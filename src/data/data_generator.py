"""
Data generation orchestrator for the Prometheus project.

This module provides the DataGenerator class that coordinates Ising model simulation
across temperature ranges with parallel processing, progress tracking, and data validation.
"""

import numpy as np
import logging
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import pickle
import hashlib

from .ising_simulator import IsingSimulator, SpinConfiguration
from .equilibration import (
    TemperatureSweepProtocol, 
    TemperatureSweepResult,
    create_default_protocols
)
from ..utils.config import PrometheusConfig, IsingConfig
from ..utils.logging_utils import get_logger, LoggingContext


@dataclass
class GenerationProgress:
    """Progress tracking for data generation."""
    total_temperatures: int
    completed_temperatures: int
    total_configurations: int
    generated_configurations: int
    start_time: float
    current_temperature: Optional[float] = None
    estimated_completion_time: Optional[float] = None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_temperatures == 0:
            return 0.0
        return (self.completed_temperatures / self.total_temperatures) * 100.0
    
    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time in seconds."""
        return time.time() - self.start_time
    
    def estimate_completion_time(self) -> Optional[float]:
        """Estimate remaining time in seconds."""
        if self.completed_temperatures == 0:
            return None
        
        elapsed = self.elapsed_time
        rate = self.completed_temperatures / elapsed
        remaining_temps = self.total_temperatures - self.completed_temperatures
        
        if rate > 0:
            return remaining_temps / rate
        return None


@dataclass
class ValidationResult:
    """Results from data validation checks."""
    is_valid: bool
    total_configurations: int
    invalid_configurations: int
    validation_errors: List[str]
    spin_value_errors: int
    shape_errors: int
    temperature_errors: int
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate as percentage."""
        if self.total_configurations == 0:
            return 0.0
        return (self.invalid_configurations / self.total_configurations) * 100.0


def _generate_temperature_batch(args: Tuple) -> Tuple[int, List[SpinConfiguration], Dict[str, Any]]:
    """
    Worker function for parallel temperature processing.
    
    This function is designed to be used with multiprocessing and must be
    at module level for pickling.
    
    Args:
        args: Tuple containing (temp_index, temperature, lattice_size, n_configs, protocols)
        
    Returns:
        Tuple of (temp_index, configurations, metadata)
    """
    temp_index, temperature, lattice_size, n_configs, equilibration_params, measurement_params = args
    
    # Create simulator
    simulator = IsingSimulator(lattice_size=lattice_size, temperature=temperature)
    
    # Recreate protocols from parameters
    from .equilibration import EquilibrationProtocol, MeasurementProtocol
    
    equilibration = EquilibrationProtocol(**equilibration_params)
    measurement = MeasurementProtocol(**measurement_params)
    
    # Run equilibration
    equilibration_result = equilibration.equilibrate(simulator)
    
    # Run measurements
    measurement.n_measurements = n_configs
    measurement_result = measurement.measure(simulator, equilibration_result.autocorr_time)
    
    # Prepare metadata
    metadata = {
        'temperature': temperature,
        'equilibration_converged': equilibration_result.converged,
        'equilibration_steps': equilibration_result.equilibration_steps,
        'autocorr_time': equilibration_result.autocorr_time,
        'acceptance_rate': equilibration_result.final_acceptance_rate,
        'n_configurations': len(measurement_result.configurations)
    }
    
    return temp_index, measurement_result.configurations, metadata


class DataGenerator:
    """
    Orchestrates Ising model data generation across temperature ranges.
    
    This class coordinates simulation across multiple temperatures with support for:
    - Parallel processing for multiple temperature simulations
    - Progress tracking and intermediate result saving
    - Data validation checks for spin configuration integrity
    - Configurable simulation parameters
    """
    
    def __init__(self, config: PrometheusConfig):
        """
        Initialize the DataGenerator.
        
        Args:
            config: PrometheusConfig object with simulation parameters
        """
        self.config = config
        self.ising_config = config.ising
        self.logger = get_logger(__name__)
        
        # Create output directories
        self.data_dir = Path(config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize protocols
        self.equilibration_protocol, self.measurement_protocol = create_default_protocols(
            self.ising_config.lattice_size
        )
        
        # Override protocol parameters from config
        self.equilibration_protocol.max_steps = self.ising_config.equilibration_steps
        self.measurement_protocol.n_measurements = self.ising_config.n_configs_per_temp
        
        # Progress tracking
        self.progress = None
        self.progress_callbacks: List[Callable[[GenerationProgress], None]] = []
        
        # Intermediate results storage
        self.intermediate_results: Dict[int, Tuple[List[SpinConfiguration], Dict[str, Any]]] = {}
        
        self.logger.info(f"DataGenerator initialized with lattice size {self.ising_config.lattice_size}")
        self.logger.info(f"Target configurations per temperature: {self.ising_config.n_configs_per_temp}")
    
    def add_progress_callback(self, callback: Callable[[GenerationProgress], None]) -> None:
        """Add a progress callback function."""
        self.progress_callbacks.append(callback)
    
    def _update_progress(self, completed_temps: int, current_temp: Optional[float] = None) -> None:
        """Update progress and notify callbacks."""
        if self.progress is None:
            return
        
        self.progress.completed_temperatures = completed_temps
        self.progress.current_temperature = current_temp
        self.progress.estimated_completion_time = self.progress.estimate_completion_time()
        
        # Count generated configurations
        total_configs = sum(len(configs) for configs, _ in self.intermediate_results.values())
        self.progress.generated_configurations = total_configs
        
        # Notify callbacks
        for callback in self.progress_callbacks:
            try:
                callback(self.progress)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")
    
    def _create_temperature_grid(self) -> np.ndarray:
        """Create temperature grid with dense sampling around critical temperature."""
        sweep_protocol = TemperatureSweepProtocol(
            lattice_size=self.ising_config.lattice_size,
            temp_range=self.ising_config.temperature_range,
            n_temperatures=self.ising_config.n_temperatures,
            critical_temp=self.ising_config.critical_temp,
            critical_density_factor=2.0,
            equilibration_protocol=self.equilibration_protocol,
            measurement_protocol=self.measurement_protocol
        )
        
        return sweep_protocol.generate_temperature_grid()
    
    def _prepare_worker_args(self, temperatures: np.ndarray) -> List[Tuple]:
        """Prepare arguments for parallel workers."""
        # Convert protocols to parameter dictionaries for serialization
        equilibration_params = {
            'observable_func': None,  # Will use default magnetization
            'max_steps': self.equilibration_protocol.max_steps,
            'min_steps': self.equilibration_protocol.min_steps,
            'autocorr_threshold': self.equilibration_protocol.autocorr_threshold,
            'convergence_window': self.equilibration_protocol.convergence_window,
            'check_interval': self.equilibration_protocol.check_interval
        }
        
        measurement_params = {
            'n_measurements': self.measurement_protocol.n_measurements,
            'sampling_interval': self.measurement_protocol.sampling_interval,
            'decorrelation_factor': self.measurement_protocol.decorrelation_factor
        }
        
        args_list = []
        for i, temperature in enumerate(temperatures):
            args = (
                i,  # temperature index
                temperature,
                self.ising_config.lattice_size,
                self.ising_config.n_configs_per_temp,
                equilibration_params,
                measurement_params
            )
            args_list.append(args)
        
        return args_list
    
    def generate_parallel(self, 
                         n_processes: Optional[int] = None,
                         save_intermediate: bool = True,
                         intermediate_save_interval: int = 10) -> TemperatureSweepResult:
        """
        Generate data using parallel processing across temperatures.
        
        Args:
            n_processes: Number of parallel processes (default: CPU count)
            save_intermediate: Whether to save intermediate results
            intermediate_save_interval: Save interval in number of completed temperatures
            
        Returns:
            TemperatureSweepResult with generated data
        """
        if n_processes is None:
            n_processes = min(mp.cpu_count(), 8)  # Limit to 8 processes
        
        self.logger.info(f"Starting parallel data generation with {n_processes} processes")
        
        with LoggingContext("Parallel Data Generation"):
            # Create temperature grid
            temperatures = self._create_temperature_grid()
            
            # Initialize progress tracking
            total_configs = len(temperatures) * self.ising_config.n_configs_per_temp
            self.progress = GenerationProgress(
                total_temperatures=len(temperatures),
                completed_temperatures=0,
                total_configurations=total_configs,
                generated_configurations=0,
                start_time=time.time()
            )
            
            self.logger.info(f"Generating data for {len(temperatures)} temperatures")
            self.logger.info(f"Target total configurations: {total_configs}")
            
            # Prepare worker arguments
            worker_args = self._prepare_worker_args(temperatures)
            
            # Storage for results
            all_configurations = [None] * len(temperatures)
            all_metadata = [None] * len(temperatures)
            
            # Execute parallel processing
            with ProcessPoolExecutor(max_workers=n_processes) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(_generate_temperature_batch, args): args[0] 
                    for args in worker_args
                }
                
                completed_count = 0
                
                # Process completed tasks
                for future in as_completed(future_to_index):
                    temp_index = future_to_index[future]
                    
                    try:
                        temp_index, configurations, metadata = future.result()
                        
                        # Store results
                        all_configurations[temp_index] = configurations
                        all_metadata[temp_index] = metadata
                        self.intermediate_results[temp_index] = (configurations, metadata)
                        
                        completed_count += 1
                        current_temp = temperatures[temp_index]
                        
                        self.logger.info(f"Completed temperature {temp_index + 1}/{len(temperatures)}: "
                                       f"T={current_temp:.4f}, {len(configurations)} configs")
                        
                        # Update progress
                        self._update_progress(completed_count, current_temp)
                        
                        # Save intermediate results if requested
                        if save_intermediate and completed_count % intermediate_save_interval == 0:
                            self._save_intermediate_results(completed_count)
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process temperature {temp_index}: {e}")
                        raise
            
            # Create final result
            result = TemperatureSweepResult(
                temperatures=temperatures,
                configurations_per_temp=all_configurations,
                equilibration_results=[],  # Not stored in parallel version
                measurement_results=[],    # Not stored in parallel version
                total_configurations=sum(len(configs) for configs in all_configurations),
                metadata={
                    'generation_time_seconds': self.progress.elapsed_time,
                    'lattice_size': self.ising_config.lattice_size,
                    'temp_range': self.ising_config.temperature_range,
                    'critical_temp': self.ising_config.critical_temp,
                    'n_processes': n_processes,
                    'per_temp_metadata': all_metadata
                }
            )
            
            self.logger.info(f"Parallel generation complete: {result.total_configurations} configurations, "
                           f"time={self.progress.elapsed_time:.1f}s")
            
            return result
    
    def generate_sequential(self) -> TemperatureSweepResult:
        """
        Generate data using sequential processing (single-threaded).
        
        Returns:
            TemperatureSweepResult with generated data
        """
        self.logger.info("Starting sequential data generation")
        
        with LoggingContext("Sequential Data Generation"):
            # Create temperature sweep protocol
            sweep_protocol = TemperatureSweepProtocol(
                lattice_size=self.ising_config.lattice_size,
                temp_range=self.ising_config.temperature_range,
                n_temperatures=self.ising_config.n_temperatures,
                critical_temp=self.ising_config.critical_temp,
                critical_density_factor=2.0,
                equilibration_protocol=self.equilibration_protocol,
                measurement_protocol=self.measurement_protocol
            )
            
            # Add progress callback
            def progress_callback(temp_idx: int, total_temps: int, current_temp: float):
                if self.progress is None:
                    return
                self._update_progress(temp_idx, current_temp)
            
            # Initialize progress tracking
            total_configs = self.ising_config.n_temperatures * self.ising_config.n_configs_per_temp
            self.progress = GenerationProgress(
                total_temperatures=self.ising_config.n_temperatures,
                completed_temperatures=0,
                total_configurations=total_configs,
                generated_configurations=0,
                start_time=time.time()
            )
            
            # Run sweep
            result = sweep_protocol.run_sweep(
                target_configs_per_temp=self.ising_config.n_configs_per_temp,
                progress_callback=progress_callback
            )
            
            self.logger.info(f"Sequential generation complete: {result.total_configurations} configurations")
            
            return result
    
    def _save_intermediate_results(self, completed_count: int) -> None:
        """Save intermediate results to disk."""
        try:
            intermediate_path = self.data_dir / f"intermediate_results_{completed_count}.pkl"
            
            with open(intermediate_path, 'wb') as f:
                pickle.dump(self.intermediate_results, f)
            
            self.logger.info(f"Saved intermediate results: {completed_count} temperatures completed")
            
        except Exception as e:
            self.logger.warning(f"Failed to save intermediate results: {e}")
    
    def validate_configurations(self, 
                              configurations: List[List[SpinConfiguration]],
                              temperatures: np.ndarray) -> ValidationResult:
        """
        Validate spin configurations for integrity and correctness.
        
        Args:
            configurations: List of configuration lists per temperature
            temperatures: Array of temperatures
            
        Returns:
            ValidationResult with validation statistics
        """
        self.logger.info("Starting configuration validation")
        
        total_configs = 0
        invalid_configs = 0
        validation_errors = []
        spin_value_errors = 0
        shape_errors = 0
        temperature_errors = 0
        
        expected_shape = self.ising_config.lattice_size
        
        for temp_idx, temp_configs in enumerate(configurations):
            expected_temp = temperatures[temp_idx]
            
            for config_idx, config in enumerate(temp_configs):
                total_configs += 1
                config_valid = True
                
                # Check spin values (should be +1 or -1)
                unique_spins = np.unique(config.spins)
                if not np.array_equal(np.sort(unique_spins), np.array([-1, 1])):
                    if len(unique_spins) > 2 or not all(spin in [-1, 1] for spin in unique_spins):
                        spin_value_errors += 1
                        config_valid = False
                        validation_errors.append(
                            f"Invalid spin values at T={expected_temp:.4f}, config {config_idx}: {unique_spins}"
                        )
                
                # Check shape
                if config.spins.shape != expected_shape:
                    shape_errors += 1
                    config_valid = False
                    validation_errors.append(
                        f"Invalid shape at T={expected_temp:.4f}, config {config_idx}: "
                        f"expected {expected_shape}, got {config.spins.shape}"
                    )
                
                # Check temperature consistency
                if abs(config.temperature - expected_temp) > 1e-6:
                    temperature_errors += 1
                    config_valid = False
                    validation_errors.append(
                        f"Temperature mismatch at config {config_idx}: "
                        f"expected {expected_temp:.6f}, got {config.temperature:.6f}"
                    )
                
                if not config_valid:
                    invalid_configs += 1
        
        result = ValidationResult(
            is_valid=(invalid_configs == 0),
            total_configurations=total_configs,
            invalid_configurations=invalid_configs,
            validation_errors=validation_errors[:100],  # Limit error list size
            spin_value_errors=spin_value_errors,
            shape_errors=shape_errors,
            temperature_errors=temperature_errors
        )
        
        self.logger.info(f"Validation complete: {total_configs} configurations checked")
        if result.is_valid:
            self.logger.info("All configurations are valid")
        else:
            self.logger.warning(f"Found {invalid_configs} invalid configurations ({result.error_rate:.2f}% error rate)")
            self.logger.warning(f"Errors: {spin_value_errors} spin values, {shape_errors} shapes, {temperature_errors} temperatures")
        
        return result
    
    def generate_dataset(self, 
                        use_parallel: bool = True,
                        n_processes: Optional[int] = None,
                        validate_data: bool = True) -> Tuple[TemperatureSweepResult, Optional[ValidationResult]]:
        """
        Generate complete dataset with optional validation.
        
        Args:
            use_parallel: Whether to use parallel processing
            n_processes: Number of processes for parallel generation
            validate_data: Whether to validate generated data
            
        Returns:
            Tuple of (TemperatureSweepResult, ValidationResult or None)
        """
        self.logger.info("Starting dataset generation")
        
        # Generate data
        if use_parallel:
            result = self.generate_parallel(n_processes=n_processes)
        else:
            result = self.generate_sequential()
        
        # Validate if requested
        validation_result = None
        if validate_data:
            validation_result = self.validate_configurations(
                result.configurations_per_temp,
                result.temperatures
            )
            
            if not validation_result.is_valid:
                self.logger.error("Data validation failed - dataset may be corrupted")
                # Could raise exception here if strict validation is required
        
        self.logger.info(f"Dataset generation complete: {result.total_configurations} total configurations")
        
        return result, validation_result
    
    def save_dataset(self, 
                    result: TemperatureSweepResult,
                    output_path: Optional[str] = None,
                    include_temperature_labels: bool = False) -> str:
        """
        Save generated dataset to file.
        
        Args:
            result: TemperatureSweepResult to save
            output_path: Output file path (default: auto-generated)
            include_temperature_labels: Whether to include temperature labels
            
        Returns:
            Path where dataset was saved
        """
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"ising_dataset_{timestamp}.npz"
            output_path = str(self.data_dir / filename)
        
        # Flatten configurations
        all_configs = []
        all_temps = []
        all_magnetizations = []
        all_energies = []
        
        for temp_idx, temp_configs in enumerate(result.configurations_per_temp):
            temperature = result.temperatures[temp_idx]
            
            for config in temp_configs:
                all_configs.append(config.spins)
                all_temps.append(temperature)
                all_magnetizations.append(config.magnetization)
                all_energies.append(config.energy)
        
        # Convert to numpy arrays
        spin_data = np.array(all_configs)
        temp_data = np.array(all_temps)
        mag_data = np.array(all_magnetizations)
        energy_data = np.array(all_energies)
        
        # Create save dictionary
        save_dict = {
            'spin_configurations': spin_data,
            'magnetizations': mag_data,
            'energies': energy_data,
            'metadata': {
                'lattice_size': self.ising_config.lattice_size,
                'n_configurations': len(all_configs),
                'n_temperatures': len(result.temperatures),
                'temp_range': self.ising_config.temperature_range,
                'critical_temp': self.ising_config.critical_temp,
                'generation_time': result.metadata.get('generation_time_seconds', 0),
                'config_hash': self._compute_config_hash()
            }
        }
        
        # Include temperature labels if requested
        if include_temperature_labels:
            save_dict['temperatures'] = temp_data
        
        # Save to file
        np.savez_compressed(output_path, **save_dict)
        
        self.logger.info(f"Dataset saved to {output_path}")
        self.logger.info(f"Dataset contains {len(all_configs)} configurations")
        
        return output_path
    
    def _compute_config_hash(self) -> str:
        """Compute hash of configuration for reproducibility tracking."""
        config_str = (
            f"{self.ising_config.lattice_size}"
            f"{self.ising_config.temperature_range}"
            f"{self.ising_config.n_temperatures}"
            f"{self.ising_config.critical_temp}"
            f"{self.ising_config.n_configs_per_temp}"
            f"{self.config.seed}"
        )
        return hashlib.md5(config_str.encode()).hexdigest()[:16]


def create_test_dataloader(config: PrometheusConfig, data_path: Optional[str] = None) -> 'DataLoader':
    """
    Create a test DataLoader for model validation.
    
    Args:
        config: Prometheus configuration
        data_path: Optional path to existing dataset. If None, looks for latest dataset.
        
    Returns:
        DataLoader for test data
    """
    from torch.utils.data import DataLoader
    from .preprocessing import DataProcessor
    
    logger = get_logger(__name__)
    
    # Initialize data processor
    processor = DataProcessor(config)
    
    # Find dataset if not provided
    if data_path is None:
        data_dir = Path(config.data.output_dir)
        
        # Look for processed HDF5 files
        h5_files = list(data_dir.glob("*_processed_*.h5"))
        if h5_files:
            # Use the most recent one
            data_path = str(max(h5_files, key=lambda p: p.stat().st_mtime))
            logger.info(f"Using existing processed dataset: {data_path}")
        else:
            # Look for raw NPZ files
            npz_files = list(data_dir.glob("ising_dataset_*.npz"))
            if npz_files:
                # Use the most recent one and process it
                raw_path = str(max(npz_files, key=lambda p: p.stat().st_mtime))
                logger.info(f"Processing raw dataset: {raw_path}")
                data_path = processor.process_dataset(raw_path)
            else:
                raise FileNotFoundError(
                    f"No dataset files found in {data_dir}. "
                    "Please generate data first using scripts/generate_data.py"
                )
    
    # Create dataloaders
    _, _, test_loader = processor.create_dataloaders(
        hdf5_path=data_path,
        batch_size=config.training.batch_size,
        load_physics=True  # Load physics quantities for validation
    )
    
    logger.info(f"Created test DataLoader with {len(test_loader)} batches")
    return test_loader