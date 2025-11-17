"""
3D Data Generation Pipeline for Ising Model.

This module implements a comprehensive 3D data generation pipeline that extends
the existing 2D framework to support 3D Ising model simulations with proper
temperature sweeps, system size variations, and data quality validation.
"""

import numpy as np
import logging
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass
from pathlib import Path
import pickle
import hashlib
import h5py

from .enhanced_monte_carlo import EnhancedMonteCarloSimulator, SpinConfiguration3D
from .equilibration_3d import Enhanced3DEquilibrationProtocol, Enhanced3DEquilibrationResult
from ..models.physics_models import Ising3DModel
import logging


@dataclass
class DataGenerationConfig3D:
    """Configuration for 3D data generation."""
    temperature_range: Tuple[float, float] = (3.0, 6.0)
    temperature_resolution: int = 61  # Number of temperature points
    system_sizes: List[int] = None  # Will default to [8, 16, 32]
    n_configs_per_temp: int = 1000
    sampling_interval: int = 100  # Steps between configuration samples
    equilibration_quality_threshold: float = 0.7
    parallel_processes: Optional[int] = None
    output_dir: str = "data"
    
    def __post_init__(self):
        if self.system_sizes is None:
            self.system_sizes = [8, 16, 32]


@dataclass
class GenerationProgress3D:
    """Progress tracking for 3D data generation."""
    total_system_sizes: int
    completed_system_sizes: int
    current_system_size: Optional[int]
    total_temperatures: int
    completed_temperatures: int
    current_temperature: Optional[float]
    total_configurations: int
    generated_configurations: int
    start_time: float
    estimated_completion_time: Optional[float] = None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate overall progress percentage."""
        if self.total_configurations == 0:
            return 0.0
        return (self.generated_configurations / self.total_configurations) * 100.0
    
    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time in seconds."""
        return time.time() - self.start_time


@dataclass
class SystemSizeResult3D:
    """Results for a single system size."""
    system_size: int
    lattice_shape: Tuple[int, int, int]
    temperatures: np.ndarray
    configurations: List[List[SpinConfiguration3D]]  # [temp_idx][config_idx]
    equilibration_results: List[Enhanced3DEquilibrationResult]
    magnetization_curves: np.ndarray  # Shape: (n_temps, n_configs)
    energy_curves: np.ndarray  # Shape: (n_temps, n_configs)
    generation_time_seconds: float
    metadata: Dict[str, Any]


@dataclass
class Dataset3DResult:
    """Complete 3D dataset generation result."""
    system_size_results: Dict[int, SystemSizeResult3D]
    config: DataGenerationConfig3D
    total_configurations: int
    total_generation_time: float
    theoretical_tc: float
    validation_results: Dict[str, Any]
    metadata: Dict[str, Any]


def _generate_3d_temperature_batch(args: Tuple) -> Tuple[int, int, float, List[SpinConfiguration3D], Enhanced3DEquilibrationResult]:
    """
    Worker function for parallel 3D temperature processing.
    
    Args:
        args: Tuple containing (system_size, temp_index, temperature, n_configs, sampling_interval)
        
    Returns:
        Tuple of (system_size, temp_index, temperature, configurations, equilibration_result)
    """
    system_size, temp_index, temperature, n_configs, sampling_interval = args
    
    # Create 3D lattice shape (cubic)
    lattice_shape = (system_size, system_size, system_size)
    
    # Create 3D simulator
    simulator = EnhancedMonteCarloSimulator(
        lattice_size=lattice_shape,
        temperature=temperature,
        coupling=1.0,
        magnetic_field=0.0
    )
    
    # Create equilibration protocol
    equilibration_protocol = Enhanced3DEquilibrationProtocol(
        max_steps=100000,
        min_steps=5000,
        energy_autocorr_threshold=0.05,
        magnetization_autocorr_threshold=0.05,
        convergence_window=500,
        check_interval=200,
        quality_threshold=0.7
    )
    
    # Equilibrate system
    equilibration_result = equilibration_protocol.equilibrate_3d(simulator)
    
    # Generate configurations
    configurations = []
    
    if equilibration_result.converged:
        # Sample configurations with proper spacing
        for config_idx in range(n_configs):
            # Perform sampling interval steps between configurations
            for _ in range(sampling_interval):
                simulator.metropolis_step()
            
            # Store configuration
            config = simulator.get_configuration()
            configurations.append(config)
    else:
        # If equilibration failed, still generate some configurations but mark them
        logger = logging.getLogger(__name__)
        logger.warning(f"Equilibration failed for L={system_size}, T={temperature:.4f}")
        
        # Generate configurations anyway but with warning metadata
        for config_idx in range(n_configs):
            for _ in range(sampling_interval):
                simulator.metropolis_step()
            
            config = simulator.get_configuration()
            config.metadata['equilibration_failed'] = True
            configurations.append(config)
    
    return system_size, temp_index, temperature, configurations, equilibration_result


class DataGenerator3D:
    """
    Comprehensive 3D Ising data generation pipeline.
    
    Generates datasets for multiple system sizes with proper temperature sweeps,
    equilibration validation, and data quality assessment.
    """
    
    def __init__(self, config: DataGenerationConfig3D):
        """
        Initialize 3D data generator.
        
        Args:
            config: DataGenerationConfig3D with generation parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize physics model for theoretical values
        self.physics_model = Ising3DModel(coupling_strength=1.0)
        self.theoretical_tc = self.physics_model.get_theoretical_tc()
        
        # Progress tracking
        self.progress = None
        self.progress_callbacks: List[Callable[[GenerationProgress3D], None]] = []
        
        self.logger.info(f"3D DataGenerator initialized")
        self.logger.info(f"System sizes: {config.system_sizes}")
        self.logger.info(f"Temperature range: {config.temperature_range}")
        self.logger.info(f"Theoretical Tc: {self.theoretical_tc:.4f}")
    
    def add_progress_callback(self, callback: Callable[[GenerationProgress3D], None]) -> None:
        """Add a progress callback function."""
        self.progress_callbacks.append(callback)
    
    def _update_progress(self, 
                        completed_sizes: int = 0,
                        current_size: Optional[int] = None,
                        completed_temps: int = 0,
                        current_temp: Optional[float] = None,
                        generated_configs: int = 0) -> None:
        """Update progress and notify callbacks."""
        if self.progress is None:
            return
        
        self.progress.completed_system_sizes = completed_sizes
        self.progress.current_system_size = current_size
        self.progress.completed_temperatures = completed_temps
        self.progress.current_temperature = current_temp
        self.progress.generated_configurations = generated_configs
        
        # Notify callbacks
        for callback in self.progress_callbacks:
            try:
                callback(self.progress)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")
    
    def _create_temperature_grid(self) -> np.ndarray:
        """
        Create temperature grid with dense sampling around critical temperature.
        
        Returns:
            Array of temperature values
        """
        temp_min, temp_max = self.config.temperature_range
        n_temps = self.config.temperature_resolution
        
        # Create denser sampling around critical temperature
        tc = self.theoretical_tc
        
        # Define regions: below Tc, around Tc, above Tc
        tc_window = 0.5  # Window around Tc for dense sampling
        
        # Allocate temperatures
        n_below = int(0.3 * n_temps)
        n_around = int(0.4 * n_temps)
        n_above = n_temps - n_below - n_around
        
        # Below Tc
        temps_below = np.linspace(temp_min, tc - tc_window/2, n_below, endpoint=False)
        
        # Around Tc (dense sampling)
        temps_around = np.linspace(tc - tc_window/2, tc + tc_window/2, n_around, endpoint=False)
        
        # Above Tc
        temps_above = np.linspace(tc + tc_window/2, temp_max, n_above, endpoint=True)
        
        # Combine and sort
        temperatures = np.concatenate([temps_below, temps_around, temps_above])
        temperatures = np.sort(temperatures)
        
        self.logger.info(f"Created temperature grid: {len(temperatures)} points from {temp_min:.2f} to {temp_max:.2f}")
        self.logger.info(f"Dense sampling around Tc={tc:.3f} ± {tc_window/2:.3f}")
        
        return temperatures
    
    def _generate_system_size_data(self, 
                                  system_size: int,
                                  temperatures: np.ndarray,
                                  use_parallel: bool = True) -> SystemSizeResult3D:
        """
        Generate data for a single system size across all temperatures.
        
        Args:
            system_size: Linear system size (creates cubic lattice)
            temperatures: Array of temperatures to simulate
            use_parallel: Whether to use parallel processing
            
        Returns:
            SystemSizeResult3D with generated data
        """
        self.logger.info(f"Generating data for system size L={system_size}")
        
        start_time = time.time()
        lattice_shape = (system_size, system_size, system_size)
        
        # Storage for results
        all_configurations = [None] * len(temperatures)
        all_equilibration_results = [None] * len(temperatures)
        
        if use_parallel and len(temperatures) > 1:
            # Parallel processing across temperatures
            n_processes = self.config.parallel_processes or min(mp.cpu_count(), len(temperatures))
            
            # Prepare worker arguments
            worker_args = [
                (system_size, temp_idx, temp, self.config.n_configs_per_temp, self.config.sampling_interval)
                for temp_idx, temp in enumerate(temperatures)
            ]
            
            with ProcessPoolExecutor(max_workers=n_processes) as executor:
                # Submit all tasks
                future_to_temp = {
                    executor.submit(_generate_3d_temperature_batch, args): args[1]
                    for args in worker_args
                }
                
                completed_temps = 0
                
                # Process completed tasks
                for future in as_completed(future_to_temp):
                    temp_idx = future_to_temp[future]
                    
                    try:
                        size, temp_idx, temperature, configurations, equilibration_result = future.result()
                        
                        # Store results
                        all_configurations[temp_idx] = configurations
                        all_equilibration_results[temp_idx] = equilibration_result
                        
                        completed_temps += 1
                        
                        self.logger.info(f"L={system_size}: Completed T={temperature:.4f} "
                                       f"({completed_temps}/{len(temperatures)}), "
                                       f"{len(configurations)} configs, "
                                       f"equilibrated={equilibration_result.converged}")
                        
                        # Update progress
                        total_configs = sum(len(configs) for configs in all_configurations if configs is not None)
                        self._update_progress(
                            current_size=system_size,
                            completed_temps=completed_temps,
                            current_temp=temperature,
                            generated_configs=total_configs
                        )
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process L={system_size}, T_idx={temp_idx}: {e}")
                        raise
        
        else:
            # Sequential processing
            for temp_idx, temperature in enumerate(temperatures):
                self.logger.info(f"L={system_size}: Processing T={temperature:.4f} ({temp_idx+1}/{len(temperatures)})")
                
                # Create simulator
                simulator = EnhancedMonteCarloSimulator(
                    lattice_size=lattice_shape,
                    temperature=temperature,
                    coupling=1.0,
                    magnetic_field=0.0
                )
                
                # Equilibrate
                equilibration_protocol = Enhanced3DEquilibrationProtocol()
                equilibration_result = equilibration_protocol.equilibrate_3d(simulator)
                
                # Generate configurations
                configurations = []
                for config_idx in range(self.config.n_configs_per_temp):
                    for _ in range(self.config.sampling_interval):
                        simulator.metropolis_step()
                    
                    config = simulator.get_configuration()
                    configurations.append(config)
                
                all_configurations[temp_idx] = configurations
                all_equilibration_results[temp_idx] = equilibration_result
                
                # Update progress
                total_configs = sum(len(configs) for configs in all_configurations if configs is not None)
                self._update_progress(
                    current_size=system_size,
                    completed_temps=temp_idx + 1,
                    current_temp=temperature,
                    generated_configs=total_configs
                )
        
        # Process results and create magnetization/energy curves
        magnetization_curves = np.zeros((len(temperatures), self.config.n_configs_per_temp))
        energy_curves = np.zeros((len(temperatures), self.config.n_configs_per_temp))
        
        for temp_idx, configs in enumerate(all_configurations):
            for config_idx, config in enumerate(configs):
                magnetization_curves[temp_idx, config_idx] = abs(config.magnetization)
                energy_curves[temp_idx, config_idx] = config.energy
        
        generation_time = time.time() - start_time
        
        # Create metadata
        metadata = {
            'system_size': system_size,
            'lattice_shape': lattice_shape,
            'n_sites': np.prod(lattice_shape),
            'n_temperatures': len(temperatures),
            'n_configs_per_temp': self.config.n_configs_per_temp,
            'sampling_interval': self.config.sampling_interval,
            'generation_time_seconds': generation_time,
            'theoretical_tc': self.theoretical_tc,
            'equilibration_success_rate': sum(1 for eq in all_equilibration_results if eq.converged) / len(all_equilibration_results)
        }
        
        result = SystemSizeResult3D(
            system_size=system_size,
            lattice_shape=lattice_shape,
            temperatures=temperatures,
            configurations=all_configurations,
            equilibration_results=all_equilibration_results,
            magnetization_curves=magnetization_curves,
            energy_curves=energy_curves,
            generation_time_seconds=generation_time,
            metadata=metadata
        )
        
        self.logger.info(f"Completed L={system_size}: {len(temperatures)} temperatures, "
                        f"{len(temperatures) * self.config.n_configs_per_temp} configurations, "
                        f"time={generation_time:.1f}s")
        
        return result
    
    def generate_complete_dataset(self, use_parallel: bool = True) -> Dataset3DResult:
        """
        Generate complete 3D dataset for all system sizes.
        
        Args:
            use_parallel: Whether to use parallel processing
            
        Returns:
            Dataset3DResult with complete dataset
        """
        self.logger.info("Starting complete 3D dataset generation")
        
        start_time = time.time()
        
        # Create temperature grid
        temperatures = self._create_temperature_grid()
        
        # Initialize progress tracking
        total_configs = (len(self.config.system_sizes) * 
                        len(temperatures) * 
                        self.config.n_configs_per_temp)
        
        self.progress = GenerationProgress3D(
            total_system_sizes=len(self.config.system_sizes),
            completed_system_sizes=0,
            current_system_size=None,
            total_temperatures=len(temperatures),
            completed_temperatures=0,
            current_temperature=None,
            total_configurations=total_configs,
            generated_configurations=0,
            start_time=start_time
        )
        
        self.logger.info(f"Target: {len(self.config.system_sizes)} system sizes, "
                        f"{len(temperatures)} temperatures, "
                        f"{total_configs} total configurations")
        
        # Generate data for each system size
        system_size_results = {}
        
        for size_idx, system_size in enumerate(self.config.system_sizes):
            self.logger.info(f"Processing system size {size_idx+1}/{len(self.config.system_sizes)}: L={system_size}")
            
            # Generate data for this system size
            size_result = self._generate_system_size_data(
                system_size=system_size,
                temperatures=temperatures,
                use_parallel=use_parallel
            )
            
            system_size_results[system_size] = size_result
            
            # Update progress
            self._update_progress(
                completed_sizes=size_idx + 1,
                current_size=system_size
            )
        
        total_generation_time = time.time() - start_time
        
        # Validate dataset
        validation_results = self._validate_complete_dataset(system_size_results, temperatures)
        
        # Create final result
        result = Dataset3DResult(
            system_size_results=system_size_results,
            config=self.config,
            total_configurations=sum(
                len(size_result.temperatures) * self.config.n_configs_per_temp
                for size_result in system_size_results.values()
            ),
            total_generation_time=total_generation_time,
            theoretical_tc=self.theoretical_tc,
            validation_results=validation_results,
            metadata={
                'generation_timestamp': time.time(),
                'system_sizes': self.config.system_sizes,
                'temperature_range': self.config.temperature_range,
                'n_temperatures': len(temperatures),
                'n_configs_per_temp': self.config.n_configs_per_temp,
                'total_generation_time_seconds': total_generation_time,
                'parallel_processing': use_parallel
            }
        )
        
        self.logger.info(f"Complete 3D dataset generation finished: "
                        f"{result.total_configurations} configurations, "
                        f"time={total_generation_time:.1f}s")
        
        return result
    
    def _validate_complete_dataset(self, 
                                  system_size_results: Dict[int, SystemSizeResult3D],
                                  temperatures: np.ndarray) -> Dict[str, Any]:
        """
        Validate the complete dataset for quality and consistency.
        
        Args:
            system_size_results: Results for all system sizes
            temperatures: Temperature array
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info("Validating complete 3D dataset")
        
        validation = {
            'is_valid': True,
            'total_configurations': 0,
            'invalid_configurations': 0,
            'system_size_validation': {},
            'magnetization_validation': {},
            'equilibration_validation': {},
            'issues': []
        }
        
        for system_size, size_result in system_size_results.items():
            size_validation = {
                'configurations_count': 0,
                'invalid_spins': 0,
                'shape_errors': 0,
                'equilibration_failures': 0,
                'magnetization_issues': 0
            }
            
            expected_shape = (system_size, system_size, system_size)
            
            for temp_idx, temp_configs in enumerate(size_result.configurations):
                temperature = temperatures[temp_idx]
                
                for config in temp_configs:
                    size_validation['configurations_count'] += 1
                    validation['total_configurations'] += 1
                    
                    # Check spin values
                    unique_spins = np.unique(config.spins)
                    if not np.array_equal(np.sort(unique_spins), np.array([-1, 1])):
                        size_validation['invalid_spins'] += 1
                        validation['invalid_configurations'] += 1
                    
                    # Check shape
                    if config.spins.shape != expected_shape:
                        size_validation['shape_errors'] += 1
                        validation['invalid_configurations'] += 1
                    
                    # Check equilibration status
                    if config.metadata.get('equilibration_failed', False):
                        size_validation['equilibration_failures'] += 1
                    
                    # Check magnetization reasonableness
                    if abs(config.magnetization) > 1.0:
                        size_validation['magnetization_issues'] += 1
                        validation['invalid_configurations'] += 1
            
            validation['system_size_validation'][system_size] = size_validation
            
            # Check equilibration success rate
            eq_results = size_result.equilibration_results
            eq_success_rate = sum(1 for eq in eq_results if eq.converged) / len(eq_results)
            validation['equilibration_validation'][system_size] = {
                'success_rate': eq_success_rate,
                'total_temperatures': len(eq_results),
                'successful_equilibrations': sum(1 for eq in eq_results if eq.converged)
            }
            
            if eq_success_rate < 0.8:
                validation['issues'].append(f"Low equilibration success rate for L={system_size}: {eq_success_rate:.2f}")
        
        # Validate magnetization curves
        for system_size, size_result in system_size_results.items():
            mag_curves = size_result.magnetization_curves
            
            # Check for proper transition behavior around Tc
            tc_idx = np.argmin(np.abs(temperatures - self.theoretical_tc))
            
            # Magnetization should be high at low T, low at high T
            low_t_mag = np.mean(mag_curves[:len(temperatures)//4])  # First quarter
            high_t_mag = np.mean(mag_curves[-len(temperatures)//4:])  # Last quarter
            
            mag_validation = {
                'low_temperature_magnetization': low_t_mag,
                'high_temperature_magnetization': high_t_mag,
                'transition_sharpness': low_t_mag - high_t_mag,
                'tc_index': tc_idx,
                'tc_magnetization': np.mean(mag_curves[tc_idx])
            }
            
            validation['magnetization_validation'][system_size] = mag_validation
            
            # Check for reasonable transition
            if mag_validation['transition_sharpness'] < 0.2:
                validation['issues'].append(f"Weak magnetization transition for L={system_size}")
        
        # Overall validation
        error_rate = validation['invalid_configurations'] / validation['total_configurations'] if validation['total_configurations'] > 0 else 0
        if error_rate > 0.01:  # More than 1% errors
            validation['is_valid'] = False
            validation['issues'].append(f"High error rate: {error_rate:.3f}")
        
        validation['error_rate'] = error_rate
        
        self.logger.info(f"Dataset validation complete: {validation['total_configurations']} configurations checked")
        if validation['is_valid']:
            self.logger.info("Dataset validation passed")
        else:
            self.logger.warning(f"Dataset validation issues: {validation['issues']}")
        
        return validation
    
    def save_dataset(self, 
                    result: Dataset3DResult,
                    output_path: Optional[str] = None,
                    format: str = 'hdf5') -> str:
        """
        Save 3D dataset to file.
        
        Args:
            result: Dataset3DResult to save
            output_path: Output file path (default: auto-generated)
            format: File format ('hdf5' or 'npz')
            
        Returns:
            Path where dataset was saved
        """
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if format == 'hdf5':
                filename = f"ising_3d_dataset_{timestamp}.h5"
            else:
                filename = f"ising_3d_dataset_{timestamp}.npz"
            output_path = str(self.output_dir / filename)
        
        if format == 'hdf5':
            self._save_hdf5(result, output_path)
        else:
            self._save_npz(result, output_path)
        
        self.logger.info(f"3D dataset saved to {output_path}")
        return output_path
    
    def _save_hdf5(self, result: Dataset3DResult, output_path: str) -> None:
        """Save dataset in HDF5 format."""
        with h5py.File(output_path, 'w') as f:
            # Global metadata
            f.attrs['total_configurations'] = result.total_configurations
            f.attrs['total_generation_time'] = result.total_generation_time
            f.attrs['theoretical_tc'] = result.theoretical_tc
            f.attrs['n_system_sizes'] = len(result.system_size_results)
            f.attrs['system_sizes'] = list(result.system_size_results.keys())
            
            # Save config
            config_group = f.create_group('config')
            config_group.attrs['temperature_range'] = result.config.temperature_range
            config_group.attrs['temperature_resolution'] = result.config.temperature_resolution
            config_group.attrs['n_configs_per_temp'] = result.config.n_configs_per_temp
            config_group.attrs['sampling_interval'] = result.config.sampling_interval
            
            # Save data for each system size
            for system_size, size_result in result.system_size_results.items():
                size_group = f.create_group(f'system_size_{system_size}')
                
                # Metadata
                size_group.attrs['system_size'] = system_size
                size_group.attrs['lattice_shape'] = size_result.lattice_shape
                size_group.attrs['n_temperatures'] = len(size_result.temperatures)
                size_group.attrs['generation_time'] = size_result.generation_time_seconds
                
                # Temperature array
                size_group.create_dataset('temperatures', data=size_result.temperatures)
                
                # Magnetization and energy curves
                size_group.create_dataset('magnetization_curves', data=size_result.magnetization_curves)
                size_group.create_dataset('energy_curves', data=size_result.energy_curves)
                
                # Spin configurations
                all_spins = []
                all_temps = []
                all_mags = []
                all_energies = []
                
                for temp_idx, temp_configs in enumerate(size_result.configurations):
                    temperature = size_result.temperatures[temp_idx]
                    
                    for config in temp_configs:
                        all_spins.append(config.spins)
                        all_temps.append(temperature)
                        all_mags.append(config.magnetization)
                        all_energies.append(config.energy)
                
                size_group.create_dataset('spin_configurations', data=np.array(all_spins))
                size_group.create_dataset('configuration_temperatures', data=np.array(all_temps))
                size_group.create_dataset('configuration_magnetizations', data=np.array(all_mags))
                size_group.create_dataset('configuration_energies', data=np.array(all_energies))
    
    def _save_npz(self, result: Dataset3DResult, output_path: str) -> None:
        """Save dataset in NPZ format."""
        save_dict = {
            'metadata': result.metadata,
            'total_configurations': result.total_configurations,
            'theoretical_tc': result.theoretical_tc,
            'system_sizes': list(result.system_size_results.keys())
        }
        
        # Add data for each system size
        for system_size, size_result in result.system_size_results.items():
            prefix = f'size_{system_size}_'
            
            save_dict[f'{prefix}temperatures'] = size_result.temperatures
            save_dict[f'{prefix}magnetization_curves'] = size_result.magnetization_curves
            save_dict[f'{prefix}energy_curves'] = size_result.energy_curves
            
            # Flatten configurations
            all_spins = []
            all_temps = []
            all_mags = []
            all_energies = []
            
            for temp_idx, temp_configs in enumerate(size_result.configurations):
                temperature = size_result.temperatures[temp_idx]
                
                for config in temp_configs:
                    all_spins.append(config.spins)
                    all_temps.append(temperature)
                    all_mags.append(config.magnetization)
                    all_energies.append(config.energy)
            
            save_dict[f'{prefix}spin_configurations'] = np.array(all_spins)
            save_dict[f'{prefix}configuration_temperatures'] = np.array(all_temps)
            save_dict[f'{prefix}configuration_magnetizations'] = np.array(all_mags)
            save_dict[f'{prefix}configuration_energies'] = np.array(all_energies)
        
        np.savez_compressed(output_path, **save_dict)


def create_default_3d_config() -> DataGenerationConfig3D:
    """Create default configuration for 3D data generation."""
    return DataGenerationConfig3D(
        temperature_range=(3.0, 6.0),
        temperature_resolution=61,
        system_sizes=[8, 16, 32],
        n_configs_per_temp=1000,
        sampling_interval=100,
        equilibration_quality_threshold=0.7,
        parallel_processes=None,
        output_dir="data"
    )


def generate_3d_ising_dataset(config: Optional[DataGenerationConfig3D] = None,
                             use_parallel: bool = True,
                             save_dataset: bool = True,
                             output_format: str = 'hdf5') -> Dataset3DResult:
    """
    Main function to generate complete 3D Ising dataset.
    
    This function implements the comprehensive 3D data generation pipeline
    as specified in task 3.1, creating temperature sweeps T ∈ [3.0, 6.0]
    with appropriate resolution, generating configurations for system sizes
    L ∈ {8, 16, 32}, and implementing 1000 configurations per temperature
    with proper sampling intervals.
    
    Args:
        config: DataGenerationConfig3D with generation parameters (uses default if None)
        use_parallel: Whether to use parallel processing for temperature batches
        save_dataset: Whether to save the generated dataset to file
        output_format: File format for saving ('hdf5' or 'npz')
        
    Returns:
        Dataset3DResult with complete generated dataset
        
    Raises:
        ValueError: If invalid configuration parameters are provided
        RuntimeError: If data generation fails
    """
    logger = logging.getLogger(__name__)
    
    # Use default config if none provided
    if config is None:
        config = create_default_3d_config()
        logger.info("Using default 3D data generation configuration")
    
    # Validate configuration
    _validate_generation_config(config)
    
    # Create data generator
    generator = DataGenerator3D(config)
    
    # Add progress logging callback
    def log_progress(progress: GenerationProgress3D):
        if progress.current_temperature is not None:
            logger.info(f"Progress: {progress.progress_percentage:.1f}% - "
                       f"L={progress.current_system_size}, T={progress.current_temperature:.3f}, "
                       f"{progress.generated_configurations}/{progress.total_configurations} configs")
    
    generator.add_progress_callback(log_progress)
    
    logger.info("Starting comprehensive 3D Ising dataset generation")
    logger.info(f"Configuration: {config.system_sizes} system sizes, "
               f"T ∈ [{config.temperature_range[0]}, {config.temperature_range[1]}], "
               f"{config.n_configs_per_temp} configs/temp")
    
    try:
        # Generate complete dataset
        result = generator.generate_complete_dataset(use_parallel=use_parallel)
        
        # Validate generated dataset
        if not result.validation_results['is_valid']:
            logger.warning(f"Dataset validation issues: {result.validation_results['issues']}")
        
        # Save dataset if requested
        if save_dataset:
            output_path = generator.save_dataset(result, format=output_format)
            logger.info(f"Dataset saved to: {output_path}")
        
        # Log summary
        logger.info(f"3D dataset generation completed successfully:")
        logger.info(f"  Total configurations: {result.total_configurations}")
        logger.info(f"  Generation time: {result.total_generation_time:.1f}s")
        logger.info(f"  System sizes: {list(result.system_size_results.keys())}")
        logger.info(f"  Theoretical Tc: {result.theoretical_tc:.3f}")
        
        return result
        
    except Exception as e:
        logger.error(f"3D dataset generation failed: {e}")
        raise RuntimeError(f"Failed to generate 3D dataset: {e}") from e


def _validate_generation_config(config: DataGenerationConfig3D) -> None:
    """
    Validate data generation configuration parameters.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: If configuration parameters are invalid
    """
    # Validate temperature range
    if config.temperature_range[0] >= config.temperature_range[1]:
        raise ValueError("Invalid temperature range: min must be less than max")
    
    if config.temperature_range[0] <= 0:
        raise ValueError("Temperature must be positive")
    
    # Validate system sizes
    if not config.system_sizes or any(size <= 0 for size in config.system_sizes):
        raise ValueError("System sizes must be positive integers")
    
    if any(size > 64 for size in config.system_sizes):
        raise ValueError("System sizes larger than 64 may cause memory issues")
    
    # Validate other parameters
    if config.n_configs_per_temp <= 0:
        raise ValueError("Number of configurations per temperature must be positive")
    
    if config.sampling_interval <= 0:
        raise ValueError("Sampling interval must be positive")
    
    if not (0 < config.equilibration_quality_threshold <= 1):
        raise ValueError("Equilibration quality threshold must be between 0 and 1")
    
    if config.temperature_resolution <= 0:
        raise ValueError("Temperature resolution must be positive")


def load_3d_dataset(file_path: str) -> Dataset3DResult:
    """
    Load a previously saved 3D dataset.
    
    Args:
        file_path: Path to the saved dataset file
        
    Returns:
        Dataset3DResult loaded from file
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported or corrupted
    """
    logger = logging.getLogger(__name__)
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    logger.info(f"Loading 3D dataset from: {file_path}")
    
    try:
        if file_path.suffix == '.h5':
            return _load_hdf5_dataset(file_path)
        elif file_path.suffix == '.npz':
            return _load_npz_dataset(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise ValueError(f"Failed to load dataset from {file_path}: {e}") from e


def _load_hdf5_dataset(file_path: Path) -> Dataset3DResult:
    """Load dataset from HDF5 format."""
    # Implementation would go here - placeholder for now
    raise NotImplementedError("HDF5 loading not yet implemented")


def _load_npz_dataset(file_path: Path) -> Dataset3DResult:
    """Load dataset from NPZ format."""
    # Implementation would go here - placeholder for now
    raise NotImplementedError("NPZ loading not yet implemented")