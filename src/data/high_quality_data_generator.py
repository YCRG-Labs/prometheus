"""
High-Quality Equilibrated Data Generator

This module implements task 8.3: Generate high-quality equilibrated data with strong phase transitions
by creating enhanced 3D Monte Carlo with 100k+ equilibration steps and dense temperature sampling.
"""

import numpy as np
import h5py
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from .enhanced_monte_carlo import EnhancedMonteCarloSimulator
from .equilibration_3d import EquilibrationValidator3D
from ..models.physics_models import Ising3DModel
from ..utils.logging_utils import get_logger


@dataclass
class HighQualityDataConfig:
    """Configuration for high-quality data generation."""
    system_sizes: List[int]
    temperature_range: Tuple[float, float]
    critical_temperature: float
    temperature_resolution: float
    n_configurations_per_temp: int
    equilibration_steps: int
    sampling_interval: int
    max_correlation_time_factor: float
    target_magnetization_range: Tuple[float, float]
    quality_validation_enabled: bool
    parallel_processing: bool
    n_processes: Optional[int]


@dataclass
class DataQualityMetrics:
    """Container for data quality assessment metrics."""
    temperature: float
    n_configurations: int
    equilibration_achieved: bool
    equilibration_steps_used: int
    autocorrelation_time: float
    magnetization_mean: float
    magnetization_std: float
    magnetization_range: Tuple[float, float]
    energy_convergence: bool
    phase_transition_strength: float
    quality_score: float


@dataclass
class HighQualityDataset:
    """Container for high-quality dataset with metadata."""
    configurations: np.ndarray
    temperatures: np.ndarray
    magnetizations: np.ndarray
    energies: np.ndarray
    system_size: int
    quality_metrics: List[DataQualityMetrics]
    generation_config: HighQualityDataConfig
    total_generation_time: float
    validation_passed: bool


class HighQualityDataGenerator:
    """
    Generator for high-quality equilibrated Monte Carlo data with strong phase transitions.
    
    Key features:
    1. Enhanced equilibration with 100k+ steps
    2. Dense temperature sampling around Tc (±0.5K with 0.05K resolution)
    3. 500+ configurations per temperature for better statistics
    4. Validation of magnetization phase transition (M > 0.5 below Tc, M < 0.1 above Tc)
    5. Comprehensive quality assessment and validation
    """
    
    def __init__(self, 
                 random_seed: Optional[int] = None,
                 verbose: bool = True):
        """
        Initialize high-quality data generator.
        
        Args:
            random_seed: Random seed for reproducibility
            verbose: Whether to print detailed progress information
        """
        self.random_seed = random_seed
        self.verbose = verbose
        self.logger = get_logger(__name__)
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate_high_quality_3d_dataset(self,
                                       config: HighQualityDataConfig,
                                       output_path: Optional[str] = None) -> HighQualityDataset:
        """
        Generate high-quality 3D Ising dataset with enhanced equilibration.
        
        Args:
            config: HighQualityDataConfig with generation parameters
            output_path: Optional path to save dataset
            
        Returns:
            HighQualityDataset with generated data and quality metrics
        """
        self.logger.info("Starting high-quality 3D dataset generation")
        start_time = time.time()
        
        # Generate temperature schedule with dense sampling around Tc
        temperatures = self._generate_dense_temperature_schedule(config)
        
        self.logger.info(f"Generated {len(temperatures)} temperature points")
        self.logger.info(f"Temperature range: {np.min(temperatures):.3f} - {np.max(temperatures):.3f}")
        
        # Generate data for each system size
        datasets = []
        
        for system_size in config.system_sizes:
            self.logger.info(f"Generating data for system size L = {system_size}")
            
            dataset = self._generate_single_size_dataset(
                system_size, temperatures, config
            )
            
            datasets.append(dataset)
        
        # For now, return the largest system size dataset
        # In a full implementation, you might want to return all sizes
        main_dataset = max(datasets, key=lambda d: d.system_size)
        
        total_time = time.time() - start_time
        main_dataset.total_generation_time = total_time
        
        # Validate overall dataset quality
        main_dataset.validation_passed = self._validate_dataset_quality(main_dataset, config)
        
        # Save dataset if path provided
        if output_path:
            self._save_dataset(main_dataset, output_path)
        
        self.logger.info(f"High-quality dataset generation completed in {total_time:.2f} seconds")
        self.logger.info(f"Dataset validation: {'PASSED' if main_dataset.validation_passed else 'FAILED'}")
        
        return main_dataset
    
    def _generate_dense_temperature_schedule(self, config: HighQualityDataConfig) -> np.ndarray:
        """Generate dense temperature sampling around critical temperature."""
        
        temp_min, temp_max = config.temperature_range
        tc = config.critical_temperature
        resolution = config.temperature_resolution
        
        # Create three regions with different sampling densities
        
        # Region 1: Far below Tc (coarser sampling)
        far_below_tc = np.arange(temp_min, tc - 0.5, resolution * 2)
        
        # Region 2: Near Tc (dense sampling)
        near_tc_min = max(temp_min, tc - 0.5)
        near_tc_max = min(temp_max, tc + 0.5)
        near_tc = np.arange(near_tc_min, near_tc_max + resolution, resolution)
        
        # Region 3: Far above Tc (coarser sampling)
        far_above_tc = np.arange(tc + 0.5 + resolution * 2, temp_max + resolution * 2, resolution * 2)
        
        # Combine all regions
        all_temps = np.concatenate([far_below_tc, near_tc, far_above_tc])
        
        # Remove duplicates and sort
        temperatures = np.unique(all_temps)
        
        # Filter to ensure we're within the specified range
        temperatures = temperatures[(temperatures >= temp_min) & (temperatures <= temp_max)]
        
        return temperatures
    
    def _generate_single_size_dataset(self,
                                    system_size: int,
                                    temperatures: np.ndarray,
                                    config: HighQualityDataConfig) -> HighQualityDataset:
        """Generate dataset for a single system size."""
        
        self.logger.info(f"Generating data for L = {system_size}")
        
        # Initialize physics model and simulator
        model = Ising3DModel()
        simulator = EnhancedMonteCarloSimulator(
            model=model,
            system_size=system_size,
            random_seed=self.random_seed
        )
        
        # Initialize equilibration validator
        validator = EquilibrationValidator3D()
        
        # Storage for results
        all_configurations = []
        all_temperatures = []
        all_magnetizations = []
        all_energies = []
        quality_metrics = []
        
        # Generate data for each temperature
        if config.parallel_processing and len(temperatures) > 1:
            # Parallel processing
            results = self._generate_parallel(
                system_size, temperatures, config
            )
            
            # Collect results
            for temp_result in results:
                temp, configs, mags, energies, metrics = temp_result
                
                all_configurations.extend(configs)
                all_temperatures.extend([temp] * len(configs))
                all_magnetizations.extend(mags)
                all_energies.extend(energies)
                quality_metrics.append(metrics)
        
        else:
            # Sequential processing
            for i, temperature in enumerate(temperatures):
                if self.verbose:
                    self.logger.info(f"Processing T = {temperature:.4f} ({i+1}/{len(temperatures)})")
                
                temp_result = self._generate_temperature_data(
                    simulator, validator, temperature, config
                )
                
                configs, mags, energies, metrics = temp_result
                
                all_configurations.extend(configs)
                all_temperatures.extend([temperature] * len(configs))
                all_magnetizations.extend(mags)
                all_energies.extend(energies)
                quality_metrics.append(metrics)
        
        # Convert to numpy arrays
        configurations_array = np.array(all_configurations)
        temperatures_array = np.array(all_temperatures)
        magnetizations_array = np.array(all_magnetizations)
        energies_array = np.array(all_energies)
        
        return HighQualityDataset(
            configurations=configurations_array,
            temperatures=temperatures_array,
            magnetizations=magnetizations_array,
            energies=energies_array,
            system_size=system_size,
            quality_metrics=quality_metrics,
            generation_config=config,
            total_generation_time=0.0,  # Will be set later
            validation_passed=False  # Will be set later
        )
    
    def _generate_parallel(self,
                         system_size: int,
                         temperatures: np.ndarray,
                         config: HighQualityDataConfig) -> List[Tuple]:
        """Generate data using parallel processing."""
        
        n_processes = config.n_processes or min(mp.cpu_count(), len(temperatures))
        
        self.logger.info(f"Using {n_processes} processes for parallel generation")
        
        # Create tasks for each temperature
        tasks = [(system_size, temp, config, self.random_seed + i if self.random_seed else None) 
                for i, temp in enumerate(temperatures)]
        
        results = []
        
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            # Submit all tasks
            future_to_temp = {
                executor.submit(_generate_temperature_data_worker, task): task[1] 
                for task in tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_temp):
                temp = future_to_temp[future]
                try:
                    result = future.result()
                    results.append((temp, *result))
                    
                    if self.verbose:
                        completed = len(results)
                        total = len(temperatures)
                        self.logger.info(f"Completed T = {temp:.4f} ({completed}/{total})")
                        
                except Exception as e:
                    self.logger.error(f"Temperature {temp} failed: {e}")
        
        # Sort results by temperature
        results.sort(key=lambda x: x[0])
        
        return results
    
    def _generate_temperature_data(self,
                                 simulator: EnhancedMonteCarloSimulator,
                                 validator: EquilibrationValidator3D,
                                 temperature: float,
                                 config: HighQualityDataConfig) -> Tuple[List, List, List, DataQualityMetrics]:
        """Generate data for a single temperature."""
        
        # Enhanced equilibration with adaptive steps
        equilibration_steps = self._determine_equilibration_steps(temperature, config)
        
        # Perform equilibration
        equilibration_result = simulator.equilibrate_enhanced(
            temperature=temperature,
            n_steps=equilibration_steps,
            convergence_threshold=1e-5,
            max_correlation_time_factor=config.max_correlation_time_factor
        )
        
        if not equilibration_result.converged:
            self.logger.warning(f"Equilibration failed for T = {temperature:.4f}")
        
        # Generate configurations with proper sampling interval
        sampling_interval = max(config.sampling_interval, 
                              int(equilibration_result.autocorrelation_time * 2))
        
        configurations = []
        magnetizations = []
        energies = []
        
        for i in range(config.n_configurations_per_temp):
            # Perform Monte Carlo steps between samples
            simulator.monte_carlo_steps(temperature, sampling_interval)
            
            # Record configuration
            config_copy = simulator.configuration.copy()
            configurations.append(config_copy)
            
            # Compute observables
            magnetization = simulator.model.compute_magnetization(config_copy)
            energy = simulator.model.compute_energy(config_copy)
            
            magnetizations.append(magnetization)
            energies.append(energy)
        
        # Compute quality metrics
        quality_metrics = self._compute_temperature_quality_metrics(
            temperature, configurations, magnetizations, energies, 
            equilibration_result, config
        )
        
        return configurations, magnetizations, energies, quality_metrics
    
    def _determine_equilibration_steps(self, temperature: float, config: HighQualityDataConfig) -> int:
        """Determine number of equilibration steps based on temperature."""
        
        base_steps = config.equilibration_steps
        tc = config.critical_temperature
        
        # Increase equilibration steps near critical temperature
        distance_from_tc = abs(temperature - tc)
        temp_range = config.temperature_range[1] - config.temperature_range[0]
        normalized_distance = distance_from_tc / (0.5 * temp_range)
        
        # Near Tc, use more steps (up to 3x base steps)
        if normalized_distance < 0.1:
            multiplier = 3.0
        elif normalized_distance < 0.2:
            multiplier = 2.0
        elif normalized_distance < 0.5:
            multiplier = 1.5
        else:
            multiplier = 1.0
        
        equilibration_steps = int(base_steps * multiplier)
        
        # Ensure minimum steps
        equilibration_steps = max(equilibration_steps, 50000)
        
        return equilibration_steps
    
    def _compute_temperature_quality_metrics(self,
                                           temperature: float,
                                           configurations: List[np.ndarray],
                                           magnetizations: List[float],
                                           energies: List[float],
                                           equilibration_result: Any,
                                           config: HighQualityDataConfig) -> DataQualityMetrics:
        """Compute quality metrics for a single temperature."""
        
        mag_array = np.array(magnetizations)
        energy_array = np.array(energies)
        
        # Basic statistics
        mag_mean = np.mean(np.abs(mag_array))
        mag_std = np.std(mag_array)
        mag_range = (np.min(mag_array), np.max(mag_array))
        
        # Phase transition strength assessment
        tc = config.critical_temperature
        phase_transition_strength = self._assess_phase_transition_strength(
            temperature, mag_mean, tc, config
        )
        
        # Overall quality score
        quality_score = self._compute_quality_score(
            temperature, mag_mean, mag_std, equilibration_result, 
            phase_transition_strength, config
        )
        
        return DataQualityMetrics(
            temperature=temperature,
            n_configurations=len(configurations),
            equilibration_achieved=equilibration_result.converged,
            equilibration_steps_used=equilibration_result.steps_used,
            autocorrelation_time=equilibration_result.autocorrelation_time,
            magnetization_mean=mag_mean,
            magnetization_std=mag_std,
            magnetization_range=mag_range,
            energy_convergence=equilibration_result.energy_converged,
            phase_transition_strength=phase_transition_strength,
            quality_score=quality_score
        )
    
    def _assess_phase_transition_strength(self,
                                        temperature: float,
                                        magnetization_mean: float,
                                        critical_temperature: float,
                                        config: HighQualityDataConfig) -> float:
        """Assess the strength of the phase transition signal."""
        
        target_low, target_high = config.target_magnetization_range
        
        if temperature < critical_temperature:
            # Below Tc: expect high magnetization (M > 0.5)
            if magnetization_mean > target_low:
                strength = min(1.0, magnetization_mean / target_low)
            else:
                strength = magnetization_mean / target_low
        else:
            # Above Tc: expect low magnetization (M < 0.1)
            if magnetization_mean < target_high:
                strength = min(1.0, (target_high - magnetization_mean) / target_high)
            else:
                strength = max(0.0, 1.0 - (magnetization_mean - target_high) / target_high)
        
        return strength
    
    def _compute_quality_score(self,
                             temperature: float,
                             mag_mean: float,
                             mag_std: float,
                             equilibration_result: Any,
                             phase_transition_strength: float,
                             config: HighQualityDataConfig) -> float:
        """Compute overall quality score for temperature data."""
        
        score = 0.0
        
        # Equilibration quality (30%)
        if equilibration_result.converged:
            equilibration_score = 1.0
        else:
            equilibration_score = 0.5  # Partial credit
        score += 0.3 * equilibration_score
        
        # Phase transition strength (40%)
        score += 0.4 * phase_transition_strength
        
        # Statistical quality (20%)
        # Good statistics means reasonable standard deviation
        if mag_mean > 0:
            relative_std = mag_std / mag_mean
            # Prefer relative std between 0.1 and 0.5
            if 0.1 <= relative_std <= 0.5:
                stats_score = 1.0
            elif relative_std < 0.1:
                stats_score = relative_std / 0.1
            else:
                stats_score = max(0.0, 1.0 - (relative_std - 0.5) / 0.5)
        else:
            stats_score = 0.5
        
        score += 0.2 * stats_score
        
        # Autocorrelation quality (10%)
        if hasattr(equilibration_result, 'autocorrelation_time'):
            # Prefer reasonable autocorrelation times (not too large)
            autocorr_time = equilibration_result.autocorrelation_time
            if autocorr_time < 100:
                autocorr_score = 1.0
            elif autocorr_time < 500:
                autocorr_score = 1.0 - (autocorr_time - 100) / 400
            else:
                autocorr_score = 0.1
        else:
            autocorr_score = 0.5
        
        score += 0.1 * autocorr_score
        
        return min(1.0, score)
    
    def _validate_dataset_quality(self, dataset: HighQualityDataset, config: HighQualityDataConfig) -> bool:
        """Validate overall dataset quality."""
        
        if not config.quality_validation_enabled:
            return True
        
        validation_passed = True
        tc = config.critical_temperature
        
        # Check phase transition requirements
        below_tc_metrics = [m for m in dataset.quality_metrics if m.temperature < tc]
        above_tc_metrics = [m for m in dataset.quality_metrics if m.temperature > tc]
        
        # Validate magnetization below Tc (should be > 0.5)
        if below_tc_metrics:
            below_tc_mags = [m.magnetization_mean for m in below_tc_metrics]
            avg_mag_below = np.mean(below_tc_mags)
            
            if avg_mag_below < config.target_magnetization_range[0]:
                self.logger.warning(f"Low magnetization below Tc: {avg_mag_below:.3f} < {config.target_magnetization_range[0]}")
                validation_passed = False
        
        # Validate magnetization above Tc (should be < 0.1)
        if above_tc_metrics:
            above_tc_mags = [m.magnetization_mean for m in above_tc_metrics]
            avg_mag_above = np.mean(above_tc_mags)
            
            if avg_mag_above > config.target_magnetization_range[1]:
                self.logger.warning(f"High magnetization above Tc: {avg_mag_above:.3f} > {config.target_magnetization_range[1]}")
                validation_passed = False
        
        # Check equilibration success rate
        equilibrated_count = sum(1 for m in dataset.quality_metrics if m.equilibration_achieved)
        equilibration_rate = equilibrated_count / len(dataset.quality_metrics)
        
        if equilibration_rate < 0.9:
            self.logger.warning(f"Low equilibration success rate: {equilibration_rate:.2f}")
            validation_passed = False
        
        # Check overall quality scores
        quality_scores = [m.quality_score for m in dataset.quality_metrics]
        avg_quality = np.mean(quality_scores)
        
        if avg_quality < 0.7:
            self.logger.warning(f"Low average quality score: {avg_quality:.3f}")
            validation_passed = False
        
        return validation_passed
    
    def _save_dataset(self, dataset: HighQualityDataset, output_path: str) -> None:
        """Save dataset to HDF5 file."""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_file, 'w') as f:
            # Save main data
            f.create_dataset('configurations', data=dataset.configurations, compression='gzip')
            f.create_dataset('temperatures', data=dataset.temperatures)
            f.create_dataset('magnetizations', data=dataset.magnetizations)
            f.create_dataset('energies', data=dataset.energies)
            
            # Save metadata
            f.attrs['system_size'] = dataset.system_size
            f.attrs['total_generation_time'] = dataset.total_generation_time
            f.attrs['validation_passed'] = dataset.validation_passed
            
            # Save configuration
            config_group = f.create_group('config')
            config_group.attrs['system_sizes'] = dataset.generation_config.system_sizes
            config_group.attrs['temperature_range'] = dataset.generation_config.temperature_range
            config_group.attrs['critical_temperature'] = dataset.generation_config.critical_temperature
            config_group.attrs['temperature_resolution'] = dataset.generation_config.temperature_resolution
            config_group.attrs['n_configurations_per_temp'] = dataset.generation_config.n_configurations_per_temp
            config_group.attrs['equilibration_steps'] = dataset.generation_config.equilibration_steps
            
            # Save quality metrics
            metrics_group = f.create_group('quality_metrics')
            
            for i, metrics in enumerate(dataset.quality_metrics):
                metric_group = metrics_group.create_group(f'temp_{i}')
                metric_group.attrs['temperature'] = metrics.temperature
                metric_group.attrs['n_configurations'] = metrics.n_configurations
                metric_group.attrs['equilibration_achieved'] = metrics.equilibration_achieved
                metric_group.attrs['magnetization_mean'] = metrics.magnetization_mean
                metric_group.attrs['magnetization_std'] = metrics.magnetization_std
                metric_group.attrs['quality_score'] = metrics.quality_score
        
        self.logger.info(f"Dataset saved to {output_file}")


def _generate_temperature_data_worker(task_args: Tuple) -> Tuple:
    """Worker function for parallel temperature data generation."""
    
    system_size, temperature, config, seed = task_args
    
    # Initialize components for this worker
    model = Ising3DModel()
    simulator = EnhancedMonteCarloSimulator(
        model=model,
        system_size=system_size,
        random_seed=seed
    )
    validator = EquilibrationValidator3D()
    
    # Create generator instance
    generator = HighQualityDataGenerator(random_seed=seed, verbose=False)
    
    # Generate data for this temperature
    return generator._generate_temperature_data(simulator, validator, temperature, config)


def create_high_quality_data_config(
    system_sizes: List[int] = [16, 32],
    temperature_range: Tuple[float, float] = (3.5, 5.5),
    critical_temperature: float = 4.511,
    temperature_resolution: float = 0.05,
    n_configurations_per_temp: int = 500,
    equilibration_steps: int = 100000,
    sampling_interval: int = 100,
    max_correlation_time_factor: float = 10.0,
    target_magnetization_range: Tuple[float, float] = (0.5, 0.1),
    quality_validation_enabled: bool = True,
    parallel_processing: bool = True,
    n_processes: Optional[int] = None
) -> HighQualityDataConfig:
    """
    Create configuration for high-quality data generation.
    
    Args:
        system_sizes: List of system sizes to generate
        temperature_range: (min_temp, max_temp) range
        critical_temperature: Critical temperature for dense sampling
        temperature_resolution: Temperature step size near Tc
        n_configurations_per_temp: Number of configurations per temperature
        equilibration_steps: Base number of equilibration steps
        sampling_interval: Steps between configuration samples
        max_correlation_time_factor: Maximum correlation time multiplier
        target_magnetization_range: (min_below_tc, max_above_tc) for validation
        quality_validation_enabled: Whether to validate data quality
        parallel_processing: Whether to use parallel processing
        n_processes: Number of processes (None for auto)
        
    Returns:
        HighQualityDataConfig instance
    """
    return HighQualityDataConfig(
        system_sizes=system_sizes,
        temperature_range=temperature_range,
        critical_temperature=critical_temperature,
        temperature_resolution=temperature_resolution,
        n_configurations_per_temp=n_configurations_per_temp,
        equilibration_steps=equilibration_steps,
        sampling_interval=sampling_interval,
        max_correlation_time_factor=max_correlation_time_factor,
        target_magnetization_range=target_magnetization_range,
        quality_validation_enabled=quality_validation_enabled,
        parallel_processing=parallel_processing,
        n_processes=n_processes
    )


def create_high_quality_data_generator(random_seed: Optional[int] = None,
                                     verbose: bool = True) -> HighQualityDataGenerator:
    """
    Factory function to create a HighQualityDataGenerator.
    
    Args:
        random_seed: Random seed for reproducibility
        verbose: Whether to print detailed progress information
        
    Returns:
        Configured HighQualityDataGenerator instance
    """
    return HighQualityDataGenerator(
        random_seed=random_seed,
        verbose=verbose
    )