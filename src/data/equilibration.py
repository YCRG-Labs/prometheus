"""
Equilibration and measurement protocols for Monte Carlo simulations.

This module provides utilities for:
- Detecting equilibration through autocorrelation analysis
- Managing measurement phases with configurable sampling
- Temperature sweep functionality for data generation
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import time

from .ising_simulator import IsingSimulator, SpinConfiguration


@dataclass
class EquilibrationResult:
    """Results from equilibration analysis."""
    converged: bool
    equilibration_steps: int
    autocorr_time: float
    final_acceptance_rate: float
    convergence_history: List[float]
    metadata: Dict[str, Any]


@dataclass
class MeasurementResult:
    """Results from measurement phase."""
    configurations: List[SpinConfiguration]
    measurement_steps: int
    sampling_interval: int
    decorrelation_factor: float
    metadata: Dict[str, Any]


@dataclass
class TemperatureSweepResult:
    """Results from complete temperature sweep."""
    temperatures: np.ndarray
    configurations_per_temp: List[List[SpinConfiguration]]
    equilibration_results: List[EquilibrationResult]
    measurement_results: List[MeasurementResult]
    total_configurations: int
    metadata: Dict[str, Any]


class EquilibrationProtocol:
    """Protocol for detecting equilibration in Monte Carlo simulations."""
    
    def __init__(self,
                 observable_func: Optional[Callable[[IsingSimulator], float]] = None,
                 max_steps: int = 50000,
                 min_steps: int = 1000,
                 autocorr_threshold: float = 0.1,
                 convergence_window: int = 200,
                 check_interval: int = 100):
        """
        Initialize equilibration protocol.
        
        Args:
            observable_func: Function to compute observable (default: magnetization)
            max_steps: Maximum equilibration steps
            min_steps: Minimum equilibration steps
            autocorr_threshold: Threshold for autocorrelation convergence
            convergence_window: Window size for convergence checking
            check_interval: Interval between convergence checks
        """
        self.observable_func = observable_func or (lambda sim: abs(sim.calculate_magnetization()))
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.autocorr_threshold = autocorr_threshold
        self.convergence_window = convergence_window
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)
    
    def equilibrate(self, simulator: IsingSimulator) -> EquilibrationResult:
        """
        Equilibrate the simulator and detect convergence.
        
        Args:
            simulator: IsingSimulator instance
            
        Returns:
            EquilibrationResult with convergence information
        """
        self.logger.debug(f"Starting equilibration at T={simulator.temperature:.4f}")
        
        observable_history = []
        step_count = 0
        converged = False
        
        while step_count < self.max_steps:
            # Perform Monte Carlo steps
            for _ in range(self.check_interval):
                simulator.metropolis_step()
                step_count += 1
            
            # Calculate observable
            observable = self.observable_func(simulator)
            observable_history.append(observable)
            
            # Check for convergence after minimum steps
            if step_count >= self.min_steps and len(observable_history) >= self.convergence_window:
                autocorr_time = self._estimate_autocorr_time(observable_history[-self.convergence_window:])
                
                if autocorr_time < self.autocorr_threshold:
                    converged = True
                    break
        
        # Calculate final autocorrelation time
        if len(observable_history) >= 10:
            final_autocorr_time = self._estimate_autocorr_time(observable_history[-min(len(observable_history), 100):])
        else:
            final_autocorr_time = float('inf')
        
        result = EquilibrationResult(
            converged=converged,
            equilibration_steps=step_count,
            autocorr_time=final_autocorr_time,
            final_acceptance_rate=simulator.get_acceptance_rate(),
            convergence_history=observable_history,
            metadata={
                'temperature': simulator.temperature,
                'max_steps': self.max_steps,
                'min_steps': self.min_steps,
                'check_interval': self.check_interval
            }
        )
        
        self.logger.debug(f"Equilibration complete: {step_count} steps, converged={converged}")
        return result
    
    def _estimate_autocorr_time(self, data: List[float]) -> float:
        """Estimate autocorrelation time using simple exponential decay fit."""
        if len(data) < 10:
            return float('inf')
        
        data_array = np.array(data)
        data_centered = data_array - np.mean(data_array)
        
        # Calculate autocorrelation function
        n = len(data_centered)
        autocorr = np.correlate(data_centered, data_centered, mode='full')
        autocorr = autocorr[n-1:]  # Take positive lags only
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find first zero crossing or decay to 1/e
        threshold = 1.0 / np.e
        for i, val in enumerate(autocorr[1:], 1):
            if val <= threshold or val <= 0:
                return float(i)
        
        return float(len(autocorr))


class MeasurementProtocol:
    """Protocol for measurement phase after equilibration."""
    
    def __init__(self,
                 n_measurements: int = 1000,
                 sampling_interval: int = 10,
                 decorrelation_factor: float = 2.0):
        """
        Initialize measurement protocol.
        
        Args:
            n_measurements: Number of measurements to collect
            sampling_interval: Steps between measurements
            decorrelation_factor: Factor to multiply autocorr time for decorrelation
        """
        self.n_measurements = n_measurements
        self.sampling_interval = sampling_interval
        self.decorrelation_factor = decorrelation_factor
        self.logger = logging.getLogger(__name__)
    
    def measure(self, simulator: IsingSimulator, autocorr_time: float) -> MeasurementResult:
        """
        Perform measurement phase.
        
        Args:
            simulator: Equilibrated IsingSimulator
            autocorr_time: Autocorrelation time from equilibration
            
        Returns:
            MeasurementResult with collected configurations
        """
        # Adjust sampling interval based on autocorrelation time
        if np.isfinite(autocorr_time):
            autocorr_based_interval = int(autocorr_time * self.decorrelation_factor)
        else:
            # Use a reasonable default if autocorr_time is infinite
            autocorr_based_interval = self.sampling_interval * 5
        
        effective_interval = max(self.sampling_interval, autocorr_based_interval)
        
        self.logger.debug(f"Starting measurements: {self.n_measurements} configs, "
                         f"interval={effective_interval}")
        
        configurations = []
        total_steps = 0
        
        for i in range(self.n_measurements):
            # Perform decorrelation steps
            for _ in range(effective_interval):
                simulator.metropolis_step()
                total_steps += 1
            
            # Collect configuration
            config = simulator.get_configuration()
            configurations.append(config)
        
        result = MeasurementResult(
            configurations=configurations,
            measurement_steps=total_steps,
            sampling_interval=effective_interval,
            decorrelation_factor=self.decorrelation_factor,
            metadata={
                'temperature': simulator.temperature,
                'autocorr_time': autocorr_time,
                'n_measurements': len(configurations)
            }
        )
        
        self.logger.debug(f"Measurements complete: {len(configurations)} configurations")
        return result


class TemperatureSweepProtocol:
    """Protocol for sweeping across temperature range."""
    
    def __init__(self,
                 lattice_size: Tuple[int, int],
                 temp_range: Tuple[float, float],
                 n_temperatures: int,
                 critical_temp: float,
                 critical_density_factor: float = 2.0,
                 equilibration_protocol: Optional[EquilibrationProtocol] = None,
                 measurement_protocol: Optional[MeasurementProtocol] = None):
        """
        Initialize temperature sweep protocol.
        
        Args:
            lattice_size: Size of the lattice
            temp_range: (min_temp, max_temp) range
            n_temperatures: Number of temperature points
            critical_temp: Critical temperature for dense sampling
            critical_density_factor: Factor for increased density around critical point
            equilibration_protocol: Equilibration protocol (default: auto-created)
            measurement_protocol: Measurement protocol (default: auto-created)
        """
        self.lattice_size = lattice_size
        self.temp_range = temp_range
        self.n_temperatures = n_temperatures
        self.critical_temp = critical_temp
        self.critical_density_factor = critical_density_factor
        
        # Create default protocols if not provided
        if equilibration_protocol is None:
            self.equilibration_protocol = EquilibrationProtocol()
        else:
            self.equilibration_protocol = equilibration_protocol
            
        if measurement_protocol is None:
            self.measurement_protocol = MeasurementProtocol()
        else:
            self.measurement_protocol = measurement_protocol
        
        self.logger = logging.getLogger(__name__)
    
    def generate_temperature_grid(self) -> np.ndarray:
        """Generate temperature grid with dense sampling around critical temperature."""
        min_temp, max_temp = self.temp_range
        
        # Define critical region width (10% of total range)
        critical_width = (max_temp - min_temp) * 0.1
        critical_min = max(min_temp, self.critical_temp - critical_width / 2)
        critical_max = min(max_temp, self.critical_temp + critical_width / 2)
        
        # Number of points in critical region (increased density)
        n_critical = int(self.n_temperatures * critical_width / (max_temp - min_temp) * self.critical_density_factor)
        n_critical = min(n_critical, self.n_temperatures // 2)  # Don't use more than half points
        
        # Remaining points for non-critical regions
        n_remaining = self.n_temperatures - n_critical
        
        # Generate temperature points
        temperatures = []
        
        # Lower temperature region
        if critical_min > min_temp:
            n_lower = int(n_remaining * (critical_min - min_temp) / (max_temp - min_temp - critical_width))
            if n_lower > 0:
                lower_temps = np.linspace(min_temp, critical_min, n_lower, endpoint=False)
                temperatures.extend(lower_temps)
        
        # Critical region (dense sampling)
        if n_critical > 0:
            critical_temps = np.linspace(critical_min, critical_max, n_critical, endpoint=True)
            temperatures.extend(critical_temps)
        
        # Upper temperature region
        if critical_max < max_temp:
            n_upper = n_remaining - len(temperatures) + n_critical
            if n_upper > 0:
                upper_temps = np.linspace(critical_max, max_temp, n_upper, endpoint=False)[1:]  # Skip overlap
                temperatures.extend(upper_temps)
        
        # Sort and ensure we have the right number of points
        temperatures = np.array(sorted(set(temperatures)))
        
        # If we have too few points, fill with linear spacing
        if len(temperatures) < self.n_temperatures:
            additional_temps = np.linspace(min_temp, max_temp, self.n_temperatures - len(temperatures))
            temperatures = np.array(sorted(set(list(temperatures) + list(additional_temps))))
        
        # Trim to exact number if needed
        if len(temperatures) > self.n_temperatures:
            temperatures = temperatures[:self.n_temperatures]
        
        return temperatures
    
    def run_sweep(self, 
                  target_configs_per_temp: int = 1000,
                  progress_callback: Optional[Callable[[int, int, float], None]] = None) -> TemperatureSweepResult:
        """
        Run complete temperature sweep generating spin configurations.
        
        Args:
            target_configs_per_temp: Target number of configurations per temperature
            progress_callback: Optional callback for progress updates (temp_idx, total_temps, current_temp)
            
        Returns:
            TemperatureSweepResult with all collected data
        """
        temperatures = self.generate_temperature_grid()
        
        self.logger.info(f"Starting temperature sweep: {len(temperatures)} temperatures, "
                        f"{target_configs_per_temp} configs each")
        self.logger.info(f"Temperature range: {temperatures[0]:.3f} to {temperatures[-1]:.3f}")
        
        # Initialize simulator
        simulator = IsingSimulator(
            lattice_size=self.lattice_size,
            temperature=temperatures[0]
        )
        
        # Storage for results
        all_configurations = []
        equilibration_results = []
        measurement_results = []
        
        start_time = time.time()
        total_configs = 0
        
        for i, temperature in enumerate(temperatures):
            temp_start_time = time.time()
            
            # Update temperature
            simulator.set_temperature(temperature)
            simulator.randomize_lattice()  # Start from random state
            
            self.logger.info(f"Processing temperature {i+1}/{len(temperatures)}: T={temperature:.4f}")
            
            # Equilibration phase
            equilibration_result = self.equilibration_protocol.equilibrate(simulator)
            equilibration_results.append(equilibration_result)
            
            if not equilibration_result.converged:
                self.logger.warning(f"Equilibration did not converge at T={temperature:.4f}")
            
            # Measurement phase
            # Adjust number of measurements based on target
            self.measurement_protocol.n_measurements = target_configs_per_temp
            measurement_result = self.measurement_protocol.measure(
                simulator, 
                equilibration_result.autocorr_time
            )
            measurement_results.append(measurement_result)
            
            # Store configurations
            all_configurations.append(measurement_result.configurations)
            total_configs += len(measurement_result.configurations)
            
            temp_end_time = time.time()
            
            self.logger.info(f"Temperature {temperature:.4f} complete: "
                           f"{len(measurement_result.configurations)} configs, "
                           f"time={temp_end_time - temp_start_time:.1f}s")
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, len(temperatures), temperature)
        
        end_time = time.time()
        
        # Create final result
        result = TemperatureSweepResult(
            temperatures=temperatures,
            configurations_per_temp=all_configurations,
            equilibration_results=equilibration_results,
            measurement_results=measurement_results,
            total_configurations=total_configs,
            metadata={
                'sweep_time_seconds': end_time - start_time,
                'lattice_size': self.lattice_size,
                'temp_range': self.temp_range,
                'critical_temp': self.critical_temp,
                'target_configs_per_temp': target_configs_per_temp,
                'actual_n_temperatures': len(temperatures)
            }
        )
        
        self.logger.info(f"Temperature sweep complete: {total_configs} total configurations, "
                        f"time={end_time - start_time:.1f}s")
        
        return result
    
    def save_configurations(self, 
                           result: TemperatureSweepResult, 
                           output_path: str,
                           include_temperature_labels: bool = False) -> None:
        """
        Save configurations to file for ML training.
        
        Args:
            result: TemperatureSweepResult to save
            output_path: Path to save configurations
            include_temperature_labels: Whether to include temperature labels
        """
        # Flatten all configurations
        all_configs = []
        all_temps = []
        
        for temp_idx, temp_configs in enumerate(result.configurations_per_temp):
            temperature = result.temperatures[temp_idx]
            
            for config in temp_configs:
                all_configs.append(config.spins)
                all_temps.append(temperature)
        
        # Convert to numpy arrays
        spin_data = np.array(all_configs)  # Shape: (n_configs, height, width)
        temp_data = np.array(all_temps)    # Shape: (n_configs,)
        
        # Save data
        save_dict = {
            'spin_configurations': spin_data,
            'metadata': {
                'lattice_size': result.metadata['lattice_size'],
                'n_configurations': len(all_configs),
                'n_temperatures': len(result.temperatures),
                'temp_range': result.metadata['temp_range'],
                'critical_temp': result.metadata['critical_temp']
            }
        }
        
        if include_temperature_labels:
            save_dict['temperatures'] = temp_data
        
        np.savez_compressed(output_path, **save_dict)
        
        self.logger.info(f"Saved {len(all_configs)} configurations to {output_path}")


def create_default_protocols(lattice_size: Tuple[int, int] = (32, 32)) -> Tuple[EquilibrationProtocol, MeasurementProtocol]:
    """
    Create default equilibration and measurement protocols optimized for Ising model.
    
    Args:
        lattice_size: Size of the lattice
        
    Returns:
        Tuple of (equilibration_protocol, measurement_protocol)
    """
    # Scale parameters based on lattice size
    n_sites = lattice_size[0] * lattice_size[1]
    
    # Equilibration protocol
    equilibration = EquilibrationProtocol(
        observable_func=lambda sim: abs(sim.calculate_magnetization()),
        max_steps=max(50000, n_sites * 50),  # Scale with system size
        min_steps=max(1000, n_sites * 2),
        autocorr_threshold=0.1,
        convergence_window=200,
        check_interval=max(100, n_sites // 10)
    )
    
    # Measurement protocol
    measurement = MeasurementProtocol(
        n_measurements=1000,
        sampling_interval=max(10, n_sites // 100),  # Scale with system size
        decorrelation_factor=2.0
    )
    
    return equilibration, measurement


def run_standard_temperature_sweep(
    lattice_size: Tuple[int, int] = (32, 32),
    n_configurations: int = 100000,
    output_path: Optional[str] = None
) -> TemperatureSweepResult:
    """
    Run a standard temperature sweep for Ising model data generation.
    
    This function implements the standard protocol for generating training data
    as specified in the requirements.
    
    Args:
        lattice_size: Size of the lattice (default: 32x32)
        n_configurations: Target total number of configurations
        output_path: Optional path to save configurations
        
    Returns:
        TemperatureSweepResult with generated data
    """
    # Calculate configurations per temperature
    n_temperatures = 100  # Standard number of temperature points
    configs_per_temp = n_configurations // n_temperatures
    
    # Create protocols
    equilibration, measurement = create_default_protocols(lattice_size)
    
    # Create temperature sweep
    sweep = TemperatureSweepProtocol(
        lattice_size=lattice_size,
        temp_range=(1.5, 3.0),  # Range around critical temperature
        n_temperatures=n_temperatures,
        critical_temp=2.269,  # Known critical temperature for 2D Ising
        critical_density_factor=2.0,
        equilibration_protocol=equilibration,
        measurement_protocol=measurement
    )
    
    # Run sweep
    result = sweep.run_sweep(target_configs_per_temp=configs_per_temp)
    
    # Save if path provided
    if output_path:
        sweep.save_configurations(result, output_path, include_temperature_labels=False)
    
    return result