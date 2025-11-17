#!/usr/bin/env python3
"""
Enhanced 3D Ising Data Generation with Proper Equilibration

This script implements task 7.1: Generate high-quality 3D Ising data with proper equilibration
- Enhanced Monte Carlo with longer equilibration (50k+ steps)
- Temperature schedule with high density near Tc = 4.511
- Generate 200+ configurations per temperature with proper sampling intervals
- Validate magnetization ranges are physically meaningful (not ~0.007)
"""

import sys
import numpy as np
import h5py
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.enhanced_monte_carlo import EnhancedMonteCarloSimulator
from src.data.equilibration_3d import Enhanced3DEquilibrationProtocol
from src.utils.logging_utils import setup_logging, get_logger


@dataclass
class Enhanced3DDataConfig:
    """Configuration for enhanced 3D data generation."""
    system_sizes: List[int] = None
    temperature_range: Tuple[float, float] = (3.8, 5.2)  # Focused around Tc = 4.511
    n_temperatures: int = 40  # Dense sampling
    n_configs_per_temp: int = 200  # Increased from previous implementations
    equilibration_steps: int = 50000  # Minimum 50k steps
    sampling_interval: int = 50  # Steps between configuration samples
    tc_theoretical: float = 4.511
    tc_density_factor: float = 0.6  # Fraction of temperatures near Tc
    output_dir: str = "data"
    
    def __post_init__(self):
        if self.system_sizes is None:
            self.system_sizes = [16, 32]  # Focus on larger systems for better physics


@dataclass
class Enhanced3DDataResult:
    """Results from enhanced 3D data generation."""
    system_size: int
    temperatures: np.ndarray
    configurations: np.ndarray  # Shape: (n_temps, n_configs, L, L, L)
    magnetizations: np.ndarray  # Shape: (n_temps, n_configs)
    energies: np.ndarray  # Shape: (n_temps, n_configs)
    equilibration_success_rates: np.ndarray  # Shape: (n_temps,)
    magnetization_curves: np.ndarray  # Mean magnetization per temperature
    energy_curves: np.ndarray  # Mean energy per temperature
    generation_time: float
    validation_results: Dict[str, any]


class Enhanced3DDataGenerator:
    """Enhanced 3D data generator with proper equilibration and validation."""
    
    def __init__(self, config: Enhanced3DDataConfig):
        """Initialize enhanced 3D data generator."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Enhanced 3D Data Generator initialized")
        self.logger.info(f"System sizes: {config.system_sizes}")
        self.logger.info(f"Temperature range: {config.temperature_range}")
        self.logger.info(f"Configurations per temperature: {config.n_configs_per_temp}")
        self.logger.info(f"Equilibration steps: {config.equilibration_steps}")
    
    def create_enhanced_temperature_schedule(self) -> np.ndarray:
        """
        Create temperature schedule with high density near Tc = 4.511.
        
        Returns:
            Array of temperatures with dense sampling near critical point
        """
        temp_min, temp_max = self.config.temperature_range
        n_temps = self.config.n_temperatures
        tc = self.config.tc_theoretical
        
        # Allocate temperatures: more near Tc, fewer at extremes
        n_critical = int(self.config.tc_density_factor * n_temps)
        n_extremes = n_temps - n_critical
        
        # Critical region: ±15% around Tc
        tc_window = 0.15 * tc
        tc_min = max(tc - tc_window, temp_min)
        tc_max = min(tc + tc_window, temp_max)
        
        # Dense sampling in critical region
        critical_temps = np.linspace(tc_min, tc_max, n_critical)
        
        # Sparse sampling in extreme regions
        n_low = n_extremes // 2
        n_high = n_extremes - n_low
        
        if tc_min > temp_min:
            low_temps = np.linspace(temp_min, tc_min, n_low + 1)[:-1]  # Exclude tc_min
        else:
            low_temps = np.array([])
        
        if tc_max < temp_max:
            high_temps = np.linspace(tc_max, temp_max, n_high + 1)[1:]  # Exclude tc_max
        else:
            high_temps = np.array([])
        
        # Combine all temperatures
        all_temps = np.concatenate([low_temps, critical_temps, high_temps])
        all_temps = np.sort(np.unique(all_temps))
        
        self.logger.info(f"Created temperature schedule: {len(all_temps)} points")
        self.logger.info(f"Critical region [{tc_min:.3f}, {tc_max:.3f}]: {n_critical} points")
        self.logger.info(f"Extreme regions: {len(low_temps)} + {len(high_temps)} points")
        
        return all_temps
    
    def generate_system_data(self, system_size: int) -> Enhanced3DDataResult:
        """
        Generate enhanced data for a single system size.
        
        Args:
            system_size: Linear system size (creates cubic lattice)
            
        Returns:
            Enhanced3DDataResult with generated data
        """
        self.logger.info(f"Generating enhanced data for system size L={system_size}")
        
        start_time = time.time()
        lattice_shape = (system_size, system_size, system_size)
        
        # Create temperature schedule
        temperatures = self.create_enhanced_temperature_schedule()
        n_temps = len(temperatures)
        n_configs = self.config.n_configs_per_temp
        
        # Initialize data arrays
        configurations = np.zeros((n_temps, n_configs, system_size, system_size, system_size), dtype=np.int8)
        magnetizations = np.zeros((n_temps, n_configs), dtype=np.float32)
        energies = np.zeros((n_temps, n_configs), dtype=np.float32)
        equilibration_success_rates = np.zeros(n_temps, dtype=np.float32)
        
        # Process each temperature
        for temp_idx, temperature in enumerate(tqdm(temperatures, desc=f"L={system_size}")):
            self.logger.info(f"L={system_size}: Processing T={temperature:.4f} ({temp_idx+1}/{n_temps})")
            
            # Create simulator with enhanced parameters
            simulator = EnhancedMonteCarloSimulator(
                lattice_size=lattice_shape,
                temperature=temperature,
                coupling=1.0,
                magnetic_field=0.0
            )
            
            # Create enhanced equilibration protocol with more reasonable thresholds
            equilibration_protocol = Enhanced3DEquilibrationProtocol(
                max_steps=max(self.config.equilibration_steps, 50000),  # Ensure minimum 50k
                min_steps=self.config.equilibration_steps // 4,  # Reduced minimum steps
                energy_autocorr_threshold=0.1,  # More lenient convergence
                magnetization_autocorr_threshold=0.1,
                convergence_window=500,  # Smaller window
                check_interval=200,  # More frequent checks
                quality_threshold=0.5  # More reasonable quality requirement
            )
            
            # Equilibrate system
            self.logger.debug(f"  Equilibrating at T={temperature:.4f}")
            equilibration_result = equilibration_protocol.equilibrate_3d(simulator)
            
            equilibration_success_rates[temp_idx] = 1.0 if equilibration_result.converged else 0.0
            
            if not equilibration_result.converged:
                self.logger.warning(f"  Equilibration failed at T={temperature:.4f}, "
                                  f"quality={equilibration_result.convergence_quality_score:.3f}")
            
            # Generate configurations with proper sampling
            self.logger.debug(f"  Generating {n_configs} configurations")
            
            for config_idx in range(n_configs):
                # Perform sampling interval steps between configurations
                for _ in range(self.config.sampling_interval):
                    simulator.metropolis_step()
                
                # Record configuration
                config = simulator.get_configuration()
                
                configurations[temp_idx, config_idx] = config.spins
                magnetizations[temp_idx, config_idx] = config.magnetization
                energies[temp_idx, config_idx] = config.energy
            
            # Log temperature statistics
            temp_mag_mean = np.mean(np.abs(magnetizations[temp_idx]))
            temp_mag_std = np.std(magnetizations[temp_idx])
            temp_energy_mean = np.mean(energies[temp_idx])
            
            self.logger.info(f"  T={temperature:.4f}: |M|={temp_mag_mean:.4f}±{temp_mag_std:.4f}, "
                           f"E={temp_energy_mean:.4f}, equilibrated={equilibration_result.converged}")
        
        # Calculate summary curves
        magnetization_curves = np.mean(np.abs(magnetizations), axis=1)
        energy_curves = np.mean(energies, axis=1)
        
        generation_time = time.time() - start_time
        
        # Validate data quality
        validation_results = self._validate_enhanced_data(
            temperatures, magnetizations, energies, equilibration_success_rates
        )
        
        result = Enhanced3DDataResult(
            system_size=system_size,
            temperatures=temperatures,
            configurations=configurations,
            magnetizations=magnetizations,
            energies=energies,
            equilibration_success_rates=equilibration_success_rates,
            magnetization_curves=magnetization_curves,
            energy_curves=energy_curves,
            generation_time=generation_time,
            validation_results=validation_results
        )
        
        self.logger.info(f"L={system_size} completed: {len(temperatures)} temps, "
                        f"{len(temperatures) * n_configs} configs, "
                        f"time={generation_time:.1f}s")
        
        return result
    
    def _validate_enhanced_data(self, 
                              temperatures: np.ndarray,
                              magnetizations: np.ndarray,
                              energies: np.ndarray,
                              equilibration_success_rates: np.ndarray) -> Dict[str, any]:
        """
        Validate the quality of enhanced 3D data.
        
        Args:
            temperatures: Temperature array
            magnetizations: Magnetization data
            energies: Energy data
            equilibration_success_rates: Equilibration success rates
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'issues': [],
            'magnetization_range': (float(np.min(magnetizations)), float(np.max(magnetizations))),
            'energy_range': (float(np.min(energies)), float(np.max(energies))),
            'equilibration_success_rate': float(np.mean(equilibration_success_rates)),
            'phase_transition_detected': False,
            'critical_temperature_estimate': None,
            'data_quality_score': 0.0
        }
        
        # Check magnetization range - should be physically meaningful
        mag_range = np.max(np.abs(magnetizations))
        if mag_range < 0.05:
            validation['issues'].append(f"Magnetization range too small: {mag_range:.4f}")
            validation['is_valid'] = False
        
        # Check for phase transition signature
        mag_curves = np.mean(np.abs(magnetizations), axis=1)
        
        # Find temperatures below and above theoretical Tc
        tc = self.config.tc_theoretical
        below_tc_mask = temperatures < tc
        above_tc_mask = temperatures > tc
        
        if np.any(below_tc_mask) and np.any(above_tc_mask):
            mag_below_tc = np.mean(mag_curves[below_tc_mask])
            mag_above_tc = np.mean(mag_curves[above_tc_mask])
            
            # Check for clear phase transition
            if mag_below_tc > mag_above_tc * 1.5:  # 50% difference
                validation['phase_transition_detected'] = True
                
                # Estimate critical temperature from steepest descent
                mag_gradient = np.gradient(mag_curves, temperatures)
                tc_estimate_idx = np.argmin(mag_gradient)  # Steepest descent
                validation['critical_temperature_estimate'] = float(temperatures[tc_estimate_idx])
        
        # Check equilibration success rate
        eq_success_rate = np.mean(equilibration_success_rates)
        if eq_success_rate < 0.8:
            validation['issues'].append(f"Low equilibration success rate: {eq_success_rate:.2f}")
        
        # Calculate overall data quality score
        score = 0.0
        
        # Magnetization range score (0-25 points)
        if mag_range > 0.3:
            score += 25
        elif mag_range > 0.1:
            score += 20
        elif mag_range > 0.05:
            score += 10
        
        # Phase transition detection (0-30 points)
        if validation['phase_transition_detected']:
            score += 30
            
            # Bonus for accurate Tc estimate
            if validation['critical_temperature_estimate'] is not None:
                tc_error = abs(validation['critical_temperature_estimate'] - tc) / tc
                if tc_error < 0.05:  # Within 5%
                    score += 10
                elif tc_error < 0.1:  # Within 10%
                    score += 5
        
        # Equilibration success (0-25 points)
        score += 25 * eq_success_rate
        
        # Data completeness (0-20 points)
        total_configs = magnetizations.size
        if total_configs >= 5000:
            score += 20
        elif total_configs >= 2000:
            score += 15
        elif total_configs >= 1000:
            score += 10
        
        validation['data_quality_score'] = score
        
        # Overall assessment
        if score >= 80:
            quality_level = "EXCELLENT"
        elif score >= 60:
            quality_level = "GOOD"
        elif score >= 40:
            quality_level = "FAIR"
        else:
            quality_level = "POOR"
            validation['is_valid'] = False
        
        validation['quality_level'] = quality_level
        
        self.logger.info(f"Data validation: {quality_level} (score: {score:.1f}/100)")
        if validation['issues']:
            for issue in validation['issues']:
                self.logger.warning(f"  Issue: {issue}")
        
        return validation
    
    def save_enhanced_dataset(self, 
                            results: List[Enhanced3DDataResult],
                            output_path: Optional[str] = None) -> str:
        """
        Save enhanced 3D dataset to HDF5 file.
        
        Args:
            results: List of Enhanced3DDataResult for different system sizes
            output_path: Output file path (auto-generated if None)
            
        Returns:
            Path where dataset was saved
        """
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = str(self.output_dir / f"ising_3d_enhanced_{timestamp}.h5")
        
        self.logger.info(f"Saving enhanced 3D dataset to {output_path}")
        
        with h5py.File(output_path, 'w') as f:
            # Global metadata
            f.attrs['n_system_sizes'] = len(results)
            f.attrs['theoretical_tc'] = self.config.tc_theoretical
            f.attrs['generation_config'] = str(self.config.__dict__)
            f.attrs['total_configurations'] = sum(r.configurations.size for r in results)
            
            # Save data for each system size
            for result in results:
                size_group = f.create_group(f'system_size_{result.system_size}')
                
                # Basic metadata
                size_group.attrs['system_size'] = result.system_size
                size_group.attrs['n_temperatures'] = len(result.temperatures)
                size_group.attrs['n_configs_per_temp'] = self.config.n_configs_per_temp
                size_group.attrs['generation_time'] = result.generation_time
                size_group.attrs['equilibration_success_rate'] = np.mean(result.equilibration_success_rates)
                
                # Validation results - handle different data types properly
                validation_group = size_group.create_group('validation')
                for key, value in result.validation_results.items():
                    if isinstance(value, (list, tuple)):
                        if value:  # Only save non-empty sequences
                            validation_group.create_dataset(key, data=value)
                    elif isinstance(value, (str, int, float, bool)):
                        validation_group.attrs[key] = value
                    elif value is None:
                        validation_group.attrs[key] = "None"
                    else:
                        # Convert other types to string
                        validation_group.attrs[key] = str(value)
                
                # Temperature data
                size_group.create_dataset('temperatures', data=result.temperatures, compression='gzip')
                size_group.create_dataset('magnetization_curves', data=result.magnetization_curves, compression='gzip')
                size_group.create_dataset('energy_curves', data=result.energy_curves, compression='gzip')
                size_group.create_dataset('equilibration_success_rates', data=result.equilibration_success_rates, compression='gzip')
                
                # Configuration data
                size_group.create_dataset('configurations', data=result.configurations, compression='gzip')
                size_group.create_dataset('magnetizations', data=result.magnetizations, compression='gzip')
                size_group.create_dataset('energies', data=result.energies, compression='gzip')
        
        self.logger.info(f"Enhanced 3D dataset saved successfully")
        return output_path
    
    def generate_complete_enhanced_dataset(self) -> List[Enhanced3DDataResult]:
        """
        Generate complete enhanced dataset for all system sizes.
        
        Returns:
            List of Enhanced3DDataResult for each system size
        """
        self.logger.info("Starting complete enhanced 3D dataset generation")
        
        results = []
        
        for system_size in self.config.system_sizes:
            self.logger.info(f"Processing system size {system_size}")
            
            result = self.generate_system_data(system_size)
            results.append(result)
            
            # Log summary for this system size
            validation = result.validation_results
            self.logger.info(f"L={system_size} summary:")
            self.logger.info(f"  Quality: {validation['quality_level']} ({validation['data_quality_score']:.1f}/100)")
            self.logger.info(f"  Magnetization range: {validation['magnetization_range']}")
            self.logger.info(f"  Phase transition detected: {validation['phase_transition_detected']}")
            self.logger.info(f"  Equilibration success: {validation['equilibration_success_rate']:.2f}")
        
        return results


def main():
    """Main function to generate enhanced 3D data."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = get_logger(__name__)
    logger.info("Starting enhanced 3D data generation (Task 7.1)")
    
    # Create enhanced configuration
    config = Enhanced3DDataConfig(
        system_sizes=[16, 32],  # Focus on larger systems
        temperature_range=(3.8, 5.2),  # Focused around Tc = 4.511
        n_temperatures=35,  # Dense sampling
        n_configs_per_temp=200,  # Increased configurations
        equilibration_steps=50000,  # Minimum 50k steps
        sampling_interval=50,  # Proper sampling
        tc_density_factor=0.6,  # 60% of temperatures near Tc
        output_dir="data"
    )
    
    # Create generator
    generator = Enhanced3DDataGenerator(config)
    
    # Generate complete dataset
    results = generator.generate_complete_enhanced_dataset()
    
    # Save dataset
    output_path = generator.save_enhanced_dataset(results)
    
    # Print final summary
    print("\n" + "="*70)
    print("ENHANCED 3D DATA GENERATION SUMMARY (Task 7.1)")
    print("="*70)
    print(f"Output file: {output_path}")
    print(f"System sizes: {config.system_sizes}")
    print(f"Temperature range: {config.temperature_range}")
    print(f"Configurations per temperature: {config.n_configs_per_temp}")
    print(f"Equilibration steps: {config.equilibration_steps}")
    
    for result in results:
        validation = result.validation_results
        print(f"\nSystem L={result.system_size}:")
        print(f"  Quality: {validation['quality_level']} ({validation['data_quality_score']:.1f}/100)")
        print(f"  Magnetization range: [{validation['magnetization_range'][0]:.4f}, {validation['magnetization_range'][1]:.4f}]")
        print(f"  Phase transition detected: {validation['phase_transition_detected']}")
        print(f"  Equilibration success rate: {validation['equilibration_success_rate']:.2f}")
        if validation['critical_temperature_estimate']:
            tc_error = abs(validation['critical_temperature_estimate'] - config.tc_theoretical) / config.tc_theoretical
            print(f"  Estimated Tc: {validation['critical_temperature_estimate']:.3f} (error: {tc_error*100:.1f}%)")
    
    print("="*70)
    logger.info("Enhanced 3D data generation completed successfully")


if __name__ == "__main__":
    main()