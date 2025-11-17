#!/usr/bin/env python3
"""
Generate High-Quality 3D Ising Data

This script generates properly equilibrated 3D Ising data with:
- Better equilibration protocols
- More configurations near critical temperature
- Proper magnetization scaling
- Enhanced data quality validation
"""

import sys
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.enhanced_monte_carlo import EnhancedMonteCarloSimulator3D
from src.data.equilibration_3d import EquilibrationProtocol3D
from src.utils.config import PrometheusConfig
from src.utils.logging_utils import setup_logging, get_logger


class HighQuality3DDataGenerator:
    """Generate high-quality 3D Ising data with proper equilibration."""
    
    def __init__(self, lattice_size=32, random_seed=42):
        self.lattice_size = lattice_size
        self.random_seed = random_seed
        self.logger = get_logger(__name__)
        
        # Enhanced parameters for better data quality
        self.equilibration_steps = 50000  # Increased from default
        self.measurement_steps = 20000    # Increased from default
        self.measurement_interval = 10    # Measure every 10 steps
        
        # Critical temperature for 3D Ising
        self.tc_theoretical = 4.511
        
    def generate_enhanced_dataset(self, 
                                n_temperatures=30,
                                n_configs_per_temp=50,
                                temp_range_factor=0.4,
                                output_path="data/ising_3d_high_quality.h5"):
        """
        Generate enhanced 3D dataset with focus on critical region.
        
        Args:
            n_temperatures: Number of temperature points
            n_configs_per_temp: Configurations per temperature
            temp_range_factor: Temperature range as fraction around Tc
            output_path: Output file path
        """
        
        self.logger.info("Starting high-quality 3D data generation")
        self.logger.info(f"Parameters:")
        self.logger.info(f"  Lattice size: {self.lattice_size}³")
        self.logger.info(f"  Temperatures: {n_temperatures}")
        self.logger.info(f"  Configs per temp: {n_configs_per_temp}")
        self.logger.info(f"  Equilibration steps: {self.equilibration_steps}")
        self.logger.info(f"  Measurement steps: {self.measurement_steps}")
        
        # Create temperature array focused on critical region
        temp_range = temp_range_factor * self.tc_theoretical
        temp_min = self.tc_theoretical - temp_range
        temp_max = self.tc_theoretical + temp_range
        
        # Use denser sampling near Tc
        temperatures = self._create_critical_focused_temperatures(
            temp_min, temp_max, n_temperatures
        )
        
        self.logger.info(f"Temperature range: [{temp_min:.3f}, {temp_max:.3f}]")
        self.logger.info(f"Critical temperature: {self.tc_theoretical:.3f}")
        
        # Initialize data storage
        total_configs = n_temperatures * n_configs_per_temp
        configurations = np.zeros((total_configs, self.lattice_size, self.lattice_size, self.lattice_size), dtype=np.int8)
        magnetizations = np.zeros(total_configs, dtype=np.float32)
        energies = np.zeros(total_configs, dtype=np.float32)
        temperature_array = np.zeros(total_configs, dtype=np.float32)
        
        # Generate data for each temperature
        config_idx = 0
        
        for temp_idx, temperature in enumerate(tqdm(temperatures, desc="Generating temperatures")):
            self.logger.info(f"Processing temperature {temperature:.4f} ({temp_idx+1}/{n_temperatures})")
            
            # Create simulator with enhanced parameters
            simulator = EnhancedMonteCarloSimulator3D(
                lattice_size=(self.lattice_size, self.lattice_size, self.lattice_size),
                temperature=temperature,
                random_seed=self.random_seed + temp_idx
            )
            
            # Enhanced equilibration protocol
            equilibration_protocol = EquilibrationProtocol3D(
                min_steps=self.equilibration_steps,
                max_steps=self.equilibration_steps * 2,
                convergence_threshold=1e-4,
                check_interval=1000
            )
            
            # Equilibrate system
            self.logger.debug(f"  Equilibrating at T={temperature:.4f}")
            equilibration_result = equilibration_protocol.equilibrate(simulator)
            
            if not equilibration_result.converged:
                self.logger.warning(f"  Equilibration may not have converged at T={temperature:.4f}")
            
            # Generate configurations
            self.logger.debug(f"  Generating {n_configs_per_temp} configurations")
            
            for config_num in range(n_configs_per_temp):
                # Run measurement steps between configurations
                for _ in range(self.measurement_steps // self.measurement_interval):
                    for _ in range(self.measurement_interval):
                        simulator.metropolis_step()
                
                # Record configuration
                config = simulator.get_configuration()
                magnetization = simulator.calculate_magnetization()
                energy = simulator.calculate_energy()
                
                configurations[config_idx] = config.spins
                magnetizations[config_idx] = magnetization
                energies[config_idx] = energy
                temperature_array[config_idx] = temperature
                
                config_idx += 1
        
        # Validate data quality
        self.logger.info("Validating data quality")
        quality_report = self._validate_data_quality(
            configurations, magnetizations, energies, temperature_array, temperatures
        )
        
        # Save data
        self.logger.info(f"Saving data to {output_path}")
        self._save_enhanced_dataset(
            output_path, configurations, magnetizations, energies, 
            temperature_array, temperatures, quality_report
        )
        
        self.logger.info("High-quality 3D data generation completed")
        
        return output_path, quality_report
    
    def _create_critical_focused_temperatures(self, temp_min, temp_max, n_temperatures):
        """Create temperature array with denser sampling near Tc."""
        
        # Split temperatures: 60% near critical region, 40% uniform
        n_critical = int(0.6 * n_temperatures)
        n_uniform = n_temperatures - n_critical
        
        # Critical region (±10% around Tc)
        critical_range = 0.1 * self.tc_theoretical
        critical_min = self.tc_theoretical - critical_range
        critical_max = self.tc_theoretical + critical_range
        
        # Ensure critical region is within bounds
        critical_min = max(critical_min, temp_min)
        critical_max = min(critical_max, temp_max)
        
        # Generate temperature arrays
        critical_temps = np.linspace(critical_min, critical_max, n_critical)
        
        # Uniform temperatures outside critical region
        uniform_low = np.linspace(temp_min, critical_min, n_uniform // 2 + 1)[:-1]
        uniform_high = np.linspace(critical_max, temp_max, n_uniform // 2 + 1)[1:]
        
        # Combine and sort
        all_temperatures = np.concatenate([uniform_low, critical_temps, uniform_high])
        all_temperatures = np.unique(all_temperatures)  # Remove duplicates
        
        # Ensure we have the right number of temperatures
        if len(all_temperatures) != n_temperatures:
            # Fallback to uniform distribution
            all_temperatures = np.linspace(temp_min, temp_max, n_temperatures)
        
        return np.sort(all_temperatures)
    
    def _validate_data_quality(self, configurations, magnetizations, energies, temperatures, unique_temps):
        """Validate the quality of generated data."""
        
        quality_report = {
            'total_configurations': len(configurations),
            'temperature_range': (float(np.min(temperatures)), float(np.max(temperatures))),
            'magnetization_range': (float(np.min(magnetizations)), float(np.max(magnetizations))),
            'energy_range': (float(np.min(energies)), float(np.max(energies))),
            'phase_transition_visible': False,
            'data_quality_score': 0.0,
            'issues': []
        }
        
        # Check magnetization range
        mag_range = np.max(np.abs(magnetizations))
        if mag_range < 0.1:
            quality_report['issues'].append("Magnetization range too small - possible equilibration issues")
        
        # Check for phase transition signature
        tc_idx = np.argmin(np.abs(unique_temps - self.tc_theoretical))
        
        # Analyze magnetization vs temperature
        mean_mags_by_temp = []
        std_mags_by_temp = []
        
        for temp in unique_temps:
            temp_mask = np.abs(temperatures - temp) < 0.01
            temp_mags = magnetizations[temp_mask]
            
            if len(temp_mags) > 0:
                mean_mags_by_temp.append(np.mean(np.abs(temp_mags)))
                std_mags_by_temp.append(np.std(temp_mags))
            else:
                mean_mags_by_temp.append(0)
                std_mags_by_temp.append(0)
        
        mean_mags_by_temp = np.array(mean_mags_by_temp)
        std_mags_by_temp = np.array(std_mags_by_temp)
        
        # Check for phase transition
        below_tc = unique_temps < self.tc_theoretical
        above_tc = unique_temps > self.tc_theoretical
        
        if np.any(below_tc) and np.any(above_tc):
            mag_below = np.mean(mean_mags_by_temp[below_tc])
            mag_above = np.mean(mean_mags_by_temp[above_tc])
            
            if mag_below > mag_above * 1.2:  # 20% difference
                quality_report['phase_transition_visible'] = True
        
        # Calculate overall quality score
        score = 0.0
        
        # Magnetization range score (0-30 points)
        if mag_range > 0.3:
            score += 30
        elif mag_range > 0.1:
            score += 20
        elif mag_range > 0.05:
            score += 10
        
        # Phase transition visibility (0-40 points)
        if quality_report['phase_transition_visible']:
            score += 40
        
        # Data completeness (0-20 points)
        if len(configurations) > 1000:
            score += 20
        elif len(configurations) > 500:
            score += 15
        elif len(configurations) > 100:
            score += 10
        
        # Temperature coverage (0-10 points)
        temp_coverage = (np.max(temperatures) - np.min(temperatures)) / (2 * self.tc_theoretical)
        score += min(10, temp_coverage * 10)
        
        quality_report['data_quality_score'] = score
        
        # Quality assessment
        if score >= 80:
            quality_level = "EXCELLENT"
        elif score >= 60:
            quality_level = "GOOD"
        elif score >= 40:
            quality_level = "FAIR"
        else:
            quality_level = "POOR"
        
        quality_report['quality_level'] = quality_level
        
        self.logger.info(f"Data quality assessment: {quality_level} (score: {score:.1f}/100)")
        
        return quality_report
    
    def _save_enhanced_dataset(self, output_path, configurations, magnetizations, 
                             energies, temperatures, unique_temps, quality_report):
        """Save the enhanced dataset to HDF5 file."""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            # Main data
            f.create_dataset('configurations', data=configurations, compression='gzip')
            f.create_dataset('magnetizations', data=magnetizations, compression='gzip')
            f.create_dataset('energies', data=energies, compression='gzip')
            f.create_dataset('temperatures', data=temperatures, compression='gzip')
            
            # Metadata
            metadata_group = f.create_group('metadata')
            metadata_group.attrs['lattice_size'] = self.lattice_size
            metadata_group.attrs['n_configurations'] = len(configurations)
            metadata_group.attrs['n_temperatures'] = len(unique_temps)
            metadata_group.attrs['temperature_range'] = (np.min(temperatures), np.max(temperatures))
            metadata_group.attrs['theoretical_tc'] = self.tc_theoretical
            metadata_group.attrs['equilibration_steps'] = self.equilibration_steps
            metadata_group.attrs['measurement_steps'] = self.measurement_steps
            metadata_group.attrs['random_seed'] = self.random_seed
            
            # Quality report
            quality_group = f.create_group('quality_report')
            for key, value in quality_report.items():
                if isinstance(value, list):
                    if value:  # Only save non-empty lists
                        quality_group.create_dataset(key, data=value)
                else:
                    quality_group.attrs[key] = value
            
            # Temperature array
            f.create_dataset('unique_temperatures', data=unique_temps, compression='gzip')
        
        self.logger.info(f"Dataset saved: {len(configurations)} configurations")
        self.logger.info(f"Quality level: {quality_report['quality_level']}")


def main():
    """Generate high-quality 3D Ising data."""
    
    # Setup logging
    config = PrometheusConfig()
    setup_logging(config.logging)
    
    logger = get_logger(__name__)
    logger.info("Starting high-quality 3D data generation")
    
    # Create generator
    generator = HighQuality3DDataGenerator(
        lattice_size=32,
        random_seed=42
    )
    
    # Generate enhanced dataset
    output_path, quality_report = generator.generate_enhanced_dataset(
        n_temperatures=25,          # Focused on critical region
        n_configs_per_temp=40,      # More configs per temperature
        temp_range_factor=0.35,     # Narrower range around Tc
        output_path="data/ising_3d_high_quality.h5"
    )
    
    logger.info(f"High-quality 3D data generation completed")
    logger.info(f"Output: {output_path}")
    logger.info(f"Quality: {quality_report['quality_level']} ({quality_report['data_quality_score']:.1f}/100)")
    
    # Print summary
    print("\n" + "="*60)
    print("HIGH-QUALITY 3D DATA GENERATION SUMMARY")
    print("="*60)
    print(f"Output file: {output_path}")
    print(f"Total configurations: {quality_report['total_configurations']}")
    print(f"Temperature range: {quality_report['temperature_range']}")
    print(f"Magnetization range: {quality_report['magnetization_range']}")
    print(f"Phase transition visible: {quality_report['phase_transition_visible']}")
    print(f"Data quality: {quality_report['quality_level']} ({quality_report['data_quality_score']:.1f}/100)")
    
    if quality_report['issues']:
        print("\nIssues found:")
        for issue in quality_report['issues']:
            print(f"  - {issue}")
    
    print("="*60)


if __name__ == "__main__":
    main()