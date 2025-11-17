#!/usr/bin/env python3
"""
Multi-System Validation Script for Task 11

This script validates the Prometheus system on multiple physical systems:
- 2D Ising model (target ≥65% accuracy)
- 3D XY model (target ≥60% accuracy)
- 3-state Potts model (target ≥60% accuracy)

Uses the real accuracy validation pipeline with proper train/test splits.
"""

import numpy as np
import h5py
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.unified_monte_carlo import (
    create_ising_simulator, create_xy_simulator, create_potts_simulator
)
from src.validation.real_accuracy_validation_pipeline import (
    RealAccuracyValidationPipeline, RealValidationConfig
)


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Set up logging configuration."""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                log_dir / f'multi_system_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            ),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def generate_2d_ising_data(
    lattice_size: Tuple[int, int] = (32, 32),
    temperature_range: Tuple[float, float] = (1.5, 3.5),
    n_temperatures: int = 40,
    n_configs_per_temp: int = 150,
    equilibration_steps: int = 50000,
    sampling_interval: int = 100,
    output_file: Path = None
) -> str:
    """
    Generate 2D Ising model dataset.
    
    Args:
        lattice_size: 2D lattice dimensions
        temperature_range: Temperature range
        n_temperatures: Number of temperature points
        n_configs_per_temp: Configurations per temperature
        equilibration_steps: Equilibration steps
        sampling_interval: Sampling interval
        output_file: Output HDF5 file path
        
    Returns:
        Path to generated data file
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Generating 2D Ising data: lattice_size={lattice_size}, "
               f"T_range={temperature_range}, n_temps={n_temperatures}")
    
    # Create 2D Ising simulator
    simulator = create_ising_simulator(
        lattice_size=lattice_size,
        temperature=1.0,
        coupling_strength=1.0,
        magnetic_field=0.0
    )
    
    # Generate temperature series
    result = simulator.simulate_temperature_series(
        temperature_range=temperature_range,
        n_temperatures=n_temperatures,
        n_configs_per_temp=n_configs_per_temp,
        sampling_interval=sampling_interval,
        equilibration_steps=equilibration_steps
    )
    
    logger.info(f"Generated {len(result.configurations)} configurations")
    
    # Save to HDF5
    if output_file is None:
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        output_file = data_dir / f'ising_2d_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'
    
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('configurations', data=result.configurations)
        f.create_dataset('temperatures', data=result.temperatures)
        f.create_dataset('magnetizations', data=result.order_parameters)
        f.create_dataset('energies', data=result.energies)
        
        # Metadata
        meta = f.create_group('metadata')
        meta.attrs['model_name'] = '2D Ising'
        meta.attrs['lattice_size'] = lattice_size
        meta.attrs['temperature_range'] = temperature_range
        meta.attrs['n_temperatures'] = n_temperatures
        meta.attrs['n_configs_per_temp'] = n_configs_per_temp
        meta.attrs['theoretical_tc'] = 2.269  # 2D Ising critical temperature
        meta.attrs['theoretical_beta'] = 0.125  # 2D Ising β exponent
        meta.attrs['theoretical_nu'] = 1.0  # 2D Ising ν exponent
    
    logger.info(f"Saved 2D Ising data to {output_file}")
    return str(output_file)


def generate_xy_data(
    lattice_size: Tuple[int, int] = (32, 32),
    temperature_range: Tuple[float, float] = (0.5, 1.5),
    n_temperatures: int = 40,
    n_configs_per_temp: int = 150,
    equilibration_steps: int = 50000,
    sampling_interval: int = 100,
    output_file: Path = None
) -> str:
    """
    Generate 2D XY model dataset.
    
    Args:
        lattice_size: 2D lattice dimensions
        temperature_range: Temperature range
        n_temperatures: Number of temperature points
        n_configs_per_temp: Configurations per temperature
        equilibration_steps: Equilibration steps
        sampling_interval: Sampling interval
        output_file: Output HDF5 file path
        
    Returns:
        Path to generated data file
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Generating 2D XY data: lattice_size={lattice_size}, "
               f"T_range={temperature_range}, n_temps={n_temperatures}")
    
    # Create XY simulator
    simulator = create_xy_simulator(
        lattice_size=lattice_size,
        temperature=1.0,
        coupling_strength=1.0
    )
    
    # Generate temperature series
    result = simulator.simulate_temperature_series(
        temperature_range=temperature_range,
        n_temperatures=n_temperatures,
        n_configs_per_temp=n_configs_per_temp,
        sampling_interval=sampling_interval,
        equilibration_steps=equilibration_steps
    )
    
    logger.info(f"Generated {len(result.configurations)} configurations")
    
    # For XY model, convert angles to (cos, sin) representation for VAE
    n_configs = len(result.configurations)
    height, width = lattice_size
    
    # Create 2-channel representation
    xy_configs = np.zeros((n_configs, 2, height, width))
    xy_configs[:, 0, :, :] = np.cos(result.configurations)
    xy_configs[:, 1, :, :] = np.sin(result.configurations)
    
    # Save to HDF5
    if output_file is None:
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        output_file = data_dir / f'xy_2d_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'
    
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('configurations', data=xy_configs)
        f.create_dataset('temperatures', data=result.temperatures)
        f.create_dataset('magnetizations', data=result.order_parameters)
        f.create_dataset('energies', data=result.energies)
        
        # Metadata
        meta = f.create_group('metadata')
        meta.attrs['model_name'] = '2D XY'
        meta.attrs['lattice_size'] = lattice_size
        meta.attrs['temperature_range'] = temperature_range
        meta.attrs['n_temperatures'] = n_temperatures
        meta.attrs['n_configs_per_temp'] = n_configs_per_temp
        meta.attrs['theoretical_tc'] = 0.893  # KT transition temperature
        meta.attrs['theoretical_beta'] = 0.0  # No conventional order parameter
        meta.attrs['theoretical_nu'] = float('inf')  # KT transition
    
    logger.info(f"Saved 2D XY data to {output_file}")
    return str(output_file)


def generate_potts_data(
    lattice_size: Tuple[int, int] = (32, 32),
    temperature_range: Tuple[float, float] = (0.5, 1.5),
    n_temperatures: int = 40,
    n_configs_per_temp: int = 150,
    equilibration_steps: int = 50000,
    sampling_interval: int = 100,
    output_file: Path = None
) -> str:
    """
    Generate 3-state Potts model dataset.
    
    Args:
        lattice_size: 2D lattice dimensions
        temperature_range: Temperature range
        n_temperatures: Number of temperature points
        n_configs_per_temp: Configurations per temperature
        equilibration_steps: Equilibration steps
        sampling_interval: Sampling interval
        output_file: Output HDF5 file path
        
    Returns:
        Path to generated data file
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Generating 3-state Potts data: lattice_size={lattice_size}, "
               f"T_range={temperature_range}, n_temps={n_temperatures}")
    
    # Create Potts simulator
    simulator = create_potts_simulator(
        lattice_size=lattice_size,
        temperature=1.0,
        coupling_strength=1.0
    )
    
    # Generate temperature series
    result = simulator.simulate_temperature_series(
        temperature_range=temperature_range,
        n_temperatures=n_temperatures,
        n_configs_per_temp=n_configs_per_temp,
        sampling_interval=sampling_interval,
        equilibration_steps=equilibration_steps
    )
    
    logger.info(f"Generated {len(result.configurations)} configurations")
    
    # For Potts model, convert to one-hot encoding
    n_configs = len(result.configurations)
    height, width = lattice_size
    n_states = 3
    
    # One-hot encode
    potts_configs = np.zeros((n_configs, n_states, height, width))
    for i, config in enumerate(result.configurations):
        for state in range(n_states):
            potts_configs[i, state, :, :] = (config == state).astype(float)
    
    # Save to HDF5
    if output_file is None:
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        output_file = data_dir / f'potts_3state_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'
    
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('configurations', data=potts_configs)
        f.create_dataset('temperatures', data=result.temperatures)
        f.create_dataset('magnetizations', data=result.order_parameters)
        f.create_dataset('energies', data=result.energies)
        
        # Metadata
        meta = f.create_group('metadata')
        meta.attrs['model_name'] = '3-state Potts'
        meta.attrs['lattice_size'] = lattice_size
        meta.attrs['temperature_range'] = temperature_range
        meta.attrs['n_temperatures'] = n_temperatures
        meta.attrs['n_configs_per_temp'] = n_configs_per_temp
        meta.attrs['theoretical_tc'] = 1.005  # 3-state Potts critical temperature
        meta.attrs['theoretical_beta'] = 0.111  # 3-state Potts β exponent
        meta.attrs['theoretical_nu'] = 0.833  # 3-state Potts ν exponent
    
    logger.info(f"Saved 3-state Potts data to {output_file}")
    return str(output_file)


def validate_system(
    data_file: str,
    system_name: str,
    system_type: str,
    target_accuracy: float,
    config: RealValidationConfig
) -> Dict[str, Any]:
    """
    Validate a single physical system.
    
    Args:
        data_file: Path to data file
        system_name: Display name of system
        system_type: System type identifier
        target_accuracy: Target accuracy threshold
        config: Validation configuration
        
    Returns:
        Dictionary with validation results
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Validating {system_name}")
    logger.info(f"Target accuracy: {target_accuracy}%")
    logger.info(f"{'='*60}\n")
    
    # Create validation pipeline
    pipeline = RealAccuracyValidationPipeline(config)
    
    # Run validation
    try:
        results = pipeline.validate_system_accuracy(
            data_file_path=data_file,
            system_type=system_type
        )
        
        # Check if target met
        meets_target = results.overall_accuracy >= target_accuracy
        
        logger.info(f"\n{system_name} Results:")
        logger.info(f"  Overall Accuracy: {results.overall_accuracy:.1f}%")
        logger.info(f"  Target: {target_accuracy}%")
        logger.info(f"  Status: {'✓ PASSED' if meets_target else '✗ FAILED'}")
        logger.info(f"  Reliability Grade: {results.reliability_grade}")
        
        return {
            'system_name': system_name,
            'system_type': system_type,
            'overall_accuracy': results.overall_accuracy,
            'target_accuracy': target_accuracy,
            'meets_target': meets_target,
            'reliability_grade': results.reliability_grade,
            'latent_mag_correlation': results.latent_magnetization_correlation,
            'extraction_quality': results.extraction_quality_score,
            'runtime': results.total_runtime,
            'full_results': results
        }
        
    except Exception as e:
        logger.error(f"Validation failed for {system_name}: {e}")
        return {
            'system_name': system_name,
            'system_type': system_type,
            'overall_accuracy': 0.0,
            'target_accuracy': target_accuracy,
            'meets_target': False,
            'reliability_grade': 'F',
            'error': str(e)
        }


def create_multi_system_report(
    results: List[Dict[str, Any]],
    output_dir: Path
) -> None:
    """
    Create comprehensive multi-system validation report.
    
    Args:
        results: List of validation results for each system
        output_dir: Output directory for report
    """
    logger = logging.getLogger(__name__)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary report
    report = {
        'validation_timestamp': datetime.now().isoformat(),
        'systems_tested': len(results),
        'systems_passed': sum(1 for r in results if r['meets_target']),
        'overall_success_rate': sum(1 for r in results if r['meets_target']) / len(results) * 100,
        'system_results': []
    }
    
    for result in results:
        system_summary = {
            'system_name': result['system_name'],
            'system_type': result['system_type'],
            'overall_accuracy': result['overall_accuracy'],
            'target_accuracy': result['target_accuracy'],
            'meets_target': result['meets_target'],
            'reliability_grade': result['reliability_grade'],
            'latent_mag_correlation': result.get('latent_mag_correlation', 0.0),
            'extraction_quality': result.get('extraction_quality', 0.0),
            'runtime': result.get('runtime', 0.0)
        }
        report['system_results'].append(system_summary)
    
    # Save JSON report
    report_file = output_dir / 'multi_system_validation_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\nMulti-System Validation Report saved to {report_file}")
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("MULTI-SYSTEM VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Systems Tested: {report['systems_tested']}")
    logger.info(f"Systems Passed: {report['systems_passed']}")
    logger.info(f"Success Rate: {report['overall_success_rate']:.1f}%")
    logger.info(f"\nIndividual System Results:")
    
    for result in results:
        status = "✓ PASS" if result['meets_target'] else "✗ FAIL"
        logger.info(f"  {result['system_name']:20s}: {result['overall_accuracy']:5.1f}% "
                   f"(target: {result['target_accuracy']:.0f}%) {status}")
    
    logger.info(f"{'='*60}\n")


def main():
    """Main function for multi-system validation."""
    parser = argparse.ArgumentParser(
        description='Validate Prometheus on multiple physical systems'
    )
    parser.add_argument('--systems', nargs='+', 
                       choices=['2d_ising', 'xy', 'potts', 'all'],
                       default=['all'],
                       help='Systems to validate')
    parser.add_argument('--generate-data', action='store_true',
                       help='Generate new data (otherwise use existing)')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--output-dir', type=str, 
                       default='results/multi_system_validation',
                       help='Output directory')
    parser.add_argument('--vae-epochs', type=int, default=50,
                       help='VAE training epochs')
    parser.add_argument('--n-configs', type=int, default=150,
                       help='Configurations per temperature')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    logger.info("Starting multi-system validation")
    logger.info(f"Systems to validate: {args.systems}")
    
    # Determine which systems to test
    if 'all' in args.systems:
        systems_to_test = ['2d_ising', 'xy', 'potts']
    else:
        systems_to_test = args.systems
    
    # Create validation configuration
    val_config = RealValidationConfig(
        vae_epochs=args.vae_epochs,
        vae_batch_size=64,
        vae_learning_rate=1e-3,
        use_physics_informed_loss=True,
        bootstrap_samples=1000,
        random_seed=42,
        save_results=True,
        results_dir=args.output_dir,
        create_visualizations=True
    )
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)
    
    results = []
    
    # Test 2D Ising
    if '2d_ising' in systems_to_test:
        if args.generate_data:
            data_file = generate_2d_ising_data(
                n_configs_per_temp=args.n_configs
            )
        else:
            # Look for existing file
            data_file = str(data_dir / 'ising_2d_latest.h5')
            if not Path(data_file).exists():
                logger.info("No existing 2D Ising data found, generating...")
                data_file = generate_2d_ising_data(
                    n_configs_per_temp=args.n_configs
                )
        
        result = validate_system(
            data_file=data_file,
            system_name='2D Ising Model',
            system_type='ising_2d',
            target_accuracy=65.0,
            config=val_config
        )
        results.append(result)
    
    # Test 2D XY
    if 'xy' in systems_to_test:
        if args.generate_data:
            data_file = generate_xy_data(
                n_configs_per_temp=args.n_configs
            )
        else:
            data_file = str(data_dir / 'xy_2d_latest.h5')
            if not Path(data_file).exists():
                logger.info("No existing XY data found, generating...")
                data_file = generate_xy_data(
                    n_configs_per_temp=args.n_configs
                )
        
        result = validate_system(
            data_file=data_file,
            system_name='2D XY Model',
            system_type='xy_2d',
            target_accuracy=60.0,
            config=val_config
        )
        results.append(result)
    
    # Test 3-state Potts
    if 'potts' in systems_to_test:
        if args.generate_data:
            data_file = generate_potts_data(
                n_configs_per_temp=args.n_configs
            )
        else:
            data_file = str(data_dir / 'potts_3state_latest.h5')
            if not Path(data_file).exists():
                logger.info("No existing Potts data found, generating...")
                data_file = generate_potts_data(
                    n_configs_per_temp=args.n_configs
                )
        
        result = validate_system(
            data_file=data_file,
            system_name='3-State Potts Model',
            system_type='potts_3state',
            target_accuracy=60.0,
            config=val_config
        )
        results.append(result)
    
    # Create comprehensive report
    output_dir = Path(args.output_dir)
    create_multi_system_report(results, output_dir)
    
    logger.info("Multi-system validation completed")


if __name__ == '__main__':
    main()
