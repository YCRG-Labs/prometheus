#!/usr/bin/env python3
"""
Script to generate comprehensive 3D Ising dataset.

This script implements task 3.1 from the PRE paper specification:
- Create temperature sweep T ∈ [3.0, 6.0] with appropriate resolution
- Generate configurations for system sizes L ∈ {8, 16, 32}
- Implement 1000 configurations per temperature with proper sampling intervals
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_generator_3d import (
    generate_3d_ising_dataset,
    create_default_3d_config,
    DataGenerationConfig3D
)
import logging


def main():
    """Main function to generate 3D Ising dataset."""
    parser = argparse.ArgumentParser(description='Generate 3D Ising dataset for PRE paper')
    
    # Configuration options
    parser.add_argument('--temp-min', type=float, default=3.0,
                       help='Minimum temperature (default: 3.0)')
    parser.add_argument('--temp-max', type=float, default=6.0,
                       help='Maximum temperature (default: 6.0)')
    parser.add_argument('--temp-resolution', type=int, default=61,
                       help='Number of temperature points (default: 61)')
    parser.add_argument('--system-sizes', type=int, nargs='+', default=[8, 16, 32],
                       help='System sizes to generate (default: 8 16 32)')
    parser.add_argument('--configs-per-temp', type=int, default=1000,
                       help='Configurations per temperature (default: 1000)')
    parser.add_argument('--sampling-interval', type=int, default=100,
                       help='Monte Carlo steps between samples (default: 100)')
    
    # Processing options
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    parser.add_argument('--processes', type=int, default=None,
                       help='Number of parallel processes (default: auto)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory (default: data)')
    parser.add_argument('--output-format', choices=['hdf5', 'npz'], default='hdf5',
                       help='Output file format (default: hdf5)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save dataset to file')
    
    # Quality options
    parser.add_argument('--quality-threshold', type=float, default=0.7,
                       help='Equilibration quality threshold (default: 0.7)')
    
    # Logging options
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level (default: INFO)')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Log file path (default: console only)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=getattr(logging, args.log_level), 
                       filename=args.log_file if args.log_file else None)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting 3D Ising dataset generation")
    logger.info(f"Arguments: {vars(args)}")
    
    # Create configuration
    config = DataGenerationConfig3D(
        temperature_range=(args.temp_min, args.temp_max),
        temperature_resolution=args.temp_resolution,
        system_sizes=args.system_sizes,
        n_configs_per_temp=args.configs_per_temp,
        sampling_interval=args.sampling_interval,
        equilibration_quality_threshold=args.quality_threshold,
        parallel_processes=args.processes,
        output_dir=args.output_dir
    )
    
    logger.info(f"Configuration:")
    logger.info(f"  Temperature range: {config.temperature_range}")
    logger.info(f"  Temperature resolution: {config.temperature_resolution}")
    logger.info(f"  System sizes: {config.system_sizes}")
    logger.info(f"  Configurations per temperature: {config.n_configs_per_temp}")
    logger.info(f"  Sampling interval: {config.sampling_interval}")
    logger.info(f"  Quality threshold: {config.equilibration_quality_threshold}")
    logger.info(f"  Parallel processing: {not args.no_parallel}")
    logger.info(f"  Output directory: {config.output_dir}")
    
    # Estimate total configurations and time
    total_configs = (len(config.system_sizes) * 
                    config.temperature_resolution * 
                    config.n_configs_per_temp)
    
    logger.info(f"Estimated total configurations: {total_configs:,}")
    
    # Estimate time (rough approximation)
    # Assume ~0.1 seconds per configuration for small systems, scaling with size
    avg_size = sum(config.system_sizes) / len(config.system_sizes)
    time_per_config = 0.1 * (avg_size / 16) ** 2  # Quadratic scaling approximation
    estimated_time_hours = (total_configs * time_per_config) / 3600
    
    logger.info(f"Estimated generation time: {estimated_time_hours:.1f} hours")
    
    # Confirm with user for large datasets
    if total_configs > 50000:
        response = input(f"This will generate {total_configs:,} configurations "
                        f"(~{estimated_time_hours:.1f} hours). Continue? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            logger.info("Generation cancelled by user")
            return
    
    try:
        # Generate dataset
        start_time = time.time()
        
        result = generate_3d_ising_dataset(
            config=config,
            use_parallel=not args.no_parallel,
            save_dataset=not args.no_save,
            output_format=args.output_format
        )
        
        generation_time = time.time() - start_time
        
        # Print summary
        logger.info("=" * 60)
        logger.info("3D ISING DATASET GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total configurations generated: {result.total_configurations:,}")
        logger.info(f"System sizes: {list(result.system_size_results.keys())}")
        logger.info(f"Temperature range: {config.temperature_range}")
        logger.info(f"Generation time: {generation_time:.1f} seconds ({generation_time/3600:.2f} hours)")
        logger.info(f"Theoretical Tc: {result.theoretical_tc:.3f}")
        
        # Validation summary
        validation = result.validation_results
        logger.info(f"Dataset validation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
        logger.info(f"Error rate: {validation['error_rate']:.4f}")
        
        if validation['issues']:
            logger.warning("Validation issues:")
            for issue in validation['issues']:
                logger.warning(f"  - {issue}")
        
        # System size summaries
        logger.info("\nSystem size summaries:")
        for size, size_result in result.system_size_results.items():
            eq_success_rate = size_result.metadata['equilibration_success_rate']
            logger.info(f"  L={size}: {len(size_result.temperatures)} temperatures, "
                       f"equilibration success: {eq_success_rate:.1%}")
        
        logger.info("\nDataset generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        raise


if __name__ == "__main__":
    main()