#!/usr/bin/env python3
"""
Data Generation Script for Prometheus Project

This script generates and preprocesses Ising model data for VAE training.
It creates comprehensive datasets across temperature ranges with proper
validation and preprocessing for the machine learning pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataGenerator, DataPreprocessor
from src.utils.config import PrometheusConfig, ConfigLoader
from src.utils.logging_utils import setup_logging
from src.utils.reproducibility import ReproducibilityManager


def main():
    parser = argparse.ArgumentParser(description='Generate Ising model training data')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='data', help='Output directory')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    parser.add_argument('--n-processes', type=int, help='Number of parallel processes')
    parser.add_argument('--validate', action='store_true', default=True, help='Validate generated data')
    
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader()
    if args.config:
        config = config_loader.load_config(args.config)
    else:
        config = PrometheusConfig()
    
    # Override data directory if specified
    if args.output_dir:
        config.data_dir = args.output_dir
    
    # Setup logging
    setup_logging(config.logging)
    
    # Set random seeds for reproducibility
    repro_manager = ReproducibilityManager(config.seed)
    repro_manager.set_seeds()
    
    print("=" * 60)
    print("Prometheus Ising Model Data Generation")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Lattice size: {config.ising.lattice_size}")
    print(f"  Temperature range: {config.ising.temperature_range}")
    print(f"  Number of temperatures: {config.ising.n_temperatures}")
    print(f"  Configurations per temperature: {config.ising.n_configs_per_temp}")
    print(f"  Total target configurations: {config.ising.n_temperatures * config.ising.n_configs_per_temp}")
    print(f"  Output directory: {config.data_dir}")
    print(f"  Parallel processing: {args.parallel}")
    if args.parallel and args.n_processes:
        print(f"  Number of processes: {args.n_processes}")
    print()
    
    # Initialize data generator
    generator = DataGenerator(config)
    
    # Add progress callback
    def progress_callback(progress):
        percent = progress.progress_percentage
        elapsed = progress.elapsed_time
        eta = progress.estimate_completion_time()
        
        print(f"\rProgress: {percent:.1f}% ({progress.completed_temperatures}/{progress.total_temperatures} temps) "
              f"- Elapsed: {elapsed:.1f}s", end="")
        if eta:
            print(f" - ETA: {eta:.1f}s", end="")
        
        if progress.completed_temperatures == progress.total_temperatures:
            print()  # New line when complete
    
    generator.add_progress_callback(progress_callback)
    
    # Generate data
    print("Starting data generation...")
    result, validation_result = generator.generate_dataset(
        use_parallel=args.parallel,
        n_processes=args.n_processes,
        validate_data=args.validate
    )
    
    print(f"Data generation complete!")
    print(f"  Generated {result.total_configurations} configurations")
    print(f"  Actual temperatures: {len(result.temperatures)}")
    
    if validation_result:
        if validation_result.is_valid:
            print(f"  Data validation: PASSED")
        else:
            print(f"  Data validation: FAILED ({validation_result.error_rate:.2f}% error rate)")
            print(f"    Errors: {validation_result.spin_value_errors} spin values, "
                  f"{validation_result.shape_errors} shapes, {validation_result.temperature_errors} temperatures")
    
    # Save raw data
    raw_path = generator.save_dataset(result, include_temperature_labels=False)
    print(f"  Raw data saved to: {raw_path}")
    
    # Preprocess data
    print("\nStarting data preprocessing...")
    preprocessor = DataPreprocessor(config)
    
    hdf5_path = preprocessor.process_dataset(
        result,
        normalization_method='sigmoid',
        split_ratios=(0.7, 0.15, 0.15)
    )
    
    print(f"Preprocessing complete!")
    print(f"  Processed data saved to: {hdf5_path}")
    
    # Load and display dataset info
    info = preprocessor.load_dataset_info(hdf5_path)
    print(f"  Dataset info:")
    print(f"    Total configurations: {info['n_configurations']}")
    print(f"    Configuration shape: {info['configuration_shape']}")
    print(f"    Train/Val/Test split: {info['train_size']}/{info['val_size']}/{info['test_size']}")
    print(f"    Split ratios: {info['split_ratios']}")
    print(f"    Normalization: {info['normalization_method']}")
    
    # Create sample dataloaders
    print("\nCreating sample DataLoaders...")
    train_loader, val_loader, test_loader = preprocessor.create_dataloaders(
        hdf5_path,
        batch_size=32,
        num_workers=0  # Avoid multiprocessing issues
    )
    
    print(f"  Train loader: {len(train_loader)} batches")
    print(f"  Val loader: {len(val_loader)} batches")
    print(f"  Test loader: {len(test_loader)} batches")
    
    # Test loading a batch
    print("\nTesting data loading...")
    for batch in train_loader:
        print(f"  Sample batch shape: {batch.shape}")
        print(f"  Sample batch dtype: {batch.dtype}")
        print(f"  Sample batch range: [{batch.min():.3f}, {batch.max():.3f}]")
        break
    
    print("\n" + "=" * 60)
    print("Data generation and preprocessing complete!")
    print("=" * 60)
    print(f"Ready for VAE training with:")
    print(f"  Dataset: {hdf5_path}")
    print(f"  Total samples: {info['n_configurations']}")
    print(f"  Input shape: {info['configuration_shape']}")


if __name__ == "__main__":
    main()