#!/usr/bin/env python3
"""
Comprehensive Parameter Sweep Execution Script

This script executes systematic hyperparameter optimization experiments
across all architecture and training parameter combinations with automated
experiment tracking and convergence analysis.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from utils.config import PrometheusConfig
from utils.logging_utils import setup_logging
from data.data_generator import DataGenerator
from data.preprocessing import DataPreprocessor
from optimization.parameter_sweep import ParameterSweepOrchestrator, SweepConfig
from optimization.experiment_tracking import ExperimentTracker, ExperimentConfig

logger = logging.getLogger(__name__)


def setup_data_loaders(config: PrometheusConfig) -> tuple:
    """
    Setup data loaders for parameter sweep experiments.
    
    Args:
        config: Prometheus configuration
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info("Setting up data loaders for parameter sweep")
    
    # Check if processed data exists
    data_dir = Path("data")
    processed_files = list(data_dir.glob("ising_processed_*.h5"))
    
    if not processed_files:
        logger.info("No processed data found, generating new dataset")
        
        # Generate data
        data_generator = DataGenerator(config)
        raw_data_path = data_generator.generate_dataset()
        
        # Preprocess data
        preprocessor = DataPreprocessor(config)
        processed_data_path = preprocessor.preprocess_dataset(raw_data_path)
    else:
        # Use most recent processed data
        processed_data_path = max(processed_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using existing processed data: {processed_data_path}")
    
    # Create data loaders
    preprocessor = DataPreprocessor(config)
    train_loader, val_loader, test_loader = preprocessor.create_data_loaders(
        processed_data_path,
        batch_size=config.training.batch_size
    )
    
    logger.info(f"Data loaders created:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def create_sweep_config(args) -> SweepConfig:
    """
    Create parameter sweep configuration from command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        SweepConfig object
    """
    # Define comprehensive parameter ranges
    latent_dims = [2, 4, 8, 16] if not args.quick else [2, 4]
    beta_values = [0.1, 0.5, 1.0, 2.0, 4.0] if not args.quick else [0.5, 1.0, 2.0]
    encoder_layers = [3, 4, 5] if not args.quick else [3, 4]
    activations = ['relu', 'leaky_relu', 'elu'] if not args.quick else ['relu', 'leaky_relu']
    learning_rates = [1e-4, 5e-4, 1e-3, 2e-3] if not args.quick else [5e-4, 1e-3]
    batch_sizes = [64, 128, 256] if not args.quick else [128, 256]
    
    # Override with command line arguments if provided
    if args.latent_dims:
        latent_dims = [int(x) for x in args.latent_dims.split(',')]
    if args.beta_values:
        beta_values = [float(x) for x in args.beta_values.split(',')]
    if args.encoder_layers:
        encoder_layers = [int(x) for x in args.encoder_layers.split(',')]
    if args.activations:
        activations = args.activations.split(',')
    if args.learning_rates:
        learning_rates = [float(x) for x in args.learning_rates.split(',')]
    if args.batch_sizes:
        batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    
    sweep_config = SweepConfig(
        latent_dims=latent_dims,
        beta_values=beta_values,
        encoder_layers=encoder_layers,
        activations=activations,
        learning_rates=learning_rates,
        batch_sizes=batch_sizes,
        max_experiments=args.max_experiments,
        parallel_jobs=args.parallel_jobs,
        timeout_per_experiment=args.timeout,
        strategy=args.strategy,
        random_seed=args.seed,
        enable_early_stopping=args.enable_early_stopping,
        convergence_patience=args.convergence_patience,
        min_improvement=args.min_improvement
    )
    
    return sweep_config


def run_parameter_sweep(config_path: str, args) -> dict:
    """
    Execute comprehensive parameter sweep.
    
    Args:
        config_path: Path to configuration file
        args: Command line arguments
        
    Returns:
        Dictionary containing sweep results
    """
    # Load configuration
    config = PrometheusConfig.from_yaml(config_path)
    
    # Setup data loaders
    train_loader, val_loader, test_loader = setup_data_loaders(config)
    
    # Create sweep configuration
    sweep_config = create_sweep_config(args)
    
    # Log sweep parameters
    logger.info("Parameter sweep configuration:")
    logger.info(f"  Latent dimensions: {sweep_config.latent_dims}")
    logger.info(f"  Beta values: {sweep_config.beta_values}")
    logger.info(f"  Encoder layers: {sweep_config.encoder_layers}")
    logger.info(f"  Activations: {sweep_config.activations}")
    logger.info(f"  Learning rates: {sweep_config.learning_rates}")
    logger.info(f"  Batch sizes: {sweep_config.batch_sizes}")
    logger.info(f"  Strategy: {sweep_config.strategy}")
    logger.info(f"  Max experiments: {sweep_config.max_experiments}")
    logger.info(f"  Parallel jobs: {sweep_config.parallel_jobs}")
    
    # Calculate total possible experiments
    total_combinations = (
        len(sweep_config.latent_dims) * len(sweep_config.beta_values) *
        len(sweep_config.encoder_layers) * len(sweep_config.activations) *
        len(sweep_config.learning_rates) * len(sweep_config.batch_sizes)
    )
    
    actual_experiments = min(total_combinations, sweep_config.max_experiments or total_combinations)
    logger.info(f"Total possible combinations: {total_combinations}")
    logger.info(f"Experiments to run: {actual_experiments}")
    
    # Estimate time
    estimated_time_per_exp = 300  # 5 minutes per experiment (conservative estimate)
    estimated_total_time = (actual_experiments * estimated_time_per_exp) / sweep_config.parallel_jobs
    logger.info(f"Estimated total time: {estimated_total_time / 3600:.1f} hours")
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize parameter sweep orchestrator
    orchestrator = ParameterSweepOrchestrator(
        base_config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        sweep_config=sweep_config,
        results_dir=str(results_dir)
    )
    
    # Execute parameter sweep
    logger.info("Starting comprehensive parameter sweep")
    start_time = time.time()
    
    try:
        sweep_results = orchestrator.run_parameter_sweep()
        
        total_time = time.time() - start_time
        logger.info(f"Parameter sweep completed in {total_time / 3600:.2f} hours")
        
        # Log summary results
        summary = sweep_results.get('sweep_summary', {})
        logger.info("Sweep Summary:")
        logger.info(f"  Total experiments: {summary.get('total_experiments', 0)}")
        logger.info(f"  Successful experiments: {summary.get('successful_experiments', 0)}")
        logger.info(f"  Best overall score: {summary.get('best_overall_score', 0):.4f}")
        logger.info(f"  Convergence achieved: {summary.get('convergence_achieved', False)}")
        
        # Log best configuration
        best_config = sweep_results.get('best_configuration')
        if best_config:
            logger.info("Best Configuration:")
            arch_config = best_config.get('architecture_config', {})
            logger.info(f"  Latent dim: {arch_config.get('latent_dim')}")
            logger.info(f"  Beta: {arch_config.get('beta')}")
            logger.info(f"  Encoder layers: {arch_config.get('encoder_layers')}")
            logger.info(f"  Activation: {arch_config.get('activation')}")
            
            train_config = best_config.get('training_config', {})
            logger.info(f"  Learning rate: {train_config.get('learning_rate')}")
            logger.info(f"  Batch size: {train_config.get('batch_size')}")
            
            physics_metrics = best_config.get('physics_metrics', {})
            logger.info(f"  Physics score: {physics_metrics.get('overall_physics_score', 0):.4f}")
            logger.info(f"  Order param correlation: {physics_metrics.get('order_parameter_correlation', 0):.4f}")
            logger.info(f"  Critical temp error: {physics_metrics.get('critical_temperature_error', 1.0):.4f}")
        
        # Generate sensitivity analysis
        logger.info("Generating parameter sensitivity analysis")
        sensitivity_analysis = orchestrator.generate_sensitivity_analysis()
        
        if sensitivity_analysis:
            logger.info("Parameter Sensitivity Analysis:")
            for param, metrics in sensitivity_analysis.items():
                logger.info(f"  {param}:")
                for metric, values in metrics.items():
                    correlation = values.get('correlation', 0)
                    r2 = values.get('r2_score', 0)
                    logger.info(f"    {metric}: correlation={correlation:.3f}, R²={r2:.3f}")
        
        return sweep_results
        
    except Exception as e:
        logger.error(f"Parameter sweep failed: {str(e)}")
        raise


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run comprehensive parameter sweep experiments")
    
    # Configuration
    parser.add_argument("--config", type=str, default="config/enhanced_training.yaml",
                       help="Path to configuration file")
    parser.add_argument("--results-dir", type=str, default="results/parameter_sweep",
                       help="Directory to save results")
    
    # Parameter ranges (comma-separated values)
    parser.add_argument("--latent-dims", type=str,
                       help="Latent dimensions to test (e.g., '2,4,8,16')")
    parser.add_argument("--beta-values", type=str,
                       help="Beta values to test (e.g., '0.1,0.5,1.0,2.0,4.0')")
    parser.add_argument("--encoder-layers", type=str,
                       help="Encoder layer counts to test (e.g., '3,4,5')")
    parser.add_argument("--activations", type=str,
                       help="Activation functions to test (e.g., 'relu,leaky_relu,elu')")
    parser.add_argument("--learning-rates", type=str,
                       help="Learning rates to test (e.g., '1e-4,5e-4,1e-3')")
    parser.add_argument("--batch-sizes", type=str,
                       help="Batch sizes to test (e.g., '64,128,256')")
    
    # Execution parameters
    parser.add_argument("--max-experiments", type=int,
                       help="Maximum number of experiments to run")
    parser.add_argument("--parallel-jobs", type=int, default=1,
                       help="Number of parallel jobs")
    parser.add_argument("--timeout", type=int, default=3600,
                       help="Timeout per experiment in seconds")
    parser.add_argument("--strategy", type=str, default="grid_search",
                       choices=["grid_search", "random_search", "bayesian"],
                       help="Optimization strategy")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    # Early stopping
    parser.add_argument("--enable-early-stopping", action="store_true", default=True,
                       help="Enable early stopping based on convergence")
    parser.add_argument("--convergence-patience", type=int, default=20,
                       help="Patience for convergence detection")
    parser.add_argument("--min-improvement", type=float, default=0.001,
                       help="Minimum improvement threshold for convergence")
    
    # Quick mode for testing
    parser.add_argument("--quick", action="store_true",
                       help="Run with reduced parameter ranges for quick testing")
    
    # Logging
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    log_dir = Path(args.results_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(
        log_level=args.log_level,
        log_file=str(log_dir / f"parameter_sweep_{int(time.time())}.log")
    )
    
    logger.info("Starting comprehensive parameter sweep")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Quick mode: {args.quick}")
    
    try:
        # Run parameter sweep
        results = run_parameter_sweep(args.config, args)
        
        logger.info("Parameter sweep completed successfully")
        logger.info(f"Results saved to: {args.results_dir}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("PARAMETER SWEEP COMPLETED")
        print("="*60)
        
        summary = results.get('sweep_summary', {})
        print(f"Total experiments: {summary.get('total_experiments', 0)}")
        print(f"Successful experiments: {summary.get('successful_experiments', 0)}")
        print(f"Best overall score: {summary.get('best_overall_score', 0):.4f}")
        print(f"Convergence achieved: {summary.get('convergence_achieved', False)}")
        
        best_config = results.get('best_configuration')
        if best_config:
            print("\nBest Configuration:")
            arch_config = best_config.get('architecture_config', {})
            print(f"  Latent dimensions: {arch_config.get('latent_dim')}")
            print(f"  Beta value: {arch_config.get('beta')}")
            print(f"  Encoder layers: {arch_config.get('encoder_layers')}")
            print(f"  Activation: {arch_config.get('activation')}")
            
            physics_metrics = best_config.get('physics_metrics', {})
            print(f"  Physics score: {physics_metrics.get('overall_physics_score', 0):.4f}")
            print(f"  Order parameter correlation: {physics_metrics.get('order_parameter_correlation', 0):.4f}")
            print(f"  Critical temperature error: {physics_metrics.get('critical_temperature_error', 1.0):.4f}")
        
        print(f"\nDetailed results saved to: {args.results_dir}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Parameter sweep failed: {str(e)}")
        print(f"\nERROR: Parameter sweep failed - {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()