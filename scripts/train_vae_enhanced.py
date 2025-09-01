#!/usr/bin/env python3
"""
Enhanced VAE Training Script

This script demonstrates the enhanced training capabilities including:
- Advanced learning rate scheduling (cosine annealing, warm restarts)
- Data augmentation (rotations, reflections, spin flips)
- Ensemble training with multiple random initializations
- Progressive training starting with lower resolution

Usage:
    python scripts/train_vae_enhanced.py --config config/enhanced_training.yaml
    python scripts/train_vae_enhanced.py --method ensemble --ensemble-size 5
    python scripts/train_vae_enhanced.py --method progressive --progressive-stages 3
"""

import argparse
import logging
import sys
from pathlib import Path
import time
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import ConfigLoader, PrometheusConfig
from utils.logging_utils import setup_logging
from utils.reproducibility import set_reproducible_seeds
from data.preprocessing import DataPreprocessor, IsingDataset
from training.enhanced_trainer import EnhancedTrainingPipeline, create_enhanced_training_config
from models.vae import ConvolutionalVAE


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced VAE Training")
    
    # Configuration
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/example.yaml",
        help="Path to configuration file"
    )
    
    # Training method
    parser.add_argument(
        "--method",
        type=str,
        choices=["standard", "ensemble", "progressive", "auto"],
        default="auto",
        help="Training method to use"
    )
    
    # Data
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to processed HDF5 dataset"
    )
    
    # Ensemble options
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=5,
        help="Number of ensemble members"
    )
    
    # Progressive options
    parser.add_argument(
        "--progressive-stages",
        type=int,
        default=3,
        help="Number of progressive training stages"
    )
    
    # Augmentation options
    parser.add_argument(
        "--augmentation",
        type=str,
        choices=["none", "conservative", "standard", "aggressive"],
        default="standard",
        help="Data augmentation type"
    )
    
    parser.add_argument(
        "--augmentation-prob",
        type=float,
        default=0.5,
        help="Data augmentation probability"
    )
    
    # Advanced scheduler options
    parser.add_argument(
        "--advanced-scheduler",
        type=str,
        choices=["none", "cosine_warm_restarts", "warmup_cosine", "cyclic", "adaptive"],
        default="none",
        help="Advanced learning rate scheduler"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/enhanced_training",
        help="Output directory for results"
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (overrides config)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (overrides config)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate (overrides config)"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for training"
    )
    
    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    return parser.parse_args()


def setup_enhanced_config(args) -> PrometheusConfig:
    """Set up enhanced configuration based on arguments."""
    # Load base configuration
    config_loader = ConfigLoader()
    config = config_loader.load_config(args.config)
    
    # Create enhanced configuration
    overrides = {}
    
    # Training method specific settings
    if args.method == "ensemble":
        overrides['ensemble_size'] = args.ensemble_size
    elif args.method == "progressive":
        overrides['progressive_stages'] = args.progressive_stages
    
    # Augmentation settings
    if args.augmentation != "none":
        overrides['use_augmentation'] = True
        overrides['augmentation_type'] = args.augmentation
        overrides['augmentation_probability'] = args.augmentation_prob
    else:
        overrides['use_augmentation'] = False
    
    # Advanced scheduler settings
    if args.advanced_scheduler != "none":
        overrides['advanced_scheduler'] = args.advanced_scheduler
        
        # Set default parameters for each scheduler type
        scheduler_params = {}
        if args.advanced_scheduler == "cosine_warm_restarts":
            scheduler_params = {
                'T_0': 10,
                'T_mult': 2,
                'eta_min': 1e-6
            }
        elif args.advanced_scheduler == "warmup_cosine":
            scheduler_params = {
                'warmup_epochs': 5,
                'max_epochs': args.epochs or config.training.num_epochs,
                'eta_min': 1e-6
            }
        elif args.advanced_scheduler == "cyclic":
            scheduler_params = {
                'base_lr': 1e-5,
                'max_lr': 1e-2,
                'step_size_up': 2000,
                'mode': 'triangular'
            }
        elif args.advanced_scheduler == "adaptive":
            scheduler_params = {
                'patience': 10,
                'factor': 0.5,
                'threshold': 1e-4
            }
        
        overrides['scheduler_params'] = scheduler_params
    
    # Override training parameters if specified
    if args.epochs:
        overrides['num_epochs'] = args.epochs
    if args.batch_size:
        overrides['batch_size'] = args.batch_size
    if args.learning_rate:
        overrides['learning_rate'] = args.learning_rate
    
    # Create enhanced config
    enhanced_config = create_enhanced_training_config(
        config, args.method, **overrides
    )
    
    # Override global settings
    enhanced_config.seed = args.seed
    enhanced_config.device = args.device
    
    return enhanced_config


def load_datasets(data_path: str, config: PrometheusConfig):
    """Load training datasets."""
    logging.info(f"Loading datasets from {data_path}")
    
    # Create datasets
    train_dataset = IsingDataset(data_path, split='train', load_physics=False)
    val_dataset = IsingDataset(data_path, split='val', load_physics=False)
    test_dataset = IsingDataset(data_path, split='test', load_physics=False)
    
    logging.info(f"Loaded datasets: train={len(train_dataset)}, "
                f"val={len(val_dataset)}, test={len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting enhanced VAE training")
    logger.info(f"Training method: {args.method}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Setup configuration
        config = setup_enhanced_config(args)
        logger.info("Configuration loaded and enhanced")
        
        # Set reproducible seeds
        set_reproducible_seeds(config.seed)
        logger.info(f"Set random seed to {config.seed}")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_loader = ConfigLoader()
        config_loader.save_config(config, output_dir / "training_config.yaml")
        
        # Load datasets
        train_dataset, val_dataset, test_dataset = load_datasets(args.data_path, config)
        
        # Create enhanced training pipeline
        pipeline = EnhancedTrainingPipeline(config)
        
        # Log training configuration
        logger.info("Training Configuration:")
        logger.info(f"  Method: {args.method}")
        logger.info(f"  Epochs: {config.training.num_epochs}")
        logger.info(f"  Batch size: {config.training.batch_size}")
        logger.info(f"  Learning rate: {config.training.learning_rate}")
        logger.info(f"  Augmentation: {config.training.use_augmentation} ({config.training.augmentation_type})")
        logger.info(f"  Advanced scheduler: {config.training.advanced_scheduler}")
        
        if config.training.use_ensemble:
            logger.info(f"  Ensemble size: {config.training.ensemble_size}")
        if config.training.use_progressive:
            logger.info(f"  Progressive stages: {config.training.progressive_stages}")
        
        # Execute training
        start_time = time.time()
        
        if args.method == "auto":
            results = pipeline.train_auto(train_dataset, val_dataset)
        elif args.method == "ensemble":
            results = pipeline.train_ensemble(train_dataset, val_dataset)
        elif args.method == "progressive":
            results = pipeline.train_progressive(train_dataset, val_dataset)
        else:  # standard
            results = pipeline.train_standard(train_dataset, val_dataset)
        
        total_time = time.time() - start_time
        
        # Log results
        logger.info(f"Training completed in {total_time:.1f}s")
        logger.info(f"Method used: {results['method']}")
        
        if 'best_val_loss' in results:
            logger.info(f"Best validation loss: {results['best_val_loss']:.6f}")
        elif 'ensemble_results' in results:
            ensemble_results = results['ensemble_results']
            logger.info(f"Ensemble best loss: {ensemble_results['ensemble_best_loss']:.6f}")
            logger.info(f"Ensemble mean loss: {ensemble_results['ensemble_mean_loss']:.6f} ± {ensemble_results['ensemble_std_loss']:.6f}")
        
        # Save results
        save_path = pipeline.save_results(str(output_dir))
        logger.info(f"Results saved to {save_path}")
        
        # Test model on test set if available
        if test_dataset is not None:
            logger.info("Evaluating on test set")
            
            trained_model = pipeline.get_trained_model()
            if trained_model is not None:
                test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=config.training.batch_size,
                    shuffle=False,
                    num_workers=4
                )
                
                trained_model.eval()
                test_loss = 0.0
                n_batches = 0
                
                with torch.no_grad():
                    for batch in test_loader:
                        if isinstance(batch, (list, tuple)):
                            data = batch[0]
                        else:
                            data = batch
                        
                        data = data.to(pipeline.device)
                        reconstruction, mu, logvar = trained_model(data)
                        
                        loss_dict = trained_model.compute_loss(
                            data, reconstruction, mu, logvar, reduction='mean'
                        )
                        
                        test_loss += loss_dict['total_loss'].item()
                        n_batches += 1
                
                test_loss /= n_batches
                logger.info(f"Test loss: {test_loss:.6f}")
        
        logger.info("Enhanced VAE training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()