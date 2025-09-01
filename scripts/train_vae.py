#!/usr/bin/env python3
"""
VAE Training Script for Prometheus Project

This script trains a Convolutional Variational Autoencoder on Ising model data
to learn latent representations of spin configurations.
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ConvolutionalVAE
from src.training import VAETrainer, EarlyStopping, LearningRateScheduler
from src.data import DataPreprocessor
from src.utils.config import PrometheusConfig, ConfigLoader
from src.utils.logging_utils import setup_logging
from src.utils.reproducibility import ReproducibilityManager


def main():
    parser = argparse.ArgumentParser(description='Train VAE on Ising model data')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True, help='Path to preprocessed HDF5 dataset')
    parser.add_argument('--output-dir', type=str, help='Output directory for models and results')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume training from')
    parser.add_argument('--epochs', type=int, help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--learning-rate', type=float, help='Learning rate (overrides config)')
    parser.add_argument('--beta', type=float, help='Beta parameter for β-VAE (overrides config)')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], help='Device to use')
    parser.add_argument('--no-early-stopping', action='store_true', help='Disable early stopping')
    parser.add_argument('--validate-only', action='store_true', help='Only run validation on existing model')
    
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader()
    if args.config:
        config = config_loader.load_config(args.config)
    else:
        config = PrometheusConfig()
    
    # Override configuration with command line arguments
    if args.output_dir:
        config.models_dir = args.output_dir
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.beta:
        config.vae.beta = args.beta
    if args.device:
        config.device = args.device
    
    # Setup logging
    setup_logging(config.logging)
    
    # Set random seeds for reproducibility
    repro_manager = ReproducibilityManager(config.seed)
    repro_manager.set_seeds()
    
    # Determine device
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    
    print("=" * 60)
    print("Prometheus VAE Training")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Dataset: {args.data}")
    print(f"  Device: {device}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Number of epochs: {config.training.num_epochs}")
    print(f"  Beta parameter: {config.vae.beta}")
    print(f"  Latent dimensions: {config.vae.latent_dim}")
    print(f"  Early stopping: {not args.no_early_stopping}")
    if args.resume:
        print(f"  Resume from: {args.resume}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    preprocessor = DataPreprocessor(config)
    
    # Verify dataset exists
    dataset_path = Path(args.data)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}")
    
    # Load dataset info
    dataset_info = preprocessor.load_dataset_info(args.data)
    print(f"  Total configurations: {dataset_info['n_configurations']}")
    print(f"  Configuration shape: {dataset_info['configuration_shape']}")
    print(f"  Train/Val/Test split: {dataset_info['train_size']}/{dataset_info['val_size']}/{dataset_info['test_size']}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = preprocessor.create_dataloaders(
        args.data,
        batch_size=config.training.batch_size,
        num_workers=4 if device.type == 'cuda' else 0
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Initialize model
    print("\nInitializing VAE model...")
    model = ConvolutionalVAE(
        input_shape=tuple(config.vae.input_shape),
        latent_dim=config.vae.latent_dim,
        encoder_channels=config.vae.encoder_channels,
        decoder_channels=config.vae.decoder_channels,
        kernel_sizes=config.vae.kernel_sizes
    ).to(device)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize trainer
    trainer = VAETrainer(model, config, device)
    
    # Note: Early stopping and learning rate scheduler are handled internally by VAETrainer
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming training from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
        print(f"  Resuming from epoch {trainer.current_epoch}")
        print(f"  Best validation loss: {trainer.best_val_loss}")
    
    # Validation-only mode
    if args.validate_only:
        if not args.resume:
            raise ValueError("--validate-only requires --resume to specify model checkpoint")
        
        print("\nRunning validation only...")
        val_metrics = trainer.validate_epoch(val_loader)
        test_metrics = trainer.validate_epoch(test_loader)
        
        print(f"Validation Results:")
        print(f"  Loss: {val_metrics['loss']:.6f}")
        print(f"  Reconstruction Loss: {val_metrics['recon_loss']:.6f}")
        print(f"  KL Divergence: {val_metrics['kl_loss']:.6f}")
        
        print(f"Test Results:")
        print(f"  Loss: {test_metrics['loss']:.6f}")
        print(f"  Reconstruction Loss: {test_metrics['recon_loss']:.6f}")
        print(f"  KL Divergence: {test_metrics['kl_loss']:.6f}")
        
        return
    
    # Training loop
    remaining_epochs = config.training.num_epochs - trainer.current_epoch
    print(f"\nStarting training for {remaining_epochs} epochs...")
    print("-" * 60)
    
    try:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.training.num_epochs,
            enable_early_stopping=not args.no_early_stopping
        )
        
        print("\nTraining completed successfully!")
        
        # Final evaluation
        print("\nFinal evaluation...")
        final_val_metrics = trainer.validate_epoch(val_loader)
        final_test_metrics = trainer.validate_epoch(test_loader)
        
        print(f"Final Validation Results:")
        print(f"  Loss: {final_val_metrics['loss']:.6f}")
        print(f"  Reconstruction Loss: {final_val_metrics['recon_loss']:.6f}")
        print(f"  KL Divergence: {final_val_metrics['kl_loss']:.6f}")
        
        print(f"Final Test Results:")
        print(f"  Loss: {final_test_metrics['loss']:.6f}")
        print(f"  Reconstruction Loss: {final_test_metrics['recon_loss']:.6f}")
        print(f"  KL Divergence: {final_test_metrics['kl_loss']:.6f}")
        
        # Save final model
        final_model_path = trainer.save_model("final_model.pth")
        print(f"\nFinal model saved to: {final_model_path}")
        
        # Save training history
        history_path = Path(config.models_dir) / "training_history.json"
        trainer.save_training_history(history, str(history_path))
        print(f"Training history saved to: {history_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        print("Saving current model state...")
        
        interrupted_path = trainer.save_checkpoint("interrupted_checkpoint.pth")
        print(f"Checkpoint saved to: {interrupted_path}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        print("Saving current model state...")
        
        error_path = trainer.save_checkpoint("error_checkpoint.pth")
        print(f"Checkpoint saved to: {error_path}")
        raise
    
    print("\n" + "=" * 60)
    print("VAE Training Complete!")
    print("=" * 60)
    print(f"Model ready for latent space analysis.")
    print(f"Use the trained model for phase detection and order parameter discovery.")


if __name__ == "__main__":
    main()