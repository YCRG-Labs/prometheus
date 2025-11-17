#!/usr/bin/env python3
"""
3D VAE Training Script for Prometheus Project

This script implements task 5.1: Train VAE on 3D Ising configurations
- Execute training with consistent hyperparameters from 2D case
- Monitor training curves and convergence behavior
- Save trained model checkpoints for analysis
"""

import argparse
import sys
import os
import time
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import h5py

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.adaptive_vae import VAEFactory
from src.models.vae_3d import ConvolutionalVAE3D
from src.data.data_loader_utils import AdaptiveDataLoader, DatasetFactory
from src.utils.config import PrometheusConfig


def generate_small_3d_dataset(output_path: str, system_sizes=[32], n_temps=11, n_configs=50):
    """Generate a small 3D dataset for training using direct Monte Carlo."""
    print("Generating small 3D dataset for training...")
    
    # Simple 3D Ising Monte Carlo implementation
    def ising_3d_energy(config):
        """Compute 3D Ising energy with periodic boundary conditions."""
        L = config.shape[0]
        energy = 0.0
        
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    spin = config[i, j, k]
                    # 6 nearest neighbors in 3D
                    neighbors = [
                        config[(i+1)%L, j, k], config[(i-1)%L, j, k],
                        config[i, (j+1)%L, k], config[i, (j-1)%L, k],
                        config[i, j, (k+1)%L], config[i, j, (k-1)%L]
                    ]
                    energy -= spin * sum(neighbors)
        
        return energy / 2  # Avoid double counting
    
    def metropolis_step_3d(config, temperature):
        """Single Metropolis step for 3D Ising model."""
        L = config.shape[0]
        i, j, k = np.random.randint(0, L, 3)
        
        # Calculate energy change
        spin = config[i, j, k]
        neighbors = [
            config[(i+1)%L, j, k], config[(i-1)%L, j, k],
            config[i, (j+1)%L, k], config[i, (j-1)%L, k],
            config[i, j, (k+1)%L], config[i, j, (k-1)%L]
        ]
        
        delta_E = 2 * spin * sum(neighbors)
        
        # Accept or reject
        if delta_E <= 0 or np.random.random() < np.exp(-delta_E / temperature):
            config[i, j, k] = -spin
        
        return config
    
    # Generate configurations
    all_configs = []
    all_temps = []
    all_magnetizations = []
    all_energies = []
    
    temperatures = np.linspace(3.0, 6.0, n_temps)
    
    for L in system_sizes:
        print(f"Generating configurations for L={L}...")
        
        for temp in temperatures:
            configs_at_temp = []
            
            # Initialize random configuration
            config = np.random.choice([-1, 1], size=(L, L, L))
            
            # Equilibration
            for _ in range(1000):
                config = metropolis_step_3d(config, temp)
            
            # Sampling
            for _ in range(n_configs):
                # Sample with interval
                for _ in range(50):
                    config = metropolis_step_3d(config, temp)
                
                # Store configuration and properties
                config_copy = config.copy()
                magnetization = np.mean(config_copy)
                energy = ising_3d_energy(config_copy)
                
                all_configs.append(config_copy)
                all_temps.append(temp)
                all_magnetizations.append(magnetization)
                all_energies.append(energy)
    
    # Convert to arrays - handle different system sizes
    # Group by system size to handle different shapes
    configs_by_size = {}
    temps_by_size = {}
    mags_by_size = {}
    energies_by_size = {}
    
    idx = 0
    for L in system_sizes:
        n_configs_this_size = n_temps * n_configs
        
        configs_this_size = []
        temps_this_size = []
        mags_this_size = []
        energies_this_size = []
        
        for i in range(n_configs_this_size):
            configs_this_size.append(all_configs[idx])
            temps_this_size.append(all_temps[idx])
            mags_this_size.append(all_magnetizations[idx])
            energies_this_size.append(all_energies[idx])
            idx += 1
        
        configs_by_size[L] = np.array(configs_this_size)
        temps_by_size[L] = np.array(temps_this_size)
        mags_by_size[L] = np.array(mags_this_size)
        energies_by_size[L] = np.array(energies_this_size)
    
    # For simplicity, use only the smallest system size for training
    L_train = min(system_sizes)
    all_configs = configs_by_size[L_train]
    all_temps = temps_by_size[L_train]
    all_magnetizations = mags_by_size[L_train]
    all_energies = energies_by_size[L_train]
    
    print(f"Generated {len(all_configs)} configurations")
    
    # Save to HDF5
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('configurations', data=all_configs)
        f.create_dataset('temperatures', data=all_temps)
        f.create_dataset('magnetizations', data=all_magnetizations)
        f.create_dataset('energies', data=all_energies)
        
        # Create splits
        n_total = len(all_configs)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        n_test = n_total - n_train - n_val
        
        indices = np.random.permutation(n_total)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]
        test_indices = indices[n_train+n_val:]
        
        splits_group = f.create_group('splits')
        splits_group.create_dataset('train_indices', data=train_indices)
        splits_group.create_dataset('val_indices', data=val_indices)
        splits_group.create_dataset('test_indices', data=test_indices)
        
        # Add split sizes as attributes
        splits_group.attrs['train_size'] = n_train
        splits_group.attrs['val_size'] = n_val
        splits_group.attrs['test_size'] = n_test
        splits_group.attrs['split_ratios'] = [0.7, 0.15, 0.15]
        
        # Metadata
        metadata = f.create_group('metadata')
        metadata.attrs['system_sizes'] = system_sizes
        metadata.attrs['temperature_range'] = [3.0, 6.0]
        metadata.attrs['n_configurations'] = n_total
        metadata.attrs['theoretical_tc'] = 4.511
        metadata.attrs['critical_temperature'] = 4.511
        metadata.attrs['creation_time'] = str(time.time())
        L_min = min(system_sizes)
        metadata.attrs['lattice_size'] = [L_min, L_min, L_min]  # 3D lattice size
        metadata.attrs['normalization_method'] = 'none'
        
        # Statistics
        stats_group = f.create_group('statistics')
        config_stats = stats_group.create_group('configurations')
        config_stats.attrs['mean'] = np.mean(all_configs)
        config_stats.attrs['std'] = np.std(all_configs)
        config_stats.attrs['min'] = np.min(all_configs)
        config_stats.attrs['max'] = np.max(all_configs)
    
    print(f"Saved 3D dataset to: {output_path}")
    return output_path


class Simple3DTrainer:
    """Simplified trainer for 3D VAE to avoid circular imports."""
    
    def __init__(self, model, device, learning_rate=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, (list, tuple)):
                data = batch[0]
            else:
                data = batch
            
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            
            reconstruction, mu, logvar = self.model(data)
            loss_dict = self.model.compute_loss(data, reconstruction, mu, logvar)
            
            loss_dict['total_loss'].backward()
            self.optimizer.step()
            
            total_loss += loss_dict['total_loss'].item()
            total_recon_loss += loss_dict['reconstruction_loss'].item()
            total_kl_loss += loss_dict['kl_loss'].item()
        
        n_batches = len(train_loader)
        return {
            'total_loss': total_loss / n_batches,
            'reconstruction_loss': total_recon_loss / n_batches,
            'kl_loss': total_kl_loss / n_batches
        }
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    data = batch[0]
                else:
                    data = batch
                
                data = data.to(self.device)
                
                reconstruction, mu, logvar = self.model(data)
                loss_dict = self.model.compute_loss(data, reconstruction, mu, logvar)
                
                total_loss += loss_dict['total_loss'].item()
                total_recon_loss += loss_dict['reconstruction_loss'].item()
                total_kl_loss += loss_dict['kl_loss'].item()
        
        n_batches = len(val_loader)
        return {
            'total_loss': total_loss / n_batches,
            'reconstruction_loss': total_recon_loss / n_batches,
            'kl_loss': total_kl_loss / n_batches
        }
    
    def train(self, train_loader, val_loader, num_epochs, patience=10):
        """Full training loop."""
        history = {'train_loss': [], 'val_loss': [], 'train_recon': [], 'val_recon': [], 'train_kl': [], 'val_kl': []}
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_metrics['total_loss'])
            
            # Record metrics
            history['train_loss'].append(train_metrics['total_loss'])
            history['val_loss'].append(val_metrics['total_loss'])
            history['train_recon'].append(train_metrics['reconstruction_loss'])
            history['val_recon'].append(val_metrics['reconstruction_loss'])
            history['train_kl'].append(train_metrics['kl_loss'])
            history['val_kl'].append(val_metrics['kl_loss'])
            
            # Check for best model
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_metrics['total_loss']:.6f}, "
                      f"Val Loss: {val_metrics['total_loss']:.6f}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return history
    
    def save_model(self, path):
        """Save the best model."""
        if self.best_model_state is not None:
            torch.save({
                'model_state_dict': self.best_model_state,
                'best_val_loss': self.best_val_loss,
                'model_config': {
                    'input_shape': self.model.input_shape,
                    'latent_dim': self.model.latent_dim,
                    'beta': self.model.beta
                }
            }, path)


def create_3d_config() -> PrometheusConfig:
    """Create configuration for 3D VAE training."""
    config = PrometheusConfig()
    
    # Training parameters - consistent with 2D case
    config.training.num_epochs = 50
    config.training.batch_size = 32
    config.training.learning_rate = 1e-3
    config.training.optimizer = "adam"
    config.training.scheduler = "reducelronplateau"
    config.training.early_stopping_patience = 10
    config.training.checkpoint_interval = 10
    
    # VAE parameters - identical hyperparameters from 2D implementation
    config.vae.latent_dim = 2
    config.vae.beta = 1.0  # β=1.0 maintained from 2D implementation
    config.vae.encoder_channels = [32, 64, 128]
    config.vae.decoder_channels = [128, 64, 32, 1]
    config.vae.kernel_sizes = [3, 3, 3]
    
    # Output directories
    config.models_dir = "models/3d_vae"
    config.results_dir = "results/3d_vae_training"
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Train 3D VAE on Ising configurations')
    parser.add_argument('--data', type=str, help='Path to 3D HDF5 dataset (will generate if not provided)')
    parser.add_argument('--output-dir', type=str, default='models/3d_vae', help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto', help='Device to use')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--generate-data', action='store_true', help='Generate new 3D dataset')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Create configuration
    config = create_3d_config()
    
    # Override config with command line arguments
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.output_dir:
        config.models_dir = args.output_dir
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    config.device = str(device)
    
    print("=" * 60)
    print("3D VAE Training for Prometheus")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Number of epochs: {config.training.num_epochs}")
    print(f"Beta parameter: {config.vae.beta}")
    print(f"Latent dimensions: {config.vae.latent_dim}")
    print()
    
    # Handle dataset
    dataset_path = args.data
    
    if args.generate_data or dataset_path is None:
        # Generate small 3D dataset
        dataset_path = generate_small_3d_dataset("data/ising_3d_small.h5")
        print(f"Generated 3D dataset: {dataset_path}")
    
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Load dataset directly
    print("Loading 3D dataset...")
    
    class Simple3DDataset(torch.utils.data.Dataset):
        def __init__(self, hdf5_path, split='train'):
            self.hdf5_path = hdf5_path
            self.split = split
            
            with h5py.File(hdf5_path, 'r') as f:
                self.indices = f[f'splits/{split}_indices'][:]
                self.configs = f['configurations'][:]
                
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            config_idx = self.indices[idx]
            config = self.configs[config_idx]
            
            # Add channel dimension and convert to tensor
            config = torch.from_numpy(config).float().unsqueeze(0)  # (1, D, H, W)
            return config
    
    # Create datasets
    train_dataset = Simple3DDataset(dataset_path, 'train')
    val_dataset = Simple3DDataset(dataset_path, 'val')
    test_dataset = Simple3DDataset(dataset_path, 'test')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=0)
    
    # Get input shape from first batch
    sample_batch = next(iter(train_loader))
    input_shape = sample_batch.shape[1:]  # Remove batch dimension
    print(f"Dataset loaded successfully")
    print(f"Configuration shape: {sample_batch.shape}")
    print(f"VAE input shape: {input_shape}")
    print(f"Train/Val/Test sizes: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create 3D VAE model
    print("\nInitializing 3D VAE model...")
    model = ConvolutionalVAE3D(
        input_shape=input_shape,
        latent_dim=config.vae.latent_dim,
        encoder_channels=config.vae.encoder_channels,
        decoder_channels=config.vae.decoder_channels,
        kernel_sizes=config.vae.kernel_sizes,
        beta=config.vae.beta
    )
    
    print(f"Model type: {type(model).__name__}")
    print(f"Input shape: {model.input_shape}")
    print(f"Latent dimensions: {model.latent_dim}")
    print(f"Beta parameter: {model.beta}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create output directories
    Path(config.models_dir).mkdir(parents=True, exist_ok=True)
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = Simple3DTrainer(model, device, config.training.learning_rate)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # Training loop
    print(f"\nStarting 3D VAE training for {config.training.num_epochs} epochs...")
    print("-" * 60)
    
    try:
        start_time = time.time()
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.training.num_epochs,
            patience=config.training.early_stopping_patience
        )
        
        training_time = time.time() - start_time
        
        print("\n3D VAE training completed successfully!")
        print(f"Training time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        print(f"Best validation loss: {trainer.best_val_loss:.6f}")
        
        # Final evaluation
        print("\nFinal evaluation...")
        final_val_metrics = trainer.validate_epoch(val_loader)
        final_test_metrics = trainer.validate_epoch(test_loader)
        
        # Save best model
        model_path = Path(config.models_dir) / "best_model.pth"
        trainer.save_model(model_path)
        
        print(f"Final Validation Results:")
        print(f"  Total Loss: {final_val_metrics['total_loss']:.6f}")
        print(f"  Reconstruction Loss: {final_val_metrics['reconstruction_loss']:.6f}")
        print(f"  KL Divergence: {final_val_metrics['kl_loss']:.6f}")
        
        print(f"Final Test Results:")
        print(f"  Total Loss: {final_test_metrics['total_loss']:.6f}")
        print(f"  Reconstruction Loss: {final_test_metrics['reconstruction_loss']:.6f}")
        print(f"  KL Divergence: {final_test_metrics['kl_loss']:.6f}")
        
        # Save training summary
        summary = {
            'dataset_path': dataset_path,
            'model_info': {
                'type': '3D VAE',
                'input_shape': input_shape,
                'latent_dim': model.latent_dim,
                'beta': model.beta,
                'parameters': sum(p.numel() for p in model.parameters())
            },
            'training_config': {
                'epochs': config.training.num_epochs,
                'batch_size': config.training.batch_size,
                'learning_rate': config.training.learning_rate,
                'optimizer': config.training.optimizer
            },
            'results': {
                'training_time': training_time,
                'best_val_loss': trainer.best_val_loss,
                'final_val_metrics': final_val_metrics,
                'final_test_metrics': final_test_metrics
            }
        }
        
        # Save summary
        import json
        summary_path = Path(config.models_dir) / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nTraining summary saved to: {summary_path}")
        print(f"Best model saved to: {Path(config.models_dir) / 'best_model.pth'}")
        print(f"Final checkpoint saved to: {Path(config.models_dir) / 'final_checkpoint.pth'}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        print("Saving current model state...")
        model_path = Path(config.models_dir) / "interrupted_model.pth"
        trainer.save_model(model_path)
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        print("Saving current model state...")
        model_path = Path(config.models_dir) / "error_model.pth"
        trainer.save_model(model_path)
        raise
    
    print("\n" + "=" * 60)
    print("3D VAE Training Complete!")
    print("=" * 60)
    print("Model ready for latent space analysis and order parameter extraction.")


if __name__ == "__main__":
    main()