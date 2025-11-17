#!/usr/bin/env python3
"""
Physics-Informed 2D VAE Training (Task 7.2)

This script implements task 7.2: Retrain 2D VAE with physics-informed loss and better hyperparameters
- Implement physics-informed loss that encourages temperature/magnetization correlations
- Use beta-VAE with warmup schedule for better latent space learning
- Add deeper encoder/decoder architecture with proper regularization
- Validate latent space shows strong correlations with physical properties

Requirements addressed:
- 2.1: Physics-informed loss implementation
- 2.2: Beta-VAE with warmup schedule
- 2.3: Enhanced architecture with regularization
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
from pathlib import Path
import json
from tqdm import tqdm
import time
from typing import Dict, Tuple, List
from scipy.stats import pearsonr

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vae import ConvolutionalVAE


class PhysicsInformedVAE(ConvolutionalVAE):
    """Enhanced VAE with physics-informed loss and deeper architecture."""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (1, 32, 32), latent_dim: int = 2):
        # Enhanced architecture with proper regularization
        encoder_channels = [32, 64, 128]
        decoder_channels = [128, 64, 32, 1]
        kernel_sizes = [3, 3, 3]
        
        super().__init__(
            input_shape=input_shape,
            latent_dim=latent_dim,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            kernel_sizes=kernel_sizes,
            beta=1.0  # Will be controlled by warmup schedule
        )
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def compute_physics_informed_loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        temperatures: torch.Tensor,
        magnetizations: torch.Tensor,
        physics_weight: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed loss that encourages correlations with physical properties.
        
        This loss function combines standard VAE loss with a physics-informed term that
        encourages the latent space to correlate with physical properties (temperature
        and magnetization).
        """
        # Standard VAE loss
        vae_losses = self.compute_loss(x, reconstruction, mu, logvar)
        
        # Physics-informed loss: encourage correlation between latent dimensions and physical properties
        physics_loss = torch.tensor(0.0, device=x.device)
        
        if len(mu) > 1:  # Need at least 2 samples for correlation
            # For each latent dimension, compute correlation with temperature and magnetization
            for i in range(self.latent_dim):
                latent_dim_i = mu[:, i]
                
                # Temperature correlation loss (negative correlation encourages positive correlation)
                temp_corr_loss = -self._compute_correlation_loss(latent_dim_i, temperatures)
                
                # Magnetization correlation loss
                mag_corr_loss = -self._compute_correlation_loss(latent_dim_i, magnetizations)
                
                # Add the stronger correlation (take minimum since we want negative values)
                physics_loss += torch.min(temp_corr_loss, mag_corr_loss)
        
        # Total loss with physics-informed term
        total_loss = vae_losses['total_loss'] + physics_weight * physics_loss
        
        return {
            **vae_losses,
            'physics_loss': physics_loss,
            'total_loss_with_physics': total_loss,
            'physics_weight': physics_weight
        }
    
    def _compute_correlation_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable correlation loss between two tensors.
        
        Returns negative correlation to encourage positive correlation when minimized.
        """
        # Center the variables
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        
        # Compute correlation coefficient
        numerator = (x_centered * y_centered).sum()
        denominator = torch.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())
        
        # Add small epsilon to avoid division by zero
        correlation = numerator / (denominator + 1e-8)
        
        return correlation


class PhysicsInformed2DVAETrainer:
    """Trainer for physics-informed 2D VAE with beta warmup and enhanced architecture."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Enhanced training configuration
        self.config = {
            'epochs': 100,
            'batch_size': 64,
            'learning_rate': 1e-3,
            'beta_warmup_epochs': 20,  # Beta warmup schedule
            'beta_max': 1.0,
            'physics_weight': 0.1,
            'patience': 15,
            'min_delta': 1e-4,
            'weight_decay': 1e-4,  # L2 regularization
            'scheduler_patience': 5,
            'scheduler_factor': 0.5
        }
    
    def load_2d_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load 2D Ising data from file."""
        print(f"Loading 2D data from: {data_path}")
        
        if data_path.endswith('.npz'):
            data = np.load(data_path, allow_pickle=True)
            
            # Handle different data formats
            if 'configurations' in data:
                configurations = data['configurations']
                temperatures = data['temperatures']
                if 'magnetizations' in data:
                    magnetizations = data['magnetizations']
                else:
                    magnetizations = np.mean(configurations, axis=(1, 2))
            elif 'spin_configurations' in data:
                configurations = data['spin_configurations']
                magnetizations = data['magnetizations']
                
                # Generate temperatures based on metadata
                metadata = data['metadata'].item()
                temp_range = metadata['temp_range']
                n_temps = metadata['n_temperatures']
                n_configs_per_temp = metadata['n_configurations'] // n_temps
                
                temperatures = np.repeat(
                    np.linspace(temp_range[0], temp_range[1], n_temps),
                    n_configs_per_temp
                )[:len(configurations)]
            else:
                raise ValueError(f"Unknown data format in {data_path}")
                
        elif data_path.endswith('.h5'):
            with h5py.File(data_path, 'r') as f:
                configurations = f['configurations'][:]
                temperatures = f['temperatures'][:]
                if 'magnetizations' in f:
                    magnetizations = f['magnetizations'][:]
                else:
                    magnetizations = np.mean(configurations, axis=(1, 2))
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        print(f"Loaded {len(configurations)} configurations")
        print(f"Configuration shape: {configurations.shape}")
        print(f"Temperature range: [{temperatures.min():.3f}, {temperatures.max():.3f}]")
        print(f"Magnetization range: [{magnetizations.min():.3f}, {magnetizations.max():.3f}]")
        
        return configurations, temperatures, magnetizations
    
    def prepare_data(
        self,
        configurations: np.ndarray,
        temperatures: np.ndarray,
        magnetizations: np.ndarray,
        train_split: float = 0.8,
        val_split: float = 0.1
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data loaders for training, validation, and testing."""
        n_samples = len(configurations)
        n_train = int(n_samples * train_split)
        n_val = int(n_samples * val_split)
        
        # Random shuffle
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        # Prepare tensors
        def create_loader(idx):
            configs = torch.FloatTensor(configurations[idx])
            temps = torch.FloatTensor(temperatures[idx])
            mags = torch.FloatTensor(magnetizations[idx])
            
            # Add channel dimension if needed
            if len(configs.shape) == 3:
                configs = configs.unsqueeze(1)
            
            dataset = TensorDataset(configs, temps, mags)
            return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)
        
        train_loader = create_loader(train_idx)
        val_loader = create_loader(val_idx)
        test_loader = create_loader(test_idx)
        
        print(f"Data splits - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        return train_loader, val_loader, test_loader
    
    def get_beta_schedule(self, epoch: int) -> float:
        """Get beta value for current epoch using warmup schedule."""
        if epoch < self.config['beta_warmup_epochs']:
            # Linear warmup from 0 to beta_max
            return (epoch / self.config['beta_warmup_epochs']) * self.config['beta_max']
        else:
            return self.config['beta_max']
    
    def train_epoch(
        self,
        model: PhysicsInformedVAE,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        
        # Update beta for this epoch
        beta = self.get_beta_schedule(epoch)
        model.set_beta(beta)
        
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_physics_loss = 0
        n_batches = 0
        
        for batch_configs, batch_temps, batch_mags in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch_configs = batch_configs.to(self.device)
            batch_temps = batch_temps.to(self.device)
            batch_mags = batch_mags.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstruction, mu, logvar = model(batch_configs)
            
            # Compute physics-informed loss
            losses = model.compute_physics_informed_loss(
                batch_configs, reconstruction, mu, logvar,
                batch_temps, batch_mags, self.config['physics_weight']
            )
            
            # Backward pass
            losses['total_loss_with_physics'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            total_loss += losses['total_loss_with_physics'].item()
            total_recon_loss += losses['reconstruction_loss'].item()
            total_kl_loss += losses['kl_loss'].item()
            total_physics_loss += losses['physics_loss'].item()
            n_batches += 1
        
        return {
            'train_loss': total_loss / n_batches,
            'train_recon_loss': total_recon_loss / n_batches,
            'train_kl_loss': total_kl_loss / n_batches,
            'train_physics_loss': total_physics_loss / n_batches,
            'beta': beta
        }
    
    def validate_epoch(
        self,
        model: PhysicsInformedVAE,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Validate for one epoch."""
        model.eval()
        
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_physics_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch_configs, batch_temps, batch_mags in val_loader:
                batch_configs = batch_configs.to(self.device)
                batch_temps = batch_temps.to(self.device)
                batch_mags = batch_mags.to(self.device)
                
                # Forward pass
                reconstruction, mu, logvar = model(batch_configs)
                
                # Compute physics-informed loss
                losses = model.compute_physics_informed_loss(
                    batch_configs, reconstruction, mu, logvar,
                    batch_temps, batch_mags, self.config['physics_weight']
                )
                
                # Accumulate losses
                total_loss += losses['total_loss_with_physics'].item()
                total_recon_loss += losses['reconstruction_loss'].item()
                total_kl_loss += losses['kl_loss'].item()
                total_physics_loss += losses['physics_loss'].item()
                n_batches += 1
        
        return {
            'val_loss': total_loss / n_batches,
            'val_recon_loss': total_recon_loss / n_batches,
            'val_kl_loss': total_kl_loss / n_batches,
            'val_physics_loss': total_physics_loss / n_batches
        }
    
    def evaluate_model_quality(
        self,
        model: PhysicsInformedVAE,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate model quality including correlations with physical properties."""
        model.eval()
        
        all_latents = []
        all_temps = []
        all_mags = []
        all_recons = []
        all_originals = []
        
        with torch.no_grad():
            for batch_configs, batch_temps, batch_mags in test_loader:
                batch_configs = batch_configs.to(self.device)
                
                # Get latent representations and reconstructions
                mu, _ = model.encode(batch_configs)
                reconstruction = model.decode(mu)
                
                all_latents.append(mu.cpu().numpy())
                all_temps.append(batch_temps.numpy())
                all_mags.append(batch_mags.numpy())
                all_recons.append(reconstruction.cpu().numpy())
                all_originals.append(batch_configs.cpu().numpy())
        
        # Concatenate all results
        latents = np.concatenate(all_latents, axis=0)
        temperatures = np.concatenate(all_temps, axis=0)
        magnetizations = np.concatenate(all_mags, axis=0)
        reconstructions = np.concatenate(all_recons, axis=0)
        originals = np.concatenate(all_originals, axis=0)
        
        # Compute correlations for each latent dimension
        correlations = {}
        for i in range(latents.shape[1]):
            temp_corr, _ = pearsonr(latents[:, i], temperatures)
            mag_corr, _ = pearsonr(latents[:, i], magnetizations)
            correlations[f'latent_{i}_temp_corr'] = temp_corr
            correlations[f'latent_{i}_mag_corr'] = mag_corr
        
        # Find best correlations
        temp_correlations = [abs(correlations[f'latent_{i}_temp_corr']) for i in range(latents.shape[1])]
        mag_correlations = [abs(correlations[f'latent_{i}_mag_corr']) for i in range(latents.shape[1])]
        
        max_temp_corr = max(temp_correlations)
        max_mag_corr = max(mag_correlations)
        
        # Compute reconstruction error
        recon_error = np.mean((originals - reconstructions) ** 2)
        
        # Overall quality score
        correlation_score = (max_temp_corr + max_mag_corr) * 50  # Scale to 0-100
        reconstruction_score = max(0, 50 - recon_error * 100)  # Scale reconstruction quality
        quality_score = correlation_score + reconstruction_score
        
        # Quality level
        if quality_score >= 80:
            quality_level = "Excellent"
        elif quality_score >= 60:
            quality_level = "Good"
        elif quality_score >= 40:
            quality_level = "Fair"
        else:
            quality_level = "Poor"
        
        return {
            'correlations': correlations,
            'max_temp_correlation': max_temp_corr,
            'max_mag_correlation': max_mag_corr,
            'mean_reconstruction_error': recon_error,
            'quality_score': quality_score,
            'quality_level': quality_level,
            'latent_statistics': {
                'mean': np.mean(latents, axis=0).tolist(),
                'std': np.std(latents, axis=0).tolist(),
                'range': (np.min(latents, axis=0).tolist(), np.max(latents, axis=0).tolist())
            }
        }
    
    def train_physics_informed_vae(
        self,
        data_path: str,
        output_dir: str,
        latent_dim: int = 2
    ) -> Tuple[PhysicsInformedVAE, Dict[str, List], Dict[str, float]]:
        """Train physics-informed 2D VAE with enhanced architecture and beta warmup."""
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load and prepare data
        configurations, temperatures, magnetizations = self.load_2d_data(data_path)
        train_loader, val_loader, test_loader = self.prepare_data(
            configurations, temperatures, magnetizations
        )
        
        # Determine input shape
        sample_batch = next(iter(train_loader))
        input_shape = sample_batch[0].shape[1:]  # Remove batch dimension
        print(f"Input shape: {input_shape}")
        
        # Create model
        model = PhysicsInformedVAE(input_shape=input_shape, latent_dim=latent_dim)
        model = model.to(self.device)
        
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=self.config['scheduler_patience'],
            factor=self.config['scheduler_factor'],
            verbose=True
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon_loss': [],
            'val_recon_loss': [],
            'train_kl_loss': [],
            'val_kl_loss': [],
            'train_physics_loss': [],
            'val_physics_loss': [],
            'beta_schedule': []
        }
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training for {self.config['epochs']} epochs...")
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            # Train
            train_metrics = self.train_epoch(model, train_loader, optimizer, epoch)
            
            # Validate
            val_metrics = self.validate_epoch(model, val_loader)
            
            # Update history
            for key, value in train_metrics.items():
                if key in history:
                    history[key].append(value)
            
            for key, value in val_metrics.items():
                if key in history:
                    history[key].append(value)
            
            # Learning rate scheduling
            scheduler.step(val_metrics['val_loss'])
            
            # Early stopping check
            if val_metrics['val_loss'] < best_val_loss - self.config['min_delta']:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': best_val_loss,
                    'config': self.config
                }, output_path / 'best_model.pth')
                
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.config['epochs']} - "
                      f"Train Loss: {train_metrics['train_loss']:.4f}, "
                      f"Val Loss: {val_metrics['val_loss']:.4f}, "
                      f"Beta: {train_metrics['beta']:.3f}, "
                      f"Physics Loss: {train_metrics['train_physics_loss']:.4f}")
            
            # Early stopping
            if patience_counter >= self.config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Load best model for evaluation
        checkpoint = torch.load(output_path / 'best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate model quality
        quality_metrics = self.evaluate_model_quality(model, test_loader)
        
        # Save results
        results = {
            'training_history': history,
            'quality_metrics': quality_metrics,
            'config': self.config,
            'training_time': training_time,
            'best_epoch': checkpoint['epoch'],
            'best_val_loss': checkpoint['val_loss']
        }
        
        with open(output_path / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save final model
        torch.save(model.state_dict(), output_path / 'final_model.pth')
        
        print(f"Results saved to: {output_path}")
        
        return model, history, quality_metrics


def main():
    """Main training function."""
    trainer = PhysicsInformed2DVAETrainer()
    
    # Configuration
    data_path = "data/ising_dataset_20250831_145819.npz"
    output_dir = "models/physics_informed_2d_vae"
    
    if not Path(data_path).exists():
        print(f"Data file not found: {data_path}")
        print("Please ensure 2D Ising data is available")
        return
    
    # Train model
    model, history, quality = trainer.train_physics_informed_vae(
        data_path=data_path,
        output_dir=output_dir,
        latent_dim=2
    )
    
    # Print final results
    print("\n" + "="*60)
    print("PHYSICS-INFORMED 2D VAE TRAINING COMPLETE")
    print("="*60)
    print(f"Model quality: {quality['quality_level']} ({quality['quality_score']:.1f}/100)")
    print(f"Best temperature correlation: {quality['max_temp_correlation']:.3f}")
    print(f"Best magnetization correlation: {quality['max_mag_correlation']:.3f}")
    print(f"Reconstruction error: {quality['mean_reconstruction_error']:.4f}")
    
    print("\nLatent correlations:")
    for key, value in quality['correlations'].items():
        print(f"  {key}: {value:.3f}")
    
    print(f"\nModel saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()