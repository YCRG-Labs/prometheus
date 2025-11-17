#!/usr/bin/env python3
"""
Enhanced VAE Training Script

This script trains VAE models with improved hyperparameters and training procedures
to ensure they learn meaningful physics representations for critical exponent extraction.

Uses the enhanced physics-informed loss function with:
- Temperature ordering loss (weight 1.5)
- Critical region enhancement loss (weight 1.0)
- Magnetization correlation loss (weight 2.0)
- Learning rate scheduling with ReduceLROnPlateau
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
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ConvolutionalVAE, ConvolutionalVAE3D
from src.training.enhanced_physics_vae import (
    EnhancedPhysicsVAETrainer,
    EnhancedPhysicsLossWeights,
    create_enhanced_physics_trainer
)
from src.utils.config import PrometheusConfig
from src.utils.logging_utils import setup_logging, get_logger


class EnhancedVAETrainer:
    """Enhanced VAE trainer with physics-aware training procedures using new enhanced physics loss."""
    
    def __init__(self, model_type='2d', device=None):
        self.model_type = model_type
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.logger = get_logger(__name__)
        
        # Enhanced training parameters
        self.enhanced_config = {
            'learning_rate': 1e-3,      # Learning rate as per spec
            'batch_size': 64,           # Larger batch size for better gradients
            'epochs': 200,              # More epochs for better convergence
            'patience': 20,             # Early stopping patience (as per spec)
            'weight_decay': 1e-5,       # L2 regularization
        }
        
        # Enhanced physics loss weights (as per spec requirements)
        self.loss_weights = EnhancedPhysicsLossWeights(
            reconstruction=1.0,
            kl_divergence=1.0,
            magnetization_correlation=2.0,  # Increased from 1.0 to 2.0 (Task 4.3)
            energy_consistency=1.0,
            temperature_ordering=1.5,        # New component (Task 4.1)
            critical_enhancement=1.0         # New component (Task 4.2)
        )
        
    def load_data(self, data_path):
        """Load and preprocess data for training."""
        self.logger.info(f"Loading data from {data_path}")
        
        if data_path.endswith('.npz'):
            # 2D data
            data = np.load(data_path, allow_pickle=True)
            configurations = data['spin_configurations']
            magnetizations = data['magnetizations']
            
            # Extract temperatures
            metadata = data['metadata'].item()
            n_temps = metadata['n_temperatures']
            temp_min, temp_max = metadata['temp_range']
            temperatures = np.linspace(temp_min, temp_max, n_temps)
            n_configs_per_temp = len(configurations) // n_temps
            temp_array = np.repeat(temperatures, n_configs_per_temp)
            
            # Normalize configurations to [0, 1]
            configurations = (configurations + 1) / 2.0
            
        elif data_path.endswith('.h5'):
            # 3D data
            with h5py.File(data_path, 'r') as f:
                configurations = f['configurations'][:]
                magnetizations = f['magnetizations'][:]
                temp_array = f['temperatures'][:]
            
            # Normalize configurations to [0, 1]
            configurations = (configurations + 1) / 2.0
        
        else:
            raise ValueError(f"Unsupported data format: {data_path}")
        
        self.logger.info(f"Loaded {len(configurations)} configurations")
        self.logger.info(f"Configuration shape: {configurations.shape[1:]}")
        self.logger.info(f"Temperature range: [{np.min(temp_array):.3f}, {np.max(temp_array):.3f}]")
        self.logger.info(f"Magnetization range: [{np.min(magnetizations):.4f}, {np.max(magnetizations):.4f}]")
        
        return configurations, magnetizations, temp_array
    
    def create_model(self, input_shape, latent_dim=2):
        """Create VAE model with enhanced architecture."""
        
        if self.model_type == '2d':
            model = ConvolutionalVAE(
                input_shape=(1,) + input_shape,
                latent_dim=latent_dim,
                encoder_channels=[32, 64, 128],  # 3 layers for 32x32 input
                decoder_channels=[128, 64, 32, 1],  # Matching decoder
                kernel_sizes=[3, 3, 3],
                beta=1.0  # Will be scheduled during training
            )
        else:  # 3d
            model = ConvolutionalVAE3D(
                input_shape=(1,) + input_shape,
                latent_dim=latent_dim,
                encoder_channels=[16, 32, 64],  # 3 layers for 32^3 input
                decoder_channels=[64, 32, 16, 1],  # Matching decoder
                kernel_sizes=[3, 3, 3],
                beta=1.0
            )
        
        return model.to(self.device)
    

    
    def train_enhanced_vae(self, data_path, output_dir, latent_dim=2):
        """Train VAE with enhanced physics-informed loss function."""
        
        self.logger.info("Starting enhanced VAE training with physics-informed loss")
        self.logger.info(f"Loss weights: mag_corr={self.loss_weights.magnetization_correlation}, "
                        f"temp_order={self.loss_weights.temperature_ordering}, "
                        f"critical_enh={self.loss_weights.critical_enhancement}")
        
        # Load data
        configurations, magnetizations, temperatures = self.load_data(data_path)
        
        # Create model
        input_shape = configurations.shape[1:]
        model = self.create_model(input_shape, latent_dim)
        
        self.logger.info(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
        
        # Prepare data
        train_size = int(0.8 * len(configurations))
        
        # Convert to tensors - normalize to [-1, 1] for VAE
        config_tensor = torch.FloatTensor(configurations * 2.0 - 1.0)  # [0,1] -> [-1,1]
        if len(config_tensor.shape) == 3:  # 2D data (N, H, W)
            config_tensor = config_tensor.unsqueeze(1)  # Add channel dimension -> (N, 1, H, W)
        elif len(config_tensor.shape) == 4:  # 3D data (N, D, H, W)
            config_tensor = config_tensor.unsqueeze(1)  # Add channel dimension -> (N, 1, D, H, W)
        
        temp_tensor = torch.FloatTensor(temperatures)
        mag_tensor = torch.FloatTensor(magnetizations)
        
        # Split data
        train_configs = config_tensor[:train_size]
        val_configs = config_tensor[train_size:]
        
        train_temps = temp_tensor[:train_size]
        val_temps = temp_tensor[train_size:]
        
        train_mags = mag_tensor[:train_size]
        val_mags = mag_tensor[train_size:]
        
        # Create enhanced physics trainer
        trainer = create_enhanced_physics_trainer(
            model=model,
            device=self.device,
            learning_rate=self.enhanced_config['learning_rate'],
            loss_weights=self.loss_weights
        )
        
        # Training loop with enhanced physics loss
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {
            'train_loss': [], 'val_loss': [], 
            'train_recon': [], 'val_recon': [],
            'train_kl': [], 'val_kl': [],
            'train_mag': [], 'val_mag': [],
            'train_energy': [], 'val_energy': [],
            'train_temp_order': [], 'val_temp_order': [],
            'train_critical': [], 'val_critical': [],
            'learning_rates': []
        }
        
        self.logger.info("Starting training loop with enhanced physics loss")
        
        for epoch in range(self.enhanced_config['epochs']):
            # Training phase
            train_losses = trainer.train_epoch(
                configurations=train_configs,
                temperatures=train_temps,
                magnetizations=train_mags,
                energies=None,  # Can add energy if available
                batch_size=self.enhanced_config['batch_size']
            )
            
            # Validation phase
            val_losses = trainer.validate(
                configurations=val_configs,
                temperatures=val_temps,
                magnetizations=val_mags,
                energies=None
            )
            
            # Record history
            training_history['train_loss'].append(train_losses['total'])
            training_history['val_loss'].append(val_losses['total'])
            training_history['train_recon'].append(train_losses['reconstruction'])
            training_history['val_recon'].append(val_losses['reconstruction'])
            training_history['train_kl'].append(train_losses['kl'])
            training_history['val_kl'].append(val_losses['kl'])
            training_history['train_mag'].append(train_losses['magnetization'])
            training_history['val_mag'].append(val_losses['magnetization'])
            training_history['train_energy'].append(train_losses['energy'])
            training_history['val_energy'].append(val_losses['energy'])
            training_history['train_temp_order'].append(train_losses['temperature_ordering'])
            training_history['val_temp_order'].append(val_losses['temperature_ordering'])
            training_history['train_critical'].append(train_losses['critical_enhancement'])
            training_history['val_critical'].append(val_losses['critical_enhancement'])
            training_history['learning_rates'].append(trainer.optimizer.param_groups[0]['lr'])
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch+1:3d}: Train Loss: {train_losses['total']:.4f}, "
                f"Val Loss: {val_losses['total']:.4f}, "
                f"Mag: {train_losses['magnetization']:.4f}, "
                f"TempOrd: {train_losses['temperature_ordering']:.4f}, "
                f"Critical: {train_losses['critical_enhancement']:.4f}, "
                f"LR: {trainer.optimizer.param_groups[0]['lr']:.2e}"
            )
            
            # Early stopping and model saving
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                patience_counter = 0
                
                # Save best model
                self._save_model(model, output_dir, 'best_model.pth', epoch, training_history)
                self.logger.info(f"  → New best model saved (val_loss: {best_val_loss:.4f})")
                
            else:
                patience_counter += 1
                
                if patience_counter >= self.enhanced_config['patience']:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Save final model and training history
        self._save_model(model, output_dir, 'final_model.pth', epoch, training_history)
        self._save_training_plots(training_history, output_dir)
        
        # Validate model quality
        model_quality = self._validate_model_quality(model, val_configs, val_temps, val_mags)
        
        self.logger.info("Enhanced VAE training completed")
        self.logger.info(f"Best validation loss: {best_val_loss:.4f}")
        self.logger.info(f"Model quality: {model_quality['quality_level']}")
        self.logger.info(f"Magnetization correlation: {model_quality['max_mag_correlation']:.3f}")
        
        return model, training_history, model_quality
    
    def _save_model(self, model, output_dir, filename, epoch, training_history):
        """Save model checkpoint with metadata."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'model_type': self.model_type,
            'enhanced_config': self.enhanced_config,
            'training_history': training_history
        }
        
        torch.save(checkpoint, output_path / filename)
        
        # Also save training summary
        summary = {
            'model_type': self.model_type,
            'epoch': epoch,
            'enhanced_config': self.enhanced_config,
            'final_train_loss': training_history['train_loss'][-1] if training_history['train_loss'] else None,
            'final_val_loss': training_history['val_loss'][-1] if training_history['val_loss'] else None,
            'best_val_loss': min(training_history['val_loss']) if training_history['val_loss'] else None
        }
        
        with open(output_path / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _save_training_plots(self, training_history, output_dir):
        """Save training progress plots with enhanced physics loss components."""
        
        output_path = Path(output_dir)
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        # Total loss
        axes[0, 0].plot(training_history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(training_history['val_loss'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reconstruction loss
        axes[0, 1].plot(training_history['train_recon'], label='Train', linewidth=2)
        axes[0, 1].plot(training_history['val_recon'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Reconstruction Loss', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # KL divergence
        axes[0, 2].plot(training_history['train_kl'], label='Train', linewidth=2)
        axes[0, 2].plot(training_history['val_kl'], label='Validation', linewidth=2)
        axes[0, 2].set_title('KL Divergence', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Magnetization correlation loss (NEW - weight 2.0)
        axes[1, 0].plot(training_history['train_mag'], label='Train', linewidth=2, color='purple')
        axes[1, 0].plot(training_history['val_mag'], label='Validation', linewidth=2, color='orange')
        axes[1, 0].set_title('Magnetization Correlation Loss (w=2.0)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Temperature ordering loss (NEW - weight 1.5)
        axes[1, 1].plot(training_history['train_temp_order'], label='Train', linewidth=2, color='green')
        axes[1, 1].plot(training_history['val_temp_order'], label='Validation', linewidth=2, color='red')
        axes[1, 1].set_title('Temperature Ordering Loss (w=1.5)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Critical enhancement loss (NEW - weight 1.0)
        axes[1, 2].plot(training_history['train_critical'], label='Train', linewidth=2, color='brown')
        axes[1, 2].plot(training_history['val_critical'], label='Validation', linewidth=2, color='pink')
        axes[1, 2].set_title('Critical Enhancement Loss (w=1.0)', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # Energy consistency loss
        axes[2, 0].plot(training_history['train_energy'], label='Train', linewidth=2, color='teal')
        axes[2, 0].plot(training_history['val_energy'], label='Validation', linewidth=2, color='coral')
        axes[2, 0].set_title('Energy Consistency Loss', fontsize=12, fontweight='bold')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Loss')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Learning rate (with ReduceLROnPlateau)
        axes[2, 1].plot(training_history['learning_rates'], linewidth=2, color='darkblue')
        axes[2, 1].set_title('Learning Rate (ReduceLROnPlateau)', fontsize=12, fontweight='bold')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('LR')
        axes[2, 1].set_yscale('log')
        axes[2, 1].grid(True, alpha=0.3)
        
        # Loss components comparison
        axes[2, 2].plot(training_history['train_recon'], label='Recon', linewidth=2, alpha=0.7)
        axes[2, 2].plot(training_history['train_kl'], label='KL', linewidth=2, alpha=0.7)
        axes[2, 2].plot(training_history['train_mag'], label='Mag (2.0x)', linewidth=2, alpha=0.7)
        axes[2, 2].plot(training_history['train_temp_order'], label='TempOrd (1.5x)', linewidth=2, alpha=0.7)
        axes[2, 2].plot(training_history['train_critical'], label='Critical', linewidth=2, alpha=0.7)
        axes[2, 2].set_title('All Loss Components (Train)', fontsize=12, fontweight='bold')
        axes[2, 2].set_xlabel('Epoch')
        axes[2, 2].set_ylabel('Loss')
        axes[2, 2].legend(fontsize=8)
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training plots saved to {output_path / 'training_progress.png'}")
    
    def _validate_model_quality(self, model, val_configs, val_temps, val_mags):
        """Validate the quality of the trained model."""
        
        model.eval()
        
        with torch.no_grad():
            # Sample validation data
            sample_size = min(100, len(val_configs))
            sample_configs = val_configs[:sample_size].to(self.device)
            sample_temps = val_temps[:sample_size]
            sample_mags = val_mags[:sample_size]
            
            # Get latent representations
            mu, logvar = model.encode(sample_configs)
            z = model.reparameterize(mu, logvar)
            
            # Reconstruct
            recon = model.decode(z)
            
            # Calculate reconstruction error
            recon_error = torch.mean((sample_configs - recon)**2).item()
            
            # Check latent space correlations
            z_np = mu.cpu().numpy()
            
            correlations = {}
            for dim in range(z_np.shape[1]):
                temp_corr = np.corrcoef(sample_temps, z_np[:, dim])[0, 1]
                mag_corr = np.corrcoef(sample_mags, z_np[:, dim])[0, 1]
                
                correlations[f'z{dim}_temp_corr'] = temp_corr if not np.isnan(temp_corr) else 0.0
                correlations[f'z{dim}_mag_corr'] = mag_corr if not np.isnan(mag_corr) else 0.0
            
            # Calculate quality score
            quality_score = 0.0
            
            # Reconstruction quality (0-40 points)
            if recon_error < 0.1:
                quality_score += 40
            elif recon_error < 0.5:
                quality_score += 30
            elif recon_error < 1.0:
                quality_score += 20
            elif recon_error < 2.0:
                quality_score += 10
            
            # Correlation quality (0-60 points)
            max_temp_corr = max([abs(correlations[k]) for k in correlations if 'temp_corr' in k])
            max_mag_corr = max([abs(correlations[k]) for k in correlations if 'mag_corr' in k])
            
            if max_temp_corr > 0.7:
                quality_score += 30
            elif max_temp_corr > 0.5:
                quality_score += 20
            elif max_temp_corr > 0.3:
                quality_score += 10
            
            if max_mag_corr > 0.7:
                quality_score += 30
            elif max_mag_corr > 0.5:
                quality_score += 20
            elif max_mag_corr > 0.3:
                quality_score += 10
            
            # Quality level
            if quality_score >= 80:
                quality_level = "EXCELLENT"
            elif quality_score >= 60:
                quality_level = "GOOD"
            elif quality_score >= 40:
                quality_level = "FAIR"
            else:
                quality_level = "POOR"
            
            quality_report = {
                'quality_score': quality_score,
                'quality_level': quality_level,
                'reconstruction_error': recon_error,
                'correlations': correlations,
                'max_temp_correlation': max_temp_corr,
                'max_mag_correlation': max_mag_corr
            }
            
            return quality_report


def main():
    """Main training function."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced VAE Training')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--model-type', type=str, choices=['2d', '3d'], required=True,
                       help='Model type (2d or 3d)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for trained model')
    parser.add_argument('--latent-dim', type=int, default=2,
                       help='Latent space dimensionality')
    
    args = parser.parse_args()
    
    # Setup logging
    config = PrometheusConfig()
    setup_logging(config.logging)
    
    logger = get_logger(__name__)
    logger.info("Starting enhanced VAE training")
    
    # Create trainer
    trainer = EnhancedVAETrainer(model_type=args.model_type)
    
    # Train model
    model, history, quality = trainer.train_enhanced_vae(
        data_path=args.data_path,
        output_dir=args.output_dir,
        latent_dim=args.latent_dim
    )
    
    logger.info("Enhanced VAE training completed")
    
    # Print summary
    print("\n" + "="*60)
    print("ENHANCED VAE TRAINING SUMMARY")
    print("="*60)
    print(f"Model type: {args.model_type}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Model quality: {quality['quality_level']} ({quality['quality_score']:.1f}/100)")
    print(f"Reconstruction error: {quality['reconstruction_error']:.4f}")
    print(f"Max temperature correlation: {quality['max_temp_correlation']:.3f}")
    print(f"Max magnetization correlation: {quality['max_mag_correlation']:.3f}")
    print("="*60)


if __name__ == "__main__":
    main()