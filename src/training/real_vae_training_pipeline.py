"""
Real VAE Training Pipeline on Physics Data

This module implements task 13.2: Real VAE training pipeline that actually trains
VAE models on Monte Carlo configurations with proper encoder/decoder architecture
and physics-informed regularization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import time
import h5py

# Simple logger function
def get_logger(name):
    return logging.getLogger(name)


@dataclass
class RealVAETrainingConfig:
    """Configuration for real VAE training."""
    # Model architecture
    latent_dim: int = 2
    encoder_channels: List[int] = None
    decoder_channels: List[int] = None
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 100
    beta: float = 1.0  # KL divergence weight
    
    # Physics-informed loss parameters
    physics_weight: float = 0.1
    temperature_correlation_weight: float = 0.05
    magnetization_correlation_weight: float = 0.05
    
    # Training options
    use_physics_informed_loss: bool = True
    use_beta_warmup: bool = True
    warmup_epochs: int = 20
    
    # Data parameters
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Device and logging
    device: str = 'auto'
    save_checkpoints: bool = True
    checkpoint_dir: str = 'models/real_vae_training'
    
    def __post_init__(self):
        if self.encoder_channels is None:
            self.encoder_channels = [32, 64, 128]
        if self.decoder_channels is None:
            self.decoder_channels = [128, 64, 32, 1]


@dataclass
class RealVAETrainingResults:
    """Results from real VAE training."""
    model_state_dict: Dict[str, Any]
    training_losses: List[float]
    validation_losses: List[float]
    reconstruction_losses: List[float]
    kl_losses: List[float]
    physics_losses: List[float]
    
    # Final metrics
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    best_epoch: int
    
    # Model quality metrics
    latent_magnetization_correlation: float
    latent_temperature_correlation: float
    reconstruction_quality: float
    
    # Training info
    total_epochs: int
    training_time: float
    config: RealVAETrainingConfig


class PhysicsDataset(Dataset):
    """Dataset for physics configurations with metadata."""
    
    def __init__(self, 
                 configurations: np.ndarray,
                 temperatures: np.ndarray,
                 magnetizations: np.ndarray,
                 energies: Optional[np.ndarray] = None):
        """
        Initialize physics dataset.
        
        Args:
            configurations: Array of spin configurations
            temperatures: Temperature for each configuration
            magnetizations: Magnetization for each configuration
            energies: Energy for each configuration (optional)
        """
        self.configurations = torch.FloatTensor(configurations)
        self.temperatures = torch.FloatTensor(temperatures)
        self.magnetizations = torch.FloatTensor(magnetizations)
        
        if energies is not None:
            self.energies = torch.FloatTensor(energies)
        else:
            self.energies = None
        
        # Ensure configurations have channel dimension
        if len(self.configurations.shape) == 3:  # 2D: (N, H, W)
            self.configurations = self.configurations.unsqueeze(1)  # (N, 1, H, W)
        elif len(self.configurations.shape) == 4:  # 3D: (N, D, H, W)
            self.configurations = self.configurations.unsqueeze(1)  # (N, 1, D, H, W)
    
    def __len__(self):
        return len(self.configurations)
    
    def __getitem__(self, idx):
        item = {
            'configuration': self.configurations[idx],
            'temperature': self.temperatures[idx],
            'magnetization': self.magnetizations[idx]
        }
        
        if self.energies is not None:
            item['energy'] = self.energies[idx]
        
        return item


class RealVAEEncoder(nn.Module):
    """Real VAE encoder for 2D/3D physics configurations."""
    
    def __init__(self, 
                 input_shape: Tuple[int, ...],
                 latent_dim: int = 2,
                 channels: List[int] = None):
        super().__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        if channels is None:
            channels = [32, 64, 128]
        
        # Determine if 2D or 3D
        self.is_3d = len(input_shape) == 4  # (C, D, H, W)
        
        # Build convolutional layers
        conv_layers = []
        in_channels = input_shape[0]
        
        for out_channels in channels:
            if self.is_3d:
                conv_layers.extend([
                    nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU()
                ])
            else:
                conv_layers.extend([
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_output = self.conv_layers(dummy_input)
            self.flattened_size = conv_output.numel()
        
        # Fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
    
    def forward(self, x):
        # Convolutional encoding
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)  # Flatten
        
        # Latent parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class RealVAEDecoder(nn.Module):
    """Real VAE decoder for 2D/3D physics configurations."""
    
    def __init__(self,
                 output_shape: Tuple[int, ...],
                 latent_dim: int = 2,
                 channels: List[int] = None):
        super().__init__()
        
        self.output_shape = output_shape
        self.latent_dim = latent_dim
        
        if channels is None:
            channels = [128, 64, 32]
        
        # Determine if 2D or 3D
        self.is_3d = len(output_shape) == 4  # (C, D, H, W)
        
        # Calculate initial spatial dimensions after upsampling
        if self.is_3d:
            # Start from small 3D volume
            self.init_depth = max(2, output_shape[1] // (2 ** len(channels)))
            self.init_height = max(2, output_shape[2] // (2 ** len(channels)))
            self.init_width = max(2, output_shape[3] // (2 ** len(channels)))
            self.init_size = channels[0] * self.init_depth * self.init_height * self.init_width
        else:
            # Start from small 2D feature map
            self.init_height = max(2, output_shape[1] // (2 ** len(channels)))
            self.init_width = max(2, output_shape[2] // (2 ** len(channels)))
            self.init_size = channels[0] * self.init_height * self.init_width
        
        # Fully connected layer from latent to initial feature map
        self.fc = nn.Linear(latent_dim, self.init_size)
        
        # Build deconvolutional layers
        deconv_layers = []
        in_channels = channels[0]
        
        for i, out_channels in enumerate(channels[1:] + [output_shape[0]]):
            if self.is_3d:
                deconv_layers.extend([
                    nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm3d(out_channels) if i < len(channels) - 1 else nn.Identity(),
                    nn.ReLU() if i < len(channels) - 1 else nn.Tanh()
                ])
            else:
                deconv_layers.extend([
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(out_channels) if i < len(channels) - 1 else nn.Identity(),
                    nn.ReLU() if i < len(channels) - 1 else nn.Tanh()
                ])
            in_channels = out_channels
        
        self.deconv_layers = nn.Sequential(*deconv_layers)
    
    def forward(self, z):
        # Fully connected expansion
        h = self.fc(z)
        
        # Reshape to initial feature map
        if self.is_3d:
            h = h.view(h.size(0), -1, self.init_depth, self.init_height, self.init_width)
        else:
            h = h.view(h.size(0), -1, self.init_height, self.init_width)
        
        # Deconvolutional decoding
        output = self.deconv_layers(h)
        
        # Ensure correct output shape
        if self.is_3d:
            output = nn.functional.interpolate(output, size=self.output_shape[1:], mode='trilinear', align_corners=False)
        else:
            output = nn.functional.interpolate(output, size=self.output_shape[1:], mode='bilinear', align_corners=False)
        
        return output


class RealVAE(nn.Module):
    """Real VAE model for physics configurations."""
    
    def __init__(self,
                 input_shape: Tuple[int, ...],
                 latent_dim: int = 2,
                 encoder_channels: List[int] = None,
                 decoder_channels: List[int] = None,
                 beta: float = 1.0):
        super().__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Create encoder and decoder
        self.encoder = RealVAEEncoder(input_shape, latent_dim, encoder_channels)
        self.decoder = RealVAEDecoder(input_shape, latent_dim, decoder_channels)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon_x = self.decoder(z)
        
        return recon_x, mu, logvar, z
    
    def encode(self, x):
        """Encode input to latent space."""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class RealVAETrainingPipeline:
    """Real VAE training pipeline for physics data."""
    
    def __init__(self, config: RealVAETrainingConfig):
        """Initialize real VAE training pipeline."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Set device
        if config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)
        
        self.logger.info(f"Real VAE training pipeline initialized on {self.device}")
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # Results tracking
        self.training_losses = []
        self.validation_losses = []
        self.reconstruction_losses = []
        self.kl_losses = []
        self.physics_losses = []
    
    def prepare_data(self,
                    configurations: np.ndarray,
                    temperatures: np.ndarray,
                    magnetizations: np.ndarray,
                    energies: Optional[np.ndarray] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare physics data for training.
        
        Args:
            configurations: Spin configurations
            temperatures: Temperature values
            magnetizations: Magnetization values
            energies: Energy values (optional)
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        self.logger.info("Preparing physics data for training")
        
        # Create dataset
        dataset = PhysicsDataset(configurations, temperatures, magnetizations, energies)
        
        # Calculate split sizes
        total_size = len(dataset)
        test_size = int(self.config.test_split * total_size)
        val_size = int(self.config.validation_split * total_size)
        train_size = total_size - val_size - test_size
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for compatibility
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        self.logger.info(f"Data split: {train_size} train, {val_size} val, {test_size} test")
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def create_model(self, input_shape: Tuple[int, ...]) -> RealVAE:
        """Create real VAE model."""
        self.logger.info(f"Creating real VAE model for input shape {input_shape}")
        
        self.model = RealVAE(
            input_shape=input_shape,
            latent_dim=self.config.latent_dim,
            encoder_channels=self.config.encoder_channels,
            decoder_channels=self.config.decoder_channels,
            beta=self.config.beta
        )
        
        self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        num_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Created real VAE with {num_params} parameters")
        
        return self.model
    
    def compute_loss(self, 
                    batch: Dict[str, torch.Tensor],
                    recon_x: torch.Tensor,
                    mu: torch.Tensor,
                    logvar: torch.Tensor,
                    z: torch.Tensor,
                    epoch: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss including reconstruction, KL, and physics-informed terms.
        
        Args:
            batch: Input batch
            recon_x: Reconstructed configurations
            mu: Latent means
            logvar: Latent log variances
            z: Latent samples
            epoch: Current epoch (for beta warmup)
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        x = batch['configuration']
        temperatures = batch['temperature']
        magnetizations = batch['magnetization']
        
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # Beta warmup
        if self.config.use_beta_warmup and epoch < self.config.warmup_epochs:
            beta = self.config.beta * (epoch / self.config.warmup_epochs)
        else:
            beta = self.config.beta
        
        # Physics-informed loss
        physics_loss = torch.tensor(0.0, device=self.device)
        
        if self.config.use_physics_informed_loss:
            # Temperature correlation loss
            if self.config.temperature_correlation_weight > 0:
                # Encourage latent dimensions to correlate with temperature
                temp_expanded = temperatures.unsqueeze(1).expand(-1, self.config.latent_dim)
                temp_corr_loss = 1.0 - torch.abs(self._compute_correlation(z, temp_expanded))
                physics_loss += self.config.temperature_correlation_weight * temp_corr_loss.mean()
            
            # Magnetization correlation loss
            if self.config.magnetization_correlation_weight > 0:
                # Encourage at least one latent dimension to correlate with magnetization
                mag_expanded = magnetizations.unsqueeze(1).expand(-1, self.config.latent_dim)
                mag_corr = torch.abs(self._compute_correlation(z, mag_expanded))
                mag_corr_loss = 1.0 - torch.max(mag_corr)  # Maximize best correlation
                physics_loss += self.config.magnetization_correlation_weight * mag_corr_loss
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss + self.config.physics_weight * physics_loss
        
        loss_components = {
            'reconstruction': recon_loss.item(),
            'kl': kl_loss.item(),
            'physics': physics_loss.item(),
            'beta': beta,
            'total': total_loss.item()
        }
        
        return total_loss, loss_components
    
    def _compute_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute correlation between tensors."""
        # Center the data
        x_centered = x - x.mean(dim=0, keepdim=True)
        y_centered = y - y.mean(dim=0, keepdim=True)
        
        # Compute correlation
        numerator = (x_centered * y_centered).sum(dim=0)
        denominator = torch.sqrt((x_centered ** 2).sum(dim=0) * (y_centered ** 2).sum(dim=0))
        
        # Avoid division by zero
        correlation = numerator / (denominator + 1e-8)
        
        return correlation
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'kl': 0.0,
            'physics': 0.0
        }
        
        num_batches = 0
        
        for batch in self.train_loader:
            # Move batch to device
            for key in batch:
                batch[key] = batch[key].to(self.device)
            
            # Forward pass
            recon_x, mu, logvar, z = self.model(batch['configuration'])
            
            # Compute loss
            loss, loss_components = self.compute_loss(batch, recon_x, mu, logvar, z, epoch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                if key in loss_components:
                    epoch_losses[key] += loss_components[key]
                elif key == 'total':
                    epoch_losses[key] += loss_components['total']
            
            num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        epoch_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'kl': 0.0,
            'physics': 0.0
        }
        
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                for key in batch:
                    batch[key] = batch[key].to(self.device)
                
                # Forward pass
                recon_x, mu, logvar, z = self.model(batch['configuration'])
                
                # Compute loss
                loss, loss_components = self.compute_loss(batch, recon_x, mu, logvar, z, epoch)
                
                # Accumulate losses
                for key in epoch_losses:
                    if key in loss_components:
                        epoch_losses[key] += loss_components[key]
                    elif key == 'total':
                        epoch_losses[key] += loss_components['total']
                
                num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def train(self, 
             configurations: np.ndarray,
             temperatures: np.ndarray,
             magnetizations: np.ndarray,
             energies: Optional[np.ndarray] = None) -> RealVAETrainingResults:
        """
        Train the real VAE on physics data.
        
        Args:
            configurations: Spin configurations
            temperatures: Temperature values
            magnetizations: Magnetization values
            energies: Energy values (optional)
            
        Returns:
            RealVAETrainingResults with training results
        """
        self.logger.info("Starting real VAE training on physics data")
        start_time = time.time()
        
        # Prepare data
        self.prepare_data(configurations, temperatures, magnetizations, energies)
        
        # Create model
        input_shape = (1,) + configurations.shape[1:]  # Add channel dimension
        self.create_model(input_shape)
        
        # Create checkpoint directory
        if self.config.save_checkpoints:
            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch(epoch)
            
            # Validate epoch
            val_losses = self.validate_epoch(epoch)
            
            # Track losses
            self.training_losses.append(train_losses['total'])
            self.validation_losses.append(val_losses['total'])
            self.reconstruction_losses.append(train_losses['reconstruction'])
            self.kl_losses.append(train_losses['kl'])
            self.physics_losses.append(train_losses['physics'])
            
            # Check for best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.best_model_state = self.model.state_dict().copy()
                best_epoch = epoch
            
            # Log progress
            if epoch % 10 == 0 or epoch == self.config.num_epochs - 1:
                self.logger.info(
                    f"Epoch {epoch:3d}: "
                    f"Train Loss: {train_losses['total']:.4f}, "
                    f"Val Loss: {val_losses['total']:.4f}, "
                    f"Recon: {train_losses['reconstruction']:.4f}, "
                    f"KL: {train_losses['kl']:.4f}, "
                    f"Physics: {train_losses['physics']:.4f}"
                )
            
            # Save checkpoint
            if self.config.save_checkpoints and (epoch % 20 == 0 or epoch == self.config.num_epochs - 1):
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_losses['total'],
                    'val_loss': val_losses['total']
                }, checkpoint_path)
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Compute final metrics
        final_metrics = self._compute_final_metrics(configurations, temperatures, magnetizations)
        
        training_time = time.time() - start_time
        
        # Create results
        results = RealVAETrainingResults(
            model_state_dict=self.model.state_dict(),
            training_losses=self.training_losses,
            validation_losses=self.validation_losses,
            reconstruction_losses=self.reconstruction_losses,
            kl_losses=self.kl_losses,
            physics_losses=self.physics_losses,
            final_train_loss=self.training_losses[-1],
            final_val_loss=self.validation_losses[-1],
            best_val_loss=self.best_val_loss,
            best_epoch=best_epoch,
            latent_magnetization_correlation=final_metrics['latent_magnetization_correlation'],
            latent_temperature_correlation=final_metrics['latent_temperature_correlation'],
            reconstruction_quality=final_metrics['reconstruction_quality'],
            total_epochs=self.config.num_epochs,
            training_time=training_time,
            config=self.config
        )
        
        self.logger.info(f"Real VAE training completed in {training_time:.2f} seconds")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f} at epoch {best_epoch}")
        
        return results
    
    def _compute_final_metrics(self, 
                              configurations: np.ndarray,
                              temperatures: np.ndarray,
                              magnetizations: np.ndarray) -> Dict[str, float]:
        """Compute final quality metrics."""
        self.model.eval()
        
        # Sample subset for evaluation
        n_samples = min(1000, len(configurations))
        indices = np.random.choice(len(configurations), n_samples, replace=False)
        
        sample_configs = torch.FloatTensor(configurations[indices])
        sample_temps = temperatures[indices]
        sample_mags = magnetizations[indices]
        
        # Add channel dimension
        if len(sample_configs.shape) == 3:
            sample_configs = sample_configs.unsqueeze(1)
        elif len(sample_configs.shape) == 4:
            sample_configs = sample_configs.unsqueeze(1)
        
        sample_configs = sample_configs.to(self.device)
        
        with torch.no_grad():
            # Get latent representations
            z, mu, logvar = self.model.encode(sample_configs)
            
            # Get reconstructions
            recon_x, _, _, _ = self.model(sample_configs)
            
            # Compute reconstruction quality
            recon_error = nn.functional.mse_loss(recon_x, sample_configs).item()
            reconstruction_quality = 1.0 / (1.0 + recon_error)  # Higher is better
            
            # Compute correlations
            z_np = z.cpu().numpy()
            
            # Find best latent dimension for magnetization correlation
            mag_correlations = []
            temp_correlations = []
            
            for dim in range(z_np.shape[1]):
                # Magnetization correlation
                mag_corr = np.corrcoef(z_np[:, dim], np.abs(sample_mags))[0, 1]
                if not np.isnan(mag_corr):
                    mag_correlations.append(abs(mag_corr))
                else:
                    mag_correlations.append(0.0)
                
                # Temperature correlation
                temp_corr = np.corrcoef(z_np[:, dim], sample_temps)[0, 1]
                if not np.isnan(temp_corr):
                    temp_correlations.append(abs(temp_corr))
                else:
                    temp_correlations.append(0.0)
            
            best_mag_correlation = max(mag_correlations) if mag_correlations else 0.0
            best_temp_correlation = max(temp_correlations) if temp_correlations else 0.0
        
        return {
            'latent_magnetization_correlation': best_mag_correlation,
            'latent_temperature_correlation': best_temp_correlation,
            'reconstruction_quality': reconstruction_quality
        }


def create_real_vae_training_pipeline(config: Optional[RealVAETrainingConfig] = None) -> RealVAETrainingPipeline:
    """
    Factory function to create real VAE training pipeline.
    
    Args:
        config: Training configuration (uses default if None)
        
    Returns:
        Configured RealVAETrainingPipeline instance
    """
    if config is None:
        config = RealVAETrainingConfig()
    
    return RealVAETrainingPipeline(config)


def load_physics_data_from_file(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load physics data from HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
        
    Returns:
        Tuple of (configurations, temperatures, magnetizations, energies)
    """
    with h5py.File(filepath, 'r') as f:
        configurations = f['configurations'][:]
        temperatures = f['temperatures'][:]
        magnetizations = f['magnetizations'][:]
        
        # Energies are optional
        if 'energies' in f:
            energies = f['energies'][:]
        else:
            energies = None
    
    return configurations, temperatures, magnetizations, energies