#!/usr/bin/env python3
"""
Physics-Informed 3D VAE Training (Task 7.3)

This script implements task 7.3: Retrain 3D VAE with improved data and architecture
- Use high-quality 3D data from task 7.1 for training
- Implement 3D-specific architecture optimizations for memory efficiency
- Add physics-informed loss to encourage meaningful latent representations
- Validate latent coordinates are not constant (current issue: z1: 2.25-2.27)

Requirements addressed:
- 2.1: 3D VAE architecture with physics-informed loss
- 2.2: Memory-efficient 3D processing
- 2.3: Enhanced latent space learning
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
import matplotlib.pyplot as plt

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vae_3d import ConvolutionalVAE3D


class PhysicsInformed3DVAE(ConvolutionalVAE3D):
    """Enhanced 3D VAE with physics-informed loss and memory-efficient architecture."""
    
    def __init__(self, input_shape: Tuple[int, int, int, int] = (1, 32, 32, 32), latent_dim: int = 2):
        # Memory-efficient 3D architecture
        encoder_channels = [16, 32, 64]  # Reduced channels for memory efficiency
        decoder_channels = [64, 32, 16, 1]
        kernel_sizes = [3, 3, 3]
        
        super().__init__(
            input_shape=input_shape,
            latent_dim=latent_dim,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            kernel_sizes=kernel_sizes,
            beta=1.0  # Will be controlled by warmup schedule
        )
        
        # Add batch normalization for better training stability
        self.use_batch_norm = True
        
    def compute_physics_informed_loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        temperatures: torch.Tensor,
        magnetizations: torch.Tensor,
        physics_weight: float = 0.1,
        diversity_weight: float = 0.05
    ) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed loss for 3D VAE with additional diversity regularization.
        
        This loss function addresses the constant latent coordinate issue by:
        1. Encouraging correlation with physical properties
        2. Adding diversity regularization to prevent collapsed latent dimensions
        3. Using temperature-dependent physics weighting
        """
        # Standard VAE loss
        vae_losses = self.compute_loss(x, reconstruction, mu, logvar)
        
        # Physics-informed loss: encourage correlation between latent dimensions and physical properties
        physics_loss = torch.tensor(0.0, device=x.device)
        diversity_loss = torch.tensor(0.0, device=x.device)
        
        if len(mu) > 1:  # Need at least 2 samples for correlation
            # Physics correlation loss
            for i in range(self.latent_dim):
                latent_dim_i = mu[:, i]
                
                # Temperature correlation loss
                temp_corr_loss = -self._compute_correlation_loss(latent_dim_i, temperatures)
                
                # Magnetization correlation loss
                mag_corr_loss = -self._compute_correlation_loss(latent_dim_i, magnetizations)
                
                # Take the stronger correlation (minimum since we want negative values)
                physics_loss += torch.min(temp_corr_loss, mag_corr_loss)
            
            # Diversity regularization to prevent constant latent coordinates
            # Encourage latent dimensions to have non-zero variance
            for i in range(self.latent_dim):
                latent_var = torch.var(mu[:, i])
                # Penalize low variance (add negative log variance with clipping)
                latent_var_clamped = torch.clamp(latent_var, 1e-6, 10.0)
                diversity_loss += -torch.log(latent_var_clamped)
            
            # Encourage different latent dimensions to be uncorrelated
            if self.latent_dim > 1:
                for i in range(self.latent_dim):
                    for j in range(i + 1, self.latent_dim):
                        latent_corr = self._compute_correlation_loss(mu[:, i], mu[:, j])
                        # Penalize high correlation between latent dimensions
                        diversity_loss += torch.abs(latent_corr)
        
        # Adaptive physics weighting based on temperature spread (with safeguards)
        if len(temperatures) > 1:
            temp_spread = torch.std(temperatures)
            adaptive_physics_weight = physics_weight * (1.0 + torch.clamp(temp_spread, 0.0, 2.0))
        else:
            adaptive_physics_weight = physics_weight
        
        # Total loss with physics-informed and diversity terms
        total_loss = (vae_losses['total_loss'] + 
                     adaptive_physics_weight * physics_loss + 
                     diversity_weight * diversity_loss)
        
        return {
            **vae_losses,
            'physics_loss': physics_loss,
            'diversity_loss': diversity_loss,
            'total_loss_with_physics': total_loss,
            'physics_weight': adaptive_physics_weight,
            'diversity_weight': diversity_weight
        }
    
    def _compute_correlation_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable correlation loss between two tensors.
        
        Returns negative correlation to encourage positive correlation when minimized.
        """
        # Check for valid inputs
        if len(x) < 2 or len(y) < 2:
            return torch.tensor(0.0, device=x.device)
        
        # Center the variables
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        
        # Compute variances
        x_var = (x_centered ** 2).sum()
        y_var = (y_centered ** 2).sum()
        
        # Check for zero variance (constant values)
        if x_var < 1e-8 or y_var < 1e-8:
            return torch.tensor(0.0, device=x.device)
        
        # Compute correlation coefficient
        numerator = (x_centered * y_centered).sum()
        denominator = torch.sqrt(x_var * y_var)
        
        # Clamp correlation to avoid numerical issues
        correlation = torch.clamp(numerator / (denominator + 1e-8), -1.0, 1.0)
        
        return correlation


class PhysicsInformed3DVAETrainer:
    """Trainer for physics-informed 3D VAE with memory optimization and enhanced architecture."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Memory and GPU information
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Enhanced training configuration optimized for 3D
        self.config = {
            'epochs': 100,              # Reduced epochs for initial testing
            'batch_size': 16,           # Increased batch size for better gradients
            'learning_rate': 1e-4,      # Even lower learning rate for stability
            'beta_warmup_epochs': 20,   # Shorter warmup for testing
            'beta_max': 1.0,           # Lower beta to start
            'physics_weight': 0.05,     # Much lower physics weight initially
            'diversity_weight': 0.01,   # Lower diversity weight
            'patience': 15,             # Reduced patience for testing
            'min_delta': 1e-4,         # Larger delta for stability
            'weight_decay': 1e-5,      # Lower weight decay
            'scheduler_patience': 5,    # Scheduler patience
            'scheduler_factor': 0.8,    # Scheduler factor
            'gradient_clip': 0.5,       # Lower gradient clipping
            'accumulation_steps': 2     # Reduced accumulation steps
        }
    
    def load_3d_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load high-quality 3D Ising data from HDF5 file."""
        print(f"Loading 3D data from: {data_path}")
        
        with h5py.File(data_path, 'r') as f:
            configurations = f['configurations'][:]
            temperatures = f['temperatures'][:]
            magnetizations = f['magnetizations'][:]
            
            # Load metadata for validation
            if 'metadata' in f:
                metadata = dict(f['metadata'].attrs)
                print(f"Data metadata: {metadata}")
            
            # Load quality report if available
            if 'quality_report' in f:
                quality_attrs = dict(f['quality_report'].attrs)
                print(f"Data quality: {quality_attrs.get('quality_level', 'Unknown')}")
        
        print(f"Loaded {len(configurations)} configurations")
        print(f"Configuration shape: {configurations.shape}")
        print(f"Temperature range: [{temperatures.min():.3f}, {temperatures.max():.3f}]")
        print(f"Magnetization range: [{magnetizations.min():.4f}, {magnetizations.max():.4f}]")
        
        # Validate data quality
        self._validate_data_quality(configurations, temperatures, magnetizations)
        
        return configurations, temperatures, magnetizations
    
    def _validate_data_quality(self, configurations: np.ndarray, temperatures: np.ndarray, magnetizations: np.ndarray):
        """Validate the quality of loaded 3D data."""
        issues = []
        
        # Check magnetization range
        mag_range = np.max(np.abs(magnetizations))
        if mag_range < 0.1:
            issues.append(f"Low magnetization range: {mag_range:.4f}")
        
        # Check temperature coverage
        temp_range = temperatures.max() - temperatures.min()
        if temp_range < 1.0:
            issues.append(f"Narrow temperature range: {temp_range:.3f}")
        
        # Check for phase transition signature
        tc_theoretical = 4.511
        near_tc_mask = np.abs(temperatures - tc_theoretical) < 0.2
        if not np.any(near_tc_mask):
            issues.append("No data near critical temperature")
        
        # Check configuration diversity
        unique_configs = len(np.unique(configurations.reshape(len(configurations), -1), axis=0))
        if unique_configs < len(configurations) * 0.9:
            issues.append(f"Low configuration diversity: {unique_configs}/{len(configurations)}")
        
        if issues:
            print("Data quality issues detected:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Data quality validation passed")
    
    def prepare_data(
        self,
        configurations: np.ndarray,
        temperatures: np.ndarray,
        magnetizations: np.ndarray,
        train_split: float = 0.7,
        val_split: float = 0.15
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare memory-efficient data loaders for 3D training."""
        n_samples = len(configurations)
        n_train = int(n_samples * train_split)
        n_val = int(n_samples * val_split)
        
        # Random shuffle with fixed seed for reproducibility
        np.random.seed(42)
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        # Prepare tensors with memory optimization
        def create_loader(idx, shuffle=True):
            configs = configurations[idx].astype(np.float32)  # Use float32 for memory efficiency
            temps = temperatures[idx].astype(np.float32)
            mags = magnetizations[idx].astype(np.float32)
            
            # Normalize configurations to [0, 1] range
            configs = (configs + 1.0) / 2.0
            
            # Add channel dimension
            configs = configs[:, np.newaxis, :, :, :]  # Add channel dimension
            
            # Convert to tensors
            config_tensor = torch.FloatTensor(configs)
            temp_tensor = torch.FloatTensor(temps)
            mag_tensor = torch.FloatTensor(mags)
            
            dataset = TensorDataset(config_tensor, temp_tensor, mag_tensor)
            return DataLoader(
                dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=shuffle,
                num_workers=0,  # Avoid multiprocessing issues with 3D data
                pin_memory=torch.cuda.is_available()
            )
        
        train_loader = create_loader(train_idx, shuffle=True)
        val_loader = create_loader(val_idx, shuffle=False)
        test_loader = create_loader(test_idx, shuffle=False)
        
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
        model: PhysicsInformed3DVAE,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch with gradient accumulation."""
        model.train()
        
        # Update beta for this epoch
        beta = self.get_beta_schedule(epoch)
        model.set_beta(beta)
        
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_physics_loss = 0
        total_diversity_loss = 0
        n_batches = 0
        
        # Gradient accumulation
        optimizer.zero_grad()
        
        for batch_idx, (batch_configs, batch_temps, batch_mags) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            batch_configs = batch_configs.to(self.device)
            batch_temps = batch_temps.to(self.device)
            batch_mags = batch_mags.to(self.device)
            
            # Forward pass
            reconstruction, mu, logvar = model(batch_configs)
            
            # Compute physics-informed loss
            losses = model.compute_physics_informed_loss(
                batch_configs, reconstruction, mu, logvar,
                batch_temps, batch_mags, 
                self.config['physics_weight'],
                self.config['diversity_weight']
            )
            
            # Scale loss for gradient accumulation
            loss = losses['total_loss_with_physics'] / self.config['accumulation_steps']
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.config['accumulation_steps'] == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.config['gradient_clip'])
                
                optimizer.step()
                optimizer.zero_grad()
            
            # Accumulate losses (unscaled for reporting)
            total_loss += losses['total_loss_with_physics'].item()
            total_recon_loss += losses['reconstruction_loss'].item()
            total_kl_loss += losses['kl_loss'].item()
            total_physics_loss += losses['physics_loss'].item()
            total_diversity_loss += losses['diversity_loss'].item()
            n_batches += 1
        
        # Handle remaining gradients
        if n_batches % self.config['accumulation_steps'] != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.config['gradient_clip'])
            optimizer.step()
            optimizer.zero_grad()
        
        return {
            'train_loss': total_loss / n_batches,
            'train_recon_loss': total_recon_loss / n_batches,
            'train_kl_loss': total_kl_loss / n_batches,
            'train_physics_loss': total_physics_loss / n_batches,
            'train_diversity_loss': total_diversity_loss / n_batches,
            'beta': beta
        }
    
    def validate_epoch(
        self,
        model: PhysicsInformed3DVAE,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Validate for one epoch."""
        model.eval()
        
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_physics_loss = 0
        total_diversity_loss = 0
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
                    batch_temps, batch_mags,
                    self.config['physics_weight'],
                    self.config['diversity_weight']
                )
                
                # Accumulate losses
                total_loss += losses['total_loss_with_physics'].item()
                total_recon_loss += losses['reconstruction_loss'].item()
                total_kl_loss += losses['kl_loss'].item()
                total_physics_loss += losses['physics_loss'].item()
                total_diversity_loss += losses['diversity_loss'].item()
                n_batches += 1
        
        return {
            'val_loss': total_loss / n_batches,
            'val_recon_loss': total_recon_loss / n_batches,
            'val_kl_loss': total_kl_loss / n_batches,
            'val_physics_loss': total_physics_loss / n_batches,
            'val_diversity_loss': total_diversity_loss / n_batches
        }
    
    def evaluate_model_quality(
        self,
        model: PhysicsInformed3DVAE,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate 3D model quality including latent space analysis."""
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
        
        # Analyze latent space quality
        latent_analysis = self._analyze_latent_space(latents, temperatures, magnetizations)
        
        # Compute reconstruction error
        recon_error = np.mean((originals - reconstructions) ** 2)
        
        # Overall quality assessment
        quality_score = self._compute_quality_score(latent_analysis, recon_error)
        
        return {
            'latent_analysis': latent_analysis,
            'mean_reconstruction_error': recon_error,
            'quality_score': quality_score['score'],
            'quality_level': quality_score['level'],
            'latent_statistics': {
                'mean': np.mean(latents, axis=0).tolist(),
                'std': np.std(latents, axis=0).tolist(),
                'range': (np.min(latents, axis=0).tolist(), np.max(latents, axis=0).tolist()),
                'variance': np.var(latents, axis=0).tolist()
            }
        }
    
    def _analyze_latent_space(self, latents: np.ndarray, temperatures: np.ndarray, magnetizations: np.ndarray) -> Dict:
        """Analyze latent space quality and correlations."""
        analysis = {
            'correlations': {},
            'latent_diversity': {},
            'constant_dimensions': []
        }
        
        # Compute correlations for each latent dimension
        for i in range(latents.shape[1]):
            latent_dim = latents[:, i]
            
            # Check if dimension is constant (the main issue we're trying to fix)
            latent_std = np.std(latent_dim)
            latent_range = np.max(latent_dim) - np.min(latent_dim)
            
            analysis['latent_diversity'][f'latent_{i}'] = {
                'std': latent_std,
                'range': latent_range,
                'is_constant': latent_std < 0.01  # Flag as constant if std < 0.01
            }
            
            if latent_std < 0.01:
                analysis['constant_dimensions'].append(i)
            
            # Compute correlations with physical properties
            if latent_std > 1e-8:  # Only compute correlations for non-constant dimensions
                temp_corr, temp_p = pearsonr(latent_dim, temperatures)
                mag_corr, mag_p = pearsonr(latent_dim, magnetizations)
                
                analysis['correlations'][f'latent_{i}_temp'] = {
                    'correlation': temp_corr,
                    'p_value': temp_p,
                    'significant': temp_p < 0.05
                }
                
                analysis['correlations'][f'latent_{i}_mag'] = {
                    'correlation': mag_corr,
                    'p_value': mag_p,
                    'significant': mag_p < 0.05
                }
            else:
                analysis['correlations'][f'latent_{i}_temp'] = {
                    'correlation': 0.0,
                    'p_value': 1.0,
                    'significant': False
                }
                
                analysis['correlations'][f'latent_{i}_mag'] = {
                    'correlation': 0.0,
                    'p_value': 1.0,
                    'significant': False
                }
        
        # Find best correlations
        temp_correlations = [abs(analysis['correlations'][f'latent_{i}_temp']['correlation']) 
                           for i in range(latents.shape[1])]
        mag_correlations = [abs(analysis['correlations'][f'latent_{i}_mag']['correlation']) 
                          for i in range(latents.shape[1])]
        
        analysis['max_temp_correlation'] = max(temp_correlations) if temp_correlations else 0.0
        analysis['max_mag_correlation'] = max(mag_correlations) if mag_correlations else 0.0
        analysis['n_constant_dimensions'] = len(analysis['constant_dimensions'])
        
        return analysis
    
    def _compute_quality_score(self, latent_analysis: Dict, recon_error: float) -> Dict:
        """Compute overall model quality score."""
        score = 0.0
        
        # Latent diversity score (0-40 points)
        n_constant = latent_analysis['n_constant_dimensions']
        n_total = len(latent_analysis['latent_diversity'])
        
        if n_constant == 0:
            score += 40  # No constant dimensions
        elif n_constant == 1:
            score += 25  # One constant dimension
        elif n_constant < n_total:
            score += 10  # Some diversity
        # else: 0 points for all constant dimensions
        
        # Correlation score (0-40 points)
        max_temp_corr = latent_analysis['max_temp_correlation']
        max_mag_corr = latent_analysis['max_mag_correlation']
        
        if max_temp_corr > 0.7:
            score += 20
        elif max_temp_corr > 0.5:
            score += 15
        elif max_temp_corr > 0.3:
            score += 10
        elif max_temp_corr > 0.1:
            score += 5
        
        if max_mag_corr > 0.7:
            score += 20
        elif max_mag_corr > 0.5:
            score += 15
        elif max_mag_corr > 0.3:
            score += 10
        elif max_mag_corr > 0.1:
            score += 5
        
        # Reconstruction score (0-20 points)
        if recon_error < 0.01:
            score += 20
        elif recon_error < 0.05:
            score += 15
        elif recon_error < 0.1:
            score += 10
        elif recon_error < 0.2:
            score += 5
        
        # Quality level
        if score >= 80:
            level = "Excellent"
        elif score >= 60:
            level = "Good"
        elif score >= 40:
            level = "Fair"
        else:
            level = "Poor"
        
        return {'score': score, 'level': level}
    
    def train_physics_informed_3d_vae(
        self,
        data_path: str,
        output_dir: str,
        latent_dim: int = 2
    ) -> Tuple[PhysicsInformed3DVAE, Dict[str, List], Dict[str, float]]:
        """Train physics-informed 3D VAE with enhanced architecture and memory optimization."""
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load and prepare data
        configurations, temperatures, magnetizations = self.load_3d_data(data_path)
        train_loader, val_loader, test_loader = self.prepare_data(
            configurations, temperatures, magnetizations
        )
        
        # Determine input shape
        sample_batch = next(iter(train_loader))
        input_shape = sample_batch[0].shape[1:]  # Remove batch dimension
        print(f"Input shape: {input_shape}")
        
        # Create model
        model = PhysicsInformed3DVAE(input_shape=input_shape, latent_dim=latent_dim)
        model = model.to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model created with {total_params} total parameters ({trainable_params} trainable)")
        
        # Memory usage estimation
        if torch.cuda.is_available():
            model_memory = total_params * 4 / 1e9  # Approximate memory in GB
            print(f"Estimated model memory: {model_memory:.2f} GB")
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999)
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
            'train_diversity_loss': [],
            'val_diversity_loss': [],
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
                      f"Physics: {train_metrics['train_physics_loss']:.4f}, "
                      f"Diversity: {train_metrics['train_diversity_loss']:.4f}")
            
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
        
        # Save results (convert numpy types to Python types for JSON serialization)
        def convert_numpy_types(obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, np.bool8)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        results = {
            'training_history': convert_numpy_types(history),
            'quality_metrics': convert_numpy_types(quality_metrics),
            'config': convert_numpy_types(self.config),
            'training_time': float(training_time),
            'best_epoch': int(checkpoint['epoch']),
            'best_val_loss': float(checkpoint['val_loss'])
        }
        
        with open(output_path / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save final model
        torch.save(model.state_dict(), output_path / 'final_model.pth')
        
        # Create training plots
        self._create_training_plots(history, output_path)
        
        print(f"Results saved to: {output_path}")
        
        return model, history, quality_metrics
    
    def _create_training_plots(self, history: Dict[str, List], output_path: Path):
        """Create comprehensive training plots for 3D VAE."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Loss curves
        axes[0, 0].plot(history['train_loss'], label='Train', alpha=0.8)
        axes[0, 0].plot(history['val_loss'], label='Validation', alpha=0.8)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reconstruction loss
        axes[0, 1].plot(history['train_recon_loss'], label='Train', alpha=0.8)
        axes[0, 1].plot(history['val_recon_loss'], label='Validation', alpha=0.8)
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # KL loss
        axes[0, 2].plot(history['train_kl_loss'], label='Train', alpha=0.8)
        axes[0, 2].plot(history['val_kl_loss'], label='Validation', alpha=0.8)
        axes[0, 2].set_title('KL Divergence')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Physics loss
        axes[1, 0].plot(history['train_physics_loss'], label='Train', alpha=0.8)
        axes[1, 0].plot(history['val_physics_loss'], label='Validation', alpha=0.8)
        axes[1, 0].set_title('Physics Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Diversity loss
        axes[1, 1].plot(history['train_diversity_loss'], label='Train', alpha=0.8)
        axes[1, 1].plot(history['val_diversity_loss'], label='Validation', alpha=0.8)
        axes[1, 1].set_title('Diversity Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Beta schedule
        axes[1, 2].plot(history['beta_schedule'], alpha=0.8)
        axes[1, 2].set_title('Beta Schedule')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Beta')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main training function."""
    trainer = PhysicsInformed3DVAETrainer()
    
    # Configuration
    data_path = "data/ising_3d_high_quality.h5"
    output_dir = "models/physics_informed_3d_vae"
    
    # Check for high-quality data file
    if not Path(data_path).exists():
        # Try alternative data files
        alternative_files = [
            "data/ising_3d_small.h5",  # This has the right structure
            "data/ising_3d_enhanced_20251031_111625.h5",
            "data/test_enhanced_3d_data.h5"
        ]
        
        for alt_file in alternative_files:
            if Path(alt_file).exists():
                data_path = alt_file
                print(f"Using alternative data file: {data_path}")
                break
        else:
            print(f"No suitable 3D data file found. Checked:")
            print(f"  - data/ising_3d_high_quality.h5")
            for alt_file in alternative_files:
                print(f"  - {alt_file}")
            print("Please ensure high-quality 3D data is available from task 7.1")
            return
    
    # Train model
    model, history, quality = trainer.train_physics_informed_3d_vae(
        data_path=data_path,
        output_dir=output_dir,
        latent_dim=2
    )
    
    # Print final results
    print("\n" + "="*70)
    print("PHYSICS-INFORMED 3D VAE TRAINING COMPLETE")
    print("="*70)
    print(f"Model quality: {quality['quality_level']} ({quality['quality_score']:.1f}/100)")
    print(f"Reconstruction error: {quality['mean_reconstruction_error']:.4f}")
    
    # Latent space analysis
    latent_analysis = quality['latent_analysis']
    print(f"\nLatent space analysis:")
    print(f"  Constant dimensions: {latent_analysis['n_constant_dimensions']}/{len(latent_analysis['latent_diversity'])}")
    print(f"  Max temperature correlation: {latent_analysis['max_temp_correlation']:.3f}")
    print(f"  Max magnetization correlation: {latent_analysis['max_mag_correlation']:.3f}")
    
    # Individual latent dimension analysis
    print(f"\nIndividual latent dimensions:")
    for i in range(len(latent_analysis['latent_diversity'])):
        diversity = latent_analysis['latent_diversity'][f'latent_{i}']
        temp_corr = latent_analysis['correlations'][f'latent_{i}_temp']['correlation']
        mag_corr = latent_analysis['correlations'][f'latent_{i}_mag']['correlation']
        
        status = "CONSTANT" if diversity['is_constant'] else "ACTIVE"
        print(f"  Latent {i}: {status} (std={diversity['std']:.4f}, "
              f"temp_corr={temp_corr:.3f}, mag_corr={mag_corr:.3f})")
    
    print(f"\nModel saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()