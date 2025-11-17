"""
Enhanced Physics-Informed VAE Training

This module provides enhanced VAE training with stronger physics constraints
to improve critical exponent extraction accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EnhancedPhysicsLossWeights:
    """Weights for different physics loss components."""
    reconstruction: float = 1.0
    kl_divergence: float = 1.0
    magnetization_correlation: float = 2.0  # Increased weight
    energy_consistency: float = 1.0
    temperature_ordering: float = 1.5  # New: enforce temperature ordering
    critical_enhancement: float = 1.0  # New: enhance critical region


class EnhancedPhysicsLoss(nn.Module):
    """
    Enhanced physics-informed loss function with stronger constraints.
    """
    
    def __init__(self, weights: Optional[EnhancedPhysicsLossWeights] = None):
        """
        Initialize enhanced physics loss.
        
        Args:
            weights: Loss component weights
        """
        super().__init__()
        
        if weights is None:
            weights = EnhancedPhysicsLossWeights()
        
        self.weights = weights
    
    def forward(self, 
                recon_x: torch.Tensor,
                x: torch.Tensor,
                mu: torch.Tensor,
                logvar: torch.Tensor,
                z: torch.Tensor,
                temperatures: Optional[torch.Tensor] = None,
                magnetizations: Optional[torch.Tensor] = None,
                energies: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute enhanced physics-informed loss.
        
        Args:
            recon_x: Reconstructed configurations
            x: Original configurations
            mu: Latent mean
            logvar: Latent log variance
            z: Latent samples
            temperatures: Temperature values
            magnetizations: Magnetization values
            energies: Energy values
            
        Returns:
            Dictionary of loss components
        """
        batch_size = x.size(0)
        
        # 1. Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / batch_size
        
        # 2. KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        # 3. Magnetization correlation loss (enhanced)
        mag_loss = torch.tensor(0.0, device=x.device)
        if magnetizations is not None:
            # Encourage strong correlation between latent dimensions and magnetization
            for dim in range(z.size(1)):
                z_dim = z[:, dim]
                
                # Pearson correlation
                z_centered = z_dim - z_dim.mean()
                mag_centered = magnetizations - magnetizations.mean()
                
                correlation = (z_centered * mag_centered).sum() / (
                    torch.sqrt((z_centered ** 2).sum() * (mag_centered ** 2).sum()) + 1e-8
                )
                
                # Maximize absolute correlation for at least one dimension
                mag_loss += (1.0 - torch.abs(correlation)) / z.size(1)
        
        # 4. Energy consistency loss
        energy_loss = torch.tensor(0.0, device=x.device)
        if energies is not None and z.size(1) > 1:
            # Encourage correlation between latent space and energy
            z_energy = z[:, 1] if z.size(1) > 1 else z[:, 0]
            
            z_centered = z_energy - z_energy.mean()
            e_centered = energies - energies.mean()
            
            correlation = (z_centered * e_centered).sum() / (
                torch.sqrt((z_centered ** 2).sum() * (e_centered ** 2).sum()) + 1e-8
            )
            
            energy_loss = 1.0 - torch.abs(correlation)
        
        # 5. Temperature ordering loss (NEW)
        temp_order_loss = torch.tensor(0.0, device=x.device)
        if temperatures is not None and z.size(1) > 0:
            # Latent space should have monotonic relationship with temperature
            # Sort by temperature and check if latent values follow
            sorted_indices = torch.argsort(temperatures)
            z_sorted = z[sorted_indices, 0]  # Use first latent dimension
            
            # Compute differences
            z_diffs = z_sorted[1:] - z_sorted[:-1]
            
            # Penalize non-monotonic behavior
            # Allow both increasing and decreasing, but penalize reversals
            sign_changes = (z_diffs[1:] * z_diffs[:-1]) < 0
            temp_order_loss = sign_changes.float().mean()
        
        # 6. Critical region enhancement loss (NEW)
        critical_loss = torch.tensor(0.0, device=x.device)
        if temperatures is not None and magnetizations is not None:
            # Identify critical region (around Tc ≈ 4.5 for 3D Ising)
            tc_estimate = 4.5
            critical_mask = torch.abs(temperatures - tc_estimate) < 0.5
            
            if critical_mask.sum() > 0:
                # In critical region, latent space should show high variance
                z_critical = z[critical_mask]
                z_non_critical = z[~critical_mask]
                
                if len(z_critical) > 0 and len(z_non_critical) > 0:
                    var_critical = z_critical.var(dim=0).mean()
                    var_non_critical = z_non_critical.var(dim=0).mean()
                    
                    # Encourage higher variance in critical region
                    critical_loss = F.relu(var_non_critical - var_critical)
        
        # Combine losses
        total_loss = (
            self.weights.reconstruction * recon_loss +
            self.weights.kl_divergence * kl_loss +
            self.weights.magnetization_correlation * mag_loss +
            self.weights.energy_consistency * energy_loss +
            self.weights.temperature_ordering * temp_order_loss +
            self.weights.critical_enhancement * critical_loss
        )
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'kl': kl_loss,
            'magnetization': mag_loss,
            'energy': energy_loss,
            'temperature_ordering': temp_order_loss,
            'critical_enhancement': critical_loss
        }


class EnhancedPhysicsVAETrainer:
    """
    Enhanced VAE trainer with stronger physics constraints.
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 learning_rate: float = 1e-3,
                 loss_weights: Optional[EnhancedPhysicsLossWeights] = None):
        """
        Initialize enhanced trainer.
        
        Args:
            model: VAE model
            device: Torch device
            learning_rate: Learning rate
            loss_weights: Physics loss weights
        """
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = EnhancedPhysicsLoss(loss_weights)
        
        # Learning rate scheduler for better convergence
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
    
    def train_epoch(self,
                   configurations: torch.Tensor,
                   temperatures: Optional[torch.Tensor] = None,
                   magnetizations: Optional[torch.Tensor] = None,
                   energies: Optional[torch.Tensor] = None,
                   batch_size: int = 32) -> Dict[str, float]:
        """
        Train for one epoch with enhanced physics loss.
        
        Args:
            configurations: Spin configurations
            temperatures: Temperature values
            magnetizations: Magnetization values
            energies: Energy values
            batch_size: Batch size
            
        Returns:
            Dictionary of average losses
        """
        self.model.train()
        
        n_samples = len(configurations)
        indices = torch.randperm(n_samples)
        
        epoch_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'kl': 0.0,
            'magnetization': 0.0,
            'energy': 0.0,
            'temperature_ordering': 0.0,
            'critical_enhancement': 0.0
        }
        
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            
            # Get batch data
            batch_configs = configurations[batch_indices].to(self.device)
            batch_temps = temperatures[batch_indices].to(self.device) if temperatures is not None else None
            batch_mags = magnetizations[batch_indices].to(self.device) if magnetizations is not None else None
            batch_energies = energies[batch_indices].to(self.device) if energies is not None else None
            
            # Forward pass
            recon_batch, mu, logvar = self.model(batch_configs)
            z = self.model.reparameterize(mu, logvar)
            
            # Compute loss
            losses = self.loss_fn(
                recon_batch, batch_configs, mu, logvar, z,
                batch_temps, batch_mags, batch_energies
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            
            n_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        # Update learning rate
        self.scheduler.step(epoch_losses['total'])
        
        return epoch_losses
    
    def validate(self,
                configurations: torch.Tensor,
                temperatures: Optional[torch.Tensor] = None,
                magnetizations: Optional[torch.Tensor] = None,
                energies: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Validate model with enhanced physics loss.
        
        Args:
            configurations: Validation configurations
            temperatures: Temperature values
            magnetizations: Magnetization values
            energies: Energy values
            
        Returns:
            Dictionary of validation losses
        """
        self.model.eval()
        
        with torch.no_grad():
            configs = configurations.to(self.device)
            temps = temperatures.to(self.device) if temperatures is not None else None
            mags = magnetizations.to(self.device) if magnetizations is not None else None
            engs = energies.to(self.device) if energies is not None else None
            
            # Forward pass
            recon, mu, logvar = self.model(configs)
            z = self.model.reparameterize(mu, logvar)
            
            # Compute loss
            losses = self.loss_fn(recon, configs, mu, logvar, z, temps, mags, engs)
            
            # Convert to float
            return {key: val.item() for key, val in losses.items()}


def create_enhanced_physics_trainer(model: nn.Module,
                                   device: torch.device,
                                   learning_rate: float = 1e-3,
                                   loss_weights: Optional[EnhancedPhysicsLossWeights] = None) -> EnhancedPhysicsVAETrainer:
    """
    Factory function to create enhanced physics VAE trainer.
    
    Args:
        model: VAE model
        device: Torch device
        learning_rate: Learning rate
        loss_weights: Physics loss weights
        
    Returns:
        Configured EnhancedPhysicsVAETrainer instance
    """
    return EnhancedPhysicsVAETrainer(model, device, learning_rate, loss_weights)