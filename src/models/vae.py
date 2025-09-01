"""
Convolutional Variational Autoencoder for Spin System Analysis

This module implements the complete VAE architecture that combines the encoder
and decoder components with proper loss functions for unsupervised learning
of order parameters in spin systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional

from .encoder import ConvolutionalEncoder
from .decoder import ConvolutionalDecoder


class ConvolutionalVAE(nn.Module):
    """
    Complete Convolutional Variational Autoencoder.
    
    Combines encoder and decoder networks with ELBO loss function for
    unsupervised learning of latent representations of spin configurations.
    Supports β-VAE formulation for controlling the balance between
    reconstruction and regularization.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 32, 32),
        latent_dim: int = 2,
        encoder_channels: List[int] = [32, 64, 128],
        decoder_channels: List[int] = [128, 64, 32, 1],
        kernel_sizes: List[int] = [3, 3, 3],
        beta: float = 1.0
    ):
        """
        Initialize the Convolutional VAE.
        
        Args:
            input_shape: Shape of input tensor (channels, height, width)
            latent_dim: Dimensionality of latent space
            encoder_channels: Channel sizes for encoder conv layers
            decoder_channels: Channel sizes for decoder conv layers
            kernel_sizes: Kernel sizes for conv layers
            beta: Beta parameter for β-VAE (weight of KL divergence)
        """
        super(ConvolutionalVAE, self).__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Initialize encoder and decoder
        self.encoder = ConvolutionalEncoder(
            input_shape=input_shape,
            latent_dim=latent_dim,
            channels=encoder_channels,
            kernel_sizes=kernel_sizes
        )
        
        self.decoder = ConvolutionalDecoder(
            latent_dim=latent_dim,
            output_shape=input_shape,
            channels=decoder_channels,
            kernel_sizes=kernel_sizes
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (mu, logvar) for latent distribution
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstruction.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            
        Returns:
            Reconstructed tensor of shape (batch_size, channels, height, width)
        """
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        return self.encoder.reparameterize(mu, logvar)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the complete VAE.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        # Encode to latent distribution parameters
        mu, logvar = self.encode(x)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)
        
        # Decode to reconstruction
        reconstruction = self.decode(z)
        
        return reconstruction, mu, logvar
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        reconstruction: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor,
        reduction: str = 'mean'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ELBO loss with reconstruction and KL divergence terms.
        
        Args:
            x: Original input tensor
            reconstruction: Reconstructed tensor from decoder
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            reduction: Reduction method ('mean', 'sum', or 'none')
            
        Returns:
            Dictionary containing loss components
        """
        batch_size = x.size(0)
        
        # Reconstruction loss (Binary Cross Entropy)
        # Convert spin values from [-1, 1] to [0, 1] for BCE
        x_normalized = (x + 1.0) / 2.0
        recon_normalized = (reconstruction + 1.0) / 2.0
        
        recon_loss = F.binary_cross_entropy(
            recon_normalized, 
            x_normalized, 
            reduction='none'
        )
        
        # Sum over spatial dimensions, average over batch if specified
        recon_loss = recon_loss.view(batch_size, -1).sum(dim=1)
        
        # KL divergence loss
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        # Total ELBO loss (negative ELBO, so we minimize)
        total_loss = recon_loss + self.beta * kl_loss
        
        # Apply reduction
        if reduction == 'mean':
            recon_loss = recon_loss.mean()
            kl_loss = kl_loss.mean()
            total_loss = total_loss.mean()
        elif reduction == 'sum':
            recon_loss = recon_loss.sum()
            kl_loss = kl_loss.sum()
            total_loss = total_loss.sum()
        # If reduction == 'none', keep per-sample losses
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'beta_weighted_kl': self.beta * kl_loss
        }
    
    def compute_elbo(
        self, 
        x: torch.Tensor, 
        reconstruction: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Evidence Lower Bound (ELBO).
        
        Args:
            x: Original input tensor
            reconstruction: Reconstructed tensor from decoder
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            ELBO value (higher is better)
        """
        loss_dict = self.compute_loss(x, reconstruction, mu, logvar, reduction='mean')
        
        # ELBO is negative of the loss (since we minimize negative ELBO)
        elbo = -loss_dict['total_loss']
        
        return elbo
    
    def sample(self, num_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Generate samples from the learned latent distribution.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Generated samples of shape (num_samples, channels, height, width)
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Sample from prior distribution N(0, I)
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # Decode samples
        with torch.no_grad():
            samples = self.decode(z)
        
        return samples
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input through the VAE (encode then decode).
        
        Args:
            x: Input tensor to reconstruct
            
        Returns:
            Reconstructed tensor
        """
        with torch.no_grad():
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            reconstruction = self.decode(z)
        
        return reconstruction
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation of input (using mean of distribution).
        
        Args:
            x: Input tensor
            
        Returns:
            Latent representation (mean of q(z|x))
        """
        with torch.no_grad():
            mu, _ = self.encode(x)
        
        return mu
    
    def set_beta(self, beta: float) -> None:
        """
        Update the beta parameter for β-VAE.
        
        Args:
            beta: New beta value
        """
        self.beta = beta