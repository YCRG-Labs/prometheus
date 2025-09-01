"""
Convolutional Encoder for Variational Autoencoder

This module implements the encoder component of the VAE that compresses
32x32 spin configurations into a 2D latent space representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class ConvolutionalEncoder(nn.Module):
    """
    Convolutional encoder that compresses spin configurations to latent space.
    
    The encoder takes 32x32 spin configurations and progressively reduces
    dimensionality through convolutional layers to produce mean and log-variance
    parameters for the latent distribution.
    """
    
    def __init__(
        self, 
        input_shape: Tuple[int, int, int] = (1, 32, 32),
        latent_dim: int = 2,
        channels: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [3, 3, 3],
        activation: str = 'relu'
    ):
        """
        Initialize the convolutional encoder.
        
        Args:
            input_shape: Shape of input tensor (channels, height, width)
            latent_dim: Dimensionality of latent space
            channels: Number of channels for each convolutional layer
            kernel_sizes: Kernel sizes for each convolutional layer
            activation: Activation function ('relu', 'leaky_relu', 'elu', 'gelu', 'swish')
        """
        super(ConvolutionalEncoder, self).__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.activation_name = activation
        
        # Create activation function
        self.activation = self._create_activation_function(activation)
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_shape[0]
        
        for out_channels, kernel_size in zip(channels, kernel_sizes):
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=1
                )
            )
            in_channels = out_channels
        
        # Calculate the flattened size after convolutions
        self.flattened_size = self._calculate_flattened_size()
        
        # Fully connected layers for latent parameters
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
    
    def _create_activation_function(self, activation: str) -> nn.Module:
        """Create activation function from string name."""
        activation_map = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'silu': nn.SiLU()
        }
        
        if activation.lower() not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}. Supported: {list(activation_map.keys())}")
        
        return activation_map[activation.lower()]
        
    def _calculate_flattened_size(self) -> int:
        """Calculate the size of flattened feature maps after convolutions."""
        with torch.no_grad():
            # Create dummy input to calculate output size
            dummy_input = torch.zeros(1, *self.input_shape)
            x = dummy_input
            
            for conv_layer in self.conv_layers:
                x = F.relu(conv_layer(x))
            
            return x.numel()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (mu, logvar) tensors for latent distribution parameters
        """
        # Apply convolutional layers with specified activation
        for conv_layer in self.conv_layers:
            x = self.activation(conv_layer(x))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Generate latent distribution parameters
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        if self.training:
            # Sample from standard normal and transform
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference, use mean
            return mu