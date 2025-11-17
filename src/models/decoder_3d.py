"""
3D Convolutional Decoder for Variational Autoencoder

This module implements the 3D decoder component of the VAE that reconstructs
3D spin configurations from latent space representations, extending the 2D
implementation to handle three-dimensional output data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import math


class ConvolutionalDecoder3D(nn.Module):
    """
    3D Convolutional decoder that reconstructs 3D spin configurations from latent space.
    
    The decoder takes latent vectors and progressively increases dimensionality
    through 3D transposed convolutional layers to produce reconstructed 3D spin
    configurations.
    """
    
    def __init__(
        self,
        latent_dim: int = 2,
        output_shape: Tuple[int, int, int, int] = (1, 32, 32, 32),
        channels: List[int] = [128, 64, 32, 1],
        kernel_sizes: List[int] = [3, 3, 3]
    ):
        """
        Initialize the 3D convolutional decoder.
        
        Args:
            latent_dim: Dimensionality of latent space
            output_shape: Shape of output tensor (channels, depth, height, width)
            channels: Number of channels for each transposed convolutional layer
            kernel_sizes: Kernel sizes for each transposed convolutional layer
        """
        super(ConvolutionalDecoder3D, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        
        # Calculate initial spatial dimensions after first linear layer
        # We need to work backwards from the final output size
        self.initial_depth = 4   # Start with 4x4x4 feature maps
        self.initial_height = 4
        self.initial_width = 4
        self.initial_channels = channels[0]
        
        # Linear layer to project latent vector to initial feature maps
        self.fc_project = nn.Linear(
            latent_dim, 
            self.initial_channels * self.initial_depth * self.initial_height * self.initial_width
        )
        
        # Build 3D transposed convolutional layers
        self.deconv_layers = nn.ModuleList()
        
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            kernel_size = kernel_sizes[i] if i < len(kernel_sizes) else 3
            
            self.deconv_layers.append(
                nn.ConvTranspose3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=1,
                    output_padding=1
                )
            )
        
        # Batch normalization layers (except for output layer)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm3d(channels[i + 1]) 
            for i in range(len(channels) - 2)
        ])
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the 3D decoder.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            
        Returns:
            Reconstructed tensor of shape (batch_size, channels, depth, height, width)
        """
        batch_size = z.size(0)
        
        # Project latent vector to initial feature maps
        x = self.fc_project(z)
        x = x.view(
            batch_size, 
            self.initial_channels, 
            self.initial_depth,
            self.initial_height, 
            self.initial_width
        )
        
        # Apply 3D transposed convolutional layers
        for i, deconv_layer in enumerate(self.deconv_layers):
            x = deconv_layer(x)
            
            # Apply batch normalization and activation (except for output layer)
            if i < len(self.batch_norms):
                x = self.batch_norms[i](x)
                x = F.relu(x)
        
        # Apply final activation (tanh to match spin values in [-1, 1])
        x = torch.tanh(x)
        
        return x
    
    def _calculate_output_size(self, input_size: int, kernel_size: int, 
                              stride: int, padding: int, output_padding: int) -> int:
        """
        Calculate output size for 3D transposed convolution.
        
        Args:
            input_size: Size of input dimension
            kernel_size: Size of convolution kernel
            stride: Stride of convolution
            padding: Padding applied
            output_padding: Additional padding on output
            
        Returns:
            Output size after transposed convolution
        """
        return (input_size - 1) * stride - 2 * padding + kernel_size + output_padding