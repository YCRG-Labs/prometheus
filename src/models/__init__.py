"""
Models module for the Prometheus project.

This module contains the neural network architectures for the Variational
Autoencoder used to discover order parameters in spin systems, supporting
both 2D and 3D configurations.
"""

from .encoder import ConvolutionalEncoder
from .decoder import ConvolutionalDecoder
from .vae import ConvolutionalVAE

# 3D model components
from .encoder_3d import ConvolutionalEncoder3D
from .decoder_3d import ConvolutionalDecoder3D
from .vae_3d import ConvolutionalVAE3D

# Adaptive architecture management
from .adaptive_vae import AdaptiveVAEManager, VAEFactory

__all__ = [
    # 2D components
    'ConvolutionalEncoder',
    'ConvolutionalDecoder', 
    'ConvolutionalVAE',
    
    # 3D components
    'ConvolutionalEncoder3D',
    'ConvolutionalDecoder3D',
    'ConvolutionalVAE3D',
    
    # Adaptive management
    'AdaptiveVAEManager',
    'VAEFactory'
]