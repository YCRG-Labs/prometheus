"""
Models module for the Prometheus project.

This module contains the neural network architectures for the Variational
Autoencoder used to discover order parameters in spin systems.
"""

from .encoder import ConvolutionalEncoder
from .decoder import ConvolutionalDecoder
from .vae import ConvolutionalVAE

__all__ = [
    'ConvolutionalEncoder',
    'ConvolutionalDecoder', 
    'ConvolutionalVAE'
]