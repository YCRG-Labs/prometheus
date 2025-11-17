"""
Adaptive VAE Manager for Multi-Dimensional Input Processing

This module implements an adaptive VAE architecture manager that can handle
both 2D and 3D input data by automatically selecting the appropriate
architecture while maintaining consistent hyperparameters.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Any, Union, Optional

from .vae import ConvolutionalVAE
from .vae_3d import ConvolutionalVAE3D


class AdaptiveVAEManager:
    """
    Manager class for creating VAE architectures adapted to different input dimensions.
    
    Automatically selects between 2D and 3D VAE implementations based on input shape
    while preserving hyperparameters (β=1.0) from the original 2D implementation.
    """
    
    def __init__(self, base_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adaptive VAE manager.
        
        Args:
            base_config: Base configuration dictionary with VAE parameters
        """
        self.base_config = base_config or self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration matching 2D implementation."""
        return {
            'latent_dim': 2,
            'encoder_channels': [32, 64, 128],
            'decoder_channels': [128, 64, 32, 1],
            'kernel_sizes': [3, 3, 3],
            'beta': 1.0  # Maintain identical hyperparameter from 2D implementation
        }
    
    def create_vae_for_input_shape(
        self,
        input_shape: Tuple[int, ...],
        **kwargs
    ) -> Union[ConvolutionalVAE, ConvolutionalVAE3D]:
        """
        Create appropriate VAE architecture based on input shape.
        
        Args:
            input_shape: Shape of input data (channels, height, width) for 2D
                        or (channels, depth, height, width) for 3D
            **kwargs: Additional parameters to override base config
            
        Returns:
            Appropriate VAE instance (2D or 3D)
        """
        # Merge base config with provided kwargs
        config = {**self.base_config, **kwargs}
        
        if len(input_shape) == 3:
            # 2D input: (channels, height, width)
            return self._create_2d_vae(input_shape, config)
        elif len(input_shape) == 4:
            # 3D input: (channels, depth, height, width)
            return self._create_3d_vae(input_shape, config)
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}. "
                           f"Expected 3D (channels, height, width) or "
                           f"4D (channels, depth, height, width)")
    
    def _create_2d_vae(
        self,
        input_shape: Tuple[int, int, int],
        config: Dict[str, Any]
    ) -> ConvolutionalVAE:
        """Create 2D VAE with given configuration."""
        return ConvolutionalVAE(
            input_shape=input_shape,
            latent_dim=config['latent_dim'],
            encoder_channels=config['encoder_channels'],
            decoder_channels=config['decoder_channels'],
            kernel_sizes=config['kernel_sizes'],
            beta=config['beta']
        )
    
    def _create_3d_vae(
        self,
        input_shape: Tuple[int, int, int, int],
        config: Dict[str, Any]
    ) -> ConvolutionalVAE3D:
        """Create 3D VAE with given configuration."""
        return ConvolutionalVAE3D(
            input_shape=input_shape,
            latent_dim=config['latent_dim'],
            encoder_channels=config['encoder_channels'],
            decoder_channels=config['decoder_channels'],
            kernel_sizes=config['kernel_sizes'],
            beta=config['beta']
        )
    
    def adapt_config_for_3d(
        self,
        base_2d_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapt 2D configuration for 3D processing while maintaining hyperparameters.
        
        Args:
            base_2d_config: Configuration from 2D implementation
            
        Returns:
            Adapted configuration for 3D processing
        """
        adapted_config = base_2d_config.copy()
        
        # Maintain identical hyperparameters
        # β=1.0 is preserved from 2D implementation
        # Channel configurations remain the same
        # Kernel sizes remain the same
        
        return adapted_config
    
    def get_model_info(self, model: Union[ConvolutionalVAE, ConvolutionalVAE3D]) -> Dict[str, Any]:
        """
        Get information about the model architecture.
        
        Args:
            model: VAE model instance
            
        Returns:
            Dictionary with model information
        """
        model_type = "2D" if isinstance(model, ConvolutionalVAE) else "3D"
        
        return {
            'model_type': model_type,
            'input_shape': model.input_shape,
            'latent_dim': model.latent_dim,
            'beta': model.beta,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    
    def validate_hyperparameter_consistency(
        self,
        model_2d: ConvolutionalVAE,
        model_3d: ConvolutionalVAE3D
    ) -> bool:
        """
        Validate that 2D and 3D models have consistent hyperparameters.
        
        Args:
            model_2d: 2D VAE model
            model_3d: 3D VAE model
            
        Returns:
            True if hyperparameters are consistent
        """
        # Check critical hyperparameters
        checks = [
            model_2d.beta == model_3d.beta,  # β parameter must be identical
            model_2d.latent_dim == model_3d.latent_dim,  # Latent dimension
            len(model_2d.encoder.channels) == len(model_3d.encoder.channels),  # Architecture depth
        ]
        
        return all(checks)


class VAEFactory:
    """
    Factory class for creating VAE models with standardized configurations.
    """
    
    @staticmethod
    def create_prometheus_vae(
        input_shape: Tuple[int, ...],
        model_type: str = "auto"
    ) -> Union[ConvolutionalVAE, ConvolutionalVAE3D]:
        """
        Create VAE with Prometheus-specific configuration.
        
        Args:
            input_shape: Shape of input data
            model_type: Type of model ("2d", "3d", or "auto")
            
        Returns:
            Configured VAE model
        """
        # Prometheus standard configuration
        prometheus_config = {
            'latent_dim': 2,
            'encoder_channels': [32, 64, 128],
            'decoder_channels': [128, 64, 32, 1],
            'kernel_sizes': [3, 3, 3],
            'beta': 1.0  # Standard β-VAE parameter
        }
        
        manager = AdaptiveVAEManager(prometheus_config)
        
        if model_type == "auto":
            return manager.create_vae_for_input_shape(input_shape)
        elif model_type == "2d":
            if len(input_shape) != 3:
                raise ValueError(f"2D model requires 3D input shape, got {input_shape}")
            return manager._create_2d_vae(input_shape, prometheus_config)
        elif model_type == "3d":
            if len(input_shape) != 4:
                raise ValueError(f"3D model requires 4D input shape, got {input_shape}")
            return manager._create_3d_vae(input_shape, prometheus_config)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
    
    @staticmethod
    def create_from_config(
        config_dict: Dict[str, Any],
        input_shape: Tuple[int, ...]
    ) -> Union[ConvolutionalVAE, ConvolutionalVAE3D]:
        """
        Create VAE from configuration dictionary.
        
        Args:
            config_dict: Configuration parameters
            input_shape: Shape of input data
            
        Returns:
            Configured VAE model
        """
        manager = AdaptiveVAEManager(config_dict)
        return manager.create_vae_for_input_shape(input_shape)