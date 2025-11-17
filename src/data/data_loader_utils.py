"""
Unified Data Loading Utilities for 2D and 3D Configurations

This module provides utilities for automatically detecting and loading
both 2D and 3D datasets with consistent interfaces, enabling seamless
switching between different dimensional data.
"""

import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Optional, Any, Union
from pathlib import Path
import logging

from .preprocessing import IsingDataset, DataPreprocessor
from .preprocessing_3d import IsingDataset3D, DataPreprocessor3D
from ..utils.config import PrometheusConfig
from ..utils.logging_utils import get_logger


class AdaptiveDataLoader:
    """
    Adaptive data loader that automatically handles 2D and 3D datasets.
    
    Automatically detects the dimensionality of the dataset and creates
    appropriate DataLoaders while maintaining consistent interfaces.
    """
    
    def __init__(self, config: PrometheusConfig):
        """
        Initialize adaptive data loader.
        
        Args:
            config: PrometheusConfig with data loading parameters
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize preprocessors for both 2D and 3D
        self.preprocessor_2d = DataPreprocessor(config)
        self.preprocessor_3d = DataPreprocessor3D(config)
    
    def detect_dataset_type(self, hdf5_path: str) -> str:
        """
        Detect whether dataset contains 2D or 3D configurations.
        
        Args:
            hdf5_path: Path to HDF5 dataset file
            
        Returns:
            Dataset type ('2d' or '3d')
        """
        with h5py.File(hdf5_path, 'r') as f:
            config_shape = f['configurations'].shape
            
            # Check dimensionality based on configuration shape
            if len(config_shape) == 4:  # (N, H, W) or (N, C, H, W)
                return '2d'
            elif len(config_shape) == 5:  # (N, C, D, H, W) or (N, D, H, W)
                return '3d'
            else:
                raise ValueError(f"Unsupported configuration shape: {config_shape}")
    
    def create_dataloaders(self,
                          hdf5_path: str,
                          batch_size: Optional[int] = None,
                          num_workers: int = 4,
                          pin_memory: bool = True,
                          load_physics: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create appropriate DataLoaders based on dataset dimensionality.
        
        Args:
            hdf5_path: Path to HDF5 dataset file
            batch_size: Batch size (default: from config)
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for GPU transfer
            load_physics: Whether to load physical quantities
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        dataset_type = self.detect_dataset_type(hdf5_path)
        self.logger.info(f"Detected {dataset_type.upper()} dataset: {hdf5_path}")
        
        if dataset_type == '2d':
            return self.preprocessor_2d.create_dataloaders(
                hdf5_path=hdf5_path,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                load_physics=load_physics
            )
        elif dataset_type == '3d':
            return self.preprocessor_3d.create_dataloaders(
                hdf5_path=hdf5_path,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                load_physics=load_physics
            )
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    def load_dataset_info(self, hdf5_path: str) -> Dict[str, Any]:
        """
        Load dataset information with automatic type detection.
        
        Args:
            hdf5_path: Path to HDF5 dataset file
            
        Returns:
            Dictionary with dataset information including type
        """
        dataset_type = self.detect_dataset_type(hdf5_path)
        
        if dataset_type == '2d':
            info = self.preprocessor_2d.load_dataset_info(hdf5_path)
        elif dataset_type == '3d':
            info = self.preprocessor_3d.load_dataset_info(hdf5_path)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        # Add type information
        info['dataset_type'] = dataset_type
        
        return info
    
    def get_input_shape(self, hdf5_path: str) -> Tuple[int, ...]:
        """
        Get the input shape for VAE architecture selection.
        
        Args:
            hdf5_path: Path to HDF5 dataset file
            
        Returns:
            Input shape tuple for VAE creation
        """
        info = self.load_dataset_info(hdf5_path)
        config_shape = info['configuration_shape']
        
        # Ensure channel dimension is included
        if info['dataset_type'] == '2d':
            if len(config_shape) == 2:  # (H, W)
                return (1, config_shape[0], config_shape[1])  # Add channel dim
            else:  # (C, H, W)
                return config_shape
        elif info['dataset_type'] == '3d':
            if len(config_shape) == 3:  # (D, H, W)
                return (1, config_shape[0], config_shape[1], config_shape[2])  # Add channel dim
            else:  # (C, D, H, W)
                return config_shape
        else:
            raise ValueError(f"Unsupported dataset type: {info['dataset_type']}")


class DatasetFactory:
    """
    Factory class for creating datasets and data loaders.
    """
    
    @staticmethod
    def create_dataset(
        hdf5_path: str,
        split: str = 'train',
        transform: Optional[callable] = None,
        load_physics: bool = False
    ) -> Union[IsingDataset, IsingDataset3D]:
        """
        Create appropriate dataset based on file contents.
        
        Args:
            hdf5_path: Path to HDF5 dataset file
            split: Dataset split ('train', 'val', 'test')
            transform: Optional transform to apply
            load_physics: Whether to load physical quantities
            
        Returns:
            Appropriate dataset instance
        """
        # Detect dataset type
        with h5py.File(hdf5_path, 'r') as f:
            config_shape = f['configurations'].shape
        
        if len(config_shape) == 4:  # 2D dataset
            return IsingDataset(
                hdf5_path=hdf5_path,
                split=split,
                transform=transform,
                load_physics=load_physics
            )
        elif len(config_shape) == 5:  # 3D dataset
            return IsingDataset3D(
                hdf5_path=hdf5_path,
                split=split,
                transform=transform,
                load_physics=load_physics
            )
        else:
            raise ValueError(f"Unsupported configuration shape: {config_shape}")
    
    @staticmethod
    def create_dataloader(
        dataset: Union[IsingDataset, IsingDataset3D],
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False
    ) -> DataLoader:
        """
        Create DataLoader for any dataset type.
        
        Args:
            dataset: Dataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            drop_last: Whether to drop last incomplete batch
            
        Returns:
            DataLoader instance
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )


class ConfigurationNormalizer:
    """
    Unified normalization utilities for both 2D and 3D configurations.
    """
    
    @staticmethod
    def normalize_configurations(
        configurations: np.ndarray,
        method: str = 'sigmoid'
    ) -> np.ndarray:
        """
        Normalize configurations regardless of dimensionality.
        
        Args:
            configurations: Configuration array (2D or 3D)
            method: Normalization method
            
        Returns:
            Normalized configurations
        """
        from .preprocessing import DataNormalizer
        normalizer = DataNormalizer(method=method)
        return normalizer.normalize(configurations)
    
    @staticmethod
    def denormalize_configurations(
        configurations: np.ndarray,
        method: str = 'sigmoid'
    ) -> np.ndarray:
        """
        Denormalize configurations regardless of dimensionality.
        
        Args:
            configurations: Normalized configuration array
            method: Normalization method used
            
        Returns:
            Denormalized configurations
        """
        from .preprocessing import DataNormalizer
        normalizer = DataNormalizer(method=method)
        return normalizer.denormalize(configurations)
    
    @staticmethod
    def convert_to_tensor(
        configurations: np.ndarray,
        add_channel_dim: bool = True
    ) -> torch.Tensor:
        """
        Convert configurations to PyTorch tensor with proper dimensions.
        
        Args:
            configurations: Configuration array
            add_channel_dim: Whether to add channel dimension if missing
            
        Returns:
            PyTorch tensor with proper shape
        """
        tensor = torch.from_numpy(configurations).float()
        
        if add_channel_dim:
            if len(tensor.shape) == 2:  # (H, W) -> (1, H, W)
                tensor = tensor.unsqueeze(0)
            elif len(tensor.shape) == 3:  
                # Could be (N, H, W) for batch of 2D or (D, H, W) for single 3D
                # Check if first dimension is likely batch size (> 8) or spatial (≤ 8)
                if tensor.shape[0] > 8:  # Likely batch dimension
                    tensor = tensor.unsqueeze(1)  # (N, H, W) -> (N, 1, H, W)
                else:  # Likely spatial dimension
                    tensor = tensor.unsqueeze(0)  # (D, H, W) -> (1, D, H, W)
            elif len(tensor.shape) == 4:  # Batch of 3D: (N, D, H, W) -> (N, 1, D, H, W)
                tensor = tensor.unsqueeze(1)
            elif len(tensor.shape) == 5:  # Already has channel dim: (N, C, D, H, W)
                pass  # No change needed
        
        return tensor


def get_data_info_summary(hdf5_path: str) -> str:
    """
    Get a formatted summary of dataset information.
    
    Args:
        hdf5_path: Path to HDF5 dataset file
        
    Returns:
        Formatted string with dataset summary
    """
    loader = AdaptiveDataLoader(PrometheusConfig())
    info = loader.load_dataset_info(hdf5_path)
    
    summary = f"""
Dataset Summary: {Path(hdf5_path).name}
{'='*50}
Type: {info['dataset_type'].upper()}
Configurations: {info['n_configurations']:,}
Shape: {info['configuration_shape']}
Lattice Size: {info['lattice_size']}
Temperature Range: {info['temperature_range'][0]:.3f} - {info['temperature_range'][1]:.3f}
Critical Temperature: {info['critical_temperature']:.3f}
Normalization: {info['normalization_method']}

Split Information:
  Train: {info['train_size']:,} ({info['split_ratios'][0]:.1%})
  Validation: {info['val_size']:,} ({info['split_ratios'][1]:.1%})
  Test: {info['test_size']:,} ({info['split_ratios'][2]:.1%})

Created: {info['creation_time']}
"""
    
    if 'statistics' in info:
        stats = info['statistics']
        if 'configurations' in stats:
            config_stats = stats['configurations']
            summary += f"""
Configuration Statistics:
  Mean: {config_stats['mean']:.4f}
  Std: {config_stats['std']:.4f}
  Range: [{config_stats['min']:.4f}, {config_stats['max']:.4f}]
"""
    
    return summary