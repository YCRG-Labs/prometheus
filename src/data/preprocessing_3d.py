"""
3D Data preprocessing and storage system for the Prometheus project.

This module extends the 2D preprocessing utilities to handle 3D spin configurations,
providing utilities for:
- 3D data normalization and tensor conversion
- HDF5 dataset creation with proper 3D schema
- Train/validation/test split with stratified sampling for 3D data
- PyTorch DataLoader classes for efficient 3D batch loading
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Tuple, Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass
import time

from .data_generator_3d import Dataset3DResult, SystemSizeResult3D
from ..utils.config import PrometheusConfig
from ..utils.logging_utils import get_logger, LoggingContext
from .preprocessing import DataNormalizer, StratifiedTemperatureSampler, DatasetSplit


@dataclass
class DatasetMetadata3D:
    """Metadata for 3D HDF5 dataset."""
    lattice_size: Tuple[int, int, int]  # 3D lattice dimensions
    n_configurations: int
    n_temperatures: int
    temperature_range: Tuple[float, float]
    critical_temperature: float
    normalization_method: str
    split_info: DatasetSplit
    creation_time: str
    config_hash: str
    data_shape: Tuple[int, ...]  # (N, 1, D, H, W) for 3D data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for HDF5 storage."""
        return {
            'lattice_size': self.lattice_size,
            'n_configurations': self.n_configurations,
            'n_temperatures': self.n_temperatures,
            'temperature_range': self.temperature_range,
            'critical_temperature': self.critical_temperature,
            'normalization_method': self.normalization_method,
            'train_size': self.split_info.train_size,
            'val_size': self.split_info.val_size,
            'test_size': self.split_info.test_size,
            'split_ratios': self.split_info.split_ratios,
            'creation_time': self.creation_time,
            'config_hash': self.config_hash,
            'data_shape': self.data_shape
        }


class DataNormalizer3D(DataNormalizer):
    """
    Extends DataNormalizer to handle 3D spin configurations.
    
    Maintains identical normalization methods as 2D implementation
    but handles the additional spatial dimension.
    """
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize 3D spin configuration data.
        
        Args:
            data: Input data with spin values {-1, +1} and shape (N, D, H, W)
            
        Returns:
            Normalized data with same shape
        """
        # Use parent class normalization - works element-wise
        return super().normalize(data)
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """
        Denormalize 3D data back to original spin values.
        
        Args:
            data: Normalized 3D data
            
        Returns:
            Denormalized data (approximately {-1, +1})
        """
        # Use parent class denormalization - works element-wise
        return super().denormalize(data)


class HDF5DatasetWriter3D:
    """Writes processed 3D data to HDF5 format with proper schema."""
    
    def __init__(self, output_path: str, compression: str = 'gzip'):
        """
        Initialize 3D HDF5 writer.
        
        Args:
            output_path: Path for output HDF5 file
            compression: Compression method ('gzip', 'lzf', 'szip', None)
        """
        self.output_path = Path(output_path)
        self.compression = compression
        self.logger = get_logger(__name__)
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def write_dataset(self,
                     configurations: np.ndarray,
                     temperatures: np.ndarray,
                     magnetizations: np.ndarray,
                     energies: np.ndarray,
                     split: DatasetSplit,
                     metadata: DatasetMetadata3D) -> None:
        """
        Write complete 3D dataset to HDF5 file.
        
        Args:
            configurations: Normalized 3D spin configurations (N, D, H, W)
            temperatures: Temperature values
            magnetizations: Magnetization values
            energies: Energy values
            split: Dataset split information
            metadata: Dataset metadata
        """
        self.logger.info(f"Writing 3D dataset to {self.output_path}")
        
        with h5py.File(self.output_path, 'w') as f:
            # Write main data arrays
            self._write_data_arrays(f, configurations, temperatures, magnetizations, energies)
            
            # Write split indices
            self._write_split_indices(f, split)
            
            # Write metadata
            self._write_metadata(f, metadata)
            
            # Write dataset statistics
            self._write_statistics(f, configurations, temperatures, magnetizations, energies)
        
        self.logger.info(f"3D dataset written successfully: {self.output_path}")
        self.logger.info(f"File size: {self.output_path.stat().st_size / (1024**2):.1f} MB")
    
    def _write_data_arrays(self, 
                          f: h5py.File,
                          configurations: np.ndarray,
                          temperatures: np.ndarray,
                          magnetizations: np.ndarray,
                          energies: np.ndarray) -> None:
        """Write main 3D data arrays to HDF5 file."""
        # 3D Configurations (main data)
        f.create_dataset(
            'configurations',
            data=configurations,
            compression=self.compression,
            compression_opts=9 if self.compression == 'gzip' else None,
            shuffle=True,
            fletcher32=True
        )
        
        # Physical quantities (same as 2D)
        f.create_dataset('temperatures', data=temperatures, compression=self.compression)
        f.create_dataset('magnetizations', data=magnetizations, compression=self.compression)
        f.create_dataset('energies', data=energies, compression=self.compression)
        
        self.logger.debug(f"Written 3D data arrays: configurations {configurations.shape}")
    
    def _write_split_indices(self, f: h5py.File, split: DatasetSplit) -> None:
        """Write train/val/test split indices (same as 2D)."""
        splits_group = f.create_group('splits')
        
        splits_group.create_dataset('train_indices', data=split.train_indices)
        splits_group.create_dataset('val_indices', data=split.val_indices)
        splits_group.create_dataset('test_indices', data=split.test_indices)
        
        # Split metadata
        splits_group.attrs['train_size'] = split.train_size
        splits_group.attrs['val_size'] = split.val_size
        splits_group.attrs['test_size'] = split.test_size
        splits_group.attrs['split_ratios'] = split.split_ratios
        
        self.logger.debug("Written split indices")
    
    def _write_metadata(self, f: h5py.File, metadata: DatasetMetadata3D) -> None:
        """Write 3D dataset metadata."""
        meta_group = f.create_group('metadata')
        
        # Convert metadata to attributes
        meta_dict = metadata.to_dict()
        for key, value in meta_dict.items():
            if isinstance(value, (tuple, list)):
                meta_group.attrs[key] = np.array(value)
            else:
                meta_group.attrs[key] = value
        
        self.logger.debug("Written 3D metadata")
    
    def _write_statistics(self,
                         f: h5py.File,
                         configurations: np.ndarray,
                         temperatures: np.ndarray,
                         magnetizations: np.ndarray,
                         energies: np.ndarray) -> None:
        """Write 3D dataset statistics."""
        stats_group = f.create_group('statistics')
        
        # 3D Configuration statistics
        config_stats = stats_group.create_group('configurations')
        config_stats.attrs['mean'] = np.mean(configurations)
        config_stats.attrs['std'] = np.std(configurations)
        config_stats.attrs['min'] = np.min(configurations)
        config_stats.attrs['max'] = np.max(configurations)
        config_stats.attrs['shape'] = configurations.shape
        
        # Temperature statistics (same as 2D)
        temp_stats = stats_group.create_group('temperatures')
        temp_stats.attrs['mean'] = np.mean(temperatures)
        temp_stats.attrs['std'] = np.std(temperatures)
        temp_stats.attrs['min'] = np.min(temperatures)
        temp_stats.attrs['max'] = np.max(temperatures)
        temp_stats.attrs['unique_count'] = len(np.unique(temperatures))
        
        # Magnetization statistics (same as 2D)
        mag_stats = stats_group.create_group('magnetizations')
        mag_stats.attrs['mean'] = np.mean(magnetizations)
        mag_stats.attrs['std'] = np.std(magnetizations)
        mag_stats.attrs['min'] = np.min(magnetizations)
        mag_stats.attrs['max'] = np.max(magnetizations)
        
        # Energy statistics (same as 2D)
        energy_stats = stats_group.create_group('energies')
        energy_stats.attrs['mean'] = np.mean(energies)
        energy_stats.attrs['std'] = np.std(energies)
        energy_stats.attrs['min'] = np.min(energies)
        energy_stats.attrs['max'] = np.max(energies)
        
        self.logger.debug("Written 3D statistics")


class IsingDataset3D(Dataset):
    """PyTorch Dataset for 3D Ising model configurations."""
    
    def __init__(self, 
                 hdf5_path: str,
                 split: str = 'train',
                 transform: Optional[callable] = None,
                 load_physics: bool = False):
        """
        Initialize 3D Ising dataset.
        
        Args:
            hdf5_path: Path to HDF5 dataset file
            split: Dataset split ('train', 'val', 'test')
            transform: Optional transform to apply to configurations
            load_physics: Whether to load physical quantities (temperature, magnetization, energy)
        """
        self.hdf5_path = Path(hdf5_path)
        self.split = split
        self.transform = transform
        self.load_physics = load_physics
        self.logger = get_logger(__name__)
        
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}")
        
        # Load dataset information
        self._load_dataset_info()
        
        self.logger.info(f"Initialized 3D {split} dataset: {len(self)} samples")
    
    def _load_dataset_info(self) -> None:
        """Load 3D dataset information and split indices."""
        with h5py.File(self.hdf5_path, 'r') as f:
            # Load split indices
            splits_group = f['splits']
            self.indices = splits_group[f'{self.split}_indices'][:]
            
            # Load metadata
            metadata_group = f['metadata']
            self.lattice_size = tuple(metadata_group.attrs['lattice_size'])
            self.data_shape = tuple(metadata_group.attrs['data_shape'])
            
            # Store dataset size
            self.n_samples = len(self.indices)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Get 3D dataset item.
        
        Args:
            idx: Sample index
            
        Returns:
            3D configuration tensor, or tuple with physics data if load_physics=True
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        # Get actual data index
        data_idx = self.indices[idx]
        
        with h5py.File(self.hdf5_path, 'r') as f:
            # Load 3D configuration
            config = f['configurations'][data_idx]
            config_tensor = torch.from_numpy(config).float()
            
            # Add channel dimension if needed (for 3D CNN)
            if len(config_tensor.shape) == 3:  # (D, H, W)
                config_tensor = config_tensor.unsqueeze(0)  # Add channel dimension -> (1, D, H, W)
            
            # Apply transform if provided
            if self.transform:
                config_tensor = self.transform(config_tensor)
            
            if not self.load_physics:
                return config_tensor
            
            # Load physical quantities if requested
            temperature = torch.tensor(f['temperatures'][data_idx]).float()
            magnetization = torch.tensor(f['magnetizations'][data_idx]).float()
            energy = torch.tensor(f['energies'][data_idx]).float()
            
            return config_tensor, temperature, magnetization, energy
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get 3D dataset statistics."""
        with h5py.File(self.hdf5_path, 'r') as f:
            stats = {}
            
            # Load split-specific statistics
            configs = f['configurations'][self.indices]
            temps = f['temperatures'][self.indices]
            mags = f['magnetizations'][self.indices]
            energies = f['energies'][self.indices]
            
            stats['configurations'] = {
                'mean': float(np.mean(configs)),
                'std': float(np.std(configs)),
                'min': float(np.min(configs)),
                'max': float(np.max(configs)),
                'shape': configs.shape
            }
            
            stats['temperatures'] = {
                'mean': float(np.mean(temps)),
                'std': float(np.std(temps)),
                'min': float(np.min(temps)),
                'max': float(np.max(temps)),
                'unique_count': len(np.unique(temps))
            }
            
            stats['magnetizations'] = {
                'mean': float(np.mean(mags)),
                'std': float(np.std(mags)),
                'min': float(np.min(mags)),
                'max': float(np.max(mags))
            }
            
            stats['energies'] = {
                'mean': float(np.mean(energies)),
                'std': float(np.std(energies)),
                'min': float(np.min(energies)),
                'max': float(np.max(energies))
            }
            
            return stats


class DataPreprocessor3D:
    """Main class for 3D data preprocessing and storage pipeline."""
    
    def __init__(self, config: PrometheusConfig):
        """
        Initialize 3D data preprocessor.
        
        Args:
            config: PrometheusConfig with preprocessing parameters
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.normalizer = DataNormalizer3D(method='sigmoid')  # Default normalization
        
        # Create output directory
        self.data_dir = Path(config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def process_dataset(self,
                       result: Dataset3DResult,
                       output_path: Optional[str] = None,
                       normalization_method: str = 'sigmoid',
                       split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                       system_size: Optional[int] = None) -> str:
        """
        Process complete 3D dataset from Dataset3DResult to HDF5.
        
        Args:
            result: Dataset3DResult from 3D data generation
            output_path: Output HDF5 file path
            normalization_method: Normalization method for configurations
            split_ratios: Train/val/test split ratios
            system_size: Specific system size to process (if None, uses largest available)
            
        Returns:
            Path to created HDF5 file
        """
        self.logger.info("Starting 3D dataset preprocessing")
        
        with LoggingContext("3D Dataset Preprocessing"):
            # Generate output path if not provided
            if output_path is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                size_suffix = f"_L{system_size}" if system_size else ""
                output_path = str(self.data_dir / f"ising_3d_processed_{timestamp}{size_suffix}.h5")
            
            # Select system size to process
            if system_size is None:
                system_size = max(result.system_size_results.keys())
                self.logger.info(f"Using largest available system size: L={system_size}")
            
            if system_size not in result.system_size_results:
                raise ValueError(f"System size {system_size} not found in results")
            
            system_result = result.system_size_results[system_size]
            
            # Extract and flatten 3D data
            configurations, temperatures, magnetizations, energies = self._extract_3d_data(system_result)
            
            # Normalize configurations
            self.normalizer = DataNormalizer3D(method=normalization_method)
            normalized_configs = self.normalizer.normalize(configurations)
            
            # Create stratified splits
            sampler = StratifiedTemperatureSampler(
                temperatures=temperatures,
                split_ratios=split_ratios,
                random_state=self.config.seed
            )
            split = sampler.create_splits()
            
            # Create 3D metadata
            metadata = DatasetMetadata3D(
                lattice_size=(system_size, system_size, system_size),
                n_configurations=len(configurations),
                n_temperatures=len(system_result.temperatures),
                temperature_range=(float(np.min(system_result.temperatures)), 
                                 float(np.max(system_result.temperatures))),
                critical_temperature=4.511,  # 3D Ising critical temperature
                normalization_method=normalization_method,
                split_info=split,
                creation_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                config_hash=f"3d_L{system_size}",
                data_shape=normalized_configs.shape
            )
            
            # Write to HDF5
            writer = HDF5DatasetWriter3D(output_path)
            writer.write_dataset(
                configurations=normalized_configs,
                temperatures=temperatures,
                magnetizations=magnetizations,
                energies=energies,
                split=split,
                metadata=metadata
            )
            
            self.logger.info(f"3D dataset preprocessing complete: {output_path}")
            
            return output_path
    
    def _extract_3d_data(self, system_result: SystemSizeResult3D) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract and flatten 3D data from SystemSizeResult3D."""
        self.logger.info("Extracting and flattening 3D configuration data")
        
        all_configs = []
        all_temps = []
        all_mags = []
        all_energies = []
        
        for temp_idx, temp_configs in enumerate(system_result.configurations_per_temp):
            temperature = system_result.temperatures[temp_idx]
            
            for config in temp_configs:
                all_configs.append(config.spins)
                all_temps.append(temperature)
                all_mags.append(config.magnetization)
                all_energies.append(config.energy)
        
        # Convert to numpy arrays
        configurations = np.array(all_configs, dtype=np.float32)
        temperatures = np.array(all_temps, dtype=np.float32)
        magnetizations = np.array(all_mags, dtype=np.float32)
        energies = np.array(all_energies, dtype=np.float32)
        
        self.logger.info(f"Extracted {len(configurations)} 3D configurations")
        self.logger.info(f"3D configuration shape: {configurations.shape}")
        
        return configurations, temperatures, magnetizations, energies
    
    def create_dataloaders(self,
                          hdf5_path: str,
                          batch_size: Optional[int] = None,
                          num_workers: int = 4,
                          pin_memory: bool = True,
                          load_physics: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for 3D train/val/test splits.
        
        Args:
            hdf5_path: Path to processed 3D HDF5 dataset
            batch_size: Batch size (default: from config)
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for GPU transfer
            load_physics: Whether to load physical quantities
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if batch_size is None:
            batch_size = self.config.training.batch_size
        
        self.logger.info(f"Creating 3D DataLoaders with batch_size={batch_size}")
        
        # Create 3D datasets
        train_dataset = IsingDataset3D(hdf5_path, split='train', load_physics=load_physics)
        val_dataset = IsingDataset3D(hdf5_path, split='val', load_physics=load_physics)
        test_dataset = IsingDataset3D(hdf5_path, split='test', load_physics=load_physics)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True  # For consistent batch sizes during training
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        self.logger.info(f"Created 3D DataLoaders: train={len(train_loader)} batches, "
                        f"val={len(val_loader)} batches, test={len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader
    
    def load_dataset_info(self, hdf5_path: str) -> Dict[str, Any]:
        """
        Load 3D dataset information from HDF5 file.
        
        Args:
            hdf5_path: Path to 3D HDF5 dataset
            
        Returns:
            Dictionary with 3D dataset information
        """
        with h5py.File(hdf5_path, 'r') as f:
            info = {}
            
            # Basic info
            info['n_configurations'] = f['configurations'].shape[0]
            info['configuration_shape'] = f['configurations'].shape[1:]
            info['is_3d'] = len(f['configurations'].shape) == 5  # (N, 1, D, H, W)
            
            # Split info
            splits = f['splits']
            info['train_size'] = splits.attrs['train_size']
            info['val_size'] = splits.attrs['val_size']
            info['test_size'] = splits.attrs['test_size']
            info['split_ratios'] = tuple(splits.attrs['split_ratios'])
            
            # Metadata
            metadata = f['metadata']
            info['lattice_size'] = tuple(metadata.attrs['lattice_size'])
            info['temperature_range'] = tuple(metadata.attrs['temperature_range'])
            info['critical_temperature'] = metadata.attrs['critical_temperature']
            
            # Handle string attributes (may or may not need decoding)
            norm_method = metadata.attrs['normalization_method']
            info['normalization_method'] = norm_method.decode('utf-8') if isinstance(norm_method, bytes) else norm_method
            
            creation_time = metadata.attrs['creation_time']
            info['creation_time'] = creation_time.decode('utf-8') if isinstance(creation_time, bytes) else creation_time
            
            # Statistics
            if 'statistics' in f:
                stats = f['statistics']
                info['statistics'] = {}
                for group_name in stats.keys():
                    group = stats[group_name]
                    info['statistics'][group_name] = dict(group.attrs)
        
        return info