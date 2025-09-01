"""
Data preprocessing and storage system for the Prometheus project.

This module provides utilities for:
- Data normalization and tensor conversion
- HDF5 dataset creation with proper schema
- Train/validation/test split with stratified sampling
- PyTorch DataLoader classes for efficient batch loading
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import logging
from typing import Tuple, Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass
import time
from sklearn.model_selection import train_test_split

from .data_generator import TemperatureSweepResult
from ..utils.config import PrometheusConfig
from ..utils.logging_utils import get_logger, LoggingContext


@dataclass
class DatasetSplit:
    """Information about dataset splits."""
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    train_size: int
    val_size: int
    test_size: int
    split_ratios: Tuple[float, float, float]  # (train, val, test)
    
    @property
    def total_size(self) -> int:
        return self.train_size + self.val_size + self.test_size


@dataclass
class DatasetMetadata:
    """Metadata for HDF5 dataset."""
    lattice_size: Tuple[int, int]
    n_configurations: int
    n_temperatures: int
    temperature_range: Tuple[float, float]
    critical_temperature: float
    normalization_method: str
    split_info: DatasetSplit
    creation_time: str
    config_hash: str
    data_shape: Tuple[int, ...]
    
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


class DataNormalizer:
    """Handles different normalization methods for spin configurations."""
    
    def __init__(self, method: str = "tanh"):
        """
        Initialize normalizer.
        
        Args:
            method: Normalization method ('tanh', 'sigmoid', 'minmax', 'none')
        """
        self.method = method
        self.logger = get_logger(__name__)
        
        if method not in ['tanh', 'sigmoid', 'minmax', 'none']:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize spin configuration data.
        
        Args:
            data: Input data with spin values {-1, +1}
            
        Returns:
            Normalized data
        """
        if self.method == 'none':
            return data.astype(np.float32)
        
        elif self.method == 'tanh':
            # Map {-1, +1} to approximately {-0.76, +0.76} using tanh
            # This provides a natural mapping that preserves the bipolar nature
            return np.tanh(data * 1.0).astype(np.float32)
        
        elif self.method == 'sigmoid':
            # Map {-1, +1} to {0, 1} using sigmoid-like transformation
            return ((data + 1) / 2).astype(np.float32)
        
        elif self.method == 'minmax':
            # Standard min-max normalization to [0, 1]
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max == data_min:
                return np.zeros_like(data, dtype=np.float32)
            return ((data - data_min) / (data_max - data_min)).astype(np.float32)
        
        else:
            raise ValueError(f"Normalization method {self.method} not implemented")
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """
        Denormalize data back to original spin values.
        
        Args:
            data: Normalized data
            
        Returns:
            Denormalized data (approximately {-1, +1})
        """
        if self.method == 'none':
            return data
        
        elif self.method == 'tanh':
            # Inverse tanh mapping
            return np.arctanh(np.clip(data, -0.99, 0.99))
        
        elif self.method == 'sigmoid':
            # Map [0, 1] back to {-1, +1}
            return (data * 2 - 1)
        
        elif self.method == 'minmax':
            # Map [0, 1] back to {-1, +1}
            return (data * 2 - 1)
        
        else:
            raise ValueError(f"Denormalization method {self.method} not implemented")


class StratifiedTemperatureSampler:
    """Handles stratified sampling by temperature for train/val/test splits."""
    
    def __init__(self, 
                 temperatures: np.ndarray,
                 split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                 random_state: int = 42):
        """
        Initialize stratified sampler.
        
        Args:
            temperatures: Array of temperatures for each configuration
            split_ratios: (train, val, test) split ratios
            random_state: Random seed for reproducibility
        """
        self.temperatures = temperatures
        self.split_ratios = split_ratios
        self.random_state = random_state
        self.logger = get_logger(__name__)
        
        # Validate split ratios
        if abs(sum(split_ratios) - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        if any(ratio <= 0 for ratio in split_ratios):
            raise ValueError("All split ratios must be positive")
    
    def create_splits(self) -> DatasetSplit:
        """
        Create stratified train/validation/test splits.
        
        Returns:
            DatasetSplit with indices for each split
        """
        self.logger.info(f"Creating stratified splits with ratios {self.split_ratios}")
        
        n_samples = len(self.temperatures)
        indices = np.arange(n_samples)
        
        # Get unique temperatures and their counts
        unique_temps, temp_counts = np.unique(self.temperatures, return_counts=True)
        self.logger.info(f"Found {len(unique_temps)} unique temperatures")
        
        # Initialize split arrays
        train_indices = []
        val_indices = []
        test_indices = []
        
        # For each temperature, split proportionally
        for temp in unique_temps:
            temp_mask = self.temperatures == temp
            temp_indices = indices[temp_mask]
            n_temp = len(temp_indices)
            
            if n_temp < 3:
                # If too few samples, put all in training
                train_indices.extend(temp_indices)
                continue
            
            # Calculate split sizes for this temperature
            n_train = max(1, int(n_temp * self.split_ratios[0]))
            n_val = max(1, int(n_temp * self.split_ratios[1]))
            n_test = n_temp - n_train - n_val
            
            # Ensure we don't exceed available samples
            if n_test < 0:
                n_val = max(1, n_temp - n_train - 1)
                n_test = n_temp - n_train - n_val
            
            # Randomly shuffle indices for this temperature
            np.random.seed(self.random_state + int(temp * 1000))  # Deterministic per temperature
            shuffled_indices = np.random.permutation(temp_indices)
            
            # Split indices
            train_indices.extend(shuffled_indices[:n_train])
            val_indices.extend(shuffled_indices[n_train:n_train + n_val])
            if n_test > 0:
                test_indices.extend(shuffled_indices[n_train + n_val:])
        
        # Convert to numpy arrays and shuffle
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        test_indices = np.array(test_indices)
        
        # Final shuffle of each split
        np.random.seed(self.random_state)
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)
        
        split = DatasetSplit(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            train_size=len(train_indices),
            val_size=len(val_indices),
            test_size=len(test_indices),
            split_ratios=self.split_ratios
        )
        
        self.logger.info(f"Split created: train={split.train_size}, val={split.val_size}, test={split.test_size}")
        
        # Validate splits
        self._validate_splits(split)
        
        return split
    
    def _validate_splits(self, split: DatasetSplit) -> None:
        """Validate that splits are correct and non-overlapping."""
        all_indices = np.concatenate([split.train_indices, split.val_indices, split.test_indices])
        
        # Check for overlaps
        if len(np.unique(all_indices)) != len(all_indices):
            raise ValueError("Split indices overlap")
        
        # Check coverage
        if len(all_indices) != len(self.temperatures):
            raise ValueError("Split indices don't cover all samples")
        
        # Check temperature distribution
        train_temps = self.temperatures[split.train_indices]
        val_temps = self.temperatures[split.val_indices]
        test_temps = self.temperatures[split.test_indices]
        
        unique_temps = np.unique(self.temperatures)
        train_unique = np.unique(train_temps)
        val_unique = np.unique(val_temps)
        test_unique = np.unique(test_temps)
        
        # Ensure training set has good temperature coverage
        coverage_ratio = len(train_unique) / len(unique_temps)
        if coverage_ratio < 0.8:
            self.logger.warning(f"Training set only covers {coverage_ratio:.1%} of temperatures")
        
        self.logger.info("Split validation passed")


class HDF5DatasetWriter:
    """Writes processed data to HDF5 format with proper schema."""
    
    def __init__(self, output_path: str, compression: str = 'gzip'):
        """
        Initialize HDF5 writer.
        
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
                     metadata: DatasetMetadata) -> None:
        """
        Write complete dataset to HDF5 file.
        
        Args:
            configurations: Normalized spin configurations
            temperatures: Temperature values
            magnetizations: Magnetization values
            energies: Energy values
            split: Dataset split information
            metadata: Dataset metadata
        """
        self.logger.info(f"Writing dataset to {self.output_path}")
        
        with h5py.File(self.output_path, 'w') as f:
            # Write main data arrays
            self._write_data_arrays(f, configurations, temperatures, magnetizations, energies)
            
            # Write split indices
            self._write_split_indices(f, split)
            
            # Write metadata
            self._write_metadata(f, metadata)
            
            # Write dataset statistics
            self._write_statistics(f, configurations, temperatures, magnetizations, energies)
        
        self.logger.info(f"Dataset written successfully: {self.output_path}")
        self.logger.info(f"File size: {self.output_path.stat().st_size / (1024**2):.1f} MB")
    
    def _write_data_arrays(self, 
                          f: h5py.File,
                          configurations: np.ndarray,
                          temperatures: np.ndarray,
                          magnetizations: np.ndarray,
                          energies: np.ndarray) -> None:
        """Write main data arrays to HDF5 file."""
        # Configurations (main data)
        f.create_dataset(
            'configurations',
            data=configurations,
            compression=self.compression,
            compression_opts=9 if self.compression == 'gzip' else None,
            shuffle=True,
            fletcher32=True
        )
        
        # Physical quantities
        f.create_dataset('temperatures', data=temperatures, compression=self.compression)
        f.create_dataset('magnetizations', data=magnetizations, compression=self.compression)
        f.create_dataset('energies', data=energies, compression=self.compression)
        
        self.logger.debug(f"Written data arrays: configurations {configurations.shape}")
    
    def _write_split_indices(self, f: h5py.File, split: DatasetSplit) -> None:
        """Write train/val/test split indices."""
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
    
    def _write_metadata(self, f: h5py.File, metadata: DatasetMetadata) -> None:
        """Write dataset metadata."""
        meta_group = f.create_group('metadata')
        
        # Convert metadata to attributes
        meta_dict = metadata.to_dict()
        for key, value in meta_dict.items():
            if isinstance(value, (tuple, list)):
                meta_group.attrs[key] = np.array(value)
            else:
                meta_group.attrs[key] = value
        
        self.logger.debug("Written metadata")
    
    def _write_statistics(self,
                         f: h5py.File,
                         configurations: np.ndarray,
                         temperatures: np.ndarray,
                         magnetizations: np.ndarray,
                         energies: np.ndarray) -> None:
        """Write dataset statistics."""
        stats_group = f.create_group('statistics')
        
        # Configuration statistics
        config_stats = stats_group.create_group('configurations')
        config_stats.attrs['mean'] = np.mean(configurations)
        config_stats.attrs['std'] = np.std(configurations)
        config_stats.attrs['min'] = np.min(configurations)
        config_stats.attrs['max'] = np.max(configurations)
        
        # Temperature statistics
        temp_stats = stats_group.create_group('temperatures')
        temp_stats.attrs['mean'] = np.mean(temperatures)
        temp_stats.attrs['std'] = np.std(temperatures)
        temp_stats.attrs['min'] = np.min(temperatures)
        temp_stats.attrs['max'] = np.max(temperatures)
        temp_stats.attrs['unique_count'] = len(np.unique(temperatures))
        
        # Magnetization statistics
        mag_stats = stats_group.create_group('magnetizations')
        mag_stats.attrs['mean'] = np.mean(magnetizations)
        mag_stats.attrs['std'] = np.std(magnetizations)
        mag_stats.attrs['min'] = np.min(magnetizations)
        mag_stats.attrs['max'] = np.max(magnetizations)
        
        # Energy statistics
        energy_stats = stats_group.create_group('energies')
        energy_stats.attrs['mean'] = np.mean(energies)
        energy_stats.attrs['std'] = np.std(energies)
        energy_stats.attrs['min'] = np.min(energies)
        energy_stats.attrs['max'] = np.max(energies)
        
        self.logger.debug("Written statistics")


class IsingDataset(Dataset):
    """PyTorch Dataset for Ising model configurations."""
    
    def __init__(self, 
                 hdf5_path: str,
                 split: str = 'train',
                 transform: Optional[callable] = None,
                 load_physics: bool = False):
        """
        Initialize Ising dataset.
        
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
        
        self.logger.info(f"Initialized {split} dataset: {len(self)} samples")
    
    def _load_dataset_info(self) -> None:
        """Load dataset information and split indices."""
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
        Get dataset item.
        
        Args:
            idx: Sample index
            
        Returns:
            Configuration tensor, or tuple with physics data if load_physics=True
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        # Get actual data index
        data_idx = self.indices[idx]
        
        with h5py.File(self.hdf5_path, 'r') as f:
            # Load configuration
            config = f['configurations'][data_idx]
            config_tensor = torch.from_numpy(config).float()
            
            # Add channel dimension if needed (for CNN)
            if len(config_tensor.shape) == 2:
                config_tensor = config_tensor.unsqueeze(0)  # Add channel dimension
            
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
        """Get dataset statistics."""
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
                'max': float(np.max(configs))
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


class DataPreprocessor:
    """Main class for data preprocessing and storage pipeline."""
    
    def __init__(self, config: PrometheusConfig):
        """
        Initialize data preprocessor.
        
        Args:
            config: PrometheusConfig with preprocessing parameters
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.normalizer = DataNormalizer(method='sigmoid')  # Default normalization
        
        # Create output directory
        self.data_dir = Path(config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def process_dataset(self,
                       result: TemperatureSweepResult,
                       output_path: Optional[str] = None,
                       normalization_method: str = 'sigmoid',
                       split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> str:
        """
        Process complete dataset from TemperatureSweepResult to HDF5.
        
        Args:
            result: TemperatureSweepResult from data generation
            output_path: Output HDF5 file path
            normalization_method: Normalization method for configurations
            split_ratios: Train/val/test split ratios
            
        Returns:
            Path to created HDF5 file
        """
        self.logger.info("Starting dataset preprocessing")
        
        with LoggingContext("Dataset Preprocessing"):
            # Generate output path if not provided
            if output_path is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = str(self.data_dir / f"ising_processed_{timestamp}.h5")
            
            # Extract and flatten data
            configurations, temperatures, magnetizations, energies = self._extract_data(result)
            
            # Normalize configurations
            self.normalizer = DataNormalizer(method=normalization_method)
            normalized_configs = self.normalizer.normalize(configurations)
            
            # Create stratified splits
            sampler = StratifiedTemperatureSampler(
                temperatures=temperatures,
                split_ratios=split_ratios,
                random_state=self.config.seed
            )
            split = sampler.create_splits()
            
            # Create metadata
            metadata = DatasetMetadata(
                lattice_size=self.config.ising.lattice_size,
                n_configurations=len(configurations),
                n_temperatures=len(result.temperatures),
                temperature_range=self.config.ising.temperature_range,
                critical_temperature=self.config.ising.critical_temp,
                normalization_method=normalization_method,
                split_info=split,
                creation_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                config_hash=result.metadata.get('config_hash', 'unknown'),
                data_shape=normalized_configs.shape
            )
            
            # Write to HDF5
            writer = HDF5DatasetWriter(output_path)
            writer.write_dataset(
                configurations=normalized_configs,
                temperatures=temperatures,
                magnetizations=magnetizations,
                energies=energies,
                split=split,
                metadata=metadata
            )
            
            self.logger.info(f"Dataset preprocessing complete: {output_path}")
            
            return output_path
    
    def _extract_data(self, result: TemperatureSweepResult) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract and flatten data from TemperatureSweepResult."""
        self.logger.info("Extracting and flattening configuration data")
        
        all_configs = []
        all_temps = []
        all_mags = []
        all_energies = []
        
        for temp_idx, temp_configs in enumerate(result.configurations_per_temp):
            temperature = result.temperatures[temp_idx]
            
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
        
        self.logger.info(f"Extracted {len(configurations)} configurations")
        self.logger.info(f"Configuration shape: {configurations.shape}")
        
        return configurations, temperatures, magnetizations, energies
    
    def create_dataloaders(self,
                          hdf5_path: str,
                          batch_size: Optional[int] = None,
                          num_workers: int = 4,
                          pin_memory: bool = True,
                          load_physics: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for train/val/test splits.
        
        Args:
            hdf5_path: Path to processed HDF5 dataset
            batch_size: Batch size (default: from config)
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for GPU transfer
            load_physics: Whether to load physical quantities
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if batch_size is None:
            batch_size = self.config.training.batch_size
        
        self.logger.info(f"Creating DataLoaders with batch_size={batch_size}")
        
        # Create datasets
        train_dataset = IsingDataset(hdf5_path, split='train', load_physics=load_physics)
        val_dataset = IsingDataset(hdf5_path, split='val', load_physics=load_physics)
        test_dataset = IsingDataset(hdf5_path, split='test', load_physics=load_physics)
        
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
        
        self.logger.info(f"Created DataLoaders: train={len(train_loader)} batches, "
                        f"val={len(val_loader)} batches, test={len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader
    
    def load_dataset_info(self, hdf5_path: str) -> Dict[str, Any]:
        """
        Load dataset information from HDF5 file.
        
        Args:
            hdf5_path: Path to HDF5 dataset
            
        Returns:
            Dictionary with dataset information
        """
        with h5py.File(hdf5_path, 'r') as f:
            info = {}
            
            # Basic info
            info['n_configurations'] = f['configurations'].shape[0]
            info['configuration_shape'] = f['configurations'].shape[1:]
            
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