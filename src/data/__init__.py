"""
Data module for the Prometheus project.

This module provides utilities for:
- Ising model simulation and Monte Carlo methods
- Data generation and orchestration
- Data preprocessing and storage
- PyTorch dataset classes and data loaders
"""

from .ising_simulator import IsingSimulator, SpinConfiguration
from .equilibration import (
    EquilibrationProtocol,
    MeasurementProtocol, 
    TemperatureSweepProtocol,
    TemperatureSweepResult,
    create_default_protocols,
    run_standard_temperature_sweep
)
from .data_generator import (
    DataGenerator,
    GenerationProgress,
    ValidationResult
)
from .preprocessing import (
    DataPreprocessor,
    DataNormalizer,
    IsingDataset,
    HDF5DatasetWriter,
    DatasetMetadata,
    DatasetSplit
)

__all__ = [
    # Simulation
    'IsingSimulator',
    'SpinConfiguration',
    
    # Equilibration and measurement
    'EquilibrationProtocol',
    'MeasurementProtocol',
    'TemperatureSweepProtocol', 
    'TemperatureSweepResult',
    'create_default_protocols',
    'run_standard_temperature_sweep',
    
    # Data generation
    'DataGenerator',
    'GenerationProgress',
    'ValidationResult',
    
    # Preprocessing
    'DataPreprocessor',
    'DataNormalizer',
    'IsingDataset',
    'HDF5DatasetWriter',
    'DatasetMetadata',
    'DatasetSplit'
]