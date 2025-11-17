"""
Data module for the Prometheus project.

This module provides utilities for:
- Ising model simulation and Monte Carlo methods
- Data generation and orchestration
- Data preprocessing and storage
- PyTorch dataset classes and data loaders
"""

from .ising_simulator import IsingSimulator, SpinConfiguration
from .enhanced_monte_carlo import (
    EnhancedMonteCarloSimulator,
    SpinConfiguration3D,
    create_2d_simulator,
    create_3d_simulator
)
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
from .data_generator_3d import (
    DataGenerator3D,
    DataGenerationConfig3D,
    Dataset3DResult,
    SystemSizeResult3D,
    GenerationProgress3D,
    generate_3d_ising_dataset,
    create_default_3d_config
)
from .equilibration_3d import (
    Enhanced3DEquilibrationProtocol,
    Enhanced3DEquilibrationResult,
    EquilibrationQualityMetrics,
    create_enhanced_3d_equilibration_protocol
)
from .preprocessing import (
    DataPreprocessor,
    DataNormalizer,
    IsingDataset,
    HDF5DatasetWriter,
    DatasetMetadata,
    DatasetSplit
)
from .preprocessing_3d import (
    DataPreprocessor3D,
    DataNormalizer3D,
    IsingDataset3D,
    HDF5DatasetWriter3D,
    DatasetMetadata3D
)
from .data_loader_utils import (
    AdaptiveDataLoader,
    DatasetFactory,
    ConfigurationNormalizer,
    get_data_info_summary
)

__all__ = [
    # Simulation
    'IsingSimulator',
    'SpinConfiguration',
    'EnhancedMonteCarloSimulator',
    'SpinConfiguration3D',
    'create_2d_simulator',
    'create_3d_simulator',
    
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
    
    # 3D Data generation
    'DataGenerator3D',
    'DataGenerationConfig3D',
    'Dataset3DResult',
    'SystemSizeResult3D',
    'GenerationProgress3D',
    'generate_3d_ising_dataset',
    'create_default_3d_config',
    
    # 3D Equilibration
    'Enhanced3DEquilibrationProtocol',
    'Enhanced3DEquilibrationResult',
    'EquilibrationQualityMetrics',
    'create_enhanced_3d_equilibration_protocol',
    
    # Preprocessing
    'DataPreprocessor',
    'DataNormalizer',
    'IsingDataset',
    'HDF5DatasetWriter',
    'DatasetMetadata',
    'DatasetSplit',
    
    # 3D Preprocessing
    'DataPreprocessor3D',
    'DataNormalizer3D',
    'IsingDataset3D',
    'HDF5DatasetWriter3D',
    'DatasetMetadata3D',
    
    # Adaptive data loading
    'AdaptiveDataLoader',
    'DatasetFactory',
    'ConfigurationNormalizer',
    'get_data_info_summary'
]