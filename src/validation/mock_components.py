"""
Mock components for validation pipeline testing.

This module provides mock implementations of components that may not
be available during testing.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class MockLatentRepresentation:
    """Mock LatentRepresentation for testing."""
    z1: np.ndarray
    z2: np.ndarray
    temperatures: np.ndarray
    magnetizations: np.ndarray
    energies: np.ndarray
    reconstruction_errors: np.ndarray
    sample_indices: np.ndarray
    
    @property
    def n_samples(self) -> int:
        return len(self.temperatures)
    
    @property
    def latent_coords(self) -> np.ndarray:
        return np.column_stack([self.z1, self.z2])


class MockEnhancedMonteCarloSimulator:
    """Mock Enhanced Monte Carlo Simulator for testing."""
    
    def __init__(self, lattice_size, model_type='ising', enhanced_equilibration=True):
        self.lattice_size = lattice_size
        self.model_type = model_type
        self.enhanced_equilibration = enhanced_equilibration
    
    def generate_equilibrated_configurations(self, temperature, n_configurations, 
                                           equilibration_steps=50000, measurement_interval=100):
        """Generate mock equilibrated configurations."""
        
        # Create mock configurations
        if len(self.lattice_size) == 2:
            configs = [np.random.choice([-1, 1], self.lattice_size) for _ in range(n_configurations)]
        else:  # 3D
            configs = [np.random.choice([-1, 1], self.lattice_size) for _ in range(n_configurations)]
        
        # Create mock magnetizations and energies
        mags = [np.mean(config) for config in configs]
        energies = [np.random.normal(-2.0, 0.5) for _ in range(n_configurations)]
        
        return configs, mags, energies


# MockVAECriticalExponentAnalyzer has been removed as part of task 13.1
# Use the real VAECriticalExponentAnalyzer from src.analysis.vae_based_critical_exponent_analyzer instead


def get_mock_logger(name):
    """Get mock logger for testing."""
    import logging
    return logging.getLogger(name)