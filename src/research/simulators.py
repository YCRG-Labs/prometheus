"""
Monte Carlo simulators for model variants.

This module provides simulator implementations for different types of Ising
model variants, including standard nearest-neighbor, long-range, and disordered
models.
"""

import numpy as np
from typing import Dict, Optional, Any
from .base_types import ModelVariantConfig
from .base_plugin import BaseSimulator, ModelVariantPlugin


class StandardIsingSimulator(BaseSimulator):
    """Standard Metropolis Monte Carlo simulator for Ising models.
    
    This simulator works with any ModelVariantPlugin and implements the
    standard Metropolis algorithm for Monte Carlo sampling.
    """
    
    def _initialize_configuration(self) -> np.ndarray:
        """Initialize random spin configuration."""
        if hasattr(self.model, 'dimensions'):
            dims = self.model.dimensions
        else:
            dims = 2  # Default to 2D
        
        if dims == 2:
            shape = (self.lattice_size, self.lattice_size)
        elif dims == 3:
            shape = (self.lattice_size, self.lattice_size, self.lattice_size)
        else:
            raise ValueError(f"Unsupported dimensions: {dims}")
        
        # Random ±1 spins
        return 2 * np.random.randint(0, 2, size=shape) - 1
    
    def equilibrate(self, n_steps: int) -> None:
        """Equilibrate the system using Monte Carlo sweeps."""
        for _ in range(n_steps):
            self.monte_carlo_step()
    
    def measure(self, n_samples: int, n_steps_between: int) -> Dict[str, np.ndarray]:
        """Perform measurements after equilibration."""
        configurations = []
        magnetizations = []
        energies = []
        
        for _ in range(n_samples):
            # Perform MC steps between measurements
            for _ in range(n_steps_between):
                self.monte_carlo_step()
            
            # Measure observables
            configurations.append(self.configuration.copy())
            magnetizations.append(self.model.compute_magnetization(self.configuration))
            energies.append(self.model.compute_energy(self.configuration))
        
        return {
            'configurations': np.array(configurations),
            'magnetizations': np.array(magnetizations),
            'energies': np.array(energies)
        }


def create_simulator_for_variant(config: ModelVariantConfig, lattice_size: int,
                                 temperature: float, seed: Optional[int] = None,
                                 **kwargs) -> BaseSimulator:
    """Factory function to create appropriate simulator for a model variant.
    
    Args:
        config: Model variant configuration
        lattice_size: Linear size of the lattice
        temperature: Simulation temperature
        seed: Random seed for reproducibility
        **kwargs: Additional simulator arguments
        
    Returns:
        BaseSimulator instance configured for the variant
        
    Raises:
        NotImplementedError: If no simulator available for variant type
    """
    # For now, we'll create a placeholder model plugin
    # In subtask 2.2, we'll implement specific model plugins
    
    # Import model plugins (will be implemented in 2.2)
    from . import model_plugins
    
    # Create appropriate model plugin based on config
    if config.interaction_type == 'nearest_neighbor' and config.disorder_type is None:
        model = model_plugins.StandardIsingModel(config)
    elif config.interaction_type == 'long_range':
        model = model_plugins.LongRangeIsingModel(config)
    elif config.disorder_type == 'quenched':
        model = model_plugins.QuenchedDisorderModel(config)
    elif config.lattice_geometry in ['triangular', 'honeycomb', 'kagome']:
        model = model_plugins.FrustratedGeometryModel(config)
    else:
        raise NotImplementedError(
            f"No simulator available for interaction_type='{config.interaction_type}', "
            f"disorder_type='{config.disorder_type}', "
            f"lattice_geometry='{config.lattice_geometry}'"
        )
    
    # Create simulator with the model
    return StandardIsingSimulator(model, lattice_size, temperature, seed)

