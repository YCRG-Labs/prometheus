"""
Template for creating custom model variant plugins.

This module provides a template that researchers can use as a starting point
for implementing custom Ising model variants. Copy this file and modify the
CustomModelPlugin class to implement your specific model.

Usage:
    1. Copy this file to a new file (e.g., my_custom_model.py)
    2. Rename CustomModelPlugin to your model name
    3. Implement the three required methods
    4. Register your plugin with the PluginRegistry
    5. Use it in the discovery pipeline

Example:
    >>> from src.research.plugin_registry import get_global_plugin_registry
    >>> from my_custom_model import MyCustomModel
    >>> 
    >>> registry = get_global_plugin_registry()
    >>> registry.register_plugin(MyCustomModel, 'my_custom_model')
"""

import numpy as np
from typing import Dict, Any, Tuple
from .base_types import ModelVariantConfig
from .base_plugin import ModelVariantPlugin, SpinFlipProposal


class CustomModelPlugin(ModelVariantPlugin):
    """Template for custom Ising model variant.
    
    Replace this docstring with a description of your model, including:
    - The energy function
    - The interaction type
    - Any special properties or behaviors
    - References to relevant papers if applicable
    
    Example:
        Energy: E = -Σ_<i,j> J_ij s_i s_j - h Σ_i s_i
        where J_ij follows some custom rule
    """
    
    def __init__(self, config: ModelVariantConfig):
        """Initialize custom model.
        
        Args:
            config: Model variant configuration containing:
                - dimensions: 2 or 3
                - lattice_geometry: Lattice type
                - interaction_params: Dict with model-specific parameters
                - disorder_strength: Disorder strength (if applicable)
                - external_field: External magnetic field
        """
        self.config = config
        self.dimensions = config.dimensions
        self.h = config.external_field
        
        # Extract model-specific parameters from config
        # Example: self.custom_param = config.interaction_params.get('custom_param', 1.0)
        
        # Initialize any precomputed quantities
        # Example: self._precompute_couplings()
        
        # Set theoretical properties if known
        # self.theoretical_tc = None  # Critical temperature
        # self.theoretical_exponents = None  # Dict of exponents
    
    def compute_energy(self, configuration: np.ndarray) -> float:
        """Compute total energy of the system.
        
        This is the most important method - it defines your model's physics.
        
        Args:
            configuration: Spin configuration array of shape:
                          - (L, L) for 2D systems
                          - (L, L, L) for 3D systems
                          where L is the lattice size
                          
        Returns:
            Total energy of the configuration (float)
            
        Implementation notes:
            - Use periodic boundary conditions: configuration[(i+1) % L, j]
            - Be careful about double-counting bonds
            - Include external field contribution: -h * Σ_i s_i
            - Optimize for performance if possible (vectorization)
        """
        energy = 0.0
        
        # TODO: Implement your energy calculation here
        # Example for nearest-neighbor:
        # if self.dimensions == 2:
        #     L = configuration.shape[0]
        #     # Horizontal bonds
        #     energy -= np.sum(configuration[:, :-1] * configuration[:, 1:])
        #     # Vertical bonds
        #     energy -= np.sum(configuration[:-1, :] * configuration[1:, :])
        
        # External field contribution
        energy -= self.h * np.sum(configuration)
        
        return energy
    
    def propose_spin_flip(self, configuration: np.ndarray,
                         site: Tuple[int, ...]) -> SpinFlipProposal:
        """Propose a Monte Carlo spin flip move.
        
        This method computes the energy change for flipping a spin at the
        given site. It's called many times during simulation, so optimize
        for performance.
        
        Args:
            configuration: Current spin configuration
            site: Coordinates of site to flip, e.g., (i, j) for 2D
            
        Returns:
            SpinFlipProposal with:
                - site: The site coordinates (same as input)
                - energy_change: ΔE if flip is accepted
                - acceptance_probability: Set to 1.0 (temperature applied by simulator)
                
        Implementation notes:
            - Only compute LOCAL energy change (don't recalculate full energy)
            - For nearest-neighbor: ΔE = 2 * s_i * (J * Σ_neighbors + h)
            - For long-range: ΔE = 2 * s_i * (Σ_j J_ij * s_j + h)
            - Use periodic boundaries: neighbor_i = (i + 1) % L
        """
        spin = configuration[site]
        
        # TODO: Compute local field from interactions
        # Example for nearest-neighbor 2D:
        # if self.dimensions == 2:
        #     i, j = site
        #     L = configuration.shape[0]
        #     field = 0.0
        #     field += configuration[(i+1) % L, j]
        #     field += configuration[(i-1) % L, j]
        #     field += configuration[i, (j+1) % L]
        #     field += configuration[i, (j-1) % L]
        
        field = 0.0  # Replace with your calculation
        
        # Energy change for flipping spin
        delta_e = 2 * spin * (field + self.h)
        
        return SpinFlipProposal(site, delta_e, 1.0)
    
    def get_theoretical_properties(self) -> Dict[str, Any]:
        """Return theoretical properties if known.
        
        Returns:
            Dictionary with keys:
                - 'tc': Theoretical critical temperature (float or None)
                - 'exponents': Dict of theoretical exponents (or None)
                  Example: {'beta': 0.125, 'nu': 1.0, 'gamma': 1.75}
                - 'universality_class': Name of universality class (str or None)
                  Example: '2D_Ising', 'mean_field', 'custom_class'
                  
        Implementation notes:
            - Return None for unknown properties
            - If your model belongs to a known universality class, specify it
            - Theoretical values are used for validation and comparison
        """
        return {
            'tc': None,  # TODO: Set if known
            'exponents': None,  # TODO: Set if known
            'universality_class': 'custom'  # TODO: Set appropriate class
        }
    
    # Optional: Override these methods if needed
    
    def validate_configuration(self, configuration: np.ndarray) -> bool:
        """Validate that a configuration is physically valid.
        
        Default implementation checks spins are ±1. Override if you need
        different validation (e.g., for continuous spins, constrained systems).
        """
        return super().validate_configuration(configuration)
    
    def compute_magnetization(self, configuration: np.ndarray) -> float:
        """Compute magnetization (order parameter).
        
        Default implementation returns mean spin. Override if your model
        has a different order parameter (e.g., staggered magnetization,
        nematic order, etc.).
        """
        return super().compute_magnetization(configuration)


# Example: More complex custom model with additional features
class AdvancedCustomModelPlugin(ModelVariantPlugin):
    """Example of a more complex custom model with precomputation.
    
    This example shows how to:
    - Precompute interaction matrices for efficiency
    - Handle model-specific parameters
    - Implement caching for repeated calculations
    """
    
    def __init__(self, config: ModelVariantConfig):
        self.config = config
        self.dimensions = config.dimensions
        self.h = config.external_field
        
        # Extract custom parameters
        self.custom_param = config.interaction_params.get('custom_param', 1.0)
        
        # Precomputation cache
        self._interaction_matrix = None
        self._cached_lattice_size = None
    
    def _precompute_interactions(self, lattice_size: int) -> None:
        """Precompute interaction matrix for given lattice size."""
        if self._cached_lattice_size == lattice_size:
            return  # Already computed
        
        self._cached_lattice_size = lattice_size
        
        # TODO: Precompute your interaction matrix
        # Example: distance-dependent interactions
        # n_sites = lattice_size ** self.dimensions
        # self._interaction_matrix = np.zeros((n_sites, n_sites))
        # ... compute interactions ...
    
    def compute_energy(self, configuration: np.ndarray) -> float:
        lattice_size = configuration.shape[0]
        self._precompute_interactions(lattice_size)
        
        # Use precomputed interactions
        energy = 0.0
        # TODO: Implement using self._interaction_matrix
        
        energy -= self.h * np.sum(configuration)
        return energy
    
    def propose_spin_flip(self, configuration: np.ndarray,
                         site: Tuple[int, ...]) -> SpinFlipProposal:
        lattice_size = configuration.shape[0]
        self._precompute_interactions(lattice_size)
        
        spin = configuration[site]
        
        # Use precomputed interactions for efficient field calculation
        field = 0.0
        # TODO: Implement using self._interaction_matrix
        
        delta_e = 2 * spin * (field + self.h)
        return SpinFlipProposal(site, delta_e, 1.0)
    
    def get_theoretical_properties(self) -> Dict[str, Any]:
        return {
            'tc': None,
            'exponents': None,
            'universality_class': 'advanced_custom'
        }


# Quick reference for common patterns:

"""
PATTERN 1: Nearest-Neighbor Interactions (2D)
----------------------------------------------
def compute_energy(self, configuration):
    L = configuration.shape[0]
    energy = 0.0
    # Horizontal bonds
    energy -= self.J * np.sum(configuration[:, :-1] * configuration[:, 1:])
    # Vertical bonds
    energy -= self.J * np.sum(configuration[:-1, :] * configuration[1:, :])
    energy -= self.h * np.sum(configuration)
    return energy

def propose_spin_flip(self, configuration, site):
    i, j = site
    L = configuration.shape[0]
    spin = configuration[i, j]
    neighbor_sum = (configuration[(i+1)%L, j] + configuration[(i-1)%L, j] +
                   configuration[i, (j+1)%L] + configuration[i, (j-1)%L])
    delta_e = 2 * spin * (self.J * neighbor_sum + self.h)
    return SpinFlipProposal(site, delta_e, 1.0)


PATTERN 2: Long-Range Interactions
-----------------------------------
def _precompute_couplings(self, L):
    n_sites = L ** 2
    self._couplings = np.zeros((n_sites, n_sites))
    for i in range(n_sites):
        for j in range(i+1, n_sites):
            r = distance(i, j, L)  # Implement distance calculation
            coupling = self.J0 / (r ** self.alpha)
            self._couplings[i, j] = coupling
            self._couplings[j, i] = coupling

def compute_energy(self, configuration):
    L = configuration.shape[0]
    self._precompute_couplings(L)
    spins = configuration.flatten()
    energy = 0.0
    for i in range(len(spins)):
        for j in range(i+1, len(spins)):
            energy -= self._couplings[i, j] * spins[i] * spins[j]
    energy -= self.h * np.sum(spins)
    return energy


PATTERN 3: Random Disorder
---------------------------
def _generate_disorder(self, L):
    if self._cached_L == L:
        return
    self._cached_L = L
    np.random.seed(self.seed)
    # Generate random couplings
    self._couplings = np.random.normal(1.0, self.disorder_strength, (L, L, 4))

def compute_energy(self, configuration):
    L = configuration.shape[0]
    self._generate_disorder(L)
    energy = 0.0
    for i in range(L):
        for j in range(L):
            spin = configuration[i, j]
            energy -= self._couplings[i,j,0] * spin * configuration[i,(j+1)%L]
            energy -= self._couplings[i,j,1] * spin * configuration[(i+1)%L,j]
    energy /= 2  # Avoid double counting
    energy -= self.h * np.sum(configuration)
    return energy
"""
