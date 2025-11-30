"""
Built-in model variant plugins.

This module provides implementations of common Ising model variants including
standard nearest-neighbor, long-range interactions, quenched disorder, and
frustrated geometries.
"""

import numpy as np
from typing import Dict, Any, Tuple
from .base_types import ModelVariantConfig
from .base_plugin import ModelVariantPlugin, SpinFlipProposal


class StandardIsingModel(ModelVariantPlugin):
    """Standard nearest-neighbor Ising model.
    
    Energy: E = -J Σ_<i,j> s_i s_j - h Σ_i s_i
    where <i,j> denotes nearest-neighbor pairs.
    """
    
    def __init__(self, config: ModelVariantConfig):
        """Initialize standard Ising model.
        
        Args:
            config: Model variant configuration
        """
        self.config = config
        self.dimensions = config.dimensions
        self.J = config.interaction_params.get('J', 1.0)
        self.h = config.external_field
        
        # Theoretical properties for standard Ising
        if config.dimensions == 2:
            self.theoretical_tc = 2.269  # 2/ln(1+sqrt(2))
            self.theoretical_exponents = {
                'beta': 0.125,
                'nu': 1.0,
                'gamma': 1.75,
                'alpha': 0.0
            }
        elif config.dimensions == 3:
            self.theoretical_tc = 4.511
            self.theoretical_exponents = {
                'beta': 0.326,
                'nu': 0.630,
                'gamma': 1.237,
                'alpha': 0.110
            }
    
    def compute_energy(self, configuration: np.ndarray) -> float:
        """Compute total energy of the configuration."""
        energy = 0.0
        
        # Nearest-neighbor interaction energy
        if self.dimensions == 2:
            # Horizontal bonds
            energy -= self.J * np.sum(configuration[:, :-1] * configuration[:, 1:])
            # Vertical bonds
            energy -= self.J * np.sum(configuration[:-1, :] * configuration[1:, :])
        elif self.dimensions == 3:
            # X-direction bonds
            energy -= self.J * np.sum(configuration[:, :, :-1] * configuration[:, :, 1:])
            # Y-direction bonds
            energy -= self.J * np.sum(configuration[:, :-1, :] * configuration[:, 1:, :])
            # Z-direction bonds
            energy -= self.J * np.sum(configuration[:-1, :, :] * configuration[1:, :, :])
        
        # External field energy
        energy -= self.h * np.sum(configuration)
        
        return energy
    
    def propose_spin_flip(self, configuration: np.ndarray,
                         site: Tuple[int, ...]) -> SpinFlipProposal:
        """Propose a spin flip at the given site."""
        # Compute energy change for flipping spin at site
        spin = configuration[site]
        
        # Sum of neighbor spins
        neighbor_sum = 0.0
        if self.dimensions == 2:
            i, j = site
            L = configuration.shape[0]
            # Periodic boundary conditions
            neighbor_sum += configuration[(i+1) % L, j]
            neighbor_sum += configuration[(i-1) % L, j]
            neighbor_sum += configuration[i, (j+1) % L]
            neighbor_sum += configuration[i, (j-1) % L]
        elif self.dimensions == 3:
            i, j, k = site
            L = configuration.shape[0]
            neighbor_sum += configuration[(i+1) % L, j, k]
            neighbor_sum += configuration[(i-1) % L, j, k]
            neighbor_sum += configuration[i, (j+1) % L, k]
            neighbor_sum += configuration[i, (j-1) % L, k]
            neighbor_sum += configuration[i, j, (k+1) % L]
            neighbor_sum += configuration[i, j, (k-1) % L]
        
        # Energy change: ΔE = 2 * spin * (J * neighbor_sum + h)
        delta_e = 2 * spin * (self.J * neighbor_sum + self.h)
        
        # Acceptance probability will be computed by simulator with temperature
        return SpinFlipProposal(site, delta_e, 1.0)
    
    def get_theoretical_properties(self) -> Dict[str, Any]:
        """Return theoretical properties."""
        return {
            'tc': self.theoretical_tc if hasattr(self, 'theoretical_tc') else None,
            'exponents': self.theoretical_exponents if hasattr(self, 'theoretical_exponents') else None,
            'universality_class': f'{self.dimensions}D_Ising'
        }


class LongRangeIsingModel(ModelVariantPlugin):
    """Long-range Ising model with power-law interactions.
    
    Energy: E = -Σ_{i<j} J(r_ij) s_i s_j
    where J(r) = J_0 / r^α
    """
    
    def __init__(self, config: ModelVariantConfig):
        """Initialize long-range Ising model.
        
        Args:
            config: Model variant configuration with 'alpha' parameter
        """
        self.config = config
        self.dimensions = config.dimensions
        self.alpha = config.interaction_params.get('alpha', 2.5)
        self.J0 = config.interaction_params.get('J0', 1.0)
        self.h = config.external_field
        
        # Precompute coupling matrix for efficiency
        self._couplings = None
        self._lattice_size = None
    
    def _precompute_couplings(self, lattice_size: int) -> None:
        """Precompute coupling matrix for given lattice size."""
        if self._lattice_size == lattice_size and self._couplings is not None:
            return
        
        self._lattice_size = lattice_size
        
        if self.dimensions == 2:
            # Create distance matrix
            coords = np.arange(lattice_size)
            x, y = np.meshgrid(coords, coords, indexing='ij')
            
            # Compute all pairwise distances (with periodic boundaries)
            n_sites = lattice_size ** 2
            self._couplings = np.zeros((n_sites, n_sites))
            
            for i in range(n_sites):
                ix, iy = i // lattice_size, i % lattice_size
                for j in range(i+1, n_sites):
                    jx, jy = j // lattice_size, j % lattice_size
                    
                    # Minimum image convention
                    dx = min(abs(ix - jx), lattice_size - abs(ix - jx))
                    dy = min(abs(iy - jy), lattice_size - abs(iy - jy))
                    r = np.sqrt(dx**2 + dy**2)
                    
                    if r > 0:
                        coupling = self.J0 / (r ** self.alpha)
                        self._couplings[i, j] = coupling
                        self._couplings[j, i] = coupling
        
        # For 3D, similar but more expensive - simplified for now
    
    def compute_energy(self, configuration: np.ndarray) -> float:
        """Compute total energy with long-range interactions."""
        lattice_size = configuration.shape[0]
        self._precompute_couplings(lattice_size)
        
        # Flatten configuration
        spins = configuration.flatten()
        
        # Energy = -Σ_{i<j} J_ij s_i s_j
        energy = 0.0
        n_sites = len(spins)
        for i in range(n_sites):
            for j in range(i+1, n_sites):
                energy -= self._couplings[i, j] * spins[i] * spins[j]
        
        # External field
        energy -= self.h * np.sum(spins)
        
        return energy
    
    def propose_spin_flip(self, configuration: np.ndarray,
                         site: Tuple[int, ...]) -> SpinFlipProposal:
        """Propose spin flip with long-range interactions."""
        lattice_size = configuration.shape[0]
        self._precompute_couplings(lattice_size)
        
        # Convert site to linear index
        if self.dimensions == 2:
            i, j = site
            site_idx = i * lattice_size + j
        else:
            raise NotImplementedError("3D long-range not fully implemented")
        
        spins = configuration.flatten()
        spin = spins[site_idx]
        
        # Compute field from all other spins
        field = 0.0
        for j in range(len(spins)):
            if j != site_idx:
                field += self._couplings[site_idx, j] * spins[j]
        
        # Energy change
        delta_e = 2 * spin * (field + self.h)
        
        return SpinFlipProposal(site, delta_e, 1.0)
    
    def get_theoretical_properties(self) -> Dict[str, Any]:
        """Return theoretical properties."""
        # For α > 2, belongs to short-range universality class
        # For α < 2, mean-field behavior
        # For α = 2, marginal case
        
        if self.alpha > 2:
            universality_class = f'{self.dimensions}D_Ising'
        elif self.alpha < 2:
            universality_class = 'mean_field'
        else:
            universality_class = 'marginal'
        
        return {
            'tc': None,  # Depends on α
            'exponents': None,
            'universality_class': universality_class
        }


class QuenchedDisorderModel(ModelVariantPlugin):
    """Ising model with quenched disorder (random couplings).
    
    Energy: E = -Σ_<i,j> J_ij s_i s_j
    where J_ij are random couplings drawn from a distribution.
    """
    
    def __init__(self, config: ModelVariantConfig, seed: int = None):
        """Initialize quenched disorder model.
        
        Args:
            config: Model variant configuration
            seed: Random seed for disorder realization
        """
        self.config = config
        self.dimensions = config.dimensions
        self.disorder_strength = config.disorder_strength
        self.h = config.external_field
        
        # Random couplings (quenched - fixed for all simulations)
        self.seed = seed
        self._couplings = None
        self._lattice_size = None
    
    def _generate_disorder(self, lattice_size: int) -> None:
        """Generate random couplings for the lattice."""
        if self._lattice_size == lattice_size and self._couplings is not None:
            return
        
        self._lattice_size = lattice_size
        
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Generate random couplings: J_ij ~ N(1, disorder_strength)
        if self.dimensions == 2:
            shape = (lattice_size, lattice_size, 4)  # 4 nearest neighbors
        else:
            shape = (lattice_size, lattice_size, lattice_size, 6)  # 6 nearest neighbors
        
        self._couplings = np.random.normal(
            loc=1.0,
            scale=self.disorder_strength,
            size=shape
        )
    
    def compute_energy(self, configuration: np.ndarray) -> float:
        """Compute energy with disordered couplings."""
        lattice_size = configuration.shape[0]
        self._generate_disorder(lattice_size)
        
        energy = 0.0
        
        if self.dimensions == 2:
            L = lattice_size
            for i in range(L):
                for j in range(L):
                    spin = configuration[i, j]
                    # Right neighbor
                    energy -= self._couplings[i, j, 0] * spin * configuration[i, (j+1) % L]
                    # Down neighbor
                    energy -= self._couplings[i, j, 1] * spin * configuration[(i+1) % L, j]
            # Divide by 2 to avoid double counting
            energy /= 2
        
        # External field
        energy -= self.h * np.sum(configuration)
        
        return energy
    
    def propose_spin_flip(self, configuration: np.ndarray,
                         site: Tuple[int, ...]) -> SpinFlipProposal:
        """Propose spin flip with disordered couplings."""
        lattice_size = configuration.shape[0]
        self._generate_disorder(lattice_size)
        
        spin = configuration[site]
        
        # Compute local field from neighbors
        field = 0.0
        if self.dimensions == 2:
            i, j = site
            L = lattice_size
            # Sum over neighbors with their couplings
            field += self._couplings[i, j, 0] * configuration[i, (j+1) % L]
            field += self._couplings[i, (j-1) % L, 0] * configuration[i, (j-1) % L]
            field += self._couplings[i, j, 1] * configuration[(i+1) % L, j]
            field += self._couplings[(i-1) % L, j, 1] * configuration[(i-1) % L, j]
        
        delta_e = 2 * spin * (field + self.h)
        
        return SpinFlipProposal(site, delta_e, 1.0)
    
    def get_theoretical_properties(self) -> Dict[str, Any]:
        """Return theoretical properties."""
        return {
            'tc': None,  # Disorder changes Tc
            'exponents': None,  # May have different exponents
            'universality_class': f'{self.dimensions}D_random_Ising'
        }


class FrustratedGeometryModel(ModelVariantPlugin):
    """Ising model on frustrated geometries (triangular, honeycomb, kagome).
    
    These geometries can exhibit geometric frustration leading to novel
    phase transition behavior.
    """
    
    def __init__(self, config: ModelVariantConfig):
        """Initialize frustrated geometry model.
        
        Args:
            config: Model variant configuration
        """
        self.config = config
        self.dimensions = config.dimensions
        self.geometry = config.lattice_geometry
        self.J = config.interaction_params.get('J', 1.0)
        self.h = config.external_field
        
        if self.dimensions != 2:
            raise NotImplementedError("Frustrated geometries only implemented for 2D")
        
        if self.geometry not in ['triangular', 'honeycomb', 'kagome']:
            raise ValueError(f"Unsupported geometry: {self.geometry}")
    
    def _get_neighbors(self, site: Tuple[int, int], lattice_size: int) -> list:
        """Get neighbor sites based on geometry."""
        i, j = site
        L = lattice_size
        
        if self.geometry == 'triangular':
            # Triangular lattice: 6 neighbors
            neighbors = [
                ((i+1) % L, j),
                ((i-1) % L, j),
                (i, (j+1) % L),
                (i, (j-1) % L),
                ((i+1) % L, (j+1) % L),
                ((i-1) % L, (j-1) % L),
            ]
        elif self.geometry == 'honeycomb':
            # Honeycomb: 3 neighbors (bipartite lattice)
            # Simplified: alternate between A and B sublattices
            if (i + j) % 2 == 0:  # A sublattice
                neighbors = [
                    ((i+1) % L, j),
                    (i, (j+1) % L),
                    ((i-1) % L, (j+1) % L),
                ]
            else:  # B sublattice
                neighbors = [
                    ((i-1) % L, j),
                    (i, (j-1) % L),
                    ((i+1) % L, (j-1) % L),
                ]
        elif self.geometry == 'kagome':
            # Kagome: 4 neighbors (corner-sharing triangles)
            neighbors = [
                ((i+1) % L, j),
                ((i-1) % L, j),
                (i, (j+1) % L),
                (i, (j-1) % L),
            ]
        else:
            neighbors = []
        
        return neighbors
    
    def compute_energy(self, configuration: np.ndarray) -> float:
        """Compute energy for frustrated geometry."""
        lattice_size = configuration.shape[0]
        energy = 0.0
        
        # Sum over all bonds (avoiding double counting)
        for i in range(lattice_size):
            for j in range(lattice_size):
                spin = configuration[i, j]
                neighbors = self._get_neighbors((i, j), lattice_size)
                
                for ni, nj in neighbors:
                    if (ni > i) or (ni == i and nj > j):  # Avoid double counting
                        energy -= self.J * spin * configuration[ni, nj]
        
        # External field
        energy -= self.h * np.sum(configuration)
        
        return energy
    
    def propose_spin_flip(self, configuration: np.ndarray,
                         site: Tuple[int, ...]) -> SpinFlipProposal:
        """Propose spin flip on frustrated geometry."""
        lattice_size = configuration.shape[0]
        spin = configuration[site]
        
        # Sum over neighbors
        neighbors = self._get_neighbors(site, lattice_size)
        neighbor_sum = sum(configuration[ni, nj] for ni, nj in neighbors)
        
        delta_e = 2 * spin * (self.J * neighbor_sum + self.h)
        
        return SpinFlipProposal(site, delta_e, 1.0)
    
    def get_theoretical_properties(self) -> Dict[str, Any]:
        """Return theoretical properties."""
        return {
            'tc': None,  # Geometry-dependent
            'exponents': None,
            'universality_class': f'{self.geometry}_Ising'
        }

