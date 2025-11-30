"""
Abstract base classes for model variant plugins and simulators.

This module defines the plugin architecture that allows researchers to easily
add new Ising model variants without modifying core system code.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np


class SpinFlipProposal:
    """Proposal for a Monte Carlo spin flip move.
    
    Attributes:
        site: Coordinates of the site to flip
        energy_change: Change in energy if flip is accepted
        acceptance_probability: Probability of accepting the flip
    """
    
    def __init__(self, site: Tuple[int, ...], energy_change: float,
                 acceptance_probability: float):
        self.site = site
        self.energy_change = energy_change
        self.acceptance_probability = acceptance_probability


class ModelVariantPlugin(ABC):
    """Abstract base class for custom Ising model variants.
    
    Researchers can extend this class to implement custom model variants with
    novel interaction types, geometries, or energy functions. The plugin system
    automatically integrates custom models with the discovery pipeline.
    
    Example:
        class LongRangeIsingPlugin(ModelVariantPlugin):
            def __init__(self, alpha: float = 2.5, dimensions: int = 2):
                self.alpha = alpha
                self.dimensions = dimensions
                self._precompute_couplings()
            
            def compute_energy(self, configuration: np.ndarray) -> float:
                # Custom energy calculation with long-range interactions
                pass
            
            def propose_spin_flip(self, configuration: np.ndarray,
                                 site: Tuple[int, ...]) -> SpinFlipProposal:
                # Custom spin flip with long-range energy difference
                pass
    """
    
    @abstractmethod
    def compute_energy(self, configuration: np.ndarray) -> float:
        """Compute total energy of the system.
        
        Args:
            configuration: Spin configuration array
            
        Returns:
            Total energy of the configuration
        """
        pass
    
    @abstractmethod
    def propose_spin_flip(self, configuration: np.ndarray,
                         site: Tuple[int, ...]) -> SpinFlipProposal:
        """Propose a Monte Carlo spin flip move.
        
        Args:
            configuration: Current spin configuration
            site: Coordinates of site to potentially flip
            
        Returns:
            SpinFlipProposal with energy change and acceptance probability
        """
        pass
    
    @abstractmethod
    def get_theoretical_properties(self) -> Dict[str, Any]:
        """Return theoretical properties if known.
        
        Returns:
            Dictionary with keys:
                - 'tc': Theoretical critical temperature (or None)
                - 'exponents': Dict of theoretical exponents (or None)
                - 'universality_class': Name of universality class (or None)
        """
        pass
    
    def validate_configuration(self, configuration: np.ndarray) -> bool:
        """Validate that a configuration is physically valid.
        
        Args:
            configuration: Spin configuration to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        # Default implementation: check that spins are ±1
        return np.all(np.isin(configuration, [-1, 1]))
    
    def compute_magnetization(self, configuration: np.ndarray) -> float:
        """Compute magnetization of the configuration.
        
        Args:
            configuration: Spin configuration
            
        Returns:
            Magnetization (mean spin value)
        """
        return np.mean(configuration)
    
    def compute_energy_per_site(self, configuration: np.ndarray) -> float:
        """Compute energy per site.
        
        Args:
            configuration: Spin configuration
            
        Returns:
            Energy per site
        """
        total_energy = self.compute_energy(configuration)
        n_sites = configuration.size
        return total_energy / n_sites


class BaseSimulator(ABC):
    """Abstract base class for Monte Carlo simulators.
    
    This class defines the interface for Monte Carlo simulators that work with
    model variant plugins. Simulators handle equilibration, measurement, and
    data collection.
    """
    
    def __init__(self, model: ModelVariantPlugin, lattice_size: int,
                 temperature: float, seed: Optional[int] = None):
        """Initialize simulator.
        
        Args:
            model: Model variant plugin to simulate
            lattice_size: Linear size of the lattice
            temperature: Simulation temperature
            seed: Random seed for reproducibility
        """
        self.model = model
        self.lattice_size = lattice_size
        self.temperature = temperature
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize configuration
        self.configuration = self._initialize_configuration()
    
    @abstractmethod
    def _initialize_configuration(self) -> np.ndarray:
        """Initialize spin configuration.
        
        Returns:
            Initial spin configuration
        """
        pass
    
    @abstractmethod
    def equilibrate(self, n_steps: int) -> None:
        """Equilibrate the system.
        
        Args:
            n_steps: Number of Monte Carlo steps for equilibration
        """
        pass
    
    @abstractmethod
    def measure(self, n_samples: int, n_steps_between: int) -> Dict[str, np.ndarray]:
        """Perform measurements after equilibration.
        
        Args:
            n_samples: Number of independent samples to collect
            n_steps_between: Number of MC steps between samples
            
        Returns:
            Dictionary with keys:
                - 'configurations': Array of spin configurations
                - 'magnetizations': Array of magnetization values
                - 'energies': Array of energy values
        """
        pass
    
    def monte_carlo_step(self) -> None:
        """Perform one Monte Carlo sweep (one attempted flip per site).
        
        This is a default implementation that can be overridden for
        specialized update schemes.
        """
        shape = self.configuration.shape
        n_sites = self.configuration.size
        
        for _ in range(n_sites):
            # Choose random site
            site = tuple(np.random.randint(0, s) for s in shape)
            
            # Propose flip
            proposal = self.model.propose_spin_flip(self.configuration, site)
            
            # Apply temperature to acceptance probability
            # Metropolis: P = min(1, exp(-ΔE/T))
            beta = 1.0 / self.temperature if self.temperature > 0 else np.inf
            acceptance_prob = min(1.0, np.exp(-beta * proposal.energy_change))
            
            # Accept or reject
            if np.random.random() < acceptance_prob:
                self.configuration[site] *= -1
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current state of the simulator.
        
        Returns:
            Dictionary with current configuration, energy, magnetization
        """
        return {
            'configuration': self.configuration.copy(),
            'energy': self.model.compute_energy(self.configuration),
            'magnetization': self.model.compute_magnetization(self.configuration),
            'temperature': self.temperature
        }
