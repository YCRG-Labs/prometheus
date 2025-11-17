"""
Enhanced Monte Carlo Simulator for 3D Ising Model.

This module extends the existing 2D Ising simulator to support 3D lattice geometries
with 6-neighbor interactions and proper periodic boundary conditions in all dimensions.
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import time

from .ising_simulator import IsingSimulator, SpinConfiguration


@dataclass
class SpinConfiguration3D:
    """Data structure for storing 3D spin configuration and associated properties."""
    spins: np.ndarray  # Shape: (depth, height, width), values: {-1, +1}
    temperature: float
    magnetization: float
    energy: float
    metadata: Dict[str, Any]


class EnhancedMonteCarloSimulator:
    """
    Enhanced Monte Carlo Simulator supporting both 2D and 3D Ising models.
    
    Extends the existing 2D implementation to handle 3D lattice geometries with
    6-neighbor interactions while maintaining compatibility with 2D systems.
    
    The 3D Ising model is defined by the Hamiltonian:
    H = -J * Σ(s_i * s_j) - h * Σ(s_i)
    
    Where:
    - J is the coupling constant (set to 1.0)
    - h is the external magnetic field (set to 0.0)
    - s_i are spin values (+1 or -1)
    - The sum is over nearest neighbors (4 in 2D, 6 in 3D)
    """
    
    def __init__(self, 
                 lattice_size: Union[Tuple[int, int], Tuple[int, int, int]], 
                 temperature: float,
                 coupling: float = 1.0, 
                 magnetic_field: float = 0.0):
        """
        Initialize the enhanced Monte Carlo simulator.
        
        Args:
            lattice_size: (height, width) for 2D or (depth, height, width) for 3D
            temperature: Temperature in units where k_B = 1
            coupling: Coupling constant J (default: 1.0)
            magnetic_field: External magnetic field h (default: 0.0)
        """
        self.lattice_size = lattice_size
        self.dimensions = len(lattice_size)
        self.temperature = temperature
        self.coupling = coupling
        self.magnetic_field = magnetic_field
        self.beta = 1.0 / temperature  # Inverse temperature
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Validate dimensions
        if self.dimensions not in [2, 3]:
            raise ValueError(f"Only 2D and 3D lattices are supported, got {self.dimensions}D")
        
        # Set up dimension-specific parameters
        if self.dimensions == 2:
            self.height, self.width = lattice_size
            self.depth = None
            self.n_neighbors = 4
        else:  # 3D
            self.depth, self.height, self.width = lattice_size
            self.n_neighbors = 6
        
        # Initialize lattice with random spins
        self.lattice = np.random.choice([-1, 1], size=lattice_size)
        
        # Precompute exponentials for efficiency
        self._precompute_exponentials()
        
        # Statistics tracking
        self.step_count = 0
        self.accepted_moves = 0
        
        self.logger.info(f"Initialized {self.dimensions}D Enhanced Monte Carlo simulator: "
                        f"lattice_size={lattice_size}, T={temperature}")
    
    def _precompute_exponentials(self) -> None:
        """Precompute exponential factors for Metropolis acceptance probabilities."""
        # Possible energy differences for spin flip: ΔE = 2*J*s_i*(sum of neighbors) + 2*h*s_i
        # For J=1, h=0:
        # - 2D: 4 neighbors, possible ΔE values: -8, -4, 0, 4, 8
        # - 3D: 6 neighbors, possible ΔE values: -12, -8, -4, 0, 4, 8, 12
        # We only need exp(-β*ΔE) for ΔE > 0 (since we always accept ΔE ≤ 0)
        
        self.exp_factors = {}
        max_neighbors = self.n_neighbors
        
        for delta_e in range(4, 4 * max_neighbors + 1, 4):  # 4, 8, 12, ... up to 4*n_neighbors
            self.exp_factors[delta_e] = np.exp(-self.beta * delta_e)
        
        self.logger.debug(f"Precomputed exponential factors for {self.dimensions}D: {self.exp_factors}")
    
    def _get_neighbors_2d(self, i: int, j: int) -> list:
        """
        Get 4 nearest neighbors for 2D lattice with periodic boundary conditions.
        
        Args:
            i, j: Lattice coordinates
            
        Returns:
            List of neighbor coordinates
        """
        neighbors = [
            ((i-1) % self.height, j),  # Up
            ((i+1) % self.height, j),  # Down
            (i, (j-1) % self.width),   # Left
            (i, (j+1) % self.width)    # Right
        ]
        return neighbors
    
    def _get_neighbors_3d(self, k: int, i: int, j: int) -> list:
        """
        Get 6 nearest neighbors for 3D lattice with periodic boundary conditions.
        
        Args:
            k, i, j: Lattice coordinates (depth, height, width)
            
        Returns:
            List of neighbor coordinates
        """
        neighbors = [
            ((k-1) % self.depth, i, j),  # Front
            ((k+1) % self.depth, i, j),  # Back
            (k, (i-1) % self.height, j), # Up
            (k, (i+1) % self.height, j), # Down
            (k, i, (j-1) % self.width),  # Left
            (k, i, (j+1) % self.width)   # Right
        ]
        return neighbors
    
    def _calculate_local_energy(self, *coords) -> float:
        """
        Calculate the local energy contribution of spin at given coordinates.
        
        Args:
            *coords: Lattice coordinates (i, j) for 2D or (k, i, j) for 3D
            
        Returns:
            Local energy contribution
        """
        if self.dimensions == 2:
            i, j = coords
            spin = self.lattice[i, j]
            neighbors = self._get_neighbors_2d(i, j)
        else:  # 3D
            k, i, j = coords
            spin = self.lattice[k, i, j]
            neighbors = self._get_neighbors_3d(k, i, j)
        
        # Sum over nearest neighbors
        neighbor_sum = sum(self.lattice[neighbor] for neighbor in neighbors)
        
        # Local energy: -J * s_i * (sum of neighbors) - h * s_i
        local_energy = -self.coupling * spin * neighbor_sum - self.magnetic_field * spin
        return local_energy
    
    def _calculate_energy_difference(self, *coords) -> float:
        """
        Calculate energy difference if spin at given coordinates is flipped.
        
        Args:
            *coords: Lattice coordinates (i, j) for 2D or (k, i, j) for 3D
            
        Returns:
            Energy difference ΔE = E_new - E_old
        """
        if self.dimensions == 2:
            i, j = coords
            spin = self.lattice[i, j]
            neighbors = self._get_neighbors_2d(i, j)
        else:  # 3D
            k, i, j = coords
            spin = self.lattice[k, i, j]
            neighbors = self._get_neighbors_3d(k, i, j)
        
        # Sum over nearest neighbors
        neighbor_sum = sum(self.lattice[neighbor] for neighbor in neighbors)
        
        # Energy difference for flipping spin: ΔE = 2 * J * s_i * neighbor_sum + 2 * h * s_i
        delta_e = 2 * self.coupling * spin * neighbor_sum + 2 * self.magnetic_field * spin
        return delta_e
    
    def metropolis_step(self) -> bool:
        """
        Perform one Metropolis Monte Carlo step.
        
        Randomly selects a spin, calculates the energy difference for flipping it,
        and accepts or rejects the move based on the Metropolis criterion.
        
        Returns:
            True if the move was accepted, False otherwise
        """
        # Randomly select a lattice site
        if self.dimensions == 2:
            i = np.random.randint(0, self.height)
            j = np.random.randint(0, self.width)
            coords = (i, j)
        else:  # 3D
            k = np.random.randint(0, self.depth)
            i = np.random.randint(0, self.height)
            j = np.random.randint(0, self.width)
            coords = (k, i, j)
        
        # Calculate energy difference for flipping this spin
        delta_e = self._calculate_energy_difference(*coords)
        
        # Metropolis acceptance criterion
        if delta_e <= 0:
            # Always accept moves that decrease energy
            accept = True
        else:
            # Accept with probability exp(-β * ΔE)
            if delta_e in self.exp_factors:
                acceptance_prob = self.exp_factors[delta_e]
            else:
                acceptance_prob = np.exp(-self.beta * delta_e)
            
            accept = np.random.random() < acceptance_prob
        
        # Apply the move if accepted
        if accept:
            self.lattice[coords] *= -1  # Flip the spin
            self.accepted_moves += 1
        
        self.step_count += 1
        return accept
    
    def sweep(self) -> int:
        """
        Perform one Monte Carlo sweep (N Metropolis steps where N is lattice size).
        
        Returns:
            Number of accepted moves in this sweep
        """
        initial_accepted = self.accepted_moves
        n_sites = np.prod(self.lattice_size)
        
        for _ in range(n_sites):
            self.metropolis_step()
        
        return self.accepted_moves - initial_accepted
    
    def calculate_magnetization(self) -> float:
        """
        Calculate the magnetization of the current configuration.
        
        Returns:
            Magnetization per spin: M = (1/N) * Σ(s_i)
        """
        total_magnetization = np.sum(self.lattice)
        n_sites = np.prod(self.lattice_size)
        magnetization_per_spin = total_magnetization / n_sites
        return magnetization_per_spin
    
    def calculate_energy(self) -> float:
        """
        Calculate the total energy of the current configuration.
        
        Returns:
            Total energy of the system
        """
        energy = 0.0
        
        if self.dimensions == 2:
            # 2D energy calculation
            for i in range(self.height):
                for j in range(self.width):
                    spin = self.lattice[i, j]
                    
                    # Count each pair only once by considering only right and down neighbors
                    right_neighbor = self.lattice[i, (j+1) % self.width]
                    down_neighbor = self.lattice[(i+1) % self.height, j]
                    
                    # Interaction energy with neighbors
                    energy -= self.coupling * spin * (right_neighbor + down_neighbor)
                    
                    # Magnetic field energy
                    energy -= self.magnetic_field * spin
        else:  # 3D
            # 3D energy calculation
            for k in range(self.depth):
                for i in range(self.height):
                    for j in range(self.width):
                        spin = self.lattice[k, i, j]
                        
                        # Count each pair only once by considering only positive direction neighbors
                        back_neighbor = self.lattice[(k+1) % self.depth, i, j]
                        down_neighbor = self.lattice[k, (i+1) % self.height, j]
                        right_neighbor = self.lattice[k, i, (j+1) % self.width]
                        
                        # Interaction energy with neighbors
                        energy -= self.coupling * spin * (back_neighbor + down_neighbor + right_neighbor)
                        
                        # Magnetic field energy
                        energy -= self.magnetic_field * spin
        
        return energy
    
    def calculate_energy_per_spin(self) -> float:
        """Calculate energy per spin."""
        total_energy = self.calculate_energy()
        n_sites = np.prod(self.lattice_size)
        return total_energy / n_sites
    
    def get_acceptance_rate(self) -> float:
        """Get the current acceptance rate."""
        if self.step_count == 0:
            return 0.0
        return self.accepted_moves / self.step_count
    
    def reset_statistics(self) -> None:
        """Reset step count and acceptance statistics."""
        self.step_count = 0
        self.accepted_moves = 0
    
    def get_configuration(self) -> Union[SpinConfiguration, SpinConfiguration3D]:
        """
        Get current spin configuration with calculated properties.
        
        Returns:
            SpinConfiguration (2D) or SpinConfiguration3D (3D) object with current state
        """
        magnetization = self.calculate_magnetization()
        energy = self.calculate_energy_per_spin()
        
        metadata = {
            'step_count': self.step_count,
            'acceptance_rate': self.get_acceptance_rate(),
            'lattice_size': self.lattice_size,
            'dimensions': self.dimensions,
            'coupling': self.coupling,
            'magnetic_field': self.magnetic_field,
            'n_neighbors': self.n_neighbors
        }
        
        if self.dimensions == 2:
            return SpinConfiguration(
                spins=self.lattice.copy(),
                temperature=self.temperature,
                magnetization=magnetization,
                energy=energy,
                metadata=metadata
            )
        else:  # 3D
            return SpinConfiguration3D(
                spins=self.lattice.copy(),
                temperature=self.temperature,
                magnetization=magnetization,
                energy=energy,
                metadata=metadata
            )
    
    def set_temperature(self, temperature: float) -> None:
        """
        Change the temperature and update related parameters.
        
        Args:
            temperature: New temperature value
        """
        self.temperature = temperature
        self.beta = 1.0 / temperature
        self._precompute_exponentials()
        self.logger.debug(f"Temperature changed to {temperature}")
    
    def randomize_lattice(self) -> None:
        """Randomize the lattice configuration."""
        self.lattice = np.random.choice([-1, 1], size=self.lattice_size)
        self.reset_statistics()
        self.logger.debug("Lattice randomized")
    
    def set_ordered_state(self, spin_value: int = 1) -> None:
        """
        Set lattice to completely ordered state.
        
        Args:
            spin_value: Spin value to set all sites to (+1 or -1)
        """
        if spin_value not in [-1, 1]:
            raise ValueError("spin_value must be +1 or -1")
        
        self.lattice.fill(spin_value)
        self.reset_statistics()
        self.logger.debug(f"Lattice set to ordered state with spin {spin_value}")
    
    def copy(self) -> 'EnhancedMonteCarloSimulator':
        """Create a copy of the simulator with the same configuration."""
        new_simulator = EnhancedMonteCarloSimulator(
            lattice_size=self.lattice_size,
            temperature=self.temperature,
            coupling=self.coupling,
            magnetic_field=self.magnetic_field
        )
        new_simulator.lattice = self.lattice.copy()
        new_simulator.step_count = self.step_count
        new_simulator.accepted_moves = self.accepted_moves
        return new_simulator
    
    def get_lattice_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the lattice configuration.
        
        Returns:
            Dictionary with lattice properties and statistics
        """
        n_sites = np.prod(self.lattice_size)
        
        info = {
            'dimensions': self.dimensions,
            'lattice_size': self.lattice_size,
            'n_sites': n_sites,
            'n_neighbors': self.n_neighbors,
            'temperature': self.temperature,
            'coupling': self.coupling,
            'magnetic_field': self.magnetic_field,
            'current_magnetization': self.calculate_magnetization(),
            'current_energy_per_spin': self.calculate_energy_per_spin(),
            'step_count': self.step_count,
            'acceptance_rate': self.get_acceptance_rate(),
            'spin_up_fraction': np.sum(self.lattice == 1) / n_sites,
            'spin_down_fraction': np.sum(self.lattice == -1) / n_sites
        }
        
        return info
    
    def validate_lattice_integrity(self) -> Dict[str, Any]:
        """
        Validate the integrity of the lattice configuration.
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'errors': []
        }
        
        # Check spin values
        unique_spins = np.unique(self.lattice)
        if not np.array_equal(np.sort(unique_spins), np.array([-1, 1])):
            if len(unique_spins) > 2 or not all(spin in [-1, 1] for spin in unique_spins):
                validation['is_valid'] = False
                validation['errors'].append(f"Invalid spin values found: {unique_spins}")
        
        # Check lattice shape
        if self.lattice.shape != self.lattice_size:
            validation['is_valid'] = False
            validation['errors'].append(
                f"Lattice shape mismatch: expected {self.lattice_size}, got {self.lattice.shape}"
            )
        
        # Check dimensions consistency
        if len(self.lattice_size) != self.dimensions:
            validation['is_valid'] = False
            validation['errors'].append(
                f"Dimension mismatch: lattice_size has {len(self.lattice_size)} dimensions, "
                f"but simulator is configured for {self.dimensions}D"
            )
        
        return validation


def create_2d_simulator(lattice_size: Tuple[int, int], temperature: float, **kwargs) -> EnhancedMonteCarloSimulator:
    """
    Convenience function to create a 2D Enhanced Monte Carlo simulator.
    
    Args:
        lattice_size: (height, width) of the 2D lattice
        temperature: Temperature value
        **kwargs: Additional parameters for the simulator
        
    Returns:
        EnhancedMonteCarloSimulator configured for 2D
    """
    return EnhancedMonteCarloSimulator(lattice_size=lattice_size, temperature=temperature, **kwargs)


def create_3d_simulator(lattice_size: Tuple[int, int, int], temperature: float, **kwargs) -> EnhancedMonteCarloSimulator:
    """
    Convenience function to create a 3D Enhanced Monte Carlo simulator.
    
    Args:
        lattice_size: (depth, height, width) of the 3D lattice
        temperature: Temperature value
        **kwargs: Additional parameters for the simulator
        
    Returns:
        EnhancedMonteCarloSimulator configured for 3D
    """
    return EnhancedMonteCarloSimulator(lattice_size=lattice_size, temperature=temperature, **kwargs)