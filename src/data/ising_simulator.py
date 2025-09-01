"""
Ising Model Monte Carlo Simulator using the Metropolis algorithm.

This module implements a 2D Ising model simulator for generating spin configurations
across different temperatures. The simulator uses the Metropolis algorithm for
proper statistical sampling and includes methods for equilibration and measurement.
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import time


@dataclass
class SpinConfiguration:
    """Data structure for storing spin configuration and associated properties."""
    spins: np.ndarray  # Shape: (height, width), values: {-1, +1}
    temperature: float
    magnetization: float
    energy: float
    metadata: Dict[str, Any]


class IsingSimulator:
    """
    2D Ising Model Monte Carlo Simulator using Metropolis algorithm.
    
    The Ising model is defined by the Hamiltonian:
    H = -J * Σ(s_i * s_j) - h * Σ(s_i)
    
    Where:
    - J is the coupling constant (set to 1.0)
    - h is the external magnetic field (set to 0.0)
    - s_i are spin values (+1 or -1)
    - The sum is over nearest neighbors
    """
    
    def __init__(self, lattice_size: Tuple[int, int], temperature: float, 
                 coupling: float = 1.0, magnetic_field: float = 0.0):
        """
        Initialize the Ising simulator.
        
        Args:
            lattice_size: (height, width) of the 2D lattice
            temperature: Temperature in units where k_B = 1
            coupling: Coupling constant J (default: 1.0)
            magnetic_field: External magnetic field h (default: 0.0)
        """
        self.height, self.width = lattice_size
        self.temperature = temperature
        self.coupling = coupling
        self.magnetic_field = magnetic_field
        self.beta = 1.0 / temperature  # Inverse temperature
        
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        # Initialize lattice with random spins
        self.lattice = np.random.choice([-1, 1], size=(self.height, self.width))
        
        # Precompute exponentials for efficiency
        self._precompute_exponentials()
        
        # Statistics tracking
        self.step_count = 0
        self.accepted_moves = 0
        
        self.logger.info(f"Initialized Ising simulator: {self.height}x{self.width} lattice, T={temperature}")
    
    def _precompute_exponentials(self) -> None:
        """Precompute exponential factors for Metropolis acceptance probabilities."""
        # Possible energy differences for spin flip: ΔE = 2*J*s_i*(sum of neighbors) + 2*h*s_i
        # For J=1, h=0, and 4 neighbors, possible values are: -8, -4, 0, 4, 8
        # We only need exp(-β*ΔE) for ΔE > 0 (since we always accept ΔE ≤ 0)
        self.exp_factors = {}
        for delta_e in [4, 8]:  # Only positive energy differences
            self.exp_factors[delta_e] = np.exp(-self.beta * delta_e)
        
        self.logger.debug(f"Precomputed exponential factors: {self.exp_factors}")
    
    def _calculate_local_energy(self, i: int, j: int) -> float:
        """
        Calculate the local energy contribution of spin at position (i, j).
        
        Args:
            i, j: Lattice coordinates
            
        Returns:
            Local energy contribution
        """
        spin = self.lattice[i, j]
        
        # Sum over nearest neighbors with periodic boundary conditions
        neighbor_sum = (
            self.lattice[(i-1) % self.height, j] +  # Up
            self.lattice[(i+1) % self.height, j] +  # Down
            self.lattice[i, (j-1) % self.width] +   # Left
            self.lattice[i, (j+1) % self.width]     # Right
        )
        
        # Local energy: -J * s_i * (sum of neighbors) - h * s_i
        local_energy = -self.coupling * spin * neighbor_sum - self.magnetic_field * spin
        return local_energy
    
    def _calculate_energy_difference(self, i: int, j: int) -> float:
        """
        Calculate energy difference if spin at (i, j) is flipped.
        
        Args:
            i, j: Lattice coordinates
            
        Returns:
            Energy difference ΔE = E_new - E_old
        """
        spin = self.lattice[i, j]
        
        # Sum over nearest neighbors
        neighbor_sum = (
            self.lattice[(i-1) % self.height, j] +
            self.lattice[(i+1) % self.height, j] +
            self.lattice[i, (j-1) % self.width] +
            self.lattice[i, (j+1) % self.width]
        )
        
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
        i = np.random.randint(0, self.height)
        j = np.random.randint(0, self.width)
        
        # Calculate energy difference for flipping this spin
        delta_e = self._calculate_energy_difference(i, j)
        
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
            self.lattice[i, j] *= -1  # Flip the spin
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
        n_sites = self.height * self.width
        
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
        magnetization_per_spin = total_magnetization / (self.height * self.width)
        return magnetization_per_spin
    
    def calculate_energy(self) -> float:
        """
        Calculate the total energy of the current configuration.
        
        Returns:
            Total energy of the system
        """
        energy = 0.0
        
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
        
        return energy
    
    def calculate_energy_per_spin(self) -> float:
        """Calculate energy per spin."""
        total_energy = self.calculate_energy()
        return total_energy / (self.height * self.width)
    
    def get_acceptance_rate(self) -> float:
        """Get the current acceptance rate."""
        if self.step_count == 0:
            return 0.0
        return self.accepted_moves / self.step_count
    
    def reset_statistics(self) -> None:
        """Reset step count and acceptance statistics."""
        self.step_count = 0
        self.accepted_moves = 0
    
    def get_configuration(self) -> SpinConfiguration:
        """
        Get current spin configuration with calculated properties.
        
        Returns:
            SpinConfiguration object with current state
        """
        magnetization = self.calculate_magnetization()
        energy = self.calculate_energy_per_spin()
        
        metadata = {
            'step_count': self.step_count,
            'acceptance_rate': self.get_acceptance_rate(),
            'lattice_size': (self.height, self.width),
            'coupling': self.coupling,
            'magnetic_field': self.magnetic_field
        }
        
        return SpinConfiguration(
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
        self.lattice = np.random.choice([-1, 1], size=(self.height, self.width))
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
    
    def copy(self) -> 'IsingSimulator':
        """Create a copy of the simulator with the same configuration."""
        new_simulator = IsingSimulator(
            lattice_size=(self.height, self.width),
            temperature=self.temperature,
            coupling=self.coupling,
            magnetic_field=self.magnetic_field
        )
        new_simulator.lattice = self.lattice.copy()
        new_simulator.step_count = self.step_count
        new_simulator.accepted_moves = self.accepted_moves
        return new_simulator