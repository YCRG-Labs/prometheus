"""
Physics Models for Statistical Mechanics Systems.

This module implements physics model classes for different statistical mechanics systems
including 3D Ising, Potts, and XY models. Each model provides methods for energy calculation,
magnetization computation, and spin flip proposals following a unified interface.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Union
from dataclasses import dataclass
import logging


@dataclass
class SpinFlipProposal:
    """Data structure for spin flip proposals."""
    site: Tuple[int, ...]
    old_spin: Union[int, float, np.ndarray]
    new_spin: Union[int, float, np.ndarray]
    energy_difference: float


class PhysicsModel(ABC):
    """
    Abstract base class for physics models in statistical mechanics.
    
    Provides a unified interface for different statistical mechanics models
    including energy calculations, magnetization computation, and Monte Carlo
    spin flip proposals.
    """
    
    def __init__(self, model_name: str, dimensions: int, coupling_strength: float = 1.0):
        """
        Initialize physics model.
        
        Args:
            model_name: Name of the physics model
            dimensions: Spatial dimensions (2 or 3)
            coupling_strength: Coupling constant J
        """
        self.model_name = model_name
        self.dimensions = dimensions
        self.coupling_strength = coupling_strength
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def compute_energy(self, configuration: np.ndarray) -> float:
        """
        Compute total energy of the configuration.
        
        Args:
            configuration: Spin configuration array
            
        Returns:
            Total energy of the system
        """
        pass
    
    @abstractmethod
    def compute_magnetization(self, configuration: np.ndarray) -> float:
        """
        Compute magnetization of the configuration.
        
        Args:
            configuration: Spin configuration array
            
        Returns:
            Magnetization per spin
        """
        pass
    
    @abstractmethod
    def get_theoretical_tc(self) -> float:
        """
        Get theoretical critical temperature.
        
        Returns:
            Theoretical critical temperature
        """
        pass
    
    @abstractmethod
    def get_theoretical_exponents(self) -> Dict[str, float]:
        """
        Get theoretical critical exponents.
        
        Returns:
            Dictionary of critical exponents
        """
        pass
    
    @abstractmethod
    def propose_spin_flip(self, configuration: np.ndarray, site: Tuple[int, ...]) -> SpinFlipProposal:
        """
        Propose a spin flip at the given site.
        
        Args:
            configuration: Current spin configuration
            site: Lattice site coordinates
            
        Returns:
            SpinFlipProposal with details of the proposed move
        """
        pass
    
    @abstractmethod
    def get_neighbors(self, site: Tuple[int, ...], lattice_shape: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """
        Get nearest neighbors of a lattice site with periodic boundary conditions.
        
        Args:
            site: Lattice site coordinates
            lattice_shape: Shape of the lattice
            
        Returns:
            List of neighbor coordinates
        """
        pass


class Ising2DModel(PhysicsModel):
    """
    2D Ising Model implementation with nearest-neighbor interactions.
    
    The 2D Ising model is defined by the Hamiltonian:
    H = -J * Σ(s_i * s_j) - h * Σ(s_i)
    
    Where:
    - J is the coupling constant
    - h is the external magnetic field (set to 0)
    - s_i are spin values (+1 or -1)
    - The sum is over nearest neighbors (4 in 2D)
    """
    
    def __init__(self, coupling_strength: float = 1.0, magnetic_field: float = 0.0):
        """
        Initialize 2D Ising model.
        
        Args:
            coupling_strength: Coupling constant J (default: 1.0)
            magnetic_field: External magnetic field h (default: 0.0)
        """
        super().__init__("Ising2D", 2, coupling_strength)
        self.magnetic_field = magnetic_field
        
        # Theoretical values for 2D Ising model (Onsager solution)
        self.theoretical_tc = 2.269 * coupling_strength  # Critical temperature for J=1
        self.theoretical_exponents = {
            'beta': 0.125,   # Magnetization exponent
            'nu': 1.0,       # Correlation length exponent
            'gamma': 1.75,   # Susceptibility exponent
            'alpha': 0.0,    # Specific heat exponent (logarithmic divergence)
            'delta': 15.0,   # Critical isotherm exponent
            'eta': 0.25      # Anomalous dimension
        }
        
        self.logger.info(f"Initialized 2D Ising model: J={coupling_strength}, h={magnetic_field}")
    
    def compute_energy(self, configuration: np.ndarray) -> float:
        """
        Compute total energy of the 2D Ising configuration.
        
        Uses the Hamiltonian H = -J * Σ(s_i * s_j) - h * Σ(s_i)
        where the sum is over nearest neighbor pairs.
        
        Args:
            configuration: 2D spin configuration array with shape (height, width)
            
        Returns:
            Total energy of the system
        """
        if configuration.ndim != 2:
            raise ValueError(f"Expected 2D configuration, got {configuration.ndim}D")
        
        height, width = configuration.shape
        energy = 0.0
        
        # Interaction energy: count each pair only once
        for i in range(height):
            for j in range(width):
                spin = configuration[i, j]
                
                # Consider only positive direction neighbors to avoid double counting
                # Down neighbor (i+1)
                down_neighbor = configuration[(i + 1) % height, j]
                energy -= self.coupling_strength * spin * down_neighbor
                
                # Right neighbor (j+1)
                right_neighbor = configuration[i, (j + 1) % width]
                energy -= self.coupling_strength * spin * right_neighbor
                
                # Magnetic field energy
                energy -= self.magnetic_field * spin
        
        return energy
    
    def compute_magnetization(self, configuration: np.ndarray) -> float:
        """
        Compute magnetization per spin of the 2D configuration.
        
        Args:
            configuration: 2D spin configuration array
            
        Returns:
            Magnetization per spin: M = (1/N) * Σ(s_i)
        """
        if configuration.ndim != 2:
            raise ValueError(f"Expected 2D configuration, got {configuration.ndim}D")
        
        total_magnetization = np.sum(configuration)
        n_sites = np.prod(configuration.shape)
        return total_magnetization / n_sites
    
    def get_theoretical_tc(self) -> float:
        """
        Get theoretical critical temperature for 2D Ising model.
        
        Returns:
            Theoretical critical temperature Tc ≈ 2.269 (for J=1)
        """
        return self.theoretical_tc
    
    def get_theoretical_exponents(self) -> Dict[str, float]:
        """
        Get theoretical critical exponents for 2D Ising model.
        
        Returns:
            Dictionary of critical exponents
        """
        return self.theoretical_exponents.copy()
    
    def get_neighbors(self, site: Tuple[int, int], lattice_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get 4 nearest neighbors for 2D lattice with periodic boundary conditions.
        
        Args:
            site: Lattice site coordinates (i, j)
            lattice_shape: Shape of the lattice (height, width)
            
        Returns:
            List of 4 neighbor coordinates
        """
        i, j = site
        height, width = lattice_shape
        
        neighbors = [
            ((i + 1) % height, j),  # Down
            ((i - 1) % height, j),  # Up
            (i, (j + 1) % width),   # Right
            (i, (j - 1) % width)    # Left
        ]
        
        return neighbors
    
    def propose_spin_flip(self, configuration: np.ndarray, site: Tuple[int, int]) -> SpinFlipProposal:
        """
        Propose a spin flip at the given site for 2D Ising model.
        
        Args:
            configuration: Current 2D spin configuration
            site: Lattice site coordinates (i, j)
            
        Returns:
            SpinFlipProposal with energy difference
        """
        if configuration.ndim != 2:
            raise ValueError(f"Expected 2D configuration, got {configuration.ndim}D")
        
        i, j = site
        old_spin = configuration[i, j]
        new_spin = -old_spin  # Flip spin
        
        # Calculate energy difference
        neighbors = self.get_neighbors(site, configuration.shape)
        neighbor_sum = sum(configuration[ni, nj] for ni, nj in neighbors)
        
        # ΔE = E_new - E_old = -J * (new_spin - old_spin) * Σ(neighbors) - h * (new_spin - old_spin)
        delta_e = -2 * self.coupling_strength * old_spin * neighbor_sum - 2 * self.magnetic_field * old_spin
        
        return SpinFlipProposal(
            site=site,
            old_spin=old_spin,
            new_spin=new_spin,
            energy_difference=delta_e
        )
    
    def validate_configuration(self, configuration: np.ndarray) -> Dict[str, Any]:
        """
        Validate the 2D Ising configuration.
        
        Args:
            configuration: 2D spin configuration to validate
            
        Returns:
            Dictionary with validation results
        """
        is_valid = True
        issues = []
        
        if configuration.ndim != 2:
            is_valid = False
            issues.append(f"Expected 2D configuration, got {configuration.ndim}D")
        
        unique_values = np.unique(configuration)
        if not np.all(np.isin(unique_values, [-1, 1])):
            is_valid = False
            issues.append(f"Invalid spin values: {unique_values}. Expected only -1 and +1")
        
        return {
            'is_valid': is_valid,
            'issues': issues,
            'shape': configuration.shape,
            'unique_values': unique_values.tolist()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the 2D Ising model.
        
        Returns:
            Dictionary with model properties
        """
        return {
            'model_name': self.model_name,
            'dimensions': self.dimensions,
            'coupling_strength': self.coupling_strength,
            'magnetic_field': self.magnetic_field,
            'theoretical_tc': self.get_theoretical_tc(),
            'theoretical_exponents': self.get_theoretical_exponents()
        }


class Ising3DModel(PhysicsModel):
    """
    3D Ising Model implementation with nearest-neighbor interactions.
    
    The 3D Ising model is defined by the Hamiltonian:
    H = -J * Σ(s_i * s_j) - h * Σ(s_i)
    
    Where:
    - J is the coupling constant
    - h is the external magnetic field (set to 0)
    - s_i are spin values (+1 or -1)
    - The sum is over nearest neighbors (6 in 3D)
    """
    
    def __init__(self, coupling_strength: float = 1.0, magnetic_field: float = 0.0):
        """
        Initialize 3D Ising model.
        
        Args:
            coupling_strength: Coupling constant J (default: 1.0)
            magnetic_field: External magnetic field h (default: 0.0)
        """
        super().__init__("Ising3D", 3, coupling_strength)
        self.magnetic_field = magnetic_field
        
        # Theoretical values for 3D Ising model
        self.theoretical_tc = 4.511  # Critical temperature for J=1
        self.theoretical_exponents = {
            'beta': 0.326,   # Magnetization exponent
            'nu': 0.630,     # Correlation length exponent
            'gamma': 1.237,  # Susceptibility exponent
            'alpha': 0.110,  # Specific heat exponent
            'delta': 4.789,  # Critical isotherm exponent
            'eta': 0.036     # Anomalous dimension
        }
        
        self.logger.info(f"Initialized 3D Ising model: J={coupling_strength}, h={magnetic_field}")
    
    def compute_energy(self, configuration: np.ndarray) -> float:
        """
        Compute total energy of the 3D Ising configuration.
        
        Uses the Hamiltonian H = -J * Σ(s_i * s_j) - h * Σ(s_i)
        where the sum is over nearest neighbor pairs.
        
        Args:
            configuration: 3D spin configuration array with shape (depth, height, width)
            
        Returns:
            Total energy of the system
        """
        if configuration.ndim != 3:
            raise ValueError(f"Expected 3D configuration, got {configuration.ndim}D")
        
        depth, height, width = configuration.shape
        energy = 0.0
        
        # Interaction energy: count each pair only once
        for k in range(depth):
            for i in range(height):
                for j in range(width):
                    spin = configuration[k, i, j]
                    
                    # Consider only positive direction neighbors to avoid double counting
                    # Back neighbor (k+1)
                    back_neighbor = configuration[(k + 1) % depth, i, j]
                    energy -= self.coupling_strength * spin * back_neighbor
                    
                    # Down neighbor (i+1)
                    down_neighbor = configuration[k, (i + 1) % height, j]
                    energy -= self.coupling_strength * spin * down_neighbor
                    
                    # Right neighbor (j+1)
                    right_neighbor = configuration[k, i, (j + 1) % width]
                    energy -= self.coupling_strength * spin * right_neighbor
                    
                    # Magnetic field energy
                    energy -= self.magnetic_field * spin
        
        return energy
    
    def compute_magnetization(self, configuration: np.ndarray) -> float:
        """
        Compute magnetization per spin of the 3D configuration.
        
        Args:
            configuration: 3D spin configuration array
            
        Returns:
            Magnetization per spin: M = (1/N) * Σ(s_i)
        """
        if configuration.ndim != 3:
            raise ValueError(f"Expected 3D configuration, got {configuration.ndim}D")
        
        total_magnetization = np.sum(configuration)
        n_sites = np.prod(configuration.shape)
        return total_magnetization / n_sites
    
    def get_theoretical_tc(self) -> float:
        """
        Get theoretical critical temperature for 3D Ising model.
        
        Returns:
            Theoretical critical temperature Tc ≈ 4.511 (for J=1)
        """
        return self.theoretical_tc * self.coupling_strength
    
    def get_theoretical_exponents(self) -> Dict[str, float]:
        """
        Get theoretical critical exponents for 3D Ising model.
        
        Returns:
            Dictionary of critical exponents
        """
        return self.theoretical_exponents.copy()
    
    def get_neighbors(self, site: Tuple[int, int, int], lattice_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """
        Get 6 nearest neighbors for 3D lattice with periodic boundary conditions.
        
        Args:
            site: Lattice site coordinates (k, i, j)
            lattice_shape: Shape of the lattice (depth, height, width)
            
        Returns:
            List of 6 neighbor coordinates
        """
        k, i, j = site
        depth, height, width = lattice_shape
        
        neighbors = [
            ((k - 1) % depth, i, j),  # Front
            ((k + 1) % depth, i, j),  # Back
            (k, (i - 1) % height, j), # Up
            (k, (i + 1) % height, j), # Down
            (k, i, (j - 1) % width),  # Left
            (k, i, (j + 1) % width)   # Right
        ]
        
        return neighbors
    
    def compute_local_energy(self, configuration: np.ndarray, site: Tuple[int, int, int]) -> float:
        """
        Compute local energy contribution of spin at given site.
        
        Args:
            configuration: 3D spin configuration array
            site: Lattice site coordinates (k, i, j)
            
        Returns:
            Local energy contribution
        """
        k, i, j = site
        spin = configuration[k, i, j]
        
        # Get neighbors and sum their spins
        neighbors = self.get_neighbors(site, configuration.shape)
        neighbor_sum = sum(configuration[neighbor] for neighbor in neighbors)
        
        # Local energy: -J * s_i * (sum of neighbors) - h * s_i
        local_energy = -self.coupling_strength * spin * neighbor_sum - self.magnetic_field * spin
        return local_energy
    
    def compute_energy_difference(self, configuration: np.ndarray, site: Tuple[int, int, int]) -> float:
        """
        Compute energy difference if spin at given site is flipped.
        
        Args:
            configuration: 3D spin configuration array
            site: Lattice site coordinates (k, i, j)
            
        Returns:
            Energy difference ΔE = E_new - E_old
        """
        k, i, j = site
        spin = configuration[k, i, j]
        
        # Get neighbors and sum their spins
        neighbors = self.get_neighbors(site, configuration.shape)
        neighbor_sum = sum(configuration[neighbor] for neighbor in neighbors)
        
        # Energy difference for flipping spin: ΔE = 2 * J * s_i * neighbor_sum + 2 * h * s_i
        delta_e = 2 * self.coupling_strength * spin * neighbor_sum + 2 * self.magnetic_field * spin
        return delta_e
    
    def propose_spin_flip(self, configuration: np.ndarray, site: Tuple[int, int, int]) -> SpinFlipProposal:
        """
        Propose a spin flip at the given 3D lattice site.
        
        Args:
            configuration: Current 3D spin configuration
            site: Lattice site coordinates (k, i, j)
            
        Returns:
            SpinFlipProposal with details of the proposed move
        """
        k, i, j = site
        old_spin = configuration[k, i, j]
        new_spin = -old_spin  # Flip the spin
        
        # Compute energy difference
        energy_difference = self.compute_energy_difference(configuration, site)
        
        return SpinFlipProposal(
            site=site,
            old_spin=old_spin,
            new_spin=new_spin,
            energy_difference=energy_difference
        )
    
    def validate_configuration(self, configuration: np.ndarray) -> Dict[str, Any]:
        """
        Validate that the configuration is valid for 3D Ising model.
        
        Args:
            configuration: 3D spin configuration array
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'errors': []
        }
        
        # Check dimensions
        if configuration.ndim != 3:
            validation['is_valid'] = False
            validation['errors'].append(f"Expected 3D array, got {configuration.ndim}D")
            return validation
        
        # Check spin values
        unique_spins = np.unique(configuration)
        valid_spins = np.array([-1, 1])
        
        if not np.array_equal(np.sort(unique_spins), valid_spins):
            if len(unique_spins) > 2 or not all(spin in [-1, 1] for spin in unique_spins):
                validation['is_valid'] = False
                validation['errors'].append(f"Invalid spin values: {unique_spins}. Expected only -1 and +1")
        
        # Check for NaN or infinite values
        if not np.isfinite(configuration).all():
            validation['is_valid'] = False
            validation['errors'].append("Configuration contains NaN or infinite values")
        
        return validation
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the 3D Ising model.
        
        Returns:
            Dictionary with model properties and parameters
        """
        return {
            'model_name': self.model_name,
            'dimensions': self.dimensions,
            'coupling_strength': self.coupling_strength,
            'magnetic_field': self.magnetic_field,
            'theoretical_tc': self.get_theoretical_tc(),
            'theoretical_exponents': self.get_theoretical_exponents(),
            'n_neighbors': 6,
            'spin_values': [-1, 1],
            'hamiltonian': 'H = -J * Σ(s_i * s_j) - h * Σ(s_i)'
        }


class Potts3StateModel(PhysicsModel):
    """
    Q=3 Potts Model implementation with nearest-neighbor interactions.
    
    The Q=3 Potts model is defined by the Hamiltonian:
    H = -J * Σ δ(s_i, s_j)
    
    Where:
    - J is the coupling constant
    - δ(s_i, s_j) is the Kronecker delta (1 if s_i = s_j, 0 otherwise)
    - s_i are spin states ∈ {0, 1, 2}
    - The sum is over nearest neighbor pairs
    
    The Q=3 Potts model exhibits a first-order phase transition.
    """
    
    def __init__(self, coupling_strength: float = 1.0, dimensions: int = 2):
        """
        Initialize Q=3 Potts model.
        
        Args:
            coupling_strength: Coupling constant J (default: 1.0)
            dimensions: Spatial dimensions (2 or 3, default: 2)
        """
        super().__init__("Potts3State", dimensions, coupling_strength)
        self.q_states = 3  # Number of Potts states {0, 1, 2}
        
        # Theoretical values for Q=3 Potts model
        if dimensions == 2:
            # 2D Q=3 Potts model
            self.theoretical_tc = 1.005 * coupling_strength  # Critical temperature for J=1
            self.theoretical_exponents = {
                'beta': 0.125,   # First-order transition - discontinuous magnetization
                'nu': 0.833,     # Correlation length exponent
                'gamma': 0.875,  # Susceptibility exponent
                'alpha': 0.333,  # Specific heat exponent (first-order)
                'delta': 8.0,    # Critical isotherm exponent
                'eta': 0.25      # Anomalous dimension
            }
        else:  # 3D
            # 3D Q=3 Potts model
            self.theoretical_tc = 0.995 * coupling_strength  # Approximate for 3D
            self.theoretical_exponents = {
                'beta': 0.125,   # First-order transition
                'nu': 0.67,      # Correlation length exponent
                'gamma': 1.0,    # Susceptibility exponent
                'alpha': 0.5,    # Specific heat exponent
                'delta': 9.0,    # Critical isotherm exponent
                'eta': 0.1       # Anomalous dimension
            }
        
        self.logger.info(f"Initialized {dimensions}D Q=3 Potts model: J={coupling_strength}")
    
    def compute_energy(self, configuration: np.ndarray) -> float:
        """
        Compute total energy of the Q=3 Potts configuration.
        
        Uses the Hamiltonian H = -J * Σ δ(s_i, s_j)
        where the sum is over nearest neighbor pairs.
        
        Args:
            configuration: Spin configuration array with values in {0, 1, 2}
            
        Returns:
            Total energy of the system
        """
        if self.dimensions == 2:
            if configuration.ndim != 2:
                raise ValueError(f"Expected 2D configuration, got {configuration.ndim}D")
            height, width = configuration.shape
            energy = 0.0
            
            # Count each pair only once
            for i in range(height):
                for j in range(width):
                    spin = configuration[i, j]
                    
                    # Right neighbor
                    right_neighbor = configuration[i, (j + 1) % width]
                    if spin == right_neighbor:
                        energy -= self.coupling_strength
                    
                    # Down neighbor
                    down_neighbor = configuration[(i + 1) % height, j]
                    if spin == down_neighbor:
                        energy -= self.coupling_strength
        
        elif self.dimensions == 3:
            if configuration.ndim != 3:
                raise ValueError(f"Expected 3D configuration, got {configuration.ndim}D")
            depth, height, width = configuration.shape
            energy = 0.0
            
            # Count each pair only once
            for k in range(depth):
                for i in range(height):
                    for j in range(width):
                        spin = configuration[k, i, j]
                        
                        # Back neighbor (k+1)
                        back_neighbor = configuration[(k + 1) % depth, i, j]
                        if spin == back_neighbor:
                            energy -= self.coupling_strength
                        
                        # Down neighbor (i+1)
                        down_neighbor = configuration[k, (i + 1) % height, j]
                        if spin == down_neighbor:
                            energy -= self.coupling_strength
                        
                        # Right neighbor (j+1)
                        right_neighbor = configuration[k, i, (j + 1) % width]
                        if spin == right_neighbor:
                            energy -= self.coupling_strength
        
        return energy
    
    def compute_magnetization(self, configuration: np.ndarray) -> float:
        """
        Compute order parameter for Q=3 Potts model.
        
        For Potts model, we use the largest state fraction as order parameter:
        M = (n_max - n_avg) / N where n_max is the count of most frequent state
        and n_avg = N/Q is the average count per state.
        
        Args:
            configuration: Spin configuration array with values in {0, 1, 2}
            
        Returns:
            Order parameter (0 for disordered, >0 for ordered)
        """
        n_sites = np.prod(configuration.shape)
        
        # Count occurrences of each state
        state_counts = np.bincount(configuration.flatten(), minlength=self.q_states)
        
        # Order parameter: (n_max - N/Q) / (N - N/Q) = (n_max - N/Q) / (N(Q-1)/Q)
        n_max = np.max(state_counts)
        n_avg = n_sites / self.q_states
        
        if n_sites == n_avg:  # Avoid division by zero
            return 0.0
        
        order_parameter = (n_max - n_avg) / (n_sites - n_avg)
        return order_parameter
    
    def get_theoretical_tc(self) -> float:
        """
        Get theoretical critical temperature for Q=3 Potts model.
        
        Returns:
            Theoretical critical temperature
        """
        return self.theoretical_tc
    
    def get_theoretical_exponents(self) -> Dict[str, float]:
        """
        Get theoretical critical exponents for Q=3 Potts model.
        
        Returns:
            Dictionary of critical exponents
        """
        return self.theoretical_exponents.copy()
    
    def get_neighbors(self, site: Tuple[int, ...], lattice_shape: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """
        Get nearest neighbors with periodic boundary conditions.
        
        Args:
            site: Lattice site coordinates
            lattice_shape: Shape of the lattice
            
        Returns:
            List of neighbor coordinates
        """
        if self.dimensions == 2:
            i, j = site
            height, width = lattice_shape
            neighbors = [
                ((i - 1) % height, j),  # Up
                ((i + 1) % height, j),  # Down
                (i, (j - 1) % width),   # Left
                (i, (j + 1) % width)    # Right
            ]
        elif self.dimensions == 3:
            k, i, j = site
            depth, height, width = lattice_shape
            neighbors = [
                ((k - 1) % depth, i, j),  # Front
                ((k + 1) % depth, i, j),  # Back
                (k, (i - 1) % height, j), # Up
                (k, (i + 1) % height, j), # Down
                (k, i, (j - 1) % width),  # Left
                (k, i, (j + 1) % width)   # Right
            ]
        
        return neighbors
    
    def compute_energy_difference(self, configuration: np.ndarray, site: Tuple[int, ...]) -> Dict[int, float]:
        """
        Compute energy difference for changing spin at given site to each possible state.
        
        Args:
            configuration: Current spin configuration
            site: Lattice site coordinates
            
        Returns:
            Dictionary mapping new_state -> energy_difference
        """
        current_state = configuration[site]
        neighbors = self.get_neighbors(site, configuration.shape)
        
        # Count neighbors in each state
        neighbor_states = [configuration[neighbor] for neighbor in neighbors]
        neighbor_counts = np.bincount(neighbor_states, minlength=self.q_states)
        
        energy_differences = {}
        
        for new_state in range(self.q_states):
            if new_state == current_state:
                energy_differences[new_state] = 0.0
            else:
                # Energy change = -J * (new_matches - old_matches)
                old_matches = neighbor_counts[current_state]
                new_matches = neighbor_counts[new_state]
                delta_e = -self.coupling_strength * (new_matches - old_matches)
                energy_differences[new_state] = delta_e
        
        return energy_differences
    
    def propose_spin_flip(self, configuration: np.ndarray, site: Tuple[int, ...]) -> SpinFlipProposal:
        """
        Propose a spin state change at the given lattice site.
        
        For Potts model, we randomly select a new state different from current state.
        
        Args:
            configuration: Current spin configuration
            site: Lattice site coordinates
            
        Returns:
            SpinFlipProposal with details of the proposed move
        """
        old_state = configuration[site]
        
        # Choose a new state different from current state
        possible_states = [s for s in range(self.q_states) if s != old_state]
        new_state = np.random.choice(possible_states)
        
        # Compute energy difference
        energy_differences = self.compute_energy_difference(configuration, site)
        energy_difference = energy_differences[new_state]
        
        return SpinFlipProposal(
            site=site,
            old_spin=old_state,
            new_spin=new_state,
            energy_difference=energy_difference
        )
    
    def validate_configuration(self, configuration: np.ndarray) -> Dict[str, Any]:
        """
        Validate that the configuration is valid for Q=3 Potts model.
        
        Args:
            configuration: Spin configuration array
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'errors': []
        }
        
        # Check dimensions
        if configuration.ndim != self.dimensions:
            validation['is_valid'] = False
            validation['errors'].append(f"Expected {self.dimensions}D array, got {configuration.ndim}D")
            return validation
        
        # Check spin values
        unique_spins = np.unique(configuration)
        valid_spins = np.arange(self.q_states)
        
        if not np.all(np.isin(unique_spins, valid_spins)):
            validation['is_valid'] = False
            validation['errors'].append(f"Invalid spin values: {unique_spins}. Expected values in {valid_spins}")
        
        # Check for NaN or infinite values
        if not np.isfinite(configuration).all():
            validation['is_valid'] = False
            validation['errors'].append("Configuration contains NaN or infinite values")
        
        return validation
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the Q=3 Potts model.
        
        Returns:
            Dictionary with model properties and parameters
        """
        return {
            'model_name': self.model_name,
            'dimensions': self.dimensions,
            'coupling_strength': self.coupling_strength,
            'q_states': self.q_states,
            'theoretical_tc': self.get_theoretical_tc(),
            'theoretical_exponents': self.get_theoretical_exponents(),
            'n_neighbors': 4 if self.dimensions == 2 else 6,
            'spin_values': list(range(self.q_states)),
            'hamiltonian': 'H = -J * Σ δ(s_i, s_j)',
            'transition_type': 'first_order'
        }


class XY2DModel(PhysicsModel):
    """
    2D XY Model implementation with continuous O(2) spins.
    
    The 2D XY model is defined by the Hamiltonian:
    H = -J * Σ cos(θ_i - θ_j)
    
    Where:
    - J is the coupling constant
    - θ_i are angle variables ∈ [0, 2π)
    - Spins are represented as s_i = (cos θ_i, sin θ_i)
    - The sum is over nearest neighbor pairs
    
    The 2D XY model exhibits a Kosterlitz-Thouless (KT) topological transition.
    """
    
    def __init__(self, coupling_strength: float = 1.0):
        """
        Initialize 2D XY model.
        
        Args:
            coupling_strength: Coupling constant J (default: 1.0)
        """
        super().__init__("XY2D", 2, coupling_strength)
        
        # Theoretical values for 2D XY model
        # Note: KT transition doesn't have conventional critical exponents
        self.theoretical_tc = 0.893 * coupling_strength  # KT transition temperature
        self.theoretical_exponents = {
            'beta': None,    # No conventional order parameter
            'nu': None,      # No diverging correlation length
            'gamma': None,   # No diverging susceptibility
            'alpha': None,   # No specific heat divergence
            'delta': None,   # No critical isotherm
            'eta': 0.25      # Universal jump in superfluid density
        }
        
        self.logger.info(f"Initialized 2D XY model: J={coupling_strength}")
    
    def compute_energy(self, configuration: np.ndarray) -> float:
        """
        Compute total energy of the 2D XY configuration.
        
        Uses the Hamiltonian H = -J * Σ cos(θ_i - θ_j)
        where the sum is over nearest neighbor pairs.
        
        Args:
            configuration: Angle configuration array with shape (height, width)
                          Values are angles in [0, 2π)
            
        Returns:
            Total energy of the system
        """
        if configuration.ndim != 2:
            raise ValueError(f"Expected 2D configuration, got {configuration.ndim}D")
        
        height, width = configuration.shape
        energy = 0.0
        
        # Count each pair only once
        for i in range(height):
            for j in range(width):
                angle = configuration[i, j]
                
                # Right neighbor
                right_neighbor = configuration[i, (j + 1) % width]
                energy -= self.coupling_strength * np.cos(angle - right_neighbor)
                
                # Down neighbor
                down_neighbor = configuration[(i + 1) % height, j]
                energy -= self.coupling_strength * np.cos(angle - down_neighbor)
        
        return energy
    
    def compute_magnetization(self, configuration: np.ndarray) -> float:
        """
        Compute magnetization vector magnitude for 2D XY model.
        
        The magnetization vector is M = (1/N) * Σ (cos θ_i, sin θ_i)
        We return |M| as the order parameter.
        
        Args:
            configuration: Angle configuration array
            
        Returns:
            Magnitude of magnetization vector |M|
        """
        n_sites = np.prod(configuration.shape)
        
        # Compute magnetization vector components
        mx = np.sum(np.cos(configuration)) / n_sites
        my = np.sum(np.sin(configuration)) / n_sites
        
        # Return magnitude
        magnetization = np.sqrt(mx**2 + my**2)
        return magnetization
    
    def compute_vorticity(self, configuration: np.ndarray) -> float:
        """
        Compute total vorticity (topological charge) of the configuration.
        
        Vorticity is computed by summing phase differences around elementary plaquettes.
        
        Args:
            configuration: Angle configuration array
            
        Returns:
            Total vorticity of the system
        """
        height, width = configuration.shape
        total_vorticity = 0.0
        
        for i in range(height):
            for j in range(width):
                # Get angles at corners of plaquette
                theta1 = configuration[i, j]
                theta2 = configuration[i, (j + 1) % width]
                theta3 = configuration[(i + 1) % height, (j + 1) % width]
                theta4 = configuration[(i + 1) % height, j]
                
                # Compute phase differences around plaquette
                dtheta1 = self._angle_difference(theta2, theta1)
                dtheta2 = self._angle_difference(theta3, theta2)
                dtheta3 = self._angle_difference(theta4, theta3)
                dtheta4 = self._angle_difference(theta1, theta4)
                
                # Sum phase differences
                phase_sum = dtheta1 + dtheta2 + dtheta3 + dtheta4
                
                # Vorticity is phase_sum / (2π)
                vorticity = phase_sum / (2 * np.pi)
                total_vorticity += vorticity
        
        return total_vorticity
    
    def _angle_difference(self, theta2: float, theta1: float) -> float:
        """
        Compute angle difference θ2 - θ1 in range [-π, π].
        
        Args:
            theta2, theta1: Angles in [0, 2π)
            
        Returns:
            Angle difference in [-π, π]
        """
        diff = theta2 - theta1
        # Wrap to [-π, π]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff <= -np.pi:
            diff += 2 * np.pi
        return diff
    
    def get_theoretical_tc(self) -> float:
        """
        Get theoretical KT transition temperature for 2D XY model.
        
        Returns:
            Theoretical KT transition temperature
        """
        return self.theoretical_tc
    
    def get_theoretical_exponents(self) -> Dict[str, float]:
        """
        Get theoretical critical exponents for 2D XY model.
        
        Note: Most exponents are None for KT transition.
        
        Returns:
            Dictionary of critical exponents
        """
        return self.theoretical_exponents.copy()
    
    def get_neighbors(self, site: Tuple[int, int], lattice_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get 4 nearest neighbors for 2D lattice with periodic boundary conditions.
        
        Args:
            site: Lattice site coordinates (i, j)
            lattice_shape: Shape of the lattice (height, width)
            
        Returns:
            List of neighbor coordinates
        """
        i, j = site
        height, width = lattice_shape
        
        neighbors = [
            ((i - 1) % height, j),  # Up
            ((i + 1) % height, j),  # Down
            (i, (j - 1) % width),   # Left
            (i, (j + 1) % width)    # Right
        ]
        
        return neighbors
    
    def compute_energy_difference(self, configuration: np.ndarray, site: Tuple[int, int], new_angle: float) -> float:
        """
        Compute energy difference for changing angle at given site.
        
        Args:
            configuration: Current angle configuration
            site: Lattice site coordinates
            new_angle: Proposed new angle
            
        Returns:
            Energy difference ΔE = E_new - E_old
        """
        i, j = site
        old_angle = configuration[i, j]
        
        neighbors = self.get_neighbors(site, configuration.shape)
        
        old_energy = 0.0
        new_energy = 0.0
        
        for neighbor in neighbors:
            neighbor_angle = configuration[neighbor]
            
            # Old energy contribution
            old_energy -= self.coupling_strength * np.cos(old_angle - neighbor_angle)
            
            # New energy contribution
            new_energy -= self.coupling_strength * np.cos(new_angle - neighbor_angle)
        
        return new_energy - old_energy
    
    def propose_spin_flip(self, configuration: np.ndarray, site: Tuple[int, int]) -> SpinFlipProposal:
        """
        Propose an angle change at the given lattice site.
        
        For XY model, we propose a small random change to the angle.
        
        Args:
            configuration: Current angle configuration
            site: Lattice site coordinates
            
        Returns:
            SpinFlipProposal with details of the proposed move
        """
        old_angle = configuration[site]
        
        # Propose small random change (can be tuned for better acceptance rate)
        delta_angle = np.random.uniform(-0.5, 0.5)  # Small angle change
        new_angle = (old_angle + delta_angle) % (2 * np.pi)
        
        # Compute energy difference
        energy_difference = self.compute_energy_difference(configuration, site, new_angle)
        
        return SpinFlipProposal(
            site=site,
            old_spin=old_angle,
            new_spin=new_angle,
            energy_difference=energy_difference
        )
    
    def validate_configuration(self, configuration: np.ndarray) -> Dict[str, Any]:
        """
        Validate that the configuration is valid for 2D XY model.
        
        Args:
            configuration: Angle configuration array
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'errors': []
        }
        
        # Check dimensions
        if configuration.ndim != 2:
            validation['is_valid'] = False
            validation['errors'].append(f"Expected 2D array, got {configuration.ndim}D")
            return validation
        
        # Check angle values
        if not np.all((configuration >= 0) & (configuration < 2 * np.pi)):
            validation['is_valid'] = False
            validation['errors'].append("Angles must be in range [0, 2π)")
        
        # Check for NaN or infinite values
        if not np.isfinite(configuration).all():
            validation['is_valid'] = False
            validation['errors'].append("Configuration contains NaN or infinite values")
        
        return validation
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the 2D XY model.
        
        Returns:
            Dictionary with model properties and parameters
        """
        return {
            'model_name': self.model_name,
            'dimensions': self.dimensions,
            'coupling_strength': self.coupling_strength,
            'theoretical_tc': self.get_theoretical_tc(),
            'theoretical_exponents': self.get_theoretical_exponents(),
            'n_neighbors': 4,
            'spin_values': 'continuous angles [0, 2π)',
            'hamiltonian': 'H = -J * Σ cos(θ_i - θ_j)',
            'transition_type': 'kosterlitz_thouless'
        }


def create_ising_2d_model(coupling_strength: float = 1.0, magnetic_field: float = 0.0) -> Ising2DModel:
    """
    Convenience function to create a 2D Ising model.
    
    Args:
        coupling_strength: Coupling constant J (default: 1.0)
        magnetic_field: External magnetic field h (default: 0.0)
        
    Returns:
        Ising2DModel instance
    """
    return Ising2DModel(coupling_strength=coupling_strength, magnetic_field=magnetic_field)


def create_ising_3d_model(coupling_strength: float = 1.0, magnetic_field: float = 0.0) -> Ising3DModel:
    """
    Convenience function to create a 3D Ising model.
    
    Args:
        coupling_strength: Coupling constant J (default: 1.0)
        magnetic_field: External magnetic field h (default: 0.0)
        
    Returns:
        Ising3DModel instance
    """
    return Ising3DModel(coupling_strength=coupling_strength, magnetic_field=magnetic_field)


def create_potts_3state_model(coupling_strength: float = 1.0, dimensions: int = 2) -> Potts3StateModel:
    """
    Convenience function to create a Q=3 Potts model.
    
    Args:
        coupling_strength: Coupling constant J (default: 1.0)
        dimensions: Spatial dimensions (2 or 3, default: 2)
        
    Returns:
        Potts3StateModel instance
    """
    return Potts3StateModel(coupling_strength=coupling_strength, dimensions=dimensions)


def create_xy_2d_model(coupling_strength: float = 1.0) -> XY2DModel:
    """
    Convenience function to create a 2D XY model.
    
    Args:
        coupling_strength: Coupling constant J (default: 1.0)
        
    Returns:
        XY2DModel instance
    """
    return XY2DModel(coupling_strength=coupling_strength)