"""
Unified Monte Carlo Simulator for Multiple Physics Models.

This module provides a unified Monte Carlo simulator that can handle different
statistical mechanics models including Ising, Potts, and XY models through
a common interface using the PhysicsModel abstraction.
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, Union, List
from dataclasses import dataclass
import time

from ..models.physics_models import PhysicsModel, SpinFlipProposal


@dataclass
class SimulationResult:
    """Data structure for storing simulation results."""
    configurations: np.ndarray
    temperatures: np.ndarray
    order_parameters: np.ndarray
    energies: np.ndarray
    model_info: Dict[str, Any]
    simulation_metadata: Dict[str, Any]


@dataclass
class EquilibrationResult:
    """Data structure for equilibration analysis results."""
    is_equilibrated: bool
    equilibration_steps: int
    final_energy: float
    final_order_parameter: float
    energy_series: np.ndarray
    order_parameter_series: np.ndarray
    convergence_metrics: Dict[str, float]


class UnifiedMonteCarloSimulator:
    """
    Unified Monte Carlo Simulator for multiple physics models.
    
    This simulator can handle different statistical mechanics models through
    the PhysicsModel interface, providing a common simulation framework for
    Ising, Potts, XY, and other models.
    """
    
    def __init__(self, 
                 physics_model: PhysicsModel,
                 lattice_size: Union[Tuple[int, int], Tuple[int, int, int]],
                 temperature: float):
        """
        Initialize the unified Monte Carlo simulator.
        
        Args:
            physics_model: Physics model instance (Ising, Potts, XY, etc.)
            lattice_size: (height, width) for 2D or (depth, height, width) for 3D
            temperature: Temperature in units where k_B = 1
        """
        self.physics_model = physics_model
        self.lattice_size = lattice_size
        self.dimensions = len(lattice_size)
        self.temperature = temperature
        self.beta = 1.0 / temperature
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Validate dimensions match model
        if self.dimensions != physics_model.dimensions:
            raise ValueError(f"Lattice dimensions ({self.dimensions}) don't match "
                           f"model dimensions ({physics_model.dimensions})")
        
        # Initialize lattice based on model type
        self._initialize_lattice()
        
        # Statistics tracking
        self.step_count = 0
        self.accepted_moves = 0
        
        self.logger.info(f"Initialized Unified Monte Carlo simulator: "
                        f"model={physics_model.model_name}, "
                        f"lattice_size={lattice_size}, T={temperature}")
    
    def _initialize_lattice(self) -> None:
        """Initialize lattice configuration based on physics model."""
        model_name = self.physics_model.model_name
        
        if model_name in ["Ising2D", "Ising3D"]:
            # Random ±1 spins
            self.lattice = np.random.choice([-1, 1], size=self.lattice_size)
        
        elif model_name == "Potts3State":
            # Random states {0, 1, 2}
            self.lattice = np.random.choice([0, 1, 2], size=self.lattice_size)
        
        elif model_name == "XY2D":
            # Random angles [0, 2π)
            self.lattice = np.random.uniform(0, 2*np.pi, size=self.lattice_size)
        
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        self.logger.debug(f"Initialized {model_name} lattice with shape {self.lattice_size}")
    
    def metropolis_step(self) -> bool:
        """
        Perform one Metropolis Monte Carlo step.
        
        Returns:
            True if the move was accepted, False otherwise
        """
        # Randomly select a lattice site
        if self.dimensions == 2:
            site = (np.random.randint(0, self.lattice_size[0]),
                   np.random.randint(0, self.lattice_size[1]))
        else:  # 3D
            site = (np.random.randint(0, self.lattice_size[0]),
                   np.random.randint(0, self.lattice_size[1]),
                   np.random.randint(0, self.lattice_size[2]))
        
        # Get spin flip proposal from physics model
        proposal = self.physics_model.propose_spin_flip(self.lattice, site)
        
        # Metropolis acceptance criterion
        if proposal.energy_difference <= 0:
            # Always accept moves that decrease energy
            accept = True
        else:
            # Accept with probability exp(-β * ΔE)
            acceptance_prob = np.exp(-self.beta * proposal.energy_difference)
            accept = np.random.random() < acceptance_prob
        
        # Apply the move if accepted
        if accept:
            self.lattice[site] = proposal.new_spin
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
    
    def equilibrate(self, 
                   n_steps: int = 10000,
                   convergence_threshold: float = 1e-4,
                   check_interval: int = 1000) -> EquilibrationResult:
        """
        Equilibrate the system and monitor convergence.
        
        Args:
            n_steps: Maximum number of equilibration steps
            convergence_threshold: Threshold for energy convergence
            check_interval: Interval for checking convergence
            
        Returns:
            EquilibrationResult with equilibration analysis
        """
        energy_series = []
        order_parameter_series = []
        
        self.logger.info(f"Starting equilibration: max_steps={n_steps}, "
                        f"threshold={convergence_threshold}")
        
        for step in range(0, n_steps, check_interval):
            # Perform sweeps
            for _ in range(check_interval):
                self.sweep()
            
            # Calculate current properties
            current_energy = self.physics_model.compute_energy(self.lattice)
            current_order_param = self.physics_model.compute_magnetization(self.lattice)
            
            energy_series.append(current_energy)
            order_parameter_series.append(current_order_param)
            
            # Check convergence (simple moving average)
            if len(energy_series) >= 5:
                recent_energies = energy_series[-5:]
                energy_std = np.std(recent_energies)
                energy_mean = np.mean(recent_energies)
                
                if energy_mean != 0:
                    relative_std = energy_std / abs(energy_mean)
                    if relative_std < convergence_threshold:
                        self.logger.info(f"Equilibration converged at step {step + check_interval}")
                        return EquilibrationResult(
                            is_equilibrated=True,
                            equilibration_steps=step + check_interval,
                            final_energy=current_energy,
                            final_order_parameter=current_order_param,
                            energy_series=np.array(energy_series),
                            order_parameter_series=np.array(order_parameter_series),
                            convergence_metrics={'relative_energy_std': relative_std}
                        )
        
        # If we reach here, equilibration didn't converge
        self.logger.warning(f"Equilibration did not converge within {n_steps} steps")
        return EquilibrationResult(
            is_equilibrated=False,
            equilibration_steps=n_steps,
            final_energy=energy_series[-1] if energy_series else 0.0,
            final_order_parameter=order_parameter_series[-1] if order_parameter_series else 0.0,
            energy_series=np.array(energy_series),
            order_parameter_series=np.array(order_parameter_series),
            convergence_metrics={'relative_energy_std': float('inf')}
        )
    
    def generate_configurations(self,
                              n_configs: int = 1000,
                              sampling_interval: int = 100,
                              equilibration_steps: int = 10000) -> List[np.ndarray]:
        """
        Generate equilibrium configurations.
        
        Args:
            n_configs: Number of configurations to generate
            sampling_interval: Steps between configuration samples
            equilibration_steps: Initial equilibration steps
            
        Returns:
            List of configuration arrays
        """
        self.logger.info(f"Generating {n_configs} configurations with "
                        f"sampling_interval={sampling_interval}")
        
        # Equilibrate first
        if equilibration_steps > 0:
            self.equilibrate(n_steps=equilibration_steps)
        
        configurations = []
        
        for i in range(n_configs):
            # Perform sampling interval sweeps
            for _ in range(sampling_interval):
                self.sweep()
            
            # Store configuration
            configurations.append(self.lattice.copy())
            
            if (i + 1) % 100 == 0:
                self.logger.debug(f"Generated {i + 1}/{n_configs} configurations")
        
        return configurations
    
    def simulate_temperature_series(self,
                                  temperature_range: Tuple[float, float],
                                  n_temperatures: int = 50,
                                  n_configs_per_temp: int = 1000,
                                  sampling_interval: int = 100,
                                  equilibration_steps: int = 10000) -> SimulationResult:
        """
        Simulate across a range of temperatures.
        
        Args:
            temperature_range: (T_min, T_max) temperature range
            n_temperatures: Number of temperature points
            n_configs_per_temp: Configurations per temperature
            sampling_interval: Steps between samples
            equilibration_steps: Equilibration steps per temperature
            
        Returns:
            SimulationResult with all data
        """
        t_min, t_max = temperature_range
        temperatures = np.linspace(t_min, t_max, n_temperatures)
        
        all_configurations = []
        all_order_parameters = []
        all_energies = []
        
        self.logger.info(f"Starting temperature series simulation: "
                        f"T ∈ [{t_min:.3f}, {t_max:.3f}], {n_temperatures} points")
        
        for i, temp in enumerate(temperatures):
            self.set_temperature(temp)
            
            # Generate configurations at this temperature
            configs = self.generate_configurations(
                n_configs=n_configs_per_temp,
                sampling_interval=sampling_interval,
                equilibration_steps=equilibration_steps
            )
            
            # Calculate properties for each configuration
            temp_order_params = []
            temp_energies = []
            
            for config in configs:
                order_param = self.physics_model.compute_magnetization(config)
                energy = self.physics_model.compute_energy(config)
                temp_order_params.append(order_param)
                temp_energies.append(energy)
            
            all_configurations.extend(configs)
            all_order_parameters.extend(temp_order_params)
            all_energies.extend(temp_energies)
            
            self.logger.info(f"Completed temperature {i+1}/{n_temperatures}: "
                           f"T={temp:.3f}, <M>={np.mean(temp_order_params):.4f}")
        
        # Create temperature array matching configurations
        temp_array = np.repeat(temperatures, n_configs_per_temp)
        
        return SimulationResult(
            configurations=np.array(all_configurations),
            temperatures=temp_array,
            order_parameters=np.array(all_order_parameters),
            energies=np.array(all_energies),
            model_info=self.physics_model.get_model_info(),
            simulation_metadata={
                'temperature_range': temperature_range,
                'n_temperatures': n_temperatures,
                'n_configs_per_temp': n_configs_per_temp,
                'sampling_interval': sampling_interval,
                'equilibration_steps': equilibration_steps,
                'lattice_size': self.lattice_size,
                'total_configurations': len(all_configurations)
            }
        )
    
    def set_temperature(self, temperature: float) -> None:
        """
        Change the temperature and update related parameters.
        
        Args:
            temperature: New temperature value
        """
        self.temperature = temperature
        self.beta = 1.0 / temperature
        self.logger.debug(f"Temperature changed to {temperature}")
    
    def get_acceptance_rate(self) -> float:
        """Get the current acceptance rate."""
        if self.step_count == 0:
            return 0.0
        return self.accepted_moves / self.step_count
    
    def reset_statistics(self) -> None:
        """Reset step count and acceptance statistics."""
        self.step_count = 0
        self.accepted_moves = 0
    
    def randomize_lattice(self) -> None:
        """Randomize the lattice configuration."""
        self._initialize_lattice()
        self.reset_statistics()
        self.logger.debug("Lattice randomized")
    
    def get_current_properties(self) -> Dict[str, float]:
        """
        Get current thermodynamic properties.
        
        Returns:
            Dictionary with current properties
        """
        return {
            'temperature': self.temperature,
            'energy': self.physics_model.compute_energy(self.lattice),
            'order_parameter': self.physics_model.compute_magnetization(self.lattice),
            'acceptance_rate': self.get_acceptance_rate(),
            'step_count': self.step_count
        }
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the current lattice configuration.
        
        Returns:
            Dictionary with validation results
        """
        return self.physics_model.validate_configuration(self.lattice)
    
    def get_simulation_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the simulation.
        
        Returns:
            Dictionary with simulation properties
        """
        return {
            'physics_model': self.physics_model.get_model_info(),
            'lattice_size': self.lattice_size,
            'dimensions': self.dimensions,
            'temperature': self.temperature,
            'current_properties': self.get_current_properties(),
            'validation': self.validate_configuration()
        }


def create_potts_simulator(lattice_size: Union[Tuple[int, int], Tuple[int, int, int]],
                          temperature: float,
                          coupling_strength: float = 1.0) -> UnifiedMonteCarloSimulator:
    """
    Convenience function to create a Potts model simulator.
    
    Args:
        lattice_size: Lattice dimensions
        temperature: Temperature value
        coupling_strength: Coupling constant J
        
    Returns:
        UnifiedMonteCarloSimulator configured for Potts model
    """
    from ..models.physics_models import create_potts_3state_model
    
    dimensions = len(lattice_size)
    potts_model = create_potts_3state_model(coupling_strength=coupling_strength, 
                                           dimensions=dimensions)
    
    return UnifiedMonteCarloSimulator(potts_model, lattice_size, temperature)


def create_xy_simulator(lattice_size: Tuple[int, int],
                       temperature: float,
                       coupling_strength: float = 1.0) -> UnifiedMonteCarloSimulator:
    """
    Convenience function to create a 2D XY model simulator.
    
    Args:
        lattice_size: 2D lattice dimensions (height, width)
        temperature: Temperature value
        coupling_strength: Coupling constant J
        
    Returns:
        UnifiedMonteCarloSimulator configured for XY model
    """
    from ..models.physics_models import create_xy_2d_model
    
    if len(lattice_size) != 2:
        raise ValueError("XY model only supports 2D lattices")
    
    xy_model = create_xy_2d_model(coupling_strength=coupling_strength)
    
    return UnifiedMonteCarloSimulator(xy_model, lattice_size, temperature)


def create_ising_simulator(lattice_size: Union[Tuple[int, int], Tuple[int, int, int]],
                          temperature: float,
                          coupling_strength: float = 1.0,
                          magnetic_field: float = 0.0) -> UnifiedMonteCarloSimulator:
    """
    Convenience function to create an Ising model simulator.
    
    Args:
        lattice_size: Lattice dimensions (2D or 3D)
        temperature: Temperature value
        coupling_strength: Coupling constant J
        magnetic_field: External magnetic field h
        
    Returns:
        UnifiedMonteCarloSimulator configured for Ising model
    """
    from ..models.physics_models import create_ising_2d_model, create_ising_3d_model
    
    dimensions = len(lattice_size)
    
    if dimensions == 2:
        ising_model = create_ising_2d_model(coupling_strength=coupling_strength,
                                           magnetic_field=magnetic_field)
    elif dimensions == 3:
        ising_model = create_ising_3d_model(coupling_strength=coupling_strength,
                                           magnetic_field=magnetic_field)
    else:
        raise ValueError(f"Unsupported dimensions: {dimensions}. Only 2D and 3D are supported.")
    
    return UnifiedMonteCarloSimulator(ising_model, lattice_size, temperature)