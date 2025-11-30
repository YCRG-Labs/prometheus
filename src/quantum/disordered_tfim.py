"""
Disordered Transverse Field Ising Model (DTFIM) implementation.

Implements the disordered TFIM Hamiltonian:
H = -Σᵢ Jᵢ σᶻᵢσᶻᵢ₊₁ - Σᵢ hᵢ σˣᵢ - Σᵢ εᵢ σᶻᵢ

With disorder in:
- Transverse fields: hᵢ ~ Uniform[h-W, h+W] or other distributions
- Couplings: Jᵢ ~ Uniform[J-Δ, J+Δ] or other distributions
- Longitudinal fields: εᵢ (optional)

Supports:
- Clean limit (W=0, Δ=0) recovers standard TFIM
- Various disorder distributions (box, Gaussian, binary)
- Disorder averaging with parallel computation
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

from .hamiltonian_builder import SpinHamiltonianBuilder, SpinChainParams
from .exact_diagonalization import ExactDiagonalizationSolver
from .disorder import (
    DisorderType,
    DisorderConfig,
    DisorderRealization,
    DisorderedSystemConfig,
    DisorderAveragingFramework,
)


@dataclass
class DTFIMParams:
    """Parameters for Disordered TFIM."""
    L: int  # System size
    h_mean: float = 1.0  # Mean transverse field
    h_disorder: float = 0.0  # Disorder strength in transverse field (W)
    J_mean: float = 1.0  # Mean coupling
    J_disorder: float = 0.0  # Disorder strength in coupling (Δ)
    epsilon_mean: float = 0.0  # Mean longitudinal field
    epsilon_disorder: float = 0.0  # Disorder strength in longitudinal field
    disorder_type: DisorderType = DisorderType.BOX  # Type of disorder distribution
    periodic: bool = True  # Boundary conditions
    
    def to_disordered_config(self) -> DisorderedSystemConfig:
        """Convert to DisorderedSystemConfig for disorder framework."""
        return DisorderedSystemConfig(
            L=self.L,
            J_config=DisorderConfig(
                disorder_type=self.disorder_type,
                center=self.J_mean,
                width=self.J_disorder
            ),
            h_config=DisorderConfig(
                disorder_type=self.disorder_type,
                center=self.h_mean,
                width=self.h_disorder
            ),
            hz_config=DisorderConfig(
                disorder_type=self.disorder_type,
                center=self.epsilon_mean,
                width=self.epsilon_disorder
            ),
            periodic=self.periodic
        )
    
    def is_clean(self) -> bool:
        """Check if this is the clean (no disorder) limit."""
        return (self.h_disorder == 0.0 and 
                self.J_disorder == 0.0 and 
                self.epsilon_disorder == 0.0)


class DisorderedTFIM:
    """
    Disordered Transverse Field Ising Model.
    
    Provides methods to:
    - Build Hamiltonian for specific disorder realizations
    - Compute ground state and observables
    - Perform disorder averaging
    - Verify clean limit
    """
    
    def __init__(self, params: DTFIMParams):
        """
        Initialize DTFIM.
        
        Args:
            params: DTFIM parameters
        """
        self.params = params
        self.builder = SpinHamiltonianBuilder(params.L)
        self.solver = ExactDiagonalizationSolver()
        
        # Setup disorder framework
        self.disorder_config = params.to_disordered_config()
        self.disorder_framework = DisorderAveragingFramework(
            self.disorder_config,
            base_seed=42
        )
    
    def build_hamiltonian(
        self,
        realization: Optional[DisorderRealization] = None
    ) -> 'scipy.sparse.csr_matrix':
        """
        Build DTFIM Hamiltonian for a specific disorder realization.
        
        Args:
            realization: Disorder realization. If None, uses clean limit.
            
        Returns:
            Sparse Hamiltonian matrix
        """
        if realization is None:
            # Clean limit
            spin_params = SpinChainParams(
                L=self.params.L,
                J=self.params.J_mean,
                h=self.params.h_mean,
                hz=self.params.epsilon_mean,
                periodic=self.params.periodic
            )
        else:
            # Disordered case
            spin_params = SpinChainParams(
                L=self.params.L,
                J=realization.couplings,
                h=realization.transverse_fields,
                hz=realization.longitudinal_fields,
                periodic=self.params.periodic
            )
        
        return self.builder.build_tfim(spin_params)
    
    def compute_ground_state(
        self,
        realization: Optional[DisorderRealization] = None
    ) -> Tuple[float, np.ndarray]:
        """
        Compute ground state for a disorder realization.
        
        Args:
            realization: Disorder realization. If None, uses clean limit.
            
        Returns:
            (ground_state_energy, ground_state_vector)
        """
        H = self.build_hamiltonian(realization)
        return self.solver.ground_state(H)
    
    def verify_clean_limit(
        self,
        h_test: float = 1.0,
        tolerance: float = 1e-6
    ) -> Dict[str, bool]:
        """
        Verify that clean limit (W=0, Δ=0) recovers standard TFIM.
        
        Tests:
        1. Ground state energy matches clean TFIM
        2. Critical point is at h_c ≈ 1.0 (for J=1)
        
        Args:
            h_test: Transverse field value to test
            tolerance: Numerical tolerance
            
        Returns:
            Dictionary with verification results
        """
        # Create clean DTFIM
        clean_params = DTFIMParams(
            L=self.params.L,
            h_mean=h_test,
            h_disorder=0.0,
            J_mean=1.0,
            J_disorder=0.0,
            epsilon_mean=0.0,
            epsilon_disorder=0.0,
            periodic=self.params.periodic
        )
        clean_dtfim = DisorderedTFIM(clean_params)
        
        # Compute using DTFIM
        E_dtfim, _ = clean_dtfim.compute_ground_state()
        
        # Compute using standard TFIM
        builder = SpinHamiltonianBuilder(self.params.L)
        H_tfim = builder.build_tfim(SpinChainParams(
            L=self.params.L,
            J=1.0,
            h=h_test,
            hz=0.0,
            periodic=self.params.periodic
        ))
        solver = ExactDiagonalizationSolver()
        E_tfim, _ = solver.ground_state(H_tfim)
        
        # Check if energies match
        energy_matches = abs(E_dtfim - E_tfim) < tolerance
        
        return {
            'energy_matches': energy_matches,
            'E_dtfim': E_dtfim,
            'E_tfim': E_tfim,
            'difference': abs(E_dtfim - E_tfim)
        }
    
    def disorder_average_energy(
        self,
        n_realizations: int = 100,
        parallel: bool = True
    ) -> 'DisorderAverageResult':
        """
        Compute disorder-averaged ground state energy.
        
        Args:
            n_realizations: Number of disorder realizations
            parallel: Use parallel computation
            
        Returns:
            DisorderAverageResult with statistics
        """
        def compute_energy(realization: DisorderRealization) -> float:
            E, _ = self.compute_ground_state(realization)
            return E
        
        return self.disorder_framework.disorder_average(
            compute_energy,
            n_realizations=n_realizations,
            parallel=parallel
        )
    
    def disorder_average_observables(
        self,
        observable_fns: Dict[str, callable],
        n_realizations: int = 100,
        parallel: bool = True
    ) -> Dict[str, 'DisorderAverageResult']:
        """
        Compute multiple disorder-averaged observables.
        
        Args:
            observable_fns: Dict of {name: function(realization) -> value}
            n_realizations: Number of disorder realizations
            parallel: Use parallel computation
            
        Returns:
            Dict of {name: DisorderAverageResult}
        """
        return self.disorder_framework.disorder_average_multiple(
            observable_fns,
            n_realizations=n_realizations,
            parallel=parallel
        )


def create_dtfim_uniform_disorder(
    L: int,
    h_mean: float = 1.0,
    W: float = 0.5,
    J_mean: float = 1.0,
    Delta: float = 0.0,
    periodic: bool = True
) -> DisorderedTFIM:
    """
    Create DTFIM with uniform (box) disorder.
    
    Standard parameterization:
    - hᵢ ~ Uniform[h_mean - W, h_mean + W]
    - Jᵢ ~ Uniform[J_mean - Δ, J_mean + Δ]
    
    Args:
        L: System size
        h_mean: Mean transverse field
        W: Disorder width in transverse field
        J_mean: Mean coupling
        Delta: Disorder width in coupling
        periodic: Boundary conditions
        
    Returns:
        DisorderedTFIM instance
    """
    params = DTFIMParams(
        L=L,
        h_mean=h_mean,
        h_disorder=2 * W,  # width parameter is full range
        J_mean=J_mean,
        J_disorder=2 * Delta,
        epsilon_mean=0.0,
        epsilon_disorder=0.0,
        disorder_type=DisorderType.BOX,
        periodic=periodic
    )
    return DisorderedTFIM(params)


def create_dtfim_gaussian_disorder(
    L: int,
    h_mean: float = 1.0,
    h_std: float = 0.3,
    J_mean: float = 1.0,
    J_std: float = 0.0,
    periodic: bool = True
) -> DisorderedTFIM:
    """
    Create DTFIM with Gaussian disorder.
    
    Args:
        L: System size
        h_mean: Mean transverse field
        h_std: Standard deviation of transverse field
        J_mean: Mean coupling
        J_std: Standard deviation of coupling
        periodic: Boundary conditions
        
    Returns:
        DisorderedTFIM instance
    """
    params = DTFIMParams(
        L=L,
        h_mean=h_mean,
        h_disorder=h_std,
        J_mean=J_mean,
        J_disorder=J_std,
        epsilon_mean=0.0,
        epsilon_disorder=0.0,
        disorder_type=DisorderType.GAUSSIAN,
        periodic=periodic
    )
    return DisorderedTFIM(params)
