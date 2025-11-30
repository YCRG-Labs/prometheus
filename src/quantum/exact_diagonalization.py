"""
Exact diagonalization solver using Lanczos algorithm.

Implements efficient ground state computation for sparse Hamiltonians
using the Lanczos iterative method from scipy.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, LinearOperator
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class GroundStateResult:
    """Result of ground state computation."""
    energy: float
    state: np.ndarray
    converged: bool
    iterations: int
    residual: float


@dataclass
class SpectrumResult:
    """Result of spectrum computation."""
    energies: np.ndarray
    states: np.ndarray
    converged: bool


class ExactDiagonalizationSolver:
    """
    Exact diagonalization solver using Lanczos algorithm.
    
    Uses scipy's eigsh (ARPACK) for efficient sparse eigenvalue computation.
    Suitable for systems up to ~20 sites (Hilbert space dim ~10^6).
    """
    
    def __init__(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-12,
        random_seed: Optional[int] = None
    ):
        """
        Initialize solver.
        
        Args:
            max_iterations: Maximum Lanczos iterations
            tolerance: Convergence tolerance for eigenvalues
            random_seed: Random seed for initial vector (for reproducibility)
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)
    
    def ground_state(
        self,
        H: sparse.spmatrix,
        initial_state: Optional[np.ndarray] = None,
        return_info: bool = False
    ) -> Tuple[float, np.ndarray]:
        """
        Compute ground state energy and wavefunction using Lanczos.
        
        Args:
            H: Sparse Hamiltonian matrix
            initial_state: Initial guess for Lanczos (random if None)
            return_info: If True, return GroundStateResult with extra info
            
        Returns:
            (energy, state) tuple, or GroundStateResult if return_info=True
        """
        dim = H.shape[0]
        
        # Generate initial vector if not provided
        if initial_state is None:
            v0 = self._rng.standard_normal(dim)
            if np.iscomplexobj(H):
                v0 = v0 + 1j * self._rng.standard_normal(dim)
            v0 /= np.linalg.norm(v0)
        else:
            v0 = initial_state / np.linalg.norm(initial_state)
        
        # Use eigsh for sparse eigenvalue problem
        # which='SA' finds smallest algebraic eigenvalue (ground state)
        try:
            energies, states = eigsh(
                H,
                k=1,
                which='SA',
                v0=v0,
                maxiter=self.max_iterations,
                tol=self.tolerance,
                return_eigenvectors=True
            )
            
            energy = energies[0]
            state = states[:, 0]
            
            # Normalize state
            state = state / np.linalg.norm(state)
            
            # Compute residual ||H|ψ⟩ - E|ψ⟩||
            residual = np.linalg.norm(H @ state - energy * state)
            converged = residual < self.tolerance * 100
            
            if return_info:
                return GroundStateResult(
                    energy=energy,
                    state=state,
                    converged=converged,
                    iterations=self.max_iterations,  # eigsh doesn't expose this
                    residual=residual
                )
            
            return energy, state
            
        except Exception as e:
            raise RuntimeError(f"Lanczos failed to converge: {e}")
    
    def low_energy_spectrum(
        self,
        H: sparse.spmatrix,
        n_states: int = 10,
        initial_state: Optional[np.ndarray] = None
    ) -> SpectrumResult:
        """
        Compute lowest n_states eigenvalues and eigenvectors.
        
        Args:
            H: Sparse Hamiltonian matrix
            n_states: Number of lowest states to compute
            initial_state: Initial guess for Lanczos
            
        Returns:
            SpectrumResult with energies and states
        """
        dim = H.shape[0]
        
        # Can't request more states than dimension
        n_states = min(n_states, dim - 2)
        
        if initial_state is None:
            v0 = self._rng.standard_normal(dim)
            if np.iscomplexobj(H):
                v0 = v0 + 1j * self._rng.standard_normal(dim)
            v0 /= np.linalg.norm(v0)
        else:
            v0 = initial_state / np.linalg.norm(initial_state)
        
        try:
            energies, states = eigsh(
                H,
                k=n_states,
                which='SA',
                v0=v0,
                maxiter=self.max_iterations,
                tol=self.tolerance,
                return_eigenvectors=True
            )
            
            # Sort by energy (eigsh doesn't guarantee order)
            idx = np.argsort(energies)
            energies = energies[idx]
            states = states[:, idx]
            
            return SpectrumResult(
                energies=energies,
                states=states,
                converged=True
            )
            
        except Exception as e:
            raise RuntimeError(f"Spectrum computation failed: {e}")
    
    def energy_gap(self, H: sparse.spmatrix) -> float:
        """
        Compute energy gap between ground and first excited state.
        
        Args:
            H: Sparse Hamiltonian matrix
            
        Returns:
            Energy gap Δ = E₁ - E₀
        """
        result = self.low_energy_spectrum(H, n_states=2)
        return result.energies[1] - result.energies[0]
    
    def full_diagonalization(
        self,
        H: sparse.spmatrix
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full diagonalization (for small systems only).
        
        Warning: Only use for dim < 4096 (L ≤ 12).
        
        Args:
            H: Hamiltonian matrix
            
        Returns:
            (energies, states) tuple with all eigenvalues/vectors
        """
        dim = H.shape[0]
        if dim > 4096:
            raise ValueError(
                f"Full diagonalization not recommended for dim={dim}. "
                "Use ground_state() or low_energy_spectrum() instead."
            )
        
        # Convert to dense and use numpy
        H_dense = H.toarray()
        energies, states = np.linalg.eigh(H_dense)
        
        return energies, states
    
    def expectation_value(
        self,
        operator: sparse.spmatrix,
        state: np.ndarray
    ) -> complex:
        """
        Compute expectation value ⟨ψ|O|ψ⟩.
        
        Args:
            operator: Operator matrix
            state: Quantum state vector
            
        Returns:
            Expectation value (real if operator is Hermitian)
        """
        return np.vdot(state, operator @ state)
    
    def variance(
        self,
        operator: sparse.spmatrix,
        state: np.ndarray
    ) -> float:
        """
        Compute variance ⟨O²⟩ - ⟨O⟩².
        
        Args:
            operator: Operator matrix
            state: Quantum state vector
            
        Returns:
            Variance (always real and non-negative)
        """
        exp_O = self.expectation_value(operator, state)
        exp_O2 = self.expectation_value(operator @ operator, state)
        return np.real(exp_O2 - exp_O ** 2)
