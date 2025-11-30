"""
Observable calculations for quantum spin systems.

Computes local observables (σx, σy, σz) and correlation functions ⟨σᵢσⱼ⟩.
"""

import numpy as np
from scipy import sparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .hamiltonian_builder import SpinHamiltonianBuilder


@dataclass
class LocalObservables:
    """Local observable measurements at each site."""
    sigma_x: np.ndarray  # ⟨σˣᵢ⟩ for each site
    sigma_y: np.ndarray  # ⟨σʸᵢ⟩ for each site
    sigma_z: np.ndarray  # ⟨σᶻᵢ⟩ for each site
    
    @property
    def magnetization_x(self) -> float:
        """Total x-magnetization per site."""
        return np.mean(self.sigma_x)
    
    @property
    def magnetization_y(self) -> float:
        """Total y-magnetization per site."""
        return np.mean(self.sigma_y)
    
    @property
    def magnetization_z(self) -> float:
        """Total z-magnetization per site."""
        return np.mean(self.sigma_z)
    
    @property
    def magnetization(self) -> float:
        """Total magnetization magnitude per site."""
        return np.sqrt(
            self.magnetization_x**2 + 
            self.magnetization_y**2 + 
            self.magnetization_z**2
        )


@dataclass
class CorrelationFunctions:
    """Two-point correlation functions."""
    zz: np.ndarray  # ⟨σᶻᵢσᶻⱼ⟩ matrix
    xx: np.ndarray  # ⟨σˣᵢσˣⱼ⟩ matrix
    yy: np.ndarray  # ⟨σʸᵢσʸⱼ⟩ matrix
    
    @property
    def connected_zz(self) -> np.ndarray:
        """Connected correlation ⟨σᶻᵢσᶻⱼ⟩ - ⟨σᶻᵢ⟩⟨σᶻⱼ⟩."""
        L = self.zz.shape[0]
        # Need local observables for this - compute diagonal
        sz = np.diag(self.zz)
        return self.zz - np.outer(sz, sz)


class ObservableCalculator:
    """
    Calculate observables for quantum spin states.
    
    Provides efficient computation of:
    - Local observables ⟨σˣᵢ⟩, ⟨σʸᵢ⟩, ⟨σᶻᵢ⟩
    - Correlation functions ⟨σᵢσⱼ⟩
    - Susceptibilities and other derived quantities
    """
    
    def __init__(self, L: int):
        """
        Initialize calculator for L-site system.
        
        Args:
            L: Number of sites
        """
        self.L = L
        self.builder = SpinHamiltonianBuilder(L)
        
        # Cache operators for efficiency
        self._sigma_x_cache: Dict[int, sparse.csr_matrix] = {}
        self._sigma_y_cache: Dict[int, sparse.csr_matrix] = {}
        self._sigma_z_cache: Dict[int, sparse.csr_matrix] = {}
    
    def _get_sigma_x(self, site: int) -> sparse.csr_matrix:
        """Get cached σˣ operator."""
        if site not in self._sigma_x_cache:
            self._sigma_x_cache[site] = self.builder.build_sigma_x(site)
        return self._sigma_x_cache[site]
    
    def _get_sigma_y(self, site: int) -> sparse.csr_matrix:
        """Get cached σʸ operator."""
        if site not in self._sigma_y_cache:
            self._sigma_y_cache[site] = self.builder.build_sigma_y(site)
        return self._sigma_y_cache[site]
    
    def _get_sigma_z(self, site: int) -> sparse.csr_matrix:
        """Get cached σᶻ operator."""
        if site not in self._sigma_z_cache:
            self._sigma_z_cache[site] = self.builder.build_sigma_z(site)
        return self._sigma_z_cache[site]
    
    def expectation(
        self, 
        operator: sparse.spmatrix, 
        state: np.ndarray
    ) -> complex:
        """Compute ⟨ψ|O|ψ⟩."""
        return np.vdot(state, operator @ state)
    
    def local_observables(self, state: np.ndarray) -> LocalObservables:
        """
        Compute all local observables ⟨σˣᵢ⟩, ⟨σʸᵢ⟩, ⟨σᶻᵢ⟩.
        
        Args:
            state: Quantum state vector
            
        Returns:
            LocalObservables with measurements at each site
        """
        sigma_x = np.zeros(self.L)
        sigma_y = np.zeros(self.L, dtype=complex)
        sigma_z = np.zeros(self.L)
        
        for i in range(self.L):
            sigma_x[i] = np.real(self.expectation(self._get_sigma_x(i), state))
            sigma_y[i] = self.expectation(self._get_sigma_y(i), state)
            sigma_z[i] = np.real(self.expectation(self._get_sigma_z(i), state))
        
        # σʸ expectation should be real for real Hamiltonians
        sigma_y = np.real(sigma_y)
        
        return LocalObservables(
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            sigma_z=sigma_z
        )
    
    def correlation_zz(
        self, 
        state: np.ndarray,
        sites: Optional[List[Tuple[int, int]]] = None
    ) -> np.ndarray:
        """
        Compute ⟨σᶻᵢσᶻⱼ⟩ correlation function.
        
        Args:
            state: Quantum state vector
            sites: List of (i, j) pairs, or None for all pairs
            
        Returns:
            L×L matrix of correlations (or values for specified pairs)
        """
        if sites is None:
            # Compute full correlation matrix
            corr = np.zeros((self.L, self.L))
            for i in range(self.L):
                for j in range(i, self.L):
                    if i == j:
                        # ⟨σᶻᵢσᶻᵢ⟩ = 1 always
                        corr[i, i] = 1.0
                    else:
                        op = self.builder.build_zz_interaction(i, j)
                        corr[i, j] = np.real(self.expectation(op, state))
                        corr[j, i] = corr[i, j]
            return corr
        else:
            # Compute only specified pairs
            values = []
            for i, j in sites:
                if i == j:
                    values.append(1.0)
                else:
                    op = self.builder.build_zz_interaction(i, j)
                    values.append(np.real(self.expectation(op, state)))
            return np.array(values)
    
    def correlation_xx(
        self, 
        state: np.ndarray,
        sites: Optional[List[Tuple[int, int]]] = None
    ) -> np.ndarray:
        """
        Compute ⟨σˣᵢσˣⱼ⟩ correlation function.
        
        Args:
            state: Quantum state vector
            sites: List of (i, j) pairs, or None for all pairs
            
        Returns:
            L×L matrix of correlations
        """
        if sites is None:
            corr = np.zeros((self.L, self.L))
            for i in range(self.L):
                for j in range(i, self.L):
                    if i == j:
                        corr[i, i] = 1.0
                    else:
                        op = self.builder.build_xx_interaction(i, j)
                        corr[i, j] = np.real(self.expectation(op, state))
                        corr[j, i] = corr[i, j]
            return corr
        else:
            values = []
            for i, j in sites:
                if i == j:
                    values.append(1.0)
                else:
                    op = self.builder.build_xx_interaction(i, j)
                    values.append(np.real(self.expectation(op, state)))
            return np.array(values)
    
    def correlation_yy(
        self, 
        state: np.ndarray,
        sites: Optional[List[Tuple[int, int]]] = None
    ) -> np.ndarray:
        """
        Compute ⟨σʸᵢσʸⱼ⟩ correlation function.
        
        Args:
            state: Quantum state vector
            sites: List of (i, j) pairs, or None for all pairs
            
        Returns:
            L×L matrix of correlations
        """
        if sites is None:
            corr = np.zeros((self.L, self.L))
            for i in range(self.L):
                for j in range(i, self.L):
                    if i == j:
                        corr[i, i] = 1.0
                    else:
                        op = self.builder.build_yy_interaction(i, j)
                        corr[i, j] = np.real(self.expectation(op, state))
                        corr[j, i] = corr[i, j]
            return corr
        else:
            values = []
            for i, j in sites:
                if i == j:
                    values.append(1.0)
                else:
                    op = self.builder.build_yy_interaction(i, j)
                    values.append(np.real(self.expectation(op, state)))
            return np.array(values)
    
    def all_correlations(self, state: np.ndarray) -> CorrelationFunctions:
        """
        Compute all two-point correlation functions.
        
        Args:
            state: Quantum state vector
            
        Returns:
            CorrelationFunctions with zz, xx, yy matrices
        """
        return CorrelationFunctions(
            zz=self.correlation_zz(state),
            xx=self.correlation_xx(state),
            yy=self.correlation_yy(state)
        )
    
    def correlation_length(
        self, 
        state: np.ndarray,
        correlation_type: str = 'zz'
    ) -> float:
        """
        Estimate correlation length from exponential decay.
        
        Fits |⟨σᵢσⱼ⟩| ~ exp(-|i-j|/ξ) for large |i-j|.
        
        Args:
            state: Quantum state vector
            correlation_type: 'zz', 'xx', or 'yy'
            
        Returns:
            Estimated correlation length ξ
        """
        if correlation_type == 'zz':
            corr = self.correlation_zz(state)
        elif correlation_type == 'xx':
            corr = self.correlation_xx(state)
        elif correlation_type == 'yy':
            corr = self.correlation_yy(state)
        else:
            raise ValueError(f"Unknown correlation type: {correlation_type}")
        
        # Get correlation vs distance (use site 0 as reference)
        distances = np.arange(1, self.L // 2 + 1)
        correlations = np.abs([corr[0, d] for d in distances])
        
        # Avoid log(0)
        mask = correlations > 1e-15
        if np.sum(mask) < 2:
            return float('inf')  # No decay detected
        
        # Linear fit to log(|C(r)|) = -r/ξ + const
        log_corr = np.log(correlations[mask])
        r = distances[mask]
        
        # Simple linear regression
        slope, _ = np.polyfit(r, log_corr, 1)
        
        if slope >= 0:
            return float('inf')  # No decay
        
        return -1.0 / slope
    
    def susceptibility(
        self, 
        state: np.ndarray,
        direction: str = 'z'
    ) -> float:
        """
        Compute magnetic susceptibility χ = Σᵢⱼ ⟨σᵢσⱼ⟩.
        
        Args:
            state: Quantum state vector
            direction: 'x', 'y', or 'z'
            
        Returns:
            Susceptibility per site
        """
        if direction == 'z':
            corr = self.correlation_zz(state)
        elif direction == 'x':
            corr = self.correlation_xx(state)
        elif direction == 'y':
            corr = self.correlation_yy(state)
        else:
            raise ValueError(f"Unknown direction: {direction}")
        
        return np.sum(corr) / self.L
    
    def structure_factor(
        self, 
        state: np.ndarray,
        k: float,
        correlation_type: str = 'zz'
    ) -> float:
        """
        Compute structure factor S(k) = Σᵢⱼ exp(ik(i-j)) ⟨σᵢσⱼ⟩.
        
        Args:
            state: Quantum state vector
            k: Wavevector (in units of 2π/L)
            correlation_type: 'zz', 'xx', or 'yy'
            
        Returns:
            Structure factor at wavevector k
        """
        if correlation_type == 'zz':
            corr = self.correlation_zz(state)
        elif correlation_type == 'xx':
            corr = self.correlation_xx(state)
        elif correlation_type == 'yy':
            corr = self.correlation_yy(state)
        else:
            raise ValueError(f"Unknown correlation type: {correlation_type}")
        
        S_k = 0.0
        for i in range(self.L):
            for j in range(self.L):
                S_k += np.exp(1j * k * (i - j)) * corr[i, j]
        
        return np.real(S_k) / self.L
