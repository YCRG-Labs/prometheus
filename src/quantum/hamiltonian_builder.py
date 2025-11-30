"""
Sparse Hamiltonian builder for spin-1/2 chains.

Implements efficient sparse matrix construction for quantum spin systems
using the computational basis |↑↑...↑⟩, |↑↑...↓⟩, etc.
"""

import numpy as np
from scipy import sparse
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass


@dataclass
class SpinChainParams:
    """Parameters for a spin-1/2 chain Hamiltonian."""
    L: int  # Number of sites
    J: Union[float, np.ndarray] = 1.0  # Coupling strength (can be site-dependent)
    h: Union[float, np.ndarray] = 0.0  # Transverse field (can be site-dependent)
    hz: Union[float, np.ndarray] = 0.0  # Longitudinal field (can be site-dependent)
    periodic: bool = True  # Periodic boundary conditions


class SpinHamiltonianBuilder:
    """
    Build sparse Hamiltonians for spin-1/2 chains.
    
    Uses computational basis where state |n⟩ corresponds to binary representation
    of n, with bit i representing spin at site i (0=↓, 1=↑).
    
    Supports:
    - Transverse Field Ising Model (TFIM)
    - XXZ model
    - Custom Hamiltonians with arbitrary terms
    """
    
    def __init__(self, L: int):
        """
        Initialize builder for L-site spin chain.
        
        Args:
            L: Number of sites in the chain
        """
        if L < 2:
            raise ValueError("Chain length must be at least 2")
        if L > 20:
            raise ValueError("Chain length > 20 not supported (Hilbert space too large)")
        
        self.L = L
        self.dim = 2 ** L  # Hilbert space dimension
        
        # Precompute bit masks for efficiency
        self._bit_masks = [1 << i for i in range(L)]
    
    def _get_spin(self, state: int, site: int) -> int:
        """Get spin value at site (0 or 1) for given basis state."""
        return (state >> site) & 1
    
    def _flip_spin(self, state: int, site: int) -> int:
        """Flip spin at site and return new state."""
        return state ^ self._bit_masks[site]
    
    def build_sigma_z(self, site: int) -> sparse.csr_matrix:
        """
        Build σᶻ operator at given site.
        
        σᶻ|↑⟩ = +|↑⟩, σᶻ|↓⟩ = -|↓⟩
        
        Args:
            site: Site index (0 to L-1)
            
        Returns:
            Sparse matrix representation of σᶻ
        """
        if not 0 <= site < self.L:
            raise ValueError(f"Site {site} out of range [0, {self.L-1}]")
        
        diag = np.array([
            2 * self._get_spin(n, site) - 1 
            for n in range(self.dim)
        ], dtype=np.float64)
        
        return sparse.diags(diag, format='csr')
    
    def build_sigma_x(self, site: int) -> sparse.csr_matrix:
        """
        Build σˣ operator at given site.
        
        σˣ|↑⟩ = |↓⟩, σˣ|↓⟩ = |↑⟩
        
        Args:
            site: Site index (0 to L-1)
            
        Returns:
            Sparse matrix representation of σˣ
        """
        if not 0 <= site < self.L:
            raise ValueError(f"Site {site} out of range [0, {self.L-1}]")
        
        rows = []
        cols = []
        data = []
        
        for n in range(self.dim):
            m = self._flip_spin(n, site)
            rows.append(m)
            cols.append(n)
            data.append(1.0)
        
        return sparse.csr_matrix(
            (data, (rows, cols)), 
            shape=(self.dim, self.dim),
            dtype=np.float64
        )
    
    def build_sigma_y(self, site: int) -> sparse.csr_matrix:
        """
        Build σʸ operator at given site.
        
        σʸ = -i|↓⟩⟨↑| + i|↑⟩⟨↓|
        
        Args:
            site: Site index (0 to L-1)
            
        Returns:
            Sparse matrix representation of σʸ (complex)
        """
        if not 0 <= site < self.L:
            raise ValueError(f"Site {site} out of range [0, {self.L-1}]")
        
        rows = []
        cols = []
        data = []
        
        for n in range(self.dim):
            m = self._flip_spin(n, site)
            spin_n = self._get_spin(n, site)
            # If spin is up (1), we get -i; if down (0), we get +i
            phase = -1j if spin_n == 1 else 1j
            rows.append(m)
            cols.append(n)
            data.append(phase)
        
        return sparse.csr_matrix(
            (data, (rows, cols)), 
            shape=(self.dim, self.dim),
            dtype=np.complex128
        )
    
    def build_sigma_plus(self, site: int) -> sparse.csr_matrix:
        """Build σ⁺ = (σˣ + iσʸ)/2 raising operator."""
        return (self.build_sigma_x(site) + 1j * self.build_sigma_y(site)) / 2
    
    def build_sigma_minus(self, site: int) -> sparse.csr_matrix:
        """Build σ⁻ = (σˣ - iσʸ)/2 lowering operator."""
        return (self.build_sigma_x(site) - 1j * self.build_sigma_y(site)) / 2

    def build_zz_interaction(self, site1: int, site2: int) -> sparse.csr_matrix:
        """
        Build σᶻᵢσᶻⱼ interaction term.
        
        Args:
            site1, site2: Site indices
            
        Returns:
            Sparse matrix for σᶻᵢσᶻⱼ
        """
        diag = np.array([
            (2 * self._get_spin(n, site1) - 1) * (2 * self._get_spin(n, site2) - 1)
            for n in range(self.dim)
        ], dtype=np.float64)
        
        return sparse.diags(diag, format='csr')
    
    def build_xx_interaction(self, site1: int, site2: int) -> sparse.csr_matrix:
        """
        Build σˣᵢσˣⱼ interaction term.
        
        Args:
            site1, site2: Site indices
            
        Returns:
            Sparse matrix for σˣᵢσˣⱼ
        """
        rows = []
        cols = []
        data = []
        
        for n in range(self.dim):
            # Flip both spins
            m = self._flip_spin(self._flip_spin(n, site1), site2)
            rows.append(m)
            cols.append(n)
            data.append(1.0)
        
        return sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(self.dim, self.dim),
            dtype=np.float64
        )
    
    def build_yy_interaction(self, site1: int, site2: int) -> sparse.csr_matrix:
        """
        Build σʸᵢσʸⱼ interaction term.
        
        Args:
            site1, site2: Site indices
            
        Returns:
            Sparse matrix for σʸᵢσʸⱼ
        """
        rows = []
        cols = []
        data = []
        
        for n in range(self.dim):
            m = self._flip_spin(self._flip_spin(n, site1), site2)
            spin1 = self._get_spin(n, site1)
            spin2 = self._get_spin(n, site2)
            # Phase from σʸ: -i if up, +i if down
            phase1 = -1 if spin1 == 1 else 1
            phase2 = -1 if spin2 == 1 else 1
            # Product of phases (i * i = -1 cancels the signs)
            rows.append(m)
            cols.append(n)
            data.append(-phase1 * phase2)  # -1 from i*i
        
        return sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(self.dim, self.dim),
            dtype=np.float64
        )
    
    def build_tfim(self, params: SpinChainParams) -> sparse.csr_matrix:
        """
        Build Transverse Field Ising Model Hamiltonian.
        
        H = -Σᵢ Jᵢ σᶻᵢσᶻᵢ₊₁ - Σᵢ hᵢ σˣᵢ - Σᵢ hzᵢ σᶻᵢ
        
        Args:
            params: SpinChainParams with J, h, hz fields
            
        Returns:
            Sparse Hamiltonian matrix
        """
        if params.L != self.L:
            raise ValueError(f"Params L={params.L} doesn't match builder L={self.L}")
        
        # Convert scalar to array if needed
        J = np.atleast_1d(params.J)
        h = np.atleast_1d(params.h)
        hz = np.atleast_1d(params.hz)
        
        # Broadcast to full length if scalar
        if len(J) == 1:
            J = np.full(self.L, J[0])
        if len(h) == 1:
            h = np.full(self.L, h[0])
        if len(hz) == 1:
            hz = np.full(self.L, hz[0])
        
        # Initialize Hamiltonian
        H = sparse.csr_matrix((self.dim, self.dim), dtype=np.float64)
        
        # ZZ interactions
        n_bonds = self.L if params.periodic else self.L - 1
        for i in range(n_bonds):
            j = (i + 1) % self.L
            H = H - J[i] * self.build_zz_interaction(i, j)
        
        # Transverse field (σˣ)
        for i in range(self.L):
            if h[i] != 0:
                H = H - h[i] * self.build_sigma_x(i)
        
        # Longitudinal field (σᶻ)
        for i in range(self.L):
            if hz[i] != 0:
                H = H - hz[i] * self.build_sigma_z(i)
        
        return H
    
    def build_xxz(self, params: SpinChainParams, delta: float = 1.0) -> sparse.csr_matrix:
        """
        Build XXZ model Hamiltonian.
        
        H = Σᵢ J(σˣᵢσˣᵢ₊₁ + σʸᵢσʸᵢ₊₁ + Δσᶻᵢσᶻᵢ₊₁) - Σᵢ hᵢ σᶻᵢ
        
        Args:
            params: SpinChainParams
            delta: Anisotropy parameter Δ
            
        Returns:
            Sparse Hamiltonian matrix
        """
        if params.L != self.L:
            raise ValueError(f"Params L={params.L} doesn't match builder L={self.L}")
        
        J = np.atleast_1d(params.J)
        h = np.atleast_1d(params.h)
        
        if len(J) == 1:
            J = np.full(self.L, J[0])
        if len(h) == 1:
            h = np.full(self.L, h[0])
        
        H = sparse.csr_matrix((self.dim, self.dim), dtype=np.float64)
        
        n_bonds = self.L if params.periodic else self.L - 1
        for i in range(n_bonds):
            j = (i + 1) % self.L
            H = H + J[i] * (
                self.build_xx_interaction(i, j) +
                self.build_yy_interaction(i, j) +
                delta * self.build_zz_interaction(i, j)
            )
        
        # External field
        for i in range(self.L):
            if h[i] != 0:
                H = H - h[i] * self.build_sigma_z(i)
        
        return H
    
    def build_heisenberg(self, params: SpinChainParams) -> sparse.csr_matrix:
        """Build isotropic Heisenberg model (XXZ with Δ=1)."""
        return self.build_xxz(params, delta=1.0)
    
    def build_custom(
        self, 
        terms: List[Tuple[str, List[int], float]]
    ) -> sparse.csr_matrix:
        """
        Build custom Hamiltonian from list of terms.
        
        Args:
            terms: List of (operator_type, sites, coefficient)
                   operator_type: 'X', 'Y', 'Z', 'XX', 'YY', 'ZZ', 'XY', etc.
                   sites: List of site indices
                   coefficient: Numerical coefficient
        
        Returns:
            Sparse Hamiltonian matrix
        """
        H = sparse.csr_matrix((self.dim, self.dim), dtype=np.complex128)
        
        op_map = {
            'X': self.build_sigma_x,
            'Y': self.build_sigma_y,
            'Z': self.build_sigma_z,
        }
        
        for op_type, sites, coeff in terms:
            if len(op_type) == 1:
                # Single-site operator
                H = H + coeff * op_map[op_type](sites[0])
            elif len(op_type) == 2:
                # Two-site operator
                if op_type == 'XX':
                    H = H + coeff * self.build_xx_interaction(sites[0], sites[1])
                elif op_type == 'YY':
                    H = H + coeff * self.build_yy_interaction(sites[0], sites[1])
                elif op_type == 'ZZ':
                    H = H + coeff * self.build_zz_interaction(sites[0], sites[1])
                else:
                    # General two-site: build as product
                    op1 = op_map[op_type[0]](sites[0])
                    op2 = op_map[op_type[1]](sites[1])
                    H = H + coeff * (op1 @ op2)
        
        # Convert to real if possible
        if np.allclose(H.toarray().imag, 0):
            H = H.real
        
        return sparse.csr_matrix(H)
