"""
Entanglement calculations for quantum spin systems.

Computes von Neumann entanglement entropy and related quantities
for bipartitions of spin chains.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class EntanglementResult:
    """Result of entanglement calculation."""
    entropy: float  # von Neumann entropy S = -Tr(ρ log ρ)
    schmidt_values: np.ndarray  # Schmidt coefficients
    schmidt_rank: int  # Number of non-zero Schmidt values
    subsystem_size: int  # Size of subsystem A


@dataclass
class EntanglementSpectrum:
    """Full entanglement spectrum."""
    eigenvalues: np.ndarray  # Eigenvalues of reduced density matrix
    entanglement_energies: np.ndarray  # ξᵢ = -log(λᵢ)
    entropy: float


class EntanglementCalculator:
    """
    Calculate entanglement properties of quantum states.
    
    Computes:
    - von Neumann entanglement entropy
    - Rényi entropies
    - Entanglement spectrum
    - Schmidt decomposition
    """
    
    def __init__(self, L: int):
        """
        Initialize calculator for L-site system.
        
        Args:
            L: Number of sites in the chain
        """
        self.L = L
        self.dim = 2 ** L
    
    def _reshape_state(
        self, 
        state: np.ndarray, 
        subsystem_A: List[int]
    ) -> np.ndarray:
        """
        Reshape state vector for bipartition.
        
        Reorders and reshapes |ψ⟩ into matrix form for SVD.
        
        Args:
            state: Full state vector (dim = 2^L)
            subsystem_A: List of site indices in subsystem A
            
        Returns:
            Matrix of shape (dim_A, dim_B) for SVD
        """
        L_A = len(subsystem_A)
        L_B = self.L - L_A
        dim_A = 2 ** L_A
        dim_B = 2 ** L_B
        
        # Get subsystem B (complement of A)
        subsystem_B = [i for i in range(self.L) if i not in subsystem_A]
        
        # Create mapping from original basis to (A, B) basis
        # This is the key step for arbitrary bipartitions
        psi_matrix = np.zeros((dim_A, dim_B), dtype=state.dtype)
        
        for n in range(self.dim):
            # Extract bits for subsystem A and B
            bits_A = 0
            bits_B = 0
            
            for i, site in enumerate(subsystem_A):
                if (n >> site) & 1:
                    bits_A |= (1 << i)
            
            for i, site in enumerate(subsystem_B):
                if (n >> site) & 1:
                    bits_B |= (1 << i)
            
            psi_matrix[bits_A, bits_B] = state[n]
        
        return psi_matrix
    
    def reduced_density_matrix(
        self, 
        state: np.ndarray, 
        subsystem_A: List[int]
    ) -> np.ndarray:
        """
        Compute reduced density matrix ρ_A = Tr_B(|ψ⟩⟨ψ|).
        
        Args:
            state: Full state vector
            subsystem_A: List of site indices in subsystem A
            
        Returns:
            Reduced density matrix for subsystem A
        """
        psi_matrix = self._reshape_state(state, subsystem_A)
        # ρ_A = ψ ψ† (matrix multiplication)
        return psi_matrix @ psi_matrix.conj().T
    
    def schmidt_decomposition(
        self, 
        state: np.ndarray, 
        subsystem_A: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Schmidt decomposition of state.
        
        |ψ⟩ = Σᵢ λᵢ |φᵢ⟩_A ⊗ |χᵢ⟩_B
        
        Args:
            state: Full state vector
            subsystem_A: List of site indices in subsystem A
            
        Returns:
            (schmidt_values, U, V) where U contains |φᵢ⟩_A, V contains |χᵢ⟩_B
        """
        psi_matrix = self._reshape_state(state, subsystem_A)
        U, s, Vh = np.linalg.svd(psi_matrix, full_matrices=False)
        return s, U, Vh
    
    def von_neumann_entropy(
        self, 
        state: np.ndarray, 
        subsystem_A: Optional[List[int]] = None,
        cut_position: Optional[int] = None
    ) -> float:
        """
        Compute von Neumann entanglement entropy S = -Tr(ρ log ρ).
        
        Args:
            state: Full state vector
            subsystem_A: List of site indices in subsystem A
                        If None, uses cut_position
            cut_position: Position of bipartition cut (sites 0..cut-1 in A)
                         If None, uses L//2
            
        Returns:
            Entanglement entropy in natural log units (nats)
        """
        if subsystem_A is None:
            if cut_position is None:
                cut_position = self.L // 2
            subsystem_A = list(range(cut_position))
        
        # Get Schmidt values via SVD
        schmidt_values, _, _ = self.schmidt_decomposition(state, subsystem_A)
        
        # Compute entropy S = -Σᵢ λᵢ² log(λᵢ²)
        # where λᵢ are Schmidt values (singular values)
        probs = schmidt_values ** 2
        
        # Filter out zeros to avoid log(0)
        probs = probs[probs > 1e-15]
        
        entropy = -np.sum(probs * np.log(probs))
        
        return entropy
    
    def renyi_entropy(
        self, 
        state: np.ndarray, 
        alpha: float,
        subsystem_A: Optional[List[int]] = None,
        cut_position: Optional[int] = None
    ) -> float:
        """
        Compute Rényi entropy S_α = (1/(1-α)) log(Tr(ρ^α)).
        
        Args:
            state: Full state vector
            alpha: Rényi index (α > 0, α ≠ 1)
            subsystem_A: List of site indices in subsystem A
            cut_position: Position of bipartition cut
            
        Returns:
            Rényi entropy
        """
        if alpha == 1:
            return self.von_neumann_entropy(state, subsystem_A, cut_position)
        
        if subsystem_A is None:
            if cut_position is None:
                cut_position = self.L // 2
            subsystem_A = list(range(cut_position))
        
        schmidt_values, _, _ = self.schmidt_decomposition(state, subsystem_A)
        probs = schmidt_values ** 2
        probs = probs[probs > 1e-15]
        
        return np.log(np.sum(probs ** alpha)) / (1 - alpha)
    
    def entanglement_spectrum(
        self, 
        state: np.ndarray, 
        subsystem_A: Optional[List[int]] = None,
        cut_position: Optional[int] = None
    ) -> EntanglementSpectrum:
        """
        Compute full entanglement spectrum.
        
        The entanglement spectrum consists of "entanglement energies"
        ξᵢ = -log(λᵢ) where λᵢ are eigenvalues of ρ_A.
        
        Args:
            state: Full state vector
            subsystem_A: List of site indices in subsystem A
            cut_position: Position of bipartition cut
            
        Returns:
            EntanglementSpectrum with eigenvalues and energies
        """
        if subsystem_A is None:
            if cut_position is None:
                cut_position = self.L // 2
            subsystem_A = list(range(cut_position))
        
        schmidt_values, _, _ = self.schmidt_decomposition(state, subsystem_A)
        eigenvalues = schmidt_values ** 2
        
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        
        # Compute entanglement energies (avoid log(0))
        mask = eigenvalues > 1e-15
        entanglement_energies = np.full_like(eigenvalues, np.inf)
        entanglement_energies[mask] = -np.log(eigenvalues[mask])
        
        entropy = -np.sum(eigenvalues[mask] * np.log(eigenvalues[mask]))
        
        return EntanglementSpectrum(
            eigenvalues=eigenvalues,
            entanglement_energies=entanglement_energies,
            entropy=entropy
        )
    
    def half_chain_entropy(self, state: np.ndarray) -> float:
        """
        Compute entanglement entropy for half-chain bipartition.
        
        This is the most common measure for 1D systems.
        
        Args:
            state: Full state vector
            
        Returns:
            Half-chain entanglement entropy
        """
        return self.von_neumann_entropy(state, cut_position=self.L // 2)
    
    def entropy_profile(self, state: np.ndarray) -> np.ndarray:
        """
        Compute entanglement entropy for all bipartition cuts.
        
        S(l) = entropy of subsystem [0, 1, ..., l-1]
        
        Args:
            state: Full state vector
            
        Returns:
            Array of entropies S(1), S(2), ..., S(L-1)
        """
        entropies = np.zeros(self.L - 1)
        for l in range(1, self.L):
            entropies[l - 1] = self.von_neumann_entropy(state, cut_position=l)
        return entropies
    
    def mutual_information(
        self, 
        state: np.ndarray, 
        subsystem_A: List[int],
        subsystem_B: List[int]
    ) -> float:
        """
        Compute mutual information I(A:B) = S(A) + S(B) - S(A∪B).
        
        Args:
            state: Full state vector
            subsystem_A: Sites in region A
            subsystem_B: Sites in region B (must be disjoint from A)
            
        Returns:
            Mutual information
        """
        # Check disjoint
        if set(subsystem_A) & set(subsystem_B):
            raise ValueError("Subsystems A and B must be disjoint")
        
        S_A = self.von_neumann_entropy(state, subsystem_A)
        S_B = self.von_neumann_entropy(state, subsystem_B)
        S_AB = self.von_neumann_entropy(state, subsystem_A + subsystem_B)
        
        return S_A + S_B - S_AB
    
    def entanglement_entropy_result(
        self, 
        state: np.ndarray, 
        subsystem_A: Optional[List[int]] = None,
        cut_position: Optional[int] = None
    ) -> EntanglementResult:
        """
        Compute full entanglement result with all details.
        
        Args:
            state: Full state vector
            subsystem_A: List of site indices in subsystem A
            cut_position: Position of bipartition cut
            
        Returns:
            EntanglementResult with entropy, Schmidt values, etc.
        """
        if subsystem_A is None:
            if cut_position is None:
                cut_position = self.L // 2
            subsystem_A = list(range(cut_position))
        
        schmidt_values, _, _ = self.schmidt_decomposition(state, subsystem_A)
        
        # Compute entropy
        probs = schmidt_values ** 2
        mask = probs > 1e-15
        entropy = -np.sum(probs[mask] * np.log(probs[mask]))
        
        # Schmidt rank (number of non-negligible values)
        schmidt_rank = np.sum(schmidt_values > 1e-10)
        
        return EntanglementResult(
            entropy=entropy,
            schmidt_values=schmidt_values,
            schmidt_rank=schmidt_rank,
            subsystem_size=len(subsystem_A)
        )
