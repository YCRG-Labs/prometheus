"""
Floquet (Periodically Driven) Ising Model implementation.

Implements time-periodic Hamiltonian for studying non-equilibrium quantum phases:
H(t) = H₁ for 0 < t < T/2
H(t) = H₂ for T/2 < t < T

Key features:
- Floquet operator computation U(T) = exp(-iH₂T/2) exp(-iH₁T/2)
- Quasienergy spectrum (eigenvalues of U(T))
- Time crystal signature detection
- Heating vs localization analysis
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from scipy import sparse
from scipy.linalg import expm

from .hamiltonian_builder import SpinHamiltonianBuilder
from .exact_diagonalization import ExactDiagonalizationSolver


@dataclass
class FloquetIsingParams:
    """Parameters for Floquet Ising Model."""
    L: int  # System size
    J: float = 1.0  # Ising coupling
    h1: float = 1.0  # Transverse field in first half-period
    h2: float = 0.5  # Transverse field in second half-period
    T: float = 1.0  # Driving period
    periodic: bool = True  # Boundary conditions
    
    def __post_init__(self):
        if self.T <= 0:
            raise ValueError(f"Period T must be positive, got {self.T}")
        if self.L > 16:
            raise ValueError(
                f"Floquet calculations are expensive. L={self.L} > 16 not recommended. "
                "Use L ≤ 16 for basic exploration."
            )


class FloquetIsing:
    """
    Floquet (Periodically Driven) Ising Model.
    
    Implements a two-step driving protocol:
    - Step 1 (0 to T/2): H₁ = -J Σᵢ σᶻᵢσᶻᵢ₊₁ - h₁ Σᵢ σˣᵢ
    - Step 2 (T/2 to T): H₂ = -J Σᵢ σᶻᵢσᶻᵢ₊₁ - h₂ Σᵢ σˣᵢ
    
    The Floquet operator is: U(T) = exp(-iH₂T/2) exp(-iH₁T/2)
    """
    
    def __init__(self, params: FloquetIsingParams):
        """
        Initialize Floquet Ising model.
        
        Args:
            params: Floquet Ising parameters
        """
        self.params = params
        self.builder = SpinHamiltonianBuilder(params.L)
        self.solver = ExactDiagonalizationSolver()
        
        # Build the two Hamiltonians
        self.H1 = self._build_hamiltonian(params.h1)
        self.H2 = self._build_hamiltonian(params.h2)
    
    def _build_hamiltonian(self, h: float) -> sparse.csr_matrix:
        """
        Build TFIM Hamiltonian with specified transverse field.
        
        H = -J Σᵢ σᶻᵢσᶻᵢ₊₁ - h Σᵢ σˣᵢ
        
        Args:
            h: Transverse field strength
            
        Returns:
            Sparse Hamiltonian matrix
        """
        L = self.params.L
        dim = 2 ** L
        
        # Transverse field term
        H = sparse.csr_matrix((dim, dim), dtype=np.float64)
        for i in range(L):
            sigma_x = self.builder.build_sigma_x(i)
            H = H - h * sigma_x
        
        # Ising interaction term
        n_bonds = L if self.params.periodic else L - 1
        for i in range(n_bonds):
            j = (i + 1) % L
            zz_term = self.builder.build_zz_interaction(i, j)
            H = H - self.params.J * zz_term
        
        return H
    
    def compute_floquet_operator(self) -> np.ndarray:
        """
        Compute Floquet operator U(T) = exp(-iH₂T/2) exp(-iH₁T/2).
        
        Note: For large systems, this is expensive as it requires
        matrix exponentials of dense matrices.
        
        Returns:
            Floquet operator as dense matrix
        """
        # Convert to dense for expm
        H1_dense = self.H1.toarray()
        H2_dense = self.H2.toarray()
        
        # Compute evolution operators
        # U₁ = exp(-iH₁T/2)
        U1 = expm(-1j * H1_dense * self.params.T / 2)
        
        # U₂ = exp(-iH₂T/2)
        U2 = expm(-1j * H2_dense * self.params.T / 2)
        
        # Floquet operator: U(T) = U₂ U₁
        U_floquet = U2 @ U1
        
        return U_floquet
    
    def compute_quasienergy_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute quasienergy spectrum from Floquet operator.
        
        Quasienergies ε are defined via: U(T) |ψ⟩ = exp(-iεT) |ψ⟩
        They are extracted from eigenvalues: λ = exp(-iεT)
        
        Returns:
            (quasienergies, floquet_states) tuple
            - quasienergies: Array of quasienergies in [-π/T, π/T)
            - floquet_states: Floquet eigenstates (columns)
        """
        U = self.compute_floquet_operator()
        
        # Diagonalize Floquet operator
        eigenvalues, eigenstates = np.linalg.eig(U)
        
        # Extract quasienergies from eigenvalues
        # λ = exp(-iεT) => ε = i ln(λ) / T
        quasienergies = np.angle(eigenvalues) / self.params.T
        
        # Sort by quasienergy
        idx = np.argsort(quasienergies)
        quasienergies = quasienergies[idx]
        eigenstates = eigenstates[:, idx]
        
        return quasienergies, eigenstates
    
    def compute_quasienergy_gap(self) -> float:
        """
        Compute gap in quasienergy spectrum.
        
        Returns:
            Minimum non-zero gap between adjacent quasienergies
        """
        quasienergies, _ = self.compute_quasienergy_spectrum()
        
        # Compute gaps between adjacent levels
        gaps = np.diff(quasienergies)
        
        # Filter out near-zero gaps (degeneracies)
        non_zero_gaps = gaps[gaps > 1e-10]
        
        if len(non_zero_gaps) == 0:
            # All levels are degenerate
            return 0.0
        
        # Return minimum non-zero gap
        return np.min(non_zero_gaps)
    
    def compute_time_evolved_state(
        self,
        initial_state: np.ndarray,
        n_periods: int
    ) -> np.ndarray:
        """
        Evolve state for n_periods under Floquet dynamics.
        
        Args:
            initial_state: Initial quantum state
            n_periods: Number of driving periods
            
        Returns:
            State after n_periods
        """
        U = self.compute_floquet_operator()
        
        # Apply Floquet operator n times
        state = initial_state.copy()
        for _ in range(n_periods):
            state = U @ state
        
        return state
    
    def compute_stroboscopic_observable(
        self,
        operator: sparse.spmatrix,
        initial_state: np.ndarray,
        n_periods: int
    ) -> np.ndarray:
        """
        Compute observable at stroboscopic times t = nT.
        
        Args:
            operator: Observable operator
            initial_state: Initial state
            n_periods: Number of periods to evolve
            
        Returns:
            Array of expectation values at each period
        """
        U = self.compute_floquet_operator()
        
        expectation_values = np.zeros(n_periods + 1)
        state = initial_state.copy()
        
        # Initial value
        expectation_values[0] = np.real(
            np.vdot(state, operator @ state)
        )
        
        # Evolve and measure
        for n in range(1, n_periods + 1):
            state = U @ state
            expectation_values[n] = np.real(
                np.vdot(state, operator @ state)
            )
        
        return expectation_values


class FloquetIsingExplorer:
    """
    Coarse exploration of Floquet Ising parameter space.
    
    Searches for time crystal signatures and other non-equilibrium phases.
    """
    
    def __init__(self, L: int = 12, J: float = 1.0):
        """
        Initialize explorer.
        
        Args:
            L: System size (keep ≤ 12 for speed)
            J: Ising coupling strength
        """
        if L > 14:
            print(f"Warning: L={L} may be slow. Consider L ≤ 12 for coarse scan.")
        
        self.L = L
        self.J = J
    
    def coarse_scan(
        self,
        h1_range: Tuple[float, float] = (0.5, 2.0),
        h2_range: Tuple[float, float] = (0.5, 2.0),
        T_range: Tuple[float, float] = (0.5, 2.0),
        n_h1: int = 5,
        n_h2: int = 5,
        n_T: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Coarse scan of Floquet parameter space.
        
        Args:
            h1_range: Range of h1 values
            h2_range: Range of h2 values
            T_range: Range of period values
            n_h1: Number of h1 points
            n_h2: Number of h2 points
            n_T: Number of T points
            
        Returns:
            Dictionary with scan results
        """
        h1_values = np.linspace(h1_range[0], h1_range[1], n_h1)
        h2_values = np.linspace(h2_range[0], h2_range[1], n_h2)
        T_values = np.linspace(T_range[0], T_range[1], n_T)
        
        # Storage for results
        quasienergy_gaps = np.zeros((n_h1, n_h2, n_T))
        
        total_points = n_h1 * n_h2 * n_T
        print(f"Scanning {n_h1}×{n_h2}×{n_T} = {total_points} parameter points...")
        
        count = 0
        for i, h1 in enumerate(h1_values):
            for j, h2 in enumerate(h2_values):
                for k, T in enumerate(T_values):
                    params = FloquetIsingParams(
                        L=self.L,
                        J=self.J,
                        h1=h1,
                        h2=h2,
                        T=T
                    )
                    
                    model = FloquetIsing(params)
                    gap = model.compute_quasienergy_gap()
                    
                    quasienergy_gaps[i, j, k] = gap
                    
                    count += 1
                    if count % 10 == 0:
                        print(f"  Progress: {count}/{total_points}")
        
        return {
            'h1_values': h1_values,
            'h2_values': h2_values,
            'T_values': T_values,
            'quasienergy_gaps': quasienergy_gaps,
        }
    
    def detect_time_crystal_signatures(
        self,
        params: FloquetIsingParams,
        n_periods: int = 20
    ) -> Dict[str, any]:
        """
        Detect potential time crystal signatures.
        
        Time crystals show:
        1. Subharmonic response (period doubling)
        2. Long-lived oscillations
        3. Robustness to perturbations
        
        Args:
            params: Floquet parameters to test
            n_periods: Number of periods to evolve
            
        Returns:
            Dictionary with time crystal indicators
        """
        model = FloquetIsing(params)
        
        # Start from a simple initial state (all spins up in z)
        dim = 2 ** params.L
        initial_state = np.zeros(dim)
        initial_state[0] = 1.0  # |↑↑...↑⟩
        
        # Measure magnetization over time
        # Build total magnetization operator
        mag_z_op = sparse.csr_matrix((dim, dim), dtype=np.float64)
        for i in range(params.L):
            sigma_z = model.builder.build_sigma_z(i)
            mag_z_op = mag_z_op + sigma_z
        mag_z_op = mag_z_op / params.L
        
        # Compute stroboscopic magnetization
        mag_trajectory = model.compute_stroboscopic_observable(
            mag_z_op,
            initial_state,
            n_periods
        )
        
        # Analyze for period doubling
        # Check if oscillation has period 2T instead of T
        # Simple check: compare odd and even time steps
        odd_values = mag_trajectory[1::2]
        even_values = mag_trajectory[2::2]
        
        # If time crystal, odd and even should be different
        period_doubling_strength = np.abs(np.mean(odd_values) - np.mean(even_values))
        
        # Check for long-lived oscillations (not decaying to zero)
        final_amplitude = np.abs(mag_trajectory[-1])
        
        return {
            'magnetization_trajectory': mag_trajectory,
            'period_doubling_strength': period_doubling_strength,
            'final_amplitude': final_amplitude,
            'has_time_crystal_signature': (
                period_doubling_strength > 0.1 and final_amplitude > 0.1
            )
        }
    
    def identify_anomalies(
        self,
        scan_results: Dict[str, np.ndarray],
        threshold: float = 2.0
    ) -> List[Dict]:
        """
        Identify anomalous regions in parameter space.
        
        Looks for:
        - Unusually small quasienergy gaps (potential phase transitions)
        - Sharp changes in gap (crossovers)
        
        Args:
            scan_results: Results from coarse_scan
            threshold: Anomaly detection threshold (in std devs)
            
        Returns:
            List of anomaly dictionaries
        """
        anomalies = []
        
        gaps = scan_results['quasienergy_gaps']
        h1_values = scan_results['h1_values']
        h2_values = scan_results['h2_values']
        T_values = scan_results['T_values']
        
        # Find small gaps
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        
        # Compute gradients
        grad_h1 = np.abs(np.gradient(gaps, axis=0))
        grad_h2 = np.abs(np.gradient(gaps, axis=1))
        grad_T = np.abs(np.gradient(gaps, axis=2))
        
        mean_grad = np.mean([np.mean(grad_h1), np.mean(grad_h2), np.mean(grad_T)])
        std_grad = np.std([np.std(grad_h1), np.std(grad_h2), np.std(grad_T)])
        
        for i in range(len(h1_values)):
            for j in range(len(h2_values)):
                for k in range(len(T_values)):
                    # Check for small gaps
                    if gaps[i, j, k] < mean_gap - threshold * std_gap:
                        anomalies.append({
                            'type': 'small_quasienergy_gap',
                            'h1': h1_values[i],
                            'h2': h2_values[j],
                            'T': T_values[k],
                            'gap': gaps[i, j, k],
                            'significance': (mean_gap - gaps[i, j, k]) / std_gap
                        })
                    
                    # Check for sharp changes
                    max_grad = max(grad_h1[i, j, k], grad_h2[i, j, k], grad_T[i, j, k])
                    if max_grad > mean_grad + threshold * std_grad:
                        anomalies.append({
                            'type': 'sharp_transition',
                            'h1': h1_values[i],
                            'h2': h2_values[j],
                            'T': T_values[k],
                            'gap': gaps[i, j, k],
                            'gradient': max_grad,
                            'significance': (max_grad - mean_grad) / std_grad
                        })
        
        print(f"Found {len(anomalies)} anomalies")
        return anomalies


def create_standard_floquet_ising(
    L: int,
    h1: float = 1.5,
    h2: float = 0.5,
    T: float = 1.0,
    J: float = 1.0
) -> FloquetIsing:
    """
    Create standard Floquet Ising model with typical parameters.
    
    Args:
        L: System size
        h1: Strong transverse field (first half-period)
        h2: Weak transverse field (second half-period)
        T: Driving period
        J: Ising coupling
        
    Returns:
        FloquetIsing instance
    """
    params = FloquetIsingParams(
        L=L,
        J=J,
        h1=h1,
        h2=h2,
        T=T,
        periodic=True
    )
    return FloquetIsing(params)
