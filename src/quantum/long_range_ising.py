"""
Long-Range Quantum Ising Model implementation.
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from scipy import sparse

from .hamiltonian_builder import SpinHamiltonianBuilder
from .exact_diagonalization import ExactDiagonalizationSolver
from .entanglement import EntanglementCalculator


@dataclass
class LongRangeIsingParams:
    """Parameters for Long-Range Ising Model."""
    L: int
    J0: float = 1.0
    alpha: float = 2.0
    h: float = 1.0
    periodic: bool = True
    ewald_cutoff: Optional[float] = None
    
    def __post_init__(self):
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")


class LongRangeIsing:
    """Long-Range Quantum Ising Model."""
    
    def __init__(self, params: LongRangeIsingParams):
        self.params = params
        self.solver = ExactDiagonalizationSolver()
        self.builder = SpinHamiltonianBuilder(params.L)
        self.entanglement_calc = EntanglementCalculator(params.L)
        self.coupling_matrix = self._compute_coupling_matrix()
    
    def _compute_coupling_matrix(self) -> np.ndarray:
        L = self.params.L
        J = np.zeros((L, L))
        
        for i in range(L):
            for j in range(i + 1, L):
                if self.params.periodic:
                    r = min(abs(i - j), L - abs(i - j))
                else:
                    r = abs(i - j)
                
                if r > 0:
                    J[i, j] = self.params.J0 / (r ** self.params.alpha)
                    J[j, i] = J[i, j]
        
        return J
    
    def build_hamiltonian(self) -> sparse.csr_matrix:
        L = self.params.L
        dim = 2 ** L
        
        H_field = sparse.csr_matrix((dim, dim), dtype=np.float64)
        for i in range(L):
            sigma_x = self.builder.build_sigma_x(i)
            H_field = H_field - self.params.h * sigma_x
        
        H_interaction = sparse.csr_matrix((dim, dim), dtype=np.float64)
        for i in range(L):
            for j in range(i + 1, L):
                if self.coupling_matrix[i, j] != 0:
                    zz_term = self.builder.build_zz_interaction(i, j)
                    H_interaction = H_interaction - self.coupling_matrix[i, j] * zz_term
        
        return H_field + H_interaction
    
    def compute_ground_state(self) -> Tuple[float, np.ndarray]:
        H = self.build_hamiltonian()
        energy, state = self.solver.ground_state(H)
        return energy, state
    
    def compute_observables(self, state: Optional[np.ndarray] = None) -> Dict[str, float]:
        if state is None:
            _, state = self.compute_ground_state()
        
        L = self.params.L
        
        mag_z = 0.0
        for i in range(L):
            sigma_z = self.builder.build_sigma_z(i)
            mag_z += np.real(state.conj() @ sigma_z @ state)
        mag_z /= L
        
        mag_x = 0.0
        for i in range(L):
            sigma_x = self.builder.build_sigma_x(i)
            mag_x += np.real(state.conj() @ sigma_x @ state)
        mag_x /= L
        
        correlations = []
        for i in range(L):
            for j in range(i + 1, min(i + 5, L)):
                zz_term = self.builder.build_zz_interaction(i, j)
                corr = np.real(state.conj() @ zz_term @ state)
                correlations.append(corr)
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        return {
            'magnetization_z': abs(mag_z),
            'magnetization_x': mag_x,
            'correlation': avg_correlation,
        }
    
    def compute_entanglement_entropy(self, state: Optional[np.ndarray] = None,
                                    cut: Optional[int] = None) -> float:
        if state is None:
            _, state = self.compute_ground_state()
        
        if cut is None:
            cut = self.params.L // 2
        
        return self.entanglement_calc.von_neumann_entropy(state, cut_position=cut)



class LongRangeIsingExplorer:
    """Coarse exploration of Long-Range Ising parameter space."""
    
    def __init__(self, L: int = 12, J0: float = 1.0):
        self.L = L
        self.J0 = J0
    
    def coarse_scan(self,
                   alpha_range: Tuple[float, float] = (1.5, 3.5),
                   h_range: Tuple[float, float] = (0.0, 2.0),
                   n_alpha: int = 10,
                   n_h: int = 10) -> Dict[str, np.ndarray]:
        alphas = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
        hs = np.linspace(h_range[0], h_range[1], n_h)
        
        energies = np.zeros((n_alpha, n_h))
        magnetizations_z = np.zeros((n_alpha, n_h))
        magnetizations_x = np.zeros((n_alpha, n_h))
        correlations = np.zeros((n_alpha, n_h))
        entropies = np.zeros((n_alpha, n_h))
        
        print(f"Scanning {n_alpha}×{n_h} = {n_alpha * n_h} parameter points...")
        
        for i, alpha in enumerate(alphas):
            for j, h in enumerate(hs):
                params = LongRangeIsingParams(
                    L=self.L,
                    J0=self.J0,
                    alpha=alpha,
                    h=h
                )
                
                model = LongRangeIsing(params)
                energy, state = model.compute_ground_state()
                observables = model.compute_observables(state)
                entropy = model.compute_entanglement_entropy(state)
                
                energies[i, j] = energy
                magnetizations_z[i, j] = observables['magnetization_z']
                magnetizations_x[i, j] = observables['magnetization_x']
                correlations[i, j] = observables['correlation']
                entropies[i, j] = entropy
                
                if (i * n_h + j + 1) % 10 == 0:
                    print(f"  Progress: {i * n_h + j + 1}/{n_alpha * n_h}")
        
        return {
            'alphas': alphas,
            'hs': hs,
            'energies': energies,
            'magnetizations_z': magnetizations_z,
            'magnetizations_x': magnetizations_x,
            'correlations': correlations,
            'entropies': entropies,
        }
    
    def identify_anomalies(self, scan_results: Dict[str, np.ndarray],
                          threshold: float = 2.0) -> List[Dict]:
        anomalies = []
        
        mag_z = scan_results['magnetizations_z']
        entropy = scan_results['entropies']
        
        grad_mag_h = np.abs(np.gradient(mag_z, axis=1))
        grad_entropy_alpha = np.abs(np.gradient(entropy, axis=0))
        
        mean_grad_h = np.mean(grad_mag_h)
        std_grad_h = np.std(grad_mag_h)
        
        alphas = scan_results['alphas']
        hs = scan_results['hs']
        
        for i in range(len(alphas)):
            for j in range(len(hs)):
                if grad_mag_h[i, j] > mean_grad_h + threshold * std_grad_h:
                    anomalies.append({
                        'type': 'phase_transition',
                        'alpha': alphas[i],
                        'h': hs[j],
                        'gradient': grad_mag_h[i, j],
                        'magnetization': mag_z[i, j],
                        'entropy': entropy[i, j],
                    })
                
                if grad_entropy_alpha[i, j] > np.mean(grad_entropy_alpha) + threshold * np.std(grad_entropy_alpha):
                    anomalies.append({
                        'type': 'universality_crossover',
                        'alpha': alphas[i],
                        'h': hs[j],
                        'gradient': grad_entropy_alpha[i, j],
                        'magnetization': mag_z[i, j],
                        'entropy': entropy[i, j],
                    })
        
        print(f"Found {len(anomalies)} anomalies")
        return anomalies
