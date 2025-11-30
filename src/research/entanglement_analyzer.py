"""
Entanglement Analysis Module for Quantum Phase Validation.

Implements Task 16: Entanglement analysis
- 16.1 Entanglement entropy scaling S(L)
- 16.2 Check: area law vs log(L) corrections
- 16.3 Central charge extraction (if CFT)
- 16.4 Entanglement spectrum analysis

This module provides comprehensive entanglement analysis for validating
quantum phase transitions in the Disordered TFIM.

Key Physics:
- At a quantum critical point (QCP), entanglement entropy scales as:
  S(L) = (c/3) * log(L) + const  (for 1D CFT)
  where c is the central charge
- In gapped phases: S(L) → const (area law in 1D)
- Entanglement spectrum reveals topological properties
- For disordered systems: log corrections may be modified

References:
- Calabrese & Cardy (2004): Entanglement entropy in 1D CFT
- Refael & Moore (2004): Entanglement in random spin chains
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path
import logging
from scipy.optimize import curve_fit
from scipy.stats import linregress
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from ..quantum.disordered_tfim import DisorderedTFIM, DTFIMParams
from ..quantum.entanglement import EntanglementCalculator, EntanglementSpectrum
from ..quantum.observables import ObservableCalculator


@dataclass
class EntanglementDataPoint:
    """Data point for entanglement analysis at a specific (L, h, W)."""
    L: int  # System size
    h: float  # Transverse field
    W: float  # Disorder strength
    n_realizations: int
    
    # Entanglement entropy (half-chain)
    entropy: float
    entropy_std: float
    
    # Entanglement spectrum statistics
    spectrum_gap: float  # Gap in entanglement spectrum
    spectrum_gap_std: float
    schmidt_rank: float  # Average Schmidt rank
    schmidt_rank_std: float
    
    # Additional quantities
    renyi_2: float  # Rényi-2 entropy
    renyi_2_std: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'L': self.L,
            'h': self.h,
            'W': self.W,
            'n_realizations': self.n_realizations,
            'entropy': self.entropy,
            'entropy_std': self.entropy_std,
            'spectrum_gap': self.spectrum_gap,
            'spectrum_gap_std': self.spectrum_gap_std,
            'schmidt_rank': self.schmidt_rank,
            'schmidt_rank_std': self.schmidt_rank_std,
            'renyi_2': self.renyi_2,
            'renyi_2_std': self.renyi_2_std,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntanglementDataPoint':
        return cls(**data)


@dataclass
class EntropyScalingResult:
    """Result of entropy scaling analysis S(L)."""
    system_sizes: List[int]
    entropies: np.ndarray
    entropy_errors: np.ndarray
    
    # Fit results
    scaling_type: str  # 'logarithmic', 'area_law', 'power_law'
    fit_params: Dict[str, float]
    fit_errors: Dict[str, float]
    fit_quality: float  # R²
    
    # Central charge (if logarithmic)
    central_charge: Optional[float] = None
    central_charge_error: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'system_sizes': self.system_sizes,
            'entropies': self.entropies.tolist(),
            'entropy_errors': self.entropy_errors.tolist(),
            'scaling_type': self.scaling_type,
            'fit_params': self.fit_params,
            'fit_errors': self.fit_errors,
            'fit_quality': self.fit_quality,
            'central_charge': self.central_charge,
            'central_charge_error': self.central_charge_error,
        }


@dataclass
class AreaLawCheckResult:
    """Result of area law vs log(L) check."""
    is_area_law: bool
    is_logarithmic: bool
    is_critical: bool  # True if logarithmic scaling detected
    
    # Fit comparison
    area_law_r2: float
    log_law_r2: float
    
    # Log correction coefficient (if present)
    log_coefficient: float
    log_coefficient_error: float
    
    # Interpretation
    interpretation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_area_law': self.is_area_law,
            'is_logarithmic': self.is_logarithmic,
            'is_critical': self.is_critical,
            'area_law_r2': self.area_law_r2,
            'log_law_r2': self.log_law_r2,
            'log_coefficient': self.log_coefficient,
            'log_coefficient_error': self.log_coefficient_error,
            'interpretation': self.interpretation,
        }


@dataclass
class CentralChargeResult:
    """Result of central charge extraction."""
    central_charge: float
    central_charge_error: float
    
    # Fit details
    fit_quality: float
    system_sizes_used: List[int]
    
    # Comparison with known values
    closest_cft: str  # e.g., 'Ising (c=1/2)', 'Free fermion (c=1)'
    deviation_from_closest: float
    
    # Validity
    is_valid: bool
    notes: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'central_charge': self.central_charge,
            'central_charge_error': self.central_charge_error,
            'fit_quality': self.fit_quality,
            'system_sizes_used': self.system_sizes_used,
            'closest_cft': self.closest_cft,
            'deviation_from_closest': self.deviation_from_closest,
            'is_valid': self.is_valid,
            'notes': self.notes,
        }


@dataclass
class EntanglementSpectrumResult:
    """Result of entanglement spectrum analysis."""
    system_size: int
    h: float
    W: float
    
    # Spectrum statistics
    eigenvalues: np.ndarray  # Eigenvalues of reduced density matrix
    entanglement_energies: np.ndarray  # ξᵢ = -log(λᵢ)
    
    # Gap analysis
    spectrum_gap: float  # ξ₁ - ξ₀
    gap_ratio: float  # (ξ₂ - ξ₁) / (ξ₁ - ξ₀)
    
    # Degeneracy analysis
    degeneracy_pattern: List[int]  # Multiplicities of low-lying levels
    
    # Topological indicators
    is_topological: bool
    topological_signature: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'system_size': self.system_size,
            'h': self.h,
            'W': self.W,
            'eigenvalues': self.eigenvalues.tolist(),
            'entanglement_energies': self.entanglement_energies.tolist(),
            'spectrum_gap': self.spectrum_gap,
            'gap_ratio': self.gap_ratio,
            'degeneracy_pattern': self.degeneracy_pattern,
            'is_topological': self.is_topological,
            'topological_signature': self.topological_signature,
        }


@dataclass
class EntanglementAnalysisResult:
    """Complete result of entanglement analysis (Task 16)."""
    # Parameters
    system_sizes: List[int]
    h: float  # Transverse field (at or near critical point)
    W: float  # Disorder strength
    n_realizations: int
    
    # Task 16.1: Entropy scaling
    entropy_scaling: EntropyScalingResult
    
    # Task 16.2: Area law check
    area_law_check: AreaLawCheckResult
    
    # Task 16.3: Central charge
    central_charge: CentralChargeResult
    
    # Task 16.4: Entanglement spectrum
    spectrum_results: List[EntanglementSpectrumResult]
    
    # Raw data
    data_points: List[EntanglementDataPoint]
    
    # Metadata
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'system_sizes': self.system_sizes,
            'h': self.h,
            'W': self.W,
            'n_realizations': self.n_realizations,
            'entropy_scaling': self.entropy_scaling.to_dict(),
            'area_law_check': self.area_law_check.to_dict(),
            'central_charge': self.central_charge.to_dict(),
            'spectrum_results': [sr.to_dict() for sr in self.spectrum_results],
            'data_points': [dp.to_dict() for dp in self.data_points],
            'metadata': self.metadata,
        }
    
    def save(self, filepath: str):
        """Save results to JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        def convert_numpy(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        data = convert_numpy(self.to_dict())
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def generate_report(self) -> str:
        """Generate human-readable report."""
        lines = []
        lines.append("=" * 80)
        lines.append("ENTANGLEMENT ANALYSIS REPORT (Task 16)")
        lines.append("=" * 80)
        lines.append("")
        
        lines.append("PARAMETERS")
        lines.append("-" * 40)
        lines.append(f"  System sizes: {self.system_sizes}")
        lines.append(f"  Transverse field h: {self.h}")
        lines.append(f"  Disorder strength W: {self.W}")
        lines.append(f"  Disorder realizations: {self.n_realizations}")
        lines.append("")
        
        lines.append("TASK 16.1: ENTROPY SCALING S(L)")
        lines.append("-" * 40)
        lines.append(f"  Scaling type: {self.entropy_scaling.scaling_type}")
        lines.append(f"  Fit quality (R²): {self.entropy_scaling.fit_quality:.4f}")
        for param, value in self.entropy_scaling.fit_params.items():
            error = self.entropy_scaling.fit_errors.get(param, 0)
            lines.append(f"  {param}: {value:.4f} ± {error:.4f}")
        lines.append("")
        
        lines.append("TASK 16.2: AREA LAW CHECK")
        lines.append("-" * 40)
        lines.append(f"  Is area law: {self.area_law_check.is_area_law}")
        lines.append(f"  Is logarithmic: {self.area_law_check.is_logarithmic}")
        lines.append(f"  Is critical: {self.area_law_check.is_critical}")
        lines.append(f"  Area law R²: {self.area_law_check.area_law_r2:.4f}")
        lines.append(f"  Log law R²: {self.area_law_check.log_law_r2:.4f}")
        lines.append(f"  Interpretation: {self.area_law_check.interpretation}")
        lines.append("")
        
        lines.append("TASK 16.3: CENTRAL CHARGE")
        lines.append("-" * 40)
        if self.central_charge.is_valid:
            lines.append(f"  c = {self.central_charge.central_charge:.4f} ± "
                        f"{self.central_charge.central_charge_error:.4f}")
            lines.append(f"  Closest CFT: {self.central_charge.closest_cft}")
            lines.append(f"  Deviation: {self.central_charge.deviation_from_closest:.4f}")
        else:
            lines.append(f"  Central charge extraction failed: {self.central_charge.notes}")
        lines.append("")
        
        lines.append("TASK 16.4: ENTANGLEMENT SPECTRUM")
        lines.append("-" * 40)
        for sr in self.spectrum_results:
            lines.append(f"  L = {sr.system_size}:")
            lines.append(f"    Spectrum gap: {sr.spectrum_gap:.4f}")
            lines.append(f"    Gap ratio: {sr.gap_ratio:.4f}")
            lines.append(f"    Degeneracy pattern: {sr.degeneracy_pattern[:5]}...")
            lines.append(f"    Topological: {sr.is_topological}")
        lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)


# Known CFT central charges for comparison
KNOWN_CENTRAL_CHARGES = {
    'Ising': 0.5,
    'Free fermion': 1.0,
    'Tricritical Ising': 0.7,
    'Three-state Potts': 0.8,
    'XY': 1.0,
    'Heisenberg': 1.0,
}



class EntanglementAnalyzer:
    """
    Comprehensive entanglement analysis for quantum phase validation.
    
    Implements Task 16:
    - 16.1 Entanglement entropy scaling S(L)
    - 16.2 Check: area law vs log(L) corrections
    - 16.3 Central charge extraction (if CFT)
    - 16.4 Entanglement spectrum analysis
    """
    
    def __init__(
        self,
        system_sizes: List[int] = None,
        n_disorder_realizations: int = 100,
        J_mean: float = 1.0,
        periodic: bool = True,
        random_seed: int = 42
    ):
        """
        Initialize entanglement analyzer.
        
        Args:
            system_sizes: List of system sizes (default: [8, 12, 16, 20, 24])
            n_disorder_realizations: Number of disorder realizations per point
            J_mean: Mean coupling strength
            periodic: Use periodic boundary conditions
            random_seed: Random seed for reproducibility
        """
        self.system_sizes = system_sizes or [8, 12, 16, 20, 24]
        self.n_disorder_realizations = n_disorder_realizations
        self.J_mean = J_mean
        self.periodic = periodic
        self.random_seed = random_seed
        
        self.logger = logging.getLogger(__name__)
    
    def _compute_entanglement_data(
        self,
        L: int,
        h: float,
        W: float,
        n_realizations: int
    ) -> EntanglementDataPoint:
        """
        Compute entanglement data at a single (L, h, W) point.
        
        Args:
            L: System size
            h: Mean transverse field
            W: Disorder strength
            n_realizations: Number of disorder realizations
            
        Returns:
            EntanglementDataPoint with computed quantities
        """
        # Create DTFIM
        params = DTFIMParams(
            L=L,
            h_mean=h,
            h_disorder=W,
            J_mean=self.J_mean,
            J_disorder=0.0,
            periodic=self.periodic
        )
        dtfim = DisorderedTFIM(params)
        
        # Entanglement calculator
        ent_calc = EntanglementCalculator(L)
        
        # Collect data over disorder realizations
        entropy_list = []
        renyi_2_list = []
        spectrum_gap_list = []
        schmidt_rank_list = []
        
        for i in range(n_realizations):
            realization = dtfim.disorder_framework.realization_generator.generate_single(
                realization_index=i
            )
            
            # Compute ground state
            E, state = dtfim.compute_ground_state(realization)
            
            # Half-chain entanglement entropy
            entropy = ent_calc.half_chain_entropy(state)
            entropy_list.append(entropy)
            
            # Rényi-2 entropy
            renyi_2 = ent_calc.renyi_entropy(state, alpha=2.0, cut_position=L // 2)
            renyi_2_list.append(renyi_2)
            
            # Entanglement spectrum
            spectrum = ent_calc.entanglement_spectrum(state, cut_position=L // 2)
            
            # Spectrum gap (difference between two lowest entanglement energies)
            finite_energies = spectrum.entanglement_energies[
                np.isfinite(spectrum.entanglement_energies)
            ]
            if len(finite_energies) >= 2:
                spectrum_gap = finite_energies[1] - finite_energies[0]
                spectrum_gap_list.append(spectrum_gap)
            
            # Schmidt rank
            schmidt_rank = np.sum(spectrum.eigenvalues > 1e-10)
            schmidt_rank_list.append(schmidt_rank)
        
        # Compute statistics
        n = len(entropy_list)
        
        return EntanglementDataPoint(
            L=L,
            h=h,
            W=W,
            n_realizations=n_realizations,
            entropy=np.mean(entropy_list),
            entropy_std=np.std(entropy_list, ddof=1) if n > 1 else 0.0,
            spectrum_gap=np.mean(spectrum_gap_list) if spectrum_gap_list else 0.0,
            spectrum_gap_std=np.std(spectrum_gap_list, ddof=1) if len(spectrum_gap_list) > 1 else 0.0,
            schmidt_rank=np.mean(schmidt_rank_list),
            schmidt_rank_std=np.std(schmidt_rank_list, ddof=1) if n > 1 else 0.0,
            renyi_2=np.mean(renyi_2_list),
            renyi_2_std=np.std(renyi_2_list, ddof=1) if n > 1 else 0.0,
        )
    
    def compute_entropy_scaling(
        self,
        h: float,
        W: float,
        parallel: bool = True,
        max_workers: Optional[int] = None
    ) -> Tuple[List[EntanglementDataPoint], EntropyScalingResult]:
        """
        Task 16.1: Compute entanglement entropy scaling S(L).
        
        Computes S(L) for all system sizes and fits to determine scaling.
        
        Args:
            h: Transverse field (should be at or near critical point)
            W: Disorder strength
            parallel: Use parallel computation
            max_workers: Maximum number of parallel workers
            
        Returns:
            (data_points, EntropyScalingResult)
        """
        self.logger.info("=" * 60)
        self.logger.info("TASK 16.1: ENTANGLEMENT ENTROPY SCALING S(L)")
        self.logger.info("=" * 60)
        self.logger.info(f"h = {h}, W = {W}")
        self.logger.info(f"System sizes: {self.system_sizes}")
        
        data_points = []
        start_time = time.time()
        
        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._compute_entanglement_data, L, h, W, 
                        self.n_disorder_realizations
                    ): L
                    for L in self.system_sizes
                }
                
                for future in as_completed(futures):
                    L = futures[future]
                    try:
                        point = future.result()
                        data_points.append(point)
                        self.logger.info(f"  L = {L}: S = {point.entropy:.4f} ± {point.entropy_std:.4f}")
                    except Exception as e:
                        self.logger.error(f"Failed at L = {L}: {e}")
        else:
            for L in self.system_sizes:
                try:
                    point = self._compute_entanglement_data(
                        L, h, W, self.n_disorder_realizations
                    )
                    data_points.append(point)
                    self.logger.info(f"  L = {L}: S = {point.entropy:.4f} ± {point.entropy_std:.4f}")
                except Exception as e:
                    self.logger.error(f"Failed at L = {L}: {e}")
        
        # Sort by system size
        data_points.sort(key=lambda p: p.L)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Data collection completed in {elapsed:.1f} seconds")
        
        # Fit entropy scaling
        L_values = np.array([p.L for p in data_points])
        S_values = np.array([p.entropy for p in data_points])
        S_errors = np.array([p.entropy_std for p in data_points])
        
        # Try different scaling forms
        scaling_result = self._fit_entropy_scaling(L_values, S_values, S_errors)
        
        return data_points, scaling_result
    
    def _fit_entropy_scaling(
        self,
        L_values: np.ndarray,
        S_values: np.ndarray,
        S_errors: np.ndarray
    ) -> EntropyScalingResult:
        """
        Fit entropy scaling to determine scaling type.
        
        Tries:
        1. Logarithmic: S = (c/3) * log(L) + const (CFT at criticality)
        2. Area law: S = const (gapped phase)
        3. Power law: S = a * L^α + const (unusual scaling)
        
        Args:
            L_values: System sizes
            S_values: Entanglement entropies
            S_errors: Entropy errors
            
        Returns:
            EntropyScalingResult with best fit
        """
        results = {}
        
        # 1. Logarithmic fit: S = a * log(L) + b
        try:
            def log_func(L, a, b):
                return a * np.log(L) + b
            
            popt_log, pcov_log = curve_fit(
                log_func, L_values, S_values,
                sigma=S_errors if np.all(S_errors > 0) else None,
                p0=[0.5, 0.0]
            )
            
            S_pred_log = log_func(L_values, *popt_log)
            ss_res = np.sum((S_values - S_pred_log) ** 2)
            ss_tot = np.sum((S_values - np.mean(S_values)) ** 2)
            r2_log = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            results['logarithmic'] = {
                'params': {'a': popt_log[0], 'b': popt_log[1]},
                'errors': {'a': np.sqrt(pcov_log[0, 0]), 'b': np.sqrt(pcov_log[1, 1])},
                'r2': r2_log,
                'central_charge': 3 * popt_log[0],  # c = 3a
                'central_charge_error': 3 * np.sqrt(pcov_log[0, 0]),
            }
        except Exception as e:
            self.logger.warning(f"Logarithmic fit failed: {e}")
            results['logarithmic'] = {'r2': 0}
        
        # 2. Area law (constant): S = const
        try:
            const_mean = np.mean(S_values)
            const_std = np.std(S_values)
            ss_res = np.sum((S_values - const_mean) ** 2)
            ss_tot = np.sum((S_values - np.mean(S_values)) ** 2)
            r2_const = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            # For constant fit, R² is always 0 by definition
            # Use variance as quality metric instead
            variance_ratio = const_std / const_mean if const_mean > 0 else float('inf')
            
            results['area_law'] = {
                'params': {'const': const_mean},
                'errors': {'const': const_std},
                'r2': 1 - variance_ratio,  # Higher is better (less variation)
                'variance_ratio': variance_ratio,
            }
        except Exception as e:
            self.logger.warning(f"Area law fit failed: {e}")
            results['area_law'] = {'r2': 0}
        
        # 3. Power law: S = a * L^α + b
        try:
            def power_func(L, a, alpha, b):
                return a * np.power(L, alpha) + b
            
            popt_pow, pcov_pow = curve_fit(
                power_func, L_values, S_values,
                sigma=S_errors if np.all(S_errors > 0) else None,
                p0=[0.1, 0.5, 0.0],
                bounds=([0, 0, -10], [10, 2, 10])
            )
            
            S_pred_pow = power_func(L_values, *popt_pow)
            ss_res = np.sum((S_values - S_pred_pow) ** 2)
            ss_tot = np.sum((S_values - np.mean(S_values)) ** 2)
            r2_pow = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            results['power_law'] = {
                'params': {'a': popt_pow[0], 'alpha': popt_pow[1], 'b': popt_pow[2]},
                'errors': {
                    'a': np.sqrt(pcov_pow[0, 0]),
                    'alpha': np.sqrt(pcov_pow[1, 1]),
                    'b': np.sqrt(pcov_pow[2, 2])
                },
                'r2': r2_pow,
            }
        except Exception as e:
            self.logger.warning(f"Power law fit failed: {e}")
            results['power_law'] = {'r2': 0}
        
        # Determine best scaling type
        # Prefer logarithmic if R² is good (indicates criticality)
        log_r2 = results.get('logarithmic', {}).get('r2', 0)
        area_r2 = results.get('area_law', {}).get('r2', 0)
        pow_r2 = results.get('power_law', {}).get('r2', 0)
        
        # Decision logic
        if log_r2 > 0.9:
            best_type = 'logarithmic'
        elif area_r2 > 0.9 and log_r2 < 0.7:
            best_type = 'area_law'
        elif pow_r2 > log_r2 and pow_r2 > 0.9:
            best_type = 'power_law'
        else:
            # Default to logarithmic if entropy increases with L
            if S_values[-1] > S_values[0]:
                best_type = 'logarithmic'
            else:
                best_type = 'area_law'
        
        best_result = results.get(best_type, {})
        
        self.logger.info(f"Best scaling type: {best_type}")
        self.logger.info(f"  R² = {best_result.get('r2', 0):.4f}")
        
        return EntropyScalingResult(
            system_sizes=L_values.tolist(),
            entropies=S_values,
            entropy_errors=S_errors,
            scaling_type=best_type,
            fit_params=best_result.get('params', {}),
            fit_errors=best_result.get('errors', {}),
            fit_quality=best_result.get('r2', 0),
            central_charge=best_result.get('central_charge'),
            central_charge_error=best_result.get('central_charge_error'),
        )

    
    def check_area_law(
        self,
        data_points: List[EntanglementDataPoint],
        entropy_scaling: EntropyScalingResult
    ) -> AreaLawCheckResult:
        """
        Task 16.2: Check area law vs log(L) corrections.
        
        In 1D:
        - Gapped phases: S → const (area law)
        - Critical points: S ~ (c/3) * log(L) (logarithmic violation)
        - Disordered critical: S ~ (c_eff/3) * log(L) with modified c
        
        Args:
            data_points: Entanglement data points
            entropy_scaling: Result from entropy scaling analysis
            
        Returns:
            AreaLawCheckResult
        """
        self.logger.info("=" * 60)
        self.logger.info("TASK 16.2: AREA LAW CHECK")
        self.logger.info("=" * 60)
        
        L_values = np.array([p.L for p in data_points])
        S_values = np.array([p.entropy for p in data_points])
        S_errors = np.array([p.entropy_std for p in data_points])
        
        # Fit area law (constant)
        S_mean = np.mean(S_values)
        S_std = np.std(S_values)
        ss_res_area = np.sum((S_values - S_mean) ** 2)
        ss_tot = np.sum((S_values - S_mean) ** 2)
        
        # For constant fit, compute how well constant describes data
        # Use coefficient of variation as metric
        cv = S_std / S_mean if S_mean > 0 else float('inf')
        area_law_r2 = max(0, 1 - cv)  # Higher is better
        
        # Fit logarithmic: S = a * log(L) + b
        try:
            log_L = np.log(L_values)
            slope, intercept, r_value, p_value, std_err = linregress(log_L, S_values)
            log_law_r2 = r_value ** 2
            log_coefficient = slope
            log_coefficient_error = std_err
        except Exception:
            log_law_r2 = 0
            log_coefficient = 0
            log_coefficient_error = float('inf')
        
        # Determine which law fits better
        # Use Bayesian Information Criterion (BIC) for model selection
        n = len(S_values)
        
        # Area law: 1 parameter (const)
        ss_area = np.sum((S_values - S_mean) ** 2)
        bic_area = n * np.log(ss_area / n + 1e-10) + 1 * np.log(n)
        
        # Log law: 2 parameters (slope, intercept)
        S_pred_log = slope * log_L + intercept
        ss_log = np.sum((S_values - S_pred_log) ** 2)
        bic_log = n * np.log(ss_log / n + 1e-10) + 2 * np.log(n)
        
        # Lower BIC is better
        is_area_law = bic_area < bic_log
        is_logarithmic = bic_log < bic_area
        
        # Additional check: is the log coefficient significant?
        # If slope is small relative to error, it's effectively area law
        if log_coefficient_error > 0:
            t_stat = abs(log_coefficient) / log_coefficient_error
            is_significant_log = t_stat > 2.0  # ~95% confidence
        else:
            is_significant_log = False
        
        is_critical = is_logarithmic and is_significant_log
        
        # Interpretation
        if is_critical:
            if log_coefficient > 0.1:
                interpretation = (
                    f"Logarithmic scaling detected (coefficient = {log_coefficient:.4f}). "
                    f"System appears to be at or near a quantum critical point. "
                    f"Effective central charge c_eff ≈ {3 * log_coefficient:.2f}."
                )
            else:
                interpretation = (
                    f"Weak logarithmic correction detected (coefficient = {log_coefficient:.4f}). "
                    f"System may be near criticality or have weak log corrections."
                )
        elif is_area_law:
            interpretation = (
                f"Area law satisfied (S ≈ {S_mean:.4f} ± {S_std:.4f}). "
                f"System appears to be in a gapped phase."
            )
        else:
            interpretation = (
                f"Scaling behavior unclear. Area law R² = {area_law_r2:.3f}, "
                f"Log law R² = {log_law_r2:.3f}. Further analysis needed."
            )
        
        self.logger.info(f"  Area law R²: {area_law_r2:.4f}")
        self.logger.info(f"  Log law R²: {log_law_r2:.4f}")
        self.logger.info(f"  Log coefficient: {log_coefficient:.4f} ± {log_coefficient_error:.4f}")
        self.logger.info(f"  Is critical: {is_critical}")
        self.logger.info(f"  Interpretation: {interpretation}")
        
        return AreaLawCheckResult(
            is_area_law=is_area_law,
            is_logarithmic=is_logarithmic,
            is_critical=is_critical,
            area_law_r2=area_law_r2,
            log_law_r2=log_law_r2,
            log_coefficient=log_coefficient,
            log_coefficient_error=log_coefficient_error,
            interpretation=interpretation,
        )
    
    def extract_central_charge(
        self,
        data_points: List[EntanglementDataPoint],
        entropy_scaling: EntropyScalingResult
    ) -> CentralChargeResult:
        """
        Task 16.3: Extract central charge (if CFT).
        
        For 1D CFT at criticality:
        S(L) = (c/3) * log(L) + const
        
        For periodic boundary conditions:
        S(L) = (c/3) * log((L/π) * sin(πl/L)) + const
        where l is subsystem size (L/2 for half-chain)
        
        Args:
            data_points: Entanglement data points
            entropy_scaling: Result from entropy scaling analysis
            
        Returns:
            CentralChargeResult
        """
        self.logger.info("=" * 60)
        self.logger.info("TASK 16.3: CENTRAL CHARGE EXTRACTION")
        self.logger.info("=" * 60)
        
        L_values = np.array([p.L for p in data_points])
        S_values = np.array([p.entropy for p in data_points])
        S_errors = np.array([p.entropy_std for p in data_points])
        
        # Method 1: Simple logarithmic fit
        # S = (c/3) * log(L) + const
        try:
            log_L = np.log(L_values)
            slope, intercept, r_value, p_value, std_err = linregress(log_L, S_values)
            
            c_simple = 3 * slope
            c_simple_error = 3 * std_err
            r2_simple = r_value ** 2
            
            self.logger.info(f"  Simple fit: c = {c_simple:.4f} ± {c_simple_error:.4f}")
            self.logger.info(f"  R² = {r2_simple:.4f}")
        except Exception as e:
            self.logger.warning(f"Simple fit failed: {e}")
            c_simple = 0
            c_simple_error = float('inf')
            r2_simple = 0
        
        # Method 2: CFT formula for periodic BC
        # S = (c/3) * log((L/π) * sin(π * l/L)) + const
        # For l = L/2: sin(π/2) = 1, so S = (c/3) * log(L/π) + const
        try:
            def cft_formula(L, c, const):
                return (c / 3) * np.log(L / np.pi) + const
            
            popt, pcov = curve_fit(
                cft_formula, L_values, S_values,
                sigma=S_errors if np.all(S_errors > 0) else None,
                p0=[0.5, 0.0]
            )
            
            c_cft = popt[0]
            c_cft_error = np.sqrt(pcov[0, 0])
            
            S_pred = cft_formula(L_values, *popt)
            ss_res = np.sum((S_values - S_pred) ** 2)
            ss_tot = np.sum((S_values - np.mean(S_values)) ** 2)
            r2_cft = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            self.logger.info(f"  CFT formula fit: c = {c_cft:.4f} ± {c_cft_error:.4f}")
            self.logger.info(f"  R² = {r2_cft:.4f}")
        except Exception as e:
            self.logger.warning(f"CFT formula fit failed: {e}")
            c_cft = c_simple
            c_cft_error = c_simple_error
            r2_cft = r2_simple
        
        # Use better fit
        if r2_cft > r2_simple:
            central_charge = c_cft
            central_charge_error = c_cft_error
            fit_quality = r2_cft
        else:
            central_charge = c_simple
            central_charge_error = c_simple_error
            fit_quality = r2_simple
        
        # Find closest known CFT
        closest_cft = 'Unknown'
        min_deviation = float('inf')
        
        for cft_name, c_known in KNOWN_CENTRAL_CHARGES.items():
            deviation = abs(central_charge - c_known)
            if deviation < min_deviation:
                min_deviation = deviation
                closest_cft = f"{cft_name} (c={c_known})"
        
        # Determine validity
        is_valid = (
            fit_quality > 0.8 and
            central_charge > 0 and
            central_charge_error < central_charge
        )
        
        if not is_valid:
            if fit_quality < 0.8:
                notes = "Poor fit quality - system may not be at criticality"
            elif central_charge <= 0:
                notes = "Negative or zero central charge - unphysical"
            else:
                notes = "Large error bars - insufficient data"
        else:
            notes = ""
        
        self.logger.info(f"  Final: c = {central_charge:.4f} ± {central_charge_error:.4f}")
        self.logger.info(f"  Closest CFT: {closest_cft}")
        self.logger.info(f"  Valid: {is_valid}")
        
        return CentralChargeResult(
            central_charge=central_charge,
            central_charge_error=central_charge_error,
            fit_quality=fit_quality,
            system_sizes_used=L_values.tolist(),
            closest_cft=closest_cft,
            deviation_from_closest=min_deviation,
            is_valid=is_valid,
            notes=notes,
        )

    
    def analyze_entanglement_spectrum(
        self,
        L: int,
        h: float,
        W: float,
        n_realizations: int = 50
    ) -> EntanglementSpectrumResult:
        """
        Task 16.4: Analyze entanglement spectrum for a single system size.
        
        The entanglement spectrum {ξᵢ = -log(λᵢ)} reveals:
        - Topological properties (degeneracies)
        - Universal features at criticality
        - Gap structure
        
        Args:
            L: System size
            h: Transverse field
            W: Disorder strength
            n_realizations: Number of disorder realizations
            
        Returns:
            EntanglementSpectrumResult
        """
        self.logger.info(f"  Analyzing entanglement spectrum for L = {L}")
        
        # Create DTFIM
        params = DTFIMParams(
            L=L,
            h_mean=h,
            h_disorder=W,
            J_mean=self.J_mean,
            J_disorder=0.0,
            periodic=self.periodic
        )
        dtfim = DisorderedTFIM(params)
        ent_calc = EntanglementCalculator(L)
        
        # Collect spectra over disorder realizations
        all_eigenvalues = []
        all_energies = []
        gap_list = []
        gap_ratio_list = []
        
        for i in range(n_realizations):
            realization = dtfim.disorder_framework.realization_generator.generate_single(
                realization_index=i
            )
            
            E, state = dtfim.compute_ground_state(realization)
            spectrum = ent_calc.entanglement_spectrum(state, cut_position=L // 2)
            
            # Store eigenvalues
            all_eigenvalues.append(spectrum.eigenvalues)
            all_energies.append(spectrum.entanglement_energies)
            
            # Compute gap and gap ratio
            finite_energies = spectrum.entanglement_energies[
                np.isfinite(spectrum.entanglement_energies)
            ]
            if len(finite_energies) >= 2:
                gap = finite_energies[1] - finite_energies[0]
                gap_list.append(gap)
                
                if len(finite_energies) >= 3 and gap > 1e-10:
                    gap_ratio = (finite_energies[2] - finite_energies[1]) / gap
                    gap_ratio_list.append(gap_ratio)
        
        # Average spectrum
        max_len = max(len(ev) for ev in all_eigenvalues)
        padded_eigenvalues = np.zeros((n_realizations, max_len))
        padded_energies = np.full((n_realizations, max_len), np.inf)
        
        for i, (ev, en) in enumerate(zip(all_eigenvalues, all_energies)):
            padded_eigenvalues[i, :len(ev)] = ev
            padded_energies[i, :len(en)] = en
        
        avg_eigenvalues = np.mean(padded_eigenvalues, axis=0)
        avg_energies = np.nanmean(np.where(np.isinf(padded_energies), np.nan, padded_energies), axis=0)
        
        # Compute degeneracy pattern
        # Group levels by energy (within tolerance)
        degeneracy_pattern = self._compute_degeneracy_pattern(avg_energies)
        
        # Check for topological signatures
        # In topological phases, entanglement spectrum often shows characteristic degeneracies
        is_topological, topological_signature = self._check_topological_signature(
            avg_energies, degeneracy_pattern
        )
        
        avg_gap = np.mean(gap_list) if gap_list else 0.0
        avg_gap_ratio = np.mean(gap_ratio_list) if gap_ratio_list else 0.0
        
        self.logger.info(f"    Spectrum gap: {avg_gap:.4f}")
        self.logger.info(f"    Gap ratio: {avg_gap_ratio:.4f}")
        self.logger.info(f"    Degeneracy pattern: {degeneracy_pattern[:5]}")
        self.logger.info(f"    Topological: {is_topological}")
        
        return EntanglementSpectrumResult(
            system_size=L,
            h=h,
            W=W,
            eigenvalues=avg_eigenvalues,
            entanglement_energies=avg_energies,
            spectrum_gap=avg_gap,
            gap_ratio=avg_gap_ratio,
            degeneracy_pattern=degeneracy_pattern,
            is_topological=is_topological,
            topological_signature=topological_signature,
        )
    
    def _compute_degeneracy_pattern(
        self,
        energies: np.ndarray,
        tolerance: float = 0.1
    ) -> List[int]:
        """
        Compute degeneracy pattern from entanglement energies.
        
        Groups levels that are within tolerance of each other.
        
        Args:
            energies: Entanglement energies
            tolerance: Energy tolerance for grouping
            
        Returns:
            List of degeneracies for each level
        """
        finite_energies = energies[np.isfinite(energies)]
        if len(finite_energies) == 0:
            return []
        
        # Sort energies
        sorted_energies = np.sort(finite_energies)
        
        # Group by proximity
        degeneracies = []
        current_group = [sorted_energies[0]]
        
        for e in sorted_energies[1:]:
            if e - current_group[-1] < tolerance:
                current_group.append(e)
            else:
                degeneracies.append(len(current_group))
                current_group = [e]
        
        degeneracies.append(len(current_group))
        
        return degeneracies
    
    def _check_topological_signature(
        self,
        energies: np.ndarray,
        degeneracy_pattern: List[int]
    ) -> Tuple[bool, str]:
        """
        Check for topological signatures in entanglement spectrum.
        
        Topological phases often show:
        - Even degeneracies (2-fold, 4-fold)
        - Specific patterns related to edge modes
        
        Args:
            energies: Entanglement energies
            degeneracy_pattern: Degeneracy pattern
            
        Returns:
            (is_topological, signature_description)
        """
        if len(degeneracy_pattern) < 2:
            return False, "Insufficient data"
        
        # Check for 2-fold degeneracy (common in Z2 topological phases)
        has_twofold = any(d == 2 for d in degeneracy_pattern[:5])
        
        # Check for even degeneracies throughout
        all_even = all(d % 2 == 0 for d in degeneracy_pattern[:5] if d > 0)
        
        # Check for characteristic gap structure
        finite_energies = energies[np.isfinite(energies)]
        if len(finite_energies) >= 3:
            gap1 = finite_energies[1] - finite_energies[0]
            gap2 = finite_energies[2] - finite_energies[1]
            
            # Large gap ratio can indicate topological protection
            if gap1 > 0 and gap2 / gap1 > 2:
                has_protected_gap = True
            else:
                has_protected_gap = False
        else:
            has_protected_gap = False
        
        # Determine topological signature
        if has_twofold and has_protected_gap:
            return True, "Z2 topological (2-fold degeneracy with protected gap)"
        elif all_even and len(degeneracy_pattern) >= 3:
            return True, "Possible topological (all even degeneracies)"
        elif has_twofold:
            return False, "2-fold degeneracy present but no protected gap"
        else:
            return False, "No clear topological signature"
    
    def run_full_analysis(
        self,
        h: float,
        W: float,
        parallel: bool = True,
        output_dir: Optional[str] = None
    ) -> EntanglementAnalysisResult:
        """
        Run complete entanglement analysis (Tasks 16.1-16.4).
        
        Args:
            h: Transverse field (at or near critical point)
            W: Disorder strength
            parallel: Use parallel computation
            output_dir: Directory to save results
            
        Returns:
            EntanglementAnalysisResult with complete analysis
        """
        self.logger.info("=" * 80)
        self.logger.info("ENTANGLEMENT ANALYSIS (Task 16)")
        self.logger.info("=" * 80)
        self.logger.info(f"h = {h}, W = {W}")
        self.logger.info(f"System sizes: {self.system_sizes}")
        self.logger.info(f"Disorder realizations: {self.n_disorder_realizations}")
        
        start_time = time.time()
        
        # Task 16.1: Entropy scaling
        self.logger.info("\n" + "=" * 60)
        self.logger.info("TASK 16.1: ENTANGLEMENT ENTROPY SCALING")
        self.logger.info("=" * 60)
        data_points, entropy_scaling = self.compute_entropy_scaling(h, W, parallel)
        
        # Task 16.2: Area law check
        area_law_check = self.check_area_law(data_points, entropy_scaling)
        
        # Task 16.3: Central charge extraction
        central_charge = self.extract_central_charge(data_points, entropy_scaling)
        
        # Task 16.4: Entanglement spectrum analysis
        self.logger.info("\n" + "=" * 60)
        self.logger.info("TASK 16.4: ENTANGLEMENT SPECTRUM ANALYSIS")
        self.logger.info("=" * 60)
        
        spectrum_results = []
        for L in self.system_sizes:
            try:
                spectrum = self.analyze_entanglement_spectrum(
                    L, h, W, n_realizations=min(50, self.n_disorder_realizations)
                )
                spectrum_results.append(spectrum)
            except Exception as e:
                self.logger.error(f"Spectrum analysis failed for L = {L}: {e}")
        
        elapsed = time.time() - start_time
        
        # Create result
        result = EntanglementAnalysisResult(
            system_sizes=self.system_sizes,
            h=h,
            W=W,
            n_realizations=self.n_disorder_realizations,
            entropy_scaling=entropy_scaling,
            area_law_check=area_law_check,
            central_charge=central_charge,
            spectrum_results=spectrum_results,
            data_points=data_points,
            metadata={
                'computation_time_minutes': elapsed / 60,
                'J_mean': self.J_mean,
                'periodic': self.periodic,
            }
        )
        
        # Save results
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            result.save(str(output_path / 'entanglement_analysis.json'))
            
            with open(output_path / 'entanglement_report.txt', 'w') as f:
                f.write(result.generate_report())
            
            self.logger.info(f"\nResults saved to {output_dir}")
        
        self.logger.info("\n" + result.generate_report())
        
        return result



def run_task16_entanglement_analysis(
    h: float = 1.0,
    W: float = 0.5,
    system_sizes: List[int] = None,
    n_realizations: int = 50,
    output_dir: str = "results/task16_entanglement",
    parallel: bool = True
) -> EntanglementAnalysisResult:
    """
    Main function to run Task 16: Entanglement analysis.
    
    Implements:
    - 16.1 Entanglement entropy scaling S(L)
    - 16.2 Check: area law vs log(L) corrections
    - 16.3 Central charge extraction (if CFT)
    - 16.4 Entanglement spectrum analysis
    
    Args:
        h: Transverse field (should be at or near critical point)
        W: Disorder strength
        system_sizes: System sizes (default: [8, 12, 16, 20, 24])
        n_realizations: Number of disorder realizations
        output_dir: Output directory
        parallel: Use parallel computation
        
    Returns:
        EntanglementAnalysisResult
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if system_sizes is None:
        system_sizes = [8, 12, 16, 20, 24]
    
    analyzer = EntanglementAnalyzer(
        system_sizes=system_sizes,
        n_disorder_realizations=n_realizations,
    )
    
    result = analyzer.run_full_analysis(
        h=h,
        W=W,
        parallel=parallel,
        output_dir=output_dir,
    )
    
    return result


def analyze_entanglement_at_criticality(
    hc: float,
    W: float,
    system_sizes: List[int] = None,
    n_realizations: int = 100,
    output_dir: Optional[str] = None,
) -> EntanglementAnalysisResult:
    """
    Analyze entanglement at a known critical point.
    
    This is a convenience function for validating quantum phase transitions
    by analyzing entanglement properties at the critical point.
    
    Args:
        hc: Critical point (transverse field)
        W: Disorder strength
        system_sizes: System sizes
        n_realizations: Number of disorder realizations
        output_dir: Output directory
        
    Returns:
        EntanglementAnalysisResult
    """
    return run_task16_entanglement_analysis(
        h=hc,
        W=W,
        system_sizes=system_sizes,
        n_realizations=n_realizations,
        output_dir=output_dir,
        parallel=True,
    )
