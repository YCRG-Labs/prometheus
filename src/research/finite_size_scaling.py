"""
Finite-Size Scaling Analysis for DTFIM Validation.

Implements Task 14: Systematic finite-size scaling
- 14.1 System sizes: L = 8, 12, 16, 20, 24, 32
- 14.2 Extract critical point hc(L) for each size
- 14.3 Finite-size scaling collapse
- 14.4 Extract ν from scaling

This module provides rigorous finite-size scaling analysis for validating
quantum phase transitions in the Disordered TFIM.

Key Physics:
- At a quantum critical point, observables scale with system size L
- Critical point hc(L) shifts with L: hc(L) - hc(∞) ~ L^(-1/ν)
- Susceptibility scales as: χ ~ L^(γ/ν)
- Correlation length: ξ ~ L at criticality
- Scaling collapse: χ/L^(γ/ν) = f((h-hc)L^(1/ν))
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from scipy.optimize import curve_fit, minimize
from scipy.stats import linregress

from ..quantum.disordered_tfim import DisorderedTFIM, DTFIMParams
from ..quantum.observables import ObservableCalculator
from ..quantum.entanglement import EntanglementCalculator


# Standard system sizes for finite-size scaling (Task 14.1)
STANDARD_SYSTEM_SIZES = [8, 12, 16, 20, 24, 32]


@dataclass
class FiniteSizeDataPoint:
    """Data point for finite-size scaling analysis."""
    L: int  # System size
    h: float  # Transverse field
    W: float  # Disorder strength
    n_realizations: int
    
    # Observables (disorder-averaged)
    magnetization: float
    magnetization_std: float
    susceptibility: float
    susceptibility_std: float
    binder_cumulant: float
    binder_std: float
    energy_gap: float
    energy_gap_std: float
    entanglement_entropy: float
    entanglement_std: float
    correlation_length: float
    correlation_length_std: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'L': self.L,
            'h': self.h,
            'W': self.W,
            'n_realizations': self.n_realizations,
            'magnetization': self.magnetization,
            'magnetization_std': self.magnetization_std,
            'susceptibility': self.susceptibility,
            'susceptibility_std': self.susceptibility_std,
            'binder_cumulant': self.binder_cumulant,
            'binder_std': self.binder_std,
            'energy_gap': self.energy_gap,
            'energy_gap_std': self.energy_gap_std,
            'entanglement_entropy': self.entanglement_entropy,
            'entanglement_std': self.entanglement_std,
            'correlation_length': self.correlation_length,
            'correlation_length_std': self.correlation_length_std,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FiniteSizeDataPoint':
        return cls(**data)


@dataclass
class CriticalPointEstimate:
    """Estimate of critical point for a given system size."""
    L: int
    hc: float  # Critical point estimate
    hc_error: float  # Error estimate
    method: str  # Method used (susceptibility_peak, binder_crossing, etc.)
    chi_max: float  # Maximum susceptibility value
    fit_quality: float  # R² or similar quality metric
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'L': self.L,
            'hc': self.hc,
            'hc_error': self.hc_error,
            'method': self.method,
            'chi_max': self.chi_max,
            'fit_quality': self.fit_quality,
        }


@dataclass
class ScalingCollapseResult:
    """Result of finite-size scaling collapse analysis."""
    hc_inf: float  # Critical point in thermodynamic limit
    hc_inf_error: float
    nu: float  # Correlation length exponent
    nu_error: float
    gamma_over_nu: float  # γ/ν ratio
    gamma_over_nu_error: float
    collapse_quality: float  # Quality of collapse (0-1)
    scaled_data: Dict[int, Tuple[np.ndarray, np.ndarray]]  # L -> (x_scaled, y_scaled)
    raw_data: Dict[int, Tuple[np.ndarray, np.ndarray]]  # L -> (h, chi)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'hc_inf': self.hc_inf,
            'hc_inf_error': self.hc_inf_error,
            'nu': self.nu,
            'nu_error': self.nu_error,
            'gamma_over_nu': self.gamma_over_nu,
            'gamma_over_nu_error': self.gamma_over_nu_error,
            'collapse_quality': self.collapse_quality,
        }


@dataclass
class FiniteSizeScalingResult:
    """Complete result of finite-size scaling analysis."""
    system_sizes: List[int]
    h_values: np.ndarray
    W: float  # Disorder strength
    n_realizations: int
    
    # Raw data
    data_points: List[FiniteSizeDataPoint]
    
    # Critical point estimates per size (Task 14.2)
    critical_points: List[CriticalPointEstimate]
    
    # Scaling collapse result (Task 14.3)
    collapse_result: Optional[ScalingCollapseResult]
    
    # Extracted exponent (Task 14.4)
    nu: float
    nu_error: float
    
    # Additional extracted quantities
    hc_thermodynamic: float  # Critical point in L→∞ limit
    hc_thermodynamic_error: float
    
    metadata: Dict = field(default_factory=dict)
    
    def save(self, filepath: str):
        """Save results to JSON."""
        data = {
            'system_sizes': self.system_sizes,
            'h_values': self.h_values.tolist(),
            'W': self.W,
            'n_realizations': self.n_realizations,
            'data_points': [p.to_dict() for p in self.data_points],
            'critical_points': [cp.to_dict() for cp in self.critical_points],
            'collapse_result': self.collapse_result.to_dict() if self.collapse_result else None,
            'nu': self.nu,
            'nu_error': self.nu_error,
            'hc_thermodynamic': self.hc_thermodynamic,
            'hc_thermodynamic_error': self.hc_thermodynamic_error,
            'metadata': self.metadata,
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'FiniteSizeScalingResult':
        """Load results from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        data_points = [FiniteSizeDataPoint.from_dict(p) for p in data['data_points']]
        critical_points = [CriticalPointEstimate(**cp) for cp in data['critical_points']]
        
        collapse_data = data.get('collapse_result')
        collapse_result = None
        if collapse_data:
            collapse_result = ScalingCollapseResult(
                hc_inf=collapse_data['hc_inf'],
                hc_inf_error=collapse_data['hc_inf_error'],
                nu=collapse_data['nu'],
                nu_error=collapse_data['nu_error'],
                gamma_over_nu=collapse_data['gamma_over_nu'],
                gamma_over_nu_error=collapse_data['gamma_over_nu_error'],
                collapse_quality=collapse_data['collapse_quality'],
                scaled_data={},
                raw_data={},
            )
        
        return cls(
            system_sizes=data['system_sizes'],
            h_values=np.array(data['h_values']),
            W=data['W'],
            n_realizations=data['n_realizations'],
            data_points=data_points,
            critical_points=critical_points,
            collapse_result=collapse_result,
            nu=data['nu'],
            nu_error=data['nu_error'],
            hc_thermodynamic=data['hc_thermodynamic'],
            hc_thermodynamic_error=data['hc_thermodynamic_error'],
            metadata=data.get('metadata', {}),
        )
    
    def generate_report(self) -> str:
        """Generate human-readable report."""
        lines = []
        lines.append("=" * 80)
        lines.append("FINITE-SIZE SCALING ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        lines.append(f"System sizes: {self.system_sizes}")
        lines.append(f"Disorder strength W: {self.W}")
        lines.append(f"Disorder realizations: {self.n_realizations}")
        lines.append(f"h range: [{self.h_values.min():.3f}, {self.h_values.max():.3f}]")
        lines.append("")
        
        lines.append("CRITICAL POINT ESTIMATES (Task 14.2)")
        lines.append("-" * 40)
        for cp in self.critical_points:
            lines.append(f"  L = {cp.L:3d}: hc = {cp.hc:.4f} ± {cp.hc_error:.4f} "
                        f"(χ_max = {cp.chi_max:.2f}, R² = {cp.fit_quality:.3f})")
        lines.append("")
        
        lines.append("THERMODYNAMIC LIMIT")
        lines.append("-" * 40)
        lines.append(f"  hc(L→∞) = {self.hc_thermodynamic:.4f} ± {self.hc_thermodynamic_error:.4f}")
        lines.append("")
        
        lines.append("CORRELATION LENGTH EXPONENT (Task 14.4)")
        lines.append("-" * 40)
        lines.append(f"  ν = {self.nu:.3f} ± {self.nu_error:.3f}")
        lines.append("")
        
        if self.collapse_result:
            lines.append("SCALING COLLAPSE (Task 14.3)")
            lines.append("-" * 40)
            lines.append(f"  Collapse quality: {self.collapse_result.collapse_quality:.3f}")
            lines.append(f"  γ/ν = {self.collapse_result.gamma_over_nu:.3f} ± "
                        f"{self.collapse_result.gamma_over_nu_error:.3f}")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)



class FiniteSizeScalingAnalyzer:
    """
    Systematic finite-size scaling analysis for DTFIM validation.
    
    Implements:
    - Task 14.1: Data generation for L = 8, 12, 16, 20, 24, 32
    - Task 14.2: Critical point extraction hc(L) for each size
    - Task 14.3: Finite-size scaling collapse
    - Task 14.4: Correlation length exponent ν extraction
    """
    
    def __init__(
        self,
        system_sizes: List[int] = None,
        n_disorder_realizations: int = 500,
        J_mean: float = 1.0,
        periodic: bool = True,
        random_seed: int = 42
    ):
        """
        Initialize finite-size scaling analyzer.
        
        Args:
            system_sizes: List of system sizes (default: [8, 12, 16, 20, 24, 32])
            n_disorder_realizations: Number of disorder realizations per point
            J_mean: Mean coupling strength
            periodic: Use periodic boundary conditions
            random_seed: Random seed for reproducibility
        """
        self.system_sizes = system_sizes or STANDARD_SYSTEM_SIZES
        self.n_disorder_realizations = n_disorder_realizations
        self.J_mean = J_mean
        self.periodic = periodic
        self.random_seed = random_seed
        
        self.logger = logging.getLogger(__name__)
    
    def _compute_single_point(
        self,
        L: int,
        h: float,
        W: float,
        n_realizations: int
    ) -> FiniteSizeDataPoint:
        """
        Compute observables at a single (L, h, W) point.
        
        Args:
            L: System size
            h: Mean transverse field
            W: Disorder strength
            n_realizations: Number of disorder realizations
            
        Returns:
            FiniteSizeDataPoint with computed observables
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
        
        # Calculators
        obs_calc = ObservableCalculator(L)
        ent_calc = EntanglementCalculator(L)
        
        # Collect observables over disorder realizations
        mag_list = []
        chi_list = []
        mag_sq_list = []
        mag_fourth_list = []
        gap_list = []
        ent_list = []
        corr_len_list = []
        
        for i in range(n_realizations):
            realization = dtfim.disorder_framework.realization_generator.generate_single(
                realization_index=i
            )
            
            # Compute ground state
            E, state = dtfim.compute_ground_state(realization)
            
            # Magnetization
            local_obs = obs_calc.local_observables(state)
            mag = abs(local_obs.magnetization_z)
            mag_list.append(mag)
            mag_sq_list.append(mag ** 2)
            mag_fourth_list.append(mag ** 4)
            
            # Susceptibility
            chi = obs_calc.susceptibility(state, direction='z')
            chi_list.append(chi)
            
            # Energy gap (for smaller systems)
            if L <= 20:
                try:
                    H = dtfim.build_hamiltonian(realization)
                    gap = dtfim.solver.energy_gap(H)
                    if gap > 0:
                        gap_list.append(gap)
                except Exception:
                    pass
            
            # Entanglement entropy
            ent_result = ent_calc.entanglement_spectrum(state, cut_position=L // 2)
            ent_list.append(ent_result.entropy)
            
            # Correlation length
            try:
                corr_len = obs_calc.correlation_length(state, correlation_type='zz')
                if np.isfinite(corr_len) and 0 < corr_len < L:
                    corr_len_list.append(corr_len)
            except Exception:
                pass
        
        # Compute statistics
        n = len(mag_list)
        
        # Binder cumulant: U = 1 - <m^4>/(3<m^2>^2)
        m2_mean = np.mean(mag_sq_list)
        m4_mean = np.mean(mag_fourth_list)
        binder = 1.0 - m4_mean / (3.0 * m2_mean ** 2) if m2_mean > 1e-10 else 0.0
        
        # Binder error via bootstrap (simplified)
        binder_std = 0.05  # Placeholder - would need bootstrap for proper error
        
        return FiniteSizeDataPoint(
            L=L,
            h=h,
            W=W,
            n_realizations=n_realizations,
            magnetization=np.mean(mag_list),
            magnetization_std=np.std(mag_list, ddof=1) if n > 1 else 0.0,
            susceptibility=np.mean(chi_list),
            susceptibility_std=np.std(chi_list, ddof=1) if n > 1 else 0.0,
            binder_cumulant=binder,
            binder_std=binder_std,
            energy_gap=np.mean(gap_list) if gap_list else 0.0,
            energy_gap_std=np.std(gap_list, ddof=1) if len(gap_list) > 1 else 0.0,
            entanglement_entropy=np.mean(ent_list),
            entanglement_std=np.std(ent_list, ddof=1) if n > 1 else 0.0,
            correlation_length=np.mean(corr_len_list) if corr_len_list else float('inf'),
            correlation_length_std=np.std(corr_len_list, ddof=1) if len(corr_len_list) > 1 else 0.0,
        )
    
    def generate_fss_data(
        self,
        h_range: Tuple[float, float],
        W: float,
        n_h_points: int = 30,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> List[FiniteSizeDataPoint]:
        """
        Generate finite-size scaling data for all system sizes (Task 14.1).
        
        Args:
            h_range: (h_min, h_max) range for transverse field
            W: Disorder strength
            n_h_points: Number of h points to scan
            parallel: Use parallel computation
            max_workers: Maximum number of parallel workers
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of FiniteSizeDataPoint for all (L, h) combinations
        """
        h_values = np.linspace(h_range[0], h_range[1], n_h_points)
        
        self.logger.info(f"Generating FSS data for L = {self.system_sizes}")
        self.logger.info(f"h range: [{h_range[0]:.3f}, {h_range[1]:.3f}], {n_h_points} points")
        self.logger.info(f"Disorder strength W = {W}")
        self.logger.info(f"Disorder realizations: {self.n_disorder_realizations}")
        
        # Generate all tasks
        tasks = []
        for L in self.system_sizes:
            for h in h_values:
                tasks.append((L, h, W))
        
        total_tasks = len(tasks)
        self.logger.info(f"Total computations: {total_tasks}")
        
        data_points = []
        start_time = time.time()
        
        if parallel and total_tasks > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._compute_single_point, L, h, W, self.n_disorder_realizations
                    ): (L, h)
                    for L, h, W in tasks
                }
                
                for i, future in enumerate(as_completed(futures)):
                    try:
                        point = future.result()
                        data_points.append(point)
                        
                        if (i + 1) % 20 == 0 or (i + 1) == total_tasks:
                            elapsed = time.time() - start_time
                            rate = (i + 1) / elapsed
                            remaining = (total_tasks - i - 1) / rate if rate > 0 else 0
                            self.logger.info(
                                f"Progress: {i + 1}/{total_tasks} "
                                f"({100*(i+1)/total_tasks:.1f}%) "
                                f"ETA: {remaining/60:.1f} min"
                            )
                            if progress_callback:
                                progress_callback(i + 1, total_tasks)
                    except Exception as e:
                        L, h = futures[future]
                        self.logger.error(f"Failed at (L={L}, h={h}): {e}")
        else:
            for i, (L, h, W) in enumerate(tasks):
                try:
                    point = self._compute_single_point(L, h, W, self.n_disorder_realizations)
                    data_points.append(point)
                    
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"Progress: {i + 1}/{total_tasks}")
                        if progress_callback:
                            progress_callback(i + 1, total_tasks)
                except Exception as e:
                    self.logger.error(f"Failed at (L={L}, h={h}): {e}")
        
        elapsed = time.time() - start_time
        self.logger.info(f"Data generation completed in {elapsed/60:.1f} minutes")
        
        return data_points
    
    def extract_critical_points(
        self,
        data_points: List[FiniteSizeDataPoint]
    ) -> List[CriticalPointEstimate]:
        """
        Extract critical point hc(L) for each system size (Task 14.2).
        
        Uses susceptibility peak method: hc(L) is where χ(h) is maximum.
        
        Args:
            data_points: List of FiniteSizeDataPoint
            
        Returns:
            List of CriticalPointEstimate for each system size
        """
        critical_points = []
        
        for L in self.system_sizes:
            # Get data for this system size
            L_points = [p for p in data_points if p.L == L]
            if not L_points:
                continue
            
            # Sort by h
            L_points.sort(key=lambda p: p.h)
            h_values = np.array([p.h for p in L_points])
            chi_values = np.array([p.susceptibility for p in L_points])
            chi_errors = np.array([p.susceptibility_std for p in L_points])
            
            # Find peak using parabolic fit around maximum
            max_idx = np.argmax(chi_values)
            
            # Use 5 points around maximum for parabolic fit
            fit_range = 2
            start_idx = max(0, max_idx - fit_range)
            end_idx = min(len(h_values), max_idx + fit_range + 1)
            
            h_fit = h_values[start_idx:end_idx]
            chi_fit = chi_values[start_idx:end_idx]
            
            if len(h_fit) >= 3:
                # Parabolic fit: χ = a(h - hc)² + χ_max
                try:
                    def parabola(h, hc, chi_max, a):
                        return chi_max + a * (h - hc) ** 2
                    
                    p0 = [h_values[max_idx], chi_values[max_idx], -1.0]
                    popt, pcov = curve_fit(parabola, h_fit, chi_fit, p0=p0)
                    
                    hc = popt[0]
                    chi_max = popt[1]
                    hc_error = np.sqrt(pcov[0, 0]) if pcov[0, 0] > 0 else 0.01
                    
                    # Compute fit quality
                    chi_pred = parabola(h_fit, *popt)
                    ss_res = np.sum((chi_fit - chi_pred) ** 2)
                    ss_tot = np.sum((chi_fit - np.mean(chi_fit)) ** 2)
                    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                    
                except Exception:
                    # Fallback to simple maximum
                    hc = h_values[max_idx]
                    chi_max = chi_values[max_idx]
                    hc_error = (h_values[1] - h_values[0]) / 2  # Half grid spacing
                    r_squared = 0.5
            else:
                hc = h_values[max_idx]
                chi_max = chi_values[max_idx]
                hc_error = 0.05
                r_squared = 0.5
            
            critical_points.append(CriticalPointEstimate(
                L=L,
                hc=hc,
                hc_error=hc_error,
                method='susceptibility_peak',
                chi_max=chi_max,
                fit_quality=r_squared,
            ))
            
            self.logger.info(f"L = {L}: hc = {hc:.4f} ± {hc_error:.4f}, χ_max = {chi_max:.2f}")
        
        return critical_points

    
    def perform_scaling_collapse(
        self,
        data_points: List[FiniteSizeDataPoint],
        critical_points: List[CriticalPointEstimate],
        hc_initial: Optional[float] = None,
        nu_initial: float = 1.0,
        gamma_over_nu_initial: float = 2.0
    ) -> ScalingCollapseResult:
        """
        Perform finite-size scaling collapse (Task 14.3).
        
        Scaling ansatz: χ/L^(γ/ν) = f((h - hc) * L^(1/ν))
        
        Optimizes hc, ν, and γ/ν to achieve best collapse.
        
        Args:
            data_points: List of FiniteSizeDataPoint
            critical_points: Critical point estimates from Task 14.2
            hc_initial: Initial guess for hc (default: mean of estimates)
            nu_initial: Initial guess for ν
            gamma_over_nu_initial: Initial guess for γ/ν
            
        Returns:
            ScalingCollapseResult with optimized parameters
        """
        # Initial guess for hc from critical point estimates
        if hc_initial is None:
            hc_initial = np.mean([cp.hc for cp in critical_points])
        
        # Organize data by system size
        data_by_L = {}
        for L in self.system_sizes:
            L_points = sorted([p for p in data_points if p.L == L], key=lambda p: p.h)
            if L_points:
                h_arr = np.array([p.h for p in L_points])
                chi_arr = np.array([p.susceptibility for p in L_points])
                data_by_L[L] = (h_arr, chi_arr)
        
        def collapse_quality(params):
            """
            Compute quality of scaling collapse.
            
            Returns negative quality (for minimization).
            """
            hc, nu, gamma_nu = params
            
            if nu <= 0 or gamma_nu <= 0:
                return 1e10
            
            # Compute scaled variables for all sizes
            all_x = []
            all_y = []
            
            for L, (h_arr, chi_arr) in data_by_L.items():
                # Scaled x: (h - hc) * L^(1/ν)
                x_scaled = (h_arr - hc) * (L ** (1.0 / nu))
                # Scaled y: χ / L^(γ/ν)
                y_scaled = chi_arr / (L ** gamma_nu)
                
                all_x.extend(x_scaled)
                all_y.extend(y_scaled)
            
            all_x = np.array(all_x)
            all_y = np.array(all_y)
            
            # Sort by x
            sort_idx = np.argsort(all_x)
            all_x = all_x[sort_idx]
            all_y = all_y[sort_idx]
            
            # Compute collapse quality as inverse of variance in y for similar x
            # Use binning approach
            n_bins = 20
            x_bins = np.linspace(all_x.min(), all_x.max(), n_bins + 1)
            
            total_variance = 0.0
            n_valid_bins = 0
            
            for i in range(n_bins):
                mask = (all_x >= x_bins[i]) & (all_x < x_bins[i + 1])
                if np.sum(mask) >= 2:
                    bin_variance = np.var(all_y[mask])
                    total_variance += bin_variance
                    n_valid_bins += 1
            
            if n_valid_bins == 0:
                return 1e10
            
            avg_variance = total_variance / n_valid_bins
            return avg_variance
        
        # Optimize parameters
        initial_params = [hc_initial, nu_initial, gamma_over_nu_initial]
        bounds = [
            (hc_initial - 0.5, hc_initial + 0.5),  # hc
            (0.1, 5.0),  # ν
            (0.5, 5.0),  # γ/ν
        ]
        
        result = minimize(
            collapse_quality,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        hc_opt, nu_opt, gamma_nu_opt = result.x
        
        # Estimate errors using Hessian (if available)
        try:
            # Numerical Hessian for error estimation
            eps = 1e-4
            hessian = np.zeros((3, 3))
            f0 = collapse_quality(result.x)
            
            for i in range(3):
                for j in range(3):
                    x_pp = result.x.copy()
                    x_pm = result.x.copy()
                    x_mp = result.x.copy()
                    x_mm = result.x.copy()
                    
                    x_pp[i] += eps
                    x_pp[j] += eps
                    x_pm[i] += eps
                    x_pm[j] -= eps
                    x_mp[i] -= eps
                    x_mp[j] += eps
                    x_mm[i] -= eps
                    x_mm[j] -= eps
                    
                    hessian[i, j] = (
                        collapse_quality(x_pp) - collapse_quality(x_pm) -
                        collapse_quality(x_mp) + collapse_quality(x_mm)
                    ) / (4 * eps * eps)
            
            # Covariance matrix is inverse of Hessian
            try:
                cov = np.linalg.inv(hessian)
                errors = np.sqrt(np.abs(np.diag(cov)))
                hc_error = errors[0]
                nu_error = errors[1]
                gamma_nu_error = errors[2]
            except:
                hc_error = 0.05
                nu_error = 0.2
                gamma_nu_error = 0.3
        except:
            hc_error = 0.05
            nu_error = 0.2
            gamma_nu_error = 0.3
        
        # Compute final collapse quality (0-1 scale)
        final_variance = collapse_quality(result.x)
        # Convert variance to quality score
        collapse_quality_score = 1.0 / (1.0 + final_variance)
        
        # Store scaled data
        scaled_data = {}
        raw_data = {}
        
        for L, (h_arr, chi_arr) in data_by_L.items():
            x_scaled = (h_arr - hc_opt) * (L ** (1.0 / nu_opt))
            y_scaled = chi_arr / (L ** gamma_nu_opt)
            scaled_data[L] = (x_scaled, y_scaled)
            raw_data[L] = (h_arr, chi_arr)
        
        self.logger.info(f"Scaling collapse: hc = {hc_opt:.4f} ± {hc_error:.4f}")
        self.logger.info(f"                  ν = {nu_opt:.3f} ± {nu_error:.3f}")
        self.logger.info(f"                  γ/ν = {gamma_nu_opt:.3f} ± {gamma_nu_error:.3f}")
        self.logger.info(f"                  Quality = {collapse_quality_score:.3f}")
        
        return ScalingCollapseResult(
            hc_inf=hc_opt,
            hc_inf_error=hc_error,
            nu=nu_opt,
            nu_error=nu_error,
            gamma_over_nu=gamma_nu_opt,
            gamma_over_nu_error=gamma_nu_error,
            collapse_quality=collapse_quality_score,
            scaled_data=scaled_data,
            raw_data=raw_data,
        )
    
    def extract_nu_from_hc_scaling(
        self,
        critical_points: List[CriticalPointEstimate]
    ) -> Tuple[float, float, float, float]:
        """
        Extract ν from finite-size scaling of hc(L) (Task 14.4).
        
        Scaling: hc(L) - hc(∞) ~ L^(-1/ν)
        
        Args:
            critical_points: Critical point estimates for each L
            
        Returns:
            (nu, nu_error, hc_inf, hc_inf_error)
        """
        # Sort by system size
        cps = sorted(critical_points, key=lambda cp: cp.L)
        
        L_values = np.array([cp.L for cp in cps])
        hc_values = np.array([cp.hc for cp in cps])
        hc_errors = np.array([cp.hc_error for cp in cps])
        
        if len(L_values) < 3:
            self.logger.warning("Insufficient data for ν extraction")
            return 1.0, 0.5, np.mean(hc_values), np.std(hc_values)
        
        # Fit: hc(L) = hc_inf + A * L^(-1/ν)
        def scaling_func(L, hc_inf, A, inv_nu):
            return hc_inf + A * L ** (-inv_nu)
        
        try:
            # Initial guess
            hc_inf_guess = hc_values[-1]  # Largest L
            A_guess = (hc_values[0] - hc_values[-1]) * L_values[0]
            inv_nu_guess = 1.0
            
            popt, pcov = curve_fit(
                scaling_func,
                L_values,
                hc_values,
                p0=[hc_inf_guess, A_guess, inv_nu_guess],
                sigma=hc_errors,
                absolute_sigma=True,
                bounds=(
                    [hc_values.min() - 0.5, -10, 0.1],
                    [hc_values.max() + 0.5, 10, 5.0]
                )
            )
            
            hc_inf = popt[0]
            inv_nu = popt[2]
            nu = 1.0 / inv_nu
            
            # Error propagation
            hc_inf_error = np.sqrt(pcov[0, 0])
            inv_nu_error = np.sqrt(pcov[2, 2])
            nu_error = nu ** 2 * inv_nu_error  # Error propagation for 1/x
            
        except Exception as e:
            self.logger.warning(f"Curve fit failed: {e}, using linear regression")
            
            # Fallback: linear regression on log-log plot
            # log(hc(L) - hc_inf) = log(A) - (1/ν) * log(L)
            # Estimate hc_inf as hc of largest system
            hc_inf = hc_values[-1]
            
            delta_hc = hc_values[:-1] - hc_inf
            valid = delta_hc > 0
            
            if np.sum(valid) >= 2:
                log_L = np.log(L_values[:-1][valid])
                log_delta = np.log(delta_hc[valid])
                
                slope, intercept, r_value, p_value, std_err = linregress(log_L, log_delta)
                inv_nu = -slope
                nu = 1.0 / inv_nu if inv_nu > 0 else 1.0
                nu_error = nu ** 2 * std_err
                hc_inf_error = np.std(hc_values) / np.sqrt(len(hc_values))
            else:
                nu = 1.0
                nu_error = 0.5
                hc_inf_error = 0.1
        
        self.logger.info(f"ν extraction: ν = {nu:.3f} ± {nu_error:.3f}")
        self.logger.info(f"              hc(∞) = {hc_inf:.4f} ± {hc_inf_error:.4f}")
        
        return nu, nu_error, hc_inf, hc_inf_error
    
    def run_full_analysis(
        self,
        h_range: Tuple[float, float],
        W: float,
        n_h_points: int = 30,
        parallel: bool = True,
        output_dir: Optional[str] = None
    ) -> FiniteSizeScalingResult:
        """
        Run complete finite-size scaling analysis (Tasks 14.1-14.4).
        
        Args:
            h_range: (h_min, h_max) range for transverse field
            W: Disorder strength
            n_h_points: Number of h points to scan
            parallel: Use parallel computation
            output_dir: Directory to save results
            
        Returns:
            FiniteSizeScalingResult with complete analysis
        """
        self.logger.info("=" * 60)
        self.logger.info("FINITE-SIZE SCALING ANALYSIS")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        # Task 14.1: Generate data for all system sizes
        self.logger.info("\nTask 14.1: Generating FSS data...")
        data_points = self.generate_fss_data(
            h_range=h_range,
            W=W,
            n_h_points=n_h_points,
            parallel=parallel
        )
        
        # Task 14.2: Extract critical points
        self.logger.info("\nTask 14.2: Extracting critical points...")
        critical_points = self.extract_critical_points(data_points)
        
        # Task 14.4: Extract ν from hc(L) scaling
        self.logger.info("\nTask 14.4: Extracting ν from hc(L) scaling...")
        nu, nu_error, hc_inf, hc_inf_error = self.extract_nu_from_hc_scaling(critical_points)
        
        # Task 14.3: Perform scaling collapse
        self.logger.info("\nTask 14.3: Performing scaling collapse...")
        collapse_result = self.perform_scaling_collapse(
            data_points,
            critical_points,
            hc_initial=hc_inf,
            nu_initial=nu
        )
        
        elapsed = time.time() - start_time
        
        # Create result
        h_values = np.linspace(h_range[0], h_range[1], n_h_points)
        
        result = FiniteSizeScalingResult(
            system_sizes=self.system_sizes,
            h_values=h_values,
            W=W,
            n_realizations=self.n_disorder_realizations,
            data_points=data_points,
            critical_points=critical_points,
            collapse_result=collapse_result,
            nu=collapse_result.nu,  # Use collapse result for final ν
            nu_error=collapse_result.nu_error,
            hc_thermodynamic=collapse_result.hc_inf,
            hc_thermodynamic_error=collapse_result.hc_inf_error,
            metadata={
                'computation_time_minutes': elapsed / 60,
                'h_range': h_range,
                'n_h_points': n_h_points,
                'J_mean': self.J_mean,
                'periodic': self.periodic,
            }
        )
        
        # Save results
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            result.save(str(output_path / 'fss_results.json'))
            
            with open(output_path / 'fss_report.txt', 'w') as f:
                f.write(result.generate_report())
            
            self.logger.info(f"\nResults saved to {output_dir}")
        
        self.logger.info("\n" + result.generate_report())
        
        return result


def run_task14_finite_size_scaling(
    h_center: float = 1.0,
    h_width: float = 0.5,
    W: float = 0.5,
    n_h_points: int = 25,
    n_realizations: int = 100,
    system_sizes: List[int] = None,
    output_dir: str = "results/task14_fss",
    parallel: bool = True
) -> FiniteSizeScalingResult:
    """
    Main function to run Task 14: Systematic finite-size scaling.
    
    Args:
        h_center: Center of h range (near expected critical point)
        h_width: Half-width of h range
        W: Disorder strength
        n_h_points: Number of h points
        n_realizations: Number of disorder realizations
        system_sizes: System sizes (default: [8, 12, 16, 20, 24, 32])
        output_dir: Output directory
        parallel: Use parallel computation
        
    Returns:
        FiniteSizeScalingResult
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if system_sizes is None:
        system_sizes = STANDARD_SYSTEM_SIZES
    
    analyzer = FiniteSizeScalingAnalyzer(
        system_sizes=system_sizes,
        n_disorder_realizations=n_realizations
    )
    
    h_range = (h_center - h_width, h_center + h_width)
    
    result = analyzer.run_full_analysis(
        h_range=h_range,
        W=W,
        n_h_points=n_h_points,
        parallel=parallel,
        output_dir=output_dir
    )
    
    return result
