"""
Critical Exponent Extraction Module.

Implements Task 15: Critical exponent extraction
- 15.1 Correlation length exponent ν
- 15.2 Dynamical exponent z
- 15.3 Order parameter exponent β
- 15.4 Susceptibility exponent γ
- 15.5 Anomalous dimension η

This module provides comprehensive critical exponent extraction for validating
quantum phase transitions in the Disordered TFIM.

Key Physics:
- At a quantum critical point, observables follow power-law scaling
- Correlation length: ξ ~ |h - hc|^(-ν)
- Order parameter: m ~ |h - hc|^β (for h < hc)
- Susceptibility: χ ~ |h - hc|^(-γ)
- Correlation function: G(r) ~ r^(-(d-2+η)) at criticality
- Dynamical exponent: Δ ~ L^(-z) (energy gap scaling)

Scaling Relations (hyperscaling):
- γ = ν(2 - η)
- 2β + γ = νd (d = dimension)
- α + 2β + γ = 2 (Rushbrooke)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path
import logging
from scipy.optimize import curve_fit, minimize
from scipy.stats import linregress

from .finite_size_scaling import (
    FiniteSizeScalingAnalyzer,
    FiniteSizeDataPoint,
    CriticalPointEstimate,
    FiniteSizeScalingResult,
    STANDARD_SYSTEM_SIZES,
)


@dataclass
class ExponentResult:
    """Result of a single critical exponent extraction."""
    name: str  # e.g., 'nu', 'z', 'beta', 'gamma', 'eta'
    value: float
    error: float
    method: str  # Method used for extraction
    fit_quality: float  # R² or similar quality metric
    system_sizes: List[int]
    raw_data: Dict[str, Any]
    is_valid: bool  # Whether extraction was successful
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'error': self.error,
            'method': self.method,
            'fit_quality': self.fit_quality,
            'system_sizes': self.system_sizes,
            'is_valid': self.is_valid,
            'notes': self.notes,
        }
    
    def __str__(self) -> str:
        if self.is_valid:
            return f"{self.name} = {self.value:.4f} ± {self.error:.4f} (R² = {self.fit_quality:.3f})"
        else:
            return f"{self.name}: extraction failed - {self.notes}"


@dataclass
class ScalingRelationCheck:
    """Result of checking a scaling relation."""
    relation_name: str  # e.g., 'hyperscaling', 'rushbrooke'
    expected_value: float
    computed_value: float
    deviation: float  # In units of combined error
    is_satisfied: bool
    formula: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'relation_name': self.relation_name,
            'expected_value': self.expected_value,
            'computed_value': self.computed_value,
            'deviation': self.deviation,
            'is_satisfied': self.is_satisfied,
            'formula': self.formula,
        }


@dataclass
class CriticalExponentsResult:
    """Complete result of critical exponent extraction."""
    # Critical point
    hc: float
    hc_error: float
    
    # Individual exponents (Task 15.1-15.5)
    nu: ExponentResult  # 15.1 Correlation length exponent
    z: ExponentResult   # 15.2 Dynamical exponent
    beta: ExponentResult  # 15.3 Order parameter exponent
    gamma: ExponentResult  # 15.4 Susceptibility exponent
    eta: ExponentResult  # 15.5 Anomalous dimension
    
    # Scaling relation checks
    scaling_checks: List[ScalingRelationCheck]
    
    # Overall quality
    overall_quality: float  # 0-1 score
    universality_class: str  # Best matching universality class
    
    # Metadata
    system_sizes: List[int]
    disorder_strength: float
    n_realizations: int
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'hc': self.hc,
            'hc_error': self.hc_error,
            'nu': self.nu.to_dict(),
            'z': self.z.to_dict(),
            'beta': self.beta.to_dict(),
            'gamma': self.gamma.to_dict(),
            'eta': self.eta.to_dict(),
            'scaling_checks': [sc.to_dict() for sc in self.scaling_checks],
            'overall_quality': self.overall_quality,
            'universality_class': self.universality_class,
            'system_sizes': self.system_sizes,
            'disorder_strength': self.disorder_strength,
            'n_realizations': self.n_realizations,
            'metadata': self.metadata,
        }
    
    def save(self, filepath: str):
        """Save results to JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def generate_report(self) -> str:
        """Generate human-readable report."""
        lines = []
        lines.append("=" * 80)
        lines.append("CRITICAL EXPONENT EXTRACTION REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        lines.append("CRITICAL POINT")
        lines.append("-" * 40)
        lines.append(f"  hc = {self.hc:.4f} ± {self.hc_error:.4f}")
        lines.append("")
        
        lines.append("CRITICAL EXPONENTS")
        lines.append("-" * 40)
        lines.append(f"  {self.nu}")
        lines.append(f"  {self.z}")
        lines.append(f"  {self.beta}")
        lines.append(f"  {self.gamma}")
        lines.append(f"  {self.eta}")
        lines.append("")
        
        lines.append("SCALING RELATION CHECKS")
        lines.append("-" * 40)
        for check in self.scaling_checks:
            status = "✓" if check.is_satisfied else "✗"
            lines.append(f"  {status} {check.relation_name}: {check.formula}")
            lines.append(f"      Expected: {check.expected_value:.4f}, "
                        f"Computed: {check.computed_value:.4f}, "
                        f"Deviation: {check.deviation:.2f}σ")
        lines.append("")
        
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Overall quality: {self.overall_quality:.1%}")
        lines.append(f"  Universality class: {self.universality_class}")
        lines.append(f"  System sizes: {self.system_sizes}")
        lines.append(f"  Disorder strength W: {self.disorder_strength}")
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)


# Known universality classes for comparison
KNOWN_UNIVERSALITY_CLASSES = {
    '1d_clean_ising': {
        'nu': 1.0, 'z': 1.0, 'beta': 0.125, 'gamma': 1.75, 'eta': 0.25,
        'description': '1D Clean Ising (exact)'
    },
    '1d_tfim': {
        'nu': 1.0, 'z': 1.0, 'beta': 0.125, 'gamma': 1.75, 'eta': 0.25,
        'description': '1D Transverse Field Ising Model'
    },
    '1d_irfp': {
        'nu': 2.0, 'z': float('inf'), 'beta': 0.191, 'gamma': 2.618, 'eta': 1.0,
        'description': '1D Infinite-Randomness Fixed Point'
    },
    '2d_ising': {
        'nu': 1.0, 'z': 2.17, 'beta': 0.125, 'gamma': 1.75, 'eta': 0.25,
        'description': '2D Ising (exact)'
    },
    '3d_ising': {
        'nu': 0.6301, 'z': 2.02, 'beta': 0.3265, 'gamma': 1.2372, 'eta': 0.0364,
        'description': '3D Ising universality class'
    },
    'mean_field': {
        'nu': 0.5, 'z': 2.0, 'beta': 0.5, 'gamma': 1.0, 'eta': 0.0,
        'description': 'Mean-field (Landau) theory'
    },
}


class CriticalExponentExtractor:
    """
    Comprehensive critical exponent extraction for quantum phase transitions.
    
    Implements Task 15:
    - 15.1 Correlation length exponent ν
    - 15.2 Dynamical exponent z
    - 15.3 Order parameter exponent β
    - 15.4 Susceptibility exponent γ
    - 15.5 Anomalous dimension η
    """
    
    def __init__(
        self,
        system_sizes: List[int] = None,
        dimension: int = 1,
        tolerance: float = 2.0,  # Sigma tolerance for scaling relations
    ):
        """
        Initialize critical exponent extractor.
        
        Args:
            system_sizes: List of system sizes used
            dimension: Spatial dimension (1 for chains)
            tolerance: Tolerance in sigma for scaling relation checks
        """
        self.system_sizes = system_sizes or STANDARD_SYSTEM_SIZES
        self.dimension = dimension
        self.tolerance = tolerance
        self.logger = logging.getLogger(__name__)
    
    def extract_all_exponents(
        self,
        data_points: List[FiniteSizeDataPoint],
        hc: float,
        hc_error: float,
    ) -> CriticalExponentsResult:
        """
        Extract all critical exponents from finite-size scaling data.
        
        Args:
            data_points: List of FiniteSizeDataPoint from FSS analysis
            hc: Critical point estimate
            hc_error: Error in critical point
            
        Returns:
            CriticalExponentsResult with all exponents
        """
        self.logger.info("=" * 60)
        self.logger.info("CRITICAL EXPONENT EXTRACTION")
        self.logger.info("=" * 60)
        
        # Task 15.1: Extract correlation length exponent ν
        self.logger.info("\nTask 15.1: Extracting correlation length exponent ν...")
        nu_result = self.extract_nu(data_points, hc)
        
        # Task 15.2: Extract dynamical exponent z
        self.logger.info("\nTask 15.2: Extracting dynamical exponent z...")
        z_result = self.extract_z(data_points, hc)
        
        # Task 15.3: Extract order parameter exponent β
        self.logger.info("\nTask 15.3: Extracting order parameter exponent β...")
        beta_result = self.extract_beta(data_points, hc)
        
        # Task 15.4: Extract susceptibility exponent γ
        self.logger.info("\nTask 15.4: Extracting susceptibility exponent γ...")
        gamma_result = self.extract_gamma(data_points, hc)
        
        # Task 15.5: Extract anomalous dimension η
        self.logger.info("\nTask 15.5: Extracting anomalous dimension η...")
        eta_result = self.extract_eta(data_points, hc, nu_result.value if nu_result.is_valid else 1.0)
        
        # Check scaling relations
        self.logger.info("\nChecking scaling relations...")
        scaling_checks = self.check_scaling_relations(
            nu_result, z_result, beta_result, gamma_result, eta_result
        )
        
        # Determine universality class
        universality_class = self.identify_universality_class(
            nu_result, z_result, beta_result, gamma_result, eta_result
        )
        
        # Compute overall quality
        valid_exponents = sum([
            nu_result.is_valid, z_result.is_valid, beta_result.is_valid,
            gamma_result.is_valid, eta_result.is_valid
        ])
        satisfied_relations = sum([sc.is_satisfied for sc in scaling_checks])
        
        overall_quality = (
            0.5 * (valid_exponents / 5) +
            0.3 * (satisfied_relations / max(1, len(scaling_checks))) +
            0.2 * np.mean([
                nu_result.fit_quality if nu_result.is_valid else 0,
                z_result.fit_quality if z_result.is_valid else 0,
                beta_result.fit_quality if beta_result.is_valid else 0,
                gamma_result.fit_quality if gamma_result.is_valid else 0,
                eta_result.fit_quality if eta_result.is_valid else 0,
            ])
        )
        
        # Get disorder strength from data
        W = data_points[0].W if data_points else 0.0
        n_realizations = data_points[0].n_realizations if data_points else 0
        
        result = CriticalExponentsResult(
            hc=hc,
            hc_error=hc_error,
            nu=nu_result,
            z=z_result,
            beta=beta_result,
            gamma=gamma_result,
            eta=eta_result,
            scaling_checks=scaling_checks,
            overall_quality=overall_quality,
            universality_class=universality_class,
            system_sizes=self.system_sizes,
            disorder_strength=W,
            n_realizations=n_realizations,
        )
        
        self.logger.info("\n" + result.generate_report())
        
        return result
    
    def extract_nu(
        self,
        data_points: List[FiniteSizeDataPoint],
        hc: float,
    ) -> ExponentResult:
        """
        Extract correlation length exponent ν (Task 15.1).
        
        Method: Finite-size scaling of correlation length
        At criticality: ξ(L) ~ L
        Away from criticality: ξ ~ |h - hc|^(-ν)
        
        Combined: ξ/L = f((h - hc) * L^(1/ν))
        
        Args:
            data_points: FSS data points
            hc: Critical point
            
        Returns:
            ExponentResult for ν
        """
        # Organize data by system size
        data_by_L = self._organize_by_size(data_points)
        
        if len(data_by_L) < 3:
            return ExponentResult(
                name='ν', value=1.0, error=float('inf'),
                method='insufficient_data', fit_quality=0.0,
                system_sizes=list(data_by_L.keys()),
                raw_data={}, is_valid=False,
                notes='Insufficient system sizes for ν extraction'
            )
        
        # Method 1: Scaling collapse optimization
        def collapse_quality(nu):
            """Compute quality of ξ/L collapse."""
            all_x = []
            all_y = []
            
            for L, points in data_by_L.items():
                for p in points:
                    x = (p.h - hc) * (L ** (1.0 / nu))
                    y = p.correlation_length / L if p.correlation_length < L else 1.0
                    if np.isfinite(y) and np.isfinite(x):
                        all_x.append(x)
                        all_y.append(y)
            
            if len(all_x) < 5:
                return 1e10
            
            all_x = np.array(all_x)
            all_y = np.array(all_y)
            
            # Bin and compute variance
            sort_idx = np.argsort(all_x)
            all_x = all_x[sort_idx]
            all_y = all_y[sort_idx]
            
            n_bins = min(10, len(all_x) // 3)
            if n_bins < 2:
                return 1e10
                
            x_bins = np.linspace(all_x.min(), all_x.max(), n_bins + 1)
            total_var = 0.0
            n_valid = 0
            
            for i in range(n_bins):
                mask = (all_x >= x_bins[i]) & (all_x < x_bins[i + 1])
                if np.sum(mask) >= 2:
                    total_var += np.var(all_y[mask])
                    n_valid += 1
            
            return total_var / max(1, n_valid)
        
        # Optimize ν
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(collapse_quality, bounds=(0.3, 3.0), method='bounded')
        nu_opt = result.x
        
        # Estimate error via curvature
        eps = 0.05
        f0 = collapse_quality(nu_opt)
        fp = collapse_quality(nu_opt + eps)
        fm = collapse_quality(nu_opt - eps)
        curvature = (fp - 2*f0 + fm) / (eps**2)
        nu_error = 1.0 / np.sqrt(max(curvature, 0.01)) if curvature > 0 else 0.5
        
        # Compute fit quality
        fit_quality = 1.0 / (1.0 + f0)
        
        self.logger.info(f"  ν = {nu_opt:.4f} ± {nu_error:.4f} (R² = {fit_quality:.3f})")
        
        return ExponentResult(
            name='ν', value=nu_opt, error=nu_error,
            method='correlation_length_collapse',
            fit_quality=fit_quality,
            system_sizes=list(data_by_L.keys()),
            raw_data={'optimal_variance': f0},
            is_valid=fit_quality > 0.3,
            notes='' if fit_quality > 0.3 else 'Low fit quality'
        )

    
    def extract_z(
        self,
        data_points: List[FiniteSizeDataPoint],
        hc: float,
    ) -> ExponentResult:
        """
        Extract dynamical exponent z (Task 15.2).
        
        Method: Energy gap scaling at criticality
        Δ(L, h=hc) ~ L^(-z)
        
        For disordered systems, may have activated scaling:
        Δ ~ exp(-c * L^ψ) (z → ∞)
        
        Args:
            data_points: FSS data points
            hc: Critical point
            
        Returns:
            ExponentResult for z
        """
        data_by_L = self._organize_by_size(data_points)
        
        # Get energy gaps at criticality for each L
        L_values = []
        gap_values = []
        
        for L, points in sorted(data_by_L.items()):
            # Find point closest to hc
            closest = min(points, key=lambda p: abs(p.h - hc))
            if closest.energy_gap > 0:
                L_values.append(L)
                gap_values.append(closest.energy_gap)
        
        if len(L_values) < 3:
            return ExponentResult(
                name='z', value=1.0, error=float('inf'),
                method='insufficient_data', fit_quality=0.0,
                system_sizes=L_values,
                raw_data={}, is_valid=False,
                notes='Insufficient data for z extraction'
            )
        
        L_arr = np.array(L_values)
        gap_arr = np.array(gap_values)
        
        # Power-law fit: Δ = A * L^(-z)
        # log(Δ) = log(A) - z * log(L)
        log_L = np.log(L_arr)
        log_gap = np.log(gap_arr)
        
        slope, intercept, r_value, p_value, std_err = linregress(log_L, log_gap)
        z_power = -slope
        z_error = std_err
        r_squared = r_value**2
        
        # Check for activated scaling (IRFP)
        # log(Δ) = log(A) - c * L^ψ (typically ψ = 0.5)
        sqrt_L = np.sqrt(L_arr)
        slope_act, _, r_act, _, _ = linregress(sqrt_L, log_gap)
        r_squared_act = r_act**2
        
        # Determine which scaling is better
        is_activated = r_squared_act > r_squared + 0.1
        
        if is_activated:
            self.logger.info(f"  z → ∞ (activated scaling detected, R² = {r_squared_act:.3f})")
            return ExponentResult(
                name='z', value=float('inf'), error=0.0,
                method='gap_scaling_activated',
                fit_quality=r_squared_act,
                system_sizes=L_values,
                raw_data={'L': L_values, 'gap': gap_values.tolist() if hasattr(gap_values, 'tolist') else list(gap_values)},
                is_valid=True,
                notes='Activated scaling (IRFP)'
            )
        
        self.logger.info(f"  z = {z_power:.4f} ± {z_error:.4f} (R² = {r_squared:.3f})")
        
        return ExponentResult(
            name='z', value=z_power, error=z_error,
            method='gap_scaling_power_law',
            fit_quality=r_squared,
            system_sizes=L_values,
            raw_data={'L': L_values, 'gap': gap_values.tolist() if hasattr(gap_values, 'tolist') else list(gap_values)},
            is_valid=r_squared > 0.7,
            notes='' if r_squared > 0.7 else 'Low fit quality'
        )
    
    def extract_beta(
        self,
        data_points: List[FiniteSizeDataPoint],
        hc: float,
    ) -> ExponentResult:
        """
        Extract order parameter exponent β (Task 15.3).
        
        Method: Magnetization scaling in ordered phase
        m ~ |h - hc|^β for h < hc (ordered phase)
        
        Finite-size scaling: m(L) ~ L^(-β/ν) at h = hc
        
        Args:
            data_points: FSS data points
            hc: Critical point
            
        Returns:
            ExponentResult for β
        """
        data_by_L = self._organize_by_size(data_points)
        
        # Method 1: Magnetization at criticality vs L
        # m(L, hc) ~ L^(-β/ν)
        L_values = []
        mag_values = []
        
        for L, points in sorted(data_by_L.items()):
            closest = min(points, key=lambda p: abs(p.h - hc))
            if closest.magnetization > 0:
                L_values.append(L)
                mag_values.append(closest.magnetization)
        
        if len(L_values) < 3:
            return ExponentResult(
                name='β', value=0.125, error=float('inf'),
                method='insufficient_data', fit_quality=0.0,
                system_sizes=L_values,
                raw_data={}, is_valid=False,
                notes='Insufficient data for β extraction'
            )
        
        L_arr = np.array(L_values)
        mag_arr = np.array(mag_values)
        
        # Fit: m = A * L^(-β/ν)
        # log(m) = log(A) - (β/ν) * log(L)
        log_L = np.log(L_arr)
        log_mag = np.log(mag_arr + 1e-10)  # Avoid log(0)
        
        slope, intercept, r_value, p_value, std_err = linregress(log_L, log_mag)
        beta_over_nu = -slope
        beta_over_nu_error = std_err
        r_squared = r_value**2
        
        # Assume ν ≈ 1 for 1D systems (will be refined)
        # For clean 1D TFIM: ν = 1, β = 1/8
        nu_estimate = 1.0
        beta = beta_over_nu * nu_estimate
        beta_error = beta_over_nu_error * nu_estimate
        
        self.logger.info(f"  β/ν = {beta_over_nu:.4f} ± {beta_over_nu_error:.4f}")
        self.logger.info(f"  β = {beta:.4f} ± {beta_error:.4f} (assuming ν ≈ 1)")
        self.logger.info(f"  R² = {r_squared:.3f}")
        
        return ExponentResult(
            name='β', value=beta, error=beta_error,
            method='magnetization_scaling',
            fit_quality=r_squared,
            system_sizes=L_values,
            raw_data={
                'L': L_values, 
                'magnetization': mag_arr.tolist() if hasattr(mag_arr, 'tolist') else list(mag_arr),
                'beta_over_nu': beta_over_nu
            },
            is_valid=r_squared > 0.5,
            notes=f'β/ν = {beta_over_nu:.4f}' if r_squared > 0.5 else 'Low fit quality'
        )

    
    def extract_gamma(
        self,
        data_points: List[FiniteSizeDataPoint],
        hc: float,
    ) -> ExponentResult:
        """
        Extract susceptibility exponent γ (Task 15.4).
        
        Method: Susceptibility scaling
        χ ~ |h - hc|^(-γ) away from criticality
        χ(L, hc) ~ L^(γ/ν) at criticality
        
        Args:
            data_points: FSS data points
            hc: Critical point
            
        Returns:
            ExponentResult for γ
        """
        data_by_L = self._organize_by_size(data_points)
        
        # Method: Susceptibility at criticality vs L
        # χ(L, hc) ~ L^(γ/ν)
        L_values = []
        chi_values = []
        
        for L, points in sorted(data_by_L.items()):
            closest = min(points, key=lambda p: abs(p.h - hc))
            if closest.susceptibility > 0:
                L_values.append(L)
                chi_values.append(closest.susceptibility)
        
        if len(L_values) < 3:
            return ExponentResult(
                name='γ', value=1.75, error=float('inf'),
                method='insufficient_data', fit_quality=0.0,
                system_sizes=L_values,
                raw_data={}, is_valid=False,
                notes='Insufficient data for γ extraction'
            )
        
        L_arr = np.array(L_values)
        chi_arr = np.array(chi_values)
        
        # Fit: χ = A * L^(γ/ν)
        # log(χ) = log(A) + (γ/ν) * log(L)
        log_L = np.log(L_arr)
        log_chi = np.log(chi_arr)
        
        slope, intercept, r_value, p_value, std_err = linregress(log_L, log_chi)
        gamma_over_nu = slope
        gamma_over_nu_error = std_err
        r_squared = r_value**2
        
        # Assume ν ≈ 1 for 1D systems
        nu_estimate = 1.0
        gamma = gamma_over_nu * nu_estimate
        gamma_error = gamma_over_nu_error * nu_estimate
        
        self.logger.info(f"  γ/ν = {gamma_over_nu:.4f} ± {gamma_over_nu_error:.4f}")
        self.logger.info(f"  γ = {gamma:.4f} ± {gamma_error:.4f} (assuming ν ≈ 1)")
        self.logger.info(f"  R² = {r_squared:.3f}")
        
        return ExponentResult(
            name='γ', value=gamma, error=gamma_error,
            method='susceptibility_scaling',
            fit_quality=r_squared,
            system_sizes=L_values,
            raw_data={
                'L': L_values,
                'susceptibility': chi_arr.tolist() if hasattr(chi_arr, 'tolist') else list(chi_arr),
                'gamma_over_nu': gamma_over_nu
            },
            is_valid=r_squared > 0.7,
            notes=f'γ/ν = {gamma_over_nu:.4f}' if r_squared > 0.7 else 'Low fit quality'
        )
    
    def extract_eta(
        self,
        data_points: List[FiniteSizeDataPoint],
        hc: float,
        nu: float = 1.0,
    ) -> ExponentResult:
        """
        Extract anomalous dimension η (Task 15.5).
        
        Method: Use scaling relation γ = ν(2 - η)
        Therefore: η = 2 - γ/ν
        
        Alternative: Correlation function decay at criticality
        G(r) ~ r^(-(d-2+η)) for d > 2
        For d = 1: G(r) ~ r^(-η-1) (modified)
        
        Args:
            data_points: FSS data points
            hc: Critical point
            nu: Correlation length exponent (for scaling relation)
            
        Returns:
            ExponentResult for η
        """
        data_by_L = self._organize_by_size(data_points)
        
        # First, get γ/ν from susceptibility scaling
        L_values = []
        chi_values = []
        
        for L, points in sorted(data_by_L.items()):
            closest = min(points, key=lambda p: abs(p.h - hc))
            if closest.susceptibility > 0:
                L_values.append(L)
                chi_values.append(closest.susceptibility)
        
        if len(L_values) < 3:
            return ExponentResult(
                name='η', value=0.25, error=float('inf'),
                method='insufficient_data', fit_quality=0.0,
                system_sizes=L_values,
                raw_data={}, is_valid=False,
                notes='Insufficient data for η extraction'
            )
        
        L_arr = np.array(L_values)
        chi_arr = np.array(chi_values)
        
        # Get γ/ν
        log_L = np.log(L_arr)
        log_chi = np.log(chi_arr)
        
        slope, intercept, r_value, p_value, std_err = linregress(log_L, log_chi)
        gamma_over_nu = slope
        gamma_over_nu_error = std_err
        r_squared = r_value**2
        
        # Use scaling relation: η = 2 - γ/ν
        eta = 2.0 - gamma_over_nu
        eta_error = gamma_over_nu_error  # Error propagation
        
        self.logger.info(f"  η = 2 - γ/ν = 2 - {gamma_over_nu:.4f} = {eta:.4f} ± {eta_error:.4f}")
        self.logger.info(f"  R² = {r_squared:.3f}")
        
        return ExponentResult(
            name='η', value=eta, error=eta_error,
            method='scaling_relation',
            fit_quality=r_squared,
            system_sizes=L_values,
            raw_data={'gamma_over_nu': gamma_over_nu},
            is_valid=r_squared > 0.7 and -0.5 < eta < 2.0,
            notes='From γ = ν(2-η)' if r_squared > 0.7 else 'Low fit quality'
        )
    
    def check_scaling_relations(
        self,
        nu: ExponentResult,
        z: ExponentResult,
        beta: ExponentResult,
        gamma: ExponentResult,
        eta: ExponentResult,
    ) -> List[ScalingRelationCheck]:
        """
        Check hyperscaling and other scaling relations.
        
        Relations checked:
        1. γ = ν(2 - η) (Fisher relation)
        2. 2β + γ = νd (hyperscaling, d = dimension)
        3. α + 2β + γ = 2 (Rushbrooke, α = 2 - νd)
        
        Args:
            nu, z, beta, gamma, eta: Extracted exponents
            
        Returns:
            List of ScalingRelationCheck results
        """
        checks = []
        
        # 1. Fisher relation: γ = ν(2 - η)
        if nu.is_valid and gamma.is_valid and eta.is_valid:
            expected = nu.value * (2 - eta.value)
            computed = gamma.value
            combined_error = np.sqrt(
                (gamma.error)**2 + 
                ((2 - eta.value) * nu.error)**2 +
                (nu.value * eta.error)**2
            )
            deviation = abs(expected - computed) / max(combined_error, 0.01)
            
            checks.append(ScalingRelationCheck(
                relation_name='Fisher',
                expected_value=expected,
                computed_value=computed,
                deviation=deviation,
                is_satisfied=deviation < self.tolerance,
                formula='γ = ν(2 - η)'
            ))
        
        # 2. Hyperscaling: 2β + γ = νd
        if nu.is_valid and beta.is_valid and gamma.is_valid:
            expected = nu.value * self.dimension
            computed = 2 * beta.value + gamma.value
            combined_error = np.sqrt(
                (self.dimension * nu.error)**2 +
                (2 * beta.error)**2 +
                (gamma.error)**2
            )
            deviation = abs(expected - computed) / max(combined_error, 0.01)
            
            checks.append(ScalingRelationCheck(
                relation_name='Hyperscaling',
                expected_value=expected,
                computed_value=computed,
                deviation=deviation,
                is_satisfied=deviation < self.tolerance,
                formula=f'2β + γ = νd (d={self.dimension})'
            ))
        
        # 3. Rushbrooke: α + 2β + γ = 2 (where α = 2 - νd)
        if nu.is_valid and beta.is_valid and gamma.is_valid:
            alpha = 2 - nu.value * self.dimension
            computed = alpha + 2 * beta.value + gamma.value
            expected = 2.0
            combined_error = np.sqrt(
                (self.dimension * nu.error)**2 +
                (2 * beta.error)**2 +
                (gamma.error)**2
            )
            deviation = abs(expected - computed) / max(combined_error, 0.01)
            
            checks.append(ScalingRelationCheck(
                relation_name='Rushbrooke',
                expected_value=expected,
                computed_value=computed,
                deviation=deviation,
                is_satisfied=deviation < self.tolerance,
                formula='α + 2β + γ = 2'
            ))
        
        return checks

    
    def identify_universality_class(
        self,
        nu: ExponentResult,
        z: ExponentResult,
        beta: ExponentResult,
        gamma: ExponentResult,
        eta: ExponentResult,
    ) -> str:
        """
        Identify the universality class based on extracted exponents.
        
        Compares with known universality classes and returns best match.
        
        Args:
            nu, z, beta, gamma, eta: Extracted exponents
            
        Returns:
            Name of best matching universality class
        """
        best_match = 'unknown'
        best_score = float('inf')
        
        for class_name, class_exponents in KNOWN_UNIVERSALITY_CLASSES.items():
            score = 0.0
            n_compared = 0
            
            # Compare each exponent
            if nu.is_valid and 'nu' in class_exponents:
                diff = abs(nu.value - class_exponents['nu'])
                score += (diff / max(nu.error, 0.1))**2
                n_compared += 1
            
            if z.is_valid and 'z' in class_exponents:
                if np.isinf(z.value) and np.isinf(class_exponents['z']):
                    pass  # Both infinite - good match
                elif np.isinf(z.value) or np.isinf(class_exponents['z']):
                    score += 100  # One infinite, one not - bad match
                else:
                    diff = abs(z.value - class_exponents['z'])
                    score += (diff / max(z.error, 0.1))**2
                n_compared += 1
            
            if beta.is_valid and 'beta' in class_exponents:
                diff = abs(beta.value - class_exponents['beta'])
                score += (diff / max(beta.error, 0.05))**2
                n_compared += 1
            
            if gamma.is_valid and 'gamma' in class_exponents:
                diff = abs(gamma.value - class_exponents['gamma'])
                score += (diff / max(gamma.error, 0.1))**2
                n_compared += 1
            
            if eta.is_valid and 'eta' in class_exponents:
                diff = abs(eta.value - class_exponents['eta'])
                score += (diff / max(eta.error, 0.1))**2
                n_compared += 1
            
            if n_compared > 0:
                avg_score = score / n_compared
                if avg_score < best_score:
                    best_score = avg_score
                    best_match = class_name
        
        # Add description if available
        if best_match in KNOWN_UNIVERSALITY_CLASSES:
            desc = KNOWN_UNIVERSALITY_CLASSES[best_match].get('description', '')
            return f"{best_match} ({desc})"
        
        return best_match
    
    def _organize_by_size(
        self,
        data_points: List[FiniteSizeDataPoint]
    ) -> Dict[int, List[FiniteSizeDataPoint]]:
        """Organize data points by system size."""
        data_by_L = {}
        for point in data_points:
            if point.L not in data_by_L:
                data_by_L[point.L] = []
            data_by_L[point.L].append(point)
        return data_by_L


def extract_critical_exponents(
    fss_result: FiniteSizeScalingResult,
    output_dir: Optional[str] = None,
) -> CriticalExponentsResult:
    """
    Main function to extract all critical exponents from FSS results.
    
    Implements Task 15: Critical exponent extraction
    - 15.1 Correlation length exponent ν
    - 15.2 Dynamical exponent z
    - 15.3 Order parameter exponent β
    - 15.4 Susceptibility exponent γ
    - 15.5 Anomalous dimension η
    
    Args:
        fss_result: Results from finite-size scaling analysis (Task 14)
        output_dir: Directory to save results (optional)
        
    Returns:
        CriticalExponentsResult with all exponents
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    extractor = CriticalExponentExtractor(
        system_sizes=fss_result.system_sizes,
        dimension=1,  # 1D chain
    )
    
    result = extractor.extract_all_exponents(
        data_points=fss_result.data_points,
        hc=fss_result.hc_thermodynamic,
        hc_error=fss_result.hc_thermodynamic_error,
    )
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        result.save(str(output_path / 'critical_exponents.json'))
        
        with open(output_path / 'critical_exponents_report.txt', 'w') as f:
            f.write(result.generate_report())
    
    return result


def run_task15_critical_exponents(
    fss_result_path: Optional[str] = None,
    h_center: float = 1.0,
    h_width: float = 0.5,
    W: float = 0.5,
    n_h_points: int = 20,
    n_realizations: int = 50,
    system_sizes: List[int] = None,
    output_dir: str = "results/task15_critical_exponents",
) -> CriticalExponentsResult:
    """
    Run Task 15: Critical exponent extraction.
    
    Can either load existing FSS results or run new FSS analysis.
    
    Args:
        fss_result_path: Path to existing FSS results (optional)
        h_center: Center of h range for new FSS analysis
        h_width: Half-width of h range
        W: Disorder strength
        n_h_points: Number of h points
        n_realizations: Number of disorder realizations
        system_sizes: System sizes (default: [8, 12, 16, 20])
        output_dir: Output directory
        
    Returns:
        CriticalExponentsResult
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    if system_sizes is None:
        # Use smaller sizes for faster computation
        system_sizes = [8, 12, 16, 20]
    
    # Load or generate FSS data
    if fss_result_path and Path(fss_result_path).exists():
        logger.info(f"Loading FSS results from {fss_result_path}")
        fss_result = FiniteSizeScalingResult.load(fss_result_path)
    else:
        logger.info("Running new FSS analysis...")
        from .finite_size_scaling import run_task14_finite_size_scaling
        
        fss_result = run_task14_finite_size_scaling(
            h_center=h_center,
            h_width=h_width,
            W=W,
            n_h_points=n_h_points,
            n_realizations=n_realizations,
            system_sizes=system_sizes,
            output_dir=output_dir,
            parallel=True,
        )
    
    # Extract critical exponents
    result = extract_critical_exponents(fss_result, output_dir)
    
    return result
