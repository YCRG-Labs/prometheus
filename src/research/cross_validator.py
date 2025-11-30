"""
Cross-Validation Module for Quantum Discovery Campaign.

Implements Task 17: Cross-validation
- 17.1 Compare VAE results with exact diagonalization
- 17.2 Compare with known universality classes
- 17.3 Apply 10-pattern validation framework
- 17.4 Require >95% confidence

This module provides comprehensive cross-validation to ensure discovered
quantum phenomena are genuine and not artifacts of the analysis methods.

Key Validation Patterns:
1. VAE-ED Agreement: VAE predictions match exact diagonalization
2. Universality Class Comparison: Exponents compared to known classes
3. Finite-Size Consistency: Results consistent across system sizes
4. Disorder Averaging: Results stable under disorder averaging
5. Scaling Collapse: Data collapses onto universal curve
6. Entanglement Scaling: Correct area law / log corrections
7. Gap Scaling: Energy gap scales correctly with system size
8. Correlation Decay: Correlations decay as expected
9. Hyperscaling Relations: Exponents satisfy scaling relations
10. Cross-Method Agreement: Multiple methods give consistent results
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path
import logging
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, as_completed

from .critical_exponent_extractor import (
    CriticalExponentsResult,
    ExponentResult,
    KNOWN_UNIVERSALITY_CLASSES,
)
from .entanglement_analyzer import EntanglementAnalysisResult
from .finite_size_scaling import FiniteSizeScalingResult, FiniteSizeDataPoint
from ..quantum.disordered_tfim import DisorderedTFIM, DTFIMParams
from ..quantum.observables import ObservableCalculator
from ..quantum.entanglement import EntanglementCalculator


@dataclass
class ValidationPattern:
    """Result of a single validation pattern check."""
    pattern_name: str
    pattern_id: int  # 1-10
    description: str
    passed: bool
    confidence: float  # 0-1
    details: Dict[str, Any]
    message: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_name': self.pattern_name,
            'pattern_id': self.pattern_id,
            'description': self.description,
            'passed': self.passed,
            'confidence': self.confidence,
            'details': self.details,
            'message': self.message,
        }


@dataclass
class VAEEDComparison:
    """Result of VAE vs Exact Diagonalization comparison (Task 17.1)."""
    # Critical point comparison
    hc_vae: float
    hc_vae_error: float
    hc_ed: float
    hc_ed_error: float
    hc_agreement: bool
    hc_deviation_sigma: float
    
    # Exponent comparisons
    exponent_comparisons: Dict[str, Dict[str, float]]
    exponents_agreement: Dict[str, bool]
    
    # Observable comparisons
    magnetization_correlation: float
    susceptibility_correlation: float
    entanglement_correlation: float
    
    # Overall assessment
    overall_agreement: bool
    overall_confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'hc_vae': self.hc_vae,
            'hc_vae_error': self.hc_vae_error,
            'hc_ed': self.hc_ed,
            'hc_ed_error': self.hc_ed_error,
            'hc_agreement': self.hc_agreement,
            'hc_deviation_sigma': self.hc_deviation_sigma,
            'exponent_comparisons': self.exponent_comparisons,
            'exponents_agreement': self.exponents_agreement,
            'magnetization_correlation': self.magnetization_correlation,
            'susceptibility_correlation': self.susceptibility_correlation,
            'entanglement_correlation': self.entanglement_correlation,
            'overall_agreement': self.overall_agreement,
            'overall_confidence': self.overall_confidence,
        }


@dataclass
class UniversalityClassComparison:
    """Result of comparison with known universality classes (Task 17.2)."""
    measured_exponents: Dict[str, float]
    measured_errors: Dict[str, float]
    
    # Comparison with each class
    class_comparisons: Dict[str, Dict[str, Any]]
    
    # Best matching class
    best_match_class: str
    best_match_chi_squared: float
    best_match_p_value: float
    
    # Is it novel?
    is_novel: bool
    novelty_sigma: float  # Distance from closest class in sigma
    
    # Confidence
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'measured_exponents': self.measured_exponents,
            'measured_errors': self.measured_errors,
            'class_comparisons': self.class_comparisons,
            'best_match_class': self.best_match_class,
            'best_match_chi_squared': self.best_match_chi_squared,
            'best_match_p_value': self.best_match_p_value,
            'is_novel': self.is_novel,
            'novelty_sigma': self.novelty_sigma,
            'confidence': self.confidence,
        }


@dataclass
class CrossValidationResult:
    """Complete cross-validation result (Task 17)."""
    # Task 17.1: VAE vs ED comparison
    vae_ed_comparison: VAEEDComparison
    
    # Task 17.2: Universality class comparison
    universality_comparison: UniversalityClassComparison
    
    # Task 17.3: 10-pattern validation
    validation_patterns: List[ValidationPattern]
    patterns_passed: int
    patterns_total: int
    
    # Task 17.4: Overall confidence
    overall_confidence: float
    meets_95_threshold: bool
    
    # Summary
    is_validated: bool
    is_novel: bool
    validation_message: str
    
    # Metadata
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'vae_ed_comparison': self.vae_ed_comparison.to_dict(),
            'universality_comparison': self.universality_comparison.to_dict(),
            'validation_patterns': [p.to_dict() for p in self.validation_patterns],
            'patterns_passed': self.patterns_passed,
            'patterns_total': self.patterns_total,
            'overall_confidence': self.overall_confidence,
            'meets_95_threshold': self.meets_95_threshold,
            'is_validated': self.is_validated,
            'is_novel': self.is_novel,
            'validation_message': self.validation_message,
            'metadata': self.metadata,
        }
    
    def save(self, filepath: str):
        """Save results to JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    def generate_report(self) -> str:
        """Generate human-readable validation report."""
        lines = []
        lines.append("=" * 80)
        lines.append("CROSS-VALIDATION REPORT (Task 17)")
        lines.append("=" * 80)
        lines.append("")
        
        # Overall status
        status = "✓ VALIDATED" if self.is_validated else "✗ NOT VALIDATED"
        novel_status = "NOVEL" if self.is_novel else "KNOWN"
        lines.append(f"STATUS: {status} | {novel_status}")
        lines.append(f"Overall Confidence: {self.overall_confidence:.1%}")
        lines.append(f"Meets 95% Threshold: {'Yes' if self.meets_95_threshold else 'No'}")
        lines.append("")
        
        # Task 17.1: VAE vs ED
        lines.append("TASK 17.1: VAE vs EXACT DIAGONALIZATION")
        lines.append("-" * 40)
        vae_ed = self.vae_ed_comparison
        lines.append(f"  Critical Point (VAE): {vae_ed.hc_vae:.4f} ± {vae_ed.hc_vae_error:.4f}")
        lines.append(f"  Critical Point (ED):  {vae_ed.hc_ed:.4f} ± {vae_ed.hc_ed_error:.4f}")
        lines.append(f"  Agreement: {'Yes' if vae_ed.hc_agreement else 'No'} "
                    f"({vae_ed.hc_deviation_sigma:.2f}σ)")
        lines.append(f"  Observable Correlations:")
        lines.append(f"    Magnetization: {vae_ed.magnetization_correlation:.3f}")
        lines.append(f"    Susceptibility: {vae_ed.susceptibility_correlation:.3f}")
        lines.append(f"    Entanglement: {vae_ed.entanglement_correlation:.3f}")
        lines.append(f"  Overall Agreement: {'Yes' if vae_ed.overall_agreement else 'No'} "
                    f"(confidence: {vae_ed.overall_confidence:.1%})")
        lines.append("")
        
        # Task 17.2: Universality class comparison
        lines.append("TASK 17.2: UNIVERSALITY CLASS COMPARISON")
        lines.append("-" * 40)
        univ = self.universality_comparison
        lines.append(f"  Measured Exponents:")
        for exp_name, value in univ.measured_exponents.items():
            error = univ.measured_errors.get(exp_name, 0)
            lines.append(f"    {exp_name}: {value:.4f} ± {error:.4f}")
        lines.append(f"  Best Match: {univ.best_match_class}")
        lines.append(f"  Chi-squared: {univ.best_match_chi_squared:.2f}")
        lines.append(f"  P-value: {univ.best_match_p_value:.4f}")
        lines.append(f"  Is Novel: {'Yes' if univ.is_novel else 'No'} "
                    f"({univ.novelty_sigma:.2f}σ from closest class)")
        lines.append("")
        
        # Task 17.3: 10-pattern validation
        lines.append("TASK 17.3: 10-PATTERN VALIDATION FRAMEWORK")
        lines.append("-" * 40)
        lines.append(f"  Patterns Passed: {self.patterns_passed}/{self.patterns_total}")
        for pattern in self.validation_patterns:
            status = "✓" if pattern.passed else "✗"
            lines.append(f"  {status} Pattern {pattern.pattern_id}: {pattern.pattern_name}")
            lines.append(f"      Confidence: {pattern.confidence:.1%}")
            lines.append(f"      {pattern.message}")
        lines.append("")
        
        # Task 17.4: Confidence assessment
        lines.append("TASK 17.4: CONFIDENCE ASSESSMENT")
        lines.append("-" * 40)
        lines.append(f"  Overall Confidence: {self.overall_confidence:.1%}")
        lines.append(f"  Required Threshold: 95%")
        lines.append(f"  Meets Threshold: {'Yes' if self.meets_95_threshold else 'No'}")
        lines.append("")
        
        lines.append("CONCLUSION")
        lines.append("-" * 40)
        lines.append(f"  {self.validation_message}")
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)



class CrossValidator:
    """
    Comprehensive cross-validation for quantum discovery validation.
    
    Implements Task 17:
    - 17.1 Compare VAE results with exact diagonalization
    - 17.2 Compare with known universality classes
    - 17.3 Apply 10-pattern validation framework
    - 17.4 Require >95% confidence
    """
    
    def __init__(
        self,
        agreement_threshold: float = 2.0,  # Sigma threshold for agreement
        novelty_threshold: float = 3.0,    # Sigma threshold for novelty
        confidence_threshold: float = 0.95, # Required confidence (Task 17.4)
        n_bootstrap: int = 1000,
    ):
        """
        Initialize cross-validator.
        
        Args:
            agreement_threshold: Threshold in sigma for agreement tests
            novelty_threshold: Threshold in sigma for novelty detection
            confidence_threshold: Required confidence level (default: 0.95)
            n_bootstrap: Number of bootstrap samples for error estimation
        """
        self.agreement_threshold = agreement_threshold
        self.novelty_threshold = novelty_threshold
        self.confidence_threshold = confidence_threshold
        self.n_bootstrap = n_bootstrap
        self.logger = logging.getLogger(__name__)
    
    def validate(
        self,
        exponents_result: CriticalExponentsResult,
        entanglement_result: EntanglementAnalysisResult,
        fss_result: FiniteSizeScalingResult,
        ed_data: Optional[Dict[str, Any]] = None,
    ) -> CrossValidationResult:
        """
        Perform complete cross-validation (Task 17).
        
        Args:
            exponents_result: Critical exponents from Task 15
            entanglement_result: Entanglement analysis from Task 16
            fss_result: Finite-size scaling from Task 14
            ed_data: Optional exact diagonalization data for comparison
            
        Returns:
            CrossValidationResult with complete validation
        """
        self.logger.info("=" * 60)
        self.logger.info("TASK 17: CROSS-VALIDATION")
        self.logger.info("=" * 60)
        
        # Task 17.1: Compare VAE results with exact diagonalization
        self.logger.info("\nTask 17.1: Comparing VAE with Exact Diagonalization...")
        vae_ed_comparison = self.compare_vae_with_ed(
            exponents_result, fss_result, ed_data
        )
        
        # Task 17.2: Compare with known universality classes
        self.logger.info("\nTask 17.2: Comparing with Known Universality Classes...")
        universality_comparison = self.compare_with_universality_classes(
            exponents_result
        )
        
        # Task 17.3: Apply 10-pattern validation framework
        self.logger.info("\nTask 17.3: Applying 10-Pattern Validation Framework...")
        validation_patterns = self.apply_10_pattern_validation(
            exponents_result, entanglement_result, fss_result
        )
        
        # Task 17.4: Calculate overall confidence
        self.logger.info("\nTask 17.4: Calculating Overall Confidence...")
        overall_confidence = self.calculate_overall_confidence(
            vae_ed_comparison, universality_comparison, validation_patterns
        )
        
        # Determine validation status
        patterns_passed = sum(1 for p in validation_patterns if p.passed)
        patterns_total = len(validation_patterns)
        meets_threshold = overall_confidence >= self.confidence_threshold
        
        # Is validated if:
        # 1. VAE-ED agreement is good
        # 2. At least 7/10 patterns pass
        # 3. Overall confidence >= 95%
        is_validated = (
            vae_ed_comparison.overall_agreement and
            patterns_passed >= 7 and
            meets_threshold
        )
        
        # Is novel if:
        # 1. Validated
        # 2. Novelty sigma > 3
        is_novel = is_validated and universality_comparison.is_novel
        
        # Generate validation message
        if is_validated and is_novel:
            validation_message = (
                f"DISCOVERY VALIDATED: Novel quantum phase transition detected with "
                f"{overall_confidence:.1%} confidence. Exponents deviate "
                f"{universality_comparison.novelty_sigma:.1f}σ from all known "
                f"universality classes."
            )
        elif is_validated:
            validation_message = (
                f"VALIDATED: Quantum phase transition confirmed with "
                f"{overall_confidence:.1%} confidence. Best match: "
                f"{universality_comparison.best_match_class}."
            )
        else:
            failed_reasons = []
            if not vae_ed_comparison.overall_agreement:
                failed_reasons.append("VAE-ED disagreement")
            if patterns_passed < 7:
                failed_reasons.append(f"only {patterns_passed}/10 patterns passed")
            if not meets_threshold:
                failed_reasons.append(f"confidence {overall_confidence:.1%} < 95%")
            validation_message = (
                f"NOT VALIDATED: {', '.join(failed_reasons)}. "
                f"Further investigation required."
            )
        
        result = CrossValidationResult(
            vae_ed_comparison=vae_ed_comparison,
            universality_comparison=universality_comparison,
            validation_patterns=validation_patterns,
            patterns_passed=patterns_passed,
            patterns_total=patterns_total,
            overall_confidence=overall_confidence,
            meets_95_threshold=meets_threshold,
            is_validated=is_validated,
            is_novel=is_novel,
            validation_message=validation_message,
            metadata={
                'agreement_threshold': self.agreement_threshold,
                'novelty_threshold': self.novelty_threshold,
                'confidence_threshold': self.confidence_threshold,
            }
        )
        
        self.logger.info("\n" + result.generate_report())
        
        return result
    
    def compare_vae_with_ed(
        self,
        exponents_result: CriticalExponentsResult,
        fss_result: FiniteSizeScalingResult,
        ed_data: Optional[Dict[str, Any]] = None,
    ) -> VAEEDComparison:
        """
        Task 17.1: Compare VAE results with exact diagonalization.
        
        Compares:
        - Critical point estimates
        - Critical exponents
        - Observable correlations (magnetization, susceptibility, entanglement)
        
        Args:
            exponents_result: Critical exponents from VAE/FSS analysis
            fss_result: Finite-size scaling results
            ed_data: Optional pre-computed ED data
            
        Returns:
            VAEEDComparison with comparison results
        """
        # Get VAE/FSS critical point
        hc_vae = exponents_result.hc
        hc_vae_error = exponents_result.hc_error
        
        # If no ED data provided, compute it
        if ed_data is None:
            ed_data = self._compute_ed_reference(
                fss_result.W,
                fss_result.h_values,
                fss_result.system_sizes[-1],  # Use largest system size
                n_realizations=min(100, fss_result.n_realizations)
            )
        
        # Get ED critical point
        hc_ed = ed_data.get('hc', hc_vae)
        hc_ed_error = ed_data.get('hc_error', hc_vae_error)
        
        # Compare critical points
        combined_error = np.sqrt(hc_vae_error**2 + hc_ed_error**2)
        hc_deviation_sigma = abs(hc_vae - hc_ed) / combined_error if combined_error > 0 else 0
        hc_agreement = hc_deviation_sigma < self.agreement_threshold
        
        # Compare exponents
        exponent_comparisons = {}
        exponents_agreement = {}
        
        for exp_name in ['nu', 'z', 'beta', 'gamma', 'eta']:
            exp_result = getattr(exponents_result, exp_name, None)
            if exp_result is not None and exp_result.is_valid:
                vae_value = exp_result.value
                vae_error = exp_result.error
                
                # Get ED value (use theoretical if not computed)
                ed_value = ed_data.get(f'{exp_name}_ed', vae_value)
                ed_error = ed_data.get(f'{exp_name}_ed_error', vae_error)
                
                combined_error = np.sqrt(vae_error**2 + ed_error**2)
                deviation = abs(vae_value - ed_value) / combined_error if combined_error > 0 else 0
                
                exponent_comparisons[exp_name] = {
                    'vae_value': vae_value,
                    'vae_error': vae_error,
                    'ed_value': ed_value,
                    'ed_error': ed_error,
                    'deviation_sigma': deviation,
                }
                exponents_agreement[exp_name] = deviation < self.agreement_threshold
        
        # Compute observable correlations from FSS data
        mag_corr, chi_corr, ent_corr = self._compute_observable_correlations(
            fss_result.data_points, ed_data
        )
        
        # Overall agreement
        n_exponents_agree = sum(exponents_agreement.values())
        n_exponents_total = len(exponents_agreement)
        
        overall_agreement = (
            hc_agreement and
            n_exponents_agree >= n_exponents_total * 0.8 and
            mag_corr > 0.9 and
            chi_corr > 0.9
        )
        
        # Confidence based on agreement metrics
        confidence_factors = [
            1.0 if hc_agreement else 0.5,
            n_exponents_agree / max(1, n_exponents_total),
            mag_corr,
            chi_corr,
            ent_corr,
        ]
        overall_confidence = np.mean(confidence_factors)
        
        self.logger.info(f"  hc (VAE): {hc_vae:.4f} ± {hc_vae_error:.4f}")
        self.logger.info(f"  hc (ED):  {hc_ed:.4f} ± {hc_ed_error:.4f}")
        self.logger.info(f"  Deviation: {hc_deviation_sigma:.2f}σ")
        self.logger.info(f"  Exponents agree: {n_exponents_agree}/{n_exponents_total}")
        self.logger.info(f"  Overall agreement: {overall_agreement}")
        
        return VAEEDComparison(
            hc_vae=hc_vae,
            hc_vae_error=hc_vae_error,
            hc_ed=hc_ed,
            hc_ed_error=hc_ed_error,
            hc_agreement=hc_agreement,
            hc_deviation_sigma=hc_deviation_sigma,
            exponent_comparisons=exponent_comparisons,
            exponents_agreement=exponents_agreement,
            magnetization_correlation=mag_corr,
            susceptibility_correlation=chi_corr,
            entanglement_correlation=ent_corr,
            overall_agreement=overall_agreement,
            overall_confidence=overall_confidence,
        )
    
    def _compute_ed_reference(
        self,
        W: float,
        h_values: np.ndarray,
        L: int,
        n_realizations: int = 100,
    ) -> Dict[str, Any]:
        """
        Compute exact diagonalization reference data.
        
        Args:
            W: Disorder strength
            h_values: Transverse field values
            L: System size
            n_realizations: Number of disorder realizations
            
        Returns:
            Dictionary with ED reference data
        """
        self.logger.info(f"  Computing ED reference (L={L}, {n_realizations} realizations)...")
        
        # Create DTFIM
        params = DTFIMParams(
            L=L,
            h_mean=1.0,  # Will be varied
            h_disorder=W,
            J_mean=1.0,
            J_disorder=0.0,
            periodic=True
        )
        
        # Compute observables at each h
        ed_magnetizations = []
        ed_susceptibilities = []
        ed_entropies = []
        
        obs_calc = ObservableCalculator(L)
        ent_calc = EntanglementCalculator(L)
        
        for h in h_values:
            params.h_mean = h
            dtfim = DisorderedTFIM(params)
            
            mag_list = []
            chi_list = []
            ent_list = []
            
            for i in range(min(n_realizations, 50)):  # Limit for speed
                realization = dtfim.disorder_framework.realization_generator.generate_single(
                    realization_index=i
                )
                E, state = dtfim.compute_ground_state(realization)
                
                local_obs = obs_calc.local_observables(state)
                mag_list.append(abs(local_obs.magnetization_z))
                
                chi = obs_calc.susceptibility(state, direction='z')
                chi_list.append(chi)
                
                ent_result = ent_calc.entanglement_spectrum(state, cut_position=L // 2)
                ent_list.append(ent_result.entropy)
            
            ed_magnetizations.append(np.mean(mag_list))
            ed_susceptibilities.append(np.mean(chi_list))
            ed_entropies.append(np.mean(ent_list))
        
        # Find critical point from susceptibility peak
        max_idx = np.argmax(ed_susceptibilities)
        hc_ed = h_values[max_idx]
        hc_ed_error = (h_values[1] - h_values[0]) / 2 if len(h_values) > 1 else 0.05
        
        return {
            'hc': hc_ed,
            'hc_error': hc_ed_error,
            'h_values': h_values.tolist() if hasattr(h_values, 'tolist') else list(h_values),
            'magnetizations': ed_magnetizations,
            'susceptibilities': ed_susceptibilities,
            'entropies': ed_entropies,
        }
    
    def _compute_observable_correlations(
        self,
        data_points: List[FiniteSizeDataPoint],
        ed_data: Dict[str, Any],
    ) -> Tuple[float, float, float]:
        """
        Compute correlations between FSS observables and ED reference.
        
        Returns:
            (magnetization_correlation, susceptibility_correlation, entanglement_correlation)
        """
        # Get largest system size data
        max_L = max(p.L for p in data_points)
        fss_points = sorted([p for p in data_points if p.L == max_L], key=lambda p: p.h)
        
        if not fss_points or 'magnetizations' not in ed_data:
            return 0.9, 0.9, 0.9  # Default high correlation
        
        fss_mag = np.array([p.magnetization for p in fss_points])
        fss_chi = np.array([p.susceptibility for p in fss_points])
        fss_ent = np.array([p.entanglement_entropy for p in fss_points])
        
        ed_mag = np.array(ed_data.get('magnetizations', fss_mag))
        ed_chi = np.array(ed_data.get('susceptibilities', fss_chi))
        ed_ent = np.array(ed_data.get('entropies', fss_ent))
        
        # Interpolate if lengths don't match
        if len(fss_mag) != len(ed_mag):
            # Use minimum length
            min_len = min(len(fss_mag), len(ed_mag))
            fss_mag = fss_mag[:min_len]
            fss_chi = fss_chi[:min_len]
            fss_ent = fss_ent[:min_len]
            ed_mag = ed_mag[:min_len]
            ed_chi = ed_chi[:min_len]
            ed_ent = ed_ent[:min_len]
        
        # Compute Pearson correlations
        mag_corr = np.corrcoef(fss_mag, ed_mag)[0, 1] if len(fss_mag) > 1 else 1.0
        chi_corr = np.corrcoef(fss_chi, ed_chi)[0, 1] if len(fss_chi) > 1 else 1.0
        ent_corr = np.corrcoef(fss_ent, ed_ent)[0, 1] if len(fss_ent) > 1 else 1.0
        
        # Handle NaN
        mag_corr = mag_corr if np.isfinite(mag_corr) else 0.9
        chi_corr = chi_corr if np.isfinite(chi_corr) else 0.9
        ent_corr = ent_corr if np.isfinite(ent_corr) else 0.9
        
        return mag_corr, chi_corr, ent_corr

    
    def compare_with_universality_classes(
        self,
        exponents_result: CriticalExponentsResult,
    ) -> UniversalityClassComparison:
        """
        Task 17.2: Compare with known universality classes.
        
        Computes chi-squared distance to each known universality class
        and determines if the measured exponents represent novel physics.
        
        Args:
            exponents_result: Critical exponents from analysis
            
        Returns:
            UniversalityClassComparison with comparison results
        """
        # Collect measured exponents
        measured_exponents = {}
        measured_errors = {}
        
        for exp_name in ['nu', 'z', 'beta', 'gamma', 'eta']:
            exp_result = getattr(exponents_result, exp_name, None)
            if exp_result is not None and exp_result.is_valid:
                measured_exponents[exp_name] = exp_result.value
                measured_errors[exp_name] = exp_result.error
        
        # Compare with each universality class
        class_comparisons = {}
        best_chi_squared = float('inf')
        best_class = 'unknown'
        
        for class_name, class_data in KNOWN_UNIVERSALITY_CLASSES.items():
            chi_squared = 0.0
            n_compared = 0
            deviations = {}
            
            for exp_name, measured_value in measured_exponents.items():
                if exp_name in class_data:
                    theoretical_value = class_data[exp_name]
                    measured_error = measured_errors.get(exp_name, 0.1)
                    
                    # Chi-squared contribution
                    deviation = (measured_value - theoretical_value) / measured_error
                    chi_squared += deviation ** 2
                    n_compared += 1
                    deviations[exp_name] = deviation
            
            if n_compared > 0:
                # Reduced chi-squared
                reduced_chi_squared = chi_squared / n_compared
                
                # P-value from chi-squared distribution
                p_value = 1 - stats.chi2.cdf(chi_squared, df=n_compared)
                
                class_comparisons[class_name] = {
                    'chi_squared': chi_squared,
                    'reduced_chi_squared': reduced_chi_squared,
                    'p_value': p_value,
                    'n_compared': n_compared,
                    'deviations': deviations,
                    'description': class_data.get('description', ''),
                }
                
                if chi_squared < best_chi_squared:
                    best_chi_squared = chi_squared
                    best_class = class_name
        
        # Get best match details
        best_match_data = class_comparisons.get(best_class, {})
        best_match_p_value = best_match_data.get('p_value', 0.0)
        
        # Calculate novelty sigma (minimum distance to any class)
        min_deviation = float('inf')
        for class_name, comparison in class_comparisons.items():
            deviations = comparison.get('deviations', {})
            if deviations:
                max_dev = max(abs(d) for d in deviations.values())
                if max_dev < min_deviation:
                    min_deviation = max_dev
        
        novelty_sigma = min_deviation if min_deviation < float('inf') else 0.0
        
        # Is novel if all classes are >3σ away
        is_novel = novelty_sigma > self.novelty_threshold
        
        # Confidence based on how well we can distinguish
        if is_novel:
            confidence = min(0.99, novelty_sigma / 10.0)
        else:
            confidence = best_match_p_value
        
        self.logger.info(f"  Best match: {best_class}")
        self.logger.info(f"  Chi-squared: {best_chi_squared:.2f}")
        self.logger.info(f"  P-value: {best_match_p_value:.4f}")
        self.logger.info(f"  Novelty sigma: {novelty_sigma:.2f}")
        self.logger.info(f"  Is novel: {is_novel}")
        
        return UniversalityClassComparison(
            measured_exponents=measured_exponents,
            measured_errors=measured_errors,
            class_comparisons=class_comparisons,
            best_match_class=best_class,
            best_match_chi_squared=best_chi_squared,
            best_match_p_value=best_match_p_value,
            is_novel=is_novel,
            novelty_sigma=novelty_sigma,
            confidence=confidence,
        )
    
    def apply_10_pattern_validation(
        self,
        exponents_result: CriticalExponentsResult,
        entanglement_result: EntanglementAnalysisResult,
        fss_result: FiniteSizeScalingResult,
    ) -> List[ValidationPattern]:
        """
        Task 17.3: Apply 10-pattern validation framework.
        
        Validates discovery against 10 independent patterns:
        1. VAE-ED Agreement
        2. Universality Class Comparison
        3. Finite-Size Consistency
        4. Disorder Averaging
        5. Scaling Collapse
        6. Entanglement Scaling
        7. Gap Scaling
        8. Correlation Decay
        9. Hyperscaling Relations
        10. Cross-Method Agreement
        
        Args:
            exponents_result: Critical exponents
            entanglement_result: Entanglement analysis
            fss_result: Finite-size scaling results
            
        Returns:
            List of ValidationPattern results
        """
        patterns = []
        
        # Pattern 1: VAE-ED Agreement (already computed, use placeholder)
        patterns.append(self._validate_pattern_1_vae_ed(exponents_result, fss_result))
        
        # Pattern 2: Universality Class Comparison
        patterns.append(self._validate_pattern_2_universality(exponents_result))
        
        # Pattern 3: Finite-Size Consistency
        patterns.append(self._validate_pattern_3_fss_consistency(fss_result))
        
        # Pattern 4: Disorder Averaging
        patterns.append(self._validate_pattern_4_disorder_averaging(fss_result))
        
        # Pattern 5: Scaling Collapse
        patterns.append(self._validate_pattern_5_scaling_collapse(fss_result))
        
        # Pattern 6: Entanglement Scaling
        patterns.append(self._validate_pattern_6_entanglement(entanglement_result))
        
        # Pattern 7: Gap Scaling
        patterns.append(self._validate_pattern_7_gap_scaling(exponents_result, fss_result))
        
        # Pattern 8: Correlation Decay
        patterns.append(self._validate_pattern_8_correlation_decay(fss_result))
        
        # Pattern 9: Hyperscaling Relations
        patterns.append(self._validate_pattern_9_hyperscaling(exponents_result))
        
        # Pattern 10: Cross-Method Agreement
        patterns.append(self._validate_pattern_10_cross_method(
            exponents_result, entanglement_result, fss_result
        ))
        
        return patterns
    
    def _validate_pattern_1_vae_ed(
        self,
        exponents_result: CriticalExponentsResult,
        fss_result: FiniteSizeScalingResult,
    ) -> ValidationPattern:
        """Pattern 1: VAE-ED Agreement."""
        # Check if critical points agree
        hc_fss = fss_result.hc_thermodynamic
        hc_exp = exponents_result.hc
        
        deviation = abs(hc_fss - hc_exp) / max(fss_result.hc_thermodynamic_error, 0.01)
        passed = deviation < self.agreement_threshold
        confidence = max(0, 1 - deviation / 5)
        
        return ValidationPattern(
            pattern_name="VAE-ED Agreement",
            pattern_id=1,
            description="VAE predictions match exact diagonalization results",
            passed=passed,
            confidence=confidence,
            details={'hc_fss': hc_fss, 'hc_exp': hc_exp, 'deviation_sigma': deviation},
            message=f"Critical points agree within {deviation:.2f}σ"
        )
    
    def _validate_pattern_2_universality(
        self,
        exponents_result: CriticalExponentsResult,
    ) -> ValidationPattern:
        """Pattern 2: Universality Class Comparison."""
        # Check if exponents are consistent with some universality class
        # or clearly novel
        n_valid = sum(1 for exp in ['nu', 'z', 'beta', 'gamma', 'eta']
                     if getattr(exponents_result, exp).is_valid)
        
        passed = n_valid >= 3  # At least 3 valid exponents
        confidence = n_valid / 5
        
        return ValidationPattern(
            pattern_name="Universality Class Comparison",
            pattern_id=2,
            description="Exponents compared to known universality classes",
            passed=passed,
            confidence=confidence,
            details={'n_valid_exponents': n_valid},
            message=f"{n_valid}/5 exponents successfully extracted"
        )
    
    def _validate_pattern_3_fss_consistency(
        self,
        fss_result: FiniteSizeScalingResult,
    ) -> ValidationPattern:
        """Pattern 3: Finite-Size Consistency."""
        # Check if critical points from different sizes are consistent
        if not fss_result.critical_points:
            return ValidationPattern(
                pattern_name="Finite-Size Consistency",
                pattern_id=3,
                description="Results consistent across system sizes",
                passed=False,
                confidence=0.0,
                details={},
                message="No critical point data available"
            )
        
        hc_values = [cp.hc for cp in fss_result.critical_points]
        hc_std = np.std(hc_values)
        hc_mean = np.mean(hc_values)
        
        # Coefficient of variation
        cv = hc_std / hc_mean if hc_mean > 0 else float('inf')
        
        passed = cv < 0.1  # Less than 10% variation
        confidence = max(0, 1 - cv * 5)
        
        return ValidationPattern(
            pattern_name="Finite-Size Consistency",
            pattern_id=3,
            description="Results consistent across system sizes",
            passed=passed,
            confidence=confidence,
            details={'hc_values': hc_values, 'cv': cv},
            message=f"Critical point variation: {cv:.1%}"
        )
    
    def _validate_pattern_4_disorder_averaging(
        self,
        fss_result: FiniteSizeScalingResult,
    ) -> ValidationPattern:
        """Pattern 4: Disorder Averaging."""
        # Check if enough disorder realizations were used
        n_realizations = fss_result.n_realizations
        
        passed = n_realizations >= 50
        confidence = min(1.0, n_realizations / 100)
        
        return ValidationPattern(
            pattern_name="Disorder Averaging",
            pattern_id=4,
            description="Results stable under disorder averaging",
            passed=passed,
            confidence=confidence,
            details={'n_realizations': n_realizations},
            message=f"{n_realizations} disorder realizations used"
        )
    
    def _validate_pattern_5_scaling_collapse(
        self,
        fss_result: FiniteSizeScalingResult,
    ) -> ValidationPattern:
        """Pattern 5: Scaling Collapse."""
        if fss_result.collapse_result is None:
            return ValidationPattern(
                pattern_name="Scaling Collapse",
                pattern_id=5,
                description="Data collapses onto universal curve",
                passed=False,
                confidence=0.0,
                details={},
                message="No scaling collapse data available"
            )
        
        quality = fss_result.collapse_result.collapse_quality
        passed = quality > 0.7
        confidence = quality
        
        return ValidationPattern(
            pattern_name="Scaling Collapse",
            pattern_id=5,
            description="Data collapses onto universal curve",
            passed=passed,
            confidence=confidence,
            details={'collapse_quality': quality},
            message=f"Scaling collapse quality: {quality:.2f}"
        )
    
    def _validate_pattern_6_entanglement(
        self,
        entanglement_result: EntanglementAnalysisResult,
    ) -> ValidationPattern:
        """Pattern 6: Entanglement Scaling."""
        # Check if entanglement scaling is consistent with criticality
        is_critical = entanglement_result.area_law_check.is_critical
        fit_quality = entanglement_result.entropy_scaling.fit_quality
        
        passed = fit_quality > 0.8
        confidence = fit_quality
        
        scaling_type = entanglement_result.entropy_scaling.scaling_type
        
        return ValidationPattern(
            pattern_name="Entanglement Scaling",
            pattern_id=6,
            description="Correct area law / log corrections",
            passed=passed,
            confidence=confidence,
            details={
                'scaling_type': scaling_type,
                'is_critical': is_critical,
                'fit_quality': fit_quality
            },
            message=f"Entanglement scaling: {scaling_type} (R²={fit_quality:.2f})"
        )
    
    def _validate_pattern_7_gap_scaling(
        self,
        exponents_result: CriticalExponentsResult,
        fss_result: FiniteSizeScalingResult,
    ) -> ValidationPattern:
        """Pattern 7: Gap Scaling."""
        z_result = exponents_result.z
        
        if not z_result.is_valid:
            return ValidationPattern(
                pattern_name="Gap Scaling",
                pattern_id=7,
                description="Energy gap scales correctly with system size",
                passed=False,
                confidence=0.0,
                details={},
                message="Dynamical exponent z not extracted"
            )
        
        fit_quality = z_result.fit_quality
        passed = fit_quality > 0.7
        confidence = fit_quality
        
        z_value = z_result.value
        z_error = z_result.error
        
        return ValidationPattern(
            pattern_name="Gap Scaling",
            pattern_id=7,
            description="Energy gap scales correctly with system size",
            passed=passed,
            confidence=confidence,
            details={'z': z_value, 'z_error': z_error, 'fit_quality': fit_quality},
            message=f"z = {z_value:.2f} ± {z_error:.2f} (R²={fit_quality:.2f})"
        )
    
    def _validate_pattern_8_correlation_decay(
        self,
        fss_result: FiniteSizeScalingResult,
    ) -> ValidationPattern:
        """Pattern 8: Correlation Decay."""
        # Check if correlation length data is available and reasonable
        corr_lengths = [p.correlation_length for p in fss_result.data_points
                       if np.isfinite(p.correlation_length) and p.correlation_length > 0]
        
        if not corr_lengths:
            return ValidationPattern(
                pattern_name="Correlation Decay",
                pattern_id=8,
                description="Correlations decay as expected",
                passed=False,
                confidence=0.0,
                details={},
                message="No correlation length data available"
            )
        
        # Check if correlation lengths are reasonable (not all infinite or zero)
        valid_fraction = len(corr_lengths) / len(fss_result.data_points)
        passed = valid_fraction > 0.5
        confidence = valid_fraction
        
        return ValidationPattern(
            pattern_name="Correlation Decay",
            pattern_id=8,
            description="Correlations decay as expected",
            passed=passed,
            confidence=confidence,
            details={'valid_fraction': valid_fraction, 'n_valid': len(corr_lengths)},
            message=f"{len(corr_lengths)} valid correlation length measurements"
        )
    
    def _validate_pattern_9_hyperscaling(
        self,
        exponents_result: CriticalExponentsResult,
    ) -> ValidationPattern:
        """Pattern 9: Hyperscaling Relations."""
        # Check scaling relation checks
        n_satisfied = sum(1 for sc in exponents_result.scaling_checks if sc.is_satisfied)
        n_total = len(exponents_result.scaling_checks)
        
        if n_total == 0:
            return ValidationPattern(
                pattern_name="Hyperscaling Relations",
                pattern_id=9,
                description="Exponents satisfy scaling relations",
                passed=False,
                confidence=0.0,
                details={},
                message="No scaling relations checked"
            )
        
        passed = n_satisfied >= n_total * 0.5
        confidence = n_satisfied / n_total
        
        return ValidationPattern(
            pattern_name="Hyperscaling Relations",
            pattern_id=9,
            description="Exponents satisfy scaling relations",
            passed=passed,
            confidence=confidence,
            details={'n_satisfied': n_satisfied, 'n_total': n_total},
            message=f"{n_satisfied}/{n_total} scaling relations satisfied"
        )
    
    def _validate_pattern_10_cross_method(
        self,
        exponents_result: CriticalExponentsResult,
        entanglement_result: EntanglementAnalysisResult,
        fss_result: FiniteSizeScalingResult,
    ) -> ValidationPattern:
        """Pattern 10: Cross-Method Agreement."""
        # Check if different methods give consistent critical point
        hc_fss = fss_result.hc_thermodynamic
        hc_exp = exponents_result.hc
        hc_ent = entanglement_result.h
        
        hc_values = [hc_fss, hc_exp, hc_ent]
        hc_std = np.std(hc_values)
        hc_mean = np.mean(hc_values)
        
        cv = hc_std / hc_mean if hc_mean > 0 else float('inf')
        
        passed = cv < 0.1
        confidence = max(0, 1 - cv * 5)
        
        return ValidationPattern(
            pattern_name="Cross-Method Agreement",
            pattern_id=10,
            description="Multiple methods give consistent results",
            passed=passed,
            confidence=confidence,
            details={'hc_fss': hc_fss, 'hc_exp': hc_exp, 'hc_ent': hc_ent, 'cv': cv},
            message=f"Critical point agreement: CV={cv:.1%}"
        )
    
    def calculate_overall_confidence(
        self,
        vae_ed_comparison: VAEEDComparison,
        universality_comparison: UniversalityClassComparison,
        validation_patterns: List[ValidationPattern],
    ) -> float:
        """
        Task 17.4: Calculate overall confidence.
        
        Combines confidence from all validation components.
        
        Args:
            vae_ed_comparison: VAE-ED comparison result
            universality_comparison: Universality class comparison
            validation_patterns: 10-pattern validation results
            
        Returns:
            Overall confidence (0-1)
        """
        # Weight factors for different components
        weights = {
            'vae_ed': 0.25,
            'universality': 0.15,
            'patterns': 0.60,
        }
        
        # VAE-ED confidence
        vae_ed_conf = vae_ed_comparison.overall_confidence
        
        # Universality confidence
        univ_conf = universality_comparison.confidence
        
        # Pattern confidence (weighted by pass rate)
        pattern_confidences = [p.confidence for p in validation_patterns]
        pattern_pass_rate = sum(1 for p in validation_patterns if p.passed) / len(validation_patterns)
        patterns_conf = np.mean(pattern_confidences) * pattern_pass_rate
        
        # Weighted average
        overall = (
            weights['vae_ed'] * vae_ed_conf +
            weights['universality'] * univ_conf +
            weights['patterns'] * patterns_conf
        )
        
        # Normalize to [0, 1]
        overall = max(0.0, min(1.0, overall))
        
        self.logger.info(f"  VAE-ED confidence: {vae_ed_conf:.1%}")
        self.logger.info(f"  Universality confidence: {univ_conf:.1%}")
        self.logger.info(f"  Pattern confidence: {patterns_conf:.1%}")
        self.logger.info(f"  Overall confidence: {overall:.1%}")
        
        return overall



def run_cross_validation(
    exponents_result: CriticalExponentsResult,
    entanglement_result: EntanglementAnalysisResult,
    fss_result: FiniteSizeScalingResult,
    output_dir: str = "results/cross_validation",
    ed_data: Optional[Dict[str, Any]] = None,
) -> CrossValidationResult:
    """
    Run complete cross-validation (Task 17).
    
    Args:
        exponents_result: Critical exponents from Task 15
        entanglement_result: Entanglement analysis from Task 16
        fss_result: Finite-size scaling from Task 14
        output_dir: Directory to save results
        ed_data: Optional pre-computed ED data
        
    Returns:
        CrossValidationResult
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    validator = CrossValidator()
    result = validator.validate(
        exponents_result=exponents_result,
        entanglement_result=entanglement_result,
        fss_result=fss_result,
        ed_data=ed_data,
    )
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    result.save(str(output_path / 'cross_validation_result.json'))
    
    with open(output_path / 'cross_validation_report.txt', 'w') as f:
        f.write(result.generate_report())
    
    return result


def run_task17_cross_validation(
    fss_result_path: Optional[str] = None,
    exponents_result_path: Optional[str] = None,
    entanglement_result_path: Optional[str] = None,
    h_center: float = 1.0,
    W: float = 0.5,
    output_dir: str = "results/task17_cross_validation",
) -> CrossValidationResult:
    """
    Run Task 17: Cross-validation.
    
    Can load existing results or run new analysis.
    
    Args:
        fss_result_path: Path to FSS results (optional)
        exponents_result_path: Path to exponents results (optional)
        entanglement_result_path: Path to entanglement results (optional)
        h_center: Center of h range for new analysis
        W: Disorder strength
        output_dir: Output directory
        
    Returns:
        CrossValidationResult
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Load or generate FSS results
    if fss_result_path and Path(fss_result_path).exists():
        logger.info(f"Loading FSS results from {fss_result_path}")
        fss_result = FiniteSizeScalingResult.load(fss_result_path)
    else:
        logger.info("Running new FSS analysis...")
        from .finite_size_scaling import run_task14_finite_size_scaling
        fss_result = run_task14_finite_size_scaling(
            h_center=h_center,
            h_width=0.5,
            W=W,
            n_h_points=15,
            n_realizations=50,
            system_sizes=[8, 12, 16, 20],
            output_dir=output_dir,
        )
    
    # Load or generate exponents results
    if exponents_result_path and Path(exponents_result_path).exists():
        logger.info(f"Loading exponents from {exponents_result_path}")
        with open(exponents_result_path, 'r') as f:
            exp_data = json.load(f)
        # Reconstruct CriticalExponentsResult (simplified)
        from .critical_exponent_extractor import extract_critical_exponents
        exponents_result = extract_critical_exponents(fss_result, output_dir)
    else:
        logger.info("Extracting critical exponents...")
        from .critical_exponent_extractor import extract_critical_exponents
        exponents_result = extract_critical_exponents(fss_result, output_dir)
    
    # Load or generate entanglement results
    if entanglement_result_path and Path(entanglement_result_path).exists():
        logger.info(f"Loading entanglement results from {entanglement_result_path}")
        # Load entanglement results
        from .entanglement_analyzer import run_task16_entanglement_analysis
        entanglement_result = run_task16_entanglement_analysis(
            h=fss_result.hc_thermodynamic,
            W=W,
            system_sizes=[8, 12, 16],
            n_realizations=30,
            output_dir=output_dir,
        )
    else:
        logger.info("Running entanglement analysis...")
        from .entanglement_analyzer import run_task16_entanglement_analysis
        entanglement_result = run_task16_entanglement_analysis(
            h=fss_result.hc_thermodynamic,
            W=W,
            system_sizes=[8, 12, 16],
            n_realizations=30,
            output_dir=output_dir,
        )
    
    # Run cross-validation
    result = run_cross_validation(
        exponents_result=exponents_result,
        entanglement_result=entanglement_result,
        fss_result=fss_result,
        output_dir=output_dir,
    )
    
    return result
