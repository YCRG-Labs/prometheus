"""
Validation Framework for rigorous statistical hypothesis validation.

This module provides comprehensive statistical validation tools for testing
research hypotheses about phase transitions, including bootstrap confidence
intervals, hypothesis testing, and effect size calculations.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy import stats
from dataclasses import dataclass

from .base_types import ValidationResult, VAEAnalysisResults, SimulationData
from .phenomena_detector import NovelPhenomenonDetector
from ..utils.logging_utils import get_logger


class ValidationFramework:
    """Statistical validation of research hypotheses.
    
    This class provides rigorous statistical methods for validating research
    hypotheses about phase transitions, including:
    - Bootstrap confidence intervals for exponent predictions
    - Hypothesis testing (t-tests, chi-square)
    - Effect size calculations (Cohen's d)
    - Universality class membership validation
    - Phase transition order validation
    - Multiple comparison corrections
    
    Attributes:
        n_bootstrap: Number of bootstrap samples for CI calculation
        alpha: Significance level for hypothesis tests
        phenomena_detector: Detector for universality class comparisons
        logger: Logger instance
    """
    
    def __init__(
        self,
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
        anomaly_threshold: float = 3.0
    ):
        """Initialize validation framework.
        
        Args:
            n_bootstrap: Number of bootstrap samples for confidence intervals
            alpha: Significance level for hypothesis tests (default: 0.05)
            anomaly_threshold: Threshold for anomaly detection in sigma
        """
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.phenomena_detector = NovelPhenomenonDetector(anomaly_threshold)
        self.logger = get_logger(__name__)
        self.logger.info(
            f"Initialized ValidationFramework with {n_bootstrap} bootstrap samples "
            f"and α={alpha}"
        )
    
    def validate_exponent_prediction(
        self,
        hypothesis_id: str,
        predicted: float,
        measured: float,
        measured_error: float,
        predicted_error: Optional[float] = None
    ) -> ValidationResult:
        """Validate predicted vs measured exponent with bootstrap CI.
        
        Performs statistical validation of an exponent prediction using:
        - Bootstrap confidence intervals
        - Two-tailed t-test
        - Effect size calculation (Cohen's d)
        
        Args:
            hypothesis_id: ID of the hypothesis being validated
            predicted: Predicted exponent value
            measured: Measured exponent value
            measured_error: Standard error of measured value
            predicted_error: Optional error in prediction
            
        Returns:
            ValidationResult with validation outcome
        """
        self.logger.info(
            f"Validating exponent prediction: predicted={predicted:.4f}, "
            f"measured={measured:.4f}±{measured_error:.4f}"
        )
        
        # Generate bootstrap samples for measured value
        bootstrap_samples = np.random.normal(
            measured, measured_error, self.n_bootstrap
        )
        
        # Calculate bootstrap confidence interval
        ci_lower = np.percentile(bootstrap_samples, 100 * self.alpha / 2)
        ci_upper = np.percentile(bootstrap_samples, 100 * (1 - self.alpha / 2))
        
        # Check if prediction falls within CI
        validated = ci_lower <= predicted <= ci_upper
        
        # Calculate p-value using t-test
        # Test if measured value differs from predicted
        if predicted_error is not None:
            # Use combined error
            combined_error = np.sqrt(measured_error**2 + predicted_error**2)
        else:
            combined_error = measured_error
        
        t_statistic = (measured - predicted) / combined_error
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=self.n_bootstrap - 1))
        
        # Calculate effect size (Cohen's d)
        cohens_d = abs(measured - predicted) / measured_error
        
        # Calculate confidence based on p-value and CI overlap
        if validated:
            # Confidence increases as measured value approaches predicted
            deviation = abs(measured - predicted) / measured_error
            confidence = max(0.0, 1.0 - (deviation / 3.0))  # 3-sigma rule
        else:
            # Low confidence if prediction outside CI
            confidence = min(0.5, p_value)
        
        # Create message
        if validated:
            message = (
                f"Hypothesis validated: measured value {measured:.4f}±{measured_error:.4f} "
                f"consistent with prediction {predicted:.4f} "
                f"(95% CI: [{ci_lower:.4f}, {ci_upper:.4f}], p={p_value:.4f})"
            )
        else:
            message = (
                f"Hypothesis refuted: measured value {measured:.4f}±{measured_error:.4f} "
                f"outside 95% CI [{ci_lower:.4f}, {ci_upper:.4f}] for prediction {predicted:.4f} "
                f"(p={p_value:.4f})"
            )
        
        result = ValidationResult(
            hypothesis_id=hypothesis_id,
            validated=validated,
            confidence=confidence,
            p_values={'exponent': p_value},
            effect_sizes={'cohens_d': cohens_d},
            bootstrap_intervals={'exponent': (ci_lower, ci_upper)},
            message=message
        )
        
        self.logger.info(f"Validation result: {message}")
        
        return result
    
    def validate_universality_class(
        self,
        hypothesis_id: str,
        measured_exponents: Dict[str, float],
        measured_errors: Dict[str, float],
        class_name: str,
        apply_correction: bool = True
    ) -> ValidationResult:
        """Validate universality class membership with multiple comparison correction.
        
        Tests if measured exponents are consistent with a known universality class,
        applying Bonferroni correction for multiple comparisons.
        
        Args:
            hypothesis_id: ID of the hypothesis being validated
            measured_exponents: Dictionary of measured exponent values
            measured_errors: Dictionary of measurement errors
            class_name: Name of universality class to test against
            apply_correction: Whether to apply Bonferroni correction
            
        Returns:
            ValidationResult with validation outcome
        """
        self.logger.info(
            f"Validating universality class '{class_name}' membership "
            f"for {len(measured_exponents)} exponents"
        )
        
        # Get universality class from detector
        if class_name not in self.phenomena_detector.universality_classes:
            raise ValueError(f"Unknown universality class: {class_name}")
        
        univ_class = self.phenomena_detector.universality_classes[class_name]
        
        # Test each exponent
        p_values = {}
        effect_sizes = {}
        bootstrap_intervals = {}
        all_validated = True
        
        n_comparisons = len(measured_exponents)
        corrected_alpha = self.alpha / n_comparisons if apply_correction else self.alpha
        
        for exp_name, measured_value in measured_exponents.items():
            if exp_name not in univ_class.exponents:
                self.logger.warning(
                    f"Exponent '{exp_name}' not in universality class '{class_name}'"
                )
                continue
            
            theoretical_value = univ_class.exponents[exp_name]
            measured_error = measured_errors.get(exp_name, 0.05)
            theoretical_error = univ_class.exponent_errors.get(exp_name, 0.01)
            
            # Bootstrap confidence interval
            bootstrap_samples = np.random.normal(
                measured_value, measured_error, self.n_bootstrap
            )
            ci_lower = np.percentile(bootstrap_samples, 100 * corrected_alpha / 2)
            ci_upper = np.percentile(bootstrap_samples, 100 * (1 - corrected_alpha / 2))
            
            bootstrap_intervals[exp_name] = (ci_lower, ci_upper)
            
            # Z-test (two-tailed)
            combined_error = np.sqrt(measured_error**2 + theoretical_error**2)
            z_statistic = (measured_value - theoretical_value) / combined_error
            p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
            p_values[exp_name] = p_value
            
            # Effect size
            cohens_d = abs(measured_value - theoretical_value) / measured_error
            effect_sizes[f'{exp_name}_cohens_d'] = cohens_d
            
            # Check if validated
            if p_value < corrected_alpha:
                all_validated = False
                self.logger.warning(
                    f"Exponent '{exp_name}' fails validation: "
                    f"measured={measured_value:.4f}, theoretical={theoretical_value:.4f}, "
                    f"p={p_value:.4e} < {corrected_alpha:.4e}"
                )
        
        # Calculate overall confidence
        if all_validated:
            # Average p-value as confidence metric
            avg_p_value = np.mean(list(p_values.values()))
            confidence = 1.0 - avg_p_value
        else:
            # Low confidence if any exponent fails
            min_p_value = min(p_values.values())
            confidence = min(0.5, min_p_value)
        
        # Create message
        if all_validated:
            message = (
                f"Hypothesis validated: all {n_comparisons} exponents consistent with "
                f"'{class_name}' universality class "
                f"(Bonferroni-corrected α={corrected_alpha:.4f})"
            )
        else:
            failed_exponents = [
                name for name, p in p_values.items() if p < corrected_alpha
            ]
            message = (
                f"Hypothesis refuted: {len(failed_exponents)} exponent(s) "
                f"({', '.join(failed_exponents)}) inconsistent with '{class_name}' "
                f"universality class"
            )
        
        result = ValidationResult(
            hypothesis_id=hypothesis_id,
            validated=all_validated,
            confidence=confidence,
            p_values=p_values,
            effect_sizes=effect_sizes,
            bootstrap_intervals=bootstrap_intervals,
            message=message
        )
        
        self.logger.info(f"Validation result: {message}")
        
        return result
    
    def validate_phase_transition_order(
        self,
        hypothesis_id: str,
        simulation_data: SimulationData,
        predicted_order: int,
        critical_temperature: float
    ) -> ValidationResult:
        """Validate phase transition order (first-order vs continuous).
        
        Detects discontinuities in the order parameter to determine if the
        transition is first-order or continuous (second-order).
        
        Args:
            hypothesis_id: ID of the hypothesis being validated
            simulation_data: Raw simulation data with magnetizations
            predicted_order: Predicted transition order (1 or 2)
            critical_temperature: Critical temperature estimate
            
        Returns:
            ValidationResult with validation outcome
        """
        self.logger.info(
            f"Validating phase transition order: predicted={predicted_order}, "
            f"Tc={critical_temperature:.4f}"
        )
        
        if predicted_order not in [1, 2]:
            raise ValueError(f"Transition order must be 1 or 2, got {predicted_order}")
        
        # Get magnetization data
        temperatures = simulation_data.temperatures
        magnetizations = simulation_data.magnetizations  # Shape: (n_temps, n_samples)
        
        # Calculate mean absolute magnetization
        mean_mag = np.mean(np.abs(magnetizations), axis=1)
        
        # Find temperature closest to Tc
        tc_idx = np.argmin(np.abs(temperatures - critical_temperature))
        
        # Check for discontinuity around Tc
        window_size = min(3, len(temperatures) // 4)
        start_idx = max(0, tc_idx - window_size)
        end_idx = min(len(temperatures), tc_idx + window_size + 1)
        
        if end_idx - start_idx < 3:
            # Not enough data
            return ValidationResult(
                hypothesis_id=hypothesis_id,
                validated=False,
                confidence=0.0,
                p_values={},
                effect_sizes={},
                bootstrap_intervals={},
                message="Insufficient temperature points for order validation"
            )
        
        # Calculate maximum jump in magnetization
        mag_window = mean_mag[start_idx:end_idx]
        mag_diff = np.abs(np.diff(mag_window))
        max_jump = np.max(mag_diff)
        max_jump_idx = np.argmax(mag_diff)
        
        # Calculate typical fluctuation size
        mag_std = np.std(magnetizations, axis=1)
        typical_fluctuation = np.mean(mag_std[start_idx:end_idx])
        
        # Determine if first-order based on jump size
        # First-order: jump > 5 * typical fluctuation
        jump_ratio = max_jump / typical_fluctuation if typical_fluctuation > 0 else 0
        is_first_order = jump_ratio > 5.0
        
        # Validate against prediction
        if predicted_order == 1:
            validated = is_first_order
        else:  # predicted_order == 2
            validated = not is_first_order
        
        # Calculate confidence
        if validated:
            # Confidence based on how clear the signature is
            if predicted_order == 1:
                # Strong first-order signature increases confidence
                confidence = min(0.95, jump_ratio / 10.0)
            else:
                # Weak discontinuity increases confidence for continuous
                confidence = max(0.5, 1.0 - (jump_ratio / 10.0))
        else:
            # Low confidence if prediction wrong
            confidence = 0.2
        
        # Chi-square test for discontinuity
        # Compare magnetization distribution before and after Tc
        mag_before = magnetizations[start_idx:tc_idx, :].flatten()
        mag_after = magnetizations[tc_idx:end_idx, :].flatten()
        
        if len(mag_before) > 0 and len(mag_after) > 0:
            # Kolmogorov-Smirnov test
            ks_statistic, ks_p_value = stats.ks_2samp(mag_before, mag_after)
        else:
            ks_statistic, ks_p_value = 0.0, 1.0
        
        # Create message
        order_str = "first-order" if is_first_order else "continuous (second-order)"
        predicted_str = "first-order" if predicted_order == 1 else "continuous (second-order)"
        
        if validated:
            message = (
                f"Hypothesis validated: transition is {order_str} as predicted. "
                f"Jump ratio: {jump_ratio:.2f}, KS p-value: {ks_p_value:.4f}"
            )
        else:
            message = (
                f"Hypothesis refuted: transition appears {order_str}, "
                f"but {predicted_str} was predicted. "
                f"Jump ratio: {jump_ratio:.2f}, KS p-value: {ks_p_value:.4f}"
            )
        
        result = ValidationResult(
            hypothesis_id=hypothesis_id,
            validated=validated,
            confidence=confidence,
            p_values={'ks_test': ks_p_value},
            effect_sizes={
                'jump_ratio': jump_ratio,
                'max_jump': max_jump,
                'typical_fluctuation': typical_fluctuation
            },
            bootstrap_intervals={},
            message=message
        )
        
        self.logger.info(f"Validation result: {message}")
        
        return result
    
    def validate_hypothesis_comprehensive(
        self,
        hypothesis_id: str,
        vae_results: VAEAnalysisResults,
        simulation_data: SimulationData,
        predicted_exponents: Dict[str, float],
        predicted_errors: Optional[Dict[str, float]] = None,
        universality_class: Optional[str] = None,
        predicted_order: Optional[int] = None
    ) -> Dict[str, ValidationResult]:
        """Perform comprehensive validation of a research hypothesis.
        
        Validates multiple aspects of a hypothesis including exponent predictions,
        universality class membership, and transition order.
        
        Args:
            hypothesis_id: ID of the hypothesis
            vae_results: VAE analysis results with measured exponents
            simulation_data: Raw simulation data
            predicted_exponents: Dictionary of predicted exponent values
            predicted_errors: Optional dictionary of prediction errors
            universality_class: Optional universality class to test
            predicted_order: Optional predicted transition order (1 or 2)
            
        Returns:
            Dictionary mapping validation type to ValidationResult
        """
        self.logger.info(
            f"Performing comprehensive validation for hypothesis '{hypothesis_id}'"
        )
        
        results = {}
        
        # Validate individual exponent predictions
        for exp_name, predicted_value in predicted_exponents.items():
            if exp_name in vae_results.exponents:
                measured_value = vae_results.exponents[exp_name]
                measured_error = vae_results.exponent_errors.get(exp_name, 0.05)
                predicted_error = (
                    predicted_errors.get(exp_name) if predicted_errors else None
                )
                
                result = self.validate_exponent_prediction(
                    hypothesis_id=f"{hypothesis_id}_{exp_name}",
                    predicted=predicted_value,
                    measured=measured_value,
                    measured_error=measured_error,
                    predicted_error=predicted_error
                )
                results[f'exponent_{exp_name}'] = result
        
        # Validate universality class if specified
        if universality_class is not None:
            result = self.validate_universality_class(
                hypothesis_id=f"{hypothesis_id}_universality",
                measured_exponents=vae_results.exponents,
                measured_errors=vae_results.exponent_errors,
                class_name=universality_class
            )
            results['universality_class'] = result
        
        # Validate transition order if specified
        if predicted_order is not None:
            result = self.validate_phase_transition_order(
                hypothesis_id=f"{hypothesis_id}_order",
                simulation_data=simulation_data,
                predicted_order=predicted_order,
                critical_temperature=vae_results.critical_temperature
            )
            results['transition_order'] = result
        
        # Calculate overall validation
        all_validated = all(r.validated for r in results.values())
        avg_confidence = np.mean([r.confidence for r in results.values()])
        
        self.logger.info(
            f"Comprehensive validation complete: "
            f"{sum(r.validated for r in results.values())}/{len(results)} tests passed, "
            f"average confidence: {avg_confidence:.2%}"
        )
        
        return results
    
    def generate_validation_report(
        self,
        validation_results: Dict[str, ValidationResult]
    ) -> str:
        """Generate a human-readable validation report.
        
        Args:
            validation_results: Dictionary of validation results
            
        Returns:
            Formatted validation report string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary
        n_total = len(validation_results)
        n_validated = sum(1 for r in validation_results.values() if r.validated)
        avg_confidence = np.mean([r.confidence for r in validation_results.values()])
        
        report_lines.append(f"Total Tests: {n_total}")
        report_lines.append(f"Validated: {n_validated} ({n_validated/n_total*100:.1f}%)")
        report_lines.append(f"Average Confidence: {avg_confidence:.2%}")
        report_lines.append("")
        
        # Individual results
        report_lines.append("INDIVIDUAL TEST RESULTS:")
        report_lines.append("-" * 80)
        
        for test_name, result in validation_results.items():
            status = "✓ VALIDATED" if result.validated else "✗ REFUTED"
            report_lines.append(f"\n{test_name.upper()}: {status}")
            report_lines.append(f"  Confidence: {result.confidence:.2%}")
            report_lines.append(f"  Message: {result.message}")
            
            if result.p_values:
                report_lines.append(f"  P-values: {result.p_values}")
            
            if result.effect_sizes:
                report_lines.append(f"  Effect sizes: {result.effect_sizes}")
            
            if result.bootstrap_intervals:
                report_lines.append(f"  Bootstrap CIs: {result.bootstrap_intervals}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
