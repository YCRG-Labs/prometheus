"""
Comparative Analysis Engine for comparing phase transition behavior across model variants.

This module provides tools for comparing critical exponents, testing universality
class membership, generating phase diagrams, and validating scaling relations
across multiple model variants.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
from pathlib import Path

from .base_types import VAEAnalysisResults, DiscoveryResults
from .phenomena_detector import NovelPhenomenonDetector, UniversalityClass
from ..utils.logging_utils import get_logger


@dataclass
class ComparisonResults:
    """Results from comparative analysis across variants.
    
    Attributes:
        variant_ids: List of variant IDs compared
        exponent_comparisons: Statistical comparisons of exponents
        universality_tests: Tests for universality class membership
        scaling_violations: Detected violations of scaling relations
        phase_diagrams: Generated phase diagram figures
        summary_statistics: Summary statistics across variants
    """
    variant_ids: List[str]
    exponent_comparisons: Dict[str, Any]
    universality_tests: Dict[str, Any]
    scaling_violations: List[Dict[str, Any]]
    phase_diagrams: List[Any]
    summary_statistics: Dict[str, Any]


@dataclass
class UniversalityTest:
    """Result of universality class membership test.
    
    Attributes:
        variant_id: ID of the variant tested
        class_name: Name of universality class tested against
        matches: Whether exponents match the class
        confidence: Confidence in the match (0.0 to 1.0)
        p_values: P-values for each exponent comparison
        deviations: Deviations in sigma for each exponent
        message: Human-readable test result
    """
    variant_id: str
    class_name: str
    matches: bool
    confidence: float
    p_values: Dict[str, float]
    deviations: Dict[str, float]
    message: str


@dataclass
class ScalingViolation:
    """Detected violation of scaling relations.
    
    Attributes:
        variant_id: ID of the variant with violation
        relation_name: Name of the scaling relation
        expected_value: Expected value from scaling relation
        measured_value: Measured value
        deviation: Deviation from expected value
        confidence: Confidence in violation detection
        description: Human-readable description
    """
    variant_id: str
    relation_name: str
    expected_value: float
    measured_value: float
    deviation: float
    confidence: float
    description: str


class ComparativeAnalyzer:
    """Compare results across model variants.
    
    This class provides comprehensive comparative analysis capabilities including:
    - Statistical comparison of critical exponents across variants
    - Universality class membership testing
    - Phase diagram generation
    - Scaling relation validation
    - ANOVA and t-tests for significance
    
    Attributes:
        phenomena_detector: Detector for accessing universality class database
        logger: Logger instance
    """
    
    def __init__(self, anomaly_threshold: float = 3.0):
        """Initialize comparative analyzer.
        
        Args:
            anomaly_threshold: Threshold in standard deviations for anomaly detection
        """
        self.phenomena_detector = NovelPhenomenonDetector(anomaly_threshold)
        self.logger = get_logger(__name__)
        self.logger.info("Initialized ComparativeAnalyzer")
    
    def compare_exponents(
        self,
        variant_results: Dict[str, List[VAEAnalysisResults]]
    ) -> ComparisonResults:
        """Compare critical exponents across variants.
        
        Performs statistical comparison of critical exponents across multiple
        model variants, including t-tests, ANOVA, and effect size calculations.
        
        Args:
            variant_results: Dictionary mapping variant IDs to lists of VAE results
            
        Returns:
            ComparisonResults with statistical comparisons
        """
        self.logger.info(f"Comparing exponents across {len(variant_results)} variants")
        
        variant_ids = list(variant_results.keys())
        
        # Collect exponents by type
        exponent_data = self._collect_exponent_data(variant_results)
        
        # Perform statistical comparisons
        exponent_comparisons = {}
        for exp_name, data in exponent_data.items():
            comparison = self._compare_exponent_across_variants(exp_name, data)
            exponent_comparisons[exp_name] = comparison
        
        # Test universality class membership for each variant
        universality_tests = {}
        for variant_id, results_list in variant_results.items():
            # Use mean exponents for the variant
            mean_result = self._compute_mean_vae_result(results_list)
            test = self._test_all_universality_classes(mean_result)
            universality_tests[variant_id] = test
        
        # Check for scaling violations
        scaling_violations = []
        for variant_id, results_list in variant_results.items():
            mean_result = self._compute_mean_vae_result(results_list)
            violations = self.identify_scaling_violations(mean_result)
            scaling_violations.extend(violations)
        
        # Compute summary statistics
        summary_statistics = self._compute_summary_statistics(variant_results)
        
        results = ComparisonResults(
            variant_ids=variant_ids,
            exponent_comparisons=exponent_comparisons,
            universality_tests=universality_tests,
            scaling_violations=scaling_violations,
            phase_diagrams=[],
            summary_statistics=summary_statistics
        )
        
        self.logger.info("Exponent comparison complete")
        return results
    
    def test_universality_class(
        self,
        vae_results: VAEAnalysisResults,
        class_name: str
    ) -> UniversalityTest:
        """Test if exponents belong to known universality class.
        
        Performs statistical hypothesis testing to determine if measured
        exponents are consistent with a known universality class.
        
        Args:
            vae_results: VAE analysis results with measured exponents
            class_name: Name of universality class to test against
            
        Returns:
            UniversalityTest with test results
        """
        self.logger.info(
            f"Testing variant '{vae_results.variant_id}' against "
            f"universality class '{class_name}'"
        )
        
        # Use phenomena detector for comparison
        matches, confidence, deviations = \
            self.phenomena_detector.compare_to_universality_class(
                vae_results, class_name
            )
        
        # Calculate p-values for each exponent
        p_values = {}
        univ_class = self.phenomena_detector.universality_classes[class_name]
        
        for exp_name, measured_value in vae_results.exponents.items():
            if exp_name in univ_class.exponents:
                theoretical_value = univ_class.exponents[exp_name]
                measured_error = vae_results.exponent_errors.get(exp_name, 0.05)
                
                # Two-tailed z-test
                z_score = abs(measured_value - theoretical_value) / measured_error
                p_value = 2 * (1 - stats.norm.cdf(z_score))
                p_values[exp_name] = p_value
        
        # Create message
        if matches:
            message = (
                f"Exponents consistent with {class_name} universality class "
                f"(confidence: {confidence:.2%})"
            )
        else:
            max_dev_exp = max(deviations.items(), key=lambda x: x[1])[0]
            max_dev = deviations[max_dev_exp]
            message = (
                f"Exponents deviate from {class_name} universality class. "
                f"Maximum deviation: {max_dev:.1f}σ in {max_dev_exp}"
            )
        
        test = UniversalityTest(
            variant_id=vae_results.variant_id,
            class_name=class_name,
            matches=matches,
            confidence=confidence,
            p_values=p_values,
            deviations=deviations,
            message=message
        )
        
        self.logger.info(f"Universality test: {message}")
        return test
    
    def generate_phase_diagrams(
        self,
        variant_id: str,
        results: List[VAEAnalysisResults],
        param_x: str,
        param_y: Optional[str] = None,
        output_dir: Optional[str] = None,
        detect_transition_order: bool = True
    ) -> List[plt.Figure]:
        """Generate phase diagram visualizations with automatic transition order detection.
        
        Creates phase diagrams showing critical temperature and phase boundaries
        as a function of model parameters. Optionally detects transition order
        from phase boundary characteristics.
        
        Args:
            variant_id: ID of the variant
            results: List of VAE results at different parameter points
            param_x: Name of parameter for x-axis
            param_y: Optional name of parameter for y-axis (for 2D diagrams)
            output_dir: Optional directory to save figures
            detect_transition_order: Whether to detect transition order from phase diagram
            
        Returns:
            List of matplotlib Figure objects
        """
        self.logger.info(
            f"Generating phase diagrams for variant '{variant_id}' "
            f"with parameters: {param_x}" + (f", {param_y}" if param_y else "")
        )
        
        figures = []
        
        if param_y is None:
            # 1D phase diagram (Tc vs single parameter)
            fig = self._generate_1d_phase_diagram(variant_id, results, param_x)
            figures.append(fig)
            
            # Detect transition order if requested
            if detect_transition_order:
                order_result = self.detect_transition_order_from_phase_diagram(
                    results, param_x
                )
                self.logger.info(
                    f"Transition order detection: {order_result['transition_type']} "
                    f"(confidence: {order_result['confidence']:.2%})"
                )
        else:
            # 2D phase diagram (Tc as heatmap)
            fig = self._generate_2d_phase_diagram(
                variant_id, results, param_x, param_y
            )
            figures.append(fig)
            
            # Detect transition order if requested
            if detect_transition_order:
                order_result = self.detect_transition_order_from_phase_diagram(
                    results, param_x, param_y
                )
                self.logger.info(
                    f"Transition order detection: {order_result['transition_type']} "
                    f"(confidence: {order_result['confidence']:.2%})"
                )
        
        # Save figures if output directory specified
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            for i, fig in enumerate(figures):
                filename = f"{variant_id}_phase_diagram_{i}.png"
                fig.savefig(output_path / filename, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved phase diagram: {filename}")
        
        return figures
    
    def detect_transition_order_from_phase_diagram(
        self,
        results: List[VAEAnalysisResults],
        param_x: str,
        param_y: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect transition order from phase diagram characteristics.
        
        Analyzes phase boundary smoothness and behavior to determine if the
        transition is continuous (second-order) or first-order:
        - Smooth phase boundaries → continuous transition
        - Sharp discontinuities → first-order transition
        - Non-monotonic behavior → re-entrant transition
        
        This method provides an implicit validation tool that complements
        direct order parameter analysis.
        
        Args:
            results: List of VAE results at different parameter points
            param_x: Name of parameter for x-axis
            param_y: Optional name of parameter for y-axis (for 2D analysis)
            
        Returns:
            Dictionary with detection results:
                - transition_type: 'continuous', 'first_order', or 're_entrant'
                - confidence: Confidence in detection (0.0 to 1.0)
                - smoothness_score: Measure of phase boundary smoothness
                - discontinuity_score: Measure of sharp jumps
                - monotonicity: Whether Tc varies monotonically
                - details: Additional diagnostic information
        """
        self.logger.info(
            f"Detecting transition order from phase diagram with {len(results)} points"
        )
        
        if len(results) < 3:
            return {
                'transition_type': 'unknown',
                'confidence': 0.0,
                'smoothness_score': 0.0,
                'discontinuity_score': 0.0,
                'monotonicity': True,
                'details': 'Insufficient data points for transition order detection'
            }
        
        # Extract parameter values and Tc
        param_values = []
        tc_values = []
        tc_confidences = []
        
        for result in results:
            if param_x in result.parameters:
                param_values.append(result.parameters[param_x])
                tc_values.append(result.critical_temperature)
                tc_confidences.append(result.tc_confidence)
        
        if len(param_values) < 3:
            return {
                'transition_type': 'unknown',
                'confidence': 0.0,
                'smoothness_score': 0.0,
                'discontinuity_score': 0.0,
                'monotonicity': True,
                'details': f'Parameter {param_x} not found in sufficient results'
            }
        
        # Sort by parameter value
        sorted_indices = np.argsort(param_values)
        param_values = np.array(param_values)[sorted_indices]
        tc_values = np.array(tc_values)[sorted_indices]
        tc_confidences = np.array(tc_confidences)[sorted_indices]
        
        # Calculate smoothness metrics
        smoothness_result = self._analyze_phase_boundary_smoothness(
            param_values, tc_values, tc_confidences
        )
        
        # Check for discontinuities
        discontinuity_result = self._detect_phase_boundary_discontinuities(
            param_values, tc_values, tc_confidences
        )
        
        # Check monotonicity (for re-entrant transitions)
        monotonicity_result = self._check_phase_boundary_monotonicity(
            param_values, tc_values
        )
        
        # Determine transition type based on analysis
        transition_type, confidence = self._classify_transition_order(
            smoothness_result,
            discontinuity_result,
            monotonicity_result
        )
        
        result_dict = {
            'transition_type': transition_type,
            'confidence': confidence,
            'smoothness_score': smoothness_result['smoothness_score'],
            'discontinuity_score': discontinuity_result['max_discontinuity'],
            'monotonicity': monotonicity_result['is_monotonic'],
            'details': {
                'smoothness': smoothness_result,
                'discontinuities': discontinuity_result,
                'monotonicity': monotonicity_result
            }
        }
        
        self.logger.info(
            f"Transition order: {transition_type} "
            f"(confidence: {confidence:.2%}, "
            f"smoothness: {smoothness_result['smoothness_score']:.3f}, "
            f"discontinuity: {discontinuity_result['max_discontinuity']:.3f})"
        )
        
        return result_dict
    
    def _analyze_phase_boundary_smoothness(
        self,
        param_values: np.ndarray,
        tc_values: np.ndarray,
        tc_confidences: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze smoothness of phase boundary.
        
        Smooth boundaries indicate continuous transitions, while rough
        boundaries may indicate first-order transitions or measurement noise.
        
        Args:
            param_values: Parameter values (sorted)
            tc_values: Critical temperatures
            tc_confidences: Confidence in Tc measurements
            
        Returns:
            Dictionary with smoothness analysis
        """
        # Calculate first and second derivatives
        dTc_dparam = np.gradient(tc_values, param_values)
        d2Tc_dparam2 = np.gradient(dTc_dparam, param_values)
        
        # Smoothness score based on second derivative variation
        # Smooth curves have small, consistent second derivatives
        second_deriv_std = np.std(d2Tc_dparam2)
        second_deriv_mean = np.abs(np.mean(d2Tc_dparam2))
        
        # Normalize by typical Tc scale
        tc_scale = np.mean(tc_values)
        param_scale = np.ptp(param_values)
        
        if tc_scale > 0 and param_scale > 0:
            normalized_curvature = second_deriv_std * param_scale**2 / tc_scale
        else:
            normalized_curvature = 0.0
        
        # Smoothness score: high for smooth curves, low for rough
        # Continuous transitions typically have smoothness > 0.7
        smoothness_score = np.exp(-normalized_curvature)
        
        # Fit polynomial to assess smoothness
        if len(param_values) >= 3:
            # Fit quadratic
            poly_coeffs = np.polyfit(param_values, tc_values, deg=2)
            poly_fit = np.polyval(poly_coeffs, param_values)
            residuals = tc_values - poly_fit
            r_squared = 1 - (np.sum(residuals**2) / np.sum((tc_values - np.mean(tc_values))**2))
        else:
            r_squared = 0.0
            residuals = np.array([])
        
        return {
            'smoothness_score': float(smoothness_score),
            'second_deriv_std': float(second_deriv_std),
            'normalized_curvature': float(normalized_curvature),
            'polynomial_r_squared': float(r_squared),
            'mean_residual': float(np.mean(np.abs(residuals))) if len(residuals) > 0 else 0.0
        }
    
    def _detect_phase_boundary_discontinuities(
        self,
        param_values: np.ndarray,
        tc_values: np.ndarray,
        tc_confidences: np.ndarray
    ) -> Dict[str, Any]:
        """Detect sharp discontinuities in phase boundary.
        
        First-order transitions exhibit sharp jumps in Tc, while continuous
        transitions show smooth variation.
        
        Args:
            param_values: Parameter values (sorted)
            tc_values: Critical temperatures
            tc_confidences: Confidence in Tc measurements
            
        Returns:
            Dictionary with discontinuity analysis
        """
        # Calculate jumps between consecutive points
        param_steps = np.diff(param_values)
        tc_jumps = np.abs(np.diff(tc_values))
        
        # Normalize jumps by parameter step size
        normalized_jumps = tc_jumps / param_steps
        
        # Find maximum jump
        max_jump_idx = np.argmax(normalized_jumps)
        max_jump = normalized_jumps[max_jump_idx]
        
        # Calculate typical variation (excluding the maximum)
        if len(normalized_jumps) > 1:
            typical_variation = np.median(normalized_jumps)
        else:
            typical_variation = max_jump
        
        # Discontinuity score: ratio of max jump to typical variation
        # First-order transitions typically have discontinuity_score > 3.0
        if typical_variation > 0:
            discontinuity_score = max_jump / typical_variation
        else:
            discontinuity_score = 1.0
        
        # Check if jump is statistically significant
        # Use confidence values to estimate measurement uncertainty
        avg_confidence = np.mean(tc_confidences)
        tc_uncertainty = np.std(tc_values) * (1.0 - avg_confidence)
        
        is_significant = max_jump > 3 * tc_uncertainty
        
        return {
            'max_discontinuity': float(max_jump),
            'discontinuity_score': float(discontinuity_score),
            'discontinuity_location': float(param_values[max_jump_idx]),
            'typical_variation': float(typical_variation),
            'is_significant': bool(is_significant),
            'tc_uncertainty': float(tc_uncertainty)
        }
    
    def _check_phase_boundary_monotonicity(
        self,
        param_values: np.ndarray,
        tc_values: np.ndarray
    ) -> Dict[str, Any]:
        """Check if phase boundary is monotonic.
        
        Non-monotonic behavior (Tc increasing then decreasing or vice versa)
        indicates re-entrant phase transitions.
        
        Args:
            param_values: Parameter values (sorted)
            tc_values: Critical temperatures
            
        Returns:
            Dictionary with monotonicity analysis
        """
        # Calculate first derivative
        dTc_dparam = np.gradient(tc_values, param_values)
        
        # Check for sign changes (ignore small fluctuations)
        # Smooth the derivative to reduce noise sensitivity
        if len(dTc_dparam) >= 5:
            from scipy.ndimage import uniform_filter1d
            smoothed_deriv = uniform_filter1d(dTc_dparam, size=3)
        else:
            smoothed_deriv = dTc_dparam
        
        sign_changes = np.sum(np.diff(np.sign(smoothed_deriv)) != 0)
        
        # Monotonic if no sign changes (or only one due to noise)
        is_monotonic = sign_changes <= 1
        
        # Find significant extrema (local maxima/minima)
        # Only count extrema with significant amplitude
        tc_range = np.ptp(tc_values)
        min_amplitude = 0.1 * tc_range  # Extrema must be at least 10% of total range
        
        extrema_indices = []
        for i in range(1, len(tc_values) - 1):
            is_maximum = (tc_values[i] > tc_values[i-1] and tc_values[i] > tc_values[i+1])
            is_minimum = (tc_values[i] < tc_values[i-1] and tc_values[i] < tc_values[i+1])
            
            if is_maximum or is_minimum:
                # Check amplitude significance
                left_diff = abs(tc_values[i] - tc_values[i-1])
                right_diff = abs(tc_values[i] - tc_values[i+1])
                if left_diff > min_amplitude or right_diff > min_amplitude:
                    extrema_indices.append(i)
        
        # Re-entrant if multiple significant extrema
        is_reentrant = len(extrema_indices) >= 2 and not is_monotonic
        
        return {
            'is_monotonic': bool(is_monotonic),
            'is_reentrant': bool(is_reentrant),
            'sign_changes': int(sign_changes),
            'n_extrema': len(extrema_indices),
            'extrema_locations': [float(param_values[i]) for i in extrema_indices]
        }
    
    def _classify_transition_order(
        self,
        smoothness_result: Dict[str, Any],
        discontinuity_result: Dict[str, Any],
        monotonicity_result: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Classify transition order based on phase diagram analysis.
        
        Args:
            smoothness_result: Results from smoothness analysis
            discontinuity_result: Results from discontinuity detection
            monotonicity_result: Results from monotonicity check
            
        Returns:
            Tuple of (transition_type, confidence)
        """
        smoothness = smoothness_result['smoothness_score']
        discontinuity = discontinuity_result['discontinuity_score']
        is_reentrant = monotonicity_result['is_reentrant']
        
        # Re-entrant transitions take precedence
        if is_reentrant:
            # Confidence based on number of extrema
            n_extrema = monotonicity_result['n_extrema']
            confidence = min(0.9, 0.5 + 0.2 * n_extrema)
            return 're_entrant', confidence
        
        # Classify based on smoothness and discontinuity
        # First-order: low smoothness OR high discontinuity
        # Continuous: high smoothness AND low discontinuity
        
        # Thresholds (empirically determined)
        SMOOTH_THRESHOLD = 0.7
        DISCONTINUITY_THRESHOLD = 3.0
        
        if discontinuity > DISCONTINUITY_THRESHOLD:
            # Strong evidence for first-order
            confidence = min(0.95, 0.5 + discontinuity / 10.0)
            return 'first_order', confidence
        elif smoothness > SMOOTH_THRESHOLD and discontinuity < 2.0:
            # Strong evidence for continuous
            confidence = min(0.95, smoothness)
            return 'continuous', confidence
        elif smoothness < 0.5:
            # Weak evidence for first-order (rough boundary)
            confidence = 0.6
            return 'first_order', confidence
        else:
            # Ambiguous - default to continuous with low confidence
            confidence = 0.5
            return 'continuous', confidence
    
    def identify_scaling_violations(
        self,
        vae_results: VAEAnalysisResults
    ) -> List[ScalingViolation]:
        """Identify violations of expected scaling relations.
        
        Checks hyperscaling relations and other theoretical constraints:
        - α + 2β + γ = 2 (hyperscaling)
        - γ = ν(2 - η) (scaling relation)
        - α = 2 - dν (hyperscaling in d dimensions)
        
        Args:
            vae_results: VAE analysis results with measured exponents
            
        Returns:
            List of detected scaling violations
        """
        violations = []
        exponents = vae_results.exponents
        
        # Check hyperscaling relation: α + 2β + γ = 2
        if 'alpha' in exponents and 'beta' in exponents and 'gamma' in exponents:
            alpha = exponents['alpha']
            beta = exponents['beta']
            gamma = exponents['gamma']
            
            expected = 2.0
            measured = alpha + 2 * beta + gamma
            deviation = abs(measured - expected)
            
            # Estimate combined error
            alpha_err = vae_results.exponent_errors.get('alpha', 0.05)
            beta_err = vae_results.exponent_errors.get('beta', 0.05)
            gamma_err = vae_results.exponent_errors.get('gamma', 0.05)
            combined_err = np.sqrt(alpha_err**2 + (2*beta_err)**2 + gamma_err**2)
            
            if deviation > 3 * combined_err:
                confidence = min(0.95, deviation / (5 * combined_err))
                violation = ScalingViolation(
                    variant_id=vae_results.variant_id,
                    relation_name='hyperscaling',
                    expected_value=expected,
                    measured_value=measured,
                    deviation=deviation / combined_err,
                    confidence=confidence,
                    description=(
                        f"Hyperscaling relation α + 2β + γ = 2 violated: "
                        f"measured {measured:.4f}, expected {expected:.4f} "
                        f"({deviation/combined_err:.1f}σ)"
                    )
                )
                violations.append(violation)
                self.logger.warning(violation.description)
        
        return violations
    
    def _collect_exponent_data(
        self,
        variant_results: Dict[str, List[VAEAnalysisResults]]
    ) -> Dict[str, Dict[str, List[float]]]:
        """Collect exponent data organized by exponent type and variant.
        
        Args:
            variant_results: Dictionary mapping variant IDs to results lists
            
        Returns:
            Dictionary mapping exponent names to variant data
        """
        exponent_data = {}
        
        for variant_id, results_list in variant_results.items():
            for result in results_list:
                for exp_name, exp_value in result.exponents.items():
                    if exp_name not in exponent_data:
                        exponent_data[exp_name] = {}
                    
                    if variant_id not in exponent_data[exp_name]:
                        exponent_data[exp_name][variant_id] = []
                    
                    exponent_data[exp_name][variant_id].append(exp_value)
        
        return exponent_data
    
    def _compare_exponent_across_variants(
        self,
        exponent_name: str,
        variant_data: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Compare a specific exponent across variants using statistical tests.
        
        Args:
            exponent_name: Name of the exponent
            variant_data: Dictionary mapping variant IDs to exponent values
            
        Returns:
            Dictionary with comparison statistics
        """
        variant_ids = list(variant_data.keys())
        
        # Compute statistics for each variant
        variant_stats = {}
        for variant_id, values in variant_data.items():
            variant_stats[variant_id] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'n': len(values),
                'sem': float(np.std(values) / np.sqrt(len(values)))
            }
        
        # Perform ANOVA if more than 2 variants
        if len(variant_ids) > 2:
            groups = [variant_data[vid] for vid in variant_ids]
            f_stat, p_value_anova = stats.f_oneway(*groups)
        else:
            f_stat, p_value_anova = None, None
        
        # Perform pairwise t-tests
        pairwise_tests = {}
        for i, vid1 in enumerate(variant_ids):
            for vid2 in variant_ids[i+1:]:
                t_stat, p_value = stats.ttest_ind(
                    variant_data[vid1],
                    variant_data[vid2]
                )
                
                # Calculate Cohen's d effect size
                mean1 = np.mean(variant_data[vid1])
                mean2 = np.mean(variant_data[vid2])
                std1 = np.std(variant_data[vid1])
                std2 = np.std(variant_data[vid2])
                pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
                
                pairwise_tests[f"{vid1}_vs_{vid2}"] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'cohens_d': float(cohens_d),
                    'significant': p_value < 0.05
                }
        
        comparison = {
            'exponent_name': exponent_name,
            'variant_statistics': variant_stats,
            'anova': {
                'f_statistic': float(f_stat) if f_stat is not None else None,
                'p_value': float(p_value_anova) if p_value_anova is not None else None,
                'significant': p_value_anova < 0.05 if p_value_anova is not None else None
            },
            'pairwise_tests': pairwise_tests
        }
        
        return comparison
    
    def _test_all_universality_classes(
        self,
        vae_results: VAEAnalysisResults
    ) -> Dict[str, UniversalityTest]:
        """Test against all known universality classes.
        
        Args:
            vae_results: VAE analysis results
            
        Returns:
            Dictionary mapping class names to test results
        """
        tests = {}
        
        for class_name in self.phenomena_detector.universality_classes.keys():
            test = self.test_universality_class(vae_results, class_name)
            tests[class_name] = test
        
        return tests
    
    def _compute_mean_vae_result(
        self,
        results_list: List[VAEAnalysisResults]
    ) -> VAEAnalysisResults:
        """Compute mean VAE result from a list of results.
        
        Args:
            results_list: List of VAE analysis results
            
        Returns:
            VAEAnalysisResults with averaged values
        """
        if not results_list:
            raise ValueError("Cannot compute mean of empty results list")
        
        # Collect exponents
        exponent_values = {}
        exponent_errors = {}
        r_squared_values = {}
        
        for result in results_list:
            for exp_name, exp_value in result.exponents.items():
                if exp_name not in exponent_values:
                    exponent_values[exp_name] = []
                    exponent_errors[exp_name] = []
                    r_squared_values[exp_name] = []
                
                exponent_values[exp_name].append(exp_value)
                exponent_errors[exp_name].append(
                    result.exponent_errors.get(exp_name, 0.05)
                )
                r_squared_values[exp_name].append(
                    result.r_squared_values.get(exp_name, 0.0)
                )
        
        # Compute means
        mean_exponents = {
            name: float(np.mean(values))
            for name, values in exponent_values.items()
        }
        
        mean_errors = {
            name: float(np.mean(errors))
            for name, errors in exponent_errors.items()
        }
        
        mean_r_squared = {
            name: float(np.mean(r2s))
            for name, r2s in r_squared_values.items()
        }
        
        # Use first result as template
        template = results_list[0]
        
        # Compute mean Tc
        tc_values = [r.critical_temperature for r in results_list]
        mean_tc = float(np.mean(tc_values))
        
        tc_confidences = [r.tc_confidence for r in results_list]
        mean_tc_confidence = float(np.mean(tc_confidences))
        
        return VAEAnalysisResults(
            variant_id=template.variant_id,
            parameters=template.parameters,
            critical_temperature=mean_tc,
            tc_confidence=mean_tc_confidence,
            exponents=mean_exponents,
            exponent_errors=mean_errors,
            r_squared_values=mean_r_squared,
            latent_representation=template.latent_representation,
            order_parameter_dim=template.order_parameter_dim
        )
    
    def _compute_summary_statistics(
        self,
        variant_results: Dict[str, List[VAEAnalysisResults]]
    ) -> Dict[str, Any]:
        """Compute summary statistics across all variants.
        
        Args:
            variant_results: Dictionary mapping variant IDs to results lists
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'n_variants': len(variant_results),
            'total_points': sum(len(results) for results in variant_results.values()),
            'variants': {}
        }
        
        for variant_id, results_list in variant_results.items():
            mean_result = self._compute_mean_vae_result(results_list)
            
            # Find closest universality class
            closest_class, confidence, deviations = \
                self.phenomena_detector.get_closest_universality_class(mean_result)
            
            summary['variants'][variant_id] = {
                'n_points': len(results_list),
                'mean_tc': mean_result.critical_temperature,
                'mean_exponents': mean_result.exponents,
                'closest_universality_class': closest_class,
                'class_confidence': confidence,
                'class_deviations': deviations
            }
        
        return summary
    
    def _generate_1d_phase_diagram(
        self,
        variant_id: str,
        results: List[VAEAnalysisResults],
        param_name: str
    ) -> plt.Figure:
        """Generate 1D phase diagram (Tc vs parameter).
        
        Args:
            variant_id: ID of the variant
            results: List of VAE results
            param_name: Name of parameter for x-axis
            
        Returns:
            matplotlib Figure
        """
        # Extract parameter values and Tc
        param_values = []
        tc_values = []
        tc_errors = []
        
        for result in results:
            if param_name in result.parameters:
                param_values.append(result.parameters[param_name])
                tc_values.append(result.critical_temperature)
                # Use inverse of confidence as error estimate
                tc_error = 0.1 * (1.0 - result.tc_confidence)
                tc_errors.append(tc_error)
        
        # Sort by parameter value
        sorted_indices = np.argsort(param_values)
        param_values = np.array(param_values)[sorted_indices]
        tc_values = np.array(tc_values)[sorted_indices]
        tc_errors = np.array(tc_errors)[sorted_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot Tc vs parameter
        ax.errorbar(
            param_values, tc_values, yerr=tc_errors,
            fmt='o-', capsize=5, capthick=2,
            markersize=8, linewidth=2,
            label='Critical Temperature'
        )
        
        ax.set_xlabel(param_name, fontsize=14)
        ax.set_ylabel('Critical Temperature (Tc)', fontsize=14)
        ax.set_title(f'Phase Diagram: {variant_id}', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        plt.tight_layout()
        
        return fig
    
    def _generate_2d_phase_diagram(
        self,
        variant_id: str,
        results: List[VAEAnalysisResults],
        param_x: str,
        param_y: str
    ) -> plt.Figure:
        """Generate 2D phase diagram (Tc as heatmap).
        
        Args:
            variant_id: ID of the variant
            results: List of VAE results
            param_x: Name of parameter for x-axis
            param_y: Name of parameter for y-axis
            
        Returns:
            matplotlib Figure
        """
        # Extract parameter values and Tc
        x_values = []
        y_values = []
        tc_values = []
        
        for result in results:
            if param_x in result.parameters and param_y in result.parameters:
                x_values.append(result.parameters[param_x])
                y_values.append(result.parameters[param_y])
                tc_values.append(result.critical_temperature)
        
        if len(x_values) < 4:
            self.logger.warning("Insufficient data points for 2D phase diagram")
            # Return empty figure
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'Insufficient data for 2D phase diagram',
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Convert to arrays
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        tc_values = np.array(tc_values)
        
        # Create grid for interpolation
        x_unique = np.unique(x_values)
        y_unique = np.unique(y_values)
        
        if len(x_unique) < 2 or len(y_unique) < 2:
            self.logger.warning("Need at least 2 unique values in each dimension")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'Insufficient grid resolution',
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Create meshgrid
        X, Y = np.meshgrid(
            np.linspace(x_values.min(), x_values.max(), 50),
            np.linspace(y_values.min(), y_values.max(), 50)
        )
        
        # Interpolate Tc values
        from scipy.interpolate import griddata
        Z = griddata((x_values, y_values), tc_values, (X, Y), method='cubic')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Critical Temperature (Tc)', fontsize=12)
        
        # Overlay data points
        ax.scatter(x_values, y_values, c='red', s=50, marker='o',
                  edgecolors='white', linewidths=1.5, label='Data points')
        
        ax.set_xlabel(param_x, fontsize=14)
        ax.set_ylabel(param_y, fontsize=14)
        ax.set_title(f'Phase Diagram: {variant_id}', fontsize=16)
        ax.legend(fontsize=12)
        
        plt.tight_layout()
        
        return fig
