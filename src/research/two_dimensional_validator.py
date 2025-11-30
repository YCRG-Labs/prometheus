"""
Two-Dimensional Validation Space for comprehensive hypothesis validation.

This module implements a novel validation approach that combines:
- Bootstrap confidence intervals (individual exponent predictions)
- ANOVA statistical tests (comparative analysis across variants)

This creates a 2D validation space with axes:
- X-axis: Statistical significance (p-values from ANOVA/t-tests)
- Y-axis: Effect size (practical importance, Cohen's d)

The 2D space enables classification of findings into regions:
- VALIDATED: High significance + High effect size
- REFUTED: High significance + Low effect size (statistically different but not important)
- INCONCLUSIVE: Low significance + High effect size (potentially important, needs more data)
- LIKELY_FALSE_POSITIVE: Low significance + Low effect size
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
from pathlib import Path

from .base_types import VAEAnalysisResults, ValidationResult
from .validation_framework import ValidationFramework
from .comparative_analyzer import ComparativeAnalyzer
from ..utils.logging_utils import get_logger


@dataclass
class TwoDValidationResult:
    """Result from two-dimensional validation.
    
    Attributes:
        hypothesis_id: ID of the hypothesis
        region: Classification region (VALIDATED, REFUTED, INCONCLUSIVE, FALSE_POSITIVE)
        statistical_significance: P-value or significance score (0.0 to 1.0)
        effect_size: Cohen's d or similar effect size metric
        bootstrap_ci: Bootstrap confidence interval for individual prediction
        anova_result: ANOVA result for comparative analysis
        confidence: Overall confidence in the result (0.0 to 1.0)
        recommendation: Human-readable recommendation
        details: Additional diagnostic information
    """
    hypothesis_id: str
    region: str
    statistical_significance: float
    effect_size: float
    bootstrap_ci: Tuple[float, float]
    anova_result: Dict[str, Any]
    confidence: float
    recommendation: str
    details: Dict[str, Any]


class TwoDimensionalValidator:
    """Two-dimensional validation combining bootstrap CIs and ANOVA.
    
    This class implements a novel validation methodology that combines:
    1. Bootstrap confidence intervals for individual exponent predictions
    2. ANOVA for comparative analysis across model variants
    
    The combination creates a 2D validation space that provides more robust
    validation than either method alone:
    - Bootstrap CIs test if measured value matches prediction
    - ANOVA tests if variants differ significantly
    - Together: "Is this variant different AND does it match predictions?"
    
    The 2D space enables nuanced classification:
    - High significance + High effect → Validated finding
    - High significance + Low effect → False positive (not practically important)
    - Low significance + High effect → Needs more data (potentially important)
    - Low significance + Low effect → Likely false positive
    
    Attributes:
        validation_framework: Framework for bootstrap CI validation
        comparative_analyzer: Analyzer for ANOVA and comparative tests
        significance_threshold: Threshold for statistical significance (default: 0.05)
        effect_size_threshold: Threshold for practical importance (default: 0.5)
        logger: Logger instance
    """
    
    def __init__(
        self,
        n_bootstrap: int = 1000,
        significance_threshold: float = 0.05,
        effect_size_threshold: float = 0.5,
        anomaly_threshold: float = 3.0
    ):
        """Initialize two-dimensional validator.
        
        Args:
            n_bootstrap: Number of bootstrap samples for CI calculation
            significance_threshold: P-value threshold for significance (default: 0.05)
            effect_size_threshold: Cohen's d threshold for practical importance (default: 0.5)
            anomaly_threshold: Threshold for anomaly detection in sigma
        """
        self.validation_framework = ValidationFramework(
            n_bootstrap=n_bootstrap,
            alpha=significance_threshold,
            anomaly_threshold=anomaly_threshold
        )
        self.comparative_analyzer = ComparativeAnalyzer(
            anomaly_threshold=anomaly_threshold
        )
        self.significance_threshold = significance_threshold
        self.effect_size_threshold = effect_size_threshold
        self.logger = get_logger(__name__)
        self.logger.info(
            f"Initialized TwoDimensionalValidator with "
            f"significance_threshold={significance_threshold}, "
            f"effect_size_threshold={effect_size_threshold}"
        )
    
    def validate_hypothesis_2d(
        self,
        hypothesis_id: str,
        predicted_exponent: float,
        measured_exponent: float,
        measured_error: float,
        variant_results: Optional[Dict[str, List[VAEAnalysisResults]]] = None,
        exponent_name: str = 'beta',
        predicted_error: Optional[float] = None
    ) -> TwoDValidationResult:
        """Perform two-dimensional validation of a hypothesis.
        
        Combines bootstrap CI validation (individual prediction) with ANOVA
        (comparative analysis) to create a 2D validation space.
        
        Args:
            hypothesis_id: ID of the hypothesis
            predicted_exponent: Predicted exponent value
            measured_exponent: Measured exponent value
            measured_error: Standard error of measured value
            variant_results: Optional dict of results from multiple variants for ANOVA
            exponent_name: Name of the exponent being validated
            predicted_error: Optional error in prediction
            
        Returns:
            TwoDValidationResult with classification and recommendations
        """
        self.logger.info(
            f"Performing 2D validation for hypothesis '{hypothesis_id}' "
            f"on exponent '{exponent_name}'"
        )
        
        # Step 1: Bootstrap CI validation (individual prediction)
        bootstrap_result = self.validation_framework.validate_exponent_prediction(
            hypothesis_id=hypothesis_id,
            predicted=predicted_exponent,
            measured=measured_exponent,
            measured_error=measured_error,
            predicted_error=predicted_error
        )
        
        bootstrap_ci = bootstrap_result.bootstrap_intervals['exponent']
        bootstrap_p_value = bootstrap_result.p_values['exponent']
        bootstrap_effect_size = bootstrap_result.effect_sizes['cohens_d']
        
        # Step 2: ANOVA validation (comparative analysis)
        if variant_results is not None and len(variant_results) > 1:
            # Perform ANOVA across variants
            anova_result = self._perform_anova_analysis(
                variant_results, exponent_name
            )
            anova_p_value = anova_result['p_value']
            anova_effect_size = anova_result['effect_size']
        else:
            # No comparative data available
            anova_result = {
                'p_value': None,
                'effect_size': None,
                'message': 'No comparative data available for ANOVA'
            }
            anova_p_value = None
            anova_effect_size = None
        
        # Step 3: Combine results into 2D validation space
        # Use bootstrap for individual validation, ANOVA for comparative
        if anova_p_value is not None:
            # Both bootstrap and ANOVA available
            # Statistical significance: combine p-values (Fisher's method)
            combined_p_value = self._combine_p_values(
                bootstrap_p_value, anova_p_value
            )
            # Effect size: use maximum (most conservative)
            combined_effect_size = max(bootstrap_effect_size, anova_effect_size)
        else:
            # Only bootstrap available
            combined_p_value = bootstrap_p_value
            combined_effect_size = bootstrap_effect_size
        
        # Step 4: Classify into 2D validation region
        region, confidence, recommendation = self._classify_2d_region(
            p_value=combined_p_value,
            effect_size=combined_effect_size,
            bootstrap_validated=bootstrap_result.validated
        )
        
        # Create result
        result = TwoDValidationResult(
            hypothesis_id=hypothesis_id,
            region=region,
            statistical_significance=1.0 - combined_p_value,  # Convert to significance score
            effect_size=combined_effect_size,
            bootstrap_ci=bootstrap_ci,
            anova_result=anova_result,
            confidence=confidence,
            recommendation=recommendation,
            details={
                'bootstrap_result': {
                    'validated': bootstrap_result.validated,
                    'p_value': bootstrap_p_value,
                    'effect_size': bootstrap_effect_size,
                    'ci': bootstrap_ci
                },
                'anova_result': anova_result,
                'combined_p_value': combined_p_value,
                'exponent_name': exponent_name,
                'predicted': predicted_exponent,
                'measured': measured_exponent,
                'measured_error': measured_error
            }
        )
        
        self.logger.info(
            f"2D validation result: {region} "
            f"(significance: {result.statistical_significance:.3f}, "
            f"effect_size: {result.effect_size:.3f}, "
            f"confidence: {confidence:.2%})"
        )
        
        return result
    
    def validate_multiple_exponents_2d(
        self,
        hypothesis_id: str,
        predicted_exponents: Dict[str, float],
        measured_exponents: Dict[str, float],
        measured_errors: Dict[str, float],
        variant_results: Optional[Dict[str, List[VAEAnalysisResults]]] = None,
        predicted_errors: Optional[Dict[str, float]] = None
    ) -> Dict[str, TwoDValidationResult]:
        """Validate multiple exponents using 2D validation.
        
        Args:
            hypothesis_id: ID of the hypothesis
            predicted_exponents: Dictionary of predicted exponent values
            measured_exponents: Dictionary of measured exponent values
            measured_errors: Dictionary of measurement errors
            variant_results: Optional dict of results from multiple variants
            predicted_errors: Optional dictionary of prediction errors
            
        Returns:
            Dictionary mapping exponent names to validation results
        """
        self.logger.info(
            f"Performing 2D validation for {len(predicted_exponents)} exponents"
        )
        
        results = {}
        
        for exp_name in predicted_exponents.keys():
            if exp_name not in measured_exponents:
                self.logger.warning(
                    f"Exponent '{exp_name}' not in measured exponents, skipping"
                )
                continue
            
            predicted = predicted_exponents[exp_name]
            measured = measured_exponents[exp_name]
            error = measured_errors.get(exp_name, 0.05)
            pred_error = predicted_errors.get(exp_name) if predicted_errors else None
            
            result = self.validate_hypothesis_2d(
                hypothesis_id=f"{hypothesis_id}_{exp_name}",
                predicted_exponent=predicted,
                measured_exponent=measured,
                measured_error=error,
                variant_results=variant_results,
                exponent_name=exp_name,
                predicted_error=pred_error
            )
            
            results[exp_name] = result
        
        return results
    
    def generate_2d_validation_plot(
        self,
        validation_results: Dict[str, TwoDValidationResult],
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """Generate 2D validation space visualization.
        
        Creates a scatter plot with:
        - X-axis: Statistical significance (1 - p_value)
        - Y-axis: Effect size (Cohen's d)
        - Color-coded regions: VALIDATED, REFUTED, INCONCLUSIVE, FALSE_POSITIVE
        - Points labeled with exponent names
        
        Args:
            validation_results: Dictionary of validation results
            output_path: Optional path to save figure
            title: Optional custom title
            
        Returns:
            matplotlib Figure
        """
        self.logger.info(
            f"Generating 2D validation plot for {len(validation_results)} results"
        )
        
        # Extract data
        exponent_names = []
        significance_scores = []
        effect_sizes = []
        regions = []
        
        for exp_name, result in validation_results.items():
            exponent_names.append(exp_name)
            significance_scores.append(result.statistical_significance)
            effect_sizes.append(result.effect_size)
            regions.append(result.region)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define region boundaries
        sig_threshold = 1.0 - self.significance_threshold
        effect_threshold = self.effect_size_threshold
        
        # Draw region boundaries
        ax.axhline(y=effect_threshold, color='gray', linestyle='--', linewidth=1.5,
                  alpha=0.5, label='Effect size threshold')
        ax.axvline(x=sig_threshold, color='gray', linestyle='--', linewidth=1.5,
                  alpha=0.5, label='Significance threshold')
        
        # Color regions
        # VALIDATED: high significance + high effect
        ax.fill_between([sig_threshold, 1.0], effect_threshold, 5.0,
                       color='green', alpha=0.1, label='VALIDATED')
        
        # REFUTED: high significance + low effect
        ax.fill_between([sig_threshold, 1.0], 0, effect_threshold,
                       color='red', alpha=0.1, label='REFUTED')
        
        # INCONCLUSIVE: low significance + high effect
        ax.fill_between([0, sig_threshold], effect_threshold, 5.0,
                       color='yellow', alpha=0.1, label='INCONCLUSIVE')
        
        # FALSE_POSITIVE: low significance + low effect
        ax.fill_between([0, sig_threshold], 0, effect_threshold,
                       color='gray', alpha=0.1, label='FALSE_POSITIVE')
        
        # Plot points
        region_colors = {
            'VALIDATED': 'green',
            'REFUTED': 'red',
            'INCONCLUSIVE': 'orange',
            'FALSE_POSITIVE': 'gray'
        }
        
        for i, exp_name in enumerate(exponent_names):
            color = region_colors.get(regions[i], 'blue')
            ax.scatter(
                significance_scores[i], effect_sizes[i],
                c=color, s=200, marker='o',
                edgecolors='black', linewidths=2,
                alpha=0.7, zorder=10
            )
            
            # Label point
            ax.annotate(
                exp_name,
                (significance_scores[i], effect_sizes[i]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
            )
        
        # Set labels and title
        ax.set_xlabel('Statistical Significance (1 - p-value)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Effect Size (Cohen\'s d)', fontsize=14, fontweight='bold')
        
        if title is None:
            title = '2D Validation Space: Statistical Significance vs Effect Size'
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Set axis limits
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.1, max(5.0, max(effect_sizes) * 1.2))
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
        
        # Add legend
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        
        # Save if path provided
        if output_path is not None:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved 2D validation plot to {output_path}")
        
        return fig
    
    def generate_validation_report_2d(
        self,
        validation_results: Dict[str, TwoDValidationResult]
    ) -> str:
        """Generate comprehensive 2D validation report.
        
        Args:
            validation_results: Dictionary of validation results
            
        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("TWO-DIMENSIONAL VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary statistics
        n_total = len(validation_results)
        region_counts = {}
        for result in validation_results.values():
            region = result.region
            region_counts[region] = region_counts.get(region, 0) + 1
        
        report_lines.append("SUMMARY:")
        report_lines.append(f"  Total Exponents: {n_total}")
        for region, count in sorted(region_counts.items()):
            percentage = count / n_total * 100
            report_lines.append(f"  {region}: {count} ({percentage:.1f}%)")
        report_lines.append("")
        
        # Individual results by region
        for region in ['VALIDATED', 'INCONCLUSIVE', 'REFUTED', 'FALSE_POSITIVE']:
            region_results = {
                name: result for name, result in validation_results.items()
                if result.region == region
            }
            
            if not region_results:
                continue
            
            report_lines.append(f"\n{region} FINDINGS:")
            report_lines.append("-" * 80)
            
            for exp_name, result in region_results.items():
                report_lines.append(f"\n{exp_name.upper()}:")
                report_lines.append(f"  Statistical Significance: {result.statistical_significance:.3f}")
                report_lines.append(f"  Effect Size: {result.effect_size:.3f}")
                report_lines.append(f"  Confidence: {result.confidence:.2%}")
                report_lines.append(f"  Bootstrap CI: [{result.bootstrap_ci[0]:.4f}, {result.bootstrap_ci[1]:.4f}]")
                
                if result.anova_result['p_value'] is not None:
                    report_lines.append(f"  ANOVA p-value: {result.anova_result['p_value']:.4e}")
                
                report_lines.append(f"  Recommendation: {result.recommendation}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append("INTERPRETATION GUIDE:")
        report_lines.append("-" * 80)
        report_lines.append("VALIDATED: High statistical significance + High effect size")
        report_lines.append("  → Finding is both statistically significant and practically important")
        report_lines.append("")
        report_lines.append("REFUTED: High statistical significance + Low effect size")
        report_lines.append("  → Statistically different but not practically important (false positive)")
        report_lines.append("")
        report_lines.append("INCONCLUSIVE: Low statistical significance + High effect size")
        report_lines.append("  → Potentially important but needs more data for statistical confidence")
        report_lines.append("")
        report_lines.append("FALSE_POSITIVE: Low statistical significance + Low effect size")
        report_lines.append("  → Likely a false positive, not worth further investigation")
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def _perform_anova_analysis(
        self,
        variant_results: Dict[str, List[VAEAnalysisResults]],
        exponent_name: str
    ) -> Dict[str, Any]:
        """Perform ANOVA analysis across variants for a specific exponent.
        
        Args:
            variant_results: Dictionary mapping variant IDs to results lists
            exponent_name: Name of the exponent to analyze
            
        Returns:
            Dictionary with ANOVA results
        """
        # Collect exponent values by variant
        variant_values = {}
        for variant_id, results_list in variant_results.items():
            values = []
            for result in results_list:
                if exponent_name in result.exponents:
                    values.append(result.exponents[exponent_name])
            
            if values:
                variant_values[variant_id] = values
        
        if len(variant_values) < 2:
            return {
                'p_value': None,
                'effect_size': None,
                'f_statistic': None,
                'message': 'Insufficient variants for ANOVA'
            }
        
        # Perform ANOVA
        groups = list(variant_values.values())
        f_statistic, p_value = stats.f_oneway(*groups)
        
        # Calculate effect size (eta-squared)
        # eta^2 = SS_between / SS_total
        all_values = np.concatenate(groups)
        grand_mean = np.mean(all_values)
        
        ss_between = sum(
            len(group) * (np.mean(group) - grand_mean)**2
            for group in groups
        )
        ss_total = np.sum((all_values - grand_mean)**2)
        
        eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
        
        # Convert eta-squared to Cohen's f for consistency
        # f = sqrt(eta^2 / (1 - eta^2))
        if eta_squared < 1.0:
            cohens_f = np.sqrt(eta_squared / (1.0 - eta_squared))
        else:
            cohens_f = 10.0  # Very large effect
        
        return {
            'p_value': float(p_value),
            'effect_size': float(cohens_f),
            'f_statistic': float(f_statistic),
            'eta_squared': float(eta_squared),
            'n_variants': len(variant_values),
            'message': f'ANOVA across {len(variant_values)} variants'
        }
    
    def _combine_p_values(self, p1: float, p2: float) -> float:
        """Combine two p-values using Fisher's method.
        
        Args:
            p1: First p-value
            p2: Second p-value
            
        Returns:
            Combined p-value
        """
        # Fisher's method: -2 * sum(log(p_i)) ~ chi-square(2k)
        # where k is the number of p-values
        chi_square_stat = -2 * (np.log(p1) + np.log(p2))
        combined_p = 1.0 - stats.chi2.cdf(chi_square_stat, df=4)
        
        return combined_p
    
    def _classify_2d_region(
        self,
        p_value: float,
        effect_size: float,
        bootstrap_validated: bool
    ) -> Tuple[str, float, str]:
        """Classify result into 2D validation region.
        
        Args:
            p_value: Combined p-value
            effect_size: Combined effect size
            bootstrap_validated: Whether bootstrap CI validation passed
            
        Returns:
            Tuple of (region, confidence, recommendation)
        """
        # Convert p-value to significance score
        significance = 1.0 - p_value
        
        # Classify based on thresholds
        is_significant = significance >= (1.0 - self.significance_threshold)
        is_large_effect = effect_size >= self.effect_size_threshold
        
        if is_significant and is_large_effect:
            # VALIDATED: Both statistically significant and practically important
            region = 'VALIDATED'
            confidence = min(0.95, significance * (effect_size / 2.0))
            recommendation = (
                "Finding is validated: statistically significant and practically important. "
                "Proceed with confidence."
            )
        
        elif is_significant and not is_large_effect:
            # REFUTED: Statistically significant but not practically important
            region = 'REFUTED'
            confidence = 0.7
            recommendation = (
                "Finding is refuted: statistically significant but effect size too small "
                "to be practically important. Likely a false positive."
            )
        
        elif not is_significant and is_large_effect:
            # INCONCLUSIVE: Large effect but not statistically significant
            region = 'INCONCLUSIVE'
            confidence = 0.5
            recommendation = (
                "Finding is inconclusive: large effect size suggests practical importance, "
                "but statistical significance is lacking. Collect more data."
            )
        
        else:
            # FALSE_POSITIVE: Neither significant nor large effect
            region = 'FALSE_POSITIVE'
            confidence = 0.3
            recommendation = (
                "Finding is likely a false positive: neither statistically significant "
                "nor practically important. Not worth further investigation."
            )
        
        # Adjust confidence based on bootstrap validation
        if bootstrap_validated and region in ['VALIDATED', 'INCONCLUSIVE']:
            confidence = min(0.95, confidence + 0.1)
        elif not bootstrap_validated and region in ['VALIDATED']:
            confidence = max(0.5, confidence - 0.2)
        
        return region, confidence, recommendation
