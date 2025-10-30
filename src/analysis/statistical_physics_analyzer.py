"""
Statistical Physics Analyzer

This module provides enhanced statistical validation and uncertainty quantification
for physics properties discovered in the Prometheus phase discovery system.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.optimize import curve_fit, minimize
from scipy.stats import bootstrap
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from .latent_analysis import LatentRepresentation
from .phase_detection import PhaseDetectionResult
from .enhanced_validation_types import (
    ConfidenceInterval, EnsembleAnalysisResult, HypothesisTestResults,
    StatisticalValidationError, ViolationSeverity, PhysicsViolation
)
from ..utils.logging_utils import get_logger


class StatisticalPhysicsAnalyzer:
    """
    Enhanced statistical validation and uncertainty quantification for physics properties.
    
    Provides bootstrap confidence intervals, ensemble analysis, hypothesis testing,
    and uncertainty quantification for critical exponents, phase boundaries, and
    other physics properties.
    """
    
    def __init__(self, 
                 confidence_level: float = 0.95,
                 n_bootstrap_default: int = 1000,
                 random_seed: Optional[int] = None,
                 n_jobs: int = -1):
        """
        Initialize statistical physics analyzer.
        
        Args:
            confidence_level: Default confidence level for intervals (0-1)
            n_bootstrap_default: Default number of bootstrap samples
            random_seed: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.confidence_level = confidence_level
        self.n_bootstrap_default = n_bootstrap_default
        self.random_seed = random_seed
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.logger = get_logger(__name__)
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.logger.info(f"Statistical analyzer initialized: confidence={confidence_level}, "
                        f"bootstrap_samples={n_bootstrap_default}, n_jobs={self.n_jobs}")
    
    def compute_bootstrap_confidence_intervals(self,
                                             data: np.ndarray,
                                             statistic_func: Callable,
                                             n_bootstrap: Optional[int] = None,
                                             confidence_level: Optional[float] = None,
                                             method: str = 'percentile') -> ConfidenceInterval:
        """
        Compute bootstrap confidence intervals for a given statistic.
        
        Args:
            data: Input data array
            statistic_func: Function to compute statistic from data
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (0-1)
            method: Bootstrap method ('percentile', 'bca', 'basic')
            
        Returns:
            ConfidenceInterval object with bounds and metadata
        """
        if n_bootstrap is None:
            n_bootstrap = self.n_bootstrap_default
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        self.logger.debug(f"Computing bootstrap CI: n_bootstrap={n_bootstrap}, "
                         f"confidence={confidence_level}, method={method}")
        
        try:
            # Ensure data is numpy array
            data = np.asarray(data)
            
            if len(data) == 0:
                raise StatisticalValidationError("Empty data array provided")
            
            # Compute original statistic
            original_stat = statistic_func(data)
            
            # Generate bootstrap samples
            bootstrap_stats = []
            rng = np.random.RandomState(self.random_seed)
            
            for _ in range(n_bootstrap):
                # Resample with replacement
                bootstrap_sample = rng.choice(data, size=len(data), replace=True)
                try:
                    bootstrap_stat = statistic_func(bootstrap_sample)
                    if np.isfinite(bootstrap_stat):
                        bootstrap_stats.append(bootstrap_stat)
                except Exception as e:
                    self.logger.warning(f"Bootstrap sample failed: {e}")
                    continue
            
            if len(bootstrap_stats) < n_bootstrap * 0.5:
                raise StatisticalValidationError(
                    f"Too many bootstrap failures: {len(bootstrap_stats)}/{n_bootstrap}"
                )
            
            bootstrap_stats = np.array(bootstrap_stats)
            
            # Compute confidence interval based on method
            alpha = 1 - confidence_level
            
            if method == 'percentile':
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                lower_bound = np.percentile(bootstrap_stats, lower_percentile)
                upper_bound = np.percentile(bootstrap_stats, upper_percentile)
                bias_correction = None
                
            elif method == 'bca':
                # Bias-corrected and accelerated (BCa) method
                lower_bound, upper_bound, bias_correction = self._compute_bca_interval(
                    data, statistic_func, bootstrap_stats, original_stat, confidence_level
                )
                
            elif method == 'basic':
                # Basic bootstrap method
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                bootstrap_lower = np.percentile(bootstrap_stats, lower_percentile)
                bootstrap_upper = np.percentile(bootstrap_stats, upper_percentile)
                lower_bound = 2 * original_stat - bootstrap_upper
                upper_bound = 2 * original_stat - bootstrap_lower
                bias_correction = None
                
            else:
                raise ValueError(f"Unknown bootstrap method: {method}")
            
            # Create confidence interval object
            ci = ConfidenceInterval(
                lower_bound=float(lower_bound),
                upper_bound=float(upper_bound),
                confidence_level=confidence_level,
                method=f"bootstrap_{method}",
                n_bootstrap_samples=len(bootstrap_stats),
                bias_correction=bias_correction
            )
            
            self.logger.debug(f"Bootstrap CI computed: [{lower_bound:.4f}, {upper_bound:.4f}]")
            
            return ci
            
        except Exception as e:
            self.logger.error(f"Bootstrap confidence interval computation failed: {e}")
            raise StatisticalValidationError(f"Bootstrap CI computation failed: {e}")
    
    def _compute_bca_interval(self,
                             data: np.ndarray,
                             statistic_func: Callable,
                             bootstrap_stats: np.ndarray,
                             original_stat: float,
                             confidence_level: float) -> Tuple[float, float, float]:
        """
        Compute bias-corrected and accelerated (BCa) confidence interval.
        
        Args:
            data: Original data
            statistic_func: Statistic function
            bootstrap_stats: Bootstrap statistics
            original_stat: Original statistic value
            confidence_level: Confidence level
            
        Returns:
            Tuple of (lower_bound, upper_bound, bias_correction)
        """
        n = len(data)
        alpha = 1 - confidence_level
        
        # Bias correction
        n_less = np.sum(bootstrap_stats < original_stat)
        bias_correction = stats.norm.ppf(n_less / len(bootstrap_stats))
        
        # Acceleration correction using jackknife
        jackknife_stats = []
        for i in range(n):
            jackknife_sample = np.concatenate([data[:i], data[i+1:]])
            try:
                jackknife_stat = statistic_func(jackknife_sample)
                jackknife_stats.append(jackknife_stat)
            except:
                jackknife_stats.append(np.nan)
        
        jackknife_stats = np.array(jackknife_stats)
        valid_jackknife = jackknife_stats[np.isfinite(jackknife_stats)]
        
        if len(valid_jackknife) < n * 0.5:
            # Fall back to percentile method if jackknife fails
            self.logger.warning("Jackknife failed, falling back to percentile method")
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            lower_bound = np.percentile(bootstrap_stats, lower_percentile)
            upper_bound = np.percentile(bootstrap_stats, upper_percentile)
            return lower_bound, upper_bound, bias_correction
        
        jackknife_mean = np.mean(valid_jackknife)
        acceleration = np.sum((jackknife_mean - valid_jackknife)**3) / (
            6 * (np.sum((jackknife_mean - valid_jackknife)**2))**1.5
        )
        
        # Adjusted percentiles
        z_alpha_2 = stats.norm.ppf(alpha / 2)
        z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)
        
        alpha_1 = stats.norm.cdf(bias_correction + (bias_correction + z_alpha_2) / 
                                (1 - acceleration * (bias_correction + z_alpha_2)))
        alpha_2 = stats.norm.cdf(bias_correction + (bias_correction + z_1_alpha_2) / 
                                (1 - acceleration * (bias_correction + z_1_alpha_2)))
        
        # Ensure percentiles are valid
        alpha_1 = np.clip(alpha_1, 0.001, 0.999)
        alpha_2 = np.clip(alpha_2, 0.001, 0.999)
        
        lower_bound = np.percentile(bootstrap_stats, alpha_1 * 100)
        upper_bound = np.percentile(bootstrap_stats, alpha_2 * 100)
        
        return lower_bound, upper_bound, bias_correction
    
    def perform_ensemble_analysis(self,
                                ensemble_data: List[LatentRepresentation],
                                properties: Optional[List[str]] = None) -> EnsembleAnalysisResult:
        """
        Perform ensemble analysis across multiple simulation runs.
        
        Args:
            ensemble_data: List of latent representations from different runs
            properties: List of properties to analyze (default: all available)
            
        Returns:
            EnsembleAnalysisResult with statistics across ensemble
        """
        if not ensemble_data:
            raise StatisticalValidationError("Empty ensemble data provided")
        
        n_ensemble = len(ensemble_data)
        self.logger.info(f"Performing ensemble analysis on {n_ensemble} runs")
        
        # Default properties to analyze
        if properties is None:
            properties = ['z1_mean', 'z2_mean', 'z1_std', 'z2_std', 'magnetization_mean']
        
        try:
            # Extract properties from each ensemble member
            ensemble_properties = {prop: [] for prop in properties}
            
            for i, latent_repr in enumerate(ensemble_data):
                try:
                    # Extract standard properties
                    if 'z1_mean' in properties:
                        ensemble_properties['z1_mean'].append(np.mean(latent_repr.z1))
                    if 'z2_mean' in properties:
                        ensemble_properties['z2_mean'].append(np.mean(latent_repr.z2))
                    if 'z1_std' in properties:
                        ensemble_properties['z1_std'].append(np.std(latent_repr.z1))
                    if 'z2_std' in properties:
                        ensemble_properties['z2_std'].append(np.std(latent_repr.z2))
                    if 'magnetization_mean' in properties:
                        ensemble_properties['magnetization_mean'].append(
                            np.mean(np.abs(latent_repr.magnetizations))
                        )
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract properties from ensemble member {i}: {e}")
                    continue
            
            # Compute ensemble statistics
            ensemble_mean = {}
            ensemble_std = {}
            ensemble_confidence_intervals = {}
            
            for prop in properties:
                if prop in ensemble_properties and ensemble_properties[prop]:
                    prop_values = np.array(ensemble_properties[prop])
                    
                    # Basic statistics
                    ensemble_mean[prop] = float(np.mean(prop_values))
                    ensemble_std[prop] = float(np.std(prop_values, ddof=1))
                    
                    # Confidence interval for the mean
                    if len(prop_values) > 1:
                        sem = stats.sem(prop_values)
                        ci_width = stats.t.ppf((1 + self.confidence_level) / 2, 
                                             len(prop_values) - 1) * sem
                        
                        ensemble_confidence_intervals[prop] = ConfidenceInterval(
                            lower_bound=ensemble_mean[prop] - ci_width,
                            upper_bound=ensemble_mean[prop] + ci_width,
                            confidence_level=self.confidence_level,
                            method="t_distribution"
                        )
            
            # Inter-run correlations
            inter_run_correlations = self._compute_inter_run_correlations(ensemble_properties)
            
            # Convergence analysis
            convergence_analysis = self._analyze_ensemble_convergence(ensemble_properties)
            
            # Outlier detection
            outlier_detection = self._detect_ensemble_outliers(ensemble_properties)
            
            result = EnsembleAnalysisResult(
                n_ensemble_members=n_ensemble,
                ensemble_mean=ensemble_mean,
                ensemble_std=ensemble_std,
                ensemble_confidence_intervals=ensemble_confidence_intervals,
                inter_run_correlations=inter_run_correlations,
                convergence_analysis=convergence_analysis,
                outlier_detection=outlier_detection
            )
            
            self.logger.info(f"Ensemble analysis completed for {len(properties)} properties")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ensemble analysis failed: {e}")
            raise StatisticalValidationError(f"Ensemble analysis failed: {e}")
    
    def _compute_inter_run_correlations(self, 
                                      ensemble_properties: Dict[str, List[float]]) -> Dict[str, float]:
        """Compute correlations between different properties across runs."""
        correlations = {}
        
        properties = list(ensemble_properties.keys())
        for i, prop1 in enumerate(properties):
            for prop2 in properties[i+1:]:
                if (ensemble_properties[prop1] and ensemble_properties[prop2] and
                    len(ensemble_properties[prop1]) == len(ensemble_properties[prop2])):
                    
                    try:
                        corr_coeff = np.corrcoef(ensemble_properties[prop1], 
                                               ensemble_properties[prop2])[0, 1]
                        if np.isfinite(corr_coeff):
                            correlations[f"{prop1}_vs_{prop2}"] = float(corr_coeff)
                    except:
                        continue
        
        return correlations
    
    def _analyze_ensemble_convergence(self, 
                                    ensemble_properties: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze convergence of ensemble properties."""
        convergence_analysis = {}
        
        for prop, values in ensemble_properties.items():
            if len(values) < 3:
                continue
                
            values = np.array(values)
            n_runs = len(values)
            
            # Running mean convergence
            running_means = np.cumsum(values) / np.arange(1, n_runs + 1)
            
            # Convergence metric: relative change in last 25% of runs
            if n_runs >= 4:
                split_point = max(2, n_runs // 4)
                early_mean = np.mean(running_means[:split_point])
                late_mean = np.mean(running_means[-split_point:])
                
                if early_mean != 0:
                    convergence_metric = abs(late_mean - early_mean) / abs(early_mean)
                else:
                    convergence_metric = abs(late_mean - early_mean)
                
                convergence_analysis[prop] = {
                    'running_means': running_means.tolist(),
                    'convergence_metric': float(convergence_metric),
                    'is_converged': convergence_metric < 0.05  # 5% threshold
                }
        
        return convergence_analysis
    
    def _detect_ensemble_outliers(self, 
                                ensemble_properties: Dict[str, List[float]]) -> Dict[str, List[int]]:
        """Detect outliers in ensemble using modified Z-score."""
        outlier_detection = {}
        
        for prop, values in ensemble_properties.items():
            if len(values) < 3:
                continue
                
            values = np.array(values)
            
            # Modified Z-score using median absolute deviation
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            
            if mad == 0:
                # Use standard deviation if MAD is zero
                modified_z_scores = np.abs(values - np.mean(values)) / np.std(values)
            else:
                modified_z_scores = 0.6745 * (values - median) / mad
            
            # Outliers are points with |modified Z-score| > 3.5
            outlier_indices = np.where(np.abs(modified_z_scores) > 3.5)[0].tolist()
            
            if outlier_indices:
                outlier_detection[prop] = outlier_indices
        
        return outlier_detection
    
    def test_physics_hypotheses(self,
                               observed_values: Dict[str, float],
                               theoretical_values: Dict[str, float],
                               uncertainties: Optional[Dict[str, float]] = None) -> List[HypothesisTestResults]:
        """
        Test physics hypotheses using statistical methods.
        
        Args:
            observed_values: Dictionary of observed physics property values
            theoretical_values: Dictionary of theoretical/expected values
            uncertainties: Optional uncertainties for observed values
            
        Returns:
            List of HypothesisTestResults for each tested property
        """
        self.logger.info(f"Testing physics hypotheses for {len(observed_values)} properties")
        
        results = []
        
        for property_name in observed_values.keys():
            if property_name not in theoretical_values:
                self.logger.warning(f"No theoretical value for property: {property_name}")
                continue
            
            observed = observed_values[property_name]
            theoretical = theoretical_values[property_name]
            uncertainty = uncertainties.get(property_name) if uncertainties else None
            
            try:
                # Perform appropriate statistical test
                if uncertainty is not None and uncertainty > 0:
                    # Z-test when uncertainty is known
                    test_result = self._perform_z_test(
                        observed, theoretical, uncertainty, property_name
                    )
                else:
                    # One-sample t-test (assuming we have sample data)
                    # For single values, we'll use a simple comparison
                    test_result = self._perform_simple_comparison_test(
                        observed, theoretical, property_name
                    )
                
                results.append(test_result)
                
            except Exception as e:
                self.logger.error(f"Hypothesis test failed for {property_name}: {e}")
                continue
        
        self.logger.info(f"Completed {len(results)} hypothesis tests")
        return results
    
    def _perform_z_test(self,
                       observed: float,
                       theoretical: float,
                       uncertainty: float,
                       property_name: str) -> HypothesisTestResults:
        """Perform Z-test for hypothesis testing with known uncertainty."""
        
        # Z-statistic
        z_statistic = (observed - theoretical) / uncertainty
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
        
        # Critical value for 95% confidence
        critical_value = stats.norm.ppf(1 - 0.05/2)
        
        # Effect size (Cohen's d equivalent)
        effect_size = abs(observed - theoretical) / uncertainty
        
        return HypothesisTestResults(
            test_name="z_test",
            null_hypothesis=f"{property_name} equals theoretical value {theoretical}",
            alternative_hypothesis=f"{property_name} differs from theoretical value",
            test_statistic=float(z_statistic),
            p_value=float(p_value),
            critical_value=float(critical_value),
            reject_null=abs(z_statistic) > critical_value,
            confidence_level=0.05,
            effect_size=float(effect_size)
        )
    
    def _perform_simple_comparison_test(self,
                                      observed: float,
                                      theoretical: float,
                                      property_name: str) -> HypothesisTestResults:
        """Perform simple comparison test for single values."""
        
        # Simple relative difference as test statistic
        if theoretical != 0:
            relative_diff = abs(observed - theoretical) / abs(theoretical)
            test_statistic = relative_diff
        else:
            test_statistic = abs(observed - theoretical)
        
        # Heuristic p-value based on relative difference
        # This is a simplified approach for demonstration
        if relative_diff < 0.01:  # 1% difference
            p_value = 0.9
        elif relative_diff < 0.05:  # 5% difference
            p_value = 0.1
        elif relative_diff < 0.1:  # 10% difference
            p_value = 0.05
        else:
            p_value = 0.01
        
        critical_value = 0.05  # 5% relative difference threshold
        
        return HypothesisTestResults(
            test_name="relative_difference_test",
            null_hypothesis=f"{property_name} equals theoretical value {theoretical}",
            alternative_hypothesis=f"{property_name} differs significantly from theoretical value",
            test_statistic=float(test_statistic),
            p_value=float(p_value),
            critical_value=float(critical_value),
            reject_null=test_statistic > critical_value,
            confidence_level=0.05,
            effect_size=float(test_statistic)
        )
    
    def estimate_phase_boundary_uncertainty(self,
                                          temperatures: np.ndarray,
                                          order_parameter: np.ndarray,
                                          n_bootstrap: Optional[int] = None) -> Dict[str, Any]:
        """
        Estimate uncertainty in phase boundary location.
        
        Args:
            temperatures: Temperature array
            order_parameter: Order parameter values
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with phase boundary uncertainty estimates
        """
        if n_bootstrap is None:
            n_bootstrap = self.n_bootstrap_default
        
        self.logger.info("Estimating phase boundary uncertainty")
        
        try:
            # Define function to find critical temperature
            def find_critical_temp(temps, order_param):
                # Find temperature where order parameter drops most rapidly
                if len(order_param) < 3:
                    return np.nan
                
                # Compute derivative (finite differences)
                dorder_dt = np.gradient(order_param, temps)
                
                # Find minimum (most negative) derivative
                critical_idx = np.argmin(dorder_dt)
                return temps[critical_idx]
            
            # Bootstrap confidence interval for critical temperature
            data_pairs = np.column_stack([temperatures, order_parameter])
            
            def bootstrap_critical_temp(data_sample):
                # Sort by temperature to maintain order
                sorted_data = data_sample[np.argsort(data_sample[:, 0])]
                return find_critical_temp(sorted_data[:, 0], sorted_data[:, 1])
            
            tc_ci = self.compute_bootstrap_confidence_intervals(
                data_pairs, bootstrap_critical_temp, n_bootstrap
            )
            
            # Original critical temperature estimate
            original_tc = find_critical_temp(temperatures, order_parameter)
            
            # Transition width estimation
            def estimate_transition_width(temps, order_param):
                # Find 10% and 90% points of order parameter range
                op_min, op_max = np.min(order_param), np.max(order_param)
                op_range = op_max - op_min
                
                if op_range == 0:
                    return np.nan
                
                # Find temperatures at 10% and 90% of range
                target_10 = op_min + 0.1 * op_range
                target_90 = op_min + 0.9 * op_range
                
                # Interpolate to find corresponding temperatures
                try:
                    temp_10 = np.interp(target_10, order_param, temps)
                    temp_90 = np.interp(target_90, order_param, temps)
                    return abs(temp_90 - temp_10)
                except:
                    return np.nan
            
            def bootstrap_transition_width(data_sample):
                sorted_data = data_sample[np.argsort(data_sample[:, 0])]
                return estimate_transition_width(sorted_data[:, 0], sorted_data[:, 1])
            
            width_ci = self.compute_bootstrap_confidence_intervals(
                data_pairs, bootstrap_transition_width, n_bootstrap
            )
            
            result = {
                'critical_temperature': float(original_tc) if np.isfinite(original_tc) else None,
                'critical_temperature_ci': tc_ci,
                'transition_width': float(estimate_transition_width(temperatures, order_parameter)),
                'transition_width_ci': width_ci,
                'boundary_uncertainty_region': {
                    'lower_bound': tc_ci.lower_bound,
                    'upper_bound': tc_ci.upper_bound,
                    'width': tc_ci.width
                }
            }
            
            self.logger.info(f"Phase boundary uncertainty estimated: "
                           f"T_c = {original_tc:.3f} [{tc_ci.lower_bound:.3f}, {tc_ci.upper_bound:.3f}]")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Phase boundary uncertainty estimation failed: {e}")
            raise StatisticalValidationError(f"Phase boundary uncertainty estimation failed: {e}")
    
    def compute_statistical_significance(self,
                                       observed_property: float,
                                       theoretical_property: float,
                                       uncertainty: float,
                                       test_type: str = 'two_tailed') -> Dict[str, float]:
        """
        Compute statistical significance for physics property validation.
        
        Args:
            observed_property: Observed value of physics property
            theoretical_property: Theoretical/expected value
            uncertainty: Uncertainty in observed value
            test_type: Type of test ('two_tailed', 'one_tailed')
            
        Returns:
            Dictionary with statistical significance metrics
        """
        try:
            # Z-score calculation
            if uncertainty <= 0:
                raise StatisticalValidationError("Uncertainty must be positive")
            
            z_score = (observed_property - theoretical_property) / uncertainty
            
            # P-value calculation
            if test_type == 'two_tailed':
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            elif test_type == 'one_tailed':
                p_value = 1 - stats.norm.cdf(abs(z_score))
            else:
                raise ValueError(f"Unknown test type: {test_type}")
            
            # Significance levels
            significance_levels = {
                'p_0_05': p_value < 0.05,
                'p_0_01': p_value < 0.01,
                'p_0_001': p_value < 0.001
            }
            
            # Effect size (standardized difference)
            effect_size = abs(z_score)
            
            # Confidence interval for the difference
            alpha = 0.05
            margin_of_error = stats.norm.ppf(1 - alpha/2) * uncertainty
            difference = observed_property - theoretical_property
            
            result = {
                'z_score': float(z_score),
                'p_value': float(p_value),
                'effect_size': float(effect_size),
                'significance_levels': significance_levels,
                'difference': float(difference),
                'difference_ci_lower': float(difference - margin_of_error),
                'difference_ci_upper': float(difference + margin_of_error),
                'is_significant_05': significance_levels['p_0_05'],
                'is_significant_01': significance_levels['p_0_01']
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Statistical significance computation failed: {e}")
            raise StatisticalValidationError(f"Statistical significance computation failed: {e}")
    
    def validate_critical_exponent_statistics(self,
                                            critical_exponents: Dict[str, float],
                                            theoretical_exponents: Dict[str, float],
                                            uncertainties: Optional[Dict[str, float]] = None,
                                            n_bootstrap: Optional[int] = None) -> Dict[str, Any]:
        """
        Validate critical exponents with comprehensive statistical analysis.
        
        Args:
            critical_exponents: Computed critical exponents
            theoretical_exponents: Theoretical values
            uncertainties: Uncertainties in computed values
            n_bootstrap: Number of bootstrap samples for CI computation
            
        Returns:
            Dictionary with comprehensive statistical validation results
        """
        if n_bootstrap is None:
            n_bootstrap = self.n_bootstrap_default
        
        self.logger.info("Performing statistical validation of critical exponents")
        
        validation_results = {}
        
        for exponent_name in critical_exponents.keys():
            if exponent_name not in theoretical_exponents:
                continue
            
            observed = critical_exponents[exponent_name]
            theoretical = theoretical_exponents[exponent_name]
            uncertainty = uncertainties.get(exponent_name) if uncertainties else None
            
            try:
                # Basic comparison
                absolute_error = abs(observed - theoretical)
                relative_error = absolute_error / abs(theoretical) if theoretical != 0 else absolute_error
                
                # Statistical significance if uncertainty is available
                significance_results = None
                if uncertainty is not None and uncertainty > 0:
                    significance_results = self.compute_statistical_significance(
                        observed, theoretical, uncertainty
                    )
                
                # Hypothesis test
                hypothesis_test = self._perform_z_test(
                    observed, theoretical, uncertainty or 0.1, exponent_name
                ) if uncertainty else self._perform_simple_comparison_test(
                    observed, theoretical, exponent_name
                )
                
                validation_results[exponent_name] = {
                    'observed_value': float(observed),
                    'theoretical_value': float(theoretical),
                    'absolute_error': float(absolute_error),
                    'relative_error': float(relative_error),
                    'uncertainty': float(uncertainty) if uncertainty else None,
                    'statistical_significance': significance_results,
                    'hypothesis_test': hypothesis_test,
                    'is_consistent': (significance_results['is_significant_05'] == False 
                                    if significance_results else relative_error < 0.1)
                }
                
            except Exception as e:
                self.logger.error(f"Validation failed for exponent {exponent_name}: {e}")
                continue
        
        # Overall consistency assessment
        consistent_exponents = sum(1 for result in validation_results.values() 
                                 if result['is_consistent'])
        total_exponents = len(validation_results)
        consistency_fraction = consistent_exponents / total_exponents if total_exponents > 0 else 0
        
        overall_results = {
            'individual_exponents': validation_results,
            'overall_consistency': {
                'consistent_exponents': consistent_exponents,
                'total_exponents': total_exponents,
                'consistency_fraction': float(consistency_fraction),
                'is_overall_consistent': consistency_fraction >= 0.75  # 75% threshold
            }
        }
        
        self.logger.info(f"Critical exponent validation completed: "
                        f"{consistent_exponents}/{total_exponents} consistent")
        
        return overall_results
    
    def generate_physics_violation_from_statistics(self,
                                                 property_name: str,
                                                 statistical_results: Dict[str, Any],
                                                 severity_threshold: float = 0.01) -> Optional[PhysicsViolation]:
        """
        Generate physics violation based on statistical analysis results.
        
        Args:
            property_name: Name of the physics property
            statistical_results: Results from statistical analysis
            severity_threshold: P-value threshold for determining severity
            
        Returns:
            PhysicsViolation object if violation is detected, None otherwise
        """
        try:
            # Check if there's a significant deviation
            if 'hypothesis_test' in statistical_results:
                hypothesis_test = statistical_results['hypothesis_test']
                
                if hypothesis_test.reject_null:
                    # Determine severity based on p-value and effect size
                    p_value = hypothesis_test.p_value
                    effect_size = hypothesis_test.effect_size
                    
                    if p_value < 0.001 or effect_size > 3:
                        severity = ViolationSeverity.CRITICAL
                    elif p_value < 0.01 or effect_size > 2:
                        severity = ViolationSeverity.HIGH
                    elif p_value < 0.05 or effect_size > 1:
                        severity = ViolationSeverity.MEDIUM
                    else:
                        severity = ViolationSeverity.LOW
                    
                    # Generate violation description
                    observed = statistical_results.get('observed_value', 'unknown')
                    theoretical = statistical_results.get('theoretical_value', 'unknown')
                    
                    description = (f"Statistical analysis indicates significant deviation in {property_name}: "
                                 f"observed = {observed}, theoretical = {theoretical}, "
                                 f"p-value = {p_value:.2e}")
                    
                    physics_explanation = (f"The {property_name} shows statistically significant "
                                         f"deviation from theoretical predictions, which may indicate "
                                         f"systematic errors, finite-size effects, or limitations in "
                                         f"the theoretical model.")
                    
                    suggested_investigation = (f"1. Check for systematic errors in {property_name} computation\n"
                                             f"2. Verify finite-size scaling behavior\n"
                                             f"3. Consider alternative theoretical models\n"
                                             f"4. Increase statistical sampling if uncertainty is large")
                    
                    return PhysicsViolation(
                        violation_type=f"statistical_deviation_{property_name}",
                        severity=severity,
                        description=description,
                        suggested_investigation=suggested_investigation,
                        physics_explanation=physics_explanation,
                        quantitative_measure=float(p_value),
                        threshold_value=float(severity_threshold),
                        confidence_level=float(hypothesis_test.confidence_level)
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to generate physics violation for {property_name}: {e}")
            return None


class UncertaintyQuantifier:
    """
    Specialized class for uncertainty quantification in physics properties.
    
    Provides methods for propagating uncertainties, estimating systematic errors,
    and quantifying model uncertainties in physics calculations.
    """
    
    def __init__(self, statistical_analyzer: StatisticalPhysicsAnalyzer):
        """
        Initialize uncertainty quantifier.
        
        Args:
            statistical_analyzer: Parent statistical analyzer instance
        """
        self.statistical_analyzer = statistical_analyzer
        self.logger = get_logger(__name__)
    
    def propagate_uncertainty(self,
                            function: Callable,
                            variables: Dict[str, float],
                            uncertainties: Dict[str, float],
                            method: str = 'monte_carlo',
                            n_samples: int = 10000) -> Dict[str, float]:
        """
        Propagate uncertainties through a function using Monte Carlo or analytical methods.
        
        Args:
            function: Function to propagate uncertainty through
            variables: Dictionary of variable names and values
            uncertainties: Dictionary of variable names and uncertainties
            method: Propagation method ('monte_carlo', 'linear', 'quadratic')
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary with propagated uncertainty statistics
        """
        self.logger.debug(f"Propagating uncertainty using {method} method")
        
        try:
            if method == 'monte_carlo':
                return self._monte_carlo_propagation(function, variables, uncertainties, n_samples)
            elif method == 'linear':
                return self._linear_propagation(function, variables, uncertainties)
            elif method == 'quadratic':
                return self._quadratic_propagation(function, variables, uncertainties)
            else:
                raise ValueError(f"Unknown propagation method: {method}")
                
        except Exception as e:
            self.logger.error(f"Uncertainty propagation failed: {e}")
            raise StatisticalValidationError(f"Uncertainty propagation failed: {e}")
    
    def _monte_carlo_propagation(self,
                               function: Callable,
                               variables: Dict[str, float],
                               uncertainties: Dict[str, float],
                               n_samples: int) -> Dict[str, float]:
        """Monte Carlo uncertainty propagation."""
        
        # Generate random samples for each variable
        samples = {}
        for var_name, var_value in variables.items():
            if var_name in uncertainties:
                uncertainty = uncertainties[var_name]
                # Assume Gaussian distribution
                samples[var_name] = np.random.normal(var_value, uncertainty, n_samples)
            else:
                # No uncertainty - use constant value
                samples[var_name] = np.full(n_samples, var_value)
        
        # Evaluate function for all sample combinations
        results = []
        for i in range(n_samples):
            sample_vars = {name: samples[name][i] for name in variables.keys()}
            try:
                result = function(**sample_vars)
                if np.isfinite(result):
                    results.append(result)
            except:
                continue
        
        if len(results) < n_samples * 0.1:
            raise StatisticalValidationError("Too many function evaluation failures in Monte Carlo")
        
        results = np.array(results)
        
        return {
            'mean': float(np.mean(results)),
            'std': float(np.std(results)),
            'uncertainty': float(np.std(results)),
            'median': float(np.median(results)),
            'percentile_2_5': float(np.percentile(results, 2.5)),
            'percentile_97_5': float(np.percentile(results, 97.5)),
            'n_successful_samples': len(results)
        }
    
    def _linear_propagation(self,
                          function: Callable,
                          variables: Dict[str, float],
                          uncertainties: Dict[str, float]) -> Dict[str, float]:
        """Linear (first-order) uncertainty propagation."""
        
        # Compute partial derivatives numerically
        epsilon = 1e-8
        partials = {}
        
        base_result = function(**variables)
        
        for var_name, var_value in variables.items():
            if var_name in uncertainties:
                # Compute partial derivative
                perturbed_vars = variables.copy()
                perturbed_vars[var_name] = var_value + epsilon
                
                try:
                    perturbed_result = function(**perturbed_vars)
                    partial = (perturbed_result - base_result) / epsilon
                    partials[var_name] = partial
                except:
                    partials[var_name] = 0.0
        
        # Linear uncertainty propagation
        variance = sum((partials[var_name] * uncertainties[var_name])**2 
                      for var_name in uncertainties.keys() 
                      if var_name in partials)
        
        uncertainty = np.sqrt(variance)
        
        return {
            'mean': float(base_result),
            'uncertainty': float(uncertainty),
            'partial_derivatives': {k: float(v) for k, v in partials.items()}
        }
    
    def _quadratic_propagation(self,
                             function: Callable,
                             variables: Dict[str, float],
                             uncertainties: Dict[str, float]) -> Dict[str, float]:
        """Quadratic (second-order) uncertainty propagation."""
        
        # This is a simplified implementation
        # Full second-order would require Hessian matrix computation
        linear_result = self._linear_propagation(function, variables, uncertainties)
        
        # Add second-order correction (simplified)
        epsilon = 1e-6
        base_result = function(**variables)
        
        second_order_variance = 0.0
        for var_name, var_value in variables.items():
            if var_name in uncertainties:
                uncertainty = uncertainties[var_name]
                
                # Compute second derivative
                vars_plus = variables.copy()
                vars_minus = variables.copy()
                vars_plus[var_name] = var_value + epsilon
                vars_minus[var_name] = var_value - epsilon
                
                try:
                    result_plus = function(**vars_plus)
                    result_minus = function(**vars_minus)
                    second_derivative = (result_plus - 2*base_result + result_minus) / (epsilon**2)
                    
                    # Second-order contribution
                    second_order_variance += 0.5 * (second_derivative * uncertainty**2)**2
                except:
                    continue
        
        total_variance = linear_result['uncertainty']**2 + second_order_variance
        total_uncertainty = np.sqrt(total_variance)
        
        return {
            'mean': linear_result['mean'],
            'uncertainty': float(total_uncertainty),
            'linear_uncertainty': linear_result['uncertainty'],
            'second_order_correction': float(np.sqrt(second_order_variance)),
            'partial_derivatives': linear_result['partial_derivatives']
        }
    
    def estimate_systematic_uncertainties(self,
                                        measurements: List[float],
                                        measurement_conditions: List[Dict[str, Any]],
                                        systematic_sources: List[str]) -> Dict[str, Any]:
        """
        Estimate systematic uncertainties from multiple measurements under different conditions.
        
        Args:
            measurements: List of measurement values
            measurement_conditions: List of condition dictionaries for each measurement
            systematic_sources: List of potential systematic error sources
            
        Returns:
            Dictionary with systematic uncertainty estimates
        """
        self.logger.info("Estimating systematic uncertainties")
        
        try:
            measurements = np.array(measurements)
            n_measurements = len(measurements)
            
            if n_measurements < 2:
                raise StatisticalValidationError("Need at least 2 measurements for systematic analysis")
            
            # Basic statistics
            mean_measurement = np.mean(measurements)
            std_measurement = np.std(measurements, ddof=1)
            
            # Analyze variation with conditions
            systematic_analysis = {}
            
            for source in systematic_sources:
                # Extract values for this systematic source from conditions
                source_values = []
                corresponding_measurements = []
                
                for i, conditions in enumerate(measurement_conditions):
                    if source in conditions:
                        source_values.append(conditions[source])
                        corresponding_measurements.append(measurements[i])
                
                if len(source_values) >= 2:
                    # Analyze correlation between systematic source and measurements
                    try:
                        correlation = np.corrcoef(source_values, corresponding_measurements)[0, 1]
                        
                        # Estimate systematic uncertainty contribution
                        source_range = max(source_values) - min(source_values)
                        measurement_range = max(corresponding_measurements) - min(corresponding_measurements)
                        
                        # Simple linear model for systematic effect
                        if source_range > 0:
                            systematic_sensitivity = measurement_range / source_range
                            systematic_uncertainty = systematic_sensitivity * np.std(source_values)
                        else:
                            systematic_uncertainty = 0.0
                        
                        systematic_analysis[source] = {
                            'correlation': float(correlation) if np.isfinite(correlation) else 0.0,
                            'sensitivity': float(systematic_sensitivity) if 'systematic_sensitivity' in locals() else 0.0,
                            'uncertainty_contribution': float(systematic_uncertainty),
                            'source_range': float(source_range),
                            'measurement_range': float(measurement_range)
                        }
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to analyze systematic source {source}: {e}")
                        continue
            
            # Total systematic uncertainty (assuming uncorrelated sources)
            total_systematic_variance = sum(
                analysis['uncertainty_contribution']**2 
                for analysis in systematic_analysis.values()
            )
            total_systematic_uncertainty = np.sqrt(total_systematic_variance)
            
            # Combined uncertainty (statistical + systematic)
            statistical_uncertainty = std_measurement / np.sqrt(n_measurements)  # Standard error
            combined_uncertainty = np.sqrt(statistical_uncertainty**2 + total_systematic_uncertainty**2)
            
            result = {
                'mean_measurement': float(mean_measurement),
                'statistical_uncertainty': float(statistical_uncertainty),
                'systematic_uncertainty': float(total_systematic_uncertainty),
                'combined_uncertainty': float(combined_uncertainty),
                'systematic_sources': systematic_analysis,
                'uncertainty_budget': {
                    'statistical_fraction': float(statistical_uncertainty**2 / combined_uncertainty**2) if combined_uncertainty > 0 else 1.0,
                    'systematic_fraction': float(total_systematic_uncertainty**2 / combined_uncertainty**2) if combined_uncertainty > 0 else 0.0
                }
            }
            
            self.logger.info(f"Systematic uncertainty analysis completed: "
                           f"statistical={statistical_uncertainty:.4f}, "
                           f"systematic={total_systematic_uncertainty:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Systematic uncertainty estimation failed: {e}")
            raise StatisticalValidationError(f"Systematic uncertainty estimation failed: {e}")
    
    def quantify_model_uncertainty(self,
                                 models: List[Callable],
                                 model_names: List[str],
                                 input_data: Dict[str, Any],
                                 weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Quantify model uncertainty by comparing predictions from different models.
        
        Args:
            models: List of model functions
            model_names: Names of the models
            input_data: Input data for model evaluation
            weights: Optional weights for model averaging
            
        Returns:
            Dictionary with model uncertainty quantification
        """
        self.logger.info(f"Quantifying model uncertainty across {len(models)} models")
        
        try:
            if len(models) != len(model_names):
                raise ValueError("Number of models must match number of model names")
            
            if weights is not None and len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            
            # Evaluate all models
            model_predictions = {}
            successful_models = []
            
            for i, (model, name) in enumerate(zip(models, model_names)):
                try:
                    prediction = model(**input_data)
                    if np.isfinite(prediction):
                        model_predictions[name] = float(prediction)
                        successful_models.append(i)
                except Exception as e:
                    self.logger.warning(f"Model {name} evaluation failed: {e}")
                    continue
            
            if len(model_predictions) < 2:
                raise StatisticalValidationError("Need at least 2 successful model evaluations")
            
            predictions = np.array(list(model_predictions.values()))
            
            # Model statistics
            if weights is not None:
                # Weighted statistics
                model_weights = np.array([weights[i] for i in successful_models])
                model_weights = model_weights / np.sum(model_weights)  # Normalize
                
                weighted_mean = np.average(predictions, weights=model_weights)
                weighted_variance = np.average((predictions - weighted_mean)**2, weights=model_weights)
                model_uncertainty = np.sqrt(weighted_variance)
            else:
                # Unweighted statistics
                weighted_mean = np.mean(predictions)
                model_uncertainty = np.std(predictions, ddof=1)
            
            # Model spread analysis
            prediction_range = np.max(predictions) - np.min(predictions)
            relative_spread = prediction_range / abs(weighted_mean) if weighted_mean != 0 else prediction_range
            
            # Model agreement metrics
            pairwise_differences = []
            for i in range(len(predictions)):
                for j in range(i+1, len(predictions)):
                    pairwise_differences.append(abs(predictions[i] - predictions[j]))
            
            mean_pairwise_difference = np.mean(pairwise_differences) if pairwise_differences else 0.0
            
            result = {
                'model_predictions': model_predictions,
                'ensemble_mean': float(weighted_mean),
                'model_uncertainty': float(model_uncertainty),
                'prediction_range': float(prediction_range),
                'relative_spread': float(relative_spread),
                'mean_pairwise_difference': float(mean_pairwise_difference),
                'n_successful_models': len(model_predictions),
                'model_agreement_score': float(1.0 / (1.0 + relative_spread))  # Higher is better agreement
            }
            
            self.logger.info(f"Model uncertainty quantified: "
                           f"ensemble_mean={weighted_mean:.4f} ± {model_uncertainty:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Model uncertainty quantification failed: {e}")
            raise StatisticalValidationError(f"Model uncertainty quantification failed: {e}")
    
    def compute_prediction_intervals(self,
                                   predictions: np.ndarray,
                                   uncertainties: np.ndarray,
                                   confidence_level: float = 0.95,
                                   method: str = 'gaussian') -> Dict[str, Any]:
        """
        Compute prediction intervals that account for both statistical and systematic uncertainties.
        
        Args:
            predictions: Array of predicted values
            uncertainties: Array of uncertainties for each prediction
            confidence_level: Confidence level for intervals
            method: Method for interval computation ('gaussian', 'bootstrap', 'quantile')
            
        Returns:
            Dictionary with prediction intervals
        """
        self.logger.debug(f"Computing prediction intervals using {method} method")
        
        try:
            predictions = np.asarray(predictions)
            uncertainties = np.asarray(uncertainties)
            
            if len(predictions) != len(uncertainties):
                raise ValueError("Predictions and uncertainties must have same length")
            
            if method == 'gaussian':
                # Assume Gaussian distribution
                alpha = 1 - confidence_level
                z_score = stats.norm.ppf(1 - alpha/2)
                
                lower_bounds = predictions - z_score * uncertainties
                upper_bounds = predictions + z_score * uncertainties
                
            elif method == 'bootstrap':
                # Bootstrap prediction intervals
                n_bootstrap = 1000
                bootstrap_predictions = []
                
                for pred, unc in zip(predictions, uncertainties):
                    bootstrap_samples = np.random.normal(pred, unc, n_bootstrap)
                    bootstrap_predictions.append(bootstrap_samples)
                
                bootstrap_predictions = np.array(bootstrap_predictions)
                
                alpha = 1 - confidence_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                lower_bounds = np.percentile(bootstrap_predictions, lower_percentile, axis=1)
                upper_bounds = np.percentile(bootstrap_predictions, upper_percentile, axis=1)
                
            elif method == 'quantile':
                # Quantile-based intervals (assumes empirical distribution)
                alpha = 1 - confidence_level
                
                # Simple approach: use uncertainty as scale parameter
                lower_bounds = predictions - uncertainties * stats.norm.ppf(1 - alpha/2)
                upper_bounds = predictions + uncertainties * stats.norm.ppf(1 - alpha/2)
                
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Compute interval statistics
            interval_widths = upper_bounds - lower_bounds
            mean_interval_width = np.mean(interval_widths)
            relative_interval_widths = interval_widths / np.abs(predictions)
            mean_relative_width = np.mean(relative_interval_widths[np.isfinite(relative_interval_widths)])
            
            result = {
                'lower_bounds': lower_bounds.tolist(),
                'upper_bounds': upper_bounds.tolist(),
                'interval_widths': interval_widths.tolist(),
                'mean_interval_width': float(mean_interval_width),
                'mean_relative_width': float(mean_relative_width) if np.isfinite(mean_relative_width) else None,
                'confidence_level': confidence_level,
                'method': method
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction interval computation failed: {e}")
            raise StatisticalValidationError(f"Prediction interval computation failed: {e}")


# Add uncertainty quantification methods to StatisticalPhysicsAnalyzer
def _add_uncertainty_methods(cls):
    """Add uncertainty quantification methods to StatisticalPhysicsAnalyzer."""
    
    def __init_uncertainty_quantifier__(self):
        """Initialize uncertainty quantifier as part of analyzer."""
        if not hasattr(self, '_uncertainty_quantifier'):
            self._uncertainty_quantifier = UncertaintyQuantifier(self)
        return self._uncertainty_quantifier
    
    # Add property for uncertainty quantifier
    cls.uncertainty_quantifier = property(__init_uncertainty_quantifier__)
    
    # Add convenience methods that delegate to uncertainty quantifier
    def propagate_uncertainty(self, *args, **kwargs):
        return self.uncertainty_quantifier.propagate_uncertainty(*args, **kwargs)
    
    def estimate_systematic_uncertainties(self, *args, **kwargs):
        return self.uncertainty_quantifier.estimate_systematic_uncertainties(*args, **kwargs)
    
    def quantify_model_uncertainty(self, *args, **kwargs):
        return self.uncertainty_quantifier.quantify_model_uncertainty(*args, **kwargs)
    
    def compute_prediction_intervals(self, *args, **kwargs):
        return self.uncertainty_quantifier.compute_prediction_intervals(*args, **kwargs)
    
    # Add methods to class
    cls.propagate_uncertainty = propagate_uncertainty
    cls.estimate_systematic_uncertainties = estimate_systematic_uncertainties
    cls.quantify_model_uncertainty = quantify_model_uncertainty
    cls.compute_prediction_intervals = compute_prediction_intervals
    
    return cls

# Apply the decorator to add uncertainty methods
StatisticalPhysicsAnalyzer = _add_uncertainty_methods(StatisticalPhysicsAnalyzer)