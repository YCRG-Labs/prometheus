"""
Statistical Validation Framework

This module implements task 11.1: Add statistical validation framework
- Implement error bar computation and confidence interval analysis
- Add finite-size scaling validation across different system sizes
- Create comprehensive quality metrics for all physics results
"""

import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from ..analysis.latent_analysis import LatentRepresentation
    from ..utils.logging_utils import get_logger
except ImportError:
    # Fallback for testing
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    try:
        from analysis.latent_analysis import LatentRepresentation
        from utils.logging_utils import get_logger
    except ImportError:
        # Mock for testing
        class LatentRepresentation:
            pass
        
        def get_logger(name):
            return logging.getLogger(name)


@dataclass
class ErrorBarResult:
    """Container for error bar computation results."""
    values: np.ndarray
    errors: np.ndarray
    confidence_intervals: List[Tuple[float, float]]
    confidence_level: float
    method: str
    bootstrap_samples: Optional[int] = None
    statistical_significance: Optional[float] = None


@dataclass
class ConfidenceIntervalResult:
    """Container for confidence interval analysis results."""
    mean: float
    std: float
    confidence_level: float
    lower_bound: float
    upper_bound: float
    method: str
    sample_size: int
    degrees_of_freedom: Optional[int] = None
    t_statistic: Optional[float] = None
    p_value: Optional[float] = None


@dataclass
class FiniteSizeScalingResult:
    """Container for finite-size scaling validation results."""
    system_sizes: List[int]
    scaling_exponent: float
    scaling_exponent_error: float
    scaling_amplitude: float
    scaling_amplitude_error: float
    r_squared: float
    p_value: float
    confidence_interval: Tuple[float, float]
    theoretical_exponent: Optional[float] = None
    scaling_accuracy: Optional[float] = None
    scaling_validation_passed: bool = False


@dataclass
class QualityMetrics:
    """Container for comprehensive quality metrics."""
    # Statistical quality
    statistical_significance: float
    confidence_level: float
    sample_adequacy: float
    
    # Physics quality
    physics_consistency: float
    theoretical_agreement: float
    universality_class_match: float
    
    # Data quality
    data_completeness: float
    equilibration_quality: float
    sampling_efficiency: float
    
    # Model quality
    model_convergence: float
    reconstruction_quality: float
    latent_space_quality: float
    
    # Overall quality score
    overall_quality_score: float


@dataclass
class SystemValidationMetrics:
    """Container for system-specific validation metrics."""
    system_type: str
    system_sizes: List[int]
    
    # Critical exponent results with error bars
    beta_result: Optional[ErrorBarResult] = None
    nu_result: Optional[ErrorBarResult] = None
    gamma_result: Optional[ErrorBarResult] = None
    
    # Critical temperature with confidence intervals
    tc_result: Optional[ConfidenceIntervalResult] = None
    
    # Finite-size scaling validation
    finite_size_scaling: Optional[Dict[str, FiniteSizeScalingResult]] = None
    
    # Quality assessment
    quality_metrics: Optional[QualityMetrics] = None
    
    # Overall validation status
    validation_passed: bool = False
    validation_score: float = 0.0


class StatisticalValidationFramework:
    """
    Comprehensive statistical validation framework for physics results.
    
    Features:
    1. Error bar computation using multiple methods (bootstrap, analytical, jackknife)
    2. Confidence interval analysis with proper statistical tests
    3. Finite-size scaling validation across different system sizes
    4. Comprehensive quality metrics for physics results
    5. Statistical significance testing and validation
    """
    
    def __init__(self,
                 confidence_level: float = 0.95,
                 bootstrap_samples: int = 10000,
                 significance_threshold: float = 0.05,
                 random_seed: Optional[int] = None):
        """
        Initialize statistical validation framework.
        
        Args:
            confidence_level: Confidence level for intervals (default: 0.95)
            bootstrap_samples: Number of bootstrap samples
            significance_threshold: P-value threshold for significance
            random_seed: Random seed for reproducibility
        """
        self.confidence_level = confidence_level
        self.bootstrap_samples = bootstrap_samples
        self.significance_threshold = significance_threshold
        self.random_seed = random_seed
        self.logger = get_logger(__name__)
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def compute_error_bars(self,
                          data: np.ndarray,
                          method: str = 'bootstrap',
                          confidence_level: Optional[float] = None) -> ErrorBarResult:
        """
        Compute error bars using specified method.
        
        Args:
            data: Input data array
            method: Method to use ('bootstrap', 'analytical', 'jackknife')
            confidence_level: Confidence level (uses instance default if None)
            
        Returns:
            ErrorBarResult with computed error bars
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        self.logger.info(f"Computing error bars using {method} method")
        
        if method == 'bootstrap':
            return self._compute_bootstrap_error_bars(data, confidence_level)
        elif method == 'analytical':
            return self._compute_analytical_error_bars(data, confidence_level)
        elif method == 'jackknife':
            return self._compute_jackknife_error_bars(data, confidence_level)
        else:
            raise ValueError(f"Unknown error bar method: {method}")
    
    def _compute_bootstrap_error_bars(self,
                                    data: np.ndarray,
                                    confidence_level: float) -> ErrorBarResult:
        """Compute error bars using bootstrap resampling."""
        
        if len(data) < 2:
            raise ValueError("Need at least 2 data points for bootstrap")
        
        # Bootstrap resampling
        rng = np.random.RandomState(self.random_seed)
        bootstrap_samples = []
        
        for _ in range(self.bootstrap_samples):
            # Resample with replacement
            indices = rng.choice(len(data), size=len(data), replace=True)
            bootstrap_sample = data[indices]
            bootstrap_samples.append(np.mean(bootstrap_sample))
        
        bootstrap_samples = np.array(bootstrap_samples)
        
        # Compute statistics
        mean_value = np.mean(data)
        bootstrap_std = np.std(bootstrap_samples)
        
        # Confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_samples, lower_percentile)
        ci_upper = np.percentile(bootstrap_samples, upper_percentile)
        
        # Statistical significance (two-tailed test)
        t_stat = abs(mean_value) / (bootstrap_std + 1e-10)
        p_value = 2 * (1 - stats.norm.cdf(t_stat))
        
        return ErrorBarResult(
            values=np.array([mean_value]),
            errors=np.array([bootstrap_std]),
            confidence_intervals=[(ci_lower, ci_upper)],
            confidence_level=confidence_level,
            method='bootstrap',
            bootstrap_samples=self.bootstrap_samples,
            statistical_significance=p_value
        )
    
    def _compute_analytical_error_bars(self,
                                     data: np.ndarray,
                                     confidence_level: float) -> ErrorBarResult:
        """Compute error bars using analytical methods."""
        
        if len(data) < 2:
            raise ValueError("Need at least 2 data points for analytical error bars")
        
        # Basic statistics
        mean_value = np.mean(data)
        std_value = np.std(data, ddof=1)  # Sample standard deviation
        sem = std_value / np.sqrt(len(data))  # Standard error of mean
        
        # Confidence interval using t-distribution
        df = len(data) - 1
        t_critical = stats.t.ppf((1 + confidence_level) / 2, df)
        
        ci_lower = mean_value - t_critical * sem
        ci_upper = mean_value + t_critical * sem
        
        # Statistical significance
        t_stat = abs(mean_value) / (sem + 1e-10)
        p_value = 2 * (1 - stats.t.cdf(t_stat, df))
        
        return ErrorBarResult(
            values=np.array([mean_value]),
            errors=np.array([sem]),
            confidence_intervals=[(ci_lower, ci_upper)],
            confidence_level=confidence_level,
            method='analytical',
            statistical_significance=p_value
        )
    
    def _compute_jackknife_error_bars(self,
                                    data: np.ndarray,
                                    confidence_level: float) -> ErrorBarResult:
        """Compute error bars using jackknife resampling."""
        
        if len(data) < 3:
            raise ValueError("Need at least 3 data points for jackknife")
        
        n = len(data)
        jackknife_samples = []
        
        # Jackknife resampling (leave-one-out)
        for i in range(n):
            jackknife_sample = np.concatenate([data[:i], data[i+1:]])
            jackknife_samples.append(np.mean(jackknife_sample))
        
        jackknife_samples = np.array(jackknife_samples)
        
        # Jackknife statistics
        mean_value = np.mean(data)
        jackknife_mean = np.mean(jackknife_samples)
        jackknife_var = (n - 1) * np.var(jackknife_samples, ddof=0)
        jackknife_std = np.sqrt(jackknife_var)
        
        # Confidence interval
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf((1 + confidence_level) / 2, n - 1)
        
        ci_lower = mean_value - t_critical * jackknife_std
        ci_upper = mean_value + t_critical * jackknife_std
        
        # Statistical significance
        t_stat = abs(mean_value) / (jackknife_std + 1e-10)
        p_value = 2 * (1 - stats.t.cdf(t_stat, n - 1))
        
        return ErrorBarResult(
            values=np.array([mean_value]),
            errors=np.array([jackknife_std]),
            confidence_intervals=[(ci_lower, ci_upper)],
            confidence_level=confidence_level,
            method='jackknife',
            statistical_significance=p_value
        )
    
    def analyze_confidence_intervals(self,
                                   data: np.ndarray,
                                   theoretical_value: Optional[float] = None) -> ConfidenceIntervalResult:
        """
        Perform comprehensive confidence interval analysis.
        
        Args:
            data: Input data array
            theoretical_value: Theoretical value for comparison
            
        Returns:
            ConfidenceIntervalResult with detailed analysis
        """
        if len(data) < 2:
            raise ValueError("Need at least 2 data points for confidence interval analysis")
        
        # Basic statistics
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1)
        n = len(data)
        sem = std_val / np.sqrt(n)
        
        # Degrees of freedom
        df = n - 1
        
        # T-statistic and confidence interval
        t_critical = stats.t.ppf((1 + self.confidence_level) / 2, df)
        
        lower_bound = mean_val - t_critical * sem
        upper_bound = mean_val + t_critical * sem
        
        # Statistical tests
        if theoretical_value is not None:
            # One-sample t-test against theoretical value
            t_stat = (mean_val - theoretical_value) / (sem + 1e-10)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        else:
            # Test against zero
            t_stat = mean_val / (sem + 1e-10)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        return ConfidenceIntervalResult(
            mean=mean_val,
            std=std_val,
            confidence_level=self.confidence_level,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            method='t-distribution',
            sample_size=n,
            degrees_of_freedom=df,
            t_statistic=t_stat,
            p_value=p_value
        )
    
    def validate_finite_size_scaling(self,
                                   system_sizes: List[int],
                                   observables: List[np.ndarray],
                                   theoretical_exponent: Optional[float] = None,
                                   observable_name: str = "observable") -> FiniteSizeScalingResult:
        """
        Validate finite-size scaling behavior across different system sizes.
        
        Args:
            system_sizes: List of system sizes (L values)
            observables: List of observable arrays for each system size
            theoretical_exponent: Theoretical scaling exponent
            observable_name: Name of the observable for logging
            
        Returns:
            FiniteSizeScalingResult with scaling analysis
        """
        if len(system_sizes) != len(observables):
            raise ValueError("Number of system sizes must match number of observable arrays")
        
        if len(system_sizes) < 3:
            raise ValueError("Need at least 3 system sizes for finite-size scaling analysis")
        
        self.logger.info(f"Validating finite-size scaling for {observable_name}")
        
        # Compute mean observables for each system size
        mean_observables = []
        observable_errors = []
        
        for i, obs_array in enumerate(observables):
            if len(obs_array) == 0:
                raise ValueError(f"Empty observable array for system size {system_sizes[i]}")
            
            mean_obs = np.mean(obs_array)
            
            # Compute error (standard error of mean)
            if len(obs_array) > 1:
                obs_error = np.std(obs_array, ddof=1) / np.sqrt(len(obs_array))
            else:
                obs_error = 0.1 * abs(mean_obs)  # 10% error estimate
            
            mean_observables.append(mean_obs)
            observable_errors.append(obs_error)
        
        mean_observables = np.array(mean_observables)
        observable_errors = np.array(observable_errors)
        system_sizes_array = np.array(system_sizes, dtype=float)
        
        # Remove invalid data points
        valid_mask = (mean_observables > 0) & (system_sizes_array > 0) & np.isfinite(mean_observables)
        
        if np.sum(valid_mask) < 3:
            raise ValueError("Insufficient valid data points for scaling analysis")
        
        mean_observables = mean_observables[valid_mask]
        observable_errors = observable_errors[valid_mask]
        system_sizes_array = system_sizes_array[valid_mask]
        
        # Fit power law: observable ~ L^exponent
        # Use log-log fit: log(observable) = exponent * log(L) + log(amplitude)
        
        log_sizes = np.log(system_sizes_array)
        log_observables = np.log(mean_observables)
        log_errors = observable_errors / (mean_observables + 1e-10)
        
        # Weighted linear fit
        try:
            weights = 1.0 / (log_errors**2 + 1e-10)
            
            # Fit log(observable) = a * log(L) + b
            popt, pcov = curve_fit(
                lambda x, a, b: a * x + b,
                log_sizes, log_observables,
                sigma=1.0/np.sqrt(weights),
                absolute_sigma=False
            )
            
            scaling_exponent = popt[0]
            log_amplitude = popt[1]
            scaling_amplitude = np.exp(log_amplitude)
            
            # Parameter errors
            param_errors = np.sqrt(np.diag(pcov))
            scaling_exponent_error = param_errors[0]
            scaling_amplitude_error = scaling_amplitude * param_errors[1]
            
        except Exception as e:
            self.logger.warning(f"Weighted fit failed, using unweighted fit: {e}")
            
            # Fallback to unweighted fit
            popt, pcov = np.polyfit(log_sizes, log_observables, 1, cov=True)
            
            scaling_exponent = popt[0]
            log_amplitude = popt[1]
            scaling_amplitude = np.exp(log_amplitude)
            
            param_errors = np.sqrt(np.diag(pcov))
            scaling_exponent_error = param_errors[0]
            scaling_amplitude_error = scaling_amplitude * param_errors[1]
        
        # Compute R-squared
        log_observables_pred = scaling_exponent * log_sizes + log_amplitude
        ss_res = np.sum((log_observables - log_observables_pred)**2)
        ss_tot = np.sum((log_observables - np.mean(log_observables))**2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))
        
        # Statistical significance
        n_points = len(log_sizes)
        df = n_points - 2
        
        if df > 0 and scaling_exponent_error > 0:
            t_stat = abs(scaling_exponent) / scaling_exponent_error
            p_value = 2 * (1 - stats.t.cdf(t_stat, df))
        else:
            p_value = 1.0
        
        # Confidence interval for scaling exponent
        if df > 0:
            t_critical = stats.t.ppf((1 + self.confidence_level) / 2, df)
            ci_lower = scaling_exponent - t_critical * scaling_exponent_error
            ci_upper = scaling_exponent + t_critical * scaling_exponent_error
        else:
            ci_lower = scaling_exponent - scaling_exponent_error
            ci_upper = scaling_exponent + scaling_exponent_error
        
        # Validation against theoretical value
        scaling_accuracy = None
        scaling_validation_passed = False
        
        if theoretical_exponent is not None:
            scaling_accuracy = (1 - abs(scaling_exponent - theoretical_exponent) / 
                              abs(theoretical_exponent)) * 100
            
            # Check if theoretical value is within confidence interval
            scaling_validation_passed = (ci_lower <= theoretical_exponent <= ci_upper)
        
        # Overall validation
        if scaling_validation_passed is False and theoretical_exponent is not None:
            # Check if at least reasonably close (within 2 standard deviations)
            scaling_validation_passed = (abs(scaling_exponent - theoretical_exponent) <= 
                                       2 * scaling_exponent_error)
        
        result = FiniteSizeScalingResult(
            system_sizes=list(system_sizes_array.astype(int)),
            scaling_exponent=scaling_exponent,
            scaling_exponent_error=scaling_exponent_error,
            scaling_amplitude=scaling_amplitude,
            scaling_amplitude_error=scaling_amplitude_error,
            r_squared=r_squared,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            theoretical_exponent=theoretical_exponent,
            scaling_accuracy=scaling_accuracy,
            scaling_validation_passed=scaling_validation_passed
        )
        
        self.logger.info(f"Finite-size scaling: exponent = {scaling_exponent:.4f} ± {scaling_exponent_error:.4f}")
        if scaling_accuracy is not None:
            self.logger.info(f"Scaling accuracy: {scaling_accuracy:.1f}%")
        
        return result
    
    def compute_quality_metrics(self,
                              system_data: Dict[str, Any],
                              validation_results: Dict[str, Any]) -> QualityMetrics:
        """
        Compute comprehensive quality metrics for physics results.
        
        Args:
            system_data: Dictionary containing system data and results
            validation_results: Dictionary containing validation results
            
        Returns:
            QualityMetrics with comprehensive quality assessment
        """
        self.logger.info("Computing comprehensive quality metrics")
        
        # Initialize quality scores
        statistical_significance = 0.0
        confidence_level = self.confidence_level
        sample_adequacy = 0.0
        physics_consistency = 0.0
        theoretical_agreement = 0.0
        universality_class_match = 0.0
        data_completeness = 0.0
        equilibration_quality = 0.0
        sampling_efficiency = 0.0
        model_convergence = 0.0
        reconstruction_quality = 0.0
        latent_space_quality = 0.0
        
        # Statistical quality assessment
        p_values = []
        
        # Collect p-values from various tests
        if 'error_bar_results' in validation_results:
            for result in validation_results['error_bar_results'].values():
                if hasattr(result, 'statistical_significance') and result.statistical_significance:
                    p_values.append(result.statistical_significance)
        
        if 'confidence_intervals' in validation_results:
            for result in validation_results['confidence_intervals'].values():
                if hasattr(result, 'p_value') and result.p_value:
                    p_values.append(result.p_value)
        
        if 'finite_size_scaling' in validation_results:
            for result in validation_results['finite_size_scaling'].values():
                if hasattr(result, 'p_value') and result.p_value:
                    p_values.append(result.p_value)
        
        # Statistical significance score
        if p_values:
            # Use Bonferroni correction for multiple testing
            corrected_alpha = self.significance_threshold / len(p_values)
            significant_tests = sum(1 for p in p_values if p < corrected_alpha)
            statistical_significance = significant_tests / len(p_values)
        
        # Sample adequacy assessment
        sample_sizes = []
        if 'sample_sizes' in system_data:
            sample_sizes = system_data['sample_sizes']
        elif 'data_arrays' in system_data:
            sample_sizes = [len(arr) for arr in system_data['data_arrays'] if hasattr(arr, '__len__')]
        
        if sample_sizes:
            min_adequate_size = 30  # Rule of thumb for CLT
            adequate_samples = sum(1 for size in sample_sizes if size >= min_adequate_size)
            sample_adequacy = adequate_samples / len(sample_sizes)
        
        # Physics consistency assessment
        physics_scores = []
        
        # Check critical exponent values
        if 'critical_exponents' in system_data:
            exponents = system_data['critical_exponents']
            
            # Beta exponent should be positive and reasonable
            if 'beta' in exponents:
                beta_val = exponents['beta']
                if 0.05 <= beta_val <= 0.6:
                    physics_scores.append(1.0)
                elif 0.01 <= beta_val <= 1.0:
                    physics_scores.append(0.7)
                else:
                    physics_scores.append(0.2)
            
            # Nu exponent should be positive and reasonable
            if 'nu' in exponents:
                nu_val = abs(exponents['nu'])  # Take absolute value
                if 0.3 <= nu_val <= 1.5:
                    physics_scores.append(1.0)
                elif 0.1 <= nu_val <= 2.0:
                    physics_scores.append(0.7)
                else:
                    physics_scores.append(0.2)
        
        if physics_scores:
            physics_consistency = np.mean(physics_scores)
        
        # Theoretical agreement assessment
        agreement_scores = []
        
        if 'finite_size_scaling' in validation_results:
            for result in validation_results['finite_size_scaling'].values():
                if hasattr(result, 'scaling_accuracy') and result.scaling_accuracy is not None:
                    agreement_scores.append(result.scaling_accuracy / 100.0)
        
        if 'accuracy_percentages' in system_data:
            for accuracy in system_data['accuracy_percentages']:
                if accuracy is not None:
                    agreement_scores.append(accuracy / 100.0)
        
        if agreement_scores:
            theoretical_agreement = np.mean(agreement_scores)
        
        # Universality class match (simplified assessment)
        if 'system_type' in system_data:
            system_type = system_data['system_type']
            
            # Check if extracted exponents match expected universality class
            if 'critical_exponents' in system_data:
                exponents = system_data['critical_exponents']
                
                if system_type == 'ising_2d':
                    # 2D Ising: β ≈ 0.125, ν ≈ 1.0
                    beta_match = 0.0
                    nu_match = 0.0
                    
                    if 'beta' in exponents:
                        beta_error = abs(exponents['beta'] - 0.125) / 0.125
                        beta_match = max(0, 1 - beta_error)
                    
                    if 'nu' in exponents:
                        nu_error = abs(abs(exponents['nu']) - 1.0) / 1.0
                        nu_match = max(0, 1 - nu_error)
                    
                    universality_class_match = (beta_match + nu_match) / 2
                
                elif system_type == 'ising_3d':
                    # 3D Ising: β ≈ 0.326, ν ≈ 0.630
                    beta_match = 0.0
                    nu_match = 0.0
                    
                    if 'beta' in exponents:
                        beta_error = abs(exponents['beta'] - 0.326) / 0.326
                        beta_match = max(0, 1 - beta_error)
                    
                    if 'nu' in exponents:
                        nu_error = abs(abs(exponents['nu']) - 0.630) / 0.630
                        nu_match = max(0, 1 - nu_error)
                    
                    universality_class_match = (beta_match + nu_match) / 2
        
        # Data completeness assessment
        completeness_scores = []
        
        if 'expected_data_points' in system_data and 'actual_data_points' in system_data:
            expected = system_data['expected_data_points']
            actual = system_data['actual_data_points']
            completeness_scores.append(min(1.0, actual / expected))
        
        if 'temperature_coverage' in system_data:
            # Check if temperature range covers critical region adequately
            temp_coverage = system_data['temperature_coverage']
            if temp_coverage >= 0.8:  # 80% coverage threshold
                completeness_scores.append(1.0)
            else:
                completeness_scores.append(temp_coverage)
        
        if completeness_scores:
            data_completeness = np.mean(completeness_scores)
        else:
            data_completeness = 0.8  # Default reasonable value
        
        # Equilibration quality (if available)
        if 'equilibration_metrics' in system_data:
            eq_metrics = system_data['equilibration_metrics']
            
            if 'convergence_achieved' in eq_metrics:
                equilibration_quality = 1.0 if eq_metrics['convergence_achieved'] else 0.3
            elif 'equilibration_score' in eq_metrics:
                equilibration_quality = eq_metrics['equilibration_score']
        else:
            equilibration_quality = 0.7  # Default reasonable value
        
        # Sampling efficiency
        if 'autocorrelation_times' in system_data:
            autocorr_times = system_data['autocorrelation_times']
            if 'sampling_intervals' in system_data:
                sampling_intervals = system_data['sampling_intervals']
                
                # Good sampling: interval >> autocorrelation time
                efficiency_ratios = []
                for autocorr, interval in zip(autocorr_times, sampling_intervals):
                    if autocorr > 0:
                        ratio = interval / autocorr
                        efficiency = min(1.0, ratio / 5.0)  # 5x autocorr time is good
                        efficiency_ratios.append(efficiency)
                
                if efficiency_ratios:
                    sampling_efficiency = np.mean(efficiency_ratios)
        
        if sampling_efficiency == 0.0:
            sampling_efficiency = 0.7  # Default reasonable value
        
        # Model convergence (if available)
        if 'model_metrics' in system_data:
            model_metrics = system_data['model_metrics']
            
            if 'training_converged' in model_metrics:
                model_convergence = 1.0 if model_metrics['training_converged'] else 0.3
            elif 'convergence_score' in model_metrics:
                model_convergence = model_metrics['convergence_score']
        else:
            model_convergence = 0.8  # Default reasonable value
        
        # Reconstruction quality (if available)
        if 'reconstruction_metrics' in system_data:
            recon_metrics = system_data['reconstruction_metrics']
            
            if 'r_squared' in recon_metrics:
                reconstruction_quality = max(0, recon_metrics['r_squared'])
            elif 'reconstruction_score' in recon_metrics:
                reconstruction_quality = recon_metrics['reconstruction_score']
        else:
            reconstruction_quality = 0.7  # Default reasonable value
        
        # Latent space quality (if available)
        if 'latent_metrics' in system_data:
            latent_metrics = system_data['latent_metrics']
            
            quality_components = []
            
            if 'magnetization_correlation' in latent_metrics:
                quality_components.append(abs(latent_metrics['magnetization_correlation']))
            
            if 'temperature_correlation' in latent_metrics:
                quality_components.append(abs(latent_metrics['temperature_correlation']))
            
            if 'phase_separability' in latent_metrics:
                # Normalize separability score
                separability = latent_metrics['phase_separability']
                quality_components.append(min(1.0, separability / 3.0))
            
            if quality_components:
                latent_space_quality = np.mean(quality_components)
        else:
            latent_space_quality = 0.6  # Default reasonable value
        
        # Compute overall quality score
        quality_components = [
            statistical_significance * 0.15,
            sample_adequacy * 0.10,
            physics_consistency * 0.15,
            theoretical_agreement * 0.15,
            universality_class_match * 0.10,
            data_completeness * 0.10,
            equilibration_quality * 0.10,
            sampling_efficiency * 0.05,
            model_convergence * 0.05,
            reconstruction_quality * 0.025,
            latent_space_quality * 0.025
        ]
        
        overall_quality_score = sum(quality_components)
        
        return QualityMetrics(
            statistical_significance=statistical_significance,
            confidence_level=confidence_level,
            sample_adequacy=sample_adequacy,
            physics_consistency=physics_consistency,
            theoretical_agreement=theoretical_agreement,
            universality_class_match=universality_class_match,
            data_completeness=data_completeness,
            equilibration_quality=equilibration_quality,
            sampling_efficiency=sampling_efficiency,
            model_convergence=model_convergence,
            reconstruction_quality=reconstruction_quality,
            latent_space_quality=latent_space_quality,
            overall_quality_score=overall_quality_score
        )
    
    def validate_system_comprehensive(self,
                                    system_data: Dict[str, Any],
                                    theoretical_values: Optional[Dict[str, float]] = None) -> SystemValidationMetrics:
        """
        Perform comprehensive validation for a single system.
        
        Args:
            system_data: Dictionary containing all system data
            theoretical_values: Dictionary of theoretical values for comparison
            
        Returns:
            SystemValidationMetrics with complete validation results
        """
        system_type = system_data.get('system_type', 'unknown')
        system_sizes = system_data.get('system_sizes', [])
        
        self.logger.info(f"Performing comprehensive validation for {system_type}")
        
        validation_results = {}
        
        # Error bar computation for critical exponents
        error_bar_results = {}
        
        if 'critical_exponent_data' in system_data:
            exponent_data = system_data['critical_exponent_data']
            
            for exponent_name, data_array in exponent_data.items():
                if len(data_array) > 1:
                    try:
                        error_result = self.compute_error_bars(data_array, method='bootstrap')
                        error_bar_results[exponent_name] = error_result
                    except Exception as e:
                        self.logger.warning(f"Error bar computation failed for {exponent_name}: {e}")
        
        validation_results['error_bar_results'] = error_bar_results
        
        # Confidence interval analysis
        confidence_intervals = {}
        
        if 'critical_temperature_data' in system_data:
            tc_data = system_data['critical_temperature_data']
            theoretical_tc = theoretical_values.get('tc') if theoretical_values else None
            
            try:
                ci_result = self.analyze_confidence_intervals(tc_data, theoretical_tc)
                confidence_intervals['critical_temperature'] = ci_result
            except Exception as e:
                self.logger.warning(f"Confidence interval analysis failed for Tc: {e}")
        
        validation_results['confidence_intervals'] = confidence_intervals
        
        # Finite-size scaling validation
        finite_size_scaling = {}
        
        if 'finite_size_data' in system_data and len(system_sizes) >= 3:
            fs_data = system_data['finite_size_data']
            
            for observable_name, size_data in fs_data.items():
                if len(size_data) == len(system_sizes):
                    theoretical_exponent = None
                    if theoretical_values and f'{observable_name}_scaling_exponent' in theoretical_values:
                        theoretical_exponent = theoretical_values[f'{observable_name}_scaling_exponent']
                    
                    try:
                        fs_result = self.validate_finite_size_scaling(
                            system_sizes, size_data, theoretical_exponent, observable_name
                        )
                        finite_size_scaling[observable_name] = fs_result
                    except Exception as e:
                        self.logger.warning(f"Finite-size scaling failed for {observable_name}: {e}")
        
        validation_results['finite_size_scaling'] = finite_size_scaling
        
        # Quality metrics computation
        try:
            quality_metrics = self.compute_quality_metrics(system_data, validation_results)
        except Exception as e:
            self.logger.warning(f"Quality metrics computation failed: {e}")
            quality_metrics = QualityMetrics(
                statistical_significance=0.0,
                confidence_level=self.confidence_level,
                sample_adequacy=0.0,
                physics_consistency=0.0,
                theoretical_agreement=0.0,
                universality_class_match=0.0,
                data_completeness=0.0,
                equilibration_quality=0.0,
                sampling_efficiency=0.0,
                model_convergence=0.0,
                reconstruction_quality=0.0,
                latent_space_quality=0.0,
                overall_quality_score=0.0
            )
        
        # Overall validation assessment
        validation_score = quality_metrics.overall_quality_score
        
        # Validation passes if quality score is above threshold and key metrics are good
        validation_passed = (
            validation_score >= 0.6 and
            quality_metrics.statistical_significance >= 0.5 and
            quality_metrics.physics_consistency >= 0.5
        )
        
        return SystemValidationMetrics(
            system_type=system_type,
            system_sizes=system_sizes,
            beta_result=error_bar_results.get('beta'),
            nu_result=error_bar_results.get('nu'),
            gamma_result=error_bar_results.get('gamma'),
            tc_result=confidence_intervals.get('critical_temperature'),
            finite_size_scaling=finite_size_scaling if finite_size_scaling else None,
            quality_metrics=quality_metrics,
            validation_passed=validation_passed,
            validation_score=validation_score
        )
    
    def generate_validation_report(self,
                                 validation_results: List[SystemValidationMetrics],
                                 output_dir: str = "results/statistical_validation") -> Dict[str, Any]:
        """
        Generate comprehensive validation report with visualizations.
        
        Args:
            validation_results: List of system validation results
            output_dir: Output directory for report files
            
        Returns:
            Dictionary containing report summary
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Generating statistical validation report in {output_path}")
        
        # Create summary statistics
        report_summary = {
            'total_systems': len(validation_results),
            'systems_passed': sum(1 for r in validation_results if r.validation_passed),
            'average_quality_score': np.mean([r.validation_score for r in validation_results]),
            'validation_timestamp': np.datetime64('now').isoformat()
        }
        
        # Save detailed results
        detailed_results = {
            'summary': report_summary,
            'system_results': [asdict(result) for result in validation_results]
        }
        
        json_path = output_path / "statistical_validation_results.json"
        with open(json_path, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        # Generate visualizations
        self._create_validation_visualizations(validation_results, output_path)
        
        # Generate text report
        self._generate_text_validation_report(validation_results, output_path)
        
        self.logger.info("Statistical validation report generated successfully")
        
        return report_summary
    
    def _create_validation_visualizations(self,
                                        validation_results: List[SystemValidationMetrics],
                                        output_path: Path):
        """Create comprehensive validation visualizations."""
        
        # Quality metrics comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Overall quality scores
        ax = axes[0, 0]
        system_names = [r.system_type for r in validation_results]
        quality_scores = [r.validation_score for r in validation_results]
        
        bars = ax.bar(range(len(system_names)), quality_scores,
                     color=['green' if score >= 0.6 else 'red' for score in quality_scores])
        ax.axhline(0.6, color='black', linestyle='--', label='Threshold (0.6)')
        ax.set_xlabel('System')
        ax.set_ylabel('Quality Score')
        ax.set_title('Overall Quality Assessment')
        ax.set_xticks(range(len(system_names)))
        ax.set_xticklabels(system_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Quality component breakdown
        ax = axes[0, 1]
        
        if validation_results and validation_results[0].quality_metrics:
            quality_components = [
                'Statistical\nSignificance',
                'Physics\nConsistency',
                'Theoretical\nAgreement',
                'Data\nCompleteness',
                'Model\nConvergence'
            ]
            
            # Average across all systems
            avg_components = []
            for component in ['statistical_significance', 'physics_consistency', 
                            'theoretical_agreement', 'data_completeness', 'model_convergence']:
                values = [getattr(r.quality_metrics, component) for r in validation_results 
                         if r.quality_metrics]
                avg_components.append(np.mean(values) if values else 0)
            
            bars = ax.bar(range(len(quality_components)), avg_components, color='skyblue')
            ax.set_xlabel('Quality Component')
            ax.set_ylabel('Average Score')
            ax.set_title('Quality Component Breakdown')
            ax.set_xticks(range(len(quality_components)))
            ax.set_xticklabels(quality_components, rotation=45)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Finite-size scaling results
        ax = axes[1, 0]
        
        scaling_accuracies = []
        scaling_systems = []
        
        for result in validation_results:
            if result.finite_size_scaling:
                for obs_name, fs_result in result.finite_size_scaling.items():
                    if fs_result.scaling_accuracy is not None:
                        scaling_accuracies.append(fs_result.scaling_accuracy)
                        scaling_systems.append(f"{result.system_type}\n{obs_name}")
        
        if scaling_accuracies:
            bars = ax.bar(range(len(scaling_systems)), scaling_accuracies,
                         color=['green' if acc >= 70 else 'orange' if acc >= 50 else 'red' 
                               for acc in scaling_accuracies])
            ax.axhline(70, color='black', linestyle='--', label='Target (70%)')
            ax.set_xlabel('System/Observable')
            ax.set_ylabel('Scaling Accuracy (%)')
            ax.set_title('Finite-Size Scaling Validation')
            ax.set_xticks(range(len(scaling_systems)))
            ax.set_xticklabels(scaling_systems, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Statistical significance summary
        ax = axes[1, 1]
        
        significance_data = []
        for result in validation_results:
            if result.quality_metrics:
                significance_data.append(result.quality_metrics.statistical_significance)
        
        if significance_data:
            ax.hist(significance_data, bins=10, alpha=0.7, color='lightblue', edgecolor='black')
            ax.axvline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
            ax.axvline(np.mean(significance_data), color='green', linestyle='-',
                      label=f'Mean ({np.mean(significance_data):.2f})')
            ax.set_xlabel('Statistical Significance Score')
            ax.set_ylabel('Number of Systems')
            ax.set_title('Statistical Significance Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_path / "statistical_validation_summary.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _generate_text_validation_report(self,
                                       validation_results: List[SystemValidationMetrics],
                                       output_path: Path):
        """Generate detailed text validation report."""
        
        report_path = output_path / "statistical_validation_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("STATISTICAL VALIDATION FRAMEWORK REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Validation Framework Configuration:\n")
            f.write(f"  Confidence Level: {self.confidence_level:.1%}\n")
            f.write(f"  Bootstrap Samples: {self.bootstrap_samples}\n")
            f.write(f"  Significance Threshold: {self.significance_threshold}\n\n")
            
            f.write("SYSTEM VALIDATION RESULTS\n")
            f.write("-" * 40 + "\n\n")
            
            for result in validation_results:
                f.write(f"System: {result.system_type}\n")
                f.write(f"  System Sizes: {result.system_sizes}\n")
                f.write(f"  Validation Passed: {result.validation_passed}\n")
                f.write(f"  Quality Score: {result.validation_score:.3f}\n\n")
                
                if result.quality_metrics:
                    qm = result.quality_metrics
                    f.write(f"  Quality Metrics:\n")
                    f.write(f"    Statistical Significance: {qm.statistical_significance:.3f}\n")
                    f.write(f"    Physics Consistency: {qm.physics_consistency:.3f}\n")
                    f.write(f"    Theoretical Agreement: {qm.theoretical_agreement:.3f}\n")
                    f.write(f"    Data Completeness: {qm.data_completeness:.3f}\n")
                    f.write(f"    Overall Quality: {qm.overall_quality_score:.3f}\n\n")
                
                if result.finite_size_scaling:
                    f.write(f"  Finite-Size Scaling Results:\n")
                    for obs_name, fs_result in result.finite_size_scaling.items():
                        f.write(f"    {obs_name}:\n")
                        f.write(f"      Scaling Exponent: {fs_result.scaling_exponent:.4f} ± {fs_result.scaling_exponent_error:.4f}\n")
                        f.write(f"      R²: {fs_result.r_squared:.3f}\n")
                        if fs_result.scaling_accuracy is not None:
                            f.write(f"      Accuracy: {fs_result.scaling_accuracy:.1f}%\n")
                        f.write(f"      Validation Passed: {fs_result.scaling_validation_passed}\n")
                    f.write("\n")
                
                f.write("-" * 40 + "\n\n")
            
            # Summary statistics
            total_systems = len(validation_results)
            passed_systems = sum(1 for r in validation_results if r.validation_passed)
            avg_quality = np.mean([r.validation_score for r in validation_results])
            
            f.write("VALIDATION SUMMARY\n")
            f.write("-" * 40 + "\n\n")
            f.write(f"Total Systems Validated: {total_systems}\n")
            f.write(f"Systems Passed: {passed_systems}\n")
            f.write(f"Success Rate: {passed_systems/total_systems:.1%}\n")
            f.write(f"Average Quality Score: {avg_quality:.3f}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF STATISTICAL VALIDATION REPORT\n")
            f.write("=" * 80 + "\n")


def create_statistical_validation_framework(confidence_level: float = 0.95,
                                          bootstrap_samples: int = 10000,
                                          significance_threshold: float = 0.05,
                                          random_seed: Optional[int] = None) -> StatisticalValidationFramework:
    """
    Factory function to create StatisticalValidationFramework.
    
    Args:
        confidence_level: Confidence level for intervals (default: 0.95)
        bootstrap_samples: Number of bootstrap samples
        significance_threshold: P-value threshold for significance
        random_seed: Random seed for reproducibility
        
    Returns:
        Configured StatisticalValidationFramework instance
    """
    return StatisticalValidationFramework(
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        significance_threshold=significance_threshold,
        random_seed=random_seed
    )