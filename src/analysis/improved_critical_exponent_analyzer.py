"""
Improved Critical Exponent Extraction Framework

This module provides enhanced critical exponent analysis with better accuracy
through improved critical temperature detection, data preprocessing, and fitting methods.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from scipy.optimize import curve_fit, minimize_scalar, differential_evolution
from scipy.stats import linregress, bootstrap
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
import warnings

from .latent_analysis import LatentRepresentation
from .phase_detection import PhaseDetectionResult
from .enhanced_validation_types import (
    CriticalExponentValidation, ConfidenceInterval, 
    CriticalExponentError, UniversalityClass
)
from .numerical_stability_fixes import safe_log, safe_divide
from ..utils.logging_utils import get_logger


@dataclass
class ImprovedPowerLawFitResult:
    """Enhanced container for power-law fitting results with additional accuracy metrics."""
    exponent: float
    amplitude: float
    exponent_error: float
    amplitude_error: float
    r_squared: float
    p_value: float
    fit_range: Tuple[float, float]
    residuals: np.ndarray
    confidence_interval: Optional[ConfidenceInterval] = None
    critical_temperature_used: float = None
    data_quality_score: float = None
    finite_size_correction: Optional[float] = None


class ImprovedCriticalTemperatureDetector:
    """
    Enhanced critical temperature detection using multiple methods and ensemble averaging.
    """
    
    def __init__(self, smoothing_window: int = 5, bootstrap_samples: int = 500):
        """Initialize improved critical temperature detector.
        
        Args:
            smoothing_window: Window size for smoothing operations
            bootstrap_samples: Number of bootstrap samples for confidence estimation
        """
        self.smoothing_window = smoothing_window
        self.bootstrap_samples = bootstrap_samples
        self.logger = get_logger(__name__)
    
    def detect_critical_temperature(self, 
                                  temperatures: np.ndarray,
                                  order_parameter: np.ndarray,
                                  method: str = 'ensemble') -> Tuple[float, float]:
        """
        Detect critical temperature using improved methods.
        
        Args:
            temperatures: Temperature array
            order_parameter: Order parameter values
            method: Detection method ('ensemble', 'derivative', 'susceptibility', 
                   'variance', 'inflection', 'binder')
            
        Returns:
            Tuple of (critical_temperature, confidence)
        """
        self.logger.info(f"Detecting critical temperature using {method} method")
        
        if method == 'ensemble':
            return self._ensemble_detection(temperatures, order_parameter)
        elif method == 'derivative':
            return self._derivative_method(temperatures, order_parameter)
        elif method == 'susceptibility':
            return self._susceptibility_method(temperatures, order_parameter)
        elif method == 'variance':
            return self._variance_method(temperatures, order_parameter)
        elif method == 'inflection':
            return self._inflection_method(temperatures, order_parameter)
        elif method == 'binder':
            return self._binder_cumulant_method(temperatures, order_parameter)
        else:
            raise ValueError(f"Unknown detection method: {method}")
    
    def _ensemble_detection(self, temperatures: np.ndarray, order_parameter: np.ndarray) -> Tuple[float, float]:
        """Ensemble method combining multiple detection approaches."""
        methods = ['derivative', 'susceptibility', 'variance', 'inflection']
        tc_estimates = []
        confidences = []
        method_names = []
        
        for method in methods:
            try:
                tc, conf = getattr(self, f'_{method}_method')(temperatures, order_parameter)
                if np.isfinite(tc) and conf > 0.1:  # Only use reasonable estimates
                    tc_estimates.append(tc)
                    confidences.append(conf)
                    method_names.append(method)
                    self.logger.debug(f"Method {method}: Tc={tc:.4f}, confidence={conf:.3f}")
            except Exception as e:
                self.logger.warning(f"Method {method} failed: {e}")
                continue
        
        if not tc_estimates:
            # Fallback to simple derivative method
            self.logger.warning("All ensemble methods failed, using derivative fallback")
            return self._derivative_method(temperatures, order_parameter)
        
        # Weighted average based on confidence
        weights = np.array(confidences)
        weights = weights / np.sum(weights)
        
        tc_ensemble = np.average(tc_estimates, weights=weights)
        
        # Enhanced confidence calculation considering method agreement
        mean_confidence = np.mean(confidences)
        tc_std = np.std(tc_estimates)
        tc_mean = np.mean(tc_estimates)
        
        # Agreement factor: lower std relative to mean indicates better agreement
        agreement_factor = 1.0 - min(1.0, tc_std / abs(tc_mean)) if tc_mean != 0 else 0.5
        
        # Combine mean confidence with agreement factor
        confidence_ensemble = mean_confidence * agreement_factor
        
        self.logger.info(f"Ensemble Tc={tc_ensemble:.4f}, confidence={confidence_ensemble:.3f} from {len(tc_estimates)} methods")
        
        return tc_ensemble, confidence_ensemble
    
    def _derivative_method(self, temperatures: np.ndarray, order_parameter: np.ndarray) -> Tuple[float, float]:
        """Detect Tc using maximum derivative of order parameter."""
        # Smooth the data first
        if len(order_parameter) > self.smoothing_window:
            smoothed_op = savgol_filter(order_parameter, self.smoothing_window, 3)
        else:
            smoothed_op = order_parameter
        
        # Calculate derivative
        derivative = np.gradient(smoothed_op, temperatures)
        
        # Find minimum (most negative) derivative
        min_idx = np.argmin(derivative)
        tc = temperatures[min_idx]
        
        # Estimate confidence based on sharpness of minimum
        derivative_range = np.max(derivative) - np.min(derivative)
        confidence = abs(derivative[min_idx]) / derivative_range if derivative_range > 0 else 0.5
        
        return tc, confidence
    
    def _susceptibility_method(self, temperatures: np.ndarray, order_parameter: np.ndarray) -> Tuple[float, float]:
        """Detect Tc using susceptibility (variance) maximum."""
        # Calculate susceptibility as variance in temperature bins
        n_bins = min(20, len(temperatures) // 5)
        temp_bins = np.linspace(np.min(temperatures), np.max(temperatures), n_bins + 1)
        temp_centers = (temp_bins[:-1] + temp_bins[1:]) / 2
        
        susceptibilities = []
        for i in range(len(temp_bins) - 1):
            mask = (temperatures >= temp_bins[i]) & (temperatures < temp_bins[i + 1])
            if np.sum(mask) > 1:
                susceptibilities.append(np.var(order_parameter[mask]))
            else:
                susceptibilities.append(0)
        
        susceptibilities = np.array(susceptibilities)
        
        if len(susceptibilities) == 0:
            return np.mean(temperatures), 0.1
        
        # Find maximum susceptibility
        max_idx = np.argmax(susceptibilities)
        tc = temp_centers[max_idx]
        
        # Confidence based on peak sharpness
        max_susc = susceptibilities[max_idx]
        mean_susc = np.mean(susceptibilities)
        confidence = (max_susc - mean_susc) / max_susc if max_susc > 0 else 0.1
        
        return tc, confidence
    
    def _variance_method(self, temperatures: np.ndarray, order_parameter: np.ndarray) -> Tuple[float, float]:
        """Detect Tc using variance (fluctuation) maximum."""
        # Calculate variance in temperature bins
        n_bins = min(25, len(temperatures) // 5)
        temp_bins = np.linspace(np.min(temperatures), np.max(temperatures), n_bins + 1)
        temp_centers = (temp_bins[:-1] + temp_bins[1:]) / 2
        
        variances = []
        for i in range(len(temp_bins) - 1):
            mask = (temperatures >= temp_bins[i]) & (temperatures < temp_bins[i + 1])
            if np.sum(mask) > 2:
                variances.append(np.var(order_parameter[mask]))
            else:
                variances.append(0)
        
        variances = np.array(variances)
        
        if len(variances) == 0 or np.max(variances) == 0:
            return np.mean(temperatures), 0.1
        
        # Smooth variances to reduce noise
        if len(variances) > self.smoothing_window:
            variances_smooth = savgol_filter(variances, min(self.smoothing_window, len(variances) - 1 if len(variances) % 2 == 0 else len(variances)), 3)
        else:
            variances_smooth = variances
        
        # Find maximum variance
        max_idx = np.argmax(variances_smooth)
        tc = temp_centers[max_idx]
        
        # Confidence based on peak prominence
        max_var = variances_smooth[max_idx]
        mean_var = np.mean(variances_smooth)
        std_var = np.std(variances_smooth)
        
        # Peak prominence relative to background
        prominence = (max_var - mean_var) / std_var if std_var > 0 else 0
        confidence = min(1.0, prominence / 3.0)  # Normalize to [0, 1]
        confidence = max(0.1, confidence)  # Minimum confidence
        
        return tc, confidence
    
    def _inflection_method(self, temperatures: np.ndarray, order_parameter: np.ndarray) -> Tuple[float, float]:
        """Detect Tc using inflection point (maximum second derivative)."""
        # Smooth the data first
        if len(order_parameter) > self.smoothing_window:
            smoothed_op = savgol_filter(order_parameter, self.smoothing_window, 3)
        else:
            smoothed_op = order_parameter
        
        # Calculate first derivative
        first_deriv = np.gradient(smoothed_op, temperatures)
        
        # Calculate second derivative
        second_deriv = np.gradient(first_deriv, temperatures)
        
        # Find maximum absolute second derivative (inflection point)
        abs_second_deriv = np.abs(second_deriv)
        max_idx = np.argmax(abs_second_deriv)
        tc = temperatures[max_idx]
        
        # Confidence based on sharpness of inflection
        max_second_deriv = abs_second_deriv[max_idx]
        mean_second_deriv = np.mean(abs_second_deriv)
        std_second_deriv = np.std(abs_second_deriv)
        
        # Sharpness relative to background
        sharpness = (max_second_deriv - mean_second_deriv) / std_second_deriv if std_second_deriv > 0 else 0
        confidence = min(1.0, sharpness / 4.0)  # Normalize to [0, 1]
        confidence = max(0.1, confidence)  # Minimum confidence
        
        return tc, confidence
    
    def _binder_cumulant_method(self, temperatures: np.ndarray, order_parameter: np.ndarray) -> Tuple[float, float]:
        """Detect Tc using Binder cumulant crossing (simplified version)."""
        # Calculate fourth-order Binder cumulant
        n_bins = min(15, len(temperatures) // 4)
        temp_bins = np.linspace(np.min(temperatures), np.max(temperatures), n_bins + 1)
        temp_centers = (temp_bins[:-1] + temp_bins[1:]) / 2
        
        binder_values = []
        for i in range(len(temp_bins) - 1):
            mask = (temperatures >= temp_bins[i]) & (temperatures < temp_bins[i + 1])
            if np.sum(mask) > 3:
                op_bin = order_parameter[mask]
                m2 = np.mean(op_bin**2)
                m4 = np.mean(op_bin**4)
                if m2 > 0:
                    binder = 1 - m4 / (3 * m2**2)
                    binder_values.append(binder)
                else:
                    binder_values.append(0)
            else:
                binder_values.append(0)
        
        binder_values = np.array(binder_values)
        
        if len(binder_values) < 3:
            return np.mean(temperatures), 0.1
        
        # Find where Binder cumulant crosses universal value (≈ 0.61 for 3D Ising)
        universal_value = 0.61
        crossings = []
        
        for i in range(len(binder_values) - 1):
            if ((binder_values[i] - universal_value) * (binder_values[i + 1] - universal_value)) < 0:
                # Linear interpolation for crossing point
                t1, t2 = temp_centers[i], temp_centers[i + 1]
                b1, b2 = binder_values[i], binder_values[i + 1]
                tc_cross = t1 + (universal_value - b1) * (t2 - t1) / (b2 - b1)
                crossings.append(tc_cross)
        
        if crossings:
            tc = np.mean(crossings)
            confidence = 0.8  # High confidence for Binder cumulant method
        else:
            # Fallback: find minimum deviation from universal value
            deviations = np.abs(binder_values - universal_value)
            min_idx = np.argmin(deviations)
            tc = temp_centers[min_idx]
            confidence = 1 - deviations[min_idx]  # Confidence based on closeness
        
        return tc, confidence
    
    def estimate_tc_with_confidence(self, 
                                   temperatures: np.ndarray,
                                   order_parameter: np.ndarray,
                                   method: str = 'ensemble') -> Dict[str, Any]:
        """
        Estimate critical temperature with comprehensive confidence metrics.
        
        Args:
            temperatures: Temperature array
            order_parameter: Order parameter values
            method: Detection method to use
            
        Returns:
            Dictionary with Tc estimate, confidence, and detailed metrics
        """
        self.logger.info(f"Estimating Tc with confidence using {method} method")
        
        # Get primary estimate
        tc_primary, confidence_primary = self.detect_critical_temperature(
            temperatures, order_parameter, method
        )
        
        # Bootstrap confidence interval
        tc_bootstrap = []
        rng = np.random.RandomState(42)
        
        n_data = len(temperatures)
        for _ in range(self.bootstrap_samples):
            # Bootstrap resample
            indices = rng.choice(n_data, size=n_data, replace=True)
            boot_temps = temperatures[indices]
            boot_op = order_parameter[indices]
            
            try:
                tc_boot, _ = self.detect_critical_temperature(boot_temps, boot_op, method)
                if np.isfinite(tc_boot):
                    tc_bootstrap.append(tc_boot)
            except Exception:
                continue
        
        # Calculate bootstrap statistics
        if len(tc_bootstrap) > 10:
            tc_bootstrap = np.array(tc_bootstrap)
            tc_mean = np.mean(tc_bootstrap)
            tc_std = np.std(tc_bootstrap)
            tc_ci_lower = np.percentile(tc_bootstrap, 2.5)
            tc_ci_upper = np.percentile(tc_bootstrap, 97.5)
            
            # Bootstrap confidence based on CI width
            ci_width = tc_ci_upper - tc_ci_lower
            relative_ci_width = ci_width / abs(tc_mean) if tc_mean != 0 else 1.0
            bootstrap_confidence = max(0.1, 1.0 - min(1.0, relative_ci_width))
        else:
            tc_mean = tc_primary
            tc_std = 0.0
            tc_ci_lower = tc_primary
            tc_ci_upper = tc_primary
            bootstrap_confidence = 0.5
        
        # Get individual method estimates for ensemble
        if method == 'ensemble':
            method_estimates = {}
            for m in ['derivative', 'susceptibility', 'variance', 'inflection']:
                try:
                    tc_m, conf_m = getattr(self, f'_{m}_method')(temperatures, order_parameter)
                    if np.isfinite(tc_m):
                        method_estimates[m] = {'tc': tc_m, 'confidence': conf_m}
                except Exception:
                    continue
            
            # Method agreement score
            if len(method_estimates) > 1:
                tc_values = [v['tc'] for v in method_estimates.values()]
                agreement_std = np.std(tc_values)
                agreement_mean = np.mean(tc_values)
                agreement_score = 1.0 - min(1.0, agreement_std / abs(agreement_mean)) if agreement_mean != 0 else 0.5
            else:
                agreement_score = 0.5
        else:
            method_estimates = {method: {'tc': tc_primary, 'confidence': confidence_primary}}
            agreement_score = 1.0
        
        # Combined confidence score
        combined_confidence = (confidence_primary * 0.4 + 
                             bootstrap_confidence * 0.4 + 
                             agreement_score * 0.2)
        
        result = {
            'tc_estimate': tc_primary,
            'tc_mean_bootstrap': tc_mean,
            'tc_std_bootstrap': tc_std,
            'tc_ci_lower': tc_ci_lower,
            'tc_ci_upper': tc_ci_upper,
            'confidence_primary': confidence_primary,
            'confidence_bootstrap': bootstrap_confidence,
            'confidence_combined': combined_confidence,
            'method_estimates': method_estimates,
            'agreement_score': agreement_score,
            'n_bootstrap_samples': len(tc_bootstrap)
        }
        
        self.logger.info(f"Tc estimate: {tc_primary:.4f} ± {tc_std:.4f}, "
                        f"CI: [{tc_ci_lower:.4f}, {tc_ci_upper:.4f}], "
                        f"confidence: {combined_confidence:.3f}")
        
        return result


class ImprovedPowerLawFitter:
    """
    Enhanced power-law fitting with better preprocessing and accuracy.
    """
    
    def __init__(self, 
                 min_points: int = 8,
                 bootstrap_samples: int = 2000,
                 random_seed: Optional[int] = None):
        """Initialize improved power-law fitter."""
        self.min_points = min_points
        self.bootstrap_samples = bootstrap_samples
        self.random_seed = random_seed
        self.logger = get_logger(__name__)
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def fit_power_law_improved(self,
                              temperatures: np.ndarray,
                              observable: np.ndarray,
                              critical_temperature: float,
                              exponent_type: str = 'beta',
                              adaptive_range: bool = True) -> ImprovedPowerLawFitResult:
        """
        Improved power-law fitting with adaptive range selection and better preprocessing.
        
        Args:
            temperatures: Temperature array
            observable: Observable values
            critical_temperature: Critical temperature
            exponent_type: Type of exponent ('beta', 'nu', 'gamma')
            adaptive_range: Whether to use adaptive fitting range
            
        Returns:
            ImprovedPowerLawFitResult with enhanced accuracy metrics
        """
        self.logger.info(f"Improved power-law fitting for {exponent_type} exponent")
        
        # Preprocess data
        temps_clean, obs_clean = self._preprocess_data(temperatures, observable, critical_temperature)
        
        if len(temps_clean) < self.min_points:
            raise CriticalExponentError(f"Insufficient clean data points: {len(temps_clean)}")
        
        # Select optimal fitting range
        if adaptive_range:
            fit_range = self._select_optimal_range(temps_clean, obs_clean, critical_temperature, exponent_type)
        else:
            fit_range = self._default_range(temps_clean, critical_temperature, exponent_type)
        
        # Apply fitting range
        range_mask = (temps_clean >= fit_range[0]) & (temps_clean <= fit_range[1])
        fit_temps = temps_clean[range_mask]
        fit_obs = obs_clean[range_mask]
        
        if len(fit_temps) < self.min_points:
            raise CriticalExponentError(f"Insufficient points in fitting range: {len(fit_temps)}")
        
        # Perform multiple fitting methods and select best
        results = []
        
        # Method 1: Weighted least squares in log space
        try:
            result1 = self._weighted_log_fit(fit_temps, fit_obs, critical_temperature, exponent_type)
            results.append(('weighted_log', result1))
        except Exception as e:
            self.logger.warning(f"Weighted log fit failed: {e}")
        
        # Method 2: Robust nonlinear fit
        try:
            result2 = self._robust_nonlinear_fit(fit_temps, fit_obs, critical_temperature, exponent_type)
            results.append(('robust_nonlinear', result2))
        except Exception as e:
            self.logger.warning(f"Robust nonlinear fit failed: {e}")
        
        # Method 3: Orthogonal distance regression
        try:
            result3 = self._orthogonal_regression_fit(fit_temps, fit_obs, critical_temperature, exponent_type)
            results.append(('orthogonal', result3))
        except Exception as e:
            self.logger.warning(f"Orthogonal regression failed: {e}")
        
        if not results:
            raise CriticalExponentError("All fitting methods failed")
        
        # Select best result based on multiple criteria
        best_result = self._select_best_fit(results)
        
        # Add metadata
        best_result.fit_range = fit_range
        best_result.critical_temperature_used = critical_temperature
        best_result.data_quality_score = self._compute_data_quality_score(fit_temps, fit_obs)
        
        # Compute enhanced confidence interval
        try:
            best_result.confidence_interval = self._compute_enhanced_bootstrap_ci(
                fit_temps, fit_obs, critical_temperature, exponent_type
            )
        except Exception as e:
            self.logger.warning(f"Enhanced bootstrap CI failed: {e}")
        
        return best_result
    
    def _preprocess_data(self, temperatures: np.ndarray, observable: np.ndarray, 
                        critical_temperature: float) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data to remove outliers and improve quality."""
        # Remove NaN and infinite values
        valid_mask = np.isfinite(temperatures) & np.isfinite(observable)
        temps = temperatures[valid_mask]
        obs = observable[valid_mask]
        
        # Remove zero or negative observables for log fitting
        positive_mask = obs > 0
        temps = temps[positive_mask]
        obs = obs[positive_mask]
        
        # Remove outliers using modified Z-score
        if len(obs) > 5:
            median_obs = np.median(obs)
            mad_obs = np.median(np.abs(obs - median_obs))
            
            if mad_obs > 0:
                modified_z_scores = 0.6745 * (obs - median_obs) / mad_obs
                outlier_mask = np.abs(modified_z_scores) < 3.5  # Keep non-outliers
                temps = temps[outlier_mask]
                obs = obs[outlier_mask]
        
        # Sort by temperature
        sort_idx = np.argsort(temps)
        temps = temps[sort_idx]
        obs = obs[sort_idx]
        
        return temps, obs
    
    def _select_optimal_range(self, temperatures: np.ndarray, observable: np.ndarray,
                             critical_temperature: float, exponent_type: str) -> Tuple[float, float]:
        """Select optimal fitting range based on data quality and physics."""
        
        if exponent_type == 'beta':
            # For β: use temperatures below Tc
            below_tc = temperatures < critical_temperature
            if np.sum(below_tc) < self.min_points:
                # Extend slightly above Tc if needed
                temp_range = np.max(temperatures) - np.min(temperatures)
                t_max = critical_temperature + 0.1 * temp_range
            else:
                t_max = critical_temperature
            
            # Find optimal lower bound by testing different ranges
            temp_diffs = critical_temperature - temperatures[temperatures < critical_temperature]
            if len(temp_diffs) > 0:
                # Use range where power law is most linear in log space
                optimal_range = self._find_most_linear_range(temperatures, observable, critical_temperature, 'below')
                t_min = max(np.min(temperatures), critical_temperature - optimal_range)
            else:
                t_min = np.min(temperatures)
                
        elif exponent_type in ['nu', 'gamma']:
            # For ν and γ: use temperatures on both sides of Tc
            temp_range = np.max(temperatures) - np.min(temperatures)
            optimal_range = self._find_most_linear_range(temperatures, observable, critical_temperature, 'both')
            
            t_min = max(np.min(temperatures), critical_temperature - optimal_range)
            t_max = min(np.max(temperatures), critical_temperature + optimal_range)
        
        else:
            # Default range
            temp_range = np.max(temperatures) - np.min(temperatures)
            t_min = critical_temperature - 0.2 * temp_range
            t_max = critical_temperature + 0.2 * temp_range
        
        return (t_min, t_max)
    
    def _find_most_linear_range(self, temperatures: np.ndarray, observable: np.ndarray,
                               critical_temperature: float, side: str) -> float:
        """Find the temperature range where power law is most linear."""
        
        # Test different ranges
        temp_range = np.max(temperatures) - np.min(temperatures)
        test_ranges = np.linspace(0.05 * temp_range, 0.4 * temp_range, 10)
        
        best_r_squared = -1
        best_range = 0.2 * temp_range
        
        for test_range in test_ranges:
            try:
                if side == 'below':
                    mask = (temperatures >= critical_temperature - test_range) & (temperatures < critical_temperature)
                elif side == 'above':
                    mask = (temperatures > critical_temperature) & (temperatures <= critical_temperature + test_range)
                else:  # both
                    mask = (temperatures >= critical_temperature - test_range) & (temperatures <= critical_temperature + test_range)
                
                if np.sum(mask) < self.min_points:
                    continue
                
                test_temps = temperatures[mask]
                test_obs = observable[mask]
                
                # Quick linear fit in log space
                reduced_temps = np.abs(test_temps - critical_temperature)
                reduced_temps = np.maximum(reduced_temps, 1e-10)
                
                log_temps = safe_log(reduced_temps)
                log_obs = safe_log(test_obs)
                
                slope, intercept, r_value, p_value, std_err = linregress(log_temps, log_obs)
                
                if r_value**2 > best_r_squared:
                    best_r_squared = r_value**2
                    best_range = test_range
                    
            except Exception:
                continue
        
        return best_range
    
    def _weighted_log_fit(self, temperatures: np.ndarray, observable: np.ndarray,
                         critical_temperature: float, exponent_type: str) -> ImprovedPowerLawFitResult:
        """Weighted least squares fit in log space with error weighting."""
        
        reduced_temp = np.abs(temperatures - critical_temperature)
        reduced_temp = np.maximum(reduced_temp, 1e-10)
        
        log_reduced_temp = safe_log(reduced_temp)
        log_obs = safe_log(observable)
        
        # Compute weights based on distance from Tc and observable magnitude
        temp_weights = 1.0 / (1.0 + (reduced_temp / np.std(reduced_temp))**2)
        obs_weights = 1.0 / (1.0 + np.abs(log_obs - np.mean(log_obs)))
        weights = temp_weights * obs_weights
        weights = weights / np.sum(weights) * len(weights)  # Normalize
        
        # Weighted linear regression
        W = np.diag(weights)
        X = np.column_stack([np.ones(len(log_reduced_temp)), log_reduced_temp])
        y = log_obs
        
        # Solve weighted least squares: (X^T W X)^-1 X^T W y
        XTW = X.T @ W
        XTWX = XTW @ X
        XTWy = XTW @ y
        
        params = np.linalg.solve(XTWX, XTWy)
        intercept, slope = params
        
        # Calculate errors
        y_pred = X @ params
        residuals = y - y_pred
        weighted_residuals = np.sqrt(weights) * residuals
        
        # Covariance matrix
        mse = np.sum(weighted_residuals**2) / (len(y) - 2)
        cov_matrix = mse * np.linalg.inv(XTWX)
        
        intercept_error = np.sqrt(cov_matrix[0, 0])
        slope_error = np.sqrt(cov_matrix[1, 1])
        
        # R-squared for weighted regression
        ss_res = np.sum(weighted_residuals**2)
        y_mean = np.average(y, weights=weights)
        ss_tot = np.sum(weights * (y - y_mean)**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # P-value approximation
        t_stat = abs(slope) / slope_error if slope_error > 0 else 0
        p_value = 2 * (1 - 0.5 * (1 + np.tanh(t_stat - 2)))  # Rough approximation
        
        return ImprovedPowerLawFitResult(
            exponent=slope,
            amplitude=np.exp(intercept),
            exponent_error=slope_error,
            amplitude_error=np.exp(intercept) * intercept_error,
            r_squared=r_squared,
            p_value=p_value,
            fit_range=(0, 0),  # Will be set later
            residuals=residuals
        )
    
    def _robust_nonlinear_fit(self, temperatures: np.ndarray, observable: np.ndarray,
                             critical_temperature: float, exponent_type: str) -> ImprovedPowerLawFitResult:
        """Robust nonlinear fitting with outlier resistance."""
        
        def power_law_func(t, amplitude, exponent):
            reduced_temp = np.abs(t - critical_temperature)
            reduced_temp = np.maximum(reduced_temp, 1e-10)
            return amplitude * (reduced_temp ** exponent)
        
        # Initial guess based on log fit
        try:
            reduced_temp = np.abs(temperatures - critical_temperature)
            reduced_temp = np.maximum(reduced_temp, 1e-10)
            log_temps = safe_log(reduced_temp)
            log_obs = safe_log(observable)
            slope, intercept, _, _, _ = linregress(log_temps, log_obs)
            initial_amplitude = np.exp(intercept)
            initial_exponent = slope
        except:
            initial_amplitude = np.median(observable)
            initial_exponent = -0.5 if exponent_type == 'beta' else -1.0
        
        # Bounds based on physics with better numerical stability
        if exponent_type == 'beta':
            bounds = ([1e-6, -3], [1e3, 3])  # More restrictive bounds
        elif exponent_type == 'nu':
            bounds = ([1e-6, -3], [1e3, 0])
        else:  # gamma
            bounds = ([1e-6, -3], [1e3, 0])
        
        try:
            # Use robust loss function (Huber loss equivalent)
            def robust_residuals(params):
                amplitude, exponent = params
                predicted = power_law_func(temperatures, amplitude, exponent)
                residuals = observable - predicted
                # Huber loss
                delta = np.std(residuals)
                huber_loss = np.where(np.abs(residuals) <= delta,
                                     0.5 * residuals**2,
                                     delta * (np.abs(residuals) - 0.5 * delta))
                return np.sum(huber_loss)
            
            # Use differential evolution for global optimization
            result = differential_evolution(
                robust_residuals,
                bounds,
                seed=self.random_seed,
                maxiter=1000,
                atol=1e-8
            )
            
            if not result.success:
                raise RuntimeError("Optimization failed")
            
            amplitude, exponent = result.x
            
            # Estimate errors using finite differences
            def objective(params):
                return robust_residuals(params)
            
            # Numerical Hessian approximation
            eps = 1e-6
            hess = np.zeros((2, 2))
            
            for i in range(2):
                for j in range(2):
                    params_pp = result.x.copy()
                    params_pm = result.x.copy()
                    params_mp = result.x.copy()
                    params_mm = result.x.copy()
                    
                    params_pp[i] += eps
                    params_pp[j] += eps
                    params_pm[i] += eps
                    params_pm[j] -= eps
                    params_mp[i] -= eps
                    params_mp[j] += eps
                    params_mm[i] -= eps
                    params_mm[j] -= eps
                    
                    hess[i, j] = (objective(params_pp) - objective(params_pm) - 
                                 objective(params_mp) + objective(params_mm)) / (4 * eps**2)
            
            # Covariance matrix (inverse Hessian)
            try:
                cov_matrix = np.linalg.inv(hess)
                amplitude_error = np.sqrt(abs(cov_matrix[0, 0]))
                exponent_error = np.sqrt(abs(cov_matrix[1, 1]))
            except:
                amplitude_error = 0.1 * amplitude
                exponent_error = 0.1 * abs(exponent)
            
            # Calculate R-squared
            predicted = power_law_func(temperatures, amplitude, exponent)
            ss_res = np.sum((observable - predicted)**2)
            ss_tot = np.sum((observable - np.mean(observable))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Residuals
            residuals = observable - predicted
            
            # P-value approximation
            t_stat = abs(exponent) / exponent_error if exponent_error > 0 else 0
            p_value = 2 * (1 - 0.5 * (1 + np.tanh(t_stat - 2)))
            
            return ImprovedPowerLawFitResult(
                exponent=exponent,
                amplitude=amplitude,
                exponent_error=exponent_error,
                amplitude_error=amplitude_error,
                r_squared=r_squared,
                p_value=p_value,
                fit_range=(0, 0),
                residuals=residuals
            )
            
        except Exception as e:
            raise CriticalExponentError(f"Robust nonlinear fit failed: {e}")
    
    def _orthogonal_regression_fit(self, temperatures: np.ndarray, observable: np.ndarray,
                                  critical_temperature: float, exponent_type: str) -> ImprovedPowerLawFitResult:
        """Orthogonal distance regression for better handling of errors in both variables."""
        
        # For now, implement a simplified version using total least squares
        reduced_temp = np.abs(temperatures - critical_temperature)
        reduced_temp = np.maximum(reduced_temp, 1e-10)
        
        log_reduced_temp = safe_log(reduced_temp)
        log_obs = safe_log(observable)
        
        # Center the data
        x_mean = np.mean(log_reduced_temp)
        y_mean = np.mean(log_obs)
        
        x_centered = log_reduced_temp - x_mean
        y_centered = log_obs - y_mean
        
        # SVD for total least squares
        A = np.column_stack([x_centered, y_centered])
        U, s, Vt = np.linalg.svd(A)
        
        # The slope is -V[0,1]/V[1,1] where V is the last column of Vt
        V = Vt.T
        slope = -V[0, -1] / V[1, -1]
        intercept = y_mean - slope * x_mean
        
        # Calculate errors (simplified)
        y_pred = intercept + slope * log_reduced_temp
        residuals = log_obs - y_pred
        
        n = len(log_reduced_temp)
        mse = np.sum(residuals**2) / (n - 2)
        
        # Standard error of slope
        x_var = np.sum((log_reduced_temp - x_mean)**2)
        slope_error = np.sqrt(mse / x_var) if x_var > 0 else 0.1 * abs(slope)
        
        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((log_obs - y_mean)**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # P-value
        t_stat = abs(slope) / slope_error if slope_error > 0 else 0
        p_value = 2 * (1 - 0.5 * (1 + np.tanh(t_stat - 2)))
        
        return ImprovedPowerLawFitResult(
            exponent=slope,
            amplitude=np.exp(intercept),
            exponent_error=slope_error,
            amplitude_error=np.exp(intercept) * np.sqrt(mse),
            r_squared=r_squared,
            p_value=p_value,
            fit_range=(0, 0),
            residuals=residuals
        )
    
    def _select_best_fit(self, results: List[Tuple[str, ImprovedPowerLawFitResult]]) -> ImprovedPowerLawFitResult:
        """Select best fit based on multiple criteria."""
        
        if len(results) == 1:
            return results[0][1]
        
        # Score each result
        scores = []
        for method, result in results:
            score = 0
            
            # R-squared contribution (40%)
            score += 0.4 * max(0, result.r_squared)
            
            # P-value contribution (20%) - prefer significant results
            score += 0.2 * (1 - result.p_value) if result.p_value < 0.05 else 0
            
            # Relative error contribution (20%) - prefer smaller relative errors
            rel_error = result.exponent_error / abs(result.exponent) if result.exponent != 0 else 1
            score += 0.2 * max(0, 1 - rel_error)
            
            # Physics reasonableness (20%) - prefer physically reasonable exponents
            if -3 < result.exponent < 3:  # Reasonable range for critical exponents
                score += 0.2
            
            scores.append(score)
        
        # Select result with highest score
        best_idx = np.argmax(scores)
        return results[best_idx][1]
    
    def _compute_data_quality_score(self, temperatures: np.ndarray, observable: np.ndarray) -> float:
        """Compute a data quality score based on various metrics."""
        
        score = 0
        
        # Number of points (normalized to 0-1)
        n_points = len(temperatures)
        score += min(1.0, n_points / 20) * 0.3
        
        # Temperature range coverage
        temp_range = np.max(temperatures) - np.min(temperatures)
        temp_std = np.std(temperatures)
        coverage = temp_range / temp_std if temp_std > 0 else 0
        score += min(1.0, coverage / 5) * 0.3
        
        # Observable signal-to-noise ratio
        obs_mean = np.mean(observable)
        obs_std = np.std(observable)
        snr = obs_mean / obs_std if obs_std > 0 else 0
        score += min(1.0, snr / 10) * 0.4
        
        return score
    
    def _compute_enhanced_bootstrap_ci(self, temperatures: np.ndarray, observable: np.ndarray,
                                     critical_temperature: float, exponent_type: str) -> ConfidenceInterval:
        """Enhanced bootstrap confidence interval with bias correction."""
        
        n_data = len(temperatures)
        bootstrap_exponents = []
        
        rng = np.random.RandomState(self.random_seed)
        
        for _ in range(self.bootstrap_samples):
            # Bootstrap resample
            indices = rng.choice(n_data, size=n_data, replace=True)
            boot_temps = temperatures[indices]
            boot_obs = observable[indices]
            
            try:
                # Use the weighted log fit for bootstrap (fastest and most stable)
                result = self._weighted_log_fit(boot_temps, boot_obs, critical_temperature, exponent_type)
                bootstrap_exponents.append(result.exponent)
            except Exception:
                continue
        
        if len(bootstrap_exponents) < self.bootstrap_samples * 0.5:
            raise CriticalExponentError("Too many bootstrap failures")
        
        bootstrap_exponents = np.array(bootstrap_exponents)
        
        # Bias-corrected percentile method
        alpha = 0.05  # 95% confidence interval
        
        # Compute bias correction
        original_result = self._weighted_log_fit(temperatures, observable, critical_temperature, exponent_type)
        original_exponent = original_result.exponent
        
        n_less = np.sum(bootstrap_exponents < original_exponent)
        bias_correction = n_less / len(bootstrap_exponents)
        
        # Adjust percentiles for bias
        lower_percentile = max(0.1, min(99.9, (alpha / 2 + bias_correction) * 100))
        upper_percentile = max(0.1, min(99.9, (1 - alpha / 2 + bias_correction) * 100))
        
        lower_bound = np.percentile(bootstrap_exponents, lower_percentile)
        upper_bound = np.percentile(bootstrap_exponents, upper_percentile)
        
        return ConfidenceInterval(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=0.95,
            method="bootstrap_bias_corrected",
            n_bootstrap_samples=len(bootstrap_exponents),
            bias_correction=bias_correction
        )
    
    def _default_range(self, temperatures: np.ndarray, critical_temperature: float, 
                      exponent_type: str) -> Tuple[float, float]:
        """Default fitting range when adaptive selection fails."""
        temp_range = np.max(temperatures) - np.min(temperatures)
        
        if exponent_type == 'beta':
            t_min = critical_temperature - 0.3 * temp_range
            t_max = critical_temperature
        else:
            t_min = critical_temperature - 0.2 * temp_range
            t_max = critical_temperature + 0.2 * temp_range
        
        return (max(t_min, np.min(temperatures)), min(t_max, np.max(temperatures)))


def create_improved_critical_exponent_analyzer(system_type: str = 'ising_3d',
                                             bootstrap_samples: int = 2000,
                                             random_seed: Optional[int] = None) -> 'ImprovedCriticalExponentAnalyzer':
    """
    Factory function to create an improved CriticalExponentAnalyzer.
    
    Args:
        system_type: Type of physical system
        bootstrap_samples: Number of bootstrap samples
        random_seed: Random seed for reproducibility
        
    Returns:
        Configured ImprovedCriticalExponentAnalyzer instance
    """
    
    tc_detector = ImprovedCriticalTemperatureDetector()
    fitter = ImprovedPowerLawFitter(bootstrap_samples=bootstrap_samples, random_seed=random_seed)
    
    return ImprovedCriticalExponentAnalyzer(tc_detector, fitter, system_type)


class ImprovedCriticalExponentAnalyzer:
    """
    Enhanced critical exponent analyzer with improved accuracy.
    """
    
    def __init__(self, 
                 tc_detector: ImprovedCriticalTemperatureDetector,
                 fitter: ImprovedPowerLawFitter,
                 system_type: str = 'ising_3d'):
        """Initialize improved analyzer."""
        self.tc_detector = tc_detector
        self.fitter = fitter
        self.system_type = system_type
        self.logger = get_logger(__name__)
        
        # Theoretical exponents
        self.theoretical_exponents = {
            'ising_2d': {'beta': 0.125, 'nu': 1.0, 'gamma': 1.75},
            'ising_3d': {'beta': 0.326, 'nu': 0.630, 'gamma': 1.237},
            'xy_2d': {'beta': 0.125, 'nu': 1.0, 'gamma': 1.75},
            'heisenberg_3d': {'beta': 0.365, 'nu': 0.705, 'gamma': 1.386}
        }
    
    def analyze_with_improved_accuracy(self, 
                                     latent_repr: LatentRepresentation,
                                     auto_detect_tc: bool = True) -> Dict[str, Any]:
        """
        Perform improved critical exponent analysis with enhanced accuracy.
        
        Args:
            latent_repr: LatentRepresentation object
            auto_detect_tc: Whether to automatically detect critical temperature
            
        Returns:
            Dictionary with improved analysis results
        """
        self.logger.info("Starting improved critical exponent analysis")
        
        results = {
            'system_type': self.system_type,
            'theoretical_exponents': self.theoretical_exponents.get(self.system_type, {}),
            'extracted_exponents': {},
            'accuracy_metrics': {},
            'critical_temperature': None
        }
        
        # Step 1: Improved critical temperature detection
        if auto_detect_tc:
            tc, tc_confidence = self.tc_detector.detect_critical_temperature(
                latent_repr.temperatures, 
                np.abs(latent_repr.magnetizations),
                method='ensemble'
            )
            results['critical_temperature'] = tc
            results['tc_confidence'] = tc_confidence
            self.logger.info(f"Detected Tc = {tc:.4f} (confidence: {tc_confidence:.3f})")
        else:
            # Use theoretical value
            theoretical_tc = {
                'ising_2d': 2.269,
                'ising_3d': 4.511,
                'xy_2d': 0.893,
                'heisenberg_3d': 1.443
            }
            tc = theoretical_tc.get(self.system_type, 2.269)
            results['critical_temperature'] = tc
            results['tc_confidence'] = 0.9
        
        # Step 2: Extract β exponent with improved method
        try:
            beta_result = self.fitter.fit_power_law_improved(
                latent_repr.temperatures,
                np.abs(latent_repr.magnetizations),
                tc,
                exponent_type='beta',
                adaptive_range=True
            )
            
            results['extracted_exponents']['beta'] = {
                'value': beta_result.exponent,
                'error': beta_result.exponent_error,
                'r_squared': beta_result.r_squared,
                'confidence_interval': (
                    beta_result.confidence_interval.lower_bound,
                    beta_result.confidence_interval.upper_bound
                ) if beta_result.confidence_interval else None,
                'data_quality': beta_result.data_quality_score
            }
            
        except Exception as e:
            self.logger.error(f"β exponent extraction failed: {e}")
            results['extracted_exponents']['beta'] = None
        
        # Step 3: Extract ν exponent using improved correlation length
        try:
            # Compute improved correlation length
            temps_binned, corr_lengths = self._compute_improved_correlation_length(latent_repr, tc)
            
            if len(temps_binned) >= self.fitter.min_points:
                nu_result = self.fitter.fit_power_law_improved(
                    temps_binned,
                    corr_lengths,
                    tc,
                    exponent_type='nu',
                    adaptive_range=True
                )
                
                results['extracted_exponents']['nu'] = {
                    'value': nu_result.exponent,
                    'error': nu_result.exponent_error,
                    'r_squared': nu_result.r_squared,
                    'confidence_interval': (
                        nu_result.confidence_interval.lower_bound,
                        nu_result.confidence_interval.upper_bound
                    ) if nu_result.confidence_interval else None,
                    'data_quality': nu_result.data_quality_score
                }
            else:
                self.logger.warning("Insufficient data for ν exponent")
                results['extracted_exponents']['nu'] = None
                
        except Exception as e:
            self.logger.error(f"ν exponent extraction failed: {e}")
            results['extracted_exponents']['nu'] = None
        
        # Step 4: Compute accuracy metrics
        results['accuracy_metrics'] = self._compute_accuracy_metrics(results)
        
        self.logger.info("Improved critical exponent analysis completed")
        
        return results
    
    def _compute_improved_correlation_length(self, latent_repr: LatentRepresentation, 
                                           critical_temperature: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute improved correlation length using multiple latent dimensions and proper scaling."""
        
        # Use more temperature bins for better resolution
        n_temp_bins = min(30, len(np.unique(latent_repr.temperatures)) // 2)
        temp_min, temp_max = np.min(latent_repr.temperatures), np.max(latent_repr.temperatures)
        temp_bins = np.linspace(temp_min, temp_max, n_temp_bins + 1)
        temp_centers = (temp_bins[:-1] + temp_bins[1:]) / 2
        
        correlation_lengths = []
        valid_temps = []
        
        for i in range(len(temp_bins) - 1):
            temp_mask = (latent_repr.temperatures >= temp_bins[i]) & \
                       (latent_repr.temperatures < temp_bins[i + 1])
            
            if np.sum(temp_mask) >= 8:  # Require more points per bin
                # Use all available latent dimensions
                z1_bin = latent_repr.z1[temp_mask]
                z2_bin = latent_repr.z2[temp_mask]
                
                # Compute correlation length as the characteristic length scale
                # Use the variance and higher moments to estimate correlation length
                
                # Method 1: Combined variance (traditional)
                z1_var = np.var(z1_bin)
                z2_var = np.var(z2_bin)
                
                # Method 2: Spatial correlation estimate
                # Use the spread in latent space as proxy for correlation length
                z_combined = np.column_stack([z1_bin, z2_bin])
                z_center = np.mean(z_combined, axis=0)
                distances = np.linalg.norm(z_combined - z_center, axis=1)
                characteristic_length = np.std(distances)
                
                # Method 3: Susceptibility-based estimate
                # χ ∝ ξ^(2-η) where η is anomalous dimension
                magnetizations_bin = latent_repr.magnetizations[temp_mask]
                susceptibility = np.var(magnetizations_bin)
                
                # Combine methods with physics-motivated weighting
                corr_length_var = np.sqrt(z1_var + z2_var)
                corr_length_spatial = characteristic_length
                corr_length_susc = np.sqrt(susceptibility)
                
                # Weight based on distance from Tc
                temp_center = temp_centers[i]
                distance_from_tc = abs(temp_center - critical_temperature)
                
                # Near Tc, use susceptibility-based; far from Tc, use variance-based
                if distance_from_tc < 0.1 * (temp_max - temp_min):
                    # Near critical point
                    corr_length = 0.5 * corr_length_susc + 0.3 * corr_length_spatial + 0.2 * corr_length_var
                else:
                    # Away from critical point
                    corr_length = 0.4 * corr_length_var + 0.4 * corr_length_spatial + 0.2 * corr_length_susc
                
                if corr_length > 0 and np.isfinite(corr_length):
                    correlation_lengths.append(corr_length)
                    valid_temps.append(temp_center)
        
        return np.array(valid_temps), np.array(correlation_lengths)
    
    def _compute_accuracy_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute comprehensive accuracy metrics."""
        
        accuracy_metrics = {}
        theoretical = results['theoretical_exponents']
        extracted = results['extracted_exponents']
        
        for exponent_name in ['beta', 'nu']:
            if exponent_name in theoretical and extracted.get(exponent_name) is not None:
                theoretical_value = theoretical[exponent_name]
                extracted_data = extracted[exponent_name]
                measured_value = extracted_data['value']
                
                # Relative error
                rel_error = abs(measured_value - theoretical_value) / theoretical_value
                accuracy_metrics[f'{exponent_name}_relative_error'] = rel_error
                accuracy_metrics[f'{exponent_name}_accuracy_percent'] = (1 - rel_error) * 100
                
                # Statistical significance
                if extracted_data.get('confidence_interval'):
                    ci_lower, ci_upper = extracted_data['confidence_interval']
                    ci_contains_theoretical = ci_lower <= theoretical_value <= ci_upper
                    accuracy_metrics[f'{exponent_name}_ci_contains_theoretical'] = ci_contains_theoretical
                    
                    # CI width as measure of precision
                    ci_width = ci_upper - ci_lower
                    relative_ci_width = ci_width / abs(measured_value) if measured_value != 0 else float('inf')
                    accuracy_metrics[f'{exponent_name}_relative_precision'] = relative_ci_width
                
                # Data quality score
                if 'data_quality' in extracted_data:
                    accuracy_metrics[f'{exponent_name}_data_quality'] = extracted_data['data_quality']
        
        # Overall accuracy score
        if 'beta_accuracy_percent' in accuracy_metrics and 'nu_accuracy_percent' in accuracy_metrics:
            overall_accuracy = (accuracy_metrics['beta_accuracy_percent'] + 
                              accuracy_metrics['nu_accuracy_percent']) / 2
            accuracy_metrics['overall_accuracy_percent'] = overall_accuracy
        
        return accuracy_metrics