"""
Robust Power-Law Fitting System

This module implements task 8.1: Fix power-law fitting numerical instabilities
by resolving optimization bounds issues, implementing robust parameter initialization,
and adding fallback fitting methods.
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
import logging
from scipy.optimize import curve_fit, minimize_scalar, differential_evolution, least_squares
from scipy.stats import linregress
from scipy import optimize
import matplotlib.pyplot as plt

from .numerical_stability_fixes import safe_log, safe_divide
from ..utils.logging_utils import get_logger


@dataclass
class RobustFitResult:
    """Container for robust power-law fitting results."""
    exponent: float
    amplitude: float
    exponent_error: float
    amplitude_error: float
    r_squared: float
    p_value: float
    fit_range: Tuple[float, float]
    residuals: np.ndarray
    method_used: str
    convergence_info: Dict[str, Any]
    parameter_bounds_used: Tuple[Tuple[float, float], Tuple[float, float]]
    initial_guess: Tuple[float, float]


class RobustPowerLawFitter:
    """
    Robust power-law fitting system that addresses numerical instabilities.
    
    Key improvements:
    1. Proper bounds checking and validation
    2. Multiple fitting methods with fallbacks
    3. Robust parameter initialization
    4. Convergence validation
    5. Parameter reasonableness checks
    """
    
    def __init__(self, 
                 min_points: int = 8,
                 max_iterations: int = 10000,
                 tolerance: float = 1e-8,
                 random_seed: Optional[int] = None):
        """
        Initialize robust power-law fitter.
        
        Args:
            min_points: Minimum number of points required for fitting
            max_iterations: Maximum iterations for optimization
            tolerance: Convergence tolerance
            random_seed: Random seed for reproducibility
        """
        self.min_points = min_points
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_seed = random_seed
        self.logger = get_logger(__name__)
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def fit_power_law_robust(self,
                           temperatures: np.ndarray,
                           observable: np.ndarray,
                           critical_temperature: float,
                           exponent_type: str = 'beta') -> RobustFitResult:
        """
        Robust power-law fitting with comprehensive error handling.
        
        Args:
            temperatures: Temperature array
            observable: Observable values
            critical_temperature: Critical temperature
            exponent_type: Type of exponent ('beta', 'nu', 'gamma')
            
        Returns:
            RobustFitResult with fitting results and diagnostics
        """
        self.logger.info(f"Starting robust power-law fitting for {exponent_type} exponent")
        
        # Validate and preprocess inputs
        temps_clean, obs_clean = self._validate_and_preprocess_data(
            temperatures, observable, critical_temperature, exponent_type
        )
        
        if len(temps_clean) < self.min_points:
            raise ValueError(f"Insufficient data points: {len(temps_clean)} < {self.min_points}")
        
        # Generate robust parameter bounds and initial guess
        bounds, initial_guess = self._generate_robust_bounds_and_guess(
            temps_clean, obs_clean, critical_temperature, exponent_type
        )
        
        # Validate bounds
        self._validate_bounds(bounds, initial_guess)
        
        # Try multiple fitting methods in order of preference
        fitting_methods = [
            ('weighted_least_squares', self._fit_weighted_least_squares),
            ('robust_curve_fit', self._fit_robust_curve_fit),
            ('differential_evolution', self._fit_differential_evolution),
            ('least_squares_robust', self._fit_least_squares_robust),
            ('log_linear_fallback', self._fit_log_linear_fallback)
        ]
        
        best_result = None
        best_score = -np.inf
        
        for method_name, method_func in fitting_methods:
            try:
                self.logger.debug(f"Trying fitting method: {method_name}")
                
                result = method_func(
                    temps_clean, obs_clean, critical_temperature, 
                    exponent_type, bounds, initial_guess
                )
                
                # Validate result
                if self._validate_fit_result(result, exponent_type):
                    # Score the result
                    score = self._score_fit_result(result)
                    
                    if score > best_score:
                        best_score = score
                        best_result = result
                        self.logger.info(f"New best result from {method_name}: score = {score:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Method {method_name} failed: {e}")
                continue
        
        if best_result is None:
            raise RuntimeError("All fitting methods failed")
        
        self.logger.info(f"Best fitting method: {best_result.method_used}")
        self.logger.info(f"Final exponent: {best_result.exponent:.4f} ± {best_result.exponent_error:.4f}")
        
        return best_result
    
    def _validate_and_preprocess_data(self, 
                                    temperatures: np.ndarray,
                                    observable: np.ndarray,
                                    critical_temperature: float,
                                    exponent_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and preprocess input data."""
        
        # Check input arrays
        if len(temperatures) != len(observable):
            raise ValueError("Temperature and observable arrays must have same length")
        
        if not np.all(np.isfinite(temperatures)) or not np.all(np.isfinite(observable)):
            self.logger.warning("Input data contains non-finite values, filtering...")
        
        # Remove non-finite values
        finite_mask = np.isfinite(temperatures) & np.isfinite(observable)
        temps = temperatures[finite_mask]
        obs = observable[finite_mask]
        
        # Remove zero or negative observables (required for log fitting)
        positive_mask = obs > 0
        temps = temps[positive_mask]
        obs = obs[positive_mask]
        
        if len(temps) == 0:
            raise ValueError("No valid positive observable values found")
        
        # For beta exponent, use only temperatures below Tc
        if exponent_type == 'beta':
            below_tc_mask = temps < critical_temperature
            if np.sum(below_tc_mask) >= self.min_points:
                temps = temps[below_tc_mask]
                obs = obs[below_tc_mask]
            else:
                self.logger.warning(f"Few points below Tc for beta fitting: {np.sum(below_tc_mask)}")
        
        # Remove outliers using robust statistics
        temps, obs = self._remove_outliers(temps, obs)
        
        # Sort by temperature
        sort_idx = np.argsort(temps)
        temps = temps[sort_idx]
        obs = obs[sort_idx]
        
        return temps, obs
    
    def _remove_outliers(self, temperatures: np.ndarray, observable: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers using modified Z-score method."""
        
        if len(observable) <= 5:
            return temperatures, observable
        
        # Use modified Z-score (more robust than standard Z-score)
        median_obs = np.median(observable)
        mad_obs = np.median(np.abs(observable - median_obs))
        
        if mad_obs == 0:
            # All values are the same, no outliers to remove
            return temperatures, observable
        
        modified_z_scores = 0.6745 * (observable - median_obs) / mad_obs
        outlier_mask = np.abs(modified_z_scores) < 3.5  # Keep non-outliers
        
        n_removed = len(observable) - np.sum(outlier_mask)
        if n_removed > 0:
            self.logger.debug(f"Removed {n_removed} outliers from {len(observable)} points")
        
        return temperatures[outlier_mask], observable[outlier_mask]
    
    def _generate_robust_bounds_and_guess(self, 
                                        temperatures: np.ndarray,
                                        observable: np.ndarray,
                                        critical_temperature: float,
                                        exponent_type: str) -> Tuple[Tuple[Tuple[float, float], Tuple[float, float]], Tuple[float, float]]:
        """Generate robust parameter bounds and initial guess."""
        
        # Compute reduced temperature
        reduced_temp = np.abs(temperatures - critical_temperature)
        reduced_temp = np.maximum(reduced_temp, 1e-10)  # Avoid zero
        
        # Estimate initial parameters using log-linear fit
        try:
            log_reduced_temp = safe_log(reduced_temp)
            log_obs = safe_log(observable)
            
            slope, intercept, r_value, p_value, std_err = linregress(log_reduced_temp, log_obs)
            
            initial_amplitude = np.exp(intercept)
            initial_exponent = slope
            
        except Exception as e:
            self.logger.warning(f"Log-linear estimation failed: {e}, using default values")
            initial_amplitude = np.median(observable)
            
            # Default exponents based on physics
            if exponent_type == 'beta':
                initial_exponent = -0.3  # Typical beta values are positive, but we fit |T-Tc|^exp
            elif exponent_type == 'nu':
                initial_exponent = -0.6  # Typical nu values
            else:  # gamma
                initial_exponent = -1.2  # Typical gamma values
        
        # Generate bounds based on exponent type and physics constraints
        if exponent_type == 'beta':
            # Beta exponent: m ∝ (Tc - T)^β, β > 0
            # But we fit |T - Tc|^exp, so exp should be positive
            amplitude_bounds = (1e-6, max(1e3, 10 * np.max(observable)))
            exponent_bounds = (0.01, 2.0)  # Physical range for beta
            
        elif exponent_type == 'nu':
            # Nu exponent: ξ ∝ |T - Tc|^(-ν), ν > 0
            # So exp should be negative
            amplitude_bounds = (1e-6, max(1e3, 10 * np.max(observable)))
            exponent_bounds = (-2.0, -0.1)  # Physical range for -nu
            
        else:  # gamma
            # Gamma exponent: χ ∝ |T - Tc|^(-γ), γ > 0
            # So exp should be negative
            amplitude_bounds = (1e-6, max(1e3, 10 * np.max(observable)))
            exponent_bounds = (-3.0, -0.1)  # Physical range for -gamma
        
        # Ensure initial guess is within bounds
        initial_amplitude = np.clip(initial_amplitude, 
                                  amplitude_bounds[0] * 1.1, 
                                  amplitude_bounds[1] * 0.9)
        initial_exponent = np.clip(initial_exponent,
                                 exponent_bounds[0] * 0.9,
                                 exponent_bounds[1] * 0.9)
        
        bounds = (amplitude_bounds, exponent_bounds)
        initial_guess = (initial_amplitude, initial_exponent)
        
        self.logger.debug(f"Generated bounds: amplitude {amplitude_bounds}, exponent {exponent_bounds}")
        self.logger.debug(f"Initial guess: amplitude {initial_amplitude:.4e}, exponent {initial_exponent:.4f}")
        
        return bounds, initial_guess
    
    def _validate_bounds(self, bounds: Tuple[Tuple[float, float], Tuple[float, float]], 
                        initial_guess: Tuple[float, float]) -> None:
        """Validate that bounds are properly formed and initial guess is within bounds."""
        
        amplitude_bounds, exponent_bounds = bounds
        initial_amplitude, initial_exponent = initial_guess
        
        # Check bounds are properly ordered
        if amplitude_bounds[0] >= amplitude_bounds[1]:
            raise ValueError(f"Invalid amplitude bounds: {amplitude_bounds[0]} >= {amplitude_bounds[1]}")
        
        if exponent_bounds[0] >= exponent_bounds[1]:
            raise ValueError(f"Invalid exponent bounds: {exponent_bounds[0]} >= {exponent_bounds[1]}")
        
        # Check initial guess is within bounds
        if not (amplitude_bounds[0] <= initial_amplitude <= amplitude_bounds[1]):
            raise ValueError(f"Initial amplitude {initial_amplitude} not in bounds {amplitude_bounds}")
        
        if not (exponent_bounds[0] <= initial_exponent <= exponent_bounds[1]):
            raise ValueError(f"Initial exponent {initial_exponent} not in bounds {exponent_bounds}")
        
        self.logger.debug("Bounds validation passed")
    
    def _fit_weighted_least_squares(self, 
                                  temperatures: np.ndarray,
                                  observable: np.ndarray,
                                  critical_temperature: float,
                                  exponent_type: str,
                                  bounds: Tuple[Tuple[float, float], Tuple[float, float]],
                                  initial_guess: Tuple[float, float]) -> RobustFitResult:
        """Weighted least squares fitting in log space."""
        
        reduced_temp = np.abs(temperatures - critical_temperature)
        reduced_temp = np.maximum(reduced_temp, 1e-10)
        
        log_reduced_temp = safe_log(reduced_temp)
        log_obs = safe_log(observable)
        
        # Compute weights (inverse variance weighting)
        # Weight by distance from critical temperature and observable magnitude
        temp_weights = 1.0 / (1.0 + (reduced_temp / np.std(reduced_temp))**2)
        obs_weights = 1.0 / (1.0 + np.abs(log_obs - np.mean(log_obs)))
        weights = temp_weights * obs_weights
        weights = weights / np.sum(weights) * len(weights)  # Normalize
        
        # Weighted linear regression
        W = np.diag(weights)
        X = np.column_stack([np.ones(len(log_reduced_temp)), log_reduced_temp])
        y = log_obs
        
        try:
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
            
            intercept_error = np.sqrt(abs(cov_matrix[0, 0]))
            slope_error = np.sqrt(abs(cov_matrix[1, 1]))
            
            # R-squared for weighted regression
            ss_res = np.sum(weighted_residuals**2)
            y_mean = np.average(y, weights=weights)
            ss_tot = np.sum(weights * (y - y_mean)**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # P-value approximation
            t_stat = abs(slope) / slope_error if slope_error > 0 else 0
            p_value = 2 * (1 - 0.5 * (1 + np.tanh(t_stat - 2)))
            
            # Check if result is within bounds
            amplitude = np.exp(intercept)
            exponent = slope
            
            amplitude_bounds, exponent_bounds = bounds
            
            if not (amplitude_bounds[0] <= amplitude <= amplitude_bounds[1]):
                raise ValueError(f"Fitted amplitude {amplitude} outside bounds {amplitude_bounds}")
            
            if not (exponent_bounds[0] <= exponent <= exponent_bounds[1]):
                raise ValueError(f"Fitted exponent {exponent} outside bounds {exponent_bounds}")
            
            return RobustFitResult(
                exponent=exponent,
                amplitude=amplitude,
                exponent_error=slope_error,
                amplitude_error=amplitude * intercept_error,
                r_squared=r_squared,
                p_value=p_value,
                fit_range=(np.min(temperatures), np.max(temperatures)),
                residuals=residuals,
                method_used='weighted_least_squares',
                convergence_info={'converged': True, 'iterations': 1},
                parameter_bounds_used=bounds,
                initial_guess=initial_guess
            )
            
        except Exception as e:
            raise RuntimeError(f"Weighted least squares failed: {e}")
    
    def _fit_robust_curve_fit(self,
                            temperatures: np.ndarray,
                            observable: np.ndarray,
                            critical_temperature: float,
                            exponent_type: str,
                            bounds: Tuple[Tuple[float, float], Tuple[float, float]],
                            initial_guess: Tuple[float, float]) -> RobustFitResult:
        """Robust curve_fit with proper bounds handling."""
        
        def power_law_func(t, amplitude, exponent):
            reduced_temp = np.abs(t - critical_temperature)
            reduced_temp = np.maximum(reduced_temp, 1e-10)
            return amplitude * (reduced_temp ** exponent)
        
        # Convert bounds to scipy format
        amplitude_bounds, exponent_bounds = bounds
        scipy_bounds = ([amplitude_bounds[0], exponent_bounds[0]], 
                       [amplitude_bounds[1], exponent_bounds[1]])
        
        try:
            # Use curve_fit with robust loss function
            popt, pcov = curve_fit(
                power_law_func, 
                temperatures, 
                observable,
                p0=initial_guess,
                bounds=scipy_bounds,
                maxfev=self.max_iterations,
                method='trf',  # Trust Region Reflective algorithm
                loss='soft_l1',  # Robust loss function
                f_scale=np.std(observable)  # Scale for robust loss
            )
            
            amplitude, exponent = popt
            
            # Calculate parameter errors
            param_errors = np.sqrt(np.diag(pcov))
            amplitude_error, exponent_error = param_errors
            
            # Calculate R-squared and residuals
            predicted = power_law_func(temperatures, amplitude, exponent)
            residuals = observable - predicted
            
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((observable - np.mean(observable))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # P-value approximation
            t_stat = abs(exponent) / exponent_error if exponent_error > 0 else 0
            p_value = 2 * (1 - 0.5 * (1 + np.tanh(t_stat - 2)))
            
            return RobustFitResult(
                exponent=exponent,
                amplitude=amplitude,
                exponent_error=exponent_error,
                amplitude_error=amplitude_error,
                r_squared=r_squared,
                p_value=p_value,
                fit_range=(np.min(temperatures), np.max(temperatures)),
                residuals=residuals,
                method_used='robust_curve_fit',
                convergence_info={'converged': True, 'covariance_available': True},
                parameter_bounds_used=bounds,
                initial_guess=initial_guess
            )
            
        except Exception as e:
            raise RuntimeError(f"Robust curve_fit failed: {e}")
    
    def _fit_differential_evolution(self,
                                  temperatures: np.ndarray,
                                  observable: np.ndarray,
                                  critical_temperature: float,
                                  exponent_type: str,
                                  bounds: Tuple[Tuple[float, float], Tuple[float, float]],
                                  initial_guess: Tuple[float, float]) -> RobustFitResult:
        """Global optimization using differential evolution."""
        
        def power_law_func(t, amplitude, exponent):
            reduced_temp = np.abs(t - critical_temperature)
            reduced_temp = np.maximum(reduced_temp, 1e-10)
            return amplitude * (reduced_temp ** exponent)
        
        def objective(params):
            amplitude, exponent = params
            try:
                predicted = power_law_func(temperatures, amplitude, exponent)
                # Use Huber loss for robustness
                residuals = observable - predicted
                delta = np.std(residuals)
                huber_loss = np.where(np.abs(residuals) <= delta,
                                     0.5 * residuals**2,
                                     delta * (np.abs(residuals) - 0.5 * delta))
                return np.sum(huber_loss)
            except:
                return np.inf
        
        # Convert bounds for differential evolution
        amplitude_bounds, exponent_bounds = bounds
        de_bounds = [amplitude_bounds, exponent_bounds]
        
        try:
            result = differential_evolution(
                objective,
                de_bounds,
                seed=self.random_seed,
                maxiter=1000,
                atol=self.tolerance,
                popsize=15
            )
            
            if not result.success:
                raise RuntimeError(f"Differential evolution failed to converge: {result.message}")
            
            amplitude, exponent = result.x
            
            # Estimate errors using finite differences
            def obj_single_param(param_val, param_idx):
                params = list(result.x)
                params[param_idx] = param_val
                return objective(params)
            
            # Numerical Hessian for error estimation
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
            
            # Calculate R-squared and residuals
            predicted = power_law_func(temperatures, amplitude, exponent)
            residuals = observable - predicted
            
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((observable - np.mean(observable))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # P-value approximation
            t_stat = abs(exponent) / exponent_error if exponent_error > 0 else 0
            p_value = 2 * (1 - 0.5 * (1 + np.tanh(t_stat - 2)))
            
            return RobustFitResult(
                exponent=exponent,
                amplitude=amplitude,
                exponent_error=exponent_error,
                amplitude_error=amplitude_error,
                r_squared=r_squared,
                p_value=p_value,
                fit_range=(np.min(temperatures), np.max(temperatures)),
                residuals=residuals,
                method_used='differential_evolution',
                convergence_info={'converged': True, 'function_evaluations': result.nfev},
                parameter_bounds_used=bounds,
                initial_guess=initial_guess
            )
            
        except Exception as e:
            raise RuntimeError(f"Differential evolution failed: {e}")
    
    def _fit_least_squares_robust(self,
                                temperatures: np.ndarray,
                                observable: np.ndarray,
                                critical_temperature: float,
                                exponent_type: str,
                                bounds: Tuple[Tuple[float, float], Tuple[float, float]],
                                initial_guess: Tuple[float, float]) -> RobustFitResult:
        """Robust least squares using scipy.optimize.least_squares."""
        
        def power_law_residuals(params, t, obs):
            amplitude, exponent = params
            reduced_temp = np.abs(t - critical_temperature)
            reduced_temp = np.maximum(reduced_temp, 1e-10)
            predicted = amplitude * (reduced_temp ** exponent)
            return obs - predicted
        
        # Convert bounds
        amplitude_bounds, exponent_bounds = bounds
        lower_bounds = [amplitude_bounds[0], exponent_bounds[0]]
        upper_bounds = [amplitude_bounds[1], exponent_bounds[1]]
        
        try:
            result = least_squares(
                power_law_residuals,
                initial_guess,
                args=(temperatures, observable),
                bounds=(lower_bounds, upper_bounds),
                loss='soft_l1',  # Robust loss function
                f_scale=np.std(observable),
                max_nfev=self.max_iterations
            )
            
            if not result.success:
                raise RuntimeError(f"Least squares failed: {result.message}")
            
            amplitude, exponent = result.x
            
            # Calculate parameter errors from Jacobian
            try:
                # Covariance matrix from Jacobian
                jac = result.jac
                cov_matrix = np.linalg.inv(jac.T @ jac)
                amplitude_error = np.sqrt(abs(cov_matrix[0, 0]))
                exponent_error = np.sqrt(abs(cov_matrix[1, 1]))
            except:
                amplitude_error = 0.1 * amplitude
                exponent_error = 0.1 * abs(exponent)
            
            # Calculate R-squared and residuals
            residuals = result.fun
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((observable - np.mean(observable))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # P-value approximation
            t_stat = abs(exponent) / exponent_error if exponent_error > 0 else 0
            p_value = 2 * (1 - 0.5 * (1 + np.tanh(t_stat - 2)))
            
            return RobustFitResult(
                exponent=exponent,
                amplitude=amplitude,
                exponent_error=exponent_error,
                amplitude_error=amplitude_error,
                r_squared=r_squared,
                p_value=p_value,
                fit_range=(np.min(temperatures), np.max(temperatures)),
                residuals=residuals,
                method_used='least_squares_robust',
                convergence_info={'converged': True, 'function_evaluations': result.nfev},
                parameter_bounds_used=bounds,
                initial_guess=initial_guess
            )
            
        except Exception as e:
            raise RuntimeError(f"Robust least squares failed: {e}")
    
    def _fit_log_linear_fallback(self,
                               temperatures: np.ndarray,
                               observable: np.ndarray,
                               critical_temperature: float,
                               exponent_type: str,
                               bounds: Tuple[Tuple[float, float], Tuple[float, float]],
                               initial_guess: Tuple[float, float]) -> RobustFitResult:
        """Fallback log-linear fitting method."""
        
        reduced_temp = np.abs(temperatures - critical_temperature)
        reduced_temp = np.maximum(reduced_temp, 1e-10)
        
        log_reduced_temp = safe_log(reduced_temp)
        log_obs = safe_log(observable)
        
        # Simple linear regression
        slope, intercept, r_value, p_value, std_err = linregress(log_reduced_temp, log_obs)
        
        amplitude = np.exp(intercept)
        exponent = slope
        
        # Check bounds
        amplitude_bounds, exponent_bounds = bounds
        
        # Clip to bounds if necessary
        if amplitude < amplitude_bounds[0] or amplitude > amplitude_bounds[1]:
            self.logger.warning(f"Amplitude {amplitude} outside bounds, clipping")
            amplitude = np.clip(amplitude, amplitude_bounds[0], amplitude_bounds[1])
        
        if exponent < exponent_bounds[0] or exponent > exponent_bounds[1]:
            self.logger.warning(f"Exponent {exponent} outside bounds, clipping")
            exponent = np.clip(exponent, exponent_bounds[0], exponent_bounds[1])
        
        # Calculate residuals in original space
        predicted_log = intercept + slope * log_reduced_temp
        residuals_log = log_obs - predicted_log
        
        # Convert to original space for R-squared calculation
        predicted = np.exp(predicted_log)
        residuals = observable - predicted
        
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((observable - np.mean(observable))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return RobustFitResult(
            exponent=exponent,
            amplitude=amplitude,
            exponent_error=std_err,
            amplitude_error=amplitude * std_err,  # Approximate
            r_squared=r_squared,
            p_value=p_value,
            fit_range=(np.min(temperatures), np.max(temperatures)),
            residuals=residuals,
            method_used='log_linear_fallback',
            convergence_info={'converged': True, 'r_value': r_value},
            parameter_bounds_used=bounds,
            initial_guess=initial_guess
        )
    
    def _validate_fit_result(self, result: RobustFitResult, exponent_type: str) -> bool:
        """Validate that fit result is physically reasonable."""
        
        # Check for NaN or infinite values
        if not np.isfinite(result.exponent) or not np.isfinite(result.amplitude):
            return False
        
        if not np.isfinite(result.exponent_error) or not np.isfinite(result.amplitude_error):
            return False
        
        # Check that errors are reasonable (not too large)
        if result.exponent_error > abs(result.exponent):
            return False
        
        if result.amplitude_error > result.amplitude:
            return False
        
        # Check R-squared is reasonable
        if result.r_squared < 0 or result.r_squared > 1:
            return False
        
        # Physics-based checks
        if exponent_type == 'beta':
            # Beta should be positive (we fit |T-Tc|^beta)
            if result.exponent <= 0 or result.exponent > 2:
                return False
        elif exponent_type in ['nu', 'gamma']:
            # Nu and gamma should be negative (we fit |T-Tc|^(-nu))
            if result.exponent >= 0 or result.exponent < -5:
                return False
        
        return True
    
    def _score_fit_result(self, result: RobustFitResult) -> float:
        """Score fit result for method selection."""
        
        score = 0
        
        # R-squared contribution (40%)
        score += 0.4 * max(0, result.r_squared)
        
        # P-value contribution (20%) - prefer significant results
        score += 0.2 * (1 - result.p_value) if result.p_value < 0.05 else 0
        
        # Relative error contribution (20%) - prefer smaller relative errors
        rel_error = result.exponent_error / abs(result.exponent) if result.exponent != 0 else 1
        score += 0.2 * max(0, 1 - rel_error)
        
        # Physics reasonableness (20%) - prefer physically reasonable values
        physics_score = 0
        if -3 < result.exponent < 3:  # Reasonable range
            physics_score += 0.5
        if result.amplitude > 0:  # Positive amplitude
            physics_score += 0.5
        
        score += 0.2 * physics_score
        
        return score


def create_robust_power_law_fitter(min_points: int = 8,
                                 max_iterations: int = 10000,
                                 tolerance: float = 1e-8,
                                 random_seed: Optional[int] = None) -> RobustPowerLawFitter:
    """
    Factory function to create a RobustPowerLawFitter.
    
    Args:
        min_points: Minimum number of points required for fitting
        max_iterations: Maximum iterations for optimization
        tolerance: Convergence tolerance
        random_seed: Random seed for reproducibility
        
    Returns:
        Configured RobustPowerLawFitter instance
    """
    return RobustPowerLawFitter(
        min_points=min_points,
        max_iterations=max_iterations,
        tolerance=tolerance,
        random_seed=random_seed
    )