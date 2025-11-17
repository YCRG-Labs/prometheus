"""
Critical Exponent Extraction Framework

This module implements robust power-law fitting and critical exponent extraction
for phase transition analysis in the Prometheus system.
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
class PowerLawFitResult:
    """
    Container for power-law fitting results.
    
    Attributes:
        exponent: Fitted critical exponent value
        amplitude: Fitted amplitude parameter
        exponent_error: Standard error of exponent
        amplitude_error: Standard error of amplitude
        r_squared: Coefficient of determination
        p_value: P-value for significance test
        fit_range: Temperature range used for fitting
        residuals: Fitting residuals
        confidence_interval: Bootstrap confidence interval for exponent
    """
    exponent: float
    amplitude: float
    exponent_error: float
    amplitude_error: float
    r_squared: float
    p_value: float
    fit_range: Tuple[float, float]
    residuals: np.ndarray
    confidence_interval: Optional[ConfidenceInterval] = None


@dataclass
class CriticalExponentResults:
    """
    Container for all critical exponent analysis results.
    
    Attributes:
        beta_result: Beta exponent (order parameter) fitting result
        nu_result: Nu exponent (correlation length) fitting result
        gamma_result: Gamma exponent (susceptibility) fitting result
        critical_temperature: Critical temperature used for analysis
        system_type: Type of physical system analyzed
        universality_class: Identified universality class
        validation: Validation against theoretical predictions
    """
    beta_result: Optional[PowerLawFitResult] = None
    nu_result: Optional[PowerLawFitResult] = None
    gamma_result: Optional[PowerLawFitResult] = None
    critical_temperature: Optional[float] = None
    system_type: str = "unknown"
    universality_class: UniversalityClass = UniversalityClass.UNKNOWN
    validation: Optional[CriticalExponentValidation] = None


class PowerLawFitter:
    """
    Robust power-law fitting system for critical behavior analysis.
    
    Implements multiple fitting methods including linear regression in log-log space,
    nonlinear least squares, and bootstrap confidence intervals.
    """
    
    def __init__(self, 
                 min_points: int = 5,
                 max_fit_range: float = 0.5,
                 bootstrap_samples: int = 1000,
                 random_seed: Optional[int] = None):
        """
        Initialize power-law fitter.
        
        Args:
            min_points: Minimum number of points required for fitting
            max_fit_range: Maximum relative distance from Tc for fitting
            bootstrap_samples: Number of bootstrap samples for confidence intervals
            random_seed: Random seed for reproducibility
        """
        self.min_points = min_points
        self.max_fit_range = max_fit_range
        self.bootstrap_samples = bootstrap_samples
        self.random_seed = random_seed
        self.logger = get_logger(__name__)
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def fit_power_law(self,
                     temperatures: np.ndarray,
                     observable: np.ndarray,
                     critical_temperature: float,
                     fit_method: str = 'log_linear',
                     fit_range_factor: float = 0.2) -> PowerLawFitResult:
        """
        Fit power law to observable data near critical temperature.
        
        For order parameter: m ∝ (Tc - T)^β for T < Tc
        For correlation length: ξ ∝ |T - Tc|^(-ν)
        For susceptibility: χ ∝ |T - Tc|^(-γ)
        
        Args:
            temperatures: Temperature array
            observable: Observable values (order parameter, correlation length, etc.)
            critical_temperature: Critical temperature estimate
            fit_method: Fitting method ('log_linear', 'nonlinear', 'robust')
            fit_range_factor: Fraction of temperature range around Tc to use
            
        Returns:
            PowerLawFitResult with fitting results and statistics
        """
        self.logger.info(f"Fitting power law using {fit_method} method")
        
        # Validate inputs
        if len(temperatures) != len(observable):
            raise CriticalExponentError("Temperature and observable arrays must have same length")
        
        if len(temperatures) < self.min_points:
            raise CriticalExponentError(f"Insufficient data points: {len(temperatures)} < {self.min_points}")
        
        # Define fitting range around critical temperature
        temp_range = np.max(temperatures) - np.min(temperatures)
        fit_range = fit_range_factor * temp_range
        
        # Select data points in fitting range
        fit_mask = np.abs(temperatures - critical_temperature) <= fit_range
        
        if np.sum(fit_mask) < self.min_points:
            # Expand range if too few points
            fit_range = self.max_fit_range * temp_range
            fit_mask = np.abs(temperatures - critical_temperature) <= fit_range
            
            if np.sum(fit_mask) < self.min_points:
                raise CriticalExponentError(f"Insufficient points in fitting range: {np.sum(fit_mask)}")
        
        fit_temps = temperatures[fit_mask]
        fit_obs = observable[fit_mask]
        
        # Remove zero or negative observable values for log fitting
        valid_mask = fit_obs > 0
        if np.sum(valid_mask) < self.min_points:
            raise CriticalExponentError("Insufficient positive observable values for power-law fitting")
        
        fit_temps = fit_temps[valid_mask]
        fit_obs = fit_obs[valid_mask]
        
        # Perform fitting based on method
        if fit_method == 'log_linear':
            result = self._fit_log_linear(fit_temps, fit_obs, critical_temperature)
        elif fit_method == 'nonlinear':
            result = self._fit_nonlinear(fit_temps, fit_obs, critical_temperature)
        elif fit_method == 'robust':
            result = self._fit_robust(fit_temps, fit_obs, critical_temperature)
        else:
            raise ValueError(f"Unknown fitting method: {fit_method}")
        
        # Add fitting range information
        result.fit_range = (np.min(fit_temps), np.max(fit_temps))
        
        # Compute bootstrap confidence interval
        try:
            result.confidence_interval = self._compute_bootstrap_ci(
                fit_temps, fit_obs, critical_temperature, fit_method
            )
        except Exception as e:
            self.logger.warning(f"Bootstrap confidence interval computation failed: {e}")
            result.confidence_interval = None
        
        self.logger.info(f"Power-law fit completed: exponent = {result.exponent:.4f} ± {result.exponent_error:.4f}")
        
        return result
    
    def _fit_log_linear(self,
                       temperatures: np.ndarray,
                       observable: np.ndarray,
                       critical_temperature: float) -> PowerLawFitResult:
        """Fit power law using linear regression in log-log space."""
        
        # Calculate reduced temperature |T - Tc|
        reduced_temp = np.abs(temperatures - critical_temperature)
        
        # Remove points too close to Tc (avoid log(0))
        min_reduced_temp = 1e-6
        valid_mask = reduced_temp > min_reduced_temp
        
        if np.sum(valid_mask) < self.min_points:
            raise CriticalExponentError("Too few points away from critical temperature")
        
        reduced_temp = reduced_temp[valid_mask]
        fit_obs = observable[valid_mask]
        
        # Log-log regression: log(obs) = log(A) + β * log(|T - Tc|)
        log_reduced_temp = safe_log(reduced_temp)
        log_obs = safe_log(fit_obs)
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(log_reduced_temp, log_obs)
        
        # Extract parameters
        exponent = slope
        amplitude = np.exp(intercept)
        exponent_error = std_err
        
        # Estimate amplitude error using error propagation
        amplitude_error = amplitude * std_err  # Approximate
        
        # Calculate residuals
        predicted_log = intercept + slope * log_reduced_temp
        residuals = log_obs - predicted_log
        
        return PowerLawFitResult(
            exponent=exponent,
            amplitude=amplitude,
            exponent_error=exponent_error,
            amplitude_error=amplitude_error,
            r_squared=r_value**2,
            p_value=p_value,
            fit_range=(0, 0),  # Will be set later
            residuals=residuals
        )
    
    def _fit_nonlinear(self,
                      temperatures: np.ndarray,
                      observable: np.ndarray,
                      critical_temperature: float) -> PowerLawFitResult:
        """Fit power law using nonlinear least squares."""
        
        # Define power law function
        def power_law(t, amplitude, exponent):
            reduced_temp = np.abs(t - critical_temperature)
            # Add small offset to avoid division by zero
            reduced_temp = np.maximum(reduced_temp, 1e-10)
            return amplitude * (reduced_temp ** exponent)
        
        # Initial parameter guess
        p0 = [np.mean(observable), -0.5]  # Reasonable starting values
        
        try:
            # Perform curve fitting
            popt, pcov = curve_fit(
                power_law, temperatures, observable, p0=p0,
                bounds=([0, -5], [np.inf, 5]),  # Reasonable bounds
                maxfev=10000
            )
            
            amplitude, exponent = popt
            param_errors = np.sqrt(np.diag(pcov))
            amplitude_error, exponent_error = param_errors
            
            # Calculate R-squared
            predicted = power_law(temperatures, amplitude, exponent)
            ss_res = np.sum((observable - predicted) ** 2)
            ss_tot = np.sum((observable - np.mean(observable)) ** 2)
            r_squared = 1 - safe_divide(ss_res, ss_tot, fill_value=0.0)
            
            # Calculate residuals
            residuals = observable - predicted
            
            # Approximate p-value (simplified)
            n = len(temperatures)
            if n > 2:
                t_stat = safe_divide(np.abs(exponent), exponent_error, fill_value=0.0)
                # Rough approximation for p-value
                p_value = 2 * (1 - 0.5 * (1 + np.tanh(t_stat - 2)))
            else:
                p_value = 1.0
            
            return PowerLawFitResult(
                exponent=exponent,
                amplitude=amplitude,
                exponent_error=exponent_error,
                amplitude_error=amplitude_error,
                r_squared=r_squared,
                p_value=p_value,
                fit_range=(0, 0),  # Will be set later
                residuals=residuals
            )
            
        except Exception as e:
            raise CriticalExponentError(f"Nonlinear fitting failed: {e}")
    
    def _fit_robust(self,
                   temperatures: np.ndarray,
                   observable: np.ndarray,
                   critical_temperature: float) -> PowerLawFitResult:
        """Fit power law using robust optimization methods."""
        
        # Try multiple methods and select best result
        methods = ['log_linear', 'nonlinear']
        results = []
        
        for method in methods:
            try:
                if method == 'log_linear':
                    result = self._fit_log_linear(temperatures, observable, critical_temperature)
                else:
                    result = self._fit_nonlinear(temperatures, observable, critical_temperature)
                results.append((method, result))
            except Exception as e:
                self.logger.warning(f"Method {method} failed: {e}")
                continue
        
        if not results:
            raise CriticalExponentError("All fitting methods failed")
        
        # Select result with best R-squared
        best_method, best_result = max(results, key=lambda x: x[1].r_squared)
        
        self.logger.info(f"Robust fitting selected {best_method} method (R² = {best_result.r_squared:.4f})")
        
        return best_result
    
    def _compute_bootstrap_ci(self,
                             temperatures: np.ndarray,
                             observable: np.ndarray,
                             critical_temperature: float,
                             fit_method: str) -> ConfidenceInterval:
        """Compute bootstrap confidence interval for exponent."""
        
        n_data = len(temperatures)
        bootstrap_exponents = []
        
        rng = np.random.RandomState(self.random_seed)
        
        for _ in range(self.bootstrap_samples):
            # Bootstrap resample
            indices = rng.choice(n_data, size=n_data, replace=True)
            boot_temps = temperatures[indices]
            boot_obs = observable[indices]
            
            try:
                # Fit power law to bootstrap sample
                if fit_method == 'log_linear':
                    result = self._fit_log_linear(boot_temps, boot_obs, critical_temperature)
                elif fit_method == 'nonlinear':
                    result = self._fit_nonlinear(boot_temps, boot_obs, critical_temperature)
                else:
                    result = self._fit_robust(boot_temps, boot_obs, critical_temperature)
                
                bootstrap_exponents.append(result.exponent)
                
            except Exception:
                continue  # Skip failed bootstrap samples
        
        if len(bootstrap_exponents) < self.bootstrap_samples * 0.5:
            raise CriticalExponentError("Too many bootstrap failures")
        
        bootstrap_exponents = np.array(bootstrap_exponents)
        
        # Compute confidence interval
        alpha = 0.05  # 95% confidence interval
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_exponents, lower_percentile)
        upper_bound = np.percentile(bootstrap_exponents, upper_percentile)
        
        return ConfidenceInterval(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=0.95,
            method="bootstrap_percentile",
            n_bootstrap_samples=len(bootstrap_exponents)
        )


class CriticalExponentAnalyzer:
    """
    Main class for critical exponent extraction and analysis.
    
    Provides methods to extract β, ν, and γ exponents from simulation data
    and validate against theoretical predictions.
    """
    
    def __init__(self,
                 fitter: Optional[PowerLawFitter] = None,
                 theoretical_exponents: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Initialize critical exponent analyzer.
        
        Args:
            fitter: PowerLawFitter instance (creates default if None)
            theoretical_exponents: Dictionary of theoretical exponent values by system type
        """
        self.fitter = fitter or PowerLawFitter()
        self.logger = get_logger(__name__)
        
        # Default theoretical exponents for common systems
        if theoretical_exponents is None:
            self.theoretical_exponents = {
                'ising_2d': {'beta': 0.125, 'nu': 1.0, 'gamma': 1.75},
                'ising_3d': {'beta': 0.326, 'nu': 0.630, 'gamma': 1.237},
                'xy_2d': {'beta': 0.125, 'nu': 1.0, 'gamma': 1.75},  # KT transition
                'heisenberg_3d': {'beta': 0.365, 'nu': 0.705, 'gamma': 1.386},
                'potts_3_2d': {'beta': 0.111, 'nu': 0.833, 'gamma': 1.556}
            }
        else:
            self.theoretical_exponents = theoretical_exponents
    
    def extract_beta_exponent(self,
                             temperatures: np.ndarray,
                             magnetizations: np.ndarray,
                             critical_temperature: float,
                             fit_method: str = 'robust') -> PowerLawFitResult:
        """
        Extract β exponent from magnetization data.
        
        Fits: m ∝ (Tc - T)^β for T < Tc
        
        Args:
            temperatures: Temperature array
            magnetizations: Magnetization values (order parameter)
            critical_temperature: Critical temperature estimate
            fit_method: Fitting method to use
            
        Returns:
            PowerLawFitResult for β exponent
        """
        self.logger.info("Extracting β exponent from magnetization data")
        
        # Use only temperatures below Tc for β exponent
        below_tc_mask = temperatures < critical_temperature
        
        if np.sum(below_tc_mask) < self.fitter.min_points:
            self.logger.warning("Few points below Tc, using all data")
            below_tc_mask = np.ones_like(temperatures, dtype=bool)
        
        fit_temps = temperatures[below_tc_mask]
        fit_mags = np.abs(magnetizations[below_tc_mask])  # Use absolute value
        
        return self.fitter.fit_power_law(
            fit_temps, fit_mags, critical_temperature, fit_method
        )
    
    def extract_nu_exponent(self,
                           temperatures: np.ndarray,
                           correlation_lengths: np.ndarray,
                           critical_temperature: float,
                           fit_method: str = 'robust') -> PowerLawFitResult:
        """
        Extract ν exponent from correlation length data.
        
        Fits: ξ ∝ |T - Tc|^(-ν)
        
        Args:
            temperatures: Temperature array
            correlation_lengths: Correlation length values
            critical_temperature: Critical temperature estimate
            fit_method: Fitting method to use
            
        Returns:
            PowerLawFitResult for ν exponent
        """
        self.logger.info("Extracting ν exponent from correlation length data")
        
        # Use data on both sides of Tc for ν exponent
        return self.fitter.fit_power_law(
            temperatures, correlation_lengths, critical_temperature, fit_method
        )
    
    def compute_correlation_length_from_latent(self,
                                             latent_repr: LatentRepresentation,
                                             n_temp_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute correlation length from latent space fluctuations.
        
        Uses the variance of latent coordinates as a proxy for correlation length.
        
        Args:
            latent_repr: LatentRepresentation object
            n_temp_bins: Number of temperature bins
            
        Returns:
            Tuple of (temperatures, correlation_lengths)
        """
        self.logger.info("Computing correlation length from latent space")
        
        # Create temperature bins
        temp_min, temp_max = np.min(latent_repr.temperatures), np.max(latent_repr.temperatures)
        temp_bins = np.linspace(temp_min, temp_max, n_temp_bins + 1)
        temp_centers = (temp_bins[:-1] + temp_bins[1:]) / 2
        
        correlation_lengths = []
        valid_temps = []
        
        for i in range(len(temp_bins) - 1):
            temp_mask = (latent_repr.temperatures >= temp_bins[i]) & \
                       (latent_repr.temperatures < temp_bins[i + 1])
            
            if np.sum(temp_mask) >= 5:  # Minimum points per bin
                # Use combined variance of both latent dimensions as correlation length proxy
                z1_var = np.var(latent_repr.z1[temp_mask])
                z2_var = np.var(latent_repr.z2[temp_mask])
                corr_length = np.sqrt(z1_var + z2_var)
                
                correlation_lengths.append(corr_length)
                valid_temps.append(temp_centers[i])
        
        return np.array(valid_temps), np.array(correlation_lengths)
    
    def analyze_system_exponents(self,
                                latent_repr: LatentRepresentation,
                                critical_temperature: float,
                                system_type: str = 'ising_3d') -> CriticalExponentResults:
        """
        Perform complete critical exponent analysis for a system.
        
        Args:
            latent_repr: LatentRepresentation object
            critical_temperature: Critical temperature estimate
            system_type: Type of physical system for theoretical comparison
            
        Returns:
            CriticalExponentResults with all extracted exponents
        """
        self.logger.info(f"Analyzing critical exponents for {system_type} system")
        
        results = CriticalExponentResults(
            critical_temperature=critical_temperature,
            system_type=system_type
        )
        
        try:
            # Extract β exponent from magnetization
            results.beta_result = self.extract_beta_exponent(
                latent_repr.temperatures,
                latent_repr.magnetizations,
                critical_temperature
            )
            
        except Exception as e:
            self.logger.error(f"β exponent extraction failed: {e}")
            results.beta_result = None
        
        try:
            # Extract ν exponent from correlation length
            temps, corr_lengths = self.compute_correlation_length_from_latent(latent_repr)
            
            if len(temps) >= self.fitter.min_points:
                results.nu_result = self.extract_nu_exponent(
                    temps, corr_lengths, critical_temperature
                )
            else:
                self.logger.warning("Insufficient data for ν exponent extraction")
                results.nu_result = None
                
        except Exception as e:
            self.logger.error(f"ν exponent extraction failed: {e}")
            results.nu_result = None
        
        # Validate against theoretical predictions
        try:
            results.validation = self._validate_exponents(results, system_type)
            results.universality_class = self._identify_universality_class(results)
        except Exception as e:
            self.logger.error(f"Exponent validation failed: {e}")
            results.validation = None
        
        self.logger.info("Critical exponent analysis completed")
        
        return results
    
    def _validate_exponents(self,
                           results: CriticalExponentResults,
                           system_type: str) -> CriticalExponentValidation:
        """Validate extracted exponents against theoretical predictions."""
        
        if system_type not in self.theoretical_exponents:
            raise CriticalExponentError(f"Unknown system type: {system_type}")
        
        theoretical = self.theoretical_exponents[system_type]
        
        validation = CriticalExponentValidation(
            beta_exponent=results.beta_result.exponent if results.beta_result else None,
            beta_theoretical=theoretical['beta'],
            beta_confidence_interval=(
                results.beta_result.confidence_interval.lower_bound,
                results.beta_result.confidence_interval.upper_bound
            ) if results.beta_result and results.beta_result.confidence_interval else None,
            beta_deviation=safe_divide(
                np.abs(results.beta_result.exponent - theoretical['beta']), 
                theoretical['beta'],
                fill_value=0.0
            ) if results.beta_result else None
        )
        
        if results.nu_result:
            validation.nu_exponent = results.nu_result.exponent
            validation.nu_theoretical = theoretical['nu']
            validation.nu_confidence_interval = (
                results.nu_result.confidence_interval.lower_bound,
                results.nu_result.confidence_interval.upper_bound
            ) if results.nu_result.confidence_interval else None
            validation.nu_deviation = safe_divide(
                np.abs(results.nu_result.exponent - theoretical['nu']),
                theoretical['nu'],
                fill_value=0.0
            )
        
        # Check universality class match
        beta_match = (validation.beta_deviation < 0.2) if validation.beta_deviation else False
        nu_match = (validation.nu_deviation < 0.2) if validation.nu_deviation else False
        
        validation.universality_class_match = beta_match and nu_match
        
        return validation
    
    def _identify_universality_class(self, results: CriticalExponentResults) -> UniversalityClass:
        """Identify universality class based on extracted exponents."""
        
        if not results.beta_result:
            return UniversalityClass.UNKNOWN
        
        beta_exp = results.beta_result.exponent
        nu_exp = results.nu_result.exponent if results.nu_result else None
        
        # Compare with known universality classes
        best_match = UniversalityClass.UNKNOWN
        min_distance = float('inf')
        
        for system_name, exponents in self.theoretical_exponents.items():
            # Calculate distance in exponent space
            beta_dist = (beta_exp - exponents['beta'])**2
            
            if nu_exp is not None:
                nu_dist = (nu_exp - exponents['nu'])**2
                total_dist = np.sqrt(beta_dist + nu_dist)
            else:
                total_dist = np.sqrt(beta_dist)
            
            if total_dist < min_distance:
                min_distance = total_dist
                
                # Map system names to universality classes
                if 'ising_2d' in system_name:
                    best_match = UniversalityClass.ISING_2D
                elif 'ising_3d' in system_name:
                    best_match = UniversalityClass.ISING_3D
                elif 'xy' in system_name:
                    best_match = UniversalityClass.XY_2D
                elif 'heisenberg' in system_name:
                    best_match = UniversalityClass.HEISENBERG_3D
                elif 'potts' in system_name:
                    best_match = UniversalityClass.POTTS_3_2D
        
        # Only return match if distance is reasonable
        if min_distance < 0.5:  # Threshold for reasonable match
            return best_match
        else:
            return UniversalityClass.UNKNOWN
    
    def visualize_power_law_fits(self,
                                results: CriticalExponentResults,
                                latent_repr: LatentRepresentation,
                                figsize: Tuple[int, int] = (15, 5)) -> Figure:
        """
        Create visualization of power-law fits and critical exponents.
        
        Args:
            results: CriticalExponentResults object
            latent_repr: Original latent representation
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure with power-law fit visualizations
        """
        n_plots = sum([results.beta_result is not None, results.nu_result is not None])
        
        if n_plots == 0:
            raise ValueError("No valid exponent results to visualize")
        
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot β exponent fit
        if results.beta_result is not None:
            ax = axes[plot_idx]
            
            # Get data below Tc
            below_tc_mask = latent_repr.temperatures < results.critical_temperature
            fit_temps = latent_repr.temperatures[below_tc_mask]
            fit_mags = np.abs(latent_repr.magnetizations[below_tc_mask])
            
            # Plot data points
            ax.scatter(results.critical_temperature - fit_temps, fit_mags, 
                      alpha=0.6, s=20, label='Data')
            
            # Plot power-law fit
            reduced_temps = results.critical_temperature - fit_temps
            fit_range = results.beta_result.fit_range
            fit_mask = (fit_temps >= fit_range[0]) & (fit_temps <= fit_range[1])
            
            if np.any(fit_mask):
                fit_reduced = reduced_temps[fit_mask]
                fit_predicted = results.beta_result.amplitude * (fit_reduced ** results.beta_result.exponent)
                
                # Sort for smooth line
                sort_idx = np.argsort(fit_reduced)
                ax.plot(fit_reduced[sort_idx], fit_predicted[sort_idx], 
                       'r-', linewidth=2, 
                       label=f'β = {results.beta_result.exponent:.3f} ± {results.beta_result.exponent_error:.3f}')
            
            ax.set_xlabel('Tc - T')
            ax.set_ylabel('|Magnetization|')
            ax.set_title('β Exponent Fit')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # Plot ν exponent fit
        if results.nu_result is not None:
            ax = axes[plot_idx]
            
            # Get correlation length data
            temps, corr_lengths = self.compute_correlation_length_from_latent(latent_repr)
            
            # Plot data points
            reduced_temps = np.abs(temps - results.critical_temperature)
            ax.scatter(reduced_temps, corr_lengths, alpha=0.6, s=20, label='Data')
            
            # Plot power-law fit
            fit_range = results.nu_result.fit_range
            fit_mask = (temps >= fit_range[0]) & (temps <= fit_range[1])
            
            if np.any(fit_mask):
                fit_reduced = reduced_temps[fit_mask]
                fit_predicted = results.nu_result.amplitude * (fit_reduced ** results.nu_result.exponent)
                
                # Sort for smooth line
                sort_idx = np.argsort(fit_reduced)
                ax.plot(fit_reduced[sort_idx], fit_predicted[sort_idx], 
                       'r-', linewidth=2,
                       label=f'ν = {results.nu_result.exponent:.3f} ± {results.nu_result.exponent_error:.3f}')
            
            ax.set_xlabel('|T - Tc|')
            ax.set_ylabel('Correlation Length')
            ax.set_title('ν Exponent Fit')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_exponent_comparison_table(self,
                                       results_dict: Dict[str, CriticalExponentResults]) -> Dict[str, Any]:
        """
        Create comparison table of critical exponents across different systems.
        
        Args:
            results_dict: Dictionary mapping system names to CriticalExponentResults
            
        Returns:
            Dictionary with comparison table data
        """
        comparison_data = {
            'system': [],
            'beta_measured': [],
            'beta_theoretical': [],
            'beta_error_percent': [],
            'nu_measured': [],
            'nu_theoretical': [],
            'nu_error_percent': [],
            'universality_class': []
        }
        
        for system_name, results in results_dict.items():
            comparison_data['system'].append(system_name)
            
            # β exponent data
            if results.beta_result and results.validation:
                comparison_data['beta_measured'].append(f"{results.beta_result.exponent:.3f}")
                comparison_data['beta_theoretical'].append(f"{results.validation.beta_theoretical:.3f}")
                comparison_data['beta_error_percent'].append(f"{results.validation.beta_deviation * 100:.1f}%")
            else:
                comparison_data['beta_measured'].append("N/A")
                comparison_data['beta_theoretical'].append("N/A")
                comparison_data['beta_error_percent'].append("N/A")
            
            # ν exponent data
            if results.nu_result and results.validation and results.validation.nu_exponent:
                comparison_data['nu_measured'].append(f"{results.nu_result.exponent:.3f}")
                comparison_data['nu_theoretical'].append(f"{results.validation.nu_theoretical:.3f}")
                comparison_data['nu_error_percent'].append(f"{results.validation.nu_deviation * 100:.1f}%")
            else:
                comparison_data['nu_measured'].append("N/A")
                comparison_data['nu_theoretical'].append("N/A")
                comparison_data['nu_error_percent'].append("N/A")
            
            # Universality class
            comparison_data['universality_class'].append(results.universality_class.value)
        
        return comparison_data


def create_critical_exponent_analyzer(system_type: str = 'ising_3d',
                                    bootstrap_samples: int = 1000,
                                    random_seed: Optional[int] = None) -> CriticalExponentAnalyzer:
    """
    Factory function to create a CriticalExponentAnalyzer with appropriate settings.
    
    Args:
        system_type: Type of physical system
        bootstrap_samples: Number of bootstrap samples for confidence intervals
        random_seed: Random seed for reproducibility
        
    Returns:
        Configured CriticalExponentAnalyzer instance
    """
    fitter = PowerLawFitter(
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed
    )
    
    return CriticalExponentAnalyzer(fitter=fitter)