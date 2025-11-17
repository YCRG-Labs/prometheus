#!/usr/bin/env python3
"""
Robust Critical Exponent Extraction Script

This script implements key accuracy improvements while maintaining stability:
1. Better critical temperature detection
2. Improved data preprocessing 
3. Adaptive fitting ranges
4. Enhanced error estimation
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.signal import savgol_filter

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.latent_analysis import LatentRepresentation
from src.utils.config import PrometheusConfig
from src.utils.logging_utils import setup_logging, get_logger


class RobustCriticalTemperatureDetector:
    """Robust critical temperature detection using multiple methods."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def detect_critical_temperature(self, temperatures, order_parameter):
        """Detect critical temperature using ensemble of methods."""
        
        # Method 1: Maximum derivative
        tc1, conf1 = self._derivative_method(temperatures, order_parameter)
        
        # Method 2: Susceptibility maximum
        tc2, conf2 = self._susceptibility_method(temperatures, order_parameter)
        
        # Method 3: Inflection point
        tc3, conf3 = self._inflection_method(temperatures, order_parameter)
        
        # Weighted ensemble
        estimates = [tc1, tc2, tc3]
        confidences = [conf1, conf2, conf3]
        
        # Remove invalid estimates
        valid_estimates = []
        valid_confidences = []
        
        for tc, conf in zip(estimates, confidences):
            if np.isfinite(tc) and conf > 0.1:
                valid_estimates.append(tc)
                valid_confidences.append(conf)
        
        if not valid_estimates:
            # Fallback to temperature range center
            return np.mean(temperatures), 0.3
        
        # Weighted average
        weights = np.array(valid_confidences)
        weights = weights / np.sum(weights)
        
        tc_ensemble = np.average(valid_estimates, weights=weights)
        confidence = np.mean(valid_confidences) * (1 - np.std(valid_estimates) / np.mean(valid_estimates))
        
        self.logger.info(f"Detected Tc = {tc_ensemble:.4f} (methods: {len(valid_estimates)}, confidence: {confidence:.3f})")
        
        return tc_ensemble, confidence
    
    def _derivative_method(self, temperatures, order_parameter):
        """Find Tc using maximum absolute derivative."""
        try:
            # Smooth data first
            if len(order_parameter) > 7:
                smoothed = savgol_filter(order_parameter, 7, 3)
            else:
                smoothed = order_parameter
            
            # Calculate derivative
            derivative = np.gradient(smoothed, temperatures)
            
            # Find maximum absolute derivative
            max_idx = np.argmax(np.abs(derivative))
            tc = temperatures[max_idx]
            
            # Confidence based on derivative magnitude
            max_deriv = np.abs(derivative[max_idx])
            mean_deriv = np.mean(np.abs(derivative))
            confidence = min(1.0, max_deriv / (mean_deriv + 1e-10))
            
            return tc, confidence
            
        except Exception as e:
            self.logger.warning(f"Derivative method failed: {e}")
            return np.mean(temperatures), 0.1
    
    def _susceptibility_method(self, temperatures, order_parameter):
        """Find Tc using susceptibility (variance) maximum."""
        try:
            # Bin temperatures
            n_bins = min(20, len(np.unique(temperatures)) // 2)
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
            
            # Find maximum
            max_idx = np.argmax(susceptibilities)
            tc = temp_centers[max_idx]
            
            # Confidence
            max_susc = susceptibilities[max_idx]
            mean_susc = np.mean(susceptibilities)
            confidence = (max_susc - mean_susc) / (max_susc + 1e-10)
            
            return tc, min(1.0, confidence)
            
        except Exception as e:
            self.logger.warning(f"Susceptibility method failed: {e}")
            return np.mean(temperatures), 0.1
    
    def _inflection_method(self, temperatures, order_parameter):
        """Find Tc using inflection point (second derivative)."""
        try:
            # Smooth data
            if len(order_parameter) > 9:
                smoothed = savgol_filter(order_parameter, 9, 3)
            else:
                smoothed = order_parameter
            
            # Second derivative
            first_deriv = np.gradient(smoothed, temperatures)
            second_deriv = np.gradient(first_deriv, temperatures)
            
            # Find maximum absolute second derivative
            max_idx = np.argmax(np.abs(second_deriv))
            tc = temperatures[max_idx]
            
            # Confidence
            max_second_deriv = np.abs(second_deriv[max_idx])
            mean_second_deriv = np.mean(np.abs(second_deriv))
            confidence = min(1.0, max_second_deriv / (mean_second_deriv + 1e-10))
            
            return tc, confidence
            
        except Exception as e:
            self.logger.warning(f"Inflection method failed: {e}")
            return np.mean(temperatures), 0.1


class RobustPowerLawFitter:
    """Robust power-law fitting with improved preprocessing and error handling."""
    
    def __init__(self, min_points=8, bootstrap_samples=1000):
        self.min_points = min_points
        self.bootstrap_samples = bootstrap_samples
        self.logger = get_logger(__name__)
    
    def fit_power_law(self, temperatures, observable, critical_temperature, exponent_type='beta'):
        """Fit power law with robust methods."""
        
        # Preprocess data
        temps_clean, obs_clean = self._preprocess_data(temperatures, observable, critical_temperature)
        
        if len(temps_clean) < self.min_points:
            raise ValueError(f"Insufficient clean data: {len(temps_clean)} < {self.min_points}")
        
        # Select fitting range adaptively
        fit_range = self._select_fitting_range(temps_clean, obs_clean, critical_temperature, exponent_type)
        
        # Apply range
        range_mask = (temps_clean >= fit_range[0]) & (temps_clean <= fit_range[1])
        fit_temps = temps_clean[range_mask]
        fit_obs = obs_clean[range_mask]
        
        if len(fit_temps) < self.min_points:
            # Expand range if needed
            temp_range = np.max(temps_clean) - np.min(temps_clean)
            if exponent_type == 'beta':
                fit_range = (critical_temperature - 0.4 * temp_range, critical_temperature)
            else:
                fit_range = (critical_temperature - 0.3 * temp_range, critical_temperature + 0.3 * temp_range)
            
            range_mask = (temps_clean >= fit_range[0]) & (temps_clean <= fit_range[1])
            fit_temps = temps_clean[range_mask]
            fit_obs = obs_clean[range_mask]
        
        if len(fit_temps) < self.min_points:
            raise ValueError(f"Still insufficient data after range expansion: {len(fit_temps)}")
        
        # Try multiple fitting methods
        results = []
        
        # Method 1: Log-linear regression
        try:
            result1 = self._log_linear_fit(fit_temps, fit_obs, critical_temperature)
            results.append(('log_linear', result1))
        except Exception as e:
            self.logger.warning(f"Log-linear fit failed: {e}")
        
        # Method 2: Nonlinear fit with bounds
        try:
            result2 = self._bounded_nonlinear_fit(fit_temps, fit_obs, critical_temperature, exponent_type)
            results.append(('nonlinear', result2))
        except Exception as e:
            self.logger.warning(f"Nonlinear fit failed: {e}")
        
        if not results:
            raise ValueError("All fitting methods failed")
        
        # Select best result
        best_result = self._select_best_result(results)
        
        # Add bootstrap confidence interval
        try:
            ci_lower, ci_upper = self._bootstrap_confidence_interval(
                fit_temps, fit_obs, critical_temperature, exponent_type
            )
            best_result['confidence_interval'] = (ci_lower, ci_upper)
        except Exception as e:
            self.logger.warning(f"Bootstrap CI failed: {e}")
            best_result['confidence_interval'] = None
        
        best_result['fit_range'] = fit_range
        best_result['n_points'] = len(fit_temps)
        
        return best_result
    
    def _preprocess_data(self, temperatures, observable, critical_temperature):
        """Clean and preprocess data."""
        # Remove invalid values
        valid_mask = np.isfinite(temperatures) & np.isfinite(observable) & (observable > 0)
        temps = temperatures[valid_mask]
        obs = observable[valid_mask]
        
        # Remove outliers using IQR method
        if len(obs) > 10:
            q1, q3 = np.percentile(obs, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_mask = (obs >= lower_bound) & (obs <= upper_bound)
            temps = temps[outlier_mask]
            obs = obs[outlier_mask]
        
        # Sort by temperature
        sort_idx = np.argsort(temps)
        return temps[sort_idx], obs[sort_idx]
    
    def _select_fitting_range(self, temperatures, observable, critical_temperature, exponent_type):
        """Select optimal fitting range based on linearity in log space."""
        
        temp_range = np.max(temperatures) - np.min(temperatures)
        
        if exponent_type == 'beta':
            # For beta: only use T < Tc
            below_tc = temperatures < critical_temperature
            if np.sum(below_tc) < self.min_points:
                t_max = critical_temperature + 0.05 * temp_range
            else:
                t_max = critical_temperature
            
            # Test different lower bounds
            test_ranges = np.linspace(0.1 * temp_range, 0.5 * temp_range, 8)
            best_r2 = -1
            best_range = 0.3 * temp_range
            
            for test_range in test_ranges:
                t_min = critical_temperature - test_range
                mask = (temperatures >= t_min) & (temperatures < t_max)
                
                if np.sum(mask) < self.min_points:
                    continue
                
                try:
                    test_temps = temperatures[mask]
                    test_obs = observable[mask]
                    
                    reduced_temps = critical_temperature - test_temps
                    reduced_temps = np.maximum(reduced_temps, 1e-10)
                    
                    log_temps = np.log(reduced_temps)
                    log_obs = np.log(test_obs)
                    
                    _, _, r_value, _, _ = linregress(log_temps, log_obs)
                    
                    if r_value**2 > best_r2:
                        best_r2 = r_value**2
                        best_range = test_range
                        
                except Exception:
                    continue
            
            return (critical_temperature - best_range, t_max)
        
        else:
            # For nu: use both sides of Tc
            test_ranges = np.linspace(0.1 * temp_range, 0.4 * temp_range, 8)
            best_r2 = -1
            best_range = 0.25 * temp_range
            
            for test_range in test_ranges:
                t_min = critical_temperature - test_range
                t_max = critical_temperature + test_range
                mask = (temperatures >= t_min) & (temperatures <= t_max)
                
                if np.sum(mask) < self.min_points:
                    continue
                
                try:
                    test_temps = temperatures[mask]
                    test_obs = observable[mask]
                    
                    reduced_temps = np.abs(test_temps - critical_temperature)
                    reduced_temps = np.maximum(reduced_temps, 1e-10)
                    
                    log_temps = np.log(reduced_temps)
                    log_obs = np.log(test_obs)
                    
                    _, _, r_value, _, _ = linregress(log_temps, log_obs)
                    
                    if r_value**2 > best_r2:
                        best_r2 = r_value**2
                        best_range = test_range
                        
                except Exception:
                    continue
            
            return (critical_temperature - best_range, critical_temperature + best_range)
    
    def _log_linear_fit(self, temperatures, observable, critical_temperature):
        """Log-linear regression fit."""
        
        if len(temperatures) < 3:
            raise ValueError("Insufficient data for log-linear fit")
        
        # Calculate reduced temperature
        reduced_temp = np.abs(temperatures - critical_temperature)
        reduced_temp = np.maximum(reduced_temp, 1e-10)
        
        log_temps = np.log(reduced_temp)
        log_obs = np.log(observable)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(log_temps, log_obs)
        
        # Calculate residuals
        predicted_log = intercept + slope * log_temps
        residuals = log_obs - predicted_log
        
        return {
            'exponent': slope,
            'amplitude': np.exp(intercept),
            'exponent_error': std_err,
            'amplitude_error': np.exp(intercept) * std_err,  # Approximate
            'r_squared': r_value**2,
            'p_value': p_value,
            'residuals': residuals,
            'method': 'log_linear'
        }
    
    def _bounded_nonlinear_fit(self, temperatures, observable, critical_temperature, exponent_type):
        """Nonlinear fit with physical bounds."""
        
        def power_law(t, amplitude, exponent):
            reduced_temp = np.abs(t - critical_temperature)
            reduced_temp = np.maximum(reduced_temp, 1e-10)
            return amplitude * (reduced_temp ** exponent)
        
        # Initial guess from log fit
        try:
            log_result = self._log_linear_fit(temperatures, observable, critical_temperature)
            initial_amplitude = log_result['amplitude']
            initial_exponent = log_result['exponent']
        except:
            initial_amplitude = np.median(observable)
            initial_exponent = -0.5 if exponent_type == 'beta' else -1.0
        
        # Physical bounds
        if exponent_type == 'beta':
            bounds = ([0, -2], [np.inf, 2])
        else:  # nu or gamma
            bounds = ([0, -5], [np.inf, 0])
        
        # Fit
        popt, pcov = curve_fit(
            power_law, temperatures, observable,
            p0=[initial_amplitude, initial_exponent],
            bounds=bounds,
            maxfev=5000
        )
        
        amplitude, exponent = popt
        param_errors = np.sqrt(np.diag(pcov))
        
        # Calculate R-squared
        predicted = power_law(temperatures, amplitude, exponent)
        ss_res = np.sum((observable - predicted)**2)
        ss_tot = np.sum((observable - np.mean(observable))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # P-value approximation
        t_stat = abs(exponent) / param_errors[1] if param_errors[1] > 0 else 0
        p_value = 2 * (1 - 0.5 * (1 + np.tanh(t_stat - 2)))
        
        return {
            'exponent': exponent,
            'amplitude': amplitude,
            'exponent_error': param_errors[1],
            'amplitude_error': param_errors[0],
            'r_squared': r_squared,
            'p_value': p_value,
            'residuals': observable - predicted,
            'method': 'nonlinear'
        }
    
    def _select_best_result(self, results):
        """Select best result based on R-squared and physical reasonableness."""
        
        if len(results) == 1:
            return results[0][1]
        
        best_score = -1
        best_result = None
        
        for method, result in results:
            # Score based on R-squared (70%) and physical reasonableness (30%)
            r2_score = max(0, result['r_squared'])
            
            # Physical reasonableness: prefer exponents in reasonable range
            exponent = result['exponent']
            if -3 < exponent < 3:
                physics_score = 1.0
            elif -5 < exponent < 5:
                physics_score = 0.5
            else:
                physics_score = 0.0
            
            total_score = 0.7 * r2_score + 0.3 * physics_score
            
            if total_score > best_score:
                best_score = total_score
                best_result = result
        
        return best_result
    
    def _bootstrap_confidence_interval(self, temperatures, observable, critical_temperature, exponent_type):
        """Bootstrap confidence interval for exponent."""
        
        n_data = len(temperatures)
        bootstrap_exponents = []
        
        np.random.seed(42)  # For reproducibility
        
        for _ in range(min(500, self.bootstrap_samples)):  # Limit for speed
            # Bootstrap sample
            indices = np.random.choice(n_data, size=n_data, replace=True)
            boot_temps = temperatures[indices]
            boot_obs = observable[indices]
            
            try:
                # Use log-linear fit for bootstrap (faster)
                result = self._log_linear_fit(boot_temps, boot_obs, critical_temperature)
                bootstrap_exponents.append(result['exponent'])
            except:
                continue
        
        if len(bootstrap_exponents) < 50:
            raise ValueError("Too few successful bootstrap samples")
        
        bootstrap_exponents = np.array(bootstrap_exponents)
        
        # 95% confidence interval
        ci_lower = np.percentile(bootstrap_exponents, 2.5)
        ci_upper = np.percentile(bootstrap_exponents, 97.5)
        
        return ci_lower, ci_upper


def load_data_and_extract_latent(data_path):
    """Load data and create improved latent representation."""
    logger = get_logger(__name__)
    
    if data_path.endswith('.npz'):
        # 2D data
        logger.info(f"Loading 2D data from {data_path}")
        data = np.load(data_path, allow_pickle=True)
        
        configurations = data['spin_configurations']
        magnetizations = data['magnetizations']
        energies = data['energies']
        metadata = data['metadata'].item()
        
        # Extract temperatures
        n_temps = metadata['n_temperatures']
        temp_min, temp_max = metadata['temp_range']
        temperatures = np.linspace(temp_min, temp_max, n_temps)
        n_configs_per_temp = configurations.shape[0] // n_temps
        temp_array = np.repeat(temperatures, n_configs_per_temp)
        
        system_type = 'ising_2d'
        
    elif data_path.endswith('.h5'):
        # 3D data
        logger.info(f"Loading 3D data from {data_path}")
        with h5py.File(data_path, 'r') as f:
            configurations = f['configurations'][:]
            magnetizations = f['magnetizations'][:]
            energies = f['energies'][:]
            temp_array = f['temperatures'][:]
        
        system_type = 'ising_3d'
    
    else:
        raise ValueError(f"Unsupported data format: {data_path}")
    
    # Create improved latent representation
    logger.info("Creating improved order parameter representations")
    
    # Enhanced order parameter 1: Absolute magnetization
    z1 = np.abs(magnetizations)
    
    # Enhanced order parameter 2: Local susceptibility
    unique_temps = np.unique(temp_array)
    z2 = np.zeros_like(magnetizations)
    
    for temp in unique_temps:
        temp_mask = np.abs(temp_array - temp) < 0.01
        if np.sum(temp_mask) > 1:
            local_susceptibility = np.var(magnetizations[temp_mask])
            z2[temp_mask] = local_susceptibility
    
    # Smooth z2 to reduce noise
    if len(z2) > 11:
        z2 = savgol_filter(z2, 11, 3)
    
    # Create LatentRepresentation
    latent_repr = LatentRepresentation(
        z1=z1,
        z2=z2,
        temperatures=temp_array,
        magnetizations=magnetizations,
        energies=energies,
        reconstruction_errors=np.zeros_like(z1),
        sample_indices=np.arange(len(z1))
    )
    
    return latent_repr, system_type


def analyze_critical_exponents_robust(latent_repr, system_type):
    """Perform robust critical exponent analysis."""
    logger = get_logger(__name__)
    
    # Theoretical values
    theoretical_exponents = {
        'ising_2d': {'beta': 0.125, 'nu': 1.0},
        'ising_3d': {'beta': 0.326, 'nu': 0.630}
    }
    
    theoretical = theoretical_exponents.get(system_type, {'beta': 0.3, 'nu': 0.6})
    
    results = {
        'system_type': system_type,
        'theoretical': theoretical,
        'extracted': {},
        'accuracy': {}
    }
    
    # Step 1: Detect critical temperature
    tc_detector = RobustCriticalTemperatureDetector()
    tc, tc_confidence = tc_detector.detect_critical_temperature(
        latent_repr.temperatures, 
        latent_repr.z1  # Use absolute magnetization
    )
    
    results['critical_temperature'] = tc
    results['tc_confidence'] = tc_confidence
    
    # Step 2: Extract β exponent
    fitter = RobustPowerLawFitter()
    
    try:
        beta_result = fitter.fit_power_law(
            latent_repr.temperatures,
            latent_repr.z1,  # Absolute magnetization
            tc,
            exponent_type='beta'
        )
        
        results['extracted']['beta'] = beta_result
        
        # Calculate accuracy
        beta_accuracy = (1 - abs(beta_result['exponent'] - theoretical['beta']) / theoretical['beta']) * 100
        results['accuracy']['beta_accuracy_percent'] = max(0, beta_accuracy)
        
    except Exception as e:
        logger.error(f"β exponent extraction failed: {e}")
        results['extracted']['beta'] = None
    
    # Step 3: Extract ν exponent using correlation length proxy
    try:
        # Compute correlation length from latent space variance
        temps_binned, corr_lengths = compute_correlation_length_binned(latent_repr, tc)
        
        if len(temps_binned) >= 8:
            nu_result = fitter.fit_power_law(
                temps_binned,
                corr_lengths,
                tc,
                exponent_type='nu'
            )
            
            results['extracted']['nu'] = nu_result
            
            # Calculate accuracy
            nu_accuracy = (1 - abs(nu_result['exponent'] - theoretical['nu']) / theoretical['nu']) * 100
            results['accuracy']['nu_accuracy_percent'] = max(0, nu_accuracy)
            
        else:
            logger.warning("Insufficient binned data for ν exponent")
            results['extracted']['nu'] = None
            
    except Exception as e:
        logger.error(f"ν exponent extraction failed: {e}")
        results['extracted']['nu'] = None
    
    # Overall accuracy
    if results['extracted']['beta'] and results['extracted']['nu']:
        overall_accuracy = (results['accuracy']['beta_accuracy_percent'] + 
                          results['accuracy']['nu_accuracy_percent']) / 2
        results['accuracy']['overall_accuracy_percent'] = overall_accuracy
    
    return results


def compute_correlation_length_binned(latent_repr, critical_temperature):
    """Compute correlation length using temperature binning."""
    
    # Create temperature bins
    n_bins = min(25, len(np.unique(latent_repr.temperatures)) // 2)
    temp_min, temp_max = np.min(latent_repr.temperatures), np.max(latent_repr.temperatures)
    temp_bins = np.linspace(temp_min, temp_max, n_bins + 1)
    temp_centers = (temp_bins[:-1] + temp_bins[1:]) / 2
    
    correlation_lengths = []
    valid_temps = []
    
    for i in range(len(temp_bins) - 1):
        temp_mask = (latent_repr.temperatures >= temp_bins[i]) & \
                   (latent_repr.temperatures < temp_bins[i + 1])
        
        if np.sum(temp_mask) >= 5:
            # Use combined variance of latent dimensions
            z1_var = np.var(latent_repr.z1[temp_mask])
            z2_var = np.var(latent_repr.z2[temp_mask])
            
            # Correlation length proxy
            corr_length = np.sqrt(z1_var + z2_var)
            
            if corr_length > 0 and np.isfinite(corr_length):
                correlation_lengths.append(corr_length)
                valid_temps.append(temp_centers[i])
    
    return np.array(valid_temps), np.array(correlation_lengths)


def print_robust_results(results):
    """Print formatted results."""
    
    print("\n" + "=" * 60)
    print(f"ROBUST {results['system_type'].upper()} CRITICAL EXPONENT ANALYSIS")
    print("=" * 60)
    
    print(f"System: {results['system_type']}")
    print(f"Critical Temperature: {results['critical_temperature']:.4f} (confidence: {results['tc_confidence']:.3f})")
    print()
    
    theoretical = results['theoretical']
    extracted = results['extracted']
    accuracy = results['accuracy']
    
    # β exponent
    if extracted.get('beta'):
        beta = extracted['beta']
        print("β EXPONENT (Order Parameter):")
        print(f"  Measured: {beta['exponent']:.4f} ± {beta['exponent_error']:.4f}")
        print(f"  Theoretical: {theoretical['beta']:.4f}")
        print(f"  Accuracy: {accuracy.get('beta_accuracy_percent', 0):.1f}%")
        print(f"  R²: {beta['r_squared']:.4f}")
        print(f"  Method: {beta['method']}")
        print(f"  Fit points: {beta['n_points']}")
        
        if beta.get('confidence_interval'):
            ci_lower, ci_upper = beta['confidence_interval']
            print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            # Check if theoretical is in CI
            in_ci = ci_lower <= theoretical['beta'] <= ci_upper
            print(f"  Theoretical in CI: {'✓' if in_ci else '✗'}")
        print()
    
    # ν exponent
    if extracted.get('nu'):
        nu = extracted['nu']
        print("ν EXPONENT (Correlation Length):")
        print(f"  Measured: {nu['exponent']:.4f} ± {nu['exponent_error']:.4f}")
        print(f"  Theoretical: {theoretical['nu']:.4f}")
        print(f"  Accuracy: {accuracy.get('nu_accuracy_percent', 0):.1f}%")
        print(f"  R²: {nu['r_squared']:.4f}")
        print(f"  Method: {nu['method']}")
        print(f"  Fit points: {nu['n_points']}")
        
        if nu.get('confidence_interval'):
            ci_lower, ci_upper = nu['confidence_interval']
            print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            # Check if theoretical is in CI
            in_ci = ci_lower <= theoretical['nu'] <= ci_upper
            print(f"  Theoretical in CI: {'✓' if in_ci else '✗'}")
        print()
    
    # Overall performance
    if 'overall_accuracy_percent' in accuracy:
        overall_acc = accuracy['overall_accuracy_percent']
        print("OVERALL PERFORMANCE:")
        print(f"  Combined Accuracy: {overall_acc:.1f}%")
        
        if overall_acc >= 85:
            rating = "Excellent"
        elif overall_acc >= 75:
            rating = "Good"
        elif overall_acc >= 65:
            rating = "Fair"
        else:
            rating = "Needs Improvement"
        
        print(f"  Performance Rating: {rating}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Robust critical exponent extraction')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to data file (.npz for 2D, .h5 for 3D)')
    parser.add_argument('--output-dir', type=str, default='results/robust_analysis',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Setup logging
    config = PrometheusConfig()
    setup_logging(config.logging)
    
    logger = get_logger(__name__)
    logger.info("Starting robust critical exponent analysis")
    
    try:
        # Load data and create latent representation
        latent_repr, system_type = load_data_and_extract_latent(args.data_path)
        
        # Analyze critical exponents
        results = analyze_critical_exponents_robust(latent_repr, system_type)
        
        # Print results
        print_robust_results(results)
        
        # Save results
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / f"{system_type}_robust_results.npz"
        np.savez(results_file, **results)
        
        logger.info(f"Results saved to {results_file}")
        logger.info("Robust analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()