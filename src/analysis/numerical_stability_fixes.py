"""
Numerical Stability Fixes for Critical Exponent Extraction

This module provides fixes for numerical stability issues identified in the accuracy assessment:
- Division by zero errors in gradient calculations
- Robust fitting bounds problems
- Invalid value handling in power-law fitting
"""

import numpy as np
from typing import Tuple, Optional
from scipy.optimize import curve_fit, OptimizeWarning
import warnings


def safe_log(x: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Safely compute logarithm, avoiding log(0) and log(negative).
    
    Args:
        x: Input array
        epsilon: Small positive value to add to avoid log(0)
        
    Returns:
        Logarithm of x with numerical stability
    """
    x_safe = np.maximum(x, epsilon)
    return np.log(x_safe)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, 
                epsilon: float = 1e-10, fill_value: float = 0.0) -> np.ndarray:
    """
    Safely divide arrays, avoiding division by zero.
    
    Args:
        numerator: Numerator array
        denominator: Denominator array
        epsilon: Small value to add to denominator
        fill_value: Value to use when denominator is zero
        
    Returns:
        Division result with numerical stability
    """
    # Add epsilon to denominator to avoid division by zero
    denominator_safe = np.where(np.abs(denominator) < epsilon, epsilon, denominator)
    
    # Perform division
    result = numerator / denominator_safe
    
    # Replace inf/nan with fill_value
    result = np.where(np.isfinite(result), result, fill_value)
    
    return result


def safe_power_law_fit(x: np.ndarray, y: np.ndarray, 
                      initial_guess: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Safely fit power law y = A * x^b with numerical stability.
    
    Args:
        x: Independent variable (must be positive)
        y: Dependent variable (must be positive)
        initial_guess: Initial guess for (amplitude, exponent)
        
    Returns:
        Tuple of (parameters, covariance, success)
    """
    # Filter valid data
    valid_mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    if len(x_valid) < 3:
        return np.array([1.0, 1.0]), np.eye(2), False
    
    # Log-space fitting for better numerical stability
    log_x = safe_log(x_valid)
    log_y = safe_log(y_valid)
    
    # Linear fit in log space: log(y) = log(A) + b*log(x)
    try:
        # Use polyfit which is more stable than direct linear regression
        coeffs = np.polyfit(log_x, log_y, 1)
        exponent = coeffs[0]
        log_amplitude = coeffs[1]
        amplitude = np.exp(log_amplitude)
        
        # Estimate covariance from residuals
        y_pred = amplitude * (x_valid ** exponent)
        residuals = y_valid - y_pred
        residual_var = np.var(residuals)
        
        # Simple covariance estimate
        cov = np.eye(2) * residual_var
        
        return np.array([amplitude, exponent]), cov, True
        
    except Exception as e:
        warnings.warn(f"Power law fit failed: {e}")
        return np.array([1.0, 1.0]), np.eye(2), False


def safe_gradient(y: np.ndarray, x: np.ndarray, 
                 edge_order: int = 1) -> np.ndarray:
    """
    Safely compute gradient, avoiding numerical issues.
    
    Args:
        y: Function values
        x: Independent variable values
        edge_order: Order of edge approximation
        
    Returns:
        Gradient with numerical stability
    """
    if len(y) < 2 or len(x) < 2:
        return np.zeros_like(y)
    
    # Ensure x is sorted
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Compute differences
    dx = np.diff(x_sorted)
    dy = np.diff(y_sorted)
    
    # Avoid division by zero in dx
    dx_safe = np.where(np.abs(dx) < 1e-10, 1e-10, dx)
    
    # Compute gradient
    gradient = dy / dx_safe
    
    # Extend to match input size (use edge values)
    if edge_order == 1:
        gradient = np.concatenate([[gradient[0]], gradient])
    else:
        gradient = np.concatenate([gradient, [gradient[-1]]])
    
    # Unsort to match original order
    unsort_idx = np.argsort(sort_idx)
    gradient = gradient[unsort_idx]
    
    return gradient


def robust_power_law_bounds(x_data: np.ndarray, y_data: np.ndarray,
                           exponent_range: Tuple[float, float] = (-5.0, 5.0),
                           amplitude_range: Optional[Tuple[float, float]] = None) -> Tuple[Tuple, Tuple]:
    """
    Compute robust bounds for power-law fitting.
    
    Args:
        x_data: Independent variable data
        y_data: Dependent variable data
        exponent_range: Range for exponent parameter
        amplitude_range: Range for amplitude parameter (auto if None)
        
    Returns:
        Tuple of (lower_bounds, upper_bounds)
    """
    # Estimate amplitude range from data if not provided
    if amplitude_range is None:
        y_min = np.min(y_data[y_data > 0]) if np.any(y_data > 0) else 1e-10
        y_max = np.max(y_data[y_data > 0]) if np.any(y_data > 0) else 1.0
        
        # Amplitude should be in reasonable range around data
        amplitude_min = y_min * 0.01
        amplitude_max = y_max * 100
    else:
        amplitude_min, amplitude_max = amplitude_range
    
    # Ensure bounds are valid
    amplitude_min = max(amplitude_min, 1e-10)
    amplitude_max = max(amplitude_max, amplitude_min * 10)
    
    exponent_min, exponent_max = exponent_range
    
    # Ensure exponent bounds are valid
    if exponent_min >= exponent_max:
        exponent_min = -5.0
        exponent_max = 5.0
    
    lower_bounds = (amplitude_min, exponent_min)
    upper_bounds = (amplitude_max, exponent_max)
    
    return lower_bounds, upper_bounds


def clean_data_for_fitting(x: np.ndarray, y: np.ndarray,
                          remove_outliers: bool = True,
                          outlier_threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Clean data for fitting by removing invalid values and optionally outliers.
    
    Args:
        x: Independent variable
        y: Dependent variable
        remove_outliers: Whether to remove outliers
        outlier_threshold: Number of standard deviations for outlier detection
        
    Returns:
        Tuple of (x_clean, y_clean, valid_mask)
    """
    # Remove invalid values
    valid_mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    
    if not np.any(valid_mask):
        return np.array([]), np.array([]), valid_mask
    
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    
    if remove_outliers and len(y_clean) > 10:
        # Remove outliers in log space for power-law data
        log_y = safe_log(y_clean)
        
        # Z-score method
        mean_log_y = np.mean(log_y)
        std_log_y = np.std(log_y)
        
        if std_log_y > 0:
            z_scores = np.abs((log_y - mean_log_y) / std_log_y)
            outlier_mask = z_scores < outlier_threshold
            
            x_clean = x_clean[outlier_mask]
            y_clean = y_clean[outlier_mask]
            
            # Update valid_mask
            temp_mask = valid_mask.copy()
            temp_mask[valid_mask] = outlier_mask
            valid_mask = temp_mask
    
    return x_clean, y_clean, valid_mask


def estimate_initial_guess(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Estimate initial guess for power-law parameters.
    
    Args:
        x: Independent variable (positive)
        y: Dependent variable (positive)
        
    Returns:
        Tuple of (amplitude, exponent) initial guess
    """
    if len(x) < 2 or len(y) < 2:
        return (1.0, 1.0)
    
    # Use log-space linear regression for initial guess
    log_x = safe_log(x)
    log_y = safe_log(y)
    
    try:
        # Simple linear regression
        coeffs = np.polyfit(log_x, log_y, 1)
        exponent = coeffs[0]
        log_amplitude = coeffs[1]
        amplitude = np.exp(log_amplitude)
        
        # Ensure reasonable values
        amplitude = np.clip(amplitude, 1e-10, 1e10)
        exponent = np.clip(exponent, -5, 5)
        
        return (amplitude, exponent)
        
    except:
        return (1.0, 1.0)


def validate_fit_result(params: np.ndarray, x: np.ndarray, y: np.ndarray,
                       max_relative_error: float = 10.0) -> bool:
    """
    Validate that fit result is reasonable.
    
    Args:
        params: Fitted parameters [amplitude, exponent]
        x: Independent variable
        y: Dependent variable
        max_relative_error: Maximum acceptable relative error
        
    Returns:
        True if fit is valid, False otherwise
    """
    if len(params) != 2:
        return False
    
    amplitude, exponent = params
    
    # Check parameter reasonableness
    if not np.isfinite(amplitude) or not np.isfinite(exponent):
        return False
    
    if amplitude <= 0 or amplitude > 1e10:
        return False
    
    if abs(exponent) > 10:
        return False
    
    # Check fit quality
    try:
        y_pred = amplitude * (x ** exponent)
        
        if not np.all(np.isfinite(y_pred)):
            return False
        
        relative_errors = np.abs((y - y_pred) / (y + 1e-10))
        mean_relative_error = np.mean(relative_errors)
        
        if mean_relative_error > max_relative_error:
            return False
        
        return True
        
    except:
        return False


# Convenience function for complete safe power-law fitting
def fit_power_law_safe(x: np.ndarray, y: np.ndarray,
                      exponent_range: Tuple[float, float] = (-5.0, 5.0),
                      remove_outliers: bool = True) -> dict:
    """
    Complete safe power-law fitting with all stability measures.
    
    Args:
        x: Independent variable
        y: Dependent variable
        exponent_range: Range for exponent
        remove_outliers: Whether to remove outliers
        
    Returns:
        Dictionary with fit results and quality metrics
    """
    # Clean data
    x_clean, y_clean, valid_mask = clean_data_for_fitting(x, y, remove_outliers)
    
    if len(x_clean) < 3:
        return {
            'success': False,
            'amplitude': 1.0,
            'exponent': 1.0,
            'amplitude_error': 0.0,
            'exponent_error': 0.0,
            'r_squared': 0.0,
            'n_points': len(x_clean),
            'message': 'Insufficient valid data points'
        }
    
    # Estimate initial guess
    initial_guess = estimate_initial_guess(x_clean, y_clean)
    
    # Compute bounds
    lower_bounds, upper_bounds = robust_power_law_bounds(x_clean, y_clean, exponent_range)
    
    # Fit power law
    params, cov, success = safe_power_law_fit(x_clean, y_clean, initial_guess)
    
    if not success:
        return {
            'success': False,
            'amplitude': params[0],
            'exponent': params[1],
            'amplitude_error': 0.0,
            'exponent_error': 0.0,
            'r_squared': 0.0,
            'n_points': len(x_clean),
            'message': 'Fitting failed'
        }
    
    # Validate result
    if not validate_fit_result(params, x_clean, y_clean):
        return {
            'success': False,
            'amplitude': params[0],
            'exponent': params[1],
            'amplitude_error': 0.0,
            'exponent_error': 0.0,
            'r_squared': 0.0,
            'n_points': len(x_clean),
            'message': 'Fit validation failed'
        }
    
    # Compute R-squared
    amplitude, exponent = params
    y_pred = amplitude * (x_clean ** exponent)
    ss_res = np.sum((y_clean - y_pred) ** 2)
    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Extract errors from covariance
    amplitude_error = np.sqrt(cov[0, 0]) if cov[0, 0] > 0 else 0.0
    exponent_error = np.sqrt(cov[1, 1]) if cov[1, 1] > 0 else 0.0
    
    return {
        'success': True,
        'amplitude': amplitude,
        'exponent': exponent,
        'amplitude_error': amplitude_error,
        'exponent_error': exponent_error,
        'r_squared': r_squared,
        'n_points': len(x_clean),
        'message': 'Fit successful'
    }