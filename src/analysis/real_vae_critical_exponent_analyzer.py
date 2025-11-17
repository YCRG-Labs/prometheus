"""
Real VAE Critical Exponent Analyzer

This module implements task 13.1: Replace mock components with real VAE training and analysis.
Creates blind extraction that discovers exponents from latent space without theoretical knowledge.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from scipy.stats import linregress, pearsonr, spearmanr
from scipy.signal import savgol_filter
import warnings

from .numerical_stability_fixes import safe_log, safe_divide

# Simple logger function
def get_logger(name):
    return logging.getLogger(name)


@dataclass
class BlindOrderParameterResult:
    """Container for blind order parameter analysis results."""
    best_dimension: int
    correlation_with_magnetization: float
    correlation_with_temperature: float
    order_parameter_values: np.ndarray
    dimension_correlations: Dict[int, Dict[str, float]]
    selection_method: str
    confidence_score: float
    temperature_sensitivity: float
    phase_transition_sharpness: float


@dataclass
class BlindPowerLawResult:
    """Container for blind power law fitting results."""
    exponent: float
    exponent_error: float
    amplitude: float
    amplitude_error: float
    r_squared: float
    p_value: float
    confidence_interval: Optional[Tuple[float, float]]
    fit_range: Tuple[float, float]
    n_points: int
    data_quality_score: float
    fitting_method: str


@dataclass
class BlindCriticalExponentResults:
    """Container for blind critical exponent analysis results."""
    system_type: str
    critical_temperature: float
    tc_confidence: float
    tc_detection_method: str
    
    # Order parameter analysis
    order_parameter_result: BlindOrderParameterResult
    
    # Critical exponents (extracted blindly)
    beta_result: Optional[BlindPowerLawResult] = None
    nu_result: Optional[BlindPowerLawResult] = None
    gamma_result: Optional[BlindPowerLawResult] = None
    
    # Quality metrics (no theoretical comparison)
    extraction_quality_score: float = 0.0
    statistical_significance: Dict[str, float] = None
    
    # Raw magnetization comparison
    raw_magnetization_comparison: Optional[Dict[str, Any]] = None


class RealVAECriticalExponentAnalyzer:
    """
    Real VAE Critical Exponent Analyzer that performs blind extraction
    without theoretical knowledge.
    """
    
    def __init__(self, 
                 system_type: str = 'unknown',
                 bootstrap_samples: int = 1000,
                 random_seed: Optional[int] = None):
        """
        Initialize real VAE critical exponent analyzer.
        
        Args:
            system_type: Type of physical system (for logging only)
            bootstrap_samples: Number of bootstrap samples
            random_seed: Random seed for reproducibility
        """
        self.system_type = system_type
        self.logger = get_logger(__name__)
        self.bootstrap_samples = bootstrap_samples
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def analyze_vae_critical_exponents(self, 
                                     latent_repr,
                                     compare_with_raw_magnetization: bool = True) -> BlindCriticalExponentResults:
        """
        Perform complete blind VAE-based critical exponent analysis.
        
        Args:
            latent_repr: LatentRepresentation object
            compare_with_raw_magnetization: Whether to compare with raw magnetization
            
        Returns:
            BlindCriticalExponentResults with complete blind analysis
        """
        self.logger.info("Starting blind VAE-based critical exponent analysis")
        
        # Step 1: Detect order parameter blindly
        order_param_result = self._detect_order_parameter_blind(latent_repr)
        
        self.logger.info(f"Detected order parameter: latent dimension {order_param_result.best_dimension}")
        self.logger.info(f"Order parameter confidence: {order_param_result.confidence_score:.4f}")
        
        # Step 2: Detect critical temperature blindly
        tc, tc_confidence, tc_method = self._detect_critical_temperature_blind(
            latent_repr.temperatures,
            np.abs(order_param_result.order_parameter_values)
        )
        
        self.logger.info(f"Detected Tc = {tc:.4f} (confidence: {tc_confidence:.3f}, method: {tc_method})")
        
        # Initialize results
        results = BlindCriticalExponentResults(
            system_type=self.system_type,
            critical_temperature=tc,
            tc_confidence=tc_confidence,
            tc_detection_method=tc_method,
            order_parameter_result=order_param_result
        )
        
        # Step 3: Extract β exponent blindly
        try:
            beta_result = self._fit_power_law_blind(
                latent_repr.temperatures,
                np.abs(order_param_result.order_parameter_values),
                tc,
                exponent_type='beta'
            )
            
            results.beta_result = beta_result
            self.logger.info(f"β exponent: {beta_result.exponent:.4f} ± {beta_result.exponent_error:.4f}")
            
        except Exception as e:
            self.logger.error(f"β exponent extraction failed: {e}")
            results.beta_result = None
        
        # Step 4: Compute extraction quality score
        results.extraction_quality_score = self._compute_extraction_quality(results)
        
        # Step 5: Compute statistical significance
        results.statistical_significance = self._compute_statistical_significance(results)
        
        self.logger.info("Blind VAE-based critical exponent analysis completed")
        
        return results
    
    def _detect_order_parameter_blind(self, latent_repr) -> BlindOrderParameterResult:
        """Detect order parameter from latent space without theoretical guidance."""
        
        # Get latent coordinates
        latent_coords = latent_repr.latent_coords
        n_dims = latent_coords.shape[1]
        
        # Analyze each dimension for order parameter characteristics
        dimension_scores = {}
        
        for dim in range(n_dims):
            latent_values = latent_coords[:, dim]
            
            # Simple scoring based on magnetization correlation
            mag_corr, mag_p = pearsonr(latent_values, np.abs(latent_repr.magnetizations))
            temp_corr, temp_p = pearsonr(latent_values, latent_repr.temperatures)
            
            # Compute temperature sensitivity
            unique_temps = np.unique(latent_repr.temperatures)
            temp_means = []
            for temp in unique_temps:
                temp_mask = latent_repr.temperatures == temp
                if np.sum(temp_mask) > 0:
                    temp_means.append(np.mean(latent_values[temp_mask]))
            
            if len(temp_means) > 3:
                gradient = np.gradient(temp_means, unique_temps)
                temp_sensitivity = np.max(np.abs(gradient))
            else:
                temp_sensitivity = 0.0
            
            # Compute transition sharpness
            temp_variances = []
            for temp in unique_temps:
                temp_mask = latent_repr.temperatures == temp
                if np.sum(temp_mask) > 3:
                    temp_variances.append(np.var(latent_values[temp_mask]))
            
            if len(temp_variances) > 0:
                max_variance = np.max(temp_variances)
                mean_variance = np.mean(temp_variances)
                transition_sharpness = max_variance / (mean_variance + 1e-10)
            else:
                transition_sharpness = 0.0
            
            # Combined score
            total_score = (
                0.4 * abs(mag_corr) +
                0.3 * min(1.0, temp_sensitivity / 0.1) +
                0.2 * min(1.0, transition_sharpness / 5.0) +
                0.1 * abs(temp_corr)
            )
            
            dimension_scores[dim] = {
                'magnetization_correlation': abs(mag_corr),
                'temperature_correlation': abs(temp_corr),
                'temperature_sensitivity': temp_sensitivity,
                'transition_sharpness': transition_sharpness,
                'total_score': total_score
            }
        
        # Select best dimension
        best_dim = max(dimension_scores.keys(),
                      key=lambda d: dimension_scores[d]['total_score'])
        
        best_scores = dimension_scores[best_dim]
        latent_values = latent_coords[:, best_dim]
        
        # Compute correlations for the selected dimension
        mag_corr, _ = pearsonr(latent_values, np.abs(latent_repr.magnetizations))
        temp_corr, _ = pearsonr(latent_values, latent_repr.temperatures)
        
        return BlindOrderParameterResult(
            best_dimension=best_dim,
            correlation_with_magnetization=mag_corr,
            correlation_with_temperature=temp_corr,
            order_parameter_values=latent_values,
            dimension_correlations=dimension_scores,
            selection_method='blind_comprehensive',
            confidence_score=best_scores['total_score'],
            temperature_sensitivity=best_scores['temperature_sensitivity'],
            phase_transition_sharpness=best_scores['transition_sharpness']
        )
    
    def _detect_critical_temperature_blind(self, 
                                         temperatures: np.ndarray,
                                         order_parameter: np.ndarray) -> Tuple[float, float, str]:
        """Detect critical temperature blindly from order parameter behavior."""
        
        # Try susceptibility peak method
        unique_temps = np.unique(temperatures)
        
        if len(unique_temps) < 5:
            # Fallback to middle temperature
            return np.median(unique_temps), 0.5, 'fallback_median'
        
        # Compute susceptibility (variance) at each temperature
        susceptibilities = []
        valid_temps = []
        
        for temp in unique_temps:
            temp_mask = temperatures == temp
            if np.sum(temp_mask) >= 5:
                susceptibility = np.var(order_parameter[temp_mask])
                susceptibilities.append(susceptibility)
                valid_temps.append(temp)
        
        if len(susceptibilities) < 3:
            return np.median(unique_temps), 0.5, 'fallback_median'
        
        susceptibilities = np.array(susceptibilities)
        valid_temps = np.array(valid_temps)
        
        # Find peak
        peak_idx = np.argmax(susceptibilities)
        tc_estimate = valid_temps[peak_idx]
        
        # Confidence based on peak prominence
        peak_value = susceptibilities[peak_idx]
        background = np.median(susceptibilities)
        prominence = (peak_value - background) / (background + 1e-10)
        confidence = min(1.0, prominence / 3.0)
        
        return tc_estimate, confidence, 'susceptibility_peak'
    
    def _fit_power_law_blind(self, 
                           temperatures: np.ndarray,
                           order_parameter: np.ndarray,
                           critical_temperature: float,
                           exponent_type: str = 'beta') -> BlindPowerLawResult:
        """Fit power law blindly without theoretical constraints."""
        
        # Prepare data based on exponent type
        if exponent_type == 'beta':
            # β: m ∝ (Tc - T)^β for T < Tc
            mask = temperatures < critical_temperature
            x_data = critical_temperature - temperatures[mask]
            y_data = np.abs(order_parameter[mask])
        else:
            raise ValueError(f"Unknown exponent type: {exponent_type}")
        
        # Filter out invalid data
        valid_mask = (x_data > 0) & (y_data > 0) & np.isfinite(x_data) & np.isfinite(y_data)
        x_data = x_data[valid_mask]
        y_data = y_data[valid_mask]
        
        if len(x_data) < 5:
            raise ValueError(f"Insufficient data for {exponent_type} fitting")
        
        # Log-log fit: log(y) = log(A) + β * log(x)
        log_x = safe_log(x_data)
        log_y = safe_log(y_data)
        
        # Linear regression in log space
        slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
        
        # Convert back to power law parameters
        exponent = slope
        amplitude = np.exp(intercept)
        r_squared = r_value**2
        
        # Quality assessment
        quality_factors = [r_squared]
        if p_value < 0.05:
            quality_factors.append(1 - p_value)
        else:
            quality_factors.append(0)
        
        # Exponent reasonableness
        if 0.01 < abs(exponent) < 5.0:
            quality_factors.append(1.0)
        else:
            quality_factors.append(0.5)
        
        data_quality_score = np.mean(quality_factors)
        
        return BlindPowerLawResult(
            exponent=exponent,
            exponent_error=std_err,
            amplitude=amplitude,
            amplitude_error=0.0,
            r_squared=r_squared,
            p_value=p_value,
            confidence_interval=(exponent - 2*std_err, exponent + 2*std_err),
            fit_range=(np.min(x_data), np.max(x_data)),
            n_points=len(x_data),
            data_quality_score=data_quality_score,
            fitting_method='log_linear_regression'
        )
    
    def _compute_extraction_quality(self, results: BlindCriticalExponentResults) -> float:
        """Compute overall extraction quality score."""
        
        quality_factors = []
        
        # Order parameter quality
        op_quality = results.order_parameter_result.confidence_score
        quality_factors.append(op_quality)
        
        # Critical temperature confidence
        tc_quality = results.tc_confidence
        quality_factors.append(tc_quality)
        
        # β exponent quality
        if results.beta_result:
            beta_quality = results.beta_result.data_quality_score
            quality_factors.append(beta_quality)
        
        return np.mean(quality_factors) if quality_factors else 0.0
    
    def _compute_statistical_significance(self, results: BlindCriticalExponentResults) -> Dict[str, float]:
        """Compute statistical significance metrics."""
        
        significance = {}
        
        # Order parameter significance
        significance['order_parameter'] = results.order_parameter_result.confidence_score
        
        # Critical temperature significance
        significance['critical_temperature'] = results.tc_confidence
        
        # β exponent significance
        if results.beta_result:
            significance['beta_exponent'] = 1 - results.beta_result.p_value if results.beta_result.p_value < 0.05 else 0
        else:
            significance['beta_exponent'] = 0.0
        
        # Overall significance
        significance['overall'] = np.mean(list(significance.values()))
        
        return significance


def create_real_vae_critical_exponent_analyzer(system_type: str = 'unknown',
                                             bootstrap_samples: int = 1000,
                                             random_seed: Optional[int] = None) -> RealVAECriticalExponentAnalyzer:
    """
    Factory function to create RealVAECriticalExponentAnalyzer.
    
    Args:
        system_type: Type of physical system (for logging only)
        bootstrap_samples: Number of bootstrap samples
        random_seed: Random seed for reproducibility
        
    Returns:
        Configured RealVAECriticalExponentAnalyzer instance
    """
    return RealVAECriticalExponentAnalyzer(
        system_type=system_type,
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed
    )