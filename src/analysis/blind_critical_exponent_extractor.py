"""
Blind Critical Exponent Extraction from Real Latent Space

This module implements task 13.3: Extract critical exponents from actual VAE latent
representations without theoretical guidance. Implements unsupervised order parameter
identification and power-law fitting to real latent space behavior.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from scipy.stats import linregress, pearsonr, spearmanr
from scipy.optimize import curve_fit, minimize_scalar, differential_evolution
from scipy.signal import savgol_filter, find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings

from .numerical_stability_fixes import safe_log, safe_divide

# Simple logger function
def get_logger(name):
    return logging.getLogger(name)


@dataclass
class BlindOrderParameterAnalysis:
    """Results from blind order parameter identification."""
    selected_dimension: int
    order_parameter_values: np.ndarray
    selection_confidence: float
    
    # Analysis metrics
    temperature_sensitivity: float
    phase_transition_sharpness: float
    monotonicity_score: float
    variance_ratio: float
    
    # Correlation metrics (for validation, not used in selection)
    magnetization_correlation: float
    temperature_correlation: float
    
    # Selection method details
    dimension_scores: Dict[int, Dict[str, float]]
    selection_method: str
    quality_metrics: Dict[str, float]


@dataclass
class BlindCriticalTemperatureDetection:
    """Results from blind critical temperature detection."""
    critical_temperature: float
    detection_confidence: float
    detection_method: str
    
    # Supporting evidence
    susceptibility_peak: Optional[float]
    derivative_peak: Optional[float]
    variance_peak: Optional[float]
    
    # Quality metrics
    peak_prominence: float
    peak_width: float
    signal_to_noise: float
    
    # Method comparison
    method_results: Dict[str, float]
    ensemble_weight: Dict[str, float]


@dataclass
class BlindPowerLawFit:
    """Results from blind power law fitting."""
    exponent: float
    exponent_error: float
    amplitude: float
    amplitude_error: float
    
    # Fit quality
    r_squared: float
    p_value: float
    chi_squared: float
    degrees_of_freedom: int
    
    # Confidence intervals
    exponent_ci_lower: float
    exponent_ci_upper: float
    
    # Fitting details
    temperature_range: Tuple[float, float]
    n_data_points: int
    fitting_method: str
    optimization_success: bool
    
    # Quality assessment
    fit_quality_score: float
    residual_analysis: Dict[str, float]


@dataclass
class BlindCriticalExponentResults:
    """Complete results from blind critical exponent extraction."""
    system_identifier: str
    
    # Order parameter analysis
    order_parameter_analysis: BlindOrderParameterAnalysis
    
    # Critical temperature detection
    tc_detection: BlindCriticalTemperatureDetection
    
    # Critical exponents
    beta_exponent: Optional[BlindPowerLawFit]
    nu_exponent: Optional[BlindPowerLawFit]
    gamma_exponent: Optional[BlindPowerLawFit]
    
    # Overall quality assessment
    extraction_quality_score: float
    reliability_metrics: Dict[str, float]
    
    # Comparison with raw magnetization (for validation)
    raw_magnetization_comparison: Optional[Dict[str, Any]]


class BlindOrderParameterIdentifier:
    """Identifies order parameter from latent space without theoretical knowledge."""
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize blind order parameter identifier."""
        self.logger = get_logger(__name__)
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def identify_order_parameter(self,
                                latent_representations: np.ndarray,
                                temperatures: np.ndarray,
                                magnetizations: Optional[np.ndarray] = None) -> BlindOrderParameterAnalysis:
        """
        Identify order parameter from latent space without theoretical guidance.
        
        Args:
            latent_representations: Latent space coordinates (N, latent_dim)
            temperatures: Temperature values (N,)
            magnetizations: Magnetization values for validation only (N,)
            
        Returns:
            BlindOrderParameterAnalysis with identified order parameter
        """
        self.logger.info("Starting blind order parameter identification")
        
        n_samples, latent_dim = latent_representations.shape
        
        # Analyze each latent dimension
        dimension_scores = {}
        
        for dim in range(latent_dim):
            latent_values = latent_representations[:, dim]
            
            # Compute various quality metrics
            scores = self._analyze_dimension_quality(latent_values, temperatures)
            dimension_scores[dim] = scores
        
        # Select best dimension based on comprehensive scoring
        best_dim = self._select_best_dimension(dimension_scores)
        
        # Extract order parameter values
        order_parameter_values = latent_representations[:, best_dim]
        
        # Compute final quality metrics
        quality_metrics = self._compute_quality_metrics(
            order_parameter_values, temperatures, dimension_scores[best_dim]
        )
        
        # Compute correlations for validation (not used in selection)
        mag_correlation = 0.0
        temp_correlation = 0.0
        
        if magnetizations is not None:
            mag_corr, _ = pearsonr(order_parameter_values, np.abs(magnetizations))
            mag_correlation = mag_corr if not np.isnan(mag_corr) else 0.0
        
        temp_corr, _ = pearsonr(order_parameter_values, temperatures)
        temp_correlation = temp_corr if not np.isnan(temp_corr) else 0.0
        
        self.logger.info(f"Selected latent dimension {best_dim} as order parameter")
        self.logger.info(f"Selection confidence: {quality_metrics['selection_confidence']:.3f}")
        
        return BlindOrderParameterAnalysis(
            selected_dimension=best_dim,
            order_parameter_values=order_parameter_values,
            selection_confidence=quality_metrics['selection_confidence'],
            temperature_sensitivity=dimension_scores[best_dim]['temperature_sensitivity'],
            phase_transition_sharpness=dimension_scores[best_dim]['transition_sharpness'],
            monotonicity_score=dimension_scores[best_dim]['monotonicity_score'],
            variance_ratio=dimension_scores[best_dim]['variance_ratio'],
            magnetization_correlation=mag_correlation,
            temperature_correlation=temp_correlation,
            dimension_scores=dimension_scores,
            selection_method='comprehensive_blind',
            quality_metrics=quality_metrics
        )
    
    def _analyze_dimension_quality(self, 
                                  latent_values: np.ndarray,
                                  temperatures: np.ndarray) -> Dict[str, float]:
        """Analyze quality metrics for a single latent dimension."""
        
        # 1. Temperature sensitivity
        temp_sensitivity = self._compute_temperature_sensitivity(latent_values, temperatures)
        
        # 2. Phase transition sharpness
        transition_sharpness = self._compute_transition_sharpness(latent_values, temperatures)
        
        # 3. Monotonicity score
        monotonicity_score = self._compute_monotonicity_score(latent_values, temperatures)
        
        # 4. Variance ratio (high/low temperature variance ratio)
        variance_ratio = self._compute_variance_ratio(latent_values, temperatures)
        
        # 5. Signal-to-noise ratio
        signal_to_noise = self._compute_signal_to_noise(latent_values, temperatures)
        
        # 6. Dynamic range
        dynamic_range = self._compute_dynamic_range(latent_values, temperatures)
        
        return {
            'temperature_sensitivity': temp_sensitivity,
            'transition_sharpness': transition_sharpness,
            'monotonicity_score': monotonicity_score,
            'variance_ratio': variance_ratio,
            'signal_to_noise': signal_to_noise,
            'dynamic_range': dynamic_range
        }
    
    def _compute_temperature_sensitivity(self, 
                                       latent_values: np.ndarray,
                                       temperatures: np.ndarray) -> float:
        """Compute temperature sensitivity of latent dimension."""
        
        # Bin by temperature
        unique_temps = np.unique(temperatures)
        if len(unique_temps) < 3:
            return 0.0
        
        temp_means = []
        for temp in sorted(unique_temps):
            temp_mask = temperatures == temp
            if np.sum(temp_mask) > 0:
                temp_means.append(np.mean(latent_values[temp_mask]))
        
        if len(temp_means) < 3:
            return 0.0
        
        # Compute gradient
        temp_means = np.array(temp_means)
        gradient = np.gradient(temp_means, sorted(unique_temps))
        
        # Return maximum absolute gradient (normalized)
        max_gradient = np.max(np.abs(gradient))
        temp_range = np.max(unique_temps) - np.min(unique_temps)
        
        return max_gradient / (temp_range + 1e-10)
    
    def _compute_transition_sharpness(self,
                                    latent_values: np.ndarray,
                                    temperatures: np.ndarray) -> float:
        """Compute sharpness of phase transition."""
        
        # Compute variance at each temperature
        unique_temps = np.unique(temperatures)
        temp_variances = []
        
        for temp in unique_temps:
            temp_mask = temperatures == temp
            if np.sum(temp_mask) > 3:
                temp_variances.append(np.var(latent_values[temp_mask]))
        
        if len(temp_variances) < 3:
            return 0.0
        
        temp_variances = np.array(temp_variances)
        
        # Sharpness is ratio of max variance to mean variance
        max_variance = np.max(temp_variances)
        mean_variance = np.mean(temp_variances)
        
        return max_variance / (mean_variance + 1e-10)
    
    def _compute_monotonicity_score(self,
                                  latent_values: np.ndarray,
                                  temperatures: np.ndarray) -> float:
        """Compute monotonicity score (order parameters should be monotonic)."""
        
        # Bin by temperature
        unique_temps = np.unique(temperatures)
        if len(unique_temps) < 3:
            return 0.0
        
        temp_means = []
        for temp in sorted(unique_temps):
            temp_mask = temperatures == temp
            if np.sum(temp_mask) > 0:
                temp_means.append(np.mean(latent_values[temp_mask]))
        
        if len(temp_means) < 3:
            return 0.0
        
        # Compute Spearman correlation (rank-based monotonicity)
        try:
            spearman_r, _ = spearmanr(sorted(unique_temps), temp_means)
            return abs(spearman_r) if not np.isnan(spearman_r) else 0.0
        except:
            return 0.0
    
    def _compute_variance_ratio(self,
                              latent_values: np.ndarray,
                              temperatures: np.ndarray) -> float:
        """Compute ratio of high-temperature to low-temperature variance."""
        
        # Split temperatures into high and low
        temp_median = np.median(temperatures)
        
        low_temp_mask = temperatures <= temp_median
        high_temp_mask = temperatures > temp_median
        
        if np.sum(low_temp_mask) < 5 or np.sum(high_temp_mask) < 5:
            return 1.0
        
        low_temp_var = np.var(latent_values[low_temp_mask])
        high_temp_var = np.var(latent_values[high_temp_mask])
        
        # Return ratio (expect high temp variance > low temp variance for order parameter)
        return high_temp_var / (low_temp_var + 1e-10)
    
    def _compute_signal_to_noise(self,
                               latent_values: np.ndarray,
                               temperatures: np.ndarray) -> float:
        """Compute signal-to-noise ratio."""
        
        # Signal: variance across temperature means
        unique_temps = np.unique(temperatures)
        temp_means = []
        
        for temp in unique_temps:
            temp_mask = temperatures == temp
            if np.sum(temp_mask) > 0:
                temp_means.append(np.mean(latent_values[temp_mask]))
        
        if len(temp_means) < 2:
            return 0.0
        
        signal_variance = np.var(temp_means)
        
        # Noise: average within-temperature variance
        temp_variances = []
        for temp in unique_temps:
            temp_mask = temperatures == temp
            if np.sum(temp_mask) > 3:
                temp_variances.append(np.var(latent_values[temp_mask]))
        
        if len(temp_variances) == 0:
            return 0.0
        
        noise_variance = np.mean(temp_variances)
        
        return signal_variance / (noise_variance + 1e-10)
    
    def _compute_dynamic_range(self,
                             latent_values: np.ndarray,
                             temperatures: np.ndarray) -> float:
        """Compute dynamic range of latent values."""
        
        # Compute range relative to standard deviation
        value_range = np.max(latent_values) - np.min(latent_values)
        value_std = np.std(latent_values)
        
        return value_range / (value_std + 1e-10)
    
    def _select_best_dimension(self, dimension_scores: Dict[int, Dict[str, float]]) -> int:
        """Select best dimension based on comprehensive scoring."""
        
        # Weights for different criteria
        weights = {
            'temperature_sensitivity': 0.25,
            'transition_sharpness': 0.20,
            'monotonicity_score': 0.20,
            'variance_ratio': 0.15,
            'signal_to_noise': 0.15,
            'dynamic_range': 0.05
        }
        
        # Compute weighted scores
        total_scores = {}
        
        for dim, scores in dimension_scores.items():
            total_score = 0.0
            
            for metric, weight in weights.items():
                if metric in scores:
                    # Normalize score (some metrics need transformation)
                    if metric == 'variance_ratio':
                        # Convert to score (prefer ratio > 1, but not too large)
                        normalized_score = min(1.0, max(0.0, (scores[metric] - 1.0) / 2.0))
                    else:
                        # Most metrics are already in [0, 1] or can be normalized
                        normalized_score = min(1.0, max(0.0, scores[metric]))
                    
                    total_score += weight * normalized_score
            
            total_scores[dim] = total_score
        
        # Select dimension with highest score
        best_dim = max(total_scores.keys(), key=lambda d: total_scores[d])
        
        return best_dim
    
    def _compute_quality_metrics(self,
                               order_parameter_values: np.ndarray,
                               temperatures: np.ndarray,
                               dimension_scores: Dict[str, float]) -> Dict[str, float]:
        """Compute overall quality metrics for selected order parameter."""
        
        # Selection confidence based on how much better this dimension is
        all_scores = [sum(scores.values()) for scores in self.dimension_scores.values()] if hasattr(self, 'dimension_scores') else [1.0]
        current_score = sum(dimension_scores.values())
        
        if len(all_scores) > 1:
            sorted_scores = sorted(all_scores, reverse=True)
            if len(sorted_scores) > 1:
                selection_confidence = (sorted_scores[0] - sorted_scores[1]) / (sorted_scores[0] + 1e-10)
            else:
                selection_confidence = 1.0
        else:
            selection_confidence = 1.0
        
        # Overall quality score
        quality_score = current_score / len(dimension_scores)
        
        return {
            'selection_confidence': min(1.0, selection_confidence),
            'overall_quality': min(1.0, quality_score)
        }


class BlindCriticalTemperatureDetector:
    """Detects critical temperature without theoretical knowledge."""
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize blind critical temperature detector."""
        self.logger = get_logger(__name__)
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def detect_critical_temperature(self,
                                  temperatures: np.ndarray,
                                  order_parameter: np.ndarray) -> BlindCriticalTemperatureDetection:
        """
        Detect critical temperature from order parameter behavior.
        
        Args:
            temperatures: Temperature values
            order_parameter: Order parameter values
            
        Returns:
            BlindCriticalTemperatureDetection with detected Tc
        """
        self.logger.info("Starting blind critical temperature detection")
        
        # Try multiple detection methods
        method_results = {}
        
        # Method 1: Susceptibility peak
        tc_susceptibility = self._detect_tc_susceptibility_peak(temperatures, order_parameter)
        method_results['susceptibility_peak'] = tc_susceptibility
        
        # Method 2: Derivative peak
        tc_derivative = self._detect_tc_derivative_peak(temperatures, order_parameter)
        method_results['derivative_peak'] = tc_derivative
        
        # Method 3: Variance peak
        tc_variance = self._detect_tc_variance_peak(temperatures, order_parameter)
        method_results['variance_peak'] = tc_variance
        
        # Method 4: Inflection point
        tc_inflection = self._detect_tc_inflection_point(temperatures, order_parameter)
        method_results['inflection_point'] = tc_inflection
        
        # Ensemble method: weighted average
        tc_ensemble, ensemble_weights = self._ensemble_tc_detection(method_results, temperatures, order_parameter)
        
        # Assess detection quality
        detection_confidence = self._assess_detection_confidence(method_results, tc_ensemble)
        
        # Get supporting evidence
        susceptibility_peak = method_results.get('susceptibility_peak')
        derivative_peak = method_results.get('derivative_peak')
        variance_peak = method_results.get('variance_peak')
        
        # Compute quality metrics
        peak_prominence, peak_width, signal_to_noise = self._compute_detection_quality_metrics(
            temperatures, order_parameter, tc_ensemble
        )
        
        self.logger.info(f"Detected Tc = {tc_ensemble:.4f} (confidence: {detection_confidence:.3f})")
        
        return BlindCriticalTemperatureDetection(
            critical_temperature=tc_ensemble,
            detection_confidence=detection_confidence,
            detection_method='ensemble',
            susceptibility_peak=susceptibility_peak,
            derivative_peak=derivative_peak,
            variance_peak=variance_peak,
            peak_prominence=peak_prominence,
            peak_width=peak_width,
            signal_to_noise=signal_to_noise,
            method_results=method_results,
            ensemble_weight=ensemble_weights
        )
    
    def _detect_tc_susceptibility_peak(self,
                                     temperatures: np.ndarray,
                                     order_parameter: np.ndarray) -> Optional[float]:
        """Detect Tc from susceptibility (variance) peak."""
        
        # Bin by temperature
        unique_temps = np.unique(temperatures)
        if len(unique_temps) < 5:
            return None
        
        susceptibilities = []
        valid_temps = []
        
        for temp in unique_temps:
            temp_mask = temperatures == temp
            if np.sum(temp_mask) >= 5:
                susceptibility = np.var(order_parameter[temp_mask])
                susceptibilities.append(susceptibility)
                valid_temps.append(temp)
        
        if len(susceptibilities) < 3:
            return None
        
        susceptibilities = np.array(susceptibilities)
        valid_temps = np.array(valid_temps)
        
        # Find peak
        peak_idx = np.argmax(susceptibilities)
        
        return valid_temps[peak_idx]
    
    def _detect_tc_derivative_peak(self,
                                 temperatures: np.ndarray,
                                 order_parameter: np.ndarray) -> Optional[float]:
        """Detect Tc from derivative peak."""
        
        # Bin by temperature and compute means
        unique_temps = np.unique(temperatures)
        if len(unique_temps) < 5:
            return None
        
        temp_means = []
        valid_temps = []
        
        for temp in sorted(unique_temps):
            temp_mask = temperatures == temp
            if np.sum(temp_mask) > 0:
                temp_means.append(np.mean(order_parameter[temp_mask]))
                valid_temps.append(temp)
        
        if len(temp_means) < 5:
            return None
        
        temp_means = np.array(temp_means)
        valid_temps = np.array(valid_temps)
        
        # Smooth if enough points
        if len(temp_means) > 7:
            smoothed_means = savgol_filter(temp_means, 5, 2)
        else:
            smoothed_means = temp_means
        
        # Compute derivative
        derivative = np.gradient(smoothed_means, valid_temps)
        
        # Find peak in absolute derivative
        abs_derivative = np.abs(derivative)
        peak_idx = np.argmax(abs_derivative)
        
        return valid_temps[peak_idx]
    
    def _detect_tc_variance_peak(self,
                               temperatures: np.ndarray,
                               order_parameter: np.ndarray) -> Optional[float]:
        """Detect Tc from variance peak (alternative to susceptibility)."""
        
        # Use sliding window to compute local variance
        unique_temps = np.unique(temperatures)
        if len(unique_temps) < 5:
            return None
        
        temp_variances = []
        valid_temps = []
        
        for temp in unique_temps:
            temp_mask = temperatures == temp
            if np.sum(temp_mask) >= 3:
                local_variance = np.var(order_parameter[temp_mask])
                temp_variances.append(local_variance)
                valid_temps.append(temp)
        
        if len(temp_variances) < 3:
            return None
        
        temp_variances = np.array(temp_variances)
        valid_temps = np.array(valid_temps)
        
        # Find peak
        peak_idx = np.argmax(temp_variances)
        
        return valid_temps[peak_idx]
    
    def _detect_tc_inflection_point(self,
                                  temperatures: np.ndarray,
                                  order_parameter: np.ndarray) -> Optional[float]:
        """Detect Tc from inflection point in order parameter curve."""
        
        # Bin by temperature
        unique_temps = np.unique(temperatures)
        if len(unique_temps) < 7:
            return None
        
        temp_means = []
        valid_temps = []
        
        for temp in sorted(unique_temps):
            temp_mask = temperatures == temp
            if np.sum(temp_mask) > 0:
                temp_means.append(np.mean(order_parameter[temp_mask]))
                valid_temps.append(temp)
        
        if len(temp_means) < 7:
            return None
        
        temp_means = np.array(temp_means)
        valid_temps = np.array(valid_temps)
        
        # Smooth
        smoothed_means = savgol_filter(temp_means, 5, 2)
        
        # Compute second derivative
        second_derivative = np.gradient(np.gradient(smoothed_means, valid_temps), valid_temps)
        
        # Find inflection point (zero crossing of second derivative)
        sign_changes = np.diff(np.sign(second_derivative))
        inflection_indices = np.where(sign_changes != 0)[0]
        
        if len(inflection_indices) == 0:
            return None
        
        # Choose inflection point with largest absolute second derivative change
        best_idx = inflection_indices[np.argmax(np.abs(sign_changes[inflection_indices]))]
        
        return valid_temps[best_idx]
    
    def _ensemble_tc_detection(self,
                             method_results: Dict[str, Optional[float]],
                             temperatures: np.ndarray,
                             order_parameter: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Combine multiple Tc detection methods."""
        
        # Filter valid results
        valid_results = {method: tc for method, tc in method_results.items() if tc is not None}
        
        if not valid_results:
            # Fallback to median temperature
            return np.median(temperatures), {'fallback': 1.0}
        
        # Compute weights based on method reliability
        weights = {}
        
        for method, tc in valid_results.items():
            # Weight based on how reasonable the Tc value is
            temp_range = np.max(temperatures) - np.min(temperatures)
            temp_center = np.min(temperatures) + temp_range / 2
            
            # Prefer Tc values near the center of temperature range
            distance_from_center = abs(tc - temp_center) / (temp_range / 2)
            centrality_weight = max(0.1, 1.0 - distance_from_center)
            
            # Method-specific reliability weights
            method_weights = {
                'susceptibility_peak': 0.3,
                'derivative_peak': 0.25,
                'variance_peak': 0.25,
                'inflection_point': 0.2
            }
            
            method_reliability = method_weights.get(method, 0.1)
            
            weights[method] = centrality_weight * method_reliability
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {method: w / total_weight for method, w in weights.items()}
        else:
            weights = {method: 1.0 / len(weights) for method in weights}
        
        # Compute weighted average
        tc_ensemble = sum(tc * weights[method] for method, tc in valid_results.items())
        
        return tc_ensemble, weights
    
    def _assess_detection_confidence(self,
                                   method_results: Dict[str, Optional[float]],
                                   tc_ensemble: float) -> float:
        """Assess confidence in Tc detection."""
        
        valid_results = [tc for tc in method_results.values() if tc is not None]
        
        if len(valid_results) < 2:
            return 0.5
        
        # Compute agreement between methods
        deviations = [abs(tc - tc_ensemble) for tc in valid_results]
        mean_deviation = np.mean(deviations)
        
        # Convert to confidence (lower deviation = higher confidence)
        max_reasonable_deviation = 0.5  # Adjust based on typical temperature ranges
        confidence = max(0.0, 1.0 - mean_deviation / max_reasonable_deviation)
        
        # Bonus for having multiple methods agree
        method_bonus = min(0.3, (len(valid_results) - 1) * 0.1)
        
        return min(1.0, confidence + method_bonus)
    
    def _compute_detection_quality_metrics(self,
                                         temperatures: np.ndarray,
                                         order_parameter: np.ndarray,
                                         tc_detected: float) -> Tuple[float, float, float]:
        """Compute quality metrics for Tc detection."""
        
        # Peak prominence: how much the susceptibility peak stands out
        unique_temps = np.unique(temperatures)
        susceptibilities = []
        
        for temp in unique_temps:
            temp_mask = temperatures == temp
            if np.sum(temp_mask) >= 3:
                susceptibilities.append(np.var(order_parameter[temp_mask]))
        
        if len(susceptibilities) < 3:
            return 0.0, 0.0, 0.0
        
        susceptibilities = np.array(susceptibilities)
        
        # Peak prominence
        max_susceptibility = np.max(susceptibilities)
        median_susceptibility = np.median(susceptibilities)
        peak_prominence = (max_susceptibility - median_susceptibility) / (median_susceptibility + 1e-10)
        
        # Peak width (estimate)
        peak_width = 0.1  # Placeholder - would need more sophisticated analysis
        
        # Signal-to-noise ratio
        signal = np.std(susceptibilities)
        noise = np.mean(susceptibilities) * 0.1  # Estimate noise as 10% of signal
        signal_to_noise = signal / (noise + 1e-10)
        
        return peak_prominence, peak_width, signal_to_noise


class BlindPowerLawFitter:
    """Fits power laws to critical behavior without theoretical constraints."""
    
    def __init__(self, 
                 bootstrap_samples: int = 1000,
                 random_seed: Optional[int] = None):
        """Initialize blind power law fitter."""
        self.logger = get_logger(__name__)
        self.bootstrap_samples = bootstrap_samples
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def fit_power_law_blind(self,
                          temperatures: np.ndarray,
                          order_parameter: np.ndarray,
                          critical_temperature: float,
                          exponent_type: str = 'beta') -> BlindPowerLawFit:
        """
        Fit power law blindly without theoretical constraints.
        
        Args:
            temperatures: Temperature values
            order_parameter: Order parameter values
            critical_temperature: Critical temperature
            exponent_type: Type of exponent ('beta', 'nu', 'gamma')
            
        Returns:
            BlindPowerLawFit with fitting results
        """
        self.logger.info(f"Fitting {exponent_type} exponent blindly")
        
        # Prepare data based on exponent type
        if exponent_type == 'beta':
            # β: m ∝ (Tc - T)^β for T < Tc
            mask = temperatures < critical_temperature
            x_data = critical_temperature - temperatures[mask]
            y_data = np.abs(order_parameter[mask])
        elif exponent_type == 'nu':
            # ν: ξ ∝ |T - Tc|^(-ν)
            mask = temperatures != critical_temperature
            x_data = np.abs(temperatures[mask] - critical_temperature)
            y_data = np.abs(order_parameter[mask])  # Assuming order_parameter is correlation length
        elif exponent_type == 'gamma':
            # γ: χ ∝ |T - Tc|^(-γ)
            mask = temperatures != critical_temperature
            x_data = np.abs(temperatures[mask] - critical_temperature)
            y_data = np.abs(order_parameter[mask])  # Assuming order_parameter is susceptibility
        else:
            raise ValueError(f"Unknown exponent type: {exponent_type}")
        
        # Filter valid data
        valid_mask = (x_data > 0) & (y_data > 0) & np.isfinite(x_data) & np.isfinite(y_data)
        x_data = x_data[valid_mask]
        y_data = y_data[valid_mask]
        
        if len(x_data) < 5:
            raise ValueError(f"Insufficient data for {exponent_type} fitting: {len(x_data)} points")
        
        # Try multiple fitting methods
        fit_results = []
        
        # Method 1: Log-linear regression
        try:
            result_log = self._fit_log_linear(x_data, y_data)
            fit_results.append(('log_linear', result_log))
        except Exception as e:
            self.logger.warning(f"Log-linear fitting failed: {e}")
        
        # Method 2: Nonlinear least squares
        try:
            result_nls = self._fit_nonlinear_least_squares(x_data, y_data)
            fit_results.append(('nonlinear_ls', result_nls))
        except Exception as e:
            self.logger.warning(f"Nonlinear least squares fitting failed: {e}")
        
        # Method 3: Robust fitting
        try:
            result_robust = self._fit_robust(x_data, y_data)
            fit_results.append(('robust', result_robust))
        except Exception as e:
            self.logger.warning(f"Robust fitting failed: {e}")
        
        if not fit_results:
            raise ValueError(f"All fitting methods failed for {exponent_type}")
        
        # Select best fit based on quality metrics
        best_method, best_result = self._select_best_fit(fit_results, x_data, y_data)
        
        # Compute bootstrap confidence intervals
        ci_lower, ci_upper = self._compute_bootstrap_ci(x_data, y_data, best_method)
        
        # Compute residual analysis
        residual_analysis = self._analyze_residuals(x_data, y_data, best_result)
        
        # Compute overall fit quality score
        fit_quality_score = self._compute_fit_quality_score(best_result, residual_analysis)
        
        self.logger.info(f"{exponent_type} exponent: {best_result['exponent']:.4f} ± {best_result['exponent_error']:.4f}")
        
        return BlindPowerLawFit(
            exponent=best_result['exponent'],
            exponent_error=best_result['exponent_error'],
            amplitude=best_result['amplitude'],
            amplitude_error=best_result.get('amplitude_error', 0.0),
            r_squared=best_result['r_squared'],
            p_value=best_result.get('p_value', 1.0),
            chi_squared=best_result.get('chi_squared', 0.0),
            degrees_of_freedom=len(x_data) - 2,
            exponent_ci_lower=ci_lower,
            exponent_ci_upper=ci_upper,
            temperature_range=(np.min(x_data), np.max(x_data)),
            n_data_points=len(x_data),
            fitting_method=best_method,
            optimization_success=best_result.get('success', True),
            fit_quality_score=fit_quality_score,
            residual_analysis=residual_analysis
        )
    
    def _fit_log_linear(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, Any]:
        """Fit using log-linear regression."""
        
        log_x = safe_log(x_data)
        log_y = safe_log(y_data)
        
        # Linear regression in log space
        slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
        
        return {
            'exponent': slope,
            'exponent_error': std_err,
            'amplitude': np.exp(intercept),
            'r_squared': r_value**2,
            'p_value': p_value,
            'success': True
        }
    
    def _fit_nonlinear_least_squares(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, Any]:
        """Fit using nonlinear least squares."""
        
        def power_law(x, amplitude, exponent):
            return amplitude * (x ** exponent)
        
        # Initial guess from log-linear fit
        log_result = self._fit_log_linear(x_data, y_data)
        initial_guess = [log_result['amplitude'], log_result['exponent']]
        
        # Fit with bounds
        bounds = ([1e-10, -10], [1e10, 10])
        
        popt, pcov = curve_fit(power_law, x_data, y_data, p0=initial_guess, bounds=bounds)
        
        # Compute R-squared
        y_pred = power_law(x_data, *popt)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Parameter errors from covariance matrix
        param_errors = np.sqrt(np.diag(pcov))
        
        return {
            'exponent': popt[1],
            'exponent_error': param_errors[1],
            'amplitude': popt[0],
            'amplitude_error': param_errors[0],
            'r_squared': r_squared,
            'chi_squared': ss_res,
            'success': True
        }
    
    def _fit_robust(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, Any]:
        """Fit using robust method (less sensitive to outliers)."""
        
        # Use log-linear as base, but with outlier removal
        log_x = safe_log(x_data)
        log_y = safe_log(y_data)
        
        # Remove outliers using IQR method
        q1, q3 = np.percentile(log_y, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (log_y >= lower_bound) & (log_y <= upper_bound)
        
        if np.sum(outlier_mask) < 5:
            # Not enough data after outlier removal
            return self._fit_log_linear(x_data, y_data)
        
        # Fit on cleaned data
        slope, intercept, r_value, p_value, std_err = linregress(log_x[outlier_mask], log_y[outlier_mask])
        
        return {
            'exponent': slope,
            'exponent_error': std_err,
            'amplitude': np.exp(intercept),
            'r_squared': r_value**2,
            'p_value': p_value,
            'success': True
        }
    
    def _select_best_fit(self, 
                        fit_results: List[Tuple[str, Dict[str, Any]]],
                        x_data: np.ndarray,
                        y_data: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        """Select best fit based on quality metrics."""
        
        best_score = -1
        best_method = None
        best_result = None
        
        for method, result in fit_results:
            # Quality score based on R-squared and parameter reasonableness
            r_squared = result.get('r_squared', 0)
            exponent = result.get('exponent', 0)
            
            # Penalize unreasonable exponents
            if abs(exponent) > 10:
                exponent_penalty = 0.5
            elif abs(exponent) > 5:
                exponent_penalty = 0.8
            else:
                exponent_penalty = 1.0
            
            # Combined score
            score = r_squared * exponent_penalty
            
            if score > best_score:
                best_score = score
                best_method = method
                best_result = result
        
        return best_method, best_result
    
    def _compute_bootstrap_ci(self, 
                            x_data: np.ndarray,
                            y_data: np.ndarray,
                            method: str) -> Tuple[float, float]:
        """Compute bootstrap confidence intervals."""
        
        bootstrap_exponents = []
        
        for _ in range(min(self.bootstrap_samples, 100)):  # Limit for performance
            # Bootstrap sample
            indices = np.random.choice(len(x_data), len(x_data), replace=True)
            x_boot = x_data[indices]
            y_boot = y_data[indices]
            
            try:
                if method == 'log_linear':
                    result = self._fit_log_linear(x_boot, y_boot)
                elif method == 'nonlinear_ls':
                    result = self._fit_nonlinear_least_squares(x_boot, y_boot)
                elif method == 'robust':
                    result = self._fit_robust(x_boot, y_boot)
                else:
                    continue
                
                bootstrap_exponents.append(result['exponent'])
            except:
                continue
        
        if len(bootstrap_exponents) < 10:
            # Fallback to parameter error
            return -1.0, 1.0
        
        # Compute 95% confidence interval
        ci_lower = np.percentile(bootstrap_exponents, 2.5)
        ci_upper = np.percentile(bootstrap_exponents, 97.5)
        
        return ci_lower, ci_upper
    
    def _analyze_residuals(self, 
                         x_data: np.ndarray,
                         y_data: np.ndarray,
                         fit_result: Dict[str, Any]) -> Dict[str, float]:
        """Analyze fitting residuals."""
        
        # Compute predicted values
        amplitude = fit_result['amplitude']
        exponent = fit_result['exponent']
        y_pred = amplitude * (x_data ** exponent)
        
        # Residuals
        residuals = y_data - y_pred
        relative_residuals = residuals / (y_data + 1e-10)
        
        return {
            'mean_absolute_error': np.mean(np.abs(residuals)),
            'root_mean_square_error': np.sqrt(np.mean(residuals**2)),
            'mean_relative_error': np.mean(np.abs(relative_residuals)),
            'max_relative_error': np.max(np.abs(relative_residuals)),
            'residual_autocorrelation': np.corrcoef(residuals[:-1], residuals[1:])[0, 1] if len(residuals) > 1 else 0.0
        }
    
    def _compute_fit_quality_score(self, 
                                 fit_result: Dict[str, Any],
                                 residual_analysis: Dict[str, float]) -> float:
        """Compute overall fit quality score."""
        
        # Components of quality score
        r_squared = fit_result.get('r_squared', 0)
        
        # Penalize high relative errors
        mean_rel_error = residual_analysis.get('mean_relative_error', 1.0)
        error_penalty = max(0.0, 1.0 - mean_rel_error)
        
        # Penalize unreasonable exponents
        exponent = abs(fit_result.get('exponent', 0))
        if exponent > 5:
            exponent_penalty = 0.5
        elif exponent > 2:
            exponent_penalty = 0.8
        else:
            exponent_penalty = 1.0
        
        # Combined score
        quality_score = 0.5 * r_squared + 0.3 * error_penalty + 0.2 * exponent_penalty
        
        return min(1.0, max(0.0, quality_score))


class BlindCriticalExponentExtractor:
    """Main class for blind critical exponent extraction."""
    
    def __init__(self, 
                 bootstrap_samples: int = 1000,
                 random_seed: Optional[int] = None):
        """Initialize blind critical exponent extractor."""
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.order_param_identifier = BlindOrderParameterIdentifier(random_seed)
        self.tc_detector = BlindCriticalTemperatureDetector(random_seed)
        self.power_law_fitter = BlindPowerLawFitter(bootstrap_samples, random_seed)
    
    def extract_critical_exponents_blind(self,
                                       latent_representations: np.ndarray,
                                       temperatures: np.ndarray,
                                       magnetizations: Optional[np.ndarray] = None,
                                       system_identifier: str = 'unknown') -> BlindCriticalExponentResults:
        """
        Extract critical exponents blindly from latent representations.
        
        Args:
            latent_representations: Latent space coordinates (N, latent_dim)
            temperatures: Temperature values (N,)
            magnetizations: Magnetization values for validation (N,)
            system_identifier: Identifier for the system
            
        Returns:
            BlindCriticalExponentResults with complete analysis
        """
        self.logger.info("Starting blind critical exponent extraction")
        
        # Step 1: Identify order parameter
        order_param_analysis = self.order_param_identifier.identify_order_parameter(
            latent_representations, temperatures, magnetizations
        )
        
        # Step 2: Detect critical temperature
        tc_detection = self.tc_detector.detect_critical_temperature(
            temperatures, order_param_analysis.order_parameter_values
        )
        
        # Step 3: Extract β exponent
        beta_exponent = None
        try:
            beta_exponent = self.power_law_fitter.fit_power_law_blind(
                temperatures,
                order_param_analysis.order_parameter_values,
                tc_detection.critical_temperature,
                exponent_type='beta'
            )
        except Exception as e:
            self.logger.error(f"β exponent extraction failed: {e}")
        
        # Step 4: Extract ν exponent (if correlation length data available)
        nu_exponent = None
        # Note: This would require correlation length computation from latent space
        # For now, we skip this as it requires more sophisticated analysis
        
        # Step 5: Compute overall quality
        extraction_quality_score = self._compute_extraction_quality(
            order_param_analysis, tc_detection, beta_exponent, nu_exponent
        )
        
        # Step 6: Compute reliability metrics
        reliability_metrics = self._compute_reliability_metrics(
            order_param_analysis, tc_detection, beta_exponent
        )
        
        # Step 7: Compare with raw magnetization (if available)
        raw_magnetization_comparison = None
        if magnetizations is not None:
            raw_magnetization_comparison = self._compare_with_raw_magnetization(
                temperatures, magnetizations, tc_detection.critical_temperature
            )
        
        results = BlindCriticalExponentResults(
            system_identifier=system_identifier,
            order_parameter_analysis=order_param_analysis,
            tc_detection=tc_detection,
            beta_exponent=beta_exponent,
            nu_exponent=nu_exponent,
            gamma_exponent=None,  # Not implemented
            extraction_quality_score=extraction_quality_score,
            reliability_metrics=reliability_metrics,
            raw_magnetization_comparison=raw_magnetization_comparison
        )
        
        self.logger.info("Blind critical exponent extraction completed")
        self.logger.info(f"Overall quality score: {extraction_quality_score:.3f}")
        
        return results
    
    def _compute_extraction_quality(self,
                                  order_param_analysis: BlindOrderParameterAnalysis,
                                  tc_detection: BlindCriticalTemperatureDetection,
                                  beta_exponent: Optional[BlindPowerLawFit],
                                  nu_exponent: Optional[BlindPowerLawFit]) -> float:
        """Compute overall extraction quality score."""
        
        quality_components = []
        
        # Order parameter quality
        quality_components.append(order_param_analysis.selection_confidence)
        
        # Critical temperature detection quality
        quality_components.append(tc_detection.detection_confidence)
        
        # β exponent quality
        if beta_exponent:
            quality_components.append(beta_exponent.fit_quality_score)
        
        # ν exponent quality
        if nu_exponent:
            quality_components.append(nu_exponent.fit_quality_score)
        
        return np.mean(quality_components) if quality_components else 0.0
    
    def _compute_reliability_metrics(self,
                                   order_param_analysis: BlindOrderParameterAnalysis,
                                   tc_detection: BlindCriticalTemperatureDetection,
                                   beta_exponent: Optional[BlindPowerLawFit]) -> Dict[str, float]:
        """Compute reliability metrics."""
        
        metrics = {}
        
        # Order parameter reliability
        metrics['order_parameter_reliability'] = order_param_analysis.selection_confidence
        
        # Critical temperature reliability
        metrics['tc_detection_reliability'] = tc_detection.detection_confidence
        
        # Exponent fitting reliability
        if beta_exponent:
            metrics['beta_fitting_reliability'] = beta_exponent.fit_quality_score
        else:
            metrics['beta_fitting_reliability'] = 0.0
        
        # Overall reliability
        metrics['overall_reliability'] = np.mean(list(metrics.values()))
        
        return metrics
    
    def _compare_with_raw_magnetization(self,
                                      temperatures: np.ndarray,
                                      magnetizations: np.ndarray,
                                      critical_temperature: float) -> Dict[str, Any]:
        """Compare VAE-based results with raw magnetization approach."""
        
        try:
            # Extract β exponent from raw magnetization
            raw_beta = self.power_law_fitter.fit_power_law_blind(
                temperatures,
                np.abs(magnetizations),
                critical_temperature,
                exponent_type='beta'
            )
            
            return {
                'raw_magnetization_beta': {
                    'exponent': raw_beta.exponent,
                    'error': raw_beta.exponent_error,
                    'r_squared': raw_beta.r_squared,
                    'quality_score': raw_beta.fit_quality_score
                }
            }
        except Exception as e:
            self.logger.warning(f"Raw magnetization comparison failed: {e}")
            return {'raw_magnetization_beta': None}


def create_blind_critical_exponent_extractor(bootstrap_samples: int = 1000,
                                           random_seed: Optional[int] = None) -> BlindCriticalExponentExtractor:
    """
    Factory function to create BlindCriticalExponentExtractor.
    
    Args:
        bootstrap_samples: Number of bootstrap samples for error estimation
        random_seed: Random seed for reproducibility
        
    Returns:
        Configured BlindCriticalExponentExtractor instance
    """
    return BlindCriticalExponentExtractor(bootstrap_samples, random_seed)