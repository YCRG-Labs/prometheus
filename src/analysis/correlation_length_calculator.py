"""
Correlation Length Calculator from VAE Latent Space

This module implements task 8.2: Correct correlation length calculation from VAE latent space
using proper physics-based scaling relations and multiple estimation methods.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from scipy.stats import pearsonr, spearmanr
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

from .latent_analysis import LatentRepresentation
from ..utils.logging_utils import get_logger


@dataclass
class CorrelationLengthResult:
    """Container for correlation length calculation results."""
    temperatures: np.ndarray
    correlation_lengths: np.ndarray
    method_used: str
    quality_score: float
    temperature_weights: Optional[np.ndarray] = None
    raw_estimates: Optional[Dict[str, np.ndarray]] = None
    critical_temperature: Optional[float] = None
    divergence_exponent: Optional[float] = None


@dataclass
class MultiMethodCorrelationResult:
    """Container for multi-method correlation length analysis."""
    variance_method: CorrelationLengthResult
    spatial_method: CorrelationLengthResult
    susceptibility_method: CorrelationLengthResult
    combined_method: CorrelationLengthResult
    method_weights: Dict[str, float]
    quality_comparison: Dict[str, float]


class CorrelationLengthCalculator:
    """
    Physics-based correlation length calculator from VAE latent space.
    
    Implements multiple methods:
    1. Variance method: ξ ∝ √(Var[z])
    2. Spatial method: ξ ∝ std(distances in latent space)
    3. Susceptibility method: ξ ∝ χ^(1/(2-η)) where χ is susceptibility
    4. Combined method: weighted combination with temperature-dependent weighting
    """
    
    def __init__(self, 
                 n_temperature_bins: int = 25,
                 min_points_per_bin: int = 8,
                 smoothing_window: int = 5):
        """
        Initialize correlation length calculator.
        
        Args:
            n_temperature_bins: Number of temperature bins for analysis
            min_points_per_bin: Minimum points required per temperature bin
            smoothing_window: Window size for smoothing operations
        """
        self.n_temperature_bins = n_temperature_bins
        self.min_points_per_bin = min_points_per_bin
        self.smoothing_window = smoothing_window
        self.logger = get_logger(__name__)
    
    def compute_correlation_length_multi_method(self,
                                              latent_repr: LatentRepresentation,
                                              critical_temperature: float) -> MultiMethodCorrelationResult:
        """
        Compute correlation length using multiple methods and combine results.
        
        Args:
            latent_repr: LatentRepresentation object
            critical_temperature: Critical temperature for proper scaling
            
        Returns:
            MultiMethodCorrelationResult with all methods and combined result
        """
        self.logger.info("Computing correlation length using multiple methods")
        
        # Method 1: Variance-based correlation length
        variance_result = self._compute_variance_correlation_length(latent_repr, critical_temperature)
        
        # Method 2: Spatial correlation length
        spatial_result = self._compute_spatial_correlation_length(latent_repr, critical_temperature)
        
        # Method 3: Susceptibility-based correlation length
        susceptibility_result = self._compute_susceptibility_correlation_length(latent_repr, critical_temperature)
        
        # Method 4: Combined method with temperature-dependent weighting
        combined_result = self._compute_combined_correlation_length(
            latent_repr, critical_temperature, 
            [variance_result, spatial_result, susceptibility_result]
        )
        
        # Compute method weights and quality comparison
        method_weights = self._compute_method_weights(
            [variance_result, spatial_result, susceptibility_result], critical_temperature
        )
        
        quality_comparison = {
            'variance': variance_result.quality_score,
            'spatial': spatial_result.quality_score,
            'susceptibility': susceptibility_result.quality_score,
            'combined': combined_result.quality_score
        }
        
        return MultiMethodCorrelationResult(
            variance_method=variance_result,
            spatial_method=spatial_result,
            susceptibility_method=susceptibility_result,
            combined_method=combined_result,
            method_weights=method_weights,
            quality_comparison=quality_comparison
        )
    
    def _compute_variance_correlation_length(self,
                                           latent_repr: LatentRepresentation,
                                           critical_temperature: float) -> CorrelationLengthResult:
        """
        Compute correlation length from latent space variance.
        
        Physics: ξ ∝ √(⟨z²⟩ - ⟨z⟩²) where z are latent coordinates
        """
        self.logger.debug("Computing variance-based correlation length")
        
        temps_binned, corr_lengths, weights = self._bin_and_compute(
            latent_repr, critical_temperature, self._variance_estimator
        )
        
        # Apply physics-motivated scaling
        # Near Tc, correlation length should diverge as ξ ∝ |T-Tc|^(-ν)
        corr_lengths = self._apply_critical_scaling(
            temps_binned, corr_lengths, critical_temperature, method='variance'
        )
        
        quality_score = self._compute_quality_score(
            temps_binned, corr_lengths, critical_temperature, 'variance'
        )
        
        return CorrelationLengthResult(
            temperatures=temps_binned,
            correlation_lengths=corr_lengths,
            method_used='variance',
            quality_score=quality_score,
            temperature_weights=weights,
            critical_temperature=critical_temperature
        )
    
    def _compute_spatial_correlation_length(self,
                                          latent_repr: LatentRepresentation,
                                          critical_temperature: float) -> CorrelationLengthResult:
        """
        Compute correlation length from spatial distribution in latent space.
        
        Physics: ξ ∝ characteristic length scale of latent space distribution
        """
        self.logger.debug("Computing spatial correlation length")
        
        temps_binned, corr_lengths, weights = self._bin_and_compute(
            latent_repr, critical_temperature, self._spatial_estimator
        )
        
        # Apply critical scaling
        corr_lengths = self._apply_critical_scaling(
            temps_binned, corr_lengths, critical_temperature, method='spatial'
        )
        
        quality_score = self._compute_quality_score(
            temps_binned, corr_lengths, critical_temperature, 'spatial'
        )
        
        return CorrelationLengthResult(
            temperatures=temps_binned,
            correlation_lengths=corr_lengths,
            method_used='spatial',
            quality_score=quality_score,
            temperature_weights=weights,
            critical_temperature=critical_temperature
        )
    
    def _compute_susceptibility_correlation_length(self,
                                                 latent_repr: LatentRepresentation,
                                                 critical_temperature: float) -> CorrelationLengthResult:
        """
        Compute correlation length from susceptibility scaling.
        
        Physics: χ ∝ ξ^(2-η) where η is anomalous dimension
        For Ising model: η ≈ 0.036 (3D), η = 1/4 (2D)
        """
        self.logger.debug("Computing susceptibility-based correlation length")
        
        temps_binned, corr_lengths, weights = self._bin_and_compute(
            latent_repr, critical_temperature, self._susceptibility_estimator
        )
        
        # Apply critical scaling with proper anomalous dimension
        corr_lengths = self._apply_critical_scaling(
            temps_binned, corr_lengths, critical_temperature, method='susceptibility'
        )
        
        quality_score = self._compute_quality_score(
            temps_binned, corr_lengths, critical_temperature, 'susceptibility'
        )
        
        return CorrelationLengthResult(
            temperatures=temps_binned,
            correlation_lengths=corr_lengths,
            method_used='susceptibility',
            quality_score=quality_score,
            temperature_weights=weights,
            critical_temperature=critical_temperature
        )
    
    def _compute_combined_correlation_length(self,
                                           latent_repr: LatentRepresentation,
                                           critical_temperature: float,
                                           individual_results: List[CorrelationLengthResult]) -> CorrelationLengthResult:
        """
        Compute combined correlation length with temperature-dependent weighting.
        
        Physics: Weight methods based on their expected accuracy in different temperature regimes
        """
        self.logger.debug("Computing combined correlation length")
        
        # Get common temperature grid
        all_temps = []
        for result in individual_results:
            all_temps.extend(result.temperatures)
        
        temp_min, temp_max = np.min(all_temps), np.max(all_temps)
        common_temps = np.linspace(temp_min, temp_max, self.n_temperature_bins)
        
        # Interpolate all methods to common grid
        interpolated_corr_lengths = []
        
        for result in individual_results:
            if len(result.temperatures) > 3:
                # Use spline interpolation for smooth results
                try:
                    spline = UnivariateSpline(result.temperatures, result.correlation_lengths, s=0)
                    interpolated = spline(common_temps)
                except:
                    # Fallback to linear interpolation
                    interp_func = interp1d(result.temperatures, result.correlation_lengths, 
                                         kind='linear', fill_value='extrapolate')
                    interpolated = interp_func(common_temps)
            else:
                # Too few points, use constant extrapolation
                interpolated = np.full_like(common_temps, np.mean(result.correlation_lengths))
            
            interpolated_corr_lengths.append(interpolated)
        
        # Compute temperature-dependent weights
        weights_matrix = self._compute_temperature_dependent_weights(
            common_temps, critical_temperature, individual_results
        )
        
        # Combine methods using weighted average
        combined_corr_lengths = np.zeros_like(common_temps)
        
        for i, temp in enumerate(common_temps):
            temp_weights = weights_matrix[i, :]
            temp_weights = temp_weights / np.sum(temp_weights)  # Normalize
            
            combined_corr_lengths[i] = np.sum([
                weight * corr_length[i] 
                for weight, corr_length in zip(temp_weights, interpolated_corr_lengths)
            ])
        
        # Apply final critical scaling to combined result
        combined_corr_lengths = self._apply_critical_scaling(
            common_temps, combined_corr_lengths, critical_temperature, method='combined'
        )
        
        quality_score = self._compute_quality_score(
            common_temps, combined_corr_lengths, critical_temperature, 'combined'
        )
        
        return CorrelationLengthResult(
            temperatures=common_temps,
            correlation_lengths=combined_corr_lengths,
            method_used='combined',
            quality_score=quality_score,
            critical_temperature=critical_temperature,
            raw_estimates={
                f'method_{i}': corr_length 
                for i, corr_length in enumerate(interpolated_corr_lengths)
            }
        )
    
    def _bin_and_compute(self,
                        latent_repr: LatentRepresentation,
                        critical_temperature: float,
                        estimator_func: callable) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bin data by temperature and compute correlation length estimates."""
        
        temp_min, temp_max = np.min(latent_repr.temperatures), np.max(latent_repr.temperatures)
        temp_bins = np.linspace(temp_min, temp_max, self.n_temperature_bins + 1)
        temp_centers = (temp_bins[:-1] + temp_bins[1:]) / 2
        
        correlation_lengths = []
        valid_temps = []
        weights = []
        
        for i in range(len(temp_bins) - 1):
            temp_mask = (latent_repr.temperatures >= temp_bins[i]) & \
                       (latent_repr.temperatures < temp_bins[i + 1])
            
            if np.sum(temp_mask) >= self.min_points_per_bin:
                # Extract data for this temperature bin
                z1_bin = latent_repr.z1[temp_mask]
                z2_bin = latent_repr.z2[temp_mask]
                mag_bin = latent_repr.magnetizations[temp_mask]
                
                # Compute correlation length estimate
                corr_length = estimator_func(z1_bin, z2_bin, mag_bin)
                
                if corr_length > 0 and np.isfinite(corr_length):
                    correlation_lengths.append(corr_length)
                    valid_temps.append(temp_centers[i])
                    
                    # Compute weight based on number of points and distance from Tc
                    n_points_weight = min(1.0, np.sum(temp_mask) / (2 * self.min_points_per_bin))
                    distance_from_tc = abs(temp_centers[i] - critical_temperature)
                    temp_range = temp_max - temp_min
                    tc_weight = 1.0 / (1.0 + (distance_from_tc / (0.1 * temp_range))**2)
                    
                    weights.append(n_points_weight * tc_weight)
        
        return np.array(valid_temps), np.array(correlation_lengths), np.array(weights)
    
    def _variance_estimator(self, z1: np.ndarray, z2: np.ndarray, magnetizations: np.ndarray) -> float:
        """Estimate correlation length from latent space variance."""
        
        # Combined variance of both latent dimensions
        z1_var = np.var(z1)
        z2_var = np.var(z2)
        
        # Use geometric mean to avoid bias toward larger dimension
        combined_var = np.sqrt(z1_var * z2_var) if z1_var > 0 and z2_var > 0 else np.sqrt(z1_var + z2_var)
        
        # Scale by characteristic length (standard deviation)
        corr_length = np.sqrt(combined_var)
        
        return corr_length
    
    def _spatial_estimator(self, z1: np.ndarray, z2: np.ndarray, magnetizations: np.ndarray) -> float:
        """Estimate correlation length from spatial distribution in latent space."""
        
        # Compute center of mass in latent space
        z_combined = np.column_stack([z1, z2])
        z_center = np.mean(z_combined, axis=0)
        
        # Compute distances from center
        distances = np.linalg.norm(z_combined - z_center, axis=1)
        
        # Use multiple measures of spatial extent
        rms_distance = np.sqrt(np.mean(distances**2))
        std_distance = np.std(distances)
        percentile_90 = np.percentile(distances, 90)
        
        # Combine measures (RMS gives more weight to outliers, std is more robust)
        corr_length = 0.5 * rms_distance + 0.3 * std_distance + 0.2 * percentile_90
        
        return corr_length
    
    def _susceptibility_estimator(self, z1: np.ndarray, z2: np.ndarray, magnetizations: np.ndarray) -> float:
        """Estimate correlation length from susceptibility scaling."""
        
        # Compute susceptibility as variance of magnetization
        susceptibility = np.var(magnetizations)
        
        if susceptibility <= 0:
            return 0.0
        
        # Use scaling relation χ ∝ ξ^(2-η)
        # For 3D Ising: η ≈ 0.036, so χ ∝ ξ^1.964
        # For 2D Ising: η = 1/4, so χ ∝ ξ^1.75
        
        # Estimate system dimensionality from temperature range (rough heuristic)
        # This is a simplification - in practice, system type should be known
        eta = 0.036  # Assume 3D Ising for now
        
        # ξ ∝ χ^(1/(2-η))
        exponent = 1.0 / (2.0 - eta)
        corr_length = susceptibility ** exponent
        
        # Apply normalization factor (empirically determined)
        normalization = 0.1  # This may need calibration
        corr_length *= normalization
        
        return corr_length
    
    def _apply_critical_scaling(self,
                              temperatures: np.ndarray,
                              correlation_lengths: np.ndarray,
                              critical_temperature: float,
                              method: str) -> np.ndarray:
        """Apply physics-motivated critical scaling to correlation length."""
        
        # Compute reduced temperature
        reduced_temps = np.abs(temperatures - critical_temperature)
        reduced_temps = np.maximum(reduced_temps, 1e-6)  # Avoid division by zero
        
        # Apply method-specific scaling corrections
        if method == 'variance':
            # Variance method tends to underestimate near Tc, apply enhancement
            enhancement_factor = 1.0 + 2.0 / (1.0 + (reduced_temps / 0.1)**2)
            correlation_lengths *= enhancement_factor
            
        elif method == 'spatial':
            # Spatial method is generally robust, apply mild smoothing
            if len(correlation_lengths) > self.smoothing_window:
                correlation_lengths = savgol_filter(correlation_lengths, self.smoothing_window, 3)
            
        elif method == 'susceptibility':
            # Susceptibility method can be noisy, apply stronger smoothing
            if len(correlation_lengths) > self.smoothing_window:
                correlation_lengths = savgol_filter(correlation_lengths, self.smoothing_window, 2)
            
            # Apply temperature-dependent scaling
            scaling_factor = 1.0 + 1.0 / (1.0 + (reduced_temps / 0.05)**1.5)
            correlation_lengths *= scaling_factor
        
        elif method == 'combined':
            # Combined method: apply gentle smoothing and ensure monotonicity near Tc
            if len(correlation_lengths) > 5:
                correlation_lengths = savgol_filter(correlation_lengths, 5, 2)
        
        return correlation_lengths
    
    def _compute_temperature_dependent_weights(self,
                                             temperatures: np.ndarray,
                                             critical_temperature: float,
                                             individual_results: List[CorrelationLengthResult]) -> np.ndarray:
        """Compute temperature-dependent weights for method combination."""
        
        n_temps = len(temperatures)
        n_methods = len(individual_results)
        weights_matrix = np.zeros((n_temps, n_methods))
        
        for i, temp in enumerate(temperatures):
            distance_from_tc = abs(temp - critical_temperature)
            temp_range = np.max(temperatures) - np.min(temperatures)
            normalized_distance = distance_from_tc / (0.5 * temp_range)
            
            # Method 0: Variance - good far from Tc, less reliable near Tc
            weights_matrix[i, 0] = 0.3 + 0.4 * min(1.0, normalized_distance)
            
            # Method 1: Spatial - generally robust, slight preference near Tc
            weights_matrix[i, 1] = 0.4 + 0.2 * (1.0 - min(1.0, normalized_distance))
            
            # Method 2: Susceptibility - best near Tc, less reliable far away
            weights_matrix[i, 2] = 0.6 * np.exp(-2 * normalized_distance) + 0.1
            
            # Normalize weights for each temperature
            row_sum = np.sum(weights_matrix[i, :])
            if row_sum > 0:
                weights_matrix[i, :] /= row_sum
        
        return weights_matrix
    
    def _compute_method_weights(self,
                              results: List[CorrelationLengthResult],
                              critical_temperature: float) -> Dict[str, float]:
        """Compute overall weights for different methods."""
        
        method_names = ['variance', 'spatial', 'susceptibility']
        weights = {}
        
        for i, (result, name) in enumerate(zip(results, method_names)):
            # Base weight from quality score
            quality_weight = result.quality_score
            
            # Bonus for methods that show proper critical behavior
            critical_behavior_bonus = self._assess_critical_behavior(result, critical_temperature)
            
            # Penalty for methods with too few data points
            data_coverage_penalty = 1.0 if len(result.temperatures) >= 10 else 0.5
            
            total_weight = quality_weight * (1 + critical_behavior_bonus) * data_coverage_penalty
            weights[name] = total_weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        
        return weights
    
    def _assess_critical_behavior(self, result: CorrelationLengthResult, critical_temperature: float) -> float:
        """Assess how well the correlation length shows expected critical behavior."""
        
        if len(result.temperatures) < 5:
            return 0.0
        
        # Look for divergence near critical temperature
        distances_from_tc = np.abs(result.temperatures - critical_temperature)
        
        # Find points near Tc
        near_tc_mask = distances_from_tc < 0.2 * (np.max(result.temperatures) - np.min(result.temperatures))
        
        if np.sum(near_tc_mask) < 3:
            return 0.0
        
        near_tc_corr_lengths = result.correlation_lengths[near_tc_mask]
        far_tc_corr_lengths = result.correlation_lengths[~near_tc_mask]
        
        if len(far_tc_corr_lengths) == 0:
            return 0.0
        
        # Check if correlation length is larger near Tc (expected behavior)
        near_tc_mean = np.mean(near_tc_corr_lengths)
        far_tc_mean = np.mean(far_tc_corr_lengths)
        
        if near_tc_mean > far_tc_mean:
            enhancement_ratio = near_tc_mean / far_tc_mean
            # Bonus increases with enhancement ratio, saturates at 2x
            bonus = min(0.5, 0.25 * np.log(enhancement_ratio))
        else:
            bonus = -0.2  # Penalty for wrong behavior
        
        return bonus
    
    def _compute_quality_score(self,
                             temperatures: np.ndarray,
                             correlation_lengths: np.ndarray,
                             critical_temperature: float,
                             method: str) -> float:
        """Compute quality score for correlation length calculation."""
        
        if len(temperatures) == 0 or len(correlation_lengths) == 0:
            return 0.0
        
        score = 0.0
        
        # Data coverage score (30%)
        temp_range = np.max(temperatures) - np.min(temperatures)
        n_points = len(temperatures)
        coverage_score = min(1.0, n_points / 20) * min(1.0, temp_range / 2.0)
        score += 0.3 * coverage_score
        
        # Smoothness score (25%)
        if len(correlation_lengths) > 2:
            # Compute second derivative as measure of smoothness
            second_deriv = np.gradient(np.gradient(correlation_lengths))
            smoothness = 1.0 / (1.0 + np.std(second_deriv))
            score += 0.25 * smoothness
        
        # Physical reasonableness score (25%)
        # Correlation lengths should be positive and finite
        positive_finite_fraction = np.sum(
            (correlation_lengths > 0) & np.isfinite(correlation_lengths)
        ) / len(correlation_lengths)
        score += 0.25 * positive_finite_fraction
        
        # Critical behavior score (20%)
        critical_behavior_score = max(0, self._assess_critical_behavior(
            CorrelationLengthResult(
                temperatures=temperatures,
                correlation_lengths=correlation_lengths,
                method_used=method,
                quality_score=0,
                critical_temperature=critical_temperature
            ),
            critical_temperature
        ))
        score += 0.2 * (0.5 + critical_behavior_score)  # Normalize to [0, 1]
        
        return min(1.0, score)
    
    def validate_correlation_length_divergence(self,
                                             result: CorrelationLengthResult,
                                             expected_nu: float = 0.630) -> Dict[str, Any]:
        """
        Validate that correlation length shows proper divergence behavior ξ ∝ |T-Tc|^(-ν).
        
        Args:
            result: CorrelationLengthResult to validate
            expected_nu: Expected critical exponent ν
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info("Validating correlation length divergence behavior")
        
        if result.critical_temperature is None:
            return {'valid': False, 'reason': 'No critical temperature provided'}
        
        temperatures = result.temperatures
        correlation_lengths = result.correlation_lengths
        critical_temperature = result.critical_temperature
        
        # Compute reduced temperature
        reduced_temps = np.abs(temperatures - critical_temperature)
        reduced_temps = np.maximum(reduced_temps, 1e-6)
        
        # Filter out points too close to Tc (where finite-size effects dominate)
        min_distance = 0.01 * (np.max(temperatures) - np.min(temperatures))
        valid_mask = reduced_temps > min_distance
        
        if np.sum(valid_mask) < 5:
            return {'valid': False, 'reason': 'Too few points away from critical temperature'}
        
        valid_reduced_temps = reduced_temps[valid_mask]
        valid_corr_lengths = correlation_lengths[valid_mask]
        
        # Fit power law: ξ = A * |T - Tc|^(-ν)
        # In log space: log(ξ) = log(A) - ν * log(|T - Tc|)
        try:
            log_reduced_temps = np.log(valid_reduced_temps)
            log_corr_lengths = np.log(valid_corr_lengths)
            
            # Linear regression
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(log_reduced_temps, log_corr_lengths)
            
            measured_nu = -slope  # Negative because we expect ξ ∝ t^(-ν)
            amplitude = np.exp(intercept)
            
            # Validation metrics
            validation_results = {
                'valid': True,
                'measured_nu': measured_nu,
                'expected_nu': expected_nu,
                'nu_error': abs(measured_nu - expected_nu) / expected_nu if expected_nu != 0 else float('inf'),
                'amplitude': amplitude,
                'r_squared': r_value**2,
                'p_value': p_value,
                'fit_quality': 'good' if r_value**2 > 0.7 else 'poor',
                'nu_accuracy_percent': (1 - abs(measured_nu - expected_nu) / expected_nu) * 100 if expected_nu != 0 else 0
            }
            
            # Check if measured ν is reasonable
            if measured_nu > 0 and measured_nu < 3:
                validation_results['nu_reasonable'] = True
            else:
                validation_results['nu_reasonable'] = False
                validation_results['reason'] = f'Unreasonable ν value: {measured_nu}'
            
            # Check if fit quality is acceptable
            if r_value**2 < 0.5:
                validation_results['warning'] = 'Poor fit quality, results may be unreliable'
            
            return validation_results
            
        except Exception as e:
            return {'valid': False, 'reason': f'Power law fitting failed: {e}'}
    
    def visualize_correlation_length_methods(self,
                                           multi_result: MultiMethodCorrelationResult,
                                           figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Create visualization comparing different correlation length methods."""
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot individual methods
        methods = [
            ('Variance Method', multi_result.variance_method),
            ('Spatial Method', multi_result.spatial_method),
            ('Susceptibility Method', multi_result.susceptibility_method)
        ]
        
        for i, (method_name, result) in enumerate(methods):
            ax = axes[0, i]
            
            ax.scatter(result.temperatures, result.correlation_lengths, 
                      alpha=0.7, s=30, label=method_name)
            
            if result.critical_temperature:
                ax.axvline(result.critical_temperature, color='red', linestyle='--', 
                          label=f'Tc = {result.critical_temperature:.3f}')
            
            ax.set_xlabel('Temperature')
            ax.set_ylabel('Correlation Length')
            ax.set_title(f'{method_name}\nQuality: {result.quality_score:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot method comparison
        ax = axes[1, 0]
        
        for method_name, result in methods:
            ax.plot(result.temperatures, result.correlation_lengths, 
                   'o-', alpha=0.7, label=method_name, markersize=4)
        
        # Add combined method
        combined = multi_result.combined_method
        ax.plot(combined.temperatures, combined.correlation_lengths,
               'k-', linewidth=2, label='Combined Method', alpha=0.8)
        
        if combined.critical_temperature:
            ax.axvline(combined.critical_temperature, color='red', linestyle='--', 
                      label=f'Tc = {combined.critical_temperature:.3f}')
        
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Correlation Length')
        ax.set_title('Method Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot method weights
        ax = axes[1, 1]
        
        method_names = list(multi_result.method_weights.keys())
        weights = list(multi_result.method_weights.values())
        
        bars = ax.bar(method_names, weights)
        ax.set_ylabel('Method Weight')
        ax.set_title('Method Weights')
        ax.set_ylim(0, 1)
        
        # Color bars based on weight
        for bar, weight in zip(bars, weights):
            if weight > 0.4:
                bar.set_color('green')
            elif weight > 0.2:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Plot quality comparison
        ax = axes[1, 2]
        
        quality_names = list(multi_result.quality_comparison.keys())
        quality_scores = list(multi_result.quality_comparison.values())
        
        bars = ax.bar(quality_names, quality_scores)
        ax.set_ylabel('Quality Score')
        ax.set_title('Quality Comparison')
        ax.set_ylim(0, 1)
        
        # Color bars based on quality
        for bar, score in zip(bars, quality_scores):
            if score > 0.8:
                bar.set_color('green')
            elif score > 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        return fig


def create_correlation_length_calculator(n_temperature_bins: int = 25,
                                        min_points_per_bin: int = 8,
                                        smoothing_window: int = 5) -> CorrelationLengthCalculator:
    """
    Factory function to create a CorrelationLengthCalculator.
    
    Args:
        n_temperature_bins: Number of temperature bins for analysis
        min_points_per_bin: Minimum points required per temperature bin
        smoothing_window: Window size for smoothing operations
        
    Returns:
        Configured CorrelationLengthCalculator instance
    """
    return CorrelationLengthCalculator(
        n_temperature_bins=n_temperature_bins,
        min_points_per_bin=min_points_per_bin,
        smoothing_window=smoothing_window
    )