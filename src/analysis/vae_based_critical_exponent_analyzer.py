"""
VAE-Based Critical Exponent Extraction Framework

This module implements task 7.4: Use VAE latent representations as order parameters
for critical exponent extraction instead of raw magnetization.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from scipy.optimize import curve_fit, minimize_scalar
from scipy.stats import linregress, pearsonr, spearmanr
from scipy.signal import savgol_filter
import warnings

from .latent_analysis import LatentRepresentation, LatentAnalyzer
from .improved_critical_exponent_analyzer import (
    ImprovedCriticalTemperatureDetector, ImprovedPowerLawFitter, 
    ImprovedPowerLawFitResult
)
from .enhanced_validation_types import (
    CriticalExponentValidation, ConfidenceInterval, 
    CriticalExponentError, UniversalityClass
)
from ..utils.logging_utils import get_logger


@dataclass
class VAEOrderParameterResult:
    """Container for VAE-based order parameter analysis results."""
    best_dimension: int
    correlation_with_magnetization: float
    correlation_with_temperature: float
    order_parameter_values: np.ndarray
    dimension_correlations: Dict[int, Dict[str, float]]
    selection_method: str
    confidence_score: float


@dataclass
class VAECriticalExponentResults:
    """Container for VAE-based critical exponent analysis results."""
    system_type: str
    critical_temperature: float
    tc_confidence: float
    
    # Order parameter analysis
    order_parameter_result: VAEOrderParameterResult
    
    # Critical exponents
    beta_result: Optional[ImprovedPowerLawFitResult] = None
    nu_result: Optional[ImprovedPowerLawFitResult] = None
    gamma_result: Optional[ImprovedPowerLawFitResult] = None
    
    # Accuracy metrics
    theoretical_exponents: Dict[str, float] = None
    accuracy_metrics: Dict[str, float] = None
    
    # Comparison with raw magnetization
    raw_magnetization_comparison: Optional[Dict[str, Any]] = None


class VAEOrderParameterSelector:
    """
    Intelligent selector for optimal VAE latent dimension to use as order parameter.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def select_optimal_order_parameter(self, 
                                     latent_repr: LatentRepresentation,
                                     method: str = 'comprehensive') -> VAEOrderParameterResult:
        """
        Select the optimal latent dimension to use as order parameter.
        
        Args:
            latent_repr: LatentRepresentation object
            method: Selection method ('correlation', 'temperature_sensitivity', 'comprehensive')
            
        Returns:
            VAEOrderParameterResult with optimal dimension and analysis
        """
        self.logger.info(f"Selecting optimal order parameter using {method} method")
        
        if method == 'correlation':
            return self._correlation_based_selection(latent_repr)
        elif method == 'temperature_sensitivity':
            return self._temperature_sensitivity_selection(latent_repr)
        elif method == 'comprehensive':
            return self._comprehensive_selection(latent_repr)
        else:
            raise ValueError(f"Unknown selection method: {method}")
    
    def _correlation_based_selection(self, latent_repr: LatentRepresentation) -> VAEOrderParameterResult:
        """Select dimension with highest correlation to magnetization."""
        
        # Analyze correlations for each latent dimension
        dimension_correlations = {}
        
        # Get latent coordinates (assuming 2D latent space)
        latent_coords = latent_repr.latent_coords
        n_dims = latent_coords.shape[1]
        
        abs_magnetizations = np.abs(latent_repr.magnetizations)
        
        for dim in range(n_dims):
            latent_values = latent_coords[:, dim]
            
            # Pearson correlation with magnetization
            pearson_r, pearson_p = pearsonr(latent_values, abs_magnetizations)
            
            # Spearman correlation (rank-based)
            spearman_r, spearman_p = spearmanr(latent_values, abs_magnetizations)
            
            # Correlation with temperature
            temp_pearson_r, temp_pearson_p = pearsonr(latent_values, latent_repr.temperatures)
            
            dimension_correlations[dim] = {
                'magnetization_pearson_r': pearson_r,
                'magnetization_pearson_p': pearson_p,
                'magnetization_spearman_r': spearman_r,
                'magnetization_spearman_p': spearman_p,
                'temperature_pearson_r': temp_pearson_r,
                'temperature_pearson_p': temp_pearson_p,
                'abs_magnetization_correlation': abs(pearson_r),
                'abs_temperature_correlation': abs(temp_pearson_r)
            }
        
        # Select dimension with highest absolute correlation to magnetization
        best_dim = max(dimension_correlations.keys(), 
                      key=lambda d: dimension_correlations[d]['abs_magnetization_correlation'])
        
        best_corr = dimension_correlations[best_dim]
        
        return VAEOrderParameterResult(
            best_dimension=best_dim,
            correlation_with_magnetization=best_corr['magnetization_pearson_r'],
            correlation_with_temperature=best_corr['temperature_pearson_r'],
            order_parameter_values=latent_coords[:, best_dim],
            dimension_correlations=dimension_correlations,
            selection_method='correlation',
            confidence_score=best_corr['abs_magnetization_correlation']
        )
    
    def _temperature_sensitivity_selection(self, latent_repr: LatentRepresentation) -> VAEOrderParameterResult:
        """Select dimension with highest temperature sensitivity around critical point."""
        
        latent_coords = latent_repr.latent_coords
        n_dims = latent_coords.shape[1]
        
        # Estimate critical temperature (rough estimate)
        theoretical_tc = {'ising_2d': 2.269, 'ising_3d': 4.511}
        # Use temperature range to guess system type
        temp_range = np.max(latent_repr.temperatures) - np.min(latent_repr.temperatures)
        if temp_range > 3:
            estimated_tc = 4.511  # 3D system
        else:
            estimated_tc = 2.269  # 2D system
        
        dimension_sensitivities = {}
        
        for dim in range(n_dims):
            latent_values = latent_coords[:, dim]
            
            # Compute temperature derivative (sensitivity)
            sensitivity = self._compute_temperature_sensitivity(
                latent_repr.temperatures, latent_values, estimated_tc
            )
            
            # Compute variance across temperature
            temp_variance = self._compute_temperature_variance(
                latent_repr.temperatures, latent_values
            )
            
            dimension_sensitivities[dim] = {
                'temperature_sensitivity': sensitivity,
                'temperature_variance': temp_variance,
                'combined_score': sensitivity * temp_variance
            }
        
        # Select dimension with highest combined score
        best_dim = max(dimension_sensitivities.keys(),
                      key=lambda d: dimension_sensitivities[d]['combined_score'])
        
        # Also compute correlations for the selected dimension
        latent_values = latent_coords[:, best_dim]
        mag_corr, _ = pearsonr(latent_values, np.abs(latent_repr.magnetizations))
        temp_corr, _ = pearsonr(latent_values, latent_repr.temperatures)
        
        return VAEOrderParameterResult(
            best_dimension=best_dim,
            correlation_with_magnetization=mag_corr,
            correlation_with_temperature=temp_corr,
            order_parameter_values=latent_values,
            dimension_correlations=dimension_sensitivities,
            selection_method='temperature_sensitivity',
            confidence_score=dimension_sensitivities[best_dim]['combined_score']
        )
    
    def _comprehensive_selection(self, latent_repr: LatentRepresentation) -> VAEOrderParameterResult:
        """Comprehensive selection combining multiple criteria."""
        
        # Get results from both methods
        corr_result = self._correlation_based_selection(latent_repr)
        temp_result = self._temperature_sensitivity_selection(latent_repr)
        
        latent_coords = latent_repr.latent_coords
        n_dims = latent_coords.shape[1]
        
        # Compute comprehensive scores for each dimension
        comprehensive_scores = {}
        
        for dim in range(n_dims):
            latent_values = latent_coords[:, dim]
            
            # Correlation score (40%)
            mag_corr = abs(corr_result.dimension_correlations[dim]['magnetization_pearson_r'])
            corr_score = mag_corr
            
            # Temperature sensitivity score (30%)
            if dim in temp_result.dimension_correlations:
                temp_sens = temp_result.dimension_correlations[dim]['combined_score']
                # Normalize temperature sensitivity
                max_temp_sens = max(temp_result.dimension_correlations[d]['combined_score'] 
                                  for d in temp_result.dimension_correlations)
                temp_score = temp_sens / max_temp_sens if max_temp_sens > 0 else 0
            else:
                temp_score = 0
            
            # Physical reasonableness score (20%)
            # Prefer dimensions that show clear phase transition behavior
            physics_score = self._compute_physics_score(latent_values, latent_repr.temperatures)
            
            # Statistical significance score (10%)
            mag_p_value = corr_result.dimension_correlations[dim]['magnetization_pearson_p']
            stat_score = 1 - mag_p_value if mag_p_value < 0.05 else 0
            
            # Combined score
            total_score = (0.4 * corr_score + 0.3 * temp_score + 
                          0.2 * physics_score + 0.1 * stat_score)
            
            comprehensive_scores[dim] = {
                'correlation_score': corr_score,
                'temperature_score': temp_score,
                'physics_score': physics_score,
                'statistical_score': stat_score,
                'total_score': total_score
            }
        
        # Select dimension with highest total score
        best_dim = max(comprehensive_scores.keys(),
                      key=lambda d: comprehensive_scores[d]['total_score'])
        
        # Get correlations for best dimension
        latent_values = latent_coords[:, best_dim]
        mag_corr, _ = pearsonr(latent_values, np.abs(latent_repr.magnetizations))
        temp_corr, _ = pearsonr(latent_values, latent_repr.temperatures)
        
        return VAEOrderParameterResult(
            best_dimension=best_dim,
            correlation_with_magnetization=mag_corr,
            correlation_with_temperature=temp_corr,
            order_parameter_values=latent_values,
            dimension_correlations=comprehensive_scores,
            selection_method='comprehensive',
            confidence_score=comprehensive_scores[best_dim]['total_score']
        )
    
    def _compute_temperature_sensitivity(self, temperatures: np.ndarray, 
                                       latent_values: np.ndarray,
                                       critical_temp: float) -> float:
        """Compute temperature sensitivity around critical point."""
        
        # Focus on region around critical temperature
        temp_range = np.max(temperatures) - np.min(temperatures)
        critical_window = 0.2 * temp_range
        
        critical_mask = np.abs(temperatures - critical_temp) <= critical_window
        
        if np.sum(critical_mask) < 5:
            # Use all data if too few points near critical temperature
            critical_mask = np.ones_like(temperatures, dtype=bool)
        
        crit_temps = temperatures[critical_mask]
        crit_latent = latent_values[critical_mask]
        
        # Compute derivative (gradient)
        if len(crit_temps) > 3:
            # Sort by temperature
            sort_idx = np.argsort(crit_temps)
            sorted_temps = crit_temps[sort_idx]
            sorted_latent = crit_latent[sort_idx]
            
            # Smooth if enough points
            if len(sorted_latent) > 7:
                smoothed_latent = savgol_filter(sorted_latent, 7, 3)
            else:
                smoothed_latent = sorted_latent
            
            # Compute gradient
            gradient = np.gradient(smoothed_latent, sorted_temps)
            
            # Return maximum absolute gradient as sensitivity measure
            return np.max(np.abs(gradient))
        else:
            return 0.0
    
    def _compute_temperature_variance(self, temperatures: np.ndarray, 
                                    latent_values: np.ndarray) -> float:
        """Compute variance of latent values across temperature."""
        
        # Bin by temperature and compute variance
        unique_temps = np.unique(temperatures)
        
        if len(unique_temps) < 3:
            return np.var(latent_values)
        
        temp_means = []
        for temp in unique_temps:
            temp_mask = temperatures == temp
            if np.sum(temp_mask) > 0:
                temp_means.append(np.mean(latent_values[temp_mask]))
        
        return np.var(temp_means) if len(temp_means) > 1 else 0.0
    
    def _compute_physics_score(self, latent_values: np.ndarray, 
                             temperatures: np.ndarray) -> float:
        """Compute physics-based score for order parameter quality."""
        
        # Look for monotonic behavior expected in order parameters
        unique_temps = np.unique(temperatures)
        
        if len(unique_temps) < 3:
            return 0.5
        
        temp_means = []
        for temp in sorted(unique_temps):
            temp_mask = temperatures == temp
            if np.sum(temp_mask) > 0:
                temp_means.append(np.mean(latent_values[temp_mask]))
        
        temp_means = np.array(temp_means)
        
        # Check for monotonic trend (order parameter should decrease with temperature)
        # Compute Spearman correlation (rank-based monotonicity)
        try:
            spearman_r, _ = spearmanr(sorted(unique_temps), temp_means)
            monotonicity_score = abs(spearman_r)
        except:
            monotonicity_score = 0.0
        
        # Check for clear transition (large change in order parameter)
        if len(temp_means) > 1:
            relative_change = (np.max(temp_means) - np.min(temp_means)) / (np.std(temp_means) + 1e-10)
            transition_score = min(1.0, relative_change / 5.0)  # Normalize
        else:
            transition_score = 0.0
        
        # Combined physics score
        physics_score = 0.6 * monotonicity_score + 0.4 * transition_score
        
        return physics_score


class VAECriticalExponentAnalyzer:
    """
    Main analyzer for VAE-based critical exponent extraction.
    """
    
    def __init__(self, 
                 system_type: str = 'ising_3d',
                 bootstrap_samples: int = 2000,
                 random_seed: Optional[int] = None):
        """
        Initialize VAE-based critical exponent analyzer.
        
        Args:
            system_type: Type of physical system
            bootstrap_samples: Number of bootstrap samples
            random_seed: Random seed for reproducibility
        """
        self.system_type = system_type
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.order_param_selector = VAEOrderParameterSelector()
        self.tc_detector = ImprovedCriticalTemperatureDetector()
        self.fitter = ImprovedPowerLawFitter(
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed
        )
        
        # Theoretical exponents
        self.theoretical_exponents = {
            'ising_2d': {'beta': 0.125, 'nu': 1.0, 'gamma': 1.75},
            'ising_3d': {'beta': 0.326, 'nu': 0.630, 'gamma': 1.237},
            'xy_2d': {'beta': 0.125, 'nu': 1.0, 'gamma': 1.75},
            'heisenberg_3d': {'beta': 0.365, 'nu': 0.705, 'gamma': 1.386}
        }
    
    def analyze_vae_critical_exponents(self, 
                                     latent_repr: LatentRepresentation,
                                     compare_with_raw_magnetization: bool = True) -> VAECriticalExponentResults:
        """
        Perform complete VAE-based critical exponent analysis.
        
        Args:
            latent_repr: LatentRepresentation object
            compare_with_raw_magnetization: Whether to compare with raw magnetization approach
            
        Returns:
            VAECriticalExponentResults with complete analysis
        """
        self.logger.info("Starting VAE-based critical exponent analysis")
        
        # Step 1: Select optimal order parameter from latent dimensions
        order_param_result = self.order_param_selector.select_optimal_order_parameter(
            latent_repr, method='comprehensive'
        )
        
        self.logger.info(f"Selected latent dimension {order_param_result.best_dimension} as order parameter")
        self.logger.info(f"Correlation with magnetization: {order_param_result.correlation_with_magnetization:.4f}")
        
        # Step 2: Detect critical temperature using VAE order parameter
        tc, tc_confidence = self.tc_detector.detect_critical_temperature(
            latent_repr.temperatures,
            np.abs(order_param_result.order_parameter_values),
            method='ensemble'
        )
        
        self.logger.info(f"Detected Tc = {tc:.4f} (confidence: {tc_confidence:.3f})")
        
        # Initialize results
        results = VAECriticalExponentResults(
            system_type=self.system_type,
            critical_temperature=tc,
            tc_confidence=tc_confidence,
            order_parameter_result=order_param_result,
            theoretical_exponents=self.theoretical_exponents.get(self.system_type, {})
        )
        
        # Step 3: Extract β exponent using VAE order parameter
        try:
            beta_result = self.fitter.fit_power_law_improved(
                latent_repr.temperatures,
                np.abs(order_param_result.order_parameter_values),
                tc,
                exponent_type='beta',
                adaptive_range=True
            )
            
            results.beta_result = beta_result
            self.logger.info(f"β exponent: {beta_result.exponent:.4f} ± {beta_result.exponent_error:.4f}")
            
        except Exception as e:
            self.logger.error(f"β exponent extraction failed: {e}")
            results.beta_result = None
        
        # Step 4: Extract ν exponent using correlation length from VAE latent space
        try:
            # Compute correlation length from latent space variance
            temps_binned, corr_lengths = self._compute_vae_correlation_length(latent_repr, tc)
            
            if len(temps_binned) >= self.fitter.min_points:
                nu_result = self.fitter.fit_power_law_improved(
                    temps_binned,
                    corr_lengths,
                    tc,
                    exponent_type='nu',
                    adaptive_range=True
                )
                
                results.nu_result = nu_result
                self.logger.info(f"ν exponent: {nu_result.exponent:.4f} ± {nu_result.exponent_error:.4f}")
            else:
                self.logger.warning("Insufficient data for ν exponent extraction")
                results.nu_result = None
                
        except Exception as e:
            self.logger.error(f"ν exponent extraction failed: {e}")
            results.nu_result = None
        
        # Step 5: Compute accuracy metrics
        results.accuracy_metrics = self._compute_vae_accuracy_metrics(results)
        
        # Step 6: Compare with raw magnetization approach if requested
        if compare_with_raw_magnetization:
            results.raw_magnetization_comparison = self._compare_with_raw_magnetization(
                latent_repr, tc, results
            )
        
        self.logger.info("VAE-based critical exponent analysis completed")
        
        return results
    
    def _compute_vae_correlation_length(self, 
                                      latent_repr: LatentRepresentation,
                                      critical_temperature: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute correlation length from VAE latent space."""
        
        # Use enhanced method that considers all latent dimensions
        n_temp_bins = min(25, len(np.unique(latent_repr.temperatures)) // 2)
        temp_min, temp_max = np.min(latent_repr.temperatures), np.max(latent_repr.temperatures)
        temp_bins = np.linspace(temp_min, temp_max, n_temp_bins + 1)
        temp_centers = (temp_bins[:-1] + temp_bins[1:]) / 2
        
        correlation_lengths = []
        valid_temps = []
        
        for i in range(len(temp_bins) - 1):
            temp_mask = (latent_repr.temperatures >= temp_bins[i]) & \
                       (latent_repr.temperatures < temp_bins[i + 1])
            
            if np.sum(temp_mask) >= 8:
                # Use all latent dimensions for correlation length
                z1_bin = latent_repr.z1[temp_mask]
                z2_bin = latent_repr.z2[temp_mask]
                
                # Method 1: Combined variance
                z1_var = np.var(z1_bin)
                z2_var = np.var(z2_bin)
                
                # Method 2: Spatial correlation in latent space
                latent_coords = np.column_stack([z1_bin, z2_bin])
                center = np.mean(latent_coords, axis=0)
                distances = np.linalg.norm(latent_coords - center, axis=1)
                spatial_corr = np.std(distances)
                
                # Method 3: Susceptibility-based (from magnetization variance)
                mag_bin = latent_repr.magnetizations[temp_mask]
                susceptibility = np.var(mag_bin)
                
                # Combine methods with physics-motivated weighting
                temp_center = temp_centers[i]
                distance_from_tc = abs(temp_center - critical_temperature)
                temp_range = temp_max - temp_min
                
                # Near Tc: emphasize susceptibility; far from Tc: emphasize variance
                if distance_from_tc < 0.15 * temp_range:
                    # Near critical point
                    corr_length = 0.4 * np.sqrt(susceptibility) + 0.4 * spatial_corr + 0.2 * np.sqrt(z1_var + z2_var)
                else:
                    # Away from critical point
                    corr_length = 0.5 * np.sqrt(z1_var + z2_var) + 0.3 * spatial_corr + 0.2 * np.sqrt(susceptibility)
                
                if corr_length > 0 and np.isfinite(corr_length):
                    correlation_lengths.append(corr_length)
                    valid_temps.append(temp_center)
        
        return np.array(valid_temps), np.array(correlation_lengths)
    
    def _compute_vae_accuracy_metrics(self, results: VAECriticalExponentResults) -> Dict[str, float]:
        """Compute accuracy metrics for VAE-based analysis."""
        
        accuracy_metrics = {}
        theoretical = results.theoretical_exponents
        
        # β exponent accuracy
        if results.beta_result and 'beta' in theoretical:
            beta_measured = results.beta_result.exponent
            beta_theoretical = theoretical['beta']
            
            beta_rel_error = abs(beta_measured - beta_theoretical) / beta_theoretical
            accuracy_metrics['beta_relative_error'] = beta_rel_error
            accuracy_metrics['beta_accuracy_percent'] = (1 - beta_rel_error) * 100
            
            # Check if theoretical value is in confidence interval
            if results.beta_result.confidence_interval:
                ci_lower = results.beta_result.confidence_interval.lower_bound
                ci_upper = results.beta_result.confidence_interval.upper_bound
                accuracy_metrics['beta_ci_contains_theoretical'] = ci_lower <= beta_theoretical <= ci_upper
        
        # ν exponent accuracy
        if results.nu_result and 'nu' in theoretical:
            nu_measured = results.nu_result.exponent
            nu_theoretical = theoretical['nu']
            
            nu_rel_error = abs(nu_measured - nu_theoretical) / nu_theoretical
            accuracy_metrics['nu_relative_error'] = nu_rel_error
            accuracy_metrics['nu_accuracy_percent'] = (1 - nu_rel_error) * 100
            
            # Check if theoretical value is in confidence interval
            if results.nu_result.confidence_interval:
                ci_lower = results.nu_result.confidence_interval.lower_bound
                ci_upper = results.nu_result.confidence_interval.upper_bound
                accuracy_metrics['nu_ci_contains_theoretical'] = ci_lower <= nu_theoretical <= ci_upper
        
        # Overall accuracy
        if 'beta_accuracy_percent' in accuracy_metrics and 'nu_accuracy_percent' in accuracy_metrics:
            overall_accuracy = (accuracy_metrics['beta_accuracy_percent'] + 
                              accuracy_metrics['nu_accuracy_percent']) / 2
            accuracy_metrics['overall_accuracy_percent'] = overall_accuracy
        
        # Order parameter quality metrics
        accuracy_metrics['order_parameter_magnetization_correlation'] = abs(
            results.order_parameter_result.correlation_with_magnetization
        )
        accuracy_metrics['order_parameter_confidence'] = results.order_parameter_result.confidence_score
        
        return accuracy_metrics
    
    def _compare_with_raw_magnetization(self, 
                                      latent_repr: LatentRepresentation,
                                      critical_temperature: float,
                                      vae_results: VAECriticalExponentResults) -> Dict[str, Any]:
        """Compare VAE-based results with raw magnetization approach."""
        
        self.logger.info("Comparing VAE results with raw magnetization approach")
        
        comparison = {}
        
        try:
            # Extract β exponent using raw magnetization
            raw_beta_result = self.fitter.fit_power_law_improved(
                latent_repr.temperatures,
                np.abs(latent_repr.magnetizations),
                critical_temperature,
                exponent_type='beta',
                adaptive_range=True
            )
            
            comparison['raw_magnetization_beta'] = {
                'exponent': raw_beta_result.exponent,
                'error': raw_beta_result.exponent_error,
                'r_squared': raw_beta_result.r_squared
            }
            
            # Compare accuracies if theoretical value available
            if 'beta' in vae_results.theoretical_exponents:
                theoretical_beta = vae_results.theoretical_exponents['beta']
                
                raw_error = abs(raw_beta_result.exponent - theoretical_beta) / theoretical_beta
                raw_accuracy = (1 - raw_error) * 100
                
                comparison['beta_accuracy_comparison'] = {
                    'raw_magnetization_accuracy': raw_accuracy,
                    'vae_accuracy': vae_results.accuracy_metrics.get('beta_accuracy_percent', 0),
                    'improvement': vae_results.accuracy_metrics.get('beta_accuracy_percent', 0) - raw_accuracy
                }
            
        except Exception as e:
            self.logger.warning(f"Raw magnetization β extraction failed: {e}")
            comparison['raw_magnetization_beta'] = None
        
        # Compare order parameter quality
        raw_mag_temp_corr, _ = pearsonr(np.abs(latent_repr.magnetizations), latent_repr.temperatures)
        vae_op_temp_corr = vae_results.order_parameter_result.correlation_with_temperature
        
        comparison['order_parameter_comparison'] = {
            'raw_magnetization_temp_correlation': raw_mag_temp_corr,
            'vae_order_parameter_temp_correlation': vae_op_temp_corr,
            'vae_magnetization_correlation': vae_results.order_parameter_result.correlation_with_magnetization
        }
        
        return comparison
    
    def visualize_vae_analysis(self, 
                             results: VAECriticalExponentResults,
                             latent_repr: LatentRepresentation,
                             figsize: Tuple[int, int] = (20, 12)) -> Figure:
        """Create comprehensive visualization of VAE-based analysis."""
        
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        
        # Plot 1: Latent space with order parameter highlighting
        ax = axes[0, 0]
        scatter = ax.scatter(latent_repr.z1, latent_repr.z2, 
                           c=np.abs(results.order_parameter_result.order_parameter_values),
                           cmap='viridis', alpha=0.6, s=20)
        ax.set_xlabel('Latent Dimension 0 (z₁)')
        ax.set_ylabel('Latent Dimension 1 (z₂)')
        ax.set_title(f'Latent Space\n(Order Parameter: Dim {results.order_parameter_result.best_dimension})')
        plt.colorbar(scatter, ax=ax, label='|Order Parameter|')
        
        # Plot 2: Order parameter vs temperature
        ax = axes[0, 1]
        ax.scatter(latent_repr.temperatures, 
                  np.abs(results.order_parameter_result.order_parameter_values),
                  alpha=0.6, s=15)
        ax.axvline(results.critical_temperature, color='red', linestyle='--', 
                  label=f'Tc = {results.critical_temperature:.3f}')
        ax.set_xlabel('Temperature')
        ax.set_ylabel('|VAE Order Parameter|')
        ax.set_title('VAE Order Parameter vs Temperature')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: β exponent fit
        if results.beta_result:
            ax = axes[0, 2]
            
            # Get data below Tc for β fit
            below_tc_mask = latent_repr.temperatures < results.critical_temperature
            fit_temps = latent_repr.temperatures[below_tc_mask]
            fit_op = np.abs(results.order_parameter_result.order_parameter_values[below_tc_mask])
            
            reduced_temps = results.critical_temperature - fit_temps
            
            ax.scatter(reduced_temps, fit_op, alpha=0.6, s=15, label='Data')
            
            # Plot power law fit
            if len(reduced_temps) > 0:
                temp_range = np.logspace(np.log10(np.min(reduced_temps[reduced_temps > 0])),
                                       np.log10(np.max(reduced_temps)), 100)
                fit_curve = results.beta_result.amplitude * (temp_range ** results.beta_result.exponent)
                ax.plot(temp_range, fit_curve, 'r-', linewidth=2,
                       label=f'β = {results.beta_result.exponent:.3f} ± {results.beta_result.exponent_error:.3f}')
            
            ax.set_xlabel('Tc - T')
            ax.set_ylabel('|VAE Order Parameter|')
            ax.set_title('β Exponent Fit (VAE)')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            axes[0, 2].text(0.5, 0.5, 'β fit failed', ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('β Exponent Fit (Failed)')
        
        # Plot 4: ν exponent fit
        if results.nu_result:
            ax = axes[0, 3]
            
            # Get correlation length data
            temps_binned, corr_lengths = self._compute_vae_correlation_length(latent_repr, results.critical_temperature)
            
            reduced_temps = np.abs(temps_binned - results.critical_temperature)
            
            ax.scatter(reduced_temps, corr_lengths, alpha=0.6, s=15, label='Data')
            
            # Plot power law fit
            if len(reduced_temps) > 0:
                temp_range = np.logspace(np.log10(np.min(reduced_temps[reduced_temps > 0])),
                                       np.log10(np.max(reduced_temps)), 100)
                fit_curve = results.nu_result.amplitude * (temp_range ** results.nu_result.exponent)
                ax.plot(temp_range, fit_curve, 'r-', linewidth=2,
                       label=f'ν = {results.nu_result.exponent:.3f} ± {results.nu_result.exponent_error:.3f}')
            
            ax.set_xlabel('|T - Tc|')
            ax.set_ylabel('Correlation Length')
            ax.set_title('ν Exponent Fit (VAE)')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            axes[0, 3].text(0.5, 0.5, 'ν fit failed', ha='center', va='center', transform=axes[0, 3].transAxes)
            axes[0, 3].set_title('ν Exponent Fit (Failed)')
        
        # Plot 5: Dimension correlations
        ax = axes[1, 0]
        dims = list(results.order_parameter_result.dimension_correlations.keys())
        if results.order_parameter_result.selection_method == 'comprehensive':
            scores = [results.order_parameter_result.dimension_correlations[d]['total_score'] for d in dims]
            ax.bar(dims, scores)
            ax.set_ylabel('Comprehensive Score')
        else:
            corrs = [abs(results.order_parameter_result.dimension_correlations[d].get('magnetization_pearson_r', 0)) for d in dims]
            ax.bar(dims, corrs)
            ax.set_ylabel('|Correlation with Magnetization|')
        
        ax.set_xlabel('Latent Dimension')
        ax.set_title('Dimension Selection Scores')
        ax.axvline(results.order_parameter_result.best_dimension, color='red', linestyle='--', 
                  label='Selected')
        ax.legend()
        
        # Plot 6: Accuracy comparison
        ax = axes[1, 1]
        if results.accuracy_metrics:
            metrics = []
            values = []
            
            if 'beta_accuracy_percent' in results.accuracy_metrics:
                metrics.append('β Accuracy')
                values.append(results.accuracy_metrics['beta_accuracy_percent'])
            
            if 'nu_accuracy_percent' in results.accuracy_metrics:
                metrics.append('ν Accuracy')
                values.append(results.accuracy_metrics['nu_accuracy_percent'])
            
            if 'overall_accuracy_percent' in results.accuracy_metrics:
                metrics.append('Overall')
                values.append(results.accuracy_metrics['overall_accuracy_percent'])
            
            if metrics:
                bars = ax.bar(metrics, values)
                ax.set_ylabel('Accuracy (%)')
                ax.set_title('VAE-Based Accuracy')
                ax.set_ylim(0, 100)
                
                # Color bars based on accuracy
                for bar, val in zip(bars, values):
                    if val >= 90:
                        bar.set_color('green')
                    elif val >= 80:
                        bar.set_color('orange')
                    else:
                        bar.set_color('red')
        
        # Plot 7: Raw magnetization comparison
        if results.raw_magnetization_comparison:
            ax = axes[1, 2]
            ax.scatter(latent_repr.temperatures, np.abs(latent_repr.magnetizations),
                      alpha=0.6, s=15, label='Raw Magnetization')
            ax.scatter(latent_repr.temperatures, 
                      np.abs(results.order_parameter_result.order_parameter_values),
                      alpha=0.6, s=15, label='VAE Order Parameter')
            ax.axvline(results.critical_temperature, color='red', linestyle='--', 
                      label=f'Tc = {results.critical_temperature:.3f}')
            ax.set_xlabel('Temperature')
            ax.set_ylabel('Order Parameter')
            ax.set_title('Order Parameter Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 8: Summary statistics
        ax = axes[1, 3]
        ax.axis('off')
        
        # Create summary text
        summary_text = f"VAE-Based Critical Exponent Analysis\n\n"
        summary_text += f"System: {results.system_type}\n"
        summary_text += f"Critical Temperature: {results.critical_temperature:.4f}\n"
        summary_text += f"Order Parameter: Latent Dim {results.order_parameter_result.best_dimension}\n"
        summary_text += f"OP-Magnetization Correlation: {results.order_parameter_result.correlation_with_magnetization:.4f}\n\n"
        
        if results.beta_result:
            theoretical_beta = results.theoretical_exponents.get('beta', 0)
            summary_text += f"β Exponent:\n"
            summary_text += f"  Measured: {results.beta_result.exponent:.4f} ± {results.beta_result.exponent_error:.4f}\n"
            summary_text += f"  Theoretical: {theoretical_beta:.4f}\n"
            if 'beta_accuracy_percent' in results.accuracy_metrics:
                summary_text += f"  Accuracy: {results.accuracy_metrics['beta_accuracy_percent']:.1f}%\n\n"
        
        if results.nu_result:
            theoretical_nu = results.theoretical_exponents.get('nu', 0)
            summary_text += f"ν Exponent:\n"
            summary_text += f"  Measured: {results.nu_result.exponent:.4f} ± {results.nu_result.exponent_error:.4f}\n"
            summary_text += f"  Theoretical: {theoretical_nu:.4f}\n"
            if 'nu_accuracy_percent' in results.accuracy_metrics:
                summary_text += f"  Accuracy: {results.accuracy_metrics['nu_accuracy_percent']:.1f}%\n\n"
        
        if 'overall_accuracy_percent' in results.accuracy_metrics:
            summary_text += f"Overall Accuracy: {results.accuracy_metrics['overall_accuracy_percent']:.1f}%"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        return fig


def create_vae_critical_exponent_analyzer(system_type: str = 'ising_3d',
                                        bootstrap_samples: int = 2000,
                                        random_seed: Optional[int] = None) -> VAECriticalExponentAnalyzer:
    """
    Factory function to create VAECriticalExponentAnalyzer.
    
    Args:
        system_type: Type of physical system
        bootstrap_samples: Number of bootstrap samples
        random_seed: Random seed for reproducibility
        
    Returns:
        Configured VAECriticalExponentAnalyzer instance
    """
    return VAECriticalExponentAnalyzer(
        system_type=system_type,
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed
    )