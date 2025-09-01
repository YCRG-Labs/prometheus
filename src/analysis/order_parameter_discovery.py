"""
Order Parameter Discovery Algorithms

This module implements algorithms for discovering order parameters from latent
space representations, including correlation analysis with physical quantities,
statistical significance testing, and comparison with theoretical predictions.
"""

import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Tuple, Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import logging
from pathlib import Path

from .latent_analysis import LatentRepresentation
from ..utils.logging_utils import get_logger


@dataclass
class CorrelationResult:
    """
    Results from correlation analysis between latent dimensions and physical quantities.
    
    Attributes:
        correlation_coefficient: Pearson correlation coefficient
        p_value: Statistical significance p-value
        confidence_interval: 95% confidence interval for correlation
        sample_size: Number of samples used in correlation
        is_significant: Whether correlation is statistically significant (p < 0.05)
    """
    correlation_coefficient: float
    p_value: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    is_significant: bool
    
    @property
    def strength_description(self) -> str:
        """Describe correlation strength in human-readable terms."""
        abs_corr = abs(self.correlation_coefficient)
        if abs_corr < 0.1:
            return "negligible"
        elif abs_corr < 0.3:
            return "weak"
        elif abs_corr < 0.5:
            return "moderate"
        elif abs_corr < 0.7:
            return "strong"
        else:
            return "very strong"


@dataclass
class OrderParameterCandidate:
    """
    Candidate order parameter discovered from latent space analysis.
    
    Attributes:
        latent_dimension: Which latent dimension ('z1' or 'z2')
        correlation_with_magnetization: Correlation result with magnetization
        correlation_with_energy: Correlation result with energy
        temperature_dependence: Temperature dependence characteristics
        critical_behavior: Analysis of behavior near critical temperature
        confidence_score: Overall confidence in this being a valid order parameter
    """
    latent_dimension: str
    correlation_with_magnetization: CorrelationResult
    correlation_with_energy: CorrelationResult
    temperature_dependence: Dict[str, Any]
    critical_behavior: Dict[str, Any]
    confidence_score: float
    
    @property
    def is_valid_order_parameter(self) -> bool:
        """Check if this is likely a valid order parameter."""
        return (self.correlation_with_magnetization.is_significant and
                abs(self.correlation_with_magnetization.correlation_coefficient) > 0.5 and
                self.confidence_score > 0.7)


class OrderParameterAnalyzer:
    """
    Main class for discovering and analyzing order parameters from latent representations.
    
    Implements correlation analysis, statistical significance testing, and comparison
    with theoretical predictions to identify which latent dimensions correspond to
    physical order parameters.
    """
    
    def __init__(self, critical_temperature: float = 2.269):
        """
        Initialize OrderParameterAnalyzer.
        
        Args:
            critical_temperature: Theoretical critical temperature for comparison
        """
        self.critical_temperature = critical_temperature
        self.logger = get_logger(__name__)
        
        self.logger.info(f"OrderParameterAnalyzer initialized with T_c = {critical_temperature}")
    
    def analyze_correlations(self,
                           latent_repr: LatentRepresentation,
                           confidence_level: float = 0.95) -> Dict[str, Dict[str, CorrelationResult]]:
        """
        Perform comprehensive correlation analysis between latent dimensions and physical quantities.
        
        Args:
            latent_repr: LatentRepresentation to analyze
            confidence_level: Confidence level for statistical tests
            
        Returns:
            Dictionary with correlation results for each latent dimension
        """
        self.logger.info("Performing correlation analysis between latent dimensions and physical quantities")
        
        results = {}
        
        # Physical quantities to correlate with
        physical_quantities = {
            'magnetization': latent_repr.magnetizations,
            'abs_magnetization': np.abs(latent_repr.magnetizations),
            'energy': latent_repr.energies,
            'temperature': latent_repr.temperatures
        }
        
        # Analyze each latent dimension
        for dim_name, dim_values in [('z1', latent_repr.z1), ('z2', latent_repr.z2)]:
            dim_results = {}
            
            for phys_name, phys_values in physical_quantities.items():
                corr_result = self._compute_correlation_with_significance(
                    dim_values, phys_values, confidence_level
                )
                dim_results[phys_name] = corr_result
                
                self.logger.debug(f"{dim_name} vs {phys_name}: r={corr_result.correlation_coefficient:.3f}, "
                                f"p={corr_result.p_value:.3e}, significant={corr_result.is_significant}")
            
            results[dim_name] = dim_results
        
        return results
    
    def _compute_correlation_with_significance(self,
                                             x: np.ndarray,
                                             y: np.ndarray,
                                             confidence_level: float) -> CorrelationResult:
        """
        Compute correlation coefficient with statistical significance testing.
        
        Args:
            x: First variable
            y: Second variable
            confidence_level: Confidence level for interval estimation
            
        Returns:
            CorrelationResult with correlation and significance information
        """
        # Remove any NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        
        if len(x_clean) < 3:
            # Not enough data for meaningful correlation
            return CorrelationResult(
                correlation_coefficient=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                sample_size=len(x_clean),
                is_significant=False
            )
        
        # Compute Pearson correlation
        corr_coef, p_value = stats.pearsonr(x_clean, y_clean)
        
        # Compute confidence interval using Fisher transformation
        n = len(x_clean)
        alpha = 1 - confidence_level
        
        # Fisher z-transformation
        z = np.arctanh(corr_coef)
        se = 1 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(1 - alpha/2)
        
        z_lower = z - z_crit * se
        z_upper = z + z_crit * se
        
        # Transform back to correlation scale
        ci_lower = np.tanh(z_lower)
        ci_upper = np.tanh(z_upper)
        
        return CorrelationResult(
            correlation_coefficient=float(corr_coef),
            p_value=float(p_value),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            sample_size=n,
            is_significant=p_value < 0.05
        )
    
    def discover_order_parameters(self, latent_repr: LatentRepresentation) -> List[OrderParameterCandidate]:
        """
        Discover order parameter candidates from latent representation.
        
        Args:
            latent_repr: LatentRepresentation to analyze
            
        Returns:
            List of OrderParameterCandidate objects ranked by confidence
        """
        self.logger.info("Discovering order parameter candidates")
        
        # Perform correlation analysis
        correlations = self.analyze_correlations(latent_repr)
        
        candidates = []
        
        # Analyze each latent dimension as potential order parameter
        for dim_name in ['z1', 'z2']:
            dim_values = latent_repr.z1 if dim_name == 'z1' else latent_repr.z2
            
            # Get correlations for this dimension
            dim_correlations = correlations[dim_name]
            
            # Analyze temperature dependence
            temp_dependence = self._analyze_temperature_dependence(dim_values, latent_repr.temperatures)
            
            # Analyze critical behavior
            critical_behavior = self._analyze_critical_behavior(
                dim_values, latent_repr.temperatures, latent_repr.magnetizations
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                dim_correlations, temp_dependence, critical_behavior
            )
            
            # Create candidate
            candidate = OrderParameterCandidate(
                latent_dimension=dim_name,
                correlation_with_magnetization=dim_correlations['abs_magnetization'],
                correlation_with_energy=dim_correlations['energy'],
                temperature_dependence=temp_dependence,
                critical_behavior=critical_behavior,
                confidence_score=confidence_score
            )
            
            candidates.append(candidate)
            
            self.logger.info(f"Order parameter candidate {dim_name}: "
                           f"mag_corr={candidate.correlation_with_magnetization.correlation_coefficient:.3f}, "
                           f"confidence={confidence_score:.3f}")
        
        # Sort by confidence score
        candidates.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return candidates
    
    def _analyze_temperature_dependence(self,
                                      latent_values: np.ndarray,
                                      temperatures: np.ndarray) -> Dict[str, Any]:
        """
        Analyze how latent dimension varies with temperature.
        
        Args:
            latent_values: Values of latent dimension
            temperatures: Temperature values
            
        Returns:
            Dictionary with temperature dependence analysis
        """
        # Get unique temperatures and compute statistics
        unique_temps = np.unique(temperatures)
        temp_means = []
        temp_stds = []
        temp_counts = []
        
        for temp in unique_temps:
            temp_mask = temperatures == temp
            temp_vals = latent_values[temp_mask]
            temp_means.append(np.mean(temp_vals))
            temp_stds.append(np.std(temp_vals))
            temp_counts.append(len(temp_vals))
        
        temp_means = np.array(temp_means)
        temp_stds = np.array(temp_stds)
        
        # Analyze trends
        temp_range = unique_temps[-1] - unique_temps[0]
        mean_range = np.max(temp_means) - np.min(temp_means)
        
        # Check for monotonic behavior
        temp_gradient = np.gradient(temp_means, unique_temps)
        is_monotonic = np.all(temp_gradient >= 0) or np.all(temp_gradient <= 0)
        
        # Check for critical point behavior (rapid change near T_c)
        critical_region_mask = np.abs(unique_temps - self.critical_temperature) < 0.2
        if np.any(critical_region_mask):
            critical_gradient = np.mean(np.abs(temp_gradient[critical_region_mask]))
            overall_gradient = np.mean(np.abs(temp_gradient))
            has_critical_enhancement = critical_gradient > 1.5 * overall_gradient
        else:
            has_critical_enhancement = False
        
        return {
            'unique_temperatures': unique_temps.tolist(),
            'temperature_means': temp_means.tolist(),
            'temperature_stds': temp_stds.tolist(),
            'temperature_range': float(temp_range),
            'mean_range': float(mean_range),
            'is_monotonic': bool(is_monotonic),
            'has_critical_enhancement': bool(has_critical_enhancement),
            'mean_gradient': float(np.mean(temp_gradient)),
            'gradient_std': float(np.std(temp_gradient))
        }
    
    def _analyze_critical_behavior(self,
                                 latent_values: np.ndarray,
                                 temperatures: np.ndarray,
                                 magnetizations: np.ndarray) -> Dict[str, Any]:
        """
        Analyze behavior near critical temperature.
        
        Args:
            latent_values: Values of latent dimension
            temperatures: Temperature values
            magnetizations: Magnetization values
            
        Returns:
            Dictionary with critical behavior analysis
        """
        # Define temperature regions
        critical_tolerance = 0.1
        low_temp_mask = temperatures < (self.critical_temperature - critical_tolerance)
        high_temp_mask = temperatures > (self.critical_temperature + critical_tolerance)
        critical_mask = ~(low_temp_mask | high_temp_mask)
        
        analysis = {}
        
        # Analyze separation between phases
        if np.any(low_temp_mask) and np.any(high_temp_mask):
            low_temp_mean = np.mean(latent_values[low_temp_mask])
            high_temp_mean = np.mean(latent_values[high_temp_mask])
            phase_separation = abs(high_temp_mean - low_temp_mean)
            
            # Compare with magnetization separation
            low_temp_mag = np.mean(np.abs(magnetizations[low_temp_mask]))
            high_temp_mag = np.mean(np.abs(magnetizations[high_temp_mask]))
            mag_separation = abs(low_temp_mag - high_temp_mag)
            
            analysis['phase_separation'] = float(phase_separation)
            analysis['magnetization_separation'] = float(mag_separation)
            analysis['separation_ratio'] = float(phase_separation / (mag_separation + 1e-10))
        else:
            analysis['phase_separation'] = 0.0
            analysis['magnetization_separation'] = 0.0
            analysis['separation_ratio'] = 0.0
        
        # Analyze variance near critical point
        if np.any(critical_mask):
            critical_variance = np.var(latent_values[critical_mask])
            overall_variance = np.var(latent_values)
            variance_enhancement = critical_variance / (overall_variance + 1e-10)
            
            analysis['critical_variance'] = float(critical_variance)
            analysis['variance_enhancement'] = float(variance_enhancement)
        else:
            analysis['critical_variance'] = 0.0
            analysis['variance_enhancement'] = 1.0
        
        # Check for order parameter-like behavior (should be ~0 above T_c, non-zero below)
        if np.any(low_temp_mask) and np.any(high_temp_mask):
            low_temp_std = np.std(latent_values[low_temp_mask])
            high_temp_std = np.std(latent_values[high_temp_mask])
            
            # Order parameter should have broader distribution below T_c
            std_ratio = low_temp_std / (high_temp_std + 1e-10)
            analysis['std_ratio_low_high'] = float(std_ratio)
            
            # Check if high-temp values are centered around zero
            high_temp_mean_abs = abs(np.mean(latent_values[high_temp_mask]))
            high_temp_std_norm = high_temp_std / (np.std(latent_values) + 1e-10)
            
            analysis['high_temp_centering'] = float(high_temp_mean_abs / (high_temp_std + 1e-10))
            analysis['high_temp_std_normalized'] = float(high_temp_std_norm)
        
        return analysis
    
    def _calculate_confidence_score(self,
                                  correlations: Dict[str, CorrelationResult],
                                  temp_dependence: Dict[str, Any],
                                  critical_behavior: Dict[str, Any]) -> float:
        """
        Calculate overall confidence score for order parameter candidate.
        
        Args:
            correlations: Correlation results with physical quantities
            temp_dependence: Temperature dependence analysis
            critical_behavior: Critical behavior analysis
            
        Returns:
            Confidence score between 0 and 1
        """
        score = 0.0
        
        # Correlation with magnetization (most important)
        mag_corr = correlations['abs_magnetization']
        if mag_corr.is_significant:
            score += 0.4 * min(abs(mag_corr.correlation_coefficient), 1.0)
        
        # Temperature dependence
        if temp_dependence['is_monotonic']:
            score += 0.1
        
        if temp_dependence['has_critical_enhancement']:
            score += 0.15
        
        # Phase separation
        if critical_behavior.get('phase_separation', 0) > 0.1:
            score += 0.15
        
        # Variance enhancement near critical point
        variance_enhancement = critical_behavior.get('variance_enhancement', 1.0)
        if variance_enhancement > 1.2:
            score += 0.1
        
        # Order parameter behavior (high-temp centering)
        high_temp_centering = critical_behavior.get('high_temp_centering', float('inf'))
        if high_temp_centering < 0.5:  # Well-centered around zero at high T
            score += 0.1
        
        return min(score, 1.0)
    
    def compare_with_theoretical(self,
                               candidates: List[OrderParameterCandidate],
                               latent_repr: LatentRepresentation) -> Dict[str, Any]:
        """
        Compare discovered order parameters with theoretical predictions.
        
        Args:
            candidates: List of order parameter candidates
            latent_repr: LatentRepresentation for comparison
            
        Returns:
            Dictionary with comparison results
        """
        self.logger.info("Comparing discovered order parameters with theoretical predictions")
        
        if not candidates:
            return {'error': 'No order parameter candidates provided'}
        
        # Get best candidate
        best_candidate = candidates[0]
        best_dim_values = (latent_repr.z1 if best_candidate.latent_dimension == 'z1' 
                          else latent_repr.z2)
        
        comparison = {
            'best_candidate': best_candidate.latent_dimension,
            'magnetization_correlation': best_candidate.correlation_with_magnetization.correlation_coefficient,
            'confidence_score': best_candidate.confidence_score
        }
        
        # Compare temperature dependence
        unique_temps = np.unique(latent_repr.temperatures)
        discovered_means = []
        theoretical_magnetization = []
        
        for temp in unique_temps:
            temp_mask = latent_repr.temperatures == temp
            
            # Discovered order parameter
            discovered_mean = np.mean(best_dim_values[temp_mask])
            discovered_means.append(discovered_mean)
            
            # Theoretical magnetization (simplified Ising model)
            if temp < self.critical_temperature:
                # Below critical temperature, approximate magnetization
                theoretical_mag = np.sqrt(max(0, 1 - (temp / self.critical_temperature) ** 4))
            else:
                theoretical_mag = 0.0
            theoretical_magnetization.append(theoretical_mag)
        
        discovered_means = np.array(discovered_means)
        theoretical_magnetization = np.array(theoretical_magnetization)
        
        # Normalize both to [0, 1] for comparison
        if np.max(discovered_means) > np.min(discovered_means):
            discovered_normalized = (discovered_means - np.min(discovered_means)) / (np.max(discovered_means) - np.min(discovered_means))
        else:
            discovered_normalized = np.zeros_like(discovered_means)
        
        # Compute similarity metrics
        if len(discovered_normalized) > 1:
            correlation_with_theory = np.corrcoef(discovered_normalized, theoretical_magnetization)[0, 1]
            rmse = np.sqrt(np.mean((discovered_normalized - theoretical_magnetization) ** 2))
        else:
            correlation_with_theory = 0.0
            rmse = 1.0
        
        comparison.update({
            'temperatures': unique_temps.tolist(),
            'discovered_order_parameter': discovered_normalized.tolist(),
            'theoretical_magnetization': theoretical_magnetization.tolist(),
            'correlation_with_theory': float(correlation_with_theory),
            'rmse_with_theory': float(rmse),
            'similarity_score': float(max(0, correlation_with_theory))
        })
        
        self.logger.info(f"Theoretical comparison: correlation={correlation_with_theory:.3f}, RMSE={rmse:.3f}")
        
        return comparison
    
    def visualize_order_parameter_discovery(self,
                                          candidates: List[OrderParameterCandidate],
                                          latent_repr: LatentRepresentation,
                                          figsize: Tuple[int, int] = (15, 10)) -> Figure:
        """
        Create comprehensive visualization of order parameter discovery results.
        
        Args:
            candidates: List of order parameter candidates
            latent_repr: LatentRepresentation for visualization
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure with order parameter analysis plots
        """
        self.logger.info("Creating order parameter discovery visualization")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        if not candidates:
            fig.suptitle("No Order Parameter Candidates Found")
            return fig
        
        # Get best candidate
        best_candidate = candidates[0]
        best_dim_name = best_candidate.latent_dimension
        best_dim_values = (latent_repr.z1 if best_dim_name == 'z1' else latent_repr.z2)
        other_dim_name = 'z2' if best_dim_name == 'z1' else 'z1'
        other_dim_values = (latent_repr.z2 if best_dim_name == 'z1' else latent_repr.z1)
        
        # Plot 1: Correlation with magnetization
        axes[0].scatter(np.abs(latent_repr.magnetizations), best_dim_values, 
                       c=latent_repr.temperatures, cmap='viridis', alpha=0.6, s=20)
        axes[0].set_xlabel('|Magnetization|')
        axes[0].set_ylabel(f'Latent Dimension {best_dim_name}')
        axes[0].set_title(f'Correlation with Magnetization\nr = {best_candidate.correlation_with_magnetization.correlation_coefficient:.3f}')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Temperature dependence
        unique_temps = np.array(best_candidate.temperature_dependence['unique_temperatures'])
        temp_means = np.array(best_candidate.temperature_dependence['temperature_means'])
        temp_stds = np.array(best_candidate.temperature_dependence['temperature_stds'])
        
        axes[1].errorbar(unique_temps, temp_means, yerr=temp_stds, 
                        fmt='o-', capsize=3, capthick=1, alpha=0.8)
        axes[1].axvline(self.critical_temperature, color='red', linestyle='--', 
                       label=f'T_c = {self.critical_temperature}')
        axes[1].set_xlabel('Temperature')
        axes[1].set_ylabel(f'Mean {best_dim_name}')
        axes[1].set_title('Temperature Dependence')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Phase separation in latent space
        low_temp_mask = latent_repr.temperatures < self.critical_temperature
        high_temp_mask = latent_repr.temperatures > self.critical_temperature
        
        axes[2].scatter(best_dim_values[low_temp_mask], other_dim_values[low_temp_mask],
                       c='blue', alpha=0.6, s=20, label=f'T < {self.critical_temperature}')
        axes[2].scatter(best_dim_values[high_temp_mask], other_dim_values[high_temp_mask],
                       c='red', alpha=0.6, s=20, label=f'T > {self.critical_temperature}')
        axes[2].set_xlabel(f'Latent Dimension {best_dim_name}')
        axes[2].set_ylabel(f'Latent Dimension {other_dim_name}')
        axes[2].set_title('Phase Separation in Latent Space')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Comparison with theoretical magnetization
        comparison = self.compare_with_theoretical(candidates, latent_repr)
        if 'temperatures' in comparison:
            comp_temps = np.array(comparison['temperatures'])
            discovered_op = np.array(comparison['discovered_order_parameter'])
            theoretical_mag = np.array(comparison['theoretical_magnetization'])
            
            axes[3].plot(comp_temps, discovered_op, 'o-', label='Discovered Order Parameter', alpha=0.8)
            axes[3].plot(comp_temps, theoretical_mag, 's-', label='Theoretical Magnetization', alpha=0.8)
            axes[3].axvline(self.critical_temperature, color='red', linestyle='--', alpha=0.7)
            axes[3].set_xlabel('Temperature')
            axes[3].set_ylabel('Normalized Value')
            axes[3].set_title(f'Comparison with Theory\nr = {comparison["correlation_with_theory"]:.3f}')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        
        # Plot 5: Confidence scores for all candidates
        candidate_names = [c.latent_dimension for c in candidates]
        confidence_scores = [c.confidence_score for c in candidates]
        mag_correlations = [abs(c.correlation_with_magnetization.correlation_coefficient) for c in candidates]
        
        x_pos = np.arange(len(candidates))
        width = 0.35
        
        axes[4].bar(x_pos - width/2, confidence_scores, width, label='Confidence Score', alpha=0.8)
        axes[4].bar(x_pos + width/2, mag_correlations, width, label='|Mag Correlation|', alpha=0.8)
        axes[4].set_xlabel('Latent Dimension')
        axes[4].set_ylabel('Score')
        axes[4].set_title('Order Parameter Candidate Scores')
        axes[4].set_xticks(x_pos)
        axes[4].set_xticklabels(candidate_names)
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)
        
        # Plot 6: Distribution comparison
        axes[5].hist(best_dim_values[low_temp_mask], bins=30, alpha=0.6, 
                    label=f'T < {self.critical_temperature}', density=True, color='blue')
        axes[5].hist(best_dim_values[high_temp_mask], bins=30, alpha=0.6,
                    label=f'T > {self.critical_temperature}', density=True, color='red')
        axes[5].set_xlabel(f'Latent Dimension {best_dim_name}')
        axes[5].set_ylabel('Density')
        axes[5].set_title('Distribution Comparison')
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.suptitle(f'Order Parameter Discovery Analysis\nBest Candidate: {best_dim_name} (Confidence: {best_candidate.confidence_score:.3f})', 
                    fontsize=14, y=0.98)
        
        return fig
    
    def generate_analysis_report(self,
                               candidates: List[OrderParameterCandidate],
                               latent_repr: LatentRepresentation) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report for order parameter discovery.
        
        Args:
            candidates: List of order parameter candidates
            latent_repr: LatentRepresentation for analysis
            
        Returns:
            Dictionary with complete analysis report
        """
        self.logger.info("Generating comprehensive order parameter analysis report")
        
        report = {
            'summary': {
                'n_candidates': len(candidates),
                'n_samples': latent_repr.n_samples,
                'temperature_range': (float(np.min(latent_repr.temperatures)), 
                                    float(np.max(latent_repr.temperatures))),
                'critical_temperature': self.critical_temperature
            },
            'candidates': [],
            'theoretical_comparison': {},
            'recommendations': []
        }
        
        # Analyze each candidate
        for candidate in candidates:
            candidate_report = {
                'latent_dimension': candidate.latent_dimension,
                'confidence_score': candidate.confidence_score,
                'is_valid': candidate.is_valid_order_parameter,
                'magnetization_correlation': {
                    'coefficient': candidate.correlation_with_magnetization.correlation_coefficient,
                    'p_value': candidate.correlation_with_magnetization.p_value,
                    'is_significant': candidate.correlation_with_magnetization.is_significant,
                    'strength': candidate.correlation_with_magnetization.strength_description
                },
                'energy_correlation': {
                    'coefficient': candidate.correlation_with_energy.correlation_coefficient,
                    'p_value': candidate.correlation_with_energy.p_value,
                    'is_significant': candidate.correlation_with_energy.is_significant
                },
                'temperature_dependence': candidate.temperature_dependence,
                'critical_behavior': candidate.critical_behavior
            }
            report['candidates'].append(candidate_report)
        
        # Theoretical comparison
        if candidates:
            report['theoretical_comparison'] = self.compare_with_theoretical(candidates, latent_repr)
        
        # Generate recommendations
        if candidates:
            best_candidate = candidates[0]
            
            if best_candidate.is_valid_order_parameter:
                report['recommendations'].append(
                    f"Strong order parameter candidate found: {best_candidate.latent_dimension} "
                    f"shows {best_candidate.correlation_with_magnetization.strength_description} "
                    f"correlation with magnetization (r = {best_candidate.correlation_with_magnetization.correlation_coefficient:.3f})"
                )
            else:
                report['recommendations'].append(
                    "No strong order parameter candidates identified. "
                    "Consider adjusting VAE architecture or training parameters."
                )
            
            # Additional recommendations based on analysis
            if best_candidate.confidence_score < 0.5:
                report['recommendations'].append(
                    "Low confidence scores suggest the latent space may not have learned "
                    "clear physical structure. Consider increasing β in β-VAE or adjusting latent dimension."
                )
            
            if not best_candidate.temperature_dependence['is_monotonic']:
                report['recommendations'].append(
                    "Non-monotonic temperature dependence detected. This may indicate "
                    "the model has learned complex but non-physical representations."
                )
        else:
            report['recommendations'].append("No order parameter candidates found.")
        
        return report