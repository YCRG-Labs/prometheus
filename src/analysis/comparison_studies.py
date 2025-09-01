"""
Comprehensive Comparison and Ablation Studies

This module provides comprehensive comparison studies including PCA baseline,
beta-VAE ablation studies, architecture comparisons, and statistical significance
testing for all discovered correlations and critical temperatures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import torch

from ..utils.logging_utils import get_logger
from .latent_analysis import LatentRepresentation
from .order_parameter_discovery import OrderParameterCandidate
from .phase_detection import PhaseDetectionResult


@dataclass
class ComparisonResult:
    """Container for comparison study results."""
    method_name: str
    latent_representation: np.ndarray
    order_parameter_correlation: float
    critical_temperature: Optional[float]
    physics_consistency_score: float
    computational_cost: float
    additional_metrics: Dict[str, Any]


@dataclass
class AblationResult:
    """Container for ablation study results."""
    parameter_name: str
    parameter_value: Any
    order_parameter_correlation: float
    critical_temperature_error: float
    physics_consistency_score: float
    training_time: float
    convergence_epochs: int


class ComparisonStudies:
    """
    Comprehensive comparison and ablation studies system.
    
    Provides detailed analysis including:
    - PCA baseline comparison
    - Beta-VAE ablation studies
    - Architecture comparison studies
    - Statistical significance testing
    """
    
    def __init__(self):
        """Initialize comparison studies system."""
        self.logger = get_logger(__name__)
        
        # Publication settings
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = sns.color_palette("husl", 10)
        
    def run_pca_baseline(self,
                        spin_configs: np.ndarray,
                        temperatures: np.ndarray,
                        magnetizations: np.ndarray,
                        n_components: int = 2) -> ComparisonResult:
        """
        Run PCA baseline for dimensionality reduction comparison.
        
        Args:
            spin_configs: Spin configurations [N, H, W]
            temperatures: Temperature values
            magnetizations: Magnetization values
            n_components: Number of PCA components
            
        Returns:
            ComparisonResult with PCA analysis
        """
        self.logger.info("Running PCA baseline comparison")
        
        # Flatten spin configurations
        n_samples = spin_configs.shape[0]
        flattened_configs = spin_configs.reshape(n_samples, -1)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        pca_representation = pca.fit_transform(flattened_configs)
        
        # Calculate correlation with magnetization
        correlations = []
        for i in range(n_components):
            corr, _ = pearsonr(np.abs(magnetizations), np.abs(pca_representation[:, i]))
            correlations.append(abs(corr))
        
        best_correlation = max(correlations)
        best_component = np.argmax(correlations)
        
        # Estimate critical temperature using clustering on best component
        critical_temp = self._estimate_critical_temperature_1d(
            pca_representation[:, best_component], temperatures
        )
        
        # Calculate physics consistency score
        physics_score = self._calculate_physics_consistency(
            best_correlation, critical_temp, 2.269
        )
        
        return ComparisonResult(
            method_name="PCA",
            latent_representation=pca_representation,
            order_parameter_correlation=best_correlation,
            critical_temperature=critical_temp,
            physics_consistency_score=physics_score,
            computational_cost=0.1,  # Relative cost
            additional_metrics={
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
                'best_component': best_component,
                'all_correlations': correlations
            }
        )
    
    def run_tsne_baseline(self,
                         spin_configs: np.ndarray,
                         temperatures: np.ndarray,
                         magnetizations: np.ndarray,
                         n_components: int = 2,
                         perplexity: float = 30.0) -> ComparisonResult:
        """
        Run t-SNE baseline for dimensionality reduction comparison.
        
        Args:
            spin_configs: Spin configurations [N, H, W]
            temperatures: Temperature values
            magnetizations: Magnetization values
            n_components: Number of t-SNE components
            perplexity: t-SNE perplexity parameter
            
        Returns:
            ComparisonResult with t-SNE analysis
        """
        self.logger.info("Running t-SNE baseline comparison")
        
        # Flatten spin configurations
        n_samples = spin_configs.shape[0]
        flattened_configs = spin_configs.reshape(n_samples, -1)
        
        # Apply t-SNE (use subset for computational efficiency)
        max_samples = min(1000, n_samples)
        indices = np.random.choice(n_samples, max_samples, replace=False)
        
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        tsne_representation = tsne.fit_transform(flattened_configs[indices])
        
        # Calculate correlation with magnetization
        correlations = []
        for i in range(n_components):
            corr, _ = pearsonr(np.abs(magnetizations[indices]), np.abs(tsne_representation[:, i]))
            correlations.append(abs(corr))
        
        best_correlation = max(correlations)
        best_component = np.argmax(correlations)
        
        # Estimate critical temperature
        critical_temp = self._estimate_critical_temperature_1d(
            tsne_representation[:, best_component], temperatures[indices]
        )
        
        # Calculate physics consistency score
        physics_score = self._calculate_physics_consistency(
            best_correlation, critical_temp, 2.269
        )
        
        return ComparisonResult(
            method_name="t-SNE",
            latent_representation=tsne_representation,
            order_parameter_correlation=best_correlation,
            critical_temperature=critical_temp,
            physics_consistency_score=physics_score,
            computational_cost=10.0,  # Relative cost
            additional_metrics={
                'perplexity': perplexity,
                'best_component': best_component,
                'all_correlations': correlations,
                'sample_indices': indices
            }
        )
    
    def run_beta_ablation_study(self,
                               vae_results: Dict[float, Dict[str, Any]],
                               theoretical_tc: float = 2.269) -> List[AblationResult]:
        """
        Run beta-VAE ablation study showing impact of different beta values.
        
        Args:
            vae_results: Dictionary mapping beta values to VAE results
            theoretical_tc: Theoretical critical temperature
            
        Returns:
            List of AblationResult objects
        """
        self.logger.info("Running beta-VAE ablation study")
        
        ablation_results = []
        
        for beta_value, results in vae_results.items():
            # Extract metrics from results
            order_param_corr = results.get('order_parameter_correlation', 0.0)
            discovered_tc = results.get('critical_temperature', theoretical_tc)
            tc_error = abs(discovered_tc - theoretical_tc) / theoretical_tc * 100
            training_time = results.get('training_time', 0.0)
            convergence_epochs = results.get('convergence_epochs', 0)
            
            # Calculate physics consistency score
            physics_score = self._calculate_physics_consistency(
                order_param_corr, discovered_tc, theoretical_tc
            )
            
            ablation_result = AblationResult(
                parameter_name="beta",
                parameter_value=beta_value,
                order_parameter_correlation=order_param_corr,
                critical_temperature_error=tc_error,
                physics_consistency_score=physics_score,
                training_time=training_time,
                convergence_epochs=convergence_epochs
            )
            
            ablation_results.append(ablation_result)
        
        return ablation_results
    
    def run_architecture_comparison(self,
                                  architecture_results: Dict[str, Dict[str, Any]],
                                  theoretical_tc: float = 2.269) -> List[ComparisonResult]:
        """
        Run architecture comparison study for different latent dimensions and layer depths.
        
        Args:
            architecture_results: Dictionary mapping architecture names to results
            theoretical_tc: Theoretical critical temperature
            
        Returns:
            List of ComparisonResult objects
        """
        self.logger.info("Running architecture comparison study")
        
        comparison_results = []
        
        for arch_name, results in architecture_results.items():
            # Extract metrics from results
            order_param_corr = results.get('order_parameter_correlation', 0.0)
            discovered_tc = results.get('critical_temperature', theoretical_tc)
            training_time = results.get('training_time', 0.0)
            latent_repr = results.get('latent_representation', np.array([]))
            
            # Calculate physics consistency score
            physics_score = self._calculate_physics_consistency(
                order_param_corr, discovered_tc, theoretical_tc
            )
            
            comparison_result = ComparisonResult(
                method_name=arch_name,
                latent_representation=latent_repr,
                order_parameter_correlation=order_param_corr,
                critical_temperature=discovered_tc,
                physics_consistency_score=physics_score,
                computational_cost=training_time,
                additional_metrics=results
            )
            
            comparison_results.append(comparison_result)
        
        return comparison_results
    
    def test_statistical_significance(self,
                                    correlations: List[float],
                                    critical_temperatures: List[float],
                                    theoretical_tc: float = 2.269,
                                    alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform statistical significance testing for discovered correlations and critical temperatures.
        
        Args:
            correlations: List of order parameter correlations
            critical_temperatures: List of discovered critical temperatures
            theoretical_tc: Theoretical critical temperature
            alpha: Significance level
            
        Returns:
            Dictionary with statistical test results
        """
        self.logger.info("Performing statistical significance testing")
        
        results = {}
        
        # Test correlation significance
        if correlations:
            # One-sample t-test against zero correlation
            corr_stat, corr_pvalue = stats.ttest_1samp(correlations, 0.0)
            
            # Test if correlations are significantly different from random (0.1)
            random_threshold = 0.1
            corr_vs_random_stat, corr_vs_random_pvalue = stats.ttest_1samp(
                correlations, random_threshold
            )
            
            results['correlation_tests'] = {
                'mean_correlation': np.mean(correlations),
                'std_correlation': np.std(correlations),
                'vs_zero': {
                    'statistic': corr_stat,
                    'p_value': corr_pvalue,
                    'significant': corr_pvalue < alpha
                },
                'vs_random': {
                    'statistic': corr_vs_random_stat,
                    'p_value': corr_vs_random_pvalue,
                    'significant': corr_vs_random_pvalue < alpha
                },
                'confidence_interval': stats.t.interval(
                    1 - alpha, len(correlations) - 1,
                    loc=np.mean(correlations),
                    scale=stats.sem(correlations)
                )
            }
        
        # Test critical temperature accuracy
        if critical_temperatures:
            # One-sample t-test against theoretical value
            tc_stat, tc_pvalue = stats.ttest_1samp(critical_temperatures, theoretical_tc)
            
            # Calculate relative errors
            relative_errors = [(tc - theoretical_tc) / theoretical_tc * 100 
                             for tc in critical_temperatures]
            
            # Test if errors are significantly different from zero
            error_stat, error_pvalue = stats.ttest_1samp(relative_errors, 0.0)
            
            results['critical_temperature_tests'] = {
                'mean_tc': np.mean(critical_temperatures),
                'std_tc': np.std(critical_temperatures),
                'mean_relative_error': np.mean(relative_errors),
                'std_relative_error': np.std(relative_errors),
                'vs_theoretical': {
                    'statistic': tc_stat,
                    'p_value': tc_pvalue,
                    'significant': tc_pvalue < alpha
                },
                'error_vs_zero': {
                    'statistic': error_stat,
                    'p_value': error_pvalue,
                    'significant': error_pvalue < alpha
                },
                'confidence_interval': stats.t.interval(
                    1 - alpha, len(critical_temperatures) - 1,
                    loc=np.mean(critical_temperatures),
                    scale=stats.sem(critical_temperatures)
                )
            }
        
        # Combined significance test
        if correlations and critical_temperatures:
            # Test if methods with higher correlations also have more accurate Tc
            corr_accuracy = [-abs(tc - theoretical_tc) for tc in critical_temperatures]
            spearman_corr, spearman_p = spearmanr(correlations, corr_accuracy)
            
            results['combined_tests'] = {
                'correlation_vs_accuracy': {
                    'spearman_correlation': spearman_corr,
                    'p_value': spearman_p,
                    'significant': spearman_p < alpha
                }
            }
        
        return results
    
    def plot_baseline_comparison(self,
                               vae_result: ComparisonResult,
                               pca_result: ComparisonResult,
                               tsne_result: Optional[ComparisonResult] = None,
                               figsize: Tuple[int, int] = (15, 10)) -> Figure:
        """
        Create baseline comparison visualization.
        
        Args:
            vae_result: VAE comparison result
            pca_result: PCA comparison result
            tsne_result: t-SNE comparison result (optional)
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure with baseline comparison
        """
        self.logger.info("Creating baseline comparison visualization")
        
        methods = [vae_result, pca_result]
        if tsne_result:
            methods.append(tsne_result)
        
        fig, axes = plt.subplots(2, len(methods), figsize=figsize)
        
        if len(methods) == 2:
            axes = axes.reshape(2, 2)
        
        # Plot latent representations
        for i, method in enumerate(methods):
            ax_latent = axes[0, i]
            
            if method.latent_representation.shape[1] >= 2:
                scatter = ax_latent.scatter(
                    method.latent_representation[:, 0],
                    method.latent_representation[:, 1],
                    c=np.arange(len(method.latent_representation)),
                    cmap='viridis', alpha=0.6, s=20
                )
                
                ax_latent.set_xlabel('Component 1', fontsize=12)
                ax_latent.set_ylabel('Component 2', fontsize=12)
                ax_latent.set_title(f'{method.method_name} Representation', 
                                  fontsize=14, fontweight='bold')
                ax_latent.grid(True, alpha=0.3)
                ax_latent.set_aspect('equal', adjustable='box')
        
        # Plot comparison metrics
        ax_metrics = axes[1, :]
        
        # Prepare data for comparison
        method_names = [m.method_name for m in methods]
        correlations = [m.order_parameter_correlation for m in methods]
        physics_scores = [m.physics_consistency_score for m in methods]
        comp_costs = [m.computational_cost for m in methods]
        
        # Metrics comparison
        x_pos = np.arange(len(method_names))
        width = 0.25
        
        if len(methods) == 2:
            ax_comp = axes[1, 0]
        else:
            ax_comp = plt.subplot(2, len(methods), len(methods) + 1)
        
        bars1 = ax_comp.bar(x_pos - width, correlations, width, 
                           label='Order Parameter Correlation', alpha=0.8)
        bars2 = ax_comp.bar(x_pos, physics_scores, width, 
                           label='Physics Consistency', alpha=0.8)
        bars3 = ax_comp.bar(x_pos + width, np.array(comp_costs) / max(comp_costs), width, 
                           label='Computational Cost (Normalized)', alpha=0.8)
        
        ax_comp.set_xlabel('Method', fontsize=12)
        ax_comp.set_ylabel('Score', fontsize=12)
        ax_comp.set_title('Method Comparison', fontsize=14, fontweight='bold')
        ax_comp.set_xticks(x_pos)
        ax_comp.set_xticklabels(method_names)
        ax_comp.legend()
        ax_comp.grid(True, alpha=0.3)
        
        # Add value annotations
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax_comp.annotate(f'{height:.3f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)
        
        # Critical temperature comparison (if available)
        if len(methods) == 2:
            ax_tc = axes[1, 1]
        else:
            ax_tc = plt.subplot(2, len(methods), 2 * len(methods))
        
        tc_values = [m.critical_temperature for m in methods if m.critical_temperature]
        tc_methods = [m.method_name for m in methods if m.critical_temperature]
        
        if tc_values:
            theoretical_tc = 2.269
            tc_errors = [abs(tc - theoretical_tc) / theoretical_tc * 100 for tc in tc_values]
            
            bars = ax_tc.bar(tc_methods, tc_errors, alpha=0.8, color='red')
            ax_tc.axhline(y=5.0, color='black', linestyle='--', alpha=0.7, 
                         label='5% Error Threshold')
            
            ax_tc.set_xlabel('Method', fontsize=12)
            ax_tc.set_ylabel('Critical Temperature Error (%)', fontsize=12)
            ax_tc.set_title('Critical Temperature Accuracy', fontsize=14, fontweight='bold')
            ax_tc.legend()
            ax_tc.grid(True, alpha=0.3)
            
            # Add value annotations
            for bar, error in zip(bars, tc_errors):
                height = bar.get_height()
                ax_tc.annotate(f'{error:.1f}%',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3), textcoords="offset points",
                              ha='center', va='bottom', fontsize=10)
        else:
            ax_tc.text(0.5, 0.5, 'No Critical Temperature\nData Available', 
                      transform=ax_tc.transAxes, ha='center', va='center',
                      fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
        
        plt.suptitle('Dimensionality Reduction Method Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_ablation_study(self,
                           ablation_results: List[AblationResult],
                           figsize: Tuple[int, int] = (15, 10)) -> Figure:
        """
        Create ablation study visualization.
        
        Args:
            ablation_results: List of ablation study results
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure with ablation study analysis
        """
        self.logger.info("Creating ablation study visualization")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Extract data
        param_values = [r.parameter_value for r in ablation_results]
        correlations = [r.order_parameter_correlation for r in ablation_results]
        tc_errors = [r.critical_temperature_error for r in ablation_results]
        physics_scores = [r.physics_consistency_score for r in ablation_results]
        training_times = [r.training_time for r in ablation_results]
        
        # Plot 1: Order parameter correlation vs parameter value
        ax1 = axes[0, 0]
        ax1.plot(param_values, correlations, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Beta Value', fontsize=12)
        ax1.set_ylabel('Order Parameter Correlation', fontsize=12)
        ax1.set_title('Order Parameter Discovery vs Beta', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Add annotations for best performance
        best_idx = np.argmax(correlations)
        ax1.annotate(f'Best: β={param_values[best_idx]:.1f}\nr={correlations[best_idx]:.3f}',
                    xy=(param_values[best_idx], correlations[best_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Plot 2: Critical temperature error vs parameter value
        ax2 = axes[0, 1]
        ax2.plot(param_values, tc_errors, 'ro-', linewidth=2, markersize=8)
        ax2.axhline(y=5.0, color='black', linestyle='--', alpha=0.7, 
                   label='5% Error Threshold')
        ax2.set_xlabel('Beta Value', fontsize=12)
        ax2.set_ylabel('Critical Temperature Error (%)', fontsize=12)
        ax2.set_title('Critical Temperature Accuracy vs Beta', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        # Add annotations for best performance
        best_tc_idx = np.argmin(tc_errors)
        ax2.annotate(f'Best: β={param_values[best_tc_idx]:.1f}\nError={tc_errors[best_tc_idx]:.1f}%',
                    xy=(param_values[best_tc_idx], tc_errors[best_tc_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Plot 3: Physics consistency score vs parameter value
        ax3 = axes[1, 0]
        ax3.plot(param_values, physics_scores, 'go-', linewidth=2, markersize=8)
        ax3.axhline(y=0.8, color='black', linestyle='--', alpha=0.7, 
                   label='Target Score (0.8)')
        ax3.set_xlabel('Beta Value', fontsize=12)
        ax3.set_ylabel('Physics Consistency Score', fontsize=12)
        ax3.set_title('Overall Physics Consistency vs Beta', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        
        # Plot 4: Training efficiency vs parameter value
        ax4 = axes[1, 1]
        
        # Normalize training times for comparison
        norm_times = np.array(training_times) / max(training_times)
        
        # Create efficiency score (high correlation, low error, low time)
        efficiency_scores = np.array(correlations) * (1 - np.array(tc_errors) / 100) * (1 - norm_times)
        
        ax4.plot(param_values, efficiency_scores, 'mo-', linewidth=2, markersize=8)
        ax4.set_xlabel('Beta Value', fontsize=12)
        ax4.set_ylabel('Training Efficiency Score', fontsize=12)
        ax4.set_title('Training Efficiency vs Beta', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        
        # Add annotations for best efficiency
        best_eff_idx = np.argmax(efficiency_scores)
        ax4.annotate(f'Best: β={param_values[best_eff_idx]:.1f}\nScore={efficiency_scores[best_eff_idx]:.3f}',
                    xy=(param_values[best_eff_idx], efficiency_scores[best_eff_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.suptitle('Beta-VAE Ablation Study Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_statistical_significance(self,
                                    significance_results: Dict[str, Any],
                                    figsize: Tuple[int, int] = (12, 8)) -> Figure:
        """
        Create statistical significance testing visualization.
        
        Args:
            significance_results: Statistical test results
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure with significance analysis
        """
        self.logger.info("Creating statistical significance visualization")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Correlation significance
        if 'correlation_tests' in significance_results:
            ax1 = axes[0, 0]
            corr_data = significance_results['correlation_tests']
            
            mean_corr = corr_data['mean_correlation']
            ci_low, ci_high = corr_data['confidence_interval']
            
            # Bar plot with confidence interval
            ax1.bar(['Order Parameter\nCorrelation'], [mean_corr], 
                   yerr=[[mean_corr - ci_low], [ci_high - mean_corr]], 
                   capsize=10, alpha=0.7, color='blue')
            
            # Add significance indicators
            vs_zero_sig = corr_data['vs_zero']['significant']
            vs_random_sig = corr_data['vs_random']['significant']
            
            sig_text = f"vs Zero: {'***' if vs_zero_sig else 'ns'}\nvs Random: {'***' if vs_random_sig else 'ns'}"
            ax1.text(0.05, 0.95, sig_text, transform=ax1.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax1.set_ylabel('Correlation Coefficient', fontsize=12)
            ax1.set_title('Order Parameter Correlation\nSignificance', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Critical temperature significance
        if 'critical_temperature_tests' in significance_results:
            ax2 = axes[0, 1]
            tc_data = significance_results['critical_temperature_tests']
            
            mean_error = tc_data['mean_relative_error']
            theoretical_tc = 2.269
            
            # Error bar plot
            ax2.bar(['Critical Temperature\nError'], [abs(mean_error)], 
                   alpha=0.7, color='red')
            
            # Add significance indicators
            vs_theoretical_sig = tc_data['vs_theoretical']['significant']
            error_vs_zero_sig = tc_data['error_vs_zero']['significant']
            
            sig_text = f"vs Theoretical: {'***' if vs_theoretical_sig else 'ns'}\nError vs Zero: {'***' if error_vs_zero_sig else 'ns'}"
            ax2.text(0.05, 0.95, sig_text, transform=ax2.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax2.axhline(y=5.0, color='black', linestyle='--', alpha=0.7, 
                       label='5% Threshold')
            ax2.set_ylabel('Relative Error (%)', fontsize=12)
            ax2.set_title('Critical Temperature\nAccuracy Significance', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Combined significance
        if 'combined_tests' in significance_results:
            ax3 = axes[1, 0]
            combined_data = significance_results['combined_tests']
            
            spearman_corr = combined_data['correlation_vs_accuracy']['spearman_correlation']
            spearman_sig = combined_data['correlation_vs_accuracy']['significant']
            
            # Correlation strength visualization
            ax3.bar(['Correlation vs\nAccuracy'], [abs(spearman_corr)], 
                   alpha=0.7, color='green')
            
            sig_text = f"Spearman r = {spearman_corr:.3f}\n{'***' if spearman_sig else 'ns'}"
            ax3.text(0.05, 0.95, sig_text, transform=ax3.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax3.set_ylabel('|Correlation Coefficient|', fontsize=12)
            ax3.set_title('Correlation-Accuracy\nRelationship', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: P-value summary
        ax4 = axes[1, 1]
        
        # Collect all p-values
        p_values = []
        test_names = []
        
        if 'correlation_tests' in significance_results:
            corr_data = significance_results['correlation_tests']
            p_values.extend([corr_data['vs_zero']['p_value'], 
                           corr_data['vs_random']['p_value']])
            test_names.extend(['Corr vs Zero', 'Corr vs Random'])
        
        if 'critical_temperature_tests' in significance_results:
            tc_data = significance_results['critical_temperature_tests']
            p_values.extend([tc_data['vs_theoretical']['p_value'], 
                           tc_data['error_vs_zero']['p_value']])
            test_names.extend(['Tc vs Theoretical', 'Error vs Zero'])
        
        if 'combined_tests' in significance_results:
            combined_data = significance_results['combined_tests']
            p_values.append(combined_data['correlation_vs_accuracy']['p_value'])
            test_names.append('Corr-Acc Relation')
        
        if p_values:
            # Log scale p-value plot
            log_p_values = [-np.log10(p) for p in p_values]
            colors = ['red' if p < 0.05 else 'gray' for p in p_values]
            
            bars = ax4.bar(range(len(test_names)), log_p_values, color=colors, alpha=0.7)
            ax4.axhline(y=-np.log10(0.05), color='black', linestyle='--', 
                       alpha=0.7, label='α = 0.05')
            
            ax4.set_xlabel('Statistical Test', fontsize=12)
            ax4.set_ylabel('-log₁₀(p-value)', fontsize=12)
            ax4.set_title('Statistical Significance\nSummary', fontsize=14, fontweight='bold')
            ax4.set_xticks(range(len(test_names)))
            ax4.set_xticklabels(test_names, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Add p-value annotations
            for bar, p_val in zip(bars, p_values):
                height = bar.get_height()
                ax4.annotate(f'p={p_val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, rotation=90)
        
        plt.suptitle('Statistical Significance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_architecture_comparison(self,
                                   architecture_results: List[ComparisonResult],
                                   figsize: Tuple[int, int] = (15, 12)) -> Figure:
        """
        Create architecture comparison visualization for different latent dimensions and layer depths.
        
        Args:
            architecture_results: List of architecture comparison results
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure with architecture comparison analysis
        """
        self.logger.info("Creating architecture comparison visualization")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Extract data
        arch_names = [r.method_name for r in architecture_results]
        correlations = [r.order_parameter_correlation for r in architecture_results]
        physics_scores = [r.physics_consistency_score for r in architecture_results]
        comp_costs = [r.computational_cost for r in architecture_results]
        critical_temps = [r.critical_temperature for r in architecture_results if r.critical_temperature]
        
        # Parse architecture parameters from names (assuming format like "LatentDim_2_Layers_3")
        latent_dims = []
        layer_counts = []
        for name in arch_names:
            try:
                parts = name.split('_')
                if 'LatentDim' in parts:
                    dim_idx = parts.index('LatentDim') + 1
                    latent_dims.append(int(parts[dim_idx]))
                else:
                    latent_dims.append(2)  # Default
                
                if 'Layers' in parts:
                    layer_idx = parts.index('Layers') + 1
                    layer_counts.append(int(parts[layer_idx]))
                else:
                    layer_counts.append(3)  # Default
            except (ValueError, IndexError):
                latent_dims.append(2)
                layer_counts.append(3)
        
        # Plot 1: Order parameter correlation by architecture
        ax1 = axes[0, 0]
        bars = ax1.bar(range(len(arch_names)), correlations, alpha=0.7, color='blue')
        ax1.set_xlabel('Architecture', fontsize=12)
        ax1.set_ylabel('Order Parameter Correlation', fontsize=12)
        ax1.set_title('Order Parameter Discovery\nby Architecture', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(arch_names)))
        ax1.set_xticklabels(arch_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value annotations
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax1.annotate(f'{corr:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Physics consistency by architecture
        ax2 = axes[0, 1]
        bars = ax2.bar(range(len(arch_names)), physics_scores, alpha=0.7, color='green')
        ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Target (0.8)')
        ax2.set_xlabel('Architecture', fontsize=12)
        ax2.set_ylabel('Physics Consistency Score', fontsize=12)
        ax2.set_title('Physics Consistency\nby Architecture', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(arch_names)))
        ax2.set_xticklabels(arch_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value annotations
        for bar, score in zip(bars, physics_scores):
            height = bar.get_height()
            ax2.annotate(f'{score:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        
        # Plot 3: Computational cost by architecture
        ax3 = axes[0, 2]
        bars = ax3.bar(range(len(arch_names)), comp_costs, alpha=0.7, color='orange')
        ax3.set_xlabel('Architecture', fontsize=12)
        ax3.set_ylabel('Computational Cost (relative)', fontsize=12)
        ax3.set_title('Computational Cost\nby Architecture', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(len(arch_names)))
        ax3.set_xticklabels(arch_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Correlation vs latent dimension
        ax4 = axes[1, 0]
        unique_dims = sorted(set(latent_dims))
        dim_correlations = []
        dim_stds = []
        
        for dim in unique_dims:
            dim_corrs = [correlations[i] for i, d in enumerate(latent_dims) if d == dim]
            dim_correlations.append(np.mean(dim_corrs))
            dim_stds.append(np.std(dim_corrs) if len(dim_corrs) > 1 else 0)
        
        ax4.errorbar(unique_dims, dim_correlations, yerr=dim_stds, 
                    marker='o', linewidth=2, markersize=8, capsize=5)
        ax4.set_xlabel('Latent Dimension', fontsize=12)
        ax4.set_ylabel('Order Parameter Correlation', fontsize=12)
        ax4.set_title('Performance vs\nLatent Dimension', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(unique_dims)
        
        # Plot 5: Correlation vs layer count
        ax5 = axes[1, 1]
        unique_layers = sorted(set(layer_counts))
        layer_correlations = []
        layer_stds = []
        
        for layers in unique_layers:
            layer_corrs = [correlations[i] for i, l in enumerate(layer_counts) if l == layers]
            layer_correlations.append(np.mean(layer_corrs))
            layer_stds.append(np.std(layer_corrs) if len(layer_corrs) > 1 else 0)
        
        ax5.errorbar(unique_layers, layer_correlations, yerr=layer_stds,
                    marker='s', linewidth=2, markersize=8, capsize=5, color='red')
        ax5.set_xlabel('Number of Layers', fontsize=12)
        ax5.set_ylabel('Order Parameter Correlation', fontsize=12)
        ax5.set_title('Performance vs\nLayer Depth', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.set_xticks(unique_layers)
        
        # Plot 6: Efficiency scatter plot (correlation vs computational cost)
        ax6 = axes[1, 2]
        
        # Create efficiency score (high correlation, low cost)
        max_cost = max(comp_costs) if comp_costs else 1
        efficiency_scores = [corr / (cost / max_cost + 0.1) for corr, cost in zip(correlations, comp_costs)]
        
        scatter = ax6.scatter(comp_costs, correlations, c=efficiency_scores, 
                            cmap='viridis', s=100, alpha=0.7)
        
        # Add architecture labels
        for i, name in enumerate(arch_names):
            ax6.annotate(name.replace('_', '\n'), 
                        xy=(comp_costs[i], correlations[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        ax6.set_xlabel('Computational Cost (relative)', fontsize=12)
        ax6.set_ylabel('Order Parameter Correlation', fontsize=12)
        ax6.set_title('Efficiency Analysis\n(Color = Efficiency Score)', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label('Efficiency Score', fontsize=10)
        
        plt.suptitle('Architecture Comparison Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def _estimate_critical_temperature_1d(self, 
                                        latent_values: np.ndarray, 
                                        temperatures: np.ndarray) -> float:
        """Estimate critical temperature from 1D latent values using simple clustering."""
        # Sort by temperature
        sorted_indices = np.argsort(temperatures)
        sorted_temps = temperatures[sorted_indices]
        sorted_values = latent_values[sorted_indices]
        
        # Find point of maximum change
        if len(sorted_values) > 10:
            # Calculate moving average derivative
            window_size = len(sorted_values) // 10
            derivatives = []
            temp_points = []
            
            for i in range(window_size, len(sorted_values) - window_size):
                before = np.mean(sorted_values[i-window_size:i])
                after = np.mean(sorted_values[i:i+window_size])
                temp_before = np.mean(sorted_temps[i-window_size:i])
                temp_after = np.mean(sorted_temps[i:i+window_size])
                
                if temp_after != temp_before:
                    derivative = abs(after - before) / (temp_after - temp_before)
                    derivatives.append(derivative)
                    temp_points.append(sorted_temps[i])
            
            if derivatives:
                max_change_idx = np.argmax(derivatives)
                return temp_points[max_change_idx]
        
        # Fallback: return median temperature
        return np.median(temperatures)
    
    def _calculate_physics_consistency(self, 
                                     correlation: float, 
                                     discovered_tc: float, 
                                     theoretical_tc: float) -> float:
        """Calculate overall physics consistency score."""
        # Correlation component (0-1)
        corr_score = min(1.0, abs(correlation))
        
        # Critical temperature accuracy component (0-1)
        tc_error = abs(discovered_tc - theoretical_tc) / theoretical_tc
        tc_score = max(0.0, 1.0 - tc_error / 0.1)  # 10% error gives 0 score
        
        # Combined score (weighted average)
        physics_score = 0.6 * corr_score + 0.4 * tc_score
        
        return physics_score
    
    def generate_comparison_report(self,
                                 vae_result: ComparisonResult,
                                 baseline_results: List[ComparisonResult],
                                 ablation_results: List[AblationResult],
                                 significance_results: Dict[str, Any],
                                 architecture_results: Optional[List[ComparisonResult]] = None,
                                 output_dir: str = 'results/comparison_studies') -> Dict[str, str]:
        """
        Generate comprehensive comparison and ablation study report.
        
        Args:
            vae_result: VAE comparison result
            baseline_results: List of baseline method results
            ablation_results: List of ablation study results
            significance_results: Statistical significance test results
            architecture_results: List of architecture comparison results (optional)
            output_dir: Output directory for results
            
        Returns:
            Dictionary mapping plot names to saved file paths
        """
        self.logger.info("Generating comprehensive comparison study report")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_plots = {}
        
        try:
            # Generate comparison plots
            plots = {}
            
            # Baseline comparison
            if len(baseline_results) >= 1:
                pca_result = baseline_results[0]
                tsne_result = baseline_results[1] if len(baseline_results) > 1 else None
                plots['baseline_comparison'] = self.plot_baseline_comparison(
                    vae_result, pca_result, tsne_result
                )
            
            # Ablation study
            if ablation_results:
                plots['ablation_study'] = self.plot_ablation_study(ablation_results)
            
            # Statistical significance
            if significance_results:
                plots['statistical_significance'] = self.plot_statistical_significance(significance_results)
            
            # Architecture comparison
            if architecture_results:
                plots['architecture_comparison'] = self.plot_architecture_comparison(architecture_results)
            
            # Save all plots
            for plot_name, fig in plots.items():
                file_path = output_path / f"{plot_name}.png"
                fig.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
                saved_plots[plot_name] = str(file_path)
                
                # Also save as PDF for publication
                pdf_path = output_path / f"{plot_name}.pdf"
                fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
                
                plt.close(fig)
                
                self.logger.info(f"Saved comparison study plot: {file_path}")
            
            # Save summary report
            self._save_comparison_summary(
                vae_result, baseline_results, ablation_results, 
                significance_results, architecture_results, output_path
            )
            
        except Exception as e:
            self.logger.error(f"Error generating comparison studies: {e}")
            raise
        
        return saved_plots
    
    def _save_comparison_summary(self,
                               vae_result: ComparisonResult,
                               baseline_results: List[ComparisonResult],
                               ablation_results: List[AblationResult],
                               significance_results: Dict[str, Any],
                               architecture_results: Optional[List[ComparisonResult]],
                               output_path: Path) -> None:
        """Save comparison study summary to text file."""
        summary_path = output_path / "comparison_summary.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Comprehensive Comparison and Ablation Study Summary\n")
            f.write("=" * 60 + "\n\n")
            
            # VAE results
            f.write("VAE Results:\n")
            f.write(f"  Order Parameter Correlation: {vae_result.order_parameter_correlation:.4f}\n")
            f.write(f"  Critical Temperature: {vae_result.critical_temperature:.4f}\n")
            f.write(f"  Physics Consistency Score: {vae_result.physics_consistency_score:.4f}\n\n")
            
            # Baseline comparisons
            f.write("Baseline Method Comparisons:\n")
            for baseline in baseline_results:
                f.write(f"  {baseline.method_name}:\n")
                f.write(f"    Order Parameter Correlation: {baseline.order_parameter_correlation:.4f}\n")
                f.write(f"    Physics Consistency Score: {baseline.physics_consistency_score:.4f}\n")
                f.write(f"    Computational Cost (relative): {baseline.computational_cost:.1f}\n\n")
            
            # Ablation study results
            if ablation_results:
                f.write("Beta-VAE Ablation Study:\n")
                best_corr = max(ablation_results, key=lambda x: x.order_parameter_correlation)
                best_tc = min(ablation_results, key=lambda x: x.critical_temperature_error)
                best_physics = max(ablation_results, key=lambda x: x.physics_consistency_score)
                
                f.write(f"  Best Order Parameter Correlation: β={best_corr.parameter_value:.1f} (r={best_corr.order_parameter_correlation:.4f})\n")
                f.write(f"  Best Critical Temperature Accuracy: β={best_tc.parameter_value:.1f} (error={best_tc.critical_temperature_error:.1f}%)\n")
                f.write(f"  Best Physics Consistency: β={best_physics.parameter_value:.1f} (score={best_physics.physics_consistency_score:.4f})\n\n")
            
            # Architecture comparison results
            if architecture_results:
                f.write("Architecture Comparison Study:\n")
                best_arch_corr = max(architecture_results, key=lambda x: x.order_parameter_correlation)
                best_arch_physics = max(architecture_results, key=lambda x: x.physics_consistency_score)
                most_efficient = min(architecture_results, key=lambda x: x.computational_cost)
                
                f.write(f"  Best Order Parameter Correlation: {best_arch_corr.method_name} (r={best_arch_corr.order_parameter_correlation:.4f})\n")
                f.write(f"  Best Physics Consistency: {best_arch_physics.method_name} (score={best_arch_physics.physics_consistency_score:.4f})\n")
                f.write(f"  Most Efficient: {most_efficient.method_name} (cost={most_efficient.computational_cost:.1f})\n\n")
            
            # Statistical significance
            if significance_results:
                f.write("Statistical Significance Results:\n")
                
                if 'correlation_tests' in significance_results:
                    corr_data = significance_results['correlation_tests']
                    f.write(f"  Order Parameter Correlation vs Zero: p={corr_data['vs_zero']['p_value']:.6f}\n")
                    f.write(f"  Order Parameter Correlation vs Random: p={corr_data['vs_random']['p_value']:.6f}\n")
                
                if 'critical_temperature_tests' in significance_results:
                    tc_data = significance_results['critical_temperature_tests']
                    f.write(f"  Critical Temperature vs Theoretical: p={tc_data['vs_theoretical']['p_value']:.6f}\n")
                    f.write(f"  Critical Temperature Error vs Zero: p={tc_data['error_vs_zero']['p_value']:.6f}\n")
        
        self.logger.info(f"Comparison study summary saved: {summary_path}")