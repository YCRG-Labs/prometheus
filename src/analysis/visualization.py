"""
Visualization and Reporting System

This module provides publication-ready plotting functions for latent space analysis,
order parameter comparisons, phase diagrams, and comprehensive analysis reports.
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
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import warnings

from .latent_analysis import LatentRepresentation
from .phase_detection import PhaseDetectionResult, ClusteringResult
from .order_parameter_discovery import OrderParameterCandidate
from .physics_validation import ValidationMetrics, PhysicsValidator
from ..utils.logging_utils import get_logger


# Set publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Publication settings
PUBLICATION_SETTINGS = {
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
}


class PublicationVisualizer:
    """
    Publication-ready visualization system for physics analysis results.
    
    Creates high-quality figures suitable for scientific publications with
    proper formatting, colormaps, and annotations.
    """
    
    def __init__(self, style: str = 'publication'):
        """
        Initialize visualizer with specified style.
        
        Args:
            style: Visualization style ('publication', 'presentation', 'notebook')
        """
        self.style = style
        self.logger = get_logger(__name__)
        
        # Apply style settings
        if style == 'publication':
            plt.rcParams.update(PUBLICATION_SETTINGS)
        
        self.logger.info(f"Publication visualizer initialized with {style} style")
    
    def plot_latent_space_analysis(self,
                                  latent_repr: LatentRepresentation,
                                  order_param_candidates: List[OrderParameterCandidate],
                                  figsize: Tuple[int, int] = (15, 10)) -> Figure:
        """
        Create comprehensive latent space analysis visualization.
        
        Args:
            latent_repr: Latent space representation
            order_param_candidates: List of order parameter candidates
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure with latent space analysis
        """
        self.logger.info("Creating latent space analysis visualization")
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Latent space colored by temperature
        ax1 = fig.add_subplot(gs[0, 0])
        scatter1 = ax1.scatter(
            latent_repr.z1, latent_repr.z2,
            c=latent_repr.temperatures,
            cmap='coolwarm',
            alpha=0.6,
            s=20,
            edgecolors='none'
        )
        
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Temperature', fontsize=12)
        
        ax1.set_xlabel('Latent Dimension 1 (z₁)', fontsize=12)
        ax1.set_ylabel('Latent Dimension 2 (z₂)', fontsize=12)
        ax1.set_title('Latent Space\n(Temperature)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        
        # Plot 2: Latent space colored by magnetization
        ax2 = fig.add_subplot(gs[0, 1])
        scatter2 = ax2.scatter(
            latent_repr.z1, latent_repr.z2,
            c=np.abs(latent_repr.magnetizations),
            cmap='viridis',
            alpha=0.6,
            s=20,
            edgecolors='none'
        )
        
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('|Magnetization|', fontsize=12)
        
        ax2.set_xlabel('Latent Dimension 1 (z₁)', fontsize=12)
        ax2.set_ylabel('Latent Dimension 2 (z₂)', fontsize=12)
        ax2.set_title('Latent Space\n(Magnetization)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')
        
        # Plot 3: Order parameter correlation
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Get primary order parameter (best candidate)
        if not order_param_candidates:
            raise ValueError("No order parameter candidates provided")
        
        best_candidate = order_param_candidates[0]
        if best_candidate.latent_dimension == 'z1':
            primary_latent = latent_repr.z1
            primary_label = 'z₁'
        else:
            primary_latent = latent_repr.z2
            primary_label = 'z₂'
        
        ax3.scatter(
            np.abs(latent_repr.magnetizations),
            primary_latent,
            alpha=0.6,
            s=20,
            c=latent_repr.temperatures,
            cmap='coolwarm',
            edgecolors='none'
        )
        
        # Add correlation line
        correlation = best_candidate.correlation_with_magnetization.correlation_coefficient
        ax3.text(0.05, 0.95, f'r = {correlation:.3f}', 
                transform=ax3.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax3.set_xlabel('|Magnetization|', fontsize=12)
        ax3.set_ylabel(f'Primary Order Parameter ({primary_label})', fontsize=12)
        ax3.set_title('Order Parameter\nCorrelation', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Temperature dependence of latent dimensions
        ax4 = fig.add_subplot(gs[1, :2])
        
        # Bin data by temperature for cleaner visualization
        unique_temps = np.unique(latent_repr.temperatures)
        z1_means = []
        z2_means = []
        z1_stds = []
        z2_stds = []
        
        for temp in unique_temps:
            temp_mask = latent_repr.temperatures == temp
            z1_temp = latent_repr.z1[temp_mask]
            z2_temp = latent_repr.z2[temp_mask]
            
            z1_means.append(np.mean(z1_temp))
            z2_means.append(np.mean(z2_temp))
            z1_stds.append(np.std(z1_temp))
            z2_stds.append(np.std(z2_temp))
        
        z1_means = np.array(z1_means)
        z2_means = np.array(z2_means)
        z1_stds = np.array(z1_stds)
        z2_stds = np.array(z2_stds)
        
        # Plot with error bars
        ax4.errorbar(unique_temps, z1_means, yerr=z1_stds, 
                    label='z₁', marker='o', linewidth=2, capsize=3)
        ax4.errorbar(unique_temps, z2_means, yerr=z2_stds, 
                    label='z₂', marker='s', linewidth=2, capsize=3)
        
        ax4.set_xlabel('Temperature', fontsize=12)
        ax4.set_ylabel('Latent Variable Value', fontsize=12)
        ax4.set_title('Temperature Dependence of Latent Variables', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Reconstruction error analysis
        ax5 = fig.add_subplot(gs[1, 2])
        
        # Bin reconstruction errors by temperature
        recon_means = []
        recon_stds = []
        
        for temp in unique_temps:
            temp_mask = latent_repr.temperatures == temp
            recon_temp = latent_repr.reconstruction_errors[temp_mask]
            recon_means.append(np.mean(recon_temp))
            recon_stds.append(np.std(recon_temp))
        
        ax5.errorbar(unique_temps, recon_means, yerr=recon_stds,
                    marker='o', linewidth=2, capsize=3, color='red')
        
        ax5.set_xlabel('Temperature', fontsize=12)
        ax5.set_ylabel('Reconstruction Error', fontsize=12)
        ax5.set_title('Reconstruction Quality\nvs Temperature', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        plt.suptitle('Latent Space Analysis Summary', fontsize=16, fontweight='bold', y=0.98)
        
        return fig
    
    def plot_order_parameter_comparison(self,
                                      latent_repr: LatentRepresentation,
                                      order_param_candidates: List[OrderParameterCandidate],
                                      theoretical_tc: float = 2.269,
                                      figsize: Tuple[int, int] = (15, 5)) -> Figure:
        """
        Create order parameter vs temperature comparison with theoretical expectations.
        
        Args:
            latent_repr: Latent space representation
            order_param_candidates: List of order parameter candidates
            theoretical_tc: Theoretical critical temperature
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure with order parameter comparison
        """
        self.logger.info("Creating order parameter comparison visualization")
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Get primary order parameter (best candidate)
        if not order_param_candidates:
            raise ValueError("No order parameter candidates provided")
        
        best_candidate = order_param_candidates[0]
        if best_candidate.latent_dimension == 'z1':
            discovered_order_param = latent_repr.z1
            primary_label = 'z₁'
        else:
            discovered_order_param = latent_repr.z2
            primary_label = 'z₂'
        
        theoretical_order_param = np.abs(latent_repr.magnetizations)
        
        # Plot 1: Discovered order parameter vs temperature
        ax1 = axes[0]
        
        # Bin data for cleaner visualization
        unique_temps = np.unique(latent_repr.temperatures)
        discovered_means = []
        discovered_stds = []
        
        for temp in unique_temps:
            temp_mask = latent_repr.temperatures == temp
            param_temp = discovered_order_param[temp_mask]
            discovered_means.append(np.mean(param_temp))
            discovered_stds.append(np.std(param_temp))
        
        ax1.errorbar(unique_temps, discovered_means, yerr=discovered_stds,
                    marker='o', linewidth=2, capsize=3, label='Discovered', color='blue')
        
        ax1.axvline(theoretical_tc, color='red', linestyle='--', alpha=0.7, 
                   label=f'T_c = {theoretical_tc:.3f}')
        
        ax1.set_xlabel('Temperature', fontsize=12)
        ax1.set_ylabel(f'Discovered Order Parameter ({primary_label})', fontsize=12)
        ax1.set_title('AI-Discovered Order Parameter', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Theoretical order parameter vs temperature
        ax2 = axes[1]
        
        theoretical_means = []
        theoretical_stds = []
        
        for temp in unique_temps:
            temp_mask = latent_repr.temperatures == temp
            param_temp = theoretical_order_param[temp_mask]
            theoretical_means.append(np.mean(param_temp))
            theoretical_stds.append(np.std(param_temp))
        
        ax2.errorbar(unique_temps, theoretical_means, yerr=theoretical_stds,
                    marker='s', linewidth=2, capsize=3, label='Theoretical', color='green')
        
        ax2.axvline(theoretical_tc, color='red', linestyle='--', alpha=0.7,
                   label=f'T_c = {theoretical_tc:.3f}')
        
        ax2.set_xlabel('Temperature', fontsize=12)
        ax2.set_ylabel('Theoretical Order Parameter (|M|)', fontsize=12)
        ax2.set_title('Theoretical Magnetization', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Direct comparison
        ax3 = axes[2]
        
        # Scatter plot with temperature coloring
        scatter = ax3.scatter(
            theoretical_order_param,
            discovered_order_param,
            c=latent_repr.temperatures,
            cmap='coolwarm',
            alpha=0.6,
            s=20,
            edgecolors='none'
        )
        
        # Add perfect correlation line
        param_min = min(np.min(theoretical_order_param), np.min(discovered_order_param))
        param_max = max(np.max(theoretical_order_param), np.max(discovered_order_param))
        ax3.plot([param_min, param_max], [param_min, param_max], 
                'k--', alpha=0.5, label='Perfect Correlation')
        
        # Add correlation coefficient
        correlation = best_candidate.correlation_with_magnetization.correlation_coefficient
        ax3.text(0.05, 0.95, f'r = {correlation:.3f}', 
                transform=ax3.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Temperature', fontsize=12)
        
        ax3.set_xlabel('Theoretical Order Parameter (|M|)', fontsize=12)
        ax3.set_ylabel(f'Discovered Order Parameter ({primary_label})', fontsize=12)
        ax3.set_title('Order Parameter Correlation', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Order Parameter Discovery Validation', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_phase_diagram(self,
                          latent_repr: LatentRepresentation,
                          phase_detection_result: PhaseDetectionResult,
                          clustering_result: Optional[ClusteringResult] = None,
                          theoretical_tc: float = 2.269,
                          figsize: Tuple[int, int] = (15, 10)) -> Figure:
        """
        Create comprehensive phase diagram with critical point identification.
        
        Args:
            latent_repr: Latent space representation
            phase_detection_result: Phase detection results
            clustering_result: Optional clustering results
            theoretical_tc: Theoretical critical temperature
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure with phase diagram
        """
        self.logger.info("Creating phase diagram visualization")
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        discovered_tc = phase_detection_result.critical_temperature
        transition_region = phase_detection_result.transition_region
        
        # Plot 1: Phase regions in latent space
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Define phase regions
        low_temp_mask = latent_repr.temperatures < transition_region[0]
        high_temp_mask = latent_repr.temperatures > transition_region[1]
        critical_mask = ~(low_temp_mask | high_temp_mask)
        
        ax1.scatter(latent_repr.z1[low_temp_mask], latent_repr.z2[low_temp_mask],
                   c='blue', alpha=0.6, s=20, label='Ordered Phase')
        ax1.scatter(latent_repr.z1[high_temp_mask], latent_repr.z2[high_temp_mask],
                   c='red', alpha=0.6, s=20, label='Disordered Phase')
        ax1.scatter(latent_repr.z1[critical_mask], latent_repr.z2[critical_mask],
                   c='orange', alpha=0.8, s=25, label='Critical Region')
        
        ax1.set_xlabel('Latent Dimension 1 (z₁)', fontsize=12)
        ax1.set_ylabel('Latent Dimension 2 (z₂)', fontsize=12)
        ax1.set_title('Phase Regions in Latent Space', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        
        # Plot 2: Temperature distribution with phase boundaries
        ax2 = fig.add_subplot(gs[0, 1])
        
        scatter = ax2.scatter(latent_repr.z1, latent_repr.z2,
                            c=latent_repr.temperatures, cmap='coolwarm',
                            alpha=0.6, s=20)
        
        # Add temperature contours if possible
        try:
            # Create grid for interpolation
            z1_grid = np.linspace(latent_repr.z1.min(), latent_repr.z1.max(), 50)
            z2_grid = np.linspace(latent_repr.z2.min(), latent_repr.z2.max(), 50)
            Z1_grid, Z2_grid = np.meshgrid(z1_grid, z2_grid)
            
            # Interpolate temperature
            temp_grid = griddata(
                (latent_repr.z1, latent_repr.z2),
                latent_repr.temperatures,
                (Z1_grid, Z2_grid),
                method='linear'
            )
            
            # Add critical temperature contours
            contour_levels = [discovered_tc, theoretical_tc]
            contour_colors = ['black', 'red']
            contour_styles = ['-', '--']
            
            for level, color, style in zip(contour_levels, contour_colors, contour_styles):
                contour = ax2.contour(Z1_grid, Z2_grid, temp_grid, 
                                    levels=[level], colors=[color], 
                                    linewidths=2, linestyles=[style])
                ax2.clabel(contour, inline=True, fontsize=10, 
                          fmt=f'T = {level:.3f}')
            
        except Exception as e:
            self.logger.warning(f"Could not create temperature contours: {e}")
        
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Temperature', fontsize=12)
        
        ax2.set_xlabel('Latent Dimension 1 (z₁)', fontsize=12)
        ax2.set_ylabel('Latent Dimension 2 (z₂)', fontsize=12)
        ax2.set_title('Temperature Distribution\nwith Critical Points', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')
        
        # Plot 3: Clustering results (if available)
        ax3 = fig.add_subplot(gs[0, 2])
        
        if clustering_result is not None:
            unique_clusters = np.unique(clustering_result.cluster_labels)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))
            
            for i, cluster_id in enumerate(unique_clusters):
                cluster_mask = clustering_result.cluster_labels == cluster_id
                ax3.scatter(latent_repr.z1[cluster_mask], latent_repr.z2[cluster_mask],
                           c=[colors[i]], alpha=0.6, s=20, label=f'Cluster {cluster_id}')
            
            # Plot cluster centers
            ax3.scatter(clustering_result.cluster_centers[:, 0],
                       clustering_result.cluster_centers[:, 1],
                       c='black', marker='x', s=100, linewidths=3, label='Centers')
            
            ax3.set_title('Cluster-Based\nPhase Detection', fontsize=14, fontweight='bold')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No Clustering\nResults Available', 
                    transform=ax3.transAxes, ha='center', va='center',
                    fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgray'))
            ax3.set_title('Clustering Analysis', fontsize=14, fontweight='bold')
        
        ax3.set_xlabel('Latent Dimension 1 (z₁)', fontsize=12)
        ax3.set_ylabel('Latent Dimension 2 (z₂)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal', adjustable='box')
        
        # Plot 4: Critical temperature comparison
        ax4 = fig.add_subplot(gs[1, :2])
        
        # Create temperature histogram with phase regions
        temp_bins = np.linspace(latent_repr.temperatures.min(), 
                               latent_repr.temperatures.max(), 50)
        
        ax4.hist(latent_repr.temperatures, bins=temp_bins, alpha=0.7, 
                density=True, color='lightblue', edgecolor='black')
        
        # Mark critical temperatures
        ax4.axvline(discovered_tc, color='black', linewidth=3, 
                   label=f'Discovered T_c = {discovered_tc:.3f}')
        ax4.axvline(theoretical_tc, color='red', linewidth=3, linestyle='--',
                   label=f'Theoretical T_c = {theoretical_tc:.3f}')
        
        # Mark transition region
        ax4.axvspan(transition_region[0], transition_region[1], 
                   alpha=0.3, color='orange', label='Transition Region')
        
        ax4.set_xlabel('Temperature', fontsize=12)
        ax4.set_ylabel('Density', fontsize=12)
        ax4.set_title('Critical Temperature Detection', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Phase transition sharpness
        ax5 = fig.add_subplot(gs[1, 2])
        
        # Calculate order parameter vs temperature
        unique_temps = np.unique(latent_repr.temperatures)
        magnetization_means = []
        
        for temp in unique_temps:
            temp_mask = latent_repr.temperatures == temp
            mag_temp = np.abs(latent_repr.magnetizations[temp_mask])
            magnetization_means.append(np.mean(mag_temp))
        
        ax5.plot(unique_temps, magnetization_means, 'bo-', linewidth=2, 
                label='|Magnetization|')
        
        ax5.axvline(discovered_tc, color='black', linewidth=2, 
                   label=f'Discovered T_c')
        ax5.axvline(theoretical_tc, color='red', linewidth=2, linestyle='--',
                   label=f'Theoretical T_c')
        
        ax5.set_xlabel('Temperature', fontsize=12)
        ax5.set_ylabel('|Magnetization|', fontsize=12)
        ax5.set_title('Phase Transition\nSharpness', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        plt.suptitle('Phase Diagram Analysis', fontsize=16, fontweight='bold', y=0.98)
        
        return fig
    
    def plot_validation_summary(self,
                              validation_metrics: ValidationMetrics,
                              figsize: Tuple[int, int] = (12, 8)) -> Figure:
        """
        Create comprehensive validation summary visualization.
        
        Args:
            validation_metrics: Physics validation results
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure with validation summary
        """
        self.logger.info("Creating validation summary visualization")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Validation scores radar chart
        ax1 = axes[0, 0]
        
        categories = ['Order Parameter\nCorrelation', 'Critical Temperature\nAccuracy', 
                     'Energy\nConservation', 'Magnetization\nConservation']
        
        values = [
            abs(validation_metrics.order_parameter_correlation),
            1.0 - (validation_metrics.critical_temperature_relative_error / 100.0),
            validation_metrics.energy_conservation_score,
            validation_metrics.magnetization_conservation_score
        ]
        
        # Ensure values are in [0, 1] range
        values = [max(0, min(1, v)) for v in values]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax1 = plt.subplot(2, 2, 1, projection='polar')
        ax1.plot(angles, values, 'o-', linewidth=2, color='blue')
        ax1.fill(angles, values, alpha=0.25, color='blue')
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories, fontsize=10)
        ax1.set_ylim(0, 1)
        ax1.set_title('Validation Scores', fontsize=14, fontweight='bold', pad=20)
        ax1.grid(True)
        
        # Plot 2: Critical temperature comparison
        ax2 = axes[0, 1]
        
        theoretical_tc = validation_metrics.theoretical_comparison['onsager_critical_temperature']
        discovered_tc = validation_metrics.theoretical_comparison['discovered_critical_temperature']
        error = validation_metrics.critical_temperature_error
        
        bars = ax2.bar(['Theoretical\n(Onsager)', 'Discovered\n(AI)'], 
                      [theoretical_tc, discovered_tc],
                      color=['red', 'blue'], alpha=0.7)
        
        # Add error annotation
        ax2.annotate(f'Error: {error:.4f}\n({validation_metrics.critical_temperature_relative_error:.1f}%)',
                    xy=(1, discovered_tc), xytext=(1.2, discovered_tc + 0.1),
                    arrowprops=dict(arrowstyle='->', color='black'),
                    fontsize=10, ha='left')
        
        ax2.set_ylabel('Critical Temperature', fontsize=12)
        ax2.set_title('Critical Temperature Comparison', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Conservation scores
        ax3 = axes[1, 0]
        
        conservation_scores = [
            validation_metrics.energy_conservation_score,
            validation_metrics.magnetization_conservation_score
        ]
        conservation_labels = ['Energy', 'Magnetization']
        
        bars = ax3.bar(conservation_labels, conservation_scores, 
                      color=['orange', 'green'], alpha=0.7)
        
        # Add score annotations
        for bar, score in zip(bars, conservation_scores):
            height = bar.get_height()
            ax3.annotate(f'{score:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        ax3.set_ylabel('Conservation Score', fontsize=12)
        ax3.set_title('Conservation Law Validation', fontsize=14, fontweight='bold')
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Overall physics consistency
        ax4 = axes[1, 1]
        
        overall_score = validation_metrics.physics_consistency_score
        
        # Create gauge-like visualization
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        ax4 = plt.subplot(2, 2, 4, projection='polar')
        ax4.plot(theta, r, 'k-', linewidth=3)
        ax4.fill_between(theta, 0, r, alpha=0.1, color='gray')
        
        # Color regions
        if overall_score >= 0.8:
            color = 'green'
            status = 'EXCELLENT'
        elif overall_score >= 0.6:
            color = 'orange'
            status = 'GOOD'
        elif overall_score >= 0.4:
            color = 'yellow'
            status = 'FAIR'
        else:
            color = 'red'
            status = 'POOR'
        
        # Fill up to score level
        score_theta = np.linspace(0, np.pi * overall_score, 50)
        score_r = np.ones_like(score_theta)
        ax4.fill_between(score_theta, 0, score_r, alpha=0.7, color=color)
        
        # Add score text
        ax4.text(np.pi/2, 0.5, f'{overall_score:.3f}\n{status}', 
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax4.set_ylim(0, 1)
        ax4.set_theta_zero_location('W')
        ax4.set_theta_direction(1)
        ax4.set_thetagrids([0, 45, 90, 135, 180], 
                          ['0.0', '0.25', '0.5', '0.75', '1.0'])
        ax4.set_title('Overall Physics\nConsistency Score', 
                     fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle('Physics Validation Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def save_figure(self, 
                   fig: Figure, 
                   filename: str, 
                   output_dir: str = 'results/figures',
                   formats: List[str] = ['png', 'pdf']) -> List[str]:
        """
        Save figure in multiple formats with publication settings.
        
        Args:
            fig: Matplotlib figure to save
            filename: Base filename (without extension)
            output_dir: Output directory
            formats: List of formats to save ('png', 'pdf', 'svg', 'eps')
            
        Returns:
            List of saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        for fmt in formats:
            file_path = output_path / f"{filename}.{fmt}"
            
            # Format-specific settings
            save_kwargs = {'bbox_inches': 'tight', 'pad_inches': 0.1}
            
            if fmt == 'png':
                save_kwargs['dpi'] = 300
            elif fmt == 'pdf':
                save_kwargs['dpi'] = 300
                save_kwargs['backend'] = 'pdf'
            
            fig.savefig(file_path, format=fmt, **save_kwargs)
            saved_files.append(str(file_path))
            
            self.logger.info(f"Figure saved: {file_path}")
        
        return saved_files


class AnalysisReporter:
    """
    Comprehensive analysis reporting system that combines all validation
    metrics and visualizations into publication-ready reports.
    """
    
    def __init__(self, output_dir: str = 'results/reports'):
        """
        Initialize analysis reporter.
        
        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
        
        self.visualizer = PublicationVisualizer()
        
        self.logger.info(f"Analysis reporter initialized: {output_dir}")
    
    def generate_comprehensive_report(self,
                                    latent_repr: LatentRepresentation,
                                    order_param_candidates: List[OrderParameterCandidate],
                                    phase_detection_result: PhaseDetectionResult,
                                    validation_metrics: ValidationMetrics,
                                    clustering_result: Optional[ClusteringResult] = None,
                                    report_name: str = 'prometheus_analysis_report') -> Dict[str, Any]:
        """
        Generate comprehensive analysis report with all visualizations and metrics.
        
        Args:
            latent_repr: Latent space representation
            order_param_candidates: List of order parameter candidates
            phase_detection_result: Phase detection results
            validation_metrics: Physics validation metrics
            clustering_result: Optional clustering results
            report_name: Base name for report files
            
        Returns:
            Dictionary with report metadata and file paths
        """
        self.logger.info(f"Generating comprehensive analysis report: {report_name}")
        
        report_dir = self.output_dir / report_name
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate all visualizations
        figures = {}
        
        # 1. Latent space analysis
        fig_latent = self.visualizer.plot_latent_space_analysis(
            latent_repr, order_param_candidates
        )
        figures['latent_analysis'] = self.visualizer.save_figure(
            fig_latent, 'latent_space_analysis', str(report_dir)
        )
        plt.close(fig_latent)
        
        # 2. Order parameter comparison
        fig_order = self.visualizer.plot_order_parameter_comparison(
            latent_repr, order_param_candidates
        )
        figures['order_parameter'] = self.visualizer.save_figure(
            fig_order, 'order_parameter_comparison', str(report_dir)
        )
        plt.close(fig_order)
        
        # 3. Phase diagram
        fig_phase = self.visualizer.plot_phase_diagram(
            latent_repr, phase_detection_result, clustering_result
        )
        figures['phase_diagram'] = self.visualizer.save_figure(
            fig_phase, 'phase_diagram', str(report_dir)
        )
        plt.close(fig_phase)
        
        # 4. Validation summary
        fig_validation = self.visualizer.plot_validation_summary(validation_metrics)
        figures['validation_summary'] = self.visualizer.save_figure(
            fig_validation, 'validation_summary', str(report_dir)
        )
        plt.close(fig_validation)
        
        # Generate text report
        validator = PhysicsValidator()
        text_report = validator.generate_validation_report(
            validation_metrics, str(report_dir / 'validation_report.txt')
        )
        
        # Generate summary statistics
        summary_stats = self._generate_summary_statistics(
            latent_repr, order_param_candidates, phase_detection_result, validation_metrics
        )
        
        # Save summary as JSON
        import json
        with open(report_dir / 'summary_statistics.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Create HTML report (optional)
        html_report_path = self._generate_html_report(
            report_dir, figures, summary_stats, text_report
        )
        
        report_metadata = {
            'report_name': report_name,
            'report_directory': str(report_dir),
            'figures': figures,
            'text_report': str(report_dir / 'validation_report.txt'),
            'summary_statistics': str(report_dir / 'summary_statistics.json'),
            'html_report': html_report_path,
            'generation_timestamp': str(np.datetime64('now')),
            'summary_stats': summary_stats
        }
        
        self.logger.info(f"Comprehensive report generated: {report_dir}")
        
        return report_metadata
    
    def _generate_summary_statistics(self,
                                   latent_repr: LatentRepresentation,
                                   order_param_candidates: List[OrderParameterCandidate],
                                   phase_detection_result: PhaseDetectionResult,
                                   validation_metrics: ValidationMetrics) -> Dict[str, Any]:
        """Generate summary statistics for the analysis."""
        return {
            'dataset_info': {
                'n_samples': int(latent_repr.n_samples),
                'temperature_range': [
                    float(np.min(latent_repr.temperatures)),
                    float(np.max(latent_repr.temperatures))
                ],
                'latent_space_statistics': latent_repr.get_statistics()
            },
            'order_parameter_discovery': {
                'primary_dimension': order_param_candidates[0].latent_dimension if order_param_candidates else 'unknown',
                'correlation_with_magnetization': float(
                    order_param_candidates[0].correlation_with_magnetization.correlation_coefficient
                ) if order_param_candidates else 0.0,
                'statistical_significance': float(
                    order_param_candidates[0].correlation_with_magnetization.p_value
                ) if order_param_candidates else 1.0,
                'discovery_confidence': float(
                    order_param_candidates[0].confidence_score
                ) if order_param_candidates else 0.0
            },
            'phase_transition_detection': {
                'discovered_critical_temperature': float(phase_detection_result.critical_temperature),
                'theoretical_critical_temperature': 2.269,
                'detection_method': phase_detection_result.method,
                'transition_region': [
                    float(phase_detection_result.transition_region[0]),
                    float(phase_detection_result.transition_region[1])
                ],
                'detection_confidence': float(phase_detection_result.confidence)
            },
            'physics_validation': {
                'overall_consistency_score': float(validation_metrics.physics_consistency_score),
                'order_parameter_correlation': float(validation_metrics.order_parameter_correlation),
                'critical_temperature_error_percent': float(validation_metrics.critical_temperature_relative_error),
                'energy_conservation_score': float(validation_metrics.energy_conservation_score),
                'magnetization_conservation_score': float(validation_metrics.magnetization_conservation_score),
                'validation_status': 'PASS' if validation_metrics.physics_consistency_score >= 0.6 else 'FAIL'
            }
        }
    
    def _generate_html_report(self,
                            report_dir: Path,
                            figures: Dict[str, List[str]],
                            summary_stats: Dict[str, Any],
                            text_report: str) -> str:
        """Generate HTML report with embedded figures and statistics."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prometheus Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; margin-bottom: 40px; }}
                .section {{ margin-bottom: 30px; }}
                .figure {{ text-align: center; margin: 20px 0; }}
                .stats {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; }}
                .pass {{ color: green; font-weight: bold; }}
                .fail {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Prometheus: AI-Driven Physics Discovery Report</h1>
                <p>Unsupervised Discovery of Order Parameters and Phase Transitions in the 2D Ising Model</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="stats">
                    <div class="metric">Overall Physics Consistency Score: <strong>{summary_stats['physics_validation']['overall_consistency_score']:.3f}</strong></div>
                    <div class="metric">Order Parameter Correlation: <strong>{summary_stats['physics_validation']['order_parameter_correlation']:.3f}</strong></div>
                    <div class="metric">Critical Temperature Error: <strong>{summary_stats['physics_validation']['critical_temperature_error_percent']:.1f}%</strong></div>
                    <div class="metric">Validation Status: <span class="{'pass' if summary_stats['physics_validation']['validation_status'] == 'PASS' else 'fail'}">{summary_stats['physics_validation']['validation_status']}</span></div>
                </div>
            </div>
            
            <div class="section">
                <h2>Latent Space Analysis</h2>
                <div class="figure">
                    <img src="latent_space_analysis.png" alt="Latent Space Analysis" style="max-width: 100%;">
                </div>
            </div>
            
            <div class="section">
                <h2>Order Parameter Discovery</h2>
                <div class="figure">
                    <img src="order_parameter_comparison.png" alt="Order Parameter Comparison" style="max-width: 100%;">
                </div>
            </div>
            
            <div class="section">
                <h2>Phase Diagram</h2>
                <div class="figure">
                    <img src="phase_diagram.png" alt="Phase Diagram" style="max-width: 100%;">
                </div>
            </div>
            
            <div class="section">
                <h2>Validation Summary</h2>
                <div class="figure">
                    <img src="validation_summary.png" alt="Validation Summary" style="max-width: 100%;">
                </div>
            </div>
            
            <div class="section">
                <h2>Detailed Validation Report</h2>
                <pre style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; white-space: pre-wrap;">{text_report}</pre>
            </div>
        </body>
        </html>
        """
        
        html_path = report_dir / 'analysis_report.html'
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return str(html_path)