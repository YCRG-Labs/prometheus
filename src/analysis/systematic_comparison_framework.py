"""
Systematic Comparison Framework for Publication Materials

This module implements a comprehensive comparison framework for generating
publication-ready comparative analysis between 2D and 3D systems, order
parameter discovery, and critical temperature accuracy across all physics systems.
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
import json
from datetime import datetime

from ..utils.logging_utils import get_logger


@dataclass
class SystemComparisonData:
    """Container for system comparison data."""
    system_name: str
    dimensionality: str  # "2D" or "3D"
    model_type: str  # "Ising", "Potts", "XY"
    
    # Latent space data
    latent_z1: np.ndarray
    latent_z2: np.ndarray
    temperatures: np.ndarray
    magnetizations: np.ndarray
    
    # Physics results
    discovered_tc: float
    theoretical_tc: float
    tc_accuracy_percent: float
    
    # Order parameter analysis
    order_parameter_correlation: float
    best_latent_dimension: str  # "z1" or "z2"
    
    # Critical exponents (if available)
    beta_exponent: Optional[float] = None
    nu_exponent: Optional[float] = None
    beta_theoretical: Optional[float] = None
    nu_theoretical: Optional[float] = None
    
    # Additional metrics
    physics_consistency_score: float = 0.0
    data_quality_score: float = 0.0


@dataclass
class ComparisonResults:
    """Container for comparison analysis results."""
    systems_compared: List[str]
    
    # Phase separation analysis
    phase_separation_quality: Dict[str, float]
    latent_space_clustering: Dict[str, float]
    
    # Order parameter comparison
    order_parameter_accuracy: Dict[str, float]
    correlation_comparison: Dict[str, float]
    
    # Critical temperature comparison
    tc_detection_accuracy: Dict[str, float]
    tc_error_statistics: Dict[str, Dict[str, float]]
    
    # Cross-system validation
    universality_validation: Dict[str, bool]
    method_consistency: Dict[str, float]


class SystematicComparisonFramework:
    """
    Comprehensive systematic comparison framework for publication materials.
    
    Provides systematic comparison across:
    - 2D vs 3D phase separation in latent space
    - Order parameter discovery accuracy
    - Critical temperature detection across all systems
    - Cross-system validation and universality
    """
    
    def __init__(self):
        """Initialize systematic comparison framework."""
        self.logger = get_logger(__name__)
        
        # Publication settings
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = {
            '2D_Ising': '#1f77b4',
            '3D_Ising': '#ff7f0e', 
            'Potts': '#2ca02c',
            'XY': '#d62728'
        }
        
    def load_system_data(self, 
                        system_results: Dict[str, Dict[str, Any]]) -> List[SystemComparisonData]:
        """
        Load and organize system data for comparison.
        
        Args:
            system_results: Dictionary mapping system names to their results
            
        Returns:
            List of SystemComparisonData objects
        """
        self.logger.info("Loading system data for comparison framework")
        
        comparison_data = []
        
        for system_name, results in system_results.items():
            try:
                # Parse system information
                if '2d' in system_name.lower() or '2D' in system_name:
                    dimensionality = "2D"
                elif '3d' in system_name.lower() or '3D' in system_name:
                    dimensionality = "3D"
                else:
                    dimensionality = "Unknown"
                
                if 'ising' in system_name.lower():
                    model_type = "Ising"
                elif 'potts' in system_name.lower():
                    model_type = "Potts"
                elif 'xy' in system_name.lower():
                    model_type = "XY"
                else:
                    model_type = "Unknown"
                
                # Extract required data
                latent_data = results.get('latent_representation', {})
                physics_data = results.get('physics_results', {})
                validation_data = results.get('validation_metrics', {})
                
                system_data = SystemComparisonData(
                    system_name=system_name,
                    dimensionality=dimensionality,
                    model_type=model_type,
                    latent_z1=latent_data.get('z1', np.array([])),
                    latent_z2=latent_data.get('z2', np.array([])),
                    temperatures=results.get('temperatures', np.array([])),
                    magnetizations=results.get('magnetizations', np.array([])),
                    discovered_tc=physics_data.get('critical_temperature', 0.0),
                    theoretical_tc=physics_data.get('theoretical_tc', 0.0),
                    tc_accuracy_percent=physics_data.get('tc_accuracy_percent', 0.0),
                    order_parameter_correlation=physics_data.get('order_parameter_correlation', 0.0),
                    best_latent_dimension=physics_data.get('best_latent_dimension', 'z1'),
                    beta_exponent=physics_data.get('beta_exponent'),
                    nu_exponent=physics_data.get('nu_exponent'),
                    beta_theoretical=physics_data.get('beta_theoretical'),
                    nu_theoretical=physics_data.get('nu_theoretical'),
                    physics_consistency_score=validation_data.get('physics_consistency_score', 0.0),
                    data_quality_score=validation_data.get('data_quality_score', 0.0)
                )
                
                comparison_data.append(system_data)
                
            except Exception as e:
                self.logger.warning(f"Error loading data for system {system_name}: {e}")
                continue
        
        self.logger.info(f"Loaded {len(comparison_data)} systems for comparison")
        return comparison_data
    
    def generate_2d_vs_3d_phase_separation_plots(self,
                                                comparison_data: List[SystemComparisonData],
                                                figsize: Tuple[int, int] = (16, 12)) -> Figure:
        """
        Generate 2D vs 3D phase separation comparison plots in latent space.
        
        Args:
            comparison_data: List of system comparison data
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure with 2D vs 3D phase separation analysis
        """
        self.logger.info("Generating 2D vs 3D phase separation comparison plots")
        
        # Separate 2D and 3D systems
        systems_2d = [s for s in comparison_data if s.dimensionality == "2D"]
        systems_3d = [s for s in comparison_data if s.dimensionality == "3D"]
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: 2D Ising latent space
        if systems_2d:
            ax1 = axes[0, 0]
            system_2d = systems_2d[0]  # Assume first is Ising
            
            scatter = ax1.scatter(
                system_2d.latent_z1, system_2d.latent_z2,
                c=system_2d.temperatures, cmap='coolwarm',
                alpha=0.6, s=30, edgecolors='black', linewidth=0.5
            )
            
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Temperature', fontsize=10)
            
            ax1.set_xlabel('z₁', fontsize=12)
            ax1.set_ylabel('z₂', fontsize=12)
            ax1.set_title('2D Ising Model\nLatent Space (Temperature)', 
                         fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal', adjustable='box')
            
            # Add critical temperature annotation
            ax1.text(0.05, 0.95, f'Tc = {system_2d.discovered_tc:.3f}\n(Theory: {system_2d.theoretical_tc:.3f})',
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: 3D Ising latent space
        if systems_3d:
            ax2 = axes[0, 1]
            system_3d = systems_3d[0]  # Assume first is Ising
            
            scatter = ax2.scatter(
                system_3d.latent_z1, system_3d.latent_z2,
                c=system_3d.temperatures, cmap='coolwarm',
                alpha=0.6, s=30, edgecolors='black', linewidth=0.5
            )
            
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Temperature', fontsize=10)
            
            ax2.set_xlabel('z₁', fontsize=12)
            ax2.set_ylabel('z₂', fontsize=12)
            ax2.set_title('3D Ising Model\nLatent Space (Temperature)', 
                         fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal', adjustable='box')
            
            # Add critical temperature annotation
            ax2.text(0.05, 0.95, f'Tc = {system_3d.discovered_tc:.3f}\n(Theory: {system_3d.theoretical_tc:.3f})',
                    transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 3: Phase separation quality comparison
        ax3 = axes[0, 2]
        
        system_names = []
        separation_scores = []
        
        for system in comparison_data:
            if len(system.latent_z1) > 0 and len(system.latent_z2) > 0:
                # Calculate phase separation quality using temperature clustering
                separation_score = self._calculate_phase_separation_quality(
                    system.latent_z1, system.latent_z2, system.temperatures
                )
                system_names.append(f"{system.dimensionality}\n{system.model_type}")
                separation_scores.append(separation_score)
        
        if separation_scores:
            bars = ax3.bar(range(len(system_names)), separation_scores, 
                          alpha=0.7, color=[self.colors.get(f"{s.dimensionality}_{s.model_type}", 'gray') 
                                          for s in comparison_data[:len(separation_scores)]])
            
            ax3.set_xlabel('System', fontsize=12)
            ax3.set_ylabel('Phase Separation Quality', fontsize=12)
            ax3.set_title('Phase Separation Quality\nComparison', fontsize=14, fontweight='bold')
            ax3.set_xticks(range(len(system_names)))
            ax3.set_xticklabels(system_names)
            ax3.grid(True, alpha=0.3)
            
            # Add value annotations
            for bar, score in zip(bars, separation_scores):
                height = bar.get_height()
                ax3.annotate(f'{score:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
        
        # Plot 4: 2D magnetization coloring
        if systems_2d:
            ax4 = axes[1, 0]
            system_2d = systems_2d[0]
            
            scatter = ax4.scatter(
                system_2d.latent_z1, system_2d.latent_z2,
                c=np.abs(system_2d.magnetizations), cmap='viridis',
                alpha=0.6, s=30, edgecolors='black', linewidth=0.5
            )
            
            cbar = plt.colorbar(scatter, ax=ax4)
            cbar.set_label('|Magnetization|', fontsize=10)
            
            ax4.set_xlabel('z₁', fontsize=12)
            ax4.set_ylabel('z₂', fontsize=12)
            ax4.set_title('2D Ising Model\nLatent Space (Magnetization)', 
                         fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.set_aspect('equal', adjustable='box')
            
            # Add correlation annotation
            ax4.text(0.05, 0.95, f'r = {system_2d.order_parameter_correlation:.3f}',
                    transform=ax4.transAxes, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Plot 5: 3D magnetization coloring
        if systems_3d:
            ax5 = axes[1, 1]
            system_3d = systems_3d[0]
            
            scatter = ax5.scatter(
                system_3d.latent_z1, system_3d.latent_z2,
                c=np.abs(system_3d.magnetizations), cmap='viridis',
                alpha=0.6, s=30, edgecolors='black', linewidth=0.5
            )
            
            cbar = plt.colorbar(scatter, ax=ax5)
            cbar.set_label('|Magnetization|', fontsize=10)
            
            ax5.set_xlabel('z₁', fontsize=12)
            ax5.set_ylabel('z₂', fontsize=12)
            ax5.set_title('3D Ising Model\nLatent Space (Magnetization)', 
                         fontsize=14, fontweight='bold')
            ax5.grid(True, alpha=0.3)
            ax5.set_aspect('equal', adjustable='box')
            
            # Add correlation annotation
            ax5.text(0.05, 0.95, f'r = {system_3d.order_parameter_correlation:.3f}',
                    transform=ax5.transAxes, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Plot 6: Dimensionality comparison summary
        ax6 = axes[1, 2]
        
        # Compare 2D vs 3D performance
        if systems_2d and systems_3d:
            metrics = ['Order Parameter\nCorrelation', 'Critical Temperature\nAccuracy (%)', 
                      'Physics\nConsistency']
            
            system_2d = systems_2d[0]
            system_3d = systems_3d[0]
            
            values_2d = [
                system_2d.order_parameter_correlation,
                100 - abs(system_2d.tc_accuracy_percent),  # Convert error to accuracy
                system_2d.physics_consistency_score
            ]
            
            values_3d = [
                system_3d.order_parameter_correlation,
                100 - abs(system_3d.tc_accuracy_percent),  # Convert error to accuracy
                system_3d.physics_consistency_score
            ]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            bars1 = ax6.bar(x - width/2, values_2d, width, label='2D Ising', 
                           alpha=0.8, color=self.colors['2D_Ising'])
            bars2 = ax6.bar(x + width/2, values_3d, width, label='3D Ising', 
                           alpha=0.8, color=self.colors['3D_Ising'])
            
            ax6.set_xlabel('Metric', fontsize=12)
            ax6.set_ylabel('Score', fontsize=12)
            ax6.set_title('2D vs 3D Performance\nComparison', fontsize=14, fontweight='bold')
            ax6.set_xticks(x)
            ax6.set_xticklabels(metrics)
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            
            # Add value annotations
            for bars, values in [(bars1, values_2d), (bars2, values_3d)]:
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax6.annotate(f'{value:.3f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('2D vs 3D Phase Separation Analysis in Latent Space', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def generate_order_parameter_comparison_plots(self,
                                                comparison_data: List[SystemComparisonData],
                                                figsize: Tuple[int, int] = (15, 10)) -> Figure:
        """
        Generate order parameter comparison plots (discovered vs theoretical).
        
        Args:
            comparison_data: List of system comparison data
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure with order parameter comparison analysis
        """
        self.logger.info("Generating order parameter comparison plots")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: Order parameter correlation comparison
        ax1 = axes[0, 0]
        
        system_names = [f"{s.dimensionality} {s.model_type}" for s in comparison_data]
        correlations = [s.order_parameter_correlation for s in comparison_data]
        colors = [self.colors.get(f"{s.dimensionality}_{s.model_type}", 'gray') 
                 for s in comparison_data]
        
        bars = ax1.bar(range(len(system_names)), correlations, 
                      alpha=0.7, color=colors)
        
        ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, 
                   label='Target Correlation (0.8)')
        ax1.set_xlabel('System', fontsize=12)
        ax1.set_ylabel('Order Parameter Correlation', fontsize=12)
        ax1.set_title('Order Parameter Discovery\nAccuracy', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(system_names)))
        ax1.set_xticklabels(system_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value annotations
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax1.annotate(f'{corr:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Order parameter vs magnetization for each system
        for i, system in enumerate(comparison_data[:2]):  # Show first 2 systems
            ax = axes[0, i+1]
            
            if len(system.latent_z1) > 0 and len(system.magnetizations) > 0:
                # Use best latent dimension
                if system.best_latent_dimension == 'z1':
                    latent_values = system.latent_z1
                else:
                    latent_values = system.latent_z2
                
                scatter = ax.scatter(
                    np.abs(system.magnetizations), latent_values,
                    c=system.temperatures, cmap='coolwarm',
                    alpha=0.6, s=20, edgecolors='black', linewidth=0.5
                )
                
                # Add correlation line
                if len(latent_values) > 1:
                    z = np.polyfit(np.abs(system.magnetizations), latent_values, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(np.abs(system.magnetizations).min(), 
                                       np.abs(system.magnetizations).max(), 100)
                    ax.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2)
                
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Temperature', fontsize=8)
                
                ax.set_xlabel('|Magnetization|', fontsize=12)
                ax.set_ylabel(f'Order Parameter ({system.best_latent_dimension})', fontsize=12)
                ax.set_title(f'{system.dimensionality} {system.model_type}\nr = {system.order_parameter_correlation:.3f}', 
                           fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
        
        # Plot 3: Theoretical vs discovered order parameter behavior
        ax3 = axes[1, 0]
        
        # Create theoretical magnetization curves
        for system in comparison_data:
            if system.model_type == "Ising" and len(system.temperatures) > 0:
                # Theoretical Ising magnetization behavior
                T = system.temperatures
                Tc = system.theoretical_tc
                
                # Approximate theoretical magnetization
                theoretical_m = np.zeros_like(T)
                below_tc = T < Tc
                if system.dimensionality == "2D":
                    theoretical_m[below_tc] = (1 - T[below_tc]/Tc)**0.125  # β = 1/8
                else:  # 3D
                    theoretical_m[below_tc] = (1 - T[below_tc]/Tc)**0.326  # β ≈ 0.326
                
                ax3.plot(T, theoretical_m, '--', 
                        label=f'{system.dimensionality} Theoretical',
                        color=self.colors.get(f"{system.dimensionality}_{system.model_type}", 'gray'),
                        linewidth=2, alpha=0.8)
                
                # Discovered magnetization
                if len(system.magnetizations) > 0:
                    sorted_indices = np.argsort(system.temperatures)
                    ax3.plot(system.temperatures[sorted_indices], 
                           np.abs(system.magnetizations[sorted_indices]), 'o-',
                           label=f'{system.dimensionality} Discovered',
                           color=self.colors.get(f"{system.dimensionality}_{system.model_type}", 'gray'),
                           alpha=0.6, markersize=4)
        
        ax3.set_xlabel('Temperature', fontsize=12)
        ax3.set_ylabel('|Magnetization|', fontsize=12)
        ax3.set_title('Theoretical vs Discovered\nOrder Parameter', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Order parameter quality metrics
        ax4 = axes[1, 1]
        
        metrics = ['Correlation\nAccuracy', 'Phase Transition\nSharpness', 'Temperature\nSensitivity']
        
        # Calculate quality metrics for each system
        quality_scores = []
        for system in comparison_data:
            corr_accuracy = min(1.0, system.order_parameter_correlation / 0.8)
            
            # Phase transition sharpness (based on temperature range)
            if len(system.temperatures) > 0:
                temp_range = system.temperatures.max() - system.temperatures.min()
                sharpness = min(1.0, 5.0 / temp_range)  # Prefer narrower ranges
            else:
                sharpness = 0.0
            
            # Temperature sensitivity (correlation with temperature)
            if len(system.temperatures) > 0 and len(system.latent_z1) > 0:
                if system.best_latent_dimension == 'z1':
                    temp_corr = abs(np.corrcoef(system.temperatures, system.latent_z1)[0, 1])
                else:
                    temp_corr = abs(np.corrcoef(system.temperatures, system.latent_z2)[0, 1])
                sensitivity = min(1.0, temp_corr)
            else:
                sensitivity = 0.0
            
            quality_scores.append([corr_accuracy, sharpness, sensitivity])
        
        # Create grouped bar chart
        x = np.arange(len(metrics))
        width = 0.8 / len(comparison_data)
        
        for i, (system, scores) in enumerate(zip(comparison_data, quality_scores)):
            ax4.bar(x + i * width, scores, width, 
                   label=f'{system.dimensionality} {system.model_type}',
                   alpha=0.7, color=self.colors.get(f"{system.dimensionality}_{system.model_type}", 'gray'))
        
        ax4.set_xlabel('Quality Metric', fontsize=12)
        ax4.set_ylabel('Score', fontsize=12)
        ax4.set_title('Order Parameter\nQuality Assessment', fontsize=14, fontweight='bold')
        ax4.set_xticks(x + width * (len(comparison_data) - 1) / 2)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Cross-system order parameter consistency
        ax5 = axes[1, 2]
        
        # Compare order parameter discovery consistency across systems
        if len(comparison_data) >= 2:
            # Create consistency matrix
            consistency_matrix = np.zeros((len(comparison_data), len(comparison_data)))
            
            for i, system1 in enumerate(comparison_data):
                for j, system2 in enumerate(comparison_data):
                    if i == j:
                        consistency_matrix[i, j] = 1.0
                    else:
                        # Calculate consistency based on correlation similarity
                        corr_diff = abs(system1.order_parameter_correlation - 
                                      system2.order_parameter_correlation)
                        consistency = max(0.0, 1.0 - corr_diff)
                        consistency_matrix[i, j] = consistency
            
            im = ax5.imshow(consistency_matrix, cmap='RdYlGn', vmin=0, vmax=1)
            
            # Add text annotations
            for i in range(len(comparison_data)):
                for j in range(len(comparison_data)):
                    text = ax5.text(j, i, f'{consistency_matrix[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontweight='bold')
            
            ax5.set_xticks(range(len(comparison_data)))
            ax5.set_yticks(range(len(comparison_data)))
            ax5.set_xticklabels([f'{s.dimensionality}\n{s.model_type}' for s in comparison_data])
            ax5.set_yticklabels([f'{s.dimensionality}\n{s.model_type}' for s in comparison_data])
            ax5.set_title('Cross-System Order Parameter\nConsistency', fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax5)
            cbar.set_label('Consistency Score', fontsize=10)
        
        plt.suptitle('Order Parameter Discovery Comparison Analysis', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def generate_critical_temperature_accuracy_comparison(self,
                                                        comparison_data: List[SystemComparisonData],
                                                        figsize: Tuple[int, int] = (15, 10)) -> Figure:
        """
        Generate critical temperature accuracy comparison across all systems.
        
        Args:
            comparison_data: List of system comparison data
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure with critical temperature accuracy analysis
        """
        self.logger.info("Generating critical temperature accuracy comparison")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: Critical temperature accuracy comparison
        ax1 = axes[0, 0]
        
        system_names = [f"{s.dimensionality}\n{s.model_type}" for s in comparison_data]
        tc_errors = [abs(s.tc_accuracy_percent) for s in comparison_data]
        colors = [self.colors.get(f"{s.dimensionality}_{s.model_type}", 'gray') 
                 for s in comparison_data]
        
        bars = ax1.bar(range(len(system_names)), tc_errors, 
                      alpha=0.7, color=colors)
        
        ax1.axhline(y=5.0, color='red', linestyle='--', alpha=0.7, 
                   label='5% Error Threshold')
        ax1.axhline(y=10.0, color='orange', linestyle='--', alpha=0.7, 
                   label='10% Error Threshold')
        
        ax1.set_xlabel('System', fontsize=12)
        ax1.set_ylabel('Critical Temperature Error (%)', fontsize=12)
        ax1.set_title('Critical Temperature\nDetection Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(system_names)))
        ax1.set_xticklabels(system_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value annotations
        for bar, error in zip(bars, tc_errors):
            height = bar.get_height()
            ax1.annotate(f'{error:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Discovered vs theoretical critical temperatures
        ax2 = axes[0, 1]
        
        discovered_tcs = [s.discovered_tc for s in comparison_data]
        theoretical_tcs = [s.theoretical_tc for s in comparison_data]
        
        # Perfect correlation line
        min_tc = min(min(discovered_tcs), min(theoretical_tcs))
        max_tc = max(max(discovered_tcs), max(theoretical_tcs))
        ax2.plot([min_tc, max_tc], [min_tc, max_tc], 'k--', alpha=0.7, 
                label='Perfect Agreement')
        
        # Scatter plot
        for i, system in enumerate(comparison_data):
            ax2.scatter(system.theoretical_tc, system.discovered_tc, 
                       s=100, alpha=0.7, 
                       color=self.colors.get(f"{system.dimensionality}_{system.model_type}", 'gray'),
                       label=f'{system.dimensionality} {system.model_type}',
                       edgecolors='black', linewidth=1)
        
        ax2.set_xlabel('Theoretical Tc', fontsize=12)
        ax2.set_ylabel('Discovered Tc', fontsize=12)
        ax2.set_title('Discovered vs Theoretical\nCritical Temperature', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add error bars or confidence regions
        for system in comparison_data:
            error_margin = abs(system.discovered_tc - system.theoretical_tc)
            ax2.errorbar(system.theoretical_tc, system.discovered_tc, 
                        yerr=error_margin*0.1, fmt='none', 
                        color=self.colors.get(f"{system.dimensionality}_{system.model_type}", 'gray'),
                        alpha=0.5, capsize=3)
        
        # Plot 3: Critical temperature detection method comparison
        ax3 = axes[0, 2]
        
        # Group by model type to compare dimensionalities
        model_types = list(set(s.model_type for s in comparison_data))
        
        x = np.arange(len(model_types))
        width = 0.35
        
        errors_2d = []
        errors_3d = []
        
        for model_type in model_types:
            systems_2d = [s for s in comparison_data if s.model_type == model_type and s.dimensionality == "2D"]
            systems_3d = [s for s in comparison_data if s.model_type == model_type and s.dimensionality == "3D"]
            
            error_2d = systems_2d[0].tc_accuracy_percent if systems_2d else 0
            error_3d = systems_3d[0].tc_accuracy_percent if systems_3d else 0
            
            errors_2d.append(abs(error_2d))
            errors_3d.append(abs(error_3d))
        
        bars1 = ax3.bar(x - width/2, errors_2d, width, label='2D Systems', 
                       alpha=0.8, color=self.colors['2D_Ising'])
        bars2 = ax3.bar(x + width/2, errors_3d, width, label='3D Systems', 
                       alpha=0.8, color=self.colors['3D_Ising'])
        
        ax3.set_xlabel('Model Type', fontsize=12)
        ax3.set_ylabel('Critical Temperature Error (%)', fontsize=12)
        ax3.set_title('2D vs 3D Critical Temperature\nDetection Accuracy', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_types)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Temperature detection confidence analysis
        ax4 = axes[1, 0]
        
        # Calculate detection confidence based on phase transition sharpness
        confidence_scores = []
        for system in comparison_data:
            if len(system.temperatures) > 0 and len(system.magnetizations) > 0:
                # Calculate transition sharpness around discovered Tc
                tc = system.discovered_tc
                temp_window = 0.5  # Temperature window around Tc
                
                near_tc = np.abs(system.temperatures - tc) < temp_window
                if np.sum(near_tc) > 5:  # Need enough points
                    mag_near_tc = np.abs(system.magnetizations[near_tc])
                    temp_near_tc = system.temperatures[near_tc]
                    
                    # Calculate magnetization gradient
                    if len(mag_near_tc) > 1:
                        gradient = np.gradient(mag_near_tc, temp_near_tc)
                        sharpness = np.max(np.abs(gradient))
                        confidence = min(1.0, sharpness * 10)  # Scale to [0,1]
                    else:
                        confidence = 0.5
                else:
                    confidence = 0.5
            else:
                confidence = 0.5
            
            confidence_scores.append(confidence)
        
        bars = ax4.bar(range(len(system_names)), confidence_scores, 
                      alpha=0.7, color=colors)
        
        ax4.set_xlabel('System', fontsize=12)
        ax4.set_ylabel('Detection Confidence', fontsize=12)
        ax4.set_title('Critical Temperature\nDetection Confidence', fontsize=14, fontweight='bold')
        ax4.set_xticks(range(len(system_names)))
        ax4.set_xticklabels(system_names)
        ax4.grid(True, alpha=0.3)
        
        # Add value annotations
        for bar, conf in zip(bars, confidence_scores):
            height = bar.get_height()
            ax4.annotate(f'{conf:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        # Plot 5: Error distribution analysis
        ax5 = axes[1, 1]
        
        # Create histogram of errors
        all_errors = [abs(s.tc_accuracy_percent) for s in comparison_data]
        
        ax5.hist(all_errors, bins=10, alpha=0.7, color='skyblue', 
                edgecolor='black', density=True)
        
        # Add statistics
        mean_error = np.mean(all_errors)
        std_error = np.std(all_errors)
        
        ax5.axvline(mean_error, color='red', linestyle='-', linewidth=2, 
                   label=f'Mean: {mean_error:.1f}%')
        ax5.axvline(mean_error + std_error, color='red', linestyle='--', 
                   alpha=0.7, label=f'±1σ: {std_error:.1f}%')
        ax5.axvline(mean_error - std_error, color='red', linestyle='--', alpha=0.7)
        
        ax5.set_xlabel('Critical Temperature Error (%)', fontsize=12)
        ax5.set_ylabel('Density', fontsize=12)
        ax5.set_title('Error Distribution\nAcross All Systems', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Accuracy improvement potential
        ax6 = axes[1, 2]
        
        # Calculate potential improvement based on order parameter correlation
        current_errors = [abs(s.tc_accuracy_percent) for s in comparison_data]
        correlations = [s.order_parameter_correlation for s in comparison_data]
        
        # Estimate potential improvement (higher correlation should lead to better Tc detection)
        potential_errors = [error * (1 - corr) for error, corr in zip(current_errors, correlations)]
        improvements = [curr - pot for curr, pot in zip(current_errors, potential_errors)]
        
        x = np.arange(len(system_names))
        width = 0.35
        
        bars1 = ax6.bar(x - width/2, current_errors, width, label='Current Error', 
                       alpha=0.8, color='lightcoral')
        bars2 = ax6.bar(x + width/2, potential_errors, width, label='Potential Error', 
                       alpha=0.8, color='lightgreen')
        
        ax6.set_xlabel('System', fontsize=12)
        ax6.set_ylabel('Critical Temperature Error (%)', fontsize=12)
        ax6.set_title('Accuracy Improvement\nPotential', fontsize=14, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(system_names)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Add improvement annotations
        for i, (curr, pot, imp) in enumerate(zip(current_errors, potential_errors, improvements)):
            if imp > 0.1:  # Only show significant improvements
                ax6.annotate(f'↓{imp:.1f}%',
                           xy=(i, (curr + pot) / 2),
                           xytext=(0, 0), textcoords="offset points",
                           ha='center', va='center', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.suptitle('Critical Temperature Detection Accuracy Analysis', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def perform_comprehensive_comparison_analysis(self,
                                                comparison_data: List[SystemComparisonData]) -> ComparisonResults:
        """
        Perform comprehensive comparison analysis across all systems.
        
        Args:
            comparison_data: List of system comparison data
            
        Returns:
            ComparisonResults with detailed analysis
        """
        self.logger.info("Performing comprehensive comparison analysis")
        
        systems_compared = [f"{s.dimensionality}_{s.model_type}" for s in comparison_data]
        
        # Phase separation analysis
        phase_separation_quality = {}
        latent_space_clustering = {}
        
        for system in comparison_data:
            system_key = f"{system.dimensionality}_{system.model_type}"
            
            if len(system.latent_z1) > 0 and len(system.latent_z2) > 0:
                # Calculate phase separation quality
                separation_quality = self._calculate_phase_separation_quality(
                    system.latent_z1, system.latent_z2, system.temperatures
                )
                phase_separation_quality[system_key] = separation_quality
                
                # Calculate clustering quality
                clustering_quality = self._calculate_clustering_quality(
                    system.latent_z1, system.latent_z2, system.temperatures, system.discovered_tc
                )
                latent_space_clustering[system_key] = clustering_quality
        
        # Order parameter comparison
        order_parameter_accuracy = {}
        correlation_comparison = {}
        
        for system in comparison_data:
            system_key = f"{system.dimensionality}_{system.model_type}"
            
            # Order parameter accuracy (correlation with magnetization)
            order_parameter_accuracy[system_key] = system.order_parameter_correlation
            
            # Correlation comparison (relative to best possible)
            correlation_comparison[system_key] = min(1.0, system.order_parameter_correlation / 0.9)
        
        # Critical temperature comparison
        tc_detection_accuracy = {}
        tc_error_statistics = {}
        
        for system in comparison_data:
            system_key = f"{system.dimensionality}_{system.model_type}"
            
            # Detection accuracy (inverse of error)
            error_percent = abs(system.tc_accuracy_percent)
            accuracy = max(0.0, 100.0 - error_percent) / 100.0
            tc_detection_accuracy[system_key] = accuracy
            
            # Error statistics
            tc_error_statistics[system_key] = {
                'absolute_error': abs(system.discovered_tc - system.theoretical_tc),
                'relative_error_percent': error_percent,
                'theoretical_tc': system.theoretical_tc,
                'discovered_tc': system.discovered_tc
            }
        
        # Cross-system validation
        universality_validation = {}
        method_consistency = {}
        
        # Check universality (similar performance for same model type)
        model_types = list(set(s.model_type for s in comparison_data))
        for model_type in model_types:
            systems_of_type = [s for s in comparison_data if s.model_type == model_type]
            
            if len(systems_of_type) > 1:
                # Check if performance is consistent across dimensions
                correlations = [s.order_parameter_correlation for s in systems_of_type]
                tc_errors = [abs(s.tc_accuracy_percent) for s in systems_of_type]
                
                corr_consistency = 1.0 - (np.std(correlations) / (np.mean(correlations) + 1e-6))
                tc_consistency = 1.0 - (np.std(tc_errors) / (np.mean(tc_errors) + 1e-6))
                
                universality_validation[model_type] = (corr_consistency > 0.7 and tc_consistency > 0.7)
                method_consistency[model_type] = (corr_consistency + tc_consistency) / 2
            else:
                universality_validation[model_type] = True  # Single system, assume valid
                method_consistency[model_type] = 1.0
        
        return ComparisonResults(
            systems_compared=systems_compared,
            phase_separation_quality=phase_separation_quality,
            latent_space_clustering=latent_space_clustering,
            order_parameter_accuracy=order_parameter_accuracy,
            correlation_comparison=correlation_comparison,
            tc_detection_accuracy=tc_detection_accuracy,
            tc_error_statistics=tc_error_statistics,
            universality_validation=universality_validation,
            method_consistency=method_consistency
        )
    
    def _calculate_phase_separation_quality(self,
                                          z1: np.ndarray,
                                          z2: np.ndarray,
                                          temperatures: np.ndarray) -> float:
        """Calculate phase separation quality in latent space."""
        if len(z1) == 0 or len(z2) == 0 or len(temperatures) == 0:
            return 0.0
        
        try:
            # Calculate temperature-based clustering
            # High and low temperature regions should be well separated
            temp_median = np.median(temperatures)
            
            high_temp_mask = temperatures > temp_median
            low_temp_mask = temperatures <= temp_median
            
            if np.sum(high_temp_mask) == 0 or np.sum(low_temp_mask) == 0:
                return 0.0
            
            # Calculate centroids
            high_temp_centroid = np.array([np.mean(z1[high_temp_mask]), np.mean(z2[high_temp_mask])])
            low_temp_centroid = np.array([np.mean(z1[low_temp_mask]), np.mean(z2[low_temp_mask])])
            
            # Calculate separation distance
            separation_distance = np.linalg.norm(high_temp_centroid - low_temp_centroid)
            
            # Calculate within-cluster variance
            high_temp_variance = np.var(z1[high_temp_mask]) + np.var(z2[high_temp_mask])
            low_temp_variance = np.var(z1[low_temp_mask]) + np.var(z2[low_temp_mask])
            avg_variance = (high_temp_variance + low_temp_variance) / 2
            
            # Quality score: separation / variance (higher is better)
            if avg_variance > 0:
                quality = separation_distance / np.sqrt(avg_variance)
                return min(1.0, quality / 5.0)  # Normalize to [0,1]
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Error calculating phase separation quality: {e}")
            return 0.0
    
    def _calculate_clustering_quality(self,
                                    z1: np.ndarray,
                                    z2: np.ndarray,
                                    temperatures: np.ndarray,
                                    critical_temperature: float) -> float:
        """Calculate clustering quality around critical temperature."""
        if len(z1) == 0 or len(z2) == 0 or len(temperatures) == 0:
            return 0.0
        
        try:
            # Define temperature regions
            temp_window = 0.5
            below_tc = temperatures < (critical_temperature - temp_window)
            above_tc = temperatures > (critical_temperature + temp_window)
            near_tc = np.abs(temperatures - critical_temperature) <= temp_window
            
            if np.sum(below_tc) == 0 or np.sum(above_tc) == 0:
                return 0.0
            
            # Calculate silhouette-like score
            points_below = np.column_stack([z1[below_tc], z2[below_tc]])
            points_above = np.column_stack([z1[above_tc], z2[above_tc]])
            
            if len(points_below) == 0 or len(points_above) == 0:
                return 0.0
            
            # Calculate average intra-cluster distance
            intra_below = np.mean([np.linalg.norm(p1 - p2) 
                                 for i, p1 in enumerate(points_below) 
                                 for p2 in points_below[i+1:]])
            intra_above = np.mean([np.linalg.norm(p1 - p2) 
                                 for i, p1 in enumerate(points_above) 
                                 for p2 in points_above[i+1:]])
            
            # Calculate average inter-cluster distance
            inter_distance = np.mean([np.linalg.norm(p1 - p2) 
                                    for p1 in points_below 
                                    for p2 in points_above])
            
            # Clustering quality: inter / intra (higher is better)
            avg_intra = (intra_below + intra_above) / 2
            if avg_intra > 0:
                quality = inter_distance / avg_intra
                return min(1.0, quality / 3.0)  # Normalize to [0,1]
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Error calculating clustering quality: {e}")
            return 0.0
    
    def save_comparison_results(self,
                              comparison_results: ComparisonResults,
                              output_dir: str = 'results/publication/comparison_studies') -> str:
        """
        Save comparison results to JSON file.
        
        Args:
            comparison_results: Comparison analysis results
            output_dir: Output directory
            
        Returns:
            Path to saved results file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / "systematic_comparison_results.json"
        
        # Convert to serializable format
        results_dict = {
            'generation_time': datetime.now().isoformat(),
            'systems_compared': comparison_results.systems_compared,
            'phase_separation_quality': comparison_results.phase_separation_quality,
            'latent_space_clustering': comparison_results.latent_space_clustering,
            'order_parameter_accuracy': comparison_results.order_parameter_accuracy,
            'correlation_comparison': comparison_results.correlation_comparison,
            'tc_detection_accuracy': comparison_results.tc_detection_accuracy,
            'tc_error_statistics': comparison_results.tc_error_statistics,
            'universality_validation': comparison_results.universality_validation,
            'method_consistency': comparison_results.method_consistency
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        self.logger.info(f"Comparison results saved to {results_file}")
        return str(results_file)
    
    def generate_systematic_comparison_report(self,
                                            comparison_data: List[SystemComparisonData],
                                            output_dir: str = 'results/publication/comparison_studies') -> Dict[str, str]:
        """
        Generate complete systematic comparison report with all plots and analysis.
        
        Args:
            comparison_data: List of system comparison data
            output_dir: Output directory for results
            
        Returns:
            Dictionary mapping plot names to saved file paths
        """
        self.logger.info("Generating systematic comparison report")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_plots = {}
        
        try:
            # Generate all comparison plots
            plots = {}
            
            # 2D vs 3D phase separation plots
            plots['2d_vs_3d_phase_separation'] = self.generate_2d_vs_3d_phase_separation_plots(comparison_data)
            
            # Order parameter comparison plots
            plots['order_parameter_comparison'] = self.generate_order_parameter_comparison_plots(comparison_data)
            
            # Critical temperature accuracy comparison
            plots['critical_temperature_accuracy'] = self.generate_critical_temperature_accuracy_comparison(comparison_data)
            
            # Save all plots
            for plot_name, fig in plots.items():
                file_path = output_path / f"{plot_name}.png"
                fig.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
                saved_plots[plot_name] = str(file_path)
                
                # Also save as PDF for publication
                pdf_path = output_path / f"{plot_name}.pdf"
                fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
                
                plt.close(fig)
                
                self.logger.info(f"Saved comparison plot: {file_path}")
            
            # Perform comprehensive analysis
            comparison_results = self.perform_comprehensive_comparison_analysis(comparison_data)
            
            # Save analysis results
            results_file = self.save_comparison_results(comparison_results, output_dir)
            saved_plots['analysis_results'] = results_file
            
            # Generate summary report
            summary_file = self._generate_comparison_summary_report(
                comparison_data, comparison_results, output_path
            )
            saved_plots['summary_report'] = summary_file
            
        except Exception as e:
            self.logger.error(f"Error generating systematic comparison report: {e}")
            raise
        
        return saved_plots
    
    def _generate_comparison_summary_report(self,
                                          comparison_data: List[SystemComparisonData],
                                          comparison_results: ComparisonResults,
                                          output_path: Path) -> str:
        """Generate comprehensive summary report."""
        summary_file = output_path / "systematic_comparison_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Systematic Comparison Framework - Summary Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Systems analyzed: {len(comparison_data)}\n\n")
            
            # System overview
            f.write("Systems Overview:\n")
            f.write("-" * 20 + "\n")
            for system in comparison_data:
                f.write(f"  {system.system_name}:\n")
                f.write(f"    Dimensionality: {system.dimensionality}\n")
                f.write(f"    Model Type: {system.model_type}\n")
                f.write(f"    Theoretical Tc: {system.theoretical_tc:.4f}\n")
                f.write(f"    Discovered Tc: {system.discovered_tc:.4f}\n")
                f.write(f"    Tc Error: {abs(system.tc_accuracy_percent):.2f}%\n")
                f.write(f"    Order Parameter Correlation: {system.order_parameter_correlation:.4f}\n\n")
            
            # Phase separation analysis
            f.write("Phase Separation Quality:\n")
            f.write("-" * 25 + "\n")
            for system_key, quality in comparison_results.phase_separation_quality.items():
                f.write(f"  {system_key}: {quality:.4f}\n")
            f.write("\n")
            
            # Order parameter analysis
            f.write("Order Parameter Discovery:\n")
            f.write("-" * 25 + "\n")
            best_system = max(comparison_results.order_parameter_accuracy.items(), key=lambda x: x[1])
            f.write(f"  Best performing system: {best_system[0]} (r = {best_system[1]:.4f})\n")
            
            avg_correlation = np.mean(list(comparison_results.order_parameter_accuracy.values()))
            f.write(f"  Average correlation: {avg_correlation:.4f}\n")
            
            systems_above_threshold = sum(1 for corr in comparison_results.order_parameter_accuracy.values() if corr > 0.8)
            f.write(f"  Systems above 0.8 correlation: {systems_above_threshold}/{len(comparison_results.order_parameter_accuracy)}\n\n")
            
            # Critical temperature analysis
            f.write("Critical Temperature Detection:\n")
            f.write("-" * 30 + "\n")
            best_tc_system = max(comparison_results.tc_detection_accuracy.items(), key=lambda x: x[1])
            f.write(f"  Most accurate system: {best_tc_system[0]} (accuracy = {best_tc_system[1]:.4f})\n")
            
            avg_accuracy = np.mean(list(comparison_results.tc_detection_accuracy.values()))
            f.write(f"  Average accuracy: {avg_accuracy:.4f}\n")
            
            errors = [stats['relative_error_percent'] for stats in comparison_results.tc_error_statistics.values()]
            systems_below_5_percent = sum(1 for error in errors if error < 5.0)
            f.write(f"  Systems with <5% error: {systems_below_5_percent}/{len(errors)}\n\n")
            
            # Universality validation
            f.write("Universality Validation:\n")
            f.write("-" * 22 + "\n")
            for model_type, is_valid in comparison_results.universality_validation.items():
                consistency = comparison_results.method_consistency.get(model_type, 0.0)
                f.write(f"  {model_type}: {'PASS' if is_valid else 'FAIL'} (consistency = {consistency:.4f})\n")
            f.write("\n")
            
            # Overall assessment
            f.write("Overall Assessment:\n")
            f.write("-" * 18 + "\n")
            
            overall_correlation = avg_correlation
            overall_tc_accuracy = avg_accuracy
            overall_consistency = np.mean(list(comparison_results.method_consistency.values()))
            
            f.write(f"  Order Parameter Discovery: {overall_correlation:.4f} ({'EXCELLENT' if overall_correlation > 0.8 else 'GOOD' if overall_correlation > 0.6 else 'NEEDS IMPROVEMENT'})\n")
            f.write(f"  Critical Temperature Detection: {overall_tc_accuracy:.4f} ({'EXCELLENT' if overall_tc_accuracy > 0.9 else 'GOOD' if overall_tc_accuracy > 0.8 else 'NEEDS IMPROVEMENT'})\n")
            f.write(f"  Method Consistency: {overall_consistency:.4f} ({'EXCELLENT' if overall_consistency > 0.8 else 'GOOD' if overall_consistency > 0.6 else 'NEEDS IMPROVEMENT'})\n")
            
            overall_score = (overall_correlation + overall_tc_accuracy + overall_consistency) / 3
            f.write(f"\n  OVERALL SCORE: {overall_score:.4f} ({'PUBLICATION READY' if overall_score > 0.8 else 'GOOD PROGRESS' if overall_score > 0.6 else 'NEEDS IMPROVEMENT'})\n")
        
        self.logger.info(f"Summary report saved to {summary_file}")
        return str(summary_file)