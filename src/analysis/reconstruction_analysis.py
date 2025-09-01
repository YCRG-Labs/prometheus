"""
Reconstruction Quality Analysis

This module provides comprehensive reconstruction quality analysis including
side-by-side comparisons, error heatmaps, interpolation visualizations,
and quantitative metrics across temperature ranges.
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
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from scipy.interpolate import griddata

from ..utils.logging_utils import get_logger


@dataclass
class ReconstructionMetrics:
    """Container for reconstruction quality metrics."""
    mse: float
    mae: float
    ssim_score: float
    psnr: float
    temperature: float
    magnetization: float


class ReconstructionAnalyzer:
    """
    Comprehensive reconstruction quality analysis system.
    
    Provides detailed analysis of VAE reconstruction quality including:
    - Side-by-side comparison grids
    - Reconstruction error heatmaps
    - Latent space interpolation
    - Quantitative metrics (MSE, SSIM, PSNR)
    """
    
    def __init__(self):
        """Initialize reconstruction analyzer."""
        self.logger = get_logger(__name__)
        
        # Publication settings
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = sns.color_palette("husl", 8)
        
    def calculate_reconstruction_metrics(self,
                                      original: np.ndarray,
                                      reconstructed: np.ndarray,
                                      temperature: float,
                                      magnetization: float) -> ReconstructionMetrics:
        """
        Calculate comprehensive reconstruction quality metrics.
        
        Args:
            original: Original spin configuration
            reconstructed: Reconstructed spin configuration
            temperature: Temperature of the configuration
            magnetization: Magnetization of the configuration
            
        Returns:
            ReconstructionMetrics object with all quality metrics
        """
        # Ensure arrays are 2D
        if original.ndim > 2:
            original = original.squeeze()
        if reconstructed.ndim > 2:
            reconstructed = reconstructed.squeeze()
        
        # Calculate MSE
        mse = np.mean((original - reconstructed) ** 2)
        
        # Calculate MAE
        mae = np.mean(np.abs(original - reconstructed))
        
        # Calculate SSIM (structural similarity)
        # Convert to range [0, 1] for SSIM calculation
        orig_norm = (original + 1) / 2
        recon_norm = (reconstructed + 1) / 2
        ssim_score = ssim(orig_norm, recon_norm, data_range=1.0)
        
        # Calculate PSNR (Peak Signal-to-Noise Ratio)
        if mse > 0:
            max_pixel = 2.0  # Range is [-1, 1], so max difference is 2
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        else:
            psnr = float('inf')
        
        return ReconstructionMetrics(
            mse=mse,
            mae=mae,
            ssim_score=ssim_score,
            psnr=psnr,
            temperature=temperature,
            magnetization=magnetization
        )
    
    def create_comparison_grid(self,
                             original_configs: np.ndarray,
                             reconstructed_configs: np.ndarray,
                             temperatures: np.ndarray,
                             magnetizations: np.ndarray,
                             n_examples: int = 12,
                             figsize: Tuple[int, int] = (15, 10)) -> Figure:
        """
        Create side-by-side comparison grids of original vs reconstructed configurations.
        
        Args:
            original_configs: Original spin configurations [N, H, W]
            reconstructed_configs: Reconstructed spin configurations [N, H, W]
            temperatures: Temperature values for each configuration
            magnetizations: Magnetization values for each configuration
            n_examples: Number of examples to show
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure with comparison grid
        """
        self.logger.info("Creating reconstruction comparison grid")
        
        # Select diverse examples across temperature range
        n_examples = min(n_examples, len(original_configs))
        
        # Sort by temperature and select evenly spaced examples
        temp_indices = np.argsort(temperatures)
        selected_indices = temp_indices[np.linspace(0, len(temp_indices)-1, n_examples, dtype=int)]
        
        # Create figure with subplots
        n_cols = 4  # Original, Reconstructed, Difference, Error
        n_rows = n_examples
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(selected_indices):
            original = original_configs[idx].squeeze()
            reconstructed = reconstructed_configs[idx].squeeze()
            temp = temperatures[idx]
            mag = magnetizations[idx]
            
            # Calculate difference and error
            difference = original - reconstructed
            error_map = np.abs(difference)
            
            # Calculate metrics
            metrics = self.calculate_reconstruction_metrics(original, reconstructed, temp, mag)
            
            # Plot original
            ax_orig = axes[i, 0]
            im1 = ax_orig.imshow(original, cmap='RdBu', vmin=-1, vmax=1, aspect='equal')
            ax_orig.set_title(f'Original\nT={temp:.3f}', fontsize=10)
            ax_orig.axis('off')
            
            # Plot reconstructed
            ax_recon = axes[i, 1]
            im2 = ax_recon.imshow(reconstructed, cmap='RdBu', vmin=-1, vmax=1, aspect='equal')
            ax_recon.set_title(f'Reconstructed\nMSE={metrics.mse:.4f}', fontsize=10)
            ax_recon.axis('off')
            
            # Plot difference
            ax_diff = axes[i, 2]
            diff_max = max(np.abs(difference.min()), np.abs(difference.max()))
            im3 = ax_diff.imshow(difference, cmap='seismic', vmin=-diff_max, vmax=diff_max, aspect='equal')
            ax_diff.set_title(f'Difference\nMAE={metrics.mae:.4f}', fontsize=10)
            ax_diff.axis('off')
            
            # Plot error heatmap
            ax_error = axes[i, 3]
            im4 = ax_error.imshow(error_map, cmap='hot', vmin=0, vmax=error_map.max(), aspect='equal')
            ax_error.set_title(f'Error Map\nSSIM={metrics.ssim_score:.3f}', fontsize=10)
            ax_error.axis('off')
            
            # Add colorbar for the first row
            if i == 0:
                plt.colorbar(im1, ax=ax_orig, fraction=0.046, pad=0.04)
                plt.colorbar(im2, ax=ax_recon, fraction=0.046, pad=0.04)
                plt.colorbar(im3, ax=ax_diff, fraction=0.046, pad=0.04)
                plt.colorbar(im4, ax=ax_error, fraction=0.046, pad=0.04)
        
        # Add column headers
        col_titles = ['Original', 'Reconstructed', 'Difference', 'Error Map']
        for j, title in enumerate(col_titles):
            axes[0, j].text(0.5, 1.15, title, transform=axes[0, j].transAxes,
                          ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        plt.suptitle('Reconstruction Quality Comparison Grid', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def create_error_heatmaps(self,
                            original_configs: np.ndarray,
                            reconstructed_configs: np.ndarray,
                            temperatures: np.ndarray,
                            magnetizations: np.ndarray,
                            figsize: Tuple[int, int] = (15, 10)) -> Figure:
        """
        Generate reconstruction error heatmaps across different temperature regimes.
        
        Args:
            original_configs: Original spin configurations
            reconstructed_configs: Reconstructed spin configurations
            temperatures: Temperature values
            magnetizations: Magnetization values
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure with error heatmaps
        """
        self.logger.info("Creating reconstruction error heatmaps")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Define temperature regimes
        temp_min, temp_max = temperatures.min(), temperatures.max()
        temp_ranges = [
            (temp_min, temp_min + (temp_max - temp_min) * 0.33, "Low T (Ordered)"),
            (temp_min + (temp_max - temp_min) * 0.33, temp_min + (temp_max - temp_min) * 0.67, "Mid T (Critical)"),
            (temp_min + (temp_max - temp_min) * 0.67, temp_max, "High T (Disordered)")
        ]
        
        # Calculate metrics for each regime
        regime_metrics = []
        for temp_low, temp_high, regime_name in temp_ranges:
            regime_mask = (temperatures >= temp_low) & (temperatures <= temp_high)
            regime_indices = np.where(regime_mask)[0]
            
            if len(regime_indices) > 0:
                # Calculate average error map for this regime
                error_maps = []
                mse_values = []
                ssim_values = []
                
                for idx in regime_indices:
                    original = original_configs[idx].squeeze()
                    reconstructed = reconstructed_configs[idx].squeeze()
                    
                    error_map = np.abs(original - reconstructed)
                    error_maps.append(error_map)
                    
                    metrics = self.calculate_reconstruction_metrics(
                        original, reconstructed, temperatures[idx], magnetizations[idx]
                    )
                    mse_values.append(metrics.mse)
                    ssim_values.append(metrics.ssim_score)
                
                avg_error_map = np.mean(error_maps, axis=0)
                avg_mse = np.mean(mse_values)
                avg_ssim = np.mean(ssim_values)
                
                regime_metrics.append({
                    'name': regime_name,
                    'error_map': avg_error_map,
                    'mse': avg_mse,
                    'ssim': avg_ssim,
                    'temp_range': (temp_low, temp_high),
                    'n_samples': len(regime_indices)
                })
        
        # Plot error heatmaps for each regime
        for i, regime in enumerate(regime_metrics):
            ax = axes[0, i]
            im = ax.imshow(regime['error_map'], cmap='hot', aspect='equal')
            ax.set_title(f"{regime['name']}\nMSE={regime['mse']:.4f}, SSIM={regime['ssim']:.3f}", 
                        fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Plot 4: Overall error statistics
        ax4 = axes[1, 0]
        
        regime_names = [r['name'] for r in regime_metrics]
        mse_values = [r['mse'] for r in regime_metrics]
        ssim_values = [r['ssim'] for r in regime_metrics]
        
        x_pos = np.arange(len(regime_names))
        
        ax4_twin = ax4.twinx()
        bars1 = ax4.bar(x_pos - 0.2, mse_values, 0.4, label='MSE', color='red', alpha=0.7)
        bars2 = ax4_twin.bar(x_pos + 0.2, ssim_values, 0.4, label='SSIM', color='blue', alpha=0.7)
        
        ax4.set_xlabel('Temperature Regime', fontsize=12)
        ax4.set_ylabel('MSE', fontsize=12, color='red')
        ax4_twin.set_ylabel('SSIM', fontsize=12, color='blue')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([name.split()[0] for name in regime_names])
        ax4.set_title('Error Metrics by Regime', fontsize=12, fontweight='bold')
        
        # Add value annotations
        for bar, value in zip(bars1, mse_values):
            height = bar.get_height()
            ax4.annotate(f'{value:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
        for bar, value in zip(bars2, ssim_values):
            height = bar.get_height()
            ax4_twin.annotate(f'{value:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
        # Plot 5: Error vs Temperature scatter
        ax5 = axes[1, 1]
        
        all_mse = []
        all_temps = []
        
        for i in range(len(original_configs)):
            original = original_configs[i].squeeze()
            reconstructed = reconstructed_configs[i].squeeze()
            metrics = self.calculate_reconstruction_metrics(
                original, reconstructed, temperatures[i], magnetizations[i]
            )
            all_mse.append(metrics.mse)
            all_temps.append(temperatures[i])
        
        scatter = ax5.scatter(all_temps, all_mse, c=np.abs(magnetizations), 
                            cmap='viridis', alpha=0.6, s=20)
        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label('|Magnetization|', fontsize=10)
        
        ax5.set_xlabel('Temperature', fontsize=12)
        ax5.set_ylabel('MSE', fontsize=12)
        ax5.set_title('Reconstruction Error vs Temperature', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: SSIM vs Temperature scatter
        ax6 = axes[1, 2]
        
        all_ssim = []
        for i in range(len(original_configs)):
            original = original_configs[i].squeeze()
            reconstructed = reconstructed_configs[i].squeeze()
            metrics = self.calculate_reconstruction_metrics(
                original, reconstructed, temperatures[i], magnetizations[i]
            )
            all_ssim.append(metrics.ssim_score)
        
        scatter = ax6.scatter(all_temps, all_ssim, c=np.abs(magnetizations), 
                            cmap='viridis', alpha=0.6, s=20)
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label('|Magnetization|', fontsize=10)
        
        ax6.set_xlabel('Temperature', fontsize=12)
        ax6.set_ylabel('SSIM', fontsize=12)
        ax6.set_title('Structural Similarity vs Temperature', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Reconstruction Error Analysis Across Temperature Regimes', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def create_interpolation_visualization(self,
                                         model: torch.nn.Module,
                                         latent_samples: np.ndarray,
                                         temperatures: np.ndarray,
                                         n_interpolations: int = 8,
                                         n_steps: int = 10,
                                         figsize: Tuple[int, int] = (15, 12)) -> Figure:
        """
        Create latent space interpolation visualizations showing smooth transitions.
        
        Args:
            model: Trained VAE model
            latent_samples: Latent space samples [N, latent_dim]
            temperatures: Corresponding temperatures
            n_interpolations: Number of interpolation paths to show
            n_steps: Number of steps in each interpolation
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure with interpolation visualization
        """
        self.logger.info("Creating latent space interpolation visualization")
        
        model.eval()
        device = next(model.parameters()).device
        
        fig, axes = plt.subplots(n_interpolations, n_steps + 2, figsize=figsize)
        
        # Select diverse pairs for interpolation
        temp_indices = np.argsort(temperatures)
        
        for i in range(n_interpolations):
            # Select two points with different temperatures
            idx1 = temp_indices[i * len(temp_indices) // (n_interpolations * 2)]
            idx2 = temp_indices[-(i * len(temp_indices) // (n_interpolations * 2) + 1)]
            
            z1 = latent_samples[idx1]
            z2 = latent_samples[idx2]
            temp1 = temperatures[idx1]
            temp2 = temperatures[idx2]
            
            # Create interpolation path
            alphas = np.linspace(0, 1, n_steps)
            interpolated_z = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                interpolated_z.append(z_interp)
            
            interpolated_z = np.array(interpolated_z)
            
            # Generate reconstructions
            with torch.no_grad():
                z_tensor = torch.FloatTensor(interpolated_z).to(device)
                reconstructions = model.decode(z_tensor).cpu().numpy()
            
            # Plot start point
            ax_start = axes[i, 0]
            ax_start.imshow(reconstructions[0].squeeze(), cmap='RdBu', vmin=-1, vmax=1, aspect='equal')
            ax_start.set_title(f'Start\nT={temp1:.3f}', fontsize=10)
            ax_start.axis('off')
            
            # Plot interpolation steps
            for j in range(n_steps):
                ax = axes[i, j + 1]
                ax.imshow(reconstructions[j].squeeze(), cmap='RdBu', vmin=-1, vmax=1, aspect='equal')
                ax.set_title(f'Step {j+1}', fontsize=10)
                ax.axis('off')
            
            # Plot end point
            ax_end = axes[i, -1]
            ax_end.imshow(reconstructions[-1].squeeze(), cmap='RdBu', vmin=-1, vmax=1, aspect='equal')
            ax_end.set_title(f'End\nT={temp2:.3f}', fontsize=10)
            ax_end.axis('off')
            
            # Add interpolation path info
            axes[i, 0].text(-0.1, 0.5, f'Path {i+1}', transform=axes[i, 0].transAxes,
                          rotation=90, ha='center', va='center', fontsize=12, fontweight='bold')
        
        plt.suptitle('Latent Space Interpolation: Smooth Transitions Between Configurations', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def calculate_quantitative_metrics(self,
                                     original_configs: np.ndarray,
                                     reconstructed_configs: np.ndarray,
                                     temperatures: np.ndarray,
                                     magnetizations: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive quantitative reconstruction metrics.
        
        Args:
            original_configs: Original spin configurations
            reconstructed_configs: Reconstructed spin configurations
            temperatures: Temperature values
            magnetizations: Magnetization values
            
        Returns:
            Dictionary with comprehensive metrics
        """
        self.logger.info("Calculating quantitative reconstruction metrics")
        
        all_metrics = []
        
        for i in range(len(original_configs)):
            original = original_configs[i].squeeze()
            reconstructed = reconstructed_configs[i].squeeze()
            
            metrics = self.calculate_reconstruction_metrics(
                original, reconstructed, temperatures[i], magnetizations[i]
            )
            all_metrics.append(metrics)
        
        # Aggregate statistics
        mse_values = [m.mse for m in all_metrics]
        mae_values = [m.mae for m in all_metrics]
        ssim_values = [m.ssim_score for m in all_metrics]
        psnr_values = [m.psnr for m in all_metrics if np.isfinite(m.psnr)]
        
        # Temperature-based analysis
        temp_bins = np.linspace(temperatures.min(), temperatures.max(), 5)
        temp_bin_metrics = {}
        
        for i in range(len(temp_bins) - 1):
            bin_mask = (temperatures >= temp_bins[i]) & (temperatures < temp_bins[i + 1])
            bin_indices = np.where(bin_mask)[0]
            
            if len(bin_indices) > 0:
                bin_mse = [mse_values[j] for j in bin_indices]
                bin_ssim = [ssim_values[j] for j in bin_indices]
                
                temp_bin_metrics[f'bin_{i}'] = {
                    'temp_range': (temp_bins[i], temp_bins[i + 1]),
                    'n_samples': len(bin_indices),
                    'mse_mean': np.mean(bin_mse),
                    'mse_std': np.std(bin_mse),
                    'ssim_mean': np.mean(bin_ssim),
                    'ssim_std': np.std(bin_ssim)
                }
        
        return {
            'overall_metrics': {
                'mse_mean': np.mean(mse_values),
                'mse_std': np.std(mse_values),
                'mse_min': np.min(mse_values),
                'mse_max': np.max(mse_values),
                'mae_mean': np.mean(mae_values),
                'mae_std': np.std(mae_values),
                'ssim_mean': np.mean(ssim_values),
                'ssim_std': np.std(ssim_values),
                'ssim_min': np.min(ssim_values),
                'ssim_max': np.max(ssim_values),
                'psnr_mean': np.mean(psnr_values) if psnr_values else 0,
                'psnr_std': np.std(psnr_values) if psnr_values else 0,
                'n_samples': len(all_metrics)
            },
            'temperature_analysis': temp_bin_metrics,
            'detailed_metrics': all_metrics
        }
    
    def generate_reconstruction_report(self,
                                     model: torch.nn.Module,
                                     original_configs: np.ndarray,
                                     reconstructed_configs: np.ndarray,
                                     latent_samples: np.ndarray,
                                     temperatures: np.ndarray,
                                     magnetizations: np.ndarray,
                                     output_dir: str = 'results/reconstruction_analysis') -> Dict[str, str]:
        """
        Generate comprehensive reconstruction quality report.
        
        Args:
            model: Trained VAE model
            original_configs: Original spin configurations
            reconstructed_configs: Reconstructed spin configurations
            latent_samples: Latent space samples
            temperatures: Temperature values
            magnetizations: Magnetization values
            output_dir: Output directory for results
            
        Returns:
            Dictionary mapping plot names to saved file paths
        """
        self.logger.info("Generating comprehensive reconstruction quality report")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_plots = {}
        
        try:
            # Generate all reconstruction analysis plots
            plots = {
                'comparison_grid': self.create_comparison_grid(
                    original_configs, reconstructed_configs, temperatures, magnetizations
                ),
                'error_heatmaps': self.create_error_heatmaps(
                    original_configs, reconstructed_configs, temperatures, magnetizations
                ),
                'interpolation_visualization': self.create_interpolation_visualization(
                    model, latent_samples, temperatures
                )
            }
            
            # Save all plots
            for plot_name, fig in plots.items():
                file_path = output_path / f"{plot_name}.png"
                fig.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
                saved_plots[plot_name] = str(file_path)
                
                # Also save as PDF for publication
                pdf_path = output_path / f"{plot_name}.pdf"
                fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
                
                plt.close(fig)
                
                self.logger.info(f"Saved reconstruction analysis plot: {file_path}")
            
            # Calculate and save quantitative metrics
            metrics = self.calculate_quantitative_metrics(
                original_configs, reconstructed_configs, temperatures, magnetizations
            )
            
            metrics_path = output_path / "reconstruction_metrics.txt"
            self._save_metrics_summary(metrics, metrics_path)
            
        except Exception as e:
            self.logger.error(f"Error generating reconstruction analysis: {e}")
            raise
        
        return saved_plots
    
    def _save_metrics_summary(self, metrics: Dict[str, Any], output_path: Path) -> None:
        """Save reconstruction metrics summary to text file."""
        with open(output_path, 'w') as f:
            f.write("Reconstruction Quality Metrics Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall metrics
            overall = metrics['overall_metrics']
            f.write("Overall Reconstruction Quality:\n")
            f.write(f"  MSE: {overall['mse_mean']:.6f} ± {overall['mse_std']:.6f}\n")
            f.write(f"  MAE: {overall['mae_mean']:.6f} ± {overall['mae_std']:.6f}\n")
            f.write(f"  SSIM: {overall['ssim_mean']:.4f} ± {overall['ssim_std']:.4f}\n")
            f.write(f"  PSNR: {overall['psnr_mean']:.2f} ± {overall['psnr_std']:.2f} dB\n")
            f.write(f"  Total Samples: {overall['n_samples']}\n\n")
            
            # Temperature analysis
            f.write("Temperature-Based Analysis:\n")
            for bin_name, bin_data in metrics['temperature_analysis'].items():
                temp_low, temp_high = bin_data['temp_range']
                f.write(f"  Temperature Range [{temp_low:.3f}, {temp_high:.3f}]:\n")
                f.write(f"    Samples: {bin_data['n_samples']}\n")
                f.write(f"    MSE: {bin_data['mse_mean']:.6f} ± {bin_data['mse_std']:.6f}\n")
                f.write(f"    SSIM: {bin_data['ssim_mean']:.4f} ± {bin_data['ssim_std']:.4f}\n\n")
        
        self.logger.info(f"Reconstruction metrics summary saved: {output_path}")