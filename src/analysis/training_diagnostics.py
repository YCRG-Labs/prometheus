"""
Advanced Training Diagnostics and Analysis

This module provides comprehensive training diagnostics including detailed loss curves,
learning rate schedules, latent space evolution, and gradient analysis for publication-ready
training analysis.
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
import torch.nn as nn
from collections import defaultdict

from ..utils.logging_utils import get_logger


@dataclass
class TrainingMetrics:
    """Container for training metrics and diagnostics."""
    epoch: int
    train_loss: float
    val_loss: Optional[float]
    reconstruction_loss: float
    kl_loss: float
    learning_rate: float
    gradient_norm: Optional[float]
    latent_samples: Optional[np.ndarray]
    reconstruction_samples: Optional[np.ndarray]


class TrainingDiagnostics:
    """
    Advanced training diagnostics and visualization system.
    
    Provides comprehensive analysis of training progress including:
    - Detailed loss decomposition
    - Learning rate schedule tracking
    - Gradient norm monitoring
    - Latent space evolution
    - Reconstruction quality analysis
    """
    
    def __init__(self):
        """Initialize training diagnostics system."""
        self.logger = get_logger(__name__)
        self.metrics_history = []
        self.gradient_history = defaultdict(list)
        self.latent_evolution = []
        
        # Publication settings
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = sns.color_palette("husl", 8)
        
    def record_epoch_metrics(self, 
                           epoch: int,
                           train_loss: float,
                           val_loss: Optional[float],
                           reconstruction_loss: float,
                           kl_loss: float,
                           learning_rate: float,
                           model: Optional[nn.Module] = None,
                           gradient_norm: Optional[float] = None,
                           latent_samples: Optional[np.ndarray] = None,
                           reconstruction_samples: Optional[np.ndarray] = None) -> None:
        """
        Record metrics for a training epoch.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss (optional)
            reconstruction_loss: Reconstruction component of loss
            kl_loss: KL divergence component of loss
            learning_rate: Current learning rate
            model: Model for gradient analysis (optional)
            gradient_norm: Pre-calculated gradient norm (optional)
            latent_samples: Sample latent representations (optional)
            reconstruction_samples: Sample reconstructions (optional)
        """
        # Calculate gradient norm if model provided, otherwise use provided value
        if gradient_norm is None and model is not None:
            gradient_norm = self._calculate_gradient_norm(model)
        
        # Create metrics record
        metrics = TrainingMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            reconstruction_loss=reconstruction_loss,
            kl_loss=kl_loss,
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
            latent_samples=latent_samples,
            reconstruction_samples=reconstruction_samples
        )
        
        self.metrics_history.append(metrics)
        
        # Record gradient information by layer
        if model is not None:
            self._record_gradient_by_layer(model, epoch)
        
        # Record latent evolution
        if latent_samples is not None:
            self.latent_evolution.append({
                'epoch': epoch,
                'latent_samples': latent_samples.copy()
            })
    
    def _calculate_gradient_norm(self, model: nn.Module) -> float:
        """Calculate total gradient norm across all parameters."""
        total_norm = 0.0
        param_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        return (total_norm ** 0.5) if param_count > 0 else 0.0
    
    def _record_gradient_by_layer(self, model: nn.Module, epoch: int) -> None:
        """Record gradient norms by layer."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                self.gradient_history[name].append({
                    'epoch': epoch,
                    'gradient_norm': grad_norm
                })
    
    def plot_detailed_loss_curves(self, figsize: Tuple[int, int] = (15, 10)) -> Figure:
        """
        Create detailed training loss curves with separate reconstruction and KL terms.
        
        Args:
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure with detailed loss analysis
        """
        self.logger.info("Creating detailed loss curves visualization")
        
        if not self.metrics_history:
            raise ValueError("No training metrics recorded")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Extract data
        epochs = [m.epoch for m in self.metrics_history]
        train_losses = [m.train_loss for m in self.metrics_history]
        val_losses = [m.val_loss for m in self.metrics_history if m.val_loss is not None]
        val_epochs = [m.epoch for m in self.metrics_history if m.val_loss is not None]
        recon_losses = [m.reconstruction_loss for m in self.metrics_history]
        kl_losses = [m.kl_loss for m in self.metrics_history]
        
        # Plot 1: Overall loss curves
        ax1 = axes[0, 0]
        ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
        if val_losses:
            ax1.plot(val_epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Total Loss', fontsize=12)
        ax1.set_title('Overall Training Progress', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Loss components
        ax2 = axes[0, 1]
        ax2.plot(epochs, recon_losses, 'g-', linewidth=2, label='Reconstruction Loss', alpha=0.8)
        ax2.plot(epochs, kl_losses, 'm-', linewidth=2, label='KL Divergence', alpha=0.8)
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss Component', fontsize=12)
        ax2.set_title('Loss Decomposition', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Plot 3: Loss ratio analysis
        ax3 = axes[1, 0]
        loss_ratios = [kl / (recon + 1e-8) for recon, kl in zip(recon_losses, kl_losses)]
        ax3.plot(epochs, loss_ratios, 'orange', linewidth=2, alpha=0.8)
        
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('KL / Reconstruction Ratio', fontsize=12)
        ax3.set_title('Loss Balance Analysis', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Loss smoothed trends
        ax4 = axes[1, 1]
        
        # Apply smoothing
        window_size = max(1, len(epochs) // 20)
        if window_size > 1:
            train_smooth = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
            recon_smooth = np.convolve(recon_losses, np.ones(window_size)/window_size, mode='valid')
            kl_smooth = np.convolve(kl_losses, np.ones(window_size)/window_size, mode='valid')
            smooth_epochs = epochs[window_size-1:]
            
            ax4.plot(smooth_epochs, train_smooth, 'b-', linewidth=2, label='Total (Smoothed)', alpha=0.8)
            ax4.plot(smooth_epochs, recon_smooth, 'g-', linewidth=2, label='Reconstruction (Smoothed)', alpha=0.8)
            ax4.plot(smooth_epochs, kl_smooth, 'm-', linewidth=2, label='KL (Smoothed)', alpha=0.8)
        else:
            ax4.plot(epochs, train_losses, 'b-', linewidth=2, label='Total', alpha=0.8)
            ax4.plot(epochs, recon_losses, 'g-', linewidth=2, label='Reconstruction', alpha=0.8)
            ax4.plot(epochs, kl_losses, 'm-', linewidth=2, label='KL', alpha=0.8)
        
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Loss (Smoothed)', fontsize=12)
        ax4.set_title('Smoothed Loss Trends', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.suptitle('Detailed Training Loss Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_learning_rate_schedule(self, figsize: Tuple[int, int] = (12, 8)) -> Figure:
        """
        Create learning rate schedule plots and convergence analysis.
        
        Args:
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure with learning rate analysis
        """
        self.logger.info("Creating learning rate schedule visualization")
        
        if not self.metrics_history:
            raise ValueError("No training metrics recorded")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Extract data
        epochs = [m.epoch for m in self.metrics_history]
        learning_rates = [m.learning_rate for m in self.metrics_history]
        train_losses = [m.train_loss for m in self.metrics_history]
        
        # Plot 1: Learning rate schedule
        ax1 = axes[0, 0]
        ax1.plot(epochs, learning_rates, 'b-', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Learning Rate', fontsize=12)
        ax1.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Loss vs Learning Rate
        ax2 = axes[0, 1]
        scatter = ax2.scatter(learning_rates, train_losses, c=epochs, 
                            cmap='viridis', alpha=0.6, s=20)
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Epoch', fontsize=12)
        
        ax2.set_xlabel('Learning Rate', fontsize=12)
        ax2.set_ylabel('Training Loss', fontsize=12)
        ax2.set_title('Loss vs Learning Rate', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        # Plot 3: Learning rate changes
        ax3 = axes[1, 0]
        lr_changes = []
        change_epochs = []
        
        for i in range(1, len(learning_rates)):
            if learning_rates[i] != learning_rates[i-1]:
                lr_change = learning_rates[i] / learning_rates[i-1]
                lr_changes.append(lr_change)
                change_epochs.append(epochs[i])
        
        if lr_changes:
            ax3.scatter(change_epochs, lr_changes, c='red', s=50, alpha=0.8)
            ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            
            for epoch, change in zip(change_epochs, lr_changes):
                ax3.annotate(f'{change:.3f}', (epoch, change), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('LR Change Ratio', fontsize=12)
        ax3.set_title('Learning Rate Adjustments', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Convergence analysis
        ax4 = axes[1, 1]
        
        # Calculate loss improvement rate
        loss_improvements = []
        improvement_epochs = []
        window_size = max(1, len(train_losses) // 10)
        
        for i in range(window_size, len(train_losses)):
            recent_avg = np.mean(train_losses[i-window_size:i])
            current_loss = train_losses[i]
            improvement = (recent_avg - current_loss) / recent_avg if recent_avg > 0 else 0
            loss_improvements.append(improvement)
            improvement_epochs.append(epochs[i])
        
        if loss_improvements:
            ax4.plot(improvement_epochs, loss_improvements, 'g-', linewidth=2, alpha=0.8)
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Loss Improvement Rate', fontsize=12)
        ax4.set_title('Convergence Analysis', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Learning Rate Schedule and Convergence Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_latent_space_evolution(self, figsize: Tuple[int, int] = (15, 10)) -> Figure:
        """
        Create latent space evolution visualization during training epochs.
        
        Args:
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure with latent space evolution
        """
        self.logger.info("Creating latent space evolution visualization")
        
        if not self.latent_evolution:
            raise ValueError("No latent space evolution data recorded")
        
        # Select key epochs for visualization
        n_snapshots = min(6, len(self.latent_evolution))
        snapshot_indices = np.linspace(0, len(self.latent_evolution)-1, n_snapshots, dtype=int)
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # Color map for consistency
        cmap = plt.cm.viridis
        
        for i, idx in enumerate(snapshot_indices):
            ax = axes[i]
            epoch_data = self.latent_evolution[idx]
            epoch = epoch_data['epoch']
            latent_samples = epoch_data['latent_samples']
            
            if latent_samples.shape[1] >= 2:
                # Plot 2D latent space
                scatter = ax.scatter(latent_samples[:, 0], latent_samples[:, 1], 
                                   c=np.arange(len(latent_samples)), cmap=cmap,
                                   alpha=0.6, s=20)
                
                ax.set_xlabel('Latent Dimension 1 (z₁)', fontsize=10)
                ax.set_ylabel('Latent Dimension 2 (z₂)', fontsize=10)
                ax.set_title(f'Epoch {epoch}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')
                
                # Add statistics
                z1_std = np.std(latent_samples[:, 0])
                z2_std = np.std(latent_samples[:, 1])
                ax.text(0.05, 0.95, f'σ₁={z1_std:.2f}\nσ₂={z2_std:.2f}', 
                       transform=ax.transAxes, fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for i in range(n_snapshots, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Latent Space Evolution During Training', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_gradient_analysis(self, figsize: Tuple[int, int] = (15, 10)) -> Figure:
        """
        Create gradient norm tracking and optimization landscape analysis.
        
        Args:
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure with gradient analysis
        """
        self.logger.info("Creating gradient analysis visualization")
        
        if not self.metrics_history or not any(m.gradient_norm for m in self.metrics_history):
            raise ValueError("No gradient information recorded")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Extract gradient data
        epochs = [m.epoch for m in self.metrics_history if m.gradient_norm is not None]
        gradient_norms = [m.gradient_norm for m in self.metrics_history if m.gradient_norm is not None]
        
        # Plot 1: Overall gradient norm
        ax1 = axes[0, 0]
        ax1.plot(epochs, gradient_norms, 'b-', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Gradient Norm', fontsize=12)
        ax1.set_title('Overall Gradient Norm', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Gradient norm distribution
        ax2 = axes[0, 1]
        ax2.hist(gradient_norms, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(np.mean(gradient_norms), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(gradient_norms):.3f}')
        ax2.axvline(np.median(gradient_norms), color='orange', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(gradient_norms):.3f}')
        
        ax2.set_xlabel('Gradient Norm', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Gradient Norm Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Gradient norm by layer (if available)
        ax3 = axes[1, 0]
        
        if self.gradient_history:
            # Select a few key layers for visualization
            layer_names = list(self.gradient_history.keys())
            selected_layers = layer_names[:min(5, len(layer_names))]
            
            for i, layer_name in enumerate(selected_layers):
                layer_data = self.gradient_history[layer_name]
                layer_epochs = [d['epoch'] for d in layer_data]
                layer_norms = [d['gradient_norm'] for d in layer_data]
                
                ax3.plot(layer_epochs, layer_norms, linewidth=2, alpha=0.8,
                        label=layer_name.split('.')[-1][:10], color=self.colors[i])
            
            ax3.set_xlabel('Epoch', fontsize=12)
            ax3.set_ylabel('Gradient Norm', fontsize=12)
            ax3.set_title('Gradient Norms by Layer', fontsize=14, fontweight='bold')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
        else:
            ax3.text(0.5, 0.5, 'No per-layer\ngradient data available', 
                    transform=ax3.transAxes, ha='center', va='center',
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
        
        # Plot 4: Gradient stability analysis
        ax4 = axes[1, 1]
        
        # Calculate gradient stability (rolling standard deviation)
        window_size = max(1, len(gradient_norms) // 10)
        if window_size > 1 and len(gradient_norms) > window_size:
            gradient_stability = []
            stability_epochs = []
            
            for i in range(window_size, len(gradient_norms)):
                window_std = np.std(gradient_norms[i-window_size:i])
                gradient_stability.append(window_std)
                stability_epochs.append(epochs[i])
            
            ax4.plot(stability_epochs, gradient_stability, 'purple', linewidth=2, alpha=0.8)
            ax4.set_xlabel('Epoch', fontsize=12)
            ax4.set_ylabel('Gradient Stability (Std)', fontsize=12)
            ax4.set_title('Gradient Stability Analysis', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor stability analysis', 
                    transform=ax4.transAxes, ha='center', va='center',
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
        
        plt.suptitle('Gradient Analysis and Optimization Landscape', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def generate_comprehensive_report(self, output_dir: str = 'results/training_diagnostics') -> Dict[str, str]:
        """
        Generate comprehensive training diagnostics report with all visualizations.
        
        Args:
            output_dir: Directory to save diagnostic plots
            
        Returns:
            Dictionary mapping plot names to saved file paths
        """
        self.logger.info("Generating comprehensive training diagnostics report")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_plots = {}
        
        try:
            # Generate all diagnostic plots
            plots = {
                'loss_curves': self.plot_detailed_loss_curves(),
                'learning_rate_schedule': self.plot_learning_rate_schedule(),
                'gradient_analysis': self.plot_gradient_analysis()
            }
            
            # Add latent evolution if available
            if self.latent_evolution:
                plots['latent_evolution'] = self.plot_latent_space_evolution()
            
            # Save all plots
            for plot_name, fig in plots.items():
                file_path = output_path / f"{plot_name}.png"
                fig.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
                saved_plots[plot_name] = str(file_path)
                
                # Also save as PDF for publication
                pdf_path = output_path / f"{plot_name}.pdf"
                fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
                
                plt.close(fig)
                
                self.logger.info(f"Saved training diagnostic plot: {file_path}")
            
            # Generate summary statistics
            self._save_training_summary(output_path)
            
        except Exception as e:
            self.logger.error(f"Error generating training diagnostics: {e}")
            raise
        
        return saved_plots
    
    def _save_training_summary(self, output_path: Path) -> None:
        """Save training summary statistics to text file."""
        if not self.metrics_history:
            return
        
        summary_path = output_path / "training_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("Training Diagnostics Summary\n")
            f.write("=" * 40 + "\n\n")
            
            # Basic statistics
            final_metrics = self.metrics_history[-1]
            f.write(f"Total Epochs: {final_metrics.epoch}\n")
            f.write(f"Final Training Loss: {final_metrics.train_loss:.6f}\n")
            if final_metrics.val_loss:
                f.write(f"Final Validation Loss: {final_metrics.val_loss:.6f}\n")
            f.write(f"Final Reconstruction Loss: {final_metrics.reconstruction_loss:.6f}\n")
            f.write(f"Final KL Loss: {final_metrics.kl_loss:.6f}\n")
            f.write(f"Final Learning Rate: {final_metrics.learning_rate:.2e}\n")
            
            # Loss statistics
            train_losses = [m.train_loss for m in self.metrics_history]
            f.write(f"\nLoss Statistics:\n")
            f.write(f"  Best Training Loss: {min(train_losses):.6f}\n")
            f.write(f"  Loss Reduction: {(train_losses[0] - train_losses[-1]) / train_losses[0] * 100:.1f}%\n")
            
            # Gradient statistics
            gradient_norms = [m.gradient_norm for m in self.metrics_history if m.gradient_norm]
            if gradient_norms:
                f.write(f"\nGradient Statistics:\n")
                f.write(f"  Mean Gradient Norm: {np.mean(gradient_norms):.6f}\n")
                f.write(f"  Max Gradient Norm: {max(gradient_norms):.6f}\n")
                f.write(f"  Min Gradient Norm: {min(gradient_norms):.6f}\n")
        
        self.logger.info(f"Training summary saved: {summary_path}")