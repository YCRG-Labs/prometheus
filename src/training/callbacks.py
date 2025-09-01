"""
Training Callbacks and Utilities

This module implements training callbacks including early stopping,
learning rate scheduling, gradient clipping, and training progress
visualization and logging integration.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import numpy as np

import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


class EarlyStopping:
    """
    Early stopping callback with configurable patience and improvement thresholds.
    
    Monitors a specified metric and stops training when no improvement
    is observed for a specified number of epochs (patience).
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        """
        Initialize early stopping callback.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for metrics that should decrease, 'max' for metrics that should increase
            restore_best_weights: Whether to restore model to best weights when stopping
            verbose: Whether to log early stopping events
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.logger = logging.getLogger(__name__)
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.inf
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
    
    def __call__(self, current_value: float, model: torch.nn.Module) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            current_value: Current value of the monitored metric
            model: Model to potentially save best weights from
            
        Returns:
            True if training should be stopped, False otherwise
        """
        if self.monitor_op(current_value - self.min_delta, self.best):
            self.best = current_value
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.wait
                if self.verbose:
                    self.logger.info(
                        f"Early stopping triggered after {self.patience} epochs "
                        f"without improvement. Best value: {self.best:.6f}"
                    )
                return True
        
        return False
    
    def restore_weights(self, model: torch.nn.Module) -> None:
        """Restore best weights to the model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                self.logger.info("Restored best model weights")


class LearningRateScheduler:
    """
    Enhanced learning rate scheduler with multiple scheduling strategies.
    
    Provides additional scheduling options beyond PyTorch's built-in schedulers,
    including custom decay functions and metric-based scheduling.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = 'plateau',
        **kwargs
    ):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: Type of scheduler ('plateau', 'exponential', 'cosine', 'step')
            **kwargs: Additional arguments for specific schedulers
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.logger = logging.getLogger(__name__)
        
        if scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get('mode', 'min'),
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 5),
                threshold=kwargs.get('threshold', 1e-4),
                verbose=kwargs.get('verbose', True)
            )
        elif scheduler_type == 'exponential':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=kwargs.get('gamma', 0.95)
            )
        elif scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('T_max', 100),
                eta_min=kwargs.get('eta_min', 0)
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    def step(self, metric: Optional[float] = None) -> None:
        """
        Step the learning rate scheduler.
        
        Args:
            metric: Metric value for plateau scheduler
        """
        if self.scheduler_type == 'plateau' and metric is not None:
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
    
    def get_last_lr(self) -> List[float]:
        """Get the last learning rate(s)."""
        return self.scheduler.get_last_lr()


class GradientClipper:
    """
    Gradient clipping utility for numerical stability during training.
    
    Provides various gradient clipping strategies including norm-based
    and value-based clipping with monitoring capabilities.
    """
    
    def __init__(
        self,
        clip_type: str = 'norm',
        clip_value: float = 1.0,
        monitor: bool = True
    ):
        """
        Initialize gradient clipper.
        
        Args:
            clip_type: Type of clipping ('norm' or 'value')
            clip_value: Clipping threshold
            monitor: Whether to monitor gradient statistics
        """
        self.clip_type = clip_type
        self.clip_value = clip_value
        self.monitor = monitor
        self.logger = logging.getLogger(__name__)
        
        self.gradient_norms = []
        self.clipping_events = 0
        self.total_steps = 0
    
    def clip_gradients(self, model: torch.nn.Module) -> float:
        """
        Clip gradients of model parameters.
        
        Args:
            model: PyTorch model
            
        Returns:
            Gradient norm before clipping
        """
        self.total_steps += 1
        
        if self.clip_type == 'norm':
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                self.clip_value
            ).item()
        elif self.clip_type == 'value':
            torch.nn.utils.clip_grad_value_(
                model.parameters(), 
                self.clip_value
            )
            # Calculate norm for monitoring
            grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
        else:
            raise ValueError(f"Unsupported clip_type: {self.clip_type}")
        
        if self.monitor:
            self.gradient_norms.append(grad_norm)
            if grad_norm > self.clip_value:
                self.clipping_events += 1
        
        return grad_norm
    
    def get_statistics(self) -> Dict[str, float]:
        """Get gradient clipping statistics."""
        if not self.gradient_norms:
            return {}
        
        return {
            'mean_grad_norm': np.mean(self.gradient_norms),
            'max_grad_norm': np.max(self.gradient_norms),
            'clipping_rate': self.clipping_events / self.total_steps if self.total_steps > 0 else 0.0,
            'total_clipping_events': self.clipping_events
        }


class TrainingProgressVisualizer:
    """
    Training progress visualization and logging integration.
    
    Creates real-time plots of training metrics and integrates with
    logging systems for comprehensive training monitoring.
    """
    
    def __init__(
        self,
        save_dir: str = "results/training_plots",
        update_frequency: int = 10,
        plot_metrics: Optional[List[str]] = None
    ):
        """
        Initialize training progress visualizer.
        
        Args:
            save_dir: Directory to save plots
            update_frequency: How often to update plots (in epochs)
            plot_metrics: List of metrics to plot (None for all)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.update_frequency = update_frequency
        self.plot_metrics = plot_metrics or ['total_loss', 'reconstruction_loss', 'kl_loss']
        self.logger = logging.getLogger(__name__)
        
        self.training_history = {}
        self.epoch_count = 0
    
    def update(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Update training history and create plots if needed.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics for current epoch
            val_metrics: Validation metrics for current epoch (optional)
        """
        self.epoch_count = epoch
        
        # Update history
        for metric, value in train_metrics.items():
            if f'train_{metric}' not in self.training_history:
                self.training_history[f'train_{metric}'] = []
            self.training_history[f'train_{metric}'].append(value)
        
        if val_metrics:
            for metric, value in val_metrics.items():
                if f'val_{metric}' not in self.training_history:
                    self.training_history[f'val_{metric}'] = []
                self.training_history[f'val_{metric}'].append(value)
        
        # Create plots if it's time
        if epoch % self.update_frequency == 0:
            self.create_plots()
    
    def create_plots(self) -> None:
        """Create and save training progress plots."""
        if not self.training_history:
            return
        
        epochs = range(1, self.epoch_count + 1)
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Progress - Epoch {self.epoch_count}', fontsize=16)
        
        # Plot total loss
        ax = axes[0, 0]
        if 'train_total_loss' in self.training_history:
            ax.plot(epochs, self.training_history['train_total_loss'], 'b-', label='Train')
        if 'val_total_loss' in self.training_history:
            ax.plot(epochs, self.training_history['val_total_loss'], 'r-', label='Validation')
        ax.set_title('Total Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
        # Plot reconstruction loss
        ax = axes[0, 1]
        if 'train_reconstruction_loss' in self.training_history:
            ax.plot(epochs, self.training_history['train_reconstruction_loss'], 'b-', label='Train')
        if 'val_reconstruction_loss' in self.training_history:
            ax.plot(epochs, self.training_history['val_reconstruction_loss'], 'r-', label='Validation')
        ax.set_title('Reconstruction Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
        # Plot KL divergence
        ax = axes[1, 0]
        if 'train_kl_loss' in self.training_history:
            ax.plot(epochs, self.training_history['train_kl_loss'], 'b-', label='Train')
        if 'val_kl_loss' in self.training_history:
            ax.plot(epochs, self.training_history['val_kl_loss'], 'r-', label='Validation')
        ax.set_title('KL Divergence')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('KL Loss')
        ax.legend()
        ax.grid(True)
        
        # Plot learning rate if available
        ax = axes[1, 1]
        if hasattr(self, 'learning_rates') and self.learning_rates:
            ax.plot(epochs[:len(self.learning_rates)], self.learning_rates, 'g-')
            ax.set_title('Learning Rate')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_yscale('log')
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, 'Learning Rate\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Learning Rate')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.save_dir / f'training_progress_epoch_{self.epoch_count}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training progress plot saved: {plot_path}")
    
    def update_learning_rate(self, lr: float) -> None:
        """Update learning rate for plotting."""
        if not hasattr(self, 'learning_rates'):
            self.learning_rates = []
        self.learning_rates.append(lr)
    
    def save_final_plots(self) -> None:
        """Save final comprehensive training plots."""
        if not self.training_history:
            return
        
        epochs = range(1, self.epoch_count + 1)
        
        # Create comprehensive final plot
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Final Training Results', fontsize=16)
        
        # Total loss
        ax = axes[0, 0]
        if 'train_total_loss' in self.training_history:
            ax.plot(epochs, self.training_history['train_total_loss'], 'b-', label='Train', linewidth=2)
        if 'val_total_loss' in self.training_history:
            ax.plot(epochs, self.training_history['val_total_loss'], 'r-', label='Validation', linewidth=2)
        ax.set_title('Total Loss', fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Reconstruction loss
        ax = axes[0, 1]
        if 'train_reconstruction_loss' in self.training_history:
            ax.plot(epochs, self.training_history['train_reconstruction_loss'], 'b-', label='Train', linewidth=2)
        if 'val_reconstruction_loss' in self.training_history:
            ax.plot(epochs, self.training_history['val_reconstruction_loss'], 'r-', label='Validation', linewidth=2)
        ax.set_title('Reconstruction Loss', fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # KL divergence
        ax = axes[0, 2]
        if 'train_kl_loss' in self.training_history:
            ax.plot(epochs, self.training_history['train_kl_loss'], 'b-', label='Train', linewidth=2)
        if 'val_kl_loss' in self.training_history:
            ax.plot(epochs, self.training_history['val_kl_loss'], 'r-', label='Validation', linewidth=2)
        ax.set_title('KL Divergence', fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('KL Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate
        ax = axes[1, 0]
        if hasattr(self, 'learning_rates') and self.learning_rates:
            ax.plot(epochs[:len(self.learning_rates)], self.learning_rates, 'g-', linewidth=2)
            ax.set_yscale('log')
        ax.set_title('Learning Rate', fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.grid(True, alpha=0.3)
        
        # Loss components comparison
        ax = axes[1, 1]
        if 'train_reconstruction_loss' in self.training_history and 'train_kl_loss' in self.training_history:
            ax.plot(epochs, self.training_history['train_reconstruction_loss'], 'b-', label='Reconstruction', linewidth=2)
            ax.plot(epochs, self.training_history['train_kl_loss'], 'r-', label='KL Divergence', linewidth=2)
            ax.set_title('Loss Components (Train)', fontsize=14)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Training summary statistics
        ax = axes[1, 2]
        ax.axis('off')
        
        # Calculate summary statistics
        summary_text = "Training Summary\n\n"
        if 'train_total_loss' in self.training_history:
            final_train_loss = self.training_history['train_total_loss'][-1]
            min_train_loss = min(self.training_history['train_total_loss'])
            summary_text += f"Final Train Loss: {final_train_loss:.6f}\n"
            summary_text += f"Best Train Loss: {min_train_loss:.6f}\n\n"
        
        if 'val_total_loss' in self.training_history:
            final_val_loss = self.training_history['val_total_loss'][-1]
            min_val_loss = min(self.training_history['val_total_loss'])
            summary_text += f"Final Val Loss: {final_val_loss:.6f}\n"
            summary_text += f"Best Val Loss: {min_val_loss:.6f}\n\n"
        
        summary_text += f"Total Epochs: {self.epoch_count}\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save final plot
        final_plot_path = self.save_dir / 'final_training_results.png'
        plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Final training plot saved: {final_plot_path}")


class TrainingLogger:
    """
    Enhanced training logger with structured logging and metrics tracking.
    
    Provides comprehensive logging of training events, metrics, and system
    information with integration to various logging backends.
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: str = "vae_training",
        log_level: str = "INFO"
    ):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name of the experiment
            log_level: Logging level
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # Set up logger
        self.logger = logging.getLogger(f"training.{experiment_name}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create file handler
        log_file = self.log_dir / f"{experiment_name}_{int(time.time())}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.file_handler = file_handler  # Keep reference for cleanup
        
        self.logger.info(f"Training logger initialized for experiment: {experiment_name}")
    
    def close(self):
        """Close the logger and file handlers."""
        if hasattr(self, 'file_handler'):
            self.file_handler.close()
            self.logger.removeHandler(self.file_handler)
    
    def log_epoch_metrics(
        self, 
        epoch: int, 
        train_metrics: Dict[str, float], 
        val_metrics: Optional[Dict[str, float]] = None,
        epoch_time: float = 0.0
    ) -> None:
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics (optional)
            epoch_time: Time taken for the epoch
        """
        log_msg = f"Epoch {epoch} - "
        log_msg += f"Train: {', '.join([f'{k}={v:.6f}' for k, v in train_metrics.items()])}"
        
        if val_metrics:
            log_msg += f" | Val: {', '.join([f'{k}={v:.6f}' for k, v in val_metrics.items()])}"
        
        if epoch_time > 0:
            log_msg += f" | Time: {epoch_time:.2f}s"
        
        self.logger.info(log_msg)
    
    def log_training_start(self, config: Dict[str, Any]) -> None:
        """Log training start with configuration."""
        self.logger.info("=" * 50)
        self.logger.info("TRAINING STARTED")
        self.logger.info("=" * 50)
        self.logger.info(f"Configuration: {config}")
    
    def log_training_end(self, total_time: float, best_metrics: Dict[str, float]) -> None:
        """Log training completion."""
        self.logger.info("=" * 50)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("=" * 50)
        self.logger.info(f"Total training time: {total_time:.2f}s")
        self.logger.info(f"Best metrics: {best_metrics}")