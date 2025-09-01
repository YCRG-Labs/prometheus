"""
VAE Training Pipeline with Monitoring and Checkpointing

This module implements the VAETrainer class that orchestrates the training
process for Variational Autoencoders with comprehensive monitoring,
checkpointing, and validation capabilities.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from ..models.vae import ConvolutionalVAE
from ..utils.config import PrometheusConfig
from .callbacks import EarlyStopping, GradientClipper, TrainingProgressVisualizer, TrainingLogger
from .advanced_schedulers import create_advanced_scheduler
from .data_augmentation import create_standard_augmentation, create_conservative_augmentation, create_aggressive_augmentation, AugmentedDataset
from .ensemble_training import EnsembleTrainer
from .progressive_training import ProgressiveTrainer, create_default_progressive_schedule


class VAETrainer:
    """
    Comprehensive VAE training pipeline with monitoring and checkpointing.
    
    Handles training and validation epochs, loss computation, optimizer
    coordination, metrics tracking, and model checkpointing with best
    model saving based on validation loss.
    """
    
    def __init__(
        self,
        model: ConvolutionalVAE,
        config: PrometheusConfig,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the VAE trainer.
        
        Args:
            model: ConvolutionalVAE model to train
            config: Configuration object with training parameters
            device: Device to run training on (auto-detected if None)
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set up device
        if device is None:
            if config.device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(config.device)
        else:
            self.device = device
        
        self.model.to(self.device)
        self.logger.info(f"Training on device: {self.device}")
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.training_history = defaultdict(list)
        
        # Initialize callbacks
        self.early_stopping = None
        self.gradient_clipper = GradientClipper(clip_type='norm', clip_value=1.0)
        self.progress_visualizer = TrainingProgressVisualizer(
            save_dir=f"{config.results_dir}/training_plots"
        )
        self.training_logger = TrainingLogger(
            log_dir=config.logging.log_dir,
            experiment_name="vae_training"
        )
        
        # Create directories
        self.models_dir = Path(config.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("VAETrainer initialized successfully")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_name = self.config.training.optimizer.lower()
        lr = self.config.training.learning_rate
        
        if optimizer_name == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == "adamw":
            optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        self.logger.info(f"Created {optimizer_name} optimizer with lr={lr}")
        return optimizer
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        # Check for advanced scheduler first
        if self.config.training.advanced_scheduler != "none":
            try:
                scheduler = create_advanced_scheduler(
                    self.optimizer,
                    self.config.training.advanced_scheduler,
                    **self.config.training.scheduler_params
                )
                self.logger.info(f"Created advanced scheduler: {self.config.training.advanced_scheduler}")
                return scheduler
            except Exception as e:
                self.logger.warning(f"Failed to create advanced scheduler: {e}, falling back to standard")
        
        # Standard schedulers
        scheduler_name = self.config.training.scheduler.lower()
        
        if scheduler_name == "none" or scheduler_name == "":
            return None
        elif scheduler_name == "reducelronplateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        elif scheduler_name == "steplr":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_name == "cosineannealinglr":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        
        self.logger.info(f"Created {scheduler_name} scheduler")
        return scheduler
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Execute one training epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Dictionary of training metrics for the epoch
        """
        self.model.train()
        epoch_metrics = defaultdict(float)
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Handle both labeled and unlabeled data
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                data, _ = batch
            else:
                data = batch
            
            # Move data to device
            data = data.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            reconstruction, mu, logvar = self.model(data)
            
            # Compute loss
            loss_dict = self.model.compute_loss(
                data, reconstruction, mu, logvar, reduction='mean'
            )
            
            # Backward pass
            loss_dict['total_loss'].backward()
            
            # Gradient clipping for numerical stability
            grad_norm = self.gradient_clipper.clip_gradients(self.model)
            
            # Optimizer step
            self.optimizer.step()
            
            # Accumulate metrics
            for key, value in loss_dict.items():
                epoch_metrics[key] += value.item()
            
            # Log progress
            if batch_idx % 50 == 0:
                self.logger.debug(
                    f"Train Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, "
                    f"Loss: {loss_dict['total_loss'].item():.6f}"
                )
        
        # Average metrics over batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return dict(epoch_metrics)
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Execute one validation epoch.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary of validation metrics for the epoch
        """
        self.model.eval()
        epoch_metrics = defaultdict(float)
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in val_loader:
                # Handle both labeled and unlabeled data
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    data, _ = batch
                else:
                    data = batch
                
                # Move data to device
                data = data.to(self.device)
                
                # Forward pass
                reconstruction, mu, logvar = self.model(data)
                
                # Compute loss
                loss_dict = self.model.compute_loss(
                    data, reconstruction, mu, logvar, reduction='mean'
                )
                
                # Accumulate metrics
                for key, value in loss_dict.items():
                    epoch_metrics[key] += value.item()
        
        # Average metrics over batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return dict(epoch_metrics)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None,
        enable_early_stopping: bool = True
    ) -> Dict[str, List[float]]:
        """
        Execute complete training loop.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            num_epochs: Number of epochs to train (uses config if None)
            enable_early_stopping: Whether to enable early stopping
            
        Returns:
            Training history dictionary
        """
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs
        
        # Initialize early stopping if enabled
        if enable_early_stopping and val_loader is not None:
            self.early_stopping = EarlyStopping(
                patience=self.config.training.early_stopping_patience,
                mode='min',
                verbose=True
            )
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.training_logger.log_training_start({
            'num_epochs': num_epochs,
            'batch_size': self.config.training.batch_size,
            'learning_rate': self.config.training.learning_rate,
            'early_stopping': enable_early_stopping and val_loader is not None
        })
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            epoch_start_time = time.time()
            
            # Training epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validation epoch
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loader is not None:
                        self.scheduler.step(val_metrics['total_loss'])
                    else:
                        self.scheduler.step(train_metrics['total_loss'])
                else:
                    self.scheduler.step()
            
            # Record metrics
            for key, value in train_metrics.items():
                self.training_history[f'train_{key}'].append(value)
            
            for key, value in val_metrics.items():
                self.training_history[f'val_{key}'].append(value)
            
            # Check for best model
            current_val_loss = val_metrics.get('total_loss', train_metrics['total_loss'])
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.logger.info(f"New best model at epoch {self.current_epoch} with val_loss: {current_val_loss:.6f}")
            
            # Early stopping check
            if self.early_stopping is not None:
                if self.early_stopping(current_val_loss, self.model):
                    self.logger.info(f"Early stopping triggered at epoch {self.current_epoch}")
                    if self.early_stopping.restore_best_weights:
                        self.early_stopping.restore_weights(self.model)
                    break
            
            # Update progress visualizer
            current_lr = self.get_current_learning_rate()
            self.progress_visualizer.update_learning_rate(current_lr)
            self.progress_visualizer.update(self.current_epoch, train_metrics, val_metrics)
            
            # Checkpointing
            if self.current_epoch % self.config.training.checkpoint_interval == 0:
                self.save_checkpoint()
            
            # Logging
            epoch_time = time.time() - epoch_start_time
            self.training_logger.log_epoch_metrics(
                self.current_epoch, train_metrics, val_metrics, epoch_time
            )
            self.logger.info(
                f"Epoch {self.current_epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['total_loss']:.6f}, "
                f"Train Recon: {train_metrics['reconstruction_loss']:.6f}, "
                f"Train KL: {train_metrics['kl_loss']:.6f}"
                + (f", Val Loss: {val_metrics['total_loss']:.6f}" if val_metrics else "")
                + f" - Time: {epoch_time:.2f}s, LR: {current_lr:.2e}"
            )
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        
        # Log training completion
        best_metrics = {'best_val_loss': self.best_val_loss}
        self.training_logger.log_training_end(total_time, best_metrics)
        
        # Save final visualizations
        self.progress_visualizer.save_final_plots()
        
        # Log gradient clipping statistics
        grad_stats = self.gradient_clipper.get_statistics()
        if grad_stats:
            self.logger.info(f"Gradient clipping statistics: {grad_stats}")
        
        # Close training logger to release file handles
        self.training_logger.close()
        
        # Save final model and best model
        self.save_checkpoint(is_final=True)
        if self.best_model_state is not None:
            self.save_best_model()
        
        return dict(self.training_history)
    
    def save_checkpoint(self, is_final: bool = False) -> None:
        """
        Save training checkpoint.
        
        Args:
            is_final: Whether this is the final checkpoint
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'training_history': dict(self.training_history),
            'config': self.config
        }
        
        if is_final:
            checkpoint_path = self.models_dir / "final_checkpoint.pth"
        else:
            checkpoint_path = self.models_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def save_best_model(self) -> None:
        """Save the best model state."""
        if self.best_model_state is None:
            self.logger.warning("No best model state available to save")
            return
        
        best_model_path = self.models_dir / "best_model.pth"
        
        model_info = {
            'model_state_dict': self.best_model_state,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'model_architecture': {
                'input_shape': self.model.input_shape,
                'latent_dim': self.model.latent_dim,
                'beta': self.model.beta
            }
        }
        
        torch.save(model_info, best_model_path)
        self.logger.info(f"Best model saved: {best_model_path} (val_loss: {self.best_val_loss:.6f})")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = defaultdict(list, checkpoint['training_history'])
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path} (epoch {self.current_epoch})")
    
    def get_training_metrics(self) -> Dict[str, List[float]]:
        """Get current training history."""
        return dict(self.training_history)
    
    def get_current_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def compute_reconstruction_accuracy(self, data_loader: DataLoader) -> float:
        """
        Compute reconstruction accuracy on a dataset.
        
        Args:
            data_loader: DataLoader for the dataset
            
        Returns:
            Mean reconstruction accuracy (percentage of correctly reconstructed spins)
        """
        self.model.eval()
        total_accuracy = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # Handle both labeled and unlabeled data
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    data, _ = batch
                else:
                    data = batch
                
                data = data.to(self.device)
                reconstruction, _, _ = self.model(data)
                
                # Convert to spin values and compute accuracy
                data_spins = torch.sign(data)
                recon_spins = torch.sign(reconstruction)
                
                # Compute per-sample accuracy
                correct = (data_spins == recon_spins).float()
                accuracy = correct.view(data.size(0), -1).mean(dim=1)
                
                total_accuracy += accuracy.sum().item()
                num_samples += data.size(0)
        
        return total_accuracy / num_samples if num_samples > 0 else 0.0