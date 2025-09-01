"""
Progressive Training Implementation

This module implements progressive training that starts with lower resolution
configurations and gradually increases complexity. This approach can improve
convergence and help the model learn hierarchical representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
import time
import copy

# Conditional import to avoid issues when running standalone
try:
    from ..models.vae import ConvolutionalVAE
except ImportError:
    # For standalone testing, create a dummy class
    class ConvolutionalVAE:
        pass
# Conditional imports to avoid issues when running standalone
try:
    from ..utils.config import PrometheusConfig
    from .trainer import VAETrainer
    from .data_augmentation import IsingAugmentation
except ImportError:
    # For standalone testing, create dummy classes
    class PrometheusConfig:
        pass
    class VAETrainer:
        pass
    class IsingAugmentation:
        pass


class ProgressiveResolutionScheduler:
    """
    Scheduler for progressive resolution training.
    
    Manages the transition between different resolution stages during training.
    """
    
    def __init__(self,
                 resolution_schedule: List[Tuple[int, int]],
                 epochs_per_stage: List[int],
                 transition_epochs: int = 5):
        """
        Initialize progressive resolution scheduler.
        
        Args:
            resolution_schedule: List of (height, width) tuples for each stage
            epochs_per_stage: Number of epochs to train at each resolution
            transition_epochs: Number of epochs for smooth transition between stages
        """
        self.resolution_schedule = resolution_schedule
        self.epochs_per_stage = epochs_per_stage
        self.transition_epochs = transition_epochs
        
        if len(resolution_schedule) != len(epochs_per_stage):
            raise ValueError("Resolution schedule and epochs per stage must have same length")
        
        self.current_stage = 0
        self.current_epoch = 0
        self.total_stages = len(resolution_schedule)
        self.logger = logging.getLogger(__name__)
        
        # Compute stage boundaries
        self.stage_boundaries = []
        cumulative_epochs = 0
        for epochs in epochs_per_stage:
            cumulative_epochs += epochs
            self.stage_boundaries.append(cumulative_epochs)
        
        self.logger.info(f"Initialized progressive scheduler with {self.total_stages} stages")
        for i, (res, epochs) in enumerate(zip(resolution_schedule, epochs_per_stage)):
            self.logger.info(f"Stage {i}: {res} resolution for {epochs} epochs")
    
    def get_current_resolution(self) -> Tuple[int, int]:
        """Get current training resolution."""
        return self.resolution_schedule[self.current_stage]
    
    def get_current_stage(self) -> int:
        """Get current training stage."""
        return self.current_stage
    
    def step(self) -> bool:
        """
        Step the scheduler to next epoch.
        
        Returns:
            True if stage changed, False otherwise
        """
        self.current_epoch += 1
        
        # Check if we need to advance to next stage
        if (self.current_stage < self.total_stages - 1 and 
            self.current_epoch >= self.stage_boundaries[self.current_stage]):
            
            old_stage = self.current_stage
            self.current_stage += 1
            
            self.logger.info(f"Advanced from stage {old_stage} to stage {self.current_stage}")
            self.logger.info(f"New resolution: {self.get_current_resolution()}")
            
            return True
        
        return False
    
    def is_in_transition(self) -> bool:
        """Check if currently in transition between stages."""
        if self.current_stage == 0:
            return False
        
        stage_start_epoch = self.stage_boundaries[self.current_stage - 1]
        epochs_in_stage = self.current_epoch - stage_start_epoch
        
        return epochs_in_stage <= self.transition_epochs
    
    def get_transition_weight(self) -> float:
        """
        Get transition weight for blending between stages.
        
        Returns:
            Weight between 0 (previous stage) and 1 (current stage)
        """
        if not self.is_in_transition():
            return 1.0
        
        stage_start_epoch = self.stage_boundaries[self.current_stage - 1]
        epochs_in_stage = self.current_epoch - stage_start_epoch
        
        return epochs_in_stage / self.transition_epochs


class ResolutionTransform:
    """
    Transform for changing resolution of Ising configurations.
    
    Handles upsampling and downsampling while preserving spin structure.
    """
    
    def __init__(self, target_size: Tuple[int, int], method: str = 'nearest'):
        """
        Initialize resolution transform.
        
        Args:
            target_size: Target (height, width) size
            method: Interpolation method ('nearest', 'bilinear', 'area')
        """
        self.target_size = target_size
        self.method = method
        self.logger = logging.getLogger(__name__)
        
        if method not in ['nearest', 'bilinear', 'area']:
            raise ValueError(f"Unknown interpolation method: {method}")
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply resolution transform.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Transformed tensor with target resolution
        """
        if x.shape[-2:] == self.target_size:
            return x
        
        # Use appropriate interpolation
        if self.method == 'nearest':
            # Preserve discrete spin values
            return F.interpolate(x, size=self.target_size, mode='nearest')
        elif self.method == 'bilinear':
            # Smooth interpolation
            result = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
            # Optionally threshold to preserve spin-like values
            return torch.tanh(result * 2)  # Soft thresholding
        elif self.method == 'area':
            # Area-based downsampling
            return F.interpolate(x, size=self.target_size, mode='area')
        else:
            raise ValueError(f"Unknown method: {self.method}")


class ProgressiveDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that provides progressive resolution training data.
    """
    
    def __init__(self,
                 base_dataset: torch.utils.data.Dataset,
                 scheduler: ProgressiveResolutionScheduler,
                 interpolation_method: str = 'nearest'):
        """
        Initialize progressive dataset.
        
        Args:
            base_dataset: Base dataset with full resolution data
            scheduler: Progressive resolution scheduler
            interpolation_method: Method for resolution changes
        """
        self.base_dataset = base_dataset
        self.scheduler = scheduler
        self.interpolation_method = interpolation_method
        self.logger = logging.getLogger(__name__)
        
        # Cache transforms for efficiency
        self.transform_cache = {}
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Any:
        """Get item with current resolution."""
        item = self.base_dataset[idx]
        
        # Get current resolution
        current_resolution = self.scheduler.get_current_resolution()
        
        # Handle both single tensor and tuple returns
        if isinstance(item, tuple):
            data, *rest = item
            data = self._transform_resolution(data, current_resolution)
            return (data, *rest)
        else:
            return self._transform_resolution(item, current_resolution)
    
    def _transform_resolution(self, data: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """Transform data to target resolution."""
        # Check cache first
        cache_key = (tuple(data.shape), target_size, self.interpolation_method)
        
        if cache_key not in self.transform_cache:
            self.transform_cache[cache_key] = ResolutionTransform(
                target_size, self.interpolation_method
            )
        
        transform = self.transform_cache[cache_key]
        return transform(data.unsqueeze(0) if len(data.shape) == 3 else data).squeeze(0)


class ProgressiveVAE(nn.Module):
    """
    VAE with progressive training capabilities.
    
    Supports dynamic architecture adaptation for different resolutions.
    """
    
    def __init__(self,
                 base_vae: ConvolutionalVAE,
                 resolution_schedule: List[Tuple[int, int]]):
        """
        Initialize progressive VAE.
        
        Args:
            base_vae: Base VAE model for full resolution
            resolution_schedule: List of training resolutions
        """
        super().__init__()
        self.base_vae = base_vae
        self.resolution_schedule = resolution_schedule
        self.current_resolution = resolution_schedule[-1]  # Start with full resolution
        self.logger = logging.getLogger(__name__)
        
        # Create resolution-specific input transforms
        self.input_transforms = nn.ModuleDict()
        for i, resolution in enumerate(resolution_schedule):
            if resolution != resolution_schedule[-1]:  # Not full resolution
                self.input_transforms[f'stage_{i}'] = ResolutionTransform(
                    resolution_schedule[-1], method='bilinear'
                )
    
    def set_resolution(self, resolution: Tuple[int, int]) -> None:
        """Set current training resolution."""
        if resolution not in self.resolution_schedule:
            raise ValueError(f"Resolution {resolution} not in schedule")
        
        self.current_resolution = resolution
        self.logger.debug(f"Set VAE resolution to {resolution}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with resolution adaptation.
        
        Args:
            x: Input tensor at current resolution
            
        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        # If input is not at full resolution, upscale it
        if x.shape[-2:] != self.resolution_schedule[-1]:
            # Find appropriate transform
            stage_idx = None
            for i, res in enumerate(self.resolution_schedule):
                if res == x.shape[-2:]:
                    stage_idx = i
                    break
            
            if stage_idx is not None and f'stage_{stage_idx}' in self.input_transforms:
                x = self.input_transforms[f'stage_{stage_idx}'](x)
        
        # Forward through base VAE
        reconstruction, mu, logvar = self.base_vae(x)
        
        # If we need to downsample reconstruction to match input resolution
        if reconstruction.shape[-2:] != self.current_resolution:
            transform = ResolutionTransform(self.current_resolution, method='bilinear')
            reconstruction = transform(reconstruction)
        
        return reconstruction, mu, logvar
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode with resolution adaptation."""
        # Upscale if needed
        if x.shape[-2:] != self.resolution_schedule[-1]:
            transform = ResolutionTransform(self.resolution_schedule[-1], method='bilinear')
            x = transform(x)
        
        return self.base_vae.encode(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode with resolution adaptation."""
        reconstruction = self.base_vae.decode(z)
        
        # Downsample if needed
        if reconstruction.shape[-2:] != self.current_resolution:
            transform = ResolutionTransform(self.current_resolution, method='bilinear')
            reconstruction = transform(reconstruction)
        
        return reconstruction


class ProgressiveTrainer:
    """
    Trainer for progressive resolution training.
    """
    
    def __init__(self,
                 model: ConvolutionalVAE,
                 config: PrometheusConfig,
                 resolution_schedule: Optional[List[Tuple[int, int]]] = None,
                 epochs_per_stage: Optional[List[int]] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize progressive trainer.
        
        Args:
            model: Base VAE model
            config: Configuration object
            resolution_schedule: List of training resolutions
            epochs_per_stage: Epochs to train at each resolution
            device: Training device
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Default resolution schedule if not provided
        if resolution_schedule is None:
            base_size = config.vae.input_shape[1:]  # Assume square
            resolution_schedule = [
                (base_size[0] // 4, base_size[1] // 4),  # 8x8
                (base_size[0] // 2, base_size[1] // 2),  # 16x16
                base_size  # 32x32
            ]
        
        if epochs_per_stage is None:
            total_epochs = config.training.num_epochs
            epochs_per_stage = [
                total_epochs // 4,  # 25% at lowest resolution
                total_epochs // 4,  # 25% at medium resolution
                total_epochs // 2   # 50% at full resolution
            ]
        
        # Initialize scheduler
        self.scheduler = ProgressiveResolutionScheduler(
            resolution_schedule, epochs_per_stage
        )
        
        # Create progressive VAE
        self.progressive_vae = ProgressiveVAE(model, resolution_schedule)
        
        # Set up device
        if device is None:
            if config.device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(config.device)
        else:
            self.device = device
        
        self.progressive_vae.to(self.device)
        
        # Create base trainer
        self.base_trainer = VAETrainer(self.progressive_vae, config, self.device)
        
        # Training state
        self.training_history = {}
        self.stage_histories = []
        
        self.logger.info("Progressive trainer initialized")
        self.logger.info(f"Resolution schedule: {resolution_schedule}")
        self.logger.info(f"Epochs per stage: {epochs_per_stage}")
    
    def train(self,
              train_dataset: torch.utils.data.Dataset,
              val_dataset: Optional[torch.utils.data.Dataset] = None,
              batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute progressive training.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size for training
            
        Returns:
            Progressive training results
        """
        if batch_size is None:
            batch_size = self.config.training.batch_size
        
        self.logger.info("Starting progressive training")
        start_time = time.time()
        
        # Create progressive datasets
        progressive_train = ProgressiveDataset(train_dataset, self.scheduler)
        progressive_val = ProgressiveDataset(val_dataset, self.scheduler) if val_dataset else None
        
        # Training loop
        current_stage = -1
        stage_start_epoch = 0
        
        for epoch in range(sum(self.scheduler.epochs_per_stage)):
            # Check for stage change
            stage_changed = self.scheduler.step()
            
            if stage_changed or current_stage == -1:
                current_stage = self.scheduler.get_current_stage()
                current_resolution = self.scheduler.get_current_resolution()
                
                self.logger.info(f"Starting stage {current_stage} with resolution {current_resolution}")
                
                # Update VAE resolution
                self.progressive_vae.set_resolution(current_resolution)
                
                # Create new data loaders for current resolution
                train_loader = torch.utils.data.DataLoader(
                    progressive_train,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True
                )
                
                val_loader = None
                if progressive_val:
                    val_loader = torch.utils.data.DataLoader(
                        progressive_val,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True
                    )
                
                stage_start_epoch = epoch
            
            # Train one epoch
            train_metrics = self.base_trainer.train_epoch(train_loader)
            val_metrics = {}
            if val_loader:
                val_metrics = self.base_trainer.validate_epoch(val_loader)
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch+1}, Stage {current_stage}, "
                f"Resolution {current_resolution}, "
                f"Train Loss: {train_metrics['total_loss']:.6f}"
                + (f", Val Loss: {val_metrics['total_loss']:.6f}" if val_metrics else "")
            )
            
            # Store metrics
            for key, value in train_metrics.items():
                if f'train_{key}' not in self.training_history:
                    self.training_history[f'train_{key}'] = []
                self.training_history[f'train_{key}'].append(value)
            
            for key, value in val_metrics.items():
                if f'val_{key}' not in self.training_history:
                    self.training_history[f'val_{key}'] = []
                self.training_history[f'val_{key}'].append(value)
        
        total_time = time.time() - start_time
        
        results = {
            'training_history': self.training_history,
            'total_time': total_time,
            'resolution_schedule': self.scheduler.resolution_schedule,
            'epochs_per_stage': self.scheduler.epochs_per_stage,
            'final_resolution': self.scheduler.get_current_resolution()
        }
        
        self.logger.info(f"Progressive training completed in {total_time:.1f}s")
        
        return results
    
    def get_model(self) -> ConvolutionalVAE:
        """Get the underlying VAE model."""
        return self.progressive_vae.base_vae


def create_default_progressive_schedule(base_resolution: Tuple[int, int],
                                      n_stages: int = 3) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Create default progressive training schedule.
    
    Args:
        base_resolution: Final training resolution
        n_stages: Number of progressive stages
        
    Returns:
        Tuple of (resolution_schedule, epochs_per_stage)
    """
    if n_stages < 2:
        raise ValueError("Need at least 2 stages for progressive training")
    
    # Create resolution schedule
    resolution_schedule = []
    h, w = base_resolution
    
    for i in range(n_stages):
        # Start from 1/4 resolution and work up
        scale_factor = 2 ** (n_stages - 1 - i)
        stage_h = max(4, h // scale_factor)  # Minimum 4x4
        stage_w = max(4, w // scale_factor)
        resolution_schedule.append((stage_h, stage_w))
    
    # Create epoch schedule (more epochs at higher resolutions)
    total_epochs = 100  # Default
    epochs_per_stage = []
    
    for i in range(n_stages):
        if i == n_stages - 1:  # Final stage gets most epochs
            epochs = total_epochs // 2
        else:
            epochs = total_epochs // (2 * (n_stages - 1))
        epochs_per_stage.append(epochs)
    
    # Adjust to match total
    epochs_per_stage[-1] = total_epochs - sum(epochs_per_stage[:-1])
    
    return resolution_schedule, epochs_per_stage