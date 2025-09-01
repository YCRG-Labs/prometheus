"""
Advanced Learning Rate Schedulers for Enhanced Training

This module implements advanced learning rate scheduling strategies including
cosine annealing with warm restarts, custom decay functions, and adaptive
scheduling based on training metrics.
"""

import math
import torch
import torch.optim as optim
from typing import Dict, List, Optional, Callable, Any
import numpy as np
import logging


class CosineAnnealingWarmRestarts(optim.lr_scheduler._LRScheduler):
    """
    Cosine Annealing with Warm Restarts scheduler.
    
    Implements the SGDR (Stochastic Gradient Descent with Warm Restarts) algorithm
    with cosine annealing learning rate schedule and periodic restarts.
    """
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_0: int,
                 T_mult: int = 1,
                 eta_min: float = 0,
                 last_epoch: int = -1,
                 verbose: bool = False):
        """
        Initialize cosine annealing with warm restarts scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            T_0: Number of iterations for the first restart
            T_mult: Factor to increase T_i after each restart
            eta_min: Minimum learning rate
            last_epoch: The index of last epoch
            verbose: If True, prints a message to stdout for each update
        """
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        self.logger = logging.getLogger(__name__)
        
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for current epoch."""
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_i:
            return [
                self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs
            ]
        else:
            return [self.eta_min for _ in self.base_lrs]
    
    def step(self, epoch: Optional[int] = None) -> None:
        """Step the scheduler."""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.T_i = self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        
        self.last_epoch = epoch
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.
    
    Combines linear warmup with cosine annealing for improved training stability.
    """
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_epochs: int,
                 max_epochs: int,
                 eta_min: float = 0,
                 last_epoch: int = -1):
        """
        Initialize warmup cosine scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            warmup_epochs: Number of warmup epochs
            max_epochs: Total number of training epochs
            eta_min: Minimum learning rate
            last_epoch: The index of last epoch
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [
                self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]


class CyclicLRScheduler(optim.lr_scheduler._LRScheduler):
    """
    Cyclic Learning Rate scheduler.
    
    Implements cyclic learning rates that oscillate between base and max learning rates.
    """
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 base_lr: float,
                 max_lr: float,
                 step_size_up: int,
                 step_size_down: Optional[int] = None,
                 mode: str = 'triangular',
                 gamma: float = 1.0,
                 scale_fn: Optional[Callable] = None,
                 scale_mode: str = 'cycle',
                 last_epoch: int = -1):
        """
        Initialize cyclic learning rate scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            base_lr: Lower boundary of learning rate
            max_lr: Upper boundary of learning rate
            step_size_up: Number of training iterations in the increasing half of a cycle
            step_size_down: Number of training iterations in the decreasing half of a cycle
            mode: One of 'triangular', 'triangular2', 'exp_range'
            gamma: Constant in 'exp_range' scaling function
            scale_fn: Custom scaling function
            scale_mode: 'cycle' or 'iterations'
            last_epoch: The index of last epoch
        """
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.mode = mode
        self.gamma = gamma
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        
        self.cycle_size = self.step_size_up + self.step_size_down
        
        super(CyclicLRScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for current epoch."""
        cycle = math.floor(1 + self.last_epoch / self.cycle_size)
        x = 1 + self.last_epoch / self.cycle_size - cycle
        
        if x <= self.step_size_up / self.cycle_size:
            scale_factor = x * self.cycle_size / self.step_size_up
        else:
            scale_factor = (x * self.cycle_size - self.step_size_up) / self.step_size_down
            scale_factor = 1 - scale_factor
        
        if self.scale_fn is None:
            if self.mode == 'triangular':
                scale_factor = scale_factor
            elif self.mode == 'triangular2':
                scale_factor = scale_factor / (2 ** (cycle - 1))
            elif self.mode == 'exp_range':
                scale_factor = scale_factor * (self.gamma ** self.last_epoch)
        else:
            if self.scale_mode == 'cycle':
                scale_factor = self.scale_fn(cycle)
            else:
                scale_factor = self.scale_fn(self.last_epoch)
        
        return [self.base_lr + (self.max_lr - self.base_lr) * scale_factor 
                for _ in self.base_lrs]


class AdaptiveLRScheduler:
    """
    Adaptive learning rate scheduler based on training metrics.
    
    Adjusts learning rate based on loss plateaus, gradient norms, and other metrics.
    """
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 patience: int = 10,
                 factor: float = 0.5,
                 threshold: float = 1e-4,
                 cooldown: int = 0,
                 min_lr: float = 0,
                 eps: float = 1e-8,
                 verbose: bool = False):
        """
        Initialize adaptive scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            patience: Number of epochs with no improvement after which learning rate will be reduced
            factor: Factor by which the learning rate will be reduced
            threshold: Threshold for measuring the new optimum
            cooldown: Number of epochs to wait before resuming normal operation
            min_lr: Lower bound on the learning rate
            eps: Minimal decay applied to lr
            verbose: If True, prints a message to stdout for each update
        """
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.threshold = threshold
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        self.verbose = verbose
        
        self.best = None
        self.num_bad_epochs = 0
        self.mode_worse = None
        self.is_better = None
        self.cooldown_counter = 0
        self.last_epoch = 0
        
        self._init_is_better(mode='min', threshold=threshold, threshold_mode='rel')
        self.logger = logging.getLogger(__name__)
    
    def _init_is_better(self, mode: str, threshold: float, threshold_mode: str) -> None:
        """Initialize comparison functions."""
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')
        
        if mode == 'min':
            self.mode_worse = float('inf')
        else:
            self.mode_worse = -float('inf')
        
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        
        if threshold_mode == 'rel':
            rel_epsilon = 1. - threshold if mode == 'min' else 1. + threshold
            self.is_better = lambda a, best: a < best * rel_epsilon
        else:
            self.is_better = lambda a, best: a < best - threshold if mode == 'min' else a > best + threshold
    
    def step(self, metrics: float, epoch: Optional[int] = None) -> None:
        """Step the scheduler with current metrics."""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        if self.is_better(metrics, self.best):
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
        
        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
    
    def _reduce_lr(self, epoch: int) -> None:
        """Reduce learning rate."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    self.logger.info(f'Epoch {epoch}: reducing learning rate of group {i} to {new_lr:.4e}.')
    
    @property
    def in_cooldown(self) -> bool:
        """Check if scheduler is in cooldown period."""
        return self.cooldown_counter > 0
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state dictionary."""
        return {
            'best': self.best,
            'num_bad_epochs': self.num_bad_epochs,
            'cooldown_counter': self.cooldown_counter,
            'last_epoch': self.last_epoch
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary."""
        self.best = state_dict['best']
        self.num_bad_epochs = state_dict['num_bad_epochs']
        self.cooldown_counter = state_dict['cooldown_counter']
        self.last_epoch = state_dict['last_epoch']


def create_advanced_scheduler(optimizer: torch.optim.Optimizer,
                            scheduler_type: str,
                            **kwargs) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Factory function to create advanced learning rate schedulers.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler to create
        **kwargs: Additional arguments for the scheduler
        
    Returns:
        Configured learning rate scheduler
    """
    if scheduler_type == 'cosine_warm_restarts':
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get('T_0', 10),
            T_mult=kwargs.get('T_mult', 2),
            eta_min=kwargs.get('eta_min', 0),
            verbose=kwargs.get('verbose', False)
        )
    
    elif scheduler_type == 'warmup_cosine':
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=kwargs.get('warmup_epochs', 5),
            max_epochs=kwargs.get('max_epochs', 100),
            eta_min=kwargs.get('eta_min', 0)
        )
    
    elif scheduler_type == 'cyclic':
        return CyclicLRScheduler(
            optimizer,
            base_lr=kwargs.get('base_lr', 1e-5),
            max_lr=kwargs.get('max_lr', 1e-2),
            step_size_up=kwargs.get('step_size_up', 2000),
            mode=kwargs.get('mode', 'triangular')
        )
    
    elif scheduler_type == 'adaptive':
        return AdaptiveLRScheduler(
            optimizer,
            patience=kwargs.get('patience', 10),
            factor=kwargs.get('factor', 0.5),
            threshold=kwargs.get('threshold', 1e-4),
            verbose=kwargs.get('verbose', False)
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")