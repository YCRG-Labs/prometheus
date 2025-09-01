"""
Ensemble Training with Multiple Random Initializations

This module implements ensemble training strategies that train multiple models
with different random initializations and combine their predictions for
improved robustness and physics consistency.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
import time
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from ..models.vae import ConvolutionalVAE
from ..utils.config import PrometheusConfig
from .trainer import VAETrainer
from .advanced_schedulers import create_advanced_scheduler


class EnsembleMember:
    """
    Individual ensemble member with its own model and training state.
    """
    
    def __init__(self, 
                 model: ConvolutionalVAE,
                 trainer: VAETrainer,
                 member_id: int,
                 seed: int):
        """
        Initialize ensemble member.
        
        Args:
            model: VAE model for this ensemble member
            trainer: Trainer for this ensemble member
            member_id: Unique identifier for this member
            seed: Random seed used for initialization
        """
        self.model = model
        self.trainer = trainer
        self.member_id = member_id
        self.seed = seed
        self.training_history = {}
        self.best_val_loss = float('inf')
        self.is_trained = False
        self.logger = logging.getLogger(__name__)
    
    def train(self, 
              train_loader: torch.utils.data.DataLoader,
              val_loader: Optional[torch.utils.data.DataLoader] = None,
              num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Train this ensemble member.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            
        Returns:
            Training history dictionary
        """
        self.logger.info(f"Training ensemble member {self.member_id} with seed {self.seed}")
        
        # Set random seed for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # Train the model
        history = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs
        )
        
        self.training_history = history
        self.best_val_loss = self.trainer.best_val_loss
        self.is_trained = True
        
        self.logger.info(f"Ensemble member {self.member_id} training complete. "
                        f"Best val loss: {self.best_val_loss:.6f}")
        
        return history
    
    def encode(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode data using this ensemble member.
        
        Args:
            data: Input data tensor
            
        Returns:
            Tuple of (latent_mean, latent_logvar)
        """
        if not self.is_trained:
            raise RuntimeError(f"Ensemble member {self.member_id} has not been trained")
        
        self.model.eval()
        with torch.no_grad():
            mu, logvar = self.model.encode(data)
        
        return mu, logvar
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation using this ensemble member.
        
        Args:
            latent: Latent representation tensor
            
        Returns:
            Decoded reconstruction
        """
        if not self.is_trained:
            raise RuntimeError(f"Ensemble member {self.member_id} has not been trained")
        
        self.model.eval()
        with torch.no_grad():
            reconstruction = self.model.decode(latent)
        
        return reconstruction
    
    def save(self, save_path: str) -> None:
        """Save ensemble member state."""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'member_id': self.member_id,
            'seed': self.seed,
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'is_trained': self.is_trained
        }
        
        torch.save(save_dict, save_path)
        self.logger.info(f"Ensemble member {self.member_id} saved to {save_path}")
    
    def load(self, load_path: str) -> None:
        """Load ensemble member state."""
        checkpoint = torch.load(load_path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.member_id = checkpoint['member_id']
        self.seed = checkpoint['seed']
        self.training_history = checkpoint['training_history']
        self.best_val_loss = checkpoint['best_val_loss']
        self.is_trained = checkpoint['is_trained']
        
        self.logger.info(f"Ensemble member {self.member_id} loaded from {load_path}")


class EnsembleTrainer:
    """
    Ensemble trainer that manages multiple VAE models with different initializations.
    """
    
    def __init__(self, 
                 config: PrometheusConfig,
                 n_members: int = 5,
                 base_seed: int = 42,
                 device: Optional[torch.device] = None):
        """
        Initialize ensemble trainer.
        
        Args:
            config: Configuration object
            n_members: Number of ensemble members
            base_seed: Base seed for generating member seeds
            device: Device to train on
        """
        self.config = config
        self.n_members = n_members
        self.base_seed = base_seed
        self.logger = logging.getLogger(__name__)
        
        # Set up device
        if device is None:
            if config.device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(config.device)
        else:
            self.device = device
        
        # Initialize ensemble members
        self.members: List[EnsembleMember] = []
        self._initialize_members()
        
        # Training state
        self.is_trained = False
        self.ensemble_history = {}
        
        # Create output directory
        self.ensemble_dir = Path(config.models_dir) / "ensemble"
        self.ensemble_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized ensemble trainer with {n_members} members")
    
    def _initialize_members(self) -> None:
        """Initialize ensemble members with different random seeds."""
        for i in range(self.n_members):
            # Generate unique seed for this member
            member_seed = self.base_seed + i * 1000
            
            # Set seed for model initialization
            torch.manual_seed(member_seed)
            np.random.seed(member_seed)
            
            # Create model
            model = ConvolutionalVAE(
                input_shape=self.config.vae.input_shape,
                latent_dim=self.config.vae.latent_dim,
                encoder_channels=self.config.vae.encoder_channels,
                decoder_channels=self.config.vae.decoder_channels,
                beta=self.config.vae.beta
            )
            
            # Create trainer
            trainer = VAETrainer(model, self.config, self.device)
            
            # Create ensemble member
            member = EnsembleMember(model, trainer, i, member_seed)
            self.members.append(member)
            
            self.logger.debug(f"Initialized ensemble member {i} with seed {member_seed}")
    
    def train_sequential(self,
                        train_loader: torch.utils.data.DataLoader,
                        val_loader: Optional[torch.utils.data.DataLoader] = None,
                        num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Train ensemble members sequentially.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs per member
            
        Returns:
            Ensemble training results
        """
        self.logger.info("Starting sequential ensemble training")
        start_time = time.time()
        
        member_histories = []
        
        for i, member in enumerate(self.members):
            self.logger.info(f"Training ensemble member {i+1}/{self.n_members}")
            
            member_start_time = time.time()
            history = member.train(train_loader, val_loader, num_epochs)
            member_time = time.time() - member_start_time
            
            member_histories.append(history)
            
            self.logger.info(f"Member {i+1} completed in {member_time:.1f}s, "
                           f"best val loss: {member.best_val_loss:.6f}")
        
        total_time = time.time() - start_time
        self.is_trained = True
        
        # Aggregate results
        results = self._aggregate_training_results(member_histories, total_time)
        self.ensemble_history = results
        
        self.logger.info(f"Sequential ensemble training completed in {total_time:.1f}s")
        
        return results
    
    def train_parallel(self,
                      train_loader: torch.utils.data.DataLoader,
                      val_loader: Optional[torch.utils.data.DataLoader] = None,
                      num_epochs: Optional[int] = None,
                      max_workers: Optional[int] = None) -> Dict[str, Any]:
        """
        Train ensemble members in parallel (if multiple GPUs available).
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs per member
            max_workers: Maximum number of parallel workers
            
        Returns:
            Ensemble training results
        """
        if max_workers is None:
            max_workers = min(self.n_members, mp.cpu_count())
        
        self.logger.info(f"Starting parallel ensemble training with {max_workers} workers")
        
        # For now, fall back to sequential training as parallel GPU training
        # requires more complex setup with multiple processes
        self.logger.warning("Parallel training not fully implemented, using sequential")
        return self.train_sequential(train_loader, val_loader, num_epochs)
    
    def _aggregate_training_results(self, 
                                  member_histories: List[Dict[str, List[float]]],
                                  total_time: float) -> Dict[str, Any]:
        """Aggregate training results from all ensemble members."""
        results = {
            'member_histories': member_histories,
            'total_training_time': total_time,
            'n_members': self.n_members,
            'member_best_losses': [member.best_val_loss for member in self.members],
            'ensemble_best_loss': min(member.best_val_loss for member in self.members),
            'ensemble_mean_loss': np.mean([member.best_val_loss for member in self.members]),
            'ensemble_std_loss': np.std([member.best_val_loss for member in self.members])
        }
        
        # Compute ensemble statistics for each metric
        if member_histories:
            all_metrics = set()
            for history in member_histories:
                all_metrics.update(history.keys())
            
            ensemble_stats = {}
            for metric in all_metrics:
                metric_values = []
                for history in member_histories:
                    if metric in history and history[metric]:
                        metric_values.append(history[metric][-1])  # Final value
                
                if metric_values:
                    ensemble_stats[f'{metric}_mean'] = np.mean(metric_values)
                    ensemble_stats[f'{metric}_std'] = np.std(metric_values)
                    ensemble_stats[f'{metric}_min'] = np.min(metric_values)
                    ensemble_stats[f'{metric}_max'] = np.max(metric_values)
            
            results['ensemble_stats'] = ensemble_stats
        
        return results
    
    def predict_ensemble(self, 
                        data: torch.Tensor,
                        return_uncertainty: bool = True) -> Dict[str, torch.Tensor]:
        """
        Make ensemble predictions with uncertainty estimation.
        
        Args:
            data: Input data tensor
            return_uncertainty: Whether to compute uncertainty estimates
            
        Returns:
            Dictionary with ensemble predictions and uncertainties
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble has not been trained")
        
        data = data.to(self.device)
        
        # Collect predictions from all members
        reconstructions = []
        latent_means = []
        latent_logvars = []
        
        for member in self.members:
            member.model.eval()
            with torch.no_grad():
                recon, mu, logvar = member.model(data)
                reconstructions.append(recon)
                latent_means.append(mu)
                latent_logvars.append(logvar)
        
        # Stack predictions
        reconstructions = torch.stack(reconstructions, dim=0)  # [n_members, batch_size, ...]
        latent_means = torch.stack(latent_means, dim=0)
        latent_logvars = torch.stack(latent_logvars, dim=0)
        
        # Compute ensemble statistics
        results = {
            'reconstruction_mean': torch.mean(reconstructions, dim=0),
            'latent_mean': torch.mean(latent_means, dim=0),
            'latent_logvar_mean': torch.mean(latent_logvars, dim=0)
        }
        
        if return_uncertainty:
            results.update({
                'reconstruction_std': torch.std(reconstructions, dim=0),
                'reconstruction_var': torch.var(reconstructions, dim=0),
                'latent_mean_std': torch.std(latent_means, dim=0),
                'latent_logvar_std': torch.std(latent_logvars, dim=0),
                'epistemic_uncertainty': torch.var(latent_means, dim=0),  # Model uncertainty
                'aleatoric_uncertainty': torch.mean(torch.exp(latent_logvars), dim=0)  # Data uncertainty
            })
        
        return results
    
    def encode_ensemble(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode data using ensemble with uncertainty estimation.
        
        Args:
            data: Input data tensor
            
        Returns:
            Dictionary with ensemble encoding results
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble has not been trained")
        
        data = data.to(self.device)
        
        # Collect encodings from all members
        latent_means = []
        latent_logvars = []
        
        for member in self.members:
            mu, logvar = member.encode(data)
            latent_means.append(mu)
            latent_logvars.append(logvar)
        
        # Stack and compute statistics
        latent_means = torch.stack(latent_means, dim=0)
        latent_logvars = torch.stack(latent_logvars, dim=0)
        
        return {
            'latent_mean': torch.mean(latent_means, dim=0),
            'latent_std': torch.std(latent_means, dim=0),
            'latent_logvar_mean': torch.mean(latent_logvars, dim=0),
            'epistemic_uncertainty': torch.var(latent_means, dim=0),
            'aleatoric_uncertainty': torch.mean(torch.exp(latent_logvars), dim=0),
            'total_uncertainty': torch.var(latent_means, dim=0) + torch.mean(torch.exp(latent_logvars), dim=0)
        }
    
    def get_best_member(self) -> EnsembleMember:
        """Get the ensemble member with the best validation loss."""
        if not self.is_trained:
            raise RuntimeError("Ensemble has not been trained")
        
        best_member = min(self.members, key=lambda m: m.best_val_loss)
        return best_member
    
    def get_member_diversity(self) -> Dict[str, float]:
        """
        Compute diversity metrics for ensemble members.
        
        Returns:
            Dictionary with diversity statistics
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble has not been trained")
        
        # Compute pairwise differences in final losses
        losses = [member.best_val_loss for member in self.members]
        
        diversity_metrics = {
            'loss_std': np.std(losses),
            'loss_range': np.max(losses) - np.min(losses),
            'loss_cv': np.std(losses) / np.mean(losses) if np.mean(losses) > 0 else 0
        }
        
        return diversity_metrics
    
    def save_ensemble(self, save_dir: Optional[str] = None) -> str:
        """
        Save entire ensemble to directory.
        
        Args:
            save_dir: Directory to save ensemble (default: auto-generated)
            
        Returns:
            Path where ensemble was saved
        """
        if save_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_dir = str(self.ensemble_dir / f"ensemble_{timestamp}")
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual members
        for member in self.members:
            member_path = save_path / f"member_{member.member_id}.pth"
            member.save(str(member_path))
        
        # Save ensemble metadata
        ensemble_metadata = {
            'n_members': self.n_members,
            'base_seed': self.base_seed,
            'config': self.config,
            'is_trained': self.is_trained,
            'ensemble_history': self.ensemble_history
        }
        
        metadata_path = save_path / "ensemble_metadata.pth"
        torch.save(ensemble_metadata, metadata_path)
        
        self.logger.info(f"Ensemble saved to {save_path}")
        return str(save_path)
    
    def load_ensemble(self, load_dir: str) -> None:
        """
        Load ensemble from directory.
        
        Args:
            load_dir: Directory containing saved ensemble
        """
        load_path = Path(load_dir)
        
        # Load ensemble metadata
        metadata_path = load_path / "ensemble_metadata.pth"
        metadata = torch.load(metadata_path, map_location='cpu')
        
        self.n_members = metadata['n_members']
        self.base_seed = metadata['base_seed']
        self.is_trained = metadata['is_trained']
        self.ensemble_history = metadata['ensemble_history']
        
        # Load individual members
        self.members = []
        for i in range(self.n_members):
            member_path = load_path / f"member_{i}.pth"
            
            # Create new member
            member_seed = self.base_seed + i * 1000
            torch.manual_seed(member_seed)
            
            model = ConvolutionalVAE(
                input_shape=self.config.vae.input_shape,
                latent_dim=self.config.vae.latent_dim,
                encoder_channels=self.config.vae.encoder_channels,
                decoder_channels=self.config.vae.decoder_channels,
                beta=self.config.vae.beta
            )
            
            trainer = VAETrainer(model, self.config, self.device)
            member = EnsembleMember(model, trainer, i, member_seed)
            
            # Load member state
            member.load(str(member_path))
            self.members.append(member)
        
        self.logger.info(f"Ensemble loaded from {load_path}")


def create_ensemble_with_different_architectures(config: PrometheusConfig,
                                               architecture_variants: List[Dict[str, Any]],
                                               device: Optional[torch.device] = None) -> EnsembleTrainer:
    """
    Create ensemble with different architecture variants.
    
    Args:
        config: Base configuration
        architecture_variants: List of architecture modifications
        device: Device to use
        
    Returns:
        EnsembleTrainer with diverse architectures
    """
    # This would require modifying EnsembleTrainer to support different architectures
    # For now, return standard ensemble
    return EnsembleTrainer(config, len(architecture_variants), device=device)