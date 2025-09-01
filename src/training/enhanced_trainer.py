"""
Enhanced Training Pipeline with Advanced Features

This module provides an enhanced training pipeline that integrates all
advanced training features including progressive training, ensemble methods,
data augmentation, and advanced learning rate scheduling.
"""

import torch
import torch.utils.data
import numpy as np
import logging
import copy
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import time

from ..models.vae import ConvolutionalVAE
from ..utils.config import PrometheusConfig
from ..data.preprocessing import IsingDataset
from .data_augmentation import AugmentedDataset
from .trainer import VAETrainer
from .ensemble_training import EnsembleTrainer
from .progressive_training import ProgressiveTrainer, create_default_progressive_schedule
from .data_augmentation import (
    create_standard_augmentation, 
    create_conservative_augmentation, 
    create_aggressive_augmentation
)
from .advanced_schedulers import create_advanced_scheduler


class EnhancedTrainingPipeline:
    """
    Enhanced training pipeline with all advanced features.
    
    Provides a unified interface for training VAE models with:
    - Advanced learning rate scheduling
    - Data augmentation
    - Ensemble training
    - Progressive training
    """
    
    def __init__(self, 
                 config: PrometheusConfig,
                 device: Optional[torch.device] = None):
        """
        Initialize enhanced training pipeline.
        
        Args:
            config: Configuration object with training parameters
            device: Device to train on (auto-detected if None)
        """
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
        
        self.logger.info(f"Enhanced training pipeline initialized on {self.device}")
        
        # Training components
        self.model = None
        self.trainer = None
        self.ensemble_trainer = None
        self.progressive_trainer = None
        
        # Training results
        self.training_results = {}
        
    def create_model(self) -> ConvolutionalVAE:
        """Create VAE model based on configuration."""
        model = ConvolutionalVAE(
            input_shape=self.config.vae.input_shape,
            latent_dim=self.config.vae.latent_dim,
            encoder_channels=self.config.vae.encoder_channels,
            decoder_channels=self.config.vae.decoder_channels,
            beta=self.config.vae.beta
        )
        
        model.to(self.device)
        self.model = model
        
        self.logger.info(f"Created VAE model with {sum(p.numel() for p in model.parameters())} parameters")
        
        return model
    
    def prepare_datasets(self, 
                        train_dataset: torch.utils.data.Dataset,
                        val_dataset: Optional[torch.utils.data.Dataset] = None,
                        test_dataset: Optional[torch.utils.data.Dataset] = None) -> Tuple[
                            torch.utils.data.Dataset, 
                            Optional[torch.utils.data.Dataset], 
                            Optional[torch.utils.data.Dataset]
                        ]:
        """
        Prepare datasets with augmentation if enabled.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            
        Returns:
            Tuple of (prepared_train, prepared_val, prepared_test)
        """
        prepared_train = train_dataset
        prepared_val = val_dataset
        prepared_test = test_dataset
        
        # Apply data augmentation to training set if enabled
        if self.config.training.use_augmentation:
            augmentation_type = self.config.training.augmentation_type
            augmentation_prob = self.config.training.augmentation_probability
            
            if augmentation_type == "standard":
                augmentation = create_standard_augmentation()
            elif augmentation_type == "conservative":
                augmentation = create_conservative_augmentation()
            elif augmentation_type == "aggressive":
                augmentation = create_aggressive_augmentation()
            else:
                self.logger.warning(f"Unknown augmentation type: {augmentation_type}, using standard")
                augmentation = create_standard_augmentation()
            
            prepared_train = AugmentedDataset(
                train_dataset, 
                augmentation, 
                augment_probability=augmentation_prob
            )
            
            self.logger.info(f"Applied {augmentation_type} augmentation with probability {augmentation_prob}")
        
        return prepared_train, prepared_val, prepared_test
    
    def train_standard(self,
                      train_dataset: torch.utils.data.Dataset,
                      val_dataset: Optional[torch.utils.data.Dataset] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Train using standard single-model approach.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            **kwargs: Additional training arguments
            
        Returns:
            Training results dictionary
        """
        self.logger.info("Starting standard training")
        
        # Create model if not exists
        if self.model is None:
            self.create_model()
        
        # Prepare datasets
        train_data, val_data, _ = self.prepare_datasets(train_dataset, val_dataset)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = None
        if val_data is not None:
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                drop_last=False
            )
        
        # Create trainer
        self.trainer = VAETrainer(self.model, self.config, self.device)
        
        # Train model
        start_time = time.time()
        history = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=kwargs.get('num_epochs', self.config.training.num_epochs)
        )
        training_time = time.time() - start_time
        
        results = {
            'method': 'standard',
            'training_history': history,
            'training_time': training_time,
            'best_val_loss': self.trainer.best_val_loss,
            'model': self.model,
            'trainer': self.trainer
        }
        
        self.training_results = results
        self.logger.info(f"Standard training completed in {training_time:.1f}s")
        
        return results
    
    def train_ensemble(self,
                      train_dataset: torch.utils.data.Dataset,
                      val_dataset: Optional[torch.utils.data.Dataset] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Train using ensemble approach.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            **kwargs: Additional training arguments
            
        Returns:
            Ensemble training results
        """
        self.logger.info("Starting ensemble training")
        
        # Prepare datasets
        train_data, val_data, _ = self.prepare_datasets(train_dataset, val_dataset)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = None
        if val_data is not None:
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                drop_last=False
            )
        
        # Create ensemble trainer
        self.ensemble_trainer = EnsembleTrainer(
            config=self.config,
            n_members=self.config.training.ensemble_size,
            base_seed=self.config.training.ensemble_base_seed,
            device=self.device
        )
        
        # Train ensemble
        start_time = time.time()
        ensemble_results = self.ensemble_trainer.train_sequential(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=kwargs.get('num_epochs', self.config.training.num_epochs)
        )
        training_time = time.time() - start_time
        
        results = {
            'method': 'ensemble',
            'ensemble_results': ensemble_results,
            'training_time': training_time,
            'ensemble_trainer': self.ensemble_trainer,
            'best_member': self.ensemble_trainer.get_best_member(),
            'diversity_metrics': self.ensemble_trainer.get_member_diversity()
        }
        
        self.training_results = results
        self.logger.info(f"Ensemble training completed in {training_time:.1f}s")
        
        return results
    
    def train_progressive(self,
                         train_dataset: torch.utils.data.Dataset,
                         val_dataset: Optional[torch.utils.data.Dataset] = None,
                         **kwargs) -> Dict[str, Any]:
        """
        Train using progressive approach.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            **kwargs: Additional training arguments
            
        Returns:
            Progressive training results
        """
        self.logger.info("Starting progressive training")
        
        # Create model if not exists
        if self.model is None:
            self.create_model()
        
        # Prepare datasets (no augmentation wrapper for progressive training)
        train_data = train_dataset
        val_data = val_dataset
        
        # Create progressive schedule
        base_resolution = self.config.vae.input_shape[1:]
        resolution_schedule, epochs_per_stage = create_default_progressive_schedule(
            base_resolution, self.config.training.progressive_stages
        )
        
        # Adjust epochs based on configuration
        total_epochs = kwargs.get('num_epochs', self.config.training.num_epochs)
        epochs_ratio = self.config.training.progressive_epochs_ratio
        
        if len(epochs_ratio) == len(epochs_per_stage):
            epochs_per_stage = [int(total_epochs * ratio) for ratio in epochs_ratio]
            # Ensure total matches
            epochs_per_stage[-1] = total_epochs - sum(epochs_per_stage[:-1])
        
        # Create progressive trainer
        self.progressive_trainer = ProgressiveTrainer(
            model=self.model,
            config=self.config,
            resolution_schedule=resolution_schedule,
            epochs_per_stage=epochs_per_stage,
            device=self.device
        )
        
        # Train progressively
        start_time = time.time()
        progressive_results = self.progressive_trainer.train(
            train_dataset=train_data,
            val_dataset=val_data,
            batch_size=self.config.training.batch_size
        )
        training_time = time.time() - start_time
        
        results = {
            'method': 'progressive',
            'progressive_results': progressive_results,
            'training_time': training_time,
            'progressive_trainer': self.progressive_trainer,
            'model': self.progressive_trainer.get_model(),
            'resolution_schedule': resolution_schedule,
            'epochs_per_stage': epochs_per_stage
        }
        
        self.training_results = results
        self.logger.info(f"Progressive training completed in {training_time:.1f}s")
        
        return results
    
    def train_auto(self,
                   train_dataset: torch.utils.data.Dataset,
                   val_dataset: Optional[torch.utils.data.Dataset] = None,
                   **kwargs) -> Dict[str, Any]:
        """
        Automatically select and execute best training method based on configuration.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            **kwargs: Additional training arguments
            
        Returns:
            Training results from selected method
        """
        self.logger.info("Auto-selecting training method based on configuration")
        
        # Priority: Progressive > Ensemble > Standard
        if self.config.training.use_progressive:
            self.logger.info("Selected progressive training")
            return self.train_progressive(train_dataset, val_dataset, **kwargs)
        elif self.config.training.use_ensemble:
            self.logger.info("Selected ensemble training")
            return self.train_ensemble(train_dataset, val_dataset, **kwargs)
        else:
            self.logger.info("Selected standard training")
            return self.train_standard(train_dataset, val_dataset, **kwargs)
    
    def get_trained_model(self) -> Optional[ConvolutionalVAE]:
        """Get the trained model (best from ensemble if applicable)."""
        if not self.training_results:
            return None
        
        method = self.training_results.get('method')
        
        if method == 'standard' or method == 'progressive':
            return self.training_results.get('model')
        elif method == 'ensemble':
            best_member = self.training_results.get('best_member')
            return best_member.model if best_member else None
        
        return None
    
    def save_results(self, save_dir: str) -> str:
        """
        Save training results to directory.
        
        Args:
            save_dir: Directory to save results
            
        Returns:
            Path where results were saved
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save based on training method
        method = self.training_results.get('method', 'unknown')
        
        if method == 'standard':
            # Save model and trainer
            model_path = save_path / "trained_model.pth"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'training_history': self.training_results['training_history'],
                'config': self.config
            }, model_path)
            
        elif method == 'ensemble':
            # Save ensemble
            ensemble_path = self.ensemble_trainer.save_ensemble(str(save_path / "ensemble"))
            
        elif method == 'progressive':
            # Save progressive model
            model_path = save_path / "progressive_model.pth"
            torch.save({
                'model_state_dict': self.progressive_trainer.get_model().state_dict(),
                'progressive_results': self.training_results['progressive_results'],
                'config': self.config
            }, model_path)
        
        # Save general results
        results_path = save_path / "training_results.pth"
        torch.save(self.training_results, results_path)
        
        self.logger.info(f"Training results saved to {save_path}")
        return str(save_path)
    
    def load_results(self, load_dir: str) -> None:
        """
        Load training results from directory.
        
        Args:
            load_dir: Directory containing saved results
        """
        load_path = Path(load_dir)
        
        # Load general results
        results_path = load_path / "training_results.pth"
        if results_path.exists():
            self.training_results = torch.load(results_path, map_location='cpu')
            
            method = self.training_results.get('method')
            
            if method == 'ensemble' and (load_path / "ensemble").exists():
                # Load ensemble
                self.ensemble_trainer = EnsembleTrainer(
                    self.config, device=self.device
                )
                self.ensemble_trainer.load_ensemble(str(load_path / "ensemble"))
                
            self.logger.info(f"Training results loaded from {load_path}")
        else:
            raise FileNotFoundError(f"No training results found in {load_path}")


def create_enhanced_training_config(base_config: PrometheusConfig,
                                  training_method: str = "auto",
                                  **overrides) -> PrometheusConfig:
    """
    Create enhanced training configuration with optimized settings.
    
    Args:
        base_config: Base configuration to modify
        training_method: Training method ("standard", "ensemble", "progressive", "auto")
        **overrides: Configuration overrides
        
    Returns:
        Enhanced configuration object
    """
    config = copy.deepcopy(base_config)
    
    # Apply method-specific optimizations
    if training_method == "ensemble":
        config.training.use_ensemble = True
        config.training.ensemble_size = overrides.get('ensemble_size', 5)
        config.training.use_augmentation = True
        config.training.augmentation_type = "conservative"
        
    elif training_method == "progressive":
        config.training.use_progressive = True
        config.training.progressive_stages = overrides.get('progressive_stages', 3)
        config.training.use_augmentation = True
        config.training.augmentation_type = "standard"
        
    elif training_method == "auto":
        # Enable multiple methods for auto-selection
        config.training.use_augmentation = True
        config.training.augmentation_type = "standard"
        
    # Apply advanced scheduler if specified
    if 'advanced_scheduler' in overrides:
        config.training.advanced_scheduler = overrides['advanced_scheduler']
        config.training.scheduler_params = overrides.get('scheduler_params', {})
    
    # Apply other overrides
    for key, value in overrides.items():
        if hasattr(config.training, key):
            setattr(config.training, key, value)
    
    return config