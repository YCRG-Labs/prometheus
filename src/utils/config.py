"""
Configuration loading and validation utilities for the Prometheus project.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field


@dataclass
class IsingConfig:
    """Configuration for Ising model simulation."""
    lattice_size: tuple = (32, 32)
    temperature_range: tuple = (1.5, 3.0)
    n_temperatures: int = 100
    critical_temp: float = 2.269
    n_configs_per_temp: int = 1000
    equilibration_steps: int = 10000
    measurement_steps: int = 1000


@dataclass
class VAEConfig:
    """Configuration for VAE model architecture."""
    input_shape: tuple = (1, 32, 32)
    latent_dim: int = 2
    encoder_channels: list = field(default_factory=lambda: [32, 64, 128])
    decoder_channels: list = field(default_factory=lambda: [128, 64, 32, 1])
    kernel_sizes: list = field(default_factory=lambda: [3, 3, 3])
    beta: float = 1.0


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    batch_size: int = 128
    learning_rate: float = 1e-3
    num_epochs: int = 100
    optimizer: str = "Adam"
    scheduler: str = "ReduceLROnPlateau"
    early_stopping_patience: int = 10
    checkpoint_interval: int = 10
    
    # Advanced scheduler parameters
    advanced_scheduler: str = "none"  # "cosine_warm_restarts", "warmup_cosine", "cyclic", "adaptive"
    scheduler_params: dict = field(default_factory=dict)
    
    # Data augmentation parameters
    use_augmentation: bool = True
    augmentation_type: str = "standard"  # "standard", "conservative", "aggressive"
    augmentation_probability: float = 0.5
    
    # Ensemble training parameters
    use_ensemble: bool = False
    ensemble_size: int = 5
    ensemble_base_seed: int = 42
    
    # Progressive training parameters
    use_progressive: bool = False
    progressive_stages: int = 3
    progressive_epochs_ratio: list = field(default_factory=lambda: [0.25, 0.25, 0.5])


@dataclass
class LoggingConfig:
    """Configuration for logging system."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_output: bool = True
    console_output: bool = True
    log_dir: str = "logs"


@dataclass
class PrometheusConfig:
    """Main configuration container for the Prometheus project."""
    ising: IsingConfig = field(default_factory=IsingConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda"
    data_dir: str = "data"
    results_dir: str = "results"
    models_dir: str = "models"


class ConfigLoader:
    """Utility class for loading and validating YAML configuration files."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(__name__)
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> PrometheusConfig:
        """
        Load configuration from YAML file or use defaults.
        
        Args:
            config_path: Path to YAML configuration file. If None, uses default config.
            
        Returns:
            PrometheusConfig: Validated configuration object.
        """
        if config_path is None:
            self.logger.info("No config file specified, using default configuration")
            return PrometheusConfig()
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            self.logger.info(f"Loaded configuration from {config_path}")
            return self._dict_to_config(config_dict)
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> PrometheusConfig:
        """Convert dictionary to PrometheusConfig with validation."""
        config = PrometheusConfig()
        
        # Update Ising configuration
        if 'ising' in config_dict:
            ising_dict = config_dict['ising']
            config.ising = IsingConfig(
                lattice_size=tuple(ising_dict.get('lattice_size', config.ising.lattice_size)),
                temperature_range=tuple(ising_dict.get('temperature_range', config.ising.temperature_range)),
                n_temperatures=ising_dict.get('n_temperatures', config.ising.n_temperatures),
                critical_temp=ising_dict.get('critical_temp', config.ising.critical_temp),
                n_configs_per_temp=ising_dict.get('n_configs_per_temp', config.ising.n_configs_per_temp),
                equilibration_steps=ising_dict.get('equilibration_steps', config.ising.equilibration_steps),
                measurement_steps=ising_dict.get('measurement_steps', config.ising.measurement_steps)
            )
        
        # Update VAE configuration
        if 'vae' in config_dict:
            vae_dict = config_dict['vae']
            config.vae = VAEConfig(
                input_shape=tuple(vae_dict.get('input_shape', config.vae.input_shape)),
                latent_dim=vae_dict.get('latent_dim', config.vae.latent_dim),
                encoder_channels=vae_dict.get('encoder_channels', config.vae.encoder_channels),
                decoder_channels=vae_dict.get('decoder_channels', config.vae.decoder_channels),
                kernel_sizes=vae_dict.get('kernel_sizes', config.vae.kernel_sizes),
                beta=vae_dict.get('beta', config.vae.beta)
            )
        
        # Update Training configuration
        if 'training' in config_dict:
            training_dict = config_dict['training']
            config.training = TrainingConfig(
                batch_size=training_dict.get('batch_size', config.training.batch_size),
                learning_rate=training_dict.get('learning_rate', config.training.learning_rate),
                num_epochs=training_dict.get('num_epochs', config.training.num_epochs),
                optimizer=training_dict.get('optimizer', config.training.optimizer),
                scheduler=training_dict.get('scheduler', config.training.scheduler),
                early_stopping_patience=training_dict.get('early_stopping_patience', config.training.early_stopping_patience),
                checkpoint_interval=training_dict.get('checkpoint_interval', config.training.checkpoint_interval),
                advanced_scheduler=training_dict.get('advanced_scheduler', config.training.advanced_scheduler),
                scheduler_params=training_dict.get('scheduler_params', config.training.scheduler_params),
                use_augmentation=training_dict.get('use_augmentation', config.training.use_augmentation),
                augmentation_type=training_dict.get('augmentation_type', config.training.augmentation_type),
                augmentation_probability=training_dict.get('augmentation_probability', config.training.augmentation_probability),
                use_ensemble=training_dict.get('use_ensemble', config.training.use_ensemble),
                ensemble_size=training_dict.get('ensemble_size', config.training.ensemble_size),
                ensemble_base_seed=training_dict.get('ensemble_base_seed', config.training.ensemble_base_seed),
                use_progressive=training_dict.get('use_progressive', config.training.use_progressive),
                progressive_stages=training_dict.get('progressive_stages', config.training.progressive_stages),
                progressive_epochs_ratio=training_dict.get('progressive_epochs_ratio', config.training.progressive_epochs_ratio)
            )
        
        # Update Logging configuration
        if 'logging' in config_dict:
            logging_dict = config_dict['logging']
            config.logging = LoggingConfig(
                level=logging_dict.get('level', config.logging.level),
                format=logging_dict.get('format', config.logging.format),
                file_output=logging_dict.get('file_output', config.logging.file_output),
                console_output=logging_dict.get('console_output', config.logging.console_output),
                log_dir=logging_dict.get('log_dir', config.logging.log_dir)
            )
        
        # Update global settings
        config.seed = config_dict.get('seed', config.seed)
        config.device = config_dict.get('device', config.device)
        config.data_dir = config_dict.get('data_dir', config.data_dir)
        config.results_dir = config_dict.get('results_dir', config.results_dir)
        config.models_dir = config_dict.get('models_dir', config.models_dir)
        
        self._validate_config(config)
        return config
    
    def _validate_config(self, config: PrometheusConfig) -> None:
        """Validate configuration parameters."""
        # Validate Ising parameters
        if len(config.ising.lattice_size) != 2:
            raise ValueError("lattice_size must be a 2-tuple")
        if config.ising.lattice_size[0] <= 0 or config.ising.lattice_size[1] <= 0:
            raise ValueError("lattice_size dimensions must be positive")
        if config.ising.temperature_range[0] >= config.ising.temperature_range[1]:
            raise ValueError("temperature_range must be (min, max) with min < max")
        if config.ising.n_temperatures <= 0:
            raise ValueError("n_temperatures must be positive")
        if config.ising.n_configs_per_temp <= 0:
            raise ValueError("n_configs_per_temp must be positive")
        
        # Validate VAE parameters
        if len(config.vae.input_shape) != 3:
            raise ValueError("input_shape must be a 3-tuple (channels, height, width)")
        if config.vae.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if config.vae.beta < 0:
            raise ValueError("beta must be non-negative")
        
        # Validate training parameters
        if config.training.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if config.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if config.training.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        
        # Validate logging parameters
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config.logging.level not in valid_levels:
            raise ValueError(f"logging level must be one of {valid_levels}")
        
        self.logger.info("Configuration validation passed")
    
    def save_config(self, config: PrometheusConfig, output_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            'ising': {
                'lattice_size': list(config.ising.lattice_size),
                'temperature_range': list(config.ising.temperature_range),
                'n_temperatures': config.ising.n_temperatures,
                'critical_temp': config.ising.critical_temp,
                'n_configs_per_temp': config.ising.n_configs_per_temp,
                'equilibration_steps': config.ising.equilibration_steps,
                'measurement_steps': config.ising.measurement_steps
            },
            'vae': {
                'input_shape': list(config.vae.input_shape),
                'latent_dim': config.vae.latent_dim,
                'encoder_channels': config.vae.encoder_channels,
                'decoder_channels': config.vae.decoder_channels,
                'kernel_sizes': config.vae.kernel_sizes,
                'beta': config.vae.beta
            },
            'training': {
                'batch_size': config.training.batch_size,
                'learning_rate': config.training.learning_rate,
                'num_epochs': config.training.num_epochs,
                'optimizer': config.training.optimizer,
                'scheduler': config.training.scheduler,
                'early_stopping_patience': config.training.early_stopping_patience,
                'checkpoint_interval': config.training.checkpoint_interval,
                'advanced_scheduler': config.training.advanced_scheduler,
                'scheduler_params': config.training.scheduler_params,
                'use_augmentation': config.training.use_augmentation,
                'augmentation_type': config.training.augmentation_type,
                'augmentation_probability': config.training.augmentation_probability,
                'use_ensemble': config.training.use_ensemble,
                'ensemble_size': config.training.ensemble_size,
                'ensemble_base_seed': config.training.ensemble_base_seed,
                'use_progressive': config.training.use_progressive,
                'progressive_stages': config.training.progressive_stages,
                'progressive_epochs_ratio': config.training.progressive_epochs_ratio
            },
            'logging': {
                'level': config.logging.level,
                'format': config.logging.format,
                'file_output': config.logging.file_output,
                'console_output': config.logging.console_output,
                'log_dir': config.logging.log_dir
            },
            'seed': config.seed,
            'device': config.device,
            'data_dir': config.data_dir,
            'results_dir': config.results_dir,
            'models_dir': config.models_dir
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Configuration saved to {output_path}")