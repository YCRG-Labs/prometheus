"""
Architecture Search Space and Optimization Framework

This module defines the search space for VAE architectures and implements
systematic exploration of different configurations to optimize physics
consistency and performance.
"""

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Any, Iterator, Tuple, Optional
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureConfig:
    """Configuration for a specific VAE architecture."""
    latent_dim: int
    beta: float
    encoder_layers: int
    activation: str
    encoder_channels: List[int] = field(default_factory=list)
    decoder_channels: List[int] = field(default_factory=list)
    kernel_sizes: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        """Generate channel configurations based on number of layers."""
        if not self.encoder_channels:
            # Generate encoder channels: start with 32, double each layer
            base_channels = 32
            self.encoder_channels = [base_channels * (2 ** i) for i in range(self.encoder_layers)]
        
        if not self.decoder_channels:
            # Generate decoder channels: reverse of encoder + output channel
            self.decoder_channels = list(reversed(self.encoder_channels)) + [1]
        
        if not self.kernel_sizes:
            # Use 3x3 kernels for all layers
            self.kernel_sizes = [3] * self.encoder_layers
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model initialization."""
        return {
            'latent_dim': self.latent_dim,
            'encoder_channels': self.encoder_channels,
            'decoder_channels': self.decoder_channels,
            'kernel_sizes': self.kernel_sizes,
            'beta': self.beta
        }
    
    def get_activation_function(self) -> nn.Module:
        """Get the activation function module."""
        activation_map = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU()
        }
        
        if self.activation not in activation_map:
            raise ValueError(f"Unsupported activation: {self.activation}")
        
        return activation_map[self.activation]


class ArchitectureSearchSpace:
    """
    Defines the search space for VAE architecture optimization.
    
    Provides systematic exploration of latent dimensions, beta values,
    encoder/decoder depths, and activation functions.
    """
    
    def __init__(
        self,
        latent_dims: List[int] = [2, 4, 8, 16],
        beta_values: List[float] = [0.1, 0.5, 1.0, 2.0, 4.0],
        encoder_layers: List[int] = [3, 4, 5],
        activations: List[str] = ['relu', 'leaky_relu', 'elu']
    ):
        """
        Initialize the architecture search space.
        
        Args:
            latent_dims: List of latent space dimensions to explore
            beta_values: List of beta-VAE parameters to test
            encoder_layers: List of encoder layer depths to try
            activations: List of activation functions to test
        """
        self.latent_dims = latent_dims
        self.beta_values = beta_values
        self.encoder_layers = encoder_layers
        self.activations = activations
        
        self.total_configurations = (
            len(latent_dims) * len(beta_values) * 
            len(encoder_layers) * len(activations)
        )
        
        logger.info(f"Architecture search space initialized with {self.total_configurations} configurations")
        logger.info(f"Latent dims: {latent_dims}")
        logger.info(f"Beta values: {beta_values}")
        logger.info(f"Encoder layers: {encoder_layers}")
        logger.info(f"Activations: {activations}")
    
    def generate_configurations(self) -> Iterator[ArchitectureConfig]:
        """
        Generate all possible architecture configurations.
        
        Yields:
            ArchitectureConfig objects for each combination
        """
        for latent_dim, beta, layers, activation in itertools.product(
            self.latent_dims, self.beta_values, self.encoder_layers, self.activations
        ):
            yield ArchitectureConfig(
                latent_dim=latent_dim,
                beta=beta,
                encoder_layers=layers,
                activation=activation
            )
    
    def get_configuration_by_index(self, index: int) -> ArchitectureConfig:
        """
        Get a specific configuration by index.
        
        Args:
            index: Index of the configuration (0 to total_configurations-1)
            
        Returns:
            ArchitectureConfig for the specified index
        """
        if index >= self.total_configurations:
            raise IndexError(f"Index {index} out of range (max: {self.total_configurations-1})")
        
        # Convert linear index to multi-dimensional indices
        n_activations = len(self.activations)
        n_layers = len(self.encoder_layers)
        n_betas = len(self.beta_values)
        
        activation_idx = index % n_activations
        layer_idx = (index // n_activations) % n_layers
        beta_idx = (index // (n_activations * n_layers)) % n_betas
        latent_idx = index // (n_activations * n_layers * n_betas)
        
        return ArchitectureConfig(
            latent_dim=self.latent_dims[latent_idx],
            beta=self.beta_values[beta_idx],
            encoder_layers=self.encoder_layers[layer_idx],
            activation=self.activations[activation_idx]
        )
    
    def get_baseline_configuration(self) -> ArchitectureConfig:
        """
        Get a baseline configuration for comparison.
        
        Returns:
            Standard configuration with middle-range parameters
        """
        return ArchitectureConfig(
            latent_dim=2,  # Standard 2D latent space
            beta=1.0,      # Standard VAE
            encoder_layers=3,  # Moderate depth
            activation='relu'  # Standard activation
        )
    
    def get_configuration_summary(self, config: ArchitectureConfig) -> str:
        """
        Get a human-readable summary of a configuration.
        
        Args:
            config: Architecture configuration
            
        Returns:
            String summary of the configuration
        """
        return (
            f"LatentDim={config.latent_dim}, Beta={config.beta}, "
            f"Layers={config.encoder_layers}, Activation={config.activation}"
        )


class ArchitectureOptimizer:
    """
    Orchestrates systematic architecture optimization experiments.
    
    Manages the execution of training runs across different architectures
    and tracks performance metrics for comparison.
    """
    
    def __init__(
        self,
        search_space: ArchitectureSearchSpace,
        base_config: Dict[str, Any],
        results_dir: str = "results/architecture_optimization"
    ):
        """
        Initialize the architecture optimizer.
        
        Args:
            search_space: Search space defining architectures to explore
            base_config: Base configuration for training (non-architecture params)
            results_dir: Directory to save optimization results
        """
        self.search_space = search_space
        self.base_config = base_config
        self.results_dir = results_dir
        
        # Results tracking
        self.experiment_results = []
        self.best_config = None
        self.best_score = float('-inf')
        
        logger.info(f"Architecture optimizer initialized")
        logger.info(f"Total configurations to explore: {search_space.total_configurations}")
    
    def create_experiment_config(self, arch_config: ArchitectureConfig) -> Dict[str, Any]:
        """
        Create a complete experiment configuration.
        
        Args:
            arch_config: Architecture configuration
            
        Returns:
            Complete configuration dictionary for training
        """
        config = self.base_config.copy()
        
        # Update VAE architecture parameters
        config['vae'].update(arch_config.to_dict())
        
        # Add architecture-specific training adjustments
        if arch_config.latent_dim > 8:
            # Larger latent spaces may need more training
            config['training']['num_epochs'] = int(config['training']['num_epochs'] * 1.2)
        
        if arch_config.encoder_layers > 4:
            # Deeper networks may need lower learning rates
            config['training']['learning_rate'] *= 0.8
        
        return config
    
    def run_single_experiment(
        self, 
        arch_config: ArchitectureConfig,
        experiment_id: int,
        trainer_factory,
        evaluator
    ) -> Dict[str, Any]:
        """
        Run a single architecture experiment.
        
        Args:
            arch_config: Architecture configuration to test
            experiment_id: Unique identifier for this experiment
            trainer_factory: Function to create trainer instances
            evaluator: Physics consistency evaluator
            
        Returns:
            Dictionary containing experiment results
        """
        logger.info(f"Starting experiment {experiment_id}: {self.search_space.get_configuration_summary(arch_config)}")
        
        try:
            # Create experiment configuration
            exp_config = self.create_experiment_config(arch_config)
            
            # Create and train model
            trainer = trainer_factory(exp_config, arch_config)
            training_history = trainer.train()
            
            # Evaluate physics consistency
            physics_metrics = evaluator.evaluate_model(trainer.model, trainer.device)
            
            # Compile results
            results = {
                'experiment_id': experiment_id,
                'architecture': arch_config.to_dict(),
                'training_metrics': {
                    'final_train_loss': training_history['train_total_loss'][-1],
                    'final_val_loss': training_history['val_total_loss'][-1],
                    'best_val_loss': trainer.best_val_loss,
                    'num_epochs_trained': len(training_history['train_total_loss'])
                },
                'physics_metrics': physics_metrics,
                'overall_score': self._compute_overall_score(training_history, physics_metrics),
                'config_summary': self.search_space.get_configuration_summary(arch_config)
            }
            
            logger.info(f"Experiment {experiment_id} completed. Score: {results['overall_score']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {str(e)}")
            return {
                'experiment_id': experiment_id,
                'architecture': arch_config.to_dict(),
                'error': str(e),
                'overall_score': float('-inf'),
                'config_summary': self.search_space.get_configuration_summary(arch_config)
            }
    
    def _compute_overall_score(
        self, 
        training_history: Dict[str, List[float]], 
        physics_metrics: Dict[str, float]
    ) -> float:
        """
        Compute overall score combining training and physics metrics.
        
        Args:
            training_history: Training metrics history
            physics_metrics: Physics consistency metrics
            
        Returns:
            Overall score (higher is better)
        """
        # Training performance (lower loss is better)
        train_score = 1.0 / (1.0 + training_history['val_total_loss'][-1])
        
        # Physics consistency (higher is better)
        physics_score = physics_metrics.get('overall_physics_score', 0.0)
        
        # Order parameter correlation (higher is better)
        correlation_score = physics_metrics.get('order_parameter_correlation', 0.0)
        
        # Critical temperature accuracy (lower error is better)
        temp_error = physics_metrics.get('critical_temperature_error', 1.0)
        temp_score = 1.0 / (1.0 + temp_error)
        
        # Weighted combination (emphasize physics consistency)
        overall_score = (
            0.2 * train_score +
            0.4 * physics_score +
            0.3 * correlation_score +
            0.1 * temp_score
        )
        
        return overall_score
    
    def get_top_configurations(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the top N performing configurations.
        
        Args:
            n: Number of top configurations to return
            
        Returns:
            List of top configuration results
        """
        if not self.experiment_results:
            return []
        
        # Sort by overall score (descending)
        sorted_results = sorted(
            self.experiment_results, 
            key=lambda x: x.get('overall_score', float('-inf')), 
            reverse=True
        )
        
        return sorted_results[:n]
    
    def save_results_summary(self, filepath: str) -> None:
        """
        Save optimization results summary to file.
        
        Args:
            filepath: Path to save results summary
        """
        import json
        from pathlib import Path
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'total_experiments': len(self.experiment_results),
            'search_space': {
                'latent_dims': self.search_space.latent_dims,
                'beta_values': self.search_space.beta_values,
                'encoder_layers': self.search_space.encoder_layers,
                'activations': self.search_space.activations
            },
            'top_configurations': self.get_top_configurations(10),
            'best_overall_score': self.best_score if self.best_config else None,
            'experiment_results': self.experiment_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Results summary saved to {filepath}")