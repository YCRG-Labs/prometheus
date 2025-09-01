"""
Hyperparameter Optimization Framework

This module implements systematic hyperparameter optimization for VAE
architectures, including grid search, random search, and Bayesian
optimization approaches.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
import numpy as np

import torch
from torch.utils.data import DataLoader

from .architecture_search import ArchitectureSearchSpace, ArchitectureConfig
from .physics_metrics import PhysicsConsistencyEvaluator
from ..models.vae import ConvolutionalVAE
from ..training.trainer import VAETrainer
from ..utils.config import PrometheusConfig

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Results from a single hyperparameter optimization experiment."""
    experiment_id: int
    architecture_config: Dict[str, Any]
    training_config: Dict[str, Any]
    training_metrics: Dict[str, float]
    physics_metrics: Dict[str, float]
    overall_score: float
    training_time: float
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class HyperparameterOptimizer:
    """
    Comprehensive hyperparameter optimization framework.
    
    Supports systematic exploration of VAE architectures and training
    configurations to maximize physics consistency and performance.
    """
    
    def __init__(
        self,
        base_config: PrometheusConfig,
        search_space: ArchitectureSearchSpace,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        results_dir: str = "results/hyperparameter_optimization"
    ):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            base_config: Base configuration for experiments
            search_space: Architecture search space
            train_loader: Training data loader
            val_loader: Validation data loader  
            test_loader: Test data loader
            results_dir: Directory to save optimization results
        """
        self.base_config = base_config
        self.search_space = search_space
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.results_dir = Path(results_dir)
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize physics evaluator
        self.physics_evaluator = PhysicsConsistencyEvaluator(
            test_loader=test_loader,
            critical_temperature=base_config.ising.critical_temp
        )
        
        # Results tracking
        self.optimization_results = []
        self.best_result = None
        self.experiment_counter = 0
        
        logger.info(f"Hyperparameter optimizer initialized")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Total configurations to explore: {search_space.total_configurations}")
    
    def create_model_from_config(self, arch_config: ArchitectureConfig) -> ConvolutionalVAE:
        """
        Create VAE model from architecture configuration.
        
        Args:
            arch_config: Architecture configuration
            
        Returns:
            Initialized ConvolutionalVAE model
        """
        # Update input shape based on base config
        input_shape = (
            1, 
            self.base_config.ising.lattice_size[0], 
            self.base_config.ising.lattice_size[1]
        )
        
        model = ConvolutionalVAE(
            input_shape=input_shape,
            latent_dim=arch_config.latent_dim,
            encoder_channels=arch_config.encoder_channels,
            decoder_channels=arch_config.decoder_channels,
            kernel_sizes=arch_config.kernel_sizes,
            beta=arch_config.beta
        )
        
        return model
    
    def create_trainer_config(self, arch_config: ArchitectureConfig) -> PrometheusConfig:
        """
        Create training configuration with architecture-specific adjustments.
        
        Args:
            arch_config: Architecture configuration
            
        Returns:
            Updated PrometheusConfig for training
        """
        config = self.base_config.copy()
        
        # Architecture-specific training adjustments
        if arch_config.latent_dim > 8:
            # Larger latent spaces may benefit from more epochs
            config.training.num_epochs = int(config.training.num_epochs * 1.3)
        
        if arch_config.encoder_layers > 4:
            # Deeper networks may need lower learning rates
            config.training.learning_rate *= 0.7
        
        if arch_config.beta > 2.0:
            # High beta values may need more careful training
            config.training.learning_rate *= 0.8
            config.training.num_epochs = int(config.training.num_epochs * 1.2)
        
        return config
    
    def run_single_experiment(self, arch_config: ArchitectureConfig) -> OptimizationResult:
        """
        Run a single hyperparameter optimization experiment.
        
        Args:
            arch_config: Architecture configuration to test
            
        Returns:
            OptimizationResult containing experiment outcomes
        """
        self.experiment_counter += 1
        experiment_id = self.experiment_counter
        
        logger.info(f"Starting experiment {experiment_id}/{self.search_space.total_configurations}")
        logger.info(f"Config: {self.search_space.get_configuration_summary(arch_config)}")
        
        start_time = time.time()
        
        try:
            # Create model and training configuration
            model = self.create_model_from_config(arch_config)
            training_config = self.create_trainer_config(arch_config)
            
            # Create trainer
            trainer = VAETrainer(
                model=model,
                config=training_config,
                device=None  # Auto-detect device
            )
            
            # Train model
            training_history = trainer.train(
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                enable_early_stopping=True
            )
            
            # Evaluate physics consistency
            physics_metrics = self.physics_evaluator.evaluate_model(
                model=trainer.model,
                device=trainer.device
            )
            
            # Compute training metrics summary
            training_metrics = {
                'final_train_loss': training_history['train_total_loss'][-1],
                'final_val_loss': training_history['val_total_loss'][-1],
                'best_val_loss': trainer.best_val_loss,
                'final_reconstruction_loss': training_history['train_reconstruction_loss'][-1],
                'final_kl_loss': training_history['train_kl_loss'][-1],
                'num_epochs_trained': len(training_history['train_total_loss']),
                'convergence_epoch': self._find_convergence_epoch(training_history)
            }
            
            # Compute overall score
            overall_score = self._compute_overall_score(training_metrics, physics_metrics)
            
            training_time = time.time() - start_time
            
            # Create result object
            result = OptimizationResult(
                experiment_id=experiment_id,
                architecture_config=arch_config.to_dict(),
                training_config={
                    'num_epochs': training_config.training.num_epochs,
                    'learning_rate': training_config.training.learning_rate,
                    'batch_size': training_config.training.batch_size,
                    'beta': arch_config.beta
                },
                training_metrics=training_metrics,
                physics_metrics=physics_metrics,
                overall_score=overall_score,
                training_time=training_time
            )
            
            logger.info(f"Experiment {experiment_id} completed successfully")
            logger.info(f"Overall score: {overall_score:.4f}, Physics score: {physics_metrics.get('overall_physics_score', 0):.4f}")
            logger.info(f"Training time: {training_time:.1f}s")
            
            return result
            
        except Exception as e:
            training_time = time.time() - start_time
            logger.error(f"Experiment {experiment_id} failed: {str(e)}")
            
            return OptimizationResult(
                experiment_id=experiment_id,
                architecture_config=arch_config.to_dict(),
                training_config={},
                training_metrics={},
                physics_metrics={},
                overall_score=float('-inf'),
                training_time=training_time,
                error=str(e)
            )
    
    def run_grid_search(self, max_experiments: Optional[int] = None) -> List[OptimizationResult]:
        """
        Run systematic grid search over all configurations.
        
        Args:
            max_experiments: Maximum number of experiments to run (None for all)
            
        Returns:
            List of optimization results
        """
        logger.info("Starting grid search optimization")
        
        configurations = list(self.search_space.generate_configurations())
        
        if max_experiments is not None:
            configurations = configurations[:max_experiments]
            logger.info(f"Limited to {max_experiments} experiments")
        
        total_experiments = len(configurations)
        logger.info(f"Running {total_experiments} experiments")
        
        for i, arch_config in enumerate(configurations):
            logger.info(f"Progress: {i+1}/{total_experiments}")
            
            result = self.run_single_experiment(arch_config)
            self.optimization_results.append(result)
            
            # Update best result
            if result.overall_score > (self.best_result.overall_score if self.best_result else float('-inf')):
                self.best_result = result
                logger.info(f"New best configuration found! Score: {result.overall_score:.4f}")
            
            # Save intermediate results
            if (i + 1) % 10 == 0:
                self.save_results(f"intermediate_results_{i+1}.json")
        
        logger.info("Grid search optimization completed")
        logger.info(f"Best overall score: {self.best_result.overall_score:.4f}")
        
        return self.optimization_results
    
    def run_random_search(self, num_experiments: int, seed: Optional[int] = None) -> List[OptimizationResult]:
        """
        Run random search over configuration space.
        
        Args:
            num_experiments: Number of random experiments to run
            seed: Random seed for reproducibility
            
        Returns:
            List of optimization results
        """
        if seed is not None:
            np.random.seed(seed)
        
        logger.info(f"Starting random search with {num_experiments} experiments")
        
        # Generate random configurations
        all_configs = list(self.search_space.generate_configurations())
        selected_indices = np.random.choice(
            len(all_configs), 
            size=min(num_experiments, len(all_configs)), 
            replace=False
        )
        
        selected_configs = [all_configs[i] for i in selected_indices]
        
        for i, arch_config in enumerate(selected_configs):
            logger.info(f"Random search progress: {i+1}/{num_experiments}")
            
            result = self.run_single_experiment(arch_config)
            self.optimization_results.append(result)
            
            # Update best result
            if result.overall_score > (self.best_result.overall_score if self.best_result else float('-inf')):
                self.best_result = result
                logger.info(f"New best configuration found! Score: {result.overall_score:.4f}")
        
        logger.info("Random search optimization completed")
        return self.optimization_results
    
    def _compute_overall_score(
        self, 
        training_metrics: Dict[str, float], 
        physics_metrics: Dict[str, float]
    ) -> float:
        """
        Compute overall optimization score.
        
        Args:
            training_metrics: Training performance metrics
            physics_metrics: Physics consistency metrics
            
        Returns:
            Overall score (higher is better)
        """
        # Training performance component (lower loss is better)
        val_loss = training_metrics.get('best_val_loss', float('inf'))
        training_score = 1.0 / (1.0 + val_loss) if val_loss != float('inf') else 0.0
        
        # Physics consistency components
        physics_score = physics_metrics.get('overall_physics_score', 0.0)
        correlation_score = physics_metrics.get('order_parameter_correlation', 0.0)
        
        # Critical temperature accuracy (lower error is better)
        temp_error = physics_metrics.get('critical_temperature_error', 1.0)
        temp_score = 1.0 / (1.0 + temp_error)
        
        # Reconstruction quality
        recon_loss = training_metrics.get('final_reconstruction_loss', float('inf'))
        recon_score = 1.0 / (1.0 + recon_loss) if recon_loss != float('inf') else 0.0
        
        # Weighted combination (emphasize physics consistency)
        overall_score = (
            0.15 * training_score +     # Overall training performance
            0.40 * physics_score +      # Physics consistency (highest weight)
            0.25 * correlation_score +  # Order parameter correlation
            0.15 * temp_score +         # Critical temperature accuracy
            0.05 * recon_score          # Reconstruction quality
        )
        
        return overall_score
    
    def _find_convergence_epoch(self, training_history: Dict[str, List[float]]) -> int:
        """
        Find the epoch where training converged.
        
        Args:
            training_history: Training metrics history
            
        Returns:
            Epoch number where convergence was achieved
        """
        val_losses = training_history.get('val_total_loss', [])
        if len(val_losses) < 10:
            return len(val_losses)
        
        # Look for plateau in validation loss
        window_size = 5
        for i in range(window_size, len(val_losses)):
            recent_losses = val_losses[i-window_size:i]
            if max(recent_losses) - min(recent_losses) < 0.001:
                return i
        
        return len(val_losses)
    
    def get_top_results(self, n: int = 10) -> List[OptimizationResult]:
        """
        Get top N optimization results.
        
        Args:
            n: Number of top results to return
            
        Returns:
            List of top optimization results
        """
        if not self.optimization_results:
            return []
        
        # Sort by overall score (descending)
        sorted_results = sorted(
            self.optimization_results,
            key=lambda x: x.overall_score,
            reverse=True
        )
        
        return sorted_results[:n]
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze optimization results and provide insights.
        
        Returns:
            Dictionary containing analysis results
        """
        if not self.optimization_results:
            return {}
        
        # Filter out failed experiments
        successful_results = [r for r in self.optimization_results if r.error is None]
        
        if not successful_results:
            return {'error': 'No successful experiments'}
        
        # Extract metrics for analysis
        scores = [r.overall_score for r in successful_results]
        physics_scores = [r.physics_metrics.get('overall_physics_score', 0) for r in successful_results]
        correlations = [r.physics_metrics.get('order_parameter_correlation', 0) for r in successful_results]
        
        # Analyze by architecture parameters
        latent_dim_analysis = self._analyze_by_parameter('latent_dim', successful_results)
        beta_analysis = self._analyze_by_parameter('beta', successful_results)
        layers_analysis = self._analyze_by_parameter('encoder_layers', successful_results)
        activation_analysis = self._analyze_by_parameter('activation', successful_results)
        
        analysis = {
            'total_experiments': len(self.optimization_results),
            'successful_experiments': len(successful_results),
            'failed_experiments': len(self.optimization_results) - len(successful_results),
            'best_overall_score': max(scores),
            'mean_overall_score': np.mean(scores),
            'std_overall_score': np.std(scores),
            'best_physics_score': max(physics_scores),
            'mean_physics_score': np.mean(physics_scores),
            'best_correlation': max(correlations),
            'mean_correlation': np.mean(correlations),
            'parameter_analysis': {
                'latent_dim': latent_dim_analysis,
                'beta': beta_analysis,
                'encoder_layers': layers_analysis,
                'activation': activation_analysis
            },
            'top_configurations': [r.to_dict() for r in self.get_top_results(5)]
        }
        
        return analysis
    
    def _analyze_by_parameter(self, param_name: str, results: List[OptimizationResult]) -> Dict[str, Any]:
        """
        Analyze results grouped by a specific parameter.
        
        Args:
            param_name: Name of parameter to analyze
            results: List of optimization results
            
        Returns:
            Analysis results for the parameter
        """
        param_groups = {}
        
        for result in results:
            param_value = result.architecture_config.get(param_name)
            if param_value not in param_groups:
                param_groups[param_value] = []
            param_groups[param_value].append(result.overall_score)
        
        analysis = {}
        for value, scores in param_groups.items():
            analysis[str(value)] = {
                'count': len(scores),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'max_score': max(scores),
                'min_score': min(scores)
            }
        
        return analysis
    
    def save_results(self, filename: str = "optimization_results.json") -> None:
        """
        Save optimization results to file.
        
        Args:
            filename: Name of file to save results
        """
        filepath = self.results_dir / filename
        
        results_data = {
            'optimization_summary': {
                'total_experiments': len(self.optimization_results),
                'best_score': self.best_result.overall_score if self.best_result else None,
                'search_space': {
                    'latent_dims': self.search_space.latent_dims,
                    'beta_values': self.search_space.beta_values,
                    'encoder_layers': self.search_space.encoder_layers,
                    'activations': self.search_space.activations
                }
            },
            'results': [r.to_dict() for r in self.optimization_results],
            'analysis': self.analyze_results()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> None:
        """
        Load optimization results from file.
        
        Args:
            filepath: Path to results file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct results objects
        self.optimization_results = []
        for result_dict in data['results']:
            result = OptimizationResult(**result_dict)
            self.optimization_results.append(result)
        
        # Find best result
        if self.optimization_results:
            self.best_result = max(self.optimization_results, key=lambda x: x.overall_score)
        
        logger.info(f"Loaded {len(self.optimization_results)} results from {filepath}")