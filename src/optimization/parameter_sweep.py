"""
Comprehensive Parameter Sweep Framework

This module implements systematic hyperparameter optimization experiments
with comprehensive parameter sweeps, automated experiment tracking,
and convergence analysis.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import torch
from torch.utils.data import DataLoader

from .hyperparameter_optimizer import HyperparameterOptimizer, OptimizationResult
from .architecture_search import ArchitectureSearchSpace, ArchitectureConfig
from .experiment_tracking import ExperimentTracker, ExperimentConfig
from .physics_metrics import PhysicsConsistencyEvaluator
from ..models.vae import ConvolutionalVAE
from ..training.trainer import VAETrainer
from ..utils.config import PrometheusConfig

logger = logging.getLogger(__name__)


@dataclass
class SweepConfig:
    """Configuration for parameter sweep experiments."""
    # Search space parameters
    latent_dims: List[int] = None
    beta_values: List[float] = None
    encoder_layers: List[int] = None
    activations: List[str] = None
    learning_rates: List[float] = None
    batch_sizes: List[int] = None
    
    # Execution parameters
    max_experiments: Optional[int] = None
    parallel_jobs: int = 1
    timeout_per_experiment: int = 3600  # 1 hour
    
    # Optimization strategy
    strategy: str = "grid_search"  # "grid_search", "random_search", "bayesian"
    random_seed: Optional[int] = 42
    
    # Early stopping
    enable_early_stopping: bool = True
    convergence_patience: int = 20
    min_improvement: float = 0.001
    
    def __post_init__(self):
        """Set default values if not provided."""
        if self.latent_dims is None:
            self.latent_dims = [2, 4, 8, 16]
        if self.beta_values is None:
            self.beta_values = [0.1, 0.5, 1.0, 2.0, 4.0]
        if self.encoder_layers is None:
            self.encoder_layers = [3, 4, 5]
        if self.activations is None:
            self.activations = ['relu', 'leaky_relu', 'elu']
        if self.learning_rates is None:
            self.learning_rates = [1e-4, 5e-4, 1e-3]
        if self.batch_sizes is None:
            self.batch_sizes = [64, 128, 256]


class ParameterSweepOrchestrator:
    """
    Orchestrates comprehensive parameter sweep experiments.
    
    Manages systematic exploration of hyperparameter space with automated
    experiment tracking, convergence monitoring, and result analysis.
    """
    
    def __init__(
        self,
        base_config: PrometheusConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        sweep_config: SweepConfig,
        results_dir: str = "results/parameter_sweep"
    ):
        """
        Initialize the parameter sweep orchestrator.
        
        Args:
            base_config: Base Prometheus configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            sweep_config: Parameter sweep configuration
            results_dir: Directory to save results
        """
        self.base_config = base_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.sweep_config = sweep_config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize search space
        self.search_space = ArchitectureSearchSpace(
            latent_dims=sweep_config.latent_dims,
            beta_values=sweep_config.beta_values,
            encoder_layers=sweep_config.encoder_layers,
            activations=sweep_config.activations
        )
        
        # Initialize experiment tracker
        experiment_config = ExperimentConfig(
            experiment_name=f"parameter_sweep_{int(time.time())}",
            enable_mlflow=True,
            save_models=True,
            save_plots=True
        )
        self.experiment_tracker = ExperimentTracker(
            config=experiment_config,
            results_dir=str(self.results_dir / "experiments")
        )
        
        # Initialize physics evaluator
        self.physics_evaluator = PhysicsConsistencyEvaluator(
            test_loader=test_loader,
            critical_temperature=base_config.ising.critical_temp
        )
        
        # State tracking
        self.completed_experiments = []
        self.best_result = None
        self.convergence_history = []
        self.is_converged = False
        
        logger.info(f"Parameter sweep orchestrator initialized")
        logger.info(f"Search space size: {self.search_space.total_configurations}")
        logger.info(f"Strategy: {sweep_config.strategy}")
        logger.info(f"Max experiments: {sweep_config.max_experiments}")
        logger.info(f"Parallel jobs: {sweep_config.parallel_jobs}")
    
    def run_parameter_sweep(self) -> Dict[str, Any]:
        """
        Execute comprehensive parameter sweep.
        
        Returns:
            Dictionary containing sweep results and analysis
        """
        logger.info("Starting comprehensive parameter sweep")
        start_time = time.time()
        
        try:
            # Generate experiment configurations
            experiment_configs = self._generate_experiment_configurations()
            
            # Execute experiments
            if self.sweep_config.parallel_jobs > 1:
                results = self._run_parallel_experiments(experiment_configs)
            else:
                results = self._run_sequential_experiments(experiment_configs)
            
            # Analyze results
            analysis = self._analyze_sweep_results(results)
            
            # Generate convergence plots
            plot_paths = self.experiment_tracker.generate_convergence_plots()
            
            # Perform parameter importance analysis
            importance_analysis = self.experiment_tracker.analyze_parameter_importance()
            
            # Export comprehensive summary
            summary_path = self.experiment_tracker.export_results_summary()
            
            total_time = time.time() - start_time
            
            # Compile final results
            sweep_results = {
                'sweep_summary': {
                    'total_experiments': len(results),
                    'successful_experiments': len([r for r in results if r.error is None]),
                    'total_time': total_time,
                    'best_overall_score': self.best_result.overall_score if self.best_result else 0,
                    'convergence_achieved': self.is_converged
                },
                'best_configuration': self.best_result.to_dict() if self.best_result else None,
                'top_configurations': self._get_top_configurations(results, n=10),
                'parameter_importance': importance_analysis,
                'convergence_analysis': analysis.get('convergence_analysis', {}),
                'physics_analysis': analysis.get('physics_analysis', {}),
                'generated_plots': plot_paths,
                'summary_file': summary_path
            }
            
            # Save sweep results
            self._save_sweep_results(sweep_results)
            
            logger.info(f"Parameter sweep completed in {total_time:.1f}s")
            logger.info(f"Best score: {self.best_result.overall_score:.4f}" if self.best_result else "No successful experiments")
            
            return sweep_results
            
        except Exception as e:
            logger.error(f"Parameter sweep failed: {str(e)}")
            raise
    
    def _generate_experiment_configurations(self) -> List[Tuple[ArchitectureConfig, Dict[str, Any]]]:
        """
        Generate experiment configurations based on sweep strategy.
        
        Returns:
            List of (architecture_config, training_config) tuples
        """
        configs = []
        
        if self.sweep_config.strategy == "grid_search":
            configs = self._generate_grid_search_configs()
        elif self.sweep_config.strategy == "random_search":
            configs = self._generate_random_search_configs()
        elif self.sweep_config.strategy == "bayesian":
            configs = self._generate_bayesian_search_configs()
        else:
            raise ValueError(f"Unknown strategy: {self.sweep_config.strategy}")
        
        # Limit number of experiments if specified
        if self.sweep_config.max_experiments is not None:
            configs = configs[:self.sweep_config.max_experiments]
        
        logger.info(f"Generated {len(configs)} experiment configurations")
        return configs
    
    def _generate_grid_search_configs(self) -> List[Tuple[ArchitectureConfig, Dict[str, Any]]]:
        """Generate configurations for grid search."""
        configs = []
        
        # Generate all architecture combinations
        arch_configs = list(self.search_space.generate_configurations())
        
        # For each architecture, try different training parameters
        for arch_config in arch_configs:
            for lr in self.sweep_config.learning_rates:
                for batch_size in self.sweep_config.batch_sizes:
                    training_config = {
                        'learning_rate': lr,
                        'batch_size': batch_size
                    }
                    configs.append((arch_config, training_config))
        
        return configs
    
    def _generate_random_search_configs(self) -> List[Tuple[ArchitectureConfig, Dict[str, Any]]]:
        """Generate configurations for random search."""
        if self.sweep_config.random_seed is not None:
            np.random.seed(self.sweep_config.random_seed)
        
        configs = []
        max_configs = self.sweep_config.max_experiments or 100
        
        for _ in range(max_configs):
            # Random architecture parameters
            arch_config = ArchitectureConfig(
                latent_dim=np.random.choice(self.sweep_config.latent_dims),
                beta=np.random.choice(self.sweep_config.beta_values),
                encoder_layers=np.random.choice(self.sweep_config.encoder_layers),
                activation=np.random.choice(self.sweep_config.activations)
            )
            
            # Random training parameters
            training_config = {
                'learning_rate': np.random.choice(self.sweep_config.learning_rates),
                'batch_size': np.random.choice(self.sweep_config.batch_sizes)
            }
            
            configs.append((arch_config, training_config))
        
        return configs
    
    def _generate_bayesian_search_configs(self) -> List[Tuple[ArchitectureConfig, Dict[str, Any]]]:
        """Generate configurations for Bayesian optimization."""
        # For now, fall back to random search
        # TODO: Implement proper Bayesian optimization with scikit-optimize
        logger.warning("Bayesian optimization not yet implemented, using random search")
        return self._generate_random_search_configs()
    
    def _run_sequential_experiments(
        self, 
        experiment_configs: List[Tuple[ArchitectureConfig, Dict[str, Any]]]
    ) -> List[OptimizationResult]:
        """
        Run experiments sequentially.
        
        Args:
            experiment_configs: List of experiment configurations
            
        Returns:
            List of optimization results
        """
        results = []
        
        for i, (arch_config, train_config) in enumerate(experiment_configs):
            logger.info(f"Running experiment {i+1}/{len(experiment_configs)}")
            
            # Check for convergence
            if self._check_convergence():
                logger.info("Convergence detected, stopping early")
                break
            
            # Run single experiment
            result = self._run_single_experiment(arch_config, train_config, i+1)
            results.append(result)
            
            # Track with experiment tracker
            self.experiment_tracker.track_optimization_experiment(result)
            
            # Update best result
            if result.error is None and (self.best_result is None or result.overall_score > self.best_result.overall_score):
                self.best_result = result
                logger.info(f"New best configuration! Score: {result.overall_score:.4f}")
            
            # Update convergence history
            self._update_convergence_history(result)
        
        return results
    
    def _run_parallel_experiments(
        self, 
        experiment_configs: List[Tuple[ArchitectureConfig, Dict[str, Any]]]
    ) -> List[OptimizationResult]:
        """
        Run experiments in parallel.
        
        Args:
            experiment_configs: List of experiment configurations
            
        Returns:
            List of optimization results
        """
        logger.info(f"Running experiments with {self.sweep_config.parallel_jobs} parallel jobs")
        
        results = []
        
        # Split configs into batches for parallel processing
        batch_size = self.sweep_config.parallel_jobs
        config_batches = [
            experiment_configs[i:i + batch_size] 
            for i in range(0, len(experiment_configs), batch_size)
        ]
        
        experiment_id = 1
        
        for batch in config_batches:
            # Check for convergence
            if self._check_convergence():
                logger.info("Convergence detected, stopping early")
                break
            
            # Run batch in parallel
            batch_results = self._run_experiment_batch(batch, experiment_id)
            results.extend(batch_results)
            
            # Track results
            for result in batch_results:
                self.experiment_tracker.track_optimization_experiment(result)
                
                # Update best result
                if result.error is None and (self.best_result is None or result.overall_score > self.best_result.overall_score):
                    self.best_result = result
                    logger.info(f"New best configuration! Score: {result.overall_score:.4f}")
                
                # Update convergence history
                self._update_convergence_history(result)
            
            experiment_id += len(batch)
        
        return results
    
    def _run_experiment_batch(
        self, 
        batch_configs: List[Tuple[ArchitectureConfig, Dict[str, Any]]], 
        start_id: int
    ) -> List[OptimizationResult]:
        """
        Run a batch of experiments in parallel.
        
        Args:
            batch_configs: Batch of experiment configurations
            start_id: Starting experiment ID
            
        Returns:
            List of optimization results
        """
        # Note: For true parallel processing, we would need to serialize
        # the data loaders and models. For now, we'll run sequentially
        # but this provides the framework for future parallel implementation.
        
        results = []
        for i, (arch_config, train_config) in enumerate(batch_configs):
            experiment_id = start_id + i
            result = self._run_single_experiment(arch_config, train_config, experiment_id)
            results.append(result)
        
        return results
    
    def _run_single_experiment(
        self, 
        arch_config: ArchitectureConfig, 
        train_config: Dict[str, Any], 
        experiment_id: int
    ) -> OptimizationResult:
        """
        Run a single optimization experiment.
        
        Args:
            arch_config: Architecture configuration
            train_config: Training configuration
            experiment_id: Unique experiment identifier
            
        Returns:
            OptimizationResult
        """
        logger.info(f"Starting experiment {experiment_id}")
        logger.info(f"Architecture: {self.search_space.get_configuration_summary(arch_config)}")
        logger.info(f"Training: LR={train_config['learning_rate']}, BS={train_config['batch_size']}")
        
        start_time = time.time()
        
        try:
            # Create model
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
            
            # Create training configuration
            training_config = self.base_config.copy()
            training_config.training.learning_rate = train_config['learning_rate']
            training_config.training.batch_size = train_config['batch_size']
            
            # Architecture-specific adjustments
            if arch_config.latent_dim > 8:
                training_config.training.num_epochs = int(training_config.training.num_epochs * 1.2)
            if arch_config.encoder_layers > 4:
                training_config.training.learning_rate *= 0.8
            
            # Create trainer
            trainer = VAETrainer(
                model=model,
                config=training_config,
                device=None  # Auto-detect
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
            
            # Compile training metrics
            training_metrics = {
                'final_train_loss': training_history['train_total_loss'][-1],
                'final_val_loss': training_history['val_total_loss'][-1],
                'best_val_loss': trainer.best_val_loss,
                'final_reconstruction_loss': training_history['train_reconstruction_loss'][-1],
                'final_kl_loss': training_history['train_kl_loss'][-1],
                'num_epochs_trained': len(training_history['train_total_loss'])
            }
            
            # Compute overall score
            overall_score = self._compute_overall_score(training_metrics, physics_metrics)
            
            training_time = time.time() - start_time
            
            result = OptimizationResult(
                experiment_id=experiment_id,
                architecture_config=arch_config.to_dict(),
                training_config=train_config,
                training_metrics=training_metrics,
                physics_metrics=physics_metrics,
                overall_score=overall_score,
                training_time=training_time
            )
            
            logger.info(f"Experiment {experiment_id} completed successfully")
            logger.info(f"Overall score: {overall_score:.4f}")
            logger.info(f"Physics score: {physics_metrics.get('overall_physics_score', 0):.4f}")
            logger.info(f"Training time: {training_time:.1f}s")
            
            return result
            
        except Exception as e:
            training_time = time.time() - start_time
            logger.error(f"Experiment {experiment_id} failed: {str(e)}")
            
            return OptimizationResult(
                experiment_id=experiment_id,
                architecture_config=arch_config.to_dict(),
                training_config=train_config,
                training_metrics={},
                physics_metrics={},
                overall_score=float('-inf'),
                training_time=training_time,
                error=str(e)
            )
    
    def _compute_overall_score(
        self, 
        training_metrics: Dict[str, float], 
        physics_metrics: Dict[str, float]
    ) -> float:
        """Compute overall optimization score."""
        # Training performance component
        val_loss = training_metrics.get('best_val_loss', float('inf'))
        training_score = 1.0 / (1.0 + val_loss) if val_loss != float('inf') else 0.0
        
        # Physics consistency components
        physics_score = physics_metrics.get('overall_physics_score', 0.0)
        correlation_score = physics_metrics.get('order_parameter_correlation', 0.0)
        
        # Critical temperature accuracy
        temp_error = physics_metrics.get('critical_temperature_error', 1.0)
        temp_score = 1.0 / (1.0 + temp_error)
        
        # Weighted combination (emphasize physics consistency)
        overall_score = (
            0.15 * training_score +     # Training performance
            0.40 * physics_score +      # Physics consistency (highest weight)
            0.25 * correlation_score +  # Order parameter correlation
            0.20 * temp_score          # Critical temperature accuracy
        )
        
        return overall_score
    
    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        if not self.sweep_config.enable_early_stopping:
            return False
        
        if len(self.convergence_history) < self.sweep_config.convergence_patience:
            return False
        
        # Check if recent improvements are below threshold
        recent_scores = self.convergence_history[-self.sweep_config.convergence_patience:]
        improvements = [recent_scores[i] - recent_scores[i-1] for i in range(1, len(recent_scores))]
        
        # Check if all recent improvements are below threshold
        if all(imp < self.sweep_config.min_improvement for imp in improvements):
            self.is_converged = True
            return True
        
        return False
    
    def _update_convergence_history(self, result: OptimizationResult) -> None:
        """Update convergence tracking history."""
        if result.error is None:
            # Track best score so far
            current_best = self.best_result.overall_score if self.best_result else float('-inf')
            best_score = max(current_best, result.overall_score)
            self.convergence_history.append(best_score)
    
    def _get_top_configurations(self, results: List[OptimizationResult], n: int = 10) -> List[Dict[str, Any]]:
        """Get top N configurations from results."""
        successful_results = [r for r in results if r.error is None]
        sorted_results = sorted(successful_results, key=lambda x: x.overall_score, reverse=True)
        return [r.to_dict() for r in sorted_results[:n]]
    
    def _analyze_sweep_results(self, results: List[OptimizationResult]) -> Dict[str, Any]:
        """Analyze parameter sweep results."""
        successful_results = [r for r in results if r.error is None]
        
        if not successful_results:
            return {'error': 'No successful experiments'}
        
        # Extract metrics
        scores = [r.overall_score for r in successful_results]
        physics_scores = [r.physics_metrics.get('overall_physics_score', 0) for r in successful_results]
        correlations = [r.physics_metrics.get('order_parameter_correlation', 0) for r in successful_results]
        temp_errors = [r.physics_metrics.get('critical_temperature_error', 1.0) for r in successful_results]
        
        # Convergence analysis
        convergence_analysis = {
            'total_experiments': len(results),
            'successful_experiments': len(successful_results),
            'convergence_achieved': self.is_converged,
            'best_score_evolution': self.convergence_history,
            'final_best_score': max(scores) if scores else 0,
            'score_improvement': (max(scores) - min(scores)) if len(scores) > 1 else 0
        }
        
        # Physics analysis
        physics_analysis = {
            'best_physics_score': max(physics_scores) if physics_scores else 0,
            'mean_physics_score': np.mean(physics_scores) if physics_scores else 0,
            'best_correlation': max(correlations) if correlations else 0,
            'mean_correlation': np.mean(correlations) if correlations else 0,
            'experiments_above_correlation_threshold': sum(1 for c in correlations if c > 0.7),
            'experiments_above_physics_threshold': sum(1 for s in physics_scores if s > 0.8),
            'best_temp_accuracy': min(temp_errors) if temp_errors else 1.0,
            'mean_temp_error': np.mean(temp_errors) if temp_errors else 1.0
        }
        
        return {
            'convergence_analysis': convergence_analysis,
            'physics_analysis': physics_analysis
        }
    
    def _save_sweep_results(self, results: Dict[str, Any]) -> None:
        """Save parameter sweep results."""
        results_file = self.results_dir / "parameter_sweep_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Parameter sweep results saved to {results_file}")
    
    def generate_sensitivity_analysis(self) -> Dict[str, Any]:
        """
        Generate parameter sensitivity analysis.
        
        Returns:
            Dictionary containing sensitivity analysis results
        """
        if len(self.completed_experiments) < 20:
            logger.warning("Need at least 20 experiments for meaningful sensitivity analysis")
            return {}
        
        # Convert results to DataFrame for analysis
        data = []
        for result in self.completed_experiments:
            if result.error is not None:
                continue
            
            row = {
                'overall_score': result.overall_score,
                **result.architecture_config,
                **result.training_config,
                **result.physics_metrics
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Perform sensitivity analysis
        sensitivity_results = {}
        
        # Parameters to analyze
        param_columns = ['latent_dim', 'beta', 'encoder_layers', 'learning_rate', 'batch_size']
        target_metrics = ['overall_score', 'overall_physics_score', 'order_parameter_correlation']
        
        for param in param_columns:
            if param not in df.columns:
                continue
            
            param_sensitivity = {}
            
            for metric in target_metrics:
                if metric not in df.columns:
                    continue
                
                # Calculate correlation
                correlation = df[param].corr(df[metric])
                
                # Calculate variance explained
                try:
                    from sklearn.linear_model import LinearRegression
                    from sklearn.metrics import r2_score
                    
                    X = df[[param]].values.reshape(-1, 1)
                    y = df[metric].values
                    
                    model = LinearRegression().fit(X, y)
                    y_pred = model.predict(X)
                    r2 = r2_score(y, y_pred)
                    
                    param_sensitivity[metric] = {
                        'correlation': correlation,
                        'r2_score': r2,
                        'coefficient': model.coef_[0] if len(model.coef_) > 0 else 0
                    }
                    
                except ImportError:
                    param_sensitivity[metric] = {
                        'correlation': correlation,
                        'r2_score': 0,
                        'coefficient': 0
                    }
            
            sensitivity_results[param] = param_sensitivity
        
        return sensitivity_results