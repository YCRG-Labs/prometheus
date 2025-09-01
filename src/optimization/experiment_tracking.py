"""
Experiment Tracking Framework with MLflow Integration

This module implements comprehensive experiment tracking for hyperparameter
optimization, including MLflow integration, parameter importance analysis,
and convergence visualization.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Using local tracking only.")

from .hyperparameter_optimizer import OptimizationResult
from ..utils.config import PrometheusConfig

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    experiment_name: str
    tracking_uri: Optional[str] = None
    artifact_location: Optional[str] = None
    enable_mlflow: bool = True
    save_models: bool = True
    save_plots: bool = True
    log_frequency: int = 10


class ExperimentTracker:
    """
    Comprehensive experiment tracking system with MLflow integration.
    
    Tracks hyperparameter optimization experiments, logs metrics and artifacts,
    and provides analysis tools for parameter importance and convergence.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        results_dir: str = "results/experiments"
    ):
        """
        Initialize the experiment tracker.
        
        Args:
            config: Experiment tracking configuration
            results_dir: Directory to save local results
        """
        self.config = config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MLflow if available and enabled
        self.mlflow_enabled = MLFLOW_AVAILABLE and config.enable_mlflow
        if self.mlflow_enabled:
            self._setup_mlflow()
        else:
            logger.info("MLflow tracking disabled or unavailable")
        
        # Experiment state
        self.current_run_id = None
        self.experiment_results = []
        self.parameter_importance = {}
        
        logger.info(f"Experiment tracker initialized")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"MLflow enabled: {self.mlflow_enabled}")
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        try:
            if self.config.tracking_uri:
                mlflow.set_tracking_uri(self.config.tracking_uri)
            
            # Set or create experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(
                        name=self.config.experiment_name,
                        artifact_location=self.config.artifact_location
                    )
                else:
                    experiment_id = experiment.experiment_id
                
                mlflow.set_experiment(experiment_id=experiment_id)
                logger.info(f"MLflow experiment set: {self.config.experiment_name}")
                
            except Exception as e:
                logger.warning(f"Failed to setup MLflow experiment: {e}")
                self.mlflow_enabled = False
                
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")
            self.mlflow_enabled = False
    
    def start_run(
        self, 
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Start a new experiment run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags for the run
            
        Returns:
            Run ID
        """
        if self.mlflow_enabled:
            run = mlflow.start_run(run_name=run_name, tags=tags)
            self.current_run_id = run.info.run_id
            logger.info(f"Started MLflow run: {self.current_run_id}")
        else:
            # Generate local run ID
            self.current_run_id = f"local_run_{int(time.time())}"
            logger.info(f"Started local run: {self.current_run_id}")
        
        return self.current_run_id
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """
        Log experiment parameters.
        
        Args:
            params: Dictionary of parameters to log
        """
        if self.mlflow_enabled and mlflow.active_run():
            # MLflow has limits on parameter values
            filtered_params = {}
            for key, value in params.items():
                if isinstance(value, (int, float, str, bool)):
                    filtered_params[key] = value
                elif isinstance(value, (list, tuple)):
                    filtered_params[key] = str(value)
                else:
                    filtered_params[key] = str(value)
            
            mlflow.log_params(filtered_params)
        
        # Always save locally
        self._save_local_data('parameters', params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log experiment metrics.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if self.mlflow_enabled and mlflow.active_run():
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    mlflow.log_metric(key, value, step=step)
        
        # Always save locally
        self._save_local_data('metrics', metrics, step)
    
    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None) -> None:
        """
        Log an artifact (file).
        
        Args:
            artifact_path: Path to the artifact file
            artifact_name: Optional name for the artifact
        """
        if self.mlflow_enabled and mlflow.active_run():
            mlflow.log_artifact(artifact_path, artifact_name)
        
        # Copy to local results directory
        local_path = self.results_dir / self.current_run_id / "artifacts"
        local_path.mkdir(parents=True, exist_ok=True)
        
        import shutil
        dest_name = artifact_name or Path(artifact_path).name
        shutil.copy2(artifact_path, local_path / dest_name)
    
    def log_model(self, model, model_name: str = "vae_model") -> None:
        """
        Log a trained model.
        
        Args:
            model: PyTorch model to log
            model_name: Name for the model
        """
        if not self.config.save_models:
            return
        
        if self.mlflow_enabled and mlflow.active_run():
            mlflow.pytorch.log_model(model, model_name)
        
        # Save locally
        local_path = self.results_dir / self.current_run_id / "models"
        local_path.mkdir(parents=True, exist_ok=True)
        
        import torch
        torch.save(model.state_dict(), local_path / f"{model_name}.pth")
    
    def end_run(self) -> None:
        """End the current experiment run."""
        if self.mlflow_enabled and mlflow.active_run():
            mlflow.end_run()
        
        logger.info(f"Ended run: {self.current_run_id}")
        self.current_run_id = None
    
    def track_optimization_experiment(
        self,
        result: OptimizationResult,
        model = None,
        training_history: Optional[Dict[str, List[float]]] = None
    ) -> None:
        """
        Track a complete optimization experiment.
        
        Args:
            result: Optimization result to track
            model: Optional trained model
            training_history: Optional training metrics history
        """
        # Start run with descriptive name
        run_name = f"exp_{result.experiment_id}_{self._get_config_summary(result.architecture_config)}"
        self.start_run(run_name=run_name)
        
        try:
            # Log architecture parameters
            arch_params = {f"arch_{k}": v for k, v in result.architecture_config.items()}
            self.log_parameters(arch_params)
            
            # Log training parameters
            train_params = {f"train_{k}": v for k, v in result.training_config.items()}
            self.log_parameters(train_params)
            
            # Log final metrics
            final_metrics = {
                'overall_score': result.overall_score,
                'training_time': result.training_time,
                **{f"train_{k}": v for k, v in result.training_metrics.items()},
                **{f"physics_{k}": v for k, v in result.physics_metrics.items()}
            }
            self.log_metrics(final_metrics)
            
            # Log training history if available
            if training_history:
                for epoch, (train_loss, val_loss) in enumerate(zip(
                    training_history.get('train_total_loss', []),
                    training_history.get('val_total_loss', [])
                )):
                    self.log_metrics({
                        'epoch_train_loss': train_loss,
                        'epoch_val_loss': val_loss
                    }, step=epoch)
            
            # Log model if provided
            if model is not None:
                self.log_model(model, f"vae_exp_{result.experiment_id}")
            
            # Save result object
            self._save_optimization_result(result)
            
            # Add to experiment results
            self.experiment_results.append(result)
            
        finally:
            self.end_run()
    
    def analyze_parameter_importance(self) -> Dict[str, Any]:
        """
        Analyze parameter importance from tracked experiments.
        
        Returns:
            Dictionary containing parameter importance analysis
        """
        if len(self.experiment_results) < 10:
            logger.warning("Need at least 10 experiments for meaningful parameter importance analysis")
            return {}
        
        # Convert results to DataFrame
        df = self._results_to_dataframe()
        
        # Calculate correlation-based importance
        correlation_importance = self._calculate_correlation_importance(df)
        
        # Calculate variance-based importance
        variance_importance = self._calculate_variance_importance(df)
        
        # Calculate mutual information importance
        mi_importance = self._calculate_mutual_information_importance(df)
        
        # Combine importance measures
        combined_importance = self._combine_importance_measures(
            correlation_importance, variance_importance, mi_importance
        )
        
        self.parameter_importance = {
            'correlation_based': correlation_importance,
            'variance_based': variance_importance,
            'mutual_information': mi_importance,
            'combined': combined_importance,
            'analysis_summary': self._summarize_importance_analysis(combined_importance)
        }
        
        return self.parameter_importance
    
    def generate_convergence_plots(self, save_path: Optional[str] = None) -> List[str]:
        """
        Generate optimization convergence plots.
        
        Args:
            save_path: Optional path to save plots
            
        Returns:
            List of generated plot file paths
        """
        if len(self.experiment_results) < 5:
            logger.warning("Need at least 5 experiments for convergence plots")
            return []
        
        plot_paths = []
        
        # Create plots directory
        plots_dir = Path(save_path) if save_path else self.results_dir / "convergence_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Overall score convergence
        plot_path = self._plot_score_convergence(plots_dir)
        plot_paths.append(plot_path)
        
        # 2. Parameter vs performance scatter plots
        scatter_paths = self._plot_parameter_performance_scatter(plots_dir)
        plot_paths.extend(scatter_paths)
        
        # 3. Best configurations over time
        best_config_path = self._plot_best_configurations_timeline(plots_dir)
        plot_paths.append(best_config_path)
        
        # 4. Physics metrics evolution
        physics_path = self._plot_physics_metrics_evolution(plots_dir)
        plot_paths.append(physics_path)
        
        # 5. Parameter importance heatmap
        if self.parameter_importance:
            importance_path = self._plot_parameter_importance_heatmap(plots_dir)
            plot_paths.append(importance_path)
        
        logger.info(f"Generated {len(plot_paths)} convergence plots in {plots_dir}")
        return plot_paths
    
    def _get_config_summary(self, config: Dict[str, Any]) -> str:
        """Get a short summary of configuration."""
        latent_dim = config.get('latent_dim', 'unk')
        beta = config.get('beta', 'unk')
        layers = config.get('encoder_layers', 'unk')
        activation = config.get('activation', 'unk')
        return f"ld{latent_dim}_b{beta}_l{layers}_{activation}"
    
    def _save_local_data(self, data_type: str, data: Any, step: Optional[int] = None) -> None:
        """Save data locally."""
        if not self.current_run_id:
            return
        
        run_dir = self.results_dir / self.current_run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{data_type}.json"
        if step is not None:
            filename = f"{data_type}_step_{step}.json"
        
        filepath = run_dir / filename
        
        # Load existing data if it exists
        existing_data = {}
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    existing_data = json.load(f)
            except:
                pass
        
        # Update with new data
        if step is not None:
            if 'steps' not in existing_data:
                existing_data['steps'] = {}
            existing_data['steps'][str(step)] = data
        else:
            existing_data.update(data)
        
        # Save updated data
        with open(filepath, 'w') as f:
            json.dump(existing_data, f, indent=2, default=str)
    
    def _save_optimization_result(self, result: OptimizationResult) -> None:
        """Save optimization result locally."""
        if not self.current_run_id:
            return
        
        run_dir = self.results_dir / self.current_run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        with open(run_dir / "optimization_result.json", 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
    
    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert experiment results to pandas DataFrame."""
        data = []
        for result in self.experiment_results:
            if result.error is not None:
                continue  # Skip failed experiments
            
            row = {
                'experiment_id': result.experiment_id,
                'overall_score': result.overall_score,
                'training_time': result.training_time,
                **result.architecture_config,
                **result.training_config,
                **result.training_metrics,
                **result.physics_metrics
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _calculate_correlation_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate parameter importance based on correlation with overall score."""
        target = 'overall_score'
        if target not in df.columns:
            return {}
        
        importance = {}
        for col in df.columns:
            if col == target or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            try:
                correlation = abs(df[col].corr(df[target]))
                if not np.isnan(correlation):
                    importance[col] = correlation
            except:
                pass
        
        return importance
    
    def _calculate_variance_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate parameter importance based on variance in outcomes."""
        target = 'overall_score'
        if target not in df.columns:
            return {}
        
        importance = {}
        for col in df.columns:
            if col == target or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            try:
                # Group by parameter value and calculate variance in target
                grouped = df.groupby(pd.cut(df[col], bins=5, duplicates='drop'))[target]
                between_group_var = grouped.mean().var()
                within_group_var = grouped.var().mean()
                
                if within_group_var > 0:
                    f_ratio = between_group_var / within_group_var
                    importance[col] = f_ratio
            except:
                pass
        
        return importance
    
    def _calculate_mutual_information_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate parameter importance using mutual information."""
        try:
            from sklearn.feature_selection import mutual_info_regression
            from sklearn.preprocessing import LabelEncoder
            
            target = 'overall_score'
            if target not in df.columns:
                return {}
            
            # Prepare features
            feature_cols = [col for col in df.columns 
                          if col != target and pd.api.types.is_numeric_dtype(df[col])]
            
            if len(feature_cols) == 0:
                return {}
            
            X = df[feature_cols].fillna(0)
            y = df[target].fillna(0)
            
            # Calculate mutual information
            mi_scores = mutual_info_regression(X, y, random_state=42)
            
            importance = dict(zip(feature_cols, mi_scores))
            return importance
            
        except ImportError:
            logger.warning("scikit-learn not available for mutual information calculation")
            return {}
        except Exception as e:
            logger.warning(f"Mutual information calculation failed: {e}")
            return {}
    
    def _combine_importance_measures(
        self, 
        correlation: Dict[str, float],
        variance: Dict[str, float],
        mutual_info: Dict[str, float]
    ) -> Dict[str, float]:
        """Combine different importance measures."""
        all_params = set(correlation.keys()) | set(variance.keys()) | set(mutual_info.keys())
        
        combined = {}
        for param in all_params:
            scores = []
            
            if param in correlation:
                scores.append(correlation[param])
            if param in variance:
                # Normalize variance importance
                max_var = max(variance.values()) if variance.values() else 1.0
                scores.append(variance[param] / max_var)
            if param in mutual_info:
                # Normalize MI importance
                max_mi = max(mutual_info.values()) if mutual_info.values() else 1.0
                scores.append(mutual_info[param] / max_mi)
            
            if scores:
                combined[param] = np.mean(scores)
        
        return combined
    
    def _summarize_importance_analysis(self, importance: Dict[str, float]) -> Dict[str, Any]:
        """Summarize parameter importance analysis."""
        if not importance:
            return {}
        
        sorted_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'most_important': sorted_params[0][0] if sorted_params else None,
            'least_important': sorted_params[-1][0] if sorted_params else None,
            'top_3_parameters': [param for param, _ in sorted_params[:3]],
            'importance_scores': dict(sorted_params),
            'total_parameters_analyzed': len(importance)
        }
    
    def _plot_score_convergence(self, plots_dir: Path) -> str:
        """Plot overall score convergence over experiments."""
        plt.figure(figsize=(10, 6))
        
        scores = [r.overall_score for r in self.experiment_results if r.error is None]
        experiment_ids = [r.experiment_id for r in self.experiment_results if r.error is None]
        
        # Running best score
        running_best = []
        best_so_far = float('-inf')
        for score in scores:
            best_so_far = max(best_so_far, score)
            running_best.append(best_so_far)
        
        plt.plot(experiment_ids, scores, 'o-', alpha=0.6, label='Individual Scores')
        plt.plot(experiment_ids, running_best, 'r-', linewidth=2, label='Best Score So Far')
        
        plt.xlabel('Experiment ID')
        plt.ylabel('Overall Score')
        plt.title('Optimization Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = plots_dir / "score_convergence.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_parameter_performance_scatter(self, plots_dir: Path) -> List[str]:
        """Plot parameter vs performance scatter plots."""
        df = self._results_to_dataframe()
        if df.empty:
            return []
        
        plot_paths = []
        param_cols = ['latent_dim', 'beta', 'encoder_layers']
        
        for param in param_cols:
            if param not in df.columns:
                continue
            
            plt.figure(figsize=(8, 6))
            plt.scatter(df[param], df['overall_score'], alpha=0.6)
            plt.xlabel(param.replace('_', ' ').title())
            plt.ylabel('Overall Score')
            plt.title(f'Performance vs {param.replace("_", " ").title()}')
            plt.grid(True, alpha=0.3)
            
            plot_path = plots_dir / f"scatter_{param}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_paths.append(str(plot_path))
        
        return plot_paths
    
    def _plot_best_configurations_timeline(self, plots_dir: Path) -> str:
        """Plot timeline of best configurations."""
        plt.figure(figsize=(12, 8))
        
        # Get best configuration at each experiment
        best_configs = []
        best_score = float('-inf')
        
        for result in self.experiment_results:
            if result.error is None and result.overall_score > best_score:
                best_score = result.overall_score
                best_configs.append({
                    'experiment_id': result.experiment_id,
                    'score': result.overall_score,
                    'latent_dim': result.architecture_config.get('latent_dim', 0),
                    'beta': result.architecture_config.get('beta', 0),
                    'layers': result.architecture_config.get('encoder_layers', 0)
                })
        
        if not best_configs:
            return ""
        
        # Create subplot for each parameter
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        exp_ids = [c['experiment_id'] for c in best_configs]
        scores = [c['score'] for c in best_configs]
        
        # Score evolution
        axes[0, 0].plot(exp_ids, scores, 'o-')
        axes[0, 0].set_title('Best Score Evolution')
        axes[0, 0].set_ylabel('Overall Score')
        
        # Parameter evolution
        axes[0, 1].plot(exp_ids, [c['latent_dim'] for c in best_configs], 'o-')
        axes[0, 1].set_title('Best Latent Dimension')
        axes[0, 1].set_ylabel('Latent Dimension')
        
        axes[1, 0].plot(exp_ids, [c['beta'] for c in best_configs], 'o-')
        axes[1, 0].set_title('Best Beta Value')
        axes[1, 0].set_ylabel('Beta')
        axes[1, 0].set_xlabel('Experiment ID')
        
        axes[1, 1].plot(exp_ids, [c['layers'] for c in best_configs], 'o-')
        axes[1, 1].set_title('Best Encoder Layers')
        axes[1, 1].set_ylabel('Encoder Layers')
        axes[1, 1].set_xlabel('Experiment ID')
        
        plt.tight_layout()
        
        plot_path = plots_dir / "best_configurations_timeline.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_physics_metrics_evolution(self, plots_dir: Path) -> str:
        """Plot evolution of physics metrics."""
        plt.figure(figsize=(12, 8))
        
        # Extract physics metrics
        physics_data = []
        for result in self.experiment_results:
            if result.error is None:
                physics_data.append({
                    'experiment_id': result.experiment_id,
                    'physics_score': result.physics_metrics.get('overall_physics_score', 0),
                    'correlation': result.physics_metrics.get('order_parameter_correlation', 0),
                    'temp_accuracy': result.physics_metrics.get('temperature_detection_accuracy', 0)
                })
        
        if not physics_data:
            return ""
        
        df_physics = pd.DataFrame(physics_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Physics score
        axes[0, 0].plot(df_physics['experiment_id'], df_physics['physics_score'], 'o-')
        axes[0, 0].set_title('Physics Consistency Score')
        axes[0, 0].set_ylabel('Physics Score')
        
        # Order parameter correlation
        axes[0, 1].plot(df_physics['experiment_id'], df_physics['correlation'], 'o-')
        axes[0, 1].set_title('Order Parameter Correlation')
        axes[0, 1].set_ylabel('Correlation')
        
        # Temperature detection accuracy
        axes[1, 0].plot(df_physics['experiment_id'], df_physics['temp_accuracy'], 'o-')
        axes[1, 0].set_title('Critical Temperature Detection')
        axes[1, 0].set_ylabel('Detection Accuracy')
        axes[1, 0].set_xlabel('Experiment ID')
        
        # Combined view
        axes[1, 1].plot(df_physics['experiment_id'], df_physics['physics_score'], 'o-', label='Physics Score')
        axes[1, 1].plot(df_physics['experiment_id'], df_physics['correlation'], 's-', label='Correlation')
        axes[1, 1].plot(df_physics['experiment_id'], df_physics['temp_accuracy'], '^-', label='Temp Accuracy')
        axes[1, 1].set_title('All Physics Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_xlabel('Experiment ID')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        plot_path = plots_dir / "physics_metrics_evolution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_parameter_importance_heatmap(self, plots_dir: Path) -> str:
        """Plot parameter importance heatmap."""
        if not self.parameter_importance or 'combined' not in self.parameter_importance:
            return ""
        
        importance = self.parameter_importance['combined']
        if not importance:
            return ""
        
        # Create heatmap data
        params = list(importance.keys())
        scores = list(importance.values())
        
        plt.figure(figsize=(10, 6))
        
        # Sort by importance
        sorted_items = sorted(zip(params, scores), key=lambda x: x[1], reverse=True)
        sorted_params, sorted_scores = zip(*sorted_items)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(sorted_params))
        plt.barh(y_pos, sorted_scores)
        plt.yticks(y_pos, sorted_params)
        plt.xlabel('Importance Score')
        plt.title('Parameter Importance Analysis')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, score in enumerate(sorted_scores):
            plt.text(score + 0.01, i, f'{score:.3f}', va='center')
        
        plt.tight_layout()
        
        plot_path = plots_dir / "parameter_importance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def export_results_summary(self, filepath: Optional[str] = None) -> str:
        """
        Export comprehensive results summary.
        
        Args:
            filepath: Optional path to save summary
            
        Returns:
            Path to saved summary file
        """
        if filepath is None:
            filepath = self.results_dir / "experiment_summary.json"
        
        # Compile comprehensive summary
        summary = {
            'experiment_info': {
                'experiment_name': self.config.experiment_name,
                'total_experiments': len(self.experiment_results),
                'successful_experiments': len([r for r in self.experiment_results if r.error is None]),
                'failed_experiments': len([r for r in self.experiment_results if r.error is not None])
            },
            'best_results': {
                'top_5_experiments': [r.to_dict() for r in sorted(
                    [r for r in self.experiment_results if r.error is None],
                    key=lambda x: x.overall_score, reverse=True
                )[:5]]
            },
            'parameter_importance': self.parameter_importance,
            'convergence_analysis': self._analyze_convergence(),
            'physics_consistency_analysis': self._analyze_physics_consistency()
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Experiment summary exported to {filepath}")
        return str(filepath)
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze optimization convergence."""
        if len(self.experiment_results) < 5:
            return {}
        
        scores = [r.overall_score for r in self.experiment_results if r.error is None]
        
        # Calculate convergence metrics
        best_scores = []
        best_so_far = float('-inf')
        for score in scores:
            best_so_far = max(best_so_far, score)
            best_scores.append(best_so_far)
        
        # Find convergence point (where improvement becomes minimal)
        improvements = [best_scores[i] - best_scores[i-1] for i in range(1, len(best_scores))]
        convergence_threshold = 0.001
        
        convergence_point = len(improvements)
        for i, improvement in enumerate(improvements):
            if improvement < convergence_threshold:
                # Check if next few improvements are also small
                if i + 3 < len(improvements) and all(imp < convergence_threshold for imp in improvements[i:i+3]):
                    convergence_point = i
                    break
        
        return {
            'final_best_score': best_scores[-1] if best_scores else 0,
            'convergence_point': convergence_point,
            'total_improvement': best_scores[-1] - best_scores[0] if len(best_scores) > 1 else 0,
            'convergence_rate': np.mean(improvements) if improvements else 0,
            'score_variance': np.var(scores) if scores else 0
        }
    
    def _analyze_physics_consistency(self) -> Dict[str, Any]:
        """Analyze physics consistency across experiments."""
        successful_results = [r for r in self.experiment_results if r.error is None]
        
        if not successful_results:
            return {}
        
        physics_scores = [r.physics_metrics.get('overall_physics_score', 0) for r in successful_results]
        correlations = [r.physics_metrics.get('order_parameter_correlation', 0) for r in successful_results]
        temp_accuracies = [r.physics_metrics.get('temperature_detection_accuracy', 0) for r in successful_results]
        
        return {
            'best_physics_score': max(physics_scores) if physics_scores else 0,
            'mean_physics_score': np.mean(physics_scores) if physics_scores else 0,
            'physics_score_std': np.std(physics_scores) if physics_scores else 0,
            'best_correlation': max(correlations) if correlations else 0,
            'mean_correlation': np.mean(correlations) if correlations else 0,
            'experiments_above_correlation_threshold': sum(1 for c in correlations if c > 0.7),
            'experiments_above_physics_threshold': sum(1 for s in physics_scores if s > 0.8),
            'temperature_detection_success_rate': np.mean(temp_accuracies) if temp_accuracies else 0
        }