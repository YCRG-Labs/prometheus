"""
Discovery Pipeline for automated exploration of model variant parameter spaces.

This module orchestrates the end-to-end discovery workflow, including parameter
space exploration, Monte Carlo simulation, VAE analysis, and novel phenomena
detection.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np

from .base_types import (
    ModelVariantConfig,
    SimulationData,
    VAEAnalysisResults,
    LatentRepresentation,
    NovelPhenomenon,
    ExplorationStrategy,
    DiscoveryResults
)
from .model_registry import ModelVariantRegistry
from .parameter_explorer import ParameterSpaceExplorer, ParameterPoint
from .phenomena_detector import NovelPhenomenonDetector
from .prometheus_integration import PrometheusIntegration, PrometheusConfig
from .error_recovery import (
    ErrorRecoveryManager,
    ErrorContext,
    ErrorCategory,
    RecoveryAction
)
from .resource_manager import ResourceManager, ParallelExecutor
from ..utils.logging_utils import get_logger


@dataclass
class DiscoveryConfig:
    """Configuration for discovery pipeline.
    
    Attributes:
        variant_id: ID of the model variant to explore
        exploration_strategy: Strategy for parameter space exploration
        simulation_params: Monte Carlo simulation parameters
        vae_config: VAE training configuration
        analysis_config: Critical exponent extraction settings
        checkpoint_interval: Number of points between checkpoints
        output_dir: Directory for results output
        enable_parallel: Enable parallel execution of parameter points
        max_parallel_tasks: Maximum number of parallel tasks
        max_memory_gb: Maximum memory to use (GB)
    """
    variant_id: str
    exploration_strategy: ExplorationStrategy
    simulation_params: Dict[str, Any]
    vae_config: Dict[str, Any]
    analysis_config: Dict[str, Any]
    checkpoint_interval: int = 10
    output_dir: str = 'results/discovery'
    enable_parallel: bool = False
    max_parallel_tasks: int = 4
    max_memory_gb: float = 16.0


class DiscoveryPipeline:
    """End-to-end discovery workflow orchestrator.
    
    The discovery pipeline coordinates parameter space exploration, Monte Carlo
    simulations, VAE analysis, and novel phenomena detection. It provides
    checkpointing for long-running explorations and progress monitoring.
    
    Attributes:
        config: Discovery pipeline configuration
        registry: Model variant registry
        explorer: Parameter space explorer
        logger: Logger instance
        _current_point_index: Index of current parameter point
        _results: Accumulated results
        _start_time: Pipeline start time
    """
    
    def __init__(self, config: DiscoveryConfig, registry: ModelVariantRegistry):
        """Initialize discovery pipeline.
        
        Args:
            config: Pipeline configuration
            registry: Model variant registry for accessing model definitions
        """
        self.config = config
        self.registry = registry
        self.logger = get_logger(__name__)
        
        # Get model variant configuration
        self.variant_config = registry.get_variant(config.variant_id)
        
        # Initialize parameter space explorer
        # Extract parameter ranges from exploration strategy or use defaults
        parameter_ranges = self._get_parameter_ranges()
        self.explorer = ParameterSpaceExplorer(
            config.variant_id,
            parameter_ranges
        )
        
        # Initialize Prometheus integration layer
        system_type = self._infer_system_type()
        prometheus_config = PrometheusConfig(
            system_type=system_type,
            random_seed=config.simulation_params.get('seed', 42),
            use_ensemble=config.analysis_config.get('use_ensemble', True),
            enable_caching=config.analysis_config.get('enable_caching', True),
            enable_profiling=config.analysis_config.get('enable_profiling', False),
            vae_params=config.vae_config,
            analysis_params=config.analysis_config
        )
        self.prometheus = PrometheusIntegration(prometheus_config)
        
        # Initialize phenomena detector
        self.phenomena_detector = NovelPhenomenonDetector(
            anomaly_threshold=config.analysis_config.get('anomaly_threshold', 3.0)
        )
        
        # Initialize error recovery manager
        self.error_recovery = ErrorRecoveryManager(
            max_retries=config.analysis_config.get('max_retries', 3)
        )
        
        # Initialize resource manager and parallel executor
        self.resource_manager = ResourceManager(
            max_memory_gb=config.max_memory_gb,
            n_parallel=config.max_parallel_tasks,
            gpu_enabled=config.vae_config.get('use_gpu', True)
        )
        
        if config.enable_parallel:
            self.parallel_executor = ParallelExecutor(
                resource_manager=self.resource_manager,
                use_processes=True,
                max_workers=config.max_parallel_tasks
            )
        else:
            self.parallel_executor = None
        
        # Pipeline state
        self._current_point_index = 0
        self._results: List[VAEAnalysisResults] = []
        self._novel_phenomena: List[NovelPhenomenon] = []
        self._start_time = 0.0
        
        # Create output directory
        self.output_path = Path(config.output_dir) / config.variant_id
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized discovery pipeline for variant '{config.variant_id}'")
    
    def run_exploration(self, resume_from: Optional[str] = None) -> DiscoveryResults:
        """Execute full exploration workflow with checkpointing.
        
        This is the main entry point for the discovery pipeline. It generates
        parameter points, runs simulations, performs VAE analysis, and detects
        novel phenomena.
        
        Args:
            resume_from: Path to checkpoint file to resume from (optional)
            
        Returns:
            DiscoveryResults containing all exploration outcomes
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Discovery Pipeline Exploration")
        self.logger.info("=" * 80)
        
        # Load checkpoint if resuming
        if resume_from is not None:
            self.load_checkpoint(resume_from)
            self.logger.info(f"Resumed from checkpoint: {resume_from}")
        else:
            self._start_time = time.time()
        
        # Generate parameter points to explore
        parameter_points = self.explorer.generate_sampling_points(
            self.config.exploration_strategy
        )
        
        self.logger.info(f"Generated {len(parameter_points)} parameter points to explore")
        self.logger.info(f"Exploration strategy: {self.config.exploration_strategy.method}")
        
        # Check if parallel execution is enabled
        if self.config.enable_parallel and self.parallel_executor is not None:
            return self._run_exploration_parallel(parameter_points, resume_from is not None)
        
        # Iterate through parameter points (sequential execution)
        param_names = sorted(self.explorer.parameter_ranges.keys())
        
        for i in range(self._current_point_index, len(parameter_points)):
            self._current_point_index = i
            point_params = parameter_points[i]
            
            # Convert array to parameter dictionary
            params_dict = {
                name: float(point_params[j])
                for j, name in enumerate(param_names)
            }
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Exploring point {i+1}/{len(parameter_points)}")
            self.logger.info(f"Parameters: {params_dict}")
            self.logger.info(f"{'='*60}")
            
            try:
                # Run simulation for this parameter point with error recovery
                sim_data = self._simulate_parameter_point_with_recovery(params_dict)
                if sim_data is None:
                    self.logger.warning(f"Skipping point {i+1} due to simulation failure")
                    continue
                
                # Analyze with VAE with error recovery
                vae_results = self._analyze_with_vae_with_recovery(sim_data)
                if vae_results is None:
                    self.logger.warning(f"Skipping point {i+1} due to analysis failure")
                    continue
                
                # Store results
                self._results.append(vae_results)
                
                # Update explorer with results
                uncertainty = self._estimate_uncertainty(vae_results)
                self.explorer.update_explored_point(
                    params_dict,
                    vae_results,
                    uncertainty
                )
                
                # Detect novel phenomena using the comprehensive detector
                try:
                    phenomena = self._detect_novel_phenomena(vae_results, sim_data)
                    if phenomena:
                        self._novel_phenomena.extend(phenomena)
                        self.logger.info(f"Detected {len(phenomena)} novel phenomena at this point")
                except Exception as e:
                    self.logger.warning(f"Error in phenomena detection: {e}")
                    # Continue without phenomena detection
                
                # Checkpoint if needed
                if (i + 1) % self.config.checkpoint_interval == 0:
                    checkpoint_path = self.output_path / f"checkpoint_{i+1}.json"
                    self.save_checkpoint(str(checkpoint_path))
                    self.logger.info(f"Saved checkpoint: {checkpoint_path}")
                
            except Exception as e:
                self.logger.error(f"Unexpected error exploring point {i+1}: {e}", exc_info=True)
                # Continue with next point
                continue
        
        # Calculate execution time
        execution_time = time.time() - self._start_time
        
        # Create final results
        results = DiscoveryResults(
            variant_id=self.config.variant_id,
            exploration_strategy=self.config.exploration_strategy,
            n_points_explored=len(self._results),
            vae_results=self._results,
            novel_phenomena=self._novel_phenomena,
            phase_diagrams=[],  # Will be generated in comparative analysis
            execution_time=execution_time,
            checkpoint_path=str(self.output_path / "final_checkpoint.json")
        )
        
        # Save final checkpoint
        self.save_checkpoint(results.checkpoint_path)
        
        # Save results summary
        self._save_results_summary(results)
        
        # Log error statistics
        error_stats = self.error_recovery.get_error_statistics()
        
        self.logger.info("=" * 80)
        self.logger.info("Discovery Pipeline Exploration Complete")
        self.logger.info(f"Explored {results.n_points_explored} parameter points")
        self.logger.info(f"Detected {len(results.novel_phenomena)} novel phenomena")
        self.logger.info(f"Execution time: {execution_time:.2f} seconds")
        
        if error_stats['total_errors'] > 0:
            self.logger.info(f"\nError Recovery Statistics:")
            self.logger.info(f"  Total errors handled: {error_stats['total_errors']}")
            self.logger.info(f"  Errors by category:")
            for category, count in error_stats['by_category'].items():
                self.logger.info(f"    {category}: {count}")
            if error_stats['most_common']:
                self.logger.info(f"  Most common: {error_stats['most_common']['category']} "
                               f"({error_stats['most_common']['count']} occurrences)")
        
        self.logger.info("=" * 80)
        
        return results
    
    def _run_exploration_parallel(
        self,
        parameter_points: np.ndarray,
        is_resuming: bool
    ) -> DiscoveryResults:
        """Execute exploration workflow with parallel execution.
        
        Args:
            parameter_points: Array of parameter points to explore
            is_resuming: Whether resuming from checkpoint
            
        Returns:
            DiscoveryResults containing all exploration outcomes
        """
        self.logger.info("Using PARALLEL execution mode")
        
        param_names = sorted(self.explorer.parameter_ranges.keys())
        
        # Prepare tasks for parallel execution
        tasks = []
        for i in range(self._current_point_index, len(parameter_points)):
            point_params = parameter_points[i]
            params_dict = {
                name: float(point_params[j])
                for j, name in enumerate(param_names)
            }
            task_id = f"point_{i}"
            tasks.append((task_id, (i, params_dict)))
        
        # Define worker function for parallel execution
        def process_parameter_point(task_id: str, task_data: Tuple[int, Dict]) -> Dict[str, Any]:
            """Process a single parameter point."""
            i, params_dict = task_data
            
            try:
                # Allocate resources for simulation
                lattice_size = self.config.simulation_params.get('lattice_size', 32)
                n_samples = self.config.simulation_params.get('n_samples', 100)
                
                if not self.resource_manager.allocate_simulation(
                    task_id, lattice_size, n_samples
                ):
                    return {'success': False, 'error': 'Resource allocation failed'}
                
                # Run simulation
                sim_data = self._simulate_parameter_point_with_recovery(params_dict)
                if sim_data is None:
                    return {'success': False, 'error': 'Simulation failed'}
                
                # Allocate resources for VAE training
                data_size = len(sim_data.temperatures) * n_samples
                use_gpu = self.config.vae_config.get('use_gpu', True)
                
                success, gpu_id = self.resource_manager.allocate_vae_training(
                    f"{task_id}_vae", data_size, use_gpu=use_gpu
                )
                
                if not success:
                    return {'success': False, 'error': 'VAE resource allocation failed'}
                
                # Update VAE config with GPU assignment
                if gpu_id is not None:
                    self.config.vae_config['device'] = f'cuda:{gpu_id}'
                
                # Analyze with VAE
                vae_results = self._analyze_with_vae_with_recovery(sim_data)
                if vae_results is None:
                    return {'success': False, 'error': 'VAE analysis failed'}
                
                # Detect novel phenomena
                phenomena = []
                try:
                    phenomena = self._detect_novel_phenomena(vae_results, sim_data)
                except Exception as e:
                    self.logger.warning(f"Phenomena detection failed: {e}")
                
                return {
                    'success': True,
                    'index': i,
                    'params': params_dict,
                    'vae_results': vae_results,
                    'phenomena': phenomena
                }
                
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        # Execute in parallel with progress tracking
        def progress_callback(completed: int, total: int):
            pct = 100 * completed / total
            self.logger.info(f"Progress: {completed}/{total} ({pct:.1f}%)")
            
            # Checkpoint periodically
            if completed % self.config.checkpoint_interval == 0:
                checkpoint_path = self.output_path / f"checkpoint_{completed}.json"
                self.save_checkpoint(str(checkpoint_path))
        
        execution_results = self.parallel_executor.execute_parallel(
            process_parameter_point,
            tasks,
            progress_callback
        )
        
        # Process results
        for task_id, result in execution_results['results'].items():
            if result['success']:
                self._results.append(result['vae_results'])
                if result['phenomena']:
                    self._novel_phenomena.extend(result['phenomena'])
                
                # Update explorer
                uncertainty = self._estimate_uncertainty(result['vae_results'])
                self.explorer.update_explored_point(
                    result['params'],
                    result['vae_results'],
                    uncertainty
                )
        
        # Calculate execution time
        execution_time = time.time() - self._start_time
        
        # Create final results
        results = DiscoveryResults(
            variant_id=self.config.variant_id,
            exploration_strategy=self.config.exploration_strategy,
            n_points_explored=len(self._results),
            vae_results=self._results,
            novel_phenomena=self._novel_phenomena,
            phase_diagrams=[],
            execution_time=execution_time,
            checkpoint_path=str(self.output_path / "final_checkpoint.json")
        )
        
        # Save final checkpoint
        self.save_checkpoint(results.checkpoint_path)
        self._save_results_summary(results)
        
        # Log results
        self.logger.info("=" * 80)
        self.logger.info("Parallel Discovery Pipeline Complete")
        self.logger.info(f"Explored {results.n_points_explored} parameter points")
        self.logger.info(f"Detected {len(results.novel_phenomena)} novel phenomena")
        self.logger.info(f"Execution time: {execution_time:.2f} seconds")
        self.logger.info(f"Success rate: {execution_results['success_rate']:.1%}")
        
        # Log resource usage
        resource_summary = self.resource_manager.get_allocation_summary()
        self.logger.info(f"\nResource Usage:")
        self.logger.info(f"  Peak memory: {resource_summary['total_memory_mb']:.1f} MB")
        self.logger.info(f"  Peak CPU cores: {resource_summary['total_cpu_cores']}")
        if self.resource_manager.gpu_enabled:
            self.logger.info(f"  GPU usage: {resource_summary['gpu_usage']}")
        
        self.logger.info("=" * 80)
        
        return results
    
    def _simulate_parameter_point_with_recovery(
        self,
        params: Dict[str, float]
    ) -> Optional[SimulationData]:
        """Run Monte Carlo simulation with error recovery.
        
        Args:
            params: Parameter values for this point
            
        Returns:
            SimulationData or None if recovery failed
        """
        try:
            return self._simulate_parameter_point(params)
        except Exception as e:
            self.logger.warning(f"Simulation error: {e}")
            
            # Create error context
            context = ErrorContext(
                category=self.error_recovery.categorize_error(e, "simulation"),
                error=e,
                parameters=self.config.simulation_params.copy(),
                attempt_number=1
            )
            
            # Execute recovery
            def recovery_fn(adjusted_params: Dict[str, Any]) -> SimulationData:
                # Update config with adjusted parameters
                old_params = self.config.simulation_params.copy()
                self.config.simulation_params.update(adjusted_params)
                try:
                    return self._simulate_parameter_point(params)
                finally:
                    # Restore original parameters
                    self.config.simulation_params = old_params
            
            return self.error_recovery.execute_recovery(context, recovery_fn)
    
    def _simulate_parameter_point(self, params: Dict[str, float]) -> SimulationData:
        """Run Monte Carlo simulation for a parameter point.
        
        Args:
            params: Parameter values for this point
            
        Returns:
            SimulationData containing configurations and observables
        """
        self.logger.info("Running Monte Carlo simulation...")
        
        # Extract simulation parameters
        sim_params = self.config.simulation_params
        lattice_size = sim_params.get('lattice_size', 32)
        n_temperatures = sim_params.get('n_temperatures', 20)
        n_samples = sim_params.get('n_samples', 100)
        n_equilibration = sim_params.get('n_equilibration', 1000)
        n_steps_between = sim_params.get('n_steps_between', 10)
        seed = sim_params.get('seed', None)
        
        # Generate temperature range
        # Use theoretical Tc if available, otherwise estimate
        if self.variant_config.theoretical_tc is not None:
            tc_estimate = self.variant_config.theoretical_tc
        else:
            tc_estimate = 4.5  # Default estimate
        
        t_min = sim_params.get('t_min', tc_estimate * 0.7)
        t_max = sim_params.get('t_max', tc_estimate * 1.3)
        temperatures = np.linspace(t_min, t_max, n_temperatures)
        
        # Storage for results
        all_configurations = []
        all_magnetizations = []
        all_energies = []
        
        # Run simulation at each temperature
        for temp in temperatures:
            self.logger.debug(f"  Temperature: {temp:.4f}")
            
            # Create simulator for this temperature
            simulator = self.registry.create_simulator(
                self.config.variant_id,
                lattice_size=lattice_size,
                temperature=temp,
                seed=seed
            )
            
            # Equilibrate
            simulator.equilibrate(n_equilibration)
            
            # Measure
            measurements = simulator.measure(n_samples, n_steps_between)
            
            all_configurations.append(measurements['configurations'])
            all_magnetizations.append(measurements['magnetizations'])
            all_energies.append(measurements['energies'])
        
        # Stack results
        configurations = np.array(all_configurations)  # (n_temps, n_samples, *lattice_shape)
        magnetizations = np.array(all_magnetizations)  # (n_temps, n_samples)
        energies = np.array(all_energies)  # (n_temps, n_samples)
        
        # Create SimulationData object
        sim_data = SimulationData(
            variant_id=self.config.variant_id,
            parameters=params,
            temperatures=temperatures,
            configurations=configurations,
            magnetizations=magnetizations,
            energies=energies,
            metadata={
                'lattice_size': lattice_size,
                'n_samples': n_samples,
                'n_equilibration': n_equilibration,
                'n_steps_between': n_steps_between,
                'seed': seed,
            }
        )
        
        self.logger.info(f"Simulation complete: {n_temperatures} temperatures, {n_samples} samples each")
        
        return sim_data

    def _analyze_with_vae_with_recovery(
        self,
        simulation_data: SimulationData
    ) -> Optional[VAEAnalysisResults]:
        """Apply VAE analysis with error recovery.
        
        Args:
            simulation_data: Monte Carlo simulation results
            
        Returns:
            VAEAnalysisResults or None if recovery failed
        """
        try:
            return self._analyze_with_vae(simulation_data)
        except Exception as e:
            self.logger.warning(f"VAE analysis error: {e}")
            
            # Create error context
            context = ErrorContext(
                category=self.error_recovery.categorize_error(e, "vae_training"),
                error=e,
                parameters=self.config.vae_config.copy(),
                attempt_number=1
            )
            
            # Execute recovery
            def recovery_fn(adjusted_params: Dict[str, Any]) -> VAEAnalysisResults:
                # Update config with adjusted parameters
                old_vae_config = self.config.vae_config.copy()
                old_analysis_config = self.config.analysis_config.copy()
                
                # Apply adjustments to both configs
                self.config.vae_config.update(adjusted_params)
                self.config.analysis_config.update(adjusted_params)
                
                # Update Prometheus config
                self.prometheus.config.vae_params.update(adjusted_params)
                self.prometheus.config.analysis_params.update(adjusted_params)
                
                try:
                    return self._analyze_with_vae(simulation_data)
                finally:
                    # Restore original parameters
                    self.config.vae_config = old_vae_config
                    self.config.analysis_config = old_analysis_config
            
            return self.error_recovery.execute_recovery(context, recovery_fn)
    
    def _analyze_with_vae(self, simulation_data: SimulationData) -> VAEAnalysisResults:
        """Apply Prometheus VAE analysis pipeline to simulation data.
        
        This method uses the PrometheusIntegration layer to apply the proven
        Prometheus VAE analysis pipeline with reproducibility guarantees.
        
        Args:
            simulation_data: Monte Carlo simulation results
            
        Returns:
            VAEAnalysisResults with extracted critical exponents
        """
        self.logger.info("Performing VAE analysis via Prometheus integration...")
        
        # Use Prometheus integration layer for analysis
        # This handles:
        # - Data format conversion
        # - Reproducibility (fixed random seeds)
        # - VAE training and latent extraction
        # - Critical exponent extraction
        # - Optional ensemble methods
        # - Result caching
        vae_results = self.prometheus.analyze_simulation_data(
            simulation_data,
            auto_detect_tc=True,
            compare_with_raw_mag=False
        )
        
        return vae_results
    
    def _detect_novel_phenomena(
        self,
        vae_results: VAEAnalysisResults,
        simulation_data: SimulationData
    ) -> List[NovelPhenomenon]:
        """Detect novel phenomena using comprehensive detector.
        
        This method uses the NovelPhenomenonDetector to identify various types
        of novel phase transition behavior including anomalous exponents,
        first-order transitions, and other unusual phenomena.
        
        Args:
            vae_results: VAE analysis results for this point
            simulation_data: Raw simulation data for additional checks
            
        Returns:
            List of detected novel phenomena
        """
        # Use the comprehensive phenomena detector
        phenomena = self.phenomena_detector.detect_all_phenomena(
            vae_results,
            simulation_data
        )
        
        # Log detected phenomena
        if phenomena:
            self.logger.info(f"Novel phenomena detection summary:")
            for p in phenomena:
                self.logger.info(f"  [{p.phenomenon_type}] {p.description}")
                self.logger.info(f"    Confidence: {p.confidence:.2%}")
        
        return phenomena
    
    def _estimate_uncertainty(self, vae_results: VAEAnalysisResults) -> float:
        """Estimate uncertainty at a parameter point.
        
        Args:
            vae_results: VAE analysis results
            
        Returns:
            Uncertainty estimate (0.0 to 1.0)
        """
        # Use R² as inverse measure of uncertainty
        # Low R² = high uncertainty
        r_squared = vae_results.r_squared_values.get('beta', 0.0)
        
        # Also consider confidence in Tc detection
        tc_confidence = vae_results.tc_confidence
        
        # Combined uncertainty (lower is better)
        uncertainty = 1.0 - (r_squared * tc_confidence)
        
        return max(0.0, min(1.0, uncertainty))
    
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save pipeline state for resumption.
        
        Args:
            checkpoint_path: Path to save checkpoint file
        """
        checkpoint_data = {
            'config': {
                'variant_id': self.config.variant_id,
                'exploration_strategy': {
                    'method': self.config.exploration_strategy.method,
                    'n_points': self.config.exploration_strategy.n_points,
                    'refinement_iterations': self.config.exploration_strategy.refinement_iterations,
                },
                'simulation_params': self.config.simulation_params,
                'vae_config': self.config.vae_config,
                'analysis_config': self.config.analysis_config,
                'checkpoint_interval': self.config.checkpoint_interval,
                'output_dir': self.config.output_dir,
            },
            'state': {
                'current_point_index': self._current_point_index,
                'start_time': self._start_time,
                'n_results': len(self._results),
                'n_phenomena': len(self._novel_phenomena),
            },
            'explorer_state': self.explorer.get_exploration_summary(),
        }
        
        # Save to file
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self.logger.debug(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load pipeline state from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        # Restore state
        state = checkpoint_data['state']
        self._current_point_index = state['current_point_index']
        self._start_time = state['start_time']
        
        # Note: Full restoration would require loading all results
        # For now, we just restore the index to continue from where we left off
        
        self.logger.info(f"Loaded checkpoint from point {self._current_point_index}")
    
    def _get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter ranges for exploration.
        
        Returns:
            Dictionary of parameter ranges
        """
        # For now, use temperature as the primary parameter
        # In full implementation, this would be configurable
        
        if self.variant_config.theoretical_tc is not None:
            tc = self.variant_config.theoretical_tc
        else:
            tc = 4.5  # Default
        
        return {
            'temperature': (tc * 0.7, tc * 1.3)
        }
    
    def _infer_system_type(self) -> str:
        """Infer system type from variant configuration.
        
        Returns:
            System type string for VAE analyzer
        """
        # Map variant configuration to system type
        if self.variant_config.dimensions == 2:
            return 'ising_2d'
        elif self.variant_config.dimensions == 3:
            return 'ising_3d'
        else:
            return 'ising_3d'  # Default
    
    def _save_results_summary(self, results: DiscoveryResults) -> None:
        """Save results summary to file.
        
        Args:
            results: Discovery results to summarize
        """
        summary_path = self.output_path / "results_summary.json"
        
        summary = {
            'variant_id': results.variant_id,
            'exploration_strategy': results.exploration_strategy.method,
            'n_points_explored': results.n_points_explored,
            'n_novel_phenomena': len(results.novel_phenomena),
            'execution_time': results.execution_time,
            'novel_phenomena': [
                {
                    'type': p.phenomenon_type,
                    'parameters': p.parameters,
                    'description': p.description,
                    'confidence': p.confidence,
                }
                for p in results.novel_phenomena
            ],
            'exponent_summary': self._summarize_exponents(results.vae_results),
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Results summary saved: {summary_path}")
    
    def _summarize_exponents(self, vae_results: List[VAEAnalysisResults]) -> Dict[str, Any]:
        """Summarize extracted exponents across all points.
        
        Args:
            vae_results: List of VAE analysis results
            
        Returns:
            Summary statistics for exponents
        """
        if not vae_results:
            return {}
        
        # Collect beta exponents
        beta_values = [r.exponents.get('beta', np.nan) for r in vae_results]
        beta_values = [b for b in beta_values if not np.isnan(b)]
        
        if not beta_values:
            return {}
        
        return {
            'beta': {
                'mean': float(np.mean(beta_values)),
                'std': float(np.std(beta_values)),
                'min': float(np.min(beta_values)),
                'max': float(np.max(beta_values)),
                'n_points': len(beta_values),
            }
        }
