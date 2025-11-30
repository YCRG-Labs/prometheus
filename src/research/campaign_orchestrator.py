"""
Discovery Campaign Orchestrator for systematic exploration of Ising model variants.

This module coordinates the entire discovery campaign workflow, including target
selection, exploration planning, validation, discovery assessment, and publication
generation.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np

from .base_types import (
    ModelVariantConfig,
    SimulationData,
    VAEAnalysisResults,
    DiscoveryResults,
    ExplorationStrategy,
)
from .model_registry import ModelVariantRegistry
from .discovery_pipeline import DiscoveryPipeline, DiscoveryConfig
from .unified_validation_pipeline import UnifiedValidationPipeline
from .discovery_assessor import DiscoveryAssessor, PhysicsDiscovery
from .publication_generator import PublicationGenerator
from ..utils.logging_utils import get_logger


@dataclass
class TargetVariant:
    """A variant selected for exploration.
    
    Attributes:
        variant_id: Unique identifier for the variant
        name: Human-readable name
        description: Description of the variant
        model_config: Model configuration
        theoretical_predictions: Predicted critical exponents (if available)
        literature_references: List of relevant papers
        priority_score: Discovery potential score (0.0 to 1.0)
        estimated_compute_hours: Estimated computational cost
        discovery_potential: Potential for novel physics (0.0 to 1.0)
    """
    variant_id: str
    name: str
    description: str
    model_config: ModelVariantConfig
    theoretical_predictions: Optional[Dict[str, float]] = None
    literature_references: List[str] = field(default_factory=list)
    priority_score: float = 0.0
    estimated_compute_hours: float = 0.0
    discovery_potential: float = 0.0


@dataclass
class CampaignConfig:
    """Configuration for discovery campaign.
    
    Attributes:
        campaign_name: Name of the campaign
        computational_budget: Total GPU-hours available
        target_variants: List of variant IDs to explore
        validation_threshold: Minimum confidence for novel physics claims
        output_directory: Directory for all campaign outputs
        enable_parallel: Enable parallel execution
        max_parallel_tasks: Maximum parallel tasks
        checkpoint_interval: Points between checkpoints
    """
    campaign_name: str
    computational_budget: float
    target_variants: List[str]
    validation_threshold: float = 0.90
    output_directory: str = 'results/discovery_campaign'
    enable_parallel: bool = False
    max_parallel_tasks: int = 4
    checkpoint_interval: int = 10


@dataclass
class CampaignResults:
    """Results from complete discovery campaign.
    
    Attributes:
        campaign_name: Name of the campaign
        variants_explored: List of variant IDs explored
        total_points_explored: Total parameter points explored
        discoveries: List of validated discoveries
        execution_time: Total execution time in seconds
        computational_cost: Total GPU-hours used
        checkpoint_path: Path to final checkpoint
    """
    campaign_name: str
    variants_explored: List[str]
    total_points_explored: int
    discoveries: List[Any]  # Will be PhysicsDiscovery objects
    execution_time: float
    computational_cost: float
    checkpoint_path: str


class DiscoveryCampaignOrchestrator:
    """Orchestrate entire discovery campaign across multiple variants.
    
    This class coordinates the systematic exploration of multiple Ising model
    variants, applying validation patterns and generating publication-ready
    outputs for all discoveries.
    
    Attributes:
        config: Campaign configuration
        registry: Model variant registry
        logger: Logger instance
        output_path: Path to campaign output directory
        campaign_db: Database of all campaign results
    """
    
    def __init__(self, config: CampaignConfig):
        """Initialize campaign orchestrator.
        
        Args:
            config: Campaign configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Create output directory structure
        self.output_path = Path(config.output_directory) / config.campaign_name
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_path / 'variants').mkdir(exist_ok=True)
        (self.output_path / 'discoveries').mkdir(exist_ok=True)
        (self.output_path / 'publications').mkdir(exist_ok=True)
        (self.output_path / 'checkpoints').mkdir(exist_ok=True)
        (self.output_path / 'provenance').mkdir(exist_ok=True)
        
        # Initialize model registry
        self.registry = ModelVariantRegistry()
        
        # Initialize validation and assessment components
        self.validation_pipeline = UnifiedValidationPipeline()
        self.discovery_assessor = DiscoveryAssessor(
            novelty_threshold=3.0,
            confidence_threshold=config.validation_threshold
        )
        self.publication_generator = PublicationGenerator()
        
        # Campaign state
        self.campaign_db = {
            'variants': {},
            'discoveries': [],
            'provenance': [],
            'start_time': None,
            'total_compute_hours': 0.0,
            'progress': {
                'current_variant': None,
                'variants_completed': 0,
                'total_variants': len(config.target_variants),
                'discoveries_found': 0,
            }
        }
        
        self.logger.info(f"Initialized discovery campaign: {config.campaign_name}")
        self.logger.info(f"Output directory: {self.output_path}")
        self.logger.info(f"Computational budget: {config.computational_budget} GPU-hours")
        self.logger.info(f"Validation threshold: {config.validation_threshold}")
        self.logger.info(f"Target variants: {len(config.target_variants)}")
    
    def run_campaign(self) -> CampaignResults:
        """Execute complete discovery campaign.
        
        Returns:
            CampaignResults with all discoveries and statistics
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Starting Discovery Campaign: {self.config.campaign_name}")
        self.logger.info("=" * 80)
        
        start_time = time.time()
        self.campaign_db['start_time'] = start_time
        
        # Save campaign configuration
        self._save_campaign_config()
        
        # Initialize provenance tracking
        self._initialize_provenance()
        
        # Explore each target variant
        variants_explored = []
        total_points = 0
        
        for variant_id in self.config.target_variants:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Exploring variant: {variant_id}")
            self.logger.info(f"{'='*80}")
            
            try:
                # Explore this variant
                variant_results = self.explore_variant(variant_id)
                
                # Store results in campaign database
                self.campaign_db['variants'][variant_id] = {
                    'n_points_explored': variant_results.n_points_explored,
                    'novel_phenomena': len(variant_results.novel_phenomena),
                    'execution_time': variant_results.execution_time,
                }
                
                variants_explored.append(variant_id)
                total_points += variant_results.n_points_explored
                
                # Save checkpoint after each variant
                self._save_campaign_checkpoint()
                
            except Exception as e:
                self.logger.error(f"Error exploring variant {variant_id}: {e}", exc_info=True)
                continue
        
        # Calculate total execution time
        execution_time = time.time() - start_time
        
        # Create campaign results
        results = CampaignResults(
            campaign_name=self.config.campaign_name,
            variants_explored=variants_explored,
            total_points_explored=total_points,
            discoveries=self.campaign_db['discoveries'],
            execution_time=execution_time,
            computational_cost=self.campaign_db['total_compute_hours'],
            checkpoint_path=str(self.output_path / 'checkpoints' / 'final_checkpoint.json')
        )
        
        # Save final results
        self._save_campaign_results(results)
        
        self.logger.info("=" * 80)
        self.logger.info("Discovery Campaign Complete")
        self.logger.info(f"Variants explored: {len(variants_explored)}")
        self.logger.info(f"Total points explored: {total_points}")
        self.logger.info(f"Discoveries: {len(results.discoveries)}")
        self.logger.info(f"Execution time: {execution_time/3600:.2f} hours")
        self.logger.info(f"Computational cost: {results.computational_cost:.1f} GPU-hours")
        self.logger.info("=" * 80)
        
        return results
    
    def explore_variant(self, variant_id: str) -> DiscoveryResults:
        """Explore a single variant completely.
        
        This method:
        1. Executes exploration plan
        2. Applies validation pipeline
        3. Assesses for novel physics
        4. Generates publications if discoveries found
        
        Args:
            variant_id: ID of variant to explore
            
        Returns:
            DiscoveryResults for this variant
        """
        self.logger.info(f"Setting up exploration for variant: {variant_id}")
        
        # Update progress
        self.campaign_db['progress']['current_variant'] = variant_id
        
        # Create exploration configuration
        # Use default parameters for now - will be customized per variant later
        exploration_strategy = ExplorationStrategy(
            method='adaptive',
            n_points=100,
            refinement_iterations=2
        )
        
        simulation_params = {
            'lattice_size': 32,
            'n_temperatures': 50,
            'n_samples': 500,
            'n_equilibration': 2000,
            'n_steps_between': 10,
            'seed': 42,
        }
        
        vae_config = {
            'latent_dim': 8,
            'hidden_dims': [64, 32],
            'learning_rate': 1e-3,
            'batch_size': 32,
            'n_epochs': 100,
            'use_gpu': True,
        }
        
        analysis_config = {
            'use_ensemble': True,
            'enable_caching': True,
            'anomaly_threshold': 3.0,
            'max_retries': 3,
        }
        
        discovery_config = DiscoveryConfig(
            variant_id=variant_id,
            exploration_strategy=exploration_strategy,
            simulation_params=simulation_params,
            vae_config=vae_config,
            analysis_config=analysis_config,
            checkpoint_interval=self.config.checkpoint_interval,
            output_dir=str(self.output_path / 'variants'),
            enable_parallel=self.config.enable_parallel,
            max_parallel_tasks=self.config.max_parallel_tasks,
        )
        
        # Create discovery pipeline
        pipeline = DiscoveryPipeline(discovery_config, self.registry)
        
        # Run exploration
        self.logger.info(f"Running exploration for {variant_id}...")
        results = pipeline.run_exploration()
        
        # Apply validation and assessment to each novel phenomenon
        for phenomenon in results.novel_phenomena:
            self.logger.info(f"Validating phenomenon: {phenomenon.phenomenon_type}")
            
            # Apply validation pipeline
            validation_report = self.validation_pipeline.validate(
                results.vae_results,
                results.simulation_data
            )
            
            # Assess for novel physics
            discovery = self.discovery_assessor.assess_novelty(
                results.vae_results,
                validation_report,
                theoretical_predictions=None,  # TODO: Add theoretical predictions
                variant_description=variant_id
            )
            
            # If novel physics discovered, generate publication
            if discovery:
                self.logger.info(
                    f"Novel physics discovered! Type: {discovery.discovery_type}, "
                    f"Significance: {discovery.significance}"
                )
                
                # Store discovery
                self.campaign_db['discoveries'].append(discovery)
                self.campaign_db['progress']['discoveries_found'] += 1
                
                # Generate publication package
                self.logger.info("Generating publication package...")
                pub_package = self.publication_generator.generate_package(
                    discovery=discovery,
                    vae_results=results.vae_results,
                    validation_report=validation_report,
                    simulation_data=results.simulation_data,
                    output_dir=self.output_path / 'publications' / discovery.discovery_id
                )
                
                self.logger.info(f"Publication package generated: {pub_package.metadata['output_directory']}")
        
        # Update progress
        self.campaign_db['progress']['variants_completed'] += 1
        
        # Record provenance
        self._record_variant_provenance(variant_id, discovery_config, results)
        
        # Generate progress report
        self._generate_progress_report()
        
        return results
    
    def _save_campaign_config(self) -> None:
        """Save campaign configuration to file."""
        config_path = self.output_path / 'campaign_config.json'
        
        config_dict = {
            'campaign_name': self.config.campaign_name,
            'computational_budget': self.config.computational_budget,
            'target_variants': self.config.target_variants,
            'validation_threshold': self.config.validation_threshold,
            'enable_parallel': self.config.enable_parallel,
            'max_parallel_tasks': self.config.max_parallel_tasks,
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"Saved campaign configuration: {config_path}")
    
    def _initialize_provenance(self) -> None:
        """Initialize provenance tracking system."""
        provenance_path = self.output_path / 'provenance' / 'campaign_provenance.json'
        
        provenance_data = {
            'campaign_name': self.config.campaign_name,
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'configuration': {
                'computational_budget': self.config.computational_budget,
                'validation_threshold': self.config.validation_threshold,
            },
            'variants': [],
        }
        
        with open(provenance_path, 'w') as f:
            json.dump(provenance_data, f, indent=2)
        
        self.logger.info(f"Initialized provenance tracking: {provenance_path}")
    
    def _record_variant_provenance(
        self,
        variant_id: str,
        config: DiscoveryConfig,
        results: DiscoveryResults
    ) -> None:
        """Record provenance for a variant exploration.
        
        Args:
            variant_id: Variant ID
            config: Discovery configuration used
            results: Exploration results
        """
        provenance_path = self.output_path / 'provenance' / f'{variant_id}_provenance.json'
        
        provenance_data = {
            'variant_id': variant_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'configuration': {
                'exploration_strategy': {
                    'method': config.exploration_strategy.method,
                    'n_points': config.exploration_strategy.n_points,
                },
                'simulation_params': config.simulation_params,
                'vae_config': config.vae_config,
            },
            'results': {
                'n_points_explored': results.n_points_explored,
                'novel_phenomena': len(results.novel_phenomena),
                'execution_time': results.execution_time,
            },
        }
        
        with open(provenance_path, 'w') as f:
            json.dump(provenance_data, f, indent=2)
        
        self.logger.info(f"Recorded provenance: {provenance_path}")
    
    def _save_campaign_checkpoint(self) -> None:
        """Save campaign checkpoint."""
        checkpoint_path = self.output_path / 'checkpoints' / f'checkpoint_{int(time.time())}.json'
        
        with open(checkpoint_path, 'w') as f:
            json.dump(self.campaign_db, f, indent=2)
        
        self.logger.info(f"Saved campaign checkpoint: {checkpoint_path}")
    
    def _save_campaign_results(self, results: CampaignResults) -> None:
        """Save final campaign results.
        
        Args:
            results: Campaign results to save
        """
        results_path = self.output_path / 'campaign_results.json'
        
        results_dict = {
            'campaign_name': results.campaign_name,
            'variants_explored': results.variants_explored,
            'total_points_explored': results.total_points_explored,
            'n_discoveries': len(results.discoveries),
            'execution_time': results.execution_time,
            'computational_cost': results.computational_cost,
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        self.logger.info(f"Saved campaign results: {results_path}")

    
    def _generate_progress_report(self) -> None:
        """Generate progress report for the campaign.
        
        This method implements Task 7.2: Progress monitoring
        """
        progress = self.campaign_db['progress']
        
        report_path = self.output_path / 'progress_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Discovery Campaign Progress Report: {self.config.campaign_name}\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall progress
            f.write("Overall Progress:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Variants completed: {progress['variants_completed']} / {progress['total_variants']}\n")
            completion_pct = (progress['variants_completed'] / progress['total_variants']) * 100
            f.write(f"Completion: {completion_pct:.1f}%\n")
            f.write(f"Current variant: {progress['current_variant']}\n")
            f.write(f"Discoveries found: {progress['discoveries_found']}\n\n")
            
            # Computational resources
            f.write("Computational Resources:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total compute used: {self.campaign_db['total_compute_hours']:.1f} GPU-hours\n")
            f.write(f"Budget: {self.config.computational_budget:.1f} GPU-hours\n")
            budget_used_pct = (self.campaign_db['total_compute_hours'] / self.config.computational_budget) * 100
            f.write(f"Budget used: {budget_used_pct:.1f}%\n\n")
            
            # Time estimation
            if self.campaign_db['start_time'] and progress['variants_completed'] > 0:
                elapsed_time = time.time() - self.campaign_db['start_time']
                avg_time_per_variant = elapsed_time / progress['variants_completed']
                remaining_variants = progress['total_variants'] - progress['variants_completed']
                estimated_remaining_time = avg_time_per_variant * remaining_variants
                
                f.write("Time Estimation:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Elapsed time: {elapsed_time/3600:.2f} hours\n")
                f.write(f"Avg time per variant: {avg_time_per_variant/3600:.2f} hours\n")
                f.write(f"Estimated remaining: {estimated_remaining_time/3600:.2f} hours\n")
                f.write(f"Estimated completion: {(elapsed_time + estimated_remaining_time)/3600:.2f} hours from start\n\n")
            
            # Discoveries summary
            if self.campaign_db['discoveries']:
                f.write("Discoveries Summary:\n")
                f.write("-" * 40 + "\n")
                for i, discovery in enumerate(self.campaign_db['discoveries'], 1):
                    f.write(f"{i}. {discovery.variant_id}\n")
                    f.write(f"   Type: {discovery.discovery_type}\n")
                    f.write(f"   Significance: {discovery.significance}\n")
                    f.write(f"   Publication potential: {discovery.publication_potential}\n")
                    f.write(f"   Confidence: {discovery.validation_confidence:.2%}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write(f"Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
        
        self.logger.info(f"Generated progress report: {report_path}")
    
    def save_checkpoint(self, checkpoint_name: Optional[str] = None) -> Path:
        """Save campaign checkpoint for resumption.
        
        This method implements Task 7.3: Checkpoint and resumption
        
        Args:
            checkpoint_name: Optional name for checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_{int(time.time())}"
        
        checkpoint_path = self.output_path / 'checkpoints' / f'{checkpoint_name}.json'
        
        # Prepare checkpoint data
        checkpoint_data = {
            'campaign_name': self.config.campaign_name,
            'checkpoint_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'computational_budget': self.config.computational_budget,
                'target_variants': self.config.target_variants,
                'validation_threshold': self.config.validation_threshold,
            },
            'state': {
                'variants_completed': self.campaign_db['progress']['variants_completed'],
                'current_variant': self.campaign_db['progress']['current_variant'],
                'discoveries_found': self.campaign_db['progress']['discoveries_found'],
                'total_compute_hours': self.campaign_db['total_compute_hours'],
                'start_time': self.campaign_db['start_time'],
            },
            'variants': self.campaign_db['variants'],
            'discoveries': [
                {
                    'discovery_id': d.discovery_id,
                    'variant_id': d.variant_id,
                    'discovery_type': d.discovery_type,
                    'significance': d.significance,
                    'publication_potential': d.publication_potential,
                    'validation_confidence': float(d.validation_confidence),
                }
                for d in self.campaign_db['discoveries']
            ],
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load campaign checkpoint for resumption.
        
        This method implements Task 7.3: Checkpoint and resumption
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        # Restore campaign state
        self.campaign_db['start_time'] = checkpoint_data['state']['start_time']
        self.campaign_db['total_compute_hours'] = checkpoint_data['state']['total_compute_hours']
        self.campaign_db['progress']['variants_completed'] = checkpoint_data['state']['variants_completed']
        self.campaign_db['progress']['current_variant'] = checkpoint_data['state']['current_variant']
        self.campaign_db['progress']['discoveries_found'] = checkpoint_data['state']['discoveries_found']
        self.campaign_db['variants'] = checkpoint_data['variants']
        
        # Note: Full discovery objects would need to be reconstructed from saved files
        # For now, we just track the count
        
        self.logger.info(f"Checkpoint loaded successfully")
        self.logger.info(f"Resuming from variant {checkpoint_data['state']['variants_completed']} of {len(self.config.target_variants)}")
    
    def resume_campaign(self, checkpoint_path: Path) -> CampaignResults:
        """Resume campaign from checkpoint.
        
        This method implements Task 7.3: Checkpoint and resumption
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            CampaignResults from resumed campaign
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Resuming Discovery Campaign: {self.config.campaign_name}")
        self.logger.info("=" * 80)
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        
        # Determine which variants still need to be explored
        completed_count = self.campaign_db['progress']['variants_completed']
        remaining_variants = self.config.target_variants[completed_count:]
        
        self.logger.info(f"Resuming with {len(remaining_variants)} variants remaining")
        
        # Continue exploration
        variants_explored = self.config.target_variants[:completed_count]
        total_points = sum(
            self.campaign_db['variants'][v]['n_points_explored']
            for v in variants_explored
            if v in self.campaign_db['variants']
        )
        
        for variant_id in remaining_variants:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Exploring variant: {variant_id}")
            self.logger.info(f"{'='*80}")
            
            try:
                # Explore this variant
                variant_results = self.explore_variant(variant_id)
                
                # Store results
                self.campaign_db['variants'][variant_id] = {
                    'n_points_explored': variant_results.n_points_explored,
                    'novel_phenomena': len(variant_results.novel_phenomena),
                    'execution_time': variant_results.execution_time,
                }
                
                variants_explored.append(variant_id)
                total_points += variant_results.n_points_explored
                
                # Save checkpoint after each variant
                self.save_checkpoint()
                
            except Exception as e:
                self.logger.error(f"Error exploring variant {variant_id}: {e}", exc_info=True)
                # Save checkpoint even on error
                self.save_checkpoint(f"error_checkpoint_{variant_id}")
                continue
        
        # Calculate total execution time
        execution_time = time.time() - self.campaign_db['start_time']
        
        # Create campaign results
        results = CampaignResults(
            campaign_name=self.config.campaign_name,
            variants_explored=variants_explored,
            total_points_explored=total_points,
            discoveries=self.campaign_db['discoveries'],
            execution_time=execution_time,
            computational_cost=self.campaign_db['total_compute_hours'],
            checkpoint_path=str(self.output_path / 'checkpoints' / 'final_checkpoint.json')
        )
        
        # Save final results
        self._save_campaign_results(results)
        
        self.logger.info("=" * 80)
        self.logger.info("Discovery Campaign Complete (Resumed)")
        self.logger.info(f"Variants explored: {len(variants_explored)}")
        self.logger.info(f"Total points explored: {total_points}")
        self.logger.info(f"Discoveries: {len(results.discoveries)}")
        self.logger.info(f"Total execution time: {execution_time/3600:.2f} hours")
        self.logger.info(f"Computational cost: {results.computational_cost:.1f} GPU-hours")
        self.logger.info("=" * 80)
        
        return results
    
    def monitor_resources(self) -> Dict[str, Any]:
        """Monitor computational resource usage.
        
        This method implements Task 7.2: Progress monitoring
        
        Returns:
            Dictionary with resource usage statistics
        """
        progress = self.campaign_db['progress']
        
        # Calculate resource usage
        compute_used = self.campaign_db['total_compute_hours']
        compute_budget = self.config.computational_budget
        compute_remaining = compute_budget - compute_used
        compute_used_pct = (compute_used / compute_budget) * 100 if compute_budget > 0 else 0
        
        # Calculate time statistics
        elapsed_time = 0
        avg_time_per_variant = 0
        estimated_remaining_time = 0
        
        if self.campaign_db['start_time'] and progress['variants_completed'] > 0:
            elapsed_time = time.time() - self.campaign_db['start_time']
            avg_time_per_variant = elapsed_time / progress['variants_completed']
            remaining_variants = progress['total_variants'] - progress['variants_completed']
            estimated_remaining_time = avg_time_per_variant * remaining_variants
        
        resource_stats = {
            'computational_resources': {
                'compute_used_gpu_hours': compute_used,
                'compute_budget_gpu_hours': compute_budget,
                'compute_remaining_gpu_hours': compute_remaining,
                'compute_used_percent': compute_used_pct,
            },
            'time_statistics': {
                'elapsed_time_hours': elapsed_time / 3600,
                'avg_time_per_variant_hours': avg_time_per_variant / 3600,
                'estimated_remaining_hours': estimated_remaining_time / 3600,
                'estimated_total_hours': (elapsed_time + estimated_remaining_time) / 3600,
            },
            'progress': {
                'variants_completed': progress['variants_completed'],
                'total_variants': progress['total_variants'],
                'completion_percent': (progress['variants_completed'] / progress['total_variants']) * 100,
                'current_variant': progress['current_variant'],
                'discoveries_found': progress['discoveries_found'],
            },
        }
        
        return resource_stats
    
    def estimate_time_to_completion(self) -> float:
        """Estimate time to campaign completion.
        
        This method implements Task 7.2: Progress monitoring
        
        Returns:
            Estimated hours to completion
        """
        progress = self.campaign_db['progress']
        
        if not self.campaign_db['start_time'] or progress['variants_completed'] == 0:
            return 0.0
        
        elapsed_time = time.time() - self.campaign_db['start_time']
        avg_time_per_variant = elapsed_time / progress['variants_completed']
        remaining_variants = progress['total_variants'] - progress['variants_completed']
        estimated_remaining_time = avg_time_per_variant * remaining_variants
        
        return estimated_remaining_time / 3600  # Convert to hours
    
    def handle_failure(self, variant_id: str, error: Exception) -> None:
        """Handle failure during variant exploration.
        
        This method implements Task 7.3: Checkpoint and resumption (graceful failure handling)
        
        Args:
            variant_id: ID of variant that failed
            error: Exception that occurred
        """
        self.logger.error(f"Failure during exploration of {variant_id}: {error}", exc_info=True)
        
        # Save error checkpoint
        error_checkpoint_path = self.save_checkpoint(f"error_{variant_id}_{int(time.time())}")
        
        # Record failure in campaign database
        if 'failures' not in self.campaign_db:
            self.campaign_db['failures'] = []
        
        self.campaign_db['failures'].append({
            'variant_id': variant_id,
            'error': str(error),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'checkpoint': str(error_checkpoint_path),
        })
        
        # Save failure report
        failure_report_path = self.output_path / 'failures' / f'{variant_id}_failure.txt'
        failure_report_path.parent.mkdir(exist_ok=True)
        
        with open(failure_report_path, 'w') as f:
            f.write(f"Failure Report: {variant_id}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error: {error}\n\n")
            f.write(f"Checkpoint saved: {error_checkpoint_path}\n\n")
            f.write("Campaign can be resumed from this checkpoint.\n")
        
        self.logger.info(f"Failure handled. Checkpoint saved: {error_checkpoint_path}")
        self.logger.info(f"Failure report saved: {failure_report_path}")
