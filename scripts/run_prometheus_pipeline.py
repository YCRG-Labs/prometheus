#!/usr/bin/env python3
"""
Prometheus Pipeline Orchestrator

This script coordinates the complete end-to-end pipeline for the Prometheus project:
1. Data generation and preprocessing
2. VAE model training
3. Latent space analysis and physics discovery
4. Visualization and reporting

The pipeline supports state management, resumption, error recovery, and comprehensive
configuration validation.
"""

import argparse
import sys
import os
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.config import PrometheusConfig, ConfigLoader
from utils.logging_utils import setup_logging
from utils.reproducibility import ReproducibilityManager


class PipelineState:
    """Manages pipeline execution state and progress tracking."""
    
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.state = {
            'pipeline_id': None,
            'start_time': None,
            'current_stage': None,
            'completed_stages': [],
            'failed_stages': [],
            'stage_outputs': {},
            'configuration': {},
            'error_log': []
        }
        self.load_state()
    
    def load_state(self):
        """Load pipeline state from file if it exists."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    self.state.update(json.load(f))
            except Exception as e:
                print(f"Warning: Could not load pipeline state: {e}")
    
    def save_state(self):
        """Save current pipeline state to file."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save pipeline state: {e}")
    
    def start_pipeline(self, pipeline_id: str, config: Dict):
        """Initialize a new pipeline run."""
        self.state.update({
            'pipeline_id': pipeline_id,
            'start_time': datetime.now().isoformat(),
            'current_stage': None,
            'completed_stages': [],
            'failed_stages': [],
            'stage_outputs': {},
            'configuration': config,
            'error_log': []
        })
        self.save_state()
    
    def start_stage(self, stage_name: str):
        """Mark the start of a pipeline stage."""
        self.state['current_stage'] = stage_name
        self.save_state()
    
    def complete_stage(self, stage_name: str, outputs: Dict):
        """Mark successful completion of a pipeline stage."""
        if stage_name not in self.state['completed_stages']:
            self.state['completed_stages'].append(stage_name)
        self.state['stage_outputs'][stage_name] = outputs
        self.state['current_stage'] = None
        if stage_name in self.state['failed_stages']:
            self.state['failed_stages'].remove(stage_name)
        self.save_state()
    
    def fail_stage(self, stage_name: str, error: str):
        """Mark failure of a pipeline stage."""
        if stage_name not in self.state['failed_stages']:
            self.state['failed_stages'].append(stage_name)
        self.state['error_log'].append({
            'stage': stage_name,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
        self.state['current_stage'] = None
        self.save_state()
    
    def is_stage_completed(self, stage_name: str) -> bool:
        """Check if a stage has been completed successfully."""
        return stage_name in self.state['completed_stages']
    
    def get_stage_output(self, stage_name: str) -> Optional[Dict]:
        """Get outputs from a completed stage."""
        return self.state['stage_outputs'].get(stage_name)


class PrometheusOrchestrator:
    """Main pipeline orchestrator for the Prometheus project."""
    
    def __init__(self, config: PrometheusConfig, state_file: Path):
        self.config = config
        self.state = PipelineState(state_file)
        self.scripts_dir = Path(__file__).parent
        self.project_root = self.scripts_dir.parent  # Go up one level from scripts/
        
        # Define pipeline stages
        self.stages = [
            'data_generation',
            'vae_training', 
            'latent_analysis',
            'visualization'
        ]
        
        # Stage dependencies
        self.dependencies = {
            'data_generation': [],
            'vae_training': ['data_generation'],
            'latent_analysis': ['vae_training'],
            'visualization': ['latent_analysis']
        }
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate pipeline configuration and dependencies."""
        errors = []
        
        # Check required directories
        required_dirs = [
            Path(self.config.data_dir),
            Path(self.config.models_dir),
            Path(self.config.results_dir)
        ]
        
        for dir_path in required_dirs:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create directory {dir_path}: {e}")
        
        # Check script files exist
        required_scripts = [
            'generate_data.py',
            'train_vae.py',
            'analyze_latent_space.py',
            'generate_visualizations.py'
        ]
        
        for script in required_scripts:
            script_path = self.scripts_dir / script
            if not script_path.exists():
                errors.append(f"Required script not found: {script_path}")
        
        # Validate configuration parameters
        if self.config.ising.lattice_size[0] <= 0 or self.config.ising.lattice_size[1] <= 0:
            errors.append("Invalid lattice size in configuration")
        
        if self.config.ising.n_temperatures <= 0:
            errors.append("Number of temperatures must be positive")
        
        if self.config.ising.n_configs_per_temp <= 0:
            errors.append("Number of configurations per temperature must be positive")
        
        if self.config.vae.latent_dim <= 0:
            errors.append("Latent dimension must be positive")
        
        if self.config.training.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        if self.config.training.num_epochs <= 0:
            errors.append("Number of epochs must be positive")
        
        return len(errors) == 0, errors
    
    def run_stage_script(self, script_name: str, args: List[str]) -> Tuple[bool, str, str]:
        """Run a pipeline stage script with error handling."""
        script_path = self.scripts_dir / script_name
        cmd = [sys.executable, str(script_path)] + args
        
        # Set up environment with PYTHONPATH to include project root
        env = os.environ.copy()
        project_root = str(self.project_root)
        
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = project_root
        
        try:
            print(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout
                env=env
            )
            
            success = result.returncode == 0
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", "Script execution timed out (2 hours)"
        except Exception as e:
            return False, "", f"Script execution failed: {e}"
    
    def run_data_generation(self, force: bool = False) -> bool:
        """Run data generation stage."""
        stage_name = 'data_generation'
        
        if self.state.is_stage_completed(stage_name) and not force:
            print(f"  Stage '{stage_name}' already completed, skipping...")
            return True
        
        print(f"\n{'='*60}")
        print(f"STAGE: Data Generation")
        print(f"{'='*60}")
        
        self.state.start_stage(stage_name)
        
        # Prepare arguments
        args = [
            '--output-dir', str(self.config.data_dir),
            '--parallel',
            '--validate'
        ]
        
        # Run data generation script
        success, stdout, stderr = self.run_stage_script('generate_data.py', args)
        
        if success:
            # Parse outputs from stdout to find dataset path
            dataset_path = None
            for line in stdout.split('\n'):
                if 'Processed data saved to:' in line:
                    dataset_path = line.split('Processed data saved to:')[1].strip()
                    break
            
            outputs = {
                'dataset_path': dataset_path,
                'data_dir': str(self.config.data_dir)
            }
            
            self.state.complete_stage(stage_name, outputs)
            print(f"✓ Data generation completed successfully")
            return True
        else:
            error_msg = f"Data generation failed:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            self.state.fail_stage(stage_name, error_msg)
            print(f"✗ Data generation failed")
            print(f"Error: {stderr}")
            return False
    
    def run_vae_training(self, force: bool = False) -> bool:
        """Run VAE training stage."""
        stage_name = 'vae_training'
        
        if self.state.is_stage_completed(stage_name) and not force:
            print(f"  Stage '{stage_name}' already completed, skipping...")
            return True
        
        print(f"\n{'='*60}")
        print(f"STAGE: VAE Training")
        print(f"{'='*60}")
        
        # Check dependencies
        data_outputs = self.state.get_stage_output('data_generation')
        if not data_outputs or not data_outputs.get('dataset_path'):
            print("✗ Data generation stage not completed or dataset path not found")
            return False
        
        dataset_path = data_outputs['dataset_path']
        if not Path(dataset_path).exists():
            print(f"✗ Dataset not found: {dataset_path}")
            return False
        
        self.state.start_stage(stage_name)
        
        # Prepare arguments
        args = [
            '--data', dataset_path,
            '--output-dir', str(self.config.models_dir)
        ]
        
        # Run VAE training script
        success, stdout, stderr = self.run_stage_script('train_vae.py', args)
        
        if success:
            # Parse outputs to find model path
            model_path = None
            for line in stdout.split('\n'):
                if 'Final model saved to:' in line:
                    model_path = line.split('Final model saved to:')[1].strip()
                    break
            
            outputs = {
                'model_path': model_path,
                'models_dir': str(self.config.models_dir)
            }
            
            self.state.complete_stage(stage_name, outputs)
            print(f"✓ VAE training completed successfully")
            return True
        else:
            error_msg = f"VAE training failed:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            self.state.fail_stage(stage_name, error_msg)
            print(f"✗ VAE training failed")
            print(f"Error: {stderr}")
            return False
    
    def run_latent_analysis(self, force: bool = False) -> bool:
        """Run latent space analysis stage."""
        stage_name = 'latent_analysis'
        
        if self.state.is_stage_completed(stage_name) and not force:
            print(f"  Stage '{stage_name}' already completed, skipping...")
            return True
        
        print(f"\n{'='*60}")
        print(f"STAGE: Latent Space Analysis")
        print(f"{'='*60}")
        
        # Check dependencies
        data_outputs = self.state.get_stage_output('data_generation')
        training_outputs = self.state.get_stage_output('vae_training')
        
        if not data_outputs or not training_outputs:
            print("✗ Required previous stages not completed")
            return False
        
        dataset_path = data_outputs['dataset_path']
        model_path = training_outputs['model_path']
        
        if not Path(dataset_path).exists():
            print(f"✗ Dataset not found: {dataset_path}")
            return False
        
        if not Path(model_path).exists():
            print(f"✗ Model not found: {model_path}")
            return False
        
        self.state.start_stage(stage_name)
        
        # Prepare arguments
        args = [
            '--model', model_path,
            '--data', dataset_path,
            '--output-dir', str(self.config.results_dir),
            '--save-latent-coords',
            '--include-temperature-labels'
        ]
        
        # Run analysis script
        success, stdout, stderr = self.run_stage_script('analyze_latent_space.py', args)
        
        if success:
            # Parse outputs
            results_file = Path(self.config.results_dir) / "analysis_results.json"
            detailed_file = Path(self.config.results_dir) / "detailed_analysis.npz"
            
            outputs = {
                'results_file': str(results_file),
                'detailed_file': str(detailed_file),
                'results_dir': str(self.config.results_dir)
            }
            
            self.state.complete_stage(stage_name, outputs)
            print(f"✓ Latent space analysis completed successfully")
            return True
        else:
            error_msg = f"Latent analysis failed:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            self.state.fail_stage(stage_name, error_msg)
            print(f"✗ Latent space analysis failed")
            print(f"Error: {stderr}")
            return False
    
    def run_visualization(self, force: bool = False) -> bool:
        """Run visualization generation stage."""
        stage_name = 'visualization'
        
        if self.state.is_stage_completed(stage_name) and not force:
            print(f"  Stage '{stage_name}' already completed, skipping...")
            return True
        
        print(f"\n{'='*60}")
        print(f"STAGE: Visualization Generation")
        print(f"{'='*60}")
        
        # Check dependencies
        analysis_outputs = self.state.get_stage_output('latent_analysis')
        
        if not analysis_outputs:
            print("✗ Latent analysis stage not completed")
            return False
        
        results_file = analysis_outputs['results_file']
        detailed_file = analysis_outputs['detailed_file']
        
        if not Path(results_file).exists():
            print(f"✗ Analysis results not found: {results_file}")
            return False
        
        if not Path(detailed_file).exists():
            print(f"✗ Detailed analysis data not found: {detailed_file}")
            return False
        
        self.state.start_stage(stage_name)
        
        # Prepare arguments
        args = [
            '--analysis-results', results_file,
            '--detailed-data', detailed_file,
            '--output-dir', str(self.config.results_dir),
            '--format', 'png',
            '--style', 'publication',
            '--no-show'
        ]
        
        # Run visualization script
        success, stdout, stderr = self.run_stage_script('generate_visualizations.py', args)
        
        if success:
            outputs = {
                'figures_dir': str(self.config.results_dir),
                'report_file': str(Path(self.config.results_dir) / "analysis_report.md"),
                'index_file': str(Path(self.config.results_dir) / "figure_index.md")
            }
            
            self.state.complete_stage(stage_name, outputs)
            print(f"✓ Visualization generation completed successfully")
            return True
        else:
            error_msg = f"Visualization failed:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            self.state.fail_stage(stage_name, error_msg)
            print(f"✗ Visualization generation failed")
            print(f"Error: {stderr}")
            return False
    
    def run_pipeline(self, stages: Optional[List[str]] = None, force_stages: Optional[List[str]] = None) -> bool:
        """Run the complete pipeline or specified stages."""
        
        if stages is None:
            stages = self.stages
        
        if force_stages is None:
            force_stages = []
        
        print(f"\n{'='*60}")
        print(f"PROMETHEUS PIPELINE EXECUTION")
        print(f"{'='*60}")
        print(f"Pipeline ID: {self.state.state['pipeline_id']}")
        print(f"Stages to run: {', '.join(stages)}")
        if force_stages:
            print(f"Force re-run: {', '.join(force_stages)}")
        print()
        
        # Stage execution mapping
        stage_functions = {
            'data_generation': self.run_data_generation,
            'vae_training': self.run_vae_training,
            'latent_analysis': self.run_latent_analysis,
            'visualization': self.run_visualization
        }
        
        success = True
        
        for stage in stages:
            if stage not in stage_functions:
                print(f"✗ Unknown stage: {stage}")
                success = False
                continue
            
            # Check dependencies
            for dep in self.dependencies.get(stage, []):
                if not self.state.is_stage_completed(dep):
                    print(f"✗ Stage '{stage}' requires '{dep}' to be completed first")
                    success = False
                    break
            
            if not success:
                break
            
            # Run stage
            force = stage in force_stages
            stage_success = stage_functions[stage](force=force)
            
            if not stage_success:
                success = False
                break
        
        return success
    
    def print_pipeline_status(self):
        """Print current pipeline status."""
        print(f"\n{'='*60}")
        print(f"PIPELINE STATUS")
        print(f"{'='*60}")
        
        if self.state.state['pipeline_id']:
            print(f"Pipeline ID: {self.state.state['pipeline_id']}")
            print(f"Started: {self.state.state['start_time']}")
            print(f"Current stage: {self.state.state['current_stage'] or 'None'}")
            print()
            
            print("Stage Status:")
            for stage in self.stages:
                if stage in self.state.state['completed_stages']:
                    status = "✓ COMPLETED"
                elif stage in self.state.state['failed_stages']:
                    status = "✗ FAILED"
                elif stage == self.state.state['current_stage']:
                    status = "⏳ RUNNING"
                else:
                    status = "⏸ PENDING"
                
                print(f"  {stage}: {status}")
            
            if self.state.state['error_log']:
                print(f"\nRecent Errors:")
                for error in self.state.state['error_log'][-3:]:  # Show last 3 errors
                    print(f"  {error['stage']}: {error['error'][:100]}...")
        else:
            print("No pipeline execution found.")


def main():
    parser = argparse.ArgumentParser(description='Prometheus Pipeline Orchestrator')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--stages', type=str, nargs='+', 
                       choices=['data_generation', 'vae_training', 'latent_analysis', 'visualization'],
                       help='Specific stages to run (default: all)')
    parser.add_argument('--force', type=str, nargs='+',
                       choices=['data_generation', 'vae_training', 'latent_analysis', 'visualization'],
                       help='Force re-run of specified stages')
    parser.add_argument('--resume', action='store_true', help='Resume from previous pipeline state')
    parser.add_argument('--status', action='store_true', help='Show pipeline status and exit')
    parser.add_argument('--reset', action='store_true', help='Reset pipeline state')
    parser.add_argument('--state-file', type=str, default='.prometheus_pipeline_state.json',
                       help='Pipeline state file path')
    parser.add_argument('--validate-only', action='store_true', help='Only validate configuration')
    
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader()
    if args.config:
        config = config_loader.load_config(args.config)
    else:
        config = PrometheusConfig()
    
    # Setup logging
    setup_logging(config.logging)
    
    # Set random seeds
    repro_manager = ReproducibilityManager(config.seed)
    repro_manager.set_seeds()
    
    # Initialize orchestrator
    state_file = Path(args.state_file)
    orchestrator = PrometheusOrchestrator(config, state_file)
    
    # Handle status request
    if args.status:
        orchestrator.print_pipeline_status()
        return
    
    # Handle reset request
    if args.reset:
        if state_file.exists():
            state_file.unlink()
            print("Pipeline state reset.")
        else:
            print("No pipeline state to reset.")
        return
    
    # Validate configuration
    print("Validating pipeline configuration...")
    is_valid, errors = orchestrator.validate_configuration()
    
    if not is_valid:
        print("✗ Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        if not args.validate_only:
            sys.exit(1)
        return
    
    print("✓ Configuration validation passed")
    
    if args.validate_only:
        print("Configuration is valid. Pipeline ready to run.")
        return
    
    # Initialize pipeline if not resuming
    if not args.resume or not orchestrator.state.state['pipeline_id']:
        pipeline_id = f"prometheus_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        orchestrator.state.start_pipeline(pipeline_id, config.__dict__)
        print(f"Started new pipeline: {pipeline_id}")
    else:
        print(f"Resuming pipeline: {orchestrator.state.state['pipeline_id']}")
    
    # Run pipeline
    try:
        success = orchestrator.run_pipeline(
            stages=args.stages,
            force_stages=args.force or []
        )
        
        if success:
            print(f"\n{'='*60}")
            print(f"PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            
            # Print final results summary
            analysis_outputs = orchestrator.state.get_stage_output('latent_analysis')
            if analysis_outputs and Path(analysis_outputs['results_file']).exists():
                with open(analysis_outputs['results_file'], 'r') as f:
                    results = json.load(f)
                
                print(f"Key Findings:")
                print(f"  • Critical Temperature: {results['phase_detection']['critical_temperature']:.4f}")
                print(f"  • Theoretical: {config.ising.critical_temp:.4f}")
                error_pct = abs(results['phase_detection']['critical_temperature'] - config.ising.critical_temp) / config.ising.critical_temp * 100
                print(f"  • Relative Error: {error_pct:.2f}%")
                print(f"  • Order Parameter Correlation: {results['order_parameters']['magnetization_correlation']:.4f}")
                print(f"  • Phase Separation Quality: {results['clustering']['silhouette_score']:.4f}")
            
            viz_outputs = orchestrator.state.get_stage_output('visualization')
            if viz_outputs:
                print(f"\nResults available at:")
                print(f"  • Analysis Report: {viz_outputs['report_file']}")
                print(f"  • Figure Index: {viz_outputs['index_file']}")
                print(f"  • Results Directory: {viz_outputs['figures_dir']}")
        
        else:
            print(f"\n{'='*60}")
            print(f"PIPELINE FAILED")
            print(f"{'='*60}")
            orchestrator.print_pipeline_status()
            sys.exit(1)
    
    except KeyboardInterrupt:
        print(f"\nPipeline interrupted by user.")
        print(f"State saved. Use --resume to continue from where you left off.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\nPipeline failed with unexpected error: {e}")
        orchestrator.print_pipeline_status()
        sys.exit(1)


if __name__ == "__main__":
    main()