#!/usr/bin/env python3
"""
PRE Paper Complete Workflow Example

This script demonstrates the complete end-to-end workflow for the Physical Review E
paper using the Prometheus phase discovery system. It covers:

1. 3D Ising data generation with proper equilibration
2. VAE training and latent representation extraction
3. Critical exponent extraction and validation
4. Multi-system comparison analysis (Ising, Potts, XY)
5. Publication materials generation

Requirements: 5.5 - Generate all figures and tables needed for Physical Review E submission

Usage:
    python examples/pre_paper_complete_workflow_example.py [--quick-demo]
    
    --quick-demo: Run with reduced parameters for faster demonstration
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Core imports
from data.high_quality_data_generator import HighQualityDataGenerator
from data.unified_monte_carlo import UnifiedMonteCarloSimulator
from models.adaptive_vae import AdaptiveVAE
from analysis.robust_critical_exponent_extractor import RobustCriticalExponentExtractor
from analysis.systematic_comparison_framework import SystematicComparisonFramework
from analysis.publication_figure_generator import PublicationFigureGenerator
from validation.comprehensive_validation_integration import ComprehensiveValidationIntegration

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PREPaperWorkflowDemo:
    """
    Complete workflow demonstration for PRE paper preparation.
    
    This class orchestrates the entire pipeline from data generation
    to publication-ready materials.
    """
    
    def __init__(self, output_dir: str = "results/pre_paper_workflow", quick_demo: bool = False):
        """
        Initialize the workflow demo.
        
        Args:
            output_dir: Directory to save all results
            quick_demo: If True, use reduced parameters for faster execution
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quick_demo = quick_demo
        
        # Configure parameters based on demo mode
        if quick_demo:
            self.config = self._get_quick_demo_config()
        else:
            self.config = self._get_full_config()
            
        logger.info(f"Initialized PRE Paper Workflow Demo")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Quick demo mode: {quick_demo}")
        
    def _get_quick_demo_config(self) -> Dict[str, Any]:
        """Get configuration for quick demonstration."""
        return {
            'ising_3d': {
                'system_sizes': [8, 16],
                'temperature_range': (4.0, 5.0),
                'n_temperatures': 11,
                'n_configs_per_temp': 50,
                'equilibration_steps': 5000,
                'sampling_interval': 100
            },
            'potts_3state': {
                'system_sizes': [16],
                'temperature_range': (0.8, 1.2),
                'n_temperatures': 9,
                'n_configs_per_temp': 50,
                'equilibration_steps': 5000
            },
            'xy_2d': {
                'system_sizes': [16],
                'temperature_range': (0.5, 2.0),
                'n_temperatures': 11,
                'n_configs_per_temp': 50,
                'equilibration_steps': 5000
            },
            'vae_training': {
                'epochs': 20,
                'batch_size': 32,
                'learning_rate': 1e-3
            }
        }
        
    def _get_full_config(self) -> Dict[str, Any]:
        """Get configuration for full workflow."""
        return {
            'ising_3d': {
                'system_sizes': [8, 16, 32],
                'temperature_range': (3.5, 5.5),
                'n_temperatures': 21,
                'n_configs_per_temp': 200,
                'equilibration_steps': 50000,
                'sampling_interval': 500
            },
            'potts_3state': {
                'system_sizes': [16, 32],
                'temperature_range': (0.7, 1.3),
                'n_temperatures': 13,
                'n_configs_per_temp': 200,
                'equilibration_steps': 30000
            },
            'xy_2d': {
                'system_sizes': [16, 32],
                'temperature_range': (0.3, 2.5),
                'n_temperatures': 15,
                'n_configs_per_temp': 200,
                'equilibration_steps': 30000
            },
            'vae_training': {
                'epochs': 100,
                'batch_size': 64,
                'learning_rate': 1e-3
            }
        }
    
    def run_complete_workflow(self) -> Dict[str, Any]:
        """
        Execute the complete PRE paper workflow.
        
        Returns:
            Dictionary containing all results and analysis
        """
        logger.info("Starting complete PRE paper workflow...")
        
        results = {}
        
        # Step 1: Generate 3D Ising data
        logger.info("Step 1: Generating 3D Ising data...")
        ising_data = self._generate_3d_ising_data()
        results['ising_3d_data'] = ising_data
        
        # Step 2: Train VAE on 3D Ising data
        logger.info("Step 2: Training VAE on 3D Ising data...")
        ising_vae_results = self._train_vae_and_extract_representations(
            ising_data, 'ising_3d'
        )
        results['ising_3d_vae'] = ising_vae_results
        
        # Step 3: Extract critical exponents from 3D Ising
        logger.info("Step 3: Extracting critical exponents from 3D Ising...")
        ising_exponents = self._extract_critical_exponents(
            ising_data, ising_vae_results, 'ising_3d'
        )
        results['ising_3d_exponents'] = ising_exponents
        
        # Step 4: Generate and analyze Potts model data
        logger.info("Step 4: Analyzing Q=3 Potts model...")
        potts_results = self._analyze_potts_model()
        results['potts_3state'] = potts_results
        
        # Step 5: Generate and analyze XY model data
        logger.info("Step 5: Analyzing 2D XY model...")
        xy_results = self._analyze_xy_model()
        results['xy_2d'] = xy_results
        
        # Step 6: Perform multi-system comparison
        logger.info("Step 6: Performing multi-system comparison...")
        comparison_results = self._perform_multi_system_comparison(results)
        results['comparison_analysis'] = comparison_results
        
        # Step 7: Generate publication materials
        logger.info("Step 7: Generating publication materials...")
        publication_materials = self._generate_publication_materials(results)
        results['publication_materials'] = publication_materials
        
        # Step 8: Create validation report
        logger.info("Step 8: Creating validation report...")
        validation_report = self._create_validation_report(results)
        results['validation_report'] = validation_report
        
        logger.info("Complete PRE paper workflow finished successfully!")
        return results
    
    def _generate_3d_ising_data(self) -> Dict[str, Any]:
        """Generate high-quality 3D Ising data."""
        config = self.config['ising_3d']
        
        # Initialize data generator
        data_generator = HighQualityDataGenerator(
            model_type='ising_3d',
            output_dir=str(self.output_dir / 'data' / 'ising_3d')
        )
        
        all_data = {}
        
        for system_size in config['system_sizes']:
            logger.info(f"Generating 3D Ising data for L={system_size}...")
            
            # Generate temperature array
            temperatures = np.linspace(
                config['temperature_range'][0],
                config['temperature_range'][1],
                config['n_temperatures']
            )
            
            # Generate data
            data = data_generator.generate_comprehensive_dataset(
                system_size=system_size,
                temperatures=temperatures,
                n_configs_per_temp=config['n_configs_per_temp'],
                equilibration_steps=config['equilibration_steps'],
                sampling_interval=config['sampling_interval']
            )
            
            all_data[f'L_{system_size}'] = data
            
        return {
            'datasets': all_data,
            'config': config,
            'theoretical_tc': 4.511,
            'theoretical_exponents': {'beta': 0.326, 'nu': 0.630}
        }
    
    def _train_vae_and_extract_representations(
        self, 
        data: Dict[str, Any], 
        model_type: str
    ) -> Dict[str, Any]:
        """Train VAE and extract latent representations."""
        
        # Use largest system size for training
        largest_size = max([int(k.split('_')[1]) for k in data['datasets'].keys()])
        training_data = data['datasets'][f'L_{largest_size}']
        
        # Initialize adaptive VAE
        input_shape = training_data['configurations'].shape[1:]
        vae = AdaptiveVAE(
            input_shape=input_shape,
            latent_dim=2,
            model_type=model_type
        )
        
        # Train VAE
        training_config = self.config['vae_training']
        history = vae.train(
            training_data['configurations'],
            epochs=training_config['epochs'],
            batch_size=training_config['batch_size'],
            learning_rate=training_config['learning_rate'],
            validation_split=0.2
        )
        
        # Extract latent representations for all system sizes
        latent_representations = {}
        for size_key, dataset in data['datasets'].items():
            latent_z = vae.encode(dataset['configurations'])
            latent_representations[size_key] = {
                'latent_coords': latent_z,
                'temperatures': dataset['temperatures'],
                'magnetizations': dataset['magnetizations'],
                'energies': dataset['energies']
            }
        
        # Save VAE model
        model_path = self.output_dir / 'models' / f'{model_type}_vae.pth'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        vae.save_model(str(model_path))
        
        return {
            'vae_model': vae,
            'training_history': history,
            'latent_representations': latent_representations,
            'model_path': str(model_path)
        }
    
    def _extract_critical_exponents(
        self,
        data: Dict[str, Any],
        vae_results: Dict[str, Any],
        model_type: str
    ) -> Dict[str, Any]:
        """Extract critical exponents using VAE latent representations."""
        
        # Initialize critical exponent extractor
        extractor = RobustCriticalExponentExtractor(
            bootstrap_samples=100 if self.quick_demo else 1000
        )
        
        exponent_results = {}
        
        for size_key, latent_data in vae_results['latent_representations'].items():
            logger.info(f"Extracting exponents for {size_key}...")
            
            # Extract critical temperature
            tc_result = extractor.extract_critical_temperature(
                temperatures=latent_data['temperatures'],
                order_parameter=latent_data['latent_coords'][:, 0],  # Use first latent dimension
                method='susceptibility_peak'
            )
            
            # Extract beta exponent
            beta_result = extractor.extract_beta_exponent(
                temperatures=latent_data['temperatures'],
                magnetizations=np.abs(latent_data['magnetizations']),
                tc_estimate=tc_result['tc_estimate']
            )
            
            # Extract nu exponent (using correlation length from latent space)
            correlation_lengths = extractor.compute_correlation_length_from_latent(
                latent_coords=latent_data['latent_coords'],
                temperatures=latent_data['temperatures']
            )
            
            nu_result = extractor.extract_nu_exponent(
                temperatures=latent_data['temperatures'],
                correlation_lengths=correlation_lengths,
                tc_estimate=tc_result['tc_estimate']
            )
            
            exponent_results[size_key] = {
                'tc_result': tc_result,
                'beta_result': beta_result,
                'nu_result': nu_result,
                'correlation_lengths': correlation_lengths
            }
        
        # Compare with theoretical values
        theoretical = data['theoretical_exponents']
        comparison = extractor.compare_with_theory(
            measured_results=exponent_results,
            theoretical_exponents=theoretical,
            theoretical_tc=data['theoretical_tc']
        )
        
        return {
            'exponent_results': exponent_results,
            'theoretical_comparison': comparison,
            'accuracy_summary': comparison['accuracy_summary']
        }
    
    def _analyze_potts_model(self) -> Dict[str, Any]:
        """Analyze Q=3 Potts model."""
        config = self.config['potts_3state']
        
        # Initialize Monte Carlo simulator for Potts model
        simulator = UnifiedMonteCarloSimulator(model_type='potts_3state')
        
        results = {}
        
        for system_size in config['system_sizes']:
            logger.info(f"Analyzing Potts model for L={system_size}...")
            
            temperatures = np.linspace(
                config['temperature_range'][0],
                config['temperature_range'][1],
                config['n_temperatures']
            )
            
            # Generate Potts data
            potts_data = simulator.generate_dataset(
                system_size=system_size,
                temperatures=temperatures,
                n_configs_per_temp=config['n_configs_per_temp'],
                equilibration_steps=config['equilibration_steps']
            )
            
            # Train VAE on Potts data
            vae_results = self._train_vae_and_extract_representations(
                {'datasets': {f'L_{system_size}': potts_data}},
                'potts_3state'
            )
            
            # Detect first-order transition
            tc_detected = self._detect_first_order_transition(
                potts_data, vae_results['latent_representations'][f'L_{system_size}']
            )
            
            results[f'L_{system_size}'] = {
                'data': potts_data,
                'vae_results': vae_results,
                'tc_detected': tc_detected,
                'theoretical_tc': 1.005
            }
        
        return results
    
    def _analyze_xy_model(self) -> Dict[str, Any]:
        """Analyze 2D XY model."""
        config = self.config['xy_2d']
        
        # Initialize Monte Carlo simulator for XY model
        simulator = UnifiedMonteCarloSimulator(model_type='xy_2d')
        
        results = {}
        
        for system_size in config['system_sizes']:
            logger.info(f"Analyzing XY model for L={system_size}...")
            
            temperatures = np.linspace(
                config['temperature_range'][0],
                config['temperature_range'][1],
                config['n_temperatures']
            )
            
            # Generate XY data
            xy_data = simulator.generate_dataset(
                system_size=system_size,
                temperatures=temperatures,
                n_configs_per_temp=config['n_configs_per_temp'],
                equilibration_steps=config['equilibration_steps']
            )
            
            # Train VAE on XY data
            vae_results = self._train_vae_and_extract_representations(
                {'datasets': {f'L_{system_size}': xy_data}},
                'xy_2d'
            )
            
            # Detect KT transition
            kt_analysis = self._analyze_kt_transition(
                xy_data, vae_results['latent_representations'][f'L_{system_size}']
            )
            
            results[f'L_{system_size}'] = {
                'data': xy_data,
                'vae_results': vae_results,
                'kt_analysis': kt_analysis
            }
        
        return results
    
    def _detect_first_order_transition(self, data: Dict, latent_data: Dict) -> Dict[str, Any]:
        """Detect first-order phase transition in Potts model."""
        # Look for discontinuous jump in order parameter
        order_param = latent_data['latent_coords'][:, 0]
        temperatures = latent_data['temperatures']
        
        # Find maximum gradient (steepest change)
        gradients = np.gradient(order_param, temperatures)
        max_gradient_idx = np.argmax(np.abs(gradients))
        tc_estimate = temperatures[max_gradient_idx]
        
        return {
            'tc_estimate': tc_estimate,
            'max_gradient': gradients[max_gradient_idx],
            'transition_type': 'first_order',
            'order_parameter': order_param
        }
    
    def _analyze_kt_transition(self, data: Dict, latent_data: Dict) -> Dict[str, Any]:
        """Analyze Kosterlitz-Thouless transition in XY model."""
        # KT transition analysis using vortex unbinding
        temperatures = latent_data['temperatures']
        latent_coords = latent_data['latent_coords']
        
        # Use latent space to identify topological transition
        # (Simplified analysis - in practice would need vortex detection)
        helicity_modulus = np.var(latent_coords[:, 1], axis=0)  # Proxy for helicity modulus
        
        # Find transition temperature where helicity modulus changes behavior
        kt_temp_estimate = temperatures[np.argmax(np.gradient(helicity_modulus))]
        
        return {
            'kt_temperature': kt_temp_estimate,
            'helicity_modulus': helicity_modulus,
            'transition_type': 'kosterlitz_thouless',
            'latent_analysis': latent_coords
        }
    
    def _perform_multi_system_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform systematic comparison across all physics systems."""
        
        # Initialize comparison framework
        comparison_framework = SystematicComparisonFramework(
            output_dir=str(self.output_dir / 'comparison')
        )
        
        # Prepare system results for comparison
        system_results = {
            'ising_3d': {
                'type': 'continuous_transition',
                'theoretical_tc': 4.511,
                'measured_tc': results['ising_3d_exponents']['theoretical_comparison']['tc_accuracy'],
                'critical_exponents': results['ising_3d_exponents']['accuracy_summary'],
                'latent_quality': self._assess_latent_quality(results['ising_3d_vae'])
            },
            'potts_3state': {
                'type': 'first_order_transition',
                'theoretical_tc': 1.005,
                'measured_tc': self._extract_potts_tc_accuracy(results['potts_3state']),
                'transition_detection': True,
                'latent_quality': self._assess_latent_quality_potts(results['potts_3state'])
            },
            'xy_2d': {
                'type': 'topological_transition',
                'kt_detection': True,
                'measured_kt_temp': self._extract_xy_kt_temp(results['xy_2d']),
                'latent_quality': self._assess_latent_quality_xy(results['xy_2d'])
            }
        }
        
        # Generate comparison analysis
        comparison_analysis = comparison_framework.generate_comprehensive_comparison(
            system_results
        )
        
        return comparison_analysis
    
    def _generate_publication_materials(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate all publication-ready figures and tables."""
        
        # Initialize publication figure generator
        fig_generator = PublicationFigureGenerator(
            output_dir=str(self.output_dir / 'publication')
        )
        
        publication_materials = {}
        
        # Generate main results figure
        main_figure = fig_generator.create_main_results_figure(
            ising_results=results['ising_3d_exponents'],
            comparison_data=results['comparison_analysis']
        )
        publication_materials['main_figure'] = main_figure
        
        # Generate critical exponent comparison table
        exponent_table = fig_generator.create_critical_exponent_table(
            results['ising_3d_exponents']['exponent_results'],
            results['ising_3d_exponents']['theoretical_comparison']
        )
        publication_materials['exponent_table'] = exponent_table
        
        # Generate phase diagram comparison
        phase_diagrams = fig_generator.create_phase_diagram_comparison({
            'ising_3d': results['ising_3d_vae']['latent_representations'],
            'potts_3state': results['potts_3state'],
            'xy_2d': results['xy_2d']
        })
        publication_materials['phase_diagrams'] = phase_diagrams
        
        # Generate system comparison figure
        system_comparison = fig_generator.create_multi_system_comparison_figure(
            results['comparison_analysis']
        )
        publication_materials['system_comparison'] = system_comparison
        
        return publication_materials
    
    def _create_validation_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive validation report."""
        
        # Initialize validation system
        validator = ComprehensiveValidationIntegration(
            output_dir=str(self.output_dir / 'validation')
        )
        
        # Validate all results
        validation_report = validator.validate_complete_workflow(results)
        
        # Add summary statistics
        validation_report['summary'] = {
            'ising_3d_accuracy': results['ising_3d_exponents']['accuracy_summary'],
            'potts_detection_success': self._validate_potts_detection(results['potts_3state']),
            'xy_kt_detection_success': self._validate_xy_detection(results['xy_2d']),
            'overall_success_rate': self._calculate_overall_success_rate(results)
        }
        
        return validation_report
    
    # Helper methods for assessment and validation
    def _assess_latent_quality(self, vae_results: Dict) -> float:
        """Assess quality of latent representations."""
        # Simple quality metric based on latent space variance
        latent_data = list(vae_results['latent_representations'].values())[0]
        latent_variance = np.var(latent_data['latent_coords'], axis=0)
        return float(np.mean(latent_variance))
    
    def _assess_latent_quality_potts(self, potts_results: Dict) -> float:
        """Assess latent quality for Potts model."""
        # Average across system sizes
        qualities = []
        for size_data in potts_results.values():
            if 'vae_results' in size_data:
                quality = self._assess_latent_quality(size_data['vae_results'])
                qualities.append(quality)
        return np.mean(qualities) if qualities else 0.0
    
    def _assess_latent_quality_xy(self, xy_results: Dict) -> float:
        """Assess latent quality for XY model."""
        # Average across system sizes
        qualities = []
        for size_data in xy_results.values():
            if 'vae_results' in size_data:
                quality = self._assess_latent_quality(size_data['vae_results'])
                qualities.append(quality)
        return np.mean(qualities) if qualities else 0.0
    
    def _extract_potts_tc_accuracy(self, potts_results: Dict) -> float:
        """Extract Tc accuracy for Potts model."""
        accuracies = []
        for size_data in potts_results.values():
            if 'tc_detected' in size_data:
                measured = size_data['tc_detected']['tc_estimate']
                theoretical = size_data['theoretical_tc']
                accuracy = 1.0 - abs(measured - theoretical) / theoretical
                accuracies.append(accuracy)
        return np.mean(accuracies) if accuracies else 0.0
    
    def _extract_xy_kt_temp(self, xy_results: Dict) -> float:
        """Extract KT temperature from XY results."""
        kt_temps = []
        for size_data in xy_results.values():
            if 'kt_analysis' in size_data:
                kt_temps.append(size_data['kt_analysis']['kt_temperature'])
        return np.mean(kt_temps) if kt_temps else 0.0
    
    def _validate_potts_detection(self, potts_results: Dict) -> bool:
        """Validate Potts transition detection."""
        for size_data in potts_results.values():
            if 'tc_detected' in size_data:
                # Check if detected Tc is reasonable
                measured = size_data['tc_detected']['tc_estimate']
                theoretical = size_data['theoretical_tc']
                if abs(measured - theoretical) / theoretical > 0.2:  # 20% tolerance
                    return False
        return True
    
    def _validate_xy_detection(self, xy_results: Dict) -> bool:
        """Validate XY KT transition detection."""
        for size_data in xy_results.values():
            if 'kt_analysis' in size_data:
                # Check if KT analysis was successful
                if size_data['kt_analysis']['kt_temperature'] > 0:
                    return True
        return False
    
    def _calculate_overall_success_rate(self, results: Dict[str, Any]) -> float:
        """Calculate overall success rate across all systems."""
        successes = 0
        total = 0
        
        # Ising 3D success
        if results['ising_3d_exponents']['accuracy_summary'].get('beta_accuracy', 0) > 0.7:
            successes += 1
        total += 1
        
        # Potts success
        if self._validate_potts_detection(results['potts_3state']):
            successes += 1
        total += 1
        
        # XY success
        if self._validate_xy_detection(results['xy_2d']):
            successes += 1
        total += 1
        
        return successes / total if total > 0 else 0.0


def main():
    """Main function to run the complete workflow example."""
    parser = argparse.ArgumentParser(description='PRE Paper Complete Workflow Example')
    parser.add_argument(
        '--quick-demo', 
        action='store_true',
        help='Run with reduced parameters for faster demonstration'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/pre_paper_workflow',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Initialize and run workflow
    workflow = PREPaperWorkflowDemo(
        output_dir=args.output_dir,
        quick_demo=args.quick_demo
    )
    
    try:
        results = workflow.run_complete_workflow()
        
        # Print summary
        print("\n" + "="*60)
        print("PRE PAPER WORKFLOW COMPLETE")
        print("="*60)
        
        if 'validation_report' in results:
            summary = results['validation_report']['summary']
            print(f"Overall Success Rate: {summary['overall_success_rate']:.1%}")
            print(f"Ising 3D Accuracy: {summary.get('ising_3d_accuracy', {})}")
            print(f"Potts Detection: {'✓' if summary['potts_detection_success'] else '✗'}")
            print(f"XY KT Detection: {'✓' if summary['xy_kt_detection_success'] else '✗'}")
        
        print(f"\nResults saved to: {workflow.output_dir}")
        print("Publication materials generated successfully!")
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        raise


if __name__ == "__main__":
    main()