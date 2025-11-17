#!/usr/bin/env python3
"""
Multi-System Comparison Analysis Example

This script demonstrates comparative analysis across multiple statistical mechanics
systems (Ising, Potts, XY) using the Prometheus phase discovery framework.

Requirements: 5.5 - Create example demonstrating multi-system comparison analysis

Usage:
    python examples/multi_system_comparison_example.py [--quick-demo]
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
import seaborn as sns

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from analysis.systematic_comparison_framework import SystematicComparisonFramework
from analysis.publication_figure_generator import PublicationFigureGenerator
from data.unified_monte_carlo import UnifiedMonteCarloSimulator
from models.adaptive_vae import AdaptiveVAE
from analysis.robust_critical_exponent_extractor import RobustCriticalExponentExtractor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiSystemComparisonDemo:
    """
    Demonstration of multi-system comparison analysis.
    
    This class shows how to:
    1. Analyze multiple physics systems with Prometheus
    2. Compare phase detection capabilities across systems
    3. Validate universality class identification
    4. Generate comparative publication materials
    """
    
    def __init__(self, output_dir: str = 'results/multi_system_comparison', quick_demo: bool = False):
        """
        Initialize multi-system comparison demo.
        
        Args:
            output_dir: Directory to save results
            quick_demo: If True, use reduced parameters for faster execution
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quick_demo = quick_demo
        
        # Define systems to compare
        self.systems = {
            'ising_2d': {
                'name': '2D Ising',
                'universality_class': '2D Ising',
                'transition_type': 'continuous',
                'theoretical_tc': 2.269,
                'theoretical_exponents': {'beta': 0.125, 'nu': 1.0, 'gamma': 1.75},
                'dimensions': 2
            },
            'ising_3d': {
                'name': '3D Ising',
                'universality_class': '3D Ising',
                'transition_type': 'continuous',
                'theoretical_tc': 4.511,
                'theoretical_exponents': {'beta': 0.326, 'nu': 0.630, 'gamma': 1.237},
                'dimensions': 3
            },
            'potts_3state': {
                'name': 'Q=3 Potts',
                'universality_class': 'First-order',
                'transition_type': 'first_order',
                'theoretical_tc': 1.005,
                'theoretical_exponents': None,  # First-order transition
                'dimensions': 2
            },
            'xy_2d': {
                'name': '2D XY',
                'universality_class': 'Kosterlitz-Thouless',
                'transition_type': 'topological',
                'theoretical_tc': None,  # KT transition
                'theoretical_exponents': None,
                'dimensions': 2
            }
        }
        
        # Configure parameters based on demo mode
        if quick_demo:
            self.config = self._get_quick_demo_config()
        else:
            self.config = self._get_full_config()
            
        logger.info(f"Initialized multi-system comparison demo")
        logger.info(f"Systems to analyze: {list(self.systems.keys())}")
        logger.info(f"Quick demo mode: {quick_demo}")
    
    def _get_quick_demo_config(self) -> Dict[str, Any]:
        """Get configuration for quick demonstration."""
        return {
            'system_size': 16,
            'n_temperatures': 11,
            'n_configs_per_temp': 50,
            'equilibration_steps': 5000,
            'vae_epochs': 20,
            'bootstrap_samples': 100
        }
    
    def _get_full_config(self) -> Dict[str, Any]:
        """Get configuration for full analysis."""
        return {
            'system_size': 32,
            'n_temperatures': 21,
            'n_configs_per_temp': 200,
            'equilibration_steps': 30000,
            'vae_epochs': 100,
            'bootstrap_samples': 1000
        }
    
    def run_complete_comparison(self) -> Dict[str, Any]:
        """
        Run complete multi-system comparison analysis.
        
        Returns:
            Dictionary containing all comparison results
        """
        logger.info("Starting multi-system comparison analysis...")
        
        results = {}
        
        # Step 1: Analyze each system individually
        logger.info("Step 1: Analyzing individual systems...")
        system_results = {}
        
        for system_name, system_info in self.systems.items():
            logger.info(f"Analyzing {system_info['name']}...")
            system_result = self._analyze_single_system(system_name, system_info)
            system_results[system_name] = system_result
        
        results['individual_systems'] = system_results
        
        # Step 2: Perform systematic comparison
        logger.info("Step 2: Performing systematic comparison...")
        comparison_analysis = self._perform_systematic_comparison(system_results)
        results['comparison_analysis'] = comparison_analysis
        
        # Step 3: Validate universality class identification
        logger.info("Step 3: Validating universality class identification...")
        universality_validation = self._validate_universality_classes(system_results)
        results['universality_validation'] = universality_validation
        
        # Step 4: Compare phase detection capabilities
        logger.info("Step 4: Comparing phase detection capabilities...")
        phase_detection_comparison = self._compare_phase_detection(system_results)
        results['phase_detection_comparison'] = phase_detection_comparison
        
        # Step 5: Generate comparative visualizations
        logger.info("Step 5: Generating comparative visualizations...")
        visualizations = self._generate_comparative_visualizations(results)
        results['visualizations'] = visualizations
        
        # Step 6: Create publication materials
        logger.info("Step 6: Creating publication materials...")
        publication_materials = self._create_publication_materials(results)
        results['publication_materials'] = publication_materials
        
        # Step 7: Generate summary report
        logger.info("Step 7: Generating summary report...")
        summary_report = self._generate_summary_report(results)
        results['summary_report'] = summary_report
        
        logger.info("Multi-system comparison analysis completed!")
        return results
    
    def _analyze_single_system(self, system_name: str, system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single physics system."""
        
        # Generate data for the system
        data = self._generate_system_data(system_name, system_info)
        
        # Train VAE on the data
        vae_results = self._train_system_vae(system_name, data)
        
        # Perform phase analysis
        phase_analysis = self._perform_phase_analysis(system_name, system_info, data, vae_results)
        
        # Extract critical properties (if applicable)
        critical_properties = self._extract_critical_properties(system_name, system_info, data, vae_results)
        
        return {
            'system_info': system_info,
            'data': data,
            'vae_results': vae_results,
            'phase_analysis': phase_analysis,
            'critical_properties': critical_properties
        }
    
    def _generate_system_data(self, system_name: str, system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Monte Carlo data for a specific system."""
        
        # Initialize simulator
        simulator = UnifiedMonteCarloSimulator(model_type=system_name)
        
        # Define temperature range based on system
        if system_info['theoretical_tc'] is not None:
            tc = system_info['theoretical_tc']
            temp_range = (tc - 1.0, tc + 1.0)
        else:
            # For systems without well-defined Tc
            temp_range = (0.5, 2.5)
        
        temperatures = np.linspace(temp_range[0], temp_range[1], self.config['n_temperatures'])
        
        # Generate dataset
        dataset = simulator.generate_dataset(
            system_size=self.config['system_size'],
            temperatures=temperatures,
            n_configs_per_temp=self.config['n_configs_per_temp'],
            equilibration_steps=self.config['equilibration_steps']
        )
        
        return {
            'dataset': dataset,
            'temperatures': temperatures,
            'system_size': self.config['system_size']
        }
    
    def _train_system_vae(self, system_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Train VAE for a specific system."""
        
        dataset = data['dataset']
        
        # Initialize adaptive VAE
        input_shape = dataset['configurations'].shape[1:]
        vae = AdaptiveVAE(
            input_shape=input_shape,
            latent_dim=2,
            model_type=system_name,
            physics_informed=True
        )
        
        # Train VAE
        history = vae.train(
            dataset['configurations'],
            epochs=self.config['vae_epochs'],
            batch_size=64,
            learning_rate=1e-3,
            validation_split=0.2,
            physics_targets={
                'temperatures': dataset['temperatures'],
                'magnetizations': dataset.get('magnetizations', np.zeros_like(dataset['temperatures']))
            }
        )
        
        # Extract latent representations
        latent_coords = vae.encode(dataset['configurations'])
        
        # Analyze latent space quality
        latent_quality = self._assess_latent_space_quality(
            latent_coords, dataset['temperatures'], dataset.get('magnetizations')
        )
        
        return {
            'vae_model': vae,
            'training_history': history,
            'latent_coords': latent_coords,
            'latent_quality': latent_quality
        }
    
    def _perform_phase_analysis(
        self, 
        system_name: str, 
        system_info: Dict[str, Any], 
        data: Dict[str, Any], 
        vae_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform phase analysis for a system."""
        
        temperatures = data['temperatures']
        latent_coords = vae_results['latent_coords']
        
        phase_analysis = {
            'transition_type': system_info['transition_type'],
            'phase_detection_success': False,
            'detected_tc': None,
            'phase_boundaries': [],
            'order_parameter': None
        }
        
        # Identify order parameter from latent space
        if data['dataset'].get('magnetizations') is not None:
            # Find latent dimension most correlated with magnetization
            magnetizations = data['dataset']['magnetizations']
            correlations = []
            for dim in range(latent_coords.shape[1]):
                corr = np.corrcoef(latent_coords[:, dim], np.abs(magnetizations))[0, 1]
                correlations.append(abs(corr))
            
            best_dim = np.argmax(correlations)
            order_parameter = latent_coords[:, best_dim]
            phase_analysis['order_parameter'] = order_parameter
            phase_analysis['order_parameter_correlation'] = correlations[best_dim]
        else:
            # Use first latent dimension as order parameter
            order_parameter = latent_coords[:, 0]
            phase_analysis['order_parameter'] = order_parameter
            phase_analysis['order_parameter_correlation'] = None
        
        # Detect phase transition based on transition type
        if system_info['transition_type'] == 'continuous':
            # Look for continuous phase transition
            tc_detected = self._detect_continuous_transition(temperatures, order_parameter)
            phase_analysis['detected_tc'] = tc_detected
            phase_analysis['phase_detection_success'] = tc_detected is not None
            
        elif system_info['transition_type'] == 'first_order':
            # Look for first-order transition (discontinuous jump)
            tc_detected = self._detect_first_order_transition(temperatures, order_parameter)
            phase_analysis['detected_tc'] = tc_detected
            phase_analysis['phase_detection_success'] = tc_detected is not None
            
        elif system_info['transition_type'] == 'topological':
            # Look for topological transition (KT type)
            kt_analysis = self._detect_topological_transition(temperatures, latent_coords)
            phase_analysis.update(kt_analysis)
            phase_analysis['phase_detection_success'] = kt_analysis.get('kt_detected', False)
        
        return phase_analysis
    
    def _extract_critical_properties(
        self,
        system_name: str,
        system_info: Dict[str, Any],
        data: Dict[str, Any],
        vae_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract critical properties if applicable."""
        
        critical_properties = {
            'has_critical_exponents': system_info['theoretical_exponents'] is not None,
            'extracted_exponents': {},
            'tc_accuracy': None,
            'exponent_accuracies': {}
        }
        
        if system_info['transition_type'] == 'continuous' and system_info['theoretical_exponents'] is not None:
            # Extract critical exponents for continuous transitions
            extractor = RobustCriticalExponentExtractor(
                bootstrap_samples=self.config['bootstrap_samples']
            )
            
            temperatures = data['temperatures']
            order_parameter = vae_results['latent_coords'][:, 0]  # Use first latent dimension
            
            # Extract critical temperature
            tc_result = extractor.extract_critical_temperature(
                temperatures=temperatures,
                order_parameter=order_parameter,
                method='susceptibility_peak'
            )
            
            if tc_result['tc_estimate'] is not None:
                tc_estimate = tc_result['tc_estimate']
                
                # Calculate Tc accuracy
                if system_info['theoretical_tc'] is not None:
                    tc_error = abs(tc_estimate - system_info['theoretical_tc']) / system_info['theoretical_tc']
                    critical_properties['tc_accuracy'] = 1.0 - tc_error
                
                # Extract beta exponent
                try:
                    beta_result = extractor.extract_beta_exponent(
                        temperatures=temperatures,
                        magnetizations=np.abs(order_parameter),
                        tc_estimate=tc_estimate
                    )
                    critical_properties['extracted_exponents']['beta'] = beta_result['beta_estimate']
                    
                    # Calculate beta accuracy
                    if 'beta' in system_info['theoretical_exponents']:
                        beta_theoretical = system_info['theoretical_exponents']['beta']
                        beta_error = abs(beta_result['beta_estimate'] - beta_theoretical) / beta_theoretical
                        critical_properties['exponent_accuracies']['beta'] = 1.0 - beta_error
                        
                except Exception as e:
                    logger.warning(f"Failed to extract beta exponent for {system_name}: {e}")
                
                # Extract nu exponent (simplified - using correlation length proxy)
                try:
                    correlation_lengths = np.var(vae_results['latent_coords'], axis=1)  # Simplified proxy
                    nu_result = extractor.extract_nu_exponent(
                        temperatures=temperatures,
                        correlation_lengths=correlation_lengths,
                        tc_estimate=tc_estimate
                    )
                    critical_properties['extracted_exponents']['nu'] = nu_result['nu_estimate']
                    
                    # Calculate nu accuracy
                    if 'nu' in system_info['theoretical_exponents']:
                        nu_theoretical = system_info['theoretical_exponents']['nu']
                        nu_error = abs(nu_result['nu_estimate'] - nu_theoretical) / nu_theoretical
                        critical_properties['exponent_accuracies']['nu'] = 1.0 - nu_error
                        
                except Exception as e:
                    logger.warning(f"Failed to extract nu exponent for {system_name}: {e}")
        
        return critical_properties
    
    def _perform_systematic_comparison(self, system_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform systematic comparison across all systems."""
        
        comparison_framework = SystematicComparisonFramework(
            output_dir=str(self.output_dir / 'comparison')
        )
        
        # Prepare comparison data
        comparison_data = {}
        
        for system_name, result in system_results.items():
            system_info = result['system_info']
            phase_analysis = result['phase_analysis']
            critical_properties = result['critical_properties']
            
            comparison_data[system_name] = {
                'name': system_info['name'],
                'universality_class': system_info['universality_class'],
                'transition_type': system_info['transition_type'],
                'phase_detection_success': phase_analysis['phase_detection_success'],
                'tc_accuracy': critical_properties.get('tc_accuracy'),
                'exponent_accuracies': critical_properties.get('exponent_accuracies', {}),
                'latent_quality': result['vae_results']['latent_quality']
            }
        
        # Generate comprehensive comparison
        comparison_analysis = comparison_framework.generate_comprehensive_comparison(comparison_data)
        
        return comparison_analysis
    
    def _validate_universality_classes(self, system_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate universality class identification."""
        
        validation = {
            'correct_identifications': 0,
            'total_systems': len(system_results),
            'system_validations': {}
        }
        
        for system_name, result in system_results.items():
            system_info = result['system_info']
            critical_properties = result['critical_properties']
            
            # Check if universality class can be identified from extracted exponents
            identified_correctly = False
            
            if system_info['universality_class'] in ['2D Ising', '3D Ising']:
                # Check if extracted exponents match theoretical values within tolerance
                exponent_accuracies = critical_properties.get('exponent_accuracies', {})
                if exponent_accuracies:
                    avg_accuracy = np.mean(list(exponent_accuracies.values()))
                    identified_correctly = avg_accuracy > 0.8  # 80% accuracy threshold
            
            elif system_info['universality_class'] == 'First-order':
                # Check if first-order transition was detected
                phase_analysis = result['phase_analysis']
                identified_correctly = phase_analysis['phase_detection_success']
            
            elif system_info['universality_class'] == 'Kosterlitz-Thouless':
                # Check if topological transition was detected
                phase_analysis = result['phase_analysis']
                identified_correctly = phase_analysis['phase_detection_success']
            
            validation['system_validations'][system_name] = {
                'universality_class': system_info['universality_class'],
                'identified_correctly': identified_correctly,
                'confidence': self._calculate_identification_confidence(result)
            }
            
            if identified_correctly:
                validation['correct_identifications'] += 1
        
        validation['success_rate'] = validation['correct_identifications'] / validation['total_systems']
        
        return validation
    
    def _compare_phase_detection(self, system_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare phase detection capabilities across systems."""
        
        comparison = {
            'detection_success_rates': {},
            'detection_methods': {},
            'latent_space_quality': {},
            'overall_performance': {}
        }
        
        for system_name, result in system_results.items():
            system_info = result['system_info']
            phase_analysis = result['phase_analysis']
            vae_results = result['vae_results']
            
            # Detection success
            comparison['detection_success_rates'][system_name] = {
                'success': phase_analysis['phase_detection_success'],
                'transition_type': system_info['transition_type']
            }
            
            # Detection method effectiveness
            if system_info['transition_type'] == 'continuous':
                method_effectiveness = self._evaluate_continuous_detection_method(result)
            elif system_info['transition_type'] == 'first_order':
                method_effectiveness = self._evaluate_first_order_detection_method(result)
            else:
                method_effectiveness = self._evaluate_topological_detection_method(result)
            
            comparison['detection_methods'][system_name] = method_effectiveness
            
            # Latent space quality
            comparison['latent_space_quality'][system_name] = vae_results['latent_quality']
        
        # Overall performance metrics
        success_rates = [r['success'] for r in comparison['detection_success_rates'].values()]
        comparison['overall_performance'] = {
            'total_success_rate': np.mean(success_rates),
            'continuous_transitions': np.mean([
                r['success'] for r in comparison['detection_success_rates'].values()
                if r['transition_type'] == 'continuous'
            ]),
            'first_order_transitions': np.mean([
                r['success'] for r in comparison['detection_success_rates'].values()
                if r['transition_type'] == 'first_order'
            ]),
            'topological_transitions': np.mean([
                r['success'] for r in comparison['detection_success_rates'].values()
                if r['transition_type'] == 'topological'
            ])
        }
        
        return comparison
    
    def _generate_comparative_visualizations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative visualizations across systems."""
        
        visualizations = {}
        
        # 1. Phase diagram comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        system_names = list(results['individual_systems'].keys())
        for i, system_name in enumerate(system_names):
            if i >= 4:  # Only plot first 4 systems
                break
                
            result = results['individual_systems'][system_name]
            latent_coords = result['vae_results']['latent_coords']
            temperatures = result['data']['temperatures']
            
            # Color points by temperature
            scatter = axes[i].scatter(
                latent_coords[:, 0], latent_coords[:, 1],
                c=temperatures, cmap='coolwarm', alpha=0.6
            )
            axes[i].set_title(f"{result['system_info']['name']}")
            axes[i].set_xlabel('Latent Dimension 1')
            axes[i].set_ylabel('Latent Dimension 2')
            plt.colorbar(scatter, ax=axes[i], label='Temperature')
        
        plt.tight_layout()
        plot_path = self.output_dir / 'phase_diagrams_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        visualizations['phase_diagrams'] = str(plot_path)
        plt.close()
        
        # 2. Detection success comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        system_names = []
        success_rates = []
        transition_types = []
        
        for system_name, result in results['individual_systems'].items():
            system_names.append(result['system_info']['name'])
            success_rates.append(1.0 if result['phase_analysis']['phase_detection_success'] else 0.0)
            transition_types.append(result['system_info']['transition_type'])
        
        # Create bar plot with different colors for transition types
        colors = {'continuous': 'blue', 'first_order': 'red', 'topological': 'green'}
        bar_colors = [colors[t] for t in transition_types]
        
        bars = ax.bar(system_names, success_rates, color=bar_colors, alpha=0.7)
        ax.set_ylabel('Detection Success Rate')
        ax.set_title('Phase Detection Success Across Systems')
        ax.set_ylim(0, 1.1)
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, color=colors[t], alpha=0.7, label=t.replace('_', ' ').title()) 
                          for t in set(transition_types)]
        ax.legend(handles=legend_elements)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = self.output_dir / 'detection_success_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        visualizations['detection_success'] = str(plot_path)
        plt.close()
        
        # 3. Critical exponent accuracy comparison (for applicable systems)
        systems_with_exponents = []
        beta_accuracies = []
        nu_accuracies = []
        
        for system_name, result in results['individual_systems'].items():
            if result['critical_properties']['has_critical_exponents']:
                exponent_accuracies = result['critical_properties']['exponent_accuracies']
                if exponent_accuracies:
                    systems_with_exponents.append(result['system_info']['name'])
                    beta_accuracies.append(exponent_accuracies.get('beta', 0))
                    nu_accuracies.append(exponent_accuracies.get('nu', 0))
        
        if systems_with_exponents:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(systems_with_exponents))
            width = 0.35
            
            ax.bar(x - width/2, beta_accuracies, width, label='β exponent', alpha=0.7)
            ax.bar(x + width/2, nu_accuracies, width, label='ν exponent', alpha=0.7)
            
            ax.set_ylabel('Accuracy')
            ax.set_title('Critical Exponent Extraction Accuracy')
            ax.set_xticks(x)
            ax.set_xticklabels(systems_with_exponents)
            ax.legend()
            ax.set_ylim(0, 1.1)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plot_path = self.output_dir / 'exponent_accuracy_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            visualizations['exponent_accuracy'] = str(plot_path)
            plt.close()
        
        return visualizations
    
    def _create_publication_materials(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create publication-ready materials."""
        
        # Initialize publication figure generator
        fig_generator = PublicationFigureGenerator(
            output_dir=str(self.output_dir / 'publication')
        )
        
        publication_materials = {}
        
        # Create main comparison figure
        main_figure = fig_generator.create_multi_system_comparison_figure(
            results['comparison_analysis']
        )
        publication_materials['main_figure'] = main_figure
        
        # Create comparison table
        comparison_table = self._create_comparison_table(results)
        publication_materials['comparison_table'] = comparison_table
        
        # Create universality class validation table
        universality_table = self._create_universality_validation_table(results)
        publication_materials['universality_table'] = universality_table
        
        return publication_materials
    
    def _generate_summary_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        
        summary = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'systems_analyzed': len(results['individual_systems']),
            'overall_performance': {},
            'system_summaries': {},
            'key_findings': []
        }
        
        # Overall performance
        if 'phase_detection_comparison' in results:
            comparison = results['phase_detection_comparison']
            summary['overall_performance'] = comparison['overall_performance']
        
        # Individual system summaries
        for system_name, result in results['individual_systems'].items():
            system_info = result['system_info']
            phase_analysis = result['phase_analysis']
            critical_properties = result['critical_properties']
            
            summary['system_summaries'][system_name] = {
                'name': system_info['name'],
                'universality_class': system_info['universality_class'],
                'phase_detection_success': phase_analysis['phase_detection_success'],
                'tc_accuracy': critical_properties.get('tc_accuracy'),
                'mean_exponent_accuracy': np.mean(list(critical_properties.get('exponent_accuracies', {}).values())) if critical_properties.get('exponent_accuracies') else None
            }
        
        # Key findings
        if 'universality_validation' in results:
            validation = results['universality_validation']
            summary['key_findings'].append(
                f"Universality class identification success rate: {validation['success_rate']:.1%}"
            )
        
        if 'phase_detection_comparison' in results:
            comparison = results['phase_detection_comparison']
            summary['key_findings'].append(
                f"Overall phase detection success rate: {comparison['overall_performance']['total_success_rate']:.1%}"
            )
        
        # Save summary to file
        summary_path = self.output_dir / 'multi_system_comparison_summary.json'
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary
    
    # Helper methods for specific analyses
    def _assess_latent_space_quality(
        self, 
        latent_coords: np.ndarray, 
        temperatures: np.ndarray, 
        magnetizations: np.ndarray = None
    ) -> Dict[str, float]:
        """Assess quality of latent space representations."""
        
        quality_metrics = {}
        
        # Variance in latent dimensions
        latent_variances = np.var(latent_coords, axis=0)
        quality_metrics['mean_latent_variance'] = float(np.mean(latent_variances))
        quality_metrics['latent_variance_ratio'] = float(np.max(latent_variances) / np.min(latent_variances))
        
        # Temperature correlation
        temp_correlations = []
        for dim in range(latent_coords.shape[1]):
            corr = np.corrcoef(latent_coords[:, dim], temperatures)[0, 1]
            temp_correlations.append(abs(corr))
        quality_metrics['max_temperature_correlation'] = float(np.max(temp_correlations))
        
        # Magnetization correlation (if available)
        if magnetizations is not None:
            mag_correlations = []
            for dim in range(latent_coords.shape[1]):
                corr = np.corrcoef(latent_coords[:, dim], np.abs(magnetizations))[0, 1]
                mag_correlations.append(abs(corr))
            quality_metrics['max_magnetization_correlation'] = float(np.max(mag_correlations))
        
        return quality_metrics
    
    def _detect_continuous_transition(self, temperatures: np.ndarray, order_parameter: np.ndarray) -> float:
        """Detect continuous phase transition."""
        # Look for susceptibility peak
        susceptibility = self._compute_susceptibility_simple(order_parameter, temperatures)
        if len(susceptibility) > 0:
            peak_idx = np.argmax(susceptibility)
            return temperatures[peak_idx]
        return None
    
    def _detect_first_order_transition(self, temperatures: np.ndarray, order_parameter: np.ndarray) -> float:
        """Detect first-order phase transition."""
        # Look for maximum gradient (steepest change)
        gradients = np.gradient(order_parameter, temperatures)
        max_gradient_idx = np.argmax(np.abs(gradients))
        return temperatures[max_gradient_idx]
    
    def _detect_topological_transition(self, temperatures: np.ndarray, latent_coords: np.ndarray) -> Dict[str, Any]:
        """Detect topological (KT) transition."""
        # Simplified KT detection using latent space analysis
        helicity_proxy = np.var(latent_coords[:, 1], axis=0) if latent_coords.shape[1] > 1 else 0
        
        return {
            'kt_detected': helicity_proxy > 0.01,  # Simple threshold
            'kt_temperature_estimate': temperatures[len(temperatures)//2],  # Rough estimate
            'helicity_proxy': helicity_proxy
        }
    
    def _compute_susceptibility_simple(self, order_parameter: np.ndarray, temperatures: np.ndarray) -> np.ndarray:
        """Compute susceptibility from order parameter fluctuations."""
        unique_temps = np.unique(temperatures)
        susceptibility = np.zeros_like(unique_temps)
        
        for i, temp in enumerate(unique_temps):
            temp_mask = temperatures == temp
            if np.sum(temp_mask) > 1:
                susceptibility[i] = np.var(order_parameter[temp_mask]) / temp
        
        return susceptibility
    
    def _calculate_identification_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence in universality class identification."""
        # Simple confidence metric based on various factors
        confidence = 0.0
        
        # Phase detection success
        if result['phase_analysis']['phase_detection_success']:
            confidence += 0.4
        
        # Latent space quality
        latent_quality = result['vae_results']['latent_quality']
        if latent_quality.get('max_temperature_correlation', 0) > 0.5:
            confidence += 0.3
        
        # Critical properties accuracy (if applicable)
        if result['critical_properties']['has_critical_exponents']:
            exponent_accuracies = result['critical_properties'].get('exponent_accuracies', {})
            if exponent_accuracies:
                avg_accuracy = np.mean(list(exponent_accuracies.values()))
                confidence += 0.3 * avg_accuracy
        else:
            confidence += 0.3  # For non-critical systems, successful detection is good
        
        return min(confidence, 1.0)
    
    def _evaluate_continuous_detection_method(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate effectiveness of continuous transition detection."""
        return {
            'method': 'susceptibility_peak',
            'effectiveness': 1.0 if result['phase_analysis']['phase_detection_success'] else 0.0,
            'tc_accuracy': result['critical_properties'].get('tc_accuracy', 0.0)
        }
    
    def _evaluate_first_order_detection_method(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate effectiveness of first-order transition detection."""
        return {
            'method': 'gradient_maximum',
            'effectiveness': 1.0 if result['phase_analysis']['phase_detection_success'] else 0.0,
            'discontinuity_strength': np.random.random()  # Placeholder
        }
    
    def _evaluate_topological_detection_method(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate effectiveness of topological transition detection."""
        return {
            'method': 'latent_space_analysis',
            'effectiveness': 1.0 if result['phase_analysis']['phase_detection_success'] else 0.0,
            'topological_signature_strength': np.random.random()  # Placeholder
        }
    
    def _create_comparison_table(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create comparison table for publication."""
        
        table_data = []
        
        for system_name, result in results['individual_systems'].items():
            system_info = result['system_info']
            phase_analysis = result['phase_analysis']
            critical_properties = result['critical_properties']
            
            row = {
                'System': system_info['name'],
                'Universality Class': system_info['universality_class'],
                'Transition Type': system_info['transition_type'],
                'Phase Detection': '✓' if phase_analysis['phase_detection_success'] else '✗',
                'Tc Accuracy': f"{critical_properties.get('tc_accuracy', 0):.1%}" if critical_properties.get('tc_accuracy') else 'N/A'
            }
            
            # Add exponent accuracies if applicable
            exponent_accuracies = critical_properties.get('exponent_accuracies', {})
            for exponent in ['beta', 'nu']:
                if exponent in exponent_accuracies:
                    row[f'{exponent} Accuracy'] = f"{exponent_accuracies[exponent]:.1%}"
                else:
                    row[f'{exponent} Accuracy'] = 'N/A'
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        # Save table
        table_path = self.output_dir / 'comparison_table.csv'
        df.to_csv(table_path, index=False)
        
        return df
    
    def _create_universality_validation_table(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create universality class validation table."""
        
        if 'universality_validation' not in results:
            return pd.DataFrame()
        
        validation = results['universality_validation']
        table_data = []
        
        for system_name, system_validation in validation['system_validations'].items():
            row = {
                'System': system_name,
                'Universality Class': system_validation['universality_class'],
                'Correctly Identified': '✓' if system_validation['identified_correctly'] else '✗',
                'Confidence': f"{system_validation['confidence']:.1%}"
            }
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        # Save table
        table_path = self.output_dir / 'universality_validation_table.csv'
        df.to_csv(table_path, index=False)
        
        return df


def main():
    """Main function to run multi-system comparison example."""
    parser = argparse.ArgumentParser(description='Multi-System Comparison Analysis Example')
    parser.add_argument(
        '--quick-demo',
        action='store_true',
        help='Run with reduced parameters for faster demonstration'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/multi_system_comparison_demo',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Initialize and run comparison demo
    demo = MultiSystemComparisonDemo(
        output_dir=args.output_dir,
        quick_demo=args.quick_demo
    )
    
    try:
        results = demo.run_complete_comparison()
        
        # Print summary
        print("\n" + "="*60)
        print("MULTI-SYSTEM COMPARISON ANALYSIS COMPLETE")
        print("="*60)
        
        summary = results['summary_report']
        print(f"Systems Analyzed: {summary['systems_analyzed']}")
        
        if 'overall_performance' in summary:
            performance = summary['overall_performance']
            print(f"Overall Detection Success: {performance.get('total_success_rate', 0):.1%}")
        
        if 'universality_validation' in results:
            validation = results['universality_validation']
            print(f"Universality Class ID Success: {validation['success_rate']:.1%}")
        
        print("\nSystem-by-System Results:")
        for system_name, system_summary in summary['system_summaries'].items():
            print(f"  {system_summary['name']}: "
                  f"Detection {'✓' if system_summary['phase_detection_success'] else '✗'}, "
                  f"Tc Acc: {system_summary.get('tc_accuracy', 0):.1%}")
        
        print(f"\nResults saved to: {demo.output_dir}")
        print("Publication materials generated successfully!")
        
    except Exception as e:
        logger.error(f"Multi-system comparison demo failed: {e}")
        raise


if __name__ == "__main__":
    main()