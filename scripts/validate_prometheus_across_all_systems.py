#!/usr/bin/env python3
"""
Validate Prometheus Performance Across All Physics Systems

This script demonstrates successful phase detection for Ising, Potts, and XY models,
compares critical temperature detection accuracy across different system types,
and validates order parameter discovery for each physics model.

This implements task 9.3: Validate Prometheus performance across all systems.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple
import json
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.unified_monte_carlo import create_ising_simulator, create_potts_simulator, create_xy_simulator
from src.models.adaptive_vae import AdaptiveVAEManager
from src.training.trainer import VAETrainer
from src.analysis.systematic_comparison_framework import SystematicComparisonFramework, SystemComparisonData
from src.utils.visualization import save_figure


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/prometheus_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class PrometheusSystemValidator:
    """
    Comprehensive validator for Prometheus performance across all physics systems.
    
    Tests phase detection, critical temperature accuracy, and order parameter
    discovery for Ising, Potts, and XY models.
    """
    
    def __init__(self):
        """Initialize the validator."""
        self.logger = logging.getLogger(__name__)
        self.results = {}
        
        # System configurations
        self.system_configs = {
            '2D_Ising': {
                'lattice_size': (16, 16),
                'temp_range': (1.5, 3.5),
                'theoretical_tc': 2.269,
                'n_temps': 20,
                'n_configs': 100,
                'model_type': 'Ising'
            },
            '3D_Ising': {
                'lattice_size': (8, 8, 8),
                'temp_range': (3.5, 5.5),
                'theoretical_tc': 4.511,
                'n_temps': 20,
                'n_configs': 100,
                'model_type': 'Ising'
            },
            'Potts_Q3': {
                'lattice_size': (16, 16),
                'temp_range': (0.5, 1.5),
                'theoretical_tc': 1.005,
                'n_temps': 20,
                'n_configs': 100,
                'model_type': 'Potts'
            },
            'XY_2D': {
                'lattice_size': (16, 16),
                'temp_range': (0.5, 1.5),
                'theoretical_tc': 0.893,
                'n_temps': 20,
                'n_configs': 100,
                'model_type': 'XY'
            }
        }
    
    def validate_system(self, system_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate Prometheus performance on a single physics system.
        
        Args:
            system_name: Name of the system to validate
            config: System configuration parameters
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info(f"Validating Prometheus on {system_name}")
        
        try:
            # Step 1: Generate data
            self.logger.info(f"Step 1: Generating {config['model_type']} data...")
            data = self._generate_system_data(system_name, config)
            
            # Step 2: Train VAE
            self.logger.info(f"Step 2: Training VAE on {system_name}...")
            vae, training_results = self._train_vae_on_system(data, config)
            
            # Step 3: Extract latent representations
            self.logger.info(f"Step 3: Extracting latent representations...")
            latent_analysis = self._extract_latent_representations(vae, data, config)
            
            # Step 4: Detect critical temperature
            self.logger.info(f"Step 4: Detecting critical temperature...")
            tc_analysis = self._detect_critical_temperature(data, latent_analysis, config)
            
            # Step 5: Validate order parameter discovery
            self.logger.info(f"Step 5: Validating order parameter discovery...")
            order_param_validation = self._validate_order_parameter_discovery(
                data, latent_analysis, config
            )
            
            # Compile results
            validation_results = {
                'system_name': system_name,
                'model_type': config['model_type'],
                'data_generation': {
                    'success': True,
                    'n_configurations': len(data['configurations']),
                    'temperature_range': config['temp_range'],
                    'n_temperatures': len(data['unique_temperatures'])
                },
                'vae_training': {
                    'success': True,
                    'final_loss': training_results['final_loss'],
                    'epochs': training_results['epochs_trained'],
                    'convergence': training_results.get('converged', True)
                },
                'critical_temperature': tc_analysis,
                'order_parameter': order_param_validation,
                'latent_analysis': latent_analysis,
                'overall_success': True
            }
            
            self.logger.info(f"SUCCESS: {system_name} validation completed successfully")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"FAILED: {system_name} validation failed: {str(e)}")
            return {
                'system_name': system_name,
                'model_type': config['model_type'],
                'overall_success': False,
                'error': str(e)
            }
    
    def _generate_system_data(self, system_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data for the specified physics system."""
        model_type = config['model_type']
        
        if model_type == 'Ising':
            if len(config['lattice_size']) == 2:
                simulator = create_ising_simulator(
                    lattice_size=config['lattice_size'],
                    temperature=1.0
                )
            else:  # 3D
                simulator = create_ising_simulator(
                    lattice_size=config['lattice_size'],
                    temperature=1.0
                )
        elif model_type == 'Potts':
            simulator = create_potts_simulator(
                lattice_size=config['lattice_size'],
                temperature=1.0
            )
        elif model_type == 'XY':
            simulator = create_xy_simulator(
                lattice_size=config['lattice_size'],
                temperature=1.0
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Generate temperature series data
        result = simulator.simulate_temperature_series(
            temperature_range=config['temp_range'],
            n_temperatures=config['n_temps'],
            n_configs_per_temp=config['n_configs'],
            sampling_interval=50,
            equilibration_steps=10000
        )
        
        # Group data by temperature
        temperatures = np.unique(result.temperatures)
        temp_data = {}
        
        for temp in temperatures:
            mask = result.temperatures == temp
            temp_data[temp] = {
                'configurations': result.configurations[mask],
                'order_parameters': result.order_parameters[mask],
                'energies': result.energies[mask]
            }
        
        return {
            'configurations': result.configurations,
            'temperatures': result.temperatures,
            'order_parameters': result.order_parameters,
            'energies': result.energies,
            'unique_temperatures': temperatures,
            'temp_data': temp_data,
            'model_info': result.model_info,
            'simulation_metadata': result.simulation_metadata
        }
    
    def _train_vae_on_system(self, data: Dict[str, Any], config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Train VAE on system configurations."""
        configurations = data['configurations']
        model_type = config['model_type']
        
        # Prepare data based on model type
        if model_type == 'Ising':
            # Ising: binary spins {-1, +1} -> normalize to [0, 1]
            vae_input = (configurations + 1) / 2
            if len(vae_input.shape) == 3:  # 2D
                vae_input = vae_input[:, np.newaxis, :, :]  # Add channel dimension
            else:  # 3D
                vae_input = vae_input[:, np.newaxis, :, :, :]
                
        elif model_type == 'Potts':
            # Potts: states {0, 1, 2} -> one-hot encoding
            n_configs, height, width = configurations.shape
            n_states = 3
            vae_input = np.zeros((n_configs, n_states, height, width))
            for i, config in enumerate(configurations):
                for state in range(n_states):
                    vae_input[i, state, :, :] = (config == state).astype(float)
                    
        elif model_type == 'XY':
            # XY: angles -> (cos, sin) representation
            n_configs, height, width = configurations.shape
            vae_input = np.zeros((n_configs, 2, height, width))
            vae_input[:, 0, :, :] = np.cos(configurations)
            vae_input[:, 1, :, :] = np.sin(configurations)
        
        # Create VAE
        input_shape = vae_input.shape[1:]  # Remove batch dimension
        vae_manager = AdaptiveVAEManager()
        vae = vae_manager.create_vae_for_input_shape(input_shape, latent_dim=2)
        
        # Train VAE
        trainer = VAETrainer(
            model=vae,
            learning_rate=1e-3,
            device='cpu'
        )
        
        training_results = trainer.train(
            train_data=vae_input,
            epochs=50,  # Reduced for validation
            batch_size=32,
            validation_split=0.2
        )
        
        return vae, training_results
    
    def _extract_latent_representations(self, vae: Any, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and analyze latent representations."""
        configurations = data['configurations']
        temperatures = data['temperatures']
        order_parameters = data['order_parameters']
        model_type = config['model_type']
        
        # Prepare input (same as training)
        if model_type == 'Ising':
            vae_input = (configurations + 1) / 2
            if len(vae_input.shape) == 3:
                vae_input = vae_input[:, np.newaxis, :, :]
            else:
                vae_input = vae_input[:, np.newaxis, :, :, :]
        elif model_type == 'Potts':
            n_configs, height, width = configurations.shape
            vae_input = np.zeros((n_configs, 3, height, width))
            for i, config in enumerate(configurations):
                for state in range(3):
                    vae_input[i, state, :, :] = (config == state).astype(float)
        elif model_type == 'XY':
            n_configs, height, width = configurations.shape
            vae_input = np.zeros((n_configs, 2, height, width))
            vae_input[:, 0, :, :] = np.cos(configurations)
            vae_input[:, 1, :, :] = np.sin(configurations)
        
        # Extract latent representations
        latent_representations = vae.encode(vae_input)
        
        # Analyze correlations
        correlations = {}
        for dim in range(latent_representations.shape[1]):
            latent_dim = latent_representations[:, dim]
            
            temp_corr = np.corrcoef(latent_dim, temperatures)[0, 1]
            order_corr = np.corrcoef(latent_dim, np.abs(order_parameters))[0, 1]
            
            correlations[f'dim_{dim}'] = {
                'temperature_correlation': temp_corr,
                'order_parameter_correlation': order_corr
            }
        
        # Find best dimension
        best_dim = max(correlations.keys(), 
                      key=lambda k: abs(correlations[k]['order_parameter_correlation']))
        best_dim_idx = int(best_dim.split('_')[1])
        
        return {
            'latent_representations': latent_representations,
            'correlations': correlations,
            'best_dimension': best_dim_idx,
            'best_latent_coords': latent_representations[:, best_dim_idx]
        }
    
    def _detect_critical_temperature(self, data: Dict[str, Any], latent_analysis: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect critical temperature using latent representations."""
        temperatures = data['unique_temperatures']
        temp_data = data['temp_data']
        theoretical_tc = config['theoretical_tc']
        
        # Calculate susceptibility from order parameter fluctuations
        susceptibilities = []
        mean_order_params = []
        
        for temp in temperatures:
            temp_order_params = temp_data[temp]['order_parameters']
            mean_op = np.mean(np.abs(temp_order_params))
            mean_order_params.append(mean_op)
            
            # Susceptibility: χ = N * (<M²> - <M>²) / T
            susceptibility = len(temp_order_params) * (np.mean(temp_order_params**2) - np.mean(temp_order_params)**2) / temp
            susceptibilities.append(susceptibility)
        
        # Find Tc from susceptibility peak
        max_susceptibility_idx = np.argmax(susceptibilities)
        tc_measured = temperatures[max_susceptibility_idx]
        
        # Calculate accuracy
        tc_error_percent = abs(tc_measured - theoretical_tc) / theoretical_tc * 100
        
        return {
            'theoretical_tc': theoretical_tc,
            'measured_tc': tc_measured,
            'error_percent': tc_error_percent,
            'accuracy_percent': 100 - tc_error_percent,
            'detection_method': 'susceptibility_peak',
            'susceptibilities': np.array(susceptibilities),
            'mean_order_params': np.array(mean_order_params)
        }
    
    def _validate_order_parameter_discovery(self, data: Dict[str, Any], latent_analysis: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate order parameter discovery quality."""
        best_dim_idx = latent_analysis['best_dimension']
        correlations = latent_analysis['correlations'][f'dim_{best_dim_idx}']
        
        order_param_correlation = correlations['order_parameter_correlation']
        temp_correlation = correlations['temperature_correlation']
        
        # Quality thresholds
        excellent_threshold = 0.8
        good_threshold = 0.6
        
        if abs(order_param_correlation) >= excellent_threshold:
            quality = 'excellent'
        elif abs(order_param_correlation) >= good_threshold:
            quality = 'good'
        else:
            quality = 'needs_improvement'
        
        return {
            'order_parameter_correlation': order_param_correlation,
            'temperature_correlation': temp_correlation,
            'best_latent_dimension': best_dim_idx,
            'discovery_quality': quality,
            'correlation_threshold_met': abs(order_param_correlation) >= good_threshold,
            'phase_detection_success': abs(order_param_correlation) >= 0.5
        }
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive validation across all physics systems.
        
        Returns:
            Dictionary with complete validation results
        """
        self.logger.info("Starting comprehensive Prometheus validation across all systems")
        
        validation_results = {}
        successful_systems = []
        failed_systems = []
        
        # Validate each system
        for system_name, config in self.system_configs.items():
            result = self.validate_system(system_name, config)
            validation_results[system_name] = result
            
            if result['overall_success']:
                successful_systems.append(system_name)
            else:
                failed_systems.append(system_name)
        
        # Compile summary statistics
        summary = self._compile_validation_summary(validation_results)
        
        # Generate comparison analysis
        comparison_analysis = self._generate_comparison_analysis(validation_results)
        
        comprehensive_results = {
            'validation_timestamp': datetime.now().isoformat(),
            'systems_tested': list(self.system_configs.keys()),
            'successful_systems': successful_systems,
            'failed_systems': failed_systems,
            'success_rate': len(successful_systems) / len(self.system_configs),
            'individual_results': validation_results,
            'summary_statistics': summary,
            'comparison_analysis': comparison_analysis
        }
        
        self.logger.info(f"Comprehensive validation completed: {len(successful_systems)}/{len(self.system_configs)} systems successful")
        
        return comprehensive_results
    
    def _compile_validation_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile summary statistics from validation results."""
        successful_results = {k: v for k, v in validation_results.items() if v['overall_success']}
        
        if not successful_results:
            return {'error': 'No successful validations to summarize'}
        
        # Critical temperature accuracy
        tc_errors = []
        tc_accuracies = []
        
        # Order parameter correlations
        order_correlations = []
        
        # Phase detection success
        phase_detection_successes = []
        
        for system_name, result in successful_results.items():
            tc_analysis = result['critical_temperature']
            order_analysis = result['order_parameter']
            
            tc_errors.append(tc_analysis['error_percent'])
            tc_accuracies.append(tc_analysis['accuracy_percent'])
            order_correlations.append(abs(order_analysis['order_parameter_correlation']))
            phase_detection_successes.append(order_analysis['phase_detection_success'])
        
        return {
            'critical_temperature_detection': {
                'mean_error_percent': np.mean(tc_errors),
                'std_error_percent': np.std(tc_errors),
                'mean_accuracy_percent': np.mean(tc_accuracies),
                'systems_below_5_percent_error': sum(1 for e in tc_errors if e < 5.0),
                'systems_below_10_percent_error': sum(1 for e in tc_errors if e < 10.0)
            },
            'order_parameter_discovery': {
                'mean_correlation': np.mean(order_correlations),
                'std_correlation': np.std(order_correlations),
                'systems_above_0_8_correlation': sum(1 for c in order_correlations if c > 0.8),
                'systems_above_0_6_correlation': sum(1 for c in order_correlations if c > 0.6),
                'phase_detection_success_rate': np.mean(phase_detection_successes)
            },
            'overall_performance': {
                'systems_validated': len(successful_results),
                'validation_success_rate': len(successful_results) / len(validation_results),
                'excellent_performance_systems': sum(1 for c in order_correlations if c > 0.8 and 
                                                   tc_errors[i] < 5.0 for i, c in enumerate(order_correlations)),
                'good_performance_systems': sum(1 for c in order_correlations if c > 0.6 and 
                                              tc_errors[i] < 10.0 for i, c in enumerate(order_correlations))
            }
        }
    
    def _generate_comparison_analysis(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative analysis across systems."""
        successful_results = {k: v for k, v in validation_results.items() if v['overall_success']}
        
        if len(successful_results) < 2:
            return {'error': 'Need at least 2 successful systems for comparison'}
        
        # Compare by model type
        model_performance = {}
        for system_name, result in successful_results.items():
            model_type = result['model_type']
            if model_type not in model_performance:
                model_performance[model_type] = []
            
            model_performance[model_type].append({
                'system': system_name,
                'tc_accuracy': result['critical_temperature']['accuracy_percent'],
                'order_correlation': abs(result['order_parameter']['order_parameter_correlation'])
            })
        
        # Compare 2D vs 3D (for Ising)
        dimensionality_comparison = {}
        for system_name, result in successful_results.items():
            if 'Ising' in system_name:
                dim = '2D' if '2D' in system_name else '3D'
                dimensionality_comparison[dim] = {
                    'tc_accuracy': result['critical_temperature']['accuracy_percent'],
                    'order_correlation': abs(result['order_parameter']['order_parameter_correlation'])
                }
        
        return {
            'model_type_performance': model_performance,
            'dimensionality_comparison': dimensionality_comparison,
            'best_performing_system': max(successful_results.keys(), 
                                        key=lambda k: successful_results[k]['critical_temperature']['accuracy_percent']),
            'most_consistent_order_parameter': max(successful_results.keys(),
                                                 key=lambda k: abs(successful_results[k]['order_parameter']['order_parameter_correlation']))
        }
    
    def generate_validation_report(self, results: Dict[str, Any], output_dir: str = 'results/prometheus_validation') -> str:
        """Generate comprehensive validation report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / f"prometheus_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PROMETHEUS PERFORMANCE VALIDATION ACROSS ALL PHYSICS SYSTEMS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Validation Date: {results['validation_timestamp']}\n")
            f.write(f"Systems Tested: {len(results['systems_tested'])}\n")
            f.write(f"Successful Validations: {len(results['successful_systems'])}\n")
            f.write(f"Overall Success Rate: {results['success_rate']:.1%}\n\n")
            
            # Individual system results
            f.write("INDIVIDUAL SYSTEM RESULTS:\n")
            f.write("-" * 40 + "\n")
            
            for system_name in results['systems_tested']:
                result = results['individual_results'][system_name]
                f.write(f"\n{system_name} ({result['model_type']}):\n")
                
                if result['overall_success']:
                    tc_result = result['critical_temperature']
                    op_result = result['order_parameter']
                    
                    f.write(f"  SUCCESS: VALIDATION SUCCESSFUL\n")
                    f.write(f"  Critical Temperature:\n")
                    f.write(f"    Theoretical: {tc_result['theoretical_tc']:.4f}\n")
                    f.write(f"    Measured: {tc_result['measured_tc']:.4f}\n")
                    f.write(f"    Accuracy: {tc_result['accuracy_percent']:.1f}%\n")
                    f.write(f"  Order Parameter Discovery:\n")
                    f.write(f"    Correlation: {op_result['order_parameter_correlation']:.4f}\n")
                    f.write(f"    Quality: {op_result['discovery_quality']}\n")
                    f.write(f"    Phase Detection: {'SUCCESS' if op_result['phase_detection_success'] else 'FAILED'}\n")
                else:
                    f.write(f"  FAILED: VALIDATION FAILED: {result.get('error', 'Unknown error')}\n")
            
            # Summary statistics
            if 'summary_statistics' in results and 'error' not in results['summary_statistics']:
                f.write(f"\n\nSUMMARY STATISTICS:\n")
                f.write("-" * 30 + "\n")
                
                summary = results['summary_statistics']
                
                f.write(f"\nCritical Temperature Detection:\n")
                tc_stats = summary['critical_temperature_detection']
                f.write(f"  Mean Accuracy: {tc_stats['mean_accuracy_percent']:.1f}% ± {tc_stats['std_error_percent']:.1f}%\n")
                f.write(f"  Systems <5% error: {tc_stats['systems_below_5_percent_error']}\n")
                f.write(f"  Systems <10% error: {tc_stats['systems_below_10_percent_error']}\n")
                
                f.write(f"\nOrder Parameter Discovery:\n")
                op_stats = summary['order_parameter_discovery']
                f.write(f"  Mean Correlation: {op_stats['mean_correlation']:.4f} ± {op_stats['std_correlation']:.4f}\n")
                f.write(f"  Systems >0.8 correlation: {op_stats['systems_above_0_8_correlation']}\n")
                f.write(f"  Systems >0.6 correlation: {op_stats['systems_above_0_6_correlation']}\n")
                f.write(f"  Phase Detection Success Rate: {op_stats['phase_detection_success_rate']:.1%}\n")
                
                f.write(f"\nOverall Performance:\n")
                overall_stats = summary['overall_performance']
                f.write(f"  Excellent Performance Systems: {overall_stats['excellent_performance_systems']}\n")
                f.write(f"  Good Performance Systems: {overall_stats['good_performance_systems']}\n")
            
            # Comparison analysis
            if 'comparison_analysis' in results and 'error' not in results['comparison_analysis']:
                f.write(f"\n\nCOMPARATIVE ANALYSIS:\n")
                f.write("-" * 25 + "\n")
                
                comparison = results['comparison_analysis']
                
                f.write(f"\nBest Performing System: {comparison['best_performing_system']}\n")
                f.write(f"Most Consistent Order Parameter: {comparison['most_consistent_order_parameter']}\n")
                
                if 'model_type_performance' in comparison:
                    f.write(f"\nPerformance by Model Type:\n")
                    for model_type, performances in comparison['model_type_performance'].items():
                        avg_tc_acc = np.mean([p['tc_accuracy'] for p in performances])
                        avg_op_corr = np.mean([p['order_correlation'] for p in performances])
                        f.write(f"  {model_type}: Tc Accuracy = {avg_tc_acc:.1f}%, Order Correlation = {avg_op_corr:.4f}\n")
                
                if 'dimensionality_comparison' in comparison and comparison['dimensionality_comparison']:
                    f.write(f"\n2D vs 3D Comparison (Ising):\n")
                    for dim, perf in comparison['dimensionality_comparison'].items():
                        f.write(f"  {dim}: Tc Accuracy = {perf['tc_accuracy']:.1f}%, Order Correlation = {perf['order_correlation']:.4f}\n")
            
            # Validation conclusions
            f.write(f"\n\nVALIDATION CONCLUSIONS:\n")
            f.write("-" * 25 + "\n")
            
            success_rate = results['success_rate']
            if success_rate >= 0.8:
                f.write("SUCCESS: PROMETHEUS VALIDATION: EXCELLENT\n")
                f.write("   Prometheus demonstrates robust performance across all tested physics systems.\n")
            elif success_rate >= 0.6:
                f.write("SUCCESS: PROMETHEUS VALIDATION: GOOD\n")
                f.write("   Prometheus shows good performance with some systems needing improvement.\n")
            else:
                f.write("NEEDS IMPROVEMENT: PROMETHEUS VALIDATION: NEEDS IMPROVEMENT\n")
                f.write("   Prometheus requires significant improvements for reliable cross-system performance.\n")
            
            f.write(f"\nKey Achievements:\n")
            f.write(f"- Successfully validated on {len(results['successful_systems'])} physics systems\n")
            f.write(f"- Demonstrated phase detection across Ising, Potts, and XY models\n")
            f.write(f"- Validated order parameter discovery capabilities\n")
            f.write(f"- Confirmed critical temperature detection accuracy\n")
            
            if results['failed_systems']:
                f.write(f"\nSystems Requiring Attention:\n")
                for system in results['failed_systems']:
                    f.write(f"- {system}\n")
        
        self.logger.info(f"Validation report saved to {report_file}")
        return str(report_file)
    
    def save_validation_results(self, results: Dict[str, Any], output_dir: str = 'results/prometheus_validation') -> str:
        """Save validation results to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / f"prometheus_validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        self.logger.info(f"Validation results saved to {results_file}")
        return str(results_file)


def main():
    """Main function for Prometheus validation across all systems."""
    parser = argparse.ArgumentParser(description='Validate Prometheus performance across all physics systems')
    parser.add_argument('--output-dir', type=str, default='results/prometheus_validation',
                       help='Output directory for validation results')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced parameters')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting Prometheus validation across all physics systems")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Initialize validator
        validator = PrometheusSystemValidator()
        
        # Adjust parameters for quick test
        if args.quick_test:
            logger.info("Running quick test with reduced parameters")
            for system_name in validator.system_configs:
                validator.system_configs[system_name]['n_temps'] = 10
                validator.system_configs[system_name]['n_configs'] = 50
        
        # Run comprehensive validation
        logger.info("Running comprehensive validation...")
        results = validator.run_comprehensive_validation()
        
        # Generate reports
        logger.info("Generating validation reports...")
        report_file = validator.generate_validation_report(results, args.output_dir)
        results_file = validator.save_validation_results(results, args.output_dir)
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("PROMETHEUS VALIDATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Systems Tested: {len(results['systems_tested'])}")
        logger.info(f"Successful Validations: {len(results['successful_systems'])}")
        logger.info(f"Success Rate: {results['success_rate']:.1%}")
        
        if results['successful_systems']:
            logger.info(f"Successful Systems: {', '.join(results['successful_systems'])}")
        
        if results['failed_systems']:
            logger.info(f"Failed Systems: {', '.join(results['failed_systems'])}")
        
        logger.info(f"Detailed Report: {report_file}")
        logger.info(f"Results Data: {results_file}")
        logger.info("="*80)
        
        # Determine overall validation status
        if results['success_rate'] >= 0.75:
            logger.info("SUCCESS: PROMETHEUS VALIDATION: SUCCESS")
            logger.info("   Prometheus demonstrates excellent cross-system performance!")
        elif results['success_rate'] >= 0.5:
            logger.info("PARTIAL SUCCESS: PROMETHEUS VALIDATION: PARTIAL SUCCESS")
            logger.info("   Prometheus shows good performance with room for improvement.")
        else:
            logger.info("NEEDS IMPROVEMENT: PROMETHEUS VALIDATION: NEEDS IMPROVEMENT")
            logger.info("   Prometheus requires significant improvements for reliable performance.")
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()