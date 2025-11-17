#!/usr/bin/env python3
"""
Critical Exponent Extraction and Validation Example

This script demonstrates the critical exponent extraction process using
VAE latent representations and validates the results against theoretical
predictions for different statistical mechanics systems.

Requirements: 5.5 - Demonstrate critical exponent extraction and validation

Usage:
    python examples/critical_exponent_extraction_example.py [--system ising_3d|potts|xy]
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
from scipy import stats

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from analysis.robust_critical_exponent_extractor import RobustCriticalExponentExtractor
from analysis.correlation_length_calculator import CorrelationLengthCalculator
from analysis.robust_power_law_fitter import RobustPowerLawFitter
from data.high_quality_data_generator import HighQualityDataGenerator
from models.adaptive_vae import AdaptiveVAE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CriticalExponentExtractionDemo:
    """
    Demonstration of critical exponent extraction and validation.
    
    This class shows how to:
    1. Generate high-quality data near critical points
    2. Train VAE to learn order parameter representations
    3. Extract critical exponents (β, ν, γ) from latent space
    4. Validate results against theoretical predictions
    5. Perform statistical analysis and error estimation
    """
    
    def __init__(self, system_type: str = 'ising_3d', output_dir: str = 'results/exponent_demo'):
        """
        Initialize the critical exponent extraction demo.
        
        Args:
            system_type: Type of physics system ('ising_3d', 'potts', 'xy')
            output_dir: Directory to save results
        """
        self.system_type = system_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set theoretical values based on system type
        self.theoretical_values = self._get_theoretical_values(system_type)
        
        logger.info(f"Initialized critical exponent extraction for {system_type}")
        logger.info(f"Theoretical values: {self.theoretical_values}")
    
    def _get_theoretical_values(self, system_type: str) -> Dict[str, Any]:
        """Get theoretical critical values for different systems."""
        theoretical_values = {
            'ising_3d': {
                'tc': 4.511,
                'beta': 0.326,
                'nu': 0.630,
                'gamma': 1.237,
                'universality_class': '3D Ising'
            },
            'ising_2d': {
                'tc': 2.269,
                'beta': 0.125,
                'nu': 1.0,
                'gamma': 1.75,
                'universality_class': '2D Ising'
            },
            'potts': {
                'tc': 1.005,
                'beta': None,  # First-order transition
                'nu': None,
                'gamma': None,
                'universality_class': 'First-order'
            },
            'xy': {
                'tc': None,  # KT transition
                'beta': None,
                'nu': None,
                'gamma': None,
                'universality_class': 'Kosterlitz-Thouless'
            }
        }
        
        return theoretical_values.get(system_type, theoretical_values['ising_3d'])
    
    def run_complete_extraction_demo(self) -> Dict[str, Any]:
        """
        Run complete critical exponent extraction demonstration.
        
        Returns:
            Dictionary containing all extraction results and validation
        """
        logger.info("Starting critical exponent extraction demonstration...")
        
        results = {}
        
        # Step 1: Generate high-quality data
        logger.info("Step 1: Generating high-quality data near critical point...")
        data = self._generate_critical_point_data()
        results['data'] = data
        
        # Step 2: Train VAE and extract latent representations
        logger.info("Step 2: Training VAE and extracting latent representations...")
        vae_results = self._train_vae_for_exponents(data)
        results['vae_results'] = vae_results
        
        # Step 3: Extract critical temperature
        logger.info("Step 3: Extracting critical temperature...")
        tc_results = self._extract_critical_temperature(data, vae_results)
        results['tc_results'] = tc_results
        
        # Step 4: Extract beta exponent (magnetization)
        logger.info("Step 4: Extracting beta exponent...")
        beta_results = self._extract_beta_exponent(data, vae_results, tc_results)
        results['beta_results'] = beta_results
        
        # Step 5: Extract nu exponent (correlation length)
        logger.info("Step 5: Extracting nu exponent...")
        nu_results = self._extract_nu_exponent(data, vae_results, tc_results)
        results['nu_results'] = nu_results
        
        # Step 6: Extract gamma exponent (susceptibility)
        logger.info("Step 6: Extracting gamma exponent...")
        gamma_results = self._extract_gamma_exponent(data, vae_results, tc_results)
        results['gamma_results'] = gamma_results
        
        # Step 7: Validate against theoretical predictions
        logger.info("Step 7: Validating against theoretical predictions...")
        validation_results = self._validate_extracted_exponents(results)
        results['validation'] = validation_results
        
        # Step 8: Generate analysis plots
        logger.info("Step 8: Generating analysis plots...")
        plots = self._generate_analysis_plots(results)
        results['plots'] = plots
        
        # Step 9: Create summary report
        logger.info("Step 9: Creating summary report...")
        summary = self._create_extraction_summary(results)
        results['summary'] = summary
        
        logger.info("Critical exponent extraction demonstration completed!")
        return results
    
    def _generate_critical_point_data(self) -> Dict[str, Any]:
        """Generate high-quality data focused around the critical point."""
        
        # Initialize data generator
        data_generator = HighQualityDataGenerator(
            model_type=self.system_type,
            output_dir=str(self.output_dir / 'data')
        )
        
        # Define temperature range around critical point
        tc_theoretical = self.theoretical_values['tc']
        if tc_theoretical is not None:
            # Focus on narrow range around Tc for better exponent extraction
            temp_range = (tc_theoretical - 0.5, tc_theoretical + 0.5)
            n_temperatures = 21
        else:
            # For systems without well-defined Tc, use broader range
            temp_range = (0.5, 2.5)
            n_temperatures = 15
        
        temperatures = np.linspace(temp_range[0], temp_range[1], n_temperatures)
        
        # Generate data for multiple system sizes (finite-size scaling)
        system_sizes = [16, 32, 48] if self.system_type == 'ising_3d' else [32, 64, 96]
        
        all_data = {}
        
        for L in system_sizes:
            logger.info(f"Generating data for L={L}...")
            
            dataset = data_generator.generate_comprehensive_dataset(
                system_size=L,
                temperatures=temperatures,
                n_configs_per_temp=200,  # High statistics for accurate exponents
                equilibration_steps=50000,
                sampling_interval=500
            )
            
            all_data[f'L_{L}'] = dataset
        
        return {
            'datasets': all_data,
            'temperatures': temperatures,
            'system_sizes': system_sizes,
            'theoretical_tc': tc_theoretical
        }
    
    def _train_vae_for_exponents(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Train VAE specifically optimized for critical exponent extraction."""
        
        # Use largest system size for training
        largest_L = max(data['system_sizes'])
        training_data = data['datasets'][f'L_{largest_L}']
        
        # Initialize VAE with physics-informed architecture
        input_shape = training_data['configurations'].shape[1:]
        vae = AdaptiveVAE(
            input_shape=input_shape,
            latent_dim=3,  # Use 3D latent space for better representation
            model_type=self.system_type,
            physics_informed=True  # Enable physics-informed loss
        )
        
        # Train with careful hyperparameters for critical behavior
        history = vae.train(
            training_data['configurations'],
            epochs=150,
            batch_size=64,
            learning_rate=1e-3,
            validation_split=0.2,
            beta_schedule='warmup',  # Use beta-VAE warmup for better latent learning
            physics_targets={
                'temperatures': training_data['temperatures'],
                'magnetizations': training_data['magnetizations']
            }
        )
        
        # Extract latent representations for all system sizes
        latent_representations = {}
        
        for size_key, dataset in data['datasets'].items():
            latent_z = vae.encode(dataset['configurations'])
            
            # Identify best order parameter dimension
            correlations = []
            for dim in range(latent_z.shape[1]):
                corr = np.corrcoef(latent_z[:, dim], np.abs(dataset['magnetizations']))[0, 1]
                correlations.append(abs(corr))
            
            best_dim = np.argmax(correlations)
            
            latent_representations[size_key] = {
                'latent_coords': latent_z,
                'order_parameter_dim': best_dim,
                'order_parameter': latent_z[:, best_dim],
                'correlations': correlations,
                'temperatures': dataset['temperatures'],
                'magnetizations': dataset['magnetizations'],
                'energies': dataset['energies']
            }
        
        return {
            'vae_model': vae,
            'training_history': history,
            'latent_representations': latent_representations
        }
    
    def _extract_critical_temperature(
        self, 
        data: Dict[str, Any], 
        vae_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract critical temperature using multiple methods."""
        
        extractor = RobustCriticalExponentExtractor()
        
        tc_results = {}
        
        for size_key, latent_data in vae_results['latent_representations'].items():
            
            # Method 1: Susceptibility peak
            susceptibility = self._compute_susceptibility(
                latent_data['order_parameter'],
                latent_data['temperatures']
            )
            
            tc_susceptibility = latent_data['temperatures'][np.argmax(susceptibility)]
            
            # Method 2: Specific heat peak
            specific_heat = self._compute_specific_heat(
                latent_data['energies'],
                latent_data['temperatures']
            )
            
            tc_specific_heat = latent_data['temperatures'][np.argmax(specific_heat)]
            
            # Method 3: Binder cumulant crossing (if multiple sizes available)
            binder_cumulant = self._compute_binder_cumulant(latent_data['order_parameter'])
            
            tc_results[size_key] = {
                'tc_susceptibility': tc_susceptibility,
                'tc_specific_heat': tc_specific_heat,
                'susceptibility': susceptibility,
                'specific_heat': specific_heat,
                'binder_cumulant': binder_cumulant
            }
        
        # Find consensus Tc estimate
        all_tc_estimates = []
        for result in tc_results.values():
            all_tc_estimates.extend([
                result['tc_susceptibility'],
                result['tc_specific_heat']
            ])
        
        tc_consensus = np.median(all_tc_estimates)
        tc_std = np.std(all_tc_estimates)
        
        return {
            'size_results': tc_results,
            'tc_consensus': tc_consensus,
            'tc_uncertainty': tc_std,
            'theoretical_tc': data['theoretical_tc']
        }
    
    def _extract_beta_exponent(
        self,
        data: Dict[str, Any],
        vae_results: Dict[str, Any],
        tc_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract beta exponent from magnetization scaling."""
        
        fitter = RobustPowerLawFitter()
        tc_estimate = tc_results['tc_consensus']
        
        beta_results = {}
        
        for size_key, latent_data in vae_results['latent_representations'].items():
            temperatures = latent_data['temperatures']
            order_parameter = np.abs(latent_data['order_parameter'])
            
            # Focus on temperatures below Tc
            below_tc_mask = temperatures < tc_estimate
            if np.sum(below_tc_mask) < 5:  # Need enough points
                continue
                
            T_below = temperatures[below_tc_mask]
            M_below = order_parameter[below_tc_mask]
            
            # Fit M ∝ (Tc - T)^β
            reduced_temp = (tc_estimate - T_below) / tc_estimate
            
            # Remove points too close to Tc (avoid corrections to scaling)
            fit_mask = reduced_temp > 0.01
            if np.sum(fit_mask) < 3:
                continue
                
            T_fit = reduced_temp[fit_mask]
            M_fit = M_below[fit_mask]
            
            # Perform power-law fit with bootstrap
            fit_result = fitter.fit_power_law_with_bootstrap(
                x_data=T_fit,
                y_data=M_fit,
                bootstrap_samples=1000
            )
            
            beta_results[size_key] = {
                'beta_estimate': fit_result['exponent'],
                'beta_uncertainty': fit_result['exponent_std'],
                'fit_quality': fit_result['r_squared'],
                'fit_range': (T_fit.min(), T_fit.max()),
                'bootstrap_distribution': fit_result['bootstrap_exponents']
            }
        
        # Combine results across system sizes
        all_betas = [r['beta_estimate'] for r in beta_results.values()]
        beta_final = np.mean(all_betas)
        beta_std = np.std(all_betas)
        
        return {
            'size_results': beta_results,
            'beta_final': beta_final,
            'beta_uncertainty': beta_std,
            'theoretical_beta': self.theoretical_values['beta']
        }
    
    def _extract_nu_exponent(
        self,
        data: Dict[str, Any],
        vae_results: Dict[str, Any],
        tc_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract nu exponent from correlation length scaling."""
        
        correlation_calc = CorrelationLengthCalculator()
        fitter = RobustPowerLawFitter()
        tc_estimate = tc_results['tc_consensus']
        
        nu_results = {}
        
        for size_key, latent_data in vae_results['latent_representations'].items():
            temperatures = latent_data['temperatures']
            
            # Compute correlation length from latent space
            correlation_lengths = correlation_calc.compute_from_latent_space(
                latent_coords=latent_data['latent_coords'],
                temperatures=temperatures
            )
            
            # Focus on temperatures near Tc
            near_tc_mask = np.abs(temperatures - tc_estimate) / tc_estimate < 0.3
            if np.sum(near_tc_mask) < 5:
                continue
                
            T_near = temperatures[near_tc_mask]
            xi_near = correlation_lengths[near_tc_mask]
            
            # Fit ξ ∝ |T - Tc|^(-ν)
            reduced_temp = np.abs(T_near - tc_estimate) / tc_estimate
            
            # Remove points too close to Tc
            fit_mask = reduced_temp > 0.01
            if np.sum(fit_mask) < 3:
                continue
                
            T_fit = reduced_temp[fit_mask]
            xi_fit = xi_near[fit_mask]
            
            # Perform power-law fit
            fit_result = fitter.fit_power_law_with_bootstrap(
                x_data=T_fit,
                y_data=xi_fit,
                bootstrap_samples=1000
            )
            
            nu_results[size_key] = {
                'nu_estimate': fit_result['exponent'],
                'nu_uncertainty': fit_result['exponent_std'],
                'fit_quality': fit_result['r_squared'],
                'correlation_lengths': correlation_lengths,
                'bootstrap_distribution': fit_result['bootstrap_exponents']
            }
        
        # Combine results
        all_nus = [r['nu_estimate'] for r in nu_results.values()]
        nu_final = np.mean(all_nus)
        nu_std = np.std(all_nus)
        
        return {
            'size_results': nu_results,
            'nu_final': nu_final,
            'nu_uncertainty': nu_std,
            'theoretical_nu': self.theoretical_values['nu']
        }
    
    def _extract_gamma_exponent(
        self,
        data: Dict[str, Any],
        vae_results: Dict[str, Any],
        tc_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract gamma exponent from susceptibility scaling."""
        
        fitter = RobustPowerLawFitter()
        tc_estimate = tc_results['tc_consensus']
        
        gamma_results = {}
        
        for size_key, latent_data in vae_results['latent_representations'].items():
            temperatures = latent_data['temperatures']
            
            # Compute susceptibility from order parameter fluctuations
            susceptibility = self._compute_susceptibility(
                latent_data['order_parameter'],
                temperatures
            )
            
            # Focus on temperatures above Tc
            above_tc_mask = temperatures > tc_estimate
            if np.sum(above_tc_mask) < 5:
                continue
                
            T_above = temperatures[above_tc_mask]
            chi_above = susceptibility[above_tc_mask]
            
            # Fit χ ∝ (T - Tc)^(-γ)
            reduced_temp = (T_above - tc_estimate) / tc_estimate
            
            # Remove points too close to Tc
            fit_mask = reduced_temp > 0.01
            if np.sum(fit_mask) < 3:
                continue
                
            T_fit = reduced_temp[fit_mask]
            chi_fit = chi_above[fit_mask]
            
            # Perform power-law fit
            fit_result = fitter.fit_power_law_with_bootstrap(
                x_data=T_fit,
                y_data=chi_fit,
                bootstrap_samples=1000
            )
            
            gamma_results[size_key] = {
                'gamma_estimate': fit_result['exponent'],
                'gamma_uncertainty': fit_result['exponent_std'],
                'fit_quality': fit_result['r_squared'],
                'susceptibility': susceptibility,
                'bootstrap_distribution': fit_result['bootstrap_exponents']
            }
        
        # Combine results
        all_gammas = [r['gamma_estimate'] for r in gamma_results.values()]
        gamma_final = np.mean(all_gammas)
        gamma_std = np.std(all_gammas)
        
        return {
            'size_results': gamma_results,
            'gamma_final': gamma_final,
            'gamma_uncertainty': gamma_std,
            'theoretical_gamma': self.theoretical_values['gamma']
        }
    
    def _validate_extracted_exponents(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extracted exponents against theoretical predictions."""
        
        validation = {}
        
        # Validate each exponent
        for exponent in ['beta', 'nu', 'gamma']:
            if f'{exponent}_results' in results:
                measured = results[f'{exponent}_results'][f'{exponent}_final']
                uncertainty = results[f'{exponent}_results'][f'{exponent}_uncertainty']
                theoretical = self.theoretical_values[exponent]
                
                if theoretical is not None:
                    # Calculate accuracy
                    relative_error = abs(measured - theoretical) / theoretical
                    accuracy = 1.0 - relative_error
                    
                    # Check if within error bars
                    within_error = abs(measured - theoretical) <= 2 * uncertainty
                    
                    # Statistical significance test
                    z_score = abs(measured - theoretical) / uncertainty if uncertainty > 0 else np.inf
                    p_value = 2 * (1 - stats.norm.cdf(z_score))  # Two-tailed test
                    
                    validation[exponent] = {
                        'measured': measured,
                        'theoretical': theoretical,
                        'uncertainty': uncertainty,
                        'relative_error': relative_error,
                        'accuracy': accuracy,
                        'within_error_bars': within_error,
                        'z_score': z_score,
                        'p_value': p_value,
                        'significant_difference': p_value < 0.05
                    }
        
        # Validate critical temperature
        if 'tc_results' in results:
            tc_measured = results['tc_results']['tc_consensus']
            tc_uncertainty = results['tc_results']['tc_uncertainty']
            tc_theoretical = self.theoretical_values['tc']
            
            if tc_theoretical is not None:
                tc_relative_error = abs(tc_measured - tc_theoretical) / tc_theoretical
                tc_accuracy = 1.0 - tc_relative_error
                
                validation['tc'] = {
                    'measured': tc_measured,
                    'theoretical': tc_theoretical,
                    'uncertainty': tc_uncertainty,
                    'relative_error': tc_relative_error,
                    'accuracy': tc_accuracy
                }
        
        # Overall validation summary
        accuracies = [v['accuracy'] for v in validation.values() if 'accuracy' in v]
        validation['summary'] = {
            'mean_accuracy': np.mean(accuracies) if accuracies else 0.0,
            'min_accuracy': np.min(accuracies) if accuracies else 0.0,
            'all_within_error': all(v.get('within_error_bars', False) for v in validation.values()),
            'universality_class': self.theoretical_values['universality_class']
        }
        
        return validation
    
    def _generate_analysis_plots(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analysis plots."""
        
        plots = {}
        
        # Plot 1: Critical exponent comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Beta exponent
        if 'beta_results' in results:
            beta_data = results['beta_results']
            axes[0, 0].errorbar(
                [0], [beta_data['beta_final']], 
                yerr=[beta_data['beta_uncertainty']],
                fmt='ro', label='Measured'
            )
            if self.theoretical_values['beta'] is not None:
                axes[0, 0].axhline(
                    self.theoretical_values['beta'], 
                    color='b', linestyle='--', label='Theoretical'
                )
            axes[0, 0].set_title('β Exponent')
            axes[0, 0].set_ylabel('β')
            axes[0, 0].legend()
        
        # Nu exponent
        if 'nu_results' in results:
            nu_data = results['nu_results']
            axes[0, 1].errorbar(
                [0], [nu_data['nu_final']], 
                yerr=[nu_data['nu_uncertainty']],
                fmt='ro', label='Measured'
            )
            if self.theoretical_values['nu'] is not None:
                axes[0, 1].axhline(
                    self.theoretical_values['nu'], 
                    color='b', linestyle='--', label='Theoretical'
                )
            axes[0, 1].set_title('ν Exponent')
            axes[0, 1].set_ylabel('ν')
            axes[0, 1].legend()
        
        # Gamma exponent
        if 'gamma_results' in results:
            gamma_data = results['gamma_results']
            axes[1, 0].errorbar(
                [0], [gamma_data['gamma_final']], 
                yerr=[gamma_data['gamma_uncertainty']],
                fmt='ro', label='Measured'
            )
            if self.theoretical_values['gamma'] is not None:
                axes[1, 0].axhline(
                    self.theoretical_values['gamma'], 
                    color='b', linestyle='--', label='Theoretical'
                )
            axes[1, 0].set_title('γ Exponent')
            axes[1, 0].set_ylabel('γ')
            axes[1, 0].legend()
        
        # Critical temperature
        if 'tc_results' in results:
            tc_data = results['tc_results']
            axes[1, 1].errorbar(
                [0], [tc_data['tc_consensus']], 
                yerr=[tc_data['tc_uncertainty']],
                fmt='ro', label='Measured'
            )
            if self.theoretical_values['tc'] is not None:
                axes[1, 1].axhline(
                    self.theoretical_values['tc'], 
                    color='b', linestyle='--', label='Theoretical'
                )
            axes[1, 1].set_title('Critical Temperature')
            axes[1, 1].set_ylabel('Tc')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plot_path = self.output_dir / 'critical_exponents_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plots['exponents_comparison'] = str(plot_path)
        plt.close()
        
        # Plot 2: Scaling behavior
        if 'vae_results' in results:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Get data from largest system size
            size_keys = list(results['vae_results']['latent_representations'].keys())
            largest_size_key = max(size_keys, key=lambda x: int(x.split('_')[1]))
            latent_data = results['vae_results']['latent_representations'][largest_size_key]
            
            temperatures = latent_data['temperatures']
            order_param = np.abs(latent_data['order_parameter'])
            
            # Magnetization vs temperature
            axes[0].plot(temperatures, order_param, 'bo-')
            axes[0].set_xlabel('Temperature')
            axes[0].set_ylabel('|Order Parameter|')
            axes[0].set_title('Order Parameter vs Temperature')
            if 'tc_results' in results:
                axes[0].axvline(results['tc_results']['tc_consensus'], color='r', linestyle='--', label='Tc')
                axes[0].legend()
            
            # Susceptibility vs temperature
            if 'tc_results' in results:
                size_result = results['tc_results']['size_results'][largest_size_key]
                axes[1].plot(temperatures, size_result['susceptibility'], 'go-')
                axes[1].set_xlabel('Temperature')
                axes[1].set_ylabel('Susceptibility')
                axes[1].set_title('Susceptibility vs Temperature')
                axes[1].axvline(results['tc_results']['tc_consensus'], color='r', linestyle='--', label='Tc')
                axes[1].legend()
            
            # Specific heat vs temperature
            if 'tc_results' in results:
                axes[2].plot(temperatures, size_result['specific_heat'], 'mo-')
                axes[2].set_xlabel('Temperature')
                axes[2].set_ylabel('Specific Heat')
                axes[2].set_title('Specific Heat vs Temperature')
                axes[2].axvline(results['tc_results']['tc_consensus'], color='r', linestyle='--', label='Tc')
                axes[2].legend()
            
            plt.tight_layout()
            plot_path = self.output_dir / 'scaling_behavior.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plots['scaling_behavior'] = str(plot_path)
            plt.close()
        
        return plots
    
    def _create_extraction_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive summary of extraction results."""
        
        summary = {
            'system_type': self.system_type,
            'universality_class': self.theoretical_values['universality_class'],
            'extraction_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Add extracted values
        for exponent in ['beta', 'nu', 'gamma']:
            if f'{exponent}_results' in results:
                data = results[f'{exponent}_results']
                summary[f'{exponent}_extracted'] = data[f'{exponent}_final']
                summary[f'{exponent}_uncertainty'] = data[f'{exponent}_uncertainty']
                summary[f'{exponent}_theoretical'] = self.theoretical_values[exponent]
        
        # Add critical temperature
        if 'tc_results' in results:
            tc_data = results['tc_results']
            summary['tc_extracted'] = tc_data['tc_consensus']
            summary['tc_uncertainty'] = tc_data['tc_uncertainty']
            summary['tc_theoretical'] = self.theoretical_values['tc']
        
        # Add validation summary
        if 'validation' in results:
            validation = results['validation']
            summary['mean_accuracy'] = validation['summary']['mean_accuracy']
            summary['min_accuracy'] = validation['summary']['min_accuracy']
            summary['all_within_error'] = validation['summary']['all_within_error']
        
        # Save summary to file
        summary_path = self.output_dir / 'extraction_summary.json'
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary
    
    # Helper methods for thermodynamic quantities
    def _compute_susceptibility(self, order_parameter: np.ndarray, temperatures: np.ndarray) -> np.ndarray:
        """Compute susceptibility from order parameter fluctuations."""
        # Group by temperature and compute variance
        unique_temps = np.unique(temperatures)
        susceptibility = np.zeros_like(unique_temps)
        
        for i, temp in enumerate(unique_temps):
            temp_mask = temperatures == temp
            if np.sum(temp_mask) > 1:
                susceptibility[i] = np.var(order_parameter[temp_mask]) / temp
        
        return susceptibility
    
    def _compute_specific_heat(self, energies: np.ndarray, temperatures: np.ndarray) -> np.ndarray:
        """Compute specific heat from energy fluctuations."""
        unique_temps = np.unique(temperatures)
        specific_heat = np.zeros_like(unique_temps)
        
        for i, temp in enumerate(unique_temps):
            temp_mask = temperatures == temp
            if np.sum(temp_mask) > 1:
                energy_var = np.var(energies[temp_mask])
                specific_heat[i] = energy_var / (temp**2)
        
        return specific_heat
    
    def _compute_binder_cumulant(self, order_parameter: np.ndarray) -> float:
        """Compute Binder cumulant for finite-size scaling."""
        m2 = np.mean(order_parameter**2)
        m4 = np.mean(order_parameter**4)
        
        if m2 > 0:
            return 1.0 - m4 / (3 * m2**2)
        else:
            return 0.0


def main():
    """Main function to run critical exponent extraction example."""
    parser = argparse.ArgumentParser(description='Critical Exponent Extraction Example')
    parser.add_argument(
        '--system',
        type=str,
        choices=['ising_3d', 'ising_2d', 'potts', 'xy'],
        default='ising_3d',
        help='Physics system to analyze'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/exponent_extraction_demo',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Initialize and run extraction demo
    demo = CriticalExponentExtractionDemo(
        system_type=args.system,
        output_dir=args.output_dir
    )
    
    try:
        results = demo.run_complete_extraction_demo()
        
        # Print summary
        print("\n" + "="*60)
        print("CRITICAL EXPONENT EXTRACTION COMPLETE")
        print("="*60)
        
        summary = results['summary']
        print(f"System: {summary['system_type']} ({summary['universality_class']})")
        
        if 'validation' in results:
            validation = results['validation']
            print(f"Mean Accuracy: {validation['summary']['mean_accuracy']:.1%}")
            print(f"All Within Error Bars: {'✓' if validation['summary']['all_within_error'] else '✗'}")
            
            for exponent in ['beta', 'nu', 'gamma']:
                if exponent in validation:
                    v = validation[exponent]
                    print(f"{exponent}: {v['measured']:.3f} ± {v['uncertainty']:.3f} "
                          f"(theory: {v['theoretical']:.3f}, accuracy: {v['accuracy']:.1%})")
        
        print(f"\nResults saved to: {demo.output_dir}")
        
    except Exception as e:
        logger.error(f"Extraction demo failed: {e}")
        raise


if __name__ == "__main__":
    main()