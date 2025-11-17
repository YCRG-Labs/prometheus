#!/usr/bin/env python3
"""
Extract Critical Exponents from 2D Ising Data

This script analyzes existing 2D Ising data to extract critical exponents β and ν,
comparing the results to theoretical predictions for the 2D Ising universality class.
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import (
    LatentAnalyzer, 
    CriticalExponentAnalyzer,
    create_critical_exponent_analyzer
)
from src.models import ConvolutionalVAE
from src.utils.config import PrometheusConfig, ConfigLoader
from src.utils.logging_utils import setup_logging, get_logger
import torch


def load_2d_ising_data(data_path: str):
    """Load 2D Ising data from numpy archive."""
    logger = get_logger(__name__)
    logger.info(f"Loading 2D Ising data from {data_path}")
    
    data = np.load(data_path, allow_pickle=True)
    
    configurations = data['spin_configurations']
    magnetizations = data['magnetizations']
    energies = data['energies']
    metadata = data['metadata'].item()
    
    logger.info(f"Loaded data: {configurations.shape[0]} configurations")
    logger.info(f"Lattice size: {metadata['lattice_size']}")
    logger.info(f"Temperature range: {metadata['temp_range']}")
    logger.info(f"Theoretical Tc: {metadata['critical_temp']}")
    
    # Extract temperatures from metadata
    n_temps = metadata['n_temperatures']
    temp_min, temp_max = metadata['temp_range']
    temperatures = np.linspace(temp_min, temp_max, n_temps)
    
    # Repeat temperatures for each configuration
    n_configs_per_temp = configurations.shape[0] // n_temps
    temp_array = np.repeat(temperatures, n_configs_per_temp)
    
    return {
        'configurations': configurations,
        'magnetizations': magnetizations,
        'energies': energies,
        'temperatures': temp_array,
        'metadata': metadata
    }


def load_trained_vae_model(model_path: str, config: PrometheusConfig):
    """Load a trained VAE model."""
    logger = get_logger(__name__)
    logger.info(f"Loading trained VAE model from {model_path}")
    
    # Create VAE model
    vae = ConvolutionalVAE(config.model)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        vae.load_state_dict(checkpoint['model_state_dict'])
    else:
        vae.load_state_dict(checkpoint)
    
    vae.eval()
    logger.info("VAE model loaded successfully")
    
    return vae


def extract_latent_representations(vae_model, configurations, temperatures, magnetizations, batch_size=100):
    """Extract latent representations from configurations using trained VAE."""
    logger = get_logger(__name__)
    logger.info("Extracting latent representations")
    
    n_configs = configurations.shape[0]
    latent_dim = vae_model.latent_dim
    
    # Prepare arrays for latent coordinates
    z1_coords = np.zeros(n_configs)
    z2_coords = np.zeros(n_configs)
    
    # Process in batches
    with torch.no_grad():
        for i in range(0, n_configs, batch_size):
            end_idx = min(i + batch_size, n_configs)
            batch_configs = configurations[i:end_idx]
            
            # Convert to tensor and add channel dimension
            batch_tensor = torch.FloatTensor(batch_configs).unsqueeze(1)  # (batch, 1, H, W)
            
            # Encode to latent space
            mu, logvar = vae_model.encode(batch_tensor)
            
            # Use mean of latent distribution
            z1_coords[i:end_idx] = mu[:, 0].numpy()
            if latent_dim > 1:
                z2_coords[i:end_idx] = mu[:, 1].numpy()
    
    logger.info("Latent representation extraction completed")
    
    # Create LatentRepresentation object
    from src.analysis.latent_analysis import LatentRepresentation
    
    latent_repr = LatentRepresentation(
        z1=z1_coords,
        z2=z2_coords,
        temperatures=temperatures,
        magnetizations=magnetizations,
        energies=np.zeros_like(z1_coords),  # Placeholder for energies
        reconstruction_errors=np.zeros_like(z1_coords),  # Placeholder for reconstruction errors
        sample_indices=np.arange(len(z1_coords))
    )
    
    return latent_repr


def analyze_2d_critical_exponents(latent_repr, critical_temperature=2.269):
    """Analyze critical exponents for 2D Ising system."""
    logger = get_logger(__name__)
    logger.info("Analyzing 2D Ising critical exponents")
    
    # Create critical exponent analyzer
    analyzer = create_critical_exponent_analyzer(
        system_type='ising_2d',
        bootstrap_samples=1000,
        random_seed=42
    )
    
    # Extract critical exponents
    results = analyzer.analyze_system_exponents(
        latent_repr=latent_repr,
        critical_temperature=critical_temperature,
        system_type='ising_2d'
    )
    
    return results, analyzer


def create_results_summary(results, theoretical_tc=2.269):
    """Create a summary of the critical exponent analysis results."""
    logger = get_logger(__name__)
    
    summary = {
        'system_type': '2D Ising',
        'critical_temperature_used': results.critical_temperature,
        'theoretical_tc': theoretical_tc,
        'results': {}
    }
    
    # β exponent results
    if results.beta_result:
        summary['results']['beta'] = {
            'measured': results.beta_result.exponent,
            'error': results.beta_result.exponent_error,
            'theoretical': 0.125,
            'deviation_percent': abs(results.beta_result.exponent - 0.125) / 0.125 * 100,
            'r_squared': results.beta_result.r_squared,
            'confidence_interval': (
                results.beta_result.confidence_interval.lower_bound,
                results.beta_result.confidence_interval.upper_bound
            ) if results.beta_result.confidence_interval else None
        }
    
    # ν exponent results
    if results.nu_result:
        summary['results']['nu'] = {
            'measured': results.nu_result.exponent,
            'error': results.nu_result.exponent_error,
            'theoretical': 1.0,
            'deviation_percent': abs(results.nu_result.exponent - 1.0) / 1.0 * 100,
            'r_squared': results.nu_result.r_squared,
            'confidence_interval': (
                results.nu_result.confidence_interval.lower_bound,
                results.nu_result.confidence_interval.upper_bound
            ) if results.nu_result.confidence_interval else None
        }
    
    # Validation results
    if results.validation:
        summary['validation'] = {
            'universality_class_match': results.validation.universality_class_match,
            'identified_class': results.universality_class.value
        }
    
    return summary


def print_results_summary(summary):
    """Print a formatted summary of the results."""
    print("\n" + "=" * 60)
    print("2D ISING CRITICAL EXPONENT ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"System: {summary['system_type']}")
    print(f"Critical Temperature Used: {summary['critical_temperature_used']:.3f}")
    print(f"Theoretical Tc: {summary['theoretical_tc']:.3f}")
    print()
    
    if 'beta' in summary['results']:
        beta_data = summary['results']['beta']
        print("β EXPONENT (Order Parameter):")
        print(f"  Measured: {beta_data['measured']:.4f} ± {beta_data['error']:.4f}")
        print(f"  Theoretical: {beta_data['theoretical']:.4f}")
        print(f"  Deviation: {beta_data['deviation_percent']:.1f}%")
        print(f"  R²: {beta_data['r_squared']:.4f}")
        if beta_data['confidence_interval']:
            ci_lower, ci_upper = beta_data['confidence_interval']
            print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print()
    
    if 'nu' in summary['results']:
        nu_data = summary['results']['nu']
        print("ν EXPONENT (Correlation Length):")
        print(f"  Measured: {nu_data['measured']:.4f} ± {nu_data['error']:.4f}")
        print(f"  Theoretical: {nu_data['theoretical']:.4f}")
        print(f"  Deviation: {nu_data['deviation_percent']:.1f}%")
        print(f"  R²: {nu_data['r_squared']:.4f}")
        if nu_data['confidence_interval']:
            ci_lower, ci_upper = nu_data['confidence_interval']
            print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print()
    
    if 'validation' in summary:
        val_data = summary['validation']
        print("VALIDATION:")
        print(f"  Universality Class Match: {'✓' if val_data['universality_class_match'] else '✗'}")
        print(f"  Identified Class: {val_data['identified_class']}")
    
    print("=" * 60)


def save_results_and_plots(results, analyzer, latent_repr, output_dir):
    """Save results and create visualization plots."""
    logger = get_logger(__name__)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create power-law fit visualization
    try:
        fig = analyzer.visualize_power_law_fits(results, latent_repr)
        plot_path = output_path / "2d_ising_critical_exponents.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Power-law fit plots saved to {plot_path}")
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to create power-law plots: {e}")
    
    # Save numerical results
    try:
        summary = create_results_summary(results)
        
        # Save as numpy archive
        results_path = output_path / "2d_ising_exponent_results.npz"
        np.savez(
            results_path,
            summary=summary,
            beta_exponent=results.beta_result.exponent if results.beta_result else None,
            beta_error=results.beta_result.exponent_error if results.beta_result else None,
            nu_exponent=results.nu_result.exponent if results.nu_result else None,
            nu_error=results.nu_result.exponent_error if results.nu_result else None,
            critical_temperature=results.critical_temperature
        )
        logger.info(f"Numerical results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Failed to save numerical results: {e}")


def main():
    parser = argparse.ArgumentParser(description='Extract critical exponents from 2D Ising data')
    parser.add_argument('--data-path', type=str, default='data/ising_dataset_20250831_145819.npz',
                       help='Path to 2D Ising data file')
    parser.add_argument('--model-path', type=str, help='Path to trained VAE model')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='results/2d_exponent_analysis',
                       help='Output directory for results')
    parser.add_argument('--critical-temp', type=float, default=2.269,
                       help='Critical temperature for 2D Ising model')
    
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader()
    if args.config:
        config = config_loader.load_config(args.config)
    else:
        config = PrometheusConfig()
    
    # Setup logging
    setup_logging(config.logging)
    logger = get_logger(__name__)
    
    logger.info("Starting 2D Ising critical exponent extraction")
    
    try:
        # Load 2D Ising data
        data = load_2d_ising_data(args.data_path)
        
        # Check if we have a trained model to extract latent representations
        if args.model_path and Path(args.model_path).exists():
            # Load VAE model and extract latent representations
            vae_model = load_trained_vae_model(args.model_path, config)
            latent_repr = extract_latent_representations(
                vae_model, 
                data['configurations'], 
                data['temperatures'], 
                data['magnetizations']
            )
        else:
            # Create mock latent representation using magnetization as order parameter
            logger.warning("No trained VAE model provided, using magnetization as order parameter")
            
            from src.analysis.latent_analysis import LatentRepresentation
            
            # Use magnetization and its square as latent dimensions
            z1 = data['magnetizations']
            z2 = data['magnetizations']**2
            
            latent_repr = LatentRepresentation(
                z1=z1,
                z2=z2,
                temperatures=data['temperatures'],
                magnetizations=data['magnetizations'],
                energies=data['energies'],
                reconstruction_errors=np.zeros_like(z1),  # No reconstruction errors for direct magnetization
                sample_indices=np.arange(len(z1))
            )
        
        # Analyze critical exponents
        results, analyzer = analyze_2d_critical_exponents(latent_repr, args.critical_temp)
        
        # Create and print summary
        summary = create_results_summary(results, args.critical_temp)
        print_results_summary(summary)
        
        # Save results and plots
        save_results_and_plots(results, analyzer, latent_repr, args.output_dir)
        
        logger.info("2D Ising critical exponent analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()