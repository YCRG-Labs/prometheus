#!/usr/bin/env python3
"""
Extract Critical Exponents from 3D Ising Data

This script analyzes 3D Ising data to extract critical exponents β and ν,
comparing the results to theoretical predictions for the 3D Ising universality class.
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import (
    LatentAnalyzer, 
    CriticalExponentAnalyzer,
    create_critical_exponent_analyzer
)
from src.models import ConvolutionalVAE3D
from src.utils.config import PrometheusConfig, ConfigLoader
from src.utils.logging_utils import setup_logging, get_logger


def load_3d_ising_data(data_path: str):
    """Load 3D Ising data from HDF5 file."""
    logger = get_logger(__name__)
    logger.info(f"Loading 3D Ising data from {data_path}")
    
    with h5py.File(data_path, 'r') as f:
        configurations = f['configurations'][:]
        magnetizations = f['magnetizations'][:]
        energies = f['energies'][:]
        temperatures = f['temperatures'][:]
        
        # Load metadata
        metadata = {}
        if 'metadata' in f:
            for key in f['metadata'].attrs:
                metadata[key] = f['metadata'].attrs[key]
    
    logger.info(f"Loaded 3D data: {configurations.shape[0]} configurations")
    logger.info(f"Configuration shape: {configurations.shape[1:]}")
    logger.info(f"Temperature range: [{np.min(temperatures):.3f}, {np.max(temperatures):.3f}]")
    
    return {
        'configurations': configurations,
        'magnetizations': magnetizations,
        'energies': energies,
        'temperatures': temperatures,
        'metadata': metadata
    }


def load_trained_3d_vae_model(model_path: str, config: PrometheusConfig):
    """Load a trained 3D VAE model."""
    logger = get_logger(__name__)
    logger.info(f"Loading trained 3D VAE model from {model_path}")
    
    # Create 3D VAE model with 3D input shape
    vae = ConvolutionalVAE3D(
        input_shape=(1, 32, 32, 32),  # 3D input shape
        latent_dim=config.vae.latent_dim,
        encoder_channels=[32, 64, 128],
        decoder_channels=[128, 64, 32, 1],
        kernel_sizes=[3, 3, 3],
        beta=config.vae.beta
    )
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        vae.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        vae.load_state_dict(checkpoint)
    
    vae.eval()
    logger.info("3D VAE model loaded successfully")
    
    return vae


def extract_3d_latent_representations(vae_model, configurations, temperatures, magnetizations, batch_size=50):
    """Extract latent representations from 3D configurations using trained VAE."""
    logger = get_logger(__name__)
    logger.info("Extracting latent representations from 3D configurations")
    
    n_configs = configurations.shape[0]
    latent_dim = vae_model.latent_dim
    
    # Prepare arrays for latent coordinates
    z1_coords = np.zeros(n_configs)
    z2_coords = np.zeros(n_configs)
    reconstruction_errors = np.zeros(n_configs)
    
    # Process in batches
    with torch.no_grad():
        for i in range(0, n_configs, batch_size):
            end_idx = min(i + batch_size, n_configs)
            batch_configs = configurations[i:end_idx]
            
            # Convert to tensor and add channel dimension
            batch_tensor = torch.FloatTensor(batch_configs).unsqueeze(1)  # (batch, 1, D, H, W)
            
            # Encode to latent space
            mu, logvar = vae_model.encode(batch_tensor)
            
            # Decode for reconstruction error
            z = vae_model.reparameterize(mu, logvar)
            reconstructed = vae_model.decode(z)
            
            # Calculate reconstruction error (MSE)
            recon_error = torch.mean((batch_tensor - reconstructed)**2, dim=(1, 2, 3, 4))
            reconstruction_errors[i:end_idx] = recon_error.numpy()
            
            # Use mean of latent distribution
            z1_coords[i:end_idx] = mu[:, 0].numpy()
            if latent_dim > 1:
                z2_coords[i:end_idx] = mu[:, 1].numpy()
    
    logger.info("3D latent representation extraction completed")
    
    # Create LatentRepresentation object
    from src.analysis.latent_analysis import LatentRepresentation
    
    latent_repr = LatentRepresentation(
        z1=z1_coords,
        z2=z2_coords,
        temperatures=temperatures,
        magnetizations=magnetizations,
        energies=np.zeros_like(z1_coords),  # Placeholder
        reconstruction_errors=reconstruction_errors,
        sample_indices=np.arange(len(z1_coords))
    )
    
    return latent_repr


def analyze_3d_critical_exponents(latent_repr, critical_temperature=4.511):
    """Analyze critical exponents for 3D Ising system."""
    logger = get_logger(__name__)
    logger.info("Analyzing 3D Ising critical exponents")
    
    # Create critical exponent analyzer
    analyzer = create_critical_exponent_analyzer(
        system_type='ising_3d',
        bootstrap_samples=1000,
        random_seed=42
    )
    
    # Extract critical exponents
    results = analyzer.analyze_system_exponents(
        latent_repr=latent_repr,
        critical_temperature=critical_temperature,
        system_type='ising_3d'
    )
    
    return results, analyzer


def create_3d_results_summary(results, theoretical_tc=4.511):
    """Create a summary of the 3D critical exponent analysis results."""
    logger = get_logger(__name__)
    
    summary = {
        'system_type': '3D Ising',
        'critical_temperature_used': results.critical_temperature,
        'theoretical_tc': theoretical_tc,
        'results': {}
    }
    
    # β exponent results
    if results.beta_result:
        summary['results']['beta'] = {
            'measured': results.beta_result.exponent,
            'error': results.beta_result.exponent_error,
            'theoretical': 0.326,
            'deviation_percent': abs(results.beta_result.exponent - 0.326) / 0.326 * 100,
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
            'theoretical': 0.630,
            'deviation_percent': abs(results.nu_result.exponent - 0.630) / 0.630 * 100,
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


def print_3d_results_summary(summary):
    """Print a formatted summary of the 3D results."""
    print("\n" + "=" * 60)
    print("3D ISING CRITICAL EXPONENT ANALYSIS RESULTS")
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


def save_3d_results_and_plots(results, analyzer, latent_repr, output_dir):
    """Save 3D results and create visualization plots."""
    logger = get_logger(__name__)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create power-law fit visualization
    try:
        fig = analyzer.visualize_power_law_fits(results, latent_repr)
        plot_path = output_path / "3d_ising_critical_exponents.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"3D power-law fit plots saved to {plot_path}")
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to create 3D power-law plots: {e}")
    
    # Save numerical results
    try:
        summary = create_3d_results_summary(results)
        
        # Save as numpy archive
        results_path = output_path / "3d_ising_exponent_results.npz"
        np.savez(
            results_path,
            summary=summary,
            beta_exponent=results.beta_result.exponent if results.beta_result else None,
            beta_error=results.beta_result.exponent_error if results.beta_result else None,
            nu_exponent=results.nu_result.exponent if results.nu_result else None,
            nu_error=results.nu_result.exponent_error if results.nu_result else None,
            critical_temperature=results.critical_temperature
        )
        logger.info(f"3D numerical results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Failed to save 3D numerical results: {e}")


def create_comparison_table(results_2d, results_3d):
    """Create comparison table between 2D and 3D results."""
    print("\n" + "=" * 80)
    print("2D vs 3D ISING CRITICAL EXPONENT COMPARISON")
    print("=" * 80)
    
    print(f"{'Property':<20} {'2D Measured':<15} {'2D Theory':<12} {'3D Measured':<15} {'3D Theory':<12}")
    print("-" * 80)
    
    # β exponent comparison
    if results_2d.beta_result and results_3d.beta_result:
        print(f"{'β exponent':<20} {results_2d.beta_result.exponent:<15.4f} {'0.125':<12} "
              f"{results_3d.beta_result.exponent:<15.4f} {'0.326':<12}")
    
    # ν exponent comparison
    if results_2d.nu_result and results_3d.nu_result:
        print(f"{'ν exponent':<20} {results_2d.nu_result.exponent:<15.4f} {'1.000':<12} "
              f"{results_3d.nu_result.exponent:<15.4f} {'0.630':<12}")
    
    print("-" * 80)
    
    # Accuracy comparison
    if results_2d.validation and results_3d.validation:
        beta_2d_acc = (1 - results_2d.validation.beta_deviation) * 100 if results_2d.validation.beta_deviation else 0
        beta_3d_acc = (1 - results_3d.validation.beta_deviation) * 100 if results_3d.validation.beta_deviation else 0
        
        print(f"{'β accuracy':<20} {beta_2d_acc:<15.1f}% {'':<12} {beta_3d_acc:<15.1f}% {'':<12}")
        
        if results_2d.validation.nu_deviation and results_3d.validation.nu_deviation:
            nu_2d_acc = (1 - results_2d.validation.nu_deviation) * 100
            nu_3d_acc = (1 - results_3d.validation.nu_deviation) * 100
            print(f"{'ν accuracy':<20} {nu_2d_acc:<15.1f}% {'':<12} {nu_3d_acc:<15.1f}% {'':<12}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Extract critical exponents from 3D Ising data')
    parser.add_argument('--data-path', type=str, default='data/ising_3d_small.h5',
                       help='Path to 3D Ising data file')
    parser.add_argument('--model-path', type=str, default='models/3d_vae/best_model.pth',
                       help='Path to trained 3D VAE model')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='results/3d_exponent_analysis',
                       help='Output directory for results')
    parser.add_argument('--critical-temp', type=float, default=4.511,
                       help='Critical temperature for 3D Ising model')
    parser.add_argument('--compare-2d', type=str, 
                       help='Path to 2D results for comparison')
    
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
    
    logger.info("Starting 3D Ising critical exponent extraction")
    
    try:
        # Load 3D Ising data
        data = load_3d_ising_data(args.data_path)
        
        # Check if we have a trained model to extract latent representations
        if Path(args.model_path).exists():
            # Load 3D VAE model and extract latent representations
            vae_model = load_trained_3d_vae_model(args.model_path, config)
            latent_repr = extract_3d_latent_representations(
                vae_model, 
                data['configurations'], 
                data['temperatures'], 
                data['magnetizations']
            )
        else:
            # Create mock latent representation using magnetization as order parameter
            logger.warning("No trained 3D VAE model found, using magnetization as order parameter")
            
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
                reconstruction_errors=np.zeros_like(z1),
                sample_indices=np.arange(len(z1))
            )
        
        # Analyze critical exponents
        results, analyzer = analyze_3d_critical_exponents(latent_repr, args.critical_temp)
        
        # Create and print summary
        summary = create_3d_results_summary(results, args.critical_temp)
        print_3d_results_summary(summary)
        
        # Save results and plots
        save_3d_results_and_plots(results, analyzer, latent_repr, args.output_dir)
        
        # Compare with 2D results if provided
        if args.compare_2d and Path(args.compare_2d).exists():
            try:
                # Load 2D results for comparison
                results_2d_data = np.load(args.compare_2d, allow_pickle=True)
                logger.info("Creating 2D vs 3D comparison")
                # Note: This would need the actual 2D results object for full comparison
                # For now, just indicate that comparison data was found
                print(f"\n2D comparison data loaded from: {args.compare_2d}")
            except Exception as e:
                logger.warning(f"Failed to load 2D comparison data: {e}")
        
        logger.info("3D Ising critical exponent analysis completed successfully")
        
    except Exception as e:
        logger.error(f"3D analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()