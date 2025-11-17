#!/usr/bin/env python3
"""
Improved Critical Exponent Extraction Script

This script uses enhanced methods for more accurate critical exponent extraction
from both 2D and 3D Ising data with better critical temperature detection,
data preprocessing, and fitting algorithms.
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

from src.analysis.improved_critical_exponent_analyzer import (
    create_improved_critical_exponent_analyzer,
    ImprovedCriticalTemperatureDetector
)
from src.analysis.latent_analysis import LatentRepresentation
from src.models import ConvolutionalVAE, ConvolutionalVAE3D
from src.utils.config import PrometheusConfig, ConfigLoader
from src.utils.logging_utils import setup_logging, get_logger


def load_data_smart(data_path: str):
    """Smart data loading that handles both 2D and 3D data formats."""
    logger = get_logger(__name__)
    
    if data_path.endswith('.npz'):
        # 2D data format
        logger.info(f"Loading 2D data from {data_path}")
        data = np.load(data_path, allow_pickle=True)
        
        configurations = data['spin_configurations']
        magnetizations = data['magnetizations']
        energies = data['energies']
        metadata = data['metadata'].item()
        
        # Extract temperatures
        n_temps = metadata['n_temperatures']
        temp_min, temp_max = metadata['temp_range']
        temperatures = np.linspace(temp_min, temp_max, n_temps)
        n_configs_per_temp = configurations.shape[0] // n_temps
        temp_array = np.repeat(temperatures, n_configs_per_temp)
        
        return {
            'configurations': configurations,
            'magnetizations': magnetizations,
            'energies': energies,
            'temperatures': temp_array,
            'metadata': metadata,
            'is_3d': False
        }
        
    elif data_path.endswith('.h5'):
        # 3D data format
        logger.info(f"Loading 3D data from {data_path}")
        with h5py.File(data_path, 'r') as f:
            configurations = f['configurations'][:]
            magnetizations = f['magnetizations'][:]
            energies = f['energies'][:]
            temperatures = f['temperatures'][:]
            
            metadata = {}
            if 'metadata' in f:
                for key in f['metadata'].attrs:
                    metadata[key] = f['metadata'].attrs[key]
        
        return {
            'configurations': configurations,
            'magnetizations': magnetizations,
            'energies': energies,
            'temperatures': temperatures,
            'metadata': metadata,
            'is_3d': True
        }
    
    else:
        raise ValueError(f"Unsupported data format: {data_path}")


def extract_improved_latent_representations(data, model_path=None, config=None):
    """Extract latent representations using trained model or create proxy representations."""
    logger = get_logger(__name__)
    
    if model_path and Path(model_path).exists() and config:
        logger.info("Using trained VAE model for latent extraction")
        
        if data['is_3d']:
            # Load 3D VAE
            vae = ConvolutionalVAE3D(
                input_shape=(1, 32, 32, 32),
                latent_dim=config.vae.latent_dim,
                encoder_channels=[32, 64, 128],
                decoder_channels=[128, 64, 32, 1],
                kernel_sizes=[3, 3, 3],
                beta=config.vae.beta
            )
        else:
            # Load 2D VAE
            vae = ConvolutionalVAE(
                input_shape=(1, 32, 32),
                latent_dim=config.vae.latent_dim,
                encoder_channels=[32, 64, 128],
                decoder_channels=[128, 64, 32, 1],
                kernel_sizes=[3, 3, 3],
                beta=config.vae.beta
            )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            vae.load_state_dict(checkpoint['model_state_dict'])
        else:
            vae.load_state_dict(checkpoint)
        
        vae.eval()
        
        # Extract latent coordinates
        n_configs = data['configurations'].shape[0]
        z1_coords = np.zeros(n_configs)
        z2_coords = np.zeros(n_configs)
        reconstruction_errors = np.zeros(n_configs)
        
        batch_size = 50 if data['is_3d'] else 100
        
        with torch.no_grad():
            for i in range(0, n_configs, batch_size):
                end_idx = min(i + batch_size, n_configs)
                batch_configs = data['configurations'][i:end_idx]
                
                # Add channel dimension
                if data['is_3d']:
                    batch_tensor = torch.FloatTensor(batch_configs).unsqueeze(1)  # (batch, 1, D, H, W)
                else:
                    batch_tensor = torch.FloatTensor(batch_configs).unsqueeze(1)  # (batch, 1, H, W)
                
                # Encode
                mu, logvar = vae.encode(batch_tensor)
                z = vae.reparameterize(mu, logvar)
                
                # Decode for reconstruction error
                reconstructed = vae.decode(z)
                
                if data['is_3d']:
                    recon_error = torch.mean((batch_tensor - reconstructed)**2, dim=(1, 2, 3, 4))
                else:
                    recon_error = torch.mean((batch_tensor - reconstructed)**2, dim=(1, 2, 3))
                
                reconstruction_errors[i:end_idx] = recon_error.numpy()
                z1_coords[i:end_idx] = mu[:, 0].numpy()
                if mu.shape[1] > 1:
                    z2_coords[i:end_idx] = mu[:, 1].numpy()
        
    else:
        logger.warning("No trained model available, creating improved proxy representations")
        
        # Create improved proxy using multiple order parameters
        magnetizations = data['magnetizations']
        energies = data['energies']
        temperatures = data['temperatures']
        
        # Enhanced order parameter 1: Smoothed magnetization
        z1_coords = np.abs(magnetizations)
        
        # Enhanced order parameter 2: Energy-based susceptibility proxy
        # Bin by temperature and compute local susceptibility
        unique_temps = np.unique(temperatures)
        z2_coords = np.zeros_like(magnetizations)
        
        for temp in unique_temps:
            temp_mask = np.abs(temperatures - temp) < 0.01
            if np.sum(temp_mask) > 1:
                local_mag_var = np.var(magnetizations[temp_mask])
                z2_coords[temp_mask] = local_mag_var
        
        # Smooth z2 to reduce noise
        from scipy.signal import savgol_filter
        if len(z2_coords) > 5:
            z2_coords = savgol_filter(z2_coords, min(11, len(z2_coords)//2*2+1), 3)
        
        reconstruction_errors = np.zeros_like(z1_coords)
    
    # Create LatentRepresentation
    latent_repr = LatentRepresentation(
        z1=z1_coords,
        z2=z2_coords,
        temperatures=data['temperatures'],
        magnetizations=data['magnetizations'],
        energies=data['energies'],
        reconstruction_errors=reconstruction_errors,
        sample_indices=np.arange(len(z1_coords))
    )
    
    return latent_repr


def run_improved_analysis(data_path: str, model_path: str = None, system_type: str = None, 
                         output_dir: str = "results/improved_analysis"):
    """Run improved critical exponent analysis."""
    logger = get_logger(__name__)
    
    # Load data
    data = load_data_smart(data_path)
    
    # Determine system type if not specified
    if system_type is None:
        system_type = 'ising_3d' if data['is_3d'] else 'ising_2d'
    
    logger.info(f"Analyzing {system_type} system")
    
    # Load configuration
    config = PrometheusConfig()
    
    # Extract latent representations
    latent_repr = extract_improved_latent_representations(data, model_path, config)
    
    # Create improved analyzer
    analyzer = create_improved_critical_exponent_analyzer(
        system_type=system_type,
        bootstrap_samples=2000,
        random_seed=42
    )
    
    # Run improved analysis
    results = analyzer.analyze_with_improved_accuracy(
        latent_repr,
        auto_detect_tc=True
    )
    
    # Print results
    print_improved_results(results, system_type)
    
    # Save results
    save_improved_results(results, output_dir, system_type)
    
    return results


def print_improved_results(results: dict, system_type: str):
    """Print formatted results with accuracy metrics."""
    
    print("\n" + "=" * 70)
    print(f"IMPROVED {system_type.upper()} CRITICAL EXPONENT ANALYSIS")
    print("=" * 70)
    
    print(f"System Type: {system_type}")
    print(f"Critical Temperature: {results['critical_temperature']:.4f} (confidence: {results.get('tc_confidence', 0):.3f})")
    print()
    
    theoretical = results['theoretical_exponents']
    extracted = results['extracted_exponents']
    accuracy = results['accuracy_metrics']
    
    # β exponent
    if extracted.get('beta'):
        beta_data = extracted['beta']
        print("β EXPONENT (Order Parameter):")
        print(f"  Measured: {beta_data['value']:.4f} ± {beta_data['error']:.4f}")
        print(f"  Theoretical: {theoretical['beta']:.4f}")
        print(f"  Accuracy: {accuracy.get('beta_accuracy_percent', 0):.1f}%")
        print(f"  R²: {beta_data['r_squared']:.4f}")
        print(f"  Data Quality: {beta_data.get('data_quality', 0):.3f}")
        
        if beta_data.get('confidence_interval'):
            ci_lower, ci_upper = beta_data['confidence_interval']
            print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            # Check if theoretical value is in CI
            theoretical_in_ci = ci_lower <= theoretical['beta'] <= ci_upper
            print(f"  Theoretical in CI: {'✓' if theoretical_in_ci else '✗'}")
        print()
    
    # ν exponent
    if extracted.get('nu'):
        nu_data = extracted['nu']
        print("ν EXPONENT (Correlation Length):")
        print(f"  Measured: {nu_data['value']:.4f} ± {nu_data['error']:.4f}")
        print(f"  Theoretical: {theoretical['nu']:.4f}")
        print(f"  Accuracy: {accuracy.get('nu_accuracy_percent', 0):.1f}%")
        print(f"  R²: {nu_data['r_squared']:.4f}")
        print(f"  Data Quality: {nu_data.get('data_quality', 0):.3f}")
        
        if nu_data.get('confidence_interval'):
            ci_lower, ci_upper = nu_data['confidence_interval']
            print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            # Check if theoretical value is in CI
            theoretical_in_ci = ci_lower <= theoretical['nu'] <= ci_upper
            print(f"  Theoretical in CI: {'✓' if theoretical_in_ci else '✗'}")
        print()
    
    # Overall accuracy
    if 'overall_accuracy_percent' in accuracy:
        print("OVERALL PERFORMANCE:")
        print(f"  Combined Accuracy: {accuracy['overall_accuracy_percent']:.1f}%")
        
        # Performance rating
        overall_acc = accuracy['overall_accuracy_percent']
        if overall_acc >= 90:
            rating = "Excellent"
        elif overall_acc >= 80:
            rating = "Good"
        elif overall_acc >= 70:
            rating = "Fair"
        else:
            rating = "Needs Improvement"
        
        print(f"  Performance Rating: {rating}")
    
    print("=" * 70)


def save_improved_results(results: dict, output_dir: str, system_type: str):
    """Save improved results to files."""
    logger = get_logger(__name__)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy archive
    results_file = output_path / f"{system_type}_improved_results.npz"
    
    # Prepare data for saving
    save_data = {
        'system_type': system_type,
        'critical_temperature': results['critical_temperature'],
        'tc_confidence': results.get('tc_confidence', 0),
        'theoretical_exponents': results['theoretical_exponents'],
        'accuracy_metrics': results['accuracy_metrics']
    }
    
    # Add extracted exponents
    for exp_name in ['beta', 'nu']:
        if results['extracted_exponents'].get(exp_name):
            exp_data = results['extracted_exponents'][exp_name]
            save_data[f'{exp_name}_value'] = exp_data['value']
            save_data[f'{exp_name}_error'] = exp_data['error']
            save_data[f'{exp_name}_r_squared'] = exp_data['r_squared']
            
            if exp_data.get('confidence_interval'):
                save_data[f'{exp_name}_ci_lower'] = exp_data['confidence_interval'][0]
                save_data[f'{exp_name}_ci_upper'] = exp_data['confidence_interval'][1]
    
    np.savez(results_file, **save_data)
    logger.info(f"Results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description='Improved critical exponent extraction')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to data file (.npz for 2D, .h5 for 3D)')
    parser.add_argument('--model-path', type=str,
                       help='Path to trained VAE model (optional)')
    parser.add_argument('--system-type', type=str, choices=['ising_2d', 'ising_3d'],
                       help='System type (auto-detected if not specified)')
    parser.add_argument('--output-dir', type=str, default='results/improved_analysis',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Setup logging
    config = PrometheusConfig()
    setup_logging(config.logging)
    
    logger = get_logger(__name__)
    logger.info("Starting improved critical exponent analysis")
    
    try:
        results = run_improved_analysis(
            data_path=args.data_path,
            model_path=args.model_path,
            system_type=args.system_type,
            output_dir=args.output_dir
        )
        
        logger.info("Improved analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()