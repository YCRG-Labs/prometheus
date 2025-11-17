#!/usr/bin/env python3
"""
Diagnostic Script to Identify Accuracy Issues

This script tests whether the problem is:
1. Data quality issues
2. VAE model quality issues  
3. Critical exponent extraction method issues
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ConvolutionalVAE3D
from src.analysis.latent_analysis import LatentRepresentation
from src.utils.config import PrometheusConfig
from src.utils.logging_utils import setup_logging, get_logger


def analyze_data_quality():
    """Analyze the quality of the raw data."""
    print("=" * 60)
    print("DATA QUALITY ANALYSIS")
    print("=" * 60)
    
    # 2D Data Analysis
    print("\n2D DATA:")
    data_2d = np.load('data/ising_dataset_20250831_145819.npz', allow_pickle=True)
    configs_2d = data_2d['spin_configurations']
    mags_2d = data_2d['magnetizations']
    metadata_2d = data_2d['metadata'].item()
    
    print(f"  Configurations: {configs_2d.shape}")
    print(f"  Magnetization range: [{np.min(mags_2d):.4f}, {np.max(mags_2d):.4f}]")
    print(f"  Temperature range: {metadata_2d['temp_range']}")
    print(f"  Theoretical Tc: {metadata_2d['critical_temp']}")
    
    # Check magnetization distribution
    n_temps = metadata_2d['n_temperatures']
    temp_min, temp_max = metadata_2d['temp_range']
    temperatures = np.linspace(temp_min, temp_max, n_temps)
    n_configs_per_temp = len(mags_2d) // n_temps
    temp_array = np.repeat(temperatures, n_configs_per_temp)
    
    # Analyze magnetization vs temperature
    unique_temps = np.unique(temp_array)
    mean_mags = []
    std_mags = []
    
    for temp in unique_temps:
        temp_mask = np.abs(temp_array - temp) < 0.01
        temp_mags = mags_2d[temp_mask]
        mean_mags.append(np.mean(np.abs(temp_mags)))
        std_mags.append(np.std(temp_mags))
    
    # Check if we see phase transition signature
    tc_theoretical = metadata_2d['critical_temp']
    tc_idx = np.argmin(np.abs(unique_temps - tc_theoretical))
    
    print(f"  Mean |magnetization| at Tc: {mean_mags[tc_idx]:.4f}")
    print(f"  Susceptibility at Tc: {std_mags[tc_idx]:.4f}")
    
    # Check if magnetization drops near Tc
    below_tc = unique_temps < tc_theoretical
    above_tc = unique_temps > tc_theoretical
    
    if np.any(below_tc) and np.any(above_tc):
        mag_below = np.mean([mean_mags[i] for i in range(len(unique_temps)) if below_tc[i]])
        mag_above = np.mean([mean_mags[i] for i in range(len(unique_temps)) if above_tc[i]])
        print(f"  Mean |mag| below Tc: {mag_below:.4f}")
        print(f"  Mean |mag| above Tc: {mag_above:.4f}")
        print(f"  Phase transition visible: {'YES' if mag_below > mag_above * 1.5 else 'NO'}")
    
    # 3D Data Analysis
    print("\n3D DATA:")
    with h5py.File('data/ising_3d_small.h5', 'r') as f:
        configs_3d = f['configurations'][:]
        mags_3d = f['magnetizations'][:]
        temps_3d = f['temperatures'][:]
    
    print(f"  Configurations: {configs_3d.shape}")
    print(f"  Magnetization range: [{np.min(mags_3d):.6f}, {np.max(mags_3d):.6f}]")
    print(f"  Temperature range: [{np.min(temps_3d):.3f}, {np.max(temps_3d):.3f}]")
    
    # Check 3D phase transition
    unique_temps_3d = np.unique(temps_3d)
    mean_mags_3d = []
    
    for temp in unique_temps_3d:
        temp_mask = np.abs(temps_3d - temp) < 0.01
        temp_mags = mags_3d[temp_mask]
        mean_mags_3d.append(np.mean(np.abs(temp_mags)))
    
    tc_3d_theoretical = 4.511
    print(f"  Theoretical 3D Tc: {tc_3d_theoretical}")
    
    # Check if 3D data covers critical region
    covers_tc = (np.min(temps_3d) <= tc_3d_theoretical <= np.max(temps_3d))
    print(f"  Data covers theoretical Tc: {'YES' if covers_tc else 'NO'}")
    
    if covers_tc:
        tc_idx_3d = np.argmin(np.abs(unique_temps_3d - tc_3d_theoretical))
        print(f"  Mean |magnetization| near Tc: {mean_mags_3d[tc_idx_3d]:.6f}")
    
    # Data quality verdict
    print(f"\n2D DATA QUALITY: {'GOOD' if np.max(np.abs(mags_2d)) > 0.3 else 'POOR'}")
    print(f"3D DATA QUALITY: {'GOOD' if np.max(np.abs(mags_3d)) > 0.01 else 'POOR'}")


def analyze_vae_model_quality():
    """Analyze the quality of the trained VAE model."""
    print("\n" + "=" * 60)
    print("VAE MODEL QUALITY ANALYSIS")
    print("=" * 60)
    
    # Load 3D VAE model
    config = PrometheusConfig()
    
    try:
        vae = ConvolutionalVAE3D(
            input_shape=(1, 32, 32, 32),
            latent_dim=2,
            encoder_channels=[32, 64, 128],
            decoder_channels=[128, 64, 32, 1],
            kernel_sizes=[3, 3, 3],
            beta=1.0
        )
        
        # Load trained weights
        checkpoint = torch.load('models/3d_vae/best_model.pth', map_location='cpu')
        if 'model_state_dict' in checkpoint:
            vae.load_state_dict(checkpoint['model_state_dict'])
        else:
            vae.load_state_dict(checkpoint)
        
        vae.eval()
        print("✓ 3D VAE model loaded successfully")
        
        # Test model on sample data
        with h5py.File('data/ising_3d_small.h5', 'r') as f:
            sample_configs = f['configurations'][:10]  # Take 10 samples
            sample_temps = f['temperatures'][:10]
            sample_mags = f['magnetizations'][:10]
        
        # Convert to tensor
        sample_tensor = torch.FloatTensor(sample_configs).unsqueeze(1)
        
        with torch.no_grad():
            # Encode
            mu, logvar = vae.encode(sample_tensor)
            z = vae.reparameterize(mu, logvar)
            
            # Decode
            reconstructed = vae.decode(z)
            
            # Calculate reconstruction error
            recon_error = torch.mean((sample_tensor - reconstructed)**2)
            
            print(f"  Latent space shape: {mu.shape}")
            print(f"  Latent z1 range: [{mu[:, 0].min():.4f}, {mu[:, 0].max():.4f}]")
            print(f"  Latent z2 range: [{mu[:, 1].min():.4f}, {mu[:, 1].max():.4f}]")
            print(f"  Reconstruction error: {recon_error:.4f}")
            
            # Check if latent space correlates with temperature
            z1_values = mu[:, 0].numpy()
            z2_values = mu[:, 1].numpy()
            
            temp_corr_z1 = np.corrcoef(sample_temps, z1_values)[0, 1]
            temp_corr_z2 = np.corrcoef(sample_temps, z2_values)[0, 1]
            
            print(f"  Temperature correlation with z1: {temp_corr_z1:.4f}")
            print(f"  Temperature correlation with z2: {temp_corr_z2:.4f}")
            
            # Check if latent space correlates with magnetization
            mag_corr_z1 = np.corrcoef(sample_mags, z1_values)[0, 1]
            mag_corr_z2 = np.corrcoef(sample_mags, z2_values)[0, 1]
            
            print(f"  Magnetization correlation with z1: {mag_corr_z1:.4f}")
            print(f"  Magnetization correlation with z2: {mag_corr_z2:.4f}")
            
            # Model quality verdict
            good_reconstruction = recon_error < 1.0
            good_correlation = max(abs(temp_corr_z1), abs(temp_corr_z2), abs(mag_corr_z1), abs(mag_corr_z2)) > 0.3
            
            print(f"\nVAE MODEL QUALITY: {'GOOD' if good_reconstruction and good_correlation else 'POOR'}")
            
            if not good_reconstruction:
                print("  Issue: Poor reconstruction quality")
            if not good_correlation:
                print("  Issue: Latent space doesn't correlate with physical properties")
    
    except Exception as e:
        print(f"✗ Failed to load/test VAE model: {e}")
        print("VAE MODEL QUALITY: UNAVAILABLE")


def test_critical_exponent_method():
    """Test the critical exponent extraction method with synthetic data."""
    print("\n" + "=" * 60)
    print("CRITICAL EXPONENT METHOD TEST")
    print("=" * 60)
    
    # Generate synthetic data with known exponents
    print("Generating synthetic data with known exponents...")
    
    # 2D Ising theoretical values
    beta_true = 0.125
    nu_true = 1.0
    tc_true = 2.269
    
    # Generate temperature array
    temps = np.linspace(1.5, tc_true - 0.01, 100)  # Only below Tc for beta
    
    # Generate synthetic magnetization: m = A * (Tc - T)^beta
    A = 1.0
    noise_level = 0.05
    
    reduced_temps = tc_true - temps
    synthetic_mags = A * (reduced_temps ** beta_true)
    
    # Add realistic noise
    noise = np.random.normal(0, noise_level * synthetic_mags, len(synthetic_mags))
    synthetic_mags += noise
    
    print(f"  True β exponent: {beta_true}")
    print(f"  Temperature range: [{np.min(temps):.3f}, {np.max(temps):.3f}]")
    print(f"  Magnetization range: [{np.min(synthetic_mags):.4f}, {np.max(synthetic_mags):.4f}]")
    
    # Test our fitting method
    from scipy.stats import linregress
    
    log_reduced = np.log(reduced_temps)
    log_mags = np.log(synthetic_mags)
    
    slope, intercept, r_value, p_value, std_err = linregress(log_reduced, log_mags)
    
    print(f"  Fitted β exponent: {slope:.4f} ± {std_err:.4f}")
    print(f"  R²: {r_value**2:.4f}")
    print(f"  Error: {abs(slope - beta_true):.4f}")
    print(f"  Relative error: {abs(slope - beta_true) / beta_true * 100:.1f}%")
    
    # Test correlation length fitting
    print("\nTesting correlation length fitting...")
    
    # Generate synthetic correlation length: xi = B * |T - Tc|^(-nu)
    temps_both_sides = np.linspace(tc_true - 0.5, tc_true + 0.5, 50)
    temps_both_sides = temps_both_sides[temps_both_sides != tc_true]  # Remove exact Tc
    
    B = 1.0
    reduced_temps_both = np.abs(temps_both_sides - tc_true)
    synthetic_corr_length = B * (reduced_temps_both ** (-nu_true))
    
    # Add noise
    noise_corr = np.random.normal(0, 0.1 * synthetic_corr_length, len(synthetic_corr_length))
    synthetic_corr_length += noise_corr
    
    # Fit
    log_reduced_both = np.log(reduced_temps_both)
    log_corr = np.log(synthetic_corr_length)
    
    slope_nu, intercept_nu, r_value_nu, p_value_nu, std_err_nu = linregress(log_reduced_both, log_corr)
    fitted_nu = -slope_nu  # Convert to positive nu
    
    print(f"  True ν exponent: {nu_true}")
    print(f"  Fitted ν exponent: {fitted_nu:.4f} ± {std_err_nu:.4f}")
    print(f"  R²: {r_value_nu**2:.4f}")
    print(f"  Error: {abs(fitted_nu - nu_true):.4f}")
    print(f"  Relative error: {abs(fitted_nu - nu_true) / nu_true * 100:.1f}%")
    
    # Method quality verdict
    beta_accurate = abs(slope - beta_true) / beta_true < 0.1  # Within 10%
    nu_accurate = abs(fitted_nu - nu_true) / nu_true < 0.1
    
    print(f"\nCRITICAL EXPONENT METHOD: {'GOOD' if beta_accurate and nu_accurate else 'NEEDS IMPROVEMENT'}")
    
    if not beta_accurate:
        print("  Issue: β exponent fitting not accurate enough")
    if not nu_accurate:
        print("  Issue: ν exponent fitting not accurate enough")


def main_diagnosis():
    """Run complete diagnostic analysis."""
    print("CRITICAL EXPONENT ACCURACY DIAGNOSTIC")
    print("=" * 60)
    print("This script will identify whether accuracy issues are due to:")
    print("1. Data quality problems")
    print("2. VAE model quality problems") 
    print("3. Critical exponent extraction method problems")
    print()
    
    # Run all diagnostic tests
    analyze_data_quality()
    analyze_vae_model_quality()
    test_critical_exponent_method()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    print("=" * 60)
    print()
    print("Based on the analysis above:")
    print()
    print("If DATA QUALITY is POOR:")
    print("  → Generate new data with better equilibration")
    print("  → Ensure temperature range covers critical region")
    print("  → Increase data density near Tc")
    print()
    print("If VAE MODEL QUALITY is POOR:")
    print("  → Retrain VAE with better hyperparameters")
    print("  → Use more training epochs")
    print("  → Adjust β parameter for better disentanglement")
    print()
    print("If CRITICAL EXPONENT METHOD needs improvement:")
    print("  → Implement finite-size scaling corrections")
    print("  → Add corrections to scaling terms")
    print("  → Use more sophisticated fitting procedures")
    print()
    print("MOST LIKELY ISSUE: VAE model not learning proper order parameter")
    print("RECOMMENDED FIX: Use trained VAE latent representations instead of raw magnetization")


if __name__ == "__main__":
    # Setup logging
    config = PrometheusConfig()
    setup_logging(config.logging)
    
    main_diagnosis()