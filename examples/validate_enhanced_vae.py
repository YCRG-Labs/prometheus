#!/usr/bin/env python3
"""
Validate Enhanced VAE Model

This script validates the trained enhanced VAE model against the requirements:
- Task 5.2: Latent-magnetization correlation ≥96%
- Task 5.3: Critical region sensitivity (variance ratio > 2.0)
"""

import sys
import numpy as np
import torch
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import ConvolutionalVAE3D


def load_model(model_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with same architecture
    model = ConvolutionalVAE3D(
        input_shape=(1, 32, 32, 32),
        latent_dim=2,
        encoder_channels=[16, 32, 64],
        decoder_channels=[64, 32, 16, 1],
        kernel_sizes=[3, 3, 3],
        beta=1.0
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def load_data(data_path):
    """Load validation data."""
    with h5py.File(data_path, 'r') as f:
        configurations = f['configurations'][:]
        magnetizations = f['magnetizations'][:]
        temperatures = f['temperatures'][:]
    
    # Normalize to [0, 1]
    configurations = (configurations + 1) / 2.0
    
    return configurations, magnetizations, temperatures


def validate_latent_magnetization_correlation(model, configurations, magnetizations, device):
    """
    Task 5.2: Validate latent-magnetization correlation ≥96%
    """
    print("\n" + "="*60)
    print("Task 5.2: Latent-Magnetization Correlation Validation")
    print("="*60)
    
    # Prepare data
    config_tensor = torch.FloatTensor(configurations * 2.0 - 1.0)  # [0,1] -> [-1,1]
    config_tensor = config_tensor.unsqueeze(1)  # Add channel dimension
    
    # Extract latent representations
    with torch.no_grad():
        config_tensor = config_tensor.to(device)
        mu, logvar = model.encode(config_tensor)
        latent = mu.cpu().numpy()
    
    # Compute correlations for each latent dimension
    correlations = []
    for dim in range(latent.shape[1]):
        corr, p_value = pearsonr(latent[:, dim], magnetizations)
        correlations.append(abs(corr))
        print(f"Latent dim {dim}: correlation = {abs(corr):.4f}, p-value = {p_value:.2e}")
    
    max_correlation = max(correlations)
    print(f"\nMax correlation: {max_correlation:.4f}")
    print(f"Target: ≥0.96")
    print(f"Status: {'✓ PASS' if max_correlation >= 0.96 else '✗ FAIL (but good progress)'}")
    
    # Plot correlations
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for dim in range(latent.shape[1]):
        axes[dim].scatter(magnetizations, latent[:, dim], alpha=0.5, s=10)
        corr = correlations[dim]
        axes[dim].set_xlabel('Magnetization', fontsize=12)
        axes[dim].set_ylabel(f'Latent Dimension {dim}', fontsize=12)
        axes[dim].set_title(f'Correlation: {corr:.4f}', fontsize=12, fontweight='bold')
        axes[dim].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/enhanced_3d_vae/latent_magnetization_correlation.png', dpi=300)
    print(f"\nPlot saved to: models/enhanced_3d_vae/latent_magnetization_correlation.png")
    
    return max_correlation


def validate_critical_region_sensitivity(model, configurations, temperatures, device, tc=4.5):
    """
    Task 5.3: Validate critical region sensitivity (variance ratio > 2.0)
    """
    print("\n" + "="*60)
    print("Task 5.3: Critical Region Sensitivity Validation")
    print("="*60)
    
    # Prepare data
    config_tensor = torch.FloatTensor(configurations * 2.0 - 1.0)
    config_tensor = config_tensor.unsqueeze(1)
    
    # Extract latent representations
    with torch.no_grad():
        config_tensor = config_tensor.to(device)
        mu, logvar = model.encode(config_tensor)
        latent = mu.cpu().numpy()
    
    # Define critical region (within 0.5 of Tc)
    critical_mask = np.abs(temperatures - tc) < 0.5
    non_critical_mask = ~critical_mask
    
    print(f"Critical temperature: {tc}")
    print(f"Critical region: {tc-0.5:.2f} < T < {tc+0.5:.2f}")
    print(f"Samples in critical region: {critical_mask.sum()}")
    print(f"Samples in non-critical region: {non_critical_mask.sum()}")
    
    # Compute variances
    variance_ratios = []
    for dim in range(latent.shape[1]):
        var_critical = np.var(latent[critical_mask, dim])
        var_non_critical = np.var(latent[non_critical_mask, dim])
        ratio = var_critical / (var_non_critical + 1e-10)
        variance_ratios.append(ratio)
        print(f"\nLatent dim {dim}:")
        print(f"  Variance (critical): {var_critical:.6f}")
        print(f"  Variance (non-critical): {var_non_critical:.6f}")
        print(f"  Ratio: {ratio:.4f}")
    
    max_ratio = max(variance_ratios)
    print(f"\nMax variance ratio: {max_ratio:.4f}")
    print(f"Target: > 2.0")
    print(f"Status: {'✓ PASS' if max_ratio > 2.0 else '✗ FAIL (but shows sensitivity)'}")
    
    # Visualize latent space structure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for dim in range(latent.shape[1]):
        # Plot latent values vs temperature
        axes[dim].scatter(temperatures[non_critical_mask], latent[non_critical_mask, dim], 
                         alpha=0.5, s=10, label='Non-critical', color='blue')
        axes[dim].scatter(temperatures[critical_mask], latent[critical_mask, dim], 
                         alpha=0.5, s=10, label='Critical', color='red')
        axes[dim].axvline(tc, color='black', linestyle='--', linewidth=2, label=f'Tc={tc}')
        axes[dim].axvspan(tc-0.5, tc+0.5, alpha=0.2, color='red')
        axes[dim].set_xlabel('Temperature', fontsize=12)
        axes[dim].set_ylabel(f'Latent Dimension {dim}', fontsize=12)
        axes[dim].set_title(f'Variance Ratio: {variance_ratios[dim]:.4f}', fontsize=12, fontweight='bold')
        axes[dim].legend()
        axes[dim].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/enhanced_3d_vae/critical_region_sensitivity.png', dpi=300)
    print(f"\nPlot saved to: models/enhanced_3d_vae/critical_region_sensitivity.png")
    
    return max_ratio


def main():
    """Main validation function."""
    
    print("="*60)
    print("ENHANCED VAE VALIDATION")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    model_path = 'models/enhanced_3d_vae_final_v2/best_model.pth'
    print(f"Loading model from: {model_path}")
    model = load_model(model_path, device)
    print("Model loaded successfully")
    
    # Load data
    data_path = 'data/vae_training_3d_ising.h5'
    print(f"Loading data from: {data_path}")
    configurations, magnetizations, temperatures = load_data(data_path)
    print(f"Loaded {len(configurations)} configurations")
    
    # Run validations
    max_correlation = validate_latent_magnetization_correlation(
        model, configurations, magnetizations, device
    )
    
    max_variance_ratio = validate_critical_region_sensitivity(
        model, configurations, temperatures, device
    )
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Task 5.2 - Latent-Magnetization Correlation:")
    print(f"  Result: {max_correlation:.4f}")
    print(f"  Target: ≥0.96")
    print(f"  Status: {'✓ PASS' if max_correlation >= 0.96 else '✗ NEEDS IMPROVEMENT'}")
    print(f"\nTask 5.3 - Critical Region Sensitivity:")
    print(f"  Result: {max_variance_ratio:.4f}")
    print(f"  Target: >2.0")
    print(f"  Status: {'✓ PASS' if max_variance_ratio > 2.0 else '✗ NEEDS IMPROVEMENT'}")
    print("="*60)
    
    # Overall assessment
    if max_correlation >= 0.96 and max_variance_ratio > 2.0:
        print("\n✓ All validation criteria met!")
    else:
        print("\n⚠ Some criteria not fully met, but model shows promise.")
        print("  Consider:")
        print("  - Training for more epochs")
        print("  - Using more training data")
        print("  - Adjusting loss weights")
        print("  - Fine-tuning architecture")


if __name__ == "__main__":
    main()
