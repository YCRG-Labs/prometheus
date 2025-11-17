#!/usr/bin/env python3
"""
Task 5.4: Compare Enhanced VAE with Baseline VAE Performance

This script performs a comprehensive side-by-side comparison of:
- Baseline VAE (models/3d_vae/best_model.pth)
- Enhanced VAE (models/enhanced_3d_vae_final_v2/best_model.pth)

Comparison metrics:
1. Latent-magnetization correlation
2. Critical region sensitivity (variance ratio)
3. Reconstruction quality
4. Latent space structure
5. Temperature ordering
"""

import sys
import numpy as np
import torch
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import ConvolutionalVAE3D


def load_model(model_path, device):
    """Load VAE model from checkpoint."""
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model config from checkpoint if available
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        print(f"  Using saved config: {config}")
        model = ConvolutionalVAE3D(**config)
    else:
        # Fallback to default config
        print("  No saved config found, using default")
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
    
    return model, checkpoint


def load_data(data_path):
    """Load test data."""
    print(f"Loading data from: {data_path}")
    with h5py.File(data_path, 'r') as f:
        configurations = f['configurations'][:]
        magnetizations = f['magnetizations'][:]
        temperatures = f['temperatures'][:]
    
    # Normalize to [0, 1]
    configurations = (configurations + 1) / 2.0
    
    return configurations, magnetizations, temperatures


def compute_magnetization_correlation(model, configurations, magnetizations, device):
    """Compute latent-magnetization correlation."""
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
    p_values = []
    for dim in range(latent.shape[1]):
        corr, p_value = pearsonr(latent[:, dim], magnetizations)
        correlations.append(abs(corr))
        p_values.append(p_value)
    
    max_correlation = max(correlations)
    max_dim = np.argmax(correlations)
    
    return {
        'correlations': correlations,
        'p_values': p_values,
        'max_correlation': max_correlation,
        'max_dim': max_dim,
        'latent': latent
    }


def compute_critical_sensitivity(latent, temperatures, tc=4.5):
    """Compute critical region sensitivity (variance ratio)."""
    # Define critical region
    critical_mask = np.abs(temperatures - tc) < 0.5
    non_critical_mask = ~critical_mask
    
    # Compute variances
    variance_ratios = []
    for dim in range(latent.shape[1]):
        var_critical = np.var(latent[critical_mask, dim])
        var_non_critical = np.var(latent[non_critical_mask, dim])
        ratio = var_critical / (var_non_critical + 1e-10)
        variance_ratios.append(ratio)
    
    max_ratio = max(variance_ratios)
    max_dim = np.argmax(variance_ratios)
    
    return {
        'variance_ratios': variance_ratios,
        'max_ratio': max_ratio,
        'max_dim': max_dim,
        'n_critical': critical_mask.sum(),
        'n_non_critical': non_critical_mask.sum()
    }


def compute_reconstruction_quality(model, configurations, device):
    """Compute reconstruction quality metrics."""
    # Prepare data
    config_tensor = torch.FloatTensor(configurations * 2.0 - 1.0)
    config_tensor = config_tensor.unsqueeze(1)
    
    # Reconstruct
    with torch.no_grad():
        config_tensor = config_tensor.to(device)
        mu, logvar = model.encode(config_tensor)
        z = model.reparameterize(mu, logvar)
        recon = model.decode(z)
        
        # Compute MSE
        mse = torch.mean((config_tensor - recon)**2).item()
        
        # Compute per-sample MSE
        per_sample_mse = torch.mean((config_tensor - recon)**2, dim=(1,2,3,4)).cpu().numpy()
    
    return {
        'mse': mse,
        'per_sample_mse': per_sample_mse,
        'mean_mse': np.mean(per_sample_mse),
        'std_mse': np.std(per_sample_mse)
    }


def compute_temperature_ordering(latent, temperatures):
    """Compute temperature ordering quality."""
    # For each latent dimension, compute correlation with temperature
    temp_correlations = []
    for dim in range(latent.shape[1]):
        corr, _ = pearsonr(latent[:, dim], temperatures)
        temp_correlations.append(abs(corr))
    
    max_temp_corr = max(temp_correlations)
    max_dim = np.argmax(temp_correlations)
    
    return {
        'temp_correlations': temp_correlations,
        'max_temp_correlation': max_temp_corr,
        'max_dim': max_dim
    }


def compare_models(baseline_path, enhanced_path, data_path, output_dir):
    """Perform comprehensive model comparison."""
    
    print("="*80)
    print("TASK 5.4: BASELINE VS ENHANCED VAE COMPARISON")
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load models
    baseline_model, baseline_checkpoint = load_model(baseline_path, device)
    enhanced_model, enhanced_checkpoint = load_model(enhanced_path, device)
    print()
    
    # Load data
    configurations, magnetizations, temperatures = load_data(data_path)
    print(f"Loaded {len(configurations)} configurations")
    print(f"Temperature range: [{temperatures.min():.2f}, {temperatures.max():.2f}]")
    print(f"Magnetization range: [{magnetizations.min():.4f}, {magnetizations.max():.4f}]")
    print()
    
    # Comparison 1: Magnetization Correlation
    print("-"*80)
    print("1. MAGNETIZATION CORRELATION")
    print("-"*80)
    
    baseline_mag = compute_magnetization_correlation(baseline_model, configurations, magnetizations, device)
    enhanced_mag = compute_magnetization_correlation(enhanced_model, configurations, magnetizations, device)
    
    print(f"Baseline VAE:")
    print(f"  Max correlation: {baseline_mag['max_correlation']:.4f} (dim {baseline_mag['max_dim']})")
    print(f"  All correlations: {[f'{c:.4f}' for c in baseline_mag['correlations']]}")
    
    print(f"\nEnhanced VAE:")
    print(f"  Max correlation: {enhanced_mag['max_correlation']:.4f} (dim {enhanced_mag['max_dim']})")
    print(f"  All correlations: {[f'{c:.4f}' for c in enhanced_mag['correlations']]}")
    
    improvement_mag = enhanced_mag['max_correlation'] - baseline_mag['max_correlation']
    print(f"\n✓ Improvement: {improvement_mag:+.4f} ({improvement_mag/baseline_mag['max_correlation']*100:+.2f}%)")
    print()
    
    # Comparison 2: Critical Region Sensitivity
    print("-"*80)
    print("2. CRITICAL REGION SENSITIVITY")
    print("-"*80)
    
    baseline_crit = compute_critical_sensitivity(baseline_mag['latent'], temperatures)
    enhanced_crit = compute_critical_sensitivity(enhanced_mag['latent'], temperatures)
    
    print(f"Baseline VAE:")
    print(f"  Max variance ratio: {baseline_crit['max_ratio']:.4f} (dim {baseline_crit['max_dim']})")
    print(f"  All ratios: {[f'{r:.4f}' for r in baseline_crit['variance_ratios']]}")
    
    print(f"\nEnhanced VAE:")
    print(f"  Max variance ratio: {enhanced_crit['max_ratio']:.4f} (dim {enhanced_crit['max_dim']})")
    print(f"  All ratios: {[f'{r:.4f}' for r in enhanced_crit['variance_ratios']]}")
    
    improvement_crit = enhanced_crit['max_ratio'] - baseline_crit['max_ratio']
    print(f"\n✓ Change: {improvement_crit:+.4f}")
    print()
    
    # Comparison 3: Reconstruction Quality
    print("-"*80)
    print("3. RECONSTRUCTION QUALITY")
    print("-"*80)
    
    baseline_recon = compute_reconstruction_quality(baseline_model, configurations, device)
    enhanced_recon = compute_reconstruction_quality(enhanced_model, configurations, device)
    
    print(f"Baseline VAE:")
    print(f"  MSE: {baseline_recon['mse']:.6f}")
    print(f"  Mean MSE: {baseline_recon['mean_mse']:.6f} ± {baseline_recon['std_mse']:.6f}")
    
    print(f"\nEnhanced VAE:")
    print(f"  MSE: {enhanced_recon['mse']:.6f}")
    print(f"  Mean MSE: {enhanced_recon['mean_mse']:.6f} ± {enhanced_recon['std_mse']:.6f}")
    
    improvement_recon = baseline_recon['mse'] - enhanced_recon['mse']
    print(f"\n✓ Improvement: {improvement_recon:+.6f} (lower is better)")
    print()
    
    # Comparison 4: Temperature Ordering
    print("-"*80)
    print("4. TEMPERATURE ORDERING")
    print("-"*80)
    
    baseline_temp = compute_temperature_ordering(baseline_mag['latent'], temperatures)
    enhanced_temp = compute_temperature_ordering(enhanced_mag['latent'], temperatures)
    
    print(f"Baseline VAE:")
    print(f"  Max temp correlation: {baseline_temp['max_temp_correlation']:.4f} (dim {baseline_temp['max_dim']})")
    
    print(f"\nEnhanced VAE:")
    print(f"  Max temp correlation: {enhanced_temp['max_temp_correlation']:.4f} (dim {enhanced_temp['max_dim']})")
    
    improvement_temp = enhanced_temp['max_temp_correlation'] - baseline_temp['max_temp_correlation']
    print(f"\n✓ Improvement: {improvement_temp:+.4f}")
    print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nMetric                          Baseline    Enhanced    Improvement")
    print(f"-"*80)
    print(f"Magnetization Correlation       {baseline_mag['max_correlation']:.4f}      {enhanced_mag['max_correlation']:.4f}      {improvement_mag:+.4f}")
    print(f"Critical Variance Ratio         {baseline_crit['max_ratio']:.4f}      {enhanced_crit['max_ratio']:.4f}      {improvement_crit:+.4f}")
    print(f"Reconstruction MSE              {baseline_recon['mse']:.6f}  {enhanced_recon['mse']:.6f}  {improvement_recon:+.6f}")
    print(f"Temperature Correlation         {baseline_temp['max_temp_correlation']:.4f}      {enhanced_temp['max_temp_correlation']:.4f}      {improvement_temp:+.4f}")
    
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print("="*80)
    
    findings = []
    
    if improvement_mag > 0.1:
        findings.append(f"✓ MAJOR improvement in magnetization correlation (+{improvement_mag:.4f})")
    elif improvement_mag > 0.01:
        findings.append(f"✓ Moderate improvement in magnetization correlation (+{improvement_mag:.4f})")
    else:
        findings.append(f"• Minimal change in magnetization correlation ({improvement_mag:+.4f})")
    
    if improvement_crit > 0.5:
        findings.append(f"✓ Significant improvement in critical sensitivity (+{improvement_crit:.4f})")
    elif improvement_crit > 0:
        findings.append(f"• Slight improvement in critical sensitivity (+{improvement_crit:.4f})")
    else:
        findings.append(f"⚠ Decreased critical sensitivity ({improvement_crit:.4f})")
    
    if improvement_recon > 0:
        findings.append(f"✓ Better reconstruction quality (-{abs(improvement_recon):.6f} MSE)")
    else:
        findings.append(f"• Slightly worse reconstruction (+{abs(improvement_recon):.6f} MSE)")
    
    if improvement_temp > 0.1:
        findings.append(f"✓ Much better temperature ordering (+{improvement_temp:.4f})")
    elif improvement_temp > 0:
        findings.append(f"✓ Better temperature ordering (+{improvement_temp:.4f})")
    
    for finding in findings:
        print(finding)
    
    # Create visualizations
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Magnetization correlation comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Baseline latent vs magnetization
    for dim in range(2):
        axes[0, dim].scatter(magnetizations, baseline_mag['latent'][:, dim], 
                            alpha=0.5, s=10, color='blue', label='Baseline')
        axes[0, dim].set_xlabel('Magnetization', fontsize=11)
        axes[0, dim].set_ylabel(f'Latent Dim {dim}', fontsize=11)
        axes[0, dim].set_title(f'Baseline VAE - Dim {dim}\nCorr: {baseline_mag["correlations"][dim]:.4f}', 
                              fontsize=12, fontweight='bold')
        axes[0, dim].grid(True, alpha=0.3)
    
    # Enhanced latent vs magnetization
    for dim in range(2):
        axes[1, dim].scatter(magnetizations, enhanced_mag['latent'][:, dim], 
                            alpha=0.5, s=10, color='red', label='Enhanced')
        axes[1, dim].set_xlabel('Magnetization', fontsize=11)
        axes[1, dim].set_ylabel(f'Latent Dim {dim}', fontsize=11)
        axes[1, dim].set_title(f'Enhanced VAE - Dim {dim}\nCorr: {enhanced_mag["correlations"][dim]:.4f}', 
                              fontsize=12, fontweight='bold')
        axes[1, dim].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'comparison_magnetization_correlation.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path / 'comparison_magnetization_correlation.png'}")
    plt.close()
    
    # Plot 2: Latent space structure comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Baseline latent space
    scatter1 = axes[0].scatter(baseline_mag['latent'][:, 0], baseline_mag['latent'][:, 1],
                              c=temperatures, cmap='coolwarm', alpha=0.6, s=20)
    axes[0].set_xlabel('Latent Dimension 0', fontsize=11)
    axes[0].set_ylabel('Latent Dimension 1', fontsize=11)
    axes[0].set_title('Baseline VAE - Latent Space', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Temperature')
    
    # Enhanced latent space
    scatter2 = axes[1].scatter(enhanced_mag['latent'][:, 0], enhanced_mag['latent'][:, 1],
                              c=temperatures, cmap='coolwarm', alpha=0.6, s=20)
    axes[1].set_xlabel('Latent Dimension 0', fontsize=11)
    axes[1].set_ylabel('Latent Dimension 1', fontsize=11)
    axes[1].set_title('Enhanced VAE - Latent Space', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='Temperature')
    
    plt.tight_layout()
    plt.savefig(output_path / 'comparison_latent_space.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path / 'comparison_latent_space.png'}")
    plt.close()
    
    # Plot 3: Metrics comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = ['Mag Corr', 'Crit Ratio', 'Recon MSE', 'Temp Corr']
    baseline_values = [
        baseline_mag['max_correlation'],
        baseline_crit['max_ratio'],
        baseline_recon['mse'],
        baseline_temp['max_temp_correlation']
    ]
    enhanced_values = [
        enhanced_mag['max_correlation'],
        enhanced_crit['max_ratio'],
        enhanced_recon['mse'],
        enhanced_temp['max_temp_correlation']
    ]
    
    for idx, (metric, baseline_val, enhanced_val) in enumerate(zip(metrics, baseline_values, enhanced_values)):
        ax = axes[idx // 2, idx % 2]
        x = ['Baseline', 'Enhanced']
        y = [baseline_val, enhanced_val]
        colors = ['blue', 'red']
        
        bars = ax.bar(x, y, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, y):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'comparison_metrics_bars.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path / 'comparison_metrics_bars.png'}")
    plt.close()
    
    # Save comparison results to JSON
    results = {
        'baseline': {
            'model_path': str(baseline_path),
            'magnetization_correlation': float(baseline_mag['max_correlation']),
            'critical_variance_ratio': float(baseline_crit['max_ratio']),
            'reconstruction_mse': float(baseline_recon['mse']),
            'temperature_correlation': float(baseline_temp['max_temp_correlation'])
        },
        'enhanced': {
            'model_path': str(enhanced_path),
            'magnetization_correlation': float(enhanced_mag['max_correlation']),
            'critical_variance_ratio': float(enhanced_crit['max_ratio']),
            'reconstruction_mse': float(enhanced_recon['mse']),
            'temperature_correlation': float(enhanced_temp['max_temp_correlation'])
        },
        'improvements': {
            'magnetization_correlation': float(improvement_mag),
            'critical_variance_ratio': float(improvement_crit),
            'reconstruction_mse': float(improvement_recon),
            'temperature_correlation': float(improvement_temp)
        },
        'findings': findings
    }
    
    with open(output_path / 'comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved: {output_path / 'comparison_results.json'}")
    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print("="*80)
    
    return results


def main():
    """Main comparison function."""
    
    baseline_path = 'models/3d_vae/best_model.pth'
    enhanced_path = 'models/enhanced_3d_vae_final_v2/best_model.pth'
    data_path = 'data/vae_training_3d_ising.h5'
    output_dir = 'results/vae_comparison'
    
    results = compare_models(baseline_path, enhanced_path, data_path, output_dir)
    
    print(f"\nAll results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
