"""
Example usage of the Latent Space Analysis Framework

This script demonstrates how to use the LatentAnalyzer and OrderParameterAnalyzer
to analyze trained VAE models and discover order parameters from latent representations.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis import LatentAnalyzer, OrderParameterAnalyzer
from src.models.vae import ConvolutionalVAE
from src.data.preprocessing import IsingDataset
from torch.utils.data import DataLoader


def create_example_vae():
    """Create an example VAE model for demonstration."""
    vae = ConvolutionalVAE(
        input_shape=(1, 32, 32),
        latent_dim=2,
        encoder_channels=[32, 64, 128],
        decoder_channels=[128, 64, 32, 1]
    )
    vae.eval()
    return vae


def create_synthetic_data():
    """Create synthetic data that mimics Ising model behavior."""
    n_samples = 1000
    
    # Create temperature range
    temperatures = np.random.uniform(1.5, 3.0, n_samples)
    
    # Create synthetic latent coordinates with order parameter structure
    z1 = np.zeros(n_samples)  # This will be our order parameter
    z2 = np.random.randn(n_samples)  # Random dimension
    
    magnetizations = np.zeros(n_samples)
    energies = np.random.uniform(-2, 0, n_samples)
    
    critical_temp = 2.269
    
    for i, temp in enumerate(temperatures):
        if temp < critical_temp:
            # Ordered phase: non-zero magnetization and z1
            mag = np.random.choice([-1, 1]) * np.sqrt(1 - (temp / critical_temp) ** 2)
            mag += 0.1 * np.random.randn()  # Add noise
            z1[i] = 0.8 * abs(mag) + 0.1 * np.random.randn()
        else:
            # Disordered phase: near-zero magnetization and z1
            mag = 0.1 * np.random.randn()
            z1[i] = 0.1 * np.random.randn()
        
        magnetizations[i] = mag
    
    return z1, z2, temperatures, magnetizations, energies


def main():
    """Main example function."""
    print("Latent Space Analysis Framework Example")
    print("=" * 50)
    
    # Create example VAE model
    print("1. Creating example VAE model...")
    vae = create_example_vae()
    
    # Initialize analyzers
    print("2. Initializing analyzers...")
    latent_analyzer = LatentAnalyzer(vae)
    order_param_analyzer = OrderParameterAnalyzer(critical_temperature=2.269)
    
    # Create synthetic latent representation
    print("3. Creating synthetic latent representation...")
    z1, z2, temperatures, magnetizations, energies = create_synthetic_data()
    
    from src.analysis.latent_analysis import LatentRepresentation
    latent_repr = LatentRepresentation(
        z1=z1,
        z2=z2,
        temperatures=temperatures,
        magnetizations=magnetizations,
        energies=energies,
        reconstruction_errors=np.random.uniform(0, 0.1, len(z1)),
        sample_indices=np.arange(len(z1))
    )
    
    print(f"   Created representation with {latent_repr.n_samples} samples")
    print(f"   Temperature range: {latent_repr.get_statistics()['temperature_range']}")
    
    # Analyze latent dimensions
    print("4. Analyzing latent dimensions...")
    dim_analysis = latent_analyzer.analyze_latent_dimensions(latent_repr)
    
    for dim_name, analysis in dim_analysis.items():
        print(f"   {dim_name}:")
        print(f"     Temperature correlation: {analysis['correlations']['temperature']:.3f}")
        print(f"     Magnetization correlation: {analysis['correlations']['abs_magnetization']:.3f}")
    
    # Discover order parameters
    print("5. Discovering order parameters...")
    candidates = order_param_analyzer.discover_order_parameters(latent_repr)
    
    print(f"   Found {len(candidates)} candidates:")
    for i, candidate in enumerate(candidates):
        print(f"     {i+1}. {candidate.latent_dimension}: "
              f"confidence={candidate.confidence_score:.3f}, "
              f"mag_corr={candidate.correlation_with_magnetization.correlation_coefficient:.3f}")
    
    # Compare with theoretical predictions
    print("6. Comparing with theoretical predictions...")
    comparison = order_param_analyzer.compare_with_theoretical(candidates, latent_repr)
    
    print(f"   Best candidate: {comparison['best_candidate']}")
    print(f"   Correlation with theory: {comparison['correlation_with_theory']:.3f}")
    print(f"   RMSE with theory: {comparison['rmse_with_theory']:.3f}")
    
    # Generate comprehensive report
    print("7. Generating analysis report...")
    report = order_param_analyzer.generate_analysis_report(candidates, latent_repr)
    
    print(f"   Report summary:")
    print(f"     Number of candidates: {report['summary']['n_candidates']}")
    print(f"     Temperature range: {report['summary']['temperature_range']}")
    print(f"     Recommendations: {len(report['recommendations'])}")
    
    for rec in report['recommendations']:
        print(f"       - {rec}")
    
    # Create visualizations
    print("8. Creating visualizations...")
    
    # Latent space visualization
    fig1 = latent_analyzer.visualize_latent_space(latent_repr, color_by='temperature')
    fig1.savefig('latent_space_temperature.png', dpi=150, bbox_inches='tight')
    print("   Saved: latent_space_temperature.png")
    
    # Phase separation visualization
    fig2 = latent_analyzer.visualize_phase_separation(latent_repr)
    fig2.savefig('phase_separation.png', dpi=150, bbox_inches='tight')
    print("   Saved: phase_separation.png")
    
    # Order parameter discovery visualization
    fig3 = order_param_analyzer.visualize_order_parameter_discovery(candidates, latent_repr)
    fig3.savefig('order_parameter_discovery.png', dpi=150, bbox_inches='tight')
    print("   Saved: order_parameter_discovery.png")
    
    plt.close('all')  # Close all figures to free memory
    
    print("\nExample completed successfully!")
    print("Check the generated PNG files for visualizations.")


if __name__ == "__main__":
    main()