"""
Basic Usage Example for Prometheus

This script demonstrates the most basic usage of the Prometheus system
for extracting critical exponents from the 3D Ising model.

Usage:
    python examples/basic_usage_example.py
"""

import numpy as np
from pathlib import Path

# Import Prometheus modules
from src.utils.reproducibility import set_random_seed
from src.data.ising_simulator import IsingSimulator
from src.analysis.improved_critical_exponent_analyzer import ImprovedCriticalExponentAnalyzer


def main():
    """Run basic critical exponent extraction example."""
    
    print("=" * 70)
    print("Prometheus Basic Usage Example")
    print("=" * 70)
    print()
    
    # Step 1: Setup reproducibility
    print("Step 1: Setting random seed for reproducibility...")
    SEED = 42
    set_random_seed(SEED)
    print(f"✓ Random seed set to {SEED}")
    print()
    
    # Step 2: Generate Monte Carlo data
    print("Step 2: Generating Monte Carlo data...")
    print("  - Lattice size: 32x32x32")
    print("  - Temperature range: 3.5 to 5.5")
    print("  - Number of temperatures: 20")
    print("  - Samples per temperature: 50")
    print()
    
    simulator = IsingSimulator(lattice_size=(32, 32, 32), temperature=4.5)
    temperatures = np.linspace(3.5, 5.5, 20)
    
    all_magnetizations = []
    all_temps = []
    
    for i, temp in enumerate(temperatures):
        print(f"  Generating samples at T={temp:.2f}... ({i+1}/{len(temperatures)})", end='\r')
        simulator.temperature = temp
        
        # Generate samples
        samples = simulator.generate_samples(n_samples=50)
        
        # Compute magnetization for each sample
        magnetizations = np.abs(np.mean(samples, axis=(1, 2, 3)))
        
        all_magnetizations.extend(magnetizations)
        all_temps.extend([temp] * len(magnetizations))
    
    print()
    print(f"✓ Generated {len(all_magnetizations)} total samples")
    print()
    
    # Step 3: Prepare data for analysis
    print("Step 3: Preparing data for analysis...")
    
    # Create latent representation (using magnetization as simple order parameter)
    latent_data = np.column_stack([
        all_magnetizations,  # Order parameter dimension
        all_temps            # Temperature dimension
    ])
    
    temperatures_array = np.array(all_temps)
    
    print(f"✓ Latent representation shape: {latent_data.shape}")
    print()
    
    # Step 4: Extract critical exponents
    print("Step 4: Extracting critical exponents...")
    print()
    
    analyzer = ImprovedCriticalExponentAnalyzer()
    
    # Detect critical temperature
    print("  Detecting critical temperature...")
    tc_result = analyzer.detect_tc_stable(latent_data, temperatures_array)
    tc = tc_result['tc']
    tc_error = tc_result['tc_error']
    
    print(f"  ✓ Tc = {tc:.4f} ± {tc_error:.4f}")
    print(f"    (Theoretical: 4.511)")
    print(f"    Error: {abs(tc - 4.511)/4.511 * 100:.2f}%")
    print()
    
    # Extract beta exponent
    print("  Extracting β exponent...")
    beta_result = analyzer.extract_beta_stable(latent_data, temperatures_array, tc)
    beta = beta_result['beta']
    beta_error = beta_result['beta_error']
    
    print(f"  ✓ β = {beta:.4f} ± {beta_error:.4f}")
    print(f"    (Theoretical: 0.326)")
    print(f"    Error: {abs(beta - 0.326)/0.326 * 100:.2f}%")
    print()
    
    # Extract nu exponent
    print("  Extracting ν exponent...")
    nu_result = analyzer.extract_nu_stable(latent_data, temperatures_array, tc)
    nu = nu_result['nu']
    nu_error = nu_result['nu_error']
    
    print(f"  ✓ ν = {nu:.4f} ± {nu_error:.4f}")
    print(f"    (Theoretical: 0.630)")
    print(f"    Error: {abs(nu - 0.630)/0.630 * 100:.2f}%")
    print()
    
    # Step 5: Compute overall accuracy
    print("Step 5: Computing overall accuracy...")
    
    results = {
        'tc': tc,
        'tc_error': tc_error,
        'beta': beta,
        'beta_error': beta_error,
        'nu': nu,
        'nu_error': nu_error
    }
    
    accuracy = analyzer.compute_overall_accuracy(results)
    
    print(f"✓ Overall accuracy: {accuracy:.1f}%")
    print()
    
    # Step 6: Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"Critical Temperature:  Tc = {tc:.4f} ± {tc_error:.4f}")
    print(f"Beta Exponent:         β  = {beta:.4f} ± {beta_error:.4f}")
    print(f"Nu Exponent:           ν  = {nu:.4f} ± {nu_error:.4f}")
    print()
    print(f"Overall Accuracy:      {accuracy:.1f}%")
    print()
    
    if accuracy >= 70:
        print("✓ Achieved publication-quality accuracy (≥70%)")
    else:
        print(f"⚠ Below publication-quality threshold (need ≥70%, got {accuracy:.1f}%)")
    
    print()
    print("=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
