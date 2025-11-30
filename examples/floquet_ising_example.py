"""
Example: Floquet (Periodically Driven) Ising Model

Demonstrates:
1. Basic Floquet Ising model setup
2. Quasienergy spectrum computation
3. Time crystal signature detection
4. Coarse parameter space exploration
5. Anomaly identification
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.quantum.floquet_ising import (
    FloquetIsing,
    FloquetIsingParams,
    FloquetIsingExplorer,
    create_standard_floquet_ising,
)


def example_1_basic_floquet():
    """Example 1: Basic Floquet Ising model."""
    print("=" * 60)
    print("Example 1: Basic Floquet Ising Model")
    print("=" * 60)
    
    # Create Floquet Ising model
    params = FloquetIsingParams(
        L=8,
        J=1.0,
        h1=1.5,  # Strong field in first half
        h2=0.5,  # Weak field in second half
        T=1.0,   # Driving period
        periodic=True
    )
    
    model = FloquetIsing(params)
    
    print(f"\nSystem parameters:")
    print(f"  L = {params.L} sites")
    print(f"  h1 = {params.h1} (first half-period)")
    print(f"  h2 = {params.h2} (second half-period)")
    print(f"  T = {params.T} (driving period)")
    print(f"  J = {params.J} (Ising coupling)")
    
    # Compute quasienergy spectrum
    print("\nComputing quasienergy spectrum...")
    quasienergies, floquet_states = model.compute_quasienergy_spectrum()
    
    print(f"  Number of quasienergy levels: {len(quasienergies)}")
    print(f"  Quasienergy range: [{quasienergies[0]:.4f}, {quasienergies[-1]:.4f}]")
    
    # Compute quasienergy gap
    gap = model.compute_quasienergy_gap()
    print(f"  Minimum quasienergy gap: {gap:.6f}")
    
    print("\n✓ Basic Floquet model computed successfully")


def example_2_time_evolution():
    """Example 2: Time evolution and stroboscopic measurements."""
    print("\n" + "=" * 60)
    print("Example 2: Time Evolution")
    print("=" * 60)
    
    # Create model
    model = create_standard_floquet_ising(L=8)
    
    # Initial state: all spins up in z-direction
    dim = 2 ** 8
    initial_state = np.zeros(dim)
    initial_state[0] = 1.0
    
    print(f"\nInitial state: |↑↑...↑⟩")
    
    # Build magnetization operator
    from scipy import sparse
    mag_z_op = sparse.csr_matrix((dim, dim), dtype=np.float64)
    for i in range(8):
        sigma_z = model.builder.build_sigma_z(i)
        mag_z_op = mag_z_op + sigma_z
    mag_z_op = mag_z_op / 8
    
    # Compute stroboscopic magnetization
    n_periods = 20
    print(f"Evolving for {n_periods} periods...")
    
    mag_trajectory = model.compute_stroboscopic_observable(
        mag_z_op,
        initial_state,
        n_periods
    )
    
    print(f"\nMagnetization trajectory:")
    print(f"  Initial: {mag_trajectory[0]:.4f}")
    print(f"  After 10 periods: {mag_trajectory[10]:.4f}")
    print(f"  After 20 periods: {mag_trajectory[20]:.4f}")
    
    # Check for oscillations
    oscillation_amplitude = np.max(mag_trajectory) - np.min(mag_trajectory)
    print(f"  Oscillation amplitude: {oscillation_amplitude:.4f}")
    
    print("\n✓ Time evolution computed successfully")


def example_3_time_crystal_detection():
    """Example 3: Time crystal signature detection."""
    print("\n" + "=" * 60)
    print("Example 3: Time Crystal Signature Detection")
    print("=" * 60)
    
    explorer = FloquetIsingExplorer(L=8, J=1.0)
    
    # Test a few parameter sets
    test_params = [
        FloquetIsingParams(L=8, h1=1.5, h2=0.5, T=1.0),
        FloquetIsingParams(L=8, h1=2.0, h2=0.3, T=0.8),
        FloquetIsingParams(L=8, h1=1.0, h2=1.0, T=1.0),  # No driving
    ]
    
    for i, params in enumerate(test_params):
        print(f"\nTest {i+1}: h1={params.h1}, h2={params.h2}, T={params.T}")
        
        signatures = explorer.detect_time_crystal_signatures(
            params,
            n_periods=15
        )
        
        print(f"  Period doubling strength: {signatures['period_doubling_strength']:.4f}")
        print(f"  Final amplitude: {signatures['final_amplitude']:.4f}")
        print(f"  Time crystal signature: {signatures['has_time_crystal_signature']}")
    
    print("\n✓ Time crystal detection completed")


def example_4_coarse_scan():
    """Example 4: Coarse parameter space scan."""
    print("\n" + "=" * 60)
    print("Example 4: Coarse Parameter Space Scan")
    print("=" * 60)
    
    # Use small system for speed
    explorer = FloquetIsingExplorer(L=8, J=1.0)
    
    print("\nScanning parameter space...")
    print("  h1 ∈ [0.5, 2.0]")
    print("  h2 ∈ [0.5, 2.0]")
    print("  T ∈ [0.5, 2.0]")
    
    results = explorer.coarse_scan(
        h1_range=(0.5, 2.0),
        h2_range=(0.5, 2.0),
        T_range=(0.5, 2.0),
        n_h1=4,
        n_h2=4,
        n_T=3
    )
    
    gaps = results['quasienergy_gaps']
    
    print(f"\nScan results:")
    print(f"  Total points scanned: {gaps.size}")
    print(f"  Mean quasienergy gap: {np.mean(gaps):.6f}")
    print(f"  Min quasienergy gap: {np.min(gaps):.6f}")
    print(f"  Max quasienergy gap: {np.max(gaps):.6f}")
    
    # Identify anomalies
    print("\nIdentifying anomalies...")
    anomalies = explorer.identify_anomalies(results, threshold=1.5)
    
    print(f"  Found {len(anomalies)} anomalies")
    
    if anomalies:
        print("\nTop 3 anomalies:")
        # Sort by significance
        sorted_anomalies = sorted(
            anomalies,
            key=lambda x: x.get('significance', 0),
            reverse=True
        )
        
        for i, anomaly in enumerate(sorted_anomalies[:3]):
            print(f"\n  Anomaly {i+1}:")
            print(f"    Type: {anomaly['type']}")
            print(f"    h1 = {anomaly['h1']:.3f}")
            print(f"    h2 = {anomaly['h2']:.3f}")
            print(f"    T = {anomaly['T']:.3f}")
            print(f"    Gap = {anomaly['gap']:.6f}")
            if 'significance' in anomaly:
                print(f"    Significance: {anomaly['significance']:.2f}σ")
    
    print("\n✓ Coarse scan completed")


def example_5_visualization():
    """Example 5: Visualize quasienergy spectrum and time evolution."""
    print("\n" + "=" * 60)
    print("Example 5: Visualization")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("results/floquet_ising_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    model = create_standard_floquet_ising(L=8)
    
    # 1. Quasienergy spectrum
    print("\nGenerating quasienergy spectrum plot...")
    quasienergies, _ = model.compute_quasienergy_spectrum()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Quasienergy levels
    axes[0].scatter(range(len(quasienergies)), quasienergies, s=10, alpha=0.6)
    axes[0].set_xlabel('Level index')
    axes[0].set_ylabel('Quasienergy')
    axes[0].set_title('Quasienergy Spectrum')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 2: Gap distribution
    gaps = np.diff(quasienergies)
    axes[1].hist(gaps, bins=30, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Quasienergy gap')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Gap Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    spectrum_file = output_dir / "quasienergy_spectrum.png"
    plt.savefig(spectrum_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {spectrum_file}")
    plt.close()
    
    # 2. Time evolution
    print("\nGenerating time evolution plot...")
    
    # Initial state
    dim = 2 ** 8
    initial_state = np.zeros(dim)
    initial_state[0] = 1.0
    
    # Build magnetization operator
    from scipy import sparse
    mag_z_op = sparse.csr_matrix((dim, dim), dtype=np.float64)
    for i in range(8):
        sigma_z = model.builder.build_sigma_z(i)
        mag_z_op = mag_z_op + sigma_z
    mag_z_op = mag_z_op / 8
    
    # Compute trajectory
    n_periods = 30
    mag_trajectory = model.compute_stroboscopic_observable(
        mag_z_op,
        initial_state,
        n_periods
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    times = np.arange(n_periods + 1) * model.params.T
    ax.plot(times, mag_trajectory, 'o-', markersize=4, linewidth=1.5)
    ax.set_xlabel('Time (in units of T)')
    ax.set_ylabel('Magnetization ⟨σᶻ⟩')
    ax.set_title('Stroboscopic Magnetization Evolution')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    evolution_file = output_dir / "time_evolution.png"
    plt.savefig(evolution_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {evolution_file}")
    plt.close()
    
    print("\n✓ Visualizations generated")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("FLOQUET ISING MODEL EXAMPLES")
    print("=" * 60)
    
    example_1_basic_floquet()
    example_2_time_evolution()
    example_3_time_crystal_detection()
    example_4_coarse_scan()
    example_5_visualization()
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
