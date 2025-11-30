"""
Example: Long-Range Quantum Ising Model

Demonstrates:
1. Basic long-range Ising simulation
2. Coarse parameter space exploration
3. Anomaly detection
4. Phase diagram generation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.quantum.long_range_ising import (
    LongRangeIsing,
    LongRangeIsingParams,
    LongRangeIsingExplorer
)


def example_basic_simulation():
    """Example 1: Basic long-range Ising simulation."""
    print("=" * 60)
    print("Example 1: Basic Long-Range Ising Simulation")
    print("=" * 60)
    
    # Create model with α = 2.0 (long-range regime)
    params = LongRangeIsingParams(
        L=10,
        J0=1.0,
        alpha=2.0,
        h=0.5
    )
    
    model = LongRangeIsing(params)
    
    # Compute ground state
    print(f"\nComputing ground state for L={params.L}, α={params.alpha}, h={params.h}...")
    energy, state = model.compute_ground_state()
    print(f"Ground state energy: {energy:.6f}")
    
    # Compute observables
    observables = model.compute_observables(state)
    print(f"\nObservables:")
    print(f"  Magnetization (z): {observables['magnetization_z']:.6f}")
    print(f"  Magnetization (x): {observables['magnetization_x']:.6f}")
    print(f"  Correlation: {observables['correlation']:.6f}")
    
    # Compute entanglement
    entropy = model.compute_entanglement_entropy(state)
    print(f"  Entanglement entropy: {entropy:.6f}")


def example_alpha_dependence():
    """Example 2: Study α-dependence of phase transition."""
    print("\n" + "=" * 60)
    print("Example 2: α-Dependence of Phase Transition")
    print("=" * 60)
    
    L = 10
    alphas = [1.5, 2.0, 2.5, 3.0]
    h_values = np.linspace(0.0, 2.0, 20)
    
    plt.figure(figsize=(10, 6))
    
    for alpha in alphas:
        magnetizations = []
        
        print(f"\nScanning α = {alpha}...")
        for h in h_values:
            params = LongRangeIsingParams(L=L, alpha=alpha, h=h)
            model = LongRangeIsing(params)
            _, state = model.compute_ground_state()
            obs = model.compute_observables(state)
            magnetizations.append(obs['magnetization_z'])
        
        plt.plot(h_values, magnetizations, 'o-', label=f'α = {alpha}')
    
    plt.xlabel('Transverse Field h')
    plt.ylabel('Magnetization |⟨σᶻ⟩|')
    plt.title(f'Long-Range Ising: α-Dependence (L={L})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_dir = Path('results/long_range_ising')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'alpha_dependence.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to {output_dir / 'alpha_dependence.png'}")
    plt.close()


def example_coarse_exploration():
    """Example 3: Coarse parameter space exploration."""
    print("\n" + "=" * 60)
    print("Example 3: Coarse Parameter Space Exploration")
    print("=" * 60)
    
    # Create explorer
    explorer = LongRangeIsingExplorer(L=10, J0=1.0)
    
    # Perform coarse scan
    print("\nPerforming coarse scan...")
    results = explorer.coarse_scan(
        alpha_range=(1.5, 3.5),
        h_range=(0.0, 2.0),
        n_alpha=10,
        n_h=10
    )
    
    # Identify anomalies
    print("\nIdentifying anomalies...")
    anomalies = explorer.identify_anomalies(results, threshold=1.5)
    
    # Print anomaly summary
    print(f"\nFound {len(anomalies)} anomalies:")
    for i, anom in enumerate(anomalies[:5]):  # Show first 5
        print(f"  {i+1}. Type: {anom['type']}")
        print(f"     α = {anom['alpha']:.3f}, h = {anom['h']:.3f}")
        print(f"     Gradient: {anom['gradient']:.6f}")
    
    # Create phase diagram
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Magnetization
    im1 = axes[0, 0].contourf(results['hs'], results['alphas'],
                               results['magnetizations_z'], levels=20, cmap='RdBu_r')
    axes[0, 0].set_xlabel('Transverse Field h')
    axes[0, 0].set_ylabel('Power-law Exponent α')
    axes[0, 0].set_title('Magnetization |⟨σᶻ⟩|')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Entanglement entropy
    im2 = axes[0, 1].contourf(results['hs'], results['alphas'],
                               results['entropies'], levels=20, cmap='viridis')
    axes[0, 1].set_xlabel('Transverse Field h')
    axes[0, 1].set_ylabel('Power-law Exponent α')
    axes[0, 1].set_title('Entanglement Entropy')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Correlation
    im3 = axes[1, 0].contourf(results['hs'], results['alphas'],
                               results['correlations'], levels=20, cmap='plasma')
    axes[1, 0].set_xlabel('Transverse Field h')
    axes[1, 0].set_ylabel('Power-law Exponent α')
    axes[1, 0].set_title('Correlation ⟨σᶻᵢσᶻⱼ⟩')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Anomaly map
    axes[1, 1].contourf(results['hs'], results['alphas'],
                        results['magnetizations_z'], levels=20, cmap='gray', alpha=0.3)
    
    # Plot anomalies
    phase_trans = [a for a in anomalies if a['type'] == 'phase_transition']
    univ_cross = [a for a in anomalies if a['type'] == 'universality_crossover']
    
    if phase_trans:
        axes[1, 1].scatter([a['h'] for a in phase_trans],
                          [a['alpha'] for a in phase_trans],
                          c='red', s=100, marker='o', label='Phase Transition', alpha=0.7)
    if univ_cross:
        axes[1, 1].scatter([a['h'] for a in univ_cross],
                          [a['alpha'] for a in univ_cross],
                          c='blue', s=100, marker='s', label='Universality Crossover', alpha=0.7)
    
    axes[1, 1].set_xlabel('Transverse Field h')
    axes[1, 1].set_ylabel('Power-law Exponent α')
    axes[1, 1].set_title('Anomaly Map')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    output_dir = Path('results/long_range_ising')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'phase_diagram.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved phase diagram to {output_dir / 'phase_diagram.png'}")
    plt.close()
    
    # Save results
    np.savez(output_dir / 'scan_results.npz', **results)
    print(f"Saved scan results to {output_dir / 'scan_results.npz'}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("LONG-RANGE QUANTUM ISING MODEL EXAMPLES")
    print("=" * 60)
    
    example_basic_simulation()
    example_alpha_dependence()
    example_coarse_exploration()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
