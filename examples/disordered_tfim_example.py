"""
Example demonstrating Disordered Transverse Field Ising Model (DTFIM).

Shows:
1. Clean limit verification (W=0, Δ=0 recovers standard TFIM)
2. Random transverse field disorder: hᵢ ~ Uniform[h-W, h+W]
3. Random coupling disorder: Jᵢ ~ Uniform[J-Δ, J+Δ]
4. Random longitudinal field: εᵢ (optional)
5. Disorder averaging with parallel computation
6. Comparison of different disorder types
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '.')

from src.quantum import (
    DTFIMParams,
    DisorderedTFIM,
    create_dtfim_uniform_disorder,
    create_dtfim_gaussian_disorder,
    DisorderType,
    ObservableCalculator,
    EntanglementCalculator,
)


def example_1_clean_limit_verification():
    """Example 1: Verify clean limit recovers standard TFIM."""
    print("=" * 70)
    print("Example 1: Clean Limit Verification")
    print("=" * 70)
    
    L = 10
    
    # Test at different field values
    h_values = [0.5, 1.0, 1.5, 2.0]
    
    print(f"\nVerifying clean limit for L={L} spins")
    print(f"Testing at h = {h_values}")
    
    all_match = True
    for h in h_values:
        params = DTFIMParams(
            L=L,
            h_mean=h,
            h_disorder=0.0,  # No disorder
            J_mean=1.0,
            J_disorder=0.0,  # No disorder
            periodic=True
        )
        
        dtfim = DisorderedTFIM(params)
        result = dtfim.verify_clean_limit(h_test=h)
        
        print(f"\n  h = {h}:")
        print(f"    Energy matches: {result['energy_matches']}")
        print(f"    E_DTFIM: {result['E_dtfim']:.8f}")
        print(f"    E_TFIM:  {result['E_tfim']:.8f}")
        print(f"    Difference: {result['difference']:.2e}")
        
        all_match = all_match and result['energy_matches']
    
    print(f"\n✓ Clean limit verification: {'PASSED' if all_match else 'FAILED'}")


def example_2_uniform_transverse_field_disorder():
    """Example 2: Random transverse field hᵢ ~ Uniform[h-W, h+W]."""
    print("\n" + "=" * 70)
    print("Example 2: Uniform Transverse Field Disorder")
    print("=" * 70)
    
    L = 12
    h_mean = 1.0
    W = 0.5  # Disorder strength
    
    print(f"\nSystem: L={L}, h_mean={h_mean}, W={W}")
    print(f"Transverse fields: hᵢ ~ Uniform[{h_mean-W}, {h_mean+W}]")
    
    # Create DTFIM with uniform disorder
    dtfim = create_dtfim_uniform_disorder(
        L=L,
        h_mean=h_mean,
        W=W,
        J_mean=1.0,
        Delta=0.0,  # No coupling disorder
        periodic=True
    )
    
    # Compute disorder-averaged ground state energy
    print("\nComputing disorder-averaged ground state energy...")
    result = dtfim.disorder_average_energy(n_realizations=200, parallel=True)
    
    print(f"\nResults (200 realizations):")
    print(f"  Mean energy: {result.mean:.6f} ± {result.stderr:.6f}")
    print(f"  Std dev: {result.std:.6f}")
    print(f"  95% CI: [{result.confidence_interval_95[0]:.6f}, "
          f"{result.confidence_interval_95[1]:.6f}]")
    
    # Compare with clean limit
    clean_dtfim = create_dtfim_uniform_disorder(L=L, h_mean=h_mean, W=0.0)
    E_clean, _ = clean_dtfim.compute_ground_state()
    
    print(f"\nComparison with clean limit:")
    print(f"  Clean energy: {E_clean:.6f}")
    print(f"  Disorder-averaged: {result.mean:.6f}")
    print(f"  Difference: {result.mean - E_clean:.6f}")


def example_3_coupling_disorder():
    """Example 3: Random coupling Jᵢ ~ Uniform[J-Δ, J+Δ]."""
    print("\n" + "=" * 70)
    print("Example 3: Coupling Disorder")
    print("=" * 70)
    
    L = 10
    J_mean = 1.0
    Delta = 0.3
    h = 1.0
    
    print(f"\nSystem: L={L}, J_mean={J_mean}, Δ={Delta}, h={h}")
    print(f"Couplings: Jᵢ ~ Uniform[{J_mean-Delta}, {J_mean+Delta}]")
    
    dtfim = create_dtfim_uniform_disorder(
        L=L,
        h_mean=h,
        W=0.0,  # No field disorder
        J_mean=J_mean,
        Delta=Delta,
        periodic=True
    )
    
    # Compute disorder-averaged energy
    result = dtfim.disorder_average_energy(n_realizations=200, parallel=True)
    
    print(f"\nDisorder-averaged ground state energy:")
    print(f"  {result.mean:.6f} ± {result.stderr:.6f}")


def example_4_combined_disorder():
    """Example 4: Combined field and coupling disorder."""
    print("\n" + "=" * 70)
    print("Example 4: Combined Field and Coupling Disorder")
    print("=" * 70)
    
    L = 12
    h_mean = 1.0
    W = 0.4
    J_mean = 1.0
    Delta = 0.2
    
    print(f"\nSystem: L={L}")
    print(f"  hᵢ ~ Uniform[{h_mean-W}, {h_mean+W}]")
    print(f"  Jᵢ ~ Uniform[{J_mean-Delta}, {J_mean+Delta}]")
    
    dtfim = create_dtfim_uniform_disorder(
        L=L,
        h_mean=h_mean,
        W=W,
        J_mean=J_mean,
        Delta=Delta,
        periodic=True
    )
    
    # Compute multiple observables
    print("\nComputing disorder-averaged observables...")
    
    def compute_energy(realization):
        E, _ = dtfim.compute_ground_state(realization)
        return E
    
    def compute_energy_per_site(realization):
        E, _ = dtfim.compute_ground_state(realization)
        return E / L
    
    def compute_avg_field(realization):
        return np.mean(realization.transverse_fields)
    
    def compute_avg_coupling(realization):
        return np.mean(realization.couplings)
    
    observable_fns = {
        'energy': compute_energy,
        'energy_per_site': compute_energy_per_site,
        'avg_field': compute_avg_field,
        'avg_coupling': compute_avg_coupling,
    }
    
    results = dtfim.disorder_average_observables(
        observable_fns,
        n_realizations=300,
        parallel=True
    )
    
    print(f"\nResults (300 realizations):")
    for name, result in results.items():
        print(f"  {name:20s}: {result.mean:.6f} ± {result.stderr:.6f}")


def example_5_gaussian_disorder():
    """Example 5: Gaussian disorder distribution."""
    print("\n" + "=" * 70)
    print("Example 5: Gaussian Disorder")
    print("=" * 70)
    
    L = 10
    h_mean = 1.0
    h_std = 0.3
    
    print(f"\nSystem: L={L}")
    print(f"  hᵢ ~ N({h_mean}, {h_std}²)")
    
    dtfim = create_dtfim_gaussian_disorder(
        L=L,
        h_mean=h_mean,
        h_std=h_std,
        J_mean=1.0,
        J_std=0.0,
        periodic=True
    )
    
    result = dtfim.disorder_average_energy(n_realizations=200, parallel=True)
    
    print(f"\nDisorder-averaged ground state energy:")
    print(f"  {result.mean:.6f} ± {result.stderr:.6f}")


def example_6_longitudinal_field():
    """Example 6: Optional longitudinal field disorder."""
    print("\n" + "=" * 70)
    print("Example 6: Longitudinal Field Disorder")
    print("=" * 70)
    
    L = 10
    
    print(f"\nSystem: L={L}")
    print(f"  hᵢ ~ Uniform[0.5, 1.5] (transverse)")
    print(f"  εᵢ ~ Uniform[-0.2, 0.2] (longitudinal)")
    
    params = DTFIMParams(
        L=L,
        h_mean=1.0,
        h_disorder=1.0,  # Width = 1.0 for range [0.5, 1.5]
        J_mean=1.0,
        J_disorder=0.0,
        epsilon_mean=0.0,
        epsilon_disorder=0.4,  # Width = 0.4 for range [-0.2, 0.2]
        disorder_type=DisorderType.BOX,
        periodic=True
    )
    
    dtfim = DisorderedTFIM(params)
    
    result = dtfim.disorder_average_energy(n_realizations=200, parallel=True)
    
    print(f"\nDisorder-averaged ground state energy:")
    print(f"  {result.mean:.6f} ± {result.stderr:.6f}")


def example_7_disorder_strength_scan():
    """Example 7: Scan disorder strength and observe effects."""
    print("\n" + "=" * 70)
    print("Example 7: Disorder Strength Scan")
    print("=" * 70)
    
    L = 10
    h_mean = 1.0
    W_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    print(f"\nScanning disorder strength W for L={L}, h_mean={h_mean}")
    print(f"W values: {W_values}")
    
    energies = []
    errors = []
    
    for W in W_values:
        dtfim = create_dtfim_uniform_disorder(
            L=L,
            h_mean=h_mean,
            W=W,
            J_mean=1.0,
            Delta=0.0,
            periodic=True
        )
        
        result = dtfim.disorder_average_energy(n_realizations=100, parallel=True)
        energies.append(result.mean)
        errors.append(result.stderr)
        
        print(f"  W={W:.1f}: E = {result.mean:.6f} ± {result.stderr:.6f}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(W_values, energies, yerr=errors, 
                marker='o', markersize=8, capsize=5, capthick=2,
                linewidth=2, label='Disorder-averaged energy')
    
    ax.set_xlabel('Disorder Strength W', fontsize=12)
    ax.set_ylabel('Ground State Energy', fontsize=12)
    ax.set_title(f'DTFIM: Energy vs Disorder Strength (L={L}, h={h_mean})', 
                 fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('results/dtfim_disorder_scan.png', dpi=150, bbox_inches='tight')
    print("\n  Plot saved to: results/dtfim_disorder_scan.png")
    plt.close()


def example_8_phase_diagram_preview():
    """Example 8: Preview of phase diagram exploration."""
    print("\n" + "=" * 70)
    print("Example 8: Phase Diagram Preview")
    print("=" * 70)
    
    L = 10
    h_values = np.linspace(0.5, 1.5, 5)
    W_values = np.linspace(0.0, 0.8, 5)
    
    print(f"\nExploring (h, W) parameter space for L={L}")
    print(f"  h ∈ [{h_values[0]}, {h_values[-1]}]")
    print(f"  W ∈ [{W_values[0]}, {W_values[-1]}]")
    print(f"  Grid: {len(h_values)} × {len(W_values)} = {len(h_values)*len(W_values)} points")
    
    energies = np.zeros((len(W_values), len(h_values)))
    
    print("\nComputing disorder-averaged energies...")
    for i, W in enumerate(W_values):
        for j, h in enumerate(h_values):
            dtfim = create_dtfim_uniform_disorder(
                L=L,
                h_mean=h,
                W=W,
                J_mean=1.0,
                Delta=0.0,
                periodic=True
            )
            
            result = dtfim.disorder_average_energy(n_realizations=50, parallel=True)
            energies[i, j] = result.mean
            
            print(f"  h={h:.2f}, W={W:.2f}: E={result.mean:.6f}")
    
    # Create phase diagram
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.contourf(h_values, W_values, energies, levels=20, cmap='viridis')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Ground State Energy', fontsize=12)
    
    ax.set_xlabel('Mean Transverse Field h', fontsize=12)
    ax.set_ylabel('Disorder Strength W', fontsize=12)
    ax.set_title(f'DTFIM Phase Diagram Preview (L={L})', fontsize=14)
    
    # Mark clean TFIM critical point
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, 
               alpha=0.7, label='Clean TFIM critical point')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('results/dtfim_phase_diagram_preview.png', dpi=150, bbox_inches='tight')
    print("\n  Phase diagram saved to: results/dtfim_phase_diagram_preview.png")
    plt.close()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("DISORDERED TFIM DEMONSTRATION")
    print("=" * 70)
    
    example_1_clean_limit_verification()
    example_2_uniform_transverse_field_disorder()
    example_3_coupling_disorder()
    example_4_combined_disorder()
    example_5_gaussian_disorder()
    example_6_longitudinal_field()
    example_7_disorder_strength_scan()
    example_8_phase_diagram_preview()
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
