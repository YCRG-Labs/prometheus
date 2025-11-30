"""
Example: Coarse exploration of Disordered TFIM parameter space.

Demonstrates Task 6: Systematic scan of (h, W) space to identify
anomalous regions and generate preliminary phase diagrams.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from src.research.dtfim_coarse_explorer import DTFIMCoarseExplorer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Run DTFIM coarse exploration."""
    
    print("=" * 80)
    print("DTFIM Coarse Exploration - Task 6")
    print("=" * 80)
    
    # Create explorer
    explorer = DTFIMCoarseExplorer(
        L=12,  # System size
        n_disorder_realizations=50,  # Disorder realizations per point
        J_mean=1.0,
        periodic=True,
        random_seed=42
    )
    
    # Scan parameter space
    print("\nScanning parameter space...")
    print("h ∈ [0, 2], W ∈ [0, 2]")
    print("Grid: 10x10 = 100 points")
    print("This will take several minutes...\n")
    
    result = explorer.scan_parameter_space(
        h_range=(0.0, 2.0),
        W_range=(0.0, 2.0),
        n_points=100,  # 10x10 grid
        parallel=True,
        max_workers=4
    )
    
    # Save results
    output_dir = Path("results/task6_dtfim_coarse_exploration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result.save(output_dir / "exploration_results.json")
    print(f"\nResults saved to {output_dir / 'exploration_results.json'}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPLORATION SUMMARY")
    print("=" * 80)
    print(f"Total points scanned: {len(result.scan_points)}")
    print(f"Anomalies detected: {len(result.anomalies)}")
    print(f"System size: L={result.metadata['L']}")
    print(f"Disorder realizations per point: {result.metadata['n_disorder_realizations']}")
    
    # Print top anomalies
    print("\n" + "-" * 80)
    print("TOP 10 ANOMALIES (by severity)")
    print("-" * 80)
    for i, anomaly in enumerate(result.anomalies[:10], 1):
        print(f"\n{i}. {anomaly.anomaly_type.upper()}")
        print(f"   Location: h={anomaly.h:.3f}, W={anomaly.W:.3f}")
        print(f"   Severity: {anomaly.severity:.3f}")
        print(f"   Description: {anomaly.description}")
        print(f"   Observables:")
        for key, val in anomaly.observables.items():
            print(f"     {key}: {val:.4f}")
    
    # Generate phase diagrams
    print("\n" + "-" * 80)
    print("GENERATING PHASE DIAGRAMS")
    print("-" * 80)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('DTFIM Coarse Exploration - Phase Diagrams', fontsize=16)
    
    # Magnetization
    ax = axes[0, 0]
    im = ax.contourf(
        result.h_values,
        result.W_values,
        result.phase_diagram_data['magnetization_z'],
        levels=20,
        cmap='viridis'
    )
    ax.set_xlabel('Transverse field h')
    ax.set_ylabel('Disorder strength W')
    ax.set_title('Magnetization |⟨σᶻ⟩|')
    plt.colorbar(im, ax=ax)
    
    # Susceptibility
    ax = axes[0, 1]
    chi_data = result.phase_diagram_data['susceptibility_z']
    chi_data_clipped = np.clip(chi_data, 0, np.percentile(chi_data, 95))
    im = ax.contourf(
        result.h_values,
        result.W_values,
        chi_data_clipped,
        levels=20,
        cmap='hot'
    )
    ax.set_xlabel('Transverse field h')
    ax.set_ylabel('Disorder strength W')
    ax.set_title('Susceptibility χᶻ')
    plt.colorbar(im, ax=ax)
    
    # Entanglement
    ax = axes[0, 2]
    im = ax.contourf(
        result.h_values,
        result.W_values,
        result.phase_diagram_data['entanglement_entropy'],
        levels=20,
        cmap='plasma'
    )
    ax.set_xlabel('Transverse field h')
    ax.set_ylabel('Disorder strength W')
    ax.set_title('Entanglement Entropy S')
    plt.colorbar(im, ax=ax)
    
    # Correlation length
    ax = axes[1, 0]
    corr_data = result.phase_diagram_data['correlation_length']
    corr_data_clipped = np.clip(corr_data, 0, result.metadata['L'])
    im = ax.contourf(
        result.h_values,
        result.W_values,
        corr_data_clipped,
        levels=20,
        cmap='coolwarm'
    )
    ax.set_xlabel('Transverse field h')
    ax.set_ylabel('Disorder strength W')
    ax.set_title('Correlation Length ξ')
    plt.colorbar(im, ax=ax)
    
    # Energy
    ax = axes[1, 1]
    im = ax.contourf(
        result.h_values,
        result.W_values,
        result.phase_diagram_data['energy'],
        levels=20,
        cmap='viridis'
    )
    ax.set_xlabel('Transverse field h')
    ax.set_ylabel('Disorder strength W')
    ax.set_title('Ground State Energy')
    plt.colorbar(im, ax=ax)
    
    # Anomaly map
    ax = axes[1, 2]
    ax.scatter(
        result.h_values,
        np.zeros_like(result.h_values),
        alpha=0
    )  # Dummy for axis setup
    
    # Plot anomalies by type
    anomaly_types = {}
    for anomaly in result.anomalies:
        if anomaly.anomaly_type not in anomaly_types:
            anomaly_types[anomaly.anomaly_type] = []
        anomaly_types[anomaly.anomaly_type].append(anomaly)
    
    colors = ['red', 'orange', 'yellow', 'cyan']
    for i, (atype, anomalies) in enumerate(anomaly_types.items()):
        h_vals = [a.h for a in anomalies]
        W_vals = [a.W for a in anomalies]
        severities = [a.severity * 100 for a in anomalies]
        ax.scatter(
            h_vals,
            W_vals,
            s=severities,
            c=colors[i % len(colors)],
            alpha=0.6,
            label=atype.replace('_', ' ').title()
        )
    
    ax.set_xlabel('Transverse field h')
    ax.set_ylabel('Disorder strength W')
    ax.set_title('Anomaly Map')
    ax.legend(fontsize=8)
    ax.set_xlim(result.h_values[0], result.h_values[-1])
    ax.set_ylim(result.W_values[0], result.W_values[-1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase_diagrams.png', dpi=300, bbox_inches='tight')
    print(f"Phase diagrams saved to {output_dir / 'phase_diagrams.png'}")
    
    # Generate detailed anomaly plots
    print("\nGenerating anomaly analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('DTFIM Anomaly Analysis', fontsize=16)
    
    # Susceptibility vs h for different W
    ax = axes[0, 0]
    W_samples = [0.0, 0.5, 1.0, 1.5, 2.0]
    for W_val in W_samples:
        points = [p for p in result.scan_points if abs(p.W - W_val) < 0.15]
        if points:
            points.sort(key=lambda p: p.h)
            h_vals = [p.h for p in points]
            chi_vals = [p.susceptibility_z for p in points]
            ax.plot(h_vals, chi_vals, 'o-', label=f'W={W_val:.1f}')
    ax.set_xlabel('Transverse field h')
    ax.set_ylabel('Susceptibility χᶻ')
    ax.set_title('Susceptibility vs h (different W)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Entanglement vs h for different W
    ax = axes[0, 1]
    for W_val in W_samples:
        points = [p for p in result.scan_points if abs(p.W - W_val) < 0.15]
        if points:
            points.sort(key=lambda p: p.h)
            h_vals = [p.h for p in points]
            ent_vals = [p.entanglement_entropy for p in points]
            ax.plot(h_vals, ent_vals, 'o-', label=f'W={W_val:.1f}')
    ax.set_xlabel('Transverse field h')
    ax.set_ylabel('Entanglement Entropy S')
    ax.set_title('Entanglement vs h (different W)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Magnetization vs W for different h
    ax = axes[1, 0]
    h_samples = [0.5, 1.0, 1.5, 2.0]
    for h_val in h_samples:
        points = [p for p in result.scan_points if abs(p.h - h_val) < 0.15]
        if points:
            points.sort(key=lambda p: p.W)
            W_vals = [p.W for p in points]
            mag_vals = [p.magnetization_z for p in points]
            ax.plot(W_vals, mag_vals, 'o-', label=f'h={h_val:.1f}')
    ax.set_xlabel('Disorder strength W')
    ax.set_ylabel('Magnetization |⟨σᶻ⟩|')
    ax.set_title('Magnetization vs W (different h)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Anomaly severity distribution
    ax = axes[1, 1]
    severities = [a.severity for a in result.anomalies]
    ax.hist(severities, bins=20, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Anomaly Severity')
    ax.set_ylabel('Count')
    ax.set_title('Anomaly Severity Distribution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'anomaly_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Anomaly analysis saved to {output_dir / 'anomaly_analysis.png'}")
    
    # Print key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    # Find critical point estimate (clean limit)
    clean_points = [p for p in result.scan_points if p.W < 0.1]
    if clean_points:
        max_chi_point = max(clean_points, key=lambda p: p.susceptibility_z)
        print(f"\nClean limit critical point estimate:")
        print(f"  h_c ≈ {max_chi_point.h:.3f} (expected: 1.0)")
        print(f"  χ_max = {max_chi_point.susceptibility_z:.3f}")
    
    # Disorder effects
    print(f"\nDisorder effects:")
    weak_disorder = [p for p in result.scan_points if 0.4 < p.W < 0.6]
    strong_disorder = [p for p in result.scan_points if 1.8 < p.W < 2.0]
    
    if weak_disorder and strong_disorder:
        avg_chi_weak = np.mean([p.susceptibility_z for p in weak_disorder])
        avg_chi_strong = np.mean([p.susceptibility_z for p in strong_disorder])
        print(f"  Average χ (W~0.5): {avg_chi_weak:.3f}")
        print(f"  Average χ (W~2.0): {avg_chi_strong:.3f}")
        print(f"  Ratio: {avg_chi_strong / avg_chi_weak:.2f}x")
    
    # Most interesting regions
    print(f"\nMost interesting regions for further study:")
    high_severity_anomalies = [a for a in result.anomalies if a.severity > 0.7]
    regions = {}
    for a in high_severity_anomalies:
        key = (round(a.h, 1), round(a.W, 1))
        if key not in regions:
            regions[key] = []
        regions[key].append(a)
    
    for i, (key, anomalies) in enumerate(sorted(regions.items(), key=lambda x: len(x[1]), reverse=True)[:5], 1):
        h, W = key
        print(f"  {i}. (h≈{h}, W≈{W}): {len(anomalies)} high-severity anomalies")
        types = set(a.anomaly_type for a in anomalies)
        print(f"     Types: {', '.join(types)}")
    
    print("\n" + "=" * 80)
    print("TASK 6 COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Review phase diagrams and anomaly maps")
    print("  2. Identify most promising regions for deep dive (Task 10)")
    print("  3. Consider increasing resolution in anomalous regions")
    print("  4. Compare with known disordered QCP behavior")


if __name__ == '__main__':
    main()
