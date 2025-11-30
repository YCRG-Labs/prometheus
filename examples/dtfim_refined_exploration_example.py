"""
Example: Refined exploration of DTFIM anomalous regions.

Demonstrates Task 10: Deep dive into anomalous regions with:
- 10x resolution in anomalous regions
- Larger system sizes (L = 12, 16, 20)
- More disorder realizations (100 → 1000)
- Full entanglement spectrum computation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import argparse

from src.research.dtfim_refined_explorer import (
    DTFIMRefinedExplorer,
    AnomalousRegion,
    RefinedExplorationResult,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_default_anomalous_regions() -> list:
    """
    Create default anomalous regions based on typical DTFIM phase diagram.
    
    These regions are chosen based on known physics:
    1. Critical region near h ≈ 1.0 (clean TFIM critical point)
    2. Strong disorder region (potential Griffiths phase)
    3. Intermediate disorder region (disorder-induced effects)
    """
    return [
        AnomalousRegion(
            h_center=1.0,
            W_center=0.5,
            h_width=0.3,
            W_width=0.3,
            anomaly_type="critical_region",
            severity=0.9,
            description="Near clean TFIM critical point with weak disorder"
        ),
        AnomalousRegion(
            h_center=1.0,
            W_center=1.5,
            h_width=0.3,
            W_width=0.3,
            anomaly_type="strong_disorder",
            severity=0.8,
            description="Strong disorder region - potential Griffiths phase"
        ),
        AnomalousRegion(
            h_center=0.5,
            W_center=1.0,
            h_width=0.3,
            W_width=0.3,
            anomaly_type="ordered_with_disorder",
            severity=0.7,
            description="Ordered phase with intermediate disorder"
        ),
    ]


def run_refined_exploration(
    region: AnomalousRegion,
    system_sizes: list = [12, 16, 20],
    n_realizations: int = 100,
    resolution: int = 10,
    output_dir: Path = None,
    parallel: bool = True
) -> RefinedExplorationResult:
    """
    Run refined exploration of a single anomalous region.
    
    Args:
        region: Anomalous region to explore
        system_sizes: List of system sizes for finite-size scaling
        n_realizations: Number of disorder realizations
        resolution: Resolution factor (points per axis)
        output_dir: Output directory for results
        parallel: Use parallel computation
        
    Returns:
        RefinedExplorationResult
    """
    logger.info(f"Starting refined exploration of region: {region.anomaly_type}")
    logger.info(f"  h ∈ [{region.h_range[0]:.2f}, {region.h_range[1]:.2f}]")
    logger.info(f"  W ∈ [{region.W_range[0]:.2f}, {region.W_range[1]:.2f}]")
    logger.info(f"  System sizes: {system_sizes}")
    logger.info(f"  Disorder realizations: {n_realizations}")
    logger.info(f"  Resolution: {resolution}x{resolution}")
    
    # Create explorer
    explorer = DTFIMRefinedExplorer(
        system_sizes=system_sizes,
        n_disorder_realizations=n_realizations,
        J_mean=1.0,
        periodic=True,
        random_seed=42
    )
    
    # Run exploration
    result = explorer.refine_anomalous_region(
        region=region,
        resolution_factor=resolution,
        parallel=parallel,
        max_workers=4
    )
    
    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        result_file = output_dir / f"refined_{region.anomaly_type}.json"
        result.save(str(result_file))
        logger.info(f"Results saved to {result_file}")
    
    return result


def plot_refined_results(
    result: RefinedExplorationResult,
    output_dir: Path
):
    """Generate plots for refined exploration results."""
    
    # Create figure with subplots for each system size
    n_sizes = len(result.system_sizes)
    fig, axes = plt.subplots(n_sizes, 4, figsize=(16, 4 * n_sizes))
    if n_sizes == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(
        f'Refined DTFIM Exploration: {result.anomalous_region.anomaly_type}\n'
        f'h ∈ [{result.anomalous_region.h_range[0]:.2f}, {result.anomalous_region.h_range[1]:.2f}], '
        f'W ∈ [{result.anomalous_region.W_range[0]:.2f}, {result.anomalous_region.W_range[1]:.2f}]',
        fontsize=14
    )
    
    for i, L in enumerate(result.system_sizes):
        # Magnetization
        ax = axes[i, 0]
        mag_grid = result.get_observable_grid('magnetization_z', L)
        im = ax.contourf(result.h_values, result.W_values, mag_grid, levels=20, cmap='viridis')
        ax.set_xlabel('h')
        ax.set_ylabel('W')
        ax.set_title(f'L={L}: Magnetization |⟨σᶻ⟩|')
        plt.colorbar(im, ax=ax)
        
        # Susceptibility
        ax = axes[i, 1]
        chi_grid = result.get_observable_grid('susceptibility_z', L)
        chi_clipped = np.clip(chi_grid, 0, np.nanpercentile(chi_grid, 95))
        im = ax.contourf(result.h_values, result.W_values, chi_clipped, levels=20, cmap='hot')
        ax.set_xlabel('h')
        ax.set_ylabel('W')
        ax.set_title(f'L={L}: Susceptibility χᶻ')
        plt.colorbar(im, ax=ax)
        
        # Entanglement entropy
        ax = axes[i, 2]
        ent_grid = result.get_observable_grid('entanglement_entropy', L)
        im = ax.contourf(result.h_values, result.W_values, ent_grid, levels=20, cmap='plasma')
        ax.set_xlabel('h')
        ax.set_ylabel('W')
        ax.set_title(f'L={L}: Entanglement S')
        plt.colorbar(im, ax=ax)
        
        # Binder cumulant
        ax = axes[i, 3]
        binder_grid = result.get_observable_grid('binder_cumulant', L)
        im = ax.contourf(result.h_values, result.W_values, binder_grid, levels=20, cmap='coolwarm')
        ax.set_xlabel('h')
        ax.set_ylabel('W')
        ax.set_title(f'L={L}: Binder Cumulant U')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'refined_{result.anomalous_region.anomaly_type}_phase_diagrams.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Finite-size scaling plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Finite-Size Scaling Analysis', fontsize=14)
    
    # Get middle h value for cuts
    h_mid_idx = len(result.h_values) // 2
    h_mid = result.h_values[h_mid_idx]
    
    observables = [
        ('magnetization_z', 'Magnetization |⟨σᶻ⟩|'),
        ('susceptibility_z', 'Susceptibility χᶻ'),
        ('entanglement_entropy', 'Entanglement S'),
        ('binder_cumulant', 'Binder Cumulant U'),
        ('correlation_length', 'Correlation Length ξ'),
        ('energy_gap', 'Energy Gap Δ'),
    ]
    
    for idx, (obs, label) in enumerate(observables):
        ax = axes[idx // 3, idx % 3]
        
        for L in result.system_sizes:
            if obs in result.fss_data and L in result.fss_data[obs]:
                data = result.fss_data[obs][L][:, h_mid_idx]
                ax.plot(result.W_values, data, 'o-', label=f'L={L}')
        
        ax.set_xlabel('Disorder W')
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs W (h={h_mid:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'refined_{result.anomalous_region.anomaly_type}_fss.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plots saved to {output_dir}")


def analyze_entanglement_spectrum(result: RefinedExplorationResult, output_dir: Path):
    """Analyze and plot entanglement spectrum data."""
    
    fig, axes = plt.subplots(1, len(result.system_sizes), figsize=(5 * len(result.system_sizes), 5))
    if len(result.system_sizes) == 1:
        axes = [axes]
    
    fig.suptitle('Entanglement Spectrum Analysis', fontsize=14)
    
    for i, L in enumerate(result.system_sizes):
        ax = axes[i]
        points = result.get_points_for_size(L)
        
        # Get points at different W values for fixed h near critical
        h_target = result.anomalous_region.h_center
        
        W_samples = np.linspace(result.W_values[0], result.W_values[-1], 5)
        
        for W_target in W_samples:
            # Find closest point
            closest = min(points, key=lambda p: abs(p.h - h_target) + abs(p.W - W_target))
            
            if closest.entanglement_spectrum_mean is not None:
                spectrum = closest.entanglement_spectrum_mean
                # Plot entanglement energies
                ent_energies = -np.log(spectrum[spectrum > 1e-15])
                ax.plot(range(len(ent_energies)), ent_energies, 'o-', 
                       label=f'W={closest.W:.2f}', alpha=0.7)
        
        ax.set_xlabel('Level index')
        ax.set_ylabel('Entanglement energy ξ = -log(λ)')
        ax.set_title(f'L={L}, h≈{h_target:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'entanglement_spectrum_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Run refined DTFIM exploration."""
    
    parser = argparse.ArgumentParser(description='Refined DTFIM Exploration')
    parser.add_argument('--quick', action='store_true', 
                       help='Quick mode with reduced parameters')
    parser.add_argument('--region', type=int, default=0,
                       help='Region index to explore (0-2)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("DTFIM Refined Exploration - Task 10")
    print("=" * 80)
    
    # Output directory
    output_dir = Path("results/task10_dtfim_refined_exploration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get anomalous regions
    regions = create_default_anomalous_regions()
    
    if args.quick:
        # Quick mode for testing
        system_sizes = [8, 10, 12]
        n_realizations = 50
        resolution = 5
        logger.info("Running in QUICK mode with reduced parameters")
    else:
        # Full mode as per Task 10 requirements
        system_sizes = [12, 16, 20]
        n_realizations = 1000
        resolution = 10
    
    # Select region
    region = regions[args.region % len(regions)]
    
    print(f"\nExploring region {args.region}: {region.anomaly_type}")
    print(f"  Center: h={region.h_center}, W={region.W_center}")
    print(f"  System sizes: {system_sizes}")
    print(f"  Disorder realizations: {n_realizations}")
    print(f"  Resolution: {resolution}x{resolution}")
    print()
    
    # Run exploration
    result = run_refined_exploration(
        region=region,
        system_sizes=system_sizes,
        n_realizations=n_realizations,
        resolution=resolution,
        output_dir=output_dir,
        parallel=True
    )
    
    # Generate plots
    print("\nGenerating plots...")
    plot_refined_results(result, output_dir)
    analyze_entanglement_spectrum(result, output_dir)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPLORATION SUMMARY")
    print("=" * 80)
    print(f"Region: {region.anomaly_type}")
    print(f"Total points computed: {len(result.scan_points)}")
    print(f"System sizes: {result.system_sizes}")
    print(f"Computation time: {result.metadata.get('computation_time_minutes', 0):.1f} minutes")
    
    # Print key findings for each system size
    print("\nKey findings by system size:")
    for L in result.system_sizes:
        points = result.get_points_for_size(L)
        if points:
            max_chi_point = max(points, key=lambda p: p.susceptibility_z)
            max_ent_point = max(points, key=lambda p: p.entanglement_entropy)
            
            print(f"\n  L={L}:")
            print(f"    Max susceptibility: χ={max_chi_point.susceptibility_z:.3f} "
                  f"at (h={max_chi_point.h:.3f}, W={max_chi_point.W:.3f})")
            print(f"    Max entanglement: S={max_ent_point.entanglement_entropy:.3f} "
                  f"at (h={max_ent_point.h:.3f}, W={max_ent_point.W:.3f})")
    
    print("\n" + "=" * 80)
    print("TASK 10 EXPLORATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Analyze finite-size scaling to extract critical exponents")
    print("  2. Compare entanglement spectra across disorder strengths")
    print("  3. Identify signatures of Griffiths phases or new universality")
    print("  4. Proceed to Task 11: Characterize DTFIM anomalies")


if __name__ == '__main__':
    main()
