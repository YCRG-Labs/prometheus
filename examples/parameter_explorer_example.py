"""
Example demonstrating Parameter Space Explorer functionality.

This script shows how to use the ParameterSpaceExplorer to systematically
explore parameter spaces using different strategies: grid search, random
sampling, adaptive refinement, and Bayesian optimization.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.research import (
    ParameterSpaceExplorer,
    ExplorationStrategy,
    ModelVariantRegistry,
    ModelVariantConfig,
)


def main():
    """Demonstrate parameter space exploration strategies."""
    
    print("=" * 70)
    print("Parameter Space Explorer Example")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path("results/parameter_exploration_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define a model variant with multiple parameters
    registry = ModelVariantRegistry()
    
    config = ModelVariantConfig(
        name="long_range_ising_2d",
        dimensions=2,
        lattice_geometry="square",
        interaction_type="long_range",
        interaction_params={"alpha": 2.5},
        theoretical_tc=2.5,
    )
    
    # Try to register, or use existing if already registered
    try:
        variant_id = registry.register_variant(config)
        print(f"\nRegistered new variant: {variant_id}")
    except ValueError:
        variant_id = config.name
        print(f"\nUsing existing variant: {variant_id}")
    
    # Define parameter ranges to explore
    parameter_ranges = {
        'temperature': (1.0, 4.0),
        'alpha': (2.0, 3.0),
    }
    
    # Create explorer
    explorer = ParameterSpaceExplorer(variant_id, parameter_ranges)
    print(f"\nExploring parameter space:")
    print(f"  Temperature: {parameter_ranges['temperature']}")
    print(f"  Alpha: {parameter_ranges['alpha']}")
    
    # Example 1: Grid Sampling
    print("\n" + "-" * 70)
    print("Example 1: Grid Sampling")
    print("-" * 70)
    
    strategy_grid = ExplorationStrategy(
        method='grid',
        n_points=25
    )
    
    points_grid = explorer.generate_sampling_points(strategy_grid)
    print(f"Generated {len(points_grid)} grid points")
    
    # Visualize grid sampling
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(points_grid[:, 0], points_grid[:, 1], alpha=0.6, s=50)
    ax.set_xlabel('Temperature', fontsize=12)
    ax.set_ylabel('Alpha', fontsize=12)
    ax.set_title('Grid Sampling Strategy', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "grid_sampling.png", dpi=150)
    print(f"Saved visualization: {output_dir / 'grid_sampling.png'}")
    plt.close()
    
    # Example 2: Random Sampling
    print("\n" + "-" * 70)
    print("Example 2: Random Sampling")
    print("-" * 70)
    
    strategy_random = ExplorationStrategy(
        method='random',
        n_points=25
    )
    
    points_random = explorer.generate_sampling_points(strategy_random)
    print(f"Generated {len(points_random)} random points")
    
    # Visualize random sampling
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(points_random[:, 0], points_random[:, 1], alpha=0.6, s=50, color='orange')
    ax.set_xlabel('Temperature', fontsize=12)
    ax.set_ylabel('Alpha', fontsize=12)
    ax.set_title('Random Sampling Strategy', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "random_sampling.png", dpi=150)
    print(f"Saved visualization: {output_dir / 'random_sampling.png'}")
    plt.close()
    
    # Example 3: Adaptive Refinement
    print("\n" + "-" * 70)
    print("Example 3: Adaptive Refinement")
    print("-" * 70)
    
    # Simulate some explored points with varying uncertainty
    # High uncertainty near the critical temperature (around T=2.5)
    np.random.seed(42)
    for i in range(20):
        temp = np.random.uniform(1.0, 4.0)
        alpha = np.random.uniform(2.0, 3.0)
        
        # Simulate higher uncertainty near Tc
        uncertainty = 0.3 + 0.6 * np.exp(-((temp - 2.5)**2) / 0.5)
        uncertainty += np.random.normal(0, 0.1)
        uncertainty = np.clip(uncertainty, 0.0, 1.0)
        
        explorer.update_explored_point(
            parameters={'temperature': temp, 'alpha': alpha},
            results={'dummy': True},
            uncertainty=uncertainty
        )
    
    print(f"Simulated {len(explorer.explored_points)} explored points")
    
    strategy_adaptive = ExplorationStrategy(
        method='adaptive',
        n_points=30,
        refinement_iterations=2
    )
    
    points_adaptive = explorer.generate_sampling_points(strategy_adaptive)
    print(f"Generated {len(points_adaptive)} adaptive points")
    
    # Visualize adaptive sampling
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot explored points colored by uncertainty
    explored_temps = [p.parameters['temperature'] for p in explorer.explored_points]
    explored_alphas = [p.parameters['alpha'] for p in explorer.explored_points]
    uncertainties = [p.uncertainty for p in explorer.explored_points]
    
    scatter1 = ax.scatter(
        explored_temps, explored_alphas,
        c=uncertainties, cmap='YlOrRd',
        s=100, alpha=0.7, edgecolors='black',
        label='Explored (colored by uncertainty)'
    )
    
    # Plot new adaptive points
    ax.scatter(
        points_adaptive[:, 0], points_adaptive[:, 1],
        marker='x', s=80, color='blue', linewidths=2,
        label='New adaptive points'
    )
    
    ax.set_xlabel('Temperature', fontsize=12)
    ax.set_ylabel('Alpha', fontsize=12)
    ax.set_title('Adaptive Refinement Strategy', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    cbar = plt.colorbar(scatter1, ax=ax)
    cbar.set_label('Uncertainty', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / "adaptive_sampling.png", dpi=150)
    print(f"Saved visualization: {output_dir / 'adaptive_sampling.png'}")
    plt.close()
    
    # Example 4: Bayesian Optimization
    print("\n" + "-" * 70)
    print("Example 4: Bayesian Optimization")
    print("-" * 70)
    
    strategy_bayesian = ExplorationStrategy(
        method='bayesian',
        n_points=25
    )
    
    points_bayesian = explorer.generate_sampling_points(strategy_bayesian)
    print(f"Generated {len(points_bayesian)} Bayesian optimization points")
    
    # Visualize Bayesian optimization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot explored points
    ax.scatter(
        explored_temps, explored_alphas,
        c=uncertainties, cmap='YlOrRd',
        s=100, alpha=0.7, edgecolors='black',
        label='Explored (colored by uncertainty)'
    )
    
    # Plot Bayesian points
    ax.scatter(
        points_bayesian[:, 0], points_bayesian[:, 1],
        marker='^', s=80, color='green', linewidths=1.5,
        edgecolors='darkgreen', label='Bayesian optimization points'
    )
    
    ax.set_xlabel('Temperature', fontsize=12)
    ax.set_ylabel('Alpha', fontsize=12)
    ax.set_title('Bayesian Optimization Strategy', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    cbar = plt.colorbar(scatter1, ax=ax)
    cbar.set_label('Uncertainty', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / "bayesian_sampling.png", dpi=150)
    print(f"Saved visualization: {output_dir / 'bayesian_sampling.png'}")
    plt.close()
    
    # Example 5: Anomalous Region Identification
    print("\n" + "-" * 70)
    print("Example 5: Anomalous Region Identification")
    print("-" * 70)
    
    anomalous_regions = explorer.identify_anomalous_regions({})
    print(f"Identified {len(anomalous_regions)} anomalous regions")
    
    if len(anomalous_regions) > 0:
        print("\nAnomalous regions:")
        for i, region in enumerate(anomalous_regions[:3]):  # Show first 3
            print(f"  Region {i+1}:")
            for param, (min_val, max_val) in region.items():
                print(f"    {param}: [{min_val:.3f}, {max_val:.3f}]")
    
    # Example 6: Exploration Summary
    print("\n" + "-" * 70)
    print("Example 6: Exploration Summary")
    print("-" * 70)
    
    summary = explorer.get_exploration_summary()
    print(f"\nExploration Summary:")
    print(f"  Variant ID: {summary['variant_id']}")
    print(f"  Number of parameters: {summary['n_parameters']}")
    print(f"  Points explored: {summary['n_explored']}")
    print(f"  Total points: {summary['n_total_points']}")
    print(f"  Average uncertainty: {summary['avg_uncertainty']:.3f}")
    print(f"  Maximum uncertainty: {summary['max_uncertainty']:.3f}")
    
    # Create comparison plot
    print("\n" + "-" * 70)
    print("Creating comparison visualization")
    print("-" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    strategies = [
        ('Grid', points_grid, 'blue'),
        ('Random', points_random, 'orange'),
        ('Adaptive', points_adaptive, 'red'),
        ('Bayesian', points_bayesian, 'green'),
    ]
    
    for ax, (name, points, color) in zip(axes.flat, strategies):
        ax.scatter(points[:, 0], points[:, 1], alpha=0.6, s=50, color=color)
        ax.set_xlabel('Temperature', fontsize=11)
        ax.set_ylabel('Alpha', fontsize=11)
        ax.set_title(f'{name} Sampling ({len(points)} points)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(parameter_ranges['temperature'])
        ax.set_ylim(parameter_ranges['alpha'])
    
    plt.tight_layout()
    plt.savefig(output_dir / "strategy_comparison.png", dpi=150)
    print(f"Saved comparison: {output_dir / 'strategy_comparison.png'}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("Parameter Space Explorer demonstration complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
