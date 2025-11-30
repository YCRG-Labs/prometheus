"""
Example demonstrating disorder infrastructure for quantum systems.

Shows:
1. Random coupling generation (box, Gaussian distributions)
2. Disorder realization generation
3. Disorder averaging with parallel computation
4. Statistical analysis of disorder-averaged quantities
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '.')

from src.quantum import (
    DisorderType,
    DisorderConfig,
    RandomCouplingGenerator,
    DisorderedSystemConfig,
    DisorderRealizationGenerator,
    DisorderAveragingFramework,
    DisorderStatisticalAnalyzer,
)


def example_1_random_coupling_generation():
    """Example 1: Generate random couplings with different distributions."""
    print("=" * 70)
    print("Example 1: Random Coupling Generation")
    print("=" * 70)
    
    gen = RandomCouplingGenerator(seed=42)
    
    # Box (uniform) distribution
    box_values = gen.generate_box(size=1000, center=1.0, width=0.5)
    print(f"\nBox distribution [0.75, 1.25]:")
    print(f"  Mean: {np.mean(box_values):.4f}")
    print(f"  Std:  {np.std(box_values):.4f}")
    print(f"  Min:  {np.min(box_values):.4f}")
    print(f"  Max:  {np.max(box_values):.4f}")
    
    # Gaussian distribution
    gaussian_values = gen.generate_gaussian(size=1000, mean=2.0, std=0.5)
    print(f"\nGaussian distribution N(2.0, 0.5):")
    print(f"  Mean: {np.mean(gaussian_values):.4f}")
    print(f"  Std:  {np.std(gaussian_values):.4f}")
    
    # Binary distribution
    binary_values = gen.generate_binary(size=1000, values=(-1.0, 1.0), p=0.7)
    print(f"\nBinary distribution {{-1, +1}} with p=0.7:")
    print(f"  Mean: {np.mean(binary_values):.4f}")
    print(f"  Fraction +1: {np.mean(binary_values == 1.0):.4f}")
    
    # Log-uniform distribution
    loguniform_values = gen.generate_loguniform(size=1000, low=0.1, high=10.0)
    print(f"\nLog-uniform distribution [0.1, 10.0]:")
    print(f"  Geometric mean: {np.exp(np.mean(np.log(loguniform_values))):.4f}")
    print(f"  Min: {np.min(loguniform_values):.4f}")
    print(f"  Max: {np.max(loguniform_values):.4f}")


def example_2_disorder_realizations():
    """Example 2: Generate complete disorder realizations."""
    print("\n" + "=" * 70)
    print("Example 2: Disorder Realization Generation")
    print("=" * 70)
    
    # Configure disordered system
    config = DisorderedSystemConfig(
        L=10,  # 10 spins
        J_config=DisorderConfig(
            disorder_type=DisorderType.BOX,
            center=1.0,
            width=0.5
        ),
        h_config=DisorderConfig(
            disorder_type=DisorderType.GAUSSIAN,
            center=1.0,
            width=0.3
        ),
        periodic=True
    )
    
    gen = DisorderRealizationGenerator(config, base_seed=42)
    
    # Generate single realization
    realization = gen.generate_single(0)
    print(f"\nSingle realization (L={config.L}, periodic):")
    print(f"  Number of bonds: {len(realization.couplings)}")
    print(f"  J values: {realization.couplings[:5]} ...")
    print(f"  h values: {realization.transverse_fields[:5]} ...")
    
    # Generate batch
    batch = gen.generate_batch(n_realizations=100)
    print(f"\nGenerated {len(batch)} realizations")
    
    # Statistics across realizations
    all_J = np.array([r.couplings for r in batch])
    print(f"\nStatistics across 100 realizations:")
    print(f"  Mean J: {np.mean(all_J):.4f} (expected: 1.0)")
    print(f"  Std J:  {np.std(all_J):.4f}")


def example_3_disorder_averaging():
    """Example 3: Disorder averaging with parallel computation."""
    print("\n" + "=" * 70)
    print("Example 3: Disorder Averaging")
    print("=" * 70)
    
    # Configure system
    config = DisorderedSystemConfig(
        L=12,
        J_config=DisorderConfig(DisorderType.BOX, center=1.0, width=0.5),
        h_config=DisorderConfig(DisorderType.BOX, center=1.0, width=0.3),
        periodic=True
    )
    
    framework = DisorderAveragingFramework(config, base_seed=42)
    
    # Define observable: average coupling strength
    def compute_avg_coupling(realization):
        return np.mean(realization.couplings)
    
    # Sequential computation
    print("\nSequential computation:")
    result_seq = framework.disorder_average(
        compute_avg_coupling,
        n_realizations=200,
        parallel=False
    )
    print(f"  Mean: {result_seq.mean:.4f} ± {result_seq.stderr:.4f}")
    print(f"  95% CI: [{result_seq.confidence_interval_95[0]:.4f}, "
          f"{result_seq.confidence_interval_95[1]:.4f}]")
    
    # Parallel computation
    print("\nParallel computation:")
    framework_par = DisorderAveragingFramework(config, base_seed=42)
    result_par = framework_par.disorder_average(
        compute_avg_coupling,
        n_realizations=200,
        parallel=True
    )
    print(f"  Mean: {result_par.mean:.4f} ± {result_par.stderr:.4f}")
    print(f"  Results match: {np.isclose(result_seq.mean, result_par.mean)}")


def example_4_multiple_observables():
    """Example 4: Compute multiple observables efficiently."""
    print("\n" + "=" * 70)
    print("Example 4: Multiple Observables")
    print("=" * 70)
    
    config = DisorderedSystemConfig(
        L=10,
        J_config=DisorderConfig(DisorderType.BOX, center=1.0, width=0.5),
        h_config=DisorderConfig(DisorderType.GAUSSIAN, center=2.0, width=0.5),
        periodic=True
    )
    
    framework = DisorderAveragingFramework(config, base_seed=42)
    
    # Define multiple observables
    compute_fns = {
        'J_mean': lambda r: np.mean(r.couplings),
        'J_std': lambda r: np.std(r.couplings),
        'h_mean': lambda r: np.mean(r.transverse_fields),
        'h_std': lambda r: np.std(r.transverse_fields),
        'J_min': lambda r: np.min(r.couplings),
        'J_max': lambda r: np.max(r.couplings),
    }
    
    results = framework.disorder_average_multiple(
        compute_fns,
        n_realizations=500,
        parallel=True
    )
    
    print("\nDisorder-averaged observables (500 realizations):")
    for name, result in results.items():
        print(f"  {name:10s}: {result.mean:.4f} ± {result.stderr:.4f}")


def example_5_statistical_analysis():
    """Example 5: Statistical analysis of disorder-averaged data."""
    print("\n" + "=" * 70)
    print("Example 5: Statistical Analysis")
    print("=" * 70)
    
    # Generate some disorder-averaged data
    config = DisorderedSystemConfig(
        L=8,
        J_config=DisorderConfig(DisorderType.GAUSSIAN, center=1.0, width=0.5),
        h_config=DisorderConfig(DisorderType.BOX, center=1.0, width=0.3),
        periodic=True
    )
    
    framework = DisorderAveragingFramework(config, base_seed=42)
    
    def compute_total_coupling(realization):
        return np.sum(realization.couplings)
    
    result = framework.disorder_average(
        compute_total_coupling,
        n_realizations=1000,
        parallel=True
    )
    
    # Comprehensive statistical analysis
    analysis = DisorderStatisticalAnalyzer.analyze_distribution(result.values)
    
    print("\nDistribution analysis:")
    print(f"  Mean:     {analysis.mean:.4f}")
    print(f"  Median:   {analysis.median:.4f}")
    print(f"  Std:      {analysis.std:.4f}")
    print(f"  Skewness: {analysis.skewness:.4f}")
    print(f"  Kurtosis: {analysis.kurtosis:.4f}")
    print(f"  Q25-Q75:  [{analysis.q25:.4f}, {analysis.q75:.4f}]")
    
    # Bootstrap error estimation
    estimate, std_error, ci = DisorderStatisticalAnalyzer.bootstrap_error(
        result.values,
        n_bootstrap=1000,
        seed=42
    )
    print(f"\nBootstrap analysis:")
    print(f"  Estimate: {estimate:.4f}")
    print(f"  Std error: {std_error:.4f}")
    print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    # Convergence analysis
    convergence = DisorderStatisticalAnalyzer.convergence_analysis(result.values)
    print(f"\nConvergence analysis:")
    print(f"  Final mean: {convergence['running_mean'][-1]:.4f}")
    print(f"  Final stderr: {convergence['running_stderr'][-1]:.4f}")
    
    is_converged, rel_change = DisorderStatisticalAnalyzer.is_converged(
        result.values,
        tolerance=0.01
    )
    print(f"  Converged: {is_converged} (relative change: {rel_change:.6f})")
    
    # Typical vs average (important for disordered systems)
    typical_avg = DisorderStatisticalAnalyzer.typical_vs_average(result.values)
    print(f"\nTypical vs average:")
    print(f"  Arithmetic mean: {typical_avg['arithmetic_mean']:.4f}")
    print(f"  Geometric mean:  {typical_avg['geometric_mean']:.4f}")
    print(f"  Ratio:           {typical_avg['ratio']:.4f}")


def example_6_visualization():
    """Example 6: Visualize disorder distributions and convergence."""
    print("\n" + "=" * 70)
    print("Example 6: Visualization")
    print("=" * 70)
    
    config = DisorderedSystemConfig(
        L=10,
        J_config=DisorderConfig(DisorderType.BOX, center=1.0, width=0.6),
        h_config=DisorderConfig(DisorderType.GAUSSIAN, center=1.5, width=0.4),
        periodic=True
    )
    
    framework = DisorderAveragingFramework(config, base_seed=42)
    
    def compute_observable(realization):
        # Some complex observable
        return np.mean(realization.couplings) * np.mean(realization.transverse_fields)
    
    result = framework.disorder_average(
        compute_observable,
        n_realizations=1000,
        parallel=True
    )
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Histogram of values
    ax = axes[0, 0]
    ax.hist(result.values, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(result.mean, color='red', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(np.median(result.values), color='blue', linestyle='--', 
               linewidth=2, label='Median')
    ax.set_xlabel('Observable Value')
    ax.set_ylabel('Probability Density')
    ax.set_title('Distribution of Disorder-Averaged Observable')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Convergence plot
    ax = axes[0, 1]
    convergence = DisorderStatisticalAnalyzer.convergence_analysis(result.values)
    n_samples = convergence['n_samples']
    running_mean = convergence['running_mean']
    running_stderr = convergence['running_stderr']
    
    ax.plot(n_samples, running_mean, 'b-', linewidth=2, label='Running mean')
    ax.fill_between(n_samples, 
                     running_mean - 2*running_stderr,
                     running_mean + 2*running_stderr,
                     alpha=0.3, label='±2 stderr')
    ax.axhline(result.mean, color='red', linestyle='--', label='Final mean')
    ax.set_xlabel('Number of Realizations')
    ax.set_ylabel('Observable Value')
    ax.set_title('Convergence of Disorder Average')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Sample disorder realizations
    ax = axes[1, 0]
    gen = DisorderRealizationGenerator(config, base_seed=42)
    for i in range(10):
        r = gen.generate_single(i)
        ax.plot(r.couplings, alpha=0.5, linewidth=1)
    ax.set_xlabel('Site Index')
    ax.set_ylabel('Coupling Strength J')
    ax.set_title('Sample Disorder Realizations (Couplings)')
    ax.grid(True, alpha=0.3)
    
    # 4. Q-Q plot (check for normality)
    ax = axes[1, 1]
    from scipy import stats
    stats.probplot(result.values, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Check for Normality)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/disorder_infrastructure_demo.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: results/disorder_infrastructure_demo.png")
    plt.close()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("DISORDER INFRASTRUCTURE DEMONSTRATION")
    print("=" * 70)
    
    example_1_random_coupling_generation()
    example_2_disorder_realizations()
    example_3_disorder_averaging()
    example_4_multiple_observables()
    example_5_statistical_analysis()
    example_6_visualization()
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
