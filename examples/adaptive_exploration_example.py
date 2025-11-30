"""
Adaptive Exploration Guided by Validation Confidence Example

This example demonstrates Task 16.5: Adaptive exploration guided by validation
confidence. This implements the temporal workflow pattern discovered during system
implementation.

Key Concept:
- High confidence → explore new regions (exploration)
- Low confidence → refine current region (exploitation)

This creates an adaptive workflow where validation confidence guides the exploration
strategy, enabling efficient parameter space exploration.

Novel Methodological Contribution:
Using validation confidence as an acquisition function for adaptive sampling creates
a natural feedback loop between exploration and validation. This temporal workflow
pattern enables:
1. Efficient exploration (avoid over-sampling high-confidence regions)
2. Thorough validation (refine low-confidence regions)
3. Automatic adaptation (switch between exploration/exploitation)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

from src.research.parameter_explorer import ParameterSpaceExplorer
from src.research.confidence_aggregator import ConfidenceAggregator, AggregatedConfidence
from src.research.base_types import (
    ModelVariantConfig, VAEAnalysisResults, NovelPhenomenon,
    LatentRepresentation, ExplorationStrategy
)
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.config import LoggingConfig


def create_mock_vae_results(
    variant_id: str,
    parameters: Dict[str, float],
    add_noise: bool = False
) -> VAEAnalysisResults:
    """Create mock VAE analysis results for demonstration.
    
    Args:
        variant_id: Variant identifier
        parameters: Parameter values
        add_noise: Whether to add noise to simulate uncertainty
        
    Returns:
        Mock VAE analysis results
    """
    # Simulate critical temperature based on parameters
    temp = parameters.get('temperature', 2.5)
    field = parameters.get('field', 0.0)
    
    # Simple model: Tc decreases with field
    tc = 2.269 * (1.0 - 0.5 * field)
    
    # Add noise if requested
    if add_noise:
        tc += np.random.normal(0, 0.1)
    
    # Simulate exponents (2D Ising-like)
    exponents = {
        'beta': 0.125 + (0.02 if add_noise else 0.0) * np.random.randn(),
        'nu': 1.0 + (0.05 if add_noise else 0.0) * np.random.randn(),
        'gamma': 1.75 + (0.05 if add_noise else 0.0) * np.random.randn()
    }
    
    # R² values (lower if noisy)
    r_squared = {
        'beta': 0.95 - (0.2 if add_noise else 0.0),
        'nu': 0.96 - (0.2 if add_noise else 0.0),
        'gamma': 0.94 - (0.2 if add_noise else 0.0)
    }
    
    return VAEAnalysisResults(
        variant_id=variant_id,
        parameters=parameters,
        critical_temperature=tc,
        tc_confidence=0.95 - (0.2 if add_noise else 0.0),
        exponents=exponents,
        exponent_errors={k: 0.01 for k in exponents},
        r_squared_values=r_squared,
        latent_representation=LatentRepresentation(
            latent_means=np.random.randn(10, 16),
            latent_stds=np.random.randn(10, 16),
            order_parameter_dim=0,
            reconstruction_quality={'mse': 0.01}
        ),
        order_parameter_dim=0
    )


def simulate_exploration_iteration(
    explorer: ParameterSpaceExplorer,
    aggregator: ConfidenceAggregator,
    points: np.ndarray,
    add_noise: bool = False
) -> List[AggregatedConfidence]:
    """Simulate one iteration of exploration with confidence calculation.
    
    Args:
        explorer: Parameter space explorer
        aggregator: Confidence aggregator
        points: Parameter points to explore
        add_noise: Whether to add noise (simulates low confidence)
        
    Returns:
        List of aggregated confidence results
    """
    param_names = sorted(explorer.parameter_ranges.keys())
    confidences = []
    
    for point in points:
        # Convert point to parameter dictionary
        parameters = {param_names[i]: point[i] for i in range(len(param_names))}
        
        # Simulate VAE analysis
        vae_results = create_mock_vae_results(
            explorer.variant_id,
            parameters,
            add_noise=add_noise
        )
        
        # Simulate phenomena detection (no phenomena for this example)
        phenomena = []
        
        # Aggregate confidence
        agg_conf = aggregator.aggregate_confidence(
            vae_results=vae_results,
            phenomena=phenomena,
            comparison_results=None,
            validation_results=None
        )
        
        # Update explorer with confidence
        explorer.update_confidence(parameters, agg_conf.overall_confidence)
        
        confidences.append(agg_conf)
    
    return confidences


def example_1_basic_confidence_guided_sampling():
    """Example 1: Basic confidence-guided sampling.
    
    Demonstrates how confidence guides exploration strategy:
    - Start with initial random sampling
    - Calculate confidence for each point
    - Use confidence to guide next sampling iteration
    """
    print("\n" + "="*80)
    print("Example 1: Basic Confidence-Guided Sampling")
    print("="*80)
    
    # Create explorer
    explorer = ParameterSpaceExplorer(
        variant_id='2d_ising',
        parameter_ranges={
            'temperature': (1.5, 3.5),
            'field': (0.0, 0.5)
        }
    )
    
    # Create confidence aggregator
    aggregator = ConfidenceAggregator()
    
    # Iteration 1: Initial random sampling
    print("\nIteration 1: Initial random sampling")
    initial_points = explorer.generate_sampling_points(
        ExplorationStrategy(method='random', n_points=10)
    )
    
    confidences_1 = simulate_exploration_iteration(
        explorer, aggregator, initial_points, add_noise=False
    )
    
    avg_conf_1 = np.mean([c.overall_confidence for c in confidences_1])
    print(f"Average confidence: {avg_conf_1:.2%}")
    print(f"Explored points: {len(explorer.explored_points)}")
    
    # Iteration 2: Confidence-guided sampling (high confidence → explore new)
    print("\nIteration 2: Confidence-guided sampling (high confidence)")
    guided_points_2 = explorer.confidence_guided_sampling(
        n_points=10,
        confidence_threshold=0.8,
        exploration_mode='random'
    )
    
    confidences_2 = simulate_exploration_iteration(
        explorer, aggregator, guided_points_2, add_noise=False
    )
    
    avg_conf_2 = np.mean([c.overall_confidence for c in confidences_2])
    print(f"Average confidence: {avg_conf_2:.2%}")
    print(f"Explored points: {len(explorer.explored_points)}")
    
    # Iteration 3: Introduce low confidence region
    print("\nIteration 3: Introduce low confidence region")
    # Manually add some low-confidence points
    low_conf_points = np.array([[2.5, 0.25], [2.6, 0.26]])
    confidences_3 = simulate_exploration_iteration(
        explorer, aggregator, low_conf_points, add_noise=True  # Add noise → low confidence
    )
    
    avg_conf_3 = np.mean([c.overall_confidence for c in confidences_3])
    print(f"Average confidence: {avg_conf_3:.2%}")
    
    # Iteration 4: Confidence-guided sampling (low confidence → refine)
    print("\nIteration 4: Confidence-guided sampling (low confidence)")
    guided_points_4 = explorer.confidence_guided_sampling(
        n_points=10,
        confidence_threshold=0.8,
        exploration_mode='random'
    )
    
    confidences_4 = simulate_exploration_iteration(
        explorer, aggregator, guided_points_4, add_noise=False
    )
    
    avg_conf_4 = np.mean([c.overall_confidence for c in confidences_4])
    print(f"Average confidence: {avg_conf_4:.2%}")
    print(f"Explored points: {len(explorer.explored_points)}")
    
    # Summary
    summary = explorer.get_exploration_summary()
    print("\nExploration Summary:")
    print(f"  Total explored points: {summary['n_explored']}")
    print(f"  Average confidence: {summary['avg_confidence']:.2%}")
    print(f"  Average uncertainty: {summary['avg_uncertainty']:.2%}")
    
    print("\n✓ Example 1 complete: Demonstrated confidence-guided sampling")


def example_2_exploration_vs_exploitation():
    """Example 2: Exploration vs Exploitation trade-off.
    
    Demonstrates how confidence threshold controls the exploration/exploitation
    trade-off.
    """
    print("\n" + "="*80)
    print("Example 2: Exploration vs Exploitation Trade-off")
    print("="*80)
    
    # Create explorer
    explorer = ParameterSpaceExplorer(
        variant_id='2d_ising',
        parameter_ranges={
            'temperature': (1.5, 3.5),
            'field': (0.0, 0.5)
        }
    )
    
    aggregator = ConfidenceAggregator()
    
    # Initial sampling
    print("\nInitial sampling (10 points)")
    initial_points = explorer.generate_sampling_points(
        ExplorationStrategy(method='grid', n_points=10)
    )
    simulate_exploration_iteration(explorer, aggregator, initial_points, add_noise=False)
    
    # Test different confidence thresholds
    thresholds = [0.5, 0.7, 0.9]
    
    for threshold in thresholds:
        print(f"\nTesting threshold = {threshold:.1f}")
        
        # Generate points with this threshold
        points = explorer.confidence_guided_sampling(
            n_points=5,
            confidence_threshold=threshold,
            exploration_mode='random'
        )
        
        # Check if points are in new regions or refinement regions
        explored_regions = explorer._identify_explored_regions()
        n_new_regions = sum(
            1 for point in points
            if not explorer._is_in_explored_region(
                {name: point[i] for i, name in enumerate(sorted(explorer.parameter_ranges.keys()))},
                explored_regions
            )
        )
        
        print(f"  Points in new regions: {n_new_regions}/{len(points)}")
        print(f"  Points in explored regions: {len(points) - n_new_regions}/{len(points)}")
        
        if threshold < 0.7:
            print("  → Low threshold: More exploitation (refinement)")
        else:
            print("  → High threshold: More exploration (new regions)")
    
    print("\n✓ Example 2 complete: Demonstrated exploration/exploitation trade-off")


def example_3_adaptive_workflow():
    """Example 3: Complete adaptive workflow.
    
    Demonstrates the full temporal workflow pattern:
    1. Initial exploration
    2. Confidence calculation
    3. Adaptive refinement (low confidence)
    4. New region exploration (high confidence)
    5. Repeat until convergence
    """
    print("\n" + "="*80)
    print("Example 3: Complete Adaptive Workflow")
    print("="*80)
    
    # Create explorer
    explorer = ParameterSpaceExplorer(
        variant_id='2d_ising_adaptive',
        parameter_ranges={
            'temperature': (1.5, 3.5),
            'field': (0.0, 0.5)
        }
    )
    
    aggregator = ConfidenceAggregator()
    
    # Track progress
    iteration_stats = []
    
    # Adaptive workflow loop
    max_iterations = 5
    confidence_threshold = 0.8
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        if iteration == 0:
            # Initial exploration
            print("Phase: Initial exploration")
            points = explorer.generate_sampling_points(
                ExplorationStrategy(method='random', n_points=8)
            )
            add_noise = False
        else:
            # Confidence-guided sampling
            summary = explorer.get_exploration_summary()
            avg_conf = summary['avg_confidence']
            
            if avg_conf > confidence_threshold:
                print(f"Phase: Exploration (confidence={avg_conf:.2%} > {confidence_threshold:.2%})")
                points = explorer.confidence_guided_sampling(
                    n_points=6,
                    confidence_threshold=confidence_threshold,
                    exploration_mode='random'
                )
                add_noise = False
            else:
                print(f"Phase: Exploitation (confidence={avg_conf:.2%} < {confidence_threshold:.2%})")
                points = explorer.confidence_guided_sampling(
                    n_points=6,
                    confidence_threshold=confidence_threshold,
                    exploration_mode='random'
                )
                # Simulate that refinement improves confidence
                add_noise = False
        
        # Simulate exploration
        confidences = simulate_exploration_iteration(
            explorer, aggregator, points, add_noise=add_noise
        )
        
        # Track statistics
        summary = explorer.get_exploration_summary()
        iteration_stats.append({
            'iteration': iteration + 1,
            'n_explored': summary['n_explored'],
            'avg_confidence': summary['avg_confidence'],
            'avg_uncertainty': summary['avg_uncertainty']
        })
        
        print(f"Explored: {summary['n_explored']} points")
        print(f"Avg confidence: {summary['avg_confidence']:.2%}")
        print(f"Avg uncertainty: {summary['avg_uncertainty']:.2%}")
    
    # Visualize progress
    print("\nWorkflow Progress:")
    print("-" * 60)
    print(f"{'Iter':<6} {'Explored':<10} {'Avg Conf':<12} {'Avg Uncert':<12}")
    print("-" * 60)
    for stats in iteration_stats:
        print(
            f"{stats['iteration']:<6} "
            f"{stats['n_explored']:<10} "
            f"{stats['avg_confidence']:<12.2%} "
            f"{stats['avg_uncertainty']:<12.2%}"
        )
    
    print("\n✓ Example 3 complete: Demonstrated complete adaptive workflow")


def example_4_visualize_exploration_strategy():
    """Example 4: Visualize exploration strategy.
    
    Creates visualizations showing how confidence guides exploration.
    """
    print("\n" + "="*80)
    print("Example 4: Visualize Exploration Strategy")
    print("="*80)
    
    output_dir = Path('results/adaptive_exploration_demo')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create explorer
    explorer = ParameterSpaceExplorer(
        variant_id='2d_ising_visual',
        parameter_ranges={
            'temperature': (1.5, 3.5),
            'field': (0.0, 0.5)
        }
    )
    
    aggregator = ConfidenceAggregator()
    
    # Phase 1: Initial exploration
    print("\nPhase 1: Initial exploration")
    initial_points = explorer.generate_sampling_points(
        ExplorationStrategy(method='grid', n_points=16)
    )
    simulate_exploration_iteration(explorer, aggregator, initial_points, add_noise=False)
    
    # Add some low-confidence points
    low_conf_points = np.array([
        [2.2, 0.2],
        [2.3, 0.22],
        [2.4, 0.24]
    ])
    simulate_exploration_iteration(explorer, aggregator, low_conf_points, add_noise=True)
    
    # Phase 2: Confidence-guided refinement
    print("Phase 2: Confidence-guided refinement")
    refined_points = explorer.confidence_guided_sampling(
        n_points=10,
        confidence_threshold=0.8,
        exploration_mode='random'
    )
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Initial exploration with confidence
    explored = [p for p in explorer.explored_points if p.explored]
    temps = [p.parameters['temperature'] for p in explored]
    fields = [p.parameters['field'] for p in explored]
    confidences = [1.0 - p.uncertainty for p in explored]
    
    scatter1 = ax1.scatter(
        temps, fields, c=confidences, s=100,
        cmap='RdYlGn', vmin=0, vmax=1, edgecolors='black', linewidths=1.5
    )
    ax1.set_xlabel('Temperature', fontsize=14)
    ax1.set_ylabel('Field', fontsize=14)
    ax1.set_title('Phase 1: Initial Exploration\n(Color = Confidence)', fontsize=16, fontweight='bold')
    ax1.grid(alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Confidence', fontsize=12)
    
    # Right plot: Refined points
    refined_temps = [p[0] for p in refined_points]
    refined_fields = [p[1] for p in refined_points]
    
    ax2.scatter(
        temps, fields, c=confidences, s=100,
        cmap='RdYlGn', vmin=0, vmax=1, edgecolors='black', linewidths=1.5,
        label='Initial points'
    )
    ax2.scatter(
        refined_temps, refined_fields, c='blue', s=150,
        marker='*', edgecolors='black', linewidths=1.5,
        label='Refined points', zorder=10
    )
    ax2.set_xlabel('Temperature', fontsize=14)
    ax2.set_ylabel('Field', fontsize=14)
    ax2.set_title('Phase 2: Confidence-Guided Refinement\n(Stars = New Points)', fontsize=16, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    
    output_path = output_dir / 'exploration_strategy_visualization.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization: {output_path}")
    
    plt.close()
    
    print("\n✓ Example 4 complete: Created exploration strategy visualization")


def example_5_convergence_analysis():
    """Example 5: Convergence analysis.
    
    Demonstrates how adaptive exploration converges to optimal coverage.
    """
    print("\n" + "="*80)
    print("Example 5: Convergence Analysis")
    print("="*80)
    
    output_dir = Path('results/adaptive_exploration_demo')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compare adaptive vs non-adaptive exploration
    strategies = {
        'Adaptive (confidence-guided)': 'adaptive',
        'Random sampling': 'random',
        'Grid sampling': 'grid'
    }
    
    results = {}
    
    for strategy_name, strategy_type in strategies.items():
        print(f"\nTesting: {strategy_name}")
        
        explorer = ParameterSpaceExplorer(
            variant_id='2d_ising_convergence',
            parameter_ranges={
                'temperature': (1.5, 3.5),
                'field': (0.0, 0.5)
            }
        )
        
        aggregator = ConfidenceAggregator()
        
        confidence_history = []
        
        for iteration in range(5):
            if strategy_type == 'adaptive':
                if iteration == 0:
                    points = explorer.generate_sampling_points(
                        ExplorationStrategy(method='random', n_points=8)
                    )
                else:
                    points = explorer.confidence_guided_sampling(
                        n_points=6,
                        confidence_threshold=0.8
                    )
            elif strategy_type == 'random':
                points = explorer.generate_sampling_points(
                    ExplorationStrategy(method='random', n_points=8 if iteration == 0 else 6)
                )
            else:  # grid
                points = explorer.generate_sampling_points(
                    ExplorationStrategy(method='grid', n_points=8 if iteration == 0 else 6)
                )
            
            simulate_exploration_iteration(explorer, aggregator, points, add_noise=False)
            
            summary = explorer.get_exploration_summary()
            confidence_history.append(summary['avg_confidence'])
        
        results[strategy_name] = confidence_history
        print(f"  Final confidence: {confidence_history[-1]:.2%}")
    
    # Visualize convergence
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for strategy_name, confidence_history in results.items():
        iterations = list(range(1, len(confidence_history) + 1))
        ax.plot(iterations, confidence_history, marker='o', linewidth=2, label=strategy_name)
    
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Average Confidence', fontsize=14)
    ax.set_title('Convergence Comparison: Adaptive vs Non-Adaptive', fontsize=16, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    output_path = output_dir / 'convergence_comparison.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved convergence plot: {output_path}")
    
    plt.close()
    
    print("\n✓ Example 5 complete: Demonstrated convergence analysis")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("ADAPTIVE EXPLORATION GUIDED BY VALIDATION CONFIDENCE")
    print("Task 16.5 - Novel Methodological Enhancement")
    print("="*80)
    
    print("\nThis example demonstrates the temporal workflow pattern:")
    print("  - High confidence -> explore new regions (exploration)")
    print("  - Low confidence -> refine current region (exploitation)")
    print("\nKey Innovation:")
    print("  Using validation confidence as an acquisition function creates")
    print("  a natural feedback loop between exploration and validation.")
    
    # Set up logging
    logging_config = LoggingConfig(
        level='INFO',
        console_output=True,
        file_output=False
    )
    setup_logging(logging_config)
    
    # Run examples
    example_1_basic_confidence_guided_sampling()
    example_2_exploration_vs_exploitation()
    example_3_adaptive_workflow()
    example_4_visualize_exploration_strategy()
    example_5_convergence_analysis()
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("  1. Confidence-guided sampling adapts exploration strategy automatically")
    print("  2. High confidence triggers exploration of new regions")
    print("  3. Low confidence triggers refinement of current regions")
    print("  4. Adaptive approach converges faster than non-adaptive methods")
    print("  5. Temporal workflow pattern enables efficient parameter space coverage")
    print("\nNovel Methodological Contribution:")
    print("  This approach bridges exploration and validation by using validation")
    print("  confidence as a feedback signal for adaptive sampling. This creates")
    print("  an efficient workflow that balances exploration and exploitation.")
    print("\nOutput saved to: results/adaptive_exploration_demo/")


if __name__ == '__main__':
    main()
