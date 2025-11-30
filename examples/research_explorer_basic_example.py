"""
Basic example demonstrating the Novel Ising Model Research Explorer infrastructure.

This example shows how to:
1. Create a model variant configuration
2. Define a research hypothesis
3. Use the base data structures
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.research import (
    ModelVariantConfig,
    ResearchHypothesis,
    SimulationData,
    ExplorationStrategy,
    NovelPhenomenon,
)


def main():
    """Demonstrate basic usage of research explorer infrastructure."""
    
    print("=" * 70)
    print("Novel Ising Model Research Explorer - Basic Example")
    print("=" * 70)
    
    # Example 1: Create a standard 2D Ising model configuration
    print("\n1. Creating standard 2D Ising model configuration...")
    standard_2d = ModelVariantConfig(
        name="standard_2d_ising",
        dimensions=2,
        lattice_geometry="square",
        interaction_type="nearest_neighbor",
        theoretical_tc=2.269,  # Known Onsager solution
        theoretical_exponents={
            'beta': 0.125,
            'nu': 1.0,
            'gamma': 1.75,
            'alpha': 0.0
        }
    )
    print(f"   Created: {standard_2d.name}")
    print(f"   Dimensions: {standard_2d.dimensions}D")
    print(f"   Theoretical Tc: {standard_2d.theoretical_tc}")
    print(f"   Theoretical β: {standard_2d.theoretical_exponents['beta']}")
    
    # Example 2: Create a long-range interaction model
    print("\n2. Creating long-range Ising model configuration...")
    long_range = ModelVariantConfig(
        name="long_range_2d_ising",
        dimensions=2,
        lattice_geometry="square",
        interaction_type="long_range",
        interaction_params={'alpha': 2.5},  # J(r) ~ r^(-2.5)
        theoretical_tc=None,  # Unknown - to be discovered
        theoretical_exponents=None
    )
    print(f"   Created: {long_range.name}")
    print(f"   Interaction type: {long_range.interaction_type}")
    print(f"   Power-law exponent α: {long_range.interaction_params['alpha']}")
    print(f"   Theoretical properties: Unknown (to be discovered)")
    
    # Example 3: Create a disordered model
    print("\n3. Creating quenched disorder model configuration...")
    disordered = ModelVariantConfig(
        name="disordered_2d_ising",
        dimensions=2,
        lattice_geometry="square",
        interaction_type="nearest_neighbor",
        disorder_type="quenched",
        disorder_strength=0.2,
        theoretical_tc=None,  # Modified by disorder
    )
    print(f"   Created: {disordered.name}")
    print(f"   Disorder type: {disordered.disorder_type}")
    print(f"   Disorder strength: {disordered.disorder_strength}")
    
    # Example 4: Define a research hypothesis
    print("\n4. Defining a research hypothesis...")
    hypothesis = ResearchHypothesis(
        hypothesis_id="hyp_001",
        description="Long-range Ising with α=2.5 exhibits mean-field behavior",
        variant_id="long_range_2d_ising",
        parameter_ranges={'temperature': (1.0, 5.0)},
        predictions={
            'beta': 0.5,   # Mean-field prediction
            'nu': 0.5,     # Mean-field prediction
            'gamma': 1.0   # Mean-field prediction
        },
        prediction_errors={
            'beta': 0.05,
            'nu': 0.05,
            'gamma': 0.1
        },
        universality_class="mean_field"
    )
    print(f"   Hypothesis ID: {hypothesis.hypothesis_id}")
    print(f"   Description: {hypothesis.description}")
    print(f"   Predicted β: {hypothesis.predictions['beta']} ± {hypothesis.prediction_errors['beta']}")
    print(f"   Predicted ν: {hypothesis.predictions['nu']} ± {hypothesis.prediction_errors['nu']}")
    print(f"   Status: {hypothesis.status}")
    
    # Example 5: Create an exploration strategy
    print("\n5. Creating parameter space exploration strategy...")
    strategy = ExplorationStrategy(
        method='adaptive',
        n_points=50,
        refinement_iterations=2
    )
    print(f"   Method: {strategy.method}")
    print(f"   Number of points: {strategy.n_points}")
    print(f"   Refinement iterations: {strategy.refinement_iterations}")
    
    # Example 6: Create mock simulation data
    print("\n6. Creating mock simulation data structure...")
    n_temps = 20
    n_samples = 100
    lattice_size = 32
    
    sim_data = SimulationData(
        variant_id="long_range_2d_ising",
        parameters={'alpha': 2.5},
        temperatures=np.linspace(1.0, 5.0, n_temps),
        configurations=np.random.choice([-1, 1], size=(n_temps, n_samples, lattice_size, lattice_size)),
        magnetizations=np.random.randn(n_temps, n_samples),
        energies=np.random.randn(n_temps, n_samples),
        metadata={
            'lattice_size': lattice_size,
            'equilibration_steps': 10000,
            'measurement_steps': 1000
        }
    )
    print(f"   Variant: {sim_data.variant_id}")
    print(f"   Temperature range: {sim_data.temperatures[0]:.2f} - {sim_data.temperatures[-1]:.2f}")
    print(f"   Configuration shape: {sim_data.configurations.shape}")
    print(f"   Lattice size: {sim_data.metadata['lattice_size']}")
    
    # Example 7: Create a novel phenomenon detection
    print("\n7. Creating novel phenomenon detection...")
    phenomenon = NovelPhenomenon(
        phenomenon_type='anomalous_exponents',
        variant_id="long_range_2d_ising",
        parameters={'alpha': 2.5, 'temperature': 3.2},
        description="Detected critical exponents deviate >3σ from known universality classes",
        confidence=0.85,
        supporting_evidence={
            'measured_beta': 0.42,
            'expected_beta_range': (0.125, 0.326),
            'sigma_deviation': 3.5
        }
    )
    print(f"   Type: {phenomenon.phenomenon_type}")
    print(f"   Variant: {phenomenon.variant_id}")
    print(f"   Confidence: {phenomenon.confidence:.1%}")
    print(f"   Description: {phenomenon.description}")
    
    print("\n" + "=" * 70)
    print("Infrastructure demonstration complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Implement ModelVariantRegistry for managing variants")
    print("  - Implement HypothesisManager for tracking hypotheses")
    print("  - Implement DiscoveryPipeline for automated exploration")
    print("  - Create custom model plugins extending ModelVariantPlugin")


if __name__ == "__main__":
    main()
