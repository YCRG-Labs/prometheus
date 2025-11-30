"""
Example: Discovery Pipeline for Parameter Space Exploration

This example demonstrates how to use the Discovery Pipeline to systematically
explore parameter spaces of model variants and detect novel phenomena.
"""

import numpy as np
from pathlib import Path

from src.research import (
    ModelVariantRegistry,
    ModelVariantConfig,
    DiscoveryPipeline,
    DiscoveryConfig,
    ExplorationStrategy
)


def main():
    """Run discovery pipeline example."""
    
    print("=" * 80)
    print("Discovery Pipeline Example")
    print("=" * 80)
    
    # Step 1: Create model variant registry
    print("\n1. Setting up model variant registry...")
    registry = ModelVariantRegistry()
    
    # Register a standard 3D Ising model variant
    config = ModelVariantConfig(
        name='standard_3d_ising',
        dimensions=3,
        lattice_geometry='cubic',
        interaction_type='nearest_neighbor',
        theoretical_tc=4.511,
        theoretical_exponents={
            'beta': 0.326,
            'nu': 0.630,
            'gamma': 1.237
        }
    )
    
    variant_id = registry.register_variant(config)
    print(f"   Registered variant: {variant_id}")
    
    # Step 2: Configure exploration strategy
    print("\n2. Configuring exploration strategy...")
    exploration_strategy = ExplorationStrategy(
        method='grid',
        n_points=5,  # Small number for demonstration
        refinement_iterations=0
    )
    print(f"   Method: {exploration_strategy.method}")
    print(f"   Points: {exploration_strategy.n_points}")
    
    # Step 3: Configure discovery pipeline
    print("\n3. Configuring discovery pipeline...")
    discovery_config = DiscoveryConfig(
        variant_id=variant_id,
        exploration_strategy=exploration_strategy,
        simulation_params={
            'lattice_size': 16,  # Small for demonstration
            'n_temperatures': 10,
            'n_samples': 50,
            'n_equilibration': 500,
            'n_steps_between': 10,
            'seed': 42
        },
        vae_config={
            'latent_dim': 10,
            'learning_rate': 0.001,
            'n_epochs': 50
        },
        analysis_config={
            'fit_range_factor': 0.3,
            'min_points': 5,
            'anomaly_threshold': 3.0  # Threshold for novel phenomena detection
        },
        checkpoint_interval=2,
        output_dir='results/discovery_example'
    )
    print(f"   Variant: {discovery_config.variant_id}")
    print(f"   Output: {discovery_config.output_dir}")
    
    # Step 4: Create and run discovery pipeline
    print("\n4. Running discovery pipeline...")
    print("   (This may take a few minutes...)")
    
    pipeline = DiscoveryPipeline(discovery_config, registry)
    
    try:
        results = pipeline.run_exploration()
        
        # Step 5: Display results
        print("\n" + "=" * 80)
        print("Discovery Results")
        print("=" * 80)
        
        print(f"\nExploration Summary:")
        print(f"  Variant: {results.variant_id}")
        print(f"  Points explored: {results.n_points_explored}")
        print(f"  Execution time: {results.execution_time:.2f} seconds")
        
        print(f"\nNovel Phenomena Detected: {len(results.novel_phenomena)}")
        for i, phenomenon in enumerate(results.novel_phenomena, 1):
            print(f"\n  {i}. {phenomenon.phenomenon_type}")
            print(f"     Parameters: {phenomenon.parameters}")
            print(f"     Description: {phenomenon.description}")
            print(f"     Confidence: {phenomenon.confidence:.2%}")
        
        if results.vae_results:
            print(f"\nCritical Exponent Summary:")
            beta_values = [r.exponents.get('beta', np.nan) for r in results.vae_results]
            beta_values = [b for b in beta_values if not np.isnan(b)]
            
            if beta_values:
                print(f"  β exponent:")
                print(f"    Mean: {np.mean(beta_values):.4f}")
                print(f"    Std:  {np.std(beta_values):.4f}")
                print(f"    Range: [{np.min(beta_values):.4f}, {np.max(beta_values):.4f}]")
                print(f"    Theoretical: 0.326")
        
        print(f"\nResults saved to: {results.checkpoint_path}")
        
    except Exception as e:
        print(f"\nError during exploration: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Example Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
