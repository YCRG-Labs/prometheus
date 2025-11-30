"""
Example demonstrating the Model Variant Registry.

This script shows how to:
1. Create and register model variants
2. Retrieve and list variants
3. Create simulators for variants
4. Run basic Monte Carlo simulations
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.research import ModelVariantConfig, ModelVariantRegistry

def main():
    print("=" * 60)
    print("Model Variant Registry Example")
    print("=" * 60)
    
    # Initialize registry
    registry = ModelVariantRegistry()
    
    # Register standard 2D Ising model
    print("\n1. Registering standard 2D Ising model...")
    standard_2d = ModelVariantConfig(
        name='standard_2d_ising',
        dimensions=2,
        lattice_geometry='square',
        interaction_type='nearest_neighbor',
        interaction_params={'J': 1.0},
        theoretical_tc=2.269,
        theoretical_exponents={'beta': 0.125, 'nu': 1.0}
    )
    registry.register_variant(standard_2d)
    print(f"   Registered: {standard_2d.name}")
    
    # Register long-range 2D Ising model
    print("\n2. Registering long-range 2D Ising model...")
    long_range_2d = ModelVariantConfig(
        name='long_range_2d_ising',
        dimensions=2,
        lattice_geometry='square',
        interaction_type='long_range',
        interaction_params={'alpha': 2.5, 'J0': 1.0}
    )
    registry.register_variant(long_range_2d)
    print(f"   Registered: {long_range_2d.name}")
    
    # Register quenched disorder model
    print("\n3. Registering quenched disorder model...")
    disorder_2d = ModelVariantConfig(
        name='disorder_2d_ising',
        dimensions=2,
        lattice_geometry='square',
        interaction_type='nearest_neighbor',
        disorder_type='quenched',
        disorder_strength=0.2
    )
    registry.register_variant(disorder_2d)
    print(f"   Registered: {disorder_2d.name}")
    
    # Register triangular lattice model
    print("\n4. Registering triangular lattice model...")
    triangular_2d = ModelVariantConfig(
        name='triangular_2d_ising',
        dimensions=2,
        lattice_geometry='triangular',
        interaction_type='frustrated',
        interaction_params={'J': 1.0}
    )
    registry.register_variant(triangular_2d)
    print(f"   Registered: {triangular_2d.name}")
    
    # List all variants
    print("\n5. Listing all registered variants...")
    all_variants = registry.list_variants()
    print(f"   Total variants: {len(all_variants)}")
    for vid in all_variants:
        print(f"   - {vid}")
    
    # Filter variants by dimension
    print("\n6. Filtering 2D variants...")
    variants_2d = registry.list_variants(filters={'dimensions': 2})
    print(f"   Found {len(variants_2d)} 2D variants")
    
    # Filter variants with disorder
    print("\n7. Filtering variants with disorder...")
    disorder_variants = registry.list_variants(filters={'has_disorder': True})
    print(f"   Found {len(disorder_variants)} variants with disorder:")
    for vid in disorder_variants:
        print(f"   - {vid}")
    
    # Get variant information
    print("\n8. Getting detailed information for standard 2D Ising...")
    info = registry.get_variant_info('standard_2d_ising')
    print(f"   Name: {info['name']}")
    print(f"   Dimensions: {info['dimensions']}")
    print(f"   Geometry: {info['lattice_geometry']}")
    print(f"   Interaction: {info['interaction_type']}")
    print(f"   Has theoretical Tc: {info['has_theoretical_tc']}")
    print(f"   Has theoretical exponents: {info['has_theoretical_exponents']}")
    
    # Create a simulator (this will work once we implement the model plugins)
    print("\n9. Creating simulator for standard 2D Ising...")
    try:
        simulator = registry.create_simulator(
            variant_id='standard_2d_ising',
            lattice_size=16,
            temperature=2.5,
            seed=42
        )
        print(f"   Simulator created successfully!")
        print(f"   Lattice size: {simulator.lattice_size}")
        print(f"   Temperature: {simulator.temperature}")
        
        # Run a short equilibration
        print("\n10. Running short equilibration (100 MC steps)...")
        simulator.equilibrate(n_steps=100)
        state = simulator.get_current_state()
        print(f"   Energy: {state['energy']:.2f}")
        print(f"   Magnetization: {state['magnetization']:.4f}")
        
    except Exception as e:
        print(f"   Error creating simulator: {e}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)

if __name__ == '__main__':
    main()

