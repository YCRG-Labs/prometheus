"""
Example demonstrating the Exploration Planner for discovery campaigns.

This example shows how to:
1. Create exploration plans for target variants
2. Optimize sampling strategies
3. Generate parameter points for exploration
4. Estimate computational costs
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research.target_selection_engine import TargetSelectionEngine
from src.research.exploration_planner import ExplorationPlanner
import logging


def main():
    """Run exploration planner examples."""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    print("=" * 80)
    print("EXPLORATION PLANNER DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Initialize target selection engine to get variants
    print("1. Loading target variants...")
    selector = TargetSelectionEngine()
    candidates = selector.identify_candidates()
    print(f"   Loaded {len(candidates)} candidate variants")
    print()
    
    # Initialize exploration planner
    print("2. Initializing exploration planner...")
    planner = ExplorationPlanner(
        default_lattice_sizes_2d=[32, 64],
        default_lattice_sizes_3d=[16, 32],
        cost_per_sample=0.01  # 0.01 GPU-seconds per sample
    )
    print("   Planner initialized")
    print()
    
    # Example 1: Create plan for long-range Ising
    print("=" * 80)
    print("EXAMPLE 1: Long-Range Ising Model (α=2.2)")
    print("=" * 80)
    
    # Get long-range variant
    long_range_variant = None
    for variant in candidates:
        if 'long_range_ising_alpha_2.2' in variant.variant_id:
            long_range_variant = variant
            break
    
    if long_range_variant:
        print(f"\nVariant: {long_range_variant.name}")
        print(f"Description: {long_range_variant.description}")
        print()
        
        # Create exploration plan
        print("Creating exploration plan...")
        plan = planner.create_plan(
            variant=long_range_variant,
            budget=200.0,  # 200 GPU-hours
            refinement_strategy='adaptive'
        )
        
        print(f"\nExploration Plan:")
        print(f"  Parameter ranges: {plan.parameter_ranges}")
        print(f"  Initial points: {plan.initial_points}")
        print(f"  Refinement strategy: {plan.refinement_strategy}")
        print(f"  Lattice sizes: {plan.lattice_sizes}")
        print(f"  Temperature points: {plan.temperature_points}")
        print(f"  Samples per temp: {plan.samples_per_temp}")
        print(f"  Temperature range: {plan.temperature_range}")
        print(f"  Estimated cost: {plan.estimated_cost:.1f} GPU-hours")
        print()
        
        # Optimize the plan
        print("Optimizing sampling strategy...")
        optimized_plan = planner.optimize_sampling(plan)
        
        print(f"\nOptimized Plan:")
        print(f"  Lattice sizes: {optimized_plan.lattice_sizes}")
        print(f"  Temperature points: {optimized_plan.temperature_points}")
        print(f"  Samples per temp: {optimized_plan.samples_per_temp}")
        print(f"  Estimated cost: {optimized_plan.estimated_cost:.1f} GPU-hours")
        print(f"  Cost reduction: {((plan.estimated_cost - optimized_plan.estimated_cost) / plan.estimated_cost * 100):.1f}%")
        print()
        
        # Generate parameter points
        print("Generating parameter points...")
        points = planner.generate_parameter_points(optimized_plan, strategy='latin_hypercube')
        
        print(f"\nGenerated {len(points)} parameter points")
        print("First 5 points:")
        for i, point in enumerate(points[:5], 1):
            print(f"  {i}. {point}")
        print()
    
    # Example 2: Create plan for diluted Ising
    print("=" * 80)
    print("EXAMPLE 2: Diluted Ising Model (p=0.60)")
    print("=" * 80)
    
    # Get diluted variant
    diluted_variant = None
    for variant in candidates:
        if 'diluted_ising_p_0.60' in variant.variant_id:
            diluted_variant = variant
            break
    
    if diluted_variant:
        print(f"\nVariant: {diluted_variant.name}")
        print(f"Description: {diluted_variant.description}")
        print()
        
        # Create exploration plan
        print("Creating exploration plan...")
        plan = planner.create_plan(
            variant=diluted_variant,
            budget=250.0,  # 250 GPU-hours
            refinement_strategy='grid'
        )
        
        print(f"\nExploration Plan:")
        print(f"  Parameter ranges: {plan.parameter_ranges}")
        print(f"  Initial points: {plan.initial_points}")
        print(f"  Refinement strategy: {plan.refinement_strategy}")
        print(f"  Lattice sizes: {plan.lattice_sizes}")
        print(f"  Temperature points: {plan.temperature_points}")
        print(f"  Samples per temp: {plan.samples_per_temp}")
        print(f"  Estimated cost: {plan.estimated_cost:.1f} GPU-hours")
        print()
        
        # Generate grid points
        print("Generating grid parameter points...")
        grid_points = planner.generate_parameter_points(plan, strategy='grid')
        
        print(f"\nGenerated {len(grid_points)} grid points")
        print("First 5 points:")
        for i, point in enumerate(grid_points[:5], 1):
            print(f"  {i}. {point}")
        print()
    
    # Example 3: Create plan for triangular antiferromagnet
    print("=" * 80)
    print("EXAMPLE 3: Triangular Antiferromagnet (J2/J1=0.6)")
    print("=" * 80)
    
    # Get triangular variant
    triangular_variant = None
    for variant in candidates:
        if 'triangular_afm_j2_j1_0.6' in variant.variant_id:
            triangular_variant = variant
            break
    
    if triangular_variant:
        print(f"\nVariant: {triangular_variant.name}")
        print(f"Description: {triangular_variant.description}")
        print()
        
        # Create exploration plan
        print("Creating exploration plan...")
        plan = planner.create_plan(
            variant=triangular_variant,
            budget=300.0,  # 300 GPU-hours
            refinement_strategy='adaptive'
        )
        
        print(f"\nExploration Plan:")
        print(f"  Parameter ranges: {plan.parameter_ranges}")
        print(f"  Initial points: {plan.initial_points}")
        print(f"  Refinement strategy: {plan.refinement_strategy}")
        print(f"  Lattice sizes: {plan.lattice_sizes}")
        print(f"  Temperature points: {plan.temperature_points}")
        print(f"  Samples per temp: {plan.samples_per_temp}")
        print(f"  Estimated cost: {plan.estimated_cost:.1f} GPU-hours")
        print()
        
        # Note: Frustrated systems get more samples automatically
        print("Note: Frustrated systems automatically receive:")
        print("  - Increased samples per temperature (for better equilibration)")
        print("  - More temperature points (to capture complex phase behavior)")
        print()
    
    # Example 4: Compare strategies
    print("=" * 80)
    print("EXAMPLE 4: Comparing Sampling Strategies")
    print("=" * 80)
    
    if long_range_variant:
        print(f"\nVariant: {long_range_variant.name}")
        print()
        
        strategies = ['grid', 'random', 'latin_hypercube']
        
        for strategy in strategies:
            print(f"\nStrategy: {strategy}")
            plan = planner.create_plan(
                variant=long_range_variant,
                initial_points=25,
                refinement_strategy=strategy
            )
            
            points = planner.generate_parameter_points(plan, strategy=strategy)
            
            print(f"  Generated {len(points)} points")
            print(f"  First 3 points:")
            for i, point in enumerate(points[:3], 1):
                print(f"    {i}. {point}")
        
        print()
    
    # Example 5: Budget-constrained planning
    print("=" * 80)
    print("EXAMPLE 5: Budget-Constrained Planning")
    print("=" * 80)
    
    budgets = [50.0, 100.0, 200.0, 500.0]
    
    if long_range_variant:
        print(f"\nVariant: {long_range_variant.name}")
        print()
        
        for budget in budgets:
            plan = planner.create_plan(
                variant=long_range_variant,
                budget=budget,
                refinement_strategy='adaptive'
            )
            
            print(f"Budget: {budget:.0f} GPU-hours")
            print(f"  Initial points: {plan.initial_points}")
            print(f"  Temperature points: {plan.temperature_points}")
            print(f"  Samples per temp: {plan.samples_per_temp}")
            print(f"  Estimated cost: {plan.estimated_cost:.1f} GPU-hours")
            print()
    
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("The exploration planner successfully:")
    print("  ✓ Created exploration plans for different variant types")
    print("  ✓ Optimized sampling strategies to reduce computational cost")
    print("  ✓ Generated parameter points using multiple strategies")
    print("  ✓ Adapted plans based on system complexity (disorder, frustration)")
    print("  ✓ Provided accurate cost estimates")
    print()


if __name__ == '__main__':
    main()
