"""
Example demonstrating the Target Selection Engine for discovery campaigns.

This example shows how to:
1. Initialize the target selection engine
2. Load candidate variants
3. Score discovery potential
4. Prioritize variants within budget
5. Generate selection reports
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research.target_selection_engine import (
    TargetSelectionEngine,
    SelectionCriteria,
)
from src.research.campaign_orchestrator import TargetVariant
from src.research.base_types import ModelVariantConfig


def main():
    """Run target selection example."""
    print("=" * 80)
    print("TARGET SELECTION ENGINE EXAMPLE")
    print("=" * 80)
    print()
    
    # Example 1: Initialize with default variants
    print("Example 1: Initialize with default candidate variants")
    print("-" * 80)
    
    engine = TargetSelectionEngine()
    
    candidates = engine.identify_candidates()
    print(f"Identified {len(candidates)} candidate variants")
    print()
    
    # Example 2: Score discovery potential
    print("Example 2: Score discovery potential for each variant")
    print("-" * 80)
    
    for variant in candidates[:5]:  # Show first 5
        score = engine.score_discovery_potential(variant)
        print(f"{variant.name:40s} Score: {score:.3f}")
    print()
    
    # Example 3: Prioritize within budget
    print("Example 3: Prioritize variants within computational budget")
    print("-" * 80)
    
    budget = 1000.0  # 1000 GPU-hours
    prioritized = engine.prioritize_targets(candidates, budget=budget)
    
    print(f"Budget: {budget:.1f} GPU-hours")
    print(f"Selected: {len(prioritized)} variants")
    print()
    
    print("Top 10 prioritized variants:")
    for i, variant in enumerate(prioritized[:10], 1):
        print(f"{i:2d}. {variant.name:40s} "
              f"Priority: {variant.priority_score:.3f} "
              f"Cost: {variant.estimated_compute_hours:.1f}h")
    print()
    
    # Example 4: Custom selection criteria
    print("Example 4: Custom selection criteria (emphasize novelty)")
    print("-" * 80)
    
    custom_criteria = SelectionCriteria(
        theoretical_weight=0.2,
        literature_gap_weight=0.2,
        feasibility_weight=0.1,
        novelty_weight=0.5,  # Emphasize novelty
    )
    
    engine_custom = TargetSelectionEngine(criteria=custom_criteria)
    prioritized_custom = engine_custom.prioritize_targets(budget=1000.0)
    
    print("Top 5 with novelty emphasis:")
    for i, variant in enumerate(prioritized_custom[:5], 1):
        print(f"{i}. {variant.name:40s} "
              f"Priority: {variant.priority_score:.3f}")
    print()
    
    # Example 5: Add custom variant
    print("Example 5: Add custom variant to database")
    print("-" * 80)
    
    custom_variant = TargetVariant(
        variant_id="custom_kagome_ising",
        name="Kagome Lattice Ising",
        description="Antiferromagnetic Ising on kagome lattice with extreme frustration",
        model_config=ModelVariantConfig(
            name="kagome_ising",
            dimensions=2,
            lattice_geometry='kagome',
            interaction_type='frustrated',
            interaction_params={'j2_j1_ratio': 1.0},
        ),
        theoretical_predictions=None,  # Unknown
        literature_references=[
            "Moessner & Chalker, Phys. Rev. B 58, 12049 (1998)",
        ],
        estimated_compute_hours=250.0,
    )
    
    engine.add_variant(custom_variant)
    score = engine.score_discovery_potential(custom_variant)
    print(f"Added: {custom_variant.name}")
    print(f"Discovery potential score: {score:.3f}")
    print()
    
    # Example 6: Generate selection report
    print("Example 6: Generate selection report")
    print("-" * 80)
    
    output_dir = Path('results/target_selection_demo')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / 'selection_report.txt'
    engine.generate_selection_report(prioritized[:10], str(report_path))
    print(f"Generated report: {report_path}")
    print()
    
    # Example 7: Save and load variant database
    print("Example 7: Save and load variant database")
    print("-" * 80)
    
    db_path = output_dir / 'variant_database.json'
    engine.save_variant_database(str(db_path))
    print(f"Saved database: {db_path}")
    
    # Load it back
    engine_loaded = TargetSelectionEngine(variant_database_path=str(db_path))
    loaded_candidates = engine_loaded.identify_candidates()
    print(f"Loaded {len(loaded_candidates)} variants from database")
    print()
    
    # Example 8: Budget-constrained selection
    print("Example 8: Compare different budget scenarios")
    print("-" * 80)
    
    budgets = [500, 1000, 2000, 5000]
    
    for budget in budgets:
        selected = engine.prioritize_targets(budget=budget)
        total_cost = sum(v.estimated_compute_hours for v in selected)
        print(f"Budget: {budget:5.0f}h → Selected: {len(selected):2d} variants "
              f"(Total cost: {total_cost:.1f}h)")
    print()
    
    print("=" * 80)
    print("TARGET SELECTION ENGINE EXAMPLE COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
