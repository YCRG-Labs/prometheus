"""
Example demonstrating the Dual-Purpose Universality Class Manager.

This script shows how the same universality class database serves both:
1. Detection: Finding deviations from known classes (novel physics)
2. Validation: Confirming predictions match known classes (hypothesis testing)

The manager tracks usage patterns and optimizes the database based on
which classes are most useful for discovery.
"""

import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research.universality_class_manager import (
    UniversalityClassManager,
    UniversalityClass
)
from src.research.base_types import VAEAnalysisResults


def example_1_basic_detection_tracking():
    """Example 1: Track usage for anomaly detection."""
    print("=" * 80)
    print("Example 1: Basic Detection Tracking")
    print("=" * 80)
    
    # Initialize manager
    manager = UniversalityClassManager(load_history=False)
    
    # Simulate detection on 2D Ising variant
    print("\n1. Detecting anomalies in 2D Ising variant...")
    
    # Standard 2D Ising (should match)
    record1 = manager.record_detection_use(
        class_name='ising_2d',
        variant_id='standard_2d_ising',
        exponent_name='beta',
        measured_value=0.123,
        measured_error=0.003,
        threshold=3.0
    )
    print(f"   β = 0.123 ± 0.003 vs theory 0.125")
    print(f"   Deviation: {record1.deviation:.2f}σ, Matched: {record1.matched}")
    
    # Anomalous variant (should not match)
    record2 = manager.record_detection_use(
        class_name='ising_2d',
        variant_id='long_range_2d_ising',
        exponent_name='beta',
        measured_value=0.35,
        measured_error=0.02,
        threshold=3.0
    )
    print(f"\n   β = 0.35 ± 0.02 vs theory 0.125")
    print(f"   Deviation: {record2.deviation:.2f}σ, Matched: {record2.matched}")
    print(f"   → Anomaly detected! Confidence: {record2.confidence:.2%}")
    
    # Get statistics
    stats = manager.get_statistics('ising_2d')['ising_2d']
    print(f"\n2. Statistics for 2D Ising class:")
    print(f"   Total uses: {stats.total_uses}")
    print(f"   Detection uses: {stats.detection_uses}")
    print(f"   Match rate: {stats.match_rate:.2%}")
    print(f"   Avg deviation: {stats.avg_deviation:.2f}σ")


def example_2_validation_tracking():
    """Example 2: Track usage for hypothesis validation."""
    print("\n" + "=" * 80)
    print("Example 2: Validation Tracking")
    print("=" * 80)
    
    # Initialize manager
    manager = UniversalityClassManager(load_history=False)
    
    # Hypothesis: Long-range Ising with α=2.5 belongs to mean field class
    print("\n1. Validating hypothesis: Long-range Ising → Mean Field class")
    
    predicted_exponents = {
        'beta': 0.5,
        'nu': 0.5,
        'gamma': 1.0
    }
    
    measured_exponents = {
        'beta': 0.48,
        'nu': 0.52,
        'gamma': 0.95
    }
    
    measured_errors = {
        'beta': 0.03,
        'nu': 0.04,
        'gamma': 0.08
    }
    
    records = manager.record_validation_use(
        class_name='mean_field',
        variant_id='long_range_ising_alpha_2.5',
        predicted_exponents=predicted_exponents,
        measured_exponents=measured_exponents,
        measured_errors=measured_errors,
        threshold=2.0
    )
    
    print("\n   Exponent comparisons:")
    for record in records:
        print(f"   {record.exponent_name}: deviation = {record.deviation:.2f}σ, "
              f"matched = {record.matched}, confidence = {record.confidence:.2%}")
    
    # Check if hypothesis validated
    all_matched = all(r.matched for r in records)
    avg_confidence = np.mean([r.confidence for r in records])
    
    print(f"\n2. Hypothesis validation result:")
    print(f"   All exponents matched: {all_matched}")
    print(f"   Average confidence: {avg_confidence:.2%}")
    print(f"   → Hypothesis {'VALIDATED' if all_matched else 'REFUTED'}")
    
    # Get statistics
    stats = manager.get_statistics('mean_field')['mean_field']
    print(f"\n3. Statistics for Mean Field class:")
    print(f"   Total uses: {stats.total_uses}")
    print(f"   Validation uses: {stats.validation_uses}")
    print(f"   Match rate: {stats.match_rate:.2%}")


def example_3_dual_purpose_usage():
    """Example 3: Demonstrate dual-purpose nature."""
    print("\n" + "=" * 80)
    print("Example 3: Dual-Purpose Usage")
    print("=" * 80)
    
    # Initialize manager
    manager = UniversalityClassManager(load_history=False)
    
    print("\n1. Using 3D Ising class for DETECTION...")
    
    # Detection: Check if variant deviates from 3D Ising
    manager.record_detection_use(
        class_name='ising_3d',
        variant_id='standard_3d_ising',
        exponent_name='beta',
        measured_value=0.320,
        measured_error=0.007,
        threshold=3.0
    )
    
    manager.record_detection_use(
        class_name='ising_3d',
        variant_id='standard_3d_ising',
        exponent_name='nu',
        measured_value=0.666,
        measured_error=0.013,
        threshold=3.0
    )
    
    print("   Checked standard 3D Ising for anomalies")
    
    print("\n2. Using 3D Ising class for VALIDATION...")
    
    # Validation: Confirm hypothesis that variant belongs to 3D Ising
    manager.record_validation_use(
        class_name='ising_3d',
        variant_id='standard_3d_ising',
        predicted_exponents={'beta': 0.326, 'nu': 0.630},
        measured_exponents={'beta': 0.320, 'nu': 0.666},
        measured_errors={'beta': 0.007, 'nu': 0.013},
        threshold=2.0
    )
    
    print("   Validated hypothesis that variant belongs to 3D Ising class")
    
    # Show dual-purpose statistics
    stats = manager.get_statistics('ising_3d')['ising_3d']
    print(f"\n3. Dual-purpose statistics for 3D Ising:")
    print(f"   Total uses: {stats.total_uses}")
    print(f"   Detection uses: {stats.detection_uses} ({stats.detection_uses/stats.total_uses:.1%})")
    print(f"   Validation uses: {stats.validation_uses} ({stats.validation_uses/stats.total_uses:.1%})")
    print(f"   Discovery value: {stats.discovery_value:.3f}")
    
    print("\n   → Same theoretical knowledge serves both purposes!")


def example_4_discovery_value_ranking():
    """Example 4: Rank classes by discovery value."""
    print("\n" + "=" * 80)
    print("Example 4: Discovery Value Ranking")
    print("=" * 80)
    
    # Initialize manager
    manager = UniversalityClassManager(load_history=False)
    
    # Simulate diverse usage patterns
    print("\n1. Simulating usage across multiple classes...")
    
    # 2D Ising: High usage, balanced detection/validation
    for i in range(10):
        manager.record_detection_use(
            'ising_2d', f'variant_{i}', 'beta',
            0.125 + np.random.normal(0, 0.02), 0.003, 3.0
        )
        manager.record_validation_use(
            'ising_2d', f'variant_{i}',
            {'beta': 0.125}, {'beta': 0.125 + np.random.normal(0, 0.02)},
            {'beta': 0.003}, 2.0
        )
    
    # 3D Ising: High usage, mostly detection
    for i in range(15):
        manager.record_detection_use(
            'ising_3d', f'variant_{i}', 'beta',
            0.326 + np.random.normal(0, 0.05), 0.007, 3.0
        )
    
    # Mean field: Moderate usage, high discrimination
    for i in range(5):
        # Half match, half don't
        value = 0.5 if i % 2 == 0 else 0.3
        manager.record_detection_use(
            'mean_field', f'variant_{i}', 'beta',
            value, 0.02, 3.0
        )
    
    # XY classes: Low usage
    manager.record_detection_use(
        'xy_2d', 'xy_variant', 'beta', 0.23, 0.02, 3.0
    )
    
    print("   Simulated 30+ usage records across 4 classes")
    
    # Rank by discovery value
    print("\n2. Classes ranked by discovery value:")
    most_useful = manager.get_most_useful_classes(n=5)
    
    for i, (class_name, value) in enumerate(most_useful, 1):
        stats = manager.get_statistics(class_name)[class_name]
        print(f"\n   {i}. {class_name}")
        print(f"      Discovery value: {value:.3f}")
        print(f"      Total uses: {stats.total_uses}")
        print(f"      Match rate: {stats.match_rate:.2%}")
        print(f"      Avg deviation: {stats.avg_deviation:.2f}σ")
        print(f"      Detection/Validation: {stats.detection_uses}/{stats.validation_uses}")


def example_5_usage_summary():
    """Example 5: Get comprehensive usage summary."""
    print("\n" + "=" * 80)
    print("Example 5: Usage Summary")
    print("=" * 80)
    
    # Initialize manager
    manager = UniversalityClassManager(load_history=False)
    
    # Simulate realistic usage
    print("\n1. Simulating realistic research workflow...")
    
    # Exploration phase: Mostly detection
    for i in range(20):
        class_name = np.random.choice(['ising_2d', 'ising_3d', 'mean_field'])
        manager.record_detection_use(
            class_name, f'exploration_variant_{i}', 'beta',
            np.random.uniform(0.1, 0.5), 0.01, 3.0
        )
    
    # Validation phase: Mostly validation
    for i in range(10):
        class_name = np.random.choice(['ising_2d', 'ising_3d'])
        manager.record_validation_use(
            class_name, f'validation_variant_{i}',
            {'beta': 0.3}, {'beta': 0.3 + np.random.normal(0, 0.02)},
            {'beta': 0.01}, 2.0
        )
    
    print("   Simulated 30 usage records (20 detection, 10 validation)")
    
    # Get summary
    summary = manager.get_usage_summary()
    
    print("\n2. Usage Summary:")
    print(f"   Total uses: {summary['total_uses']}")
    print(f"   Detection uses: {summary['detection_uses']} ({summary['detection_ratio']:.1%})")
    print(f"   Validation uses: {summary['validation_uses']}")
    print(f"   Classes used: {summary['num_used_classes']}/{summary['num_classes']}")
    
    print("\n3. Most used classes:")
    for class_name, uses in summary['most_used_classes']:
        print(f"   {class_name}: {uses} uses")
    
    print("\n4. Most discriminating classes:")
    for class_name, match_rate in summary['most_discriminating_classes']:
        print(f"   {class_name}: {match_rate:.2%} match rate")
    
    print("\n5. Most useful for discovery:")
    for class_name, value in summary['most_useful_for_discovery']:
        print(f"   {class_name}: {value:.3f} discovery value")


def example_6_database_optimization():
    """Example 6: Optimize database based on usage."""
    print("\n" + "=" * 80)
    print("Example 6: Database Optimization")
    print("=" * 80)
    
    # Initialize manager
    manager = UniversalityClassManager(load_history=False)
    
    # Simulate usage with clear patterns
    print("\n1. Simulating usage with optimization opportunities...")
    
    # High-value classes: 2D and 3D Ising (heavily used)
    for i in range(20):
        manager.record_detection_use(
            'ising_2d', f'variant_{i}', 'beta',
            0.125 + np.random.normal(0, 0.02), 0.003, 3.0
        )
        manager.record_detection_use(
            'ising_3d', f'variant_{i}', 'beta',
            0.326 + np.random.normal(0, 0.02), 0.007, 3.0
        )
    
    # Underutilized classes: XY models (rarely used)
    manager.record_detection_use(
        'xy_2d', 'xy_variant', 'beta', 0.23, 0.02, 3.0
    )
    
    # Coverage gap: Long-range variant that doesn't match any class
    for i in range(5):
        manager.record_detection_use(
            'mean_field', f'long_range_variant_{i}', 'beta',
            0.35, 0.02, 3.0  # Doesn't match mean field (0.5)
        )
    
    print("   Simulated 46 usage records with clear patterns")
    
    # Optimize database
    print("\n2. Analyzing database optimization opportunities...")
    optimization = manager.optimize_database(min_uses=5)
    
    print(f"\n3. Underutilized classes (< 5 uses):")
    for class_name, uses in optimization['underutilized_classes']:
        print(f"   {class_name}: {uses} uses")
    
    print(f"\n4. High-value classes (discovery value > 0.5):")
    for class_name, value in optimization['high_value_classes']:
        print(f"   {class_name}: {value:.3f}")
    
    print(f"\n5. Coverage gaps (variants not matching any class):")
    for gap in optimization['coverage_gaps']:
        print(f"   {gap['variant_id']}: best match {gap['best_match_deviation']:.2f}σ")
    
    print(f"\n6. Recommendation:")
    print(f"   {optimization['recommendation']}")


def example_7_save_and_load_history():
    """Example 7: Save and load usage history."""
    print("\n" + "=" * 80)
    print("Example 7: Save and Load History")
    print("=" * 80)
    
    # Create temporary directory
    temp_dir = Path('results/temp_universality_test')
    temp_dir.mkdir(parents=True, exist_ok=True)
    history_file = temp_dir / 'usage_history.json'
    
    # Initialize manager and record some usage
    print("\n1. Creating manager and recording usage...")
    manager1 = UniversalityClassManager(
        load_history=False,
        history_file=history_file
    )
    
    for i in range(5):
        manager1.record_detection_use(
            'ising_2d', f'variant_{i}', 'beta',
            0.125 + np.random.normal(0, 0.01), 0.003, 3.0
        )
    
    stats1 = manager1.get_statistics('ising_2d')['ising_2d']
    print(f"   Recorded {stats1.total_uses} uses")
    
    # Save history
    print(f"\n2. Saving history to {history_file}...")
    manager1.save_history()
    
    # Load history in new manager
    print("\n3. Loading history in new manager...")
    manager2 = UniversalityClassManager(
        load_history=True,
        history_file=history_file
    )
    
    stats2 = manager2.get_statistics('ising_2d')['ising_2d']
    print(f"   Loaded {stats2.total_uses} uses")
    print(f"   Match rate: {stats2.match_rate:.2%}")
    print(f"   Avg deviation: {stats2.avg_deviation:.2f}σ")
    
    # Verify consistency
    assert stats1.total_uses == stats2.total_uses
    assert abs(stats1.match_rate - stats2.match_rate) < 0.01
    print("\n   ✓ History successfully saved and loaded!")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("DUAL-PURPOSE UNIVERSALITY CLASS MANAGER EXAMPLES")
    print("=" * 80)
    print("\nDemonstrating how the same theoretical knowledge serves both")
    print("discovery (detecting deviations) and validation (confirming predictions)")
    
    example_1_basic_detection_tracking()
    example_2_validation_tracking()
    example_3_dual_purpose_usage()
    example_4_discovery_value_ranking()
    example_5_usage_summary()
    example_6_database_optimization()
    example_7_save_and_load_history()
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("""
1. DUAL PURPOSE: Same universality class database serves both:
   - Detection: Find deviations from known classes (novel physics)
   - Validation: Confirm predictions match known classes (hypothesis testing)

2. USAGE TRACKING: Manager tracks when each class is used for each purpose,
   enabling analysis of which classes are most useful for discovery.

3. DISCOVERY VALUE: Computed metric combining:
   - Usage frequency (more used = more relevant)
   - Discrimination power (match rate near 50% = most informative)
   - Anomaly detection (higher deviations = finds more anomalies)
   - Balance (used for both purposes = more versatile)

4. DATABASE OPTIMIZATION: Usage patterns reveal:
   - Underutilized classes (candidates for removal)
   - High-value classes (should be prioritized)
   - Coverage gaps (suggest new classes to add)

5. SYNERGY: The dual-purpose nature creates powerful synergy:
   - Detection identifies interesting variants
   - Validation confirms they belong to known/novel classes
   - Same theoretical knowledge serves both workflows
   - Optimization improves both detection and validation

This approach is more efficient than maintaining separate databases
for detection and validation, and enables continuous improvement based
on actual research usage patterns.
""")
    
    print("=" * 80)
    print("Examples completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
