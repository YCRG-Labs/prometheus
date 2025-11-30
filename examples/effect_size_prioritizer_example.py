"""
Example demonstrating the Effect Size Prioritization System.

This script shows how to prioritize research findings based on both
statistical significance (p-value) and practical importance (effect size),
preventing false positives and false negatives.
"""

import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research.effect_size_prioritizer import (
    EffectSizePrioritizer,
    Finding,
    FindingCategory
)


def example_1_basic_prioritization():
    """Example 1: Basic finding prioritization."""
    print("=" * 80)
    print("Example 1: Basic Finding Prioritization")
    print("=" * 80)
    
    # Initialize prioritizer
    prioritizer = EffectSizePrioritizer(alpha=0.05)
    
    print("\n1. Adding findings with different characteristics...")
    
    # Finding 1: Strong evidence (significant + large effect)
    finding1 = Finding(
        finding_id="F1",
        variant_id="long_range_ising",
        exponent_name="beta",
        measured_value=0.45,
        predicted_value=0.125,
        p_value=0.001,
        effect_size=1.5,
        sample_size=100,
        description="Long-range Ising shows mean-field behavior"
    )
    
    pf1 = prioritizer.add_finding(finding1)
    print(f"\n   Finding 1: p={finding1.p_value:.4f}, d={finding1.effect_size:.2f}")
    print(f"   Category: {pf1.finding_category.value}")
    print(f"   Priority: {pf1.priority_score:.1f}/100")
    
    # Finding 2: Investigate further (large effect but not significant)
    finding2 = Finding(
        finding_id="F2",
        variant_id="frustrated_lattice",
        exponent_name="nu",
        measured_value=0.75,
        predicted_value=0.63,
        p_value=0.15,
        effect_size=0.9,
        sample_size=30,
        description="Frustrated lattice shows different exponent"
    )
    
    pf2 = prioritizer.add_finding(finding2)
    print(f"\n   Finding 2: p={finding2.p_value:.4f}, d={finding2.effect_size:.2f}")
    print(f"   Category: {pf2.finding_category.value}")
    print(f"   Priority: {pf2.priority_score:.1f}/100")
    
    # Finding 3: False positive risk (significant but small effect)
    finding3 = Finding(
        finding_id="F3",
        variant_id="standard_2d",
        exponent_name="gamma",
        measured_value=1.76,
        predicted_value=1.75,
        p_value=0.02,
        effect_size=0.15,
        sample_size=200,
        description="Slight deviation in gamma"
    )
    
    pf3 = prioritizer.add_finding(finding3)
    print(f"\n   Finding 3: p={finding3.p_value:.4f}, d={finding3.effect_size:.2f}")
    print(f"   Category: {pf3.finding_category.value}")
    print(f"   Priority: {pf3.priority_score:.1f}/100")
    
    # Finding 4: No evidence (not significant + small effect)
    finding4 = Finding(
        finding_id="F4",
        variant_id="standard_3d",
        exponent_name="alpha",
        measured_value=0.11,
        predicted_value=0.11,
        p_value=0.80,
        effect_size=0.05,
        sample_size=50,
        description="Alpha matches prediction"
    )
    
    pf4 = prioritizer.add_finding(finding4)
    print(f"\n   Finding 4: p={finding4.p_value:.4f}, d={finding4.effect_size:.2f}")
    print(f"   Category: {pf4.finding_category.value}")
    print(f"   Priority: {pf4.priority_score:.1f}/100")
    
    print("\n2. Priority ranking:")
    top_findings = prioritizer.get_top_priorities(n=4)
    for i, pf in enumerate(top_findings, 1):
        print(f"\n   {i}. {pf.finding.finding_id} (Priority: {pf.priority_score:.1f})")
        print(f"      {pf.finding.description}")
        print(f"      Category: {pf.finding_category.value}")


def example_2_four_quadrants():
    """Example 2: Demonstrate the four quadrants."""
    print("\n" + "=" * 80)
    print("Example 2: Four Quadrants of Findings")
    print("=" * 80)
    
    prioritizer = EffectSizePrioritizer()
    
    print("\n1. Creating findings in each quadrant...")
    
    # Quadrant 1: High significance + Large effect (STRONG EVIDENCE)
    print("\n   Quadrant 1: High Significance + Large Effect")
    f1 = Finding(
        "Q1", "variant_1", "beta", 0.5, 0.125, 0.001, 1.8, 100,
        "Strong evidence for novel physics"
    )
    pf1 = prioritizer.add_finding(f1)
    print(f"   → {pf1.finding_category.value}")
    print(f"   → Priority: {pf1.priority_score:.1f}")
    
    # Quadrant 2: Low significance + Large effect (INVESTIGATE FURTHER)
    print("\n   Quadrant 2: Low Significance + Large Effect")
    f2 = Finding(
        "Q2", "variant_2", "nu", 0.8, 0.63, 0.20, 1.2, 25,
        "Large effect but needs more data"
    )
    pf2 = prioritizer.add_finding(f2)
    print(f"   → {pf2.finding_category.value}")
    print(f"   → Priority: {pf2.priority_score:.1f}")
    
    # Quadrant 3: High significance + Small effect (FALSE POSITIVE RISK)
    print("\n   Quadrant 3: High Significance + Small Effect")
    f3 = Finding(
        "Q3", "variant_3", "gamma", 1.76, 1.75, 0.01, 0.18, 500,
        "Significant but not important"
    )
    pf3 = prioritizer.add_finding(f3)
    print(f"   → {pf3.finding_category.value}")
    print(f"   → Priority: {pf3.priority_score:.1f}")
    
    # Quadrant 4: Low significance + Small effect (NO EVIDENCE)
    print("\n   Quadrant 4: Low Significance + Small Effect")
    f4 = Finding(
        "Q4", "variant_4", "alpha", 0.11, 0.11, 0.75, 0.08, 50,
        "No evidence of difference"
    )
    pf4 = prioritizer.add_finding(f4)
    print(f"   → {pf4.finding_category.value}")
    print(f"   → Priority: {pf4.priority_score:.1f}")
    
    print("\n2. Key Insight:")
    print("   The four quadrants represent different research scenarios:")
    print("   - Q1 (Strong Evidence): Publish and follow up")
    print("   - Q2 (Investigate Further): Collect more data")
    print("   - Q3 (False Positive Risk): Verify independently")
    print("   - Q4 (No Evidence): Low priority")


def example_3_false_positive_prevention():
    """Example 3: Preventing false positives."""
    print("\n" + "=" * 80)
    print("Example 3: Preventing False Positives")
    print("=" * 80)
    
    prioritizer = EffectSizePrioritizer()
    
    print("\n1. Scenario: Large sample size with tiny effect...")
    
    # With large sample, even tiny effects can be "significant"
    finding = Finding(
        "FP1",
        "standard_2d_large_sample",
        "beta",
        measured_value=0.126,
        predicted_value=0.125,
        p_value=0.001,  # Highly significant!
        effect_size=0.05,  # But negligible effect
        sample_size=10000,
        description="Tiny deviation with huge sample"
    )
    
    pf = prioritizer.add_finding(finding)
    
    print(f"\n   Measured: {finding.measured_value:.3f}")
    print(f"   Predicted: {finding.predicted_value:.3f}")
    print(f"   Difference: {abs(finding.measured_value - finding.predicted_value):.3f}")
    print(f"   P-value: {finding.p_value:.4f} (highly significant!)")
    print(f"   Effect size: {finding.effect_size:.2f} (negligible)")
    print(f"   Sample size: {finding.sample_size}")
    
    print(f"\n2. Prioritizer Classification:")
    print(f"   Category: {pf.finding_category.value}")
    print(f"   Priority: {pf.priority_score:.1f}/100")
    print(f"\n   {pf.recommendation}")
    
    print("\n3. Interpretation:")
    print("   Without effect size, this would be flagged as important")
    print("   (p < 0.001). But the effect is too small to matter.")
    print("   Prioritizer correctly identifies this as FALSE POSITIVE RISK.")


def example_4_false_negative_prevention():
    """Example 4: Preventing false negatives."""
    print("\n" + "=" * 80)
    print("Example 4: Preventing False Negatives")
    print("=" * 80)
    
    prioritizer = EffectSizePrioritizer()
    
    print("\n1. Scenario: Small sample size with large effect...")
    
    # With small sample, even large effects may not be "significant"
    finding = Finding(
        "FN1",
        "novel_variant_small_sample",
        "beta",
        measured_value=0.45,
        predicted_value=0.125,
        p_value=0.08,  # Not significant at α=0.05
        effect_size=1.5,  # But huge effect!
        sample_size=15,
        description="Large deviation with small sample"
    )
    
    pf = prioritizer.add_finding(finding)
    
    print(f"\n   Measured: {finding.measured_value:.3f}")
    print(f"   Predicted: {finding.predicted_value:.3f}")
    print(f"   Difference: {abs(finding.measured_value - finding.predicted_value):.3f}")
    print(f"   P-value: {finding.p_value:.4f} (not significant)")
    print(f"   Effect size: {finding.effect_size:.2f} (very large!)")
    print(f"   Sample size: {finding.sample_size}")
    
    print(f"\n2. Prioritizer Classification:")
    print(f"   Category: {pf.finding_category.value}")
    print(f"   Priority: {pf.priority_score:.1f}/100")
    print(f"\n   {pf.recommendation}")
    
    print("\n3. Interpretation:")
    print("   Without effect size, this would be dismissed as not")
    print("   significant (p > 0.05). But the effect is huge!")
    print("   Prioritizer correctly identifies this as INVESTIGATE FURTHER.")
    print("   → Collect more data to increase statistical power.")


def example_5_investigation_plan():
    """Example 5: Generate investigation plan."""
    print("\n" + "=" * 80)
    print("Example 5: Investigation Plan Generation")
    print("=" * 80)
    
    prioritizer = EffectSizePrioritizer()
    
    print("\n1. Adding 10 diverse findings...")
    
    # Simulate diverse findings
    np.random.seed(42)
    
    for i in range(10):
        # Random p-value and effect size
        p_value = np.random.uniform(0.001, 0.5)
        effect_size = np.random.uniform(0.05, 2.0)
        sample_size = np.random.randint(20, 200)
        
        finding = Finding(
            finding_id=f"F{i+1}",
            variant_id=f"variant_{i+1}",
            exponent_name="beta",
            measured_value=0.3 + np.random.normal(0, 0.1),
            predicted_value=0.3,
            p_value=p_value,
            effect_size=effect_size,
            sample_size=sample_size,
            description=f"Finding {i+1}"
        )
        
        prioritizer.add_finding(finding)
    
    print(f"   Added {len(prioritizer.findings)} findings")
    
    # Generate investigation plan
    print("\n2. Investigation Plan:")
    plan = prioritizer.generate_investigation_plan()
    
    print(f"\n   A. IMMEDIATE INVESTIGATION ({len(plan['immediate_investigation'])} findings)")
    print("      Strong evidence - publish and follow up")
    for pf in plan['immediate_investigation'][:3]:
        print(f"      - {pf.finding.finding_id}: p={pf.finding.p_value:.4f}, d={pf.finding.effect_size:.2f}")
    
    print(f"\n   B. COLLECT MORE DATA ({len(plan['collect_more_data'])} findings)")
    print("      Large effects but need more samples")
    for pf in plan['collect_more_data'][:3]:
        print(f"      - {pf.finding.finding_id}: p={pf.finding.p_value:.4f}, d={pf.finding.effect_size:.2f}")
    
    print(f"\n   C. VERIFY INDEPENDENTLY ({len(plan['verify_independently'])} findings)")
    print("      Significant but small effects - check for artifacts")
    for pf in plan['verify_independently'][:3]:
        print(f"      - {pf.finding.finding_id}: p={pf.finding.p_value:.4f}, d={pf.finding.effect_size:.2f}")
    
    print(f"\n   D. LOW PRIORITY ({len(plan['low_priority'])} findings)")
    print("      No evidence - deprioritize")
    for pf in plan['low_priority'][:3]:
        print(f"      - {pf.finding.finding_id}: p={pf.finding.p_value:.4f}, d={pf.finding.effect_size:.2f}")


def example_6_summary_statistics():
    """Example 6: Summary statistics."""
    print("\n" + "=" * 80)
    print("Example 6: Summary Statistics")
    print("=" * 80)
    
    prioritizer = EffectSizePrioritizer()
    
    # Add diverse findings
    np.random.seed(42)
    
    print("\n1. Adding 50 findings with realistic distribution...")
    
    for i in range(50):
        # Realistic distribution: most findings are null or small effects
        if i < 5:  # 10% strong evidence
            p_value = np.random.uniform(0.001, 0.01)
            effect_size = np.random.uniform(0.8, 2.0)
        elif i < 15:  # 20% investigate further
            p_value = np.random.uniform(0.05, 0.5)
            effect_size = np.random.uniform(0.6, 1.5)
        elif i < 25:  # 20% false positive risk
            p_value = np.random.uniform(0.001, 0.05)
            effect_size = np.random.uniform(0.05, 0.3)
        else:  # 50% no evidence
            p_value = np.random.uniform(0.1, 0.9)
            effect_size = np.random.uniform(0.0, 0.3)
        
        finding = Finding(
            finding_id=f"F{i+1}",
            variant_id=f"variant_{i+1}",
            exponent_name="beta",
            measured_value=0.3,
            predicted_value=0.3,
            p_value=p_value,
            effect_size=effect_size,
            sample_size=np.random.randint(30, 150),
            description=f"Finding {i+1}"
        )
        
        prioritizer.add_finding(finding)
    
    # Get summary
    summary = prioritizer.get_summary_statistics()
    
    print(f"\n2. Summary Statistics:")
    print(f"   Total findings: {summary['total_findings']}")
    print(f"   Average priority: {summary['avg_priority']:.1f}/100")
    
    print(f"\n3. By Category:")
    for category, count in summary['by_category'].items():
        pct = 100 * count / summary['total_findings']
        print(f"   {category}: {count} ({pct:.1f}%)")
    
    print(f"\n4. By Significance:")
    for sig, count in summary['by_significance'].items():
        pct = 100 * count / summary['total_findings']
        print(f"   {sig}: {count} ({pct:.1f}%)")
    
    print(f"\n5. By Effect Size:")
    for eff, count in summary['by_effect_size'].items():
        pct = 100 * count / summary['total_findings']
        print(f"   {eff}: {count} ({pct:.1f}%)")


def example_7_visualization():
    """Example 7: Visualize findings."""
    print("\n" + "=" * 80)
    print("Example 7: Visualization")
    print("=" * 80)
    
    prioritizer = EffectSizePrioritizer()
    
    # Add findings across the space
    np.random.seed(42)
    
    print("\n1. Adding 30 findings across significance-effect space...")
    
    for i in range(30):
        p_value = 10 ** np.random.uniform(-3, -0.5)
        effect_size = np.random.uniform(0.0, 2.0)
        
        finding = Finding(
            finding_id=f"F{i+1}",
            variant_id=f"variant_{i+1}",
            exponent_name="beta",
            measured_value=0.3,
            predicted_value=0.3,
            p_value=p_value,
            effect_size=effect_size,
            sample_size=100,
            description=f"Finding {i+1}"
        )
        
        prioritizer.add_finding(finding)
    
    # Visualize
    print("\n2. Visualization:")
    print(prioritizer.visualize_findings())
    
    print("\n3. Interpretation:")
    print("   - Top-right (●): Strong evidence - high priority")
    print("   - Bottom-right (◐): Large effect but not significant - collect more data")
    print("   - Top-left (○): Significant but small effect - false positive risk")
    print("   - Bottom-left (·): No evidence - low priority")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("EFFECT SIZE PRIORITIZATION SYSTEM EXAMPLES")
    print("=" * 80)
    print("\nDemonstrating how to prioritize findings based on both")
    print("statistical significance and practical importance")
    
    example_1_basic_prioritization()
    example_2_four_quadrants()
    example_3_false_positive_prevention()
    example_4_false_negative_prevention()
    example_5_investigation_plan()
    example_6_summary_statistics()
    example_7_visualization()
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("""
1. TWO DIMENSIONS NEEDED:
   - P-value: Is the effect real? (statistical significance)
   - Effect size: Is the effect important? (practical significance)
   - Both are required for proper interpretation

2. FOUR QUADRANTS:
   - Strong Evidence: High significance + large effect → Publish
   - Investigate Further: Low significance + large effect → More data
   - False Positive Risk: High significance + small effect → Verify
   - No Evidence: Low significance + small effect → Deprioritize

3. FALSE POSITIVE PREVENTION:
   - Large samples can make tiny effects "significant"
   - Effect size prevents wasting effort on unimportant findings
   - Example: p=0.001 but d=0.05 → Not worth investigating

4. FALSE NEGATIVE PREVENTION:
   - Small samples can make large effects "not significant"
   - Effect size highlights important findings needing more data
   - Example: p=0.08 but d=1.5 → Definitely worth more samples

5. PRIORITY SCORING:
   - Combines p-value, effect size, and sample size
   - Creates actionable priority queue
   - Focuses effort on most promising findings

6. INVESTIGATION PLAN:
   - Immediate: Strong evidence findings
   - Collect data: Large effects needing power
   - Verify: Significant but small effects
   - Low priority: No evidence

This approach prevents both false positives (statistically significant
but unimportant) and false negatives (important but not yet significant),
ensuring research effort is focused on truly meaningful findings.
""")
    
    print("=" * 80)
    print("Examples completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
