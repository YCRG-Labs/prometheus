"""
Example: Validation Triangle Cross-Validation

This example demonstrates the validation triangle, a novel methodological enhancement
that provides an over-determined system for robust validation by checking consistency
between three independent aspects:

1. Critical Exponents (measured values)
2. Universality Class (theoretical classification)
3. Scaling Relations (theoretical constraints)

The key insight: All three must be mutually consistent. If any two are satisfied,
the third is constrained. This creates powerful redundancy for validation.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research.validation_triangle import ValidationTriangle, ConsistencyStatus
from src.research.base_types import VAEAnalysisResults, LatentRepresentation
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def create_mock_vae_results(
    variant_id: str,
    exponents: dict,
    exponent_errors: dict,
    r_squared_values: dict
) -> VAEAnalysisResults:
    """Create mock VAE results for demonstration."""
    return VAEAnalysisResults(
        variant_id=variant_id,
        parameters={'system': variant_id},
        critical_temperature=2.269,
        tc_confidence=0.95,
        exponents=exponents,
        exponent_errors=exponent_errors,
        r_squared_values=r_squared_values,
        latent_representation=LatentRepresentation(
            latent_means=np.random.randn(20, 10),
            latent_stds=np.random.rand(20, 10) * 0.1,
            order_parameter_dim=0,
            reconstruction_quality={'mse': 0.01}
        ),
        order_parameter_dim=0
    )


def example_1_all_consistent():
    """Example 1: All three vertices consistent (ideal case)."""
    print("\n" + "="*80)
    print("EXAMPLE 1: All Three Vertices Consistent")
    print("="*80)
    print("\nScenario: Perfect 2D Ising model with accurate measurements.")
    print("All three vertices should be mutually consistent.")
    print()
    
    triangle = ValidationTriangle(consistency_threshold=2.5)
    
    # Create perfect 2D Ising exponents
    vae_results = create_mock_vae_results(
        variant_id='2d_ising_perfect',
        exponents={
            'beta': 0.125,
            'nu': 1.0,
            'gamma': 1.75,
            'alpha': 0.0
        },
        exponent_errors={
            'beta': 0.005,
            'nu': 0.02,
            'gamma': 0.03,
            'alpha': 0.02
        },
        r_squared_values={
            'beta': 0.98,
            'nu': 0.97,
            'gamma': 0.96,
            'alpha': 0.95
        }
    )
    
    # Validate
    validation = triangle.validate(vae_results, expected_universality_class='ising_2d', dimensions=2)
    
    print(f"Overall Status: {validation.overall_status.value.upper()}")
    print(f"Overall Confidence: {validation.overall_confidence:.2%}")
    print(f"\nMessage: {validation.message}")
    
    print(f"\nEdge Consistency:")
    for edge in validation.edges:
        status = "[OK]" if edge.consistent else "[FAIL]"
        print(f"  {status} {edge.vertex1.value} <-> {edge.vertex2.value}: {edge.confidence:.2%}")
        print(f"      {edge.details.get('message', 'No details')}")
    
    if validation.inconsistencies:
        print(f"\nInconsistencies:")
        for inc in validation.inconsistencies:
            print(f"  - {inc}")
    else:
        print(f"\n[OK] No inconsistencies detected!")
    
    if validation.recommendations:
        print(f"\nRecommendations:")
        for rec in validation.recommendations:
            print(f"  - {rec}")
    
    # Visualize
    fig = triangle.visualize_triangle(validation)
    output_dir = Path('results/validation_triangle_demo')
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'example1_all_consistent.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_dir / 'example1_all_consistent.png'}")


def example_2_bad_exponents():
    """Example 2: Exponents inconsistent (measurement error)."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Inconsistent Exponents (Measurement Error)")
    print("="*80)
    print("\nScenario: 2D Ising model but with measurement errors in exponents.")
    print("Universality class and scaling relations should be consistent,")
    print("but measured exponents deviate.")
    print()
    
    triangle = ValidationTriangle(consistency_threshold=2.5)
    
    # Create 2D Ising with bad beta measurement
    vae_results = create_mock_vae_results(
        variant_id='2d_ising_bad_measurement',
        exponents={
            'beta': 0.18,  # Should be 0.125 (measurement error)
            'nu': 1.0,
            'gamma': 1.75,
            'alpha': 0.0
        },
        exponent_errors={
            'beta': 0.02,
            'nu': 0.02,
            'gamma': 0.03,
            'alpha': 0.02
        },
        r_squared_values={
            'beta': 0.65,  # Poor fit quality
            'nu': 0.97,
            'gamma': 0.96,
            'alpha': 0.95
        }
    )
    
    # Validate
    validation = triangle.validate(vae_results, expected_universality_class='ising_2d', dimensions=2)
    
    print(f"Overall Status: {validation.overall_status.value.upper()}")
    print(f"Overall Confidence: {validation.overall_confidence:.2%}")
    print(f"\nMessage: {validation.message}")
    
    print(f"\nEdge Consistency:")
    for edge in validation.edges:
        status = "[OK]" if edge.consistent else "[FAIL]"
        print(f"  {status} {edge.vertex1.value} <-> {edge.vertex2.value}: {edge.confidence:.2%}")
        print(f"      {edge.details.get('message', 'No details')}")
    
    print(f"\nInconsistencies:")
    for inc in validation.inconsistencies:
        print(f"  - {inc}")
    
    print(f"\nRecommendations:")
    for rec in validation.recommendations:
        print(f"  - {rec}")
    
    print(f"\nAnalysis:")
    print(f"  The validation triangle correctly identifies that EXPONENTS are the problem.")
    print(f"  Universality class (2D Ising) and scaling relations are consistent,")
    print(f"  but measured exponents deviate. This suggests measurement error.")
    
    # Visualize
    fig = triangle.visualize_triangle(validation)
    output_dir = Path('results/validation_triangle_demo')
    fig.savefig(output_dir / 'example2_bad_exponents.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_dir / 'example2_bad_exponents.png'}")


def example_3_wrong_universality_class():
    """Example 3: Wrong universality class assignment."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Wrong Universality Class Assignment")
    print("="*80)
    print("\nScenario: Measured exponents satisfy scaling relations, but don't match")
    print("the expected universality class. Suggests wrong class assignment.")
    print()
    
    triangle = ValidationTriangle(consistency_threshold=2.5)
    
    # Create 3D Ising exponents but claim it's 2D
    vae_results = create_mock_vae_results(
        variant_id='3d_ising_mislabeled',
        exponents={
            'beta': 0.326,  # 3D Ising value
            'nu': 0.630,    # 3D Ising value
            'gamma': 1.237, # 3D Ising value
            'alpha': 0.110  # 3D Ising value
        },
        exponent_errors={
            'beta': 0.005,
            'nu': 0.005,
            'gamma': 0.01,
            'alpha': 0.01
        },
        r_squared_values={
            'beta': 0.96,
            'nu': 0.95,
            'gamma': 0.94,
            'alpha': 0.93
        }
    )
    
    # Validate against wrong class (2D instead of 3D)
    validation = triangle.validate(vae_results, expected_universality_class='ising_2d', dimensions=3)
    
    print(f"Overall Status: {validation.overall_status.value.upper()}")
    print(f"Overall Confidence: {validation.overall_confidence:.2%}")
    print(f"\nMessage: {validation.message}")
    
    print(f"\nEdge Consistency:")
    for edge in validation.edges:
        status = "[OK]" if edge.consistent else "[FAIL]"
        print(f"  {status} {edge.vertex1.value} <-> {edge.vertex2.value}: {edge.confidence:.2%}")
        print(f"      {edge.details.get('message', 'No details')}")
    
    print(f"\nInconsistencies:")
    for inc in validation.inconsistencies:
        print(f"  - {inc}")
    
    print(f"\nRecommendations:")
    for rec in validation.recommendations:
        print(f"  - {rec}")
    
    print(f"\nAnalysis:")
    print(f"  The validation triangle identifies UNIVERSALITY_CLASS as the problem.")
    print(f"  Exponents and scaling relations are consistent (it's actually 3D Ising),")
    print(f"  but they don't match the claimed 2D Ising class.")
    
    # Visualize
    fig = triangle.visualize_triangle(validation)
    output_dir = Path('results/validation_triangle_demo')
    fig.savefig(output_dir / 'example3_wrong_class.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_dir / 'example3_wrong_class.png'}")


def example_4_scaling_violation():
    """Example 4: Scaling relation violation (novel physics)."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Scaling Relation Violation (Novel Physics)")
    print("="*80)
    print("\nScenario: Exponents match a universality class, but violate scaling relations.")
    print("This could indicate novel physics beyond standard scaling theory.")
    print()
    
    triangle = ValidationTriangle(consistency_threshold=2.5)
    
    # Create exponents that match mean field but violate hyperscaling
    vae_results = create_mock_vae_results(
        variant_id='novel_system',
        exponents={
            'beta': 0.5,    # Mean field
            'nu': 0.5,      # Mean field
            'gamma': 1.0,   # Mean field
            'alpha': 0.3    # Should be 0.0 for hyperscaling!
        },
        exponent_errors={
            'beta': 0.01,
            'nu': 0.01,
            'gamma': 0.02,
            'alpha': 0.02
        },
        r_squared_values={
            'beta': 0.94,
            'nu': 0.93,
            'gamma': 0.92,
            'alpha': 0.91
        }
    )
    
    # Validate
    validation = triangle.validate(vae_results, expected_universality_class='mean_field', dimensions=3)
    
    print(f"Overall Status: {validation.overall_status.value.upper()}")
    print(f"Overall Confidence: {validation.overall_confidence:.2%}")
    print(f"\nMessage: {validation.message}")
    
    print(f"\nEdge Consistency:")
    for edge in validation.edges:
        status = "[OK]" if edge.consistent else "[FAIL]"
        print(f"  {status} {edge.vertex1.value} <-> {edge.vertex2.value}: {edge.confidence:.2%}")
        print(f"      {edge.details.get('message', 'No details')}")
    
    print(f"\nInconsistencies:")
    for inc in validation.inconsistencies:
        print(f"  - {inc}")
    
    print(f"\nRecommendations:")
    for rec in validation.recommendations:
        print(f"  - {rec}")
    
    print(f"\nAnalysis:")
    print(f"  The validation triangle identifies SCALING_RELATIONS as the problem.")
    print(f"  Exponents match mean field class, but hyperscaling is violated.")
    print(f"  This could indicate:")
    print(f"    - Finite-size effects")
    print(f"    - Novel physics beyond standard scaling")
    print(f"    - Crossover behavior")
    
    # Visualize
    fig = triangle.visualize_triangle(validation)
    output_dir = Path('results/validation_triangle_demo')
    fig.savefig(output_dir / 'example4_scaling_violation.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_dir / 'example4_scaling_violation.png'}")


def example_5_all_inconsistent():
    """Example 5: All three vertices inconsistent (data quality issue)."""
    print("\n" + "="*80)
    print("EXAMPLE 5: All Vertices Inconsistent (Data Quality Issue)")
    print("="*80)
    print("\nScenario: Poor quality data with inconsistent exponents, wrong class,")
    print("and scaling violations. Indicates fundamental data quality problems.")
    print()
    
    triangle = ValidationTriangle(consistency_threshold=2.5)
    
    # Create completely inconsistent exponents
    vae_results = create_mock_vae_results(
        variant_id='poor_quality_data',
        exponents={
            'beta': 0.22,   # Between 2D and 3D
            'nu': 0.85,     # Doesn't match anything
            'gamma': 1.5,   # Intermediate value
            'alpha': 0.25   # Violates hyperscaling
        },
        exponent_errors={
            'beta': 0.08,   # Large errors
            'nu': 0.10,
            'gamma': 0.15,
            'alpha': 0.12
        },
        r_squared_values={
            'beta': 0.55,   # Poor fits
            'nu': 0.52,
            'gamma': 0.58,
            'alpha': 0.48
        }
    )
    
    # Validate
    validation = triangle.validate(vae_results, expected_universality_class='ising_2d', dimensions=2)
    
    print(f"Overall Status: {validation.overall_status.value.upper()}")
    print(f"Overall Confidence: {validation.overall_confidence:.2%}")
    print(f"\nMessage: {validation.message}")
    
    print(f"\nEdge Consistency:")
    for edge in validation.edges:
        status = "[OK]" if edge.consistent else "[FAIL]"
        print(f"  {status} {edge.vertex1.value} <-> {edge.vertex2.value}: {edge.confidence:.2%}")
        print(f"      {edge.details.get('message', 'No details')}")
    
    print(f"\nInconsistencies:")
    for inc in validation.inconsistencies:
        print(f"  - {inc}")
    
    print(f"\nRecommendations:")
    for rec in validation.recommendations:
        print(f"  - {rec}")
    
    print(f"\nAnalysis:")
    print(f"  All three vertices are inconsistent - this is a red flag!")
    print(f"  Combined with poor R² values and large errors, this indicates:")
    print(f"    - Insufficient equilibration")
    print(f"    - Too few samples")
    print(f"    - Numerical instabilities")
    print(f"  Recommendation: Improve simulation quality before drawing conclusions.")
    
    # Visualize
    fig = triangle.visualize_triangle(validation)
    output_dir = Path('results/validation_triangle_demo')
    fig.savefig(output_dir / 'example5_all_inconsistent.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_dir / 'example5_all_inconsistent.png'}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("VALIDATION TRIANGLE CROSS-VALIDATION SYSTEM")
    print("="*80)
    print("\nThis example demonstrates a novel methodological enhancement:")
    print("The validation triangle provides an over-determined system for robust")
    print("validation by checking consistency between three independent aspects:")
    print()
    print("  1. Critical Exponents (measured values)")
    print("  2. Universality Class (theoretical classification)")
    print("  3. Scaling Relations (theoretical constraints)")
    print()
    print("Key Insight:")
    print("  All three must be mutually consistent. If any two are satisfied,")
    print("  the third is constrained. This creates powerful redundancy.")
    print()
    print("Triangle Structure:")
    print("        Universality Class")
    print("              /  \\")
    print("             /    \\")
    print("            /      \\")
    print("     Exponents --- Scaling Relations")
    
    # Run examples
    example_1_all_consistent()
    example_2_bad_exponents()
    example_3_wrong_universality_class()
    example_4_scaling_violation()
    example_5_all_inconsistent()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nThe validation triangle provides:")
    print("  1. Over-determined system with redundant checks")
    print("  2. Clear identification of problematic vertex")
    print("  3. Distinction between measurement error and novel physics")
    print("  4. Actionable recommendations for each scenario")
    print()
    print("When two vertices are consistent:")
    print("  - Identifies the third as problematic")
    print("  - Provides specific recommendations")
    print("  - Constrains possible explanations")
    print()
    print("This represents a novel methodological contribution enabling")
    print("more reliable validation in computational phase transition studies.")
    print()


if __name__ == '__main__':
    main()
