"""
Example demonstrating the Hypothesis Manager functionality.

This script shows how to:
1. Create research hypotheses about phase transition behavior
2. Compare predictions with known universality classes
3. Validate hypotheses against experimental results
4. Track hypothesis status and confidence
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.research import HypothesisManager, UNIVERSALITY_CLASSES


def main():
    """Demonstrate hypothesis manager functionality."""
    
    print("=" * 70)
    print("Hypothesis Manager Example")
    print("=" * 70)
    
    # Initialize hypothesis manager
    manager = HypothesisManager()
    
    # Example 1: Create hypothesis for long-range Ising model
    print("\n1. Creating hypothesis for long-range Ising model")
    print("-" * 70)
    
    hypothesis1 = manager.create_hypothesis(
        description="Long-range 2D Ising with α=2.5 belongs to mean-field universality class",
        variant_id="long_range_ising_2d",
        predictions={
            'beta': 0.5,
            'nu': 0.5,
            'gamma': 1.0,
        },
        prediction_errors={
            'beta': 0.05,
            'nu': 0.05,
            'gamma': 0.1,
        },
        universality_class='mean_field',
    )
    
    print(f"Created hypothesis: {hypothesis1.hypothesis_id}")
    print(f"Description: {hypothesis1.description}")
    print(f"Predictions: {hypothesis1.predictions}")
    print(f"Status: {hypothesis1.status}")
    
    # Example 2: Compare with universality class
    print("\n2. Comparing predictions with universality class")
    print("-" * 70)
    
    comparison = manager.compare_with_universality_class(hypothesis1.hypothesis_id)
    print(f"Universality class: {comparison['universality_class']}")
    print(f"Overall agreement: {comparison['overall_agreement']}")
    print(f"Agreement fraction: {comparison['agreement_fraction']:.1%}")
    print("\nDetailed comparisons:")
    for exponent, details in comparison['comparisons'].items():
        print(f"  {exponent}:")
        print(f"    Predicted: {details['predicted']:.3f}")
        print(f"    Class value: {details['class_value']:.3f}")
        print(f"    Difference: {details['difference']:.3f}")
        print(f"    Agrees: {details['agrees']}")
    
    # Example 3: Validate hypothesis with experimental results
    print("\n3. Validating hypothesis with experimental results")
    print("-" * 70)
    
    # Simulate experimental results (close to mean-field predictions)
    experimental_results = {
        'beta': 0.48,  # Close to 0.5
        'nu': 0.52,    # Close to 0.5
        'gamma': 0.95, # Close to 1.0
    }
    
    experimental_errors = {
        'beta': 0.03,
        'nu': 0.03,
        'gamma': 0.08,
    }
    
    print(f"Experimental results: {experimental_results}")
    print(f"Experimental errors: {experimental_errors}")
    
    validation = manager.validate_hypothesis(
        hypothesis1.hypothesis_id,
        experimental_results,
        experimental_errors,
    )
    
    print(f"\nValidation result: {validation.validated}")
    print(f"Confidence: {validation.confidence:.3f}")
    print(f"Message: {validation.message}")
    print("\nP-values:")
    for exponent, p_value in validation.p_values.items():
        print(f"  {exponent}: {p_value:.4f}")
    
    # Example 4: Create hypothesis for standard 2D Ising
    print("\n4. Creating hypothesis for standard 2D Ising model")
    print("-" * 70)
    
    hypothesis2 = manager.create_hypothesis(
        description="Standard 2D Ising model follows 2D Ising universality class",
        variant_id="standard_2d_ising",
        predictions={
            'beta': 0.125,
            'nu': 1.0,
            'gamma': 1.75,
        },
        universality_class='2d_ising',
    )
    
    print(f"Created hypothesis: {hypothesis2.hypothesis_id}")
    print(f"Description: {hypothesis2.description}")
    
    # Example 5: Validate with results that don't match
    print("\n5. Validating with results that don't match predictions")
    print("-" * 70)
    
    # Simulate experimental results that deviate significantly
    bad_results = {
        'beta': 0.35,  # Much larger than 0.125
        'nu': 0.65,    # Much smaller than 1.0
        'gamma': 1.25, # Much smaller than 1.75
    }
    
    validation2 = manager.validate_hypothesis(
        hypothesis2.hypothesis_id,
        bad_results,
    )
    
    print(f"Validation result: {validation2.validated}")
    print(f"Confidence: {validation2.confidence:.3f}")
    print(f"Message: {validation2.message}")
    
    # Example 6: List all hypotheses
    print("\n6. Listing all hypotheses")
    print("-" * 70)
    
    all_hypotheses = manager.list_hypotheses()
    print(f"Total hypotheses: {len(all_hypotheses)}")
    
    for hyp in all_hypotheses:
        print(f"\n  {hyp.hypothesis_id}:")
        print(f"    Description: {hyp.description}")
        print(f"    Status: {hyp.status}")
        print(f"    Confidence: {hyp.confidence:.3f}")
    
    # Example 7: Filter hypotheses by status
    print("\n7. Filtering hypotheses by status")
    print("-" * 70)
    
    validated_hypotheses = manager.list_hypotheses(status='validated')
    refuted_hypotheses = manager.list_hypotheses(status='refuted')
    
    print(f"Validated hypotheses: {len(validated_hypotheses)}")
    for hyp in validated_hypotheses:
        print(f"  - {hyp.hypothesis_id}: {hyp.description}")
    
    print(f"\nRefuted hypotheses: {len(refuted_hypotheses)}")
    for hyp in refuted_hypotheses:
        print(f"  - {hyp.hypothesis_id}: {hyp.description}")
    
    # Example 8: Get detailed hypothesis status
    print("\n8. Getting detailed hypothesis status")
    print("-" * 70)
    
    status = manager.get_hypothesis_status(hypothesis1.hypothesis_id)
    print(f"Hypothesis ID: {status['hypothesis_id']}")
    print(f"Status: {status['status']}")
    print(f"Confidence: {status['confidence']:.3f}")
    print(f"Predictions: {status['predictions']}")
    if status['validation_results']:
        print(f"Validated exponents: {status['validation_results']['validated_exponents']}")
    
    # Example 9: Show available universality classes
    print("\n9. Available universality classes")
    print("-" * 70)
    
    classes = manager.get_universality_classes()
    for class_name, exponents in classes.items():
        print(f"\n{class_name}:")
        for exp_name, exp_value in exponents.items():
            print(f"  {exp_name}: {exp_value:.3f}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
