"""
Example: Multiplicative Confidence Scoring System

This example demonstrates the novel methodological enhancement of combining
confidence scores from multiple independent validation layers to create a
robust overall confidence metric.

The key insight: Each validation layer provides independent evidence:
- Detection: Flags anomalies based on deviation from known classes
- Comparison: Quantifies statistical significance of differences
- Validation: Rigorously tests specific hypotheses

By multiplying these independent scores, we get a conservative assessment
that requires agreement across all layers.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research.confidence_aggregator import ConfidenceAggregator, AggregatedConfidence
from src.research.base_types import (
    NovelPhenomenon, VAEAnalysisResults, SimulationData, ValidationResult,
    LatentRepresentation
)
from src.research.comparative_analyzer import ComparisonResults
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def create_mock_vae_results(
    variant_id: str,
    exponents: dict,
    exponent_errors: dict,
    r_squared_values: dict,
    tc: float = 2.269
) -> VAEAnalysisResults:
    """Create mock VAE results for demonstration."""
    return VAEAnalysisResults(
        variant_id=variant_id,
        parameters={'temperature_range': '1.0-4.0'},
        critical_temperature=tc,
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


def example_1_strong_evidence():
    """Example 1: Strong evidence across all layers (high confidence)."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Strong Evidence Across All Layers")
    print("="*80)
    print("\nScenario: Novel long-range Ising model with clear anomalous exponents,")
    print("statistically significant differences, and validated predictions.")
    print()
    
    aggregator = ConfidenceAggregator()
    
    # Create VAE results with anomalous exponents
    vae_results = create_mock_vae_results(
        variant_id='long_range_ising_alpha_2.5',
        exponents={'beta': 0.45, 'nu': 0.55, 'gamma': 1.1},
        exponent_errors={'beta': 0.02, 'nu': 0.03, 'gamma': 0.05},
        r_squared_values={'beta': 0.95, 'nu': 0.93, 'gamma': 0.94}
    )
    
    # Create detected phenomena (high confidence)
    phenomena = [
        NovelPhenomenon(
            phenomenon_type='anomalous_exponents',
            variant_id='long_range_ising_alpha_2.5',
            parameters={'alpha': 2.5},
            description='Beta exponent deviates 8.5σ from 3D Ising',
            confidence=0.92,
            supporting_evidence={'deviation_sigma': 8.5, 'r_squared': 0.95}
        )
    ]
    
    # Create comparison results (high significance)
    comparison_results = ComparisonResults(
        variant_ids=['long_range_ising_alpha_2.5', 'standard_3d_ising'],
        exponent_comparisons={
            'beta': {
                'exponent_name': 'beta',
                'variant_statistics': {},
                'anova': {'f_statistic': 45.2, 'p_value': 1e-8, 'significant': True},
                'pairwise_tests': {}
            }
        },
        universality_tests={
            'long_range_ising_alpha_2.5': {
                'mean_field': type('Test', (), {
                    'confidence': 0.88,
                    'matches': True,
                    'p_values': {'beta': 0.02},
                    'deviations': {'beta': 1.5}
                })()
            }
        },
        scaling_violations=[],
        phase_diagrams=[],
        summary_statistics={}
    )
    
    # Create validation results (validated)
    validation_results = {
        'exponent_beta': ValidationResult(
            hypothesis_id='test_beta',
            validated=True,
            confidence=0.90,
            p_values={'exponent': 0.03},
            effect_sizes={'cohens_d': 0.8},
            bootstrap_intervals={'exponent': (0.41, 0.49)},
            message='Hypothesis validated'
        )
    }
    
    # Aggregate confidence
    aggregated = aggregator.aggregate_confidence(
        vae_results=vae_results,
        phenomena=phenomena,
        comparison_results=comparison_results,
        validation_results=validation_results
    )
    
    print(aggregated.message)
    print(f"\nDetailed Breakdown:")
    print(f"  Detection confidence: {aggregated.layer_confidences['detection']:.2%}")
    print(f"    - {len(phenomena)} phenomena detected")
    print(f"    - Max phenomenon confidence: {phenomena[0].confidence:.2%}")
    print(f"    - Weighted by fit quality (R²): {np.mean(list(vae_results.r_squared_values.values())):.2%}")
    print(f"  Comparison confidence: {aggregated.layer_confidences['comparison']:.2%}")
    print(f"    - Best universality class match: mean_field (88%)")
    print(f"  Validation confidence: {aggregated.layer_confidences['validation']:.2%}")
    print(f"    - 1/1 tests validated")
    print(f"\nOverall: {aggregated.overall_confidence:.2%} = "
          f"{aggregated.layer_confidences['detection']:.2%} × "
          f"{aggregated.layer_confidences['comparison']:.2%} × "
          f"{aggregated.layer_confidences['validation']:.2%}")
    print(f"\n[OK] {aggregated.recommendation}: This finding has strong support across all validation layers.")
    
    # Visualize
    fig = aggregator.visualize_confidence_breakdown(aggregated)
    output_dir = Path('results/confidence_aggregation_demo')
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'example1_strong_evidence.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_dir / 'example1_strong_evidence.png'}")


def example_2_weak_validation():
    """Example 2: Strong detection but weak validation (moderate confidence)."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Strong Detection but Weak Validation")
    print("="*80)
    print("\nScenario: Anomaly detected with high confidence, but hypothesis validation")
    print("shows poor agreement with predictions. Suggests detection is real but")
    print("predictions were incorrect.")
    print()
    
    aggregator = ConfidenceAggregator()
    
    # Create VAE results
    vae_results = create_mock_vae_results(
        variant_id='frustrated_triangular',
        exponents={'beta': 0.28, 'nu': 0.72, 'gamma': 1.45},
        exponent_errors={'beta': 0.03, 'nu': 0.04, 'gamma': 0.06},
        r_squared_values={'beta': 0.91, 'nu': 0.89, 'gamma': 0.90}
    )
    
    # Strong detection
    phenomena = [
        NovelPhenomenon(
            phenomenon_type='anomalous_exponents',
            variant_id='frustrated_triangular',
            parameters={'frustration': 0.5},
            description='Exponents deviate from all known universality classes',
            confidence=0.88,
            supporting_evidence={'deviation_sigma': 6.2}
        )
    ]
    
    # Moderate comparison
    comparison_results = ComparisonResults(
        variant_ids=['frustrated_triangular'],
        exponent_comparisons={},
        universality_tests={
            'frustrated_triangular': {
                'ising_2d': type('Test', (), {
                    'confidence': 0.65,
                    'matches': False,
                    'p_values': {},
                    'deviations': {}
                })()
            }
        },
        scaling_violations=[],
        phase_diagrams=[],
        summary_statistics={}
    )
    
    # Weak validation (predictions were wrong)
    validation_results = {
        'exponent_beta': ValidationResult(
            hypothesis_id='test_beta',
            validated=False,
            confidence=0.25,
            p_values={'exponent': 0.001},
            effect_sizes={'cohens_d': 2.5},
            bootstrap_intervals={'exponent': (0.22, 0.34)},
            message='Hypothesis refuted: measured outside predicted range'
        )
    }
    
    # Aggregate
    aggregated = aggregator.aggregate_confidence(
        vae_results=vae_results,
        phenomena=phenomena,
        comparison_results=comparison_results,
        validation_results=validation_results
    )
    
    print(aggregated.message)
    print(f"\nAnalysis:")
    print(f"  Detection: {aggregated.layer_confidences['detection']:.2%} (HIGH)")
    print(f"    -> Anomaly clearly detected")
    print(f"  Comparison: {aggregated.layer_confidences['comparison']:.2%} (MODERATE)")
    print(f"    -> Statistically different from known classes")
    print(f"  Validation: {aggregated.layer_confidences['validation']:.2%} (LOW)")
    print(f"    -> Predictions were incorrect")
    print(f"\nOverall: {aggregated.overall_confidence:.2%}")
    print(f"\n[WARNING] {aggregated.recommendation}: The anomaly is real, but our theoretical")
    print(f"  predictions need revision. Investigate validation layer to refine hypothesis.")
    
    # Visualize
    fig = aggregator.visualize_confidence_breakdown(aggregated)
    output_dir = Path('results/confidence_aggregation_demo')
    fig.savefig(output_dir / 'example2_weak_validation.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_dir / 'example2_weak_validation.png'}")


def example_3_false_positive():
    """Example 3: Weak evidence across all layers (likely false positive)."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Weak Evidence - Likely False Positive")
    print("="*80)
    print("\nScenario: Marginal anomaly detection with poor fit quality, no statistical")
    print("significance, and failed validation. Likely a false positive from noise.")
    print()
    
    aggregator = ConfidenceAggregator()
    
    # Create VAE results with poor fit quality
    vae_results = create_mock_vae_results(
        variant_id='noisy_system',
        exponents={'beta': 0.33, 'nu': 0.64, 'gamma': 1.25},
        exponent_errors={'beta': 0.08, 'nu': 0.10, 'gamma': 0.15},
        r_squared_values={'beta': 0.55, 'nu': 0.52, 'gamma': 0.58}  # Poor fits
    )
    
    # Weak detection
    phenomena = [
        NovelPhenomenon(
            phenomenon_type='anomalous_exponents',
            variant_id='noisy_system',
            parameters={},
            description='Beta slightly elevated (3.2σ)',
            confidence=0.35,  # Low confidence
            supporting_evidence={'deviation_sigma': 3.2, 'r_squared': 0.55}
        )
    ]
    
    # Weak comparison
    comparison_results = ComparisonResults(
        variant_ids=['noisy_system'],
        exponent_comparisons={},
        universality_tests={
            'noisy_system': {
                'ising_3d': type('Test', (), {
                    'confidence': 0.42,
                    'matches': False,
                    'p_values': {},
                    'deviations': {}
                })()
            }
        },
        scaling_violations=[],
        phase_diagrams=[],
        summary_statistics={}
    )
    
    # Failed validation
    validation_results = {
        'exponent_beta': ValidationResult(
            hypothesis_id='test_beta',
            validated=False,
            confidence=0.18,
            p_values={'exponent': 0.45},
            effect_sizes={'cohens_d': 0.3},
            bootstrap_intervals={'exponent': (0.17, 0.49)},
            message='Hypothesis refuted: large uncertainty'
        )
    }
    
    # Aggregate
    aggregated = aggregator.aggregate_confidence(
        vae_results=vae_results,
        phenomena=phenomena,
        comparison_results=comparison_results,
        validation_results=validation_results
    )
    
    print(aggregated.message)
    print(f"\nAnalysis:")
    print(f"  Detection: {aggregated.layer_confidences['detection']:.2%} (LOW)")
    print(f"    -> Marginal anomaly, poor fit quality")
    print(f"  Comparison: {aggregated.layer_confidences['comparison']:.2%} (LOW)")
    print(f"    -> Not statistically significant")
    print(f"  Validation: {aggregated.layer_confidences['validation']:.2%} (LOW)")
    print(f"    -> Failed validation tests")
    print(f"\nOverall: {aggregated.overall_confidence:.2%}")
    print(f"\n[FAIL] {aggregated.recommendation}: All layers show weak evidence.")
    print(f"  This is likely a false positive from noise or insufficient data.")
    print(f"  Recommendation: Collect more data or improve simulation quality.")
    
    # Visualize
    fig = aggregator.visualize_confidence_breakdown(aggregated)
    output_dir = Path('results/confidence_aggregation_demo')
    fig.savefig(output_dir / 'example3_false_positive.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_dir / 'example3_false_positive.png'}")


def example_4_variant_comparison():
    """Example 4: Compare confidence across multiple variants."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Confidence Comparison Across Variants")
    print("="*80)
    print("\nScenario: Compare confidence scores for multiple model variants to")
    print("prioritize which findings deserve further investigation.")
    print()
    
    aggregator = ConfidenceAggregator()
    
    # Create mock aggregated confidences for different variants
    variant_confidences = {}
    
    # Variant 1: Strong evidence
    variant_confidences['long_range_alpha_2.5'] = AggregatedConfidence(
        overall_confidence=0.85,
        layer_confidences={'detection': 0.92, 'comparison': 0.88, 'validation': 0.90},
        layer_weights={'detection': 0.33, 'comparison': 0.33, 'validation': 0.34},
        breakdown={},
        recommendation='STRONG_EVIDENCE',
        message='Strong evidence'
    )
    
    # Variant 2: Moderate evidence
    variant_confidences['frustrated_kagome'] = AggregatedConfidence(
        overall_confidence=0.58,
        layer_confidences={'detection': 0.78, 'comparison': 0.72, 'validation': 0.65},
        layer_weights={'detection': 0.33, 'comparison': 0.33, 'validation': 0.34},
        breakdown={},
        recommendation='INVESTIGATE_VALIDATION',
        message='Moderate evidence'
    )
    
    # Variant 3: Weak validation
    variant_confidences['disordered_0.3'] = AggregatedConfidence(
        overall_confidence=0.42,
        layer_confidences={'detection': 0.85, 'comparison': 0.68, 'validation': 0.32},
        layer_weights={'detection': 0.33, 'comparison': 0.33, 'validation': 0.34},
        breakdown={},
        recommendation='INVESTIGATE_VALIDATION',
        message='Weak validation'
    )
    
    # Variant 4: False positive
    variant_confidences['standard_2d'] = AggregatedConfidence(
        overall_confidence=0.15,
        layer_confidences={'detection': 0.28, 'comparison': 0.35, 'validation': 0.22},
        layer_weights={'detection': 0.33, 'comparison': 0.33, 'validation': 0.34},
        breakdown={},
        recommendation='LIKELY_FALSE_POSITIVE',
        message='Likely false positive'
    )
    
    # Visualize comparison
    fig = aggregator.compare_confidence_across_variants(variant_confidences)
    output_dir = Path('results/confidence_aggregation_demo')
    fig.savefig(output_dir / 'example4_variant_comparison.png', dpi=300, bbox_inches='tight')
    
    print("Variant Confidence Summary:")
    print("-" * 80)
    for variant, conf in sorted(variant_confidences.items(), 
                                key=lambda x: x[1].overall_confidence, 
                                reverse=True):
        print(f"\n{variant}:")
        print(f"  Overall: {conf.overall_confidence:.2%}")
        print(f"  Detection: {conf.layer_confidences['detection']:.2%}")
        print(f"  Comparison: {conf.layer_confidences['comparison']:.2%}")
        print(f"  Validation: {conf.layer_confidences['validation']:.2%}")
        print(f"  -> {conf.recommendation}")
    
    print(f"\nPriority Order (by overall confidence):")
    sorted_variants = sorted(variant_confidences.items(), 
                           key=lambda x: x[1].overall_confidence, 
                           reverse=True)
    for i, (variant, conf) in enumerate(sorted_variants, 1):
        print(f"  {i}. {variant} ({conf.overall_confidence:.2%})")
    
    print(f"\nVisualization saved to: {output_dir / 'example4_variant_comparison.png'}")


def main():
    """Run all examples."""
    
    print("\n" + "="*80)
    print("MULTIPLICATIVE CONFIDENCE SCORING SYSTEM")
    print("="*80)
    print("\nThis example demonstrates a novel methodological enhancement:")
    print("Combining confidence scores from multiple independent validation layers")
    print("to create a robust overall confidence metric.")
    print("\nKey Insight:")
    print("  overall_confidence = detection_conf * comparison_conf * validation_conf")
    print("\nThis multiplicative approach ensures:")
    print("  - All layers must agree for high confidence")
    print("  - Disagreement in any layer reduces overall confidence")
    print("  - Conservative assessment prevents false positives")
    
    # Run examples
    example_1_strong_evidence()
    example_2_weak_validation()
    example_3_false_positive()
    example_4_variant_comparison()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nThe multiplicative confidence scoring system provides:")
    print("  1. Robust assessment requiring agreement across layers")
    print("  2. Clear identification of weak points in evidence")
    print("  3. Prioritization of findings for further investigation")
    print("  4. Protection against false positives from single-layer anomalies")
    print("\nThis represents a novel methodological contribution to computational")
    print("phase transition studies, enabling more reliable discovery of novel phenomena.")
    print()


if __name__ == '__main__':
    main()
