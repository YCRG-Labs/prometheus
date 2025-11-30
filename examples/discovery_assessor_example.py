"""
Example demonstrating the Discovery Assessor for evaluating novel physics discoveries.

This example shows how to:
1. Assess whether validated findings constitute novel physics
2. Classify discovery types
3. Evaluate scientific significance
4. Compare with theoretical predictions
5. Generate discovery summaries
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.research.discovery_assessor import DiscoveryAssessor, PhysicsDiscovery
from src.research.base_types import VAEAnalysisResults, LatentRepresentation
from src.research.unified_validation_pipeline import UnifiedValidationPipeline, ValidationReport
from src.research.universality_class_manager import UniversalityClassManager


def create_mock_vae_results(variant_id: str, exponents: dict) -> VAEAnalysisResults:
    """Create mock VAE results for testing."""
    latent_rep = LatentRepresentation(
        latent_means=np.random.randn(100, 10),
        latent_stds=np.random.rand(100, 10),
        order_parameter_dim=0
    )
    
    return VAEAnalysisResults(
        variant_id=variant_id,
        parameters={'temperature_range': (1.0, 5.0)},
        critical_temperature=2.269,
        tc_confidence=0.95,
        exponents=exponents,
        exponent_errors={k: 0.02 for k in exponents.keys()},
        r_squared_values={k: 0.95 for k in exponents.keys()},
        latent_representation=latent_rep,
        order_parameter_dim=0
    )


def create_mock_validation_report(
    variant_id: str,
    validated: bool,
    confidence: float,
    n_physics_novel: int = 1
) -> ValidationReport:
    """Create mock validation report for testing."""
    from datetime import datetime
    
    return ValidationReport(
        variant_id=variant_id,
        timestamp=datetime.now(),
        overall_validated=validated,
        overall_confidence=confidence,
        recommendation="VALIDATED" if validated else "NOT_VALIDATED",
        pattern_results={
            'phenomena_detection': {
                'phenomena': [],
                'n_phenomena': n_physics_novel,
                'max_confidence': 0.9
            },
            'anomaly_classification': {
                'classified_anomalies': [],
                'n_physics_novel': n_physics_novel,
                'n_methodological': 0
            },
            'validation_triangle': None,
            'universality_management': {
                'closest_class': '2D_Ising',
                'class_confidence': 0.3,
                'deviations': {}
            }
        },
        summary="Mock validation report",
        publication_ready=validated and confidence > 0.95
    )


def example_1_novel_universality_class():
    """Example 1: Discovering a novel universality class."""
    print("=" * 70)
    print("Example 1: Novel Universality Class Discovery")
    print("=" * 70)
    print()
    
    # Create assessor
    assessor = DiscoveryAssessor(
        novelty_threshold=3.0,
        confidence_threshold=0.90
    )
    
    # Simulate discovery of novel exponents (significantly different from 2D Ising)
    # 2D Ising: β=0.125, ν=1.0, γ=1.75
    # Novel class: β=0.35, ν=0.65, γ=1.3
    vae_results = create_mock_vae_results(
        variant_id='long_range_ising_alpha_2.2',
        exponents={
            'beta': 0.35,
            'nu': 0.65,
            'gamma': 1.3
        }
    )
    
    # Create validation report (high confidence, validated)
    validation_report = create_mock_validation_report(
        variant_id='long_range_ising_alpha_2.2',
        validated=True,
        confidence=0.96,
        n_physics_novel=3
    )
    
    # Assess novelty
    discovery = assessor.assess_novelty(
        vae_results=vae_results,
        validation_report=validation_report,
        theoretical_predictions={'beta': 0.5, 'nu': 0.5, 'gamma': 1.0},  # Mean-field predictions
        variant_description="Long-range Ising model with power-law exponent α=2.2"
    )
    
    if discovery:
        print("✓ Novel physics discovered!")
        print()
        print(assessor.generate_discovery_summary(discovery))
        
        # Save discovery
        output_dir = Path('results/discoveries')
        filepath = assessor.save_discovery(discovery, output_dir)
        print(f"\nDiscovery saved to: {filepath}")
    else:
        print("✗ No novel physics detected")
    
    print()


def example_2_exotic_transition():
    """Example 2: Discovering an exotic phase transition."""
    print("=" * 70)
    print("Example 2: Exotic Phase Transition Discovery")
    print("=" * 70)
    print()
    
    assessor = DiscoveryAssessor(
        novelty_threshold=3.0,
        confidence_threshold=0.90
    )
    
    # Simulate discovery of first-order transition with unusual exponents
    vae_results = create_mock_vae_results(
        variant_id='frustrated_triangular_lattice',
        exponents={
            'beta': 0.18,
            'nu': 0.85,
            'gamma': 1.5
        }
    )
    
    validation_report = create_mock_validation_report(
        variant_id='frustrated_triangular_lattice',
        validated=True,
        confidence=0.93,
        n_physics_novel=2
    )
    
    # Assess novelty
    discovery = assessor.assess_novelty(
        vae_results=vae_results,
        validation_report=validation_report,
        theoretical_predictions=None,  # No predictions available
        variant_description="Antiferromagnetic Ising on frustrated triangular lattice"
    )
    
    if discovery:
        print("✓ Novel physics discovered!")
        print()
        print(assessor.generate_discovery_summary(discovery))
        
        print(f"\nDiscovery Type: {discovery.discovery_type}")
        print(f"Significance: {discovery.significance}")
        print(f"Publication Potential: {discovery.publication_potential}")
    else:
        print("✗ No novel physics detected")
    
    print()


def example_3_confirmed_prediction():
    """Example 3: Confirming theoretical predictions (not novel)."""
    print("=" * 70)
    print("Example 3: Confirming Theoretical Predictions (Not Novel)")
    print("=" * 70)
    print()
    
    assessor = DiscoveryAssessor(
        novelty_threshold=3.0,
        confidence_threshold=0.90
    )
    
    # Simulate standard 2D Ising results (matches known class)
    vae_results = create_mock_vae_results(
        variant_id='standard_2d_ising',
        exponents={
            'beta': 0.125,
            'nu': 1.0,
            'gamma': 1.75
        }
    )
    
    validation_report = create_mock_validation_report(
        variant_id='standard_2d_ising',
        validated=True,
        confidence=0.95,
        n_physics_novel=0  # No novel anomalies
    )
    
    # Assess novelty
    discovery = assessor.assess_novelty(
        vae_results=vae_results,
        validation_report=validation_report,
        theoretical_predictions={'beta': 0.125, 'nu': 1.0, 'gamma': 1.75},
        variant_description="Standard 2D Ising model on square lattice"
    )
    
    if discovery:
        print("✓ Novel physics discovered!")
        print(assessor.generate_discovery_summary(discovery))
    else:
        print("✗ No novel physics detected")
        print("   Reason: Results match known 2D Ising universality class")
        print("   This confirms existing theory but is not a novel discovery")
    
    print()


def example_4_classification_types():
    """Example 4: Different discovery type classifications."""
    print("=" * 70)
    print("Example 4: Discovery Type Classifications")
    print("=" * 70)
    print()
    
    assessor = DiscoveryAssessor(
        novelty_threshold=3.0,
        confidence_threshold=0.90
    )
    
    # Test different variant types
    test_cases = [
        {
            'variant_id': 'long_range_ising_alpha_2.3',
            'exponents': {'beta': 0.28, 'nu': 0.72},
            'expected_type': 'crossover_behavior'
        },
        {
            'variant_id': 'diluted_ising_p_0.6',
            'exponents': {'beta': 0.15, 'nu': 0.95},
            'expected_type': 'disorder_driven'
        },
        {
            'variant_id': 'kagome_lattice_ising',
            'exponents': {'beta': 0.20, 'nu': 0.88},
            'expected_type': 'frustration_induced'
        }
    ]
    
    for case in test_cases:
        vae_results = create_mock_vae_results(
            variant_id=case['variant_id'],
            exponents=case['exponents']
        )
        
        validation_report = create_mock_validation_report(
            variant_id=case['variant_id'],
            validated=True,
            confidence=0.92,
            n_physics_novel=2
        )
        
        discovery = assessor.assess_novelty(
            vae_results=vae_results,
            validation_report=validation_report
        )
        
        if discovery:
            print(f"Variant: {case['variant_id']}")
            print(f"  Classified as: {discovery.discovery_type}")
            print(f"  Expected: {case['expected_type']}")
            print(f"  Match: {'✓' if discovery.discovery_type == case['expected_type'] else '✗'}")
            print()


def example_5_significance_assessment():
    """Example 5: Assessing scientific significance levels."""
    print("=" * 70)
    print("Example 5: Scientific Significance Assessment")
    print("=" * 70)
    print()
    
    assessor = DiscoveryAssessor(
        novelty_threshold=3.0,
        confidence_threshold=0.90
    )
    
    # Test different significance levels
    test_cases = [
        {
            'name': 'Major Discovery',
            'exponents': {'beta': 0.40, 'nu': 0.60, 'gamma': 1.2},
            'confidence': 0.97,
            'theory_status': 'refuted',
            'expected_significance': 'major'
        },
        {
            'name': 'Moderate Discovery',
            'exponents': {'beta': 0.18, 'nu': 0.92, 'gamma': 1.65},
            'confidence': 0.92,
            'theory_status': 'extended',
            'expected_significance': 'moderate'
        },
        {
            'name': 'Minor Discovery',
            'exponents': {'beta': 0.13, 'nu': 0.98, 'gamma': 1.72},
            'confidence': 0.90,
            'theory_status': 'partial',
            'expected_significance': 'minor'
        }
    ]
    
    for case in test_cases:
        vae_results = create_mock_vae_results(
            variant_id=f"test_variant_{case['name'].replace(' ', '_').lower()}",
            exponents=case['exponents']
        )
        
        validation_report = create_mock_validation_report(
            variant_id=vae_results.variant_id,
            validated=True,
            confidence=case['confidence'],
            n_physics_novel=2
        )
        
        discovery = assessor.assess_novelty(
            vae_results=vae_results,
            validation_report=validation_report,
            theoretical_predictions={'beta': 0.125, 'nu': 1.0, 'gamma': 1.75}
        )
        
        if discovery:
            print(f"{case['name']}:")
            print(f"  Assessed Significance: {discovery.significance}")
            print(f"  Expected: {case['expected_significance']}")
            print(f"  Publication Potential: {discovery.publication_potential}")
            print(f"  Validation Confidence: {discovery.validation_confidence:.2%}")
            print()


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("DISCOVERY ASSESSOR EXAMPLES")
    print("=" * 70 + "\n")
    
    example_1_novel_universality_class()
    example_2_exotic_transition()
    example_3_confirmed_prediction()
    example_4_classification_types()
    example_5_significance_assessment()
    
    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()
