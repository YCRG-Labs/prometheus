"""
Methodological Anomaly Classification Example

This example demonstrates Task 16.6: Methodological anomaly classifier that
distinguishes physics anomalies from methodological issues.

Key Concept:
Classify anomalies into four categories:
1. physics_novel - Genuine novel physics phenomena
2. data_quality - Poor data quality issues
3. convergence - VAE training or MC convergence problems
4. finite_size_scaling - Finite-size effects

This prevents false positives from methodological issues being mistaken for
novel physics discoveries.

Novel Methodological Contribution:
Robust discrimination between physics and methodology enables confident
identification of genuine novel phenomena while avoiding wasted effort on
methodological artifacts.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import Dict

from src.research.anomaly_classifier import (
    MethodologicalAnomalyClassifier,
    AnomalyCategory,
    ClassifiedAnomaly
)
from src.research.base_types import (
    VAEAnalysisResults,
    SimulationData,
    LatentRepresentation
)
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.config import LoggingConfig


def create_mock_vae_results(
    variant_id: str,
    parameters: Dict[str, float],
    scenario: str = 'good'
) -> VAEAnalysisResults:
    """Create mock VAE results for different scenarios.
    
    Args:
        variant_id: Variant identifier
        parameters: Parameter values
        scenario: Scenario type ('good', 'poor_data', 'poor_convergence', 'small_lattice', 'novel')
        
    Returns:
        Mock VAE analysis results
    """
    # Base exponents (2D Ising-like)
    exponents = {
        'beta': 0.125,
        'nu': 1.0,
        'gamma': 1.75
    }
    
    # Modify based on scenario
    if scenario == 'poor_data':
        # Poor data quality: low R², high errors
        r_squared = {'beta': 0.55, 'nu': 0.60, 'gamma': 0.58}
        exponent_errors = {'beta': 0.05, 'nu': 0.15, 'gamma': 0.20}
        tc_confidence = 0.85
        # Exponents slightly off due to poor fits
        exponents['beta'] += 0.03
        exponents['nu'] += 0.08
        
    elif scenario == 'poor_convergence':
        # Poor convergence: low Tc confidence, moderate R²
        r_squared = {'beta': 0.75, 'nu': 0.78, 'gamma': 0.76}
        exponent_errors = {'beta': 0.02, 'nu': 0.06, 'gamma': 0.08}
        tc_confidence = 0.55
        # Exponents off due to poor Tc detection
        exponents['beta'] += 0.04
        exponents['nu'] -= 0.12
        
    elif scenario == 'small_lattice':
        # Finite-size effects: good quality but anomalous exponents
        r_squared = {'beta': 0.92, 'nu': 0.90, 'gamma': 0.91}
        exponent_errors = {'beta': 0.01, 'nu': 0.03, 'gamma': 0.04}
        tc_confidence = 0.92
        # Exponents shifted by finite-size effects
        exponents['beta'] += 0.05
        exponents['nu'] -= 0.15
        exponents['gamma'] += 0.10
        
    elif scenario == 'novel':
        # Novel physics: excellent quality, anomalous exponents
        r_squared = {'beta': 0.95, 'nu': 0.96, 'gamma': 0.94}
        exponent_errors = {'beta': 0.008, 'nu': 0.02, 'gamma': 0.03}
        tc_confidence = 0.95
        # Genuinely different exponents
        exponents['beta'] = 0.35  # Significantly different
        exponents['nu'] = 0.75
        exponents['gamma'] = 1.50
        
    else:  # 'good'
        # Good quality, standard exponents
        r_squared = {'beta': 0.95, 'nu': 0.96, 'gamma': 0.94}
        exponent_errors = {'beta': 0.008, 'nu': 0.02, 'gamma': 0.03}
        tc_confidence = 0.95
    
    return VAEAnalysisResults(
        variant_id=variant_id,
        parameters=parameters,
        critical_temperature=2.269,
        tc_confidence=tc_confidence,
        exponents=exponents,
        exponent_errors=exponent_errors,
        r_squared_values=r_squared,
        latent_representation=LatentRepresentation(
            latent_means=np.random.randn(10, 16),
            latent_stds=np.random.randn(10, 16),
            order_parameter_dim=0,
            reconstruction_quality={'mse': 0.01}
        ),
        order_parameter_dim=0
    )


def create_mock_simulation_data(lattice_size: int = 32) -> SimulationData:
    """Create mock simulation data.
    
    Args:
        lattice_size: Size of lattice
        
    Returns:
        Mock simulation data
    """
    n_temps = 20
    n_samples = 100
    
    return SimulationData(
        variant_id='2d_ising',
        parameters={'temperature': 2.269},
        temperatures=np.linspace(1.5, 3.5, n_temps),
        configurations=np.random.choice([-1, 1], size=(n_temps, n_samples, lattice_size, lattice_size)),
        magnetizations=np.random.randn(n_temps, n_samples),
        energies=np.random.randn(n_temps, n_samples),
        metadata={'lattice_size': lattice_size}
    )


def example_1_classify_different_scenarios():
    """Example 1: Classify anomalies from different scenarios.
    
    Demonstrates how the classifier distinguishes between different types
    of anomalies.
    """
    print("\n" + "="*80)
    print("Example 1: Classify Different Anomaly Scenarios")
    print("="*80)
    
    classifier = MethodologicalAnomalyClassifier(
        anomaly_threshold=3.0,
        r_squared_threshold=0.7,
        tc_confidence_threshold=0.8
    )
    
    scenarios = {
        'poor_data': 'Poor data quality (low R²)',
        'poor_convergence': 'Poor convergence (low Tc confidence)',
        'small_lattice': 'Finite-size effects (small lattice)',
        'novel': 'Novel physics (all checks pass)',
    }
    
    for scenario, description in scenarios.items():
        print(f"\n--- Scenario: {description} ---")
        
        # Create mock results
        vae_results = create_mock_vae_results(
            variant_id='2d_ising',
            parameters={'temperature': 2.269},
            scenario=scenario
        )
        
        # Add simulation data for finite-size scenario
        if scenario == 'small_lattice':
            sim_data = create_mock_simulation_data(lattice_size=16)
        else:
            sim_data = create_mock_simulation_data(lattice_size=64)
        
        # Classify
        classified = classifier.classify_anomalies(vae_results, sim_data)
        
        if classified:
            for c in classified:
                print(f"\nClassification: {c.category.value}")
                print(f"Confidence: {c.confidence:.2%}")
                print(f"Severity: {c.severity}")
                print(f"Recommendation: {c.recommendation[:100]}...")
        else:
            print("\nNo anomalies detected")
    
    print("\n✓ Example 1 complete: Demonstrated classification of different scenarios")


def example_2_classification_report():
    """Example 2: Generate comprehensive classification report.
    
    Shows how to generate a detailed report for multiple anomalies.
    """
    print("\n" + "="*80)
    print("Example 2: Comprehensive Classification Report")
    print("="*80)
    
    classifier = MethodologicalAnomalyClassifier()
    
    # Create multiple scenarios
    all_classified = []
    
    scenarios = ['poor_data', 'poor_convergence', 'small_lattice', 'novel']
    for i, scenario in enumerate(scenarios):
        vae_results = create_mock_vae_results(
            variant_id=f'variant_{i}',
            parameters={'temperature': 2.0 + i * 0.2},
            scenario=scenario
        )
        
        sim_data = create_mock_simulation_data(
            lattice_size=16 if scenario == 'small_lattice' else 64
        )
        
        classified = classifier.classify_anomalies(vae_results, sim_data)
        all_classified.extend(classified)
    
    # Generate report
    report = classifier.generate_classification_report(all_classified)
    print(report)
    
    print("\n✓ Example 2 complete: Generated comprehensive classification report")


def example_3_evidence_analysis():
    """Example 3: Analyze evidence for classification.
    
    Shows the evidence collected for each classification decision.
    """
    print("\n" + "="*80)
    print("Example 3: Evidence Analysis")
    print("="*80)
    
    classifier = MethodologicalAnomalyClassifier()
    
    # Test novel physics scenario
    print("\n--- Novel Physics Scenario ---")
    vae_results = create_mock_vae_results(
        variant_id='novel_system',
        parameters={'alpha': 2.5},
        scenario='novel'
    )
    sim_data = create_mock_simulation_data(lattice_size=64)
    
    classified = classifier.classify_anomalies(vae_results, sim_data)
    
    if classified:
        c = classified[0]
        print(f"\nCategory: {c.category.value}")
        print(f"Confidence: {c.confidence:.2%}")
        print("\nEvidence collected:")
        for key, value in c.evidence.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            elif isinstance(value, bool):
                print(f"  {key}: {'✓' if value else '✗'}")
            else:
                print(f"  {key}: {value}")
        
        print(f"\nDecision logic:")
        print(f"  Data quality good: {c.evidence['data_quality_good']}")
        print(f"  Convergence good: {c.evidence['convergence_good']}")
        print(f"  Scaling satisfied: {c.evidence['scaling_satisfied']}")
        print(f"  → Classification: {c.category.value}")
    
    print("\n✓ Example 3 complete: Analyzed evidence for classification")


def example_4_severity_levels():
    """Example 4: Demonstrate severity level determination.
    
    Shows how severity is determined for different categories.
    """
    print("\n" + "="*80)
    print("Example 4: Severity Level Determination")
    print("="*80)
    
    classifier = MethodologicalAnomalyClassifier()
    
    test_cases = [
        ('poor_data', 'Critical data quality (R²<0.5)'),
        ('poor_convergence', 'Critical convergence (Tc conf<0.5)'),
        ('small_lattice', 'Very small lattice (16×16)'),
        ('novel', 'Novel physics candidate'),
    ]
    
    print("\nSeverity levels for different scenarios:")
    print("-" * 60)
    
    for scenario, description in test_cases:
        vae_results = create_mock_vae_results(
            variant_id='test',
            parameters={'temp': 2.0},
            scenario=scenario
        )
        
        sim_data = create_mock_simulation_data(
            lattice_size=16 if scenario == 'small_lattice' else 64
        )
        
        classified = classifier.classify_anomalies(vae_results, sim_data)
        
        if classified:
            c = classified[0]
            severity_icon = {
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢'
            }.get(c.severity, '⚪')
            
            print(f"\n{severity_icon} {description}")
            print(f"   Category: {c.category.value}")
            print(f"   Severity: {c.severity.upper()}")
            print(f"   Confidence: {c.confidence:.2%}")
    
    print("\n✓ Example 4 complete: Demonstrated severity level determination")


def example_5_recommendation_generation():
    """Example 5: Generate specific recommendations.
    
    Shows the specific recommendations for each category.
    """
    print("\n" + "="*80)
    print("Example 5: Recommendation Generation")
    print("="*80)
    
    classifier = MethodologicalAnomalyClassifier()
    
    scenarios = {
        'poor_data': 'Data Quality Issues',
        'poor_convergence': 'Convergence Problems',
        'small_lattice': 'Finite-Size Effects',
        'novel': 'Novel Physics Candidate',
    }
    
    print("\nSpecific recommendations for each category:")
    print("=" * 80)
    
    for scenario, title in scenarios.items():
        print(f"\n{title}:")
        print("-" * 80)
        
        vae_results = create_mock_vae_results(
            variant_id='test',
            parameters={'temp': 2.0},
            scenario=scenario
        )
        
        sim_data = create_mock_simulation_data(
            lattice_size=16 if scenario == 'small_lattice' else 64
        )
        
        classified = classifier.classify_anomalies(vae_results, sim_data)
        
        if classified:
            c = classified[0]
            print(f"\nCategory: {c.category.value}")
            print(f"Recommendation:")
            # Format recommendation with line breaks
            rec_parts = c.recommendation.split('. ')
            for part in rec_parts:
                if part.strip():
                    print(f"  • {part.strip()}.")
    
    print("\n✓ Example 5 complete: Generated specific recommendations")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("METHODOLOGICAL ANOMALY CLASSIFICATION")
    print("Task 16.6 - Novel Methodological Enhancement")
    print("="*80)
    
    print("\nThis example demonstrates robust discrimination between:")
    print("  1. physics_novel - Genuine novel physics phenomena")
    print("  2. data_quality - Poor data quality issues")
    print("  3. convergence - VAE training or MC convergence problems")
    print("  4. finite_size_scaling - Finite-size effects")
    
    print("\nKey Innovation:")
    print("  Prevents false positives from methodological issues being")
    print("  mistaken for novel physics discoveries.")
    
    # Set up logging
    logging_config = LoggingConfig(
        level='INFO',
        console_output=True,
        file_output=False
    )
    setup_logging(logging_config)
    
    # Run examples
    example_1_classify_different_scenarios()
    example_2_classification_report()
    example_3_evidence_analysis()
    example_4_severity_levels()
    example_5_recommendation_generation()
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("  1. Classifier distinguishes physics from methodology automatically")
    print("  2. Four categories enable targeted remediation")
    print("  3. Evidence-based classification provides transparency")
    print("  4. Severity levels prioritize investigation efforts")
    print("  5. Specific recommendations guide next steps")
    
    print("\nNovel Methodological Contribution:")
    print("  Robust discrimination between physics and methodology prevents")
    print("  wasted effort on methodological artifacts and enables confident")
    print("  identification of genuine novel phenomena.")


if __name__ == '__main__':
    main()
