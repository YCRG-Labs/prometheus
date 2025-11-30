"""
Error Propagation Chain Tracker Example.

This example demonstrates how to use the ErrorPropagationTracker to:
1. Track uncertainty through the validation chain
2. Identify dominant error sources
3. Visualize error propagation
4. Generate improvement recommendations

The tracker monitors how measurement uncertainty from VAE analysis
propagates through bootstrap resampling, effect size calculation,
and anomaly threshold comparison.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.research.error_propagation_tracker import (
    ErrorPropagationTracker,
    PropagationStage
)


def example_1_single_measurement_tracking():
    """Example 1: Track error propagation for a single measurement."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Measurement Error Propagation")
    print("="*70)
    
    # Initialize tracker
    tracker = ErrorPropagationTracker(anomaly_threshold=3.0)
    
    # Simulate VAE measurement of critical exponent β
    # Theoretical value for 2D Ising: β = 0.125
    vae_params = {
        'exponent_value': 0.123,  # Measured value
        'r_squared': 0.95,  # Good fit quality
        'n_data_points': 20,  # Number of temperature points
        'fit_residuals': np.random.normal(0, 0.002, 20)  # Small residuals
    }
    
    # Simulate bootstrap resampling
    # Bootstrap samples around measured value with some variance
    bootstrap_samples = np.random.normal(0.123, 0.003, 1000)
    
    # Theoretical prediction
    predicted_value = 0.125
    predicted_uncertainty = 0.001  # Small theoretical uncertainty
    
    # Create complete propagation chain
    chain = tracker.create_propagation_chain(
        measurement_id="2D_Ising_beta",
        vae_params=vae_params,
        bootstrap_samples=bootstrap_samples,
        predicted_value=predicted_value,
        predicted_uncertainty=predicted_uncertainty
    )
    
    # Print chain details
    print(f"\nMeasurement: {chain.measurement_id}")
    print(f"Dominant Error Source: {chain.dominant_source.value}")
    print(f"Error Amplification: {chain.amplification_factor:.2f}x")
    print(f"Total Uncertainty: {chain.total_uncertainty:.4f}")
    
    print("\nStage-by-Stage Breakdown:")
    for stage in chain.stages:
        print(f"\n  {stage.stage.value}:")
        print(f"    Value: {stage.value:.4f}")
        print(f"    Uncertainty: {stage.uncertainty:.4f}")
        print(f"    Relative: {stage.relative_uncertainty:.2%}")
        if stage.metadata:
            print(f"    Metadata: {stage.metadata}")
    
    # Visualize the chain
    output_dir = Path("results/error_propagation_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tracker.visualize_propagation_chain(
        chain,
        save_path=output_dir / "single_chain.png",
        show=False
    )
    print(f"\nVisualization saved to {output_dir / 'single_chain.png'}")


def example_2_multiple_measurements_comparison():
    """Example 2: Compare error propagation across multiple measurements."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Multiple Measurements Comparison")
    print("="*70)
    
    tracker = ErrorPropagationTracker(anomaly_threshold=3.0)
    
    # Simulate measurements with different quality levels
    measurements = [
        {
            'id': '2D_Ising_beta_high_quality',
            'vae': {'exponent_value': 0.123, 'r_squared': 0.98, 'n_data_points': 30},
            'bootstrap_std': 0.002,
            'predicted': 0.125
        },
        {
            'id': '2D_Ising_beta_medium_quality',
            'vae': {'exponent_value': 0.120, 'r_squared': 0.90, 'n_data_points': 20},
            'bootstrap_std': 0.005,
            'predicted': 0.125
        },
        {
            'id': '2D_Ising_beta_low_quality',
            'vae': {'exponent_value': 0.115, 'r_squared': 0.75, 'n_data_points': 15},
            'bootstrap_std': 0.010,
            'predicted': 0.125
        },
        {
            'id': '3D_Ising_beta_high_quality',
            'vae': {'exponent_value': 0.320, 'r_squared': 0.96, 'n_data_points': 25},
            'bootstrap_std': 0.007,
            'predicted': 0.326
        },
        {
            'id': 'Long_Range_beta_anomalous',
            'vae': {'exponent_value': 0.450, 'r_squared': 0.92, 'n_data_points': 20},
            'bootstrap_std': 0.015,
            'predicted': 0.326  # Expecting 3D Ising class
        }
    ]
    
    # Create chains for all measurements
    for meas in measurements:
        bootstrap_samples = np.random.normal(
            meas['vae']['exponent_value'],
            meas['bootstrap_std'],
            1000
        )
        
        tracker.create_propagation_chain(
            measurement_id=meas['id'],
            vae_params=meas['vae'],
            bootstrap_samples=bootstrap_samples,
            predicted_value=meas['predicted'],
            predicted_uncertainty=0.001
        )
    
    print(f"\nCreated {len(tracker.chains)} propagation chains")
    
    # Identify dominant error sources
    print("\nDominant Error Sources:")
    dominant_sources = tracker.identify_dominant_error_sources()
    for stage, contribution in dominant_sources.items():
        print(f"  {stage.value}: {contribution:.1%}")
    
    # Get summary statistics
    summary = tracker.get_summary_statistics()
    print(f"\nSummary Statistics:")
    print(f"  Number of chains: {summary['n_chains']}")
    print(f"  Average amplification: {summary['amplification']['mean']:.2f}x "
          f"(±{summary['amplification']['std']:.2f})")
    print(f"  Average final uncertainty: {summary['final_uncertainty']['mean']:.4f} "
          f"(±{summary['final_uncertainty']['std']:.4f})")
    print(f"  Most dominant source: {summary['dominant_source']['stage']} "
          f"({summary['dominant_source']['contribution']:.1%})")
    
    # Visualize all chains
    output_dir = Path("results/error_propagation_demo")
    tracker.visualize_multiple_chains(
        save_path=output_dir / "multiple_chains.png",
        show=False
    )
    print(f"\nMulti-chain visualization saved to {output_dir / 'multiple_chains.png'}")


def example_3_improvement_recommendations():
    """Example 3: Generate targeted improvement recommendations."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Improvement Recommendations")
    print("="*70)
    
    tracker = ErrorPropagationTracker(anomaly_threshold=3.0)
    
    # Simulate measurements with VAE as dominant error source
    # (poor fit quality, few data points)
    for i in range(5):
        vae_params = {
            'exponent_value': 0.12 + np.random.normal(0, 0.01),
            'r_squared': 0.70 + np.random.uniform(0, 0.1),  # Poor fits
            'n_data_points': 12 + np.random.randint(0, 5)  # Few points
        }
        
        bootstrap_samples = np.random.normal(
            vae_params['exponent_value'],
            0.003,
            1000
        )
        
        tracker.create_propagation_chain(
            measurement_id=f"measurement_{i+1}",
            vae_params=vae_params,
            bootstrap_samples=bootstrap_samples,
            predicted_value=0.125,
            predicted_uncertainty=0.001
        )
    
    # Generate and print recommendations
    recommendations = tracker.generate_improvement_recommendations()
    tracker.print_recommendations(recommendations)
    
    # Show which recommendations are most important
    print("\nTop Priority Actions:")
    high_priority = [r for r in recommendations if r['priority'] == 'HIGH']
    for rec in high_priority:
        print(f"\n{rec['stage'].upper()} ({rec['contribution']:.1%} of total uncertainty)")
        print(f"  → {rec['recommendations'][0]}")  # Show top recommendation


def example_4_effect_of_data_quality():
    """Example 4: Demonstrate effect of data quality on error propagation."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Effect of Data Quality on Error Propagation")
    print("="*70)
    
    # Compare three scenarios: poor, medium, good data quality
    scenarios = {
        'Poor Quality': {
            'r_squared': 0.70,
            'n_data_points': 10,
            'bootstrap_std': 0.015
        },
        'Medium Quality': {
            'r_squared': 0.85,
            'n_data_points': 20,
            'bootstrap_std': 0.008
        },
        'High Quality': {
            'r_squared': 0.95,
            'n_data_points': 30,
            'bootstrap_std': 0.003
        }
    }
    
    results = {}
    
    for scenario_name, params in scenarios.items():
        tracker = ErrorPropagationTracker(anomaly_threshold=3.0)
        
        vae_params = {
            'exponent_value': 0.123,
            'r_squared': params['r_squared'],
            'n_data_points': params['n_data_points']
        }
        
        bootstrap_samples = np.random.normal(0.123, params['bootstrap_std'], 1000)
        
        chain = tracker.create_propagation_chain(
            measurement_id=f"beta_{scenario_name.replace(' ', '_')}",
            vae_params=vae_params,
            bootstrap_samples=bootstrap_samples,
            predicted_value=0.125,
            predicted_uncertainty=0.001
        )
        
        results[scenario_name] = {
            'amplification': chain.amplification_factor,
            'final_uncertainty': chain.stages[-1].uncertainty,
            'dominant_source': chain.dominant_source.value,
            'classification': chain.stages[-1].metadata.get('classification', 'unknown')
        }
    
    # Print comparison
    print("\nData Quality Impact on Error Propagation:")
    print(f"{'Scenario':<20} {'Amplification':<15} {'Final Unc.':<15} {'Dominant Source':<25} {'Classification'}")
    print("-" * 95)
    
    for scenario, res in results.items():
        print(f"{scenario:<20} {res['amplification']:<15.2f} "
              f"{res['final_uncertainty']:<15.4f} "
              f"{res['dominant_source']:<25} {res['classification']}")
    
    print("\nKey Insights:")
    print("  • Higher R² → Lower VAE uncertainty → Lower amplification")
    print("  • More data points → Better statistical power → Lower uncertainty")
    print("  • Better bootstrap sampling → More stable estimates")
    print("  • Quality improvements compound through the chain")


def example_5_anomaly_detection_confidence():
    """Example 5: How uncertainty affects anomaly detection confidence."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Uncertainty and Anomaly Detection Confidence")
    print("="*70)
    
    tracker = ErrorPropagationTracker(anomaly_threshold=3.0)
    
    # Test cases with different deviations from predicted value
    test_cases = [
        {'name': 'Clear Normal', 'measured': 0.125, 'predicted': 0.125, 'uncertainty': 0.003},
        {'name': 'Borderline Normal', 'measured': 0.134, 'predicted': 0.125, 'uncertainty': 0.003},
        {'name': 'Uncertain', 'measured': 0.140, 'predicted': 0.125, 'uncertainty': 0.005},
        {'name': 'Borderline Anomaly', 'measured': 0.145, 'predicted': 0.125, 'uncertainty': 0.003},
        {'name': 'Clear Anomaly', 'measured': 0.200, 'predicted': 0.125, 'uncertainty': 0.003},
    ]
    
    print("\nAnomaly Detection with Uncertainty:")
    print(f"{'Case':<20} {'Measured':<12} {'Effect Size':<12} {'Classification':<15} {'Confidence'}")
    print("-" * 75)
    
    for case in test_cases:
        vae_params = {
            'exponent_value': case['measured'],
            'r_squared': 0.95,
            'n_data_points': 20
        }
        
        bootstrap_samples = np.random.normal(case['measured'], case['uncertainty'], 1000)
        
        chain = tracker.create_propagation_chain(
            measurement_id=case['name'].replace(' ', '_'),
            vae_params=vae_params,
            bootstrap_samples=bootstrap_samples,
            predicted_value=case['predicted'],
            predicted_uncertainty=0.001
        )
        
        effect_size = chain.stages[2].value  # Effect size stage
        classification = chain.stages[3].metadata['classification']
        confidence = chain.stages[3].metadata['confidence']
        
        print(f"{case['name']:<20} {case['measured']:<12.4f} "
              f"{effect_size:<12.2f} {classification:<15} {confidence:.2f}")
    
    print("\nKey Insights:")
    print("  • Clear cases (far from threshold) have high confidence")
    print("  • Borderline cases have lower confidence")
    print("  • Higher uncertainty → lower confidence for same deviation")
    print("  • Uncertainty quantification enables nuanced classification")


def example_6_source_decomposition():
    """Example 6: Detailed decomposition of uncertainty sources."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Detailed Uncertainty Source Decomposition")
    print("="*70)
    
    tracker = ErrorPropagationTracker(anomaly_threshold=3.0)
    
    # Create a measurement with detailed tracking
    vae_params = {
        'exponent_value': 0.320,
        'r_squared': 0.92,
        'n_data_points': 25,
        'fit_residuals': np.random.normal(0, 0.005, 25)
    }
    
    bootstrap_samples = np.random.normal(0.320, 0.007, 1000)
    
    chain = tracker.create_propagation_chain(
        measurement_id="3D_Ising_beta_detailed",
        vae_params=vae_params,
        bootstrap_samples=bootstrap_samples,
        predicted_value=0.326,
        predicted_uncertainty=0.002
    )
    
    print("\nDetailed Source Decomposition:")
    
    for stage in chain.stages:
        print(f"\n{stage.stage.value.upper()}:")
        print(f"  Value: {stage.value:.4f}")
        print(f"  Total Uncertainty: {stage.uncertainty:.4f}")
        print(f"  Relative Uncertainty: {stage.relative_uncertainty:.2%}")
        
        if stage.sources:
            print(f"  Source Breakdown:")
            total_sources = sum(abs(v) for v in stage.sources.values())
            for source_name, source_value in stage.sources.items():
                contribution = abs(source_value) / total_sources if total_sources > 0 else 0
                print(f"    • {source_name}: {source_value:.4f} ({contribution:.1%})")
    
    print(f"\nOverall Chain Summary:")
    print(f"  Initial Uncertainty: {chain.stages[0].uncertainty:.4f}")
    print(f"  Final Uncertainty: {chain.stages[-1].uncertainty:.4f}")
    print(f"  Amplification Factor: {chain.amplification_factor:.2f}x")
    print(f"  Dominant Source: {chain.dominant_source.value}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("ERROR PROPAGATION CHAIN TRACKER EXAMPLES")
    print("="*70)
    
    # Run all examples
    example_1_single_measurement_tracking()
    example_2_multiple_measurements_comparison()
    example_3_improvement_recommendations()
    example_4_effect_of_data_quality()
    example_5_anomaly_detection_confidence()
    example_6_source_decomposition()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. Error propagation tracking reveals dominant uncertainty sources")
    print("  2. Data quality improvements compound through the validation chain")
    print("  3. Uncertainty quantification enables confident anomaly detection")
    print("  4. Targeted improvements can be prioritized based on contributions")
    print("  5. Visualization helps communicate uncertainty to stakeholders")
    print("\nCheck results/error_propagation_demo/ for visualizations")


if __name__ == "__main__":
    main()
