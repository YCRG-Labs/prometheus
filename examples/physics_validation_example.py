"""
Physics Validation and Visualization Example

This example demonstrates how to use the physics validation framework
and visualization system to validate AI-discovered physics results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.analysis.physics_validation import PhysicsValidator, ValidationMetrics
from src.analysis.visualization import PublicationVisualizer, AnalysisReporter
from src.analysis.latent_analysis import LatentRepresentation
from src.analysis.order_parameter_discovery import OrderParameterCandidate, CorrelationResult
from src.analysis.phase_detection import PhaseDetectionResult


def create_sample_data():
    """Create sample data for demonstration."""
    print("Creating sample data...")
    
    # Generate realistic latent space data
    n_samples = 1000
    temperatures = np.linspace(1.5, 3.0, n_samples)
    
    # Create latent coordinates that correlate with temperature
    z1 = np.random.normal(0, 1, n_samples) + 0.5 * (temperatures - 2.269)
    z2 = np.random.normal(0, 1, n_samples)
    
    # Create magnetizations with realistic phase transition
    magnetizations = np.where(
        temperatures < 2.269,
        np.random.normal(0.8, 0.1, n_samples),  # Ordered phase
        np.random.normal(0.0, 0.1, n_samples)   # Disordered phase
    )
    
    # Create energies and reconstruction errors
    energies = -2.0 + 0.5 * temperatures + np.random.normal(0, 0.1, n_samples)
    reconstruction_errors = np.random.exponential(0.01, n_samples)
    
    latent_repr = LatentRepresentation(
        z1=z1, z2=z2, temperatures=temperatures,
        magnetizations=magnetizations, energies=energies,
        reconstruction_errors=reconstruction_errors,
        sample_indices=np.arange(n_samples)
    )
    
    return latent_repr


def create_order_parameter_candidates():
    """Create sample order parameter candidates."""
    print("Creating order parameter candidates...")
    
    # Strong correlation with magnetization for z1
    mag_corr = CorrelationResult(
        correlation_coefficient=0.85,
        p_value=1e-15,
        confidence_interval=(0.82, 0.88),
        sample_size=1000,
        is_significant=True
    )
    
    # Weak correlation with energy
    energy_corr = CorrelationResult(
        correlation_coefficient=-0.25,
        p_value=0.001,
        confidence_interval=(-0.35, -0.15),
        sample_size=1000,
        is_significant=True
    )
    
    # Create primary candidate (z1 as order parameter)
    primary_candidate = OrderParameterCandidate(
        latent_dimension='z1',
        correlation_with_magnetization=mag_corr,
        correlation_with_energy=energy_corr,
        temperature_dependence={
            'is_monotonic': True,
            'has_critical_enhancement': True,
            'mean_range': 2.5,
            'gradient_std': 0.1
        },
        critical_behavior={
            'phase_separation': 1.2,
            'variance_enhancement': 1.8,
            'std_ratio_low_high': 2.1
        },
        confidence_score=0.92
    )
    
    # Create secondary candidate (z2 as comparison)
    weak_mag_corr = CorrelationResult(
        correlation_coefficient=0.15,
        p_value=0.1,
        confidence_interval=(0.05, 0.25),
        sample_size=1000,
        is_significant=False
    )
    
    secondary_candidate = OrderParameterCandidate(
        latent_dimension='z2',
        correlation_with_magnetization=weak_mag_corr,
        correlation_with_energy=energy_corr,
        temperature_dependence={
            'is_monotonic': False,
            'has_critical_enhancement': False,
            'mean_range': 0.8,
            'gradient_std': 0.05
        },
        critical_behavior={
            'phase_separation': 0.3,
            'variance_enhancement': 1.1,
            'std_ratio_low_high': 1.2
        },
        confidence_score=0.25
    )
    
    return [primary_candidate, secondary_candidate]


def create_phase_detection_result():
    """Create sample phase detection result."""
    print("Creating phase detection result...")
    
    return PhaseDetectionResult(
        critical_temperature=2.275,  # Close to theoretical 2.269
        confidence=0.88,
        method='ensemble',
        transition_region=(2.22, 2.33)
    )


def demonstrate_physics_validation():
    """Demonstrate physics validation framework."""
    print("\n" + "="*60)
    print("PHYSICS VALIDATION DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    latent_repr = create_sample_data()
    order_param_candidates = create_order_parameter_candidates()
    phase_detection_result = create_phase_detection_result()
    
    # Initialize validator
    validator = PhysicsValidator(theoretical_tc=2.269, tolerance_percent=5.0)
    
    print("\n1. Validating Order Parameter Discovery...")
    order_validation = validator.validate_order_parameter_discovery(
        latent_repr, order_param_candidates
    )
    
    print(f"   - Best candidate: {order_validation['primary_dimension']}")
    print(f"   - Correlation with magnetization: {order_validation['correlation_coefficient']:.3f}")
    print(f"   - Statistical significance: p = {order_validation['statistical_significance']:.2e}")
    print(f"   - Correlation strength: {order_validation['correlation_strength']}")
    
    print("\n2. Validating Critical Temperature...")
    tc_validation = validator.validate_critical_temperature(phase_detection_result)
    
    print(f"   - Discovered T_c: {tc_validation['discovered_tc']:.3f}")
    print(f"   - Theoretical T_c: {tc_validation['theoretical_tc']:.3f}")
    print(f"   - Relative error: {tc_validation['relative_error_percent']:.2f}%")
    print(f"   - Within tolerance: {tc_validation['within_tolerance']}")
    print(f"   - Validation status: {tc_validation['validation_status']}")
    
    print("\n3. Comprehensive Physics Validation...")
    validation_metrics = validator.comprehensive_physics_validation(
        latent_repr, order_param_candidates, phase_detection_result
    )
    
    print(f"   - Overall physics consistency: {validation_metrics.physics_consistency_score:.3f}")
    print(f"   - Order parameter correlation: {validation_metrics.order_parameter_correlation:.3f}")
    print(f"   - Critical temperature error: {validation_metrics.critical_temperature_relative_error:.2f}%")
    print(f"   - Energy conservation: {validation_metrics.energy_conservation_score:.3f}")
    print(f"   - Magnetization conservation: {validation_metrics.magnetization_conservation_score:.3f}")
    
    # Generate validation report
    print("\n4. Generating Validation Report...")
    report_text = validator.generate_validation_report(validation_metrics)
    
    # Save report
    output_dir = Path("results/physics_validation_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "validation_report.txt", "w") as f:
        f.write(report_text)
    
    print(f"   - Report saved to: {output_dir / 'validation_report.txt'}")
    
    return latent_repr, order_param_candidates, phase_detection_result, validation_metrics


def demonstrate_visualization():
    """Demonstrate visualization and reporting system."""
    print("\n" + "="*60)
    print("VISUALIZATION AND REPORTING DEMONSTRATION")
    print("="*60)
    
    # Get data from validation demonstration
    latent_repr, order_param_candidates, phase_detection_result, validation_metrics = demonstrate_physics_validation()
    
    # Initialize visualizer
    visualizer = PublicationVisualizer(style='publication')
    
    print("\n1. Creating Publication-Ready Visualizations...")
    
    # Create output directory
    output_dir = Path("results/physics_validation_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate individual plots
    print("   - Latent space analysis...")
    fig1 = visualizer.plot_latent_space_analysis(latent_repr, order_param_candidates)
    visualizer.save_figure(fig1, "latent_space_analysis", str(output_dir))
    plt.close(fig1)
    
    print("   - Order parameter comparison...")
    fig2 = visualizer.plot_order_parameter_comparison(latent_repr, order_param_candidates)
    visualizer.save_figure(fig2, "order_parameter_comparison", str(output_dir))
    plt.close(fig2)
    
    print("   - Phase diagram...")
    fig3 = visualizer.plot_phase_diagram(latent_repr, phase_detection_result)
    visualizer.save_figure(fig3, "phase_diagram", str(output_dir))
    plt.close(fig3)
    
    print("   - Validation summary...")
    fig4 = visualizer.plot_validation_summary(validation_metrics)
    visualizer.save_figure(fig4, "validation_summary", str(output_dir))
    plt.close(fig4)
    
    print("\n2. Generating Comprehensive Analysis Report...")
    
    # Initialize reporter
    reporter = AnalysisReporter(output_dir=str(output_dir))
    
    # Generate comprehensive report
    report_metadata = reporter.generate_comprehensive_report(
        latent_repr, order_param_candidates, phase_detection_result,
        validation_metrics, report_name="physics_validation_demo"
    )
    
    print(f"   - Report directory: {report_metadata['report_directory']}")
    print(f"   - Generated figures: {list(report_metadata['figures'].keys())}")
    print(f"   - HTML report: {report_metadata['html_report']}")
    print(f"   - Summary statistics: {report_metadata['summary_statistics']}")
    
    # Display key results
    summary_stats = report_metadata['summary_stats']
    
    print("\n3. Summary of Results...")
    print(f"   - Dataset: {summary_stats['dataset_info']['n_samples']} samples")
    temp_range = summary_stats['dataset_info']['temperature_range']
    print(f"   - Temperature range: {temp_range[0]:.1f} - {temp_range[1]:.1f}")
    
    order_info = summary_stats['order_parameter_discovery']
    print(f"   - Primary order parameter: {order_info['primary_dimension']}")
    print(f"   - Magnetization correlation: {order_info['correlation_with_magnetization']:.3f}")
    print(f"   - Discovery confidence: {order_info['discovery_confidence']:.3f}")
    
    phase_info = summary_stats['phase_transition_detection']
    print(f"   - Discovered T_c: {phase_info['discovered_critical_temperature']:.3f}")
    print(f"   - Theoretical T_c: {phase_info['theoretical_critical_temperature']:.3f}")
    
    physics_info = summary_stats['physics_validation']
    print(f"   - Physics consistency: {physics_info['overall_consistency_score']:.3f}")
    print(f"   - Validation status: {physics_info['validation_status']}")
    
    return report_metadata


def main():
    """Main demonstration function."""
    print("Physics Validation and Visualization Example")
    print("=" * 50)
    print("This example demonstrates the comprehensive physics validation")
    print("and visualization system for AI-discovered physics results.")
    
    try:
        # Run demonstrations
        report_metadata = demonstrate_visualization()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nAll results saved to: {report_metadata['report_directory']}")
        print("\nGenerated files:")
        print("- validation_report.txt: Detailed physics validation report")
        print("- analysis_report.html: Interactive HTML report")
        print("- summary_statistics.json: Machine-readable summary")
        print("- Multiple publication-ready figures (PNG and PDF)")
        
        print("\nKey Features Demonstrated:")
        print("✓ Order parameter validation against theoretical magnetization")
        print("✓ Critical temperature accuracy testing (within 5% tolerance)")
        print("✓ Physics consistency scoring and statistical significance")
        print("✓ Publication-ready visualization generation")
        print("✓ Comprehensive analysis reporting")
        print("✓ HTML report with embedded figures and statistics")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()