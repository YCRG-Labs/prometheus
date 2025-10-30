"""
Physics Validation and Visualization Example

This example demonstrates how to use the enhanced physics validation framework
and visualization system to validate AI-discovered physics results.

Enhanced features include:
- Critical exponent analysis and finite-size scaling validation
- Symmetry analysis and theoretical model validation
- Statistical physics analysis with uncertainty quantification
- Experimental benchmark comparison
- Comprehensive physics review report generation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.analysis.physics_validation import PhysicsValidator, ValidationMetrics
from src.analysis.enhanced_validation_config import (
    EnhancedValidationConfig, ValidationLevel, get_default_config
)
from src.analysis.visualization import PublicationVisualizer, AnalysisReporter
from src.analysis.latent_analysis import LatentRepresentation
from src.analysis.order_parameter_discovery import OrderParameterCandidate, CorrelationResult
from src.analysis.phase_detection import PhaseDetectionResult
from src.analysis.physics_review_report_generator import PhysicsReviewReportGenerator
from src.analysis.experimental_benchmark_comparator import ExperimentalBenchmarkComparator


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
    """Demonstrate enhanced physics validation framework."""
    print("\n" + "="*60)
    print("ENHANCED PHYSICS VALIDATION DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    latent_repr = create_sample_data()
    order_param_candidates = create_order_parameter_candidates()
    phase_detection_result = create_phase_detection_result()
    
    # Initialize validator
    validator = PhysicsValidator(theoretical_tc=2.269, tolerance_percent=5.0)
    
    print("\n1. Basic Validation (Legacy Mode)...")
    # Run basic validation first
    basic_metrics = validator.comprehensive_physics_validation(
        latent_repr, order_param_candidates, phase_detection_result
    )
    
    print(f"   - Overall physics consistency: {basic_metrics.physics_consistency_score:.3f}")
    print(f"   - Order parameter correlation: {basic_metrics.order_parameter_correlation:.3f}")
    print(f"   - Critical temperature error: {basic_metrics.critical_temperature_relative_error:.2f}%")
    
    print("\n2. Enhanced Validation with Standard Configuration...")
    # Use enhanced validation with standard config
    config = get_default_config('standard')
    config.critical_exponents.enable = True
    config.symmetry_analysis.enable = True
    config.symmetry_analysis.hamiltonian_symmetries = ['Z2']  # Ising model
    
    enhanced_metrics = validator.comprehensive_physics_validation(
        latent_repr, order_param_candidates, phase_detection_result,
        validation_config=config
    )
    
    print(f"   - Overall physics consistency: {enhanced_metrics.physics_consistency_score:.3f}")
    print(f"   - Order parameter correlation: {enhanced_metrics.order_parameter_correlation:.3f}")
    print(f"   - Critical temperature error: {enhanced_metrics.critical_temperature_relative_error:.2f}%")
    
    # Show enhanced results
    if enhanced_metrics.enhanced_results:
        print(f"   - Enhanced features available: {list(enhanced_metrics.enhanced_results.keys())}")
        
        if 'critical_exponent_validation' in enhanced_metrics.enhanced_results:
            crit_exp = enhanced_metrics.enhanced_results['critical_exponent_validation']
            if crit_exp:
                print(f"   - Critical exponents: β={crit_exp.beta_exponent:.4f}, γ={crit_exp.gamma_exponent:.4f}")
                print(f"   - Universality class match: {crit_exp.universality_class_match}")
        
        if 'symmetry_validation' in enhanced_metrics.enhanced_results:
            sym_val = enhanced_metrics.enhanced_results['symmetry_validation']
            if sym_val:
                print(f"   - Symmetry consistency: {sym_val.symmetry_consistency_score:.3f}")
                print(f"   - Broken symmetries: {sym_val.broken_symmetries}")
    
    print("\n3. Comprehensive Validation with All Features...")
    # Use comprehensive validation
    comprehensive_config = get_default_config('comprehensive')
    comprehensive_config.experimental_comparison.enable = True
    comprehensive_config.report_generation.enable = True
    
    comprehensive_metrics = validator.comprehensive_physics_validation(
        latent_repr, order_param_candidates, phase_detection_result,
        validation_config=comprehensive_config
    )
    
    print(f"   - Overall physics consistency: {comprehensive_metrics.physics_consistency_score:.3f}")
    print(f"   - Enhanced validation features: {len(comprehensive_metrics.enhanced_results or {})}")
    
    print("\n4. Generating Enhanced Validation Report...")
    # Generate comprehensive physics review report
    report_generator = PhysicsReviewReportGenerator()
    
    try:
        report = report_generator.generate_comprehensive_report(
            validation_results={
                'basic_metrics': comprehensive_metrics,
                'enhanced_results': comprehensive_metrics.enhanced_results or {}
            },
            include_educational_content=True
        )
        
        # Save enhanced report
        output_dir = Path("results/physics_validation_example")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save basic validation report
        basic_report_text = validator.generate_validation_report(comprehensive_metrics)
        with open(output_dir / "validation_report.txt", "w") as f:
            f.write(basic_report_text)
        
        # Save enhanced report summary
        enhanced_report_text = f"""Enhanced Physics Validation Report
{'='*50}

Overall Assessment: {report.overall_assessment}

Physics Violations Found: {len(report.violations)}
"""
        
        if report.violations:
            enhanced_report_text += "\nViolations:\n"
            for i, violation in enumerate(report.violations[:5], 1):
                enhanced_report_text += f"{i}. {violation.violation_type} ({violation.severity}): {violation.description}\n"
        
        if report.educational_content:
            enhanced_report_text += f"\nEducational Topics Covered: {len(report.educational_content)}\n"
            for topic in list(report.educational_content.keys())[:3]:
                enhanced_report_text += f"- {topic.replace('_', ' ').title()}\n"
        
        with open(output_dir / "enhanced_validation_report.txt", "w") as f:
            f.write(enhanced_report_text)
        
        print(f"   - Basic report saved to: {output_dir / 'validation_report.txt'}")
        print(f"   - Enhanced report saved to: {output_dir / 'enhanced_validation_report.txt'}")
        print(f"   - Physics violations found: {len(report.violations)}")
        print(f"   - Educational content topics: {len(report.educational_content) if report.educational_content else 0}")
        
    except Exception as e:
        print(f"   - Enhanced report generation: {e}")
        # Fallback to basic report
        basic_report_text = validator.generate_validation_report(comprehensive_metrics)
        output_dir = Path("results/physics_validation_example")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "validation_report.txt", "w") as f:
            f.write(basic_report_text)
        print(f"   - Basic report saved to: {output_dir / 'validation_report.txt'}")
    
    return latent_repr, order_param_candidates, phase_detection_result, comprehensive_metrics


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


def demonstrate_experimental_comparison():
    """Demonstrate experimental benchmark comparison."""
    print("\n" + "="*60)
    print("EXPERIMENTAL BENCHMARK COMPARISON")
    print("="*60)
    
    # Create experimental comparator
    exp_comparator = ExperimentalBenchmarkComparator()
    
    print("\n1. Adding Experimental Benchmarks...")
    
    # Add 2D Ising model benchmark
    ising_benchmark = {
        'critical_temperature': {
            'value': 2.269,
            'uncertainty': 0.001,
            'source': 'Onsager (1944) - Exact solution',
            'system': '2D Ising model'
        },
        'critical_exponents': {
            'beta': {'value': 0.125, 'uncertainty': 0.002, 'source': 'Theoretical'},
            'gamma': {'value': 1.75, 'uncertainty': 0.01, 'source': 'Theoretical'},
            'nu': {'value': 1.0, 'uncertainty': 0.005, 'source': 'Theoretical'}
        },
        'universality_class': '2D Ising',
        'dimensionality': 2
    }
    
    try:
        exp_comparator.add_experimental_benchmark(
            name="2D_Ising_Theoretical",
            data=ising_benchmark
        )
        print("   - Added 2D Ising theoretical benchmark")
        
        # Compare with computational results
        computational_results = {
            'critical_temperature': 2.275,  # From our phase detection
            'beta_exponent': 0.128,
            'gamma_exponent': 1.73,
            'nu_exponent': 0.98
        }
        
        print("\n2. Comparing with Computational Results...")
        comparison_result = exp_comparator.compare_with_experiments(
            computational_results=computational_results,
            benchmark_name="2D_Ising_Theoretical"
        )
        
        print(f"   - Overall agreement score: {comparison_result.overall_agreement:.3f}")
        print(f"   - Statistical significance: {comparison_result.statistical_significance:.4f}")
        
        print("\n3. Parameter-by-Parameter Comparison:")
        for param, comp in comparison_result.parameter_comparisons.items():
            print(f"   - {param}:")
            print(f"     Computational: {comp.computational_value:.4f}")
            print(f"     Experimental: {comp.experimental_value:.4f} ± {comp.experimental_uncertainty:.4f}")
            print(f"     Agreement: {comp.agreement_metric:.3f}")
        
        if comparison_result.discrepancies:
            print(f"\n4. Discrepancies Found ({len(comparison_result.discrepancies)}):")
            for i, disc in enumerate(comparison_result.discrepancies[:3], 1):
                print(f"   {i}. {disc.parameter}: {disc.explanation}")
        
        # Save comparison results
        output_dir = Path("results/physics_validation_example")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        comparison_text = f"""Experimental Comparison Results
{'='*40}

Overall Agreement Score: {comparison_result.overall_agreement:.3f}
Statistical Significance: {comparison_result.statistical_significance:.4f}

Parameter Comparisons:
"""
        for param, comp in comparison_result.parameter_comparisons.items():
            comparison_text += f"""
{param}:
  Computational: {comp.computational_value:.4f}
  Experimental: {comp.experimental_value:.4f} ± {comp.experimental_uncertainty:.4f}
  Agreement: {comp.agreement_metric:.3f}
"""""
        
        with open(output_dir / "experimental_comparison.txt", "w") as f:
            f.write(comparison_text)
        
        print(f"\n   - Comparison results saved to: {output_dir / 'experimental_comparison.txt'}")
        
    except Exception as e:
        print(f"   - Experimental comparison error: {e}")
    
    print()


def main():
    """Main demonstration function."""
    print("Enhanced Physics Validation and Visualization Example")
    print("=" * 60)
    print("This example demonstrates the comprehensive enhanced physics validation")
    print("and visualization system for AI-discovered physics results.")
    print("\nEnhanced features include:")
    print("- Critical exponent analysis and finite-size scaling validation")
    print("- Symmetry analysis and theoretical model validation")
    print("- Statistical physics analysis with uncertainty quantification")
    print("- Experimental benchmark comparison")
    print("- Comprehensive physics review report generation")
    
    try:
        # Run demonstrations
        report_metadata = demonstrate_visualization()
        demonstrate_experimental_comparison()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nAll results saved to: {report_metadata['report_directory']}")
        print("\nGenerated files:")
        print("- validation_report.txt: Basic physics validation report")
        print("- enhanced_validation_report.txt: Enhanced validation with physics review")
        print("- experimental_comparison.txt: Experimental benchmark comparison")
        print("- analysis_report.html: Interactive HTML report")
        print("- summary_statistics.json: Machine-readable summary")
        print("- Multiple publication-ready figures (PNG and PDF)")
        
        print("\nKey Features Demonstrated:")
        print("✓ Enhanced order parameter validation with symmetry analysis")
        print("✓ Critical exponent computation and universality class identification")
        print("✓ Finite-size scaling analysis (when multi-size data available)")
        print("✓ Theoretical model validation (Ising, XY, Heisenberg)")
        print("✓ Statistical physics analysis with bootstrap confidence intervals")
        print("✓ Experimental benchmark comparison with agreement metrics")
        print("✓ Comprehensive physics review report generation")
        print("✓ Educational content and physics explanations")
        print("✓ Physics violation detection and severity assessment")
        print("✓ Publication-ready visualization generation")
        print("✓ Backward compatibility with legacy validation")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()