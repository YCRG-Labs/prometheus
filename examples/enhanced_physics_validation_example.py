#!/usr/bin/env python3
"""
Enhanced Physics Validation Example

This example demonstrates how to use the enhanced physics validation system
with all advanced features including:
- Critical exponent analysis and finite-size scaling
- Symmetry validation and order parameter analysis
- Theoretical model validation (Ising, XY, Heisenberg)
- Statistical physics analysis with uncertainty quantification
- Experimental benchmark comparison
- Comprehensive physics review report generation
- Educational content and explanations
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from analysis.physics_validation import PhysicsValidator
from analysis.enhanced_validation_config import (
    EnhancedValidationConfig, ValidationLevel, get_default_config
)
from analysis.latent_analysis import LatentRepresentation
from analysis.phase_detection import PhaseDetectionResult
from analysis.order_parameter_discovery import OrderParameterCandidate, CorrelationResult
from analysis.theoretical_model_validator import TheoreticalModelValidator
from analysis.statistical_physics_analyzer import StatisticalPhysicsAnalyzer
from analysis.physics_review_report_generator import PhysicsReviewReportGenerator
from analysis.experimental_benchmark_comparator import ExperimentalBenchmarkComparator


def create_mock_data():
    """Create mock data for demonstration."""
    # Create mock latent representation
    n_samples = 1000
    temperatures = np.linspace(1.5, 3.0, n_samples)
    
    # Mock latent dimensions with phase transition behavior
    tc = 2.269  # Onsager critical temperature
    z1 = np.tanh(2 * (tc - temperatures)) + 0.1 * np.random.randn(n_samples)
    z2 = 0.5 * np.random.randn(n_samples)
    
    # Mock magnetizations
    magnetizations = np.tanh(2 * (tc - temperatures)) + 0.05 * np.random.randn(n_samples)
    
    latent_repr = LatentRepresentation(
        z1=z1,
        z2=z2,
        temperatures=temperatures,
        magnetizations=magnetizations,
        energies=-2 * np.ones(n_samples),  # Mock energies
        reconstruction_errors=0.01 * np.ones(n_samples)
    )
    
    # Create mock phase detection result
    phase_detection_result = PhaseDetectionResult(
        critical_temperature=tc + 0.05,  # Slight error for testing
        confidence=0.95,
        method='gradient',
        transition_region=(tc - 0.1, tc + 0.1)
    )
    
    # Create mock order parameter candidates
    correlation_result = CorrelationResult(
        correlation_coefficient=0.92,
        p_value=1e-10,
        latent_values=z1
    )
    
    order_param_candidate = OrderParameterCandidate(
        latent_dimension='z1',
        confidence_score=0.95,
        correlation_with_magnetization=correlation_result,
        is_valid_order_parameter=True
    )
    
    return latent_repr, phase_detection_result, [order_param_candidate]


def create_multi_size_mock_data():
    """Create mock data for multiple system sizes for finite-size scaling analysis."""
    system_sizes = [16, 32, 64, 128]
    multi_size_data = {}
    
    tc = 2.269  # Onsager critical temperature
    
    for L in system_sizes:
        n_samples = 500
        temperatures = np.linspace(2.0, 2.5, n_samples)
        
        # Mock finite-size scaling behavior
        # Order parameter scales as L^(-beta/nu) at Tc
        beta_over_nu = 0.125  # 2D Ising
        scaling_factor = L**(-beta_over_nu)
        
        z1 = scaling_factor * np.tanh(L**(1/1.0) * (tc - temperatures)) + 0.05 * np.random.randn(n_samples)
        z2 = 0.3 * np.random.randn(n_samples)
        
        magnetizations = scaling_factor * np.tanh(L**(1/1.0) * (tc - temperatures)) + 0.02 * np.random.randn(n_samples)
        
        multi_size_data[L] = LatentRepresentation(
            z1=z1,
            z2=z2,
            temperatures=temperatures,
            magnetizations=magnetizations,
            energies=-2 * np.ones(n_samples),
            reconstruction_errors=0.01 * np.ones(n_samples)
        )
    
    return multi_size_data


def create_experimental_benchmark_data():
    """Create mock experimental benchmark data for comparison."""
    # Mock experimental data for 2D Ising model
    experimental_data = {
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
    
    return experimental_data


def demonstrate_basic_validation():
    """Demonstrate basic validation with default settings."""
    print("=== Basic Enhanced Validation ===")
    
    # Create mock data
    latent_repr, phase_detection_result, order_param_candidates = create_mock_data()
    
    # Create validator
    validator = PhysicsValidator()
    
    # Use basic validation level
    config = get_default_config('basic')
    
    # Run validation
    metrics = validator.comprehensive_physics_validation(
        latent_repr=latent_repr,
        order_param_candidates=order_param_candidates,
        phase_detection_result=phase_detection_result,
        validation_config=config
    )
    
    print(f"Physics consistency score: {metrics.physics_consistency_score:.3f}")
    print(f"Critical temperature error: {metrics.critical_temperature_error:.3f}")
    print(f"Order parameter correlation: {metrics.order_parameter_correlation:.3f}")
    
    if metrics.enhanced_results:
        print(f"Enhanced features available: {list(metrics.enhanced_results.keys())}")
    
    print()


def demonstrate_comprehensive_validation():
    """Demonstrate comprehensive validation with all features."""
    print("=== Comprehensive Enhanced Validation ===")
    
    # Create mock data
    latent_repr, phase_detection_result, order_param_candidates = create_mock_data()
    
    # Create validator
    validator = PhysicsValidator()
    
    # Use comprehensive validation level
    config = get_default_config('comprehensive')
    
    # Enable experimental comparison for demonstration
    config.experimental_comparison.enable = True
    
    # Run validation
    metrics = validator.comprehensive_physics_validation(
        latent_repr=latent_repr,
        order_param_candidates=order_param_candidates,
        phase_detection_result=phase_detection_result,
        validation_config=config
    )
    
    print(f"Physics consistency score: {metrics.physics_consistency_score:.3f}")
    print(f"Critical temperature error: {metrics.critical_temperature_error:.3f}")
    print(f"Order parameter correlation: {metrics.order_parameter_correlation:.3f}")
    
    if metrics.enhanced_results:
        print(f"Enhanced features available: {list(metrics.enhanced_results.keys())}")
        
        # Show critical exponent results if available
        if 'critical_exponent_validation' in metrics.enhanced_results:
            crit_exp = metrics.enhanced_results['critical_exponent_validation']
            if crit_exp:
                print(f"Beta exponent: {crit_exp.beta_exponent:.4f} ± {(crit_exp.beta_confidence_interval[1] - crit_exp.beta_confidence_interval[0])/2:.4f}")
                print(f"Universality class match: {crit_exp.universality_class_match}")
        
        # Show symmetry validation results if available
        if 'symmetry_validation' in metrics.enhanced_results:
            sym_val = metrics.enhanced_results['symmetry_validation']
            if sym_val:
                print(f"Symmetry consistency score: {sym_val.symmetry_consistency_score:.3f}")
                print(f"Broken symmetries: {sym_val.broken_symmetries}")
    
    print()


def demonstrate_custom_configuration():
    """Demonstrate custom configuration creation."""
    print("=== Custom Configuration Example ===")
    
    # Create custom configuration
    config = EnhancedValidationConfig(validation_level=ValidationLevel.STANDARD)
    
    # Customize specific settings
    config.critical_exponents.bootstrap_samples = 2000
    config.statistical_analysis.confidence_level = 0.99
    config.experimental_comparison.enable = True
    config.report_generation.include_educational_content = True
    
    # Validate configuration
    warnings = config.validate_config()
    if warnings:
        print("Configuration warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    # Show configuration as dictionary
    config_dict = config.to_dict()
    print(f"Critical exponent bootstrap samples: {config_dict['critical_exponents']['bootstrap_samples']}")
    print(f"Statistical confidence level: {config_dict['statistical_analysis']['confidence_level']}")
    print(f"Experimental comparison enabled: {config_dict['experimental_comparison']['enable']}")
    
    print()


def demonstrate_configuration_persistence():
    """Demonstrate saving and loading configurations."""
    print("=== Configuration Persistence Example ===")
    
    # Create a custom configuration
    config = get_default_config('comprehensive')
    config.critical_exponents.bootstrap_samples = 5000
    config.performance.mode = config.performance.mode.__class__('thorough')
    
    # Save to file (would save if path exists)
    try:
        config.save_to_file('enhanced_validation_config.yaml')
        print("Configuration saved to enhanced_validation_config.yaml")
        
        # Load from file
        loaded_config = EnhancedValidationConfig.load_from_file('enhanced_validation_config.yaml')
        print("Configuration loaded successfully")
        print(f"Bootstrap samples: {loaded_config.critical_exponents.bootstrap_samples}")
        
    except Exception as e:
        print(f"File operations skipped: {e}")
    
    print()


def demonstrate_critical_exponent_analysis():
    """Demonstrate critical exponent analysis and finite-size scaling."""
    print("=== Critical Exponent Analysis ===")
    
    # Create multi-size data for finite-size scaling
    multi_size_data = create_multi_size_mock_data()
    latent_repr, phase_detection_result, order_param_candidates = create_mock_data()
    
    # Create validator with comprehensive config
    validator = PhysicsValidator()
    config = get_default_config('comprehensive')
    config.critical_exponents.enable = True
    config.finite_size_scaling.enable = True
    
    # Run validation
    metrics = validator.comprehensive_physics_validation(
        latent_repr=latent_repr,
        order_param_candidates=order_param_candidates,
        phase_detection_result=phase_detection_result,
        validation_config=config,
        multi_size_data=multi_size_data
    )
    
    if metrics.enhanced_results and 'critical_exponent_validation' in metrics.enhanced_results:
        crit_exp = metrics.enhanced_results['critical_exponent_validation']
        if crit_exp:
            print(f"Critical Exponents:")
            print(f"  β = {crit_exp.beta_exponent:.4f} ± {(crit_exp.beta_confidence_interval[1] - crit_exp.beta_confidence_interval[0])/2:.4f}")
            print(f"  γ = {crit_exp.gamma_exponent:.4f} ± {(crit_exp.gamma_confidence_interval[1] - crit_exp.gamma_confidence_interval[0])/2:.4f}")
            print(f"  ν = {crit_exp.nu_exponent:.4f} ± {(crit_exp.nu_confidence_interval[1] - crit_exp.nu_confidence_interval[0])/2:.4f}")
            print(f"  Universality class match: {crit_exp.universality_class_match}")
            if crit_exp.scaling_violations:
                print(f"  Scaling violations: {crit_exp.scaling_violations}")
    
    if metrics.enhanced_results and 'finite_size_scaling' in metrics.enhanced_results:
        fss = metrics.enhanced_results['finite_size_scaling']
        if fss:
            print(f"Finite-Size Scaling:")
            print(f"  Scaling collapse quality: {fss.scaling_collapse_quality:.3f}")
            print(f"  System sizes analyzed: {fss.system_sizes}")
    
    print()


def demonstrate_symmetry_analysis():
    """Demonstrate symmetry analysis and order parameter validation."""
    print("=== Symmetry Analysis ===")
    
    # Create mock data
    latent_repr, phase_detection_result, order_param_candidates = create_mock_data()
    
    # Create validator with symmetry analysis enabled
    validator = PhysicsValidator()
    config = get_default_config('comprehensive')
    config.symmetry_analysis.enable = True
    config.symmetry_analysis.hamiltonian_symmetries = ['Z2']  # Ising model symmetry
    
    # Run validation
    metrics = validator.comprehensive_physics_validation(
        latent_repr=latent_repr,
        order_param_candidates=order_param_candidates,
        phase_detection_result=phase_detection_result,
        validation_config=config
    )
    
    if metrics.enhanced_results and 'symmetry_validation' in metrics.enhanced_results:
        sym_val = metrics.enhanced_results['symmetry_validation']
        if sym_val:
            print(f"Symmetry Analysis Results:")
            print(f"  Broken symmetries: {sym_val.broken_symmetries}")
            print(f"  Order parameter symmetry: {sym_val.order_parameter_symmetry}")
            print(f"  Symmetry consistency score: {sym_val.symmetry_consistency_score:.3f}")
            if sym_val.violations:
                print(f"  Symmetry violations: {sym_val.violations}")
    
    print()


def demonstrate_theoretical_model_validation():
    """Demonstrate theoretical model validation."""
    print("=== Theoretical Model Validation ===")
    
    # Create mock data
    latent_repr, phase_detection_result, order_param_candidates = create_mock_data()
    
    # Create theoretical model validator
    model_validator = TheoreticalModelValidator()
    
    # Test Ising model validation
    print("Ising Model Validation:")
    try:
        ising_result = model_validator.validate_against_ising_model(
            latent_repr=latent_repr,
            system_size=64
        )
        print(f"  Agreement with Ising model: {ising_result.agreement_score:.3f}")
        print(f"  Critical temperature match: {ising_result.critical_temperature_match}")
        if ising_result.deviations:
            print(f"  Deviations: {ising_result.deviations}")
    except Exception as e:
        print(f"  Ising validation: {e}")
    
    # Test XY model validation
    print("XY Model Validation:")
    try:
        xy_result = model_validator.validate_against_xy_model(latent_repr=latent_repr)
        print(f"  Agreement with XY model: {xy_result.agreement_score:.3f}")
        print(f"  Vortex behavior detected: {xy_result.vortex_behavior_detected}")
    except Exception as e:
        print(f"  XY validation: {e}")
    
    print()


def demonstrate_statistical_analysis():
    """Demonstrate statistical physics analysis."""
    print("=== Statistical Physics Analysis ===")
    
    # Create mock ensemble data (multiple simulation runs)
    ensemble_data = []
    for i in range(5):
        latent_repr, _, _ = create_mock_data()
        ensemble_data.append(latent_repr)
    
    # Create statistical analyzer
    stat_analyzer = StatisticalPhysicsAnalyzer()
    
    # Perform ensemble analysis
    try:
        ensemble_result = stat_analyzer.perform_ensemble_analysis(ensemble_data)
        print(f"Ensemble Analysis Results:")
        print(f"  Mean critical temperature: {ensemble_result.mean_critical_temperature:.4f}")
        print(f"  Critical temperature std: {ensemble_result.critical_temperature_std:.4f}")
        print(f"  Ensemble consistency score: {ensemble_result.consistency_score:.3f}")
        
        if ensemble_result.confidence_intervals:
            print(f"  95% Confidence intervals:")
            for param, interval in ensemble_result.confidence_intervals.items():
                print(f"    {param}: [{interval[0]:.4f}, {interval[1]:.4f}]")
    except Exception as e:
        print(f"  Ensemble analysis: {e}")
    
    # Test hypothesis testing
    observed_values = {'critical_temperature': 2.274, 'beta_exponent': 0.128}
    theoretical_values = {'critical_temperature': 2.269, 'beta_exponent': 0.125}
    
    try:
        hypothesis_results = stat_analyzer.test_physics_hypotheses(
            observed_values=observed_values,
            theoretical_values=theoretical_values
        )
        print(f"Hypothesis Testing:")
        for param, result in hypothesis_results.test_results.items():
            print(f"  {param}: p-value = {result.p_value:.4f}, significant = {result.is_significant}")
    except Exception as e:
        print(f"  Hypothesis testing: {e}")
    
    print()


def demonstrate_experimental_comparison():
    """Demonstrate experimental benchmark comparison."""
    print("=== Experimental Benchmark Comparison ===")
    
    # Create mock data and experimental benchmarks
    latent_repr, phase_detection_result, order_param_candidates = create_mock_data()
    experimental_data = create_experimental_benchmark_data()
    
    # Create experimental comparator
    exp_comparator = ExperimentalBenchmarkComparator()
    
    # Add experimental benchmark
    try:
        exp_comparator.add_experimental_benchmark(
            name="2D_Ising_Onsager",
            data=experimental_data
        )
        
        # Perform comparison
        comparison_result = exp_comparator.compare_with_experiments(
            computational_results={
                'critical_temperature': 2.274,
                'beta_exponent': 0.128,
                'gamma_exponent': 1.73,
                'nu_exponent': 0.98
            },
            benchmark_name="2D_Ising_Onsager"
        )
        
        print(f"Experimental Comparison Results:")
        print(f"  Overall agreement score: {comparison_result.overall_agreement:.3f}")
        print(f"  Statistical significance: {comparison_result.statistical_significance:.4f}")
        
        for param, comp in comparison_result.parameter_comparisons.items():
            print(f"  {param}:")
            print(f"    Computational: {comp.computational_value:.4f}")
            print(f"    Experimental: {comp.experimental_value:.4f} ± {comp.experimental_uncertainty:.4f}")
            print(f"    Agreement: {comp.agreement_metric:.3f}")
            
        if comparison_result.discrepancies:
            print(f"  Discrepancies found: {len(comparison_result.discrepancies)}")
            for disc in comparison_result.discrepancies[:2]:  # Show first 2
                print(f"    - {disc.parameter}: {disc.explanation}")
                
    except Exception as e:
        print(f"  Experimental comparison: {e}")
    
    print()


def demonstrate_physics_review_report():
    """Demonstrate comprehensive physics review report generation."""
    print("=== Physics Review Report Generation ===")
    
    # Create mock data
    latent_repr, phase_detection_result, order_param_candidates = create_mock_data()
    
    # Create validator with comprehensive config
    validator = PhysicsValidator()
    config = get_default_config('comprehensive')
    config.report_generation.enable = True
    config.report_generation.include_educational_content = True
    config.report_generation.include_visualizations = True
    
    # Run comprehensive validation
    metrics = validator.comprehensive_physics_validation(
        latent_repr=latent_repr,
        order_param_candidates=order_param_candidates,
        phase_detection_result=phase_detection_result,
        validation_config=config
    )
    
    # Generate physics review report
    report_generator = PhysicsReviewReportGenerator()
    
    try:
        report = report_generator.generate_comprehensive_report(
            validation_results={
                'basic_metrics': metrics,
                'enhanced_results': metrics.enhanced_results or {}
            },
            include_educational_content=True
        )
        
        print(f"Physics Review Report Generated:")
        print(f"  Overall assessment: {report.overall_assessment}")
        print(f"  Number of sections: {len([s for s in [report.theoretical_consistency, report.order_parameter_validation, report.critical_behavior_analysis] if s])}")
        print(f"  Physics violations found: {len(report.violations)}")
        
        if report.violations:
            print(f"  Violation summary:")
            for violation in report.violations[:3]:  # Show first 3
                print(f"    - {violation.violation_type} ({violation.severity}): {violation.description}")
        
        if report.educational_content:
            print(f"  Educational topics covered: {len(report.educational_content)}")
            for topic in list(report.educational_content.keys())[:3]:  # Show first 3
                print(f"    - {topic}")
        
        if report.visualizations:
            print(f"  Visualizations generated: {len(report.visualizations)}")
            
    except Exception as e:
        print(f"  Report generation: {e}")
    
    print()


def demonstrate_educational_content():
    """Demonstrate educational content generation."""
    print("=== Educational Content Generation ===")
    
    # Create report generator
    report_generator = PhysicsReviewReportGenerator()
    
    # Generate educational explanations
    physics_concepts = [
        'critical_exponents',
        'universality_classes',
        'finite_size_scaling',
        'order_parameters',
        'phase_transitions'
    ]
    
    try:
        explanations = report_generator.generate_educational_explanations(physics_concepts)
        
        print(f"Educational Content Generated:")
        for concept, explanation in explanations.items():
            print(f"  {concept.replace('_', ' ').title()}:")
            print(f"    {explanation[:100]}..." if len(explanation) > 100 else f"    {explanation}")
            print()
            
    except Exception as e:
        print(f"  Educational content generation: {e}")
    
    print()


def demonstrate_legacy_compatibility():
    """Demonstrate backward compatibility with legacy dictionary configuration."""
    print("=== Legacy Compatibility Example ===")
    
    # Create mock data
    latent_repr, phase_detection_result, order_param_candidates = create_mock_data()
    
    # Create validator
    validator = PhysicsValidator()
    
    # Use legacy dictionary configuration
    legacy_config = {
        'validation_level': 'standard',
        'enable_enhanced_features': True,
        'enable_theoretical_validation': True,
        'enable_statistical_analysis': True,
        'enable_experimental_comparison': False,
        'confidence_level': 0.95,
        'n_bootstrap': 1000,
        'dimensionality': 2,
        'hamiltonian_symmetries': ['Z2']
    }
    
    # Run validation with legacy config
    metrics = validator.comprehensive_physics_validation(
        latent_repr=latent_repr,
        order_param_candidates=order_param_candidates,
        phase_detection_result=phase_detection_result,
        validation_config=legacy_config
    )
    
    print(f"Legacy validation completed successfully")
    print(f"Physics consistency score: {metrics.physics_consistency_score:.3f}")
    
    print()


def main():
    """Run all demonstration examples."""
    print("Enhanced Physics Validation System Demonstration")
    print("=" * 50)
    print("This example demonstrates all enhanced physics validation features:")
    print("- Critical exponent analysis and finite-size scaling")
    print("- Symmetry validation and order parameter analysis") 
    print("- Theoretical model validation (Ising, XY, Heisenberg)")
    print("- Statistical physics analysis with uncertainty quantification")
    print("- Experimental benchmark comparison")
    print("- Comprehensive physics review report generation")
    print("- Educational content and explanations")
    print()
    
    try:
        # Basic demonstrations
        demonstrate_basic_validation()
        demonstrate_comprehensive_validation()
        
        # Enhanced feature demonstrations
        demonstrate_critical_exponent_analysis()
        demonstrate_symmetry_analysis()
        demonstrate_theoretical_model_validation()
        demonstrate_statistical_analysis()
        demonstrate_experimental_comparison()
        demonstrate_physics_review_report()
        demonstrate_educational_content()
        
        # Configuration demonstrations
        demonstrate_custom_configuration()
        demonstrate_configuration_persistence()
        demonstrate_legacy_compatibility()
        
        print("=" * 50)
        print("All demonstrations completed successfully!")
        print("The enhanced physics validation system provides comprehensive")
        print("validation capabilities for ensuring physical correctness of")
        print("discovered phase transitions and order parameters.")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()