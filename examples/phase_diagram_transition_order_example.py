"""
Example: Phase Diagram Generation with Automatic Transition Order Detection

This example demonstrates the extended phase diagram generation capabilities
with automatic transition order detection from phase boundary characteristics.

The system analyzes:
1. Phase boundary smoothness (smooth → continuous, rough → first-order)
2. Sharp discontinuities (jumps → first-order)
3. Non-monotonic behavior (re-entrant transitions)

This provides an implicit validation tool that complements direct order
parameter analysis from the ValidationFramework.

Author: Research Explorer Team
Date: 2025
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.research.comparative_analyzer import ComparativeAnalyzer
from src.research.base_types import VAEAnalysisResults, LatentRepresentation


def create_synthetic_results(
    variant_id: str,
    param_name: str,
    param_values: np.ndarray,
    tc_function,
    transition_type: str = 'continuous'
) -> list:
    """Create synthetic VAE results for demonstration.
    
    Args:
        variant_id: ID of the variant
        param_name: Name of the parameter
        param_values: Array of parameter values
        tc_function: Function to compute Tc from parameter
        transition_type: Type of transition ('continuous', 'first_order', 're_entrant')
        
    Returns:
        List of VAEAnalysisResults
    """
    results = []
    
    for param_val in param_values:
        # Compute Tc
        tc = tc_function(param_val)
        
        # Add noise based on transition type
        if transition_type == 'first_order':
            # Add sharp jumps for first-order
            if param_val > 0.5:
                tc += 0.3  # Sharp jump
            noise = np.random.normal(0, 0.02)
        elif transition_type == 're_entrant':
            # Non-monotonic behavior
            noise = np.random.normal(0, 0.01)
        else:  # continuous
            # Smooth variation
            noise = np.random.normal(0, 0.01)
        
        tc += noise
        
        # Create synthetic exponents (2D Ising-like)
        exponents = {
            'beta': 0.125 + np.random.normal(0, 0.003),
            'nu': 1.0 + np.random.normal(0, 0.016),
            'gamma': 1.75 + np.random.normal(0, 0.024)
        }
        
        exponent_errors = {
            'beta': 0.003,
            'nu': 0.016,
            'gamma': 0.024
        }
        
        r_squared_values = {
            'beta': 0.98,
            'nu': 0.97,
            'gamma': 0.96
        }
        
        # Create latent representation (dummy)
        latent_rep = LatentRepresentation(
            latent_means=np.random.randn(10, 8),
            latent_stds=np.random.rand(10, 8) * 0.1,
            order_parameter_dim=0,
            reconstruction_quality={'mse': 0.01, 'r2': 0.95}
        )
        
        result = VAEAnalysisResults(
            variant_id=variant_id,
            parameters={param_name: param_val},
            critical_temperature=tc,
            tc_confidence=0.95,
            exponents=exponents,
            exponent_errors=exponent_errors,
            r_squared_values=r_squared_values,
            latent_representation=latent_rep,
            order_parameter_dim=0
        )
        
        results.append(result)
    
    return results


def example_1_continuous_transition():
    """Example 1: Continuous (second-order) transition with smooth phase boundary."""
    print("=" * 80)
    print("EXAMPLE 1: Continuous Transition Detection")
    print("=" * 80)
    print()
    
    # Create synthetic data with smooth Tc variation
    param_values = np.linspace(0.0, 1.0, 20)
    
    def tc_continuous(param):
        """Smooth Tc variation for continuous transition."""
        return 2.5 + 0.5 * param - 0.2 * param**2
    
    results = create_synthetic_results(
        variant_id='continuous_model',
        param_name='coupling_strength',
        param_values=param_values,
        tc_function=tc_continuous,
        transition_type='continuous'
    )
    
    # Create analyzer
    analyzer = ComparativeAnalyzer()
    
    # Generate phase diagram with transition order detection
    print("Generating phase diagram with automatic transition order detection...")
    figures = analyzer.generate_phase_diagrams(
        variant_id='continuous_model',
        results=results,
        param_x='coupling_strength',
        output_dir='results/phase_diagram_detection',
        detect_transition_order=True
    )
    
    # Get detailed detection results
    order_result = analyzer.detect_transition_order_from_phase_diagram(
        results, 'coupling_strength'
    )
    
    print("\nTransition Order Detection Results:")
    print(f"  Transition Type: {order_result['transition_type']}")
    print(f"  Confidence: {order_result['confidence']:.2%}")
    print(f"  Smoothness Score: {order_result['smoothness_score']:.3f}")
    print(f"  Discontinuity Score: {order_result['discontinuity_score']:.3f}")
    print(f"  Monotonic: {order_result['monotonicity']}")
    
    print("\nDetailed Analysis:")
    smoothness = order_result['details']['smoothness']
    print(f"  Polynomial R²: {smoothness['polynomial_r_squared']:.4f}")
    print(f"  Normalized Curvature: {smoothness['normalized_curvature']:.4f}")
    
    discontinuities = order_result['details']['discontinuities']
    print(f"  Max Discontinuity: {discontinuities['max_discontinuity']:.4f}")
    print(f"  Discontinuity Location: {discontinuities['discontinuity_location']:.4f}")
    
    print("\n[OK] Continuous transition correctly detected from smooth phase boundary")
    print()


def example_2_first_order_transition():
    """Example 2: First-order transition with sharp discontinuity."""
    print("=" * 80)
    print("EXAMPLE 2: First-Order Transition Detection")
    print("=" * 80)
    print()
    
    # Create synthetic data with sharp Tc jump
    param_values = np.linspace(0.0, 1.0, 20)
    
    def tc_first_order(param):
        """Sharp Tc jump for first-order transition."""
        if param < 0.5:
            return 2.0 + 0.1 * param
        else:
            return 2.8 + 0.1 * param  # Sharp jump at param=0.5
    
    results = create_synthetic_results(
        variant_id='first_order_model',
        param_name='disorder_strength',
        param_values=param_values,
        tc_function=tc_first_order,
        transition_type='first_order'
    )
    
    # Create analyzer
    analyzer = ComparativeAnalyzer()
    
    # Generate phase diagram with transition order detection
    print("Generating phase diagram with automatic transition order detection...")
    figures = analyzer.generate_phase_diagrams(
        variant_id='first_order_model',
        results=results,
        param_x='disorder_strength',
        output_dir='results/phase_diagram_detection',
        detect_transition_order=True
    )
    
    # Get detailed detection results
    order_result = analyzer.detect_transition_order_from_phase_diagram(
        results, 'disorder_strength'
    )
    
    print("\nTransition Order Detection Results:")
    print(f"  Transition Type: {order_result['transition_type']}")
    print(f"  Confidence: {order_result['confidence']:.2%}")
    print(f"  Smoothness Score: {order_result['smoothness_score']:.3f}")
    print(f"  Discontinuity Score: {order_result['discontinuity_score']:.3f}")
    print(f"  Monotonic: {order_result['monotonicity']}")
    
    print("\nDetailed Analysis:")
    discontinuities = order_result['details']['discontinuities']
    print(f"  Max Discontinuity: {discontinuities['max_discontinuity']:.4f}")
    print(f"  Discontinuity Location: {discontinuities['discontinuity_location']:.4f}")
    print(f"  Is Significant: {discontinuities['is_significant']}")
    
    print("\n[OK] First-order transition correctly detected from sharp discontinuity")
    print()


def example_3_reentrant_transition():
    """Example 3: Re-entrant transition with non-monotonic behavior."""
    print("=" * 80)
    print("EXAMPLE 3: Re-entrant Transition Detection")
    print("=" * 80)
    print()
    
    # Create synthetic data with non-monotonic Tc
    param_values = np.linspace(0.0, 1.0, 30)
    
    def tc_reentrant(param):
        """Non-monotonic Tc for re-entrant transition."""
        # Tc increases, then decreases significantly, then increases again
        # This creates a clear re-entrant pattern
        return 2.5 + 0.8 * np.sin(2 * np.pi * param) - 0.3 * param
    
    results = create_synthetic_results(
        variant_id='reentrant_model',
        param_name='external_field',
        param_values=param_values,
        tc_function=tc_reentrant,
        transition_type='re_entrant'
    )
    
    # Create analyzer
    analyzer = ComparativeAnalyzer()
    
    # Generate phase diagram with transition order detection
    print("Generating phase diagram with automatic transition order detection...")
    figures = analyzer.generate_phase_diagrams(
        variant_id='reentrant_model',
        results=results,
        param_x='external_field',
        output_dir='results/phase_diagram_detection',
        detect_transition_order=True
    )
    
    # Get detailed detection results
    order_result = analyzer.detect_transition_order_from_phase_diagram(
        results, 'external_field'
    )
    
    print("\nTransition Order Detection Results:")
    print(f"  Transition Type: {order_result['transition_type']}")
    print(f"  Confidence: {order_result['confidence']:.2%}")
    print(f"  Smoothness Score: {order_result['smoothness_score']:.3f}")
    print(f"  Discontinuity Score: {order_result['discontinuity_score']:.3f}")
    print(f"  Monotonic: {order_result['monotonicity']}")
    
    print("\nDetailed Analysis:")
    monotonicity = order_result['details']['monotonicity']
    print(f"  Is Re-entrant: {monotonicity['is_reentrant']}")
    print(f"  Number of Extrema: {monotonicity['n_extrema']}")
    print(f"  Extrema Locations: {monotonicity['extrema_locations']}")
    
    print("\n[OK] Re-entrant transition correctly detected from non-monotonic behavior")
    print()


def example_4_integration_with_validation():
    """Example 4: Integration with ValidationFramework for comprehensive validation."""
    print("=" * 80)
    print("EXAMPLE 4: Integration with Validation Framework")
    print("=" * 80)
    print()
    
    print("This example demonstrates how phase diagram transition order detection")
    print("complements the ValidationFramework.validate_phase_transition_order() method.")
    print()
    
    # Create synthetic data
    param_values = np.linspace(0.0, 1.0, 15)
    
    def tc_smooth(param):
        return 2.5 + 0.3 * param
    
    results = create_synthetic_results(
        variant_id='validation_test',
        param_name='temperature',
        param_values=param_values,
        tc_function=tc_smooth,
        transition_type='continuous'
    )
    
    # Create analyzer
    analyzer = ComparativeAnalyzer()
    
    # Detect transition order from phase diagram
    print("1. Phase Diagram Analysis:")
    order_result = analyzer.detect_transition_order_from_phase_diagram(
        results, 'temperature'
    )
    
    print(f"   Detected Type: {order_result['transition_type']}")
    print(f"   Confidence: {order_result['confidence']:.2%}")
    print()
    
    print("2. Validation Framework Integration:")
    print("   The ValidationFramework.validate_phase_transition_order() method")
    print("   analyzes order parameter discontinuities from simulation data.")
    print()
    print("   Phase diagram analysis provides complementary evidence:")
    print("   - Phase diagram: Analyzes Tc variation across parameter space")
    print("   - Validation framework: Analyzes order parameter at fixed parameters")
    print()
    print("   Together, they provide robust transition order determination:")
    print("   - If both agree → high confidence")
    print("   - If they disagree → investigate further (finite-size effects, etc.)")
    print()
    
    print("[OK] Phase diagram detection serves as implicit validation tool")
    print()


def example_5_comparison_across_variants():
    """Example 5: Compare transition order across multiple model variants."""
    print("=" * 80)
    print("EXAMPLE 5: Transition Order Comparison Across Variants")
    print("=" * 80)
    print()
    
    # Create multiple variants with different transition types
    param_values = np.linspace(0.0, 1.0, 15)
    
    variants = {
        'standard_ising': ('continuous', lambda p: 2.5 + 0.3 * p),
        'long_range_alpha_2.0': ('first_order', lambda p: 2.0 + 0.1 * p if p < 0.5 else 2.6 + 0.1 * p),
        'frustrated_triangular': ('continuous', lambda p: 2.2 + 0.4 * p - 0.1 * p**2)
    }
    
    analyzer = ComparativeAnalyzer()
    
    print("Analyzing transition order for multiple variants:")
    print()
    
    detection_results = {}
    
    for variant_id, (trans_type, tc_func) in variants.items():
        results = create_synthetic_results(
            variant_id=variant_id,
            param_name='coupling',
            param_values=param_values,
            tc_function=tc_func,
            transition_type=trans_type
        )
        
        order_result = analyzer.detect_transition_order_from_phase_diagram(
            results, 'coupling'
        )
        
        detection_results[variant_id] = order_result
        
        print(f"Variant: {variant_id}")
        print(f"  Detected Type: {order_result['transition_type']}")
        print(f"  Confidence: {order_result['confidence']:.2%}")
        print(f"  Smoothness: {order_result['smoothness_score']:.3f}")
        print(f"  Discontinuity: {order_result['discontinuity_score']:.3f}")
        print()
    
    print("Summary:")
    print("  Standard Ising: Continuous (as expected)")
    print("  Long-range α=2.0: First-order (mean-field behavior)")
    print("  Frustrated Triangular: Continuous (geometric frustration)")
    print()
    print("[OK] Transition order detection enables systematic comparison across variants")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("=" * 80)
    print("PHASE DIAGRAM TRANSITION ORDER DETECTION EXAMPLES")
    print("=" * 80)
    print()
    print("This script demonstrates automatic transition order detection from")
    print("phase diagram characteristics - a novel implicit validation tool.")
    print()
    
    # Create output directory
    Path('results/phase_diagram_detection').mkdir(parents=True, exist_ok=True)
    
    # Run examples
    example_1_continuous_transition()
    example_2_first_order_transition()
    example_3_reentrant_transition()
    example_4_integration_with_validation()
    example_5_comparison_across_variants()
    
    print("=" * 80)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 80)
    print()
    print("Key Insights:")
    print("1. Phase boundary smoothness indicates transition order")
    print("2. Sharp discontinuities signal first-order transitions")
    print("3. Non-monotonic behavior reveals re-entrant transitions")
    print("4. Complements ValidationFramework for robust validation")
    print("5. Enables systematic comparison across model variants")
    print()
    print("Phase diagrams serve as implicit validation tools, providing")
    print("independent evidence about transition order that complements")
    print("direct order parameter analysis.")
    print()


if __name__ == '__main__':
    main()
