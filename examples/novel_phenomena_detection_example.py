"""
Example: Novel Phenomena Detection

This example demonstrates the novel phenomena detection capabilities of the
research explorer, including detection of anomalous exponents, first-order
transitions, and other unusual phase transition behavior.
"""

import numpy as np
from pathlib import Path

from src.research import (
    NovelPhenomenonDetector,
    UniversalityClass,
    VAEAnalysisResults,
    SimulationData,
)
from src.research.base_types import LatentRepresentation


def create_mock_vae_results(
    variant_id: str,
    beta: float,
    beta_error: float = 0.02,
    tc: float = 4.5,
    r_squared: float = 0.95
) -> VAEAnalysisResults:
    """Create mock VAE results for testing."""
    
    # Create mock latent representation
    n_temps = 20
    n_samples = 100
    latent_dim = 10
    
    latent_repr = LatentRepresentation(
        latent_means=np.random.randn(n_temps, n_samples, latent_dim),
        latent_stds=np.ones((n_temps, n_samples, latent_dim)) * 0.1,
        order_parameter_dim=0,
        reconstruction_quality={'mse': 0.01}
    )
    
    return VAEAnalysisResults(
        variant_id=variant_id,
        parameters={'temperature': tc},
        critical_temperature=tc,
        tc_confidence=0.95,
        exponents={'beta': beta},
        exponent_errors={'beta': beta_error},
        r_squared_values={'beta': r_squared},
        latent_representation=latent_repr,
        order_parameter_dim=0
    )


def create_mock_simulation_data(
    variant_id: str,
    tc: float = 4.5,
    first_order: bool = False
) -> SimulationData:
    """Create mock simulation data for testing."""
    
    n_temps = 20
    n_samples = 100
    lattice_size = 16
    
    temperatures = np.linspace(tc * 0.7, tc * 1.3, n_temps)
    
    # Create magnetization data
    if first_order:
        # Create discontinuous magnetization for first-order transition
        magnetizations = np.zeros((n_temps, n_samples))
        for i, T in enumerate(temperatures):
            if T < tc:
                # Ordered phase
                magnetizations[i, :] = 0.8 + np.random.randn(n_samples) * 0.05
            else:
                # Disordered phase
                magnetizations[i, :] = 0.1 + np.random.randn(n_samples) * 0.05
    else:
        # Continuous transition
        magnetizations = np.zeros((n_temps, n_samples))
        for i, T in enumerate(temperatures):
            # Smooth transition
            m_mean = max(0, (tc - T) / tc) ** 0.326  # β ≈ 0.326 for 3D Ising
            magnetizations[i, :] = m_mean + np.random.randn(n_samples) * 0.1
    
    # Create mock configurations
    configurations = np.random.choice(
        [-1, 1],
        size=(n_temps, n_samples, lattice_size, lattice_size, lattice_size)
    )
    
    # Create mock energies
    energies = np.random.randn(n_temps, n_samples) * 0.5
    
    return SimulationData(
        variant_id=variant_id,
        parameters={'temperature': tc},
        temperatures=temperatures,
        configurations=configurations,
        magnetizations=magnetizations,
        energies=energies,
        metadata={'lattice_size': lattice_size}
    )


def example_anomalous_exponents():
    """Example: Detecting anomalous critical exponents."""
    
    print("\n" + "=" * 80)
    print("Example 1: Anomalous Critical Exponents")
    print("=" * 80)
    
    # Create detector
    detector = NovelPhenomenonDetector(anomaly_threshold=3.0)
    
    # Test case 1: Normal 3D Ising exponent (should not be flagged)
    print("\nTest 1: Normal 3D Ising exponent (beta = 0.326)")
    vae_results = create_mock_vae_results(
        variant_id='test_3d_ising',
        beta=0.326,
        beta_error=0.02
    )
    
    phenomena = detector.detect_anomalous_exponents(vae_results)
    print(f"  Detected phenomena: {len(phenomena)}")
    if phenomena:
        for p in phenomena:
            print(f"    - {p.description}")
    else:
        print("    - No anomalies detected (as expected)")
    
    # Test case 2: Anomalous exponent (should be flagged)
    print("\nTest 2: Anomalous exponent (beta = 0.60, truly anomalous)")
    vae_results = create_mock_vae_results(
        variant_id='test_anomalous',
        beta=0.60,
        beta_error=0.02
    )
    
    phenomena = detector.detect_anomalous_exponents(vae_results)
    print(f"  Detected phenomena: {len(phenomena)}")
    for p in phenomena:
        print(f"    - {p.description}")
        print(f"      Confidence: {p.confidence:.2%}")
        print(f"      Evidence: {p.supporting_evidence}")
    
    # Test case 3: Slightly off but within threshold
    print("\nTest 3: Slightly off exponent (beta = 0.35, within 3-sigma)")
    vae_results = create_mock_vae_results(
        variant_id='test_slight_deviation',
        beta=0.35,
        beta_error=0.02
    )
    
    phenomena = detector.detect_anomalous_exponents(vae_results)
    print(f"  Detected phenomena: {len(phenomena)}")
    if phenomena:
        for p in phenomena:
            print(f"    - {p.description}")
    else:
        print("    - No anomalies detected (within threshold)")


def example_first_order_transition():
    """Example: Detecting first-order phase transitions."""
    
    print("\n" + "=" * 80)
    print("Example 2: First-Order Phase Transitions")
    print("=" * 80)
    
    # Create detector
    detector = NovelPhenomenonDetector(anomaly_threshold=3.0)
    
    # Test case 1: Continuous transition (should not be flagged)
    print("\nTest 1: Continuous transition")
    vae_results = create_mock_vae_results(
        variant_id='test_continuous',
        beta=0.326
    )
    sim_data = create_mock_simulation_data(
        variant_id='test_continuous',
        first_order=False
    )
    
    phenomenon = detector.detect_first_order_transition(vae_results, sim_data)
    if phenomenon:
        print(f"  Detected: {phenomenon.description}")
    else:
        print("  No first-order transition detected (as expected)")
    
    # Test case 2: First-order transition (should be flagged)
    print("\nTest 2: First-order transition")
    vae_results = create_mock_vae_results(
        variant_id='test_first_order',
        beta=0.326
    )
    sim_data = create_mock_simulation_data(
        variant_id='test_first_order',
        first_order=True
    )
    
    phenomenon = detector.detect_first_order_transition(vae_results, sim_data)
    if phenomenon:
        print(f"  Detected: {phenomenon.description}")
        print(f"  Confidence: {phenomenon.confidence:.2%}")
        print(f"  Evidence:")
        for key, value in phenomenon.supporting_evidence.items():
            if not isinstance(value, list):  # Skip large arrays
                print(f"    - {key}: {value}")
    else:
        print("  No first-order transition detected")


def example_universality_class_comparison():
    """Example: Comparing to known universality classes."""
    
    print("\n" + "=" * 80)
    print("Example 3: Universality Class Comparison")
    print("=" * 80)
    
    # Create detector
    detector = NovelPhenomenonDetector(anomaly_threshold=3.0)
    
    # Test different exponent values
    test_cases = [
        ('3D Ising', 0.326, 'ising_3d'),
        ('2D Ising', 0.125, 'ising_2d'),
        ('Mean Field', 0.50, 'mean_field'),
        ('Unknown', 0.40, None),
    ]
    
    for name, beta, expected_class in test_cases:
        print(f"\nTest: {name} (beta = {beta})")
        
        vae_results = create_mock_vae_results(
            variant_id=f'test_{name.lower().replace(" ", "_")}',
            beta=beta,
            beta_error=0.02
        )
        
        # Find closest universality class
        closest_class, confidence, deviations = detector.get_closest_universality_class(
            vae_results
        )
        
        print(f"  Closest class: {closest_class}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Deviations: {deviations}")
        
        # Compare to specific class if expected
        if expected_class:
            matches, conf, devs = detector.compare_to_universality_class(
                vae_results, expected_class
            )
            print(f"  Matches {expected_class}: {matches} (confidence: {conf:.2%})")


def example_comprehensive_detection():
    """Example: Comprehensive detection of all phenomena types."""
    
    print("\n" + "=" * 80)
    print("Example 4: Comprehensive Phenomena Detection")
    print("=" * 80)
    
    # Create detector
    detector = NovelPhenomenonDetector(anomaly_threshold=3.0)
    
    # Create a system with both anomalous exponent and first-order transition
    print("\nTest: System with multiple novel phenomena (beta = 0.60, first-order)")
    
    vae_results = create_mock_vae_results(
        variant_id='test_complex',
        beta=0.60,  # Anomalous
        beta_error=0.02
    )
    
    sim_data = create_mock_simulation_data(
        variant_id='test_complex',
        first_order=True  # First-order
    )
    
    # Detect all phenomena
    phenomena = detector.detect_all_phenomena(vae_results, sim_data)
    
    print(f"\nDetected {len(phenomena)} novel phenomena:")
    for i, p in enumerate(phenomena, 1):
        print(f"\n  {i}. Type: {p.phenomenon_type}")
        print(f"     Description: {p.description}")
        print(f"     Confidence: {p.confidence:.2%}")
        print(f"     Parameters: {p.parameters}")


def example_custom_universality_class():
    """Example: Adding custom universality classes."""
    
    print("\n" + "=" * 80)
    print("Example 5: Custom Universality Classes")
    print("=" * 80)
    
    # Define a custom universality class
    custom_class = UniversalityClass(
        name='Custom Exotic',
        exponents={'beta': 0.42, 'nu': 0.75, 'gamma': 1.5},
        exponent_errors={'beta': 0.02, 'nu': 0.03, 'gamma': 0.05},
        description='Hypothetical exotic universality class'
    )
    
    # Create detector with custom class
    detector = NovelPhenomenonDetector(
        anomaly_threshold=3.0,
        custom_universality_classes={'custom_exotic': custom_class}
    )
    
    print(f"\nRegistered universality classes:")
    for name in detector.universality_classes.keys():
        print(f"  - {name}")
    
    # Test against custom class
    print(f"\nTest: Exponent matching custom class (beta = 0.42)")
    vae_results = create_mock_vae_results(
        variant_id='test_custom',
        beta=0.42,
        beta_error=0.02
    )
    
    closest_class, confidence, deviations = detector.get_closest_universality_class(
        vae_results
    )
    
    print(f"  Closest class: {closest_class}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  Deviations: {deviations}")


def main():
    """Run all examples."""
    
    print("=" * 80)
    print("Novel Phenomena Detection Examples")
    print("=" * 80)
    print("\nThis example demonstrates various novel phenomena detection capabilities:")
    print("  1. Anomalous critical exponents")
    print("  2. First-order phase transitions")
    print("  3. Universality class comparison")
    print("  4. Comprehensive detection")
    print("  5. Custom universality classes")
    
    try:
        example_anomalous_exponents()
        example_first_order_transition()
        example_universality_class_comparison()
        example_comprehensive_detection()
        example_custom_universality_class()
        
        print("\n" + "=" * 80)
        print("All Examples Complete")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
