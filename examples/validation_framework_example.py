"""
Example: Validation Framework for Hypothesis Testing

This example demonstrates how to use the ValidationFramework to rigorously
validate research hypotheses about phase transitions using statistical methods
including bootstrap confidence intervals, hypothesis testing, and effect sizes.
"""

import numpy as np
from pathlib import Path

from src.research import (
    ModelVariantRegistry,
    ModelVariantConfig,
    VAEAnalysisResults,
    SimulationData,
    ValidationFramework,
)
from src.research.base_types import LatentRepresentation
from src.utils.logging_utils import get_logger


def create_mock_simulation_data(
    variant_id: str,
    tc: float,
    n_temps: int = 20,
    n_samples: int = 100,
    lattice_size: int = 32,
    first_order: bool = False
) -> SimulationData:
    """Create mock simulation data for demonstration.
    
    Args:
        variant_id: ID of the variant
        tc: Critical temperature
        n_temps: Number of temperature points
        n_samples: Number of samples per temperature
        lattice_size: Linear lattice size
        first_order: Whether to simulate first-order transition
        
    Returns:
        SimulationData object
    """
    temperatures = np.linspace(tc * 0.7, tc * 1.3, n_temps)
    
    # Generate magnetizations with phase transition
    magnetizations = []
    for T in temperatures:
        if first_order:
            # First-order: sharp discontinuity at Tc
            if T < tc:
                mag = np.random.normal(0.8, 0.05, n_samples)
            else:
                mag = np.random.normal(0.1, 0.05, n_samples)
        else:
            # Continuous: smooth transition
            # Use approximate power law: m ~ (Tc - T)^beta
            if T < tc:
                mag = np.random.normal((tc - T)**0.125, 0.05, n_samples)
            else:
                mag = np.random.normal(0.0, 0.05, n_samples)
        
        magnetizations.append(mag)
    
    magnetizations = np.array(magnetizations)
    energies = np.random.randn(n_temps, n_samples)
    configurations = np.random.choice([-1, 1], size=(n_temps, n_samples, lattice_size, lattice_size))
    
    return SimulationData(
        variant_id=variant_id,
        parameters={'temperature': tc},
        temperatures=temperatures,
        configurations=configurations,
        magnetizations=magnetizations,
        energies=energies,
        metadata={'lattice_size': lattice_size, 'n_samples': n_samples}
    )


def create_mock_vae_results(
    variant_id: str,
    tc: float,
    beta: float,
    beta_error: float = 0.01
) -> VAEAnalysisResults:
    """Create mock VAE results for demonstration."""
    latent_repr = LatentRepresentation(
        latent_means=np.random.randn(20, 100, 10),
        latent_stds=np.ones((20, 100, 10)) * 0.1,
        order_parameter_dim=0,
        reconstruction_quality={'mse': 0.01}
    )
    
    return VAEAnalysisResults(
        variant_id=variant_id,
        parameters={'temperature': tc},
        critical_temperature=tc,
        tc_confidence=0.95,
        exponents={'beta': beta, 'nu': 1.0, 'gamma': 1.75},
        exponent_errors={'beta': beta_error, 'nu': 0.05, 'gamma': 0.1},
        r_squared_values={'beta': 0.95, 'nu': 0.92, 'gamma': 0.90},
        latent_representation=latent_repr,
        order_parameter_dim=0
    )


def main():
    """Run validation framework example."""
    logger = get_logger('validation_framework_example')
    logger.info("=" * 80)
    logger.info("Validation Framework Example")
    logger.info("=" * 80)
    
    # Initialize validation framework
    validator = ValidationFramework(n_bootstrap=1000, alpha=0.05)
    
    # Example 1: Validate single exponent prediction
    logger.info("\n" + "=" * 80)
    logger.info("Example 1: Single Exponent Validation")
    logger.info("=" * 80)
    
    logger.info("\nScenario: Testing if measured β matches 2D Ising prediction")
    result1 = validator.validate_exponent_prediction(
        hypothesis_id="hyp_001_beta",
        predicted=0.125,  # 2D Ising theoretical value
        measured=0.127,   # Slightly different measured value
        measured_error=0.01
    )
    
    logger.info(f"\nResult: {result1.message}")
    logger.info(f"Validated: {result1.validated}")
    logger.info(f"Confidence: {result1.confidence:.2%}")
    logger.info(f"P-value: {result1.p_values['exponent']:.4f}")
    logger.info(f"Cohen's d: {result1.effect_sizes['cohens_d']:.3f}")
    logger.info(f"95% CI: {result1.bootstrap_intervals['exponent']}")
    
    # Example 2: Validate universality class membership
    logger.info("\n" + "=" * 80)
    logger.info("Example 2: Universality Class Validation")
    logger.info("=" * 80)
    
    # Create mock results for 2D Ising
    vae_results_2d = create_mock_vae_results(
        variant_id='ising_2d',
        tc=2.269,
        beta=0.127,  # Close to theoretical 0.125
        beta_error=0.01
    )
    
    logger.info("\nScenario: Testing if system belongs to 2D Ising universality class")
    result2 = validator.validate_universality_class(
        hypothesis_id="hyp_002_universality",
        measured_exponents=vae_results_2d.exponents,
        measured_errors=vae_results_2d.exponent_errors,
        class_name='ising_2d',
        apply_correction=True
    )
    
    logger.info(f"\nResult: {result2.message}")
    logger.info(f"Validated: {result2.validated}")
    logger.info(f"Confidence: {result2.confidence:.2%}")
    logger.info(f"P-values: {result2.p_values}")
    logger.info(f"Effect sizes: {result2.effect_sizes}")
    
    # Example 3: Validate phase transition order
    logger.info("\n" + "=" * 80)
    logger.info("Example 3: Phase Transition Order Validation")
    logger.info("=" * 80)
    
    # Test continuous (second-order) transition
    logger.info("\nScenario 3a: Validating continuous (second-order) transition")
    sim_data_continuous = create_mock_simulation_data(
        variant_id='ising_2d',
        tc=2.269,
        first_order=False
    )
    
    result3a = validator.validate_phase_transition_order(
        hypothesis_id="hyp_003a_order",
        simulation_data=sim_data_continuous,
        predicted_order=2,  # Predict continuous
        critical_temperature=2.269
    )
    
    logger.info(f"\nResult: {result3a.message}")
    logger.info(f"Validated: {result3a.validated}")
    logger.info(f"Confidence: {result3a.confidence:.2%}")
    logger.info(f"Jump ratio: {result3a.effect_sizes['jump_ratio']:.2f}")
    logger.info(f"KS test p-value: {result3a.p_values['ks_test']:.4f}")
    
    # Test first-order transition
    logger.info("\nScenario 3b: Validating first-order transition")
    sim_data_first_order = create_mock_simulation_data(
        variant_id='custom_model',
        tc=3.0,
        first_order=True
    )
    
    result3b = validator.validate_phase_transition_order(
        hypothesis_id="hyp_003b_order",
        simulation_data=sim_data_first_order,
        predicted_order=1,  # Predict first-order
        critical_temperature=3.0
    )
    
    logger.info(f"\nResult: {result3b.message}")
    logger.info(f"Validated: {result3b.validated}")
    logger.info(f"Confidence: {result3b.confidence:.2%}")
    logger.info(f"Jump ratio: {result3b.effect_sizes['jump_ratio']:.2f}")
    logger.info(f"KS test p-value: {result3b.p_values['ks_test']:.4f}")
    
    # Example 4: Comprehensive hypothesis validation
    logger.info("\n" + "=" * 80)
    logger.info("Example 4: Comprehensive Hypothesis Validation")
    logger.info("=" * 80)
    
    logger.info("\nScenario: Complete validation of 2D Ising hypothesis")
    logger.info("  - Predicted exponents: β=0.125, ν=1.0, γ=1.75")
    logger.info("  - Predicted universality class: 2D Ising")
    logger.info("  - Predicted transition order: continuous (2nd order)")
    
    comprehensive_results = validator.validate_hypothesis_comprehensive(
        hypothesis_id="hyp_004_comprehensive",
        vae_results=vae_results_2d,
        simulation_data=sim_data_continuous,
        predicted_exponents={'beta': 0.125, 'nu': 1.0, 'gamma': 1.75},
        predicted_errors={'beta': 0.01, 'nu': 0.05, 'gamma': 0.1},
        universality_class='ising_2d',
        predicted_order=2
    )
    
    logger.info(f"\nComprehensive validation complete:")
    logger.info(f"  Total tests: {len(comprehensive_results)}")
    logger.info(f"  Passed: {sum(r.validated for r in comprehensive_results.values())}")
    logger.info(f"  Failed: {sum(not r.validated for r in comprehensive_results.values())}")
    
    for test_name, result in comprehensive_results.items():
        status = "✓" if result.validated else "✗"
        logger.info(f"  {status} {test_name}: confidence={result.confidence:.2%}")
    
    # Generate validation report
    logger.info("\n" + "=" * 80)
    logger.info("Validation Report")
    logger.info("=" * 80)
    
    report = validator.generate_validation_report(comprehensive_results)
    print(report)
    
    # Example 5: Testing hypothesis refutation
    logger.info("\n" + "=" * 80)
    logger.info("Example 5: Hypothesis Refutation")
    logger.info("=" * 80)
    
    logger.info("\nScenario: Testing wrong universality class (3D Ising for 2D system)")
    result5 = validator.validate_universality_class(
        hypothesis_id="hyp_005_wrong_class",
        measured_exponents=vae_results_2d.exponents,
        measured_errors=vae_results_2d.exponent_errors,
        class_name='ising_3d',  # Wrong class!
        apply_correction=True
    )
    
    logger.info(f"\nResult: {result5.message}")
    logger.info(f"Validated: {result5.validated}")
    logger.info(f"Confidence: {result5.confidence:.2%}")
    
    # Example 6: Effect of measurement uncertainty
    logger.info("\n" + "=" * 80)
    logger.info("Example 6: Effect of Measurement Uncertainty")
    logger.info("=" * 80)
    
    logger.info("\nComparing validation with different measurement errors:")
    
    for error_level in [0.005, 0.01, 0.02, 0.05]:
        result = validator.validate_exponent_prediction(
            hypothesis_id=f"hyp_006_error_{error_level}",
            predicted=0.125,
            measured=0.135,  # 0.01 away from prediction
            measured_error=error_level
        )
        logger.info(
            f"  Error={error_level:.3f}: validated={result.validated}, "
            f"confidence={result.confidence:.2%}, p={result.p_values['exponent']:.4f}"
        )
    
    logger.info("\n" + "=" * 80)
    logger.info("Validation Framework Example Complete")
    logger.info("=" * 80)
    logger.info("\nKey Takeaways:")
    logger.info("  1. Bootstrap CIs provide robust uncertainty quantification")
    logger.info("  2. Multiple comparison correction prevents false positives")
    logger.info("  3. Effect sizes complement p-values for practical significance")
    logger.info("  4. Comprehensive validation tests multiple aspects of hypotheses")
    logger.info("  5. Measurement uncertainty critically affects validation outcomes")


if __name__ == '__main__':
    main()
