"""
Example demonstrating the Unified Validation Pipeline.

This example shows how to use the unified validation pipeline to validate
a potential physics discovery by applying all 10 validation patterns in sequence.
"""

import numpy as np
from pathlib import Path
from typing import Dict
import sys
sys.path.insert(0, '.')

from src.research.unified_validation_pipeline import UnifiedValidationPipeline
from src.research.base_types import VAEAnalysisResults, SimulationData


def create_example_vae_results(variant_id: str, is_novel: bool = True) -> VAEAnalysisResults:
    """Create example VAE results for demonstration.
    
    Args:
        variant_id: ID for the variant
        is_novel: Whether to simulate novel physics or known physics
        
    Returns:
        VAEAnalysisResults
    """
    if is_novel:
        # Simulate novel physics (deviates from 2D Ising)
        exponents = {
            'beta': 0.200,  # 2D Ising is 0.125
            'nu': 0.850,    # 2D Ising is 1.0
            'gamma': 1.500  # 2D Ising is 1.75
        }
        exponent_errors = {
            'beta': 0.015,
            'nu': 0.040,
            'gamma': 0.080
        }
        r_squared_values = {
            'beta': 0.92,
            'nu': 0.89,
            'gamma': 0.91
        }
    else:
        # Simulate known 2D Ising physics
        exponents = {
            'beta': 0.125,
            'nu': 1.0,
            'gamma': 1.75
        }
        exponent_errors = {
            'beta': 0.010,
            'nu': 0.030,
            'gamma': 0.060
        }
        r_squared_values = {
            'beta': 0.95,
            'nu': 0.94,
            'gamma': 0.96
        }
    
    # Create latent representation
    from src.research.base_types import LatentRepresentation
    latent_rep = LatentRepresentation(
        latent_means=np.random.randn(100, 8),
        latent_stds=np.random.rand(100, 8) * 0.1,
        order_parameter_dim=0,
        reconstruction_quality={'mse': 0.02, 'r2': 0.88}
    )
    
    return VAEAnalysisResults(
        variant_id=variant_id,
        parameters={'temperature_range': (2.0, 2.5), 'lattice_size': 64},
        exponents=exponents,
        exponent_errors=exponent_errors,
        critical_temperature=2.269,
        tc_confidence=0.92,
        r_squared_values=r_squared_values,
        latent_representation=latent_rep,
        order_parameter_dim=0
    )


def create_bootstrap_samples(
    exponents: Dict[str, float],
    errors: Dict[str, float],
    n_samples: int = 1000
) -> Dict[str, np.ndarray]:
    """Create bootstrap samples for exponents.
    
    Args:
        exponents: Central values
        errors: Standard errors
        n_samples: Number of bootstrap samples
        
    Returns:
        Dictionary of bootstrap samples
    """
    bootstrap_samples = {}
    for exp_name, exp_value in exponents.items():
        error = errors.get(exp_name, 0.05)
        samples = np.random.normal(exp_value, error, n_samples)
        bootstrap_samples[exp_name] = samples
    
    return bootstrap_samples


def example_novel_physics_validation():
    """Example: Validate a potential novel physics discovery."""
    print("=" * 70)
    print("Example 1: Novel Physics Discovery Validation")
    print("=" * 70)
    print()
    
    # Create unified validation pipeline
    pipeline = UnifiedValidationPipeline(
        validation_threshold=0.90,
        anomaly_threshold=3.0,
        n_bootstrap=1000,
        alpha=0.05
    )
    
    # Create example VAE results (novel physics)
    vae_results = create_example_vae_results(
        variant_id="long_range_ising_alpha_2.2",
        is_novel=True
    )
    
    # Create bootstrap samples
    bootstrap_samples = create_bootstrap_samples(
        vae_results.exponents,
        vae_results.exponent_errors,
        n_samples=1000
    )
    
    # Predicted exponents (from theory or reference)
    # Using 2D Ising as reference
    predicted_exponents = {
        'beta': 0.125,
        'nu': 1.0,
        'gamma': 1.75
    }
    
    # Run validation
    print("Running unified validation pipeline...")
    print()
    
    report = pipeline.validate_discovery(
        vae_results=vae_results,
        simulation_data=None,
        predicted_exponents=predicted_exponents,
        variant_results=None,
        bootstrap_samples=bootstrap_samples,
        dimensions=2
    )
    
    # Print results
    print(report.summary)
    print()
    print(f"Recommendation: {report.recommendation}")
    print()
    print(f"Publication Ready: {report.publication_ready}")
    print()
    
    # Save report
    output_dir = Path("results/unified_validation_demo")
    filepath = pipeline.save_report(report, output_dir)
    print(f"Report saved to: {filepath}")
    print()


def example_known_physics_validation():
    """Example: Validate known physics (should not claim novel)."""
    print("=" * 70)
    print("Example 2: Known Physics Validation (Negative Control)")
    print("=" * 70)
    print()
    
    # Create unified validation pipeline
    pipeline = UnifiedValidationPipeline(
        validation_threshold=0.90,
        anomaly_threshold=3.0,
        n_bootstrap=1000,
        alpha=0.05
    )
    
    # Create example VAE results (known 2D Ising)
    vae_results = create_example_vae_results(
        variant_id="standard_2d_ising",
        is_novel=False
    )
    
    # Create bootstrap samples
    bootstrap_samples = create_bootstrap_samples(
        vae_results.exponents,
        vae_results.exponent_errors,
        n_samples=1000
    )
    
    # Predicted exponents (2D Ising)
    predicted_exponents = {
        'beta': 0.125,
        'nu': 1.0,
        'gamma': 1.75
    }
    
    # Run validation
    print("Running unified validation pipeline...")
    print()
    
    report = pipeline.validate_discovery(
        vae_results=vae_results,
        simulation_data=None,
        predicted_exponents=predicted_exponents,
        variant_results=None,
        bootstrap_samples=bootstrap_samples,
        dimensions=2
    )
    
    # Print results
    print(report.summary)
    print()
    print(f"Recommendation: {report.recommendation}")
    print()
    print(f"Publication Ready: {report.publication_ready}")
    print()
    
    # Save report
    output_dir = Path("results/unified_validation_demo")
    filepath = pipeline.save_report(report, output_dir)
    print(f"Report saved to: {filepath}")
    print()


def example_methodological_issue():
    """Example: Detect methodological issues (poor data quality)."""
    print("=" * 70)
    print("Example 3: Methodological Issue Detection")
    print("=" * 70)
    print()
    
    # Create unified validation pipeline
    pipeline = UnifiedValidationPipeline(
        validation_threshold=0.90,
        anomaly_threshold=3.0,
        n_bootstrap=1000,
        alpha=0.05
    )
    
    # Create example VAE results with poor data quality
    from src.research.base_types import LatentRepresentation
    latent_rep = LatentRepresentation(
        latent_means=np.random.randn(100, 8),
        latent_stds=np.random.rand(100, 8) * 0.2,
        order_parameter_dim=0,
        reconstruction_quality={'mse': 0.08, 'r2': 0.70}
    )
    
    vae_results = VAEAnalysisResults(
        variant_id="poor_quality_data",
        parameters={'temperature_range': (2.0, 2.5), 'lattice_size': 32},
        exponents={
            'beta': 0.200,
            'nu': 0.850,
            'gamma': 1.500
        },
        exponent_errors={
            'beta': 0.080,  # Large errors
            'nu': 0.150,
            'gamma': 0.250
        },
        critical_temperature=2.269,
        tc_confidence=0.55,  # Low Tc confidence
        r_squared_values={
            'beta': 0.62,  # Poor R²
            'nu': 0.58,
            'gamma': 0.65
        },
        latent_representation=latent_rep,
        order_parameter_dim=0
    )
    
    # Create bootstrap samples
    bootstrap_samples = create_bootstrap_samples(
        vae_results.exponents,
        vae_results.exponent_errors,
        n_samples=1000
    )
    
    # Predicted exponents
    predicted_exponents = {
        'beta': 0.125,
        'nu': 1.0,
        'gamma': 1.75
    }
    
    # Run validation
    print("Running unified validation pipeline...")
    print()
    
    report = pipeline.validate_discovery(
        vae_results=vae_results,
        simulation_data=None,
        predicted_exponents=predicted_exponents,
        variant_results=None,
        bootstrap_samples=bootstrap_samples,
        dimensions=2
    )
    
    # Print results
    print(report.summary)
    print()
    print(f"Recommendation: {report.recommendation}")
    print()
    
    # Check anomaly classification
    anomaly_results = report.pattern_results['anomaly_classification']
    print(f"Methodological Issues Detected: {anomaly_results['n_methodological']}")
    print(f"Physics-Novel Anomalies: {anomaly_results['n_physics_novel']}")
    print()
    
    # Save report
    output_dir = Path("results/unified_validation_demo")
    filepath = pipeline.save_report(report, output_dir)
    print(f"Report saved to: {filepath}")
    print()


if __name__ == "__main__":
    # Run all examples
    example_novel_physics_validation()
    print("\n" * 2)
    
    example_known_physics_validation()
    print("\n" * 2)
    
    example_methodological_issue()
