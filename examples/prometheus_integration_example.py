"""
Example: Prometheus Integration Layer

This example demonstrates how to use the PrometheusIntegration layer to
analyze simulation data with reproducibility guarantees and optimization features.

The integration layer provides:
- Clean interface to Prometheus VAE analysis
- Reproducibility with fixed random seeds
- Optional ensemble extraction
- Result caching for efficiency
- Performance profiling support
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research import (
    PrometheusIntegration,
    PrometheusConfig,
    ModelVariantRegistry,
    ModelVariantConfig,
    SimulationData,
)


def main():
    """Demonstrate Prometheus integration layer."""
    
    print("=" * 80)
    print("Prometheus Integration Layer Example")
    print("=" * 80)
    
    # =========================================================================
    # 1. Create Prometheus Integration with Configuration
    # =========================================================================
    
    print("\n1. Initializing Prometheus Integration")
    print("-" * 80)
    
    # Configure Prometheus integration
    config = PrometheusConfig(
        system_type='ising_2d',
        random_seed=42,  # For reproducibility
        use_ensemble=True,  # Use ensemble extraction methods
        enable_caching=True,  # Cache results for efficiency
        enable_profiling=False,  # Disable profiling for this example
        vae_params={
            'latent_dim': 10,
            'learning_rate': 0.001,
        },
        analysis_params={
            'anomaly_threshold': 3.0,
        }
    )
    
    # Create integration layer
    prometheus = PrometheusIntegration(config)
    
    print(f"[OK] Initialized Prometheus integration for {config.system_type}")
    print(f"  Random seed: {config.random_seed}")
    print(f"  Ensemble extraction: {config.use_ensemble}")
    print(f"  Caching: {config.enable_caching}")
    
    # Validate integration
    validation = prometheus.validate_integration()
    print("\nIntegration validation:")
    for key, value in validation.items():
        status = "[OK]" if value else "[--]"
        print(f"  {status} {key}: {value}")
    
    # =========================================================================
    # 2. Generate Simulation Data
    # =========================================================================
    
    print("\n2. Generating Simulation Data")
    print("-" * 80)
    
    # Create model variant registry
    registry = ModelVariantRegistry()
    
    # Register standard 2D Ising model
    variant_config = ModelVariantConfig(
        name="standard_2d_ising",
        dimensions=2,
        lattice_geometry='square',
        interaction_type='nearest_neighbor',
        interaction_params={'J': 1.0},
        theoretical_tc=2.269,
        theoretical_exponents={'beta': 0.125, 'nu': 1.0, 'gamma': 1.75}
    )
    
    variant_id = registry.register_variant(variant_config)
    print(f"[OK] Registered variant: {variant_id}")
    
    # Simulation parameters
    lattice_size = 32
    n_temperatures = 15
    n_samples = 50
    tc_estimate = 2.269
    
    temperatures = np.linspace(tc_estimate * 0.8, tc_estimate * 1.2, n_temperatures)
    
    print(f"  Lattice size: {lattice_size}x{lattice_size}")
    print(f"  Temperatures: {n_temperatures} points")
    print(f"  Samples per temperature: {n_samples}")
    
    # Run simulations
    print("\n  Running Monte Carlo simulations...")
    
    all_configurations = []
    all_magnetizations = []
    all_energies = []
    
    for i, temp in enumerate(temperatures):
        # Create simulator
        simulator = registry.create_simulator(
            variant_id,
            lattice_size=lattice_size,
            temperature=temp,
            seed=42
        )
        
        # Equilibrate
        simulator.equilibrate(1000)
        
        # Measure
        measurements = simulator.measure(n_samples, 10)
        
        all_configurations.append(measurements['configurations'])
        all_magnetizations.append(measurements['magnetizations'])
        all_energies.append(measurements['energies'])
        
        if (i + 1) % 5 == 0:
            print(f"    Progress: {i+1}/{n_temperatures} temperatures")
    
    # Create SimulationData object
    sim_data = SimulationData(
        variant_id=variant_id,
        parameters={'temperature_range': (temperatures[0], temperatures[-1])},
        temperatures=temperatures,
        configurations=np.array(all_configurations),
        magnetizations=np.array(all_magnetizations),
        energies=np.array(all_energies),
        metadata={
            'lattice_size': lattice_size,
            'n_samples': n_samples,
            'seed': 42,
        }
    )
    
    print(f"\n[OK] Simulation complete")
    print(f"  Data shape: {sim_data.configurations.shape}")
    
    # =========================================================================
    # 3. Analyze with Prometheus Integration
    # =========================================================================
    
    print("\n3. Analyzing with Prometheus Integration")
    print("-" * 80)
    
    # Ensure reproducibility
    prometheus.ensure_reproducibility()
    
    # Analyze simulation data
    print("\n  Running Prometheus VAE analysis...")
    results = prometheus.analyze_simulation_data(
        sim_data,
        auto_detect_tc=True,
        compare_with_raw_mag=False
    )
    
    print(f"\n[OK] Analysis complete")
    print(f"\nResults:")
    print(f"  Critical Temperature:")
    print(f"    Tc = {results.critical_temperature:.4f}")
    print(f"    Confidence = {results.tc_confidence:.2%}")
    print(f"    Theoretical = {variant_config.theoretical_tc:.4f}")
    print(f"    Error = {abs(results.critical_temperature - variant_config.theoretical_tc):.4f}")
    
    print(f"\n  Critical Exponents:")
    for exp_name, exp_value in results.exponents.items():
        error = results.exponent_errors.get(exp_name, 0.0)
        r_squared = results.r_squared_values.get(exp_name, 0.0)
        print(f"    {exp_name} = {exp_value:.4f} ± {error:.4f} (R² = {r_squared:.4f})")
        
        # Compare with theoretical if available
        if exp_name == 'beta':
            theoretical = variant_config.theoretical_exponents['beta']
            accuracy = (1 - abs(exp_value - theoretical) / theoretical) * 100
            print(f"      Theoretical = {theoretical:.4f}")
            print(f"      Accuracy = {accuracy:.1f}%")
    
    print(f"\n  Order Parameter:")
    print(f"    Dimension = {results.order_parameter_dim}")
    
    # =========================================================================
    # 4. Demonstrate Caching
    # =========================================================================
    
    print("\n4. Demonstrating Result Caching")
    print("-" * 80)
    
    # Check cache stats before
    cache_stats = prometheus.get_cache_stats()
    print(f"  Cache stats before: {cache_stats}")
    
    # Analyze same data again (should use cache)
    print("\n  Analyzing same data again (should use cache)...")
    import time
    start_time = time.time()
    results_cached = prometheus.analyze_simulation_data(sim_data)
    cache_time = time.time() - start_time
    
    print(f"  [OK] Analysis complete in {cache_time:.4f} seconds")
    print(f"  Results match: {results_cached.critical_temperature == results.critical_temperature}")
    
    # Check cache stats after
    cache_stats = prometheus.get_cache_stats()
    print(f"  Cache stats after: {cache_stats}")
    
    # =========================================================================
    # 5. Demonstrate Reproducibility
    # =========================================================================
    
    print("\n5. Demonstrating Reproducibility")
    print("-" * 80)
    
    # Clear cache to force reanalysis
    prometheus.clear_cache()
    
    # Get reproducibility info
    repro_info = prometheus.get_reproducibility_info()
    print(f"  Reproducibility info:")
    for key, value in repro_info.items():
        print(f"    {key}: {value}")
    
    # Analyze with same seed
    prometheus.ensure_reproducibility(seed=42)
    results_1 = prometheus.analyze_simulation_data(sim_data)
    
    # Clear cache and analyze again with same seed
    prometheus.clear_cache()
    prometheus.ensure_reproducibility(seed=42)
    results_2 = prometheus.analyze_simulation_data(sim_data)
    
    print(f"\n  Results with same seed:")
    print(f"    Tc (run 1) = {results_1.critical_temperature:.6f}")
    print(f"    Tc (run 2) = {results_2.critical_temperature:.6f}")
    print(f"    Match: {abs(results_1.critical_temperature - results_2.critical_temperature) < 1e-6}")
    
    # =========================================================================
    # 6. Compare Different Configurations
    # =========================================================================
    
    print("\n6. Comparing Different Configurations")
    print("-" * 80)
    
    # Create integration without ensemble
    config_no_ensemble = PrometheusConfig(
        system_type='ising_2d',
        random_seed=42,
        use_ensemble=False,  # Disable ensemble
        enable_caching=False,
    )
    
    prometheus_no_ensemble = PrometheusIntegration(config_no_ensemble)
    
    print("  Analyzing without ensemble extraction...")
    results_no_ensemble = prometheus_no_ensemble.analyze_simulation_data(sim_data)
    
    print(f"\n  Comparison:")
    print(f"    With ensemble:")
    print(f"      β = {results.exponents['beta']:.4f} ± {results.exponent_errors['beta']:.4f}")
    if 'beta_ensemble' in results.exponents:
        print(f"      β (ensemble) = {results.exponents['beta_ensemble']:.4f} ± {results.exponent_errors['beta_ensemble']:.4f}")
    
    print(f"\n    Without ensemble:")
    print(f"      β = {results_no_ensemble.exponents['beta']:.4f} ± {results_no_ensemble.exponent_errors['beta']:.4f}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    print("\nPrometheus Integration Features Demonstrated:")
    print("  [OK] Clean interface to Prometheus VAE analysis")
    print("  [OK] Reproducibility with fixed random seeds")
    print("  [OK] Optional ensemble extraction for robust estimates")
    print("  [OK] Result caching for efficiency")
    print("  [OK] Configurable analysis parameters")
    print("  [OK] Integration validation")
    
    print("\nKey Benefits:")
    print("  - Maintains Prometheus accuracy guarantees (>=70%)")
    print("  - Ensures reproducible results across runs")
    print("  - Improves efficiency with caching")
    print("  - Provides flexible configuration options")
    print("  - Simplifies integration with Research Explorer")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
