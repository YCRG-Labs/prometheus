"""
Advanced Usage Example for Prometheus

This script demonstrates advanced features including:
- Ensemble methods for robust extraction
- Comprehensive validation with bootstrap CI
- Performance optimization with caching
- Multi-method comparison

Usage:
    python examples/advanced_usage_example.py
"""

import numpy as np
from pathlib import Path

from src.utils.reproducibility import set_random_seed
from src.data.ising_simulator import IsingSimulator
from src.analysis.ensemble_extractor import EnsembleExponentExtractor
from src.analysis.validation_framework import ValidationFramework
from src.optimization.performance_optimizer import PerformanceProfiler, ResultCache


def main():
    """Run advanced critical exponent extraction example."""
    
    print("=" * 70)
    print("Prometheus Advanced Usage Example")
    print("=" * 70)
    print()
    
    # Setup
    SEED = 42
    set_random_seed(SEED)
    
    # Initialize profiler and cache
    profiler = PerformanceProfiler()
    cache = ResultCache(max_size_mb=100)
    
    # Step 1: Generate data with profiling
    print("Step 1: Generating Monte Carlo data (with profiling)...")
    
    with profiler.profile_block("data_generation"):
        simulator = IsingSimulator(lattice_size=(32, 32, 32), temperature=4.5)
        temperatures = np.linspace(3.5, 5.5, 30)
        
        all_magnetizations = []
        all_temps = []
        
        for temp in temperatures:
            simulator.temperature = temp
            samples = simulator.generate_samples(n_samples=50)
            magnetizations = np.abs(np.mean(samples, axis=(1, 2, 3)))
            
            all_magnetizations.extend(magnetizations)
            all_temps.extend([temp] * len(magnetizations))
    
    print(f"✓ Generated {len(all_magnetizations)} samples")
    print()
    
    # Step 2: Prepare latent representation
    print("Step 2: Preparing latent representation...")
    
    latent_data = np.column_stack([
        all_magnetizations,
        all_temps
    ])
    temperatures_array = np.array(all_temps)
    
    print(f"✓ Latent shape: {latent_data.shape}")
    print()
    
    # Step 3: Ensemble extraction
    print("Step 3: Extracting exponents with ensemble methods...")
    print()
    
    with profiler.profile_block("ensemble_extraction"):
        extractor = EnsembleExponentExtractor()
        
        # Detect Tc first
        from src.analysis.improved_critical_exponent_analyzer import ImprovedCriticalExponentAnalyzer
        analyzer = ImprovedCriticalExponentAnalyzer()
        tc_result = analyzer.detect_tc_stable(latent_data, temperatures_array)
        tc = tc_result['tc']
        
        print(f"  Detected Tc = {tc:.4f}")
        print()
        
        # Extract beta with ensemble
        print("  Extracting β with ensemble methods...")
        beta_results = extractor.extract_beta_ensemble(
            latent_data=latent_data,
            temperatures=temperatures_array,
            tc=tc
        )
        
        print(f"  ✓ Ensemble β = {beta_results['ensemble_beta']:.4f} ± {beta_results['ensemble_error']:.4f}")
        print(f"    Confidence: {beta_results['ensemble_confidence']:.1f}%")
        print(f"    Method agreement: {beta_results['method_agreement']:.1f}%")
        print()
        
        # Show individual methods
        print("  Individual method results:")
        for method_name, result in beta_results['method_results'].items():
            print(f"    {method_name:20s}: β = {result['value']:.4f} ± {result['error']:.4f} (R² = {result['r_squared']:.4f})")
        print()
    
    # Step 4: Comprehensive validation
    print("Step 4: Running comprehensive validation...")
    print()
    
    with profiler.profile_block("validation"):
        validator = ValidationFramework()
        
        # Bootstrap confidence intervals
        print("  Computing bootstrap confidence intervals (1000 samples)...")
        bootstrap_result = validator.bootstrap_confidence_interval(
            data=latent_data[:, 0],  # Order parameter
            temperatures=temperatures_array,
            tc=tc,
            n_bootstrap=1000,
            n_workers=4
        )
        
        print(f"  ✓ Bootstrap CI: [{bootstrap_result['ci_lower']:.4f}, {bootstrap_result['ci_upper']:.4f}]")
        print(f"    Bias: {bootstrap_result['bias']:.6f}")
        print()
        
        # Statistical tests
        print("  Running statistical tests...")
        
        # Prepare data for fitting
        temps_below_tc = temperatures_array[temperatures_array < tc]
        order_param = latent_data[:, 0][temperatures_array < tc]
        reduced_temp = (tc - temps_below_tc) / tc
        
        # Remove zeros
        mask = reduced_temp > 0
        reduced_temp = reduced_temp[mask]
        order_param = order_param[mask]
        
        if len(reduced_temp) > 10:
            # Fit power law
            log_t = np.log(reduced_temp)
            log_m = np.log(order_param + 1e-10)
            
            # Linear fit
            coeffs = np.polyfit(log_t, log_m, 1)
            fitted = np.polyval(coeffs, log_t)
            residuals = log_m - fitted
            
            # Run tests
            stat_tests = validator.run_statistical_tests(
                residuals=residuals,
                fitted_values=fitted,
                n_params=2
            )
            
            print(f"  ✓ F-test: p = {stat_tests['f_test']['p_value']:.4f} ({'significant' if stat_tests['f_test']['significant'] else 'not significant'})")
            print(f"  ✓ Shapiro-Wilk: p = {stat_tests['shapiro_wilk']['p_value']:.4f} ({'normal' if stat_tests['shapiro_wilk']['normal'] else 'not normal'})")
            print(f"  ✓ Durbin-Watson: {stat_tests['durbin_watson']['statistic']:.4f} ({'no autocorr' if stat_tests['durbin_watson']['no_autocorrelation'] else 'autocorr'})")
            print(f"  ✓ Breusch-Pagan: p = {stat_tests['breusch_pagan']['p_value']:.4f} ({'homoscedastic' if stat_tests['breusch_pagan']['homoscedastic'] else 'heteroscedastic'})")
            print()
    
    # Step 5: Performance report
    print("Step 5: Performance Report")
    print("=" * 70)
    profiler.print_report()
    print()
    
    # Step 6: Summary
    print("=" * 70)
    print("Advanced Analysis Summary")
    print("=" * 70)
    print()
    print(f"Ensemble Results:")
    print(f"  β = {beta_results['ensemble_beta']:.4f} ± {beta_results['ensemble_error']:.4f}")
    print(f"  Confidence: {beta_results['ensemble_confidence']:.1f}%")
    print(f"  Method Agreement: {beta_results['method_agreement']:.1f}%")
    print()
    print(f"Validation:")
    print(f"  Bootstrap CI: [{bootstrap_result['ci_lower']:.4f}, {bootstrap_result['ci_upper']:.4f}]")
    if len(reduced_temp) > 10:
        tests_passed = sum([
            stat_tests['f_test']['significant'],
            stat_tests['shapiro_wilk']['normal'],
            stat_tests['durbin_watson']['no_autocorrelation'],
            stat_tests['breusch_pagan']['homoscedastic']
        ])
        print(f"  Statistical Tests: {tests_passed}/4 passed")
    print()
    print("=" * 70)
    print("Advanced example complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
