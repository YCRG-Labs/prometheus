#!/usr/bin/env python3
"""
Recreate Quantum Discovery Campaign Results

This script recreates all results from the 5-month quantum discovery campaign,
including the Disordered Transverse Field Ising Model (DTFIM) analysis.

Steps:
1. Run exact diagonalization for DTFIM
2. Perform finite-size scaling analysis
3. Extract critical exponents
4. Analyze entanglement entropy
5. Cross-validate results
6. Generate publication figures

Usage:
    python scripts/recreate_quantum_discovery_results.py
    python scripts/recreate_quantum_discovery_results.py --quick  # Fast validation
    python scripts/recreate_quantum_discovery_results.py --full   # Full analysis
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_quantum_discovery_pipeline(mode: str = 'quick', output_dir: str = None):
    """Run the quantum discovery campaign pipeline."""
    
    print("=" * 70)
    print("RECREATING QUANTUM DISCOVERY CAMPAIGN RESULTS")
    print("Disordered Transverse Field Ising Model (DTFIM)")
    print("=" * 70)
    print()
    
    # Set parameters based on mode
    if mode == 'quick':
        system_sizes = [8, 10, 12]
        n_realizations = 20
        n_h_points = 15
        print("Running in QUICK mode (validation only)")
    elif mode == 'standard':
        system_sizes = [8, 12, 16]
        n_realizations = 100
        n_h_points = 30
        print("Running in STANDARD mode")
    else:  # full
        system_sizes = [8, 12, 16, 20, 24]
        n_realizations = 500
        n_h_points = 50
        print("Running in FULL mode (publication-quality)")
    
    if output_dir is None:
        output_dir = f"results/quantum_discovery_{datetime.now():%Y%m%d_%H%M%S}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_path}")
    print(f"System sizes: {system_sizes}")
    print(f"Disorder realizations: {n_realizations}")
    print(f"h points: {n_h_points}")
    print()
    
    # Import modules
    try:
        import numpy as np
        from src.quantum.disordered_tfim import DisorderedTFIM, DTFIMParams
        from src.quantum.exact_diagonalization import ExactDiagonalizationSolver
        from src.quantum.entanglement import EntanglementCalculator
        from src.research.finite_size_scaling import FiniteSizeScalingAnalyzer
        from src.research.critical_exponent_extractor import CriticalExponentExtractor
        from src.research.entanglement_analyzer import EntanglementAnalyzer
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure all quantum modules are installed.")
        sys.exit(1)
    
    results = {
        'mode': mode,
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'system_sizes': system_sizes,
            'n_realizations': n_realizations,
            'n_h_points': n_h_points
        }
    }
    
    # Step 1: Verify Clean TFIM
    print("-" * 50)
    print("Step 1: Verifying clean TFIM baseline")
    print("-" * 50)
    
    L_test = 8
    params = DTFIMParams(
        L=L_test,
        h_mean=1.0,
        h_disorder=0.0,
        J_mean=1.0,
        J_disorder=0.0,
        periodic=True
    )
    dtfim = DisorderedTFIM(params)
    
    verification = dtfim.verify_clean_limit(h_test=1.0)
    print(f"  Clean limit verification: {'PASSED' if verification['energy_matches'] else 'FAILED'}")
    print(f"  Energy difference: {verification['difference']:.2e}")
    
    results['clean_limit_verification'] = verification
    
    # Step 2: Finite-Size Scaling
    print()
    print("-" * 50)
    print("Step 2: Finite-size scaling analysis")
    print("-" * 50)
    
    W = 0.5  # Disorder strength
    h_range = (0.5, 1.5)
    
    fss_analyzer = FiniteSizeScalingAnalyzer(
        system_sizes=system_sizes,
        n_disorder_realizations=n_realizations,
        random_seed=42
    )
    
    print(f"  Generating FSS data for W = {W}...")
    data_points = fss_analyzer.generate_fss_data(
        h_range=h_range,
        W=W,
        n_h_points=n_h_points,
        parallel=True
    )
    
    print(f"  Extracting critical points...")
    critical_points = fss_analyzer.extract_critical_points(data_points)
    
    for cp in critical_points:
        print(f"    L = {cp.L}: hc = {cp.hc:.4f} ± {cp.hc_error:.4f}")
    
    # Scaling collapse
    print(f"  Performing scaling collapse...")
    collapse_result = fss_analyzer.perform_scaling_collapse(data_points, critical_points)
    
    print(f"  hc(∞) = {collapse_result.hc_inf:.4f} ± {collapse_result.hc_inf_error:.4f}")
    print(f"  Collapse quality: {collapse_result.collapse_quality:.3f}")
    
    results['fss'] = {
        'critical_points': [
            {'L': cp.L, 'hc': cp.hc, 'hc_error': cp.hc_error}
            for cp in critical_points
        ],
        'hc_inf': collapse_result.hc_inf,
        'hc_inf_error': collapse_result.hc_inf_error,
        'collapse_quality': collapse_result.collapse_quality
    }
    
    # Step 3: Critical Exponents
    print()
    print("-" * 50)
    print("Step 3: Extracting critical exponents")
    print("-" * 50)
    
    extractor = CriticalExponentExtractor(system_sizes=system_sizes)
    exponents = extractor.extract_all_exponents(
        data_points,
        hc=collapse_result.hc_inf,
        hc_error=collapse_result.hc_inf_error
    )
    
    print(f"  ν = {exponents.nu.value:.3f} ± {exponents.nu.error:.3f}")
    print(f"  z = {exponents.z.value:.3f} ± {exponents.z.error:.3f}")
    print(f"  β = {exponents.beta.value:.3f} ± {exponents.beta.error:.3f}")
    print(f"  γ = {exponents.gamma.value:.3f} ± {exponents.gamma.error:.3f}")
    print(f"  η = {exponents.eta.value:.3f} ± {exponents.eta.error:.3f}")
    
    results['exponents'] = {
        'nu': {'value': exponents.nu.value, 'error': exponents.nu.error, 'valid': exponents.nu.is_valid},
        'z': {'value': exponents.z.value, 'error': exponents.z.error, 'valid': exponents.z.is_valid},
        'beta': {'value': exponents.beta.value, 'error': exponents.beta.error, 'valid': exponents.beta.is_valid},
        'gamma': {'value': exponents.gamma.value, 'error': exponents.gamma.error, 'valid': exponents.gamma.is_valid},
        'eta': {'value': exponents.eta.value, 'error': exponents.eta.error, 'valid': exponents.eta.is_valid}
    }
    
    # Step 4: Entanglement Analysis
    print()
    print("-" * 50)
    print("Step 4: Entanglement entropy analysis")
    print("-" * 50)
    
    ent_analyzer = EntanglementAnalyzer(
        system_sizes=system_sizes,
        n_disorder_realizations=min(n_realizations, 50),
        random_seed=42
    )
    
    print(f"  Computing entropy scaling at hc = {collapse_result.hc_inf:.4f}...")
    ent_data, entropy_scaling = ent_analyzer.compute_entropy_scaling(
        h=collapse_result.hc_inf,
        W=W,
        parallel=True
    )
    
    print(f"  Scaling type: {entropy_scaling.scaling_type}")
    if entropy_scaling.central_charge is not None:
        print(f"  Central charge: c = {entropy_scaling.central_charge:.3f} ± {entropy_scaling.central_charge_error:.3f}")
    print(f"  Fit quality: {entropy_scaling.fit_quality:.3f}")
    
    results['entanglement'] = {
        'scaling_type': entropy_scaling.scaling_type,
        'central_charge': entropy_scaling.central_charge,
        'central_charge_error': entropy_scaling.central_charge_error,
        'fit_quality': entropy_scaling.fit_quality
    }
    
    # Step 5: Validation Summary
    print()
    print("-" * 50)
    print("Step 5: Validation summary")
    print("-" * 50)
    
    # Check scaling relations
    scaling_satisfied = sum(1 for sc in exponents.scaling_checks if sc.is_satisfied)
    total_relations = len(exponents.scaling_checks)
    
    print(f"  Scaling relations: {scaling_satisfied}/{total_relations} satisfied")
    print(f"  Universality class: {exponents.universality_class}")
    print(f"  Overall quality: {exponents.overall_quality:.1%}")
    
    # Compute confidence
    valid_exponents = sum([
        exponents.nu.is_valid,
        exponents.z.is_valid,
        exponents.beta.is_valid,
        exponents.gamma.is_valid,
        exponents.eta.is_valid
    ])
    
    confidence = (
        0.3 * (valid_exponents / 5) +
        0.3 * (scaling_satisfied / max(1, total_relations)) +
        0.2 * collapse_result.collapse_quality +
        0.2 * entropy_scaling.fit_quality
    )
    
    results['validation'] = {
        'valid_exponents': valid_exponents,
        'scaling_relations_satisfied': scaling_satisfied,
        'universality_class': exponents.universality_class,
        'overall_confidence': confidence
    }
    
    # Save results
    results_file = output_path / "quantum_discovery_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_file}")
    
    # Step 6: Generate Figures
    print()
    print("-" * 50)
    print("Step 6: Generating figures")
    print("-" * 50)
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Phase diagram (susceptibility)
        ax = axes[0, 0]
        for L in system_sizes:
            L_points = sorted([p for p in data_points if p.L == L], key=lambda p: p.h)
            h_vals = [p.h for p in L_points]
            chi_vals = [p.susceptibility for p in L_points]
            ax.plot(h_vals, chi_vals, 'o-', label=f'L={L}', markersize=3)
        ax.axvline(collapse_result.hc_inf, color='k', linestyle='--', label=f'hc={collapse_result.hc_inf:.3f}')
        ax.set_xlabel('Transverse field h')
        ax.set_ylabel('Susceptibility χ')
        ax.set_title('Susceptibility vs h')
        ax.legend()
        
        # Scaling collapse
        ax = axes[0, 1]
        for L, (x_scaled, y_scaled) in collapse_result.scaled_data.items():
            ax.plot(x_scaled, y_scaled, 'o', label=f'L={L}', markersize=3, alpha=0.7)
        ax.set_xlabel('$(h - h_c) L^{1/\\nu}$')
        ax.set_ylabel('$\\chi / L^{\\gamma/\\nu}$')
        ax.set_title(f'Scaling Collapse (quality={collapse_result.collapse_quality:.2f})')
        ax.legend()
        
        # Entanglement entropy
        ax = axes[1, 0]
        L_vals = [p.L for p in ent_data]
        S_vals = [p.entropy for p in ent_data]
        S_errs = [p.entropy_std for p in ent_data]
        ax.errorbar(L_vals, S_vals, yerr=S_errs, fmt='o-', capsize=3)
        ax.set_xlabel('System size L')
        ax.set_ylabel('Entanglement entropy S')
        ax.set_title(f'Entropy Scaling ({entropy_scaling.scaling_type})')
        
        # Exponent comparison
        ax = axes[1, 1]
        known_ising = {'ν': 1.0, 'z': 1.0, 'β': 0.125, 'γ': 1.75, 'η': 0.25}
        measured = {
            'ν': exponents.nu.value,
            'z': exponents.z.value,
            'β': exponents.beta.value,
            'γ': exponents.gamma.value,
            'η': exponents.eta.value
        }
        errors = {
            'ν': exponents.nu.error,
            'z': exponents.z.error,
            'β': exponents.beta.error,
            'γ': exponents.gamma.error,
            'η': exponents.eta.error
        }
        
        x = np.arange(len(known_ising))
        width = 0.35
        
        ax.bar(x - width/2, list(known_ising.values()), width, label='1D Ising (exact)', alpha=0.7)
        ax.bar(x + width/2, list(measured.values()), width, label='This work', alpha=0.7,
               yerr=list(errors.values()), capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(list(known_ising.keys()))
        ax.set_ylabel('Exponent value')
        ax.set_title('Critical Exponents Comparison')
        ax.legend()
        
        plt.tight_layout()
        fig_file = output_path / "quantum_discovery_results.png"
        plt.savefig(fig_file, dpi=150)
        plt.close()
        print(f"  Figure saved to: {fig_file}")
        
    except Exception as e:
        print(f"  Warning: Could not generate figures: {e}")
    
    # Final summary
    print()
    print("=" * 70)
    print("QUANTUM DISCOVERY RECREATION COMPLETE")
    print("=" * 70)
    print(f"  Critical point: hc = {collapse_result.hc_inf:.4f} ± {collapse_result.hc_inf_error:.4f}")
    print(f"  Universality class: {exponents.universality_class}")
    print(f"  Overall confidence: {confidence:.1%}")
    print()
    
    if confidence >= 0.95:
        print("  ✓ SUCCESS: Confidence ≥ 95%")
    elif confidence >= 0.90:
        print("  ⚠ ACCEPTABLE: Confidence ≥ 90%")
    else:
        print("  ✗ WARNING: Confidence < 90%")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Recreate quantum discovery campaign results"
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Run in quick mode (validation only)'
    )
    parser.add_argument(
        '--full', action='store_true',
        help='Run in full mode (publication-quality)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    if args.full:
        mode = 'full'
    elif args.quick:
        mode = 'quick'
    else:
        mode = 'standard'
    
    results = run_quantum_discovery_pipeline(
        mode=mode,
        output_dir=args.output_dir
    )
    
    confidence = results.get('validation', {}).get('overall_confidence', 0)
    sys.exit(0 if confidence >= 0.85 else 1)


if __name__ == "__main__":
    main()
