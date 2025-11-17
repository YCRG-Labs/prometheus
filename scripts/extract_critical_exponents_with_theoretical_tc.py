#!/usr/bin/env python3
"""
Critical Exponent Extraction with Theoretical Tc

This script uses theoretical critical temperatures and focuses on improving
the power-law fitting accuracy in the critical region.
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import linregress

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.latent_analysis import LatentRepresentation
from src.utils.config import PrometheusConfig
from src.utils.logging_utils import setup_logging, get_logger


def load_data_simple(data_path):
    """Load data with proper format detection."""
    logger = get_logger(__name__)
    
    if data_path.endswith('.npz'):
        # 2D data
        logger.info(f"Loading 2D data from {data_path}")
        data = np.load(data_path, allow_pickle=True)
        
        configurations = data['spin_configurations']
        magnetizations = data['magnetizations']
        energies = data['energies']
        metadata = data['metadata'].item()
        
        # Extract temperatures
        n_temps = metadata['n_temperatures']
        temp_min, temp_max = metadata['temp_range']
        temperatures = np.linspace(temp_min, temp_max, n_temps)
        n_configs_per_temp = configurations.shape[0] // n_temps
        temp_array = np.repeat(temperatures, n_configs_per_temp)
        
        system_type = 'ising_2d'
        theoretical_tc = 2.269
        
    elif data_path.endswith('.h5'):
        # 3D data
        logger.info(f"Loading 3D data from {data_path}")
        with h5py.File(data_path, 'r') as f:
            configurations = f['configurations'][:]
            magnetizations = f['magnetizations'][:]
            energies = f['energies'][:]
            temp_array = f['temperatures'][:]
        
        system_type = 'ising_3d'
        theoretical_tc = 4.511
    
    else:
        raise ValueError(f"Unsupported data format: {data_path}")
    
    return {
        'magnetizations': magnetizations,
        'energies': energies,
        'temperatures': temp_array,
        'system_type': system_type,
        'theoretical_tc': theoretical_tc
    }


def extract_beta_exponent_improved(temperatures, magnetizations, tc, system_type):
    """Extract β exponent with improved methods."""
    logger = get_logger(__name__)
    
    # Use only temperatures below Tc for β exponent
    below_tc_mask = temperatures < tc
    
    if np.sum(below_tc_mask) < 10:
        logger.warning("Very few points below Tc, using broader range")
        temp_range = np.max(temperatures) - np.min(temperatures)
        below_tc_mask = temperatures < (tc + 0.1 * temp_range)
    
    fit_temps = temperatures[below_tc_mask]
    fit_mags = np.abs(magnetizations[below_tc_mask])
    
    # Remove zero magnetizations
    nonzero_mask = fit_mags > 1e-10
    fit_temps = fit_temps[nonzero_mask]
    fit_mags = fit_mags[nonzero_mask]
    
    if len(fit_temps) < 5:
        raise ValueError("Insufficient data for β exponent fitting")
    
    logger.info(f"Fitting β exponent with {len(fit_temps)} points")
    
    # Method 1: Focus on region very close to Tc
    temp_range = tc - np.min(fit_temps)
    close_to_tc_range = 0.3 * temp_range  # Use closest 30% of temperature range
    
    close_mask = (tc - fit_temps) <= close_to_tc_range
    if np.sum(close_mask) >= 5:
        fit_temps_close = fit_temps[close_mask]
        fit_mags_close = fit_mags[close_mask]
        
        # Power law: m = A * (Tc - T)^β
        reduced_temps = tc - fit_temps_close
        reduced_temps = np.maximum(reduced_temps, 1e-10)
        
        # Log-log fit
        log_reduced = np.log(reduced_temps)
        log_mags = np.log(fit_mags_close)
        
        slope, intercept, r_value, p_value, std_err = linregress(log_reduced, log_mags)
        
        result = {
            'exponent': slope,
            'amplitude': np.exp(intercept),
            'exponent_error': std_err,
            'r_squared': r_value**2,
            'p_value': p_value,
            'n_points': len(fit_temps_close),
            'method': 'close_to_tc_log_linear',
            'fit_range': (np.min(fit_temps_close), np.max(fit_temps_close))
        }
        
        logger.info(f"β = {slope:.4f} ± {std_err:.4f}, R² = {r_value**2:.4f}")
        
        return result
    
    else:
        # Fallback to broader range
        reduced_temps = tc - fit_temps
        reduced_temps = np.maximum(reduced_temps, 1e-10)
        
        log_reduced = np.log(reduced_temps)
        log_mags = np.log(fit_mags)
        
        slope, intercept, r_value, p_value, std_err = linregress(log_reduced, log_mags)
        
        return {
            'exponent': slope,
            'amplitude': np.exp(intercept),
            'exponent_error': std_err,
            'r_squared': r_value**2,
            'p_value': p_value,
            'n_points': len(fit_temps),
            'method': 'broad_range_log_linear',
            'fit_range': (np.min(fit_temps), np.max(fit_temps))
        }


def extract_nu_exponent_improved(temperatures, magnetizations, tc, system_type):
    """Extract ν exponent using improved correlation length estimation."""
    logger = get_logger(__name__)
    
    # Bin temperatures and compute local susceptibility (correlation length proxy)
    unique_temps = np.unique(temperatures)
    
    # Focus on temperatures near Tc
    temp_range = np.max(temperatures) - np.min(temperatures)
    near_tc_range = 0.4 * temp_range
    
    near_tc_temps = unique_temps[np.abs(unique_temps - tc) <= near_tc_range]
    
    if len(near_tc_temps) < 5:
        near_tc_temps = unique_temps
    
    corr_lengths = []
    valid_temps = []
    
    for temp in near_tc_temps:
        # Get magnetizations at this temperature
        temp_mask = np.abs(temperatures - temp) < 0.01
        
        if np.sum(temp_mask) > 3:
            local_mags = magnetizations[temp_mask]
            
            # Use susceptibility as correlation length proxy
            # χ ∝ ξ^(2-η) where η ≈ 0 for Ising model, so χ ∝ ξ²
            susceptibility = np.var(local_mags)
            
            if susceptibility > 0:
                corr_length = np.sqrt(susceptibility)
                corr_lengths.append(corr_length)
                valid_temps.append(temp)
    
    if len(valid_temps) < 5:
        raise ValueError("Insufficient temperature points for ν exponent")
    
    corr_lengths = np.array(corr_lengths)
    valid_temps = np.array(valid_temps)
    
    logger.info(f"Fitting ν exponent with {len(valid_temps)} temperature points")
    
    # Power law: ξ = A * |T - Tc|^(-ν)
    reduced_temps = np.abs(valid_temps - tc)
    reduced_temps = np.maximum(reduced_temps, 1e-10)
    
    # Log-log fit
    log_reduced = np.log(reduced_temps)
    log_corr = np.log(corr_lengths)
    
    slope, intercept, r_value, p_value, std_err = linregress(log_reduced, log_corr)
    
    result = {
        'exponent': slope,  # This should be negative for ν
        'amplitude': np.exp(intercept),
        'exponent_error': std_err,
        'r_squared': r_value**2,
        'p_value': p_value,
        'n_points': len(valid_temps),
        'method': 'susceptibility_proxy',
        'fit_range': (np.min(valid_temps), np.max(valid_temps))
    }
    
    logger.info(f"ν = {-slope:.4f} ± {std_err:.4f}, R² = {r_value**2:.4f}")
    
    return result


def analyze_with_theoretical_tc(data):
    """Analyze critical exponents using theoretical critical temperature."""
    logger = get_logger(__name__)
    
    system_type = data['system_type']
    tc = data['theoretical_tc']
    temperatures = data['temperatures']
    magnetizations = data['magnetizations']
    
    # Theoretical values
    theoretical_exponents = {
        'ising_2d': {'beta': 0.125, 'nu': 1.0},
        'ising_3d': {'beta': 0.326, 'nu': 0.630}
    }
    
    theoretical = theoretical_exponents[system_type]
    
    logger.info(f"Analyzing {system_type} with theoretical Tc = {tc:.3f}")
    
    results = {
        'system_type': system_type,
        'critical_temperature': tc,
        'theoretical': theoretical,
        'extracted': {},
        'accuracy': {}
    }
    
    # Extract β exponent
    try:
        beta_result = extract_beta_exponent_improved(temperatures, magnetizations, tc, system_type)
        results['extracted']['beta'] = beta_result
        
        # Calculate accuracy
        beta_error = abs(beta_result['exponent'] - theoretical['beta']) / theoretical['beta']
        beta_accuracy = max(0, (1 - beta_error) * 100)
        results['accuracy']['beta_accuracy_percent'] = beta_accuracy
        
        logger.info(f"β exponent extracted: {beta_result['exponent']:.4f} (accuracy: {beta_accuracy:.1f}%)")
        
    except Exception as e:
        logger.error(f"β exponent extraction failed: {e}")
        results['extracted']['beta'] = None
    
    # Extract ν exponent
    try:
        nu_result = extract_nu_exponent_improved(temperatures, magnetizations, tc, system_type)
        
        # Convert to positive ν (since we fit ξ ∝ |T-Tc|^(-ν), slope is -ν)
        nu_result['exponent'] = -nu_result['exponent']
        
        results['extracted']['nu'] = nu_result
        
        # Calculate accuracy
        nu_error = abs(nu_result['exponent'] - theoretical['nu']) / theoretical['nu']
        nu_accuracy = max(0, (1 - nu_error) * 100)
        results['accuracy']['nu_accuracy_percent'] = nu_accuracy
        
        logger.info(f"ν exponent extracted: {nu_result['exponent']:.4f} (accuracy: {nu_accuracy:.1f}%)")
        
    except Exception as e:
        logger.error(f"ν exponent extraction failed: {e}")
        results['extracted']['nu'] = None
    
    # Overall accuracy
    if results['extracted']['beta'] and results['extracted']['nu']:
        overall_accuracy = (results['accuracy']['beta_accuracy_percent'] + 
                          results['accuracy']['nu_accuracy_percent']) / 2
        results['accuracy']['overall_accuracy_percent'] = overall_accuracy
    
    return results


def print_results_with_theoretical_tc(results):
    """Print formatted results."""
    
    print("\n" + "=" * 70)
    print(f"CRITICAL EXPONENT ANALYSIS WITH THEORETICAL Tc")
    print("=" * 70)
    
    print(f"System: {results['system_type']}")
    print(f"Theoretical Tc Used: {results['critical_temperature']:.3f}")
    print()
    
    theoretical = results['theoretical']
    extracted = results['extracted']
    accuracy = results['accuracy']
    
    # β exponent
    if extracted.get('beta'):
        beta = extracted['beta']
        print("β EXPONENT (Order Parameter):")
        print(f"  Measured: {beta['exponent']:.4f} ± {beta['exponent_error']:.4f}")
        print(f"  Theoretical: {theoretical['beta']:.4f}")
        print(f"  Accuracy: {accuracy.get('beta_accuracy_percent', 0):.1f}%")
        print(f"  R²: {beta['r_squared']:.4f}")
        print(f"  Method: {beta['method']}")
        print(f"  Fit Points: {beta['n_points']}")
        print(f"  Temperature Range: [{beta['fit_range'][0]:.3f}, {beta['fit_range'][1]:.3f}]")
        
        # Deviation analysis
        deviation = abs(beta['exponent'] - theoretical['beta'])
        relative_deviation = deviation / theoretical['beta'] * 100
        print(f"  Absolute Deviation: {deviation:.4f}")
        print(f"  Relative Deviation: {relative_deviation:.1f}%")
        print()
    
    # ν exponent
    if extracted.get('nu'):
        nu = extracted['nu']
        print("ν EXPONENT (Correlation Length):")
        print(f"  Measured: {nu['exponent']:.4f} ± {nu['exponent_error']:.4f}")
        print(f"  Theoretical: {theoretical['nu']:.4f}")
        print(f"  Accuracy: {accuracy.get('nu_accuracy_percent', 0):.1f}%")
        print(f"  R²: {nu['r_squared']:.4f}")
        print(f"  Method: {nu['method']}")
        print(f"  Fit Points: {nu['n_points']}")
        print(f"  Temperature Range: [{nu['fit_range'][0]:.3f}, {nu['fit_range'][1]:.3f}]")
        
        # Deviation analysis
        deviation = abs(nu['exponent'] - theoretical['nu'])
        relative_deviation = deviation / theoretical['nu'] * 100
        print(f"  Absolute Deviation: {deviation:.4f}")
        print(f"  Relative Deviation: {relative_deviation:.1f}%")
        print()
    
    # Overall performance
    if 'overall_accuracy_percent' in accuracy:
        overall_acc = accuracy['overall_accuracy_percent']
        print("OVERALL PERFORMANCE:")
        print(f"  Combined Accuracy: {overall_acc:.1f}%")
        
        if overall_acc >= 80:
            rating = "Excellent"
        elif overall_acc >= 70:
            rating = "Good"
        elif overall_acc >= 60:
            rating = "Fair"
        else:
            rating = "Needs Improvement"
        
        print(f"  Performance Rating: {rating}")
        
        # Improvement suggestions
        print("\nIMPROVEMENT SUGGESTIONS:")
        if overall_acc < 70:
            print("  • Use trained VAE models for better order parameter extraction")
            print("  • Increase data density near the critical temperature")
            print("  • Consider finite-size scaling corrections")
            print("  • Use more sophisticated correlation length calculations")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Critical exponent extraction with theoretical Tc')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to data file (.npz for 2D, .h5 for 3D)')
    parser.add_argument('--output-dir', type=str, default='results/theoretical_tc_analysis',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Setup logging
    config = PrometheusConfig()
    setup_logging(config.logging)
    
    logger = get_logger(__name__)
    logger.info("Starting critical exponent analysis with theoretical Tc")
    
    try:
        # Load data
        data = load_data_simple(args.data_path)
        
        # Analyze with theoretical Tc
        results = analyze_with_theoretical_tc(data)
        
        # Print results
        print_results_with_theoretical_tc(results)
        
        # Save results
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / f"{data['system_type']}_theoretical_tc_results.npz"
        np.savez(results_file, **results)
        
        logger.info(f"Results saved to {results_file}")
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()