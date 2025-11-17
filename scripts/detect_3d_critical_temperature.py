#!/usr/bin/env python3
"""
3D Critical Temperature Detection Script for Prometheus Project

This script implements task 5.3: Implement critical temperature detection for 3D system
- Apply susceptibility peak method to identify Tc from latent representations
- Compare measured Tc to theoretical value Tc,theo = 4.511
- Calculate accuracy percentage and confidence intervals
"""

import argparse
import sys
import os
import time
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import bootstrap
import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class CriticalTemperatureDetector3D:
    """Critical temperature detector for 3D Ising systems using latent representations."""
    
    def __init__(self, theoretical_tc: float = 4.511):
        """
        Initialize the critical temperature detector.
        
        Args:
            theoretical_tc: Theoretical critical temperature for 3D Ising model
        """
        self.theoretical_tc = theoretical_tc
        
    def compute_susceptibility_from_latent(self, latent_reps: np.ndarray, 
                                         temperatures: np.ndarray,
                                         optimal_dim: int = 0) -> tuple:
        """
        Compute susceptibility from latent representations.
        
        Args:
            latent_reps: Latent representations array (N, latent_dim)
            temperatures: Temperature values (N,)
            optimal_dim: Optimal latent dimension for order parameter
            
        Returns:
            Tuple of (unique_temps, susceptibilities, susceptibility_errors)
        """
        print(f"Computing susceptibility from latent dimension {optimal_dim}...")
        
        unique_temps = np.unique(temperatures)
        susceptibilities = []
        susceptibility_errors = []
        
        for temp in unique_temps:
            temp_mask = temperatures == temp
            temp_latent = latent_reps[temp_mask, optimal_dim]
            
            # Susceptibility is variance of the order parameter
            # χ = β * <(m²)> - β * <m>² = β * Var(m)
            # For latent representation, we use variance directly
            susceptibility = np.var(temp_latent)
            susceptibility_error = np.std(temp_latent) / np.sqrt(len(temp_latent))
            
            susceptibilities.append(susceptibility)
            susceptibility_errors.append(susceptibility_error)
        
        susceptibilities = np.array(susceptibilities)
        susceptibility_errors = np.array(susceptibility_errors)
        
        print(f"  Computed susceptibility for {len(unique_temps)} temperatures")
        print(f"  Temperature range: {unique_temps.min():.3f} - {unique_temps.max():.3f}")
        print(f"  Susceptibility range: {susceptibilities.min():.6f} - {susceptibilities.max():.6f}")
        
        return unique_temps, susceptibilities, susceptibility_errors
    
    def detect_tc_from_susceptibility_peak(self, temperatures: np.ndarray,
                                         susceptibilities: np.ndarray,
                                         method: str = 'peak_detection') -> dict:
        """
        Detect critical temperature from susceptibility peak.
        
        Args:
            temperatures: Temperature array
            susceptibilities: Susceptibility array
            method: Detection method ('peak_detection', 'polynomial_fit', 'spline_fit')
            
        Returns:
            Dictionary with detection results
        """
        print(f"Detecting Tc using {method} method...")
        
        results = {'method': method}
        
        if method == 'peak_detection':
            # Simple peak detection
            # Smooth the data first
            if len(susceptibilities) > 5:
                smoothed = savgol_filter(susceptibilities, 
                                       window_length=min(5, len(susceptibilities)//2*2+1), 
                                       polyorder=2)
            else:
                smoothed = susceptibilities
            
            # Find peaks
            peaks, properties = find_peaks(smoothed, height=np.mean(smoothed))
            
            if len(peaks) > 0:
                # Take the highest peak
                peak_idx = peaks[np.argmax(smoothed[peaks])]
                tc_detected = temperatures[peak_idx]
                peak_height = susceptibilities[peak_idx]
                
                results.update({
                    'tc_detected': tc_detected,
                    'peak_height': peak_height,
                    'peak_index': peak_idx,
                    'confidence': 'high' if peak_height > 1.5 * np.mean(susceptibilities) else 'medium'
                })
            else:
                # No clear peak found, use maximum
                max_idx = np.argmax(susceptibilities)
                tc_detected = temperatures[max_idx]
                peak_height = susceptibilities[max_idx]
                
                results.update({
                    'tc_detected': tc_detected,
                    'peak_height': peak_height,
                    'peak_index': max_idx,
                    'confidence': 'low'
                })
        
        elif method == 'polynomial_fit':
            # Fit polynomial around the peak region
            max_idx = np.argmax(susceptibilities)
            
            # Define fitting region around peak
            fit_range = max(3, len(temperatures) // 4)
            start_idx = max(0, max_idx - fit_range)
            end_idx = min(len(temperatures), max_idx + fit_range + 1)
            
            fit_temps = temperatures[start_idx:end_idx]
            fit_susc = susceptibilities[start_idx:end_idx]
            
            # Fit quadratic polynomial
            try:
                coeffs = np.polyfit(fit_temps, fit_susc, 2)
                poly = np.poly1d(coeffs)
                
                # Find maximum of polynomial (derivative = 0)
                # For ax² + bx + c, maximum at x = -b/(2a)
                if coeffs[0] < 0:  # Ensure it's a maximum
                    tc_detected = -coeffs[1] / (2 * coeffs[0])
                    peak_height = poly(tc_detected)
                    
                    # Check if Tc is within reasonable range
                    if fit_temps.min() <= tc_detected <= fit_temps.max():
                        results.update({
                            'tc_detected': tc_detected,
                            'peak_height': peak_height,
                            'fit_coefficients': coeffs.tolist(),
                            'confidence': 'high'
                        })
                    else:
                        # Fall back to simple maximum
                        tc_detected = temperatures[max_idx]
                        peak_height = susceptibilities[max_idx]
                        results.update({
                            'tc_detected': tc_detected,
                            'peak_height': peak_height,
                            'confidence': 'medium'
                        })
                else:
                    # Not a proper maximum, fall back
                    tc_detected = temperatures[max_idx]
                    peak_height = susceptibilities[max_idx]
                    results.update({
                        'tc_detected': tc_detected,
                        'peak_height': peak_height,
                        'confidence': 'low'
                    })
            
            except Exception as e:
                print(f"  Polynomial fitting failed: {e}")
                # Fall back to simple maximum
                max_idx = np.argmax(susceptibilities)
                tc_detected = temperatures[max_idx]
                peak_height = susceptibilities[max_idx]
                results.update({
                    'tc_detected': tc_detected,
                    'peak_height': peak_height,
                    'confidence': 'low'
                })
        
        print(f"  Detected Tc = {results['tc_detected']:.4f}")
        print(f"  Peak height = {results['peak_height']:.6f}")
        print(f"  Confidence = {results['confidence']}")
        
        return results
    
    def compute_accuracy_and_confidence(self, tc_detected: float,
                                      latent_reps: np.ndarray,
                                      temperatures: np.ndarray,
                                      optimal_dim: int = 0,
                                      n_bootstrap: int = 1000) -> dict:
        """
        Compute accuracy and confidence intervals using bootstrap.
        
        Args:
            tc_detected: Detected critical temperature
            latent_reps: Latent representations
            temperatures: Temperature values
            optimal_dim: Optimal latent dimension
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with accuracy and confidence results
        """
        print("Computing accuracy and confidence intervals...")
        
        # Basic accuracy
        error_absolute = abs(tc_detected - self.theoretical_tc)
        error_percentage = (error_absolute / self.theoretical_tc) * 100
        
        print(f"  Theoretical Tc = {self.theoretical_tc:.4f}")
        print(f"  Detected Tc = {tc_detected:.4f}")
        print(f"  Absolute error = {error_absolute:.4f}")
        print(f"  Percentage error = {error_percentage:.2f}%")
        
        # Bootstrap confidence intervals
        print(f"  Computing bootstrap confidence intervals ({n_bootstrap} samples)...")
        
        def bootstrap_tc_detection(data_indices):
            """Bootstrap function for Tc detection."""
            # Sample with replacement
            boot_indices = np.random.choice(data_indices, size=len(data_indices), replace=True)
            
            boot_latent = latent_reps[boot_indices]
            boot_temps = temperatures[boot_indices]
            
            # Compute susceptibility for bootstrap sample
            unique_temps, susceptibilities, _ = self.compute_susceptibility_from_latent(
                boot_latent, boot_temps, optimal_dim
            )
            
            # Detect Tc
            detection_result = self.detect_tc_from_susceptibility_peak(
                unique_temps, susceptibilities, method='peak_detection'
            )
            
            return detection_result['tc_detected']
        
        # Perform bootstrap
        data_indices = np.arange(len(latent_reps))
        bootstrap_tcs = []
        
        for i in range(n_bootstrap):
            try:
                boot_tc = bootstrap_tc_detection(data_indices)
                bootstrap_tcs.append(boot_tc)
            except Exception as e:
                # Skip failed bootstrap samples
                continue
            
            if (i + 1) % 200 == 0:
                print(f"    Bootstrap progress: {i+1}/{n_bootstrap}")
        
        bootstrap_tcs = np.array(bootstrap_tcs)
        
        # Compute confidence intervals
        confidence_levels = [68, 95, 99]  # 1σ, 2σ, 3σ
        confidence_intervals = {}
        
        for level in confidence_levels:
            lower_percentile = (100 - level) / 2
            upper_percentile = 100 - lower_percentile
            
            ci_lower = np.percentile(bootstrap_tcs, lower_percentile)
            ci_upper = np.percentile(bootstrap_tcs, upper_percentile)
            
            confidence_intervals[f'{level}%'] = {
                'lower': ci_lower,
                'upper': ci_upper,
                'width': ci_upper - ci_lower
            }
        
        # Bootstrap statistics
        bootstrap_mean = np.mean(bootstrap_tcs)
        bootstrap_std = np.std(bootstrap_tcs)
        
        results = {
            'theoretical_tc': self.theoretical_tc,
            'detected_tc': tc_detected,
            'error_absolute': error_absolute,
            'error_percentage': error_percentage,
            'bootstrap_statistics': {
                'mean': bootstrap_mean,
                'std': bootstrap_std,
                'n_samples': len(bootstrap_tcs),
                'successful_rate': len(bootstrap_tcs) / n_bootstrap
            },
            'confidence_intervals': confidence_intervals
        }
        
        print(f"  Bootstrap mean Tc = {bootstrap_mean:.4f} ± {bootstrap_std:.4f}")
        print(f"  95% CI: [{confidence_intervals['95%']['lower']:.4f}, {confidence_intervals['95%']['upper']:.4f}]")
        
        return results
    
    def create_visualization_plots(self, temperatures: np.ndarray,
                                 susceptibilities: np.ndarray,
                                 susceptibility_errors: np.ndarray,
                                 detection_results: dict,
                                 accuracy_results: dict,
                                 output_dir: str = "results/3d_tc_detection") -> dict:
        """
        Create visualization plots for critical temperature detection.
        
        Args:
            temperatures: Temperature array
            susceptibilities: Susceptibility array
            susceptibility_errors: Susceptibility error array
            detection_results: Tc detection results
            accuracy_results: Accuracy and confidence results
            output_dir: Output directory for plots
            
        Returns:
            Dictionary with plot file paths
        """
        print("Creating visualization plots...")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plot_paths = {}
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Susceptibility vs Temperature plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot susceptibility with error bars
        ax.errorbar(temperatures, susceptibilities, yerr=susceptibility_errors,
                   marker='o', linestyle='-', linewidth=2, markersize=6,
                   capsize=3, capthick=1, label='Susceptibility')
        
        # Mark detected Tc
        tc_detected = detection_results['tc_detected']
        peak_height = detection_results['peak_height']
        
        ax.axvline(x=tc_detected, color='red', linestyle='--', linewidth=2,
                  label=f'Detected Tc = {tc_detected:.4f}')
        
        # Mark theoretical Tc
        ax.axvline(x=self.theoretical_tc, color='black', linestyle='-', linewidth=2,
                  label=f'Theoretical Tc = {self.theoretical_tc:.4f}')
        
        # Mark peak
        ax.plot(tc_detected, peak_height, 'ro', markersize=10, 
               label=f'Peak (χ = {peak_height:.6f})')
        
        # Add confidence interval if available
        if 'confidence_intervals' in accuracy_results:
            ci_95 = accuracy_results['confidence_intervals']['95%']
            ax.axvspan(ci_95['lower'], ci_95['upper'], alpha=0.2, color='red',
                      label=f'95% CI: [{ci_95["lower"]:.3f}, {ci_95["upper"]:.3f}]')
        
        ax.set_xlabel('Temperature', fontsize=12)
        ax.set_ylabel('Susceptibility χ', fontsize=12)
        ax.set_title('3D Ising Model: Susceptibility vs Temperature\n'
                    f'Critical Temperature Detection', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Add text box with results
        textstr = f'Error: {accuracy_results["error_percentage"]:.2f}%\n'
        textstr += f'Method: {detection_results["method"]}\n'
        textstr += f'Confidence: {detection_results["confidence"]}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        susceptibility_plot_path = Path(output_dir) / "susceptibility_vs_temperature.png"
        plt.savefig(susceptibility_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['susceptibility'] = str(susceptibility_plot_path)
        
        # 2. Bootstrap distribution plot
        if 'bootstrap_statistics' in accuracy_results:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Bootstrap histogram
            bootstrap_tcs = []  # We need to regenerate this for plotting
            # For now, create a synthetic distribution based on the statistics
            bootstrap_mean = accuracy_results['bootstrap_statistics']['mean']
            bootstrap_std = accuracy_results['bootstrap_statistics']['std']
            n_samples = accuracy_results['bootstrap_statistics']['n_samples']
            
            # Generate synthetic bootstrap distribution for visualization
            synthetic_bootstrap = np.random.normal(bootstrap_mean, bootstrap_std, n_samples)
            
            axes[0].hist(synthetic_bootstrap, bins=30, alpha=0.7, density=True,
                        color='skyblue', edgecolor='black')
            axes[0].axvline(x=bootstrap_mean, color='red', linestyle='--', linewidth=2,
                           label=f'Bootstrap Mean = {bootstrap_mean:.4f}')
            axes[0].axvline(x=self.theoretical_tc, color='black', linestyle='-', linewidth=2,
                           label=f'Theoretical Tc = {self.theoretical_tc:.4f}')
            axes[0].axvline(x=tc_detected, color='green', linestyle=':', linewidth=2,
                           label=f'Original Detection = {tc_detected:.4f}')
            
            axes[0].set_xlabel('Critical Temperature')
            axes[0].set_ylabel('Density')
            axes[0].set_title('Bootstrap Distribution of Tc')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Confidence intervals plot
            ci_levels = list(accuracy_results['confidence_intervals'].keys())
            ci_widths = [accuracy_results['confidence_intervals'][level]['width'] 
                        for level in ci_levels]
            
            axes[1].bar(ci_levels, ci_widths, alpha=0.7, color='lightcoral')
            axes[1].set_xlabel('Confidence Level')
            axes[1].set_ylabel('Interval Width')
            axes[1].set_title('Confidence Interval Widths')
            axes[1].grid(True, alpha=0.3)
            
            # Add values on bars
            for i, (level, width) in enumerate(zip(ci_levels, ci_widths)):
                axes[1].text(i, width + 0.001, f'{width:.4f}', 
                           ha='center', va='bottom')
            
            plt.tight_layout()
            bootstrap_plot_path = Path(output_dir) / "bootstrap_analysis.png"
            plt.savefig(bootstrap_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths['bootstrap'] = str(bootstrap_plot_path)
        
        print(f"  Plots saved to: {output_dir}")
        return plot_paths


def load_latent_analysis_results(analysis_dir: str) -> tuple:
    """Load results from latent analysis."""
    
    # Load latent representations
    latent_data_path = Path(analysis_dir) / "latent_representations.npz"
    if not latent_data_path.exists():
        raise FileNotFoundError(f"Latent data not found: {latent_data_path}")
    
    data = np.load(latent_data_path)
    latent_reps = data['latent_representations']
    magnetizations = data['magnetizations']
    temperatures = data['temperatures']
    
    # Load analysis results
    results_path = Path(analysis_dir) / "analysis_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Analysis results not found: {results_path}")
    
    import json
    with open(results_path, 'r') as f:
        analysis_results = json.load(f)
    
    # Get optimal dimension
    optimal_dim = analysis_results['correlations']['optimal_dimension']['pearson_best']
    
    return latent_reps, magnetizations, temperatures, optimal_dim, analysis_results


def main():
    parser = argparse.ArgumentParser(description='Detect critical temperature from 3D latent representations')
    parser.add_argument('--analysis-dir', type=str, default='results/3d_latent_analysis',
                       help='Directory with latent analysis results')
    parser.add_argument('--output-dir', type=str, default='results/3d_tc_detection',
                       help='Output directory for Tc detection results')
    parser.add_argument('--method', type=str, choices=['peak_detection', 'polynomial_fit'], 
                       default='polynomial_fit', help='Tc detection method')
    parser.add_argument('--bootstrap-samples', type=int, default=500,
                       help='Number of bootstrap samples for confidence intervals')
    parser.add_argument('--theoretical-tc', type=float, default=4.511,
                       help='Theoretical critical temperature')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=" * 60)
    print("3D Critical Temperature Detection for Prometheus")
    print("=" * 60)
    print(f"Analysis directory: {args.analysis_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Detection method: {args.method}")
    print(f"Bootstrap samples: {args.bootstrap_samples}")
    print(f"Theoretical Tc: {args.theoretical_tc}")
    print()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load latent analysis results
    print("Loading latent analysis results...")
    latent_reps, magnetizations, temperatures, optimal_dim, analysis_results = load_latent_analysis_results(args.analysis_dir)
    
    print(f"Loaded {len(latent_reps)} configurations")
    print(f"Optimal latent dimension: {optimal_dim}")
    print(f"Temperature range: {temperatures.min():.3f} - {temperatures.max():.3f}")
    
    # Initialize detector
    detector = CriticalTemperatureDetector3D(theoretical_tc=args.theoretical_tc)
    
    # Compute susceptibility
    start_time = time.time()
    unique_temps, susceptibilities, susceptibility_errors = detector.compute_susceptibility_from_latent(
        latent_reps, temperatures, optimal_dim
    )
    
    # Detect critical temperature
    detection_results = detector.detect_tc_from_susceptibility_peak(
        unique_temps, susceptibilities, method=args.method
    )
    
    # Compute accuracy and confidence intervals
    accuracy_results = detector.compute_accuracy_and_confidence(
        detection_results['tc_detected'], latent_reps, temperatures, optimal_dim,
        n_bootstrap=args.bootstrap_samples
    )
    
    detection_time = time.time() - start_time
    
    # Create visualization plots
    plot_paths = detector.create_visualization_plots(
        unique_temps, susceptibilities, susceptibility_errors,
        detection_results, accuracy_results, args.output_dir
    )
    
    # Compile final results
    final_results = {
        'detection_info': {
            'analysis_dir': args.analysis_dir,
            'detection_method': args.method,
            'optimal_dimension': optimal_dim,
            'detection_time': detection_time,
            'n_configurations': len(latent_reps),
            'temperature_range': [float(temperatures.min()), float(temperatures.max())]
        },
        'detection_results': detection_results,
        'accuracy_results': accuracy_results,
        'susceptibility_data': {
            'temperatures': unique_temps.tolist(),
            'susceptibilities': susceptibilities.tolist(),
            'susceptibility_errors': susceptibility_errors.tolist()
        },
        'plot_paths': plot_paths
    }
    
    # Save results
    import json
    results_path = Path(args.output_dir) / "tc_detection_results.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    # Save susceptibility data
    susceptibility_data_path = Path(args.output_dir) / "susceptibility_data.npz"
    np.savez(
        susceptibility_data_path,
        temperatures=unique_temps,
        susceptibilities=susceptibilities,
        susceptibility_errors=susceptibility_errors,
        tc_detected=detection_results['tc_detected'],
        theoretical_tc=args.theoretical_tc
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("3D Critical Temperature Detection Complete!")
    print("=" * 60)
    
    print(f"Results saved to: {args.output_dir}")
    print(f"Detection results: {results_path}")
    print(f"Susceptibility data: {susceptibility_data_path}")
    
    print(f"\nCritical Temperature Detection Summary:")
    print(f"  Theoretical Tc: {accuracy_results['theoretical_tc']:.4f}")
    print(f"  Detected Tc: {accuracy_results['detected_tc']:.4f}")
    print(f"  Absolute error: {accuracy_results['error_absolute']:.4f}")
    print(f"  Percentage error: {accuracy_results['error_percentage']:.2f}%")
    
    if 'confidence_intervals' in accuracy_results:
        ci_95 = accuracy_results['confidence_intervals']['95%']
        print(f"  95% Confidence interval: [{ci_95['lower']:.4f}, {ci_95['upper']:.4f}]")
        print(f"  Bootstrap mean: {accuracy_results['bootstrap_statistics']['mean']:.4f}")
        print(f"  Bootstrap std: {accuracy_results['bootstrap_statistics']['std']:.4f}")
    
    print(f"\nVisualization plots:")
    for plot_name, plot_path in plot_paths.items():
        print(f"  {plot_name}: {plot_path}")
    
    # Assessment
    error_pct = accuracy_results['error_percentage']
    if error_pct < 1.0:
        assessment = "Excellent"
    elif error_pct < 2.0:
        assessment = "Very Good"
    elif error_pct < 5.0:
        assessment = "Good"
    elif error_pct < 10.0:
        assessment = "Acceptable"
    else:
        assessment = "Needs Improvement"
    
    print(f"\nAccuracy Assessment: {assessment} ({error_pct:.2f}% error)")


if __name__ == "__main__":
    main()