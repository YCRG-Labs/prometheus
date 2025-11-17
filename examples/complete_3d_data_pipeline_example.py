#!/usr/bin/env python3
"""
Complete 3D Data Generation and Quality Analysis Pipeline Example.

This example demonstrates the full implementation of tasks 3.1 and 3.2:
- Generate comprehensive 3D Ising dataset with temperature sweeps and multiple system sizes
- Perform data quality validation and magnetization analysis
- Create visualization tools for 2D slices of 3D configurations

This serves as a complete demonstration of the 3D data generation pipeline
for the PRE paper project.
"""

import sys
import os
import time
from pathlib import Path

# Add src to path for standalone execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_generator_3d import (
    generate_3d_ising_dataset,
    DataGenerationConfig3D,
    create_default_3d_config
)
from analysis.data_quality_3d import analyze_3d_dataset_quality
import logging


def demonstrate_small_scale_generation():
    """Demonstrate 3D data generation with small-scale parameters for quick testing."""
    print("=== Small-Scale 3D Data Generation Demo ===")
    
    # Create configuration for quick demonstration
    config = DataGenerationConfig3D(
        temperature_range=(4.0, 5.0),  # Narrow range around Tc
        temperature_resolution=11,      # Fewer temperature points
        system_sizes=[8, 12],          # Smaller systems
        n_configs_per_temp=50,         # Fewer configurations
        sampling_interval=50,          # Shorter sampling interval
        equilibration_quality_threshold=0.6,  # More lenient threshold
        parallel_processes=2,          # Limited parallelism
        output_dir="data/demo"
    )
    
    print(f"Configuration:")
    print(f"  Temperature range: {config.temperature_range}")
    print(f"  Temperature points: {config.temperature_resolution}")
    print(f"  System sizes: {config.system_sizes}")
    print(f"  Configs per temperature: {config.n_configs_per_temp}")
    print(f"  Total configurations: {len(config.system_sizes) * config.temperature_resolution * config.n_configs_per_temp}")
    
    # Generate dataset
    start_time = time.time()
    
    try:
        dataset = generate_3d_ising_dataset(
            config=config,
            use_parallel=True,
            save_dataset=True,
            output_format='hdf5'
        )
        
        generation_time = time.time() - start_time
        
        print(f"\nGeneration completed successfully!")
        print(f"  Time taken: {generation_time:.1f} seconds")
        print(f"  Total configurations: {dataset.total_configurations}")
        print(f"  Theoretical Tc: {dataset.theoretical_tc:.3f}")
        
        return dataset
        
    except Exception as e:
        print(f"Generation failed: {e}")
        return None


def demonstrate_quality_analysis(dataset):
    """Demonstrate comprehensive quality analysis of 3D dataset."""
    print("\n=== 3D Data Quality Analysis Demo ===")
    
    if dataset is None:
        print("No dataset available for analysis")
        return None
    
    try:
        # Perform quality analysis
        report = analyze_3d_dataset_quality(
            dataset=dataset,
            create_visualizations=True,
            output_dir="results/demo_quality_analysis"
        )
        
        print(f"\nQuality analysis completed!")
        print(f"  Overall quality score: {report.overall_quality_score:.3f}")
        print(f"  Validation status: {'PASSED' if report.validation_passed else 'FAILED'}")
        
        # Print detailed results
        print(f"\nSystem-specific results:")
        for size, mag_result in report.magnetization_analysis.items():
            tc_error = abs(mag_result.tc_estimate - dataset.theoretical_tc) / dataset.theoretical_tc * 100
            print(f"  L={size}:")
            print(f"    Tc estimate: {mag_result.tc_estimate:.3f} (error: {tc_error:.1f}%)")
            print(f"    Transition sharpness: {mag_result.transition_sharpness:.3f}")
            print(f"    Fit quality: {mag_result.fit_quality:.3f}")
        
        # Print issues and recommendations
        if report.issues_found:
            print(f"\nIssues found:")
            for issue in report.issues_found:
                print(f"  - {issue}")
        
        if report.recommendations:
            print(f"\nRecommendations:")
            for rec in report.recommendations:
                print(f"  - {rec}")
        
        return report
        
    except Exception as e:
        print(f"Quality analysis failed: {e}")
        return None


def demonstrate_full_scale_generation():
    """Demonstrate full-scale 3D data generation (commented out for safety)."""
    print("\n=== Full-Scale 3D Data Generation (Demonstration) ===")
    
    # Create full configuration as specified in requirements
    full_config = create_default_3d_config()
    
    print(f"Full-scale configuration would be:")
    print(f"  Temperature range: {full_config.temperature_range}")
    print(f"  Temperature points: {full_config.temperature_resolution}")
    print(f"  System sizes: {full_config.system_sizes}")
    print(f"  Configs per temperature: {full_config.n_configs_per_temp}")
    
    total_configs = (len(full_config.system_sizes) * 
                    full_config.temperature_resolution * 
                    full_config.n_configs_per_temp)
    
    print(f"  Total configurations: {total_configs:,}")
    
    # Estimate time
    avg_size = sum(full_config.system_sizes) / len(full_config.system_sizes)
    time_per_config = 0.1 * (avg_size / 16) ** 2
    estimated_hours = (total_configs * time_per_config) / 3600
    
    print(f"  Estimated time: {estimated_hours:.1f} hours")
    print(f"\nTo run full-scale generation, use:")
    print(f"  python scripts/generate_3d_ising_dataset.py")
    print(f"\nTo analyze the results, use:")
    print(f"  python scripts/analyze_3d_data_quality.py <dataset_file>")


def demonstrate_magnetization_curve_analysis(dataset):
    """Demonstrate magnetization curve analysis and visualization."""
    print("\n=== Magnetization Curve Analysis Demo ===")
    
    if dataset is None:
        print("No dataset available for magnetization analysis")
        return
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create magnetization plots for each system size
        fig, axes = plt.subplots(1, len(dataset.system_size_results), figsize=(15, 5))
        if len(dataset.system_size_results) == 1:
            axes = [axes]
        
        for i, (size, size_result) in enumerate(dataset.system_size_results.items()):
            temperatures = size_result.temperatures
            mag_curves = size_result.magnetization_curves
            
            # Calculate mean and std
            mean_mags = np.mean(mag_curves, axis=1)
            std_mags = np.std(mag_curves, axis=1)
            
            # Plot magnetization curve
            axes[i].errorbar(temperatures, mean_mags, yerr=std_mags, 
                           marker='o', markersize=3, linewidth=1, capsize=2)
            axes[i].axvline(dataset.theoretical_tc, color='red', linestyle='--', 
                          label=f'Theoretical Tc = {dataset.theoretical_tc:.3f}')
            axes[i].set_xlabel('Temperature')
            axes[i].set_ylabel('|Magnetization|')
            axes[i].set_title(f'L = {size}')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path("results/demo_quality_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "magnetization_curves_demo.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Magnetization curves saved to: {plot_path}")
        
        # Calculate and display susceptibility peaks
        print(f"\nSusceptibility analysis:")
        for size, size_result in dataset.system_size_results.items():
            temperatures = size_result.temperatures
            mag_curves = size_result.magnetization_curves
            
            # Calculate susceptibility
            mean_mag_squared = np.mean(mag_curves**2, axis=1)
            mean_mags = np.mean(mag_curves, axis=1)
            susceptibility = (mean_mag_squared - mean_mags**2) / temperatures
            
            # Find peak
            max_idx = np.argmax(susceptibility)
            tc_estimate = temperatures[max_idx]
            tc_error = abs(tc_estimate - dataset.theoretical_tc) / dataset.theoretical_tc * 100
            
            print(f"  L={size}: Tc estimate = {tc_estimate:.3f} (error: {tc_error:.1f}%)")
        
    except ImportError:
        print("Matplotlib not available - skipping magnetization curve plots")
    except Exception as e:
        print(f"Magnetization analysis failed: {e}")


def main():
    """Main demonstration function."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("3D Ising Data Generation and Quality Analysis Pipeline Demo")
    print("=" * 60)
    
    # Demonstrate small-scale generation
    dataset = demonstrate_small_scale_generation()
    
    # Demonstrate quality analysis
    report = demonstrate_quality_analysis(dataset)
    
    # Demonstrate magnetization curve analysis
    demonstrate_magnetization_curve_analysis(dataset)
    
    # Show full-scale configuration
    demonstrate_full_scale_generation()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    
    if dataset and report:
        print(f"✓ Successfully generated {dataset.total_configurations} 3D configurations")
        print(f"✓ Quality analysis completed with score {report.overall_quality_score:.3f}")
        print(f"✓ Validation status: {'PASSED' if report.validation_passed else 'FAILED'}")
        
        if report.validation_passed:
            print("\nThe 3D data generation pipeline is working correctly!")
            print("You can now proceed with full-scale data generation for the PRE paper.")
        else:
            print("\nSome issues were found in the generated data.")
            print("Please review the quality report and adjust parameters as needed.")
    else:
        print("Demo encountered errors - please check the logs for details.")
    
    print("\nNext steps:")
    print("1. Run full-scale generation: python scripts/generate_3d_ising_dataset.py")
    print("2. Analyze results: python scripts/analyze_3d_data_quality.py <dataset_file>")
    print("3. Proceed to task 4: Adapt Prometheus architecture for 3D processing")


if __name__ == "__main__":
    main()