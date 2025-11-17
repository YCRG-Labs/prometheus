#!/usr/bin/env python3
"""
Publication Materials Generation Example

This example demonstrates how to use the comprehensive publication materials
generation system to create all figures, tables, and analysis needed for
Physical Review E submission.

Usage:
    python examples/publication_materials_example.py
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from analysis.systematic_comparison_framework import (
    SystematicComparisonFramework, 
    SystemComparisonData
)
from analysis.critical_exponent_comparison_tables import (
    CriticalExponentComparisonTables,
    CriticalExponentData
)
from analysis.publication_figure_generator import PublicationFigureGenerator
from utils.logging_utils import get_logger


def create_mock_comparison_data() -> list:
    """Create mock system comparison data for demonstration."""
    logger = get_logger(__name__)
    logger.info("Creating mock comparison data for demonstration")
    
    # Generate synthetic latent space data
    n_points = 1000
    
    # 2D Ising system
    temperatures_2d = np.linspace(1.5, 3.5, n_points)
    tc_2d = 2.269
    
    # Create latent space with phase transition
    z1_2d = np.random.normal(0, 1, n_points)
    z2_2d = np.where(temperatures_2d < tc_2d, 
                     np.random.normal(1, 0.3, n_points),
                     np.random.normal(-1, 0.3, n_points))
    
    # Magnetization with critical behavior
    magnetizations_2d = np.abs(np.where(temperatures_2d < tc_2d,
                                       (1 - temperatures_2d/tc_2d)**0.125,
                                       np.random.normal(0, 0.1, n_points)))
    
    system_2d = SystemComparisonData(
        system_name="2D_Ising_L32",
        dimensionality="2D",
        model_type="Ising",
        latent_z1=z1_2d,
        latent_z2=z2_2d,
        temperatures=temperatures_2d,
        magnetizations=magnetizations_2d,
        discovered_tc=2.275,
        theoretical_tc=2.269,
        tc_accuracy_percent=0.26,
        order_parameter_correlation=0.892,
        best_latent_dimension="z2",
        physics_consistency_score=0.85,
        data_quality_score=0.91
    )
    
    # 3D Ising system
    temperatures_3d = np.linspace(3.5, 5.5, n_points)
    tc_3d = 4.511
    
    z1_3d = np.random.normal(0, 1, n_points)
    z2_3d = np.where(temperatures_3d < tc_3d,
                     np.random.normal(1.5, 0.4, n_points),
                     np.random.normal(-1.5, 0.4, n_points))
    
    magnetizations_3d = np.abs(np.where(temperatures_3d < tc_3d,
                                       (1 - temperatures_3d/tc_3d)**0.326,
                                       np.random.normal(0, 0.1, n_points)))
    
    system_3d = SystemComparisonData(
        system_name="3D_Ising_L16",
        dimensionality="3D",
        model_type="Ising",
        latent_z1=z1_3d,
        latent_z2=z2_3d,
        temperatures=temperatures_3d,
        magnetizations=magnetizations_3d,
        discovered_tc=4.523,
        theoretical_tc=4.511,
        tc_accuracy_percent=0.27,
        order_parameter_correlation=0.876,
        best_latent_dimension="z2",
        physics_consistency_score=0.82,
        data_quality_score=0.88
    )
    
    # Potts system
    temperatures_potts = np.linspace(0.5, 1.5, n_points)
    tc_potts = 1.005
    
    z1_potts = np.random.normal(0, 1, n_points)
    z2_potts = np.where(temperatures_potts < tc_potts,
                        np.random.normal(0.8, 0.2, n_points),
                        np.random.normal(-0.8, 0.2, n_points))
    
    magnetizations_potts = np.abs(np.where(temperatures_potts < tc_potts,
                                          (1 - temperatures_potts/tc_potts)**0.111,
                                          np.random.normal(0, 0.05, n_points)))
    
    system_potts = SystemComparisonData(
        system_name="Potts_Q3_L24",
        dimensionality="2D",
        model_type="Potts",
        latent_z1=z1_potts,
        latent_z2=z2_potts,
        temperatures=temperatures_potts,
        magnetizations=magnetizations_potts,
        discovered_tc=1.012,
        theoretical_tc=1.005,
        tc_accuracy_percent=0.70,
        order_parameter_correlation=0.834,
        best_latent_dimension="z2",
        physics_consistency_score=0.78,
        data_quality_score=0.85
    )
    
    return [system_2d, system_3d, system_potts]


def create_mock_exponent_data() -> list:
    """Create mock critical exponent data for demonstration."""
    logger = get_logger(__name__)
    logger.info("Creating mock critical exponent data for demonstration")
    
    # 2D Ising exponents
    exponent_2d = CriticalExponentData(
        system_name="2D_Ising_L32",
        dimensionality="2D",
        model_type="Ising",
        beta_measured=0.128,
        beta_theoretical=0.125,
        beta_error_percent=2.4,
        beta_confidence_interval=(0.124, 0.132),
        beta_p_value=0.032,
        nu_measured=1.03,
        nu_theoretical=1.0,
        nu_error_percent=3.0,
        nu_confidence_interval=(0.98, 1.08),
        nu_p_value=0.045,
        overall_accuracy=0.875,
        universality_class_match=True,
        statistical_significance=True
    )
    
    # 3D Ising exponents
    exponent_3d = CriticalExponentData(
        system_name="3D_Ising_L16",
        dimensionality="3D",
        model_type="Ising",
        beta_measured=0.334,
        beta_theoretical=0.326,
        beta_error_percent=2.5,
        beta_confidence_interval=(0.328, 0.340),
        beta_p_value=0.028,
        nu_measured=0.642,
        nu_theoretical=0.630,
        nu_error_percent=1.9,
        nu_confidence_interval=(0.635, 0.649),
        nu_p_value=0.041,
        overall_accuracy=0.883,
        universality_class_match=True,
        statistical_significance=True
    )
    
    # Potts exponents
    exponent_potts = CriticalExponentData(
        system_name="Potts_Q3_L24",
        dimensionality="2D",
        model_type="Potts",
        beta_measured=0.115,
        beta_theoretical=0.111,
        beta_error_percent=3.6,
        beta_confidence_interval=(0.109, 0.121),
        beta_p_value=0.052,
        nu_measured=0.851,
        nu_theoretical=0.833,
        nu_error_percent=2.2,
        nu_confidence_interval=(0.842, 0.860),
        nu_p_value=0.038,
        overall_accuracy=0.821,
        universality_class_match=True,
        statistical_significance=True
    )
    
    return [exponent_2d, exponent_3d, exponent_potts]


def main():
    """Main function to demonstrate publication materials generation."""
    logger = get_logger(__name__)
    logger.info("Starting publication materials generation example")
    
    # Create output directory
    output_dir = Path("results/publication_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create mock data
        comparison_data = create_mock_comparison_data()
        exponent_data = create_mock_exponent_data()
        
        logger.info(f"Created mock data: {len(comparison_data)} systems, {len(exponent_data)} exponent datasets")
        
        # Initialize publication materials generator
        pub_generator = PublicationFigureGenerator()
        
        # Generate complete publication package
        logger.info("Generating complete publication package...")
        
        package = pub_generator.create_complete_publication_package(
            comparison_data=comparison_data,
            exponent_data=exponent_data,
            output_dir=str(output_dir)
        )
        
        # Print summary
        print("\n" + "="*60)
        print("PUBLICATION MATERIALS GENERATION COMPLETE")
        print("="*60)
        
        print(f"\nGeneration time: {package['generation_time']}")
        print(f"Output directory: {output_dir}")
        
        print(f"\nFigures generated: {len(package['figures'])}")
        main_figs = sum(1 for key in package['figures'] if 'main_figure' in key and key.endswith('_png'))
        supp_figs = sum(1 for key in package['figures'] if 'supplementary' in key and key.endswith('_png'))
        print(f"  - Main figures: {main_figs}")
        print(f"  - Supplementary figures: {supp_figs}")
        
        print(f"\nTables generated: {len(package['tables'])}")
        csv_tables = sum(1 for key in package['tables'] if key.endswith('_csv'))
        latex_tables = sum(1 for key in package['tables'] if key.endswith('_latex'))
        print(f"  - CSV tables: {csv_tables}")
        print(f"  - LaTeX tables: {latex_tables}")
        
        print(f"\nAnalysis reports: {len(package['reports'])}")
        
        print("\nKey files created:")
        print(f"  - Master checklist: {package['checklist']}")
        print(f"  - Publication metadata: {package['metadata']}")
        
        if 'captions' in package['figures']:
            print(f"  - Figure captions: {package['figures']['captions']}")
        
        if 'latex_code' in package['figures']:
            print(f"  - LaTeX figure code: {package['figures']['latex_code']}")
        
        print("\nNext steps:")
        print("  1. Review all generated figures for clarity and accuracy")
        print("  2. Verify table formatting and statistical significance")
        print("  3. Complete manuscript text using provided materials")
        print("  4. Check publication checklist for remaining tasks")
        
        print(f"\n✓ Publication materials successfully generated in {output_dir}")
        
        # Demonstrate individual components
        print("\n" + "-"*60)
        print("COMPONENT DEMONSTRATIONS")
        print("-"*60)
        
        # Systematic comparison framework
        logger.info("Demonstrating systematic comparison framework...")
        comparison_framework = SystematicComparisonFramework()
        
        comparison_results = comparison_framework.perform_comprehensive_comparison_analysis(comparison_data)
        
        print(f"\nSystematic Comparison Results:")
        print(f"  - Systems compared: {len(comparison_results.systems_compared)}")
        print(f"  - Phase separation quality scores: {len(comparison_results.phase_separation_quality)}")
        print(f"  - Order parameter accuracy: {len(comparison_results.order_parameter_accuracy)}")
        print(f"  - Critical temperature accuracy: {len(comparison_results.tc_detection_accuracy)}")
        
        # Critical exponent tables
        logger.info("Demonstrating critical exponent tables...")
        exponent_tables = CriticalExponentComparisonTables()
        
        comprehensive_table = exponent_tables.create_comprehensive_exponent_table(exponent_data)
        summary_table = exponent_tables.create_summary_statistics_table(exponent_data)
        
        print(f"\nCritical Exponent Tables:")
        print(f"  - Comprehensive table: {comprehensive_table.shape[0]} rows × {comprehensive_table.shape[1]} columns")
        print(f"  - Summary statistics: {summary_table.shape[0]} exponent types analyzed")
        
        # Statistical significance testing
        significance_results = exponent_tables.perform_statistical_significance_testing(exponent_data)
        
        print(f"\nStatistical Significance Testing:")
        for exponent, results in significance_results.items():
            if exponent != 'overall':
                p_value = results['paired_t_test']['p_value']
                correlation = results['correlation']['coefficient']
                print(f"  - {exponent.upper()} exponent: p={p_value:.4f}, r={correlation:.4f}")
        
        logger.info("Publication materials generation example completed successfully")
        
    except Exception as e:
        logger.error(f"Error in publication materials generation: {e}")
        raise


if __name__ == "__main__":
    main()