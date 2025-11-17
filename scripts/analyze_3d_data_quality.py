#!/usr/bin/env python3
"""
Script to analyze 3D Ising dataset quality and create validation reports.

This script implements task 3.2 from the PRE paper specification:
- Compute magnetization curves and validate transition behavior around Tc ≈ 4.511
- Implement data quality checks for proper equilibration and sampling
- Create visualization tools for 2D slices of 3D configurations
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.data_quality_3d import analyze_3d_dataset_quality
from data.data_generator_3d import load_3d_dataset
import logging


def save_quality_report(report, output_path: str):
    """Save quality report to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    report_dict = {
        'dataset_summary': report.dataset_summary,
        'overall_quality_score': float(report.overall_quality_score),
        'validation_passed': report.validation_passed,
        'issues_found': report.issues_found,
        'recommendations': report.recommendations,
        'equilibration_analysis': report.equilibration_analysis,
        'system_size_reports': {}
    }
    
    # Convert system size reports
    for size, size_report in report.system_size_reports.items():
        report_dict['system_size_reports'][str(size)] = {
            'system_size': size_report['system_size'],
            'lattice_shape': size_report['lattice_shape'],
            'n_sites': size_report['n_sites'],
            'n_temperatures': size_report['n_temperatures'],
            'n_configurations': size_report['n_configurations'],
            'generation_time_minutes': size_report['generation_time_minutes'],
            'configuration_validation': size_report['configuration_validation'],
            'equilibration_analysis': size_report['equilibration_analysis']
        }
    
    # Convert magnetization analysis
    report_dict['magnetization_analysis'] = {}
    for size, mag_result in report.magnetization_analysis.items():
        report_dict['magnetization_analysis'][str(size)] = {
            'system_size': mag_result.system_size,
            'tc_estimate': float(mag_result.tc_estimate),
            'tc_confidence_interval': [float(mag_result.tc_confidence_interval[0]), 
                                     float(mag_result.tc_confidence_interval[1])],
            'transition_sharpness': float(mag_result.transition_sharpness),
            'fit_quality': float(mag_result.fit_quality),
            'metadata': mag_result.metadata
        }
    
    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2)


def print_quality_summary(report):
    """Print a summary of the quality analysis."""
    print("\n" + "="*60)
    print("3D ISING DATASET QUALITY ANALYSIS SUMMARY")
    print("="*60)
    
    # Overall status
    status = "PASSED" if report.validation_passed else "FAILED"
    print(f"Overall Validation Status: {status}")
    print(f"Quality Score: {report.overall_quality_score:.3f}/1.0")
    
    # Dataset summary
    summary = report.dataset_summary
    print(f"\nDataset Summary:")
    print(f"  Total configurations: {summary['total_configurations']:,}")
    print(f"  System sizes: {summary['system_sizes']}")
    print(f"  Temperature range: {summary['temperature_range']}")
    print(f"  Generation time: {summary['generation_time_hours']:.2f} hours")
    
    # System size analysis
    print(f"\nSystem Size Analysis:")
    for size in summary['system_sizes']:
        size_report = report.system_size_reports[size]
        config_val = size_report['configuration_validation']
        eq_analysis = size_report['equilibration_analysis']
        
        print(f"  L={size}:")
        print(f"    Configurations: {size_report['n_configurations']:,}")
        print(f"    Config validation: {'PASS' if config_val['is_valid'] else 'FAIL'}")
        print(f"    Equilibration success: {eq_analysis['success_rate']:.1%}")
        
        # Magnetization analysis
        if size in report.magnetization_analysis:
            mag_result = report.magnetization_analysis[size]
            tc_error = abs(mag_result.tc_estimate - 4.511) / 4.511 * 100
            print(f"    Tc estimate: {mag_result.tc_estimate:.3f} (error: {tc_error:.1f}%)")
            print(f"    Transition sharpness: {mag_result.transition_sharpness:.3f}")
    
    # Issues and recommendations
    if report.issues_found:
        print(f"\nIssues Found ({len(report.issues_found)}):")
        for issue in report.issues_found:
            print(f"  - {issue}")
    
    if report.recommendations:
        print(f"\nRecommendations ({len(report.recommendations)}):")
        for rec in report.recommendations:
            print(f"  - {rec}")
    
    # Equilibration summary
    eq_analysis = report.equilibration_analysis
    print(f"\nEquilibration Analysis:")
    print(f"  Overall success rate: {eq_analysis['overall_success_rate']:.1%}")
    print(f"  Average quality score: {eq_analysis['average_quality_score']:.3f}")
    
    print("\n" + "="*60)


def main():
    """Main function to analyze 3D dataset quality."""
    parser = argparse.ArgumentParser(description='Analyze 3D Ising dataset quality')
    
    # Input options
    parser.add_argument('dataset_path', type=str,
                       help='Path to 3D dataset file (.h5 or .npz)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='results/data_quality',
                       help='Output directory for analysis results (default: results/data_quality)')
    parser.add_argument('--report-file', type=str, default=None,
                       help='JSON file to save quality report (default: auto-generated)')
    
    # Visualization options
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip creating 2D slice visualizations')
    parser.add_argument('--viz-temperatures', type=int, nargs='+', default=None,
                       help='Temperature indices to visualize (default: around Tc)')
    parser.add_argument('--viz-configs', type=int, nargs='+', default=[0, 1, 2],
                       help='Configuration indices to visualize (default: 0 1 2)')
    
    # Analysis options
    parser.add_argument('--theoretical-tc', type=float, default=4.511,
                       help='Theoretical critical temperature (default: 4.511)')
    
    # Logging options
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level (default: INFO)')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Log file path (default: console only)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=getattr(logging, args.log_level),
                       filename=args.log_file if args.log_file else None)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting 3D dataset quality analysis")
    logger.info(f"Dataset path: {args.dataset_path}")
    
    # Check if dataset file exists
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset file not found: {dataset_path}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Load dataset
        logger.info("Loading 3D dataset...")
        dataset = load_3d_dataset(str(dataset_path))
        logger.info(f"Loaded dataset with {dataset.total_configurations:,} configurations")
        
        # Perform quality analysis
        logger.info("Performing quality analysis...")
        report = analyze_3d_dataset_quality(
            dataset=dataset,
            create_visualizations=not args.no_visualizations,
            output_dir=str(output_dir) if not args.no_visualizations else None
        )
        
        # Save quality report
        if args.report_file:
            report_path = args.report_file
        else:
            timestamp = dataset_path.stem
            report_path = output_dir / f"quality_report_{timestamp}.json"
        
        save_quality_report(report, str(report_path))
        logger.info(f"Quality report saved to: {report_path}")
        
        # Print summary
        print_quality_summary(report)
        
        # Return appropriate exit code
        return 0 if report.validation_passed else 1
        
    except Exception as e:
        logger.error(f"Quality analysis failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)