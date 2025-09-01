#!/usr/bin/env python3
"""
Visualization Generation Script for Prometheus Project

This script generates publication-ready figures and visualizations for the
VAE latent space analysis and physics discovery results.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import Visualizer, PhysicsValidator
from src.utils.config import PrometheusConfig, ConfigLoader
from src.utils.logging_utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description='Generate publication-ready visualizations')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--analysis-results', type=str, required=True, 
                       help='Path to analysis results JSON file')
    parser.add_argument('--detailed-data', type=str, required=True,
                       help='Path to detailed analysis NPZ file')
    parser.add_argument('--output-dir', type=str, help='Output directory for figures')
    parser.add_argument('--format', type=str, choices=['png', 'pdf', 'svg'], default='png',
                       help='Output figure format')
    parser.add_argument('--dpi', type=int, default=300, help='Figure DPI for raster formats')
    parser.add_argument('--style', type=str, choices=['publication', 'presentation', 'notebook'], 
                       default='publication', help='Figure style preset')
    parser.add_argument('--figures', type=str, nargs='+', 
                       choices=['all', 'latent_space', 'order_parameter', 'phase_diagram', 
                               'clustering', 'physics_validation', 'summary'],
                       default=['all'], help='Which figures to generate')
    parser.add_argument('--no-show', action='store_true', help='Do not display figures')
    
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader()
    if args.config:
        config = config_loader.load_config(args.config)
    else:
        config = PrometheusConfig()
    
    # Override configuration with command line arguments
    if args.output_dir:
        config.results_dir = args.output_dir
    
    # Setup logging
    setup_logging(config.logging)
    
    print("=" * 60)
    print("Prometheus Visualization Generation")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Analysis results: {args.analysis_results}")
    print(f"  Detailed data: {args.detailed_data}")
    print(f"  Output directory: {config.results_dir}")
    print(f"  Figure format: {args.format}")
    print(f"  Figure style: {args.style}")
    print(f"  Figures to generate: {args.figures}")
    print()
    
    # Create output directory
    output_dir = Path(config.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load analysis results
    print("Loading analysis results...")
    results_file = Path(args.analysis_results)
    if not results_file.exists():
        raise FileNotFoundError(f"Analysis results not found: {args.analysis_results}")
    
    with open(results_file, 'r') as f:
        analysis_results = json.load(f)
    
    # Load detailed data
    detailed_file = Path(args.detailed_data)
    if not detailed_file.exists():
        raise FileNotFoundError(f"Detailed data not found: {args.detailed_data}")
    
    detailed_data = np.load(detailed_file)
    latent_coords = detailed_data['latent_coords']
    temperatures = detailed_data['temperatures']
    magnetizations = detailed_data['magnetizations']
    energies = detailed_data['energies']
    cluster_labels = detailed_data['cluster_labels']
    order_parameter_values = detailed_data['order_parameter_values']
    
    print(f"  Loaded {len(latent_coords)} data points")
    print(f"  Latent space dimensions: {latent_coords.shape[1]}")
    print(f"  Temperature range: [{temperatures.min():.3f}, {temperatures.max():.3f}]")
    
    # Initialize visualizer
    visualizer = Visualizer(config, style=args.style)
    
    # Determine which figures to generate
    figures_to_generate = args.figures
    if 'all' in figures_to_generate:
        figures_to_generate = ['latent_space', 'order_parameter', 'phase_diagram', 
                              'clustering', 'physics_validation', 'summary']
    
    generated_figures = []
    
    # Generate latent space visualization
    if 'latent_space' in figures_to_generate:
        print("\nGenerating latent space visualization...")
        
        fig = visualizer.plot_latent_space(
            latent_coords=latent_coords,
            temperatures=temperatures,
            title="Latent Space Representation of Ising Model Configurations"
        )
        
        filename = f"latent_space.{args.format}"
        filepath = output_dir / filename
        fig.savefig(filepath, dpi=args.dpi, bbox_inches='tight')
        generated_figures.append(filepath)
        print(f"  Saved: {filepath}")
        
        if not args.no_show:
            plt.show()
        plt.close(fig)
    
    # Generate order parameter plot
    if 'order_parameter' in figures_to_generate:
        print("\nGenerating order parameter visualization...")
        
        fig = visualizer.plot_order_parameter_vs_temperature(
            temperatures=temperatures,
            order_parameter_values=order_parameter_values,
            magnetizations=magnetizations,
            critical_temperature=analysis_results['phase_detection']['critical_temperature'],
            theoretical_critical_temp=config.ising.critical_temp,
            title="Discovered Order Parameter vs Temperature"
        )
        
        filename = f"order_parameter.{args.format}"
        filepath = output_dir / filename
        fig.savefig(filepath, dpi=args.dpi, bbox_inches='tight')
        generated_figures.append(filepath)
        print(f"  Saved: {filepath}")
        
        if not args.no_show:
            plt.show()
        plt.close(fig)
    
    # Generate phase diagram
    if 'phase_diagram' in figures_to_generate:
        print("\nGenerating phase diagram...")
        
        fig = visualizer.plot_phase_diagram(
            latent_coords=latent_coords,
            temperatures=temperatures,
            cluster_labels=cluster_labels,
            critical_temperature=analysis_results['phase_detection']['critical_temperature'],
            title="Phase Diagram in Latent Space"
        )
        
        filename = f"phase_diagram.{args.format}"
        filepath = output_dir / filename
        fig.savefig(filepath, dpi=args.dpi, bbox_inches='tight')
        generated_figures.append(filepath)
        print(f"  Saved: {filepath}")
        
        if not args.no_show:
            plt.show()
        plt.close(fig)
    
    # Generate clustering analysis
    if 'clustering' in figures_to_generate:
        print("\nGenerating clustering analysis...")
        
        fig = visualizer.plot_clustering_analysis(
            latent_coords=latent_coords,
            temperatures=temperatures,
            cluster_labels=cluster_labels,
            silhouette_score=analysis_results['clustering']['silhouette_score'],
            title="Clustering Analysis of Phase Separation"
        )
        
        filename = f"clustering_analysis.{args.format}"
        filepath = output_dir / filename
        fig.savefig(filepath, dpi=args.dpi, bbox_inches='tight')
        generated_figures.append(filepath)
        print(f"  Saved: {filepath}")
        
        if not args.no_show:
            plt.show()
        plt.close(fig)
    
    # Generate physics validation plots
    if 'physics_validation' in figures_to_generate:
        print("\nGenerating physics validation plots...")
        
        # Initialize physics validator
        physics_validator = PhysicsValidator(config)
        
        # Perform validation
        validation_results = physics_validator.validate_physics_discovery(
            discovered_critical_temp=analysis_results['phase_detection']['critical_temperature'],
            discovered_order_param=order_parameter_values,
            theoretical_magnetization=magnetizations,
            temperatures=temperatures
        )
        
        fig = visualizer.plot_physics_validation(
            validation_results=validation_results,
            temperatures=temperatures,
            title="Physics Validation: Theory vs Discovery"
        )
        
        filename = f"physics_validation.{args.format}"
        filepath = output_dir / filename
        fig.savefig(filepath, dpi=args.dpi, bbox_inches='tight')
        generated_figures.append(filepath)
        print(f"  Saved: {filepath}")
        
        if not args.no_show:
            plt.show()
        plt.close(fig)
    
    # Generate summary figure
    if 'summary' in figures_to_generate:
        print("\nGenerating summary figure...")
        
        fig = visualizer.create_summary_figure(
            latent_coords=latent_coords,
            temperatures=temperatures,
            order_parameter_values=order_parameter_values,
            magnetizations=magnetizations,
            cluster_labels=cluster_labels,
            analysis_results=analysis_results,
            config=config
        )
        
        filename = f"summary_figure.{args.format}"
        filepath = output_dir / filename
        fig.savefig(filepath, dpi=args.dpi, bbox_inches='tight')
        generated_figures.append(filepath)
        print(f"  Saved: {filepath}")
        
        if not args.no_show:
            plt.show()
        plt.close(fig)
    
    # Generate analysis report
    print("\nGenerating analysis report...")
    
    report_content = visualizer.generate_analysis_report(
        analysis_results=analysis_results,
        validation_results=physics_validator.validate_physics_discovery(
            discovered_critical_temp=analysis_results['phase_detection']['critical_temperature'],
            discovered_order_param=order_parameter_values,
            theoretical_magnetization=magnetizations,
            temperatures=temperatures
        ) if 'physics_validation' in figures_to_generate else None,
        config=config
    )
    
    report_file = output_dir / "analysis_report.md"
    with open(report_file, 'w') as f:
        f.write(report_content)
    print(f"  Analysis report saved to: {report_file}")
    
    # Create figure index
    print("\nCreating figure index...")
    
    index_content = "# Prometheus Analysis Figures\n\n"
    index_content += f"Generated on: {np.datetime64('now')}\n\n"
    index_content += "## Generated Figures\n\n"
    
    for i, figure_path in enumerate(generated_figures, 1):
        figure_name = figure_path.stem.replace('_', ' ').title()
        index_content += f"{i}. **{figure_name}**: `{figure_path.name}`\n"
    
    index_content += f"\n## Analysis Results Summary\n\n"
    index_content += f"- **Critical Temperature**: {analysis_results['phase_detection']['critical_temperature']:.4f} "
    index_content += f"(theory: {config.ising.critical_temp:.4f})\n"
    index_content += f"- **Relative Error**: {abs(analysis_results['phase_detection']['critical_temperature'] - config.ising.critical_temp) / config.ising.critical_temp * 100:.2f}%\n"
    index_content += f"- **Order Parameter Correlation**: {analysis_results['order_parameters']['magnetization_correlation']:.4f}\n"
    index_content += f"- **Phase Separation Quality**: {analysis_results['clustering']['silhouette_score']:.4f}\n"
    index_content += f"- **Detection Confidence**: {analysis_results['phase_detection']['confidence']:.4f}\n"
    
    index_file = output_dir / "figure_index.md"
    with open(index_file, 'w') as f:
        f.write(index_content)
    print(f"  Figure index saved to: {index_file}")
    
    print("\n" + "=" * 60)
    print("Visualization Generation Complete!")
    print("=" * 60)
    print(f"Generated {len(generated_figures)} figures:")
    for figure_path in generated_figures:
        print(f"  • {figure_path.name}")
    print()
    print(f"All files saved to: {output_dir}")
    print(f"View the analysis report: {report_file}")
    print(f"Figure index: {index_file}")


if __name__ == "__main__":
    main()