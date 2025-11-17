#!/usr/bin/env python3
"""
Publication Materials Generation Script

This script demonstrates the complete publication materials generation system
for the Prometheus project, creating all figures, tables, and analysis needed
for Physical Review E submission.

Usage:
    python scripts/generate_publication_materials.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import logging
from utils.logging_utils import get_logger


def create_mock_system_data():
    """Create mock system comparison data for demonstration."""
    logger = get_logger(__name__)
    logger.info("Creating mock system comparison data")
    
    # Generate synthetic data for 2D Ising
    n_points = 1000
    temperatures_2d = np.linspace(1.5, 3.5, n_points)
    tc_2d = 2.269
    
    # Create latent space with clear phase transition
    z1_2d = np.random.normal(0, 1, n_points)
    z2_2d = np.where(temperatures_2d < tc_2d, 
                     np.random.normal(1, 0.3, n_points),
                     np.random.normal(-1, 0.3, n_points))
    
    # Add some noise and correlation
    z2_2d += 0.3 * z1_2d + np.random.normal(0, 0.1, n_points)
    
    # Magnetization with critical behavior
    magnetizations_2d = np.abs(np.where(temperatures_2d < tc_2d,
                                       (1 - temperatures_2d/tc_2d)**0.125,
                                       np.random.normal(0, 0.1, n_points)))
    
    # 3D Ising system
    temperatures_3d = np.linspace(3.5, 5.5, n_points)
    tc_3d = 4.511
    
    z1_3d = np.random.normal(0, 1, n_points)
    z2_3d = np.where(temperatures_3d < tc_3d,
                     np.random.normal(1.5, 0.4, n_points),
                     np.random.normal(-1.5, 0.4, n_points))
    
    z2_3d += 0.2 * z1_3d + np.random.normal(0, 0.15, n_points)
    
    magnetizations_3d = np.abs(np.where(temperatures_3d < tc_3d,
                                       (1 - temperatures_3d/tc_3d)**0.326,
                                       np.random.normal(0, 0.1, n_points)))
    
    return {
        '2D_Ising': {
            'latent_z1': z1_2d,
            'latent_z2': z2_2d,
            'temperatures': temperatures_2d,
            'magnetizations': magnetizations_2d,
            'discovered_tc': 2.275,
            'theoretical_tc': 2.269,
            'tc_accuracy_percent': 0.26,
            'order_parameter_correlation': 0.892
        },
        '3D_Ising': {
            'latent_z1': z1_3d,
            'latent_z2': z2_3d,
            'temperatures': temperatures_3d,
            'magnetizations': magnetizations_3d,
            'discovered_tc': 4.523,
            'theoretical_tc': 4.511,
            'tc_accuracy_percent': 0.27,
            'order_parameter_correlation': 0.876
        }
    }


def create_mock_exponent_data():
    """Create mock critical exponent data."""
    return {
        '2D_Ising': {
            'beta_measured': 0.128,
            'beta_theoretical': 0.125,
            'beta_error_percent': 2.4,
            'nu_measured': 1.03,
            'nu_theoretical': 1.0,
            'nu_error_percent': 3.0
        },
        '3D_Ising': {
            'beta_measured': 0.334,
            'beta_theoretical': 0.326,
            'beta_error_percent': 2.5,
            'nu_measured': 0.642,
            'nu_theoretical': 0.630,
            'nu_error_percent': 1.9
        }
    }


def generate_main_figure_1(system_data, exponent_data, output_dir):
    """Generate Main Figure 1: Comprehensive Results Overview."""
    logger = get_logger(__name__)
    logger.info("Generating Main Figure 1: Comprehensive Results Overview")
    
    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.0,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Prometheus: Unsupervised Discovery of Critical Phenomena', fontsize=16, fontweight='bold')
    
    # Panel A: 2D Ising latent space
    ax_a = axes[0, 0]
    data_2d = system_data['2D_Ising']
    
    scatter = ax_a.scatter(
        data_2d['latent_z1'], data_2d['latent_z2'],
        c=data_2d['temperatures'], cmap='coolwarm',
        alpha=0.7, s=25, edgecolors='black', linewidth=0.3
    )
    
    cbar = plt.colorbar(scatter, ax=ax_a, shrink=0.8)
    cbar.set_label('Temperature', fontsize=10)
    
    ax_a.set_xlabel('z₁', fontsize=12)
    ax_a.set_ylabel('z₂', fontsize=12)
    ax_a.set_title('(a) 2D Ising Latent Space', fontsize=14, fontweight='bold')
    ax_a.grid(True, alpha=0.3)
    ax_a.set_aspect('equal', adjustable='box')
    
    # Add critical temperature annotation
    ax_a.text(0.05, 0.95, f'Tc = {data_2d["discovered_tc"]:.3f}',
             transform=ax_a.transAxes, fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel B: 3D Ising latent space
    ax_b = axes[0, 1]
    data_3d = system_data['3D_Ising']
    
    scatter = ax_b.scatter(
        data_3d['latent_z1'], data_3d['latent_z2'],
        c=data_3d['temperatures'], cmap='coolwarm',
        alpha=0.7, s=25, edgecolors='black', linewidth=0.3
    )
    
    cbar = plt.colorbar(scatter, ax=ax_b, shrink=0.8)
    cbar.set_label('Temperature', fontsize=10)
    
    ax_b.set_xlabel('z₁', fontsize=12)
    ax_b.set_ylabel('z₂', fontsize=12)
    ax_b.set_title('(b) 3D Ising Latent Space', fontsize=14, fontweight='bold')
    ax_b.grid(True, alpha=0.3)
    ax_b.set_aspect('equal', adjustable='box')
    
    ax_b.text(0.05, 0.95, f'Tc = {data_3d["discovered_tc"]:.3f}',
             transform=ax_b.transAxes, fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel C: Order parameter correlations
    ax_c = axes[0, 2]
    
    systems = ['2D Ising', '3D Ising']
    correlations = [data_2d['order_parameter_correlation'], data_3d['order_parameter_correlation']]
    colors = ['#1f77b4', '#ff7f0e']
    
    bars = ax_c.bar(systems, correlations, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax_c.axhline(y=0.8, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Target (0.8)')
    
    ax_c.set_ylabel('Order Parameter Correlation', fontsize=12)
    ax_c.set_title('(c) Order Parameter Discovery', fontsize=14, fontweight='bold')
    ax_c.legend(fontsize=10)
    ax_c.grid(True, alpha=0.3)
    ax_c.set_ylim(0, 1.0)
    
    # Add value annotations
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        ax_c.annotate(f'{corr:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Panel D: Critical temperature accuracy
    ax_d = axes[1, 0]
    
    tc_errors = [abs(data_2d['tc_accuracy_percent']), abs(data_3d['tc_accuracy_percent'])]
    
    bars = ax_d.bar(systems, tc_errors, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax_d.axhline(y=5.0, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='5% Error')
    
    ax_d.set_ylabel('Critical Temperature Error (%)', fontsize=12)
    ax_d.set_title('(d) Critical Temperature Accuracy', fontsize=14, fontweight='bold')
    ax_d.legend(fontsize=10)
    ax_d.grid(True, alpha=0.3)
    
    # Add value annotations
    for bar, error in zip(bars, tc_errors):
        height = bar.get_height()
        ax_d.annotate(f'{error:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Panel E: Critical exponent comparison
    ax_e = axes[1, 1]
    
    beta_measured = [exponent_data['2D_Ising']['beta_measured'], exponent_data['3D_Ising']['beta_measured']]
    beta_theoretical = [exponent_data['2D_Ising']['beta_theoretical'], exponent_data['3D_Ising']['beta_theoretical']]
    
    # Perfect correlation line
    all_values = beta_measured + beta_theoretical
    min_val, max_val = min(all_values), max(all_values)
    ax_e.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.7, label='Perfect Agreement')
    
    # Scatter plot
    ax_e.scatter(beta_theoretical, beta_measured, s=100, alpha=0.8, c=colors, 
                edgecolors='black', linewidth=1.5, label='β exponent')
    
    ax_e.set_xlabel('Theoretical Value', fontsize=12)
    ax_e.set_ylabel('Measured Value', fontsize=12)
    ax_e.set_title('(e) β Exponent Validation', fontsize=14, fontweight='bold')
    ax_e.legend(fontsize=10)
    ax_e.grid(True, alpha=0.3)
    ax_e.set_aspect('equal', adjustable='box')
    
    # Panel F: Overall accuracy summary
    ax_f = axes[1, 2]
    
    # Calculate overall scores
    overall_scores = []
    for system in ['2D_Ising', '3D_Ising']:
        corr_score = system_data[system]['order_parameter_correlation']
        tc_score = 1.0 - abs(system_data[system]['tc_accuracy_percent']) / 100.0
        exp_score = 1.0 - exponent_data[system]['beta_error_percent'] / 100.0
        overall = (corr_score + tc_score + exp_score) / 3
        overall_scores.append(overall)
    
    bars = ax_f.bar(systems, overall_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax_f.axhline(y=0.8, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Target (80%)')
    
    ax_f.set_ylabel('Overall Performance Score', fontsize=12)
    ax_f.set_title('(f) Overall Performance', fontsize=14, fontweight='bold')
    ax_f.legend(fontsize=10)
    ax_f.grid(True, alpha=0.3)
    ax_f.set_ylim(0, 1.0)
    
    # Add value annotations
    for bar, score in zip(bars, overall_scores):
        height = bar.get_height()
        ax_f.annotate(f'{score:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    png_path = output_path / "main_figure_1_comprehensive_results.png"
    pdf_path = output_path / "main_figure_1_comprehensive_results.pdf"
    
    fig.savefig(png_path, dpi=300, format='png', bbox_inches='tight')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    plt.close(fig)
    
    logger.info(f"Main Figure 1 saved: {png_path}")
    return str(png_path)


def generate_critical_exponent_table(exponent_data, output_dir):
    """Generate critical exponent comparison table."""
    logger = get_logger(__name__)
    logger.info("Generating critical exponent comparison table")
    
    # Create comprehensive table
    table_data = []
    
    for system, data in exponent_data.items():
        row = {
            'System': system,
            'β (Theoretical)': f"{data['beta_theoretical']:.4f}",
            'β (Measured)': f"{data['beta_measured']:.4f}",
            'β Error (%)': f"{data['beta_error_percent']:.1f}",
            'ν (Theoretical)': f"{data['nu_theoretical']:.4f}",
            'ν (Measured)': f"{data['nu_measured']:.4f}",
            'ν Error (%)': f"{data['nu_error_percent']:.1f}",
            'Overall Accuracy': f"{1.0 - (data['beta_error_percent'] + data['nu_error_percent']) / 200.0:.3f}"
        }
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Save CSV
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_path / "critical_exponent_comparison_table.csv"
    df.to_csv(csv_path, index=False)
    
    # Create LaTeX table
    latex_content = """\\begin{table}[htbp]
\\centering
\\caption{Critical Exponent Comparison: Theoretical vs Measured Values}
\\label{tab:critical_exponents}
\\begin{tabular}{|l|c|c|c|c|c|c|c|}
\\hline
System & $\\beta_{th}$ & $\\beta_{meas}$ & $\\beta$ Error (\\%) & $\\nu_{th}$ & $\\nu_{meas}$ & $\\nu$ Error (\\%) & Accuracy \\\\
\\hline
"""
    
    for _, row in df.iterrows():
        latex_content += f"{row['System']} & {row['β (Theoretical)']} & {row['β (Measured)']} & {row['β Error (%)']} & {row['ν (Theoretical)']} & {row['ν (Measured)']} & {row['ν Error (%)']} & {row['Overall Accuracy']} \\\\\n"
    
    latex_content += """\\hline
\\end{tabular}
\\end{table}
"""
    
    # Save LaTeX
    latex_path = output_path / "critical_exponent_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    
    logger.info(f"Critical exponent table saved: {csv_path}")
    return str(csv_path), str(latex_path)


def generate_publication_checklist(output_dir):
    """Generate publication checklist."""
    checklist_content = """# Publication Materials Checklist

Generated on: {timestamp}

## Figures
- [x] Main Figure 1: Comprehensive Results Overview
- [ ] Main Figure 2: Phase Separation Analysis  
- [ ] Main Figure 3: Critical Exponent Validation
- [ ] Supplementary figures as needed

## Tables
- [x] Critical Exponent Comparison Table (CSV and LaTeX)
- [ ] Summary Statistics Table
- [ ] Universality Class Validation Table

## Analysis Reports
- [x] Mock data generation and validation
- [ ] Statistical significance testing
- [ ] Cross-system validation

## Publication Materials
- [x] Figure captions
- [x] LaTeX table formatting
- [ ] Manuscript text
- [ ] References

## Quality Assurance
- [ ] All results independently verified
- [ ] Code and data availability statements
- [ ] Reproducibility documentation
- [ ] Ethical considerations

## Submission Preparation
- [ ] Journal formatting requirements checked
- [ ] Word count within limits
- [ ] Figure and table limits respected
- [ ] Final proofreading completed

## Next Steps
1. Review generated figure for clarity and accuracy
2. Verify table formatting and values
3. Complete remaining figures and analysis
4. Write manuscript text using provided materials
""".format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    output_path = Path(output_dir)
    checklist_path = output_path / "publication_checklist.md"
    
    with open(checklist_path, 'w') as f:
        f.write(checklist_content)
    
    return str(checklist_path)


def main():
    """Main function to generate publication materials."""
    logger = get_logger(__name__)
    logger.info("Starting publication materials generation")
    
    # Create output directory
    output_dir = Path("results/publication_materials_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create mock data
        logger.info("Creating mock data for demonstration")
        system_data = create_mock_system_data()
        exponent_data = create_mock_exponent_data()
        
        # Generate main figure
        logger.info("Generating main figure")
        main_fig_path = generate_main_figure_1(system_data, exponent_data, output_dir / "figures")
        
        # Generate critical exponent table
        logger.info("Generating critical exponent table")
        csv_path, latex_path = generate_critical_exponent_table(exponent_data, output_dir / "tables")
        
        # Generate publication checklist
        logger.info("Generating publication checklist")
        checklist_path = generate_publication_checklist(output_dir)
        
        # Create summary report
        summary = {
            'generation_time': datetime.now().isoformat(),
            'output_directory': str(output_dir),
            'files_generated': {
                'main_figure_png': main_fig_path,
                'main_figure_pdf': main_fig_path.replace('.png', '.pdf'),
                'exponent_table_csv': csv_path,
                'exponent_table_latex': latex_path,
                'publication_checklist': checklist_path
            },
            'system_analysis': {
                '2D_Ising': {
                    'tc_accuracy': f"{system_data['2D_Ising']['tc_accuracy_percent']:.2f}%",
                    'order_parameter_correlation': f"{system_data['2D_Ising']['order_parameter_correlation']:.3f}",
                    'beta_exponent_error': f"{exponent_data['2D_Ising']['beta_error_percent']:.1f}%"
                },
                '3D_Ising': {
                    'tc_accuracy': f"{system_data['3D_Ising']['tc_accuracy_percent']:.2f}%",
                    'order_parameter_correlation': f"{system_data['3D_Ising']['order_parameter_correlation']:.3f}",
                    'beta_exponent_error': f"{exponent_data['3D_Ising']['beta_error_percent']:.1f}%"
                }
            }
        }
        
        # Save summary
        summary_path = output_dir / "generation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print results
        print("\n" + "="*60)
        print("PUBLICATION MATERIALS GENERATION COMPLETE")
        print("="*60)
        
        print(f"\nOutput directory: {output_dir}")
        print(f"Generation time: {summary['generation_time']}")
        
        print(f"\nFiles generated:")
        for name, path in summary['files_generated'].items():
            print(f"  - {name}: {path}")
        
        print(f"\nSystem Analysis Summary:")
        for system, metrics in summary['system_analysis'].items():
            print(f"  {system}:")
            print(f"    - Critical temperature accuracy: {metrics['tc_accuracy']}")
            print(f"    - Order parameter correlation: {metrics['order_parameter_correlation']}")
            print(f"    - β exponent error: {metrics['beta_exponent_error']}")
        
        print(f"\nNext steps:")
        print(f"  1. Review generated figure: {main_fig_path}")
        print(f"  2. Check table formatting: {csv_path}")
        print(f"  3. Follow publication checklist: {checklist_path}")
        
        print(f"\n✓ Publication materials successfully generated!")
        
        logger.info("Publication materials generation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in publication materials generation: {e}")
        raise


if __name__ == "__main__":
    main()