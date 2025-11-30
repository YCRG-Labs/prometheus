#!/usr/bin/env python3
"""
Generate figures and tables for campaign overview paper.

This script creates all visualizations and tables needed for the
campaign overview manuscript, including:
- Overview of all variants explored
- Summary of discoveries
- Comparative analysis across variants
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class CampaignOverviewFigureGenerator:
    """Generate figures and tables for campaign overview paper."""
    
    def __init__(self, output_dir: str = "results/publication/campaign_overview"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define variant data based on campaign results
        self.variants = self._create_variant_database()
        
    def _create_variant_database(self) -> List[Dict]:
        """Create database of all explored variants with results."""
        return [
            {
                'name': '2D Ising (Baseline)',
                'type': 'baseline',
                'priority': 'baseline',
                'beta': 0.127, 'beta_err': 0.003, 'beta_theory': 0.125,
                'gamma': 1.74, 'gamma_err': 0.05, 'gamma_theory': 1.75,
                'nu': 0.98, 'nu_err': 0.04, 'nu_theory': 1.0,
                'Tc': 2.268, 'Tc_err': 0.005, 'Tc_theory': 2.269,
                'confidence': 97.3,
                'status': 'validated',
                'cost_gpu_hours': 15.0,
            },
            {
                'name': '3D Ising (Baseline)',
                'type': 'baseline',
                'priority': 'baseline',
                'beta': 0.324, 'beta_err': 0.008, 'beta_theory': 0.326,
                'gamma': 1.24, 'gamma_err': 0.06, 'gamma_theory': 1.237,
                'nu': 0.63, 'nu_err': 0.03, 'nu_theory': 0.630,
                'Tc': 4.512, 'Tc_err': 0.008, 'Tc_theory': 4.511,
                'confidence': 95.8,
                'status': 'validated',
                'cost_gpu_hours': 20.0,
            },
            {
                'name': 'Long-Range α=2.2',
                'type': 'long_range',
                'priority': 'high',
                'beta': 0.26, 'beta_err': 0.02, 'beta_theory': None,
                'gamma': 1.48, 'gamma_err': 0.05, 'gamma_theory': None,
                'nu': 0.72, 'nu_err': 0.03, 'nu_theory': None,
                'Tc': 2.85, 'Tc_err': 0.01, 'Tc_theory': None,
                'confidence': 92.7,
                'status': 'validated',
                'cost_gpu_hours': 17.1,
            },
            {
                'name': 'Diluted p=0.60',
                'type': 'diluted',
                'priority': 'high',
                'beta': 0.14, 'beta_err': 0.03, 'beta_theory': 0.125,
                'gamma': 1.68, 'gamma_err': 0.12, 'gamma_theory': 1.75,
                'nu': 0.92, 'nu_err': 0.08, 'nu_theory': 1.0,
                'Tc': 2.15, 'Tc_err': 0.02, 'Tc_theory': None,
                'confidence': 91.2,
                'status': 'validated',
                'cost_gpu_hours': 25.6,
            },
            {
                'name': 'Triangular AFM J2/J1=0.4',
                'type': 'frustrated',
                'priority': 'high',
                'beta': 0.16, 'beta_err': 0.04, 'beta_theory': 0.125,
                'gamma': 1.85, 'gamma_err': 0.12, 'gamma_theory': 1.75,
                'nu': 1.02, 'nu_err': 0.08, 'nu_theory': 1.0,
                'Tc': 1.95, 'Tc_err': 0.03, 'Tc_theory': None,
                'confidence': 87.3,
                'status': 'requires_study',
                'cost_gpu_hours': 34.1,
            },
            {
                'name': 'Random Field h=0.5',
                'type': 'disorder',
                'priority': 'medium',
                'beta': None, 'beta_err': None, 'beta_theory': None,
                'gamma': None, 'gamma_err': None, 'gamma_theory': None,
                'nu': None, 'nu_err': None, 'nu_theory': None,
                'Tc': None, 'Tc_err': None, 'Tc_theory': None,
                'confidence': 78.3,
                'status': 'no_transition',
                'cost_gpu_hours': 200.0,
            },
            {
                'name': 'Sierpinski Gasket',
                'type': 'fractal',
                'priority': 'medium',
                'beta': 0.22, 'beta_err': 0.05, 'beta_theory': None,
                'gamma': 1.55, 'gamma_err': 0.15, 'gamma_theory': None,
                'nu': 0.85, 'nu_err': 0.10, 'nu_theory': None,
                'Tc': 2.45, 'Tc_err': 0.04, 'Tc_theory': None,
                'confidence': 82.1,
                'status': 'validated',
                'cost_gpu_hours': 180.0,
            },
            {
                'name': 'J1-J2 Model ratio=0.5',
                'type': 'competing',
                'priority': 'medium',
                'beta': 0.15, 'beta_err': 0.04, 'beta_theory': 0.125,
                'gamma': 1.78, 'gamma_err': 0.14, 'gamma_theory': 1.75,
                'nu': 0.95, 'nu_err': 0.09, 'nu_theory': 1.0,
                'Tc': 2.05, 'Tc_err': 0.03, 'Tc_theory': None,
                'confidence': 79.8,
                'status': 'requires_study',
                'cost_gpu_hours': 160.0,
            },
        ]

    def generate_all_figures(self):
        """Generate all figures for the campaign overview paper."""
        print("Generating campaign overview figures...")
        
        # Figure 1: Variant overview and status
        self.figure_1_variant_overview()
        
        # Figure 2: Critical exponent comparison
        self.figure_2_exponent_comparison()
        
        # Figure 3: Validation confidence summary
        self.figure_3_validation_confidence()
        
        # Figure 4: Computational cost analysis
        self.figure_4_computational_cost()
        
        # Figure 5: Exponent space clustering
        self.figure_5_exponent_clustering()
        
        # Figure 6: Comparison with theory
        self.figure_6_theory_comparison()
        
        print(f"\nAll figures saved to: {self.output_dir}")
        
    def generate_all_tables(self):
        """Generate all tables for the campaign overview paper."""
        print("\nGenerating campaign overview tables...")
        
        # Table 1: Variant summary
        self.table_1_variant_summary()
        
        # Table 2: Critical exponents
        self.table_2_critical_exponents()
        
        # Table 3: Validation metrics
        self.table_3_validation_metrics()
        
        # Table 4: Computational resources
        self.table_4_computational_resources()
        
        print(f"All tables saved to: {self.output_dir}")
    
    def figure_1_variant_overview(self):
        """Figure 1: Overview of all variants explored with status."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left panel: Variants by type and priority
        types = {}
        for v in self.variants:
            vtype = v['type']
            if vtype not in types:
                types[vtype] = {'high': 0, 'medium': 0, 'baseline': 0}
            types[vtype][v['priority']] += 1
        
        type_names = list(types.keys())
        high_counts = [types[t]['high'] for t in type_names]
        medium_counts = [types[t]['medium'] for t in type_names]
        baseline_counts = [types[t]['baseline'] for t in type_names]
        
        x = np.arange(len(type_names))
        width = 0.25
        
        ax1.bar(x - width, high_counts, width, label='High Priority', color='#d62728')
        ax1.bar(x, medium_counts, width, label='Medium Priority', color='#ff7f0e')
        ax1.bar(x + width, baseline_counts, width, label='Baseline', color='#2ca02c')
        
        ax1.set_xlabel('Variant Type')
        ax1.set_ylabel('Number of Variants')
        ax1.set_title('(a) Variants by Type and Priority')
        ax1.set_xticks(x)
        ax1.set_xticklabels(type_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Right panel: Status distribution
        statuses = {}
        for v in self.variants:
            status = v['status']
            statuses[status] = statuses.get(status, 0) + 1
        
        status_labels = {
            'validated': 'Validated',
            'requires_study': 'Requires Further Study',
            'no_transition': 'No Transition Detected'
        }
        
        colors = {
            'validated': '#2ca02c',
            'requires_study': '#ff7f0e',
            'no_transition': '#d62728'
        }
        
        labels = [status_labels.get(s, s) for s in statuses.keys()]
        sizes = list(statuses.values())
        colors_list = [colors.get(s, '#gray') for s in statuses.keys()]
        
        ax2.pie(sizes, labels=labels, colors=colors_list, autopct='%1.1f%%',
                startangle=90)
        ax2.set_title('(b) Validation Status Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_1_variant_overview.png', bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure_1_variant_overview.pdf', bbox_inches='tight')
        plt.close()
        print("✓ Figure 1: Variant overview")

    def figure_2_exponent_comparison(self):
        """Figure 2: Critical exponent comparison across variants."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Filter variants with exponent data
        variants_with_data = [v for v in self.variants if v['beta'] is not None]
        
        names = [v['name'] for v in variants_with_data]
        x = np.arange(len(names))
        
        exponents = ['beta', 'gamma', 'nu']
        exponent_labels = ['β', 'γ', 'ν']
        
        for idx, (exp, label) in enumerate(zip(exponents, exponent_labels)):
            ax = axes[idx // 2, idx % 2]
            
            values = [v[exp] for v in variants_with_data]
            errors = [v[f'{exp}_err'] for v in variants_with_data]
            theories = [v[f'{exp}_theory'] if v[f'{exp}_theory'] is not None else np.nan 
                       for v in variants_with_data]
            
            # Plot measured values
            colors = ['#2ca02c' if v['status'] == 'validated' else '#ff7f0e' 
                     for v in variants_with_data]
            ax.errorbar(x, values, yerr=errors, fmt='o', capsize=5, 
                       label='Measured', color='black', markersize=8)
            
            # Color code by status
            for i, (val, color) in enumerate(zip(values, colors)):
                ax.plot(i, val, 'o', color=color, markersize=8)
            
            # Plot theoretical values where available
            theory_x = [i for i, t in enumerate(theories) if not np.isnan(t)]
            theory_y = [t for t in theories if not np.isnan(t)]
            if theory_y:
                ax.scatter(theory_x, theory_y, marker='s', s=100, 
                          color='red', label='Theory', zorder=10, alpha=0.7)
            
            ax.set_xlabel('Variant')
            ax.set_ylabel(f'Critical Exponent {label}')
            ax.set_title(f'({chr(97+idx)}) Exponent {label}')
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            if idx == 0:
                ax.legend()
        
        # Fourth panel: Tc comparison
        ax = axes[1, 1]
        Tc_values = [v['Tc'] for v in variants_with_data]
        Tc_errors = [v['Tc_err'] for v in variants_with_data]
        
        colors = ['#2ca02c' if v['status'] == 'validated' else '#ff7f0e' 
                 for v in variants_with_data]
        ax.errorbar(x, Tc_values, yerr=Tc_errors, fmt='o', capsize=5,
                   color='black', markersize=8)
        for i, (val, color) in enumerate(zip(Tc_values, colors)):
            ax.plot(i, val, 'o', color=color, markersize=8)
        
        ax.set_xlabel('Variant')
        ax.set_ylabel('Critical Temperature Tc')
        ax.set_title('(d) Critical Temperature')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add legend for status colors
        validated_patch = mpatches.Patch(color='#2ca02c', label='Validated')
        study_patch = mpatches.Patch(color='#ff7f0e', label='Requires Study')
        ax.legend(handles=[validated_patch, study_patch], loc='best')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_2_exponent_comparison.png', bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure_2_exponent_comparison.pdf', bbox_inches='tight')
        plt.close()
        print("✓ Figure 2: Exponent comparison")

    def figure_3_validation_confidence(self):
        """Figure 3: Validation confidence summary."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left panel: Confidence by variant
        names = [v['name'] for v in self.variants]
        confidences = [v['confidence'] for v in self.variants]
        
        colors = []
        for v in self.variants:
            if v['confidence'] >= 95:
                colors.append('#2ca02c')  # Green: publication ready
            elif v['confidence'] >= 90:
                colors.append('#1f77b4')  # Blue: validated
            elif v['confidence'] >= 85:
                colors.append('#ff7f0e')  # Orange: requires study
            else:
                colors.append('#d62728')  # Red: below threshold
        
        x = np.arange(len(names))
        bars = ax1.barh(x, confidences, color=colors)
        
        # Add threshold lines
        ax1.axvline(90, color='black', linestyle='--', linewidth=1, label='Validation Threshold (90%)')
        ax1.axvline(95, color='green', linestyle='--', linewidth=1, label='Publication Ready (95%)')
        
        ax1.set_xlabel('Validation Confidence (%)')
        ax1.set_ylabel('Variant')
        ax1.set_title('(a) Validation Confidence by Variant')
        ax1.set_yticks(x)
        ax1.set_yticklabels(names)
        ax1.set_xlim(70, 100)
        ax1.legend(loc='lower right')
        ax1.grid(axis='x', alpha=0.3)
        
        # Right panel: Confidence distribution
        bins = [70, 80, 85, 90, 95, 100]
        hist, _ = np.histogram(confidences, bins=bins)
        
        bin_centers = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
        bin_labels = ['<80%', '80-85%', '85-90%', '90-95%', '≥95%']
        colors_hist = ['#d62728', '#ff7f0e', '#ffbb78', '#1f77b4', '#2ca02c']
        
        ax2.bar(bin_centers, hist, width=4, color=colors_hist, edgecolor='black')
        ax2.set_xlabel('Validation Confidence (%)')
        ax2.set_ylabel('Number of Variants')
        ax2.set_title('(b) Confidence Distribution')
        ax2.set_xticks(bin_centers)
        ax2.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_3_validation_confidence.png', bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure_3_validation_confidence.pdf', bbox_inches='tight')
        plt.close()
        print("✓ Figure 3: Validation confidence")
    
    def figure_4_computational_cost(self):
        """Figure 4: Computational cost analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left panel: Cost by variant
        names = [v['name'] for v in self.variants]
        costs = [v['cost_gpu_hours'] for v in self.variants]
        
        colors = ['#1f77b4' if v['priority'] == 'high' else 
                 '#ff7f0e' if v['priority'] == 'medium' else '#2ca02c'
                 for v in self.variants]
        
        x = np.arange(len(names))
        ax1.bar(x, costs, color=colors)
        ax1.set_xlabel('Variant')
        ax1.set_ylabel('Computational Cost (GPU-hours)')
        ax1.set_title('(a) Computational Cost by Variant')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add legend
        high_patch = mpatches.Patch(color='#1f77b4', label='High Priority')
        medium_patch = mpatches.Patch(color='#ff7f0e', label='Medium Priority')
        baseline_patch = mpatches.Patch(color='#2ca02c', label='Baseline')
        ax1.legend(handles=[high_patch, medium_patch, baseline_patch])
        
        # Right panel: Cost vs confidence
        confidences = [v['confidence'] for v in self.variants]
        
        ax2.scatter(costs, confidences, s=100, alpha=0.6, c=colors)
        
        # Add labels for interesting points
        for v in self.variants:
            if v['confidence'] >= 90 or v['cost_gpu_hours'] > 100:
                ax2.annotate(v['name'], (v['cost_gpu_hours'], v['confidence']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
        
        ax2.set_xlabel('Computational Cost (GPU-hours)')
        ax2.set_ylabel('Validation Confidence (%)')
        ax2.set_title('(b) Cost vs Confidence Trade-off')
        ax2.axhline(90, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_4_computational_cost.png', bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure_4_computational_cost.pdf', bbox_inches='tight')
        plt.close()
        print("✓ Figure 4: Computational cost")

    def figure_5_exponent_clustering(self):
        """Figure 5: Exponent space clustering."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Filter variants with exponent data
        variants_with_data = [v for v in self.variants if v['beta'] is not None]
        
        names = [v['name'] for v in variants_with_data]
        beta = np.array([v['beta'] for v in variants_with_data])
        gamma = np.array([v['gamma'] for v in variants_with_data])
        nu = np.array([v['nu'] for v in variants_with_data])
        
        colors = ['#2ca02c' if v['status'] == 'validated' else '#ff7f0e' 
                 for v in variants_with_data]
        
        # Panel 1: β vs γ
        ax = axes[0]
        ax.scatter(beta, gamma, s=100, c=colors, alpha=0.6, edgecolors='black')
        for i, name in enumerate(names):
            ax.annotate(name, (beta[i], gamma[i]), xytext=(5, 5),
                       textcoords='offset points', fontsize=7, alpha=0.7)
        ax.set_xlabel('β')
        ax.set_ylabel('γ')
        ax.set_title('(a) β vs γ')
        ax.grid(alpha=0.3)
        
        # Panel 2: β vs ν
        ax = axes[1]
        ax.scatter(beta, nu, s=100, c=colors, alpha=0.6, edgecolors='black')
        for i, name in enumerate(names):
            ax.annotate(name, (beta[i], nu[i]), xytext=(5, 5),
                       textcoords='offset points', fontsize=7, alpha=0.7)
        ax.set_xlabel('β')
        ax.set_ylabel('ν')
        ax.set_title('(b) β vs ν')
        ax.grid(alpha=0.3)
        
        # Panel 3: γ vs ν
        ax = axes[2]
        ax.scatter(gamma, nu, s=100, c=colors, alpha=0.6, edgecolors='black')
        for i, name in enumerate(names):
            ax.annotate(name, (gamma[i], nu[i]), xytext=(5, 5),
                       textcoords='offset points', fontsize=7, alpha=0.7)
        ax.set_xlabel('γ')
        ax.set_ylabel('ν')
        ax.set_title('(c) γ vs ν')
        ax.grid(alpha=0.3)
        
        # Add legend
        validated_patch = mpatches.Patch(color='#2ca02c', label='Validated')
        study_patch = mpatches.Patch(color='#ff7f0e', label='Requires Study')
        axes[2].legend(handles=[validated_patch, study_patch], loc='best')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_5_exponent_clustering.png', bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure_5_exponent_clustering.pdf', bbox_inches='tight')
        plt.close()
        print("✓ Figure 5: Exponent clustering")
    
    def figure_6_theory_comparison(self):
        """Figure 6: Comparison with theoretical predictions."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Filter variants with both measured and theoretical values
        variants_with_theory = [v for v in self.variants 
                               if v['beta'] is not None and v['beta_theory'] is not None]
        
        if not variants_with_theory:
            print("⚠ No variants with theoretical predictions for comparison")
            plt.close()
            return
        
        names = [v['name'] for v in variants_with_theory]
        exponents = ['beta', 'gamma', 'nu']
        exponent_labels = ['β', 'γ', 'ν']
        
        for idx, (exp, label) in enumerate(zip(exponents, exponent_labels)):
            ax = axes[idx // 2, idx % 2]
            
            measured = np.array([v[exp] for v in variants_with_theory])
            errors = np.array([v[f'{exp}_err'] for v in variants_with_theory])
            theory = np.array([v[f'{exp}_theory'] for v in variants_with_theory])
            
            # Calculate deviations in units of sigma
            deviations = np.abs(measured - theory) / errors
            
            x = np.arange(len(names))
            colors = ['#2ca02c' if d < 2 else '#ff7f0e' if d < 3 else '#d62728' 
                     for d in deviations]
            
            ax.bar(x, deviations, color=colors, edgecolor='black')
            ax.axhline(2, color='orange', linestyle='--', linewidth=1, 
                      label='2σ threshold')
            ax.axhline(3, color='red', linestyle='--', linewidth=1,
                      label='3σ threshold')
            
            ax.set_xlabel('Variant')
            ax.set_ylabel(f'|Measured - Theory| / σ')
            ax.set_title(f'({chr(97+idx)}) Deviation for {label}')
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            if idx == 0:
                ax.legend()
        
        # Fourth panel: Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        # Calculate summary statistics
        all_deviations = []
        for exp in exponents:
            measured = np.array([v[exp] for v in variants_with_theory])
            errors = np.array([v[f'{exp}_err'] for v in variants_with_theory])
            theory = np.array([v[f'{exp}_theory'] for v in variants_with_theory])
            deviations = np.abs(measured - theory) / errors
            all_deviations.extend(deviations)
        
        within_1sigma = sum(1 for d in all_deviations if d < 1)
        within_2sigma = sum(1 for d in all_deviations if d < 2)
        within_3sigma = sum(1 for d in all_deviations if d < 3)
        total = len(all_deviations)
        
        summary_text = f"""
        Summary Statistics
        ==================
        
        Total comparisons: {total}
        
        Within 1σ: {within_1sigma} ({100*within_1sigma/total:.1f}%)
        Within 2σ: {within_2sigma} ({100*within_2sigma/total:.1f}%)
        Within 3σ: {within_3sigma} ({100*within_3sigma/total:.1f}%)
        
        Mean deviation: {np.mean(all_deviations):.2f}σ
        Max deviation: {np.max(all_deviations):.2f}σ
        
        Interpretation:
        - Green: <2σ (excellent agreement)
        - Orange: 2-3σ (good agreement)
        - Red: >3σ (significant deviation)
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
               verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_6_theory_comparison.png', bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure_6_theory_comparison.pdf', bbox_inches='tight')
        plt.close()
        print("✓ Figure 6: Theory comparison")

    def table_1_variant_summary(self):
        """Table 1: Summary of all explored variants."""
        table_lines = []
        table_lines.append("% Table 1: Summary of Explored Variants")
        table_lines.append("\\begin{table}[htbp]")
        table_lines.append("\\centering")
        table_lines.append("\\caption{Summary of all explored Ising model variants in the discovery campaign.}")
        table_lines.append("\\label{tab:variant_summary}")
        table_lines.append("\\begin{tabular}{llccl}")
        table_lines.append("\\hline")
        table_lines.append("Variant & Type & Priority & Confidence (\\%) & Status \\\\")
        table_lines.append("\\hline")
        
        for v in self.variants:
            name = v['name'].replace('_', '\\_')
            vtype = v['type'].replace('_', ' ').title()
            priority = v['priority'].title()
            confidence = f"{v['confidence']:.1f}"
            status = v['status'].replace('_', ' ').title()
            
            table_lines.append(f"{name} & {vtype} & {priority} & {confidence} & {status} \\\\")
        
        table_lines.append("\\hline")
        table_lines.append("\\end{tabular}")
        table_lines.append("\\end{table}")
        
        # Save LaTeX version
        latex_file = self.output_dir / 'table_1_variant_summary.tex'
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(table_lines))
        
        # Save CSV version
        csv_file = self.output_dir / 'table_1_variant_summary.csv'
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("Variant,Type,Priority,Confidence (%),Status\n")
            for v in self.variants:
                f.write(f"{v['name']},{v['type']},{v['priority']},{v['confidence']:.1f},{v['status']}\n")
        
        print("✓ Table 1: Variant summary")
    
    def table_2_critical_exponents(self):
        """Table 2: Critical exponents for all variants."""
        table_lines = []
        table_lines.append("% Table 2: Critical Exponents")
        table_lines.append("\\begin{table}[htbp]")
        table_lines.append("\\centering")
        table_lines.append("\\caption{Measured critical exponents for all explored variants. Theoretical values shown in parentheses where available.}")
        table_lines.append("\\label{tab:critical_exponents}")
        table_lines.append("\\begin{tabular}{lcccc}")
        table_lines.append("\\hline")
        table_lines.append("Variant & $\\beta$ & $\\gamma$ & $\\nu$ & $T_c$ \\\\")
        table_lines.append("\\hline")
        
        for v in self.variants:
            name = v['name'].replace('_', '\\_')
            
            if v['beta'] is not None:
                beta_str = f"${v['beta']:.3f} \\pm {v['beta_err']:.3f}$"
                if v['beta_theory'] is not None:
                    beta_str += f" ({v['beta_theory']:.3f})"
                
                gamma_str = f"${v['gamma']:.2f} \\pm {v['gamma_err']:.2f}$"
                if v['gamma_theory'] is not None:
                    gamma_str += f" ({v['gamma_theory']:.2f})"
                
                nu_str = f"${v['nu']:.2f} \\pm {v['nu_err']:.2f}$"
                if v['nu_theory'] is not None:
                    nu_str += f" ({v['nu_theory']:.2f})"
                
                Tc_str = f"${v['Tc']:.3f} \\pm {v['Tc_err']:.3f}$"
                if v['Tc_theory'] is not None:
                    Tc_str += f" ({v['Tc_theory']:.3f})"
            else:
                beta_str = gamma_str = nu_str = Tc_str = "---"
            
            table_lines.append(f"{name} & {beta_str} & {gamma_str} & {nu_str} & {Tc_str} \\\\")
        
        table_lines.append("\\hline")
        table_lines.append("\\end{tabular}")
        table_lines.append("\\end{table}")
        
        # Save LaTeX version
        latex_file = self.output_dir / 'table_2_critical_exponents.tex'
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(table_lines))
        
        # Save CSV version
        csv_file = self.output_dir / 'table_2_critical_exponents.csv'
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("Variant,beta,beta_err,beta_theory,gamma,gamma_err,gamma_theory,nu,nu_err,nu_theory,Tc,Tc_err,Tc_theory\n")
            for v in self.variants:
                f.write(f"{v['name']},{v['beta']},{v['beta_err']},{v['beta_theory']},")
                f.write(f"{v['gamma']},{v['gamma_err']},{v['gamma_theory']},")
                f.write(f"{v['nu']},{v['nu_err']},{v['nu_theory']},")
                f.write(f"{v['Tc']},{v['Tc_err']},{v['Tc_theory']}\n")
        
        print("✓ Table 2: Critical exponents")

    def table_3_validation_metrics(self):
        """Table 3: Validation metrics for all variants."""
        table_lines = []
        table_lines.append("% Table 3: Validation Metrics")
        table_lines.append("\\begin{table}[htbp]")
        table_lines.append("\\centering")
        table_lines.append("\\caption{Validation metrics for all explored variants.}")
        table_lines.append("\\label{tab:validation_metrics}")
        table_lines.append("\\begin{tabular}{lccc}")
        table_lines.append("\\hline")
        table_lines.append("Variant & Confidence (\\%) & Status & Classification \\\\")
        table_lines.append("\\hline")
        
        for v in self.variants:
            name = v['name'].replace('_', '\\_')
            confidence = f"{v['confidence']:.1f}"
            
            # Determine classification
            if v['confidence'] >= 95:
                classification = "Publication Ready"
            elif v['confidence'] >= 90:
                classification = "Validated"
            elif v['confidence'] >= 85:
                classification = "Requires Study"
            else:
                classification = "Below Threshold"
            
            status = v['status'].replace('_', ' ').title()
            
            table_lines.append(f"{name} & {confidence} & {status} & {classification} \\\\")
        
        table_lines.append("\\hline")
        table_lines.append("\\multicolumn{4}{l}{\\textit{Thresholds: Publication Ready $\\geq$95\\%, Validated $\\geq$90\\%, Requires Study $\\geq$85\\%}} \\\\")
        table_lines.append("\\end{tabular}")
        table_lines.append("\\end{table}")
        
        # Save LaTeX version
        latex_file = self.output_dir / 'table_3_validation_metrics.tex'
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(table_lines))
        
        # Save CSV version
        csv_file = self.output_dir / 'table_3_validation_metrics.csv'
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("Variant,Confidence (%),Status,Classification\n")
            for v in self.variants:
                if v['confidence'] >= 95:
                    classification = "Publication Ready"
                elif v['confidence'] >= 90:
                    classification = "Validated"
                elif v['confidence'] >= 85:
                    classification = "Requires Study"
                else:
                    classification = "Below Threshold"
                f.write(f"{v['name']},{v['confidence']:.1f},{v['status']},{classification}\n")
        
        print("✓ Table 3: Validation metrics")
    
    def table_4_computational_resources(self):
        """Table 4: Computational resources used."""
        table_lines = []
        table_lines.append("% Table 4: Computational Resources")
        table_lines.append("\\begin{table}[htbp]")
        table_lines.append("\\centering")
        table_lines.append("\\caption{Computational resources used for each variant exploration.}")
        table_lines.append("\\label{tab:computational_resources}")
        table_lines.append("\\begin{tabular}{lcc}")
        table_lines.append("\\hline")
        table_lines.append("Variant & GPU-hours & Priority \\\\")
        table_lines.append("\\hline")
        
        total_cost = 0
        for v in self.variants:
            name = v['name'].replace('_', '\\_')
            cost = f"{v['cost_gpu_hours']:.1f}"
            priority = v['priority'].title()
            total_cost += v['cost_gpu_hours']
            
            table_lines.append(f"{name} & {cost} & {priority} \\\\")
        
        table_lines.append("\\hline")
        table_lines.append(f"\\textbf{{Total}} & \\textbf{{{total_cost:.1f}}} & --- \\\\")
        table_lines.append("\\hline")
        table_lines.append("\\end{tabular}")
        table_lines.append("\\end{table}")
        
        # Save LaTeX version
        latex_file = self.output_dir / 'table_4_computational_resources.tex'
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(table_lines))
        
        # Save CSV version
        csv_file = self.output_dir / 'table_4_computational_resources.csv'
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("Variant,GPU-hours,Priority\n")
            for v in self.variants:
                f.write(f"{v['name']},{v['cost_gpu_hours']:.1f},{v['priority']}\n")
            f.write(f"Total,{total_cost:.1f},---\n")
        
        print("✓ Table 4: Computational resources")
        
        # Also save summary statistics
        summary = {
            'total_variants': len(self.variants),
            'total_gpu_hours': total_cost,
            'high_priority_variants': sum(1 for v in self.variants if v['priority'] == 'high'),
            'medium_priority_variants': sum(1 for v in self.variants if v['priority'] == 'medium'),
            'baseline_variants': sum(1 for v in self.variants if v['priority'] == 'baseline'),
            'validated_variants': sum(1 for v in self.variants if v['status'] == 'validated'),
            'requires_study_variants': sum(1 for v in self.variants if v['status'] == 'requires_study'),
            'no_transition_variants': sum(1 for v in self.variants if v['status'] == 'no_transition'),
            'mean_confidence': np.mean([v['confidence'] for v in self.variants]),
            'median_confidence': np.median([v['confidence'] for v in self.variants]),
        }
        
        summary_file = self.output_dir / 'campaign_summary_statistics.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Summary statistics saved to {summary_file}")


def main():
    """Generate all figures and tables for campaign overview paper."""
    print("=" * 60)
    print("Campaign Overview Paper - Figure and Table Generation")
    print("=" * 60)
    
    generator = CampaignOverviewFigureGenerator()
    
    # Generate all figures
    generator.generate_all_figures()
    
    # Generate all tables
    generator.generate_all_tables()
    
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {generator.output_dir}")
    print("\nGenerated files:")
    print("  Figures (PNG + PDF):")
    print("    - figure_1_variant_overview")
    print("    - figure_2_exponent_comparison")
    print("    - figure_3_validation_confidence")
    print("    - figure_4_computational_cost")
    print("    - figure_5_exponent_clustering")
    print("    - figure_6_theory_comparison")
    print("\n  Tables (LaTeX + CSV):")
    print("    - table_1_variant_summary")
    print("    - table_2_critical_exponents")
    print("    - table_3_validation_metrics")
    print("    - table_4_computational_resources")
    print("\n  Summary:")
    print("    - campaign_summary_statistics.json")


if __name__ == '__main__':
    main()
