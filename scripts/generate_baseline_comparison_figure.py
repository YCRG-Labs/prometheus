#!/usr/bin/env python3
"""
Generate Figure 2: Baseline Comparison Results
Creates a comprehensive comparison visualization between Prometheus VAE and baseline methods.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_baseline_comparison_figure():
    """Create the baseline comparison figure with clean, non-overlapping layout."""
    
    # Create figure with proper spacing
    fig = plt.figure(figsize=(18, 10))
    
    # Define methods and their performance metrics
    methods = ['PCA', 't-SNE', 'Supervised\nCNN', 'Prometheus\n(Ours)']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Performance data from the paper
    order_param_corr = [0.45, 0.38, 0.72, 0.85]
    order_param_err = [0.08, 0.12, 0.06, 0.04]
    
    temp_error = [12.3, 15.7, 3.2, 0.27]
    temp_error_err = [2.1, 3.4, 0.8, 0.05]
    
    physics_score = [0.52, 0.48, 0.75, 0.88]
    physics_score_err = [0.09, 0.11, 0.07, 0.03]
    
    # Subplot 1: Order Parameter Correlation
    ax1 = plt.subplot(2, 2, 1)
    bars1 = ax1.bar(methods, order_param_corr, yerr=order_param_err, 
                    color=colors, alpha=0.8, capsize=5, width=0.6)
    ax1.set_ylabel('Order Parameter\nCorrelation', fontsize=11, fontweight='bold')
    ax1.set_title('A. Order Parameter Discovery', fontsize=12, fontweight='bold', pad=20)
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add value labels on bars with better positioning
    for bar, val in zip(bars1, order_param_corr):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax1.tick_params(axis='x', labelsize=9)
    ax1.tick_params(axis='y', labelsize=9)
    
    # Subplot 2: Critical Temperature Error
    ax2 = plt.subplot(2, 2, 2)
    bars2 = ax2.bar(methods, temp_error, yerr=temp_error_err, 
                    color=colors, alpha=0.8, capsize=5, width=0.6)
    ax2.set_ylabel('Critical Temperature\nError (%)', fontsize=11, fontweight='bold')
    ax2.set_title('B. Critical Temperature Accuracy', fontsize=12, fontweight='bold', pad=20)
    ax2.set_ylim(0, 22)
    ax2.axhline(y=5.0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add value labels on bars
    for bar, val in zip(bars2, temp_error):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%' if val > 1 else f'{val:.2f}%', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax2.tick_params(axis='x', labelsize=9)
    ax2.tick_params(axis='y', labelsize=9)
    
    # Subplot 3: Physics Consistency Score
    ax3 = plt.subplot(2, 2, 3)
    bars3 = ax3.bar(methods, physics_score, yerr=physics_score_err, 
                    color=colors, alpha=0.8, capsize=5, width=0.6)
    ax3.set_ylabel('Physics Consistency\nScore', fontsize=11, fontweight='bold')
    ax3.set_title('C. Physics Validation', fontsize=12, fontweight='bold', pad=20)
    ax3.set_ylim(0, 1.1)
    ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add value labels on bars
    for bar, val in zip(bars3, physics_score):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax3.tick_params(axis='x', labelsize=9)
    ax3.tick_params(axis='y', labelsize=9)
    
    # Subplot 4: Performance Summary
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Create a clean summary table
    summary_data = [
        ['Metric', 'PCA', 't-SNE', 'Supervised CNN', 'Prometheus'],
        ['Order Parameter Corr.', '0.45', '0.38', '0.72', '0.85'],
        ['Temp. Error (%)', '12.3', '15.7', '3.2', '0.27'],
        ['Physics Score', '0.52', '0.48', '0.75', '0.88'],
        ['Phase Separation', 'Poor', 'Poor', 'Good', 'Excellent']
    ]
    
    # Create table
    table = ax4.table(cellText=summary_data[1:], colLabels=summary_data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.2, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold')
    
    # Highlight Prometheus column
    for i in range(1, len(summary_data)):
        table[(i, 4)].set_facecolor('#96CEB4')
        table[(i, 4)].set_text_props(weight='bold')
    
    ax4.set_title('D. Performance Summary', fontsize=12, fontweight='bold', pad=20)
    
    # Add improvement annotations
    improvement_text = (
        "Key Improvements vs Baselines:\n"
        "• 89% improvement over PCA (order parameter)\n"
        "• 124% improvement over t-SNE (order parameter)\n"
        "• 91.6% reduction in temperature error vs PCA"
    )
    
    ax4.text(0.5, 0.15, improvement_text, ha='center', va='center',
             transform=ax4.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    # Overall title with proper spacing
    fig.suptitle('Baseline Comparison: Prometheus VAE vs Traditional Methods', 
                fontsize=16, fontweight='bold', y=0.96)
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.95, 
                       hspace=0.35, wspace=0.25)
    
    return fig

def main():
    """Generate and save the baseline comparison figure."""
    
    # Create the figure
    fig = create_baseline_comparison_figure()
    
    # Save the figure
    output_path = 'Paper/figures/02_baseline_comparison.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Baseline comparison figure saved to: {output_path}")
    
    # Also save as PDF for high-quality printing
    pdf_path = 'Paper/figures/02_baseline_comparison.pdf'
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"PDF version saved to: {pdf_path}")
    
    plt.show()

if __name__ == "__main__":
    main()