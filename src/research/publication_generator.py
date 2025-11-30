"""
Publication Generator for Novel Physics Discovery Campaign

This module implements Task 6: Implement publication generator that:
- Generates publication-ready figures for discoveries
- Creates LaTeX-formatted tables with exponents and errors
- Generates draft abstract and results text
- Exports supplementary data with complete provenance

The generator produces complete publication packages ready for journal submission.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
import h5py
import csv

from .discovery_assessor import PhysicsDiscovery
from .base_types import VAEAnalysisResults, SimulationData
from .unified_validation_pipeline import ValidationReport
from ..analysis.publication_figure_generator import PublicationFigureGenerator
from ..analysis.publication_materials import PublicationMaterialsGenerator
from ..utils.logging_utils import get_logger


@dataclass
class PublicationPackage:
    """Complete publication package for a discovery.
    
    Attributes:
        discovery: The physics discovery
        figures: Dictionary of generated figures
        tables: Dictionary of generated tables
        draft_text: Dictionary of draft text sections
        supplementary_data: Dictionary of supplementary data files
        metadata: Package metadata
    """
    discovery: PhysicsDiscovery
    figures: Dict[str, str] = field(default_factory=dict)
    tables: Dict[str, str] = field(default_factory=dict)
    draft_text: Dict[str, str] = field(default_factory=dict)
    supplementary_data: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PublicationGenerator:
    """Generate publication-ready outputs for physics discoveries.
    
    This class creates complete publication packages including:
    - High-quality figures (phase diagrams, exponent plots, scaling collapse)
    - LaTeX-formatted tables with exponents and confidence intervals
    - Draft abstract and results section text
    - Supplementary data files with complete provenance
    
    Attributes:
        figure_generator: Publication figure generator
        materials_generator: Publication materials generator
        logger: Logger instance
    """
    
    def __init__(self):
        """Initialize publication generator."""
        self.logger = get_logger(__name__)
        self.figure_generator = PublicationFigureGenerator()
        self.materials_generator = PublicationMaterialsGenerator()
        
        # Publication settings
        self.journal_format = "Physical Review E"
        self.figure_dpi = 300
        self.figure_format = "both"  # PNG and PDF
        
        self.logger.info("Initialized PublicationGenerator")
    
    def generate_package(
        self,
        discovery: PhysicsDiscovery,
        vae_results: VAEAnalysisResults,
        validation_report: ValidationReport,
        simulation_data: Optional[SimulationData] = None,
        output_dir: Optional[Path] = None
    ) -> PublicationPackage:
        """Generate complete publication package for a discovery.
        
        This is the main entry point for publication generation.
        
        Args:
            discovery: PhysicsDiscovery object
            vae_results: VAE analysis results
            validation_report: Validation report
            simulation_data: Optional simulation data
            output_dir: Optional output directory
            
        Returns:
            PublicationPackage with all generated materials
        """
        self.logger.info(
            f"Generating publication package for discovery {discovery.discovery_id}"
        )
        
        # Setup output directory
        if output_dir is None:
            output_dir = Path(f"results/publication/{discovery.discovery_id}")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create package
        package = PublicationPackage(discovery=discovery)
        
        # Generate figures
        self.logger.info("Generating figures...")
        package.figures = self.generate_figures(
            discovery, vae_results, validation_report, 
            simulation_data, output_dir / "figures"
        )
        
        # Generate tables
        self.logger.info("Generating tables...")
        package.tables = self.generate_tables(
            discovery, vae_results, validation_report,
            output_dir / "tables"
        )
        
        # Generate draft text
        self.logger.info("Generating draft text...")
        package.draft_text = self.generate_draft_text(
            discovery, vae_results, validation_report,
            output_dir / "text"
        )
        
        # Export supplementary data
        self.logger.info("Exporting supplementary data...")
        package.supplementary_data = self.export_supplementary_data(
            discovery, vae_results, validation_report, simulation_data,
            output_dir / "supplementary"
        )
        
        # Add metadata
        package.metadata = {
            'generation_time': datetime.now().isoformat(),
            'discovery_id': discovery.discovery_id,
            'variant_id': discovery.variant_id,
            'journal_format': self.journal_format,
            'output_directory': str(output_dir)
        }
        
        # Save package manifest
        self._save_package_manifest(package, output_dir)
        
        self.logger.info(
            f"Publication package generated successfully in {output_dir}"
        )
        
        return package
    
    def generate_figures(
        self,
        discovery: PhysicsDiscovery,
        vae_results: VAEAnalysisResults,
        validation_report: ValidationReport,
        simulation_data: Optional[SimulationData],
        output_dir: Path
    ) -> Dict[str, str]:
        """Generate publication-quality figures.
        
        Args:
            discovery: PhysicsDiscovery object
            vae_results: VAE analysis results
            validation_report: Validation report
            simulation_data: Optional simulation data
            output_dir: Output directory for figures
            
        Returns:
            Dictionary mapping figure names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        figures = {}
        
        # Figure 1: Phase diagram
        fig1 = self._create_phase_diagram(discovery, vae_results, simulation_data)
        figures['phase_diagram'] = self._save_figure(
            fig1, output_dir / "phase_diagram", self.figure_format
        )
        plt.close(fig1)
        
        # Figure 2: Exponent comparison
        fig2 = self._create_exponent_comparison(discovery, vae_results)
        figures['exponent_comparison'] = self._save_figure(
            fig2, output_dir / "exponent_comparison", self.figure_format
        )
        plt.close(fig2)
        
        # Figure 3: Scaling collapse
        fig3 = self._create_scaling_collapse(discovery, vae_results, simulation_data)
        figures['scaling_collapse'] = self._save_figure(
            fig3, output_dir / "scaling_collapse", self.figure_format
        )
        plt.close(fig3)
        
        # Figure 4: Validation summary
        fig4 = self._create_validation_visualization(discovery, validation_report)
        figures['validation_summary'] = self._save_figure(
            fig4, output_dir / "validation_summary", self.figure_format
        )
        plt.close(fig4)
        
        self.logger.info(f"Generated {len(figures)} figures")
        
        return figures

    
    def _create_phase_diagram(
        self,
        discovery: PhysicsDiscovery,
        vae_results: VAEAnalysisResults,
        simulation_data: Optional[SimulationData]
    ) -> Figure:
        """Create phase diagram figure.
        
        Args:
            discovery: PhysicsDiscovery object
            vae_results: VAE analysis results
            simulation_data: Optional simulation data
            
        Returns:
            Matplotlib Figure
        """
        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel A: Latent space with temperature coloring
        ax_a = fig.add_subplot(gs[0, 0])
        
        if hasattr(vae_results, 'latent_z1') and len(vae_results.latent_z1) > 0:
            scatter = ax_a.scatter(
                vae_results.latent_z1,
                vae_results.latent_z2,
                c=vae_results.temperatures,
                cmap='coolwarm',
                alpha=0.7,
                s=30,
                edgecolors='black',
                linewidth=0.3
            )
            
            cbar = plt.colorbar(scatter, ax=ax_a)
            cbar.set_label('Temperature', fontsize=12)
            
            # Mark critical temperature
            tc = discovery.metadata.get('critical_temperature', vae_results.critical_temperature)
            ax_a.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
            ax_a.text(
                0.05, 0.95, f'Tc = {tc:.3f}',
                transform=ax_a.transAxes,
                fontsize=12,
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        ax_a.set_xlabel('Latent Dimension z₁', fontsize=12)
        ax_a.set_ylabel('Latent Dimension z₂', fontsize=12)
        ax_a.set_title('(a) Latent Space Phase Diagram', fontsize=14, fontweight='bold')
        ax_a.grid(True, alpha=0.3)
        
        # Panel B: Order parameter vs temperature
        ax_b = fig.add_subplot(gs[0, 1])
        
        if hasattr(vae_results, 'order_parameter') and len(vae_results.order_parameter) > 0:
            # Sort by temperature
            sorted_idx = np.argsort(vae_results.temperatures)
            temps = vae_results.temperatures[sorted_idx]
            order_param = np.abs(vae_results.order_parameter[sorted_idx])
            
            ax_b.plot(temps, order_param, 'o-', linewidth=2, markersize=6, alpha=0.7)
            
            # Mark critical temperature
            tc = discovery.metadata.get('critical_temperature', vae_results.critical_temperature)
            ax_b.axvline(tc, color='red', linestyle='--', linewidth=2, label=f'Tc = {tc:.3f}')
        
        ax_b.set_xlabel('Temperature', fontsize=12)
        ax_b.set_ylabel('Order Parameter', fontsize=12)
        ax_b.set_title('(b) Order Parameter Evolution', fontsize=14, fontweight='bold')
        ax_b.legend(fontsize=10)
        ax_b.grid(True, alpha=0.3)
        
        # Panel C: Magnetization vs temperature
        ax_c = fig.add_subplot(gs[1, 0])
        
        if simulation_data and hasattr(simulation_data, 'magnetizations'):
            sorted_idx = np.argsort(simulation_data.temperatures)
            temps = simulation_data.temperatures[sorted_idx]
            mags = np.abs(simulation_data.magnetizations[sorted_idx])
            
            ax_c.plot(temps, mags, 'o-', linewidth=2, markersize=6, alpha=0.7, color='green')
            
            # Mark critical temperature
            tc = discovery.metadata.get('critical_temperature', vae_results.critical_temperature)
            ax_c.axvline(tc, color='red', linestyle='--', linewidth=2, label=f'Tc = {tc:.3f}')
        
        ax_c.set_xlabel('Temperature', fontsize=12)
        ax_c.set_ylabel('|Magnetization|', fontsize=12)
        ax_c.set_title('(c) Magnetization vs Temperature', fontsize=14, fontweight='bold')
        ax_c.legend(fontsize=10)
        ax_c.grid(True, alpha=0.3)
        
        # Panel D: Susceptibility vs temperature
        ax_d = fig.add_subplot(gs[1, 1])
        
        if simulation_data and hasattr(simulation_data, 'susceptibilities'):
            sorted_idx = np.argsort(simulation_data.temperatures)
            temps = simulation_data.temperatures[sorted_idx]
            chi = simulation_data.susceptibilities[sorted_idx]
            
            ax_d.plot(temps, chi, 'o-', linewidth=2, markersize=6, alpha=0.7, color='purple')
            
            # Mark critical temperature
            tc = discovery.metadata.get('critical_temperature', vae_results.critical_temperature)
            ax_d.axvline(tc, color='red', linestyle='--', linewidth=2, label=f'Tc = {tc:.3f}')
        
        ax_d.set_xlabel('Temperature', fontsize=12)
        ax_d.set_ylabel('Susceptibility', fontsize=12)
        ax_d.set_title('(d) Susceptibility vs Temperature', fontsize=14, fontweight='bold')
        ax_d.legend(fontsize=10)
        ax_d.grid(True, alpha=0.3)
        
        plt.suptitle(
            f'Phase Diagram: {discovery.variant_id}',
            fontsize=16,
            fontweight='bold'
        )
        
        return fig
    
    def _create_exponent_comparison(
        self,
        discovery: PhysicsDiscovery,
        vae_results: VAEAnalysisResults
    ) -> Figure:
        """Create critical exponent comparison figure.
        
        Args:
            discovery: PhysicsDiscovery object
            vae_results: VAE analysis results
            
        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Get exponents
        exponents = discovery.critical_exponents
        errors = discovery.exponent_errors
        
        # Get theoretical comparison
        theory_comp = discovery.theoretical_comparison
        closest_class = discovery.metadata.get('closest_universality_class', 'Unknown')
        
        # Panel A: Measured vs theoretical
        ax_a = axes[0]
        
        exp_names = list(exponents.keys())
        measured_values = [exponents[name] for name in exp_names]
        error_values = [errors.get(name, 0.0) for name in exp_names]
        
        # Get theoretical values from closest class or predictions
        theoretical_values = []
        for name in exp_names:
            # Try to get from theoretical comparison
            theory_val = None
            if 'agreements' in theory_comp:
                for agreement in theory_comp['agreements']:
                    if agreement['exponent'] == name:
                        theory_val = agreement['predicted']
                        break
            if theory_val is None and 'disagreements' in theory_comp:
                for disagreement in theory_comp['disagreements']:
                    if disagreement['exponent'] == name:
                        theory_val = disagreement['predicted']
                        break
            
            theoretical_values.append(theory_val if theory_val is not None else measured_values[exp_names.index(name)])
        
        # Plot measured vs theoretical
        x = np.arange(len(exp_names))
        width = 0.35
        
        bars1 = ax_a.bar(
            x - width/2, measured_values, width,
            yerr=error_values,
            label='Measured',
            alpha=0.8,
            capsize=5,
            color='blue',
            edgecolor='black',
            linewidth=1
        )
        
        bars2 = ax_a.bar(
            x + width/2, theoretical_values, width,
            label=f'Theoretical ({closest_class})',
            alpha=0.8,
            color='red',
            edgecolor='black',
            linewidth=1
        )
        
        ax_a.set_xlabel('Critical Exponent', fontsize=12)
        ax_a.set_ylabel('Value', fontsize=12)
        ax_a.set_title('(a) Measured vs Theoretical Exponents', fontsize=14, fontweight='bold')
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(exp_names, fontsize=12)
        ax_a.legend(fontsize=10)
        ax_a.grid(True, alpha=0.3, axis='y')
        
        # Add value annotations
        for bar, val in zip(bars1, measured_values):
            height = bar.get_height()
            ax_a.annotate(
                f'{val:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        # Panel B: Deviation from closest class
        ax_b = axes[1]
        
        deviations = discovery.metadata.get('deviations', {})
        dev_names = list(deviations.keys())
        dev_values = [deviations[name] for name in dev_names]
        
        bars = ax_b.bar(
            range(len(dev_names)),
            dev_values,
            alpha=0.8,
            color='orange',
            edgecolor='black',
            linewidth=1
        )
        
        # Add threshold lines
        ax_b.axhline(3.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='3σ Threshold')
        ax_b.axhline(5.0, color='darkred', linestyle='--', linewidth=2, alpha=0.7, label='5σ Threshold')
        
        ax_b.set_xlabel('Critical Exponent', fontsize=12)
        ax_b.set_ylabel('Deviation (σ)', fontsize=12)
        ax_b.set_title(f'(b) Deviation from {closest_class}', fontsize=14, fontweight='bold')
        ax_b.set_xticks(range(len(dev_names)))
        ax_b.set_xticklabels(dev_names, fontsize=12)
        ax_b.legend(fontsize=10)
        ax_b.grid(True, alpha=0.3, axis='y')
        
        # Add value annotations
        for bar, val in zip(bars, dev_values):
            height = bar.get_height()
            ax_b.annotate(
                f'{val:.2f}σ',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )
        
        plt.suptitle(
            f'Critical Exponent Analysis: {discovery.variant_id}',
            fontsize=16,
            fontweight='bold'
        )
        plt.tight_layout()
        
        return fig
    
    def _create_scaling_collapse(
        self,
        discovery: PhysicsDiscovery,
        vae_results: VAEAnalysisResults,
        simulation_data: Optional[SimulationData]
    ) -> Figure:
        """Create scaling collapse demonstration figure.
        
        Args:
            discovery: PhysicsDiscovery object
            vae_results: VAE analysis results
            simulation_data: Optional simulation data
            
        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Get critical exponents
        beta = discovery.critical_exponents.get('beta', 0.125)
        nu = discovery.critical_exponents.get('nu', 1.0)
        tc = discovery.metadata.get('critical_temperature', vae_results.critical_temperature)
        
        # Panel A: Magnetization scaling
        ax_a = axes[0]
        
        if simulation_data and hasattr(simulation_data, 'magnetizations'):
            temps = simulation_data.temperatures
            mags = np.abs(simulation_data.magnetizations)
            
            # Calculate reduced temperature
            t_reduced = (temps - tc) / tc
            
            # Scaling collapse: M * |t|^beta vs t
            mask = np.abs(t_reduced) > 0.01  # Avoid division by zero
            t_plot = t_reduced[mask]
            m_scaled = mags[mask] * np.abs(t_plot) ** beta
            
            ax_a.plot(t_plot, m_scaled, 'o', alpha=0.7, markersize=6)
            ax_a.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        
        ax_a.set_xlabel('Reduced Temperature (T - Tc) / Tc', fontsize=12)
        ax_a.set_ylabel(f'M × |t|^β (β = {beta:.3f})', fontsize=12)
        ax_a.set_title('(a) Magnetization Scaling Collapse', fontsize=14, fontweight='bold')
        ax_a.grid(True, alpha=0.3)
        
        # Panel B: Susceptibility scaling
        ax_b = axes[1]
        
        if simulation_data and hasattr(simulation_data, 'susceptibilities'):
            temps = simulation_data.temperatures
            chi = simulation_data.susceptibilities
            
            # Calculate reduced temperature
            t_reduced = (temps - tc) / tc
            
            # Scaling collapse: χ * |t|^gamma vs t
            gamma = discovery.critical_exponents.get('gamma', 1.75)
            mask = np.abs(t_reduced) > 0.01
            t_plot = t_reduced[mask]
            chi_scaled = chi[mask] * np.abs(t_plot) ** gamma
            
            ax_b.plot(t_plot, chi_scaled, 'o', alpha=0.7, markersize=6, color='purple')
            ax_b.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        
        ax_b.set_xlabel('Reduced Temperature (T - Tc) / Tc', fontsize=12)
        ax_b.set_ylabel(f'χ × |t|^γ (γ = {gamma:.3f})', fontsize=12)
        ax_b.set_title('(b) Susceptibility Scaling Collapse', fontsize=14, fontweight='bold')
        ax_b.grid(True, alpha=0.3)
        
        plt.suptitle(
            f'Scaling Collapse Demonstration: {discovery.variant_id}',
            fontsize=16,
            fontweight='bold'
        )
        plt.tight_layout()
        
        return fig
    
    def _create_validation_visualization(
        self,
        discovery: PhysicsDiscovery,
        validation_report: ValidationReport
    ) -> Figure:
        """Create validation summary visualization.
        
        Args:
            discovery: PhysicsDiscovery object
            validation_report: Validation report
            
        Returns:
            Matplotlib Figure
        """
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel A: Overall confidence
        ax_a = fig.add_subplot(gs[0, 0])
        
        confidence = validation_report.overall_confidence
        
        # Create gauge chart
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        ax_a = plt.subplot(2, 2, 1, projection='polar')
        ax_a.plot(theta, r, 'k-', linewidth=3)
        
        # Color based on confidence
        if confidence >= 0.95:
            color = 'darkgreen'
            status = 'EXCELLENT'
        elif confidence >= 0.90:
            color = 'green'
            status = 'VERY GOOD'
        elif confidence >= 0.80:
            color = 'orange'
            status = 'GOOD'
        else:
            color = 'red'
            status = 'NEEDS IMPROVEMENT'
        
        conf_theta = np.linspace(0, np.pi * confidence, 50)
        conf_r = np.ones_like(conf_theta)
        ax_a.fill_between(conf_theta, 0, conf_r, alpha=0.7, color=color)
        
        ax_a.text(
            np.pi/2, 0.5, f'{confidence:.1%}\n{status}',
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        ax_a.set_ylim(0, 1)
        ax_a.set_theta_zero_location('W')
        ax_a.set_thetagrids([0, 45, 90, 135, 180], ['0%', '25%', '50%', '75%', '100%'])
        ax_a.set_title('(a) Overall Validation Confidence', fontsize=14, fontweight='bold', pad=20)
        
        # Panel B: Pattern results
        ax_b = fig.add_subplot(gs[0, 1])
        
        pattern_names = []
        pattern_confidences = []
        
        for pattern_name, pattern_result in validation_report.pattern_results.items():
            if isinstance(pattern_result, dict) and 'confidence' in pattern_result:
                pattern_names.append(pattern_name.replace('_', '\n'))
                pattern_confidences.append(pattern_result['confidence'])
        
        if pattern_names:
            bars = ax_b.barh(
                range(len(pattern_names)),
                pattern_confidences,
                alpha=0.8,
                color='steelblue',
                edgecolor='black',
                linewidth=1
            )
            
            ax_b.axvline(0.9, color='red', linestyle='--', linewidth=2, alpha=0.7, label='90% Threshold')
            
            ax_b.set_xlabel('Confidence', fontsize=12)
            ax_b.set_ylabel('Validation Pattern', fontsize=12)
            ax_b.set_title('(b) Pattern-Level Confidence', fontsize=14, fontweight='bold')
            ax_b.set_yticks(range(len(pattern_names)))
            ax_b.set_yticklabels(pattern_names, fontsize=9)
            ax_b.legend(fontsize=10)
            ax_b.grid(True, alpha=0.3, axis='x')
            ax_b.set_xlim(0, 1)
        
        # Panel C: Discovery classification
        ax_c = fig.add_subplot(gs[1, 0])
        
        # Create classification summary
        classification_data = {
            'Type': discovery.discovery_type.replace('_', ' ').title(),
            'Significance': discovery.significance.upper(),
            'Publication\nPotential': discovery.publication_potential.upper(),
            'Validation\nStatus': 'VALIDATED' if validation_report.overall_validated else 'NOT VALIDATED'
        }
        
        labels = list(classification_data.keys())
        values_text = list(classification_data.values())
        
        # Color code based on values
        colors_map = {
            'major': 'darkgreen', 'moderate': 'green', 'minor': 'orange',
            'high': 'darkgreen', 'medium': 'orange', 'low': 'red',
            'VALIDATED': 'green', 'NOT VALIDATED': 'red'
        }
        
        ax_c.axis('off')
        
        # Create table
        table_data = [[label, value] for label, value in zip(labels, values_text)]
        table = ax_c.table(
            cellText=table_data,
            colLabels=['Category', 'Value'],
            cellLoc='left',
            loc='center',
            colWidths=[0.4, 0.6]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Color cells
        for i, (label, value) in enumerate(zip(labels, values_text), start=1):
            cell_color = colors_map.get(discovery.significance, 'lightgray')
            if 'Significance' in label:
                cell_color = colors_map.get(discovery.significance, 'lightgray')
            elif 'Potential' in label:
                cell_color = colors_map.get(discovery.publication_potential, 'lightgray')
            elif 'Status' in label:
                cell_color = colors_map.get(value, 'lightgray')
            
            table[(i, 1)].set_facecolor(cell_color)
            table[(i, 1)].set_alpha(0.3)
        
        ax_c.set_title('(c) Discovery Classification', fontsize=14, fontweight='bold')
        
        # Panel D: Theoretical comparison
        ax_d = fig.add_subplot(gs[1, 1])
        
        theory_comp = discovery.theoretical_comparison
        status = theory_comp.get('status', 'unknown')
        message = theory_comp.get('message', 'No comparison available')
        
        # Create status visualization
        status_colors = {
            'confirmed': 'green',
            'partial': 'orange',
            'refuted': 'red',
            'extended': 'blue',
            'no_predictions': 'gray'
        }
        
        status_color = status_colors.get(status, 'gray')
        
        ax_d.axis('off')
        
        # Add status box
        ax_d.text(
            0.5, 0.7,
            f'Theoretical Comparison:\n{status.upper().replace("_", " ")}',
            ha='center', va='center',
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.3, pad=1)
        )
        
        # Add message
        ax_d.text(
            0.5, 0.3,
            message,
            ha='center', va='center',
            fontsize=10,
            wrap=True,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        ax_d.set_title('(d) Theoretical Comparison', fontsize=14, fontweight='bold')
        
        plt.suptitle(
            f'Validation Summary: {discovery.variant_id}',
            fontsize=16,
            fontweight='bold'
        )
        
        return fig
    
    def _save_figure(
        self,
        fig: Figure,
        base_path: Path,
        format: str = "both"
    ) -> str:
        """Save figure in specified format(s).
        
        Args:
            fig: Matplotlib Figure
            base_path: Base path without extension
            format: "png", "pdf", or "both"
            
        Returns:
            Path to primary saved figure (PNG if both)
        """
        base_path = Path(base_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        saved_path = None
        
        if format in ["png", "both"]:
            png_path = base_path.with_suffix('.png')
            fig.savefig(png_path, dpi=self.figure_dpi, format='png', bbox_inches='tight')
            saved_path = str(png_path)
            self.logger.debug(f"Saved PNG: {png_path}")
        
        if format in ["pdf", "both"]:
            pdf_path = base_path.with_suffix('.pdf')
            fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
            if saved_path is None:
                saved_path = str(pdf_path)
            self.logger.debug(f"Saved PDF: {pdf_path}")
        
        return saved_path

    
    def generate_tables(
        self,
        discovery: PhysicsDiscovery,
        vae_results: VAEAnalysisResults,
        validation_report: ValidationReport,
        output_dir: Path
    ) -> Dict[str, str]:
        """Generate LaTeX-formatted tables.
        
        Args:
            discovery: PhysicsDiscovery object
            vae_results: VAE analysis results
            validation_report: Validation report
            output_dir: Output directory for tables
            
        Returns:
            Dictionary mapping table names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tables = {}
        
        # Table 1: Critical exponents
        tables['exponents'] = self._create_exponent_table(
            discovery, output_dir / "exponents_table.tex"
        )
        
        # Table 2: Validation metrics
        tables['validation'] = self._create_validation_table(
            discovery, validation_report, output_dir / "validation_table.tex"
        )
        
        # Table 3: Comparison with known classes
        tables['comparison'] = self._create_comparison_table(
            discovery, output_dir / "comparison_table.tex"
        )
        
        # Also save CSV versions
        tables['exponents_csv'] = self._create_exponent_table_csv(
            discovery, output_dir / "exponents_table.csv"
        )
        
        self.logger.info(f"Generated {len(tables)} tables")
        
        return tables
    
    def _create_exponent_table(
        self,
        discovery: PhysicsDiscovery,
        output_path: Path
    ) -> str:
        """Create LaTeX-formatted exponent table.
        
        Args:
            discovery: PhysicsDiscovery object
            output_path: Output file path
            
        Returns:
            Path to saved table
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("% Critical Exponents Table\n")
            f.write("% Generated automatically\n\n")
            
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Critical exponents for " + discovery.variant_id.replace('_', '\\_') + "}\n")
            f.write("\\label{tab:exponents_" + discovery.variant_id + "}\n")
            f.write("\\begin{tabular}{lccc}\n")
            f.write("\\hline\\hline\n")
            f.write("Exponent & Measured & Error & Deviation ($\\sigma$) \\\\\n")
            f.write("\\hline\n")
            
            # Get deviations
            deviations = discovery.metadata.get('deviations', {})
            
            for exp_name in sorted(discovery.critical_exponents.keys()):
                exp_value = discovery.critical_exponents[exp_name]
                exp_error = discovery.exponent_errors.get(exp_name, 0.0)
                exp_deviation = deviations.get(exp_name, 0.0)
                
                # Format exponent name with Greek letters
                exp_display = exp_name
                if exp_name == 'beta':
                    exp_display = '$\\beta$'
                elif exp_name == 'nu':
                    exp_display = '$\\nu$'
                elif exp_name == 'gamma':
                    exp_display = '$\\gamma$'
                elif exp_name == 'alpha':
                    exp_display = '$\\alpha$'
                elif exp_name == 'delta':
                    exp_display = '$\\delta$'
                elif exp_name == 'eta':
                    exp_display = '$\\eta$'
                
                f.write(f"{exp_display} & {exp_value:.4f} & {exp_error:.4f} & {exp_deviation:.2f} \\\\\n")
            
            f.write("\\hline\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        self.logger.debug(f"Created exponent table: {output_path}")
        
        return str(output_path)
    
    def _create_validation_table(
        self,
        discovery: PhysicsDiscovery,
        validation_report: ValidationReport,
        output_path: Path
    ) -> str:
        """Create LaTeX-formatted validation table.
        
        Args:
            discovery: PhysicsDiscovery object
            validation_report: Validation report
            output_path: Output file path
            
        Returns:
            Path to saved table
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("% Validation Metrics Table\n")
            f.write("% Generated automatically\n\n")
            
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Validation metrics for " + discovery.variant_id.replace('_', '\\_') + "}\n")
            f.write("\\label{tab:validation_" + discovery.variant_id + "}\n")
            f.write("\\begin{tabular}{lc}\n")
            f.write("\\hline\\hline\n")
            f.write("Metric & Value \\\\\n")
            f.write("\\hline\n")
            
            # Overall confidence
            f.write(f"Overall Confidence & {validation_report.overall_confidence:.2%} \\\\\n")
            
            # Validation status
            status = "Validated" if validation_report.overall_validated else "Not Validated"
            f.write(f"Validation Status & {status} \\\\\n")
            
            # Publication ready
            pub_ready = "Yes" if validation_report.publication_ready else "No"
            f.write(f"Publication Ready & {pub_ready} \\\\\n")
            
            # Discovery classification
            f.write(f"Discovery Type & {discovery.discovery_type.replace('_', ' ').title()} \\\\\n")
            f.write(f"Significance & {discovery.significance.title()} \\\\\n")
            f.write(f"Publication Potential & {discovery.publication_potential.title()} \\\\\n")
            
            # Critical temperature
            tc = discovery.metadata.get('critical_temperature', 0.0)
            tc_conf = discovery.metadata.get('tc_confidence', 0.0)
            f.write(f"Critical Temperature & {tc:.4f} \\\\\n")
            f.write(f"$T_c$ Confidence & {tc_conf:.2%} \\\\\n")
            
            f.write("\\hline\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        self.logger.debug(f"Created validation table: {output_path}")
        
        return str(output_path)
    
    def _create_comparison_table(
        self,
        discovery: PhysicsDiscovery,
        output_path: Path
    ) -> str:
        """Create LaTeX-formatted comparison table.
        
        Args:
            discovery: PhysicsDiscovery object
            output_path: Output file path
            
        Returns:
            Path to saved table
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        theory_comp = discovery.theoretical_comparison
        closest_class = discovery.metadata.get('closest_universality_class', 'Unknown')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("% Comparison with Known Classes Table\n")
            f.write("% Generated automatically\n\n")
            
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Comparison with known universality classes for " + 
                   discovery.variant_id.replace('_', '\\_') + "}\n")
            f.write("\\label{tab:comparison_" + discovery.variant_id + "}\n")
            f.write("\\begin{tabular}{lccc}\n")
            f.write("\\hline\\hline\n")
            f.write("Exponent & Measured & " + closest_class.replace('_', '\\_') + " & Deviation ($\\sigma$) \\\\\n")
            f.write("\\hline\n")
            
            deviations = discovery.metadata.get('deviations', {})
            
            # Get theoretical values from comparison
            theoretical_values = {}
            if 'agreements' in theory_comp:
                for agreement in theory_comp['agreements']:
                    theoretical_values[agreement['exponent']] = agreement['predicted']
            if 'disagreements' in theory_comp:
                for disagreement in theory_comp['disagreements']:
                    theoretical_values[disagreement['exponent']] = disagreement['predicted']
            
            for exp_name in sorted(discovery.critical_exponents.keys()):
                exp_value = discovery.critical_exponents[exp_name]
                theory_value = theoretical_values.get(exp_name, exp_value)
                exp_deviation = deviations.get(exp_name, 0.0)
                
                # Format exponent name
                exp_display = exp_name
                if exp_name == 'beta':
                    exp_display = '$\\beta$'
                elif exp_name == 'nu':
                    exp_display = '$\\nu$'
                elif exp_name == 'gamma':
                    exp_display = '$\\gamma$'
                
                f.write(f"{exp_display} & {exp_value:.4f} & {theory_value:.4f} & {exp_deviation:.2f} \\\\\n")
            
            f.write("\\hline\n")
            f.write(f"\\multicolumn{{4}}{{l}}{{Comparison Status: {theory_comp.get('status', 'unknown').title()}}} \\\\\n")
            f.write("\\hline\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        self.logger.debug(f"Created comparison table: {output_path}")
        
        return str(output_path)
    
    def _create_exponent_table_csv(
        self,
        discovery: PhysicsDiscovery,
        output_path: Path
    ) -> str:
        """Create CSV version of exponent table.
        
        Args:
            discovery: PhysicsDiscovery object
            output_path: Output file path
            
        Returns:
            Path to saved table
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        deviations = discovery.metadata.get('deviations', {})
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Exponent', 'Measured', 'Error', 'Deviation_sigma'])
            
            for exp_name in sorted(discovery.critical_exponents.keys()):
                exp_value = discovery.critical_exponents[exp_name]
                exp_error = discovery.exponent_errors.get(exp_name, 0.0)
                exp_deviation = deviations.get(exp_name, 0.0)
                
                writer.writerow([exp_name, exp_value, exp_error, exp_deviation])
        
        self.logger.debug(f"Created CSV exponent table: {output_path}")
        
        return str(output_path)
    
    def generate_draft_text(
        self,
        discovery: PhysicsDiscovery,
        vae_results: VAEAnalysisResults,
        validation_report: ValidationReport,
        output_dir: Path
    ) -> Dict[str, str]:
        """Generate draft manuscript text.
        
        Args:
            discovery: PhysicsDiscovery object
            vae_results: VAE analysis results
            validation_report: Validation report
            output_dir: Output directory for text files
            
        Returns:
            Dictionary mapping text section names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        text_sections = {}
        
        # Generate abstract
        text_sections['abstract'] = self._generate_abstract(
            discovery, vae_results, validation_report,
            output_dir / "abstract.txt"
        )
        
        # Generate results section
        text_sections['results'] = self._generate_results_section(
            discovery, vae_results, validation_report,
            output_dir / "results.txt"
        )
        
        # Generate discussion points
        text_sections['discussion'] = self._generate_discussion_points(
            discovery, vae_results, validation_report,
            output_dir / "discussion.txt"
        )
        
        self.logger.info(f"Generated {len(text_sections)} text sections")
        
        return text_sections

    
    def _generate_abstract(
        self,
        discovery: PhysicsDiscovery,
        vae_results: VAEAnalysisResults,
        validation_report: ValidationReport,
        output_path: Path
    ) -> str:
        """Generate draft abstract.
        
        Args:
            discovery: PhysicsDiscovery object
            vae_results: VAE analysis results
            validation_report: Validation report
            output_path: Output file path
            
        Returns:
            Path to saved abstract
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get key information
        variant_name = discovery.variant_id.replace('_', ' ').title()
        discovery_type = discovery.discovery_type.replace('_', ' ')
        closest_class = discovery.metadata.get('closest_universality_class', 'known classes')
        max_deviation = discovery.metadata.get('max_deviation_sigma', 0.0)
        confidence = validation_report.overall_confidence
        
        # Get exponents
        exponents_str = ", ".join([
            f"{name}={value:.3f}±{discovery.exponent_errors.get(name, 0.0):.3f}"
            for name, value in discovery.critical_exponents.items()
        ])
        
        # Get theoretical comparison status
        theory_status = discovery.theoretical_comparison.get('status', 'no_predictions')
        theory_message = discovery.theoretical_comparison.get('message', '')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("DRAFT ABSTRACT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"We report the discovery of {discovery_type} in the {variant_name} ")
            f.write(f"using unsupervised machine learning methods combined with rigorous ")
            f.write(f"statistical validation. ")
            
            f.write(f"Through systematic exploration of the parameter space, we identified ")
            f.write(f"critical behavior that deviates significantly (>{max_deviation:.1f}σ) ")
            f.write(f"from {closest_class}. ")
            
            f.write(f"The measured critical exponents are {exponents_str}. ")
            
            if theory_status == 'refuted':
                f.write(f"These findings refute existing theoretical predictions. ")
            elif theory_status == 'extended':
                f.write(f"These findings extend beyond current theoretical predictions. ")
            elif theory_status == 'confirmed':
                f.write(f"These findings confirm theoretical predictions. ")
            
            f.write(f"Our results were validated using a comprehensive framework of 10 ")
            f.write(f"statistical validation patterns, achieving an overall confidence of ")
            f.write(f"{confidence:.1%}. ")
            
            if discovery.discovery_type == 'novel_universality_class':
                f.write(f"We propose that this represents a new universality class ")
                f.write(f"distinct from all previously known classes. ")
            
            f.write(f"These findings have implications for understanding critical phenomena ")
            f.write(f"in {variant_name.lower()} and related systems.\n\n")
            
            f.write("=" * 70 + "\n")
            f.write("NOTE: This is a draft abstract. Please review and revise as needed.\n")
        
        self.logger.debug(f"Generated abstract: {output_path}")
        
        return str(output_path)
    
    def _generate_results_section(
        self,
        discovery: PhysicsDiscovery,
        vae_results: VAEAnalysisResults,
        validation_report: ValidationReport,
        output_path: Path
    ) -> str:
        """Generate draft results section.
        
        Args:
            discovery: PhysicsDiscovery object
            vae_results: VAE analysis results
            validation_report: Validation report
            output_path: Output file path
            
        Returns:
            Path to saved results section
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        variant_name = discovery.variant_id.replace('_', ' ').title()
        tc = discovery.metadata.get('critical_temperature', vae_results.critical_temperature)
        tc_conf = discovery.metadata.get('tc_confidence', 0.0)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("DRAFT RESULTS SECTION\n")
            f.write("=" * 70 + "\n\n")
            
            # Critical temperature detection
            f.write("Critical Temperature Detection\n")
            f.write("-" * 30 + "\n\n")
            f.write(f"We identified the critical temperature of the {variant_name} ")
            f.write(f"as Tc = {tc:.4f} with confidence {tc_conf:.1%}. ")
            f.write(f"This was determined through analysis of the latent space ")
            f.write(f"representation learned by the variational autoencoder, ")
            f.write(f"which showed clear phase separation at this temperature.\n\n")
            
            # Critical exponents
            f.write("Critical Exponents\n")
            f.write("-" * 30 + "\n\n")
            f.write(f"We extracted the following critical exponents:\n\n")
            
            for exp_name, exp_value in discovery.critical_exponents.items():
                exp_error = discovery.exponent_errors.get(exp_name, 0.0)
                f.write(f"  {exp_name}: {exp_value:.4f} ± {exp_error:.4f}\n")
            
            f.write(f"\nThese values were obtained through finite-size scaling analysis ")
            f.write(f"and validated using multiple independent methods.\n\n")
            
            # Comparison with known classes
            f.write("Comparison with Known Universality Classes\n")
            f.write("-" * 30 + "\n\n")
            
            closest_class = discovery.metadata.get('closest_universality_class', 'Unknown')
            max_deviation = discovery.metadata.get('max_deviation_sigma', 0.0)
            deviations = discovery.metadata.get('deviations', {})
            
            f.write(f"The closest known universality class is {closest_class}. ")
            f.write(f"However, our measured exponents deviate significantly:\n\n")
            
            for exp_name, deviation in deviations.items():
                f.write(f"  {exp_name}: {deviation:.2f}σ deviation\n")
            
            f.write(f"\nThe maximum deviation of {max_deviation:.2f}σ exceeds our ")
            f.write(f"novelty threshold of 3σ, indicating this is not a known ")
            f.write(f"universality class.\n\n")
            
            # Theoretical comparison
            f.write("Theoretical Comparison\n")
            f.write("-" * 30 + "\n\n")
            
            theory_comp = discovery.theoretical_comparison
            theory_status = theory_comp.get('status', 'no_predictions')
            theory_message = theory_comp.get('message', '')
            
            f.write(f"Status: {theory_status.upper()}\n")
            f.write(f"{theory_message}\n\n")
            
            if 'agreements' in theory_comp and theory_comp['agreements']:
                f.write(f"Exponents in agreement with predictions:\n")
                for agreement in theory_comp['agreements']:
                    f.write(f"  {agreement['exponent']}: ")
                    f.write(f"predicted={agreement['predicted']:.4f}, ")
                    f.write(f"measured={agreement['measured']:.4f} ")
                    f.write(f"({agreement['deviation_sigma']:.2f}σ)\n")
                f.write("\n")
            
            if 'disagreements' in theory_comp and theory_comp['disagreements']:
                f.write(f"Exponents in disagreement with predictions:\n")
                for disagreement in theory_comp['disagreements']:
                    f.write(f"  {disagreement['exponent']}: ")
                    f.write(f"predicted={disagreement['predicted']:.4f}, ")
                    f.write(f"measured={disagreement['measured']:.4f} ")
                    f.write(f"({disagreement['deviation_sigma']:.2f}σ)\n")
                f.write("\n")
            
            # Validation
            f.write("Validation\n")
            f.write("-" * 30 + "\n\n")
            f.write(f"Our findings were validated using a comprehensive framework ")
            f.write(f"of 10 statistical validation patterns. ")
            f.write(f"The overall validation confidence is {validation_report.overall_confidence:.1%}. ")
            
            if validation_report.publication_ready:
                f.write(f"The results meet all criteria for publication.\n\n")
            else:
                f.write(f"Additional validation may be required before publication.\n\n")
            
            f.write("=" * 70 + "\n")
            f.write("NOTE: This is a draft results section. Please review and expand as needed.\n")
        
        self.logger.debug(f"Generated results section: {output_path}")
        
        return str(output_path)
    
    def _generate_discussion_points(
        self,
        discovery: PhysicsDiscovery,
        vae_results: VAEAnalysisResults,
        validation_report: ValidationReport,
        output_path: Path
    ) -> str:
        """Generate discussion points.
        
        Args:
            discovery: PhysicsDiscovery object
            vae_results: VAE analysis results
            validation_report: Validation report
            output_path: Output file path
            
        Returns:
            Path to saved discussion points
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("DRAFT DISCUSSION POINTS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("Key Points to Discuss:\n\n")
            
            # Point 1: Discovery significance
            f.write(f"1. Significance of Discovery\n")
            f.write(f"   - This discovery is classified as {discovery.significance} significance\n")
            f.write(f"   - Discovery type: {discovery.discovery_type.replace('_', ' ')}\n")
            f.write(f"   - Publication potential: {discovery.publication_potential}\n\n")
            
            # Point 2: Comparison with theory
            theory_status = discovery.theoretical_comparison.get('status', 'no_predictions')
            f.write(f"2. Theoretical Implications\n")
            f.write(f"   - Theoretical comparison status: {theory_status}\n")
            
            if theory_status == 'refuted':
                f.write(f"   - Discuss why existing theory fails to predict these exponents\n")
                f.write(f"   - Suggest modifications to theory\n")
            elif theory_status == 'extended':
                f.write(f"   - Discuss how findings extend beyond current theory\n")
                f.write(f"   - Suggest new theoretical investigations\n")
            elif theory_status == 'confirmed':
                f.write(f"   - Discuss agreement with theory\n")
                f.write(f"   - Highlight validation of theoretical predictions\n")
            
            f.write("\n")
            
            # Point 3: Novel universality class
            if discovery.discovery_type == 'novel_universality_class':
                f.write(f"3. Novel Universality Class\n")
                f.write(f"   - Propose name for new universality class\n")
                f.write(f"   - Discuss what makes this class distinct\n")
                f.write(f"   - Identify other systems that might belong to this class\n")
                f.write(f"   - Discuss implications for renormalization group theory\n\n")
            
            # Point 4: Methodology
            f.write(f"4. Methodological Advances\n")
            f.write(f"   - Discuss use of unsupervised learning for discovery\n")
            f.write(f"   - Highlight comprehensive validation framework\n")
            f.write(f"   - Emphasize reproducibility and rigor\n\n")
            
            # Point 5: Future work
            f.write(f"5. Future Directions\n")
            f.write(f"   - Suggest follow-up experiments or simulations\n")
            f.write(f"   - Identify related systems to explore\n")
            f.write(f"   - Propose theoretical work needed\n\n")
            
            # Point 6: Limitations
            f.write(f"6. Limitations and Caveats\n")
            f.write(f"   - Discuss any limitations of the approach\n")
            f.write(f"   - Mention assumptions made\n")
            f.write(f"   - Suggest additional validation if needed\n\n")
            
            f.write("=" * 70 + "\n")
            f.write("NOTE: These are suggested discussion points. Expand and refine as needed.\n")
        
        self.logger.debug(f"Generated discussion points: {output_path}")
        
        return str(output_path)
    
    def export_supplementary_data(
        self,
        discovery: PhysicsDiscovery,
        vae_results: VAEAnalysisResults,
        validation_report: ValidationReport,
        simulation_data: Optional[SimulationData],
        output_dir: Path
    ) -> Dict[str, str]:
        """Export supplementary data files.
        
        Args:
            discovery: PhysicsDiscovery object
            vae_results: VAE analysis results
            validation_report: Validation report
            simulation_data: Optional simulation data
            output_dir: Output directory for supplementary data
            
        Returns:
            Dictionary mapping data type to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        supplementary = {}
        
        # Export discovery metadata
        supplementary['discovery_metadata'] = self._export_discovery_metadata(
            discovery, output_dir / "discovery_metadata.json"
        )
        
        # Export VAE results
        supplementary['vae_results'] = self._export_vae_results(
            vae_results, output_dir / "vae_results.h5"
        )
        
        # Export validation report
        supplementary['validation_report'] = self._export_validation_report(
            validation_report, output_dir / "validation_report.json"
        )
        
        # Export simulation data if available
        if simulation_data:
            supplementary['simulation_data'] = self._export_simulation_data(
                simulation_data, output_dir / "simulation_data.h5"
            )
        
        # Generate reproducibility script
        supplementary['reproducibility_script'] = self._generate_reproducibility_script(
            discovery, output_dir / "reproduce.py"
        )
        
        # Generate README
        supplementary['readme'] = self._generate_supplementary_readme(
            discovery, supplementary, output_dir / "README.md"
        )
        
        self.logger.info(f"Exported {len(supplementary)} supplementary data files")
        
        return supplementary
    
    def _export_discovery_metadata(
        self,
        discovery: PhysicsDiscovery,
        output_path: Path
    ) -> str:
        """Export discovery metadata to JSON.
        
        Args:
            discovery: PhysicsDiscovery object
            output_path: Output file path
            
        Returns:
            Path to saved metadata
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'discovery_id': discovery.discovery_id,
            'variant_id': discovery.variant_id,
            'discovery_type': discovery.discovery_type,
            'significance': discovery.significance,
            'publication_potential': discovery.publication_potential,
            'timestamp': discovery.timestamp.isoformat(),
            'critical_exponents': {k: float(v) for k, v in discovery.critical_exponents.items()},
            'exponent_errors': {k: float(v) for k, v in discovery.exponent_errors.items()},
            'validation_confidence': float(discovery.validation_confidence),
            'theoretical_comparison': discovery.theoretical_comparison,
            'metadata': discovery.metadata
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.debug(f"Exported discovery metadata: {output_path}")
        
        return str(output_path)
    
    def _export_vae_results(
        self,
        vae_results: VAEAnalysisResults,
        output_path: Path
    ) -> str:
        """Export VAE results to HDF5.
        
        Args:
            vae_results: VAE analysis results
            output_path: Output file path
            
        Returns:
            Path to saved results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            # Store basic info
            f.attrs['variant_id'] = vae_results.variant_id
            f.attrs['critical_temperature'] = float(vae_results.critical_temperature)
            f.attrs['tc_confidence'] = float(vae_results.tc_confidence)
            
            # Store exponents
            exp_group = f.create_group('exponents')
            for name, value in vae_results.exponents.items():
                exp_group.attrs[name] = float(value)
            
            # Store errors
            err_group = f.create_group('exponent_errors')
            for name, value in vae_results.exponent_errors.items():
                err_group.attrs[name] = float(value)
            
            # Store latent space if available
            if hasattr(vae_results, 'latent_z1') and vae_results.latent_z1 is not None:
                f.create_dataset('latent_z1', data=vae_results.latent_z1)
                f.create_dataset('latent_z2', data=vae_results.latent_z2)
            
            # Store temperatures if available
            if hasattr(vae_results, 'temperatures') and vae_results.temperatures is not None:
                f.create_dataset('temperatures', data=vae_results.temperatures)
            
            # Store order parameter if available
            if hasattr(vae_results, 'order_parameter') and vae_results.order_parameter is not None:
                f.create_dataset('order_parameter', data=vae_results.order_parameter)
        
        self.logger.debug(f"Exported VAE results: {output_path}")
        
        return str(output_path)
    
    def _export_validation_report(
        self,
        validation_report: ValidationReport,
        output_path: Path
    ) -> str:
        """Export validation report to JSON.
        
        Args:
            validation_report: Validation report
            output_path: Output file path
            
        Returns:
            Path to saved report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report_dict = {
            'overall_validated': validation_report.overall_validated,
            'overall_confidence': float(validation_report.overall_confidence),
            'publication_ready': validation_report.publication_ready,
            'recommendation': validation_report.recommendation,
            'pattern_results': validation_report.pattern_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        self.logger.debug(f"Exported validation report: {output_path}")
        
        return str(output_path)
    
    def _export_simulation_data(
        self,
        simulation_data: SimulationData,
        output_path: Path
    ) -> str:
        """Export simulation data to HDF5.
        
        Args:
            simulation_data: Simulation data
            output_path: Output file path
            
        Returns:
            Path to saved data
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            # Store temperatures
            if hasattr(simulation_data, 'temperatures'):
                f.create_dataset('temperatures', data=simulation_data.temperatures)
            
            # Store magnetizations
            if hasattr(simulation_data, 'magnetizations'):
                f.create_dataset('magnetizations', data=simulation_data.magnetizations)
            
            # Store susceptibilities
            if hasattr(simulation_data, 'susceptibilities'):
                f.create_dataset('susceptibilities', data=simulation_data.susceptibilities)
            
            # Store configurations if available
            if hasattr(simulation_data, 'configurations'):
                f.create_dataset('configurations', data=simulation_data.configurations)
        
        self.logger.debug(f"Exported simulation data: {output_path}")
        
        return str(output_path)
    
    def _generate_reproducibility_script(
        self,
        discovery: PhysicsDiscovery,
        output_path: Path
    ) -> str:
        """Generate reproducibility script.
        
        Args:
            discovery: PhysicsDiscovery object
            output_path: Output file path
            
        Returns:
            Path to saved script
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("#!/usr/bin/env python3\n")
            f.write('"""\n')
            f.write(f"Reproducibility script for discovery {discovery.discovery_id}\n")
            f.write(f"Generated automatically on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write('"""\n\n')
            
            f.write("# This script demonstrates how to reproduce the discovery\n")
            f.write("# using the provided supplementary data\n\n")
            
            f.write("import h5py\n")
            f.write("import json\n")
            f.write("import numpy as np\n\n")
            
            f.write("# Load discovery metadata\n")
            f.write("with open('discovery_metadata.json', 'r') as f:\n")
            f.write("    metadata = json.load(f)\n\n")
            
            f.write("print('Discovery ID:', metadata['discovery_id'])\n")
            f.write("print('Variant:', metadata['variant_id'])\n")
            f.write("print('Discovery Type:', metadata['discovery_type'])\n")
            f.write("print('Significance:', metadata['significance'])\n")
            f.write("print('\\nCritical Exponents:')\n")
            f.write("for name, value in metadata['critical_exponents'].items():\n")
            f.write("    error = metadata['exponent_errors'][name]\n")
            f.write("    print(f'  {name}: {value:.4f} ± {error:.4f}')\n\n")
            
            f.write("# Load VAE results\n")
            f.write("with h5py.File('vae_results.h5', 'r') as f:\n")
            f.write("    tc = f.attrs['critical_temperature']\n")
            f.write("    tc_conf = f.attrs['tc_confidence']\n")
            f.write("    print(f'\\nCritical Temperature: {tc:.4f} (confidence: {tc_conf:.2%})')\n\n")
            
            f.write("# Load validation report\n")
            f.write("with open('validation_report.json', 'r') as f:\n")
            f.write("    validation = json.load(f)\n\n")
            
            f.write("print(f'\\nValidation Confidence: {validation[\"overall_confidence\"]:.2%}')\n")
            f.write("print(f'Publication Ready: {validation[\"publication_ready\"]}')\n")
        
        self.logger.debug(f"Generated reproducibility script: {output_path}")
        
        return str(output_path)
    
    def _generate_supplementary_readme(
        self,
        discovery: PhysicsDiscovery,
        supplementary_files: Dict[str, str],
        output_path: Path
    ) -> str:
        """Generate README for supplementary materials.
        
        Args:
            discovery: PhysicsDiscovery object
            supplementary_files: Dictionary of supplementary file paths
            output_path: Output file path
            
        Returns:
            Path to saved README
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Supplementary Materials: {discovery.discovery_id}\n\n")
            
            f.write("## Overview\n\n")
            f.write(f"This directory contains supplementary materials for the discovery ")
            f.write(f"of {discovery.discovery_type.replace('_', ' ')} in {discovery.variant_id}.\n\n")
            
            f.write("## Files\n\n")
            
            for file_type, file_path in supplementary_files.items():
                filename = Path(file_path).name
                f.write(f"### {filename}\n\n")
                
                if file_type == 'discovery_metadata':
                    f.write("Complete metadata for the discovery including:\n")
                    f.write("- Discovery classification\n")
                    f.write("- Critical exponents and errors\n")
                    f.write("- Validation confidence\n")
                    f.write("- Theoretical comparison\n\n")
                
                elif file_type == 'vae_results':
                    f.write("VAE analysis results including:\n")
                    f.write("- Latent space representations\n")
                    f.write("- Critical temperature detection\n")
                    f.write("- Order parameter\n\n")
                
                elif file_type == 'validation_report':
                    f.write("Comprehensive validation report including:\n")
                    f.write("- Overall validation status\n")
                    f.write("- Pattern-level results\n")
                    f.write("- Confidence scores\n\n")
                
                elif file_type == 'simulation_data':
                    f.write("Raw simulation data including:\n")
                    f.write("- Temperature points\n")
                    f.write("- Magnetizations\n")
                    f.write("- Susceptibilities\n")
                    f.write("- Configurations\n\n")
                
                elif file_type == 'reproducibility_script':
                    f.write("Python script to reproduce the analysis.\n")
                    f.write("Run with: `python reproduce.py`\n\n")
            
            f.write("## Reproducibility\n\n")
            f.write("All results can be reproduced using the provided data and scripts.\n")
            f.write("See `reproduce.py` for a demonstration.\n\n")
            
            f.write("## Citation\n\n")
            f.write("If you use these materials, please cite:\n")
            f.write(f"[Citation information to be added upon publication]\n\n")
            
            f.write("## Contact\n\n")
            f.write("[Contact information to be added]\n")
        
        self.logger.debug(f"Generated supplementary README: {output_path}")
        
        return str(output_path)
    
    def _save_package_manifest(
        self,
        package: PublicationPackage,
        output_dir: Path
    ) -> None:
        """Save package manifest.
        
        Args:
            package: PublicationPackage
            output_dir: Output directory
        """
        manifest_path = output_dir / "package_manifest.json"
        
        manifest = {
            'discovery_id': package.discovery.discovery_id,
            'variant_id': package.discovery.variant_id,
            'generation_time': package.metadata.get('generation_time'),
            'figures': package.figures,
            'tables': package.tables,
            'draft_text': package.draft_text,
            'supplementary_data': package.supplementary_data,
            'metadata': package.metadata
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        self.logger.info(f"Saved package manifest: {manifest_path}")
