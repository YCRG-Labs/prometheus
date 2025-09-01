"""
Publication Materials Generator

This module provides a comprehensive system for generating publication-ready
materials including training diagnostics, reconstruction analysis, and comparison
studies for the Prometheus VAE project.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import torch
import json
from datetime import datetime

from ..utils.logging_utils import get_logger
from .training_diagnostics import TrainingDiagnostics, TrainingMetrics
from .reconstruction_analysis import ReconstructionAnalyzer, ReconstructionMetrics
from .comparison_studies import ComparisonStudies, ComparisonResult, AblationResult
from .latent_analysis import LatentRepresentation
from .order_parameter_discovery import OrderParameterCandidate
from .phase_detection import PhaseDetectionResult
from .physics_validation import ValidationMetrics


@dataclass
class PublicationDataset:
    """Container for all data needed for publication materials."""
    # Training data
    training_history: List[TrainingMetrics]
    
    # Model and reconstructions
    trained_model: torch.nn.Module
    original_configs: np.ndarray
    reconstructed_configs: np.ndarray
    latent_samples: np.ndarray
    
    # Physics analysis
    temperatures: np.ndarray
    magnetizations: np.ndarray
    latent_representation: LatentRepresentation
    order_parameter_candidates: List[OrderParameterCandidate]
    phase_detection_result: PhaseDetectionResult
    validation_metrics: ValidationMetrics
    
    # Comparison data
    baseline_results: Optional[List[ComparisonResult]] = None
    ablation_results: Optional[List[AblationResult]] = None
    architecture_results: Optional[List[ComparisonResult]] = None
    significance_results: Optional[Dict[str, Any]] = None


class PublicationMaterialsGenerator:
    """
    Comprehensive publication materials generator.
    
    Integrates training diagnostics, reconstruction analysis, and comparison
    studies to generate a complete set of publication-ready figures and reports.
    """
    
    def __init__(self):
        """Initialize publication materials generator."""
        self.logger = get_logger(__name__)
        
        # Initialize component analyzers
        self.training_diagnostics = TrainingDiagnostics()
        self.reconstruction_analyzer = ReconstructionAnalyzer()
        self.comparison_studies = ComparisonStudies()
        
        # Publication settings
        plt.style.use('seaborn-v0_8-whitegrid')
        
    def load_training_history(self, training_history: List[Dict[str, Any]]) -> None:
        """
        Load training history into diagnostics system.
        
        Args:
            training_history: List of training epoch data
        """
        self.logger.info("Loading training history for diagnostics")
        
        for epoch_data in training_history:
            self.training_diagnostics.record_epoch_metrics(
                epoch=epoch_data.get('epoch', 0),
                train_loss=epoch_data.get('train_loss', 0.0),
                val_loss=epoch_data.get('val_loss'),
                reconstruction_loss=epoch_data.get('reconstruction_loss', 0.0),
                kl_loss=epoch_data.get('kl_loss', 0.0),
                learning_rate=epoch_data.get('learning_rate', 0.001),
                gradient_norm=epoch_data.get('gradient_norm'),
                latent_samples=epoch_data.get('latent_samples')
            )
    
    def generate_training_diagnostics(self, 
                                    output_dir: str = 'results/publication/training') -> Dict[str, str]:
        """
        Generate comprehensive training diagnostics.
        
        Args:
            output_dir: Output directory for training diagnostics
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        self.logger.info("Generating training diagnostics for publication")
        
        return self.training_diagnostics.generate_comprehensive_report(output_dir)
    
    def generate_reconstruction_analysis(self,
                                       dataset: PublicationDataset,
                                       output_dir: str = 'results/publication/reconstruction') -> Dict[str, str]:
        """
        Generate comprehensive reconstruction quality analysis.
        
        Args:
            dataset: Publication dataset with all required data
            output_dir: Output directory for reconstruction analysis
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        self.logger.info("Generating reconstruction analysis for publication")
        
        return self.reconstruction_analyzer.generate_reconstruction_report(
            model=dataset.trained_model,
            original_configs=dataset.original_configs,
            reconstructed_configs=dataset.reconstructed_configs,
            latent_samples=dataset.latent_samples,
            temperatures=dataset.temperatures,
            magnetizations=dataset.magnetizations,
            output_dir=output_dir
        )
    
    def generate_comparison_studies(self,
                                  dataset: PublicationDataset,
                                  output_dir: str = 'results/publication/comparison') -> Dict[str, str]:
        """
        Generate comprehensive comparison and ablation studies.
        
        Args:
            dataset: Publication dataset with comparison data
            output_dir: Output directory for comparison studies
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        self.logger.info("Generating comparison studies for publication")
        
        if not dataset.baseline_results or not dataset.ablation_results:
            self.logger.warning("Baseline or ablation results not available, generating with available data")
        
        # Create VAE result for comparison
        vae_result = ComparisonResult(
            method_name="VAE",
            latent_representation=dataset.latent_samples,
            order_parameter_correlation=dataset.order_parameter_candidates[0].correlation_with_magnetization.correlation_coefficient if dataset.order_parameter_candidates else 0.0,
            critical_temperature=dataset.phase_detection_result.critical_temperature,
            physics_consistency_score=dataset.validation_metrics.physics_consistency_score,
            computational_cost=1.0,  # Reference cost
            additional_metrics={}
        )
        
        return self.comparison_studies.generate_comparison_report(
            vae_result=vae_result,
            baseline_results=dataset.baseline_results or [],
            ablation_results=dataset.ablation_results or [],
            significance_results=dataset.significance_results or {},
            architecture_results=dataset.architecture_results or [],
            output_dir=output_dir
        )
    
    def create_main_results_figure(self, 
                                 dataset: PublicationDataset,
                                 figsize: Tuple[int, int] = (20, 12)) -> Figure:
        """
        Create main results figure combining key findings.
        
        Args:
            dataset: Publication dataset
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure with main results
        """
        self.logger.info("Creating main results figure for publication")
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Panel A: Latent space with temperature coloring
        ax_a = fig.add_subplot(gs[0, 0])
        scatter_a = ax_a.scatter(
            dataset.latent_representation.z1,
            dataset.latent_representation.z2,
            c=dataset.temperatures,
            cmap='coolwarm',
            alpha=0.6,
            s=20
        )
        cbar_a = plt.colorbar(scatter_a, ax=ax_a)
        cbar_a.set_label('Temperature', fontsize=10)
        ax_a.set_xlabel('z₁', fontsize=12)
        ax_a.set_ylabel('z₂', fontsize=12)
        ax_a.set_title('A. Latent Space\n(Temperature)', fontsize=14, fontweight='bold')
        ax_a.grid(True, alpha=0.3)
        
        # Panel B: Latent space with magnetization coloring
        ax_b = fig.add_subplot(gs[0, 1])
        scatter_b = ax_b.scatter(
            dataset.latent_representation.z1,
            dataset.latent_representation.z2,
            c=np.abs(dataset.magnetizations),
            cmap='viridis',
            alpha=0.6,
            s=20
        )
        cbar_b = plt.colorbar(scatter_b, ax=ax_b)
        cbar_b.set_label('|Magnetization|', fontsize=10)
        ax_b.set_xlabel('z₁', fontsize=12)
        ax_b.set_ylabel('z₂', fontsize=12)
        ax_b.set_title('B. Latent Space\n(Magnetization)', fontsize=14, fontweight='bold')
        ax_b.grid(True, alpha=0.3)
        
        # Panel C: Order parameter correlation
        ax_c = fig.add_subplot(gs[0, 2])
        if dataset.order_parameter_candidates:
            best_candidate = dataset.order_parameter_candidates[0]
            if best_candidate.latent_dimension == 'z1':
                primary_latent = dataset.latent_representation.z1
            else:
                primary_latent = dataset.latent_representation.z2
            
            ax_c.scatter(
                np.abs(dataset.magnetizations),
                primary_latent,
                alpha=0.6,
                s=20,
                c=dataset.temperatures,
                cmap='coolwarm'
            )
            
            correlation = best_candidate.correlation_with_magnetization.correlation_coefficient
            ax_c.text(0.05, 0.95, f'r = {correlation:.3f}', 
                     transform=ax_c.transAxes, fontsize=12, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax_c.set_xlabel('|Magnetization|', fontsize=12)
        ax_c.set_ylabel('Order Parameter', fontsize=12)
        ax_c.set_title('C. Order Parameter\nCorrelation', fontsize=14, fontweight='bold')
        ax_c.grid(True, alpha=0.3)
        
        # Panel D: Phase detection
        ax_d = fig.add_subplot(gs[0, 3])
        
        # Temperature histogram with critical temperature
        temp_bins = np.linspace(dataset.temperatures.min(), dataset.temperatures.max(), 30)
        ax_d.hist(dataset.temperatures, bins=temp_bins, alpha=0.7, density=True, 
                 color='lightblue', edgecolor='black')
        
        discovered_tc = dataset.phase_detection_result.critical_temperature
        theoretical_tc = 2.269
        
        ax_d.axvline(discovered_tc, color='red', linewidth=3, 
                    label=f'Discovered: {discovered_tc:.3f}')
        ax_d.axvline(theoretical_tc, color='black', linewidth=3, linestyle='--',
                    label=f'Theoretical: {theoretical_tc:.3f}')
        
        ax_d.set_xlabel('Temperature', fontsize=12)
        ax_d.set_ylabel('Density', fontsize=12)
        ax_d.set_title('D. Critical Temperature\nDetection', fontsize=14, fontweight='bold')
        ax_d.legend(fontsize=10)
        ax_d.grid(True, alpha=0.3)
        
        # Panel E: Training loss curves
        ax_e = fig.add_subplot(gs[1, :2])
        if self.training_diagnostics.metrics_history:
            epochs = [m.epoch for m in self.training_diagnostics.metrics_history]
            train_losses = [m.train_loss for m in self.training_diagnostics.metrics_history]
            recon_losses = [m.reconstruction_loss for m in self.training_diagnostics.metrics_history]
            kl_losses = [m.kl_loss for m in self.training_diagnostics.metrics_history]
            
            ax_e.plot(epochs, train_losses, 'b-', linewidth=2, label='Total Loss', alpha=0.8)
            ax_e.plot(epochs, recon_losses, 'g-', linewidth=2, label='Reconstruction', alpha=0.8)
            ax_e.plot(epochs, kl_losses, 'm-', linewidth=2, label='KL Divergence', alpha=0.8)
            
            ax_e.set_xlabel('Epoch', fontsize=12)
            ax_e.set_ylabel('Loss', fontsize=12)
            ax_e.set_title('E. Training Progress', fontsize=14, fontweight='bold')
            ax_e.legend()
            ax_e.grid(True, alpha=0.3)
            ax_e.set_yscale('log')
        
        # Panel F: Reconstruction examples
        ax_f = fig.add_subplot(gs[1, 2:])
        
        # Show a few reconstruction examples
        n_examples = min(4, len(dataset.original_configs))
        example_indices = np.linspace(0, len(dataset.original_configs)-1, n_examples, dtype=int)
        
        for i, idx in enumerate(example_indices):
            # Original
            ax_orig = plt.subplot(2, n_examples*2, n_examples*2 + i*2 + 1)
            ax_orig.imshow(dataset.original_configs[idx].squeeze(), cmap='RdBu', vmin=-1, vmax=1)
            ax_orig.set_title(f'T={dataset.temperatures[idx]:.2f}', fontsize=10)
            ax_orig.axis('off')
            
            # Reconstructed
            ax_recon = plt.subplot(2, n_examples*2, n_examples*2 + i*2 + 2)
            ax_recon.imshow(dataset.reconstructed_configs[idx].squeeze(), cmap='RdBu', vmin=-1, vmax=1)
            ax_recon.set_title('Reconstructed', fontsize=10)
            ax_recon.axis('off')
        
        # Add panel F title
        fig.text(0.75, 0.65, 'F. Reconstruction Examples', fontsize=14, fontweight='bold', ha='center')
        
        # Panel G: Physics validation summary
        ax_g = fig.add_subplot(gs[2, :2])
        
        # Validation metrics radar chart
        categories = ['Order Parameter\nCorrelation', 'Critical Temperature\nAccuracy', 
                     'Energy\nConservation', 'Magnetization\nConservation']
        
        values = [
            abs(dataset.validation_metrics.order_parameter_correlation),
            1.0 - (dataset.validation_metrics.critical_temperature_relative_error / 100.0),
            dataset.validation_metrics.energy_conservation_score,
            dataset.validation_metrics.magnetization_conservation_score
        ]
        
        # Ensure values are in [0, 1] range
        values = [max(0, min(1, v)) for v in values]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax_g = plt.subplot(2, 4, 5, projection='polar')
        ax_g.plot(angles, values, 'o-', linewidth=2, color='blue')
        ax_g.fill(angles, values, alpha=0.25, color='blue')
        ax_g.set_xticks(angles[:-1])
        ax_g.set_xticklabels(categories, fontsize=8)
        ax_g.set_ylim(0, 1)
        ax_g.set_title('G. Physics Validation', fontsize=14, fontweight='bold', pad=20)
        
        # Panel H: Overall physics consistency
        ax_h = fig.add_subplot(gs[2, 2])
        
        overall_score = dataset.validation_metrics.physics_consistency_score
        
        # Gauge visualization
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        ax_h = plt.subplot(2, 4, 7, projection='polar')
        ax_h.plot(theta, r, 'k-', linewidth=3)
        
        # Color based on score
        if overall_score >= 0.8:
            color = 'green'
            status = 'EXCELLENT'
        elif overall_score >= 0.6:
            color = 'orange'
            status = 'GOOD'
        else:
            color = 'red'
            status = 'NEEDS IMPROVEMENT'
        
        score_theta = np.linspace(0, np.pi * overall_score, 50)
        score_r = np.ones_like(score_theta)
        ax_h.fill_between(score_theta, 0, score_r, alpha=0.7, color=color)
        
        ax_h.text(np.pi/2, 0.5, f'{overall_score:.3f}\n{status}', 
                 ha='center', va='center', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax_h.set_ylim(0, 1)
        ax_h.set_theta_zero_location('W')
        ax_h.set_thetagrids([0, 45, 90, 135, 180], ['0.0', '0.25', '0.5', '0.75', '1.0'])
        ax_h.set_title('H. Overall Score', fontsize=14, fontweight='bold', pad=20)
        
        # Panel I: Method comparison (if available)
        ax_i = fig.add_subplot(gs[2, 3])
        
        if dataset.baseline_results:
            methods = ['VAE'] + [r.method_name for r in dataset.baseline_results]
            vae_corr = dataset.order_parameter_candidates[0].correlation_with_magnetization.correlation_coefficient if dataset.order_parameter_candidates else 0.0
            correlations = [vae_corr] + [r.order_parameter_correlation for r in dataset.baseline_results]
            
            bars = ax_i.bar(methods, correlations, alpha=0.7)
            ax_i.set_ylabel('Order Parameter Correlation', fontsize=12)
            ax_i.set_title('I. Method Comparison', fontsize=14, fontweight='bold')
            ax_i.grid(True, alpha=0.3)
            
            # Highlight VAE
            bars[0].set_color('red')
            
            # Add value annotations
            for bar, corr in zip(bars, correlations):
                height = bar.get_height()
                ax_i.annotate(f'{corr:.3f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=10)
        else:
            ax_i.text(0.5, 0.5, 'No Baseline\nComparison\nAvailable', 
                     transform=ax_i.transAxes, ha='center', va='center',
                     fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
            ax_i.set_title('I. Method Comparison', fontsize=14, fontweight='bold')
        
        plt.suptitle('Prometheus: Unsupervised Discovery of Order Parameters in 2D Ising Model', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        return fig
    
    def create_supplementary_figures(self, 
                                   dataset: PublicationDataset,
                                   output_dir: str = 'results/publication/supplementary') -> Dict[str, str]:
        """
        Create supplementary figures for detailed analysis.
        
        Args:
            dataset: Publication dataset
            output_dir: Output directory for supplementary figures
            
        Returns:
            Dictionary mapping figure names to file paths
        """
        self.logger.info("Creating supplementary figures")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_figures = {}
        
        # Supplementary Figure 1: Detailed latent space analysis
        if hasattr(self, 'visualization_system'):
            fig_s1 = self.visualization_system.plot_latent_space_analysis(
                dataset.latent_representation,
                dataset.order_parameter_candidates
            )
            
            s1_path = output_path / "supplementary_figure_1_latent_analysis.png"
            fig_s1.savefig(s1_path, dpi=300, bbox_inches='tight')
            saved_figures['supplementary_1'] = str(s1_path)
            plt.close(fig_s1)
        
        # Supplementary Figure 2: Training diagnostics
        if self.training_diagnostics.metrics_history:
            fig_s2 = self.training_diagnostics.plot_detailed_loss_curves()
            
            s2_path = output_path / "supplementary_figure_2_training_diagnostics.png"
            fig_s2.savefig(s2_path, dpi=300, bbox_inches='tight')
            saved_figures['supplementary_2'] = str(s2_path)
            plt.close(fig_s2)
        
        # Supplementary Figure 3: Reconstruction quality
        fig_s3 = self.reconstruction_analyzer.create_comparison_grid(
            dataset.original_configs,
            dataset.reconstructed_configs,
            dataset.temperatures,
            dataset.magnetizations
        )
        
        s3_path = output_path / "supplementary_figure_3_reconstruction_quality.png"
        fig_s3.savefig(s3_path, dpi=300, bbox_inches='tight')
        saved_figures['supplementary_3'] = str(s3_path)
        plt.close(fig_s3)
        
        return saved_figures
    
    def generate_complete_publication_package(self,
                                            dataset: PublicationDataset,
                                            output_dir: str = 'results/publication') -> Dict[str, Any]:
        """
        Generate complete publication package with all materials.
        
        Args:
            dataset: Complete publication dataset
            output_dir: Base output directory
            
        Returns:
            Dictionary with all generated materials and their paths
        """
        self.logger.info("Generating complete publication package")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        package = {
            'generation_time': datetime.now().isoformat(),
            'main_figures': {},
            'supplementary_figures': {},
            'analysis_reports': {},
            'data_summaries': {}
        }
        
        try:
            # Generate main results figure
            main_fig = self.create_main_results_figure(dataset)
            main_path = output_path / "main_results_figure.png"
            main_fig.savefig(main_path, dpi=300, bbox_inches='tight')
            main_fig.savefig(output_path / "main_results_figure.pdf", format='pdf', bbox_inches='tight')
            package['main_figures']['main_results'] = str(main_path)
            plt.close(main_fig)
            
            # Generate training diagnostics
            training_plots = self.generate_training_diagnostics(
                str(output_path / 'training_diagnostics')
            )
            package['analysis_reports']['training_diagnostics'] = training_plots
            
            # Generate reconstruction analysis
            reconstruction_plots = self.generate_reconstruction_analysis(
                dataset, str(output_path / 'reconstruction_analysis')
            )
            package['analysis_reports']['reconstruction_analysis'] = reconstruction_plots
            
            # Generate comparison studies
            comparison_plots = self.generate_comparison_studies(
                dataset, str(output_path / 'comparison_studies')
            )
            package['analysis_reports']['comparison_studies'] = comparison_plots
            
            # Generate supplementary figures
            supplementary_figs = self.create_supplementary_figures(
                dataset, str(output_path / 'supplementary')
            )
            package['supplementary_figures'] = supplementary_figs
            
            # Generate data summary
            data_summary = self._create_data_summary(dataset)
            summary_path = output_path / "data_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(data_summary, f, indent=2, default=str)
            package['data_summaries']['main_summary'] = str(summary_path)
            
            # Generate publication checklist
            checklist_path = self._create_publication_checklist(package, output_path)
            package['checklist'] = str(checklist_path)
            
            self.logger.info(f"Complete publication package generated in {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating publication package: {e}")
            raise
        
        return package
    
    def _create_data_summary(self, dataset: PublicationDataset) -> Dict[str, Any]:
        """Create comprehensive data summary for publication."""
        summary = {
            'dataset_info': {
                'n_configurations': len(dataset.original_configs),
                'temperature_range': [float(dataset.temperatures.min()), float(dataset.temperatures.max())],
                'magnetization_range': [float(dataset.magnetizations.min()), float(dataset.magnetizations.max())],
                'lattice_size': list(dataset.original_configs.shape[1:]),
                'latent_dimensions': dataset.latent_samples.shape[1] if dataset.latent_samples is not None else 0
            },
            'physics_results': {
                'discovered_critical_temperature': float(dataset.phase_detection_result.critical_temperature),
                'theoretical_critical_temperature': 2.269,
                'critical_temperature_error_percent': float(dataset.validation_metrics.critical_temperature_relative_error),
                'order_parameter_correlation': float(dataset.order_parameter_candidates[0].correlation_with_magnetization.correlation_coefficient) if dataset.order_parameter_candidates else 0.0,
                'physics_consistency_score': float(dataset.validation_metrics.physics_consistency_score)
            },
            'model_performance': {
                'final_training_loss': float(self.training_diagnostics.metrics_history[-1].train_loss) if self.training_diagnostics.metrics_history else 0.0,
                'final_reconstruction_loss': float(self.training_diagnostics.metrics_history[-1].reconstruction_loss) if self.training_diagnostics.metrics_history else 0.0,
                'final_kl_loss': float(self.training_diagnostics.metrics_history[-1].kl_loss) if self.training_diagnostics.metrics_history else 0.0,
                'training_epochs': len(self.training_diagnostics.metrics_history) if self.training_diagnostics.metrics_history else 0
            }
        }
        
        # Add comparison results if available
        if dataset.baseline_results:
            summary['baseline_comparisons'] = {}
            for baseline in dataset.baseline_results:
                summary['baseline_comparisons'][baseline.method_name] = {
                    'order_parameter_correlation': float(baseline.order_parameter_correlation),
                    'physics_consistency_score': float(baseline.physics_consistency_score),
                    'computational_cost': float(baseline.computational_cost)
                }
        
        # Add ablation results if available
        if dataset.ablation_results:
            summary['ablation_study'] = {}
            for ablation in dataset.ablation_results:
                summary['ablation_study'][f'beta_{ablation.parameter_value}'] = {
                    'order_parameter_correlation': float(ablation.order_parameter_correlation),
                    'critical_temperature_error': float(ablation.critical_temperature_error),
                    'physics_consistency_score': float(ablation.physics_consistency_score)
                }
        
        return summary
    
    def _create_publication_checklist(self, package: Dict[str, Any], output_path: Path) -> str:
        """Create publication checklist with all generated materials."""
        checklist_path = output_path / "publication_checklist.md"
        
        with open(checklist_path, 'w') as f:
            f.write("# Prometheus Publication Materials Checklist\n\n")
            f.write(f"Generated on: {package['generation_time']}\n\n")
            
            f.write("## Main Figures\n")
            for name, path in package['main_figures'].items():
                f.write(f"- [x] {name}: `{path}`\n")
            
            f.write("\n## Analysis Reports\n")
            for category, plots in package['analysis_reports'].items():
                f.write(f"### {category.replace('_', ' ').title()}\n")
                for plot_name, plot_path in plots.items():
                    f.write(f"- [x] {plot_name}: `{plot_path}`\n")
            
            f.write("\n## Supplementary Figures\n")
            for name, path in package['supplementary_figures'].items():
                f.write(f"- [x] {name}: `{path}`\n")
            
            f.write("\n## Data Summaries\n")
            for name, path in package['data_summaries'].items():
                f.write(f"- [x] {name}: `{path}`\n")
            
            f.write("\n## Publication Readiness Checklist\n")
            f.write("- [ ] All figures reviewed for clarity and accuracy\n")
            f.write("- [ ] Figure captions written\n")
            f.write("- [ ] Statistical significance tests completed\n")
            f.write("- [ ] Comparison with baseline methods included\n")
            f.write("- [ ] Ablation studies documented\n")
            f.write("- [ ] Physics validation metrics meet requirements\n")
            f.write("- [ ] Code and data availability statements prepared\n")
            f.write("- [ ] Reproducibility information documented\n")
        
        return str(checklist_path)