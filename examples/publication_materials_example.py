"""
Publication Materials Generation Example

This example demonstrates how to generate comprehensive publication-ready
materials for the Prometheus VAE project, including training diagnostics,
reconstruction analysis, and comparison studies.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.publication_materials import (
    PublicationMaterialsGenerator, 
    PublicationDataset
)
from src.analysis.training_diagnostics import TrainingMetrics
from src.analysis.latent_analysis import LatentRepresentation
from src.analysis.order_parameter_discovery import (
    OrderParameterCandidate, 
    CorrelationResult
)
from src.analysis.phase_detection import PhaseDetectionResult
from src.analysis.physics_validation import ValidationMetrics
from src.analysis.comparison_studies import ComparisonResult, AblationResult
from src.models.vae import ConvolutionalVAE



def create_mock_training_history(n_epochs: int = 100) -> list:
    """Create mock training history for demonstration."""
    history = []
    
    for epoch in range(n_epochs):
        # Simulate decreasing loss with some noise
        base_loss = 2.0 * np.exp(-epoch / 30) + 0.1
        noise = np.random.normal(0, 0.05)
        
        train_loss = base_loss + noise
        val_loss = base_loss + noise * 1.2 + 0.02
        
        # Decompose loss
        reconstruction_loss = train_loss * 0.7
        kl_loss = train_loss * 0.3
        
        # Learning rate schedule
        if epoch < 20:
            lr = 0.001
        elif epoch < 60:
            lr = 0.0005
        else:
            lr = 0.0001
        
        # Mock latent samples for some epochs
        latent_samples = None
        if epoch % 10 == 0:
            latent_samples = np.random.randn(100, 2)
        
        # Mock gradient norm (decreasing over time)
        gradient_norm = 2.0 * np.exp(-epoch / 20) + 0.1 + np.random.normal(0, 0.05)
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss,
            'learning_rate': lr,
            'gradient_norm': gradient_norm,
            'latent_samples': latent_samples
        })
    
    return history


def create_mock_dataset() -> PublicationDataset:
    """Create mock dataset for demonstration."""
    logging.info("Creating mock dataset for publication materials demo")
    
    # Generate mock data
    n_samples = 1000
    lattice_size = 32
    
    # Temperature range around critical temperature
    temperatures = np.linspace(1.5, 3.0, n_samples)
    
    # Mock spin configurations (random for demo)
    original_configs = np.random.choice([-1, 1], size=(n_samples, lattice_size, lattice_size))
    
    # Mock reconstructions (with some noise)
    reconstructed_configs = original_configs + np.random.normal(0, 0.1, original_configs.shape)
    reconstructed_configs = np.clip(reconstructed_configs, -1, 1)
    
    # Mock magnetizations (temperature-dependent)
    critical_temp = 2.269
    magnetizations = np.tanh(2 * (critical_temp - temperatures)) + np.random.normal(0, 0.1, n_samples)
    
    # Mock latent samples (2D with temperature structure)
    z1 = magnetizations + np.random.normal(0, 0.2, n_samples)
    z2 = np.random.normal(0, 0.5, n_samples)
    latent_samples = np.column_stack([z1, z2])
    
    # Mock energies (temperature-dependent)
    energies = -2 * np.tanh((critical_temp - temperatures) / 0.5) + np.random.normal(0, 0.1, n_samples)
    
    # Create latent representation
    latent_repr = LatentRepresentation(
        z1=z1,
        z2=z2,
        temperatures=temperatures,
        magnetizations=magnetizations,
        energies=energies,
        reconstruction_errors=np.random.exponential(0.1, n_samples),
        sample_indices=np.arange(n_samples)
    )
    
    # Create order parameter candidate
    correlation_result = CorrelationResult(
        correlation_coefficient=0.85,
        p_value=1e-10,
        confidence_interval=(0.82, 0.88),
        sample_size=n_samples,
        is_significant=True
    )
    
    # Create energy correlation result
    energy_correlation_result = CorrelationResult(
        correlation_coefficient=0.65,
        p_value=1e-8,
        confidence_interval=(0.60, 0.70),
        sample_size=n_samples,
        is_significant=True
    )
    
    order_param_candidate = OrderParameterCandidate(
        latent_dimension='z1',
        correlation_with_magnetization=correlation_result,
        correlation_with_energy=energy_correlation_result,
        temperature_dependence={'slope': -0.5, 'r_squared': 0.85},
        critical_behavior={'transition_sharpness': 0.8, 'critical_exponent': 0.125},
        confidence_score=0.85
    )
    
    # Create phase detection result
    phase_detection = PhaseDetectionResult(
        critical_temperature=2.275,
        confidence=0.92,
        method="clustering",
        transition_region=(2.2, 2.35),
        ensemble_scores={'clustering': 0.92, 'gradient': 0.88, 'information': 0.85}
    )
    
    # Create validation metrics
    validation_metrics = ValidationMetrics(
        order_parameter_correlation=0.85,
        critical_temperature_error=0.006,
        critical_temperature_relative_error=0.27,
        energy_conservation_score=0.95,
        magnetization_conservation_score=0.92,
        physics_consistency_score=0.88,
        statistical_significance={
            'correlation_p_value': 1e-10,
            'critical_temp_p_value': 0.045,
            'conservation_p_value': 1e-8
        },
        theoretical_comparison={
            'onsager_critical_temperature': 2.269,
            'discovered_critical_temperature': 2.275
        }
    )
    
    # Create mock model
    model = ConvolutionalVAE(
        input_shape=(1, lattice_size, lattice_size),
        latent_dim=2
    )
    
    # Create training history
    training_history = create_mock_training_history()
    training_metrics = []
    for epoch_data in training_history:
        metrics = TrainingMetrics(
            epoch=epoch_data['epoch'],
            train_loss=epoch_data['train_loss'],
            val_loss=epoch_data['val_loss'],
            reconstruction_loss=epoch_data['reconstruction_loss'],
            kl_loss=epoch_data['kl_loss'],
            learning_rate=epoch_data['learning_rate'],
            gradient_norm=epoch_data['gradient_norm'],
            latent_samples=epoch_data['latent_samples'],
            reconstruction_samples=None
        )
        training_metrics.append(metrics)
    
    # Create baseline comparison results
    baseline_results = [
        ComparisonResult(
            method_name="PCA",
            latent_representation=np.random.randn(n_samples, 2),
            order_parameter_correlation=0.45,
            critical_temperature=2.35,
            physics_consistency_score=0.52,
            computational_cost=0.1,
            additional_metrics={'explained_variance_ratio': [0.6, 0.25]}
        ),
        ComparisonResult(
            method_name="t-SNE",
            latent_representation=np.random.randn(500, 2),  # Smaller for t-SNE
            order_parameter_correlation=0.38,
            critical_temperature=None,
            physics_consistency_score=0.35,
            computational_cost=10.0,
            additional_metrics={'perplexity': 30.0}
        )
    ]
    
    # Create ablation study results
    beta_values = [0.1, 0.5, 1.0, 2.0, 4.0]
    ablation_results = []
    
    for beta in beta_values:
        # Simulate beta effect (optimal around 1.0)
        corr_effect = 1.0 - 0.3 * abs(np.log(beta))
        correlation = max(0.3, min(0.9, 0.85 * corr_effect))
        
        tc_error = abs(beta - 1.0) * 2.0 + np.random.normal(0, 0.5)
        tc_error = max(0.1, tc_error)
        
        physics_score = correlation * (1.0 - tc_error / 10.0)
        
        ablation_results.append(AblationResult(
            parameter_name="beta",
            parameter_value=beta,
            order_parameter_correlation=correlation,
            critical_temperature_error=tc_error,
            physics_consistency_score=physics_score,
            training_time=100 + beta * 20,
            convergence_epochs=int(80 + beta * 10)
        ))
    
    # Create architecture comparison results
    architecture_configs = [
        ("LatentDim_2_Layers_3", 2, 3, 0.85, 1.0),
        ("LatentDim_4_Layers_3", 4, 3, 0.82, 1.5),
        ("LatentDim_8_Layers_3", 8, 3, 0.78, 2.2),
        ("LatentDim_2_Layers_4", 2, 4, 0.87, 1.3),
        ("LatentDim_2_Layers_5", 2, 5, 0.83, 1.8),
        ("LatentDim_4_Layers_4", 4, 4, 0.84, 2.0),
        ("LatentDim_8_Layers_4", 8, 4, 0.80, 3.1),
    ]
    
    architecture_results = []
    for arch_name, latent_dim, layers, base_corr, base_cost in architecture_configs:
        # Add some variation
        correlation = base_corr + np.random.normal(0, 0.02)
        correlation = max(0.3, min(0.95, correlation))
        
        # Critical temperature varies slightly
        critical_temp = 2.269 + np.random.normal(0, 0.01)
        
        # Physics consistency based on correlation and Tc accuracy
        tc_error = abs(critical_temp - 2.269) / 2.269
        physics_score = correlation * (1.0 - tc_error / 0.05)
        
        architecture_results.append(ComparisonResult(
            method_name=arch_name,
            latent_representation=np.random.randn(n_samples, latent_dim),
            order_parameter_correlation=correlation,
            critical_temperature=critical_temp,
            physics_consistency_score=physics_score,
            computational_cost=base_cost,
            additional_metrics={
                'latent_dimension': latent_dim,
                'layer_count': layers,
                'parameter_count': latent_dim * layers * 1000  # Mock parameter count
            }
        ))
    
    # Create significance test results
    significance_results = {
        'correlation_tests': {
            'mean_correlation': 0.85,
            'std_correlation': 0.05,
            'vs_zero': {'statistic': 15.2, 'p_value': 1e-12, 'significant': True},
            'vs_random': {'statistic': 12.8, 'p_value': 1e-10, 'significant': True},
            'confidence_interval': (0.82, 0.88)
        },
        'critical_temperature_tests': {
            'mean_tc': 2.275,
            'std_tc': 0.015,
            'mean_relative_error': 0.27,
            'std_relative_error': 0.1,
            'vs_theoretical': {'statistic': 2.1, 'p_value': 0.045, 'significant': True},
            'error_vs_zero': {'statistic': 1.8, 'p_value': 0.08, 'significant': False},
            'confidence_interval': (2.26, 2.29)
        },
        'combined_tests': {
            'correlation_vs_accuracy': {
                'spearman_correlation': 0.72,
                'p_value': 0.003,
                'significant': True
            }
        }
    }
    
    # Create complete dataset
    dataset = PublicationDataset(
        training_history=training_metrics,
        trained_model=model,
        original_configs=original_configs,
        reconstructed_configs=reconstructed_configs,
        latent_samples=latent_samples,
        temperatures=temperatures,
        magnetizations=magnetizations,
        latent_representation=latent_repr,
        order_parameter_candidates=[order_param_candidate],
        phase_detection_result=phase_detection,
        validation_metrics=validation_metrics,
        baseline_results=baseline_results,
        ablation_results=ablation_results,
        architecture_results=architecture_results,
        significance_results=significance_results
    )
    
    return dataset


def main():
    """Main function to demonstrate publication materials generation."""
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info("Starting publication materials generation example")
    
    # Create output directory
    output_dir = Path("results/publication_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create mock dataset
        dataset = create_mock_dataset()
        
        # Initialize publication materials generator
        pub_generator = PublicationMaterialsGenerator()
        
        # Load training history
        training_history_dicts = []
        for metrics in dataset.training_history:
            training_history_dicts.append({
                'epoch': metrics.epoch,
                'train_loss': metrics.train_loss,
                'val_loss': metrics.val_loss,
                'reconstruction_loss': metrics.reconstruction_loss,
                'kl_loss': metrics.kl_loss,
                'learning_rate': metrics.learning_rate,
                'gradient_norm': metrics.gradient_norm,
                'latent_samples': metrics.latent_samples
            })
        
        pub_generator.load_training_history(training_history_dicts)
        
        # Generate complete publication package
        logging.info("Generating complete publication package...")
        package = pub_generator.generate_complete_publication_package(
            dataset, str(output_dir)
        )
        
        # Print summary
        print("\n" + "="*60)
        print("PUBLICATION MATERIALS GENERATION COMPLETE")
        print("="*60)
        print(f"Output directory: {output_dir}")
        print(f"Generation time: {package['generation_time']}")
        
        print(f"\nMain figures generated: {len(package['main_figures'])}")
        for name, path in package['main_figures'].items():
            print(f"  - {name}: {path}")
        
        print(f"\nAnalysis reports generated: {len(package['analysis_reports'])}")
        for category, plots in package['analysis_reports'].items():
            print(f"  - {category}: {len(plots)} plots")
        
        print(f"\nSupplementary figures: {len(package['supplementary_figures'])}")
        for name, path in package['supplementary_figures'].items():
            print(f"  - {name}: {path}")
        
        print(f"\nData summaries: {len(package['data_summaries'])}")
        for name, path in package['data_summaries'].items():
            print(f"  - {name}: {path}")
        
        print(f"\nPublication checklist: {package['checklist']}")
        
        # Generate individual components for demonstration
        print("\n" + "-"*40)
        print("INDIVIDUAL COMPONENT EXAMPLES")
        print("-"*40)
        
        # Training diagnostics
        logging.info("Generating training diagnostics...")
        training_plots = pub_generator.generate_training_diagnostics(
            str(output_dir / "individual_training")
        )
        print(f"Training diagnostic plots: {len(training_plots)}")
        
        # Reconstruction analysis
        logging.info("Generating reconstruction analysis...")
        reconstruction_plots = pub_generator.generate_reconstruction_analysis(
            dataset, str(output_dir / "individual_reconstruction")
        )
        print(f"Reconstruction analysis plots: {len(reconstruction_plots)}")
        
        # Comparison studies
        logging.info("Generating comparison studies...")
        comparison_plots = pub_generator.generate_comparison_studies(
            dataset, str(output_dir / "individual_comparison")
        )
        print(f"Comparison study plots: {len(comparison_plots)}")
        
        # Create and save main results figure
        logging.info("Creating main results figure...")
        main_fig = pub_generator.create_main_results_figure(dataset)
        main_fig_path = output_dir / "example_main_results.png"
        main_fig.savefig(main_fig_path, dpi=300, bbox_inches='tight')
        plt.close(main_fig)
        print(f"Main results figure saved: {main_fig_path}")
        
        print("\n" + "="*60)
        print("EXAMPLE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Check the output directory for all generated materials: {output_dir}")
        print("Use the publication checklist to ensure all materials are ready for submission.")
        
    except Exception as e:
        logging.error(f"Error in publication materials example: {e}")
        raise


if __name__ == "__main__":
    main()