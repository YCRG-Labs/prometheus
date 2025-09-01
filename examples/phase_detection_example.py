#!/usr/bin/env python3
"""
Phase Detection Example

This example demonstrates how to use the phase transition detection system
to identify critical temperatures from latent space representations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.phase_detection import PhaseTransitionDetector
from src.analysis.latent_analysis import LatentRepresentation


def create_synthetic_phase_data(n_samples: int = 2000, 
                               critical_temp: float = 2.269) -> LatentRepresentation:
    """
    Create synthetic latent representation with clear phase transition.
    
    Args:
        n_samples: Number of samples to generate
        critical_temp: Critical temperature for phase transition
        
    Returns:
        LatentRepresentation with phase transition behavior
    """
    np.random.seed(42)
    
    # Generate temperature range
    temperatures = np.random.uniform(1.5, 3.0, n_samples)
    
    # Create latent variables with phase transition behavior
    # z1 represents order parameter (magnetization-like)
    z1 = np.tanh(2 * (critical_temp - temperatures)) + np.random.normal(0, 0.2, n_samples)
    
    # z2 represents fluctuations (peaks near critical point)
    z2 = np.exp(-((temperatures - critical_temp) / 0.3)**2) + np.random.normal(0, 0.1, n_samples)
    
    # Physical quantities
    magnetizations = np.abs(z1) * 0.8 + np.random.normal(0, 0.05, n_samples)
    energies = -2.0 * magnetizations + np.random.normal(0, 0.1, n_samples)
    reconstruction_errors = np.random.exponential(0.1, n_samples)
    sample_indices = np.arange(n_samples)
    
    return LatentRepresentation(
        z1=z1,
        z2=z2,
        temperatures=temperatures,
        magnetizations=magnetizations,
        energies=energies,
        reconstruction_errors=reconstruction_errors,
        sample_indices=sample_indices
    )


def main():
    """Main example function."""
    print("Phase Transition Detection Example")
    print("=" * 50)
    
    # Create synthetic data with known phase transition
    print("1. Creating synthetic latent space data with phase transition...")
    latent_repr = create_synthetic_phase_data(n_samples=2000)
    
    print(f"   - Generated {latent_repr.n_samples} samples")
    print(f"   - Temperature range: {np.min(latent_repr.temperatures):.2f} - {np.max(latent_repr.temperatures):.2f}")
    print(f"   - True critical temperature: 2.269")
    
    # Initialize phase detector
    print("\n2. Initializing phase transition detector...")
    detector = PhaseTransitionDetector(theoretical_tc=2.269)
    
    # Test individual methods
    print("\n3. Testing individual detection methods...")
    
    # Clustering method
    print("   a) Clustering-based detection...")
    clustering_result = detector.detect_phase_transition(latent_repr, methods=['clustering'])
    print(f"      Critical temperature: {clustering_result.critical_temperature:.3f}")
    print(f"      Confidence: {clustering_result.confidence:.3f}")
    
    # Gradient method
    print("   b) Gradient-based detection...")
    gradient_result = detector.detect_phase_transition(latent_repr, methods=['gradient'])
    print(f"      Critical temperature: {gradient_result.critical_temperature:.3f}")
    print(f"      Confidence: {gradient_result.confidence:.3f}")
    
    # Information-theoretic method
    print("   c) Information-theoretic detection...")
    info_result = detector.detect_phase_transition(latent_repr, methods=['information_theoretic'])
    print(f"      Critical temperature: {info_result.critical_temperature:.3f}")
    print(f"      Confidence: {info_result.confidence:.3f}")
    
    # Ensemble method
    print("\n4. Testing ensemble detection...")
    ensemble_result = detector.detect_phase_transition(
        latent_repr, 
        methods=['clustering', 'gradient', 'information_theoretic']
    )
    print(f"   Ensemble critical temperature: {ensemble_result.critical_temperature:.3f}")
    print(f"   Ensemble confidence: {ensemble_result.confidence:.3f}")
    print(f"   Method weights:")
    for method, details in ensemble_result.ensemble_scores.items():
        print(f"     - {method}: T_c={details['critical_temperature']:.3f}, weight={details['weight']:.3f}")
    
    # Validation against theoretical value
    print("\n5. Validation against Onsager solution...")
    validation = detector.validate_detection(ensemble_result)
    onsager_comparison = detector.compare_with_onsager_solution(ensemble_result)
    
    print(f"   Theoretical T_c (Onsager): {onsager_comparison['onsager_tc']:.3f}")
    print(f"   Detected T_c: {onsager_comparison['detected_tc']:.3f}")
    print(f"   Relative error: {onsager_comparison['relative_error_percent']:.1f}%")
    print(f"   Accuracy class: {onsager_comparison['accuracy_class']}")
    print(f"   Within 5% tolerance: {validation['within_tolerance']}")
    
    # Create comprehensive report
    print("\n6. Generating comprehensive analysis report...")
    report = detector.create_comprehensive_report(latent_repr, ensemble_result)
    
    print(f"   Report sections: {list(report.keys())}")
    print(f"   Dataset info: {report['dataset_info']['n_samples']} samples")
    print(f"   Detection method: {report['detection_results']['method']}")
    
    # Create visualizations
    print("\n7. Creating visualizations...")
    
    # Latent space visualization
    from src.analysis.latent_analysis import LatentAnalyzer
    
    # Mock VAE for analyzer (not needed for visualization)
    class MockVAE:
        def to(self, device):
            return self
        def eval(self):
            return self
    
    analyzer = LatentAnalyzer(MockVAE())
    
    # Visualize latent space colored by temperature
    fig1 = analyzer.visualize_latent_space(latent_repr, color_by='temperature')
    fig1.suptitle('Latent Space Colored by Temperature')
    
    # Visualize phase separation
    fig2 = analyzer.visualize_phase_separation(latent_repr, critical_temp=ensemble_result.critical_temperature)
    fig2.suptitle(f'Phase Separation (T_c = {ensemble_result.critical_temperature:.3f})')
    
    # Visualize clustering results
    if ensemble_result.clustering_result:
        fig3 = detector.clustering_detector.visualize_clustering(latent_repr, ensemble_result.clustering_result)
        fig3.suptitle('Clustering-based Phase Detection')
    
    # Visualize gradient analysis
    if ensemble_result.gradient_analysis:
        gradient_data = detector.gradient_detector.calculate_temperature_gradients(latent_repr)
        gradient_result = detector.gradient_detector.detect_critical_temperature(gradient_data)
        fig4 = detector.gradient_detector.visualize_gradients(gradient_data, gradient_result)
        fig4.suptitle('Gradient-based Phase Detection')
    
    print("   ✓ Visualizations created")
    
    # Save results
    output_dir = Path("results/phase_detection_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save figures
    fig1.savefig(output_dir / "latent_space_temperature.png", dpi=300, bbox_inches='tight')
    fig2.savefig(output_dir / "phase_separation.png", dpi=300, bbox_inches='tight')
    
    if ensemble_result.clustering_result:
        fig3.savefig(output_dir / "clustering_analysis.png", dpi=300, bbox_inches='tight')
    
    if ensemble_result.gradient_analysis:
        fig4.savefig(output_dir / "gradient_analysis.png", dpi=300, bbox_inches='tight')
    
    # Save latent representation
    analyzer.save_latent_representation(latent_repr, output_dir / "latent_representation.h5")
    
    print(f"\n8. Results saved to: {output_dir}")
    print("   ✓ Latent space visualization")
    print("   ✓ Phase separation plot")
    print("   ✓ Clustering analysis")
    print("   ✓ Gradient analysis")
    print("   ✓ Latent representation data")
    
    print("\n" + "=" * 50)
    print("Phase detection example completed successfully!")
    print(f"Final result: T_c = {ensemble_result.critical_temperature:.3f} ± {np.diff(ensemble_result.transition_region)[0]/2:.3f}")
    print(f"Accuracy: {onsager_comparison['accuracy_class']} ({onsager_comparison['relative_error_percent']:.1f}% error)")
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()