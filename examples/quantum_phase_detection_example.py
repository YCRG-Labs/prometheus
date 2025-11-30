"""
Example: Quantum Phase Transition Detection using VAE

Demonstrates detection of quantum critical points and extraction of
critical exponents for the transverse-field Ising model.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from src.models.quantum_vae import QuantumVAE, train_quantum_vae
from src.analysis.quantum_phase_detection import (
    QuantumPhaseDetector,
    CriticalExponentExtractor,
    validate_on_clean_tfim
)
from src.quantum.hamiltonian_builder import SpinHamiltonianBuilder, SpinChainParams
from src.quantum.exact_diagonalization import ExactDiagonalizationSolver


def generate_tfim_dataset(L, h_values, n_samples_per_h=10):
    """Generate TFIM states across phase transition."""
    print(f"Generating TFIM dataset: L={L}, {len(h_values)} h-values, {n_samples_per_h} samples each")
    
    builder = SpinHamiltonianBuilder(L)
    solver = ExactDiagonalizationSolver()
    all_states = []
    
    for i, h in enumerate(h_values):
        params = SpinChainParams(L=L, J=1.0, h=h)
        H = builder.build_tfim(params)
        
        states_at_h = []
        for _ in range(n_samples_per_h):
            energy, state = solver.ground_state(H)
            states_at_h.append(state)
        
        all_states.append(np.array(states_at_h))
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{len(h_values)} parameter points")
    
    return np.array(all_states)


def train_vae_on_tfim(L, h_values, n_samples=10, epochs=50):
    """Train VAE on TFIM states."""
    print("\n=== Training VAE on TFIM ===")
    
    # Generate training data
    states = generate_tfim_dataset(L, h_values, n_samples)
    
    # Flatten for training
    n_total = len(h_values) * n_samples
    training_states = states.reshape(n_total, -1)
    
    # Initialize VAE
    state_dim = 2**L
    latent_dim = 6
    vae = QuantumVAE(
        state_dim=state_dim,
        latent_dim=latent_dim,
        encoder_hidden=[128, 64, 32],
        decoder_hidden=[32, 64, 128]
    )
    
    # Train
    trained_vae, history = train_quantum_vae(
        vae,
        training_states,
        epochs=epochs,
        batch_size=32,
        learning_rate=1e-3
    )
    
    print(f"Training complete. Final loss: {history['total_loss'][-1]:.4f}")
    
    return trained_vae, states


def detect_qcp_all_methods(vae, states, h_values):
    """Run all QCP detection methods."""
    print("\n=== Detecting Quantum Critical Point ===")
    
    detector = QuantumPhaseDetector(vae)
    
    # Method 1: Latent variance
    print("\n1. Latent Variance Method:")
    result_var = detector.detect_qcp_latent_variance(states, h_values)
    print(f"   Detected Tc = {result_var.critical_point:.4f}")
    print(f"   Confidence = {result_var.confidence:.4f}")
    
    # Method 2: Reconstruction error
    print("\n2. Reconstruction Error Method:")
    result_recon = detector.detect_qcp_reconstruction_error(states, h_values)
    print(f"   Detected Tc = {result_recon.critical_point:.4f}")
    print(f"   Confidence = {result_recon.confidence:.4f}")
    
    # Method 3: Fidelity susceptibility
    print("\n3. Fidelity Susceptibility Method:")
    result_fid = detector.detect_qcp_fidelity_susceptibility(states, h_values)
    print(f"   Detected Tc = {result_fid.critical_point:.4f}")
    print(f"   Confidence = {result_fid.confidence:.4f}")
    
    # Method 4: Ensemble
    print("\n4. Ensemble Method:")
    result_ensemble = detector.detect_qcp_ensemble(states, h_values)
    print(f"   Detected Tc = {result_ensemble.critical_point:.4f}")
    print(f"   Confidence = {result_ensemble.confidence:.4f}")
    
    return {
        'variance': result_var,
        'reconstruction': result_recon,
        'fidelity': result_fid,
        'ensemble': result_ensemble
    }


def extract_critical_exponents(vae, states, h_values, critical_point, L):
    """Extract critical exponents."""
    print("\n=== Extracting Critical Exponents ===")
    
    extractor = CriticalExponentExtractor(vae)
    
    # Dynamical exponent z
    z, z_err = extractor.extract_dynamical_exponent(
        states, h_values, critical_point, L
    )
    print(f"\nDynamical exponent z = {z:.4f} ± {z_err:.4f}")
    print(f"Expected for TFIM: z = 1.0")
    
    return {'z': z, 'z_error': z_err}


def plot_detection_results(results, h_values, save_path=None):
    """Plot QCP detection results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Quantum Critical Point Detection Methods', fontsize=14, fontweight='bold')
    
    # Plot 1: Latent variance
    ax = axes[0, 0]
    result = results['variance']
    variances = result.supporting_evidence['variances']
    ax.plot(h_values, variances, 'b-', linewidth=2)
    ax.axvline(result.critical_point, color='r', linestyle='--', 
               label=f'Detected Tc={result.critical_point:.3f}')
    ax.axvline(1.0, color='g', linestyle=':', label='Expected Tc=1.0')
    ax.set_xlabel('Transverse Field h')
    ax.set_ylabel('Latent Variance')
    ax.set_title('Method 1: Latent Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Reconstruction error
    ax = axes[0, 1]
    result = results['reconstruction']
    errors = result.supporting_evidence['errors']
    ax.plot(h_values, errors, 'b-', linewidth=2)
    ax.axvline(result.critical_point, color='r', linestyle='--',
               label=f'Detected Tc={result.critical_point:.3f}')
    ax.axvline(1.0, color='g', linestyle=':', label='Expected Tc=1.0')
    ax.set_xlabel('Transverse Field h')
    ax.set_ylabel('Reconstruction Error')
    ax.set_title('Method 2: Reconstruction Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Fidelity susceptibility
    ax = axes[1, 0]
    result = results['fidelity']
    susceptibility = result.supporting_evidence['susceptibility']
    params = result.supporting_evidence['parameters']
    ax.plot(params, susceptibility, 'b-', linewidth=2)
    ax.axvline(result.critical_point, color='r', linestyle='--',
               label=f'Detected Tc={result.critical_point:.3f}')
    ax.axvline(1.0, color='g', linestyle=':', label='Expected Tc=1.0')
    ax.set_xlabel('Transverse Field h')
    ax.set_ylabel('Fidelity Susceptibility')
    ax.set_title('Method 3: Fidelity Susceptibility')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary comparison
    ax = axes[1, 1]
    methods = ['Variance', 'Recon Error', 'Fidelity', 'Ensemble']
    detected_Tc = [
        results['variance'].critical_point,
        results['reconstruction'].critical_point,
        results['fidelity'].critical_point,
        results['ensemble'].critical_point
    ]
    confidences = [
        results['variance'].confidence,
        results['reconstruction'].confidence,
        results['fidelity'].confidence,
        results['ensemble'].confidence
    ]
    
    x = np.arange(len(methods))
    bars = ax.bar(x, detected_Tc, color=['blue', 'orange', 'green', 'red'], alpha=0.7)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=2, label='Expected Tc=1.0')
    ax.set_ylabel('Detected Critical Point')
    ax.set_title('Method Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add confidence as text
    for i, (bar, conf) in enumerate(zip(bars, confidences)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'conf={conf:.2f}',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
    
    plt.show()


def main():
    """Main example execution."""
    print("=" * 70)
    print("Quantum Phase Transition Detection Example")
    print("System: Transverse-Field Ising Model (TFIM)")
    print("=" * 70)
    
    # Parameters
    L = 10  # System size
    h_values = np.linspace(0.5, 1.5, 40)  # Scan across phase transition
    n_samples = 8
    
    # Train VAE
    vae, states = train_vae_on_tfim(L, h_values, n_samples, epochs=30)
    
    # Detect QCP using all methods
    results = detect_qcp_all_methods(vae, states, h_values)
    
    # Extract critical exponents
    critical_point = results['ensemble'].critical_point
    exponents = extract_critical_exponents(vae, states, h_values, critical_point, L)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Expected critical point: Tc = 1.0")
    print(f"Detected critical point: Tc = {critical_point:.4f}")
    print(f"Error: {abs(critical_point - 1.0):.4f}")
    print(f"Confidence: {results['ensemble'].confidence:.4f}")
    print(f"\nExpected dynamical exponent: z = 1.0")
    print(f"Estimated dynamical exponent: z = {exponents['z']:.4f} ± {exponents['z_error']:.4f}")
    
    # Validation status
    error = abs(critical_point - 1.0)
    if error < 0.1:
        status = "✓ EXCELLENT"
    elif error < 0.2:
        status = "✓ GOOD"
    elif error < 0.3:
        status = "✓ ACCEPTABLE"
    else:
        status = "✗ NEEDS IMPROVEMENT"
    
    print(f"\nValidation Status: {status}")
    print("=" * 70)
    
    # Plot results
    output_dir = Path('results/quantum_phase_detection')
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / 'qcp_detection_results.png'
    plot_detection_results(results, h_values, save_path=plot_path)
    
    # Save numerical results
    results_file = output_dir / 'detection_results.txt'
    with open(results_file, 'w') as f:
        f.write("Quantum Phase Transition Detection Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"System: TFIM with L={L}\n")
        f.write(f"Parameter range: h ∈ [{h_values[0]:.2f}, {h_values[-1]:.2f}]\n")
        f.write(f"Number of points: {len(h_values)}\n")
        f.write(f"Samples per point: {n_samples}\n\n")
        f.write("Detection Results:\n")
        f.write("-" * 50 + "\n")
        for method, result in results.items():
            f.write(f"{method.capitalize()}:\n")
            f.write(f"  Tc = {result.critical_point:.4f}\n")
            f.write(f"  Confidence = {result.confidence:.4f}\n")
            f.write(f"  Error = {abs(result.critical_point - 1.0):.4f}\n\n")
        f.write("Critical Exponents:\n")
        f.write("-" * 50 + "\n")
        f.write(f"z = {exponents['z']:.4f} ± {exponents['z_error']:.4f}\n")
        f.write(f"Expected: z = 1.0\n")
    
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
