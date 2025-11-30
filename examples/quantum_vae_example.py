"""
Example: Quantum-Aware VAE for Quantum State Analysis

Demonstrates:
1. Training quantum VAE on quantum states
2. Detecting quantum phase transitions
3. Extracting critical exponents
4. Testing on clean TFIM
"""

import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.quantum_vae import (
    QuantumVAE,
    prepare_quantum_state_batch,
    extract_quantum_states
)
from src.analysis.quantum_phase_detection import (
    QuantumPhaseDetector,
    CriticalExponentExtractor
)
from src.quantum.hamiltonian_builder import build_tfim_hamiltonian
from src.quantum.exact_diagonalization import ExactDiagonalizationSolver


def generate_tfim_dataset(L: int, h_values: np.ndarray, n_samples: int = 20):
    """
    Generate dataset of TFIM ground states.
    
    Args:
        L: System size
        h_values: Transverse field values
        n_samples: Samples per parameter value
        
    Returns:
        (states, labels) where states is (n_params, n_samples, 2^L)
    """
    print(f"Generating TFIM dataset: L={L}, {len(h_values)} parameter points")
    
    solver = ExactDiagonalizationSolver(random_seed=42)
    states = []
    
    for i, h in enumerate(h_values):
        H = build_tfim_hamiltonian(L, J=1.0, h=h)
        
        sample_states = []
        for _ in range(n_samples):
            energy, state = solver.ground_state(H)
            sample_states.append(state)
        
        states.append(np.array(sample_states))
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{len(h_values)} parameter points")
    
    return np.array(states), h_values


def train_quantum_vae(
    states: np.ndarray,
    latent_dim: int = 8,
    n_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3
):
    """
    Train quantum VAE on quantum states.
    
    Args:
        states: Quantum states (n_params, n_samples, state_dim)
        latent_dim: Latent dimension
        n_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        
    Returns:
        Trained VAE model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Flatten parameter and sample dimensions for training
    n_params, n_samples, state_dim = states.shape
    states_flat = states.reshape(-1, state_dim)
    
    # Initialize model
    vae = QuantumVAE(
        state_dim=state_dim,
        latent_dim=latent_dim,
        beta=1.0,
        entanglement_weight=0.1
    ).to(device)
    
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    
    print(f"\nTraining Quantum VAE:")
    print(f"  State dim: {state_dim}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Total samples: {len(states_flat)}")
    print(f"  Epochs: {n_epochs}")
    
    # Training loop
    vae.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_fidelity = 0.0
        n_batches = 0
        
        # Shuffle data
        indices = np.random.permutation(len(states_flat))
        
        for i in range(0, len(states_flat), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_states = states_flat[batch_indices]
            
            # Prepare batch
            x = prepare_quantum_state_batch(batch_states, device=device)
            
            # Forward pass
            reconstruction, mu, logvar = vae(x)
            
            # Compute loss
            loss_dict = vae.compute_loss(x, reconstruction, mu, logvar)
            loss = loss_dict['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_fidelity += loss_dict['fidelity'].item()
            n_batches += 1
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / n_batches
            avg_fidelity = epoch_fidelity / n_batches
            print(f"  Epoch {epoch + 1}/{n_epochs}: Loss = {avg_loss:.4f}, Fidelity = {avg_fidelity:.4f}")
    
    print("Training complete!")
    return vae


def detect_phase_transition(vae: QuantumVAE, states: np.ndarray, h_values: np.ndarray):
    """
    Detect quantum phase transition using trained VAE.
    
    Args:
        vae: Trained quantum VAE
        states: Quantum states (n_params, n_samples, state_dim)
        h_values: Parameter values
        
    Returns:
        Detection results
    """
    print("\n" + "=" * 60)
    print("QUANTUM PHASE TRANSITION DETECTION")
    print("=" * 60)
    
    detector = QuantumPhaseDetector(vae)
    
    # Method 1: Latent variance
    print("\n1. Latent Variance Method:")
    result_var = detector.detect_qcp_latent_variance(states, h_values)
    print(f"   Detected Tc = {result_var.critical_point:.4f}")
    print(f"   Confidence = {result_var.confidence:.2f}")
    
    # Method 2: Reconstruction error
    print("\n2. Reconstruction Error Method:")
    result_recon = detector.detect_qcp_reconstruction_error(states, h_values)
    print(f"   Detected Tc = {result_recon.critical_point:.4f}")
    print(f"   Confidence = {result_recon.confidence:.2f}")
    
    # Method 3: Fidelity susceptibility
    print("\n3. Fidelity Susceptibility Method:")
    result_fid = detector.detect_qcp_fidelity_susceptibility(states, h_values)
    print(f"   Detected Tc = {result_fid.critical_point:.4f}")
    print(f"   Confidence = {result_fid.confidence:.2f}")
    
    # Ensemble method
    print("\n4. Ensemble Method:")
    result_ensemble = detector.detect_qcp_ensemble(states, h_values)
    print(f"   Detected Tc = {result_ensemble.critical_point:.4f}")
    print(f"   Confidence = {result_ensemble.confidence:.2f}")
    
    print("\n" + "=" * 60)
    print(f"EXPECTED Tc = 1.0000 (clean TFIM)")
    print(f"ERROR = {abs(result_ensemble.critical_point - 1.0):.4f}")
    print("=" * 60)
    
    return result_ensemble


def main():
    """Main example execution."""
    print("=" * 60)
    print("QUANTUM VAE EXAMPLE: TFIM Phase Transition Detection")
    print("=" * 60)
    
    # Parameters
    L = 10  # System size (2^10 = 1024 dimensional Hilbert space)
    n_points = 30  # Parameter points
    n_samples = 15  # Samples per point
    latent_dim = 6  # Latent dimension
    
    # Generate dataset
    h_values = np.linspace(0.5, 1.5, n_points)
    states, h_values = generate_tfim_dataset(L, h_values, n_samples)
    
    print(f"\nDataset shape: {states.shape}")
    print(f"State dimension: {states.shape[2]}")
    
    # Train VAE
    vae = train_quantum_vae(
        states,
        latent_dim=latent_dim,
        n_epochs=50,
        batch_size=32,
        learning_rate=1e-3
    )
    
    # Detect phase transition
    result = detect_phase_transition(vae, states, h_values)
    
    # Test quantum fidelity
    print("\n" + "=" * 60)
    print("QUANTUM FIDELITY TEST")
    print("=" * 60)
    
    vae.eval()
    test_states = prepare_quantum_state_batch(states[0][:5], device=next(vae.parameters()).device)
    
    with torch.no_grad():
        reconstruction, _, _ = vae(test_states)
        fidelity = vae.quantum_fidelity(test_states, reconstruction)
    
    print(f"Average reconstruction fidelity: {fidelity.mean().item():.4f}")
    print(f"Min fidelity: {fidelity.min().item():.4f}")
    print(f"Max fidelity: {fidelity.max().item():.4f}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)
    
    return vae, result


if __name__ == "__main__":
    vae, result = main()
