"""
Quantum-Aware Variational Autoencoder for Quantum State Analysis

This module implements a VAE specifically designed for quantum states,
handling complex amplitudes, entanglement structure, and quantum fidelity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
import numpy as np


class QuantumStateEncoder(nn.Module):
    """
    Encoder for quantum states that handles complex amplitudes.
    
    Takes quantum state vectors (with real and imaginary parts) and
    encodes them into a latent space representation that preserves
    quantum information structure.
    """
    
    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 8,
        hidden_dims: List[int] = [256, 128, 64],
        activation: str = 'relu'
    ):
        """
        Initialize quantum state encoder.
        
        Args:
            state_dim: Dimension of quantum state (2^L for L qubits)
            latent_dim: Dimensionality of latent space
            hidden_dims: Hidden layer dimensions
            activation: Activation function
        """
        super(QuantumStateEncoder, self).__init__()
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        # Input processes both real and imaginary parts
        input_dim = 2 * state_dim
        
        # Build encoder network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.encoder_net = nn.Sequential(*layers)
        
        # Latent distribution parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        }
        return activations.get(activation.lower(), nn.ReLU())
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode quantum state to latent distribution.
        
        Args:
            state: Complex quantum state (batch_size, state_dim, 2)
                   where last dim is [real, imag]
            
        Returns:
            (mu, logvar) for latent distribution
        """
        batch_size = state.size(0)
        
        # Flatten real and imaginary parts
        x = state.view(batch_size, -1)
        
        # Encode
        h = self.encoder_net(x)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu


class QuantumStateDecoder(nn.Module):
    """
    Decoder for quantum states that reconstructs complex amplitudes.
    
    Takes latent vectors and reconstructs quantum state vectors
    with proper normalization.
    """
    
    def __init__(
        self,
        latent_dim: int,
        state_dim: int,
        hidden_dims: List[int] = [64, 128, 256],
        activation: str = 'relu'
    ):
        """
        Initialize quantum state decoder.
        
        Args:
            latent_dim: Dimensionality of latent space
            state_dim: Dimension of quantum state (2^L)
            hidden_dims: Hidden layer dimensions
            activation: Activation function
        """
        super(QuantumStateDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        
        # Build decoder network
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.decoder_net = nn.Sequential(*layers)
        
        # Output layer for real and imaginary parts
        self.fc_out = nn.Linear(prev_dim, 2 * state_dim)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        }
        return activations.get(activation.lower(), nn.ReLU())
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to quantum state.
        
        Args:
            z: Latent vector (batch_size, latent_dim)
            
        Returns:
            Reconstructed quantum state (batch_size, state_dim, 2)
            with [real, imag] in last dimension
        """
        batch_size = z.size(0)
        
        # Decode
        h = self.decoder_net(z)
        out = self.fc_out(h)
        
        # Reshape to (batch, state_dim, 2) for real/imag
        state = out.view(batch_size, self.state_dim, 2)
        
        # Normalize to unit norm (quantum states must be normalized)
        norm = torch.sqrt(torch.sum(state ** 2, dim=(1, 2), keepdim=True))
        state = state / (norm + 1e-8)
        
        return state


class QuantumVAE(nn.Module):
    """
    Quantum-aware Variational Autoencoder.
    
    Designed specifically for quantum states with:
    - Complex amplitude handling
    - Entanglement-aware loss function
    - Quantum fidelity metric
    """
    
    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 8,
        encoder_hidden: List[int] = [256, 128, 64],
        decoder_hidden: List[int] = [64, 128, 256],
        beta: float = 1.0,
        entanglement_weight: float = 0.1
    ):
        """
        Initialize Quantum VAE.
        
        Args:
            state_dim: Dimension of quantum state (2^L)
            latent_dim: Dimensionality of latent space
            encoder_hidden: Hidden dimensions for encoder
            decoder_hidden: Hidden dimensions for decoder
            beta: Beta parameter for β-VAE
            entanglement_weight: Weight for entanglement loss term
        """
        super(QuantumVAE, self).__init__()
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.entanglement_weight = entanglement_weight
        
        self.encoder = QuantumStateEncoder(
            state_dim=state_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden
        )
        
        self.decoder = QuantumStateDecoder(
            latent_dim=latent_dim,
            state_dim=state_dim,
            hidden_dims=decoder_hidden
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode quantum state to latent distribution."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to quantum state."""
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        return self.encoder.reparameterize(mu, logvar)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through quantum VAE.
        
        Args:
            x: Input quantum state (batch_size, state_dim, 2)
            
        Returns:
            (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def quantum_fidelity(
        self, 
        state1: torch.Tensor, 
        state2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute quantum fidelity F = |⟨ψ₁|ψ₂⟩|².
        
        Args:
            state1: First quantum state (batch, state_dim, 2)
            state2: Second quantum state (batch, state_dim, 2)
            
        Returns:
            Fidelity for each pair (batch,)
        """
        # Convert to complex
        psi1 = torch.complex(state1[..., 0], state1[..., 1])
        psi2 = torch.complex(state2[..., 0], state2[..., 1])
        
        # Inner product ⟨ψ₁|ψ₂⟩
        inner_product = torch.sum(torch.conj(psi1) * psi2, dim=1)
        
        # Fidelity = |⟨ψ₁|ψ₂⟩|²
        fidelity = torch.abs(inner_product) ** 2
        
        return fidelity

    
    def entanglement_loss(
        self,
        state: torch.Tensor,
        reconstruction: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute entanglement-aware loss.
        
        Penalizes reconstructions that don't preserve entanglement structure.
        Uses purity as a proxy: Tr(ρ²) for reduced density matrices.
        
        Args:
            state: Original quantum state
            reconstruction: Reconstructed quantum state
            
        Returns:
            Entanglement loss
        """
        batch_size = state.size(0)
        
        # For simplicity, compute purity of half-system
        # This requires reshaping state into bipartite form
        L = int(np.log2(self.state_dim))
        L_A = L // 2
        dim_A = 2 ** L_A
        dim_B = 2 ** (L - L_A)
        
        def compute_purity(psi):
            """Compute purity of subsystem A."""
            # Convert to complex
            psi_complex = torch.complex(psi[..., 0], psi[..., 1])
            
            # Reshape to (batch, dim_A, dim_B)
            psi_matrix = psi_complex.view(batch_size, dim_A, dim_B)
            
            # Reduced density matrix: ρ_A = ψ ψ†
            rho_A = torch.bmm(psi_matrix, psi_matrix.conj().transpose(1, 2))
            
            # Purity: Tr(ρ²)
            rho_A_squared = torch.bmm(rho_A, rho_A)
            purity = torch.real(torch.diagonal(rho_A_squared, dim1=1, dim2=2).sum(dim=1))
            
            return purity
        
        try:
            purity_orig = compute_purity(state)
            purity_recon = compute_purity(reconstruction)
            
            # Loss is difference in purity
            loss = torch.mean((purity_orig - purity_recon) ** 2)
        except:
            # If computation fails (e.g., wrong dimensions), return zero
            loss = torch.tensor(0.0, device=state.device)
        
        return loss
    
    def compute_loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute quantum-aware VAE loss.
        
        Loss = Reconstruction + β * KL + λ * Entanglement
        
        Args:
            x: Original quantum state
            reconstruction: Reconstructed quantum state
            mu: Latent mean
            logvar: Latent log variance
            
        Returns:
            Dictionary of loss components
        """
        batch_size = x.size(0)
        
        # 1. Reconstruction loss using fidelity
        # We want high fidelity, so loss = 1 - fidelity
        fidelity = self.quantum_fidelity(x, reconstruction)
        recon_loss = torch.mean(1.0 - fidelity)
        
        # 2. KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.mean(kl_loss)
        
        # 3. Entanglement loss
        ent_loss = self.entanglement_loss(x, reconstruction)
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss + self.entanglement_weight * ent_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'entanglement_loss': ent_loss,
            'fidelity': torch.mean(fidelity)
        }
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation (mean of distribution)."""
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu
    
    def sample(self, num_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Generate quantum states from prior.
        
        Args:
            num_samples: Number of states to generate
            device: Device to generate on
            
        Returns:
            Generated quantum states
        """
        if device is None:
            device = next(self.parameters()).device
        
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        with torch.no_grad():
            samples = self.decode(z)
        
        return samples


def prepare_quantum_state_batch(
    states: np.ndarray,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Prepare batch of quantum states for VAE input.
    
    Args:
        states
: Array of quantum state vectors (batch, state_dim)
                Can be real or complex
        device: PyTorch device
        
    Returns:
        Tensor of shape (batch, state_dim, 2) with [real, imag]
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = states.shape[0]
    state_dim = states.shape[1]
    
    # Prepare real and imaginary parts
    if np.iscomplexobj(states):
        real_part = np.real(states)
        imag_part = np.imag(states)
    else:
        real_part = states
        imag_part = np.zeros_like(states)
    
    # Stack into (batch, state_dim, 2)
    state_tensor = np.stack([real_part, imag_part], axis=-1)
    
    return torch.tensor(state_tensor, dtype=torch.float32, device=device)


def extract_quantum_states(
    state_tensor: torch.Tensor
) -> np.ndarray:
    """
    Extract quantum states from VAE output format.
    
    Args:
        state_tensor: Tensor of shape (batch, state_dim, 2)
        
    Returns:
        Complex numpy array of shape (batch, state_dim)
    """
    state_np = state_tensor.cpu().numpy()
    real_part = state_np[..., 0]
    imag_part = state_np[..., 1]
    
    return real_part + 1j * imag_part
