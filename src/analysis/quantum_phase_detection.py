"""
Quantum Phase Transition Detection using VAE Latent Space Analysis

This module implements methods for detecting quantum critical points (QCP)
and extracting critical exponents from VAE latent space representations.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.decomposition import PCA


@dataclass
class QCPDetectionResult:
    """Result of quantum critical point detection."""
    critical_point: float  # Estimated critical parameter value
    confidence: float  # Confidence score (0-1)
    method: str  # Detection method used
    supporting_evidence: Dict  # Additional evidence


@dataclass
class CriticalExponents:
    """Critical exponents for quantum phase transition."""
    nu: Optional[float] = None  # Correlation length exponent
    z: Optional[float] = None  # Dynamical exponent
    beta: Optional[float] = None  # Order parameter exponent
    gamma: Optional[float] = None  # Susceptibility exponent
    eta: Optional[float] = None  # Anomalous dimension


class QuantumPhaseDetector:
    """
    Detect quantum phase transitions using VAE latent space analysis.
    
    Methods:
    1. Latent space clustering/separation
    2. Latent variance peaks
    3. Reconstruction error peaks
    4. Fidelity susceptibility
    """
    
    def __init__(self, vae_model: torch.nn.Module):
        """
        Initialize detector with trained VAE.
        
        Args:
            vae_model: Trained quantum VAE model
        """
        self.vae = vae_model
        self.device = next(vae_model.parameters()).device
    
    def detect_qcp_latent_variance(
        self,
        states: np.ndarray,
        parameters: np.ndarray,
        window_size: int = 5
    ) -> QCPDetectionResult:
        """
        Detect QCP from peaks in latent space variance.
        
        At critical points, latent representations show increased variance
        due to critical fluctuations.
        
        Args:
            states: Quantum states (n_params, n_samples, state_dim)
            parameters: Parameter values (n_params,)
            window_size: Smoothing window size
            
        Returns:
            QCPDetectionResult with detected critical point
        """
        n_params = len(parameters)
        latent_variances = []
        
        for i in range(n_params):
            # Get latent representations for this parameter
            state_batch = self._prepare_states(states[i])
            
            with torch.no_grad():
                mu, _ = self.vae.encode(state_batch)
                mu_np = mu.cpu().numpy()
            
            # Compute variance across samples
            variance = np.var(mu_np, axis=0).mean()
            latent_variances.append(variance)
        
        latent_variances = np.array(latent_variances)
        
        # Smooth the variance curve
        if window_size > 1:
            from scipy.ndimage import uniform_filter1d
            latent_variances = uniform_filter1d(latent_variances, size=window_size)
        
        # Find peaks
        peaks, properties = find_peaks(latent_variances, prominence=0.1)
        
        if len(peaks) == 0:
            # No clear peak, use maximum
            critical_idx = np.argmax(latent_variances)
            confidence = 0.5
        else:
            # Use most prominent peak
            critical_idx = peaks[np.argmax(properties['prominences'])]
            confidence = 0.8
        
        critical_point = parameters[critical_idx]
        
        return QCPDetectionResult(
            critical_point=critical_point,
            confidence=confidence,
            method='latent_variance',
            supporting_evidence={
                'variances': latent_variances,
                'peak_indices': peaks,
                'parameters': parameters
            }
        )
    
    def detect_qcp_reconstruction_error(
        self,
        states: np.ndarray,
        parameters: np.ndarray
    ) -> QCPDetectionResult:
        """
        Detect QCP from peaks in reconstruction error.
        
        Critical states are harder to compress, leading to higher
        reconstruction error.
        
        Args:
            states: Quantum states (n_params, n_samples, state_dim)
            parameters: Parameter values (n_params,)
            
        Returns:
            QCPDetectionResult
        """
        n_params = len(parameters)
        recon_errors = []
        
        for i in range(n_params):
            state_batch = self._prepare_states(states[i])
            
            with torch.no_grad():
                reconstruction, mu, logvar = self.vae(state_batch)
                
                # Compute fidelity
                fidelity = self.vae.quantum_fidelity(state_batch, reconstruction)
                error = 1.0 - fidelity.mean().item()
            
            recon_errors.append(error)
        
        recon_errors = np.array(recon_errors)
        
        # Find peaks
        peaks, properties = find_peaks(recon_errors, prominence=0.01)
        
        if len(peaks) == 0:
            critical_idx = np.argmax(recon_errors)
            confidence = 0.6
        else:
            critical_idx = peaks[np.argmax(properties['prominences'])]
            confidence = 0.85
        
        critical_point = parameters[critical_idx]
        
        return QCPDetectionResult(
            critical_point=critical_point,
            confidence=confidence,
            method='reconstruction_error',
            supporting_evidence={
                'errors': recon_errors,
                'peak_indices': peaks,
                'parameters': parameters
            }
        )
    
    def detect_qcp_fidelity_susceptibility(
        self,
        states: np.ndarray,
        parameters: np.ndarray
    ) -> QCPDetectionResult:
        """
        Detect QCP using fidelity susceptibility.
        
        Fidelity susceptibility χ_F = -∂²F/∂λ² peaks at QCP.
        
        Args:
            states: Quantum states (n_params, n_samples, state_dim)
            parameters: Parameter values (n_params,)
            
        Returns:
            QCPDetectionResult
        """
        n_params = len(parameters)
        fidelities = []
        
        # Compute fidelity between adjacent parameter values
        for i in range(n_params - 1):
            state1 = self._prepare_states(states[i])
            state2 = self._prepare_states(states[i + 1])
            
            with torch.no_grad():
                # Use mean states for comparison
                mu1, _ = self.vae.encode(state1)
                mu2, _ = self.vae.encode(state2)
                
                # Reconstruct from means
                recon1 = self.vae.decode(mu1.mean(dim=0, keepdim=True))
                recon2 = self.vae.decode(mu2.mean(dim=0, keepdim=True))
                
                # Fidelity between reconstructions
                fid = self.vae.quantum_fidelity(recon1, recon2).item()
            
            fidelities.append(fid)
        
        fidelities = np.array(fidelities)
        
        # Compute susceptibility (second derivative)
        dparam = np.diff(parameters)
        dfidelity = -np.log(fidelities + 1e-10)  # -log(F)
        
        # First derivative
        first_deriv = np.gradient(dfidelity, parameters[:-1])
        
        # Second derivative (susceptibility)
        susceptibility = np.abs(np.gradient(first_deriv, parameters[:-1]))
        
        # Find peak
        critical_idx = np.argmax(susceptibility)
        critical_point = parameters[critical_idx]
        
        # Confidence based on peak sharpness
        peak_value = susceptibility[critical_idx]
        mean_value = np.mean(susceptibility)
        confidence = min(0.95, peak_value / (mean_value + 1e-6) / 10)
        
        return QCPDetectionResult(
            critical_point=critical_point,
            confidence=confidence,
            method='fidelity_susceptibility',
            supporting_evidence={
                'susceptibility': susceptibility,
                'fidelities': fidelities,
                'parameters': parameters[:-1]
            }
        )
    
    def detect_qcp_ensemble(
        self,
        states: np.ndarray,
        parameters: np.ndarray
    ) -> QCPDetectionResult:
        """
        Ensemble detection using multiple methods.
        
        Combines results from all detection methods for robust estimate.
        
        Args:
            states: Quantum states (n_params, n_samples, state_dim)
            parameters: Parameter values (n_params,)
            
        Returns:
            Ensemble QCPDetectionResult
        """
        # Run all detection methods
        result_variance = self.detect_qcp_latent_variance(states, parameters)
        result_recon = self.detect_qcp_reconstruction_error(states, parameters)
        result_fidelity = self.detect_qcp_fidelity_susceptibility(states, parameters)
        
        # Weighted average of critical points
        results = [result_variance, result_recon, result_fidelity]
        weights = [r.confidence for r in results]
        total_weight = sum(weights)
        
        critical_point = sum(r.critical_point * w for r, w in zip(results, weights)) / total_weight
        
        # Confidence is average of individual confidences
        confidence = np.mean(weights)
        
        return QCPDetectionResult(
            critical_point=critical_point,
            confidence=confidence,
            method='ensemble',
            supporting_evidence={
                'individual_results': results,
                'weights': weights
            }
        )
    
    def _prepare_states(self, states: np.ndarray) -> torch.Tensor:
        """Prepare states for VAE input."""
        from ..models.quantum_vae import prepare_quantum_state_batch
        return prepare_quantum_state_batch(states, device=self.device)


class CriticalExponentExtractor:
    """
    Extract critical exponents from finite-size scaling analysis.
    
    Uses VAE latent space to identify scaling behavior near QCP.
    """
    
    def __init__(self, vae_model: torch.nn.Module):
        """
        Initialize extractor.
        
        Args:
            vae_model: Trained quantum VAE
        """
        self.vae = vae_model
        self.device = next(vae_model.parameters()).device
    
    def extract_correlation_length_exponent(
        self,
        states_by_size: Dict[int, np.ndarray],
        parameters: np.ndarray,
        critical_point: float
    ) -> Tuple[float, float]:
        """
        Extract correlation length exponent ν.
        
        Uses finite-size scaling: ξ ~ L at criticality
        and ξ ~ |h - hc|^(-ν) away from criticality.
        
        Args:
            states_by_size: Dict mapping system size L to states
            parameters: Parameter values
            critical_point: Known/estimated critical point
            
        Returns:
            (nu, error) tuple
        """
        sizes = sorted(states_by_size.keys())
        
        # For each size, find where latent variance peaks
        peak_positions = []
        
        for L in sizes:
            states = states_by_size[L]
            variances = []
            
            for i in range(len(parameters)):
                state_batch = self._prepare_states(states[i])
                
                with torch.no_grad():
                    mu, _ = self.vae.encode(state_batch)
                    variance = torch.var(mu, dim=0).mean().item()
                
                variances.append(variance)
            
            # Find peak position
            peak_idx = np.argmax(variances)
            peak_positions.append(parameters[peak_idx])
        
        peak_positions = np.array(peak_positions)
        sizes = np.array(sizes)
        
        # Fit: hc(L) - hc(∞) ~ L^(-1/ν)
        def scaling_func(L, nu, hc_inf):
            return hc_inf + (peak_positions[0] - hc_inf) * (L[0] / L) ** (1 / nu)
        
        try:
            popt, pcov = curve_fit(
                scaling_func,
                sizes,
                peak_positions,
                p0=[1.0, critical_point],
                bounds=([0.1, critical_point - 1], [5.0, critical_point + 1])
            )
            nu = popt[0]
            nu_error = np.sqrt(pcov[0, 0])
        except:
            # Fallback: simple power law fit
            log_L = np.log(sizes)
            log_delta = np.log(np.abs(peak_positions - critical_point) + 1e-10)
            
            coeffs = np.polyfit(log_L, log_delta, 1)
            nu = -1.0 / coeffs[0]
            nu_error = 0.5
        
        return nu, nu_error
    
    def extract_dynamical_exponent(
        self,
        states: np.ndarray,
        parameters: np.ndarray,
        critical_point: float,
        system_size: int
    ) -> Tuple[float, float]:
        """
        Extract dynamical exponent z.
        
        At QCP: gap Δ ~ L^(-z)
        
        Args:
            states: Quantum states near criticality
            parameters: Parameter values
            critical_point: Critical point
            system_size: System size L
            
        Returns:
            (z, error) tuple
        """
        # Find states near critical point
        critical_idx = np.argmin(np.abs(parameters - critical_point))
        
        # Use latent space gap as proxy for energy gap
        state_batch = self._prepare_states(states[critical_idx])
        
        with torch.no_grad():
            mu, logvar = self.vae.encode(state_batch)
            
            # Latent space "gap" = spread in first latent dimension
            latent_gap = torch.std(mu[:, 0]).item()
        
        # Estimate z from single size (rough estimate)
        # Would need multiple sizes for accurate extraction
        z_estimate = np.log(latent_gap) / np.log(1.0 / system_size)
        z_error = 0.5  # Large uncertainty without multiple sizes
        
        return abs(z_estimate), z_error
    
    def _prepare_states(self, states: np.ndarray) -> torch.Tensor:
        """Prepare states for VAE input."""
        from ..models.quantum_vae import prepare_quantum_state_batch
        return prepare_quantum_state_batch(states, device=self.device)


def validate_on_clean_tfim(
    vae_model: torch.nn.Module,
    L: int = 12,
    n_points: int = 50
) -> Dict:
    """
    Validate QCP detection on clean TFIM.
    
    Should detect Tc ≈ 1.0 with ν = 1, z = 1.
    
    Args:
        vae_model: Trained VAE
        L: System size
        n_points: Number of parameter points
        
    Returns:
        Validation results dictionary
    """
    from ..quantum.hamiltonian_builder import SpinHamiltonianBuilder, SpinChainParams
    from ..quantum.exact_diagonalization import ExactDiagonalizationSolver
    
    # Generate states across phase transition
    h_values = np.linspace(0.5, 1.5, n_points)
    n_samples = 10
    
    builder = SpinHamiltonianBuilder(L)
    states = []
    solver = ExactDiagonalizationSolver()
    
    for h in h_values:
        params = SpinChainParams(L=L, J=1.0, h=h)
        H = builder.build_tfim(params)
        
        sample_states = []
        for _ in range(n_samples):
            energy, state = solver.ground_state(H)
            sample_states.append(state)
        
        states.append(np.array(sample_states))
    
    states = np.array(states)
    
    # Detect QCP
    detector = QuantumPhaseDetector(vae_model)
    result = detector.detect_qcp_ensemble(states, h_values)
    
    # Extract exponents (would need multiple sizes for accurate ν)
    extractor = CriticalExponentExtractor(vae_model)
    z, z_err = extractor.extract_dynamical_exponent(
        states, h_values, result.critical_point, L
    )
    
    return {
        'detected_Tc': result.critical_point,
        'confidence': result.confidence,
        'expected_Tc': 1.0,
        'error': abs(result.critical_point - 1.0),
        'z_estimate': z,
        'z_error': z_err,
        'expected_z': 1.0
    }
