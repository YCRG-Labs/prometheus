"""
Enhanced Equilibration Validation for 3D Systems.

This module extends the existing equilibration framework to support 3D systems
with improved convergence monitoring, autocorrelation time calculation, and
equilibration quality metrics specifically designed for 3D Ising models.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass
import time
from scipy import stats

from .equilibration import EquilibrationResult, EquilibrationProtocol
from .enhanced_monte_carlo import EnhancedMonteCarloSimulator, SpinConfiguration3D


@dataclass
class Enhanced3DEquilibrationResult:
    """Enhanced results from 3D equilibration analysis."""
    converged: bool
    equilibration_steps: int
    energy_autocorr_time: float
    magnetization_autocorr_time: float
    final_acceptance_rate: float
    energy_convergence_history: List[float]
    magnetization_convergence_history: List[float]
    convergence_quality_score: float
    equilibration_quality_metrics: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class EquilibrationQualityMetrics:
    """Quality metrics for equilibration assessment."""
    energy_variance_ratio: float  # Ratio of final to initial energy variance
    magnetization_drift: float    # Linear drift in magnetization
    acceptance_rate_stability: float  # Stability of acceptance rate
    autocorr_convergence_score: float  # How well autocorr time converged
    overall_quality_score: float  # Combined quality metric (0-1)


class Enhanced3DEquilibrationProtocol:
    """
    Enhanced equilibration protocol for 3D systems with comprehensive validation.
    
    Provides improved convergence detection through multiple observables,
    robust autocorrelation time estimation, and quality assessment metrics.
    """
    
    def __init__(self,
                 max_steps: int = 100000,
                 min_steps: int = 5000,
                 energy_autocorr_threshold: float = 0.05,
                 magnetization_autocorr_threshold: float = 0.05,
                 convergence_window: int = 500,
                 check_interval: int = 200,
                 quality_threshold: float = 0.7,
                 variance_ratio_threshold: float = 0.1):
        """
        Initialize enhanced 3D equilibration protocol.
        
        Args:
            max_steps: Maximum equilibration steps
            min_steps: Minimum equilibration steps before checking convergence
            energy_autocorr_threshold: Threshold for energy autocorrelation convergence
            magnetization_autocorr_threshold: Threshold for magnetization autocorrelation
            convergence_window: Window size for convergence analysis
            check_interval: Steps between convergence checks
            quality_threshold: Minimum quality score for accepting equilibration
            variance_ratio_threshold: Maximum variance ratio for convergence
        """
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.energy_autocorr_threshold = energy_autocorr_threshold
        self.magnetization_autocorr_threshold = magnetization_autocorr_threshold
        self.convergence_window = convergence_window
        self.check_interval = check_interval
        self.quality_threshold = quality_threshold
        self.variance_ratio_threshold = variance_ratio_threshold
        
        self.logger = logging.getLogger(__name__)
        
        # Configurable thresholds for different system sizes
        self._size_dependent_thresholds = {
            'small': {'max_steps': 50000, 'min_steps': 2000},   # L <= 16
            'medium': {'max_steps': 100000, 'min_steps': 5000}, # 16 < L <= 32
            'large': {'max_steps': 200000, 'min_steps': 10000}  # L > 32
        }
    
    def equilibrate_3d(self, simulator: EnhancedMonteCarloSimulator) -> Enhanced3DEquilibrationResult:
        """
        Equilibrate 3D system with comprehensive convergence monitoring.
        
        Args:
            simulator: EnhancedMonteCarloSimulator for 3D system
            
        Returns:
            Enhanced3DEquilibrationResult with detailed convergence information
        """
        if simulator.dimensions != 3:
            raise ValueError(f"Expected 3D simulator, got {simulator.dimensions}D")
        
        self.logger.info(f"Starting 3D equilibration at T={simulator.temperature:.4f}, "
                        f"lattice_size={simulator.lattice_size}")
        
        # Adjust parameters based on system size
        self._adjust_parameters_for_system_size(simulator.lattice_size)
        
        # Initialize tracking variables
        energy_history = []
        magnetization_history = []
        acceptance_rate_history = []
        step_count = 0
        converged = False
        
        start_time = time.time()
        
        while step_count < self.max_steps:
            # Perform Monte Carlo steps
            initial_accepted = simulator.accepted_moves
            for _ in range(self.check_interval):
                simulator.metropolis_step()
                step_count += 1
            
            # Calculate observables
            energy = simulator.calculate_energy_per_spin()
            magnetization = abs(simulator.calculate_magnetization())  # Use absolute value
            acceptance_rate = simulator.get_acceptance_rate()
            
            energy_history.append(energy)
            magnetization_history.append(magnetization)
            acceptance_rate_history.append(acceptance_rate)
            
            # Check for convergence after minimum steps
            if step_count >= self.min_steps and len(energy_history) >= self.convergence_window:
                convergence_result = self._check_convergence(
                    energy_history, magnetization_history, acceptance_rate_history
                )
                
                if convergence_result['converged']:
                    converged = True
                    self.logger.debug(f"Convergence achieved at step {step_count}")
                    break
            
            # Log progress periodically
            if step_count % (self.check_interval * 10) == 0:
                self.logger.debug(f"Equilibration progress: {step_count}/{self.max_steps} steps, "
                                f"E={energy:.4f}, M={magnetization:.4f}")
        
        equilibration_time = time.time() - start_time
        
        # Calculate final autocorrelation times
        energy_autocorr_time = self._calculate_autocorr_time(energy_history)
        magnetization_autocorr_time = self._calculate_autocorr_time(magnetization_history)
        
        # Compute quality metrics
        quality_metrics = self._compute_quality_metrics(
            energy_history, magnetization_history, acceptance_rate_history
        )
        
        # Overall quality score
        quality_score = self._compute_overall_quality_score(
            energy_autocorr_time, magnetization_autocorr_time, quality_metrics
        )
        
        result = Enhanced3DEquilibrationResult(
            converged=converged and quality_score >= self.quality_threshold,
            equilibration_steps=step_count,
            energy_autocorr_time=energy_autocorr_time,
            magnetization_autocorr_time=magnetization_autocorr_time,
            final_acceptance_rate=simulator.get_acceptance_rate(),
            energy_convergence_history=energy_history,
            magnetization_convergence_history=magnetization_history,
            convergence_quality_score=quality_score,
            equilibration_quality_metrics=quality_metrics.__dict__,
            metadata={
                'temperature': simulator.temperature,
                'lattice_size': simulator.lattice_size,
                'dimensions': simulator.dimensions,
                'equilibration_time_seconds': equilibration_time,
                'max_steps': self.max_steps,
                'min_steps': self.min_steps,
                'check_interval': self.check_interval,
                'convergence_window': self.convergence_window
            }
        )
        
        self.logger.info(f"3D equilibration complete: {step_count} steps, "
                        f"converged={result.converged}, quality={quality_score:.3f}")
        
        return result
    
    def _adjust_parameters_for_system_size(self, lattice_size: Tuple[int, int, int]) -> None:
        """Adjust equilibration parameters based on system size."""
        max_dimension = max(lattice_size)
        
        if max_dimension <= 16:
            size_category = 'small'
        elif max_dimension <= 32:
            size_category = 'medium'
        else:
            size_category = 'large'
        
        thresholds = self._size_dependent_thresholds[size_category]
        self.max_steps = thresholds['max_steps']
        self.min_steps = thresholds['min_steps']
        
        self.logger.debug(f"Adjusted parameters for {size_category} system: "
                         f"max_steps={self.max_steps}, min_steps={self.min_steps}")
    
    def _check_convergence(self, 
                          energy_history: List[float], 
                          magnetization_history: List[float],
                          acceptance_rate_history: List[float]) -> Dict[str, Any]:
        """
        Check convergence based on multiple criteria.
        
        Args:
            energy_history: History of energy values
            magnetization_history: History of magnetization values
            acceptance_rate_history: History of acceptance rates
            
        Returns:
            Dictionary with convergence information
        """
        window_size = min(self.convergence_window, len(energy_history))
        
        # Calculate autocorrelation times for recent window
        recent_energy = energy_history[-window_size:]
        recent_magnetization = magnetization_history[-window_size:]
        
        energy_autocorr = self._calculate_autocorr_time(recent_energy)
        mag_autocorr = self._calculate_autocorr_time(recent_magnetization)
        
        # Check autocorrelation convergence
        energy_converged = energy_autocorr < self.energy_autocorr_threshold * window_size
        mag_converged = mag_autocorr < self.magnetization_autocorr_threshold * window_size
        
        # Check variance stability
        if len(energy_history) >= 2 * window_size:
            early_energy_var = np.var(energy_history[-2*window_size:-window_size])
            recent_energy_var = np.var(recent_energy)
            variance_ratio = abs(recent_energy_var - early_energy_var) / (early_energy_var + 1e-10)
            variance_stable = variance_ratio < self.variance_ratio_threshold
        else:
            variance_stable = False
        
        # Check acceptance rate stability
        if len(acceptance_rate_history) >= window_size:
            recent_acceptance = acceptance_rate_history[-window_size:]
            acceptance_std = np.std(recent_acceptance)
            acceptance_stable = acceptance_std < 0.05  # 5% standard deviation threshold
        else:
            acceptance_stable = False
        
        converged = energy_converged and mag_converged and variance_stable and acceptance_stable
        
        return {
            'converged': converged,
            'energy_converged': energy_converged,
            'magnetization_converged': mag_converged,
            'variance_stable': variance_stable,
            'acceptance_stable': acceptance_stable,
            'energy_autocorr_time': energy_autocorr,
            'magnetization_autocorr_time': mag_autocorr
        }
    
    def _calculate_autocorr_time(self, data: List[float]) -> float:
        """
        Calculate autocorrelation time using improved method.
        
        Args:
            data: Time series data
            
        Returns:
            Autocorrelation time
        """
        if len(data) < 20:
            return float('inf')
        
        data_array = np.array(data)
        
        # Remove linear trend if present
        if len(data_array) > 50:
            x = np.arange(len(data_array))
            slope, intercept, _, _, _ = stats.linregress(x, data_array)
            data_detrended = data_array - (slope * x + intercept)
        else:
            data_detrended = data_array - np.mean(data_array)
        
        # Calculate autocorrelation function using FFT for efficiency
        n = len(data_detrended)
        
        # Pad with zeros to avoid circular correlation
        padded_data = np.zeros(2 * n)
        padded_data[:n] = data_detrended
        
        # FFT-based autocorrelation
        fft_data = np.fft.fft(padded_data)
        autocorr = np.fft.ifft(fft_data * np.conj(fft_data)).real
        autocorr = autocorr[:n]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find integrated autocorrelation time
        # τ_int = 1 + 2 * Σ(C(t)) where sum goes until C(t) becomes negligible
        tau_int = 1.0
        for i in range(1, min(n//4, 100)):  # Don't integrate too far
            if autocorr[i] <= 0 or autocorr[i] < 0.01:  # Stop at first zero or very small value
                break
            tau_int += 2 * autocorr[i]
        
        return tau_int
    
    def _compute_quality_metrics(self, 
                                energy_history: List[float],
                                magnetization_history: List[float],
                                acceptance_rate_history: List[float]) -> EquilibrationQualityMetrics:
        """
        Compute comprehensive quality metrics for equilibration assessment.
        
        Args:
            energy_history: History of energy values
            magnetization_history: History of magnetization values
            acceptance_rate_history: History of acceptance rates
            
        Returns:
            EquilibrationQualityMetrics with computed metrics
        """
        n_points = len(energy_history)
        
        # Energy variance ratio (final vs initial)
        if n_points >= 100:
            initial_window = n_points // 10
            final_window = n_points // 10
            
            initial_energy_var = np.var(energy_history[:initial_window])
            final_energy_var = np.var(energy_history[-final_window:])
            
            if initial_energy_var > 0:
                energy_variance_ratio = final_energy_var / initial_energy_var
            else:
                energy_variance_ratio = 1.0
        else:
            energy_variance_ratio = 1.0
        
        # Magnetization drift (linear trend)
        if n_points >= 50:
            x = np.arange(n_points)
            slope, _, r_value, _, _ = stats.linregress(x, magnetization_history)
            magnetization_drift = abs(slope) * n_points  # Total drift over equilibration
        else:
            magnetization_drift = 0.0
        
        # Acceptance rate stability
        if len(acceptance_rate_history) >= 20:
            acceptance_rate_std = np.std(acceptance_rate_history[-len(acceptance_rate_history)//2:])
            acceptance_rate_stability = max(0, 1 - acceptance_rate_std / 0.1)  # Normalize by 10%
        else:
            acceptance_rate_stability = 0.5
        
        # Autocorrelation convergence score
        energy_autocorr = self._calculate_autocorr_time(energy_history)
        mag_autocorr = self._calculate_autocorr_time(magnetization_history)
        
        # Score based on how reasonable the autocorr times are
        max_reasonable_autocorr = n_points / 10  # Should be much less than total length
        energy_autocorr_score = max(0, 1 - energy_autocorr / max_reasonable_autocorr)
        mag_autocorr_score = max(0, 1 - mag_autocorr / max_reasonable_autocorr)
        autocorr_convergence_score = (energy_autocorr_score + mag_autocorr_score) / 2
        
        # Overall quality score (weighted combination)
        variance_score = max(0, 1 - energy_variance_ratio)  # Lower variance ratio is better
        drift_score = max(0, 1 - magnetization_drift / 0.1)  # Lower drift is better
        
        overall_quality_score = (
            0.3 * variance_score +
            0.2 * drift_score +
            0.3 * acceptance_rate_stability +
            0.2 * autocorr_convergence_score
        )
        
        return EquilibrationQualityMetrics(
            energy_variance_ratio=energy_variance_ratio,
            magnetization_drift=magnetization_drift,
            acceptance_rate_stability=acceptance_rate_stability,
            autocorr_convergence_score=autocorr_convergence_score,
            overall_quality_score=overall_quality_score
        )
    
    def _compute_overall_quality_score(self,
                                     energy_autocorr_time: float,
                                     magnetization_autocorr_time: float,
                                     quality_metrics: EquilibrationQualityMetrics) -> float:
        """
        Compute overall quality score combining all metrics.
        
        Args:
            energy_autocorr_time: Energy autocorrelation time
            magnetization_autocorr_time: Magnetization autocorrelation time
            quality_metrics: Computed quality metrics
            
        Returns:
            Overall quality score (0-1)
        """
        # Base score from quality metrics
        base_score = quality_metrics.overall_quality_score
        
        # Penalty for very large autocorrelation times
        max_reasonable_autocorr = 50  # Reasonable upper limit
        
        if np.isfinite(energy_autocorr_time):
            energy_penalty = min(1.0, energy_autocorr_time / max_reasonable_autocorr)
        else:
            energy_penalty = 1.0
        
        if np.isfinite(magnetization_autocorr_time):
            mag_penalty = min(1.0, magnetization_autocorr_time / max_reasonable_autocorr)
        else:
            mag_penalty = 1.0
        
        autocorr_penalty = (energy_penalty + mag_penalty) / 2
        
        # Final score with penalty
        final_score = base_score * (1 - 0.3 * autocorr_penalty)
        
        return max(0, min(1, final_score))
    
    def validate_equilibration_quality(self, result: Enhanced3DEquilibrationResult) -> Dict[str, Any]:
        """
        Validate the quality of equilibration results.
        
        Args:
            result: Enhanced3DEquilibrationResult to validate
            
        Returns:
            Dictionary with validation results and recommendations
        """
        validation = {
            'is_acceptable': result.converged and result.convergence_quality_score >= self.quality_threshold,
            'quality_score': result.convergence_quality_score,
            'issues': [],
            'recommendations': []
        }
        
        # Check specific issues
        if not result.converged:
            validation['issues'].append("System did not converge within maximum steps")
            validation['recommendations'].append("Increase max_steps or check temperature range")
        
        if result.convergence_quality_score < self.quality_threshold:
            validation['issues'].append(f"Quality score {result.convergence_quality_score:.3f} below threshold {self.quality_threshold}")
        
        if result.energy_autocorr_time > 100:
            validation['issues'].append("Energy autocorrelation time is very large")
            validation['recommendations'].append("Consider longer equilibration or different temperature")
        
        if result.magnetization_autocorr_time > 100:
            validation['issues'].append("Magnetization autocorrelation time is very large")
        
        metrics = result.equilibration_quality_metrics
        if metrics['energy_variance_ratio'] > 2.0:
            validation['issues'].append("Energy variance increased during equilibration")
            validation['recommendations'].append("System may not be properly equilibrated")
        
        if metrics['magnetization_drift'] > 0.1:
            validation['issues'].append("Significant magnetization drift detected")
            validation['recommendations'].append("Longer equilibration may be needed")
        
        return validation


def create_enhanced_3d_equilibration_protocol(system_size: Tuple[int, int, int],
                                            temperature: float) -> Enhanced3DEquilibrationProtocol:
    """
    Create an enhanced 3D equilibration protocol optimized for given system parameters.
    
    Args:
        system_size: 3D lattice size (depth, height, width)
        temperature: Temperature for equilibration
        
    Returns:
        Enhanced3DEquilibrationProtocol configured for the system
    """
    max_dimension = max(system_size)
    n_sites = np.prod(system_size)
    
    # Scale parameters based on system size
    base_max_steps = 50000
    max_steps = int(base_max_steps * (max_dimension / 32) ** 1.5)
    max_steps = min(max_steps, 500000)  # Cap at reasonable limit
    
    min_steps = max(2000, max_steps // 20)
    
    # Adjust thresholds based on temperature (closer to critical = more stringent)
    tc_3d = 4.511
    temp_factor = abs(temperature - tc_3d) / tc_3d
    
    if temp_factor < 0.1:  # Very close to critical temperature
        energy_threshold = 0.02
        mag_threshold = 0.02
        quality_threshold = 0.8
    elif temp_factor < 0.2:  # Near critical temperature
        energy_threshold = 0.05
        mag_threshold = 0.05
        quality_threshold = 0.7
    else:  # Far from critical temperature
        energy_threshold = 0.1
        mag_threshold = 0.1
        quality_threshold = 0.6
    
    return Enhanced3DEquilibrationProtocol(
        max_steps=max_steps,
        min_steps=min_steps,
        energy_autocorr_threshold=energy_threshold,
        magnetization_autocorr_threshold=mag_threshold,
        quality_threshold=quality_threshold,
        convergence_window=min(1000, max_steps // 20),
        check_interval=max(100, max_steps // 500)
    )