"""
Improved Critical Exponent Analyzer V2 with Numerical Stability

This module provides an improved critical exponent analyzer that addresses
the numerical stability issues identified in the accuracy assessment.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.stats import pearsonr
import warnings

from .numerical_stability_fixes import (
    safe_log, safe_divide, fit_power_law_safe, clean_data_for_fitting
)
from .latent_analysis import LatentRepresentation


@dataclass
class ImprovedExponentResult:
    """Container for improved exponent extraction results."""
    exponent: float
    exponent_error: float
    amplitude: float
    amplitude_error: float
    r_squared: float
    n_points: int
    success: bool
    message: str
    
    # Additional quality metrics
    relative_error_to_theory: Optional[float] = None
    confidence_level: float = 0.0


@dataclass
class ImprovedCriticalExponentResults:
    """Container for complete improved critical exponent analysis."""
    system_type: str
    critical_temperature: float
    tc_confidence: float
    
    # Critical exponents
    beta_result: Optional[ImprovedExponentResult] = None
    nu_result: Optional[ImprovedExponentResult] = None
    gamma_result: Optional[ImprovedExponentResult] = None
    
    # Overall quality
    overall_accuracy: float = 0.0
    extraction_success: bool = False
    quality_grade: str = 'F'


class ImprovedCriticalExponentAnalyzerV2:
    """
    Improved critical exponent analyzer with numerical stability fixes.
    """
    
    def __init__(self, 
                 system_type: str = 'ising_3d',
                 theoretical_values: Optional[Dict[str, float]] = None):
        """
        Initialize improved analyzer.
        
        Args:
            system_type: Type of physical system
            theoretical_values: Theoretical exponent values for comparison
        """
        self.system_type = system_type
        
        # Default theoretical values for 3D Ising
        if theoretical_values is None:
            self.theoretical_values = {
                'tc': 4.511,
                'beta': 0.326,
                'nu': 0.630,
                'gamma': 1.237
            }
        else:
            self.theoretical_values = theoretical_values
    
    def analyze_with_stability(self, 
                              latent_repr: LatentRepresentation,
                              auto_detect_tc: bool = True) -> ImprovedCriticalExponentResults:
        """
        Analyze critical exponents with numerical stability.
        
        Args:
            latent_repr: Latent representation from VAE
            auto_detect_tc: Whether to auto-detect critical temperature
            
        Returns:
            ImprovedCriticalExponentResults with stable extraction
        """
        # Detect critical temperature
        if auto_detect_tc:
            tc, tc_confidence = self._detect_tc_stable(latent_repr)
        else:
            tc = self.theoretical_values.get('tc', 4.5)
            tc_confidence = 0.5
        
        # Initialize results
        results = ImprovedCriticalExponentResults(
            system_type=self.system_type,
            critical_temperature=tc,
            tc_confidence=tc_confidence
        )
        
        # Extract β exponent
        try:
            beta_result = self._extract_beta_stable(latent_repr, tc)
            results.beta_result = beta_result
        except Exception as e:
            warnings.warn(f"Beta extraction failed: {e}")
            results.beta_result = None
        
        # Compute overall quality
        results.overall_accuracy = self._compute_overall_accuracy(results)
        results.extraction_success = results.beta_result is not None and results.beta_result.success
        results.quality_grade = self._assign_quality_grade(results.overall_accuracy)
        
        return results
    
    def _detect_tc_stable(self, latent_repr: LatentRepresentation) -> Tuple[float, float]:
        """
        Detect critical temperature with numerical stability.
        
        Args:
            latent_repr: Latent representation
            
        Returns:
            Tuple of (tc, confidence)
        """
        temperatures = latent_repr.temperatures
        
        # Use z1 as order parameter (typically most correlated with magnetization)
        order_parameter = np.abs(latent_repr.z1)
        
        # Bin by temperature
        unique_temps = np.unique(temperatures)
        
        if len(unique_temps) < 5:
            # Fallback to middle temperature
            return np.median(unique_temps), 0.3
        
        # Compute susceptibility (variance) at each temperature
        susceptibilities = []
        valid_temps = []
        
        for temp in unique_temps:
            temp_mask = temperatures == temp
            if np.sum(temp_mask) >= 3:
                # Use safe variance calculation
                temp_values = order_parameter[temp_mask]
                if len(temp_values) > 0:
                    susceptibility = np.var(temp_values)
                    if np.isfinite(susceptibility):
                        susceptibilities.append(susceptibility)
                        valid_temps.append(temp)
        
        if len(susceptibilities) < 3:
            return np.median(unique_temps), 0.3
        
        susceptibilities = np.array(susceptibilities)
        valid_temps = np.array(valid_temps)
        
        # Find peak with smoothing
        if len(susceptibilities) > 5:
            # Simple moving average smoothing
            window = 3
            smoothed = np.convolve(susceptibilities, np.ones(window)/window, mode='valid')
            peak_idx = np.argmax(smoothed) + window // 2
        else:
            peak_idx = np.argmax(susceptibilities)
        
        tc_estimate = valid_temps[peak_idx]
        
        # Confidence based on peak prominence
        peak_value = susceptibilities[peak_idx]
        background = np.median(susceptibilities)
        
        if background > 0:
            prominence = (peak_value - background) / background
            confidence = min(1.0, max(0.1, prominence / 3.0))
        else:
            confidence = 0.3
        
        return tc_estimate, confidence
    
    def _extract_beta_stable(self, 
                            latent_repr: LatentRepresentation,
                            tc: float) -> ImprovedExponentResult:
        """
        Extract β exponent with numerical stability.
        
        Args:
            latent_repr: Latent representation
            tc: Critical temperature
            
        Returns:
            ImprovedExponentResult for β exponent
        """
        temperatures = latent_repr.temperatures
        order_parameter = np.abs(latent_repr.z1)
        
        # β: m ∝ (Tc - T)^β for T < Tc
        mask = temperatures < tc
        
        if np.sum(mask) < 5:
            return ImprovedExponentResult(
                exponent=0.0,
                exponent_error=0.0,
                amplitude=0.0,
                amplitude_error=0.0,
                r_squared=0.0,
                n_points=np.sum(mask),
                success=False,
                message='Insufficient data points below Tc'
            )
        
        x_data = tc - temperatures[mask]
        y_data = order_parameter[mask]
        
        # Use safe power-law fitting
        fit_result = fit_power_law_safe(
            x_data, y_data,
            exponent_range=(0.0, 2.0),  # Physical range for β
            remove_outliers=True
        )
        
        # Compute relative error to theory if available
        relative_error = None
        if fit_result['success'] and 'beta' in self.theoretical_values:
            theoretical_beta = self.theoretical_values['beta']
            relative_error = abs(fit_result['exponent'] - theoretical_beta) / theoretical_beta
        
        # Compute confidence level
        confidence = 0.0
        if fit_result['success']:
            # Based on R-squared and number of points
            confidence = fit_result['r_squared'] * min(1.0, fit_result['n_points'] / 20.0)
        
        return ImprovedExponentResult(
            exponent=fit_result['exponent'],
            exponent_error=fit_result['exponent_error'],
            amplitude=fit_result['amplitude'],
            amplitude_error=fit_result['amplitude_error'],
            r_squared=fit_result['r_squared'],
            n_points=fit_result['n_points'],
            success=fit_result['success'],
            message=fit_result['message'],
            relative_error_to_theory=relative_error,
            confidence_level=confidence
        )
    
    def _compute_overall_accuracy(self, results: ImprovedCriticalExponentResults) -> float:
        """
        Compute overall accuracy percentage.
        
        Args:
            results: Extraction results
            
        Returns:
            Overall accuracy as percentage
        """
        accuracy_components = []
        
        # Critical temperature accuracy (20% weight)
        if 'tc' in self.theoretical_values:
            tc_error = abs(results.critical_temperature - self.theoretical_values['tc']) / self.theoretical_values['tc']
            tc_accuracy = max(0, 100 * (1 - tc_error))
            accuracy_components.append(('tc', tc_accuracy, 0.2))
        
        # Beta exponent accuracy (50% weight)
        if results.beta_result and results.beta_result.success:
            if results.beta_result.relative_error_to_theory is not None:
                beta_accuracy = max(0, 100 * (1 - results.beta_result.relative_error_to_theory))
            else:
                # Use R-squared as proxy
                beta_accuracy = results.beta_result.r_squared * 100
            
            accuracy_components.append(('beta', beta_accuracy, 0.5))
        
        # Confidence level (30% weight)
        if results.beta_result:
            confidence_score = results.beta_result.confidence_level * 100
            accuracy_components.append(('confidence', confidence_score, 0.3))
        
        # Compute weighted average
        if accuracy_components:
            total_weight = sum(weight for _, _, weight in accuracy_components)
            weighted_sum = sum(acc * weight for _, acc, weight in accuracy_components)
            return weighted_sum / total_weight
        else:
            return 0.0
    
    def _assign_quality_grade(self, accuracy: float) -> str:
        """Assign quality grade based on accuracy."""
        if accuracy >= 90:
            return 'A'
        elif accuracy >= 80:
            return 'B'
        elif accuracy >= 70:
            return 'C'
        elif accuracy >= 60:
            return 'D'
        else:
            return 'F'


def create_improved_analyzer_v2(system_type: str = 'ising_3d',
                               theoretical_values: Optional[Dict[str, float]] = None) -> ImprovedCriticalExponentAnalyzerV2:
    """
    Factory function to create improved analyzer.
    
    Args:
        system_type: Type of physical system
        theoretical_values: Theoretical exponent values
        
    Returns:
        Configured ImprovedCriticalExponentAnalyzerV2 instance
    """
    return ImprovedCriticalExponentAnalyzerV2(system_type, theoretical_values)