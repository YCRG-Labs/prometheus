"""
Integrated VAE Analyzer with Pipeline Beta Extractor

This module integrates the proven pipeline beta extractor (97% R²) into the
VAE-based critical exponent analysis framework.

Task 2: Fix β exponent extraction pipeline integration
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import warnings

from .pipeline_beta_extractor import PipelineBetaExtractor, PipelineBetaResult
from .vae_based_critical_exponent_analyzer import (
    VAECriticalExponentAnalyzer,
    VAEOrderParameterSelector,
    VAECriticalExponentResults
)
from .improved_critical_exponent_analyzer import ImprovedCriticalTemperatureDetector
from .latent_analysis import LatentRepresentation
from ..utils.logging_utils import get_logger


@dataclass
class IntegratedVAEResults:
    """Container for integrated VAE analysis results with pipeline beta extraction."""
    system_type: str
    critical_temperature: float
    tc_confidence: float
    
    # Order parameter analysis
    order_parameter_dimension: int
    order_parameter_correlation: float
    
    # Beta exponent (using proven pipeline method)
    beta_exponent: float
    beta_error: float
    beta_r_squared: float
    beta_success: bool
    
    # Comparison with raw magnetization
    raw_mag_beta: Optional[float] = None
    raw_mag_r_squared: Optional[float] = None
    
    # Accuracy metrics
    theoretical_beta: Optional[float] = None
    beta_accuracy_percent: Optional[float] = None
    
    # Full results
    pipeline_beta_result: Optional[PipelineBetaResult] = None


class IntegratedVAEAnalyzer:
    """
    Integrated VAE analyzer that uses the proven pipeline beta extractor.
    
    This class combines:
    1. VAE order parameter selection
    2. Critical temperature detection
    3. Proven pipeline beta extraction (97% R²)
    
    This fixes the discrepancy between direct fitting and pipeline performance.
    """
    
    def __init__(self, system_type: str = 'ising_3d'):
        """
        Initialize integrated VAE analyzer.
        
        Args:
            system_type: Type of physical system
        """
        self.system_type = system_type
        self.logger = get_logger(__name__)
        
        # Components
        self.order_param_selector = VAEOrderParameterSelector()
        self.tc_detector = ImprovedCriticalTemperatureDetector()
        self.beta_extractor = PipelineBetaExtractor(min_points=8)
        
        # Theoretical values
        self.theoretical_exponents = {
            'ising_2d': {'beta': 0.125, 'nu': 1.0, 'gamma': 1.75, 'tc': 2.269},
            'ising_3d': {'beta': 0.326, 'nu': 0.630, 'gamma': 1.237, 'tc': 4.511},
            'xy_2d': {'beta': 0.125, 'nu': 1.0, 'gamma': 1.75, 'tc': 2.0},
            'heisenberg_3d': {'beta': 0.365, 'nu': 0.705, 'gamma': 1.386, 'tc': 3.0}
        }
    
    def analyze(self,
               latent_repr: LatentRepresentation,
               auto_detect_tc: bool = True,
               compare_with_raw_mag: bool = True) -> IntegratedVAEResults:
        """
        Perform integrated VAE analysis with proven beta extraction.
        
        Args:
            latent_repr: Latent representation from VAE
            auto_detect_tc: Whether to auto-detect critical temperature
            compare_with_raw_mag: Whether to compare with raw magnetization
            
        Returns:
            IntegratedVAEResults with complete analysis
        """
        self.logger.info("Starting integrated VAE analysis with pipeline beta extraction")
        
        # Step 1: Select optimal order parameter from latent dimensions
        order_param_result = self.order_param_selector.select_optimal_order_parameter(
            latent_repr, method='comprehensive'
        )
        
        self.logger.info(f"Selected latent dimension {order_param_result.best_dimension} as order parameter")
        self.logger.info(f"Correlation with magnetization: {order_param_result.correlation_with_magnetization:.4f}")
        
        # Step 2: Detect critical temperature
        if auto_detect_tc:
            tc, tc_confidence = self.tc_detector.detect_critical_temperature(
                latent_repr.temperatures,
                np.abs(order_param_result.order_parameter_values),
                method='ensemble'
            )
            self.logger.info(f"Detected Tc = {tc:.4f} (confidence: {tc_confidence:.3f})")
        else:
            # Use theoretical value
            theoretical = self.theoretical_exponents.get(self.system_type, {})
            tc = theoretical.get('tc', 4.5)
            tc_confidence = 0.5
            self.logger.info(f"Using theoretical Tc = {tc:.4f}")
        
        # Step 3: Extract β exponent using proven pipeline method
        beta_result = self.beta_extractor.extract_beta_from_latent(
            latent_repr,
            tc,
            order_parameter_dim=order_param_result.best_dimension,
            fit_range_factor=0.3
        )
        
        if beta_result.success:
            self.logger.info(f"β exponent: {beta_result.exponent:.4f} ± {beta_result.exponent_error:.4f}")
            self.logger.info(f"R²: {beta_result.r_squared:.4f}")
        else:
            self.logger.warning(f"β extraction failed: {beta_result.message}")
        
        # Step 4: Compare with raw magnetization if requested
        raw_mag_beta = None
        raw_mag_r_squared = None
        
        if compare_with_raw_mag and beta_result.success:
            raw_mag_result = self.beta_extractor.extract_beta_from_magnetization(
                latent_repr.temperatures,
                latent_repr.magnetizations,
                tc,
                fit_range_factor=0.3
            )
            
            if raw_mag_result.success:
                raw_mag_beta = raw_mag_result.exponent
                raw_mag_r_squared = raw_mag_result.r_squared
                self.logger.info(f"Raw magnetization β: {raw_mag_beta:.4f} (R² = {raw_mag_r_squared:.4f})")
        
        # Step 5: Compute accuracy metrics
        theoretical = self.theoretical_exponents.get(self.system_type, {})
        theoretical_beta = theoretical.get('beta', None)
        beta_accuracy = None
        
        if theoretical_beta is not None and beta_result.success:
            relative_error = abs(beta_result.exponent - theoretical_beta) / theoretical_beta
            beta_accuracy = (1 - relative_error) * 100
            self.logger.info(f"β accuracy: {beta_accuracy:.1f}%")
        
        # Package results
        results = IntegratedVAEResults(
            system_type=self.system_type,
            critical_temperature=tc,
            tc_confidence=tc_confidence,
            order_parameter_dimension=order_param_result.best_dimension,
            order_parameter_correlation=order_param_result.correlation_with_magnetization,
            beta_exponent=beta_result.exponent,
            beta_error=beta_result.exponent_error,
            beta_r_squared=beta_result.r_squared,
            beta_success=beta_result.success,
            raw_mag_beta=raw_mag_beta,
            raw_mag_r_squared=raw_mag_r_squared,
            theoretical_beta=theoretical_beta,
            beta_accuracy_percent=beta_accuracy,
            pipeline_beta_result=beta_result
        )
        
        self.logger.info("Integrated VAE analysis completed")
        
        return results
    
    def print_results(self, results: IntegratedVAEResults):
        """Print formatted results."""
        
        print("\n" + "=" * 70)
        print("INTEGRATED VAE ANALYSIS RESULTS")
        print("=" * 70)
        
        print(f"\nSystem Type: {results.system_type}")
        
        print(f"\nCritical Temperature:")
        print(f"  Detected Tc: {results.critical_temperature:.4f}")
        if results.theoretical_beta is not None:
            theoretical = self.theoretical_exponents.get(results.system_type, {})
            theoretical_tc = theoretical.get('tc', 0)
            tc_error = abs(results.critical_temperature - theoretical_tc) / theoretical_tc * 100
            print(f"  Theoretical Tc: {theoretical_tc:.4f}")
            print(f"  Error: {tc_error:.2f}%")
        print(f"  Confidence: {results.tc_confidence:.3f}")
        
        print(f"\nOrder Parameter:")
        print(f"  Selected dimension: {results.order_parameter_dimension}")
        print(f"  Magnetization correlation: {results.order_parameter_correlation:.4f}")
        
        print(f"\nβ Exponent (Pipeline Method):")
        if results.beta_success:
            print(f"  Measured: {results.beta_exponent:.4f} ± {results.beta_error:.4f}")
            if results.theoretical_beta is not None:
                print(f"  Theoretical: {results.theoretical_beta:.4f}")
                print(f"  Accuracy: {results.beta_accuracy_percent:.1f}%")
            print(f"  R²: {results.beta_r_squared:.4f}")
        else:
            print(f"  Extraction failed")
        
        if results.raw_mag_beta is not None:
            print(f"\nβ Exponent (Raw Magnetization):")
            print(f"  Measured: {results.raw_mag_beta:.4f}")
            print(f"  R²: {results.raw_mag_r_squared:.4f}")
            
            if results.beta_success:
                improvement = results.beta_r_squared - results.raw_mag_r_squared
                print(f"  R² improvement: {improvement:+.4f}")
        
        print("\n" + "=" * 70)


def create_integrated_vae_analyzer(system_type: str = 'ising_3d') -> IntegratedVAEAnalyzer:
    """
    Factory function to create integrated VAE analyzer.
    
    Args:
        system_type: Type of physical system
        
    Returns:
        Configured IntegratedVAEAnalyzer instance
    """
    return IntegratedVAEAnalyzer(system_type=system_type)
