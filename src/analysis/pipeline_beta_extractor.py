"""
Pipeline Beta Exponent Extractor

This module provides a direct integration of the proven fit_power_law_safe method
into the critical exponent extraction pipeline. This addresses the discrepancy
between direct fitting (97% R²) and pipeline fitting (33% accuracy).

Task 2: Fix β exponent extraction pipeline integration
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import warnings

from .numerical_stability_fixes import (
    fit_power_law_safe,
    clean_data_for_fitting,
    safe_log,
    safe_divide
)
from .latent_analysis import LatentRepresentation
from ..utils.logging_utils import get_logger


@dataclass
class PipelineBetaResult:
    """Container for pipeline beta extraction results."""
    exponent: float
    exponent_error: float
    amplitude: float
    amplitude_error: float
    r_squared: float
    n_points: int
    success: bool
    message: str
    fit_range: Tuple[float, float]
    critical_temperature_used: float


class PipelineBetaExtractor:
    """
    Beta exponent extractor that uses the proven direct fitting method.
    
    This class ensures the pipeline uses the same approach that achieves
    97% R² in direct tests, eliminating the integration gap.
    """
    
    def __init__(self, min_points: int = 8):
        """
        Initialize pipeline beta extractor.
        
        Args:
            min_points: Minimum number of points required for fitting
        """
        self.min_points = min_points
        self.logger = get_logger(__name__)
    
    def extract_beta_from_latent(self,
                                 latent_repr: LatentRepresentation,
                                 critical_temperature: float,
                                 order_parameter_dim: int = 0,
                                 fit_range_factor: float = 0.3) -> PipelineBetaResult:
        """
        Extract β exponent from latent representation using proven direct method.
        
        This method applies the same data preparation and fitting approach that
        achieves 97% R² in direct tests.
        
        Args:
            latent_repr: Latent representation from VAE
            critical_temperature: Critical temperature estimate
            order_parameter_dim: Which latent dimension to use (0 for z1, 1 for z2)
            fit_range_factor: Fraction of temperature range to use for fitting
            
        Returns:
            PipelineBetaResult with extraction results
        """
        self.logger.info("Extracting β exponent using proven direct fitting method")
        
        # Step 1: Select order parameter from latent space
        if order_parameter_dim == 0:
            order_parameter = np.abs(latent_repr.z1)
        elif order_parameter_dim == 1:
            order_parameter = np.abs(latent_repr.z2)
        else:
            # Use latent_coords if available
            if hasattr(latent_repr, 'latent_coords'):
                order_parameter = np.abs(latent_repr.latent_coords[:, order_parameter_dim])
            else:
                raise ValueError(f"Invalid order parameter dimension: {order_parameter_dim}")
        
        temperatures = latent_repr.temperatures
        
        # Step 2: Select data below Tc (β exponent definition: m ∝ (Tc - T)^β for T < Tc)
        below_tc_mask = temperatures < critical_temperature
        
        if np.sum(below_tc_mask) < self.min_points:
            self.logger.warning(f"Only {np.sum(below_tc_mask)} points below Tc, using all data")
            below_tc_mask = np.ones_like(temperatures, dtype=bool)
        
        fit_temps = temperatures[below_tc_mask]
        fit_order_param = order_parameter[below_tc_mask]
        
        # Step 3: Define fitting range around Tc
        temp_range = np.max(fit_temps) - np.min(fit_temps)
        fit_range = fit_range_factor * temp_range
        
        # Select points within fitting range
        range_mask = (critical_temperature - fit_temps) <= fit_range
        
        if np.sum(range_mask) < self.min_points:
            # Expand range if needed
            self.logger.warning(f"Expanding fitting range to get minimum points")
            range_mask = np.ones_like(fit_temps, dtype=bool)
        
        final_temps = fit_temps[range_mask]
        final_order_param = fit_order_param[range_mask]
        
        # Step 4: Prepare data for power-law fitting
        # For β: m ∝ (Tc - T)^β
        # So we fit: y = A * x^β where x = (Tc - T) and y = m
        
        x_data = critical_temperature - final_temps
        y_data = final_order_param
        
        # Ensure x_data is positive (we're below Tc)
        if np.any(x_data <= 0):
            self.logger.warning("Some x_data values are non-positive, filtering")
            positive_mask = x_data > 0
            x_data = x_data[positive_mask]
            y_data = y_data[positive_mask]
        
        if len(x_data) < self.min_points:
            return PipelineBetaResult(
                exponent=0.0,
                exponent_error=0.0,
                amplitude=0.0,
                amplitude_error=0.0,
                r_squared=0.0,
                n_points=len(x_data),
                success=False,
                message=f'Insufficient data points: {len(x_data)} < {self.min_points}',
                fit_range=(np.min(final_temps), np.max(final_temps)),
                critical_temperature_used=critical_temperature
            )
        
        # Step 5: Apply proven direct fitting method
        self.logger.info(f"Fitting power law with {len(x_data)} points")
        self.logger.info(f"Temperature range: [{np.min(final_temps):.3f}, {np.max(final_temps):.3f}]")
        self.logger.info(f"Reduced temperature range: [{np.min(x_data):.4f}, {np.max(x_data):.4f}]")
        
        fit_result = fit_power_law_safe(
            x_data,
            y_data,
            exponent_range=(0.0, 2.0),  # Physical range for β exponent
            remove_outliers=True
        )
        
        # Step 6: Package results
        result = PipelineBetaResult(
            exponent=fit_result['exponent'],
            exponent_error=fit_result['exponent_error'],
            amplitude=fit_result['amplitude'],
            amplitude_error=fit_result['amplitude_error'],
            r_squared=fit_result['r_squared'],
            n_points=fit_result['n_points'],
            success=fit_result['success'],
            message=fit_result['message'],
            fit_range=(np.min(final_temps), np.max(final_temps)),
            critical_temperature_used=critical_temperature
        )
        
        if result.success:
            self.logger.info(f"β exponent: {result.exponent:.4f} ± {result.exponent_error:.4f}")
            self.logger.info(f"R²: {result.r_squared:.4f}")
        else:
            self.logger.warning(f"β extraction failed: {result.message}")
        
        return result
    
    def extract_beta_from_magnetization(self,
                                       temperatures: np.ndarray,
                                       magnetizations: np.ndarray,
                                       critical_temperature: float,
                                       fit_range_factor: float = 0.3) -> PipelineBetaResult:
        """
        Extract β exponent from raw magnetization using proven direct method.
        
        This provides a baseline comparison with the latent-based extraction.
        
        Args:
            temperatures: Temperature array
            magnetizations: Magnetization values
            critical_temperature: Critical temperature estimate
            fit_range_factor: Fraction of temperature range to use for fitting
            
        Returns:
            PipelineBetaResult with extraction results
        """
        self.logger.info("Extracting β exponent from raw magnetization")
        
        # Use absolute magnetization
        order_parameter = np.abs(magnetizations)
        
        # Select data below Tc
        below_tc_mask = temperatures < critical_temperature
        
        if np.sum(below_tc_mask) < self.min_points:
            self.logger.warning(f"Only {np.sum(below_tc_mask)} points below Tc, using all data")
            below_tc_mask = np.ones_like(temperatures, dtype=bool)
        
        fit_temps = temperatures[below_tc_mask]
        fit_mags = order_parameter[below_tc_mask]
        
        # Define fitting range
        temp_range = np.max(fit_temps) - np.min(fit_temps)
        fit_range = fit_range_factor * temp_range
        
        range_mask = (critical_temperature - fit_temps) <= fit_range
        
        if np.sum(range_mask) < self.min_points:
            range_mask = np.ones_like(fit_temps, dtype=bool)
        
        final_temps = fit_temps[range_mask]
        final_mags = fit_mags[range_mask]
        
        # Prepare data
        x_data = critical_temperature - final_temps
        y_data = final_mags
        
        # Filter positive x values
        positive_mask = x_data > 0
        x_data = x_data[positive_mask]
        y_data = y_data[positive_mask]
        
        if len(x_data) < self.min_points:
            return PipelineBetaResult(
                exponent=0.0,
                exponent_error=0.0,
                amplitude=0.0,
                amplitude_error=0.0,
                r_squared=0.0,
                n_points=len(x_data),
                success=False,
                message=f'Insufficient data points: {len(x_data)} < {self.min_points}',
                fit_range=(np.min(final_temps), np.max(final_temps)),
                critical_temperature_used=critical_temperature
            )
        
        # Apply direct fitting
        fit_result = fit_power_law_safe(
            x_data,
            y_data,
            exponent_range=(0.0, 2.0),
            remove_outliers=True
        )
        
        result = PipelineBetaResult(
            exponent=fit_result['exponent'],
            exponent_error=fit_result['exponent_error'],
            amplitude=fit_result['amplitude'],
            amplitude_error=fit_result['amplitude_error'],
            r_squared=fit_result['r_squared'],
            n_points=fit_result['n_points'],
            success=fit_result['success'],
            message=fit_result['message'],
            fit_range=(np.min(final_temps), np.max(final_temps)),
            critical_temperature_used=critical_temperature
        )
        
        if result.success:
            self.logger.info(f"β exponent (raw mag): {result.exponent:.4f} ± {result.exponent_error:.4f}")
            self.logger.info(f"R²: {result.r_squared:.4f}")
        
        return result
    
    def compare_latent_vs_magnetization(self,
                                       latent_repr: LatentRepresentation,
                                       critical_temperature: float,
                                       order_parameter_dim: int = 0) -> dict:
        """
        Compare β extraction from latent space vs raw magnetization.
        
        Args:
            latent_repr: Latent representation
            critical_temperature: Critical temperature
            order_parameter_dim: Latent dimension to use
            
        Returns:
            Dictionary with comparison results
        """
        self.logger.info("Comparing latent vs magnetization β extraction")
        
        # Extract from latent
        latent_result = self.extract_beta_from_latent(
            latent_repr, critical_temperature, order_parameter_dim
        )
        
        # Extract from magnetization
        mag_result = self.extract_beta_from_magnetization(
            latent_repr.temperatures,
            latent_repr.magnetizations,
            critical_temperature
        )
        
        comparison = {
            'latent': {
                'exponent': latent_result.exponent,
                'error': latent_result.exponent_error,
                'r_squared': latent_result.r_squared,
                'success': latent_result.success,
                'n_points': latent_result.n_points
            },
            'magnetization': {
                'exponent': mag_result.exponent,
                'error': mag_result.exponent_error,
                'r_squared': mag_result.r_squared,
                'success': mag_result.success,
                'n_points': mag_result.n_points
            }
        }
        
        # Compute improvement
        if latent_result.success and mag_result.success:
            comparison['r_squared_improvement'] = latent_result.r_squared - mag_result.r_squared
            comparison['exponent_difference'] = latent_result.exponent - mag_result.exponent
        
        return comparison


def create_pipeline_beta_extractor(min_points: int = 8) -> PipelineBetaExtractor:
    """
    Factory function to create pipeline beta extractor.
    
    Args:
        min_points: Minimum number of points for fitting
        
    Returns:
        Configured PipelineBetaExtractor instance
    """
    return PipelineBetaExtractor(min_points=min_points)
