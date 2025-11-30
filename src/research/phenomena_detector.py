"""
Novel Phenomena Detector for identifying unusual phase transition behavior.

This module implements detection algorithms for various types of novel phenomena
in phase transitions, including anomalous critical exponents, first-order
transitions, re-entrant phases, and multi-critical points.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .base_types import NovelPhenomenon, VAEAnalysisResults, SimulationData
from ..utils.logging_utils import get_logger


@dataclass
class UniversalityClass:
    """Known universality class with theoretical exponents.
    
    Attributes:
        name: Name of the universality class
        exponents: Dictionary of theoretical critical exponents
        exponent_errors: Typical experimental uncertainties
        description: Description of the universality class
    """
    name: str
    exponents: Dict[str, float]
    exponent_errors: Dict[str, float]
    description: str


class NovelPhenomenonDetector:
    """Detector for novel phase transition phenomena.
    
    This class implements various detection algorithms to identify potentially
    novel phase transition behavior, including:
    - Anomalous critical exponents (>3σ from known universality classes)
    - First-order phase transitions (discontinuities in order parameter)
    - Re-entrant phase transitions
    - Multi-critical points
    
    Attributes:
        universality_classes: Database of known universality classes
        anomaly_threshold: Threshold in standard deviations for anomaly detection
        logger: Logger instance
    """
    
    # Known universality classes with theoretical exponents
    KNOWN_UNIVERSALITY_CLASSES = {
        'ising_2d': UniversalityClass(
            name='2D Ising',
            exponents={'beta': 0.125, 'nu': 1.0, 'gamma': 1.75, 'alpha': 0.0},
            exponent_errors={'beta': 0.01, 'nu': 0.05, 'gamma': 0.05, 'alpha': 0.05},
            description='2D Ising model universality class'
        ),
        'ising_3d': UniversalityClass(
            name='3D Ising',
            exponents={'beta': 0.326, 'nu': 0.630, 'gamma': 1.237, 'alpha': 0.110},
            exponent_errors={'beta': 0.005, 'nu': 0.002, 'gamma': 0.002, 'alpha': 0.006},
            description='3D Ising model universality class'
        ),
        'mean_field': UniversalityClass(
            name='Mean Field',
            exponents={'beta': 0.5, 'nu': 0.5, 'gamma': 1.0, 'alpha': 0.0},
            exponent_errors={'beta': 0.01, 'nu': 0.01, 'gamma': 0.01, 'alpha': 0.01},
            description='Mean field (d > 4) universality class'
        ),
        'xy_2d': UniversalityClass(
            name='2D XY (BKT)',
            exponents={'beta': 0.23, 'nu': 0.67, 'gamma': 1.32, 'alpha': -0.01},
            exponent_errors={'beta': 0.02, 'nu': 0.03, 'gamma': 0.05, 'alpha': 0.05},
            description='2D XY model (Berezinskii-Kosterlitz-Thouless)'
        ),
        'xy_3d': UniversalityClass(
            name='3D XY',
            exponents={'beta': 0.346, 'nu': 0.672, 'gamma': 1.316, 'alpha': -0.013},
            exponent_errors={'beta': 0.003, 'nu': 0.001, 'gamma': 0.001, 'alpha': 0.003},
            description='3D XY model universality class'
        ),
    }
    
    def __init__(
        self,
        anomaly_threshold: float = 3.0,
        custom_universality_classes: Optional[Dict[str, UniversalityClass]] = None
    ):
        """Initialize novel phenomenon detector.
        
        Args:
            anomaly_threshold: Threshold in standard deviations for anomaly detection
            custom_universality_classes: Additional universality classes to check against
        """
        self.anomaly_threshold = anomaly_threshold
        self.logger = get_logger(__name__)
        
        # Combine known and custom universality classes
        self.universality_classes = dict(self.KNOWN_UNIVERSALITY_CLASSES)
        if custom_universality_classes:
            self.universality_classes.update(custom_universality_classes)
        
        self.logger.info(
            f"Initialized NovelPhenomenonDetector with {len(self.universality_classes)} "
            f"universality classes and {anomaly_threshold}σ threshold"
        )
    
    def detect_all_phenomena(
        self,
        vae_results: VAEAnalysisResults,
        simulation_data: Optional[SimulationData] = None
    ) -> List[NovelPhenomenon]:
        """Detect all types of novel phenomena in analysis results.
        
        Args:
            vae_results: VAE analysis results to check
            simulation_data: Optional raw simulation data for additional checks
            
        Returns:
            List of detected novel phenomena
        """
        phenomena = []
        
        # Detect anomalous exponents
        anomalous = self.detect_anomalous_exponents(vae_results)
        phenomena.extend(anomalous)
        
        # Detect first-order transitions (requires simulation data)
        if simulation_data is not None:
            first_order = self.detect_first_order_transition(vae_results, simulation_data)
            if first_order:
                phenomena.append(first_order)
        
        # Log summary
        if phenomena:
            self.logger.info(
                f"Detected {len(phenomena)} novel phenomena at parameters {vae_results.parameters}"
            )
            for p in phenomena:
                self.logger.info(f"  - {p.phenomenon_type}: {p.description}")
        
        return phenomena
    
    def detect_anomalous_exponents(
        self,
        vae_results: VAEAnalysisResults
    ) -> List[NovelPhenomenon]:
        """Detect anomalous critical exponents.
        
        Compares measured exponents against all known universality classes
        and flags those that deviate by more than the threshold.
        
        Args:
            vae_results: VAE analysis results with extracted exponents
            
        Returns:
            List of detected anomalous exponent phenomena
        """
        phenomena = []
        
        # Check each measured exponent against all universality classes
        for exponent_name, measured_value in vae_results.exponents.items():
            # Get measurement error
            measured_error = vae_results.exponent_errors.get(exponent_name, 0.05)
            
            # Find closest universality class
            min_deviation = float('inf')
            closest_class = None
            closest_theoretical = None
            
            for class_name, univ_class in self.universality_classes.items():
                if exponent_name in univ_class.exponents:
                    theoretical_value = univ_class.exponents[exponent_name]
                    theoretical_error = univ_class.exponent_errors.get(exponent_name, 0.01)
                    
                    # Combined error (measurement + theoretical uncertainty)
                    combined_error = np.sqrt(measured_error**2 + theoretical_error**2)
                    
                    # Calculate deviation in standard deviations
                    deviation = abs(measured_value - theoretical_value) / combined_error
                    
                    if deviation < min_deviation:
                        min_deviation = deviation
                        closest_class = class_name
                        closest_theoretical = theoretical_value
            
            # Check if deviation exceeds threshold
            if min_deviation > self.anomaly_threshold and closest_class is not None:
                # Get R² for quality assessment
                r_squared = vae_results.r_squared_values.get(exponent_name, 0.0)
                
                # Only flag if fit quality is reasonable (R² > 0.5)
                if r_squared > 0.5:
                    # Calculate confidence based on deviation and fit quality
                    confidence = min(0.99, (min_deviation / 10.0) * r_squared)
                    
                    phenomenon = NovelPhenomenon(
                        phenomenon_type='anomalous_exponents',
                        variant_id=vae_results.variant_id,
                        parameters=vae_results.parameters,
                        description=(
                            f"{exponent_name} = {measured_value:.4f} ± {measured_error:.4f} "
                            f"deviates {min_deviation:.1f}σ from closest universality class "
                            f"'{closest_class}' ({closest_theoretical:.4f})"
                        ),
                        confidence=confidence,
                        supporting_evidence={
                            'exponent_name': exponent_name,
                            'measured_value': measured_value,
                            'measured_error': measured_error,
                            'closest_class': closest_class,
                            'theoretical_value': closest_theoretical,
                            'deviation_sigma': min_deviation,
                            'r_squared': r_squared,
                            'critical_temperature': vae_results.critical_temperature,
                            'tc_confidence': vae_results.tc_confidence,
                        }
                    )
                    phenomena.append(phenomenon)
                    
                    self.logger.warning(
                        f"Anomalous {exponent_name} exponent detected: "
                        f"{measured_value:.4f} vs {closest_theoretical:.4f} "
                        f"({min_deviation:.1f}σ deviation)"
                    )
        
        return phenomena
    
    def detect_first_order_transition(
        self,
        vae_results: VAEAnalysisResults,
        simulation_data: SimulationData
    ) -> Optional[NovelPhenomenon]:
        """Detect first-order phase transitions.
        
        First-order transitions are characterized by discontinuities in the
        order parameter (magnetization) at the critical temperature.
        
        Args:
            vae_results: VAE analysis results with Tc
            simulation_data: Raw simulation data with magnetizations
            
        Returns:
            NovelPhenomenon if first-order transition detected, None otherwise
        """
        # Get magnetization data
        temperatures = simulation_data.temperatures
        magnetizations = simulation_data.magnetizations  # Shape: (n_temps, n_samples)
        
        # Calculate mean absolute magnetization at each temperature
        mean_mag = np.mean(np.abs(magnetizations), axis=1)
        
        # Find temperature closest to detected Tc
        tc_idx = np.argmin(np.abs(temperatures - vae_results.critical_temperature))
        
        # Check for discontinuity around Tc
        # Look at magnetization change in a window around Tc
        window_size = min(3, len(temperatures) // 4)
        start_idx = max(0, tc_idx - window_size)
        end_idx = min(len(temperatures), tc_idx + window_size + 1)
        
        if end_idx - start_idx < 3:
            # Not enough data points
            return None
        
        # Calculate maximum jump in magnetization
        mag_window = mean_mag[start_idx:end_idx]
        temp_window = temperatures[start_idx:end_idx]
        
        # Find maximum derivative (jump)
        mag_diff = np.abs(np.diff(mag_window))
        max_jump_idx = np.argmax(mag_diff)
        max_jump = mag_diff[max_jump_idx]
        
        # Calculate typical fluctuation size
        mag_std = np.std(magnetizations, axis=1)
        typical_fluctuation = np.mean(mag_std[start_idx:end_idx])
        
        # Threshold for first-order: jump > 5 * typical fluctuation
        first_order_threshold = 5.0 * typical_fluctuation
        
        if max_jump > first_order_threshold:
            # Calculate confidence based on jump size
            confidence = min(0.95, max_jump / (10.0 * typical_fluctuation))
            
            jump_temp = temp_window[max_jump_idx]
            
            phenomenon = NovelPhenomenon(
                phenomenon_type='first_order',
                variant_id=vae_results.variant_id,
                parameters=vae_results.parameters,
                description=(
                    f"First-order phase transition detected at T = {jump_temp:.4f}. "
                    f"Magnetization discontinuity: {max_jump:.4f} "
                    f"({max_jump/typical_fluctuation:.1f}× typical fluctuation)"
                ),
                confidence=confidence,
                supporting_evidence={
                    'transition_temperature': jump_temp,
                    'magnetization_jump': max_jump,
                    'typical_fluctuation': typical_fluctuation,
                    'jump_ratio': max_jump / typical_fluctuation,
                    'detected_tc': vae_results.critical_temperature,
                    'mean_magnetization': mean_mag.tolist(),
                    'temperatures': temperatures.tolist(),
                }
            )
            
            self.logger.warning(
                f"First-order transition detected at T = {jump_temp:.4f} "
                f"(jump = {max_jump:.4f})"
            )
            
            return phenomenon
        
        return None
    
    def detect_re_entrant_transition(
        self,
        vae_results_list: List[VAEAnalysisResults],
        parameter_name: str
    ) -> Optional[NovelPhenomenon]:
        """Detect re-entrant phase transitions.
        
        Re-entrant transitions occur when the system transitions from ordered
        to disordered and back to ordered as a parameter is varied.
        
        Args:
            vae_results_list: List of VAE results at different parameter values
            parameter_name: Name of the parameter being varied
            
        Returns:
            NovelPhenomenon if re-entrant transition detected, None otherwise
        """
        if len(vae_results_list) < 5:
            # Need sufficient points to detect re-entrance
            return None
        
        # Extract parameter values and critical temperatures
        param_values = []
        tc_values = []
        
        for result in vae_results_list:
            if parameter_name in result.parameters:
                param_values.append(result.parameters[parameter_name])
                tc_values.append(result.critical_temperature)
        
        if len(param_values) < 5:
            return None
        
        # Sort by parameter value
        sorted_indices = np.argsort(param_values)
        param_values = np.array(param_values)[sorted_indices]
        tc_values = np.array(tc_values)[sorted_indices]
        
        # Look for non-monotonic behavior in Tc
        # Calculate second derivative to find inflection points
        if len(tc_values) >= 5:
            # Smooth with moving average
            window = 3
            tc_smooth = np.convolve(tc_values, np.ones(window)/window, mode='valid')
            param_smooth = param_values[window//2:-(window//2)]
            
            # Calculate first derivative
            dtc_dparam = np.gradient(tc_smooth, param_smooth)
            
            # Look for sign changes in derivative (indicating re-entrance)
            sign_changes = np.diff(np.sign(dtc_dparam))
            n_sign_changes = np.sum(np.abs(sign_changes) > 0)
            
            if n_sign_changes >= 2:
                # Potential re-entrant behavior
                confidence = min(0.90, n_sign_changes / 4.0)
                
                # Find parameter range where re-entrance occurs
                change_indices = np.where(np.abs(sign_changes) > 0)[0]
                param_range = (
                    param_smooth[change_indices[0]],
                    param_smooth[change_indices[-1]]
                )
                
                phenomenon = NovelPhenomenon(
                    phenomenon_type='re_entrant',
                    variant_id=vae_results_list[0].variant_id,
                    parameters={parameter_name: f"{param_range[0]:.4f} to {param_range[1]:.4f}"},
                    description=(
                        f"Re-entrant phase transition detected in {parameter_name} range "
                        f"[{param_range[0]:.4f}, {param_range[1]:.4f}]. "
                        f"Critical temperature shows {n_sign_changes} reversals."
                    ),
                    confidence=confidence,
                    supporting_evidence={
                        'parameter_name': parameter_name,
                        'parameter_values': param_values.tolist(),
                        'tc_values': tc_values.tolist(),
                        'n_reversals': int(n_sign_changes),
                        'reversal_range': param_range,
                    }
                )
                
                self.logger.warning(
                    f"Re-entrant transition detected in {parameter_name} "
                    f"range [{param_range[0]:.4f}, {param_range[1]:.4f}]"
                )
                
                return phenomenon
        
        return None
    
    def detect_multi_critical_point(
        self,
        vae_results_grid: Dict[Tuple[float, ...], VAEAnalysisResults],
        parameter_names: List[str]
    ) -> Optional[NovelPhenomenon]:
        """Detect multi-critical points in parameter space.
        
        Multi-critical points occur where multiple phase boundaries meet,
        often exhibiting unusual critical behavior.
        
        Args:
            vae_results_grid: Dictionary mapping parameter tuples to VAE results
            parameter_names: Names of parameters in the grid
            
        Returns:
            NovelPhenomenon if multi-critical point detected, None otherwise
        """
        if len(parameter_names) < 2:
            # Need at least 2D parameter space
            return None
        
        if len(vae_results_grid) < 9:
            # Need sufficient grid points
            return None
        
        # Extract grid data
        param_points = list(vae_results_grid.keys())
        tc_values = [vae_results_grid[p].critical_temperature for p in param_points]
        
        # Convert to arrays
        param_array = np.array(param_points)
        tc_array = np.array(tc_values)
        
        # Look for convergence of phase boundaries
        # Calculate gradient of Tc in parameter space
        # A multi-critical point would show gradients converging to zero
        
        # Find point with minimum Tc gradient magnitude
        # (simplified detection - full implementation would use more sophisticated methods)
        
        # For now, flag if we find a point where Tc is locally extremal
        # This is a placeholder for more sophisticated multi-critical detection
        
        self.logger.debug("Multi-critical point detection not fully implemented")
        
        return None
    
    def compare_to_universality_class(
        self,
        vae_results: VAEAnalysisResults,
        class_name: str
    ) -> Tuple[bool, float, Dict[str, float]]:
        """Compare measured exponents to a specific universality class.
        
        Args:
            vae_results: VAE analysis results with measured exponents
            class_name: Name of universality class to compare against
            
        Returns:
            Tuple of (matches, confidence, deviations) where:
            - matches: True if exponents match within threshold
            - confidence: Confidence in the match (0.0 to 1.0)
            - deviations: Dictionary of deviations in sigma for each exponent
        """
        if class_name not in self.universality_classes:
            raise ValueError(f"Unknown universality class: {class_name}")
        
        univ_class = self.universality_classes[class_name]
        deviations = {}
        max_deviation = 0.0
        n_compared = 0
        
        # Compare each available exponent
        for exponent_name, measured_value in vae_results.exponents.items():
            if exponent_name in univ_class.exponents:
                theoretical_value = univ_class.exponents[exponent_name]
                measured_error = vae_results.exponent_errors.get(exponent_name, 0.05)
                theoretical_error = univ_class.exponent_errors.get(exponent_name, 0.01)
                
                # Combined error
                combined_error = np.sqrt(measured_error**2 + theoretical_error**2)
                
                # Deviation in sigma
                deviation = abs(measured_value - theoretical_value) / combined_error
                deviations[exponent_name] = deviation
                
                max_deviation = max(max_deviation, deviation)
                n_compared += 1
        
        if n_compared == 0:
            return False, 0.0, {}
        
        # Match if all deviations are below threshold
        matches = max_deviation < self.anomaly_threshold
        
        # Confidence decreases with deviation
        confidence = max(0.0, 1.0 - (max_deviation / (2 * self.anomaly_threshold)))
        
        return matches, confidence, deviations
    
    def get_closest_universality_class(
        self,
        vae_results: VAEAnalysisResults
    ) -> Tuple[str, float, Dict[str, float]]:
        """Find the closest matching universality class.
        
        Args:
            vae_results: VAE analysis results with measured exponents
            
        Returns:
            Tuple of (class_name, confidence, deviations)
        """
        best_class = None
        best_confidence = 0.0
        best_deviations = {}
        
        for class_name in self.universality_classes.keys():
            matches, confidence, deviations = self.compare_to_universality_class(
                vae_results, class_name
            )
            
            if confidence > best_confidence:
                best_class = class_name
                best_confidence = confidence
                best_deviations = deviations
        
        return best_class, best_confidence, best_deviations
