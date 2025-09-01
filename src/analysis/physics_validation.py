"""
Physics Validation Framework

This module provides comprehensive validation functions for comparing discovered
order parameters with theoretical expectations, statistical tests for critical
temperature accuracy, and physics consistency checks throughout the pipeline.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import warnings

from .latent_analysis import LatentRepresentation, LatentAnalyzer
from .phase_detection import PhaseDetectionResult
from .order_parameter_discovery import OrderParameterCandidate
from ..data.ising_simulator import IsingSimulator, SpinConfiguration
from ..utils.logging_utils import get_logger, LoggingContext


@dataclass
class ValidationMetrics:
    """
    Container for physics validation metrics and results.
    
    Attributes:
        order_parameter_correlation: Correlation between discovered and theoretical order parameters
        critical_temperature_error: Absolute error in critical temperature estimation
        critical_temperature_relative_error: Relative error as percentage
        energy_conservation_score: Score for energy conservation (0-1)
        magnetization_conservation_score: Score for magnetization conservation (0-1)
        physics_consistency_score: Overall physics consistency score (0-1)
        statistical_significance: P-values for various statistical tests
        theoretical_comparison: Comparison with known theoretical results
    """
    order_parameter_correlation: float
    critical_temperature_error: float
    critical_temperature_relative_error: float
    energy_conservation_score: float
    magnetization_conservation_score: float
    physics_consistency_score: float
    statistical_significance: Dict[str, float]
    theoretical_comparison: Dict[str, Any]


@dataclass
class ConservationTestResult:
    """Results from conservation law testing."""
    quantity_name: str
    mean_deviation: float
    max_deviation: float
    std_deviation: float
    conservation_score: float
    is_conserved: bool
    tolerance: float


class PhysicsValidator:
    """
    Comprehensive physics validation framework for VAE-discovered physics.
    
    Validates discovered order parameters, critical temperatures, and conservation
    laws against theoretical expectations and known results.
    """
    
    def __init__(self, 
                 theoretical_tc: float = 2.269,  # Onsager's exact solution
                 tolerance_percent: float = 5.0):
        """
        Initialize physics validator.
        
        Args:
            theoretical_tc: Theoretical critical temperature (Onsager solution)
            tolerance_percent: Tolerance percentage for critical temperature validation
        """
        self.theoretical_tc = theoretical_tc
        self.tolerance_percent = tolerance_percent
        self.tolerance_absolute = theoretical_tc * tolerance_percent / 100.0
        self.logger = get_logger(__name__)
        
        self.logger.info(f"Physics validator initialized: T_c = {theoretical_tc:.3f} ± {self.tolerance_absolute:.3f}")
    
    def validate_order_parameter_discovery(self,
                                         latent_repr: LatentRepresentation,
                                         order_param_candidates: List[OrderParameterCandidate]) -> Dict[str, Any]:
        """
        Validate discovered order parameters against theoretical magnetization.
        
        Args:
            latent_repr: Latent space representation
            order_param_candidates: List of order parameter candidates from discovery
            
        Returns:
            Dictionary with validation results and metrics
        """
        self.logger.info("Validating order parameter discovery")
        
        # Calculate theoretical magnetization (absolute value for comparison)
        theoretical_magnetization = np.abs(latent_repr.magnetizations)
        
        # Get best order parameter candidate
        if not order_param_candidates:
            raise ValueError("No order parameter candidates provided")
        
        best_candidate = order_param_candidates[0]  # Assume sorted by confidence
        
        # Get discovered order parameter values
        if best_candidate.latent_dimension == 'z1':
            discovered_order_param = latent_repr.z1
        else:
            discovered_order_param = latent_repr.z2
        
        # Calculate correlation with theoretical magnetization
        correlation_coeff = np.corrcoef(discovered_order_param, theoretical_magnetization)[0, 1]
        
        # Statistical significance test
        n_samples = len(discovered_order_param)
        t_statistic = correlation_coeff * np.sqrt((n_samples - 2) / (1 - correlation_coeff**2))
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), n_samples - 2))
        
        # Temperature-dependent analysis
        temp_analysis = self._analyze_temperature_dependence(
            latent_repr.temperatures,
            discovered_order_param,
            theoretical_magnetization
        )
        
        # Phase transition behavior validation
        phase_behavior = self._validate_phase_transition_behavior(
            latent_repr.temperatures,
            discovered_order_param,
            theoretical_magnetization
        )
        
        validation_result = {
            'correlation_coefficient': float(correlation_coeff),
            'statistical_significance': float(p_value),
            'is_significant': p_value < 0.05,
            'correlation_strength': self._interpret_correlation_strength(correlation_coeff),
            'temperature_dependence': temp_analysis,
            'phase_transition_behavior': phase_behavior,
            'primary_dimension': best_candidate.latent_dimension,
            'best_candidate': {
                'confidence_score': best_candidate.confidence_score,
                'magnetization_correlation': best_candidate.correlation_with_magnetization.correlation_coefficient,
                'is_valid': best_candidate.is_valid_order_parameter
            }
        }
        
        self.logger.info(f"Order parameter correlation: {correlation_coeff:.3f} (p={p_value:.2e})")
        
        return validation_result
    
    def validate_critical_temperature(self,
                                    phase_detection_result: PhaseDetectionResult) -> Dict[str, Any]:
        """
        Validate critical temperature estimation against theoretical value.
        
        Args:
            phase_detection_result: Results from phase detection
            
        Returns:
            Dictionary with critical temperature validation results
        """
        self.logger.info("Validating critical temperature estimation")
        
        discovered_tc = phase_detection_result.critical_temperature
        
        # Calculate errors
        absolute_error = abs(discovered_tc - self.theoretical_tc)
        relative_error = (absolute_error / self.theoretical_tc) * 100.0
        
        # Check if within tolerance
        within_tolerance = absolute_error <= self.tolerance_absolute
        
        # Confidence assessment
        confidence_score = phase_detection_result.confidence
        
        # Transition region analysis
        transition_width = (phase_detection_result.transition_region[1] - 
                          phase_detection_result.transition_region[0])
        
        # Statistical assessment based on method used
        method_reliability = self._assess_method_reliability(phase_detection_result.method)
        
        validation_result = {
            'discovered_tc': float(discovered_tc),
            'theoretical_tc': float(self.theoretical_tc),
            'absolute_error': float(absolute_error),
            'relative_error_percent': float(relative_error),
            'within_tolerance': within_tolerance,
            'tolerance_percent': self.tolerance_percent,
            'confidence_score': float(confidence_score),
            'transition_width': float(transition_width),
            'detection_method': phase_detection_result.method,
            'method_reliability': method_reliability,
            'transition_region': phase_detection_result.transition_region,
            'validation_status': 'PASS' if within_tolerance else 'FAIL'
        }
        
        status = "PASS" if within_tolerance else "FAIL"
        self.logger.info(f"Critical temperature validation: {status} "
                        f"(T_c = {discovered_tc:.3f}, error = {relative_error:.1f}%)")
        
        return validation_result
    
    def test_energy_conservation(self,
                               configurations: List[SpinConfiguration],
                               tolerance: float = 1e-10) -> ConservationTestResult:
        """
        Test energy conservation throughout the simulation pipeline.
        
        Args:
            configurations: List of spin configurations to test
            tolerance: Numerical tolerance for conservation test
            
        Returns:
            ConservationTestResult for energy conservation
        """
        self.logger.info("Testing energy conservation")
        
        if not configurations:
            raise ValueError("No configurations provided for energy conservation test")
        
        # Calculate energy for each configuration using independent method
        calculated_energies = []
        stored_energies = []
        
        for config in configurations:
            # Create simulator to recalculate energy
            simulator = IsingSimulator(
                lattice_size=config.spins.shape,
                temperature=config.temperature
            )
            simulator.lattice = config.spins.copy()
            
            # Calculate energy independently
            calculated_energy = simulator.calculate_energy_per_spin()
            calculated_energies.append(calculated_energy)
            stored_energies.append(config.energy)
        
        calculated_energies = np.array(calculated_energies)
        stored_energies = np.array(stored_energies)
        
        # Calculate deviations
        deviations = np.abs(calculated_energies - stored_energies)
        mean_deviation = np.mean(deviations)
        max_deviation = np.max(deviations)
        std_deviation = np.std(deviations)
        
        # Conservation score (1.0 = perfect conservation)
        conservation_score = np.exp(-mean_deviation / tolerance)
        is_conserved = mean_deviation <= tolerance
        
        result = ConservationTestResult(
            quantity_name='energy',
            mean_deviation=float(mean_deviation),
            max_deviation=float(max_deviation),
            std_deviation=float(std_deviation),
            conservation_score=float(conservation_score),
            is_conserved=is_conserved,
            tolerance=tolerance
        )
        
        self.logger.info(f"Energy conservation: {'PASS' if is_conserved else 'FAIL'} "
                        f"(mean deviation: {mean_deviation:.2e})")
        
        return result
    
    def test_magnetization_conservation(self,
                                      configurations: List[SpinConfiguration],
                                      tolerance: float = 1e-10) -> ConservationTestResult:
        """
        Test magnetization conservation throughout the simulation pipeline.
        
        Args:
            configurations: List of spin configurations to test
            tolerance: Numerical tolerance for conservation test
            
        Returns:
            ConservationTestResult for magnetization conservation
        """
        self.logger.info("Testing magnetization conservation")
        
        if not configurations:
            raise ValueError("No configurations provided for magnetization conservation test")
        
        # Calculate magnetization for each configuration using independent method
        calculated_magnetizations = []
        stored_magnetizations = []
        
        for config in configurations:
            # Calculate magnetization independently
            calculated_mag = np.mean(config.spins)
            calculated_magnetizations.append(calculated_mag)
            stored_magnetizations.append(config.magnetization)
        
        calculated_magnetizations = np.array(calculated_magnetizations)
        stored_magnetizations = np.array(stored_magnetizations)
        
        # Calculate deviations
        deviations = np.abs(calculated_magnetizations - stored_magnetizations)
        mean_deviation = np.mean(deviations)
        max_deviation = np.max(deviations)
        std_deviation = np.std(deviations)
        
        # Conservation score (1.0 = perfect conservation)
        conservation_score = np.exp(-mean_deviation / tolerance)
        is_conserved = mean_deviation <= tolerance
        
        result = ConservationTestResult(
            quantity_name='magnetization',
            mean_deviation=float(mean_deviation),
            max_deviation=float(max_deviation),
            std_deviation=float(std_deviation),
            conservation_score=float(conservation_score),
            is_conserved=is_conserved,
            tolerance=tolerance
        )
        
        self.logger.info(f"Magnetization conservation: {'PASS' if is_conserved else 'FAIL'} "
                        f"(mean deviation: {mean_deviation:.2e})")
        
        return result
    
    def comprehensive_physics_validation(self,
                                       latent_repr: LatentRepresentation,
                                       order_param_candidates: List[OrderParameterCandidate],
                                       phase_detection_result: PhaseDetectionResult,
                                       configurations: Optional[List[SpinConfiguration]] = None) -> ValidationMetrics:
        """
        Perform comprehensive physics validation combining all tests.
        
        Args:
            latent_repr: Latent space representation
            order_param_candidates: List of order parameter candidates
            phase_detection_result: Phase detection results
            configurations: Optional spin configurations for conservation tests
            
        Returns:
            ValidationMetrics with comprehensive validation results
        """
        self.logger.info("Performing comprehensive physics validation")
        
        with LoggingContext("Physics Validation"):
            # Validate order parameter discovery
            order_param_validation = self.validate_order_parameter_discovery(
                latent_repr, order_param_candidates
            )
            
            # Validate critical temperature
            tc_validation = self.validate_critical_temperature(phase_detection_result)
            
            # Test conservation laws if configurations provided
            energy_conservation = None
            magnetization_conservation = None
            
            if configurations:
                energy_conservation = self.test_energy_conservation(configurations)
                magnetization_conservation = self.test_magnetization_conservation(configurations)
            
            # Calculate overall physics consistency score
            physics_score = self._calculate_physics_consistency_score(
                order_param_validation,
                tc_validation,
                energy_conservation,
                magnetization_conservation
            )
            
            # Compile statistical significance results
            statistical_significance = {
                'order_parameter_correlation': order_param_validation['statistical_significance'],
                'critical_temperature_confidence': tc_validation['confidence_score']
            }
            
            # Theoretical comparison
            theoretical_comparison = {
                'onsager_critical_temperature': self.theoretical_tc,
                'discovered_critical_temperature': tc_validation['discovered_tc'],
                'order_parameter_correlation': order_param_validation['correlation_coefficient'],
                'phase_transition_sharpness': tc_validation['transition_width']
            }
            
            metrics = ValidationMetrics(
                order_parameter_correlation=order_param_validation['correlation_coefficient'],
                critical_temperature_error=tc_validation['absolute_error'],
                critical_temperature_relative_error=tc_validation['relative_error_percent'],
                energy_conservation_score=energy_conservation.conservation_score if energy_conservation else 1.0,
                magnetization_conservation_score=magnetization_conservation.conservation_score if magnetization_conservation else 1.0,
                physics_consistency_score=physics_score,
                statistical_significance=statistical_significance,
                theoretical_comparison=theoretical_comparison
            )
        
        self.logger.info(f"Physics validation completed: overall score = {physics_score:.3f}")
        
        return metrics
    
    def _analyze_temperature_dependence(self,
                                      temperatures: np.ndarray,
                                      discovered_param: np.ndarray,
                                      theoretical_param: np.ndarray) -> Dict[str, Any]:
        """Analyze temperature dependence of order parameter."""
        # Bin data by temperature
        unique_temps = np.unique(temperatures)
        
        discovered_means = []
        theoretical_means = []
        correlations_by_temp = []
        
        for temp in unique_temps:
            temp_mask = temperatures == temp
            
            if np.sum(temp_mask) > 1:
                disc_temp = discovered_param[temp_mask]
                theo_temp = theoretical_param[temp_mask]
                
                discovered_means.append(np.mean(disc_temp))
                theoretical_means.append(np.mean(theo_temp))
                
                # Correlation at this temperature
                if np.std(disc_temp) > 0 and np.std(theo_temp) > 0:
                    temp_corr = np.corrcoef(disc_temp, theo_temp)[0, 1]
                    correlations_by_temp.append(temp_corr)
                else:
                    correlations_by_temp.append(0.0)
        
        return {
            'temperatures': unique_temps.tolist(),
            'discovered_means': discovered_means,
            'theoretical_means': theoretical_means,
            'temperature_correlations': correlations_by_temp,
            'overall_temp_correlation': float(np.corrcoef(discovered_means, theoretical_means)[0, 1])
        }
    
    def _validate_phase_transition_behavior(self,
                                          temperatures: np.ndarray,
                                          discovered_param: np.ndarray,
                                          theoretical_param: np.ndarray) -> Dict[str, Any]:
        """Validate phase transition behavior around critical temperature."""
        # Separate into temperature regimes
        low_temp_mask = temperatures < (self.theoretical_tc - 0.2)
        high_temp_mask = temperatures > (self.theoretical_tc + 0.2)
        critical_mask = ~(low_temp_mask | high_temp_mask)
        
        results = {}
        
        for mask, name in [(low_temp_mask, 'low_temperature'),
                           (high_temp_mask, 'high_temperature'),
                           (critical_mask, 'critical_region')]:
            if np.sum(mask) > 0:
                disc_regime = discovered_param[mask]
                theo_regime = theoretical_param[mask]
                
                correlation = np.corrcoef(disc_regime, theo_regime)[0, 1] if len(disc_regime) > 1 else 0.0
                
                results[name] = {
                    'correlation': float(correlation),
                    'n_samples': int(np.sum(mask)),
                    'discovered_mean': float(np.mean(disc_regime)),
                    'discovered_std': float(np.std(disc_regime)),
                    'theoretical_mean': float(np.mean(theo_regime)),
                    'theoretical_std': float(np.std(theo_regime))
                }
        
        return results
    
    def _assess_method_reliability(self, method: str) -> Dict[str, Any]:
        """Assess reliability of phase detection method."""
        reliability_scores = {
            'clustering': 0.8,
            'gradient': 0.9,
            'ensemble': 0.95,
            'information_theoretic': 0.85
        }
        
        method_strengths = {
            'clustering': ['Clear phase separation', 'Visual interpretability'],
            'gradient': ['Sensitive to transitions', 'Mathematical rigor'],
            'ensemble': ['Multiple validation', 'Robust estimates'],
            'information_theoretic': ['Model-independent', 'Theoretical foundation']
        }
        
        method_weaknesses = {
            'clustering': ['Sensitive to noise', 'Requires parameter tuning'],
            'gradient': ['Requires smoothing', 'Sensitive to data quality'],
            'ensemble': ['Complex interpretation', 'Computational cost'],
            'information_theoretic': ['Statistical requirements', 'Implementation complexity']
        }
        
        return {
            'reliability_score': reliability_scores.get(method, 0.5),
            'strengths': method_strengths.get(method, []),
            'weaknesses': method_weaknesses.get(method, [])
        }
    
    def _calculate_physics_consistency_score(self,
                                           order_param_validation: Dict[str, Any],
                                           tc_validation: Dict[str, Any],
                                           energy_conservation: Optional[ConservationTestResult],
                                           magnetization_conservation: Optional[ConservationTestResult]) -> float:
        """Calculate overall physics consistency score."""
        scores = []
        weights = []
        
        # Order parameter correlation score
        corr_coeff = abs(order_param_validation['correlation_coefficient'])
        order_param_score = min(1.0, corr_coeff)  # Cap at 1.0
        scores.append(order_param_score)
        weights.append(0.3)
        
        # Critical temperature accuracy score
        tc_score = 1.0 if tc_validation['within_tolerance'] else 0.0
        # Partial credit based on relative error
        if not tc_validation['within_tolerance']:
            relative_error = tc_validation['relative_error_percent']
            tc_score = max(0.0, 1.0 - (relative_error / (2 * self.tolerance_percent)))
        
        scores.append(tc_score)
        weights.append(0.4)
        
        # Conservation scores
        if energy_conservation:
            scores.append(energy_conservation.conservation_score)
            weights.append(0.15)
        
        if magnetization_conservation:
            scores.append(magnetization_conservation.conservation_score)
            weights.append(0.15)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Calculate weighted average
        overall_score = np.average(scores, weights=weights)
        
        return float(overall_score)
    
    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret correlation coefficient strength."""
        abs_corr = abs(correlation)
        
        if abs_corr >= 0.9:
            return "Very Strong"
        elif abs_corr >= 0.7:
            return "Strong"
        elif abs_corr >= 0.5:
            return "Moderate"
        elif abs_corr >= 0.3:
            return "Weak"
        else:
            return "Very Weak"
    
    def generate_validation_report(self,
                                 validation_metrics: ValidationMetrics,
                                 output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive validation report.
        
        Args:
            validation_metrics: Validation results
            output_path: Optional path to save report
            
        Returns:
            Report text
        """
        report_lines = [
            "=" * 80,
            "PHYSICS VALIDATION REPORT",
            "=" * 80,
            "",
            "SUMMARY",
            "-" * 40,
            f"Overall Physics Consistency Score: {validation_metrics.physics_consistency_score:.3f}",
            f"Order Parameter Correlation: {validation_metrics.order_parameter_correlation:.3f}",
            f"Critical Temperature Error: {validation_metrics.critical_temperature_relative_error:.1f}%",
            f"Energy Conservation Score: {validation_metrics.energy_conservation_score:.3f}",
            f"Magnetization Conservation Score: {validation_metrics.magnetization_conservation_score:.3f}",
            "",
            "DETAILED RESULTS",
            "-" * 40,
            "",
            "Order Parameter Validation:",
            f"  - Correlation with theoretical magnetization: {validation_metrics.order_parameter_correlation:.4f}",
            f"  - Statistical significance (p-value): {validation_metrics.statistical_significance['order_parameter_correlation']:.2e}",
            "",
            "Critical Temperature Validation:",
            f"  - Theoretical T_c (Onsager): {validation_metrics.theoretical_comparison['onsager_critical_temperature']:.3f}",
            f"  - Discovered T_c: {validation_metrics.theoretical_comparison['discovered_critical_temperature']:.3f}",
            f"  - Absolute error: {validation_metrics.critical_temperature_error:.4f}",
            f"  - Relative error: {validation_metrics.critical_temperature_relative_error:.2f}%",
            f"  - Within tolerance ({self.tolerance_percent}%): {'YES' if validation_metrics.critical_temperature_relative_error <= self.tolerance_percent else 'NO'}",
            "",
            "Conservation Laws:",
            f"  - Energy conservation score: {validation_metrics.energy_conservation_score:.4f}",
            f"  - Magnetization conservation score: {validation_metrics.magnetization_conservation_score:.4f}",
            "",
            "VALIDATION STATUS",
            "-" * 40,
        ]
        
        # Overall validation status
        if validation_metrics.physics_consistency_score >= 0.8:
            status = "EXCELLENT - Physics validation passed with high confidence"
        elif validation_metrics.physics_consistency_score >= 0.6:
            status = "GOOD - Physics validation passed with moderate confidence"
        elif validation_metrics.physics_consistency_score >= 0.4:
            status = "FAIR - Physics validation shows mixed results"
        else:
            status = "POOR - Physics validation failed"
        
        report_lines.append(f"Status: {status}")
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(report_text)
            
            self.logger.info(f"Validation report saved to {output_path}")
        
        return report_text