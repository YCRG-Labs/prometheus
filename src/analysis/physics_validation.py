"""
Physics Validation Framework

This module provides comprehensive validation functions for comparing discovered
order parameters with theoretical expectations, statistical tests for critical
temperature accuracy, and physics consistency checks throughout the pipeline.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import warnings

from .latent_analysis import LatentRepresentation, LatentAnalyzer
from .phase_detection import PhaseDetectionResult
from .order_parameter_discovery import OrderParameterCandidate
from .enhanced_validation_types import (
    CriticalExponentValidation, FiniteSizeScalingResult, UniversalityClass,
    UniversalityClassResult, PhysicsViolation, ViolationSeverity,
    CriticalExponentError, FiniteSizeScalingError, SymmetryValidationResult,
    SymmetryValidationError
)
from .enhanced_validation_config import (
    EnhancedValidationConfig, ValidationLevel, get_default_config,
    create_config_from_dict
)
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
        enhanced_results: Optional enhanced validation results from advanced features
    """
    order_parameter_correlation: float
    critical_temperature_error: float
    critical_temperature_relative_error: float
    energy_conservation_score: float
    magnetization_conservation_score: float
    physics_consistency_score: float
    statistical_significance: Dict[str, float]
    theoretical_comparison: Dict[str, Any]
    enhanced_results: Optional[Dict[str, Any]] = None


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
    
    def create_validation_config(self, 
                               validation_level: str = 'standard',
                               **overrides) -> EnhancedValidationConfig:
        """
        Create a validation configuration with specified level and overrides.
        
        Args:
            validation_level: Validation level ('basic', 'standard', 'comprehensive')
            **overrides: Configuration overrides
            
        Returns:
            Enhanced validation configuration
        """
        from .enhanced_validation_config import config_manager
        return config_manager.create_custom_config(validation_level, **overrides)
    
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
                                       configurations: Optional[List[SpinConfiguration]] = None,
                                       validation_config: Optional[Union[Dict[str, Any], EnhancedValidationConfig]] = None,
                                       multi_size_data: Optional[Dict[int, LatentRepresentation]] = None,
                                       ensemble_data: Optional[List[LatentRepresentation]] = None) -> ValidationMetrics:
        """
        Perform comprehensive physics validation combining all tests including enhanced validation.
        
        Args:
            latent_repr: Latent space representation
            order_param_candidates: List of order parameter candidates
            phase_detection_result: Phase detection results
            configurations: Optional spin configurations for conservation tests
            validation_config: Optional configuration for enhanced validation features
            multi_size_data: Optional data from multiple system sizes for finite-size scaling
            ensemble_data: Optional ensemble data for statistical analysis
            
        Returns:
            ValidationMetrics with comprehensive validation results including enhanced features
        """
        self.logger.info("Performing comprehensive physics validation with enhanced features")
        
        # Initialize validation configuration
        if validation_config is None:
            config = get_default_config('standard')
        elif isinstance(validation_config, EnhancedValidationConfig):
            config = validation_config
        elif isinstance(validation_config, dict):
            # Handle legacy dictionary configuration
            if 'validation_level' in validation_config:
                config = get_default_config(validation_config['validation_level'])
                # Apply any additional overrides from the dictionary
                legacy_dict = config.get_legacy_config_dict()
                legacy_dict.update(validation_config)
                validation_config = legacy_dict
            else:
                config = create_config_from_dict(validation_config)
        else:
            config = get_default_config('standard')
        
        # Extract configuration parameters for backward compatibility
        if isinstance(validation_config, dict):
            # Legacy mode - use dictionary parameters
            validation_level = validation_config.get('validation_level', 'standard')
            enable_enhanced_features = validation_config.get('enable_enhanced_features', True)
            enable_theoretical_validation = validation_config.get('enable_theoretical_validation', True)
            enable_statistical_analysis = validation_config.get('enable_statistical_analysis', True)
            enable_experimental_comparison = validation_config.get('enable_experimental_comparison', False)
            enable_report_generation = validation_config.get('enable_report_generation', True)
        else:
            # New configuration system
            validation_level = config.validation_level.value
            enable_enhanced_features = config.enable_enhanced_features
            enable_theoretical_validation = config.theoretical_models.enable
            enable_statistical_analysis = config.statistical_analysis.enable
            enable_experimental_comparison = config.experimental_comparison.enable
            enable_report_generation = config.report_generation.enable
        
        with LoggingContext("Enhanced Physics Validation"):
            # Basic validation (backward compatible)
            order_param_validation = self.validate_order_parameter_discovery(
                latent_repr, order_param_candidates
            )
            
            tc_validation = self.validate_critical_temperature(phase_detection_result)
            
            # Test conservation laws if configurations provided
            energy_conservation = None
            magnetization_conservation = None
            
            if configurations:
                energy_conservation = self.test_energy_conservation(configurations)
                magnetization_conservation = self.test_magnetization_conservation(configurations)
            
            # Enhanced validation results storage
            enhanced_results = {}
            
            # Enhanced critical exponent validation
            if enable_enhanced_features and config.critical_exponents.enable:
                try:
                    critical_exponent_validation = self.validate_critical_exponents(
                        latent_repr, phase_detection_result
                    )
                    enhanced_results['critical_exponent_validation'] = critical_exponent_validation
                    self.logger.info("Enhanced critical exponent validation completed")
                except Exception as e:
                    self.logger.warning(f"Critical exponent validation failed: {e}")
                    enhanced_results['critical_exponent_validation'] = None
                
                # Finite-size scaling validation if multi-size data available
                if multi_size_data and config.finite_size_scaling.enable:
                    try:
                        finite_size_scaling_result = self.validate_finite_size_scaling(multi_size_data)
                        enhanced_results['finite_size_scaling_result'] = finite_size_scaling_result
                        self.logger.info("Finite-size scaling validation completed")
                    except Exception as e:
                        self.logger.warning(f"Finite-size scaling validation failed: {e}")
                        enhanced_results['finite_size_scaling_result'] = None
                
                # Symmetry validation
                if config.symmetry_validation.enable:
                    try:
                        hamiltonian_symmetries = config.symmetry_validation.hamiltonian_symmetries
                        symmetry_validation = self.validate_symmetry_properties(
                            order_param_candidates, hamiltonian_symmetries
                        )
                        enhanced_results['symmetry_validation'] = symmetry_validation
                        self.logger.info("Symmetry validation completed")
                    except Exception as e:
                        self.logger.warning(f"Symmetry validation failed: {e}")
                        enhanced_results['symmetry_validation'] = None
            
            # Theoretical model validation
            if enable_theoretical_validation and enable_enhanced_features:
                try:
                    from .theoretical_model_validator import TheoreticalModelValidator
                    
                    theoretical_validator = TheoreticalModelValidator(
                        default_dimensionality=config.theoretical_models.dimensionality
                    )
                    
                    # Validate against configured models
                    for model_type in config.theoretical_models.models_to_validate:
                        if model_type.lower() == 'ising':
                            ising_validation = theoretical_validator.validate_against_ising_model(
                                latent_repr, config.theoretical_models.system_size
                            )
                            enhanced_results['ising_model_validation'] = ising_validation
                        elif model_type.lower() == 'xy':
                            xy_validation = theoretical_validator.validate_against_xy_model(latent_repr)
                            enhanced_results['xy_model_validation'] = xy_validation
                        elif model_type.lower() == 'heisenberg':
                            heisenberg_validation = theoretical_validator.validate_against_heisenberg_model(latent_repr)
                            enhanced_results['heisenberg_model_validation'] = heisenberg_validation
                    
                    self.logger.info("Theoretical model validation completed")
                except Exception as e:
                    self.logger.warning(f"Theoretical model validation failed: {e}")
                    enhanced_results['theoretical_model_validation'] = None
            
            # Statistical physics analysis
            if enable_statistical_analysis and enable_enhanced_features:
                try:
                    from .statistical_physics_analyzer import StatisticalPhysicsAnalyzer
                    
                    statistical_analyzer = StatisticalPhysicsAnalyzer(
                        confidence_level=config.statistical_analysis.confidence_level,
                        n_bootstrap_default=config.statistical_analysis.bootstrap_samples,
                        random_seed=config.statistical_analysis.random_seed
                    )
                    
                    # Ensemble analysis if ensemble data available and enabled
                    if ensemble_data and config.statistical_analysis.ensemble_analysis:
                        ensemble_analysis = statistical_analyzer.perform_ensemble_analysis(ensemble_data)
                        enhanced_results['ensemble_analysis'] = ensemble_analysis
                    
                    # Phase boundary uncertainty estimation if enabled
                    if config.statistical_analysis.phase_boundary_uncertainty:
                        phase_boundary_uncertainty = statistical_analyzer.estimate_phase_boundary_uncertainty(
                            latent_repr.temperatures,
                            np.abs(latent_repr.magnetizations)
                        )
                        enhanced_results['phase_boundary_uncertainty'] = phase_boundary_uncertainty
                    
                    # Hypothesis testing for physics properties if enabled
                    if (config.statistical_analysis.hypothesis_testing and 
                        'critical_exponent_validation' in enhanced_results and 
                        enhanced_results['critical_exponent_validation']):
                        crit_exp = enhanced_results['critical_exponent_validation']
                        observed_values = {'beta_exponent': crit_exp.beta_exponent}
                        theoretical_values = {'beta_exponent': crit_exp.beta_theoretical}
                        uncertainties = {'beta_exponent': (crit_exp.beta_confidence_interval[1] - crit_exp.beta_confidence_interval[0]) / 2}
                        
                        hypothesis_tests = statistical_analyzer.test_physics_hypotheses(
                            observed_values, theoretical_values, uncertainties
                        )
                        enhanced_results['hypothesis_tests'] = hypothesis_tests
                    
                    self.logger.info("Statistical physics analysis completed")
                except Exception as e:
                    self.logger.warning(f"Statistical physics analysis failed: {e}")
                    enhanced_results['statistical_analysis'] = None
            
            # Experimental benchmark comparison
            if enable_experimental_comparison and enable_enhanced_features:
                try:
                    from .experimental_benchmark_comparator import ExperimentalBenchmarkComparator
                    
                    exp_comparator = ExperimentalBenchmarkComparator(
                        benchmark_data_path=config.experimental_comparison.custom_benchmark_path
                    )
                    
                    # Prepare computational results for comparison
                    computational_results = {
                        'critical_temperature': phase_detection_result.critical_temperature
                    }
                    
                    if 'critical_exponent_validation' in enhanced_results and enhanced_results['critical_exponent_validation']:
                        crit_exp = enhanced_results['critical_exponent_validation']
                        computational_results['beta_exponent'] = crit_exp.beta_exponent
                        if crit_exp.gamma_exponent is not None:
                            computational_results['gamma_exponent'] = crit_exp.gamma_exponent
                        if crit_exp.nu_exponent is not None:
                            computational_results['nu_exponent'] = crit_exp.nu_exponent
                    
                    # Compare with configured benchmarks
                    experimental_validation = exp_comparator.validate(
                        computational_results, 
                        config.experimental_comparison.benchmark_datasets
                    )
                    enhanced_results['experimental_validation'] = experimental_validation
                    
                    self.logger.info("Experimental benchmark comparison completed")
                except Exception as e:
                    self.logger.warning(f"Experimental benchmark comparison failed: {e}")
                    enhanced_results['experimental_validation'] = None
            
            # Generate comprehensive physics review report
            if enable_report_generation and enable_enhanced_features:
                try:
                    from .physics_review_report_generator import PhysicsReviewReportGenerator
                    
                    report_generator = PhysicsReviewReportGenerator(
                        validation_level=config.report_generation.validation_level
                    )
                    
                    # Compile all validation results for report generation
                    all_validation_results = {
                        'order_parameter_validation': order_param_validation,
                        'critical_temperature_validation': tc_validation,
                        'energy_conservation': energy_conservation,
                        'magnetization_conservation': magnetization_conservation,
                        **enhanced_results
                    }
                    
                    physics_review_report = report_generator.generate_comprehensive_report(
                        all_validation_results,
                        include_educational_content=config.report_generation.include_educational_content
                    )
                    enhanced_results['physics_review_report'] = physics_review_report
                    
                    self.logger.info("Physics review report generated")
                except Exception as e:
                    self.logger.warning(f"Physics review report generation failed: {e}")
                    enhanced_results['physics_review_report'] = None
            
            # Calculate enhanced physics consistency score
            physics_score = self._calculate_enhanced_physics_consistency_score(
                order_param_validation,
                tc_validation,
                energy_conservation,
                magnetization_conservation,
                enhanced_results
            )
            
            # Compile enhanced statistical significance results
            statistical_significance = {
                'order_parameter_correlation': order_param_validation['statistical_significance'],
                'critical_temperature_confidence': tc_validation['confidence_score']
            }
            
            # Add enhanced statistical significance if available
            if 'hypothesis_tests' in enhanced_results and enhanced_results['hypothesis_tests']:
                for test in enhanced_results['hypothesis_tests']:
                    statistical_significance[f"{test.test_name}_p_value"] = test.p_value
            
            # Enhanced theoretical comparison
            theoretical_comparison = {
                'onsager_critical_temperature': self.theoretical_tc,
                'discovered_critical_temperature': tc_validation['discovered_tc'],
                'order_parameter_correlation': order_param_validation['correlation_coefficient'],
                'phase_transition_sharpness': tc_validation['transition_width']
            }
            
            # Add enhanced theoretical comparisons
            if 'critical_exponent_validation' in enhanced_results and enhanced_results['critical_exponent_validation']:
                crit_exp = enhanced_results['critical_exponent_validation']
                theoretical_comparison.update({
                    'beta_exponent_computed': crit_exp.beta_exponent,
                    'beta_exponent_theoretical': crit_exp.beta_theoretical,
                    'universality_class_match': crit_exp.universality_class_match
                })
            
            # Create enhanced ValidationMetrics
            metrics = ValidationMetrics(
                order_parameter_correlation=order_param_validation['correlation_coefficient'],
                critical_temperature_error=tc_validation['absolute_error'],
                critical_temperature_relative_error=tc_validation['relative_error_percent'],
                energy_conservation_score=energy_conservation.conservation_score if energy_conservation else 1.0,
                magnetization_conservation_score=magnetization_conservation.conservation_score if magnetization_conservation else 1.0,
                physics_consistency_score=physics_score,
                statistical_significance=statistical_significance,
                theoretical_comparison=theoretical_comparison,
                enhanced_results=enhanced_results if enhanced_results else None
            )
            
        self.logger.info(f"Enhanced physics validation completed: overall score = {physics_score:.3f}")
        
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
        """Calculate overall physics consistency score (basic version for backward compatibility)."""
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
    
    def _calculate_enhanced_physics_consistency_score(self,
                                                    order_param_validation: Dict[str, Any],
                                                    tc_validation: Dict[str, Any],
                                                    energy_conservation: Optional[ConservationTestResult],
                                                    magnetization_conservation: Optional[ConservationTestResult],
                                                    enhanced_results: Dict[str, Any]) -> float:
        """Calculate enhanced physics consistency score including all validation components."""
        scores = []
        weights = []
        
        # Basic validation scores (reduced weights to accommodate enhanced features)
        corr_coeff = abs(order_param_validation['correlation_coefficient'])
        order_param_score = min(1.0, corr_coeff)
        scores.append(order_param_score)
        weights.append(0.2)
        
        # Critical temperature accuracy score
        tc_score = 1.0 if tc_validation['within_tolerance'] else 0.0
        if not tc_validation['within_tolerance']:
            relative_error = tc_validation['relative_error_percent']
            tc_score = max(0.0, 1.0 - (relative_error / (2 * self.tolerance_percent)))
        
        scores.append(tc_score)
        weights.append(0.25)
        
        # Conservation scores
        if energy_conservation:
            scores.append(energy_conservation.conservation_score)
            weights.append(0.1)
        
        if magnetization_conservation:
            scores.append(magnetization_conservation.conservation_score)
            weights.append(0.1)
        
        # Enhanced validation scores
        
        # Critical exponent validation
        if 'critical_exponent_validation' in enhanced_results and enhanced_results['critical_exponent_validation']:
            crit_exp = enhanced_results['critical_exponent_validation']
            if crit_exp.universality_class_match:
                crit_exp_score = 1.0
            else:
                # Score based on deviation from theoretical values
                beta_deviation = crit_exp.beta_deviation if crit_exp.beta_deviation is not None else 0.5
                crit_exp_score = max(0.0, 1.0 - beta_deviation / 0.3)  # Normalize by 30% deviation
            
            scores.append(crit_exp_score)
            weights.append(0.15)
        
        # Symmetry validation
        if 'symmetry_validation' in enhanced_results and enhanced_results['symmetry_validation']:
            symmetry_val = enhanced_results['symmetry_validation']
            symmetry_score = symmetry_val.symmetry_consistency_score
            scores.append(symmetry_score)
            weights.append(0.1)
        
        # Finite-size scaling validation
        if 'finite_size_scaling_result' in enhanced_results and enhanced_results['finite_size_scaling_result']:
            fss_result = enhanced_results['finite_size_scaling_result']
            fss_score = fss_result.scaling_collapse_quality
            scores.append(fss_score)
            weights.append(0.1)
        
        # Normalize weights to sum to 1
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
    
    def validate_symmetry_properties(self,
                                   order_parameters: List[OrderParameterCandidate],
                                   hamiltonian_symmetries: List[str]) -> SymmetryValidationResult:
        """
        Validate symmetry properties of discovered order parameters.
        
        Args:
            order_parameters: List of order parameter candidates
            hamiltonian_symmetries: List of symmetries of the underlying Hamiltonian
            
        Returns:
            SymmetryValidationResult with symmetry analysis
        """
        self.logger.info("Validating order parameter symmetry properties")
        
        try:
            if not order_parameters:
                raise SymmetryValidationError("No order parameters provided for symmetry validation")
            
            # Get the best order parameter candidate
            best_order_param = order_parameters[0]
            
            # Analyze symmetry breaking
            broken_symmetries = []
            symmetry_order = {}
            
            # For Ising-like systems, check Z2 symmetry breaking
            if 'Z2' in hamiltonian_symmetries:
                # Check if order parameter breaks Z2 symmetry (non-zero mean)
                if hasattr(best_order_param, 'latent_dimension'):
                    if best_order_param.latent_dimension == 'z1':
                        order_param_values = best_order_param.correlation_with_magnetization.latent_values
                    else:
                        # Assume we have access to the values somehow
                        order_param_values = np.random.randn(100)  # Placeholder
                    
                    mean_value = np.mean(order_param_values)
                    if abs(mean_value) > 0.1:  # Threshold for symmetry breaking
                        broken_symmetries.append('Z2')
                        symmetry_order['Z2'] = 2
            
            # For XY-like systems, check U(1) symmetry
            if 'U1' in hamiltonian_symmetries:
                # Check for continuous symmetry breaking
                # This would require more sophisticated analysis
                pass
            
            # For Heisenberg-like systems, check O(3) symmetry
            if 'O3' in hamiltonian_symmetries:
                # Check for vector order parameter
                pass
            
            # Determine order parameter symmetry
            if broken_symmetries:
                order_parameter_symmetry = f"Breaks {', '.join(broken_symmetries)}"
            else:
                order_parameter_symmetry = "Preserves all symmetries"
            
            # Calculate symmetry consistency score
            # This is a simplified metric based on correlation with expected behavior
            if best_order_param.is_valid_order_parameter:
                base_score = 0.8
            else:
                base_score = 0.4
            
            # Adjust based on correlation strength
            correlation_strength = abs(best_order_param.correlation_with_magnetization.correlation_coefficient)
            symmetry_consistency_score = base_score + 0.2 * correlation_strength
            symmetry_consistency_score = min(1.0, symmetry_consistency_score)
            
            # Check for violations
            violations = []
            if symmetry_consistency_score < 0.6:
                violations.append("Low symmetry consistency score")
            
            if not broken_symmetries and 'Z2' in hamiltonian_symmetries:
                # For Ising model, we expect Z2 symmetry breaking below Tc
                violations.append("Expected Z2 symmetry breaking not detected")
            
            result = SymmetryValidationResult(
                broken_symmetries=broken_symmetries,
                symmetry_order=symmetry_order,
                order_parameter_symmetry=order_parameter_symmetry,
                symmetry_consistency_score=symmetry_consistency_score,
                violations=violations
            )
            
            self.logger.info(f"Symmetry validation completed: consistency score = {symmetry_consistency_score:.3f}")
            
            return result
            
        except Exception as e:
            raise SymmetryValidationError(f"Failed to validate symmetry properties: {str(e)}") from e
    
    def validate_critical_exponents(self,
                                  latent_repr: LatentRepresentation,
                                  phase_detection_result: PhaseDetectionResult) -> CriticalExponentValidation:
        """
        Validate critical exponents against theoretical predictions.
        
        Args:
            latent_repr: Latent space representation
            phase_detection_result: Phase detection results
            
        Returns:
            CriticalExponentValidation with computed exponents and validation results
        """
        self.logger.info("Validating critical exponents")
        
        try:
            # Get critical temperature and order parameter data
            tc = phase_detection_result.critical_temperature
            temperatures = latent_repr.temperatures
            
            # Get order parameter (use magnetization as proxy)
            order_parameter = np.abs(latent_repr.magnetizations)
            
            # Compute beta exponent (order parameter scaling)
            beta_result = self._compute_beta_exponent(temperatures, order_parameter, tc)
            
            # Compute gamma exponent (susceptibility scaling) if possible
            gamma_result = self._compute_gamma_exponent(temperatures, order_parameter, tc)
            
            # Compute nu exponent (correlation length scaling) - approximated
            nu_result = self._compute_nu_exponent(temperatures, order_parameter, tc)
            
            # Identify universality class
            universality_result = self._identify_universality_class({
                'beta': beta_result['exponent'],
                'gamma': gamma_result['exponent'] if gamma_result else None,
                'nu': nu_result['exponent'] if nu_result else None
            })
            
            # Check for scaling violations
            scaling_violations = self._check_scaling_violations(
                beta_result, gamma_result, nu_result, universality_result
            )
            
            # Create validation result
            validation = CriticalExponentValidation(
                beta_exponent=beta_result['exponent'],
                beta_theoretical=beta_result['theoretical'],
                beta_confidence_interval=beta_result['confidence_interval'],
                beta_deviation=abs(beta_result['exponent'] - beta_result['theoretical']),
                
                gamma_exponent=gamma_result['exponent'] if gamma_result else None,
                gamma_theoretical=gamma_result['theoretical'] if gamma_result else None,
                gamma_confidence_interval=gamma_result['confidence_interval'] if gamma_result else None,
                gamma_deviation=abs(gamma_result['exponent'] - gamma_result['theoretical']) if gamma_result else None,
                
                nu_exponent=nu_result['exponent'] if nu_result else None,
                nu_theoretical=nu_result['theoretical'] if nu_result else None,
                nu_confidence_interval=nu_result['confidence_interval'] if nu_result else None,
                nu_deviation=abs(nu_result['exponent'] - nu_result['theoretical']) if nu_result else None,
                
                universality_class_match=universality_result['match'],
                identified_universality_class=universality_result['class'],
                scaling_violations=scaling_violations,
                power_law_fit_quality={
                    'beta_r_squared': beta_result['r_squared'],
                    'gamma_r_squared': gamma_result['r_squared'] if gamma_result else None,
                    'nu_r_squared': nu_result['r_squared'] if nu_result else None
                }
            )
            
            self.logger.info(f"Critical exponents computed: β={beta_result['exponent']:.3f}, "
                           f"universality class: {universality_result['class'].value}")
            
            return validation
            
        except Exception as e:
            raise CriticalExponentError(f"Failed to validate critical exponents: {str(e)}") from e
    
    def validate_finite_size_scaling(self,
                                   multi_size_data: Dict[int, LatentRepresentation]) -> FiniteSizeScalingResult:
        """
        Validate finite-size scaling behavior across different system sizes.
        
        Args:
            multi_size_data: Dictionary mapping system sizes to latent representations
            
        Returns:
            FiniteSizeScalingResult with scaling analysis results
        """
        self.logger.info("Validating finite-size scaling")
        
        try:
            if len(multi_size_data) < 2:
                raise FiniteSizeScalingError("Need at least 2 different system sizes for scaling analysis")
            
            system_sizes = sorted(multi_size_data.keys())
            
            # Extract order parameter data for each size
            size_data = {}
            for size in system_sizes:
                latent_repr = multi_size_data[size]
                order_param = np.abs(latent_repr.magnetizations)
                temperatures = latent_repr.temperatures
                size_data[size] = {'temperatures': temperatures, 'order_parameter': order_param}
            
            # Perform scaling collapse analysis
            scaling_result = self._perform_scaling_collapse(size_data, system_sizes)
            
            # Compute correlation length exponent
            nu_result = self._compute_correlation_length_exponent(size_data, system_sizes)
            
            # Check for finite-size corrections
            finite_size_corrections = self._analyze_finite_size_corrections(size_data, system_sizes)
            
            # Generate scaling plots data
            scaling_plots_data = self._generate_scaling_plots_data(size_data, system_sizes, scaling_result)
            
            # Check for scaling violations
            scaling_violations = self._check_finite_size_scaling_violations(
                scaling_result, nu_result, finite_size_corrections
            )
            
            result = FiniteSizeScalingResult(
                system_sizes=system_sizes,
                scaling_collapse_quality=scaling_result['quality'],
                scaling_function_parameters=scaling_result['parameters'],
                correlation_length_exponent=nu_result['exponent'],
                correlation_length_confidence_interval=nu_result['confidence_interval'],
                scaling_plots_data=scaling_plots_data,
                finite_size_corrections=finite_size_corrections,
                scaling_violations=scaling_violations
            )
            
            self.logger.info(f"Finite-size scaling analysis completed: "
                           f"collapse quality = {scaling_result['quality']:.3f}, "
                           f"ν = {nu_result['exponent']:.3f}")
            
            return result
            
        except Exception as e:
            raise FiniteSizeScalingError(f"Failed to validate finite-size scaling: {str(e)}") from e
    
    def validate_symmetry_properties(self,
                                   order_parameters: List[OrderParameterCandidate],
                                   hamiltonian_symmetries: List[str]) -> SymmetryValidationResult:
        """
        Validate symmetry properties of order parameters.
        
        Args:
            order_parameters: List of order parameter candidates
            hamiltonian_symmetries: List of symmetries in the Hamiltonian (e.g., ['Z2', 'translation', 'rotation'])
            
        Returns:
            SymmetryValidationResult with symmetry analysis results
        """
        self.logger.info("Validating symmetry properties of order parameters")
        
        try:
            if not order_parameters:
                raise SymmetryValidationError("No order parameter candidates provided")
            
            # Get the best order parameter candidate
            best_candidate = order_parameters[0]
            
            # Analyze symmetry properties
            broken_symmetries = self._identify_broken_symmetries(best_candidate, hamiltonian_symmetries)
            symmetry_order = self._compute_symmetry_order(best_candidate)
            order_parameter_symmetry = self._classify_order_parameter_symmetry(best_candidate)
            
            # Compute symmetry consistency score
            consistency_score = self._compute_symmetry_consistency_score(
                best_candidate, hamiltonian_symmetries, broken_symmetries
            )
            
            # Analyze symmetry breaking temperature
            symmetry_breaking_temp = self._analyze_symmetry_breaking_temperature(best_candidate)
            
            # Perform continuous and discrete symmetry analysis
            continuous_analysis = self._analyze_continuous_symmetries(best_candidate, hamiltonian_symmetries)
            discrete_analysis = self._analyze_discrete_symmetries(best_candidate, hamiltonian_symmetries)
            
            # Check for symmetry violations
            violations = self._check_symmetry_violations(
                best_candidate, hamiltonian_symmetries, broken_symmetries, consistency_score
            )
            
            result = SymmetryValidationResult(
                broken_symmetries=broken_symmetries,
                symmetry_order=symmetry_order,
                order_parameter_symmetry=order_parameter_symmetry,
                symmetry_consistency_score=consistency_score,
                symmetry_breaking_temperature=symmetry_breaking_temp,
                continuous_symmetry_analysis=continuous_analysis,
                discrete_symmetry_analysis=discrete_analysis,
                violations=violations
            )
            
            self.logger.info(f"Symmetry validation completed: "
                           f"broken symmetries = {broken_symmetries}, "
                           f"consistency score = {consistency_score:.3f}")
            
            return result
            
        except Exception as e:
            raise SymmetryValidationError(f"Failed to validate symmetry properties: {str(e)}") from e
    
    def analyze_order_parameter_correlations(self,
                                           order_parameters: List[OrderParameterCandidate],
                                           latent_repr: LatentRepresentation) -> Dict[str, Any]:
        """
        Analyze correlation functions and decay behavior of order parameters.
        
        Args:
            order_parameters: List of order parameter candidates
            latent_repr: Latent space representation
            
        Returns:
            Dictionary with correlation analysis results
        """
        self.logger.info("Analyzing order parameter correlation functions")
        
        try:
            if not order_parameters:
                raise ValueError("No order parameter candidates provided")
            
            best_candidate = order_parameters[0]
            
            # Get order parameter values
            if best_candidate.latent_dimension == 'z1':
                order_param_values = latent_repr.z1
            else:
                order_param_values = latent_repr.z2
            
            # Compute spatial correlation functions
            spatial_correlations = self._compute_spatial_correlations(order_param_values, latent_repr)
            
            # Analyze correlation decay
            decay_analysis = self._analyze_correlation_decay(spatial_correlations)
            
            # Compute temporal correlations if available
            temporal_correlations = self._compute_temporal_correlations(order_param_values, latent_repr)
            
            # Analyze correlation length
            correlation_length = self._compute_correlation_length(spatial_correlations, decay_analysis)
            
            return {
                'spatial_correlations': spatial_correlations,
                'decay_analysis': decay_analysis,
                'temporal_correlations': temporal_correlations,
                'correlation_length': correlation_length,
                'correlation_function_quality': decay_analysis.get('fit_quality', 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze order parameter correlations: {str(e)}")
            return {}
    
    def analyze_order_parameter_hierarchy(self,
                                        order_parameters: List[OrderParameterCandidate],
                                        latent_repr: LatentRepresentation) -> Dict[str, Any]:
        """
        Analyze hierarchy and coupling between multiple order parameters.
        
        Args:
            order_parameters: List of order parameter candidates
            latent_repr: Latent space representation
            
        Returns:
            Dictionary with hierarchy analysis results
        """
        self.logger.info("Analyzing order parameter hierarchy")
        
        try:
            if len(order_parameters) < 2:
                return {'hierarchy': 'single_order_parameter', 'coupling_analysis': {}}
            
            # Get order parameter values for all candidates
            order_param_data = {}
            for i, candidate in enumerate(order_parameters):
                if candidate.latent_dimension == 'z1':
                    order_param_data[f'param_{i}'] = latent_repr.z1
                else:
                    order_param_data[f'param_{i}'] = latent_repr.z2
            
            # Analyze coupling between order parameters
            coupling_analysis = self._analyze_order_parameter_coupling(order_param_data, latent_repr)
            
            # Determine hierarchy (primary vs secondary order parameters)
            hierarchy_analysis = self._determine_order_parameter_hierarchy(
                order_parameters, order_param_data, coupling_analysis
            )
            
            # Analyze competition between order parameters
            competition_analysis = self._analyze_order_parameter_competition(
                order_param_data, latent_repr.temperatures
            )
            
            return {
                'hierarchy': hierarchy_analysis,
                'coupling_analysis': coupling_analysis,
                'competition_analysis': competition_analysis,
                'n_order_parameters': len(order_parameters)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze order parameter hierarchy: {str(e)}")
            return {}
    
    def _identify_broken_symmetries(self,
                                  order_parameter: OrderParameterCandidate,
                                  hamiltonian_symmetries: List[str]) -> List[str]:
        """Identify which symmetries are broken by the order parameter."""
        broken_symmetries = []
        
        # For Ising model, check Z2 symmetry breaking
        if 'Z2' in hamiltonian_symmetries:
            # If order parameter has non-zero mean, Z2 symmetry is broken
            if hasattr(order_parameter, 'temperature_dependence'):
                temp_dep = order_parameter.temperature_dependence
                if 'low_temperature_mean' in temp_dep and abs(temp_dep['low_temperature_mean']) > 0.1:
                    broken_symmetries.append('Z2')
        
        # Check translational symmetry
        if 'translation' in hamiltonian_symmetries:
            # For now, assume translational symmetry is preserved (would need spatial analysis)
            pass
        
        # Check rotational symmetry
        if 'rotation' in hamiltonian_symmetries:
            # For Ising model, rotational symmetry is already broken by the Hamiltonian
            pass
        
        return broken_symmetries
    
    def _compute_symmetry_order(self, order_parameter: OrderParameterCandidate) -> Dict[str, int]:
        """Compute the order of various symmetries."""
        symmetry_order = {}
        
        # For Ising model, Z2 symmetry has order 2
        symmetry_order['Z2'] = 2
        
        # Translational symmetry order depends on lattice (assume infinite for continuous)
        symmetry_order['translation'] = -1  # -1 indicates continuous symmetry
        
        return symmetry_order
    
    def _classify_order_parameter_symmetry(self, order_parameter: OrderParameterCandidate) -> str:
        """Classify the symmetry of the order parameter itself."""
        # For Ising model, the order parameter is a scalar (Z2 symmetry)
        if order_parameter.correlation_with_magnetization.is_significant:
            return 'scalar_Z2'
        else:
            return 'unknown'
    
    def _compute_symmetry_consistency_score(self,
                                          order_parameter: OrderParameterCandidate,
                                          hamiltonian_symmetries: List[str],
                                          broken_symmetries: List[str]) -> float:
        """Compute a score for symmetry consistency."""
        score = 1.0
        
        # Check if the right symmetries are broken
        if 'Z2' in hamiltonian_symmetries and 'Z2' not in broken_symmetries:
            # For Ising model below Tc, Z2 should be broken
            if hasattr(order_parameter, 'critical_behavior'):
                crit_behavior = order_parameter.critical_behavior
                if 'below_tc_behavior' in crit_behavior:
                    score *= 0.5  # Penalize for not breaking Z2 below Tc
        
        # Reward high correlation with magnetization for Ising model
        if order_parameter.correlation_with_magnetization.is_significant:
            corr_strength = abs(order_parameter.correlation_with_magnetization.correlation_coefficient)
            score *= corr_strength
        
        return min(1.0, score)
    
    def _analyze_symmetry_breaking_temperature(self, order_parameter: OrderParameterCandidate) -> Optional[float]:
        """Analyze the temperature at which symmetry breaking occurs."""
        if hasattr(order_parameter, 'critical_behavior'):
            crit_behavior = order_parameter.critical_behavior
            if 'critical_temperature' in crit_behavior:
                return crit_behavior['critical_temperature']
        return None
    
    def _analyze_continuous_symmetries(self,
                                     order_parameter: OrderParameterCandidate,
                                     hamiltonian_symmetries: List[str]) -> Dict[str, Any]:
        """Analyze continuous symmetries (e.g., rotational, translational)."""
        continuous_analysis = {}
        
        # Translational symmetry analysis
        if 'translation' in hamiltonian_symmetries:
            continuous_analysis['translational'] = {
                'preserved': True,  # Assume preserved without spatial analysis
                'analysis_method': 'assumed_homogeneous'
            }
        
        # Rotational symmetry analysis
        if 'rotation' in hamiltonian_symmetries:
            continuous_analysis['rotational'] = {
                'preserved': False,  # Ising model breaks rotational symmetry
                'analysis_method': 'hamiltonian_structure'
            }
        
        return continuous_analysis
    
    def _analyze_discrete_symmetries(self,
                                   order_parameter: OrderParameterCandidate,
                                   hamiltonian_symmetries: List[str]) -> Dict[str, Any]:
        """Analyze discrete symmetries (e.g., Z2, reflection)."""
        discrete_analysis = {}
        
        # Z2 symmetry analysis
        if 'Z2' in hamiltonian_symmetries:
            # Check if order parameter breaks Z2 symmetry
            z2_broken = False
            if hasattr(order_parameter, 'temperature_dependence'):
                temp_dep = order_parameter.temperature_dependence
                if 'low_temperature_mean' in temp_dep:
                    z2_broken = abs(temp_dep['low_temperature_mean']) > 0.1
            
            discrete_analysis['Z2'] = {
                'broken': z2_broken,
                'breaking_mechanism': 'spontaneous' if z2_broken else 'none',
                'order_parameter_sign': 'positive' if z2_broken else 'zero'
            }
        
        return discrete_analysis
    
    def _check_symmetry_violations(self,
                                 order_parameter: OrderParameterCandidate,
                                 hamiltonian_symmetries: List[str],
                                 broken_symmetries: List[str],
                                 consistency_score: float) -> List[str]:
        """Check for symmetry-related violations."""
        violations = []
        
        # Check if consistency score is too low
        if consistency_score < 0.5:
            violations.append("Low symmetry consistency score - order parameter may not respect expected symmetries")
        
        # Check if expected symmetries are broken
        if 'Z2' in hamiltonian_symmetries and 'Z2' not in broken_symmetries:
            violations.append("Z2 symmetry not broken - unexpected for Ising model below critical temperature")
        
        # Check if unexpected symmetries are broken
        for broken_sym in broken_symmetries:
            if broken_sym not in hamiltonian_symmetries:
                violations.append(f"Unexpected symmetry breaking: {broken_sym}")
        
        return violations
    
    def _compute_spatial_correlations(self,
                                    order_param_values: np.ndarray,
                                    latent_repr: LatentRepresentation) -> Dict[str, Any]:
        """Compute spatial correlation functions."""
        # Since we don't have explicit spatial structure in latent space,
        # we'll compute correlations based on temperature proximity as a proxy
        temperatures = latent_repr.temperatures
        unique_temps = np.unique(temperatures)
        
        correlation_data = {}
        
        for temp in unique_temps:
            temp_mask = temperatures == temp
            temp_order_params = order_param_values[temp_mask]
            
            if len(temp_order_params) > 1:
                # Compute autocorrelation function
                autocorr = np.correlate(temp_order_params, temp_order_params, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                autocorr = autocorr / autocorr[0]  # Normalize
                
                correlation_data[f'temp_{temp:.3f}'] = {
                    'autocorrelation': autocorr[:min(10, len(autocorr))],  # Keep first 10 lags
                    'correlation_length_estimate': self._estimate_correlation_length_from_autocorr(autocorr)
                }
        
        return {
            'temperature_based_correlations': correlation_data,
            'method': 'temperature_proximity_proxy',
            'note': 'Using temperature proximity as spatial proxy due to latent space structure'
        }
    
    def _analyze_correlation_decay(self, spatial_correlations: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the decay behavior of correlation functions."""
        decay_analysis = {
            'decay_type': 'unknown',
            'correlation_lengths': {},
            'fit_quality': 0.0,
            'temperature_dependence': {}
        }
        
        if 'temperature_based_correlations' in spatial_correlations:
            temp_correlations = spatial_correlations['temperature_based_correlations']
            
            correlation_lengths = []
            fit_qualities = []
            
            for temp_key, temp_data in temp_correlations.items():
                if 'correlation_length_estimate' in temp_data:
                    corr_length = temp_data['correlation_length_estimate']
                    if corr_length is not None:
                        correlation_lengths.append(corr_length)
                        decay_analysis['correlation_lengths'][temp_key] = corr_length
                
                # Analyze autocorrelation decay
                if 'autocorrelation' in temp_data:
                    autocorr = temp_data['autocorrelation']
                    decay_fit = self._fit_exponential_decay(autocorr)
                    if decay_fit:
                        fit_qualities.append(decay_fit['r_squared'])
                        decay_analysis['temperature_dependence'][temp_key] = decay_fit
            
            if correlation_lengths:
                decay_analysis['mean_correlation_length'] = np.mean(correlation_lengths)
                decay_analysis['correlation_length_std'] = np.std(correlation_lengths)
            
            if fit_qualities:
                decay_analysis['fit_quality'] = np.mean(fit_qualities)
                decay_analysis['decay_type'] = 'exponential' if np.mean(fit_qualities) > 0.8 else 'non_exponential'
        
        return decay_analysis
    
    def _compute_temporal_correlations(self,
                                     order_param_values: np.ndarray,
                                     latent_repr: LatentRepresentation) -> Dict[str, Any]:
        """Compute temporal correlation functions if time series data is available."""
        # Since we don't have explicit temporal structure, we'll analyze
        # correlations based on the sequence of data points as a proxy
        
        temporal_analysis = {}
        
        # Compute lagged correlations
        max_lag = min(50, len(order_param_values) // 4)
        lagged_correlations = []
        
        for lag in range(1, max_lag + 1):
            if lag < len(order_param_values):
                corr = np.corrcoef(order_param_values[:-lag], order_param_values[lag:])[0, 1]
                if not np.isnan(corr):
                    lagged_correlations.append(corr)
                else:
                    lagged_correlations.append(0.0)
        
        temporal_analysis['lagged_correlations'] = lagged_correlations
        temporal_analysis['max_lag'] = max_lag
        
        # Analyze persistence (how long correlations remain significant)
        persistence_threshold = 0.1
        persistence_length = 0
        for i, corr in enumerate(lagged_correlations):
            if abs(corr) > persistence_threshold:
                persistence_length = i + 1
            else:
                break
        
        temporal_analysis['persistence_length'] = persistence_length
        temporal_analysis['persistence_threshold'] = persistence_threshold
        
        # Fit exponential decay to temporal correlations
        if lagged_correlations:
            decay_fit = self._fit_exponential_decay(np.array(lagged_correlations))
            temporal_analysis['decay_fit'] = decay_fit
        
        return temporal_analysis
    
    def _compute_correlation_length(self,
                                  spatial_correlations: Dict[str, Any],
                                  decay_analysis: Dict[str, Any]) -> Optional[float]:
        """Compute correlation length from correlation function decay."""
        if 'mean_correlation_length' in decay_analysis:
            return decay_analysis['mean_correlation_length']
        
        # Try to extract from temperature dependence
        if 'temperature_dependence' in decay_analysis:
            temp_dep = decay_analysis['temperature_dependence']
            correlation_lengths = []
            
            for temp_data in temp_dep.values():
                if 'correlation_length' in temp_data:
                    correlation_lengths.append(temp_data['correlation_length'])
            
            if correlation_lengths:
                return np.mean(correlation_lengths)
        
        return None
    
    def _analyze_order_parameter_coupling(self,
                                        order_param_data: Dict[str, np.ndarray],
                                        latent_repr: LatentRepresentation) -> Dict[str, Any]:
        """Analyze coupling between multiple order parameters."""
        coupling_analysis = {}
        
        param_names = list(order_param_data.keys())
        if len(param_names) >= 2:
            # Compute cross-correlations between order parameters
            param1_data = order_param_data[param_names[0]]
            param2_data = order_param_data[param_names[1]]
            
            cross_correlation = np.corrcoef(param1_data, param2_data)[0, 1]
            
            coupling_analysis['cross_correlation'] = float(cross_correlation)
            coupling_analysis['coupling_strength'] = 'strong' if abs(cross_correlation) > 0.7 else 'weak'
            
            # Analyze temperature dependence of coupling
            temperatures = latent_repr.temperatures
            unique_temps = np.unique(temperatures)
            
            temp_coupling = []
            for temp in unique_temps:
                temp_mask = temperatures == temp
                if np.sum(temp_mask) > 1:
                    temp_corr = np.corrcoef(param1_data[temp_mask], param2_data[temp_mask])[0, 1]
                    temp_coupling.append(temp_corr)
            
            coupling_analysis['temperature_dependence'] = temp_coupling
        
        return coupling_analysis
    
    def _determine_order_parameter_hierarchy(self,
                                           order_parameters: List[OrderParameterCandidate],
                                           order_param_data: Dict[str, np.ndarray],
                                           coupling_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine hierarchy between order parameters (primary vs secondary)."""
        hierarchy = {}
        
        # Rank order parameters by confidence score
        ranked_params = sorted(enumerate(order_parameters), 
                             key=lambda x: x[1].confidence_score, reverse=True)
        
        hierarchy['primary_order_parameter'] = {
            'index': ranked_params[0][0],
            'confidence_score': ranked_params[0][1].confidence_score,
            'dimension': ranked_params[0][1].latent_dimension
        }
        
        if len(ranked_params) > 1:
            hierarchy['secondary_order_parameter'] = {
                'index': ranked_params[1][0],
                'confidence_score': ranked_params[1][1].confidence_score,
                'dimension': ranked_params[1][1].latent_dimension
            }
            
            # Determine relationship type
            if 'cross_correlation' in coupling_analysis:
                cross_corr = coupling_analysis['cross_correlation']
                if abs(cross_corr) > 0.8:
                    hierarchy['relationship'] = 'strongly_coupled'
                elif abs(cross_corr) > 0.5:
                    hierarchy['relationship'] = 'moderately_coupled'
                else:
                    hierarchy['relationship'] = 'weakly_coupled'
        
        return hierarchy
    
    def _analyze_order_parameter_competition(self,
                                           order_param_data: Dict[str, np.ndarray],
                                           temperatures: np.ndarray) -> Dict[str, Any]:
        """Analyze competition between order parameters."""
        competition_analysis = {}
        
        if len(order_param_data) >= 2:
            param_names = list(order_param_data.keys())
            param1_data = order_param_data[param_names[0]]
            param2_data = order_param_data[param_names[1]]
            
            # Analyze which parameter dominates at different temperatures
            unique_temps = np.unique(temperatures)
            dominance_pattern = []
            
            for temp in unique_temps:
                temp_mask = temperatures == temp
                if np.sum(temp_mask) > 0:
                    param1_strength = np.mean(np.abs(param1_data[temp_mask]))
                    param2_strength = np.mean(np.abs(param2_data[temp_mask]))
                    
                    if param1_strength > param2_strength:
                        dominance_pattern.append('param_0')
                    else:
                        dominance_pattern.append('param_1')
            
            competition_analysis['dominance_pattern'] = dominance_pattern
            competition_analysis['temperatures'] = unique_temps.tolist()
            
            # Check for competition signatures (anti-correlation)
            overall_correlation = np.corrcoef(param1_data, param2_data)[0, 1]
            competition_analysis['competition_signature'] = overall_correlation < -0.3
        
        return competition_analysis
    
    def _estimate_correlation_length_from_autocorr(self, autocorr: np.ndarray) -> Optional[float]:
        """Estimate correlation length from autocorrelation function."""
        try:
            # Find where autocorrelation drops to 1/e
            threshold = 1.0 / np.e
            
            # Find first point where autocorr drops below threshold
            below_threshold = np.where(autocorr < threshold)[0]
            if len(below_threshold) > 0:
                return float(below_threshold[0])
            
            # If never drops below threshold, estimate from exponential fit
            if len(autocorr) > 3:
                x = np.arange(len(autocorr))
                try:
                    # Fit exponential decay: y = exp(-x/xi)
                    log_autocorr = np.log(np.maximum(autocorr, 1e-10))
                    slope, _ = np.polyfit(x, log_autocorr, 1)
                    correlation_length = -1.0 / slope if slope < 0 else None
                    return correlation_length
                except:
                    return None
            
            return None
        except:
            return None
    
    def _fit_exponential_decay(self, data: np.ndarray) -> Optional[Dict[str, float]]:
        """Fit exponential decay function to data."""
        try:
            if len(data) < 3:
                return None
            
            x = np.arange(len(data))
            
            # Remove zeros and negative values for log fit
            valid_mask = data > 1e-10
            if np.sum(valid_mask) < 3:
                return None
            
            x_valid = x[valid_mask]
            data_valid = data[valid_mask]
            
            # Fit exponential: y = A * exp(-x/tau)
            # Take log: ln(y) = ln(A) - x/tau
            log_data = np.log(data_valid)
            
            # Linear fit to log data
            coeffs = np.polyfit(x_valid, log_data, 1)
            slope, intercept = coeffs
            
            # Extract parameters
            tau = -1.0 / slope if slope < 0 else np.inf
            A = np.exp(intercept)
            
            # Compute R-squared
            log_data_pred = slope * x_valid + intercept
            ss_res = np.sum((log_data - log_data_pred) ** 2)
            ss_tot = np.sum((log_data - np.mean(log_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            return {
                'decay_constant': float(tau),
                'amplitude': float(A),
                'correlation_length': float(tau),
                'r_squared': float(r_squared),
                'fit_quality': 'good' if r_squared > 0.8 else 'poor'
            }
            
        except Exception as e:
            return None
    
    def _compute_beta_exponent(self, temperatures: np.ndarray, order_parameter: np.ndarray, 
                              tc: float) -> Dict[str, Any]:
        """Compute beta critical exponent from order parameter scaling."""
        # Focus on temperatures below Tc for beta exponent
        below_tc_mask = temperatures < tc
        if np.sum(below_tc_mask) < 5:
            raise CriticalExponentError("Insufficient data points below critical temperature for beta computation")
        
        t_below = temperatures[below_tc_mask]
        m_below = order_parameter[below_tc_mask]
        
        # Compute reduced temperature
        t_reduced = (tc - t_below) / tc
        
        # Filter out very small values to avoid log issues
        valid_mask = (t_reduced > 1e-6) & (m_below > 1e-6)
        if np.sum(valid_mask) < 3:
            raise CriticalExponentError("Insufficient valid data points for beta computation")
        
        t_red_valid = t_reduced[valid_mask]
        m_valid = m_below[valid_mask]
        
        # Power law fit: m ~ t^beta
        def power_law(t, beta, a):
            return a * np.power(t, beta)
        
        try:
            # Fit in log space for better numerical stability
            log_t = np.log(t_red_valid)
            log_m = np.log(m_valid)
            
            # Linear fit: log(m) = beta * log(t) + log(a)
            coeffs = np.polyfit(log_t, log_m, 1)
            beta_computed = coeffs[0]
            
            # Compute R-squared
            log_m_pred = np.polyval(coeffs, log_t)
            ss_res = np.sum((log_m - log_m_pred) ** 2)
            ss_tot = np.sum((log_m - np.mean(log_m)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Bootstrap confidence interval
            confidence_interval = self._bootstrap_exponent_confidence(
                log_t, log_m, n_bootstrap=1000
            )
            
            # Theoretical value for 2D Ising model
            beta_theoretical = 1.0 / 8.0  # Exact Onsager result
            
            return {
                'exponent': float(beta_computed),
                'theoretical': beta_theoretical,
                'confidence_interval': confidence_interval,
                'r_squared': float(r_squared),
                'n_points': len(t_red_valid)
            }
            
        except Exception as e:
            raise CriticalExponentError(f"Failed to compute beta exponent: {str(e)}") from e
    
    def _compute_gamma_exponent(self, temperatures: np.ndarray, order_parameter: np.ndarray,
                               tc: float) -> Optional[Dict[str, Any]]:
        """Compute gamma critical exponent from susceptibility scaling."""
        try:
            # Compute susceptibility as variance of order parameter
            unique_temps = np.unique(temperatures)
            susceptibilities = []
            valid_temps = []
            
            for temp in unique_temps:
                temp_mask = temperatures == temp
                if np.sum(temp_mask) > 1:
                    temp_order_param = order_parameter[temp_mask]
                    # Susceptibility proportional to variance
                    chi = np.var(temp_order_param)
                    if chi > 1e-10:  # Avoid numerical issues
                        susceptibilities.append(chi)
                        valid_temps.append(temp)
            
            if len(valid_temps) < 5:
                self.logger.warning("Insufficient temperature points for gamma computation")
                return None
            
            susceptibilities = np.array(susceptibilities)
            valid_temps = np.array(valid_temps)
            
            # Focus on temperatures above Tc for gamma exponent
            above_tc_mask = valid_temps > tc
            if np.sum(above_tc_mask) < 3:
                self.logger.warning("Insufficient data points above critical temperature for gamma computation")
                return None
            
            t_above = valid_temps[above_tc_mask]
            chi_above = susceptibilities[above_tc_mask]
            
            # Compute reduced temperature
            t_reduced = (t_above - tc) / tc
            
            # Filter valid points
            valid_mask = (t_reduced > 1e-6) & (chi_above > 1e-10)
            if np.sum(valid_mask) < 3:
                return None
            
            t_red_valid = t_reduced[valid_mask]
            chi_valid = chi_above[valid_mask]
            
            # Power law fit: chi ~ t^(-gamma)
            log_t = np.log(t_red_valid)
            log_chi = np.log(chi_valid)
            
            # Linear fit: log(chi) = -gamma * log(t) + log(a)
            coeffs = np.polyfit(log_t, log_chi, 1)
            gamma_computed = -coeffs[0]  # Negative because chi ~ t^(-gamma)
            
            # Compute R-squared
            log_chi_pred = np.polyval(coeffs, log_t)
            ss_res = np.sum((log_chi - log_chi_pred) ** 2)
            ss_tot = np.sum((log_chi - np.mean(log_chi)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Bootstrap confidence interval
            confidence_interval = self._bootstrap_exponent_confidence(
                log_t, log_chi, n_bootstrap=1000, negative_exponent=True
            )
            
            # Theoretical value for 2D Ising model
            gamma_theoretical = 7.0 / 4.0  # Exact Onsager result
            
            return {
                'exponent': float(gamma_computed),
                'theoretical': gamma_theoretical,
                'confidence_interval': confidence_interval,
                'r_squared': float(r_squared),
                'n_points': len(t_red_valid)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to compute gamma exponent: {str(e)}")
            return None
    
    def _compute_nu_exponent(self, temperatures: np.ndarray, order_parameter: np.ndarray,
                            tc: float) -> Optional[Dict[str, Any]]:
        """Compute nu critical exponent (correlation length) - approximated from order parameter."""
        try:
            # This is an approximation since we don't have direct correlation length data
            # We use the fact that correlation length diverges as xi ~ |t|^(-nu)
            # and relate it to order parameter fluctuations
            
            unique_temps = np.unique(temperatures)
            correlation_proxies = []
            valid_temps = []
            
            for temp in unique_temps:
                temp_mask = temperatures == temp
                if np.sum(temp_mask) > 2:
                    temp_order_param = order_parameter[temp_mask]
                    # Use standard deviation as proxy for correlation length
                    corr_proxy = np.std(temp_order_param)
                    if corr_proxy > 1e-10:
                        correlation_proxies.append(corr_proxy)
                        valid_temps.append(temp)
            
            if len(valid_temps) < 5:
                self.logger.warning("Insufficient temperature points for nu computation")
                return None
            
            correlation_proxies = np.array(correlation_proxies)
            valid_temps = np.array(valid_temps)
            
            # Focus on temperatures near Tc
            near_tc_mask = np.abs(valid_temps - tc) < 0.5
            if np.sum(near_tc_mask) < 3:
                return None
            
            t_near = valid_temps[near_tc_mask]
            corr_near = correlation_proxies[near_tc_mask]
            
            # Compute reduced temperature
            t_reduced = np.abs(t_near - tc) / tc
            
            # Filter valid points
            valid_mask = (t_reduced > 1e-6) & (corr_near > 1e-10)
            if np.sum(valid_mask) < 3:
                return None
            
            t_red_valid = t_reduced[valid_mask]
            corr_valid = corr_near[valid_mask]
            
            # Power law fit: corr ~ t^(-nu)
            log_t = np.log(t_red_valid)
            log_corr = np.log(corr_valid)
            
            # Linear fit: log(corr) = -nu * log(t) + log(a)
            coeffs = np.polyfit(log_t, log_corr, 1)
            nu_computed = -coeffs[0]  # Negative because corr ~ t^(-nu)
            
            # Compute R-squared
            log_corr_pred = np.polyval(coeffs, log_t)
            ss_res = np.sum((log_corr - log_corr_pred) ** 2)
            ss_tot = np.sum((log_corr - np.mean(log_corr)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Bootstrap confidence interval
            confidence_interval = self._bootstrap_exponent_confidence(
                log_t, log_corr, n_bootstrap=1000, negative_exponent=True
            )
            
            # Theoretical value for 2D Ising model
            nu_theoretical = 1.0  # Exact result
            
            return {
                'exponent': float(nu_computed),
                'theoretical': nu_theoretical,
                'confidence_interval': confidence_interval,
                'r_squared': float(r_squared),
                'n_points': len(t_red_valid)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to compute nu exponent: {str(e)}")
            return None
    
    def _bootstrap_exponent_confidence(self, log_x: np.ndarray, log_y: np.ndarray,
                                     n_bootstrap: int = 1000, 
                                     negative_exponent: bool = False) -> Tuple[float, float]:
        """Compute bootstrap confidence interval for power law exponent."""
        exponents = []
        n_points = len(log_x)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_points, n_points, replace=True)
            x_boot = log_x[indices]
            y_boot = log_y[indices]
            
            try:
                # Fit linear model
                coeffs = np.polyfit(x_boot, y_boot, 1)
                exponent = -coeffs[0] if negative_exponent else coeffs[0]
                exponents.append(exponent)
            except:
                continue
        
        if len(exponents) < 100:
            # Fallback to simple estimate
            return (0.0, 0.0)
        
        exponents = np.array(exponents)
        lower = np.percentile(exponents, 2.5)
        upper = np.percentile(exponents, 97.5)
        
        return (float(lower), float(upper))
    
    def _identify_universality_class(self, exponents: Dict[str, Optional[float]]) -> Dict[str, Any]:
        """Identify universality class based on critical exponents."""
        # Theoretical values for different universality classes
        universality_classes = {
            UniversalityClass.ISING_2D: {
                'beta': 1.0/8.0,
                'gamma': 7.0/4.0,
                'nu': 1.0,
                'alpha': 0.0
            },
            UniversalityClass.ISING_3D: {
                'beta': 0.3265,
                'gamma': 1.237,
                'nu': 0.630,
                'alpha': 0.110
            },
            UniversalityClass.XY_2D: {
                'beta': 1.0/8.0,
                'gamma': 7.0/4.0,
                'nu': 1.0,
                'alpha': 0.0
            },
            UniversalityClass.XY_3D: {
                'beta': 0.346,
                'gamma': 1.316,
                'nu': 0.672,
                'alpha': -0.007
            }
        }
        
        best_match = UniversalityClass.UNKNOWN
        best_score = float('inf')
        
        for uc_class, theoretical in universality_classes.items():
            score = 0.0
            n_compared = 0
            
            for exp_name, exp_value in exponents.items():
                if exp_value is not None and exp_name in theoretical:
                    theoretical_value = theoretical[exp_name]
                    relative_error = abs(exp_value - theoretical_value) / abs(theoretical_value)
                    score += relative_error
                    n_compared += 1
            
            if n_compared > 0:
                avg_score = score / n_compared
                if avg_score < best_score:
                    best_score = avg_score
                    best_match = uc_class
        
        # Consider it a match if average relative error < 20%
        is_match = best_score < 0.2
        
        return {
            'class': best_match,
            'match': is_match,
            'score': float(best_score),
            'compared_exponents': {k: v for k, v in exponents.items() if v is not None}
        }
    
    def _check_scaling_violations(self, beta_result: Dict[str, Any], 
                                gamma_result: Optional[Dict[str, Any]],
                                nu_result: Optional[Dict[str, Any]],
                                universality_result: Dict[str, Any]) -> List[str]:
        """Check for scaling law violations."""
        violations = []
        
        # Check beta exponent quality
        if beta_result['r_squared'] < 0.8:
            violations.append(f"Poor beta exponent fit quality (R² = {beta_result['r_squared']:.3f})")
        
        # Check deviation from theoretical values
        beta_deviation = abs(beta_result['exponent'] - beta_result['theoretical'])
        if beta_deviation > 0.05:  # 5% tolerance
            violations.append(f"Large beta exponent deviation from theory ({beta_deviation:.3f})")
        
        # Check gamma exponent if available
        if gamma_result:
            if gamma_result['r_squared'] < 0.7:
                violations.append(f"Poor gamma exponent fit quality (R² = {gamma_result['r_squared']:.3f})")
            
            gamma_deviation = abs(gamma_result['exponent'] - gamma_result['theoretical'])
            if gamma_deviation > 0.1:  # 10% tolerance for gamma
                violations.append(f"Large gamma exponent deviation from theory ({gamma_deviation:.3f})")
        
        # Check universality class match
        if not universality_result['match']:
            violations.append(f"No clear universality class match (best score: {universality_result['score']:.3f})")
        
        return violations
    
    def _perform_scaling_collapse(self, size_data: Dict[int, Dict[str, np.ndarray]], 
                                system_sizes: List[int]) -> Dict[str, Any]:
        """Perform finite-size scaling collapse analysis."""
        # Extract critical temperature (assume same for all sizes)
        tc_estimates = []
        for size in system_sizes:
            temps = size_data[size]['temperatures']
            order_param = size_data[size]['order_parameter']
            
            # Find temperature with maximum derivative (crude Tc estimate)
            unique_temps = np.unique(temps)
            if len(unique_temps) > 3:
                temp_means = []
                for temp in unique_temps:
                    temp_mask = temps == temp
                    temp_means.append(np.mean(order_param[temp_mask]))
                
                temp_means = np.array(temp_means)
                derivatives = np.gradient(temp_means, unique_temps)
                max_deriv_idx = np.argmax(np.abs(derivatives))
                tc_estimates.append(unique_temps[max_deriv_idx])
        
        tc_avg = np.mean(tc_estimates) if tc_estimates else self.theoretical_tc
        
        # Attempt scaling collapse
        # For 2D Ising: M(T,L) = L^(-beta/nu) * f((T-Tc)*L^(1/nu))
        beta_nu = 1.0/8.0 / 1.0  # beta/nu for 2D Ising
        nu_inv = 1.0  # 1/nu for 2D Ising
        
        collapsed_data = {}
        scaling_quality_scores = []
        
        for size in system_sizes:
            temps = size_data[size]['temperatures']
            order_param = size_data[size]['order_parameter']
            
            # Compute scaling variables
            t_reduced = (temps - tc_avg) * (size ** nu_inv)
            m_scaled = order_param * (size ** beta_nu)
            
            collapsed_data[size] = {
                'x': t_reduced,
                'y': m_scaled,
                'original_temps': temps,
                'original_order_param': order_param
            }
        
        # Assess collapse quality by checking overlap in scaled coordinates
        quality_score = self._assess_collapse_quality(collapsed_data, system_sizes)
        
        return {
            'quality': quality_score,
            'parameters': {
                'beta_over_nu': beta_nu,
                'nu_inverse': nu_inv,
                'critical_temperature': tc_avg
            },
            'collapsed_data': collapsed_data
        }
    
    def _assess_collapse_quality(self, collapsed_data: Dict[int, Dict[str, np.ndarray]], 
                               system_sizes: List[int]) -> float:
        """Assess quality of scaling collapse."""
        if len(system_sizes) < 2:
            return 0.0
        
        # Create common x-axis grid
        all_x = []
        for size in system_sizes:
            all_x.extend(collapsed_data[size]['x'])
        
        x_min, x_max = np.percentile(all_x, [10, 90])  # Use central 80% range
        x_grid = np.linspace(x_min, x_max, 50)
        
        # Interpolate each size's data onto common grid
        interpolated_curves = []
        for size in system_sizes:
            x_data = collapsed_data[size]['x']
            y_data = collapsed_data[size]['y']
            
            # Sort by x for interpolation
            sort_idx = np.argsort(x_data)
            x_sorted = x_data[sort_idx]
            y_sorted = y_data[sort_idx]
            
            # Remove duplicates and interpolate
            try:
                y_interp = np.interp(x_grid, x_sorted, y_sorted)
                interpolated_curves.append(y_interp)
            except:
                continue
        
        if len(interpolated_curves) < 2:
            return 0.0
        
        # Compute pairwise correlations between curves
        correlations = []
        for i in range(len(interpolated_curves)):
            for j in range(i+1, len(interpolated_curves)):
                try:
                    corr = np.corrcoef(interpolated_curves[i], interpolated_curves[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                except:
                    continue
        
        # Quality score is average correlation
        quality = np.mean(correlations) if correlations else 0.0
        return float(quality)
    
    def _compute_correlation_length_exponent(self, size_data: Dict[int, Dict[str, np.ndarray]], 
                                           system_sizes: List[int]) -> Dict[str, Any]:
        """Compute correlation length exponent from finite-size scaling."""
        # Use the fact that correlation length scales as xi ~ L at criticality
        # and xi ~ |t|^(-nu) away from criticality
        
        # For each size, find the peak in susceptibility (proxy for correlation length)
        size_susceptibilities = {}
        
        for size in system_sizes:
            temps = size_data[size]['temperatures']
            order_param = size_data[size]['order_parameter']
            
            unique_temps = np.unique(temps)
            susceptibilities = []
            
            for temp in unique_temps:
                temp_mask = temps == temp
                if np.sum(temp_mask) > 1:
                    temp_order_param = order_param[temp_mask]
                    chi = np.var(temp_order_param)
                    susceptibilities.append(chi)
                else:
                    susceptibilities.append(0.0)
            
            if len(susceptibilities) > 0:
                max_chi = np.max(susceptibilities)
                size_susceptibilities[size] = max_chi
        
        if len(size_susceptibilities) < 2:
            return {
                'exponent': 1.0,  # Default theoretical value
                'confidence_interval': (0.8, 1.2),
                'r_squared': 0.0
            }
        
        # Fit chi_max ~ L^(gamma/nu) for 2D Ising
        sizes = np.array(list(size_susceptibilities.keys()))
        chis = np.array(list(size_susceptibilities.values()))
        
        # Filter out zero susceptibilities
        valid_mask = chis > 1e-10
        if np.sum(valid_mask) < 2:
            return {
                'exponent': 1.0,
                'confidence_interval': (0.8, 1.2),
                'r_squared': 0.0
            }
        
        sizes_valid = sizes[valid_mask]
        chis_valid = chis[valid_mask]
        
        # Power law fit in log space
        log_sizes = np.log(sizes_valid)
        log_chis = np.log(chis_valid)
        
        try:
            coeffs = np.polyfit(log_sizes, log_chis, 1)
            gamma_over_nu = coeffs[0]
            
            # For 2D Ising: gamma/nu = 7/4 / 1 = 1.75
            # So nu = gamma / (gamma/nu) = 1.75 / gamma_over_nu
            theoretical_gamma_over_nu = 7.0/4.0
            nu_computed = theoretical_gamma_over_nu / gamma_over_nu if gamma_over_nu != 0 else 1.0
            
            # Compute R-squared
            log_chis_pred = np.polyval(coeffs, log_sizes)
            ss_res = np.sum((log_chis - log_chis_pred) ** 2)
            ss_tot = np.sum((log_chis - np.mean(log_chis)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Simple confidence interval (could be improved with bootstrap)
            nu_error = 0.1 * abs(nu_computed)  # 10% error estimate
            confidence_interval = (nu_computed - nu_error, nu_computed + nu_error)
            
            return {
                'exponent': float(nu_computed),
                'confidence_interval': confidence_interval,
                'r_squared': float(r_squared)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to compute correlation length exponent: {str(e)}")
            return {
                'exponent': 1.0,
                'confidence_interval': (0.8, 1.2),
                'r_squared': 0.0
            }
    
    def _analyze_finite_size_corrections(self, size_data: Dict[int, Dict[str, np.ndarray]], 
                                       system_sizes: List[int]) -> Dict[str, float]:
        """Analyze finite-size corrections to scaling."""
        corrections = {}
        
        # Analyze how critical temperature shifts with system size
        tc_shifts = []
        for size in system_sizes:
            temps = size_data[size]['temperatures']
            order_param = size_data[size]['order_parameter']
            
            # Find apparent critical temperature for this size
            unique_temps = np.unique(temps)
            if len(unique_temps) > 3:
                temp_means = []
                for temp in unique_temps:
                    temp_mask = temps == temp
                    temp_means.append(np.mean(order_param[temp_mask]))
                
                temp_means = np.array(temp_means)
                derivatives = np.gradient(temp_means, unique_temps)
                max_deriv_idx = np.argmax(np.abs(derivatives))
                tc_apparent = unique_temps[max_deriv_idx]
                
                # Finite-size correction: Tc(L) = Tc(inf) + a/L
                tc_shifts.append((size, tc_apparent))
        
        if len(tc_shifts) >= 2:
            sizes = np.array([shift[0] for shift in tc_shifts])
            tcs = np.array([shift[1] for shift in tc_shifts])
            
            # Fit Tc(L) = Tc_inf + a/L
            try:
                inv_sizes = 1.0 / sizes
                coeffs = np.polyfit(inv_sizes, tcs, 1)
                tc_infinite = coeffs[1]  # Intercept
                correction_amplitude = coeffs[0]  # Slope
                
                corrections['tc_infinite_extrapolation'] = float(tc_infinite)
                corrections['tc_correction_amplitude'] = float(correction_amplitude)
                
                # R-squared for the fit
                tcs_pred = np.polyval(coeffs, inv_sizes)
                ss_res = np.sum((tcs - tcs_pred) ** 2)
                ss_tot = np.sum((tcs - np.mean(tcs)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                corrections['tc_extrapolation_quality'] = float(r_squared)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze Tc finite-size corrections: {str(e)}")
        
        return corrections
    
    def _generate_scaling_plots_data(self, size_data: Dict[int, Dict[str, np.ndarray]], 
                                   system_sizes: List[int],
                                   scaling_result: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Generate data for scaling collapse plots."""
        plots_data = {}
        
        # Extract collapsed data
        collapsed_data = scaling_result['collapsed_data']
        
        # Combine all collapsed data points
        all_x_collapsed = []
        all_y_collapsed = []
        all_sizes = []
        
        for size in system_sizes:
            if size in collapsed_data:
                x_data = collapsed_data[size]['x']
                y_data = collapsed_data[size]['y']
                
                all_x_collapsed.extend(x_data)
                all_y_collapsed.extend(y_data)
                all_sizes.extend([size] * len(x_data))
        
        plots_data['collapsed_x'] = np.array(all_x_collapsed)
        plots_data['collapsed_y'] = np.array(all_y_collapsed)
        plots_data['sizes'] = np.array(all_sizes)
        
        # Original data for comparison
        all_temps = []
        all_order_params = []
        all_sizes_orig = []
        
        for size in system_sizes:
            temps = size_data[size]['temperatures']
            order_param = size_data[size]['order_parameter']
            
            all_temps.extend(temps)
            all_order_params.extend(order_param)
            all_sizes_orig.extend([size] * len(temps))
        
        plots_data['original_temperatures'] = np.array(all_temps)
        plots_data['original_order_parameters'] = np.array(all_order_params)
        plots_data['original_sizes'] = np.array(all_sizes_orig)
        
        return plots_data
    
    def _check_finite_size_scaling_violations(self, scaling_result: Dict[str, Any],
                                            nu_result: Dict[str, Any],
                                            finite_size_corrections: Dict[str, float]) -> List[str]:
        """Check for finite-size scaling violations."""
        violations = []
        
        # Check scaling collapse quality
        if scaling_result['quality'] < 0.7:
            violations.append(f"Poor scaling collapse quality ({scaling_result['quality']:.3f})")
        
        # Check correlation length exponent
        if nu_result['r_squared'] < 0.6:
            violations.append(f"Poor correlation length exponent fit (R² = {nu_result['r_squared']:.3f})")
        
        # Check if nu is reasonable (should be positive and not too large)
        nu_value = nu_result['exponent']
        if nu_value <= 0:
            violations.append(f"Unphysical correlation length exponent (ν = {nu_value:.3f})")
        elif nu_value > 3.0:
            violations.append(f"Unusually large correlation length exponent (ν = {nu_value:.3f})")
        
        # Check finite-size corrections if available
        if 'tc_extrapolation_quality' in finite_size_corrections:
            if finite_size_corrections['tc_extrapolation_quality'] < 0.5:
                violations.append("Poor finite-size correction extrapolation")
        
        return violations
    
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