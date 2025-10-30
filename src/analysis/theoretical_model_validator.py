"""
Theoretical Model Validator

This module provides comprehensive validation against established theoretical physics models
including Ising, XY, and Heisenberg models. It computes theoretical predictions and compares
them with computational results to ensure physics consistency.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
from scipy import stats, special
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import warnings

from .latent_analysis import LatentRepresentation
from .phase_detection import PhaseDetectionResult
from .order_parameter_discovery import OrderParameterCandidate
from .enhanced_validation_types import (
    TheoreticalPredictions, IsingModelValidation, XYModelValidation, 
    HeisenbergModelValidation, TheoreticalModelError, UniversalityClass,
    BaseTheoreticalModel, PhysicsViolation, ViolationSeverity
)
from ..utils.logging_utils import get_logger, LoggingContext


class IsingModel(BaseTheoreticalModel):
    """
    Theoretical Ising model implementation with exact solutions and predictions.
    """
    
    def __init__(self, dimensionality: int = 2):
        super().__init__("Ising", dimensionality)
        self.logger = get_logger(__name__)
        
        # Theoretical critical exponents for different dimensions
        self._critical_exponents = {
            2: {  # 2D Ising (exact Onsager solution)
                'beta': 1/8,
                'gamma': 7/4,
                'nu': 1.0,
                'alpha': 0.0,  # logarithmic divergence
                'delta': 15.0,
                'eta': 1/4
            },
            3: {  # 3D Ising (experimental/numerical values)
                'beta': 0.3265,
                'gamma': 1.2372,
                'nu': 0.6301,
                'alpha': 0.1096,
                'delta': 4.789,
                'eta': 0.0364
            }
        }
        
        # Critical temperatures (in units of J/k_B)
        self._critical_temperatures = {
            2: 2.269185314213022,  # Exact Onsager solution: 2/ln(1+sqrt(2))
            3: 4.51152  # Numerical estimate for 3D
        }
    
    def compute_critical_exponents(self) -> Dict[str, float]:
        """Compute theoretical critical exponents for the Ising model."""
        if self.dimensionality not in self._critical_exponents:
            raise TheoreticalModelError(f"Critical exponents not available for {self.dimensionality}D Ising model")
        
        return self._critical_exponents[self.dimensionality].copy()
    
    def compute_critical_temperature(self, system_parameters: Dict[str, Any]) -> float:
        """
        Compute theoretical critical temperature for the Ising model.
        
        Args:
            system_parameters: Dictionary containing 'J' (coupling strength) and optionally 'lattice_size'
        """
        if self.dimensionality not in self._critical_temperatures:
            raise TheoreticalModelError(f"Critical temperature not available for {self.dimensionality}D Ising model")
        
        # Base critical temperature in units of J/k_B
        tc_base = self._critical_temperatures[self.dimensionality]
        
        # Scale by coupling strength if provided
        J = system_parameters.get('J', 1.0)
        tc = tc_base * J
        
        # Apply finite-size corrections if lattice size is provided
        lattice_size = system_parameters.get('lattice_size')
        if lattice_size is not None and self.dimensionality == 2:
            # Finite-size correction for 2D Ising model
            tc_correction = self._compute_finite_size_correction(lattice_size, tc)
            tc += tc_correction
        
        return tc
    
    def compute_order_parameter_behavior(self, temperatures: np.ndarray, 
                                       system_parameters: Dict[str, Any] = None) -> np.ndarray:
        """
        Compute theoretical order parameter (magnetization) behavior.
        
        Args:
            temperatures: Array of temperatures
            system_parameters: Optional system parameters
        """
        if system_parameters is None:
            system_parameters = {}
        
        tc = self.compute_critical_temperature(system_parameters)
        exponents = self.compute_critical_exponents()
        beta = exponents['beta']
        
        # Compute magnetization using power law below Tc
        magnetization = np.zeros_like(temperatures)
        below_tc = temperatures < tc
        
        if np.any(below_tc):
            reduced_temp = (tc - temperatures[below_tc]) / tc
            magnetization[below_tc] = np.power(reduced_temp, beta)
        
        return magnetization
    
    def compute_susceptibility(self, temperatures: np.ndarray,
                             system_parameters: Dict[str, Any] = None) -> np.ndarray:
        """Compute theoretical magnetic susceptibility."""
        if system_parameters is None:
            system_parameters = {}
        
        tc = self.compute_critical_temperature(system_parameters)
        exponents = self.compute_critical_exponents()
        gamma = exponents['gamma']
        
        susceptibility = np.zeros_like(temperatures)
        
        # Above Tc
        above_tc = temperatures > tc
        if np.any(above_tc):
            reduced_temp = (temperatures[above_tc] - tc) / tc
            susceptibility[above_tc] = np.power(reduced_temp, -gamma)
        
        # Below Tc
        below_tc = temperatures < tc
        if np.any(below_tc):
            reduced_temp = (tc - temperatures[below_tc]) / tc
            susceptibility[below_tc] = np.power(reduced_temp, -gamma)
        
        return susceptibility
    
    def compute_specific_heat(self, temperatures: np.ndarray,
                            system_parameters: Dict[str, Any] = None) -> np.ndarray:
        """Compute theoretical specific heat."""
        if system_parameters is None:
            system_parameters = {}
        
        tc = self.compute_critical_temperature(system_parameters)
        exponents = self.compute_critical_exponents()
        alpha = exponents['alpha']
        
        specific_heat = np.ones_like(temperatures)  # Background value
        
        if self.dimensionality == 2:
            # 2D Ising has logarithmic divergence (alpha = 0)
            near_tc = np.abs(temperatures - tc) < 0.1 * tc
            if np.any(near_tc):
                reduced_temp = np.abs(temperatures[near_tc] - tc) / tc
                # Logarithmic divergence
                specific_heat[near_tc] = -np.log(reduced_temp + 1e-10)
        else:
            # Power law divergence for other dimensions
            near_tc = np.abs(temperatures - tc) < 0.1 * tc
            if np.any(near_tc):
                reduced_temp = np.abs(temperatures[near_tc] - tc) / tc
                specific_heat[near_tc] = np.power(reduced_temp + 1e-10, -alpha)
        
        return specific_heat
    
    def _compute_finite_size_correction(self, lattice_size: int, tc_infinite: float) -> float:
        """Compute finite-size correction to critical temperature."""
        if self.dimensionality == 2:
            # Finite-size scaling correction for 2D Ising
            nu = self._critical_exponents[2]['nu']
            correction = -1.0 / (nu * np.log(lattice_size))
            return correction * tc_infinite / lattice_size
        else:
            # Simple 1/L correction for other dimensions
            return -tc_infinite / (2 * lattice_size)


class XYModel(BaseTheoreticalModel):
    """
    Theoretical XY model implementation with Kosterlitz-Thouless transition.
    """
    
    def __init__(self, dimensionality: int = 2):
        super().__init__("XY", dimensionality)
        self.logger = get_logger(__name__)
        
        # Critical exponents for XY model
        self._critical_exponents = {
            2: {  # 2D XY (Kosterlitz-Thouless)
                'beta': None,  # No conventional order parameter
                'gamma': None,  # No conventional susceptibility divergence
                'nu': None,    # No conventional correlation length divergence
                'alpha': None, # No specific heat divergence
                'eta': 0.25,   # Universal jump in superfluid density
                'kt_transition': True
            },
            3: {  # 3D XY
                'beta': 0.3485,
                'gamma': 1.3177,
                'nu': 0.6717,
                'alpha': -0.0151,
                'delta': 4.780,
                'eta': 0.0380,
                'kt_transition': False
            }
        }
        
        # Critical temperatures (approximate values)
        self._critical_temperatures = {
            2: 0.8935,  # 2D XY Kosterlitz-Thouless temperature
            3: 2.202    # 3D XY critical temperature
        }
    
    def compute_critical_exponents(self) -> Dict[str, float]:
        """Compute theoretical critical exponents for the XY model."""
        if self.dimensionality not in self._critical_exponents:
            raise TheoreticalModelError(f"Critical exponents not available for {self.dimensionality}D XY model")
        
        exponents = self._critical_exponents[self.dimensionality].copy()
        # Remove non-numeric entries
        return {k: v for k, v in exponents.items() if isinstance(v, (int, float))}
    
    def compute_critical_temperature(self, system_parameters: Dict[str, Any]) -> float:
        """Compute theoretical critical temperature for the XY model."""
        if self.dimensionality not in self._critical_temperatures:
            raise TheoreticalModelError(f"Critical temperature not available for {self.dimensionality}D XY model")
        
        tc_base = self._critical_temperatures[self.dimensionality]
        J = system_parameters.get('J', 1.0)
        return tc_base * J
    
    def compute_order_parameter_behavior(self, temperatures: np.ndarray,
                                       system_parameters: Dict[str, Any] = None) -> np.ndarray:
        """Compute theoretical order parameter behavior for XY model."""
        if system_parameters is None:
            system_parameters = {}
        
        tc = self.compute_critical_temperature(system_parameters)
        
        if self.dimensionality == 2:
            # 2D XY model has no conventional order parameter
            # Return helicity modulus or superfluid density instead
            order_parameter = np.zeros_like(temperatures)
            below_tc = temperatures < tc
            if np.any(below_tc):
                # Approximate behavior of helicity modulus
                reduced_temp = temperatures[below_tc] / tc
                order_parameter[below_tc] = 1.0 - reduced_temp
        else:
            # 3D XY model has conventional order parameter
            exponents = self.compute_critical_exponents()
            beta = exponents['beta']
            
            order_parameter = np.zeros_like(temperatures)
            below_tc = temperatures < tc
            if np.any(below_tc):
                reduced_temp = (tc - temperatures[below_tc]) / tc
                order_parameter[below_tc] = np.power(reduced_temp, beta)
        
        return order_parameter
    
    def compute_helicity_modulus(self, temperatures: np.ndarray,
                               system_parameters: Dict[str, Any] = None) -> np.ndarray:
        """Compute helicity modulus for XY model (especially relevant for 2D)."""
        if system_parameters is None:
            system_parameters = {}
        
        tc = self.compute_critical_temperature(system_parameters)
        helicity_modulus = np.zeros_like(temperatures)
        
        if self.dimensionality == 2:
            # 2D XY: Universal jump at KT transition
            below_tc = temperatures < tc
            helicity_modulus[below_tc] = 2.0 * temperatures[below_tc] / np.pi  # Universal value
            
            # Above Tc, exponential decay
            above_tc = temperatures >= tc
            if np.any(above_tc):
                # Exponential decay with correlation length
                reduced_temp = (temperatures[above_tc] - tc) / tc
                helicity_modulus[above_tc] = np.exp(-reduced_temp)
        else:
            # 3D XY: Power law behavior
            exponents = self.compute_critical_exponents()
            if 'nu' in exponents:
                nu = exponents['nu']
                near_tc = np.abs(temperatures - tc) < 0.1 * tc
                if np.any(near_tc):
                    reduced_temp = np.abs(temperatures[near_tc] - tc) / tc
                    helicity_modulus[near_tc] = np.power(reduced_temp + 1e-10, -1/nu)
        
        return helicity_modulus
    
    def compute_vortex_density(self, temperatures: np.ndarray,
                             system_parameters: Dict[str, Any] = None) -> np.ndarray:
        """Compute vortex density for 2D XY model."""
        if self.dimensionality != 2:
            raise TheoreticalModelError("Vortex density only applicable to 2D XY model")
        
        if system_parameters is None:
            system_parameters = {}
        
        tc = self.compute_critical_temperature(system_parameters)
        vortex_density = np.zeros_like(temperatures)
        
        # Below Tc: exponentially suppressed vortices
        below_tc = temperatures < tc
        if np.any(below_tc):
            # Exponential suppression
            energy_scale = system_parameters.get('J', 1.0)
            vortex_density[below_tc] = np.exp(-2 * np.pi * energy_scale / temperatures[below_tc])
        
        # Above Tc: proliferation of vortices
        above_tc = temperatures >= tc
        if np.any(above_tc):
            # Power law increase
            reduced_temp = temperatures[above_tc] / tc
            vortex_density[above_tc] = np.power(reduced_temp, 2)
        
        return vortex_density
    
    def is_kosterlitz_thouless_transition(self) -> bool:
        """Check if this is a Kosterlitz-Thouless transition."""
        return (self.dimensionality == 2 and 
                self._critical_exponents[2].get('kt_transition', False))


class HeisenbergModel(BaseTheoreticalModel):
    """
    Theoretical Heisenberg model implementation with spin wave analysis.
    """
    
    def __init__(self, dimensionality: int = 3):
        super().__init__("Heisenberg", dimensionality)
        self.logger = get_logger(__name__)
        
        # Critical exponents for Heisenberg model
        self._critical_exponents = {
            3: {  # 3D Heisenberg
                'beta': 0.3689,
                'gamma': 1.3960,
                'nu': 0.7112,
                'alpha': -0.1336,
                'delta': 4.783,
                'eta': 0.0375
            }
        }
        
        # Critical temperatures
        self._critical_temperatures = {
            3: 1.443  # 3D Heisenberg critical temperature
        }
    
    def compute_critical_exponents(self) -> Dict[str, float]:
        """Compute theoretical critical exponents for the Heisenberg model."""
        if self.dimensionality not in self._critical_exponents:
            raise TheoreticalModelError(f"Critical exponents not available for {self.dimensionality}D Heisenberg model")
        
        return self._critical_exponents[self.dimensionality].copy()
    
    def compute_critical_temperature(self, system_parameters: Dict[str, Any]) -> float:
        """Compute theoretical critical temperature for the Heisenberg model."""
        if self.dimensionality not in self._critical_temperatures:
            raise TheoreticalModelError(f"Critical temperature not available for {self.dimensionality}D Heisenberg model")
        
        tc_base = self._critical_temperatures[self.dimensionality]
        J = system_parameters.get('J', 1.0)
        return tc_base * J
    
    def compute_order_parameter_behavior(self, temperatures: np.ndarray,
                                       system_parameters: Dict[str, Any] = None) -> np.ndarray:
        """Compute theoretical order parameter behavior for Heisenberg model."""
        if system_parameters is None:
            system_parameters = {}
        
        tc = self.compute_critical_temperature(system_parameters)
        exponents = self.compute_critical_exponents()
        beta = exponents['beta']
        
        order_parameter = np.zeros_like(temperatures)
        below_tc = temperatures < tc
        
        if np.any(below_tc):
            reduced_temp = (tc - temperatures[below_tc]) / tc
            order_parameter[below_tc] = np.power(reduced_temp, beta)
        
        return order_parameter
    
    def compute_spin_wave_contribution(self, temperatures: np.ndarray,
                                     system_parameters: Dict[str, Any] = None) -> np.ndarray:
        """Compute spin wave contribution to magnetization reduction."""
        if system_parameters is None:
            system_parameters = {}
        
        # Spin wave theory prediction for magnetization reduction
        # M(T) = M(0) * (1 - (T/T_sw)^(3/2)) for 3D
        
        tc = self.compute_critical_temperature(system_parameters)
        spin_wave_scale = system_parameters.get('spin_wave_scale', tc)
        
        magnetization_reduction = np.zeros_like(temperatures)
        
        # Only valid at low temperatures where spin wave theory applies
        low_temp_mask = temperatures < 0.5 * tc
        
        if np.any(low_temp_mask):
            low_temps = temperatures[low_temp_mask]
            if self.dimensionality == 3:
                # 3D spin wave theory: T^(3/2) dependence
                reduction = np.power(low_temps / spin_wave_scale, 1.5)
            elif self.dimensionality == 2:
                # 2D: logarithmic reduction (Mermin-Wagner theorem)
                reduction = np.log(1 + low_temps / spin_wave_scale)
            else:
                reduction = np.power(low_temps / spin_wave_scale, self.dimensionality / 2)
            
            magnetization_reduction[low_temp_mask] = reduction
        
        return magnetization_reduction
    
    def compute_magnon_dispersion(self, k_values: np.ndarray,
                                system_parameters: Dict[str, Any] = None) -> np.ndarray:
        """Compute magnon dispersion relation."""
        if system_parameters is None:
            system_parameters = {}
        
        # Simple magnon dispersion: E(k) = J * S * (1 - cos(k*a))
        J = system_parameters.get('J', 1.0)
        S = system_parameters.get('spin', 0.5)
        a = system_parameters.get('lattice_constant', 1.0)
        
        # For small k: E(k) ≈ J * S * (k*a)^2 / 2
        dispersion = J * S * (1 - np.cos(k_values * a))
        
        return dispersion
    
    def analyze_continuous_symmetry_breaking(self, temperatures: np.ndarray,
                                           order_parameter: np.ndarray) -> Dict[str, Any]:
        """Analyze continuous symmetry breaking characteristics."""
        analysis = {}
        
        # Check for smooth, continuous onset (characteristic of O(3) symmetry breaking)
        if len(temperatures) > 10 and len(order_parameter) > 10:
            # Sort by temperature
            sorted_indices = np.argsort(temperatures)
            sorted_temps = temperatures[sorted_indices]
            sorted_order = order_parameter[sorted_indices]
            
            # Measure smoothness of transition
            first_derivative = np.gradient(sorted_order, sorted_temps)
            second_derivative = np.gradient(first_derivative, sorted_temps)
            
            # Continuous transitions should have smooth derivatives
            smoothness_score = 1.0 / (1.0 + np.std(second_derivative))
            analysis['transition_smoothness'] = smoothness_score
            
            # Check for power law behavior near transition
            # Find approximate critical temperature
            max_derivative_idx = np.argmax(np.abs(first_derivative))
            tc_estimate = sorted_temps[max_derivative_idx]
            
            # Analyze power law behavior below Tc
            below_tc = sorted_temps < tc_estimate
            if np.sum(below_tc) > 5:
                below_temps = sorted_temps[below_tc]
                below_order = sorted_order[below_tc]
                
                # Fit power law: M ~ (Tc - T)^beta
                reduced_temps = (tc_estimate - below_temps) / tc_estimate
                valid_mask = (reduced_temps > 0) & (below_order > 0)
                
                if np.sum(valid_mask) > 3:
                    try:
                        log_reduced_temps = np.log(reduced_temps[valid_mask])
                        log_order = np.log(below_order[valid_mask])
                        
                        # Linear fit in log space
                        coeffs = np.polyfit(log_reduced_temps, log_order, 1)
                        beta_estimate = coeffs[0]
                        r_squared = np.corrcoef(log_reduced_temps, log_order)[0, 1]**2
                        
                        analysis['power_law_fit'] = {
                            'beta_estimate': beta_estimate,
                            'r_squared': r_squared,
                            'tc_estimate': tc_estimate
                        }
                    except:
                        analysis['power_law_fit'] = None
        
        return analysis


class TheoreticalModelValidator:
    """
    Comprehensive theoretical model validation framework.
    
    Validates computational results against established theoretical physics models
    including Ising, XY, and Heisenberg models.
    """
    
    def __init__(self, default_dimensionality: int = 2):
        """
        Initialize theoretical model validator.
        
        Args:
            default_dimensionality: Default system dimensionality
        """
        self.default_dimensionality = default_dimensionality
        self.logger = get_logger(__name__)
        
        # Initialize theoretical models
        self.models = {
            'ising': {
                2: IsingModel(2),
                3: IsingModel(3)
            },
            'xy': {
                2: XYModel(2),
                3: XYModel(3)
            },
            'heisenberg': {
                3: HeisenbergModel(3)
            }
        }
        
        self.logger.info(f"Theoretical model validator initialized with {default_dimensionality}D default")
    
    def validate_against_ising_model(self,
                                   latent_repr: LatentRepresentation,
                                   system_size: int,
                                   dimensionality: Optional[int] = None) -> IsingModelValidation:
        """
        Validate computational results against Ising model predictions.
        
        Args:
            latent_repr: Latent space representation
            system_size: System size for finite-size corrections
            dimensionality: System dimensionality (defaults to class default)
            
        Returns:
            IsingModelValidation with comparison results
        """
        self.logger.info("Validating against Ising model")
        
        try:
            dim = dimensionality or self.default_dimensionality
            
            if dim not in self.models['ising']:
                raise TheoreticalModelError(f"Ising model not available for {dim}D")
            
            ising_model = self.models['ising'][dim]
            
            # Prepare system parameters
            system_parameters = {
                'J': 1.0,  # Assume unit coupling
                'lattice_size': system_size
            }
            
            # Compute theoretical predictions
            theoretical_predictions = self.compute_theoretical_predictions(
                'ising', system_parameters, dim
            )
            
            # Extract computational results
            computational_results = self._extract_computational_results(latent_repr)
            
            # Compute agreement metrics
            agreement_metrics = self._compute_agreement_metrics(
                computational_results, theoretical_predictions
            )
            
            # Onsager solution comparison (for 2D)
            onsager_comparison = {}
            if dim == 2:
                onsager_comparison = self._compare_with_onsager_solution(
                    computational_results, theoretical_predictions
                )
            
            # Mean field comparison
            mean_field_comparison = self._compare_with_mean_field_theory(
                computational_results, theoretical_predictions
            )
            
            validation = IsingModelValidation(
                theoretical_predictions=theoretical_predictions,
                computational_results=computational_results,
                agreement_metrics=agreement_metrics,
                onsager_solution_comparison=onsager_comparison,
                mean_field_comparison=mean_field_comparison
            )
            
            self.logger.info(f"Ising model validation completed: "
                           f"Tc agreement = {agreement_metrics.get('critical_temperature_agreement', 0):.3f}")
            
            return validation
            
        except Exception as e:
            raise TheoreticalModelError(f"Failed to validate against Ising model: {str(e)}") from e
    
    def validate_against_xy_model(self,
                                latent_repr: LatentRepresentation,
                                dimensionality: Optional[int] = None) -> XYModelValidation:
        """
        Validate computational results against XY model predictions.
        
        Args:
            latent_repr: Latent space representation
            dimensionality: System dimensionality (defaults to class default)
            
        Returns:
            XYModelValidation with comparison results
        """
        self.logger.info("Validating against XY model")
        
        try:
            dim = dimensionality or self.default_dimensionality
            
            if dim not in self.models['xy']:
                raise TheoreticalModelError(f"XY model not available for {dim}D")
            
            xy_model = self.models['xy'][dim]
            
            # Compute theoretical predictions
            theoretical_predictions = self.compute_theoretical_predictions(
                'xy', {'J': 1.0}, dim
            )
            
            # Extract computational results
            computational_results = self._extract_computational_results(latent_repr)
            
            # Compute agreement metrics
            agreement_metrics = self._compute_agreement_metrics(
                computational_results, theoretical_predictions
            )
            
            # Kosterlitz-Thouless analysis (for 2D)
            kt_analysis = {}
            if dim == 2 and xy_model.is_kosterlitz_thouless_transition():
                kt_analysis = self._analyze_kosterlitz_thouless_transition(
                    latent_repr, theoretical_predictions
                )
            
            # Vortex analysis (placeholder for future implementation)
            vortex_analysis = {}
            
            validation = XYModelValidation(
                theoretical_predictions=theoretical_predictions,
                computational_results=computational_results,
                agreement_metrics=agreement_metrics,
                kosterlitz_thouless_analysis=kt_analysis,
                vortex_analysis=vortex_analysis
            )
            
            self.logger.info(f"XY model validation completed: "
                           f"Tc agreement = {agreement_metrics.get('critical_temperature_agreement', 0):.3f}")
            
            return validation
            
        except Exception as e:
            raise TheoreticalModelError(f"Failed to validate against XY model: {str(e)}") from e
    
    def validate_against_heisenberg_model(self,
                                        latent_repr: LatentRepresentation,
                                        dimensionality: Optional[int] = None) -> HeisenbergModelValidation:
        """
        Validate computational results against Heisenberg model predictions.
        
        Args:
            latent_repr: Latent space representation
            dimensionality: System dimensionality (defaults to 3)
            
        Returns:
            HeisenbergModelValidation with comparison results
        """
        self.logger.info("Validating against Heisenberg model")
        
        try:
            dim = dimensionality or 3  # Heisenberg model typically 3D
            
            if dim not in self.models['heisenberg']:
                raise TheoreticalModelError(f"Heisenberg model not available for {dim}D")
            
            # Compute theoretical predictions
            theoretical_predictions = self.compute_theoretical_predictions(
                'heisenberg', {'J': 1.0}, dim
            )
            
            # Extract computational results
            computational_results = self._extract_computational_results(latent_repr)
            
            # Compute agreement metrics
            agreement_metrics = self._compute_agreement_metrics(
                computational_results, theoretical_predictions
            )
            
            # Spin wave analysis (placeholder for future implementation)
            spin_wave_analysis = {}
            
            # Magnon analysis (placeholder for future implementation)
            magnon_analysis = {}
            
            validation = HeisenbergModelValidation(
                theoretical_predictions=theoretical_predictions,
                computational_results=computational_results,
                agreement_metrics=agreement_metrics,
                spin_wave_analysis=spin_wave_analysis,
                magnon_analysis=magnon_analysis
            )
            
            self.logger.info(f"Heisenberg model validation completed: "
                           f"Tc agreement = {agreement_metrics.get('critical_temperature_agreement', 0):.3f}")
            
            return validation
            
        except Exception as e:
            raise TheoreticalModelError(f"Failed to validate against Heisenberg model: {str(e)}") from e
    
    def compute_theoretical_predictions(self,
                                      model_type: str,
                                      system_parameters: Dict[str, Any],
                                      dimensionality: Optional[int] = None) -> TheoreticalPredictions:
        """
        Compute theoretical predictions for specified model and parameters.
        
        Args:
            model_type: Type of model ('ising', 'xy', 'heisenberg')
            system_parameters: System parameters for the model
            dimensionality: System dimensionality
            
        Returns:
            TheoreticalPredictions with computed values
        """
        dim = dimensionality or self.default_dimensionality
        
        if model_type not in self.models:
            raise TheoreticalModelError(f"Unknown model type: {model_type}")
        
        if dim not in self.models[model_type]:
            raise TheoreticalModelError(f"{model_type} model not available for {dim}D")
        
        model = self.models[model_type][dim]
        
        # Compute basic predictions
        critical_temperature = model.compute_critical_temperature(system_parameters)
        critical_exponents = model.compute_critical_exponents()
        
        # Generate temperature range for order parameter behavior
        temp_range = np.linspace(0.5 * critical_temperature, 2.0 * critical_temperature, 100)
        order_parameter_behavior = {
            'temperatures': temp_range,
            'order_parameter': model.compute_order_parameter_behavior(temp_range, system_parameters)
        }
        
        # Additional predictions based on model type
        correlation_functions = {}
        susceptibility_predictions = {}
        specific_heat_predictions = {}
        
        if hasattr(model, 'compute_susceptibility'):
            susceptibility_predictions['magnetic_susceptibility'] = model.compute_susceptibility(
                temp_range, system_parameters
            )
        
        if hasattr(model, 'compute_specific_heat'):
            specific_heat_predictions['specific_heat'] = model.compute_specific_heat(
                temp_range, system_parameters
            )
        
        predictions = TheoreticalPredictions(
            model_type=f"{model_type}_{dim}d",
            critical_temperature=critical_temperature,
            critical_exponents=critical_exponents,
            order_parameter_behavior=order_parameter_behavior,
            correlation_functions=correlation_functions,
            susceptibility_predictions=susceptibility_predictions,
            specific_heat_predictions=specific_heat_predictions
        )
        
        return predictions
    
    def select_best_model(self,
                         latent_repr: LatentRepresentation,
                         candidate_models: List[str] = None,
                         system_size: Optional[int] = None,
                         use_advanced_criteria: bool = True) -> Tuple[str, Dict[str, Any]]:
        """
        Select the best theoretical model based on agreement with computational results.
        
        Args:
            latent_repr: Latent space representation
            candidate_models: List of models to consider (defaults to all available)
            system_size: System size for finite-size corrections
            use_advanced_criteria: Whether to use advanced model selection criteria
            
        Returns:
            Tuple of (best_model_name, comparison_results)
        """
        self.logger.info("Selecting best theoretical model")
        
        if candidate_models is None:
            candidate_models = ['ising', 'xy', 'heisenberg']
        
        model_comparisons = {}
        
        for model_type in candidate_models:
            try:
                if model_type == 'ising':
                    validation = self.validate_against_ising_model(
                        latent_repr, system_size or 32
                    )
                elif model_type == 'xy':
                    validation = self.validate_against_xy_model(latent_repr)
                elif model_type == 'heisenberg':
                    validation = self.validate_against_heisenberg_model(latent_repr)
                else:
                    continue
                
                # Compute overall agreement score
                if use_advanced_criteria:
                    agreement_score = self._compute_advanced_agreement_score(
                        validation, model_type, latent_repr
                    )
                else:
                    agreement_score = self._compute_overall_agreement_score(validation.agreement_metrics)
                
                model_comparisons[model_type] = {
                    'validation': validation,
                    'agreement_score': agreement_score,
                    'critical_temperature_agreement': validation.agreement_metrics.get(
                        'critical_temperature_agreement', 0
                    ),
                    'model_specific_metrics': self._compute_model_specific_metrics(
                        validation, model_type, latent_repr
                    )
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to validate against {model_type} model: {str(e)}")
                model_comparisons[model_type] = {
                    'validation': None,
                    'agreement_score': 0.0,
                    'error': str(e)
                }
        
        # Select best model based on agreement score
        if not model_comparisons:
            raise TheoreticalModelError("No models could be validated")
        
        best_model = max(model_comparisons.keys(), 
                        key=lambda k: model_comparisons[k]['agreement_score'])
        
        # Add model ranking and confidence assessment
        sorted_models = sorted(model_comparisons.items(), 
                             key=lambda x: x[1]['agreement_score'], reverse=True)
        
        model_ranking = {model: rank + 1 for rank, (model, _) in enumerate(sorted_models)}
        
        # Compute selection confidence
        if len(sorted_models) > 1:
            best_score = sorted_models[0][1]['agreement_score']
            second_score = sorted_models[1][1]['agreement_score']
            selection_confidence = (best_score - second_score) / best_score if best_score > 0 else 0
        else:
            selection_confidence = 1.0
        
        model_comparisons['_meta'] = {
            'model_ranking': model_ranking,
            'selection_confidence': selection_confidence,
            'best_model': best_model
        }
        
        self.logger.info(f"Best model selected: {best_model} "
                        f"(score: {model_comparisons[best_model]['agreement_score']:.3f}, "
                        f"confidence: {selection_confidence:.3f})")
        
        return best_model, model_comparisons
    
    def compare_multiple_models(self,
                              latent_repr: LatentRepresentation,
                              models_to_compare: List[str],
                              system_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform detailed comparison between multiple theoretical models.
        
        Args:
            latent_repr: Latent space representation
            models_to_compare: List of model names to compare
            system_size: System size for finite-size corrections
            
        Returns:
            Dictionary with detailed comparison results
        """
        self.logger.info(f"Comparing models: {models_to_compare}")
        
        comparison_results = {}
        
        # Validate against each model
        for model_type in models_to_compare:
            try:
                if model_type == 'ising':
                    validation = self.validate_against_ising_model(latent_repr, system_size or 32)
                elif model_type == 'xy':
                    validation = self.validate_against_xy_model(latent_repr)
                elif model_type == 'heisenberg':
                    validation = self.validate_against_heisenberg_model(latent_repr)
                else:
                    continue
                
                comparison_results[model_type] = {
                    'validation': validation,
                    'theoretical_predictions': validation.theoretical_predictions,
                    'agreement_metrics': validation.agreement_metrics,
                    'model_characteristics': self._get_model_characteristics(model_type)
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to compare with {model_type} model: {str(e)}")
                comparison_results[model_type] = {'error': str(e)}
        
        # Perform cross-model analysis
        cross_analysis = self._perform_cross_model_analysis(comparison_results)
        comparison_results['cross_analysis'] = cross_analysis
        
        return comparison_results
    
    def identify_universality_class(self,
                                  critical_exponents: Dict[str, float],
                                  system_dimensionality: int = None) -> Dict[str, Any]:
        """
        Identify universality class based on critical exponents.
        
        Args:
            critical_exponents: Dictionary of computed critical exponents
            system_dimensionality: System dimensionality
            
        Returns:
            Dictionary with universality class identification results
        """
        dim = system_dimensionality or self.default_dimensionality
        
        # Known universality classes and their exponents
        universality_classes = {
            'ising_2d': {'beta': 1/8, 'gamma': 7/4, 'nu': 1.0, 'alpha': 0.0},
            'ising_3d': {'beta': 0.3265, 'gamma': 1.2372, 'nu': 0.6301, 'alpha': 0.1096},
            'xy_2d': {'eta': 0.25},  # KT transition
            'xy_3d': {'beta': 0.3485, 'gamma': 1.3177, 'nu': 0.6717, 'alpha': -0.0151},
            'heisenberg_3d': {'beta': 0.3689, 'gamma': 1.3960, 'nu': 0.7112, 'alpha': -0.1336}
        }
        
        # Compute distances to known universality classes
        class_distances = {}
        
        for class_name, class_exponents in universality_classes.items():
            distance = 0.0
            n_compared = 0
            
            for exponent, value in critical_exponents.items():
                if exponent in class_exponents:
                    theoretical_value = class_exponents[exponent]
                    relative_error = abs(value - theoretical_value) / abs(theoretical_value)
                    distance += relative_error
                    n_compared += 1
            
            if n_compared > 0:
                class_distances[class_name] = distance / n_compared
        
        # Find best match
        if class_distances:
            best_class = min(class_distances.keys(), key=lambda k: class_distances[k])
            best_distance = class_distances[best_class]
            confidence = max(0.0, 1.0 - best_distance)
        else:
            best_class = 'unknown'
            best_distance = float('inf')
            confidence = 0.0
        
        return {
            'identified_class': best_class,
            'confidence': confidence,
            'class_distances': class_distances,
            'critical_exponents_used': critical_exponents,
            'system_dimensionality': dim
        }
    
    def _extract_computational_results(self, latent_repr: LatentRepresentation) -> Dict[str, float]:
        """Extract relevant computational results from latent representation."""
        results = {}
        
        # Extract magnetization statistics
        magnetizations = np.abs(latent_repr.magnetizations)
        results['mean_magnetization'] = float(np.mean(magnetizations))
        results['max_magnetization'] = float(np.max(magnetizations))
        results['magnetization_std'] = float(np.std(magnetizations))
        
        # Extract temperature statistics
        results['min_temperature'] = float(np.min(latent_repr.temperatures))
        results['max_temperature'] = float(np.max(latent_repr.temperatures))
        results['temperature_range'] = float(np.max(latent_repr.temperatures) - 
                                           np.min(latent_repr.temperatures))
        
        # Extract latent space statistics
        results['z1_mean'] = float(np.mean(latent_repr.z1))
        results['z1_std'] = float(np.std(latent_repr.z1))
        results['z2_mean'] = float(np.mean(latent_repr.z2))
        results['z2_std'] = float(np.std(latent_repr.z2))
        
        # Correlation between latent dimensions and magnetization
        results['z1_magnetization_correlation'] = float(
            np.corrcoef(latent_repr.z1, magnetizations)[0, 1]
        )
        results['z2_magnetization_correlation'] = float(
            np.corrcoef(latent_repr.z2, magnetizations)[0, 1]
        )
        
        return results
    
    def _compute_agreement_metrics(self,
                                 computational_results: Dict[str, float],
                                 theoretical_predictions: TheoreticalPredictions) -> Dict[str, float]:
        """Compute agreement metrics between computational and theoretical results."""
        metrics = {}
        
        # Critical temperature agreement (if available)
        if 'critical_temperature_estimate' in computational_results:
            tc_comp = computational_results['critical_temperature_estimate']
            tc_theo = theoretical_predictions.critical_temperature
            tc_error = abs(tc_comp - tc_theo) / tc_theo
            metrics['critical_temperature_agreement'] = max(0.0, 1.0 - tc_error)
            metrics['critical_temperature_relative_error'] = tc_error
        
        # Order parameter correlation agreement
        max_corr = max(
            abs(computational_results.get('z1_magnetization_correlation', 0)),
            abs(computational_results.get('z2_magnetization_correlation', 0))
        )
        metrics['order_parameter_correlation'] = max_corr
        
        # Magnetization range agreement
        theo_behavior = theoretical_predictions.order_parameter_behavior
        if 'order_parameter' in theo_behavior:
            theo_max = np.max(theo_behavior['order_parameter'])
            comp_max = computational_results.get('max_magnetization', 0)
            if theo_max > 0:
                mag_agreement = min(comp_max / theo_max, theo_max / comp_max)
                metrics['magnetization_scale_agreement'] = mag_agreement
        
        return metrics
    
    def _compute_overall_agreement_score(self, agreement_metrics: Dict[str, float]) -> float:
        """Compute overall agreement score from individual metrics."""
        weights = {
            'critical_temperature_agreement': 0.4,
            'order_parameter_correlation': 0.4,
            'magnetization_scale_agreement': 0.2
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in agreement_metrics:
                score += weight * agreement_metrics[metric]
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _compare_with_onsager_solution(self,
                                     computational_results: Dict[str, float],
                                     theoretical_predictions: TheoreticalPredictions) -> Dict[str, Any]:
        """Compare with exact Onsager solution for 2D Ising model."""
        comparison = {}
        
        # Exact critical temperature
        tc_onsager = 2.269185314213022
        comparison['exact_critical_temperature'] = tc_onsager
        
        if 'critical_temperature_estimate' in computational_results:
            tc_comp = computational_results['critical_temperature_estimate']
            comparison['computational_critical_temperature'] = tc_comp
            comparison['onsager_agreement'] = 1.0 - abs(tc_comp - tc_onsager) / tc_onsager
        
        # Exact critical exponents
        exact_exponents = {
            'beta': 1/8,
            'gamma': 7/4,
            'nu': 1.0,
            'alpha': 0.0
        }
        comparison['exact_critical_exponents'] = exact_exponents
        
        return comparison
    
    def _compare_with_mean_field_theory(self,
                                      computational_results: Dict[str, float],
                                      theoretical_predictions: TheoreticalPredictions) -> Dict[str, Any]:
        """Compare with mean field theory predictions."""
        comparison = {}
        
        # Mean field critical exponents
        mf_exponents = {
            'beta': 0.5,
            'gamma': 1.0,
            'nu': 0.5,
            'alpha': 0.0
        }
        comparison['mean_field_exponents'] = mf_exponents
        
        # Compare with theoretical exponents
        theo_exponents = theoretical_predictions.critical_exponents
        comparison['deviation_from_mean_field'] = {}
        
        for exponent in ['beta', 'gamma', 'nu']:
            if exponent in theo_exponents:
                mf_value = mf_exponents[exponent]
                theo_value = theo_exponents[exponent]
                deviation = abs(theo_value - mf_value) / mf_value if mf_value != 0 else 0
                comparison['deviation_from_mean_field'][exponent] = deviation
        
        return comparison
    
    def _analyze_kosterlitz_thouless_transition(self,
                                              latent_repr: LatentRepresentation,
                                              theoretical_predictions: TheoreticalPredictions) -> Dict[str, Any]:
        """Analyze Kosterlitz-Thouless transition characteristics."""
        analysis = {}
        
        # KT transition has no conventional order parameter
        # Look for helicity modulus or superfluid density behavior
        analysis['transition_type'] = 'kosterlitz_thouless'
        analysis['has_conventional_order_parameter'] = False
        
        # Analyze temperature dependence of correlations
        temperatures = latent_repr.temperatures
        unique_temps = np.unique(temperatures)
        
        if len(unique_temps) > 5:
            # Look for exponential decay of correlations above Tc
            tc = theoretical_predictions.critical_temperature
            
            above_tc_temps = unique_temps[unique_temps > tc]
            if len(above_tc_temps) > 3:
                # Placeholder for correlation length analysis
                analysis['correlation_length_behavior'] = 'exponential_decay'
                analysis['kt_temperature_estimate'] = tc
        
        return analysis
    
    def _compute_advanced_agreement_score(self,
                                        validation: Union[IsingModelValidation, XYModelValidation, HeisenbergModelValidation],
                                        model_type: str,
                                        latent_repr: LatentRepresentation) -> float:
        """Compute advanced agreement score using model-specific criteria."""
        base_score = self._compute_overall_agreement_score(validation.agreement_metrics)
        
        # Model-specific adjustments
        if model_type == 'ising':
            # Bonus for Onsager solution agreement in 2D
            if hasattr(validation, 'onsager_solution_comparison') and validation.onsager_solution_comparison:
                onsager_agreement = validation.onsager_solution_comparison.get('onsager_agreement', 0)
                base_score += 0.1 * onsager_agreement
        
        elif model_type == 'xy':
            # Check for KT transition characteristics in 2D
            if hasattr(validation, 'kosterlitz_thouless_analysis') and validation.kosterlitz_thouless_analysis:
                kt_analysis = validation.kosterlitz_thouless_analysis
                if kt_analysis.get('transition_type') == 'kosterlitz_thouless':
                    base_score += 0.05  # Small bonus for KT characteristics
        
        elif model_type == 'heisenberg':
            # Check for continuous symmetry characteristics
            # Look for smooth order parameter behavior
            magnetizations = np.abs(latent_repr.magnetizations)
            if len(magnetizations) > 10:
                # Measure smoothness of magnetization curve
                smoothness = 1.0 / (1.0 + np.std(np.diff(magnetizations)))
                base_score += 0.05 * smoothness
        
        return min(1.0, base_score)  # Cap at 1.0
    
    def _compute_model_specific_metrics(self,
                                      validation: Union[IsingModelValidation, XYModelValidation, HeisenbergModelValidation],
                                      model_type: str,
                                      latent_repr: LatentRepresentation) -> Dict[str, Any]:
        """Compute model-specific validation metrics."""
        metrics = {}
        
        if model_type == 'ising':
            # Ising-specific metrics
            if hasattr(validation, 'onsager_solution_comparison'):
                metrics['onsager_agreement'] = validation.onsager_solution_comparison.get('onsager_agreement', 0)
            
            if hasattr(validation, 'mean_field_comparison'):
                mf_comp = validation.mean_field_comparison
                if 'deviation_from_mean_field' in mf_comp:
                    avg_deviation = np.mean(list(mf_comp['deviation_from_mean_field'].values()))
                    metrics['mean_field_deviation'] = avg_deviation
        
        elif model_type == 'xy':
            # XY-specific metrics
            if hasattr(validation, 'kosterlitz_thouless_analysis'):
                kt_analysis = validation.kosterlitz_thouless_analysis
                metrics['is_kt_transition'] = kt_analysis.get('transition_type') == 'kosterlitz_thouless'
                metrics['has_conventional_order_parameter'] = kt_analysis.get('has_conventional_order_parameter', True)
        
        elif model_type == 'heisenberg':
            # Heisenberg-specific metrics
            # Analyze continuous symmetry properties
            magnetizations = np.abs(latent_repr.magnetizations)
            if len(magnetizations) > 5:
                # Measure order parameter smoothness (continuous symmetry indicator)
                smoothness = 1.0 / (1.0 + np.std(np.diff(magnetizations)))
                metrics['order_parameter_smoothness'] = smoothness
                
                # Check for gradual onset (characteristic of continuous transitions)
                temp_sorted_indices = np.argsort(latent_repr.temperatures)
                sorted_mags = magnetizations[temp_sorted_indices]
                gradual_onset_score = self._measure_gradual_onset(sorted_mags)
                metrics['gradual_onset_score'] = gradual_onset_score
        
        return metrics
    
    def _get_model_characteristics(self, model_type: str) -> Dict[str, Any]:
        """Get characteristic properties of theoretical models."""
        characteristics = {
            'ising': {
                'symmetry': 'Z2 (discrete)',
                'order_parameter': 'magnetization',
                'transition_type': 'second_order',
                'dimensionality_support': [2, 3],
                'exact_solutions': ['2D (Onsager)'],
                'universality_class': 'Ising'
            },
            'xy': {
                'symmetry': 'U(1) (continuous)',
                'order_parameter': 'complex order parameter / helicity modulus',
                'transition_type': '2D: KT, 3D: second_order',
                'dimensionality_support': [2, 3],
                'exact_solutions': [],
                'universality_class': '2D: KT, 3D: XY'
            },
            'heisenberg': {
                'symmetry': 'O(3) (continuous)',
                'order_parameter': 'vector magnetization',
                'transition_type': 'second_order',
                'dimensionality_support': [3],
                'exact_solutions': [],
                'universality_class': 'Heisenberg'
            }
        }
        
        return characteristics.get(model_type, {})
    
    def _perform_cross_model_analysis(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform analysis across multiple models."""
        analysis = {}
        
        # Extract critical temperatures from all models
        critical_temperatures = {}
        agreement_scores = {}
        
        for model_type, results in comparison_results.items():
            if 'validation' in results and results['validation']:
                validation = results['validation']
                tc = validation.theoretical_predictions.critical_temperature
                critical_temperatures[model_type] = tc
                
                if 'agreement_metrics' in results:
                    score = self._compute_overall_agreement_score(results['agreement_metrics'])
                    agreement_scores[model_type] = score
        
        # Analyze critical temperature spread
        if len(critical_temperatures) > 1:
            tc_values = list(critical_temperatures.values())
            tc_mean = np.mean(tc_values)
            tc_std = np.std(tc_values)
            tc_range = np.max(tc_values) - np.min(tc_values)
            
            analysis['critical_temperature_analysis'] = {
                'mean': tc_mean,
                'std': tc_std,
                'range': tc_range,
                'relative_spread': tc_std / tc_mean if tc_mean > 0 else 0,
                'individual_values': critical_temperatures
            }
        
        # Analyze agreement score distribution
        if agreement_scores:
            scores = list(agreement_scores.values())
            analysis['agreement_score_analysis'] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'range': np.max(scores) - np.min(scores),
                'individual_scores': agreement_scores
            }
        
        # Model recommendation based on cross-analysis
        if agreement_scores:
            best_model = max(agreement_scores.keys(), key=lambda k: agreement_scores[k])
            best_score = agreement_scores[best_model]
            
            # Confidence in recommendation
            if len(agreement_scores) > 1:
                sorted_scores = sorted(agreement_scores.values(), reverse=True)
                confidence = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0] if sorted_scores[0] > 0 else 0
            else:
                confidence = 1.0
            
            analysis['recommendation'] = {
                'best_model': best_model,
                'confidence': confidence,
                'score': best_score
            }
        
        return analysis
    
    def _measure_gradual_onset(self, sorted_magnetizations: np.ndarray) -> float:
        """Measure how gradual the order parameter onset is."""
        if len(sorted_magnetizations) < 5:
            return 0.0
        
        # Look for smooth, gradual increase rather than sharp jump
        # Compute second derivative to measure curvature
        first_diff = np.diff(sorted_magnetizations)
        second_diff = np.diff(first_diff)
        
        # Gradual onset has small second derivatives (low curvature)
        curvature_measure = np.mean(np.abs(second_diff))
        gradual_score = 1.0 / (1.0 + 10.0 * curvature_measure)
        
        return gradual_score