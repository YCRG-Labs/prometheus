"""
Robust Critical Exponent Extractor with Validation

This module implements task 8.4: Implement robust critical exponent extraction with validation
by adding multiple fitting range optimization, cross-validation, and ensemble averaging.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from scipy.stats import bootstrap
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import warnings

from .robust_power_law_fitter import RobustPowerLawFitter, RobustFitResult
from .correlation_length_calculator import CorrelationLengthCalculator, MultiMethodCorrelationResult
from .latent_analysis import LatentRepresentation
from ..utils.logging_utils import get_logger


@dataclass
class FittingRangeResult:
    """Container for fitting range optimization results."""
    temperature_range: Tuple[float, float]
    fit_result: RobustFitResult
    range_score: float
    n_points_used: int
    range_fraction: float


@dataclass
class CrossValidationResult:
    """Container for cross-validation results."""
    fold_results: List[RobustFitResult]
    mean_exponent: float
    std_exponent: float
    cv_score: float
    consistency_score: float
    statistical_significance: float


@dataclass
class EnsembleResult:
    """Container for ensemble averaging results."""
    individual_results: List[RobustFitResult]
    ensemble_exponent: float
    ensemble_error: float
    ensemble_confidence_interval: Tuple[float, float]
    method_weights: np.ndarray
    ensemble_score: float


@dataclass
class RobustExtractionResult:
    """Container for complete robust extraction results."""
    exponent_type: str
    critical_temperature: float
    
    # Range optimization results
    optimal_range: FittingRangeResult
    range_candidates: List[FittingRangeResult]
    
    # Cross-validation results
    cross_validation: CrossValidationResult
    
    # Ensemble results
    ensemble: EnsembleResult
    
    # Final validated result
    final_exponent: float
    final_error: float
    final_confidence_interval: Tuple[float, float]
    
    # Validation metrics
    theoretical_exponent: Optional[float]
    accuracy_percent: Optional[float]
    statistical_significance: float
    validation_passed: bool
    
    # Quality metrics
    overall_quality_score: float
    robustness_score: float


class RobustCriticalExponentExtractor:
    """
    Robust critical exponent extractor with comprehensive validation.
    
    Features:
    1. Multiple fitting range optimization
    2. Cross-validation for parameter selection
    3. Statistical significance testing
    4. Ensemble averaging across multiple methods
    5. Comprehensive validation framework
    """
    
    def __init__(self,
                 n_range_candidates: int = 20,
                 cv_folds: int = 5,
                 ensemble_methods: int = 5,
                 bootstrap_samples: int = 2000,
                 significance_threshold: float = 0.05,
                 random_seed: Optional[int] = None):
        """
        Initialize robust critical exponent extractor.
        
        Args:
            n_range_candidates: Number of fitting ranges to test
            cv_folds: Number of cross-validation folds
            ensemble_methods: Number of ensemble methods
            bootstrap_samples: Number of bootstrap samples
            significance_threshold: P-value threshold for significance
            random_seed: Random seed for reproducibility
        """
        self.n_range_candidates = n_range_candidates
        self.cv_folds = cv_folds
        self.ensemble_methods = ensemble_methods
        self.bootstrap_samples = bootstrap_samples
        self.significance_threshold = significance_threshold
        self.random_seed = random_seed
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.power_law_fitter = RobustPowerLawFitter(
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed
        )
        
        self.correlation_calculator = CorrelationLengthCalculator()
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def extract_robust_critical_exponent(self,
                                       latent_repr: LatentRepresentation,
                                       critical_temperature: float,
                                       exponent_type: str,
                                       theoretical_exponent: Optional[float] = None) -> RobustExtractionResult:
        """
        Extract critical exponent using robust methods with comprehensive validation.
        
        Args:
            latent_repr: LatentRepresentation object
            critical_temperature: Critical temperature
            exponent_type: Type of exponent ('beta', 'nu', 'gamma')
            theoretical_exponent: Theoretical value for validation
            
        Returns:
            RobustExtractionResult with complete analysis
        """
        self.logger.info(f"Starting robust {exponent_type} exponent extraction")
        
        # Prepare data based on exponent type
        temperatures, observables = self._prepare_data_for_exponent(
            latent_repr, critical_temperature, exponent_type
        )
        
        if len(temperatures) < self.power_law_fitter.min_points:
            raise ValueError(f"Insufficient data points: {len(temperatures)}")
        
        # Step 1: Optimize fitting range
        self.logger.info("Optimizing fitting ranges")
        range_candidates = self._optimize_fitting_ranges(
            temperatures, observables, critical_temperature, exponent_type
        )
        
        optimal_range = max(range_candidates, key=lambda r: r.range_score)
        
        # Step 2: Cross-validation
        self.logger.info("Performing cross-validation")
        cv_result = self._perform_cross_validation(
            temperatures, observables, critical_temperature, 
            exponent_type, optimal_range.temperature_range
        )
        
        # Step 3: Ensemble averaging
        self.logger.info("Computing ensemble average")
        ensemble_result = self._compute_ensemble_average(
            temperatures, observables, critical_temperature, 
            exponent_type, range_candidates[:self.ensemble_methods]
        )
        
        # Step 4: Final validation and result selection
        final_exponent, final_error, final_ci = self._select_final_result(
            optimal_range, cv_result, ensemble_result
        )
        
        # Step 5: Compute validation metrics
        accuracy_percent = None
        if theoretical_exponent is not None:
            accuracy_percent = (1 - abs(final_exponent - theoretical_exponent) / abs(theoretical_exponent)) * 100
        
        statistical_significance = self._compute_statistical_significance(
            final_exponent, final_error
        )
        
        validation_passed = self._validate_extraction_result(
            final_exponent, final_error, theoretical_exponent, statistical_significance
        )
        
        # Step 6: Compute quality scores
        overall_quality_score = self._compute_overall_quality_score(
            optimal_range, cv_result, ensemble_result, validation_passed
        )
        
        robustness_score = self._compute_robustness_score(
            range_candidates, cv_result, ensemble_result
        )
        
        result = RobustExtractionResult(
            exponent_type=exponent_type,
            critical_temperature=critical_temperature,
            optimal_range=optimal_range,
            range_candidates=range_candidates,
            cross_validation=cv_result,
            ensemble=ensemble_result,
            final_exponent=final_exponent,
            final_error=final_error,
            final_confidence_interval=final_ci,
            theoretical_exponent=theoretical_exponent,
            accuracy_percent=accuracy_percent,
            statistical_significance=statistical_significance,
            validation_passed=validation_passed,
            overall_quality_score=overall_quality_score,
            robustness_score=robustness_score
        )
        
        self.logger.info(f"Robust {exponent_type} extraction completed")
        self.logger.info(f"Final exponent: {final_exponent:.4f} ± {final_error:.4f}")
        if accuracy_percent is not None:
            self.logger.info(f"Accuracy: {accuracy_percent:.1f}%")
        
        return result
    
    def _prepare_data_for_exponent(self,
                                 latent_repr: LatentRepresentation,
                                 critical_temperature: float,
                                 exponent_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare temperature and observable data for specific exponent type."""
        
        if exponent_type == 'beta':
            # Beta exponent: use magnetization data, preferably below Tc
            temperatures = latent_repr.temperatures
            observables = np.abs(latent_repr.magnetizations)
            
            # Filter to below Tc if enough points
            below_tc_mask = temperatures < critical_temperature
            if np.sum(below_tc_mask) >= self.power_law_fitter.min_points:
                temperatures = temperatures[below_tc_mask]
                observables = observables[below_tc_mask]
        
        elif exponent_type == 'nu':
            # Nu exponent: use correlation length data
            corr_result = self.correlation_calculator.compute_correlation_length_multi_method(
                latent_repr, critical_temperature
            )
            
            # Use the best correlation length method
            best_method = corr_result.combined_method
            temperatures = best_method.temperatures
            observables = best_method.correlation_lengths
        
        elif exponent_type == 'gamma':
            # Gamma exponent: use susceptibility data
            temperatures, susceptibilities = self._compute_susceptibility_from_latent(
                latent_repr
            )
            observables = susceptibilities
        
        else:
            raise ValueError(f"Unknown exponent type: {exponent_type}")
        
        # Remove invalid data points
        valid_mask = (observables > 0) & np.isfinite(observables) & np.isfinite(temperatures)
        temperatures = temperatures[valid_mask]
        observables = observables[valid_mask]
        
        return temperatures, observables
    
    def _compute_susceptibility_from_latent(self, latent_repr: LatentRepresentation) -> Tuple[np.ndarray, np.ndarray]:
        """Compute susceptibility from latent representation."""
        
        # Bin magnetizations by temperature
        unique_temps = np.unique(latent_repr.temperatures)
        temperatures = []
        susceptibilities = []
        
        for temp in unique_temps:
            temp_mask = latent_repr.temperatures == temp
            if np.sum(temp_mask) >= 5:  # Minimum points for variance calculation
                temp_mags = latent_repr.magnetizations[temp_mask]
                susceptibility = np.var(temp_mags)  # χ ∝ ⟨m²⟩ - ⟨m⟩²
                
                if susceptibility > 0:
                    temperatures.append(temp)
                    susceptibilities.append(susceptibility)
        
        return np.array(temperatures), np.array(susceptibilities)
    
    def _optimize_fitting_ranges(self,
                               temperatures: np.ndarray,
                               observables: np.ndarray,
                               critical_temperature: float,
                               exponent_type: str) -> List[FittingRangeResult]:
        """Optimize fitting ranges using multiple candidates."""
        
        temp_min, temp_max = np.min(temperatures), np.max(temperatures)
        temp_range = temp_max - temp_min
        
        # Generate range candidates
        range_candidates = []
        
        # Different range strategies
        if exponent_type == 'beta':
            # For beta: focus on temperatures below Tc
            max_distance = min(0.5 * temp_range, critical_temperature - temp_min)
            distances = np.linspace(0.05 * temp_range, max_distance, self.n_range_candidates)
            
            for distance in distances:
                range_min = max(temp_min, critical_temperature - distance)
                range_max = critical_temperature
                
                if range_max > range_min:
                    range_candidates.append((range_min, range_max))
        
        else:
            # For nu and gamma: use symmetric ranges around Tc
            max_distance = min(0.4 * temp_range, 
                             min(critical_temperature - temp_min, temp_max - critical_temperature))
            distances = np.linspace(0.05 * temp_range, max_distance, self.n_range_candidates)
            
            for distance in distances:
                range_min = critical_temperature - distance
                range_max = critical_temperature + distance
                
                # Ensure range is within data bounds
                range_min = max(range_min, temp_min)
                range_max = min(range_max, temp_max)
                
                if range_max > range_min:
                    range_candidates.append((range_min, range_max))
        
        # Evaluate each range candidate
        results = []
        
        for range_min, range_max in range_candidates:
            try:
                # Filter data to range
                range_mask = (temperatures >= range_min) & (temperatures <= range_max)
                
                if np.sum(range_mask) < self.power_law_fitter.min_points:
                    continue
                
                range_temps = temperatures[range_mask]
                range_obs = observables[range_mask]
                
                # Fit power law
                fit_result = self.power_law_fitter.fit_power_law_robust(
                    range_temps, range_obs, critical_temperature, exponent_type
                )
                
                # Score this range
                range_score = self._score_fitting_range(
                    fit_result, range_temps, range_obs, critical_temperature, exponent_type
                )
                
                range_result = FittingRangeResult(
                    temperature_range=(range_min, range_max),
                    fit_result=fit_result,
                    range_score=range_score,
                    n_points_used=len(range_temps),
                    range_fraction=(range_max - range_min) / temp_range
                )
                
                results.append(range_result)
                
            except Exception as e:
                self.logger.debug(f"Range ({range_min:.3f}, {range_max:.3f}) failed: {e}")
                continue
        
        if not results:
            raise RuntimeError("No valid fitting ranges found")
        
        # Sort by score
        results.sort(key=lambda r: r.range_score, reverse=True)
        
        return results
    
    def _score_fitting_range(self,
                           fit_result: RobustFitResult,
                           temperatures: np.ndarray,
                           observables: np.ndarray,
                           critical_temperature: float,
                           exponent_type: str) -> float:
        """Score a fitting range based on multiple criteria."""
        
        score = 0.0
        
        # R-squared contribution (30%)
        score += 0.3 * max(0, fit_result.r_squared)
        
        # Statistical significance (25%)
        if fit_result.p_value < 0.001:
            sig_score = 1.0
        elif fit_result.p_value < 0.01:
            sig_score = 0.8
        elif fit_result.p_value < 0.05:
            sig_score = 0.6
        else:
            sig_score = 0.2
        score += 0.25 * sig_score
        
        # Parameter precision (20%)
        if fit_result.exponent != 0:
            relative_error = fit_result.exponent_error / abs(fit_result.exponent)
            precision_score = max(0, 1 - relative_error)
        else:
            precision_score = 0
        score += 0.2 * precision_score
        
        # Number of points (15%)
        n_points = len(temperatures)
        points_score = min(1.0, n_points / 20)
        score += 0.15 * points_score
        
        # Physics reasonableness (10%)
        physics_score = self._assess_physics_reasonableness(fit_result.exponent, exponent_type)
        score += 0.1 * physics_score
        
        return score
    
    def _assess_physics_reasonableness(self, exponent: float, exponent_type: str) -> float:
        """Assess if exponent value is physically reasonable."""
        
        if exponent_type == 'beta':
            # Beta should be positive, typically 0.1 - 0.5
            if 0.05 <= exponent <= 0.6:
                return 1.0
            elif 0.01 <= exponent <= 1.0:
                return 0.7
            else:
                return 0.2
        
        elif exponent_type == 'nu':
            # Nu should be negative (we fit |T-Tc|^(-nu)), typically -0.5 to -1.0
            if -1.5 <= exponent <= -0.3:
                return 1.0
            elif -2.0 <= exponent <= -0.1:
                return 0.7
            else:
                return 0.2
        
        elif exponent_type == 'gamma':
            # Gamma should be negative, typically -1.0 to -2.0
            if -2.5 <= exponent <= -0.8:
                return 1.0
            elif -3.0 <= exponent <= -0.5:
                return 0.7
            else:
                return 0.2
        
        return 0.5
    
    def _perform_cross_validation(self,
                                temperatures: np.ndarray,
                                observables: np.ndarray,
                                critical_temperature: float,
                                exponent_type: str,
                                optimal_range: Tuple[float, float]) -> CrossValidationResult:
        """Perform cross-validation for parameter selection."""
        
        # Filter data to optimal range
        range_min, range_max = optimal_range
        range_mask = (temperatures >= range_min) & (temperatures <= range_max)
        range_temps = temperatures[range_mask]
        range_obs = observables[range_mask]
        
        if len(range_temps) < self.cv_folds:
            # Too few points for CV, use bootstrap instead
            return self._bootstrap_validation(range_temps, range_obs, critical_temperature, exponent_type)
        
        # Perform k-fold cross-validation
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)
        
        fold_results = []
        
        for train_idx, test_idx in kfold.split(range_temps):
            try:
                train_temps = range_temps[train_idx]
                train_obs = range_obs[train_idx]
                
                # Fit on training data
                fit_result = self.power_law_fitter.fit_power_law_robust(
                    train_temps, train_obs, critical_temperature, exponent_type
                )
                
                fold_results.append(fit_result)
                
            except Exception as e:
                self.logger.debug(f"CV fold failed: {e}")
                continue
        
        if not fold_results:
            raise RuntimeError("All cross-validation folds failed")
        
        # Compute CV statistics
        exponents = [r.exponent for r in fold_results]
        mean_exponent = np.mean(exponents)
        std_exponent = np.std(exponents)
        
        # CV score based on consistency
        cv_score = 1.0 / (1.0 + std_exponent / abs(mean_exponent)) if mean_exponent != 0 else 0.5
        
        # Consistency score
        consistency_score = 1.0 - (std_exponent / abs(mean_exponent)) if mean_exponent != 0 else 0.0
        consistency_score = max(0, consistency_score)
        
        # Statistical significance
        if std_exponent > 0:
            t_stat = abs(mean_exponent) / (std_exponent / np.sqrt(len(exponents)))
            # Rough p-value approximation
            statistical_significance = 2 * (1 - 0.5 * (1 + np.tanh(t_stat - 2)))
        else:
            statistical_significance = 0.001
        
        return CrossValidationResult(
            fold_results=fold_results,
            mean_exponent=mean_exponent,
            std_exponent=std_exponent,
            cv_score=cv_score,
            consistency_score=consistency_score,
            statistical_significance=statistical_significance
        )
    
    def _bootstrap_validation(self,
                            temperatures: np.ndarray,
                            observables: np.ndarray,
                            critical_temperature: float,
                            exponent_type: str) -> CrossValidationResult:
        """Perform bootstrap validation when CV is not feasible."""
        
        n_bootstrap = min(self.cv_folds * 2, 20)
        bootstrap_results = []
        
        rng = np.random.RandomState(self.random_seed)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = rng.choice(len(temperatures), size=len(temperatures), replace=True)
            boot_temps = temperatures[indices]
            boot_obs = observables[indices]
            
            try:
                fit_result = self.power_law_fitter.fit_power_law_robust(
                    boot_temps, boot_obs, critical_temperature, exponent_type
                )
                bootstrap_results.append(fit_result)
            except:
                continue
        
        if not bootstrap_results:
            raise RuntimeError("Bootstrap validation failed")
        
        # Compute statistics
        exponents = [r.exponent for r in bootstrap_results]
        mean_exponent = np.mean(exponents)
        std_exponent = np.std(exponents)
        
        cv_score = 1.0 / (1.0 + std_exponent / abs(mean_exponent)) if mean_exponent != 0 else 0.5
        consistency_score = max(0, 1.0 - (std_exponent / abs(mean_exponent))) if mean_exponent != 0 else 0.0
        
        statistical_significance = 0.05  # Conservative estimate for bootstrap
        
        return CrossValidationResult(
            fold_results=bootstrap_results,
            mean_exponent=mean_exponent,
            std_exponent=std_exponent,
            cv_score=cv_score,
            consistency_score=consistency_score,
            statistical_significance=statistical_significance
        )
    
    def _compute_ensemble_average(self,
                                temperatures: np.ndarray,
                                observables: np.ndarray,
                                critical_temperature: float,
                                exponent_type: str,
                                range_candidates: List[FittingRangeResult]) -> EnsembleResult:
        """Compute ensemble average across multiple fitting methods."""
        
        individual_results = []
        
        # Use top range candidates for ensemble
        for range_result in range_candidates:
            individual_results.append(range_result.fit_result)
        
        if len(individual_results) < 2:
            # Not enough for ensemble, use single best result
            best_result = individual_results[0]
            return EnsembleResult(
                individual_results=individual_results,
                ensemble_exponent=best_result.exponent,
                ensemble_error=best_result.exponent_error,
                ensemble_confidence_interval=(
                    best_result.exponent - 1.96 * best_result.exponent_error,
                    best_result.exponent + 1.96 * best_result.exponent_error
                ),
                method_weights=np.array([1.0]),
                ensemble_score=best_result.r_squared
            )
        
        # Compute weights based on quality scores
        weights = []
        for result in individual_results:
            weight = result.r_squared * (1 - result.p_value) if result.p_value < 1 else result.r_squared
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize
        
        # Compute weighted average
        exponents = np.array([r.exponent for r in individual_results])
        errors = np.array([r.exponent_error for r in individual_results])
        
        ensemble_exponent = np.average(exponents, weights=weights)
        
        # Ensemble error using weighted variance
        ensemble_variance = np.average((exponents - ensemble_exponent)**2, weights=weights)
        ensemble_error = np.sqrt(ensemble_variance + np.average(errors**2, weights=weights))
        
        # Confidence interval
        ensemble_ci = (
            ensemble_exponent - 1.96 * ensemble_error,
            ensemble_exponent + 1.96 * ensemble_error
        )
        
        # Ensemble score
        ensemble_score = np.average([r.r_squared for r in individual_results], weights=weights)
        
        return EnsembleResult(
            individual_results=individual_results,
            ensemble_exponent=ensemble_exponent,
            ensemble_error=ensemble_error,
            ensemble_confidence_interval=ensemble_ci,
            method_weights=weights,
            ensemble_score=ensemble_score
        )
    
    def _select_final_result(self,
                           optimal_range: FittingRangeResult,
                           cv_result: CrossValidationResult,
                           ensemble_result: EnsembleResult) -> Tuple[float, float, Tuple[float, float]]:
        """Select final result from optimal range, CV, and ensemble methods."""
        
        # Score each method
        scores = {}
        
        # Optimal range score
        scores['optimal'] = optimal_range.range_score
        
        # CV score
        scores['cv'] = cv_result.cv_score * cv_result.consistency_score
        
        # Ensemble score
        scores['ensemble'] = ensemble_result.ensemble_score
        
        # Select best method
        best_method = max(scores.keys(), key=lambda k: scores[k])
        
        if best_method == 'optimal':
            exponent = optimal_range.fit_result.exponent
            error = optimal_range.fit_result.exponent_error
            ci = (exponent - 1.96 * error, exponent + 1.96 * error)
        elif best_method == 'cv':
            exponent = cv_result.mean_exponent
            error = cv_result.std_exponent
            ci = (exponent - 1.96 * error, exponent + 1.96 * error)
        else:  # ensemble
            exponent = ensemble_result.ensemble_exponent
            error = ensemble_result.ensemble_error
            ci = ensemble_result.ensemble_confidence_interval
        
        self.logger.info(f"Selected {best_method} method for final result")
        
        return exponent, error, ci
    
    def _compute_statistical_significance(self, exponent: float, error: float) -> float:
        """Compute statistical significance (p-value) for exponent."""
        
        if error <= 0:
            return 1.0
        
        # Two-tailed t-test
        t_stat = abs(exponent) / error
        
        # Rough p-value approximation
        p_value = 2 * (1 - 0.5 * (1 + np.tanh(t_stat - 2)))
        
        return p_value
    
    def _validate_extraction_result(self,
                                  exponent: float,
                                  error: float,
                                  theoretical_exponent: Optional[float],
                                  statistical_significance: float) -> bool:
        """Validate extraction result against multiple criteria."""
        
        validation_passed = True
        
        # Check statistical significance
        if statistical_significance > self.significance_threshold:
            self.logger.warning(f"Result not statistically significant: p = {statistical_significance:.4f}")
            validation_passed = False
        
        # Check error magnitude
        if error > abs(exponent):
            self.logger.warning(f"Large relative error: {error:.4f} > {abs(exponent):.4f}")
            validation_passed = False
        
        # Check against theoretical value if available
        if theoretical_exponent is not None:
            relative_error = abs(exponent - theoretical_exponent) / abs(theoretical_exponent)
            if relative_error > 0.5:  # 50% error threshold
                self.logger.warning(f"Large deviation from theory: {relative_error:.2%}")
                validation_passed = False
        
        return validation_passed
    
    def _compute_overall_quality_score(self,
                                     optimal_range: FittingRangeResult,
                                     cv_result: CrossValidationResult,
                                     ensemble_result: EnsembleResult,
                                     validation_passed: bool) -> float:
        """Compute overall quality score for the extraction."""
        
        score = 0.0
        
        # Range optimization quality (30%)
        score += 0.3 * optimal_range.range_score
        
        # Cross-validation quality (30%)
        score += 0.3 * cv_result.cv_score
        
        # Ensemble quality (25%)
        score += 0.25 * ensemble_result.ensemble_score
        
        # Validation bonus/penalty (15%)
        if validation_passed:
            score += 0.15
        else:
            score -= 0.1
        
        return min(1.0, max(0.0, score))
    
    def _compute_robustness_score(self,
                                range_candidates: List[FittingRangeResult],
                                cv_result: CrossValidationResult,
                                ensemble_result: EnsembleResult) -> float:
        """Compute robustness score based on consistency across methods."""
        
        # Collect all exponent estimates
        all_exponents = []
        
        # From range optimization
        for candidate in range_candidates[:5]:  # Top 5 ranges
            all_exponents.append(candidate.fit_result.exponent)
        
        # From cross-validation
        for fold_result in cv_result.fold_results:
            all_exponents.append(fold_result.exponent)
        
        # From ensemble
        for result in ensemble_result.individual_results:
            all_exponents.append(result.exponent)
        
        if len(all_exponents) < 2:
            return 0.5
        
        # Compute coefficient of variation
        mean_exp = np.mean(all_exponents)
        std_exp = np.std(all_exponents)
        
        if mean_exp == 0:
            return 0.0
        
        cv_coeff = std_exp / abs(mean_exp)
        
        # Robustness score (lower CV = higher robustness)
        robustness_score = 1.0 / (1.0 + cv_coeff)
        
        return robustness_score
    
    def visualize_robust_extraction(self,
                                  result: RobustExtractionResult,
                                  figsize: Tuple[int, int] = (20, 12)) -> plt.Figure:
        """Create comprehensive visualization of robust extraction results."""
        
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        
        # Plot 1: Range optimization results
        ax = axes[0, 0]
        
        ranges = [r.temperature_range for r in result.range_candidates[:10]]
        scores = [r.range_score for r in result.range_candidates[:10]]
        
        range_centers = [(r[0] + r[1]) / 2 for r in ranges]
        range_widths = [r[1] - r[0] for r in ranges]
        
        scatter = ax.scatter(range_centers, range_widths, c=scores, cmap='viridis', s=50)
        
        # Highlight optimal range
        opt_center = (result.optimal_range.temperature_range[0] + result.optimal_range.temperature_range[1]) / 2
        opt_width = result.optimal_range.temperature_range[1] - result.optimal_range.temperature_range[0]
        ax.scatter([opt_center], [opt_width], c='red', s=100, marker='*', label='Optimal')
        
        ax.set_xlabel('Range Center')
        ax.set_ylabel('Range Width')
        ax.set_title('Fitting Range Optimization')
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='Range Score')
        
        # Plot 2: Cross-validation results
        ax = axes[0, 1]
        
        cv_exponents = [r.exponent for r in result.cross_validation.fold_results]
        ax.hist(cv_exponents, bins=min(10, len(cv_exponents)), alpha=0.7, edgecolor='black')
        ax.axvline(result.cross_validation.mean_exponent, color='red', linestyle='--', 
                  label=f'Mean: {result.cross_validation.mean_exponent:.4f}')
        
        if result.theoretical_exponent:
            ax.axvline(result.theoretical_exponent, color='green', linestyle='--',
                      label=f'Theory: {result.theoretical_exponent:.4f}')
        
        ax.set_xlabel('Exponent Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Cross-Validation Distribution')
        ax.legend()
        
        # Plot 3: Ensemble results
        ax = axes[0, 2]
        
        ensemble_exponents = [r.exponent for r in result.ensemble.individual_results]
        weights = result.ensemble.method_weights
        
        bars = ax.bar(range(len(ensemble_exponents)), ensemble_exponents, 
                     color=plt.cm.viridis(weights / np.max(weights)))
        
        ax.axhline(result.ensemble.ensemble_exponent, color='red', linestyle='--',
                  label=f'Ensemble: {result.ensemble.ensemble_exponent:.4f}')
        
        if result.theoretical_exponent:
            ax.axhline(result.theoretical_exponent, color='green', linestyle='--',
                      label=f'Theory: {result.theoretical_exponent:.4f}')
        
        ax.set_xlabel('Method Index')
        ax.set_ylabel('Exponent Value')
        ax.set_title('Ensemble Methods')
        ax.legend()
        
        # Plot 4: Final result comparison
        ax = axes[0, 3]
        
        methods = ['Optimal Range', 'Cross-Validation', 'Ensemble', 'Final']
        values = [
            result.optimal_range.fit_result.exponent,
            result.cross_validation.mean_exponent,
            result.ensemble.ensemble_exponent,
            result.final_exponent
        ]
        errors = [
            result.optimal_range.fit_result.exponent_error,
            result.cross_validation.std_exponent,
            result.ensemble.ensemble_error,
            result.final_error
        ]
        
        bars = ax.bar(methods, values, yerr=errors, capsize=5, alpha=0.7)
        
        if result.theoretical_exponent:
            ax.axhline(result.theoretical_exponent, color='green', linestyle='--',
                      label=f'Theory: {result.theoretical_exponent:.4f}')
        
        ax.set_ylabel('Exponent Value')
        ax.set_title('Method Comparison')
        ax.legend()
        plt.xticks(rotation=45)
        
        # Plot 5: Quality metrics
        ax = axes[1, 0]
        
        quality_metrics = {
            'Overall Quality': result.overall_quality_score,
            'Robustness': result.robustness_score,
            'CV Score': result.cross_validation.cv_score,
            'Ensemble Score': result.ensemble.ensemble_score
        }
        
        bars = ax.bar(quality_metrics.keys(), quality_metrics.values())
        ax.set_ylabel('Score')
        ax.set_title('Quality Metrics')
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45)
        
        # Color bars based on score
        for bar, score in zip(bars, quality_metrics.values()):
            if score > 0.8:
                bar.set_color('green')
            elif score > 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Plot 6: Accuracy assessment
        ax = axes[1, 1]
        
        if result.accuracy_percent is not None:
            accuracy_data = {
                'Accuracy (%)': result.accuracy_percent,
                'Target (%)': 70  # Target accuracy
            }
            
            bars = ax.bar(accuracy_data.keys(), accuracy_data.values())
            bars[0].set_color('green' if result.accuracy_percent > 70 else 'red')
            bars[1].set_color('blue')
            
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Accuracy Assessment')
            ax.set_ylim(0, 100)
        else:
            ax.text(0.5, 0.5, 'No theoretical\nvalue available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Accuracy Assessment')
        
        # Plot 7: Statistical significance
        ax = axes[1, 2]
        
        significance_data = {
            'P-value': result.statistical_significance,
            'Threshold': self.significance_threshold
        }
        
        bars = ax.bar(significance_data.keys(), significance_data.values())
        bars[0].set_color('green' if result.statistical_significance < self.significance_threshold else 'red')
        bars[1].set_color('blue')
        
        ax.set_ylabel('P-value')
        ax.set_title('Statistical Significance')
        ax.set_yscale('log')
        
        # Plot 8: Summary text
        ax = axes[1, 3]
        ax.axis('off')
        
        summary_text = f"Robust {result.exponent_type.upper()} Exponent Extraction\n\n"
        summary_text += f"Final Result: {result.final_exponent:.4f} ± {result.final_error:.4f}\n"
        
        if result.theoretical_exponent:
            summary_text += f"Theoretical: {result.theoretical_exponent:.4f}\n"
            summary_text += f"Accuracy: {result.accuracy_percent:.1f}%\n"
        
        summary_text += f"Statistical Significance: {result.statistical_significance:.2e}\n"
        summary_text += f"Overall Quality: {result.overall_quality_score:.3f}\n"
        summary_text += f"Robustness: {result.robustness_score:.3f}\n"
        summary_text += f"Validation: {'PASSED' if result.validation_passed else 'FAILED'}\n\n"
        
        summary_text += f"Range Optimization: {len(result.range_candidates)} candidates\n"
        summary_text += f"Cross-Validation: {len(result.cross_validation.fold_results)} folds\n"
        summary_text += f"Ensemble Methods: {len(result.ensemble.individual_results)}\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        return fig


def create_robust_critical_exponent_extractor(
    n_range_candidates: int = 20,
    cv_folds: int = 5,
    ensemble_methods: int = 5,
    bootstrap_samples: int = 2000,
    significance_threshold: float = 0.05,
    random_seed: Optional[int] = None
) -> RobustCriticalExponentExtractor:
    """
    Factory function to create a RobustCriticalExponentExtractor.
    
    Args:
        n_range_candidates: Number of fitting ranges to test
        cv_folds: Number of cross-validation folds
        ensemble_methods: Number of ensemble methods
        bootstrap_samples: Number of bootstrap samples
        significance_threshold: P-value threshold for significance
        random_seed: Random seed for reproducibility
        
    Returns:
        Configured RobustCriticalExponentExtractor instance
    """
    return RobustCriticalExponentExtractor(
        n_range_candidates=n_range_candidates,
        cv_folds=cv_folds,
        ensemble_methods=ensemble_methods,
        bootstrap_samples=bootstrap_samples,
        significance_threshold=significance_threshold,
        random_seed=random_seed
    )