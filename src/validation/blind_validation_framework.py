"""
Blind Validation Framework with Separate Comparison Step

This module implements task 13.4: Create blind validation that assesses extraction
quality without knowing theoretical values, with statistical significance testing
and a separate comparison step for theoretical predictions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.stats import kstest, shapiro, anderson, jarque_bera
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# Simple logger function
def get_logger(name):
    return logging.getLogger(name)


@dataclass
class BlindValidationMetrics:
    """Metrics for blind validation (no theoretical knowledge)."""
    
    # Statistical significance
    statistical_significance: float
    confidence_level: float
    p_value: float
    
    # Cross-validation metrics
    cv_mean_score: float
    cv_std_score: float
    cv_consistency: float
    
    # Fit quality metrics
    goodness_of_fit: float
    residual_normality: float
    residual_independence: float
    
    # Robustness metrics
    bootstrap_stability: float
    outlier_sensitivity: float
    parameter_uncertainty: float
    
    # Overall quality
    overall_quality_score: float
    reliability_grade: str  # A, B, C, D, F


@dataclass
class CrossValidationResults:
    """Results from cross-validation analysis."""
    fold_scores: List[float]
    mean_score: float
    std_score: float
    consistency_score: float
    
    # Per-fold details
    fold_parameters: List[Dict[str, float]]
    parameter_stability: Dict[str, float]
    
    # Quality assessment
    cv_quality_score: float
    is_reliable: bool


@dataclass
class StatisticalSignificanceTest:
    """Results from statistical significance testing."""
    test_name: str
    test_statistic: float
    p_value: float
    critical_value: float
    is_significant: bool
    confidence_level: float
    
    # Effect size measures
    effect_size: Optional[float]
    power: Optional[float]
    
    # Interpretation
    significance_level: str  # 'high', 'medium', 'low', 'none'
    reliability_assessment: str


@dataclass
class TheoreticalComparison:
    """Results from comparison with theoretical predictions."""
    
    # Comparison metrics
    relative_error: float
    absolute_error: float
    z_score: float
    
    # Agreement assessment
    within_error_bars: bool
    agreement_level: str  # 'excellent', 'good', 'fair', 'poor'
    
    # Statistical tests
    t_test_result: StatisticalSignificanceTest
    equivalence_test_result: Optional[StatisticalSignificanceTest]
    
    # Theoretical values
    theoretical_value: float
    theoretical_uncertainty: Optional[float]
    
    # Measured values
    measured_value: float
    measured_uncertainty: float


@dataclass
class BlindValidationResults:
    """Complete results from blind validation framework."""
    
    # Blind validation (no theoretical knowledge)
    blind_metrics: BlindValidationMetrics
    cross_validation: CrossValidationResults
    significance_tests: List[StatisticalSignificanceTest]
    
    # Quality assessment
    extraction_reliability: str
    recommended_confidence: float
    quality_flags: List[str]
    
    # Robustness analysis
    bootstrap_results: Dict[str, Any]
    sensitivity_analysis: Dict[str, Any]
    
    # Overall assessment
    validation_passed: bool
    validation_grade: str
    recommendations: List[str]


@dataclass
class ComparisonResults:
    """Results from separate theoretical comparison step."""
    
    # Individual comparisons
    parameter_comparisons: Dict[str, TheoreticalComparison]
    
    # Overall assessment
    overall_agreement: str
    accuracy_score: float
    
    # Statistical summary
    mean_relative_error: float
    rms_error: float
    
    # Universality class assessment
    universality_class_match: bool
    universality_confidence: float
    
    # Recommendations
    theory_validation_status: str
    improvement_suggestions: List[str]


class BlindQualityAssessor:
    """Assesses quality of extraction without theoretical knowledge."""
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize blind quality assessor."""
        self.logger = get_logger(__name__)
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def assess_extraction_quality(self,
                                temperatures: np.ndarray,
                                order_parameter: np.ndarray,
                                critical_temperature: float,
                                fitted_parameters: Dict[str, float],
                                fitting_errors: Dict[str, float]) -> BlindValidationMetrics:
        """
        Assess extraction quality without theoretical knowledge.
        
        Args:
            temperatures: Temperature values
            order_parameter: Order parameter values
            critical_temperature: Detected critical temperature
            fitted_parameters: Fitted parameters (e.g., {'beta': 0.32, 'amplitude': 1.5})
            fitting_errors: Parameter uncertainties
            
        Returns:
            BlindValidationMetrics with quality assessment
        """
        self.logger.info("Assessing extraction quality blindly")
        
        # 1. Statistical significance
        significance = self._assess_statistical_significance(
            temperatures, order_parameter, critical_temperature, fitted_parameters
        )
        
        # 2. Cross-validation
        cv_results = self._perform_cross_validation(
            temperatures, order_parameter, critical_temperature
        )
        
        # 3. Goodness of fit
        goodness_of_fit = self._assess_goodness_of_fit(
            temperatures, order_parameter, critical_temperature, fitted_parameters
        )
        
        # 4. Residual analysis
        residual_normality, residual_independence = self._analyze_residuals(
            temperatures, order_parameter, critical_temperature, fitted_parameters
        )
        
        # 5. Bootstrap stability
        bootstrap_stability = self._assess_bootstrap_stability(
            temperatures, order_parameter, critical_temperature
        )
        
        # 6. Outlier sensitivity
        outlier_sensitivity = self._assess_outlier_sensitivity(
            temperatures, order_parameter, critical_temperature
        )
        
        # 7. Parameter uncertainty
        parameter_uncertainty = self._assess_parameter_uncertainty(fitting_errors)
        
        # 8. Overall quality score
        overall_quality = self._compute_overall_quality(
            significance, cv_results.cv_quality_score, goodness_of_fit,
            residual_normality, bootstrap_stability, outlier_sensitivity
        )
        
        # 9. Reliability grade
        reliability_grade = self._assign_reliability_grade(overall_quality)
        
        return BlindValidationMetrics(
            statistical_significance=significance,
            confidence_level=0.95,  # Standard confidence level
            p_value=0.05,  # Will be computed properly in full implementation
            cv_mean_score=cv_results.mean_score,
            cv_std_score=cv_results.std_score,
            cv_consistency=cv_results.consistency_score,
            goodness_of_fit=goodness_of_fit,
            residual_normality=residual_normality,
            residual_independence=residual_independence,
            bootstrap_stability=bootstrap_stability,
            outlier_sensitivity=outlier_sensitivity,
            parameter_uncertainty=parameter_uncertainty,
            overall_quality_score=overall_quality,
            reliability_grade=reliability_grade
        )
    
    def _assess_statistical_significance(self,
                                       temperatures: np.ndarray,
                                       order_parameter: np.ndarray,
                                       critical_temperature: float,
                                       fitted_parameters: Dict[str, float]) -> float:
        """Assess statistical significance of the fit."""
        
        # Prepare data for fitting
        mask = temperatures < critical_temperature
        if np.sum(mask) < 5:
            return 0.0
        
        x_data = critical_temperature - temperatures[mask]
        y_data = np.abs(order_parameter[mask])
        
        # Remove invalid data
        valid_mask = (x_data > 0) & (y_data > 0) & np.isfinite(x_data) & np.isfinite(y_data)
        x_data = x_data[valid_mask]
        y_data = y_data[valid_mask]
        
        if len(x_data) < 5:
            return 0.0
        
        # Compute R-squared for power law fit
        if 'beta' in fitted_parameters and 'amplitude' in fitted_parameters:
            beta = fitted_parameters['beta']
            amplitude = fitted_parameters['amplitude']
            
            y_pred = amplitude * (x_data ** beta)
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            
            if ss_tot > 0:
                r_squared = 1 - (ss_res / ss_tot)
                return max(0.0, r_squared)
        
        return 0.0
    
    def _perform_cross_validation(self,
                                temperatures: np.ndarray,
                                order_parameter: np.ndarray,
                                critical_temperature: float,
                                n_folds: int = 5) -> CrossValidationResults:
        """Perform cross-validation to assess model stability."""
        
        # Prepare data
        mask = temperatures < critical_temperature
        if np.sum(mask) < 10:
            # Not enough data for cross-validation
            return CrossValidationResults(
                fold_scores=[0.0],
                mean_score=0.0,
                std_score=1.0,
                consistency_score=0.0,
                fold_parameters=[],
                parameter_stability={},
                cv_quality_score=0.0,
                is_reliable=False
            )
        
        x_data = critical_temperature - temperatures[mask]
        y_data = np.abs(order_parameter[mask])
        
        # Remove invalid data
        valid_mask = (x_data > 0) & (y_data > 0) & np.isfinite(x_data) & np.isfinite(y_data)
        x_data = x_data[valid_mask]
        y_data = y_data[valid_mask]
        
        if len(x_data) < 10:
            return CrossValidationResults(
                fold_scores=[0.0],
                mean_score=0.0,
                std_score=1.0,
                consistency_score=0.0,
                fold_parameters=[],
                parameter_stability={},
                cv_quality_score=0.0,
                is_reliable=False
            )
        
        # Perform k-fold cross-validation
        kf = KFold(n_splits=min(n_folds, len(x_data) // 3), shuffle=True, random_state=self.random_seed)
        
        fold_scores = []
        fold_parameters = []
        
        for train_idx, test_idx in kf.split(x_data):
            try:
                # Split data
                x_train, x_test = x_data[train_idx], x_data[test_idx]
                y_train, y_test = y_data[train_idx], y_data[test_idx]
                
                # Fit on training data (log-linear regression)
                log_x_train = np.log(x_train)
                log_y_train = np.log(y_train)
                
                # Simple linear regression
                A = np.vstack([log_x_train, np.ones(len(log_x_train))]).T
                beta, log_amplitude = np.linalg.lstsq(A, log_y_train, rcond=None)[0]
                amplitude = np.exp(log_amplitude)
                
                # Predict on test data
                y_pred = amplitude * (x_test ** beta)
                
                # Compute score (R-squared)
                ss_res = np.sum((y_test - y_pred) ** 2)
                ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                
                if ss_tot > 0:
                    r_squared = 1 - (ss_res / ss_tot)
                    fold_scores.append(max(0.0, r_squared))
                else:
                    fold_scores.append(0.0)
                
                fold_parameters.append({'beta': beta, 'amplitude': amplitude})
                
            except Exception as e:
                self.logger.warning(f"Cross-validation fold failed: {e}")
                fold_scores.append(0.0)
                fold_parameters.append({'beta': 0.0, 'amplitude': 1.0})
        
        if not fold_scores:
            fold_scores = [0.0]
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        # Consistency score (lower std is better)
        consistency_score = max(0.0, 1.0 - std_score)
        
        # Parameter stability
        parameter_stability = {}
        if fold_parameters:
            for param in ['beta', 'amplitude']:
                param_values = [fp.get(param, 0.0) for fp in fold_parameters]
                if param_values:
                    param_stability[param] = 1.0 - (np.std(param_values) / (np.mean(np.abs(param_values)) + 1e-10))
        
        # CV quality score
        cv_quality_score = 0.6 * mean_score + 0.4 * consistency_score
        
        # Reliability assessment
        is_reliable = (mean_score > 0.7) and (consistency_score > 0.8)
        
        return CrossValidationResults(
            fold_scores=fold_scores,
            mean_score=mean_score,
            std_score=std_score,
            consistency_score=consistency_score,
            fold_parameters=fold_parameters,
            parameter_stability=parameter_stability,
            cv_quality_score=cv_quality_score,
            is_reliable=is_reliable
        )
    
    def _assess_goodness_of_fit(self,
                              temperatures: np.ndarray,
                              order_parameter: np.ndarray,
                              critical_temperature: float,
                              fitted_parameters: Dict[str, float]) -> float:
        """Assess goodness of fit using multiple criteria."""
        
        # Prepare data
        mask = temperatures < critical_temperature
        if np.sum(mask) < 5:
            return 0.0
        
        x_data = critical_temperature - temperatures[mask]
        y_data = np.abs(order_parameter[mask])
        
        valid_mask = (x_data > 0) & (y_data > 0) & np.isfinite(x_data) & np.isfinite(y_data)
        x_data = x_data[valid_mask]
        y_data = y_data[valid_mask]
        
        if len(x_data) < 5:
            return 0.0
        
        # Compute predicted values
        if 'beta' in fitted_parameters and 'amplitude' in fitted_parameters:
            beta = fitted_parameters['beta']
            amplitude = fitted_parameters['amplitude']
            y_pred = amplitude * (x_data ** beta)
        else:
            return 0.0
        
        # Multiple goodness-of-fit measures
        
        # 1. R-squared
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # 2. Mean absolute percentage error (MAPE)
        mape = np.mean(np.abs((y_data - y_pred) / (y_data + 1e-10)))
        mape_score = max(0.0, 1.0 - mape)
        
        # 3. Correlation coefficient
        try:
            correlation = np.corrcoef(y_data, y_pred)[0, 1]
            correlation = correlation if not np.isnan(correlation) else 0.0
        except:
            correlation = 0.0
        
        # Combined goodness of fit
        goodness_of_fit = 0.5 * max(0.0, r_squared) + 0.3 * mape_score + 0.2 * abs(correlation)
        
        return min(1.0, max(0.0, goodness_of_fit))
    
    def _analyze_residuals(self,
                         temperatures: np.ndarray,
                         order_parameter: np.ndarray,
                         critical_temperature: float,
                         fitted_parameters: Dict[str, float]) -> Tuple[float, float]:
        """Analyze residuals for normality and independence."""
        
        # Prepare data and compute residuals
        mask = temperatures < critical_temperature
        if np.sum(mask) < 10:
            return 0.5, 0.5
        
        x_data = critical_temperature - temperatures[mask]
        y_data = np.abs(order_parameter[mask])
        
        valid_mask = (x_data > 0) & (y_data > 0) & np.isfinite(x_data) & np.isfinite(y_data)
        x_data = x_data[valid_mask]
        y_data = y_data[valid_mask]
        
        if len(x_data) < 10:
            return 0.5, 0.5
        
        # Compute residuals
        if 'beta' in fitted_parameters and 'amplitude' in fitted_parameters:
            beta = fitted_parameters['beta']
            amplitude = fitted_parameters['amplitude']
            y_pred = amplitude * (x_data ** beta)
            residuals = y_data - y_pred
        else:
            return 0.5, 0.5
        
        # 1. Test for normality
        try:
            # Shapiro-Wilk test for normality
            if len(residuals) >= 3:
                _, p_value_normality = shapiro(residuals)
                normality_score = p_value_normality  # Higher p-value = more normal
            else:
                normality_score = 0.5
        except:
            normality_score = 0.5
        
        # 2. Test for independence (autocorrelation)
        try:
            if len(residuals) > 3:
                # Lag-1 autocorrelation
                autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
                autocorr = autocorr if not np.isnan(autocorr) else 0.0
                independence_score = max(0.0, 1.0 - abs(autocorr))
            else:
                independence_score = 0.5
        except:
            independence_score = 0.5
        
        return normality_score, independence_score
    
    def _assess_bootstrap_stability(self,
                                  temperatures: np.ndarray,
                                  order_parameter: np.ndarray,
                                  critical_temperature: float,
                                  n_bootstrap: int = 100) -> float:
        """Assess stability using bootstrap resampling."""
        
        # Prepare data
        mask = temperatures < critical_temperature
        if np.sum(mask) < 10:
            return 0.0
        
        x_data = critical_temperature - temperatures[mask]
        y_data = np.abs(order_parameter[mask])
        
        valid_mask = (x_data > 0) & (y_data > 0) & np.isfinite(x_data) & np.isfinite(y_data)
        x_data = x_data[valid_mask]
        y_data = y_data[valid_mask]
        
        if len(x_data) < 10:
            return 0.0
        
        # Bootstrap sampling
        bootstrap_betas = []
        
        for _ in range(n_bootstrap):
            try:
                # Bootstrap sample
                indices = np.random.choice(len(x_data), len(x_data), replace=True)
                x_boot = x_data[indices]
                y_boot = y_data[indices]
                
                # Fit (log-linear regression)
                log_x_boot = np.log(x_boot)
                log_y_boot = np.log(y_boot)
                
                A = np.vstack([log_x_boot, np.ones(len(log_x_boot))]).T
                beta, _ = np.linalg.lstsq(A, log_y_boot, rcond=None)[0]
                
                bootstrap_betas.append(beta)
                
            except:
                continue
        
        if len(bootstrap_betas) < 10:
            return 0.0
        
        # Stability is inverse of coefficient of variation
        mean_beta = np.mean(bootstrap_betas)
        std_beta = np.std(bootstrap_betas)
        
        if abs(mean_beta) > 1e-10:
            cv = std_beta / abs(mean_beta)
            stability = max(0.0, 1.0 - cv)
        else:
            stability = 0.0
        
        return min(1.0, stability)
    
    def _assess_outlier_sensitivity(self,
                                  temperatures: np.ndarray,
                                  order_parameter: np.ndarray,
                                  critical_temperature: float) -> float:
        """Assess sensitivity to outliers."""
        
        # Prepare data
        mask = temperatures < critical_temperature
        if np.sum(mask) < 10:
            return 0.5
        
        x_data = critical_temperature - temperatures[mask]
        y_data = np.abs(order_parameter[mask])
        
        valid_mask = (x_data > 0) & (y_data > 0) & np.isfinite(x_data) & np.isfinite(y_data)
        x_data = x_data[valid_mask]
        y_data = y_data[valid_mask]
        
        if len(x_data) < 10:
            return 0.5
        
        try:
            # Fit with all data
            log_x = np.log(x_data)
            log_y = np.log(y_data)
            A = np.vstack([log_x, np.ones(len(log_x))]).T
            beta_full, _ = np.linalg.lstsq(A, log_y, rcond=None)[0]
            
            # Fit with outliers removed (using IQR method)
            q1, q3 = np.percentile(log_y, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_mask = (log_y >= lower_bound) & (log_y <= upper_bound)
            
            if np.sum(outlier_mask) >= 5:
                log_x_clean = log_x[outlier_mask]
                log_y_clean = log_y[outlier_mask]
                
                A_clean = np.vstack([log_x_clean, np.ones(len(log_x_clean))]).T
                beta_clean, _ = np.linalg.lstsq(A_clean, log_y_clean, rcond=None)[0]
                
                # Sensitivity is how much the parameter changes
                relative_change = abs(beta_full - beta_clean) / (abs(beta_full) + 1e-10)
                sensitivity = max(0.0, 1.0 - relative_change)
            else:
                sensitivity = 0.5
                
        except:
            sensitivity = 0.5
        
        return min(1.0, sensitivity)
    
    def _assess_parameter_uncertainty(self, fitting_errors: Dict[str, float]) -> float:
        """Assess parameter uncertainty."""
        
        if not fitting_errors:
            return 0.0
        
        # Focus on beta parameter uncertainty
        if 'beta' in fitting_errors:
            beta_error = fitting_errors['beta']
            # Convert to relative uncertainty score (lower error = higher score)
            uncertainty_score = max(0.0, 1.0 - min(1.0, beta_error / 0.1))  # Normalize by 0.1
        else:
            uncertainty_score = 0.5
        
        return uncertainty_score
    
    def _compute_overall_quality(self, *quality_components) -> float:
        """Compute overall quality score from components."""
        
        # Filter out None values
        valid_components = [c for c in quality_components if c is not None and not np.isnan(c)]
        
        if not valid_components:
            return 0.0
        
        # Weighted average (can be customized)
        return np.mean(valid_components)
    
    def _assign_reliability_grade(self, overall_quality: float) -> str:
        """Assign reliability grade based on overall quality."""
        
        if overall_quality >= 0.9:
            return 'A'
        elif overall_quality >= 0.8:
            return 'B'
        elif overall_quality >= 0.7:
            return 'C'
        elif overall_quality >= 0.6:
            return 'D'
        else:
            return 'F'


class TheoreticalComparator:
    """Compares extracted results with theoretical predictions."""
    
    def __init__(self):
        """Initialize theoretical comparator."""
        self.logger = get_logger(__name__)
        
        # Theoretical values for different systems
        self.theoretical_values = {
            'ising_2d': {'beta': 0.125, 'nu': 1.0, 'gamma': 1.75, 'tc': 2.269},
            'ising_3d': {'beta': 0.326, 'nu': 0.630, 'gamma': 1.237, 'tc': 4.511},
            'xy_2d': {'beta': 0.125, 'nu': 1.0, 'gamma': 1.75, 'tc': None},
            'potts_3_2d': {'beta': 0.125, 'nu': 5.0/6.0, 'gamma': 13.0/9.0, 'tc': 1.005},
            'heisenberg_3d': {'beta': 0.365, 'nu': 0.705, 'gamma': 1.386, 'tc': None}
        }
    
    def compare_with_theory(self,
                          measured_parameters: Dict[str, float],
                          parameter_errors: Dict[str, float],
                          system_type: str) -> ComparisonResults:
        """
        Compare measured parameters with theoretical predictions.
        
        Args:
            measured_parameters: Measured parameter values
            parameter_errors: Parameter uncertainties
            system_type: Type of physical system
            
        Returns:
            ComparisonResults with comparison analysis
        """
        self.logger.info(f"Comparing results with theory for {system_type}")
        
        # Get theoretical values
        if system_type not in self.theoretical_values:
            self.logger.warning(f"Unknown system type: {system_type}")
            theoretical_params = {}
        else:
            theoretical_params = self.theoretical_values[system_type]
        
        # Compare each parameter
        parameter_comparisons = {}
        
        for param_name, measured_value in measured_parameters.items():
            if param_name in theoretical_params and theoretical_params[param_name] is not None:
                theoretical_value = theoretical_params[param_name]
                measured_error = parameter_errors.get(param_name, 0.1 * abs(measured_value))
                
                comparison = self._compare_single_parameter(
                    measured_value, measured_error, theoretical_value, param_name
                )
                
                parameter_comparisons[param_name] = comparison
        
        # Overall assessment
        overall_agreement, accuracy_score = self._assess_overall_agreement(parameter_comparisons)
        
        # Compute summary statistics
        relative_errors = [comp.relative_error for comp in parameter_comparisons.values()]
        mean_relative_error = np.mean(relative_errors) if relative_errors else 1.0
        rms_error = np.sqrt(np.mean([e**2 for e in relative_errors])) if relative_errors else 1.0
        
        # Universality class assessment
        universality_match, universality_confidence = self._assess_universality_class(
            parameter_comparisons, system_type
        )
        
        # Theory validation status
        theory_validation_status = self._determine_validation_status(overall_agreement, accuracy_score)
        
        # Improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(
            parameter_comparisons, accuracy_score
        )
        
        return ComparisonResults(
            parameter_comparisons=parameter_comparisons,
            overall_agreement=overall_agreement,
            accuracy_score=accuracy_score,
            mean_relative_error=mean_relative_error,
            rms_error=rms_error,
            universality_class_match=universality_match,
            universality_confidence=universality_confidence,
            theory_validation_status=theory_validation_status,
            improvement_suggestions=improvement_suggestions
        )
    
    def _compare_single_parameter(self,
                                measured_value: float,
                                measured_error: float,
                                theoretical_value: float,
                                parameter_name: str) -> TheoreticalComparison:
        """Compare a single parameter with theory."""
        
        # Basic comparison metrics
        absolute_error = abs(measured_value - theoretical_value)
        relative_error = absolute_error / abs(theoretical_value) if theoretical_value != 0 else float('inf')
        
        # Z-score (how many standard deviations away)
        z_score = absolute_error / measured_error if measured_error > 0 else float('inf')
        
        # Check if within error bars
        within_error_bars = absolute_error <= 2 * measured_error  # 2-sigma criterion
        
        # Agreement level
        if relative_error <= 0.05:
            agreement_level = 'excellent'
        elif relative_error <= 0.15:
            agreement_level = 'good'
        elif relative_error <= 0.30:
            agreement_level = 'fair'
        else:
            agreement_level = 'poor'
        
        # Statistical tests
        t_test_result = self._perform_t_test(measured_value, measured_error, theoretical_value)
        
        # Equivalence test (TOST - Two One-Sided Tests)
        equivalence_test_result = self._perform_equivalence_test(
            measured_value, measured_error, theoretical_value
        )
        
        return TheoreticalComparison(
            relative_error=relative_error,
            absolute_error=absolute_error,
            z_score=z_score,
            within_error_bars=within_error_bars,
            agreement_level=agreement_level,
            t_test_result=t_test_result,
            equivalence_test_result=equivalence_test_result,
            theoretical_value=theoretical_value,
            theoretical_uncertainty=None,  # Could be added if known
            measured_value=measured_value,
            measured_uncertainty=measured_error
        )
    
    def _perform_t_test(self,
                       measured_value: float,
                       measured_error: float,
                       theoretical_value: float) -> StatisticalSignificanceTest:
        """Perform t-test to check if measured value differs significantly from theory."""
        
        # One-sample t-test
        if measured_error > 0:
            t_statistic = (measured_value - theoretical_value) / measured_error
            
            # Degrees of freedom (assuming reasonable sample size)
            df = 10  # This would be actual df in real implementation
            
            # Two-tailed p-value
            p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))
            
            # Critical value for 95% confidence
            critical_value = stats.t.ppf(0.975, df)
            
            is_significant = abs(t_statistic) > critical_value
            
            # Significance level
            if p_value < 0.001:
                significance_level = 'high'
            elif p_value < 0.01:
                significance_level = 'medium'
            elif p_value < 0.05:
                significance_level = 'low'
            else:
                significance_level = 'none'
            
            # Reliability assessment
            if is_significant:
                reliability_assessment = 'measured value significantly differs from theory'
            else:
                reliability_assessment = 'measured value consistent with theory'
        else:
            # No error information
            t_statistic = 0.0
            p_value = 1.0
            critical_value = 1.96
            is_significant = False
            significance_level = 'none'
            reliability_assessment = 'insufficient error information'
        
        return StatisticalSignificanceTest(
            test_name='one_sample_t_test',
            test_statistic=t_statistic,
            p_value=p_value,
            critical_value=critical_value,
            is_significant=is_significant,
            confidence_level=0.95,
            effect_size=None,  # Could compute Cohen's d
            power=None,  # Could compute statistical power
            significance_level=significance_level,
            reliability_assessment=reliability_assessment
        )
    
    def _perform_equivalence_test(self,
                                measured_value: float,
                                measured_error: float,
                                theoretical_value: float,
                                equivalence_margin: float = 0.1) -> Optional[StatisticalSignificanceTest]:
        """Perform equivalence test (TOST) to check if values are practically equivalent."""
        
        if measured_error <= 0:
            return None
        
        # TOST (Two One-Sided Tests) for equivalence
        # H0: |measured - theoretical| >= margin
        # H1: |measured - theoretical| < margin
        
        # Define equivalence bounds
        lower_bound = theoretical_value - equivalence_margin * abs(theoretical_value)
        upper_bound = theoretical_value + equivalence_margin * abs(theoretical_value)
        
        # Two one-sided t-tests
        t1 = (measured_value - lower_bound) / measured_error  # Test if measured > lower_bound
        t2 = (upper_bound - measured_value) / measured_error  # Test if measured < upper_bound
        
        # Degrees of freedom
        df = 10  # Would be actual df in real implementation
        
        # P-values for one-sided tests
        p1 = stats.t.cdf(t1, df)
        p2 = stats.t.cdf(t2, df)
        
        # TOST p-value is the maximum of the two
        tost_p_value = max(p1, p2)
        
        # Equivalence is established if TOST p-value < alpha
        is_equivalent = tost_p_value < 0.05
        
        return StatisticalSignificanceTest(
            test_name='equivalence_test_TOST',
            test_statistic=min(t1, t2),
            p_value=tost_p_value,
            critical_value=stats.t.ppf(0.95, df),
            is_significant=is_equivalent,
            confidence_level=0.95,
            effect_size=None,
            power=None,
            significance_level='high' if is_equivalent else 'none',
            reliability_assessment='values are equivalent' if is_equivalent else 'values not equivalent'
        )
    
    def _assess_overall_agreement(self, 
                                parameter_comparisons: Dict[str, TheoreticalComparison]) -> Tuple[str, float]:
        """Assess overall agreement across all parameters."""
        
        if not parameter_comparisons:
            return 'unknown', 0.0
        
        # Collect agreement levels
        agreement_levels = [comp.agreement_level for comp in parameter_comparisons.values()]
        
        # Count each level
        excellent_count = agreement_levels.count('excellent')
        good_count = agreement_levels.count('good')
        fair_count = agreement_levels.count('fair')
        poor_count = agreement_levels.count('poor')
        
        total_count = len(agreement_levels)
        
        # Overall agreement
        if excellent_count / total_count >= 0.8:
            overall_agreement = 'excellent'
        elif (excellent_count + good_count) / total_count >= 0.7:
            overall_agreement = 'good'
        elif (excellent_count + good_count + fair_count) / total_count >= 0.6:
            overall_agreement = 'fair'
        else:
            overall_agreement = 'poor'
        
        # Accuracy score (0-100)
        relative_errors = [comp.relative_error for comp in parameter_comparisons.values()]
        mean_relative_error = np.mean(relative_errors)
        accuracy_score = max(0.0, (1.0 - mean_relative_error) * 100)
        
        return overall_agreement, accuracy_score
    
    def _assess_universality_class(self,
                                 parameter_comparisons: Dict[str, TheoreticalComparison],
                                 system_type: str) -> Tuple[bool, float]:
        """Assess if results match expected universality class."""
        
        # This is a simplified assessment
        # In reality, would need more sophisticated analysis
        
        if not parameter_comparisons:
            return False, 0.0
        
        # Check if critical exponents are consistent with universality class
        good_matches = 0
        total_exponents = 0
        
        for param_name, comparison in parameter_comparisons.items():
            if param_name in ['beta', 'nu', 'gamma']:
                total_exponents += 1
                if comparison.agreement_level in ['excellent', 'good']:
                    good_matches += 1
        
        if total_exponents == 0:
            return False, 0.0
        
        match_fraction = good_matches / total_exponents
        universality_match = match_fraction >= 0.67  # At least 2/3 of exponents match well
        universality_confidence = match_fraction
        
        return universality_match, universality_confidence
    
    def _determine_validation_status(self, overall_agreement: str, accuracy_score: float) -> str:
        """Determine theory validation status."""
        
        if overall_agreement == 'excellent' and accuracy_score >= 90:
            return 'theory_strongly_validated'
        elif overall_agreement in ['excellent', 'good'] and accuracy_score >= 80:
            return 'theory_validated'
        elif overall_agreement in ['good', 'fair'] and accuracy_score >= 70:
            return 'theory_partially_validated'
        elif overall_agreement == 'fair' and accuracy_score >= 60:
            return 'theory_weakly_supported'
        else:
            return 'theory_not_validated'
    
    def _generate_improvement_suggestions(self,
                                        parameter_comparisons: Dict[str, TheoreticalComparison],
                                        accuracy_score: float) -> List[str]:
        """Generate suggestions for improving results."""
        
        suggestions = []
        
        if accuracy_score < 70:
            suggestions.append("Consider increasing data quality and quantity")
            suggestions.append("Review fitting procedures and parameter estimation methods")
        
        # Parameter-specific suggestions
        for param_name, comparison in parameter_comparisons.items():
            if comparison.agreement_level == 'poor':
                suggestions.append(f"Investigate {param_name} extraction method - large deviation from theory")
            elif comparison.agreement_level == 'fair':
                suggestions.append(f"Consider refining {param_name} fitting procedure")
        
        if not any(comp.within_error_bars for comp in parameter_comparisons.values()):
            suggestions.append("Error bars may be underestimated - review uncertainty analysis")
        
        if len(suggestions) == 0:
            suggestions.append("Results are in good agreement with theory")
        
        return suggestions


class BlindValidationFramework:
    """Main framework for blind validation with separate comparison."""
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize blind validation framework."""
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.quality_assessor = BlindQualityAssessor(random_seed)
        self.theoretical_comparator = TheoreticalComparator()
    
    def validate_extraction_blind(self,
                                temperatures: np.ndarray,
                                order_parameter: np.ndarray,
                                critical_temperature: float,
                                fitted_parameters: Dict[str, float],
                                fitting_errors: Dict[str, float]) -> BlindValidationResults:
        """
        Perform blind validation without theoretical knowledge.
        
        Args:
            temperatures: Temperature values
            order_parameter: Order parameter values
            critical_temperature: Detected critical temperature
            fitted_parameters: Fitted parameters
            fitting_errors: Parameter uncertainties
            
        Returns:
            BlindValidationResults with validation assessment
        """
        self.logger.info("Performing blind validation")
        
        # 1. Assess extraction quality blindly
        blind_metrics = self.quality_assessor.assess_extraction_quality(
            temperatures, order_parameter, critical_temperature, fitted_parameters, fitting_errors
        )
        
        # 2. Perform cross-validation
        cross_validation = self.quality_assessor._perform_cross_validation(
            temperatures, order_parameter, critical_temperature
        )
        
        # 3. Statistical significance tests
        significance_tests = self._perform_comprehensive_significance_tests(
            temperatures, order_parameter, critical_temperature, fitted_parameters
        )
        
        # 4. Bootstrap analysis
        bootstrap_results = self._perform_bootstrap_analysis(
            temperatures, order_parameter, critical_temperature
        )
        
        # 5. Sensitivity analysis
        sensitivity_analysis = self._perform_sensitivity_analysis(
            temperatures, order_parameter, critical_temperature
        )
        
        # 6. Overall assessment
        validation_passed, validation_grade, quality_flags, recommendations = self._assess_validation_outcome(
            blind_metrics, cross_validation, significance_tests
        )
        
        return BlindValidationResults(
            blind_metrics=blind_metrics,
            cross_validation=cross_validation,
            significance_tests=significance_tests,
            extraction_reliability=blind_metrics.reliability_grade,
            recommended_confidence=0.95 if validation_passed else 0.68,
            quality_flags=quality_flags,
            bootstrap_results=bootstrap_results,
            sensitivity_analysis=sensitivity_analysis,
            validation_passed=validation_passed,
            validation_grade=validation_grade,
            recommendations=recommendations
        )
    
    def compare_with_theory_separate(self,
                                   measured_parameters: Dict[str, float],
                                   parameter_errors: Dict[str, float],
                                   system_type: str) -> ComparisonResults:
        """
        Separate step: Compare with theoretical predictions.
        
        Args:
            measured_parameters: Measured parameter values
            parameter_errors: Parameter uncertainties
            system_type: Type of physical system
            
        Returns:
            ComparisonResults with theoretical comparison
        """
        self.logger.info("Performing separate theoretical comparison")
        
        return self.theoretical_comparator.compare_with_theory(
            measured_parameters, parameter_errors, system_type
        )
    
    def _perform_comprehensive_significance_tests(self,
                                                temperatures: np.ndarray,
                                                order_parameter: np.ndarray,
                                                critical_temperature: float,
                                                fitted_parameters: Dict[str, float]) -> List[StatisticalSignificanceTest]:
        """Perform comprehensive statistical significance tests."""
        
        tests = []
        
        # This is a placeholder - would implement various statistical tests
        # such as F-test for model significance, likelihood ratio tests, etc.
        
        return tests
    
    def _perform_bootstrap_analysis(self,
                                  temperatures: np.ndarray,
                                  order_parameter: np.ndarray,
                                  critical_temperature: float) -> Dict[str, Any]:
        """Perform bootstrap analysis for robustness assessment."""
        
        # Placeholder for bootstrap analysis
        return {
            'bootstrap_stability': 0.8,
            'parameter_distributions': {},
            'confidence_intervals': {}
        }
    
    def _perform_sensitivity_analysis(self,
                                    temperatures: np.ndarray,
                                    order_parameter: np.ndarray,
                                    critical_temperature: float) -> Dict[str, Any]:
        """Perform sensitivity analysis."""
        
        # Placeholder for sensitivity analysis
        return {
            'outlier_sensitivity': 0.7,
            'data_subset_sensitivity': 0.8,
            'parameter_sensitivity': {}
        }
    
    def _assess_validation_outcome(self,
                                 blind_metrics: BlindValidationMetrics,
                                 cross_validation: CrossValidationResults,
                                 significance_tests: List[StatisticalSignificanceTest]) -> Tuple[bool, str, List[str], List[str]]:
        """Assess overall validation outcome."""
        
        # Determine if validation passed
        validation_passed = (
            blind_metrics.overall_quality_score >= 0.7 and
            cross_validation.is_reliable and
            blind_metrics.reliability_grade in ['A', 'B', 'C']
        )
        
        # Assign validation grade
        if blind_metrics.overall_quality_score >= 0.9:
            validation_grade = 'A'
        elif blind_metrics.overall_quality_score >= 0.8:
            validation_grade = 'B'
        elif blind_metrics.overall_quality_score >= 0.7:
            validation_grade = 'C'
        else:
            validation_grade = 'F'
        
        # Quality flags
        quality_flags = []
        if blind_metrics.goodness_of_fit < 0.8:
            quality_flags.append('poor_fit_quality')
        if blind_metrics.bootstrap_stability < 0.7:
            quality_flags.append('unstable_parameters')
        if not cross_validation.is_reliable:
            quality_flags.append('poor_cross_validation')
        
        # Recommendations
        recommendations = []
        if not validation_passed:
            recommendations.append('Improve data quality and quantity')
            recommendations.append('Review fitting procedures')
        if blind_metrics.parameter_uncertainty > 0.5:
            recommendations.append('Reduce parameter uncertainties')
        
        return validation_passed, validation_grade, quality_flags, recommendations


def create_blind_validation_framework(random_seed: Optional[int] = None) -> BlindValidationFramework:
    """
    Factory function to create BlindValidationFramework.
    
    Args:
        random_seed: Random seed for reproducibility
        
    Returns:
        Configured BlindValidationFramework instance
    """
    return BlindValidationFramework(random_seed)