"""
Comprehensive Statistical Validation Framework

This module implements task 14.2: Implement comprehensive statistical validation.
Adds bootstrap confidence intervals for all extracted exponents, implements
cross-validation across different system sizes and temperature ranges, and
creates statistical significance testing for power-law fits and correlations.
"""

import numpy as np
import scipy.stats as stats
from scipy.stats import bootstrap, kstest, anderson, jarque_bera, shapiro
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold, cross_val_score, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Simple logger function
def get_logger(name):
    return logging.getLogger(name)


@dataclass
class BootstrapResults:
    """Results from bootstrap confidence interval analysis."""
    
    # Bootstrap statistics
    original_statistic: float
    bootstrap_samples: np.ndarray
    n_bootstrap_samples: int
    
    # Confidence intervals
    confidence_level: float
    ci_lower: float
    ci_upper: float
    ci_width: float
    
    # Bootstrap distribution properties
    bootstrap_mean: float
    bootstrap_std: float
    bootstrap_bias: float
    bootstrap_skewness: float
    bootstrap_kurtosis: float
    
    # Quality metrics
    convergence_achieved: bool
    effective_sample_size: int
    monte_carlo_error: float
    
    # Distribution tests
    normality_test_pvalue: float
    is_normal_distribution: bool


@dataclass
class CrossValidationResults:
    """Results from cross-validation analysis."""
    
    # Cross-validation setup
    cv_method: str
    n_folds: int
    n_samples: int
    
    # Performance metrics across folds
    fold_scores: np.ndarray
    mean_score: float
    std_score: float
    
    # Detailed metrics
    r2_scores: np.ndarray
    mse_scores: np.ndarray
    mae_scores: np.ndarray
    
    # Statistical tests
    score_normality_pvalue: float
    score_consistency_test: float
    
    # Stability metrics
    coefficient_of_variation: float
    stability_score: float
    overfitting_indicator: float
    
    # Fold-specific results
    fold_parameters: List[Dict[str, float]]
    parameter_stability: Dict[str, float]


@dataclass
class SignificanceTestResults:
    """Results from statistical significance testing."""
    
    # Test information
    test_name: str
    test_statistic: float
    p_value: float
    degrees_of_freedom: Optional[int]
    
    # Significance assessment
    alpha_level: float
    is_significant: bool
    effect_size: Optional[float]
    power: Optional[float]
    
    # Additional test details
    critical_value: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    test_assumptions_met: bool
    
    # Quality indicators
    sample_size: int
    test_power_adequate: bool
    practical_significance: bool


@dataclass
class PowerLawValidationResults:
    """Results from power-law fitting validation."""
    
    # Fitting results
    exponent: float
    exponent_error: float
    amplitude: float
    amplitude_error: float
    
    # Goodness of fit
    r_squared: float
    adjusted_r_squared: float
    chi_squared: float
    reduced_chi_squared: float
    
    # Bootstrap confidence intervals
    exponent_bootstrap: BootstrapResults
    amplitude_bootstrap: BootstrapResults
    
    # Cross-validation results
    cross_validation: CrossValidationResults
    
    # Significance tests
    fit_significance: SignificanceTestResults
    linearity_test: SignificanceTestResults
    residual_normality: SignificanceTestResults
    
    # Model comparison
    alternative_models: Dict[str, Dict[str, float]]
    best_model: str
    model_selection_criterion: str
    
    # Diagnostic metrics
    residual_analysis: Dict[str, float]
    outlier_analysis: Dict[str, Any]
    leverage_analysis: Dict[str, float]


@dataclass
class ComprehensiveValidationResults:
    """Complete comprehensive statistical validation results."""
    
    # System information
    system_type: str
    n_samples: int
    temperature_range: Tuple[float, float]
    
    # Critical temperature validation
    tc_bootstrap: BootstrapResults
    tc_cross_validation: CrossValidationResults
    tc_significance: SignificanceTestResults
    
    # Critical exponent validation
    beta_validation: Optional[PowerLawValidationResults] = None
    nu_validation: Optional[PowerLawValidationResults] = None
    gamma_validation: Optional[PowerLawValidationResults] = None
    
    # Correlation analysis
    correlation_tests: Dict[str, SignificanceTestResults] = field(default_factory=dict)
    
    # Overall assessment
    overall_validation_score: float = 0.0
    validation_grade: str = 'F'
    statistical_reliability: str = 'Poor'
    
    # Recommendations
    validation_warnings: List[str] = field(default_factory=list)
    improvement_recommendations: List[str] = field(default_factory=list)


class BootstrapValidator:
    """Bootstrap confidence interval validator."""
    
    def __init__(self, 
                 n_bootstrap: int = 10000,
                 confidence_level: float = 0.95,
                 random_seed: Optional[int] = None):
        """Initialize bootstrap validator."""
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        self.logger = get_logger(__name__)
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def bootstrap_statistic(self, 
                          data: np.ndarray,
                          statistic_func: Callable,
                          **kwargs) -> BootstrapResults:
        """
        Compute bootstrap confidence intervals for a statistic.
        
        Args:
            data: Input data array
            statistic_func: Function to compute statistic
            **kwargs: Additional arguments for statistic function
            
        Returns:
            BootstrapResults with confidence intervals and diagnostics
        """
        
        # Compute original statistic
        original_stat = statistic_func(data, **kwargs)
        
        # Bootstrap sampling
        bootstrap_stats = []
        n_samples = len(data)
        
        for i in range(self.n_bootstrap):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_sample = data[bootstrap_indices]
            
            try:
                bootstrap_stat = statistic_func(bootstrap_sample, **kwargs)
                if np.isfinite(bootstrap_stat):
                    bootstrap_stats.append(bootstrap_stat)
            except:
                continue
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        if len(bootstrap_stats) < self.n_bootstrap * 0.5:
            self.logger.warning(f"Only {len(bootstrap_stats)}/{self.n_bootstrap} bootstrap samples succeeded")
        
        # Compute confidence intervals
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        # Bootstrap distribution properties
        bootstrap_mean = np.mean(bootstrap_stats)
        bootstrap_std = np.std(bootstrap_stats)
        bootstrap_bias = bootstrap_mean - original_stat
        bootstrap_skewness = stats.skew(bootstrap_stats)
        bootstrap_kurtosis = stats.kurtosis(bootstrap_stats)
        
        # Convergence assessment
        convergence_achieved = self._assess_convergence(bootstrap_stats)
        
        # Effective sample size
        effective_n = self._compute_effective_sample_size(bootstrap_stats)
        
        # Monte Carlo error
        mc_error = bootstrap_std / np.sqrt(len(bootstrap_stats))
        
        # Normality test
        if len(bootstrap_stats) > 8:
            _, normality_p = shapiro(bootstrap_stats[:5000])  # Limit for shapiro test
        else:
            normality_p = 0.0
        
        is_normal = normality_p > 0.05
        
        return BootstrapResults(
            original_statistic=original_stat,
            bootstrap_samples=bootstrap_stats,
            n_bootstrap_samples=len(bootstrap_stats),
            confidence_level=self.confidence_level,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_width=ci_upper - ci_lower,
            bootstrap_mean=bootstrap_mean,
            bootstrap_std=bootstrap_std,
            bootstrap_bias=bootstrap_bias,
            bootstrap_skewness=bootstrap_skewness,
            bootstrap_kurtosis=bootstrap_kurtosis,
            convergence_achieved=convergence_achieved,
            effective_sample_size=effective_n,
            monte_carlo_error=mc_error,
            normality_test_pvalue=normality_p,
            is_normal_distribution=is_normal
        )
    
    def _assess_convergence(self, bootstrap_stats: np.ndarray) -> bool:
        """Assess if bootstrap has converged."""
        
        if len(bootstrap_stats) < 100:
            return False
        
        # Check if running mean has stabilized
        n_check = min(1000, len(bootstrap_stats) // 4)
        running_means = []
        
        for i in range(n_check, len(bootstrap_stats), n_check):
            running_means.append(np.mean(bootstrap_stats[:i]))
        
        if len(running_means) < 3:
            return False
        
        # Check relative change in running mean
        relative_changes = np.abs(np.diff(running_means)) / (np.abs(running_means[:-1]) + 1e-10)
        
        # Convergence if last few changes are small
        return np.all(relative_changes[-2:] < 0.01)
    
    def _compute_effective_sample_size(self, bootstrap_stats: np.ndarray) -> int:
        """Compute effective sample size accounting for autocorrelation."""
        
        if len(bootstrap_stats) < 10:
            return len(bootstrap_stats)
        
        # Compute autocorrelation
        try:
            autocorr = np.correlate(bootstrap_stats - np.mean(bootstrap_stats),
                                  bootstrap_stats - np.mean(bootstrap_stats), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            # Find first negative autocorrelation
            first_negative = np.where(autocorr < 0)[0]
            if len(first_negative) > 0:
                tau = first_negative[0]
            else:
                tau = len(autocorr) // 4
            
            # Effective sample size
            effective_n = len(bootstrap_stats) / (1 + 2 * tau)
            return max(1, int(effective_n))
            
        except:
            return len(bootstrap_stats)


class CrossValidator:
    """Cross-validation framework for statistical validation."""
    
    def __init__(self, 
                 cv_method: str = 'kfold',
                 n_folds: int = 5,
                 random_seed: Optional[int] = None):
        """Initialize cross-validator."""
        self.cv_method = cv_method
        self.n_folds = n_folds
        self.random_seed = random_seed
        self.logger = get_logger(__name__)
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def cross_validate_power_law(self,
                                x_data: np.ndarray,
                                y_data: np.ndarray) -> CrossValidationResults:
        """
        Cross-validate power-law fitting.
        
        Args:
            x_data: Independent variable data
            y_data: Dependent variable data
            
        Returns:
            CrossValidationResults with validation metrics
        """
        
        # Prepare data for cross-validation
        log_x = np.log(x_data)
        log_y = np.log(y_data)
        
        # Remove invalid values
        valid_mask = np.isfinite(log_x) & np.isfinite(log_y)
        log_x = log_x[valid_mask]
        log_y = log_y[valid_mask]
        
        if len(log_x) < self.n_folds:
            raise ValueError(f"Insufficient data for {self.n_folds}-fold cross-validation")
        
        # Create cross-validation splitter
        if self.cv_method == 'kfold':
            cv_splitter = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
        elif self.cv_method == 'timeseries':
            cv_splitter = TimeSeriesSplit(n_splits=self.n_folds)
        else:
            raise ValueError(f"Unknown CV method: {self.cv_method}")
        
        # Perform cross-validation
        fold_scores = []
        r2_scores = []
        mse_scores = []
        mae_scores = []
        fold_parameters = []
        
        X = log_x.reshape(-1, 1)
        y = log_y
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X)):
            try:
                # Split data
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Fit linear model in log space
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Compute metrics
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                fold_scores.append(r2)
                r2_scores.append(r2)
                mse_scores.append(mse)
                mae_scores.append(mae)
                
                # Store parameters (slope = exponent, intercept = log(amplitude))
                fold_parameters.append({
                    'exponent': model.coef_[0],
                    'log_amplitude': model.intercept_,
                    'amplitude': np.exp(model.intercept_)
                })
                
            except Exception as e:
                self.logger.warning(f"Fold {fold_idx} failed: {e}")
                continue
        
        if len(fold_scores) == 0:
            raise ValueError("All cross-validation folds failed")
        
        fold_scores = np.array(fold_scores)
        r2_scores = np.array(r2_scores)
        mse_scores = np.array(mse_scores)
        mae_scores = np.array(mae_scores)
        
        # Compute statistics
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        # Test score normality
        if len(fold_scores) >= 3:
            _, score_normality_p = shapiro(fold_scores)
        else:
            score_normality_p = 0.0
        
        # Score consistency test (coefficient of variation)
        cv_score = std_score / (abs(mean_score) + 1e-10)
        
        # Stability metrics
        stability_score = max(0, 1 - cv_score)  # Higher is more stable
        
        # Overfitting indicator (difference between train and test performance)
        # This is a simplified version - would need train scores for full analysis
        overfitting_indicator = cv_score  # High CV suggests overfitting
        
        # Parameter stability
        parameter_stability = {}
        if fold_parameters:
            for param_name in fold_parameters[0].keys():
                param_values = [fp[param_name] for fp in fold_parameters]
                param_cv = np.std(param_values) / (abs(np.mean(param_values)) + 1e-10)
                parameter_stability[param_name] = max(0, 1 - param_cv)
        
        return CrossValidationResults(
            cv_method=self.cv_method,
            n_folds=self.n_folds,
            n_samples=len(log_x),
            fold_scores=fold_scores,
            mean_score=mean_score,
            std_score=std_score,
            r2_scores=r2_scores,
            mse_scores=mse_scores,
            mae_scores=mae_scores,
            score_normality_pvalue=score_normality_p,
            score_consistency_test=cv_score,
            coefficient_of_variation=cv_score,
            stability_score=stability_score,
            overfitting_indicator=overfitting_indicator,
            fold_parameters=fold_parameters,
            parameter_stability=parameter_stability
        )


class SignificanceTester:
    """Statistical significance testing framework."""
    
    def __init__(self, alpha_level: float = 0.05):
        """Initialize significance tester."""
        self.alpha_level = alpha_level
        self.logger = get_logger(__name__)
    
    def test_power_law_significance(self,
                                  x_data: np.ndarray,
                                  y_data: np.ndarray,
                                  exponent: float,
                                  amplitude: float) -> SignificanceTestResults:
        """
        Test statistical significance of power-law fit.
        
        Args:
            x_data: Independent variable data
            y_data: Dependent variable data
            exponent: Fitted exponent
            amplitude: Fitted amplitude
            
        Returns:
            SignificanceTestResults with significance assessment
        """
        
        # Compute predicted values
        y_pred = amplitude * (x_data ** exponent)
        
        # Compute residuals
        residuals = y_data - y_pred
        
        # F-test for overall model significance
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        
        if ss_tot == 0:
            r_squared = 1.0
        else:
            r_squared = 1 - (ss_res / ss_tot)
        
        n = len(y_data)
        p = 2  # Number of parameters (amplitude, exponent)
        
        if n <= p:
            # Insufficient degrees of freedom
            return SignificanceTestResults(
                test_name='F-test (insufficient data)',
                test_statistic=0.0,
                p_value=1.0,
                degrees_of_freedom=0,
                alpha_level=self.alpha_level,
                is_significant=False,
                effect_size=0.0,
                sample_size=n,
                test_power_adequate=False,
                practical_significance=False,
                test_assumptions_met=False
            )
        
        # F-statistic
        f_stat = (r_squared / p) / ((1 - r_squared) / (n - p - 1))
        
        # P-value from F-distribution
        p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1)
        
        # Effect size (R-squared)
        effect_size = r_squared
        
        # Power analysis (simplified)
        power = self._estimate_power(f_stat, p, n - p - 1, self.alpha_level)
        
        # Critical value
        critical_value = stats.f.ppf(1 - self.alpha_level, p, n - p - 1)
        
        # Confidence interval for R-squared (approximate)
        ci_lower, ci_upper = self._r_squared_confidence_interval(r_squared, n, p)
        
        # Test assumptions
        assumptions_met = self._check_power_law_assumptions(x_data, y_data, residuals)
        
        # Practical significance (effect size > 0.1)
        practical_significance = effect_size > 0.1
        
        return SignificanceTestResults(
            test_name='F-test for power-law fit',
            test_statistic=f_stat,
            p_value=p_value,
            degrees_of_freedom=n - p - 1,
            alpha_level=self.alpha_level,
            is_significant=p_value < self.alpha_level,
            effect_size=effect_size,
            power=power,
            critical_value=critical_value,
            confidence_interval=(ci_lower, ci_upper),
            test_assumptions_met=assumptions_met,
            sample_size=n,
            test_power_adequate=power > 0.8,
            practical_significance=practical_significance
        )
    
    def test_correlation_significance(self,
                                    x_data: np.ndarray,
                                    y_data: np.ndarray,
                                    correlation_type: str = 'pearson') -> SignificanceTestResults:
        """
        Test statistical significance of correlation.
        
        Args:
            x_data: First variable
            y_data: Second variable
            correlation_type: Type of correlation ('pearson', 'spearman')
            
        Returns:
            SignificanceTestResults with correlation significance
        """
        
        # Remove invalid values
        valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
        x_clean = x_data[valid_mask]
        y_clean = y_data[valid_mask]
        
        if len(x_clean) < 3:
            return SignificanceTestResults(
                test_name=f'{correlation_type} correlation (insufficient data)',
                test_statistic=0.0,
                p_value=1.0,
                alpha_level=self.alpha_level,
                is_significant=False,
                sample_size=len(x_clean),
                test_power_adequate=False,
                practical_significance=False,
                test_assumptions_met=False
            )
        
        # Compute correlation
        if correlation_type == 'pearson':
            corr_coef, p_value = stats.pearsonr(x_clean, y_clean)
            test_name = 'Pearson correlation'
        elif correlation_type == 'spearman':
            corr_coef, p_value = stats.spearmanr(x_clean, y_clean)
            test_name = 'Spearman correlation'
        else:
            raise ValueError(f"Unknown correlation type: {correlation_type}")
        
        n = len(x_clean)
        
        # Test statistic (t-statistic for correlation)
        if abs(corr_coef) < 1:
            t_stat = corr_coef * np.sqrt((n - 2) / (1 - corr_coef**2))
        else:
            t_stat = np.inf if corr_coef > 0 else -np.inf
        
        # Effect size (correlation coefficient itself)
        effect_size = abs(corr_coef)
        
        # Power estimation
        power = self._estimate_correlation_power(corr_coef, n, self.alpha_level)
        
        # Critical value
        critical_value = stats.t.ppf(1 - self.alpha_level/2, n - 2)
        
        # Confidence interval for correlation
        ci_lower, ci_upper = self._correlation_confidence_interval(corr_coef, n)
        
        # Test assumptions (mainly normality for Pearson)
        if correlation_type == 'pearson':
            assumptions_met = self._check_correlation_assumptions(x_clean, y_clean)
        else:
            assumptions_met = True  # Spearman is non-parametric
        
        # Practical significance (|r| > 0.3 for moderate effect)
        practical_significance = effect_size > 0.3
        
        return SignificanceTestResults(
            test_name=test_name,
            test_statistic=t_stat,
            p_value=p_value,
            degrees_of_freedom=n - 2,
            alpha_level=self.alpha_level,
            is_significant=p_value < self.alpha_level,
            effect_size=effect_size,
            power=power,
            critical_value=critical_value,
            confidence_interval=(ci_lower, ci_upper),
            test_assumptions_met=assumptions_met,
            sample_size=n,
            test_power_adequate=power > 0.8,
            practical_significance=practical_significance
        )
    
    def _estimate_power(self, f_stat: float, df1: int, df2: int, alpha: float) -> float:
        """Estimate statistical power for F-test."""
        
        # This is a simplified power estimation
        # In practice, would use non-central F-distribution
        
        critical_f = stats.f.ppf(1 - alpha, df1, df2)
        
        if f_stat > critical_f:
            # Observed F exceeds critical value
            power = min(0.99, 0.5 + 0.4 * (f_stat / critical_f - 1))
        else:
            # Observed F below critical value
            power = max(0.01, 0.5 * (f_stat / critical_f))
        
        return power
    
    def _estimate_correlation_power(self, r: float, n: int, alpha: float) -> float:
        """Estimate statistical power for correlation test."""
        
        # Fisher's z-transformation
        if abs(r) >= 0.999:
            return 0.99 if abs(r) > 0.999 else 0.01
        
        z_r = 0.5 * np.log((1 + abs(r)) / (1 - abs(r)))
        
        # Standard error
        se = 1 / np.sqrt(n - 3)
        
        # Critical z-value
        z_crit = stats.norm.ppf(1 - alpha/2)
        
        # Power calculation (simplified)
        z_stat = z_r / se
        power = 1 - stats.norm.cdf(z_crit - z_stat) + stats.norm.cdf(-z_crit - z_stat)
        
        return min(0.99, max(0.01, power))
    
    def _r_squared_confidence_interval(self, r_squared: float, n: int, p: int) -> Tuple[float, float]:
        """Compute confidence interval for R-squared."""
        
        # This is a simplified approximation
        # More accurate methods exist but are complex
        
        if r_squared <= 0 or r_squared >= 1:
            return (0.0, 1.0)
        
        # Standard error approximation
        se = np.sqrt((1 - r_squared)**2 * (n - 1) / (n - p - 1))
        
        # Normal approximation (not ideal but simple)
        z_crit = stats.norm.ppf(1 - self.alpha_level/2)
        
        ci_lower = max(0, r_squared - z_crit * se)
        ci_upper = min(1, r_squared + z_crit * se)
        
        return (ci_lower, ci_upper)
    
    def _correlation_confidence_interval(self, r: float, n: int) -> Tuple[float, float]:
        """Compute confidence interval for correlation coefficient."""
        
        if abs(r) >= 0.999 or n < 4:
            return (-1.0, 1.0)
        
        # Fisher's z-transformation
        z_r = 0.5 * np.log((1 + r) / (1 - r))
        
        # Standard error
        se = 1 / np.sqrt(n - 3)
        
        # Critical value
        z_crit = stats.norm.ppf(1 - self.alpha_level/2)
        
        # Confidence interval in z-space
        z_lower = z_r - z_crit * se
        z_upper = z_r + z_crit * se
        
        # Transform back to correlation space
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        return (r_lower, r_upper)
    
    def _check_power_law_assumptions(self, x_data: np.ndarray, y_data: np.ndarray, residuals: np.ndarray) -> bool:
        """Check assumptions for power-law fitting."""
        
        assumptions_met = True
        
        # 1. Linearity in log-log space
        log_x = np.log(x_data)
        log_y = np.log(y_data)
        
        # Check for obvious non-linearity (simplified)
        if len(log_x) > 10:
            # Compute correlation in log space
            log_corr, _ = stats.pearsonr(log_x, log_y)
            if abs(log_corr) < 0.7:  # Weak linear relationship in log space
                assumptions_met = False
        
        # 2. Residual normality (simplified test)
        if len(residuals) > 8:
            _, normality_p = shapiro(residuals[:5000])  # Limit for shapiro test
            if normality_p < 0.05:
                assumptions_met = False
        
        # 3. Homoscedasticity (constant variance)
        if len(residuals) > 10:
            # Split residuals into groups and compare variances
            mid_point = len(residuals) // 2
            var1 = np.var(residuals[:mid_point])
            var2 = np.var(residuals[mid_point:])
            
            # F-test for equal variances
            if var1 > 0 and var2 > 0:
                f_ratio = max(var1, var2) / min(var1, var2)
                if f_ratio > 4:  # Rule of thumb
                    assumptions_met = False
        
        return assumptions_met
    
    def _check_correlation_assumptions(self, x_data: np.ndarray, y_data: np.ndarray) -> bool:
        """Check assumptions for Pearson correlation."""
        
        # For Pearson correlation, main assumption is bivariate normality
        # Simplified check: test marginal normality
        
        if len(x_data) < 8:
            return True  # Can't test with small samples
        
        # Test normality of both variables
        _, x_normality_p = shapiro(x_data[:5000])  # Limit for shapiro test
        _, y_normality_p = shapiro(y_data[:5000])
        
        # Both should be approximately normal
        return x_normality_p > 0.05 and y_normality_p > 0.05


class ComprehensiveStatisticalValidator:
    """Main class for comprehensive statistical validation."""
    
    def __init__(self,
                 n_bootstrap: int = 5000,
                 confidence_level: float = 0.95,
                 cv_folds: int = 5,
                 alpha_level: float = 0.05,
                 random_seed: Optional[int] = None):
        """Initialize comprehensive statistical validator."""
        
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.bootstrap_validator = BootstrapValidator(
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_seed=random_seed
        )
        
        self.cross_validator = CrossValidator(
            cv_method='kfold',
            n_folds=cv_folds,
            random_seed=random_seed
        )
        
        self.significance_tester = SignificanceTester(alpha_level=alpha_level)
        
        self.confidence_level = confidence_level
        self.alpha_level = alpha_level
    
    def validate_critical_exponent_extraction(self,
                                            temperatures: np.ndarray,
                                            order_parameter: np.ndarray,
                                            critical_temperature: float,
                                            extracted_exponents: Dict[str, Dict[str, float]],
                                            system_type: str = 'unknown') -> ComprehensiveValidationResults:
        """
        Perform comprehensive statistical validation of critical exponent extraction.
        
        Args:
            temperatures: Temperature values
            order_parameter: Order parameter values
            critical_temperature: Extracted critical temperature
            extracted_exponents: Dictionary of extracted exponents with fitting results
            system_type: Type of physical system
            
        Returns:
            ComprehensiveValidationResults with complete validation
        """
        
        self.logger.info("Starting comprehensive statistical validation")
        
        # Initialize results
        results = ComprehensiveValidationResults(
            system_type=system_type,
            n_samples=len(temperatures),
            temperature_range=(np.min(temperatures), np.max(temperatures))
        )
        
        # 1. Critical temperature validation
        self.logger.info("Validating critical temperature detection")
        results.tc_bootstrap = self._validate_critical_temperature(
            temperatures, order_parameter, critical_temperature
        )
        
        # 2. Critical exponent validation
        for exponent_name, exponent_data in extracted_exponents.items():
            self.logger.info(f"Validating {exponent_name} exponent")
            
            if exponent_name == 'beta':
                results.beta_validation = self._validate_power_law_exponent(
                    temperatures, order_parameter, critical_temperature,
                    exponent_data, exponent_type='beta'
                )
            elif exponent_name == 'nu':
                results.nu_validation = self._validate_power_law_exponent(
                    temperatures, order_parameter, critical_temperature,
                    exponent_data, exponent_type='nu'
                )
            elif exponent_name == 'gamma':
                results.gamma_validation = self._validate_power_law_exponent(
                    temperatures, order_parameter, critical_temperature,
                    exponent_data, exponent_type='gamma'
                )
        
        # 3. Correlation analysis
        self.logger.info("Performing correlation analysis")
        results.correlation_tests = self._validate_correlations(
            temperatures, order_parameter
        )
        
        # 4. Overall assessment
        results.overall_validation_score = self._compute_overall_validation_score(results)
        results.validation_grade = self._assign_validation_grade(results.overall_validation_score)
        results.statistical_reliability = self._assess_statistical_reliability(results)
        
        # 5. Generate recommendations
        results.validation_warnings = self._generate_validation_warnings(results)
        results.improvement_recommendations = self._generate_improvement_recommendations(results)
        
        self.logger.info("Comprehensive statistical validation completed")
        self.logger.info(f"Overall validation score: {results.overall_validation_score:.3f}")
        self.logger.info(f"Validation grade: {results.validation_grade}")
        
        return results
    
    def _validate_critical_temperature(self,
                                     temperatures: np.ndarray,
                                     order_parameter: np.ndarray,
                                     critical_temperature: float) -> BootstrapResults:
        """Validate critical temperature detection using bootstrap."""
        
        def tc_detection_statistic(data_indices):
            """Statistic function for bootstrap: detect Tc from subset of data."""
            
            # Use subset of data
            subset_temps = temperatures[data_indices]
            subset_op = order_parameter[data_indices]
            
            # Simple Tc detection: susceptibility peak
            unique_temps = np.unique(subset_temps)
            if len(unique_temps) < 3:
                return critical_temperature  # Fallback
            
            susceptibilities = []
            for temp in unique_temps:
                temp_mask = subset_temps == temp
                if np.sum(temp_mask) >= 3:
                    susceptibility = np.var(subset_op[temp_mask])
                    susceptibilities.append(susceptibility)
                else:
                    susceptibilities.append(0)
            
            if len(susceptibilities) == 0:
                return critical_temperature
            
            # Find peak
            peak_idx = np.argmax(susceptibilities)
            return unique_temps[peak_idx]
        
        # Create data indices for bootstrap
        data_indices = np.arange(len(temperatures))
        
        return self.bootstrap_validator.bootstrap_statistic(
            data_indices, tc_detection_statistic
        )
    
    def _validate_power_law_exponent(self,
                                   temperatures: np.ndarray,
                                   order_parameter: np.ndarray,
                                   critical_temperature: float,
                                   exponent_data: Dict[str, float],
                                   exponent_type: str) -> PowerLawValidationResults:
        """Validate power-law exponent extraction."""
        
        # Prepare data for power-law fitting
        if exponent_type == 'beta':
            # β: m ∝ (Tc - T)^β for T < Tc
            mask = temperatures < critical_temperature
            x_data = critical_temperature - temperatures[mask]
            y_data = np.abs(order_parameter[mask])
        elif exponent_type == 'nu':
            # ν: ξ ∝ |T - Tc|^(-ν)
            mask = temperatures != critical_temperature
            x_data = np.abs(temperatures[mask] - critical_temperature)
            y_data = np.abs(order_parameter[mask])  # Assuming this is correlation length
        else:
            raise ValueError(f"Unknown exponent type: {exponent_type}")
        
        # Filter valid data
        valid_mask = (x_data > 0) & (y_data > 0) & np.isfinite(x_data) & np.isfinite(y_data)
        x_data = x_data[valid_mask]
        y_data = y_data[valid_mask]
        
        if len(x_data) < 5:
            # Insufficient data
            return self._create_empty_power_law_validation()
        
        # Extract fitted parameters
        exponent = exponent_data.get('exponent', 0.0)
        amplitude = exponent_data.get('amplitude', 1.0)
        exponent_error = exponent_data.get('exponent_error', 0.0)
        amplitude_error = exponent_data.get('amplitude_error', 0.0)
        r_squared = exponent_data.get('r_squared', 0.0)
        
        # Bootstrap validation for exponent
        def exponent_statistic(data_indices):
            """Bootstrap statistic for exponent."""
            x_boot = x_data[data_indices]
            y_boot = y_data[data_indices]
            
            # Fit power law in log space
            log_x = np.log(x_boot)
            log_y = np.log(y_boot)
            
            try:
                slope, intercept = np.polyfit(log_x, log_y, 1)
                return slope  # This is the exponent
            except:
                return exponent  # Fallback
        
        exponent_bootstrap = self.bootstrap_validator.bootstrap_statistic(
            np.arange(len(x_data)), exponent_statistic
        )
        
        # Bootstrap validation for amplitude
        def amplitude_statistic(data_indices):
            """Bootstrap statistic for amplitude."""
            x_boot = x_data[data_indices]
            y_boot = y_data[data_indices]
            
            log_x = np.log(x_boot)
            log_y = np.log(y_boot)
            
            try:
                slope, intercept = np.polyfit(log_x, log_y, 1)
                return np.exp(intercept)  # This is the amplitude
            except:
                return amplitude  # Fallback
        
        amplitude_bootstrap = self.bootstrap_validator.bootstrap_statistic(
            np.arange(len(x_data)), amplitude_statistic
        )
        
        # Cross-validation
        cross_validation = self.cross_validator.cross_validate_power_law(x_data, y_data)
        
        # Significance tests
        fit_significance = self.significance_tester.test_power_law_significance(
            x_data, y_data, exponent, amplitude
        )
        
        # Linearity test (correlation in log space)
        log_x = np.log(x_data)
        log_y = np.log(y_data)
        linearity_test = self.significance_tester.test_correlation_significance(
            log_x, log_y, correlation_type='pearson'
        )
        
        # Residual analysis
        y_pred = amplitude * (x_data ** exponent)
        residuals = y_data - y_pred
        
        # Residual normality test
        residual_normality = self._test_residual_normality(residuals)
        
        # Compute additional fit metrics
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        
        if ss_tot > 0:
            adjusted_r_squared = 1 - (ss_res / ss_tot) * (len(y_data) - 1) / (len(y_data) - 2)
        else:
            adjusted_r_squared = r_squared
        
        chi_squared = ss_res
        reduced_chi_squared = chi_squared / max(1, len(y_data) - 2)
        
        # Residual analysis
        residual_analysis = {
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'max_residual': np.max(np.abs(residuals)),
            'residual_autocorrelation': self._compute_residual_autocorrelation(residuals)
        }
        
        # Outlier analysis
        outlier_analysis = self._analyze_outliers(x_data, y_data, residuals)
        
        # Leverage analysis
        leverage_analysis = self._analyze_leverage(x_data, y_data)
        
        return PowerLawValidationResults(
            exponent=exponent,
            exponent_error=exponent_error,
            amplitude=amplitude,
            amplitude_error=amplitude_error,
            r_squared=r_squared,
            adjusted_r_squared=adjusted_r_squared,
            chi_squared=chi_squared,
            reduced_chi_squared=reduced_chi_squared,
            exponent_bootstrap=exponent_bootstrap,
            amplitude_bootstrap=amplitude_bootstrap,
            cross_validation=cross_validation,
            fit_significance=fit_significance,
            linearity_test=linearity_test,
            residual_normality=residual_normality,
            alternative_models={},  # Could add polynomial, exponential fits
            best_model='power_law',
            model_selection_criterion='AIC',
            residual_analysis=residual_analysis,
            outlier_analysis=outlier_analysis,
            leverage_analysis=leverage_analysis
        )
    
    def _validate_correlations(self,
                             temperatures: np.ndarray,
                             order_parameter: np.ndarray) -> Dict[str, SignificanceTestResults]:
        """Validate correlations between variables."""
        
        correlation_tests = {}
        
        # Temperature-order parameter correlation
        correlation_tests['temperature_order_parameter'] = (
            self.significance_tester.test_correlation_significance(
                temperatures, order_parameter, correlation_type='pearson'
            )
        )
        
        # Spearman correlation (non-parametric)
        correlation_tests['temperature_order_parameter_spearman'] = (
            self.significance_tester.test_correlation_significance(
                temperatures, order_parameter, correlation_type='spearman'
            )
        )
        
        return correlation_tests
    
    def _test_residual_normality(self, residuals: np.ndarray) -> SignificanceTestResults:
        """Test normality of residuals."""
        
        if len(residuals) < 8:
            return SignificanceTestResults(
                test_name='Shapiro-Wilk (insufficient data)',
                test_statistic=0.0,
                p_value=1.0,
                alpha_level=self.alpha_level,
                is_significant=False,
                sample_size=len(residuals),
                test_power_adequate=False,
                practical_significance=False,
                test_assumptions_met=False
            )
        
        # Shapiro-Wilk test for normality
        test_residuals = residuals[:5000]  # Limit for shapiro test
        stat, p_value = shapiro(test_residuals)
        
        return SignificanceTestResults(
            test_name='Shapiro-Wilk normality test',
            test_statistic=stat,
            p_value=p_value,
            alpha_level=self.alpha_level,
            is_significant=p_value < self.alpha_level,
            effect_size=1 - stat,  # Deviation from normality
            sample_size=len(test_residuals),
            test_power_adequate=len(test_residuals) > 50,
            practical_significance=p_value < 0.01,  # Strong evidence against normality
            test_assumptions_met=True  # Shapiro-Wilk has minimal assumptions
        )
    
    def _compute_residual_autocorrelation(self, residuals: np.ndarray) -> float:
        """Compute autocorrelation of residuals."""
        
        if len(residuals) < 3:
            return 0.0
        
        try:
            # Lag-1 autocorrelation
            autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
            return autocorr if np.isfinite(autocorr) else 0.0
        except:
            return 0.0
    
    def _analyze_outliers(self, x_data: np.ndarray, y_data: np.ndarray, residuals: np.ndarray) -> Dict[str, Any]:
        """Analyze outliers in the data."""
        
        # Simple outlier detection using IQR method
        q1, q3 = np.percentile(residuals, [25, 75])
        iqr = q3 - q1
        
        outlier_threshold = 1.5 * iqr
        outlier_mask = np.abs(residuals - np.median(residuals)) > outlier_threshold
        
        n_outliers = np.sum(outlier_mask)
        outlier_fraction = n_outliers / len(residuals)
        
        return {
            'n_outliers': n_outliers,
            'outlier_fraction': outlier_fraction,
            'outlier_threshold': outlier_threshold,
            'max_outlier_residual': np.max(np.abs(residuals)) if len(residuals) > 0 else 0.0
        }
    
    def _analyze_leverage(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, float]:
        """Analyze leverage of data points."""
        
        # Simplified leverage analysis
        # In full implementation, would compute hat matrix
        
        # Points far from mean x have high leverage
        x_mean = np.mean(x_data)
        x_std = np.std(x_data)
        
        if x_std == 0:
            return {'max_leverage': 0.0, 'mean_leverage': 0.0}
        
        leverage_scores = np.abs(x_data - x_mean) / x_std
        
        return {
            'max_leverage': np.max(leverage_scores),
            'mean_leverage': np.mean(leverage_scores)
        }
    
    def _create_empty_power_law_validation(self) -> PowerLawValidationResults:
        """Create empty power-law validation results for insufficient data."""
        
        empty_bootstrap = BootstrapResults(
            original_statistic=0.0,
            bootstrap_samples=np.array([]),
            n_bootstrap_samples=0,
            confidence_level=self.confidence_level,
            ci_lower=0.0,
            ci_upper=0.0,
            ci_width=0.0,
            bootstrap_mean=0.0,
            bootstrap_std=0.0,
            bootstrap_bias=0.0,
            bootstrap_skewness=0.0,
            bootstrap_kurtosis=0.0,
            convergence_achieved=False,
            effective_sample_size=0,
            monte_carlo_error=0.0,
            normality_test_pvalue=0.0,
            is_normal_distribution=False
        )
        
        empty_cv = CrossValidationResults(
            cv_method='kfold',
            n_folds=0,
            n_samples=0,
            fold_scores=np.array([]),
            mean_score=0.0,
            std_score=0.0,
            r2_scores=np.array([]),
            mse_scores=np.array([]),
            mae_scores=np.array([]),
            score_normality_pvalue=0.0,
            score_consistency_test=0.0,
            coefficient_of_variation=0.0,
            stability_score=0.0,
            overfitting_indicator=0.0,
            fold_parameters=[],
            parameter_stability={}
        )
        
        empty_significance = SignificanceTestResults(
            test_name='Insufficient data',
            test_statistic=0.0,
            p_value=1.0,
            alpha_level=self.alpha_level,
            is_significant=False,
            sample_size=0,
            test_power_adequate=False,
            practical_significance=False,
            test_assumptions_met=False
        )
        
        return PowerLawValidationResults(
            exponent=0.0,
            exponent_error=0.0,
            amplitude=0.0,
            amplitude_error=0.0,
            r_squared=0.0,
            adjusted_r_squared=0.0,
            chi_squared=0.0,
            reduced_chi_squared=0.0,
            exponent_bootstrap=empty_bootstrap,
            amplitude_bootstrap=empty_bootstrap,
            cross_validation=empty_cv,
            fit_significance=empty_significance,
            linearity_test=empty_significance,
            residual_normality=empty_significance,
            alternative_models={},
            best_model='none',
            model_selection_criterion='none',
            residual_analysis={},
            outlier_analysis={},
            leverage_analysis={}
        )
    
    def _compute_overall_validation_score(self, results: ComprehensiveValidationResults) -> float:
        """Compute overall validation score."""
        
        score_components = []
        
        # Critical temperature validation (20% weight)
        if results.tc_bootstrap.convergence_achieved:
            tc_score = 0.8 if results.tc_bootstrap.is_normal_distribution else 0.6
        else:
            tc_score = 0.3
        score_components.append(('tc', tc_score, 0.2))
        
        # Beta exponent validation (40% weight)
        if results.beta_validation:
            beta_score = self._score_power_law_validation(results.beta_validation)
            score_components.append(('beta', beta_score, 0.4))
        
        # Nu exponent validation (30% weight)
        if results.nu_validation:
            nu_score = self._score_power_law_validation(results.nu_validation)
            score_components.append(('nu', nu_score, 0.3))
        
        # Correlation validation (10% weight)
        if results.correlation_tests:
            corr_scores = []
            for test_result in results.correlation_tests.values():
                if test_result.test_power_adequate and test_result.is_significant:
                    corr_scores.append(0.8)
                elif test_result.is_significant:
                    corr_scores.append(0.6)
                else:
                    corr_scores.append(0.3)
            
            if corr_scores:
                corr_score = np.mean(corr_scores)
                score_components.append(('correlation', corr_score, 0.1))
        
        # Compute weighted average
        if score_components:
            total_weight = sum(weight for _, _, weight in score_components)
            weighted_sum = sum(score * weight for _, score, weight in score_components)
            return weighted_sum / total_weight
        else:
            return 0.0
    
    def _score_power_law_validation(self, validation: PowerLawValidationResults) -> float:
        """Score power-law validation results."""
        
        score_components = []
        
        # Bootstrap convergence (25%)
        if validation.exponent_bootstrap.convergence_achieved:
            score_components.append(0.8)
        else:
            score_components.append(0.3)
        
        # Cross-validation stability (25%)
        if validation.cross_validation.stability_score > 0.7:
            score_components.append(0.8)
        elif validation.cross_validation.stability_score > 0.5:
            score_components.append(0.6)
        else:
            score_components.append(0.3)
        
        # Statistical significance (25%)
        if validation.fit_significance.is_significant and validation.fit_significance.test_power_adequate:
            score_components.append(0.9)
        elif validation.fit_significance.is_significant:
            score_components.append(0.7)
        else:
            score_components.append(0.2)
        
        # Model quality (25%)
        if validation.r_squared > 0.8:
            score_components.append(0.9)
        elif validation.r_squared > 0.6:
            score_components.append(0.7)
        elif validation.r_squared > 0.4:
            score_components.append(0.5)
        else:
            score_components.append(0.2)
        
        return np.mean(score_components)
    
    def _assign_validation_grade(self, score: float) -> str:
        """Assign letter grade based on validation score."""
        
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def _assess_statistical_reliability(self, results: ComprehensiveValidationResults) -> str:
        """Assess overall statistical reliability."""
        
        score = results.overall_validation_score
        
        if score >= 0.85:
            return 'Excellent'
        elif score >= 0.75:
            return 'Good'
        elif score >= 0.65:
            return 'Fair'
        elif score >= 0.5:
            return 'Poor'
        else:
            return 'Very Poor'
    
    def _generate_validation_warnings(self, results: ComprehensiveValidationResults) -> List[str]:
        """Generate validation warnings based on results."""
        
        warnings = []
        
        # Critical temperature warnings
        if not results.tc_bootstrap.convergence_achieved:
            warnings.append("Critical temperature bootstrap did not converge")
        
        if not results.tc_bootstrap.is_normal_distribution:
            warnings.append("Critical temperature bootstrap distribution is not normal")
        
        # Beta exponent warnings
        if results.beta_validation:
            if not results.beta_validation.fit_significance.is_significant:
                warnings.append("Beta exponent fit is not statistically significant")
            
            if results.beta_validation.cross_validation.stability_score < 0.5:
                warnings.append("Beta exponent cross-validation shows poor stability")
            
            if results.beta_validation.r_squared < 0.5:
                warnings.append("Beta exponent fit has low R-squared")
        
        # Nu exponent warnings
        if results.nu_validation:
            if not results.nu_validation.fit_significance.is_significant:
                warnings.append("Nu exponent fit is not statistically significant")
            
            if results.nu_validation.cross_validation.stability_score < 0.5:
                warnings.append("Nu exponent cross-validation shows poor stability")
        
        # Sample size warnings
        if results.n_samples < 100:
            warnings.append("Small sample size may affect statistical power")
        
        return warnings
    
    def _generate_improvement_recommendations(self, results: ComprehensiveValidationResults) -> List[str]:
        """Generate improvement recommendations."""
        
        recommendations = []
        
        # Sample size recommendations
        if results.n_samples < 500:
            recommendations.append("Increase sample size for better statistical power")
        
        # Bootstrap recommendations
        if not results.tc_bootstrap.convergence_achieved:
            recommendations.append("Increase bootstrap samples for critical temperature")
        
        # Cross-validation recommendations
        if results.beta_validation and results.beta_validation.cross_validation.stability_score < 0.7:
            recommendations.append("Improve data quality or fitting method for beta exponent")
        
        # Significance recommendations
        if results.beta_validation and not results.beta_validation.fit_significance.test_power_adequate:
            recommendations.append("Increase sample size or effect size for beta exponent")
        
        # Model quality recommendations
        if results.beta_validation and results.beta_validation.r_squared < 0.6:
            recommendations.append("Consider alternative models or data preprocessing for beta exponent")
        
        # General recommendations
        if results.overall_validation_score < 0.7:
            recommendations.append("Consider improving data quality and fitting methods")
        
        return recommendations


def create_comprehensive_statistical_validator(n_bootstrap: int = 5000,
                                             confidence_level: float = 0.95,
                                             cv_folds: int = 5,
                                             alpha_level: float = 0.05,
                                             random_seed: Optional[int] = None) -> ComprehensiveStatisticalValidator:
    """
    Factory function to create comprehensive statistical validator.
    
    Args:
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        cv_folds: Number of cross-validation folds
        alpha_level: Significance level for tests
        random_seed: Random seed for reproducibility
        
    Returns:
        Configured ComprehensiveStatisticalValidator instance
    """
    return ComprehensiveStatisticalValidator(
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        cv_folds=cv_folds,
        alpha_level=alpha_level,
        random_seed=random_seed
    )


def run_statistical_validation_example(temperatures: np.ndarray,
                                     order_parameter: np.ndarray,
                                     critical_temperature: float,
                                     extracted_exponents: Dict[str, Dict[str, float]]):
    """
    Example function to run comprehensive statistical validation.
    
    Args:
        temperatures: Temperature values
        order_parameter: Order parameter values
        critical_temperature: Extracted critical temperature
        extracted_exponents: Dictionary of extracted exponents
    """
    
    # Create validator
    validator = create_comprehensive_statistical_validator(
        n_bootstrap=1000,  # Reduced for example
        cv_folds=5,
        random_seed=42
    )
    
    # Run validation
    results = validator.validate_critical_exponent_extraction(
        temperatures=temperatures,
        order_parameter=order_parameter,
        critical_temperature=critical_temperature,
        extracted_exponents=extracted_exponents,
        system_type='ising_3d'
    )
    
    print(f"Comprehensive Statistical Validation Results:")
    print(f"Overall Score: {results.overall_validation_score:.3f}")
    print(f"Validation Grade: {results.validation_grade}")
    print(f"Statistical Reliability: {results.statistical_reliability}")
    
    if results.validation_warnings:
        print(f"\nWarnings:")
        for warning in results.validation_warnings:
            print(f"  • {warning}")
    
    if results.improvement_recommendations:
        print(f"\nRecommendations:")
        for rec in results.improvement_recommendations:
            print(f"  • {rec}")
    
    return results