"""
Comprehensive Validation Framework

This module implements comprehensive validation and quality assurance for
critical exponent extraction, including bootstrap confidence intervals,
cross-validation, statistical tests, and quality metrics.

Task 8: Implement comprehensive validation framework
Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from scipy import stats
from scipy.optimize import curve_fit
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from .numerical_stability_fixes import fit_power_law_safe, safe_log, safe_divide
from ..utils.logging_utils import get_logger


@dataclass
class BootstrapResult:
    """Result from bootstrap confidence interval computation."""
    mean: float
    std: float
    confidence_interval: Tuple[float, float]
    confidence_level: float
    n_samples: int
    bootstrap_distribution: np.ndarray
    bias: float
    converged: bool
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'mean': self.mean,
            'std': self.std,
            'confidence_interval': self.confidence_interval,
            'confidence_level': self.confidence_level,
            'n_samples': self.n_samples,
            'bias': self.bias,
            'converged': self.converged
        }


@dataclass
class CrossValidationResult:
    """Result from cross-validation."""
    mean_score: float
    std_score: float
    fold_scores: List[float]
    stability_score: float
    n_folds: int
    success: bool
    message: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'mean_score': self.mean_score,
            'std_score': self.std_score,
            'fold_scores': self.fold_scores,
            'stability_score': self.stability_score,
            'n_folds': self.n_folds,
            'success': self.success,
            'message': self.message
        }


@dataclass
class StatisticalTestResults:
    """Results from statistical significance tests."""
    f_test_statistic: float
    f_test_p_value: float
    f_test_significant: bool
    shapiro_statistic: float
    shapiro_p_value: float
    normality_passed: bool
    durbin_watson_statistic: float
    autocorrelation_passed: bool
    breusch_pagan_statistic: float
    breusch_pagan_p_value: float
    homoscedasticity_passed: bool
    all_tests_passed: bool
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'f_test': {
                'statistic': self.f_test_statistic,
                'p_value': self.f_test_p_value,
                'significant': self.f_test_significant
            },
            'shapiro_wilk': {
                'statistic': self.shapiro_statistic,
                'p_value': self.shapiro_p_value,
                'passed': self.normality_passed
            },
            'durbin_watson': {
                'statistic': self.durbin_watson_statistic,
                'passed': self.autocorrelation_passed
            },
            'breusch_pagan': {
                'statistic': self.breusch_pagan_statistic,
                'p_value': self.breusch_pagan_p_value,
                'passed': self.homoscedasticity_passed
            },
            'all_tests_passed': self.all_tests_passed
        }


@dataclass
class QualityMetrics:
    """Quality assessment metrics for model fits."""
    r_squared: float
    adjusted_r_squared: float
    aic: float
    bic: float
    rmse: float
    mae: float
    residual_std: float
    max_residual: float
    quality_grade: str  # EXCELLENT, GOOD, FAIR, POOR
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'r_squared': self.r_squared,
            'adjusted_r_squared': self.adjusted_r_squared,
            'aic': self.aic,
            'bic': self.bic,
            'rmse': self.rmse,
            'mae': self.mae,
            'residual_std': self.residual_std,
            'max_residual': self.max_residual,
            'quality_grade': self.quality_grade
        }


class ValidationFramework:
    """
    Comprehensive validation framework for critical exponent extraction.
    
    Provides bootstrap confidence intervals, cross-validation, statistical
    tests, and quality metrics for publication-ready validation.
    """
    
    def __init__(self,
                 n_bootstrap: int = 1000,
                 confidence_level: float = 0.95,
                 n_folds: int = 5,
                 alpha: float = 0.05,
                 parallel: bool = True,
                 n_jobs: Optional[int] = None):
        """
        Initialize validation framework.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95%)
            n_folds: Number of folds for cross-validation
            alpha: Significance level for statistical tests
            parallel: Whether to use parallel processing for bootstrap
            n_jobs: Number of parallel jobs (None = use all cores)
        """
        self.logger = get_logger(__name__)
        
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.n_folds = n_folds
        self.alpha = alpha
        self.parallel = parallel
        self.n_jobs = n_jobs
        
        self.logger.info(f"Validation framework initialized:")
        self.logger.info(f"  Bootstrap samples: {n_bootstrap}")
        self.logger.info(f"  Confidence level: {confidence_level:.1%}")
        self.logger.info(f"  Cross-validation folds: {n_folds}")
        self.logger.info(f"  Significance level: {alpha}")
        self.logger.info(f"  Parallel processing: {parallel}")
    
    def bootstrap_confidence_interval(self,
                                     x_data: np.ndarray,
                                     y_data: np.ndarray,
                                     fit_function: Callable,
                                     parameter_index: int = 0) -> BootstrapResult:
        """
        Compute bootstrap confidence intervals for fitted parameters.
        
        Args:
            x_data: Independent variable data
            y_data: Dependent variable data
            fit_function: Function that fits data and returns parameters
            parameter_index: Which parameter to compute CI for
            
        Returns:
            BootstrapResult with confidence intervals
        """
        self.logger.info(f"Computing bootstrap CI with {self.n_bootstrap} samples")
        
        n_samples = len(x_data)
        bootstrap_estimates = []
        
        # Original estimate
        try:
            original_params = fit_function(x_data, y_data)
            original_estimate = original_params[parameter_index]
        except Exception as e:
            self.logger.error(f"Original fit failed: {e}")
            return BootstrapResult(
                mean=0.0,
                std=0.0,
                confidence_interval=(0.0, 0.0),
                confidence_level=self.confidence_level,
                n_samples=0,
                bootstrap_distribution=np.array([]),
                bias=0.0,
                converged=False
            )
        
        # Bootstrap sampling
        if self.parallel and self.n_bootstrap > 100:
            # Parallel bootstrap
            bootstrap_estimates = self._parallel_bootstrap(
                x_data, y_data, fit_function, parameter_index
            )
        else:
            # Sequential bootstrap
            for i in range(self.n_bootstrap):
                # Resample with replacement
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                x_boot = x_data[indices]
                y_boot = y_data[indices]
                
                try:
                    params = fit_function(x_boot, y_boot)
                    bootstrap_estimates.append(params[parameter_index])
                except:
                    # Skip failed fits
                    continue
        
        bootstrap_estimates = np.array(bootstrap_estimates)
        
        if len(bootstrap_estimates) < 100:
            self.logger.warning(f"Only {len(bootstrap_estimates)} successful bootstrap samples")
            converged = False
        else:
            converged = True
        
        # Compute statistics
        mean_estimate = np.mean(bootstrap_estimates)
        std_estimate = np.std(bootstrap_estimates)
        bias = mean_estimate - original_estimate
        
        # Percentile method for CI
        alpha_lower = (1 - self.confidence_level) / 2
        alpha_upper = 1 - alpha_lower
        
        ci_lower = np.percentile(bootstrap_estimates, alpha_lower * 100)
        ci_upper = np.percentile(bootstrap_estimates, alpha_upper * 100)
        
        self.logger.info(f"Bootstrap complete:")
        self.logger.info(f"  Mean: {mean_estimate:.4f}")
        self.logger.info(f"  Std: {std_estimate:.4f}")
        self.logger.info(f"  {self.confidence_level:.1%} CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        self.logger.info(f"  Bias: {bias:.4f}")
        
        return BootstrapResult(
            mean=mean_estimate,
            std=std_estimate,
            confidence_interval=(ci_lower, ci_upper),
            confidence_level=self.confidence_level,
            n_samples=len(bootstrap_estimates),
            bootstrap_distribution=bootstrap_estimates,
            bias=bias,
            converged=converged
        )
    
    def _parallel_bootstrap(self,
                           x_data: np.ndarray,
                           y_data: np.ndarray,
                           fit_function: Callable,
                           parameter_index: int) -> List[float]:
        """
        Perform bootstrap sampling in parallel.
        
        Args:
            x_data: Independent variable data
            y_data: Dependent variable data
            fit_function: Fitting function
            parameter_index: Parameter index to extract
            
        Returns:
            List of bootstrap estimates
        """
        n_samples = len(x_data)
        bootstrap_estimates = []
        
        def bootstrap_iteration(seed):
            """Single bootstrap iteration."""
            np.random.seed(seed)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            x_boot = x_data[indices]
            y_boot = y_data[indices]
            
            try:
                params = fit_function(x_boot, y_boot)
                return params[parameter_index]
            except:
                return None
        
        # Use ProcessPoolExecutor for parallel bootstrap
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit all bootstrap iterations
            futures = [
                executor.submit(bootstrap_iteration, i)
                for i in range(self.n_bootstrap)
            ]
            
            # Collect results
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    bootstrap_estimates.append(result)
        
        return bootstrap_estimates
    
    def cross_validate(self,
                      x_data: np.ndarray,
                      y_data: np.ndarray,
                      fit_function: Callable,
                      score_function: Optional[Callable] = None) -> CrossValidationResult:
        """
        Perform k-fold cross-validation.
        
        Args:
            x_data: Independent variable data
            y_data: Dependent variable data
            fit_function: Function that fits data and returns predictions
            score_function: Function to compute score (default: R²)
            
        Returns:
            CrossValidationResult with stability metrics
        """
        self.logger.info(f"Performing {self.n_folds}-fold cross-validation")
        
        n_samples = len(x_data)
        fold_size = n_samples // self.n_folds
        
        if score_function is None:
            # Default: R² score
            score_function = lambda y_true, y_pred: 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
        
        fold_scores = []
        
        # K-fold cross-validation
        for fold in range(self.n_folds):
            # Split data
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < self.n_folds - 1 else n_samples
            
            test_indices = np.arange(test_start, test_end)
            train_indices = np.concatenate([
                np.arange(0, test_start),
                np.arange(test_end, n_samples)
            ])
            
            x_train, y_train = x_data[train_indices], y_data[train_indices]
            x_test, y_test = x_data[test_indices], y_data[test_indices]
            
            try:
                # Fit on training data
                y_pred = fit_function(x_train, y_train, x_test)
                
                # Score on test data
                score = score_function(y_test, y_pred)
                fold_scores.append(score)
                
                self.logger.info(f"  Fold {fold+1}/{self.n_folds}: score = {score:.4f}")
                
            except Exception as e:
                self.logger.warning(f"  Fold {fold+1} failed: {e}")
                continue
        
        if len(fold_scores) < 2:
            return CrossValidationResult(
                mean_score=0.0,
                std_score=0.0,
                fold_scores=fold_scores,
                stability_score=0.0,
                n_folds=len(fold_scores),
                success=False,
                message=f'Insufficient successful folds: {len(fold_scores)}'
            )
        
        # Compute statistics
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        # Stability score (higher is better, 1.0 = perfect stability)
        cv = std_score / abs(mean_score) if mean_score != 0 else float('inf')
        stability_score = np.exp(-cv)  # Exponential decay with CV
        
        self.logger.info(f"Cross-validation complete:")
        self.logger.info(f"  Mean score: {mean_score:.4f} ± {std_score:.4f}")
        self.logger.info(f"  Stability: {stability_score:.2%}")
        
        return CrossValidationResult(
            mean_score=mean_score,
            std_score=std_score,
            fold_scores=fold_scores,
            stability_score=stability_score,
            n_folds=len(fold_scores),
            success=True,
            message='Cross-validation successful'
        )
    
    def statistical_tests(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         n_params: int = 2) -> StatisticalTestResults:
        """
        Perform comprehensive statistical significance tests.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            n_params: Number of parameters in model
            
        Returns:
            StatisticalTestResults with all test results
        """
        self.logger.info("Performing statistical significance tests")
        
        residuals = y_true - y_pred
        n_samples = len(y_true)
        
        # 1. F-test for model significance
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        
        if ss_tot > 0:
            r_squared = 1 - ss_res / ss_tot
            f_statistic = (r_squared / n_params) / ((1 - r_squared) / (n_samples - n_params - 1))
            f_p_value = 1 - stats.f.cdf(f_statistic, n_params, n_samples - n_params - 1)
            f_significant = f_p_value < self.alpha
        else:
            f_statistic, f_p_value, f_significant = 0.0, 1.0, False
        
        self.logger.info(f"  F-test: F={f_statistic:.4f}, p={f_p_value:.4f}, significant={f_significant}")
        
        # 2. Shapiro-Wilk test for normality of residuals
        if len(residuals) >= 3:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            normality_passed = shapiro_p > self.alpha
        else:
            shapiro_stat, shapiro_p, normality_passed = 0.0, 1.0, False
        
        self.logger.info(f"  Shapiro-Wilk: W={shapiro_stat:.4f}, p={shapiro_p:.4f}, passed={normality_passed}")
        
        # 3. Durbin-Watson test for autocorrelation
        dw_stat = self._durbin_watson(residuals)
        # DW statistic: 2 = no autocorrelation, <2 = positive, >2 = negative
        # Accept if 1.5 < DW < 2.5
        autocorr_passed = 1.5 < dw_stat < 2.5
        
        self.logger.info(f"  Durbin-Watson: DW={dw_stat:.4f}, passed={autocorr_passed}")
        
        # 4. Breusch-Pagan test for homoscedasticity
        bp_stat, bp_p = self._breusch_pagan(residuals, y_pred)
        homosced_passed = bp_p > self.alpha
        
        self.logger.info(f"  Breusch-Pagan: BP={bp_stat:.4f}, p={bp_p:.4f}, passed={homosced_passed}")
        
        # Overall assessment
        all_passed = f_significant and normality_passed and autocorr_passed and homosced_passed
        
        self.logger.info(f"  All tests passed: {all_passed}")
        
        return StatisticalTestResults(
            f_test_statistic=f_statistic,
            f_test_p_value=f_p_value,
            f_test_significant=f_significant,
            shapiro_statistic=shapiro_stat,
            shapiro_p_value=shapiro_p,
            normality_passed=normality_passed,
            durbin_watson_statistic=dw_stat,
            autocorrelation_passed=autocorr_passed,
            breusch_pagan_statistic=bp_stat,
            breusch_pagan_p_value=bp_p,
            homoscedasticity_passed=homosced_passed,
            all_tests_passed=all_passed
        )
    
    def _durbin_watson(self, residuals: np.ndarray) -> float:
        """
        Compute Durbin-Watson statistic for autocorrelation.
        
        Args:
            residuals: Residuals from fit
            
        Returns:
            Durbin-Watson statistic
        """
        diff_residuals = np.diff(residuals)
        dw = np.sum(diff_residuals**2) / np.sum(residuals**2)
        return dw
    
    def _breusch_pagan(self,
                      residuals: np.ndarray,
                      fitted_values: np.ndarray) -> Tuple[float, float]:
        """
        Compute Breusch-Pagan test for homoscedasticity.
        
        Args:
            residuals: Residuals from fit
            fitted_values: Fitted values
            
        Returns:
            Tuple of (test statistic, p-value)
        """
        # Regress squared residuals on fitted values
        squared_residuals = residuals**2
        
        # Simple linear regression
        n = len(residuals)
        x_mean = np.mean(fitted_values)
        y_mean = np.mean(squared_residuals)
        
        numerator = np.sum((fitted_values - x_mean) * (squared_residuals - y_mean))
        denominator = np.sum((fitted_values - x_mean)**2)
        
        if denominator > 0:
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
            
            # Predicted squared residuals
            y_pred = slope * fitted_values + intercept
            
            # R² for auxiliary regression
            ss_res = np.sum((squared_residuals - y_pred)**2)
            ss_tot = np.sum((squared_residuals - y_mean)**2)
            
            if ss_tot > 0:
                r_squared = 1 - ss_res / ss_tot
                
                # BP statistic = n * R²
                bp_stat = n * r_squared
                
                # Chi-squared test with 1 degree of freedom
                p_value = 1 - stats.chi2.cdf(bp_stat, 1)
            else:
                bp_stat, p_value = 0.0, 1.0
        else:
            bp_stat, p_value = 0.0, 1.0
        
        return bp_stat, p_value
    
    def quality_metrics(self,
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       n_params: int = 2) -> QualityMetrics:
        """
        Compute comprehensive quality assessment metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            n_params: Number of parameters in model
            
        Returns:
            QualityMetrics with all quality measures
        """
        self.logger.info("Computing quality metrics")
        
        n_samples = len(y_true)
        residuals = y_true - y_pred
        
        # R² and adjusted R²
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        
        if ss_tot > 0:
            r_squared = 1 - ss_res / ss_tot
            adjusted_r_squared = 1 - (1 - r_squared) * (n_samples - 1) / (n_samples - n_params - 1)
        else:
            r_squared = 0.0
            adjusted_r_squared = 0.0
        
        # RMSE and MAE
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        
        # Residual statistics
        residual_std = np.std(residuals)
        max_residual = np.max(np.abs(residuals))
        
        # AIC and BIC
        # AIC = 2k - 2ln(L), BIC = k*ln(n) - 2ln(L)
        # For least squares: ln(L) = -n/2 * ln(2π) - n/2 * ln(σ²) - 1/(2σ²) * Σ(residuals²)
        # Simplified: AIC = n*ln(σ²) + 2k, BIC = n*ln(σ²) + k*ln(n)
        
        sigma_squared = ss_res / n_samples
        if sigma_squared > 0:
            aic = n_samples * np.log(sigma_squared) + 2 * n_params
            bic = n_samples * np.log(sigma_squared) + n_params * np.log(n_samples)
        else:
            aic = float('inf')
            bic = float('inf')
        
        # Quality grade
        if r_squared >= 0.95 and adjusted_r_squared >= 0.94:
            quality_grade = "EXCELLENT"
        elif r_squared >= 0.90 and adjusted_r_squared >= 0.88:
            quality_grade = "GOOD"
        elif r_squared >= 0.80 and adjusted_r_squared >= 0.75:
            quality_grade = "FAIR"
        else:
            quality_grade = "POOR"
        
        self.logger.info(f"  R²: {r_squared:.4f}")
        self.logger.info(f"  Adjusted R²: {adjusted_r_squared:.4f}")
        self.logger.info(f"  RMSE: {rmse:.4f}")
        self.logger.info(f"  MAE: {mae:.4f}")
        self.logger.info(f"  Quality grade: {quality_grade}")
        
        return QualityMetrics(
            r_squared=r_squared,
            adjusted_r_squared=adjusted_r_squared,
            aic=aic,
            bic=bic,
            rmse=rmse,
            mae=mae,
            residual_std=residual_std,
            max_residual=max_residual,
            quality_grade=quality_grade
        )


def create_validation_framework(n_bootstrap: int = 1000,
                                confidence_level: float = 0.95,
                                n_folds: int = 5,
                                alpha: float = 0.05,
                                parallel: bool = True) -> ValidationFramework:
    """
    Factory function to create validation framework.
    
    Args:
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        n_folds: Number of cross-validation folds
        alpha: Significance level for tests
        parallel: Whether to use parallel processing
        
    Returns:
        Configured ValidationFramework instance
    """
    return ValidationFramework(
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        n_folds=n_folds,
        alpha=alpha,
        parallel=parallel
    )
