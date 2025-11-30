"""
Bonferroni-Bootstrap Hybrid Validator combining complementary strengths.

This module implements a hybrid validation approach that combines:
- Bonferroni correction for specificity (prevents false positives)
- Bootstrap resampling for sensitivity (handles non-normal distributions)

Novel Discovery: The two methods have complementary strengths:
- Bonferroni: Conservative, controls family-wise error rate
- Bootstrap: Robust, makes no distributional assumptions
- Hybrid: Adaptive switching based on data characteristics

The hybrid approach provides both statistical rigor (Bonferroni) and
robustness to violations of normality assumptions (bootstrap).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats

from .base_types import ValidationResult
from ..utils.logging_utils import get_logger


@dataclass
class HybridValidationResult:
    """Result from hybrid validation combining Bonferroni and bootstrap.
    
    Attributes:
        hypothesis_id: ID of hypothesis validated
        validated: Whether hypothesis was validated
        confidence: Overall confidence (0.0 to 1.0)
        method_used: Which method was used ('bonferroni', 'bootstrap', 'both')
        bonferroni_results: Results from Bonferroni correction
        bootstrap_results: Results from bootstrap resampling
        agreement: Whether both methods agree
        recommendation: Human-readable recommendation
    """
    hypothesis_id: str
    validated: bool
    confidence: float
    method_used: str
    bonferroni_results: Dict[str, Any]
    bootstrap_results: Dict[str, Any]
    agreement: bool
    recommendation: str


@dataclass
class DataCharacteristics:
    """Characteristics of data used to select validation method.
    
    Attributes:
        n_samples: Number of samples
        normality_p_value: P-value from normality test
        is_normal: Whether data appears normally distributed
        has_outliers: Whether data contains outliers
        sample_size_adequate: Whether sample size is adequate for parametric tests
        recommendation: Recommended validation method
    """
    n_samples: int
    normality_p_value: float
    is_normal: bool
    has_outliers: bool
    sample_size_adequate: bool
    recommendation: str


class HybridValidator:
    """Hybrid validator combining Bonferroni and bootstrap methods.
    
    This class implements adaptive validation that:
    1. Analyzes data characteristics
    2. Selects appropriate method(s) based on characteristics
    3. Applies Bonferroni for specificity (prevent false positives)
    4. Applies bootstrap for sensitivity (handle non-normality)
    5. Combines results when both methods are used
    
    Key Innovation: Adaptive switching based on data characteristics
    ensures optimal validation strategy for each dataset.
    
    Attributes:
        alpha: Significance level (default: 0.05)
        n_bootstrap: Number of bootstrap samples (default: 1000)
        normality_threshold: P-value threshold for normality (default: 0.05)
        min_sample_size: Minimum sample size for parametric tests (default: 30)
        logger: Logger instance
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        n_bootstrap: int = 1000,
        normality_threshold: float = 0.05,
        min_sample_size: int = 30
    ):
        """Initialize hybrid validator.
        
        Args:
            alpha: Significance level for hypothesis tests
            n_bootstrap: Number of bootstrap samples
            normality_threshold: P-value threshold for normality test
            min_sample_size: Minimum sample size for parametric tests
        """
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.normality_threshold = normality_threshold
        self.min_sample_size = min_sample_size
        self.logger = get_logger(__name__)
        
        self.logger.info(
            f"Initialized HybridValidator with α={alpha}, "
            f"n_bootstrap={n_bootstrap}, normality_threshold={normality_threshold}"
        )
    
    def analyze_data_characteristics(
        self,
        data: np.ndarray
    ) -> DataCharacteristics:
        """Analyze data characteristics to select validation method.
        
        Args:
            data: Data array to analyze
            
        Returns:
            DataCharacteristics with analysis results
        """
        n_samples = len(data)
        
        # Test for normality (Shapiro-Wilk test)
        if n_samples >= 3:
            _, normality_p = stats.shapiro(data)
        else:
            normality_p = 0.0  # Too few samples
        
        is_normal = normality_p > self.normality_threshold
        
        # Check for outliers (IQR method)
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        has_outliers = np.any((data < lower_bound) | (data > upper_bound))
        
        # Check sample size
        sample_size_adequate = n_samples >= self.min_sample_size
        
        # Recommend method
        if is_normal and sample_size_adequate and not has_outliers:
            recommendation = 'bonferroni'
        elif not is_normal or has_outliers:
            recommendation = 'bootstrap'
        elif not sample_size_adequate:
            recommendation = 'both'  # Use both for extra confidence
        else:
            recommendation = 'both'
        
        characteristics = DataCharacteristics(
            n_samples=n_samples,
            normality_p_value=normality_p,
            is_normal=is_normal,
            has_outliers=has_outliers,
            sample_size_adequate=sample_size_adequate,
            recommendation=recommendation
        )
        
        self.logger.debug(
            f"Data characteristics: n={n_samples}, normal={is_normal}, "
            f"outliers={has_outliers}, recommendation={recommendation}"
        )
        
        return characteristics
    
    def validate_with_bonferroni(
        self,
        predicted_exponents: Dict[str, float],
        measured_exponents: Dict[str, float],
        measured_errors: Dict[str, float]
    ) -> Dict[str, Any]:
        """Validate using Bonferroni correction for multiple comparisons.
        
        Bonferroni correction controls family-wise error rate by adjusting
        significance level: α_adjusted = α / n_comparisons
        
        Args:
            predicted_exponents: Predicted exponent values
            measured_exponents: Measured exponent values
            measured_errors: Standard errors on measured values
            
        Returns:
            Dictionary with Bonferroni validation results
        """
        n_comparisons = len(predicted_exponents)
        alpha_adjusted = self.alpha / n_comparisons
        
        results = {
            'n_comparisons': n_comparisons,
            'alpha_original': self.alpha,
            'alpha_adjusted': alpha_adjusted,
            'exponent_results': {},
            'all_validated': True
        }
        
        for exponent_name in predicted_exponents.keys():
            if exponent_name not in measured_exponents:
                continue
            
            predicted = predicted_exponents[exponent_name]
            measured = measured_exponents[exponent_name]
            error = measured_errors.get(exponent_name, 0.05)
            
            # Z-test for difference
            z_score = abs(measured - predicted) / error
            p_value = 2 * (1 - stats.norm.cdf(z_score))
            
            # Compare against adjusted alpha
            validated = p_value > alpha_adjusted
            
            results['exponent_results'][exponent_name] = {
                'predicted': predicted,
                'measured': measured,
                'error': error,
                'z_score': z_score,
                'p_value': p_value,
                'p_value_adjusted': p_value * n_comparisons,  # Bonferroni adjustment
                'validated': validated
            }
            
            if not validated:
                results['all_validated'] = False
        
        self.logger.debug(
            f"Bonferroni validation: {n_comparisons} comparisons, "
            f"α_adjusted={alpha_adjusted:.4f}, validated={results['all_validated']}"
        )
        
        return results
    
    def validate_with_bootstrap(
        self,
        predicted_exponents: Dict[str, float],
        measured_data: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Validate using bootstrap confidence intervals.
        
        Bootstrap resampling makes no distributional assumptions and is
        robust to outliers and non-normality.
        
        Args:
            predicted_exponents: Predicted exponent values
            measured_data: Raw measurement data for each exponent
            
        Returns:
            Dictionary with bootstrap validation results
        """
        results = {
            'n_bootstrap': self.n_bootstrap,
            'alpha': self.alpha,
            'exponent_results': {},
            'all_validated': True
        }
        
        for exponent_name in predicted_exponents.keys():
            if exponent_name not in measured_data:
                continue
            
            predicted = predicted_exponents[exponent_name]
            data = measured_data[exponent_name]
            
            # Bootstrap resampling
            bootstrap_means = []
            for _ in range(self.n_bootstrap):
                resample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_means.append(np.mean(resample))
            
            bootstrap_means = np.array(bootstrap_means)
            
            # Compute confidence interval
            ci_lower = np.percentile(bootstrap_means, 100 * self.alpha / 2)
            ci_upper = np.percentile(bootstrap_means, 100 * (1 - self.alpha / 2))
            
            # Check if predicted value falls within CI
            validated = ci_lower <= predicted <= ci_upper
            
            # Compute p-value (proportion of bootstrap samples more extreme)
            measured_mean = np.mean(data)
            if predicted > measured_mean:
                p_value = np.mean(bootstrap_means >= predicted)
            else:
                p_value = np.mean(bootstrap_means <= predicted)
            p_value = 2 * min(p_value, 1 - p_value)  # Two-tailed
            
            results['exponent_results'][exponent_name] = {
                'predicted': predicted,
                'measured_mean': measured_mean,
                'measured_std': np.std(data),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': p_value,
                'validated': validated
            }
            
            if not validated:
                results['all_validated'] = False
        
        self.logger.debug(
            f"Bootstrap validation: {self.n_bootstrap} samples, "
            f"validated={results['all_validated']}"
        )
        
        return results
    
    def validate_hypothesis(
        self,
        hypothesis_id: str,
        predicted_exponents: Dict[str, float],
        measured_exponents: Dict[str, float],
        measured_errors: Dict[str, float],
        measured_data: Optional[Dict[str, np.ndarray]] = None,
        force_method: Optional[str] = None
    ) -> HybridValidationResult:
        """Validate hypothesis using hybrid approach.
        
        Automatically selects appropriate method(s) based on data characteristics,
        or uses forced method if specified.
        
        Args:
            hypothesis_id: ID of hypothesis being validated
            predicted_exponents: Predicted exponent values
            measured_exponents: Measured exponent values
            measured_errors: Standard errors on measured values
            measured_data: Optional raw measurement data for bootstrap
            force_method: Force specific method ('bonferroni', 'bootstrap', 'both', None)
            
        Returns:
            HybridValidationResult with validation outcome
        """
        self.logger.info(f"Validating hypothesis {hypothesis_id}")
        
        # Analyze data characteristics if raw data provided
        if measured_data is not None and force_method is None:
            # Analyze first exponent's data as representative
            first_exponent = list(measured_data.keys())[0]
            characteristics = self.analyze_data_characteristics(measured_data[first_exponent])
            method_to_use = characteristics.recommendation
        elif force_method is not None:
            method_to_use = force_method
        else:
            # Default to Bonferroni if no raw data
            method_to_use = 'bonferroni'
        
        # Apply validation method(s)
        bonferroni_results = None
        bootstrap_results = None
        
        if method_to_use in ['bonferroni', 'both']:
            bonferroni_results = self.validate_with_bonferroni(
                predicted_exponents,
                measured_exponents,
                measured_errors
            )
        
        if method_to_use in ['bootstrap', 'both'] and measured_data is not None:
            bootstrap_results = self.validate_with_bootstrap(
                predicted_exponents,
                measured_data
            )
        
        # Determine overall validation result
        if method_to_use == 'bonferroni':
            validated = bonferroni_results['all_validated']
            confidence = self._compute_bonferroni_confidence(bonferroni_results)
            agreement = True  # Only one method used
        elif method_to_use == 'bootstrap':
            validated = bootstrap_results['all_validated']
            confidence = self._compute_bootstrap_confidence(bootstrap_results)
            agreement = True  # Only one method used
        else:  # both
            bonf_validated = bonferroni_results['all_validated']
            boot_validated = bootstrap_results['all_validated']
            agreement = bonf_validated == boot_validated
            
            if agreement:
                validated = bonf_validated
                # Average confidence from both methods
                bonf_conf = self._compute_bonferroni_confidence(bonferroni_results)
                boot_conf = self._compute_bootstrap_confidence(bootstrap_results)
                confidence = (bonf_conf + boot_conf) / 2
            else:
                # Disagreement: Use more conservative result (Bonferroni)
                validated = bonf_validated
                confidence = self._compute_bonferroni_confidence(bonferroni_results) * 0.5
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            validated, confidence, method_to_use, agreement,
            bonferroni_results, bootstrap_results
        )
        
        result = HybridValidationResult(
            hypothesis_id=hypothesis_id,
            validated=validated,
            confidence=confidence,
            method_used=method_to_use,
            bonferroni_results=bonferroni_results or {},
            bootstrap_results=bootstrap_results or {},
            agreement=agreement,
            recommendation=recommendation
        )
        
        self.logger.info(
            f"Hypothesis {hypothesis_id}: validated={validated}, "
            f"confidence={confidence:.2%}, method={method_to_use}, agreement={agreement}"
        )
        
        return result
    
    def _compute_bonferroni_confidence(self, results: Dict[str, Any]) -> float:
        """Compute overall confidence from Bonferroni results.
        
        Args:
            results: Bonferroni validation results
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not results['exponent_results']:
            return 0.0
        
        # Average of (1 - p_value_adjusted) for all exponents
        confidences = []
        for exp_result in results['exponent_results'].values():
            p_adj = exp_result['p_value_adjusted']
            conf = 1.0 - min(p_adj, 1.0)
            confidences.append(conf)
        
        return np.mean(confidences)
    
    def _compute_bootstrap_confidence(self, results: Dict[str, Any]) -> float:
        """Compute overall confidence from bootstrap results.
        
        Args:
            results: Bootstrap validation results
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not results['exponent_results']:
            return 0.0
        
        # Average of (1 - p_value) for all exponents
        confidences = []
        for exp_result in results['exponent_results'].values():
            p_val = exp_result['p_value']
            conf = 1.0 - min(p_val, 1.0)
            confidences.append(conf)
        
        return np.mean(confidences)
    
    def _generate_recommendation(
        self,
        validated: bool,
        confidence: float,
        method_used: str,
        agreement: bool,
        bonferroni_results: Optional[Dict],
        bootstrap_results: Optional[Dict]
    ) -> str:
        """Generate human-readable recommendation.
        
        Args:
            validated: Whether hypothesis was validated
            confidence: Overall confidence
            method_used: Which method was used
            agreement: Whether methods agree (if both used)
            bonferroni_results: Bonferroni results
            bootstrap_results: Bootstrap results
            
        Returns:
            Recommendation string
        """
        if validated and confidence > 0.95:
            return "STRONG_VALIDATION: Hypothesis strongly supported by data."
        elif validated and confidence > 0.80:
            return "VALIDATED: Hypothesis supported by data with good confidence."
        elif validated and confidence > 0.60:
            return "WEAK_VALIDATION: Hypothesis marginally supported. Consider more data."
        elif not validated and method_used == 'both' and not agreement:
            return "DISAGREEMENT: Methods disagree. Bootstrap suggests validation but Bonferroni rejects. Likely due to non-normality or outliers. Trust bootstrap result."
        elif not validated and confidence < 0.20:
            return "STRONG_REFUTATION: Hypothesis strongly contradicted by data."
        elif not validated and confidence < 0.50:
            return "REFUTED: Hypothesis not supported by data."
        else:
            return "INCONCLUSIVE: Results ambiguous. Collect more data or refine hypothesis."
    
    def compare_methods(
        self,
        predicted_exponents: Dict[str, float],
        measured_exponents: Dict[str, float],
        measured_errors: Dict[str, float],
        measured_data: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Compare Bonferroni and bootstrap methods side-by-side.
        
        Useful for understanding when methods agree/disagree and why.
        
        Args:
            predicted_exponents: Predicted exponent values
            measured_exponents: Measured exponent values
            measured_errors: Standard errors on measured values
            measured_data: Raw measurement data
            
        Returns:
            Dictionary comparing both methods
        """
        # Run both methods
        bonferroni_results = self.validate_with_bonferroni(
            predicted_exponents,
            measured_exponents,
            measured_errors
        )
        
        bootstrap_results = self.validate_with_bootstrap(
            predicted_exponents,
            measured_data
        )
        
        # Analyze data characteristics
        first_exponent = list(measured_data.keys())[0]
        characteristics = self.analyze_data_characteristics(measured_data[first_exponent])
        
        # Compare results
        comparison = {
            'data_characteristics': characteristics,
            'bonferroni': bonferroni_results,
            'bootstrap': bootstrap_results,
            'agreement': bonferroni_results['all_validated'] == bootstrap_results['all_validated'],
            'exponent_comparison': {}
        }
        
        # Compare exponent by exponent
        for exp_name in predicted_exponents.keys():
            if exp_name in bonferroni_results['exponent_results'] and \
               exp_name in bootstrap_results['exponent_results']:
                bonf = bonferroni_results['exponent_results'][exp_name]
                boot = bootstrap_results['exponent_results'][exp_name]
                
                comparison['exponent_comparison'][exp_name] = {
                    'bonferroni_validated': bonf['validated'],
                    'bootstrap_validated': boot['validated'],
                    'agree': bonf['validated'] == boot['validated'],
                    'bonferroni_p_value': bonf['p_value'],
                    'bootstrap_p_value': boot['p_value'],
                    'bonferroni_more_conservative': bonf['p_value'] < boot['p_value']
                }
        
        return comparison
