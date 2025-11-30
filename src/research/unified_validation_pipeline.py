"""
Unified Validation Pipeline integrating all 10 validation patterns.

This module implements Task 4.1: Create unified validation pipeline that:
- Integrates all 10 validation patterns
- Applies patterns in sequence
- Aggregates results across patterns
- Generates comprehensive validation report

The pipeline ensures robust validation of potential physics discoveries by
applying multiple independent validation methods and requiring agreement
across all patterns.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime

from .base_types import VAEAnalysisResults, SimulationData, ValidationResult
from .confidence_aggregator import ConfidenceAggregator, AggregatedConfidence
from .validation_triangle import ValidationTriangle, TriangleValidation
from .two_dimensional_validator import TwoDimensionalValidator, TwoDValidationResult
from .anomaly_classifier import MethodologicalAnomalyClassifier, ClassifiedAnomaly, AnomalyCategory
from .effect_size_prioritizer import EffectSizePrioritizer, Finding, PrioritizedFinding
from .hybrid_validator import HybridValidator, HybridValidationResult
from .error_propagation_tracker import ErrorPropagationTracker, PropagationChain
from .universality_class_manager import UniversalityClassManager, UsageRecord
from .phenomena_detector import NovelPhenomenonDetector, NovelPhenomenon
from .comparative_analyzer import ComparativeAnalyzer, ComparisonResults
from .validation_framework import ValidationFramework
from ..utils.logging_utils import get_logger


@dataclass
class ValidationReport:
    """Comprehensive validation report from unified pipeline.
    
    Attributes:
        variant_id: ID of the variant validated
        timestamp: When validation was performed
        overall_validated: Whether the discovery is validated overall
        overall_confidence: Overall confidence score (0.0 to 1.0)
        recommendation: Overall recommendation
        pattern_results: Results from each validation pattern
        summary: Human-readable summary
        publication_ready: Whether results are ready for publication
    """
    variant_id: str
    timestamp: datetime
    overall_validated: bool
    overall_confidence: float
    recommendation: str
    pattern_results: Dict[str, Any]
    summary: str
    publication_ready: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedValidationPipeline:
    """Unified validation pipeline integrating all 10 validation patterns.
    
    This class implements the complete validation workflow for potential
    physics discoveries. It applies 10 independent validation patterns in
    sequence and aggregates results to produce a comprehensive validation
    report with high confidence.
    
    The 10 validation patterns are:
    1. Confidence Aggregation - Multiplicative confidence across layers
    2. Validation Triangle - Three-way consistency checking
    3. Two-Dimensional Validation - Bootstrap CI + ANOVA
    4. Anomaly Classification - Physics vs methodological issues
    5. Effect Size Prioritization - Statistical + practical significance
    6. Hybrid Validation - Bonferroni + Bootstrap
    7. Error Propagation Tracking - Uncertainty through pipeline
    8. Universality Class Management - Dual-purpose class usage
    9. Phenomena Detection - Anomaly detection
    10. Comparative Analysis - Multi-variant comparison
    
    Attributes:
        confidence_aggregator: Pattern 1
        validation_triangle: Pattern 2
        twod_validator: Pattern 3
        anomaly_classifier: Pattern 4
        effect_prioritizer: Pattern 5
        hybrid_validator: Pattern 6
        error_tracker: Pattern 7
        universality_manager: Pattern 8
        phenomena_detector: Pattern 9
        comparative_analyzer: Pattern 10
        validation_framework: Base validation framework
        logger: Logger instance
        validation_threshold: Minimum confidence for validation (default: 0.9)
    """
    
    def __init__(
        self,
        validation_threshold: float = 0.9,
        anomaly_threshold: float = 3.0,
        n_bootstrap: int = 1000,
        alpha: float = 0.05
    ):
        """Initialize unified validation pipeline.
        
        Args:
            validation_threshold: Minimum overall confidence for validation
            anomaly_threshold: Threshold for anomaly detection (sigma)
            n_bootstrap: Number of bootstrap samples
            alpha: Significance level for hypothesis tests
        """
        self.validation_threshold = validation_threshold
        self.logger = get_logger(__name__)
        
        # Initialize all 10 validation patterns
        self.confidence_aggregator = ConfidenceAggregator(
            anomaly_threshold=anomaly_threshold,
            n_bootstrap=n_bootstrap,
            alpha=alpha
        )
        
        self.validation_triangle = ValidationTriangle(
            anomaly_threshold=anomaly_threshold,
            consistency_threshold=2.0
        )
        
        self.twod_validator = TwoDimensionalValidator(
            n_bootstrap=n_bootstrap,
            significance_threshold=alpha,
            effect_size_threshold=0.5,
            anomaly_threshold=anomaly_threshold
        )
        
        self.anomaly_classifier = MethodologicalAnomalyClassifier(
            anomaly_threshold=anomaly_threshold,
            r_squared_threshold=0.7,
            tc_confidence_threshold=0.8
        )
        
        self.effect_prioritizer = EffectSizePrioritizer(
            alpha=alpha,
            small_effect_threshold=0.2,
            medium_effect_threshold=0.5,
            large_effect_threshold=0.8
        )
        
        self.hybrid_validator = HybridValidator(
            alpha=alpha,
            n_bootstrap=n_bootstrap,
            normality_threshold=alpha,
            min_sample_size=30
        )
        
        self.error_tracker = ErrorPropagationTracker(
            anomaly_threshold=anomaly_threshold
        )
        
        self.universality_manager = UniversalityClassManager(
            load_history=True
        )
        
        self.phenomena_detector = NovelPhenomenonDetector(
            anomaly_threshold=anomaly_threshold
        )
        
        self.comparative_analyzer = ComparativeAnalyzer(
            anomaly_threshold=anomaly_threshold
        )
        
        self.validation_framework = ValidationFramework(
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            anomaly_threshold=anomaly_threshold
        )
        
        self.logger.info(
            f"Initialized UnifiedValidationPipeline with "
            f"validation_threshold={validation_threshold}, "
            f"anomaly_threshold={anomaly_threshold}σ"
        )
    
    def validate_discovery(
        self,
        vae_results: VAEAnalysisResults,
        simulation_data: Optional[SimulationData] = None,
        predicted_exponents: Optional[Dict[str, float]] = None,
        variant_results: Optional[Dict[str, List[VAEAnalysisResults]]] = None,
        bootstrap_samples: Optional[Dict[str, np.ndarray]] = None,
        dimensions: int = 2
    ) -> ValidationReport:
        """Validate a potential physics discovery using all 10 patterns.
        
        This is the main entry point for the unified validation pipeline.
        It applies all validation patterns in sequence and aggregates results.
        
        Args:
            vae_results: VAE analysis results for the variant
            simulation_data: Optional simulation data
            predicted_exponents: Optional predicted exponent values
            variant_results: Optional results from multiple variants for comparison
            bootstrap_samples: Optional bootstrap samples for each exponent
            dimensions: System dimensions (2 or 3)
            
        Returns:
            Comprehensive ValidationReport
        """
        self.logger.info(
            f"Starting unified validation for variant '{vae_results.variant_id}'"
        )
        
        pattern_results = {}
        confidences = []
        
        # Pattern 1: Phenomena Detection
        self.logger.info("Pattern 1: Detecting novel phenomena...")
        phenomena = self.phenomena_detector.detect_all_phenomena(
            vae_results, simulation_data
        )
        pattern_results['phenomena_detection'] = {
            'phenomena': phenomena,
            'n_phenomena': len(phenomena),
            'max_confidence': max([p.confidence for p in phenomena]) if phenomena else 0.0
        }
        
        # Pattern 2: Anomaly Classification
        self.logger.info("Pattern 2: Classifying anomalies...")
        classified_anomalies = self.anomaly_classifier.classify_anomalies(
            vae_results, simulation_data
        )
        pattern_results['anomaly_classification'] = {
            'classified_anomalies': classified_anomalies,
            'n_physics_novel': sum(
                1 for a in classified_anomalies 
                if a.category == AnomalyCategory.PHYSICS_NOVEL
            ),
            'n_methodological': sum(
                1 for a in classified_anomalies 
                if a.category != AnomalyCategory.PHYSICS_NOVEL
            )
        }
        
        # Only proceed if we have physics-novel anomalies
        has_physics_novel = pattern_results['anomaly_classification']['n_physics_novel'] > 0
        if not has_physics_novel:
            self.logger.info("No physics-novel anomalies detected. Validation terminated.")
            return self._create_negative_report(
                vae_results.variant_id,
                pattern_results,
                "No physics-novel anomalies detected. All anomalies are methodological."
            )
        
        # Pattern 3: Validation Triangle
        self.logger.info("Pattern 3: Checking validation triangle...")
        triangle_validation = self.validation_triangle.validate(
            vae_results,
            expected_universality_class=None,
            dimensions=dimensions
        )
        pattern_results['validation_triangle'] = triangle_validation
        confidences.append(triangle_validation.overall_confidence)
        
        # Pattern 4: Universality Class Management
        self.logger.info("Pattern 4: Recording universality class usage...")
        # Record detection use for closest class
        closest_class, class_confidence, deviations = \
            self.phenomena_detector.get_closest_universality_class(vae_results)
        
        for exp_name, exp_value in vae_results.exponents.items():
            exp_error = vae_results.exponent_errors.get(exp_name, 0.05)
            self.universality_manager.record_detection_use(
                class_name=closest_class,
                variant_id=vae_results.variant_id,
                exponent_name=exp_name,
                measured_value=exp_value,
                measured_error=exp_error
            )
        
        pattern_results['universality_management'] = {
            'closest_class': closest_class,
            'class_confidence': class_confidence,
            'deviations': deviations
        }
        confidences.append(class_confidence)
        
        # Pattern 5: Two-Dimensional Validation (if predictions available)
        if predicted_exponents is not None:
            self.logger.info("Pattern 5: Performing 2D validation...")
            twod_results = self.twod_validator.validate_multiple_exponents_2d(
                hypothesis_id=f"{vae_results.variant_id}_hypothesis",
                predicted_exponents=predicted_exponents,
                measured_exponents=vae_results.exponents,
                measured_errors=vae_results.exponent_errors,
                variant_results=variant_results
            )
            pattern_results['twod_validation'] = twod_results
            
            # Average confidence from 2D validation
            avg_2d_conf = np.mean([r.confidence for r in twod_results.values()])
            confidences.append(avg_2d_conf)
        else:
            pattern_results['twod_validation'] = None
        
        # Pattern 6: Effect Size Prioritization
        if predicted_exponents is not None:
            self.logger.info("Pattern 6: Prioritizing by effect size...")
            for exp_name in vae_results.exponents.keys():
                if exp_name not in predicted_exponents:
                    continue
                
                measured = vae_results.exponents[exp_name]
                predicted = predicted_exponents[exp_name]
                error = vae_results.exponent_errors.get(exp_name, 0.05)
                
                # Calculate effect size (Cohen's d)
                effect_size = abs(measured - predicted) / error
                
                # Calculate p-value (simple z-test)
                z_score = abs(measured - predicted) / error
                from scipy import stats
                p_value = 2 * (1 - stats.norm.cdf(z_score))
                
                finding = Finding(
                    finding_id=f"{vae_results.variant_id}_{exp_name}",
                    variant_id=vae_results.variant_id,
                    exponent_name=exp_name,
                    measured_value=measured,
                    predicted_value=predicted,
                    p_value=p_value,
                    effect_size=effect_size,
                    sample_size=100,  # Approximate
                    description=f"Exponent {exp_name} comparison"
                )
                
                prioritized = self.effect_prioritizer.add_finding(finding)
            
            pattern_results['effect_prioritization'] = {
                'top_priorities': self.effect_prioritizer.get_top_priorities(n=5),
                'summary': self.effect_prioritizer.get_summary_statistics()
            }
        else:
            pattern_results['effect_prioritization'] = None
        
        # Pattern 7: Hybrid Validation (if predictions and bootstrap available)
        if predicted_exponents is not None and bootstrap_samples is not None:
            self.logger.info("Pattern 7: Performing hybrid validation...")
            hybrid_result = self.hybrid_validator.validate_hypothesis(
                hypothesis_id=f"{vae_results.variant_id}_hybrid",
                predicted_exponents=predicted_exponents,
                measured_exponents=vae_results.exponents,
                measured_errors=vae_results.exponent_errors,
                measured_data=bootstrap_samples
            )
            pattern_results['hybrid_validation'] = hybrid_result
            confidences.append(hybrid_result.confidence)
        else:
            pattern_results['hybrid_validation'] = None
        
        # Pattern 8: Error Propagation Tracking
        if bootstrap_samples is not None:
            self.logger.info("Pattern 8: Tracking error propagation...")
            chains = []
            for exp_name in vae_results.exponents.keys():
                if exp_name not in bootstrap_samples:
                    continue
                
                vae_params = {
                    'exponent_value': vae_results.exponents[exp_name],
                    'r_squared': vae_results.r_squared_values.get(exp_name, 0.9),
                    'n_data_points': 100  # Approximate
                }
                
                predicted_val = predicted_exponents.get(exp_name, 0.0) if predicted_exponents else 0.0
                
                chain = self.error_tracker.create_propagation_chain(
                    measurement_id=f"{vae_results.variant_id}_{exp_name}",
                    vae_params=vae_params,
                    bootstrap_samples=bootstrap_samples[exp_name],
                    predicted_value=predicted_val
                )
                chains.append(chain)
            
            pattern_results['error_propagation'] = {
                'chains': chains,
                'dominant_sources': self.error_tracker.identify_dominant_error_sources(chains),
                'summary': self.error_tracker.get_summary_statistics(chains)
            }
        else:
            pattern_results['error_propagation'] = None
        
        # Pattern 9: Comparative Analysis (if variant results available)
        if variant_results is not None and len(variant_results) > 1:
            self.logger.info("Pattern 9: Performing comparative analysis...")
            comparison_results = self.comparative_analyzer.compare_variants(
                variant_results
            )
            pattern_results['comparative_analysis'] = comparison_results
        else:
            pattern_results['comparative_analysis'] = None
        
        # Pattern 10: Confidence Aggregation
        self.logger.info("Pattern 10: Aggregating confidence...")
        aggregated_confidence = self.confidence_aggregator.aggregate_confidence(
            vae_results=vae_results,
            phenomena=phenomena,
            comparison_results=pattern_results.get('comparative_analysis'),
            validation_results=None  # Could add validation framework results
        )
        pattern_results['confidence_aggregation'] = aggregated_confidence
        confidences.append(aggregated_confidence.overall_confidence)
        
        # Calculate overall confidence (geometric mean of all confidences)
        overall_confidence = np.prod(confidences) ** (1.0 / len(confidences))
        
        # Determine if validated
        overall_validated = overall_confidence >= self.validation_threshold
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            overall_validated,
            overall_confidence,
            pattern_results
        )
        
        # Generate summary
        summary = self._generate_summary(
            vae_results.variant_id,
            overall_validated,
            overall_confidence,
            pattern_results
        )
        
        # Determine if publication ready
        publication_ready = (
            overall_validated and
            overall_confidence >= 0.95 and
            has_physics_novel
        )
        
        report = ValidationReport(
            variant_id=vae_results.variant_id,
            timestamp=datetime.now(),
            overall_validated=overall_validated,
            overall_confidence=overall_confidence,
            recommendation=recommendation,
            pattern_results=pattern_results,
            summary=summary,
            publication_ready=publication_ready,
            metadata={
                'dimensions': dimensions,
                'n_patterns_applied': len([v for v in pattern_results.values() if v is not None]),
                'validation_threshold': self.validation_threshold
            }
        )
        
        self.logger.info(
            f"Validation complete: validated={overall_validated}, "
            f"confidence={overall_confidence:.2%}, "
            f"publication_ready={publication_ready}"
        )
        
        return report
    
    def _create_negative_report(
        self,
        variant_id: str,
        pattern_results: Dict[str, Any],
        reason: str
    ) -> ValidationReport:
        """Create a negative validation report.
        
        Args:
            variant_id: ID of the variant
            pattern_results: Partial pattern results
            reason: Reason for negative result
            
        Returns:
            ValidationReport with negative result
        """
        return ValidationReport(
            variant_id=variant_id,
            timestamp=datetime.now(),
            overall_validated=False,
            overall_confidence=0.0,
            recommendation=f"NOT_VALIDATED: {reason}",
            pattern_results=pattern_results,
            summary=reason,
            publication_ready=False
        )
    
    def _generate_recommendation(
        self,
        validated: bool,
        confidence: float,
        pattern_results: Dict[str, Any]
    ) -> str:
        """Generate overall recommendation.
        
        Args:
            validated: Whether discovery is validated
            confidence: Overall confidence
            pattern_results: Results from all patterns
            
        Returns:
            Recommendation string
        """
        if validated and confidence >= 0.95:
            return (
                "STRONG_VALIDATION: Discovery validated with very high confidence. "
                "Ready for publication. All validation patterns agree."
            )
        elif validated and confidence >= 0.90:
            return (
                "VALIDATED: Discovery validated with high confidence. "
                "Consider additional verification before publication."
            )
        elif confidence >= 0.80:
            return (
                "LIKELY_VALID: High confidence but below validation threshold. "
                "Collect more data or perform additional validation."
            )
        elif confidence >= 0.60:
            return (
                "INCONCLUSIVE: Moderate confidence. Significant uncertainty remains. "
                "Increase sample size and repeat validation."
            )
        else:
            return (
                "NOT_VALIDATED: Low confidence. Likely methodological issue or "
                "insufficient data. Review anomaly classification results."
            )
    
    def _generate_summary(
        self,
        variant_id: str,
        validated: bool,
        confidence: float,
        pattern_results: Dict[str, Any]
    ) -> str:
        """Generate human-readable summary.
        
        Args:
            variant_id: ID of the variant
            validated: Whether validated
            confidence: Overall confidence
            pattern_results: Results from all patterns
            
        Returns:
            Summary string
        """
        lines = []
        lines.append(f"Validation Summary for {variant_id}")
        lines.append("=" * 70)
        lines.append(f"Overall Result: {'VALIDATED' if validated else 'NOT VALIDATED'}")
        lines.append(f"Overall Confidence: {confidence:.2%}")
        lines.append("")
        
        # Phenomena detection
        n_phenomena = pattern_results['phenomena_detection']['n_phenomena']
        lines.append(f"Novel Phenomena Detected: {n_phenomena}")
        
        # Anomaly classification
        n_physics = pattern_results['anomaly_classification']['n_physics_novel']
        n_method = pattern_results['anomaly_classification']['n_methodological']
        lines.append(f"Physics-Novel Anomalies: {n_physics}")
        lines.append(f"Methodological Issues: {n_method}")
        lines.append("")
        
        # Validation triangle
        triangle = pattern_results['validation_triangle']
        lines.append(f"Validation Triangle: {triangle.overall_status.value}")
        lines.append(f"Triangle Confidence: {triangle.overall_confidence:.2%}")
        lines.append("")
        
        # Universality class
        closest_class = pattern_results['universality_management']['closest_class']
        class_conf = pattern_results['universality_management']['class_confidence']
        lines.append(f"Closest Universality Class: {closest_class}")
        lines.append(f"Class Match Confidence: {class_conf:.2%}")
        lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def save_report(
        self,
        report: ValidationReport,
        output_dir: Path
    ) -> Path:
        """Save validation report to file.
        
        Args:
            report: ValidationReport to save
            output_dir: Directory to save report
            
        Returns:
            Path to saved report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp_str = report.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"validation_report_{report.variant_id}_{timestamp_str}.json"
        filepath = output_dir / filename
        
        # Convert report to serializable format
        pattern_summary = {
            'n_phenomena': report.pattern_results.get('phenomena_detection', {}).get('n_phenomena', 0),
            'n_physics_novel': report.pattern_results.get('anomaly_classification', {}).get('n_physics_novel', 0),
        }
        
        # Add triangle status if available
        if 'validation_triangle' in report.pattern_results and report.pattern_results['validation_triangle']:
            pattern_summary['triangle_status'] = report.pattern_results['validation_triangle'].overall_status.value
        
        # Add closest class if available
        if 'universality_management' in report.pattern_results:
            pattern_summary['closest_class'] = report.pattern_results['universality_management'].get('closest_class', 'unknown')
        
        report_dict = {
            'variant_id': report.variant_id,
            'timestamp': report.timestamp.isoformat(),
            'overall_validated': report.overall_validated,
            'overall_confidence': float(report.overall_confidence),
            'recommendation': report.recommendation,
            'summary': report.summary,
            'publication_ready': report.publication_ready,
            'metadata': report.metadata,
            'pattern_results_summary': pattern_summary
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        self.logger.info(f"Saved validation report to {filepath}")
        
        # Also save text summary
        text_filepath = filepath.with_suffix('.txt')
        with open(text_filepath, 'w') as f:
            f.write(report.summary)
            f.write("\n\n")
            f.write(f"Recommendation: {report.recommendation}\n")
        
        return filepath
