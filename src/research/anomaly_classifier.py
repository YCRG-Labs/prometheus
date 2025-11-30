"""
Methodological Anomaly Classifier

This module implements classification of anomalies into physics-based vs
methodological categories. This enables robust discrimination between genuine
novel physics phenomena and data quality/methodological issues.

Novel Methodological Contribution (Task 16.6):
The classifier distinguishes four categories of anomalies:
1. physics_novel - Genuine novel physics phenomena
2. data_quality - Poor data quality issues
3. convergence - VAE training or MC convergence problems
4. finite_size_scaling - Finite-size effects not properly accounted for

This prevents false positives from methodological issues being mistaken for
novel physics discoveries.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .base_types import NovelPhenomenon, VAEAnalysisResults, SimulationData
from .phenomena_detector import NovelPhenomenonDetector
from ..utils.logging_utils import get_logger


class AnomalyCategory(Enum):
    """Categories of anomalies."""
    PHYSICS_NOVEL = "physics_novel"
    DATA_QUALITY = "data_quality"
    CONVERGENCE = "convergence"
    FINITE_SIZE_SCALING = "finite_size_scaling"
    UNKNOWN = "unknown"


@dataclass
class ClassifiedAnomaly:
    """Anomaly with classification and recommendations.
    
    Attributes:
        phenomenon: The detected novel phenomenon
        category: Classification category
        confidence: Confidence in classification (0.0 to 1.0)
        evidence: Evidence supporting the classification
        recommendation: Specific recommendation for addressing the issue
        severity: Severity level (low, medium, high)
    """
    phenomenon: NovelPhenomenon
    category: AnomalyCategory
    confidence: float
    evidence: Dict[str, any]
    recommendation: str
    severity: str
    
    def __post_init__(self):
        """Validate classified anomaly."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        
        valid_severities = ['low', 'medium', 'high']
        if self.severity not in valid_severities:
            raise ValueError(f"Invalid severity: {self.severity}")


class MethodologicalAnomalyClassifier:
    """Classifier for distinguishing physics vs methodological anomalies.
    
    This class extends the NovelPhenomenonDetector with classification logic
    to distinguish between genuine novel physics phenomena and methodological
    issues that can produce anomalous-looking results.
    
    Classification Categories:
    1. **physics_novel**: Genuine novel physics phenomena
       - Consistent anomalies across multiple checks
       - High data quality and convergence
       - Scaling relations satisfied
       
    2. **data_quality**: Poor data quality issues
       - Low R² values in fits
       - High uncertainty in measurements
       - Inconsistent results
       
    3. **convergence**: VAE training or MC convergence problems
       - Poor VAE reconstruction quality
       - Low Tc confidence
       - Training instabilities
       
    4. **finite_size_scaling**: Finite-size effects
       - Anomalies that scale with system size
       - Violations of finite-size scaling relations
       - Size-dependent exponents
    
    Attributes:
        detector: NovelPhenomenonDetector instance
        r_squared_threshold: Minimum R² for good data quality
        tc_confidence_threshold: Minimum Tc confidence for convergence
        logger: Logger instance
    """
    
    def __init__(
        self,
        anomaly_threshold: float = 3.0,
        r_squared_threshold: float = 0.7,
        tc_confidence_threshold: float = 0.8
    ):
        """Initialize methodological anomaly classifier.
        
        Args:
            anomaly_threshold: Threshold for anomaly detection (sigma)
            r_squared_threshold: Minimum R² for good data quality
            tc_confidence_threshold: Minimum Tc confidence for convergence
        """
        self.detector = NovelPhenomenonDetector(anomaly_threshold)
        self.r_squared_threshold = r_squared_threshold
        self.tc_confidence_threshold = tc_confidence_threshold
        self.logger = get_logger(__name__)
        
        self.logger.info(
            f"Initialized MethodologicalAnomalyClassifier with "
            f"R²≥{r_squared_threshold}, Tc_conf≥{tc_confidence_threshold}"
        )
    
    def classify_anomalies(
        self,
        vae_results: VAEAnalysisResults,
        simulation_data: Optional[SimulationData] = None
    ) -> List[ClassifiedAnomaly]:
        """Classify all detected anomalies.
        
        Args:
            vae_results: VAE analysis results
            simulation_data: Optional simulation data for additional checks
            
        Returns:
            List of classified anomalies with recommendations
        """
        # Detect phenomena using base detector
        phenomena = self.detector.detect_all_phenomena(vae_results, simulation_data)
        
        if not phenomena:
            self.logger.info("No anomalies detected - no classification needed")
            return []
        
        # Classify each phenomenon
        classified = []
        for phenomenon in phenomena:
            classification = self._classify_single_anomaly(
                phenomenon, vae_results, simulation_data
            )
            classified.append(classification)
        
        # Log summary
        self.logger.info(f"Classified {len(classified)} anomalies:")
        for cat in AnomalyCategory:
            count = sum(1 for c in classified if c.category == cat)
            if count > 0:
                self.logger.info(f"  - {cat.value}: {count}")
        
        return classified
    
    def _classify_single_anomaly(
        self,
        phenomenon: NovelPhenomenon,
        vae_results: VAEAnalysisResults,
        simulation_data: Optional[SimulationData]
    ) -> ClassifiedAnomaly:
        """Classify a single anomaly.
        
        Args:
            phenomenon: Detected phenomenon
            vae_results: VAE analysis results
            simulation_data: Optional simulation data
            
        Returns:
            Classified anomaly with category and recommendation
        """
        # Collect evidence from multiple sources
        evidence = self._collect_evidence(phenomenon, vae_results, simulation_data)
        
        # Apply classification logic
        category, confidence = self._determine_category(evidence)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(category, evidence)
        
        # Determine severity
        severity = self._determine_severity(category, evidence)
        
        return ClassifiedAnomaly(
            phenomenon=phenomenon,
            category=category,
            confidence=confidence,
            evidence=evidence,
            recommendation=recommendation,
            severity=severity
        )
    
    def _collect_evidence(
        self,
        phenomenon: NovelPhenomenon,
        vae_results: VAEAnalysisResults,
        simulation_data: Optional[SimulationData]
    ) -> Dict[str, any]:
        """Collect evidence for classification.
        
        Args:
            phenomenon: Detected phenomenon
            vae_results: VAE analysis results
            simulation_data: Optional simulation data
            
        Returns:
            Dictionary of evidence
        """
        evidence = {}
        
        # Data quality indicators
        r_squared_values = list(vae_results.r_squared_values.values())
        evidence['avg_r_squared'] = np.mean(r_squared_values)
        evidence['min_r_squared'] = np.min(r_squared_values)
        evidence['data_quality_good'] = evidence['min_r_squared'] >= self.r_squared_threshold
        
        # Convergence indicators
        evidence['tc_confidence'] = vae_results.tc_confidence
        evidence['convergence_good'] = vae_results.tc_confidence >= self.tc_confidence_threshold
        
        # Exponent consistency
        exponent_errors = list(vae_results.exponent_errors.values())
        evidence['avg_exponent_error'] = np.mean(exponent_errors)
        evidence['max_exponent_error'] = np.max(exponent_errors)
        evidence['errors_reasonable'] = evidence['max_exponent_error'] < 0.1
        
        # Scaling relation checks
        if 'beta' in vae_results.exponents and 'nu' in vae_results.exponents:
            beta = vae_results.exponents['beta']
            nu = vae_results.exponents['nu']
            gamma = vae_results.exponents.get('gamma', 2 * beta * nu)  # Estimate if missing
            
            # Hyperscaling: alpha + 2*beta + gamma = 2
            # Estimate alpha from alpha = 2 - d*nu (assume d=2 or 3)
            # Use d=2 as default
            alpha = 2 - 2 * nu
            hyperscaling = alpha + 2 * beta + gamma
            evidence['hyperscaling_value'] = hyperscaling
            evidence['hyperscaling_deviation'] = abs(hyperscaling - 2.0)
            evidence['scaling_satisfied'] = evidence['hyperscaling_deviation'] < 0.2
        else:
            evidence['scaling_satisfied'] = None
        
        # Phenomenon-specific evidence
        evidence['phenomenon_type'] = phenomenon.phenomenon_type
        evidence['phenomenon_confidence'] = phenomenon.confidence
        
        # Finite-size indicators (if simulation data available)
        if simulation_data is not None:
            lattice_shape = simulation_data.configurations.shape[2:]
            evidence['lattice_size'] = lattice_shape[0] if lattice_shape else None
            evidence['has_simulation_data'] = True
        else:
            evidence['lattice_size'] = None
            evidence['has_simulation_data'] = False
        
        return evidence
    
    def _determine_category(
        self,
        evidence: Dict[str, any]
    ) -> Tuple[AnomalyCategory, float]:
        """Determine anomaly category from evidence.
        
        Uses a decision tree approach to classify anomalies based on
        multiple evidence sources.
        
        Args:
            evidence: Collected evidence dictionary
            
        Returns:
            Tuple of (category, confidence)
        """
        # Decision tree for classification
        
        # Check 1: Data quality
        if not evidence['data_quality_good']:
            # Poor R² values indicate data quality issues
            confidence = 1.0 - evidence['min_r_squared'] / self.r_squared_threshold
            return AnomalyCategory.DATA_QUALITY, min(0.95, confidence)
        
        # Check 2: Convergence
        if not evidence['convergence_good']:
            # Low Tc confidence indicates convergence issues
            confidence = 1.0 - evidence['tc_confidence'] / self.tc_confidence_threshold
            return AnomalyCategory.CONVERGENCE, min(0.95, confidence)
        
        # Check 3: Scaling relations
        if evidence['scaling_satisfied'] is not None and not evidence['scaling_satisfied']:
            # Scaling violations often indicate finite-size effects
            # But could also be novel physics
            if evidence['lattice_size'] is not None and evidence['lattice_size'] < 64:
                # Small lattice → likely finite-size effects
                confidence = 0.8
                return AnomalyCategory.FINITE_SIZE_SCALING, confidence
            else:
                # Large lattice → could be novel physics
                confidence = 0.6
                return AnomalyCategory.PHYSICS_NOVEL, confidence
        
        # Check 4: Error magnitudes
        if not evidence['errors_reasonable']:
            # Large errors indicate data quality or convergence issues
            confidence = 0.7
            return AnomalyCategory.DATA_QUALITY, confidence
        
        # Check 5: All checks passed → likely novel physics
        # Confidence based on phenomenon confidence and data quality
        confidence = (
            evidence['phenomenon_confidence'] * 0.5 +
            evidence['avg_r_squared'] * 0.3 +
            evidence['tc_confidence'] * 0.2
        )
        
        return AnomalyCategory.PHYSICS_NOVEL, min(0.95, confidence)
    
    def _generate_recommendation(
        self,
        category: AnomalyCategory,
        evidence: Dict[str, any]
    ) -> str:
        """Generate specific recommendation based on category.
        
        Args:
            category: Anomaly category
            evidence: Evidence dictionary
            
        Returns:
            Recommendation string
        """
        if category == AnomalyCategory.DATA_QUALITY:
            if evidence['min_r_squared'] < 0.5:
                return (
                    "CRITICAL: Very poor fit quality (R²<0.5). "
                    "Increase sampling: more temperatures, more MC samples, longer equilibration. "
                    "Check for systematic errors in data generation."
                )
            else:
                return (
                    "Moderate data quality issues (R²<0.7). "
                    "Increase sampling density near critical temperature. "
                    "Consider longer equilibration times."
                )
        
        elif category == AnomalyCategory.CONVERGENCE:
            if evidence['tc_confidence'] < 0.5:
                return (
                    "CRITICAL: Poor Tc detection (confidence<0.5). "
                    "Retrain VAE with more epochs, adjust learning rate. "
                    "Check VAE architecture is appropriate for system. "
                    "Verify temperature range covers critical region."
                )
            else:
                return (
                    "Moderate convergence issues (Tc confidence<0.8). "
                    "Retrain VAE with adjusted hyperparameters. "
                    "Consider ensemble training for robustness."
                )
        
        elif category == AnomalyCategory.FINITE_SIZE_SCALING:
            if evidence['lattice_size'] is not None and evidence['lattice_size'] < 32:
                return (
                    "CRITICAL: Very small lattice size (<32). "
                    "Increase lattice size to at least 64×64 for 2D or 32³ for 3D. "
                    "Apply finite-size scaling corrections. "
                    "Extrapolate to thermodynamic limit."
                )
            else:
                return (
                    "Finite-size effects detected. "
                    "Perform finite-size scaling analysis with multiple lattice sizes. "
                    "Extrapolate exponents to infinite size limit. "
                    "Check for logarithmic corrections."
                )
        
        elif category == AnomalyCategory.PHYSICS_NOVEL:
            return (
                "POTENTIAL NOVEL PHYSICS: All methodological checks passed. "
                "Verify with independent methods: "
                "1) Repeat with different lattice sizes "
                "2) Use alternative analysis methods (e.g., Binder cumulant) "
                "3) Check against literature for similar systems "
                "4) Perform detailed finite-size scaling analysis "
                "5) Consider submitting for peer review if confirmed"
            )
        
        else:  # UNKNOWN
            return (
                "Unable to classify anomaly definitively. "
                "Perform comprehensive diagnostics: "
                "Check data quality, convergence, and finite-size effects. "
                "Consult with domain experts."
            )
    
    def _determine_severity(
        self,
        category: AnomalyCategory,
        evidence: Dict[str, any]
    ) -> str:
        """Determine severity level.
        
        Args:
            category: Anomaly category
            evidence: Evidence dictionary
            
        Returns:
            Severity level ('low', 'medium', 'high')
        """
        if category == AnomalyCategory.PHYSICS_NOVEL:
            # Novel physics is high priority for investigation
            return 'high'
        
        elif category == AnomalyCategory.DATA_QUALITY:
            if evidence['min_r_squared'] < 0.5:
                return 'high'  # Critical data quality issues
            else:
                return 'medium'  # Moderate issues
        
        elif category == AnomalyCategory.CONVERGENCE:
            if evidence['tc_confidence'] < 0.5:
                return 'high'  # Critical convergence issues
            else:
                return 'medium'  # Moderate issues
        
        elif category == AnomalyCategory.FINITE_SIZE_SCALING:
            if evidence['lattice_size'] is not None and evidence['lattice_size'] < 32:
                return 'high'  # Very small lattice
            else:
                return 'medium'  # Moderate finite-size effects
        
        else:  # UNKNOWN
            return 'medium'
    
    def generate_classification_report(
        self,
        classified_anomalies: List[ClassifiedAnomaly]
    ) -> str:
        """Generate human-readable classification report.
        
        Args:
            classified_anomalies: List of classified anomalies
            
        Returns:
            Formatted report string
        """
        if not classified_anomalies:
            return "No anomalies detected."
        
        lines = []
        lines.append("=" * 80)
        lines.append("ANOMALY CLASSIFICATION REPORT")
        lines.append("=" * 80)
        lines.append(f"\nTotal anomalies: {len(classified_anomalies)}")
        
        # Summary by category
        lines.append("\nSummary by Category:")
        for cat in AnomalyCategory:
            count = sum(1 for c in classified_anomalies if c.category == cat)
            if count > 0:
                lines.append(f"  - {cat.value}: {count}")
        
        # Summary by severity
        lines.append("\nSummary by Severity:")
        for severity in ['high', 'medium', 'low']:
            count = sum(1 for c in classified_anomalies if c.severity == severity)
            if count > 0:
                lines.append(f"  - {severity}: {count}")
        
        # Detailed classifications
        lines.append("\n" + "=" * 80)
        lines.append("DETAILED CLASSIFICATIONS")
        lines.append("=" * 80)
        
        for i, classified in enumerate(classified_anomalies, 1):
            lines.append(f"\n{i}. {classified.phenomenon.phenomenon_type.upper()}")
            lines.append(f"   Category: {classified.category.value}")
            lines.append(f"   Confidence: {classified.confidence:.2%}")
            lines.append(f"   Severity: {classified.severity.upper()}")
            lines.append(f"   Description: {classified.phenomenon.description}")
            lines.append(f"\n   Evidence:")
            for key, value in classified.evidence.items():
                if isinstance(value, float):
                    lines.append(f"     - {key}: {value:.4f}")
                else:
                    lines.append(f"     - {key}: {value}")
            lines.append(f"\n   Recommendation:")
            # Wrap recommendation text
            rec_lines = classified.recommendation.split('. ')
            for rec_line in rec_lines:
                if rec_line:
                    lines.append(f"     {rec_line}.")
        
        # Priority actions
        lines.append("\n" + "=" * 80)
        lines.append("PRIORITY ACTIONS")
        lines.append("=" * 80)
        
        high_severity = [c for c in classified_anomalies if c.severity == 'high']
        if high_severity:
            lines.append(f"\n{len(high_severity)} HIGH PRIORITY issues require immediate attention:")
            for c in high_severity:
                lines.append(f"\n  - {c.category.value.upper()}: {c.phenomenon.description}")
                lines.append(f"    Action: {c.recommendation.split('.')[0]}.")
        else:
            lines.append("\nNo high priority issues detected.")
        
        # Novel physics candidates
        novel_physics = [c for c in classified_anomalies if c.category == AnomalyCategory.PHYSICS_NOVEL]
        if novel_physics:
            lines.append("\n" + "=" * 80)
            lines.append("NOVEL PHYSICS CANDIDATES")
            lines.append("=" * 80)
            lines.append(f"\n{len(novel_physics)} potential novel physics phenomena detected:")
            for c in novel_physics:
                lines.append(f"\n  - {c.phenomenon.description}")
                lines.append(f"    Confidence: {c.confidence:.2%}")
                lines.append(f"    Parameters: {c.phenomenon.parameters}")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)

