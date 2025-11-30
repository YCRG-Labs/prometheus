"""
Discovery Assessor for evaluating novelty and significance of physics discoveries.

This module implements Task 5: Implement discovery assessor that:
- Assesses whether findings constitute novel physics
- Classifies discovery types (novel universality class, exotic transition, etc.)
- Evaluates scientific significance and publication potential
- Compares with theoretical predictions

The assessor determines if validated findings represent genuine novel physics
worthy of publication or are variations of known phenomena.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

from .base_types import VAEAnalysisResults, SimulationData
from .unified_validation_pipeline import ValidationReport
from .universality_class_manager import UniversalityClassManager
from .phenomena_detector import NovelPhenomenonDetector
from ..utils.logging_utils import get_logger


@dataclass
class PhysicsDiscovery:
    """A validated physics discovery.
    
    Attributes:
        discovery_id: Unique identifier for the discovery
        variant_id: ID of the variant where discovery was made
        discovery_type: Type of discovery
        critical_exponents: Measured critical exponents
        exponent_errors: Errors on exponents
        validation_confidence: Overall validation confidence
        theoretical_comparison: Comparison with theoretical predictions
        significance: Scientific significance level
        publication_potential: Publication potential assessment
        timestamp: When discovery was assessed
        metadata: Additional metadata
    """
    discovery_id: str
    variant_id: str
    discovery_type: str
    critical_exponents: Dict[str, float]
    exponent_errors: Dict[str, float]
    validation_confidence: float
    theoretical_comparison: Dict[str, Any]
    significance: str
    publication_potential: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate discovery parameters."""
        valid_types = [
            'novel_universality_class',
            'exotic_transition',
            'anomalous_exponents',
            'crossover_behavior',
            'frustration_induced',
            'disorder_driven',
            'unknown'
        ]
        if self.discovery_type not in valid_types:
            raise ValueError(f"Invalid discovery type: {self.discovery_type}")
        
        valid_significance = ['major', 'moderate', 'minor']
        if self.significance not in valid_significance:
            raise ValueError(f"Invalid significance: {self.significance}")
        
        valid_potential = ['high', 'medium', 'low']
        if self.publication_potential not in valid_potential:
            raise ValueError(f"Invalid publication potential: {self.publication_potential}")


class DiscoveryAssessor:
    """Assess whether validated findings constitute novel physics.
    
    This class evaluates validation reports to determine if findings represent
    genuine novel physics discoveries worthy of publication. It:
    - Compares with all known universality classes
    - Classifies the type of discovery
    - Assesses scientific significance
    - Evaluates publication potential
    - Compares with theoretical predictions
    
    Attributes:
        universality_manager: Manager for universality class database
        phenomena_detector: Detector for novel phenomena
        novelty_threshold: Minimum deviation (sigma) to claim novelty
        confidence_threshold: Minimum confidence for discovery claim
        logger: Logger instance
    """
    
    def __init__(
        self,
        novelty_threshold: float = 3.0,
        confidence_threshold: float = 0.90,
        universality_manager: Optional[UniversalityClassManager] = None
    ):
        """Initialize discovery assessor.
        
        Args:
            novelty_threshold: Minimum deviation (sigma) from known classes
            confidence_threshold: Minimum validation confidence for discovery
            universality_manager: Optional universality class manager
        """
        self.novelty_threshold = novelty_threshold
        self.confidence_threshold = confidence_threshold
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.universality_manager = universality_manager or UniversalityClassManager()
        self.phenomena_detector = NovelPhenomenonDetector(
            anomaly_threshold=novelty_threshold
        )
        
        self.logger.info(
            f"Initialized DiscoveryAssessor with "
            f"novelty_threshold={novelty_threshold}σ, "
            f"confidence_threshold={confidence_threshold}"
        )
    
    def assess_novelty(
        self,
        vae_results: VAEAnalysisResults,
        validation_report: ValidationReport,
        theoretical_predictions: Optional[Dict[str, float]] = None,
        variant_description: Optional[str] = None
    ) -> Optional[PhysicsDiscovery]:
        """Assess whether findings constitute novel physics.
        
        This is the main entry point for discovery assessment. It evaluates
        validation results to determine if they represent genuine novel physics.
        
        Args:
            vae_results: VAE analysis results
            validation_report: Validation report from unified pipeline
            theoretical_predictions: Optional theoretical predictions
            variant_description: Optional description of the variant
            
        Returns:
            PhysicsDiscovery if novel physics is found, None otherwise
        """
        self.logger.info(
            f"Assessing novelty for variant '{vae_results.variant_id}'"
        )
        
        # Check if validation passed
        if not validation_report.overall_validated:
            self.logger.info("Validation failed. No discovery to assess.")
            return None
        
        # Check confidence threshold
        if validation_report.overall_confidence < self.confidence_threshold:
            self.logger.info(
                f"Confidence {validation_report.overall_confidence:.2%} below "
                f"threshold {self.confidence_threshold:.2%}. No discovery claimed."
            )
            return None
        
        # Check for physics-novel anomalies
        anomaly_results = validation_report.pattern_results.get('anomaly_classification', {})
        n_physics_novel = anomaly_results.get('n_physics_novel', 0)
        
        if n_physics_novel == 0:
            self.logger.info("No physics-novel anomalies. No discovery claimed.")
            return None
        
        # Compare with all known universality classes
        closest_class, class_confidence, deviations = \
            self.phenomena_detector.get_closest_universality_class(vae_results)
        
        # Check if deviations exceed novelty threshold
        max_deviation = max(deviations.values()) if deviations else 0.0
        
        if max_deviation < self.novelty_threshold:
            self.logger.info(
                f"Maximum deviation {max_deviation:.2f}σ below novelty threshold "
                f"{self.novelty_threshold}σ. Matches known class '{closest_class}'."
            )
            return None
        
        # Novel physics detected!
        self.logger.info(
            f"Novel physics detected! Maximum deviation: {max_deviation:.2f}σ "
            f"from closest class '{closest_class}'"
        )
        
        # Classify discovery type
        discovery_type = self.classify_discovery_type(
            vae_results,
            validation_report,
            closest_class,
            deviations
        )
        
        # Compare with theoretical predictions
        theoretical_comparison = self._compare_with_theory(
            vae_results,
            theoretical_predictions,
            closest_class,
            deviations
        )
        
        # Assess significance
        significance = self.assess_significance(
            vae_results,
            validation_report,
            max_deviation,
            theoretical_comparison
        )
        
        # Evaluate publication potential
        publication_potential = self._evaluate_publication_potential(
            validation_report,
            significance,
            theoretical_comparison
        )
        
        # Create discovery object
        discovery_id = f"{vae_results.variant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        discovery = PhysicsDiscovery(
            discovery_id=discovery_id,
            variant_id=vae_results.variant_id,
            discovery_type=discovery_type,
            critical_exponents=vae_results.exponents,
            exponent_errors=vae_results.exponent_errors,
            validation_confidence=validation_report.overall_confidence,
            theoretical_comparison=theoretical_comparison,
            significance=significance,
            publication_potential=publication_potential,
            metadata={
                'closest_universality_class': closest_class,
                'max_deviation_sigma': float(max_deviation),
                'deviations': {k: float(v) for k, v in deviations.items()},
                'critical_temperature': float(vae_results.critical_temperature),
                'tc_confidence': float(vae_results.tc_confidence),
                'variant_description': variant_description or '',
                'n_physics_novel_anomalies': n_physics_novel
            }
        )
        
        self.logger.info(
            f"Discovery assessed: type={discovery_type}, "
            f"significance={significance}, "
            f"publication_potential={publication_potential}"
        )
        
        return discovery
    
    def classify_discovery_type(
        self,
        vae_results: VAEAnalysisResults,
        validation_report: ValidationReport,
        closest_class: str,
        deviations: Dict[str, float]
    ) -> str:
        """Classify the type of physics discovery.
        
        Args:
            vae_results: VAE analysis results
            validation_report: Validation report
            closest_class: Closest known universality class
            deviations: Deviations from closest class
            
        Returns:
            Discovery type string
        """
        variant_id = vae_results.variant_id.lower()
        
        # Check for novel universality class
        # All exponents deviate significantly from all known classes
        all_deviate = all(dev > self.novelty_threshold for dev in deviations.values())
        
        if all_deviate and len(deviations) >= 2:
            return 'novel_universality_class'
        
        # Check for exotic transition
        # Look for first-order or unusual transition characteristics
        phenomena = validation_report.pattern_results.get('phenomena_detection', {}).get('phenomena', [])
        for phenomenon in phenomena:
            if phenomenon.phenomenon_type == 'first_order':
                return 'exotic_transition'
        
        # Check for crossover behavior
        if 'long-range' in variant_id or 'crossover' in variant_id:
            return 'crossover_behavior'
        
        # Check for frustration-induced phenomena
        if any(geom in variant_id for geom in ['triangular', 'kagome', 'frustrated']):
            return 'frustration_induced'
        
        # Check for disorder-driven phenomena
        if any(disorder in variant_id for disorder in ['diluted', 'random', 'disorder']):
            return 'disorder_driven'
        
        # Check for anomalous exponents
        # Some exponents deviate but not all
        if any(dev > self.novelty_threshold for dev in deviations.values()):
            return 'anomalous_exponents'
        
        return 'unknown'
    
    def assess_significance(
        self,
        vae_results: VAEAnalysisResults,
        validation_report: ValidationReport,
        max_deviation: float,
        theoretical_comparison: Dict[str, Any]
    ) -> str:
        """Assess scientific significance of the discovery.
        
        Significance is based on:
        - Magnitude of deviation from known physics
        - Validation confidence
        - Theoretical implications
        - Novelty of the finding
        
        Args:
            vae_results: VAE analysis results
            validation_report: Validation report
            max_deviation: Maximum deviation from known classes
            theoretical_comparison: Comparison with theory
            
        Returns:
            Significance level: 'major', 'moderate', or 'minor'
        """
        confidence = validation_report.overall_confidence
        
        # Major significance criteria:
        # - Very high deviation (>5σ) from all known classes
        # - Very high validation confidence (>95%)
        # - Refutes or extends existing theory
        # - Novel universality class
        
        is_novel_class = (
            validation_report.pattern_results.get('phenomena_detection', {})
            .get('n_phenomena', 0) > 0
        )
        
        refutes_theory = theoretical_comparison.get('status') == 'refuted'
        extends_theory = theoretical_comparison.get('status') == 'extended'
        
        if (max_deviation > 5.0 and 
            confidence > 0.95 and 
            (refutes_theory or is_novel_class)):
            return 'major'
        
        # Moderate significance criteria:
        # - Significant deviation (3-5σ)
        # - High validation confidence (90-95%)
        # - Confirms or extends theory
        # - Anomalous exponents or exotic transition
        
        if (max_deviation > 3.0 and 
            confidence > 0.90 and 
            (extends_theory or is_novel_class)):
            return 'moderate'
        
        # Minor significance:
        # - Modest deviation (just above threshold)
        # - Adequate validation confidence
        # - Incremental advance
        
        return 'minor'
    
    def _compare_with_theory(
        self,
        vae_results: VAEAnalysisResults,
        theoretical_predictions: Optional[Dict[str, float]],
        closest_class: str,
        deviations: Dict[str, float]
    ) -> Dict[str, Any]:
        """Compare findings with theoretical predictions.
        
        Args:
            vae_results: VAE analysis results
            theoretical_predictions: Theoretical predictions (if available)
            closest_class: Closest known universality class
            deviations: Deviations from closest class
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {
            'has_predictions': theoretical_predictions is not None,
            'closest_class': closest_class,
            'deviations': deviations,
            'status': 'no_predictions'
        }
        
        if theoretical_predictions is None:
            comparison['message'] = (
                f"No theoretical predictions available. "
                f"Closest known class is '{closest_class}' with "
                f"maximum deviation {max(deviations.values()):.2f}σ."
            )
            return comparison
        
        # Compare with predictions
        agreements = []
        disagreements = []
        
        for exp_name, predicted_value in theoretical_predictions.items():
            if exp_name not in vae_results.exponents:
                continue
            
            measured_value = vae_results.exponents[exp_name]
            measured_error = vae_results.exponent_errors.get(exp_name, 0.05)
            
            # Calculate deviation
            deviation = abs(measured_value - predicted_value) / measured_error
            
            if deviation <= 2.0:
                agreements.append({
                    'exponent': exp_name,
                    'predicted': predicted_value,
                    'measured': measured_value,
                    'error': measured_error,
                    'deviation_sigma': deviation
                })
            else:
                disagreements.append({
                    'exponent': exp_name,
                    'predicted': predicted_value,
                    'measured': measured_value,
                    'error': measured_error,
                    'deviation_sigma': deviation
                })
        
        comparison['agreements'] = agreements
        comparison['disagreements'] = disagreements
        
        # Determine status
        if len(disagreements) == 0 and len(agreements) > 0:
            comparison['status'] = 'confirmed'
            comparison['message'] = (
                f"All {len(agreements)} predicted exponents confirmed within 2σ."
            )
        elif len(disagreements) > 0 and len(agreements) > 0:
            comparison['status'] = 'partial'
            comparison['message'] = (
                f"{len(agreements)} exponents confirmed, "
                f"{len(disagreements)} exponents deviate from predictions."
            )
        elif len(disagreements) > 0 and len(agreements) == 0:
            comparison['status'] = 'refuted'
            comparison['message'] = (
                f"All {len(disagreements)} predicted exponents refuted. "
                f"Significant deviation from theory."
            )
        else:
            comparison['status'] = 'extended'
            comparison['message'] = (
                "Measurements extend beyond theoretical predictions. "
                "New physics not covered by existing theory."
            )
        
        return comparison
    
    def _evaluate_publication_potential(
        self,
        validation_report: ValidationReport,
        significance: str,
        theoretical_comparison: Dict[str, Any]
    ) -> str:
        """Evaluate publication potential of the discovery.
        
        Args:
            validation_report: Validation report
            significance: Scientific significance
            theoretical_comparison: Comparison with theory
            
        Returns:
            Publication potential: 'high', 'medium', or 'low'
        """
        confidence = validation_report.overall_confidence
        theory_status = theoretical_comparison.get('status', 'no_predictions')
        
        # High publication potential:
        # - Major significance
        # - Very high confidence (>95%)
        # - Refutes or extends theory
        # - Publication ready
        
        if (significance == 'major' and 
            confidence > 0.95 and 
            theory_status in ['refuted', 'extended'] and
            validation_report.publication_ready):
            return 'high'
        
        # Medium publication potential:
        # - Moderate significance
        # - High confidence (>90%)
        # - Confirms or partially agrees with theory
        
        if (significance in ['major', 'moderate'] and 
            confidence > 0.90 and
            theory_status in ['confirmed', 'partial', 'extended']):
            return 'medium'
        
        # Low publication potential:
        # - Minor significance
        # - Adequate confidence
        # - Incremental advance
        
        return 'low'
    
    def generate_discovery_summary(
        self,
        discovery: PhysicsDiscovery
    ) -> str:
        """Generate human-readable summary of the discovery.
        
        Args:
            discovery: PhysicsDiscovery object
            
        Returns:
            Summary string
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"PHYSICS DISCOVERY: {discovery.discovery_id}")
        lines.append("=" * 70)
        lines.append("")
        
        lines.append(f"Variant: {discovery.variant_id}")
        lines.append(f"Discovery Type: {discovery.discovery_type.replace('_', ' ').title()}")
        lines.append(f"Significance: {discovery.significance.upper()}")
        lines.append(f"Publication Potential: {discovery.publication_potential.upper()}")
        lines.append(f"Validation Confidence: {discovery.validation_confidence:.2%}")
        lines.append("")
        
        lines.append("Critical Exponents:")
        for exp_name, exp_value in discovery.critical_exponents.items():
            exp_error = discovery.exponent_errors.get(exp_name, 0.0)
            lines.append(f"  {exp_name}: {exp_value:.4f} ± {exp_error:.4f}")
        lines.append("")
        
        lines.append("Comparison with Known Physics:")
        closest_class = discovery.metadata.get('closest_universality_class', 'unknown')
        max_dev = discovery.metadata.get('max_deviation_sigma', 0.0)
        lines.append(f"  Closest Universality Class: {closest_class}")
        lines.append(f"  Maximum Deviation: {max_dev:.2f}σ")
        lines.append("")
        
        theory_comp = discovery.theoretical_comparison
        lines.append("Theoretical Comparison:")
        lines.append(f"  Status: {theory_comp.get('status', 'unknown').upper()}")
        lines.append(f"  {theory_comp.get('message', 'No comparison available')}")
        lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def save_discovery(
        self,
        discovery: PhysicsDiscovery,
        output_dir: Path
    ) -> Path:
        """Save discovery to file.
        
        Args:
            discovery: PhysicsDiscovery to save
            output_dir: Directory to save discovery
            
        Returns:
            Path to saved discovery file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        filename = f"discovery_{discovery.discovery_id}.json"
        filepath = output_dir / filename
        
        # Convert to serializable format
        discovery_dict = {
            'discovery_id': discovery.discovery_id,
            'variant_id': discovery.variant_id,
            'discovery_type': discovery.discovery_type,
            'critical_exponents': {k: float(v) for k, v in discovery.critical_exponents.items()},
            'exponent_errors': {k: float(v) for k, v in discovery.exponent_errors.items()},
            'validation_confidence': float(discovery.validation_confidence),
            'theoretical_comparison': discovery.theoretical_comparison,
            'significance': discovery.significance,
            'publication_potential': discovery.publication_potential,
            'timestamp': discovery.timestamp.isoformat(),
            'metadata': discovery.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(discovery_dict, f, indent=2)
        
        self.logger.info(f"Saved discovery to {filepath}")
        
        # Also save text summary
        text_filepath = filepath.with_suffix('.txt')
        with open(text_filepath, 'w') as f:
            f.write(self.generate_discovery_summary(discovery))
        
        return filepath
    
    def load_discovery(
        self,
        filepath: Path
    ) -> PhysicsDiscovery:
        """Load discovery from file.
        
        Args:
            filepath: Path to discovery file
            
        Returns:
            PhysicsDiscovery object
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        discovery = PhysicsDiscovery(
            discovery_id=data['discovery_id'],
            variant_id=data['variant_id'],
            discovery_type=data['discovery_type'],
            critical_exponents=data['critical_exponents'],
            exponent_errors=data['exponent_errors'],
            validation_confidence=data['validation_confidence'],
            theoretical_comparison=data['theoretical_comparison'],
            significance=data['significance'],
            publication_potential=data['publication_potential'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )
        
        self.logger.info(f"Loaded discovery from {filepath}")
        
        return discovery
