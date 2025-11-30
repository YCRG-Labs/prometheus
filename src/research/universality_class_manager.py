"""
Dual-Purpose Universality Class Manager with usage tracking.

This module extends the universality class database to track when classes are used
for detection vs validation, analyze which classes are most useful for discovery,
and optimize the database based on usage patterns.

Novel Discovery: The same theoretical knowledge serves both discovery (detecting
deviations from known classes) and validation (confirming predictions match classes).
This dual-purpose nature creates powerful synergy.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

from .phenomena_detector import UniversalityClass, NovelPhenomenonDetector
from ..utils.logging_utils import get_logger


@dataclass
class UsageRecord:
    """Record of a single usage of a universality class.
    
    Attributes:
        timestamp: When the class was used
        purpose: 'detection' or 'validation'
        variant_id: ID of the variant being analyzed
        exponent_name: Which exponent was compared
        deviation: Deviation in sigma from theoretical value
        matched: Whether the comparison matched (within threshold)
        confidence: Confidence in the comparison
    """
    timestamp: datetime
    purpose: str  # 'detection' or 'validation'
    variant_id: str
    exponent_name: str
    deviation: float
    matched: bool
    confidence: float


@dataclass
class ClassUsageStatistics:
    """Statistics for a universality class's usage.
    
    Attributes:
        class_name: Name of the universality class
        total_uses: Total number of times used
        detection_uses: Number of times used for detection
        validation_uses: Number of times used for validation
        match_rate: Fraction of comparisons that matched
        avg_deviation: Average deviation in sigma
        most_common_variants: Variants most often compared to this class
        most_common_exponents: Exponents most often compared
        discovery_value: Computed value for discovery (higher = more useful)
    """
    class_name: str
    total_uses: int = 0
    detection_uses: int = 0
    validation_uses: int = 0
    match_rate: float = 0.0
    avg_deviation: float = 0.0
    most_common_variants: List[Tuple[str, int]] = field(default_factory=list)
    most_common_exponents: List[Tuple[str, int]] = field(default_factory=list)
    discovery_value: float = 0.0


class UniversalityClassManager:
    """Dual-purpose universality class manager with usage tracking.
    
    This class extends the basic universality class database to track:
    - When classes are used for detection (finding deviations)
    - When classes are used for validation (confirming predictions)
    - Which classes are most useful for discovery
    - Usage patterns to optimize the database
    
    Key Innovation: The same theoretical knowledge serves both purposes,
    creating synergy between discovery and validation workflows.
    
    Attributes:
        universality_classes: Database of known universality classes
        usage_history: Complete history of class usage
        statistics: Computed statistics for each class
        logger: Logger instance
    """
    
    def __init__(
        self,
        custom_classes: Optional[Dict[str, UniversalityClass]] = None,
        load_history: bool = True,
        history_file: Optional[Path] = None
    ):
        """Initialize universality class manager.
        
        Args:
            custom_classes: Additional universality classes to include
            load_history: Whether to load usage history from file
            history_file: Path to history file (default: results/universality_usage.json)
        """
        self.logger = get_logger(__name__)
        
        # Initialize with known classes
        self.universality_classes = dict(NovelPhenomenonDetector.KNOWN_UNIVERSALITY_CLASSES)
        if custom_classes:
            self.universality_classes.update(custom_classes)
        
        # Usage tracking
        self.usage_history: List[UsageRecord] = []
        self.statistics: Dict[str, ClassUsageStatistics] = {}
        
        # Initialize statistics for each class
        for class_name in self.universality_classes.keys():
            self.statistics[class_name] = ClassUsageStatistics(class_name=class_name)
        
        # History file
        self.history_file = history_file or Path('results/universality_usage.json')
        
        # Load history if requested
        if load_history and self.history_file.exists():
            self._load_history()
        
        self.logger.info(
            f"Initialized UniversalityClassManager with {len(self.universality_classes)} classes"
        )
    
    def record_detection_use(
        self,
        class_name: str,
        variant_id: str,
        exponent_name: str,
        measured_value: float,
        measured_error: float,
        threshold: float = 3.0
    ) -> UsageRecord:
        """Record usage of a class for anomaly detection.
        
        Args:
            class_name: Name of the universality class
            variant_id: ID of the variant being analyzed
            exponent_name: Which exponent is being compared
            measured_value: Measured exponent value
            measured_error: Error on measured value
            threshold: Threshold in sigma for anomaly detection
            
        Returns:
            UsageRecord for this detection
        """
        if class_name not in self.universality_classes:
            raise ValueError(f"Unknown universality class: {class_name}")
        
        univ_class = self.universality_classes[class_name]
        
        if exponent_name not in univ_class.exponents:
            raise ValueError(
                f"Exponent {exponent_name} not in class {class_name}"
            )
        
        # Calculate deviation
        theoretical_value = univ_class.exponents[exponent_name]
        theoretical_error = univ_class.exponent_errors.get(exponent_name, 0.01)
        combined_error = np.sqrt(measured_error**2 + theoretical_error**2)
        deviation = abs(measured_value - theoretical_value) / combined_error
        
        # Check if matched (within threshold)
        matched = deviation <= threshold
        
        # Calculate confidence (higher deviation = higher confidence in anomaly)
        confidence = min(0.99, deviation / 10.0) if not matched else 1.0 - (deviation / threshold)
        
        # Create record
        record = UsageRecord(
            timestamp=datetime.now(),
            purpose='detection',
            variant_id=variant_id,
            exponent_name=exponent_name,
            deviation=deviation,
            matched=matched,
            confidence=confidence
        )
        
        # Store record
        self.usage_history.append(record)
        
        # Update statistics
        self._update_statistics(class_name, record)
        
        self.logger.debug(
            f"Detection use: {class_name} for {variant_id}.{exponent_name}, "
            f"deviation={deviation:.2f}σ, matched={matched}"
        )
        
        return record
    
    def record_validation_use(
        self,
        class_name: str,
        variant_id: str,
        predicted_exponents: Dict[str, float],
        measured_exponents: Dict[str, float],
        measured_errors: Dict[str, float],
        threshold: float = 2.0
    ) -> List[UsageRecord]:
        """Record usage of a class for hypothesis validation.
        
        Args:
            class_name: Name of the universality class
            variant_id: ID of the variant being validated
            predicted_exponents: Predicted exponent values
            measured_exponents: Measured exponent values
            measured_errors: Errors on measured values
            threshold: Threshold in sigma for validation
            
        Returns:
            List of UsageRecords for each exponent validated
        """
        if class_name not in self.universality_classes:
            raise ValueError(f"Unknown universality class: {class_name}")
        
        records = []
        
        for exponent_name in predicted_exponents.keys():
            if exponent_name not in measured_exponents:
                continue
            
            predicted_value = predicted_exponents[exponent_name]
            measured_value = measured_exponents[exponent_name]
            measured_error = measured_errors.get(exponent_name, 0.05)
            
            # Calculate deviation from prediction
            deviation = abs(measured_value - predicted_value) / measured_error
            
            # Check if validated (within threshold)
            matched = deviation <= threshold
            
            # Calculate confidence
            confidence = 1.0 - (deviation / (threshold * 2)) if matched else 0.0
            confidence = max(0.0, min(1.0, confidence))
            
            # Create record
            record = UsageRecord(
                timestamp=datetime.now(),
                purpose='validation',
                variant_id=variant_id,
                exponent_name=exponent_name,
                deviation=deviation,
                matched=matched,
                confidence=confidence
            )
            
            # Store record
            self.usage_history.append(record)
            records.append(record)
            
            # Update statistics
            self._update_statistics(class_name, record)
            
            self.logger.debug(
                f"Validation use: {class_name} for {variant_id}.{exponent_name}, "
                f"deviation={deviation:.2f}σ, matched={matched}"
            )
        
        return records
    
    def _update_statistics(self, class_name: str, record: UsageRecord) -> None:
        """Update statistics for a class based on a new usage record.
        
        Args:
            class_name: Name of the universality class
            record: Usage record to incorporate
        """
        stats = self.statistics[class_name]
        
        # Update counts
        stats.total_uses += 1
        if record.purpose == 'detection':
            stats.detection_uses += 1
        else:
            stats.validation_uses += 1
        
        # Update match rate (running average)
        old_matches = stats.match_rate * (stats.total_uses - 1)
        new_matches = old_matches + (1 if record.matched else 0)
        stats.match_rate = new_matches / stats.total_uses
        
        # Update average deviation (running average)
        old_sum = stats.avg_deviation * (stats.total_uses - 1)
        stats.avg_deviation = (old_sum + record.deviation) / stats.total_uses
        
        # Recompute discovery value
        stats.discovery_value = self._compute_discovery_value(class_name)
    
    def _compute_discovery_value(self, class_name: str) -> float:
        """Compute discovery value for a universality class.
        
        Discovery value measures how useful a class is for finding novel physics:
        - Higher usage = more relevant
        - Lower match rate = more discriminating
        - Higher average deviation = identifies more anomalies
        - Balance between detection and validation uses
        
        Args:
            class_name: Name of the universality class
            
        Returns:
            Discovery value (higher = more useful)
        """
        stats = self.statistics[class_name]
        
        if stats.total_uses == 0:
            return 0.0
        
        # Usage factor (log scale to avoid dominating)
        usage_factor = np.log1p(stats.total_uses)
        
        # Discrimination factor (lower match rate = more discriminating)
        # But not too low (need some matches to be useful)
        discrimination_factor = 1.0 - abs(stats.match_rate - 0.5)
        
        # Anomaly detection factor (higher deviation = finds more anomalies)
        anomaly_factor = min(1.0, stats.avg_deviation / 5.0)
        
        # Balance factor (prefer classes used for both purposes)
        if stats.total_uses > 0:
            detection_ratio = stats.detection_uses / stats.total_uses
            balance_factor = 1.0 - abs(detection_ratio - 0.5)
        else:
            balance_factor = 0.0
        
        # Combine factors
        discovery_value = (
            usage_factor * 0.3 +
            discrimination_factor * 0.3 +
            anomaly_factor * 0.2 +
            balance_factor * 0.2
        )
        
        return discovery_value
    
    def get_statistics(self, class_name: Optional[str] = None) -> Dict[str, ClassUsageStatistics]:
        """Get usage statistics for universality classes.
        
        Args:
            class_name: Specific class to get statistics for (None = all)
            
        Returns:
            Dictionary of statistics
        """
        if class_name is not None:
            if class_name not in self.statistics:
                raise ValueError(f"Unknown universality class: {class_name}")
            return {class_name: self.statistics[class_name]}
        
        return dict(self.statistics)
    
    def get_most_useful_classes(self, n: int = 5, purpose: Optional[str] = None) -> List[Tuple[str, float]]:
        """Get the most useful universality classes for discovery.
        
        Args:
            n: Number of classes to return
            purpose: Filter by purpose ('detection', 'validation', or None for both)
            
        Returns:
            List of (class_name, discovery_value) tuples, sorted by value
        """
        # Filter by purpose if requested
        if purpose is not None:
            filtered_stats = {}
            for class_name, stats in self.statistics.items():
                if purpose == 'detection' and stats.detection_uses > 0:
                    filtered_stats[class_name] = stats
                elif purpose == 'validation' and stats.validation_uses > 0:
                    filtered_stats[class_name] = stats
        else:
            filtered_stats = self.statistics
        
        # Sort by discovery value
        sorted_classes = sorted(
            filtered_stats.items(),
            key=lambda x: x[1].discovery_value,
            reverse=True
        )
        
        return [(name, stats.discovery_value) for name, stats in sorted_classes[:n]]
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get summary of usage patterns across all classes.
        
        Returns:
            Dictionary with summary statistics
        """
        total_uses = sum(s.total_uses for s in self.statistics.values())
        total_detection = sum(s.detection_uses for s in self.statistics.values())
        total_validation = sum(s.validation_uses for s in self.statistics.values())
        
        # Most used classes
        most_used = sorted(
            self.statistics.items(),
            key=lambda x: x[1].total_uses,
            reverse=True
        )[:5]
        
        # Most discriminating classes (match rate closest to 0.5)
        most_discriminating = sorted(
            [(name, stats) for name, stats in self.statistics.items() if stats.total_uses > 0],
            key=lambda x: abs(x[1].match_rate - 0.5),
            reverse=True
        )[:5]
        
        # Most useful for discovery
        most_useful = self.get_most_useful_classes(n=5)
        
        return {
            'total_uses': total_uses,
            'detection_uses': total_detection,
            'validation_uses': total_validation,
            'detection_ratio': total_detection / total_uses if total_uses > 0 else 0.0,
            'most_used_classes': [(name, stats.total_uses) for name, stats in most_used],
            'most_discriminating_classes': [(name, stats.match_rate) for name, stats in most_discriminating],
            'most_useful_for_discovery': most_useful,
            'num_classes': len(self.universality_classes),
            'num_used_classes': sum(1 for s in self.statistics.values() if s.total_uses > 0)
        }
    
    def optimize_database(self, min_uses: int = 10) -> Dict[str, Any]:
        """Optimize the universality class database based on usage patterns.
        
        Identifies:
        - Underutilized classes (candidates for removal)
        - High-value classes (should be prioritized)
        - Missing classes (gaps in coverage)
        
        Args:
            min_uses: Minimum uses to consider a class utilized
            
        Returns:
            Dictionary with optimization recommendations
        """
        underutilized = []
        high_value = []
        
        for class_name, stats in self.statistics.items():
            if stats.total_uses < min_uses:
                underutilized.append((class_name, stats.total_uses))
            elif stats.discovery_value > 0.5:
                high_value.append((class_name, stats.discovery_value))
        
        # Analyze coverage gaps
        # Look for variants that don't match any class well
        coverage_gaps = self._analyze_coverage_gaps()
        
        recommendations = {
            'underutilized_classes': underutilized,
            'high_value_classes': high_value,
            'coverage_gaps': coverage_gaps,
            'recommendation': self._generate_optimization_recommendation(
                underutilized, high_value, coverage_gaps
            )
        }
        
        self.logger.info(
            f"Database optimization: {len(underutilized)} underutilized, "
            f"{len(high_value)} high-value, {len(coverage_gaps)} coverage gaps"
        )
        
        return recommendations
    
    def _analyze_coverage_gaps(self) -> List[Dict[str, Any]]:
        """Analyze coverage gaps in the universality class database.
        
        Identifies variants that don't match any known class well,
        suggesting potential new classes to add.
        
        Returns:
            List of coverage gaps with details
        """
        gaps = []
        
        # Group usage records by variant
        variant_records: Dict[str, List[UsageRecord]] = {}
        for record in self.usage_history:
            if record.variant_id not in variant_records:
                variant_records[record.variant_id] = []
            variant_records[record.variant_id].append(record)
        
        # Check each variant
        for variant_id, records in variant_records.items():
            # Find best match
            best_match_deviation = min(r.deviation for r in records) if records else float('inf')
            
            # If best match is poor (>3σ), it's a coverage gap
            if best_match_deviation > 3.0:
                # Collect exponent values from records
                exponents = {}
                for record in records:
                    if record.exponent_name not in exponents:
                        exponents[record.exponent_name] = []
                    exponents[record.exponent_name].append(record.deviation)
                
                gaps.append({
                    'variant_id': variant_id,
                    'best_match_deviation': best_match_deviation,
                    'num_comparisons': len(records),
                    'avg_deviation': np.mean([r.deviation for r in records]),
                    'exponents_checked': list(exponents.keys())
                })
        
        return gaps
    
    def _generate_optimization_recommendation(
        self,
        underutilized: List[Tuple[str, int]],
        high_value: List[Tuple[str, float]],
        coverage_gaps: List[Dict[str, Any]]
    ) -> str:
        """Generate human-readable optimization recommendation.
        
        Args:
            underutilized: List of underutilized classes
            high_value: List of high-value classes
            coverage_gaps: List of coverage gaps
            
        Returns:
            Recommendation string
        """
        recommendations = []
        
        if underutilized:
            recommendations.append(
                f"Consider removing {len(underutilized)} underutilized classes: "
                f"{', '.join(name for name, _ in underutilized[:3])}"
            )
        
        if high_value:
            recommendations.append(
                f"Prioritize {len(high_value)} high-value classes: "
                f"{', '.join(name for name, _ in high_value[:3])}"
            )
        
        if coverage_gaps:
            recommendations.append(
                f"Add {len(coverage_gaps)} new classes to cover gaps in: "
                f"{', '.join(gap['variant_id'] for gap in coverage_gaps[:3])}"
            )
        
        if not recommendations:
            return "Database is well-optimized. No changes recommended."
        
        return " | ".join(recommendations)
    
    def save_history(self, filepath: Optional[Path] = None) -> None:
        """Save usage history to file.
        
        Args:
            filepath: Path to save to (default: self.history_file)
        """
        filepath = filepath or self.history_file
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert records to serializable format
        data = {
            'usage_history': [
                {
                    'timestamp': record.timestamp.isoformat(),
                    'purpose': record.purpose,
                    'variant_id': record.variant_id,
                    'exponent_name': record.exponent_name,
                    'deviation': float(record.deviation),
                    'matched': bool(record.matched),
                    'confidence': float(record.confidence)
                }
                for record in self.usage_history
            ],
            'statistics': {
                name: {
                    'class_name': stats.class_name,
                    'total_uses': int(stats.total_uses),
                    'detection_uses': int(stats.detection_uses),
                    'validation_uses': int(stats.validation_uses),
                    'match_rate': float(stats.match_rate),
                    'avg_deviation': float(stats.avg_deviation),
                    'discovery_value': float(stats.discovery_value)
                }
                for name, stats in self.statistics.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved usage history to {filepath}")
    
    def _load_history(self) -> None:
        """Load usage history from file."""
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
            
            # Load usage records
            self.usage_history = [
                UsageRecord(
                    timestamp=datetime.fromisoformat(record['timestamp']),
                    purpose=record['purpose'],
                    variant_id=record['variant_id'],
                    exponent_name=record['exponent_name'],
                    deviation=record['deviation'],
                    matched=record['matched'],
                    confidence=record['confidence']
                )
                for record in data.get('usage_history', [])
            ]
            
            # Load statistics
            for name, stats_data in data.get('statistics', {}).items():
                if name in self.statistics:
                    self.statistics[name] = ClassUsageStatistics(**stats_data)
            
            self.logger.info(
                f"Loaded {len(self.usage_history)} usage records from {self.history_file}"
            )
        except Exception as e:
            self.logger.warning(f"Failed to load history: {e}")
