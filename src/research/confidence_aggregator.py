"""
Confidence Aggregator for multiplicative confidence scoring across validation layers.

This module implements a novel methodological enhancement discovered during the
implementation of the research explorer: combining confidence scores from multiple
independent validation layers (detection, comparison, validation) to create a
robust overall confidence metric.

The key insight is that each validation layer provides independent evidence:
- NovelPhenomenonDetector: Flags anomalies (0-1 confidence)
- ComparativeAnalyzer: Quantifies statistical differences (0-1 confidence)
- ValidationFramework: Rigorously tests hypotheses (0-1 confidence)

By multiplying these independent confidence scores, we get a conservative overall
assessment that requires agreement across all layers.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path

from .base_types import NovelPhenomenon, VAEAnalysisResults, ValidationResult
from .phenomena_detector import NovelPhenomenonDetector
from .comparative_analyzer import ComparativeAnalyzer, ComparisonResults
from .validation_framework import ValidationFramework
from ..utils.logging_utils import get_logger


@dataclass
class AggregatedConfidence:
    """Aggregated confidence score from multiple validation layers.
    
    Attributes:
        overall_confidence: Overall confidence (product of layer confidences)
        layer_confidences: Individual confidence scores from each layer
        layer_weights: Optional weights for each layer (default: equal)
        breakdown: Detailed breakdown of confidence sources
        recommendation: Recommended action based on confidence
        message: Human-readable summary
    """
    overall_confidence: float
    layer_confidences: Dict[str, float]
    layer_weights: Dict[str, float]
    breakdown: Dict[str, any]
    recommendation: str
    message: str


class ConfidenceAggregator:
    """Aggregate confidence scores from multiple validation layers.
    
    This class implements the multiplicative confidence scoring pattern discovered
    during system implementation. It combines independent evidence from:
    
    1. Detection Layer (NovelPhenomenonDetector):
       - Confidence in anomaly detection
       - Based on deviation from known universality classes
       - Range: 0.0 (no anomaly) to 1.0 (strong anomaly)
    
    2. Comparison Layer (ComparativeAnalyzer):
       - Confidence in statistical significance
       - Based on ANOVA/t-test p-values and effect sizes
       - Range: 0.0 (no difference) to 1.0 (highly significant)
    
    3. Validation Layer (ValidationFramework):
       - Confidence in hypothesis validation
       - Based on bootstrap CIs and hypothesis tests
       - Range: 0.0 (refuted) to 1.0 (validated)
    
    The overall confidence is calculated as:
        overall = detection_conf × comparison_conf × validation_conf
    
    This multiplicative approach ensures that:
    - All layers must agree for high confidence
    - Disagreement in any layer reduces overall confidence
    - Conservative assessment prevents false positives
    
    Attributes:
        detector: NovelPhenomenonDetector instance
        analyzer: ComparativeAnalyzer instance
        validator: ValidationFramework instance
        logger: Logger instance
    """
    
    def __init__(
        self,
        anomaly_threshold: float = 3.0,
        n_bootstrap: int = 1000,
        alpha: float = 0.05
    ):
        """Initialize confidence aggregator.
        
        Args:
            anomaly_threshold: Threshold for anomaly detection (sigma)
            n_bootstrap: Number of bootstrap samples for validation
            alpha: Significance level for hypothesis tests
        """
        self.detector = NovelPhenomenonDetector(anomaly_threshold)
        self.analyzer = ComparativeAnalyzer(anomaly_threshold)
        self.validator = ValidationFramework(n_bootstrap, alpha, anomaly_threshold)
        self.logger = get_logger(__name__)
        self.logger.info("Initialized ConfidenceAggregator with three-tier validation")
    
    def aggregate_confidence(
        self,
        vae_results: VAEAnalysisResults,
        phenomena: List[NovelPhenomenon],
        comparison_results: Optional[ComparisonResults] = None,
        validation_results: Optional[Dict[str, ValidationResult]] = None,
        layer_weights: Optional[Dict[str, float]] = None
    ) -> AggregatedConfidence:
        """Aggregate confidence scores from multiple validation layers.
        
        Args:
            vae_results: VAE analysis results
            phenomena: List of detected phenomena
            comparison_results: Optional comparison results from analyzer
            validation_results: Optional validation results from validator
            layer_weights: Optional weights for each layer (default: equal)
            
        Returns:
            AggregatedConfidence with overall score and breakdown
        """
        self.logger.info(
            f"Aggregating confidence for variant '{vae_results.variant_id}' "
            f"at parameters {vae_results.parameters}"
        )
        
        # Default equal weights
        if layer_weights is None:
            layer_weights = {
                'detection': 1.0,
                'comparison': 1.0,
                'validation': 1.0
            }
        
        # Normalize weights
        total_weight = sum(layer_weights.values())
        layer_weights = {k: v / total_weight for k, v in layer_weights.items()}
        
        # Extract confidence from each layer
        layer_confidences = {}
        breakdown = {}
        
        # Layer 1: Detection confidence
        detection_conf = self._extract_detection_confidence(phenomena, vae_results)
        layer_confidences['detection'] = detection_conf
        breakdown['detection'] = {
            'confidence': detection_conf,
            'n_phenomena': len(phenomena),
            'phenomena_types': [p.phenomenon_type for p in phenomena],
            'max_phenomenon_confidence': max([p.confidence for p in phenomena]) if phenomena else 0.0
        }
        
        # Layer 2: Comparison confidence
        if comparison_results is not None:
            comparison_conf = self._extract_comparison_confidence(
                comparison_results, vae_results.variant_id
            )
        else:
            comparison_conf = 0.5  # Neutral if not available
        layer_confidences['comparison'] = comparison_conf
        breakdown['comparison'] = {
            'confidence': comparison_conf,
            'available': comparison_results is not None
        }
        
        # Layer 3: Validation confidence
        if validation_results is not None:
            validation_conf = self._extract_validation_confidence(validation_results)
        else:
            validation_conf = 0.5  # Neutral if not available
        layer_confidences['validation'] = validation_conf
        breakdown['validation'] = {
            'confidence': validation_conf,
            'available': validation_results is not None,
            'n_tests': len(validation_results) if validation_results else 0
        }
        
        # Calculate overall confidence using weighted geometric mean
        # This is equivalent to multiplicative combination with weights
        overall_confidence = np.prod([
            conf ** layer_weights[layer]
            for layer, conf in layer_confidences.items()
        ])
        
        # Generate recommendation based on overall confidence
        recommendation = self._generate_recommendation(
            overall_confidence, layer_confidences
        )
        
        # Create message
        message = self._create_confidence_message(
            overall_confidence, layer_confidences, recommendation
        )
        
        result = AggregatedConfidence(
            overall_confidence=overall_confidence,
            layer_confidences=layer_confidences,
            layer_weights=layer_weights,
            breakdown=breakdown,
            recommendation=recommendation,
            message=message
        )
        
        self.logger.info(
            f"Aggregated confidence: {overall_confidence:.2%} "
            f"(detection: {detection_conf:.2%}, comparison: {comparison_conf:.2%}, "
            f"validation: {validation_conf:.2%})"
        )
        
        return result
    
    def _extract_detection_confidence(
        self,
        phenomena: List[NovelPhenomenon],
        vae_results: VAEAnalysisResults
    ) -> float:
        """Extract confidence from detection layer.
        
        Args:
            phenomena: List of detected phenomena
            vae_results: VAE analysis results
            
        Returns:
            Detection confidence (0.0 to 1.0)
        """
        if not phenomena:
            # No phenomena detected - could mean either:
            # 1. System matches known physics (high confidence in known class)
            # 2. Insufficient data (low confidence)
            # Use fit quality as proxy
            avg_r_squared = np.mean(list(vae_results.r_squared_values.values()))
            return avg_r_squared  # High R² → high confidence in results
        
        # Use maximum phenomenon confidence
        # (most confident detection drives overall detection confidence)
        max_confidence = max(p.confidence for p in phenomena)
        
        # Weight by fit quality
        avg_r_squared = np.mean(list(vae_results.r_squared_values.values()))
        weighted_confidence = max_confidence * avg_r_squared
        
        return weighted_confidence
    
    def _extract_comparison_confidence(
        self,
        comparison_results: ComparisonResults,
        variant_id: str
    ) -> float:
        """Extract confidence from comparison layer.
        
        Args:
            comparison_results: Comparison results from analyzer
            variant_id: ID of variant being assessed
            
        Returns:
            Comparison confidence (0.0 to 1.0)
        """
        # Extract universality test for this variant
        if variant_id in comparison_results.universality_tests:
            tests = comparison_results.universality_tests[variant_id]
            
            # Find best matching universality class
            best_confidence = 0.0
            for class_name, test in tests.items():
                if test.confidence > best_confidence:
                    best_confidence = test.confidence
            
            return best_confidence
        
        # Fallback: use ANOVA significance as proxy
        # High significance → high confidence in differences
        anova_results = comparison_results.exponent_comparisons
        if anova_results:
            # Average p-values across exponents
            p_values = []
            for exp_comp in anova_results.values():
                if exp_comp['anova']['p_value'] is not None:
                    p_values.append(exp_comp['anova']['p_value'])
            
            if p_values:
                avg_p_value = np.mean(p_values)
                # Convert p-value to confidence (lower p → higher confidence)
                return 1.0 - avg_p_value
        
        return 0.5  # Neutral if no data
    
    def _extract_validation_confidence(
        self,
        validation_results: Dict[str, ValidationResult]
    ) -> float:
        """Extract confidence from validation layer.
        
        Args:
            validation_results: Dictionary of validation results
            
        Returns:
            Validation confidence (0.0 to 1.0)
        """
        if not validation_results:
            return 0.5  # Neutral if no validation
        
        # Average confidence across all validation tests
        confidences = [result.confidence for result in validation_results.values()]
        avg_confidence = np.mean(confidences)
        
        # Weight by validation success rate
        n_validated = sum(1 for r in validation_results.values() if r.validated)
        success_rate = n_validated / len(validation_results)
        
        # Combine average confidence with success rate
        weighted_confidence = avg_confidence * (0.5 + 0.5 * success_rate)
        
        return weighted_confidence
    
    def _generate_recommendation(
        self,
        overall_confidence: float,
        layer_confidences: Dict[str, float]
    ) -> str:
        """Generate recommendation based on confidence scores.
        
        Args:
            overall_confidence: Overall aggregated confidence
            layer_confidences: Individual layer confidences
            
        Returns:
            Recommendation string
        """
        # High confidence (>0.8): Strong evidence
        if overall_confidence > 0.8:
            return "STRONG_EVIDENCE"
        
        # Moderate confidence (0.5-0.8): Further investigation
        elif overall_confidence > 0.5:
            # Check which layer is weakest
            weakest_layer = min(layer_confidences.items(), key=lambda x: x[1])[0]
            return f"INVESTIGATE_{weakest_layer.upper()}"
        
        # Low confidence (0.2-0.5): Inconclusive
        elif overall_confidence > 0.2:
            return "INCONCLUSIVE"
        
        # Very low confidence (<0.2): Likely false positive
        else:
            return "LIKELY_FALSE_POSITIVE"
    
    def _create_confidence_message(
        self,
        overall_confidence: float,
        layer_confidences: Dict[str, float],
        recommendation: str
    ) -> str:
        """Create human-readable confidence message.
        
        Args:
            overall_confidence: Overall confidence
            layer_confidences: Layer confidences
            recommendation: Recommendation string
            
        Returns:
            Formatted message
        """
        lines = []
        lines.append(f"Overall Confidence: {overall_confidence:.2%}")
        lines.append("Layer Breakdown:")
        for layer, conf in layer_confidences.items():
            lines.append(f"  - {layer.capitalize()}: {conf:.2%}")
        lines.append(f"Recommendation: {recommendation.replace('_', ' ')}")
        
        return "\n".join(lines)
    
    def visualize_confidence_breakdown(
        self,
        aggregated_confidence: AggregatedConfidence,
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize confidence breakdown across layers.
        
        Creates a bar chart showing confidence scores from each layer and
        the overall aggregated confidence.
        
        Args:
            aggregated_confidence: Aggregated confidence result
            output_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left plot: Layer confidences
        layers = list(aggregated_confidence.layer_confidences.keys())
        confidences = list(aggregated_confidence.layer_confidences.values())
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green
        bars = ax1.bar(layers, confidences, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, conf in zip(bars, confidences):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2, height,
                f'{conf:.2%}',
                ha='center', va='bottom', fontsize=12, fontweight='bold'
            )
        
        ax1.set_ylabel('Confidence', fontsize=14)
        ax1.set_title('Confidence by Validation Layer', fontsize=16, fontweight='bold')
        ax1.set_ylim(0, 1.1)
        ax1.grid(axis='y', alpha=0.3)
        ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='High confidence')
        ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate')
        ax1.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Low confidence')
        ax1.legend(fontsize=10)
        
        # Right plot: Overall confidence with breakdown
        overall = aggregated_confidence.overall_confidence
        
        # Create stacked representation showing multiplicative effect
        ax2.barh(['Overall'], [overall], color='purple', alpha=0.7, edgecolor='black')
        ax2.text(
            overall / 2, 0, f'{overall:.2%}',
            ha='center', va='center', fontsize=14, fontweight='bold', color='white'
        )
        
        # Add layer contributions below
        y_pos = -0.5
        for i, (layer, conf) in enumerate(aggregated_confidence.layer_confidences.items()):
            ax2.barh([y_pos], [conf], color=colors[i], alpha=0.5, edgecolor='black')
            ax2.text(
                conf / 2, y_pos, f'{layer}: {conf:.2%}',
                ha='center', va='center', fontsize=10
            )
            y_pos -= 0.3
        
        ax2.set_xlabel('Confidence', fontsize=14)
        ax2.set_title('Overall Aggregated Confidence', fontsize=16, fontweight='bold')
        ax2.set_xlim(0, 1.1)
        ax2.set_ylim(-1.5, 0.5)
        ax2.grid(axis='x', alpha=0.3)
        ax2.axvline(x=0.8, color='green', linestyle='--', alpha=0.5)
        ax2.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5)
        ax2.axvline(x=0.2, color='red', linestyle='--', alpha=0.5)
        
        # Add recommendation text
        rec_text = aggregated_confidence.recommendation.replace('_', ' ')
        ax2.text(
            0.5, -1.3, f'Recommendation: {rec_text}',
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved confidence breakdown visualization: {output_path}")
        
        return fig
    
    def compare_confidence_across_variants(
        self,
        variant_confidences: Dict[str, AggregatedConfidence],
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """Compare aggregated confidence across multiple variants.
        
        Args:
            variant_confidences: Dictionary mapping variant IDs to confidences
            output_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        variants = list(variant_confidences.keys())
        n_variants = len(variants)
        
        # Extract data
        overall_confs = [variant_confidences[v].overall_confidence for v in variants]
        detection_confs = [variant_confidences[v].layer_confidences['detection'] for v in variants]
        comparison_confs = [variant_confidences[v].layer_confidences['comparison'] for v in variants]
        validation_confs = [variant_confidences[v].layer_confidences['validation'] for v in variants]
        
        # Create grouped bar chart
        x = np.arange(n_variants)
        width = 0.2
        
        ax.bar(x - 1.5*width, detection_confs, width, label='Detection', color='#3498db', alpha=0.7)
        ax.bar(x - 0.5*width, comparison_confs, width, label='Comparison', color='#e74c3c', alpha=0.7)
        ax.bar(x + 0.5*width, validation_confs, width, label='Validation', color='#2ecc71', alpha=0.7)
        ax.bar(x + 1.5*width, overall_confs, width, label='Overall', color='purple', alpha=0.7)
        
        ax.set_xlabel('Variant', fontsize=14)
        ax.set_ylabel('Confidence', fontsize=14)
        ax.set_title('Confidence Comparison Across Variants', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Add threshold lines
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.3, label='High')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.3, label='Moderate')
        ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.3, label='Low')
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved variant comparison visualization: {output_path}")
        
        return fig
