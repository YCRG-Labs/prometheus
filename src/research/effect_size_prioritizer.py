"""
Effect Size Prioritization System for research findings.

This module implements a prioritization system that classifies findings based on
both statistical significance and practical importance (effect size), creating
a priority queue for investigation.

Novel Discovery: Effect size bridges statistical and practical significance:
- Statistical significance (p-value) tells us if an effect is real
- Effect size (Cohen's d) tells us if an effect is important
- Both are needed to avoid false positives (low effect) and false negatives (high effect, low p)

The prioritizer prevents wasting effort on statistically significant but
unimportant findings, while highlighting important findings that need more data.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import heapq

from ..utils.logging_utils import get_logger


class SignificanceCategory(Enum):
    """Statistical significance category."""
    HIGHLY_SIGNIFICANT = "highly_significant"  # p < 0.01
    SIGNIFICANT = "significant"  # 0.01 <= p < 0.05
    MARGINALLY_SIGNIFICANT = "marginally_significant"  # 0.05 <= p < 0.10
    NOT_SIGNIFICANT = "not_significant"  # p >= 0.10


class EffectSizeCategory(Enum):
    """Effect size category (Cohen's d)."""
    VERY_LARGE = "very_large"  # |d| >= 1.2
    LARGE = "large"  # 0.8 <= |d| < 1.2
    MEDIUM = "medium"  # 0.5 <= |d| < 0.8
    SMALL = "small"  # 0.2 <= |d| < 0.5
    NEGLIGIBLE = "negligible"  # |d| < 0.2


class FindingCategory(Enum):
    """Combined finding category."""
    STRONG_EVIDENCE = "strong_evidence"  # High significance + large effect
    INVESTIGATE_FURTHER = "investigate_further"  # High effect but low significance (needs more data)
    FALSE_POSITIVE_RISK = "false_positive_risk"  # High significance but low effect (not important)
    NO_EVIDENCE = "no_evidence"  # Low significance + low effect


@dataclass
class Finding:
    """A research finding with statistical and practical significance.
    
    Attributes:
        finding_id: Unique identifier for the finding
        variant_id: ID of the variant being studied
        exponent_name: Name of the exponent (if applicable)
        measured_value: Measured value
        predicted_value: Predicted/reference value
        p_value: Statistical significance (p-value)
        effect_size: Practical importance (Cohen's d)
        sample_size: Number of samples
        description: Human-readable description
        metadata: Additional metadata
    """
    finding_id: str
    variant_id: str
    exponent_name: Optional[str]
    measured_value: float
    predicted_value: float
    p_value: float
    effect_size: float
    sample_size: int
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PrioritizedFinding:
    """A finding with priority score and classification.
    
    Attributes:
        finding: The original finding
        priority_score: Priority score (0-100, higher = more important)
        significance_category: Statistical significance category
        effect_size_category: Effect size category
        finding_category: Combined finding category
        recommendation: Human-readable recommendation
    """
    finding: Finding
    priority_score: float
    significance_category: SignificanceCategory
    effect_size_category: EffectSizeCategory
    finding_category: FindingCategory
    recommendation: str
    
    def __lt__(self, other):
        """Compare by priority score (for heap queue)."""
        return self.priority_score > other.priority_score  # Higher priority first


class EffectSizePrioritizer:
    """Prioritize research findings based on effect size and statistical significance.
    
    This class implements a prioritization system that:
    1. Classifies findings by statistical significance (p-value)
    2. Classifies findings by practical importance (effect size)
    3. Combines both to create four categories
    4. Assigns priority scores for investigation
    5. Maintains a priority queue of findings
    
    Key Innovation: Prevents false positives (statistically significant but
    unimportant) and false negatives (important but not yet significant) by
    considering both dimensions simultaneously.
    
    Attributes:
        alpha: Significance level threshold (default: 0.05)
        small_effect_threshold: Threshold for small effect size (default: 0.2)
        medium_effect_threshold: Threshold for medium effect size (default: 0.5)
        large_effect_threshold: Threshold for large effect size (default: 0.8)
        findings: List of all findings
        priority_queue: Priority queue of findings
        logger: Logger instance
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        small_effect_threshold: float = 0.2,
        medium_effect_threshold: float = 0.5,
        large_effect_threshold: float = 0.8
    ):
        """Initialize effect size prioritizer.
        
        Args:
            alpha: Significance level threshold
            small_effect_threshold: Threshold for small effect size
            medium_effect_threshold: Threshold for medium effect size
            large_effect_threshold: Threshold for large effect size
        """
        self.alpha = alpha
        self.small_effect_threshold = small_effect_threshold
        self.medium_effect_threshold = medium_effect_threshold
        self.large_effect_threshold = large_effect_threshold
        
        self.findings: List[PrioritizedFinding] = []
        self.priority_queue: List[PrioritizedFinding] = []
        
        self.logger = get_logger(__name__)
        self.logger.info(
            f"Initialized EffectSizePrioritizer with α={alpha}, "
            f"effect thresholds=[{small_effect_threshold}, {medium_effect_threshold}, {large_effect_threshold}]"
        )
    
    def classify_significance(self, p_value: float) -> SignificanceCategory:
        """Classify statistical significance based on p-value.
        
        Args:
            p_value: P-value from statistical test
            
        Returns:
            SignificanceCategory
        """
        if p_value < 0.01:
            return SignificanceCategory.HIGHLY_SIGNIFICANT
        elif p_value < 0.05:
            return SignificanceCategory.SIGNIFICANT
        elif p_value < 0.10:
            return SignificanceCategory.MARGINALLY_SIGNIFICANT
        else:
            return SignificanceCategory.NOT_SIGNIFICANT
    
    def classify_effect_size(self, effect_size: float) -> EffectSizeCategory:
        """Classify effect size based on Cohen's d.
        
        Args:
            effect_size: Cohen's d effect size
            
        Returns:
            EffectSizeCategory
        """
        abs_effect = abs(effect_size)
        
        if abs_effect >= 1.2:
            return EffectSizeCategory.VERY_LARGE
        elif abs_effect >= self.large_effect_threshold:
            return EffectSizeCategory.LARGE
        elif abs_effect >= self.medium_effect_threshold:
            return EffectSizeCategory.MEDIUM
        elif abs_effect >= self.small_effect_threshold:
            return EffectSizeCategory.SMALL
        else:
            return EffectSizeCategory.NEGLIGIBLE
    
    def classify_finding(
        self,
        significance: SignificanceCategory,
        effect_size_cat: EffectSizeCategory
    ) -> FindingCategory:
        """Classify finding based on both significance and effect size.
        
        Args:
            significance: Statistical significance category
            effect_size_cat: Effect size category
            
        Returns:
            FindingCategory
        """
        # High significance
        is_significant = significance in [
            SignificanceCategory.HIGHLY_SIGNIFICANT,
            SignificanceCategory.SIGNIFICANT
        ]
        
        # Large effect
        is_large_effect = effect_size_cat in [
            EffectSizeCategory.VERY_LARGE,
            EffectSizeCategory.LARGE,
            EffectSizeCategory.MEDIUM
        ]
        
        # Classify into four categories
        if is_significant and is_large_effect:
            return FindingCategory.STRONG_EVIDENCE
        elif not is_significant and is_large_effect:
            return FindingCategory.INVESTIGATE_FURTHER
        elif is_significant and not is_large_effect:
            return FindingCategory.FALSE_POSITIVE_RISK
        else:
            return FindingCategory.NO_EVIDENCE
    
    def compute_priority_score(
        self,
        p_value: float,
        effect_size: float,
        sample_size: int,
        significance: SignificanceCategory,
        effect_size_cat: EffectSizeCategory,
        finding_cat: FindingCategory
    ) -> float:
        """Compute priority score for a finding.
        
        Priority score combines:
        - Statistical significance (lower p-value = higher priority)
        - Effect size (larger effect = higher priority)
        - Sample size (larger sample = higher confidence)
        - Category bonuses
        
        Args:
            p_value: P-value
            effect_size: Cohen's d
            sample_size: Number of samples
            significance: Significance category
            effect_size_cat: Effect size category
            finding_cat: Finding category
            
        Returns:
            Priority score (0-100)
        """
        # Base score from p-value (0-40 points)
        # Use -log10(p) to give more weight to smaller p-values
        if p_value > 0:
            p_score = min(40, -np.log10(p_value) * 10)
        else:
            p_score = 40
        
        # Effect size score (0-40 points)
        effect_score = min(40, abs(effect_size) * 40)
        
        # Sample size score (0-10 points)
        # Log scale to avoid dominating
        sample_score = min(10, np.log10(sample_size + 1) * 3)
        
        # Category bonus (0-10 points)
        if finding_cat == FindingCategory.STRONG_EVIDENCE:
            category_bonus = 10
        elif finding_cat == FindingCategory.INVESTIGATE_FURTHER:
            category_bonus = 7
        elif finding_cat == FindingCategory.FALSE_POSITIVE_RISK:
            category_bonus = 3
        else:
            category_bonus = 0
        
        # Total score
        priority_score = p_score + effect_score + sample_score + category_bonus
        
        return min(100, priority_score)
    
    def generate_recommendation(
        self,
        finding_cat: FindingCategory,
        significance: SignificanceCategory,
        effect_size_cat: EffectSizeCategory,
        p_value: float,
        effect_size: float
    ) -> str:
        """Generate human-readable recommendation.
        
        Args:
            finding_cat: Finding category
            significance: Significance category
            effect_size_cat: Effect size category
            p_value: P-value
            effect_size: Cohen's d
            
        Returns:
            Recommendation string
        """
        if finding_cat == FindingCategory.STRONG_EVIDENCE:
            return (
                f"STRONG EVIDENCE: Both statistically significant (p={p_value:.4f}) "
                f"and practically important (d={effect_size:.2f}). "
                f"High priority for publication and follow-up."
            )
        elif finding_cat == FindingCategory.INVESTIGATE_FURTHER:
            return (
                f"INVESTIGATE FURTHER: Large effect size (d={effect_size:.2f}) "
                f"but not yet significant (p={p_value:.4f}). "
                f"Collect more data to increase statistical power."
            )
        elif finding_cat == FindingCategory.FALSE_POSITIVE_RISK:
            return (
                f"FALSE POSITIVE RISK: Statistically significant (p={p_value:.4f}) "
                f"but small effect size (d={effect_size:.2f}). "
                f"May not be practically important. Verify with independent data."
            )
        else:
            return (
                f"NO EVIDENCE: Neither significant (p={p_value:.4f}) "
                f"nor large effect (d={effect_size:.2f}). "
                f"Low priority for further investigation."
            )
    
    def add_finding(self, finding: Finding) -> PrioritizedFinding:
        """Add a finding and compute its priority.
        
        Args:
            finding: Finding to add
            
        Returns:
            PrioritizedFinding with computed priority
        """
        # Classify significance and effect size
        significance = self.classify_significance(finding.p_value)
        effect_size_cat = self.classify_effect_size(finding.effect_size)
        finding_cat = self.classify_finding(significance, effect_size_cat)
        
        # Compute priority score
        priority_score = self.compute_priority_score(
            finding.p_value,
            finding.effect_size,
            finding.sample_size,
            significance,
            effect_size_cat,
            finding_cat
        )
        
        # Generate recommendation
        recommendation = self.generate_recommendation(
            finding_cat,
            significance,
            effect_size_cat,
            finding.p_value,
            finding.effect_size
        )
        
        # Create prioritized finding
        prioritized = PrioritizedFinding(
            finding=finding,
            priority_score=priority_score,
            significance_category=significance,
            effect_size_category=effect_size_cat,
            finding_category=finding_cat,
            recommendation=recommendation
        )
        
        # Add to list and priority queue
        self.findings.append(prioritized)
        heapq.heappush(self.priority_queue, prioritized)
        
        self.logger.info(
            f"Added finding {finding.finding_id}: "
            f"priority={priority_score:.1f}, category={finding_cat.value}"
        )
        
        return prioritized
    
    def get_top_priorities(self, n: int = 10) -> List[PrioritizedFinding]:
        """Get top N priority findings.
        
        Args:
            n: Number of findings to return
            
        Returns:
            List of top priority findings
        """
        # Use nsmallest because we want highest priority (heap is min-heap with reversed comparison)
        return heapq.nsmallest(n, self.priority_queue)
    
    def get_findings_by_category(
        self,
        category: FindingCategory
    ) -> List[PrioritizedFinding]:
        """Get all findings in a specific category.
        
        Args:
            category: Finding category to filter by
            
        Returns:
            List of findings in that category
        """
        return [f for f in self.findings if f.finding_category == category]
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics across all findings.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.findings:
            return {
                'total_findings': 0,
                'by_category': {},
                'by_significance': {},
                'by_effect_size': {},
                'avg_priority': 0.0
            }
        
        # Count by category
        by_category = {}
        for cat in FindingCategory:
            count = sum(1 for f in self.findings if f.finding_category == cat)
            by_category[cat.value] = count
        
        # Count by significance
        by_significance = {}
        for sig in SignificanceCategory:
            count = sum(1 for f in self.findings if f.significance_category == sig)
            by_significance[sig.value] = count
        
        # Count by effect size
        by_effect_size = {}
        for eff in EffectSizeCategory:
            count = sum(1 for f in self.findings if f.effect_size_category == eff)
            by_effect_size[eff.value] = count
        
        # Average priority
        avg_priority = np.mean([f.priority_score for f in self.findings])
        
        return {
            'total_findings': len(self.findings),
            'by_category': by_category,
            'by_significance': by_significance,
            'by_effect_size': by_effect_size,
            'avg_priority': avg_priority,
            'top_priority': self.findings[0].priority_score if self.findings else 0.0
        }
    
    def generate_investigation_plan(self) -> Dict[str, List[PrioritizedFinding]]:
        """Generate investigation plan organized by priority.
        
        Returns:
            Dictionary with findings organized by action needed
        """
        plan = {
            'immediate_investigation': [],  # Strong evidence
            'collect_more_data': [],  # Investigate further
            'verify_independently': [],  # False positive risk
            'low_priority': []  # No evidence
        }
        
        for finding in self.findings:
            if finding.finding_category == FindingCategory.STRONG_EVIDENCE:
                plan['immediate_investigation'].append(finding)
            elif finding.finding_category == FindingCategory.INVESTIGATE_FURTHER:
                plan['collect_more_data'].append(finding)
            elif finding.finding_category == FindingCategory.FALSE_POSITIVE_RISK:
                plan['verify_independently'].append(finding)
            else:
                plan['low_priority'].append(finding)
        
        # Sort each category by priority
        for key in plan:
            plan[key] = sorted(plan[key], key=lambda x: x.priority_score, reverse=True)
        
        return plan
    
    def visualize_findings(self) -> str:
        """Create ASCII visualization of findings in 2D space.
        
        Returns:
            ASCII art visualization
        """
        if not self.findings:
            return "No findings to visualize"
        
        # Create 2D grid (significance vs effect size)
        grid = [[' ' for _ in range(50)] for _ in range(20)]
        
        for finding in self.findings:
            # Map p-value to y-axis (0-20)
            # Use -log10(p) for better visualization
            if finding.finding.p_value > 0:
                y = int(min(19, -np.log10(finding.finding.p_value) * 4))
            else:
                y = 19
            
            # Map effect size to x-axis (0-50)
            x = int(min(49, abs(finding.finding.effect_size) * 25))
            
            # Mark with category symbol
            if finding.finding_category == FindingCategory.STRONG_EVIDENCE:
                symbol = '●'
            elif finding.finding_category == FindingCategory.INVESTIGATE_FURTHER:
                symbol = '◐'
            elif finding.finding_category == FindingCategory.FALSE_POSITIVE_RISK:
                symbol = '○'
            else:
                symbol = '·'
            
            grid[19 - y][x] = symbol
        
        # Build visualization
        lines = []
        lines.append("Effect Size vs Statistical Significance")
        lines.append("=" * 52)
        lines.append("High │")
        lines.append("Sig  │")
        
        for row in grid:
            lines.append("     │" + ''.join(row))
        
        lines.append("Low  │" + "_" * 50)
        lines.append("     └" + "─" * 50)
        lines.append("      Low" + " " * 20 + "Effect Size" + " " * 15 + "High")
        lines.append("")
        lines.append("Legend: ● Strong Evidence  ◐ Investigate Further")
        lines.append("        ○ False Positive Risk  · No Evidence")
        
        return '\n'.join(lines)
