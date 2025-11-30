"""
Target Selection Engine for Discovery Campaign.

This module implements the target selection engine that identifies and prioritizes
Ising model variants for exploration based on theoretical predictions, literature
gaps, and computational feasibility.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np

from .base_types import ModelVariantConfig
from .campaign_orchestrator import TargetVariant
from ..utils.logging_utils import get_logger


@dataclass
class SelectionCriteria:
    """Criteria for scoring target variants.
    
    Attributes:
        theoretical_weight: Weight for theoretical prediction score (0-1)
        literature_gap_weight: Weight for literature gap score (0-1)
        feasibility_weight: Weight for computational feasibility (0-1)
        novelty_weight: Weight for novelty potential (0-1)
    """
    theoretical_weight: float = 0.3
    literature_gap_weight: float = 0.3
    feasibility_weight: float = 0.2
    novelty_weight: float = 0.2
    
    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = (self.theoretical_weight + self.literature_gap_weight + 
                self.feasibility_weight + self.novelty_weight)
        if not np.isclose(total, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total}")


class TargetSelectionEngine:
    """Select and prioritize Ising model variants for exploration.
    
    This engine identifies candidate variants from a database, scores their
    discovery potential based on multiple criteria, and prioritizes them
    within computational budget constraints.
    
    Attributes:
        criteria: Selection criteria with weights
        logger: Logger instance
        variant_database: Database of candidate variants
    """
    
    def __init__(
        self,
        criteria: Optional[SelectionCriteria] = None,
        variant_database_path: Optional[str] = None
    ):
        """Initialize target selection engine.
        
        Args:
            criteria: Selection criteria (uses defaults if None)
            variant_database_path: Path to variant database JSON file
        """
        self.criteria = criteria or SelectionCriteria()
        self.logger = get_logger(__name__)
        self.variant_database: Dict[str, TargetVariant] = {}
        
        # Load variant database if provided
        if variant_database_path:
            self.load_variant_database(variant_database_path)
        else:
            # Initialize with default candidate variants
            self._initialize_default_variants()
        
        self.logger.info("Initialized Target Selection Engine")
        self.logger.info(f"Loaded {len(self.variant_database)} candidate variants")
    
    def _initialize_default_variants(self) -> None:
        """Initialize database with default candidate variants."""
        # This will be populated with 15-20 candidate variants
        # For now, create a few examples
        
        candidates = []
        
        # 1. Long-Range Ising (α=2.0-2.5)
        for alpha in [2.0, 2.1, 2.2, 2.3, 2.4, 2.5]:
            variant = TargetVariant(
                variant_id=f"long_range_ising_alpha_{alpha:.1f}",
                name=f"Long-Range Ising (α={alpha:.1f})",
                description=f"2D Ising model with power-law interactions J(r) ~ r^(-{alpha})",
                model_config=ModelVariantConfig(
                    name=f"long_range_ising_alpha_{alpha:.1f}",
                    dimensions=2,
                    lattice_geometry='square',
                    interaction_type='long_range',
                    interaction_params={'alpha': alpha},
                    theoretical_tc=None,  # Unknown - to be discovered
                    theoretical_exponents=None,
                ),
                theoretical_predictions={
                    'beta': 0.5 if alpha <= 2.0 else None,  # Mean-field for α≤2
                    'nu': 0.5 if alpha <= 2.0 else None,
                },
                literature_references=[
                    "Fisher et al., Phys. Rev. Lett. 29, 917 (1972)",
                    "Luijten & Blöte, Phys. Rev. B 56, 8945 (1997)",
                ],
                priority_score=0.0,  # Will be calculated
                estimated_compute_hours=150.0,  # 64x64, 50 temps, 1000 samples
                discovery_potential=0.0,  # Will be calculated
            )
            candidates.append(variant)
        
        # 2. Diluted Ising (p=0.5-0.9)
        for p in [0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90]:
            variant = TargetVariant(
                variant_id=f"diluted_ising_p_{p:.2f}",
                name=f"Diluted Ising (p={p:.2f})",
                description=f"2D Ising model with site dilution (occupation probability p={p})",
                model_config=ModelVariantConfig(
                    name=f"diluted_ising_p_{p:.2f}",
                    dimensions=2,
                    lattice_geometry='square',
                    interaction_type='nearest_neighbor',
                    disorder_type='quenched',
                    disorder_strength=1.0 - p,
                    theoretical_tc=None,
                    theoretical_exponents={'beta': 0.125, 'nu': 1.0} if p > 0.593 else None,
                ),
                theoretical_predictions={
                    'beta': 0.125 if p > 0.593 else None,
                    'nu': 1.0 if p > 0.593 else None,
                },
                literature_references=[
                    "Stauffer & Aharony, Introduction to Percolation Theory (1994)",
                    "Ballesteros et al., Phys. Rev. B 58, 2740 (1998)",
                ],
                priority_score=0.0,
                estimated_compute_hours=180.0,  # Disorder averaging needed
                discovery_potential=0.0,
            )
            candidates.append(variant)
        
        # 3. Triangular Antiferromagnet
        for j2_j1 in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            variant = TargetVariant(
                variant_id=f"triangular_afm_j2_j1_{j2_j1:.1f}",
                name=f"Triangular AFM (J2/J1={j2_j1:.1f})",
                description=f"Antiferromagnetic Ising on triangular lattice with frustration J2/J1={j2_j1}",
                model_config=ModelVariantConfig(
                    name=f"triangular_afm_j2_j1_{j2_j1:.1f}",
                    dimensions=2,
                    lattice_geometry='triangular',
                    interaction_type='frustrated',
                    interaction_params={'j2_j1_ratio': j2_j1},
                    theoretical_tc=None,
                    theoretical_exponents=None,
                ),
                theoretical_predictions=None,  # Highly uncertain due to frustration
                literature_references=[
                    "Wannier, Phys. Rev. 79, 357 (1950)",
                    "Moessner & Ramirez, Phys. Today 59, 24 (2006)",
                ],
                priority_score=0.0,
                estimated_compute_hours=200.0,  # Frustration slows equilibration
                discovery_potential=0.0,
            )
            candidates.append(variant)
        
        # Store in database
        for variant in candidates:
            self.variant_database[variant.variant_id] = variant
        
        self.logger.info(f"Initialized {len(candidates)} default candidate variants")
    
    def load_variant_database(self, database_path: str) -> None:
        """Load variant database from JSON file.
        
        Args:
            database_path: Path to JSON file containing variant definitions
        """
        path = Path(database_path)
        if not path.exists():
            raise FileNotFoundError(f"Variant database not found: {database_path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Parse variants from JSON
        for variant_data in data.get('variants', []):
            # Reconstruct ModelVariantConfig
            model_config = ModelVariantConfig(**variant_data['model_config'])
            
            # Create TargetVariant
            variant = TargetVariant(
                variant_id=variant_data['variant_id'],
                name=variant_data['name'],
                description=variant_data['description'],
                model_config=model_config,
                theoretical_predictions=variant_data.get('theoretical_predictions'),
                literature_references=variant_data.get('literature_references', []),
                priority_score=variant_data.get('priority_score', 0.0),
                estimated_compute_hours=variant_data.get('estimated_compute_hours', 100.0),
                discovery_potential=variant_data.get('discovery_potential', 0.0),
            )
            
            self.variant_database[variant.variant_id] = variant
        
        self.logger.info(f"Loaded {len(self.variant_database)} variants from {database_path}")
    
    def save_variant_database(self, output_path: str) -> None:
        """Save variant database to JSON file.
        
        Args:
            output_path: Path to save JSON file
        """
        # Convert variants to serializable format
        variants_data = []
        for variant in self.variant_database.values():
            variant_dict = {
                'variant_id': variant.variant_id,
                'name': variant.name,
                'description': variant.description,
                'model_config': {
                    'name': variant.model_config.name,
                    'dimensions': variant.model_config.dimensions,
                    'lattice_geometry': variant.model_config.lattice_geometry,
                    'interaction_type': variant.model_config.interaction_type,
                    'interaction_params': variant.model_config.interaction_params,
                    'disorder_type': variant.model_config.disorder_type,
                    'disorder_strength': variant.model_config.disorder_strength,
                    'external_field': variant.model_config.external_field,
                    'theoretical_tc': variant.model_config.theoretical_tc,
                    'theoretical_exponents': variant.model_config.theoretical_exponents,
                },
                'theoretical_predictions': variant.theoretical_predictions,
                'literature_references': variant.literature_references,
                'priority_score': variant.priority_score,
                'estimated_compute_hours': variant.estimated_compute_hours,
                'discovery_potential': variant.discovery_potential,
            }
            variants_data.append(variant_dict)
        
        data = {'variants': variants_data}
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved variant database to {output_path}")
    
    def identify_candidates(self) -> List[TargetVariant]:
        """Identify candidate variants from database.
        
        Returns:
            List of all candidate variants
        """
        candidates = list(self.variant_database.values())
        self.logger.info(f"Identified {len(candidates)} candidate variants")
        return candidates
    
    def score_discovery_potential(self, variant: TargetVariant) -> float:
        """Score variant's potential for novel physics discovery.
        
        This method combines multiple factors:
        - Theoretical predictions (presence of predictions for novel behavior)
        - Literature gaps (how well-studied is this variant)
        - Computational feasibility (can we afford to explore it)
        - Novelty potential (how different from known systems)
        
        Args:
            variant: Target variant to score
            
        Returns:
            Discovery potential score (0.0 to 1.0)
        """
        # 1. Theoretical prediction score
        theoretical_score = self._score_theoretical_predictions(variant)
        
        # 2. Literature gap score
        literature_score = self._score_literature_gap(variant)
        
        # 3. Computational feasibility score
        feasibility_score = self._score_feasibility(variant)
        
        # 4. Novelty score
        novelty_score = self._score_novelty(variant)
        
        # Weighted combination
        total_score = (
            self.criteria.theoretical_weight * theoretical_score +
            self.criteria.literature_gap_weight * literature_score +
            self.criteria.feasibility_weight * feasibility_score +
            self.criteria.novelty_weight * novelty_score
        )
        
        self.logger.debug(
            f"Variant {variant.variant_id}: "
            f"theory={theoretical_score:.2f}, lit={literature_score:.2f}, "
            f"feas={feasibility_score:.2f}, nov={novelty_score:.2f}, "
            f"total={total_score:.2f}"
        )
        
        return total_score
    
    def _score_theoretical_predictions(self, variant: TargetVariant) -> float:
        """Score based on theoretical predictions.
        
        Higher score if:
        - Predictions exist for novel behavior
        - Predictions are uncertain or controversial
        - Crossover regions predicted
        
        Args:
            variant: Target variant
            
        Returns:
            Score (0.0 to 1.0)
        """
        if variant.theoretical_predictions is None:
            # No predictions = high discovery potential (unexplored territory)
            return 0.9
        
        # Check if predictions suggest novel behavior
        predictions = variant.theoretical_predictions
        
        # Check for None values (uncertain predictions)
        uncertain_count = sum(1 for v in predictions.values() if v is None)
        if uncertain_count > 0:
            # Uncertainty suggests discovery potential
            return 0.8
        
        # Check if predictions deviate from standard universality classes
        # Standard 2D Ising: beta=0.125, nu=1.0, gamma=1.75
        # Standard 3D Ising: beta=0.326, nu=0.630, gamma=1.237
        # Mean-field: beta=0.5, nu=0.5, gamma=1.0
        
        known_classes = [
            {'beta': 0.125, 'nu': 1.0, 'gamma': 1.75},  # 2D Ising
            {'beta': 0.326, 'nu': 0.630, 'gamma': 1.237},  # 3D Ising
            {'beta': 0.5, 'nu': 0.5, 'gamma': 1.0},  # Mean-field
        ]
        
        # Calculate minimum distance to known classes
        min_distance = float('inf')
        for known_class in known_classes:
            distance = 0.0
            count = 0
            for exp_name in ['beta', 'nu', 'gamma']:
                if exp_name in predictions and exp_name in known_class:
                    if predictions[exp_name] is not None:
                        distance += abs(predictions[exp_name] - known_class[exp_name])
                        count += 1
            if count > 0:
                distance /= count
                min_distance = min(min_distance, distance)
        
        # Higher distance = more novel = higher score
        if min_distance == float('inf'):
            return 0.5
        
        # Normalize distance to [0, 1] range
        # Assume distance > 0.2 is significantly different
        score = min(1.0, min_distance / 0.2)
        return score
    
    def _score_literature_gap(self, variant: TargetVariant) -> float:
        """Score based on literature gaps.
        
        Higher score if:
        - Few literature references (understudied)
        - Old references (needs modern study)
        - Conflicting results in literature
        
        Args:
            variant: Target variant
            
        Returns:
            Score (0.0 to 1.0)
        """
        n_refs = len(variant.literature_references)
        
        if n_refs == 0:
            # No references = completely unexplored
            return 1.0
        elif n_refs <= 2:
            # Few references = understudied
            return 0.8
        elif n_refs <= 5:
            # Moderate coverage
            return 0.5
        else:
            # Well-studied
            return 0.2
    
    def _score_feasibility(self, variant: TargetVariant) -> float:
        """Score computational feasibility.
        
        Higher score if:
        - Reasonable computational cost
        - Standard lattice geometry
        - No extreme parameter values
        
        Args:
            variant: Target variant
            
        Returns:
            Score (0.0 to 1.0)
        """
        # Score based on estimated compute hours
        # Assume 100 hours is ideal, >500 hours is too expensive
        hours = variant.estimated_compute_hours
        
        if hours <= 100:
            compute_score = 1.0
        elif hours <= 200:
            compute_score = 0.8
        elif hours <= 300:
            compute_score = 0.6
        elif hours <= 500:
            compute_score = 0.4
        else:
            compute_score = 0.2
        
        # Score based on lattice geometry complexity
        standard_geometries = ['square', 'cubic']
        if variant.model_config.lattice_geometry in standard_geometries:
            geometry_score = 1.0
        else:
            geometry_score = 0.7  # More complex but still feasible
        
        # Average scores
        return (compute_score + geometry_score) / 2.0
    
    def _score_novelty(self, variant: TargetVariant) -> float:
        """Score novelty potential.
        
        Higher score if:
        - Long-range interactions
        - Frustration present
        - Disorder effects
        - Non-standard geometry
        
        Args:
            variant: Target variant
            
        Returns:
            Score (0.0 to 1.0)
        """
        score = 0.0
        
        # Long-range interactions
        if variant.model_config.interaction_type == 'long_range':
            score += 0.3
        
        # Frustration
        if variant.model_config.interaction_type == 'frustrated':
            score += 0.3
        
        # Disorder
        if variant.model_config.disorder_type is not None:
            score += 0.2
        
        # Non-standard geometry
        if variant.model_config.lattice_geometry not in ['square', 'cubic']:
            score += 0.2
        
        return min(1.0, score)
    
    def prioritize_targets(
        self,
        candidates: Optional[List[TargetVariant]] = None,
        budget: Optional[float] = None
    ) -> List[TargetVariant]:
        """Prioritize variants within computational budget.
        
        Args:
            candidates: List of candidate variants (uses all if None)
            budget: Computational budget in GPU-hours (no limit if None)
            
        Returns:
            Prioritized list of variants within budget
        """
        if candidates is None:
            candidates = self.identify_candidates()
        
        # Score all candidates
        for variant in candidates:
            variant.discovery_potential = self.score_discovery_potential(variant)
            variant.priority_score = variant.discovery_potential
        
        # Sort by priority score (descending)
        sorted_variants = sorted(
            candidates,
            key=lambda v: v.priority_score,
            reverse=True
        )
        
        # Apply budget constraint if specified
        if budget is not None:
            selected = []
            total_cost = 0.0
            
            for variant in sorted_variants:
                if total_cost + variant.estimated_compute_hours <= budget:
                    selected.append(variant)
                    total_cost += variant.estimated_compute_hours
                else:
                    self.logger.info(
                        f"Budget limit reached. Excluding {variant.variant_id} "
                        f"(would exceed budget by {total_cost + variant.estimated_compute_hours - budget:.1f} hours)"
                    )
            
            self.logger.info(
                f"Selected {len(selected)}/{len(sorted_variants)} variants "
                f"within budget of {budget:.1f} GPU-hours "
                f"(total cost: {total_cost:.1f} GPU-hours)"
            )
            
            return selected
        
        return sorted_variants
    
    def get_variant(self, variant_id: str) -> Optional[TargetVariant]:
        """Get variant by ID.
        
        Args:
            variant_id: Variant identifier
            
        Returns:
            TargetVariant if found, None otherwise
        """
        return self.variant_database.get(variant_id)
    
    def add_variant(self, variant: TargetVariant) -> None:
        """Add variant to database.
        
        Args:
            variant: Target variant to add
        """
        self.variant_database[variant.variant_id] = variant
        self.logger.info(f"Added variant: {variant.variant_id}")
    
    def remove_variant(self, variant_id: str) -> None:
        """Remove variant from database.
        
        Args:
            variant_id: Variant identifier
        """
        if variant_id in self.variant_database:
            del self.variant_database[variant_id]
            self.logger.info(f"Removed variant: {variant_id}")
        else:
            self.logger.warning(f"Variant not found: {variant_id}")
    
    def generate_selection_report(
        self,
        prioritized_variants: List[TargetVariant],
        output_path: str
    ) -> None:
        """Generate detailed selection report.
        
        Args:
            prioritized_variants: List of prioritized variants
            output_path: Path to save report
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("TARGET VARIANT SELECTION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        report_lines.append(f"Total candidates evaluated: {len(self.variant_database)}")
        report_lines.append(f"Variants selected: {len(prioritized_variants)}")
        report_lines.append("")
        
        total_cost = sum(v.estimated_compute_hours for v in prioritized_variants)
        report_lines.append(f"Total estimated cost: {total_cost:.1f} GPU-hours")
        report_lines.append("")
        
        report_lines.append("Selection Criteria Weights:")
        report_lines.append(f"  Theoretical predictions: {self.criteria.theoretical_weight:.2f}")
        report_lines.append(f"  Literature gaps: {self.criteria.literature_gap_weight:.2f}")
        report_lines.append(f"  Computational feasibility: {self.criteria.feasibility_weight:.2f}")
        report_lines.append(f"  Novelty potential: {self.criteria.novelty_weight:.2f}")
        report_lines.append("")
        
        report_lines.append("=" * 80)
        report_lines.append("PRIORITIZED VARIANTS")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        for i, variant in enumerate(prioritized_variants, 1):
            report_lines.append(f"{i}. {variant.name}")
            report_lines.append(f"   ID: {variant.variant_id}")
            report_lines.append(f"   Priority Score: {variant.priority_score:.3f}")
            report_lines.append(f"   Discovery Potential: {variant.discovery_potential:.3f}")
            report_lines.append(f"   Estimated Cost: {variant.estimated_compute_hours:.1f} GPU-hours")
            report_lines.append(f"   Description: {variant.description}")
            
            if variant.theoretical_predictions:
                report_lines.append(f"   Theoretical Predictions: {variant.theoretical_predictions}")
            
            report_lines.append(f"   Literature References: {len(variant.literature_references)}")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        # Write report
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Generated selection report: {output_path}")
