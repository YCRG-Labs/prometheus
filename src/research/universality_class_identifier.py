# -*- coding: utf-8 -*-
"""
New Universality Class Identification.

This module provides tools for identifying and characterizing new universality
classes from multi-variant comparative analysis.

Requirements 6.5, 10.4, 10.5: Suggest relationships between variant properties
and universality classes, compare new classes across multiple variants to
establish universality, suggest names and theoretical interpretations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

from .variant_database import VariantDatabase, VariantRecord
from .multi_variant_analyzer import MultiVariantAnalyzer, ClusterResult
from .phenomena_detector import UniversalityClass
from ..utils.logging_utils import get_logger


@dataclass
class ProposedUniversalityClass:
    """A proposed new universality class.
    
    Attributes:
        class_name: Proposed name for the class
        member_variants: Variants belonging to this class
        characteristic_exponents: Characteristic critical exponents
        exponent_errors: Errors on exponents
        scaling_relations_verified: Whether scaling relations hold
        universality_confidence: Confidence in universality (0.0 to 1.0)
        distinguishing_features: Features that distinguish this class
        theoretical_interpretation: Proposed theoretical interpretation
        literature_support: Supporting literature references
    """
    class_name: str
    member_variants: List[str]
    characteristic_exponents: Dict[str, float]
    exponent_errors: Dict[str, float]
    scaling_relations_verified: bool
    universality_confidence: float
    distinguishing_features: List[str]
    theoretical_interpretation: str
    literature_support: List[str] = field(default_factory=list)


@dataclass
class UniversalityRelationship:
    """Relationship between variant properties and universality classes.
    
    Attributes:
        property_name: Name of variant property
        universality_class: Associated universality class
        relationship_type: Type of relationship ('determines', 'influences', 'correlates')
        strength: Strength of relationship (0.0 to 1.0)
        description: Human-readable description
    """
    property_name: str
    universality_class: str
    relationship_type: str
    strength: float
    description: str



class UniversalityClassIdentifier:
    """Identify and characterize new universality classes.
    
    This class analyzes clusters of variants to determine if they represent
    new universality classes, verifies scaling relations, and proposes
    theoretical interpretations.
    
    Attributes:
        database: Variant database
        analyzer: Multi-variant analyzer
        logger: Logger instance
    """
    
    def __init__(self, database: VariantDatabase):
        """Initialize universality class identifier.
        
        Args:
            database: Variant database
        """
        self.database = database
        self.analyzer = MultiVariantAnalyzer(database)
        self.logger = get_logger(__name__)
        self.logger.info("Initialized UniversalityClassIdentifier")
    
    def identify_new_classes(
        self,
        clusters: List[ClusterResult],
        deviation_threshold: float = 3.0,
        min_members: int = 2
    ) -> List[ProposedUniversalityClass]:
        """Identify potential new universality classes from clusters.
        
        Requirement 6.5: Suggest relationships between variant properties and
        universality classes.
        
        Requirement 10.4: Compare new classes across multiple variants to
        establish universality.
        
        Args:
            clusters: Cluster results from comparative analysis
            deviation_threshold: Threshold for deviation from known classes (sigma)
            min_members: Minimum number of variants to propose new class
            
        Returns:
            List of proposed new universality classes
        """
        self.logger.info(
            f"Identifying new universality classes from {len(clusters)} clusters"
        )
        
        proposed_classes = []
        
        for cluster in clusters:
            if cluster.cluster_size < min_members:
                continue
            
            # Check if cluster deviates from all known classes
            is_novel, max_deviation = self._check_novelty(
                cluster, deviation_threshold
            )
            
            if not is_novel:
                continue
            
            # Verify scaling relations
            scaling_verified = self._verify_scaling_relations(cluster)
            
            # Check universality across cluster members
            universality_confidence = self._assess_universality(cluster)
            
            if universality_confidence < 0.7:
                self.logger.info(
                    f"Cluster {cluster.cluster_id} has low universality confidence "
                    f"({universality_confidence:.2f}), skipping"
                )
                continue
            
            # Identify distinguishing features
            features = self._identify_distinguishing_features(cluster)
            
            # Propose name
            class_name = self._propose_class_name(cluster, features)
            
            # Generate theoretical interpretation
            interpretation = self._generate_interpretation(cluster, features)
            
            # Compute exponent errors
            exponent_errors = self._compute_cluster_exponent_errors(cluster)
            
            proposed_class = ProposedUniversalityClass(
                class_name=class_name,
                member_variants=cluster.variant_ids,
                characteristic_exponents=cluster.centroid_exponents,
                exponent_errors=exponent_errors,
                scaling_relations_verified=scaling_verified,
                universality_confidence=universality_confidence,
                distinguishing_features=features,
                theoretical_interpretation=interpretation
            )
            
            proposed_classes.append(proposed_class)
            
            self.logger.info(
                f"Proposed new universality class: {class_name} "
                f"({cluster.cluster_size} members, confidence: {universality_confidence:.2f})"
            )
        
        return proposed_classes

    
    def identify_relationships(
        self,
        proposed_classes: List[ProposedUniversalityClass]
    ) -> List[UniversalityRelationship]:
        """Identify relationships between variant properties and universality classes.
        
        Requirement 6.5: Suggest relationships between variant properties and
        universality classes.
        
        Args:
            proposed_classes: List of proposed universality classes
            
        Returns:
            List of identified relationships
        """
        self.logger.info("Identifying variant property relationships")
        
        relationships = []
        
        for proposed_class in proposed_classes:
            # Get variants in this class
            variants = [
                self.database.get_variant(vid)
                for vid in proposed_class.member_variants
            ]
            variants = [v for v in variants if v is not None]
            
            if not variants:
                continue
            
            # Analyze common properties
            # 1. Dimensions
            dimensions = [v.variant_config.dimensions for v in variants]
            if len(set(dimensions)) == 1:
                # All same dimension
                dim = dimensions[0]
                relationships.append(UniversalityRelationship(
                    property_name='dimensions',
                    universality_class=proposed_class.class_name,
                    relationship_type='determines',
                    strength=1.0,
                    description=f"All members are {dim}D systems"
                ))
            
            # 2. Interaction type
            interaction_types = [v.variant_config.interaction_type for v in variants]
            if len(set(interaction_types)) == 1:
                itype = interaction_types[0]
                relationships.append(UniversalityRelationship(
                    property_name='interaction_type',
                    universality_class=proposed_class.class_name,
                    relationship_type='determines',
                    strength=1.0,
                    description=f"All members have {itype} interactions"
                ))
            
            # 3. Lattice geometry
            geometries = [v.variant_config.lattice_geometry for v in variants]
            if len(set(geometries)) == 1:
                geom = geometries[0]
                relationships.append(UniversalityRelationship(
                    property_name='lattice_geometry',
                    universality_class=proposed_class.class_name,
                    relationship_type='influences',
                    strength=0.8,
                    description=f"All members use {geom} lattice"
                ))
            
            # 4. Disorder
            has_disorder = [v.variant_config.disorder_strength > 0 for v in variants]
            if all(has_disorder):
                relationships.append(UniversalityRelationship(
                    property_name='disorder',
                    universality_class=proposed_class.class_name,
                    relationship_type='influences',
                    strength=0.9,
                    description="All members have disorder"
                ))
            elif not any(has_disorder):
                relationships.append(UniversalityRelationship(
                    property_name='disorder',
                    universality_class=proposed_class.class_name,
                    relationship_type='influences',
                    strength=0.9,
                    description="No members have disorder"
                ))
            
            # 5. Interaction parameters
            # Check if specific parameter ranges correlate with this class
            param_names = set()
            for v in variants:
                param_names.update(v.variant_config.interaction_params.keys())
            
            for param_name in param_names:
                param_values = []
                for v in variants:
                    if param_name in v.variant_config.interaction_params:
                        param_values.append(v.variant_config.interaction_params[param_name])
                
                if len(param_values) >= 2:
                    # Check if parameter values are clustered
                    param_std = np.std(param_values)
                    param_mean = np.mean(param_values)
                    
                    if param_std / (abs(param_mean) + 1e-10) < 0.3:
                        # Tightly clustered
                        relationships.append(UniversalityRelationship(
                            property_name=param_name,
                            universality_class=proposed_class.class_name,
                            relationship_type='correlates',
                            strength=0.7,
                            description=f"{param_name} ≈ {param_mean:.3f} ± {param_std:.3f}"
                        ))
        
        self.logger.info(f"Identified {len(relationships)} relationships")
        return relationships

    
    def _check_novelty(
        self,
        cluster: ClusterResult,
        threshold: float
    ) -> Tuple[bool, float]:
        """Check if cluster represents novel physics.
        
        Args:
            cluster: Cluster result
            threshold: Deviation threshold in sigma
            
        Returns:
            Tuple of (is_novel, max_deviation)
        """
        from .phenomena_detector import NovelPhenomenonDetector
        
        detector = NovelPhenomenonDetector()
        
        # Check cluster centroid against all known classes
        max_deviation = 0.0
        
        for class_name, univ_class in detector.universality_classes.items():
            # Compute deviations for each exponent
            deviations = []
            for exp_name, measured_value in cluster.centroid_exponents.items():
                if exp_name in univ_class.exponents:
                    theoretical_value = univ_class.exponents[exp_name]
                    theoretical_error = univ_class.exponent_errors.get(exp_name, 0.01)
                    
                    # Use cluster variance as measurement error
                    measured_error = np.sqrt(cluster.intra_cluster_variance)
                    combined_error = np.sqrt(measured_error**2 + theoretical_error**2)
                    
                    deviation = abs(measured_value - theoretical_value) / combined_error
                    deviations.append(deviation)
            
            if deviations:
                class_max_dev = max(deviations)
                max_deviation = max(max_deviation, class_max_dev)
        
        is_novel = max_deviation > threshold
        
        return is_novel, max_deviation
    
    def _verify_scaling_relations(self, cluster: ClusterResult) -> bool:
        """Verify scaling relations for cluster.
        
        Args:
            cluster: Cluster result
            
        Returns:
            True if scaling relations are satisfied
        """
        exponents = cluster.centroid_exponents
        
        # Check hyperscaling relation: α + 2β + γ = 2
        if all(exp in exponents for exp in ['alpha', 'beta', 'gamma']):
            alpha = exponents['alpha']
            beta = exponents['beta']
            gamma = exponents['gamma']
            
            hyperscaling = alpha + 2 * beta + gamma
            
            # Allow 10% tolerance
            if abs(hyperscaling - 2.0) > 0.2:
                self.logger.warning(
                    f"Cluster {cluster.cluster_id} violates hyperscaling: "
                    f"α + 2β + γ = {hyperscaling:.3f} (expected 2.0)"
                )
                return False
        
        # Check γ = ν(2 - η) if all present
        if all(exp in exponents for exp in ['gamma', 'nu', 'eta']):
            gamma = exponents['gamma']
            nu = exponents['nu']
            eta = exponents['eta']
            
            expected_gamma = nu * (2 - eta)
            
            if abs(gamma - expected_gamma) / expected_gamma > 0.15:
                self.logger.warning(
                    f"Cluster {cluster.cluster_id} violates γ = ν(2-η): "
                    f"γ = {gamma:.3f}, ν(2-η) = {expected_gamma:.3f}"
                )
                return False
        
        return True
    
    def _assess_universality(self, cluster: ClusterResult) -> float:
        """Assess universality confidence for cluster.
        
        Universality means all members have similar exponents despite
        different microscopic details.
        
        Args:
            cluster: Cluster result
            
        Returns:
            Universality confidence (0.0 to 1.0)
        """
        # Get variants in cluster
        variants = [
            self.database.get_variant(vid)
            for vid in cluster.variant_ids
        ]
        variants = [v for v in variants if v is not None]
        
        if len(variants) < 2:
            return 0.0
        
        # Check diversity of microscopic properties
        dimensions = set(v.variant_config.dimensions for v in variants)
        geometries = set(v.variant_config.lattice_geometry for v in variants)
        interaction_types = set(v.variant_config.interaction_type for v in variants)
        
        # More diversity = stronger evidence for universality
        diversity_score = (
            (len(dimensions) > 1) * 0.3 +
            (len(geometries) > 1) * 0.3 +
            (len(interaction_types) > 1) * 0.4
        )
        
        # Check consistency of exponents
        # Low intra-cluster variance = high consistency
        consistency_score = np.exp(-cluster.intra_cluster_variance * 10)
        
        # Combine scores
        universality_confidence = 0.5 * diversity_score + 0.5 * consistency_score
        
        return float(universality_confidence)

    
    def _identify_distinguishing_features(
        self,
        cluster: ClusterResult
    ) -> List[str]:
        """Identify distinguishing features of cluster.
        
        Args:
            cluster: Cluster result
            
        Returns:
            List of distinguishing features
        """
        features = []
        
        # Get variants
        variants = [
            self.database.get_variant(vid)
            for vid in cluster.variant_ids
        ]
        variants = [v for v in variants if v is not None]
        
        if not variants:
            return features
        
        # Check for common features
        # 1. Dimensions
        dimensions = [v.variant_config.dimensions for v in variants]
        if len(set(dimensions)) == 1:
            features.append(f"{dimensions[0]}D systems")
        
        # 2. Interaction type
        interaction_types = [v.variant_config.interaction_type for v in variants]
        if len(set(interaction_types)) == 1:
            features.append(f"{interaction_types[0]} interactions")
        
        # 3. Disorder
        has_disorder = [v.variant_config.disorder_strength > 0 for v in variants]
        if all(has_disorder):
            features.append("disordered systems")
        elif not any(has_disorder):
            features.append("clean systems")
        
        # 4. Lattice geometry
        geometries = [v.variant_config.lattice_geometry for v in variants]
        if len(set(geometries)) == 1:
            features.append(f"{geometries[0]} lattice")
        
        # 5. Exponent characteristics
        exponents = cluster.centroid_exponents
        
        if 'beta' in exponents:
            beta = exponents['beta']
            if beta < 0.2:
                features.append("small β (weak ordering)")
            elif beta > 0.4:
                features.append("large β (strong ordering)")
        
        if 'nu' in exponents:
            nu = exponents['nu']
            if nu < 0.6:
                features.append("small ν (short correlation length)")
            elif nu > 1.0:
                features.append("large ν (long correlation length)")
        
        return features
    
    def _propose_class_name(
        self,
        cluster: ClusterResult,
        features: List[str]
    ) -> str:
        """Propose name for new universality class.
        
        Requirement 10.5: Suggest names for new classes.
        
        Args:
            cluster: Cluster result
            features: Distinguishing features
            
        Returns:
            Proposed class name
        """
        # Get representative variant
        rep_variant = self.database.get_variant(cluster.representative_variant)
        
        if rep_variant is None:
            return f"Novel_Class_{cluster.cluster_id}"
        
        # Build name from features
        name_parts = []
        
        # Dimension
        dim = rep_variant.variant_config.dimensions
        name_parts.append(f"{dim}D")
        
        # Interaction type
        itype = rep_variant.variant_config.interaction_type
        if itype != 'nearest_neighbor':
            name_parts.append(itype.replace('_', '-'))
        
        # Lattice geometry
        geom = rep_variant.variant_config.lattice_geometry
        if geom not in ['square', 'cubic']:
            name_parts.append(geom)
        
        # Disorder
        if rep_variant.variant_config.disorder_strength > 0:
            name_parts.append("disordered")
        
        # Add "Ising" if not already implied
        if 'ising' not in rep_variant.variant_config.name.lower():
            name_parts.append("Ising")
        
        class_name = "_".join(name_parts)
        
        return class_name
    
    def _generate_interpretation(
        self,
        cluster: ClusterResult,
        features: List[str]
    ) -> str:
        """Generate theoretical interpretation for new class.
        
        Requirement 10.5: Suggest theoretical interpretations for new classes.
        
        Args:
            cluster: Cluster result
            features: Distinguishing features
            
        Returns:
            Theoretical interpretation
        """
        interpretation_parts = []
        
        # Get variants
        variants = [
            self.database.get_variant(vid)
            for vid in cluster.variant_ids
        ]
        variants = [v for v in variants if v is not None]
        
        if not variants:
            return "Insufficient data for interpretation"
        
        # Analyze exponents
        exponents = cluster.centroid_exponents
        
        # Compare to mean-field
        if 'beta' in exponents and 'nu' in exponents:
            beta = exponents['beta']
            nu = exponents['nu']
            
            # Mean-field: β=0.5, ν=0.5
            if abs(beta - 0.5) < 0.1 and abs(nu - 0.5) < 0.1:
                interpretation_parts.append(
                    "Exponents consistent with mean-field behavior, "
                    "suggesting effective dimensionality d ≥ 4 or long-range interactions."
                )
            elif beta < 0.3 and nu > 0.8:
                interpretation_parts.append(
                    "Small β and large ν suggest weak first-order transition "
                    "or proximity to lower critical dimension."
                )
            elif beta > 0.4 and nu < 0.7:
                interpretation_parts.append(
                    "Large β and small ν suggest strong ordering "
                    "with short correlation length."
                )
        
        # Check for disorder effects
        has_disorder = any(v.variant_config.disorder_strength > 0 for v in variants)
        if has_disorder:
            interpretation_parts.append(
                "Disorder may modify critical behavior through "
                "Harris criterion or induce new fixed points."
            )
        
        # Check for frustration
        geometries = [v.variant_config.lattice_geometry for v in variants]
        frustrated_geometries = ['triangular', 'kagome', 'pyrochlore']
        if any(g in frustrated_geometries for g in geometries):
            interpretation_parts.append(
                "Geometric frustration may lead to exotic phases "
                "or modified critical behavior."
            )
        
        # Check for long-range interactions
        interaction_types = [v.variant_config.interaction_type for v in variants]
        if 'long_range' in interaction_types:
            interpretation_parts.append(
                "Long-range interactions can modify universality class "
                "depending on decay exponent."
            )
        
        if not interpretation_parts:
            interpretation_parts.append(
                "Novel universality class with exponents deviating from "
                "known classes. Further theoretical investigation needed."
            )
        
        return " ".join(interpretation_parts)
    
    def _compute_cluster_exponent_errors(
        self,
        cluster: ClusterResult
    ) -> Dict[str, float]:
        """Compute exponent errors for cluster.
        
        Args:
            cluster: Cluster result
            
        Returns:
            Dictionary of exponent errors
        """
        # Get variants
        variants = [
            self.database.get_variant(vid)
            for vid in cluster.variant_ids
        ]
        variants = [v for v in variants if v is not None]
        
        if not variants:
            return {}
        
        # Collect exponent values
        exponent_values = {}
        for variant in variants:
            for exp_name, exp_value in variant.measured_exponents.items():
                if exp_name not in exponent_values:
                    exponent_values[exp_name] = []
                exponent_values[exp_name].append(exp_value)
        
        # Compute standard error of mean
        exponent_errors = {}
        for exp_name, values in exponent_values.items():
            if len(values) > 1:
                exponent_errors[exp_name] = float(np.std(values) / np.sqrt(len(values)))
            else:
                exponent_errors[exp_name] = 0.05  # Default error
        
        return exponent_errors
    
    def generate_class_report(
        self,
        proposed_class: ProposedUniversalityClass,
        relationships: List[UniversalityRelationship],
        output_file: Optional[Path] = None
    ) -> str:
        """Generate detailed report for proposed universality class.
        
        Args:
            proposed_class: Proposed universality class
            relationships: Related relationships
            output_file: Optional file to save report
            
        Returns:
            Report text
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"PROPOSED UNIVERSALITY CLASS: {proposed_class.class_name}")
        lines.append("=" * 80)
        lines.append("")
        
        # Basic info
        lines.append(f"Number of Members: {len(proposed_class.member_variants)}")
        lines.append(f"Universality Confidence: {proposed_class.universality_confidence:.2%}")
        lines.append(f"Scaling Relations Verified: {proposed_class.scaling_relations_verified}")
        lines.append("")
        
        # Characteristic exponents
        lines.append("Characteristic Critical Exponents:")
        for exp_name, exp_value in proposed_class.characteristic_exponents.items():
            error = proposed_class.exponent_errors.get(exp_name, 0.0)
            lines.append(f"  {exp_name}: {exp_value:.4f} ± {error:.4f}")
        lines.append("")
        
        # Distinguishing features
        lines.append("Distinguishing Features:")
        for feature in proposed_class.distinguishing_features:
            lines.append(f"  - {feature}")
        lines.append("")
        
        # Theoretical interpretation
        lines.append("Theoretical Interpretation:")
        lines.append(f"  {proposed_class.theoretical_interpretation}")
        lines.append("")
        
        # Member variants
        lines.append("Member Variants:")
        for variant_id in proposed_class.member_variants:
            variant = self.database.get_variant(variant_id)
            if variant:
                lines.append(f"  - {variant_id} ({variant.variant_config.name})")
        lines.append("")
        
        # Relationships
        class_relationships = [
            r for r in relationships
            if r.universality_class == proposed_class.class_name
        ]
        if class_relationships:
            lines.append("Property Relationships:")
            for rel in class_relationships:
                lines.append(
                    f"  - {rel.property_name} {rel.relationship_type} "
                    f"(strength: {rel.strength:.2f}): {rel.description}"
                )
        
        lines.append("")
        lines.append("=" * 80)
        
        report_text = "\n".join(lines)
        
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            self.logger.info(f"Saved class report to {output_file}")
        
        return report_text
