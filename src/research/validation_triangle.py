"""
Validation Triangle Cross-Validation System

This module implements a novel methodological enhancement: the validation triangle,
which provides an over-determined system for robust validation by checking consistency
between three independent aspects of phase transition behavior:

1. Critical Exponents (measured values)
2. Universality Class (theoretical classification)
3. Scaling Relations (theoretical constraints)

The key insight: All three must be mutually consistent. If any two are satisfied,
the third is constrained. This creates powerful redundancy for validation and can
detect inconsistencies that indicate either:
- Measurement errors
- Wrong universality class assignment
- Violations of scaling relations (novel physics)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .base_types import VAEAnalysisResults
from .phenomena_detector import NovelPhenomenonDetector, UniversalityClass
from .comparative_analyzer import ComparativeAnalyzer, ScalingViolation
from ..utils.logging_utils import get_logger


class TriangleVertex(Enum):
    """Vertices of the validation triangle."""
    EXPONENTS = "exponents"
    UNIVERSALITY_CLASS = "universality_class"
    SCALING_RELATIONS = "scaling_relations"


class ConsistencyStatus(Enum):
    """Status of consistency check."""
    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    INSUFFICIENT_DATA = "insufficient_data"
    CONSTRAINED = "constrained"


@dataclass
class TriangleEdge:
    """Edge connecting two vertices of the validation triangle.
    
    Attributes:
        vertex1: First vertex
        vertex2: Second vertex
        consistent: Whether the two vertices are consistent
        confidence: Confidence in consistency (0.0 to 1.0)
        details: Detailed information about the consistency check
    """
    vertex1: TriangleVertex
    vertex2: TriangleVertex
    consistent: bool
    confidence: float
    details: Dict[str, any]


@dataclass
class TriangleValidation:
    """Result of validation triangle cross-validation.
    
    Attributes:
        overall_status: Overall consistency status
        overall_confidence: Overall confidence in validation
        edges: Consistency checks for each edge
        inconsistencies: List of detected inconsistencies
        recommendations: Recommended actions
        message: Human-readable summary
    """
    overall_status: ConsistencyStatus
    overall_confidence: float
    edges: List[TriangleEdge]
    inconsistencies: List[str]
    recommendations: List[str]
    message: str


class ValidationTriangle:
    """Validation triangle cross-validation system.
    
    This class implements three-way consistency checking between:
    1. Measured critical exponents
    2. Universality class assignment
    3. Scaling relation constraints
    
    The validation triangle provides an over-determined system where:
    - If exponents + universality class agree → scaling relations constrained
    - If exponents + scaling relations agree → universality class constrained
    - If universality class + scaling relations agree → exponents constrained
    
    Inconsistencies indicate potential issues:
    - All three inconsistent → measurement error or data quality issue
    - Two consistent, one inconsistent → identifies the problematic vertex
    
    Attributes:
        detector: NovelPhenomenonDetector for universality class checks
        analyzer: ComparativeAnalyzer for scaling relation checks
        consistency_threshold: Threshold for consistency (in sigma)
        logger: Logger instance
    """
    
    # Known scaling relations
    SCALING_RELATIONS = {
        'hyperscaling': {
            'formula': 'alpha + 2*beta + gamma = 2',
            'tolerance': 0.15,  # Tolerance for relation
            'description': 'Hyperscaling relation'
        },
        'gamma_nu_relation': {
            'formula': 'gamma = nu * (2 - eta)',
            'tolerance': 0.20,
            'description': 'Gamma-nu scaling relation'
        },
        'alpha_nu_relation_2d': {
            'formula': 'alpha = 2 - 2*nu',  # For d=2
            'tolerance': 0.15,
            'description': 'Alpha-nu relation (2D)'
        },
        'alpha_nu_relation_3d': {
            'formula': 'alpha = 2 - 3*nu',  # For d=3
            'tolerance': 0.15,
            'description': 'Alpha-nu relation (3D)'
        }
    }
    
    def __init__(
        self,
        anomaly_threshold: float = 3.0,
        consistency_threshold: float = 2.0
    ):
        """Initialize validation triangle.
        
        Args:
            anomaly_threshold: Threshold for anomaly detection (sigma)
            consistency_threshold: Threshold for consistency checks (sigma)
        """
        self.detector = NovelPhenomenonDetector(anomaly_threshold)
        self.analyzer = ComparativeAnalyzer(anomaly_threshold)
        self.consistency_threshold = consistency_threshold
        self.logger = get_logger(__name__)
        self.logger.info(
            f"Initialized ValidationTriangle with consistency threshold {consistency_threshold}sigma"
        )
    
    def validate(
        self,
        vae_results: VAEAnalysisResults,
        expected_universality_class: Optional[str] = None,
        dimensions: int = 2
    ) -> TriangleValidation:
        """Perform validation triangle cross-validation.
        
        Checks consistency between exponents, universality class, and scaling relations.
        
        Args:
            vae_results: VAE analysis results with measured exponents
            expected_universality_class: Optional expected universality class
            dimensions: System dimensions (2 or 3) for scaling relations
            
        Returns:
            TriangleValidation with consistency results
        """
        self.logger.info(
            f"Performing validation triangle cross-validation for variant '{vae_results.variant_id}'"
        )
        
        edges = []
        inconsistencies = []
        recommendations = []
        
        # Edge 1: Exponents <-> Universality Class
        exp_univ_edge = self._check_exponents_universality_consistency(
            vae_results, expected_universality_class
        )
        edges.append(exp_univ_edge)
        
        if not exp_univ_edge.consistent:
            inconsistencies.append(
                f"Exponents inconsistent with universality class: {exp_univ_edge.details.get('message', 'Unknown')}"
            )
        
        # Edge 2: Exponents <-> Scaling Relations
        exp_scaling_edge = self._check_exponents_scaling_consistency(
            vae_results, dimensions
        )
        edges.append(exp_scaling_edge)
        
        if not exp_scaling_edge.consistent:
            inconsistencies.append(
                f"Exponents violate scaling relations: {exp_scaling_edge.details.get('message', 'Unknown')}"
            )
        
        # Edge 3: Universality Class <-> Scaling Relations
        univ_scaling_edge = self._check_universality_scaling_consistency(
            vae_results, expected_universality_class, dimensions
        )
        edges.append(univ_scaling_edge)
        
        if not univ_scaling_edge.consistent:
            inconsistencies.append(
                f"Universality class inconsistent with scaling relations: {univ_scaling_edge.details.get('message', 'Unknown')}"
            )
        
        # Determine overall status and confidence
        n_consistent = sum(1 for edge in edges if edge.consistent)
        avg_confidence = np.mean([edge.confidence for edge in edges])
        
        if n_consistent == 3:
            overall_status = ConsistencyStatus.CONSISTENT
            overall_confidence = avg_confidence
            message = "All three vertices of validation triangle are mutually consistent"
        elif n_consistent == 2:
            overall_status = ConsistencyStatus.CONSTRAINED
            overall_confidence = avg_confidence * 0.8  # Reduced confidence
            # Identify the inconsistent vertex
            inconsistent_vertex = self._identify_inconsistent_vertex(edges)
            message = f"Two vertices consistent, {inconsistent_vertex.value} appears inconsistent"
            recommendations.append(f"Investigate {inconsistent_vertex.value} for potential issues")
        elif n_consistent == 1:
            overall_status = ConsistencyStatus.INCONSISTENT
            overall_confidence = avg_confidence * 0.5
            message = "Multiple inconsistencies detected in validation triangle"
            recommendations.append("Check data quality and measurement errors")
            recommendations.append("Consider alternative universality class")
        else:
            overall_status = ConsistencyStatus.INCONSISTENT
            overall_confidence = avg_confidence * 0.3
            message = "All vertices of validation triangle are inconsistent"
            recommendations.append("Likely measurement error or insufficient data")
            recommendations.append("Increase simulation quality or sample size")
        
        # Generate specific recommendations based on pattern
        if n_consistent == 2:
            recommendations.extend(
                self._generate_constrained_recommendations(edges, vae_results)
            )
        
        result = TriangleValidation(
            overall_status=overall_status,
            overall_confidence=overall_confidence,
            edges=edges,
            inconsistencies=inconsistencies,
            recommendations=recommendations,
            message=message
        )
        
        self.logger.info(
            f"Validation triangle: {n_consistent}/3 edges consistent, "
            f"status={overall_status.value}, confidence={overall_confidence:.2%}"
        )
        
        return result
    
    def _check_exponents_universality_consistency(
        self,
        vae_results: VAEAnalysisResults,
        expected_class: Optional[str] = None
    ) -> TriangleEdge:
        """Check consistency between measured exponents and universality class.
        
        Args:
            vae_results: VAE analysis results
            expected_class: Optional expected universality class
            
        Returns:
            TriangleEdge with consistency result
        """
        if expected_class is None:
            # Find closest universality class
            closest_class, confidence, deviations = \
                self.detector.get_closest_universality_class(vae_results)
        else:
            # Check against expected class
            matches, confidence, deviations = \
                self.detector.compare_to_universality_class(vae_results, expected_class)
            closest_class = expected_class
        
        # Check if all deviations are within threshold
        max_deviation = max(deviations.values()) if deviations else 0.0
        consistent = max_deviation < self.consistency_threshold
        
        details = {
            'universality_class': closest_class,
            'deviations': deviations,
            'max_deviation': max_deviation,
            'message': f"Max deviation: {max_deviation:.2f}sigma from {closest_class}"
        }
        
        return TriangleEdge(
            vertex1=TriangleVertex.EXPONENTS,
            vertex2=TriangleVertex.UNIVERSALITY_CLASS,
            consistent=consistent,
            confidence=confidence,
            details=details
        )
    
    def _check_exponents_scaling_consistency(
        self,
        vae_results: VAEAnalysisResults,
        dimensions: int
    ) -> TriangleEdge:
        """Check consistency between measured exponents and scaling relations.
        
        Args:
            vae_results: VAE analysis results
            dimensions: System dimensions
            
        Returns:
            TriangleEdge with consistency result
        """
        exponents = vae_results.exponents
        violations = []
        
        # Check hyperscaling relation: α + 2β + γ = 2
        if all(exp in exponents for exp in ['alpha', 'beta', 'gamma']):
            alpha = exponents['alpha']
            beta = exponents['beta']
            gamma = exponents['gamma']
            
            measured = alpha + 2 * beta + gamma
            expected = 2.0
            deviation = abs(measured - expected)
            
            # Estimate error
            alpha_err = vae_results.exponent_errors.get('alpha', 0.05)
            beta_err = vae_results.exponent_errors.get('beta', 0.05)
            gamma_err = vae_results.exponent_errors.get('gamma', 0.05)
            combined_err = np.sqrt(alpha_err**2 + (2*beta_err)**2 + gamma_err**2)
            
            deviation_sigma = deviation / combined_err if combined_err > 0 else 0
            
            if deviation_sigma > self.consistency_threshold:
                violations.append({
                    'relation': 'hyperscaling',
                    'measured': measured,
                    'expected': expected,
                    'deviation_sigma': deviation_sigma
                })
        
        # Check gamma-nu relation: γ = ν(2 - η)
        # Simplified: check if gamma/nu is reasonable (typically 1.5-2.0)
        if 'gamma' in exponents and 'nu' in exponents:
            gamma = exponents['gamma']
            nu = exponents['nu']
            
            if nu > 0.1:  # Avoid division by very small numbers
                ratio = gamma / nu
                # Typical range for (2 - η) is 1.5 to 2.0
                if ratio < 1.0 or ratio > 2.5:
                    violations.append({
                        'relation': 'gamma_nu_ratio',
                        'measured': ratio,
                        'expected_range': (1.0, 2.5),
                        'deviation_sigma': abs(ratio - 1.75) / 0.5  # Rough estimate
                    })
        
        # Check alpha-nu relation (dimension-dependent)
        if 'alpha' in exponents and 'nu' in exponents:
            alpha = exponents['alpha']
            nu = exponents['nu']
            
            if dimensions == 2:
                expected_alpha = 2 - 2 * nu
            elif dimensions == 3:
                expected_alpha = 2 - 3 * nu
            else:
                expected_alpha = None
            
            if expected_alpha is not None:
                deviation = abs(alpha - expected_alpha)
                alpha_err = vae_results.exponent_errors.get('alpha', 0.05)
                nu_err = vae_results.exponent_errors.get('nu', 0.05)
                combined_err = np.sqrt(alpha_err**2 + (dimensions * nu_err)**2)
                
                deviation_sigma = deviation / combined_err if combined_err > 0 else 0
                
                if deviation_sigma > self.consistency_threshold:
                    violations.append({
                        'relation': f'alpha_nu_{dimensions}d',
                        'measured': alpha,
                        'expected': expected_alpha,
                        'deviation_sigma': deviation_sigma
                    })
        
        # Determine consistency
        consistent = len(violations) == 0
        confidence = 1.0 - (len(violations) / 3.0)  # Reduce confidence with violations
        
        details = {
            'violations': violations,
            'n_violations': len(violations),
            'message': f"{len(violations)} scaling relation violation(s) detected" if violations else "All scaling relations satisfied"
        }
        
        return TriangleEdge(
            vertex1=TriangleVertex.EXPONENTS,
            vertex2=TriangleVertex.SCALING_RELATIONS,
            consistent=consistent,
            confidence=max(0.0, confidence),
            details=details
        )
    
    def _check_universality_scaling_consistency(
        self,
        vae_results: VAEAnalysisResults,
        expected_class: Optional[str],
        dimensions: int
    ) -> TriangleEdge:
        """Check consistency between universality class and scaling relations.
        
        This checks if the theoretical exponents from the universality class
        satisfy the scaling relations.
        
        Args:
            vae_results: VAE analysis results
            expected_class: Expected universality class
            dimensions: System dimensions
            
        Returns:
            TriangleEdge with consistency result
        """
        if expected_class is None:
            # Find closest class
            closest_class, _, _ = self.detector.get_closest_universality_class(vae_results)
        else:
            closest_class = expected_class
        
        if closest_class not in self.detector.universality_classes:
            return TriangleEdge(
                vertex1=TriangleVertex.UNIVERSALITY_CLASS,
                vertex2=TriangleVertex.SCALING_RELATIONS,
                consistent=False,
                confidence=0.0,
                details={'message': f"Unknown universality class: {closest_class}"}
            )
        
        univ_class = self.detector.universality_classes[closest_class]
        theoretical_exponents = univ_class.exponents
        
        violations = []
        
        # Check hyperscaling with theoretical exponents
        if all(exp in theoretical_exponents for exp in ['alpha', 'beta', 'gamma']):
            alpha = theoretical_exponents['alpha']
            beta = theoretical_exponents['beta']
            gamma = theoretical_exponents['gamma']
            
            measured = alpha + 2 * beta + gamma
            expected = 2.0
            deviation = abs(measured - expected)
            
            if deviation > 0.1:  # Theoretical values should be very close
                violations.append({
                    'relation': 'hyperscaling',
                    'class': closest_class,
                    'measured': measured,
                    'expected': expected,
                    'deviation': deviation
                })
        
        # Universality classes should satisfy scaling relations by construction
        # If they don't, it indicates an issue with the database
        consistent = len(violations) == 0
        confidence = 1.0 if consistent else 0.5
        
        details = {
            'universality_class': closest_class,
            'violations': violations,
            'message': f"Universality class '{closest_class}' " + 
                      ("satisfies" if consistent else "violates") + " scaling relations"
        }
        
        return TriangleEdge(
            vertex1=TriangleVertex.UNIVERSALITY_CLASS,
            vertex2=TriangleVertex.SCALING_RELATIONS,
            consistent=consistent,
            confidence=confidence,
            details=details
        )
    
    def _identify_inconsistent_vertex(self, edges: List[TriangleEdge]) -> TriangleVertex:
        """Identify which vertex is inconsistent when two edges are consistent.
        
        Args:
            edges: List of triangle edges
            
        Returns:
            The inconsistent vertex
        """
        # Count how many edges each vertex participates in that are inconsistent
        vertex_inconsistency_count = {
            TriangleVertex.EXPONENTS: 0,
            TriangleVertex.UNIVERSALITY_CLASS: 0,
            TriangleVertex.SCALING_RELATIONS: 0
        }
        
        for edge in edges:
            if not edge.consistent:
                vertex_inconsistency_count[edge.vertex1] += 1
                vertex_inconsistency_count[edge.vertex2] += 1
        
        # The vertex that appears in the most inconsistent edges is the problem
        inconsistent_vertex = max(
            vertex_inconsistency_count.items(),
            key=lambda x: x[1]
        )[0]
        
        return inconsistent_vertex
    
    def _generate_constrained_recommendations(
        self,
        edges: List[TriangleEdge],
        vae_results: VAEAnalysisResults
    ) -> List[str]:
        """Generate specific recommendations when two vertices are consistent.
        
        Args:
            edges: List of triangle edges
            vae_results: VAE analysis results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        inconsistent_vertex = self._identify_inconsistent_vertex(edges)
        
        if inconsistent_vertex == TriangleVertex.EXPONENTS:
            recommendations.append(
                "Exponents appear problematic. Check measurement quality and fit R² values."
            )
            recommendations.append(
                "Consider increasing simulation length or improving equilibration."
            )
            
            # Check R² values
            avg_r_squared = np.mean(list(vae_results.r_squared_values.values()))
            if avg_r_squared < 0.8:
                recommendations.append(
                    f"Low average R² ({avg_r_squared:.2f}). Improve data quality."
                )
        
        elif inconsistent_vertex == TriangleVertex.UNIVERSALITY_CLASS:
            recommendations.append(
                "Universality class assignment may be incorrect."
            )
            recommendations.append(
                "Try alternative universality classes or consider novel physics."
            )
            
            # Suggest closest alternative
            closest_class, _, _ = self.detector.get_closest_universality_class(vae_results)
            recommendations.append(
                f"Current best match: {closest_class}. Consider nearby classes."
            )
        
        elif inconsistent_vertex == TriangleVertex.SCALING_RELATIONS:
            recommendations.append(
                "Scaling relations violated. This may indicate:"
            )
            recommendations.append(
                "  - Finite-size effects (increase system size)"
            )
            recommendations.append(
                "  - Novel physics beyond standard scaling theory"
            )
            recommendations.append(
                "  - Crossover behavior between universality classes"
            )
        
        return recommendations
    
    def visualize_triangle(
        self,
        validation: TriangleValidation,
        output_path: Optional[str] = None
    ) -> 'plt.Figure':
        """Visualize the validation triangle with consistency status.
        
        Args:
            validation: TriangleValidation result
            output_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Define triangle vertices positions
        vertices = {
            TriangleVertex.UNIVERSALITY_CLASS: (0.5, 0.9),
            TriangleVertex.EXPONENTS: (0.1, 0.2),
            TriangleVertex.SCALING_RELATIONS: (0.9, 0.2)
        }
        
        # Draw edges with colors based on consistency
        for edge in validation.edges:
            v1_pos = vertices[edge.vertex1]
            v2_pos = vertices[edge.vertex2]
            
            color = 'green' if edge.consistent else 'red'
            linewidth = 3 if edge.consistent else 2
            linestyle = '-' if edge.consistent else '--'
            alpha = edge.confidence
            
            ax.plot(
                [v1_pos[0], v2_pos[0]],
                [v1_pos[1], v2_pos[1]],
                color=color, linewidth=linewidth, linestyle=linestyle,
                alpha=alpha, zorder=1
            )
            
            # Add confidence label at edge midpoint
            mid_x = (v1_pos[0] + v2_pos[0]) / 2
            mid_y = (v1_pos[1] + v2_pos[1]) / 2
            ax.text(
                mid_x, mid_y, f'{edge.confidence:.0%}',
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        # Draw vertices
        for vertex, pos in vertices.items():
            # Determine vertex color based on consistency
            vertex_edges = [e for e in validation.edges if vertex in [e.vertex1, e.vertex2]]
            n_consistent = sum(1 for e in vertex_edges if e.consistent)
            
            if n_consistent == 2:
                color = 'lightgreen'
            elif n_consistent == 1:
                color = 'yellow'
            else:
                color = 'lightcoral'
            
            circle = patches.Circle(
                pos, 0.08, facecolor=color, edgecolor='black',
                linewidth=2, zorder=2
            )
            ax.add_patch(circle)
            
            # Add vertex label
            label = vertex.value.replace('_', '\n').title()
            ax.text(
                pos[0], pos[1], label,
                ha='center', va='center', fontsize=11, fontweight='bold',
                zorder=3
            )
        
        # Add title and status
        status_color = {
            ConsistencyStatus.CONSISTENT: 'green',
            ConsistencyStatus.CONSTRAINED: 'orange',
            ConsistencyStatus.INCONSISTENT: 'red',
            ConsistencyStatus.INSUFFICIENT_DATA: 'gray'
        }
        
        ax.set_title(
            f'Validation Triangle\n{validation.message}',
            fontsize=14, fontweight='bold', pad=20
        )
        
        # Add status box
        status_text = f"Status: {validation.overall_status.value.upper()}\n"
        status_text += f"Confidence: {validation.overall_confidence:.0%}\n"
        status_text += f"Consistent Edges: {sum(1 for e in validation.edges if e.consistent)}/3"
        
        ax.text(
            0.5, 0.05, status_text,
            ha='center', va='center', fontsize=12,
            bbox=dict(
                boxstyle='round',
                facecolor=status_color[validation.overall_status],
                alpha=0.3,
                edgecolor='black',
                linewidth=2
            )
        )
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='green', linewidth=3, label='Consistent'),
            plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Inconsistent')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved validation triangle visualization: {output_path}")
        
        return fig
