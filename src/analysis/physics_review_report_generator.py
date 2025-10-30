"""
Physics Review Report Generator

This module provides comprehensive physics review report generation capabilities
for the Prometheus phase discovery system. It creates detailed reports with
physics violation detection, educational content, and visualization capabilities.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import matplotlib.figure as mpl_figure
from matplotlib.figure import Figure
import seaborn as sns

from .enhanced_validation_types import (
    PhysicsReviewReport, PhysicsReviewSummary, TheoreticalConsistencySection,
    OrderParameterValidationSection, CriticalBehaviorSection, 
    StatisticalValidationSection, ExperimentalComparisonSection,
    PhysicsViolation, ViolationSeverity, ViolationSummary, ValidationLevel,
    CriticalExponentValidation, SymmetryValidationResult, UniversalityClassResult,
    FiniteSizeScalingResult, EnsembleAnalysisResult, HypothesisTestResults,
    ExperimentalComparison, MetaAnalysisResult, ConfidenceInterval,
    PhysicsValidationError
)

logger = logging.getLogger(__name__)


class PhysicsReviewReportGenerator:
    """
    Comprehensive physics review report generator.
    
    This class creates detailed physics review reports with violation detection,
    educational content, and comprehensive analysis summaries.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """
        Initialize the report generator.
        
        Args:
            validation_level: Level of validation comprehensiveness
        """
        self.validation_level = validation_level
        self.violations: List[PhysicsViolation] = []
        self.educational_content_cache: Dict[str, str] = {}
        self.literature_references: Dict[str, List[str]] = {}
        
        # Initialize literature reference database
        self._initialize_literature_references()
        
        # Physics violation thresholds
        self.violation_thresholds = {
            'critical_exponent_deviation': {
                ViolationSeverity.LOW: 0.05,
                ViolationSeverity.MEDIUM: 0.10,
                ViolationSeverity.HIGH: 0.20,
                ViolationSeverity.CRITICAL: 0.30
            },
            'symmetry_consistency': {
                ViolationSeverity.LOW: 0.90,
                ViolationSeverity.MEDIUM: 0.80,
                ViolationSeverity.HIGH: 0.70,
                ViolationSeverity.CRITICAL: 0.60
            },
            'scaling_collapse_quality': {
                ViolationSeverity.LOW: 0.90,
                ViolationSeverity.MEDIUM: 0.80,
                ViolationSeverity.HIGH: 0.70,
                ViolationSeverity.CRITICAL: 0.60
            }
        }
    
    def generate_comprehensive_report(
        self,
        validation_results: Dict[str, Any],
        include_educational_content: bool = True
    ) -> PhysicsReviewReport:
        """
        Generate comprehensive physics review report.
        
        Args:
            validation_results: Dictionary containing all validation results
            include_educational_content: Whether to include educational explanations
            
        Returns:
            Complete physics review report
        """
        try:
            logger.info("Generating comprehensive physics review report")
            
            # Clear previous violations
            self.violations.clear()
            
            # Extract and validate input data
            self._validate_input_data(validation_results)
            
            # Generate report sections
            summary = self._generate_summary_section(validation_results)
            theoretical_consistency = self._generate_theoretical_consistency_section(
                validation_results
            )
            order_parameter_validation = self._generate_order_parameter_section(
                validation_results
            )
            critical_behavior = self._generate_critical_behavior_section(
                validation_results
            )
            statistical_validation = self._generate_statistical_validation_section(
                validation_results
            )
            experimental_comparison = self._generate_experimental_comparison_section(
                validation_results
            )
            
            # Generate violation summary
            violation_summary = self.generate_violation_summary(self.violations)
            
            # Generate educational content if requested
            educational_content = {}
            if include_educational_content:
                educational_content = self._generate_educational_content(validation_results)
            
            # Generate visualizations
            visualizations = self._generate_report_visualizations(validation_results)
            
            # Create comprehensive report
            report = PhysicsReviewReport(
                summary=summary,
                theoretical_consistency=theoretical_consistency,
                order_parameter_validation=order_parameter_validation,
                critical_behavior_analysis=critical_behavior,
                statistical_validation=statistical_validation,
                experimental_comparison=experimental_comparison,
                violations=self.violations.copy(),
                violation_summary=violation_summary,
                educational_content=educational_content,
                visualizations=visualizations,
                overall_assessment=self._generate_overall_assessment(violation_summary),
                generation_timestamp=datetime.now().isoformat(),
                validation_parameters={
                    'validation_level': self.validation_level.value,
                    'include_educational_content': include_educational_content
                }
            )
            
            logger.info(f"Generated report with {len(self.violations)} violations")
            return report
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            raise PhysicsValidationError(f"Failed to generate report: {e}")
    
    def generate_violation_summary(
        self,
        violations: List[PhysicsViolation]
    ) -> ViolationSummary:
        """
        Generate summary of physics violations.
        
        Args:
            violations: List of physics violations
            
        Returns:
            Summary of violations with categorization and recommendations
        """
        try:
            # Count violations by severity
            violations_by_severity = {severity: 0 for severity in ViolationSeverity}
            for violation in violations:
                violations_by_severity[violation.severity] += 1
            
            # Count violations by type
            violations_by_type = {}
            for violation in violations:
                violation_type = violation.violation_type
                violations_by_type[violation_type] = violations_by_type.get(violation_type, 0) + 1
            
            # Get critical violations
            critical_violations = [
                v for v in violations 
                if v.severity == ViolationSeverity.CRITICAL
            ]
            
            # Generate recommended actions
            recommended_actions = self._generate_recommended_actions(violations)
            
            return ViolationSummary(
                total_violations=len(violations),
                violations_by_severity=violations_by_severity,
                violations_by_type=violations_by_type,
                critical_violations=critical_violations,
                recommended_actions=recommended_actions
            )
            
        except Exception as e:
            logger.error(f"Error generating violation summary: {e}")
            raise PhysicsValidationError(f"Failed to generate violation summary: {e}")
    
    def _validate_input_data(self, validation_results: Dict[str, Any]) -> None:
        """Validate input data structure."""
        required_keys = ['critical_exponent_validation', 'symmetry_validation']
        for key in required_keys:
            if key not in validation_results:
                logger.warning(f"Missing required validation result: {key}")
    
    def _generate_summary_section(
        self, 
        validation_results: Dict[str, Any]
    ) -> PhysicsReviewSummary:
        """Generate summary section of the report."""
        # Calculate overall physics consistency score
        consistency_score = self._calculate_physics_consistency_score(validation_results)
        
        # Generate key findings
        key_findings = self._extract_key_findings(validation_results)
        
        # Get major violations (high and critical severity)
        major_violations = [
            v for v in self.violations 
            if v.severity in [ViolationSeverity.HIGH, ViolationSeverity.CRITICAL]
        ]
        
        # Generate recommendations
        recommendations = self._generate_summary_recommendations(validation_results)
        
        # Determine overall assessment
        if consistency_score >= 0.9:
            overall_assessment = "Excellent physics consistency"
        elif consistency_score >= 0.8:
            overall_assessment = "Good physics consistency with minor issues"
        elif consistency_score >= 0.7:
            overall_assessment = "Acceptable physics consistency with notable issues"
        elif consistency_score >= 0.6:
            overall_assessment = "Poor physics consistency requiring investigation"
        else:
            overall_assessment = "Critical physics consistency issues detected"
        
        return PhysicsReviewSummary(
            overall_assessment=overall_assessment,
            physics_consistency_score=consistency_score,
            validation_level=self.validation_level,
            key_findings=key_findings,
            major_violations=major_violations,
            recommendations=recommendations
        )
    
    def _generate_theoretical_consistency_section(
        self, 
        validation_results: Dict[str, Any]
    ) -> TheoreticalConsistencySection:
        """Generate theoretical consistency section."""
        # Extract critical exponent validation
        critical_exponent_validation = validation_results.get(
            'critical_exponent_validation',
            self._create_default_critical_exponent_validation()
        )
        
        # Check for critical exponent violations
        self._check_critical_exponent_violations(critical_exponent_validation)
        
        # Extract universality class result
        universality_class_result = validation_results.get(
            'universality_class_result',
            self._create_default_universality_class_result()
        )
        
        # Extract finite-size scaling result if available
        finite_size_scaling_result = validation_results.get('finite_size_scaling_result')
        if finite_size_scaling_result:
            self._check_finite_size_scaling_violations(finite_size_scaling_result)
        
        # Extract model comparisons
        model_comparisons = validation_results.get('model_comparisons', {})
        
        # Get theoretical violations
        theoretical_violations = [
            v for v in self.violations 
            if v.violation_type in ['critical_exponent', 'universality_class', 'finite_size_scaling']
        ]
        
        return TheoreticalConsistencySection(
            critical_exponent_validation=critical_exponent_validation,
            universality_class_result=universality_class_result,
            finite_size_scaling_result=finite_size_scaling_result,
            model_comparisons=model_comparisons,
            theoretical_violations=theoretical_violations
        )
    
    def _generate_order_parameter_section(
        self, 
        validation_results: Dict[str, Any]
    ) -> OrderParameterValidationSection:
        """Generate order parameter validation section."""
        # Extract symmetry validation
        symmetry_validation = validation_results.get(
            'symmetry_validation',
            self._create_default_symmetry_validation()
        )
        
        # Check for symmetry violations
        self._check_symmetry_violations(symmetry_validation)
        
        # Extract correlation and hierarchy analysis
        correlation_analysis = validation_results.get('correlation_analysis', {})
        hierarchy_analysis = validation_results.get('hierarchy_analysis', {})
        coupling_analysis = validation_results.get('coupling_analysis', {})
        
        # Get order parameter violations
        order_parameter_violations = [
            v for v in self.violations 
            if v.violation_type in ['symmetry', 'order_parameter', 'correlation']
        ]
        
        return OrderParameterValidationSection(
            symmetry_validation=symmetry_validation,
            correlation_analysis=correlation_analysis,
            hierarchy_analysis=hierarchy_analysis,
            coupling_analysis=coupling_analysis,
            order_parameter_violations=order_parameter_violations
        )
    
    def _generate_critical_behavior_section(
        self, 
        validation_results: Dict[str, Any]
    ) -> CriticalBehaviorSection:
        """Generate critical behavior analysis section."""
        phase_transition_analysis = validation_results.get('phase_transition_analysis', {})
        critical_temperature_validation = validation_results.get('critical_temperature_validation', {})
        scaling_behavior = validation_results.get('scaling_behavior', {})
        transition_sharpness = validation_results.get('transition_sharpness')
        
        # Get critical behavior violations
        critical_behavior_violations = [
            v for v in self.violations 
            if v.violation_type in ['phase_transition', 'critical_temperature', 'scaling']
        ]
        
        return CriticalBehaviorSection(
            phase_transition_analysis=phase_transition_analysis,
            critical_temperature_validation=critical_temperature_validation,
            scaling_behavior=scaling_behavior,
            transition_sharpness=transition_sharpness,
            critical_behavior_violations=critical_behavior_violations
        )
    
    def _generate_statistical_validation_section(
        self, 
        validation_results: Dict[str, Any]
    ) -> StatisticalValidationSection:
        """Generate statistical validation section."""
        ensemble_analysis = validation_results.get('ensemble_analysis')
        hypothesis_tests = validation_results.get('hypothesis_tests', [])
        confidence_intervals = validation_results.get('confidence_intervals', {})
        uncertainty_quantification = validation_results.get('uncertainty_quantification', {})
        
        # Get statistical violations
        statistical_violations = [
            v for v in self.violations 
            if v.violation_type in ['statistical', 'confidence_interval', 'hypothesis_test']
        ]
        
        return StatisticalValidationSection(
            ensemble_analysis=ensemble_analysis,
            hypothesis_tests=hypothesis_tests,
            confidence_intervals=confidence_intervals,
            uncertainty_quantification=uncertainty_quantification,
            statistical_violations=statistical_violations
        )
    
    def _generate_experimental_comparison_section(
        self, 
        validation_results: Dict[str, Any]
    ) -> ExperimentalComparisonSection:
        """Generate experimental comparison section."""
        experimental_comparisons = validation_results.get('experimental_comparisons', [])
        meta_analysis = validation_results.get('meta_analysis')
        agreement_summary = validation_results.get('agreement_summary', {})
        
        # Get experimental violations
        experimental_violations = [
            v for v in self.violations 
            if v.violation_type in ['experimental_comparison', 'meta_analysis']
        ]
        
        return ExperimentalComparisonSection(
            experimental_comparisons=experimental_comparisons,
            meta_analysis=meta_analysis,
            agreement_summary=agreement_summary,
            experimental_violations=experimental_violations
        )
    
    def _check_critical_exponent_violations(
        self, 
        critical_exponent_validation: CriticalExponentValidation
    ) -> None:
        """Check for critical exponent violations and add them to the violations list."""
        # Check beta exponent deviation
        if critical_exponent_validation.beta_deviation is not None:
            severity = self._determine_violation_severity(
                'critical_exponent_deviation',
                critical_exponent_validation.beta_deviation
            )
            
            if severity != ViolationSeverity.LOW or self.validation_level == ValidationLevel.COMPREHENSIVE:
                violation = PhysicsViolation(
                    violation_type='critical_exponent',
                    severity=severity,
                    description=f"Beta critical exponent deviates by {critical_exponent_validation.beta_deviation:.3f} from theoretical value",
                    suggested_investigation="Check finite-size effects, system equilibration, and order parameter definition",
                    physics_explanation="Critical exponents are universal quantities that should match theoretical predictions within statistical uncertainty",
                    quantitative_measure=critical_exponent_validation.beta_deviation,
                    threshold_value=self.violation_thresholds['critical_exponent_deviation'][severity]
                )
                self.violations.append(violation)
        
        # Check universality class match
        if not critical_exponent_validation.universality_class_match:
            violation = PhysicsViolation(
                violation_type='universality_class',
                severity=ViolationSeverity.HIGH,
                description="Critical exponents do not match expected universality class",
                suggested_investigation="Verify system symmetries, dimensionality, and interaction range",
                physics_explanation="Systems in the same universality class should have identical critical exponents"
            )
            self.violations.append(violation)
    
    def _check_symmetry_violations(
        self, 
        symmetry_validation: SymmetryValidationResult
    ) -> None:
        """Check for symmetry violations and add them to the violations list."""
        if symmetry_validation.symmetry_consistency_score < 0.8:
            severity = self._determine_violation_severity(
                'symmetry_consistency',
                symmetry_validation.symmetry_consistency_score
            )
            
            violation = PhysicsViolation(
                violation_type='symmetry',
                severity=severity,
                description=f"Low symmetry consistency score: {symmetry_validation.symmetry_consistency_score:.3f}",
                suggested_investigation="Check order parameter definition and symmetry breaking mechanism",
                physics_explanation="Order parameters should respect the symmetries of the underlying Hamiltonian",
                quantitative_measure=symmetry_validation.symmetry_consistency_score,
                threshold_value=0.8
            )
            self.violations.append(violation)
    
    def _check_finite_size_scaling_violations(
        self, 
        finite_size_scaling_result: FiniteSizeScalingResult
    ) -> None:
        """Check for finite-size scaling violations."""
        if finite_size_scaling_result.scaling_collapse_quality < 0.8:
            severity = self._determine_violation_severity(
                'scaling_collapse_quality',
                finite_size_scaling_result.scaling_collapse_quality
            )
            
            violation = PhysicsViolation(
                violation_type='finite_size_scaling',
                severity=severity,
                description=f"Poor scaling collapse quality: {finite_size_scaling_result.scaling_collapse_quality:.3f}",
                suggested_investigation="Check system sizes, correlation length estimation, and finite-size corrections",
                physics_explanation="Finite-size scaling should produce good data collapse when properly scaled",
                quantitative_measure=finite_size_scaling_result.scaling_collapse_quality,
                threshold_value=0.8
            )
            self.violations.append(violation)
    
    def _determine_violation_severity(
        self, 
        violation_type: str, 
        value: float
    ) -> ViolationSeverity:
        """Determine violation severity based on thresholds."""
        if violation_type not in self.violation_thresholds:
            return ViolationSeverity.MEDIUM
        
        thresholds = self.violation_thresholds[violation_type]
        
        if violation_type == 'critical_exponent_deviation':
            # Higher values are worse for deviations
            if value >= thresholds[ViolationSeverity.CRITICAL]:
                return ViolationSeverity.CRITICAL
            elif value >= thresholds[ViolationSeverity.HIGH]:
                return ViolationSeverity.HIGH
            elif value >= thresholds[ViolationSeverity.MEDIUM]:
                return ViolationSeverity.MEDIUM
            else:
                return ViolationSeverity.LOW
        else:
            # Lower values are worse for quality scores
            if value <= thresholds[ViolationSeverity.CRITICAL]:
                return ViolationSeverity.CRITICAL
            elif value <= thresholds[ViolationSeverity.HIGH]:
                return ViolationSeverity.HIGH
            elif value <= thresholds[ViolationSeverity.MEDIUM]:
                return ViolationSeverity.MEDIUM
            else:
                return ViolationSeverity.LOW
    
    def _calculate_physics_consistency_score(
        self, 
        validation_results: Dict[str, Any]
    ) -> float:
        """Calculate overall physics consistency score."""
        scores = []
        
        # Critical exponent consistency
        if 'critical_exponent_validation' in validation_results:
            crit_exp = validation_results['critical_exponent_validation']
            if hasattr(crit_exp, 'beta_deviation') and crit_exp.beta_deviation is not None:
                # Convert deviation to score (lower deviation = higher score)
                score = max(0.0, 1.0 - crit_exp.beta_deviation / 0.3)
                scores.append(score)
        
        # Symmetry consistency
        if 'symmetry_validation' in validation_results:
            sym_val = validation_results['symmetry_validation']
            if hasattr(sym_val, 'symmetry_consistency_score'):
                scores.append(sym_val.symmetry_consistency_score)
        
        # Finite-size scaling quality
        if 'finite_size_scaling_result' in validation_results:
            fss = validation_results['finite_size_scaling_result']
            if hasattr(fss, 'scaling_collapse_quality'):
                scores.append(fss.scaling_collapse_quality)
        
        # Return average score or default
        return np.mean(scores) if scores else 0.8
    
    def _extract_key_findings(self, validation_results: Dict[str, Any]) -> List[str]:
        """Extract key findings from validation results."""
        findings = []
        
        # Critical exponent findings
        if 'critical_exponent_validation' in validation_results:
            crit_exp = validation_results['critical_exponent_validation']
            if hasattr(crit_exp, 'universality_class_match') and crit_exp.universality_class_match:
                findings.append("Critical exponents match expected universality class")
            elif hasattr(crit_exp, 'identified_universality_class'):
                findings.append(f"Identified universality class: {crit_exp.identified_universality_class.value}")
        
        # Symmetry findings
        if 'symmetry_validation' in validation_results:
            sym_val = validation_results['symmetry_validation']
            if hasattr(sym_val, 'broken_symmetries') and sym_val.broken_symmetries:
                findings.append(f"Broken symmetries detected: {', '.join(sym_val.broken_symmetries)}")
        
        return findings
    
    def _generate_summary_recommendations(
        self, 
        validation_results: Dict[str, Any]
    ) -> List[str]:
        """Generate summary recommendations based on validation results."""
        recommendations = []
        
        # Check for critical violations
        critical_violations = [
            v for v in self.violations 
            if v.severity == ViolationSeverity.CRITICAL
        ]
        
        if critical_violations:
            recommendations.append("Address critical physics violations before proceeding")
        
        # Check for high severity violations
        high_violations = [
            v for v in self.violations 
            if v.severity == ViolationSeverity.HIGH
        ]
        
        if high_violations:
            recommendations.append("Investigate high-severity physics inconsistencies")
        
        # General recommendations based on validation level
        if self.validation_level == ValidationLevel.BASIC:
            recommendations.append("Consider upgrading to comprehensive validation for publication")
        
        return recommendations
    
    def _generate_recommended_actions(
        self, 
        violations: List[PhysicsViolation]
    ) -> List[str]:
        """Generate recommended actions based on violations."""
        actions = []
        
        # Group violations by type
        violation_types = {}
        for violation in violations:
            vtype = violation.violation_type
            if vtype not in violation_types:
                violation_types[vtype] = []
            violation_types[vtype].append(violation)
        
        # Generate type-specific recommendations
        for vtype, vlist in violation_types.items():
            if vtype == 'critical_exponent':
                actions.append("Verify critical exponent calculations and finite-size scaling")
            elif vtype == 'symmetry':
                actions.append("Check order parameter symmetry properties and breaking mechanism")
            elif vtype == 'finite_size_scaling':
                actions.append("Improve finite-size scaling analysis with larger system sizes")
        
        return actions
    
    def _generate_overall_assessment(self, violation_summary: ViolationSummary) -> str:
        """Generate overall assessment based on violation summary."""
        critical_count = violation_summary.violations_by_severity.get(ViolationSeverity.CRITICAL, 0)
        high_count = violation_summary.violations_by_severity.get(ViolationSeverity.HIGH, 0)
        total_count = violation_summary.total_violations
        
        if critical_count > 0:
            return f"Critical physics issues detected ({critical_count} critical violations). Immediate attention required."
        elif high_count > 0:
            return f"Significant physics concerns identified ({high_count} high-severity violations). Investigation recommended."
        elif total_count > 5:
            return f"Multiple minor physics issues detected ({total_count} total violations). Review suggested."
        elif total_count > 0:
            return f"Minor physics inconsistencies found ({total_count} violations). Generally acceptable."
        else:
            return "Excellent physics consistency. No violations detected."
    
    def _generate_educational_content(
        self, 
        validation_results: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate educational content for physics concepts."""
        educational_content = {}
        
        # Identify physics concepts present in the validation results
        concepts = self._identify_physics_concepts(validation_results)
        
        # Generate explanations for each concept
        for concept in concepts:
            if concept not in self.educational_content_cache:
                explanation = self._generate_concept_explanation(concept)
                self.educational_content_cache[concept] = explanation
            
            educational_content[concept] = self.educational_content_cache[concept]
        
        return educational_content
    
    def _create_default_critical_exponent_validation(self) -> CriticalExponentValidation:
        """Create default critical exponent validation for missing data."""
        return CriticalExponentValidation(
            beta_exponent=0.0,
            beta_theoretical=0.0,
            beta_confidence_interval=(0.0, 0.0),
            beta_deviation=0.0,
            universality_class_match=False
        )
    
    def _create_default_universality_class_result(self) -> UniversalityClassResult:
        """Create default universality class result for missing data."""
        from .enhanced_validation_types import UniversalityClass
        return UniversalityClassResult(
            identified_class=UniversalityClass.UNKNOWN,
            confidence_score=0.0,
            critical_exponents_match={}
        )
    
    def _create_default_symmetry_validation(self) -> SymmetryValidationResult:
        """Create default symmetry validation for missing data."""
        return SymmetryValidationResult(
            broken_symmetries=[],
            symmetry_order={},
            order_parameter_symmetry="unknown",
            symmetry_consistency_score=0.0
        )
    
    def generate_educational_explanations(
        self,
        physics_concepts: List[str]
    ) -> Dict[str, str]:
        """
        Generate educational explanations for physics concepts.
        
        Args:
            physics_concepts: List of physics concepts to explain
            
        Returns:
            Dictionary mapping concepts to their explanations
        """
        explanations = {}
        
        for concept in physics_concepts:
            if concept not in self.educational_content_cache:
                explanation = self._generate_concept_explanation(concept)
                self.educational_content_cache[concept] = explanation
            
            explanations[concept] = self.educational_content_cache[concept]
        
        return explanations
    
    def _identify_physics_concepts(self, validation_results: Dict[str, Any]) -> List[str]:
        """Identify physics concepts present in validation results."""
        concepts = []
        
        # Check for critical exponents
        if 'critical_exponent_validation' in validation_results:
            concepts.extend(['critical_exponents', 'universality_classes', 'phase_transitions'])
        
        # Check for symmetry analysis
        if 'symmetry_validation' in validation_results:
            concepts.extend(['symmetry_breaking', 'order_parameters'])
        
        # Check for finite-size scaling
        if 'finite_size_scaling_result' in validation_results:
            concepts.extend(['finite_size_scaling', 'correlation_length'])
        
        # Check for experimental comparisons
        if 'experimental_comparisons' in validation_results:
            concepts.append('experimental_validation')
        
        # Check for statistical analysis
        if 'ensemble_analysis' in validation_results or 'hypothesis_tests' in validation_results:
            concepts.append('statistical_physics')
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(concepts))
    
    def _generate_concept_explanation(self, concept: str) -> str:
        """Generate educational explanation for a specific physics concept."""
        explanations = {
            'critical_exponents': self._explain_critical_exponents(),
            'universality_classes': self._explain_universality_classes(),
            'phase_transitions': self._explain_phase_transitions(),
            'symmetry_breaking': self._explain_symmetry_breaking(),
            'order_parameters': self._explain_order_parameters(),
            'finite_size_scaling': self._explain_finite_size_scaling(),
            'correlation_length': self._explain_correlation_length(),
            'experimental_validation': self._explain_experimental_validation(),
            'statistical_physics': self._explain_statistical_physics()
        }
        
        return explanations.get(concept, f"Educational content for '{concept}' not available.")
    
    def _explain_critical_exponents(self) -> str:
        """Explain critical exponents."""
        return """
Critical Exponents:

Critical exponents are universal numbers that characterize the behavior of physical quantities 
near phase transitions. They describe how thermodynamic quantities diverge or vanish as the 
system approaches the critical point.

Key critical exponents:
• β (beta): Order parameter exponent - describes how the order parameter vanishes approaching Tc
• γ (gamma): Susceptibility exponent - characterizes divergence of magnetic susceptibility
• ν (nu): Correlation length exponent - describes divergence of correlation length
• α (alpha): Specific heat exponent - characterizes specific heat behavior

These exponents are universal within each universality class, meaning all systems with the same 
symmetries and dimensionality share identical critical exponents regardless of microscopic details.

Literature: Goldenfeld, N. "Lectures on Phase Transitions and the Renormalization Group" (1992)
"""
    
    def _explain_universality_classes(self) -> str:
        """Explain universality classes."""
        return """
Universality Classes:

Universality is one of the most profound concepts in statistical physics. Systems belong to the 
same universality class if they have identical critical exponents, despite having different 
microscopic interactions.

Classification depends on:
• Spatial dimensionality (d)
• Order parameter dimensionality (n)
• Symmetries of the Hamiltonian
• Range of interactions

Common universality classes:
• Ising (n=1): Binary order parameter, Z₂ symmetry
• XY (n=2): Planar spins, O(2) symmetry, includes Kosterlitz-Thouless transitions
• Heisenberg (n=3): 3D spins, O(3) symmetry

This universality allows predictions about new systems based on known results from the same class.

Literature: Cardy, J. "Scaling and Renormalization in Statistical Physics" (1996)
"""
    
    def _explain_phase_transitions(self) -> str:
        """Explain phase transitions."""
        return """
Phase Transitions:

Phase transitions are abrupt changes in the macroscopic properties of a system due to 
collective behavior of microscopic constituents. They represent fundamental changes in 
the organization of matter.

Types of transitions:
• First-order: Discontinuous order parameter, latent heat present
• Second-order (continuous): Continuous order parameter, critical fluctuations
• Kosterlitz-Thouless: Topological transitions with exponential behavior

Critical phenomena near second-order transitions:
• Diverging correlation length
• Power-law behavior of thermodynamic quantities
• Scale invariance and self-similarity
• Universal critical exponents

Understanding phase transitions is crucial for materials science, condensed matter physics, 
and many other fields.

Literature: Stanley, H.E. "Introduction to Phase Transitions and Critical Phenomena" (1971)
"""
    
    def _explain_symmetry_breaking(self) -> str:
        """Explain symmetry breaking."""
        return """
Symmetry Breaking:

Symmetry breaking occurs when a system's ground state has lower symmetry than its Hamiltonian. 
This is a fundamental mechanism for phase transitions and the emergence of order.

Types of symmetry breaking:
• Spontaneous: System chooses one of many equivalent ground states
• Explicit: External field breaks the symmetry
• Continuous: O(n) symmetries, leads to Goldstone modes
• Discrete: Z_n symmetries, no Goldstone modes

Order parameters:
• Quantify the degree of symmetry breaking
• Zero in the symmetric phase, non-zero in the broken phase
• Transform according to the broken symmetry

Examples:
• Ferromagnetism: Rotational symmetry → preferred spin direction
• Crystallization: Translational symmetry → periodic lattice

Literature: Anderson, P.W. "Basic Notions of Condensed Matter Physics" (1984)
"""
    
    def _explain_order_parameters(self) -> str:
        """Explain order parameters."""
        return """
Order Parameters:

Order parameters are quantities that distinguish different phases of matter. They provide 
a mathematical description of the degree of order in a system.

Properties of order parameters:
• Zero in the disordered phase
• Non-zero in the ordered phase
• Transform according to the system's symmetries
• Determine the universality class

Examples:
• Magnetization in ferromagnets
• Density difference in liquid-gas transitions
• Complex amplitude in superconductors
• Nematic director in liquid crystals

Primary vs. Secondary order parameters:
• Primary: Directly coupled to the ordering field
• Secondary: Induced by coupling to primary order parameter

The choice of order parameter is crucial for understanding the physics and determining 
the correct theoretical description.

Literature: Chaikin, P.M. & Lubensky, T.C. "Principles of Condensed Matter Physics" (1995)
"""
    
    def _explain_finite_size_scaling(self) -> str:
        """Explain finite-size scaling."""
        return """
Finite-Size Scaling:

Finite-size scaling describes how critical behavior is modified in finite systems. 
It's essential for extracting critical exponents from numerical simulations.

Key concepts:
• Correlation length ξ becomes comparable to system size L near criticality
• Scaling variable: t·L^(1/ν) where t = |T-Tc|/Tc
• Data collapse: Properly scaled data from different system sizes collapse onto universal curves

Scaling forms:
• Thermodynamic quantities: X(t,L) = L^(x/ν) f(tL^(1/ν))
• Where x is the critical exponent for quantity X

Applications:
• Determine critical temperature with high precision
• Extract critical exponents
• Verify universality class predictions
• Account for finite-size corrections

Proper finite-size scaling analysis is crucial for reliable critical exponent determination 
in computational studies.

Literature: Barber, M.N. "Finite-size scaling" in Phase Transitions and Critical Phenomena (1983)
"""
    
    def _explain_correlation_length(self) -> str:
        """Explain correlation length."""
        return """
Correlation Length:

The correlation length ξ characterizes the range over which fluctuations are correlated 
in a system. It's a fundamental length scale that diverges at critical points.

Mathematical definition:
• ξ = lim(r→∞) r / ln[G(0)/G(r)]
• Where G(r) is the correlation function

Critical behavior:
• ξ ~ |t|^(-ν) as T → Tc
• ν is the correlation length critical exponent
• Divergence leads to scale invariance at criticality

Physical significance:
• Measures the size of correlated regions
• Determines finite-size effects in simulations
• Controls the approach to thermodynamic limit
• Related to response functions via fluctuation-dissipation theorem

In finite systems:
• ξ cannot exceed system size L
• Finite-size scaling emerges when ξ ~ L
• Critical behavior is modified by finite-size effects

Literature: Ma, S.K. "Modern Theory of Critical Phenomena" (1976)
"""
    
    def _explain_experimental_validation(self) -> str:
        """Explain experimental validation."""
        return """
Experimental Validation:

Comparing computational results with experimental data is crucial for validating 
theoretical models and simulation methods.

Key considerations:
• Statistical significance of differences
• Systematic vs. random errors
• Sample preparation and measurement conditions
• Finite-size and finite-time effects in experiments

Common discrepancies:
• Impurities and defects in real materials
• Non-equilibrium effects
• Finite measurement time scales
• Instrumental resolution limits

Meta-analysis approaches:
• Combine results from multiple experimental studies
• Account for different measurement techniques
• Assess publication bias and systematic trends
• Provide robust estimates with uncertainty quantification

Best practices:
• Use multiple experimental datasets when available
• Consider measurement uncertainties properly
• Account for systematic differences between techniques
• Report confidence intervals and statistical significance

Literature: Privman, V. "Finite Size Scaling and Numerical Simulation of Statistical Systems" (1990)
"""
    
    def _explain_statistical_physics(self) -> str:
        """Explain statistical physics validation."""
        return """
Statistical Physics Validation:

Statistical validation ensures that computational results are statistically reliable 
and physically meaningful.

Key statistical concepts:
• Bootstrap confidence intervals: Non-parametric uncertainty estimation
• Hypothesis testing: Formal comparison with theoretical predictions
• Ensemble analysis: Combining results from multiple independent runs
• Error propagation: Proper treatment of uncertainties

Important considerations:
• Autocorrelation in time series data
• Finite-size effects and extrapolation
• Systematic vs. statistical errors
• Multiple testing corrections

Validation procedures:
• Test against known analytical results
• Compare with established experimental data
• Verify scaling laws and universal behavior
• Check consistency across different system sizes

Statistical significance:
• p-values for hypothesis tests
• Confidence intervals for parameter estimates
• Effect sizes for practical significance
• Power analysis for detecting true effects

Literature: Landau, D.P. & Binder, K. "A Guide to Monte Carlo Simulations in Statistical Physics" (2014)
"""
    
    def _initialize_literature_references(self) -> None:
        """Initialize literature reference database."""
        self.literature_references = {
            'critical_exponents': [
                'Goldenfeld, N. "Lectures on Phase Transitions and the Renormalization Group" (1992)',
                'Fisher, M.E. "The theory of equilibrium critical phenomena" Rep. Prog. Phys. 30, 615 (1967)',
                'Kadanoff, L.P. "Scaling laws for Ising models near Tc" Physics 2, 263 (1966)'
            ],
            'universality_classes': [
                'Cardy, J. "Scaling and Renormalization in Statistical Physics" (1996)',
                'Wilson, K.G. "The renormalization group: Critical phenomena and the Kondo problem" Rev. Mod. Phys. 47, 773 (1975)',
                'Pelissetto, A. & Vicari, E. "Critical phenomena and renormalization-group theory" Phys. Rep. 368, 549 (2002)'
            ],
            'phase_transitions': [
                'Stanley, H.E. "Introduction to Phase Transitions and Critical Phenomena" (1971)',
                'Landau, L.D. & Lifshitz, E.M. "Statistical Physics" (1980)',
                'Yeomans, J.M. "Statistical Mechanics of Phase Transitions" (1992)'
            ],
            'symmetry_breaking': [
                'Anderson, P.W. "Basic Notions of Condensed Matter Physics" (1984)',
                'Goldstone, J. "Field theories with superconductor solutions" Nuovo Cimento 19, 154 (1961)',
                'Nambu, Y. & Jona-Lasinio, G. "Dynamical model of elementary particles" Phys. Rev. 122, 345 (1961)'
            ],
            'order_parameters': [
                'Chaikin, P.M. & Lubensky, T.C. "Principles of Condensed Matter Physics" (1995)',
                'de Gennes, P.G. & Prost, J. "The Physics of Liquid Crystals" (1993)',
                'Tinkham, M. "Introduction to Superconductivity" (1996)'
            ],
            'finite_size_scaling': [
                'Barber, M.N. "Finite-size scaling" in Phase Transitions and Critical Phenomena Vol. 8 (1983)',
                'Privman, V. "Finite Size Scaling and Numerical Simulation of Statistical Systems" (1990)',
                'Binder, K. "Finite size scaling analysis of Ising model block distribution functions" Z. Phys. B 43, 119 (1981)'
            ],
            'correlation_length': [
                'Ma, S.K. "Modern Theory of Critical Phenomena" (1976)',
                'Fisher, M.E. "Correlation functions and the critical region of simple fluids" J. Math. Phys. 5, 944 (1964)',
                'Ornstein, L.S. & Zernike, F. "Accidental deviations of density and opalescence" Proc. Akad. Sci. (Amsterdam) 17, 793 (1914)'
            ],
            'experimental_validation': [
                'Privman, V. "Finite Size Scaling and Numerical Simulation of Statistical Systems" (1990)',
                'Guggenheim, E.A. "The principle of corresponding states" J. Chem. Phys. 13, 253 (1945)',
                'Levelt Sengers, J.M.H. "Critical exponents at the turn of the century" Physica A 82, 319 (1976)'
            ],
            'statistical_physics': [
                'Landau, D.P. & Binder, K. "A Guide to Monte Carlo Simulations in Statistical Physics" (2014)',
                'Newman, M.E.J. & Barkema, G.T. "Monte Carlo Methods in Statistical Physics" (1999)',
                'Efron, B. & Tibshirani, R.J. "An Introduction to the Bootstrap" (1993)'
            ]
        }
    
    def get_literature_references(self, concept: str) -> List[str]:
        """
        Get literature references for a specific physics concept.
        
        Args:
            concept: Physics concept name
            
        Returns:
            List of literature references
        """
        return self.literature_references.get(concept, [])
    
    def add_literature_reference(self, concept: str, reference: str) -> None:
        """
        Add a literature reference for a physics concept.
        
        Args:
            concept: Physics concept name
            reference: Literature reference string
        """
        if concept not in self.literature_references:
            self.literature_references[concept] = []
        
        if reference not in self.literature_references[concept]:
            self.literature_references[concept].append(reference)
    
    def generate_bibliography(self, concepts: List[str]) -> Dict[str, List[str]]:
        """
        Generate bibliography for given physics concepts.
        
        Args:
            concepts: List of physics concepts
            
        Returns:
            Dictionary mapping concepts to their literature references
        """
        bibliography = {}
        
        for concept in concepts:
            if concept in self.literature_references:
                bibliography[concept] = self.literature_references[concept].copy()
        
        return bibliography
    
    def _generate_report_visualizations(
        self, 
        validation_results: Dict[str, Any]
    ) -> Dict[str, Figure]:
        """Generate visualizations for the physics review report."""
        visualizations = {}
        
        try:
            # Set style for consistent appearance
            plt.style.use('seaborn-v0_8')
            
            # Generate critical exponent visualization
            if 'critical_exponent_validation' in validation_results:
                fig = self._create_critical_exponent_plot(
                    validation_results['critical_exponent_validation']
                )
                if fig is not None:
                    visualizations['critical_exponents'] = fig
            
            # Generate symmetry analysis visualization
            if 'symmetry_validation' in validation_results:
                fig = self._create_symmetry_analysis_plot(
                    validation_results['symmetry_validation']
                )
                if fig is not None:
                    visualizations['symmetry_analysis'] = fig
            
            # Generate finite-size scaling visualization
            if 'finite_size_scaling_result' in validation_results:
                fig = self._create_finite_size_scaling_plot(
                    validation_results['finite_size_scaling_result']
                )
                if fig is not None:
                    visualizations['finite_size_scaling'] = fig
            
            # Generate violation summary visualization
            if self.violations:
                fig = self._create_violation_summary_plot(self.violations)
                if fig is not None:
                    visualizations['violation_summary'] = fig
            
            # Generate physics consistency dashboard
            fig = self._create_physics_consistency_dashboard(validation_results)
            if fig is not None:
                visualizations['consistency_dashboard'] = fig
            
        except Exception as e:
            logger.warning(f"Error generating visualizations: {e}")
        
        return visualizations
    
    def _create_critical_exponent_plot(
        self, 
        critical_exponent_validation: CriticalExponentValidation
    ) -> Optional[Figure]:
        """Create critical exponent comparison plot."""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Prepare data for plotting
            exponents = []
            computed_values = []
            theoretical_values = []
            confidence_intervals = []
            
            # Beta exponent
            if critical_exponent_validation.beta_exponent is not None:
                exponents.append('β')
                computed_values.append(critical_exponent_validation.beta_exponent)
                theoretical_values.append(critical_exponent_validation.beta_theoretical)
                confidence_intervals.append(critical_exponent_validation.beta_confidence_interval)
            
            # Gamma exponent
            if critical_exponent_validation.gamma_exponent is not None:
                exponents.append('γ')
                computed_values.append(critical_exponent_validation.gamma_exponent)
                theoretical_values.append(critical_exponent_validation.gamma_theoretical)
                confidence_intervals.append(critical_exponent_validation.gamma_confidence_interval)
            
            # Nu exponent
            if critical_exponent_validation.nu_exponent is not None:
                exponents.append('ν')
                computed_values.append(critical_exponent_validation.nu_exponent)
                theoretical_values.append(critical_exponent_validation.nu_theoretical)
                confidence_intervals.append(critical_exponent_validation.nu_confidence_interval)
            
            if not exponents:
                plt.close(fig)
                return None
            
            x_pos = np.arange(len(exponents))
            
            # Plot computed values with error bars
            yerr_lower = [cv - ci[0] for cv, ci in zip(computed_values, confidence_intervals)]
            yerr_upper = [ci[1] - cv for cv, ci in zip(computed_values, confidence_intervals)]
            
            ax.errorbar(x_pos, computed_values, 
                       yerr=[yerr_lower, yerr_upper],
                       fmt='o', capsize=5, capthick=2, 
                       label='Computed', color='blue', markersize=8)
            
            # Plot theoretical values
            ax.scatter(x_pos, theoretical_values, 
                      marker='s', s=100, color='red', 
                      label='Theoretical', zorder=5)
            
            # Customize plot
            ax.set_xlabel('Critical Exponent', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title('Critical Exponent Validation', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(exponents)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add universality class information
            if hasattr(critical_exponent_validation, 'identified_universality_class'):
                uc = critical_exponent_validation.identified_universality_class
                ax.text(0.02, 0.98, f'Universality Class: {uc.value}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating critical exponent plot: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None
    
    def _create_symmetry_analysis_plot(
        self, 
        symmetry_validation: SymmetryValidationResult
    ) -> Optional[Figure]:
        """Create symmetry analysis visualization."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot 1: Symmetry consistency score
            consistency_score = symmetry_validation.symmetry_consistency_score
            
            # Create gauge-like plot for consistency score
            theta = np.linspace(0, np.pi, 100)
            r = np.ones_like(theta)
            
            ax1.plot(theta, r, 'k-', linewidth=2)
            ax1.fill_between(theta, 0, r, alpha=0.3, color='lightgray')
            
            # Color code the score
            if consistency_score >= 0.8:
                color = 'green'
            elif consistency_score >= 0.6:
                color = 'orange'
            else:
                color = 'red'
            
            # Add score indicator
            score_theta = np.pi * (1 - consistency_score)
            ax1.plot([score_theta, score_theta], [0, 1], color=color, linewidth=4)
            ax1.scatter([score_theta], [1], color=color, s=100, zorder=5)
            
            ax1.set_xlim(0, np.pi)
            ax1.set_ylim(0, 1.2)
            ax1.set_title('Symmetry Consistency Score', fontweight='bold')
            ax1.text(np.pi/2, 0.5, f'{consistency_score:.3f}', 
                    ha='center', va='center', fontsize=16, fontweight='bold')
            ax1.set_xticks([0, np.pi/2, np.pi])
            ax1.set_xticklabels(['1.0', '0.5', '0.0'])
            ax1.set_yticks([])
            
            # Plot 2: Broken symmetries
            if symmetry_validation.broken_symmetries:
                broken_symmetries = symmetry_validation.broken_symmetries
                y_pos = np.arange(len(broken_symmetries))
                
                ax2.barh(y_pos, [1] * len(broken_symmetries), color='red', alpha=0.7)
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(broken_symmetries)
                ax2.set_xlabel('Broken')
                ax2.set_title('Symmetry Breaking Analysis', fontweight='bold')
                ax2.set_xlim(0, 1.2)
                
                # Add symmetry order information if available
                if symmetry_validation.symmetry_order:
                    for i, symmetry in enumerate(broken_symmetries):
                        if symmetry in symmetry_validation.symmetry_order:
                            order = symmetry_validation.symmetry_order[symmetry]
                            ax2.text(1.1, i, f'n={order}', va='center', fontsize=10)
            else:
                ax2.text(0.5, 0.5, 'No Broken\nSymmetries', 
                        ha='center', va='center', transform=ax2.transAxes,
                        fontsize=14, color='green', fontweight='bold')
                ax2.set_title('Symmetry Breaking Analysis', fontweight='bold')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating symmetry analysis plot: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None
    
    def _create_finite_size_scaling_plot(
        self, 
        finite_size_scaling_result: FiniteSizeScalingResult
    ) -> Optional[Figure]:
        """Create finite-size scaling visualization."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot 1: System sizes and scaling quality
            system_sizes = finite_size_scaling_result.system_sizes
            scaling_quality = finite_size_scaling_result.scaling_collapse_quality
            
            ax1.scatter(system_sizes, [scaling_quality] * len(system_sizes), 
                       s=100, alpha=0.7, color='blue')
            ax1.axhline(y=scaling_quality, color='blue', linestyle='--', alpha=0.7)
            ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Threshold')
            
            ax1.set_xlabel('System Size L')
            ax1.set_ylabel('Scaling Collapse Quality')
            ax1.set_title('Finite-Size Scaling Quality', fontweight='bold')
            ax1.set_ylim(0, 1.1)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Add quality assessment text
            if scaling_quality >= 0.9:
                quality_text = "Excellent"
                quality_color = "green"
            elif scaling_quality >= 0.8:
                quality_text = "Good"
                quality_color = "orange"
            else:
                quality_text = "Poor"
                quality_color = "red"
            
            ax1.text(0.02, 0.98, f'Quality: {quality_text}\n({scaling_quality:.3f})', 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=quality_color, alpha=0.3))
            
            # Plot 2: Correlation length scaling
            if hasattr(finite_size_scaling_result, 'correlation_length_exponent'):
                nu_exp = finite_size_scaling_result.correlation_length_exponent
                nu_ci = finite_size_scaling_result.correlation_length_confidence_interval
                
                # Create mock data for visualization (in real implementation, use actual data)
                t_values = np.logspace(-3, -0.5, 50)
                xi_values = t_values**(-nu_exp)
                
                ax2.loglog(t_values, xi_values, 'b-', linewidth=2, 
                          label=f'ν = {nu_exp:.3f}')
                
                # Add confidence interval
                xi_upper = t_values**(-nu_ci[1])
                xi_lower = t_values**(-nu_ci[0])
                ax2.fill_between(t_values, xi_lower, xi_upper, alpha=0.3, color='blue')
                
                ax2.set_xlabel('|t| = |T - Tc|/Tc')
                ax2.set_ylabel('Correlation Length ξ')
                ax2.set_title('Correlation Length Scaling', fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Correlation Length\nData Not Available', 
                        ha='center', va='center', transform=ax2.transAxes,
                        fontsize=12, style='italic')
                ax2.set_title('Correlation Length Scaling', fontweight='bold')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating finite-size scaling plot: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None
    
    def _create_violation_summary_plot(
        self, 
        violations: List[PhysicsViolation]
    ) -> Optional[Figure]:
        """Create violation summary visualization."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot 1: Violations by severity
            severity_counts = {}
            for violation in violations:
                severity = violation.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            if severity_counts:
                severities = list(severity_counts.keys())
                counts = list(severity_counts.values())
                
                # Color code by severity
                colors = []
                for severity in severities:
                    if severity == 'critical':
                        colors.append('red')
                    elif severity == 'high':
                        colors.append('orange')
                    elif severity == 'medium':
                        colors.append('yellow')
                    else:
                        colors.append('lightblue')
                
                bars = ax1.bar(severities, counts, color=colors, alpha=0.7)
                ax1.set_xlabel('Severity Level')
                ax1.set_ylabel('Number of Violations')
                ax1.set_title('Violations by Severity', fontweight='bold')
                
                # Add count labels on bars
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                            f'{count}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 2: Violations by type
            type_counts = {}
            for violation in violations:
                vtype = violation.violation_type
                type_counts[vtype] = type_counts.get(vtype, 0) + 1
            
            if type_counts:
                # Create pie chart for violation types
                types = list(type_counts.keys())
                counts = list(type_counts.values())
                
                wedges, texts, autotexts = ax2.pie(counts, labels=types, autopct='%1.1f%%',
                                                  startangle=90, textprops={'fontsize': 10})
                ax2.set_title('Violations by Type', fontweight='bold')
                
                # Make percentage text bold
                for autotext in autotexts:
                    autotext.set_fontweight('bold')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating violation summary plot: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None
    
    def _create_physics_consistency_dashboard(
        self, 
        validation_results: Dict[str, Any]
    ) -> Optional[Figure]:
        """Create overall physics consistency dashboard."""
        try:
            fig = plt.figure(figsize=(14, 8))
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            
            # Overall consistency score (large gauge)
            ax_main = fig.add_subplot(gs[0, :2])
            consistency_score = self._calculate_physics_consistency_score(validation_results)
            
            # Create gauge plot
            theta = np.linspace(0, np.pi, 100)
            r = np.ones_like(theta)
            
            ax_main.plot(theta, r, 'k-', linewidth=3)
            ax_main.fill_between(theta, 0, r, alpha=0.2, color='lightgray')
            
            # Color zones
            ax_main.fill_between(theta[theta <= np.pi*0.2], 0, 
                               r[theta <= np.pi*0.2], alpha=0.3, color='red')
            ax_main.fill_between(theta[(theta > np.pi*0.2) & (theta <= np.pi*0.4)], 0, 
                               r[(theta > np.pi*0.2) & (theta <= np.pi*0.4)], alpha=0.3, color='orange')
            ax_main.fill_between(theta[(theta > np.pi*0.4) & (theta <= np.pi*0.8)], 0, 
                               r[(theta > np.pi*0.4) & (theta <= np.pi*0.8)], alpha=0.3, color='yellow')
            ax_main.fill_between(theta[theta > np.pi*0.8], 0, 
                               r[theta > np.pi*0.8], alpha=0.3, color='green')
            
            # Score indicator
            score_theta = np.pi * (1 - consistency_score)
            ax_main.plot([score_theta, score_theta], [0, 1], color='black', linewidth=6)
            ax_main.scatter([score_theta], [1], color='black', s=200, zorder=5)
            
            ax_main.set_xlim(0, np.pi)
            ax_main.set_ylim(0, 1.2)
            ax_main.set_title('Overall Physics Consistency Score', fontsize=16, fontweight='bold')
            ax_main.text(np.pi/2, 0.3, f'{consistency_score:.3f}', 
                        ha='center', va='center', fontsize=24, fontweight='bold')
            ax_main.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
            ax_main.set_xticklabels(['1.0', '0.75', '0.5', '0.25', '0.0'])
            ax_main.set_yticks([])
            
            # Individual component scores
            components = ['Critical\nExponents', 'Symmetry\nAnalysis', 'Finite-Size\nScaling']
            component_scores = []
            
            # Extract component scores
            if 'critical_exponent_validation' in validation_results:
                crit_exp = validation_results['critical_exponent_validation']
                if hasattr(crit_exp, 'beta_deviation') and crit_exp.beta_deviation is not None:
                    score = max(0.0, 1.0 - crit_exp.beta_deviation / 0.3)
                    component_scores.append(score)
                else:
                    component_scores.append(0.8)
            else:
                component_scores.append(0.5)
            
            if 'symmetry_validation' in validation_results:
                sym_val = validation_results['symmetry_validation']
                if hasattr(sym_val, 'symmetry_consistency_score'):
                    component_scores.append(sym_val.symmetry_consistency_score)
                else:
                    component_scores.append(0.8)
            else:
                component_scores.append(0.5)
            
            if 'finite_size_scaling_result' in validation_results:
                fss = validation_results['finite_size_scaling_result']
                if hasattr(fss, 'scaling_collapse_quality'):
                    component_scores.append(fss.scaling_collapse_quality)
                else:
                    component_scores.append(0.8)
            else:
                component_scores.append(0.5)
            
            # Component scores bar chart
            ax_components = fig.add_subplot(gs[0, 2])
            colors = ['green' if score >= 0.8 else 'orange' if score >= 0.6 else 'red' 
                     for score in component_scores]
            
            bars = ax_components.bar(range(len(components)), component_scores, 
                                   color=colors, alpha=0.7)
            ax_components.set_ylim(0, 1)
            ax_components.set_ylabel('Score')
            ax_components.set_title('Component Scores', fontweight='bold')
            ax_components.set_xticks(range(len(components)))
            ax_components.set_xticklabels(components, rotation=45, ha='right')
            
            # Add score labels
            for bar, score in zip(bars, component_scores):
                height = bar.get_height()
                ax_components.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                 f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # Summary statistics
            ax_summary = fig.add_subplot(gs[1, :])
            ax_summary.axis('off')
            
            # Create summary text
            total_violations = len(self.violations)
            critical_violations = len([v for v in self.violations if v.severity == ViolationSeverity.CRITICAL])
            high_violations = len([v for v in self.violations if v.severity == ViolationSeverity.HIGH])
            
            summary_text = f"""
Physics Validation Summary:
• Overall Consistency Score: {consistency_score:.3f}
• Total Violations: {total_violations}
• Critical Violations: {critical_violations}
• High-Severity Violations: {high_violations}
• Validation Level: {self.validation_level.value.title()}
            """
            
            ax_summary.text(0.1, 0.8, summary_text, transform=ax_summary.transAxes,
                          fontsize=12, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            
            # Add recommendations
            if critical_violations > 0:
                recommendation = "⚠️ Critical issues require immediate attention"
                color = 'red'
            elif high_violations > 0:
                recommendation = "⚠️ High-severity issues should be investigated"
                color = 'orange'
            elif total_violations > 5:
                recommendation = "ℹ️ Multiple minor issues detected"
                color = 'blue'
            else:
                recommendation = "✅ Physics validation passed successfully"
                color = 'green'
            
            ax_summary.text(0.6, 0.8, f"Recommendation:\n{recommendation}", 
                          transform=ax_summary.transAxes, fontsize=12, 
                          verticalalignment='top', color=color, fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor=color, alpha=0.1))
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating physics consistency dashboard: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None