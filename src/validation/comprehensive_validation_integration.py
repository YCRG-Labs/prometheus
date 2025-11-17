"""
Comprehensive Validation Integration

This module integrates all validation components for task 11:
- Statistical validation framework (task 11.1)
- Final validation and quality assurance system (task 11.2)
- Provides unified interface for comprehensive validation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from .statistical_validation_framework import (
        StatisticalValidationFramework, SystemValidationMetrics,
        ErrorBarResult, ConfidenceIntervalResult, FiniteSizeScalingResult, QualityMetrics
    )
    from .final_validation_system import (
        FinalValidationSystem, FinalValidationReport,
        EquilibrationValidationResult, DataQualityResult, PhysicsConsistencyResult
    )
    from ..utils.logging_utils import get_logger
except ImportError:
    # Fallback for testing
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    try:
        from validation.statistical_validation_framework import (
            StatisticalValidationFramework, SystemValidationMetrics
        )
        from validation.final_validation_system import (
            FinalValidationSystem, FinalValidationReport
        )
        from utils.logging_utils import get_logger
    except ImportError:
        # Mock for testing
        class StatisticalValidationFramework:
            pass
        
        class FinalValidationSystem:
            pass
        
        class SystemValidationMetrics:
            pass
        
        class FinalValidationReport:
            pass
        
        def get_logger(name):
            return logging.getLogger(name)


@dataclass
class ComprehensiveValidationResult:
    """Container for comprehensive validation results."""
    validation_timestamp: str
    
    # Statistical validation results
    statistical_validation: Dict[str, SystemValidationMetrics]
    statistical_summary: Dict[str, Any]
    
    # Final validation results
    final_validation: FinalValidationReport
    
    # Integrated assessment
    overall_validation_passed: bool
    overall_validation_score: float
    
    # Performance metrics
    total_validation_time: float
    systems_validated: int
    
    # Consolidated recommendations
    priority_issues: List[str]
    actionable_recommendations: List[str]
    
    # Quality assurance summary
    quality_assurance_passed: bool
    ready_for_publication: bool


class ComprehensiveValidationIntegration:
    """
    Comprehensive validation integration system that combines all validation components.
    
    This class provides a unified interface for:
    1. Statistical validation framework (error bars, confidence intervals, finite-size scaling)
    2. Final validation system (equilibration, data quality, physics consistency)
    3. Integrated quality assurance and recommendations
    """
    
    def __init__(self,
                 confidence_level: float = 0.95,
                 bootstrap_samples: int = 10000,
                 significance_threshold: float = 0.05,
                 quality_threshold: float = 0.7,
                 equilibration_threshold: float = 1e-4,
                 parallel_validation: bool = True,
                 random_seed: Optional[int] = None):
        """
        Initialize comprehensive validation integration.
        
        Args:
            confidence_level: Confidence level for statistical tests
            bootstrap_samples: Number of bootstrap samples
            significance_threshold: P-value threshold for significance
            quality_threshold: Minimum quality score threshold
            equilibration_threshold: Threshold for equilibration convergence
            parallel_validation: Whether to run validations in parallel
            random_seed: Random seed for reproducibility
        """
        self.confidence_level = confidence_level
        self.bootstrap_samples = bootstrap_samples
        self.significance_threshold = significance_threshold
        self.quality_threshold = quality_threshold
        self.equilibration_threshold = equilibration_threshold
        self.parallel_validation = parallel_validation
        self.random_seed = random_seed
        self.logger = get_logger(__name__)
        
        # Initialize validation frameworks
        self.statistical_framework = StatisticalValidationFramework(
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            significance_threshold=significance_threshold,
            random_seed=random_seed
        )
        
        self.final_validation_system = FinalValidationSystem(
            equilibration_threshold=equilibration_threshold,
            quality_threshold=quality_threshold,
            random_seed=random_seed
        )
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def run_comprehensive_validation(self,
                                   systems_data: Dict[str, Dict[str, Any]],
                                   output_dir: str = "results/comprehensive_validation") -> ComprehensiveValidationResult:
        """
        Run comprehensive validation across all systems and components.
        
        Args:
            systems_data: Dictionary containing data for all systems to validate
            output_dir: Output directory for validation results
            
        Returns:
            ComprehensiveValidationResult with complete validation assessment
        """
        self.logger.info("Starting comprehensive validation integration")
        start_time = time.time()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Phase 1: Statistical Validation
        self.logger.info("Phase 1: Running statistical validation framework")
        statistical_results = self._run_statistical_validation(systems_data)
        
        # Phase 2: Final Validation System
        self.logger.info("Phase 2: Running final validation system")
        final_validation_result = self.final_validation_system.run_final_validation(
            systems_data, str(output_path / "final_validation")
        )
        
        # Phase 3: Integration and Assessment
        self.logger.info("Phase 3: Integrating validation results")
        integrated_result = self._integrate_validation_results(
            statistical_results, final_validation_result, time.time() - start_time
        )
        
        # Phase 4: Generate Comprehensive Report
        self.logger.info("Phase 4: Generating comprehensive validation report")
        self._generate_comprehensive_report(integrated_result, output_path)
        
        self.logger.info(f"Comprehensive validation completed in {integrated_result.total_validation_time:.1f}s")
        self.logger.info(f"Overall validation: {'PASSED' if integrated_result.overall_validation_passed else 'FAILED'}")
        
        return integrated_result
    
    def _run_statistical_validation(self,
                                  systems_data: Dict[str, Dict[str, Any]]) -> Dict[str, SystemValidationMetrics]:
        """Run statistical validation for all systems."""
        
        statistical_results = {}
        
        if self.parallel_validation:
            statistical_results = self._run_statistical_validation_parallel(systems_data)
        else:
            statistical_results = self._run_statistical_validation_sequential(systems_data)
        
        return statistical_results
    
    def _run_statistical_validation_parallel(self,
                                           systems_data: Dict[str, Dict[str, Any]]) -> Dict[str, SystemValidationMetrics]:
        """Run statistical validation in parallel."""
        
        statistical_results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit validation tasks
            future_to_system = {
                executor.submit(self._validate_single_system_statistical, system_name, system_data): system_name
                for system_name, system_data in systems_data.items()
            }
            
            # Collect results
            for future in as_completed(future_to_system):
                system_name = future_to_system[future]
                try:
                    result = future.result()
                    statistical_results[system_name] = result
                    self.logger.info(f"Statistical validation completed for {system_name}")
                except Exception as e:
                    self.logger.error(f"Statistical validation failed for {system_name}: {e}")
                    # Create placeholder failed result
                    statistical_results[system_name] = self._create_failed_statistical_result(system_name)
        
        return statistical_results
    
    def _run_statistical_validation_sequential(self,
                                             systems_data: Dict[str, Dict[str, Any]]) -> Dict[str, SystemValidationMetrics]:
        """Run statistical validation sequentially."""
        
        statistical_results = {}
        
        for system_name, system_data in systems_data.items():
            try:
                result = self._validate_single_system_statistical(system_name, system_data)
                statistical_results[system_name] = result
                self.logger.info(f"Statistical validation completed for {system_name}")
            except Exception as e:
                self.logger.error(f"Statistical validation failed for {system_name}: {e}")
                statistical_results[system_name] = self._create_failed_statistical_result(system_name)
        
        return statistical_results
    
    def _validate_single_system_statistical(self,
                                          system_name: str,
                                          system_data: Dict[str, Any]) -> SystemValidationMetrics:
        """Run statistical validation for a single system."""
        
        # Prepare theoretical values
        theoretical_values = None
        if 'config' in system_data:
            config = system_data['config']
            theoretical_values = {
                'tc': config.get('theoretical_tc'),
                **config.get('theoretical_exponents', {})
            }
        
        # Run comprehensive statistical validation
        result = self.statistical_framework.validate_system_comprehensive(
            system_data, theoretical_values
        )
        
        return result
    
    def _create_failed_statistical_result(self, system_name: str) -> SystemValidationMetrics:
        """Create a failed statistical validation result."""
        
        # Create minimal failed result
        from .statistical_validation_framework import QualityMetrics
        
        failed_quality = QualityMetrics(
            statistical_significance=0.0,
            confidence_level=self.confidence_level,
            sample_adequacy=0.0,
            physics_consistency=0.0,
            theoretical_agreement=0.0,
            universality_class_match=0.0,
            data_completeness=0.0,
            equilibration_quality=0.0,
            sampling_efficiency=0.0,
            model_convergence=0.0,
            reconstruction_quality=0.0,
            latent_space_quality=0.0,
            overall_quality_score=0.0
        )
        
        return SystemValidationMetrics(
            system_type=system_name,
            system_sizes=[],
            quality_metrics=failed_quality,
            validation_passed=False,
            validation_score=0.0
        )
    
    def _integrate_validation_results(self,
                                    statistical_results: Dict[str, SystemValidationMetrics],
                                    final_validation: FinalValidationReport,
                                    total_time: float) -> ComprehensiveValidationResult:
        """Integrate all validation results into comprehensive assessment."""
        
        # Statistical validation summary
        statistical_summary = self._compute_statistical_summary(statistical_results)
        
        # Overall validation assessment
        statistical_passed = statistical_summary['systems_passed'] >= len(statistical_results) * 0.8
        final_passed = final_validation.validation_passed
        
        overall_validation_passed = statistical_passed and final_passed
        
        # Overall validation score (weighted combination)
        statistical_score = statistical_summary['average_validation_score']
        final_score = final_validation.overall_validation_score
        
        overall_validation_score = (
            statistical_score * 0.4 +
            final_score * 0.6
        )
        
        # Priority issues and recommendations
        priority_issues = self._identify_priority_issues(statistical_results, final_validation)
        actionable_recommendations = self._generate_actionable_recommendations(
            statistical_results, final_validation
        )
        
        # Quality assurance assessment
        quality_assurance_passed = (
            overall_validation_score >= self.quality_threshold and
            len(priority_issues) <= 2 and
            final_validation.all_systems_equilibrated and
            final_validation.all_data_quality_passed
        )
        
        # Publication readiness
        ready_for_publication = (
            quality_assurance_passed and
            overall_validation_score >= 0.8 and
            statistical_summary['average_validation_score'] >= 0.75
        )
        
        return ComprehensiveValidationResult(
            validation_timestamp=str(np.datetime64('now')),
            statistical_validation=statistical_results,
            statistical_summary=statistical_summary,
            final_validation=final_validation,
            overall_validation_passed=overall_validation_passed,
            overall_validation_score=overall_validation_score,
            total_validation_time=total_time,
            systems_validated=len(statistical_results),
            priority_issues=priority_issues,
            actionable_recommendations=actionable_recommendations,
            quality_assurance_passed=quality_assurance_passed,
            ready_for_publication=ready_for_publication
        )
    
    def _compute_statistical_summary(self,
                                   statistical_results: Dict[str, SystemValidationMetrics]) -> Dict[str, Any]:
        """Compute summary statistics for statistical validation results."""
        
        if not statistical_results:
            return {
                'systems_passed': 0,
                'total_systems': 0,
                'success_rate': 0.0,
                'average_validation_score': 0.0,
                'average_quality_score': 0.0
            }
        
        systems_passed = sum(1 for result in statistical_results.values() if result.validation_passed)
        total_systems = len(statistical_results)
        success_rate = systems_passed / total_systems
        
        validation_scores = [result.validation_score for result in statistical_results.values()]
        average_validation_score = np.mean(validation_scores)
        
        quality_scores = [
            result.quality_metrics.overall_quality_score 
            for result in statistical_results.values() 
            if result.quality_metrics
        ]
        average_quality_score = np.mean(quality_scores) if quality_scores else 0.0
        
        return {
            'systems_passed': systems_passed,
            'total_systems': total_systems,
            'success_rate': success_rate,
            'average_validation_score': average_validation_score,
            'average_quality_score': average_quality_score
        }
    
    def _identify_priority_issues(self,
                                statistical_results: Dict[str, SystemValidationMetrics],
                                final_validation: FinalValidationReport) -> List[str]:
        """Identify priority issues that need immediate attention."""
        
        priority_issues = []
        
        # Critical statistical issues
        for system_name, result in statistical_results.items():
            if not result.validation_passed:
                priority_issues.append(f"Statistical validation failed for {system_name}")
            
            if result.quality_metrics and result.quality_metrics.statistical_significance < 0.3:
                priority_issues.append(f"Poor statistical significance for {system_name}")
            
            if result.quality_metrics and result.quality_metrics.physics_consistency < 0.5:
                priority_issues.append(f"Physics consistency issues in {system_name}")
        
        # Critical final validation issues
        if not final_validation.all_systems_equilibrated:
            priority_issues.append("Some systems failed equilibration validation")
        
        if not final_validation.all_data_quality_passed:
            priority_issues.append("Data quality issues detected")
        
        if not final_validation.all_physics_consistent:
            priority_issues.append("Physics consistency validation failed")
        
        # Add most critical issues from final validation
        critical_final_issues = [
            issue for issue in final_validation.critical_issues 
            if any(keyword in issue.lower() for keyword in ['failed', 'error', 'inconsistent'])
        ]
        priority_issues.extend(critical_final_issues[:3])  # Top 3 critical issues
        
        return priority_issues[:10]  # Limit to top 10 priority issues
    
    def _generate_actionable_recommendations(self,
                                          statistical_results: Dict[str, SystemValidationMetrics],
                                          final_validation: FinalValidationReport) -> List[str]:
        """Generate actionable recommendations for improvement."""
        
        recommendations = []
        
        # Statistical validation recommendations
        low_quality_systems = [
            name for name, result in statistical_results.items()
            if result.quality_metrics and result.quality_metrics.overall_quality_score < self.quality_threshold
        ]
        
        if low_quality_systems:
            recommendations.append(
                f"Improve data quality and model training for systems: {', '.join(low_quality_systems[:3])}"
            )
        
        # Check for common issues across systems
        poor_significance_systems = [
            name for name, result in statistical_results.items()
            if result.quality_metrics and result.quality_metrics.statistical_significance < 0.5
        ]
        
        if len(poor_significance_systems) > 1:
            recommendations.append(
                "Increase sample sizes and improve statistical power across multiple systems"
            )
        
        poor_physics_systems = [
            name for name, result in statistical_results.items()
            if result.quality_metrics and result.quality_metrics.physics_consistency < 0.6
        ]
        
        if len(poor_physics_systems) > 1:
            recommendations.append(
                "Review physics model implementations and theoretical parameter values"
            )
        
        # Final validation recommendations
        if final_validation.average_equilibration_quality < self.quality_threshold:
            recommendations.append(
                "Increase equilibration time and improve convergence monitoring"
            )
        
        if final_validation.average_data_quality < self.quality_threshold:
            recommendations.append(
                "Improve Monte Carlo simulation parameters and data generation quality"
            )
        
        if final_validation.average_physics_consistency < self.quality_threshold:
            recommendations.append(
                "Validate physics implementations against known theoretical results"
            )
        
        # Add specific recommendations from final validation
        specific_recommendations = [
            rec for rec in final_validation.improvement_recommendations
            if any(keyword in rec.lower() for keyword in ['increase', 'improve', 'check', 'validate'])
        ]
        recommendations.extend(specific_recommendations[:5])  # Top 5 specific recommendations
        
        # General recommendations based on overall performance
        overall_score = (
            np.mean([r.validation_score for r in statistical_results.values()]) * 0.4 +
            final_validation.overall_validation_score * 0.6
        )
        
        if overall_score < 0.6:
            recommendations.append(
                "Consider fundamental review of simulation and analysis methodology"
            )
        elif overall_score < 0.8:
            recommendations.append(
                "Focus on improving weakest performing validation components"
            )
        else:
            recommendations.append(
                "Validation performance is good - consider minor optimizations for publication"
            )
        
        return recommendations[:15]  # Limit to top 15 recommendations
    
    def _generate_comprehensive_report(self,
                                     result: ComprehensiveValidationResult,
                                     output_path: Path):
        """Generate comprehensive validation report with all components."""
        
        # Save comprehensive JSON report
        json_path = output_path / "comprehensive_validation_report.json"
        with open(json_path, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        # Generate comprehensive text report
        self._generate_comprehensive_text_report(result, output_path)
        
        # Generate comprehensive visualizations
        self._generate_comprehensive_visualizations(result, output_path)
        
        self.logger.info(f"Comprehensive validation report generated in {output_path}")
    
    def _generate_comprehensive_text_report(self,
                                          result: ComprehensiveValidationResult,
                                          output_path: Path):
        """Generate comprehensive text report."""
        
        report_path = output_path / "comprehensive_validation_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("COMPREHENSIVE VALIDATION AND ERROR ANALYSIS REPORT\n")
            f.write("=" * 100 + "\n\n")
            
            f.write(f"Validation Timestamp: {result.validation_timestamp}\n")
            f.write(f"Total Validation Time: {result.total_validation_time:.1f} seconds\n")
            f.write(f"Systems Validated: {result.systems_validated}\n\n")
            
            # Overall Assessment
            f.write("OVERALL VALIDATION ASSESSMENT\n")
            f.write("-" * 50 + "\n")
            f.write(f"Overall Validation: {'✅ PASSED' if result.overall_validation_passed else '❌ FAILED'}\n")
            f.write(f"Overall Score: {result.overall_validation_score:.3f}\n")
            f.write(f"Quality Assurance: {'✅ PASSED' if result.quality_assurance_passed else '❌ FAILED'}\n")
            f.write(f"Publication Ready: {'✅ YES' if result.ready_for_publication else '❌ NO'}\n\n")
            
            # Statistical Validation Summary
            f.write("STATISTICAL VALIDATION SUMMARY\n")
            f.write("-" * 50 + "\n")
            stat_summary = result.statistical_summary
            f.write(f"Systems Passed: {stat_summary['systems_passed']}/{stat_summary['total_systems']}\n")
            f.write(f"Success Rate: {stat_summary['success_rate']:.1%}\n")
            f.write(f"Average Validation Score: {stat_summary['average_validation_score']:.3f}\n")
            f.write(f"Average Quality Score: {stat_summary['average_quality_score']:.3f}\n\n")
            
            # Final Validation Summary
            f.write("FINAL VALIDATION SUMMARY\n")
            f.write("-" * 50 + "\n")
            final_val = result.final_validation
            f.write(f"All Systems Equilibrated: {final_val.all_systems_equilibrated}\n")
            f.write(f"All Data Quality Passed: {final_val.all_data_quality_passed}\n")
            f.write(f"All Physics Consistent: {final_val.all_physics_consistent}\n")
            f.write(f"Average Equilibration Quality: {final_val.average_equilibration_quality:.3f}\n")
            f.write(f"Average Data Quality: {final_val.average_data_quality:.3f}\n")
            f.write(f"Average Physics Consistency: {final_val.average_physics_consistency:.3f}\n\n")
            
            # Priority Issues
            if result.priority_issues:
                f.write("PRIORITY ISSUES\n")
                f.write("-" * 50 + "\n")
                for i, issue in enumerate(result.priority_issues, 1):
                    f.write(f"{i:2d}. {issue}\n")
                f.write("\n")
            
            # Actionable Recommendations
            if result.actionable_recommendations:
                f.write("ACTIONABLE RECOMMENDATIONS\n")
                f.write("-" * 50 + "\n")
                for i, rec in enumerate(result.actionable_recommendations, 1):
                    f.write(f"{i:2d}. {rec}\n")
                f.write("\n")
            
            # System-by-System Details
            f.write("SYSTEM-BY-SYSTEM VALIDATION DETAILS\n")
            f.write("-" * 50 + "\n")
            
            for system_name, stat_result in result.statistical_validation.items():
                f.write(f"\nSystem: {system_name}\n")
                f.write(f"  Statistical Validation: {'PASSED' if stat_result.validation_passed else 'FAILED'}\n")
                f.write(f"  Validation Score: {stat_result.validation_score:.3f}\n")
                
                if stat_result.quality_metrics:
                    qm = stat_result.quality_metrics
                    f.write(f"  Quality Metrics:\n")
                    f.write(f"    Statistical Significance: {qm.statistical_significance:.3f}\n")
                    f.write(f"    Physics Consistency: {qm.physics_consistency:.3f}\n")
                    f.write(f"    Theoretical Agreement: {qm.theoretical_agreement:.3f}\n")
                    f.write(f"    Overall Quality: {qm.overall_quality_score:.3f}\n")
                
                # Add final validation details if available
                if system_name in final_val.equilibration_results:
                    eq_result = final_val.equilibration_results[system_name]
                    f.write(f"  Equilibration: {'PASSED' if eq_result.is_equilibrated else 'FAILED'}\n")
                    f.write(f"  Equilibration Quality: {eq_result.convergence_quality_score:.3f}\n")
                
                if system_name in final_val.data_quality_results:
                    dq_result = final_val.data_quality_results[system_name]
                    f.write(f"  Data Quality Score: {dq_result.overall_quality_score:.3f}\n")
                
                if system_name in final_val.physics_consistency_results:
                    pc_result = final_val.physics_consistency_results[system_name]
                    f.write(f"  Physics Consistency Score: {pc_result.physics_consistency_score:.3f}\n")
            
            f.write("\n" + "=" * 100 + "\n")
            f.write("END OF COMPREHENSIVE VALIDATION REPORT\n")
            f.write("=" * 100 + "\n")
    
    def _generate_comprehensive_visualizations(self,
                                             result: ComprehensiveValidationResult,
                                             output_path: Path):
        """Generate comprehensive validation visualizations."""
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Plot 1: Overall validation summary
        ax = axes[0, 0]
        
        categories = ['Statistical\nValidation', 'Final\nValidation', 'Overall\nValidation']
        scores = [
            result.statistical_summary['average_validation_score'],
            result.final_validation.overall_validation_score,
            result.overall_validation_score
        ]
        
        bars = ax.bar(categories, scores,
                     color=['green' if score >= 0.7 else 'orange' if score >= 0.5 else 'red' 
                           for score in scores])
        
        ax.axhline(0.7, color='black', linestyle='--', label='Quality Threshold')
        ax.set_ylabel('Validation Score')
        ax.set_title('Comprehensive Validation Summary')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add score labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: System-wise validation comparison
        ax = axes[0, 1]
        
        system_names = list(result.statistical_validation.keys())
        if system_names:
            stat_scores = [result.statistical_validation[name].validation_score for name in system_names]
            
            # Get final validation scores
            final_scores = []
            for name in system_names:
                eq_score = 0.0
                dq_score = 0.0
                pc_score = 0.0
                
                if name in result.final_validation.equilibration_results:
                    eq_score = result.final_validation.equilibration_results[name].convergence_quality_score
                
                if name in result.final_validation.data_quality_results:
                    dq_score = result.final_validation.data_quality_results[name].overall_quality_score
                
                if name in result.final_validation.physics_consistency_results:
                    pc_score = result.final_validation.physics_consistency_results[name].physics_consistency_score
                
                final_scores.append((eq_score + dq_score + pc_score) / 3)
            
            x = np.arange(len(system_names))
            width = 0.35
            
            ax.bar(x - width/2, stat_scores, width, label='Statistical Validation', alpha=0.8)
            ax.bar(x + width/2, final_scores, width, label='Final Validation', alpha=0.8)
            
            ax.axhline(0.7, color='black', linestyle='--', alpha=0.7)
            ax.set_xlabel('System')
            ax.set_ylabel('Validation Score')
            ax.set_title('System-wise Validation Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(system_names, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Quality component breakdown
        ax = axes[0, 2]
        
        if result.statistical_validation:
            # Average quality components across all systems
            quality_components = {
                'Statistical\nSignificance': [],
                'Physics\nConsistency': [],
                'Theoretical\nAgreement': [],
                'Data\nCompleteness': [],
                'Overall\nQuality': []
            }
            
            for stat_result in result.statistical_validation.values():
                if stat_result.quality_metrics:
                    qm = stat_result.quality_metrics
                    quality_components['Statistical\nSignificance'].append(qm.statistical_significance)
                    quality_components['Physics\nConsistency'].append(qm.physics_consistency)
                    quality_components['Theoretical\nAgreement'].append(qm.theoretical_agreement)
                    quality_components['Data\nCompleteness'].append(qm.data_completeness)
                    quality_components['Overall\nQuality'].append(qm.overall_quality_score)
            
            component_names = list(quality_components.keys())
            component_means = [np.mean(quality_components[name]) if quality_components[name] else 0 
                             for name in component_names]
            
            bars = ax.bar(range(len(component_names)), component_means, color='skyblue', alpha=0.7)
            ax.set_xlabel('Quality Component')
            ax.set_ylabel('Average Score')
            ax.set_title('Quality Component Breakdown')
            ax.set_xticks(range(len(component_names)))
            ax.set_xticklabels(component_names, rotation=45)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            # Add values on bars
            for bar, mean_val in zip(bars, component_means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Validation timeline and performance
        ax = axes[1, 0]
        
        performance_metrics = [
            'Systems\nValidated',
            'Total Time\n(minutes)',
            'Success Rate\n(%)',
            'Publication\nReady'
        ]
        
        performance_values = [
            result.systems_validated,
            result.total_validation_time / 60,
            result.statistical_summary['success_rate'] * 100,
            100 if result.ready_for_publication else 0
        ]
        
        # Normalize values for display
        normalized_values = [
            performance_values[0] / 10,  # Assume max 10 systems
            min(1.0, performance_values[1] / 30),  # Assume max 30 minutes
            performance_values[2] / 100,
            performance_values[3] / 100
        ]
        
        bars = ax.bar(range(len(performance_metrics)), normalized_values, 
                     color=['blue', 'orange', 'green', 'purple'], alpha=0.7)
        
        ax.set_xlabel('Performance Metric')
        ax.set_ylabel('Normalized Score')
        ax.set_title('Validation Performance Metrics')
        ax.set_xticks(range(len(performance_metrics)))
        ax.set_xticklabels(performance_metrics, rotation=45)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add actual values as text
        for bar, actual_val in zip(bars, performance_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{actual_val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 5: Issues and recommendations summary
        ax = axes[1, 1]
        
        issue_categories = ['Priority\nIssues', 'Actionable\nRecommendations']
        issue_counts = [len(result.priority_issues), len(result.actionable_recommendations)]
        
        bars = ax.bar(issue_categories, issue_counts, 
                     color=['red' if count > 5 else 'orange' if count > 2 else 'green' 
                           for count in issue_counts], alpha=0.7)
        
        ax.set_xlabel('Category')
        ax.set_ylabel('Count')
        ax.set_title('Issues and Recommendations')
        ax.grid(True, alpha=0.3)
        
        # Add count labels
        for bar, count in zip(bars, issue_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   str(count), ha='center', va='bottom', fontweight='bold')
        
        # Plot 6: Final assessment dashboard
        ax = axes[1, 2]
        ax.axis('off')
        
        # Create assessment dashboard
        dashboard_text = f"VALIDATION DASHBOARD\n\n"
        dashboard_text += f"Overall Status: {'✅ PASSED' if result.overall_validation_passed else '❌ FAILED'}\n"
        dashboard_text += f"Overall Score: {result.overall_validation_score:.3f}\n\n"
        
        dashboard_text += f"Quality Assurance: {'✅ PASSED' if result.quality_assurance_passed else '❌ FAILED'}\n"
        dashboard_text += f"Publication Ready: {'✅ YES' if result.ready_for_publication else '❌ NO'}\n\n"
        
        dashboard_text += f"Validation Components:\n"
        dashboard_text += f"• Statistical: {result.statistical_summary['success_rate']:.1%} success\n"
        dashboard_text += f"• Equilibration: {'✅' if result.final_validation.all_systems_equilibrated else '❌'}\n"
        dashboard_text += f"• Data Quality: {'✅' if result.final_validation.all_data_quality_passed else '❌'}\n"
        dashboard_text += f"• Physics Consistency: {'✅' if result.final_validation.all_physics_consistent else '❌'}\n\n"
        
        dashboard_text += f"Performance:\n"
        dashboard_text += f"• Systems: {result.systems_validated}\n"
        dashboard_text += f"• Time: {result.total_validation_time:.1f}s\n"
        dashboard_text += f"• Issues: {len(result.priority_issues)}\n"
        dashboard_text += f"• Recommendations: {len(result.actionable_recommendations)}\n"
        
        ax.text(0.05, 0.95, dashboard_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        fig.savefig(output_path / "comprehensive_validation_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


def create_comprehensive_validation_integration(confidence_level: float = 0.95,
                                              bootstrap_samples: int = 10000,
                                              significance_threshold: float = 0.05,
                                              quality_threshold: float = 0.7,
                                              equilibration_threshold: float = 1e-4,
                                              parallel_validation: bool = True,
                                              random_seed: Optional[int] = None) -> ComprehensiveValidationIntegration:
    """
    Factory function to create ComprehensiveValidationIntegration.
    
    Args:
        confidence_level: Confidence level for statistical tests
        bootstrap_samples: Number of bootstrap samples
        significance_threshold: P-value threshold for significance
        quality_threshold: Minimum quality score threshold
        equilibration_threshold: Threshold for equilibration convergence
        parallel_validation: Whether to run validations in parallel
        random_seed: Random seed for reproducibility
        
    Returns:
        Configured ComprehensiveValidationIntegration instance
    """
    return ComprehensiveValidationIntegration(
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        significance_threshold=significance_threshold,
        quality_threshold=quality_threshold,
        equilibration_threshold=equilibration_threshold,
        parallel_validation=parallel_validation,
        random_seed=random_seed
    )