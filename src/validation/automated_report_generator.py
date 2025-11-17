"""
Automated Report Generation System

This module provides automated generation of comprehensive reports for reproducible research.
It creates standardized reports with consistent formatting and validation metrics.

Requirements: 5.5, 6.5 - Implement automated report generation for reproducible research
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
from jinja2 import Template
import logging

logger = logging.getLogger(__name__)

@dataclass
class ReportMetadata:
    """Metadata for automated reports."""
    title: str
    author: str
    generation_timestamp: str
    version: str
    project_name: str
    description: str
    tags: List[str]

@dataclass
class ValidationResult:
    """Standardized validation result."""
    test_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    recommendations: List[str]

@dataclass
class SystemResult:
    """Results for a single physics system."""
    system_name: str
    system_type: str
    phase_detection_success: bool
    critical_temperature: Optional[float]
    critical_exponents: Dict[str, float]
    accuracy_metrics: Dict[str, float]
    validation_results: List[ValidationResult]

class AutomatedReportGenerator:
    """
    Automated report generation system for reproducible research.
    
    This class provides standardized report generation with:
    - Consistent formatting and structure
    - Automated validation and quality checks
    - Publication-ready outputs
    - Reproducibility documentation
    """
    
    def __init__(self, output_dir: str = "reports", template_dir: Optional[str] = None):
        """
        Initialize the automated report generator.
        
        Args:
            output_dir: Directory to save generated reports
            template_dir: Directory containing report templates
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up template directory
        if template_dir is None:
            self.template_dir = Path(__file__).parent / "report_templates"
        else:
            self.template_dir = Path(template_dir)
        
        # Create subdirectories
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        
        # Initialize report templates
        self._initialize_templates()
        
        logger.info(f"Initialized automated report generator")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _initialize_templates(self):
        """Initialize report templates."""
        self.templates = {
            'main_report': self._get_main_report_template(),
            'validation_report': self._get_validation_report_template(),
            'system_summary': self._get_system_summary_template(),
            'executive_summary': self._get_executive_summary_template()
        }
    
    def generate_comprehensive_report(
        self,
        metadata: ReportMetadata,
        system_results: List[SystemResult],
        validation_results: List[ValidationResult],
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Generate comprehensive automated report.
        
        Args:
            metadata: Report metadata
            system_results: Results for all analyzed systems
            validation_results: Overall validation results
            additional_data: Additional data to include in report
            
        Returns:
            Dictionary mapping report types to file paths
        """
        logger.info("Generating comprehensive automated report...")
        
        # Prepare report data
        report_data = self._prepare_report_data(
            metadata, system_results, validation_results, additional_data
        )
        
        # Generate different report formats
        generated_reports = {}
        
        # Main markdown report
        main_report_path = self._generate_main_report(report_data)
        generated_reports['main_report'] = str(main_report_path)
        
        # Validation report
        validation_report_path = self._generate_validation_report(report_data)
        generated_reports['validation_report'] = str(validation_report_path)
        
        # Executive summary
        executive_summary_path = self._generate_executive_summary(report_data)
        generated_reports['executive_summary'] = str(executive_summary_path)
        
        # JSON data export
        json_export_path = self._generate_json_export(report_data)
        generated_reports['json_export'] = str(json_export_path)
        
        # Generate figures and tables
        figures_paths = self._generate_report_figures(report_data)
        tables_paths = self._generate_report_tables(report_data)
        
        generated_reports.update(figures_paths)
        generated_reports.update(tables_paths)
        
        # Create reproducibility package
        reproducibility_path = self._create_reproducibility_package(report_data)
        generated_reports['reproducibility_package'] = str(reproducibility_path)
        
        logger.info(f"Comprehensive report generated successfully")
        logger.info(f"Generated {len(generated_reports)} report components")
        
        return generated_reports
    
    def _prepare_report_data(
        self,
        metadata: ReportMetadata,
        system_results: List[SystemResult],
        validation_results: List[ValidationResult],
        additional_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare all data for report generation."""
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(system_results, validation_results)
        
        # Prepare system comparison data
        comparison_data = self._prepare_system_comparison_data(system_results)
        
        # Prepare validation summary
        validation_summary = self._prepare_validation_summary(validation_results)
        
        report_data = {
            'metadata': asdict(metadata),
            'summary_statistics': summary_stats,
            'system_results': [asdict(result) for result in system_results],
            'validation_results': [asdict(result) for result in validation_results],
            'comparison_data': comparison_data,
            'validation_summary': validation_summary,
            'generation_info': {
                'generated_at': datetime.now().isoformat(),
                'generator_version': '1.0.0',
                'total_systems': len(system_results),
                'total_validations': len(validation_results)
            }
        }
        
        # Add additional data if provided
        if additional_data:
            report_data['additional_data'] = additional_data
        
        return report_data
    
    def _calculate_summary_statistics(
        self,
        system_results: List[SystemResult],
        validation_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """Calculate summary statistics for the report."""
        
        # System-level statistics
        total_systems = len(system_results)
        successful_detections = sum(1 for result in system_results if result.phase_detection_success)
        detection_success_rate = successful_detections / total_systems if total_systems > 0 else 0.0
        
        # Accuracy statistics
        all_accuracies = []
        for result in system_results:
            all_accuracies.extend(result.accuracy_metrics.values())
        
        mean_accuracy = np.mean(all_accuracies) if all_accuracies else 0.0
        std_accuracy = np.std(all_accuracies) if all_accuracies else 0.0
        
        # Validation statistics
        total_validations = len(validation_results)
        passed_validations = sum(1 for result in validation_results if result.passed)
        validation_success_rate = passed_validations / total_validations if total_validations > 0 else 0.0
        
        # Overall quality score
        validation_scores = [result.score for result in validation_results]
        overall_quality_score = np.mean(validation_scores) if validation_scores else 0.0
        
        return {
            'total_systems_analyzed': total_systems,
            'successful_phase_detections': successful_detections,
            'phase_detection_success_rate': detection_success_rate,
            'mean_accuracy': mean_accuracy,
            'accuracy_std': std_accuracy,
            'total_validations': total_validations,
            'passed_validations': passed_validations,
            'validation_success_rate': validation_success_rate,
            'overall_quality_score': overall_quality_score,
            'quality_grade': self._calculate_quality_grade(overall_quality_score)
        }
    
    def _calculate_quality_grade(self, score: float) -> str:
        """Calculate quality grade based on score."""
        if score >= 9.0:
            return 'A+ (Excellent)'
        elif score >= 8.0:
            return 'A (Very Good)'
        elif score >= 7.0:
            return 'B (Good)'
        elif score >= 6.0:
            return 'C (Acceptable)'
        else:
            return 'D (Needs Improvement)'
    
    def _prepare_system_comparison_data(self, system_results: List[SystemResult]) -> Dict[str, Any]:
        """Prepare data for system comparison analysis."""
        
        comparison_data = {
            'system_names': [],
            'system_types': [],
            'detection_success': [],
            'critical_temperatures': [],
            'accuracy_scores': [],
            'universality_classes': []
        }
        
        for result in system_results:
            comparison_data['system_names'].append(result.system_name)
            comparison_data['system_types'].append(result.system_type)
            comparison_data['detection_success'].append(result.phase_detection_success)
            comparison_data['critical_temperatures'].append(result.critical_temperature)
            
            # Calculate average accuracy for this system
            if result.accuracy_metrics:
                avg_accuracy = np.mean(list(result.accuracy_metrics.values()))
            else:
                avg_accuracy = 0.0
            comparison_data['accuracy_scores'].append(avg_accuracy)
            
            # Determine universality class based on system type
            universality_class = self._determine_universality_class(result.system_type)
            comparison_data['universality_classes'].append(universality_class)
        
        return comparison_data
    
    def _determine_universality_class(self, system_type: str) -> str:
        """Determine universality class from system type."""
        universality_mapping = {
            'ising_2d': '2D Ising',
            'ising_3d': '3D Ising',
            'potts_3state': 'First-order',
            'xy_2d': 'Kosterlitz-Thouless'
        }
        return universality_mapping.get(system_type, 'Unknown')
    
    def _prepare_validation_summary(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Prepare validation summary data."""
        
        # Group validations by category
        validation_categories = {}
        for result in validation_results:
            category = result.test_name.split('_')[0]  # First part of test name
            if category not in validation_categories:
                validation_categories[category] = []
            validation_categories[category].append(result)
        
        # Calculate category-wise statistics
        category_stats = {}
        for category, results in validation_categories.items():
            total_tests = len(results)
            passed_tests = sum(1 for r in results if r.passed)
            avg_score = np.mean([r.score for r in results])
            
            category_stats[category] = {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
                'average_score': avg_score,
                'grade': self._calculate_quality_grade(avg_score)
            }
        
        return {
            'category_statistics': category_stats,
            'total_categories': len(validation_categories),
            'overall_recommendations': self._generate_overall_recommendations(validation_results)
        }
    
    def _generate_overall_recommendations(self, validation_results: List[ValidationResult]) -> List[str]:
        """Generate overall recommendations based on validation results."""
        
        recommendations = []
        
        # Check for failed validations
        failed_validations = [r for r in validation_results if not r.passed]
        if failed_validations:
            recommendations.append(f"Address {len(failed_validations)} failed validation(s)")
        
        # Check for low scores
        low_score_validations = [r for r in validation_results if r.score < 7.0]
        if low_score_validations:
            recommendations.append(f"Improve {len(low_score_validations)} validation(s) with low scores")
        
        # Add specific recommendations from validation results
        for result in validation_results:
            recommendations.extend(result.recommendations)
        
        # Remove duplicates and limit to top recommendations
        unique_recommendations = list(set(recommendations))
        return unique_recommendations[:10]  # Top 10 recommendations
    
    def _generate_main_report(self, report_data: Dict[str, Any]) -> Path:
        """Generate main markdown report."""
        
        template = self.templates['main_report']
        content = template.render(**report_data)
        
        report_path = self.output_dir / f"main_report_{report_data['generation_info']['generated_at'][:10]}.md"
        
        with open(report_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Main report generated: {report_path}")
        return report_path
    
    def _generate_validation_report(self, report_data: Dict[str, Any]) -> Path:
        """Generate detailed validation report."""
        
        template = self.templates['validation_report']
        content = template.render(**report_data)
        
        report_path = self.output_dir / f"validation_report_{report_data['generation_info']['generated_at'][:10]}.md"
        
        with open(report_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Validation report generated: {report_path}")
        return report_path
    
    def _generate_executive_summary(self, report_data: Dict[str, Any]) -> Path:
        """Generate executive summary."""
        
        template = self.templates['executive_summary']
        content = template.render(**report_data)
        
        summary_path = self.output_dir / f"executive_summary_{report_data['generation_info']['generated_at'][:10]}.md"
        
        with open(summary_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Executive summary generated: {summary_path}")
        return summary_path
    
    def _generate_json_export(self, report_data: Dict[str, Any]) -> Path:
        """Generate JSON export of all data."""
        
        json_path = self.output_dir / f"report_data_{report_data['generation_info']['generated_at'][:10]}.json"
        
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"JSON export generated: {json_path}")
        return json_path
    
    def _generate_report_figures(self, report_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate all report figures."""
        
        figures = {}
        
        # Figure 1: System comparison
        fig1_path = self._create_system_comparison_figure(report_data)
        figures['system_comparison_figure'] = str(fig1_path)
        
        # Figure 2: Validation results
        fig2_path = self._create_validation_results_figure(report_data)
        figures['validation_results_figure'] = str(fig2_path)
        
        # Figure 3: Accuracy distribution
        fig3_path = self._create_accuracy_distribution_figure(report_data)
        figures['accuracy_distribution_figure'] = str(fig3_path)
        
        return figures
    
    def _create_system_comparison_figure(self, report_data: Dict[str, Any]) -> Path:
        """Create system comparison figure."""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        comparison_data = report_data['comparison_data']
        
        # Detection success rates
        systems = comparison_data['system_names']
        success_rates = [1.0 if success else 0.0 for success in comparison_data['detection_success']]
        
        axes[0].bar(systems, success_rates, alpha=0.7)
        axes[0].set_ylabel('Detection Success Rate')
        axes[0].set_title('Phase Detection Success by System')
        axes[0].set_ylim(0, 1.1)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
        
        # Accuracy scores
        accuracy_scores = comparison_data['accuracy_scores']
        axes[1].bar(systems, accuracy_scores, alpha=0.7, color='orange')
        axes[1].set_ylabel('Average Accuracy (%)')
        axes[1].set_title('Average Accuracy by System')
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        fig_path = self.output_dir / 'figures' / 'system_comparison.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig_path
    
    def _create_validation_results_figure(self, report_data: Dict[str, Any]) -> Path:
        """Create validation results figure."""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        validation_summary = report_data['validation_summary']
        category_stats = validation_summary['category_statistics']
        
        categories = list(category_stats.keys())
        success_rates = [stats['success_rate'] for stats in category_stats.values()]
        avg_scores = [stats['average_score'] for stats in category_stats.values()]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, success_rates, width, label='Success Rate', alpha=0.7)
        ax.bar(x + width/2, [score/10 for score in avg_scores], width, label='Avg Score (scaled)', alpha=0.7)
        
        ax.set_ylabel('Rate / Score')
        ax.set_title('Validation Results by Category')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        fig_path = self.output_dir / 'figures' / 'validation_results.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig_path
    
    def _create_accuracy_distribution_figure(self, report_data: Dict[str, Any]) -> Path:
        """Create accuracy distribution figure."""
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Collect all accuracy values
        all_accuracies = []
        for system_result in report_data['system_results']:
            all_accuracies.extend(system_result['accuracy_metrics'].values())
        
        if all_accuracies:
            ax.hist(all_accuracies, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(all_accuracies), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(all_accuracies):.1f}%')
            ax.set_xlabel('Accuracy (%)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Accuracy Scores')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No accuracy data available', 
                   transform=ax.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        
        fig_path = self.output_dir / 'figures' / 'accuracy_distribution.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig_path
    
    def _generate_report_tables(self, report_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate all report tables."""
        
        tables = {}
        
        # Table 1: System summary
        table1_path = self._create_system_summary_table(report_data)
        tables['system_summary_table'] = str(table1_path)
        
        # Table 2: Validation summary
        table2_path = self._create_validation_summary_table(report_data)
        tables['validation_summary_table'] = str(table2_path)
        
        return tables
    
    def _create_system_summary_table(self, report_data: Dict[str, Any]) -> Path:
        """Create system summary table."""
        
        table_data = []
        
        for system_result in report_data['system_results']:
            row = {
                'System': system_result['system_name'],
                'Type': system_result['system_type'],
                'Phase Detection': '✓' if system_result['phase_detection_success'] else '✗',
                'Critical Temperature': system_result['critical_temperature'] if system_result['critical_temperature'] else 'N/A',
                'Average Accuracy': f"{np.mean(list(system_result['accuracy_metrics'].values())):.1f}%" if system_result['accuracy_metrics'] else 'N/A'
            }
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        table_path = self.output_dir / 'tables' / 'system_summary.csv'
        df.to_csv(table_path, index=False)
        
        return table_path
    
    def _create_validation_summary_table(self, report_data: Dict[str, Any]) -> Path:
        """Create validation summary table."""
        
        table_data = []
        
        for validation_result in report_data['validation_results']:
            row = {
                'Test Name': validation_result['test_name'],
                'Status': 'PASS' if validation_result['passed'] else 'FAIL',
                'Score': f"{validation_result['score']:.1f}",
                'Recommendations': '; '.join(validation_result['recommendations'][:3])  # First 3 recommendations
            }
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        table_path = self.output_dir / 'tables' / 'validation_summary.csv'
        df.to_csv(table_path, index=False)
        
        return table_path
    
    def _create_reproducibility_package(self, report_data: Dict[str, Any]) -> Path:
        """Create reproducibility package."""
        
        reproducibility_data = {
            'metadata': report_data['metadata'],
            'generation_info': report_data['generation_info'],
            'environment': {
                'python_version': '3.8+',
                'required_packages': [
                    'numpy>=1.19.0',
                    'scipy>=1.5.0',
                    'matplotlib>=3.3.0',
                    'pandas>=1.1.0',
                    'torch>=1.7.0'
                ],
                'hardware_requirements': 'GPU recommended for VAE training'
            },
            'data_sources': {
                'monte_carlo_parameters': 'Documented in data generation scripts',
                'vae_architectures': 'Documented in model configuration files',
                'analysis_parameters': 'Documented in analysis scripts'
            },
            'validation_procedures': {
                'statistical_tests': 'Bootstrap confidence intervals with 1000 samples',
                'physics_validation': 'Comparison with theoretical predictions',
                'computational_validation': 'Convergence and stability testing'
            }
        }
        
        package_path = self.output_dir / 'reproducibility_package.yaml'
        
        with open(package_path, 'w') as f:
            yaml.dump(reproducibility_data, f, default_flow_style=False)
        
        logger.info(f"Reproducibility package created: {package_path}")
        return package_path
    
    # Template methods
    def _get_main_report_template(self) -> Template:
        """Get main report template."""
        template_str = """# {{ metadata.title }}

**Author:** {{ metadata.author }}  
**Generated:** {{ generation_info.generated_at }}  
**Version:** {{ metadata.version }}

## Executive Summary

{{ metadata.description }}

### Key Statistics
- **Systems Analyzed:** {{ summary_statistics.total_systems_analyzed }}
- **Phase Detection Success Rate:** {{ "%.1f"|format(summary_statistics.phase_detection_success_rate * 100) }}%
- **Mean Accuracy:** {{ "%.1f"|format(summary_statistics.mean_accuracy) }}%
- **Overall Quality Score:** {{ "%.1f"|format(summary_statistics.overall_quality_score) }}/10 ({{ summary_statistics.quality_grade }})

## System Results

{% for system in system_results %}
### {{ system.system_name }}
- **Type:** {{ system.system_type }}
- **Phase Detection:** {{ "✓" if system.phase_detection_success else "✗" }}
- **Critical Temperature:** {{ system.critical_temperature if system.critical_temperature else "N/A" }}
- **Critical Exponents:** {{ system.critical_exponents }}
- **Accuracy Metrics:** {{ system.accuracy_metrics }}

{% endfor %}

## Validation Summary

**Overall Validation Success Rate:** {{ "%.1f"|format(validation_summary.category_statistics.get('overall', {}).get('success_rate', 0) * 100) }}%

### Validation Categories
{% for category, stats in validation_summary.category_statistics.items() %}
- **{{ category.title() }}:** {{ stats.passed_tests }}/{{ stats.total_tests }} tests passed ({{ "%.1f"|format(stats.success_rate * 100) }}%)
{% endfor %}

## Recommendations

{% for recommendation in validation_summary.overall_recommendations %}
- {{ recommendation }}
{% endfor %}

---
*This report was automatically generated by the Automated Report Generation System.*
"""
        return Template(template_str)
    
    def _get_validation_report_template(self) -> Template:
        """Get validation report template."""
        template_str = """# Validation Report: {{ metadata.title }}

**Generated:** {{ generation_info.generated_at }}

## Validation Overview

This report provides detailed validation results for all tested components.

### Summary Statistics
- **Total Validations:** {{ summary_statistics.total_validations }}
- **Passed Validations:** {{ summary_statistics.passed_validations }}
- **Success Rate:** {{ "%.1f"|format(summary_statistics.validation_success_rate * 100) }}%
- **Average Score:** {{ "%.1f"|format(summary_statistics.overall_quality_score) }}/10

## Detailed Validation Results

{% for validation in validation_results %}
### {{ validation.test_name }}
- **Status:** {{ "PASS" if validation.passed else "FAIL" }}
- **Score:** {{ "%.1f"|format(validation.score) }}/10
- **Details:** {{ validation.details }}
- **Recommendations:**
{% for rec in validation.recommendations %}
  - {{ rec }}
{% endfor %}

{% endfor %}

## Category Analysis

{% for category, stats in validation_summary.category_statistics.items() %}
### {{ category.title() }} Category
- **Tests:** {{ stats.total_tests }}
- **Passed:** {{ stats.passed_tests }}
- **Success Rate:** {{ "%.1f"|format(stats.success_rate * 100) }}%
- **Average Score:** {{ "%.1f"|format(stats.average_score) }}/10
- **Grade:** {{ stats.grade }}

{% endfor %}
"""
        return Template(template_str)
    
    def _get_system_summary_template(self) -> Template:
        """Get system summary template."""
        template_str = """# System Summary: {{ system.system_name }}

**System Type:** {{ system.system_type }}  
**Phase Detection Success:** {{ "✓" if system.phase_detection_success else "✗" }}

## Critical Properties
- **Critical Temperature:** {{ system.critical_temperature if system.critical_temperature else "N/A" }}
- **Critical Exponents:** {{ system.critical_exponents }}

## Accuracy Metrics
{% for metric, value in system.accuracy_metrics.items() %}
- **{{ metric }}:** {{ "%.1f"|format(value) }}%
{% endfor %}

## Validation Results
{% for validation in system.validation_results %}
- **{{ validation.test_name }}:** {{ "PASS" if validation.passed else "FAIL" }} ({{ "%.1f"|format(validation.score) }}/10)
{% endfor %}
"""
        return Template(template_str)
    
    def _get_executive_summary_template(self) -> Template:
        """Get executive summary template."""
        template_str = """# Executive Summary: {{ metadata.title }}

## Project Overview
{{ metadata.description }}

## Key Achievements
- Successfully analyzed {{ summary_statistics.total_systems_analyzed }} physics systems
- Achieved {{ "%.1f"|format(summary_statistics.phase_detection_success_rate * 100) }}% phase detection success rate
- Maintained {{ "%.1f"|format(summary_statistics.mean_accuracy) }}% average accuracy across all metrics
- Completed {{ summary_statistics.total_validations }} validation tests with {{ "%.1f"|format(summary_statistics.validation_success_rate * 100) }}% success rate

## Quality Assessment
**Overall Quality Score:** {{ "%.1f"|format(summary_statistics.overall_quality_score) }}/10 ({{ summary_statistics.quality_grade }})

## Recommendations for Publication
{% for recommendation in validation_summary.overall_recommendations[:5] %}
{{ loop.index }}. {{ recommendation }}
{% endfor %}

## Conclusion
{% if summary_statistics.overall_quality_score >= 8.0 %}
This work demonstrates high quality and is ready for publication consideration.
{% elif summary_statistics.overall_quality_score >= 7.0 %}
This work shows good quality with minor improvements needed before publication.
{% else %}
This work requires significant improvements before publication consideration.
{% endif %}
"""
        return Template(template_str)


# Utility functions for creating report components
def create_system_result(
    system_name: str,
    system_type: str,
    phase_detection_success: bool,
    critical_temperature: Optional[float] = None,
    critical_exponents: Optional[Dict[str, float]] = None,
    accuracy_metrics: Optional[Dict[str, float]] = None,
    validation_results: Optional[List[ValidationResult]] = None
) -> SystemResult:
    """Create a SystemResult object with default values."""
    return SystemResult(
        system_name=system_name,
        system_type=system_type,
        phase_detection_success=phase_detection_success,
        critical_temperature=critical_temperature,
        critical_exponents=critical_exponents or {},
        accuracy_metrics=accuracy_metrics or {},
        validation_results=validation_results or []
    )

def create_validation_result(
    test_name: str,
    passed: bool,
    score: float,
    details: Optional[Dict[str, Any]] = None,
    recommendations: Optional[List[str]] = None
) -> ValidationResult:
    """Create a ValidationResult object with default values."""
    return ValidationResult(
        test_name=test_name,
        passed=passed,
        score=score,
        details=details or {},
        recommendations=recommendations or []
    )

def create_report_metadata(
    title: str,
    author: str = "Automated System",
    version: str = "1.0.0",
    project_name: str = "Physics Analysis Project",
    description: str = "Automated analysis report",
    tags: Optional[List[str]] = None
) -> ReportMetadata:
    """Create ReportMetadata object with default values."""
    return ReportMetadata(
        title=title,
        author=author,
        generation_timestamp=datetime.now().isoformat(),
        version=version,
        project_name=project_name,
        description=description,
        tags=tags or []
    )