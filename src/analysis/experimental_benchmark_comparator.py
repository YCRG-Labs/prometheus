"""
Experimental Benchmark Comparator

This module provides functionality for comparing computational physics results
with experimental benchmark data, including statistical comparison methods,
agreement metrics, and meta-analysis capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import json
import logging
from scipy import stats
from scipy.stats import chi2
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from .enhanced_validation_types import (
    ExperimentalComparison,
    MetaAnalysisResult,
    PhysicsViolation,
    ViolationSeverity,
    ExperimentalComparisonError,
    BaseEnhancedValidator
)

logger = logging.getLogger(__name__)


class ExperimentalBenchmarkComparator(BaseEnhancedValidator):
    """
    Comparator for validating computational results against experimental benchmarks.
    
    This class provides methods for loading experimental data, performing statistical
    comparisons, computing agreement metrics, and conducting meta-analysis across
    multiple experimental datasets.
    """
    
    def __init__(self, benchmark_data_path: Optional[str] = None):
        """
        Initialize the experimental benchmark comparator.
        
        Args:
            benchmark_data_path: Path to directory containing experimental benchmark data
        """
        super().__init__()
        self.benchmark_data_path = Path(benchmark_data_path) if benchmark_data_path else None
        self.loaded_datasets: Dict[str, Dict[str, Any]] = {}
        self._initialize_default_benchmarks()
    
    def _initialize_default_benchmarks(self) -> None:
        """Initialize default experimental benchmark datasets."""
        # Default Ising model benchmarks from literature
        self.default_benchmarks = {
            "ising_2d_onsager": {
                "system": "2D Ising Model",
                "reference": "Onsager (1944)",
                "critical_temperature": 2.269185,
                "critical_temperature_uncertainty": 0.000001,
                "beta_exponent": 0.125,
                "beta_uncertainty": 0.001,
                "gamma_exponent": 1.75,
                "gamma_uncertainty": 0.01,
                "nu_exponent": 1.0,
                "nu_uncertainty": 0.01,
                "dimensionality": 2,
                "lattice_type": "square"
            },
            "ising_3d_experimental": {
                "system": "3D Ising Model",
                "reference": "Pelissetto & Vicari (2002)",
                "critical_temperature": 4.511526,
                "critical_temperature_uncertainty": 0.000010,
                "beta_exponent": 0.32653,
                "beta_uncertainty": 0.00010,
                "gamma_exponent": 1.2372,
                "gamma_uncertainty": 0.0005,
                "nu_exponent": 0.63002,
                "nu_uncertainty": 0.00010,
                "dimensionality": 3,
                "lattice_type": "cubic"
            },
            "xy_2d_kosterlitz_thouless": {
                "system": "2D XY Model",
                "reference": "Kosterlitz & Thouless (1973)",
                "critical_temperature": 0.8935,
                "critical_temperature_uncertainty": 0.0005,
                "eta_exponent": 0.25,
                "eta_uncertainty": 0.01,
                "dimensionality": 2,
                "transition_type": "BKT"
            },
            "heisenberg_3d_experimental": {
                "system": "3D Heisenberg Model",
                "reference": "Campostrini et al. (2006)",
                "critical_temperature": 1.4431,
                "critical_temperature_uncertainty": 0.0002,
                "beta_exponent": 0.3689,
                "beta_uncertainty": 0.0003,
                "gamma_exponent": 1.3960,
                "gamma_uncertainty": 0.0009,
                "nu_exponent": 0.7112,
                "nu_uncertainty": 0.0005,
                "dimensionality": 3,
                "lattice_type": "cubic"
            }
        }
    
    def load_experimental_benchmark(self, dataset_name: str) -> Dict[str, Any]:
        """
        Load experimental benchmark dataset.
        
        Args:
            dataset_name: Name of the experimental dataset to load
            
        Returns:
            Dictionary containing experimental benchmark data
            
        Raises:
            ExperimentalComparisonError: If dataset cannot be loaded
        """
        try:
            # First check if it's a default benchmark
            if dataset_name in self.default_benchmarks:
                dataset = self.default_benchmarks[dataset_name].copy()
                self.loaded_datasets[dataset_name] = dataset
                logger.info(f"Loaded default benchmark dataset: {dataset_name}")
                return dataset
            
            # Try to load from file if benchmark_data_path is provided
            if self.benchmark_data_path and self.benchmark_data_path.exists():
                dataset_file = self.benchmark_data_path / f"{dataset_name}.json"
                if dataset_file.exists():
                    with open(dataset_file, 'r') as f:
                        dataset = json.load(f)
                    self.loaded_datasets[dataset_name] = dataset
                    logger.info(f"Loaded benchmark dataset from file: {dataset_name}")
                    return dataset
                
                # Try CSV format
                csv_file = self.benchmark_data_path / f"{dataset_name}.csv"
                if csv_file.exists():
                    df = pd.read_csv(csv_file)
                    dataset = df.to_dict('records')[0] if len(df) > 0 else {}
                    self.loaded_datasets[dataset_name] = dataset
                    logger.info(f"Loaded benchmark dataset from CSV: {dataset_name}")
                    return dataset
            
            raise ExperimentalComparisonError(
                f"Experimental dataset '{dataset_name}' not found in default benchmarks "
                f"or benchmark data path"
            )
            
        except Exception as e:
            raise ExperimentalComparisonError(
                f"Failed to load experimental dataset '{dataset_name}': {str(e)}"
            )
    
    def compare_with_experimental_data(
        self,
        computational_results: Dict[str, float],
        experimental_dataset: str,
        properties_to_compare: Optional[List[str]] = None
    ) -> List[ExperimentalComparison]:
        """
        Compare computational results with experimental data.
        
        Args:
            computational_results: Dictionary of computed physics properties
            experimental_dataset: Name of experimental dataset to compare against
            properties_to_compare: List of properties to compare (if None, compare all available)
            
        Returns:
            List of ExperimentalComparison objects
        """
        try:
            # Load experimental data
            exp_data = self.load_experimental_benchmark(experimental_dataset)
            
            # Determine properties to compare
            if properties_to_compare is None:
                # Find common properties between computational and experimental data
                comp_props = set(computational_results.keys())
                exp_props = set(key for key in exp_data.keys() 
                              if not key.endswith('_uncertainty') and 
                              key not in ['system', 'reference', 'dimensionality', 'lattice_type', 'transition_type'])
                properties_to_compare = list(comp_props.intersection(exp_props))
            
            comparisons = []
            
            for prop in properties_to_compare:
                if prop not in computational_results:
                    logger.warning(f"Property '{prop}' not found in computational results")
                    continue
                
                if prop not in exp_data:
                    logger.warning(f"Property '{prop}' not found in experimental data")
                    continue
                
                # Get experimental uncertainty
                uncertainty_key = f"{prop}_uncertainty"
                exp_uncertainty = exp_data.get(uncertainty_key, 0.0)
                
                # Compute comparison metrics
                comp_value = computational_results[prop]
                exp_value = exp_data[prop]
                
                # Agreement metric (normalized absolute difference)
                if exp_uncertainty > 0:
                    agreement_metric = 1.0 - min(1.0, abs(comp_value - exp_value) / exp_uncertainty)
                else:
                    # Use relative difference if no uncertainty available
                    if exp_value != 0:
                        agreement_metric = 1.0 - min(1.0, abs(comp_value - exp_value) / abs(exp_value))
                    else:
                        agreement_metric = 1.0 if comp_value == exp_value else 0.0
                
                # Z-score for statistical significance
                if exp_uncertainty > 0:
                    z_score = abs(comp_value - exp_value) / exp_uncertainty
                    # Two-tailed p-value
                    statistical_significance = 2 * (1 - stats.norm.cdf(abs(z_score)))
                else:
                    z_score = 0.0
                    statistical_significance = 1.0 if comp_value == exp_value else 0.0
                
                # Generate discrepancy explanation if needed
                discrepancy_explanation = None
                if agreement_metric < 0.8:  # Threshold for significant discrepancy
                    discrepancy_explanation = self._generate_discrepancy_explanation(
                        prop, comp_value, exp_value, exp_uncertainty, z_score
                    )
                
                comparison = ExperimentalComparison(
                    experimental_dataset=experimental_dataset,
                    computational_value=comp_value,
                    experimental_value=exp_value,
                    experimental_uncertainty=exp_uncertainty,
                    agreement_metric=agreement_metric,
                    statistical_significance=statistical_significance,
                    z_score=z_score,
                    discrepancy_explanation=discrepancy_explanation
                )
                
                comparisons.append(comparison)
                
                # Check for violations
                if agreement_metric < 0.5:  # Severe disagreement
                    violation = PhysicsViolation(
                        violation_type="experimental_disagreement",
                        severity=ViolationSeverity.HIGH,
                        description=f"Severe disagreement with experimental data for {prop}",
                        suggested_investigation=f"Review computational method for {prop} calculation",
                        physics_explanation=discrepancy_explanation or "Unknown cause of disagreement",
                        quantitative_measure=agreement_metric,
                        threshold_value=0.5
                    )
                    self.add_violation(violation)
            
            logger.info(f"Completed comparison with {experimental_dataset}: {len(comparisons)} properties compared")
            return comparisons
            
        except Exception as e:
            raise ExperimentalComparisonError(
                f"Failed to compare with experimental data '{experimental_dataset}': {str(e)}"
            )
    
    def _generate_discrepancy_explanation(
        self,
        property_name: str,
        comp_value: float,
        exp_value: float,
        exp_uncertainty: float,
        z_score: float
    ) -> str:
        """Generate explanation for discrepancy between computational and experimental results."""
        
        relative_diff = abs(comp_value - exp_value) / abs(exp_value) if exp_value != 0 else float('inf')
        
        explanations = []
        
        if z_score > 3:
            explanations.append("Statistical significance suggests systematic error")
        
        if relative_diff > 0.1:
            explanations.append("Large relative difference may indicate model limitations")
        
        if property_name in ['critical_temperature', 'beta_exponent', 'gamma_exponent', 'nu_exponent']:
            if comp_value > exp_value:
                explanations.append("Overestimation may be due to finite-size effects or insufficient equilibration")
            else:
                explanations.append("Underestimation may be due to inadequate sampling or model approximations")
        
        if not explanations:
            explanations.append("Discrepancy within expected computational uncertainty")
        
        return "; ".join(explanations)
    
    def compute_agreement_metrics(
        self,
        comparisons: List[ExperimentalComparison]
    ) -> Dict[str, float]:
        """
        Compute overall agreement metrics across multiple comparisons.
        
        Args:
            comparisons: List of experimental comparisons
            
        Returns:
            Dictionary of agreement metrics
        """
        if not comparisons:
            return {}
        
        agreement_scores = [comp.agreement_metric for comp in comparisons]
        z_scores = [comp.z_score for comp in comparisons]
        p_values = [comp.statistical_significance for comp in comparisons]
        
        # Overall agreement metrics
        metrics = {
            'mean_agreement': np.mean(agreement_scores),
            'median_agreement': np.median(agreement_scores),
            'min_agreement': np.min(agreement_scores),
            'agreement_std': np.std(agreement_scores),
            'fraction_good_agreement': np.mean([score > 0.8 for score in agreement_scores]),
            'mean_z_score': np.mean(z_scores),
            'max_z_score': np.max(z_scores),
            'fraction_significant': np.mean([p < 0.05 for p in p_values]),
            'n_comparisons': len(comparisons)
        }
        
        # Chi-square goodness of fit test
        if len(z_scores) > 1:
            chi2_stat = np.sum(np.array(z_scores)**2)
            chi2_p_value = 1 - chi2.cdf(chi2_stat, len(z_scores))
            metrics['chi2_statistic'] = chi2_stat
            metrics['chi2_p_value'] = chi2_p_value
        
        return metrics
    
    def perform_meta_analysis(
        self,
        experimental_comparisons: List[ExperimentalComparison],
        property_name: str
    ) -> MetaAnalysisResult:
        """
        Perform meta-analysis across multiple experimental datasets.
        
        Args:
            experimental_comparisons: List of experimental comparisons for the same property
            property_name: Name of the property being analyzed
            
        Returns:
            MetaAnalysisResult object
        """
        try:
            if not experimental_comparisons:
                raise ExperimentalComparisonError("No experimental comparisons provided for meta-analysis")
            
            # Extract values and weights
            exp_values = []
            exp_uncertainties = []
            comp_values = []
            weights = []
            
            for comp in experimental_comparisons:
                exp_values.append(comp.experimental_value)
                exp_uncertainties.append(comp.experimental_uncertainty)
                comp_values.append(comp.computational_value)
                
                # Weight by inverse variance (if uncertainty available)
                if comp.experimental_uncertainty > 0:
                    weight = 1.0 / (comp.experimental_uncertainty ** 2)
                else:
                    weight = 1.0
                weights.append(weight)
            
            exp_values = np.array(exp_values)
            exp_uncertainties = np.array(exp_uncertainties)
            comp_values = np.array(comp_values)
            weights = np.array(weights)
            
            # Weighted pooled estimates
            total_weight = np.sum(weights)
            pooled_exp_estimate = np.sum(weights * exp_values) / total_weight
            pooled_comp_estimate = np.sum(weights * comp_values) / total_weight
            pooled_uncertainty = np.sqrt(1.0 / total_weight)
            
            # Heterogeneity analysis (I² statistic)
            if len(experimental_comparisons) > 1:
                # Q statistic
                q_stat = np.sum(weights * (exp_values - pooled_exp_estimate)**2)
                df = len(experimental_comparisons) - 1
                
                # I² statistic (percentage of variation due to heterogeneity)
                if q_stat > df:
                    i_squared = ((q_stat - df) / q_stat) * 100
                else:
                    i_squared = 0.0
                
                heterogeneity_statistic = i_squared
            else:
                heterogeneity_statistic = 0.0
            
            # Forest plot data preparation
            forest_plot_data = {
                'studies': [comp.experimental_dataset for comp in experimental_comparisons],
                'experimental_values': exp_values.tolist(),
                'experimental_uncertainties': exp_uncertainties.tolist(),
                'computational_values': comp_values.tolist(),
                'weights': weights.tolist(),
                'pooled_experimental': pooled_exp_estimate,
                'pooled_computational': pooled_comp_estimate,
                'pooled_uncertainty': pooled_uncertainty
            }
            
            # Publication bias analysis (Egger's test if enough studies)
            publication_bias_analysis = {}
            if len(experimental_comparisons) >= 3:
                # Simple funnel plot asymmetry test
                effect_sizes = comp_values - exp_values
                standard_errors = exp_uncertainties
                
                if np.all(standard_errors > 0):
                    # Linear regression of effect size on standard error
                    try:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            standard_errors, effect_sizes
                        )
                        publication_bias_analysis = {
                            'egger_slope': slope,
                            'egger_intercept': intercept,
                            'egger_p_value': p_value,
                            'asymmetry_detected': p_value < 0.05
                        }
                    except Exception as e:
                        logger.warning(f"Could not perform Egger's test: {e}")
            
            result = MetaAnalysisResult(
                n_studies=len(experimental_comparisons),
                pooled_estimate=pooled_exp_estimate,
                pooled_uncertainty=pooled_uncertainty,
                heterogeneity_statistic=heterogeneity_statistic,
                individual_studies=experimental_comparisons,
                forest_plot_data=forest_plot_data,
                publication_bias_analysis=publication_bias_analysis
            )
            
            logger.info(f"Completed meta-analysis for {property_name}: {len(experimental_comparisons)} studies")
            return result
            
        except Exception as e:
            raise ExperimentalComparisonError(
                f"Failed to perform meta-analysis for {property_name}: {str(e)}"
            )
    
    def validate_against_multiple_benchmarks(
        self,
        computational_results: Dict[str, float],
        benchmark_datasets: List[str],
        properties_to_compare: Optional[List[str]] = None
    ) -> Dict[str, List[ExperimentalComparison]]:
        """
        Validate computational results against multiple experimental benchmarks.
        
        Args:
            computational_results: Dictionary of computed physics properties
            benchmark_datasets: List of experimental dataset names
            properties_to_compare: List of properties to compare
            
        Returns:
            Dictionary mapping dataset names to comparison results
        """
        all_comparisons = {}
        
        for dataset in benchmark_datasets:
            try:
                comparisons = self.compare_with_experimental_data(
                    computational_results, dataset, properties_to_compare
                )
                all_comparisons[dataset] = comparisons
                logger.info(f"Completed validation against {dataset}")
            except Exception as e:
                logger.error(f"Failed validation against {dataset}: {e}")
                all_comparisons[dataset] = []
        
        return all_comparisons
    
    def generate_comparison_summary(
        self,
        all_comparisons: Dict[str, List[ExperimentalComparison]]
    ) -> Dict[str, Any]:
        """
        Generate summary of comparisons across multiple datasets.
        
        Args:
            all_comparisons: Dictionary of comparison results by dataset
            
        Returns:
            Summary dictionary with overall statistics
        """
        # Flatten all comparisons
        all_comp_list = []
        for dataset_comps in all_comparisons.values():
            all_comp_list.extend(dataset_comps)
        
        if not all_comp_list:
            return {"error": "No successful comparisons found"}
        
        # Overall agreement metrics
        overall_metrics = self.compute_agreement_metrics(all_comp_list)
        
        # Per-dataset summary
        dataset_summaries = {}
        for dataset, comparisons in all_comparisons.items():
            if comparisons:
                dataset_metrics = self.compute_agreement_metrics(comparisons)
                dataset_summaries[dataset] = dataset_metrics
        
        # Property-wise analysis
        property_summaries = {}
        properties = set()
        for comp in all_comp_list:
            # Extract property name from dataset (simplified approach)
            for prop in ['critical_temperature', 'beta_exponent', 'gamma_exponent', 'nu_exponent']:
                if prop in comp.experimental_dataset or abs(comp.experimental_value - 
                    self.default_benchmarks.get(comp.experimental_dataset, {}).get(prop, float('inf'))) < 1e-6:
                    properties.add(prop)
        
        for prop in properties:
            prop_comparisons = []
            for comp in all_comp_list:
                # Check if this comparison is for this property
                exp_data = self.loaded_datasets.get(comp.experimental_dataset, {})
                if prop in exp_data and abs(comp.experimental_value - exp_data[prop]) < 1e-6:
                    prop_comparisons.append(comp)
            
            if prop_comparisons:
                property_summaries[prop] = self.compute_agreement_metrics(prop_comparisons)
        
        summary = {
            'overall_metrics': overall_metrics,
            'dataset_summaries': dataset_summaries,
            'property_summaries': property_summaries,
            'total_comparisons': len(all_comp_list),
            'successful_datasets': len([d for d, c in all_comparisons.items() if c]),
            'violations': [v.__dict__ for v in self.violations]
        }
        
        return summary
    
    def validate(
        self,
        computational_results: Dict[str, float],
        benchmark_datasets: Optional[List[str]] = None,
        properties_to_compare: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Main validation method for experimental benchmark comparison.
        
        Args:
            computational_results: Dictionary of computed physics properties
            benchmark_datasets: List of experimental datasets (if None, use defaults)
            properties_to_compare: List of properties to compare
            
        Returns:
            Comprehensive validation results
        """
        if benchmark_datasets is None:
            benchmark_datasets = list(self.default_benchmarks.keys())
        
        # Clear previous violations
        self.clear_violations()
        
        # Perform comparisons
        all_comparisons = self.validate_against_multiple_benchmarks(
            computational_results, benchmark_datasets, properties_to_compare
        )
        
        # Generate summary
        summary = self.generate_comparison_summary(all_comparisons)
        
        # Perform meta-analysis for each property
        meta_analyses = {}
        if properties_to_compare:
            for prop in properties_to_compare:
                prop_comparisons = []
                for dataset_comps in all_comparisons.values():
                    for comp in dataset_comps:
                        # Check if this comparison is for this property
                        exp_data = self.loaded_datasets.get(comp.experimental_dataset, {})
                        if prop in exp_data and abs(comp.experimental_value - exp_data[prop]) < 1e-6:
                            prop_comparisons.append(comp)
                
                if len(prop_comparisons) > 1:
                    try:
                        meta_analysis = self.perform_meta_analysis(prop_comparisons, prop)
                        meta_analyses[prop] = meta_analysis
                    except Exception as e:
                        logger.warning(f"Meta-analysis failed for {prop}: {e}")
        
        return {
            'comparisons': all_comparisons,
            'summary': summary,
            'meta_analyses': meta_analyses,
            'violations': self.violations
        }
    
    def generate_discrepancy_analysis(
        self,
        comparisons: List[ExperimentalComparison],
        threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Generate detailed analysis of discrepancies between computational and experimental results.
        
        Args:
            comparisons: List of experimental comparisons
            threshold: Agreement threshold below which to analyze discrepancies
            
        Returns:
            Dictionary containing discrepancy analysis
        """
        discrepant_comparisons = [comp for comp in comparisons if comp.agreement_metric < threshold]
        
        if not discrepant_comparisons:
            return {
                'n_discrepancies': 0,
                'message': 'No significant discrepancies found'
            }
        
        # Categorize discrepancies
        discrepancy_categories = {
            'severe': [comp for comp in discrepant_comparisons if comp.agreement_metric < 0.5],
            'moderate': [comp for comp in discrepant_comparisons if 0.5 <= comp.agreement_metric < 0.7],
            'mild': [comp for comp in discrepant_comparisons if 0.7 <= comp.agreement_metric < threshold]
        }
        
        # Analyze patterns
        systematic_bias = np.mean([comp.computational_value - comp.experimental_value 
                                 for comp in discrepant_comparisons])
        
        # Property-specific analysis
        property_analysis = {}
        for comp in discrepant_comparisons:
            dataset = comp.experimental_dataset
            if dataset not in property_analysis:
                property_analysis[dataset] = []
            property_analysis[dataset].append({
                'agreement': comp.agreement_metric,
                'z_score': comp.z_score,
                'explanation': comp.discrepancy_explanation
            })
        
        # Generate recommendations
        recommendations = []
        if len(discrepancy_categories['severe']) > 0:
            recommendations.append("Review computational methodology for severe discrepancies")
        if abs(systematic_bias) > 0.1:
            recommendations.append("Investigate systematic bias in computational results")
        if len(discrepant_comparisons) / len(comparisons) > 0.5:
            recommendations.append("Consider model limitations or parameter adjustments")
        
        return {
            'n_discrepancies': len(discrepant_comparisons),
            'discrepancy_categories': {k: len(v) for k, v in discrepancy_categories.items()},
            'systematic_bias': systematic_bias,
            'property_analysis': property_analysis,
            'recommendations': recommendations,
            'discrepant_comparisons': discrepant_comparisons
        }
    
    def create_forest_plot(
        self,
        meta_analysis_result: MetaAnalysisResult,
        property_name: str,
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Create forest plot for meta-analysis results.
        
        Args:
            meta_analysis_result: Meta-analysis results
            property_name: Name of the property being plotted
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        forest_data = meta_analysis_result.forest_plot_data
        studies = forest_data['studies']
        exp_values = np.array(forest_data['experimental_values'])
        exp_uncertainties = np.array(forest_data['experimental_uncertainties'])
        comp_values = np.array(forest_data['computational_values'])
        weights = np.array(forest_data['weights'])
        
        # Normalize weights for plotting
        normalized_weights = weights / np.max(weights) * 100
        
        y_positions = np.arange(len(studies))
        
        # Plot experimental values with error bars
        ax.errorbar(exp_values, y_positions, xerr=exp_uncertainties, 
                   fmt='s', color='blue', label='Experimental', 
                   markersize=normalized_weights/10, alpha=0.7)
        
        # Plot computational values
        ax.scatter(comp_values, y_positions, marker='o', color='red', 
                  s=normalized_weights, label='Computational', alpha=0.7)
        
        # Plot pooled estimates
        pooled_exp = forest_data['pooled_experimental']
        pooled_comp = forest_data['pooled_computational']
        pooled_uncertainty = forest_data['pooled_uncertainty']
        
        ax.axvline(pooled_exp, color='blue', linestyle='--', alpha=0.8, 
                  label=f'Pooled Experimental: {pooled_exp:.4f}')
        ax.axvline(pooled_comp, color='red', linestyle='--', alpha=0.8,
                  label=f'Pooled Computational: {pooled_comp:.4f}')
        
        # Formatting
        ax.set_yticks(y_positions)
        ax.set_yticklabels(studies)
        ax.set_xlabel(f'{property_name}')
        ax.set_title(f'Forest Plot: {property_name}\n'
                    f'Meta-analysis of {meta_analysis_result.n_studies} studies')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add heterogeneity information
        ax.text(0.02, 0.98, f'I² = {meta_analysis_result.heterogeneity_statistic:.1f}%',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_agreement_visualization(
        self,
        comparisons: List[ExperimentalComparison],
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Create visualization of agreement metrics across comparisons.
        
        Args:
            comparisons: List of experimental comparisons
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract data
        datasets = [comp.experimental_dataset for comp in comparisons]
        agreement_scores = [comp.agreement_metric for comp in comparisons]
        z_scores = [comp.z_score for comp in comparisons]
        comp_values = [comp.computational_value for comp in comparisons]
        exp_values = [comp.experimental_value for comp in comparisons]
        
        # Agreement scores bar plot
        ax1.bar(range(len(datasets)), agreement_scores, alpha=0.7)
        ax1.set_xticks(range(len(datasets)))
        ax1.set_xticklabels(datasets, rotation=45, ha='right')
        ax1.set_ylabel('Agreement Score')
        ax1.set_title('Agreement Scores by Dataset')
        ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Good Agreement')
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Poor Agreement')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Z-scores
        colors = ['green' if z < 2 else 'orange' if z < 3 else 'red' for z in z_scores]
        ax2.bar(range(len(datasets)), z_scores, color=colors, alpha=0.7)
        ax2.set_xticks(range(len(datasets)))
        ax2.set_xticklabels(datasets, rotation=45, ha='right')
        ax2.set_ylabel('Z-Score')
        ax2.set_title('Statistical Significance (Z-Scores)')
        ax2.axhline(y=2, color='orange', linestyle='--', alpha=0.7, label='p < 0.05')
        ax2.axhline(y=3, color='red', linestyle='--', alpha=0.7, label='p < 0.001')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Computational vs Experimental scatter
        ax3.scatter(exp_values, comp_values, alpha=0.7, s=60)
        min_val = min(min(exp_values), min(comp_values))
        max_val = max(max(exp_values), max(comp_values))
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect Agreement')
        ax3.set_xlabel('Experimental Values')
        ax3.set_ylabel('Computational Values')
        ax3.set_title('Computational vs Experimental Values')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Agreement score distribution
        ax4.hist(agreement_scores, bins=10, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Agreement Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Agreement Scores')
        ax4.axvline(x=np.mean(agreement_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(agreement_scores):.3f}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_funnel_plot(
        self,
        meta_analysis_result: MetaAnalysisResult,
        property_name: str,
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Create funnel plot for publication bias assessment.
        
        Args:
            meta_analysis_result: Meta-analysis results
            property_name: Name of the property being plotted
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        forest_data = meta_analysis_result.forest_plot_data
        exp_values = np.array(forest_data['experimental_values'])
        comp_values = np.array(forest_data['computational_values'])
        exp_uncertainties = np.array(forest_data['experimental_uncertainties'])
        
        # Calculate effect sizes (difference between computational and experimental)
        effect_sizes = comp_values - exp_values
        standard_errors = exp_uncertainties
        
        # Only plot if we have uncertainties
        valid_indices = standard_errors > 0
        if not np.any(valid_indices):
            ax.text(0.5, 0.5, 'No uncertainty data available for funnel plot',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'Funnel Plot: {property_name}')
            return fig
        
        effect_sizes = effect_sizes[valid_indices]
        standard_errors = standard_errors[valid_indices]
        
        # Create funnel plot
        ax.scatter(effect_sizes, 1/standard_errors, alpha=0.7, s=60)
        
        # Add funnel lines (pseudo 95% confidence limits)
        if len(effect_sizes) > 0:
            pooled_effect = np.mean(effect_sizes)
            se_range = np.linspace(np.min(1/standard_errors), np.max(1/standard_errors), 100)
            
            # Funnel lines
            upper_limit = pooled_effect + 1.96 / se_range
            lower_limit = pooled_effect - 1.96 / se_range
            
            ax.plot(upper_limit, se_range, 'r--', alpha=0.7, label='95% CI')
            ax.plot(lower_limit, se_range, 'r--', alpha=0.7)
            ax.axvline(pooled_effect, color='blue', linestyle='-', alpha=0.7, 
                      label=f'Pooled Effect: {pooled_effect:.4f}')
        
        ax.set_xlabel('Effect Size (Computational - Experimental)')
        ax.set_ylabel('Precision (1/Standard Error)')
        ax.set_title(f'Funnel Plot: {property_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add publication bias test results if available
        if meta_analysis_result.publication_bias_analysis:
            bias_analysis = meta_analysis_result.publication_bias_analysis
            if 'egger_p_value' in bias_analysis:
                p_val = bias_analysis['egger_p_value']
                bias_text = f"Egger's test p-value: {p_val:.4f}"
                if p_val < 0.05:
                    bias_text += "\n(Asymmetry detected)"
                ax.text(0.02, 0.98, bias_text, transform=ax.transAxes, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_comprehensive_visualization_report(
        self,
        validation_results: Dict[str, Any],
        output_dir: Optional[str] = None
    ) -> Dict[str, Figure]:
        """
        Generate comprehensive visualization report for experimental validation.
        
        Args:
            validation_results: Results from validate() method
            output_dir: Optional directory to save plots
            
        Returns:
            Dictionary of generated figures
        """
        figures = {}
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Overall agreement visualization
        all_comparisons = []
        for dataset_comps in validation_results['comparisons'].values():
            all_comparisons.extend(dataset_comps)
        
        if all_comparisons:
            save_path = str(output_path / 'agreement_overview.png') if output_dir else None
            figures['agreement_overview'] = self.create_agreement_visualization(
                all_comparisons, save_path
            )
        
        # Meta-analysis forest plots
        meta_analyses = validation_results.get('meta_analyses', {})
        for property_name, meta_result in meta_analyses.items():
            save_path = str(output_path / f'forest_plot_{property_name}.png') if output_dir else None
            figures[f'forest_plot_{property_name}'] = self.create_forest_plot(
                meta_result, property_name, save_path
            )
            
            # Funnel plot for publication bias
            save_path = str(output_path / f'funnel_plot_{property_name}.png') if output_dir else None
            figures[f'funnel_plot_{property_name}'] = self.create_funnel_plot(
                meta_result, property_name, save_path
            )
        
        return figures