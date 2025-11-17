"""
Critical Exponent Comparison Tables Generator

This module generates comprehensive tables showing theoretical vs measured exponents,
error percentages, confidence intervals, and statistical significance testing for
all physics systems analyzed by Prometheus.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from scipy import stats
from datetime import datetime
import json

from ..utils.logging_utils import get_logger


@dataclass
class CriticalExponentData:
    """Container for critical exponent data."""
    system_name: str
    dimensionality: str  # "2D" or "3D"
    model_type: str  # "Ising", "Potts", "XY"
    
    # Beta exponent (magnetization)
    beta_measured: Optional[float] = None
    beta_theoretical: Optional[float] = None
    beta_error_percent: Optional[float] = None
    beta_confidence_interval: Optional[Tuple[float, float]] = None
    beta_p_value: Optional[float] = None
    
    # Nu exponent (correlation length)
    nu_measured: Optional[float] = None
    nu_theoretical: Optional[float] = None
    nu_error_percent: Optional[float] = None
    nu_confidence_interval: Optional[Tuple[float, float]] = None
    nu_p_value: Optional[float] = None
    
    # Gamma exponent (susceptibility) - if available
    gamma_measured: Optional[float] = None
    gamma_theoretical: Optional[float] = None
    gamma_error_percent: Optional[float] = None
    gamma_confidence_interval: Optional[Tuple[float, float]] = None
    gamma_p_value: Optional[float] = None
    
    # Overall quality metrics
    overall_accuracy: Optional[float] = None
    universality_class_match: Optional[bool] = None
    statistical_significance: Optional[bool] = None


@dataclass
class ExponentComparisonSummary:
    """Summary of exponent comparison analysis."""
    total_systems: int
    systems_with_beta: int
    systems_with_nu: int
    systems_with_gamma: int
    
    # Accuracy statistics
    beta_accuracy_stats: Dict[str, float]
    nu_accuracy_stats: Dict[str, float]
    gamma_accuracy_stats: Dict[str, float]
    
    # Universality class validation
    universality_validation: Dict[str, bool]
    
    # Statistical significance summary
    significance_summary: Dict[str, Dict[str, Any]]


class CriticalExponentComparisonTables:
    """
    Comprehensive critical exponent comparison tables generator.
    
    Creates publication-ready tables with:
    - Theoretical vs measured exponents
    - Error percentages and confidence intervals
    - Statistical significance testing
    - Universality class validation
    """
    
    def __init__(self):
        """Initialize critical exponent comparison tables generator."""
        self.logger = get_logger(__name__)
        
        # Theoretical exponent values for different systems
        self.theoretical_exponents = {
            '2D_Ising': {'beta': 0.125, 'nu': 1.0, 'gamma': 1.75},
            '3D_Ising': {'beta': 0.326, 'nu': 0.630, 'gamma': 1.237},
            '2D_Potts_Q3': {'beta': 0.111, 'nu': 0.833, 'gamma': 1.556},  # Q=3 Potts
            '3D_Potts_Q3': {'beta': 0.354, 'nu': 0.679, 'gamma': 1.970},  # Q=3 Potts
            '2D_XY': {'beta': 0.231, 'nu': 1.0, 'gamma': 1.32}  # XY model (approximate)
        }
        
    def load_exponent_data(self, 
                          system_results: Dict[str, Dict[str, Any]]) -> List[CriticalExponentData]:
        """
        Load critical exponent data from system results.
        
        Args:
            system_results: Dictionary mapping system names to their results
            
        Returns:
            List of CriticalExponentData objects
        """
        self.logger.info("Loading critical exponent data for comparison tables")
        
        exponent_data = []
        
        for system_name, results in system_results.items():
            try:
                # Parse system information
                if '2d' in system_name.lower() or '2D' in system_name:
                    dimensionality = "2D"
                elif '3d' in system_name.lower() or '3D' in system_name:
                    dimensionality = "3D"
                else:
                    dimensionality = "Unknown"
                
                if 'ising' in system_name.lower():
                    model_type = "Ising"
                elif 'potts' in system_name.lower():
                    model_type = "Potts"
                elif 'xy' in system_name.lower():
                    model_type = "XY"
                else:
                    model_type = "Unknown"
                
                # Get theoretical values
                system_key = f"{dimensionality}_{model_type}"
                if 'potts' in system_name.lower():
                    system_key += "_Q3"  # Assume Q=3 Potts
                
                theoretical = self.theoretical_exponents.get(system_key, {})
                
                # Extract exponent data
                exponents = results.get('critical_exponents', {})
                
                # Beta exponent
                beta_measured = exponents.get('beta', {}).get('value')
                beta_theoretical = theoretical.get('beta')
                beta_error_percent = None
                beta_confidence_interval = exponents.get('beta', {}).get('confidence_interval')
                beta_p_value = exponents.get('beta', {}).get('p_value')
                
                if beta_measured is not None and beta_theoretical is not None:
                    beta_error_percent = abs(beta_measured - beta_theoretical) / beta_theoretical * 100
                
                # Nu exponent
                nu_measured = exponents.get('nu', {}).get('value')
                nu_theoretical = theoretical.get('nu')
                nu_error_percent = None
                nu_confidence_interval = exponents.get('nu', {}).get('confidence_interval')
                nu_p_value = exponents.get('nu', {}).get('p_value')
                
                if nu_measured is not None and nu_theoretical is not None:
                    nu_error_percent = abs(nu_measured - nu_theoretical) / nu_theoretical * 100
                
                # Gamma exponent
                gamma_measured = exponents.get('gamma', {}).get('value')
                gamma_theoretical = theoretical.get('gamma')
                gamma_error_percent = None
                gamma_confidence_interval = exponents.get('gamma', {}).get('confidence_interval')
                gamma_p_value = exponents.get('gamma', {}).get('p_value')
                
                if gamma_measured is not None and gamma_theoretical is not None:
                    gamma_error_percent = abs(gamma_measured - gamma_theoretical) / gamma_theoretical * 100
                
                # Calculate overall accuracy
                errors = []
                if beta_error_percent is not None:
                    errors.append(beta_error_percent)
                if nu_error_percent is not None:
                    errors.append(nu_error_percent)
                if gamma_error_percent is not None:
                    errors.append(gamma_error_percent)
                
                overall_accuracy = None
                if errors:
                    avg_error = np.mean(errors)
                    overall_accuracy = max(0.0, 100.0 - avg_error) / 100.0
                
                # Universality class match (all exponents within 20% of theoretical)
                universality_class_match = None
                if errors:
                    universality_class_match = all(error < 20.0 for error in errors)
                
                # Statistical significance (any p-value < 0.05)
                p_values = [p for p in [beta_p_value, nu_p_value, gamma_p_value] if p is not None]
                statistical_significance = None
                if p_values:
                    statistical_significance = any(p < 0.05 for p in p_values)
                
                exponent_data.append(CriticalExponentData(
                    system_name=system_name,
                    dimensionality=dimensionality,
                    model_type=model_type,
                    beta_measured=beta_measured,
                    beta_theoretical=beta_theoretical,
                    beta_error_percent=beta_error_percent,
                    beta_confidence_interval=beta_confidence_interval,
                    beta_p_value=beta_p_value,
                    nu_measured=nu_measured,
                    nu_theoretical=nu_theoretical,
                    nu_error_percent=nu_error_percent,
                    nu_confidence_interval=nu_confidence_interval,
                    nu_p_value=nu_p_value,
                    gamma_measured=gamma_measured,
                    gamma_theoretical=gamma_theoretical,
                    gamma_error_percent=gamma_error_percent,
                    gamma_confidence_interval=gamma_confidence_interval,
                    gamma_p_value=gamma_p_value,
                    overall_accuracy=overall_accuracy,
                    universality_class_match=universality_class_match,
                    statistical_significance=statistical_significance
                ))
                
            except Exception as e:
                self.logger.warning(f"Error loading exponent data for system {system_name}: {e}")
                continue
        
        self.logger.info(f"Loaded exponent data for {len(exponent_data)} systems")
        return exponent_data
    
    def create_comprehensive_exponent_table(self,
                                          exponent_data: List[CriticalExponentData]) -> pd.DataFrame:
        """
        Create comprehensive critical exponent comparison table.
        
        Args:
            exponent_data: List of critical exponent data
            
        Returns:
            Pandas DataFrame with comprehensive exponent comparison
        """
        self.logger.info("Creating comprehensive critical exponent table")
        
        table_data = []
        
        for data in exponent_data:
            row = {
                'System': f"{data.dimensionality} {data.model_type}",
                'Dimensionality': data.dimensionality,
                'Model': data.model_type,
            }
            
            # Beta exponent columns
            if data.beta_measured is not None:
                row['β (Measured)'] = f"{data.beta_measured:.4f}"
                row['β (Theoretical)'] = f"{data.beta_theoretical:.4f}" if data.beta_theoretical else "N/A"
                row['β Error (%)'] = f"{data.beta_error_percent:.1f}" if data.beta_error_percent else "N/A"
                
                if data.beta_confidence_interval:
                    ci_low, ci_high = data.beta_confidence_interval
                    row['β 95% CI'] = f"[{ci_low:.4f}, {ci_high:.4f}]"
                else:
                    row['β 95% CI'] = "N/A"
                
                row['β p-value'] = f"{data.beta_p_value:.4f}" if data.beta_p_value else "N/A"
            else:
                row.update({
                    'β (Measured)': "N/A",
                    'β (Theoretical)': f"{data.beta_theoretical:.4f}" if data.beta_theoretical else "N/A",
                    'β Error (%)': "N/A",
                    'β 95% CI': "N/A",
                    'β p-value': "N/A"
                })
            
            # Nu exponent columns
            if data.nu_measured is not None:
                row['ν (Measured)'] = f"{data.nu_measured:.4f}"
                row['ν (Theoretical)'] = f"{data.nu_theoretical:.4f}" if data.nu_theoretical else "N/A"
                row['ν Error (%)'] = f"{data.nu_error_percent:.1f}" if data.nu_error_percent else "N/A"
                
                if data.nu_confidence_interval:
                    ci_low, ci_high = data.nu_confidence_interval
                    row['ν 95% CI'] = f"[{ci_low:.4f}, {ci_high:.4f}]"
                else:
                    row['ν 95% CI'] = "N/A"
                
                row['ν p-value'] = f"{data.nu_p_value:.4f}" if data.nu_p_value else "N/A"
            else:
                row.update({
                    'ν (Measured)': "N/A",
                    'ν (Theoretical)': f"{data.nu_theoretical:.4f}" if data.nu_theoretical else "N/A",
                    'ν Error (%)': "N/A",
                    'ν 95% CI': "N/A",
                    'ν p-value': "N/A"
                })
            
            # Gamma exponent columns (if available)
            if data.gamma_measured is not None:
                row['γ (Measured)'] = f"{data.gamma_measured:.4f}"
                row['γ (Theoretical)'] = f"{data.gamma_theoretical:.4f}" if data.gamma_theoretical else "N/A"
                row['γ Error (%)'] = f"{data.gamma_error_percent:.1f}" if data.gamma_error_percent else "N/A"
                
                if data.gamma_confidence_interval:
                    ci_low, ci_high = data.gamma_confidence_interval
                    row['γ 95% CI'] = f"[{ci_low:.4f}, {ci_high:.4f}]"
                else:
                    row['γ 95% CI'] = "N/A"
                
                row['γ p-value'] = f"{data.gamma_p_value:.4f}" if data.gamma_p_value else "N/A"
            else:
                row.update({
                    'γ (Measured)': "N/A",
                    'γ (Theoretical)': f"{data.gamma_theoretical:.4f}" if data.gamma_theoretical else "N/A",
                    'γ Error (%)': "N/A",
                    'γ 95% CI': "N/A",
                    'γ p-value': "N/A"
                })
            
            # Overall metrics
            row['Overall Accuracy'] = f"{data.overall_accuracy:.3f}" if data.overall_accuracy else "N/A"
            row['Universality Match'] = "✓" if data.universality_class_match else "✗" if data.universality_class_match is not None else "N/A"
            row['Statistically Significant'] = "✓" if data.statistical_significance else "✗" if data.statistical_significance is not None else "N/A"
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        return df
    
    def create_summary_statistics_table(self,
                                      exponent_data: List[CriticalExponentData]) -> pd.DataFrame:
        """
        Create summary statistics table for critical exponents.
        
        Args:
            exponent_data: List of critical exponent data
            
        Returns:
            Pandas DataFrame with summary statistics
        """
        self.logger.info("Creating summary statistics table")
        
        # Collect data for statistics
        beta_errors = [d.beta_error_percent for d in exponent_data if d.beta_error_percent is not None]
        nu_errors = [d.nu_error_percent for d in exponent_data if d.nu_error_percent is not None]
        gamma_errors = [d.gamma_error_percent for d in exponent_data if d.gamma_error_percent is not None]
        
        overall_accuracies = [d.overall_accuracy for d in exponent_data if d.overall_accuracy is not None]
        
        # Calculate statistics
        stats_data = []
        
        # Beta exponent statistics
        if beta_errors:
            stats_data.append({
                'Exponent': 'β (Magnetization)',
                'N Systems': len(beta_errors),
                'Mean Error (%)': f"{np.mean(beta_errors):.2f}",
                'Std Error (%)': f"{np.std(beta_errors):.2f}",
                'Min Error (%)': f"{np.min(beta_errors):.2f}",
                'Max Error (%)': f"{np.max(beta_errors):.2f}",
                'Systems < 10% Error': f"{sum(1 for e in beta_errors if e < 10.0)}/{len(beta_errors)}",
                'Systems < 20% Error': f"{sum(1 for e in beta_errors if e < 20.0)}/{len(beta_errors)}"
            })
        
        # Nu exponent statistics
        if nu_errors:
            stats_data.append({
                'Exponent': 'ν (Correlation Length)',
                'N Systems': len(nu_errors),
                'Mean Error (%)': f"{np.mean(nu_errors):.2f}",
                'Std Error (%)': f"{np.std(nu_errors):.2f}",
                'Min Error (%)': f"{np.min(nu_errors):.2f}",
                'Max Error (%)': f"{np.max(nu_errors):.2f}",
                'Systems < 10% Error': f"{sum(1 for e in nu_errors if e < 10.0)}/{len(nu_errors)}",
                'Systems < 20% Error': f"{sum(1 for e in nu_errors if e < 20.0)}/{len(nu_errors)}"
            })
        
        # Gamma exponent statistics
        if gamma_errors:
            stats_data.append({
                'Exponent': 'γ (Susceptibility)',
                'N Systems': len(gamma_errors),
                'Mean Error (%)': f"{np.mean(gamma_errors):.2f}",
                'Std Error (%)': f"{np.std(gamma_errors):.2f}",
                'Min Error (%)': f"{np.min(gamma_errors):.2f}",
                'Max Error (%)': f"{np.max(gamma_errors):.2f}",
                'Systems < 10% Error': f"{sum(1 for e in gamma_errors if e < 10.0)}/{len(gamma_errors)}",
                'Systems < 20% Error': f"{sum(1 for e in gamma_errors if e < 20.0)}/{len(gamma_errors)}"
            })
        
        # Overall statistics
        if overall_accuracies:
            # Convert accuracies to errors for consistency
            overall_errors = [(1 - acc) * 100 for acc in overall_accuracies]
            stats_data.append({
                'Exponent': 'Overall (All Exponents)',
                'N Systems': len(overall_errors),
                'Mean Error (%)': f"{np.mean(overall_errors):.2f}",
                'Std Error (%)': f"{np.std(overall_errors):.2f}",
                'Min Error (%)': f"{np.min(overall_errors):.2f}",
                'Max Error (%)': f"{np.max(overall_errors):.2f}",
                'Systems < 10% Error': f"{sum(1 for e in overall_errors if e < 10.0)}/{len(overall_errors)}",
                'Systems < 20% Error': f"{sum(1 for e in overall_errors if e < 20.0)}/{len(overall_errors)}"
            })
        
        df = pd.DataFrame(stats_data)
        return df
    
    def create_universality_class_table(self,
                                      exponent_data: List[CriticalExponentData]) -> pd.DataFrame:
        """
        Create universality class validation table.
        
        Args:
            exponent_data: List of critical exponent data
            
        Returns:
            Pandas DataFrame with universality class analysis
        """
        self.logger.info("Creating universality class validation table")
        
        # Group by universality class
        universality_classes = {}
        
        for data in exponent_data:
            class_key = f"{data.dimensionality}_{data.model_type}"
            if class_key not in universality_classes:
                universality_classes[class_key] = []
            universality_classes[class_key].append(data)
        
        table_data = []
        
        for class_name, systems in universality_classes.items():
            # Get theoretical values
            theoretical = self.theoretical_exponents.get(class_name, {})
            if not theoretical and 'Potts' in class_name:
                theoretical = self.theoretical_exponents.get(f"{class_name}_Q3", {})
            
            # Calculate class statistics
            beta_errors = [s.beta_error_percent for s in systems if s.beta_error_percent is not None]
            nu_errors = [s.nu_error_percent for s in systems if s.nu_error_percent is not None]
            gamma_errors = [s.gamma_error_percent for s in systems if s.gamma_error_percent is not None]
            
            universality_matches = [s.universality_class_match for s in systems if s.universality_class_match is not None]
            
            row = {
                'Universality Class': class_name.replace('_', ' '),
                'N Systems': len(systems),
                'β Theoretical': f"{theoretical.get('beta', 'N/A'):.4f}" if theoretical.get('beta') else "N/A",
                'ν Theoretical': f"{theoretical.get('nu', 'N/A'):.4f}" if theoretical.get('nu') else "N/A",
                'γ Theoretical': f"{theoretical.get('gamma', 'N/A'):.4f}" if theoretical.get('gamma') else "N/A",
            }
            
            # Beta statistics
            if beta_errors:
                row['β Mean Error (%)'] = f"{np.mean(beta_errors):.1f}"
                row['β Systems < 20%'] = f"{sum(1 for e in beta_errors if e < 20.0)}/{len(beta_errors)}"
            else:
                row['β Mean Error (%)'] = "N/A"
                row['β Systems < 20%'] = "N/A"
            
            # Nu statistics
            if nu_errors:
                row['ν Mean Error (%)'] = f"{np.mean(nu_errors):.1f}"
                row['ν Systems < 20%'] = f"{sum(1 for e in nu_errors if e < 20.0)}/{len(nu_errors)}"
            else:
                row['ν Mean Error (%)'] = "N/A"
                row['ν Systems < 20%'] = "N/A"
            
            # Gamma statistics
            if gamma_errors:
                row['γ Mean Error (%)'] = f"{np.mean(gamma_errors):.1f}"
                row['γ Systems < 20%'] = f"{sum(1 for e in gamma_errors if e < 20.0)}/{len(gamma_errors)}"
            else:
                row['γ Mean Error (%)'] = "N/A"
                row['γ Systems < 20%'] = "N/A"
            
            # Universality validation
            if universality_matches:
                match_rate = sum(universality_matches) / len(universality_matches)
                row['Universality Match Rate'] = f"{match_rate:.2f}"
                row['Class Validation'] = "PASS" if match_rate >= 0.8 else "PARTIAL" if match_rate >= 0.5 else "FAIL"
            else:
                row['Universality Match Rate'] = "N/A"
                row['Class Validation'] = "N/A"
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        return df
    
    def perform_statistical_significance_testing(self,
                                               exponent_data: List[CriticalExponentData]) -> Dict[str, Any]:
        """
        Perform statistical significance testing for critical exponents.
        
        Args:
            exponent_data: List of critical exponent data
            
        Returns:
            Dictionary with statistical test results
        """
        self.logger.info("Performing statistical significance testing")
        
        results = {}
        
        # Collect measured vs theoretical values
        beta_measured = [d.beta_measured for d in exponent_data if d.beta_measured is not None and d.beta_theoretical is not None]
        beta_theoretical = [d.beta_theoretical for d in exponent_data if d.beta_measured is not None and d.beta_theoretical is not None]
        
        nu_measured = [d.nu_measured for d in exponent_data if d.nu_measured is not None and d.nu_theoretical is not None]
        nu_theoretical = [d.nu_theoretical for d in exponent_data if d.nu_measured is not None and d.nu_theoretical is not None]
        
        gamma_measured = [d.gamma_measured for d in exponent_data if d.gamma_measured is not None and d.gamma_theoretical is not None]
        gamma_theoretical = [d.gamma_theoretical for d in exponent_data if d.gamma_measured is not None and d.gamma_theoretical is not None]
        
        # Beta exponent tests
        if len(beta_measured) > 1:
            # Paired t-test: measured vs theoretical
            t_stat, p_value = stats.ttest_rel(beta_measured, beta_theoretical)
            
            # Correlation test
            corr_coeff, corr_p = stats.pearsonr(beta_measured, beta_theoretical)
            
            # Wilcoxon signed-rank test (non-parametric)
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(beta_measured, beta_theoretical)
            
            results['beta'] = {
                'n_systems': len(beta_measured),
                'paired_t_test': {'statistic': t_stat, 'p_value': p_value},
                'correlation': {'coefficient': corr_coeff, 'p_value': corr_p},
                'wilcoxon_test': {'statistic': wilcoxon_stat, 'p_value': wilcoxon_p},
                'mean_measured': np.mean(beta_measured),
                'mean_theoretical': np.mean(beta_theoretical),
                'mean_difference': np.mean(np.array(beta_measured) - np.array(beta_theoretical))
            }
        
        # Nu exponent tests
        if len(nu_measured) > 1:
            t_stat, p_value = stats.ttest_rel(nu_measured, nu_theoretical)
            corr_coeff, corr_p = stats.pearsonr(nu_measured, nu_theoretical)
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(nu_measured, nu_theoretical)
            
            results['nu'] = {
                'n_systems': len(nu_measured),
                'paired_t_test': {'statistic': t_stat, 'p_value': p_value},
                'correlation': {'coefficient': corr_coeff, 'p_value': corr_p},
                'wilcoxon_test': {'statistic': wilcoxon_stat, 'p_value': wilcoxon_p},
                'mean_measured': np.mean(nu_measured),
                'mean_theoretical': np.mean(nu_theoretical),
                'mean_difference': np.mean(np.array(nu_measured) - np.array(nu_theoretical))
            }
        
        # Gamma exponent tests
        if len(gamma_measured) > 1:
            t_stat, p_value = stats.ttest_rel(gamma_measured, gamma_theoretical)
            corr_coeff, corr_p = stats.pearsonr(gamma_measured, gamma_theoretical)
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(gamma_measured, gamma_theoretical)
            
            results['gamma'] = {
                'n_systems': len(gamma_measured),
                'paired_t_test': {'statistic': t_stat, 'p_value': p_value},
                'correlation': {'coefficient': corr_coeff, 'p_value': corr_p},
                'wilcoxon_test': {'statistic': wilcoxon_stat, 'p_value': wilcoxon_p},
                'mean_measured': np.mean(gamma_measured),
                'mean_theoretical': np.mean(gamma_theoretical),
                'mean_difference': np.mean(np.array(gamma_measured) - np.array(gamma_theoretical))
            }
        
        # Overall accuracy test
        overall_accuracies = [d.overall_accuracy for d in exponent_data if d.overall_accuracy is not None]
        if overall_accuracies:
            # Test if overall accuracy is significantly different from random (0.5)
            t_stat, p_value = stats.ttest_1samp(overall_accuracies, 0.5)
            
            # Test if accuracy is significantly above threshold (0.8)
            t_stat_thresh, p_value_thresh = stats.ttest_1samp(overall_accuracies, 0.8)
            
            results['overall'] = {
                'n_systems': len(overall_accuracies),
                'vs_random': {'statistic': t_stat, 'p_value': p_value},
                'vs_threshold': {'statistic': t_stat_thresh, 'p_value': p_value_thresh},
                'mean_accuracy': np.mean(overall_accuracies),
                'std_accuracy': np.std(overall_accuracies)
            }
        
        return results   
 
    def create_statistical_significance_table(self,
                                            significance_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Create statistical significance testing table.
        
        Args:
            significance_results: Results from statistical significance testing
            
        Returns:
            Pandas DataFrame with significance test results
        """
        self.logger.info("Creating statistical significance table")
        
        table_data = []
        
        for exponent, results in significance_results.items():
            if exponent == 'overall':
                continue  # Handle separately
            
            row = {
                'Exponent': exponent.upper(),
                'N Systems': results['n_systems'],
                'Mean Measured': f"{results['mean_measured']:.4f}",
                'Mean Theoretical': f"{results['mean_theoretical']:.4f}",
                'Mean Difference': f"{results['mean_difference']:.4f}",
            }
            
            # Paired t-test results
            t_test = results['paired_t_test']
            row['t-statistic'] = f"{t_test['statistic']:.3f}"
            row['t-test p-value'] = f"{t_test['p_value']:.4f}"
            row['t-test Significant'] = "✓" if t_test['p_value'] < 0.05 else "✗"
            
            # Correlation results
            corr = results['correlation']
            row['Correlation'] = f"{corr['coefficient']:.4f}"
            row['Corr p-value'] = f"{corr['p_value']:.4f}"
            row['Corr Significant'] = "✓" if corr['p_value'] < 0.05 else "✗"
            
            # Wilcoxon test results
            wilcoxon = results['wilcoxon_test']
            row['Wilcoxon p-value'] = f"{wilcoxon['p_value']:.4f}"
            row['Wilcoxon Significant'] = "✓" if wilcoxon['p_value'] < 0.05 else "✗"
            
            table_data.append(row)
        
        # Add overall results if available
        if 'overall' in significance_results:
            overall = significance_results['overall']
            row = {
                'Exponent': 'OVERALL',
                'N Systems': overall['n_systems'],
                'Mean Measured': f"{overall['mean_accuracy']:.4f}",
                'Mean Theoretical': "0.800",  # Target accuracy
                'Mean Difference': f"{overall['mean_accuracy'] - 0.8:.4f}",
                't-statistic': f"{overall['vs_threshold']['statistic']:.3f}",
                't-test p-value': f"{overall['vs_threshold']['p_value']:.4f}",
                't-test Significant': "✓" if overall['vs_threshold']['p_value'] < 0.05 else "✗",
                'Correlation': "N/A",
                'Corr p-value': "N/A",
                'Corr Significant': "N/A",
                'Wilcoxon p-value': "N/A",
                'Wilcoxon Significant': "N/A"
            }
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        return df
    
    def create_latex_tables(self,
                          comprehensive_table: pd.DataFrame,
                          summary_table: pd.DataFrame,
                          universality_table: pd.DataFrame,
                          significance_table: pd.DataFrame) -> Dict[str, str]:
        """
        Create LaTeX-formatted tables for publication.
        
        Args:
            comprehensive_table: Comprehensive exponent comparison table
            summary_table: Summary statistics table
            universality_table: Universality class validation table
            significance_table: Statistical significance table
            
        Returns:
            Dictionary mapping table names to LaTeX strings
        """
        self.logger.info("Creating LaTeX-formatted tables")
        
        latex_tables = {}
        
        # Comprehensive table (main results)
        latex_tables['comprehensive'] = self._create_latex_comprehensive_table(comprehensive_table)
        
        # Summary statistics table
        latex_tables['summary'] = self._create_latex_summary_table(summary_table)
        
        # Universality class table
        latex_tables['universality'] = self._create_latex_universality_table(universality_table)
        
        # Statistical significance table
        latex_tables['significance'] = self._create_latex_significance_table(significance_table)
        
        return latex_tables
    
    def _create_latex_comprehensive_table(self, df: pd.DataFrame) -> str:
        """Create LaTeX version of comprehensive table."""
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Critical Exponent Comparison: Theoretical vs Measured Values}\n"
        latex += "\\label{tab:critical_exponents}\n"
        latex += "\\begin{tabular}{|l|c|c|c|c|c|c|c|c|}\n"
        latex += "\\hline\n"
        latex += "System & $\\beta_{th}$ & $\\beta_{meas}$ & Error (\\%) & $\\nu_{th}$ & $\\nu_{meas}$ & Error (\\%) & Accuracy & Match \\\\\n"
        latex += "\\hline\n"
        
        for _, row in df.iterrows():
            system = row['System'].replace('_', '\\_')
            beta_th = row['β (Theoretical)']
            beta_meas = row['β (Measured)']
            beta_err = row['β Error (%)']
            nu_th = row['ν (Theoretical)']
            nu_meas = row['ν (Measured)']
            nu_err = row['ν Error (%)']
            accuracy = row['Overall Accuracy']
            match = row['Universality Match']
            
            latex += f"{system} & {beta_th} & {beta_meas} & {beta_err} & {nu_th} & {nu_meas} & {nu_err} & {accuracy} & {match} \\\\\n"
        
        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def _create_latex_summary_table(self, df: pd.DataFrame) -> str:
        """Create LaTeX version of summary statistics table."""
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Critical Exponent Accuracy Summary Statistics}\n"
        latex += "\\label{tab:exponent_summary}\n"
        latex += "\\begin{tabular}{|l|c|c|c|c|c|}\n"
        latex += "\\hline\n"
        latex += "Exponent & N Systems & Mean Error (\\%) & Std Error (\\%) & $<$10\\% Error & $<$20\\% Error \\\\\n"
        latex += "\\hline\n"
        
        for _, row in df.iterrows():
            exponent = row['Exponent']
            n_systems = row['N Systems']
            mean_err = row['Mean Error (%)']
            std_err = row['Std Error (%)']
            under_10 = row['Systems < 10% Error']
            under_20 = row['Systems < 20% Error']
            
            latex += f"{exponent} & {n_systems} & {mean_err} & {std_err} & {under_10} & {under_20} \\\\\n"
        
        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def _create_latex_universality_table(self, df: pd.DataFrame) -> str:
        """Create LaTeX version of universality class table."""
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Universality Class Validation}\n"
        latex += "\\label{tab:universality}\n"
        latex += "\\begin{tabular}{|l|c|c|c|c|c|c|}\n"
        latex += "\\hline\n"
        latex += "Class & N & $\\beta$ Error (\\%) & $\\nu$ Error (\\%) & Match Rate & Validation \\\\\n"
        latex += "\\hline\n"
        
        for _, row in df.iterrows():
            class_name = row['Universality Class'].replace('_', '\\_')
            n_systems = row['N Systems']
            beta_err = row['β Mean Error (%)']
            nu_err = row['ν Mean Error (%)']
            match_rate = row['Universality Match Rate']
            validation = row['Class Validation']
            
            latex += f"{class_name} & {n_systems} & {beta_err} & {nu_err} & {match_rate} & {validation} \\\\\n"
        
        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def _create_latex_significance_table(self, df: pd.DataFrame) -> str:
        """Create LaTeX version of statistical significance table."""
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Statistical Significance Testing Results}\n"
        latex += "\\label{tab:significance}\n"
        latex += "\\begin{tabular}{|l|c|c|c|c|c|}\n"
        latex += "\\hline\n"
        latex += "Exponent & N & Mean Diff & t-test p & Correlation & Wilcoxon p \\\\\n"
        latex += "\\hline\n"
        
        for _, row in df.iterrows():
            exponent = row['Exponent']
            n_systems = row['N Systems']
            mean_diff = row['Mean Difference']
            t_p = row['t-test p-value']
            correlation = row['Correlation']
            wilcoxon_p = row['Wilcoxon p-value']
            
            latex += f"{exponent} & {n_systems} & {mean_diff} & {t_p} & {correlation} & {wilcoxon_p} \\\\\n"
        
        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def create_exponent_comparison_plots(self,
                                       exponent_data: List[CriticalExponentData],
                                       figsize: Tuple[int, int] = (15, 12)) -> Figure:
        """
        Create comprehensive critical exponent comparison plots.
        
        Args:
            exponent_data: List of critical exponent data
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure with exponent comparison plots
        """
        self.logger.info("Creating critical exponent comparison plots")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: Beta exponent comparison
        ax1 = axes[0, 0]
        
        beta_measured = [d.beta_measured for d in exponent_data if d.beta_measured is not None]
        beta_theoretical = [d.beta_theoretical for d in exponent_data if d.beta_measured is not None and d.beta_theoretical is not None]
        system_names = [f"{d.dimensionality} {d.model_type}" for d in exponent_data if d.beta_measured is not None]
        
        if beta_measured and beta_theoretical:
            # Perfect correlation line
            min_val = min(min(beta_measured), min(beta_theoretical))
            max_val = max(max(beta_measured), max(beta_theoretical))
            ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Perfect Agreement')
            
            # Scatter plot
            colors = plt.cm.Set1(np.linspace(0, 1, len(beta_measured)))
            scatter = ax1.scatter(beta_theoretical, beta_measured, c=colors, s=100, alpha=0.7, edgecolors='black')
            
            # Add system labels
            for i, name in enumerate(system_names):
                ax1.annotate(name, (beta_theoretical[i], beta_measured[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax1.set_xlabel('Theoretical β', fontsize=12)
            ax1.set_ylabel('Measured β', fontsize=12)
            ax1.set_title('β Exponent: Measured vs Theoretical', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Nu exponent comparison
        ax2 = axes[0, 1]
        
        nu_measured = [d.nu_measured for d in exponent_data if d.nu_measured is not None]
        nu_theoretical = [d.nu_theoretical for d in exponent_data if d.nu_measured is not None and d.nu_theoretical is not None]
        nu_system_names = [f"{d.dimensionality} {d.model_type}" for d in exponent_data if d.nu_measured is not None]
        
        if nu_measured and nu_theoretical:
            min_val = min(min(nu_measured), min(nu_theoretical))
            max_val = max(max(nu_measured), max(nu_theoretical))
            ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Perfect Agreement')
            
            colors = plt.cm.Set2(np.linspace(0, 1, len(nu_measured)))
            scatter = ax2.scatter(nu_theoretical, nu_measured, c=colors, s=100, alpha=0.7, edgecolors='black')
            
            for i, name in enumerate(nu_system_names):
                ax2.annotate(name, (nu_theoretical[i], nu_measured[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax2.set_xlabel('Theoretical ν', fontsize=12)
            ax2.set_ylabel('Measured ν', fontsize=12)
            ax2.set_title('ν Exponent: Measured vs Theoretical', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Error distribution
        ax3 = axes[0, 2]
        
        beta_errors = [d.beta_error_percent for d in exponent_data if d.beta_error_percent is not None]
        nu_errors = [d.nu_error_percent for d in exponent_data if d.nu_error_percent is not None]
        
        if beta_errors or nu_errors:
            bins = np.linspace(0, 50, 20)
            
            if beta_errors:
                ax3.hist(beta_errors, bins=bins, alpha=0.7, label='β Exponent', color='blue', density=True)
            if nu_errors:
                ax3.hist(nu_errors, bins=bins, alpha=0.7, label='ν Exponent', color='red', density=True)
            
            ax3.axvline(10, color='orange', linestyle='--', alpha=0.7, label='10% Error')
            ax3.axvline(20, color='red', linestyle='--', alpha=0.7, label='20% Error')
            
            ax3.set_xlabel('Error (%)', fontsize=12)
            ax3.set_ylabel('Density', fontsize=12)
            ax3.set_title('Error Distribution', fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Accuracy by system
        ax4 = axes[1, 0]
        
        systems = [f"{d.dimensionality}\n{d.model_type}" for d in exponent_data if d.overall_accuracy is not None]
        accuracies = [d.overall_accuracy for d in exponent_data if d.overall_accuracy is not None]
        
        if systems and accuracies:
            colors = ['green' if acc > 0.8 else 'orange' if acc > 0.6 else 'red' for acc in accuracies]
            bars = ax4.bar(range(len(systems)), accuracies, color=colors, alpha=0.7)
            
            ax4.axhline(0.8, color='red', linestyle='--', alpha=0.7, label='Target (80%)')
            ax4.set_xlabel('System', fontsize=12)
            ax4.set_ylabel('Overall Accuracy', fontsize=12)
            ax4.set_title('Overall Accuracy by System', fontsize=14, fontweight='bold')
            ax4.set_xticks(range(len(systems)))
            ax4.set_xticklabels(systems)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Add value annotations
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax4.annotate(f'{acc:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
        
        # Plot 5: Universality class validation
        ax5 = axes[1, 1]
        
        # Group by universality class
        universality_classes = {}
        for data in exponent_data:
            class_key = f"{data.dimensionality}_{data.model_type}"
            if class_key not in universality_classes:
                universality_classes[class_key] = []
            universality_classes[class_key].append(data)
        
        class_names = []
        match_rates = []
        
        for class_name, systems in universality_classes.items():
            matches = [s.universality_class_match for s in systems if s.universality_class_match is not None]
            if matches:
                match_rate = sum(matches) / len(matches)
                class_names.append(class_name.replace('_', '\n'))
                match_rates.append(match_rate)
        
        if class_names and match_rates:
            colors = ['green' if rate >= 0.8 else 'orange' if rate >= 0.5 else 'red' for rate in match_rates]
            bars = ax5.bar(range(len(class_names)), match_rates, color=colors, alpha=0.7)
            
            ax5.axhline(0.8, color='red', linestyle='--', alpha=0.7, label='Target (80%)')
            ax5.set_xlabel('Universality Class', fontsize=12)
            ax5.set_ylabel('Match Rate', fontsize=12)
            ax5.set_title('Universality Class Validation', fontsize=14, fontweight='bold')
            ax5.set_xticks(range(len(class_names)))
            ax5.set_xticklabels(class_names)
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # Add value annotations
            for bar, rate in zip(bars, match_rates):
                height = bar.get_height()
                ax5.annotate(f'{rate:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
        
        # Plot 6: Confidence intervals
        ax6 = axes[1, 2]
        
        # Show confidence intervals for available exponents
        systems_with_ci = []
        beta_cis = []
        nu_cis = []
        
        for data in exponent_data:
            if data.beta_confidence_interval or data.nu_confidence_interval:
                systems_with_ci.append(f"{data.dimensionality}\n{data.model_type}")
                
                if data.beta_confidence_interval:
                    ci_low, ci_high = data.beta_confidence_interval
                    beta_cis.append((data.beta_measured, ci_low, ci_high))
                else:
                    beta_cis.append(None)
                
                if data.nu_confidence_interval:
                    ci_low, ci_high = data.nu_confidence_interval
                    nu_cis.append((data.nu_measured, ci_low, ci_high))
                else:
                    nu_cis.append(None)
        
        if systems_with_ci:
            x_pos = np.arange(len(systems_with_ci))
            
            # Plot beta confidence intervals
            for i, ci_data in enumerate(beta_cis):
                if ci_data:
                    measured, ci_low, ci_high = ci_data
                    ax6.errorbar(i - 0.1, measured, yerr=[[measured - ci_low], [ci_high - measured]], 
                               fmt='bo', capsize=5, label='β' if i == 0 else "")
            
            # Plot nu confidence intervals
            for i, ci_data in enumerate(nu_cis):
                if ci_data:
                    measured, ci_low, ci_high = ci_data
                    ax6.errorbar(i + 0.1, measured, yerr=[[measured - ci_low], [ci_high - measured]], 
                               fmt='ro', capsize=5, label='ν' if i == 0 else "")
            
            ax6.set_xlabel('System', fontsize=12)
            ax6.set_ylabel('Exponent Value', fontsize=12)
            ax6.set_title('95% Confidence Intervals', fontsize=14, fontweight='bold')
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels(systems_with_ci)
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Critical Exponent Comparison Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def generate_comprehensive_exponent_report(self,
                                             exponent_data: List[CriticalExponentData],
                                             output_dir: str = 'results/publication/exponent_tables') -> Dict[str, str]:
        """
        Generate comprehensive critical exponent comparison report.
        
        Args:
            exponent_data: List of critical exponent data
            output_dir: Output directory for results
            
        Returns:
            Dictionary mapping output names to file paths
        """
        self.logger.info("Generating comprehensive critical exponent report")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_outputs = {}
        
        try:
            # Create all tables
            comprehensive_table = self.create_comprehensive_exponent_table(exponent_data)
            summary_table = self.create_summary_statistics_table(exponent_data)
            universality_table = self.create_universality_class_table(exponent_data)
            
            # Perform statistical testing
            significance_results = self.perform_statistical_significance_testing(exponent_data)
            significance_table = self.create_statistical_significance_table(significance_results)
            
            # Save CSV tables
            comprehensive_table.to_csv(output_path / "comprehensive_exponent_table.csv", index=False)
            summary_table.to_csv(output_path / "summary_statistics_table.csv", index=False)
            universality_table.to_csv(output_path / "universality_class_table.csv", index=False)
            significance_table.to_csv(output_path / "statistical_significance_table.csv", index=False)
            
            saved_outputs['comprehensive_csv'] = str(output_path / "comprehensive_exponent_table.csv")
            saved_outputs['summary_csv'] = str(output_path / "summary_statistics_table.csv")
            saved_outputs['universality_csv'] = str(output_path / "universality_class_table.csv")
            saved_outputs['significance_csv'] = str(output_path / "statistical_significance_table.csv")
            
            # Create LaTeX tables
            latex_tables = self.create_latex_tables(
                comprehensive_table, summary_table, universality_table, significance_table
            )
            
            # Save LaTeX tables
            for table_name, latex_content in latex_tables.items():
                latex_file = output_path / f"{table_name}_table.tex"
                with open(latex_file, 'w') as f:
                    f.write(latex_content)
                saved_outputs[f'{table_name}_latex'] = str(latex_file)
            
            # Create comparison plots
            comparison_fig = self.create_exponent_comparison_plots(exponent_data)
            plot_path = output_path / "critical_exponent_comparison_plots.png"
            comparison_fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            comparison_fig.savefig(output_path / "critical_exponent_comparison_plots.pdf", 
                                 format='pdf', bbox_inches='tight')
            plt.close(comparison_fig)
            saved_outputs['comparison_plots'] = str(plot_path)
            
            # Save statistical results
            results_file = output_path / "statistical_significance_results.json"
            with open(results_file, 'w') as f:
                json.dump(significance_results, f, indent=2, default=str)
            saved_outputs['significance_results'] = str(results_file)
            
            # Generate summary report
            summary_report = self._generate_exponent_summary_report(
                exponent_data, significance_results, output_path
            )
            saved_outputs['summary_report'] = summary_report
            
            self.logger.info(f"Critical exponent report generated in {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating critical exponent report: {e}")
            raise
        
        return saved_outputs
    
    def _generate_exponent_summary_report(self,
                                        exponent_data: List[CriticalExponentData],
                                        significance_results: Dict[str, Any],
                                        output_path: Path) -> str:
        """Generate comprehensive summary report for critical exponents."""
        summary_file = output_path / "critical_exponent_summary_report.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Critical Exponent Comparison Analysis - Summary Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total systems analyzed: {len(exponent_data)}\n\n")
            
            # Beta exponent analysis
            beta_data = [d for d in exponent_data if d.beta_measured is not None]
            if beta_data:
                f.write("β Exponent (Magnetization) Analysis:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Systems with β measurements: {len(beta_data)}\n")
                
                beta_errors = [d.beta_error_percent for d in beta_data if d.beta_error_percent is not None]
                if beta_errors:
                    f.write(f"  Mean error: {np.mean(beta_errors):.2f}%\n")
                    f.write(f"  Standard deviation: {np.std(beta_errors):.2f}%\n")
                    f.write(f"  Best accuracy: {np.min(beta_errors):.2f}%\n")
                    f.write(f"  Worst accuracy: {np.max(beta_errors):.2f}%\n")
                    f.write(f"  Systems < 10% error: {sum(1 for e in beta_errors if e < 10.0)}/{len(beta_errors)}\n")
                    f.write(f"  Systems < 20% error: {sum(1 for e in beta_errors if e < 20.0)}/{len(beta_errors)}\n")
                
                if 'beta' in significance_results:
                    beta_sig = significance_results['beta']
                    f.write(f"  Statistical significance (t-test): p = {beta_sig['paired_t_test']['p_value']:.4f}\n")
                    f.write(f"  Correlation with theory: r = {beta_sig['correlation']['coefficient']:.4f}\n")
                
                f.write("\n")
            
            # Nu exponent analysis
            nu_data = [d for d in exponent_data if d.nu_measured is not None]
            if nu_data:
                f.write("ν Exponent (Correlation Length) Analysis:\n")
                f.write("-" * 45 + "\n")
                f.write(f"  Systems with ν measurements: {len(nu_data)}\n")
                
                nu_errors = [d.nu_error_percent for d in nu_data if d.nu_error_percent is not None]
                if nu_errors:
                    f.write(f"  Mean error: {np.mean(nu_errors):.2f}%\n")
                    f.write(f"  Standard deviation: {np.std(nu_errors):.2f}%\n")
                    f.write(f"  Best accuracy: {np.min(nu_errors):.2f}%\n")
                    f.write(f"  Worst accuracy: {np.max(nu_errors):.2f}%\n")
                    f.write(f"  Systems < 10% error: {sum(1 for e in nu_errors if e < 10.0)}/{len(nu_errors)}\n")
                    f.write(f"  Systems < 20% error: {sum(1 for e in nu_errors if e < 20.0)}/{len(nu_errors)}\n")
                
                if 'nu' in significance_results:
                    nu_sig = significance_results['nu']
                    f.write(f"  Statistical significance (t-test): p = {nu_sig['paired_t_test']['p_value']:.4f}\n")
                    f.write(f"  Correlation with theory: r = {nu_sig['correlation']['coefficient']:.4f}\n")
                
                f.write("\n")
            
            # Universality class validation
            f.write("Universality Class Validation:\n")
            f.write("-" * 30 + "\n")
            
            universality_classes = {}
            for data in exponent_data:
                class_key = f"{data.dimensionality}_{data.model_type}"
                if class_key not in universality_classes:
                    universality_classes[class_key] = []
                universality_classes[class_key].append(data)
            
            for class_name, systems in universality_classes.items():
                matches = [s.universality_class_match for s in systems if s.universality_class_match is not None]
                if matches:
                    match_rate = sum(matches) / len(matches)
                    f.write(f"  {class_name.replace('_', ' ')}: {match_rate:.2f} match rate ({sum(matches)}/{len(matches)} systems)\n")
            
            f.write("\n")
            
            # Overall assessment
            f.write("Overall Assessment:\n")
            f.write("-" * 18 + "\n")
            
            all_errors = []
            if beta_data:
                all_errors.extend([d.beta_error_percent for d in beta_data if d.beta_error_percent is not None])
            if nu_data:
                all_errors.extend([d.nu_error_percent for d in nu_data if d.nu_error_percent is not None])
            
            if all_errors:
                overall_mean_error = np.mean(all_errors)
                systems_excellent = sum(1 for e in all_errors if e < 10.0)
                systems_good = sum(1 for e in all_errors if e < 20.0)
                
                f.write(f"  Overall mean error: {overall_mean_error:.2f}%\n")
                f.write(f"  Excellent accuracy (<10% error): {systems_excellent}/{len(all_errors)} ({systems_excellent/len(all_errors)*100:.1f}%)\n")
                f.write(f"  Good accuracy (<20% error): {systems_good}/{len(all_errors)} ({systems_good/len(all_errors)*100:.1f}%)\n")
                
                if overall_mean_error < 10.0:
                    assessment = "EXCELLENT - Publication ready"
                elif overall_mean_error < 20.0:
                    assessment = "GOOD - Acceptable for publication"
                else:
                    assessment = "NEEDS IMPROVEMENT - Requires further optimization"
                
                f.write(f"\n  OVERALL ASSESSMENT: {assessment}\n")
            
            # Recommendations
            f.write("\nRecommendations:\n")
            f.write("-" * 15 + "\n")
            
            if all_errors:
                worst_systems = [d for d in exponent_data 
                               if (d.beta_error_percent and d.beta_error_percent > 20.0) or 
                                  (d.nu_error_percent and d.nu_error_percent > 20.0)]
                
                if worst_systems:
                    f.write("  Systems requiring improvement:\n")
                    for system in worst_systems:
                        f.write(f"    - {system.dimensionality} {system.model_type}\n")
                
                if overall_mean_error > 15.0:
                    f.write("  - Consider improving data quality and equilibration\n")
                    f.write("  - Optimize VAE architecture and training parameters\n")
                    f.write("  - Increase system sizes for better finite-size scaling\n")
                else:
                    f.write("  - Results are suitable for publication\n")
                    f.write("  - Consider adding more physics systems for broader validation\n")
        
        self.logger.info(f"Exponent summary report saved to {summary_file}")
        return str(summary_file)