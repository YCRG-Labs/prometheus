"""
Data Quality Validation and Magnetization Analysis for 3D Ising Data.

This module provides comprehensive data quality validation, magnetization curve analysis,
and visualization tools for 3D Ising model datasets, including transition behavior
validation around the critical temperature.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from scipy.ndimage import gaussian_filter1d
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import h5py

from ..data.data_generator_3d import Dataset3DResult, SystemSizeResult3D
from ..models.physics_models import Ising3DModel
import logging


@dataclass
class MagnetizationAnalysisResult:
    """Results from magnetization curve analysis."""
    system_size: int
    temperatures: np.ndarray
    mean_magnetization: np.ndarray
    std_magnetization: np.ndarray
    susceptibility: np.ndarray
    tc_estimate: float
    tc_confidence_interval: Tuple[float, float]
    transition_sharpness: float
    fit_quality: float
    metadata: Dict[str, Any]


@dataclass
class DataQualityReport3D:
    """Comprehensive data quality report for 3D dataset."""
    dataset_summary: Dict[str, Any]
    system_size_reports: Dict[int, Dict[str, Any]]
    magnetization_analysis: Dict[int, MagnetizationAnalysisResult]
    equilibration_analysis: Dict[str, Any]
    overall_quality_score: float
    issues_found: List[str]
    recommendations: List[str]
    validation_passed: bool


class DataQualityAnalyzer3D:
    """
    Comprehensive data quality analyzer for 3D Ising datasets.
    
    Provides validation of data quality, magnetization curve analysis,
    transition behavior validation around Tc ≈ 4.511, and visualization
    tools for 2D slices of 3D configurations.
    """
    
    def __init__(self, theoretical_tc: float = 4.511):
        """
        Initialize 3D data quality analyzer.
        
        Args:
            theoretical_tc: Theoretical critical temperature for 3D Ising model
        """
        self.theoretical_tc = theoretical_tc
        self.logger = get_logger(__name__)
        
        # Initialize physics model for reference
        self.physics_model = Ising3DModel(coupling_strength=1.0)
        
        self.logger.info(f"Initialized 3D data quality analyzer with Tc={theoretical_tc:.3f}")
    
    def analyze_dataset_quality(self, dataset: Dataset3DResult) -> DataQualityReport3D:
        """
        Perform comprehensive quality analysis of 3D dataset.
        
        Args:
            dataset: Dataset3DResult to analyze
            
        Returns:
            DataQualityReport3D with complete analysis results
        """
        self.logger.info("Starting comprehensive 3D dataset quality analysis")
        
        # Dataset summary
        dataset_summary = self._create_dataset_summary(dataset)
        
        # Analyze each system size
        system_size_reports = {}
        magnetization_analysis = {}
        
        for system_size, size_result in dataset.system_size_results.items():
            self.logger.info(f"Analyzing system size L={system_size}")
            
            # System-specific quality analysis
            size_report = self._analyze_system_size_quality(size_result)
            system_size_reports[system_size] = size_report
            
            # Magnetization curve analysis
            mag_analysis = self._analyze_magnetization_curves(size_result)
            magnetization_analysis[system_size] = mag_analysis
        
        # Equilibration analysis across all systems
        equilibration_analysis = self._analyze_equilibration_quality(dataset)
        
        # Overall quality assessment
        overall_quality, issues, recommendations = self._assess_overall_quality(
            dataset_summary, system_size_reports, magnetization_analysis, equilibration_analysis
        )
        
        # Create comprehensive report
        report = DataQualityReport3D(
            dataset_summary=dataset_summary,
            system_size_reports=system_size_reports,
            magnetization_analysis=magnetization_analysis,
            equilibration_analysis=equilibration_analysis,
            overall_quality_score=overall_quality,
            issues_found=issues,
            recommendations=recommendations,
            validation_passed=overall_quality >= 0.7 and len(issues) == 0
        )
        
        self.logger.info(f"Quality analysis complete: score={overall_quality:.3f}, "
                        f"validation={'PASSED' if report.validation_passed else 'FAILED'}")
        
        return report
    
    def _create_dataset_summary(self, dataset: Dataset3DResult) -> Dict[str, Any]:
        """Create summary statistics for the dataset."""
        summary = {
            'total_configurations': dataset.total_configurations,
            'system_sizes': list(dataset.system_size_results.keys()),
            'n_system_sizes': len(dataset.system_size_results),
            'theoretical_tc': dataset.theoretical_tc,
            'generation_time_hours': dataset.total_generation_time / 3600,
            'temperature_range': dataset.config.temperature_range,
            'n_temperatures': dataset.config.temperature_resolution,
            'configs_per_temp': dataset.config.n_configs_per_temp,
            'sampling_interval': dataset.config.sampling_interval
        }
        
        # Calculate total lattice sites across all systems
        total_sites = 0
        for size_result in dataset.system_size_results.values():
            n_sites = np.prod(size_result.lattice_shape)
            n_configs = len(size_result.temperatures) * dataset.config.n_configs_per_temp
            total_sites += n_sites * n_configs
        
        summary['total_lattice_sites_simulated'] = total_sites
        
        return summary
    
    def _analyze_system_size_quality(self, size_result: SystemSizeResult3D) -> Dict[str, Any]:
        """Analyze data quality for a specific system size."""
        system_size = size_result.system_size
        
        # Configuration validation
        config_validation = self._validate_configurations(size_result)
        
        # Energy analysis
        energy_analysis = self._analyze_energy_curves(size_result)
        
        # Magnetization statistics
        mag_stats = self._compute_magnetization_statistics(size_result)
        
        # Equilibration success analysis
        eq_analysis = self._analyze_equilibration_success(size_result)
        
        report = {
            'system_size': system_size,
            'lattice_shape': size_result.lattice_shape,
            'n_sites': np.prod(size_result.lattice_shape),
            'n_temperatures': len(size_result.temperatures),
            'n_configurations': len(size_result.temperatures) * len(size_result.configurations[0]),
            'generation_time_minutes': size_result.generation_time_seconds / 60,
            'configuration_validation': config_validation,
            'energy_analysis': energy_analysis,
            'magnetization_statistics': mag_stats,
            'equilibration_analysis': eq_analysis
        }
        
        return report
    
    def _validate_configurations(self, size_result: SystemSizeResult3D) -> Dict[str, Any]:
        """Validate spin configurations for correctness."""
        validation = {
            'total_configs': 0,
            'invalid_spins': 0,
            'shape_errors': 0,
            'nan_values': 0,
            'energy_outliers': 0,
            'magnetization_outliers': 0
        }
        
        expected_shape = size_result.lattice_shape
        
        for temp_idx, temp_configs in enumerate(size_result.configurations):
            for config in temp_configs:
                validation['total_configs'] += 1
                
                # Check spin values
                unique_spins = np.unique(config.spins)
                if not np.array_equal(np.sort(unique_spins), np.array([-1, 1])):
                    validation['invalid_spins'] += 1
                
                # Check shape
                if config.spins.shape != expected_shape:
                    validation['shape_errors'] += 1
                
                # Check for NaN values
                if not np.isfinite(config.spins).all():
                    validation['nan_values'] += 1
                
                # Check for reasonable energy values
                if abs(config.energy) > 10:  # Reasonable bound for energy per spin
                    validation['energy_outliers'] += 1
                
                # Check for reasonable magnetization values
                if abs(config.magnetization) > 1.0:
                    validation['magnetization_outliers'] += 1
        
        # Calculate error rates
        total = validation['total_configs']
        validation['error_rates'] = {
            'invalid_spins_rate': validation['invalid_spins'] / total if total > 0 else 0,
            'shape_error_rate': validation['shape_errors'] / total if total > 0 else 0,
            'nan_rate': validation['nan_values'] / total if total > 0 else 0,
            'energy_outlier_rate': validation['energy_outliers'] / total if total > 0 else 0,
            'magnetization_outlier_rate': validation['magnetization_outliers'] / total if total > 0 else 0
        }
        
        validation['is_valid'] = all(rate < 0.01 for rate in validation['error_rates'].values())
        
        return validation
    
    def _analyze_energy_curves(self, size_result: SystemSizeResult3D) -> Dict[str, Any]:
        """Analyze energy curves for physical consistency."""
        energy_curves = size_result.energy_curves
        temperatures = size_result.temperatures
        
        # Calculate mean and std energy at each temperature
        mean_energies = np.mean(energy_curves, axis=1)
        std_energies = np.std(energy_curves, axis=1)
        
        # Check for monotonic behavior (energy should generally increase with temperature)
        energy_gradient = np.gradient(mean_energies, temperatures)
        monotonic_fraction = np.sum(energy_gradient > 0) / len(energy_gradient)
        
        # Check for reasonable energy range
        energy_range = np.max(mean_energies) - np.min(mean_energies)
        
        # Specific heat estimation (derivative of energy)
        specific_heat = np.gradient(mean_energies, temperatures)
        
        analysis = {
            'mean_energies': mean_energies,
            'std_energies': std_energies,
            'energy_range': energy_range,
            'monotonic_fraction': monotonic_fraction,
            'specific_heat': specific_heat,
            'energy_gradient': energy_gradient,
            'is_physically_reasonable': monotonic_fraction > 0.8 and energy_range > 0.5
        }
        
        return analysis
    
    def _compute_magnetization_statistics(self, size_result: SystemSizeResult3D) -> Dict[str, Any]:
        """Compute comprehensive magnetization statistics."""
        mag_curves = size_result.magnetization_curves
        temperatures = size_result.temperatures
        
        # Basic statistics
        mean_mags = np.mean(mag_curves, axis=1)
        std_mags = np.std(mag_curves, axis=1)
        
        # Susceptibility (χ = N * <M²> - <M>²) / T)
        mean_mag_squared = np.mean(mag_curves**2, axis=1)
        susceptibility = (mean_mag_squared - mean_mags**2) / temperatures
        
        # Find transition region
        tc_idx = np.argmin(np.abs(temperatures - self.theoretical_tc))
        
        stats = {
            'mean_magnetization': mean_mags,
            'std_magnetization': std_mags,
            'susceptibility': susceptibility,
            'max_susceptibility': np.max(susceptibility),
            'max_susceptibility_temperature': temperatures[np.argmax(susceptibility)],
            'tc_index': tc_idx,
            'tc_magnetization': mean_mags[tc_idx],
            'low_temp_magnetization': np.mean(mean_mags[:len(temperatures)//4]),
            'high_temp_magnetization': np.mean(mean_mags[-len(temperatures)//4:])
        }
        
        # Transition sharpness
        stats['transition_sharpness'] = stats['low_temp_magnetization'] - stats['high_temp_magnetization']
        
        return stats
    
    def _analyze_magnetization_curves(self, size_result: SystemSizeResult3D) -> MagnetizationAnalysisResult:
        """Perform detailed magnetization curve analysis with Tc estimation."""
        mag_curves = size_result.magnetization_curves
        temperatures = size_result.temperatures
        
        # Calculate statistics
        mean_mags = np.mean(mag_curves, axis=1)
        std_mags = np.std(mag_curves, axis=1)
        
        # Susceptibility calculation
        mean_mag_squared = np.mean(mag_curves**2, axis=1)
        susceptibility = (mean_mag_squared - mean_mags**2) / temperatures
        
        # Estimate Tc from susceptibility peak
        tc_estimate, tc_confidence = self._estimate_tc_from_susceptibility(temperatures, susceptibility)
        
        # Transition sharpness
        low_temp_mag = np.mean(mean_mags[:len(temperatures)//4])
        high_temp_mag = np.mean(mean_mags[-len(temperatures)//4:])
        transition_sharpness = low_temp_mag - high_temp_mag
        
        # Fit quality assessment
        fit_quality = self._assess_magnetization_fit_quality(temperatures, mean_mags, tc_estimate)
        
        result = MagnetizationAnalysisResult(
            system_size=size_result.system_size,
            temperatures=temperatures,
            mean_magnetization=mean_mags,
            std_magnetization=std_mags,
            susceptibility=susceptibility,
            tc_estimate=tc_estimate,
            tc_confidence_interval=tc_confidence,
            transition_sharpness=transition_sharpness,
            fit_quality=fit_quality,
            metadata={
                'theoretical_tc': self.theoretical_tc,
                'tc_error_percent': abs(tc_estimate - self.theoretical_tc) / self.theoretical_tc * 100,
                'max_susceptibility': np.max(susceptibility),
                'max_susceptibility_temperature': temperatures[np.argmax(susceptibility)]
            }
        )
        
        return result
    
    def _estimate_tc_from_susceptibility(self, temperatures: np.ndarray, 
                                       susceptibility: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        """Estimate critical temperature from susceptibility peak."""
        # Find peak in susceptibility
        max_idx = np.argmax(susceptibility)
        tc_estimate = temperatures[max_idx]
        
        # Estimate confidence interval using neighboring points
        window = 3
        start_idx = max(0, max_idx - window)
        end_idx = min(len(temperatures), max_idx + window + 1)
        
        local_temps = temperatures[start_idx:end_idx]
        local_susc = susceptibility[start_idx:end_idx]
        
        # Fit parabola around peak for better estimate
        try:
            coeffs = np.polyfit(local_temps, local_susc, 2)
            if coeffs[0] < 0:  # Parabola opens downward (valid peak)
                tc_refined = -coeffs[1] / (2 * coeffs[0])
                if local_temps[0] <= tc_refined <= local_temps[-1]:
                    tc_estimate = tc_refined
        except:
            pass  # Use simple max if fitting fails
        
        # Confidence interval based on temperature resolution
        temp_resolution = temperatures[1] - temperatures[0]
        confidence_interval = (tc_estimate - temp_resolution, tc_estimate + temp_resolution)
        
        return tc_estimate, confidence_interval
    
    def _assess_magnetization_fit_quality(self, temperatures: np.ndarray, 
                                        magnetization: np.ndarray, 
                                        tc_estimate: float) -> float:
        """Assess quality of magnetization curve fit to expected behavior."""
        # Check for proper transition behavior
        tc_idx = np.argmin(np.abs(temperatures - tc_estimate))
        
        # Low temperature should have high magnetization
        low_temp_region = temperatures < tc_estimate - 0.5
        high_temp_region = temperatures > tc_estimate + 0.5
        
        if np.any(low_temp_region) and np.any(high_temp_region):
            low_temp_mag = np.mean(magnetization[low_temp_region])
            high_temp_mag = np.mean(magnetization[high_temp_region])
            
            # Quality based on transition sharpness and expected behavior
            transition_quality = (low_temp_mag - high_temp_mag) / low_temp_mag if low_temp_mag > 0 else 0
            
            # Check for monotonic decrease around transition
            transition_region = np.abs(temperatures - tc_estimate) < 1.0
            if np.any(transition_region):
                transition_temps = temperatures[transition_region]
                transition_mags = magnetization[transition_region]
                
                # Sort by temperature and check for general decreasing trend
                sorted_indices = np.argsort(transition_temps)
                sorted_mags = transition_mags[sorted_indices]
                
                # Calculate fraction of decreasing steps
                decreasing_steps = np.sum(np.diff(sorted_mags) < 0)
                total_steps = len(sorted_mags) - 1
                monotonic_quality = decreasing_steps / total_steps if total_steps > 0 else 0
            else:
                monotonic_quality = 0.5
            
            # Combined quality score
            fit_quality = 0.6 * transition_quality + 0.4 * monotonic_quality
        else:
            fit_quality = 0.0
        
        return max(0, min(1, fit_quality))
    
    def _analyze_equilibration_success(self, size_result: SystemSizeResult3D) -> Dict[str, Any]:
        """Analyze equilibration success rates and quality."""
        eq_results = size_result.equilibration_results
        
        analysis = {
            'total_temperatures': len(eq_results),
            'successful_equilibrations': sum(1 for eq in eq_results if eq.converged),
            'failed_equilibrations': sum(1 for eq in eq_results if not eq.converged),
            'success_rate': sum(1 for eq in eq_results if eq.converged) / len(eq_results),
            'average_equilibration_steps': np.mean([eq.equilibration_steps for eq in eq_results]),
            'average_quality_score': np.mean([eq.convergence_quality_score for eq in eq_results]),
            'average_energy_autocorr_time': np.mean([eq.energy_autocorr_time for eq in eq_results if np.isfinite(eq.energy_autocorr_time)]),
            'average_mag_autocorr_time': np.mean([eq.magnetization_autocorr_time for eq in eq_results if np.isfinite(eq.magnetization_autocorr_time)])
        }
        
        # Temperature-dependent analysis
        temperatures = size_result.temperatures
        tc_region = np.abs(temperatures - self.theoretical_tc) < 0.5
        
        if np.any(tc_region):
            tc_eq_results = [eq for i, eq in enumerate(eq_results) if tc_region[i]]
            analysis['tc_region_success_rate'] = sum(1 for eq in tc_eq_results if eq.converged) / len(tc_eq_results)
        else:
            analysis['tc_region_success_rate'] = analysis['success_rate']
        
        return analysis
    
    def _analyze_equilibration_quality(self, dataset: Dataset3DResult) -> Dict[str, Any]:
        """Analyze equilibration quality across all system sizes."""
        all_eq_results = []
        
        for size_result in dataset.system_size_results.values():
            all_eq_results.extend(size_result.equilibration_results)
        
        analysis = {
            'total_equilibrations': len(all_eq_results),
            'overall_success_rate': sum(1 for eq in all_eq_results if eq.converged) / len(all_eq_results),
            'average_quality_score': np.mean([eq.convergence_quality_score for eq in all_eq_results]),
            'quality_score_std': np.std([eq.convergence_quality_score for eq in all_eq_results]),
            'failed_equilibrations_by_size': {}
        }
        
        # Analyze failures by system size
        for system_size, size_result in dataset.system_size_results.items():
            eq_results = size_result.equilibration_results
            failed_count = sum(1 for eq in eq_results if not eq.converged)
            analysis['failed_equilibrations_by_size'][system_size] = {
                'failed_count': failed_count,
                'total_count': len(eq_results),
                'failure_rate': failed_count / len(eq_results)
            }
        
        return analysis
    
    def _assess_overall_quality(self, dataset_summary: Dict[str, Any],
                              system_reports: Dict[int, Dict[str, Any]],
                              mag_analysis: Dict[int, MagnetizationAnalysisResult],
                              eq_analysis: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
        """Assess overall dataset quality and provide recommendations."""
        issues = []
        recommendations = []
        quality_scores = []
        
        # Configuration validation quality
        config_quality_scores = []
        for size, report in system_reports.items():
            config_val = report['configuration_validation']
            if not config_val['is_valid']:
                issues.append(f"Configuration validation failed for L={size}")
            
            # Score based on error rates
            error_rates = config_val['error_rates']
            max_error_rate = max(error_rates.values())
            config_score = max(0, 1 - max_error_rate * 10)  # Penalize errors heavily
            config_quality_scores.append(config_score)
        
        config_quality = np.mean(config_quality_scores) if config_quality_scores else 0
        quality_scores.append(config_quality)
        
        # Equilibration quality
        eq_success_rate = eq_analysis['overall_success_rate']
        eq_quality_score = eq_analysis['average_quality_score']
        
        if eq_success_rate < 0.8:
            issues.append(f"Low equilibration success rate: {eq_success_rate:.2%}")
            recommendations.append("Consider increasing equilibration time or adjusting temperature range")
        
        if eq_quality_score < 0.7:
            issues.append(f"Low equilibration quality score: {eq_quality_score:.3f}")
        
        equilibration_quality = 0.7 * eq_success_rate + 0.3 * eq_quality_score
        quality_scores.append(equilibration_quality)
        
        # Magnetization analysis quality
        mag_quality_scores = []
        for size, mag_result in mag_analysis.items():
            tc_error = abs(mag_result.tc_estimate - self.theoretical_tc) / self.theoretical_tc
            
            if tc_error > 0.1:  # More than 10% error
                issues.append(f"Large Tc estimation error for L={size}: {tc_error:.1%}")
            
            if mag_result.transition_sharpness < 0.2:
                issues.append(f"Weak magnetization transition for L={size}")
                recommendations.append(f"Check equilibration quality for L={size}")
            
            # Quality score based on Tc accuracy and transition sharpness
            tc_score = max(0, 1 - tc_error * 5)  # Penalize Tc errors
            transition_score = min(1, mag_result.transition_sharpness / 0.5)  # Normalize transition sharpness
            fit_score = mag_result.fit_quality
            
            mag_score = 0.4 * tc_score + 0.3 * transition_score + 0.3 * fit_score
            mag_quality_scores.append(mag_score)
        
        magnetization_quality = np.mean(mag_quality_scores) if mag_quality_scores else 0
        quality_scores.append(magnetization_quality)
        
        # Overall quality score
        overall_quality = np.mean(quality_scores)
        
        # Additional recommendations
        if overall_quality < 0.7:
            recommendations.append("Overall dataset quality is below acceptable threshold")
            recommendations.append("Consider regenerating data with improved parameters")
        
        if len(issues) == 0 and overall_quality >= 0.8:
            recommendations.append("Dataset quality is excellent - suitable for publication")
        
        return overall_quality, issues, recommendations
    
    def create_visualization_2d_slices(self, size_result: SystemSizeResult3D,
                                     temperature_indices: Optional[List[int]] = None,
                                     config_indices: Optional[List[int]] = None,
                                     slice_axis: int = 0,
                                     output_dir: Optional[str] = None) -> List[str]:
        """
        Create visualization tools for 2D slices of 3D configurations.
        
        Args:
            size_result: SystemSizeResult3D to visualize
            temperature_indices: Indices of temperatures to visualize (default: around Tc)
            config_indices: Indices of configurations to visualize (default: first few)
            slice_axis: Axis along which to take slices (0, 1, or 2)
            output_dir: Directory to save plots (default: current directory)
            
        Returns:
            List of paths to generated plot files
        """
        if output_dir is None:
            output_dir = "."
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default temperature indices around Tc
        if temperature_indices is None:
            tc_idx = np.argmin(np.abs(size_result.temperatures - self.theoretical_tc))
            temperature_indices = [max(0, tc_idx-2), tc_idx, min(len(size_result.temperatures)-1, tc_idx+2)]
        
        # Default configuration indices
        if config_indices is None:
            config_indices = [0, 1, 2]  # First few configurations
        
        plot_files = []
        
        for temp_idx in temperature_indices:
            if temp_idx >= len(size_result.temperatures):
                continue
                
            temperature = size_result.temperatures[temp_idx]
            temp_configs = size_result.configurations[temp_idx]
            
            for config_idx in config_indices:
                if config_idx >= len(temp_configs):
                    continue
                
                config = temp_configs[config_idx]
                spins = config.spins
                
                # Create figure with multiple slices
                n_slices = min(4, spins.shape[slice_axis])
                fig, axes = plt.subplots(1, n_slices, figsize=(4*n_slices, 4))
                if n_slices == 1:
                    axes = [axes]
                
                slice_indices = np.linspace(0, spins.shape[slice_axis]-1, n_slices, dtype=int)
                
                for i, slice_idx in enumerate(slice_indices):
                    if slice_axis == 0:
                        slice_data = spins[slice_idx, :, :]
                    elif slice_axis == 1:
                        slice_data = spins[:, slice_idx, :]
                    else:  # slice_axis == 2
                        slice_data = spins[:, :, slice_idx]
                    
                    im = axes[i].imshow(slice_data, cmap='RdBu', vmin=-1, vmax=1, origin='lower')
                    axes[i].set_title(f'Slice {slice_idx}')
                    axes[i].set_aspect('equal')
                
                plt.suptitle(f'L={size_result.system_size}, T={temperature:.3f}, '
                           f'Config {config_idx}, M={config.magnetization:.3f}')
                plt.tight_layout()
                
                # Save plot
                filename = f'3d_slices_L{size_result.system_size}_T{temperature:.3f}_C{config_idx}.png'
                filepath = output_dir / filename
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close()
                
                plot_files.append(str(filepath))
        
        self.logger.info(f"Generated {len(plot_files)} 2D slice visualizations in {output_dir}")
        return plot_files


def analyze_3d_dataset_quality(dataset: Dataset3DResult,
                             create_visualizations: bool = True,
                             output_dir: Optional[str] = None) -> DataQualityReport3D:
    """
    Main function to analyze 3D dataset quality and create validation report.
    
    This function implements task 3.2 data quality validation and magnetization analysis:
    - Compute magnetization curves and validate transition behavior around Tc ≈ 4.511
    - Implement data quality checks for proper equilibration and sampling
    - Create visualization tools for 2D slices of 3D configurations
    
    Args:
        dataset: Dataset3DResult to analyze
        create_visualizations: Whether to create 2D slice visualizations
        output_dir: Directory for output files
        
    Returns:
        DataQualityReport3D with comprehensive analysis results
    """
    logger = get_logger(__name__)
    
    logger.info("Starting comprehensive 3D dataset quality analysis")
    
    # Create analyzer
    analyzer = DataQualityAnalyzer3D(theoretical_tc=dataset.theoretical_tc)
    
    # Perform quality analysis
    report = analyzer.analyze_dataset_quality(dataset)
    
    # Create visualizations if requested
    if create_visualizations and output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create 2D slice visualizations for each system size
        for system_size, size_result in dataset.system_size_results.items():
            viz_dir = output_path / f"visualizations_L{system_size}"
            
            try:
                plot_files = analyzer.create_visualization_2d_slices(
                    size_result=size_result,
                    output_dir=str(viz_dir)
                )
                logger.info(f"Created {len(plot_files)} visualizations for L={system_size}")
            except Exception as e:
                logger.warning(f"Failed to create visualizations for L={system_size}: {e}")
    
    # Log summary
    logger.info(f"Quality analysis complete:")
    logger.info(f"  Overall quality score: {report.overall_quality_score:.3f}")
    logger.info(f"  Validation status: {'PASSED' if report.validation_passed else 'FAILED'}")
    logger.info(f"  Issues found: {len(report.issues_found)}")
    logger.info(f"  Recommendations: {len(report.recommendations)}")
    
    if report.issues_found:
        logger.warning("Issues found:")
        for issue in report.issues_found:
            logger.warning(f"  - {issue}")
    
    if report.recommendations:
        logger.info("Recommendations:")
        for rec in report.recommendations:
            logger.info(f"  - {rec}")
    
    return report