"""
Final Validation and Quality Assurance System

This module implements task 11.2: Create final validation and quality assurance system
- Validate proper equilibration through energy convergence monitoring
- Implement comprehensive data quality checks for all generated datasets
- Add final physics consistency validation across all results
"""

import numpy as np
import h5py
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks
import warnings

try:
    from ..analysis.latent_analysis import LatentRepresentation
    from ..utils.logging_utils import get_logger
    from .statistical_validation_framework import StatisticalValidationFramework
except ImportError:
    # Fallback for testing
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    try:
        from analysis.latent_analysis import LatentRepresentation
        from utils.logging_utils import get_logger
        from validation.statistical_validation_framework import StatisticalValidationFramework
    except ImportError:
        # Mock for testing
        class LatentRepresentation:
            pass
        
        class StatisticalValidationFramework:
            pass
        
        def get_logger(name):
            return logging.getLogger(name)


@dataclass
class EquilibrationValidationResult:
    """Container for equilibration validation results."""
    is_equilibrated: bool
    convergence_step: int
    energy_autocorrelation_time: float
    magnetization_autocorrelation_time: float
    final_energy_variance: float
    convergence_quality_score: float
    equilibration_method: str
    validation_details: Dict[str, Any]


@dataclass
class DataQualityResult:
    """Container for data quality assessment results."""
    dataset_name: str
    total_configurations: int
    valid_configurations: int
    data_completeness: float
    
    # Temperature coverage
    temperature_range_coverage: float
    critical_region_coverage: float
    
    # Physical consistency
    magnetization_range_valid: bool
    energy_range_valid: bool
    phase_transition_visible: bool
    
    # Statistical quality
    sample_size_adequate: bool
    sampling_interval_adequate: bool
    
    # Overall quality
    overall_quality_score: float
    quality_issues: List[str]
    recommendations: List[str]


@dataclass
class PhysicsConsistencyResult:
    """Container for physics consistency validation results."""
    system_type: str
    
    # Critical behavior validation
    critical_temperature_consistent: bool
    critical_exponents_consistent: bool
    universality_class_match: bool
    
    # Phase transition validation
    phase_separation_clear: bool
    order_parameter_behavior_correct: bool
    susceptibility_peak_present: bool
    
    # Scaling behavior validation
    finite_size_scaling_valid: bool
    critical_scaling_valid: bool
    
    # Overall physics consistency
    physics_consistency_score: float
    consistency_issues: List[str]
    physics_recommendations: List[str]


@dataclass
class FinalValidationReport:
    """Container for final comprehensive validation report."""
    validation_timestamp: str
    
    # System-level results
    equilibration_results: Dict[str, EquilibrationValidationResult]
    data_quality_results: Dict[str, DataQualityResult]
    physics_consistency_results: Dict[str, PhysicsConsistencyResult]
    
    # Overall assessment
    all_systems_equilibrated: bool
    all_data_quality_passed: bool
    all_physics_consistent: bool
    
    # Summary metrics
    average_equilibration_quality: float
    average_data_quality: float
    average_physics_consistency: float
    overall_validation_score: float
    
    # Final recommendations
    critical_issues: List[str]
    improvement_recommendations: List[str]
    validation_passed: bool


class FinalValidationSystem:
    """
    Final validation and quality assurance system for comprehensive physics validation.
    
    Features:
    1. Equilibration validation through energy convergence monitoring
    2. Comprehensive data quality checks for all datasets
    3. Physics consistency validation across all results
    4. Final quality assurance and recommendation system
    """
    
    def __init__(self,
                 equilibration_threshold: float = 1e-4,
                 autocorr_window_factor: int = 10,
                 min_equilibration_steps: int = 1000,
                 quality_threshold: float = 0.7,
                 random_seed: Optional[int] = None):
        """
        Initialize final validation system.
        
        Args:
            equilibration_threshold: Threshold for energy convergence
            autocorr_window_factor: Factor for autocorrelation window size
            min_equilibration_steps: Minimum steps for equilibration
            quality_threshold: Minimum quality score threshold
            random_seed: Random seed for reproducibility
        """
        self.equilibration_threshold = equilibration_threshold
        self.autocorr_window_factor = autocorr_window_factor
        self.min_equilibration_steps = min_equilibration_steps
        self.quality_threshold = quality_threshold
        self.random_seed = random_seed
        self.logger = get_logger(__name__)
        
        # Initialize statistical validation framework
        try:
            self.statistical_framework = StatisticalValidationFramework(
                random_seed=random_seed
            )
        except TypeError:
            # Handle case where StatisticalValidationFramework is a mock
            self.statistical_framework = None
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def validate_equilibration(self,
                             energy_series: np.ndarray,
                             magnetization_series: Optional[np.ndarray] = None,
                             system_name: str = "system") -> EquilibrationValidationResult:
        """
        Validate proper equilibration through energy convergence monitoring.
        
        Args:
            energy_series: Time series of energy values
            magnetization_series: Optional time series of magnetization values
            system_name: Name of the system for logging
            
        Returns:
            EquilibrationValidationResult with detailed equilibration analysis
        """
        self.logger.info(f"Validating equilibration for {system_name}")
        
        if len(energy_series) < self.min_equilibration_steps:
            self.logger.warning(f"Insufficient equilibration steps: {len(energy_series)} < {self.min_equilibration_steps}")
            return self._create_failed_equilibration_result("Insufficient equilibration steps")
        
        # Method 1: Energy variance convergence
        convergence_step_variance = self._detect_convergence_by_variance(energy_series)
        
        # Method 2: Running average convergence
        convergence_step_average = self._detect_convergence_by_running_average(energy_series)
        
        # Method 3: Autocorrelation analysis
        energy_autocorr_time = self._compute_autocorrelation_time(energy_series)
        
        mag_autocorr_time = 0.0
        if magnetization_series is not None and len(magnetization_series) == len(energy_series):
            mag_autocorr_time = self._compute_autocorrelation_time(magnetization_series)
        
        # Determine final convergence step
        convergence_step = max(convergence_step_variance, convergence_step_average)
        convergence_step = max(convergence_step, int(5 * energy_autocorr_time))  # At least 5 autocorr times
        
        # Check if equilibrated
        is_equilibrated = (
            convergence_step < len(energy_series) * 0.9 and  # Converged before 90% of simulation
            convergence_step >= min(self.min_equilibration_steps, len(energy_series) // 4) and  # More lenient for short series
            energy_autocorr_time > 0 and
            energy_autocorr_time < len(energy_series) / 5  # More lenient autocorr time
        )
        
        # Compute final energy variance (after equilibration)
        if convergence_step < len(energy_series):
            equilibrated_energies = energy_series[convergence_step:]
            final_energy_variance = np.var(equilibrated_energies)
        else:
            final_energy_variance = np.var(energy_series)
        
        # Quality score assessment
        quality_components = []
        
        # Convergence quality (earlier convergence is better)
        conv_quality = max(0, 1 - convergence_step / len(energy_series))
        quality_components.append(conv_quality * 0.3)
        
        # Autocorrelation quality (shorter autocorr time is better)
        if energy_autocorr_time > 0:
            autocorr_quality = max(0, 1 - energy_autocorr_time / (len(energy_series) / 10))
        else:
            autocorr_quality = 0.5
        quality_components.append(autocorr_quality * 0.3)
        
        # Variance stability (lower final variance is better)
        if len(energy_series) > convergence_step:
            pre_equilib_var = np.var(energy_series[:convergence_step]) if convergence_step > 0 else np.var(energy_series)
            post_equilib_var = final_energy_variance
            
            if pre_equilib_var > 0:
                variance_improvement = max(0, 1 - post_equilib_var / pre_equilib_var)
            else:
                variance_improvement = 0.5
        else:
            variance_improvement = 0.3
        
        quality_components.append(variance_improvement * 0.4)
        
        convergence_quality_score = sum(quality_components)
        
        # Validation details
        validation_details = {
            'convergence_step_variance': convergence_step_variance,
            'convergence_step_average': convergence_step_average,
            'total_steps': len(energy_series),
            'convergence_fraction': convergence_step / len(energy_series),
            'energy_autocorr_time': energy_autocorr_time,
            'magnetization_autocorr_time': mag_autocorr_time,
            'final_energy_variance': final_energy_variance,
            'quality_components': {
                'convergence_quality': conv_quality,
                'autocorrelation_quality': autocorr_quality,
                'variance_improvement': variance_improvement
            }
        }
        
        result = EquilibrationValidationResult(
            is_equilibrated=is_equilibrated,
            convergence_step=convergence_step,
            energy_autocorrelation_time=energy_autocorr_time,
            magnetization_autocorrelation_time=mag_autocorr_time,
            final_energy_variance=final_energy_variance,
            convergence_quality_score=convergence_quality_score,
            equilibration_method="multi_method_analysis",
            validation_details=validation_details
        )
        
        self.logger.info(f"Equilibration validation for {system_name}: "
                        f"{'PASSED' if is_equilibrated else 'FAILED'} "
                        f"(quality: {convergence_quality_score:.3f})")
        
        return result
    
    def _detect_convergence_by_variance(self, energy_series: np.ndarray) -> int:
        """Detect convergence by monitoring energy variance in sliding windows."""
        
        window_size = max(100, len(energy_series) // 20)
        variances = []
        
        for i in range(window_size, len(energy_series)):
            window_energies = energy_series[i-window_size:i]
            variances.append(np.var(window_energies))
        
        if not variances:
            return len(energy_series) // 2
        
        variances = np.array(variances)
        
        # Find where variance stabilizes (derivative approaches zero)
        if len(variances) > 10:
            variance_derivative = np.gradient(variances)
            
            # Find where derivative is consistently small
            stable_threshold = np.std(variance_derivative) * 0.1
            
            for i in range(len(variance_derivative) - 5):
                if all(abs(variance_derivative[i:i+5]) < stable_threshold):
                    return window_size + i
        
        # Fallback: use point where variance is below threshold
        mean_variance = np.mean(variances)
        threshold = mean_variance * (1 + self.equilibration_threshold)
        
        for i, var in enumerate(variances):
            if var < threshold:
                return window_size + i
        
        return len(energy_series) // 2
    
    def _detect_convergence_by_running_average(self, energy_series: np.ndarray) -> int:
        """Detect convergence by monitoring running average stability."""
        
        window_size = max(50, len(energy_series) // 30)
        running_averages = []
        
        for i in range(window_size, len(energy_series)):
            window_energies = energy_series[i-window_size:i]
            running_averages.append(np.mean(window_energies))
        
        if not running_averages:
            return len(energy_series) // 2
        
        running_averages = np.array(running_averages)
        
        # Find where running average stabilizes
        if len(running_averages) > 10:
            avg_derivative = np.gradient(running_averages)
            
            # Find where derivative is consistently small
            stable_threshold = np.std(avg_derivative) * self.equilibration_threshold
            
            for i in range(len(avg_derivative) - 5):
                if all(abs(avg_derivative[i:i+5]) < stable_threshold):
                    return window_size + i
        
        return len(energy_series) // 2
    
    def _compute_autocorrelation_time(self, time_series: np.ndarray) -> float:
        """Compute autocorrelation time for a time series."""
        
        if len(time_series) < 10:
            return 1.0
        
        # Center the data
        centered_series = time_series - np.mean(time_series)
        
        # Compute autocorrelation function
        n = len(centered_series)
        autocorr = np.correlate(centered_series, centered_series, mode='full')
        autocorr = autocorr[n-1:]  # Take positive lags only
        
        # Normalize
        autocorr = autocorr / autocorr[0]
        
        # Find where autocorrelation drops to 1/e
        target = 1.0 / np.e
        
        for i, corr in enumerate(autocorr):
            if corr <= target:
                return float(i)
        
        # If never drops to 1/e, return a reasonable estimate
        return min(len(time_series) / 10, 100.0)
    
    def _create_failed_equilibration_result(self, reason: str) -> EquilibrationValidationResult:
        """Create a failed equilibration result."""
        return EquilibrationValidationResult(
            is_equilibrated=False,
            convergence_step=0,
            energy_autocorrelation_time=0.0,
            magnetization_autocorrelation_time=0.0,
            final_energy_variance=0.0,
            convergence_quality_score=0.0,
            equilibration_method="failed",
            validation_details={'failure_reason': reason}
        )
    
    def validate_data_quality(self,
                            dataset_path: str,
                            system_config: Dict[str, Any],
                            dataset_name: str = "dataset") -> DataQualityResult:
        """
        Implement comprehensive data quality checks for generated datasets.
        
        Args:
            dataset_path: Path to the dataset file
            system_config: System configuration parameters
            dataset_name: Name of the dataset for logging
            
        Returns:
            DataQualityResult with comprehensive quality assessment
        """
        self.logger.info(f"Validating data quality for {dataset_name}")
        
        quality_issues = []
        recommendations = []
        
        try:
            # Load dataset
            if dataset_path.endswith('.h5') or dataset_path.endswith('.hdf5'):
                data = self._load_h5_dataset(dataset_path)
            elif dataset_path.endswith('.npz'):
                data = self._load_npz_dataset(dataset_path)
            else:
                raise ValueError(f"Unsupported dataset format: {dataset_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset {dataset_path}: {e}")
            return self._create_failed_data_quality_result(dataset_name, f"Failed to load dataset: {e}")
        
        # Basic data completeness
        total_configs = len(data.get('configurations', []))
        valid_configs = total_configs
        
        # Check for invalid configurations
        if 'configurations' in data:
            configs = data['configurations']
            valid_mask = np.all(np.isfinite(configs), axis=tuple(range(1, configs.ndim)))
            valid_configs = np.sum(valid_mask)
        
        data_completeness = valid_configs / total_configs if total_configs > 0 else 0.0
        
        if data_completeness < 0.95:
            quality_issues.append(f"Low data completeness: {data_completeness:.1%}")
            recommendations.append("Check Monte Carlo simulation for numerical issues")
        
        # Temperature coverage validation
        temperatures = data.get('temperatures', np.array([]))
        temp_range_coverage = 0.0
        critical_region_coverage = 0.0
        
        if len(temperatures) > 0:
            expected_temp_range = system_config.get('temperature_range', (0, 10))
            actual_temp_range = (np.min(temperatures), np.max(temperatures))
            
            expected_span = expected_temp_range[1] - expected_temp_range[0]
            actual_span = actual_temp_range[1] - actual_temp_range[0]
            
            temp_range_coverage = min(1.0, actual_span / expected_span) if expected_span > 0 else 0.0
            
            # Critical region coverage
            theoretical_tc = system_config.get('theoretical_tc', 0.0)
            if theoretical_tc > 0:
                critical_window = 0.5  # ±0.5 around Tc
                critical_temps = temperatures[
                    (temperatures >= theoretical_tc - critical_window) & 
                    (temperatures <= theoretical_tc + critical_window)
                ]
                critical_region_coverage = len(critical_temps) / len(temperatures)
        
        if temp_range_coverage < 0.8:
            quality_issues.append(f"Insufficient temperature range coverage: {temp_range_coverage:.1%}")
            recommendations.append("Extend temperature range in data generation")
        
        if critical_region_coverage < 0.2:
            quality_issues.append(f"Poor critical region coverage: {critical_region_coverage:.1%}")
            recommendations.append("Increase sampling density near critical temperature")
        
        # Physical consistency checks
        magnetizations = data.get('magnetizations', np.array([]))
        energies = data.get('energies', np.array([]))
        
        magnetization_range_valid = True
        energy_range_valid = True
        phase_transition_visible = False
        
        if len(magnetizations) > 0:
            # Magnetization should be in reasonable range
            mag_range = (np.min(magnetizations), np.max(magnetizations))
            
            # For Ising models, magnetization should be between -1 and 1
            if mag_range[0] < -1.1 or mag_range[1] > 1.1:
                magnetization_range_valid = False
                quality_issues.append(f"Magnetization out of physical range: {mag_range}")
                recommendations.append("Check Monte Carlo implementation for magnetization calculation")
            
            # Check for phase transition visibility
            if len(temperatures) > 0 and len(temperatures) == len(magnetizations):
                # Look for clear transition in magnetization vs temperature
                unique_temps = np.unique(temperatures)
                if len(unique_temps) > 5:
                    temp_mags = []
                    for temp in unique_temps:
                        temp_mask = temperatures == temp
                        temp_mags.append(np.mean(np.abs(magnetizations[temp_mask])))
                    
                    # Check if there's a clear drop in magnetization
                    max_mag = np.max(temp_mags)
                    min_mag = np.min(temp_mags)
                    
                    if max_mag > 0.3 and min_mag < 0.2 and (max_mag - min_mag) > 0.3:
                        phase_transition_visible = True
        
        if len(energies) > 0:
            # Energy should be reasonable (negative for ferromagnetic Ising)
            energy_range = (np.min(energies), np.max(energies))
            
            # For Ising models, energy should be negative
            if system_config.get('system_type', '').startswith('ising'):
                if energy_range[1] > 0.1:  # Allow small positive values due to numerical precision
                    energy_range_valid = False
                    quality_issues.append(f"Positive energies in Ising model: {energy_range}")
                    recommendations.append("Check energy calculation in Monte Carlo simulation")
        
        if not phase_transition_visible:
            quality_issues.append("Phase transition not clearly visible in data")
            recommendations.append("Increase temperature resolution near critical point")
        
        # Statistical quality checks
        expected_configs_per_temp = system_config.get('n_configs_per_temp', 100)
        sample_size_adequate = (total_configs / len(np.unique(temperatures)) >= expected_configs_per_temp * 0.8 
                               if len(temperatures) > 0 else False)
        
        if not sample_size_adequate:
            quality_issues.append("Insufficient sample size per temperature")
            recommendations.append("Increase number of configurations per temperature")
        
        # Sampling interval adequacy (simplified check)
        sampling_interval_adequate = True
        if 'sampling_interval' in system_config:
            expected_interval = system_config['sampling_interval']
            if expected_interval < 10:  # Very short intervals might lead to correlation
                sampling_interval_adequate = False
                quality_issues.append("Sampling interval may be too short")
                recommendations.append("Increase sampling interval to reduce autocorrelation")
        
        # Overall quality score
        quality_components = [
            data_completeness * 0.2,
            temp_range_coverage * 0.15,
            critical_region_coverage * 0.15,
            (1.0 if magnetization_range_valid else 0.0) * 0.15,
            (1.0 if energy_range_valid else 0.0) * 0.15,
            (1.0 if phase_transition_visible else 0.0) * 0.1,
            (1.0 if sample_size_adequate else 0.0) * 0.05,
            (1.0 if sampling_interval_adequate else 0.0) * 0.05
        ]
        
        overall_quality_score = sum(quality_components)
        
        result = DataQualityResult(
            dataset_name=dataset_name,
            total_configurations=total_configs,
            valid_configurations=valid_configs,
            data_completeness=data_completeness,
            temperature_range_coverage=temp_range_coverage,
            critical_region_coverage=critical_region_coverage,
            magnetization_range_valid=magnetization_range_valid,
            energy_range_valid=energy_range_valid,
            phase_transition_visible=phase_transition_visible,
            sample_size_adequate=sample_size_adequate,
            sampling_interval_adequate=sampling_interval_adequate,
            overall_quality_score=overall_quality_score,
            quality_issues=quality_issues,
            recommendations=recommendations
        )
        
        self.logger.info(f"Data quality validation for {dataset_name}: "
                        f"score = {overall_quality_score:.3f}, "
                        f"issues = {len(quality_issues)}")
        
        return result
    
    def _load_h5_dataset(self, dataset_path: str) -> Dict[str, np.ndarray]:
        """Load HDF5 dataset."""
        data = {}
        
        with h5py.File(dataset_path, 'r') as f:
            for key in f.keys():
                data[key] = f[key][:]
        
        return data
    
    def _load_npz_dataset(self, dataset_path: str) -> Dict[str, np.ndarray]:
        """Load NPZ dataset."""
        return dict(np.load(dataset_path))
    
    def _create_failed_data_quality_result(self, dataset_name: str, reason: str) -> DataQualityResult:
        """Create a failed data quality result."""
        return DataQualityResult(
            dataset_name=dataset_name,
            total_configurations=0,
            valid_configurations=0,
            data_completeness=0.0,
            temperature_range_coverage=0.0,
            critical_region_coverage=0.0,
            magnetization_range_valid=False,
            energy_range_valid=False,
            phase_transition_visible=False,
            sample_size_adequate=False,
            sampling_interval_adequate=False,
            overall_quality_score=0.0,
            quality_issues=[reason],
            recommendations=["Fix data loading issues before proceeding"]
        )
    
    def validate_physics_consistency(self,
                                   analysis_results: Dict[str, Any],
                                   system_config: Dict[str, Any],
                                   system_name: str = "system") -> PhysicsConsistencyResult:
        """
        Add final physics consistency validation across all results.
        
        Args:
            analysis_results: Dictionary containing all analysis results
            system_config: System configuration parameters
            system_name: Name of the system for logging
            
        Returns:
            PhysicsConsistencyResult with comprehensive physics validation
        """
        self.logger.info(f"Validating physics consistency for {system_name}")
        
        system_type = system_config.get('system_type', 'unknown')
        consistency_issues = []
        physics_recommendations = []
        
        # Critical temperature consistency
        critical_temperature_consistent = True
        theoretical_tc = system_config.get('theoretical_tc', 0.0)
        
        if 'critical_temperature' in analysis_results and theoretical_tc > 0:
            measured_tc = analysis_results['critical_temperature']
            tc_error = abs(measured_tc - theoretical_tc) / theoretical_tc
            
            if tc_error > 0.1:  # 10% error threshold
                critical_temperature_consistent = False
                consistency_issues.append(f"Critical temperature error: {tc_error:.1%}")
                physics_recommendations.append("Improve critical temperature detection method")
        
        # Critical exponents consistency
        critical_exponents_consistent = True
        theoretical_exponents = system_config.get('theoretical_exponents', {})
        
        if 'critical_exponents' in analysis_results and theoretical_exponents:
            measured_exponents = analysis_results['critical_exponents']
            
            for exp_name, theoretical_value in theoretical_exponents.items():
                if exp_name in measured_exponents:
                    measured_value = measured_exponents[exp_name]
                    exp_error = abs(measured_value - theoretical_value) / abs(theoretical_value)
                    
                    if exp_error > 0.3:  # 30% error threshold for exponents
                        critical_exponents_consistent = False
                        consistency_issues.append(f"{exp_name} exponent error: {exp_error:.1%}")
                        physics_recommendations.append(f"Improve {exp_name} exponent extraction")
        
        # Universality class match
        universality_class_match = True
        
        if system_type in ['ising_2d', 'ising_3d'] and 'critical_exponents' in analysis_results:
            exponents = analysis_results['critical_exponents']
            
            if system_type == 'ising_2d':
                # 2D Ising universality class
                expected_beta = 0.125
                expected_nu = 1.0
                
                if 'beta' in exponents:
                    beta_error = abs(exponents['beta'] - expected_beta) / expected_beta
                    if beta_error > 0.5:
                        universality_class_match = False
                        consistency_issues.append("β exponent inconsistent with 2D Ising universality class")
                
                if 'nu' in exponents:
                    nu_error = abs(abs(exponents['nu']) - expected_nu) / expected_nu
                    if nu_error > 0.5:
                        universality_class_match = False
                        consistency_issues.append("ν exponent inconsistent with 2D Ising universality class")
            
            elif system_type == 'ising_3d':
                # 3D Ising universality class
                expected_beta = 0.326
                expected_nu = 0.630
                
                if 'beta' in exponents:
                    beta_error = abs(exponents['beta'] - expected_beta) / expected_beta
                    if beta_error > 0.5:
                        universality_class_match = False
                        consistency_issues.append("β exponent inconsistent with 3D Ising universality class")
                
                if 'nu' in exponents:
                    nu_error = abs(abs(exponents['nu']) - expected_nu) / expected_nu
                    if nu_error > 0.5:
                        universality_class_match = False
                        consistency_issues.append("ν exponent inconsistent with 3D Ising universality class")
        
        # Phase separation validation
        phase_separation_clear = True
        
        if 'latent_representation' in analysis_results:
            latent_repr = analysis_results['latent_representation']
            
            # Check if phases are well separated in latent space
            if hasattr(latent_repr, 'z1') and hasattr(latent_repr, 'temperatures'):
                z1 = latent_repr.z1
                temperatures = latent_repr.temperatures
                
                if theoretical_tc > 0:
                    below_tc_mask = temperatures < theoretical_tc
                    above_tc_mask = temperatures >= theoretical_tc
                    
                    if np.sum(below_tc_mask) > 0 and np.sum(above_tc_mask) > 0:
                        z1_below = np.mean(z1[below_tc_mask])
                        z1_above = np.mean(z1[above_tc_mask])
                        z1_std = np.std(z1)
                        
                        # Check separation in units of standard deviation
                        separation = abs(z1_below - z1_above) / (z1_std + 1e-10)
                        
                        if separation < 1.0:  # Less than 1 standard deviation separation
                            phase_separation_clear = False
                            consistency_issues.append("Poor phase separation in latent space")
                            physics_recommendations.append("Improve VAE training for better phase separation")
        
        # Order parameter behavior validation
        order_parameter_behavior_correct = True
        
        if 'order_parameter_analysis' in analysis_results:
            op_analysis = analysis_results['order_parameter_analysis']
            
            # Check if order parameter shows expected critical behavior
            if 'magnetization_vs_temperature' in op_analysis:
                mag_vs_temp = op_analysis['magnetization_vs_temperature']
                
                # Should show clear transition around Tc
                if theoretical_tc > 0:
                    # Check if magnetization drops significantly at Tc
                    temps_near_tc = []
                    mags_near_tc = []
                    
                    for temp, mag in mag_vs_temp:
                        if abs(temp - theoretical_tc) < 0.5:
                            temps_near_tc.append(temp)
                            mags_near_tc.append(abs(mag))
                    
                    if len(mags_near_tc) > 2:
                        mag_variation = np.max(mags_near_tc) - np.min(mags_near_tc)
                        if mag_variation < 0.2:  # Should see significant variation
                            order_parameter_behavior_correct = False
                            consistency_issues.append("Order parameter shows insufficient critical behavior")
        
        # Susceptibility peak validation
        susceptibility_peak_present = True
        
        if 'susceptibility_analysis' in analysis_results:
            susc_analysis = analysis_results['susceptibility_analysis']
            
            if 'susceptibility_vs_temperature' in susc_analysis:
                susc_vs_temp = susc_analysis['susceptibility_vs_temperature']
                
                # Should show peak near Tc
                if theoretical_tc > 0 and len(susc_vs_temp) > 5:
                    temps, susceptibilities = zip(*susc_vs_temp)
                    temps = np.array(temps)
                    susceptibilities = np.array(susceptibilities)
                    
                    # Find peak
                    peak_indices, _ = find_peaks(susceptibilities, height=np.mean(susceptibilities))
                    
                    if len(peak_indices) == 0:
                        susceptibility_peak_present = False
                        consistency_issues.append("No susceptibility peak found")
                        physics_recommendations.append("Check susceptibility calculation method")
                    else:
                        # Check if peak is near theoretical Tc
                        peak_temps = temps[peak_indices]
                        closest_peak = peak_temps[np.argmin(np.abs(peak_temps - theoretical_tc))]
                        
                        if abs(closest_peak - theoretical_tc) > 0.5:
                            susceptibility_peak_present = False
                            consistency_issues.append("Susceptibility peak far from theoretical Tc")
        
        # Finite-size scaling validation
        finite_size_scaling_valid = True
        
        if 'finite_size_scaling' in analysis_results:
            fs_results = analysis_results['finite_size_scaling']
            
            for observable, fs_result in fs_results.items():
                if hasattr(fs_result, 'scaling_validation_passed'):
                    if not fs_result.scaling_validation_passed:
                        finite_size_scaling_valid = False
                        consistency_issues.append(f"Finite-size scaling failed for {observable}")
                        physics_recommendations.append(f"Improve finite-size scaling analysis for {observable}")
        
        # Critical scaling validation
        critical_scaling_valid = True
        
        if 'critical_scaling' in analysis_results:
            scaling_results = analysis_results['critical_scaling']
            
            # Check if critical scaling follows expected power laws
            for quantity, scaling_data in scaling_results.items():
                if 'scaling_exponent' in scaling_data and 'expected_exponent' in scaling_data:
                    measured_exp = scaling_data['scaling_exponent']
                    expected_exp = scaling_data['expected_exponent']
                    
                    scaling_error = abs(measured_exp - expected_exp) / abs(expected_exp)
                    
                    if scaling_error > 0.4:  # 40% error threshold for scaling
                        critical_scaling_valid = False
                        consistency_issues.append(f"Critical scaling error for {quantity}: {scaling_error:.1%}")
        
        # Overall physics consistency score
        consistency_components = [
            (1.0 if critical_temperature_consistent else 0.0) * 0.2,
            (1.0 if critical_exponents_consistent else 0.0) * 0.25,
            (1.0 if universality_class_match else 0.0) * 0.2,
            (1.0 if phase_separation_clear else 0.0) * 0.1,
            (1.0 if order_parameter_behavior_correct else 0.0) * 0.1,
            (1.0 if susceptibility_peak_present else 0.0) * 0.05,
            (1.0 if finite_size_scaling_valid else 0.0) * 0.05,
            (1.0 if critical_scaling_valid else 0.0) * 0.05
        ]
        
        physics_consistency_score = sum(consistency_components)
        
        if not universality_class_match:
            physics_recommendations.append("Verify system implementation matches expected universality class")
        
        if len(consistency_issues) == 0:
            physics_recommendations.append("Physics consistency validation passed - results are reliable")
        
        result = PhysicsConsistencyResult(
            system_type=system_type,
            critical_temperature_consistent=critical_temperature_consistent,
            critical_exponents_consistent=critical_exponents_consistent,
            universality_class_match=universality_class_match,
            phase_separation_clear=phase_separation_clear,
            order_parameter_behavior_correct=order_parameter_behavior_correct,
            susceptibility_peak_present=susceptibility_peak_present,
            finite_size_scaling_valid=finite_size_scaling_valid,
            critical_scaling_valid=critical_scaling_valid,
            physics_consistency_score=physics_consistency_score,
            consistency_issues=consistency_issues,
            physics_recommendations=physics_recommendations
        )
        
        self.logger.info(f"Physics consistency validation for {system_name}: "
                        f"score = {physics_consistency_score:.3f}, "
                        f"issues = {len(consistency_issues)}")
        
        return result
    
    def run_final_validation(self,
                           systems_data: Dict[str, Dict[str, Any]],
                           output_dir: str = "results/final_validation") -> FinalValidationReport:
        """
        Run comprehensive final validation across all systems.
        
        Args:
            systems_data: Dictionary containing data for all systems
            output_dir: Output directory for validation results
            
        Returns:
            FinalValidationReport with complete validation results
        """
        self.logger.info("Starting final comprehensive validation")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize result containers
        equilibration_results = {}
        data_quality_results = {}
        physics_consistency_results = {}
        
        # Validate each system
        for system_name, system_data in systems_data.items():
            self.logger.info(f"Validating system: {system_name}")
            
            # Equilibration validation
            if 'energy_series' in system_data:
                energy_series = system_data['energy_series']
                mag_series = system_data.get('magnetization_series')
                
                eq_result = self.validate_equilibration(
                    energy_series, mag_series, system_name
                )
                equilibration_results[system_name] = eq_result
            
            # Data quality validation
            if 'dataset_path' in system_data:
                dataset_path = system_data['dataset_path']
                system_config = system_data.get('config', {})
                
                dq_result = self.validate_data_quality(
                    dataset_path, system_config, system_name
                )
                data_quality_results[system_name] = dq_result
            
            # Physics consistency validation
            if 'analysis_results' in system_data:
                analysis_results = system_data['analysis_results']
                system_config = system_data.get('config', {})
                
                pc_result = self.validate_physics_consistency(
                    analysis_results, system_config, system_name
                )
                physics_consistency_results[system_name] = pc_result
        
        # Compute overall assessment
        all_systems_equilibrated = all(
            result.is_equilibrated for result in equilibration_results.values()
        )
        
        all_data_quality_passed = all(
            result.overall_quality_score >= self.quality_threshold 
            for result in data_quality_results.values()
        )
        
        all_physics_consistent = all(
            result.physics_consistency_score >= self.quality_threshold
            for result in physics_consistency_results.values()
        )
        
        # Summary metrics
        avg_equilibration_quality = np.mean([
            result.convergence_quality_score for result in equilibration_results.values()
        ]) if equilibration_results else 0.0
        
        avg_data_quality = np.mean([
            result.overall_quality_score for result in data_quality_results.values()
        ]) if data_quality_results else 0.0
        
        avg_physics_consistency = np.mean([
            result.physics_consistency_score for result in physics_consistency_results.values()
        ]) if physics_consistency_results else 0.0
        
        overall_validation_score = (
            avg_equilibration_quality * 0.3 +
            avg_data_quality * 0.4 +
            avg_physics_consistency * 0.3
        )
        
        # Generate recommendations
        critical_issues = []
        improvement_recommendations = []
        
        # Collect critical issues
        for system_name, result in equilibration_results.items():
            if not result.is_equilibrated:
                critical_issues.append(f"{system_name}: Equilibration failed")
        
        for system_name, result in data_quality_results.items():
            if result.overall_quality_score < self.quality_threshold:
                critical_issues.append(f"{system_name}: Data quality below threshold")
            critical_issues.extend([f"{system_name}: {issue}" for issue in result.quality_issues])
        
        for system_name, result in physics_consistency_results.items():
            if result.physics_consistency_score < self.quality_threshold:
                critical_issues.append(f"{system_name}: Physics consistency below threshold")
            critical_issues.extend([f"{system_name}: {issue}" for issue in result.consistency_issues])
        
        # Collect improvement recommendations
        for system_name, result in data_quality_results.items():
            improvement_recommendations.extend([
                f"{system_name}: {rec}" for rec in result.recommendations
            ])
        
        for system_name, result in physics_consistency_results.items():
            improvement_recommendations.extend([
                f"{system_name}: {rec}" for rec in result.physics_recommendations
            ])
        
        # Overall validation status
        validation_passed = (
            all_systems_equilibrated and
            all_data_quality_passed and
            all_physics_consistent and
            overall_validation_score >= self.quality_threshold
        )
        
        # Create final report
        final_report = FinalValidationReport(
            validation_timestamp=str(np.datetime64('now')),
            equilibration_results=equilibration_results,
            data_quality_results=data_quality_results,
            physics_consistency_results=physics_consistency_results,
            all_systems_equilibrated=all_systems_equilibrated,
            all_data_quality_passed=all_data_quality_passed,
            all_physics_consistent=all_physics_consistent,
            average_equilibration_quality=avg_equilibration_quality,
            average_data_quality=avg_data_quality,
            average_physics_consistency=avg_physics_consistency,
            overall_validation_score=overall_validation_score,
            critical_issues=critical_issues,
            improvement_recommendations=improvement_recommendations,
            validation_passed=validation_passed
        )
        
        # Save results
        self._save_final_validation_report(final_report, output_path)
        
        # Generate visualizations
        self._create_final_validation_visualizations(final_report, output_path)
        
        self.logger.info(f"Final validation completed: {'PASSED' if validation_passed else 'FAILED'}")
        self.logger.info(f"Overall validation score: {overall_validation_score:.3f}")
        
        return final_report
    
    def _save_final_validation_report(self,
                                    report: FinalValidationReport,
                                    output_path: Path):
        """Save final validation report to files."""
        
        # Save JSON report
        json_path = output_path / "final_validation_report.json"
        with open(json_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        # Save text summary
        text_path = output_path / "final_validation_summary.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("FINAL VALIDATION AND QUALITY ASSURANCE REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Validation Timestamp: {report.validation_timestamp}\n")
            f.write(f"Overall Validation: {'PASSED' if report.validation_passed else 'FAILED'}\n")
            f.write(f"Overall Score: {report.overall_validation_score:.3f}\n\n")
            
            f.write("SUMMARY ASSESSMENT\n")
            f.write("-" * 40 + "\n")
            f.write(f"All Systems Equilibrated: {report.all_systems_equilibrated}\n")
            f.write(f"All Data Quality Passed: {report.all_data_quality_passed}\n")
            f.write(f"All Physics Consistent: {report.all_physics_consistent}\n\n")
            
            f.write("AVERAGE SCORES\n")
            f.write("-" * 40 + "\n")
            f.write(f"Equilibration Quality: {report.average_equilibration_quality:.3f}\n")
            f.write(f"Data Quality: {report.average_data_quality:.3f}\n")
            f.write(f"Physics Consistency: {report.average_physics_consistency:.3f}\n\n")
            
            if report.critical_issues:
                f.write("CRITICAL ISSUES\n")
                f.write("-" * 40 + "\n")
                for issue in report.critical_issues:
                    f.write(f"• {issue}\n")
                f.write("\n")
            
            if report.improvement_recommendations:
                f.write("IMPROVEMENT RECOMMENDATIONS\n")
                f.write("-" * 40 + "\n")
                for rec in report.improvement_recommendations[:10]:  # Limit to top 10
                    f.write(f"• {rec}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
        
        self.logger.info(f"Final validation report saved to {output_path}")
    
    def _create_final_validation_visualizations(self,
                                              report: FinalValidationReport,
                                              output_path: Path):
        """Create comprehensive final validation visualizations."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Overall validation summary
        ax = axes[0, 0]
        
        categories = ['Equilibration', 'Data Quality', 'Physics Consistency']
        scores = [
            report.average_equilibration_quality,
            report.average_data_quality,
            report.average_physics_consistency
        ]
        
        bars = ax.bar(categories, scores, 
                     color=['green' if score >= 0.7 else 'orange' if score >= 0.5 else 'red' 
                           for score in scores])
        
        ax.axhline(0.7, color='black', linestyle='--', label='Quality Threshold')
        ax.set_ylabel('Average Score')
        ax.set_title('Final Validation Summary')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add score labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: System-wise validation status
        ax = axes[0, 1]
        
        system_names = list(report.equilibration_results.keys())
        if system_names:
            eq_status = [1 if report.equilibration_results[name].is_equilibrated else 0 
                        for name in system_names]
            dq_status = [1 if report.data_quality_results.get(name, type('', (), {'overall_quality_score': 0})).overall_quality_score >= 0.7 else 0
                        for name in system_names]
            pc_status = [1 if report.physics_consistency_results.get(name, type('', (), {'physics_consistency_score': 0})).physics_consistency_score >= 0.7 else 0
                        for name in system_names]
            
            x = np.arange(len(system_names))
            width = 0.25
            
            ax.bar(x - width, eq_status, width, label='Equilibration', alpha=0.8)
            ax.bar(x, dq_status, width, label='Data Quality', alpha=0.8)
            ax.bar(x + width, pc_status, width, label='Physics Consistency', alpha=0.8)
            
            ax.set_xlabel('System')
            ax.set_ylabel('Validation Status (1=Pass, 0=Fail)')
            ax.set_title('System-wise Validation Status')
            ax.set_xticks(x)
            ax.set_xticklabels(system_names, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Quality score distribution
        ax = axes[1, 0]
        
        all_scores = []
        score_labels = []
        
        for name in system_names:
            if name in report.equilibration_results:
                all_scores.append(report.equilibration_results[name].convergence_quality_score)
                score_labels.append(f"{name}_eq")
            
            if name in report.data_quality_results:
                all_scores.append(report.data_quality_results[name].overall_quality_score)
                score_labels.append(f"{name}_dq")
            
            if name in report.physics_consistency_results:
                all_scores.append(report.physics_consistency_results[name].physics_consistency_score)
                score_labels.append(f"{name}_pc")
        
        if all_scores:
            ax.hist(all_scores, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(0.7, color='red', linestyle='--', label='Quality Threshold')
            ax.axvline(np.mean(all_scores), color='green', linestyle='-',
                      label=f'Mean ({np.mean(all_scores):.3f})')
            ax.set_xlabel('Quality Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Quality Score Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Issues and recommendations summary
        ax = axes[1, 1]
        ax.axis('off')
        
        # Create summary text
        summary_text = f"FINAL VALIDATION SUMMARY\n\n"
        summary_text += f"Overall Status: {'✅ PASSED' if report.validation_passed else '❌ FAILED'}\n"
        summary_text += f"Overall Score: {report.overall_validation_score:.3f}\n\n"
        
        summary_text += f"System Status:\n"
        summary_text += f"• Equilibrated: {report.all_systems_equilibrated}\n"
        summary_text += f"• Data Quality: {report.all_data_quality_passed}\n"
        summary_text += f"• Physics Consistent: {report.all_physics_consistent}\n\n"
        
        summary_text += f"Critical Issues: {len(report.critical_issues)}\n"
        for issue in report.critical_issues[:3]:
            summary_text += f"• {issue[:40]}...\n"
        
        if len(report.critical_issues) > 3:
            summary_text += f"• ... and {len(report.critical_issues) - 3} more\n"
        
        summary_text += f"\nRecommendations: {len(report.improvement_recommendations)}\n"
        for rec in report.improvement_recommendations[:3]:
            summary_text += f"• {rec[:40]}...\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        fig.savefig(output_path / "final_validation_summary.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


def create_final_validation_system(equilibration_threshold: float = 1e-4,
                                 autocorr_window_factor: int = 10,
                                 min_equilibration_steps: int = 1000,
                                 quality_threshold: float = 0.7,
                                 random_seed: Optional[int] = None) -> FinalValidationSystem:
    """
    Factory function to create FinalValidationSystem.
    
    Args:
        equilibration_threshold: Threshold for energy convergence
        autocorr_window_factor: Factor for autocorrelation window size
        min_equilibration_steps: Minimum steps for equilibration
        quality_threshold: Minimum quality score threshold
        random_seed: Random seed for reproducibility
        
    Returns:
        Configured FinalValidationSystem instance
    """
    return FinalValidationSystem(
        equilibration_threshold=equilibration_threshold,
        autocorr_window_factor=autocorr_window_factor,
        min_equilibration_steps=min_equilibration_steps,
        quality_threshold=quality_threshold,
        random_seed=random_seed
    )