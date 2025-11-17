"""
Baseline Accuracy Measurement System

This module implements task 14.1: Measure baseline accuracy with real Monte Carlo data.
Tests critical exponent extraction on actual 3D Ising Monte Carlo simulations and
compares raw magnetization vs VAE-enhanced approaches on identical datasets.
Documents realistic accuracy expectations (likely 40-70% range, not 98%).
"""

import numpy as np
import torch
import h5py
import json
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import warnings
import logging
from scipy.stats import pearsonr, spearmanr

# Import real components
from ..training.real_vae_training_pipeline import (
    RealVAETrainingPipeline, RealVAETrainingConfig, create_real_vae_training_pipeline,
    load_physics_data_from_file
)
from ..analysis.blind_critical_exponent_extractor import (
    BlindCriticalExponentExtractor, create_blind_critical_exponent_extractor
)
from ..analysis.robust_critical_exponent_extractor import RobustCriticalExponentExtractor
from ..data.enhanced_monte_carlo import EnhancedMonteCarloSimulator
from ..models.physics_models import Ising3DModel

# Simple logger function
def get_logger(name):
    return logging.getLogger(name)


@dataclass
class BaselineAccuracyConfig:
    """Configuration for baseline accuracy measurement."""
    
    # Data parameters
    use_existing_data: bool = True
    data_file_path: Optional[str] = None
    generate_new_data: bool = False
    
    # Monte Carlo parameters (if generating new data)
    system_sizes: List[int] = None
    temperature_range: Tuple[float, float] = (3.5, 5.5)
    n_temperatures: int = 40
    n_configs_per_temp: int = 200
    equilibration_steps: int = 50000
    sampling_interval: int = 100
    
    # VAE training parameters
    vae_epochs: int = 100
    vae_batch_size: int = 64
    vae_learning_rate: float = 1e-3
    vae_beta: float = 1.0
    use_physics_informed_loss: bool = True
    
    # Analysis parameters
    bootstrap_samples: int = 500
    confidence_level: float = 0.95
    random_seed: Optional[int] = 42
    
    # Comparison parameters
    compare_system_sizes: bool = True
    compare_temperature_ranges: bool = True
    compare_data_quality: bool = True
    
    # Output parameters
    save_results: bool = True
    results_dir: str = 'results/baseline_accuracy'
    create_visualizations: bool = True
    
    def __post_init__(self):
        if self.system_sizes is None:
            self.system_sizes = [16, 32]


@dataclass
class MethodAccuracyResults:
    """Results for a single method (VAE or raw magnetization)."""
    method_name: str
    
    # Critical temperature
    tc_measured: float
    tc_theoretical: float
    tc_error_percent: float
    tc_confidence: float
    
    # Critical exponents
    beta_measured: Optional[float] = None
    beta_error_percent: Optional[float] = None
    beta_r_squared: Optional[float] = None
    beta_confidence_interval: Optional[Tuple[float, float]] = None
    
    nu_measured: Optional[float] = None
    nu_error_percent: Optional[float] = None
    nu_r_squared: Optional[float] = None
    nu_confidence_interval: Optional[Tuple[float, float]] = None
    
    # Overall metrics
    overall_accuracy: float = 0.0
    extraction_success: bool = False
    data_quality_score: float = 0.0
    
    # Method-specific metrics
    method_specific_metrics: Dict[str, Any] = None
    
    # Performance metrics
    computation_time: float = 0.0
    memory_usage: float = 0.0


@dataclass
class BaselineAccuracyResults:
    """Complete baseline accuracy measurement results."""
    
    # Configuration used
    config: BaselineAccuracyConfig
    
    # Data information
    data_source: str
    n_samples: int
    temperature_range: Tuple[float, float]
    system_sizes_tested: List[int]
    
    # Method results (required fields)
    vae_results: MethodAccuracyResults
    raw_magnetization_results: MethodAccuracyResults
    
    # Theoretical values (with defaults)
    theoretical_tc: float = 4.511
    theoretical_beta: float = 0.326
    theoretical_nu: float = 0.630
    
    # Comparative analysis (with defaults)
    vae_improvement_percent: float = 0.0
    better_method: str = 'unknown'
    accuracy_difference: float = 0.0
    
    # Data quality assessment (with defaults)
    data_quality_metrics: Optional[Dict[str, float]] = None
    
    # System size analysis
    system_size_analysis: Optional[Dict[int, Dict[str, Any]]] = None
    
    # Realistic expectations
    realistic_accuracy_range: Tuple[float, float] = (40.0, 70.0)
    meets_realistic_expectations: bool = False
    
    # Overall assessment
    total_runtime: float = 0.0
    assessment_grade: str = 'F'
    key_findings: Optional[List[str]] = None


class BaselineAccuracyMeasurement:
    """Main class for measuring baseline accuracy with real Monte Carlo data."""
    
    def __init__(self, config: BaselineAccuracyConfig):
        """Initialize baseline accuracy measurement system."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Set random seed for reproducibility
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
            torch.manual_seed(config.random_seed)
        
        # Initialize components
        self.vae_trainer = None
        self.blind_extractor = None
        self.robust_extractor = None
        
        self.logger.info("Baseline accuracy measurement system initialized")
    
    def measure_baseline_accuracy(self) -> BaselineAccuracyResults:
        """
        Measure baseline accuracy using real Monte Carlo data.
        
        Returns:
            BaselineAccuracyResults with comprehensive accuracy assessment
        """
        self.logger.info("Starting baseline accuracy measurement")
        start_time = time.time()
        
        # Create results directory
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Load or generate data
            self.logger.info("Loading/generating Monte Carlo data")
            data_info = self._load_or_generate_data()
            
            # Step 2: Assess data quality
            self.logger.info("Assessing data quality")
            data_quality_metrics = self._assess_data_quality(data_info)
            
            # Step 3: Test VAE-enhanced approach
            self.logger.info("Testing VAE-enhanced approach")
            vae_results = self._test_vae_approach(data_info)
            
            # Step 4: Test raw magnetization approach
            self.logger.info("Testing raw magnetization approach")
            raw_results = self._test_raw_magnetization_approach(data_info)
            
            # Step 5: Perform system size analysis (if multiple sizes)
            system_size_analysis = None
            if len(self.config.system_sizes) > 1:
                self.logger.info("Performing system size analysis")
                system_size_analysis = self._analyze_system_size_effects(data_info)
            
            # Step 6: Compute comparative analysis
            comparative_analysis = self._compute_comparative_analysis(vae_results, raw_results)
            
            # Step 7: Assess realistic expectations
            realistic_assessment = self._assess_realistic_expectations(vae_results, raw_results)
            
            total_time = time.time() - start_time
            
            # Create results object
            results = BaselineAccuracyResults(
                config=self.config,
                data_source=data_info['source'],
                n_samples=data_info['n_samples'],
                temperature_range=data_info['temperature_range'],
                system_sizes_tested=data_info['system_sizes'],
                vae_results=vae_results,
                raw_magnetization_results=raw_results,
                vae_improvement_percent=comparative_analysis['improvement_percent'],
                better_method=comparative_analysis['better_method'],
                accuracy_difference=comparative_analysis['accuracy_difference'],
                data_quality_metrics=data_quality_metrics,
                system_size_analysis=system_size_analysis,
                realistic_accuracy_range=realistic_assessment['accuracy_range'],
                meets_realistic_expectations=realistic_assessment['meets_expectations'],
                total_runtime=total_time,
                assessment_grade=realistic_assessment['grade'],
                key_findings=realistic_assessment['key_findings']
            )
            
            # Save results
            if self.config.save_results:
                self._save_results(results, results_dir)
            
            # Create visualizations
            if self.config.create_visualizations:
                self._create_visualizations(results, results_dir)
            
            self.logger.info(f"Baseline accuracy measurement completed in {total_time:.2f} seconds")
            self._log_summary(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Baseline accuracy measurement failed: {e}")
            raise
    
    def _load_or_generate_data(self) -> Dict[str, Any]:
        """Load existing data or generate new Monte Carlo data."""
        
        if self.config.use_existing_data and self.config.data_file_path:
            # Load existing data
            self.logger.info(f"Loading existing data from {self.config.data_file_path}")
            
            try:
                configurations, temperatures, magnetizations, energies = load_physics_data_from_file(
                    self.config.data_file_path
                )
                
                # Determine system sizes from configuration shapes
                if len(configurations.shape) == 4:  # 2D: (N, H, W)
                    system_sizes = [configurations.shape[1]]
                elif len(configurations.shape) == 5:  # 3D: (N, D, H, W)
                    system_sizes = [configurations.shape[1]]
                else:
                    system_sizes = [32]  # Default assumption
                
                return {
                    'source': f'existing_file:{self.config.data_file_path}',
                    'configurations': configurations,
                    'temperatures': temperatures,
                    'magnetizations': magnetizations,
                    'energies': energies,
                    'n_samples': len(configurations),
                    'temperature_range': (np.min(temperatures), np.max(temperatures)),
                    'system_sizes': system_sizes
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to load existing data: {e}")
                self.logger.info("Falling back to generating new data")
        
        # Generate new data
        self.logger.info("Generating new Monte Carlo data")
        return self._generate_new_monte_carlo_data()
    
    def _generate_new_monte_carlo_data(self) -> Dict[str, Any]:
        """Generate new high-quality Monte Carlo data."""
        
        # Create 3D Ising model
        model = Ising3DModel()
        
        # Use largest system size for generation
        system_size = max(self.config.system_sizes)
        
        # Create enhanced Monte Carlo simulator
        simulator = EnhancedMonteCarloSimulator(
            model=model,
            system_size=system_size,
            random_seed=self.config.random_seed
        )
        
        # Generate temperature array with high density near Tc
        tc_theoretical = 4.511
        temp_min, temp_max = self.config.temperature_range
        
        # Create temperature schedule with emphasis near Tc
        n_temps = self.config.n_temperatures
        temp_low = np.linspace(temp_min, tc_theoretical - 0.3, n_temps // 3)
        temp_critical = np.linspace(tc_theoretical - 0.3, tc_theoretical + 0.3, n_temps // 3)
        temp_high = np.linspace(tc_theoretical + 0.3, temp_max, n_temps // 3)
        temperatures_schedule = np.concatenate([temp_low, temp_critical, temp_high])
        
        # Generate configurations
        all_configurations = []
        all_temperatures = []
        all_magnetizations = []
        all_energies = []
        
        for temp in temperatures_schedule:
            self.logger.info(f"Generating data at T = {temp:.3f}")
            
            # Equilibrate
            equilibrated_config = simulator.equilibrate(
                temperature=temp,
                n_steps=self.config.equilibration_steps
            )
            
            # Generate configurations
            configs, temps, mags, energies = simulator.generate_configurations(
                temperature=temp,
                n_configurations=self.config.n_configs_per_temp,
                sampling_interval=self.config.sampling_interval,
                initial_config=equilibrated_config
            )
            
            all_configurations.extend(configs)
            all_temperatures.extend(temps)
            all_magnetizations.extend(mags)
            all_energies.extend(energies)
        
        # Convert to arrays
        configurations = np.array(all_configurations)
        temperatures = np.array(all_temperatures)
        magnetizations = np.array(all_magnetizations)
        energies = np.array(all_energies)
        
        self.logger.info(f"Generated {len(configurations)} configurations")
        self.logger.info(f"Temperature range: {np.min(temperatures):.3f} - {np.max(temperatures):.3f}")
        self.logger.info(f"Magnetization range: {np.min(magnetizations):.3f} - {np.max(magnetizations):.3f}")
        
        return {
            'source': 'newly_generated',
            'configurations': configurations,
            'temperatures': temperatures,
            'magnetizations': magnetizations,
            'energies': energies,
            'n_samples': len(configurations),
            'temperature_range': (np.min(temperatures), np.max(temperatures)),
            'system_sizes': [system_size]
        }
    
    def _assess_data_quality(self, data_info: Dict[str, Any]) -> Dict[str, float]:
        """Assess the quality of the Monte Carlo data."""
        
        temperatures = data_info['temperatures']
        magnetizations = data_info['magnetizations']
        energies = data_info['energies']
        
        quality_metrics = {}
        
        # 1. Temperature coverage
        temp_min, temp_max = data_info['temperature_range']
        tc_theoretical = 4.511
        
        if temp_min <= tc_theoretical <= temp_max:
            temp_coverage = 1.0
        else:
            temp_coverage = 0.0
        
        quality_metrics['temperature_coverage'] = temp_coverage
        
        # 2. Phase transition visibility
        unique_temps = np.unique(temperatures)
        temp_magnetizations = []
        
        for temp in unique_temps:
            temp_mask = temperatures == temp
            if np.sum(temp_mask) > 0:
                temp_magnetizations.append(np.mean(np.abs(magnetizations[temp_mask])))
        
        if len(temp_magnetizations) > 5:
            mag_range = np.max(temp_magnetizations) - np.min(temp_magnetizations)
            phase_transition_visibility = min(1.0, mag_range / 0.5)  # Expect ~0.5 range
        else:
            phase_transition_visibility = 0.0
        
        quality_metrics['phase_transition_visibility'] = phase_transition_visibility
        
        # 3. Data density near critical temperature
        critical_region_mask = np.abs(temperatures - tc_theoretical) < 0.5
        critical_density = np.sum(critical_region_mask) / len(temperatures)
        quality_metrics['critical_region_density'] = min(1.0, critical_density / 0.3)  # Expect ~30%
        
        # 4. Magnetization quality
        mag_std = np.std(magnetizations)
        mag_mean = np.mean(np.abs(magnetizations))
        
        if mag_mean > 0:
            mag_signal_to_noise = mag_mean / (mag_std + 1e-10)
            quality_metrics['magnetization_signal_to_noise'] = min(1.0, mag_signal_to_noise / 2.0)
        else:
            quality_metrics['magnetization_signal_to_noise'] = 0.0
        
        # 5. Energy consistency
        if energies is not None:
            energy_temp_corr, _ = pearsonr(energies, temperatures)
            quality_metrics['energy_temperature_correlation'] = abs(energy_temp_corr)
        else:
            quality_metrics['energy_temperature_correlation'] = 0.0
        
        # 6. Sample size adequacy
        n_samples = data_info['n_samples']
        sample_adequacy = min(1.0, n_samples / 1000)  # Expect at least 1000 samples
        quality_metrics['sample_size_adequacy'] = sample_adequacy
        
        # Overall data quality score
        quality_metrics['overall_data_quality'] = np.mean(list(quality_metrics.values()))
        
        self.logger.info(f"Data quality assessment:")
        for metric, value in quality_metrics.items():
            self.logger.info(f"  {metric}: {value:.3f}")
        
        return quality_metrics
    
    def _test_vae_approach(self, data_info: Dict[str, Any]) -> MethodAccuracyResults:
        """Test VAE-enhanced critical exponent extraction."""
        
        start_time = time.time()
        
        try:
            # Create VAE training configuration
            vae_config = RealVAETrainingConfig(
                batch_size=self.config.vae_batch_size,
                learning_rate=self.config.vae_learning_rate,
                num_epochs=self.config.vae_epochs,
                beta=self.config.vae_beta,
                use_physics_informed_loss=self.config.use_physics_informed_loss,
                validation_split=0.2,
                test_split=0.2,
                random_seed=self.config.random_seed
            )
            
            # Create and train VAE
            vae_trainer = create_real_vae_training_pipeline(vae_config)
            
            vae_training_results = vae_trainer.train(
                data_info['configurations'],
                data_info['temperatures'],
                data_info['magnetizations'],
                data_info['energies']
            )
            
            # Extract latent representations
            latent_representations = self._extract_latent_representations(
                vae_trainer, data_info
            )
            
            # Perform blind critical exponent extraction
            blind_extractor = create_blind_critical_exponent_extractor(
                bootstrap_samples=self.config.bootstrap_samples,
                random_seed=self.config.random_seed
            )
            
            extraction_results = blind_extractor.extract_critical_exponents_blind(
                latent_representations=latent_representations,
                temperatures=data_info['temperatures'],
                magnetizations=data_info['magnetizations'],
                system_identifier='ising_3d'
            )
            
            # Compute accuracy metrics
            tc_error = abs(extraction_results.tc_detection.critical_temperature - 4.511) / 4.511 * 100
            
            beta_measured = None
            beta_error = None
            beta_r_squared = None
            beta_ci = None
            
            if extraction_results.beta_exponent:
                beta_measured = extraction_results.beta_exponent.exponent
                beta_error = abs(beta_measured - 0.326) / 0.326 * 100
                beta_r_squared = extraction_results.beta_exponent.r_squared
                beta_ci = (extraction_results.beta_exponent.exponent_ci_lower,
                          extraction_results.beta_exponent.exponent_ci_upper)
            
            # Overall accuracy (weighted average)
            accuracy_components = []
            
            # Tc accuracy (20% weight)
            tc_accuracy = max(0, 100 - tc_error)
            accuracy_components.append(('tc', tc_accuracy, 0.2))
            
            # Beta accuracy (40% weight)
            if beta_error is not None:
                beta_accuracy = max(0, 100 - beta_error)
                accuracy_components.append(('beta', beta_accuracy, 0.4))
            
            # Order parameter quality (40% weight)
            op_quality = extraction_results.order_parameter_analysis.selection_confidence * 100
            accuracy_components.append(('order_param', op_quality, 0.4))
            
            # Compute weighted average
            total_weight = sum(weight for _, _, weight in accuracy_components)
            weighted_sum = sum(acc * weight for _, acc, weight in accuracy_components)
            overall_accuracy = weighted_sum / total_weight if total_weight > 0 else 0.0
            
            computation_time = time.time() - start_time
            
            # Method-specific metrics
            method_metrics = {
                'latent_magnetization_correlation': vae_training_results.latent_magnetization_correlation,
                'reconstruction_quality': vae_training_results.reconstruction_quality,
                'order_parameter_dimension': extraction_results.order_parameter_analysis.selected_dimension,
                'order_parameter_confidence': extraction_results.order_parameter_analysis.selection_confidence,
                'tc_detection_confidence': extraction_results.tc_detection.detection_confidence,
                'extraction_quality_score': extraction_results.extraction_quality_score
            }
            
            return MethodAccuracyResults(
                method_name='VAE-enhanced',
                tc_measured=extraction_results.tc_detection.critical_temperature,
                tc_theoretical=4.511,
                tc_error_percent=tc_error,
                tc_confidence=extraction_results.tc_detection.detection_confidence,
                beta_measured=beta_measured,
                beta_error_percent=beta_error,
                beta_r_squared=beta_r_squared,
                beta_confidence_interval=beta_ci,
                overall_accuracy=overall_accuracy,
                extraction_success=extraction_results.beta_exponent is not None,
                data_quality_score=extraction_results.extraction_quality_score,
                method_specific_metrics=method_metrics,
                computation_time=computation_time
            )
            
        except Exception as e:
            self.logger.error(f"VAE approach failed: {e}")
            
            return MethodAccuracyResults(
                method_name='VAE-enhanced',
                tc_measured=0.0,
                tc_theoretical=4.511,
                tc_error_percent=100.0,
                tc_confidence=0.0,
                overall_accuracy=0.0,
                extraction_success=False,
                data_quality_score=0.0,
                computation_time=time.time() - start_time
            )
    
    def _test_raw_magnetization_approach(self, data_info: Dict[str, Any]) -> MethodAccuracyResults:
        """Test raw magnetization critical exponent extraction."""
        
        start_time = time.time()
        
        try:
            # Create robust extractor for raw magnetization
            self.robust_extractor = RobustCriticalExponentExtractor(
                bootstrap_samples=self.config.bootstrap_samples,
                random_seed=self.config.random_seed
            )
            
            # Perform extraction using raw magnetization
            extraction_results = self.robust_extractor.extract_critical_exponents_robust(
                temperatures=data_info['temperatures'],
                order_parameter=np.abs(data_info['magnetizations']),
                system_type='ising_3d'
            )
            
            # Compute accuracy metrics
            tc_error = abs(extraction_results.critical_temperature - 4.511) / 4.511 * 100
            
            beta_measured = None
            beta_error = None
            beta_r_squared = None
            beta_ci = None
            
            if extraction_results.beta_result:
                beta_measured = extraction_results.beta_result.exponent
                beta_error = abs(beta_measured - 0.326) / 0.326 * 100
                beta_r_squared = extraction_results.beta_result.r_squared
                beta_ci = (extraction_results.beta_result.confidence_interval_lower,
                          extraction_results.beta_result.confidence_interval_upper)
            
            # Overall accuracy (simpler for raw magnetization)
            accuracy_components = []
            
            # Tc accuracy (30% weight)
            tc_accuracy = max(0, 100 - tc_error)
            accuracy_components.append(('tc', tc_accuracy, 0.3))
            
            # Beta accuracy (70% weight)
            if beta_error is not None:
                beta_accuracy = max(0, 100 - beta_error)
                accuracy_components.append(('beta', beta_accuracy, 0.7))
            
            # Compute weighted average
            total_weight = sum(weight for _, _, weight in accuracy_components)
            weighted_sum = sum(acc * weight for _, acc, weight in accuracy_components)
            overall_accuracy = weighted_sum / total_weight if total_weight > 0 else 0.0
            
            computation_time = time.time() - start_time
            
            # Method-specific metrics
            method_metrics = {
                'tc_detection_method': extraction_results.tc_detection_method,
                'fitting_quality': extraction_results.beta_result.fit_quality if extraction_results.beta_result else 0.0,
                'data_points_used': extraction_results.beta_result.n_data_points if extraction_results.beta_result else 0
            }
            
            return MethodAccuracyResults(
                method_name='Raw magnetization',
                tc_measured=extraction_results.critical_temperature,
                tc_theoretical=4.511,
                tc_error_percent=tc_error,
                tc_confidence=extraction_results.tc_confidence,
                beta_measured=beta_measured,
                beta_error_percent=beta_error,
                beta_r_squared=beta_r_squared,
                beta_confidence_interval=beta_ci,
                overall_accuracy=overall_accuracy,
                extraction_success=extraction_results.beta_result is not None,
                data_quality_score=extraction_results.overall_quality_score,
                method_specific_metrics=method_metrics,
                computation_time=computation_time
            )
            
        except Exception as e:
            self.logger.error(f"Raw magnetization approach failed: {e}")
            
            return MethodAccuracyResults(
                method_name='Raw magnetization',
                tc_measured=0.0,
                tc_theoretical=4.511,
                tc_error_percent=100.0,
                tc_confidence=0.0,
                overall_accuracy=0.0,
                extraction_success=False,
                data_quality_score=0.0,
                computation_time=time.time() - start_time
            )
    
    def _extract_latent_representations(self, vae_trainer, data_info: Dict[str, Any]) -> np.ndarray:
        """Extract latent representations from trained VAE."""
        
        # Load trained model
        model = vae_trainer.model
        model.eval()
        
        # Prepare data
        configurations = torch.FloatTensor(data_info['configurations'])
        
        # Add channel dimension if needed
        if len(configurations.shape) == 3:  # 2D: (N, H, W)
            configurations = configurations.unsqueeze(1)  # (N, 1, H, W)
        elif len(configurations.shape) == 4:  # 3D: (N, D, H, W)
            configurations = configurations.unsqueeze(1)  # (N, 1, D, H, W)
        
        configurations = configurations.to(vae_trainer.device)
        
        # Extract latent representations
        latent_representations = []
        
        with torch.no_grad():
            batch_size = 64
            for i in range(0, len(configurations), batch_size):
                batch = configurations[i:i + batch_size]
                z, mu, logvar = model.encode(batch)
                latent_representations.append(z.cpu().numpy())
        
        return np.concatenate(latent_representations, axis=0)
    
    def _analyze_system_size_effects(self, data_info: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """Analyze accuracy across different system sizes."""
        
        # This is a placeholder for system size analysis
        # In practice, would need data for multiple system sizes
        
        analysis = {}
        
        for size in self.config.system_sizes:
            analysis[size] = {
                'accuracy_vae': 0.0,  # Would compute actual accuracy
                'accuracy_raw': 0.0,
                'data_quality': 0.0,
                'finite_size_effects': 0.0
            }
        
        return analysis
    
    def _compute_comparative_analysis(self, 
                                    vae_results: MethodAccuracyResults,
                                    raw_results: MethodAccuracyResults) -> Dict[str, Any]:
        """Compute comparative analysis between methods."""
        
        # Improvement calculation
        improvement_percent = vae_results.overall_accuracy - raw_results.overall_accuracy
        
        # Determine better method
        if vae_results.overall_accuracy > raw_results.overall_accuracy:
            better_method = 'VAE-enhanced'
        elif raw_results.overall_accuracy > vae_results.overall_accuracy:
            better_method = 'Raw magnetization'
        else:
            better_method = 'Tie'
        
        # Accuracy difference
        accuracy_difference = abs(vae_results.overall_accuracy - raw_results.overall_accuracy)
        
        return {
            'improvement_percent': improvement_percent,
            'better_method': better_method,
            'accuracy_difference': accuracy_difference
        }
    
    def _assess_realistic_expectations(self, 
                                     vae_results: MethodAccuracyResults,
                                     raw_results: MethodAccuracyResults) -> Dict[str, Any]:
        """Assess results against realistic expectations."""
        
        # Realistic accuracy range for critical exponent extraction
        realistic_range = (40.0, 70.0)
        
        # Check if either method meets realistic expectations
        vae_meets = realistic_range[0] <= vae_results.overall_accuracy <= realistic_range[1]
        raw_meets = realistic_range[0] <= raw_results.overall_accuracy <= realistic_range[1]
        meets_expectations = vae_meets or raw_meets
        
        # Determine grade
        best_accuracy = max(vae_results.overall_accuracy, raw_results.overall_accuracy)
        
        if best_accuracy >= 80:
            grade = 'A'
        elif best_accuracy >= 70:
            grade = 'B'
        elif best_accuracy >= 60:
            grade = 'C'
        elif best_accuracy >= 50:
            grade = 'D'
        else:
            grade = 'F'
        
        # Key findings
        key_findings = []
        
        if vae_results.overall_accuracy > raw_results.overall_accuracy:
            key_findings.append(f"VAE approach shows {vae_results.overall_accuracy - raw_results.overall_accuracy:.1f}% improvement")
        elif raw_results.overall_accuracy > vae_results.overall_accuracy:
            key_findings.append(f"Raw magnetization approach performs {raw_results.overall_accuracy - vae_results.overall_accuracy:.1f}% better")
        else:
            key_findings.append("Both methods show similar performance")
        
        if best_accuracy < realistic_range[0]:
            key_findings.append(f"Accuracy ({best_accuracy:.1f}%) below realistic expectations ({realistic_range[0]}-{realistic_range[1]}%)")
        elif best_accuracy > realistic_range[1]:
            key_findings.append(f"Accuracy ({best_accuracy:.1f}%) exceeds realistic expectations")
        else:
            key_findings.append(f"Accuracy ({best_accuracy:.1f}%) within realistic range")
        
        if vae_results.extraction_success and not raw_results.extraction_success:
            key_findings.append("VAE approach more robust - raw magnetization failed")
        elif raw_results.extraction_success and not vae_results.extraction_success:
            key_findings.append("Raw magnetization more robust - VAE approach failed")
        
        return {
            'accuracy_range': realistic_range,
            'meets_expectations': meets_expectations,
            'grade': grade,
            'key_findings': key_findings
        }
    
    def _save_results(self, results: BaselineAccuracyResults, results_dir: Path):
        """Save baseline accuracy results to files."""
        
        # Convert to dictionary for JSON serialization
        results_dict = asdict(results)
        
        # Save main results
        results_file = results_dir / 'baseline_accuracy_results.json'
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {results_file}")
    
    def _create_visualizations(self, results: BaselineAccuracyResults, results_dir: Path):
        """Create visualization plots for baseline accuracy results."""
        
        try:
            # Create comparison figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Overall accuracy comparison
            ax = axes[0, 0]
            methods = ['VAE-enhanced', 'Raw magnetization']
            accuracies = [results.vae_results.overall_accuracy, results.raw_magnetization_results.overall_accuracy]
            
            bars = ax.bar(methods, accuracies, color=['red', 'blue'], alpha=0.7)
            ax.set_ylabel('Overall Accuracy (%)')
            ax.set_title('Method Comparison: Overall Accuracy')
            ax.set_ylim(0, 100)
            
            # Add realistic expectation band
            ax.axhspan(results.realistic_accuracy_range[0], results.realistic_accuracy_range[1], 
                      alpha=0.2, color='green', label='Realistic Range')
            
            # Add accuracy values on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax.legend()
            
            # Plot 2: Critical temperature accuracy
            ax = axes[0, 1]
            tc_errors = [results.vae_results.tc_error_percent, results.raw_magnetization_results.tc_error_percent]
            
            bars = ax.bar(methods, tc_errors, color=['red', 'blue'], alpha=0.7)
            ax.set_ylabel('Critical Temperature Error (%)')
            ax.set_title('Critical Temperature Detection Accuracy')
            
            for bar, error in zip(bars, tc_errors):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{error:.2f}%', ha='center', va='bottom', fontweight='bold')
            
            # Plot 3: Beta exponent accuracy
            ax = axes[1, 0]
            beta_accuracies = []
            
            if results.vae_results.beta_error_percent is not None:
                beta_accuracies.append(max(0, 100 - results.vae_results.beta_error_percent))
            else:
                beta_accuracies.append(0)
            
            if results.raw_magnetization_results.beta_error_percent is not None:
                beta_accuracies.append(max(0, 100 - results.raw_magnetization_results.beta_error_percent))
            else:
                beta_accuracies.append(0)
            
            bars = ax.bar(methods, beta_accuracies, color=['red', 'blue'], alpha=0.7)
            ax.set_ylabel('β Exponent Accuracy (%)')
            ax.set_title('β Exponent Extraction Accuracy')
            ax.set_ylim(0, 100)
            
            for bar, acc in zip(bars, beta_accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # Plot 4: Performance summary
            ax = axes[1, 1]
            ax.axis('off')
            
            summary_text = f"""Baseline Accuracy Assessment Summary

Data Source: {results.data_source}
Samples: {results.n_samples:,}
Temperature Range: {results.temperature_range[0]:.2f} - {results.temperature_range[1]:.2f}

VAE-Enhanced Method:
  Overall Accuracy: {results.vae_results.overall_accuracy:.1f}%
  Tc Error: {results.vae_results.tc_error_percent:.2f}%
  β Accuracy: {beta_accuracies[0]:.1f}%
  Success: {'Yes' if results.vae_results.extraction_success else 'No'}

Raw Magnetization Method:
  Overall Accuracy: {results.raw_magnetization_results.overall_accuracy:.1f}%
  Tc Error: {results.raw_magnetization_results.tc_error_percent:.2f}%
  β Accuracy: {beta_accuracies[1]:.1f}%
  Success: {'Yes' if results.raw_magnetization_results.extraction_success else 'No'}

Assessment:
  Better Method: {results.better_method}
  Improvement: {results.vae_improvement_percent:+.1f}%
  Grade: {results.assessment_grade}
  Meets Expectations: {'Yes' if results.meets_realistic_expectations else 'No'}

Runtime: {results.total_runtime:.1f}s
"""
            
            ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            
            # Save figure
            fig_path = results_dir / 'baseline_accuracy_comparison.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Visualization saved to {fig_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create visualizations: {e}")
    
    def _log_summary(self, results: BaselineAccuracyResults):
        """Log summary of baseline accuracy results."""
        
        self.logger.info("=" * 60)
        self.logger.info("BASELINE ACCURACY MEASUREMENT SUMMARY")
        self.logger.info("=" * 60)
        
        self.logger.info(f"Data: {results.data_source}")
        self.logger.info(f"Samples: {results.n_samples:,}")
        self.logger.info(f"Temperature range: {results.temperature_range[0]:.2f} - {results.temperature_range[1]:.2f}")
        
        self.logger.info(f"\nVAE-Enhanced Method:")
        self.logger.info(f"  Overall accuracy: {results.vae_results.overall_accuracy:.1f}%")
        self.logger.info(f"  Tc error: {results.vae_results.tc_error_percent:.2f}%")
        if results.vae_results.beta_error_percent is not None:
            self.logger.info(f"  β error: {results.vae_results.beta_error_percent:.1f}%")
        self.logger.info(f"  Success: {'Yes' if results.vae_results.extraction_success else 'No'}")
        
        self.logger.info(f"\nRaw Magnetization Method:")
        self.logger.info(f"  Overall accuracy: {results.raw_magnetization_results.overall_accuracy:.1f}%")
        self.logger.info(f"  Tc error: {results.raw_magnetization_results.tc_error_percent:.2f}%")
        if results.raw_magnetization_results.beta_error_percent is not None:
            self.logger.info(f"  β error: {results.raw_magnetization_results.beta_error_percent:.1f}%")
        self.logger.info(f"  Success: {'Yes' if results.raw_magnetization_results.extraction_success else 'No'}")
        
        self.logger.info(f"\nComparative Assessment:")
        self.logger.info(f"  Better method: {results.better_method}")
        self.logger.info(f"  VAE improvement: {results.vae_improvement_percent:+.1f}%")
        self.logger.info(f"  Assessment grade: {results.assessment_grade}")
        self.logger.info(f"  Meets realistic expectations: {'Yes' if results.meets_realistic_expectations else 'No'}")
        
        self.logger.info(f"\nKey Findings:")
        for finding in results.key_findings:
            self.logger.info(f"  • {finding}")
        
        self.logger.info(f"\nRealistic accuracy range: {results.realistic_accuracy_range[0]:.0f}-{results.realistic_accuracy_range[1]:.0f}%")
        self.logger.info(f"Runtime: {results.total_runtime:.1f} seconds")


def create_baseline_accuracy_measurement(config: Optional[BaselineAccuracyConfig] = None) -> BaselineAccuracyMeasurement:
    """
    Factory function to create baseline accuracy measurement system.
    
    Args:
        config: Configuration for baseline accuracy measurement
        
    Returns:
        Configured BaselineAccuracyMeasurement instance
    """
    if config is None:
        config = BaselineAccuracyConfig()
    
    return BaselineAccuracyMeasurement(config)


def run_baseline_accuracy_example(data_file_path: Optional[str] = None):
    """
    Example function to run baseline accuracy measurement.
    
    Args:
        data_file_path: Optional path to existing data file
    """
    # Create configuration
    config = BaselineAccuracyConfig(
        use_existing_data=data_file_path is not None,
        data_file_path=data_file_path,
        generate_new_data=data_file_path is None,
        vae_epochs=50,  # Reduced for example
        bootstrap_samples=200,  # Reduced for speed
        save_results=True,
        create_visualizations=True
    )
    
    # Create and run measurement
    measurement = create_baseline_accuracy_measurement(config)
    results = measurement.measure_baseline_accuracy()
    
    print(f"\nBaseline Accuracy Measurement Results:")
    print(f"VAE Overall Accuracy: {results.vae_results.overall_accuracy:.1f}%")
    print(f"Raw Magnetization Accuracy: {results.raw_magnetization_results.overall_accuracy:.1f}%")
    print(f"Better Method: {results.better_method}")
    print(f"Assessment Grade: {results.assessment_grade}")
    print(f"Meets Realistic Expectations: {'Yes' if results.meets_realistic_expectations else 'No'}")
    
    return results