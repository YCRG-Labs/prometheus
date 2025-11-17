"""
Real Accuracy Validation Pipeline

This module implements task 13.5: Updated accuracy validation pipeline that uses
real VAE training instead of mocks, removes hardcoded theoretical values from
analysis code, and implements proper train/test splits with real Monte Carlo data.
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

# Import real components (no mock dependencies)
from ..training.real_vae_training_pipeline import (
    RealVAETrainingPipeline, RealVAETrainingConfig, create_real_vae_training_pipeline,
    load_physics_data_from_file
)
from ..analysis.blind_critical_exponent_extractor import (
    BlindCriticalExponentExtractor, create_blind_critical_exponent_extractor
)
from ..validation.blind_validation_framework import (
    BlindValidationFramework, create_blind_validation_framework
)
from ..data.enhanced_monte_carlo import EnhancedMonteCarloSimulator

# Simple logger function
def get_logger(name):
    return logging.getLogger(name)


@dataclass
class RealValidationConfig:
    """Configuration for real accuracy validation."""
    
    # Data parameters
    train_split: float = 0.7
    validation_split: float = 0.15
    test_split: float = 0.15
    
    # VAE training parameters
    vae_epochs: int = 100
    vae_batch_size: int = 64
    vae_learning_rate: float = 1e-3
    vae_beta: float = 1.0
    use_physics_informed_loss: bool = True
    
    # Extraction parameters
    bootstrap_samples: int = 1000
    random_seed: Optional[int] = 42
    
    # Validation parameters
    confidence_level: float = 0.95
    target_accuracy_threshold: float = 70.0  # Realistic target (not 90%)
    
    # Output parameters
    save_results: bool = True
    results_dir: str = 'results/real_validation'
    create_visualizations: bool = True


@dataclass
class RealValidationResults:
    """Results from real accuracy validation pipeline."""
    
    # Training results
    vae_training_results: Any  # RealVAETrainingResults
    training_success: bool
    
    # Extraction results
    blind_extraction_results: Any  # BlindCriticalExponentResults
    extraction_success: bool
    
    # Validation results
    blind_validation_results: Any  # BlindValidationResults
    validation_passed: bool
    
    # Theoretical comparison (separate step)
    theoretical_comparison: Optional[Any] = None  # ComparisonResults
    
    # Overall assessment
    overall_accuracy: float
    meets_target_accuracy: bool
    reliability_grade: str
    
    # Quality metrics
    latent_magnetization_correlation: float
    reconstruction_quality: float
    extraction_quality_score: float
    
    # Performance metrics
    total_runtime: float
    training_time: float
    extraction_time: float
    validation_time: float
    
    # Configuration used
    config: RealValidationConfig


class RealAccuracyValidationPipeline:
    """Real accuracy validation pipeline using actual VAE training and blind extraction."""
    
    def __init__(self, config: RealValidationConfig):
        """Initialize real accuracy validation pipeline."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Set random seed for reproducibility
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
            torch.manual_seed(config.random_seed)
        
        # Initialize components
        self.vae_trainer = None
        self.blind_extractor = None
        self.validation_framework = None
        
        # Results storage
        self.results = None
        
        self.logger.info("Real accuracy validation pipeline initialized")
    
    def validate_system_accuracy(self,
                                data_file_path: str,
                                system_type: str = 'ising_3d') -> RealValidationResults:
        """
        Validate system accuracy using real VAE training and blind extraction.
        
        Args:
            data_file_path: Path to HDF5 file with physics data
            system_type: Type of physical system (for theoretical comparison only)
            
        Returns:
            RealValidationResults with complete validation assessment
        """
        self.logger.info(f"Starting real accuracy validation for {system_type}")
        start_time = time.time()
        
        # Create results directory
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Load and split data
            self.logger.info("Loading and splitting physics data")
            train_data, val_data, test_data = self._load_and_split_data(data_file_path)
            
            # Step 2: Train real VAE on physics data
            self.logger.info("Training real VAE on physics data")
            training_start = time.time()
            vae_results = self._train_real_vae(train_data, val_data)
            training_time = time.time() - training_start
            
            # Step 3: Extract latent representations from test data
            self.logger.info("Extracting latent representations from test data")
            latent_representations = self._extract_latent_representations(vae_results, test_data)
            
            # Step 4: Perform blind critical exponent extraction
            self.logger.info("Performing blind critical exponent extraction")
            extraction_start = time.time()
            extraction_results = self._perform_blind_extraction(latent_representations, test_data)
            extraction_time = time.time() - extraction_start
            
            # Step 5: Perform blind validation (no theoretical knowledge)
            self.logger.info("Performing blind validation")
            validation_start = time.time()
            validation_results = self._perform_blind_validation(
                test_data, extraction_results
            )
            validation_time = time.time() - validation_start
            
            # Step 6: Separate theoretical comparison (if requested)
            theoretical_comparison = None
            if system_type != 'unknown':
                self.logger.info("Performing separate theoretical comparison")
                theoretical_comparison = self._perform_theoretical_comparison(
                    extraction_results, system_type
                )
            
            # Step 7: Compute overall assessment
            overall_assessment = self._compute_overall_assessment(
                vae_results, extraction_results, validation_results, theoretical_comparison
            )
            
            total_time = time.time() - start_time
            
            # Create results object
            results = RealValidationResults(
                vae_training_results=vae_results,
                training_success=vae_results is not None,
                blind_extraction_results=extraction_results,
                extraction_success=extraction_results is not None,
                blind_validation_results=validation_results,
                validation_passed=validation_results.validation_passed if validation_results else False,
                theoretical_comparison=theoretical_comparison,
                overall_accuracy=overall_assessment['overall_accuracy'],
                meets_target_accuracy=overall_assessment['meets_target_accuracy'],
                reliability_grade=overall_assessment['reliability_grade'],
                latent_magnetization_correlation=overall_assessment['latent_magnetization_correlation'],
                reconstruction_quality=overall_assessment['reconstruction_quality'],
                extraction_quality_score=overall_assessment['extraction_quality_score'],
                total_runtime=total_time,
                training_time=training_time,
                extraction_time=extraction_time,
                validation_time=validation_time,
                config=self.config
            )
            
            # Save results
            if self.config.save_results:
                self._save_results(results, results_dir)
            
            # Create visualizations
            if self.config.create_visualizations:
                self._create_visualizations(results, results_dir)
            
            self.results = results
            
            self.logger.info(f"Real accuracy validation completed in {total_time:.2f} seconds")
            self.logger.info(f"Overall accuracy: {results.overall_accuracy:.1f}%")
            self.logger.info(f"Reliability grade: {results.reliability_grade}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Real accuracy validation failed: {e}")
            raise
    
    def _load_and_split_data(self, data_file_path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Load physics data and split into train/validation/test sets."""
        
        # Load data from HDF5 file
        configurations, temperatures, magnetizations, energies = load_physics_data_from_file(data_file_path)
        
        # Create proper train/test splits (no data leakage)
        n_total = len(configurations)
        
        # Shuffle indices
        indices = np.random.permutation(n_total)
        
        # Split indices
        n_train = int(self.config.train_split * n_total)
        n_val = int(self.config.validation_split * n_total)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # Create data splits
        train_data = {
            'configurations': configurations[train_indices],
            'temperatures': temperatures[train_indices],
            'magnetizations': magnetizations[train_indices],
            'energies': energies[train_indices] if energies is not None else None
        }
        
        val_data = {
            'configurations': configurations[val_indices],
            'temperatures': temperatures[val_indices],
            'magnetizations': magnetizations[val_indices],
            'energies': energies[val_indices] if energies is not None else None
        }
        
        test_data = {
            'configurations': configurations[test_indices],
            'temperatures': temperatures[test_indices],
            'magnetizations': magnetizations[test_indices],
            'energies': energies[test_indices] if energies is not None else None
        }
        
        self.logger.info(f"Data split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")
        
        return train_data, val_data, test_data
    
    def _train_real_vae(self, train_data: Dict[str, np.ndarray], val_data: Dict[str, np.ndarray]):
        """Train real VAE on physics data."""
        
        # Create VAE training configuration
        vae_config = RealVAETrainingConfig(
            batch_size=self.config.vae_batch_size,
            learning_rate=self.config.vae_learning_rate,
            num_epochs=self.config.vae_epochs,
            beta=self.config.vae_beta,
            use_physics_informed_loss=self.config.use_physics_informed_loss,
            validation_split=0.0,  # We already have validation data
            test_split=0.0,  # We already have test data
            random_seed=self.config.random_seed
        )
        
        # Create VAE training pipeline
        self.vae_trainer = create_real_vae_training_pipeline(vae_config)
        
        # Combine train and validation data for training
        combined_configs = np.concatenate([train_data['configurations'], val_data['configurations']])
        combined_temps = np.concatenate([train_data['temperatures'], val_data['temperatures']])
        combined_mags = np.concatenate([train_data['magnetizations'], val_data['magnetizations']])
        
        combined_energies = None
        if train_data['energies'] is not None and val_data['energies'] is not None:
            combined_energies = np.concatenate([train_data['energies'], val_data['energies']])
        
        # Train VAE
        try:
            vae_results = self.vae_trainer.train(
                combined_configs, combined_temps, combined_mags, combined_energies
            )
            
            self.logger.info(f"VAE training completed successfully")
            self.logger.info(f"Final validation loss: {vae_results.final_val_loss:.4f}")
            self.logger.info(f"Latent-magnetization correlation: {vae_results.latent_magnetization_correlation:.4f}")
            
            return vae_results
            
        except Exception as e:
            self.logger.error(f"VAE training failed: {e}")
            return None
    
    def _extract_latent_representations(self, vae_results, test_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract latent representations from test data using trained VAE."""
        
        if vae_results is None:
            raise ValueError("VAE training failed - cannot extract latent representations")
        
        # Load trained model
        model = self.vae_trainer.model
        model.eval()
        
        # Prepare test data
        test_configs = torch.FloatTensor(test_data['configurations'])
        
        # Add channel dimension if needed
        if len(test_configs.shape) == 3:  # 2D: (N, H, W)
            test_configs = test_configs.unsqueeze(1)  # (N, 1, H, W)
        elif len(test_configs.shape) == 4:  # 3D: (N, D, H, W)
            test_configs = test_configs.unsqueeze(1)  # (N, 1, D, H, W)
        
        test_configs = test_configs.to(self.vae_trainer.device)
        
        # Extract latent representations
        latent_representations = []
        
        with torch.no_grad():
            batch_size = 64
            for i in range(0, len(test_configs), batch_size):
                batch = test_configs[i:i + batch_size]
                z, mu, logvar = model.encode(batch)
                latent_representations.append(z.cpu().numpy())
        
        latent_representations = np.concatenate(latent_representations, axis=0)
        
        self.logger.info(f"Extracted latent representations: shape {latent_representations.shape}")
        
        return latent_representations
    
    def _perform_blind_extraction(self, latent_representations: np.ndarray, test_data: Dict[str, np.ndarray]):
        """Perform blind critical exponent extraction from latent representations."""
        
        # Create blind extractor
        self.blind_extractor = create_blind_critical_exponent_extractor(
            bootstrap_samples=self.config.bootstrap_samples,
            random_seed=self.config.random_seed
        )
        
        # Perform blind extraction (no theoretical knowledge used)
        try:
            extraction_results = self.blind_extractor.extract_critical_exponents_blind(
                latent_representations=latent_representations,
                temperatures=test_data['temperatures'],
                magnetizations=test_data['magnetizations'],  # For validation only
                system_identifier='unknown_system'  # No theoretical knowledge
            )
            
            self.logger.info("Blind extraction completed successfully")
            self.logger.info(f"Detected Tc: {extraction_results.tc_detection.critical_temperature:.4f}")
            self.logger.info(f"Order parameter dimension: {extraction_results.order_parameter_analysis.selected_dimension}")
            
            if extraction_results.beta_exponent:
                self.logger.info(f"β exponent: {extraction_results.beta_exponent.exponent:.4f} ± {extraction_results.beta_exponent.exponent_error:.4f}")
            
            return extraction_results
            
        except Exception as e:
            self.logger.error(f"Blind extraction failed: {e}")
            return None
    
    def _perform_blind_validation(self, test_data: Dict[str, np.ndarray], extraction_results):
        """Perform blind validation without theoretical knowledge."""
        
        if extraction_results is None:
            return None
        
        # Create validation framework
        self.validation_framework = create_blind_validation_framework(
            random_seed=self.config.random_seed
        )
        
        # Prepare parameters for validation
        fitted_parameters = {}
        fitting_errors = {}
        
        if extraction_results.beta_exponent:
            fitted_parameters['beta'] = extraction_results.beta_exponent.exponent
            fitted_parameters['amplitude'] = extraction_results.beta_exponent.amplitude
            fitting_errors['beta'] = extraction_results.beta_exponent.exponent_error
            fitting_errors['amplitude'] = extraction_results.beta_exponent.amplitude_error
        
        # Perform blind validation
        try:
            validation_results = self.validation_framework.validate_extraction_blind(
                temperatures=test_data['temperatures'],
                order_parameter=extraction_results.order_parameter_analysis.order_parameter_values,
                critical_temperature=extraction_results.tc_detection.critical_temperature,
                fitted_parameters=fitted_parameters,
                fitting_errors=fitting_errors
            )
            
            self.logger.info("Blind validation completed successfully")
            self.logger.info(f"Validation passed: {validation_results.validation_passed}")
            self.logger.info(f"Validation grade: {validation_results.validation_grade}")
            self.logger.info(f"Overall quality score: {validation_results.blind_metrics.overall_quality_score:.3f}")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Blind validation failed: {e}")
            return None
    
    def _perform_theoretical_comparison(self, extraction_results, system_type: str):
        """Separate step: Compare with theoretical predictions."""
        
        if extraction_results is None or not self.validation_framework:
            return None
        
        # Prepare measured parameters
        measured_parameters = {}
        parameter_errors = {}
        
        if extraction_results.beta_exponent:
            measured_parameters['beta'] = extraction_results.beta_exponent.exponent
            parameter_errors['beta'] = extraction_results.beta_exponent.exponent_error
        
        if extraction_results.nu_exponent:
            measured_parameters['nu'] = extraction_results.nu_exponent.exponent
            parameter_errors['nu'] = extraction_results.nu_exponent.exponent_error
        
        # Add critical temperature
        measured_parameters['tc'] = extraction_results.tc_detection.critical_temperature
        parameter_errors['tc'] = 0.1  # Estimate uncertainty
        
        # Perform theoretical comparison
        try:
            comparison_results = self.validation_framework.compare_with_theory_separate(
                measured_parameters=measured_parameters,
                parameter_errors=parameter_errors,
                system_type=system_type
            )
            
            self.logger.info("Theoretical comparison completed")
            self.logger.info(f"Overall agreement: {comparison_results.overall_agreement}")
            self.logger.info(f"Accuracy score: {comparison_results.accuracy_score:.1f}%")
            
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"Theoretical comparison failed: {e}")
            return None
    
    def _compute_overall_assessment(self, vae_results, extraction_results, validation_results, theoretical_comparison) -> Dict[str, Any]:
        """Compute overall assessment of the validation."""
        
        assessment = {
            'overall_accuracy': 0.0,
            'meets_target_accuracy': False,
            'reliability_grade': 'F',
            'latent_magnetization_correlation': 0.0,
            'reconstruction_quality': 0.0,
            'extraction_quality_score': 0.0
        }
        
        # VAE quality metrics
        if vae_results:
            assessment['latent_magnetization_correlation'] = vae_results.latent_magnetization_correlation
            assessment['reconstruction_quality'] = vae_results.reconstruction_quality
        
        # Extraction quality
        if extraction_results:
            assessment['extraction_quality_score'] = extraction_results.extraction_quality_score
        
        # Overall accuracy computation
        accuracy_components = []
        
        # 1. VAE training quality (30%)
        if vae_results:
            vae_quality = (
                0.4 * assessment['latent_magnetization_correlation'] +
                0.3 * assessment['reconstruction_quality'] +
                0.3 * (1.0 - min(1.0, vae_results.final_val_loss))
            )
            accuracy_components.append(('vae_quality', vae_quality, 0.3))
        
        # 2. Extraction quality (40%)
        if extraction_results:
            extraction_quality = assessment['extraction_quality_score']
            accuracy_components.append(('extraction_quality', extraction_quality, 0.4))
        
        # 3. Validation quality (30%)
        if validation_results:
            validation_quality = validation_results.blind_metrics.overall_quality_score
            accuracy_components.append(('validation_quality', validation_quality, 0.3))
        
        # Compute weighted average
        if accuracy_components:
            total_weight = sum(weight for _, _, weight in accuracy_components)
            weighted_sum = sum(score * weight for _, score, weight in accuracy_components)
            overall_accuracy = (weighted_sum / total_weight) * 100
        else:
            overall_accuracy = 0.0
        
        assessment['overall_accuracy'] = overall_accuracy
        assessment['meets_target_accuracy'] = overall_accuracy >= self.config.target_accuracy_threshold
        
        # Reliability grade
        if overall_accuracy >= 90:
            assessment['reliability_grade'] = 'A'
        elif overall_accuracy >= 80:
            assessment['reliability_grade'] = 'B'
        elif overall_accuracy >= 70:
            assessment['reliability_grade'] = 'C'
        elif overall_accuracy >= 60:
            assessment['reliability_grade'] = 'D'
        else:
            assessment['reliability_grade'] = 'F'
        
        return assessment
    
    def _save_results(self, results: RealValidationResults, results_dir: Path):
        """Save validation results to files."""
        
        # Save main results as JSON
        results_dict = asdict(results)
        
        # Convert non-serializable objects to summaries
        if results.vae_training_results:
            results_dict['vae_training_results'] = {
                'final_train_loss': results.vae_training_results.final_train_loss,
                'final_val_loss': results.vae_training_results.final_val_loss,
                'latent_magnetization_correlation': results.vae_training_results.latent_magnetization_correlation,
                'reconstruction_quality': results.vae_training_results.reconstruction_quality,
                'training_time': results.vae_training_results.training_time
            }
        
        if results.blind_extraction_results:
            results_dict['blind_extraction_results'] = {
                'critical_temperature': results.blind_extraction_results.tc_detection.critical_temperature,
                'tc_confidence': results.blind_extraction_results.tc_detection.detection_confidence,
                'order_parameter_dimension': results.blind_extraction_results.order_parameter_analysis.selected_dimension,
                'extraction_quality_score': results.blind_extraction_results.extraction_quality_score,
                'beta_exponent': results.blind_extraction_results.beta_exponent.exponent if results.blind_extraction_results.beta_exponent else None,
                'beta_error': results.blind_extraction_results.beta_exponent.exponent_error if results.blind_extraction_results.beta_exponent else None
            }
        
        # Save to JSON
        results_file = results_dir / 'validation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {results_file}")
    
    def _create_visualizations(self, results: RealValidationResults, results_dir: Path):
        """Create visualization plots."""
        
        try:
            # Create summary figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plot 1: Training curves
            if results.vae_training_results:
                ax = axes[0, 0]
                epochs = range(len(results.vae_training_results.training_losses))
                ax.plot(epochs, results.vae_training_results.training_losses, label='Training Loss')
                ax.plot(epochs, results.vae_training_results.validation_losses, label='Validation Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('VAE Training Curves')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Plot 2: Accuracy metrics
            ax = axes[0, 1]
            metrics = ['Overall', 'VAE Quality', 'Extraction', 'Validation']
            values = [
                results.overall_accuracy,
                results.latent_magnetization_correlation * 100,
                results.extraction_quality_score * 100,
                results.blind_validation_results.blind_metrics.overall_quality_score * 100 if results.blind_validation_results else 0
            ]
            
            bars = ax.bar(metrics, values)
            ax.set_ylabel('Score (%)')
            ax.set_title('Quality Metrics')
            ax.set_ylim(0, 100)
            
            # Color bars based on performance
            for bar, val in zip(bars, values):
                if val >= 80:
                    bar.set_color('green')
                elif val >= 70:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            # Plot 3: Performance summary
            ax = axes[1, 0]
            ax.axis('off')
            
            summary_text = f"""Real Accuracy Validation Summary
            
Overall Accuracy: {results.overall_accuracy:.1f}%
Reliability Grade: {results.reliability_grade}
Target Met: {'Yes' if results.meets_target_accuracy else 'No'}

VAE Training:
  Success: {'Yes' if results.training_success else 'No'}
  Latent-Mag Correlation: {results.latent_magnetization_correlation:.3f}
  Reconstruction Quality: {results.reconstruction_quality:.3f}

Extraction:
  Success: {'Yes' if results.extraction_success else 'No'}
  Quality Score: {results.extraction_quality_score:.3f}

Validation:
  Passed: {'Yes' if results.validation_passed else 'No'}
  
Runtime: {results.total_runtime:.1f}s
"""
            
            ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace')
            
            # Plot 4: Time breakdown
            ax = axes[1, 1]
            times = [results.training_time, results.extraction_time, results.validation_time]
            labels = ['Training', 'Extraction', 'Validation']
            
            ax.pie(times, labels=labels, autopct='%1.1f%%')
            ax.set_title('Time Breakdown')
            
            plt.tight_layout()
            
            # Save figure
            fig_path = results_dir / 'validation_summary.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Visualization saved to {fig_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create visualizations: {e}")


def create_real_accuracy_validation_pipeline(config: Optional[RealValidationConfig] = None) -> RealAccuracyValidationPipeline:
    """
    Factory function to create real accuracy validation pipeline.
    
    Args:
        config: Validation configuration (uses default if None)
        
    Returns:
        Configured RealAccuracyValidationPipeline instance
    """
    if config is None:
        config = RealValidationConfig()
    
    return RealAccuracyValidationPipeline(config)


def run_real_validation_example(data_file_path: str, system_type: str = 'ising_3d'):
    """
    Example function to run real accuracy validation.
    
    Args:
        data_file_path: Path to physics data file
        system_type: Type of physical system
    """
    # Create configuration
    config = RealValidationConfig(
        vae_epochs=50,  # Reduced for example
        target_accuracy_threshold=70.0,  # Realistic target
        save_results=True,
        create_visualizations=True
    )
    
    # Create pipeline
    pipeline = create_real_accuracy_validation_pipeline(config)
    
    # Run validation
    results = pipeline.validate_system_accuracy(data_file_path, system_type)
    
    print(f"Real Accuracy Validation Results:")
    print(f"Overall Accuracy: {results.overall_accuracy:.1f}%")
    print(f"Reliability Grade: {results.reliability_grade}")
    print(f"Target Met: {'Yes' if results.meets_target_accuracy else 'No'}")
    print(f"Training Success: {'Yes' if results.training_success else 'No'}")
    print(f"Extraction Success: {'Yes' if results.extraction_success else 'No'}")
    print(f"Validation Passed: {'Yes' if results.validation_passed else 'No'}")
    
    return results