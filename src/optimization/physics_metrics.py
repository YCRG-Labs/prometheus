"""
Physics Consistency Evaluation Metrics

This module implements comprehensive evaluation of VAE models for physics
consistency, including order parameter correlation, critical temperature
accuracy, and overall physics score computation.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from scipy import stats
from scipy.optimize import minimize_scalar

from ..models.vae import ConvolutionalVAE
from ..analysis.latent_analysis import LatentAnalyzer
from ..analysis.phase_detection import PhaseDetector

logger = logging.getLogger(__name__)


class PhysicsConsistencyEvaluator:
    """
    Comprehensive evaluator for physics consistency of VAE models.
    
    Evaluates models on multiple physics-based metrics including order
    parameter discovery, critical temperature detection, and overall
    physics consistency scores.
    """
    
    def __init__(
        self,
        test_loader: DataLoader,
        critical_temperature: float = 2.269,
        tolerance: float = 0.05
    ):
        """
        Initialize the physics consistency evaluator.
        
        Args:
            test_loader: DataLoader for test dataset
            critical_temperature: Theoretical critical temperature
            tolerance: Acceptable tolerance for critical temperature detection
        """
        self.test_loader = test_loader
        self.theoretical_tc = critical_temperature
        self.tolerance = tolerance
        
        logger.info(f"Physics evaluator initialized")
        logger.info(f"Theoretical Tc: {critical_temperature}")
        logger.info(f"Tolerance: {tolerance * 100}%")
    
    def evaluate_model(
        self, 
        model: ConvolutionalVAE, 
        device: torch.device
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of model physics consistency.
        
        Args:
            model: Trained VAE model to evaluate
            device: Device to run evaluation on
            
        Returns:
            Dictionary of physics consistency metrics
        """
        logger.info("Starting comprehensive physics evaluation")
        
        try:
            # Extract latent representations and physical quantities
            latent_data = self._extract_latent_data(model, device)
            
            # Evaluate order parameter correlation
            order_param_metrics = self._evaluate_order_parameter_correlation(latent_data)
            
            # Evaluate critical temperature detection
            critical_temp_metrics = self._evaluate_critical_temperature_detection(latent_data)
            
            # Evaluate phase separation quality
            phase_separation_metrics = self._evaluate_phase_separation(latent_data)
            
            # Evaluate reconstruction quality across temperatures
            reconstruction_metrics = self._evaluate_reconstruction_quality(model, device)
            
            # Compute overall physics consistency score
            overall_score = self._compute_overall_physics_score(
                order_param_metrics,
                critical_temp_metrics,
                phase_separation_metrics,
                reconstruction_metrics
            )
            
            # Combine all metrics
            physics_metrics = {
                **order_param_metrics,
                **critical_temp_metrics,
                **phase_separation_metrics,
                **reconstruction_metrics,
                'overall_physics_score': overall_score
            }
            
            logger.info(f"Physics evaluation completed. Overall score: {overall_score:.4f}")
            
            return physics_metrics
            
        except Exception as e:
            logger.error(f"Physics evaluation failed: {str(e)}")
            return {
                'overall_physics_score': 0.0,
                'order_parameter_correlation': 0.0,
                'critical_temperature_error': 1.0,
                'error': str(e)
            }
    
    def _extract_latent_data(self, model: ConvolutionalVAE, device: torch.device) -> Dict[str, np.ndarray]:
        """
        Extract latent representations and associated physical quantities.
        
        Args:
            model: VAE model
            device: Device to run on
            
        Returns:
            Dictionary containing latent coordinates, temperatures, and physical quantities
        """
        model.eval()
        
        latent_coords = []
        temperatures = []
        magnetizations = []
        energies = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 2:
                        data, temp = batch
                    elif len(batch) == 4:
                        data, temp, mag, energy = batch
                        magnetizations.extend(mag.cpu().numpy())
                        energies.extend(energy.cpu().numpy())
                    else:
                        data = batch[0]
                        temp = batch[1] if len(batch) > 1 else torch.zeros(data.size(0))
                else:
                    data = batch
                    temp = torch.zeros(data.size(0))  # Dummy temperatures
                
                data = data.to(device)
                
                # Get latent representation (mean of distribution)
                mu, _ = model.encode(data)
                latent_coords.append(mu.cpu().numpy())
                temperatures.extend(temp.cpu().numpy())
                
                # Calculate magnetization if not provided
                if len(magnetizations) == 0:
                    batch_magnetizations = self._calculate_magnetization(data.cpu().numpy())
                    magnetizations.extend(batch_magnetizations)
        
        latent_coords = np.vstack(latent_coords)
        temperatures = np.array(temperatures)
        magnetizations = np.array(magnetizations)
        
        return {
            'latent_coords': latent_coords,
            'temperatures': temperatures,
            'magnetizations': magnetizations,
            'energies': np.array(energies) if energies else np.array([])
        }
    
    def _calculate_magnetization(self, spin_configs: np.ndarray) -> List[float]:
        """
        Calculate magnetization for spin configurations.
        
        Args:
            spin_configs: Array of spin configurations
            
        Returns:
            List of magnetization values
        """
        magnetizations = []
        for config in spin_configs:
            # Convert from [-1, 1] range if needed
            spins = np.sign(config.squeeze())
            mag = np.abs(np.mean(spins))
            magnetizations.append(mag)
        
        return magnetizations
    
    def _evaluate_order_parameter_correlation(self, latent_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate correlation between latent dimensions and order parameters.
        
        Args:
            latent_data: Dictionary containing latent and physical data
            
        Returns:
            Dictionary of order parameter correlation metrics
        """
        latent_coords = latent_data['latent_coords']
        magnetizations = latent_data['magnetizations']
        
        if len(magnetizations) == 0:
            return {
                'order_parameter_correlation': 0.0,
                'best_latent_dimension': 0,
                'correlation_p_value': 1.0
            }
        
        # Find best correlation with magnetization
        best_correlation = 0.0
        best_dimension = 0
        best_p_value = 1.0
        
        for dim in range(latent_coords.shape[1]):
            # Calculate correlation with magnetization
            correlation, p_value = stats.pearsonr(latent_coords[:, dim], magnetizations)
            abs_correlation = abs(correlation)
            
            if abs_correlation > abs(best_correlation):
                best_correlation = correlation
                best_dimension = dim
                best_p_value = p_value
        
        # Calculate R² score for best dimension
        r2 = r2_score(magnetizations, latent_coords[:, best_dimension])
        
        return {
            'order_parameter_correlation': abs(best_correlation),
            'order_parameter_r2': max(0, r2),  # Ensure non-negative
            'best_latent_dimension': best_dimension,
            'correlation_p_value': best_p_value,
            'correlation_significance': 1.0 if best_p_value < 0.05 else 0.0
        }
    
    def _evaluate_critical_temperature_detection(self, latent_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate critical temperature detection accuracy.
        
        Args:
            latent_data: Dictionary containing latent and physical data
            
        Returns:
            Dictionary of critical temperature detection metrics
        """
        latent_coords = latent_data['latent_coords']
        temperatures = latent_data['temperatures']
        
        if len(np.unique(temperatures)) < 5:
            return {
                'critical_temperature_error': 1.0,
                'detected_critical_temperature': 0.0,
                'temperature_detection_accuracy': 0.0
            }
        
        try:
            # Use phase detector to find critical temperature
            phase_detector = PhaseDetector()
            
            # Detect critical temperature using multiple methods
            clustering_tc = self._detect_tc_clustering(latent_coords, temperatures)
            gradient_tc = self._detect_tc_gradient(latent_coords, temperatures)
            
            # Use average of methods as detected temperature
            detected_tc = np.mean([tc for tc in [clustering_tc, gradient_tc] if tc > 0])
            
            if detected_tc == 0:
                detected_tc = self.theoretical_tc  # Fallback
            
            # Calculate error metrics
            absolute_error = abs(detected_tc - self.theoretical_tc)
            relative_error = absolute_error / self.theoretical_tc
            
            # Check if within tolerance
            within_tolerance = relative_error <= self.tolerance
            
            return {
                'detected_critical_temperature': detected_tc,
                'critical_temperature_error': relative_error,
                'critical_temperature_absolute_error': absolute_error,
                'temperature_detection_accuracy': 1.0 if within_tolerance else 0.0,
                'clustering_tc': clustering_tc,
                'gradient_tc': gradient_tc
            }
            
        except Exception as e:
            logger.warning(f"Critical temperature detection failed: {str(e)}")
            return {
                'critical_temperature_error': 1.0,
                'detected_critical_temperature': 0.0,
                'temperature_detection_accuracy': 0.0
            }
    
    def _detect_tc_clustering(self, latent_coords: np.ndarray, temperatures: np.ndarray) -> float:
        """
        Detect critical temperature using clustering analysis.
        
        Args:
            latent_coords: Latent space coordinates
            temperatures: Temperature values
            
        Returns:
            Detected critical temperature
        """
        try:
            from sklearn.cluster import KMeans
            
            # Perform K-means clustering (k=2 for ordered/disordered phases)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(latent_coords)
            
            # Find temperature where cluster assignment changes most
            temp_sorted_indices = np.argsort(temperatures)
            sorted_temps = temperatures[temp_sorted_indices]
            sorted_labels = cluster_labels[temp_sorted_indices]
            
            # Look for the temperature with maximum cluster transition
            transition_scores = []
            unique_temps = np.unique(sorted_temps)
            
            for i in range(1, len(unique_temps) - 1):
                temp = unique_temps[i]
                temp_mask = sorted_temps == temp
                
                # Calculate cluster purity at this temperature
                temp_labels = sorted_labels[temp_mask]
                if len(temp_labels) > 0:
                    cluster_0_frac = np.mean(temp_labels == 0)
                    cluster_1_frac = np.mean(temp_labels == 1)
                    # Transition score is highest when clusters are mixed
                    transition_score = 2 * cluster_0_frac * cluster_1_frac
                    transition_scores.append((temp, transition_score))
            
            if transition_scores:
                # Find temperature with highest transition score
                best_temp = max(transition_scores, key=lambda x: x[1])[0]
                return best_temp
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _detect_tc_gradient(self, latent_coords: np.ndarray, temperatures: np.ndarray) -> float:
        """
        Detect critical temperature using gradient analysis.
        
        Args:
            latent_coords: Latent space coordinates
            temperatures: Temperature values
            
        Returns:
            Detected critical temperature
        """
        try:
            # Use the dimension with highest magnetization correlation
            if latent_coords.shape[1] >= 2:
                # Calculate variance of each dimension vs temperature
                temp_bins = np.linspace(temperatures.min(), temperatures.max(), 20)
                max_gradient = 0.0
                best_tc = 0.0
                
                for dim in range(latent_coords.shape[1]):
                    # Bin data by temperature
                    binned_means = []
                    bin_centers = []
                    
                    for i in range(len(temp_bins) - 1):
                        temp_mask = (temperatures >= temp_bins[i]) & (temperatures < temp_bins[i + 1])
                        if np.sum(temp_mask) > 0:
                            binned_means.append(np.mean(latent_coords[temp_mask, dim]))
                            bin_centers.append((temp_bins[i] + temp_bins[i + 1]) / 2)
                    
                    if len(binned_means) > 3:
                        # Calculate gradient
                        gradients = np.gradient(binned_means, bin_centers)
                        max_grad_idx = np.argmax(np.abs(gradients))
                        
                        if abs(gradients[max_grad_idx]) > max_gradient:
                            max_gradient = abs(gradients[max_grad_idx])
                            best_tc = bin_centers[max_grad_idx]
                
                return best_tc if best_tc > 0 else 0.0
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _evaluate_phase_separation(self, latent_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate quality of phase separation in latent space.
        
        Args:
            latent_data: Dictionary containing latent and physical data
            
        Returns:
            Dictionary of phase separation quality metrics
        """
        latent_coords = latent_data['latent_coords']
        temperatures = latent_data['temperatures']
        
        try:
            # Separate high and low temperature phases
            temp_median = np.median(temperatures)
            low_temp_mask = temperatures < temp_median
            high_temp_mask = temperatures >= temp_median
            
            if np.sum(low_temp_mask) == 0 or np.sum(high_temp_mask) == 0:
                return {
                    'phase_separation_quality': 0.0,
                    'inter_phase_distance': 0.0,
                    'intra_phase_compactness': 0.0
                }
            
            low_temp_coords = latent_coords[low_temp_mask]
            high_temp_coords = latent_coords[high_temp_mask]
            
            # Calculate centroids
            low_temp_centroid = np.mean(low_temp_coords, axis=0)
            high_temp_centroid = np.mean(high_temp_coords, axis=0)
            
            # Inter-phase distance (higher is better)
            inter_phase_distance = np.linalg.norm(low_temp_centroid - high_temp_centroid)
            
            # Intra-phase compactness (lower is better, so we invert)
            low_temp_spread = np.mean(np.linalg.norm(low_temp_coords - low_temp_centroid, axis=1))
            high_temp_spread = np.mean(np.linalg.norm(high_temp_coords - high_temp_centroid, axis=1))
            avg_intra_phase_spread = (low_temp_spread + high_temp_spread) / 2
            
            # Phase separation quality (ratio of inter to intra distances)
            if avg_intra_phase_spread > 0:
                separation_quality = inter_phase_distance / avg_intra_phase_spread
            else:
                separation_quality = inter_phase_distance
            
            # Normalize to [0, 1] range
            separation_quality = min(1.0, separation_quality / 10.0)
            
            return {
                'phase_separation_quality': separation_quality,
                'inter_phase_distance': inter_phase_distance,
                'intra_phase_compactness': 1.0 / (1.0 + avg_intra_phase_spread),
                'low_temp_spread': low_temp_spread,
                'high_temp_spread': high_temp_spread
            }
            
        except Exception as e:
            logger.warning(f"Phase separation evaluation failed: {str(e)}")
            return {
                'phase_separation_quality': 0.0,
                'inter_phase_distance': 0.0,
                'intra_phase_compactness': 0.0
            }
    
    def _evaluate_reconstruction_quality(self, model: ConvolutionalVAE, device: torch.device) -> Dict[str, float]:
        """
        Evaluate reconstruction quality across different temperature regimes.
        
        Args:
            model: VAE model
            device: Device to run on
            
        Returns:
            Dictionary of reconstruction quality metrics
        """
        model.eval()
        
        reconstruction_errors = []
        temperature_bins = {'low': [], 'medium': [], 'high': []}
        
        with torch.no_grad():
            for batch in self.test_loader:
                # Handle different batch formats
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    data, temp = batch[0], batch[1]
                else:
                    data = batch if not isinstance(batch, (list, tuple)) else batch[0]
                    temp = torch.zeros(data.size(0))
                
                data = data.to(device)
                
                # Get reconstruction
                reconstruction, mu, logvar = model(data)
                
                # Calculate reconstruction error (MSE)
                mse = torch.mean((data - reconstruction) ** 2, dim=[1, 2, 3])
                reconstruction_errors.extend(mse.cpu().numpy())
                
                # Bin by temperature
                for i, t in enumerate(temp.cpu().numpy()):
                    error = mse[i].item()
                    if t < self.theoretical_tc - 0.2:
                        temperature_bins['low'].append(error)
                    elif t > self.theoretical_tc + 0.2:
                        temperature_bins['high'].append(error)
                    else:
                        temperature_bins['medium'].append(error)
        
        # Calculate metrics
        overall_recon_error = np.mean(reconstruction_errors)
        
        # Temperature-specific reconstruction quality
        low_temp_error = np.mean(temperature_bins['low']) if temperature_bins['low'] else overall_recon_error
        high_temp_error = np.mean(temperature_bins['high']) if temperature_bins['high'] else overall_recon_error
        medium_temp_error = np.mean(temperature_bins['medium']) if temperature_bins['medium'] else overall_recon_error
        
        # Reconstruction consistency (lower variance is better)
        recon_consistency = 1.0 / (1.0 + np.std(reconstruction_errors))
        
        return {
            'overall_reconstruction_error': overall_recon_error,
            'reconstruction_quality_score': 1.0 / (1.0 + overall_recon_error),
            'low_temp_reconstruction_error': low_temp_error,
            'high_temp_reconstruction_error': high_temp_error,
            'critical_temp_reconstruction_error': medium_temp_error,
            'reconstruction_consistency': recon_consistency
        }
    
    def _compute_overall_physics_score(
        self,
        order_param_metrics: Dict[str, float],
        critical_temp_metrics: Dict[str, float],
        phase_separation_metrics: Dict[str, float],
        reconstruction_metrics: Dict[str, float]
    ) -> float:
        """
        Compute overall physics consistency score.
        
        Args:
            order_param_metrics: Order parameter correlation metrics
            critical_temp_metrics: Critical temperature detection metrics
            phase_separation_metrics: Phase separation quality metrics
            reconstruction_metrics: Reconstruction quality metrics
            
        Returns:
            Overall physics consistency score (0 to 1)
        """
        # Extract key metrics with defaults
        order_correlation = order_param_metrics.get('order_parameter_correlation', 0.0)
        temp_accuracy = critical_temp_metrics.get('temperature_detection_accuracy', 0.0)
        temp_error = critical_temp_metrics.get('critical_temperature_error', 1.0)
        phase_separation = phase_separation_metrics.get('phase_separation_quality', 0.0)
        recon_quality = reconstruction_metrics.get('reconstruction_quality_score', 0.0)
        
        # Convert temperature error to score (lower error is better)
        temp_score = 1.0 / (1.0 + temp_error)
        
        # Weighted combination of physics metrics
        physics_score = (
            0.35 * order_correlation +      # Order parameter discovery (highest weight)
            0.25 * temp_score +             # Critical temperature accuracy
            0.15 * temp_accuracy +          # Binary temperature detection accuracy
            0.15 * phase_separation +       # Phase separation quality
            0.10 * recon_quality           # Reconstruction quality
        )
        
        # Ensure score is in [0, 1] range
        physics_score = max(0.0, min(1.0, physics_score))
        
        return physics_score