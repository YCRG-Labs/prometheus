"""
Phase Transition Detection System

This module implements various algorithms for detecting phase transitions in latent space
representations, including clustering-based methods, gradient-based detection, and
information-theoretic measures.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import NearestNeighbors
import warnings

from .latent_analysis import LatentRepresentation
from ..utils.logging_utils import get_logger, LoggingContext


@dataclass
class ClusteringResult:
    """
    Container for clustering analysis results.
    
    Attributes:
        n_clusters: Number of clusters used
        cluster_labels: Cluster assignment for each sample
        cluster_centers: Cluster center coordinates
        inertia: Within-cluster sum of squares
        silhouette_score: Silhouette coefficient
        calinski_harabasz_score: Calinski-Harabasz index
        davies_bouldin_score: Davies-Bouldin index
        temperature_separation: Temperature statistics by cluster
    """
    n_clusters: int
    cluster_labels: np.ndarray
    cluster_centers: np.ndarray
    inertia: float
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    temperature_separation: Dict[str, Any]


@dataclass
class PhaseDetectionResult:
    """
    Container for phase transition detection results.
    
    Attributes:
        critical_temperature: Estimated critical temperature
        confidence: Confidence measure for the estimate
        method: Detection method used
        transition_region: Temperature range of transition
        clustering_result: Associated clustering analysis
        gradient_analysis: Gradient-based analysis results
        ensemble_scores: Scores from multiple detection methods
    """
    critical_temperature: float
    confidence: float
    method: str
    transition_region: Tuple[float, float]
    clustering_result: Optional[ClusteringResult] = None
    gradient_analysis: Optional[Dict[str, Any]] = None
    ensemble_scores: Optional[Dict[str, float]] = None


class ClusteringPhaseDetector:
    """
    Clustering-based phase transition detection using K-means clustering
    in latent space to identify phase boundaries.
    """
    
    def __init__(self, 
                 max_clusters: int = 10,
                 random_state: int = 42):
        """
        Initialize clustering-based phase detector.
        
        Args:
            max_clusters: Maximum number of clusters to test
            random_state: Random seed for reproducibility
        """
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.logger = get_logger(__name__)
        
    def find_optimal_clusters(self, 
                            latent_coords: np.ndarray,
                            min_clusters: int = 2) -> Tuple[int, Dict[str, List[float]]]:
        """
        Find optimal number of clusters using multiple validation metrics.
        
        Args:
            latent_coords: Latent space coordinates (N, 2)
            min_clusters: Minimum number of clusters to test
            
        Returns:
            Tuple of (optimal_n_clusters, validation_scores)
        """
        self.logger.info(f"Finding optimal number of clusters (range: {min_clusters}-{self.max_clusters})")
        
        # Standardize coordinates for clustering
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(latent_coords)
        
        # Test different numbers of clusters
        cluster_range = range(min_clusters, min(self.max_clusters + 1, len(latent_coords) // 10))
        
        scores = {
            'inertia': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': []
        }
        
        for n_clusters in cluster_range:
            # Fit K-means
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cluster_labels = kmeans.fit_predict(coords_scaled)
            
            # Calculate validation metrics
            scores['inertia'].append(kmeans.inertia_)
            
            if len(np.unique(cluster_labels)) > 1:  # Need at least 2 clusters for these metrics
                scores['silhouette'].append(silhouette_score(coords_scaled, cluster_labels))
                scores['calinski_harabasz'].append(calinski_harabasz_score(coords_scaled, cluster_labels))
                scores['davies_bouldin'].append(davies_bouldin_score(coords_scaled, cluster_labels))
            else:
                scores['silhouette'].append(-1)
                scores['calinski_harabasz'].append(0)
                scores['davies_bouldin'].append(float('inf'))
        
        # Find optimal number using elbow method for inertia and best silhouette score
        optimal_n_clusters = self._find_elbow_point(list(cluster_range), scores['inertia'])
        
        # Validate with silhouette score
        if len(scores['silhouette']) > 0:
            silhouette_optimal = cluster_range[np.argmax(scores['silhouette'])]
            
            # Use silhouette score if it suggests a reasonable number
            if abs(silhouette_optimal - optimal_n_clusters) <= 2:
                optimal_n_clusters = silhouette_optimal
        
        self.logger.info(f"Optimal number of clusters: {optimal_n_clusters}")
        
        return optimal_n_clusters, scores
    
    def _find_elbow_point(self, x_values: List[int], y_values: List[float]) -> int:
        """
        Find elbow point in curve using the maximum curvature method.
        
        Args:
            x_values: X-axis values (number of clusters)
            y_values: Y-axis values (inertia scores)
            
        Returns:
            X-value at elbow point
        """
        if len(x_values) < 3:
            return x_values[0] if x_values else 2
        
        # Normalize values to [0, 1] range
        x_norm = np.array(x_values, dtype=float)
        y_norm = np.array(y_values, dtype=float)
        
        x_norm = (x_norm - x_norm.min()) / (x_norm.max() - x_norm.min())
        y_norm = (y_norm - y_norm.min()) / (y_norm.max() - y_norm.min())
        
        # Calculate distances from each point to line connecting first and last points
        line_vec = np.array([x_norm[-1] - x_norm[0], y_norm[-1] - y_norm[0]])
        line_vec_norm = line_vec / np.linalg.norm(line_vec)
        
        distances = []
        for i in range(len(x_norm)):
            point_vec = np.array([x_norm[i] - x_norm[0], y_norm[i] - y_norm[0]])
            # Distance from point to line
            cross_product = np.cross(point_vec, line_vec_norm)
            distances.append(abs(cross_product))
        
        # Return x-value with maximum distance (elbow point)
        elbow_idx = np.argmax(distances)
        return x_values[elbow_idx]
    
    def perform_clustering(self, 
                          latent_repr: LatentRepresentation,
                          n_clusters: Optional[int] = None) -> ClusteringResult:
        """
        Perform K-means clustering on latent space coordinates.
        
        Args:
            latent_repr: LatentRepresentation to cluster
            n_clusters: Number of clusters (auto-determined if None)
            
        Returns:
            ClusteringResult with clustering analysis
        """
        self.logger.info("Performing K-means clustering on latent space")
        
        latent_coords = latent_repr.latent_coords
        
        # Find optimal number of clusters if not specified
        if n_clusters is None:
            n_clusters, _ = self.find_optimal_clusters(latent_coords)
        
        # Standardize coordinates
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(latent_coords)
        
        # Perform clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        cluster_labels = kmeans.fit_predict(coords_scaled)
        
        # Transform cluster centers back to original scale
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        # Calculate validation metrics
        silhouette = silhouette_score(coords_scaled, cluster_labels) if n_clusters > 1 else -1
        calinski_harabasz = calinski_harabasz_score(coords_scaled, cluster_labels) if n_clusters > 1 else 0
        davies_bouldin = davies_bouldin_score(coords_scaled, cluster_labels) if n_clusters > 1 else float('inf')
        
        # Analyze temperature separation by cluster
        temperature_separation = self._analyze_temperature_separation(
            cluster_labels, latent_repr.temperatures
        )
        
        result = ClusteringResult(
            n_clusters=n_clusters,
            cluster_labels=cluster_labels,
            cluster_centers=cluster_centers,
            inertia=kmeans.inertia_,
            silhouette_score=silhouette,
            calinski_harabasz_score=calinski_harabasz,
            davies_bouldin_score=davies_bouldin,
            temperature_separation=temperature_separation
        )
        
        self.logger.info(f"Clustering completed: {n_clusters} clusters, silhouette={silhouette:.3f}")
        
        return result
    
    def _analyze_temperature_separation(self, 
                                      cluster_labels: np.ndarray,
                                      temperatures: np.ndarray) -> Dict[str, Any]:
        """
        Analyze how well clusters separate different temperature regimes.
        
        Args:
            cluster_labels: Cluster assignments for each sample
            temperatures: Temperature values for each sample
            
        Returns:
            Dictionary with temperature separation analysis
        """
        unique_clusters = np.unique(cluster_labels)
        cluster_temps = {}
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_temp_values = temperatures[cluster_mask]
            
            cluster_temps[f'cluster_{cluster_id}'] = {
                'mean_temperature': float(np.mean(cluster_temp_values)),
                'std_temperature': float(np.std(cluster_temp_values)),
                'min_temperature': float(np.min(cluster_temp_values)),
                'max_temperature': float(np.max(cluster_temp_values)),
                'n_samples': int(np.sum(cluster_mask)),
                'temperature_range': float(np.max(cluster_temp_values) - np.min(cluster_temp_values))
            }
        
        # Calculate overall separation quality
        cluster_means = [cluster_temps[f'cluster_{i}']['mean_temperature'] for i in unique_clusters]
        separation_quality = np.std(cluster_means) / np.mean([cluster_temps[f'cluster_{i}']['std_temperature'] 
                                                             for i in unique_clusters])
        
        return {
            'cluster_statistics': cluster_temps,
            'separation_quality': float(separation_quality),
            'temperature_range_total': float(np.max(temperatures) - np.min(temperatures)),
            'n_clusters': len(unique_clusters)
        }
    
    def detect_phase_boundary(self, 
                            clustering_result: ClusteringResult,
                            latent_repr: LatentRepresentation) -> PhaseDetectionResult:
        """
        Detect phase transition boundary from clustering results.
        
        Args:
            clustering_result: Results from clustering analysis
            latent_repr: Original latent representation
            
        Returns:
            PhaseDetectionResult with critical temperature estimate
        """
        self.logger.info("Detecting phase boundary from clustering results")
        
        # Identify clusters corresponding to different phases
        cluster_stats = clustering_result.temperature_separation['cluster_statistics']
        
        # Sort clusters by mean temperature
        cluster_ids = []
        mean_temps = []
        
        for cluster_key, stats in cluster_stats.items():
            cluster_id = int(cluster_key.split('_')[1])
            cluster_ids.append(cluster_id)
            mean_temps.append(stats['mean_temperature'])
        
        # Sort by temperature
        sorted_indices = np.argsort(mean_temps)
        sorted_cluster_ids = [cluster_ids[i] for i in sorted_indices]
        sorted_mean_temps = [mean_temps[i] for i in sorted_indices]
        
        # Find the largest temperature gap between adjacent clusters
        if len(sorted_mean_temps) >= 2:
            temp_gaps = []
            gap_positions = []
            
            for i in range(len(sorted_mean_temps) - 1):
                gap = sorted_mean_temps[i + 1] - sorted_mean_temps[i]
                temp_gaps.append(gap)
                gap_positions.append((sorted_mean_temps[i] + sorted_mean_temps[i + 1]) / 2)
            
            # Critical temperature is at the largest gap
            max_gap_idx = np.argmax(temp_gaps)
            critical_temp = gap_positions[max_gap_idx]
            
            # Estimate confidence based on gap size relative to temperature range
            total_temp_range = sorted_mean_temps[-1] - sorted_mean_temps[0]
            confidence = temp_gaps[max_gap_idx] / total_temp_range if total_temp_range > 0 else 0
            
            # Define transition region
            transition_width = temp_gaps[max_gap_idx]
            transition_region = (
                critical_temp - transition_width / 2,
                critical_temp + transition_width / 2
            )
            
        else:
            # Fallback for single cluster or no clear separation
            critical_temp = np.mean(latent_repr.temperatures)
            confidence = 0.0
            temp_std = np.std(latent_repr.temperatures)
            transition_region = (critical_temp - temp_std, critical_temp + temp_std)
        
        result = PhaseDetectionResult(
            critical_temperature=critical_temp,
            confidence=confidence,
            method='clustering',
            transition_region=transition_region,
            clustering_result=clustering_result
        )
        
        self.logger.info(f"Phase boundary detected: T_c = {critical_temp:.3f} ± {transition_width/2:.3f}")
        
        return result
    
    def visualize_clustering(self,
                           latent_repr: LatentRepresentation,
                           clustering_result: ClusteringResult,
                           figsize: Tuple[int, int] = (15, 5)) -> Figure:
        """
        Create visualization of clustering results and phase boundaries.
        
        Args:
            latent_repr: Original latent representation
            clustering_result: Clustering analysis results
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure with clustering visualizations
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: Clusters in latent space
        unique_clusters = np.unique(clustering_result.cluster_labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = clustering_result.cluster_labels == cluster_id
            axes[0].scatter(
                latent_repr.z1[cluster_mask],
                latent_repr.z2[cluster_mask],
                c=[colors[i]],
                alpha=0.6,
                s=20,
                label=f'Cluster {cluster_id}'
            )
        
        # Plot cluster centers
        axes[0].scatter(
            clustering_result.cluster_centers[:, 0],
            clustering_result.cluster_centers[:, 1],
            c='black',
            marker='x',
            s=100,
            linewidths=3,
            label='Centers'
        )
        
        axes[0].set_xlabel('Latent Dimension 1 (z₁)')
        axes[0].set_ylabel('Latent Dimension 2 (z₂)')
        axes[0].set_title(f'K-means Clustering (k={clustering_result.n_clusters})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect('equal', adjustable='box')
        
        # Plot 2: Temperature distribution by cluster
        cluster_stats = clustering_result.temperature_separation['cluster_statistics']
        
        cluster_ids = []
        mean_temps = []
        std_temps = []
        
        for cluster_key, stats in cluster_stats.items():
            cluster_id = int(cluster_key.split('_')[1])
            cluster_ids.append(cluster_id)
            mean_temps.append(stats['mean_temperature'])
            std_temps.append(stats['std_temperature'])
        
        # Sort by mean temperature
        sorted_indices = np.argsort(mean_temps)
        sorted_cluster_ids = [cluster_ids[i] for i in sorted_indices]
        sorted_mean_temps = [mean_temps[i] for i in sorted_indices]
        sorted_std_temps = [std_temps[i] for i in sorted_indices]
        
        x_pos = np.arange(len(sorted_cluster_ids))
        axes[1].bar(x_pos, sorted_mean_temps, yerr=sorted_std_temps, 
                   capsize=5, alpha=0.7, color=colors[:len(sorted_cluster_ids)])
        
        axes[1].set_xlabel('Cluster (sorted by temperature)')
        axes[1].set_ylabel('Mean Temperature')
        axes[1].set_title('Temperature Distribution by Cluster')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels([f'C{cid}' for cid in sorted_cluster_ids])
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Cluster assignments vs temperature
        scatter = axes[2].scatter(
            latent_repr.temperatures,
            clustering_result.cluster_labels,
            c=clustering_result.cluster_labels,
            cmap='Set1',
            alpha=0.6,
            s=20
        )
        
        axes[2].set_xlabel('Temperature')
        axes[2].set_ylabel('Cluster Assignment')
        axes[2].set_title('Cluster Assignment vs Temperature')
        axes[2].grid(True, alpha=0.3)
        
        # Add cluster boundaries as vertical lines
        for i in range(len(sorted_mean_temps) - 1):
            boundary_temp = (sorted_mean_temps[i] + sorted_mean_temps[i + 1]) / 2
            axes[2].axvline(boundary_temp, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        return fig


class GradientPhaseDetector:
    """
    Gradient-based phase transition detection using derivatives of latent variables
    with respect to temperature to identify critical points.
    """
    
    def __init__(self, 
                 smoothing_sigma: float = 0.5,
                 min_temp_points: int = 10):
        """
        Initialize gradient-based phase detector.
        
        Args:
            smoothing_sigma: Gaussian smoothing parameter for gradient calculation
            min_temp_points: Minimum number of temperature points required
        """
        self.smoothing_sigma = smoothing_sigma
        self.min_temp_points = min_temp_points
        self.logger = get_logger(__name__)
    
    def calculate_temperature_gradients(self, 
                                      latent_repr: LatentRepresentation,
                                      n_temp_bins: int = 50) -> Dict[str, np.ndarray]:
        """
        Calculate gradients of latent variables with respect to temperature.
        
        Args:
            latent_repr: LatentRepresentation to analyze
            n_temp_bins: Number of temperature bins for gradient calculation
            
        Returns:
            Dictionary with temperature points and gradients for each latent dimension
        """
        self.logger.info("Calculating temperature gradients of latent variables")
        
        # Create temperature bins
        temp_min, temp_max = np.min(latent_repr.temperatures), np.max(latent_repr.temperatures)
        temp_bins = np.linspace(temp_min, temp_max, n_temp_bins)
        temp_centers = (temp_bins[:-1] + temp_bins[1:]) / 2
        
        # Calculate mean latent values in each temperature bin
        z1_means = []
        z2_means = []
        z1_stds = []
        z2_stds = []
        valid_temps = []
        
        for i in range(len(temp_bins) - 1):
            temp_mask = (latent_repr.temperatures >= temp_bins[i]) & \
                       (latent_repr.temperatures < temp_bins[i + 1])
            
            if np.sum(temp_mask) >= self.min_temp_points:
                z1_bin = latent_repr.z1[temp_mask]
                z2_bin = latent_repr.z2[temp_mask]
                
                z1_means.append(np.mean(z1_bin))
                z2_means.append(np.mean(z2_bin))
                z1_stds.append(np.std(z1_bin))
                z2_stds.append(np.std(z2_bin))
                valid_temps.append(temp_centers[i])
        
        if len(valid_temps) < 5:
            raise ValueError(f"Insufficient temperature points for gradient calculation: {len(valid_temps)}")
        
        # Convert to arrays
        valid_temps = np.array(valid_temps)
        z1_means = np.array(z1_means)
        z2_means = np.array(z2_means)
        z1_stds = np.array(z1_stds)
        z2_stds = np.array(z2_stds)
        
        # Apply smoothing
        if self.smoothing_sigma > 0:
            z1_means_smooth = gaussian_filter1d(z1_means, sigma=self.smoothing_sigma)
            z2_means_smooth = gaussian_filter1d(z2_means, sigma=self.smoothing_sigma)
            z1_stds_smooth = gaussian_filter1d(z1_stds, sigma=self.smoothing_sigma)
            z2_stds_smooth = gaussian_filter1d(z2_stds, sigma=self.smoothing_sigma)
        else:
            z1_means_smooth = z1_means
            z2_means_smooth = z2_means
            z1_stds_smooth = z1_stds
            z2_stds_smooth = z2_stds
        
        # Calculate gradients using finite differences
        dt = valid_temps[1] - valid_temps[0]  # Assume uniform spacing
        
        z1_gradient = np.gradient(z1_means_smooth, dt)
        z2_gradient = np.gradient(z2_means_smooth, dt)
        z1_std_gradient = np.gradient(z1_stds_smooth, dt)
        z2_std_gradient = np.gradient(z2_stds_smooth, dt)
        
        # Calculate second derivatives for curvature analysis
        z1_curvature = np.gradient(z1_gradient, dt)
        z2_curvature = np.gradient(z2_gradient, dt)
        
        return {
            'temperatures': valid_temps,
            'z1_means': z1_means_smooth,
            'z2_means': z2_means_smooth,
            'z1_stds': z1_stds_smooth,
            'z2_stds': z2_stds_smooth,
            'z1_gradient': z1_gradient,
            'z2_gradient': z2_gradient,
            'z1_std_gradient': z1_std_gradient,
            'z2_std_gradient': z2_std_gradient,
            'z1_curvature': z1_curvature,
            'z2_curvature': z2_curvature
        }
    
    def detect_critical_temperature(self, 
                                  gradient_data: Dict[str, np.ndarray]) -> PhaseDetectionResult:
        """
        Detect critical temperature from gradient analysis.
        
        Args:
            gradient_data: Results from calculate_temperature_gradients
            
        Returns:
            PhaseDetectionResult with critical temperature estimate
        """
        self.logger.info("Detecting critical temperature from gradients")
        
        temperatures = gradient_data['temperatures']
        
        # Method 1: Maximum absolute gradient
        z1_grad_abs = np.abs(gradient_data['z1_gradient'])
        z2_grad_abs = np.abs(gradient_data['z2_gradient'])
        
        # Combined gradient magnitude
        grad_magnitude = np.sqrt(z1_grad_abs**2 + z2_grad_abs**2)
        max_grad_idx = np.argmax(grad_magnitude)
        tc_gradient = temperatures[max_grad_idx]
        
        # Method 2: Maximum curvature (second derivative)
        z1_curv_abs = np.abs(gradient_data['z1_curvature'])
        z2_curv_abs = np.abs(gradient_data['z2_curvature'])
        
        curv_magnitude = np.sqrt(z1_curv_abs**2 + z2_curv_abs**2)
        max_curv_idx = np.argmax(curv_magnitude)
        tc_curvature = temperatures[max_curv_idx]
        
        # Method 3: Maximum standard deviation gradient (fluctuation peak)
        std_grad_magnitude = np.sqrt(gradient_data['z1_std_gradient']**2 + 
                                   gradient_data['z2_std_gradient']**2)
        max_std_grad_idx = np.argmax(std_grad_magnitude)
        tc_fluctuation = temperatures[max_std_grad_idx]
        
        # Ensemble estimate (weighted average)
        estimates = [tc_gradient, tc_curvature, tc_fluctuation]
        weights = [grad_magnitude[max_grad_idx], 
                  curv_magnitude[max_curv_idx],
                  std_grad_magnitude[max_std_grad_idx]]
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        tc_ensemble = np.average(estimates, weights=weights)
        
        # Estimate confidence based on agreement between methods
        estimate_std = np.std(estimates)
        temp_range = temperatures[-1] - temperatures[0]
        confidence = max(0, 1 - (estimate_std / temp_range) * 2)  # Scale to [0, 1]
        
        # Define transition region based on gradient width
        gradient_threshold = grad_magnitude[max_grad_idx] * 0.5
        above_threshold = grad_magnitude > gradient_threshold
        
        if np.any(above_threshold):
            transition_indices = np.where(above_threshold)[0]
            transition_start = temperatures[transition_indices[0]]
            transition_end = temperatures[transition_indices[-1]]
        else:
            # Fallback: use standard deviation around estimate
            temp_std = estimate_std if estimate_std > 0 else 0.1
            transition_start = tc_ensemble - temp_std
            transition_end = tc_ensemble + temp_std
        
        gradient_analysis = {
            'tc_gradient': float(tc_gradient),
            'tc_curvature': float(tc_curvature),
            'tc_fluctuation': float(tc_fluctuation),
            'tc_ensemble': float(tc_ensemble),
            'gradient_magnitude': grad_magnitude.tolist(),
            'curvature_magnitude': curv_magnitude.tolist(),
            'std_gradient_magnitude': std_grad_magnitude.tolist(),
            'method_weights': weights.tolist()
        }
        
        result = PhaseDetectionResult(
            critical_temperature=tc_ensemble,
            confidence=confidence,
            method='gradient',
            transition_region=(transition_start, transition_end),
            gradient_analysis=gradient_analysis
        )
        
        self.logger.info(f"Gradient-based T_c = {tc_ensemble:.3f} (confidence: {confidence:.3f})")
        
        return result
    
    def visualize_gradients(self,
                          gradient_data: Dict[str, np.ndarray],
                          detection_result: PhaseDetectionResult,
                          figsize: Tuple[int, int] = (15, 10)) -> Figure:
        """
        Create visualization of gradient analysis results.
        
        Args:
            gradient_data: Results from gradient calculation
            detection_result: Phase detection results
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure with gradient visualizations
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        temperatures = gradient_data['temperatures']
        tc = detection_result.critical_temperature
        
        # Plot 1: Latent variable means vs temperature
        axes[0, 0].plot(temperatures, gradient_data['z1_means'], 'b-', label='z₁ mean', linewidth=2)
        axes[0, 0].plot(temperatures, gradient_data['z2_means'], 'r-', label='z₂ mean', linewidth=2)
        axes[0, 0].axvline(tc, color='black', linestyle='--', alpha=0.7, label=f'T_c = {tc:.3f}')
        axes[0, 0].set_xlabel('Temperature')
        axes[0, 0].set_ylabel('Latent Variable Mean')
        axes[0, 0].set_title('Latent Variables vs Temperature')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Gradients vs temperature
        axes[0, 1].plot(temperatures, gradient_data['z1_gradient'], 'b-', label='dz₁/dT', linewidth=2)
        axes[0, 1].plot(temperatures, gradient_data['z2_gradient'], 'r-', label='dz₂/dT', linewidth=2)
        axes[0, 1].axvline(tc, color='black', linestyle='--', alpha=0.7, label=f'T_c = {tc:.3f}')
        axes[0, 1].set_xlabel('Temperature')
        axes[0, 1].set_ylabel('Gradient (dz/dT)')
        axes[0, 1].set_title('First Derivatives')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Curvature vs temperature
        axes[0, 2].plot(temperatures, gradient_data['z1_curvature'], 'b-', label='d²z₁/dT²', linewidth=2)
        axes[0, 2].plot(temperatures, gradient_data['z2_curvature'], 'r-', label='d²z₂/dT²', linewidth=2)
        axes[0, 2].axvline(tc, color='black', linestyle='--', alpha=0.7, label=f'T_c = {tc:.3f}')
        axes[0, 2].set_xlabel('Temperature')
        axes[0, 2].set_ylabel('Curvature (d²z/dT²)')
        axes[0, 2].set_title('Second Derivatives')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Standard deviations vs temperature
        axes[1, 0].plot(temperatures, gradient_data['z1_stds'], 'b-', label='z₁ std', linewidth=2)
        axes[1, 0].plot(temperatures, gradient_data['z2_stds'], 'r-', label='z₂ std', linewidth=2)
        axes[1, 0].axvline(tc, color='black', linestyle='--', alpha=0.7, label=f'T_c = {tc:.3f}')
        axes[1, 0].set_xlabel('Temperature')
        axes[1, 0].set_ylabel('Standard Deviation')
        axes[1, 0].set_title('Latent Variable Fluctuations')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Combined gradient magnitude
        grad_magnitude = np.sqrt(gradient_data['z1_gradient']**2 + gradient_data['z2_gradient']**2)
        axes[1, 1].plot(temperatures, grad_magnitude, 'g-', linewidth=2, label='|∇z|')
        axes[1, 1].axvline(tc, color='black', linestyle='--', alpha=0.7, label=f'T_c = {tc:.3f}')
        axes[1, 1].set_xlabel('Temperature')
        axes[1, 1].set_ylabel('Gradient Magnitude')
        axes[1, 1].set_title('Combined Gradient Magnitude')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Method comparison
        if detection_result.gradient_analysis:
            methods = ['Gradient', 'Curvature', 'Fluctuation']
            tc_values = [
                detection_result.gradient_analysis['tc_gradient'],
                detection_result.gradient_analysis['tc_curvature'],
                detection_result.gradient_analysis['tc_fluctuation']
            ]
            weights = detection_result.gradient_analysis['method_weights']
            
            bars = axes[1, 2].bar(methods, tc_values, alpha=0.7, 
                                color=['blue', 'red', 'green'])
            
            # Add weight annotations
            for bar, weight in zip(bars, weights):
                height = bar.get_height()
                axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'w={weight:.2f}', ha='center', va='bottom')
            
            axes[1, 2].axhline(tc, color='black', linestyle='--', alpha=0.7, 
                              label=f'Ensemble T_c = {tc:.3f}')
            axes[1, 2].set_ylabel('Critical Temperature')
            axes[1, 2].set_title('Method Comparison')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class InformationTheoreticDetector:
    """
    Information-theoretic phase transition detection using mutual information
    and entropy measures to identify critical points.
    """
    
    def __init__(self, 
                 n_temp_bins: int = 30,
                 n_neighbors: int = 5):
        """
        Initialize information-theoretic detector.
        
        Args:
            n_temp_bins: Number of temperature bins for analysis
            n_neighbors: Number of neighbors for mutual information estimation
        """
        self.n_temp_bins = n_temp_bins
        self.n_neighbors = n_neighbors
        self.logger = get_logger(__name__)
    
    def calculate_mutual_information(self, 
                                   latent_repr: LatentRepresentation) -> Dict[str, Any]:
        """
        Calculate mutual information between latent variables and temperature.
        
        Args:
            latent_repr: LatentRepresentation to analyze
            
        Returns:
            Dictionary with mutual information analysis results
        """
        self.logger.info("Calculating mutual information between latent variables and temperature")
        
        # Calculate mutual information for each latent dimension
        mi_z1_temp = mutual_info_regression(
            latent_repr.z1.reshape(-1, 1), 
            latent_repr.temperatures,
            n_neighbors=self.n_neighbors,
            random_state=42
        )[0]
        
        mi_z2_temp = mutual_info_regression(
            latent_repr.z2.reshape(-1, 1),
            latent_repr.temperatures,
            n_neighbors=self.n_neighbors,
            random_state=42
        )[0]
        
        # Calculate mutual information between latent dimensions
        mi_z1_z2 = mutual_info_regression(
            latent_repr.z1.reshape(-1, 1),
            latent_repr.z2,
            n_neighbors=self.n_neighbors,
            random_state=42
        )[0]
        
        # Temperature-binned analysis
        temp_min, temp_max = np.min(latent_repr.temperatures), np.max(latent_repr.temperatures)
        temp_bins = np.linspace(temp_min, temp_max, self.n_temp_bins + 1)
        temp_centers = (temp_bins[:-1] + temp_bins[1:]) / 2
        
        mi_z1_binned = []
        mi_z2_binned = []
        entropy_z1_binned = []
        entropy_z2_binned = []
        valid_temp_centers = []
        
        for i in range(len(temp_bins) - 1):
            temp_mask = (latent_repr.temperatures >= temp_bins[i]) & \
                       (latent_repr.temperatures < temp_bins[i + 1])
            
            if np.sum(temp_mask) >= 10:  # Minimum samples for reliable estimation
                z1_bin = latent_repr.z1[temp_mask]
                z2_bin = latent_repr.z2[temp_mask]
                
                # Calculate entropy (using histogram-based estimation)
                z1_hist, _ = np.histogram(z1_bin, bins=20, density=True)
                z2_hist, _ = np.histogram(z2_bin, bins=20, density=True)
                
                # Add small epsilon to avoid log(0)
                epsilon = 1e-10
                z1_hist = z1_hist + epsilon
                z2_hist = z2_hist + epsilon
                
                entropy_z1 = -np.sum(z1_hist * np.log(z1_hist))
                entropy_z2 = -np.sum(z2_hist * np.log(z2_hist))
                
                entropy_z1_binned.append(entropy_z1)
                entropy_z2_binned.append(entropy_z2)
                
                # Calculate local mutual information if enough samples
                if len(z1_bin) >= 20:
                    mi_z1_local = mutual_info_regression(
                        z1_bin.reshape(-1, 1),
                        z2_bin,
                        n_neighbors=min(self.n_neighbors, len(z1_bin) // 4),
                        random_state=42
                    )[0]
                    mi_z2_local = mi_z1_local  # Symmetric
                else:
                    mi_z1_local = 0
                    mi_z2_local = 0
                
                mi_z1_binned.append(mi_z1_local)
                mi_z2_binned.append(mi_z2_local)
                valid_temp_centers.append(temp_centers[i])
        
        return {
            'mi_z1_temperature': float(mi_z1_temp),
            'mi_z2_temperature': float(mi_z2_temp),
            'mi_z1_z2': float(mi_z1_z2),
            'temperatures': np.array(valid_temp_centers),
            'mi_z1_binned': np.array(mi_z1_binned),
            'mi_z2_binned': np.array(mi_z2_binned),
            'entropy_z1_binned': np.array(entropy_z1_binned),
            'entropy_z2_binned': np.array(entropy_z2_binned)
        }
    
    def detect_critical_temperature(self, 
                                  mi_data: Dict[str, Any]) -> PhaseDetectionResult:
        """
        Detect critical temperature from information-theoretic measures.
        
        Args:
            mi_data: Results from mutual information calculation
            
        Returns:
            PhaseDetectionResult with critical temperature estimate
        """
        self.logger.info("Detecting critical temperature from information-theoretic measures")
        
        temperatures = mi_data['temperatures']
        
        if len(temperatures) < 5:
            raise ValueError("Insufficient temperature points for information-theoretic analysis")
        
        # Method 1: Maximum entropy (maximum disorder/information)
        entropy_combined = mi_data['entropy_z1_binned'] + mi_data['entropy_z2_binned']
        max_entropy_idx = np.argmax(entropy_combined)
        tc_entropy = temperatures[max_entropy_idx]
        
        # Method 2: Maximum mutual information between latent dimensions
        max_mi_idx = np.argmax(mi_data['mi_z1_binned'])
        tc_mutual_info = temperatures[max_mi_idx]
        
        # Method 3: Maximum gradient of entropy
        entropy_gradient = np.gradient(entropy_combined)
        max_entropy_grad_idx = np.argmax(np.abs(entropy_gradient))
        tc_entropy_gradient = temperatures[max_entropy_grad_idx]
        
        # Ensemble estimate
        estimates = [tc_entropy, tc_mutual_info, tc_entropy_gradient]
        
        # Weight by the magnitude of the corresponding measures
        weights = [
            entropy_combined[max_entropy_idx],
            mi_data['mi_z1_binned'][max_mi_idx],
            np.abs(entropy_gradient[max_entropy_grad_idx])
        ]
        
        # Normalize weights
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        tc_ensemble = np.average(estimates, weights=weights)
        
        # Estimate confidence based on agreement between methods
        estimate_std = np.std(estimates)
        temp_range = temperatures[-1] - temperatures[0]
        confidence = max(0, 1 - (estimate_std / temp_range) * 2)
        
        # Define transition region
        transition_width = estimate_std if estimate_std > 0 else 0.1
        transition_region = (tc_ensemble - transition_width, tc_ensemble + transition_width)
        
        result = PhaseDetectionResult(
            critical_temperature=tc_ensemble,
            confidence=confidence,
            method='information_theoretic',
            transition_region=transition_region
        )
        
        self.logger.info(f"Information-theoretic T_c = {tc_ensemble:.3f} (confidence: {confidence:.3f})")
        
        return result


class PhaseTransitionDetector:
    """
    Main class for phase transition detection combining multiple methods.
    """
    
    def __init__(self, 
                 theoretical_tc: float = 2.269,
                 random_state: int = 42):
        """
        Initialize phase transition detector.
        
        Args:
            theoretical_tc: Theoretical critical temperature for validation
            random_state: Random seed for reproducibility
        """
        self.theoretical_tc = theoretical_tc
        self.random_state = random_state
        self.logger = get_logger(__name__)
        
        # Initialize detection methods
        self.clustering_detector = ClusteringPhaseDetector(random_state=random_state)
        self.gradient_detector = GradientPhaseDetector()
        self.info_detector = InformationTheoreticDetector()
    
    def detect_phase_transition(self, 
                              latent_repr: LatentRepresentation,
                              methods: List[str] = ['clustering', 'gradient', 'information_theoretic']) -> PhaseDetectionResult:
        """
        Detect phase transition using specified methods and ensemble combination.
        
        Args:
            latent_repr: LatentRepresentation to analyze
            methods: List of detection methods to use
            
        Returns:
            PhaseDetectionResult with ensemble analysis
        """
        self.logger.info(f"Detecting phase transition using methods: {methods}")
        
        results = {}
        
        # Clustering-based detection
        if 'clustering' in methods:
            with LoggingContext("Clustering-based detection"):
                clustering_result = self.clustering_detector.perform_clustering(latent_repr)
                clustering_detection = self.clustering_detector.detect_phase_boundary(
                    clustering_result, latent_repr
                )
                results['clustering'] = clustering_detection
        
        # Gradient-based detection
        if 'gradient' in methods:
            with LoggingContext("Gradient-based detection"):
                gradient_data = self.gradient_detector.calculate_temperature_gradients(latent_repr)
                gradient_detection = self.gradient_detector.detect_critical_temperature(gradient_data)
                results['gradient'] = gradient_detection
        
        # Information-theoretic detection
        if 'information_theoretic' in methods:
            with LoggingContext("Information-theoretic detection"):
                mi_data = self.info_detector.calculate_mutual_information(latent_repr)
                info_detection = self.info_detector.detect_critical_temperature(mi_data)
                results['information_theoretic'] = info_detection
        
        if not results:
            raise ValueError(f"No valid detection methods specified: {methods}")
        
        # Ensemble combination of multiple methods
        if len(results) > 1:
            ensemble_result = self._combine_detection_results(results, latent_repr)
        else:
            # Single method result
            ensemble_result = list(results.values())[0]
            ensemble_result.ensemble_scores = {
                method: result.critical_temperature for method, result in results.items()
            }
        
        # Add validation against theoretical value
        theoretical_error = abs(ensemble_result.critical_temperature - self.theoretical_tc)
        theoretical_accuracy = 1.0 - (theoretical_error / self.theoretical_tc)
        
        self.logger.info(f"Ensemble critical temperature: {ensemble_result.critical_temperature:.3f}")
        self.logger.info(f"Theoretical T_c: {self.theoretical_tc:.3f}")
        self.logger.info(f"Accuracy: {theoretical_accuracy:.1%}")
        
        return ensemble_result
    
    def _combine_detection_results(self, 
                                 results: Dict[str, PhaseDetectionResult],
                                 latent_repr: LatentRepresentation) -> PhaseDetectionResult:
        """
        Combine results from multiple detection methods using ensemble approach.
        
        Args:
            results: Dictionary of detection results by method
            latent_repr: Original latent representation for context
            
        Returns:
            Combined PhaseDetectionResult
        """
        self.logger.info("Combining detection results using ensemble method")
        
        # Extract critical temperatures and confidences
        tc_values = []
        confidences = []
        methods = []
        
        for method, result in results.items():
            tc_values.append(result.critical_temperature)
            confidences.append(result.confidence)
            methods.append(method)
        
        tc_values = np.array(tc_values)
        confidences = np.array(confidences)
        
        # Weight by confidence and inverse variance
        weights = confidences.copy()
        
        # Additional weighting based on agreement with theoretical value
        theoretical_errors = np.abs(tc_values - self.theoretical_tc)
        theoretical_weights = 1.0 / (1.0 + theoretical_errors)
        
        # Combine weights
        combined_weights = weights * theoretical_weights
        
        # Normalize weights
        if np.sum(combined_weights) > 0:
            combined_weights = combined_weights / np.sum(combined_weights)
        else:
            combined_weights = np.ones(len(combined_weights)) / len(combined_weights)
        
        # Calculate ensemble critical temperature
        tc_ensemble = np.average(tc_values, weights=combined_weights)
        
        # Calculate ensemble confidence based on agreement and individual confidences
        tc_std = np.std(tc_values)
        temp_range = np.max(latent_repr.temperatures) - np.min(latent_repr.temperatures)
        agreement_factor = max(0, 1 - (tc_std / temp_range) * 4)  # Penalize disagreement
        
        ensemble_confidence = np.average(confidences, weights=combined_weights) * agreement_factor
        
        # Define ensemble transition region
        transition_regions = [result.transition_region for result in results.values()]
        transition_starts = [region[0] for region in transition_regions]
        transition_ends = [region[1] for region in transition_regions]
        
        ensemble_transition_start = np.average(transition_starts, weights=combined_weights)
        ensemble_transition_end = np.average(transition_ends, weights=combined_weights)
        
        # Create ensemble scores dictionary
        ensemble_scores = {}
        for method, tc_value, confidence, weight in zip(methods, tc_values, confidences, combined_weights):
            ensemble_scores[method] = {
                'critical_temperature': float(tc_value),
                'confidence': float(confidence),
                'weight': float(weight)
            }
        
        # Create ensemble result
        ensemble_result = PhaseDetectionResult(
            critical_temperature=tc_ensemble,
            confidence=ensemble_confidence,
            method='ensemble',
            transition_region=(ensemble_transition_start, ensemble_transition_end),
            ensemble_scores=ensemble_scores
        )
        
        # Include individual method results for reference
        if 'clustering' in results:
            ensemble_result.clustering_result = results['clustering'].clustering_result
        if 'gradient' in results:
            ensemble_result.gradient_analysis = results['gradient'].gradient_analysis
        
        self.logger.info(f"Ensemble combination: T_c = {tc_ensemble:.3f}, confidence = {ensemble_confidence:.3f}")
        
        return ensemble_result
    
    def validate_detection(self, 
                          detection_result: PhaseDetectionResult,
                          tolerance: float = 0.05) -> Dict[str, Any]:
        """
        Validate phase detection results against theoretical expectations.
        
        Args:
            detection_result: Phase detection results to validate
            tolerance: Acceptable relative error tolerance
            
        Returns:
            Dictionary with validation metrics
        """
        self.logger.info("Validating phase detection results")
        
        detected_tc = detection_result.critical_temperature
        theoretical_error = abs(detected_tc - self.theoretical_tc)
        relative_error = theoretical_error / self.theoretical_tc
        
        validation = {
            'detected_tc': float(detected_tc),
            'theoretical_tc': float(self.theoretical_tc),
            'absolute_error': float(theoretical_error),
            'relative_error': float(relative_error),
            'within_tolerance': relative_error <= tolerance,
            'accuracy': float(1.0 - relative_error),
            'confidence': float(detection_result.confidence),
            'method': detection_result.method,
            'transition_region': detection_result.transition_region
        }
        
        if validation['within_tolerance']:
            self.logger.info(f"✓ Detection successful: {relative_error:.1%} error (< {tolerance:.1%} tolerance)")
        else:
            self.logger.warning(f"✗ Detection outside tolerance: {relative_error:.1%} error (> {tolerance:.1%} tolerance)")
        
        return validation
    
    def compare_with_onsager_solution(self, 
                                    detection_result: PhaseDetectionResult) -> Dict[str, Any]:
        """
        Compare detection results with theoretical Onsager solution for 2D Ising model.
        
        Args:
            detection_result: Phase detection results to compare
            
        Returns:
            Dictionary with detailed comparison metrics
        """
        self.logger.info("Comparing detection results with Onsager solution")
        
        # Onsager's exact solution: T_c = 2J / ln(1 + √2) ≈ 2.269 (for J=1)
        onsager_tc = 2.0 / np.log(1 + np.sqrt(2))
        
        detected_tc = detection_result.critical_temperature
        absolute_error = abs(detected_tc - onsager_tc)
        relative_error = absolute_error / onsager_tc
        
        # Classification of accuracy
        if relative_error <= 0.01:
            accuracy_class = "Excellent"
        elif relative_error <= 0.05:
            accuracy_class = "Good"
        elif relative_error <= 0.10:
            accuracy_class = "Acceptable"
        else:
            accuracy_class = "Poor"
        
        comparison = {
            'onsager_tc': float(onsager_tc),
            'detected_tc': float(detected_tc),
            'absolute_error': float(absolute_error),
            'relative_error': float(relative_error),
            'relative_error_percent': float(relative_error * 100),
            'accuracy_class': accuracy_class,
            'method': detection_result.method,
            'confidence': float(detection_result.confidence),
            'transition_region': detection_result.transition_region,
            'within_1_percent': relative_error <= 0.01,
            'within_5_percent': relative_error <= 0.05,
            'within_10_percent': relative_error <= 0.10
        }
        
        # Add ensemble method details if available
        if detection_result.ensemble_scores:
            comparison['ensemble_details'] = detection_result.ensemble_scores
        
        self.logger.info(f"Onsager comparison: {relative_error:.1%} error, {accuracy_class} accuracy")
        
        return comparison
    
    def create_comprehensive_report(self,
                                  latent_repr: LatentRepresentation,
                                  detection_result: PhaseDetectionResult) -> Dict[str, Any]:
        """
        Create comprehensive analysis report combining all detection methods and validation.
        
        Args:
            latent_repr: Original latent representation
            detection_result: Phase detection results
            
        Returns:
            Dictionary with complete analysis report
        """
        self.logger.info("Creating comprehensive phase detection report")
        
        report = {
            'dataset_info': {
                'n_samples': latent_repr.n_samples,
                'temperature_range': (
                    float(np.min(latent_repr.temperatures)),
                    float(np.max(latent_repr.temperatures))
                ),
                'latent_space_statistics': latent_repr.get_statistics()
            },
            'detection_results': {
                'critical_temperature': float(detection_result.critical_temperature),
                'confidence': float(detection_result.confidence),
                'method': detection_result.method,
                'transition_region': detection_result.transition_region
            },
            'validation': self.validate_detection(detection_result),
            'onsager_comparison': self.compare_with_onsager_solution(detection_result)
        }
        
        # Add method-specific details
        if detection_result.clustering_result:
            report['clustering_analysis'] = {
                'n_clusters': detection_result.clustering_result.n_clusters,
                'silhouette_score': float(detection_result.clustering_result.silhouette_score),
                'temperature_separation': detection_result.clustering_result.temperature_separation
            }
        
        if detection_result.gradient_analysis:
            report['gradient_analysis'] = detection_result.gradient_analysis
        
        if detection_result.ensemble_scores:
            report['ensemble_analysis'] = detection_result.ensemble_scores
        
        return report