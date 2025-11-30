"""
Parameter Space Explorer for systematic exploration of model variant parameter spaces.

This module provides tools for generating sampling points across parameter spaces
using various strategies including grid search, adaptive refinement, Bayesian
optimization, and random sampling.

Includes confidence-guided adaptive exploration (Task 16.5) that uses validation
confidence to guide the exploration strategy:
- High confidence → explore new regions
- Low confidence → refine current region
"""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from dataclasses import dataclass
from .base_types import ExplorationStrategy
from ..utils.logging_utils import get_logger


@dataclass
class ParameterPoint:
    """A single point in parameter space.
    
    Attributes:
        parameters: Dictionary of parameter values
        uncertainty: Estimated uncertainty at this point (for adaptive methods)
        explored: Whether this point has been explored
        results: Results from exploration (if explored)
    """
    parameters: Dict[str, float]
    uncertainty: float = 1.0
    explored: bool = False
    results: Optional[Any] = None


class ParameterSpaceExplorer:
    """Explore parameter spaces of model variants systematically.
    
    This class provides various strategies for sampling parameter spaces,
    including grid search, adaptive refinement, Bayesian optimization,
    and random sampling.
    
    Attributes:
        variant_id: ID of the model variant being explored
        parameter_ranges: Dictionary mapping parameter names to (min, max) tuples
        explored_points: List of parameter points that have been explored
    """
    
    def __init__(
        self,
        variant_id: str,
        parameter_ranges: Dict[str, Tuple[float, float]]
    ):
        """Initialize parameter space explorer.
        
        Args:
            variant_id: ID of the model variant to explore
            parameter_ranges: Dictionary of parameter ranges, e.g.,
                {'temperature': (1.0, 5.0), 'field': (0.0, 1.0)}
        
        Raises:
            ValueError: If parameter ranges are invalid
        """
        self.variant_id = variant_id
        self.parameter_ranges = parameter_ranges
        self.explored_points: List[ParameterPoint] = []
        self.logger = get_logger(__name__)
        
        # Validate parameter ranges
        for param_name, (min_val, max_val) in parameter_ranges.items():
            if min_val >= max_val:
                raise ValueError(
                    f"Invalid range for {param_name}: min ({min_val}) >= max ({max_val})"
                )
    
    def generate_sampling_points(
        self,
        strategy: ExplorationStrategy
    ) -> np.ndarray:
        """Generate parameter points to sample based on strategy.
        
        Args:
            strategy: Exploration strategy configuration
        
        Returns:
            Array of parameter points, shape (n_points, n_parameters)
        
        Raises:
            ValueError: If strategy method is unknown
        """
        if strategy.method == 'grid':
            return self._grid_sampling(strategy.n_points)
        elif strategy.method == 'random':
            return self._random_sampling(strategy.n_points)
        elif strategy.method == 'adaptive':
            return self._adaptive_sampling(strategy)
        elif strategy.method == 'bayesian':
            return self._bayesian_sampling(strategy)
        else:
            raise ValueError(f"Unknown exploration method: {strategy.method}")
    
    def _grid_sampling(self, n_points: int) -> np.ndarray:
        """Generate uniform grid sampling across parameter space.
        
        Args:
            n_points: Total number of points to generate
        
        Returns:
            Array of parameter points
        """
        n_params = len(self.parameter_ranges)
        points_per_dim = int(np.ceil(n_points ** (1.0 / n_params)))
        
        # Create grid for each parameter
        grids = []
        param_names = sorted(self.parameter_ranges.keys())
        
        for param_name in param_names:
            min_val, max_val = self.parameter_ranges[param_name]
            grid = np.linspace(min_val, max_val, points_per_dim)
            grids.append(grid)
        
        # Create meshgrid
        mesh = np.meshgrid(*grids, indexing='ij')
        
        # Flatten and combine
        points = np.column_stack([m.ravel() for m in mesh])
        
        # Limit to requested number of points
        if len(points) > n_points:
            indices = np.random.choice(len(points), n_points, replace=False)
            points = points[indices]
        
        return points

    
    def _random_sampling(self, n_points: int) -> np.ndarray:
        """Generate random sampling across parameter space.
        
        Args:
            n_points: Number of points to generate
        
        Returns:
            Array of parameter points
        """
        n_params = len(self.parameter_ranges)
        points = np.zeros((n_points, n_params))
        
        param_names = sorted(self.parameter_ranges.keys())
        for i, param_name in enumerate(param_names):
            min_val, max_val = self.parameter_ranges[param_name]
            points[:, i] = np.random.uniform(min_val, max_val, n_points)
        
        return points
    
    def _adaptive_sampling(self, strategy: ExplorationStrategy) -> np.ndarray:
        """Generate adaptive sampling with refinement in high-uncertainty regions.
        
        This method starts with a coarse grid and iteratively refines regions
        with high uncertainty or interesting behavior.
        
        Args:
            strategy: Exploration strategy with refinement parameters
        
        Returns:
            Array of parameter points
        """
        # Start with initial grid
        initial_points_per_iteration = strategy.n_points // (strategy.refinement_iterations + 1)
        points = self._grid_sampling(initial_points_per_iteration)
        
        # If no refinement iterations, return initial grid
        if strategy.refinement_iterations == 0:
            return points
        
        # For adaptive refinement, we need explored points with uncertainty
        # If we have explored points, use them for refinement
        if len(self.explored_points) > 0:
            # Identify high-uncertainty regions
            high_uncertainty_regions = self._identify_high_uncertainty_regions()
            
            # Generate refined points in those regions
            refined_points = self._generate_refined_points(
                high_uncertainty_regions,
                initial_points_per_iteration
            )
            
            points = np.vstack([points, refined_points])
        
        return points
    
    def _identify_high_uncertainty_regions(self) -> List[Dict[str, Tuple[float, float]]]:
        """Identify parameter regions with high uncertainty.
        
        Returns:
            List of parameter range dictionaries for high-uncertainty regions
        """
        if len(self.explored_points) == 0:
            return []
        
        # Sort points by uncertainty
        sorted_points = sorted(
            self.explored_points,
            key=lambda p: p.uncertainty,
            reverse=True
        )
        
        # Take top 20% most uncertain points
        n_top = max(1, len(sorted_points) // 5)
        top_uncertain = sorted_points[:n_top]
        
        # Define regions around these points
        regions = []
        param_names = sorted(self.parameter_ranges.keys())
        
        for point in top_uncertain:
            region = {}
            for param_name in param_names:
                param_value = point.parameters[param_name]
                min_val, max_val = self.parameter_ranges[param_name]
                
                # Create region ±10% around the point
                range_size = max_val - min_val
                delta = 0.1 * range_size
                
                region_min = max(min_val, param_value - delta)
                region_max = min(max_val, param_value + delta)
                
                region[param_name] = (region_min, region_max)
            
            regions.append(region)
        
        return regions
    
    def _generate_refined_points(
        self,
        regions: List[Dict[str, Tuple[float, float]]],
        n_points_per_region: int
    ) -> np.ndarray:
        """Generate refined sampling points in specified regions.
        
        Args:
            regions: List of parameter range dictionaries
            n_points_per_region: Number of points to generate per region
        
        Returns:
            Array of refined parameter points
        """
        if len(regions) == 0:
            return np.array([]).reshape(0, len(self.parameter_ranges))
        
        all_points = []
        param_names = sorted(self.parameter_ranges.keys())
        
        for region in regions:
            # Generate random points in this region
            region_points = np.zeros((n_points_per_region, len(param_names)))
            
            for i, param_name in enumerate(param_names):
                min_val, max_val = region[param_name]
                region_points[:, i] = np.random.uniform(
                    min_val, max_val, n_points_per_region
                )
            
            all_points.append(region_points)
        
        return np.vstack(all_points)

    
    def _bayesian_sampling(self, strategy: ExplorationStrategy) -> np.ndarray:
        """Generate sampling points using Bayesian optimization.
        
        This method uses a Gaussian process to model the parameter space
        and selects points that maximize an acquisition function (expected
        improvement or upper confidence bound).
        
        Args:
            strategy: Exploration strategy configuration
        
        Returns:
            Array of parameter points
        """
        # Start with a small initial random sample
        n_initial = min(10, strategy.n_points // 2)
        initial_points = self._random_sampling(n_initial)
        
        # If we don't have enough explored points for GP, return random sampling
        if len(self.explored_points) < 3:
            return self._random_sampling(strategy.n_points)
        
        # Build Gaussian Process model from explored points
        X_train, y_train = self._prepare_gp_training_data()
        
        if len(X_train) < 3:
            # Not enough data for GP, fall back to random
            return self._random_sampling(strategy.n_points)
        
        # Generate candidate points
        n_candidates = 1000
        candidates = self._random_sampling(n_candidates)
        
        # Compute acquisition function for each candidate
        acquisition_values = self._compute_acquisition_function(
            candidates, X_train, y_train
        )
        
        # Select top points based on acquisition function
        n_new = strategy.n_points - n_initial
        top_indices = np.argsort(acquisition_values)[-n_new:]
        selected_points = candidates[top_indices]
        
        # Combine initial and selected points
        all_points = np.vstack([initial_points, selected_points])
        
        return all_points
    
    def _prepare_gp_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for Gaussian Process from explored points.
        
        Returns:
            Tuple of (X_train, y_train) where X_train is parameter values
            and y_train is the objective (negative uncertainty for exploration)
        """
        param_names = sorted(self.parameter_ranges.keys())
        n_points = len(self.explored_points)
        n_params = len(param_names)
        
        X_train = np.zeros((n_points, n_params))
        y_train = np.zeros(n_points)
        
        for i, point in enumerate(self.explored_points):
            for j, param_name in enumerate(param_names):
                X_train[i, j] = point.parameters[param_name]
            
            # Use negative uncertainty as objective (we want to explore uncertain regions)
            y_train[i] = -point.uncertainty
        
        return X_train, y_train
    
    def _compute_acquisition_function(
        self,
        candidates: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> np.ndarray:
        """Compute acquisition function for candidate points.
        
        Uses Upper Confidence Bound (UCB) acquisition function:
        UCB(x) = μ(x) + κ * σ(x)
        
        Args:
            candidates: Candidate parameter points
            X_train: Training parameter values
            y_train: Training objective values
        
        Returns:
            Acquisition values for each candidate
        """
        # Simple GP implementation using RBF kernel
        # For production, consider using scikit-learn's GaussianProcessRegressor
        
        # Compute kernel matrix for training data
        K = self._rbf_kernel(X_train, X_train)
        K += 1e-6 * np.eye(len(K))  # Add noise for numerical stability
        
        # Compute kernel between candidates and training data
        K_star = self._rbf_kernel(candidates, X_train)
        
        # Compute mean and variance predictions
        try:
            K_inv = np.linalg.inv(K)
            mu = K_star @ K_inv @ y_train
            
            K_star_star = self._rbf_kernel(candidates, candidates, diag_only=True)
            var = K_star_star - np.sum(K_star @ K_inv * K_star, axis=1)
            sigma = np.sqrt(np.maximum(var, 0))
        except np.linalg.LinAlgError:
            # If matrix inversion fails, fall back to random exploration
            mu = np.zeros(len(candidates))
            sigma = np.ones(len(candidates))
        
        # UCB acquisition function with κ=2.0
        kappa = 2.0
        ucb = mu + kappa * sigma
        
        return ucb
    
    def _rbf_kernel(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        diag_only: bool = False
    ) -> np.ndarray:
        """Compute RBF (Gaussian) kernel between two sets of points.
        
        K(x, x') = exp(-||x - x'||² / (2 * length_scale²))
        
        Args:
            X1: First set of points (n1, n_features)
            X2: Second set of points (n2, n_features)
            diag_only: If True, only compute diagonal elements
        
        Returns:
            Kernel matrix (n1, n2) or diagonal (n1,) if diag_only=True
        """
        length_scale = 1.0
        
        if diag_only:
            # Diagonal is always 1 for RBF kernel
            return np.ones(len(X1))
        
        # Compute squared Euclidean distances
        X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
        distances_sq = X1_norm + X2_norm - 2 * X1 @ X2.T
        
        # Compute RBF kernel
        K = np.exp(-distances_sq / (2 * length_scale**2))
        
        return K

    
    def adaptive_refinement(
        self,
        current_results: Dict[str, Any],
        metric: str = 'uncertainty'
    ) -> np.ndarray:
        """Generate refined sampling based on current results.
        
        This method analyzes current exploration results and generates
        additional sampling points in regions that need more exploration.
        
        Args:
            current_results: Dictionary containing current exploration results
            metric: Metric to use for refinement ('uncertainty', 'gradient', 'anomaly')
        
        Returns:
            Array of refined parameter points
        
        Raises:
            ValueError: If metric is unknown
        """
        if metric == 'uncertainty':
            return self._refine_by_uncertainty(current_results)
        elif metric == 'gradient':
            return self._refine_by_gradient(current_results)
        elif metric == 'anomaly':
            return self._refine_by_anomaly(current_results)
        else:
            raise ValueError(f"Unknown refinement metric: {metric}")
    
    def _refine_by_uncertainty(self, results: Dict[str, Any]) -> np.ndarray:
        """Refine sampling in high-uncertainty regions.
        
        Args:
            results: Current exploration results
        
        Returns:
            Array of refined parameter points
        """
        # Identify high-uncertainty regions
        regions = self._identify_high_uncertainty_regions()
        
        if len(regions) == 0:
            # No high-uncertainty regions, sample uniformly
            return self._random_sampling(10)
        
        # Generate refined points
        return self._generate_refined_points(regions, n_points_per_region=5)
    
    def _refine_by_gradient(self, results: Dict[str, Any]) -> np.ndarray:
        """Refine sampling in regions with high gradients.
        
        High gradients indicate rapid changes in properties, often near
        phase transitions.
        
        Args:
            results: Current exploration results
        
        Returns:
            Array of refined parameter points
        """
        if len(self.explored_points) < 2:
            return self._random_sampling(10)
        
        # Compute approximate gradients between neighboring points
        param_names = sorted(self.parameter_ranges.keys())
        high_gradient_points = []
        
        for i, point in enumerate(self.explored_points):
            if not point.explored or point.results is None:
                continue
            
            # Find nearby points
            nearby = self._find_nearby_points(point, radius=0.2)
            
            if len(nearby) == 0:
                continue
            
            # Compute gradient magnitude
            gradient = self._estimate_gradient(point, nearby)
            
            if gradient > 0.5:  # Threshold for "high" gradient
                high_gradient_points.append(point)
        
        if len(high_gradient_points) == 0:
            return self._random_sampling(10)
        
        # Create regions around high-gradient points
        regions = []
        for point in high_gradient_points:
            region = {}
            for param_name in param_names:
                param_value = point.parameters[param_name]
                min_val, max_val = self.parameter_ranges[param_name]
                range_size = max_val - min_val
                delta = 0.05 * range_size  # Smaller region for gradient refinement
                
                region[param_name] = (
                    max(min_val, param_value - delta),
                    min(max_val, param_value + delta)
                )
            regions.append(region)
        
        return self._generate_refined_points(regions, n_points_per_region=5)
    
    def _refine_by_anomaly(self, results: Dict[str, Any]) -> np.ndarray:
        """Refine sampling near anomalous behavior.
        
        Args:
            results: Current exploration results
        
        Returns:
            Array of refined parameter points
        """
        # Look for anomalous points in results
        anomalous_points = []
        
        for point in self.explored_points:
            if not point.explored or point.results is None:
                continue
            
            # Check if point has anomalous properties
            if self._is_anomalous(point):
                anomalous_points.append(point)
        
        if len(anomalous_points) == 0:
            return self._random_sampling(10)
        
        # Create regions around anomalous points
        param_names = sorted(self.parameter_ranges.keys())
        regions = []
        
        for point in anomalous_points:
            region = {}
            for param_name in param_names:
                param_value = point.parameters[param_name]
                min_val, max_val = self.parameter_ranges[param_name]
                range_size = max_val - min_val
                delta = 0.1 * range_size
                
                region[param_name] = (
                    max(min_val, param_value - delta),
                    min(max_val, param_value + delta)
                )
            regions.append(region)
        
        return self._generate_refined_points(regions, n_points_per_region=5)
    
    def _find_nearby_points(
        self,
        point: ParameterPoint,
        radius: float = 0.2
    ) -> List[ParameterPoint]:
        """Find points within a given radius in parameter space.
        
        Args:
            point: Reference point
            radius: Search radius (as fraction of parameter range)
        
        Returns:
            List of nearby points
        """
        nearby = []
        param_names = sorted(self.parameter_ranges.keys())
        
        for other in self.explored_points:
            if other is point or not other.explored:
                continue
            
            # Compute normalized distance
            distance = 0.0
            for param_name in param_names:
                min_val, max_val = self.parameter_ranges[param_name]
                range_size = max_val - min_val
                
                diff = point.parameters[param_name] - other.parameters[param_name]
                normalized_diff = diff / range_size
                distance += normalized_diff**2
            
            distance = np.sqrt(distance)
            
            if distance < radius:
                nearby.append(other)
        
        return nearby
    
    def _estimate_gradient(
        self,
        point: ParameterPoint,
        nearby: List[ParameterPoint]
    ) -> float:
        """Estimate gradient magnitude at a point.
        
        Args:
            point: Reference point
            nearby: Nearby points for gradient estimation
        
        Returns:
            Estimated gradient magnitude
        """
        if len(nearby) == 0:
            return 0.0
        
        # Use uncertainty as the quantity to compute gradient of
        # In practice, this could be any property (Tc, exponents, etc.)
        value = point.uncertainty
        
        gradients = []
        param_names = sorted(self.parameter_ranges.keys())
        
        for other in nearby:
            # Compute distance
            distance = 0.0
            for param_name in param_names:
                diff = point.parameters[param_name] - other.parameters[param_name]
                distance += diff**2
            distance = np.sqrt(distance)
            
            if distance < 1e-10:
                continue
            
            # Compute gradient
            value_diff = abs(value - other.uncertainty)
            gradient = value_diff / distance
            gradients.append(gradient)
        
        if len(gradients) == 0:
            return 0.0
        
        return np.mean(gradients)
    
    def _is_anomalous(self, point: ParameterPoint) -> bool:
        """Check if a point exhibits anomalous behavior.
        
        Args:
            point: Point to check
        
        Returns:
            True if point is anomalous
        """
        # Simple heuristic: high uncertainty indicates anomalous behavior
        # In practice, this would check for anomalous exponents, phase transitions, etc.
        return point.uncertainty > 0.7
    
    def identify_anomalous_regions(
        self,
        results: Dict[str, Any]
    ) -> List[Dict[str, Tuple[float, float]]]:
        """Identify parameter regions with anomalous behavior.
        
        Args:
            results: Exploration results containing VAE analysis outputs
        
        Returns:
            List of parameter range dictionaries for anomalous regions
        """
        anomalous_points = []
        
        for point in self.explored_points:
            if not point.explored or point.results is None:
                continue
            
            if self._is_anomalous(point):
                anomalous_points.append(point)
        
        # Create regions around anomalous points
        param_names = sorted(self.parameter_ranges.keys())
        regions = []
        
        for point in anomalous_points:
            region = {}
            for param_name in param_names:
                param_value = point.parameters[param_name]
                min_val, max_val = self.parameter_ranges[param_name]
                range_size = max_val - min_val
                delta = 0.15 * range_size
                
                region[param_name] = (
                    max(min_val, param_value - delta),
                    min(max_val, param_value + delta)
                )
            regions.append(region)
        
        return regions
    
    def update_explored_point(
        self,
        parameters: Dict[str, float],
        results: Any,
        uncertainty: float
    ) -> None:
        """Update an explored point with results.
        
        Args:
            parameters: Parameter values for the point
            results: Exploration results
            uncertainty: Estimated uncertainty at this point
        """
        # Find or create point
        point = None
        for p in self.explored_points:
            if self._parameters_match(p.parameters, parameters):
                point = p
                break
        
        if point is None:
            point = ParameterPoint(parameters=parameters.copy())
            self.explored_points.append(point)
        
        # Update point
        point.explored = True
        point.results = results
        point.uncertainty = uncertainty
    
    def _parameters_match(
        self,
        params1: Dict[str, float],
        params2: Dict[str, float],
        tolerance: float = 1e-6
    ) -> bool:
        """Check if two parameter dictionaries match within tolerance.
        
        Args:
            params1: First parameter dictionary
            params2: Second parameter dictionary
            tolerance: Matching tolerance
        
        Returns:
            True if parameters match
        """
        if set(params1.keys()) != set(params2.keys()):
            return False
        
        for key in params1:
            if abs(params1[key] - params2[key]) > tolerance:
                return False
        
        return True
    
    def confidence_guided_sampling(
        self,
        n_points: int,
        confidence_threshold: float = 0.8,
        exploration_mode: str = 'adaptive'
    ) -> np.ndarray:
        """Generate sampling points guided by validation confidence.
        
        This implements the temporal workflow pattern discovered during system
        implementation: use validation confidence to guide exploration strategy.
        
        Strategy:
        - High confidence (>threshold) → Move to new regions (exploration)
        - Low confidence (<threshold) → Refine current region (exploitation)
        
        This creates an adaptive workflow where:
        1. Initial exploration identifies interesting regions
        2. Low confidence triggers local refinement
        3. High confidence triggers move to unexplored regions
        4. Process repeats until parameter space is well-characterized
        
        Args:
            n_points: Number of points to generate
            confidence_threshold: Threshold for switching between exploration/exploitation
            exploration_mode: Mode for new region exploration ('random', 'grid', 'bayesian')
        
        Returns:
            Array of parameter points guided by confidence
        """
        if len(self.explored_points) == 0:
            # No explored points yet, start with initial sampling
            return self._random_sampling(n_points)
        
        # Calculate average confidence from explored points
        explored = [p for p in self.explored_points if p.explored]
        if not explored:
            return self._random_sampling(n_points)
        
        # Use inverse uncertainty as proxy for confidence
        # High uncertainty → low confidence → need refinement
        # Low uncertainty → high confidence → can explore new regions
        confidences = [1.0 - p.uncertainty for p in explored]
        avg_confidence = np.mean(confidences)
        
        self.logger.info(
            f"Confidence-guided sampling: avg_confidence={avg_confidence:.2%}, "
            f"threshold={confidence_threshold:.2%}"
        )
        
        # Decision: explore new regions or refine current region?
        if avg_confidence > confidence_threshold:
            # High confidence → explore new regions
            self.logger.info("High confidence detected - exploring new regions")
            return self._explore_new_regions(n_points, exploration_mode)
        else:
            # Low confidence → refine current region
            self.logger.info("Low confidence detected - refining current region")
            return self._refine_low_confidence_regions(n_points)
    
    def _explore_new_regions(
        self,
        n_points: int,
        mode: str = 'random'
    ) -> np.ndarray:
        """Explore new regions of parameter space.
        
        Identifies unexplored regions and generates sampling points there.
        
        Args:
            n_points: Number of points to generate
            mode: Exploration mode ('random', 'grid', 'bayesian')
        
        Returns:
            Array of parameter points in unexplored regions
        """
        # Identify explored regions
        explored_regions = self._identify_explored_regions()
        
        # Generate candidates in unexplored regions
        candidates = []
        max_attempts = n_points * 10
        attempts = 0
        
        param_names = sorted(self.parameter_ranges.keys())
        
        while len(candidates) < n_points and attempts < max_attempts:
            # Generate candidate point
            if mode == 'grid':
                # Grid sampling in unexplored regions
                candidate = self._generate_grid_point_in_unexplored(explored_regions)
            elif mode == 'bayesian':
                # Bayesian optimization for unexplored regions
                candidate = self._generate_bayesian_point_in_unexplored(explored_regions)
            else:
                # Random sampling in unexplored regions
                candidate = {}
                for param_name in param_names:
                    min_val, max_val = self.parameter_ranges[param_name]
                    candidate[param_name] = np.random.uniform(min_val, max_val)
            
            # Check if candidate is in unexplored region
            if not self._is_in_explored_region(candidate, explored_regions):
                candidates.append(candidate)
            
            attempts += 1
        
        # Convert to array
        if not candidates:
            # Fallback to random sampling if no unexplored regions found
            return self._random_sampling(n_points)
        
        points = np.zeros((len(candidates), len(param_names)))
        for i, candidate in enumerate(candidates):
            for j, param_name in enumerate(param_names):
                points[i, j] = candidate[param_name]
        
        return points
    
    def _refine_low_confidence_regions(self, n_points: int) -> np.ndarray:
        """Refine regions with low validation confidence.
        
        Identifies points with low confidence (high uncertainty) and generates
        additional sampling points nearby to improve confidence.
        
        Args:
            n_points: Number of refinement points to generate
        
        Returns:
            Array of parameter points for refinement
        """
        # Find low-confidence points (high uncertainty)
        explored = [p for p in self.explored_points if p.explored]
        
        # Sort by uncertainty (descending)
        low_confidence_points = sorted(explored, key=lambda p: p.uncertainty, reverse=True)
        
        # Take top 30% lowest confidence points
        n_low_conf = max(1, len(low_confidence_points) // 3)
        low_confidence_points = low_confidence_points[:n_low_conf]
        
        # Create refined regions around these points
        param_names = sorted(self.parameter_ranges.keys())
        regions = []
        
        for point in low_confidence_points:
            region = {}
            for param_name in param_names:
                param_value = point.parameters[param_name]
                min_val, max_val = self.parameter_ranges[param_name]
                range_size = max_val - min_val
                
                # Smaller refinement region (5% of range)
                delta = 0.05 * range_size
                
                region[param_name] = (
                    max(min_val, param_value - delta),
                    min(max_val, param_value + delta)
                )
            regions.append(region)
        
        # Generate refined points
        points_per_region = max(1, n_points // len(regions))
        return self._generate_refined_points(regions, points_per_region)
    
    def _identify_explored_regions(self) -> List[Dict[str, Tuple[float, float]]]:
        """Identify regions that have been explored.
        
        Returns:
            List of parameter range dictionaries for explored regions
        """
        if len(self.explored_points) == 0:
            return []
        
        explored = [p for p in self.explored_points if p.explored]
        if not explored:
            return []
        
        param_names = sorted(self.parameter_ranges.keys())
        regions = []
        
        # Create regions around each explored point
        for point in explored:
            region = {}
            for param_name in param_names:
                param_value = point.parameters[param_name]
                min_val, max_val = self.parameter_ranges[param_name]
                range_size = max_val - min_val
                
                # Region size based on density of explored points
                delta = 0.15 * range_size
                
                region[param_name] = (
                    max(min_val, param_value - delta),
                    min(max_val, param_value + delta)
                )
            regions.append(region)
        
        return regions
    
    def _is_in_explored_region(
        self,
        point: Dict[str, float],
        explored_regions: List[Dict[str, Tuple[float, float]]]
    ) -> bool:
        """Check if a point is within any explored region.
        
        Args:
            point: Parameter point to check
            explored_regions: List of explored region definitions
        
        Returns:
            True if point is in an explored region
        """
        for region in explored_regions:
            in_region = True
            for param_name, (min_val, max_val) in region.items():
                if not (min_val <= point[param_name] <= max_val):
                    in_region = False
                    break
            
            if in_region:
                return True
        
        return False
    
    def _generate_grid_point_in_unexplored(
        self,
        explored_regions: List[Dict[str, Tuple[float, float]]]
    ) -> Dict[str, float]:
        """Generate a grid point in unexplored region.
        
        Args:
            explored_regions: List of explored regions to avoid
        
        Returns:
            Parameter point dictionary
        """
        # Simple implementation: random point (grid would require more complex logic)
        param_names = sorted(self.parameter_ranges.keys())
        point = {}
        for param_name in param_names:
            min_val, max_val = self.parameter_ranges[param_name]
            point[param_name] = np.random.uniform(min_val, max_val)
        return point
    
    def _generate_bayesian_point_in_unexplored(
        self,
        explored_regions: List[Dict[str, Tuple[float, float]]]
    ) -> Dict[str, float]:
        """Generate a Bayesian optimization point in unexplored region.
        
        Args:
            explored_regions: List of explored regions to avoid
        
        Returns:
            Parameter point dictionary
        """
        # Simple implementation: random point (Bayesian would require GP model)
        param_names = sorted(self.parameter_ranges.keys())
        point = {}
        for param_name in param_names:
            min_val, max_val = self.parameter_ranges[param_name]
            point[param_name] = np.random.uniform(min_val, max_val)
        return point
    
    def update_confidence(
        self,
        parameters: Dict[str, float],
        confidence: float
    ) -> None:
        """Update confidence score for a parameter point.
        
        This method allows external validation systems (e.g., ConfidenceAggregator)
        to update the confidence scores used for adaptive exploration.
        
        Args:
            parameters: Parameter values for the point
            confidence: Validation confidence (0.0 to 1.0)
        """
        # Find or create point
        point = None
        for p in self.explored_points:
            if self._parameters_match(p.parameters, parameters):
                point = p
                break
        
        if point is None:
            point = ParameterPoint(parameters=parameters.copy())
            self.explored_points.append(point)
        
        # Update uncertainty (inverse of confidence)
        point.uncertainty = 1.0 - confidence
        point.explored = True
    
    def get_exploration_summary(self) -> Dict[str, Any]:
        """Get summary of exploration progress.
        
        Returns:
            Dictionary with exploration statistics
        """
        n_explored = sum(1 for p in self.explored_points if p.explored)
        
        if n_explored == 0:
            avg_uncertainty = 1.0
            max_uncertainty = 1.0
            avg_confidence = 0.0
        else:
            explored = [p for p in self.explored_points if p.explored]
            avg_uncertainty = np.mean([p.uncertainty for p in explored])
            max_uncertainty = np.max([p.uncertainty for p in explored])
            avg_confidence = 1.0 - avg_uncertainty
        
        return {
            'variant_id': self.variant_id,
            'n_parameters': len(self.parameter_ranges),
            'parameter_ranges': self.parameter_ranges,
            'n_explored': n_explored,
            'n_total_points': len(self.explored_points),
            'avg_uncertainty': float(avg_uncertainty),
            'max_uncertainty': float(max_uncertainty),
            'avg_confidence': float(avg_confidence),
        }
