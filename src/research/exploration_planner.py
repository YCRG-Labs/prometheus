"""
Exploration Planner for Discovery Campaign.

This module implements the exploration planner that designs exploration strategies
for target variants, including parameter space sampling, lattice sizes, temperature
points, and adaptive refinement strategies.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

from .campaign_orchestrator import TargetVariant
from .base_types import ModelVariantConfig
from ..utils.logging_utils import get_logger


@dataclass
class ExplorationPlan:
    """Plan for exploring a target variant.
    
    Attributes:
        variant: Target variant to explore
        parameter_ranges: Dictionary mapping parameter names to (min, max) ranges
        initial_points: Number of initial parameter points to sample
        refinement_strategy: Strategy for adaptive refinement ('adaptive', 'grid', 'random')
        lattice_sizes: List of lattice sizes to use
        temperature_points: Number of temperature points per parameter setting
        samples_per_temp: Number of Monte Carlo samples per temperature
        estimated_cost: Estimated computational cost in GPU-hours
        temperature_range: (T_min, T_max) for temperature sweep
        refinement_threshold: Confidence threshold below which to refine
        max_refinement_depth: Maximum depth of adaptive refinement
    """
    variant: TargetVariant
    parameter_ranges: Dict[str, Tuple[float, float]]
    initial_points: int
    refinement_strategy: str
    lattice_sizes: List[int]
    temperature_points: int
    samples_per_temp: int
    estimated_cost: float
    temperature_range: Tuple[float, float] = (1.0, 4.0)
    refinement_threshold: float = 0.8
    max_refinement_depth: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'variant_id': self.variant.variant_id,
            'variant_name': self.variant.name,
            'parameter_ranges': self.parameter_ranges,
            'initial_points': self.initial_points,
            'refinement_strategy': self.refinement_strategy,
            'lattice_sizes': self.lattice_sizes,
            'temperature_points': self.temperature_points,
            'samples_per_temp': self.samples_per_temp,
            'estimated_cost': self.estimated_cost,
            'temperature_range': self.temperature_range,
            'refinement_threshold': self.refinement_threshold,
            'max_refinement_depth': self.max_refinement_depth,
        }


class ExplorationPlanner:
    """Plan exploration strategy for target variants.
    
    This planner designs efficient exploration strategies that balance
    thoroughness with computational cost, using adaptive refinement to
    focus resources on interesting regions of parameter space.
    
    Attributes:
        logger: Logger instance
        default_lattice_sizes_2d: Default lattice sizes for 2D systems
        default_lattice_sizes_3d: Default lattice sizes for 3D systems
        cost_per_sample: Estimated GPU-seconds per Monte Carlo sample
    """
    
    def __init__(
        self,
        default_lattice_sizes_2d: Optional[List[int]] = None,
        default_lattice_sizes_3d: Optional[List[int]] = None,
        cost_per_sample: float = 0.01
    ):
        """Initialize exploration planner.
        
        Args:
            default_lattice_sizes_2d: Default lattice sizes for 2D (e.g., [32, 64])
            default_lattice_sizes_3d: Default lattice sizes for 3D (e.g., [16, 32])
            cost_per_sample: Estimated GPU-seconds per MC sample
        """
        self.logger = get_logger(__name__)
        self.default_lattice_sizes_2d = default_lattice_sizes_2d or [32, 64]
        self.default_lattice_sizes_3d = default_lattice_sizes_3d or [16, 32]
        self.cost_per_sample = cost_per_sample
        
        self.logger.info("Initialized Exploration Planner")
    
    def create_plan(
        self,
        variant: TargetVariant,
        budget: Optional[float] = None,
        initial_points: Optional[int] = None,
        refinement_strategy: str = 'adaptive'
    ) -> ExplorationPlan:
        """Create exploration plan for a target variant.
        
        Args:
            variant: Target variant to explore
            budget: Computational budget in GPU-hours (uses variant estimate if None)
            initial_points: Number of initial points (auto-determined if None)
            refinement_strategy: 'adaptive', 'grid', or 'random'
            
        Returns:
            ExplorationPlan with complete exploration strategy
        """
        self.logger.info(f"Creating exploration plan for {variant.name}")
        
        # Determine parameter ranges
        parameter_ranges = self._determine_parameter_ranges(variant)
        
        # Determine lattice sizes based on dimensionality
        lattice_sizes = self._determine_lattice_sizes(variant)
        
        # Determine number of initial points
        if initial_points is None:
            initial_points = self._determine_initial_points(
                variant, parameter_ranges, budget
            )
        
        # Determine temperature points and samples
        temp_points, samples_per_temp = self._determine_sampling_strategy(
            variant, budget
        )
        
        # Estimate temperature range
        temp_range = self._estimate_temperature_range(variant)
        
        # Calculate estimated cost
        estimated_cost = self._estimate_cost(
            initial_points=initial_points,
            lattice_sizes=lattice_sizes,
            temp_points=temp_points,
            samples_per_temp=samples_per_temp,
            dimensions=variant.model_config.dimensions
        )
        
        # Create plan
        plan = ExplorationPlan(
            variant=variant,
            parameter_ranges=parameter_ranges,
            initial_points=initial_points,
            refinement_strategy=refinement_strategy,
            lattice_sizes=lattice_sizes,
            temperature_points=temp_points,
            samples_per_temp=samples_per_temp,
            estimated_cost=estimated_cost,
            temperature_range=temp_range,
            refinement_threshold=0.8,
            max_refinement_depth=3,
        )
        
        self.logger.info(
            f"Created plan: {initial_points} initial points, "
            f"{temp_points} temps, {samples_per_temp} samples/temp, "
            f"estimated cost: {estimated_cost:.1f} GPU-hours"
        )
        
        return plan
    
    def _determine_parameter_ranges(
        self, variant: TargetVariant
    ) -> Dict[str, Tuple[float, float]]:
        """Determine parameter ranges to explore.
        
        Args:
            variant: Target variant
            
        Returns:
            Dictionary mapping parameter names to (min, max) ranges
        """
        ranges = {}
        config = variant.model_config
        
        # Long-range interactions: explore alpha parameter
        if config.interaction_type == 'long_range':
            if 'alpha' in config.interaction_params:
                # Explore around specified alpha
                alpha = config.interaction_params['alpha']
                ranges['alpha'] = (max(1.5, alpha - 0.2), min(3.5, alpha + 0.2))
            else:
                # Default range for long-range exploration
                ranges['alpha'] = (2.0, 2.5)
        
        # Diluted/disordered systems: explore disorder strength
        if config.disorder_type is not None:
            if config.disorder_strength is not None:
                # Explore around specified disorder
                disorder = config.disorder_strength
                ranges['disorder'] = (max(0.0, disorder - 0.1), min(1.0, disorder + 0.1))
            else:
                # Default disorder range
                ranges['disorder'] = (0.1, 0.5)
        
        # Frustrated systems: explore frustration parameter
        if config.interaction_type == 'frustrated':
            if 'j2_j1_ratio' in config.interaction_params:
                # Explore around specified ratio
                ratio = config.interaction_params['j2_j1_ratio']
                ranges['j2_j1'] = (max(0.0, ratio - 0.2), min(2.0, ratio + 0.2))
            else:
                # Default frustration range
                ranges['j2_j1'] = (0.0, 1.0)
        
        # External field: explore field strength
        if config.external_field is not None and config.external_field != 0.0:
            field = config.external_field
            ranges['field'] = (0.0, abs(field) * 2.0)
        
        # If no specific parameters, explore temperature only
        if not ranges:
            self.logger.info(f"No parameter ranges specified for {variant.variant_id}, will explore temperature only")
        
        return ranges
    
    def _determine_lattice_sizes(self, variant: TargetVariant) -> List[int]:
        """Determine appropriate lattice sizes.
        
        Args:
            variant: Target variant
            
        Returns:
            List of lattice sizes
        """
        dimensions = variant.model_config.dimensions
        
        if dimensions == 2:
            # 2D systems: use larger lattices
            return self.default_lattice_sizes_2d.copy()
        elif dimensions == 3:
            # 3D systems: use smaller lattices (more expensive)
            return self.default_lattice_sizes_3d.copy()
        else:
            # Default to 2D sizes
            self.logger.warning(f"Unknown dimensionality {dimensions}, using 2D defaults")
            return self.default_lattice_sizes_2d.copy()
    
    def _determine_initial_points(
        self,
        variant: TargetVariant,
        parameter_ranges: Dict[str, Tuple[float, float]],
        budget: Optional[float]
    ) -> int:
        """Determine number of initial parameter points.
        
        Args:
            variant: Target variant
            parameter_ranges: Parameter ranges to explore
            budget: Computational budget
            
        Returns:
            Number of initial points
        """
        n_params = len(parameter_ranges)
        
        if n_params == 0:
            # No parameters to explore, just one "point"
            return 1
        elif n_params == 1:
            # Single parameter: use 10-20 points
            return 15
        elif n_params == 2:
            # Two parameters: use grid of 5x5 to 10x10
            return 50  # ~7x7 grid
        else:
            # Multiple parameters: use Latin hypercube or adaptive sampling
            # Start with fewer points, rely on refinement
            return 30
    
    def _determine_sampling_strategy(
        self,
        variant: TargetVariant,
        budget: Optional[float]
    ) -> Tuple[int, int]:
        """Determine temperature points and samples per temperature.
        
        Args:
            variant: Target variant
            budget: Computational budget
            
        Returns:
            (temperature_points, samples_per_temp)
        """
        # Base strategy on variant complexity
        config = variant.model_config
        
        # Start with defaults
        temp_points = 50
        samples_per_temp = 1000
        
        # Adjust for disorder (needs more samples for averaging)
        if config.disorder_type is not None:
            samples_per_temp = 1500
            self.logger.debug("Increased samples for disorder averaging")
        
        # Adjust for frustration (needs more equilibration)
        if config.interaction_type == 'frustrated':
            samples_per_temp = 2000
            temp_points = 70
            self.logger.debug("Increased samples and temps for frustrated system")
        
        # Adjust for 3D (more expensive, use fewer temps)
        if config.dimensions == 3:
            temp_points = max(40, int(temp_points * 0.8))
            self.logger.debug(f"Reduced temps to {temp_points} for 3D system")
        
        return temp_points, samples_per_temp
    
    def _estimate_temperature_range(
        self, variant: TargetVariant
    ) -> Tuple[float, float]:
        """Estimate appropriate temperature range.
        
        Args:
            variant: Target variant
            
        Returns:
            (T_min, T_max) temperature range
        """
        config = variant.model_config
        
        # Use theoretical Tc if available
        if config.theoretical_tc is not None:
            tc = config.theoretical_tc
            # Explore ±50% around Tc
            return (tc * 0.5, tc * 1.5)
        
        # Otherwise use defaults based on dimensionality
        if config.dimensions == 2:
            # 2D Ising: Tc ≈ 2.269
            return (1.5, 3.5)
        elif config.dimensions == 3:
            # 3D Ising: Tc ≈ 4.511
            return (3.0, 6.0)
        else:
            # Default range
            return (1.0, 4.0)
    
    def _estimate_cost(
        self,
        initial_points: int,
        lattice_sizes: List[int],
        temp_points: int,
        samples_per_temp: int,
        dimensions: int
    ) -> float:
        """Estimate computational cost in GPU-hours.
        
        Args:
            initial_points: Number of parameter points
            lattice_sizes: List of lattice sizes
            temp_points: Temperature points per parameter setting
            samples_per_temp: Samples per temperature
            dimensions: System dimensionality
            
        Returns:
            Estimated cost in GPU-hours
        """
        total_samples = 0
        
        for lattice_size in lattice_sizes:
            # Calculate system size
            if dimensions == 2:
                system_size = lattice_size ** 2
            elif dimensions == 3:
                system_size = lattice_size ** 3
            else:
                system_size = lattice_size ** dimensions
            
            # Total samples for this lattice size
            samples = initial_points * temp_points * samples_per_temp
            
            # Cost scales with system size
            cost_factor = system_size / 1000.0  # Normalize to 32x32 2D
            total_samples += samples * cost_factor
        
        # Convert to GPU-hours
        gpu_seconds = total_samples * self.cost_per_sample
        gpu_hours = gpu_seconds / 3600.0
        
        # Add overhead for refinement (estimate 50% additional)
        gpu_hours *= 1.5
        
        return gpu_hours
    
    def optimize_sampling(self, plan: ExplorationPlan) -> ExplorationPlan:
        """Optimize sampling strategy for efficiency.
        
        This method adjusts the exploration plan to maximize efficiency
        while maintaining scientific rigor.
        
        Args:
            plan: Initial exploration plan
            
        Returns:
            Optimized exploration plan
        """
        self.logger.info(f"Optimizing sampling strategy for {plan.variant.name}")
        
        # Create a copy to modify
        optimized = ExplorationPlan(
            variant=plan.variant,
            parameter_ranges=plan.parameter_ranges.copy(),
            initial_points=plan.initial_points,
            refinement_strategy=plan.refinement_strategy,
            lattice_sizes=plan.lattice_sizes.copy(),
            temperature_points=plan.temperature_points,
            samples_per_temp=plan.samples_per_temp,
            estimated_cost=plan.estimated_cost,
            temperature_range=plan.temperature_range,
            refinement_threshold=plan.refinement_threshold,
            max_refinement_depth=plan.max_refinement_depth,
        )
        
        # Optimization 1: Use coarse-to-fine lattice strategy
        # Start with smaller lattices, only use larger ones for validation
        if len(optimized.lattice_sizes) > 1:
            # Keep only smallest lattice for initial exploration
            initial_lattice = [min(optimized.lattice_sizes)]
            optimized.lattice_sizes = initial_lattice
            self.logger.debug(f"Optimized to use initial lattice size: {initial_lattice}")
        
        # Optimization 2: Reduce temperature points for initial sweep
        # Use fewer temps initially, refine around Tc later
        if optimized.temperature_points > 40:
            optimized.temperature_points = 40
            self.logger.debug("Reduced initial temperature points to 40")
        
        # Optimization 3: Adjust samples based on system complexity
        # Simple systems need fewer samples
        config = optimized.variant.model_config
        if (config.disorder_type is None and 
            config.interaction_type not in ['frustrated', 'long_range']):
            # Simple system: reduce samples
            optimized.samples_per_temp = max(500, int(optimized.samples_per_temp * 0.7))
            self.logger.debug(f"Reduced samples to {optimized.samples_per_temp} for simple system")
        
        # Recalculate cost
        optimized.estimated_cost = self._estimate_cost(
            initial_points=optimized.initial_points,
            lattice_sizes=optimized.lattice_sizes,
            temp_points=optimized.temperature_points,
            samples_per_temp=optimized.samples_per_temp,
            dimensions=optimized.variant.model_config.dimensions
        )
        
        cost_reduction = ((plan.estimated_cost - optimized.estimated_cost) / 
                         plan.estimated_cost * 100)
        
        self.logger.info(
            f"Optimization complete: cost reduced by {cost_reduction:.1f}% "
            f"({plan.estimated_cost:.1f} → {optimized.estimated_cost:.1f} GPU-hours)"
        )
        
        return optimized
    
    def generate_parameter_points(
        self,
        plan: ExplorationPlan,
        strategy: Optional[str] = None
    ) -> List[Dict[str, float]]:
        """Generate parameter points for exploration.
        
        Args:
            plan: Exploration plan
            strategy: Sampling strategy ('grid', 'random', 'latin_hypercube')
                     Uses plan.refinement_strategy if None
            
        Returns:
            List of parameter dictionaries
        """
        if strategy is None:
            strategy = plan.refinement_strategy
        
        if not plan.parameter_ranges:
            # No parameters to vary, return single empty point
            return [{}]
        
        param_names = list(plan.parameter_ranges.keys())
        n_params = len(param_names)
        n_points = plan.initial_points
        
        if strategy == 'grid':
            points = self._generate_grid_points(plan.parameter_ranges, n_points)
        elif strategy == 'random':
            points = self._generate_random_points(plan.parameter_ranges, n_points)
        elif strategy == 'latin_hypercube' or strategy == 'adaptive':
            points = self._generate_latin_hypercube_points(plan.parameter_ranges, n_points)
        else:
            self.logger.warning(f"Unknown strategy '{strategy}', using grid")
            points = self._generate_grid_points(plan.parameter_ranges, n_points)
        
        self.logger.info(f"Generated {len(points)} parameter points using {strategy} strategy")
        
        return points
    
    def _generate_grid_points(
        self,
        parameter_ranges: Dict[str, Tuple[float, float]],
        n_points: int
    ) -> List[Dict[str, float]]:
        """Generate grid of parameter points."""
        param_names = list(parameter_ranges.keys())
        n_params = len(param_names)
        
        # Determine grid size per dimension
        points_per_dim = int(np.ceil(n_points ** (1.0 / n_params)))
        
        # Generate grid for each parameter
        grids = []
        for param_name in param_names:
            min_val, max_val = parameter_ranges[param_name]
            grid = np.linspace(min_val, max_val, points_per_dim)
            grids.append(grid)
        
        # Create meshgrid
        meshgrids = np.meshgrid(*grids, indexing='ij')
        
        # Flatten and create parameter dictionaries
        points = []
        for i in range(meshgrids[0].size):
            point = {}
            for j, param_name in enumerate(param_names):
                point[param_name] = float(meshgrids[j].flat[i])
            points.append(point)
        
        return points
    
    def _generate_random_points(
        self,
        parameter_ranges: Dict[str, Tuple[float, float]],
        n_points: int
    ) -> List[Dict[str, float]]:
        """Generate random parameter points."""
        points = []
        
        for _ in range(n_points):
            point = {}
            for param_name, (min_val, max_val) in parameter_ranges.items():
                point[param_name] = np.random.uniform(min_val, max_val)
            points.append(point)
        
        return points
    
    def _generate_latin_hypercube_points(
        self,
        parameter_ranges: Dict[str, Tuple[float, float]],
        n_points: int
    ) -> List[Dict[str, float]]:
        """Generate Latin hypercube sample of parameter points."""
        param_names = list(parameter_ranges.keys())
        n_params = len(param_names)
        
        # Generate Latin hypercube sample in [0, 1]^n
        # Simple implementation: stratified random sampling
        points_normalized = np.zeros((n_points, n_params))
        
        for i in range(n_params):
            # Create stratified samples
            intervals = np.linspace(0, 1, n_points + 1)
            samples = np.random.uniform(intervals[:-1], intervals[1:])
            # Shuffle to break correlation between dimensions
            np.random.shuffle(samples)
            points_normalized[:, i] = samples
        
        # Scale to actual parameter ranges
        points = []
        for i in range(n_points):
            point = {}
            for j, param_name in enumerate(param_names):
                min_val, max_val = parameter_ranges[param_name]
                value = min_val + points_normalized[i, j] * (max_val - min_val)
                point[param_name] = float(value)
            points.append(point)
        
        return points
    
    def save_plan(self, plan: ExplorationPlan, output_path: str) -> None:
        """Save exploration plan to file.
        
        Args:
            plan: Exploration plan
            output_path: Path to save plan (JSON format)
        """
        import json
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(plan.to_dict(), f, indent=2)
        
        self.logger.info(f"Saved exploration plan to {output_path}")
