"""
Core data structures for the Novel Ising Model Research Explorer.

This module defines the fundamental data types used throughout the research
exploration system, including model configurations, hypotheses, simulation
results, and analysis outputs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np


@dataclass
class ModelVariantConfig:
    """Configuration for an Ising model variant.
    
    Attributes:
        name: Unique identifier for the model variant
        dimensions: Spatial dimensions (2 or 3)
        lattice_geometry: Type of lattice ('square', 'triangular', 'honeycomb', etc.)
        interaction_type: Type of interactions ('nearest_neighbor', 'long_range', 'frustrated')
        interaction_params: Parameters for interaction (e.g., {'alpha': 2.5} for long-range)
        disorder_type: Type of disorder ('quenched', 'annealed', None)
        disorder_strength: Strength of disorder (0.0 = no disorder)
        external_field: External magnetic field strength
        custom_energy_function: Optional custom energy calculation function
        theoretical_tc: Known theoretical critical temperature (if available)
        theoretical_exponents: Known theoretical critical exponents (if available)
    """
    name: str
    dimensions: int
    lattice_geometry: str = 'square'
    interaction_type: str = 'nearest_neighbor'
    interaction_params: Dict[str, float] = field(default_factory=dict)
    disorder_type: Optional[str] = None
    disorder_strength: float = 0.0
    external_field: float = 0.0
    custom_energy_function: Optional[Callable] = None
    theoretical_tc: Optional[float] = None
    theoretical_exponents: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.dimensions not in [2, 3]:
            raise ValueError(f"Dimensions must be 2 or 3, got {self.dimensions}")
        
        if self.disorder_strength < 0.0 or self.disorder_strength > 1.0:
            raise ValueError(f"Disorder strength must be in [0, 1], got {self.disorder_strength}")
        
        valid_geometries = ['square', 'cubic', 'triangular', 'honeycomb', 'kagome']
        if self.lattice_geometry not in valid_geometries:
            raise ValueError(f"Unknown lattice geometry: {self.lattice_geometry}")


@dataclass
class ResearchHypothesis:
    """Definition of a testable research hypothesis.
    
    Attributes:
        hypothesis_id: Unique identifier for the hypothesis
        description: Human-readable description of the hypothesis
        variant_id: ID of the model variant this hypothesis applies to
        parameter_ranges: Parameter ranges to explore (e.g., {'temperature': (1.0, 5.0)})
        predictions: Predicted critical exponent values (e.g., {'beta': 0.35, 'nu': 0.65})
        prediction_errors: Acceptable error margins for predictions
        universality_class: Expected universality class (if applicable)
        status: Current validation status ('pending', 'validated', 'refuted', 'inconclusive')
        confidence: Confidence level in validation (0.0 to 1.0)
        validation_results: Detailed validation results (populated after validation)
    """
    hypothesis_id: str
    description: str
    variant_id: str
    parameter_ranges: Dict[str, Tuple[float, float]]
    predictions: Dict[str, float]
    prediction_errors: Dict[str, float] = field(default_factory=dict)
    universality_class: Optional[str] = None
    status: str = 'pending'
    confidence: float = 0.0
    validation_results: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate hypothesis parameters."""
        valid_statuses = ['pending', 'validated', 'refuted', 'inconclusive']
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid status: {self.status}")
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


@dataclass
class SimulationData:
    """Monte Carlo simulation output data.
    
    Attributes:
        variant_id: ID of the model variant that was simulated
        parameters: Parameter values used in simulation
        temperatures: Array of temperature values
        configurations: Spin configurations (shape: n_temps, n_samples, *lattice_shape)
        magnetizations: Magnetization values for each configuration
        energies: Energy values for each configuration
        metadata: Additional metadata (equilibration steps, MC steps, etc.)
    """
    variant_id: str
    parameters: Dict[str, float]
    temperatures: np.ndarray
    configurations: np.ndarray
    magnetizations: np.ndarray
    energies: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatentRepresentation:
    """Latent space representation from VAE analysis.
    
    Attributes:
        latent_means: Mean values in latent space
        latent_stds: Standard deviations in latent space
        order_parameter_dim: Dimension identified as order parameter
        reconstruction_quality: Quality metrics for reconstruction
    """
    latent_means: np.ndarray
    latent_stds: np.ndarray
    order_parameter_dim: int
    reconstruction_quality: Dict[str, float] = field(default_factory=dict)


@dataclass
class VAEAnalysisResults:
    """Results from Prometheus VAE analysis pipeline.
    
    Attributes:
        variant_id: ID of the model variant analyzed
        parameters: Parameter values for this analysis
        critical_temperature: Detected critical temperature
        tc_confidence: Confidence in Tc detection
        exponents: Extracted critical exponents (e.g., {'beta': 0.35, 'nu': 0.65})
        exponent_errors: Error estimates for exponents
        r_squared_values: R² values for power law fits
        latent_representation: Latent space representation
        order_parameter_dim: Dimension identified as order parameter
    """
    variant_id: str
    parameters: Dict[str, float]
    critical_temperature: float
    tc_confidence: float
    exponents: Dict[str, float]
    exponent_errors: Dict[str, float]
    r_squared_values: Dict[str, float]
    latent_representation: LatentRepresentation
    order_parameter_dim: int


@dataclass
class NovelPhenomenon:
    """Detected novel phase transition phenomenon.
    
    Attributes:
        phenomenon_type: Type of phenomenon ('anomalous_exponents', 'first_order', etc.)
        variant_id: ID of the model variant where phenomenon was detected
        parameters: Parameter values where phenomenon occurs
        description: Human-readable description
        confidence: Confidence in detection (0.0 to 1.0)
        supporting_evidence: Supporting data and metrics
    """
    phenomenon_type: str
    variant_id: str
    parameters: Dict[str, float]
    description: str
    confidence: float
    supporting_evidence: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate phenomenon parameters."""
        valid_types = [
            'anomalous_exponents',
            'first_order',
            're_entrant',
            'multi_critical',
            'unknown'
        ]
        if self.phenomenon_type not in valid_types:
            raise ValueError(f"Invalid phenomenon type: {self.phenomenon_type}")
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


@dataclass
class ExplorationStrategy:
    """Strategy for parameter space exploration.
    
    Attributes:
        method: Exploration method ('grid', 'adaptive', 'bayesian', 'random')
        n_points: Number of parameter points to sample
        refinement_iterations: Number of adaptive refinement iterations
        focus_regions: Specific regions to explore more densely
    """
    method: str
    n_points: int
    refinement_iterations: int = 0
    focus_regions: Optional[List[Dict[str, Any]]] = None
    
    def __post_init__(self):
        """Validate exploration strategy."""
        valid_methods = ['grid', 'adaptive', 'bayesian', 'random']
        if self.method not in valid_methods:
            raise ValueError(f"Invalid exploration method: {self.method}")
        
        if self.n_points <= 0:
            raise ValueError(f"n_points must be positive, got {self.n_points}")


@dataclass
class DiscoveryResults:
    """Complete discovery pipeline results.
    
    Attributes:
        variant_id: ID of the model variant explored
        exploration_strategy: Strategy used for exploration
        n_points_explored: Number of parameter points explored
        vae_results: List of VAE analysis results for each point
        novel_phenomena: List of detected novel phenomena
        phase_diagrams: Generated phase diagram figures
        execution_time: Total execution time in seconds
        checkpoint_path: Path to checkpoint file
    """
    variant_id: str
    exploration_strategy: ExplorationStrategy
    n_points_explored: int
    vae_results: List[VAEAnalysisResults]
    novel_phenomena: List[NovelPhenomenon]
    phase_diagrams: List[Any] = field(default_factory=list)  # matplotlib figures
    execution_time: float = 0.0
    checkpoint_path: str = ""


@dataclass
class ValidationResult:
    """Result of hypothesis validation.
    
    Attributes:
        hypothesis_id: ID of the hypothesis validated
        validated: Whether hypothesis was validated
        confidence: Confidence in validation result (0.0 to 1.0)
        p_values: P-values for statistical tests
        effect_sizes: Effect sizes for comparisons
        bootstrap_intervals: Bootstrap confidence intervals
        message: Human-readable validation message
    """
    hypothesis_id: str
    validated: bool
    confidence: float
    p_values: Dict[str, float]
    effect_sizes: Dict[str, float]
    bootstrap_intervals: Dict[str, Tuple[float, float]]
    message: str
    
    def __post_init__(self):
        """Validate result parameters."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
