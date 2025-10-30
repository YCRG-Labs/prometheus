"""
Enhanced Physics Validation Data Structures and Interfaces

This module provides enhanced data classes, interfaces, and exception classes
for comprehensive physics validation in the Prometheus phase discovery system.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod
from matplotlib.figure import Figure
from enum import Enum

from .latent_analysis import LatentRepresentation
from .phase_detection import PhaseDetectionResult
from .order_parameter_discovery import OrderParameterCandidate


# ============================================================================
# Exception Classes
# ============================================================================

class PhysicsValidationError(Exception):
    """Base exception for physics validation errors."""
    pass


class TheoreticalModelError(PhysicsValidationError):
    """Error in theoretical model validation."""
    pass


class CriticalExponentError(PhysicsValidationError):
    """Error in critical exponent computation."""
    pass


class SymmetryValidationError(PhysicsValidationError):
    """Error in symmetry analysis."""
    pass


class ExperimentalComparisonError(PhysicsValidationError):
    """Error in experimental benchmark comparison."""
    pass


class StatisticalValidationError(PhysicsValidationError):
    """Error in statistical validation procedures."""
    pass


class FiniteSizeScalingError(PhysicsValidationError):
    """Error in finite-size scaling analysis."""
    pass


# ============================================================================
# Enums and Constants
# ============================================================================

class ViolationSeverity(Enum):
    """Severity levels for physics violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class UniversalityClass(Enum):
    """Known universality classes for phase transitions."""
    ISING_2D = "ising_2d"
    ISING_3D = "ising_3d"
    XY_2D = "xy_2d"
    XY_3D = "xy_3d"
    HEISENBERG_3D = "heisenberg_3d"
    POTTS_3_2D = "potts_3_2d"
    UNKNOWN = "unknown"


class ValidationLevel(Enum):
    """Validation comprehensiveness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


# ============================================================================
# Enhanced Validation Result Data Classes
# ============================================================================

@dataclass
class CriticalExponentValidation:
    """Results from critical exponent analysis and validation."""
    beta_exponent: float
    beta_theoretical: float
    beta_confidence_interval: Tuple[float, float]
    beta_deviation: float
    
    gamma_exponent: Optional[float] = None
    gamma_theoretical: Optional[float] = None
    gamma_confidence_interval: Optional[Tuple[float, float]] = None
    gamma_deviation: Optional[float] = None
    
    nu_exponent: Optional[float] = None
    nu_theoretical: Optional[float] = None
    nu_confidence_interval: Optional[Tuple[float, float]] = None
    nu_deviation: Optional[float] = None
    
    alpha_exponent: Optional[float] = None
    alpha_theoretical: Optional[float] = None
    alpha_confidence_interval: Optional[Tuple[float, float]] = None
    alpha_deviation: Optional[float] = None
    
    universality_class_match: bool = False
    identified_universality_class: UniversalityClass = UniversalityClass.UNKNOWN
    scaling_violations: List[str] = None
    power_law_fit_quality: Dict[str, float] = None
    
    def __post_init__(self):
        if self.scaling_violations is None:
            self.scaling_violations = []
        if self.power_law_fit_quality is None:
            self.power_law_fit_quality = {}


@dataclass
class FiniteSizeScalingResult:
    """Results from finite-size scaling analysis."""
    system_sizes: List[int]
    scaling_collapse_quality: float
    scaling_function_parameters: Dict[str, float]
    correlation_length_exponent: float
    correlation_length_confidence_interval: Tuple[float, float]
    
    scaling_plots_data: Dict[str, np.ndarray] = None
    finite_size_corrections: Dict[str, float] = None
    scaling_violations: List[str] = None
    
    def __post_init__(self):
        if self.scaling_plots_data is None:
            self.scaling_plots_data = {}
        if self.finite_size_corrections is None:
            self.finite_size_corrections = {}
        if self.scaling_violations is None:
            self.scaling_violations = []


@dataclass
class SymmetryValidationResult:
    """Results from symmetry analysis of order parameters."""
    broken_symmetries: List[str]
    symmetry_order: Dict[str, int]
    order_parameter_symmetry: str
    symmetry_consistency_score: float
    
    symmetry_breaking_temperature: Optional[float] = None
    continuous_symmetry_analysis: Optional[Dict[str, Any]] = None
    discrete_symmetry_analysis: Optional[Dict[str, Any]] = None
    violations: List[str] = None
    
    def __post_init__(self):
        if self.violations is None:
            self.violations = []


@dataclass
class PhysicsViolation:
    """Detailed information about a physics validation violation."""
    violation_type: str
    severity: ViolationSeverity
    description: str
    suggested_investigation: str
    physics_explanation: str
    
    literature_references: List[str] = None
    quantitative_measure: Optional[float] = None
    threshold_value: Optional[float] = None
    confidence_level: Optional[float] = None
    
    def __post_init__(self):
        if self.literature_references is None:
            self.literature_references = []


@dataclass
class UniversalityClassResult:
    """Results from universality class identification."""
    identified_class: UniversalityClass
    confidence_score: float
    critical_exponents_match: Dict[str, bool]
    
    alternative_classes: List[Tuple[UniversalityClass, float]] = None
    dimensionality_analysis: Optional[Dict[str, Any]] = None
    symmetry_analysis: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.alternative_classes is None:
            self.alternative_classes = []


@dataclass
class TheoreticalPredictions:
    """Theoretical predictions for comparison with computational results."""
    model_type: str
    critical_temperature: float
    critical_exponents: Dict[str, float]
    order_parameter_behavior: Dict[str, Any]
    
    correlation_functions: Optional[Dict[str, Callable]] = None
    susceptibility_predictions: Optional[Dict[str, float]] = None
    specific_heat_predictions: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.correlation_functions is None:
            self.correlation_functions = {}


@dataclass
class IsingModelValidation:
    """Validation results against Ising model predictions."""
    theoretical_predictions: TheoreticalPredictions
    computational_results: Dict[str, float]
    agreement_metrics: Dict[str, float]
    
    onsager_solution_comparison: Optional[Dict[str, Any]] = None
    mean_field_comparison: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.onsager_solution_comparison is None:
            self.onsager_solution_comparison = {}
        if self.mean_field_comparison is None:
            self.mean_field_comparison = {}


@dataclass
class XYModelValidation:
    """Validation results against XY model predictions."""
    theoretical_predictions: TheoreticalPredictions
    computational_results: Dict[str, float]
    agreement_metrics: Dict[str, float]
    
    kosterlitz_thouless_analysis: Optional[Dict[str, Any]] = None
    vortex_analysis: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.kosterlitz_thouless_analysis is None:
            self.kosterlitz_thouless_analysis = {}
        if self.vortex_analysis is None:
            self.vortex_analysis = {}


@dataclass
class HeisenbergModelValidation:
    """Validation results against Heisenberg model predictions."""
    theoretical_predictions: TheoreticalPredictions
    computational_results: Dict[str, float]
    agreement_metrics: Dict[str, float]
    
    spin_wave_analysis: Optional[Dict[str, Any]] = None
    magnon_analysis: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.spin_wave_analysis is None:
            self.spin_wave_analysis = {}
        if self.magnon_analysis is None:
            self.magnon_analysis = {}


@dataclass
class ConfidenceInterval:
    """Statistical confidence interval with metadata."""
    lower_bound: float
    upper_bound: float
    confidence_level: float
    method: str
    
    n_bootstrap_samples: Optional[int] = None
    bias_correction: Optional[float] = None
    
    @property
    def width(self) -> float:
        """Width of the confidence interval."""
        return self.upper_bound - self.lower_bound
    
    @property
    def center(self) -> float:
        """Center point of the confidence interval."""
        return (self.lower_bound + self.upper_bound) / 2.0


@dataclass
class EnsembleAnalysisResult:
    """Results from ensemble analysis across multiple simulation runs."""
    n_ensemble_members: int
    ensemble_mean: Dict[str, float]
    ensemble_std: Dict[str, float]
    ensemble_confidence_intervals: Dict[str, ConfidenceInterval]
    
    inter_run_correlations: Optional[Dict[str, float]] = None
    convergence_analysis: Optional[Dict[str, Any]] = None
    outlier_detection: Optional[Dict[str, List[int]]] = None
    
    def __post_init__(self):
        if self.inter_run_correlations is None:
            self.inter_run_correlations = {}
        if self.convergence_analysis is None:
            self.convergence_analysis = {}
        if self.outlier_detection is None:
            self.outlier_detection = {}


@dataclass
class HypothesisTestResults:
    """Results from statistical hypothesis testing."""
    test_name: str
    null_hypothesis: str
    alternative_hypothesis: str
    test_statistic: float
    p_value: float
    critical_value: float
    
    reject_null: bool
    confidence_level: float = 0.05
    effect_size: Optional[float] = None
    power_analysis: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.power_analysis is None:
            self.power_analysis = {}


@dataclass
class ExperimentalComparison:
    """Comparison between computational and experimental results."""
    experimental_dataset: str
    computational_value: float
    experimental_value: float
    experimental_uncertainty: float
    
    agreement_metric: float
    statistical_significance: float
    z_score: float
    
    discrepancy_explanation: Optional[str] = None
    systematic_errors: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.systematic_errors is None:
            self.systematic_errors = []


@dataclass
class MetaAnalysisResult:
    """Results from meta-analysis across multiple experimental datasets."""
    n_studies: int
    pooled_estimate: float
    pooled_uncertainty: float
    heterogeneity_statistic: float
    
    individual_studies: List[ExperimentalComparison] = None
    forest_plot_data: Optional[Dict[str, Any]] = None
    publication_bias_analysis: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.individual_studies is None:
            self.individual_studies = []
        if self.forest_plot_data is None:
            self.forest_plot_data = {}
        if self.publication_bias_analysis is None:
            self.publication_bias_analysis = {}


# ============================================================================
# Enhanced Report Data Structures
# ============================================================================

@dataclass
class PhysicsReviewSummary:
    """Summary section of physics review report."""
    overall_assessment: str
    physics_consistency_score: float
    validation_level: ValidationLevel
    
    key_findings: List[str] = None
    major_violations: List[PhysicsViolation] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.key_findings is None:
            self.key_findings = []
        if self.major_violations is None:
            self.major_violations = []
        if self.recommendations is None:
            self.recommendations = []


@dataclass
class TheoreticalConsistencySection:
    """Theoretical consistency section of physics review report."""
    critical_exponent_validation: CriticalExponentValidation
    universality_class_result: UniversalityClassResult
    finite_size_scaling_result: Optional[FiniteSizeScalingResult] = None
    
    model_comparisons: Dict[str, Any] = None
    theoretical_violations: List[PhysicsViolation] = None
    
    def __post_init__(self):
        if self.model_comparisons is None:
            self.model_comparisons = {}
        if self.theoretical_violations is None:
            self.theoretical_violations = []


@dataclass
class OrderParameterValidationSection:
    """Order parameter validation section of physics review report."""
    symmetry_validation: SymmetryValidationResult
    correlation_analysis: Dict[str, Any]
    hierarchy_analysis: Dict[str, Any]
    
    coupling_analysis: Optional[Dict[str, Any]] = None
    order_parameter_violations: List[PhysicsViolation] = None
    
    def __post_init__(self):
        if self.coupling_analysis is None:
            self.coupling_analysis = {}
        if self.order_parameter_violations is None:
            self.order_parameter_violations = []


@dataclass
class CriticalBehaviorSection:
    """Critical behavior analysis section of physics review report."""
    phase_transition_analysis: Dict[str, Any]
    critical_temperature_validation: Dict[str, Any]
    scaling_behavior: Dict[str, Any]
    
    transition_sharpness: Optional[float] = None
    critical_behavior_violations: List[PhysicsViolation] = None
    
    def __post_init__(self):
        if self.critical_behavior_violations is None:
            self.critical_behavior_violations = []


@dataclass
class StatisticalValidationSection:
    """Statistical validation section of physics review report."""
    ensemble_analysis: Optional[EnsembleAnalysisResult] = None
    hypothesis_tests: List[HypothesisTestResults] = None
    confidence_intervals: Dict[str, ConfidenceInterval] = None
    
    uncertainty_quantification: Optional[Dict[str, Any]] = None
    statistical_violations: List[PhysicsViolation] = None
    
    def __post_init__(self):
        if self.hypothesis_tests is None:
            self.hypothesis_tests = []
        if self.confidence_intervals is None:
            self.confidence_intervals = {}
        if self.statistical_violations is None:
            self.statistical_violations = []


@dataclass
class ExperimentalComparisonSection:
    """Experimental comparison section of physics review report."""
    experimental_comparisons: List[ExperimentalComparison] = None
    meta_analysis: Optional[MetaAnalysisResult] = None
    
    agreement_summary: Optional[Dict[str, Any]] = None
    experimental_violations: List[PhysicsViolation] = None
    
    def __post_init__(self):
        if self.experimental_comparisons is None:
            self.experimental_comparisons = []
        if self.agreement_summary is None:
            self.agreement_summary = {}
        if self.experimental_violations is None:
            self.experimental_violations = []


@dataclass
class ViolationSummary:
    """Summary of all physics violations found during validation."""
    total_violations: int
    violations_by_severity: Dict[ViolationSeverity, int]
    violations_by_type: Dict[str, int]
    
    critical_violations: List[PhysicsViolation] = None
    recommended_actions: List[str] = None
    
    def __post_init__(self):
        if self.critical_violations is None:
            self.critical_violations = []
        if self.recommended_actions is None:
            self.recommended_actions = []


@dataclass
class PhysicsReviewReport:
    """Comprehensive physics review report."""
    summary: PhysicsReviewSummary
    theoretical_consistency: TheoreticalConsistencySection
    order_parameter_validation: OrderParameterValidationSection
    critical_behavior_analysis: CriticalBehaviorSection
    statistical_validation: StatisticalValidationSection
    experimental_comparison: ExperimentalComparisonSection
    
    violations: List[PhysicsViolation] = None
    violation_summary: Optional[ViolationSummary] = None
    educational_content: Dict[str, str] = None
    visualizations: Dict[str, Figure] = None
    
    overall_assessment: str = ""
    generation_timestamp: Optional[str] = None
    validation_parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.violations is None:
            self.violations = []
        if self.educational_content is None:
            self.educational_content = {}
        if self.visualizations is None:
            self.visualizations = {}
        if self.validation_parameters is None:
            self.validation_parameters = {}


# ============================================================================
# Interface Protocols
# ============================================================================

class EnhancedPhysicsValidatorProtocol(Protocol):
    """Protocol for enhanced physics validation components."""
    
    def validate_critical_exponents(
        self, 
        latent_repr: LatentRepresentation,
        phase_detection_result: PhaseDetectionResult
    ) -> CriticalExponentValidation:
        """Validate critical exponents against theoretical predictions."""
        ...
    
    def validate_finite_size_scaling(
        self,
        multi_size_data: Dict[int, LatentRepresentation]
    ) -> FiniteSizeScalingResult:
        """Validate finite-size scaling behavior."""
        ...
    
    def validate_symmetry_properties(
        self,
        order_parameters: List[OrderParameterCandidate],
        hamiltonian_symmetries: List[str]
    ) -> SymmetryValidationResult:
        """Validate symmetry properties of order parameters."""
        ...
    
    def validate_universality_class(
        self,
        critical_exponents: Dict[str, float],
        system_dimensionality: int
    ) -> UniversalityClassResult:
        """Identify and validate universality class."""
        ...


class TheoreticalModelValidatorProtocol(Protocol):
    """Protocol for theoretical model validation components."""
    
    def validate_against_ising_model(
        self,
        latent_repr: LatentRepresentation,
        system_size: int
    ) -> IsingModelValidation:
        """Validate against Ising model predictions."""
        ...
    
    def validate_against_xy_model(
        self,
        latent_repr: LatentRepresentation
    ) -> XYModelValidation:
        """Validate against XY model predictions."""
        ...
    
    def validate_against_heisenberg_model(
        self,
        latent_repr: LatentRepresentation
    ) -> HeisenbergModelValidation:
        """Validate against Heisenberg model predictions."""
        ...
    
    def compute_theoretical_predictions(
        self,
        model_type: str,
        system_parameters: Dict[str, Any]
    ) -> TheoreticalPredictions:
        """Compute theoretical predictions for given model and parameters."""
        ...


class StatisticalPhysicsAnalyzerProtocol(Protocol):
    """Protocol for statistical physics analysis components."""
    
    def compute_bootstrap_confidence_intervals(
        self,
        data: np.ndarray,
        statistic_func: Callable,
        n_bootstrap: int = 1000
    ) -> ConfidenceInterval:
        """Compute bootstrap confidence intervals."""
        ...
    
    def perform_ensemble_analysis(
        self,
        ensemble_data: List[LatentRepresentation]
    ) -> EnsembleAnalysisResult:
        """Perform ensemble analysis across multiple runs."""
        ...
    
    def test_physics_hypotheses(
        self,
        observed_values: Dict[str, float],
        theoretical_values: Dict[str, float]
    ) -> List[HypothesisTestResults]:
        """Test physics hypotheses using statistical methods."""
        ...


class PhysicsReviewReportGeneratorProtocol(Protocol):
    """Protocol for physics review report generation components."""
    
    def generate_comprehensive_report(
        self,
        validation_results: Dict[str, Any],
        include_educational_content: bool = True
    ) -> PhysicsReviewReport:
        """Generate comprehensive physics review report."""
        ...
    
    def generate_violation_summary(
        self,
        violations: List[PhysicsViolation]
    ) -> ViolationSummary:
        """Generate summary of physics violations."""
        ...
    
    def generate_educational_explanations(
        self,
        physics_concepts: List[str]
    ) -> Dict[str, str]:
        """Generate educational explanations for physics concepts."""
        ...


class ExperimentalBenchmarkComparatorProtocol(Protocol):
    """Protocol for experimental benchmark comparison components."""
    
    def compare_with_experimental_data(
        self,
        computational_results: Dict[str, float],
        experimental_dataset: str
    ) -> List[ExperimentalComparison]:
        """Compare computational results with experimental data."""
        ...
    
    def perform_meta_analysis(
        self,
        experimental_comparisons: List[ExperimentalComparison]
    ) -> MetaAnalysisResult:
        """Perform meta-analysis across multiple experimental datasets."""
        ...
    
    def load_experimental_benchmark(
        self,
        dataset_name: str
    ) -> Dict[str, Any]:
        """Load experimental benchmark dataset."""
        ...


# ============================================================================
# Abstract Base Classes
# ============================================================================

class BaseEnhancedValidator(ABC):
    """Abstract base class for enhanced validation components."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.violations: List[PhysicsViolation] = []
    
    @abstractmethod
    def validate(self, *args, **kwargs) -> Any:
        """Perform validation and return results."""
        pass
    
    def add_violation(self, violation: PhysicsViolation) -> None:
        """Add a physics violation to the list."""
        self.violations.append(violation)
    
    def clear_violations(self) -> None:
        """Clear all recorded violations."""
        self.violations.clear()
    
    def get_violations_by_severity(self, severity: ViolationSeverity) -> List[PhysicsViolation]:
        """Get violations filtered by severity level."""
        return [v for v in self.violations if v.severity == severity]


class BaseTheoreticalModel(ABC):
    """Abstract base class for theoretical model implementations."""
    
    def __init__(self, model_name: str, dimensionality: int):
        self.model_name = model_name
        self.dimensionality = dimensionality
    
    @abstractmethod
    def compute_critical_exponents(self) -> Dict[str, float]:
        """Compute theoretical critical exponents."""
        pass
    
    @abstractmethod
    def compute_critical_temperature(self, system_parameters: Dict[str, Any]) -> float:
        """Compute theoretical critical temperature."""
        pass
    
    @abstractmethod
    def compute_order_parameter_behavior(self, temperatures: np.ndarray) -> np.ndarray:
        """Compute theoretical order parameter behavior."""
        pass