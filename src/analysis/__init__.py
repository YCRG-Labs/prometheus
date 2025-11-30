"""
Analysis module for the Prometheus project.

This module provides tools for analyzing trained VAE models and discovering
physical insights from latent space representations.
"""

from .latent_analysis import LatentAnalyzer, LatentRepresentation
from .order_parameter_discovery import OrderParameterAnalyzer, CorrelationResult, OrderParameterCandidate
from .data_quality_3d import (
    DataQualityAnalyzer3D,
    DataQualityReport3D,
    MagnetizationAnalysisResult,
    analyze_3d_dataset_quality
)
from .critical_exponent_analyzer import (
    CriticalExponentAnalyzer,
    PowerLawFitter,
    PowerLawFitResult,
    CriticalExponentResults,
    create_critical_exponent_analyzer
)

# Publication materials generation
from .systematic_comparison_framework import (
    SystematicComparisonFramework,
    SystemComparisonData,
    ComparisonResults
)
from .critical_exponent_comparison_tables import (
    CriticalExponentComparisonTables,
    CriticalExponentData,
    ExponentComparisonSummary
)
from .publication_figure_generator import (
    PublicationFigureGenerator,
    PublicationFigureSpec,
    PublicationPackage
)

# Quantum phase transition detection
from .quantum_phase_detection import (
    QuantumPhaseDetector,
    CriticalExponentExtractor,
    QCPDetectionResult,
    CriticalExponents
)

# Enhanced validation types and interfaces
from .enhanced_validation_types import (
    # Exception classes
    PhysicsValidationError,
    TheoreticalModelError,
    CriticalExponentError,
    SymmetryValidationError,
    ExperimentalComparisonError,
    StatisticalValidationError,
    FiniteSizeScalingError,
    
    # Enums
    ViolationSeverity,
    UniversalityClass,
    ValidationLevel,
    
    # Enhanced validation result data classes
    CriticalExponentValidation,
    FiniteSizeScalingResult,
    SymmetryValidationResult,
    PhysicsViolation,
    UniversalityClassResult,
    TheoreticalPredictions,
    IsingModelValidation,
    XYModelValidation,
    HeisenbergModelValidation,
    ConfidenceInterval,
    EnsembleAnalysisResult,
    HypothesisTestResults,
    ExperimentalComparison,
    MetaAnalysisResult,
    
    # Enhanced report data structures
    PhysicsReviewSummary,
    TheoreticalConsistencySection,
    OrderParameterValidationSection,
    CriticalBehaviorSection,
    StatisticalValidationSection,
    ExperimentalComparisonSection,
    ViolationSummary,
    PhysicsReviewReport,
    
    # Interface protocols
    EnhancedPhysicsValidatorProtocol,
    TheoreticalModelValidatorProtocol,
    StatisticalPhysicsAnalyzerProtocol,
    PhysicsReviewReportGeneratorProtocol,
    ExperimentalBenchmarkComparatorProtocol,
    
    # Abstract base classes
    BaseEnhancedValidator,
    BaseTheoreticalModel,
)

__all__ = [
    'LatentAnalyzer',
    'LatentRepresentation', 
    'OrderParameterAnalyzer',
    'CorrelationResult',
    'OrderParameterCandidate',
    
    # 3D Data quality analysis
    'DataQualityAnalyzer3D',
    'DataQualityReport3D',
    'MagnetizationAnalysisResult',
    'analyze_3d_dataset_quality',
    
    # Critical exponent analysis
    'CriticalExponentAnalyzer',
    'PowerLawFitter',
    'PowerLawFitResult',
    'CriticalExponentResults',
    'create_critical_exponent_analyzer',
    
    # Quantum phase transition detection
    'QuantumPhaseDetector',
    'CriticalExponentExtractor',
    'QCPDetectionResult',
    'CriticalExponents',
    
    # Publication materials generation
    'SystematicComparisonFramework',
    'SystemComparisonData',
    'ComparisonResults',
    'CriticalExponentComparisonTables',
    'CriticalExponentData',
    'ExponentComparisonSummary',
    'PublicationFigureGenerator',
    'PublicationFigureSpec',
    'PublicationPackage',
    
    # Enhanced validation exceptions
    'PhysicsValidationError',
    'TheoreticalModelError',
    'CriticalExponentError',
    'SymmetryValidationError',
    'ExperimentalComparisonError',
    'StatisticalValidationError',
    'FiniteSizeScalingError',
    
    # Enhanced validation enums
    'ViolationSeverity',
    'UniversalityClass',
    'ValidationLevel',
    
    # Enhanced validation data classes
    'CriticalExponentValidation',
    'FiniteSizeScalingResult',
    'SymmetryValidationResult',
    'PhysicsViolation',
    'UniversalityClassResult',
    'TheoreticalPredictions',
    'IsingModelValidation',
    'XYModelValidation',
    'HeisenbergModelValidation',
    'ConfidenceInterval',
    'EnsembleAnalysisResult',
    'HypothesisTestResults',
    'ExperimentalComparison',
    'MetaAnalysisResult',
    
    # Enhanced report structures
    'PhysicsReviewSummary',
    'TheoreticalConsistencySection',
    'OrderParameterValidationSection',
    'CriticalBehaviorSection',
    'StatisticalValidationSection',
    'ExperimentalComparisonSection',
    'ViolationSummary',
    'PhysicsReviewReport',
    
    # Interface protocols
    'EnhancedPhysicsValidatorProtocol',
    'TheoreticalModelValidatorProtocol',
    'StatisticalPhysicsAnalyzerProtocol',
    'PhysicsReviewReportGeneratorProtocol',
    'ExperimentalBenchmarkComparatorProtocol',
    
    # Abstract base classes
    'BaseEnhancedValidator',
    'BaseTheoreticalModel',
]