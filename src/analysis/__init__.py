"""
Analysis module for the Prometheus project.

This module provides tools for analyzing trained VAE models and discovering
physical insights from latent space representations.
"""

from .latent_analysis import LatentAnalyzer, LatentRepresentation
from .order_parameter_discovery import OrderParameterAnalyzer, CorrelationResult, OrderParameterCandidate

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