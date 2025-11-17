"""
Validation Module

This module provides comprehensive validation and quality assurance
for the Prometheus critical exponent extraction system.

Task 11 Implementation:
- Statistical validation framework (task 11.1)
- Final validation and quality assurance system (task 11.2)
- Comprehensive validation integration
"""

# Temporarily commented out due to import issues
# from .accuracy_validation_pipeline import (
#     AccuracyValidationPipeline,
#     ValidationMetrics,
#     ModelQualityMetrics,
#     SystemValidationResult,
#     PipelineValidationResult,
#     create_accuracy_validation_pipeline
# )

from .statistical_validation_framework import (
    StatisticalValidationFramework,
    ErrorBarResult,
    ConfidenceIntervalResult,
    FiniteSizeScalingResult,
    QualityMetrics,
    SystemValidationMetrics,
    create_statistical_validation_framework
)

from .final_validation_system import (
    FinalValidationSystem,
    EquilibrationValidationResult,
    DataQualityResult,
    PhysicsConsistencyResult,
    FinalValidationReport,
    create_final_validation_system
)

from .comprehensive_validation_integration import (
    ComprehensiveValidationIntegration,
    ComprehensiveValidationResult,
    create_comprehensive_validation_integration
)

__all__ = [
    # Accuracy validation pipeline (existing) - temporarily commented out
    # 'AccuracyValidationPipeline',
    # 'ValidationMetrics',
    # 'ModelQualityMetrics',
    # 'SystemValidationResult',
    # 'PipelineValidationResult',
    # 'create_accuracy_validation_pipeline',
    
    # Statistical validation framework (task 11.1)
    'StatisticalValidationFramework',
    'ErrorBarResult',
    'ConfidenceIntervalResult',
    'FiniteSizeScalingResult',
    'QualityMetrics',
    'SystemValidationMetrics',
    'create_statistical_validation_framework',
    
    # Final validation system (task 11.2)
    'FinalValidationSystem',
    'EquilibrationValidationResult',
    'DataQualityResult',
    'PhysicsConsistencyResult',
    'FinalValidationReport',
    'create_final_validation_system',
    
    # Comprehensive validation integration
    'ComprehensiveValidationIntegration',
    'ComprehensiveValidationResult',
    'create_comprehensive_validation_integration'
]