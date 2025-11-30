"""
Error Recovery Manager for Discovery Pipeline.

This module provides comprehensive error handling and recovery strategies for
the discovery pipeline, including simulation errors, VAE training failures,
and analysis errors.
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np

from ..utils.logging_utils import get_logger


class ErrorCategory(Enum):
    """Categories of errors that can occur in the pipeline."""
    SIMULATION_ERROR = "simulation"
    VAE_TRAINING_ERROR = "vae_training"
    ANALYSIS_ERROR = "analysis"
    DATA_ERROR = "data"
    RESOURCE_ERROR = "resource"
    UNKNOWN_ERROR = "unknown"


class RecoveryAction(Enum):
    """Actions to take when recovering from errors."""
    RETRY = "retry"
    SKIP = "skip"
    ADJUST_PARAMETERS = "adjust_parameters"
    REDUCE_RESOURCES = "reduce_resources"
    FAIL = "fail"


@dataclass
class ErrorContext:
    """Context information about an error.
    
    Attributes:
        category: Category of the error
        error: The original exception
        parameters: Parameters that caused the error
        attempt_number: Number of attempts made so far
        metadata: Additional context information
    """
    category: ErrorCategory
    error: Exception
    parameters: Dict[str, Any]
    attempt_number: int = 1
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RecoveryStrategy:
    """Strategy for recovering from an error.
    
    Attributes:
        action: Action to take
        max_retries: Maximum number of retry attempts
        parameter_adjustments: Parameter adjustments to apply
        message: Human-readable explanation
    """
    action: RecoveryAction
    max_retries: int = 3
    parameter_adjustments: Optional[Dict[str, Any]] = None
    message: str = ""


class ErrorRecoveryManager:
    """Manage error recovery in discovery pipeline.
    
    This class provides intelligent error handling and recovery strategies
    for various types of errors that can occur during exploration.
    
    Attributes:
        logger: Logger instance
        max_retries: Default maximum retry attempts
        error_history: History of errors encountered
    """
    
    def __init__(self, max_retries: int = 3):
        """Initialize error recovery manager.
        
        Args:
            max_retries: Default maximum number of retry attempts
        """
        self.logger = get_logger(__name__)
        self.max_retries = max_retries
        self.error_history: list[ErrorContext] = []
    
    def handle_simulation_error(
        self,
        error: Exception,
        params: Dict[str, Any]
    ) -> RecoveryStrategy:
        """Determine recovery action for simulation errors.
        
        Common simulation errors:
        - Non-equilibrated systems
        - Numerical instabilities
        - Invalid configurations
        - Temperature out of range
        
        Args:
            error: The exception that occurred
            params: Simulation parameters
            
        Returns:
            RecoveryStrategy with recommended action
        """
        error_msg = str(error).lower()
        
        # Categorize the error
        if "equilibrat" in error_msg or "convergence" in error_msg:
            # Non-equilibrated system - increase equilibration steps
            return RecoveryStrategy(
                action=RecoveryAction.ADJUST_PARAMETERS,
                max_retries=2,
                parameter_adjustments={
                    'n_equilibration': params.get('n_equilibration', 1000) * 2,
                    'n_steps_between': params.get('n_steps_between', 10) * 2
                },
                message="Increasing equilibration steps to improve convergence"
            )
        
        elif "numerical" in error_msg or "overflow" in error_msg or "nan" in error_msg:
            # Numerical instability - adjust temperature step size
            return RecoveryStrategy(
                action=RecoveryAction.ADJUST_PARAMETERS,
                max_retries=2,
                parameter_adjustments={
                    'n_temperatures': max(10, params.get('n_temperatures', 20) // 2),
                    't_min': params.get('t_min', 2.0) * 1.1,
                    't_max': params.get('t_max', 5.0) * 0.9
                },
                message="Reducing temperature range to avoid numerical instabilities"
            )
        
        elif "invalid" in error_msg or "configuration" in error_msg:
            # Invalid configuration - validate and skip
            self.logger.warning(f"Invalid configuration detected: {error}")
            return RecoveryStrategy(
                action=RecoveryAction.SKIP,
                message="Skipping parameter point due to invalid configuration"
            )
        
        elif "memory" in error_msg or "resource" in error_msg:
            # Resource error - reduce batch size
            return RecoveryStrategy(
                action=RecoveryAction.REDUCE_RESOURCES,
                max_retries=2,
                parameter_adjustments={
                    'lattice_size': max(16, params.get('lattice_size', 32) // 2),
                    'n_samples': max(50, params.get('n_samples', 100) // 2)
                },
                message="Reducing resource usage due to memory constraints"
            )
        
        else:
            # Unknown error - retry with same parameters
            return RecoveryStrategy(
                action=RecoveryAction.RETRY,
                max_retries=self.max_retries,
                message=f"Retrying simulation after error: {error}"
            )
    
    def handle_vae_error(
        self,
        error: Exception,
        data_params: Dict[str, Any]
    ) -> RecoveryStrategy:
        """Determine recovery action for VAE training errors.
        
        Common VAE errors:
        - Poor convergence
        - Latent collapse
        - GPU memory errors
        - NaN in loss
        
        Args:
            error: The exception that occurred
            data_params: VAE training parameters
            
        Returns:
            RecoveryStrategy with recommended action
        """
        error_msg = str(error).lower()
        
        if "convergence" in error_msg or "loss" in error_msg:
            # Poor convergence - adjust learning rate and epochs
            return RecoveryStrategy(
                action=RecoveryAction.ADJUST_PARAMETERS,
                max_retries=2,
                parameter_adjustments={
                    'learning_rate': data_params.get('learning_rate', 1e-3) * 0.5,
                    'n_epochs': min(200, data_params.get('n_epochs', 100) * 2),
                    'patience': data_params.get('patience', 10) + 5
                },
                message="Adjusting learning rate and epochs for better convergence"
            )
        
        elif "collapse" in error_msg or "latent" in error_msg:
            # Latent collapse - increase beta parameter
            return RecoveryStrategy(
                action=RecoveryAction.ADJUST_PARAMETERS,
                max_retries=2,
                parameter_adjustments={
                    'beta': min(1.0, data_params.get('beta', 0.1) * 2),
                    'latent_dim': max(2, data_params.get('latent_dim', 10) - 2)
                },
                message="Increasing beta parameter to prevent latent collapse"
            )
        
        elif "memory" in error_msg or "cuda" in error_msg or "gpu" in error_msg:
            # GPU memory error - reduce batch size
            return RecoveryStrategy(
                action=RecoveryAction.REDUCE_RESOURCES,
                max_retries=2,
                parameter_adjustments={
                    'batch_size': max(16, data_params.get('batch_size', 64) // 2),
                    'use_gpu': False  # Fall back to CPU if needed
                },
                message="Reducing batch size or falling back to CPU due to GPU memory constraints"
            )
        
        elif "nan" in error_msg or "inf" in error_msg:
            # NaN in loss - reduce learning rate and add gradient clipping
            return RecoveryStrategy(
                action=RecoveryAction.ADJUST_PARAMETERS,
                max_retries=2,
                parameter_adjustments={
                    'learning_rate': data_params.get('learning_rate', 1e-3) * 0.1,
                    'gradient_clip': 1.0,
                    'weight_decay': data_params.get('weight_decay', 0.0) + 1e-5
                },
                message="Reducing learning rate and adding gradient clipping to handle NaN"
            )
        
        else:
            # Unknown error - retry once
            return RecoveryStrategy(
                action=RecoveryAction.RETRY,
                max_retries=1,
                message=f"Retrying VAE training after error: {error}"
            )
    
    def handle_analysis_error(
        self,
        error: Exception,
        results: Dict[str, Any]
    ) -> RecoveryStrategy:
        """Determine recovery action for analysis errors.
        
        Common analysis errors:
        - Insufficient data points
        - Poor fit quality (R² < 0.7)
        - Exponent out of physical range
        - Critical temperature detection failure
        
        Args:
            error: The exception that occurred
            results: Analysis results or parameters
            
        Returns:
            RecoveryStrategy with recommended action
        """
        error_msg = str(error).lower()
        
        if "insufficient" in error_msg or "data" in error_msg:
            # Insufficient data points - increase temperature sampling
            return RecoveryStrategy(
                action=RecoveryAction.ADJUST_PARAMETERS,
                max_retries=1,
                parameter_adjustments={
                    'n_temperatures': results.get('n_temperatures', 20) * 2,
                    'n_samples': results.get('n_samples', 100) * 2
                },
                message="Increasing temperature sampling for better analysis"
            )
        
        elif "fit" in error_msg or "r2" in error_msg or "r²" in error_msg:
            # Poor fit quality - flag for manual review but continue
            r_squared = results.get('r_squared', 0.0)
            if r_squared < 0.5:
                return RecoveryStrategy(
                    action=RecoveryAction.SKIP,
                    message=f"Skipping point due to poor fit quality (R²={r_squared:.3f})"
                )
            else:
                return RecoveryStrategy(
                    action=RecoveryAction.RETRY,
                    max_retries=1,
                    message=f"Flagging point for review (R²={r_squared:.3f})"
                )
        
        elif "exponent" in error_msg or "range" in error_msg:
            # Exponent out of physical range - mark as anomalous
            return RecoveryStrategy(
                action=RecoveryAction.SKIP,
                message="Marking point as anomalous due to unphysical exponent values"
            )
        
        elif "critical" in error_msg or "tc" in error_msg:
            # Critical temperature detection failure - widen search range
            return RecoveryStrategy(
                action=RecoveryAction.ADJUST_PARAMETERS,
                max_retries=1,
                parameter_adjustments={
                    't_min': results.get('t_min', 2.0) * 0.8,
                    't_max': results.get('t_max', 5.0) * 1.2,
                    'n_temperatures': results.get('n_temperatures', 20) + 10
                },
                message="Widening temperature range for Tc detection"
            )
        
        else:
            # Unknown error - skip this point
            return RecoveryStrategy(
                action=RecoveryAction.SKIP,
                message=f"Skipping point after analysis error: {error}"
            )
    
    def execute_recovery(
        self,
        context: ErrorContext,
        recovery_fn: Callable[[Dict[str, Any]], Any]
    ) -> Optional[Any]:
        """Execute recovery strategy with retry logic.
        
        Args:
            context: Error context information
            recovery_fn: Function to retry with adjusted parameters
            
        Returns:
            Result from recovery function, or None if recovery failed
        """
        # Record error in history
        self.error_history.append(context)
        
        # Determine recovery strategy based on error category
        if context.category == ErrorCategory.SIMULATION_ERROR:
            strategy = self.handle_simulation_error(context.error, context.parameters)
        elif context.category == ErrorCategory.VAE_TRAINING_ERROR:
            strategy = self.handle_vae_error(context.error, context.parameters)
        elif context.category == ErrorCategory.ANALYSIS_ERROR:
            strategy = self.handle_analysis_error(context.error, context.parameters)
        else:
            # Unknown category - default to skip
            strategy = RecoveryStrategy(
                action=RecoveryAction.SKIP,
                message=f"Skipping due to unknown error category: {context.category}"
            )
        
        # Log recovery strategy
        self.logger.info(f"Recovery strategy: {strategy.action.value}")
        self.logger.info(f"Message: {strategy.message}")
        
        # Execute recovery action
        if strategy.action == RecoveryAction.SKIP:
            self.logger.warning("Skipping parameter point")
            return None
        
        elif strategy.action == RecoveryAction.FAIL:
            self.logger.error("Recovery failed, propagating error")
            raise context.error
        
        elif strategy.action in [RecoveryAction.RETRY, RecoveryAction.ADJUST_PARAMETERS, RecoveryAction.REDUCE_RESOURCES]:
            # Retry with adjusted parameters
            adjusted_params = context.parameters.copy()
            if strategy.parameter_adjustments:
                adjusted_params.update(strategy.parameter_adjustments)
                self.logger.info(f"Adjusted parameters: {strategy.parameter_adjustments}")
            
            # Retry loop
            for attempt in range(strategy.max_retries):
                try:
                    self.logger.info(f"Retry attempt {attempt + 1}/{strategy.max_retries}")
                    result = recovery_fn(adjusted_params)
                    self.logger.info("Recovery successful")
                    return result
                
                except Exception as e:
                    self.logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
                    if attempt == strategy.max_retries - 1:
                        self.logger.error("All retry attempts exhausted")
                        return None
            
            return None
        
        else:
            self.logger.error(f"Unknown recovery action: {strategy.action}")
            return None
    
    def categorize_error(self, error: Exception, stage: str) -> ErrorCategory:
        """Categorize an error based on its type and stage.
        
        Args:
            error: The exception to categorize
            stage: Pipeline stage where error occurred
            
        Returns:
            ErrorCategory for the error
        """
        error_msg = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Check stage first
        if stage == "simulation":
            return ErrorCategory.SIMULATION_ERROR
        elif stage == "vae_training":
            return ErrorCategory.VAE_TRAINING_ERROR
        elif stage == "analysis":
            return ErrorCategory.ANALYSIS_ERROR
        
        # Check error message/type
        if "memory" in error_msg or "resource" in error_msg:
            return ErrorCategory.RESOURCE_ERROR
        elif "data" in error_msg or "shape" in error_msg or "dimension" in error_msg:
            return ErrorCategory.DATA_ERROR
        elif "valueerror" in error_type or "typeerror" in error_type:
            return ErrorCategory.DATA_ERROR
        else:
            return ErrorCategory.UNKNOWN_ERROR
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about errors encountered.
        
        Returns:
            Dictionary with error statistics
        """
        if not self.error_history:
            return {
                'total_errors': 0,
                'by_category': {},
                'most_common': None
            }
        
        # Count by category
        category_counts = {}
        for context in self.error_history:
            cat = context.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Find most common
        most_common = max(category_counts.items(), key=lambda x: x[1])
        
        return {
            'total_errors': len(self.error_history),
            'by_category': category_counts,
            'most_common': {
                'category': most_common[0],
                'count': most_common[1]
            }
        }
    
    def clear_history(self) -> None:
        """Clear error history."""
        self.error_history.clear()
        self.logger.info("Error history cleared")
