"""
Optimization module for hyperparameter tuning and architecture search.

This module provides tools for systematic exploration of VAE architectures
and training configurations to optimize physics consistency and performance.
"""

from .hyperparameter_optimizer import HyperparameterOptimizer
from .architecture_search import ArchitectureSearchSpace, ArchitectureOptimizer
from .physics_metrics import PhysicsConsistencyEvaluator

__all__ = [
    'HyperparameterOptimizer',
    'ArchitectureSearchSpace', 
    'ArchitectureOptimizer',
    'PhysicsConsistencyEvaluator'
]