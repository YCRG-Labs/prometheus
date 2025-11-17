"""
Optimization module for hyperparameter tuning and architecture search.

This module provides tools for systematic exploration of VAE architectures
and training configurations to optimize physics consistency and performance.
"""

# Import existing modules (may have dependency issues)
try:
    from .hyperparameter_optimizer import HyperparameterOptimizer
    from .architecture_search import ArchitectureSearchSpace, ArchitectureOptimizer
    from .physics_metrics import PhysicsConsistencyEvaluator
    _LEGACY_IMPORTS_AVAILABLE = True
except ImportError:
    _LEGACY_IMPORTS_AVAILABLE = False

# Import new performance optimizer (standalone, no dependencies)
from .performance_optimizer import (
    PerformanceProfiler,
    ResultCache,
    ProgressTracker,
    get_profiler,
    get_cache,
    profile,
    cached,
    create_progress_tracker
)

__all__ = [
    'PerformanceProfiler',
    'ResultCache',
    'ProgressTracker',
    'get_profiler',
    'get_cache',
    'profile',
    'cached',
    'create_progress_tracker'
]

if _LEGACY_IMPORTS_AVAILABLE:
    __all__.extend([
        'HyperparameterOptimizer',
        'ArchitectureSearchSpace', 
        'ArchitectureOptimizer',
        'PhysicsConsistencyEvaluator'
    ])