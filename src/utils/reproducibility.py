"""
Reproducibility utilities for ensuring deterministic results.

This module provides utilities for setting random seeds across all libraries
used in the Prometheus system to ensure reproducible results.
"""

import random
import numpy as np
import os
from typing import Optional, Dict, Any


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for all random number generators used in the system.
    
    This function sets seeds for:
    - Python's built-in random module
    - NumPy
    - PyTorch (if available)
    - CUDA (if available)
    
    Args:
        seed: Random seed value (default: 42)
        
    Example:
        >>> from src.utils.reproducibility import set_random_seed
        >>> set_random_seed(42)
        >>> # All subsequent random operations will be deterministic
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch seeds if available
    try:
        import torch
        torch.manual_seed(seed)
        
        # Set CUDA seed if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
            # Make CUDA operations deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not installed
    
    # Set environment variable for hash seed (Python 3.3+)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_reproducibility_info() -> dict:
    """
    Get information about reproducibility settings.
    
    Returns:
        Dictionary containing:
        - python_seed: Current Python random seed state
        - numpy_seed: Current NumPy random seed state
        - torch_available: Whether PyTorch is available
        - cuda_available: Whether CUDA is available
        - cudnn_deterministic: Whether cuDNN deterministic mode is enabled
        
    Example:
        >>> info = get_reproducibility_info()
        >>> print(f"PyTorch available: {info['torch_available']}")
    """
    info = {
        'python_seed': random.getstate()[1][0],  # Get first value from state
        'numpy_seed': np.random.get_state()[1][0],  # Get first value from state
        'torch_available': False,
        'cuda_available': False,
        'cudnn_deterministic': False,
    }
    
    try:
        import torch
        info['torch_available'] = True
        info['cuda_available'] = torch.cuda.is_available()
        info['cudnn_deterministic'] = torch.backends.cudnn.deterministic
    except ImportError:
        pass
    
    return info


class ReproducibleContext:
    """
    Context manager for reproducible operations.
    
    This context manager temporarily sets a random seed for a block of code,
    then restores the previous random state.
    
    Example:
        >>> with ReproducibleContext(42):
        ...     # All random operations here are deterministic
        ...     data = np.random.randn(100)
        >>> # Previous random state is restored
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize reproducible context.
        
        Args:
            seed: Random seed to use within the context
        """
        self.seed = seed
        self.python_state = None
        self.numpy_state = None
        self.torch_state = None
        self.torch_cuda_state = None
        
    def __enter__(self):
        """Save current random states and set new seed."""
        # Save current states
        self.python_state = random.getstate()
        self.numpy_state = np.random.get_state()
        
        try:
            import torch
            self.torch_state = torch.get_rng_state()
            if torch.cuda.is_available():
                self.torch_cuda_state = torch.cuda.get_rng_state_all()
        except ImportError:
            pass
        
        # Set new seed
        set_random_seed(self.seed)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous random states."""
        # Restore states
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)
        
        try:
            import torch
            if self.torch_state is not None:
                torch.set_rng_state(self.torch_state)
            if self.torch_cuda_state is not None:
                torch.cuda.set_rng_state_all(self.torch_cuda_state)
        except ImportError:
            pass
        
        return False  # Don't suppress exceptions


def validate_reproducibility(func, seed: int = 42, n_runs: int = 2) -> bool:
    """
    Validate that a function produces reproducible results.
    
    Args:
        func: Function to test (should take no arguments)
        seed: Random seed to use
        n_runs: Number of runs to compare
        
    Returns:
        True if all runs produce identical results, False otherwise
        
    Example:
        >>> def my_random_function():
        ...     return np.random.randn(10)
        >>> is_reproducible = validate_reproducibility(my_random_function)
    """
    results = []
    
    for _ in range(n_runs):
        set_random_seed(seed)
        result = func()
        results.append(result)
    
    # Check if all results are equal
    first_result = results[0]
    for result in results[1:]:
        if isinstance(first_result, np.ndarray):
            if not np.allclose(first_result, result):
                return False
        else:
            if first_result != result:
                return False
    
    return True


# Aliases for backward compatibility
set_random_seeds = set_random_seed


def get_random_state() -> Dict[str, Any]:
    """
    Get the current state of all random number generators.
    
    Returns:
        Dictionary containing the state of all RNGs
        
    Example:
        >>> state = get_random_state()
        >>> # ... do some random operations ...
        >>> set_random_state(state)  # Restore state
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
    }
    
    try:
        import torch
        state['torch'] = torch.get_rng_state()
        if torch.cuda.is_available():
            state['torch_cuda'] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    
    return state


def set_random_state(state: Dict[str, Any]) -> None:
    """
    Restore the state of all random number generators.
    
    Args:
        state: Dictionary containing RNG states (from get_random_state())
        
    Example:
        >>> state = get_random_state()
        >>> # ... do some random operations ...
        >>> set_random_state(state)  # Restore state
    """
    if 'python' in state:
        random.setstate(state['python'])
    
    if 'numpy' in state:
        np.random.set_state(state['numpy'])
    
    try:
        import torch
        if 'torch' in state:
            torch.set_rng_state(state['torch'])
        if 'torch_cuda' in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['torch_cuda'])
    except ImportError:
        pass


# Legacy aliases for backward compatibility
class ReproducibilityManager:
    """Legacy class for backward compatibility."""
    
    @staticmethod
    def set_seed(seed: int = 42):
        """Set random seed (legacy method)."""
        set_random_seed(seed)
    
    @staticmethod
    def get_state():
        """Get random state (legacy method)."""
        return get_random_state()
    
    @staticmethod
    def set_state(state):
        """Set random state (legacy method)."""
        set_random_state(state)


def setup_reproducibility(seed: int = 42):
    """
    Setup reproducibility for the entire system (legacy function).
    
    Args:
        seed: Random seed to use
    """
    set_random_seed(seed)


def ensure_deterministic_operations():
    """
    Ensure all operations are deterministic (legacy function).
    
    This is automatically handled by set_random_seed().
    """
    pass  # Already handled in set_random_seed


class ReproducibilityContext(ReproducibleContext):
    """Legacy alias for ReproducibleContext."""
    pass
