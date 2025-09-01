"""
Reproducibility utilities for the Prometheus project.
Manages random seeds across NumPy, PyTorch, and Python random modules.
"""

import os
import random
import logging
import platform
import subprocess
from typing import Dict, Any, Optional
from datetime import datetime
import hashlib
import json

import numpy as np

# PyTorch imports with error handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class ReproducibilityManager:
    """Manages reproducibility settings and environment tracking."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.logger = logging.getLogger(__name__)
        self.environment_info = {}
        
    def set_seeds(self, seed: Optional[int] = None) -> None:
        """
        Set random seeds for all relevant libraries.
        
        Args:
            seed: Random seed to use. If None, uses the instance seed.
        """
        if seed is not None:
            self.seed = seed
        
        # Set Python random seed
        random.seed(self.seed)
        self.logger.info(f"Set Python random seed to {self.seed}")
        
        # Set NumPy seed
        np.random.seed(self.seed)
        self.logger.info(f"Set NumPy random seed to {self.seed}")
        
        # Set PyTorch seeds if available
        if TORCH_AVAILABLE:
            torch.manual_seed(self.seed)
            self.logger.info(f"Set PyTorch CPU seed to {self.seed}")
            
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
                self.logger.info(f"Set PyTorch CUDA seed to {self.seed}")
                
                # Additional CUDA reproducibility settings
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                self.logger.info("Set CUDA deterministic mode")
        else:
            self.logger.warning("PyTorch not available, skipping PyTorch seed setting")
        
        # Set environment variable for hash seed (affects Python's hash() function)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        self.logger.info(f"Set PYTHONHASHSEED to {self.seed}")
    
    def capture_environment_info(self) -> Dict[str, Any]:
        """
        Capture comprehensive environment information for reproducibility.
        
        Returns:
            Dict containing environment information.
        """
        env_info = {
            'timestamp': datetime.now().isoformat(),
            'seed': self.seed,
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'python_implementation': platform.python_implementation()
            },
            'packages': self._get_package_versions(),
            'environment_variables': self._get_relevant_env_vars(),
            'hardware': self._get_hardware_info()
        }
        
        self.environment_info = env_info
        self.logger.info("Captured environment information")
        return env_info
    
    def _get_package_versions(self) -> Dict[str, str]:
        """Get versions of relevant packages."""
        packages = {}
        
        # Core packages
        try:
            packages['numpy'] = np.__version__
        except:
            packages['numpy'] = 'not available'
        
        if TORCH_AVAILABLE:
            try:
                packages['torch'] = torch.__version__
            except:
                packages['torch'] = 'not available'
        else:
            packages['torch'] = 'not installed'
        
        # Try to get other common packages
        package_names = ['scipy', 'matplotlib', 'h5py', 'yaml', 'sklearn']
        for pkg_name in package_names:
            try:
                pkg = __import__(pkg_name)
                packages[pkg_name] = getattr(pkg, '__version__', 'version unknown')
            except ImportError:
                packages[pkg_name] = 'not installed'
        
        return packages
    
    def _get_relevant_env_vars(self) -> Dict[str, str]:
        """Get relevant environment variables."""
        relevant_vars = [
            'PYTHONHASHSEED',
            'CUDA_VISIBLE_DEVICES',
            'OMP_NUM_THREADS',
            'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS'
        ]
        
        env_vars = {}
        for var in relevant_vars:
            env_vars[var] = os.environ.get(var, 'not set')
        
        return env_vars
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        hardware_info = {}
        
        # CPU information
        try:
            if platform.system() == "Linux":
                with open('/proc/cpuinfo', 'r') as f:
                    cpu_info = f.read()
                    # Extract CPU model
                    for line in cpu_info.split('\n'):
                        if 'model name' in line:
                            hardware_info['cpu_model'] = line.split(':')[1].strip()
                            break
            else:
                hardware_info['cpu_model'] = platform.processor()
        except:
            hardware_info['cpu_model'] = 'unknown'
        
        # GPU information
        if TORCH_AVAILABLE and torch.cuda.is_available():
            hardware_info['cuda_available'] = True
            hardware_info['cuda_version'] = torch.version.cuda
            hardware_info['gpu_count'] = torch.cuda.device_count()
            hardware_info['gpu_names'] = [
                torch.cuda.get_device_name(i) 
                for i in range(torch.cuda.device_count())
            ]
        else:
            hardware_info['cuda_available'] = False
        
        return hardware_info
    
    def save_environment_info(self, filepath: str) -> None:
        """
        Save environment information to a JSON file.
        
        Args:
            filepath: Path to save the environment information.
        """
        if not self.environment_info:
            self.capture_environment_info()
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.environment_info, f, indent=2)
        
        self.logger.info(f"Environment information saved to {filepath}")
    
    def generate_experiment_hash(self, config_dict: Dict[str, Any]) -> str:
        """
        Generate a unique hash for the experiment based on configuration and environment.
        
        Args:
            config_dict: Configuration dictionary.
            
        Returns:
            Hexadecimal hash string.
        """
        # Combine configuration and key environment info
        hash_data = {
            'config': config_dict,
            'seed': self.seed,
            'python_version': platform.python_version(),
            'packages': self._get_package_versions()
        }
        
        # Create hash
        hash_string = json.dumps(hash_data, sort_keys=True)
        experiment_hash = hashlib.sha256(hash_string.encode()).hexdigest()[:16]
        
        self.logger.info(f"Generated experiment hash: {experiment_hash}")
        return experiment_hash
    
    def verify_reproducibility(self, reference_hash: str, config_dict: Dict[str, Any]) -> bool:
        """
        Verify if current environment can reproduce results from reference hash.
        
        Args:
            reference_hash: Hash from previous experiment.
            config_dict: Current configuration.
            
        Returns:
            True if environment is compatible for reproduction.
        """
        current_hash = self.generate_experiment_hash(config_dict)
        is_reproducible = current_hash == reference_hash
        
        if is_reproducible:
            self.logger.info("Environment verified for reproducibility")
        else:
            self.logger.warning(
                f"Environment may not be reproducible. "
                f"Current hash: {current_hash}, Reference hash: {reference_hash}"
            )
        
        return is_reproducible


def setup_reproducibility(seed: int = 42) -> ReproducibilityManager:
    """
    Convenience function to set up reproducibility with a given seed.
    
    Args:
        seed: Random seed to use across all libraries.
        
    Returns:
        ReproducibilityManager instance.
    """
    manager = ReproducibilityManager(seed)
    manager.set_seeds()
    manager.capture_environment_info()
    return manager


def ensure_deterministic_operations():
    """Ensure deterministic operations for PyTorch if available."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        # Additional settings for deterministic behavior
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        # Set environment variable for deterministic CUDA operations
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        logger = logging.getLogger(__name__)
        logger.info("Enabled deterministic CUDA operations")


class ReproducibilityContext:
    """Context manager for reproducible code blocks."""
    
    def __init__(self, seed: int):
        self.seed = seed
        self.original_states = {}
        
    def __enter__(self):
        # Save current states
        self.original_states['python'] = random.getstate()
        self.original_states['numpy'] = np.random.get_state()
        
        if TORCH_AVAILABLE:
            self.original_states['torch'] = torch.get_rng_state()
            if torch.cuda.is_available():
                self.original_states['torch_cuda'] = torch.cuda.get_rng_state_all()
        
        # Set new seed
        manager = ReproducibilityManager(self.seed)
        manager.set_seeds()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original states
        random.setstate(self.original_states['python'])
        np.random.set_state(self.original_states['numpy'])
        
        if TORCH_AVAILABLE:
            torch.set_rng_state(self.original_states['torch'])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(self.original_states['torch_cuda'])