"""
Resource Manager for computational resource allocation and tracking.

This module provides resource management capabilities including:
- Memory usage tracking and allocation
- GPU resource management
- Parallel execution coordination
- Task cleanup and resource freeing

Task 14: Performance optimization and resource management
Requirements: 8.3, 3.1, 3.2
"""

import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..utils.logging_utils import get_logger


@dataclass
class ResourceAllocation:
    """Resource allocation information."""
    task_id: str
    memory_mb: float
    gpu_id: Optional[int] = None
    cpu_cores: int = 1
    allocated_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'task_id': self.task_id,
            'memory_mb': self.memory_mb,
            'gpu_id': self.gpu_id,
            'cpu_cores': self.cpu_cores,
            'allocated_at': self.allocated_at
        }


@dataclass
class ResourceStats:
    """System resource statistics."""
    total_memory_mb: float
    available_memory_mb: float
    used_memory_mb: float
    memory_percent: float
    cpu_count: int
    cpu_percent: float
    gpu_available: bool
    gpu_count: int = 0
    gpu_memory_mb: List[float] = field(default_factory=list)
    gpu_utilization: List[float] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'total_memory_mb': self.total_memory_mb,
            'available_memory_mb': self.available_memory_mb,
            'used_memory_mb': self.used_memory_mb,
            'memory_percent': self.memory_percent,
            'cpu_count': self.cpu_count,
            'cpu_percent': self.cpu_percent,
            'gpu_available': self.gpu_available,
            'gpu_count': self.gpu_count,
            'gpu_memory_mb': self.gpu_memory_mb,
            'gpu_utilization': self.gpu_utilization
        }


class ResourceManager:
    """
    Manage computational resources for discovery pipeline.
    
    Tracks memory usage, GPU allocation, and coordinates parallel execution
    to prevent resource exhaustion.
    
    Attributes:
        max_memory_gb: Maximum memory to use (GB)
        n_parallel: Maximum number of parallel tasks
        gpu_enabled: Whether to use GPU acceleration
        logger: Logger instance
        allocations: Current resource allocations
        completed_tasks: Set of completed task IDs
    """
    
    def __init__(
        self,
        max_memory_gb: float = 16.0,
        n_parallel: int = 4,
        gpu_enabled: bool = True,
        reserve_memory_gb: float = 2.0
    ):
        """
        Initialize resource manager.
        
        Args:
            max_memory_gb: Maximum memory to allocate (GB)
            n_parallel: Maximum parallel tasks
            gpu_enabled: Enable GPU usage
            reserve_memory_gb: Memory to reserve for system (GB)
        """
        self.max_memory_gb = max_memory_gb
        self.n_parallel = n_parallel
        self.gpu_enabled = gpu_enabled and TORCH_AVAILABLE
        self.reserve_memory_gb = reserve_memory_gb
        self.logger = get_logger(__name__)
        
        # Track allocations
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.completed_tasks: set = set()
        
        # GPU tracking
        self.gpu_count = 0
        self.gpu_allocations: Dict[int, List[str]] = {}  # gpu_id -> task_ids
        
        if self.gpu_enabled:
            self._initialize_gpu()
        
        # Log initial state
        stats = self.get_resource_stats()
        self.logger.info(f"ResourceManager initialized:")
        self.logger.info(f"  Max memory: {max_memory_gb:.1f} GB")
        self.logger.info(f"  Parallel tasks: {n_parallel}")
        self.logger.info(f"  GPU enabled: {self.gpu_enabled}")
        if self.gpu_enabled:
            self.logger.info(f"  GPUs available: {self.gpu_count}")
    
    def _initialize_gpu(self):
        """Initialize GPU tracking."""
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, GPU support disabled")
            self.gpu_enabled = False
            return
        
        if torch.cuda.is_available():
            self.gpu_count = torch.cuda.device_count()
            for gpu_id in range(self.gpu_count):
                self.gpu_allocations[gpu_id] = []
            self.logger.info(f"Detected {self.gpu_count} GPU(s)")
        else:
            self.logger.info("No CUDA GPUs detected")
            self.gpu_enabled = False
    
    def get_resource_stats(self) -> ResourceStats:
        """
        Get current system resource statistics.
        
        Returns:
            ResourceStats object with current system state
        """
        if not PSUTIL_AVAILABLE:
            # Return dummy stats if psutil not available
            return ResourceStats(
                total_memory_mb=self.max_memory_gb * 1024,
                available_memory_mb=self.max_memory_gb * 1024 * 0.5,
                used_memory_mb=self.max_memory_gb * 1024 * 0.5,
                memory_percent=50.0,
                cpu_count=mp.cpu_count(),
                cpu_percent=0.0,
                gpu_available=self.gpu_enabled,
                gpu_count=self.gpu_count
            )
        
        # Get memory stats
        mem = psutil.virtual_memory()
        total_memory_mb = mem.total / 1024 / 1024
        available_memory_mb = mem.available / 1024 / 1024
        used_memory_mb = mem.used / 1024 / 1024
        memory_percent = mem.percent
        
        # Get CPU stats
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get GPU stats
        gpu_memory_mb = []
        gpu_utilization = []
        if self.gpu_enabled and TORCH_AVAILABLE:
            for gpu_id in range(self.gpu_count):
                try:
                    # Get GPU memory
                    mem_allocated = torch.cuda.memory_allocated(gpu_id) / 1024 / 1024
                    gpu_memory_mb.append(mem_allocated)
                    
                    # GPU utilization requires nvidia-ml-py3
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_utilization.append(util.gpu)
                    except:
                        gpu_utilization.append(0.0)
                except:
                    gpu_memory_mb.append(0.0)
                    gpu_utilization.append(0.0)
        
        return ResourceStats(
            total_memory_mb=total_memory_mb,
            available_memory_mb=available_memory_mb,
            used_memory_mb=used_memory_mb,
            memory_percent=memory_percent,
            cpu_count=cpu_count,
            cpu_percent=cpu_percent,
            gpu_available=self.gpu_enabled,
            gpu_count=self.gpu_count,
            gpu_memory_mb=gpu_memory_mb,
            gpu_utilization=gpu_utilization
        )
    
    def allocate_simulation(
        self,
        task_id: str,
        lattice_size: int,
        n_samples: int = 100
    ) -> bool:
        """
        Check if resources available for Monte Carlo simulation.
        
        Args:
            task_id: Unique task identifier
            lattice_size: Size of lattice (per dimension)
            n_samples: Number of samples to generate
            
        Returns:
            True if resources can be allocated, False otherwise
        """
        # Estimate memory requirements
        # Rough estimate: lattice_size^3 * n_samples * 8 bytes (float64)
        estimated_memory_mb = (lattice_size ** 3) * n_samples * 8 / 1024 / 1024
        
        # Add overhead (20%)
        estimated_memory_mb *= 1.2
        
        return self._allocate_resources(
            task_id=task_id,
            memory_mb=estimated_memory_mb,
            cpu_cores=1,
            gpu_id=None
        )
    
    def allocate_vae_training(
        self,
        task_id: str,
        data_size: int,
        batch_size: int = 32,
        use_gpu: bool = True
    ) -> Tuple[bool, Optional[int]]:
        """
        Check if resources available for VAE training.
        
        Args:
            task_id: Unique task identifier
            data_size: Number of training samples
            batch_size: Batch size for training
            use_gpu: Whether to use GPU
            
        Returns:
            Tuple of (success, gpu_id)
        """
        # Estimate memory requirements
        # Model size + batch data + gradients
        estimated_memory_mb = (data_size * 8 / 1024 / 1024) * 0.3  # 30% of data size
        estimated_memory_mb += 500  # Model and optimizer overhead
        
        # Try to allocate GPU if requested
        gpu_id = None
        if use_gpu and self.gpu_enabled:
            gpu_id = self._find_available_gpu()
            if gpu_id is None:
                self.logger.warning(f"No GPU available for task {task_id}, using CPU")
        
        success = self._allocate_resources(
            task_id=task_id,
            memory_mb=estimated_memory_mb,
            cpu_cores=2 if gpu_id is None else 1,
            gpu_id=gpu_id
        )
        
        return success, gpu_id
    
    def _allocate_resources(
        self,
        task_id: str,
        memory_mb: float,
        cpu_cores: int = 1,
        gpu_id: Optional[int] = None
    ) -> bool:
        """
        Internal method to allocate resources.
        
        Args:
            task_id: Task identifier
            memory_mb: Memory required (MB)
            cpu_cores: CPU cores required
            gpu_id: GPU ID if GPU required
            
        Returns:
            True if allocation successful
        """
        # Check if already allocated
        if task_id in self.allocations:
            self.logger.warning(f"Task {task_id} already has resources allocated")
            return True
        
        # Check parallel task limit
        active_tasks = len(self.allocations) - len(self.completed_tasks)
        if active_tasks >= self.n_parallel:
            self.logger.debug(f"Parallel task limit reached ({self.n_parallel})")
            return False
        
        # Check memory availability
        stats = self.get_resource_stats()
        available_memory_mb = stats.available_memory_mb - (self.reserve_memory_gb * 1024)
        
        # Calculate currently allocated memory
        allocated_memory_mb = sum(
            alloc.memory_mb for alloc in self.allocations.values()
            if alloc.task_id not in self.completed_tasks
        )
        
        if allocated_memory_mb + memory_mb > available_memory_mb:
            self.logger.debug(
                f"Insufficient memory: need {memory_mb:.1f} MB, "
                f"available {available_memory_mb - allocated_memory_mb:.1f} MB"
            )
            return False
        
        # Check GPU availability
        if gpu_id is not None:
            if gpu_id >= self.gpu_count:
                self.logger.error(f"Invalid GPU ID: {gpu_id}")
                return False
            
            # Check if GPU is overloaded
            if len(self.gpu_allocations[gpu_id]) >= 2:  # Max 2 tasks per GPU
                self.logger.debug(f"GPU {gpu_id} is at capacity")
                return False
        
        # Allocate resources
        allocation = ResourceAllocation(
            task_id=task_id,
            memory_mb=memory_mb,
            gpu_id=gpu_id,
            cpu_cores=cpu_cores
        )
        
        self.allocations[task_id] = allocation
        
        if gpu_id is not None:
            self.gpu_allocations[gpu_id].append(task_id)
        
        self.logger.debug(
            f"Allocated resources for {task_id}: "
            f"{memory_mb:.1f} MB, {cpu_cores} cores"
            + (f", GPU {gpu_id}" if gpu_id is not None else "")
        )
        
        return True
    
    def _find_available_gpu(self) -> Optional[int]:
        """
        Find GPU with least allocations.
        
        Returns:
            GPU ID or None if all busy
        """
        if not self.gpu_enabled or self.gpu_count == 0:
            return None
        
        # Find GPU with fewest allocations
        min_allocations = float('inf')
        best_gpu = None
        
        for gpu_id in range(self.gpu_count):
            n_allocs = len(self.gpu_allocations[gpu_id])
            if n_allocs < min_allocations and n_allocs < 2:  # Max 2 per GPU
                min_allocations = n_allocs
                best_gpu = gpu_id
        
        return best_gpu
    
    def mark_completed(self, task_id: str):
        """
        Mark task as completed (but keep allocation for tracking).
        
        Args:
            task_id: Task identifier
        """
        if task_id not in self.allocations:
            self.logger.warning(f"Task {task_id} not found in allocations")
            return
        
        self.completed_tasks.add(task_id)
        self.logger.debug(f"Task {task_id} marked as completed")
    
    def cleanup_completed_tasks(self):
        """
        Free resources from completed tasks.
        
        Removes allocations for tasks marked as completed.
        """
        tasks_to_remove = []
        
        for task_id in self.completed_tasks:
            if task_id in self.allocations:
                allocation = self.allocations[task_id]
                
                # Remove from GPU allocations
                if allocation.gpu_id is not None:
                    if task_id in self.gpu_allocations[allocation.gpu_id]:
                        self.gpu_allocations[allocation.gpu_id].remove(task_id)
                
                tasks_to_remove.append(task_id)
        
        # Remove allocations
        for task_id in tasks_to_remove:
            del self.allocations[task_id]
        
        # Clear completed set
        self.completed_tasks.clear()
        
        if tasks_to_remove:
            self.logger.info(f"Cleaned up {len(tasks_to_remove)} completed tasks")
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """
        Get summary of current allocations.
        
        Returns:
            Dictionary with allocation statistics
        """
        active_allocations = [
            alloc for task_id, alloc in self.allocations.items()
            if task_id not in self.completed_tasks
        ]
        
        total_memory_mb = sum(alloc.memory_mb for alloc in active_allocations)
        total_cpu_cores = sum(alloc.cpu_cores for alloc in active_allocations)
        
        gpu_usage = {
            gpu_id: len(tasks) for gpu_id, tasks in self.gpu_allocations.items()
        }
        
        return {
            'active_tasks': len(active_allocations),
            'completed_tasks': len(self.completed_tasks),
            'total_memory_mb': total_memory_mb,
            'total_cpu_cores': total_cpu_cores,
            'gpu_usage': gpu_usage,
            'max_parallel': self.n_parallel
        }


class ParallelExecutor:
    """
    Parallel execution coordinator for discovery pipeline.
    
    Manages parallel execution of parameter point simulations with
    resource allocation and error handling.
    """
    
    def __init__(
        self,
        resource_manager: ResourceManager,
        use_processes: bool = True,
        max_workers: Optional[int] = None
    ):
        """
        Initialize parallel executor.
        
        Args:
            resource_manager: Resource manager for allocation
            use_processes: Use processes (True) or threads (False)
            max_workers: Maximum worker processes/threads (None = auto)
        """
        self.resource_manager = resource_manager
        self.use_processes = use_processes
        self.max_workers = max_workers or resource_manager.n_parallel
        self.logger = get_logger(__name__)
        
        self.logger.info(
            f"ParallelExecutor initialized: "
            f"{'processes' if use_processes else 'threads'}, "
            f"max_workers={self.max_workers}"
        )
    
    def execute_parallel(
        self,
        func: Callable,
        tasks: List[Tuple[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute function in parallel for multiple tasks.
        
        Args:
            func: Function to execute (must accept task_id and task_data)
            tasks: List of (task_id, task_data) tuples
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping task_id to result
        """
        results = {}
        errors = {}
        
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for task_id, task_data in tasks:
                future = executor.submit(func, task_id, task_data)
                future_to_task[future] = task_id
            
            # Collect results as they complete
            completed = 0
            total = len(tasks)
            
            for future in as_completed(future_to_task):
                task_id = future_to_task[future]
                completed += 1
                
                try:
                    result = future.result()
                    results[task_id] = result
                    self.resource_manager.mark_completed(task_id)
                    
                    self.logger.debug(f"Task {task_id} completed successfully")
                    
                except Exception as e:
                    self.logger.error(f"Task {task_id} failed: {e}")
                    errors[task_id] = str(e)
                
                # Progress callback
                if progress_callback:
                    progress_callback(completed, total)
                
                # Periodic cleanup
                if completed % 10 == 0:
                    self.resource_manager.cleanup_completed_tasks()
        
        # Final cleanup
        self.resource_manager.cleanup_completed_tasks()
        
        self.logger.info(
            f"Parallel execution complete: "
            f"{len(results)} succeeded, {len(errors)} failed"
        )
        
        return {
            'results': results,
            'errors': errors,
            'success_rate': len(results) / total if total > 0 else 0.0
        }


def estimate_simulation_memory(
    lattice_size: int,
    n_samples: int,
    dimensions: int = 3
) -> float:
    """
    Estimate memory requirements for Monte Carlo simulation.
    
    Args:
        lattice_size: Lattice size per dimension
        n_samples: Number of samples
        dimensions: Number of dimensions (2 or 3)
        
    Returns:
        Estimated memory in MB
    """
    # Configuration storage: lattice_size^dimensions * n_samples * 8 bytes
    config_memory = (lattice_size ** dimensions) * n_samples * 8 / 1024 / 1024
    
    # Observables storage (magnetization, energy, etc.)
    observables_memory = n_samples * 8 * 10 / 1024 / 1024  # ~10 observables
    
    # Working memory (current state, etc.)
    working_memory = (lattice_size ** dimensions) * 8 / 1024 / 1024
    
    # Total with 20% overhead
    total_memory = (config_memory + observables_memory + working_memory) * 1.2
    
    return total_memory


def estimate_vae_memory(
    n_samples: int,
    lattice_size: int,
    dimensions: int = 3,
    latent_dim: int = 10
) -> float:
    """
    Estimate memory requirements for VAE training.
    
    Args:
        n_samples: Number of training samples
        lattice_size: Lattice size per dimension
        dimensions: Number of dimensions
        latent_dim: Latent space dimensionality
        
    Returns:
        Estimated memory in MB
    """
    # Input data
    input_size = (lattice_size ** dimensions) * n_samples * 4 / 1024 / 1024  # float32
    
    # Model parameters (rough estimate)
    model_memory = 100  # MB
    
    # Optimizer state (2x model parameters for Adam)
    optimizer_memory = 200  # MB
    
    # Batch processing overhead
    batch_memory = (lattice_size ** dimensions) * 32 * 4 / 1024 / 1024  # batch_size=32
    
    # Total with 30% overhead
    total_memory = (input_size + model_memory + optimizer_memory + batch_memory) * 1.3
    
    return total_memory
