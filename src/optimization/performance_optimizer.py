"""
Performance Optimization Module

This module provides performance optimization utilities including:
- Code profiling and bottleneck identification
- Result caching with automatic invalidation
- Progress indicators for long-running operations
- Memory management utilities

Task 12: Performance optimization and efficiency
Requirements: 6.1, 6.3, 6.4, 6.5
"""

import time
import functools
import hashlib
import pickle
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from contextlib import contextmanager

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback progress indicator
    class tqdm:
        def __init__(self, iterable=None, desc=None, total=None, **kwargs):
            self.iterable = iterable
            self.desc = desc
            self.total = total or (len(iterable) if iterable else 0)
            self.n = 0
            
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.update(1)
        
        def update(self, n=1):
            self.n += n
            if self.total > 0:
                pct = 100 * self.n / self.total
                print(f"\r{self.desc}: {self.n}/{self.total} ({pct:.1f}%)", end='', flush=True)
        
        def close(self):
            print()
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            self.close()

from ..utils.logging_utils import get_logger


@dataclass
class ProfileResult:
    """Result from code profiling."""
    function_name: str
    execution_time: float
    memory_used: Optional[float] = None
    call_count: int = 1
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'function_name': self.function_name,
            'execution_time': self.execution_time,
            'memory_used': self.memory_used,
            'call_count': self.call_count
        }


class PerformanceProfiler:
    """
    Performance profiler for identifying bottlenecks.
    
    Tracks execution time and memory usage for functions.
    """
    
    def __init__(self):
        """Initialize profiler."""
        self.logger = get_logger(__name__)
        self.profiles: Dict[str, ProfileResult] = {}
        
    def profile(self, func: Callable) -> Callable:
        """
        Decorator to profile a function.
        
        Args:
            func: Function to profile
            
        Returns:
            Wrapped function with profiling
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Track memory before (if available)
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
                mem_available = True
            except ImportError:
                mem_before = 0
                mem_available = False
            
            # Time execution
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Track memory after
            if mem_available:
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = mem_after - mem_before
            else:
                memory_used = None
            
            # Update profile
            if func_name in self.profiles:
                profile = self.profiles[func_name]
                profile.execution_time += execution_time
                profile.call_count += 1
                if memory_used is not None and profile.memory_used is not None:
                    profile.memory_used += memory_used
            else:
                self.profiles[func_name] = ProfileResult(
                    function_name=func_name,
                    execution_time=execution_time,
                    memory_used=memory_used,
                    call_count=1
                )
            
            return result
        
        return wrapper
    
    @contextmanager
    def profile_block(self, block_name: str):
        """
        Context manager to profile a code block.
        
        Args:
            block_name: Name for the profiled block
            
        Example:
            with profiler.profile_block("data_loading"):
                data = load_data()
        """
        # Track memory before
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            mem_available = True
        except ImportError:
            mem_before = 0
            mem_available = False
        
        # Time execution
        start_time = time.time()
        
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            
            # Track memory after
            if mem_available:
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = mem_after - mem_before
            else:
                memory_used = None
            
            # Update profile
            if block_name in self.profiles:
                profile = self.profiles[block_name]
                profile.execution_time += execution_time
                profile.call_count += 1
                if memory_used is not None and profile.memory_used is not None:
                    profile.memory_used += memory_used
            else:
                self.profiles[block_name] = ProfileResult(
                    function_name=block_name,
                    execution_time=execution_time,
                    memory_used=memory_used,
                    call_count=1
                )
    
    def get_report(self, sort_by: str = 'time') -> str:
        """
        Get profiling report.
        
        Args:
            sort_by: Sort by 'time' or 'memory'
            
        Returns:
            Formatted report string
        """
        if not self.profiles:
            return "No profiling data available"
        
        # Sort profiles
        if sort_by == 'time':
            sorted_profiles = sorted(
                self.profiles.values(),
                key=lambda p: p.execution_time,
                reverse=True
            )
        elif sort_by == 'memory':
            sorted_profiles = sorted(
                self.profiles.values(),
                key=lambda p: p.memory_used if p.memory_used else 0,
                reverse=True
            )
        else:
            sorted_profiles = list(self.profiles.values())
        
        # Build report
        lines = ["=" * 80]
        lines.append("PERFORMANCE PROFILING REPORT")
        lines.append("=" * 80)
        lines.append(f"{'Function':<50} {'Time (s)':<12} {'Calls':<8} {'Memory (MB)':<12}")
        lines.append("-" * 80)
        
        total_time = sum(p.execution_time for p in self.profiles.values())
        
        for profile in sorted_profiles:
            time_pct = 100 * profile.execution_time / total_time if total_time > 0 else 0
            mem_str = f"{profile.memory_used:.2f}" if profile.memory_used else "N/A"
            
            lines.append(
                f"{profile.function_name:<50} "
                f"{profile.execution_time:>8.3f} ({time_pct:>4.1f}%) "
                f"{profile.call_count:>6} "
                f"{mem_str:>10}"
            )
        
        lines.append("-" * 80)
        lines.append(f"{'TOTAL':<50} {total_time:>8.3f} (100.0%)")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def save_report(self, filepath: Path):
        """
        Save profiling report to file.
        
        Args:
            filepath: Path to save report
        """
        report = self.get_report()
        filepath.write_text(report)
        self.logger.info(f"Profiling report saved to {filepath}")
    
    def reset(self):
        """Reset all profiling data."""
        self.profiles.clear()


class ResultCache:
    """
    Result caching system with automatic invalidation.
    
    Caches function results based on input arguments to avoid
    redundant computation.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, max_size_mb: float = 1000):
        """
        Initialize result cache.
        
        Args:
            cache_dir: Directory for cache files (None = memory only)
            max_size_mb: Maximum cache size in MB
        """
        self.logger = get_logger(__name__)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_size_mb = max_size_mb
        self.memory_cache: Dict[str, Any] = {}
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Result cache initialized at {self.cache_dir}")
        else:
            self.logger.info("Result cache initialized (memory only)")
    
    def _compute_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """
        Compute cache key from function name and arguments.
        
        Args:
            func_name: Function name
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Cache key string
        """
        # Create hashable representation
        key_data = {
            'func': func_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        key_str = json.dumps(key_data, sort_keys=True)
        
        # Hash to fixed-length key
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def cached(self, func: Callable) -> Callable:
        """
        Decorator to cache function results.
        
        Args:
            func: Function to cache
            
        Returns:
            Wrapped function with caching
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            cache_key = self._compute_key(func_name, args, kwargs)
            
            # Check memory cache first
            if cache_key in self.memory_cache:
                self.logger.debug(f"Cache hit (memory): {func_name}")
                return self.memory_cache[cache_key]
            
            # Check disk cache
            if self.cache_dir:
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                if cache_file.exists():
                    try:
                        with open(cache_file, 'rb') as f:
                            result = pickle.load(f)
                        self.logger.debug(f"Cache hit (disk): {func_name}")
                        # Store in memory cache for faster access
                        self.memory_cache[cache_key] = result
                        return result
                    except Exception as e:
                        self.logger.warning(f"Failed to load cache: {e}")
            
            # Cache miss - compute result
            self.logger.debug(f"Cache miss: {func_name}")
            result = func(*args, **kwargs)
            
            # Store in memory cache
            self.memory_cache[cache_key] = result
            
            # Store in disk cache
            if self.cache_dir:
                try:
                    cache_file = self.cache_dir / f"{cache_key}.pkl"
                    with open(cache_file, 'wb') as f:
                        pickle.dump(result, f)
                    self.logger.debug(f"Cached to disk: {func_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to cache result: {e}")
            
            # Check cache size and clean if needed
            self._check_cache_size()
            
            return result
        
        return wrapper
    
    def _check_cache_size(self):
        """Check cache size and clean if exceeds limit."""
        if not self.cache_dir:
            return
        
        # Calculate total cache size
        total_size_mb = sum(
            f.stat().st_size for f in self.cache_dir.glob("*.pkl")
        ) / 1024 / 1024
        
        if total_size_mb > self.max_size_mb:
            self.logger.warning(
                f"Cache size ({total_size_mb:.1f} MB) exceeds limit ({self.max_size_mb} MB)"
            )
            # Remove oldest files
            cache_files = sorted(
                self.cache_dir.glob("*.pkl"),
                key=lambda f: f.stat().st_mtime
            )
            
            while total_size_mb > self.max_size_mb * 0.8 and cache_files:
                oldest = cache_files.pop(0)
                size_mb = oldest.stat().st_size / 1024 / 1024
                oldest.unlink()
                total_size_mb -= size_mb
                self.logger.debug(f"Removed old cache file: {oldest.name}")
    
    def clear(self):
        """Clear all cached results."""
        self.memory_cache.clear()
        
        if self.cache_dir:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            self.logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        stats = {
            'memory_entries': len(self.memory_cache),
            'disk_entries': 0,
            'total_size_mb': 0.0
        }
        
        if self.cache_dir:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            stats['disk_entries'] = len(cache_files)
            stats['total_size_mb'] = sum(
                f.stat().st_size for f in cache_files
            ) / 1024 / 1024
        
        return stats


class ProgressTracker:
    """
    Progress tracking for long-running operations.
    
    Provides progress bars with ETA estimation and cancellation support.
    """
    
    def __init__(self, desc: str = "Processing", total: Optional[int] = None):
        """
        Initialize progress tracker.
        
        Args:
            desc: Description of the operation
            total: Total number of items (None for unknown)
        """
        self.desc = desc
        self.total = total
        self.pbar = None
        self.cancelled = False
        
    def __enter__(self):
        """Start progress tracking."""
        self.pbar = tqdm(total=self.total, desc=self.desc, unit='it')
        return self
    
    def __exit__(self, *args):
        """Stop progress tracking."""
        if self.pbar:
            self.pbar.close()
    
    def update(self, n: int = 1):
        """
        Update progress.
        
        Args:
            n: Number of items completed
        """
        if self.pbar:
            self.pbar.update(n)
    
    def set_description(self, desc: str):
        """
        Update description.
        
        Args:
            desc: New description
        """
        if self.pbar:
            self.pbar.set_description(desc)
    
    def cancel(self):
        """Cancel the operation."""
        self.cancelled = True
        if self.pbar:
            self.pbar.set_description(f"{self.desc} (CANCELLED)")
    
    def is_cancelled(self) -> bool:
        """Check if operation was cancelled."""
        return self.cancelled


def create_progress_tracker(desc: str = "Processing",
                            total: Optional[int] = None) -> ProgressTracker:
    """
    Factory function to create progress tracker.
    
    Args:
        desc: Description of the operation
        total: Total number of items
        
    Returns:
        ProgressTracker instance
    """
    return ProgressTracker(desc=desc, total=total)


# Global profiler instance
_global_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
    """Get global profiler instance."""
    return _global_profiler


# Global cache instance
_global_cache: Optional[ResultCache] = None


def get_cache(cache_dir: Optional[Path] = None) -> ResultCache:
    """
    Get global cache instance.
    
    Args:
        cache_dir: Cache directory (only used on first call)
        
    Returns:
        ResultCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = ResultCache(cache_dir=cache_dir)
    return _global_cache


def profile(func: Callable) -> Callable:
    """
    Convenience decorator for profiling.
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapped function
    """
    return get_profiler().profile(func)


def cached(func: Callable) -> Callable:
    """
    Convenience decorator for caching.
    
    Args:
        func: Function to cache
        
    Returns:
        Wrapped function
    """
    return get_cache().cached(func)
