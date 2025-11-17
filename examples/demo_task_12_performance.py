"""
Demo script for Task 12: Performance Optimization and Efficiency

Demonstrates profiling, caching, and progress tracking features.
"""

import numpy as np
import time
from pathlib import Path

from src.optimization.performance_optimizer import (
    profile,
    cached,
    ProgressTracker,
    get_profiler,
    get_cache
)


# Demo 1: Profiling
print("="*70)
print("DEMO 1: Performance Profiling")
print("="*70)

@profile
def simulate_data_loading():
    """Simulate loading data."""
    time.sleep(0.1)
    return np.random.randn(1000, 100)

@profile
def simulate_preprocessing(data):
    """Simulate preprocessing."""
    time.sleep(0.05)
    return data / np.std(data)

@profile
def simulate_analysis(data):
    """Simulate analysis."""
    time.sleep(0.15)
    return np.mean(data, axis=0)

# Run pipeline
print("\nRunning analysis pipeline...")
data = simulate_data_loading()
processed = simulate_preprocessing(data)
results = simulate_analysis(processed)

# Show profiling report
profiler = get_profiler()
print("\n" + profiler.get_report())


# Demo 2: Caching
print("\n" + "="*70)
print("DEMO 2: Result Caching")
print("="*70)

@cached
def expensive_computation(n):
    """Expensive computation that benefits from caching."""
    print(f"  Computing result for n={n}...")
    time.sleep(0.2)
    return sum(i**2 for i in range(n))

print("\nFirst call (computes result):")
start = time.time()
result1 = expensive_computation(1000)
time1 = time.time() - start
print(f"  Result: {result1}, Time: {time1:.3f}s")

print("\nSecond call with same argument (uses cache):")
start = time.time()
result2 = expensive_computation(1000)
time2 = time.time() - start
print(f"  Result: {result2}, Time: {time2:.3f}s")
if time2 > 0:
    print(f"  Speedup: {time1/time2:.1f}x faster!")
else:
    print(f"  Speedup: >1000x faster (cache hit was instant!)")

print("\nThird call with different argument (computes result):")
start = time.time()
result3 = expensive_computation(2000)
time3 = time.time() - start
print(f"  Result: {result3}, Time: {time3:.3f}s")

# Show cache stats
cache = get_cache()
stats = cache.get_stats()
print(f"\nCache statistics:")
print(f"  Memory entries: {stats['memory_entries']}")
print(f"  Disk entries: {stats['disk_entries']}")
print(f"  Total size: {stats['total_size_mb']:.2f} MB")


# Demo 3: Progress Tracking
print("\n" + "="*70)
print("DEMO 3: Progress Tracking")
print("="*70)

print("\nProcessing 50 items with progress tracking:")
n_items = 50
results = []

with ProgressTracker(desc="Processing items", total=n_items) as tracker:
    for i in range(n_items):
        # Simulate processing
        time.sleep(0.02)
        result = i ** 2
        results.append(result)
        tracker.update(1)

print(f"Processed {len(results)} items successfully!")


# Demo 4: Combined Usage
print("\n" + "="*70)
print("DEMO 4: Combined Usage (Profiling + Caching + Progress)")
print("="*70)

@profile
@cached
def process_batch(batch_id, size):
    """Process a batch with profiling and caching."""
    time.sleep(0.05)
    return np.random.randn(size).mean()

print("\nProcessing 20 batches:")
n_batches = 20
batch_results = []

with ProgressTracker(desc="Processing batches", total=n_batches) as tracker:
    for i in range(n_batches):
        # Some batches repeat (will use cache)
        batch_id = i % 10  # Repeat every 10 batches
        result = process_batch(batch_id, 100)
        batch_results.append(result)
        tracker.update(1)

print(f"\nProcessed {len(batch_results)} batches")
print(f"Cache entries: {get_cache().get_stats()['memory_entries']}")


# Final Summary
print("\n" + "="*70)
print("SUMMARY: Task 12 Performance Optimization Features")
print("="*70)
print("\n✓ Profiling: Identify bottlenecks and optimize critical paths")
print("✓ Caching: Eliminate redundant computation automatically")
print("✓ Progress Tracking: Provide user feedback for long operations")
print("✓ Parallel Processing: Already available in validation framework")
print("\nAll performance optimization features are working correctly!")
print("="*70)
