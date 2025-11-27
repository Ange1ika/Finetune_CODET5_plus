"""
Performance Profiling Utilities

This module provides timing decorators and performance profiling tools
for identifying bottlenecks in training and data loading pipelines.

Usage:
    from profiling_utils import timing, print_timing_stats, reset_timing_stats
    
    @timing
    def my_function():
        pass
    
    # After your code runs
    print_timing_stats()
"""

import time
import functools
from collections import defaultdict


# Global dictionary to store timing statistics
TIMING_STATS = defaultdict(lambda: {
    'calls': 0, 
    'total_time': 0, 
    'avg_time': 0, 
    'min_time': float('inf'), 
    'max_time': 0
})


def timing(func):
    """
    Decorator to measure and log function execution time.
    Tracks: total time, average time, min/max times, number of calls.
    
    Usage:
        @timing
        def my_function():
            pass
    
    The decorator will:
    - Measure execution time for each call
    - Update global statistics (TIMING_STATS)
    - Log warnings for slow functions (>0.1 seconds)
    - Generate comprehensive report via print_timing_stats()
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        
        # Update statistics
        stats = TIMING_STATS[func.__name__]
        stats['calls'] += 1
        stats['total_time'] += elapsed
        stats['avg_time'] = stats['total_time'] / stats['calls']
        stats['min_time'] = min(stats['min_time'], elapsed)
        stats['max_time'] = max(stats['max_time'], elapsed)
        
        # Log if function takes significant time (>0.1 seconds)
        if elapsed > 0.1:
            print(f"⏱️  {func.__name__} took {elapsed:.3f}s")
        
        return result
    return wrapper


def print_timing_stats():
    """
    Print comprehensive timing statistics for all profiled functions.
    
    Displays a formatted table with:
    - Function name
    - Number of calls
    - Total time spent
    - Average time per call
    - Minimum time
    - Maximum time
    
    Functions are sorted by total time (descending) to highlight
    the biggest bottlenecks first.
    """
    if not TIMING_STATS:
        print("No timing statistics collected.")
        return
    
    print("\n" + "="*80)
    print("PERFORMANCE PROFILING RESULTS")
    print("="*80)
    print(f"{'Function':<40} {'Calls':>8} {'Total(s)':>10} {'Avg(s)':>10} {'Min(s)':>10} {'Max(s)':>10}")
    print("-"*80)
    
    # Sort by total time (descending)
    sorted_stats = sorted(TIMING_STATS.items(), key=lambda x: x[1]['total_time'], reverse=True)
    
    for func_name, stats in sorted_stats:
        print(f"{func_name:<40} {stats['calls']:>8} {stats['total_time']:>10.3f} "
              f"{stats['avg_time']:>10.3f} {stats['min_time']:>10.3f} {stats['max_time']:>10.3f}")
    
    print("="*80)
    print(f"Total profiled time: {sum(s['total_time'] for s in TIMING_STATS.values()):.3f}s")
    print("="*80 + "\n")


def reset_timing_stats():
    """
    Reset all timing statistics.
    
    Clears the TIMING_STATS dictionary, useful when you want to:
    - Profile different phases separately
    - Reset between training runs
    - Clear stats after warmup phase
    """
    TIMING_STATS.clear()


def get_timing_stats():
    """
    Get the current timing statistics dictionary.
    
    Returns:
        dict: Dictionary mapping function names to their timing statistics
    """
    return dict(TIMING_STATS)


def print_bottleneck_summary(top_n=5):
    """
    Print a summary of the top N bottlenecks.
    
    Args:
        top_n (int): Number of top bottlenecks to display
    """
    if not TIMING_STATS:
        print("No timing statistics collected.")
        return
    
    sorted_stats = sorted(TIMING_STATS.items(), key=lambda x: x[1]['total_time'], reverse=True)
    
    print("\n" + "="*60)
    print(f"TOP {top_n} BOTTLENECKS")
    print("="*60)
    
    total_time = sum(s['total_time'] for s in TIMING_STATS.values())
    
    for i, (func_name, stats) in enumerate(sorted_stats[:top_n], 1):
        percentage = (stats['total_time'] / total_time * 100) if total_time > 0 else 0
        print(f"{i}. {func_name}")
        print(f"   Total time: {stats['total_time']:.3f}s ({percentage:.1f}% of total)")
        print(f"   Calls: {stats['calls']}, Avg: {stats['avg_time']:.3f}s")
        print()
    
    print("="*60 + "\n")
