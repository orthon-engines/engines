"""
PRISM Memory Management Utilities

Compute → Write → Release → Next Step

Prevents memory bloat by explicitly releasing objects after writing to disk.
Critical for running on 16GB Mac without force-quit dialogs.
"""

import gc
import sys
import logging
from functools import wraps
from typing import Any, List, Optional
import platform

try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    # Windows doesn't have resource module
    HAS_RESOURCE = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


logger = logging.getLogger(__name__)


# =============================================================================
# Memory Size Estimation
# =============================================================================

def get_size_mb(obj: Any) -> float:
    """
    Get approximate memory size in MB.

    For Polars DataFrames, uses estimated_size().
    For other objects, uses sys.getsizeof() (underestimates).
    """
    if HAS_POLARS and isinstance(obj, pl.DataFrame):
        try:
            return obj.estimated_size('mb')
        except Exception:
            return sys.getsizeof(obj) / 1e6

    if HAS_POLARS and isinstance(obj, pl.LazyFrame):
        return 0.0  # LazyFrames don't hold data

    # For dicts/lists, try to sum children
    if isinstance(obj, dict):
        total = sys.getsizeof(obj)
        for k, v in obj.items():
            total += get_size_mb(k) * 1e6
            total += get_size_mb(v) * 1e6
        return total / 1e6

    if isinstance(obj, (list, tuple)):
        total = sys.getsizeof(obj)
        for item in obj:
            total += get_size_mb(item) * 1e6
        return total / 1e6

    return sys.getsizeof(obj) / 1e6


# =============================================================================
# Memory Usage Reporting
# =============================================================================

def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    if HAS_RESOURCE:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in bytes on Linux, KB on Mac
        if platform.system() == 'Darwin':
            return usage.ru_maxrss / 1e6  # Mac: bytes -> MB
        else:
            return usage.ru_maxrss / 1e3  # Linux: KB -> MB

    # Fallback: try psutil
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1e6
    except ImportError:
        return 0.0


def memory_report(prefix: str = "") -> float:
    """
    Print current memory usage.

    Args:
        prefix: Optional prefix for the message

    Returns:
        Current memory usage in MB
    """
    usage_mb = get_memory_usage_mb()
    msg = f"Memory: {usage_mb:.1f} MB"
    if prefix:
        msg = f"{prefix}: {msg}"
    print(msg)
    logger.info(msg)
    return usage_mb


# =============================================================================
# Explicit Memory Release
# =============================================================================

def release(*objects, verbose: bool = False, name: str = None) -> int:
    """
    Explicitly release objects and garbage collect.

    Args:
        *objects: Objects to delete
        verbose: If True, print what's being released
        name: Optional name for logging

    Returns:
        Number of objects collected by GC
    """
    if verbose:
        total_mb = 0
        for obj in objects:
            if obj is not None:
                size = get_size_mb(obj)
                total_mb += size
                obj_name = name or obj.__class__.__name__
                print(f"  Releasing {obj_name}: ~{size:.1f} MB")

        if total_mb > 0:
            print(f"  Total releasing: ~{total_mb:.1f} MB")

    # Delete references (caller must not use these after!)
    # Note: This doesn't actually delete - caller must do `del obj` or reassign
    # This function triggers gc.collect()

    collected = gc.collect()

    if verbose and collected > 0:
        print(f"  GC collected {collected} objects")

    return collected


def force_gc(verbose: bool = False) -> int:
    """
    Force garbage collection.

    Args:
        verbose: If True, print collection stats

    Returns:
        Number of objects collected
    """
    collected = gc.collect()
    if verbose:
        print(f"GC collected {collected} objects")
        memory_report("After GC")
    return collected


# =============================================================================
# Memory Guard Decorator
# =============================================================================

def memory_guard(max_mb: float = 8000, warn_only: bool = True):
    """
    Decorator to monitor function memory usage.

    Args:
        max_mb: Maximum allowed memory increase in MB
        warn_only: If False, raises MemoryError on exceed

    Usage:
        @memory_guard(max_mb=2000)
        def process_large_data():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            before = get_memory_usage_mb()

            try:
                result = func(*args, **kwargs)
            finally:
                after = get_memory_usage_mb()
                delta = after - before

                if delta > max_mb:
                    msg = f"{func.__name__} used {delta:.0f} MB (limit: {max_mb} MB)"
                    logger.warning(msg)
                    print(f"WARNING: {msg}")

                    if not warn_only:
                        raise MemoryError(msg)

            return result
        return wrapper
    return decorator


# =============================================================================
# Context Manager for Memory Tracking
# =============================================================================

class MemoryTracker:
    """
    Context manager to track memory usage in a block.

    Usage:
        with MemoryTracker("Processing signals") as tracker:
            do_stuff()
        print(f"Used {tracker.delta_mb:.1f} MB")
    """

    def __init__(self, name: str = "", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_mb = 0.0
        self.end_mb = 0.0
        self.delta_mb = 0.0

    def __enter__(self):
        self.start_mb = get_memory_usage_mb()
        if self.verbose and self.name:
            print(f"[{self.name}] Starting: {self.start_mb:.1f} MB")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_mb = get_memory_usage_mb()
        self.delta_mb = self.end_mb - self.start_mb

        if self.verbose:
            sign = "+" if self.delta_mb >= 0 else ""
            if self.name:
                print(f"[{self.name}] Finished: {self.end_mb:.1f} MB ({sign}{self.delta_mb:.1f} MB)")
            else:
                print(f"Memory delta: {sign}{self.delta_mb:.1f} MB")

        return False  # Don't suppress exceptions


# =============================================================================
# Batch Processing Helper
# =============================================================================

def batch_iterator(items: List, batch_size: int = 500, verbose: bool = False):
    """
    Iterate over items in batches with memory tracking.

    Args:
        items: List of items to process
        batch_size: Number of items per batch
        verbose: If True, print progress and memory

    Yields:
        (batch_num, batch_items) tuples
    """
    n_items = len(items)
    n_batches = (n_items + batch_size - 1) // batch_size

    for batch_num in range(n_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, n_items)
        batch = items[start_idx:end_idx]

        if verbose:
            memory = get_memory_usage_mb()
            print(f"Batch {batch_num + 1}/{n_batches} "
                  f"[{start_idx}:{end_idx}] - Memory: {memory:.1f} MB")

        yield batch_num, batch

        # Force GC between batches
        gc.collect()


# =============================================================================
# Polars Memory-Efficient Patterns
# =============================================================================

def read_parquet_lazy(path: str) -> 'pl.LazyFrame':
    """
    Read parquet lazily (no immediate memory allocation).

    Use .collect() only when you need the data.
    """
    if not HAS_POLARS:
        raise ImportError("Polars required")
    return pl.scan_parquet(path)


def write_and_release(
    df: 'pl.DataFrame',
    path: str,
    verbose: bool = False
) -> str:
    """
    Write DataFrame to parquet and release memory.

    Args:
        df: DataFrame to write
        path: Output path
        verbose: If True, print memory info

    Returns:
        Output path
    """
    if not HAS_POLARS:
        raise ImportError("Polars required")

    if verbose:
        size = get_size_mb(df)
        print(f"Writing {len(df)} rows (~{size:.1f} MB) to {path}")

    df.write_parquet(path)

    # Caller should do: del df; gc.collect()
    # We can't delete their reference

    if verbose:
        print(f"Wrote {path}")

    return path


def append_parquet(
    df: 'pl.DataFrame',
    path: str,
    verbose: bool = False
) -> str:
    """
    Append DataFrame to existing parquet file.

    Memory-efficient: reads existing, concatenates, writes, releases.

    Args:
        df: DataFrame to append
        path: Parquet file path
        verbose: If True, print progress

    Returns:
        Output path
    """
    if not HAS_POLARS:
        raise ImportError("Polars required")

    import os

    if os.path.exists(path):
        # Read existing
        existing = pl.read_parquet(path)
        if verbose:
            print(f"Appending {len(df)} rows to existing {len(existing)} rows")

        # Concatenate
        combined = pl.concat([existing, df], how='diagonal')

        # Release existing
        del existing
        gc.collect()

        # Write
        combined.write_parquet(path)

        # Release combined
        del combined
        gc.collect()
    else:
        # First write
        df.write_parquet(path)

    if verbose:
        print(f"Wrote {path}")

    return path


# =============================================================================
# Summary Report
# =============================================================================

def memory_summary() -> dict:
    """
    Get memory summary for logging/debugging.

    Returns:
        Dict with memory stats
    """
    current = get_memory_usage_mb()

    summary = {
        'current_mb': current,
        'platform': platform.system(),
        'python_version': platform.python_version(),
    }

    # Try to get more details
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        summary.update({
            'rss_mb': mem_info.rss / 1e6,
            'vms_mb': mem_info.vms / 1e6,
            'percent': process.memory_percent(),
        })
    except ImportError:
        pass

    return summary


def print_memory_summary():
    """Print formatted memory summary."""
    summary = memory_summary()
    print("\n" + "=" * 50)
    print("MEMORY SUMMARY")
    print("=" * 50)
    print(f"Current usage: {summary['current_mb']:.1f} MB")
    if 'percent' in summary:
        print(f"System percent: {summary['percent']:.1f}%")
    if 'rss_mb' in summary:
        print(f"RSS: {summary['rss_mb']:.1f} MB")
        print(f"VMS: {summary['vms_mb']:.1f} MB")
    print("=" * 50 + "\n")
