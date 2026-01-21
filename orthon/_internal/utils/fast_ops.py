"""
PRISM Fast Operations Utility

Provides optimized operations using:
- bottleneck: Fast NaN handling and rolling windows
- numexpr: Fast element-wise math (for large arrays)

These functions automatically fall back to numpy/scipy if
optional dependencies are not installed.

Usage:
    from orthon._internal.utils.fast_ops import nanmean, nanstd, move_mean, drop_nan
"""

import numpy as np
from typing import Optional

# Try to import bottleneck for fast NaN operations
try:
    import bottleneck as bn
    HAS_BOTTLENECK = True
except ImportError:
    HAS_BOTTLENECK = False

# Try to import numexpr for fast element-wise math
try:
    import numexpr as ne
    HAS_NUMEXPR = True
except ImportError:
    HAS_NUMEXPR = False


# =============================================================================
# NaN-aware statistics (2-10x faster with bottleneck)
# =============================================================================

def nanmean(arr: np.ndarray, axis: Optional[int] = None) -> float:
    """Fast NaN-aware mean."""
    if HAS_BOTTLENECK:
        return bn.nanmean(arr, axis=axis)
    return np.nanmean(arr, axis=axis)


def nanstd(arr: np.ndarray, axis: Optional[int] = None, ddof: int = 0) -> float:
    """Fast NaN-aware standard deviation."""
    if HAS_BOTTLENECK:
        return bn.nanstd(arr, axis=axis, ddof=ddof)
    return np.nanstd(arr, axis=axis, ddof=ddof)


def nanvar(arr: np.ndarray, axis: Optional[int] = None, ddof: int = 0) -> float:
    """Fast NaN-aware variance."""
    if HAS_BOTTLENECK:
        return bn.nanvar(arr, axis=axis, ddof=ddof)
    return np.nanvar(arr, axis=axis, ddof=ddof)


def nanmin(arr: np.ndarray, axis: Optional[int] = None) -> float:
    """Fast NaN-aware minimum."""
    if HAS_BOTTLENECK:
        return bn.nanmin(arr, axis=axis)
    return np.nanmin(arr, axis=axis)


def nanmax(arr: np.ndarray, axis: Optional[int] = None) -> float:
    """Fast NaN-aware maximum."""
    if HAS_BOTTLENECK:
        return bn.nanmax(arr, axis=axis)
    return np.nanmax(arr, axis=axis)


def nansum(arr: np.ndarray, axis: Optional[int] = None) -> float:
    """Fast NaN-aware sum."""
    if HAS_BOTTLENECK:
        return bn.nansum(arr, axis=axis)
    return np.nansum(arr, axis=axis)


def nanmedian(arr: np.ndarray, axis: Optional[int] = None) -> float:
    """Fast NaN-aware median."""
    if HAS_BOTTLENECK:
        return bn.nanmedian(arr, axis=axis)
    return np.nanmedian(arr, axis=axis)


# =============================================================================
# NaN filtering (fast path)
# =============================================================================

def drop_nan(arr: np.ndarray) -> np.ndarray:
    """
    Remove NaN values from array (1D only).

    Equivalent to: arr[~np.isnan(arr)]
    """
    return arr[~np.isnan(arr)]


def count_nan(arr: np.ndarray) -> int:
    """Count NaN values in array."""
    if HAS_BOTTLENECK:
        return int(bn.nansum(np.isnan(arr)))
    return int(np.sum(np.isnan(arr)))


def any_nan(arr: np.ndarray) -> bool:
    """Check if array contains any NaN values."""
    if HAS_BOTTLENECK:
        return bn.anynan(arr)
    return np.any(np.isnan(arr))


def all_nan(arr: np.ndarray) -> bool:
    """Check if all values in array are NaN."""
    if HAS_BOTTLENECK:
        return bn.allnan(arr)
    return np.all(np.isnan(arr))


# =============================================================================
# Rolling windows (10-100x faster with bottleneck)
# =============================================================================

def move_mean(arr: np.ndarray, window: int, min_count: Optional[int] = None) -> np.ndarray:
    """
    Moving (rolling) mean.

    Args:
        arr: Input array
        window: Window size
        min_count: Minimum non-NaN values required (default: window)

    Returns:
        Array of rolling means (NaN at edges)
    """
    if HAS_BOTTLENECK:
        return bn.move_mean(arr, window, min_count=min_count)
    # Fallback using numpy convolution (slower)
    if min_count is None:
        min_count = window
    result = np.convolve(arr, np.ones(window) / window, mode='valid')
    # Pad with NaN to match input length
    padding = np.full(window - 1, np.nan)
    return np.concatenate([padding, result])


def move_std(arr: np.ndarray, window: int, min_count: Optional[int] = None, ddof: int = 0) -> np.ndarray:
    """
    Moving (rolling) standard deviation.

    Args:
        arr: Input array
        window: Window size
        min_count: Minimum non-NaN values required (default: window)
        ddof: Degrees of freedom

    Returns:
        Array of rolling stds (NaN at edges)
    """
    if HAS_BOTTLENECK:
        return bn.move_std(arr, window, min_count=min_count, ddof=ddof)
    # Fallback (slower)
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = np.std(arr[i - window + 1:i + 1], ddof=ddof)
    return result


def move_sum(arr: np.ndarray, window: int, min_count: Optional[int] = None) -> np.ndarray:
    """Moving (rolling) sum."""
    if HAS_BOTTLENECK:
        return bn.move_sum(arr, window, min_count=min_count)
    # Fallback using cumsum trick
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    result = cumsum[window:] - cumsum[:-window]
    padding = np.full(window - 1, np.nan)
    return np.concatenate([padding, result])


def move_min(arr: np.ndarray, window: int, min_count: Optional[int] = None) -> np.ndarray:
    """Moving (rolling) minimum."""
    if HAS_BOTTLENECK:
        return bn.move_min(arr, window, min_count=min_count)
    # Fallback (slower)
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = np.min(arr[i - window + 1:i + 1])
    return result


def move_max(arr: np.ndarray, window: int, min_count: Optional[int] = None) -> np.ndarray:
    """Moving (rolling) maximum."""
    if HAS_BOTTLENECK:
        return bn.move_max(arr, window, min_count=min_count)
    # Fallback (slower)
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = np.max(arr[i - window + 1:i + 1])
    return result


# =============================================================================
# Element-wise math (faster for large arrays with numexpr)
# =============================================================================

def fast_eval(expr: str, local_dict: dict) -> np.ndarray:
    """
    Evaluate element-wise expression.

    Uses numexpr for large arrays (>10000 elements), numpy otherwise.

    Args:
        expr: Expression string (e.g., "a * b + c")
        local_dict: Variable bindings (e.g., {"a": arr1, "b": arr2, "c": arr3})

    Returns:
        Result array
    """
    # Check if arrays are large enough to benefit from numexpr
    if HAS_NUMEXPR:
        for val in local_dict.values():
            if isinstance(val, np.ndarray) and val.size > 10000:
                return ne.evaluate(expr, local_dict=local_dict)
    # Fallback to eval (safe because we control the inputs)
    return eval(expr, {"np": np, "__builtins__": {}}, local_dict)


def squared_diff_sum(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute sum of squared differences: sum((a - b)^2)

    Optimized for distance calculations.
    """
    if HAS_NUMEXPR and a.size > 10000:
        return ne.evaluate("sum((a - b) ** 2)")
    return np.sum((a - b) ** 2)


# =============================================================================
# Utility info
# =============================================================================

def get_available_optimizations() -> dict:
    """Return info about available optimizations."""
    return {
        "bottleneck": HAS_BOTTLENECK,
        "numexpr": HAS_NUMEXPR,
    }
