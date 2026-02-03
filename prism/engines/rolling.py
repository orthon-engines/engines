"""
Generic Rolling Engine.

ONE function to replace 16 rolling_*.py files.

Instead of:
    rolling_hurst.py, rolling_entropy.py, rolling_kurtosis.py, ...

We have:
    rolling.compute(statistics.compute, values, window, stride)

The manifest specifies which engines to run in rolling mode.
This module applies any engine function over rolling windows.
"""

import numpy as np
from typing import Dict, Any, Callable, Optional, List


def compute(
    engine_fn: Callable[[np.ndarray], Dict[str, float]],
    values: np.ndarray,
    window: int,
    stride: int,
    engine_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """
    Apply any engine function over rolling windows.
    
    Args:
        engine_fn: Function with signature f(array) -> dict
        values: Signal values
        window: Window size
        stride: Step size between windows
        engine_params: Optional params to pass to engine_fn
        
    Returns:
        dict with rolling_{key} arrays for each output key
        
    Example:
        from prism.engines.signal import memory
        rolling_hurst = compute(memory.compute, values, window=200, stride=50)
        # Returns {'rolling_hurst': array, 'rolling_hurst_r2': array}
    """
    values = np.asarray(values).flatten()
    n = len(values)
    engine_params = engine_params or {}
    
    if n < window:
        # Not enough data - run engine once to get output keys
        sample = engine_fn(values, **engine_params) if engine_params else engine_fn(values)
        return {f'rolling_{k}': np.full(n, np.nan) for k in sample.keys()}
    
    # Get output keys from first computation
    first_chunk = values[:window]
    sample_output = engine_fn(first_chunk, **engine_params) if engine_params else engine_fn(first_chunk)
    output_keys = list(sample_output.keys())
    
    # Initialize result arrays
    results = {f'rolling_{k}': np.full(n, np.nan) for k in output_keys}
    
    # Compute over rolling windows
    for i in range(0, n - window + 1, stride):
        chunk = values[i:i + window]
        output = engine_fn(chunk, **engine_params) if engine_params else engine_fn(chunk)
        
        # Place results at window end
        idx = i + window - 1
        for key, val in output.items():
            results[f'rolling_{key}'][idx] = val
    
    return results


def compute_multi(
    engines: Dict[str, Callable],
    values: np.ndarray,
    window: int,
    stride: int,
) -> Dict[str, np.ndarray]:
    """
    Apply multiple engines over rolling windows efficiently.
    
    Args:
        engines: Dict of name -> engine_fn
        values: Signal values
        window: Window size
        stride: Step size
        
    Returns:
        Combined dict with all rolling outputs
    """
    values = np.asarray(values).flatten()
    n = len(values)
    
    if n < window:
        results = {}
        for name, fn in engines.items():
            sample = fn(values)
            for k in sample.keys():
                results[f'rolling_{k}'] = np.full(n, np.nan)
        return results
    
    # Initialize from first window
    first_chunk = values[:window]
    results = {}
    for name, fn in engines.items():
        sample = fn(first_chunk)
        for k in sample.keys():
            results[f'rolling_{k}'] = np.full(n, np.nan)
    
    # Compute all engines per window (more cache-friendly)
    for i in range(0, n - window + 1, stride):
        chunk = values[i:i + window]
        idx = i + window - 1
        
        for name, fn in engines.items():
            output = fn(chunk)
            for key, val in output.items():
                results[f'rolling_{key}'][idx] = val
    
    return results


# ============================================================
# CONVENIENCE WRAPPERS (for backwards compatibility)
# ============================================================

def rolling_statistics(values: np.ndarray, window: int, stride: int) -> Dict[str, np.ndarray]:
    """Rolling statistics (kurtosis, skewness, crest_factor)."""
    from prism.engines.signal.statistics import compute as stats_compute
    return compute(stats_compute, values, window, stride)


def rolling_memory(values: np.ndarray, window: int, stride: int) -> Dict[str, np.ndarray]:
    """Rolling memory (hurst, hurst_r2)."""
    from prism.engines.signal.memory import compute as memory_compute
    return compute(memory_compute, values, window, stride)


def rolling_complexity(values: np.ndarray, window: int, stride: int) -> Dict[str, np.ndarray]:
    """Rolling complexity (entropies)."""
    from prism.engines.signal.complexity import compute as complexity_compute
    return compute(complexity_compute, values, window, stride)


def rolling_spectral(values: np.ndarray, window: int, stride: int, fs: float = 1.0) -> Dict[str, np.ndarray]:
    """Rolling spectral analysis."""
    from prism.engines.signal.spectral import compute as spectral_compute
    return compute(spectral_compute, values, window, stride, {'fs': fs})


def rolling_trend(values: np.ndarray, window: int, stride: int) -> Dict[str, np.ndarray]:
    """Rolling trend analysis."""
    from prism.engines.signal.trend import compute as trend_compute
    return compute(trend_compute, values, window, stride)


# Derivatives are computed via prism.primitives.individual.calculus.derivative()
# Do not duplicate compute logic here - rolling.py is a wrapper pattern only.
