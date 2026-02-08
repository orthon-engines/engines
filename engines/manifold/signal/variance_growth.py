"""
Variance Growth Engine.

Measures how variance grows with time scale.
Used for detecting non-stationarity and regime changes.
"""

import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    """
    Compute variance growth rate.

    Measures how variance of cumulative sum grows with sample size.
    For stationary processes, variance grows linearly (slope ~ 1).
    For trending/non-stationary, variance grows faster (slope > 1).

    Args:
        y: Signal values

    Returns:
        dict with variance_growth_rate, variance_ratio
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    if n < 20:
        return {
            'variance_growth_rate': np.nan,
            'variance_ratio': np.nan,
        }

    try:
        # Compute cumulative sum (de-meaned)
        y_centered = y - np.mean(y)
        cumsum = np.cumsum(y_centered)

        # Measure variance at different window sizes
        sizes = []
        variances = []

        for k in [n // 8, n // 4, n // 2, n]:
            if k < 4:
                continue
            # Variance of cumsum at this scale
            var_k = np.var(cumsum[:k])
            if var_k > 0:
                sizes.append(k)
                variances.append(var_k)

        if len(sizes) < 2:
            return {
                'variance_growth_rate': np.nan,
                'variance_ratio': np.nan,
            }

        # Fit log-log slope: var ~ n^alpha
        log_sizes = np.log(sizes)
        log_vars = np.log(variances)
        slope, _ = np.polyfit(log_sizes, log_vars, 1)

        # Variance ratio: var(full) / var(half) normalized by size ratio
        var_full = np.var(cumsum)
        var_half = np.var(cumsum[:n // 2])
        if var_half > 0:
            variance_ratio = (var_full / var_half) / 2.0  # Should be ~1 for stationary
        else:
            variance_ratio = np.nan

        return {
            'variance_growth_rate': float(slope),
            'variance_ratio': float(variance_ratio),
        }

    except Exception:
        return {
            'variance_growth_rate': np.nan,
            'variance_ratio': np.nan,
        }
