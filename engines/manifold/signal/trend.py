"""
Trend Engine.

Computes trend analysis measures:
- Linear trend slope and R²
- Detrended residual std
- CUSUM range

Thin wrapper over primitives/individual/stationarity.py.
"""

import numpy as np
from typing import Dict

from engines.primitives.individual.stationarity import (
    trend as _trend,
    mann_kendall_test,
)


def compute(y: np.ndarray) -> Dict[str, float]:
    """
    Compute trend properties.
    
    Args:
        y: Signal values
        
    Returns:
        dict with trend_slope, trend_r2, detrend_std, cusum_range
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)
    
    if n < 3:
        return {
            'trend_slope': np.nan,
            'trend_r2': np.nan,
            'detrend_std': np.nan,
            'cusum_range': np.nan,
        }
    
    # Linear fit
    x = np.arange(n)
    slope, intercept = np.polyfit(x, y, 1)
    
    # R² (coefficient of determination)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Detrended residual std
    residuals = y - y_pred
    detrend_std = float(np.std(residuals))
    
    # CUSUM range (cumulative sum of deviations)
    mean_y = np.mean(y)
    cusum = np.cumsum(y - mean_y)
    cusum_range = float(np.max(cusum) - np.min(cusum))
    
    # Normalize by std for scale invariance
    std_y = np.std(y)
    if std_y > 0:
        cusum_range_norm = cusum_range / std_y
    else:
        cusum_range_norm = 0.0
    
    return {
        'trend_slope': float(slope),
        'trend_r2': float(r2),
        'detrend_std': detrend_std,
        'cusum_range': cusum_range_norm,
    }


def compute_mann_kendall(y: np.ndarray) -> Dict[str, float]:
    """
    Non-parametric trend test.
    
    Returns:
        dict with mk_stat, mk_pvalue, mk_slope (Theil-Sen)
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    
    if len(y) < 10:
        return {
            'mk_stat': np.nan,
            'mk_pvalue': np.nan,
            'mk_slope': np.nan,
        }
    
    try:
        stat, pvalue, slope = mann_kendall_test(y)
        return {
            'mk_stat': float(stat),
            'mk_pvalue': float(pvalue),
            'mk_slope': float(slope),
        }
    except Exception:
        return {
            'mk_stat': np.nan,
            'mk_pvalue': np.nan,
            'mk_slope': np.nan,
        }


def compute_rate_of_change(y: np.ndarray) -> Dict[str, float]:
    """
    Compute rate of change statistics.
    
    Returns:
        dict with mean_roc, std_roc, max_roc (all normalized)
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    
    if len(y) < 3:
        return {
            'mean_roc': np.nan,
            'std_roc': np.nan,
            'max_roc': np.nan,
        }
    
    # First difference
    diff = np.diff(y)
    
    # Normalize by signal std for scale invariance
    std_y = np.std(y)
    if std_y > 0:
        diff_norm = diff / std_y
    else:
        diff_norm = diff
    
    return {
        'mean_roc': float(np.mean(np.abs(diff_norm))),
        'std_roc': float(np.std(diff_norm)),
        'max_roc': float(np.max(np.abs(diff_norm))),
    }
