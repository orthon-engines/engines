"""
Correlation Engine.

Computes pairwise correlation measures (symmetric):
- Pearson correlation
- Spearman correlation
- Cross-correlation
- Mutual information

Thin wrapper over primitives/pairwise/correlation.py.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple

from engines.primitives.pairwise.correlation import (
    correlation as _correlation,
    cross_correlation as _cross_correlation,
    lag_at_max_xcorr as _lag_at_max_xcorr,
)
from engines.primitives.pairwise.information import (
    mutual_information as _mutual_info,
)


def compute(
    x: np.ndarray,
    y: np.ndarray,
    method: str = 'pearson',
) -> Dict[str, float]:
    """
    Compute correlation between two signals.
    
    Args:
        x, y: Signal values (must be same length)
        method: 'pearson' or 'spearman'
        
    Returns:
        dict with correlation, correlation_abs
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # Align lengths
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    
    # Remove NaN pairs
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    
    if len(x) < 3:
        return {
            'correlation': np.nan,
            'correlation_abs': np.nan,
        }
    
    corr = _correlation(x, y, method=method)
    
    return {
        'correlation': float(corr),
        'correlation_abs': float(abs(corr)),
    }


def compute_cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 50,
) -> Dict[str, Any]:
    """
    Compute cross-correlation and lag at maximum.
    
    Args:
        x, y: Signal values
        max_lag: Maximum lag to compute
        
    Returns:
        dict with max_xcorr, lag_at_max, xcorr_symmetric
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    
    if len(x) < max_lag + 10:
        return {
            'max_xcorr': np.nan,
            'lag_at_max': np.nan,
            'xcorr_symmetric': np.nan,
        }
    
    # Full cross-correlation
    xcorr = _cross_correlation(x, y, max_lag=max_lag)
    
    # Find maximum
    center = len(xcorr) // 2
    max_idx = np.argmax(np.abs(xcorr))
    max_xcorr = xcorr[max_idx]
    lag_at_max = max_idx - center
    
    # Symmetry measure (xcorr at +lag vs -lag)
    if center > 0:
        left = xcorr[:center]
        right = xcorr[center+1:][::-1]
        min_len = min(len(left), len(right))
        if min_len > 0:
            xcorr_symmetric = float(np.corrcoef(left[:min_len], right[:min_len])[0, 1])
        else:
            xcorr_symmetric = np.nan
    else:
        xcorr_symmetric = np.nan
    
    return {
        'max_xcorr': float(max_xcorr),
        'lag_at_max': int(lag_at_max),
        'xcorr_symmetric': xcorr_symmetric,
    }


def compute_mutual_info(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute mutual information between two signals.
    
    Args:
        x, y: Signal values
        n_bins: Number of bins for discretization
        
    Returns:
        dict with mutual_info, normalized_mi
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    
    if len(x) < n_bins * 2:
        return {
            'mutual_info': np.nan,
            'normalized_mi': np.nan,
        }
    
    mi = _mutual_info(x, y, n_bins=n_bins)
    
    # Normalize by entropy
    # MI / sqrt(H(X) * H(Y)) for normalized measure
    hx = _entropy_1d(x, n_bins)
    hy = _entropy_1d(y, n_bins)
    
    if hx > 0 and hy > 0:
        normalized = mi / np.sqrt(hx * hy)
    else:
        normalized = 0.0
    
    return {
        'mutual_info': float(mi),
        'normalized_mi': float(normalized),
    }


def _entropy_1d(x: np.ndarray, n_bins: int) -> float:
    """Compute entropy of 1D distribution."""
    counts, _ = np.histogram(x, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))


def compute_all(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 50,
) -> Dict[str, Any]:
    """
    Compute all correlation measures.
    
    Args:
        x, y: Signal values
        max_lag: Maximum lag for cross-correlation
        
    Returns:
        Combined dict with all measures
    """
    result = {}
    result.update(compute(x, y, method='pearson'))
    result['spearman'] = compute(x, y, method='spearman')['correlation']
    result.update(compute_cross_correlation(x, y, max_lag=max_lag))
    result.update(compute_mutual_info(x, y))
    return result
