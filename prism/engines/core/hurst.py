"""
Hurst Exponent Computation
==========================

Long-range memory estimation via:
- Rescaled Range (R/S) analysis
- Detrended Fluctuation Analysis (DFA)

Interpretation:
    H < 0.5: Anti-persistent (mean-reverting)
    H = 0.5: Random walk (no memory)
    H > 0.5: Persistent (trending)

CANONICAL INTERFACE:
    def compute(observations: pd.DataFrame) -> pd.DataFrame
    Input:  [entity_id, signal_id, I, y]
    Output: [entity_id, signal_id, hurst, hurst_r2, hurst_method]
"""

ENGINE_TYPE = 'signal'

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any


def compute_rs(y: np.ndarray, min_window: int = 10) -> Dict[str, Any]:
    """
    Hurst exponent via Rescaled Range (R/S) analysis.
    
    Classical method. Best for stationary series.
    """
    n = len(y)
    
    if n < min_window * 2:
        return {'hurst': 0.5, 'r2': 0.0, 'method': 'rs'}
    
    # Window sizes (powers of 2)
    max_k = int(np.floor(np.log2(n / min_window)))
    if max_k < 2:
        return {'hurst': 0.5, 'r2': 0.0, 'method': 'rs'}
    
    window_sizes = [int(n / (2**k)) for k in range(max_k + 1)]
    window_sizes = [w for w in window_sizes if w >= min_window]
    
    rs_values = []
    
    for ws in window_sizes:
        n_windows = n // ws
        rs_list = []
        
        for i in range(n_windows):
            window = y[i * ws:(i + 1) * ws]
            mean = np.mean(window)
            cumsum = np.cumsum(window - mean)
            r = np.max(cumsum) - np.min(cumsum)
            s = np.std(window, ddof=1)
            
            if s > 0:
                rs_list.append(r / s)
        
        if rs_list:
            rs_values.append((ws, np.mean(rs_list)))
    
    if len(rs_values) < 2:
        return {'hurst': 0.5, 'r2': 0.0, 'method': 'rs'}
    
    # Log-log regression
    log_n = np.log([x[0] for x in rs_values])
    log_rs = np.log([x[1] for x in rs_values])
    slope, _, r_value, _, _ = stats.linregress(log_n, log_rs)
    
    return {
        'hurst': float(np.clip(slope, 0, 1)),
        'r2': float(r_value ** 2),
        'method': 'rs'
    }


def compute_dfa(y: np.ndarray, min_window: int = 10) -> Dict[str, Any]:
    """
    Hurst exponent via Detrended Fluctuation Analysis (DFA).
    
    More robust to non-stationarity than R/S.
    """
    n = len(y)
    
    if n < min_window * 4:
        return {'hurst': 0.5, 'r2': 0.0, 'method': 'dfa'}
    
    # Integrate (cumsum of deviations from mean)
    integrated = np.cumsum(y - np.mean(y))
    
    # Log-spaced window sizes
    max_window = n // 4
    window_sizes = []
    w = min_window
    while w <= max_window:
        window_sizes.append(w)
        w = int(w * 1.5)
    
    if len(window_sizes) < 3:
        return {'hurst': 0.5, 'r2': 0.0, 'method': 'dfa'}
    
    fluctuations = []
    
    for ws in window_sizes:
        n_windows = n // ws
        f2_list = []
        
        for i in range(n_windows):
            segment = integrated[i * ws:(i + 1) * ws]
            x = np.arange(ws)
            slope, intercept = np.polyfit(x, segment, 1)
            trend = slope * x + intercept
            f2 = np.mean((segment - trend) ** 2)
            f2_list.append(f2)
        
        if f2_list:
            fluctuations.append((ws, np.sqrt(np.mean(f2_list))))
    
    if len(fluctuations) < 3:
        return {'hurst': 0.5, 'r2': 0.0, 'method': 'dfa'}
    
    log_n = np.log([x[0] for x in fluctuations])
    log_f = np.log([x[1] for x in fluctuations])
    slope, _, r_value, _, _ = stats.linregress(log_n, log_f)
    
    return {
        'hurst': float(np.clip(slope, 0, 1)),
        'r2': float(r_value ** 2),
        'method': 'dfa'
    }


def _compute_array(y: np.ndarray, method: str = 'dfa') -> Dict[str, Any]:
    """Internal: compute Hurst from numpy array."""
    y = np.asarray(y).flatten()

    if method == 'dfa':
        return compute_dfa(y)
    elif method == 'rs':
        return compute_rs(y)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute(observations: pd.DataFrame, method: str = 'dfa') -> pd.DataFrame:
    """
    Compute Hurst exponent.

    CANONICAL INTERFACE:
        Input:  observations [entity_id, signal_id, I, y]
        Output: primitives [entity_id, signal_id, hurst, hurst_r2, hurst_method]

    Args:
        observations: DataFrame with columns [entity_id, signal_id, I, y]
        method: 'dfa' (default, robust) or 'rs' (classical)

    Returns:
        DataFrame with Hurst exponent per entity/signal
    """
    results = []

    for (entity_id, signal_id), group in observations.groupby(['entity_id', 'signal_id']):
        y = group.sort_values('I')['y'].values

        try:
            result = _compute_array(y, method=method)
            results.append({
                'entity_id': entity_id,
                'signal_id': signal_id,
                'hurst': result['hurst'],
                'hurst_r2': result['r2'],
                'hurst_method': result['method'],
            })
        except Exception:
            results.append({
                'entity_id': entity_id,
                'signal_id': signal_id,
                'hurst': np.nan,
                'hurst_r2': np.nan,
                'hurst_method': method,
            })

    return pd.DataFrame(results)
