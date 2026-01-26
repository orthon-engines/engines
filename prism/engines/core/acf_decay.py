"""
Autocorrelation Decay Analysis
==============================

Determines if ACF decays exponentially (short-range dependence)
or as a power-law (long-range dependence).

Power-law decay indicates long memory and is associated with
Hurst exponent > 0.5.

Supports three computation modes:
    - static: Entire signal → single value
    - windowed: Rolling windows → time series
    - point: At time t → single value
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional


def compute(
    series: np.ndarray,
    mode: str = 'static',
    t: Optional[int] = None,
    window_size: int = 200,
    step_size: int = 20,
    max_lag: int = 50,
) -> Dict[str, Any]:
    """
    Determine autocorrelation decay type.

    Args:
        series: 1D numpy array of observations
        mode: 'static', 'windowed', or 'point'
        t: Time index for point mode
        window_size: Window size for windowed/point modes
        step_size: Step between windows for windowed mode
        max_lag: Maximum lag to consider for ACF

    Returns:
        mode='static': {'decay_type': str, 'half_life': float, ...}
        mode='windowed': {'decay_type': array, 'half_life': array, 't': array, ...}
        mode='point': {'decay_type': str, 'half_life': float, 't': int, ...}
    """
    series = np.asarray(series).flatten()

    if mode == 'static':
        return _compute_static(series, max_lag)
    elif mode == 'windowed':
        return _compute_windowed(series, window_size, step_size, max_lag)
    elif mode == 'point':
        return _compute_point(series, t, window_size, max_lag)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'static', 'windowed', or 'point'.")


def _compute_static(series: np.ndarray, max_lag: int = 50) -> Dict[str, Any]:
    """Compute ACF decay on entire signal."""
    n = len(series)
    max_lag = min(max_lag, n // 3)

    if max_lag < 5:
        return {
            'decay_type': 'exponential',
            'half_life': 1.0,
            'exponential_r2': 0.0,
            'power_law_r2': 0.0
        }

    # Compute ACF
    centered = series - np.mean(series)
    acf = np.correlate(centered, centered, mode='full')
    acf = acf[n-1:n-1+max_lag+1]
    acf = acf / acf[0] if acf[0] != 0 else acf

    lags = np.arange(1, len(acf))
    acf_values = np.abs(acf[1:])

    # Filter out zeros/negatives for log
    valid = acf_values > 0.01
    if np.sum(valid) < 3:
        return {
            'decay_type': 'exponential',
            'half_life': 1.0,
            'exponential_r2': 0.0,
            'power_law_r2': 0.0
        }

    lags_valid = lags[valid]
    acf_valid = acf_values[valid]

    # Fit exponential: log(ACF) = -λ * lag
    log_acf = np.log(acf_valid)
    exp_slope, _, exp_r, _, _ = stats.linregress(lags_valid, log_acf)
    exp_r2 = exp_r ** 2

    # Fit power law: log(ACF) = -α * log(lag)
    log_lags = np.log(lags_valid)
    _, _, pow_r, _, _ = stats.linregress(log_lags, log_acf)
    pow_r2 = pow_r ** 2

    # Half-life from exponential fit
    if exp_slope < 0:
        half_life = -np.log(2) / exp_slope
    else:
        half_life = float(max_lag)

    # Better fit wins (power law needs to be notably better)
    if pow_r2 > exp_r2 + 0.05:
        decay_type = 'power_law'
    else:
        decay_type = 'exponential'

    return {
        'decay_type': decay_type,
        'half_life': float(half_life),
        'exponential_r2': float(exp_r2),
        'power_law_r2': float(pow_r2)
    }


def _compute_windowed(
    series: np.ndarray,
    window_size: int,
    step_size: int,
    max_lag: int = 50,
) -> Dict[str, Any]:
    """Compute ACF decay over rolling windows."""
    n = len(series)

    if n < window_size:
        return {
            'decay_type': np.array([]),
            'half_life': np.array([]),
            'exponential_r2': np.array([]),
            'power_law_r2': np.array([]),
            't': np.array([]),
            'window_size': window_size,
            'step_size': step_size,
        }

    t_values = []
    decay_types = []
    half_life_values = []
    exp_r2_values = []
    pow_r2_values = []

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        window = series[start:end]

        result = _compute_static(window, max_lag)

        t_values.append(start + window_size // 2)
        decay_types.append(result['decay_type'])
        half_life_values.append(result['half_life'])
        exp_r2_values.append(result['exponential_r2'])
        pow_r2_values.append(result['power_law_r2'])

    return {
        'decay_type': np.array(decay_types),
        'half_life': np.array(half_life_values),
        'exponential_r2': np.array(exp_r2_values),
        'power_law_r2': np.array(pow_r2_values),
        't': np.array(t_values),
        'window_size': window_size,
        'step_size': step_size,
    }


def _compute_point(
    series: np.ndarray,
    t: int,
    window_size: int,
    max_lag: int = 50,
) -> Dict[str, Any]:
    """Compute ACF decay at specific time t."""
    if t is None:
        raise ValueError("t is required for point mode")

    n = len(series)

    # Center window on t
    half_window = window_size // 2
    start = max(0, t - half_window)
    end = min(n, start + window_size)

    if end - start < window_size:
        start = max(0, end - window_size)

    window = series[start:end]

    if len(window) < 20:
        return {
            'decay_type': 'exponential',
            'half_life': 1.0,
            'exponential_r2': 0.0,
            'power_law_r2': 0.0,
            't': t,
            'window_start': start,
            'window_end': end,
        }

    result = _compute_static(window, max_lag)
    result['t'] = t
    result['window_start'] = start
    result['window_end'] = end

    return result
