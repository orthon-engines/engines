"""
Spectral Features
=================

Frequency domain characteristics:
    - Centroid: Center of mass of power spectrum
    - Bandwidth: Spread around centroid
    - Rolloff: Frequency below which 85% of energy lies
    - Low/High ratio: Energy distribution

These distinguish:
    - Narrowband: Dominant frequency (periodic)
    - Broadband: Energy spread (noise-like)
    - 1/f: Power-law spectrum (complex)

Supports three computation modes:
    - static: Entire signal → single value
    - windowed: Rolling windows → time series
    - point: At time t → single value
"""

import numpy as np
from scipy.fft import fft, fftfreq
from typing import Dict, Any, Optional


def compute(
    series: np.ndarray,
    mode: str = 'static',
    t: Optional[int] = None,
    window_size: int = 200,
    step_size: int = 20,
) -> Dict[str, Any]:
    """
    Compute spectral features.

    Args:
        series: 1D numpy array of observations
        mode: 'static', 'windowed', or 'point'
        t: Time index for point mode
        window_size: Window size for windowed/point modes
        step_size: Step between windows for windowed mode

    Returns:
        mode='static': {'centroid': float, 'bandwidth': float, ...}
        mode='windowed': {'centroid': array, 'bandwidth': array, 't': array, ...}
        mode='point': {'centroid': float, 'bandwidth': float, 't': int, ...}
    """
    series = np.asarray(series).flatten()

    if mode == 'static':
        return _compute_static(series)
    elif mode == 'windowed':
        return _compute_windowed(series, window_size, step_size)
    elif mode == 'point':
        return _compute_point(series, t, window_size)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'static', 'windowed', or 'point'.")


def _compute_static(series: np.ndarray) -> Dict[str, Any]:
    """Compute spectral features on entire signal."""
    n = len(series)

    if n < 16:
        return {
            'centroid': 0.25,
            'bandwidth': 0.1,
            'low_high_ratio': 1.0,
            'rolloff': 0.25
        }

    # FFT
    fft_vals = fft(series - np.mean(series))
    power = np.abs(fft_vals[:n//2]) ** 2
    freqs = fftfreq(n)[:n//2]

    # Exclude DC
    power = power[1:]
    freqs = freqs[1:]

    if len(freqs) == 0 or np.sum(power) < 1e-10:
        return {
            'centroid': 0.25,
            'bandwidth': 0.1,
            'low_high_ratio': 1.0,
            'rolloff': 0.25
        }

    # Normalize power
    power = power / np.sum(power)

    # Spectral centroid (center of mass)
    centroid = np.sum(freqs * power)

    # Spectral bandwidth (spread around centroid)
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * power))

    # Low/high frequency ratio (split at 0.1 Nyquist)
    low_mask = freqs < 0.1
    high_mask = freqs >= 0.1

    low_power = np.sum(power[low_mask]) if np.any(low_mask) else 0
    high_power = np.sum(power[high_mask]) if np.any(high_mask) else 1e-10

    low_high_ratio = low_power / high_power

    # Rolloff frequency (85% energy)
    cumsum = np.cumsum(power)
    rolloff_idx = np.searchsorted(cumsum, 0.85)
    rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]

    return {
        'centroid': float(centroid),
        'bandwidth': float(bandwidth),
        'low_high_ratio': float(low_high_ratio),
        'rolloff': float(rolloff)
    }


def _compute_windowed(
    series: np.ndarray,
    window_size: int,
    step_size: int,
) -> Dict[str, Any]:
    """Compute spectral features over rolling windows."""
    n = len(series)

    if n < window_size:
        return {
            'centroid': np.array([]),
            'bandwidth': np.array([]),
            'low_high_ratio': np.array([]),
            'rolloff': np.array([]),
            't': np.array([]),
            'window_size': window_size,
            'step_size': step_size,
        }

    t_values = []
    centroid_values = []
    bandwidth_values = []
    low_high_ratio_values = []
    rolloff_values = []

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        window = series[start:end]

        result = _compute_static(window)

        t_values.append(start + window_size // 2)
        centroid_values.append(result['centroid'])
        bandwidth_values.append(result['bandwidth'])
        low_high_ratio_values.append(result['low_high_ratio'])
        rolloff_values.append(result['rolloff'])

    return {
        'centroid': np.array(centroid_values),
        'bandwidth': np.array(bandwidth_values),
        'low_high_ratio': np.array(low_high_ratio_values),
        'rolloff': np.array(rolloff_values),
        't': np.array(t_values),
        'window_size': window_size,
        'step_size': step_size,
    }


def _compute_point(
    series: np.ndarray,
    t: int,
    window_size: int,
) -> Dict[str, Any]:
    """Compute spectral features at specific time t."""
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

    if len(window) < 16:
        return {
            'centroid': 0.25,
            'bandwidth': 0.1,
            'low_high_ratio': 1.0,
            'rolloff': 0.25,
            't': t,
            'window_start': start,
            'window_end': end,
        }

    result = _compute_static(window)
    result['t'] = t
    result['window_start'] = start
    result['window_end'] = end

    return result
