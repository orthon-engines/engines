"""
Wavelet Multi-Scale Decomposition
=================================

Decomposes signal into time-frequency components at multiple scales.

Key outputs:
    - energy_by_scale: How energy is distributed across scales
    - dominant_scale: Scale with maximum energy
    - scale_entropy: Uniformity of energy distribution

A shift in dominant scale is an EARLY WARNING signal
of regime change.

Supports three computation modes:
    - static: Entire signal → single value
    - windowed: Rolling windows → time series
    - point: At time t → single value

Dependencies: pywt (PyWavelets) - falls back to scipy.signal if unavailable
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional


def compute(
    series: np.ndarray,
    mode: str = 'static',
    t: Optional[int] = None,
    window_size: int = 200,
    step_size: int = 20,
    wavelet: str = 'db4',
    max_level: int = None,
) -> Dict[str, Any]:
    """
    Compute wavelet decomposition features.

    Args:
        series: 1D numpy array of observations
        mode: 'static', 'windowed', or 'point'
        t: Time index for point mode
        window_size: Window size for windowed/point modes
        step_size: Step between windows for windowed mode
        wavelet: Wavelet type (default: Daubechies-4)
        max_level: Maximum decomposition level

    Returns:
        mode='static': {'dominant_scale': int, 'scale_entropy': float, ...}
        mode='windowed': {'dominant_scale': array, 'scale_entropy': array, 't': array, ...}
        mode='point': {'dominant_scale': int, 'scale_entropy': float, 't': int, ...}
    """
    series = np.asarray(series).flatten()

    if mode == 'static':
        return _compute_static(series, wavelet, max_level)
    elif mode == 'windowed':
        return _compute_windowed(series, window_size, step_size, wavelet, max_level)
    elif mode == 'point':
        return _compute_point(series, t, window_size, wavelet, max_level)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'static', 'windowed', or 'point'.")


def _compute_static(
    series: np.ndarray,
    wavelet: str = 'db4',
    max_level: int = None,
) -> Dict[str, Any]:
    """Compute wavelet features on entire signal."""
    n = len(series)

    if n < 16:
        return _empty_result()

    # Try pywt first, fall back to scipy
    try:
        details, approx, energy_by_scale = _compute_pywt(series, wavelet, max_level)
    except ImportError:
        details, approx, energy_by_scale = _compute_scipy_fallback(series)

    if not energy_by_scale:
        return _empty_result()

    # Dominant scale (index of max energy)
    dominant_scale = int(np.argmax(energy_by_scale))

    # Scale entropy
    energy_arr = np.array(energy_by_scale)
    energy_arr = energy_arr[energy_arr > 0]
    if len(energy_arr) > 0:
        scale_entropy = -np.sum(energy_arr * np.log(energy_arr + 1e-10))
    else:
        scale_entropy = 0.0

    # Energy ratio: low frequencies vs high frequencies
    mid = len(energy_by_scale) // 2
    low_energy = sum(energy_by_scale[:mid]) if mid > 0 else 0
    high_energy = sum(energy_by_scale[mid:]) if mid < len(energy_by_scale) else 1e-10
    energy_ratio = low_energy / (high_energy + 1e-10)

    # Detail statistics (from finest scale if available)
    detail_mean = 0.0
    detail_std = 0.0
    detail_kurtosis = 0.0
    if details and len(details) > 0:
        finest = details[-1]
        detail_mean = float(np.mean(np.abs(finest)))
        detail_std = float(np.std(finest))
        if detail_std > 0:
            detail_kurtosis = float(stats.kurtosis(finest))

    # Approximation (trend) statistics
    approx_slope = 0.0
    approx_curvature = 0.0
    if len(approx) > 2:
        x = np.arange(len(approx))
        coeffs = np.polyfit(x, approx, 2)
        approx_curvature = float(coeffs[0])
        approx_slope = float(coeffs[1])

    return {
        'energy_by_scale': energy_by_scale,
        'dominant_scale': dominant_scale,
        'scale_entropy': float(scale_entropy),
        'energy_ratio_low_high': float(energy_ratio),
        'detail_mean': detail_mean,
        'detail_std': detail_std,
        'detail_kurtosis': detail_kurtosis,
        'approx_slope': approx_slope,
        'approx_curvature': approx_curvature
    }


def _compute_windowed(
    series: np.ndarray,
    window_size: int,
    step_size: int,
    wavelet: str = 'db4',
    max_level: int = None,
) -> Dict[str, Any]:
    """Compute wavelet features over rolling windows."""
    n = len(series)

    if n < window_size:
        return {
            'dominant_scale': np.array([]),
            'scale_entropy': np.array([]),
            'energy_ratio_low_high': np.array([]),
            'detail_mean': np.array([]),
            'detail_std': np.array([]),
            'detail_kurtosis': np.array([]),
            'approx_slope': np.array([]),
            'approx_curvature': np.array([]),
            't': np.array([]),
            'window_size': window_size,
            'step_size': step_size,
        }

    t_values = []
    dominant_scale_values = []
    scale_entropy_values = []
    energy_ratio_values = []
    detail_mean_values = []
    detail_std_values = []
    detail_kurtosis_values = []
    approx_slope_values = []
    approx_curvature_values = []

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        window = series[start:end]

        result = _compute_static(window, wavelet, max_level)

        t_values.append(start + window_size // 2)
        dominant_scale_values.append(result['dominant_scale'])
        scale_entropy_values.append(result['scale_entropy'])
        energy_ratio_values.append(result['energy_ratio_low_high'])
        detail_mean_values.append(result['detail_mean'])
        detail_std_values.append(result['detail_std'])
        detail_kurtosis_values.append(result['detail_kurtosis'])
        approx_slope_values.append(result['approx_slope'])
        approx_curvature_values.append(result['approx_curvature'])

    return {
        'dominant_scale': np.array(dominant_scale_values),
        'scale_entropy': np.array(scale_entropy_values),
        'energy_ratio_low_high': np.array(energy_ratio_values),
        'detail_mean': np.array(detail_mean_values),
        'detail_std': np.array(detail_std_values),
        'detail_kurtosis': np.array(detail_kurtosis_values),
        'approx_slope': np.array(approx_slope_values),
        'approx_curvature': np.array(approx_curvature_values),
        't': np.array(t_values),
        'window_size': window_size,
        'step_size': step_size,
    }


def _compute_point(
    series: np.ndarray,
    t: int,
    window_size: int,
    wavelet: str = 'db4',
    max_level: int = None,
) -> Dict[str, Any]:
    """Compute wavelet features at specific time t."""
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
        result = _empty_result()
        result['t'] = t
        result['window_start'] = start
        result['window_end'] = end
        return result

    result = _compute_static(window, wavelet, max_level)
    result['t'] = t
    result['window_start'] = start
    result['window_end'] = end

    return result


def _compute_pywt(
    series: np.ndarray,
    wavelet: str,
    max_level: int
) -> Tuple[List[np.ndarray], np.ndarray, List[float]]:
    """Compute using PyWavelets."""
    import pywt

    n = len(series)
    if max_level is None:
        max_level = min(pywt.dwt_max_level(n, wavelet), 8)

    if max_level < 1:
        return [], np.array([]), []

    coeffs = pywt.wavedec(series, wavelet, level=max_level)
    approx = coeffs[0]
    details = coeffs[1:]

    # Energy by scale
    energy_by_scale = []
    total_energy = 0
    for d in details:
        e = np.sum(d ** 2)
        energy_by_scale.append(e)
        total_energy += e

    if total_energy > 0:
        energy_by_scale = [e / total_energy for e in energy_by_scale]

    return details, approx, energy_by_scale


def _compute_scipy_fallback(series: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray, List[float]]:
    """Fallback using scipy continuous wavelet transform."""
    from scipy.signal import ricker, cwt

    n = len(series)
    widths = np.arange(1, min(n // 4, 32))

    if len(widths) < 2:
        return [], np.array([]), []

    cwtmatr = cwt(series, ricker, widths)

    energy_by_scale = []
    total_energy = 0
    for row in cwtmatr:
        e = np.sum(row ** 2)
        energy_by_scale.append(e)
        total_energy += e

    if total_energy > 0:
        energy_by_scale = [e / total_energy for e in energy_by_scale]

    return [], np.array([]), energy_by_scale


def _empty_result() -> Dict[str, Any]:
    """Return empty result for insufficient data."""
    return {
        'energy_by_scale': [],
        'dominant_scale': 0,
        'scale_entropy': 0.0,
        'energy_ratio_low_high': 1.0,
        'detail_mean': 0.0,
        'detail_std': 0.0,
        'detail_kurtosis': 0.0,
        'approx_slope': 0.0,
        'approx_curvature': 0.0
    }
