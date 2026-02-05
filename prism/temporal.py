"""
PRISM Temporal Analysis Primitives

Pure mathematical functions for time-domain analysis.
All functions take numpy arrays and return numbers or arrays.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional, List


def autocorrelation(
    values: np.ndarray,
    max_lag: Optional[int] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute autocorrelation function.

    Args:
        values: Input time series
        max_lag: Maximum lag to compute (default: len(values) - 1)
        normalize: If True, normalize to range [-1, 1]

    Returns:
        Autocorrelation values for lags 0 to max_lag
    """
    values = np.asarray(values, dtype=np.float64)
    values = values - np.nanmean(values)
    n = len(values)

    if max_lag is None:
        max_lag = n - 1

    # Full autocorrelation
    autocorr = np.correlate(values, values, mode='full')
    autocorr = autocorr[n - 1:]  # Take positive lags only

    if normalize and autocorr[0] != 0:
        autocorr = autocorr / autocorr[0]

    return autocorr[:max_lag + 1]


def autocorrelation_decay(
    values: np.ndarray,
    threshold: float = 1/np.e
) -> float:
    """
    Compute autocorrelation decay time (lag at which ACF drops below threshold).

    Args:
        values: Input time series
        threshold: Threshold for decay (default: 1/e ≈ 0.368)

    Returns:
        Decay lag (or len(values) if threshold never reached)
    """
    acf = autocorrelation(values, normalize=True)

    # Find first lag where ACF drops below threshold
    below_threshold = np.where(np.abs(acf) < threshold)[0]

    if len(below_threshold) == 0:
        return float(len(values))

    return float(below_threshold[0])


def trend_fit(
    values: np.ndarray,
    order: int = 1
) -> Tuple[np.ndarray, float]:
    """
    Fit polynomial trend to time series.

    Args:
        values: Input time series
        order: Polynomial order (1 for linear, 2 for quadratic, etc.)

    Returns:
        Tuple of (coefficients, r_squared)
        Coefficients are in descending order [a_n, a_{n-1}, ..., a_0]
    """
    values = np.asarray(values, dtype=np.float64)
    x = np.arange(len(values))

    # Remove NaN values
    valid = ~np.isnan(values)
    if np.sum(valid) < order + 1:
        return np.zeros(order + 1), 0.0

    coeffs = np.polyfit(x[valid], values[valid], order)

    # Calculate R-squared
    fitted = np.polyval(coeffs, x[valid])
    ss_res = np.sum((values[valid] - fitted) ** 2)
    ss_tot = np.sum((values[valid] - np.mean(values[valid])) ** 2)

    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return coeffs, float(r_squared)


def rate_of_change(
    values: np.ndarray,
    method: str = 'difference'
) -> np.ndarray:
    """
    Compute rate of change (first derivative).

    Args:
        values: Input time series
        method: 'difference' (simple diff) or 'gradient' (numpy gradient)

    Returns:
        Rate of change array
    """
    values = np.asarray(values, dtype=np.float64)

    if method == 'difference':
        return np.diff(values)
    elif method == 'gradient':
        return np.gradient(values)
    else:
        raise ValueError(f"Unknown method: {method}")


def turning_points(values: np.ndarray) -> int:
    """
    Count turning points (local extrema).

    A turning point is where the signal changes from increasing to
    decreasing or vice versa.

    Args:
        values: Input time series

    Returns:
        Number of turning points
    """
    values = np.asarray(values, dtype=np.float64)
    diff = np.diff(values)

    # Sign changes in the difference indicate turning points
    sign_changes = np.diff(np.sign(diff))
    return int(np.sum(sign_changes != 0))


def zero_crossings(values: np.ndarray) -> int:
    """
    Count zero crossings.

    Related to signal frequency content.
    More crossings = higher frequency content.

    Args:
        values: Input time series

    Returns:
        Number of zero crossings
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    return int(np.sum(np.diff(np.signbit(values))))


def mean_crossings(values: np.ndarray) -> int:
    """
    Count mean crossings.

    Args:
        values: Input time series

    Returns:
        Number of mean crossings
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    centered = values - np.mean(values)
    return int(np.sum(np.diff(np.signbit(centered))))


def peak_detection(
    values: np.ndarray,
    height: Optional[float] = None,
    distance: Optional[int] = None,
    prominence: Optional[float] = None
) -> Tuple[np.ndarray, dict]:
    """
    Detect peaks in signal.

    Args:
        values: Input time series
        height: Minimum peak height
        distance: Minimum distance between peaks
        prominence: Minimum peak prominence

    Returns:
        Tuple of (peak_indices, properties_dict)
    """
    values = np.asarray(values, dtype=np.float64)

    kwargs = {}
    if height is not None:
        kwargs['height'] = height
    if distance is not None:
        kwargs['distance'] = distance
    if prominence is not None:
        kwargs['prominence'] = prominence

    peaks, properties = signal.find_peaks(values, **kwargs)
    return peaks, properties


def envelope_extraction(
    values: np.ndarray,
    method: str = 'hilbert'
) -> np.ndarray:
    """
    Extract signal envelope.

    Args:
        values: Input time series
        method: 'hilbert' for Hilbert transform envelope

    Returns:
        Envelope signal
    """
    values = np.asarray(values, dtype=np.float64)

    if method == 'hilbert':
        analytic = signal.hilbert(values)
        envelope = np.abs(analytic)
    else:
        raise ValueError(f"Unknown method: {method}")

    return envelope


def moving_average(
    values: np.ndarray,
    window: int,
    mode: str = 'valid'
) -> np.ndarray:
    """
    Compute moving average.

    Args:
        values: Input time series
        window: Window size
        mode: 'valid', 'same', or 'full'

    Returns:
        Smoothed signal
    """
    values = np.asarray(values, dtype=np.float64)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode=mode)


def detrend(
    values: np.ndarray,
    order: int = 1
) -> np.ndarray:
    """
    Remove polynomial trend from signal.

    Args:
        values: Input time series
        order: Polynomial order to remove

    Returns:
        Detrended signal
    """
    values = np.asarray(values, dtype=np.float64)
    x = np.arange(len(values))

    valid = ~np.isnan(values)
    coeffs = np.polyfit(x[valid], values[valid], order)
    trend = np.polyval(coeffs, x)

    return values - trend


def segment_signal(
    values: np.ndarray,
    window_size: int,
    stride: int
) -> np.ndarray:
    """
    Segment signal into overlapping windows.

    Args:
        values: Input time series
        window_size: Size of each window
        stride: Step between windows

    Returns:
        2D array of shape (n_windows, window_size)
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    # Calculate number of windows
    n_windows = (n - window_size) // stride + 1

    if n_windows <= 0:
        return np.array([values])

    # Create windows
    windows = np.array([
        values[i * stride:i * stride + window_size]
        for i in range(n_windows)
    ])

    return windows
