"""
Seasonality Engine
==================

Detects and measures seasonal patterns in time series data.

Metrics:
    - seasonal_strength: Strength of seasonal component (0-1)
    - trend_strength: Strength of trend component (0-1)
    - residual_strength: Strength of residual/noise (0-1)
    - dominant_period: Most likely seasonal period
    - is_seasonal: Boolean indicating significant seasonality

Usage:
    from prism.engines.typology.seasonality import compute_seasonality
    result = compute_seasonality(values, period=12)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy import signal as sp_signal
from scipy import stats


def compute_seasonality(
    values: np.ndarray,
    period: Optional[int] = None,
    robust: bool = True
) -> Dict[str, Any]:
    """
    Compute seasonality metrics using STL-like decomposition.

    Args:
        values: 1D array of time series values
        period: Seasonal period (None = auto-detect)
        robust: Use robust (median-based) decomposition

    Returns:
        Dictionary with seasonality metrics
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]

    n = len(values)
    if n < 20:
        return _empty_result("Insufficient data (need >= 20 points)")

    # Auto-detect period if not provided
    if period is None:
        period = _detect_period(values)

    if period < 2 or period > n // 2:
        return _empty_result(f"Invalid period: {period}")

    # STL-like decomposition
    trend, seasonal, residual = _stl_decompose(values, period, robust)

    # Compute strengths (variance-based)
    var_detrended = np.var(values - trend)
    var_deseasonalized = np.var(values - seasonal)
    var_residual = np.var(residual)
    var_total = np.var(values)

    if var_total > 0:
        # Seasonal strength: 1 - Var(R) / Var(detrended)
        seasonal_strength = max(0, 1 - var_residual / var_detrended) if var_detrended > 0 else 0

        # Trend strength: 1 - Var(R) / Var(deseasonalized)
        trend_strength = max(0, 1 - var_residual / var_deseasonalized) if var_deseasonalized > 0 else 0

        # Residual proportion
        residual_strength = var_residual / var_total
    else:
        seasonal_strength = 0.0
        trend_strength = 0.0
        residual_strength = 1.0

    # Seasonal amplitude
    seasonal_amplitude = np.max(seasonal) - np.min(seasonal)

    # Is seasonal? Threshold at 0.3 strength
    is_seasonal = seasonal_strength > 0.3

    # ACF-based seasonality check
    acf_seasonal, acf_peaks = _acf_seasonality(values, period)

    return {
        "seasonal_strength": float(seasonal_strength),
        "trend_strength": float(trend_strength),
        "residual_strength": float(residual_strength),
        "dominant_period": int(period),
        "is_seasonal": bool(is_seasonal),
        "seasonal_amplitude": float(seasonal_amplitude),
        "acf_at_period": float(acf_seasonal),
        "acf_peaks": acf_peaks,
        "trend_mean": float(np.mean(trend)),
        "trend_range": float(np.max(trend) - np.min(trend)),
        "residual_std": float(np.std(residual)),
        "n_observations": n,
    }


def _detect_period(values: np.ndarray) -> int:
    """
    Auto-detect seasonal period using periodogram.
    """
    n = len(values)

    # Detrend
    detrended = values - np.linspace(values[0], values[-1], n)

    # Periodogram
    freqs, psd = sp_signal.periodogram(detrended)

    # Find dominant frequency (excluding DC)
    psd[0] = 0  # Exclude DC component

    if len(psd) > 1:
        peak_idx = np.argmax(psd[1:]) + 1
        if freqs[peak_idx] > 0:
            period = int(round(1 / freqs[peak_idx]))
            period = max(2, min(period, n // 3))
            return period

    # Default to common periods based on length
    if n >= 365:
        return 7  # Weekly
    elif n >= 24:
        return 12  # Monthly/Quarterly
    else:
        return max(2, n // 4)


def _stl_decompose(
    values: np.ndarray,
    period: int,
    robust: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    STL-like seasonal decomposition.

    Returns:
        trend: Trend component
        seasonal: Seasonal component
        residual: Residual component
    """
    n = len(values)

    # Try statsmodels first
    try:
        from statsmodels.tsa.seasonal import STL
        stl = STL(values, period=period, robust=robust)
        result = stl.fit()
        return result.trend, result.seasonal, result.resid
    except ImportError:
        pass

    # Fallback: simple moving average decomposition
    return _simple_decompose(values, period, robust)


def _simple_decompose(
    values: np.ndarray,
    period: int,
    robust: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple seasonal decomposition (moving average method).
    """
    n = len(values)

    # 1. Extract trend using centered moving average
    if period % 2 == 0:
        # Even period: use 2xperiod MA
        ma_window = period + 1
    else:
        ma_window = period

    trend = _moving_average(values, ma_window)

    # 2. Detrend
    detrended = values - trend

    # 3. Average seasonal pattern
    seasonal_pattern = np.zeros(period)
    counts = np.zeros(period)

    for i in range(n):
        idx = i % period
        if not np.isnan(detrended[i]):
            if robust:
                # Will compute median later
                pass
            seasonal_pattern[idx] += detrended[i]
            counts[idx] += 1

    # Compute averages
    for i in range(period):
        if counts[i] > 0:
            seasonal_pattern[i] /= counts[i]

    # Center seasonal pattern
    seasonal_pattern -= np.mean(seasonal_pattern)

    # 4. Extend seasonal pattern to full length
    seasonal = np.array([seasonal_pattern[i % period] for i in range(n)])

    # 5. Residual
    residual = values - trend - seasonal

    return trend, seasonal, residual


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """
    Centered moving average with edge handling.
    """
    n = len(values)
    result = np.full(n, np.nan)

    half = window // 2

    # Convolution for interior
    kernel = np.ones(window) / window
    conv = np.convolve(values, kernel, mode='valid')

    # Place centered
    start = half
    end = start + len(conv)
    result[start:end] = conv

    # Fill edges with nearest valid
    result[:start] = result[start]
    result[end:] = result[end - 1]

    return result


def _acf_seasonality(values: np.ndarray, period: int) -> Tuple[float, list]:
    """
    Check seasonality using autocorrelation function.

    Returns:
        acf_at_period: ACF value at the seasonal period
        acf_peaks: List of significant ACF peaks
    """
    n = len(values)
    max_lag = min(n // 2, period * 3)

    # Compute ACF
    mean = np.mean(values)
    var = np.var(values)

    if var == 0:
        return None, []

    acf = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acf.append(1.0)
        else:
            cov = np.mean((values[:-lag] - mean) * (values[lag:] - mean))
            acf.append(cov / var)

    acf = np.array(acf)

    # ACF at seasonal period
    if period <= max_lag:
        acf_at_period = acf[period]
    else:
        acf_at_period = 0.0

    # Find peaks
    # Significance threshold: 2/sqrt(n)
    threshold = 2 / np.sqrt(n)

    peaks = []
    for i in range(1, len(acf) - 1):
        if acf[i] > acf[i-1] and acf[i] > acf[i+1] and acf[i] > threshold:
            peaks.append({"lag": i, "acf": float(acf[i])})

    return float(acf_at_period), peaks


def _empty_result(reason: str) -> Dict[str, Any]:
    """Return empty result with reason."""
    return {
        "seasonal_strength": 0.0,
        "trend_strength": 0.0,
        "residual_strength": 1.0,
        "dominant_period": 0,
        "is_seasonal": False,
        "seasonal_amplitude": 0.0,
        "acf_at_period": 0.0,
        "acf_peaks": [],
        "trend_mean": np.nan,
        "trend_range": np.nan,
        "residual_std": np.nan,
        "n_observations": 0,
        "error": reason,
    }


def classify_seasonality(result: Dict[str, Any]) -> str:
    """
    Classify seasonality pattern.

    Returns:
        'strong_seasonal': seasonal_strength > 0.7
        'moderate_seasonal': 0.4 < seasonal_strength <= 0.7
        'weak_seasonal': 0.2 < seasonal_strength <= 0.4
        'no_seasonality': seasonal_strength <= 0.2
    """
    strength = result.get("seasonal_strength", 0)

    if strength > 0.7:
        return "strong_seasonal"
    elif strength > 0.4:
        return "moderate_seasonal"
    elif strength > 0.2:
        return "weak_seasonal"
    else:
        return "no_seasonality"


def decompose_multiple_seasonalities(
    values: np.ndarray,
    periods: list
) -> Dict[str, Any]:
    """
    Decompose time series with multiple seasonal periods.

    Args:
        values: 1D array of time series values
        periods: List of seasonal periods to extract

    Returns:
        Dictionary with components for each period
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    # Sort periods from shortest to longest
    periods = sorted(periods)

    result = {"seasonal_components": {}}
    residual = values.copy()

    # Extract each seasonal component
    for period in periods:
        if period >= n // 2:
            continue

        # Decompose current residual
        trend, seasonal, new_residual = _simple_decompose(residual, period, robust=True)

        # Store seasonal component
        result["seasonal_components"][period] = {
            "pattern": seasonal.tolist(),
            "amplitude": float(np.max(seasonal) - np.min(seasonal)),
            "strength": float(np.var(seasonal) / np.var(residual)) if np.var(residual) > 0 else 0,
        }

        # Update residual
        residual = new_residual

    result["final_residual_std"] = float(np.std(residual))
    result["total_seasonal_explained"] = float(1 - np.var(residual) / np.var(values)) if np.var(values) > 0 else 0

    return result
