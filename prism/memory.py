"""
PRISM Memory and Long-Range Dependence Primitives

Pure mathematical functions for measuring long-range dependence,
persistence, and memory in time series.
"""

import numpy as np
from typing import Tuple, Optional


def hurst_exponent(
    values: np.ndarray,
    method: str = 'rs'
) -> float:
    """
    Compute Hurst exponent.

    The Hurst exponent measures long-range dependence:
    - H < 0.5: Anti-persistent (mean-reverting)
    - H = 0.5: Random walk (no memory)
    - H > 0.5: Persistent (trending)

    Args:
        values: Input time series
        method: 'rs' (rescaled range) or 'dfa' (detrended fluctuation)

    Returns:
        Hurst exponent (0 to 1)
    """
    values = np.asarray(values, dtype=np.float64)

    if method == 'rs':
        return _hurst_rs(values)
    elif method == 'dfa':
        return _hurst_dfa(values)
    else:
        raise ValueError(f"Unknown method: {method}")


def _hurst_rs(values: np.ndarray) -> float:
    """Compute Hurst exponent using rescaled range (R/S) method."""
    n = len(values)

    if n < 20:
        return 0.5  # Not enough data

    # Use multiple segment sizes
    segment_sizes = []
    rs_values = []

    for size in [n // 8, n // 4, n // 2, n]:
        if size < 10:
            continue

        # Split into segments
        n_segments = n // size
        if n_segments < 1:
            continue

        segment_rs = []
        for i in range(n_segments):
            segment = values[i * size:(i + 1) * size]

            # Mean-adjusted cumulative sum
            mean = np.mean(segment)
            cumsum = np.cumsum(segment - mean)

            # Range
            R = np.max(cumsum) - np.min(cumsum)

            # Standard deviation
            S = np.std(segment, ddof=1)

            if S > 0:
                segment_rs.append(R / S)

        if len(segment_rs) > 0:
            segment_sizes.append(size)
            rs_values.append(np.mean(segment_rs))

    if len(segment_sizes) < 2:
        return 0.5

    # Fit log(R/S) vs log(size) to get Hurst exponent
    log_sizes = np.log(segment_sizes)
    log_rs = np.log(rs_values)

    slope, _ = np.polyfit(log_sizes, log_rs, 1)

    return float(np.clip(slope, 0, 1))


def _hurst_dfa(values: np.ndarray) -> float:
    """Compute Hurst exponent using detrended fluctuation analysis."""
    n = len(values)

    if n < 20:
        return 0.5

    # Integration (cumulative sum of deviations from mean)
    y = np.cumsum(values - np.mean(values))

    # Try multiple scales
    scales = np.unique(np.floor(np.logspace(1, np.log10(n // 4), 10))).astype(int)
    scales = scales[scales >= 4]

    if len(scales) < 2:
        return 0.5

    fluctuations = []

    for scale in scales:
        # Divide into segments
        n_segments = n // scale
        if n_segments < 1:
            continue

        rms_list = []

        for i in range(n_segments):
            segment = y[i * scale:(i + 1) * scale]

            # Fit linear trend
            x = np.arange(len(segment))
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)

            # RMS of detrended segment
            rms = np.sqrt(np.mean((segment - trend) ** 2))
            rms_list.append(rms)

        if len(rms_list) > 0:
            fluctuations.append(np.mean(rms_list))
        else:
            fluctuations.append(0)

    # Remove zeros
    valid_mask = np.array(fluctuations) > 0
    if np.sum(valid_mask) < 2:
        return 0.5

    valid_scales = scales[valid_mask]
    valid_fluct = np.array(fluctuations)[valid_mask]

    # Fit log-log
    log_scales = np.log(valid_scales)
    log_fluct = np.log(valid_fluct)

    slope, _ = np.polyfit(log_scales, log_fluct, 1)

    return float(np.clip(slope, 0, 1))


def detrended_fluctuation_analysis(
    values: np.ndarray,
    min_scale: int = 4,
    max_scale: Optional[int] = None,
    n_scales: int = 10
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Perform detrended fluctuation analysis.

    Args:
        values: Input time series
        min_scale: Minimum scale size
        max_scale: Maximum scale size (default: len(values) // 4)
        n_scales: Number of scale points

    Returns:
        Tuple of (scales, fluctuations, alpha)
        alpha is the DFA exponent (related to Hurst: H = alpha)
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    if max_scale is None:
        max_scale = n // 4

    # Integration
    y = np.cumsum(values - np.mean(values))

    # Generate scales
    scales = np.unique(np.floor(np.logspace(
        np.log10(min_scale),
        np.log10(max_scale),
        n_scales
    ))).astype(int)

    fluctuations = []

    for scale in scales:
        n_segments = n // scale
        if n_segments < 1:
            fluctuations.append(np.nan)
            continue

        rms_list = []

        for i in range(n_segments):
            segment = y[i * scale:(i + 1) * scale]

            # Linear detrend
            x = np.arange(len(segment))
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)

            rms = np.sqrt(np.mean((segment - trend) ** 2))
            rms_list.append(rms)

        fluctuations.append(np.mean(rms_list))

    fluctuations = np.array(fluctuations)

    # Remove invalid values
    valid = ~np.isnan(fluctuations) & (fluctuations > 0)
    if np.sum(valid) < 2:
        return scales, fluctuations, 0.5

    # Fit to get alpha
    log_scales = np.log(scales[valid])
    log_fluct = np.log(fluctuations[valid])

    alpha, _ = np.polyfit(log_scales, log_fluct, 1)

    return scales, fluctuations, float(alpha)


def rescaled_range(
    values: np.ndarray,
    segment_size: Optional[int] = None
) -> float:
    """
    Compute rescaled range (R/S) statistic.

    Args:
        values: Input time series
        segment_size: Size of segment (default: full length)

    Returns:
        R/S statistic
    """
    values = np.asarray(values, dtype=np.float64)

    if segment_size is None:
        segment_size = len(values)

    # Mean-adjusted cumulative sum
    mean = np.mean(values[:segment_size])
    cumsum = np.cumsum(values[:segment_size] - mean)

    # Range
    R = np.max(cumsum) - np.min(cumsum)

    # Standard deviation
    S = np.std(values[:segment_size], ddof=1)

    if S == 0:
        return 0.0

    return float(R / S)


def long_range_correlation(
    values: np.ndarray,
    max_lag: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    """
    Analyze long-range correlation via autocorrelation decay.

    Args:
        values: Input time series
        max_lag: Maximum lag to analyze

    Returns:
        Tuple of (autocorrelation_values, decay_exponent)
        Decay exponent < 1 indicates long-range correlations
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    if max_lag is None:
        max_lag = n // 4

    # Compute autocorrelation
    values_centered = values - np.mean(values)
    autocorr = np.correlate(values_centered, values_centered, mode='full')
    autocorr = autocorr[n - 1:]
    autocorr = autocorr / autocorr[0]  # Normalize

    acf = autocorr[:max_lag + 1]

    # Fit power-law decay: ACF(τ) ~ τ^(-α)
    lags = np.arange(1, len(acf))
    acf_positive = acf[1:]

    # Only use positive ACF values
    valid = acf_positive > 0
    if np.sum(valid) < 2:
        return acf, 1.0

    log_lags = np.log(lags[valid])
    log_acf = np.log(acf_positive[valid])

    decay_exp, _ = np.polyfit(log_lags, log_acf, 1)

    return acf, float(-decay_exp)


def variance_growth(
    values: np.ndarray,
    max_lag: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    """
    Analyze variance growth with aggregation scale.

    For random walks, variance grows linearly with lag.
    For anti-persistent series, variance grows sub-linearly.
    For persistent series, variance grows super-linearly.

    Args:
        values: Input time series
        max_lag: Maximum aggregation scale

    Returns:
        Tuple of (scales, scaling_exponent)
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    if max_lag is None:
        max_lag = n // 4

    scales = np.arange(1, max_lag + 1)
    variances = []

    for scale in scales:
        # Aggregate at this scale
        n_agg = n // scale
        if n_agg < 2:
            variances.append(np.nan)
            continue

        aggregated = [
            np.mean(values[i * scale:(i + 1) * scale])
            for i in range(n_agg)
        ]

        variances.append(np.var(aggregated))

    variances = np.array(variances)

    # Fit power law
    valid = ~np.isnan(variances) & (variances > 0)
    if np.sum(valid) < 2:
        return scales, 1.0

    log_scales = np.log(scales[valid])
    log_var = np.log(variances[valid])

    exponent, _ = np.polyfit(log_scales, log_var, 1)

    return scales, float(exponent)
