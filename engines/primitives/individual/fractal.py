"""
Fractal Primitives (31-32)

Hurst exponent and Detrended Fluctuation Analysis.
"""

import numpy as np
from typing import Optional, Tuple

from engines.primitives.config import PRIMITIVES_CONFIG as cfg


def hurst_exponent(
    signal: np.ndarray,
    method: str = 'rs'
) -> float:
    """
    Compute Hurst exponent.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    method : str
        'rs' (rescaled range) or 'dfa' (detrended fluctuation)

    Returns
    -------
    float
        Hurst exponent H

    Notes
    -----
    H < 0.5: anti-persistent (mean-reverting)
    H = 0.5: random walk (Brownian motion)
    H > 0.5: persistent (trending)

    For financial/natural signals, H > 0.5 indicates long-range dependence.
    """
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    # Hard math floor: R/S method needs at least rs_min_k samples for any
    # subseries. The internal len(k_values) < 3 check handles marginal cases.
    if n < cfg.fractal.rs_min_k:
        return np.nan

    if method == 'dfa':
        return dfa(signal)

    # Rescaled Range (R/S) method
    max_k = min(int(n * cfg.fractal.rs_max_k_ratio), cfg.fractal.rs_max_k_cap)
    k_values = []
    rs_values = []

    for k in range(cfg.fractal.rs_min_k, max_k):
        # Divide into subseries
        n_subseries = n // k
        rs_sum = 0

        for i in range(n_subseries):
            subseries = signal[i*k:(i+1)*k]
            mean = np.mean(subseries)

            # Cumulative deviation from mean
            Y = np.cumsum(subseries - mean)

            # Range
            R = np.max(Y) - np.min(Y)

            # Standard deviation
            S = np.std(subseries, ddof=1)

            if S > 0:
                rs_sum += R / S

        if n_subseries > 0:
            rs_avg = rs_sum / n_subseries
            if rs_avg > 0:
                k_values.append(np.log(k))
                rs_values.append(np.log(rs_avg))

    if len(k_values) < 3:
        return np.nan

    # Linear fit: log(R/S) = H * log(k) + c
    H, _ = np.polyfit(k_values, rs_values, 1)

    return float(np.clip(H, 0, 1))


def dfa(
    signal: np.ndarray,
    scale_range: Tuple[int, int] = None,
    order: int = 1
) -> float:
    """
    Compute Detrended Fluctuation Analysis.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    scale_range : tuple, optional
        (min_scale, max_scale)
    order : int
        Polynomial order for detrending

    Returns
    -------
    float
        DFA exponent (similar to Hurst)

    Notes
    -----
    alpha < 0.5: anti-correlated
    alpha = 0.5: uncorrelated (white noise)
    alpha = 1.0: 1/f noise (pink noise)
    alpha = 1.5: Brownian motion
    alpha > 1.5: non-stationary, unbounded
    """
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < cfg.min_samples.dfa:
        return np.nan

    # Integrate signal
    Y = np.cumsum(signal - np.mean(signal))

    # Define scales
    if scale_range is None:
        min_scale = cfg.fractal.dfa_min_scale
        max_scale = min(int(n * cfg.fractal.dfa_max_scale_ratio), cfg.fractal.dfa_max_scale_cap)
    else:
        min_scale, max_scale = scale_range

    scales = np.unique(np.logspace(
        np.log10(min_scale),
        np.log10(max_scale),
        cfg.fractal.dfa_n_scales
    ).astype(int))

    fluctuations = []

    for scale in scales:
        n_segments = n // scale
        if n_segments < 2:
            continue

        F_sq = []

        for i in range(n_segments):
            segment = Y[i*scale:(i+1)*scale]
            x = np.arange(scale)

            # Polynomial fit (local trend)
            coeffs = np.polyfit(x, segment, order)
            trend = np.polyval(coeffs, x)

            # Fluctuation
            F_sq.append(np.mean((segment - trend) ** 2))

        if F_sq:
            fluctuations.append(np.sqrt(np.mean(F_sq)))

    if len(fluctuations) < 3:
        return np.nan

    # Linear fit in log-log space
    log_scales = np.log(scales[:len(fluctuations)])
    log_fluct = np.log(fluctuations)

    alpha, _ = np.polyfit(log_scales, log_fluct, 1)

    return float(alpha)


def hurst_r2(signal: np.ndarray) -> float:
    """
    Compute R-squared of Hurst exponent fit.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    float
        R-squared value (goodness of fit)

    Notes
    -----
    High R² indicates reliable Hurst estimate.
    Low R² may indicate non-scaling behavior.
    """
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    # Hard math floor: same as hurst_exponent
    if n < cfg.fractal.rs_min_k:
        return np.nan

    max_k = min(int(n * cfg.fractal.rs_max_k_ratio), cfg.fractal.rs_max_k_cap)
    log_k = []
    log_rs = []

    for k in range(cfg.fractal.rs_min_k, max_k):
        n_subseries = n // k
        rs_sum = 0

        for i in range(n_subseries):
            subseries = signal[i*k:(i+1)*k]
            Y = np.cumsum(subseries - np.mean(subseries))
            R = np.max(Y) - np.min(Y)
            S = np.std(subseries, ddof=1)

            if S > 0:
                rs_sum += R / S

        if n_subseries > 0:
            rs_avg = rs_sum / n_subseries
            if rs_avg > 0:
                log_k.append(np.log(k))
                log_rs.append(np.log(rs_avg))

    if len(log_k) < 3:
        return np.nan

    # Compute R²
    slope, intercept = np.polyfit(log_k, log_rs, 1)
    predicted = slope * np.array(log_k) + intercept
    ss_res = np.sum((np.array(log_rs) - predicted) ** 2)
    ss_tot = np.sum((np.array(log_rs) - np.mean(log_rs)) ** 2)

    if ss_tot == 0:
        return np.nan

    return float(1 - ss_res / ss_tot)
