"""
Statistical Primitives (1-12)

Basic statistical measures for individual signals.
"""

import numpy as np
from scipy import stats as scipy_stats
from typing import Tuple, Union, List

try:
    from rudder_primitives_rs.individual import (
        skewness as _skewness_rs,
        kurtosis as _kurtosis_rs,
        crest_factor as _crest_factor_rs,
    )
    _USE_RUST_STATS = True
except ImportError:
    _USE_RUST_STATS = False


def mean(signal: np.ndarray) -> float:
    """
    Compute arithmetic mean.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    float
        Arithmetic mean

    Notes
    -----
    mean = (1/n) * sum(x_i)
    """
    return float(np.nanmean(signal))


def std(signal: np.ndarray, ddof: int = 0) -> float:
    """
    Compute standard deviation.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    ddof : int
        Delta degrees of freedom (0 for population, 1 for sample)

    Returns
    -------
    float
        Standard deviation

    Notes
    -----
    std = sqrt((1/(n-ddof)) * sum((x_i - mean)^2))
    """
    return float(np.nanstd(signal, ddof=ddof))


def variance(signal: np.ndarray, ddof: int = 0) -> float:
    """
    Compute variance.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    ddof : int
        Delta degrees of freedom

    Returns
    -------
    float
        Variance

    Notes
    -----
    var = (1/(n-ddof)) * sum((x_i - mean)^2)
    """
    return float(np.nanvar(signal, ddof=ddof))


def min_max(signal: np.ndarray) -> Tuple[float, float]:
    """
    Compute minimum and maximum values.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    tuple
        (min_value, max_value)
    """
    return float(np.nanmin(signal)), float(np.nanmax(signal))


def percentiles(signal: np.ndarray, qs: List[float] = None) -> np.ndarray:
    """
    Compute percentiles.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    qs : list of float
        Percentiles to compute (default: [25, 50, 75])

    Returns
    -------
    np.ndarray
        Percentile values
    """
    if qs is None:
        qs = [25, 50, 75]
    return np.nanpercentile(signal, qs)


def skewness(signal: np.ndarray) -> float:
    """
    Compute skewness (third standardized moment).

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    float
        Skewness

    Notes
    -----
    Measures asymmetry of distribution.
    skewness > 0: right tail longer
    skewness < 0: left tail longer
    skewness = 0: symmetric
    """
    signal = np.asarray(signal).flatten()
    if _USE_RUST_STATS:
        clean = signal[~np.isnan(signal)]
        if len(clean) < 3:
            return np.nan
        return _skewness_rs(clean)

    return float(scipy_stats.skew(signal, nan_policy='omit'))


def kurtosis(signal: np.ndarray, fisher: bool = True) -> float:
    """
    Compute kurtosis (fourth standardized moment).

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fisher : bool
        If True, return excess kurtosis (kurtosis - 3)

    Returns
    -------
    float
        Kurtosis

    Notes
    -----
    Measures "tailedness" of distribution.
    excess kurtosis > 0: heavy tails (leptokurtic)
    excess kurtosis < 0: light tails (platykurtic)
    excess kurtosis = 0: normal-like tails (mesokurtic)
    """
    signal = np.asarray(signal).flatten()
    if _USE_RUST_STATS:
        clean = signal[~np.isnan(signal)]
        if len(clean) < 4:
            return np.nan
        return _kurtosis_rs(clean, fisher)

    return float(scipy_stats.kurtosis(signal, fisher=fisher, nan_policy='omit'))


def rms(signal: np.ndarray) -> float:
    """
    Compute root mean square.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    float
        RMS value

    Notes
    -----
    RMS = sqrt(mean(x^2))
    Related to signal power/energy.
    """
    signal = np.asarray(signal)
    return float(np.sqrt(np.nanmean(signal ** 2)))


def peak_to_peak(signal: np.ndarray) -> float:
    """
    Compute peak-to-peak amplitude.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    float
        Peak-to-peak value (max - min)
    """
    return float(np.nanmax(signal) - np.nanmin(signal))


def crest_factor(signal: np.ndarray) -> float:
    """
    Compute crest factor (peak / RMS).

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    float
        Crest factor

    Notes
    -----
    Crest factor = peak / RMS
    Indicates how "peaky" the signal is.
    Sine wave: sqrt(2) ≈ 1.414
    Square wave: 1.0
    Impulse: high value
    """
    signal = np.asarray(signal).flatten()
    if _USE_RUST_STATS:
        clean = signal[~np.isnan(signal)]
        if len(clean) == 0:
            return np.nan
        return _crest_factor_rs(clean)

    rms_val = rms(signal)
    peak_val = np.nanmax(np.abs(signal))
    if rms_val == 0:
        return np.nan
    return float(peak_val / rms_val)


def zero_crossings(signal: np.ndarray) -> int:
    """
    Count zero crossings.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    int
        Number of zero crossings

    Notes
    -----
    Related to signal frequency content.
    More crossings = higher frequency content.
    """
    signal = np.asarray(signal)
    signal = signal[~np.isnan(signal)]
    return int(np.sum(np.diff(np.signbit(signal))))


def mean_crossings(signal: np.ndarray) -> int:
    """
    Count mean crossings.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    int
        Number of mean crossings

    Notes
    -----
    Zero crossings of the zero-mean signal.
    """
    signal = np.asarray(signal)
    signal = signal[~np.isnan(signal)]
    centered = signal - np.mean(signal)
    return int(np.sum(np.diff(np.signbit(centered))))
