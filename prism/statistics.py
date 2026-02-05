"""
PRISM Statistical Primitives

Pure mathematical functions for basic statistical measures.
All functions take numpy arrays and return numbers or arrays.
"""

import numpy as np
from scipy import stats as scipy_stats
from typing import Tuple, List, Optional


def mean(values: np.ndarray) -> float:
    """
    Compute arithmetic mean.

    Args:
        values: Input array

    Returns:
        Arithmetic mean
    """
    return float(np.nanmean(values))


def variance(values: np.ndarray, ddof: int = 0) -> float:
    """
    Compute variance.

    Args:
        values: Input array
        ddof: Delta degrees of freedom (0 for population, 1 for sample)

    Returns:
        Variance
    """
    return float(np.nanvar(values, ddof=ddof))


def standard_deviation(values: np.ndarray, ddof: int = 0) -> float:
    """
    Compute standard deviation.

    Args:
        values: Input array
        ddof: Delta degrees of freedom

    Returns:
        Standard deviation
    """
    return float(np.nanstd(values, ddof=ddof))


def skewness(values: np.ndarray) -> float:
    """
    Compute skewness (third standardized moment).

    Measures asymmetry of distribution.
    - skewness > 0: right tail longer
    - skewness < 0: left tail longer
    - skewness = 0: symmetric

    Args:
        values: Input array

    Returns:
        Skewness
    """
    return float(scipy_stats.skew(values, nan_policy='omit'))


def kurtosis(values: np.ndarray, fisher: bool = True) -> float:
    """
    Compute kurtosis (fourth standardized moment).

    Measures "tailedness" of distribution.
    - excess kurtosis > 0: heavy tails (leptokurtic)
    - excess kurtosis < 0: light tails (platykurtic)
    - excess kurtosis = 0: normal-like tails (mesokurtic)

    Args:
        values: Input array
        fisher: If True, return excess kurtosis (kurtosis - 3)

    Returns:
        Kurtosis
    """
    return float(scipy_stats.kurtosis(values, fisher=fisher, nan_policy='omit'))


def rms(values: np.ndarray) -> float:
    """
    Compute root mean square.

    RMS = sqrt(mean(x^2))

    Args:
        values: Input array

    Returns:
        RMS value
    """
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.nanmean(values ** 2)))


def peak_to_peak(values: np.ndarray) -> float:
    """
    Compute peak-to-peak amplitude.

    Args:
        values: Input array

    Returns:
        Peak-to-peak value (max - min)
    """
    return float(np.nanmax(values) - np.nanmin(values))


def crest_factor(values: np.ndarray) -> float:
    """
    Compute crest factor (peak / RMS).

    Indicates how "peaky" the signal is.
    - Sine wave: sqrt(2) ≈ 1.414
    - Square wave: 1.0
    - Impulse: high value

    Args:
        values: Input array

    Returns:
        Crest factor
    """
    values = np.asarray(values, dtype=np.float64)
    rms_val = rms(values)
    peak_val = np.nanmax(np.abs(values))

    if rms_val == 0:
        return np.nan

    return float(peak_val / rms_val)


def coefficient_of_variation(values: np.ndarray) -> float:
    """
    Compute coefficient of variation (CV = std / mean).

    Scale-free measure of dispersion.

    Args:
        values: Input array

    Returns:
        Coefficient of variation
    """
    m = mean(values)
    if m == 0:
        return np.nan
    return float(standard_deviation(values) / abs(m))


def median(values: np.ndarray) -> float:
    """
    Compute median.

    Args:
        values: Input array

    Returns:
        Median value
    """
    return float(np.nanmedian(values))


def percentile(values: np.ndarray, q: float) -> float:
    """
    Compute percentile.

    Args:
        values: Input array
        q: Percentile (0-100)

    Returns:
        Percentile value
    """
    return float(np.nanpercentile(values, q))


def iqr(values: np.ndarray) -> float:
    """
    Compute interquartile range (IQR = Q3 - Q1).

    Args:
        values: Input array

    Returns:
        Interquartile range
    """
    q1 = np.nanpercentile(values, 25)
    q3 = np.nanpercentile(values, 75)
    return float(q3 - q1)


def mad(values: np.ndarray, scale: bool = True) -> float:
    """
    Compute median absolute deviation (MAD).

    MAD = median(|x - median(x)|)

    Args:
        values: Input array
        scale: If True, scale by 1.4826 for consistency with std (Gaussian)

    Returns:
        MAD value
    """
    values = np.asarray(values, dtype=np.float64)
    med = np.nanmedian(values)
    abs_deviation = np.abs(values - med)
    mad_val = np.nanmedian(abs_deviation)

    if scale:
        mad_val = mad_val * 1.4826  # Consistency factor for Gaussian

    return float(mad_val)


def moments(values: np.ndarray, max_order: int = 4) -> np.ndarray:
    """
    Compute raw moments up to specified order.

    Args:
        values: Input array
        max_order: Maximum moment order

    Returns:
        Array of moments [m1, m2, m3, ...]
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]

    result = []
    for k in range(1, max_order + 1):
        result.append(np.mean(values ** k))

    return np.array(result)


def central_moments(values: np.ndarray, max_order: int = 4) -> np.ndarray:
    """
    Compute central moments up to specified order.

    Args:
        values: Input array
        max_order: Maximum moment order

    Returns:
        Array of central moments [μ2, μ3, μ4, ...]
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    centered = values - np.mean(values)

    result = []
    for k in range(2, max_order + 1):
        result.append(np.mean(centered ** k))

    return np.array(result)
