"""
PRISM Normalization Primitives

Pure mathematical functions for data normalization.
Multiple methods with different robustness to outliers.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple


# MAD scale factor for consistency with std (assuming Gaussian)
# For normal distribution: std ≈ 1.4826 * MAD
MAD_SCALE_FACTOR = 1.4826


def zscore_normalize(
    values: np.ndarray,
    axis: Optional[int] = 0,
    ddof: int = 0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Z-score normalization: (x - mean) / std

    Standard normalization that centers data and scales by standard deviation.

    Risks:
    - Sensitive to outliers
    - Assumes approximately Gaussian distribution

    Args:
        values: Input array
        axis: Axis along which to compute (0=columns, 1=rows, None=global)
        ddof: Degrees of freedom for std calculation

    Returns:
        Tuple of (normalized_data, params_dict)
    """
    values = np.asarray(values, dtype=np.float64)

    mean = np.nanmean(values, axis=axis, keepdims=True)
    std = np.nanstd(values, axis=axis, ddof=ddof, keepdims=True)

    # Avoid division by zero
    std = np.where(std < 1e-10, 1.0, std)

    normalized = (values - mean) / std

    params = {
        'method': 'zscore',
        'mean': np.squeeze(mean) if axis is not None else mean,
        'std': np.squeeze(std) if axis is not None else std,
    }

    return normalized, params


def robust_normalize(
    values: np.ndarray,
    axis: Optional[int] = 0,
    quantile_range: Tuple[float, float] = (25.0, 75.0)
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Robust normalization: (x - median) / IQR

    Uses median and interquartile range for robustness to outliers.

    Args:
        values: Input array
        axis: Axis along which to compute
        quantile_range: Quantiles for IQR (default 25th-75th)

    Returns:
        Tuple of (normalized_data, params_dict)
    """
    values = np.asarray(values, dtype=np.float64)

    median = np.nanmedian(values, axis=axis, keepdims=True)

    q_low, q_high = quantile_range
    q1 = np.nanpercentile(values, q_low, axis=axis, keepdims=True)
    q3 = np.nanpercentile(values, q_high, axis=axis, keepdims=True)
    iqr = q3 - q1

    # Avoid division by zero
    iqr = np.where(iqr < 1e-10, 1.0, iqr)

    normalized = (values - median) / iqr

    params = {
        'method': 'robust',
        'median': np.squeeze(median) if axis is not None else median,
        'iqr': np.squeeze(iqr) if axis is not None else iqr,
        'q1': np.squeeze(q1) if axis is not None else q1,
        'q3': np.squeeze(q3) if axis is not None else q3,
    }

    return normalized, params


def mad_normalize(
    values: np.ndarray,
    axis: Optional[int] = 0,
    scale: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    MAD normalization: (x - median) / MAD

    Most robust normalization method using Median Absolute Deviation.
    Can handle up to 50% outliers (50% breakdown point).

    Args:
        values: Input array
        axis: Axis along which to compute
        scale: If True, scale MAD to be consistent with std for Gaussian

    Returns:
        Tuple of (normalized_data, params_dict)
    """
    values = np.asarray(values, dtype=np.float64)

    median = np.nanmedian(values, axis=axis, keepdims=True)

    # MAD = median(|x - median(x)|)
    abs_deviation = np.abs(values - median)
    mad = np.nanmedian(abs_deviation, axis=axis, keepdims=True)

    # Scale to be consistent with std for Gaussian
    if scale:
        mad = mad * MAD_SCALE_FACTOR

    # Avoid division by zero
    mad = np.where(mad < 1e-10, 1.0, mad)

    normalized = (values - median) / mad

    params = {
        'method': 'mad',
        'median': np.squeeze(median) if axis is not None else median,
        'mad': np.squeeze(mad) if axis is not None else mad,
        'scaled': scale,
    }

    return normalized, params


def minmax_normalize(
    values: np.ndarray,
    axis: Optional[int] = 0,
    feature_range: Tuple[float, float] = (0.0, 1.0)
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Min-max normalization: scale to [min, max] range.

    Preserves distribution shape but sensitive to outliers.

    Args:
        values: Input array
        axis: Axis along which to compute
        feature_range: Output range (default [0, 1])

    Returns:
        Tuple of (normalized_data, params_dict)
    """
    values = np.asarray(values, dtype=np.float64)
    new_min, new_max = feature_range

    data_min = np.nanmin(values, axis=axis, keepdims=True)
    data_max = np.nanmax(values, axis=axis, keepdims=True)
    data_range = data_max - data_min

    # Avoid division by zero
    data_range = np.where(data_range < 1e-10, 1.0, data_range)

    # Scale to [0, 1] then to target range
    normalized = (values - data_min) / data_range
    normalized = normalized * (new_max - new_min) + new_min

    params = {
        'method': 'minmax',
        'data_min': np.squeeze(data_min) if axis is not None else data_min,
        'data_max': np.squeeze(data_max) if axis is not None else data_max,
        'feature_range': feature_range,
    }

    return normalized, params


def quantile_normalize(
    values: np.ndarray,
    n_quantiles: int = 100
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Quantile normalization: transform to uniform distribution.

    Maps values to their quantile positions.

    Args:
        values: Input array
        n_quantiles: Number of quantile bins

    Returns:
        Tuple of (normalized_data, params_dict)
    """
    values = np.asarray(values, dtype=np.float64).flatten()

    # Compute quantile positions
    sorted_idx = np.argsort(values)
    normalized = np.empty_like(values)
    normalized[sorted_idx] = np.linspace(0, 1, len(values))

    params = {
        'method': 'quantile',
        'n_quantiles': n_quantiles,
    }

    return normalized, params


def inverse_normalize(
    normalized_data: np.ndarray,
    params: Dict[str, Any]
) -> np.ndarray:
    """
    Inverse transform normalized data back to original scale.

    Args:
        normalized_data: Normalized array
        params: Parameters from normalize call

    Returns:
        Data in original scale
    """
    normalized_data = np.asarray(normalized_data, dtype=np.float64)
    method = params.get('method', 'zscore')

    if method == 'none':
        return normalized_data.copy()

    elif method == 'zscore':
        mean = params['mean']
        std = params['std']
        if np.ndim(mean) < normalized_data.ndim:
            mean = np.expand_dims(mean, axis=0)
            std = np.expand_dims(std, axis=0)
        return normalized_data * std + mean

    elif method == 'robust':
        median = params['median']
        iqr = params['iqr']
        if np.ndim(median) < normalized_data.ndim:
            median = np.expand_dims(median, axis=0)
            iqr = np.expand_dims(iqr, axis=0)
        return normalized_data * iqr + median

    elif method == 'mad':
        median = params['median']
        mad = params['mad']
        if np.ndim(median) < normalized_data.ndim:
            median = np.expand_dims(median, axis=0)
            mad = np.expand_dims(mad, axis=0)
        return normalized_data * mad + median

    elif method == 'minmax':
        data_min = params['data_min']
        data_max = params['data_max']
        new_min, new_max = params['feature_range']
        if np.ndim(data_min) < normalized_data.ndim:
            data_min = np.expand_dims(data_min, axis=0)
            data_max = np.expand_dims(data_max, axis=0)
        return ((normalized_data - new_min) / (new_max - new_min)
                * (data_max - data_min) + data_min)

    else:
        raise ValueError(f"Unknown method in params: {method}")


def normalize(
    values: np.ndarray,
    method: str = "zscore",
    axis: Optional[int] = 0,
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Unified normalization interface.

    Args:
        values: Input array
        method: One of 'zscore', 'robust', 'mad', 'minmax', 'quantile', 'none'
        axis: Axis along which to compute
        **kwargs: Method-specific parameters

    Returns:
        Tuple of (normalized_data, params_dict)
    """
    method = method.lower()

    if method == "none":
        return np.asarray(values, dtype=np.float64).copy(), {'method': 'none'}
    elif method == "zscore":
        return zscore_normalize(values, axis=axis, **kwargs)
    elif method == "robust":
        return robust_normalize(values, axis=axis, **kwargs)
    elif method == "mad":
        return mad_normalize(values, axis=axis, **kwargs)
    elif method == "minmax":
        return minmax_normalize(values, axis=axis, **kwargs)
    elif method == "quantile":
        return quantile_normalize(values, **kwargs)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def recommend_method(values: np.ndarray, axis: Optional[int] = 0) -> Dict[str, Any]:
    """
    Recommend normalization method based on data characteristics.

    Args:
        values: Input array
        axis: Axis for analysis

    Returns:
        Dict with recommended method and supporting statistics
    """
    values = np.asarray(values, dtype=np.float64)

    # Compute distribution statistics
    mean_val = np.nanmean(values, axis=axis)
    median_val = np.nanmedian(values, axis=axis)

    # Kurtosis (excess kurtosis: 0 for Gaussian)
    centered = values - np.nanmean(values, axis=axis, keepdims=True)
    m4 = np.nanmean(centered ** 4, axis=axis)
    m2 = np.nanmean(centered ** 2, axis=axis)
    kurtosis = np.where(m2 > 0, m4 / (m2 ** 2) - 3, 0)

    # Skewness
    m3 = np.nanmean(centered ** 3, axis=axis)
    skewness = np.where(m2 > 0, m3 / (m2 ** 1.5), 0)

    # Detect outliers via IQR method
    q1 = np.nanpercentile(values, 25, axis=axis)
    q3 = np.nanpercentile(values, 75, axis=axis)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    if axis == 0:
        outlier_mask = (values < lower) | (values > upper)
        outlier_fraction = np.nanmean(outlier_mask, axis=0)
    else:
        outlier_fraction = np.nanmean((values < lower) | (values > upper))

    # Mean statistics for recommendation
    mean_kurtosis = float(np.nanmean(kurtosis))
    mean_outlier_frac = float(np.nanmean(outlier_fraction))
    mean_skewness = float(np.nanmean(np.abs(skewness)))

    # Decision logic
    if mean_outlier_frac > 0.05:
        recommended = "mad"
        reason = f"High outlier fraction ({mean_outlier_frac:.1%})"
    elif mean_kurtosis > 3:
        if mean_kurtosis > 10:
            recommended = "mad"
            reason = f"Very heavy tails (kurtosis={mean_kurtosis:.1f})"
        else:
            recommended = "robust"
            reason = f"Heavy tails (kurtosis={mean_kurtosis:.1f})"
    elif mean_skewness > 1:
        recommended = "robust"
        reason = f"Skewed distribution (|skewness|={mean_skewness:.1f})"
    else:
        recommended = "zscore"
        reason = "Approximately Gaussian distribution"

    return {
        'recommended_method': recommended,
        'reason': reason,
        'statistics': {
            'mean_kurtosis': mean_kurtosis,
            'mean_abs_skewness': mean_skewness,
            'outlier_fraction': mean_outlier_frac,
        }
    }
