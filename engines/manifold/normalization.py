"""
Normalization Engine
====================

Pure computation engine for data normalization.
Supports multiple methods with trade-offs for different data distributions.

Methods:
- zscore: Standard (x - mean) / std - assumes Gaussian, sensitive to outliers
- robust: (x - median) / IQR - robust to outliers, assumes symmetric distribution
- mad: (x - median) / MAD - most robust, works for heavy-tailed distributions
- minmax: Scale to [0, 1] range - preserves distribution shape

ENGINES computes numbers. ORTHON classifies.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Literal
from enum import Enum


class NormMethod(str, Enum):
    """Normalization methods with increasing robustness to outliers."""
    ZSCORE = "zscore"      # Standard: sensitive to outliers
    ROBUST = "robust"      # IQR-based: moderate robustness
    MAD = "mad"            # MAD-based: most robust
    MINMAX = "minmax"      # Range-based: preserves shape
    NONE = "none"          # No normalization


# MAD scale factor for consistency with std (assuming Gaussian)
# For normal distribution: std ≈ 1.4826 * MAD
MAD_SCALE_FACTOR = 1.4826


def compute_zscore(
    data: np.ndarray,
    axis: Optional[int] = 0,
    ddof: int = 0
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Z-score normalization: (x - mean) / std

    Risks:
    - Sensitive to outliers (single extreme value compresses distribution)
    - Assumes Gaussian distribution
    - Temporal contamination if computed globally

    Use when:
    - Data is approximately Gaussian
    - No significant outliers present
    - Baseline period is known to be clean

    Args:
        data: Input array (N x D for axis=0 normalizes each column)
        axis: Axis along which to compute statistics (0=columns, 1=rows, None=global)
        ddof: Degrees of freedom for std calculation

    Returns:
        Tuple of (normalized_data, params_dict)
        params_dict contains 'mean' and 'std' for inverse transform
    """
    data = np.asarray(data, dtype=np.float64)

    mean = np.nanmean(data, axis=axis, keepdims=True)
    std = np.nanstd(data, axis=axis, ddof=ddof, keepdims=True)

    # Avoid division by zero - use 1.0 to preserve original values for constant features
    std = np.where(std < 1e-10, 1.0, std)

    normalized = (data - mean) / std

    params = {
        'method': 'zscore',
        'mean': np.squeeze(mean) if axis is not None else mean,
        'std': np.squeeze(std) if axis is not None else std,
    }

    return normalized, params


def compute_robust(
    data: np.ndarray,
    axis: Optional[int] = 0,
    quantile_range: Tuple[float, float] = (25.0, 75.0)
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Robust normalization: (x - median) / IQR

    Advantages over z-score:
    - Median is robust to outliers (50% breakdown point)
    - IQR ignores extreme tails

    Risks:
    - Assumes symmetric distribution around median
    - Less efficient than std for Gaussian data

    Use when:
    - Outliers are present or suspected
    - Distribution is roughly symmetric but heavy-tailed

    Args:
        data: Input array
        axis: Axis along which to compute statistics
        quantile_range: Quantiles for IQR (default 25th-75th)

    Returns:
        Tuple of (normalized_data, params_dict)
    """
    data = np.asarray(data, dtype=np.float64)

    median = np.nanmedian(data, axis=axis, keepdims=True)

    q_low, q_high = quantile_range
    q1 = np.nanpercentile(data, q_low, axis=axis, keepdims=True)
    q3 = np.nanpercentile(data, q_high, axis=axis, keepdims=True)
    iqr = q3 - q1

    # Avoid division by zero
    iqr = np.where(iqr < 1e-10, 1.0, iqr)

    normalized = (data - median) / iqr

    params = {
        'method': 'robust',
        'median': np.squeeze(median) if axis is not None else median,
        'iqr': np.squeeze(iqr) if axis is not None else iqr,
        'q1': np.squeeze(q1) if axis is not None else q1,
        'q3': np.squeeze(q3) if axis is not None else q3,
    }

    return normalized, params


def compute_mad(
    data: np.ndarray,
    axis: Optional[int] = 0,
    scale: bool = True
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    MAD normalization: (x - median) / MAD

    MAD = Median Absolute Deviation = median(|x - median(x)|)

    Most robust normalization method:
    - 50% breakdown point (can handle up to 50% outliers)
    - Works for asymmetric and heavy-tailed distributions
    - Consistent with std for Gaussian (when scaled by 1.4826)

    Risks:
    - Less efficient than std for truly Gaussian data
    - MAD can be zero for discrete data with >50% at one value

    Use when:
    - Unknown distribution characteristics
    - Suspected outliers or anomalies in data
    - Heavy-tailed distributions (financial, industrial)

    Args:
        data: Input array
        axis: Axis along which to compute statistics
        scale: If True, scale MAD to be consistent with std for Gaussian

    Returns:
        Tuple of (normalized_data, params_dict)
    """
    data = np.asarray(data, dtype=np.float64)

    median = np.nanmedian(data, axis=axis, keepdims=True)

    # MAD = median(|x - median(x)|)
    abs_deviation = np.abs(data - median)
    mad = np.nanmedian(abs_deviation, axis=axis, keepdims=True)

    # Scale to be consistent with std for Gaussian
    if scale:
        mad = mad * MAD_SCALE_FACTOR

    # Avoid division by zero
    mad = np.where(mad < 1e-10, 1.0, mad)

    normalized = (data - median) / mad

    params = {
        'method': 'mad',
        'median': np.squeeze(median) if axis is not None else median,
        'mad': np.squeeze(mad) if axis is not None else mad,
        'scaled': scale,
    }

    return normalized, params


def compute_minmax(
    data: np.ndarray,
    axis: Optional[int] = 0,
    feature_range: Tuple[float, float] = (0.0, 1.0)
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Min-max normalization: scale to [min, max] range.

    Advantages:
    - Preserves distribution shape
    - Bounded output range

    Risks:
    - Extremely sensitive to outliers (single outlier affects entire scale)
    - No centering - doesn't account for central tendency

    Use when:
    - Bounded output is required (e.g., neural network input)
    - Data is already clean with no outliers

    Args:
        data: Input array
        axis: Axis along which to compute statistics
        feature_range: Output range (default [0, 1])

    Returns:
        Tuple of (normalized_data, params_dict)
    """
    data = np.asarray(data, dtype=np.float64)
    new_min, new_max = feature_range

    data_min = np.nanmin(data, axis=axis, keepdims=True)
    data_max = np.nanmax(data, axis=axis, keepdims=True)
    data_range = data_max - data_min

    # Avoid division by zero
    data_range = np.where(data_range < 1e-10, 1.0, data_range)

    # Scale to [0, 1] then to target range
    normalized = (data - data_min) / data_range
    normalized = normalized * (new_max - new_min) + new_min

    params = {
        'method': 'minmax',
        'data_min': np.squeeze(data_min) if axis is not None else data_min,
        'data_max': np.squeeze(data_max) if axis is not None else data_max,
        'feature_range': feature_range,
    }

    return normalized, params


def normalize(
    data: np.ndarray,
    method: str = "zscore",
    axis: Optional[int] = 0,
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Unified normalization interface.

    Args:
        data: Input array
        method: One of 'zscore', 'robust', 'mad', 'minmax', 'none'
        axis: Axis along which to compute (0=columns, 1=rows, None=global)
        **kwargs: Method-specific parameters

    Returns:
        Tuple of (normalized_data, params_dict)

    Example:
        >>> data = np.array([[1, 2], [3, 100], [5, 6]])  # Note outlier in col 2
        >>> norm_z, _ = normalize(data, method='zscore')  # Outlier compresses scale
        >>> norm_mad, _ = normalize(data, method='mad')   # Robust to outlier
    """
    method = method.lower()

    if method == "none":
        return data.copy(), {'method': 'none'}
    elif method == "zscore":
        return compute_zscore(data, axis=axis, **kwargs)
    elif method == "robust":
        return compute_robust(data, axis=axis, **kwargs)
    elif method == "mad":
        return compute_mad(data, axis=axis, **kwargs)
    elif method == "minmax":
        return compute_minmax(data, axis=axis, **kwargs)
    else:
        raise ValueError(f"Unknown normalization method: {method}. "
                        f"Use one of: zscore, robust, mad, minmax, none")


def inverse_normalize(
    normalized_data: np.ndarray,
    params: Dict[str, Any]
) -> np.ndarray:
    """
    Inverse transform normalized data back to original scale.

    Args:
        normalized_data: Normalized array
        params: Parameters from normalize() call

    Returns:
        Data in original scale
    """
    method = params.get('method', 'zscore')

    if method == 'none':
        return normalized_data.copy()

    elif method == 'zscore':
        mean = params['mean']
        std = params['std']
        if mean.ndim < normalized_data.ndim:
            mean = np.expand_dims(mean, axis=0)
            std = np.expand_dims(std, axis=0)
        return normalized_data * std + mean

    elif method == 'robust':
        median = params['median']
        iqr = params['iqr']
        if median.ndim < normalized_data.ndim:
            median = np.expand_dims(median, axis=0)
            iqr = np.expand_dims(iqr, axis=0)
        return normalized_data * iqr + median

    elif method == 'mad':
        median = params['median']
        mad = params['mad']
        if median.ndim < normalized_data.ndim:
            median = np.expand_dims(median, axis=0)
            mad = np.expand_dims(mad, axis=0)
        return normalized_data * mad + median

    elif method == 'minmax':
        data_min = params['data_min']
        data_max = params['data_max']
        new_min, new_max = params['feature_range']
        if data_min.ndim < normalized_data.ndim:
            data_min = np.expand_dims(data_min, axis=0)
            data_max = np.expand_dims(data_max, axis=0)
        # Reverse: (x - new_min) / (new_max - new_min) * range + data_min
        return ((normalized_data - new_min) / (new_max - new_min)
                * (data_max - data_min) + data_min)

    else:
        raise ValueError(f"Unknown method in params: {method}")


def recommend_method(
    data: np.ndarray,
    axis: Optional[int] = 0
) -> Dict[str, Any]:
    """
    Recommend normalization method based on data characteristics.

    ENGINES computes statistics. Recommendation is informational only.

    Args:
        data: Input array
        axis: Axis for analysis

    Returns:
        Dict with recommended method and supporting statistics
    """
    data = np.asarray(data, dtype=np.float64)

    # Compute distribution statistics
    mean = np.nanmean(data, axis=axis)
    median = np.nanmedian(data, axis=axis)
    std = np.nanstd(data, axis=axis)

    # Kurtosis (excess kurtosis: 0 for Gaussian)
    centered = data - np.nanmean(data, axis=axis, keepdims=True)
    m4 = np.nanmean(centered ** 4, axis=axis)
    m2 = np.nanmean(centered ** 2, axis=axis)
    kurtosis = np.where(m2 > 0, m4 / (m2 ** 2) - 3, 0)

    # Skewness
    m3 = np.nanmean(centered ** 3, axis=axis)
    skewness = np.where(m2 > 0, m3 / (m2 ** 1.5), 0)

    # Detect outliers via IQR method
    q1 = np.nanpercentile(data, 25, axis=axis)
    q3 = np.nanpercentile(data, 75, axis=axis)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    if axis == 0:
        outlier_mask = (data < lower) | (data > upper)
        outlier_fraction = np.nanmean(outlier_mask, axis=0)
    else:
        outlier_fraction = np.nanmean((data < lower) | (data > upper))

    # Mean kurtosis and outlier fraction for recommendation
    mean_kurtosis = np.nanmean(kurtosis)
    mean_outlier_frac = np.nanmean(outlier_fraction)
    mean_skewness = np.nanmean(np.abs(skewness))

    # Decision logic
    if mean_outlier_frac > 0.05:
        # >5% outliers: use MAD
        recommended = "mad"
        reason = f"High outlier fraction ({mean_outlier_frac:.1%})"
    elif mean_kurtosis > 3:
        # Heavy-tailed: use robust or MAD
        if mean_kurtosis > 10:
            recommended = "mad"
            reason = f"Very heavy tails (kurtosis={mean_kurtosis:.1f})"
        else:
            recommended = "robust"
            reason = f"Heavy tails (kurtosis={mean_kurtosis:.1f})"
    elif mean_skewness > 1:
        # Skewed: use robust
        recommended = "robust"
        reason = f"Skewed distribution (|skewness|={mean_skewness:.1f})"
    else:
        # Approximately Gaussian: z-score is fine
        recommended = "zscore"
        reason = "Approximately Gaussian distribution"

    return {
        'recommended_method': recommended,
        'reason': reason,
        'statistics': {
            'mean_kurtosis': float(mean_kurtosis),
            'mean_abs_skewness': float(mean_skewness),
            'outlier_fraction': float(mean_outlier_frac),
        }
    }


# Convenience aliases
zscore = compute_zscore
robust_scale = compute_robust
mad_scale = compute_mad
minmax_scale = compute_minmax
