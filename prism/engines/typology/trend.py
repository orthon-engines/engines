"""
Trend Engine
============

Detects and quantifies trends in time series data.

Metrics:
    - mann_kendall_tau: Mann-Kendall trend statistic (-1 to 1)
    - mann_kendall_pvalue: p-value for trend significance
    - sen_slope: Sen's slope estimator (robust trend magnitude)
    - sen_intercept: Sen's intercept estimator
    - trend_direction: 'increasing', 'decreasing', or 'no_trend'
    - trend_strength: Normalized trend strength (0-1)

Usage:
    from prism.engines.typology.trend import compute_trend
    result = compute_trend(values)
"""

import numpy as np
from typing import Dict, Any, Optional
from scipy import stats


def compute_trend(
    values: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Compute trend metrics for a time series.

    Args:
        values: 1D array of time series values
        alpha: Significance level for trend detection

    Returns:
        Dictionary with trend metrics
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]

    n = len(values)
    if n < 10:
        return _empty_result("Insufficient data (need >= 10 points)")

    # Mann-Kendall test
    tau, mk_pvalue, s, var_s = _mann_kendall(values)

    # Sen's slope estimator
    slope, intercept = _sen_slope(values)

    # Trend direction
    if mk_pvalue < alpha:
        if tau > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"
    else:
        trend_direction = "no_trend"

    # Trend strength (normalized)
    # Based on |tau| and significance
    if mk_pvalue < alpha:
        trend_strength = abs(tau)
    else:
        trend_strength = abs(tau) * 0.5  # Discount non-significant trends

    # Linear regression for comparison
    x = np.arange(n)
    lin_slope, lin_intercept, r_value, lin_pvalue, std_err = stats.linregress(x, values)

    return {
        "mann_kendall_tau": float(tau),
        "mann_kendall_pvalue": float(mk_pvalue),
        "mann_kendall_s": float(s),
        "mann_kendall_var_s": float(var_s),
        "sen_slope": float(slope),
        "sen_intercept": float(intercept),
        "trend_direction": trend_direction,
        "trend_strength": float(trend_strength),
        "is_significant": bool(mk_pvalue < alpha),
        "linear_slope": float(lin_slope),
        "linear_r_squared": float(r_value ** 2),
        "linear_pvalue": float(lin_pvalue),
        "n_observations": n,
    }


def _mann_kendall(values: np.ndarray) -> tuple:
    """
    Mann-Kendall trend test.

    H0: No monotonic trend
    H1: Monotonic trend exists

    Returns:
        tau: Kendall's tau (-1 to 1)
        pvalue: Two-sided p-value
        s: Mann-Kendall S statistic
        var_s: Variance of S
    """
    n = len(values)

    # Calculate S statistic
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = values[j] - values[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    # Handle ties
    unique, counts = np.unique(values, return_counts=True)
    ties = counts[counts > 1]

    # Variance of S
    var_s = (n * (n - 1) * (2 * n + 5)) / 18

    # Adjust variance for ties
    if len(ties) > 0:
        tie_correction = np.sum(ties * (ties - 1) * (2 * ties + 5)) / 18
        var_s -= tie_correction

    # Kendall's tau
    n_pairs = n * (n - 1) / 2
    tau = s / n_pairs if n_pairs > 0 else 0

    # Z-score with continuity correction
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0

    # Two-sided p-value
    pvalue = 2 * (1 - stats.norm.cdf(abs(z)))

    return tau, pvalue, s, var_s


def _sen_slope(values: np.ndarray) -> tuple:
    """
    Sen's slope estimator (Theil-Sen).

    Robust estimator of linear trend.
    Median of all pairwise slopes.

    Returns:
        slope: Sen's slope
        intercept: Sen's intercept
    """
    n = len(values)
    x = np.arange(n)

    # Calculate all pairwise slopes
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = values[j] - values[i]
            if dx != 0:
                slopes.append(dy / dx)

    if len(slopes) == 0:
        return None, np.median(values)

    # Sen's slope is the median
    slope = np.median(slopes)

    # Intercept: median of (y_i - slope * x_i)
    intercepts = values - slope * x
    intercept = np.median(intercepts)

    return slope, intercept


def _empty_result(reason: str) -> Dict[str, Any]:
    """Return empty result with reason."""
    return {
        "mann_kendall_tau": np.nan,
        "mann_kendall_pvalue": np.nan,
        "mann_kendall_s": np.nan,
        "mann_kendall_var_s": np.nan,
        "sen_slope": np.nan,
        "sen_intercept": np.nan,
        "trend_direction": "unknown",
        "trend_strength": 0.0,
        "is_significant": False,
        "linear_slope": np.nan,
        "linear_r_squared": np.nan,
        "linear_pvalue": np.nan,
        "n_observations": 0,
        "error": reason,
    }


def classify_trend(result: Dict[str, Any]) -> str:
    """
    Classify trend into a category based on strength and direction.

    Returns:
        'strong_increasing': tau > 0.5 and significant
        'moderate_increasing': 0.2 < tau <= 0.5 and significant
        'weak_increasing': tau > 0 and significant
        'strong_decreasing': tau < -0.5 and significant
        'moderate_decreasing': -0.5 <= tau < -0.2 and significant
        'weak_decreasing': tau < 0 and significant
        'no_trend': Not significant
    """
    tau = result.get("mann_kendall_tau", 0)
    is_sig = result.get("is_significant", False)

    if not is_sig:
        return "no_trend"

    if tau > 0.5:
        return "strong_increasing"
    elif tau > 0.2:
        return "moderate_increasing"
    elif tau > 0:
        return "weak_increasing"
    elif tau < -0.5:
        return "strong_decreasing"
    elif tau < -0.2:
        return "moderate_decreasing"
    else:
        return "weak_decreasing"


def compute_trend_change(
    values: np.ndarray,
    window_size: int = 20
) -> Dict[str, Any]:
    """
    Detect changes in trend direction over rolling windows.

    Args:
        values: 1D array of time series values
        window_size: Size of rolling window

    Returns:
        Dictionary with trend change analysis
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    if n < window_size * 2:
        return {"error": "Insufficient data for trend change detection"}

    # Compute rolling Sen's slopes
    slopes = []
    for i in range(n - window_size + 1):
        window = values[i:i + window_size]
        slope, _ = _sen_slope(window)
        slopes.append(slope)

    slopes = np.array(slopes)

    # Find sign changes (trend reversals)
    signs = np.sign(slopes)
    sign_changes = np.where(np.diff(signs) != 0)[0]

    # Find acceleration (slope changes)
    slope_diff = np.diff(slopes)

    return {
        "n_trend_reversals": len(sign_changes),
        "reversal_indices": sign_changes.tolist(),
        "mean_slope": float(np.mean(slopes)),
        "slope_volatility": float(np.std(slopes)),
        "max_slope": float(np.max(slopes)),
        "min_slope": float(np.min(slopes)),
        "slope_range": float(np.max(slopes) - np.min(slopes)),
        "mean_acceleration": float(np.mean(slope_diff)),
    }
