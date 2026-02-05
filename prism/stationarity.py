"""
PRISM Stationarity Testing Primitives

Pure mathematical functions for testing stationarity of time series.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional


def kpss_test(
    values: np.ndarray,
    regression: str = 'c',
    nlags: Optional[int] = None
) -> Tuple[float, float, dict]:
    """
    Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for stationarity.

    Null hypothesis: The series is stationary.
    Small p-value: Reject null, series is non-stationary.

    Args:
        values: Input time series
        regression: 'c' (constant only) or 'ct' (constant and trend)
        nlags: Number of lags for HAC covariance (default: auto)

    Returns:
        Tuple of (test_statistic, p_value, critical_values_dict)
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    n = len(values)

    if n < 10:
        return np.nan, np.nan, {}

    # Fit regression
    if regression == 'c':
        # Constant only
        residuals = values - np.mean(values)
    elif regression == 'ct':
        # Constant and trend
        t = np.arange(n)
        coeffs = np.polyfit(t, values, 1)
        trend = np.polyval(coeffs, t)
        residuals = values - trend
    else:
        raise ValueError(f"Unknown regression type: {regression}")

    # Cumulative sum of residuals
    cumsum = np.cumsum(residuals)

    # Estimate long-run variance using Newey-West
    if nlags is None:
        nlags = int(4 * (n / 100) ** 0.25)

    # Short-run variance
    gamma0 = np.sum(residuals ** 2) / n

    # Add autocorrelations with Bartlett weights
    for i in range(1, nlags + 1):
        weight = 1 - i / (nlags + 1)
        gamma_i = np.sum(residuals[i:] * residuals[:-i]) / n
        gamma0 += 2 * weight * gamma_i

    # KPSS statistic
    stat = np.sum(cumsum ** 2) / (n ** 2 * gamma0)

    # Critical values (asymptotic)
    if regression == 'c':
        critical_values = {'1%': 0.739, '5%': 0.463, '10%': 0.347}
    else:
        critical_values = {'1%': 0.216, '5%': 0.146, '10%': 0.119}

    # Approximate p-value
    if stat < critical_values['10%']:
        p_value = 0.10
    elif stat < critical_values['5%']:
        p_value = 0.05
    elif stat < critical_values['1%']:
        p_value = 0.01
    else:
        p_value = 0.001

    return float(stat), float(p_value), critical_values


def augmented_dickey_fuller(
    values: np.ndarray,
    max_lag: Optional[int] = None,
    regression: str = 'c'
) -> Tuple[float, float, dict]:
    """
    Augmented Dickey-Fuller test for unit root.

    Null hypothesis: The series has a unit root (non-stationary).
    Small p-value: Reject null, series is stationary.

    Args:
        values: Input time series
        max_lag: Maximum number of lags to include
        regression: 'c' (constant), 'ct' (constant+trend), 'n' (none)

    Returns:
        Tuple of (test_statistic, p_value, critical_values_dict)
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    n = len(values)

    if n < 20:
        return np.nan, np.nan, {}

    if max_lag is None:
        max_lag = int(12 * (n / 100) ** 0.25)

    # First difference
    diff = np.diff(values)

    # Lagged level
    y_lag = values[:-1]

    # Lagged differences
    if max_lag > 0:
        diff_lags = np.column_stack([
            diff[max_lag - i - 1:-i - 1] if i > 0 else diff[max_lag - 1:-1]
            for i in range(max_lag)
        ])
        y = diff[max_lag:]
        X = np.column_stack([y_lag[max_lag:], diff_lags])
    else:
        y = diff
        X = y_lag.reshape(-1, 1)

    # Add constant/trend
    if regression == 'c':
        X = np.column_stack([np.ones(len(y)), X])
    elif regression == 'ct':
        X = np.column_stack([np.ones(len(y)), np.arange(len(y)), X])

    # OLS regression
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, {}

    # Standard error of coefficient on y_{t-1}
    if regression == 'n':
        coef_idx = 0
    elif regression == 'c':
        coef_idx = 1
    else:
        coef_idx = 2

    # Residual variance
    resid = y - X @ coeffs
    sigma2 = np.sum(resid ** 2) / (len(y) - len(coeffs))

    # Covariance matrix
    try:
        cov_matrix = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(cov_matrix[coef_idx, coef_idx])
    except np.linalg.LinAlgError:
        se = np.nan

    # ADF statistic
    gamma = coeffs[coef_idx]
    stat = gamma / se if se > 0 else np.nan

    # Critical values (MacKinnon 1994)
    if regression == 'c':
        critical_values = {'1%': -3.43, '5%': -2.86, '10%': -2.57}
    elif regression == 'ct':
        critical_values = {'1%': -3.96, '5%': -3.41, '10%': -3.12}
    else:
        critical_values = {'1%': -2.58, '5%': -1.95, '10%': -1.62}

    # Approximate p-value
    if np.isnan(stat):
        p_value = np.nan
    elif stat < critical_values['1%']:
        p_value = 0.001
    elif stat < critical_values['5%']:
        p_value = 0.01
    elif stat < critical_values['10%']:
        p_value = 0.05
    else:
        p_value = 0.10

    return float(stat), float(p_value), critical_values


def variance_ratio_test(
    values: np.ndarray,
    lags: int = 2
) -> Tuple[float, float]:
    """
    Variance ratio test for random walk hypothesis.

    Under random walk, Var(k-period return) = k * Var(1-period return)
    VR = 1 for random walk
    VR < 1 for mean reversion
    VR > 1 for momentum/trending

    Args:
        values: Input time series (typically returns or log prices)
        lags: Number of periods for comparison

    Returns:
        Tuple of (variance_ratio, z_statistic)
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    n = len(values)

    if n < lags * 2:
        return np.nan, np.nan

    # 1-period variance
    var1 = np.var(np.diff(values), ddof=1)

    # k-period variance
    diff_k = values[lags:] - values[:-lags]
    var_k = np.var(diff_k, ddof=1)

    if var1 == 0:
        return np.nan, np.nan

    # Variance ratio
    vr = var_k / (lags * var1)

    # Asymptotic variance (under null of random walk)
    asymp_var = 2 * (2 * lags - 1) * (lags - 1) / (3 * lags * n)

    # Z-statistic
    z_stat = (vr - 1) / np.sqrt(asymp_var)

    return float(vr), float(z_stat)


def runs_test(values: np.ndarray) -> Tuple[float, float]:
    """
    Runs test for randomness.

    A run is a sequence of consecutive identical signs.
    Tests whether the sequence of + and - is random.

    Args:
        values: Input time series

    Returns:
        Tuple of (z_statistic, p_value)
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]

    # Convert to signs (relative to median or mean)
    median = np.median(values)
    signs = np.sign(values - median)
    signs = signs[signs != 0]  # Remove zeros

    n = len(signs)
    if n < 10:
        return np.nan, np.nan

    # Count runs
    runs = 1
    for i in range(1, n):
        if signs[i] != signs[i - 1]:
            runs += 1

    # Count positives and negatives
    n_pos = np.sum(signs > 0)
    n_neg = np.sum(signs < 0)

    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan

    # Expected runs and variance under null
    expected_runs = 1 + 2 * n_pos * n_neg / n
    var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n ** 2 * (n - 1))

    if var_runs <= 0:
        return np.nan, np.nan

    # Z-statistic
    z_stat = (runs - expected_runs) / np.sqrt(var_runs)

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return float(z_stat), float(p_value)


def is_stationary(
    values: np.ndarray,
    significance: float = 0.05
) -> bool:
    """
    Simple stationarity check using multiple tests.

    Args:
        values: Input time series
        significance: Significance level

    Returns:
        True if series appears stationary, False otherwise
    """
    # ADF test (null = non-stationary)
    adf_stat, adf_p, _ = augmented_dickey_fuller(values)
    adf_stationary = adf_p < significance if not np.isnan(adf_p) else False

    # KPSS test (null = stationary)
    kpss_stat, kpss_p, _ = kpss_test(values)
    kpss_stationary = kpss_p > significance if not np.isnan(kpss_p) else False

    # Both tests should agree
    return adf_stationary and kpss_stationary
