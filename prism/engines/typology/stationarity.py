"""
Stationarity Engine
===================

Tests for stationarity in time series data.

Metrics:
    - adf_statistic: Augmented Dickey-Fuller test statistic
    - adf_pvalue: ADF test p-value (< 0.05 suggests stationarity)
    - kpss_statistic: KPSS test statistic
    - kpss_pvalue: KPSS test p-value (> 0.05 suggests stationarity)
    - is_stationary: Boolean combining both tests
    - unit_root_present: Boolean indicating presence of unit root

Usage:
    from prism.engines.typology.stationarity import compute_stationarity
    result = compute_stationarity(values)
"""

import numpy as np
from typing import Dict, Any, Optional
from scipy import stats


def compute_stationarity(
    values: np.ndarray,
    max_lags: Optional[int] = None,
    regression: str = "c"
) -> Dict[str, Any]:
    """
    Compute stationarity tests for a time series.

    Args:
        values: 1D array of time series values
        max_lags: Maximum lags for ADF test (None = auto)
        regression: Type of regression for ADF ('c', 'ct', 'ctt', 'n')
                   'c' = constant only (default)
                   'ct' = constant + trend
                   'ctt' = constant + linear + quadratic trend
                   'n' = no constant, no trend

    Returns:
        Dictionary with stationarity metrics
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]

    n = len(values)
    if n < 20:
        return _empty_result("Insufficient data (need >= 20 points)")

    # Augmented Dickey-Fuller test
    adf_stat, adf_pvalue = _adf_test(values, max_lags, regression)

    # KPSS test
    kpss_stat, kpss_pvalue = _kpss_test(values, regression)

    # Combined interpretation
    # ADF: reject H0 (p < 0.05) means stationary
    # KPSS: fail to reject H0 (p > 0.05) means stationary
    adf_stationary = adf_pvalue < 0.05
    kpss_stationary = kpss_pvalue > 0.05

    # Both tests should agree for confidence
    is_stationary = adf_stationary and kpss_stationary
    unit_root_present = not adf_stationary

    # Confidence based on agreement
    if adf_stationary == kpss_stationary:
        confidence = 0.9  # Tests agree
    else:
        confidence = 0.5  # Tests disagree - ambiguous

    return {
        "adf_statistic": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "kpss_statistic": float(kpss_stat),
        "kpss_pvalue": float(kpss_pvalue),
        "is_stationary": bool(is_stationary),
        "unit_root_present": bool(unit_root_present),
        "adf_stationary": bool(adf_stationary),
        "kpss_stationary": bool(kpss_stationary),
        "confidence": float(confidence),
        "n_observations": n,
    }


def _adf_test(
    values: np.ndarray,
    max_lags: Optional[int] = None,
    regression: str = "c"
) -> tuple:
    """
    Augmented Dickey-Fuller test for unit root.

    H0: Unit root present (non-stationary)
    H1: No unit root (stationary)

    Reject H0 if p-value < 0.05 (stationary)
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(values, maxlag=max_lags, regression=regression, autolag='AIC')
        return result[0], result[1]  # statistic, pvalue
    except ImportError:
        # Fallback: simple implementation
        return _simple_adf(values)


def _simple_adf(values: np.ndarray) -> tuple:
    """
    Simplified ADF test when statsmodels not available.
    Uses first differences and tests for mean reversion.
    """
    n = len(values)

    # First difference
    diff = np.diff(values)
    lagged = values[:-1]

    # Regression: diff_t = alpha + beta * y_{t-1} + epsilon
    # If beta < 0 and significant, series is stationary
    X = np.column_stack([np.ones(len(lagged)), lagged])

    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(X, diff, rcond=None)
        beta = coeffs[1]

        # Estimate standard error
        if len(residuals) > 0:
            mse = residuals[0] / (n - 2)
        else:
            mse = np.var(diff - X @ coeffs)

        XtX_inv = np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(mse * XtX_inv[1, 1])

        # Test statistic
        t_stat = beta / se_beta if se_beta > 0 else 0

        # Critical values (approximate for n > 100)
        # -3.43 (1%), -2.86 (5%), -2.57 (10%)
        if t_stat < -3.43:
            pvalue = 0.01
        elif t_stat < -2.86:
            pvalue = 0.05
        elif t_stat < -2.57:
            pvalue = 0.10
        else:
            pvalue = 0.5

        return float(t_stat), float(pvalue)
    except Exception:
        return None, None


def _kpss_test(values: np.ndarray, regression: str = "c") -> tuple:
    """
    KPSS test for stationarity.

    H0: Series is stationary
    H1: Series has unit root (non-stationary)

    Fail to reject H0 if p-value > 0.05 (stationary)
    """
    try:
        from statsmodels.tsa.stattools import kpss
        nlags = "auto" if regression == "c" else "legacy"
        result = kpss(values, regression=regression, nlags=nlags)
        return result[0], result[1]  # statistic, pvalue
    except ImportError:
        # Fallback: simple implementation
        return _simple_kpss(values)


def _simple_kpss(values: np.ndarray) -> tuple:
    """
    Simplified KPSS test when statsmodels not available.
    """
    n = len(values)

    # Demean
    y = values - np.mean(values)

    # Cumulative sum of residuals
    S = np.cumsum(y)

    # Long-run variance estimate (Newey-West)
    lags = int(4 * (n / 100) ** 0.25)
    gamma0 = np.sum(y ** 2) / n

    gamma_sum = 0
    for j in range(1, lags + 1):
        w = 1 - j / (lags + 1)  # Bartlett weights
        gamma_j = np.sum(y[j:] * y[:-j]) / n
        gamma_sum += 2 * w * gamma_j

    sigma2 = gamma0 + gamma_sum

    # KPSS statistic
    if sigma2 > 0:
        eta = np.sum(S ** 2) / (n ** 2 * sigma2)
    else:
        eta = 0

    # Critical values for level stationarity
    # 0.347 (10%), 0.463 (5%), 0.574 (2.5%), 0.739 (1%)
    if eta > 0.739:
        pvalue = 0.01
    elif eta > 0.574:
        pvalue = 0.025
    elif eta > 0.463:
        pvalue = 0.05
    elif eta > 0.347:
        pvalue = 0.10
    else:
        pvalue = 0.15

    return float(eta), float(pvalue)


def _empty_result(reason: str) -> Dict[str, Any]:
    """Return empty result with reason."""
    return {
        "adf_statistic": np.nan,
        "adf_pvalue": np.nan,
        "kpss_statistic": np.nan,
        "kpss_pvalue": np.nan,
        "is_stationary": False,
        "unit_root_present": True,
        "adf_stationary": False,
        "kpss_stationary": False,
        "confidence": 0.0,
        "n_observations": 0,
        "error": reason,
    }


# Convenience function for classification
def classify_stationarity(result: Dict[str, Any]) -> str:
    """
    Classify stationarity result into a category.

    Returns:
        'stationary': Both tests indicate stationarity
        'non_stationary': Both tests indicate non-stationarity
        'trend_stationary': ADF says stationary but KPSS doesn't (has trend)
        'difference_stationary': KPSS says stationary but ADF doesn't
        'ambiguous': Tests disagree in unexpected way
    """
    adf = result.get("adf_stationary", False)
    kpss = result.get("kpss_stationary", False)

    if adf and kpss:
        return "stationary"
    elif not adf and not kpss:
        return "non_stationary"
    elif adf and not kpss:
        return "trend_stationary"
    elif not adf and kpss:
        return "difference_stationary"
    else:
        return "ambiguous"
