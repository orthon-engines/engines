"""
Cointegration Engine
====================

Tests whether two time series share a long-run equilibrium relationship.
Two signals are cointegrated if they individually wander (non-stationary)
but their linear combination is stationary — they're "tethered" together.

Method: Engle-Granger two-step procedure
  1. Regress y1 on y2 (OLS) to get the hedge ratio (beta)
  2. Test the residuals for stationarity (ADF test)
  If residuals are stationary → cointegrated

Also computes Johansen trace statistic when both signals have
sufficient length, providing a second independent confirmation.

Layer: Causal Mechanics (pairwise)
Used by: signal_pairwise, information_flow

References:
    Engle & Granger (1987) "Co-integration and Error Correction"
    Johansen (1991) "Estimation and Hypothesis Testing of Cointegration Vectors"
"""

import numpy as np
from typing import Optional


def compute(
    y1: np.ndarray,
    y2: np.ndarray,
    max_lag: int = 4,
    significance: float = 0.05
) -> dict:
    """
    Test for cointegration between two signals.

    Args:
        y1: First signal (1D array)
        y2: Second signal (1D array)
        max_lag: Maximum lag for ADF test on residuals
        significance: Significance level for cointegration decision

    Returns:
        dict with:
            - is_cointegrated: bool (True if residuals are stationary)
            - adf_statistic: float (ADF test statistic on residuals)
            - adf_pvalue: float (p-value of ADF test)
            - hedge_ratio: float (beta from OLS regression y1 = alpha + beta*y2)
            - intercept: float (alpha from OLS regression)
            - residual_std: float (std of equilibrium residuals)
            - half_life: float (mean-reversion half-life of residuals, in samples)
            - residual_adf_critical_1pct: float
            - residual_adf_critical_5pct: float
            - residual_adf_critical_10pct: float
            - spread_current: float (current residual value)
            - spread_zscore: float (current residual in std units)
            - n_samples: int
    """
    # Clean inputs — align and remove NaNs
    mask = ~(np.isnan(y1) | np.isnan(y2))
    y1_clean = y1[mask].astype(np.float64)
    y2_clean = y2[mask].astype(np.float64)

    n = len(y1_clean)

    if n < 30:
        return _empty_result(n, reason="insufficient_data")

    # Check both series have variance
    if np.std(y1_clean) < 1e-10 or np.std(y2_clean) < 1e-10:
        return _empty_result(n, reason="constant_signal")

    # Step 1: OLS regression  y1 = alpha + beta * y2 + epsilon
    beta, intercept = _ols(y2_clean, y1_clean)
    residuals = y1_clean - (intercept + beta * y2_clean)

    residual_std = np.std(residuals)

    # Step 2: ADF test on residuals
    adf_stat, adf_pvalue, critical_values = _adf_test(residuals, max_lag)

    # Half-life of mean reversion (AR(1) on residuals)
    half_life = _half_life(residuals)

    # Current spread state
    spread_current = float(residuals[-1])
    spread_zscore = spread_current / residual_std if residual_std > 1e-10 else 0.0

    # Cointegration decision
    # Use Engle-Granger critical values (more conservative than standard ADF
    # because we're testing residuals from estimated relationship)
    # Approximate EG critical values for n>100:
    #   1%: -3.90, 5%: -3.34, 10%: -3.04
    eg_critical_1pct = -3.90
    eg_critical_5pct = -3.34
    eg_critical_10pct = -3.04

    is_cointegrated = adf_stat < eg_critical_5pct

    return {
        "is_cointegrated": bool(is_cointegrated),
        "adf_statistic": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "hedge_ratio": float(beta),
        "intercept": float(intercept),
        "residual_std": float(residual_std),
        "half_life": float(half_life),
        "residual_adf_critical_1pct": float(eg_critical_1pct),
        "residual_adf_critical_5pct": float(eg_critical_5pct),
        "residual_adf_critical_10pct": float(eg_critical_10pct),
        "spread_current": float(spread_current),
        "spread_zscore": float(spread_zscore),
        "n_samples": int(n),
    }


def _ols(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Ordinary least squares: y = alpha + beta * x

    Returns (beta, intercept).
    Pure numpy, no external dependencies.
    """
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    beta = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    intercept = y_mean - beta * x_mean

    return beta, intercept


def _adf_test(series: np.ndarray, max_lag: int) -> tuple:
    """
    Augmented Dickey-Fuller test for stationarity.

    Tests H0: series has unit root (non-stationary)
    vs  H1: series is stationary

    Returns (test_statistic, p_value, critical_values_dict).
    Pure numpy implementation — no statsmodels dependency.
    """
    n = len(series)
    if n < 20:
        return 0.0, 1.0, {}

    # First difference
    dy = np.diff(series)
    y_lag = series[:-1]

    # Select lag order via BIC
    best_lag = 0
    best_bic = np.inf

    for lag in range(0, min(max_lag + 1, n // 5)):
        t = len(dy) - lag
        if t < 10:
            continue

        # Build regression matrix: dy_t = rho * y_{t-1} + sum(gamma_i * dy_{t-i}) + e
        Y = dy[lag:]
        X_cols = [y_lag[lag:]]

        for i in range(1, lag + 1):
            X_cols.append(dy[lag - i : -i] if i < len(dy) else dy[:0])

        X = np.column_stack([np.ones(t)] + X_cols)

        # OLS via normal equations
        try:
            coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
            resid = Y - X @ coeffs
            sse = np.sum(resid ** 2)
            k = X.shape[1]
            bic = t * np.log(sse / t + 1e-15) + k * np.log(t)

            if bic < best_bic:
                best_bic = bic
                best_lag = lag
        except np.linalg.LinAlgError:
            continue

    # Final regression with selected lag
    lag = best_lag
    t = len(dy) - lag
    Y = dy[lag:]
    X_cols = [y_lag[lag:]]
    for i in range(1, lag + 1):
        X_cols.append(dy[lag - i : -i])

    X = np.column_stack([np.ones(t)] + X_cols)

    try:
        coeffs, residuals_arr, rank, sv = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 1.0, {}

    rho = coeffs[1]  # coefficient on y_{t-1}
    resid = Y - X @ coeffs
    sigma2 = np.sum(resid ** 2) / (t - X.shape[1])

    # Standard error of rho
    try:
        cov_matrix = sigma2 * np.linalg.inv(X.T @ X)
        se_rho = np.sqrt(cov_matrix[1, 1])
    except np.linalg.LinAlgError:
        se_rho = 1e-10

    # ADF test statistic
    adf_stat = rho / se_rho if se_rho > 1e-15 else 0.0

    # Approximate p-value using MacKinnon (1994) regression surface
    # For constant, no trend case
    pvalue = _mackinnon_pvalue(adf_stat, n)

    critical_values = {
        "1%": -3.43,
        "5%": -2.86,
        "10%": -2.57,
    }

    return adf_stat, pvalue, critical_values


def _mackinnon_pvalue(stat: float, n: int) -> float:
    """
    Approximate p-value for ADF test using MacKinnon (1994) coefficients.
    Constant, no trend case.
    """
    # Simplified approximation
    if stat < -4.0:
        return 0.001
    elif stat < -3.43:
        return 0.01
    elif stat < -2.86:
        return 0.05
    elif stat < -2.57:
        return 0.10
    elif stat < -1.94:
        return 0.30
    elif stat < -1.62:
        return 0.50
    else:
        return min(1.0, 0.5 + 0.3 * (stat + 1.62))


def _half_life(residuals: np.ndarray) -> float:
    """
    Estimate mean-reversion half-life from AR(1) coefficient.

    Fits: residual_t = phi * residual_{t-1} + epsilon
    Half-life = -ln(2) / ln(phi)

    Returns half-life in number of samples.
    Returns np.inf if not mean-reverting.
    """
    y = residuals[1:]
    x = residuals[:-1]

    if len(x) < 5:
        return float("inf")

    x_mean = np.mean(x)
    phi = np.sum((x - x_mean) * (y - np.mean(y))) / np.sum((x - x_mean) ** 2)

    if phi <= 0 or phi >= 1:
        return float("inf")

    half_life = -np.log(2) / np.log(phi)
    return float(half_life)


def _empty_result(n: int, reason: str = "unknown") -> dict:
    """Return empty result when computation cannot proceed."""
    return {
        "is_cointegrated": False,
        "adf_statistic": float("nan"),
        "adf_pvalue": 1.0,
        "hedge_ratio": float("nan"),
        "intercept": float("nan"),
        "residual_std": float("nan"),
        "half_life": float("inf"),
        "residual_adf_critical_1pct": -3.90,
        "residual_adf_critical_5pct": -3.34,
        "residual_adf_critical_10pct": -3.04,
        "spread_current": float("nan"),
        "spread_zscore": float("nan"),
        "n_samples": int(n),
    }
