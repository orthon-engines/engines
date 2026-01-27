"""
Cointegration Engine

Tests for long-run equilibrium relationships between signals.

CANONICAL INTERFACE:
    def compute(observations: pd.DataFrame) -> pd.DataFrame
    Input:  [entity_id, signal_id, I, y]
    Output: [entity_id, signal_a, signal_b, coint_stat, pvalue, is_cointegrated]

Two series are cointegrated if they share a common stochastic trend,
meaning they move together in the long run even if they diverge short-term.
"""

ENGINE_TYPE = 'pairwise'

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any


def compute(
    observations: pd.DataFrame,
    significance: float = 0.05,
) -> pd.DataFrame:
    """
    Test cointegration for all signal pairs.

    CANONICAL INTERFACE:
        Input:  observations [entity_id, signal_id, I, y]
        Output: primitives [entity_id, signal_a, signal_b, coint_stat,
                           pvalue, is_cointegrated]

    Parameters
    ----------
    observations : pd.DataFrame
        Must have columns: entity_id, signal_id, I, y
    significance : float, optional
        Significance level for cointegration (default: 0.05)

    Returns
    -------
    pd.DataFrame
        Pairwise cointegration test results
    """
    results = []

    for entity_id, entity_group in observations.groupby('entity_id'):
        signals = entity_group['signal_id'].unique()

        if len(signals) < 2:
            continue

        # Pivot to wide format
        try:
            wide = entity_group.pivot(index='I', columns='signal_id', values='y')
            wide = wide.sort_index().dropna()
        except Exception:
            wide = entity_group.groupby(['I', 'signal_id'])['y'].mean().unstack()
            wide = wide.sort_index().dropna()

        if len(wide) < 20:
            continue

        # Test all pairs
        for i, sig_a in enumerate(signals):
            for sig_b in signals[i + 1:]:
                if sig_a not in wide.columns or sig_b not in wide.columns:
                    continue

                try:
                    result = _engle_granger_test(
                        wide[sig_a].values,
                        wide[sig_b].values
                    )

                    results.append({
                        'entity_id': entity_id,
                        'signal_a': sig_a,
                        'signal_b': sig_b,
                        'coint_stat': result['stat'],
                        'pvalue': result['pvalue'],
                        'is_cointegrated': result['pvalue'] < significance,
                    })
                except Exception:
                    results.append({
                        'entity_id': entity_id,
                        'signal_a': sig_a,
                        'signal_b': sig_b,
                        'coint_stat': np.nan,
                        'pvalue': np.nan,
                        'is_cointegrated': False,
                    })

    return pd.DataFrame(results)


def _engle_granger_test(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Engle-Granger two-step cointegration test.

    1. Regress y on x to get residuals
    2. Test residuals for stationarity (ADF test)
    """
    n = len(x)

    # Step 1: OLS regression y = a + b*x
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta

    # Step 2: ADF test on residuals
    adf_stat, pvalue = _adf_test(residuals)

    return {
        'stat': float(adf_stat),
        'pvalue': float(pvalue),
        'beta': float(beta[1]),
    }


def _adf_test(y: np.ndarray) -> tuple:
    """
    Augmented Dickey-Fuller test for stationarity.

    Tests H0: series has unit root (non-stationary)
    vs H1: series is stationary

    Returns (test_statistic, p_value)
    """
    n = len(y)

    if n < 20:
        return np.nan, np.nan

    # Difference and lagged level
    dy = np.diff(y)
    y_lag = y[:-1]

    # Include lagged differences for augmentation
    k = min(int(np.floor(12 * (n / 100) ** 0.25)), n // 3)

    # Build regression matrix
    Y = dy[k:]
    n_obs = len(Y)

    X_list = [np.ones(n_obs), y_lag[k:]]
    for i in range(1, k + 1):
        X_list.append(dy[k - i:-i] if i < len(dy) - k else np.zeros(n_obs))

    X = np.column_stack(X_list)

    try:
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        residuals = Y - X @ beta
        se = np.sqrt(np.sum(residuals ** 2) / (n_obs - len(beta)))

        # Standard error of rho coefficient
        XtX_inv = np.linalg.inv(X.T @ X)
        se_rho = se * np.sqrt(XtX_inv[1, 1])

        # ADF statistic
        rho = beta[1]
        adf_stat = rho / se_rho

        # Approximate p-value using normal distribution (rough approximation)
        # For proper critical values, use MacKinnon tables
        # This is simplified - use statsmodels for production
        pvalue = _mackinnon_pvalue(adf_stat, n_obs)

        return adf_stat, pvalue

    except Exception:
        return np.nan, np.nan


def _mackinnon_pvalue(stat: float, nobs: int) -> float:
    """
    Approximate MacKinnon p-value for ADF test.

    Uses simplified critical values for regression with constant.
    """
    # Critical values (approximate) for n -> inf
    # 1%: -3.43, 5%: -2.86, 10%: -2.57
    if stat < -3.43:
        return 0.01
    elif stat < -2.86:
        return 0.05
    elif stat < -2.57:
        return 0.10
    else:
        # Rough interpolation
        return min(1.0, 0.5 + 0.1 * stat)
