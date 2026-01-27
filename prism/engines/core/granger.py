"""
Granger Causality Engine

Tests whether past values of X improve prediction of Y.

CANONICAL INTERFACE:
    def compute(observations: pd.DataFrame) -> pd.DataFrame
    Input:  [entity_id, signal_id, I, y]
    Output: [entity_id, source_id, target_id, granger_fstat, granger_pvalue,
             optimal_lag, is_causal]

Measures:
- F-statistic and p-value per pair
- Directional causality network
- Optimal lag structure
"""

ENGINE_TYPE = 'pairwise'

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Optional
import warnings


def compute(
    observations: pd.DataFrame,
    max_lag: int = 5,
    significance: float = 0.05,
) -> pd.DataFrame:
    """
    Compute Granger causality for all signal pairs.

    CANONICAL INTERFACE:
        Input:  observations [entity_id, signal_id, I, y]
        Output: primitives [entity_id, source_id, target_id, granger_fstat,
                           granger_pvalue, optimal_lag, is_causal]

    Parameters
    ----------
    observations : pd.DataFrame
        Must have columns: entity_id, signal_id, I, y
    max_lag : int, optional
        Maximum lag to test (default: 5)
    significance : float, optional
        Significance level for causality (default: 0.05)

    Returns
    -------
    pd.DataFrame
        Pairwise Granger causality test results
    """
    results = []

    for entity_id, entity_group in observations.groupby('entity_id'):
        signals = entity_group['signal_id'].unique()

        if len(signals) < 2:
            continue

        # Pivot to wide format for this entity
        try:
            wide = entity_group.pivot(index='I', columns='signal_id', values='y')
            wide = wide.sort_index().dropna()
        except Exception:
            wide = entity_group.groupby(['I', 'signal_id'])['y'].mean().unstack()
            wide = wide.sort_index().dropna()

        if len(wide) < max_lag + 10:
            continue

        # Test all pairs
        for source in signals:
            for target in signals:
                if source == target:
                    continue

                if source not in wide.columns or target not in wide.columns:
                    continue

                try:
                    result = _granger_test(
                        wide[source].values,
                        wide[target].values,
                        max_lag=max_lag
                    )

                    results.append({
                        'entity_id': entity_id,
                        'source_id': source,
                        'target_id': target,
                        'granger_fstat': result['f_stat'],
                        'granger_pvalue': result['p_value'],
                        'optimal_lag': result['optimal_lag'],
                        'is_causal': result['p_value'] < significance,
                    })
                except Exception:
                    results.append({
                        'entity_id': entity_id,
                        'source_id': source,
                        'target_id': target,
                        'granger_fstat': np.nan,
                        'granger_pvalue': np.nan,
                        'optimal_lag': np.nan,
                        'is_causal': False,
                    })

    return pd.DataFrame(results)


def _granger_test(
    source: np.ndarray,
    target: np.ndarray,
    max_lag: int = 5,
) -> Dict[str, Any]:
    """
    Test if source Granger-causes target.

    Uses F-test comparing restricted (only target lags) vs
    unrestricted (target + source lags) models.
    """
    n = len(target)

    if n < max_lag + 10:
        return {'f_stat': np.nan, 'p_value': np.nan, 'optimal_lag': np.nan}

    best_result = {'f_stat': 0, 'p_value': 1.0, 'optimal_lag': 1}

    for lag in range(1, max_lag + 1):
        # Build lagged matrices
        Y = target[lag:]
        n_obs = len(Y)

        # Restricted model: only target lags
        X_restricted = np.column_stack([
            target[lag - i - 1:n - i - 1] for i in range(lag)
        ])

        # Unrestricted model: target + source lags
        X_unrestricted = np.column_stack([
            X_restricted,
            *[source[lag - i - 1:n - i - 1] for i in range(lag)]
        ])

        # Add constant
        X_restricted = np.column_stack([np.ones(n_obs), X_restricted])
        X_unrestricted = np.column_stack([np.ones(n_obs), X_unrestricted])

        try:
            # Fit restricted model
            beta_r = np.linalg.lstsq(X_restricted, Y, rcond=None)[0]
            resid_r = Y - X_restricted @ beta_r
            ssr_r = np.sum(resid_r ** 2)

            # Fit unrestricted model
            beta_u = np.linalg.lstsq(X_unrestricted, Y, rcond=None)[0]
            resid_u = Y - X_unrestricted @ beta_u
            ssr_u = np.sum(resid_u ** 2)

            # F-test
            df1 = lag  # Number of restrictions
            df2 = n_obs - 2 * lag - 1  # Degrees of freedom

            if df2 <= 0 or ssr_u <= 0:
                continue

            f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
            p_value = 1 - stats.f.cdf(f_stat, df1, df2)

            if p_value < best_result['p_value']:
                best_result = {
                    'f_stat': float(f_stat),
                    'p_value': float(p_value),
                    'optimal_lag': lag,
                }

        except Exception:
            continue

    return best_result


def _ensure_stationarity(series: np.ndarray) -> np.ndarray:
    """Difference series if non-stationary."""
    from scipy.stats import normaltest

    # Simple check: if variance of differences is much smaller, difference
    diff = np.diff(series)
    if np.var(diff) < 0.5 * np.var(series):
        return diff
    return series
