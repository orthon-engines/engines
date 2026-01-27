"""
Copula Dependence Engine

Measures tail dependence and non-linear dependence structure.

CANONICAL INTERFACE:
    def compute(observations: pd.DataFrame) -> pd.DataFrame
    Input:  [entity_id, signal_id, I, y]
    Output: [entity_id, signal_a, signal_b, lower_tail, upper_tail,
             kendall_tau, spearman_rho]

Copula analysis captures:
- Tail dependence (co-movement in extremes)
- Rank-based correlation (robust to outliers)
"""

ENGINE_TYPE = 'pairwise'

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple


def compute(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute copula-based dependence for all signal pairs.

    CANONICAL INTERFACE:
        Input:  observations [entity_id, signal_id, I, y]
        Output: primitives [entity_id, signal_a, signal_b, lower_tail,
                           upper_tail, kendall_tau, spearman_rho]

    Parameters
    ----------
    observations : pd.DataFrame
        Must have columns: entity_id, signal_id, I, y

    Returns
    -------
    pd.DataFrame
        Pairwise copula dependence metrics
    """
    results = []

    for entity_id, entity_group in observations.groupby('entity_id'):
        # Pivot to wide format
        try:
            wide = entity_group.pivot(index='I', columns='signal_id', values='y')
            wide = wide.sort_index().dropna()
        except Exception:
            wide = entity_group.groupby(['I', 'signal_id'])['y'].mean().unstack()
            wide = wide.sort_index().dropna()

        signals = list(wide.columns)
        n_signals = len(signals)

        if len(wide) < 20 or n_signals < 2:
            continue

        # Transform to uniform marginals (probability integral transform)
        wide_uniform = wide.rank(pct=True)

        # Compute pairwise dependence
        for i in range(n_signals):
            for j in range(i + 1, n_signals):
                sig_a = signals[i]
                sig_b = signals[j]

                try:
                    u = wide_uniform.iloc[:, i].values
                    v = wide_uniform.iloc[:, j].values

                    # Tail dependence
                    lower_tail, upper_tail = _empirical_tail_dependence(u, v)

                    # Rank correlations
                    tau = _kendall_tau(wide.iloc[:, i].values, wide.iloc[:, j].values)
                    rho = _spearman_rho(wide.iloc[:, i].values, wide.iloc[:, j].values)

                    results.append({
                        'entity_id': entity_id,
                        'signal_a': sig_a,
                        'signal_b': sig_b,
                        'lower_tail': float(lower_tail),
                        'upper_tail': float(upper_tail),
                        'kendall_tau': float(tau),
                        'spearman_rho': float(rho),
                    })

                except Exception:
                    results.append({
                        'entity_id': entity_id,
                        'signal_a': sig_a,
                        'signal_b': sig_b,
                        'lower_tail': np.nan,
                        'upper_tail': np.nan,
                        'kendall_tau': np.nan,
                        'spearman_rho': np.nan,
                    })

    return pd.DataFrame(results)


def _empirical_tail_dependence(u: np.ndarray, v: np.ndarray) -> Tuple[float, float]:
    """
    Estimate tail dependence from empirical copula.

    Returns (lower_tail, upper_tail) dependence coefficients.
    """
    thresholds = [0.05, 0.10, 0.15]

    lower_deps = []
    upper_deps = []

    for q in thresholds:
        # Lower tail: P(V <= q | U <= q)
        mask_lower = u <= q
        if mask_lower.sum() > 0:
            lower_deps.append((v[mask_lower] <= q).mean())

        # Upper tail: P(V > 1-q | U > 1-q)
        mask_upper = u >= (1 - q)
        if mask_upper.sum() > 0:
            upper_deps.append((v[mask_upper] >= (1 - q)).mean())

    lower_tail = np.mean(lower_deps) if lower_deps else 0.0
    upper_tail = np.mean(upper_deps) if upper_deps else 0.0

    return lower_tail, upper_tail


def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Kendall's tau rank correlation."""
    tau, _ = stats.kendalltau(x, y)
    return tau if not np.isnan(tau) else 0.0


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman's rho rank correlation."""
    rho, _ = stats.spearmanr(x, y)
    return rho if not np.isnan(rho) else 0.0
