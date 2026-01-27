"""
Transfer Entropy Engine

Measures directional information flow between signals.

CANONICAL INTERFACE:
    def compute(observations: pd.DataFrame) -> pd.DataFrame
    Input:  [entity_id, signal_id, I, y]
    Output: [entity_id, source_id, target_id, transfer_entropy, normalized_te]

Transfer entropy from X to Y measures how much knowing the past of X
reduces uncertainty about Y, beyond what knowing the past of Y provides.
"""

ENGINE_TYPE = 'pairwise'

import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, Any, Optional


def compute(
    observations: pd.DataFrame,
    lag: int = 1,
    bins: int = 10,
) -> pd.DataFrame:
    """
    Compute transfer entropy for all signal pairs.

    CANONICAL INTERFACE:
        Input:  observations [entity_id, signal_id, I, y]
        Output: primitives [entity_id, source_id, target_id, transfer_entropy,
                           normalized_te]

    Parameters
    ----------
    observations : pd.DataFrame
        Must have columns: entity_id, signal_id, I, y
    lag : int, optional
        Time lag for transfer entropy (default: 1)
    bins : int, optional
        Number of bins for discretization (default: 10)

    Returns
    -------
    pd.DataFrame
        Pairwise transfer entropy values
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

        if len(wide) < lag + 10:
            continue

        # Test all pairs
        for source in signals:
            for target in signals:
                if source == target:
                    continue

                if source not in wide.columns or target not in wide.columns:
                    continue

                try:
                    te = _compute_transfer_entropy(
                        wide[source].values,
                        wide[target].values,
                        lag=lag,
                        bins=bins
                    )

                    # Compute max possible TE for normalization
                    max_te = np.log2(bins)
                    normalized_te = te / max_te if max_te > 0 else 0

                    results.append({
                        'entity_id': entity_id,
                        'source_id': source,
                        'target_id': target,
                        'transfer_entropy': te,
                        'normalized_te': normalized_te,
                    })
                except Exception:
                    results.append({
                        'entity_id': entity_id,
                        'source_id': source,
                        'target_id': target,
                        'transfer_entropy': np.nan,
                        'normalized_te': np.nan,
                    })

    return pd.DataFrame(results)


def _compute_transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    bins: int = 10,
) -> float:
    """
    Compute transfer entropy from source to target.

    TE(X->Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})

    Uses histogram-based estimation.
    """
    n = len(target)

    if n < lag + 10:
        return np.nan

    # Discretize
    source_binned = _discretize(source, bins)
    target_binned = _discretize(target, bins)

    # Build joint distributions
    # Y_t, Y_{t-1}, X_{t-1}
    y_t = target_binned[lag:]
    y_past = target_binned[:-lag]
    x_past = source_binned[:-lag]

    # Compute entropies using counts
    # H(Y_t | Y_{t-1}) = H(Y_t, Y_{t-1}) - H(Y_{t-1})
    h_y_ypast = _joint_entropy(y_t, y_past, bins)
    h_ypast = _entropy(y_past, bins)
    h_y_given_ypast = h_y_ypast - h_ypast

    # H(Y_t | Y_{t-1}, X_{t-1}) = H(Y_t, Y_{t-1}, X_{t-1}) - H(Y_{t-1}, X_{t-1})
    h_y_ypast_xpast = _triple_entropy(y_t, y_past, x_past, bins)
    h_ypast_xpast = _joint_entropy(y_past, x_past, bins)
    h_y_given_ypast_xpast = h_y_ypast_xpast - h_ypast_xpast

    # Transfer entropy
    te = h_y_given_ypast - h_y_given_ypast_xpast

    return max(0, te)  # TE should be non-negative


def _discretize(x: np.ndarray, bins: int) -> np.ndarray:
    """Discretize continuous values into bins."""
    x_min, x_max = np.min(x), np.max(x)
    if x_max == x_min:
        return np.zeros(len(x), dtype=int)

    # Equal-width bins
    bin_edges = np.linspace(x_min, x_max, bins + 1)
    return np.clip(np.digitize(x, bin_edges[1:-1]), 0, bins - 1)


def _entropy(x: np.ndarray, bins: int) -> float:
    """Compute entropy H(X)."""
    counts = Counter(x)
    n = len(x)
    probs = np.array([c / n for c in counts.values()])
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def _joint_entropy(x: np.ndarray, y: np.ndarray, bins: int) -> float:
    """Compute joint entropy H(X, Y)."""
    joint = list(zip(x, y))
    counts = Counter(joint)
    n = len(joint)
    probs = np.array([c / n for c in counts.values()])
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def _triple_entropy(x: np.ndarray, y: np.ndarray, z: np.ndarray, bins: int) -> float:
    """Compute joint entropy H(X, Y, Z)."""
    joint = list(zip(x, y, z))
    counts = Counter(joint)
    n = len(joint)
    probs = np.array([c / n for c in counts.values()])
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))
