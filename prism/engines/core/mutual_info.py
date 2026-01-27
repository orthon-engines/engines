"""
Mutual Information Engine

Measures nonlinear statistical dependence between signals.

CANONICAL INTERFACE:
    def compute(observations: pd.DataFrame) -> pd.DataFrame
    Input:  [entity_id, signal_id, I, y]
    Output: [entity_id, signal_a, signal_b, mutual_info, normalized_mi]

Mutual information captures any statistical dependence, not just linear.
High MI = signals are predictive of each other.
"""

ENGINE_TYPE = 'pairwise'

import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, Any


def compute(
    observations: pd.DataFrame,
    bins: int = 20,
    max_pairs: int = 50,
) -> pd.DataFrame:
    """
    Compute mutual information for all signal pairs.

    CANONICAL INTERFACE:
        Input:  observations [entity_id, signal_id, I, y]
        Output: primitives [entity_id, signal_a, signal_b, mutual_info,
                           normalized_mi]

    Parameters
    ----------
    observations : pd.DataFrame
        Must have columns: entity_id, signal_id, I, y
    bins : int, optional
        Number of bins for discretization (default: 20)
    max_pairs : int, optional
        Maximum number of signals to compare (default: 50)

    Returns
    -------
    pd.DataFrame
        Pairwise mutual information values
    """
    results = []

    for entity_id, entity_group in observations.groupby('entity_id'):
        signals = list(entity_group['signal_id'].unique())[:max_pairs]

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

        # Compute pairwise MI
        for i, sig_a in enumerate(signals):
            for sig_b in signals[i + 1:]:
                if sig_a not in wide.columns or sig_b not in wide.columns:
                    continue

                try:
                    mi = _mutual_information(
                        wide[sig_a].values,
                        wide[sig_b].values,
                        bins=bins
                    )

                    # Normalized MI: MI / min(H(X), H(Y))
                    h_a = _entropy(wide[sig_a].values, bins)
                    h_b = _entropy(wide[sig_b].values, bins)
                    max_mi = min(h_a, h_b)
                    normalized = mi / max_mi if max_mi > 0 else 0

                    results.append({
                        'entity_id': entity_id,
                        'signal_a': sig_a,
                        'signal_b': sig_b,
                        'mutual_info': mi,
                        'normalized_mi': normalized,
                    })
                except Exception:
                    results.append({
                        'entity_id': entity_id,
                        'signal_a': sig_a,
                        'signal_b': sig_b,
                        'mutual_info': np.nan,
                        'normalized_mi': np.nan,
                    })

    return pd.DataFrame(results)


def _mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """
    Compute mutual information I(X;Y) using histogram estimation.

    I(X;Y) = H(X) + H(Y) - H(X,Y)
    """
    # Discretize
    x_binned = _discretize(x, bins)
    y_binned = _discretize(y, bins)

    # Compute entropies
    h_x = _entropy_binned(x_binned)
    h_y = _entropy_binned(y_binned)
    h_xy = _joint_entropy(x_binned, y_binned)

    return max(0, h_x + h_y - h_xy)


def _discretize(x: np.ndarray, bins: int) -> np.ndarray:
    """Discretize continuous values into bins."""
    x_min, x_max = np.min(x), np.max(x)
    if x_max == x_min:
        return np.zeros(len(x), dtype=int)

    bin_edges = np.linspace(x_min, x_max, bins + 1)
    return np.clip(np.digitize(x, bin_edges[1:-1]), 0, bins - 1)


def _entropy(x: np.ndarray, bins: int) -> float:
    """Compute entropy H(X) of continuous variable."""
    x_binned = _discretize(x, bins)
    return _entropy_binned(x_binned)


def _entropy_binned(x: np.ndarray) -> float:
    """Compute entropy H(X) of discretized variable."""
    counts = Counter(x)
    n = len(x)
    probs = np.array([c / n for c in counts.values()])
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def _joint_entropy(x: np.ndarray, y: np.ndarray) -> float:
    """Compute joint entropy H(X,Y)."""
    joint = list(zip(x, y))
    counts = Counter(joint)
    n = len(joint)
    probs = np.array([c / n for c in counts.values()])
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))
