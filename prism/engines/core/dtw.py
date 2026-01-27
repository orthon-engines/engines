"""
Dynamic Time Warping (DTW) Engine

Measures similarity between signals allowing for temporal shifts.

CANONICAL INTERFACE:
    def compute(observations: pd.DataFrame) -> pd.DataFrame
    Input:  [entity_id, signal_id, I, y]
    Output: [entity_id, signal_a, signal_b, dtw_distance, dtw_normalized]

DTW finds optimal alignment between sequences, useful when signals
have similar shapes but are shifted or stretched in time.
"""

ENGINE_TYPE = 'pairwise'

import numpy as np
import pandas as pd
from typing import Dict, Any


def compute(
    observations: pd.DataFrame,
    max_pairs: int = 50,
) -> pd.DataFrame:
    """
    Compute DTW distance for all signal pairs.

    CANONICAL INTERFACE:
        Input:  observations [entity_id, signal_id, I, y]
        Output: primitives [entity_id, signal_a, signal_b, dtw_distance,
                           dtw_normalized]

    Parameters
    ----------
    observations : pd.DataFrame
        Must have columns: entity_id, signal_id, I, y
    max_pairs : int, optional
        Maximum number of signals to compare (default: 50)

    Returns
    -------
    pd.DataFrame
        Pairwise DTW distances
    """
    results = []

    for entity_id, entity_group in observations.groupby('entity_id'):
        signals = list(entity_group['signal_id'].unique())[:max_pairs]

        if len(signals) < 2:
            continue

        # Get series for each signal
        series = {}
        for sig in signals:
            s = entity_group[entity_group['signal_id'] == sig].sort_values('I')['y'].values
            series[sig] = s

        # Compute pairwise DTW
        for i, sig_a in enumerate(signals):
            for sig_b in signals[i + 1:]:
                try:
                    distance = _dtw_distance(series[sig_a], series[sig_b])

                    # Normalize by path length
                    n = len(series[sig_a]) + len(series[sig_b])
                    normalized = distance / n if n > 0 else 0

                    results.append({
                        'entity_id': entity_id,
                        'signal_a': sig_a,
                        'signal_b': sig_b,
                        'dtw_distance': distance,
                        'dtw_normalized': normalized,
                    })
                except Exception:
                    results.append({
                        'entity_id': entity_id,
                        'signal_a': sig_a,
                        'signal_b': sig_b,
                        'dtw_distance': np.nan,
                        'dtw_normalized': np.nan,
                    })

    return pd.DataFrame(results)


def _dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute DTW distance between two sequences.

    Uses dynamic programming to find minimum cost alignment.
    """
    n, m = len(x), len(y)

    if n == 0 or m == 0:
        return np.nan

    # Initialize cost matrix
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    # Fill cost matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dtw[i, j] = cost + min(
                dtw[i - 1, j],      # insertion
                dtw[i, j - 1],      # deletion
                dtw[i - 1, j - 1]   # match
            )

    return float(dtw[n, m])


def _dtw_distance_window(x: np.ndarray, y: np.ndarray, window: int = None) -> float:
    """
    DTW with Sakoe-Chiba band constraint for speedup.

    Only allows warping within `window` steps of diagonal.
    """
    n, m = len(x), len(y)

    if n == 0 or m == 0:
        return np.nan

    if window is None:
        window = max(n, m)

    # Initialize cost matrix
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    # Fill cost matrix with band constraint
    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m + 1, i + window + 1)

        for j in range(j_start, j_end):
            cost = abs(x[i - 1] - y[j - 1])
            dtw[i, j] = cost + min(
                dtw[i - 1, j],
                dtw[i, j - 1],
                dtw[i - 1, j - 1]
            )

    return float(dtw[n, m])
