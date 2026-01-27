"""
Divergence Engine

Measures distribution divergence between signal pairs.

CANONICAL INTERFACE:
    def compute(observations: pd.DataFrame) -> pd.DataFrame
    Input:  [entity_id, signal_id, I, y]
    Output: [entity_id, signal_a, signal_b, kl_divergence, js_divergence,
             wasserstein_distance]

Divergence measures how different two probability distributions are:
- KL divergence: Asymmetric, measures information lost
- JS divergence: Symmetric, bounded [0, 1]
- Wasserstein: Earth mover's distance, considers geometry
"""

ENGINE_TYPE = 'pairwise'

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple


def compute(
    observations: pd.DataFrame,
    bins: int = 50,
) -> pd.DataFrame:
    """
    Compute distribution divergence for all signal pairs.

    CANONICAL INTERFACE:
        Input:  observations [entity_id, signal_id, I, y]
        Output: primitives [entity_id, signal_a, signal_b, kl_divergence,
                           js_divergence, wasserstein_distance]

    Parameters
    ----------
    observations : pd.DataFrame
        Must have columns: entity_id, signal_id, I, y
    bins : int, optional
        Number of bins for histogram estimation (default: 50)

    Returns
    -------
    pd.DataFrame
        Pairwise divergence metrics
    """
    results = []

    for entity_id, entity_group in observations.groupby('entity_id'):
        signals = entity_group['signal_id'].unique()

        if len(signals) < 2:
            continue

        # Get series for each signal
        series = {}
        for sig in signals:
            s = entity_group[entity_group['signal_id'] == sig].sort_values('I')['y'].values
            if len(s) >= 20:
                series[sig] = s

        if len(series) < 2:
            continue

        signal_list = list(series.keys())

        for i in range(len(signal_list)):
            for j in range(i + 1, len(signal_list)):
                sig_a = signal_list[i]
                sig_b = signal_list[j]

                try:
                    x = series[sig_a]
                    y = series[sig_b]

                    # Compute divergences
                    kl_div, js_div = _histogram_divergences(x, y, bins)
                    wasserstein = _wasserstein_distance(x, y)

                    results.append({
                        'entity_id': entity_id,
                        'signal_a': sig_a,
                        'signal_b': sig_b,
                        'kl_divergence': float(kl_div),
                        'js_divergence': float(js_div),
                        'wasserstein_distance': float(wasserstein),
                    })

                except Exception:
                    results.append({
                        'entity_id': entity_id,
                        'signal_a': sig_a,
                        'signal_b': sig_b,
                        'kl_divergence': np.nan,
                        'js_divergence': np.nan,
                        'wasserstein_distance': np.nan,
                    })

    return pd.DataFrame(results)


def _histogram_divergences(x: np.ndarray, y: np.ndarray, bins: int) -> Tuple[float, float]:
    """
    Compute KL and JS divergence using histogram estimation.

    Returns (kl_divergence, js_divergence)
    """
    # Create common bin edges
    combined = np.concatenate([x, y])
    bin_edges = np.histogram_bin_edges(combined, bins=bins)

    # Compute histograms
    p_hist, _ = np.histogram(x, bins=bin_edges, density=True)
    q_hist, _ = np.histogram(y, bins=bin_edges, density=True)

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p = p_hist + epsilon
    q = q_hist + epsilon

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    # KL divergence: D_KL(P || Q) = sum(P * log(P/Q))
    kl_pq = np.sum(p * np.log(p / q))
    kl_qp = np.sum(q * np.log(q / p))
    kl_div = (kl_pq + kl_qp) / 2  # Symmetric version

    # JS divergence: JS = (D_KL(P||M) + D_KL(Q||M)) / 2 where M = (P+Q)/2
    m = (p + q) / 2
    js_div = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))

    return kl_div, js_div


def _wasserstein_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Wasserstein distance (Earth Mover's Distance).

    Uses scipy's implementation which computes the 1D Wasserstein distance
    between two empirical distributions.
    """
    return stats.wasserstein_distance(x, y)


def compute_signal_divergence(
    observations: pd.DataFrame,
    reference_signal: str,
    bins: int = 50,
) -> pd.DataFrame:
    """
    Compute divergence of all signals from a reference signal.

    CANONICAL INTERFACE:
        Input:  observations [entity_id, signal_id, I, y]
        Output: primitives [entity_id, signal_id, kl_from_ref, js_from_ref,
                           wasserstein_from_ref]

    Parameters
    ----------
    observations : pd.DataFrame
        Must have columns: entity_id, signal_id, I, y
    reference_signal : str
        Signal ID to use as reference
    bins : int, optional
        Number of bins for histogram estimation (default: 50)

    Returns
    -------
    pd.DataFrame
        Divergence from reference per signal
    """
    results = []

    for entity_id, entity_group in observations.groupby('entity_id'):
        # Get reference signal
        ref_data = entity_group[entity_group['signal_id'] == reference_signal]
        if len(ref_data) < 20:
            continue

        ref_values = ref_data.sort_values('I')['y'].values

        # Compute divergence for each other signal
        for signal_id in entity_group['signal_id'].unique():
            if signal_id == reference_signal:
                results.append({
                    'entity_id': entity_id,
                    'signal_id': signal_id,
                    'kl_from_ref': 0.0,
                    'js_from_ref': 0.0,
                    'wasserstein_from_ref': 0.0,
                })
                continue

            sig_data = entity_group[entity_group['signal_id'] == signal_id]
            if len(sig_data) < 20:
                results.append({
                    'entity_id': entity_id,
                    'signal_id': signal_id,
                    'kl_from_ref': np.nan,
                    'js_from_ref': np.nan,
                    'wasserstein_from_ref': np.nan,
                })
                continue

            try:
                sig_values = sig_data.sort_values('I')['y'].values
                kl_div, js_div = _histogram_divergences(sig_values, ref_values, bins)
                wasserstein = _wasserstein_distance(sig_values, ref_values)

                results.append({
                    'entity_id': entity_id,
                    'signal_id': signal_id,
                    'kl_from_ref': float(kl_div),
                    'js_from_ref': float(js_div),
                    'wasserstein_from_ref': float(wasserstein),
                })

            except Exception:
                results.append({
                    'entity_id': entity_id,
                    'signal_id': signal_id,
                    'kl_from_ref': np.nan,
                    'js_from_ref': np.nan,
                    'wasserstein_from_ref': np.nan,
                })

    return pd.DataFrame(results)
