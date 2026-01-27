"""
Basin Analysis Engine

Determines basin membership based on deviation from baseline behavior.

CANONICAL INTERFACE:
    def compute(observations: pd.DataFrame) -> pd.DataFrame
    Input:  [entity_id, signal_id, I, y]
    Output: [entity_id, signal_id, basin, basin_stability, transition_prob,
             dist_from_baseline]

Basin analysis determines whether the system has escaped its normal
operating regime by comparing current behavior to baseline.
"""

ENGINE_TYPE = 'signal'

import numpy as np
import pandas as pd
from typing import Dict, Any

try:
    from sklearn.neighbors import BallTree
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def compute(
    observations: pd.DataFrame,
    baseline_fraction: float = 0.3,
    escape_sigma: float = 3.0,
    embedding_dim: int = 3,
    delay: int = 1,
) -> pd.DataFrame:
    """
    Compute basin membership for all signals.

    CANONICAL INTERFACE:
        Input:  observations [entity_id, signal_id, I, y]
        Output: primitives [entity_id, signal_id, basin, basin_stability,
                           transition_prob, dist_from_baseline]

    Parameters
    ----------
    observations : pd.DataFrame
        Must have columns: entity_id, signal_id, I, y
    baseline_fraction : float, optional
        Fraction of data to use as baseline (default: 0.3)
    escape_sigma : float, optional
        Number of std devs for escape threshold (default: 3.0)
    embedding_dim : int, optional
        Embedding dimension for phase space (default: 3)
    delay : int, optional
        Delay for embedding (default: 1)

    Returns
    -------
    pd.DataFrame
        Basin metrics per signal
    """
    if not HAS_SKLEARN:
        return pd.DataFrame(columns=[
            'entity_id', 'signal_id', 'basin', 'basin_stability',
            'transition_prob', 'dist_from_baseline'
        ])

    results = []

    for (entity_id, signal_id), group in observations.groupby(['entity_id', 'signal_id']):
        y = group.sort_values('I')['y'].values

        if len(y) < 50:
            results.append({
                'entity_id': entity_id,
                'signal_id': signal_id,
                'basin': -1,
                'basin_stability': np.nan,
                'transition_prob': np.nan,
                'dist_from_baseline': np.nan,
            })
            continue

        try:
            result = _compute_basin(y, baseline_fraction, escape_sigma, embedding_dim, delay)
            results.append({
                'entity_id': entity_id,
                'signal_id': signal_id,
                **result
            })
        except Exception:
            results.append({
                'entity_id': entity_id,
                'signal_id': signal_id,
                'basin': -1,
                'basin_stability': np.nan,
                'transition_prob': np.nan,
                'dist_from_baseline': np.nan,
            })

    return pd.DataFrame(results)


def _compute_basin(
    signal: np.ndarray,
    baseline_fraction: float,
    escape_sigma: float,
    embedding_dim: int,
    delay: int,
) -> Dict[str, Any]:
    """Compute basin metrics for a single signal."""
    n = len(signal)

    # Create embedding
    embedded = _embed(signal, embedding_dim, delay)
    if len(embedded) < 20:
        return {
            'basin': -1,
            'basin_stability': np.nan,
            'transition_prob': np.nan,
            'dist_from_baseline': np.nan,
        }

    # Split into baseline and current
    n_baseline = int(len(embedded) * baseline_fraction)
    if n_baseline < 10:
        n_baseline = min(10, len(embedded) // 2)

    baseline_embedded = embedded[:n_baseline]
    current_embedded = embedded[n_baseline:]

    if len(current_embedded) < 5:
        current_embedded = embedded[-10:]

    # Compute baseline statistics
    baseline_centroid = np.mean(baseline_embedded, axis=0)
    baseline_dists = np.linalg.norm(baseline_embedded - baseline_centroid, axis=1)
    baseline_mean = np.mean(baseline_dists)
    baseline_std = np.std(baseline_dists)
    if baseline_std < 1e-10:
        baseline_std = 1e-10

    # Escape threshold
    escape_threshold = baseline_mean + escape_sigma * baseline_std

    # Current state analysis
    current_centroid = np.mean(current_embedded, axis=0)
    dist_from_baseline = np.linalg.norm(current_centroid - baseline_centroid)

    # Basin membership
    if dist_from_baseline < escape_threshold:
        basin = 0  # Still in baseline basin
    else:
        basin = 1  # Escaped to new basin

    # Basin stability: how deep in basin (negative = deep, positive = near boundary)
    stability = (escape_threshold - dist_from_baseline) / escape_threshold

    # Transition probability: based on distance to boundary
    boundary_distance = escape_threshold - dist_from_baseline
    transition_prob = 1 / (1 + np.exp(boundary_distance / baseline_std))

    return {
        'basin': int(basin),
        'basin_stability': float(stability),
        'transition_prob': float(transition_prob),
        'dist_from_baseline': float(dist_from_baseline),
    }


def _embed(signal: np.ndarray, dim: int, delay: int) -> np.ndarray:
    """Create Takens embedding."""
    n = len(signal)
    n_points = n - (dim - 1) * delay

    if n_points <= 0:
        return np.array([]).reshape(0, dim)

    embedded = np.zeros((n_points, dim))
    for i in range(dim):
        embedded[:, i] = signal[i * delay : i * delay + n_points]

    return embedded
