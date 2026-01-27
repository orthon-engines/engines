"""
Clustering Engine

Cluster time points into regimes based on signal values.

CANONICAL INTERFACE:
    def compute(observations: pd.DataFrame) -> pd.DataFrame
    Input:  [entity_id, signal_id, I, y]
    Output: [entity_id, I, regime_id, regime_centroid_dist]

Uses KMeans or DBSCAN to identify regimes in multivariate signal space.
"""

ENGINE_TYPE = 'point'

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Optional


def compute(
    observations: pd.DataFrame,
    n_clusters: int = 3,
    method: str = 'kmeans',
) -> pd.DataFrame:
    """
    Cluster time points into regimes.

    CANONICAL INTERFACE:
        Input:  observations [entity_id, signal_id, I, y]
        Output: primitives [entity_id, I, regime_id, regime_centroid_dist]

    Parameters
    ----------
    observations : pd.DataFrame
        Must have columns: entity_id, signal_id, I, y
    n_clusters : int, optional
        Number of clusters for KMeans (default: 3)
    method : str, optional
        Clustering method: 'kmeans' or 'dbscan' (default: 'kmeans')

    Returns
    -------
    pd.DataFrame
        Regime assignments per time point
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

        if len(wide) < n_clusters + 5:
            continue

        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(wide.values)

        # Cluster
        try:
            if method == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = model.fit_predict(X)
                # Distance to assigned centroid
                distances = np.min(
                    np.linalg.norm(X[:, np.newaxis] - model.cluster_centers_, axis=2),
                    axis=1
                )
            elif method == 'dbscan':
                model = DBSCAN(eps=0.5, min_samples=5)
                labels = model.fit_predict(X)
                distances = np.zeros(len(labels))  # DBSCAN doesn't have centroids
            else:
                raise ValueError(f"Unknown method: {method}")

            for i, (idx, label) in enumerate(zip(wide.index, labels)):
                results.append({
                    'entity_id': entity_id,
                    'I': idx,
                    'regime_id': int(label),
                    'regime_centroid_dist': float(distances[i]),
                })

        except Exception:
            # If clustering fails, assign all to regime 0
            for idx in wide.index:
                results.append({
                    'entity_id': entity_id,
                    'I': idx,
                    'regime_id': 0,
                    'regime_centroid_dist': np.nan,
                })

    return pd.DataFrame(results)


def compute_regime_stats(
    observations: pd.DataFrame,
    regimes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute statistics for each regime.

    Parameters
    ----------
    observations : pd.DataFrame
        Original observations
    regimes : pd.DataFrame
        Output from compute() with regime assignments

    Returns
    -------
    pd.DataFrame
        Statistics per regime [entity_id, regime_id, n_points, duration_mean, ...]
    """
    results = []

    merged = observations.merge(regimes, on=['entity_id', 'I'])

    for (entity_id, regime_id), group in merged.groupby(['entity_id', 'regime_id']):
        n_points = len(group)

        # Compute regime duration (consecutive time points in regime)
        I_values = group['I'].sort_values().values
        if len(I_values) > 1:
            diffs = np.diff(I_values)
            # Count runs of consecutive points
            breaks = np.where(diffs > 1.5 * np.median(diffs))[0]
            n_segments = len(breaks) + 1
            duration_mean = n_points / n_segments
        else:
            duration_mean = 1.0

        results.append({
            'entity_id': entity_id,
            'regime_id': regime_id,
            'n_points': n_points,
            'duration_mean': duration_mean,
            'pct_time': n_points / len(merged[merged['entity_id'] == entity_id]),
        })

    return pd.DataFrame(results)
