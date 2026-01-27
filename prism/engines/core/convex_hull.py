"""
Convex Hull Engine

Measures the geometric extent of signals in behavioral space.

CANONICAL INTERFACE:
    def compute(observations: pd.DataFrame) -> pd.DataFrame
    Input:  [entity_id, signal_id, I, y]
    Output: [entity_id, signal_id, centroid_distance, is_boundary]

Convex hull analysis reveals:
- Contracting volume = behavioral convergence
- Expanding volume = behavioral divergence
- Boundary signals = extreme behavioral signatures
"""

ENGINE_TYPE = 'signal'

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from typing import Dict, Any


def compute(
    observations: pd.DataFrame,
    max_hull_dims: int = 6,
) -> pd.DataFrame:
    """
    Compute convex hull metrics for all signals within each entity.

    CANONICAL INTERFACE:
        Input:  observations [entity_id, signal_id, I, y]
        Output: primitives [entity_id, signal_id, centroid_distance,
                           normalized_distance, is_boundary]

    Parameters
    ----------
    observations : pd.DataFrame
        Must have columns: entity_id, signal_id, I, y
    max_hull_dims : int, optional
        Maximum dimensions for hull computation (default: 6)

    Returns
    -------
    pd.DataFrame
        Convex hull metrics per signal
    """
    results = []

    for entity_id, entity_group in observations.groupby('entity_id'):
        # Pivot to wide format: rows=I (time), cols=signal_id, values=y
        try:
            wide = entity_group.pivot(index='I', columns='signal_id', values='y')
            wide = wide.sort_index().dropna()
        except Exception:
            wide = entity_group.groupby(['I', 'signal_id'])['y'].mean().unstack()
            wide = wide.sort_index().dropna()

        signals = list(wide.columns)
        n_signals = len(signals)

        if n_signals < 4:
            for signal_id in signals:
                results.append({
                    'entity_id': entity_id,
                    'signal_id': signal_id,
                    'centroid_distance': np.nan,
                    'normalized_distance': np.nan,
                    'is_boundary': False,
                })
            continue

        try:
            # Each signal is a row, time points are features
            X = wide.T.values  # (n_signals, n_timepoints)
            n_dims = X.shape[1]

            # Compute centroid and distances
            centroid = X.mean(axis=0)
            distances = np.linalg.norm(X - centroid, axis=1)

            # Normalize distances
            max_dist = distances.max() if distances.max() > 0 else 1.0
            normalized = distances / max_dist

            # For hull computation, reduce dimensionality if needed
            boundary_indices = set()

            if n_dims <= max_hull_dims and n_signals > n_dims:
                # Can compute hull directly
                try:
                    hull = ConvexHull(X)
                    boundary_indices = set(hull.vertices)
                except Exception:
                    pass
            else:
                # Project to lower dimensions for hull
                n_components = min(max_hull_dims, n_dims, n_signals - 1)
                if n_components >= 2:
                    try:
                        pca = PCA(n_components=n_components)
                        X_projected = pca.fit_transform(X)
                        hull = ConvexHull(X_projected)
                        boundary_indices = set(hull.vertices)
                    except Exception:
                        pass

            for i, signal_id in enumerate(signals):
                results.append({
                    'entity_id': entity_id,
                    'signal_id': signal_id,
                    'centroid_distance': float(distances[i]),
                    'normalized_distance': float(normalized[i]),
                    'is_boundary': i in boundary_indices,
                })

        except Exception:
            for signal_id in signals:
                results.append({
                    'entity_id': entity_id,
                    'signal_id': signal_id,
                    'centroid_distance': np.nan,
                    'normalized_distance': np.nan,
                    'is_boundary': False,
                })

    return pd.DataFrame(results)


def compute_entity_stats(
    observations: pd.DataFrame,
    max_hull_dims: int = 6,
) -> pd.DataFrame:
    """
    Compute entity-level convex hull statistics.

    CANONICAL INTERFACE:
        Input:  observations [entity_id, signal_id, I, y]
        Output: primitives [entity_id, hull_volume, hull_area, n_vertices,
                           centroid_avg_distance, centroid_max_distance]

    Parameters
    ----------
    observations : pd.DataFrame
        Must have columns: entity_id, signal_id, I, y
    max_hull_dims : int, optional
        Maximum dimensions for hull computation (default: 6)

    Returns
    -------
    pd.DataFrame
        Entity-level hull statistics
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

        if n_signals < 4:
            results.append({
                'entity_id': entity_id,
                'hull_volume': np.nan,
                'hull_area': np.nan,
                'n_vertices': 0,
                'centroid_avg_distance': np.nan,
                'centroid_max_distance': np.nan,
            })
            continue

        try:
            X = wide.T.values
            n_dims = X.shape[1]

            # Compute centroid stats
            centroid = X.mean(axis=0)
            distances = np.linalg.norm(X - centroid, axis=1)

            hull_volume = np.nan
            hull_area = np.nan
            n_vertices = 0

            # For hull computation, reduce dimensionality if needed
            if n_dims <= max_hull_dims and n_signals > n_dims:
                try:
                    hull = ConvexHull(X)
                    hull_volume = hull.volume
                    hull_area = hull.area
                    n_vertices = len(hull.vertices)
                except Exception:
                    pass
            else:
                n_components = min(max_hull_dims, n_dims, n_signals - 1)
                if n_components >= 2:
                    try:
                        pca = PCA(n_components=n_components)
                        X_projected = pca.fit_transform(X)
                        hull = ConvexHull(X_projected)
                        hull_volume = hull.volume
                        hull_area = hull.area
                        n_vertices = len(hull.vertices)
                    except Exception:
                        pass

            results.append({
                'entity_id': entity_id,
                'hull_volume': float(hull_volume) if not np.isnan(hull_volume) else np.nan,
                'hull_area': float(hull_area) if not np.isnan(hull_area) else np.nan,
                'n_vertices': int(n_vertices),
                'centroid_avg_distance': float(np.mean(distances)),
                'centroid_max_distance': float(np.max(distances)),
            })

        except Exception:
            results.append({
                'entity_id': entity_id,
                'hull_volume': np.nan,
                'hull_area': np.nan,
                'n_vertices': 0,
                'centroid_avg_distance': np.nan,
                'centroid_max_distance': np.nan,
            })

    return pd.DataFrame(results)
