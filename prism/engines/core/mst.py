"""
Minimum Spanning Tree (MST) Engine

Builds the minimum spanning tree of signals in behavioral space.

CANONICAL INTERFACE:
    def compute(observations: pd.DataFrame) -> pd.DataFrame
    Input:  [entity_id, signal_id, I, y]
    Output: [entity_id, signal_a, signal_b, mst_edge_weight, is_mst_edge]

MST reveals structural relationships:
- Contracting MST (shorter edges) = behavioral convergence
- Expanding MST (longer edges) = behavioral divergence
- High-degree nodes = "bridge" signals connecting clusters
"""

ENGINE_TYPE = 'pairwise'

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import Counter
from typing import Dict, Any, List, Tuple


def compute(
    observations: pd.DataFrame,
    distance_metric: str = "euclidean",
) -> pd.DataFrame:
    """
    Compute Minimum Spanning Tree for all entities.

    CANONICAL INTERFACE:
        Input:  observations [entity_id, signal_id, I, y]
        Output: primitives [entity_id, signal_a, signal_b, mst_edge_weight]

    Parameters
    ----------
    observations : pd.DataFrame
        Must have columns: entity_id, signal_id, I, y
    distance_metric : str, optional
        Distance metric for scipy.pdist (default: "euclidean")

    Returns
    -------
    pd.DataFrame
        MST edges with weights
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

        if n_signals < 2:
            continue

        try:
            # Each signal is a row, time points are features
            X = wide.T.values  # (n_signals, n_timepoints)

            # Compute pairwise distances
            distances = pdist(X, metric=distance_metric)
            distance_matrix = squareform(distances)

            # Build MST
            mst_sparse = minimum_spanning_tree(distance_matrix)
            mst_matrix = mst_sparse.toarray()

            # Extract edges
            rows, cols = np.where(mst_matrix > 0)
            for i, j in zip(rows, cols):
                results.append({
                    'entity_id': entity_id,
                    'signal_a': signals[i],
                    'signal_b': signals[j],
                    'mst_edge_weight': float(mst_matrix[i, j]),
                })

        except Exception:
            # If MST fails, return empty for this entity
            continue

    return pd.DataFrame(results)


def compute_with_stats(
    observations: pd.DataFrame,
    distance_metric: str = "euclidean",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute MST edges and per-signal statistics.

    CANONICAL INTERFACE:
        Input:  observations [entity_id, signal_id, I, y]
        Output: (edges_df, stats_df)
                edges_df: [entity_id, signal_a, signal_b, mst_edge_weight]
                stats_df: [entity_id, signal_id, mst_degree, is_hub, is_leaf]

    Parameters
    ----------
    observations : pd.DataFrame
        Must have columns: entity_id, signal_id, I, y
    distance_metric : str, optional
        Distance metric (default: "euclidean")

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (edges, stats)
    """
    edge_results = []
    stat_results = []

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

        if n_signals < 2:
            continue

        try:
            X = wide.T.values
            distances = pdist(X, metric=distance_metric)
            distance_matrix = squareform(distances)

            mst_sparse = minimum_spanning_tree(distance_matrix)
            mst_matrix = mst_sparse.toarray()

            # Extract edges
            edges = []
            rows, cols = np.where(mst_matrix > 0)
            for i, j in zip(rows, cols):
                edges.append((i, j, mst_matrix[i, j]))
                edge_results.append({
                    'entity_id': entity_id,
                    'signal_a': signals[i],
                    'signal_b': signals[j],
                    'mst_edge_weight': float(mst_matrix[i, j]),
                })

            # Compute node degrees
            degrees = _compute_node_degrees(edges, n_signals)

            for idx, signal_id in enumerate(signals):
                degree = degrees.get(idx, 0)
                stat_results.append({
                    'entity_id': entity_id,
                    'signal_id': signal_id,
                    'mst_degree': int(degree),
                    'is_hub': degree >= 3,
                    'is_leaf': degree == 1,
                })

        except Exception:
            continue

    return pd.DataFrame(edge_results), pd.DataFrame(stat_results)


def _compute_node_degrees(edges: List[Tuple[int, int, float]], n_nodes: int) -> Dict[int, int]:
    """Compute degree of each node in the MST."""
    degree_count = Counter()
    for i, j, _ in edges:
        degree_count[i] += 1
        degree_count[j] += 1

    return {node: degree_count.get(node, 0) for node in range(n_nodes)}
