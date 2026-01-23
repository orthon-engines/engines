"""
prism/modules/discovery.py - Cohort discovery algorithms

Pure functions, no I/O. Discovers behavioral cohorts from state features.
"""

import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def discover_cohorts(
    df: pl.DataFrame,
    entity_col: str = "entity_id",
    max_clusters: int = 5,
    method: str = "kmeans",
) -> pl.DataFrame:
    """
    Discover behavioral cohorts from entity features.

    Args:
        df: Input DataFrame with features per entity
        entity_col: Entity identifier column
        max_clusters: Maximum clusters to try
        method: Clustering method (kmeans, hierarchical)

    Returns:
        DataFrame with cohort assignments per entity
    """
    # Get feature columns
    feature_cols = [
        c for c in df.columns
        if c not in [entity_col, 'timestamp', 'time_col']
        and df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
    ]

    if not feature_cols:
        # No features, assign all to cohort 0
        entities = df[entity_col].unique()
        return pl.DataFrame({
            entity_col: entities,
            'cohort': [0] * len(entities),
            'cohort_confidence': [0.0] * len(entities),
        })

    # Aggregate features per entity (mean across time)
    entity_features = df.group_by(entity_col).agg([
        pl.col(c).mean().alias(c) for c in feature_cols
    ])

    entities = entity_features[entity_col].to_list()
    X = entity_features.select(feature_cols).to_numpy()

    # Remove columns that are all NaN
    col_mask = ~np.isnan(X).all(axis=0)
    X = X[:, col_mask]

    # Remove rows with NaN
    nan_mask = np.isnan(X).any(axis=1)
    X_clean = X[~nan_mask]
    entities_clean = [e for e, m in zip(entities, nan_mask) if not m]

    if len(X_clean) < 2:
        return pl.DataFrame({
            entity_col: entities,
            'cohort': [0] * len(entities),
            'cohort_confidence': [0.0] * len(entities),
        })

    # Standardize
    X_std = (X_clean - np.mean(X_clean, axis=0)) / (np.std(X_clean, axis=0) + 1e-10)

    # Find optimal k
    best_k, best_score, best_labels = _find_optimal_k(X_std, max_clusters)

    # Build result
    cohort_map = dict(zip(entities_clean, best_labels))

    results = []
    for entity in entities:
        if entity in cohort_map:
            results.append({
                entity_col: entity,
                'cohort': int(cohort_map[entity]),
                'cohort_confidence': float(best_score) if best_score > 0 else 0.0,
            })
        else:
            # Entity had NaN features
            results.append({
                entity_col: entity,
                'cohort': -1,
                'cohort_confidence': 0.0,
            })

    result_df = pl.DataFrame(results)

    # Add cohort summary
    cohort_counts = result_df.group_by('cohort').len().rename({'len': 'cohort_size'})
    result_df = result_df.join(cohort_counts, on='cohort', how='left')

    return result_df


def _find_optimal_k(
    X: np.ndarray,
    max_k: int,
) -> Tuple[int, float, np.ndarray]:
    """Find optimal number of clusters using silhouette score."""
    n_samples = len(X)
    max_k = min(max_k, n_samples - 1)

    if max_k < 2:
        return 1, 0.0, np.zeros(n_samples, dtype=int)

    best_k = 2
    best_score = -1
    best_labels = None

    for k in range(2, max_k + 1):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X)

            if len(np.unique(labels)) < 2:
                continue

            score = silhouette_score(X, labels)

            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        except Exception:
            continue

    if best_labels is None:
        best_labels = np.zeros(n_samples, dtype=int)

    return best_k, best_score, best_labels


def compute_cohort_summary(
    cohort_df: pl.DataFrame,
    state_df: pl.DataFrame,
    entity_col: str = "entity_id",
) -> pl.DataFrame:
    """
    Compute summary statistics per cohort.

    Returns DataFrame with cohort-level metrics.
    """
    # Join cohort assignments with state features
    joined = state_df.join(cohort_df.select([entity_col, 'cohort']), on=entity_col)

    # Get numeric columns
    numeric_cols = [
        c for c in state_df.columns
        if c not in [entity_col, 'timestamp']
        and state_df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
    ]

    if not numeric_cols:
        return cohort_df.group_by('cohort').len().rename({'len': 'n_entities'})

    # Aggregate by cohort
    summary = joined.group_by('cohort').agg([
        pl.col(entity_col).n_unique().alias('n_entities'),
        *[pl.col(c).mean().alias(f'{c}_mean') for c in numeric_cols[:10]],
        *[pl.col(c).std().alias(f'{c}_std') for c in numeric_cols[:10]],
    ])

    return summary
