"""
Correlated Trajectories Engine

Finds pairs of entities with similar baseline_distance trajectories.

Definition:
    Uses Pearson correlation on baseline_distance time series.
    High correlation indicates entities are degrading together.

Use cases:
    - Identifying related failures
    - Finding entities affected by same root cause
    - Cohort discovery

Config:
    correlation_threshold: float (default 0.8) - Minimum correlation to report
    min_common_windows: int (default 10) - Minimum overlapping windows
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import polars as pl


def compute(
    dynamics_data: Dict[str, pl.DataFrame] = None,
    dynamics_df: pl.DataFrame = None,
    config: Dict[str, Any] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Find pairs of entities with correlated baseline_distance trajectories.

    Args:
        dynamics_data: Dict of entity_id -> dynamics DataFrame
        dynamics_df: Alternative: single DataFrame with entity_id column
        config: Contains correlation_threshold, min_common_windows

    Returns:
        Dict with correlated pairs and summary
    """
    if config is None:
        config = {}

    threshold = config.get('correlation_threshold', 0.8)
    min_windows = config.get('min_common_windows', 10)

    # Convert single DataFrame to dict format
    if dynamics_df is not None and dynamics_data is None:
        dynamics_data = _df_to_entity_dict(dynamics_df)

    if not dynamics_data:
        return {'error': 'No dynamics data provided', 'correlated_pairs': []}

    # Build trajectory dict: entity_id -> {window: baseline_distance}
    trajectories = {}

    for entity_id, df in dynamics_data.items():
        if 'window' not in df.columns or 'baseline_distance' not in df.columns:
            continue

        traj = {}
        for row in df.iter_rows(named=True):
            window = row['window']
            bd = row['baseline_distance']
            if bd is not None and not np.isnan(bd):
                traj[window] = bd

        if len(traj) >= min_windows:
            trajectories[entity_id] = traj

    # Find correlated pairs
    entities = list(trajectories.keys())
    correlated_pairs = []
    all_correlations = []

    for i, e1 in enumerate(entities):
        for e2 in entities[i+1:]:
            # Find common windows
            common = set(trajectories[e1].keys()) & set(trajectories[e2].keys())

            if len(common) < min_windows:
                continue

            # Extract aligned values
            common_sorted = sorted(common)
            v1 = np.array([trajectories[e1][w] for w in common_sorted])
            v2 = np.array([trajectories[e2][w] for w in common_sorted])

            # Compute correlation
            if np.std(v1) > 0 and np.std(v2) > 0:
                corr = np.corrcoef(v1, v2)[0, 1]
            else:
                corr = 0.0

            all_correlations.append(corr)

            if corr > threshold:
                correlated_pairs.append({
                    'entity_1': e1,
                    'entity_2': e2,
                    'correlation': float(corr),
                    'n_common_windows': len(common),
                })

    # Sort by correlation descending
    correlated_pairs.sort(key=lambda x: x['correlation'], reverse=True)

    return {
        'threshold': threshold,
        'min_common_windows': min_windows,
        'n_entities_analyzed': len(entities),
        'n_pairs_analyzed': len(all_correlations),
        'n_correlated_pairs': len(correlated_pairs),
        'correlated_pairs': correlated_pairs,
        'mean_correlation': float(np.mean(all_correlations)) if all_correlations else None,
        'max_correlation': float(np.max(all_correlations)) if all_correlations else None,
        'most_correlated_pair': correlated_pairs[0] if correlated_pairs else None,
    }


def _df_to_entity_dict(df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    """Convert single DataFrame with entity_id to dict of DataFrames."""
    if 'entity_id' not in df.columns:
        return {'default': df}

    entities = df['entity_id'].unique().to_list()
    return {eid: df.filter(pl.col('entity_id') == eid) for eid in entities}
