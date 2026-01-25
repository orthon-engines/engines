"""
Entity Ranking Engine

Ranks entities by deviation from baseline at each window.

Output:
    Rankings sorted by baseline_distance descending (worst first).

Use cases:
    - "Which entities are furthest from baseline?"
    - "Top 10 entities by degradation"
"""

import numpy as np
from typing import Dict, Any, List, Optional
import polars as pl


def compute(
    dynamics_data: Dict[str, pl.DataFrame] = None,
    dynamics_df: pl.DataFrame = None,
    window: int = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Rank entities by baseline_distance at a given window.

    Args:
        dynamics_data: Dict of entity_id -> dynamics DataFrame
        dynamics_df: Alternative: single DataFrame with entity_id column
        window: Which window to rank (if None, uses last window)

    Returns:
        Dict with ranked entity list and summary stats
    """
    # Convert single DataFrame to dict format
    if dynamics_df is not None and dynamics_data is None:
        dynamics_data = _df_to_entity_dict(dynamics_df)

    if not dynamics_data:
        return {'error': 'No dynamics data provided', 'rankings': []}

    # Determine window
    if window is None:
        # Find latest window across all entities
        all_windows = set()
        for df in dynamics_data.values():
            if 'window' in df.columns:
                all_windows.update(df['window'].to_list())
        window = max(all_windows) if all_windows else None

    if window is None:
        return {'error': 'No window specified or found', 'rankings': []}

    # Collect rankings
    rankings = []

    for entity_id, df in dynamics_data.items():
        if 'window' not in df.columns:
            continue

        row = df.filter(pl.col('window') == window)
        if len(row) == 0:
            continue

        entry = {'entity_id': entity_id}

        if 'baseline_distance' in row.columns:
            bd = row['baseline_distance'][0]
            entry['baseline_distance'] = float(bd) if bd is not None and not np.isnan(bd) else None
        else:
            entry['baseline_distance'] = None

        if 'hd_slope' in row.columns:
            hds = row['hd_slope'][0]
            entry['hd_slope'] = float(hds) if hds is not None and not np.isnan(hds) else None
        else:
            entry['hd_slope'] = None

        if 'regime' in row.columns:
            regime = row['regime'][0]
            entry['regime'] = int(regime) if regime is not None else None
        else:
            entry['regime'] = None

        rankings.append(entry)

    # Sort by baseline_distance descending (worst first)
    rankings.sort(key=lambda x: x['baseline_distance'] if x['baseline_distance'] is not None else -np.inf, reverse=True)

    # Add rank
    for i, entry in enumerate(rankings):
        entry['rank'] = i + 1

    return {
        'window': window,
        'n_entities': len(rankings),
        'rankings': rankings,
        'worst_entity': rankings[0]['entity_id'] if rankings else None,
        'best_entity': rankings[-1]['entity_id'] if rankings else None,
    }


def compute_all_windows(
    dynamics_data: Dict[str, pl.DataFrame] = None,
    dynamics_df: pl.DataFrame = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Compute rankings for all windows.
    """
    if dynamics_df is not None and dynamics_data is None:
        dynamics_data = _df_to_entity_dict(dynamics_df)

    if not dynamics_data:
        return []

    # Get all windows
    all_windows = set()
    for df in dynamics_data.values():
        if 'window' in df.columns:
            all_windows.update(df['window'].to_list())

    results = []
    for window in sorted(all_windows):
        result = compute(dynamics_data=dynamics_data, window=window)
        results.append(result)

    return results


def _df_to_entity_dict(df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    """Convert single DataFrame with entity_id to dict of DataFrames."""
    if 'entity_id' not in df.columns:
        return {'default': df}

    entities = df['entity_id'].unique().to_list()
    return {eid: df.filter(pl.col('entity_id') == eid) for eid in entities}
