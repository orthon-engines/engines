"""
Fleet Status Engine

Computes aggregate status metrics across all entities for each window.

Output is indexed by WINDOW only (not entity).

Metrics:
    - Regime counts and percentages
    - Mean/max baseline_distance across fleet
    - Count of diverging entities (hd_slope > threshold)

Config:
    diverging_threshold: float (default 0) - hd_slope threshold for "diverging"
"""

import numpy as np
from typing import Dict, Any, List, Optional
import polars as pl


def compute(
    dynamics_data: Dict[str, pl.DataFrame] = None,
    dynamics_df: pl.DataFrame = None,
    config: Dict[str, Any] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Compute fleet-level status metrics per window.

    Args:
        dynamics_data: Dict of entity_id -> dynamics DataFrame
        dynamics_df: Alternative: single DataFrame with entity_id column
        config: Contains diverging_threshold

    Returns:
        List of dicts, one per window
    """
    if config is None:
        config = {}

    diverging_threshold = config.get('diverging_threshold', 0)

    # Convert single DataFrame to dict format
    if dynamics_df is not None and dynamics_data is None:
        dynamics_data = _df_to_entity_dict(dynamics_df)

    if not dynamics_data:
        return [{'error': 'No dynamics data provided'}]

    # Get all windows
    all_windows = set()
    for df in dynamics_data.values():
        if 'window' in df.columns:
            all_windows.update(df['window'].to_list())

    if not all_windows:
        return [{'error': 'No window column found in data'}]

    results = []
    n_entities = len(dynamics_data)

    for window in sorted(all_windows):
        window_data = {
            'window': window,
            'total_entities': n_entities,
        }

        # Collect metrics across entities
        regime_counts = {}
        baseline_distances = []
        hd_slopes = []
        entities_at_window = 0

        for entity_id, df in dynamics_data.items():
            if 'window' not in df.columns:
                continue

            row = df.filter(pl.col('window') == window)
            if len(row) == 0:
                continue

            entities_at_window += 1

            # Regime count
            if 'regime' in row.columns:
                regime = row['regime'][0]
                if regime is not None:
                    regime_counts[regime] = regime_counts.get(regime, 0) + 1

            # Baseline distance
            if 'baseline_distance' in row.columns:
                bd = row['baseline_distance'][0]
                if bd is not None and not np.isnan(bd):
                    baseline_distances.append(bd)

            # HD slope
            if 'hd_slope' in row.columns:
                hds = row['hd_slope'][0]
                if hds is not None and not np.isnan(hds):
                    hd_slopes.append(hds)

        window_data['entities_at_window'] = entities_at_window

        # Regime statistics
        for regime, count in regime_counts.items():
            window_data[f'regime_{regime}_count'] = count
            window_data[f'regime_{regime}_pct'] = count / entities_at_window if entities_at_window > 0 else 0

        # Baseline distance statistics
        if baseline_distances:
            window_data['mean_baseline_distance'] = float(np.mean(baseline_distances))
            window_data['max_baseline_distance'] = float(np.max(baseline_distances))
            window_data['std_baseline_distance'] = float(np.std(baseline_distances))
            window_data['median_baseline_distance'] = float(np.median(baseline_distances))
        else:
            window_data['mean_baseline_distance'] = None
            window_data['max_baseline_distance'] = None
            window_data['std_baseline_distance'] = None
            window_data['median_baseline_distance'] = None

        # HD slope statistics
        if hd_slopes:
            window_data['mean_hd_slope'] = float(np.mean(hd_slopes))
            window_data['max_hd_slope'] = float(np.max(hd_slopes))
            window_data['diverging_count'] = sum(1 for s in hd_slopes if s > diverging_threshold)
            window_data['diverging_pct'] = window_data['diverging_count'] / entities_at_window if entities_at_window > 0 else 0
        else:
            window_data['mean_hd_slope'] = None
            window_data['max_hd_slope'] = None
            window_data['diverging_count'] = None
            window_data['diverging_pct'] = None

        results.append(window_data)

    return results


def _df_to_entity_dict(df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    """Convert single DataFrame with entity_id to dict of DataFrames."""
    if 'entity_id' not in df.columns:
        return {'default': df}

    entities = df['entity_id'].unique().to_list()
    return {eid: df.filter(pl.col('entity_id') == eid) for eid in entities}
