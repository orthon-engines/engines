"""
Leading Indicator Engine

Finds entities that showed divergence earliest.

Definition:
    An entity is "diverging" when its hd_slope exceeds a threshold.
    The "leading indicator" is the entity that crossed this threshold first.

Use cases:
    - Early warning detection
    - Identifying which entities show problems first
    - Predictive maintenance prioritization

Config:
    diverging_threshold: float (default 0) - hd_slope threshold
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
    Find entities that showed divergence earliest.

    Args:
        dynamics_data: Dict of entity_id -> dynamics DataFrame
        dynamics_df: Alternative: single DataFrame with entity_id column
        config: Contains diverging_threshold

    Returns:
        Dict with leading indicators and summary
    """
    if config is None:
        config = {}

    threshold = config.get('diverging_threshold', 0)

    # Convert single DataFrame to dict format
    if dynamics_df is not None and dynamics_data is None:
        dynamics_data = _df_to_entity_dict(dynamics_df)

    if not dynamics_data:
        return {'error': 'No dynamics data provided', 'leading_indicators': []}

    # Find first divergence window for each entity
    first_divergence = {}

    for entity_id, df in dynamics_data.items():
        if 'window' not in df.columns or 'hd_slope' not in df.columns:
            continue

        # Filter to diverging windows
        diverging = df.filter(pl.col('hd_slope') > threshold)

        if len(diverging) > 0:
            first_window = diverging['window'].min()
            first_divergence[entity_id] = int(first_window)

    # Sort by earliest divergence
    sorted_indicators = sorted(first_divergence.items(), key=lambda x: x[1])

    # Build result
    leading_indicators = [
        {
            'entity_id': entity_id,
            'first_diverging_window': window,
            'rank': i + 1,
        }
        for i, (entity_id, window) in enumerate(sorted_indicators)
    ]

    # Summary statistics
    n_never_diverged = len(dynamics_data) - len(first_divergence)
    divergence_windows = [w for _, w in sorted_indicators]

    return {
        'threshold': threshold,
        'n_entities': len(dynamics_data),
        'n_diverged': len(first_divergence),
        'n_never_diverged': n_never_diverged,
        'leading_indicators': leading_indicators,
        'first_entity': sorted_indicators[0][0] if sorted_indicators else None,
        'first_window': sorted_indicators[0][1] if sorted_indicators else None,
        'mean_first_divergence_window': float(np.mean(divergence_windows)) if divergence_windows else None,
        'median_first_divergence_window': float(np.median(divergence_windows)) if divergence_windows else None,
    }


def _df_to_entity_dict(df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    """Convert single DataFrame with entity_id to dict of DataFrames."""
    if 'entity_id' not in df.columns:
        return {'default': df}

    entities = df['entity_id'].unique().to_list()
    return {eid: df.filter(pl.col('entity_id') == eid) for eid in entities}
