"""
PRISM Windowing Utilities — Float-based, time-scale agnostic.

NO datetime parsing. Just numeric operations on timestamp floats.

Strategies:
    count — fixed number of observations per window
    range — fixed timestamp range per window

Usage:
    from prism.utils.windowing import create_windows, get_window_bounds
    
    # Add window_id to observations
    df = create_windows(observations, config)
    
    # Get window metadata
    bounds = get_window_bounds(df)
"""

import polars as pl
from typing import Dict, Any, Optional


def create_windows(
    df: pl.DataFrame,
    config: Dict[str, Any],
    entity_col: str = 'entity_id',
    timestamp_col: str = 'timestamp',
) -> pl.DataFrame:
    """
    Add window_id column based on windowing strategy.
    
    Args:
        df: DataFrame with entity_col and timestamp_col
        config: Domain config with 'windowing' section
        entity_col: Column identifying entities
        timestamp_col: Column with float timestamps
    
    Returns:
        DataFrame with 'window_id' column added
    
    Strategies:
        count: Windows contain fixed number of observations
               window_id = (rank - 1) // stride
        
        range: Windows span fixed timestamp range
               window_id = timestamp // stride
    """
    windowing = config.get('windowing', {})
    strategy = windowing.get('strategy', 'count')
    size = windowing.get('size', 50)
    stride = windowing.get('stride', size // 2)  # Default 50% overlap
    
    if strategy == 'count':
        # Window by observation count within each entity
        # rank() gives 1-indexed position, so subtract 1
        return df.with_columns(
            ((pl.col(timestamp_col)
              .rank(method='ordinal')
              .over(entity_col) - 1) // stride)
            .cast(pl.Int64)
            .alias('window_id')
        )
    
    elif strategy == 'range':
        # Window by timestamp range (globally, not per-entity)
        # Useful when timestamp is absolute (e.g., seconds since start)
        return df.with_columns(
            (pl.col(timestamp_col) // stride)
            .cast(pl.Int64)
            .alias('window_id')
        )
    
    else:
        raise ValueError(f"Unknown windowing strategy: {strategy}")


def get_window_bounds(
    df: pl.DataFrame,
    entity_col: str = 'entity_id',
    signal_col: str = 'signal_id',
    timestamp_col: str = 'timestamp',
) -> pl.DataFrame:
    """
    Get window metadata (start, end, count) for each entity-signal-window.
    
    Returns DataFrame with:
        entity_id, signal_id, window_id, window_start, window_end, n_obs
    """
    return (
        df
        .group_by([entity_col, signal_col, 'window_id'])
        .agg([
            pl.col(timestamp_col).min().alias('window_start'),
            pl.col(timestamp_col).max().alias('window_end'),
            pl.col(timestamp_col).count().alias('n_obs'),
        ])
        .sort([entity_col, signal_col, 'window_id'])
    )


def filter_valid_windows(
    df: pl.DataFrame,
    min_observations: int = 20,
) -> pl.DataFrame:
    """
    Filter to windows with sufficient observations.
    
    Must be called AFTER aggregation to window level.
    Expects 'n_obs' column from get_window_bounds or similar.
    """
    if 'n_obs' not in df.columns:
        raise ValueError("DataFrame must have 'n_obs' column")
    
    return df.filter(pl.col('n_obs') >= min_observations)


def window_aggregate(
    df: pl.DataFrame,
    value_col: str = 'value',
    entity_col: str = 'entity_id',
    signal_col: str = 'signal_id',
    timestamp_col: str = 'timestamp',
    min_observations: int = 20,
) -> pl.DataFrame:
    """
    Aggregate observations to window level with basic stats.
    
    This is the standard pre-processing before engine computation.
    
    Returns DataFrame with one row per entity-signal-window:
        entity_id, signal_id, window_id, window_start, window_end, 
        n_obs, values (list)
    """
    return (
        df
        .group_by([entity_col, signal_col, 'window_id'])
        .agg([
            pl.col(timestamp_col).min().alias('window_start'),
            pl.col(timestamp_col).max().alias('window_end'),
            pl.col(timestamp_col).count().alias('n_obs'),
            pl.col(value_col).alias('values'),  # Keep as list for engines
        ])
        .filter(pl.col('n_obs') >= min_observations)
        .sort([entity_col, signal_col, 'window_id'])
    )


# =============================================================================
# Convenience functions for common patterns
# =============================================================================

def windows_from_config(
    observations: pl.DataFrame,
    config: Dict[str, Any],
) -> pl.DataFrame:
    """
    One-liner: Load observations, add windows, aggregate.
    
    Usage:
        windowed = windows_from_config(observations, domain_config)
        # Now ready for engine computation
    """
    min_obs = config.get('windowing', {}).get('min_observations', 20)
    
    df = create_windows(observations, config)
    return window_aggregate(df, min_observations=min_obs)


def estimate_window_count(
    n_observations: int,
    config: Dict[str, Any],
) -> int:
    """
    Estimate number of windows that will be created.
    
    Useful for progress bars and memory estimation.
    """
    windowing = config.get('windowing', {})
    size = windowing.get('size', 50)
    stride = windowing.get('stride', size // 2)
    
    if stride <= 0:
        return 1
    
    return max(1, (n_observations - size) // stride + 1)
