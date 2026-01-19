"""
PRISM Adaptive Windowing — Dynamic window sizes based on entity observation count.

The Problem:
    Hardcoded window sizes (252, 126, 63) from financial domain don't work for
    industrial data where entity lifespans vary wildly (150-500 cycles).

The Solution:
    Compute window size per-entity to guarantee minimum number of windows.

Usage:
    from prism.utils.adaptive_windows import (
        compute_adaptive_windows,
        get_entity_window_configs,
        create_entity_windows,
    )
    
    # Get config for single entity
    config = compute_adaptive_windows(entity_length=192)
    # {'size': 38, 'stride': 19, 'n_windows': 9}
    
    # Get configs for all entities
    configs = get_entity_window_configs(observations)
    
    # Create windows for a DataFrame
    windowed = create_entity_windows(observations, configs)
"""

from typing import Dict, List, Optional, Any
import logging

import polars as pl

logger = logging.getLogger(__name__)


def compute_adaptive_windows(
    entity_length: int,
    min_windows: int = 4,
    max_window_size: int = 100,
    min_window_size: int = 20,
    overlap_ratio: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute window parameters based on entity observation count.
    
    Guarantees at least `min_windows` windows per entity, subject to
    minimum window size constraints.
    
    Args:
        entity_length: Number of observations for this entity
        min_windows: Target minimum number of windows to generate
        max_window_size: Cap on window size (even for long entities)
        min_window_size: Floor on window size (need enough for metrics)
        overlap_ratio: Fraction of overlap between windows (0.5 = 50%)
    
    Returns:
        dict with:
            - size: window size in observations
            - stride: step between windows
            - n_windows: expected number of windows
            - entity_length: input entity length
            - sufficient: bool, True if enough data for min_windows
    
    Examples:
        >>> compute_adaptive_windows(192)
        {'size': 38, 'stride': 19, 'n_windows': 9, ...}
        
        >>> compute_adaptive_windows(500)
        {'size': 100, 'stride': 50, 'n_windows': 9, ...}
        
        >>> compute_adaptive_windows(50)  # Short entity
        {'size': 20, 'stride': 10, 'n_windows': 4, 'sufficient': False, ...}
    """
    # Edge case: very short entity
    if entity_length < min_window_size:
        return {
            'size': entity_length,
            'stride': max(1, entity_length // 2),
            'n_windows': 1,
            'entity_length': entity_length,
            'sufficient': False,
        }
    
    # Target window size to get min_windows with overlap
    # Formula: n_windows = (length - size) / stride + 1
    # With stride = size * (1 - overlap), solve for size:
    # n = (L - size) / (size * (1-o)) + 1
    # Approximation: size ≈ L / (n * (2-o) - 1) for 50% overlap
    
    effective_factor = min_windows * (1 + (1 - overlap_ratio)) - 1
    target_size = int(entity_length / max(1, effective_factor))
    
    # Apply bounds
    size = max(min_window_size, min(max_window_size, target_size))
    
    # Compute stride for desired overlap
    stride = max(1, int(size * (1 - overlap_ratio)))
    
    # Calculate actual number of windows
    n_windows = max(1, (entity_length - size) // stride + 1)
    
    # If we don't get enough windows, try reducing size
    attempts = 0
    while n_windows < min_windows and size > min_window_size and attempts < 10:
        size = max(min_window_size, size - 5)
        stride = max(1, int(size * (1 - overlap_ratio)))
        n_windows = max(1, (entity_length - size) // stride + 1)
        attempts += 1
    
    return {
        'size': size,
        'stride': stride,
        'n_windows': n_windows,
        'entity_length': entity_length,
        'sufficient': n_windows >= min_windows,
    }


def get_entity_window_configs(
    observations: pl.DataFrame,
    entity_col: str = 'entity_id',
    signal_col: str = 'signal_id',
    **kwargs,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute adaptive window config for each entity.
    
    Uses the MINIMUM observation count across all signals for each entity
    to ensure all signals get the same window structure.
    
    Args:
        observations: DataFrame with entity_col and signal_col
        entity_col: Column identifying entities
        signal_col: Column identifying signals
        **kwargs: Passed to compute_adaptive_windows()
    
    Returns:
        dict mapping entity_id → window config
    """
    # Get observation count per entity-signal, then take min per entity
    # This ensures all signals for an entity use the same windows
    entity_lengths = (
        observations
        .group_by([entity_col, signal_col])
        .agg(pl.count().alias('n_obs'))
        .group_by(entity_col)
        .agg(pl.col('n_obs').min().alias('min_obs'))
    )
    
    configs = {}
    for row in entity_lengths.iter_rows(named=True):
        entity_id = row[entity_col]
        n_obs = row['min_obs']
        configs[entity_id] = compute_adaptive_windows(n_obs, **kwargs)
    
    # Log summary
    sizes = [c['size'] for c in configs.values()]
    n_windows = [c['n_windows'] for c in configs.values()]
    insufficient = sum(1 for c in configs.values() if not c['sufficient'])
    
    logger.info(
        f"Adaptive windows for {len(configs)} entities: "
        f"size {min(sizes)}-{max(sizes)}, "
        f"windows {min(n_windows)}-{max(n_windows)}, "
        f"{insufficient} with insufficient data"
    )
    
    return configs


def create_entity_windows(
    df: pl.DataFrame,
    entity_col: str = 'entity_id',
    timestamp_col: str = 'timestamp',
    window_config: Optional[Dict[str, Any]] = None,
) -> pl.DataFrame:
    """
    Add window_id column to DataFrame using adaptive or fixed windowing.
    
    Args:
        df: DataFrame with entity_col and timestamp_col
        entity_col: Column identifying entities
        timestamp_col: Column with timestamps (float)
        window_config: Optional fixed config. If None, uses adaptive.
    
    Returns:
        DataFrame with window_id added
    """
    if window_config is not None:
        # Fixed windowing
        stride = window_config.get('stride', 25)
        return df.with_columns(
            ((pl.col(timestamp_col)
              .rank(method='ordinal')
              .over(entity_col) - 1) // stride)
            .cast(pl.Int64)
            .alias('window_id')
        )
    
    # Adaptive windowing - need per-entity configs
    configs = get_entity_window_configs(df, entity_col)
    
    # Apply per-entity stride
    # This is a bit tricky in Polars - we'll use a join approach
    config_df = pl.DataFrame([
        {'entity_id': eid, 'stride': cfg['stride']}
        for eid, cfg in configs.items()
    ])
    
    return (
        df
        .join(config_df, on=entity_col, how='left')
        .with_columns(
            ((pl.col(timestamp_col)
              .rank(method='ordinal')
              .over(entity_col) - 1) // pl.col('stride'))
            .cast(pl.Int64)
            .alias('window_id')
        )
        .drop('stride')
    )


def validate_window_coverage(
    observations: pl.DataFrame,
    vector: pl.DataFrame,
    entity_col: str = 'entity_id',
) -> Dict[str, Any]:
    """
    Validate that vector output has windows for all entities.
    
    Returns dict with validation results and any issues found.
    """
    obs_entities = set(observations[entity_col].unique().to_list())
    vec_entities = set(vector[entity_col].unique().to_list())
    
    missing = obs_entities - vec_entities
    
    # Windows per entity
    windows_per_entity = (
        vector
        .group_by(entity_col)
        .agg(pl.col('window_id').n_unique().alias('n_windows'))
    )
    
    min_windows = windows_per_entity['n_windows'].min()
    max_windows = windows_per_entity['n_windows'].max()
    
    return {
        'obs_entities': len(obs_entities),
        'vec_entities': len(vec_entities),
        'missing_entities': list(missing),
        'min_windows_per_entity': min_windows,
        'max_windows_per_entity': max_windows,
        'valid': len(missing) == 0 and min_windows >= 1,
    }


# =============================================================================
# Convenience function for entry points
# =============================================================================

def ensure_sufficient_windows(
    observations: pl.DataFrame,
    min_windows: int = 4,
    entity_col: str = 'entity_id',
) -> bool:
    """
    Check if adaptive windowing will produce sufficient windows.
    
    Returns True if all entities will get at least min_windows.
    Logs warnings for problematic entities.
    """
    configs = get_entity_window_configs(
        observations, 
        entity_col=entity_col,
        min_windows=min_windows,
    )
    
    insufficient = [
        (eid, cfg) for eid, cfg in configs.items() 
        if not cfg['sufficient']
    ]
    
    if insufficient:
        logger.warning(
            f"{len(insufficient)} entities have insufficient data for {min_windows} windows:"
        )
        for eid, cfg in insufficient[:5]:  # Show first 5
            logger.warning(
                f"  {eid}: {cfg['entity_length']} obs → {cfg['n_windows']} windows"
            )
        if len(insufficient) > 5:
            logger.warning(f"  ... and {len(insufficient) - 5} more")
    
    return len(insufficient) == 0
