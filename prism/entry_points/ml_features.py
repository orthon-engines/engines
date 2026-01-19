"""
PRISM ML Features — Generate ML-ready feature parquet.

Combines all PRISM layers into one denormalized feature table.
One row per entity (engine, bearing, unit). Ready for any ML framework.

Reads:  data/vector.parquet, data/geometry.parquet, data/state.parquet
Writes: data/ml_features.parquet

Usage:
    python -m prism.entry_points.ml_features
    python -m prism.entry_points.ml_features --target RUL
    python -m prism.entry_points.ml_features --testing --limit 10
"""

import argparse
import polars as pl
from pathlib import Path
from typing import List, Optional

from prism.db.parquet_store import (
    get_path,
    OBSERVATIONS,
    VECTOR,
    GEOMETRY,
    STATE,
    ML_FEATURES,
)


# =============================================================================
# FEATURE AGGREGATION
# =============================================================================

def aggregate_vector_features(vector: pl.DataFrame, entity_col: str) -> pl.DataFrame:
    """
    Aggregate 51 vector metrics per entity.
    
    For each metric, compute: mean, std, last (captures trajectory endpoint).
    This gives ~153 features from vector layer.
    """
    # Identify metric columns (exclude metadata)
    exclude_cols = {entity_col, 'signal_id', 'signal_type', 'timestamp', 'window_end', 'cohort_id'}
    metrics = [c for c in vector.columns if c not in exclude_cols and not c.startswith('_')]
    
    if not metrics:
        raise ValueError(f"No metric columns found in vector.parquet. Columns: {vector.columns}")
    
    aggs = []
    for metric in metrics:
        aggs.extend([
            pl.col(metric).mean().alias(f'vector_{metric}_mean'),
            pl.col(metric).std().alias(f'vector_{metric}_std'),
            pl.col(metric).last().alias(f'vector_{metric}_last'),
        ])
    
    return vector.group_by(entity_col).agg(aggs)


def aggregate_geometry_features(geometry: pl.DataFrame, entity_col: str) -> pl.DataFrame:
    """
    Aggregate geometry metrics per entity.
    
    Geometry captures cohort structure: PCA, clustering, MST, LOF, etc.
    """
    exclude_cols = {entity_col, 'cohort_id', 'timestamp', 'window_end', 'signal_id'}
    metrics = [c for c in geometry.columns if c not in exclude_cols and not c.startswith('_')]
    
    if not metrics:
        print("Warning: No metric columns found in geometry.parquet")
        return pl.DataFrame({entity_col: []})
    
    aggs = []
    for metric in metrics:
        aggs.extend([
            pl.col(metric).mean().alias(f'geometry_{metric}_mean'),
            pl.col(metric).std().alias(f'geometry_{metric}_std'),
            pl.col(metric).last().alias(f'geometry_{metric}_last'),
        ])
    
    return geometry.group_by(entity_col).agg(aggs)


def aggregate_state_features(state: pl.DataFrame, entity_col: str) -> pl.DataFrame:
    """
    Aggregate state metrics per entity.
    
    State captures temporal dynamics: velocity, acceleration in coupling space.
    """
    exclude_cols = {entity_col, 'timestamp', 'window_end', 'cohort_id', 'signal_id'}
    metrics = [c for c in state.columns if c not in exclude_cols and not c.startswith('_')]
    
    if not metrics:
        print("Warning: No metric columns found in state.parquet")
        return pl.DataFrame({entity_col: []})
    
    aggs = []
    for metric in metrics:
        aggs.extend([
            pl.col(metric).mean().alias(f'state_{metric}_mean'),
            pl.col(metric).last().alias(f'state_{metric}_last'),
        ])
    
    return state.group_by(entity_col).agg(aggs)


def extract_target(
    observations: pl.DataFrame, 
    entity_col: str, 
    target_col: str
) -> pl.DataFrame:
    """
    Extract target variable per entity.
    
    Takes the LAST value of target per entity (e.g., final RUL).
    """
    if target_col not in observations.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {observations.columns}")
    
    return observations.group_by(entity_col).agg(
        pl.col(target_col).last().alias('target')
    )


def detect_entity_column(df: pl.DataFrame) -> str:
    """
    Auto-detect the entity column from dataframe.
    
    Looks for common patterns: entity_id, engine_id, unit_id, bearing_id
    """
    candidates = ['entity_id', 'engine_id', 'unit_id', 'bearing_id', 'unit', 'engine']
    
    for col in candidates:
        if col in df.columns:
            return col
    
    # Fallback: look for columns ending in _id
    id_cols = [c for c in df.columns if c.endswith('_id') and c != 'signal_id']
    if id_cols:
        return id_cols[0]
    
    raise ValueError(f"Could not detect entity column. Available columns: {df.columns}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate ML-ready feature parquet from PRISM outputs'
    )
    parser.add_argument(
        '--target', type=str, default=None,
        help='Target column name for supervised learning (e.g., RUL, fault_type)'
    )
    parser.add_argument(
        '--entity', type=str, default=None,
        help='Entity column for aggregation (auto-detected if not specified)'
    )
    parser.add_argument(
        '--testing', action='store_true',
        help='Enable test mode'
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='[TESTING] Max entities to process'
    )
    args = parser.parse_args()
    
    # -------------------------------------------------------------------------
    # Load PRISM outputs
    # -------------------------------------------------------------------------
    print("Loading PRISM outputs...")
    
    vector_path = get_path(VECTOR)
    geometry_path = get_path(GEOMETRY)
    state_path = get_path(STATE)
    observations_path = get_path(OBSERVATIONS)
    
    if not Path(vector_path).exists():
        raise FileNotFoundError(f"vector.parquet not found. Run signal_vector first.")
    
    vector = pl.read_parquet(vector_path)
    print(f"  vector.parquet: {len(vector):,} rows, {len(vector.columns)} columns")
    
    # Geometry and state are optional (might not exist yet)
    geometry = None
    state = None
    
    if Path(geometry_path).exists():
        geometry = pl.read_parquet(geometry_path)
        print(f"  geometry.parquet: {len(geometry):,} rows, {len(geometry.columns)} columns")
    else:
        print("  geometry.parquet: not found (skipping)")
    
    if Path(state_path).exists():
        state = pl.read_parquet(state_path)
        print(f"  state.parquet: {len(state):,} rows, {len(state.columns)} columns")
    else:
        print("  state.parquet: not found (skipping)")
    
    # -------------------------------------------------------------------------
    # Detect entity column
    # -------------------------------------------------------------------------
    entity_col = args.entity or detect_entity_column(vector)
    print(f"\nEntity column: {entity_col}")
    
    n_entities = vector[entity_col].n_unique()
    print(f"Unique entities: {n_entities:,}")
    
    # -------------------------------------------------------------------------
    # Aggregate features
    # -------------------------------------------------------------------------
    print("\nAggregating features...")
    
    print("  Vector features...")
    ml_features = aggregate_vector_features(vector, entity_col)
    print(f"    {len(ml_features.columns) - 1} features")
    
    if geometry is not None:
        print("  Geometry features...")
        geometry_features = aggregate_geometry_features(geometry, entity_col)
        if len(geometry_features.columns) > 1:
            ml_features = ml_features.join(geometry_features, on=entity_col, how='left')
            print(f"    {len(geometry_features.columns) - 1} features")
    
    if state is not None:
        print("  State features...")
        state_features = aggregate_state_features(state, entity_col)
        if len(state_features.columns) > 1:
            ml_features = ml_features.join(state_features, on=entity_col, how='left')
            print(f"    {len(state_features.columns) - 1} features")
    
    # -------------------------------------------------------------------------
    # Add target variable
    # -------------------------------------------------------------------------
    if args.target:
        print(f"\nExtracting target: {args.target}")
        observations = pl.read_parquet(observations_path)
        target_df = extract_target(observations, entity_col, args.target)
        ml_features = ml_features.join(target_df, on=entity_col, how='left')
        
        # Report target stats
        target_values = ml_features['target'].drop_nulls()
        print(f"  Target range: {target_values.min():.2f} - {target_values.max():.2f}")
        print(f"  Target mean: {target_values.mean():.2f}")
    
    # -------------------------------------------------------------------------
    # Testing limit
    # -------------------------------------------------------------------------
    if args.testing and args.limit:
        ml_features = ml_features.head(args.limit)
        print(f"\n[TESTING] Limited to {args.limit} entities")
    
    # -------------------------------------------------------------------------
    # Handle nulls/infinities
    # -------------------------------------------------------------------------
    print("\nCleaning features...")
    
    # Replace infinities with nulls
    for col in ml_features.columns:
        if ml_features[col].dtype in [pl.Float32, pl.Float64]:
            ml_features = ml_features.with_columns(
                pl.when(pl.col(col).is_infinite())
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )
    
    # Count nulls per column
    null_counts = {col: ml_features[col].null_count() for col in ml_features.columns}
    high_null_cols = [col for col, count in null_counts.items() if count > len(ml_features) * 0.5]
    
    if high_null_cols:
        print(f"  Warning: {len(high_null_cols)} columns have >50% nulls")
    
    # Fill remaining nulls with 0 (safe default for tree models)
    ml_features = ml_features.fill_null(0)
    
    # -------------------------------------------------------------------------
    # Write output
    # -------------------------------------------------------------------------
    output_path = get_path(ML_FEATURES)
    ml_features.write_parquet(output_path)
    
    n_entities = len(ml_features)
    n_features = len(ml_features.columns) - 1  # exclude entity_id
    if args.target:
        n_features -= 1  # exclude target
    
    print(f"\n" + "="*50)
    print(f"ML FEATURES GENERATED")
    print(f"="*50)
    print(f"Entities:  {n_entities:,}")
    print(f"Features:  {n_features:,}")
    if args.target:
        print(f"Target:    {args.target}")
    print(f"Output:    {output_path}")
    print(f"="*50)


if __name__ == "__main__":
    main()
