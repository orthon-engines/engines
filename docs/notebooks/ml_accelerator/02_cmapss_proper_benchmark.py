#!/usr/bin/env python3
"""
PRISM ML Accelerator - C-MAPSS FD001 Proper Benchmark
======================================================

Proper approach using features at ALL time points during training,
not just the final cycle.

Training: For each cycle t, RUL = min(max_cycle - t, 125)  # capped at 125
Testing: Predict RUL at the cutoff point

This is the standard approach used in published benchmarks.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path('/Users/jasonrudder/prism-mac/data')
CMAPSS_DIR = DATA_DIR / 'CMAPSSData'
PRISM_DIR = DATA_DIR / 'cmapss_fd001'

RUL_CAP = 125  # Standard RUL capping


def load_official_data():
    """Load official C-MAPSS data."""
    train_df = pl.read_parquet(CMAPSS_DIR / 'train_FD001.parquet')
    test_df = pl.read_parquet(CMAPSS_DIR / 'test_FD001.parquet')
    rul_df = pl.read_parquet(CMAPSS_DIR / 'RUL_FD001.parquet')

    print("=== Official C-MAPSS FD001 Data ===")
    print(f"Train: {train_df.shape} ({train_df['unit'].n_unique()} engines)")
    print(f"Test: {test_df.shape} ({test_df['unit'].n_unique()} engines)")
    print(f"RUL: {rul_df.shape}, range {rul_df['RUL'].min()}-{rul_df['RUL'].max()}")

    return train_df, test_df, rul_df


def extract_unit_from_indicator(df: pl.DataFrame) -> pl.DataFrame:
    """Extract unit number from indicator_id."""
    return df.with_columns(
        pl.when(pl.col('indicator_id').str.contains(r'FD001_(\d+)_'))
        .then(pl.col('indicator_id').str.extract(r'FD001_(\d+)_', 1).cast(pl.Int64))
        .otherwise(None)
        .alias('unit')
    )


def compute_train_rul(train_df: pl.DataFrame) -> pl.DataFrame:
    """Compute RUL for training data (capped at 125)."""
    # Get max cycle per unit (failure point)
    max_cycles = train_df.group_by('unit').agg(pl.col('cycle').max().alias('max_cycle'))

    # Join and compute RUL
    train_with_rul = train_df.join(max_cycles, on='unit').with_columns(
        pl.when(pl.col('max_cycle') - pl.col('cycle') > RUL_CAP)
        .then(RUL_CAP)
        .otherwise(pl.col('max_cycle') - pl.col('cycle'))
        .alias('RUL')
    )

    return train_with_rul


def load_prism_vector_features():
    """Load PRISM vector features with window dates."""
    vec_path = PRISM_DIR / 'vector' / 'indicator.parquet'
    if not vec_path.exists():
        return pl.DataFrame()

    vec = pl.read_parquet(vec_path)
    vec = extract_unit_from_indicator(vec)

    # Map test units (1001-1100 -> 1-100) and add split column
    vec = vec.with_columns(
        pl.when(pl.col('unit') > 1000)
        .then(pl.col('unit') - 1000)
        .otherwise(pl.col('unit'))
        .alias('actual_unit'),

        pl.when(pl.col('unit') > 1000)
        .then(pl.lit('test'))
        .otherwise(pl.lit('train'))
        .alias('split')
    )

    print(f"\n=== PRISM Vector Features ===")
    print(f"Shape: {vec.shape}")
    train_units = vec.filter(pl.col('split') == 'train')['actual_unit'].n_unique()
    test_units = vec.filter(pl.col('split') == 'test')['actual_unit'].n_unique()
    print(f"Train units: {train_units}, Test units: {test_units}")

    return vec


def load_prism_laplace_features():
    """Load PRISM Laplace field features."""
    field_path = PRISM_DIR / 'vector' / 'indicator_field.parquet'
    if not field_path.exists():
        return pl.DataFrame()

    field = pl.read_parquet(field_path)
    field = extract_unit_from_indicator(field)

    # Map test units
    field = field.with_columns(
        pl.when(pl.col('unit') > 1000)
        .then(pl.col('unit') - 1000)
        .otherwise(pl.col('unit'))
        .alias('actual_unit'),

        pl.when(pl.col('unit') > 1000)
        .then(pl.lit('test'))
        .otherwise(pl.lit('train'))
        .alias('split')
    )

    print(f"\n=== PRISM Laplace Features ===")
    print(f"Shape: {field.shape}")

    return field


def map_dates_to_cycles(train_df: pl.DataFrame, prism_df: pl.DataFrame) -> pl.DataFrame:
    """
    Map PRISM window end dates to actual cycles.

    PRISM uses synthetic dates (base + unit_offset + cycle).
    We need to reverse-map these to actual cycles.
    """
    # Get date range per unit from training data
    # PRISM uses: obs_date = base_date + (unit - 1) * 500 + cycle
    # base_date = 2000-01-01

    # For each unit, compute the mapping
    # Actually, let's just use the window index as a proxy for "time"
    # and map to cycles based on the relative position

    # Group by unit and get unique dates
    dates_per_unit = prism_df.group_by(['actual_unit', 'split']).agg(
        pl.col('obs_date').unique().sort().alias('dates')
    )

    # For now, map the date index to a cycle fraction
    # This is approximate but works for demonstration
    return prism_df


def aggregate_vector_features_per_window(vec: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate vector features per unit per window.

    Creates one feature vector per (unit, window_date).
    """
    if len(vec) == 0:
        return pl.DataFrame()

    # Aggregate across sensors for each (unit, date, engine, metric)
    agg = vec.group_by(['actual_unit', 'split', 'obs_date', 'engine', 'metric_name']).agg([
        pl.col('metric_value').mean().alias('mean'),
    ])

    # Create feature name
    agg = agg.with_columns(
        (pl.col('engine') + '_' + pl.col('metric_name')).alias('feature_name')
    )

    # Pivot to wide format
    wide = agg.pivot(
        on='feature_name',
        index=['actual_unit', 'split', 'obs_date'],
        values='mean'
    )

    return wide


def aggregate_laplace_features_per_window(field: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate Laplace features per unit per window.
    """
    if len(field) == 0:
        return pl.DataFrame()

    # Aggregate Laplace metrics per unit per window
    agg = field.group_by(['actual_unit', 'split', 'window_end']).agg([
        pl.col('divergence').mean().alias('lap_div_mean'),
        pl.col('divergence').std().alias('lap_div_std'),
        pl.col('gradient').mean().alias('lap_grad_mean'),
        pl.col('gradient_magnitude').mean().alias('lap_gradmag_mean'),
        pl.col('laplacian').mean().alias('lap_laplacian_mean'),
        pl.col('is_source').sum().alias('lap_n_sources'),
        pl.col('is_sink').sum().alias('lap_n_sinks'),
    ])

    return agg.rename({'window_end': 'obs_date'})


def prepare_training_data(
    vec_wide: pl.DataFrame,
    laplace_wide: pl.DataFrame,
    train_df: pl.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data with RUL labels.

    For each training window, we assign RUL based on relative position.
    """
    # Get train vector data
    train_vec = vec_wide.filter(pl.col('split') == 'train').drop('split')

    # Get train laplace data
    train_lap = laplace_wide.filter(pl.col('split') == 'train').drop('split')

    # Join vector and laplace on (unit, date)
    combined = train_vec.join(
        train_lap,
        on=['actual_unit', 'obs_date'],
        how='inner'
    )

    # For each unit, compute RUL based on relative window position
    # Window at end = RUL 0, earlier windows = higher RUL

    # Get max date per unit (final window = failure)
    max_dates = combined.group_by('actual_unit').agg(
        pl.col('obs_date').max().alias('max_date')
    )

    combined = combined.join(max_dates, on='actual_unit')

    # Get max cycle per unit from train_df
    max_cycles = train_df.group_by('unit').agg(
        pl.col('cycle').max().alias('max_cycle')
    )

    combined = combined.join(
        max_cycles.rename({'unit': 'actual_unit'}),
        on='actual_unit'
    )

    # Compute approximate cycle from date
    # PRISM stride is ~7 days typically
    # Count windows from end and multiply by stride
    window_counts = combined.group_by('actual_unit').agg(
        pl.col('obs_date').n_unique().alias('n_windows')
    )
    combined = combined.join(window_counts, on='actual_unit')

    # Rank windows within each unit (1 = earliest, n = latest/failure)
    combined = combined.with_columns(
        pl.col('obs_date').rank().over('actual_unit').alias('window_rank')
    )

    # Compute RUL: at final window, RUL=0
    # Approximate: each window step = max_cycle / n_windows
    combined = combined.with_columns(
        ((pl.col('n_windows') - pl.col('window_rank')) *
         (pl.col('max_cycle') / pl.col('n_windows'))).round().cast(pl.Int64).alias('approx_RUL')
    )

    # Cap at 125
    combined = combined.with_columns(
        pl.when(pl.col('approx_RUL') > RUL_CAP)
        .then(RUL_CAP)
        .otherwise(pl.col('approx_RUL'))
        .alias('RUL')
    )

    # Get feature columns
    meta_cols = ['actual_unit', 'obs_date', 'split', 'max_date', 'max_cycle',
                 'n_windows', 'window_rank', 'approx_RUL', 'RUL']
    feature_cols = [c for c in combined.columns if c not in meta_cols]

    print(f"\nTraining data: {len(combined)} samples, {len(feature_cols)} features")
    print(f"RUL range: {combined['RUL'].min()} - {combined['RUL'].max()}")

    # Extract features and targets
    X_train = combined.select(feature_cols).to_numpy()
    y_train = combined['RUL'].to_numpy()

    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)

    return X_train, y_train, feature_cols


def prepare_test_data(
    vec_wide: pl.DataFrame,
    laplace_wide: pl.DataFrame,
    rul_df: pl.DataFrame,
    feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare test data - use FINAL window per unit.
    """
    # Get test vector data
    test_vec = vec_wide.filter(pl.col('split') == 'test').drop('split')

    # Get test laplace data
    test_lap = laplace_wide.filter(pl.col('split') == 'test').drop('split')

    # Join
    combined = test_vec.join(
        test_lap,
        on=['actual_unit', 'obs_date'],
        how='inner'
    )

    # Get final window per unit
    final_windows = combined.group_by('actual_unit').agg(
        pl.col('obs_date').max().alias('final_date')
    )

    combined = combined.join(
        final_windows,
        left_on=['actual_unit', 'obs_date'],
        right_on=['actual_unit', 'final_date'],
        how='inner'
    )

    # Sort by unit
    combined = combined.sort('actual_unit')

    # Get units that have features
    test_units = combined['actual_unit'].unique().sort().to_list()

    # Get RUL for these units
    test_rul = rul_df.filter(pl.col('unit').is_in(test_units)).sort('unit')

    # Ensure feature columns exist (pad missing with 0)
    for col in feature_cols:
        if col not in combined.columns:
            combined = combined.with_columns(pl.lit(0.0).alias(col))

    X_test = combined.select(feature_cols).to_numpy()
    y_test = test_rul['RUL'].to_numpy()

    # Handle NaN
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    print(f"\nTest data: {len(combined)} samples, {len(test_units)} units")

    return X_test, y_test, np.array(test_units)


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray):
    """Train XGBoost regressor."""
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def run_benchmark():
    """Run the proper benchmark."""
    print("=" * 70)
    print("PRISM ML ACCELERATOR - C-MAPSS FD001 PROPER BENCHMARK")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"RUL Cap: {RUL_CAP}")

    # Load data
    train_df, test_df, rul_df = load_official_data()

    # Load PRISM features
    vec = load_prism_vector_features()
    field = load_prism_laplace_features()

    # Aggregate per window
    print("\n=== Aggregating Features Per Window ===")
    vec_wide = aggregate_vector_features_per_window(vec)
    laplace_wide = aggregate_laplace_features_per_window(field)

    print(f"Vector windows: {vec_wide.shape}")
    print(f"Laplace windows: {laplace_wide.shape}")

    # Prepare training data
    print("\n=== Preparing Training Data ===")
    X_train, y_train, feature_cols = prepare_training_data(vec_wide, laplace_wide, train_df)

    # Prepare test data
    print("\n=== Preparing Test Data ===")
    X_test, y_test, test_units = prepare_test_data(vec_wide, laplace_wide, rul_df, feature_cols)

    # Train and evaluate
    print("\n=== Training XGBoost ===")
    model = train_xgboost(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)  # Ensure non-negative

    rmse = evaluate(y_test, y_pred)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Test units: {len(test_units)}/100")
    print(f"\n>>> PRISM + XGBoost RMSE: {rmse:.2f}")
    print("\n" + "-" * 50)
    print("Published Benchmarks (100 test units):")
    print("  LSTM: RMSE 13-16 (GPU)")
    print("  CNN:  RMSE 12-14 (GPU)")
    print("=" * 70)

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[-10:][::-1]
        print("\nTop 10 Features:")
        for i in top_idx:
            print(f"  {feature_cols[i]}: {importances[i]:.4f}")

    return {
        'rmse': rmse,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': len(feature_cols),
        'test_units': len(test_units),
    }


if __name__ == '__main__':
    results = run_benchmark()
