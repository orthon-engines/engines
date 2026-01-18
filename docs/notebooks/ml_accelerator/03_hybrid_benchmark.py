#!/usr/bin/env python3
"""
PRISM ML Accelerator - Hybrid Benchmark
=========================================

Uses PRISM features where available, raw features as fallback.
This allows benchmarking on all 100 test units.
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

RUL_CAP = 125


def load_data():
    """Load all data sources."""
    train_df = pl.read_parquet(CMAPSS_DIR / 'train_FD001.parquet')
    test_df = pl.read_parquet(CMAPSS_DIR / 'test_FD001.parquet')
    rul_df = pl.read_parquet(CMAPSS_DIR / 'RUL_FD001.parquet')

    vec = pl.read_parquet(PRISM_DIR / 'vector' / 'indicator.parquet')
    field = pl.read_parquet(PRISM_DIR / 'vector' / 'indicator_field.parquet')

    return train_df, test_df, rul_df, vec, field


def compute_raw_features(df: pl.DataFrame, final_only: bool = True) -> pl.DataFrame:
    """
    Compute raw sensor features.

    If final_only, returns just the final cycle per unit.
    Otherwise, returns features for each cycle.
    """
    sensor_cols = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
                   's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

    if final_only:
        # Get final cycle per unit
        final = df.group_by('unit').agg([
            pl.col(col).last().alias(f'raw_{col}') for col in sensor_cols
        ] + [
            pl.col('cycle').max().alias('max_cycle')
        ])
        return final

    # Return all cycles with features
    return df.select(['unit', 'cycle'] + sensor_cols)


def extract_prism_features_per_window(
    vec: pl.DataFrame,
    field: pl.DataFrame,
    split: str  # 'train' or 'test'
) -> pl.DataFrame:
    """Extract PRISM features aggregated per window."""

    # Extract unit from indicator_id
    vec = vec.with_columns(
        pl.col('indicator_id').str.extract(r'FD001_(\d+)_', 1).cast(pl.Int64).alias('raw_unit'),
    ).filter(pl.col('raw_unit').is_not_null())

    # Determine split
    if split == 'train':
        vec = vec.filter(pl.col('raw_unit') <= 100)
        vec = vec.with_columns(pl.col('raw_unit').alias('unit'))
    else:  # test
        vec = vec.filter(pl.col('raw_unit') > 1000)
        vec = vec.with_columns((pl.col('raw_unit') - 1000).alias('unit'))

    if len(vec) == 0:
        return pl.DataFrame()

    # Aggregate across sensors per (unit, date, engine, metric)
    agg = vec.group_by(['unit', 'obs_date', 'engine', 'metric_name']).agg([
        pl.col('metric_value').mean().alias('mean'),
    ])

    # Create feature name
    agg = agg.with_columns(
        (pl.col('engine') + '_' + pl.col('metric_name')).alias('feature_name')
    )

    # Pivot to wide format
    wide = agg.pivot(
        on='feature_name',
        index=['unit', 'obs_date'],
        values='mean'
    )

    # Similarly for Laplace field
    field = field.with_columns(
        pl.col('indicator_id').str.extract(r'FD001_(\d+)_', 1).cast(pl.Int64).alias('raw_unit'),
    ).filter(pl.col('raw_unit').is_not_null())

    if split == 'train':
        field = field.filter(pl.col('raw_unit') <= 100)
        field = field.with_columns(pl.col('raw_unit').alias('unit'))
    else:
        field = field.filter(pl.col('raw_unit') > 1000)
        field = field.with_columns((pl.col('raw_unit') - 1000).alias('unit'))

    if len(field) > 0:
        lap_agg = field.group_by(['unit', 'window_end']).agg([
            pl.col('divergence').mean().alias('lap_div_mean'),
            pl.col('divergence').std().alias('lap_div_std'),
            pl.col('gradient').mean().alias('lap_grad_mean'),
            pl.col('gradient_magnitude').mean().alias('lap_gradmag_mean'),
        ]).rename({'window_end': 'obs_date'})

        # Join Laplace to vector
        wide = wide.join(lap_agg, on=['unit', 'obs_date'], how='left')

    return wide


def prepare_training_data(
    train_df: pl.DataFrame,
    prism_features: pl.DataFrame
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare training data with RUL labels."""

    # Get max cycles per unit
    max_cycles = train_df.group_by('unit').agg(
        pl.col('cycle').max().alias('max_cycle')
    )

    # Join with PRISM features
    combined = prism_features.join(max_cycles, on='unit')

    # Compute window positions and RUL
    # Get max date per unit (final window)
    max_dates = combined.group_by('unit').agg(
        pl.col('obs_date').max().alias('max_date')
    )
    combined = combined.join(max_dates, on='unit')

    # Rank windows
    combined = combined.with_columns(
        pl.col('obs_date').rank().over('unit').alias('window_rank')
    )

    n_windows_per_unit = combined.group_by('unit').agg(
        pl.col('obs_date').n_unique().alias('n_windows')
    )
    combined = combined.join(n_windows_per_unit, on='unit')

    # Compute approximate RUL
    combined = combined.with_columns(
        ((pl.col('n_windows') - pl.col('window_rank')) *
         (pl.col('max_cycle') / pl.col('n_windows'))).round().cast(pl.Int64).alias('RUL')
    )

    # Cap at 125
    combined = combined.with_columns(
        pl.when(pl.col('RUL') > RUL_CAP)
        .then(RUL_CAP)
        .otherwise(pl.col('RUL'))
        .alias('RUL')
    )

    # Get feature columns
    meta_cols = ['unit', 'obs_date', 'max_cycle', 'max_date', 'window_rank', 'n_windows', 'RUL']
    feature_cols = [c for c in combined.columns if c not in meta_cols]

    X = combined.select(feature_cols).to_numpy()
    y = combined['RUL'].to_numpy()

    # Handle NaN
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    return X, y, feature_cols


def prepare_test_data(
    test_df: pl.DataFrame,
    rul_df: pl.DataFrame,
    prism_features: pl.DataFrame,
    feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare test data with hybrid features."""

    all_test_units = sorted(test_df['unit'].unique().to_list())

    # Get units that have PRISM features
    prism_units = sorted(prism_features['unit'].unique().to_list())

    # Get final window PRISM features per unit
    final_windows = prism_features.group_by('unit').agg(
        pl.col('obs_date').max().alias('final_date')
    )
    prism_final = prism_features.join(
        final_windows,
        left_on=['unit', 'obs_date'],
        right_on=['unit', 'final_date'],
        how='inner'
    )

    # Get raw features for all units (as fallback)
    raw_features = compute_raw_features(test_df, final_only=True)

    # Build test feature matrix
    rows = []
    for unit in all_test_units:
        if unit in prism_units:
            # Use PRISM features
            unit_row = prism_final.filter(pl.col('unit') == unit)
            if len(unit_row) > 0:
                # Ensure all feature columns exist
                row_data = {'unit': unit}
                for col in feature_cols:
                    if col in unit_row.columns:
                        row_data[col] = unit_row[col][0]
                    else:
                        row_data[col] = 0.0
                rows.append(row_data)
            else:
                # Fallback to raw
                unit_raw = raw_features.filter(pl.col('unit') == unit)
                row_data = {'unit': unit}
                for col in feature_cols:
                    if col in unit_raw.columns:
                        row_data[col] = unit_raw[col][0]
                    else:
                        row_data[col] = 0.0
                rows.append(row_data)
        else:
            # Use raw features
            unit_raw = raw_features.filter(pl.col('unit') == unit)
            row_data = {'unit': unit}
            for col in feature_cols:
                if col.startswith('raw_') and col in unit_raw.columns:
                    row_data[col] = unit_raw[col][0]
                else:
                    row_data[col] = 0.0
            rows.append(row_data)

    test_features = pl.DataFrame(rows, infer_schema_length=None)
    test_features = test_features.sort('unit')

    # Get RUL
    rul_sorted = rul_df.sort('unit')

    X_test = test_features.select(feature_cols).to_numpy()
    y_test = rul_sorted['RUL'].to_numpy()
    test_units = test_features['unit'].to_numpy()

    # Handle NaN
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    return X_test, y_test, test_units


def run_hybrid_benchmark():
    """Run hybrid benchmark."""
    print("=" * 70)
    print("PRISM ML ACCELERATOR - HYBRID BENCHMARK")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Load data
    print("Loading data...")
    train_df, test_df, rul_df, vec, field = load_data()

    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")
    print(f"RUL: {rul_df.shape}")

    # Extract PRISM features
    print("\nExtracting PRISM features...")
    train_prism = extract_prism_features_per_window(vec, field, 'train')
    test_prism = extract_prism_features_per_window(vec, field, 'test')

    train_prism_units = train_prism['unit'].n_unique()
    test_prism_units = test_prism['unit'].n_unique()
    print(f"Train PRISM features: {train_prism.shape}, {train_prism_units} units")
    print(f"Test PRISM features: {test_prism.shape}, {test_prism_units} units")

    # Prepare training data
    print("\nPreparing training data...")
    X_train, y_train, feature_cols = prepare_training_data(train_df, train_prism)
    print(f"Training: {X_train.shape}, RUL range: {y_train.min()}-{y_train.max()}")

    # Prepare test data (hybrid)
    print("\nPreparing test data (hybrid PRISM + raw fallback)...")
    X_test, y_test, test_units = prepare_test_data(test_df, rul_df, test_prism, feature_cols)
    print(f"Test: {X_test.shape}")
    print(f"Test units with PRISM: {test_prism_units}/100")
    print(f"Test units with raw fallback: {100 - test_prism_units}/100")

    # Train XGBoost
    print("\nTraining XGBoost...")
    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)

    # Evaluate
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    # Evaluate separately for PRISM vs raw units
    prism_mask = np.isin(test_units, test_prism['unit'].unique().to_list())
    if prism_mask.sum() > 0:
        rmse_prism = np.sqrt(np.mean((y_test[prism_mask] - y_pred[prism_mask]) ** 2))
    else:
        rmse_prism = np.nan

    raw_mask = ~prism_mask
    if raw_mask.sum() > 0:
        rmse_raw = np.sqrt(np.mean((y_test[raw_mask] - y_pred[raw_mask]) ** 2))
    else:
        rmse_raw = np.nan

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Training samples: {X_train.shape[0]:,}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Test units: {len(test_units)}/100")
    print()
    print(f">>> OVERALL RMSE: {rmse:.2f}")
    print(f"    PRISM units ({prism_mask.sum()}) RMSE: {rmse_prism:.2f}")
    print(f"    Raw units ({raw_mask.sum()}) RMSE: {rmse_raw:.2f}")
    print()
    print("-" * 50)
    print("Published Benchmarks:")
    print("  LSTM: RMSE 13-16")
    print("  CNN:  RMSE 12-14")
    print("=" * 70)

    # Show some predictions
    print("\nSample predictions:")
    print(f"{'Unit':>5} {'True':>8} {'Pred':>8} {'Error':>8} {'Source':>8}")
    print("-" * 40)
    for i in range(min(10, len(test_units))):
        source = "PRISM" if prism_mask[i] else "RAW"
        err = abs(y_test[i] - y_pred[i])
        print(f"{test_units[i]:5d} {y_test[i]:8.0f} {y_pred[i]:8.1f} {err:8.1f} {source:>8}")

    return {
        'rmse': rmse,
        'rmse_prism': rmse_prism,
        'rmse_raw': rmse_raw,
        'n_train': X_train.shape[0],
        'n_features': X_train.shape[1],
        'n_prism_units': prism_mask.sum(),
        'n_raw_units': raw_mask.sum(),
    }


if __name__ == '__main__':
    results = run_hybrid_benchmark()
