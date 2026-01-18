#!/usr/bin/env python3
"""
PRISM ML Accelerator - Clean Benchmark
=======================================

Clean benchmark on units with complete PRISM features.
Reports results for 58/100 test units that have PRISM coverage.

This isolates the question: "Do PRISM features improve RUL prediction?"
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
    """Load all data."""
    train_df = pl.read_parquet(CMAPSS_DIR / 'train_FD001.parquet')
    test_df = pl.read_parquet(CMAPSS_DIR / 'test_FD001.parquet')
    rul_df = pl.read_parquet(CMAPSS_DIR / 'RUL_FD001.parquet')

    vec = pl.read_parquet(PRISM_DIR / 'vector' / 'indicator.parquet')
    field = pl.read_parquet(PRISM_DIR / 'vector' / 'indicator_field.parquet')

    return train_df, test_df, rul_df, vec, field


def extract_unit(df: pl.DataFrame) -> pl.DataFrame:
    """Extract unit from indicator_id."""
    return df.with_columns(
        pl.col('indicator_id').str.extract(r'FD001_(\d+)_', 1).cast(pl.Int64).alias('raw_unit')
    ).filter(pl.col('raw_unit').is_not_null())


def get_prism_units(vec: pl.DataFrame) -> Tuple[List[int], List[int]]:
    """Get train and test units with PRISM features."""
    vec = extract_unit(vec)
    train_units = sorted([u for u in vec['raw_unit'].unique().to_list() if u <= 100])
    test_units = sorted([u - 1000 for u in vec['raw_unit'].unique().to_list() if u > 1000])
    return train_units, test_units


def aggregate_final_window_features(
    vec: pl.DataFrame,
    field: pl.DataFrame,
    units: List[int],
    is_test: bool = False
) -> pl.DataFrame:
    """
    Aggregate PRISM features at the FINAL window for each unit.

    For test: We need final window features (cutoff point)
    For train: We use final window features (failure point, RUL=0)
    """
    vec = extract_unit(vec)
    field = extract_unit(field)

    # Map units
    if is_test:
        vec = vec.filter(pl.col('raw_unit') > 1000).with_columns(
            (pl.col('raw_unit') - 1000).alias('unit')
        )
        field = field.filter(pl.col('raw_unit') > 1000).with_columns(
            (pl.col('raw_unit') - 1000).alias('unit')
        )
    else:
        vec = vec.filter(pl.col('raw_unit') <= 100).with_columns(
            pl.col('raw_unit').alias('unit')
        )
        field = field.filter(pl.col('raw_unit') <= 100).with_columns(
            pl.col('raw_unit').alias('unit')
        )

    # Filter to specific units
    vec = vec.filter(pl.col('unit').is_in(units))
    field = field.filter(pl.col('unit').is_in(units))

    # Get final window per unit
    vec_final = vec.group_by('unit').agg(pl.col('obs_date').max().alias('final_date'))
    vec = vec.join(vec_final, left_on=['unit', 'obs_date'], right_on=['unit', 'final_date'], how='inner')

    # Aggregate across sensors: (unit, engine, metric) -> mean
    vec_agg = vec.group_by(['unit', 'engine', 'metric_name']).agg([
        pl.col('metric_value').mean().alias('value')
    ])

    # Pivot to wide
    vec_agg = vec_agg.with_columns(
        (pl.col('engine') + '_' + pl.col('metric_name')).alias('feature_name')
    )
    vec_wide = vec_agg.pivot(on='feature_name', index='unit', values='value')

    # Laplace features
    if len(field) > 0:
        field_final = field.group_by('unit').agg(pl.col('window_end').max().alias('final_date'))
        field = field.join(field_final, left_on=['unit', 'window_end'], right_on=['unit', 'final_date'], how='inner')

        lap_agg = field.group_by('unit').agg([
            pl.col('divergence').mean().alias('lap_div_mean'),
            pl.col('divergence').std().alias('lap_div_std'),
            pl.col('gradient').mean().alias('lap_grad_mean'),
            pl.col('gradient_magnitude').mean().alias('lap_gradmag_mean'),
            pl.col('laplacian').mean().alias('lap_laplacian_mean'),
        ])

        vec_wide = vec_wide.join(lap_agg, on='unit', how='left')

    return vec_wide.sort('unit')


def compute_raw_sensor_features(
    df: pl.DataFrame,
    units: List[int]
) -> pl.DataFrame:
    """
    Compute raw sensor statistics from all cycles (not just final).

    For each unit, compute statistics over the full trajectory.
    """
    sensor_cols = [f's{i}' for i in range(1, 22)]

    # Filter to units
    df = df.filter(pl.col('unit').is_in(units))

    # Compute per-unit statistics
    agg_exprs = []
    for col in sensor_cols:
        agg_exprs.extend([
            pl.col(col).mean().alias(f'raw_{col}_mean'),
            pl.col(col).std().alias(f'raw_{col}_std'),
            pl.col(col).last().alias(f'raw_{col}_final'),
            (pl.col(col).last() - pl.col(col).first()).alias(f'raw_{col}_delta'),
        ])

    # Add cycle count as feature
    agg_exprs.append(pl.col('cycle').max().alias('raw_max_cycle'))

    raw_features = df.group_by('unit').agg(agg_exprs)

    return raw_features.sort('unit')


def run_clean_benchmark():
    """Run clean benchmark on PRISM-covered units."""
    print("=" * 70)
    print("PRISM ML ACCELERATOR - CLEAN BENCHMARK")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Load data
    train_df, test_df, rul_df, vec, field = load_data()
    train_prism_units, test_prism_units = get_prism_units(vec)

    print(f"Train units with PRISM: {len(train_prism_units)}/100")
    print(f"Test units with PRISM: {len(test_prism_units)}/100")
    print(f"Test units: {test_prism_units[:10]}...")

    # Get RUL for test units
    test_rul = rul_df.filter(pl.col('unit').is_in(test_prism_units)).sort('unit')
    y_test = test_rul['RUL'].to_numpy()

    # Strategy 1: PRISM features only (final window)
    print("\n--- STRATEGY 1: PRISM Features (final window) ---")
    train_prism = aggregate_final_window_features(vec, field, train_prism_units, is_test=False)
    test_prism = aggregate_final_window_features(vec, field, test_prism_units, is_test=True)

    # Get common features
    common_cols = list(set(train_prism.columns) & set(test_prism.columns) - {'unit'})
    print(f"PRISM features: {len(common_cols)}")

    X_train_prism = train_prism.select(common_cols).to_numpy()
    X_test_prism = test_prism.select(common_cols).to_numpy()

    # Train RUL = 0 (at failure point)
    y_train = np.zeros(len(X_train_prism))

    # Handle NaN
    X_train_prism = np.nan_to_num(X_train_prism, nan=0)
    X_test_prism = np.nan_to_num(X_test_prism, nan=0)

    import xgboost as xgb
    model_prism = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    model_prism.fit(X_train_prism, y_train)
    pred_prism = np.maximum(model_prism.predict(X_test_prism), 0)
    rmse_prism_only = np.sqrt(np.mean((y_test - pred_prism) ** 2))
    print(f"RMSE (PRISM only, RUL=0 training): {rmse_prism_only:.2f}")

    # Strategy 2: Raw sensor features (full trajectory stats)
    print("\n--- STRATEGY 2: Raw Features (trajectory stats) ---")
    train_raw = compute_raw_sensor_features(train_df, train_prism_units)
    test_raw = compute_raw_sensor_features(test_df, test_prism_units)

    raw_cols = list(set(train_raw.columns) & set(test_raw.columns) - {'unit'})
    print(f"Raw features: {len(raw_cols)}")

    X_train_raw = train_raw.select(raw_cols).to_numpy()
    X_test_raw = test_raw.select(raw_cols).to_numpy()
    X_train_raw = np.nan_to_num(X_train_raw, nan=0)
    X_test_raw = np.nan_to_num(X_test_raw, nan=0)

    model_raw = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    model_raw.fit(X_train_raw, y_train)
    pred_raw = np.maximum(model_raw.predict(X_test_raw), 0)
    rmse_raw = np.sqrt(np.mean((y_test - pred_raw) ** 2))
    print(f"RMSE (Raw only): {rmse_raw:.2f}")

    # Strategy 3: PRISM + Raw combined
    print("\n--- STRATEGY 3: PRISM + Raw Combined ---")
    X_train_comb = np.hstack([X_train_prism, X_train_raw])
    X_test_comb = np.hstack([X_test_prism, X_test_raw])
    print(f"Combined features: {X_train_comb.shape[1]}")

    model_comb = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    model_comb.fit(X_train_comb, y_train)
    pred_comb = np.maximum(model_comb.predict(X_test_comb), 0)
    rmse_comb = np.sqrt(np.mean((y_test - pred_comb) ** 2))
    print(f"RMSE (PRISM + Raw): {rmse_comb:.2f}")

    # Strategy 4: Use trajectory for training (better y_train)
    print("\n--- STRATEGY 4: Multi-window training (proper RUL) ---")

    # Build training set with multiple windows per engine
    vec_tr = extract_unit(vec).filter(pl.col('raw_unit') <= 100).with_columns(
        pl.col('raw_unit').alias('unit')
    )

    # Get unique dates per unit
    windows = vec_tr.group_by('unit').agg(
        pl.col('obs_date').unique().sort().alias('dates')
    )

    # Get max cycles
    max_cycles = train_df.group_by('unit').agg(pl.col('cycle').max().alias('max_cycle'))

    # Build training examples for each window
    train_rows = []
    for row in windows.iter_rows(named=True):
        unit = row['unit']
        dates = row['dates']
        max_cycle = max_cycles.filter(pl.col('unit') == unit)['max_cycle'][0]

        n_dates = len(dates)
        for i, date in enumerate(dates):
            # RUL approximation: last window = 0, earlier = higher
            approx_rul = int((n_dates - 1 - i) * max_cycle / n_dates)
            approx_rul = min(approx_rul, RUL_CAP)

            # Get features for this window
            window_data = vec_tr.filter(
                (pl.col('unit') == unit) & (pl.col('obs_date') == date)
            )

            # Aggregate across sensors
            agg = window_data.group_by(['engine', 'metric_name']).agg(
                pl.col('metric_value').mean().alias('value')
            )

            # Create feature dict
            feature_dict = {'unit': unit, 'RUL': approx_rul}
            for r in agg.iter_rows(named=True):
                feat_name = f"{r['engine']}_{r['metric_name']}"
                feature_dict[feat_name] = r['value']

            train_rows.append(feature_dict)

    train_multi = pl.DataFrame(train_rows, infer_schema_length=None)
    print(f"Multi-window training: {len(train_multi)} samples from {train_multi['unit'].n_unique()} units")
    print(f"RUL range: {train_multi['RUL'].min()}-{train_multi['RUL'].max()}")

    # Get features
    multi_feature_cols = [c for c in train_multi.columns if c not in ['unit', 'RUL']]
    # Only use features that exist in test
    multi_feature_cols = [c for c in multi_feature_cols if c in common_cols]
    print(f"Features: {len(multi_feature_cols)}")

    X_train_multi = train_multi.select(multi_feature_cols).to_numpy()
    y_train_multi = train_multi['RUL'].to_numpy()
    X_test_multi = test_prism.select([c for c in multi_feature_cols if c in test_prism.columns]).to_numpy()

    X_train_multi = np.nan_to_num(X_train_multi, nan=0)
    X_test_multi = np.nan_to_num(X_test_multi, nan=0)

    model_multi = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    model_multi.fit(X_train_multi, y_train_multi)
    pred_multi = np.maximum(model_multi.predict(X_test_multi), 0)
    rmse_multi = np.sqrt(np.mean((y_test - pred_multi) ** 2))
    print(f"RMSE (Multi-window PRISM): {rmse_multi:.2f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - C-MAPSS FD001 (58 test units with PRISM coverage)")
    print("=" * 70)
    print(f"\n{'Method':<35} {'Features':>10} {'RMSE':>10}")
    print("-" * 55)
    print(f"{'1. PRISM final window (y=0)':<35} {len(common_cols):>10} {rmse_prism_only:>10.2f}")
    print(f"{'2. Raw trajectory stats':<35} {len(raw_cols):>10} {rmse_raw:>10.2f}")
    print(f"{'3. PRISM + Raw combined':<35} {X_train_comb.shape[1]:>10} {rmse_comb:>10.2f}")
    print(f"{'4. PRISM multi-window (proper RUL)':<35} {len(multi_feature_cols):>10} {rmse_multi:>10.2f}")
    print("-" * 55)
    print("\nPublished Benchmarks (100 test units):")
    print("  LSTM: RMSE 13-16")
    print("  CNN:  RMSE 12-14")
    print()
    print("Note: Results on 58/100 test units due to incomplete PRISM coverage.")
    print("=" * 70)

    # Show feature importance for best model
    if rmse_multi < min(rmse_prism_only, rmse_raw, rmse_comb):
        best_model = model_multi
        best_features = multi_feature_cols
    else:
        best_model = model_comb
        best_features = list(common_cols) + raw_cols

    print("\nTop 10 Features (best model):")
    importances = best_model.feature_importances_
    top_idx = np.argsort(importances)[-10:][::-1]
    for i in top_idx:
        if i < len(best_features):
            print(f"  {best_features[i]}: {importances[i]:.4f}")

    return {
        'rmse_prism_only': rmse_prism_only,
        'rmse_raw': rmse_raw,
        'rmse_combined': rmse_comb,
        'rmse_multi_window': rmse_multi,
        'test_units': len(test_prism_units),
    }


if __name__ == '__main__':
    results = run_clean_benchmark()
