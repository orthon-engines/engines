"""
PRISM vs Baseline V2 - Full feature engineering.

Combines:
- Raw sensors (24 features)
- Vector-level: hurst, entropy, garch, lyapunov per source signal (pivoted to wide)
- Geometry-level: PCA, clustering, MST, LOF, distance, copula, hull
- State-level: trajectory dynamics, mode changes

Key insight: Pivot vector features from long to wide format.
"""

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


# C-MAPSS column names
COLUMNS = ['unit_id', 'cycle'] + [f'op_{i}' for i in range(1, 4)] + [f's_{i}' for i in range(1, 22)]

DATA_ROOT = Path("/Users/jasonrudder/prism-mac/data")
ML_DIR = DATA_ROOT / "machine_learning"
TRAIN_DIR = DATA_ROOT / "C-MAPPS_TRAIN"
TEST_DIR = DATA_ROOT / "C-MAPPS_TEST"


def load_cmapss(path: str) -> pd.DataFrame:
    """Load C-MAPSS txt file."""
    df = pd.read_csv(path, sep=r'\s+', header=None, names=COLUMNS)
    return df


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Add RUL column: max_cycle - current_cycle per unit."""
    max_cycles = df.groupby('unit_id')['cycle'].max().rename('max_cycle')
    df = df.merge(max_cycles, on='unit_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df = df.drop(columns=['max_cycle'])
    return df


def load_rul_file(path: str) -> np.ndarray:
    """Load ground truth RUL file."""
    with open(path, 'r') as f:
        return np.array([float(line.strip()) for line in f if line.strip()])


def pivot_vector_features(vec: pl.DataFrame, engines: list = None) -> pl.DataFrame:
    """
    Pivot vector from long to wide format.

    Input: signal_id, entity_id, timestamp, value, engine
    Output: entity_id, timestamp, hurst_signal1, hurst_signal2, entropy_signal1, ...
    """
    if engines is None:
        # Focus on most predictive engines
        engines = ['hurst', 'entropy', 'garch', 'lyapunov']

    print(f"  Pivoting vector features for engines: {engines}")

    # Filter to selected engines
    vec_filtered = vec.filter(pl.col('engine').is_in(engines))
    print(f"  Filtered rows: {len(vec_filtered):,}")

    # Create combined signal name: engine_source_signal
    vec_filtered = vec_filtered.with_columns(
        (pl.col('engine') + '_' + pl.col('source_signal')).alias('feature_name')
    )

    # Extract unit_id from entity_id
    vec_filtered = vec_filtered.with_columns(
        pl.col('entity_id').str.extract(r'U(\d+)', 1).cast(pl.Int64).alias('unit_id'),
        pl.col('timestamp').cast(pl.Int64).alias('cycle')
    )

    # Pivot to wide format
    pivoted = vec_filtered.pivot(
        values='value',
        index=['unit_id', 'cycle'],
        on='feature_name',
        aggregate_function='mean'  # In case of duplicates
    )

    print(f"  Pivoted shape: {pivoted.shape}")
    return pivoted


def main():
    cap_rul = 125

    print("="*70)
    print("PRISM vs BASELINE V2 - Full Feature Engineering")
    print("="*70)

    # =========================================================================
    # BASELINE
    # =========================================================================
    print("\n" + "="*70)
    print("BASELINE: Raw C-MAPSS Sensors")
    print("="*70)

    train_raw = load_cmapss(str(ML_DIR / "train_FD001.txt"))
    train_raw = add_rul(train_raw)
    train_raw['RUL'] = train_raw['RUL'].clip(upper=cap_rul)

    test_raw = load_cmapss(str(ML_DIR / "test_FD001.txt"))
    rul_actual = load_rul_file(str(ML_DIR / "RUL_FD001.txt"))
    rul_actual_capped = np.clip(rul_actual, 0, cap_rul)

    raw_feature_cols = [f'op_{i}' for i in range(1, 4)] + [f's_{i}' for i in range(1, 22)]

    X_train_raw = train_raw[raw_feature_cols].values
    y_train_raw = train_raw['RUL'].values

    X_train_b, X_val_b, y_train_b, y_val_b = train_test_split(
        X_train_raw, y_train_raw, test_size=0.2, random_state=42
    )

    print(f"\nTraining: {len(X_train_b):,} | Validation: {len(X_val_b):,} | Features: {len(raw_feature_cols)}")

    baseline_model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    baseline_model.fit(X_train_b, y_train_b)

    y_val_pred_b = baseline_model.predict(X_val_b)
    baseline_val_rmse = np.sqrt(mean_squared_error(y_val_b, y_val_pred_b))

    last_cycles = test_raw.groupby('unit_id').last().reset_index()
    X_test_b = last_cycles[raw_feature_cols].values
    y_test_pred_b = baseline_model.predict(X_test_b)

    baseline_test_rmse = np.sqrt(mean_squared_error(rul_actual_capped, y_test_pred_b))
    baseline_test_mae = mean_absolute_error(rul_actual_capped, y_test_pred_b)

    print(f"\nBASELINE: Val RMSE={baseline_val_rmse:.4f} | Test RMSE={baseline_test_rmse:.4f} | Test MAE={baseline_test_mae:.4f}")

    # =========================================================================
    # PRISM FEATURES
    # =========================================================================
    print("\n" + "="*70)
    print("PRISM: Raw + Vector + Geometry + State")
    print("="*70)

    # Load PRISM data
    print("\nLoading PRISM features...")

    train_vec = pl.read_parquet(TRAIN_DIR / "vector.parquet")
    train_geo = pl.read_parquet(TRAIN_DIR / "geometry.parquet")
    train_state = pl.read_parquet(TRAIN_DIR / "state.parquet")

    test_vec = pl.read_parquet(TEST_DIR / "vector.parquet")
    test_geo = pl.read_parquet(TEST_DIR / "geometry.parquet")
    test_state = pl.read_parquet(TEST_DIR / "state.parquet")

    print(f"  Train vector: {train_vec.shape}, geometry: {train_geo.shape}, state: {train_state.shape}")
    print(f"  Test vector: {test_vec.shape}, geometry: {test_geo.shape}, state: {test_state.shape}")

    # -------------------------------------------------------------------------
    # Pivot vector features
    # -------------------------------------------------------------------------
    print("\nPivoting vector features...")
    train_vec_wide = pivot_vector_features(train_vec, engines=['hurst', 'entropy', 'garch', 'lyapunov'])
    test_vec_wide = pivot_vector_features(test_vec, engines=['hurst', 'entropy', 'garch', 'lyapunov'])

    # -------------------------------------------------------------------------
    # Process geometry
    # -------------------------------------------------------------------------
    print("\nProcessing geometry features...")

    geo_exclude = ['entity_id', 'timestamp', 'signal_ids', 'computed_at', 'mode_id', 'n_features', 'n_engines']
    geo_feature_cols = [c for c in train_geo.columns if c not in geo_exclude
                        and train_geo[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

    train_geo = train_geo.with_columns([
        pl.col('entity_id').str.extract(r'U(\d+)', 1).cast(pl.Int64).alias('unit_id'),
        pl.col('timestamp').cast(pl.Int64).alias('cycle')
    ])
    test_geo = test_geo.with_columns([
        pl.col('entity_id').str.extract(r'U(\d+)', 1).cast(pl.Int64).alias('unit_id'),
        pl.col('timestamp').cast(pl.Int64).alias('cycle')
    ])

    print(f"  Geometry features: {len(geo_feature_cols)}")

    # -------------------------------------------------------------------------
    # Process state
    # -------------------------------------------------------------------------
    print("Processing state features...")

    state_exclude = ['entity_id', 'timestamp', 'state_label', 'failure_signature', 'mode_id']
    state_feature_cols = [c for c in train_state.columns if c not in state_exclude
                          and train_state[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

    train_state = train_state.with_columns([
        pl.col('entity_id').str.extract(r'U(\d+)', 1).cast(pl.Int64).alias('unit_id'),
        pl.col('timestamp').cast(pl.Int64).alias('cycle')
    ])
    test_state = test_state.with_columns([
        pl.col('entity_id').str.extract(r'U(\d+)', 1).cast(pl.Int64).alias('unit_id'),
        pl.col('timestamp').cast(pl.Int64).alias('cycle')
    ])

    print(f"  State features: {len(state_feature_cols)}")

    # -------------------------------------------------------------------------
    # Merge all features
    # -------------------------------------------------------------------------
    print("\nMerging features...")

    # Start with raw data
    train_merged = train_raw.copy()
    test_merged = test_raw.copy()

    # Merge vector (pivoted)
    vec_cols = [c for c in train_vec_wide.columns if c not in ['unit_id', 'cycle']]
    print(f"  Vector feature columns: {len(vec_cols)}")

    train_merged = train_merged.merge(
        train_vec_wide.to_pandas(),
        on=['unit_id', 'cycle'],
        how='left'
    )
    test_merged = test_merged.merge(
        test_vec_wide.to_pandas(),
        on=['unit_id', 'cycle'],
        how='left'
    )

    # Merge geometry
    train_merged = train_merged.merge(
        train_geo.select(['unit_id', 'cycle'] + geo_feature_cols).to_pandas(),
        on=['unit_id', 'cycle'],
        how='left'
    )
    test_merged = test_merged.merge(
        test_geo.select(['unit_id', 'cycle'] + geo_feature_cols).to_pandas(),
        on=['unit_id', 'cycle'],
        how='left'
    )

    # Merge state
    train_merged = train_merged.merge(
        train_state.select(['unit_id', 'cycle'] + state_feature_cols).to_pandas(),
        on=['unit_id', 'cycle'],
        how='left'
    )
    test_merged = test_merged.merge(
        test_state.select(['unit_id', 'cycle'] + state_feature_cols).to_pandas(),
        on=['unit_id', 'cycle'],
        how='left'
    )

    print(f"  Train merged: {train_merged.shape}")
    print(f"  Test merged: {test_merged.shape}")

    # -------------------------------------------------------------------------
    # Feature cleanup
    # -------------------------------------------------------------------------
    all_prism_cols = raw_feature_cols + vec_cols + geo_feature_cols + state_feature_cols

    # Keep only columns that exist and have data
    valid_cols = []
    for col in all_prism_cols:
        if col in train_merged.columns:
            non_null = train_merged[col].notna().sum()
            if non_null > 100:  # At least 100 non-null values
                valid_cols.append(col)

    print(f"\nValid feature columns: {len(valid_cols)}")
    print(f"  Raw: {len(raw_feature_cols)}")
    print(f"  Vector: {len([c for c in valid_cols if c in vec_cols])}")
    print(f"  Geometry: {len([c for c in valid_cols if c in geo_feature_cols])}")
    print(f"  State: {len([c for c in valid_cols if c in state_feature_cols])}")

    # Fill NaN and inf
    train_merged[valid_cols] = train_merged[valid_cols].fillna(0)
    test_merged[valid_cols] = test_merged[valid_cols].fillna(0)
    train_merged = train_merged.replace([np.inf, -np.inf], 0)
    test_merged = test_merged.replace([np.inf, -np.inf], 0)

    X_train_prism = train_merged[valid_cols].values
    y_train_prism = train_merged['RUL'].values

    X_train_p, X_val_p, y_train_p, y_val_p = train_test_split(
        X_train_prism, y_train_prism, test_size=0.2, random_state=42
    )

    print(f"\nTraining: {len(X_train_p):,} | Validation: {len(X_val_p):,} | Features: {len(valid_cols)}")

    # -------------------------------------------------------------------------
    # Train PRISM model with regularization
    # -------------------------------------------------------------------------
    print("\nTraining PRISM XGBoost...")

    prism_model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.5,  # More aggressive feature subsampling
        reg_alpha=0.5,  # L1 regularization
        reg_lambda=2.0,  # L2 regularization
        min_child_weight=5,  # Prevent overfitting to small samples
        random_state=42,
        n_jobs=-1,
    )
    prism_model.fit(X_train_p, y_train_p)

    y_val_pred_p = prism_model.predict(X_val_p)
    prism_val_rmse = np.sqrt(mean_squared_error(y_val_p, y_val_pred_p))

    # Test (last cycle per unit)
    last_cycles_prism = test_merged.groupby('unit_id').last().reset_index()
    X_test_p = last_cycles_prism[valid_cols].values
    y_test_pred_p = prism_model.predict(X_test_p)

    prism_test_rmse = np.sqrt(mean_squared_error(rul_actual_capped, y_test_pred_p))
    prism_test_mae = mean_absolute_error(rul_actual_capped, y_test_pred_p)

    print(f"\nPRISM: Val RMSE={prism_val_rmse:.4f} | Test RMSE={prism_test_rmse:.4f} | Test MAE={prism_test_mae:.4f}")

    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)

    print(f"\n{'Metric':<20} {'Baseline':<15} {'PRISM':<15} {'Winner':<15}")
    print("-"*70)

    val_winner = "PRISM" if prism_val_rmse < baseline_val_rmse else "Baseline"
    test_winner = "PRISM" if prism_test_rmse < baseline_test_rmse else "Baseline"
    mae_winner = "PRISM" if prism_test_mae < baseline_test_mae else "Baseline"

    print(f"{'Val RMSE':<20} {baseline_val_rmse:<15.4f} {prism_val_rmse:<15.4f} {val_winner:<15}")
    print(f"{'Test RMSE':<20} {baseline_test_rmse:<15.4f} {prism_test_rmse:<15.4f} {test_winner:<15}")
    print(f"{'Test MAE':<20} {baseline_test_mae:<15.4f} {prism_test_mae:<15.4f} {mae_winner:<15}")

    improvement = (baseline_test_rmse - prism_test_rmse) / baseline_test_rmse * 100
    if improvement > 0:
        print(f"\nPRISM improves Test RMSE by {improvement:.1f}%")
    else:
        print(f"\nBaseline wins by {-improvement:.1f}%")

    print(f"\nTarget benchmark: 6.62 RMSE")
    print(f"Best result: {min(baseline_test_rmse, prism_test_rmse):.2f}")
    print(f"Gap to target: {min(baseline_test_rmse, prism_test_rmse) - 6.62:.2f}")

    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================
    print("\n" + "="*70)
    print("TOP 30 FEATURES (PRISM model)")
    print("="*70)

    importances = prism_model.feature_importances_
    feature_importance = list(zip(valid_cols, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for i, (feat, imp) in enumerate(feature_importance[:30]):
        if feat in raw_feature_cols:
            marker = "[RAW]"
        elif feat in vec_cols:
            marker = "[VEC]"
        elif feat in geo_feature_cols:
            marker = "[GEO]"
        elif feat in state_feature_cols:
            marker = "[STATE]"
        else:
            marker = "[?]"
        print(f"  {i+1:2d}. {feat:<45} {imp:.4f}  {marker}")


if __name__ == "__main__":
    main()
