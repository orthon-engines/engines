"""
Train PRISM model on FD002 data.
"""
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import pickle
import json

# C-MAPSS columns
COLUMNS = ['unit_id', 'cycle'] + [f'op_{i}' for i in range(1, 4)] + [f's_{i}' for i in range(1, 22)]

DATA_DIR = Path("/Users/jasonrudder/prism-mac/data")
FD002_DIR = DATA_DIR / "FD002"

def load_cmapss(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r'\s+', header=None, names=COLUMNS)
    return df

def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    max_cycles = df.groupby('unit_id')['cycle'].max().rename('max_cycle')
    df = df.merge(max_cycles, on='unit_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df = df.drop(columns=['max_cycle'])
    return df

def load_rul_file(path: str) -> np.ndarray:
    with open(path, 'r') as f:
        return np.array([float(line.strip()) for line in f if line.strip()])

def pivot_vector_features(vec: pl.DataFrame, engines: list = None) -> pl.DataFrame:
    if engines is None:
        engines = ['hurst', 'entropy', 'garch', 'lyapunov']

    vec_filtered = vec.filter(pl.col('engine').is_in(engines))
    vec_filtered = vec_filtered.with_columns(
        (pl.col('engine') + '_' + pl.col('source_signal')).alias('feature_name')
    )
    vec_filtered = vec_filtered.with_columns(
        pl.col('entity_id').str.extract(r'U(\d+)', 1).cast(pl.Int64).alias('unit_id'),
        pl.col('timestamp').cast(pl.Int64).alias('cycle')
    )

    pivoted = vec_filtered.pivot(
        values='value',
        index=['unit_id', 'cycle'],
        on='feature_name',
        aggregate_function='mean'
    )
    return pivoted

def main():
    cap_rul = 125

    print("="*70)
    print("PRISM FD002 TRAINING")
    print("="*70)

    # Load raw train data
    print("\nLoading raw data...")
    train_raw = load_cmapss(str(DATA_DIR / "train_FD002.txt"))
    train_raw = add_rul(train_raw)
    train_raw['RUL'] = train_raw['RUL'].clip(upper=cap_rul)
    print(f"  Train: {len(train_raw):,} rows, {train_raw['unit_id'].nunique()} units")

    # Load test data
    test_raw = load_cmapss(str(DATA_DIR / "test_FD002.txt"))
    rul_actual = load_rul_file(str(DATA_DIR / "RUL_FD002.txt"))
    rul_actual_capped = np.clip(rul_actual, 0, cap_rul)
    print(f"  Test: {len(test_raw):,} rows, {test_raw['unit_id'].nunique()} units")

    raw_feature_cols = [f'op_{i}' for i in range(1, 4)] + [f's_{i}' for i in range(1, 22)]

    # =========================================================================
    # BASELINE
    # =========================================================================
    print("\n" + "="*70)
    print("BASELINE: Raw Sensors")
    print("="*70)

    X_train_raw = train_raw[raw_feature_cols].values
    y_train_raw = train_raw['RUL'].values

    X_train_b, X_val_b, y_train_b, y_val_b = train_test_split(
        X_train_raw, y_train_raw, test_size=0.2, random_state=42
    )

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

    print(f"Baseline: Val RMSE={baseline_val_rmse:.2f} | Test RMSE={baseline_test_rmse:.2f} | MAE={baseline_test_mae:.2f}")

    # =========================================================================
    # PRISM
    # =========================================================================
    print("\n" + "="*70)
    print("PRISM: Raw + Vector + Geometry + State")
    print("="*70)

    # Load PRISM features
    print("\nLoading PRISM features...")
    vec = pl.read_parquet(FD002_DIR / "vector.parquet")
    geo = pl.read_parquet(FD002_DIR / "geometry.parquet")
    state = pl.read_parquet(FD002_DIR / "state.parquet")

    print(f"  Vector: {len(vec):,} rows")
    print(f"  Geometry: {len(geo):,} rows")
    print(f"  State: {len(state):,} rows")

    # Pivot vector
    print("\nPivoting vector features...")
    vec_wide = pivot_vector_features(vec)
    print(f"  Pivoted: {vec_wide.shape}")

    # Process geometry
    geo_exclude = ['entity_id', 'timestamp', 'signal_ids', 'computed_at', 'mode_id', 'n_features', 'n_engines']
    geo_feature_cols = [c for c in geo.columns if c not in geo_exclude
                        and geo[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

    geo = geo.with_columns([
        pl.col('entity_id').str.extract(r'U(\d+)', 1).cast(pl.Int64).alias('unit_id'),
        pl.col('timestamp').cast(pl.Int64).alias('cycle')
    ])

    # Process state
    state_exclude = ['entity_id', 'timestamp', 'state_label', 'failure_signature', 'mode_id']
    state_feature_cols = [c for c in state.columns if c not in state_exclude
                          and state[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

    state = state.with_columns([
        pl.col('entity_id').str.extract(r'U(\d+)', 1).cast(pl.Int64).alias('unit_id'),
        pl.col('timestamp').cast(pl.Int64).alias('cycle')
    ])

    print(f"  Geometry features: {len(geo_feature_cols)}")
    print(f"  State features: {len(state_feature_cols)}")

    # Merge features
    print("\nMerging features...")
    train_merged = train_raw.copy()

    vec_cols = [c for c in vec_wide.columns if c not in ['unit_id', 'cycle']]
    train_merged = train_merged.merge(vec_wide.to_pandas(), on=['unit_id', 'cycle'], how='left')
    train_merged = train_merged.merge(
        geo.select(['unit_id', 'cycle'] + geo_feature_cols).to_pandas(),
        on=['unit_id', 'cycle'], how='left'
    )
    train_merged = train_merged.merge(
        state.select(['unit_id', 'cycle'] + state_feature_cols).to_pandas(),
        on=['unit_id', 'cycle'], how='left'
    )

    print(f"  Merged shape: {train_merged.shape}")

    # All feature columns
    all_prism_cols = raw_feature_cols + vec_cols + geo_feature_cols + state_feature_cols

    valid_cols = []
    for col in all_prism_cols:
        if col in train_merged.columns:
            non_null = train_merged[col].notna().sum()
            if non_null > 100:
                valid_cols.append(col)

    print(f"  Valid features: {len(valid_cols)}")

    # Fill NaN
    train_merged[valid_cols] = train_merged[valid_cols].fillna(0)
    train_merged = train_merged.replace([np.inf, -np.inf], 0)

    X_train_prism = train_merged[valid_cols].values
    y_train_prism = train_merged['RUL'].values

    X_train_p, X_val_p, y_train_p, y_val_p = train_test_split(
        X_train_prism, y_train_prism, test_size=0.2, random_state=42
    )

    print(f"\nTraining PRISM model...")
    print(f"  Train: {len(X_train_p):,} | Val: {len(X_val_p):,} | Features: {len(valid_cols)}")

    prism_model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.5,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
    )
    prism_model.fit(X_train_p, y_train_p)

    y_val_pred_p = prism_model.predict(X_val_p)
    prism_val_rmse = np.sqrt(mean_squared_error(y_val_p, y_val_pred_p))

    # Test - need to merge test data with PRISM features
    # For now, use raw test with entity-level PRISM features
    test_merged = test_raw.copy()
    test_merged = test_merged.merge(vec_wide.to_pandas(), on=['unit_id', 'cycle'], how='left')
    test_merged = test_merged.merge(
        geo.select(['unit_id', 'cycle'] + geo_feature_cols).to_pandas(),
        on=['unit_id', 'cycle'], how='left'
    )
    test_merged = test_merged.merge(
        state.select(['unit_id', 'cycle'] + state_feature_cols).to_pandas(),
        on=['unit_id', 'cycle'], how='left'
    )

    # Ensure all valid_cols exist
    for col in valid_cols:
        if col not in test_merged.columns:
            test_merged[col] = 0

    test_merged[valid_cols] = test_merged[valid_cols].fillna(0)
    test_merged = test_merged.replace([np.inf, -np.inf], 0)

    last_cycles_prism = test_merged.groupby('unit_id').last().reset_index()
    X_test_p = last_cycles_prism[valid_cols].values
    y_test_pred_p = prism_model.predict(X_test_p)

    prism_test_rmse = np.sqrt(mean_squared_error(rul_actual_capped, y_test_pred_p))
    prism_test_mae = mean_absolute_error(rul_actual_capped, y_test_pred_p)

    print(f"\nPRISM: Val RMSE={prism_val_rmse:.2f} | Test RMSE={prism_test_rmse:.2f} | MAE={prism_test_mae:.2f}")

    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(f"\n{'Model':<15} {'Val RMSE':<12} {'Test RMSE':<12} {'Test MAE':<12}")
    print("-"*50)
    print(f"{'Baseline':<15} {baseline_val_rmse:<12.2f} {baseline_test_rmse:<12.2f} {baseline_test_mae:<12.2f}")
    print(f"{'PRISM':<15} {prism_val_rmse:<12.2f} {prism_test_rmse:<12.2f} {prism_test_mae:<12.2f}")

    improvement = (baseline_test_rmse - prism_test_rmse) / baseline_test_rmse * 100
    if improvement > 0:
        print(f"\nPRISM improves by {improvement:.1f}%")
    else:
        print(f"\nBaseline wins by {-improvement:.1f}%")

    # Save model
    model_path = DATA_DIR / "fd002_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(prism_model, f)
    print(f"\nModel saved: {model_path}")

    # Save metadata
    metadata = {
        'dataset': 'FD002',
        'train_units': int(train_raw['unit_id'].nunique()),
        'test_units': int(test_raw['unit_id'].nunique()),
        'features': len(valid_cols),
        'baseline_test_rmse': float(baseline_test_rmse),
        'prism_test_rmse': float(prism_test_rmse),
        'improvement_pct': float(improvement),
    }
    with open(DATA_DIR / "fd002_model.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Feature importance
    print("\n" + "="*70)
    print("TOP 20 FEATURES")
    print("="*70)

    importances = prism_model.feature_importances_
    feature_importance = sorted(zip(valid_cols, importances), key=lambda x: x[1], reverse=True)

    for i, (feat, imp) in enumerate(feature_importance[:20]):
        if feat in raw_feature_cols:
            src = "RAW"
        elif feat in vec_cols:
            src = "VEC"
        elif feat in geo_feature_cols:
            src = "GEO"
        else:
            src = "STATE"
        print(f"  {i+1:2d}. {feat:<40} {imp:.4f} [{src}]")

if __name__ == "__main__":
    main()
