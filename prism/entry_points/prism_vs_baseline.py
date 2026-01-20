"""
PRISM vs Baseline XGBoost Comparison.

Uses observations + vector + geometry + state features
at per-timestamp level (not entity-level aggregation).

Compares against raw sensor baseline.
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
    """Load ground truth RUL file (one value per line)."""
    with open(path, 'r') as f:
        return np.array([float(line.strip()) for line in f if line.strip()])


def load_prism_features(data_dir: Path, is_train: bool = True) -> pl.DataFrame:
    """
    Load and merge all PRISM features at per-timestamp level.

    Returns DataFrame with entity_id, timestamp, and all features.
    """
    print(f"\nLoading PRISM features from {data_dir}...")

    # Load observations
    obs_path = data_dir / "observations.parquet"
    if obs_path.exists():
        obs = pl.read_parquet(obs_path)
        print(f"  Observations: {len(obs):,} rows, {len(obs.columns)} cols")
    else:
        raise FileNotFoundError(f"observations.parquet not found in {data_dir}")

    # Load vector
    vec_path = data_dir / "vector.parquet"
    if vec_path.exists():
        vec = pl.read_parquet(vec_path)
        print(f"  Vector: {len(vec):,} rows, {len(vec.columns)} cols")
    else:
        raise FileNotFoundError(f"vector.parquet not found in {data_dir}")

    # Load geometry
    geo_path = data_dir / "geometry.parquet"
    if geo_path.exists():
        geo = pl.read_parquet(geo_path)
        print(f"  Geometry: {len(geo):,} rows, {len(geo.columns)} cols")
    else:
        raise FileNotFoundError(f"geometry.parquet not found in {data_dir}")

    # Load state
    state_path = data_dir / "state.parquet"
    if state_path.exists():
        state = pl.read_parquet(state_path)
        print(f"  State: {len(state):,} rows, {len(state.columns)} cols")
    else:
        raise FileNotFoundError(f"state.parquet not found in {data_dir}")

    return obs, vec, geo, state


def extract_unit_timestamp(entity_id: str) -> tuple:
    """Extract unit_id and timestamp from PRISM entity_id like 'FD001_U091_C117'."""
    import re
    match = re.match(r'FD001_U(\d+)_C(\d+)', entity_id)
    if match:
        return int(match.group(1)), int(match.group(2))
    # Try simpler format
    match = re.match(r'FD001_U(\d+)', entity_id)
    if match:
        return int(match.group(1)), None
    return None, None


def main():
    cap_rul = 125

    print("="*60)
    print("PRISM vs BASELINE COMPARISON")
    print("="*60)

    # =========================================================================
    # STEP 1: Run baseline first
    # =========================================================================
    print("\n" + "="*60)
    print("BASELINE: Raw C-MAPSS Sensors")
    print("="*60)

    # Load raw train data
    train_raw = load_cmapss(str(ML_DIR / "train_FD001.txt"))
    train_raw = add_rul(train_raw)
    train_raw['RUL'] = train_raw['RUL'].clip(upper=cap_rul)

    # Load raw test data
    test_raw = load_cmapss(str(ML_DIR / "test_FD001.txt"))
    rul_actual = load_rul_file(str(ML_DIR / "RUL_FD001.txt"))
    rul_actual_capped = np.clip(rul_actual, 0, cap_rul)

    # Feature columns (raw sensors + op settings)
    raw_feature_cols = [f'op_{i}' for i in range(1, 4)] + [f's_{i}' for i in range(1, 22)]

    X_train_raw = train_raw[raw_feature_cols].values
    y_train_raw = train_raw['RUL'].values

    # Holdout validation split
    X_train_b, X_val_b, y_train_b, y_val_b = train_test_split(
        X_train_raw, y_train_raw, test_size=0.2, random_state=42
    )

    print(f"\nTraining set: {len(X_train_b):,} samples")
    print(f"Validation set: {len(X_val_b):,} samples")
    print(f"Features: {len(raw_feature_cols)} (raw sensors)")

    # Train baseline model
    baseline_model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining baseline XGBoost...")
    baseline_model.fit(X_train_b, y_train_b)

    # Baseline validation
    y_val_pred_b = baseline_model.predict(X_val_b)
    baseline_val_rmse = np.sqrt(mean_squared_error(y_val_b, y_val_pred_b))

    # Baseline test (last cycle per unit)
    last_cycles = test_raw.groupby('unit_id').last().reset_index()
    X_test_b = last_cycles[raw_feature_cols].values
    y_test_pred_b = baseline_model.predict(X_test_b)

    baseline_test_rmse = np.sqrt(mean_squared_error(rul_actual_capped, y_test_pred_b))
    baseline_test_mae = mean_absolute_error(rul_actual_capped, y_test_pred_b)

    print(f"\nBASELINE RESULTS:")
    print(f"  Validation RMSE: {baseline_val_rmse:.4f}")
    print(f"  Test RMSE: {baseline_test_rmse:.4f}")
    print(f"  Test MAE:  {baseline_test_mae:.4f}")

    # =========================================================================
    # STEP 2: PRISM features
    # =========================================================================
    print("\n" + "="*60)
    print("PRISM: Observations + Vector + Geometry + State")
    print("="*60)

    # Load PRISM data
    train_obs, train_vec, train_geo, train_state = load_prism_features(TRAIN_DIR, is_train=True)
    test_obs, test_vec, test_geo, test_state = load_prism_features(TEST_DIR, is_train=False)

    # Check schemas
    print("\nData shapes:")
    print(f"  Train observations: {train_obs.shape}")
    print(f"  Train vector: {train_vec.shape}")
    print(f"  Train geometry: {train_geo.shape}")
    print(f"  Train state: {train_state.shape}")

    # For geometry and state, we need to merge with raw data by entity+timestamp
    # First, let's understand the entity_id format in each
    print("\nEntity ID samples:")
    if 'entity_id' in train_geo.columns:
        print(f"  Geometry: {train_geo['entity_id'].head(3).to_list()}")
    if 'entity_id' in train_state.columns:
        print(f"  State: {train_state['entity_id'].head(3).to_list()}")
    if 'entity_id' in train_vec.columns:
        print(f"  Vector: {train_vec['entity_id'].head(3).to_list()}")
    if 'entity_id' in train_obs.columns:
        print(f"  Observations: {train_obs['entity_id'].head(3).to_list()}")

    # Extract timestamp column name
    ts_col = None
    for col in ['timestamp', 'obs_date', 'window_end', 'snapshot_idx']:
        if col in train_geo.columns:
            ts_col = col
            print(f"\nTimestamp column: {ts_col}")
            print(f"  Sample values: {train_geo[ts_col].head(3).to_list()}")
            break

    # Strategy: merge geometry and state features with raw train data
    # The raw data has (unit_id, cycle) which maps to timestamp

    # First, extract unit_id from PRISM entity_id
    def parse_entity_id(df: pl.DataFrame) -> pl.DataFrame:
        """Add unit_id column from entity_id."""
        if 'entity_id' in df.columns:
            # Extract unit number from 'FD001_U091' format
            df = df.with_columns(
                pl.col('entity_id').str.extract(r'U(\d+)', 1).cast(pl.Int64).alias('unit_id')
            )
        return df

    train_geo = parse_entity_id(train_geo)
    train_state = parse_entity_id(train_state)
    test_geo = parse_entity_id(test_geo)
    test_state = parse_entity_id(test_state)

    # Check for timestamp/cycle mapping
    # 'timestamp' in geometry/state = cycle number in raw data
    if 'timestamp' in train_geo.columns:
        train_geo = train_geo.with_columns(pl.col('timestamp').cast(pl.Int64).alias('cycle'))
        test_geo = test_geo.with_columns(pl.col('timestamp').cast(pl.Int64).alias('cycle'))
    if 'timestamp' in train_state.columns:
        train_state = train_state.with_columns(pl.col('timestamp').cast(pl.Int64).alias('cycle'))
        test_state = test_state.with_columns(pl.col('timestamp').cast(pl.Int64).alias('cycle'))

    print(f"\nAfter cycle mapping:")
    print(f"  Train geo has 'cycle': {'cycle' in train_geo.columns}")
    print(f"  Train state has 'cycle': {'cycle' in train_state.columns}")

    # Get numeric feature columns from geometry
    geo_exclude = ['entity_id', 'unit_id', 'cycle', 'timestamp', 'snapshot_idx', 'obs_date', 'window_end']
    geo_feature_cols = [c for c in train_geo.columns if c not in geo_exclude
                        and train_geo[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
    print(f"\nGeometry features: {len(geo_feature_cols)}")

    # Get numeric feature columns from state
    state_exclude = ['entity_id', 'unit_id', 'cycle', 'timestamp', 'snapshot_idx', 'obs_date', 'window_end']
    state_feature_cols = [c for c in train_state.columns if c not in state_exclude
                          and train_state[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
    print(f"State features: {len(state_feature_cols)}")

    # Convert to pandas for easier merging
    train_raw_df = train_raw.copy()
    test_raw_df = test_raw.copy()

    train_geo_pd = train_geo.to_pandas()
    train_state_pd = train_state.to_pandas()
    test_geo_pd = test_geo.to_pandas()
    test_state_pd = test_state.to_pandas()

    # Merge geometry features
    print("\nMerging geometry features with raw data...")
    if 'cycle' in train_geo_pd.columns:
        train_merged = train_raw_df.merge(
            train_geo_pd[['unit_id', 'cycle'] + geo_feature_cols],
            on=['unit_id', 'cycle'],
            how='left'
        )
    else:
        # Try alternative - join by unit_id only (entity-level)
        geo_agg = train_geo_pd.groupby('unit_id')[geo_feature_cols].mean().reset_index()
        train_merged = train_raw_df.merge(geo_agg, on='unit_id', how='left')

    print(f"  Train merged shape: {train_merged.shape}")

    # Merge state features
    print("Merging state features...")
    if 'cycle' in train_state_pd.columns:
        train_merged = train_merged.merge(
            train_state_pd[['unit_id', 'cycle'] + state_feature_cols],
            on=['unit_id', 'cycle'],
            how='left'
        )
    else:
        state_agg = train_state_pd.groupby('unit_id')[state_feature_cols].mean().reset_index()
        train_merged = train_merged.merge(state_agg, on='unit_id', how='left')

    print(f"  Train merged shape after state: {train_merged.shape}")

    # Same for test
    print("\nMerging test data...")
    if 'cycle' in test_geo_pd.columns:
        test_merged = test_raw_df.merge(
            test_geo_pd[['unit_id', 'cycle'] + geo_feature_cols],
            on=['unit_id', 'cycle'],
            how='left'
        )
    else:
        geo_agg = test_geo_pd.groupby('unit_id')[geo_feature_cols].mean().reset_index()
        test_merged = test_raw_df.merge(geo_agg, on='unit_id', how='left')

    if 'cycle' in test_state_pd.columns:
        test_merged = test_merged.merge(
            test_state_pd[['unit_id', 'cycle'] + state_feature_cols],
            on=['unit_id', 'cycle'],
            how='left'
        )
    else:
        state_agg = test_state_pd.groupby('unit_id')[state_feature_cols].mean().reset_index()
        test_merged = test_merged.merge(state_agg, on='unit_id', how='left')

    print(f"  Test merged shape: {test_merged.shape}")

    # All feature columns (raw + PRISM)
    prism_feature_cols = raw_feature_cols + geo_feature_cols + state_feature_cols

    # Remove any columns with all NaN
    valid_cols = []
    for col in prism_feature_cols:
        if col in train_merged.columns and train_merged[col].notna().sum() > 0:
            valid_cols.append(col)

    print(f"\nValid feature columns: {len(valid_cols)}")

    # Fill NaN with 0 (or could use mean imputation)
    train_merged[valid_cols] = train_merged[valid_cols].fillna(0)
    test_merged[valid_cols] = test_merged[valid_cols].fillna(0)

    # Check for any remaining NaN or inf
    train_merged = train_merged.replace([np.inf, -np.inf], 0)
    test_merged = test_merged.replace([np.inf, -np.inf], 0)

    X_train_prism = train_merged[valid_cols].values
    y_train_prism = train_merged['RUL'].values

    # Holdout validation split
    X_train_p, X_val_p, y_train_p, y_val_p = train_test_split(
        X_train_prism, y_train_prism, test_size=0.2, random_state=42
    )

    print(f"\nTraining set: {len(X_train_p):,} samples")
    print(f"Validation set: {len(X_val_p):,} samples")
    print(f"Features: {len(valid_cols)} (raw + PRISM)")

    # Train PRISM model - with stronger regularization to prevent overfitting
    prism_model = XGBRegressor(
        n_estimators=300,
        max_depth=4,  # Reduced from 6 to prevent overfitting
        learning_rate=0.05,  # Reduced from 0.1
        subsample=0.7,  # Reduced from 0.8
        colsample_bytree=0.6,  # Reduced from 0.8
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining PRISM XGBoost...")
    prism_model.fit(X_train_p, y_train_p)

    # PRISM validation
    y_val_pred_p = prism_model.predict(X_val_p)
    prism_val_rmse = np.sqrt(mean_squared_error(y_val_p, y_val_pred_p))

    # PRISM test (last cycle per unit)
    last_cycles_prism = test_merged.groupby('unit_id').last().reset_index()
    X_test_p = last_cycles_prism[valid_cols].values
    y_test_pred_p = prism_model.predict(X_test_p)

    prism_test_rmse = np.sqrt(mean_squared_error(rul_actual_capped, y_test_pred_p))
    prism_test_mae = mean_absolute_error(rul_actual_capped, y_test_pred_p)

    print(f"\nPRISM RESULTS:")
    print(f"  Validation RMSE: {prism_val_rmse:.4f}")
    print(f"  Test RMSE: {prism_test_rmse:.4f}")
    print(f"  Test MAE:  {prism_test_mae:.4f}")

    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)

    print(f"\n{'Metric':<20} {'Baseline':<15} {'PRISM':<15} {'Winner':<15}")
    print("-"*60)

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
    print(f"Gap to target: {prism_test_rmse - 6.62:.2f}")

    # Feature importance
    print("\n" + "="*60)
    print("TOP 20 FEATURES (PRISM model)")
    print("="*60)

    importances = prism_model.feature_importances_
    feature_importance = list(zip(valid_cols, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for feat, imp in feature_importance[:20]:
        is_prism = feat not in raw_feature_cols
        marker = "[PRISM]" if is_prism else "[RAW]"
        print(f"  {feat:<40} {imp:.4f}  {marker}")


if __name__ == "__main__":
    main()
