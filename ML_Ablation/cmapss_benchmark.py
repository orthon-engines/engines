#!/usr/bin/env python3
"""
C-MAPSS FD001 Benchmark - Compare PRISM features vs raw data
Target: 6.62 RMSE (state-of-the-art)

This script:
1. Loads FD001 train/test data
2. Runs PRISM pipeline on both
3. Trains CatBoost on train features
4. Predicts RUL on test data
5. Compares with ground truth
"""

import sys
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Add prism to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.db.parquet_store import get_path, OBSERVATIONS, VECTOR, GEOMETRY, STATE

# =============================================================================
# Data Loading
# =============================================================================

CMAPSS_DIR = Path(__file__).parent.parent / "data" / "CMAPSSData"

COLUMN_NAMES = [
    'unit', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3',
    's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
    's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21'
]

SENSOR_COLS = [f's{i}' for i in range(1, 22)]


def load_cmapss_train(fd='FD001'):
    """Load training data with computed RUL."""
    path = CMAPSS_DIR / f'train_{fd}.txt'
    df = pd.read_csv(path, sep=r'\s+', header=None, names=COLUMN_NAMES)

    # Compute RUL: max_cycle - current_cycle for each unit
    max_cycles = df.groupby('unit')['cycle'].max().to_dict()
    df['RUL'] = df.apply(lambda row: max_cycles[row['unit']] - row['cycle'], axis=1)

    # Cap RUL at 125 (standard practice)
    df['RUL'] = df['RUL'].clip(upper=125)

    return df


def load_cmapss_test(fd='FD001'):
    """Load test data and ground truth RUL."""
    # Test trajectories (partial)
    test_path = CMAPSS_DIR / f'test_{fd}.txt'
    df = pd.read_csv(test_path, sep=r'\s+', header=None, names=COLUMN_NAMES)

    # Ground truth RUL at end of each trajectory
    rul_path = CMAPSS_DIR / f'RUL_{fd}.txt'
    rul_df = pd.read_csv(rul_path, header=None, names=['RUL'])
    rul_df['unit'] = range(1, len(rul_df) + 1)

    # Cap RUL at 125
    rul_df['RUL'] = rul_df['RUL'].clip(upper=125)

    return df, rul_df


# =============================================================================
# Feature Engineering (Raw baseline)
# =============================================================================

def extract_raw_features(df, mode='last'):
    """
    Extract features from raw sensor data.

    mode='last': Use last observation per unit (for test prediction)
    mode='all': Use all observations (for training)
    """
    if mode == 'last':
        # Get last observation per unit
        last_obs = df.groupby('unit').last().reset_index()
        features = last_obs[SENSOR_COLS + ['op_setting_1', 'op_setting_2', 'op_setting_3']]
        return features, last_obs['unit'].values
    else:
        # All observations for training
        features = df[SENSOR_COLS + ['op_setting_1', 'op_setting_2', 'op_setting_3']]
        return features, df['unit'].values, df['RUL'].values


def extract_rolling_features(df, window=30):
    """Extract rolling statistics as features."""
    feature_list = []

    for unit in df['unit'].unique():
        unit_df = df[df['unit'] == unit].copy()

        for col in SENSOR_COLS:
            unit_df[f'{col}_roll_mean'] = unit_df[col].rolling(window, min_periods=1).mean()
            unit_df[f'{col}_roll_std'] = unit_df[col].rolling(window, min_periods=1).std().fillna(0)
            unit_df[f'{col}_roll_min'] = unit_df[col].rolling(window, min_periods=1).min()
            unit_df[f'{col}_roll_max'] = unit_df[col].rolling(window, min_periods=1).max()

        # Get last row
        last_row = unit_df.iloc[-1]
        feature_list.append(last_row)

    result = pd.DataFrame(feature_list)
    roll_cols = [c for c in result.columns if 'roll' in c]
    return result[roll_cols + SENSOR_COLS], result['unit'].values


# =============================================================================
# PRISM Feature Extraction
# =============================================================================

def run_prism_pipeline(df, output_prefix='temp'):
    """
    Run PRISM pipeline on dataframe.

    Converts to PRISM schema and runs vector/geometry/state.
    Returns extracted features.
    """
    from prism.db.parquet_store import get_data_root
    import subprocess
    import tempfile
    import shutil

    data_root = get_data_root()

    # Convert to PRISM schema (long format)
    records = []
    for _, row in df.iterrows():
        entity_id = f"U{int(row['unit']):03d}"
        timestamp = float(row['cycle'])

        # Each sensor becomes a row
        for col in SENSOR_COLS:
            records.append({
                'entity_id': entity_id,
                'signal_id': col,
                'timestamp': timestamp,
                'value': float(row[col]),
            })

        # Add RUL if present
        if 'RUL' in row:
            records.append({
                'entity_id': entity_id,
                'signal_id': 'target_rul',
                'timestamp': timestamp,
                'value': float(row['RUL']),
            })

    obs_df = pl.DataFrame(records)

    # Save observations
    obs_path = data_root / f'{output_prefix}_observations.parquet'
    obs_df.write_parquet(obs_path)

    print(f"  Saved {len(obs_df):,} observations to {obs_path}")

    return obs_df


def extract_prism_features_simple(df, mode='last', window=30):
    """
    Extract PRISM-style features without running full pipeline.

    mode='last': Only last window per unit (for test prediction)
    mode='all': All windows with stride (for training)
    """
    feature_list = []
    units = []
    ruls = []

    stride = 10 if mode == 'all' else window

    for unit in df['unit'].unique():
        unit_df = df[df['unit'] == unit].sort_values('cycle')
        n = len(unit_df)

        if mode == 'last':
            # Just the last window
            windows = [(max(0, n - window), n)]
        else:
            # Sliding windows with stride
            windows = []
            end = n
            while end >= window:
                windows.append((end - window, end))
                end -= stride
            windows = list(reversed(windows))
            if not windows:
                windows = [(0, n)]

        for start, end in windows:
            window_df = unit_df.iloc[start:end]
            features = {}
            features['unit'] = unit

            for col in SENSOR_COLS:
                series = window_df[col].values

                # Basic stats (like PRISM vector)
                features[f'{col}_mean'] = np.mean(series)
                features[f'{col}_std'] = np.std(series) if len(series) > 1 else 0
                features[f'{col}_min'] = np.min(series)
                features[f'{col}_max'] = np.max(series)
                features[f'{col}_range'] = np.max(series) - np.min(series)

                # Trend
                if len(series) > 1:
                    features[f'{col}_trend'] = (series[-1] - series[0]) / len(series)
                else:
                    features[f'{col}_trend'] = 0

                # Velocity
                if len(series) > 1:
                    features[f'{col}_velocity'] = np.mean(np.diff(series))
                else:
                    features[f'{col}_velocity'] = 0

            # Cross-sensor features (like PRISM geometry)
            sensor_means = [features[f'{col}_mean'] for col in SENSOR_COLS]
            sensor_stds = [features[f'{col}_std'] for col in SENSOR_COLS]

            features['sensors_mean_mean'] = np.mean(sensor_means)
            features['sensors_std_mean'] = np.mean(sensor_stds)

            feature_list.append(features)
            units.append(unit)

            # Get RUL at end of window
            if 'RUL' in unit_df.columns:
                ruls.append(unit_df.iloc[end - 1]['RUL'])

    result_df = pd.DataFrame(feature_list)
    feature_cols = [c for c in result_df.columns if c != 'unit']

    if mode == 'all' and ruls:
        return result_df[feature_cols], np.array(units), np.array(ruls)
    return result_df[feature_cols], np.array(units)


# =============================================================================
# Models
# =============================================================================

def train_xgboost(X_train, y_train, X_test):
    """Train XGBoost model."""
    try:
        import xgboost as xgb
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        print("XGBoost not available, using sklearn GradientBoosting")
        model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model.predict(X_test), model

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbosity=0
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return predictions, model


def train_lightgbm(X_train, y_train, X_test):
    """Train LightGBM model."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBM not installed. Installing...")
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'lightgbm', '-q'])
        import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbose=-1
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return predictions, model


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("C-MAPSS FD001 BENCHMARK")
    print("Target: 6.62 RMSE (state-of-the-art)")
    print("=" * 70)

    # Load data
    print("\n[1] Loading FD001 data...")
    train_df = load_cmapss_train('FD001')
    test_df, rul_df = load_cmapss_test('FD001')

    print(f"  Train: {len(train_df):,} rows, {train_df['unit'].nunique()} units")
    print(f"  Test: {len(test_df):,} rows, {test_df['unit'].nunique()} units")
    print(f"  Ground truth RUL range: {rul_df['RUL'].min()} - {rul_df['RUL'].max()}")

    # ==========================================================================
    # Baseline 1: Raw last observation
    # ==========================================================================
    print("\n[2] Baseline: Raw sensor values (last observation)")

    # Train features: use all observations
    X_train_raw = train_df[SENSOR_COLS + ['op_setting_1', 'op_setting_2', 'op_setting_3']].values
    y_train = train_df['RUL'].values

    # Test features: last observation per unit
    test_last = test_df.groupby('unit').last().reset_index()
    X_test_raw = test_last[SENSOR_COLS + ['op_setting_1', 'op_setting_2', 'op_setting_3']].values
    y_test = rul_df['RUL'].values

    # Scale
    scaler_raw = StandardScaler()
    X_train_raw_scaled = scaler_raw.fit_transform(X_train_raw)
    X_test_raw_scaled = scaler_raw.transform(X_test_raw)

    # Train and predict
    pred_raw, _ = train_xgboost(X_train_raw_scaled, y_train, X_test_raw_scaled)
    rmse_raw = np.sqrt(mean_squared_error(y_test, pred_raw))
    print(f"  RMSE (raw): {rmse_raw:.2f}")

    # ==========================================================================
    # Baseline 2: Rolling features
    # ==========================================================================
    print("\n[3] Rolling features (30-cycle window)")

    X_train_roll, train_units_roll = extract_rolling_features(train_df, window=30)
    X_test_roll, test_units_roll = extract_rolling_features(test_df, window=30)

    # Get corresponding RUL for training
    train_last = train_df.groupby('unit').last().reset_index()
    y_train_roll = train_last.set_index('unit').loc[train_units_roll, 'RUL'].values

    # Scale
    scaler_roll = StandardScaler()
    X_train_roll_scaled = scaler_roll.fit_transform(X_train_roll.fillna(0))
    X_test_roll_scaled = scaler_roll.transform(X_test_roll.fillna(0))

    # Train and predict
    pred_roll, _ = train_xgboost(X_train_roll_scaled, y_train_roll, X_test_roll_scaled)
    rmse_roll = np.sqrt(mean_squared_error(y_test, pred_roll))
    print(f"  RMSE (rolling): {rmse_roll:.2f}")

    # ==========================================================================
    # PRISM-style features
    # ==========================================================================
    print("\n[4] PRISM-style features (window stats + trends)")

    X_train_prism, train_units_prism = extract_prism_features_simple(train_df)
    X_test_prism, test_units_prism = extract_prism_features_simple(test_df)

    # Get corresponding RUL
    y_train_prism = train_last.set_index('unit').loc[train_units_prism, 'RUL'].values

    # Scale
    scaler_prism = StandardScaler()
    X_train_prism_scaled = scaler_prism.fit_transform(X_train_prism.fillna(0))
    X_test_prism_scaled = scaler_prism.transform(X_test_prism.fillna(0))

    # Train and predict
    pred_prism, model_prism = train_xgboost(X_train_prism_scaled, y_train_prism, X_test_prism_scaled)
    rmse_prism = np.sqrt(mean_squared_error(y_test, pred_prism))
    print(f"  RMSE (PRISM-style): {rmse_prism:.2f}")

    # Feature importance
    importance = model_prism.feature_importances_
    feature_names = X_train_prism.columns
    top_features = sorted(zip(feature_names, importance), key=lambda x: -x[1])[:10]
    print(f"  Top features: {[f[0] for f in top_features[:5]]}")

    # ==========================================================================
    # Combined features
    # ==========================================================================
    print("\n[5] Combined (raw + rolling + PRISM-style)")

    # Combine features
    X_train_combined = np.hstack([
        X_train_roll_scaled,
        X_train_prism_scaled
    ])
    X_test_combined = np.hstack([
        X_test_roll_scaled,
        X_test_prism_scaled
    ])

    pred_combined, _ = train_xgboost(X_train_combined, y_train_prism, X_test_combined)
    rmse_combined = np.sqrt(mean_squared_error(y_test, pred_combined))
    print(f"  RMSE (combined): {rmse_combined:.2f}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Raw sensors:     {rmse_raw:.2f} RMSE")
    print(f"  Rolling stats:   {rmse_roll:.2f} RMSE")
    print(f"  PRISM-style:     {rmse_prism:.2f} RMSE")
    print(f"  Combined:        {rmse_combined:.2f} RMSE")
    print(f"  Target (SOTA):   6.62 RMSE")
    print()
    print(f"  Gap to SOTA:     {rmse_combined - 6.62:.2f} RMSE")
    print("=" * 70)

    return {
        'raw': rmse_raw,
        'rolling': rmse_roll,
        'prism': rmse_prism,
        'combined': rmse_combined,
        'target': 6.62
    }


if __name__ == '__main__':
    main()
