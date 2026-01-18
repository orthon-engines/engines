#!/usr/bin/env python3
"""
C-MAPSS FD001 Multi-Window Benchmark with PRISM Features.

Key insight: We need training samples at VARIOUS RUL values, not just RUL=0.

For each training engine, we create multiple samples by computing features
at different points in the trajectory:
- At cycle 50: RUL = max_cycle - 50
- At cycle 100: RUL = max_cycle - 100
- At end: RUL = 0

This teaches the model the relationship between behavioral features and RUL.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

DATA_DIR = Path('/Users/jasonrudder/prism-mac/data')
CMAPSS_DIR = DATA_DIR / 'CMAPSSData'
ML_DIR = Path('/Users/jasonrudder/prism-mac/notebooks/ml_accelerator')

# Import PRISM engines
import sys
sys.path.insert(0, str(Path('/Users/jasonrudder/prism-mac')))

from prism.engines.realized_vol import compute_realized_vol
from prism.engines.hilbert import compute_hilbert
from prism.engines.hurst import compute_hurst
from prism.engines.entropy import compute_entropy
from prism.engines.garch import compute_garch
from prism.engines.rqa import compute_rqa
from prism.engines.spectral import compute_spectral
from prism.engines.wavelet import compute_wavelets
from prism.engines.lyapunov import compute_lyapunov

# RUL capping
RUL_CAP = 125

# Sensor columns
SENSOR_COLS = [f's{i}' for i in range(1, 22)]

# Engine minimum observations
ENGINE_MIN_OBS = {
    'hurst': 20,
    'entropy': 30,
    'lyapunov': 30,
    'garch': 50,
    'spectral': 40,
    'wavelet': 40,
    'rqa': 30,
    'realized_vol': 15,
    'hilbert': 20,
}

# Core engines
CORE_ENGINES = {
    'hurst': compute_hurst,
    'entropy': compute_entropy,
    'realized_vol': compute_realized_vol,
    'hilbert': compute_hilbert,
    'rqa': compute_rqa,
}

EXTENDED_ENGINES = {
    'garch': compute_garch,
    'spectral': compute_spectral,
    'wavelet': compute_wavelets,
    'lyapunov': compute_lyapunov,
}


def compute_trajectory_features(values: np.ndarray, engines: Dict = None) -> Dict[str, float]:
    """Compute PRISM features on a trajectory."""
    if engines is None:
        engines = CORE_ENGINES

    features = {}
    n = len(values)

    for engine_name, engine_func in engines.items():
        min_obs = ENGINE_MIN_OBS.get(engine_name, 15)

        if n < min_obs:
            continue

        try:
            try:
                metrics = engine_func(values, min_obs=min_obs)
            except TypeError:
                metrics = engine_func(values)

            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    try:
                        numeric_value = float(metric_value)
                        if np.isfinite(numeric_value):
                            features[f'{engine_name}_{metric_name}'] = numeric_value
                    except (TypeError, ValueError):
                        continue
        except Exception:
            continue

    return features


def compute_unit_features_at_cycle(unit_df: pl.DataFrame, cutoff_cycle: int) -> Dict[str, float]:
    """
    Compute PRISM features for a unit at a specific cutoff cycle.

    Only uses data up to and including cutoff_cycle.
    """
    # Filter to cycles up to cutoff
    partial_df = unit_df.filter(pl.col('cycle') <= cutoff_cycle)
    n = len(partial_df)

    if n < 15:
        return {}

    # Select engines based on trajectory length
    engines = CORE_ENGINES.copy()
    if n >= 50:
        engines.update(EXTENDED_ENGINES)

    # Compute features for each sensor
    sensor_features = []
    for sensor in SENSOR_COLS:
        if sensor not in partial_df.columns:
            continue

        values = partial_df[sensor].to_numpy()
        valid_mask = ~np.isnan(values)
        if valid_mask.sum() < 15:
            continue

        clean_values = values[valid_mask]
        features = compute_trajectory_features(clean_values, engines)

        if features:
            sensor_features.append(features)

    if not sensor_features:
        return {}

    # Aggregate across sensors
    all_feature_names = set()
    for sf in sensor_features:
        all_feature_names.update(sf.keys())

    all_features = {}
    for feature_name in all_feature_names:
        values = [sf.get(feature_name) for sf in sensor_features if feature_name in sf]
        if values:
            all_features[feature_name] = np.mean(values)
            all_features[f'{feature_name}_std'] = np.std(values)

    return all_features


def create_multiwindow_training_data(sample_points: List[int] = None) -> pl.DataFrame:
    """
    Create training data with samples at multiple points in each trajectory.

    Args:
        sample_points: List of cycle points to sample at.
                      Default: [30, 50, 75, 100, 125, 150, 175, 200, 225, 250]

    Returns:
        DataFrame with features and RUL labels
    """
    if sample_points is None:
        # Sample at multiple points - covers range of typical test lengths
        sample_points = [30, 50, 75, 100, 125, 150, 175, 200]

    train_df = pl.read_parquet(CMAPSS_DIR / 'train_FD001.parquet')

    # Get max cycle per unit
    max_cycles = train_df.group_by('unit').agg(
        pl.col('cycle').max().alias('max_cycle')
    )

    units = train_df['unit'].unique().sort().to_list()
    print(f"Creating multi-window training data for {len(units)} units...")
    print(f"Sample points: {sample_points}")

    rows = []
    for unit in units:
        unit_df = train_df.filter(pl.col('unit') == unit)
        max_cycle = int(max_cycles.filter(pl.col('unit') == unit)['max_cycle'][0])

        # Sample at each point
        for cutoff in sample_points:
            if cutoff > max_cycle - 10:  # Need at least 10 cycles after cutoff
                continue

            # Compute RUL at this cutoff
            rul = max_cycle - cutoff
            rul_capped = min(rul, RUL_CAP)

            # Compute features at this cutoff
            features = compute_unit_features_at_cycle(unit_df, cutoff)

            if features:
                row = {
                    'unit': unit,
                    'cutoff_cycle': cutoff,
                    'max_cycle': max_cycle,
                    'rul': rul_capped,
                }
                row.update(features)
                rows.append(row)

        # Also sample at end (RUL=0)
        features_end = compute_unit_features_at_cycle(unit_df, max_cycle)
        if features_end:
            row = {
                'unit': unit,
                'cutoff_cycle': max_cycle,
                'max_cycle': max_cycle,
                'rul': 0,
            }
            row.update(features_end)
            rows.append(row)

    result = pl.DataFrame(rows, infer_schema_length=None)
    print(f"Created {len(result)} training samples from {len(units)} units")
    print(f"RUL distribution: min={result['rul'].min()}, max={result['rul'].max()}, mean={result['rul'].mean():.1f}")

    return result


def load_test_data() -> Tuple[pl.DataFrame, np.ndarray]:
    """Load test features and RUL labels."""
    test_features = pl.read_parquet(ML_DIR / 'test_trajectory_features.parquet')

    # Load RUL
    rul_path = CMAPSS_DIR / 'RUL_FD001.txt'
    with open(rul_path) as f:
        rul_values = [int(line.strip()) for line in f]

    rul_df = pl.DataFrame({
        'unit': list(range(1, len(rul_values) + 1)),
        'rul_actual': rul_values,
    })

    # Merge and cap
    test = test_features.join(rul_df, on='unit', how='inner')
    test = test.with_columns([
        pl.when(pl.col('rul_actual') > RUL_CAP)
        .then(RUL_CAP)
        .otherwise(pl.col('rul_actual'))
        .alias('rul')
    ])

    return test


def get_common_features(train: pl.DataFrame, test: pl.DataFrame) -> list:
    """Get feature columns present in both train and test."""
    train_cols = set(train.columns)
    test_cols = set(test.columns)

    exclude = {'unit', 'n_cycles', 'max_cycle', 'cutoff_cycle', 'rul', 'rul_actual'}
    common = (train_cols & test_cols) - exclude

    return sorted(list(common))


def run_benchmark():
    """Run the multi-window benchmark."""
    print("=" * 80)
    print("C-MAPSS FD001 MULTI-WINDOW BENCHMARK")
    print("PRISM Features + XGBoost")
    print("=" * 80)
    print()

    # Create multi-window training data
    print("=== CREATING TRAINING DATA ===")
    train = create_multiwindow_training_data()

    # Load test data
    print("\n=== LOADING TEST DATA ===")
    test = load_test_data()
    print(f"Test units: {len(test)}")

    # Get common features
    features = get_common_features(train, test)
    print(f"\nCommon features: {len(features)}")

    # Prepare arrays
    X_train = train.select(features).to_numpy()
    y_train = train['rul'].to_numpy()

    X_test = test.select(features).to_numpy()
    y_test = test['rul'].to_numpy()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train range: [{y_train.min()}, {y_train.max()}]")
    print(f"y_test range: [{y_test.min()}, {y_test.max()}]")

    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # Split training data for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    print(f"\nTraining split: {len(X_tr)} train, {len(X_val)} val")

    # Train XGBoost
    print("\n=== TRAINING XGBOOST ===")
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print(f"Best iteration: {model.best_iteration}")

    # Validation performance
    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    print(f"Validation RMSE: {val_rmse:.2f}")

    # Predict on test set
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, RUL_CAP)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 80)
    print("RESULTS (100 Test Units)")
    print("=" * 80)
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.4f}")

    # Feature importance
    print("\n=== TOP 20 FEATURES ===")
    importance = model.feature_importances_
    feature_importance = list(zip(features, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for name, imp in feature_importance[:20]:
        print(f"  {name}: {imp*100:.2f}%")

    # Error distribution
    errors = y_pred - y_test
    print(f"\n=== ERROR DISTRIBUTION ===")
    print(f"Mean error: {errors.mean():+.2f}")
    print(f"Std error: {errors.std():.2f}")
    print(f"Median error: {np.median(errors):+.2f}")

    for tol in [10, 15, 20, 25, 30]:
        within = np.sum(np.abs(errors) <= tol)
        pct = within / len(errors) * 100
        print(f"Within +/-{tol}: {within}/100 ({pct:.0f}%)")

    # Sample predictions
    print("\n=== SAMPLE PREDICTIONS ===")
    print("Unit | Actual | Predicted | Error")
    print("-" * 45)
    units = test['unit'].to_numpy()
    indices = np.random.choice(len(test), min(15, len(test)), replace=False)
    for i in sorted(indices):
        unit = units[i]
        actual = y_test[i]
        pred = y_pred[i]
        error = pred - actual
        print(f"  {unit:3d} | {actual:6.0f} | {pred:9.1f} | {error:+7.1f}")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON WITH PUBLISHED BENCHMARKS")
    print("=" * 80)
    print(f"| Model | Test Units | RMSE | Hardware |")
    print(f"|-------|------------|------|----------|")
    print(f"| **PRISM + XGBoost** | **100** | **{rmse:.2f}** | Mac Mini |")
    print(f"| LSTM | 100 | 13-16 | GPU |")
    print(f"| CNN | 100 | 12-14 | GPU |")
    print(f"| Previous PRISM | 58 | 43.87 | Mac Mini |")

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'n_train_samples': len(X_train),
        'n_test': len(test),
        'n_features': len(features),
    }


if __name__ == '__main__':
    results = run_benchmark()
