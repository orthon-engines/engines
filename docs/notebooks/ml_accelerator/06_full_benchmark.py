#!/usr/bin/env python3
"""
C-MAPSS FD001 Official Benchmark with PRISM Full-Trajectory Features.

This is the official benchmark:
- Train: 100 engines, each run to failure
- Test: 100 engines, each with partial trajectory
- Ground truth: RUL_FD001.txt (RUL at cutoff for each test engine)

Features: PRISM behavioral metrics computed on full trajectory.
Model: XGBoost regressor (lightweight, interpretable).
"""

import numpy as np
import polars as pl
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

DATA_DIR = Path('/Users/jasonrudder/prism-mac/data')
CMAPSS_DIR = DATA_DIR / 'CMAPSSData'
ML_DIR = Path('/Users/jasonrudder/prism-mac/notebooks/ml_accelerator')

# RUL capping - standard practice
RUL_CAP = 125


def load_train_rul() -> pl.DataFrame:
    """
    Compute RUL for training set (RUL = max_cycle - current_cycle).
    For trajectory features, we use RUL at the final cycle (RUL = 0).
    """
    train_df = pl.read_parquet(CMAPSS_DIR / 'train_FD001.parquet')

    # Get max cycle per unit (= failure point, so RUL = 0 at that cycle)
    max_cycles = train_df.group_by('unit').agg(
        pl.col('cycle').max().alias('max_cycle')
    )

    # For training, we need labels for the FULL trajectory
    # Since we computed features on full trajectory, the label is RUL=0 at failure
    # But we can also compute RUL at any point: RUL = max_cycle - current_cycle

    # For the benchmark, we train on final-cycle features (RUL=0)
    # But for better training, we should use multi-window approach

    # Here we return the total lifetime (max_cycle) for each unit
    # The model will learn to predict RUL from behavioral features
    return max_cycles.with_columns([
        pl.lit(0).alias('rul_at_cutoff'),  # Training data: run to failure
        pl.col('max_cycle').alias('total_lifetime'),
    ])


def load_test_rul() -> pl.DataFrame:
    """Load RUL ground truth for test set."""
    rul_path = CMAPSS_DIR / 'RUL_FD001.txt'

    with open(rul_path) as f:
        rul_values = [int(line.strip()) for line in f]

    return pl.DataFrame({
        'unit': list(range(1, len(rul_values) + 1)),
        'rul_actual': rul_values,
    })


def load_features(split: str) -> pl.DataFrame:
    """Load PRISM features for a split."""
    path = ML_DIR / f'{split}_trajectory_features.parquet'
    return pl.read_parquet(path)


def prepare_train_data_simple():
    """
    Prepare training data using simple approach:
    Train on final-cycle features with RUL=0 labels.

    This is the simplest approach but not optimal.
    """
    train_features = load_features('train')
    train_rul = load_train_rul()

    # Merge
    train = train_features.join(train_rul, on='unit', how='inner')

    # Since features are from full trajectory (RUL=0 at end),
    # we can train to predict RUL=0 for all training samples
    train = train.with_columns([
        pl.lit(0).alias('rul')
    ])

    return train


def prepare_train_data_lifetime():
    """
    Prepare training data using total lifetime as proxy.

    Idea: Engines with shorter total lifetimes show different behavioral patterns
    than engines with longer lifetimes. The model can learn these patterns.

    This is not predicting RUL directly, but learning degradation patterns.
    """
    train_features = load_features('train')
    train_rul = load_train_rul()

    # Merge
    train = train_features.join(train_rul, on='unit', how='inner')

    # Use total lifetime as label (capped)
    train = train.with_columns([
        pl.when(pl.col('max_cycle') > RUL_CAP)
        .then(RUL_CAP)
        .otherwise(pl.col('max_cycle'))
        .alias('rul')
    ])

    return train


def prepare_test_data():
    """Prepare test data with ground truth RUL."""
    test_features = load_features('test')
    test_rul = load_test_rul()

    # Merge
    test = test_features.join(test_rul, on='unit', how='inner')

    # Cap RUL
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

    # Exclude non-feature columns
    exclude = {'unit', 'n_cycles', 'max_cycle', 'total_lifetime', 'rul', 'rul_actual', 'rul_at_cutoff'}

    common = (train_cols & test_cols) - exclude
    return sorted(list(common))


def run_benchmark():
    """Run the official C-MAPSS benchmark."""
    print("=" * 80)
    print("C-MAPSS FD001 OFFICIAL BENCHMARK")
    print("PRISM Full-Trajectory Features + XGBoost")
    print("=" * 80)
    print()

    # Load data
    print("Loading data...")
    train = prepare_train_data_lifetime()  # Use lifetime approach
    test = prepare_test_data()

    print(f"  Train: {len(train)} units")
    print(f"  Test: {len(test)} units")

    # Get common features
    features = get_common_features(train, test)
    print(f"  Features: {len(features)}")

    # Prepare arrays
    X_train = train.select(features).to_numpy()
    y_train = train['rul'].to_numpy()

    X_test = test.select(features).to_numpy()
    y_test = test['rul'].to_numpy()

    print(f"\n  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")

    # Handle NaN values
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # Train XGBoost
    print("\nTraining XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation on training set
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error')
    print(f"  CV RMSE: {-cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

    # Train on full training set
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Clip predictions
    y_pred = np.clip(y_pred, 0, RUL_CAP)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Test Units: {len(test)}")
    print(f"Features: {len(features)}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.4f}")

    # Feature importance
    print("\n=== TOP 15 FEATURES ===")
    importance = model.feature_importances_
    feature_importance = list(zip(features, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for name, imp in feature_importance[:15]:
        print(f"  {name}: {imp:.4f} ({imp*100:.1f}%)")

    # Comparison with published benchmarks
    print("\n" + "=" * 80)
    print("COMPARISON WITH PUBLISHED BENCHMARKS")
    print("=" * 80)
    print(f"| Model | RMSE | Notes |")
    print(f"|-------|------|-------|")
    print(f"| **PRISM + XGBoost** | **{rmse:.2f}** | Full-trajectory features, Mac Mini |")
    print(f"| LSTM | 13-16 | Deep learning, GPU required |")
    print(f"| CNN | 12-14 | Deep learning, GPU required |")
    print(f"| Deep Ensemble | 11-13 | Multiple models, GPU required |")

    # Detailed predictions
    print("\n=== SAMPLE PREDICTIONS ===")
    print("Unit | Actual | Predicted | Error")
    print("-" * 40)
    for i in range(min(10, len(test))):
        unit = test['unit'][i]
        actual = y_test[i]
        pred = y_pred[i]
        error = pred - actual
        print(f"  {unit:3d} | {actual:6.0f} | {pred:9.1f} | {error:+6.1f}")

    # Error distribution
    errors = y_pred - y_test
    print(f"\n=== ERROR DISTRIBUTION ===")
    print(f"Mean error: {errors.mean():+.2f}")
    print(f"Std error: {errors.std():.2f}")
    print(f"Median error: {np.median(errors):+.2f}")
    print(f"Min error: {errors.min():+.2f}")
    print(f"Max error: {errors.max():+.2f}")

    # Count predictions within tolerance
    for tol in [5, 10, 20, 30]:
        within = np.sum(np.abs(errors) <= tol)
        pct = within / len(errors) * 100
        print(f"Within +/-{tol}: {within}/{len(errors)} ({pct:.1f}%)")

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'n_features': len(features),
        'n_train': len(train),
        'n_test': len(test),
        'feature_importance': feature_importance[:20],
    }


if __name__ == '__main__':
    results = run_benchmark()
