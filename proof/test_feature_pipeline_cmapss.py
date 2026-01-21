#!/usr/bin/env python3
"""
Test Feature Engineering Pipeline on C-MAPSS
============================================

Apply heavy rolling window statistics + cluster normalization
to C-MAPSS FD001 and FD002, then train XGBoost.

Usage:
    python proof/test_feature_pipeline_cmapss.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orthon.features import (
    RollingFeatureEngine,
    RollingConfig,
    ClusterNormalizer,
    FeatureEngineeringPipeline,
)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import GradientBoostingRegressor


# =============================================================================
# DATA LOADING
# =============================================================================

COLS = ['unit', 'cycle', 'op_1', 'op_2', 'op_3'] + [f's_{i}' for i in range(1, 22)]

# Informative sensors
SIGNAL_COLS = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12',
               's_13', 's_14', 's_15', 's_17', 's_20', 's_21']


def find_cmapss_data():
    """Find C-MAPSS data directory."""
    candidates = [
        Path('data'),
        Path('data/CMAPSSData'),
        Path('/var/folders/2v/f2fc1dgd24x8rcn0l72b73sw0000gn/T/cmapss_data'),
        Path.home() / 'data' / 'cmapss',
    ]

    for path in candidates:
        if (path / 'train_FD001.txt').exists():
            return path

    return None


def load_cmapss(data_dir: Path, subset: str):
    """Load C-MAPSS dataset."""
    train_path = data_dir / f'train_{subset}.txt'
    test_path = data_dir / f'test_{subset}.txt'
    rul_path = data_dir / f'RUL_{subset}.txt'

    train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=COLS)
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=COLS)

    with open(rul_path, 'r') as f:
        rul_true = np.array([float(line.strip()) for line in f if line.strip()])

    # Add RUL to training data (capped at 125)
    max_cycles = train_df.groupby('unit')['cycle'].max().rename('max_cycle')
    train_df = train_df.merge(max_cycles, on='unit')
    train_df['RUL'] = (train_df['max_cycle'] - train_df['cycle']).clip(upper=125)
    train_df = train_df.drop(columns=['max_cycle'])

    return train_df, test_df, rul_true


# =============================================================================
# MAIN
# =============================================================================

def run_experiment(data_dir: Path, subset: str, n_clusters: int):
    """Run full experiment on a C-MAPSS subset."""

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {subset} ({n_clusters} clusters)")
    print('='*70)

    # Load data
    print("\n[1/5] Loading data...")
    train_df, test_df, rul_true = load_cmapss(data_dir, subset)
    print(f"  Train: {len(train_df):,} rows, {train_df['unit'].nunique()} engines")
    print(f"  Test: {len(test_df):,} rows, {test_df['unit'].nunique()} engines")

    # Create feature engineering pipeline
    print("\n[2/5] Setting up feature engineering pipeline...")

    # Configure rolling features
    rolling_config = RollingConfig(
        windows=[10, 20, 30],
        compute_mean=True,
        compute_std=True,
        compute_slope=True,
        compute_delta=True,
        compute_curvature=True,
        compute_min=True,
        compute_max=True,
        compute_range=True,
        compute_zscore=True,
        compute_volatility=True,
        # Disable slower ones
        compute_skew=False,
        compute_kurtosis=False,
        compute_quantiles=False,
        compute_iqr=False,
        compute_cv=False,
        compute_momentum=False,
        compute_autocorr=False,
        compute_entropy=False,
    )

    pipeline = FeatureEngineeringPipeline(
        n_clusters=n_clusters,
        windows=[10, 20, 30],
        op_cols=['op_1', 'op_2'],
        signal_cols=SIGNAL_COLS,
        healthy_pct=0.20,
        rolling_config=rolling_config,
    )

    # Fit and transform training data
    print("\n[3/5] Transforming training data...")
    train_feat = pipeline.fit_transform(
        data=train_df,
        entity_col='unit',
        time_col='cycle',
    )

    # Transform test data
    print("\n[4/5] Transforming test data...")
    test_feat = pipeline.transform(test_df)

    # Identify feature columns
    base_cols = set(COLS + ['RUL'])
    feature_cols = [c for c in train_feat.columns if c not in base_cols and c != 'unit']

    # Remove any columns with all NaN
    valid_cols = []
    for col in feature_cols:
        if train_feat[col].notna().sum() > len(train_feat) * 0.5:
            valid_cols.append(col)
    feature_cols = valid_cols

    print(f"\n  Generated {len(feature_cols)} features")
    print(f"  Sample features: {feature_cols[:10]}")

    # Prepare training data
    print("\n[5/5] Training XGBoost model...")

    X_train = train_feat[feature_cols].fillna(0).values
    y_train = train_feat['RUL'].values

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    # Train model
    if HAS_XGBOOST:
        model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=2,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
        )

    model.fit(X_train_s, y_train)

    # Prepare test data (last cycle per unit)
    test_last = test_feat.groupby('unit').last().reset_index()

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in test_last.columns:
            test_last[col] = 0

    X_test = test_last[feature_cols].fillna(0).values
    X_test_s = scaler.transform(X_test)

    # Predict
    y_pred = model.predict(X_test_s)
    y_pred = np.clip(y_pred, 0, 125)
    rul_capped = np.clip(rul_true, 0, 125)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(rul_capped, y_pred))
    mae = mean_absolute_error(rul_capped, y_pred)

    # SOTA benchmarks
    sota = {'FD001': 10.82, 'FD002': 11.46}[subset]
    gap = (rmse - sota) / sota * 100

    print(f"\n{'='*70}")
    print(f"RESULTS: {subset}")
    print('='*70)
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  SOTA: {sota}")
    print(f"  Gap:  {gap:+.1f}%")

    # Feature importance
    if HAS_XGBOOST and hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        top_idx = np.argsort(importance)[-15:][::-1]

        print(f"\n  Top 15 Features:")
        for i, idx in enumerate(top_idx):
            print(f"    {i+1:2d}. {feature_cols[idx]:<35} {importance[idx]:.4f}")

    return rmse, mae, gap


def main():
    print("="*70)
    print("FEATURE ENGINEERING PIPELINE TEST ON C-MAPSS")
    print("="*70)

    # Find data
    data_dir = find_cmapss_data()
    if data_dir is None:
        print("\n[ERROR] Could not find C-MAPSS data!")
        print("Please download from NASA and place in data/ directory")
        return 1

    print(f"\nData directory: {data_dir}")
    print(f"XGBoost available: {HAS_XGBOOST}")

    results = {}

    # Run FD001 (1 operating condition)
    rmse, mae, gap = run_experiment(data_dir, 'FD001', n_clusters=1)
    results['FD001'] = {'rmse': rmse, 'mae': mae, 'gap': gap}

    # Run FD002 (6 operating conditions)
    rmse, mae, gap = run_experiment(data_dir, 'FD002', n_clusters=6)
    results['FD002'] = {'rmse': rmse, 'mae': mae, 'gap': gap}

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Dataset':<10} {'RMSE':>10} {'MAE':>10} {'SOTA':>10} {'Gap':>10}")
    print("-"*50)
    print(f"{'FD001':<10} {results['FD001']['rmse']:>10.2f} {results['FD001']['mae']:>10.2f} {'10.82':>10} {results['FD001']['gap']:>+9.1f}%")
    print(f"{'FD002':<10} {results['FD002']['rmse']:>10.2f} {results['FD002']['mae']:>10.2f} {'11.46':>10} {results['FD002']['gap']:>+9.1f}%")
    print("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
