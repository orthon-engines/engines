#!/usr/bin/env python3
"""
Stacking Ensemble Test
======================

Test stacking ensemble: LightGBM + CatBoost + XGBoost with Ridge meta-learner.

Goal: Beat current best results
- FD001: 13.26 RMSE
- FD002: 14.15 RMSE
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from orthon.features import (
    RollingFeatureEngine,
    RollingConfig,
    ClusterNormalizer,
)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor  # CatBoost unavailable on Python 3.14

# =============================================================================
# CONFIG
# =============================================================================

COLS = ['unit', 'cycle', 'op_1', 'op_2', 'op_3'] + [f's_{i}' for i in range(1, 22)]
SIGNAL_COLS = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12',
               's_13', 's_14', 's_15', 's_17', 's_20', 's_21']

# Best window configuration from previous experiments
WINDOWS = [5, 10, 20, 30, 50]

DATA_DIR = Path('/var/folders/2v/f2fc1dgd24x8rcn0l72b73sw0000gn/T/cmapss_data')


def load_cmapss(subset: str):
    train_df = pd.read_csv(DATA_DIR / f'train_{subset}.txt', sep=r'\s+', header=None, names=COLS)
    test_df = pd.read_csv(DATA_DIR / f'test_{subset}.txt', sep=r'\s+', header=None, names=COLS)

    with open(DATA_DIR / f'RUL_{subset}.txt', 'r') as f:
        rul_true = np.array([float(line.strip()) for line in f if line.strip()])

    max_cycles = train_df.groupby('unit')['cycle'].max().rename('max_cycle')
    train_df = train_df.merge(max_cycles, on='unit')
    train_df['RUL'] = (train_df['max_cycle'] - train_df['cycle']).clip(upper=125)
    train_df = train_df.drop(columns=['max_cycle'])

    return train_df, test_df, rul_true


def compute_features(df, normalizer, rolling_engine, entity_col='unit', time_col='cycle'):
    """Compute all features for a dataframe."""

    # Step 1: Cluster normalize
    df_norm = normalizer.compute_healthy_distance(df)

    # Step 2: Rolling features on hd_mean
    all_dfs = []

    for unit in df_norm[entity_col].unique():
        unit_df = df_norm[df_norm[entity_col] == unit].sort_values(time_col).copy()

        # Rolling on hd_mean
        hd_feats = rolling_engine.compute(unit_df['hd_mean'].values, prefix='hd')
        for name, vals in hd_feats.items():
            unit_df[name] = vals

        # Rolling on top 3 normalized signals
        for sig in ['s_11_norm', 's_12_norm', 's_15_norm']:
            if sig in unit_df.columns:
                sig_feats = rolling_engine.compute(unit_df[sig].values, prefix=sig.replace('_norm', ''))
                for name, vals in sig_feats.items():
                    unit_df[name] = vals

        all_dfs.append(unit_df)

    return pd.concat(all_dfs, ignore_index=True)


def run_experiment(subset: str, n_clusters: int):
    """Run stacking ensemble experiment."""

    print(f"\n{'='*70}")
    print(f"{subset} | STACKING ENSEMBLE (LGB + GBR + XGB + Ridge)")
    print('='*70)

    # Load
    train_df, test_df, rul_true = load_cmapss(subset)
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    # Fit normalizer
    normalizer = ClusterNormalizer(n_clusters=n_clusters, healthy_pct=0.20)
    normalizer.fit(train_df, op_cols=['op_1', 'op_2'], signal_cols=SIGNAL_COLS,
                   entity_col='unit', time_col='cycle')

    # Rolling engine with best windows
    config = RollingConfig(
        windows=WINDOWS,
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
        compute_skew=False,
        compute_kurtosis=False,
        compute_quantiles=False,
        compute_iqr=False,
        compute_cv=False,
        compute_momentum=False,
        compute_autocorr=False,
    )
    rolling_engine = RollingFeatureEngine(config=config)

    # Transform
    print("Computing features...")
    train_feat = compute_features(train_df, normalizer, rolling_engine)
    test_feat = compute_features(test_df, normalizer, rolling_engine)

    # Get feature columns
    base_cols = set(COLS + ['RUL', 'unit'])
    feature_cols = [c for c in train_feat.columns if c not in base_cols]
    feature_cols = [c for c in feature_cols if train_feat[c].notna().sum() > len(train_feat) * 0.3]

    print(f"Total features: {len(feature_cols)}")

    # Prepare data
    X_train = train_feat[feature_cols].fillna(0).values
    y_train = train_feat['RUL'].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    # Build stacking ensemble
    print("\nBuilding stacking ensemble...")

    estimators = [
        ('lgb', LGBMRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.7,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )),
        ('gbr', GradientBoostingRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
        ('xgb', XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.7,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )),
    ]

    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1,
        passthrough=False,  # Only use base model predictions
    )

    print("Training stacking ensemble (this may take a few minutes)...")
    stack.fit(X_train_s, y_train)

    # Prepare test (last cycle per unit)
    test_last = test_feat.groupby('unit').last().reset_index()
    for col in feature_cols:
        if col not in test_last.columns:
            test_last[col] = 0

    X_test = test_last[feature_cols].fillna(0).values
    X_test_s = scaler.transform(X_test)

    # Predict with stacking
    y_pred_stack = np.clip(stack.predict(X_test_s), 0, 125)
    rul_capped = np.clip(rul_true, 0, 125)

    rmse_stack = np.sqrt(mean_squared_error(rul_capped, y_pred_stack))
    sota = {'FD001': 10.82, 'FD002': 11.46}[subset]
    prev_best = {'FD001': 13.26, 'FD002': 14.15}[subset]
    gap = (rmse_stack - sota) / sota * 100
    improvement = (prev_best - rmse_stack) / prev_best * 100

    print(f"\n*** STACKING RESULTS ***")
    print(f"  RMSE: {rmse_stack:.2f}")
    print(f"  SOTA: {sota}")
    print(f"  Gap from SOTA: {gap:+.1f}%")
    print(f"  Previous best: {prev_best}")
    print(f"  Improvement: {improvement:+.1f}%")

    # Also test individual models for comparison
    print("\n[Individual Model Comparison]")

    for name, model in estimators:
        model.fit(X_train_s, y_train)
        y_pred = np.clip(model.predict(X_test_s), 0, 125)
        rmse = np.sqrt(mean_squared_error(rul_capped, y_pred))
        print(f"  {name.upper()}: {rmse:.2f}")

    return rmse_stack, gap


def main():
    print("="*70)
    print("STACKING ENSEMBLE: LightGBM + GradientBoosting + XGBoost + Ridge")
    print("="*70)
    print(f"\nWindows: {WINDOWS}")
    print("Previous best: FD001=13.26, FD002=14.15")

    results = []

    for subset, n_clusters in [('FD001', 1), ('FD002', 6)]:
        rmse, gap = run_experiment(subset, n_clusters)
        results.append({
            'subset': subset,
            'rmse': rmse,
            'gap': gap,
        })

    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"\n{'Dataset':<10} {'Stack RMSE':>12} {'Prev Best':>12} {'SOTA':>10} {'Gap':>10}")
    print("-"*54)
    for r in results:
        prev = {'FD001': 13.26, 'FD002': 14.15}[r['subset']]
        sota = {'FD001': 10.82, 'FD002': 11.46}[r['subset']]
        print(f"{r['subset']:<10} {r['rmse']:>12.2f} {prev:>12.2f} {sota:>10.2f} {r['gap']:>+9.1f}%")

    return 0


if __name__ == '__main__':
    sys.exit(main())
