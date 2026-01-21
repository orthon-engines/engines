#!/usr/bin/env python3
"""
Extended Windows + Feature Selection Test
==========================================

Test wider window range [5, 10, 20, 30, 50] and XGBoost feature selection.
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

from xgboost import XGBRegressor

# =============================================================================
# CONFIG
# =============================================================================

COLS = ['unit', 'cycle', 'op_1', 'op_2', 'op_3'] + [f's_{i}' for i in range(1, 22)]
SIGNAL_COLS = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12',
               's_13', 's_14', 's_15', 's_17', 's_20', 's_21']

# Extended windows: short-term (5) to long-term (50)
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


def run_experiment(subset: str, n_clusters: int, top_k: int = None):
    """Run experiment with optional feature selection."""

    print(f"\n{'='*70}")
    print(f"{subset} | Windows: {WINDOWS} | Top-K: {top_k or 'all'}")
    print('='*70)

    # Load
    train_df, test_df, rul_true = load_cmapss(subset)
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    # Fit normalizer
    normalizer = ClusterNormalizer(n_clusters=n_clusters, healthy_pct=0.20)
    normalizer.fit(train_df, op_cols=['op_1', 'op_2'], signal_cols=SIGNAL_COLS,
                   entity_col='unit', time_col='cycle')

    # Rolling engine with extended windows
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

    # Train initial model
    model = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.7,
        random_state=42, n_jobs=-1, verbosity=0
    )
    model.fit(X_train_s, y_train)

    # Feature selection
    if top_k and top_k < len(feature_cols):
        importance = model.feature_importances_
        top_idx = np.argsort(importance)[-top_k:]
        selected_cols = [feature_cols[i] for i in top_idx]

        print(f"\nTop {top_k} features selected:")
        for i in np.argsort(importance)[-10:][::-1]:
            print(f"  {feature_cols[i]:<40} {importance[i]:.4f}")

        # Retrain with selected features
        X_train_sel = train_feat[selected_cols].fillna(0).values
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_sel)

        model = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.7,
            random_state=42, n_jobs=-1, verbosity=0
        )
        model.fit(X_train_s, y_train)
        feature_cols = selected_cols

    # Prepare test (last cycle per unit)
    test_last = test_feat.groupby('unit').last().reset_index()
    for col in feature_cols:
        if col not in test_last.columns:
            test_last[col] = 0

    X_test = test_last[feature_cols].fillna(0).values
    X_test_s = scaler.transform(X_test)

    # Predict
    y_pred = np.clip(model.predict(X_test_s), 0, 125)
    rul_capped = np.clip(rul_true, 0, 125)

    rmse = np.sqrt(mean_squared_error(rul_capped, y_pred))
    sota = {'FD001': 10.82, 'FD002': 11.46}[subset]
    gap = (rmse - sota) / sota * 100

    print(f"\n*** RMSE: {rmse:.2f} | SOTA: {sota} | Gap: {gap:+.1f}% ***")

    return rmse, gap, feature_cols


def main():
    print("="*70)
    print("EXTENDED WINDOWS [5, 10, 20, 30, 50] + FEATURE SELECTION")
    print("="*70)

    results = []

    # Test different configurations
    configs = [
        ('FD001', 1, None),      # All features
        ('FD001', 1, 50),        # Top 50
        ('FD001', 1, 30),        # Top 30
        ('FD002', 6, None),      # All features
        ('FD002', 6, 50),        # Top 50
        ('FD002', 6, 30),        # Top 30
    ]

    for subset, n_clusters, top_k in configs:
        rmse, gap, _ = run_experiment(subset, n_clusters, top_k)
        results.append({
            'subset': subset,
            'top_k': top_k or 'all',
            'rmse': rmse,
            'gap': gap,
        })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Dataset':<10} {'Top-K':<10} {'RMSE':>10} {'Gap':>10}")
    print("-"*40)
    for r in results:
        print(f"{r['subset']:<10} {str(r['top_k']):<10} {r['rmse']:>10.2f} {r['gap']:>+9.1f}%")

    return 0


if __name__ == '__main__':
    sys.exit(main())
