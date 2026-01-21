#!/usr/bin/env python3
"""
Feature Engineering Demo
========================

Demonstrates heavy rolling window statistics and cluster normalization.

These are the two techniques that drove our best C-MAPSS results:
- FD001: 13.36 RMSE
- FD002: 15.04 RMSE

Usage:
    python examples/feature_engineering_demo.py
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("FEATURE ENGINEERING DEMO")
    print("Rolling Window Statistics + Cluster Normalization")
    print("=" * 70)

    # Import feature engineering modules
    try:
        from orthon.features import (
            RollingFeatureEngine,
            RollingConfig,
            ClusterNormalizer,
            FeatureEngineeringPipeline,
            compute_all_rolling_features,
        )
        print("\n[OK] Imported orthon.features")
    except ImportError as e:
        print(f"\n[ERROR] Import failed: {e}")
        print("Run: pip install -e .")
        return 1

    # =========================================================================
    # 1. GENERATE SYNTHETIC DATA
    # =========================================================================
    print("\n" + "-" * 70)
    print("1. GENERATING SYNTHETIC MULTI-CONDITION DATA")
    print("-" * 70)

    np.random.seed(42)

    # Simulate 10 engines with 200 cycles each
    n_engines = 10
    n_cycles = 200
    n_signals = 5

    data_rows = []

    for engine in range(1, n_engines + 1):
        # Random operating condition for this engine
        op_condition = np.random.randint(0, 3)  # 3 regimes

        # Operating condition values
        op_1 = [0.0, 25.0, 42.0][op_condition] + np.random.randn() * 2
        op_2 = [0.84, 0.70, 0.62][op_condition] + np.random.randn() * 0.02

        for cycle in range(1, n_cycles + 1):
            # Degradation increases with cycle
            degradation = 0.05 * cycle + 0.0005 * cycle ** 1.5

            row = {
                'unit': engine,
                'cycle': cycle,
                'op_1': op_1 + np.random.randn() * 0.5,
                'op_2': op_2 + np.random.randn() * 0.005,
            }

            # Generate signals based on operating condition
            base_values = {
                0: [550, 1200, 47, 520, 8.5],    # Regime 0 baseline
                1: [610, 1350, 45, 480, 8.0],    # Regime 1 baseline
                2: [580, 1280, 46, 500, 8.2],    # Regime 2 baseline
            }

            for i, base in enumerate(base_values[op_condition]):
                noise = np.random.randn() * (base * 0.01)
                row[f's_{i+1}'] = base + degradation * (i + 1) + noise

            data_rows.append(row)

    df = pd.DataFrame(data_rows)
    print(f"\nGenerated data: {len(df):,} rows, {n_engines} engines, {n_cycles} cycles each")
    print(f"Columns: {list(df.columns)}")
    print(f"\nOperating conditions (op_1, op_2):")
    print(df.groupby('unit')[['op_1', 'op_2']].mean().round(2))

    # =========================================================================
    # 2. ROLLING WINDOW STATISTICS
    # =========================================================================
    print("\n" + "-" * 70)
    print("2. ROLLING WINDOW STATISTICS")
    print("-" * 70)

    # Configure what statistics to compute
    config = RollingConfig(
        windows=[10, 20, 30],
        compute_mean=True,
        compute_std=True,
        compute_slope=True,
        compute_delta=True,
        compute_curvature=True,
        compute_skew=True,
        compute_zscore=True,
        compute_volatility=True,
        compute_autocorr=True,
        # These are slower, disable for demo
        compute_kurtosis=False,
        compute_quantiles=False,
        compute_entropy=False,
    )

    engine = RollingFeatureEngine(config=config)

    # Compute for a single signal
    print("\n[Demo] Computing rolling features for s_1 on engine 1...")

    engine_1 = df[df['unit'] == 1].sort_values('cycle')
    s1_values = engine_1['s_1'].values

    features = engine.compute(s1_values, prefix='s1')
    print(f"\nGenerated {len(features)} features:")
    for name in sorted(features.keys())[:15]:
        vals = features[name]
        valid = ~np.isnan(vals)
        if valid.sum() > 0:
            print(f"  {name}: mean={np.nanmean(vals):.3f}, std={np.nanstd(vals):.3f}")

    print("  ...")

    # Compute for multiple signals across all entities
    print("\n[Full] Computing rolling features for all signals and engines...")

    df_rolling = compute_all_rolling_features(
        data=df,
        signal_cols=['s_1', 's_2', 's_3'],
        windows=[10, 20],
        entity_col='unit',
        sort_col='cycle',
    )

    new_cols = [c for c in df_rolling.columns if c not in df.columns]
    print(f"\nAdded {len(new_cols)} rolling feature columns")
    print(f"Sample feature names: {new_cols[:10]}")

    # =========================================================================
    # 3. CLUSTER NORMALIZATION
    # =========================================================================
    print("\n" + "-" * 70)
    print("3. CLUSTER NORMALIZATION")
    print("-" * 70)

    # Split into train/test
    train_units = list(range(1, 8))  # 7 engines for training
    test_units = list(range(8, 11))  # 3 engines for testing

    train_df = df[df['unit'].isin(train_units)].copy()
    test_df = df[df['unit'].isin(test_units)].copy()

    print(f"\nTrain: {len(train_df):,} rows ({len(train_units)} engines)")
    print(f"Test: {len(test_df):,} rows ({len(test_units)} engines)")

    # Fit cluster normalizer
    normalizer = ClusterNormalizer(
        n_clusters=3,       # 3 operating regimes
        healthy_pct=0.20,   # First 20% of life = healthy
        use_median=False,   # Use mean for baseline
        robust_std=True,    # Use robust std estimate
    )

    normalizer.fit(
        data=train_df,
        op_cols=['op_1', 'op_2'],
        signal_cols=['s_1', 's_2', 's_3', 's_4', 's_5'],
        entity_col='unit',
        time_col='cycle',
    )

    # Show learned baselines
    print("\n[Learned Baselines]")
    for cluster_id, baseline in normalizer.baselines.items():
        print(f"\n  Cluster {cluster_id} ({baseline.n_samples} healthy samples):")
        for signal, stats in list(baseline.signal_stats.items())[:3]:
            print(f"    {signal}: center={stats['center']:.2f}, spread={stats['spread']:.2f}")

    # Transform test data
    print("\n[Transforming test data...]")
    test_norm = normalizer.compute_healthy_distance(test_df)

    print(f"\nNormalized test data columns:")
    norm_cols = [c for c in test_norm.columns if c.endswith('_norm') or c.startswith('hd_')]
    print(f"  {norm_cols}")

    print(f"\nHealthy distance statistics (test set):")
    print(f"  hd_mean: {test_norm['hd_mean'].mean():.3f} (avg deviation from healthy)")
    print(f"  hd_max: {test_norm['hd_max'].mean():.3f} (worst signal deviation)")

    # =========================================================================
    # 4. FULL PIPELINE
    # =========================================================================
    print("\n" + "-" * 70)
    print("4. FULL FEATURE ENGINEERING PIPELINE")
    print("-" * 70)

    pipeline = FeatureEngineeringPipeline(
        n_clusters=3,
        windows=[10, 20],
        op_cols=['op_1', 'op_2'],
        signal_cols=['s_1', 's_2', 's_3'],
        healthy_pct=0.20,
    )

    # Fit and transform training data
    train_features = pipeline.fit_transform(
        data=train_df,
        entity_col='unit',
        time_col='cycle',
    )

    # Transform test data
    test_features = pipeline.transform(test_df)

    print(f"\nFinal feature counts:")
    print(f"  Original columns: {len(df.columns)}")
    print(f"  Final columns: {len(train_features.columns)}")
    print(f"  New features: {len(train_features.columns) - len(df.columns)}")

    # Show sample of generated features
    feature_cols = [c for c in train_features.columns if c not in df.columns]
    print(f"\nSample generated features:")
    for col in sorted(feature_cols)[:20]:
        print(f"  - {col}")
    if len(feature_cols) > 20:
        print(f"  ... and {len(feature_cols) - 20} more")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("""
Key Techniques:

1. ROLLING WINDOW STATISTICS
   - Compute features over sliding windows (10, 20, 30 cycles)
   - Captures temporal dynamics:
     * slope: degradation rate
     * delta: change from N cycles ago
     * curvature: acceleration
     * volatility: stability of signal
     * zscore: current position in window

2. CLUSTER NORMALIZATION
   - Cluster operating conditions into regimes
   - Compute healthy baselines per regime
   - Normalize by regime baseline (z-score)
   - Enables comparison across operating conditions

3. COMBINED PIPELINE
   - First normalize by cluster
   - Then compute rolling stats on normalized values
   - This is what achieved our best C-MAPSS results

Usage in Your Project:

    from orthon.features import FeatureEngineeringPipeline

    pipeline = FeatureEngineeringPipeline(
        n_clusters=6,
        windows=[10, 20, 30],
        op_cols=['op_1', 'op_2'],
        signal_cols=['s_11', 's_12', 's_15'],
    )

    train_feat = pipeline.fit_transform(train_df, entity_col='unit', time_col='cycle')
    test_feat = pipeline.transform(test_df)

    # Use features for ML
    X_train = train_feat[feature_cols].values
    y_train = train_feat['RUL'].values
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
