#!/usr/bin/env python3
"""
Orthon Demo Script
==================

This script demonstrates the core capabilities of the orthon package.

Run:
    python examples/demo_orthon.py

Requirements:
    pip install -e .
"""

import numpy as np
import sys

def main():
    print("=" * 70)
    print("ORTHON DEMO - Behavioral Geometry Engine")
    print("=" * 70)

    # Import orthon
    try:
        import orthon
        print(f"\n[OK] Orthon version: {orthon.__version__}")
    except ImportError as e:
        print(f"\n[ERROR] Failed to import orthon: {e}")
        print("Run: pip install -e .")
        return 1

    # =========================================================================
    # 1. LIST AVAILABLE ENGINES
    # =========================================================================
    print("\n" + "-" * 70)
    print("1. AVAILABLE ENGINES")
    print("-" * 70)

    print(f"\nVector Engines ({len(orthon.list_vector_engines())}):")
    for name in sorted(orthon.list_vector_engines()):
        print(f"  - {name}")

    print(f"\nGeometry Engines ({len(orthon.list_geometry_engines())}):")
    for name in sorted(orthon.list_geometry_engines()):
        print(f"  - {name}")

    print(f"\nState Engines ({len(orthon.list_state_engines())}):")
    for name in sorted(orthon.list_state_engines()):
        print(f"  - {name}")

    # =========================================================================
    # 2. VECTOR ENGINE DEMO
    # =========================================================================
    print("\n" + "-" * 70)
    print("2. VECTOR ENGINE DEMO (Single Signal Analysis)")
    print("-" * 70)

    # Generate synthetic degradation signal
    np.random.seed(42)
    t = np.arange(0, 200)
    degradation = 0.05 * t + 0.002 * t ** 1.5
    noise = np.random.randn(200) * 0.5
    signal = 550 + degradation + noise

    print(f"\nGenerated signal: 200 samples, simulating sensor degradation")
    print(f"  Mean: {signal.mean():.2f}")
    print(f"  Std:  {signal.std():.2f}")
    print(f"  Range: [{signal.min():.2f}, {signal.max():.2f}]")

    # Compute Hurst exponent
    print("\n  Computing Hurst exponent (long-range dependence)...")
    try:
        hurst = orthon.compute_hurst(signal)
        h_exp = hurst.get('hurst_exp', hurst.get('H', 'N/A'))
        print(f"  [OK] Hurst exponent: {h_exp}")
        if isinstance(h_exp, float):
            if h_exp > 0.5:
                print("       Interpretation: Persistent (trending) behavior")
            elif h_exp < 0.5:
                print("       Interpretation: Anti-persistent (mean-reverting)")
            else:
                print("       Interpretation: Random walk")
    except Exception as e:
        print(f"  [SKIP] Hurst computation failed: {e}")

    # Compute entropy
    print("\n  Computing entropy (information content)...")
    try:
        entropy = orthon.compute_entropy(signal)
        se = entropy.get('sample_entropy', entropy.get('SampEn', 'N/A'))
        print(f"  [OK] Sample entropy: {se}")
    except Exception as e:
        print(f"  [SKIP] Entropy computation failed: {e}")

    # Compute realized volatility
    print("\n  Computing realized volatility...")
    try:
        vol = orthon.compute_realized_vol(signal)
        rv = vol.get('realized_vol', vol.get('rv', 'N/A'))
        print(f"  [OK] Realized volatility: {rv}")
    except Exception as e:
        print(f"  [SKIP] Realized vol computation failed: {e}")

    # =========================================================================
    # 3. GEOMETRY ENGINE DEMO
    # =========================================================================
    print("\n" + "-" * 70)
    print("3. GEOMETRY ENGINE DEMO (Multi-Signal Structure)")
    print("-" * 70)

    # Generate correlated multi-signal data
    np.random.seed(42)
    n_samples = 100
    n_signals = 5

    # Base pattern (shared degradation)
    base = np.linspace(0, 10, n_samples)

    # Create correlated signals
    data = np.column_stack([
        base + np.random.randn(n_samples) * 0.5,          # Correlated +
        base * 1.2 + np.random.randn(n_samples) * 0.3,    # Correlated +
        -base * 0.8 + np.random.randn(n_samples) * 0.4,   # Correlated -
        np.sin(base) + np.random.randn(n_samples) * 0.2,  # Periodic
        np.random.randn(n_samples) * 2,                   # Independent noise
    ])

    print(f"\nGenerated data: {n_samples} samples x {n_signals} signals")
    print(f"  Signal 1-2: Positively correlated with degradation")
    print(f"  Signal 3: Negatively correlated")
    print(f"  Signal 4: Periodic component")
    print(f"  Signal 5: Random noise")

    # PCA
    print("\n  Computing PCA (dimensionality reduction)...")
    try:
        pca = orthon.PCAEngine(n_components=3)
        pca_result = pca.compute(data)
        evr = pca_result.get('explained_variance_ratio', [])
        if len(evr) > 0:
            print(f"  [OK] Explained variance ratios: {[f'{v:.2%}' for v in evr[:3]]}")
            print(f"       PC1 captures {evr[0]:.1%} of total variance")
    except Exception as e:
        print(f"  [SKIP] PCA failed: {e}")

    # Clustering
    print("\n  Computing clustering (signal grouping)...")
    try:
        cluster = orthon.ClusteringEngine(n_clusters=2)
        cluster_result = cluster.compute(data.T)  # Signals as rows
        labels = cluster_result.get('labels', [])
        print(f"  [OK] Cluster labels: {labels}")
    except Exception as e:
        print(f"  [SKIP] Clustering failed: {e}")

    # =========================================================================
    # 4. STATE ENGINE DEMO
    # =========================================================================
    print("\n" + "-" * 70)
    print("4. STATE ENGINE DEMO (Temporal Dynamics)")
    print("-" * 70)

    # Generate time series with causal relationship
    np.random.seed(42)
    n = 100

    # X causes Y with lag
    x = np.cumsum(np.random.randn(n))
    y = np.zeros(n)
    for i in range(3, n):
        y[i] = 0.8 * x[i - 3] + 0.2 * np.random.randn()

    ts_data = np.column_stack([x, y])

    print(f"\nGenerated time series: {n} timesteps x 2 signals")
    print(f"  Signal X: Random walk")
    print(f"  Signal Y: Caused by X with lag=3")

    # DTW
    print("\n  Computing DTW (time series similarity)...")
    try:
        dtw = orthon.DTWEngine()
        dtw_result = dtw.compute(x, y)
        dist = dtw_result.get('distance', 'N/A')
        print(f"  [OK] DTW distance: {dist}")
    except Exception as e:
        print(f"  [SKIP] DTW failed: {e}")

    # =========================================================================
    # 5. I/O DEMO
    # =========================================================================
    print("\n" + "-" * 70)
    print("5. I/O LAYER")
    print("-" * 70)

    print("\n  File constants:")
    print(f"    OBSERVATIONS = '{orthon.OBSERVATIONS}'")
    print(f"    VECTOR = '{orthon.VECTOR}'")
    print(f"    GEOMETRY = '{orthon.GEOMETRY}'")
    print(f"    STATE = '{orthon.STATE}'")

    print("\n  Available I/O functions:")
    print("    orthon.read_parquet(path)")
    print("    orthon.write_parquet_atomic(df, path)")
    print("    orthon.upsert_parquet(df, path, key_cols)")
    print("    orthon.append_parquet(df, path)")
    print("    orthon.get_path(orthon.OBSERVATIONS)")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)

    print("""
Next steps:

1. Read the full documentation:
   notebooks/orthon_complete_documentation.ipynb

2. Check the cheatsheet:
   notebooks/ORTHON_CHEATSHEET.md

3. Run the C-MAPSS RUL prediction:
   python proof/best_rul_model.py

4. Explore the CLI:
   orthon --help
   orthon --list-engines
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
