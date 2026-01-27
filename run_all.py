#!/usr/bin/env python3
"""
PRISM Pipeline: run_all.py

RUNS EVERYTHING. NO EXCEPTIONS. NO ASKING.

Usage:
    python run_all.py

Input:
    data/observations.parquet

Outputs:
    data/typology.parquet          - Signal classification
    data/primitives.parquet        - ALL per-signal metrics
    data/primitives_pairs.parquet  - ALL pairwise metrics
    data/primitives_points.parquet - ALL per-point metrics
    data/manifold.parquet          - Phase space trajectory
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Add prism to path
sys.path.insert(0, str(Path(__file__).parent))

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "observations.parquet"

# =============================================================================
# EXPLICIT ENGINE IMPORTS
# =============================================================================

from prism.engines.core import (
    # Signal-level engines (one row per signal)
    hurst,
    lyapunov,
    entropy,
    fft,
    garch,
    acf_decay,
    attractor,
    basin,
    rqa,
    lof,
    convex_hull,
    dmd,

    # Pairwise engines (one row per signal pair)
    granger,
    transfer_entropy,
    cointegration,
    dtw,
    mutual_info,
    copula,
    mst,
    divergence,

    # Point-level engines (one row per observation)
    hilbert,
    clustering,

    # System-level
    pca,
)

# =============================================================================
# EXPLICIT ENGINE LISTS
# =============================================================================

SIGNAL_ENGINES = [
    ('hurst', hurst),
    ('lyapunov', lyapunov),
    ('entropy', entropy),
    ('fft', fft),
    ('garch', garch),
    ('acf_decay', acf_decay),
    ('attractor', attractor),
    ('basin', basin),
    ('rqa', rqa),
    ('lof', lof),
    ('convex_hull', convex_hull),
    # DMD outputs per-entity modes, not per-signal - special handling
]

PAIRWISE_ENGINES = [
    ('granger', granger),
    ('transfer_entropy', transfer_entropy),
    ('cointegration', cointegration),
    ('dtw', dtw),
    ('mutual_info', mutual_info),
    ('copula', copula),
    ('mst', mst),
    ('divergence', divergence),
]

POINT_ENGINES = [
    ('hilbert', hilbert),
    ('clustering', clustering),
]


# =============================================================================
# TYPOLOGY COMPUTATION
# =============================================================================

def compute_typology(obs: pd.DataFrame) -> pd.DataFrame:
    """Classify each signal by type."""
    results = []

    for (entity_id, signal_id), group in obs.groupby(['entity_id', 'signal_id']):
        y = group['y'].dropna().values
        n = len(y)

        if n == 0:
            signal_class = 'empty'
            unique_ratio = 0.0
            variance = 0.0
            mean_val = 0.0
            std_val = 0.0
            min_val = 0.0
            max_val = 0.0
        else:
            unique_ratio = len(np.unique(y)) / n
            variance = float(np.var(y))
            mean_val = float(np.mean(y))
            std_val = float(np.std(y))
            min_val = float(np.min(y))
            max_val = float(np.max(y))

            if variance < 1e-10:
                signal_class = 'constant'
            elif unique_ratio < 0.01:
                signal_class = 'digital'
            elif unique_ratio < 0.05:
                signal_class = 'discrete'
            else:
                signal_class = 'analog'

        results.append({
            'entity_id': entity_id,
            'signal_id': signal_id,
            'n_points': n,
            'unique_ratio': round(unique_ratio, 6),
            'variance': variance,
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'signal_class': signal_class,
        })

    return pd.DataFrame(results)


# =============================================================================
# MANIFOLD TRAJECTORY
# =============================================================================

def compute_manifold_trajectory(obs: pd.DataFrame) -> pd.DataFrame:
    """Compute phase space trajectory for visualization."""
    from sklearn.decomposition import PCA

    results = []

    for entity_id, entity_group in obs.groupby('entity_id'):
        # Pivot: rows = time points, columns = signals
        try:
            wide = entity_group.pivot(index='I', columns='signal_id', values='y')
            wide = wide.sort_index().dropna()
        except Exception:
            wide = entity_group.groupby(['I', 'signal_id'])['y'].mean().unstack()
            wide = wide.sort_index().dropna()

        if len(wide) < 10 or len(wide.columns) < 2:
            continue

        # PCA for trajectory
        n_comp = min(3, len(wide.columns), len(wide))
        pca_model = PCA(n_components=n_comp)
        coords = pca_model.fit_transform(wide.values)

        # Pad to 3D
        if coords.shape[1] < 3:
            pad = np.zeros((coords.shape[0], 3 - coords.shape[1]))
            coords = np.hstack([coords, pad])

        explained = float(sum(pca_model.explained_variance_ratio_))

        for i, I in enumerate(wide.index):
            results.append({
                'entity_id': entity_id,
                'I': I,
                'manifold_x': float(coords[i, 0]),
                'manifold_y': float(coords[i, 1]),
                'manifold_z': float(coords[i, 2]),
                'explained_variance': explained,
            })

    return pd.DataFrame(results)


# =============================================================================
# BASIC DERIVATIVES (point-wise)
# =============================================================================

def compute_derivatives_pointwise(obs: pd.DataFrame) -> pd.DataFrame:
    """Per-point derivatives."""
    results = []

    for (entity_id, signal_id), group in obs.groupby(['entity_id', 'signal_id']):
        group = group.sort_values('I')
        y = group['y'].values
        I_vals = group['I'].values

        # First derivative
        dy = np.gradient(y)

        # Second derivative
        d2y = np.gradient(dy)

        for i, I in enumerate(I_vals):
            results.append({
                'entity_id': entity_id,
                'signal_id': signal_id,
                'I': I,
                'dy': float(dy[i]),
                'd2y': float(d2y[i]),
            })

    return pd.DataFrame(results)


# =============================================================================
# BASIC CORRELATIONS (pairwise)
# =============================================================================

def compute_correlation_matrix(obs: pd.DataFrame) -> pd.DataFrame:
    """Basic pairwise correlations."""
    from itertools import combinations

    results = []

    for entity_id, entity_group in obs.groupby('entity_id'):
        signals = list(entity_group['signal_id'].unique())

        if len(signals) < 2:
            continue

        # Pivot to wide
        try:
            wide = entity_group.pivot(index='I', columns='signal_id', values='y')
            wide = wide.sort_index().dropna()
        except Exception:
            continue

        if len(wide) < 10:
            continue

        for s1, s2 in combinations(signals, 2):
            if s1 not in wide.columns or s2 not in wide.columns:
                continue

            y1 = wide[s1].values
            y2 = wide[s2].values

            if np.std(y1) < 1e-10 or np.std(y2) < 1e-10:
                corr = 0.0
            else:
                corr = float(np.corrcoef(y1, y2)[0, 1])
                if np.isnan(corr):
                    corr = 0.0

            results.append({
                'entity_id': entity_id,
                'signal_a': s1,
                'signal_b': s2,
                'correlation': corr,
            })

    return pd.DataFrame(results)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    start_time = time.time()

    print("=" * 70)
    print("PRISM PIPELINE: EXPLICIT ENGINE MODE")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. LOAD OBSERVATIONS
    # -------------------------------------------------------------------------
    print("\n[1/7] Loading observations...")

    if not INPUT_FILE.exists():
        print(f"  ERROR: {INPUT_FILE} not found")
        sys.exit(1)

    obs = pd.read_parquet(INPUT_FILE)
    n_obs = len(obs)
    n_entities = obs['entity_id'].nunique()
    n_signals = obs['signal_id'].nunique()

    print(f"  {n_obs:,} observations")
    print(f"  {n_entities} entities")
    print(f"  {n_signals} signals")

    # -------------------------------------------------------------------------
    # 2. COMPUTE TYPOLOGY
    # -------------------------------------------------------------------------
    print("\n[2/7] Computing typology...")

    typology = compute_typology(obs)
    typology.to_parquet(DATA_DIR / 'typology.parquet', index=False)

    type_counts = typology['signal_class'].value_counts().to_dict()
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")
    print(f"  ✓ Saved typology.parquet ({len(typology)} signals, {len(typology.columns)} cols)")

    # -------------------------------------------------------------------------
    # 3. RUN SIGNAL ENGINES
    # -------------------------------------------------------------------------
    print("\n[3/7] Running signal engines...")
    print(f"  {len(SIGNAL_ENGINES)} engines to run")

    signal_results = [typology]  # Start with typology as base

    for name, engine in SIGNAL_ENGINES:
        try:
            result = engine.compute(obs)
            if len(result) > 0:
                signal_results.append(result)
                n_cols = len([c for c in result.columns if c not in ['entity_id', 'signal_id']])
                print(f"  ✓ {name}: {n_cols} metrics")
            else:
                print(f"  ○ {name}: empty result")
        except Exception as e:
            print(f"  ✗ {name}: {str(e)[:50]}")

    # Merge all signal results
    primitives = signal_results[0]
    for df in signal_results[1:]:
        if 'entity_id' in df.columns and 'signal_id' in df.columns:
            primitives = primitives.merge(
                df,
                on=['entity_id', 'signal_id'],
                how='left',
                suffixes=('', '_dup')
            )

    # Drop duplicate columns
    primitives = primitives.loc[:, ~primitives.columns.str.endswith('_dup')]
    primitives.to_parquet(DATA_DIR / 'primitives.parquet', index=False)
    print(f"  ✓ Saved primitives.parquet ({len(primitives)} rows × {len(primitives.columns)} cols)")

    # -------------------------------------------------------------------------
    # 4. RUN PAIRWISE ENGINES
    # -------------------------------------------------------------------------
    print("\n[4/7] Running pairwise engines...")
    print(f"  {len(PAIRWISE_ENGINES)} engines to run")

    # Start with basic correlations
    pair_results = [compute_correlation_matrix(obs)]
    if len(pair_results[0]) > 0:
        print(f"  ✓ correlation: {len(pair_results[0])} pairs")

    for name, engine in PAIRWISE_ENGINES:
        try:
            result = engine.compute(obs)
            if len(result) > 0:
                pair_results.append(result)
                n_cols = len([c for c in result.columns
                            if c not in ['entity_id', 'signal_a', 'signal_b', 'source_id', 'target_id']])
                print(f"  ✓ {name}: {len(result)} pairs, {n_cols} metrics")
            else:
                print(f"  ○ {name}: empty result")
        except Exception as e:
            print(f"  ✗ {name}: {str(e)[:50]}")

    # Normalize column names and merge
    if pair_results:
        for i, df in enumerate(pair_results):
            if 'source_id' in df.columns:
                pair_results[i] = df.rename(columns={'source_id': 'signal_a', 'target_id': 'signal_b'})

        primitives_pairs = pair_results[0]
        for df in pair_results[1:]:
            merge_cols = [c for c in ['entity_id', 'signal_a', 'signal_b']
                         if c in df.columns and c in primitives_pairs.columns]
            if merge_cols:
                primitives_pairs = primitives_pairs.merge(
                    df, on=merge_cols, how='outer', suffixes=('', '_dup')
                )
        primitives_pairs = primitives_pairs.loc[:, ~primitives_pairs.columns.str.endswith('_dup')]
    else:
        primitives_pairs = pd.DataFrame()

    primitives_pairs.to_parquet(DATA_DIR / 'primitives_pairs.parquet', index=False)
    print(f"  ✓ Saved primitives_pairs.parquet ({len(primitives_pairs)} rows × {len(primitives_pairs.columns)} cols)")

    # -------------------------------------------------------------------------
    # 5. RUN POINT ENGINES
    # -------------------------------------------------------------------------
    print("\n[5/7] Running point engines...")
    print(f"  {len(POINT_ENGINES)} engines to run")

    # Start with derivatives
    point_results = [compute_derivatives_pointwise(obs)]
    print(f"  ✓ derivatives: {len(point_results[0])} points")

    for name, engine in POINT_ENGINES:
        try:
            result = engine.compute(obs)
            if len(result) > 0:
                point_results.append(result)
                n_cols = len([c for c in result.columns if c not in ['entity_id', 'signal_id', 'I']])
                print(f"  ✓ {name}: {len(result)} points, {n_cols} metrics")
            else:
                print(f"  ○ {name}: empty result")
        except Exception as e:
            print(f"  ✗ {name}: {str(e)[:50]}")

    # Merge all point results
    primitives_points = obs.copy()
    for df in point_results:
        merge_cols = ['entity_id', 'signal_id', 'I']
        if all(c in df.columns for c in merge_cols):
            primitives_points = primitives_points.merge(
                df, on=merge_cols, how='left', suffixes=('', '_dup')
            )
        elif 'entity_id' in df.columns and 'I' in df.columns:
            # Point engines that don't have signal_id (regime detection)
            primitives_points = primitives_points.merge(
                df, on=['entity_id', 'I'], how='left', suffixes=('', '_dup')
            )

    primitives_points = primitives_points.loc[:, ~primitives_points.columns.str.endswith('_dup')]
    primitives_points.to_parquet(DATA_DIR / 'primitives_points.parquet', index=False)
    print(f"  ✓ Saved primitives_points.parquet ({len(primitives_points)} rows × {len(primitives_points.columns)} cols)")

    # -------------------------------------------------------------------------
    # 6. COMPUTE MANIFOLD TRAJECTORY
    # -------------------------------------------------------------------------
    print("\n[6/7] Computing manifold trajectory...")

    manifold = compute_manifold_trajectory(obs)
    manifold.to_parquet(DATA_DIR / 'manifold.parquet', index=False)
    print(f"  ✓ Saved manifold.parquet ({len(manifold)} trajectory points)")

    # -------------------------------------------------------------------------
    # 7. SUMMARY
    # -------------------------------------------------------------------------
    elapsed = time.time() - start_time

    print("\n[7/7] Complete!")
    print("=" * 70)
    print(f"Time: {elapsed:.1f}s")
    print("=" * 70)
    print("OUTPUTS:")
    print("=" * 70)

    outputs = [
        ('typology.parquet', typology),
        ('primitives.parquet', primitives),
        ('primitives_pairs.parquet', primitives_pairs),
        ('primitives_points.parquet', primitives_points),
        ('manifold.parquet', manifold),
    ]

    for filename, df in outputs:
        filepath = DATA_DIR / filename
        if filepath.exists():
            size = filepath.stat().st_size
            size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/1024/1024:.1f} MB"
            print(f"  {filename:30} {len(df):>10,} rows × {len(df.columns):>3} cols  [{size_str}]")
        else:
            print(f"  {filename:30} NOT CREATED")

    print("=" * 70)


if __name__ == '__main__':
    main()
