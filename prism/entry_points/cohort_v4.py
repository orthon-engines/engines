"""
Ørthon Cohort Discovery v4 - Regime-Aware Signal Matching

Key insight: Multi-regime systems need per-regime analysis.
Compute fingerprints WITHIN each operating regime, then aggregate.

Uses ALL available data:
- observations.parquet: Raw signals + operating conditions
- vector.parquet: Derived behavioral features
- geometry.parquet: Mode assignments

Input: observations.parquet, vector.parquet, geometry.parquet
Output: cohorts_v4.parquet with signal matching across entities
"""

import polars as pl
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json
import os
import warnings

warnings.filterwarnings('ignore')

DATA_PATH = Path(os.environ.get("ORTHON_DATA_PATH", "data/FD002"))


def load_all_data():
    """Load all available data."""
    obs = pl.read_parquet(DATA_PATH / "observations.parquet")
    vec = pl.read_parquet(DATA_PATH / "vector.parquet")

    geom_path = DATA_PATH / "geometry.parquet"
    geom = pl.read_parquet(geom_path) if geom_path.exists() else None

    state_path = DATA_PATH / "state.parquet"
    state = pl.read_parquet(state_path) if state_path.exists() else None

    return obs, vec, geom, state


def identify_operating_regimes(obs: pl.DataFrame, n_regimes: int = 6):
    """
    Identify operating regimes from operating condition signals.
    Uses altitude and op_mach (or similar) to cluster into regimes.
    """
    # Find operating condition signals
    op_signals = ["altitude", "op_mach", "mach", "operating_condition"]
    available = [s for s in op_signals if s in obs["signal_id"].unique().to_list()]

    if len(available) < 2:
        # Fall back to using mode_id from geometry if available
        print("  No operating condition signals found, using single regime")
        return None

    # Pivot operating conditions
    op_conditions = obs.filter(
        pl.col("signal_id").is_in(available[:2])
    ).pivot(
        values="value",
        index=["entity_id", "timestamp"],
        on="signal_id"
    ).drop_nulls()

    if len(op_conditions) < n_regimes:
        return None

    # Cluster into regimes
    op_cols = [c for c in op_conditions.columns if c not in ["entity_id", "timestamp"]]
    op_matrix = op_conditions.select(op_cols).to_numpy()

    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    regime_labels = kmeans.fit_predict(op_matrix)

    op_conditions = op_conditions.with_columns(
        pl.Series("regime", regime_labels)
    )

    print(f"  Identified {n_regimes} operating regimes:")
    for i, center in enumerate(kmeans.cluster_centers_):
        count = (regime_labels == i).sum()
        print(f"    Regime {i}: {op_cols[0]}={center[0]:.1f}, {op_cols[1]}={center[1]:.3f} ({count:,} samples)")

    return op_conditions


def build_regime_fingerprints(obs: pl.DataFrame, op_conditions: pl.DataFrame, entity_id: str, n_regimes: int = 6):
    """
    Build correlation fingerprint per signal, computed within each operating regime.
    Returns weighted aggregate fingerprint.
    """
    entity_ops = op_conditions.filter(pl.col("entity_id") == entity_id)
    entity_obs = obs.filter(pl.col("entity_id") == entity_id)

    # Join to get regime per observation
    entity_data = entity_obs.join(
        entity_ops.select(["timestamp", "regime"]),
        on="timestamp",
        how="inner"
    )

    signal_names = sorted(entity_data["signal_id"].unique().to_list())
    n_signals = len(signal_names)

    # Compute correlation matrix per regime
    regime_corrs = []
    regime_counts = []

    for regime in range(n_regimes):
        regime_data = entity_data.filter(pl.col("regime") == regime)
        regime_counts.append(len(regime_data))

        if len(regime_data) < 10:
            regime_corrs.append(np.zeros((n_signals, n_signals)))
            continue

        # Pivot to matrix
        pivot = regime_data.pivot(
            values="value",
            index="timestamp",
            on="signal_id"
        ).sort("timestamp")

        # Build matrix in consistent signal order
        matrix = np.column_stack([
            pivot[s].to_numpy() if s in pivot.columns else np.zeros(len(pivot))
            for s in signal_names
        ])

        # Handle NaN and normalize
        matrix = np.nan_to_num(matrix, nan=0.0)

        if matrix.std() > 0:
            scaler = StandardScaler()
            matrix_norm = scaler.fit_transform(matrix)
            corr = np.corrcoef(matrix_norm.T)
            corr = np.nan_to_num(corr, nan=0.0)
        else:
            corr = np.zeros((n_signals, n_signals))

        regime_corrs.append(corr)

    # Weighted aggregate across regimes
    total = sum(regime_counts)
    if total > 0:
        weights = [c / total for c in regime_counts]
        agg_corr = sum(w * c for w, c in zip(weights, regime_corrs))
    else:
        agg_corr = np.zeros((n_signals, n_signals))

    return signal_names, agg_corr


def build_vector_mode_fingerprints(vec: pl.DataFrame, entity_id: str):
    """
    Build fingerprint using vector features aggregated by mode.
    """
    entity_vec = vec.filter(pl.col("entity_id") == entity_id)

    if "mode_id" not in entity_vec.columns:
        # Fallback: just aggregate all
        agg = entity_vec.group_by("source_signal").agg([
            pl.col("value").mean().alias("mean"),
            pl.col("value").std().alias("std"),
        ])
        signal_names = sorted(agg["source_signal"].to_list())
        features = np.column_stack([
            agg.filter(pl.col("source_signal") == s).select(["mean", "std"]).to_numpy().flatten()
            if s in agg["source_signal"].to_list() else [0, 0]
            for s in signal_names
        ]).T
        return signal_names, features

    # Aggregate by mode
    mode_agg = entity_vec.group_by(["source_signal", "mode_id"]).agg([
        pl.col("value").mean().alias("mean"),
        pl.col("value").std().alias("std"),
    ])

    signal_names = sorted(mode_agg["source_signal"].unique().to_list())
    modes = sorted(mode_agg["mode_id"].drop_nulls().unique().to_list())

    features = []
    for sig in signal_names:
        sig_data = mode_agg.filter(pl.col("source_signal") == sig)
        sig_features = []
        for mode in modes:
            mode_row = sig_data.filter(pl.col("mode_id") == mode)
            if len(mode_row) > 0:
                mean_val = mode_row["mean"].to_list()[0]
                std_val = mode_row["std"].to_list()[0]
                sig_features.extend([
                    mean_val if mean_val is not None else 0,
                    std_val if std_val is not None else 0
                ])
            else:
                sig_features.extend([0, 0])
        features.append(sig_features)

    return signal_names, np.array(features)


def combined_fingerprint(obs, vec, op_conditions, entity_id, n_regimes=6):
    """
    Combine regime-aware correlation + vector mode features.
    """
    # Regime-aware correlations
    names_corr, corr_fp = build_regime_fingerprints(obs, op_conditions, entity_id, n_regimes)

    # Vector features by mode
    names_vec, vec_fp = build_vector_mode_fingerprints(vec, entity_id)

    # Align signal names
    common_names = sorted(set(names_corr) & set(names_vec))

    if len(common_names) == 0:
        return names_corr, corr_fp

    # Re-index to common names
    corr_idx = [names_corr.index(n) for n in common_names]
    vec_idx = [names_vec.index(n) for n in common_names]

    corr_fp = corr_fp[np.ix_(corr_idx, corr_idx)]
    vec_fp = vec_fp[vec_idx]

    # Normalize vector features
    if vec_fp.std() > 0:
        vec_fp = (vec_fp - vec_fp.mean()) / vec_fp.std()

    # Combine
    combined = np.hstack([
        corr_fp * 2.0,
        vec_fp * 1.0,
    ])

    return common_names, combined


def match_signals(fp_a, fp_b, names_a, names_b):
    """
    Use Hungarian algorithm for optimal 1-to-1 matching.
    """
    dist = cdist(fp_a, fp_b, metric='correlation')
    dist = np.nan_to_num(dist, nan=1.0)

    row_ind, col_ind = linear_sum_assignment(dist)

    matches = []
    for i, j in zip(row_ind, col_ind):
        matches.append({
            'signal_a': names_a[i],
            'signal_b': names_b[j],
            'similarity': 1 - dist[i, j],
            'correct': names_a[i] == names_b[j]
        })

    return matches


def cross_entity_matching(obs, vec, op_conditions, entities, n_regimes=6):
    """
    Match signals across all entity pairs.
    """
    results = {}

    # Build fingerprints for all entities
    print("\n  Building fingerprints...")
    fingerprints = {}
    for entity in entities:
        names, fp = combined_fingerprint(obs, vec, op_conditions, entity, n_regimes)
        fingerprints[entity] = (names, fp)

    # Match consecutive pairs
    print("  Matching entity pairs...")
    for i in range(len(entities) - 1):
        entity_a = entities[i]
        entity_b = entities[i + 1]

        names_a, fp_a = fingerprints[entity_a]
        names_b, fp_b = fingerprints[entity_b]

        matches = match_signals(fp_a, fp_b, names_a, names_b)
        correct = sum(1 for m in matches if m['correct'])
        accuracy = correct / len(matches) * 100

        results[f"{entity_a}_to_{entity_b}"] = {
            'matches': matches,
            'accuracy': accuracy,
            'correct': correct,
            'total': len(matches)
        }

    return results


def main():
    print("="*70)
    print("ØRTHON COHORT DISCOVERY v4")
    print("Regime-Aware Signal Matching")
    print("="*70)
    print(f"\nData path: {DATA_PATH}")

    # Load all data
    print("\nLoading data...")
    obs, vec, geom, state = load_all_data()
    print(f"  Observations: {len(obs):,}")
    print(f"  Vector: {len(vec):,}")
    print(f"  Geometry: {len(geom):,}" if geom is not None else "  Geometry: None")
    print(f"  State: {len(state):,}" if state is not None else "  State: None")

    # Identify operating regimes
    print("\nIdentifying operating regimes...")
    op_conditions = identify_operating_regimes(obs)

    if op_conditions is None:
        print("  Using single-regime fallback")
        # Create dummy single regime
        op_conditions = obs.select(["entity_id", "timestamp"]).unique().with_columns(
            pl.lit(0).alias("regime")
        )
        n_regimes = 1
    else:
        n_regimes = 6

    # Get entities
    entities = sorted(obs["entity_id"].unique().to_list())[:10]  # Limit for speed
    print(f"\nProcessing {len(entities)} entities: {entities[:5]}...")

    # Cross-entity matching
    print("\nCross-entity signal matching...")
    results = cross_entity_matching(obs, vec, op_conditions, entities, n_regimes)

    # Print results
    print("\n" + "="*70)
    print("MATCHING RESULTS")
    print("="*70)

    total_correct = 0
    total_signals = 0

    for pair, data in results.items():
        print(f"\n{pair}: {data['correct']}/{data['total']} ({data['accuracy']:.1f}%)")
        total_correct += data['correct']
        total_signals += data['total']

        # Show mismatches
        mismatches = [m for m in data['matches'] if not m['correct']]
        if mismatches:
            print("  Mismatches:")
            for m in mismatches[:3]:
                print(f"    {m['signal_a']} -> {m['signal_b']} (sim={m['similarity']:.3f})")

    overall = total_correct / total_signals * 100 if total_signals > 0 else 0
    print(f"\n{'='*70}")
    print(f"OVERALL ACCURACY: {total_correct}/{total_signals} ({overall:.1f}%)")
    print("="*70)

    # Save results
    output = {
        'data_path': str(DATA_PATH),
        'n_entities': len(entities),
        'n_regimes': n_regimes,
        'overall_accuracy': overall,
        'pair_results': {k: {'accuracy': v['accuracy'], 'correct': v['correct'], 'total': v['total']}
                         for k, v in results.items()}
    }

    with open(DATA_PATH / "cohort_matching_v4.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {DATA_PATH}/cohort_matching_v4.json")


if __name__ == "__main__":
    main()
