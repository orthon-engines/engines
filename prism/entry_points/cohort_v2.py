"""
Ørthon Cohort Discovery v2

Two-stage signal organization:
1. COHORT = Entity: Signals that share temporal structure belong together
2. SIGNAL TYPE = Behavior: Signals that behave similarly across cohorts are the same type

Input: observations.parquet, vector.parquet
Output: cohorts_v2.parquet with entity assignment + signal type labels
"""

import polars as pl
import numpy as np
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import json
import os

DATA_PATH = Path(os.environ.get("ORTHON_DATA_PATH", "data/battery"))


def load_data():
    """Load observations and vector data."""
    obs = pl.read_parquet(DATA_PATH / "observations.parquet")
    vec = pl.read_parquet(DATA_PATH / "vector.parquet")
    return obs, vec


def discover_cohorts_by_temporal_alignment(obs: pl.DataFrame) -> dict:
    """
    Stage 1: Group signals into cohorts based on temporal alignment.
    Signals from the same entity share timestamps.

    Returns: {signal_key: cohort_id} where signal_key = "entity_id::signal_id"
    """
    print("\n" + "="*70)
    print("STAGE 1: Cohort Discovery (Temporal Alignment)")
    print("="*70)

    # Get unique (entity, signal) pairs and their timestamps
    signal_timestamps = {}

    for row in obs.iter_rows(named=True):
        key = f"{row['entity_id']}::{row['signal_id']}"
        if key not in signal_timestamps:
            signal_timestamps[key] = set()
        signal_timestamps[key].add(row['timestamp'])

    # Convert to sorted tuples for comparison
    signal_ts_tuples = {k: tuple(sorted(v)) for k, v in signal_timestamps.items()}

    # Group by timestamp pattern
    ts_to_signals = {}
    for sig, ts in signal_ts_tuples.items():
        # Use hash of first/last 10 timestamps + length as fingerprint
        ts_list = list(ts)
        fingerprint = (len(ts_list), ts_list[0] if ts_list else 0, ts_list[-1] if ts_list else 0)
        if fingerprint not in ts_to_signals:
            ts_to_signals[fingerprint] = []
        ts_to_signals[fingerprint].append(sig)

    # Actually, simpler: signals from same entity_id are same cohort
    entity_signals = obs.group_by("entity_id").agg(
        pl.col("signal_id").unique().alias("signals")
    )

    cohorts = {}
    for i, row in enumerate(entity_signals.iter_rows(named=True)):
        entity = row['entity_id']
        signals = row['signals']
        cohort_id = f"cohort_{i}"
        print(f"\n  {cohort_id}: {entity}")
        print(f"    Signals ({len(signals)}): {', '.join(sorted(signals)[:5])}{'...' if len(signals) > 5 else ''}")

        for sig in signals:
            cohorts[f"{entity}::{sig}"] = {
                'cohort_id': cohort_id,
                'entity_id': entity,
                'signal_id': sig
            }

    print(f"\n  Total cohorts: {len(entity_signals)}")
    return cohorts


def discover_signal_types_by_behavior(vec: pl.DataFrame, cohorts: dict) -> dict:
    """
    Stage 2: Within each cohort, identify signal types by behavioral similarity.
    Signals with similar vector fingerprints are the same "type".

    Returns: Updated cohorts dict with signal_type added
    """
    print("\n" + "="*70)
    print("STAGE 2: Signal Type Discovery (Behavioral Fingerprints)")
    print("="*70)

    # Build behavioral fingerprint for each signal
    # Aggregate vector features per (entity, source_signal)

    fingerprints = vec.group_by(["entity_id", "source_signal"]).agg([
        pl.col("value").mean().alias("mean"),
        pl.col("value").std().alias("std"),
        pl.col("value").min().alias("min"),
        pl.col("value").max().alias("max"),
        pl.col("value").quantile(0.25).alias("q25"),
        pl.col("value").quantile(0.75).alias("q75"),
    ])

    # Pivot to get feature matrix per signal
    # Each row = one signal, columns = statistical summaries

    signal_features = {}
    for row in fingerprints.iter_rows(named=True):
        key = f"{row['entity_id']}::{row['source_signal']}"
        signal_features[key] = [
            row['mean'] or 0,
            row['std'] or 0,
            row['min'] or 0,
            row['max'] or 0,
            row['q25'] or 0,
            row['q75'] or 0,
        ]

    # Get unique signal names (without entity prefix)
    signal_names = set()
    for key in cohorts.keys():
        _, sig = key.split("::")
        signal_names.add(sig)

    print(f"\n  Unique signal names: {len(signal_names)}")

    # For each signal name, check if it behaves consistently across entities
    signal_type_map = {}

    for sig_name in sorted(signal_names):
        # Get all instances of this signal across entities
        instances = [(k, v) for k, v in signal_features.items() if k.endswith(f"::{sig_name}")]

        if len(instances) > 1:
            # Check behavioral consistency
            features = np.array([v for _, v in instances])

            # Normalize and compute variance across entities
            if features.std(axis=0).sum() > 0:
                features_norm = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)
                variance = np.var(features_norm, axis=0).mean()
            else:
                variance = 0

            signal_type_map[sig_name] = {
                'instances': len(instances),
                'cross_entity_variance': variance,
                'consistent': variance < 1.0  # Threshold for "same behavior"
            }
        else:
            signal_type_map[sig_name] = {
                'instances': len(instances),
                'cross_entity_variance': 0,
                'consistent': True
            }

    # Cluster signals by behavioral similarity (across all entities)
    # Build feature matrix: rows = signal types, columns = avg features

    signal_names_list = sorted(signal_names)
    avg_features = []

    for sig_name in signal_names_list:
        instances = [v for k, v in signal_features.items() if k.endswith(f"::{sig_name}")]
        if instances:
            avg_features.append(np.mean(instances, axis=0))
        else:
            avg_features.append([0]*6)

    avg_features = np.array(avg_features)

    # Normalize
    scaler = StandardScaler()
    if len(avg_features) > 1 and avg_features.std() > 0:
        avg_features_norm = scaler.fit_transform(avg_features)
    else:
        avg_features_norm = avg_features

    # Cluster into signal types
    n_types = min(5, len(signal_names_list))  # Up to 5 signal types

    if len(signal_names_list) > 1:
        clustering = AgglomerativeClustering(n_clusters=n_types, linkage='ward')
        type_labels = clustering.fit_predict(avg_features_norm)
    else:
        type_labels = [0]

    # Assign type names
    type_names = ['capacity', 'impedance', 'temperature', 'voltage', 'misc']
    signal_to_type = {}

    for sig_name, type_id in zip(signal_names_list, type_labels):
        signal_to_type[sig_name] = f"type_{type_id}"

    # Print signal type clusters
    print("\n  Signal Type Clusters:")
    for type_id in range(n_types):
        members = [s for s, t in signal_to_type.items() if t == f"type_{type_id}"]
        if members:
            print(f"    type_{type_id}: {', '.join(members)}")

    # Update cohorts with signal type
    for key in cohorts:
        _, sig = key.split("::")
        cohorts[key]['signal_type'] = signal_to_type.get(sig, 'unknown')

    return cohorts, signal_to_type


def cross_entity_signal_matching(obs: pl.DataFrame, cohorts: dict) -> pl.DataFrame:
    """
    Stage 3: Match signals across entities by behavior.

    For unlabeled data: "Signal 7 in Entity A behaves like Signal 12 in Entity B"
    """
    print("\n" + "="*70)
    print("STAGE 3: Cross-Entity Signal Matching")
    print("="*70)

    # Group by signal type and show which signals match
    type_to_signals = {}

    for key, info in cohorts.items():
        sig_type = info['signal_type']
        entity = info['entity_id']
        signal = info['signal_id']

        if sig_type not in type_to_signals:
            type_to_signals[sig_type] = {}
        if entity not in type_to_signals[sig_type]:
            type_to_signals[sig_type][entity] = []
        type_to_signals[sig_type][entity].append(signal)

    print("\n  Cross-Entity Signal Matching:")
    for sig_type in sorted(type_to_signals.keys()):
        print(f"\n  {sig_type}:")
        for entity in sorted(type_to_signals[sig_type].keys())[:5]:  # Limit to 5 entities for readability
            signals = type_to_signals[sig_type][entity]
            print(f"    {entity}: {', '.join(sorted(signals))}")
        if len(type_to_signals[sig_type]) > 5:
            print(f"    ... and {len(type_to_signals[sig_type]) - 5} more entities")

    return type_to_signals


def format_report(cohorts: dict, signal_types: dict, cross_match: dict) -> str:
    """Generate human-readable report."""

    lines = []
    lines.append("="*70)
    lines.append("ØRTHON COHORT ANALYSIS")
    lines.append("signal typology")
    lines.append("="*70)

    # Group by cohort (entity)
    cohort_groups = {}
    for key, info in cohorts.items():
        cid = info['cohort_id']
        if cid not in cohort_groups:
            cohort_groups[cid] = {
                'entity': info['entity_id'],
                'signals': {}
            }
        sig_type = info['signal_type']
        if sig_type not in cohort_groups[cid]['signals']:
            cohort_groups[cid]['signals'][sig_type] = []
        cohort_groups[cid]['signals'][sig_type].append(info['signal_id'])

    # Print each cohort (limit for large datasets)
    cohort_ids = sorted(cohort_groups.keys())[:10]  # First 10 cohorts
    for cid in cohort_ids:
        group = cohort_groups[cid]
        lines.append("")
        lines.append(f"COHORT: {group['entity']}")
        lines.append("─"*50)

        for sig_type in sorted(group['signals'].keys()):
            signals = sorted(group['signals'][sig_type])
            lines.append(f"  [{sig_type}]")
            lines.append(f"    {', '.join(signals)}")

    if len(cohort_groups) > 10:
        lines.append(f"\n... and {len(cohort_groups) - 10} more cohorts")

    # Cross-entity matching summary
    lines.append("")
    lines.append("="*70)
    lines.append("SIGNAL TYPE MATCHING (across cohorts)")
    lines.append("="*70)

    for sig_type in sorted(cross_match.keys()):
        lines.append(f"\n{sig_type}:")
        entities = sorted(cross_match[sig_type].keys())[:5]
        for entity in entities:
            signals = cross_match[sig_type][entity]
            lines.append(f"  {entity}: {', '.join(sorted(signals))}")
        if len(cross_match[sig_type]) > 5:
            lines.append(f"  ... ({len(cross_match[sig_type])} entities total)")

    return "\n".join(lines)


def save_results(cohorts: dict, signal_types: dict, report: str):
    """Save cohort assignments and report."""

    # Convert to DataFrame
    rows = []
    for key, info in cohorts.items():
        rows.append({
            'entity_id': info['entity_id'],
            'signal_id': info['signal_id'],
            'cohort_id': info['cohort_id'],
            'signal_type': info['signal_type'],
        })

    df = pl.DataFrame(rows)
    df.write_parquet(DATA_PATH / "cohorts_v2.parquet")
    print(f"\n  Saved: {DATA_PATH}/cohorts_v2.parquet")

    # Save report
    with open(DATA_PATH / "cohorts_v2_report.txt", "w") as f:
        f.write(report)
    print(f"  Saved: {DATA_PATH}/cohorts_v2_report.txt")

    # Save signal types
    with open(DATA_PATH / "signal_types.json", "w") as f:
        json.dump(signal_types, f, indent=2)
    print(f"  Saved: {DATA_PATH}/signal_types.json")

    return df


def main():
    print("="*70)
    print("ØRTHON COHORT DISCOVERY v2")
    print("="*70)
    print(f"Data path: {DATA_PATH}")

    # Load data
    obs, vec = load_data()
    print(f"\nObservations: {len(obs):,} rows")
    print(f"Vector features: {len(vec):,} rows")

    # Stage 1: Discover cohorts by temporal alignment
    cohorts = discover_cohorts_by_temporal_alignment(obs)

    # Stage 2: Discover signal types by behavior
    cohorts, signal_types = discover_signal_types_by_behavior(vec, cohorts)

    # Stage 3: Cross-entity matching
    cross_match = cross_entity_signal_matching(obs, cohorts)

    # Generate report
    report = format_report(cohorts, signal_types, cross_match)
    print("\n" + report)

    # Save results
    df = save_results(cohorts, signal_types, report)

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)

    return df


if __name__ == "__main__":
    main()
