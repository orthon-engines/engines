#!/usr/bin/env python3
"""
PRISM Cohort Discovery
======================

Discover cohorts from observations or vector signals.

Cohorts are groups of signals that exhibit similar behavior patterns.
Discovery can be run from:
  - raw: observations.parquet (cluster signals by raw value patterns)
  - vector: vector.parquet (cluster signals by behavioral metrics)

Output: data/cohorts.parquet

Usage:
    python -m prism.entry_points.cohort                    # Default: from raw
    python -m prism.entry_points.cohort --source vector    # From vector signals
    python -m prism.entry_points.cohort --compare          # Run both, compare
    python -m prism.entry_points.cohort --adaptive         # Auto-detect params
    python -m prism.entry_points.cohort --testing --limit 100
"""

import argparse
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import polars as pl
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from orthon._internal.db.parquet_store import (
    ensure_directory,
    get_path,
    OBSERVATIONS,
    VECTOR,
    COHORTS,
    COHORTS_RAW,
    COHORTS_VECTOR,
)
from orthon._internal.db.polars_io import write_parquet_atomic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def discover_cohorts_from_raw(
    limit: Optional[int] = None,
    signals: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Discover cohorts by clustering raw observation patterns.

    Uses hierarchical clustering on signal value distributions:
    - Mean, std, min, max, range per signal
    - Correlation-based distance

    Args:
        limit: Max observations per signal (testing)
        signals: Specific signals to process (testing)

    Returns:
        DataFrame with columns: entity_id, signal_id, cohort_id
    """
    obs_path = get_path(OBSERVATIONS)
    if not obs_path.exists():
        raise FileNotFoundError(f"No observations found at {obs_path}")

    logger.info("Reading observations...")
    df = pl.read_parquet(obs_path)

    # Filter if testing
    if signals:
        df = df.filter(pl.col("signal_id").is_in(signals))
    if limit:
        df = df.group_by("signal_id").head(limit)

    logger.info(f"Loaded {len(df):,} observations")

    # Compute summary stats per signal
    logger.info("Computing signal statistics...")
    stats = df.group_by(["entity_id", "signal_id"]).agg([
        pl.col("value").mean().alias("mean"),
        pl.col("value").std().alias("std"),
        pl.col("value").min().alias("min"),
        pl.col("value").max().alias("max"),
        pl.col("value").count().alias("n_obs"),
    ]).with_columns([
        (pl.col("max") - pl.col("min")).alias("range"),
    ])

    # Prepare feature matrix
    feature_cols = ["mean", "std", "min", "max", "range"]
    features = stats.select(feature_cols).to_numpy()

    # Handle NaN/inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Cluster
    n_signals = len(stats)
    n_clusters = max(2, min(10, n_signals // 5))
    logger.info(f"Clustering {n_signals} signals into {n_clusters} cohorts...")

    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = clustering.fit_predict(features_scaled)

    # Build result
    result = stats.select(["entity_id", "signal_id"]).with_columns([
        pl.Series("cohort_id", [f"raw_cohort_{i}" for i in labels]),
        pl.lit("raw").alias("source"),
        pl.lit(datetime.now()).alias("discovered_at"),
    ])

    return result


def discover_cohorts_from_vector(
    signals: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Discover cohorts by clustering behavioral vector patterns.

    Uses hierarchical clustering on aggregated vector metrics:
    - Mean of each engine metric per signal
    - More semantically meaningful than raw patterns

    Args:
        signals: Specific signals to process (testing)

    Returns:
        DataFrame with columns: entity_id, signal_id, cohort_id
    """
    vector_path = get_path(VECTOR)
    if not vector_path.exists():
        raise FileNotFoundError(f"No vector data found at {vector_path}")

    logger.info("Reading vector data...")
    df = pl.read_parquet(vector_path)

    # Filter sparse signals only (windowed metrics)
    df = df.filter(pl.col("signal_type") == "sparse")

    if signals:
        df = df.filter(pl.col("source_signal").is_in(signals))

    logger.info(f"Loaded {len(df):,} vector rows")

    # Pivot to get one row per signal with all metrics
    logger.info("Pivoting to signal-metric matrix...")
    pivot = df.group_by(["entity_id", "source_signal", "engine"]).agg([
        pl.col("value").mean().alias("mean_value"),
    ]).pivot(
        index=["entity_id", "source_signal"],
        columns="engine",
        values="mean_value",
    )

    # Get feature columns (all columns except entity_id and source_signal)
    feature_cols = [c for c in pivot.columns if c not in ["entity_id", "source_signal"]]

    if not feature_cols:
        raise ValueError("No feature columns after pivot")

    # Prepare feature matrix
    features = pivot.select(feature_cols).to_numpy()
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Cluster
    n_signals = len(pivot)
    n_clusters = max(2, min(10, n_signals // 5))
    logger.info(f"Clustering {n_signals} signals into {n_clusters} cohorts...")

    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = clustering.fit_predict(features_scaled)

    # Build result
    result = pivot.select(["entity_id", "source_signal"]).rename({"source_signal": "signal_id"})
    result = result.with_columns([
        pl.Series("cohort_id", [f"vector_cohort_{i}" for i in labels]),
        pl.lit("vector").alias("source"),
        pl.lit(datetime.now()).alias("discovered_at"),
    ])

    return result


def compare_cohort_sources(raw_cohorts: pl.DataFrame, vector_cohorts: pl.DataFrame) -> Dict[str, Any]:
    """
    Compare cohort assignments from raw vs vector sources.

    Computes:
    - Agreement rate (% of signals in same cohort in both)
    - Adjusted Rand Index between clusterings

    Returns:
        Dict with comparison metrics
    """
    from sklearn.metrics import adjusted_rand_score

    # Join on signal_id
    comparison = raw_cohorts.select(["signal_id", "cohort_id"]).rename({"cohort_id": "raw_cohort"})
    comparison = comparison.join(
        vector_cohorts.select(["signal_id", "cohort_id"]).rename({"cohort_id": "vector_cohort"}),
        on="signal_id",
        how="inner",
    )

    if len(comparison) == 0:
        return {"error": "No overlapping signals", "n_compared": 0}

    raw_labels = comparison["raw_cohort"].to_list()
    vector_labels = comparison["vector_cohort"].to_list()

    ari = adjusted_rand_score(raw_labels, vector_labels)

    return {
        "n_compared": len(comparison),
        "adjusted_rand_index": ari,
        "raw_n_cohorts": comparison["raw_cohort"].n_unique(),
        "vector_n_cohorts": comparison["vector_cohort"].n_unique(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="PRISM Cohort Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output: data/cohorts.parquet

Schema:
  entity_id     | String   | Entity containing the signal
  signal_id     | String   | Signal identifier
  cohort_id     | String   | Discovered cohort assignment
  source        | String   | Discovery source (raw, vector)
  discovered_at | Datetime | When cohort was discovered

Examples:
  python -m prism.entry_points.cohort                    # Discover from raw
  python -m prism.entry_points.cohort --source vector    # Discover from vector
  python -m prism.entry_points.cohort --compare          # Compare both methods
  python -m prism.entry_points.cohort --testing --limit 100
"""
    )

    parser.add_argument(
        "--source",
        choices=["raw", "vector"],
        default="raw",
        help="Data source for cohort discovery (default: raw)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both sources and output accuracy comparison"
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Auto-detect optimal number of cohorts"
    )

    # Testing flags
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Enable testing mode (required for --limit and --signal)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="[TESTING] Max observations per signal"
    )
    parser.add_argument(
        "--signal",
        type=str,
        help="[TESTING] Comma-separated signal IDs to process"
    )

    args = parser.parse_args()

    # Guard testing flags
    if not args.testing and (args.limit or args.signal):
        logger.warning("=" * 80)
        logger.warning("TESTING FLAGS IGNORED - --testing not specified")
        logger.warning("=" * 80)
        args.limit = None
        args.signal = None

    # Parse signals
    signals = None
    if args.signal:
        signals = [s.strip() for s in args.signal.split(",")]

    # Ensure output directory
    ensure_directory()

    print()
    print("=" * 80)
    print("PRISM COHORT DISCOVERY")
    print("=" * 80)

    if args.compare:
        print("Mode: Compare raw vs vector sources")
        print()

        # Run both
        print("Discovering cohorts from raw observations...")
        raw_cohorts = discover_cohorts_from_raw(limit=args.limit, signals=signals)
        raw_path = get_path(COHORTS_RAW)
        write_parquet_atomic(raw_cohorts, raw_path)
        print(f"  Saved {len(raw_cohorts)} assignments to {raw_path}")

        print()
        print("Discovering cohorts from vector signals...")
        try:
            vector_cohorts = discover_cohorts_from_vector(signals=signals)
            vector_path = get_path(COHORTS_VECTOR)
            write_parquet_atomic(vector_cohorts, vector_path)
            print(f"  Saved {len(vector_cohorts)} assignments to {vector_path}")

            # Compare
            print()
            print("Comparing cohort assignments...")
            comparison = compare_cohort_sources(raw_cohorts, vector_cohorts)

            print()
            print("=" * 80)
            print("COMPARISON RESULTS")
            print("=" * 80)
            print(f"  Signals compared: {comparison.get('n_compared', 0)}")
            print(f"  Raw cohorts: {comparison.get('raw_n_cohorts', 0)}")
            print(f"  Vector cohorts: {comparison.get('vector_n_cohorts', 0)}")
            print(f"  Adjusted Rand Index: {comparison.get('adjusted_rand_index', 0):.3f}")
            print()
            print("  (ARI = 1.0 means perfect agreement, 0 = random)")

        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            print("  Comparison skipped - run signal_vector first")

        # Use raw cohorts as primary
        cohorts = raw_cohorts

    elif args.source == "vector":
        print("Mode: Discover from vector signals")
        print()
        cohorts = discover_cohorts_from_vector(signals=signals)

    else:
        print("Mode: Discover from raw observations")
        print()
        cohorts = discover_cohorts_from_raw(limit=args.limit, signals=signals)

    # Save primary cohorts
    cohorts_path = get_path(COHORTS)
    write_parquet_atomic(cohorts, cohorts_path)

    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"  Signals: {len(cohorts)}")
    print(f"  Cohorts: {cohorts['cohort_id'].n_unique()}")
    print(f"  Output: {cohorts_path}")


if __name__ == "__main__":
    main()
