"""
Stage 08a: Cohort Discovery Entry Point
=======================================

Pure orchestration - discovers natural cohorts/clusters in signal space.

Inputs:
    - signal_vector.parquet

Output:
    - cohort_discovery.parquet

Uses clustering to identify natural groupings of signals based on
their feature vectors.
"""

import argparse
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


def run(
    signal_vector_path: str,
    output_path: str = "cohort_discovery.parquet",
    n_clusters: Optional[int] = None,
    min_cluster_size: int = 2,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Discover natural cohorts via clustering.

    Args:
        signal_vector_path: Path to signal_vector.parquet
        output_path: Output path for cohort_discovery.parquet
        n_clusters: Number of clusters (None=auto-detect)
        min_cluster_size: Minimum cluster size
        verbose: Print progress

    Returns:
        Cohort discovery DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 08a: COHORT DISCOVERY")
        print("Clustering signals into natural cohorts")
        print("=" * 70)

    # Load signal vector
    sv = pl.read_parquet(signal_vector_path)

    if verbose:
        print(f"Loaded signal_vector: {sv.shape}")

    # Get unique signals and their feature vectors
    # Group by signal_id, taking mean of each feature
    exclude_cols = {'I', 'signal_id', 'cohort', 'unit_id', 'window_idx'}
    feature_cols = [c for c in sv.columns if c not in exclude_cols
                    and sv[c].dtype in [pl.Float64, pl.Float32]]

    if len(feature_cols) == 0:
        if verbose:
            print("No numeric features found for clustering")
        return pl.DataFrame()

    # Aggregate features per signal
    signal_features = sv.group_by('signal_id').agg([
        pl.col(c).mean().alias(c) for c in feature_cols
    ])

    signals = signal_features['signal_id'].to_list()
    n_signals = len(signals)

    if verbose:
        print(f"Signals: {n_signals}")
        print(f"Features: {len(feature_cols)}")

    # Build feature matrix
    X = np.column_stack([
        signal_features[c].to_numpy() for c in feature_cols
    ])

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)

    # Standardize features
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-10
    X_scaled = (X - X_mean) / X_std

    # Clustering
    try:
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.metrics import silhouette_score

        if n_clusters is None:
            # Auto-detect optimal clusters (2 to min(10, n_signals//2))
            max_k = min(10, n_signals // 2)
            if max_k < 2:
                max_k = 2

            best_k = 2
            best_score = -1

            for k in range(2, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)

                if len(set(labels)) > 1:
                    score = silhouette_score(X_scaled, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k

            n_clusters = best_k
            if verbose:
                print(f"Auto-detected optimal clusters: {n_clusters} (silhouette: {best_score:.3f})")

        # Final clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        centers = kmeans.cluster_centers_

        # Compute distance to cluster center
        distances = np.linalg.norm(X_scaled - centers[labels], axis=1)

    except ImportError:
        if verbose:
            print("sklearn not available, using basic clustering")
        # Fallback: simple distance-based grouping
        labels = np.zeros(n_signals, dtype=int)
        distances = np.zeros(n_signals)
        n_clusters = 1

    # Build results
    results = []
    for i, signal_id in enumerate(signals):
        results.append({
            'signal_id': signal_id,
            'discovered_cohort': int(labels[i]),
            'distance_to_center': float(distances[i]),
        })

    df = pl.DataFrame(results)

    # Add cohort statistics
    cohort_stats = []
    for cohort_id in range(n_clusters):
        cohort_mask = labels == cohort_id
        cohort_size = np.sum(cohort_mask)

        if cohort_size > 0:
            cohort_distances = distances[cohort_mask]
            cohort_stats.append({
                'cohort_id': cohort_id,
                'size': int(cohort_size),
                'mean_distance': float(np.mean(cohort_distances)),
                'max_distance': float(np.max(cohort_distances)),
                'compactness': float(1.0 / (np.mean(cohort_distances) + 1e-10)),
            })

    stats_df = pl.DataFrame(cohort_stats) if cohort_stats else pl.DataFrame()

    # Write outputs
    if len(df) > 0:
        df.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {df.shape}")
        print(f"\nCohort sizes:")
        for stat in cohort_stats:
            print(f"  Cohort {stat['cohort_id']}: {stat['size']} signals")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Stage 08a: Cohort Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Discovers natural signal groupings via clustering:
  - Auto-detects optimal number of clusters
  - Computes distance to cluster center
  - Identifies cohort compactness

Example:
  python -m prism.entry_points.stage_08a_cohort_discovery \\
      signal_vector.parquet -o cohort_discovery.parquet
"""
    )
    parser.add_argument('signal_vector', help='Path to signal_vector.parquet')
    parser.add_argument('-o', '--output', default='cohort_discovery.parquet',
                        help='Output path (default: cohort_discovery.parquet)')
    parser.add_argument('--n-clusters', type=int, help='Number of clusters (auto if not specified)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.signal_vector,
        args.output,
        n_clusters=args.n_clusters,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
