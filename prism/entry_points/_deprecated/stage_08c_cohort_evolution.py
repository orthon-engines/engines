"""
Stage 08c: Cohort Evolution Entry Point
=======================================

Pure orchestration - tracks how cohorts evolve over time.

Inputs:
    - state_vector.parquet
    - cohort_membership.parquet (optional)

Output:
    - cohort_evolution.parquet

Computes cohort-level dynamics:
    - Cohort centroid trajectory
    - Cohort dispersion over time
    - Inter-cohort distances
"""

import argparse
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from itertools import combinations


def run(
    state_vector_path: str,
    cohort_membership_path: Optional[str] = None,
    output_path: str = "cohort_evolution.parquet",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Track cohort evolution over time.

    Args:
        state_vector_path: Path to state_vector.parquet
        cohort_membership_path: Path to cohort_membership.parquet (optional)
        output_path: Output path for cohort_evolution.parquet
        verbose: Print progress

    Returns:
        Cohort evolution DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 08c: COHORT EVOLUTION")
        print("Tracking cohort dynamics over time")
        print("=" * 70)

    # Load state vector
    sv = pl.read_parquet(state_vector_path)

    if verbose:
        print(f"Loaded state_vector: {sv.shape}")

    # Check for cohort column
    has_cohort = 'cohort' in sv.columns

    if not has_cohort:
        if verbose:
            print("No cohort column, computing global evolution")

    # Get feature columns
    exclude_cols = {'I', 'cohort', 'unit_id', 'window_idx', 'n_signals'}
    feature_cols = [c for c in sv.columns if c not in exclude_cols
                    and sv[c].dtype in [pl.Float64, pl.Float32]]

    if verbose:
        print(f"Feature columns: {len(feature_cols)}")

    results = []

    if has_cohort:
        cohorts = sv['cohort'].unique().to_list()
        time_points = sorted(sv['I'].unique().to_list())

        if verbose:
            print(f"Cohorts: {len(cohorts)}")
            print(f"Time points: {len(time_points)}")

        # Track each cohort over time
        for cohort in cohorts:
            cohort_data = sv.filter(pl.col('cohort') == cohort).sort('I')

            if len(cohort_data) == 0:
                continue

            # Get feature matrix
            X = np.column_stack([
                cohort_data[c].to_numpy() for c in feature_cols
            ])
            X = np.nan_to_num(X, nan=0.0)
            time_vals = cohort_data['I'].to_numpy()

            # Compute centroid at each time
            for i, I in enumerate(time_vals):
                centroid = X[i]
                centroid_norm = np.linalg.norm(centroid)

                # Compute velocity if not first point
                velocity = np.nan
                if i > 0:
                    prev_centroid = X[i-1]
                    velocity = np.linalg.norm(centroid - prev_centroid)

                results.append({
                    'cohort': cohort,
                    'I': int(I),
                    'centroid_norm': float(centroid_norm),
                    'velocity': float(velocity),
                })

        # Compute inter-cohort distances at each time point
        if len(cohorts) > 1:
            for I in time_points:
                time_data = sv.filter(pl.col('I') == I)

                cohort_centroids = {}
                for cohort in cohorts:
                    c_data = time_data.filter(pl.col('cohort') == cohort)
                    if len(c_data) > 0:
                        centroid = np.array([c_data[c].mean() for c in feature_cols])
                        cohort_centroids[cohort] = centroid

                # Pairwise distances
                for c1, c2 in combinations(cohort_centroids.keys(), 2):
                    dist = np.linalg.norm(cohort_centroids[c1] - cohort_centroids[c2])
                    results.append({
                        'cohort': f"{c1}_to_{c2}",
                        'I': int(I),
                        'centroid_norm': np.nan,
                        'velocity': np.nan,
                        'inter_cohort_distance': float(dist),
                    })
    else:
        # Global evolution (single cohort)
        time_points = sorted(sv['I'].unique().to_list())

        for i, I in enumerate(time_points):
            time_data = sv.filter(pl.col('I') == I)

            # Compute global centroid
            X = np.column_stack([
                time_data[c].to_numpy() for c in feature_cols
            ])
            X = np.nan_to_num(X, nan=0.0)

            centroid = np.mean(X, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            dispersion = np.mean(np.linalg.norm(X - centroid, axis=1))

            results.append({
                'cohort': 'global',
                'I': int(I),
                'centroid_norm': float(centroid_norm),
                'dispersion': float(dispersion),
            })

    # Build DataFrame
    df = pl.DataFrame(results) if results else pl.DataFrame()

    # Write output
    if len(df) > 0:
        df.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {df.shape}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Stage 08c: Cohort Evolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tracks cohort-level dynamics:
  - Cohort centroid trajectory
  - Cohort velocity over time
  - Inter-cohort distances

Example:
  python -m prism.entry_points.stage_08c_cohort_evolution \\
      state_vector.parquet -o cohort_evolution.parquet
"""
    )
    parser.add_argument('state_vector', help='Path to state_vector.parquet')
    parser.add_argument('--cohort-membership', help='Path to cohort_membership.parquet')
    parser.add_argument('-o', '--output', default='cohort_evolution.parquet',
                        help='Output path (default: cohort_evolution.parquet)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.state_vector,
        args.cohort_membership,
        args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
