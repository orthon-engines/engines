"""
Stage 36: Persistent Homology Entry Point
==========================================

Computes topological invariants of the state-space attractor per window.

For each cohort, the state vector traces a trajectory through feature space.
A sliding window over the trajectory gives a point cloud at each time step.
Persistent homology reveals the SHAPE of that attractor:
- β₀ = 1 means one coherent attractor
- β₀ > 1 means the attractor has fragmented
- β₁ = 1 means a dominant cycle (limit cycle)
- β₁ = 0 means no periodic structure

Inputs:
    - state_vector.parquet (centroid trajectory in feature space)

Output:
    - persistent_homology.parquet → 5_evolution/
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
from typing import Optional

from manifold.core.dynamics.persistent_homology import compute
from manifold.io.writer import write_output


# Columns that are trajectory coordinates (feature dimensions)
_META_COLS = {'I', 'n_signals', 'cohort', 'mean_distance', 'max_distance', 'std_distance'}


def run(
    state_vector_path: str,
    data_path: str = ".",
    window: int = 20,
    stride: int = 1,
    max_dim: int = 1,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute persistent homology of state-space trajectory per window.

    For each cohort:
        1. Extract feature columns as trajectory coordinates
        2. Slide a window of `window` I-steps over the trajectory
        3. Compute persistence diagram per window
        4. Record Betti numbers and persistence metrics

    Args:
        state_vector_path: Path to state_vector.parquet
        data_path: Root data directory for output
        window: Number of consecutive I-steps per point cloud
        stride: Step between windows
        max_dim: Maximum homology dimension (1 = H0 + H1)
        verbose: Print progress

    Returns:
        DataFrame with (cohort, I, betti_0, betti_1, persistence metrics)
    """
    if verbose:
        print("=" * 70)
        print("STAGE 36: PERSISTENT HOMOLOGY")
        print("Topology of the state-space attractor")
        print("=" * 70)

    # Load state vector
    sv = pl.read_parquet(state_vector_path)

    if verbose:
        print(f"Loaded state_vector: {sv.shape}")

    # Identify feature columns (everything that isn't metadata)
    feature_cols = [c for c in sv.columns if c not in _META_COLS]
    n_dims = len(feature_cols)

    if verbose:
        print(f"Feature dimensions: {n_dims} ({', '.join(feature_cols[:5])}{'...' if n_dims > 5 else ''})")
        print(f"Window: {window} steps, stride: {stride}")

    has_cohort = 'cohort' in sv.columns

    results = []

    if has_cohort:
        cohorts = sv['cohort'].unique().to_list()
        if verbose:
            print(f"Cohorts: {len(cohorts)}")

        for cohort in cohorts:
            cohort_data = sv.filter(pl.col('cohort') == cohort).sort('I')
            cohort_results = _process_trajectory(
                cohort_data, feature_cols, window, stride, max_dim, cohort
            )
            results.extend(cohort_results)

            if verbose:
                print(f"  {cohort}: {len(cohort_results)} windows")
    else:
        cohort_results = _process_trajectory(
            sv.sort('I'), feature_cols, window, stride, max_dim, 'global'
        )
        results.extend(cohort_results)

        if verbose:
            print(f"  global: {len(cohort_results)} windows")

    # Build DataFrame
    df = pl.from_dicts(results, infer_schema_length=len(results)) if results else pl.DataFrame()

    # Write output
    if len(df) > 0:
        write_output(df, data_path, 'persistent_homology', verbose=verbose)

    if verbose:
        print(f"Shape: {df.shape}")

        if len(df) > 0 and 'betti_0' in df.columns:
            # Summary stats
            b0_valid = df.filter(pl.col('betti_0').is_not_null())
            if len(b0_valid) > 0:
                print(f"\nTopology summary:")
                print(f"  b0 range: [{b0_valid['betti_0'].min()}, {b0_valid['betti_0'].max()}]")
                if 'betti_1' in df.columns:
                    b1_valid = df.filter(pl.col('betti_1').is_not_null())
                    if len(b1_valid) > 0:
                        print(f"  b1 range: [{b1_valid['betti_1'].min()}, {b1_valid['betti_1'].max()}]")

    return df


def _process_trajectory(
    data: pl.DataFrame,
    feature_cols: list,
    window: int,
    stride: int,
    max_dim: int,
    cohort: str,
) -> list:
    """
    Window a trajectory and compute persistence per window.

    Args:
        data: Sorted DataFrame for one cohort
        feature_cols: Columns to use as trajectory coordinates
        window: Window size in I-steps
        stride: Step between windows
        max_dim: Maximum homology dimension
        cohort: Cohort label

    Returns:
        List of result dicts, one per window
    """
    n = len(data)
    if n < window:
        return []

    I_values = data['I'].to_numpy()
    trajectory = data.select(feature_cols).to_numpy()

    results = []

    for start in range(0, n - window + 1, stride):
        end = start + window
        point_cloud = trajectory[start:end]
        I_end = int(I_values[end - 1])  # Window end index (canonical I)

        metrics = compute(point_cloud, min_samples=10, max_dim=max_dim)

        row = {
            'cohort': cohort,
            'I': I_end,
        }
        # Add all topology metrics (exclude engine metadata)
        for k, v in metrics.items():
            if k not in ('subsampled', 'n_points'):
                row[k] = v

        results.append(row)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Stage 36: Persistent Homology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes persistent homology of the state-space attractor.

At each window, the trajectory through feature space forms a point cloud.
Persistent homology reveals:
  β₀ = connected components (fragmentation)
  β₁ = loops (periodic structure)

Example:
  python -m manifold.stages.dynamics.persistent_homology \\
      state_vector.parquet -o persistent_homology.parquet
"""
    )
    parser.add_argument('state_vector', help='Path to state_vector.parquet')
    parser.add_argument('-d', '--data-path', default='.',
                        help='Root data directory (default: .)')
    parser.add_argument('--window', type=int, default=20,
                        help='Window size in I-steps (default: 20)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Step between windows (default: 1)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.state_vector,
        args.data_path,
        window=args.window,
        stride=args.stride,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
