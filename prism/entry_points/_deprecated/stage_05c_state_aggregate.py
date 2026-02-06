"""
Stage 05c: State Aggregate Entry Point
======================================

Pure orchestration - aggregates state vectors across cohorts/time.

Inputs:
    - state_vector.parquet

Output:
    - state_aggregate.parquet

Computes aggregate statistics of state vectors:
    - Per-cohort means and variances
    - Temporal evolution statistics
    - Cross-cohort comparisons
"""

import argparse
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any


def run(
    state_vector_path: str,
    output_path: str = "state_aggregate.parquet",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute aggregate statistics of state vectors.

    Args:
        state_vector_path: Path to state_vector.parquet
        output_path: Output path for state_aggregate.parquet
        verbose: Print progress

    Returns:
        State aggregate DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 05c: STATE AGGREGATE")
        print("Aggregate state vector statistics")
        print("=" * 70)

    # Load state vector
    sv = pl.read_parquet(state_vector_path)

    if verbose:
        print(f"Loaded state_vector: {sv.shape}")

    # Identify centroid columns
    exclude_cols = {'I', 'cohort', 'unit_id', 'window_idx', 'n_signals'}
    centroid_cols = [c for c in sv.columns if c not in exclude_cols
                     and sv[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

    if verbose:
        print(f"Centroid columns: {len(centroid_cols)}")

    results = []

    # Check if we have cohort grouping
    has_cohort = 'cohort' in sv.columns

    if has_cohort:
        cohorts = sv['cohort'].unique().to_list()
        if verbose:
            print(f"Cohorts: {len(cohorts)}")

        for cohort in cohorts:
            cohort_data = sv.filter(pl.col('cohort') == cohort)

            for col in centroid_cols:
                values = cohort_data[col].to_numpy()
                values = values[~np.isnan(values)]

                if len(values) < 2:
                    continue

                # First difference for dynamics
                diff = np.diff(values)

                results.append({
                    'cohort': cohort,
                    'component': col,
                    'n_windows': len(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'range': float(np.max(values) - np.min(values)),
                    'cv': float(np.std(values) / (np.abs(np.mean(values)) + 1e-10)),
                    # Temporal dynamics
                    'mean_velocity': float(np.mean(diff)) if len(diff) > 0 else np.nan,
                    'std_velocity': float(np.std(diff)) if len(diff) > 0 else np.nan,
                    'max_velocity': float(np.max(np.abs(diff))) if len(diff) > 0 else np.nan,
                })
    else:
        # Global aggregation (no cohort)
        for col in centroid_cols:
            values = sv[col].to_numpy()
            values = values[~np.isnan(values)]

            if len(values) < 2:
                continue

            diff = np.diff(values)

            results.append({
                'cohort': 'global',
                'component': col,
                'n_windows': len(values),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'range': float(np.max(values) - np.min(values)),
                'cv': float(np.std(values) / (np.abs(np.mean(values)) + 1e-10)),
                'mean_velocity': float(np.mean(diff)) if len(diff) > 0 else np.nan,
                'std_velocity': float(np.std(diff)) if len(diff) > 0 else np.nan,
                'max_velocity': float(np.max(np.abs(diff))) if len(diff) > 0 else np.nan,
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
        description="Stage 05c: State Aggregate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes aggregate statistics of state vectors:
  - Per-cohort means and variances
  - Temporal dynamics (velocity)
  - Range and coefficient of variation

Example:
  python -m prism.entry_points.stage_05c_state_aggregate \\
      state_vector.parquet -o state_aggregate.parquet
"""
    )
    parser.add_argument('state_vector', help='Path to state_vector.parquet')
    parser.add_argument('-o', '--output', default='state_aggregate.parquet',
                        help='Output path (default: state_aggregate.parquet)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.state_vector,
        args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
