"""
Stage 18: Segment Comparison Entry Point
========================================

Splits data at specified index boundaries, computes geometry/dynamics
independently per segment, then computes deltas between segments.

Manifest config:
    segments:
      - name: pre
        range: [0, 20]
      - name: post
        range: [21, null]   # null = end of data

Inputs:
    - signal_vector.parquet
    - state_geometry.parquet (optional, for comparison)

Output:
    - segment_comparison.parquet

Each segment gets independent eigendecomp, then deltas computed between
consecutive segments. This reveals how geometric structure changes
across boundaries (faults, elections, regime changes).
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Dict, Any, Optional

from prism.engines.state.eigendecomp import compute as compute_eigendecomp


def run(
    observations_path: str,
    output_path: str = "segment_comparison.parquet",
    segments: List[Dict[str, Any]] = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute per-segment geometry and deltas.

    Args:
        observations_path: Path to observations.parquet
        output_path: Output path for segment_comparison.parquet
        segments: List of segment definitions:
            [{'name': 'pre', 'range': [0, 20]},
             {'name': 'post', 'range': [21, None]}]
        verbose: Print progress

    Returns:
        segment_comparison DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 18: SEGMENT COMPARISON")
        print("Per-segment geometry with cross-segment deltas")
        print("=" * 70)

    if segments is None or len(segments) < 2:
        if verbose:
            print("Warning: Need at least 2 segments for comparison. Using default split at I=20.")
        segments = [
            {'name': 'pre', 'range': [0, 19]},
            {'name': 'post', 'range': [20, None]},
        ]

    if verbose:
        print(f"Segments: {len(segments)}")
        for seg in segments:
            end = seg['range'][1] if seg['range'][1] is not None else 'end'
            print(f"  {seg['name']}: I in [{seg['range'][0]}, {end}]")

    # Load observations
    obs = pl.read_parquet(observations_path)

    if verbose:
        print(f"\nLoaded observations: {obs.shape}")

    has_cohort = 'cohort' in obs.columns
    cohorts = obs['cohort'].unique().to_list() if has_cohort else ['all']
    signals = obs['signal_id'].unique().to_list()

    if verbose:
        print(f"Cohorts: {len(cohorts)}")
        print(f"Signals: {len(signals)}")

    results = []

    for cohort in cohorts:
        if has_cohort:
            cohort_data = obs.filter(pl.col('cohort') == cohort)
        else:
            cohort_data = obs

        # Get I range for this cohort
        i_max = cohort_data['I'].max()

        cohort_segments = {}

        for seg in segments:
            seg_name = seg['name']
            start_i = seg['range'][0]
            end_i = seg['range'][1] if seg['range'][1] is not None else i_max

            # Filter to segment
            seg_data = cohort_data.filter(
                (pl.col('I') >= start_i) & (pl.col('I') <= end_i)
            )

            if len(seg_data) < 10:
                continue

            # Pivot to wide format: each row = one I, each column = signal
            try:
                wide = seg_data.pivot(
                    values='value',
                    index='I',
                    on='signal_id',
                )
            except Exception:
                continue

            if wide is None or len(wide) < 5:
                continue

            # Get signal columns (exclude I)
            signal_cols = [c for c in wide.columns if c != 'I']
            if len(signal_cols) < 2:
                continue

            # Build matrix (rows = I values, columns = signals)
            matrix = wide.select(signal_cols).to_numpy()

            # Remove rows with NaN
            valid_rows = ~np.isnan(matrix).any(axis=1)
            matrix = matrix[valid_rows]

            if len(matrix) < 5:
                continue

            # Compute eigendecomp
            eigen_result = compute_eigendecomp(matrix)
            eigenvalues = eigen_result.get('eigenvalues', [])

            cohort_segments[seg_name] = {
                'n_samples': len(matrix),
                'n_signals': len(signal_cols),
                'effective_dim': eigen_result.get('effective_dim'),
                'eigenvalue_1': float(eigenvalues[0]) if len(eigenvalues) > 0 and not np.isnan(eigenvalues[0]) else None,
                'eigenvalue_2': float(eigenvalues[1]) if len(eigenvalues) > 1 and not np.isnan(eigenvalues[1]) else None,
                'explained_1': float(eigen_result.get('explained_ratio', [0])[0]) if eigen_result.get('explained_ratio') is not None else None,
                'condition_number': eigen_result.get('condition_number'),
                'spectral_gap': eigen_result.get('ratio_2_1'),
            }

        # Compute deltas between consecutive segments
        seg_names = [s['name'] for s in segments if s['name'] in cohort_segments]

        for i in range(len(seg_names) - 1):
            seg_a = seg_names[i]
            seg_b = seg_names[i + 1]

            stats_a = cohort_segments.get(seg_a, {})
            stats_b = cohort_segments.get(seg_b, {})

            if not stats_a or not stats_b:
                continue

            results.append({
                'cohort': cohort,
                'segment_a': seg_a,
                'segment_b': seg_b,
                'n_samples_a': stats_a.get('n_samples', 0),
                'n_samples_b': stats_b.get('n_samples', 0),
                'eff_dim_a': stats_a.get('effective_dim'),
                'eff_dim_b': stats_b.get('effective_dim'),
                'eff_dim_delta': (stats_b.get('effective_dim', 0) - stats_a.get('effective_dim', 0))
                                 if stats_a.get('effective_dim') and stats_b.get('effective_dim') else None,
                'eigenvalue_1_a': stats_a.get('eigenvalue_1'),
                'eigenvalue_1_b': stats_b.get('eigenvalue_1'),
                'eigenvalue_1_delta': (stats_b.get('eigenvalue_1', 0) - stats_a.get('eigenvalue_1', 0))
                                      if stats_a.get('eigenvalue_1') and stats_b.get('eigenvalue_1') else None,
                'explained_1_a': stats_a.get('explained_1'),
                'explained_1_b': stats_b.get('explained_1'),
                'condition_number_a': stats_a.get('condition_number'),
                'condition_number_b': stats_b.get('condition_number'),
                'spectral_gap_a': stats_a.get('spectral_gap'),
                'spectral_gap_b': stats_b.get('spectral_gap'),
            })

    # Build output
    if results:
        result = pl.DataFrame(results)
    else:
        result = pl.DataFrame(schema={
            'cohort': pl.Utf8,
            'segment_a': pl.Utf8,
            'segment_b': pl.Utf8,
            'n_samples_a': pl.Int64,
            'n_samples_b': pl.Int64,
            'eff_dim_a': pl.Float64,
            'eff_dim_b': pl.Float64,
            'eff_dim_delta': pl.Float64,
            'eigenvalue_1_a': pl.Float64,
            'eigenvalue_1_b': pl.Float64,
            'eigenvalue_1_delta': pl.Float64,
            'explained_1_a': pl.Float64,
            'explained_1_b': pl.Float64,
            'condition_number_a': pl.Float64,
            'condition_number_b': pl.Float64,
            'spectral_gap_a': pl.Float64,
            'spectral_gap_b': pl.Float64,
        })

    result.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {result.shape}")

        if len(result) > 0:
            print("\nEffective dimension changes:")
            valid = result.filter(pl.col('eff_dim_delta').is_not_null())
            if len(valid) > 0:
                collapsed = valid.filter(pl.col('eff_dim_delta') < -0.5)
                expanded = valid.filter(pl.col('eff_dim_delta') > 0.5)
                print(f"  Collapsed (delta < -0.5): {len(collapsed)} cohorts")
                print(f"  Expanded (delta > 0.5):   {len(expanded)} cohorts")
                print(f"  Mean delta: {valid['eff_dim_delta'].mean():.3f}")

        print()
        print("─" * 50)
        print(f"✓ {Path(output_path).absolute()}")
        print("─" * 50)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 18: Segment Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes geometry per segment and deltas between segments.

Default segments (can be overridden via manifest):
  pre:  I in [0, 20]
  post: I in [21, end]

Output schema:
  cohort, segment_a, segment_b,
  eff_dim_a, eff_dim_b, eff_dim_delta,
  eigenvalue_1_a, eigenvalue_1_b, eigenvalue_1_delta,
  ...

Example:
  python -m prism.entry_points.stage_18_segment_comparison \\
      observations.parquet -o segment_comparison.parquet --split-at 20
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('-o', '--output', default='segment_comparison.parquet',
                        help='Output path (default: segment_comparison.parquet)')
    parser.add_argument('--split-at', type=int, default=20,
                        help='I value to split at (default: 20)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    # Build segments from --split-at
    segments = [
        {'name': 'pre', 'range': [0, args.split_at]},
        {'name': 'post', 'range': [args.split_at + 1, None]},
    ]

    run(
        args.observations,
        args.output,
        segments=segments,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
