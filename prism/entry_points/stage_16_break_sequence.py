"""
Stage 16: Break Sequence Entry Point
====================================

Computes propagation order of structural breaks per cohort.

Given a reference index (e.g., fault injection at I=20), finds the FIRST break
per signal after that index and ranks signals by detection latency.

This reveals:
- Which signal breaks first (closest to root cause)
- Propagation path through the system
- Cascade speed (gap between first and last break)

Inputs:
    - breaks.parquet (from stage_00)

Output:
    - break_sequence.parquet

PRISM computes propagation order. ORTHON interprets as fault propagation paths.
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
from typing import Optional


def run(
    breaks_path: str,
    output_path: str = "break_sequence.parquet",
    reference_index: Optional[int] = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute break propagation sequence per cohort.

    Args:
        breaks_path: Path to breaks.parquet
        output_path: Output path for break_sequence.parquet
        reference_index: Sample index to measure from (e.g., 20 for TEP fault injection).
                        If None, uses I=0 (first break overall per cohort).
        verbose: Print progress

    Returns:
        break_sequence DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 16: BREAK SEQUENCE")
        print("Propagation order of structural breaks")
        print("=" * 70)

    breaks = pl.read_parquet(breaks_path)

    if verbose:
        print(f"Loaded breaks: {breaks.shape}")

    has_cohort = 'cohort' in breaks.columns

    if not has_cohort:
        if verbose:
            print("Warning: breaks.parquet has no cohort column. Creating synthetic cohort='all'")
        breaks = breaks.with_columns(pl.lit('all').alias('cohort'))

    cohorts = breaks['cohort'].unique().to_list()

    if verbose:
        print(f"Cohorts: {len(cohorts)}")
        if reference_index is not None:
            print(f"Reference index: {reference_index}")
        else:
            print("Reference index: 0 (first break overall)")

    results = []

    for cohort in cohorts:
        cohort_breaks = breaks.filter(pl.col('cohort') == cohort)

        # Filter to breaks after reference_index
        ref_idx = reference_index if reference_index is not None else 0
        if ref_idx > 0:
            cohort_breaks = cohort_breaks.filter(pl.col('I') > ref_idx)

        if len(cohort_breaks) == 0:
            continue

        # Find first break per signal (sorted by I)
        first_breaks = (
            cohort_breaks
            .sort('I')
            .group_by('signal_id')
            .first()
            .sort('I')
        )

        if len(first_breaks) == 0:
            continue

        # Compute propagation metrics
        first_I = first_breaks['I'].min()

        for rank, row in enumerate(first_breaks.iter_rows(named=True), start=1):
            results.append({
                'cohort': cohort,
                'signal_id': row['signal_id'],
                'first_break_I': row['I'],
                'detection_latency': row['I'] - ref_idx,
                'propagation_rank': rank,
                'cascade_delay': row['I'] - first_I,
                'magnitude': row['magnitude'],
                'direction': row['direction'],
                'snr': row['snr'],
                'reference_index': ref_idx,
            })

    # Build DataFrame
    if results:
        result = pl.DataFrame(results, schema={
            'cohort': pl.Utf8,
            'signal_id': pl.Utf8,
            'first_break_I': pl.UInt32,
            'detection_latency': pl.Int64,
            'propagation_rank': pl.UInt32,
            'cascade_delay': pl.Int64,
            'magnitude': pl.Float64,
            'direction': pl.Int8,
            'snr': pl.Float64,
            'reference_index': pl.Int64,
        })
    else:
        result = pl.DataFrame(schema={
            'cohort': pl.Utf8,
            'signal_id': pl.Utf8,
            'first_break_I': pl.UInt32,
            'detection_latency': pl.Int64,
            'propagation_rank': pl.UInt32,
            'cascade_delay': pl.Int64,
            'magnitude': pl.Float64,
            'direction': pl.Int8,
            'snr': pl.Float64,
            'reference_index': pl.Int64,
        })

    result.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {result.shape}")

        if len(result) > 0:
            # Summary: which signals break first most often?
            first_breakers = (
                result
                .filter(pl.col('propagation_rank') == 1)
                .group_by('signal_id')
                .agg(pl.len().alias('n_first'))
                .sort('n_first', descending=True)
            )
            print("\nSignals that break FIRST most often (potential root cause indicators):")
            for row in first_breakers.head(10).iter_rows(named=True):
                print(f"  {row['signal_id']}: {row['n_first']} cohorts")

            # Cascade speed stats
            cascade = result.filter(pl.col('cascade_delay') > 0)
            if len(cascade) > 0:
                print(f"\nCascade delay stats:")
                print(f"  Mean: {cascade['cascade_delay'].mean():.1f} samples")
                print(f"  Max:  {cascade['cascade_delay'].max()} samples")

        print()
        print("─" * 50)
        print(f"✓ {Path(output_path).absolute()}")
        print("─" * 50)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 16: Break Sequence (Propagation Order)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes propagation order of structural breaks per cohort.

For each cohort, finds the FIRST break per signal after reference_index,
then ranks signals by detection latency.

Output schema (break_sequence.parquet):
  cohort, signal_id, first_break_I, detection_latency,
  propagation_rank, cascade_delay, magnitude, direction, snr, reference_index

Example:
  python -m prism.entry_points.stage_16_break_sequence \\
      breaks.parquet -o break_sequence.parquet --reference-index 20
"""
    )
    parser.add_argument('breaks', help='Path to breaks.parquet')
    parser.add_argument('-o', '--output', default='break_sequence.parquet',
                        help='Output path (default: break_sequence.parquet)')
    parser.add_argument('--reference-index', type=int, default=None,
                        help='Reference index (e.g., 20 for fault injection). Default: 0')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.breaks,
        args.output,
        reference_index=args.reference_index,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
