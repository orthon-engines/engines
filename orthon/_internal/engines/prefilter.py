"""
Laplacian Pre-Filter
====================

Use Laplacian on raw observations to quickly identify
which signals are worth running engines on.

O(n) instead of O(n × engines × 2)

Pipeline:
---------
observations → prefilter.py → filtered_signals → signal_vector.py

Usage:
------
    python -m prism.entry_points.prefilter
"""

import argparse
import polars as pl
import numpy as np
from pathlib import Path
from typing import Tuple

from orthon._internal.db.parquet_store import get_path, get_data_root, ensure_directory, OBSERVATIONS
from orthon._internal.db.polars_io import write_parquet_atomic


def laplacian_prefilter(
    observations: pl.DataFrame,
    signal_col: str = 'signal_id',
    date_col: str = 'obs_date',
    value_col: str = 'value',
    min_variance: float = 0.001,
    min_curvature: float = 0.0001,
    verbose: bool = True,
) -> Tuple[list, list]:
    """
    Fast filter using Laplacian on raw observations.

    Returns:
        (keep_signals, skip_signals)
    """

    if verbose:
        print("=" * 70)
        print("LAPLACIAN PRE-FILTER")
        print("=" * 70)

    # Compute Laplacian per signal
    result = (
        observations
        .sort([signal_col, date_col])
        .with_columns([
            # Raw Laplacian: x(t+1) - 2*x(t) + x(t-1)
            (
                pl.col(value_col).shift(-1).over(signal_col) -
                2 * pl.col(value_col) +
                pl.col(value_col).shift(1).over(signal_col)
            ).alias('laplacian'),

            # First difference (velocity)
            (pl.col(value_col) - pl.col(value_col).shift(1).over(signal_col))
                .alias('velocity'),
        ])
    )

    # Aggregate stats per signal
    stats = (
        result
        .group_by(signal_col)
        .agg([
            # Value stats
            pl.col(value_col).std().alias('value_std'),
            pl.col(value_col).mean().alias('value_mean'),

            # Laplacian stats (curvature)
            pl.col('laplacian').std().alias('laplacian_std'),
            pl.col('laplacian').mean().abs().alias('laplacian_mean_abs'),
            pl.col('laplacian').abs().mean().alias('laplacian_abs_mean'),

            # Velocity stats
            pl.col('velocity').std().alias('velocity_std'),

            # Count
            pl.col(value_col).count().alias('n_obs'),
        ])
        .with_columns([
            # Coefficient of variation (normalized variance)
            (pl.col('value_std') / (pl.col('value_mean').abs() + 1e-10))
                .alias('cv'),

            # Relative curvature
            (pl.col('laplacian_std') / (pl.col('value_std') + 1e-10))
                .alias('relative_curvature'),
        ])
    )

    # Filter criteria
    keep = stats.filter(
        (pl.col('cv') > min_variance) &  # Has variance
        (pl.col('relative_curvature') > min_curvature) &  # Has curvature
        (pl.col('n_obs') >= 100)  # Enough data
    )[signal_col].to_list()

    skip = stats.filter(
        (pl.col('cv') <= min_variance) |
        (pl.col('relative_curvature') <= min_curvature) |
        (pl.col('n_obs') < 100)
    )[signal_col].to_list()

    if verbose:
        print(f"\n  Total signals: {len(keep) + len(skip)}")
        print(f"  Keep (has signal): {len(keep)}")
        print(f"  Skip (flat/constant): {len(skip)}")

        if skip:
            print(f"\n  Skipped signals:")
            for ind in skip[:10]:
                row = stats.filter(pl.col(signal_col) == ind).to_dicts()[0]
                print(f"    {ind}: CV={row['cv']:.4f}, Curv={row['relative_curvature']:.4f}")
            if len(skip) > 10:
                print(f"    ... and {len(skip) - 10} more")

    return keep, skip


def identify_duplicates(
    observations: pl.DataFrame,
    signal_col: str = 'signal_id',
    date_col: str = 'obs_date',
    value_col: str = 'value',
    correlation_threshold: float = 0.99,
    verbose: bool = True,
) -> dict:
    """
    Use Laplacian correlation to find near-duplicate signals.

    If two signals have nearly identical Laplacian patterns,
    they're measuring the same thing.

    Returns:
        {signal: [list of duplicates]}
    """

    if verbose:
        print("=" * 70)
        print("DUPLICATE DETECTION (Laplacian Correlation)")
        print("=" * 70)

    # Compute Laplacian per signal
    with_lap = (
        observations
        .sort([signal_col, date_col])
        .with_columns([
            (
                pl.col(value_col).shift(-1).over(signal_col) -
                2 * pl.col(value_col) +
                pl.col(value_col).shift(1).over(signal_col)
            ).alias('laplacian'),
        ])
    )

    # Pivot to wide format for correlation
    # Each column = signal's Laplacian series
    signals = with_lap[signal_col].unique().to_list()

    # Build correlation matrix via pairwise comparison
    # (More efficient methods exist, but this is clear)

    duplicates = {}
    checked = set()

    for i, ind_a in enumerate(signals):
        if ind_a in checked:
            continue

        series_a = (
            with_lap
            .filter(pl.col(signal_col) == ind_a)
            .select([date_col, 'laplacian'])
            .rename({'laplacian': 'lap_a'})
        )

        dups = []

        for ind_b in signals[i+1:]:
            if ind_b in checked:
                continue

            series_b = (
                with_lap
                .filter(pl.col(signal_col) == ind_b)
                .select([date_col, 'laplacian'])
                .rename({'laplacian': 'lap_b'})
            )

            # Join on date
            joined = series_a.join(series_b, on=date_col, how='inner')

            if len(joined) < 100:
                continue

            # Compute correlation
            corr = joined.select(
                pl.corr('lap_a', 'lap_b')
            ).item()

            if corr is not None and abs(corr) > correlation_threshold:
                dups.append((ind_b, corr))
                checked.add(ind_b)

        if dups:
            duplicates[ind_a] = dups

    if verbose:
        print(f"\n  Duplicate groups found: {len(duplicates)}")
        for ind, dups in list(duplicates.items())[:5]:
            print(f"\n  {ind}:")
            for dup, corr in dups:
                print(f"    ≈ {dup} (r={corr:.4f})")

    return duplicates


def smart_filter(
    observations: pl.DataFrame,
    signal_col: str = 'signal_id',
    date_col: str = 'obs_date',
    value_col: str = 'value',
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Combined smart filter:
    1. Remove flat/constant signals (Laplacian pre-filter)
    2. Remove near-duplicates (keep one representative)

    Returns filtered observations ready for engine computation.
    """

    if verbose:
        print("=" * 70)
        print("SMART FILTER (Laplacian-based)")
        print("=" * 70)
        print(f"\n  Input: {observations[signal_col].n_unique()} signals")

    # Step 1: Remove flat signals
    keep, skip_flat = laplacian_prefilter(
        observations,
        signal_col,
        date_col,
        value_col,
        verbose=verbose,
    )

    filtered = observations.filter(pl.col(signal_col).is_in(keep))

    # Step 2: Remove duplicates
    duplicates = identify_duplicates(
        filtered,
        signal_col,
        date_col,
        value_col,
        verbose=verbose,
    )

    # Get all duplicate signals (keep the first, remove the rest)
    remove_dups = set()
    for ind, dups in duplicates.items():
        for dup, _ in dups:
            remove_dups.add(dup)

    final = filtered.filter(~pl.col(signal_col).is_in(remove_dups))

    if verbose:
        print(f"\n  Final: {final[signal_col].n_unique()} signals")
        print(f"  Removed: {observations[signal_col].n_unique() - final[signal_col].n_unique()}")
        print(f"    - Flat/constant: {len(skip_flat)}")
        print(f"    - Duplicates: {len(remove_dups)}")

    return final


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PRISM Laplacian Pre-Filter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
O(n) pre-filter before running O(n × engines × 2) engine computation.

Uses Laplacian on raw observations to:
1. Skip flat/constant signals (no signal)
2. Identify near-duplicate signals (same Laplacian pattern)

  python -m prism.entry_points.prefilter
        """
    )
    parser.add_argument(
        '--min-variance',
        type=float,
        default=0.001,
        help='Minimum coefficient of variation (default: 0.001)'
    )
    parser.add_argument(
        '--min-curvature',
        type=float,
        default=0.0001,
        help='Minimum relative curvature (default: 0.0001)'
    )
    parser.add_argument(
        '--dup-threshold',
        type=float,
        default=0.99,
        help='Correlation threshold for duplicates (default: 0.99)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output'
    )

    args = parser.parse_args()
    ensure_directory()

    # Load observations using lazy scan (enables streaming for large files)
    obs_path = get_path(OBSERVATIONS)
    if not args.quiet:
        print(f"Loading: {obs_path}")

    # Only load columns needed for prefilter (reduces memory)
    observations = (
        pl.scan_parquet(obs_path)
        .select(['signal_id', 'timestamp', 'value'])
        .collect()
    )

    # Run smart filter
    filtered = smart_filter(
        observations,
        verbose=not args.quiet
    )

    # Save filtered signal list
    keep_signals = filtered['signal_id'].unique().to_list()

    filter_df = pl.DataFrame({
        'signal_id': keep_signals,
        'status': ['keep'] * len(keep_signals)
    })

    # prefilter is a config file, not one of the 5 core files
    output_path = get_data_root() / "prefilter.parquet"
    write_parquet_atomic(filter_df, output_path)

    if not args.quiet:
        print(f"\nSaved: {output_path}")
        print(f"Signals to process: {len(keep_signals)}")


if __name__ == '__main__':
    main()
