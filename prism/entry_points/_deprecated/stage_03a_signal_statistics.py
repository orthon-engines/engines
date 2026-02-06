"""
Stage 03a: Signal Statistics Entry Point
========================================

Pure orchestration - calls engines/signal/statistics.py for computation.

Inputs:
    - observations.parquet

Output:
    - signal_statistics.parquet

Computes per-signal statistical properties:
    - kurtosis
    - skewness
    - crest_factor
"""

import argparse
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any

from prism.engines.signal.statistics import compute as compute_statistics


def run(
    observations_path: str,
    output_path: str = "signal_statistics.parquet",
    signal_column: str = 'signal_id',
    value_column: str = 'value',
    index_column: str = 'I',
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run statistics computation for all signals.

    Args:
        observations_path: Path to observations.parquet
        output_path: Output path for signal_statistics.parquet
        signal_column: Column with signal IDs
        value_column: Column with values
        index_column: Column with time index
        verbose: Print progress

    Returns:
        Signal statistics DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 03a: SIGNAL STATISTICS")
        print("Per-signal statistical properties")
        print("=" * 70)

    # Load observations
    obs = pl.read_parquet(observations_path)

    signals = obs[signal_column].unique().to_list()
    n_signals = len(signals)

    if verbose:
        print(f"Processing {n_signals} signals...")

    results = []

    for i, signal_id in enumerate(signals):
        sig_data = obs.filter(pl.col(signal_column) == signal_id).sort(index_column)
        values = sig_data[value_column].to_numpy()
        values = values[~np.isnan(values)]

        # Get cohort if present
        cohort = None
        if 'cohort' in sig_data.columns:
            cohort = sig_data['cohort'][0]

        # Compute statistics
        stats = compute_statistics(values)

        result = {
            'signal_id': signal_id,
            'n_samples': len(values),
            **stats
        }

        if cohort is not None:
            result['cohort'] = cohort

        results.append(result)

        if verbose and (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{n_signals} signals...")

    # Build DataFrame
    df = pl.DataFrame(results) if results else pl.DataFrame()

    # Write output
    if len(df) > 0:
        df.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {df.shape}")
        if 'kurtosis' in df.columns:
            mean_kurt = df['kurtosis'].mean()
            print(f"Mean kurtosis: {mean_kurt:.3f}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Stage 03a: Signal Statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes per-signal statistics:
  - kurtosis (tail heaviness)
  - skewness (asymmetry)
  - crest_factor (peak-to-RMS ratio)

Example:
  python -m prism.entry_points.stage_03a_signal_statistics \\
      observations.parquet -o signal_statistics.parquet
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('-o', '--output', default='signal_statistics.parquet',
                        help='Output path (default: signal_statistics.parquet)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
