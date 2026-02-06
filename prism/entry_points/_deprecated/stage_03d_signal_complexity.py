"""
Stage 03d: Signal Complexity Entry Point
========================================

Pure orchestration - calls engines/signal/complexity.py for computation.

Inputs:
    - observations.parquet

Output:
    - signal_complexity.parquet

Computes per-signal complexity/entropy properties:
    - sample_entropy
    - permutation_entropy
    - approximate_entropy
"""

import argparse
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any

from prism.engines.signal.complexity import compute as compute_complexity


def run(
    observations_path: str,
    output_path: str = "signal_complexity.parquet",
    signal_column: str = 'signal_id',
    value_column: str = 'value',
    index_column: str = 'I',
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run complexity computation for all signals.

    Args:
        observations_path: Path to observations.parquet
        output_path: Output path for signal_complexity.parquet
        signal_column: Column with signal IDs
        value_column: Column with values
        index_column: Column with time index
        verbose: Print progress

    Returns:
        Signal complexity DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 03d: SIGNAL COMPLEXITY")
        print("Per-signal entropy/complexity properties")
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

        # Compute complexity metrics
        complexity_metrics = compute_complexity(values)

        result = {
            'signal_id': signal_id,
            'n_samples': len(values),
            **complexity_metrics,
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
        if 'sample_entropy' in df.columns:
            mean_se = df['sample_entropy'].mean()
            print(f"Mean sample entropy: {mean_se:.3f}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Stage 03d: Signal Complexity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes per-signal complexity properties:
  - sample_entropy (regularity measure)
  - permutation_entropy (ordinal pattern complexity)
  - approximate_entropy (similar to sample entropy)

High entropy = more random/complex
Low entropy = more regular/predictable

Example:
  python -m prism.entry_points.stage_03d_signal_complexity \\
      observations.parquet -o signal_complexity.parquet
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('-o', '--output', default='signal_complexity.parquet',
                        help='Output path (default: signal_complexity.parquet)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
