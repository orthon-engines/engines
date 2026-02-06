"""
Stage 10: Information Flow Entry Point
======================================

Pure orchestration - calls engines/pairwise/causality.py for computation.

Inputs:
    - observations.parquet

Output:
    - information_flow.parquet

Computes information-theoretic causality between signals:
    - Transfer entropy
    - Granger causality
    - Mutual information
"""

import argparse
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from itertools import combinations

from prism.engines.pairwise.causality import compute_all as compute_causality


def run(
    observations_path: str,
    output_path: str = "information_flow.parquet",
    signal_column: str = 'signal_id',
    value_column: str = 'value',
    index_column: str = 'I',
    max_pairs: int = 500,
    min_samples: int = 100,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run information flow computation for all signal pairs.

    Args:
        observations_path: Path to observations.parquet
        output_path: Output path for information_flow.parquet
        signal_column: Column with signal IDs
        value_column: Column with values
        index_column: Column with time index
        max_pairs: Maximum number of pairs to compute
        min_samples: Minimum samples required
        verbose: Print progress

    Returns:
        Information flow DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 10: INFORMATION FLOW")
        print("Pairwise causality metrics")
        print("=" * 70)

    # Load observations
    obs = pl.read_parquet(observations_path)

    # Pivot to wide format
    signals = obs[signal_column].unique().to_list()
    n_signals = len(signals)

    if verbose:
        print(f"Signals: {n_signals}")
        n_pairs = n_signals * (n_signals - 1) // 2
        print(f"Potential pairs: {n_pairs}")

    # Get time series per signal
    signal_data = {}
    for signal in signals:
        sig = obs.filter(pl.col(signal_column) == signal).sort(index_column)
        values = sig[value_column].to_numpy()
        values = values[~np.isnan(values)]
        if len(values) >= min_samples:
            signal_data[signal] = values

    available_signals = list(signal_data.keys())
    pairs = list(combinations(available_signals, 2))

    if len(pairs) > max_pairs:
        if verbose:
            print(f"Limiting to {max_pairs} pairs (of {len(pairs)})")
        pairs = pairs[:max_pairs]

    results = []

    for i, (sig_a, sig_b) in enumerate(pairs):
        x = signal_data[sig_a]
        y = signal_data[sig_b]

        # Align lengths
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]

        try:
            result = compute_causality(x, y)
            result['signal_a'] = sig_a
            result['signal_b'] = sig_b
            results.append(result)
        except Exception:
            pass

        if verbose and (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(pairs)} pairs...")

    # Build DataFrame
    df = pl.DataFrame(results) if results else pl.DataFrame()

    if len(df) > 0:
        df.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {df.shape}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Stage 10: Information Flow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes pairwise causality metrics:
  - Transfer entropy
  - Granger causality
  - Mutual information

Example:
  python -m prism.entry_points.stage_10_information_flow \\
      observations.parquet -o information_flow.parquet
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('-o', '--output', default='information_flow.parquet',
                        help='Output path (default: information_flow.parquet)')
    parser.add_argument('--max-pairs', type=int, default=500,
                        help='Maximum pairs to compute (default: 500)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        args.output,
        max_pairs=args.max_pairs,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
