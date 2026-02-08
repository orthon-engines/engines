"""
Stage 10: Information Flow Entry Point
======================================

Pure orchestration - computes causality for pairs flagged by eigenvector gating.

Inputs:
    - observations.parquet (raw time series)
    - signal_pairwise.parquet (pairs with needs_granger flags)

Output:
    - information_flow.parquet

Only computes Granger causality for pairs where needs_granger=True
(high PC co-loading indicates correlation, need Granger for direction).
"""

import argparse
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from engines.manifold.pairwise.causality import compute_all as compute_causality


def run(
    observations_path: str,
    signal_pairwise_path: str,
    output_path: str = "information_flow.parquet",
    signal_column: str = 'signal_id',
    value_column: str = 'value',
    index_column: str = 'I',
    min_samples: int = 100,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run information flow computation for Granger-flagged pairs only.

    Uses eigenvector co-loading from signal_pairwise to gate expensive
    Granger causality computation. Only pairs with needs_granger=True
    are processed.

    Args:
        observations_path: Path to observations.parquet
        signal_pairwise_path: Path to signal_pairwise.parquet (has needs_granger)
        output_path: Output path for information_flow.parquet
        signal_column: Column with signal IDs
        value_column: Column with values
        index_column: Column with time index
        min_samples: Minimum samples required
        verbose: Print progress

    Returns:
        Information flow DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 10: INFORMATION FLOW")
        print("Causality for eigenvector-gated pairs")
        print("=" * 70)

    # Load observations
    obs = pl.read_parquet(observations_path)

    # Load signal_pairwise for Granger gating
    pairwise = pl.read_parquet(signal_pairwise_path)

    if verbose:
        print(f"Observations: {len(obs)} rows")
        print(f"Pairwise: {len(pairwise)} rows")

    # Check for needs_granger column
    if 'needs_granger' not in pairwise.columns:
        if verbose:
            print("Warning: needs_granger column not found, computing all pairs")
        granger_pairs = pairwise.select(['signal_a', 'signal_b']).unique()
    else:
        # Filter to pairs needing Granger
        granger_pairs = pairwise.filter(pl.col('needs_granger') == True).select(['signal_a', 'signal_b']).unique()

    n_granger = len(granger_pairs)
    n_total = pairwise.select(['signal_a', 'signal_b']).unique().height

    if verbose:
        print(f"Pairs needing Granger: {n_granger}/{n_total} ({100*n_granger/max(n_total,1):.1f}%)")

    if n_granger == 0:
        if verbose:
            print("No pairs flagged for Granger causality")
        result = pl.DataFrame()
        result.write_parquet(output_path)
        return result

    # Build signal time series lookup
    signals = obs[signal_column].unique().to_list()
    signal_data = {}

    for signal in signals:
        sig = obs.filter(pl.col(signal_column) == signal).sort(index_column)
        values = sig[value_column].to_numpy()
        values = values[~np.isnan(values)]
        if len(values) >= min_samples:
            signal_data[signal] = values

    if verbose:
        print(f"Signals with sufficient data: {len(signal_data)}/{len(signals)}")

    # Compute causality for flagged pairs
    results = []
    pairs_list = granger_pairs.to_dicts()

    for i, pair in enumerate(pairs_list):
        signal_a = pair['signal_a']
        signal_b = pair['signal_b']

        if signal_a not in signal_data or signal_b not in signal_data:
            continue

        x = signal_data[signal_a]
        y = signal_data[signal_b]

        # Align lengths
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]

        if len(x) < min_samples:
            continue

        try:
            # Compute A → B direction
            causality_ab = compute_causality(x, y)
            # Compute B → A direction
            causality_ba = compute_causality(y, x)

            results.append({
                'signal_a': signal_a,
                'signal_b': signal_b,
                'granger_f_a_to_b': causality_ab.get('granger_f'),
                'granger_p_a_to_b': causality_ab.get('granger_p'),
                'granger_f_b_to_a': causality_ba.get('granger_f'),
                'granger_p_b_to_a': causality_ba.get('granger_p'),
                'transfer_entropy_a_to_b': causality_ab.get('transfer_entropy'),
                'transfer_entropy_b_to_a': causality_ba.get('transfer_entropy'),
                'n_samples': min_len,
            })
        except Exception as e:
            if verbose:
                print(f"  Warning: {signal_a}->{signal_b}: {e}")

        if verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_granger} pairs...")

    # Build DataFrame
    result = pl.DataFrame(results) if results else pl.DataFrame()

    if len(result) > 0:
        result.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {result.shape}")

        if len(result) > 0 and 'granger_f_a_to_b' in result.columns:
            valid_granger = result.filter(pl.col('granger_f_a_to_b').is_not_null())
            if len(valid_granger) > 0:
                print(f"\nGranger causality computed: {len(valid_granger)} pairs")

        print()
        print("─" * 50)
        print(f"✓ {Path(output_path).absolute()}")
        print("─" * 50)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 10: Information Flow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes causality metrics for eigenvector-gated pairs.

Uses needs_granger flag from signal_pairwise to avoid computing
causality for all O(n²) pairs. Only pairs with high PC co-loading
(indicating correlation) get Granger analysis to determine direction.

Example:
  python -m engines.entry_points.stage_10_information_flow \\
      observations.parquet signal_pairwise.parquet \\
      -o information_flow.parquet
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('signal_pairwise', help='Path to signal_pairwise.parquet')
    parser.add_argument('-o', '--output', default='information_flow.parquet',
                        help='Output path (default: information_flow.parquet)')
    parser.add_argument('--min-samples', type=int, default=100,
                        help='Minimum samples per signal (default: 100)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        args.signal_pairwise,
        args.output,
        min_samples=args.min_samples,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
