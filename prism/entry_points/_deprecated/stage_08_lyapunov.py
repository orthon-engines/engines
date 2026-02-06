"""
Stage 08: Lyapunov Entry Point
==============================

Pure orchestration - calls engines/dynamics/lyapunov.py for computation.

Inputs:
    - observations.parquet

Output:
    - lyapunov.parquet

Computes per-signal Lyapunov exponents:
    - lyapunov_exp: Largest Lyapunov exponent
    - stability: Classification (stable/marginal/unstable/chaotic)
    - embedding_dim, embedding_tau: Phase space parameters
    - confidence: Estimation quality
"""

import argparse
import polars as pl
import numpy as np
from pathlib import Path
from typing import Optional

from prism.engines.dynamics.lyapunov import compute as compute_lyapunov
from prism.engines.dynamics.formal_definitions import classify_stability


def run(
    observations_path: str,
    output_path: str = "lyapunov.parquet",
    signal_column: str = 'signal_id',
    value_column: str = 'value',
    index_column: str = 'I',
    min_samples: int = 200,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run Lyapunov exponent computation for all signals.

    Args:
        observations_path: Path to observations.parquet
        output_path: Output path for lyapunov.parquet
        signal_column: Column with signal IDs
        value_column: Column with values
        index_column: Column with time index
        min_samples: Minimum samples for Lyapunov computation
        verbose: Print progress

    Returns:
        Lyapunov DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 08: LYAPUNOV")
        print("Per-signal stability measurement")
        print("=" * 70)

    # Load observations
    obs = pl.read_parquet(observations_path)

    if verbose:
        n_signals = obs[signal_column].n_unique()
        print(f"Loaded: {len(obs):,} observations, {n_signals} signals")

    signals = obs[signal_column].unique().to_list()
    results = []

    for i, signal in enumerate(signals):
        signal_data = obs.filter(pl.col(signal_column) == signal).sort(index_column)
        values = signal_data[value_column].to_numpy()

        # Compute Lyapunov
        result = compute_lyapunov(values, min_samples=min_samples)

        # Add stability classification
        lyap_val = result.get('lyapunov')
        if lyap_val is not None:
            stability = classify_stability(lyap_val)
            result['stability'] = stability.value
            result['lyapunov_exp'] = lyap_val
        else:
            result['stability'] = 'unknown'
            result['lyapunov_exp'] = np.nan

        result['signal_id'] = signal
        result['n_samples'] = len(values)
        results.append(result)

        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(signals)} signals...")

    # Build DataFrame
    df = pl.DataFrame(results)
    df.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {df.shape}")

        # Summary
        stable_count = len([r for r in results
                           if r.get('stability') in ['stable', 'asymptotically_stable']])
        unstable_count = len([r for r in results
                             if r.get('stability') in ['weakly_unstable', 'chaotic', 'unstable']])
        print(f"Stable: {stable_count}, Unstable: {unstable_count}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Stage 08: Lyapunov",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes per-signal Lyapunov exponents:
  - lyapunov_exp: Largest Lyapunov exponent
  - stability: stable/marginal/unstable/chaotic
  - embedding_dim, embedding_tau: Phase space parameters

Example:
  python -m prism.entry_points.stage_08_lyapunov \\
      observations.parquet -o lyapunov.parquet
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('-o', '--output', default='lyapunov.parquet',
                        help='Output path (default: lyapunov.parquet)')
    parser.add_argument('--min-samples', type=int, default=200,
                        help='Minimum samples for Lyapunov (default: 200)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        args.output,
        min_samples=args.min_samples,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
