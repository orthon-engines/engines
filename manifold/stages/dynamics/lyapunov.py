"""
Stage 08: Lyapunov Entry Point
==============================

Pure orchestration - calls engines/dynamics/lyapunov.py for computation.

Inputs:
    - observations.parquet

Output:
    - lyapunov.parquet

Computes per-signal Lyapunov exponents:
    - Largest Lyapunov exponent (λ)
    - Embedding dimension
    - Embedding delay
    - Confidence score

ENGINES computes, ORTHON interprets:
    λ > 0: Chaos (trajectories diverge)
    λ ≈ 0: Quasi-periodic (trajectories parallel)
    λ < 0: Stable (trajectories converge)
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
from typing import Optional

from manifold.core.dynamics.lyapunov import compute as compute_lyapunov
from manifold.io.writer import write_output


def run(
    observations_path: str,
    data_path: str = ".",
    min_samples: int = 200,
    method: str = 'rosenstein',
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute Lyapunov exponents for all signals.

    Args:
        observations_path: Path to observations.parquet
        data_path: Root data directory for output
        min_samples: Minimum samples required per signal
        method: 'rosenstein' or 'kantz'
        verbose: Print progress

    Returns:
        Lyapunov DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 08: LYAPUNOV")
        print("Computing per-signal Lyapunov exponents")
        print("=" * 70)

    # Load observations
    obs = pl.read_parquet(observations_path)

    if verbose:
        print(f"Loaded observations: {obs.shape}")

    # Get unique signals
    signals = obs['signal_id'].unique().to_list()
    has_cohort = 'cohort' in obs.columns

    if verbose:
        print(f"Processing {len(signals)} signals...")

    results = []

    for signal_id in signals:
        signal_data = obs.filter(pl.col('signal_id') == signal_id).sort('I')

        if has_cohort:
            # Process per cohort
            cohorts = signal_data['cohort'].unique().to_list()
            for cohort in cohorts:
                cohort_data = signal_data.filter(pl.col('cohort') == cohort)
                values = cohort_data['value'].to_numpy()

                lyap_result = compute_lyapunov(
                    values,
                    min_samples=min_samples,
                    method=method,
                )

                results.append({
                    'signal_id': signal_id,
                    'cohort': cohort,
                    'lyapunov': lyap_result['lyapunov'],
                    'embedding_dim': lyap_result['embedding_dim'],
                    'embedding_tau': lyap_result['embedding_tau'],
                    'confidence': lyap_result['confidence'],
                    'n_samples': len(values),
                })
        else:
            values = signal_data['value'].to_numpy()

            lyap_result = compute_lyapunov(
                values,
                min_samples=min_samples,
                method=method,
            )

            results.append({
                'signal_id': signal_id,
                'lyapunov': lyap_result['lyapunov'],
                'embedding_dim': lyap_result['embedding_dim'],
                'embedding_tau': lyap_result['embedding_tau'],
                'confidence': lyap_result['confidence'],
                'n_samples': len(values),
            })

    # Build DataFrame
    result = pl.DataFrame(results) if results else pl.DataFrame()

    if len(result) > 0:
        write_output(result, data_path, 'lyapunov', verbose=verbose)

    if verbose:
        print(f"Shape: {result.shape}")

        if len(result) > 0 and 'lyapunov' in result.columns:
            valid = result.filter(pl.col('lyapunov').is_not_null())
            if len(valid) > 0:
                print(f"\nLyapunov stats (n={len(valid)}):")
                print(f"  Mean: {valid['lyapunov'].mean():.4f}")
                print(f"  Std:  {valid['lyapunov'].std():.4f}")
                print(f"  Range: [{valid['lyapunov'].min():.4f}, {valid['lyapunov'].max():.4f}]")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 08: Lyapunov",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes Lyapunov exponents for all signals.

The Lyapunov exponent measures sensitivity to initial conditions:
  λ > 0: Chaotic (exponential divergence)
  λ ≈ 0: Quasi-periodic (parallel trajectories)
  λ < 0: Stable (convergent trajectories)

Example:
  python -m engines.entry_points.stage_08_lyapunov \\
      observations.parquet -o lyapunov.parquet
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('--min-samples', type=int, default=200,
                        help='Minimum samples per signal (default: 200)')
    parser.add_argument('--method', choices=['rosenstein', 'kantz'], default='rosenstein',
                        help='Algorithm (default: rosenstein)')
    parser.add_argument('-d', '--data-path', default='.',
                        help='Root data directory (default: .)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        args.data_path,
        min_samples=args.min_samples,
        method=args.method,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
