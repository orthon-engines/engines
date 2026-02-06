"""
Stage 09: Dynamics Entry Point
==============================

Pure orchestration - calls engines/parallel/dynamics_runner.py for computation.

Inputs:
    - observations.parquet

Output:
    - dynamics.parquet

Computes dynamical systems metrics per signal:
    - Lyapunov exponents
    - RQA metrics (recurrence, determinism, laminarity)
    - Attractor properties (correlation dimension, type)
"""

import argparse
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from prism.engines.dynamics.lyapunov import compute as compute_lyapunov
from prism.engines.dynamics.attractor import compute as compute_attractor


def run(
    observations_path: str,
    output_path: str = "dynamics.parquet",
    signal_column: str = 'signal_id',
    value_column: str = 'value',
    index_column: str = 'I',
    min_samples: int = 100,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run full dynamics computation for all signals.

    Args:
        observations_path: Path to observations.parquet
        output_path: Output path for dynamics.parquet
        signal_column: Column with signal IDs
        value_column: Column with values
        index_column: Column with time index
        min_samples: Minimum samples for computation
        verbose: Print progress

    Returns:
        Dynamics DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 09: DYNAMICS")
        print("Dynamical systems metrics per signal")
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

        # Remove NaN
        values = values[~np.isnan(values)]

        if len(values) < min_samples:
            continue

        result = {
            'signal_id': signal,
            'n_samples': len(values),
        }

        # Lyapunov
        try:
            lyap_result = compute_lyapunov(values, min_samples=min_samples)
            result['lyapunov'] = lyap_result.get('lyapunov')
            result['lyapunov_confidence'] = lyap_result.get('confidence')
            result['embedding_dim'] = lyap_result.get('embedding_dim')
            result['embedding_tau'] = lyap_result.get('embedding_tau')
        except Exception:
            result['lyapunov'] = np.nan

        # Attractor
        try:
            attr_result = compute_attractor(values)
            result['correlation_dim'] = attr_result.get('correlation_dim')
            result['attractor_type'] = attr_result.get('attractor_type')
        except Exception:
            result['correlation_dim'] = np.nan
            result['attractor_type'] = 'unknown'

        results.append(result)

        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(signals)} signals...")

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
        description="Stage 09: Dynamics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes dynamical systems metrics per signal:
  - Lyapunov exponents
  - Attractor properties

Example:
  python -m prism.entry_points.stage_09_dynamics \\
      observations.parquet -o dynamics.parquet
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('-o', '--output', default='dynamics.parquet',
                        help='Output path (default: dynamics.parquet)')
    parser.add_argument('--min-samples', type=int, default=100,
                        help='Minimum samples (default: 100)')
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
