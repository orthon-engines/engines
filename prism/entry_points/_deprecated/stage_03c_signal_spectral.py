"""
Stage 03c: Signal Spectral Entry Point
======================================

Pure orchestration - calls engines/signal/spectral.py for computation.

Inputs:
    - observations.parquet

Output:
    - signal_spectral.parquet

Computes per-signal spectral properties:
    - dominant_freq
    - spectral_entropy
    - spectral_centroid
    - spectral_bandwidth
    - spectral_slope
"""

import argparse
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any

from prism.engines.signal.spectral import compute as compute_spectral


def run(
    observations_path: str,
    output_path: str = "signal_spectral.parquet",
    signal_column: str = 'signal_id',
    value_column: str = 'value',
    index_column: str = 'I',
    sample_rate: float = 1.0,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run spectral computation for all signals.

    Args:
        observations_path: Path to observations.parquet
        output_path: Output path for signal_spectral.parquet
        signal_column: Column with signal IDs
        value_column: Column with values
        index_column: Column with time index
        sample_rate: Sampling rate in Hz
        verbose: Print progress

    Returns:
        Signal spectral DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 03c: SIGNAL SPECTRAL")
        print("Per-signal spectral properties")
        print("=" * 70)

    # Load observations
    obs = pl.read_parquet(observations_path)

    signals = obs[signal_column].unique().to_list()
    n_signals = len(signals)

    if verbose:
        print(f"Processing {n_signals} signals...")
        print(f"Sample rate: {sample_rate} Hz")

    results = []

    for i, signal_id in enumerate(signals):
        sig_data = obs.filter(pl.col(signal_column) == signal_id).sort(index_column)
        values = sig_data[value_column].to_numpy()
        values = values[~np.isnan(values)]

        # Get cohort if present
        cohort = None
        if 'cohort' in sig_data.columns:
            cohort = sig_data['cohort'][0]

        # Compute spectral metrics
        spectral_metrics = compute_spectral(values, sample_rate=sample_rate)

        result = {
            'signal_id': signal_id,
            'n_samples': len(values),
            **spectral_metrics,
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
        if 'spectral_entropy' in df.columns:
            mean_se = df['spectral_entropy'].mean()
            print(f"Mean spectral entropy: {mean_se:.3f}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Stage 03c: Signal Spectral",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes per-signal spectral properties:
  - dominant_freq (peak frequency)
  - spectral_entropy (frequency spread)
  - spectral_centroid (center of mass)
  - spectral_bandwidth (spread around centroid)
  - spectral_slope (power law decay)

Example:
  python -m prism.entry_points.stage_03c_signal_spectral \\
      observations.parquet -o signal_spectral.parquet
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('-o', '--output', default='signal_spectral.parquet',
                        help='Output path (default: signal_spectral.parquet)')
    parser.add_argument('--sample-rate', type=float, default=1.0,
                        help='Sample rate in Hz (default: 1.0)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        args.output,
        sample_rate=args.sample_rate,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
