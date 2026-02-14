"""
Stage 22: Rolling FTLE Entry Point
==================================

FTLE computed at each timestep in rolling windows, not just a single
global summary. Shows how stability evolves through the system's lifecycle.

A bearing that's stable for 9000 samples and chaotic for 1000 samples
gets an average FTLE that describes neither state. Rolling FTLE gives
FTLE(I) - the stability at each moment.

Inputs:
    - observations.parquet

Output:
    - ftle_rolling.parquet

Key output:
    ftle_velocity = d(FTLE)/dI - is stability changing?
    Positive = system becoming more unstable
    Negative = system stabilizing
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
from typing import Optional

from manifold.core.dynamics.ftle import compute as compute_ftle
from manifold.io.writer import write_output


def run(
    observations_path: str,
    data_path: str = ".",
    window_size: int = 200,
    stride: int = 50,
    min_samples: int = 100,
    direction: str = 'forward',
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute rolling FTLE over time.

    Args:
        observations_path: Path to observations.parquet
        data_path: Root data directory for output
        window_size: Window size for FTLE computation
        stride: Step size between windows
        min_samples: Minimum samples for reliable FTLE
        direction: 'forward' or 'backward'
        verbose: Print progress

    Returns:
        ftle_rolling DataFrame with FTLE at each window center
    """
    if verbose:
        print("=" * 70)
        print(f"STAGE 22: ROLLING FTLE ({'BACKWARD' if direction == 'backward' else 'FORWARD'})")
        print("FTLE evolution over time - stability trajectory")
        print("=" * 70)

    obs = pl.read_parquet(observations_path)

    if verbose:
        print(f"Loaded observations: {obs.shape}")

    has_cohort = 'cohort' in obs.columns
    cohorts = obs['cohort'].unique().to_list() if has_cohort else ['all']
    signals = sorted(obs['signal_id'].unique().to_list())

    if verbose:
        print(f"Cohorts: {len(cohorts)}")
        print(f"Signals: {len(signals)}")
        print(f"Window: {window_size}, Stride: {stride}")

    results = []

    for cohort in cohorts:
        if has_cohort:
            cohort_data = obs.filter(pl.col('cohort') == cohort)
        else:
            cohort_data = obs

        for signal_id in signals:
            signal_data = cohort_data.filter(pl.col('signal_id') == signal_id).sort('I')
            values = signal_data['value'].to_numpy()
            i_values = signal_data['I'].to_numpy()

            if len(values) < window_size:
                continue

            # Rolling window FTLE
            prev_ftle = None
            signal_results = []

            for start in range(0, len(values) - window_size + 1, stride):
                window = values[start:start + window_size]
                window_i = i_values[start:start + window_size]

                # Reverse for backward FTLE
                if direction == 'backward':
                    window = window[::-1]

                # Compute FTLE
                ftle_result = compute_ftle(window, min_samples=min_samples)

                if ftle_result['ftle'] is None:
                    continue

                ftle_val = ftle_result['ftle']

                # FTLE velocity (rate of change)
                if prev_ftle is not None:
                    ftle_velocity = ftle_val - prev_ftle
                else:
                    ftle_velocity = 0.0

                signal_results.append({
                    'I': int(window_i[window_size // 2]),  # Window center
                    'cohort': cohort,
                    'signal_id': signal_id,
                    'ftle': float(ftle_val),
                    'ftle_std': float(ftle_result['ftle_std']) if ftle_result['ftle_std'] else None,
                    'ftle_velocity': float(ftle_velocity),
                    'confidence': float(ftle_result['confidence']),
                    'embedding_dim': ftle_result['embedding_dim'],
                    'embedding_tau': ftle_result['embedding_tau'],
                    'window_start': int(window_i[0]),
                    'window_end': int(window_i[-1]),
                    'direction': direction,
                })

                prev_ftle = ftle_val

            # Compute acceleration (second derivative)
            for i in range(1, len(signal_results) - 1):
                v1 = signal_results[i]['ftle_velocity']
                v0 = signal_results[i - 1]['ftle_velocity']
                signal_results[i]['ftle_acceleration'] = v1 - v0

            # First and last don't have acceleration
            if signal_results:
                signal_results[0]['ftle_acceleration'] = 0.0
                if len(signal_results) > 1:
                    signal_results[-1]['ftle_acceleration'] = 0.0

            results.extend(signal_results)

    # Build output
    if results:
        result = pl.DataFrame(results)
    else:
        result = pl.DataFrame(schema={
            'I': pl.Int64,
            'cohort': pl.Utf8,
            'signal_id': pl.Utf8,
            'ftle': pl.Float64,
            'ftle_std': pl.Float64,
            'ftle_velocity': pl.Float64,
            'ftle_acceleration': pl.Float64,
            'confidence': pl.Float64,
            'embedding_dim': pl.Int64,
            'embedding_tau': pl.Int64,
            'window_start': pl.Int64,
            'window_end': pl.Int64,
            'direction': pl.Utf8,
        })

    write_output(result, data_path, 'ftle_rolling', verbose=verbose)

    if verbose:
        print(f"Shape: {result.shape}")

        if len(result) > 0:
            print("\nRolling FTLE stats:")
            print(f"  Mean FTLE: {result['ftle'].mean():.4f}")
            print(f"  Mean |velocity|: {result['ftle_velocity'].abs().mean():.4f}")

            # Signals with highest FTLE variance over time (most dynamic stability)
            if 'signal_id' in result.columns:
                var_by_signal = (
                    result
                    .group_by('signal_id')
                    .agg(pl.col('ftle').std().alias('ftle_variability'))
                    .sort('ftle_variability', descending=True)
                    .head(5)
                )
                print("\nMost stability-variable signals:")
                for r in var_by_signal.iter_rows(named=True):
                    v = r['ftle_variability']
                    print(f"  {r['signal_id']}: std={v:.4f}" if v is not None else f"  {r['signal_id']}: std=N/A")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 22: Rolling FTLE (Stability Evolution)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes FTLE at each timestep in rolling windows.

Key outputs:
  ftle:              FTLE value at this window
  ftle_velocity:     d(FTLE)/dI - is stability changing?
  ftle_acceleration: d²(FTLE)/dI² - is change accelerating?

Positive velocity = system becoming more unstable
Negative velocity = system stabilizing

Example:
  python -m engines.entry_points.stage_22_ftle_rolling \\
      observations.parquet -o ftle_rolling.parquet --window 200 --stride 50
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('-d', '--data-path', default='.',
                        help='Root data directory (default: .)')
    parser.add_argument('--window', type=int, default=200,
                        help='Window size for FTLE (default: 200)')
    parser.add_argument('--stride', type=int, default=50,
                        help='Stride between windows (default: 50)')
    parser.add_argument('--backward', action='store_true',
                        help='Compute backward FTLE (attracting structures)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        args.data_path,
        window_size=args.window,
        stride=args.stride,
        direction='backward' if args.backward else 'forward',
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
