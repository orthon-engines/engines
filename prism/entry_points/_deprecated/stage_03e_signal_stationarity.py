"""
Stage 03e: Signal Stationarity Entry Point
==========================================

Pure orchestration - calls engines for stationarity tests.

Inputs:
    - observations.parquet

Output:
    - signal_stationarity.parquet

Computes per-signal stationarity properties:
    - ADF test (adf_stat, adf_pvalue)
    - Variance ratio test
    - Hurst exponent (memory)
    - ACF decay characteristics
"""

import argparse
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any

from prism.engines.signal.adf_stat import compute as compute_adf
from prism.engines.signal.variance_ratio import compute as compute_variance_ratio
from prism.engines.signal.memory import compute as compute_memory, compute_acf_decay


def run(
    observations_path: str,
    output_path: str = "signal_stationarity.parquet",
    signal_column: str = 'signal_id',
    value_column: str = 'value',
    index_column: str = 'I',
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run stationarity tests for all signals.

    Args:
        observations_path: Path to observations.parquet
        output_path: Output path for signal_stationarity.parquet
        signal_column: Column with signal IDs
        value_column: Column with values
        index_column: Column with time index
        verbose: Print progress

    Returns:
        Signal stationarity DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 03e: SIGNAL STATIONARITY")
        print("Per-signal stationarity tests")
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

        # Compute stationarity metrics
        adf_metrics = compute_adf(values)
        vr_metrics = compute_variance_ratio(values)
        memory_metrics = compute_memory(values)
        acf_metrics = compute_acf_decay(values)

        result = {
            'signal_id': signal_id,
            'n_samples': len(values),
            **adf_metrics,
            **vr_metrics,
            **memory_metrics,
            **acf_metrics,
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
        if 'adf_pvalue' in df.columns:
            stationary_pct = (df['adf_pvalue'] < 0.05).mean() * 100
            print(f"Stationary signals (ADF p<0.05): {stationary_pct:.1f}%")
        if 'hurst' in df.columns:
            mean_hurst = df['hurst'].mean()
            print(f"Mean Hurst exponent: {mean_hurst:.3f}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Stage 03e: Signal Stationarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes per-signal stationarity properties:
  - adf_stat, adf_pvalue (Augmented Dickey-Fuller test)
  - variance_ratio (random walk test)
  - hurst (long-range dependence)
  - acf_lag1, acf_half_life (autocorrelation decay)

ADF p < 0.05 -> stationary
Hurst > 0.5 -> persistent (trending)
Hurst < 0.5 -> anti-persistent (mean-reverting)

Example:
  python -m prism.entry_points.stage_03e_signal_stationarity \\
      observations.parquet -o signal_stationarity.parquet
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('-o', '--output', default='signal_stationarity.parquet',
                        help='Output path (default: signal_stationarity.parquet)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
