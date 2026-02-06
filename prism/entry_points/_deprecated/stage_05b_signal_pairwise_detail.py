"""
Stage 05b: Signal Pairwise Detail Entry Point
=============================================

Pure orchestration - detailed pairwise analysis between signals.

Inputs:
    - observations.parquet

Output:
    - signal_pairwise_summary.parquet
    - signal_pairwise_detail.parquet

Computes detailed pairwise relationships including:
    - Cross-correlation at multiple lags
    - Lead/lag relationships
    - Coherence
"""

import argparse
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from itertools import combinations


def compute_cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int = 20) -> Dict[str, float]:
    """Compute cross-correlation at multiple lags."""
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    # Normalize
    x = (x - np.mean(x)) / (np.std(x) + 1e-10)
    y = (y - np.mean(y)) / (np.std(y) + 1e-10)

    # Cross-correlation at lag 0
    xcorr_0 = np.sum(x * y) / n

    # Find best lag (lead/lag detection)
    best_lag = 0
    best_xcorr = xcorr_0

    for lag in range(1, min(max_lag, n // 4)):
        # Positive lag: y leads x
        xcorr_pos = np.sum(x[lag:] * y[:-lag]) / (n - lag)
        # Negative lag: x leads y
        xcorr_neg = np.sum(x[:-lag] * y[lag:]) / (n - lag)

        if abs(xcorr_pos) > abs(best_xcorr):
            best_xcorr = xcorr_pos
            best_lag = lag

        if abs(xcorr_neg) > abs(best_xcorr):
            best_xcorr = xcorr_neg
            best_lag = -lag

    return {
        'xcorr_lag0': float(xcorr_0),
        'xcorr_best': float(best_xcorr),
        'best_lag': int(best_lag),
        'abs_best_xcorr': float(abs(best_xcorr)),
    }


def run(
    observations_path: str,
    summary_output: str = "signal_pairwise_summary.parquet",
    detail_output: str = "signal_pairwise_detail.parquet",
    signal_column: str = 'signal_id',
    value_column: str = 'value',
    index_column: str = 'I',
    max_pairs: int = 500,
    verbose: bool = True,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Compute detailed pairwise signal relationships.

    Args:
        observations_path: Path to observations.parquet
        summary_output: Output path for summary
        detail_output: Output path for detail
        signal_column: Column with signal IDs
        value_column: Column with values
        index_column: Column with time index
        max_pairs: Maximum pairs to process
        verbose: Print progress

    Returns:
        Tuple of (summary DataFrame, detail DataFrame)
    """
    if verbose:
        print("=" * 70)
        print("STAGE 05b: SIGNAL PAIRWISE DETAIL")
        print("Detailed pairwise signal analysis")
        print("=" * 70)

    # Load observations
    obs = pl.read_parquet(observations_path)

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
        if len(values) >= 20:
            signal_data[signal] = values

    available_signals = list(signal_data.keys())
    pairs = list(combinations(available_signals, 2))

    if len(pairs) > max_pairs:
        if verbose:
            print(f"Limiting to {max_pairs} pairs (of {len(pairs)})")
        pairs = pairs[:max_pairs]

    summary_results = []
    detail_results = []

    for i, (sig_a, sig_b) in enumerate(pairs):
        x = signal_data[sig_a]
        y = signal_data[sig_b]

        # Align lengths
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]

        # Cross-correlation analysis
        xcorr_result = compute_cross_correlation(x, y)

        # Pearson correlation
        corr = np.corrcoef(x, y)[0, 1]

        # Summary row
        summary_results.append({
            'signal_a': sig_a,
            'signal_b': sig_b,
            'pearson_corr': float(corr) if not np.isnan(corr) else np.nan,
            **xcorr_result,
            'n_samples': min_len,
        })

        # Detail rows (correlation at multiple lags)
        for lag in range(-10, 11):
            if abs(lag) >= min_len // 2:
                continue

            if lag == 0:
                lag_corr = corr
            elif lag > 0:
                lag_corr = np.corrcoef(x[lag:], y[:-lag])[0, 1]
            else:
                lag_corr = np.corrcoef(x[:lag], y[-lag:])[0, 1]

            detail_results.append({
                'signal_a': sig_a,
                'signal_b': sig_b,
                'lag': lag,
                'correlation': float(lag_corr) if not np.isnan(lag_corr) else np.nan,
            })

        if verbose and (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(pairs)} pairs...")

    # Build DataFrames
    summary_df = pl.DataFrame(summary_results) if summary_results else pl.DataFrame()
    detail_df = pl.DataFrame(detail_results) if detail_results else pl.DataFrame()

    # Write outputs
    if len(summary_df) > 0:
        summary_df.write_parquet(summary_output)
    if len(detail_df) > 0:
        detail_df.write_parquet(detail_output)

    if verbose:
        print(f"\nSaved summary: {summary_output} ({summary_df.shape})")
        print(f"Saved detail: {detail_output} ({detail_df.shape})")

    return summary_df, detail_df


def main():
    parser = argparse.ArgumentParser(
        description="Stage 05b: Signal Pairwise Detail",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes detailed pairwise relationships:
  - Cross-correlation at multiple lags
  - Lead/lag detection
  - Lag-specific correlation profiles

Outputs two files:
  - signal_pairwise_summary.parquet (one row per pair)
  - signal_pairwise_detail.parquet (one row per pair+lag)

Example:
  python -m prism.entry_points.stage_05b_signal_pairwise_detail \\
      observations.parquet
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('--summary-output', default='signal_pairwise_summary.parquet',
                        help='Summary output path')
    parser.add_argument('--detail-output', default='signal_pairwise_detail.parquet',
                        help='Detail output path')
    parser.add_argument('--max-pairs', type=int, default=500,
                        help='Maximum pairs (default: 500)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        args.summary_output,
        args.detail_output,
        max_pairs=args.max_pairs,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
