"""
Stage 02a: Observations Windowed Entry Point
============================================

Pure orchestration - windows observations for downstream processing.

Inputs:
    - observations.parquet
    - manifest.yaml (for window/stride config)

Output:
    - observations_windowed.parquet

Creates windowed views of observations for efficient downstream computation.
Each row represents a window of data with start_I, end_I indices.
"""

import argparse
import polars as pl
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def run(
    observations_path: str,
    manifest_path: str,
    output_path: str = "observations_windowed.parquet",
    window_size: Optional[int] = None,
    stride: Optional[int] = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Create windowed observations for downstream processing.

    Args:
        observations_path: Path to observations.parquet
        manifest_path: Path to manifest.yaml
        output_path: Output path for observations_windowed.parquet
        window_size: Override window size (default from manifest)
        stride: Override stride (default from manifest)
        verbose: Print progress

    Returns:
        Windowed observations DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 02a: OBSERVATIONS WINDOWED")
        print("Creating windowed views for efficient computation")
        print("=" * 70)

    # Load manifest for window/stride config
    with open(manifest_path, 'r') as f:
        manifest = yaml.safe_load(f)

    # Get window/stride from manifest or use defaults
    system = manifest.get('system', {})
    window = window_size or system.get('window', 128)
    step = stride or system.get('stride', 64)

    if verbose:
        print(f"Window size: {window}, Stride: {step}")

    # Load observations
    obs = pl.read_parquet(observations_path)

    if verbose:
        n_rows = len(obs)
        n_signals = obs['signal_id'].n_unique()
        print(f"Loaded: {n_rows:,} observations, {n_signals} signals")

    # Create window indices per signal
    results = []

    for signal_id in obs['signal_id'].unique().to_list():
        sig_data = obs.filter(pl.col('signal_id') == signal_id).sort('I')
        n_samples = len(sig_data)

        # Get cohort if present
        cohort = None
        if 'cohort' in sig_data.columns:
            cohort = sig_data['cohort'][0]

        # Generate windows
        window_idx = 0
        for start_i in range(0, n_samples - window + 1, step):
            end_i = start_i + window

            # Get window data stats for validation
            window_data = sig_data.slice(start_i, window)
            values = window_data['value'].to_numpy()

            result = {
                'signal_id': signal_id,
                'window_idx': window_idx,
                'start_I': int(sig_data['I'][start_i]),
                'end_I': int(sig_data['I'][end_i - 1]),
                'n_samples': window,
                'window_mean': float(np.nanmean(values)),
                'window_std': float(np.nanstd(values)),
                'window_min': float(np.nanmin(values)),
                'window_max': float(np.nanmax(values)),
                'nan_count': int(np.sum(np.isnan(values))),
            }

            if cohort is not None:
                result['cohort'] = cohort

            results.append(result)
            window_idx += 1

    # Build DataFrame
    df = pl.DataFrame(results) if results else pl.DataFrame()

    # Write output
    if len(df) > 0:
        df.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {df.shape}")
        if len(df) > 0:
            n_windows = df['window_idx'].max() + 1
            print(f"Windows per signal: ~{n_windows}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Stage 02a: Observations Windowed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Creates windowed views of observations for downstream processing.

Example:
  python -m prism.entry_points.stage_02a_observations_windowed \\
      observations.parquet manifest.yaml -o observations_windowed.parquet
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('manifest', help='Path to manifest.yaml')
    parser.add_argument('-o', '--output', default='observations_windowed.parquet',
                        help='Output path (default: observations_windowed.parquet)')
    parser.add_argument('--window', type=int, help='Override window size')
    parser.add_argument('--stride', type=int, help='Override stride')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        args.manifest,
        args.output,
        window_size=args.window,
        stride=args.stride,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
