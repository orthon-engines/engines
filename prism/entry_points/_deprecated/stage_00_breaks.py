"""
Stage 00: Break Detection Entry Point
=====================================

Thin orchestrator:
1. Read observations
2. Call break detection engine per signal
3. Write breaks.parquet
4. Return summary stats for typology enrichment

Runs early in pipeline — operates on raw observations.
Stage 00 because it runs BEFORE signal_vector and can inform typology.
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional

from prism.engines.breaks import compute, summarize_breaks


def run(
    observations_path: str,
    output_path: str = "breaks.parquet",
    sensitivity: float = 1.0,
    min_spacing: int = 10,
    context_window: int = 50,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run break detection on all signals.

    Args:
        observations_path: Path to observations.parquet
        output_path: Where to write breaks.parquet
        sensitivity: Detection sensitivity (0.5=conservative, 2.0=aggressive)
        min_spacing: Minimum I between breaks
        context_window: Samples for pre/post level computation
        verbose: Print progress

    Returns:
        breaks DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 00: BREAK DETECTION")
        print("Heaviside (steps) + Dirac (impulses)")
        print("=" * 70)

    obs = pl.read_parquet(observations_path)

    # Get unique signal_ids
    signal_ids = obs['signal_id'].unique().sort().to_list()
    if verbose:
        print(f"Signals: {len(signal_ids)}")

    all_breaks = []
    summaries = {}

    for sig_id in signal_ids:
        # Extract signal values ordered by I
        signal_data = (
            obs
            .filter(pl.col('signal_id') == sig_id)
            .sort('I')
        )
        y = signal_data['value'].to_numpy()

        # Detect breaks
        breaks = compute(
            y,
            signal_id=sig_id,
            sensitivity=sensitivity,
            min_spacing=min_spacing,
            context_window=context_window,
        )

        all_breaks.extend(breaks)
        summaries[sig_id] = summarize_breaks(breaks)

        if verbose and breaks:
            print(f"  {sig_id}: {len(breaks)} breaks detected")

    # Build output DataFrame
    if all_breaks:
        breaks_df = pl.DataFrame(all_breaks, schema={
            'signal_id': pl.Utf8,
            'I': pl.UInt32,
            'magnitude': pl.Float64,
            'direction': pl.Int8,
            'sharpness': pl.Float64,
            'duration': pl.UInt32,
            'pre_level': pl.Float64,
            'post_level': pl.Float64,
            'snr': pl.Float64,
        })
    else:
        # Empty with correct schema
        breaks_df = pl.DataFrame(schema={
            'signal_id': pl.Utf8,
            'I': pl.UInt32,
            'magnitude': pl.Float64,
            'direction': pl.Int8,
            'sharpness': pl.Float64,
            'duration': pl.UInt32,
            'pre_level': pl.Float64,
            'post_level': pl.Float64,
            'snr': pl.Float64,
        })

    # Write
    breaks_df.write_parquet(output_path)

    if verbose:
        total = len(all_breaks)
        signals_with = sum(1 for s in summaries.values() if s['n_breaks'] > 0)
        print(f"\nTotal breaks: {total}")
        print(f"Signals with breaks: {signals_with}/{len(signal_ids)}")
        print(f"Output: {output_path}")

    return breaks_df


def run_from_manifest(
    manifest: Dict[str, Any],
    verbose: bool = True,
) -> pl.DataFrame:
    """Run break detection using manifest configuration."""
    obs_path = manifest['paths']['observations']
    output_dir = manifest['paths'].get('output_dir', '.')
    output_path = str(Path(output_dir) / 'breaks.parquet')

    # Sensitivity from manifest (ORTHON can tune per job)
    sensitivity = manifest.get('defaults', {}).get('break_sensitivity', 1.0)

    return run(
        observations_path=obs_path,
        output_path=output_path,
        sensitivity=sensitivity,
        verbose=verbose,
    )


def get_summaries_for_typology(
    breaks_path: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Load breaks.parquet and return per-signal summaries
    for enriching typology.

    Returns:
        Dict mapping signal_id → summary stats dict
    """
    breaks_df = pl.read_parquet(breaks_path)

    summaries = {}
    for sig_id in breaks_df['signal_id'].unique().to_list():
        sig_breaks = breaks_df.filter(pl.col('signal_id') == sig_id)
        breaks_list = sig_breaks.to_dicts()
        summaries[sig_id] = summarize_breaks(breaks_list)

    return summaries


def main():
    parser = argparse.ArgumentParser(
        description="Stage 00: Break Detection (Heaviside/Dirac)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Detects discontinuities in signals:
  - Steps (Heaviside): sustained level changes
  - Impulses (Dirac): transient spikes
  - Gradual shifts: slow regime transitions

Output schema (breaks.parquet):
  signal_id, I, magnitude, direction, sharpness,
  duration, pre_level, post_level, snr

Example:
  python -m prism.entry_points.stage_00_breaks \\
      observations.parquet -o breaks.parquet --sensitivity 1.0
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('-o', '--output', default='breaks.parquet',
                        help='Output path (default: breaks.parquet)')
    parser.add_argument('--sensitivity', type=float, default=1.0,
                        help='Detection sensitivity (0.5=conservative, 2.0=aggressive)')
    parser.add_argument('--min-spacing', type=int, default=10,
                        help='Minimum samples between breaks (default: 10)')
    parser.add_argument('--context-window', type=int, default=50,
                        help='Context window for pre/post levels (default: 50)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        args.output,
        sensitivity=args.sensitivity,
        min_spacing=args.min_spacing,
        context_window=args.context_window,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
