"""
Stage 08b: Cohort Membership Entry Point
========================================

Pure orchestration - tracks cohort membership over time.

Inputs:
    - signal_vector.parquet
    - cohort_discovery.parquet (optional, uses original cohorts otherwise)

Output:
    - cohort_membership.parquet
    - signal_cohort_waterfall.parquet

Tracks how signals move between cohorts over time (membership dynamics).
"""

import argparse
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def run(
    signal_vector_path: str,
    cohort_discovery_path: Optional[str] = None,
    membership_output: str = "cohort_membership.parquet",
    waterfall_output: str = "signal_cohort_waterfall.parquet",
    verbose: bool = True,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Track cohort membership over time.

    Args:
        signal_vector_path: Path to signal_vector.parquet
        cohort_discovery_path: Path to cohort_discovery.parquet (optional)
        membership_output: Output path for membership data
        waterfall_output: Output path for waterfall data
        verbose: Print progress

    Returns:
        Tuple of (membership DataFrame, waterfall DataFrame)
    """
    if verbose:
        print("=" * 70)
        print("STAGE 08b: COHORT MEMBERSHIP")
        print("Tracking cohort membership dynamics")
        print("=" * 70)

    # Load signal vector
    sv = pl.read_parquet(signal_vector_path)

    if verbose:
        print(f"Loaded signal_vector: {sv.shape}")

    # Load discovered cohorts if provided
    cohort_map = {}
    if cohort_discovery_path and Path(cohort_discovery_path).exists():
        cd = pl.read_parquet(cohort_discovery_path)
        for row in cd.iter_rows(named=True):
            cohort_map[row['signal_id']] = row['discovered_cohort']
        if verbose:
            print(f"Loaded cohort assignments for {len(cohort_map)} signals")
    elif 'cohort' in sv.columns:
        # Use original cohorts
        for row in sv.select(['signal_id', 'cohort']).unique().iter_rows(named=True):
            cohort_map[row['signal_id']] = row['cohort']
        if verbose:
            print(f"Using original cohorts for {len(cohort_map)} signals")

    # Get time windows
    if 'I' not in sv.columns:
        if verbose:
            print("No I column found in signal_vector")
        return pl.DataFrame(), pl.DataFrame()

    time_points = sorted(sv['I'].unique().to_list())
    signals = sv['signal_id'].unique().to_list()

    if verbose:
        print(f"Time points: {len(time_points)}")
        print(f"Signals: {len(signals)}")

    # Track membership over time
    membership_results = []
    waterfall_results = []

    # Feature columns for distance computation
    exclude_cols = {'I', 'signal_id', 'cohort', 'unit_id', 'window_idx'}
    feature_cols = [c for c in sv.columns if c not in exclude_cols
                    and sv[c].dtype in [pl.Float64, pl.Float32]]

    for signal_id in signals:
        sig_data = sv.filter(pl.col('signal_id') == signal_id).sort('I')

        cohort = cohort_map.get(signal_id, 'unknown')

        # Track feature evolution
        for i, row in enumerate(sig_data.iter_rows(named=True)):
            I = row['I']

            # Compute feature centroid distance at this time
            features = [row.get(c, np.nan) for c in feature_cols]
            feature_norm = np.linalg.norm([f for f in features if not np.isnan(f)])

            membership_results.append({
                'signal_id': signal_id,
                'I': I,
                'cohort': cohort,
                'feature_norm': float(feature_norm) if not np.isnan(feature_norm) else np.nan,
            })

            # Waterfall: track transitions (for visualization)
            if i > 0:
                waterfall_results.append({
                    'signal_id': signal_id,
                    'from_I': int(sig_data['I'][i-1]),
                    'to_I': I,
                    'cohort': cohort,
                })

    # Build DataFrames
    membership_df = pl.DataFrame(membership_results) if membership_results else pl.DataFrame()
    waterfall_df = pl.DataFrame(waterfall_results) if waterfall_results else pl.DataFrame()

    # Write outputs
    if len(membership_df) > 0:
        membership_df.write_parquet(membership_output)
    if len(waterfall_df) > 0:
        waterfall_df.write_parquet(waterfall_output)

    if verbose:
        print(f"\nSaved membership: {membership_output} ({membership_df.shape})")
        print(f"Saved waterfall: {waterfall_output} ({waterfall_df.shape})")

    return membership_df, waterfall_df


def main():
    parser = argparse.ArgumentParser(
        description="Stage 08b: Cohort Membership",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tracks cohort membership dynamics over time:
  - Signal-to-cohort assignments
  - Feature evolution within cohorts
  - Waterfall data for transition visualization

Example:
  python -m prism.entry_points.stage_08b_cohort_membership \\
      signal_vector.parquet --cohort-discovery cohort_discovery.parquet
"""
    )
    parser.add_argument('signal_vector', help='Path to signal_vector.parquet')
    parser.add_argument('--cohort-discovery', help='Path to cohort_discovery.parquet')
    parser.add_argument('--membership-output', default='cohort_membership.parquet',
                        help='Membership output path')
    parser.add_argument('--waterfall-output', default='signal_cohort_waterfall.parquet',
                        help='Waterfall output path')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.signal_vector,
        args.cohort_discovery,
        args.membership_output,
        args.waterfall_output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
