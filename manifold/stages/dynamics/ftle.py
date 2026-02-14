"""
Stage 08: FTLE Entry Point
==========================

Finite-Time Lyapunov Exponents - the correct name for what we compute.

Classical MLE assumes infinite time and ergodicity. ENGINES computes divergence
over finite windows and slides forward. That's FTLE by definition.

The FTLE framing unlocks capabilities:
    - Time-varying instability field (not just one number)
    - Ridges in FTLE field = Lagrangian Coherent Structures (LCS)
    - LCS = regime boundaries, transition corridors, failure basins

Same math that revealed the Interplanetary Transport Network in astrodynamics.
Different planets: your bearings, pumps, turbines.

Inputs:
    - observations.parquet

Output:
    - ftle.parquet (forward FTLE)
    - ftle_backward.parquet (backward FTLE, optional)

ENGINES computes FTLE, ORTHON interprets:
    Forward FTLE > 0:  Trajectories diverge (repelling structures)
    Backward FTLE > 0: Trajectories converge TO this point (attracting structures)

Bidirectional FTLE reveals full Lagrangian Coherent Structure:
    - Forward: where trajectories come FROM
    - Backward: where trajectories go TO
    - Combined: regime boundaries, transition corridors, failure basins
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
from typing import Optional

from manifold.core.dynamics.ftle import compute as compute_ftle
from manifold.core.dynamics.formal_definitions import classify_stability
from manifold.io.writer import write_output


def run(
    observations_path: str,
    data_path: str = ".",
    min_samples: int = 200,
    method: str = 'rosenstein',
    verbose: bool = True,
    intervention: dict = None,
    direction: str = 'forward',
) -> pl.DataFrame:
    """
    Compute FTLE (Finite-Time Lyapunov Exponents) for all signals.

    Args:
        observations_path: Path to observations.parquet
        data_path: Root data directory for output
        min_samples: Minimum samples required per signal
        method: 'rosenstein' or 'kantz'
        verbose: Print progress
        intervention: Optional intervention config dict with:
            - event_index: Sample index where intervention occurs
            When provided, computes pre/post intervention FTLE separately
        direction: 'forward' (default) or 'backward'
            - forward: Measures divergence going forward in time (repelling LCS)
            - backward: Measures divergence going backward in time (attracting LCS)

    Returns:
        FTLE DataFrame
    """
    backward = direction == 'backward'

    if verbose:
        print("=" * 70)
        print(f"STAGE 08: FTLE ({'BACKWARD' if backward else 'FORWARD'})")
        if backward:
            print("Backward FTLE - Attracting structures (where trajectories converge TO)")
        else:
            print("Forward FTLE - Repelling structures (where trajectories diverge FROM)")
        print("=" * 70)

    # Check for intervention mode
    intervention_enabled = intervention and intervention.get('enabled', False)
    event_index = intervention.get('event_index', 0) if intervention else 0

    if intervention_enabled and verbose:
        print(f"Intervention mode: event_index={event_index}")

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
                i_values = cohort_data['I'].to_numpy()

                if intervention_enabled:
                    # Split pre/post intervention
                    pre_mask = i_values < event_index
                    post_mask = i_values >= event_index

                    pre_values = values[pre_mask]
                    post_values = values[post_mask]

                    # For backward FTLE, reverse the time series
                    values_comp = values[::-1].copy() if backward else values
                    pre_values_comp = pre_values[::-1].copy() if backward else pre_values
                    post_values_comp = post_values[::-1].copy() if backward else post_values

                    # Compute FTLE for full, pre, and post
                    ftle_full = compute_ftle(values_comp, min_samples=min_samples, method=method)
                    ftle_pre = compute_ftle(pre_values_comp, min_samples=min(min_samples, 50), method=method) if len(pre_values) >= 20 else {'ftle': None, 'ftle_std': None, 'confidence': 0}
                    ftle_post = compute_ftle(post_values_comp, min_samples=min_samples, method=method) if len(post_values) >= min_samples else {'ftle': None, 'ftle_std': None, 'confidence': 0}

                    # Always use 'ftle' column name — direction column distinguishes
                    prefix = 'ftle'
                    ftle_val = ftle_full['ftle']
                    if ftle_val is not None:
                        stability = classify_stability(ftle_val).value
                    elif len(values) < min_samples:
                        stability = 'insufficient_data'
                    else:
                        stability = 'unknown'
                    results.append({
                        'signal_id': signal_id,
                        'cohort': cohort,
                        prefix: ftle_val,
                        f'{prefix}_std': ftle_full['ftle_std'],
                        f'{prefix}_pre': ftle_pre.get('ftle'),
                        f'{prefix}_post': ftle_post.get('ftle'),
                        f'{prefix}_delta': (ftle_post.get('ftle') - ftle_pre.get('ftle')) if ftle_pre.get('ftle') is not None and ftle_post.get('ftle') is not None else None,
                        'embedding_dim': ftle_full['embedding_dim'],
                        'embedding_tau': ftle_full['embedding_tau'],
                        'embedding_dim_method': ftle_full.get('embedding_dim_method'),
                        'tau_method': ftle_full.get('tau_method'),
                        'is_deterministic': ftle_full.get('is_deterministic'),
                        'E1_saturation_dim': ftle_full.get('E1_saturation_dim'),
                        'confidence': ftle_full['confidence'],
                        'n_samples': len(values),
                        'n_pre': len(pre_values),
                        'n_post': len(post_values),
                        'event_index': event_index,
                        'direction': direction,
                        'stability': stability,
                    })
                else:
                    # For backward FTLE, reverse the time series
                    values_comp = values[::-1].copy() if backward else values
                    ftle_result = compute_ftle(
                        values_comp,
                        min_samples=min_samples,
                        method=method,
                    )

                    prefix = 'ftle'
                    ftle_val = ftle_result['ftle']
                    if ftle_val is not None:
                        stability = classify_stability(ftle_val).value
                    elif len(values) < min_samples:
                        stability = 'insufficient_data'
                    else:
                        stability = 'unknown'
                    results.append({
                        'signal_id': signal_id,
                        'cohort': cohort,
                        prefix: ftle_val,
                        f'{prefix}_std': ftle_result['ftle_std'],
                        'embedding_dim': ftle_result['embedding_dim'],
                        'embedding_tau': ftle_result['embedding_tau'],
                        'embedding_dim_method': ftle_result.get('embedding_dim_method'),
                        'tau_method': ftle_result.get('tau_method'),
                        'is_deterministic': ftle_result.get('is_deterministic'),
                        'E1_saturation_dim': ftle_result.get('E1_saturation_dim'),
                        'confidence': ftle_result['confidence'],
                        'n_samples': len(values),
                        'direction': direction,
                        'stability': stability,
                    })
        else:
            values = signal_data['value'].to_numpy()

            # For backward FTLE, reverse the time series
            values_comp = values[::-1].copy() if backward else values
            ftle_result = compute_ftle(
                values_comp,
                min_samples=min_samples,
                method=method,
            )

            prefix = 'ftle'
            ftle_val = ftle_result['ftle']
            if ftle_val is not None:
                stability = classify_stability(ftle_val).value
            elif len(values) < min_samples:
                stability = 'insufficient_data'
            else:
                stability = 'unknown'
            results.append({
                'signal_id': signal_id,
                prefix: ftle_val,
                f'{prefix}_std': ftle_result['ftle_std'],
                'embedding_dim': ftle_result['embedding_dim'],
                'embedding_tau': ftle_result['embedding_tau'],
                'embedding_dim_method': ftle_result.get('embedding_dim_method'),
                'tau_method': ftle_result.get('tau_method'),
                'is_deterministic': ftle_result.get('is_deterministic'),
                'E1_saturation_dim': ftle_result.get('E1_saturation_dim'),
                'confidence': ftle_result['confidence'],
                'n_samples': len(values),
                'direction': direction,
                'stability': stability,
            })

    # Build DataFrame — infer schema from ALL rows to handle None → Float64
    result = pl.from_dicts(results, infer_schema_length=len(results)) if results else pl.DataFrame()

    stage_name = 'ftle_backward' if backward else 'ftle'

    if len(result) > 0:
        write_output(result, data_path, stage_name, verbose=verbose)

    if verbose:
        print(f"Shape: {result.shape}")

        if len(result) > 0 and 'ftle' in result.columns:
            valid = result.filter(pl.col('ftle').is_not_null())
            if len(valid) > 0:
                label = "Backward FTLE" if backward else "Forward FTLE"
                print(f"\n{label} stats (n={len(valid)}):")
                print(f"  Mean: {valid['ftle'].mean():.4f}")
                print(f"  Std:  {valid['ftle'].std():.4f}")
                print(f"  Range: [{valid['ftle'].min():.4f}, {valid['ftle'].max():.4f}]")

    return result


def run_bidirectional(
    observations_path: str,
    data_path: str,
    min_samples: int = 200,
    method: str = 'rosenstein',
    verbose: bool = True,
    intervention: dict = None,
) -> pl.DataFrame:
    """
    Compute both forward and backward FTLE and merge into bidirectional analysis.

    Reveals the full Lagrangian Coherent Structure:
        - Forward FTLE: Repelling structures (where trajectories diverge FROM)
        - Backward FTLE: Attracting structures (where trajectories converge TO)
        - LCS Strength: Total phase space stretching (ftle + ftle_bwd)
        - Asymmetry: Directional bias (ftle - ftle_bwd)

    Args:
        observations_path: Path to observations.parquet
        data_path: Root data directory for output
        min_samples: Minimum samples required per signal
        method: 'rosenstein' or 'kantz'
        verbose: Print progress
        intervention: Optional intervention config

    Returns:
        Merged bidirectional FTLE DataFrame
    """
    if verbose:
        print("=" * 70)
        print("BIDIRECTIONAL FTLE - Full LCS Analysis")
        print("=" * 70)

    # Run forward FTLE
    fwd = run(
        observations_path,
        data_path,
        min_samples=min_samples,
        method=method,
        verbose=verbose,
        intervention=intervention,
        direction='forward',
    )

    # Run backward FTLE
    bwd = run(
        observations_path,
        data_path,
        min_samples=min_samples,
        method=method,
        verbose=verbose,
        intervention=intervention,
        direction='backward',
    )

    # Merge forward and backward into single DataFrame via vstack
    # Both have same schema with 'ftle' column — 'direction' column distinguishes
    common_cols = sorted(set(fwd.columns) & set(bwd.columns))
    combined = pl.concat([fwd.select(common_cols), bwd.select(common_cols)], how='vertical')

    # Save merged result
    write_output(combined, data_path, 'ftle', verbose=verbose)

    if verbose:
        print()
        print("=" * 70)
        print("BIDIRECTIONAL FTLE COMPLETE")
        print("=" * 70)
        print(f"Shape: {combined.shape} (forward + backward)")

        for dir_name in ['forward', 'backward']:
            subset = combined.filter(pl.col('direction') == dir_name)
            valid = subset.filter(pl.col('ftle').is_not_null())
            if len(valid) > 0:
                print(f"\n{dir_name.title()} FTLE stats (n={len(valid)}):")
                print(f"  Mean: {valid['ftle'].mean():.4f}")
                print(f"  Std:  {valid['ftle'].std():.4f}")

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Stage 08: FTLE (Finite-Time Lyapunov Exponents)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes Finite-Time Lyapunov Exponents for all signals.

FTLE measures sensitivity to initial conditions over finite time:
  FTLE > 0: Trajectories diverge (instability)
  FTLE ≈ 0: Trajectories parallel (quasi-periodic)
  FTLE < 0: Trajectories converge (stable)

Unlike classical Lyapunov exponents, FTLE:
  - Works on finite windows → time-varying field
  - Handles transient, non-stationary data
  - Ridges = regime boundaries (Lagrangian Coherent Structures)

Example:
  python -m engines.entry_points.stage_08_ftle observations.parquet -o ftle.parquet
  python -m engines.entry_points.stage_08_ftle observations.parquet --backward
  python -m engines.entry_points.stage_08_ftle observations.parquet --bidirectional
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('--min-samples', type=int, default=200,
                        help='Minimum samples per signal (default: 200)')
    parser.add_argument('--method', choices=['rosenstein', 'kantz'], default='rosenstein',
                        help='Algorithm (default: rosenstein)')
    parser.add_argument('-d', '--data-path', default='.',
                        help='Root data directory (default: .)')
    parser.add_argument('--backward', action='store_true',
                        help='Compute backward FTLE (attracting structures)')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Compute both forward and backward FTLE with LCS metrics')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    if args.bidirectional:
        run_bidirectional(
            args.observations,
            args.data_path,
            min_samples=args.min_samples,
            method=args.method,
            verbose=not args.quiet,
        )
        return

    direction = 'backward' if args.backward else 'forward'

    run(
        args.observations,
        args.data_path,
        min_samples=args.min_samples,
        method=args.method,
        verbose=not args.quiet,
        direction=direction,
    )


if __name__ == "__main__":
    main()
