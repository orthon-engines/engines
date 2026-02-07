"""
Stage 08: FTLE Entry Point
==========================

Finite-Time Lyapunov Exponents - the correct name for what we compute.

Classical MLE assumes infinite time and ergodicity. PRISM computes divergence
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

PRISM computes FTLE, ORTHON interprets:
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

from prism.engines.dynamics.ftle import compute as compute_ftle


def run(
    observations_path: str,
    output_path: str = "ftle.parquet",
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
        output_path: Output path for ftle.parquet
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

                    # Use appropriate column prefix for forward/backward
                    prefix = 'ftle_bwd' if backward else 'ftle'
                    results.append({
                        'signal_id': signal_id,
                        'cohort': cohort,
                        prefix: ftle_full['ftle'],
                        f'{prefix}_std': ftle_full['ftle_std'],
                        f'{prefix}_pre': ftle_pre.get('ftle'),
                        f'{prefix}_post': ftle_post.get('ftle'),
                        f'{prefix}_delta': (ftle_post.get('ftle') - ftle_pre.get('ftle')) if ftle_pre.get('ftle') is not None and ftle_post.get('ftle') is not None else None,
                        'embedding_dim': ftle_full['embedding_dim'],
                        'embedding_tau': ftle_full['embedding_tau'],
                        'confidence': ftle_full['confidence'],
                        'n_samples': len(values),
                        'n_pre': len(pre_values),
                        'n_post': len(post_values),
                        'event_index': event_index,
                        'direction': direction,
                    })
                else:
                    # For backward FTLE, reverse the time series
                    values_comp = values[::-1].copy() if backward else values
                    ftle_result = compute_ftle(
                        values_comp,
                        min_samples=min_samples,
                        method=method,
                    )

                    prefix = 'ftle_bwd' if backward else 'ftle'
                    results.append({
                        'signal_id': signal_id,
                        'cohort': cohort,
                        prefix: ftle_result['ftle'],
                        f'{prefix}_std': ftle_result['ftle_std'],
                        'embedding_dim': ftle_result['embedding_dim'],
                        'embedding_tau': ftle_result['embedding_tau'],
                        'confidence': ftle_result['confidence'],
                        'n_samples': len(values),
                        'direction': direction,
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

            prefix = 'ftle_bwd' if backward else 'ftle'
            results.append({
                'signal_id': signal_id,
                prefix: ftle_result['ftle'],
                f'{prefix}_std': ftle_result['ftle_std'],
                'embedding_dim': ftle_result['embedding_dim'],
                'embedding_tau': ftle_result['embedding_tau'],
                'confidence': ftle_result['confidence'],
                'n_samples': len(values),
                'direction': direction,
            })

    # Build DataFrame
    result = pl.DataFrame(results) if results else pl.DataFrame()

    if len(result) > 0:
        result.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {result.shape}")

        col_name = 'ftle_bwd' if backward else 'ftle'
        if len(result) > 0 and col_name in result.columns:
            valid = result.filter(pl.col(col_name).is_not_null())
            if len(valid) > 0:
                label = "Backward FTLE" if backward else "Forward FTLE"
                print(f"\n{label} stats (n={len(valid)}):")
                print(f"  Mean: {valid[col_name].mean():.4f}")
                print(f"  Std:  {valid[col_name].std():.4f}")
                print(f"  Range: [{valid[col_name].min():.4f}, {valid[col_name].max():.4f}]")

        print()
        print("─" * 50)
        print(f"✓ {Path(output_path).absolute()}")
        print("─" * 50)

    return result


def run_bidirectional(
    observations_path: str,
    output_dir: str,
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
        output_dir: Directory for output files
        min_samples: Minimum samples required per signal
        method: 'rosenstein' or 'kantz'
        verbose: Print progress
        intervention: Optional intervention config

    Returns:
        Merged bidirectional FTLE DataFrame
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("BIDIRECTIONAL FTLE - Full LCS Analysis")
        print("=" * 70)

    # Run forward FTLE
    fwd = run(
        observations_path,
        str(output_dir / 'ftle.parquet'),
        min_samples=min_samples,
        method=method,
        verbose=verbose,
        intervention=intervention,
        direction='forward',
    )

    # Run backward FTLE
    bwd = run(
        observations_path,
        str(output_dir / 'ftle_backward.parquet'),
        min_samples=min_samples,
        method=method,
        verbose=verbose,
        intervention=intervention,
        direction='backward',
    )

    # Merge forward and backward
    join_cols = ['signal_id']
    if 'cohort' in fwd.columns:
        join_cols.append('cohort')

    # Select backward columns to merge (avoid duplicates)
    bwd_cols = [c for c in bwd.columns if c.startswith('ftle_bwd')]
    bwd_select = join_cols + bwd_cols

    combined = fwd.join(bwd.select(bwd_select), on=join_cols, how='left')

    # Compute LCS metrics
    ftle_col = 'ftle'
    bwd_col = 'ftle_bwd'

    if ftle_col in combined.columns and bwd_col in combined.columns:
        combined = combined.with_columns([
            # LCS strength: total stretching
            (pl.col(ftle_col) + pl.col(bwd_col)).alias('lcs_strength'),
            # Asymmetry: positive = more repelling, negative = more attracting
            (pl.col(ftle_col) - pl.col(bwd_col)).alias('ftle_asymmetry'),
        ])

    # Save bidirectional result
    output_path = output_dir / 'ftle_bidirectional.parquet'
    combined.write_parquet(output_path)

    if verbose:
        print()
        print("=" * 70)
        print("BIDIRECTIONAL FTLE COMPLETE")
        print("=" * 70)
        print(f"Saved: {output_path}")

        if 'lcs_strength' in combined.columns:
            valid = combined.filter(pl.col('lcs_strength').is_not_null())
            if len(valid) > 0:
                print(f"\nLCS metrics (n={len(valid)}):")
                print(f"  Forward FTLE:  mean={valid[ftle_col].mean():.4f}")
                print(f"  Backward FTLE: mean={valid[bwd_col].mean():.4f}")
                print(f"  LCS Strength:  mean={valid['lcs_strength'].mean():.4f}")
                print(f"  Asymmetry:     mean={valid['ftle_asymmetry'].mean():.4f}")

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
  python -m prism.entry_points.stage_08_ftle observations.parquet -o ftle.parquet
  python -m prism.entry_points.stage_08_ftle observations.parquet --backward
  python -m prism.entry_points.stage_08_ftle observations.parquet --bidirectional
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('--min-samples', type=int, default=200,
                        help='Minimum samples per signal (default: 200)')
    parser.add_argument('--method', choices=['rosenstein', 'kantz'], default='rosenstein',
                        help='Algorithm (default: rosenstein)')
    parser.add_argument('-o', '--output', default='ftle.parquet',
                        help='Output path (default: ftle.parquet)')
    parser.add_argument('--backward', action='store_true',
                        help='Compute backward FTLE (attracting structures)')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Compute both forward and backward FTLE with LCS metrics')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    if args.bidirectional:
        # Bidirectional mode: output to directory
        from pathlib import Path
        output_dir = Path(args.output).parent if args.output != 'ftle.parquet' else Path('.')
        run_bidirectional(
            args.observations,
            str(output_dir),
            min_samples=args.min_samples,
            method=args.method,
            verbose=not args.quiet,
        )
        return

    direction = 'backward' if args.backward else 'forward'

    run(
        args.observations,
        args.output,
        min_samples=args.min_samples,
        method=args.method,
        verbose=not args.quiet,
        direction=direction,
    )


if __name__ == "__main__":
    main()
