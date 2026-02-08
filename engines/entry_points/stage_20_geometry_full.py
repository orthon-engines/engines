"""
Stage 20: Full-Span Geometry Entry Point
========================================

Expanding-window eigendecomposition. At each I from min_window to N,
computes eigendecomp on ALL data from I=0 to I.

This gives the CUMULATIVE geometry trajectory - how the system's
geometric character evolves as more data is observed.

Unlike windowed state_geometry (LOCAL snapshots), geometry_full
captures the continuous ARC of dimensional evolution.

Inputs:
    - observations.parquet

Output:
    - geometry_full.parquet

Key insight: A system going from startup → normal → degradation → failure
has ONE trajectory. Windowing fragments it. Full-span preserves the arc.
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
from typing import Optional

from engines.manifold.state.eigendecomp import (
    compute as compute_eigendecomp,
    enforce_eigenvector_continuity,
    bootstrap_effective_dim,
)


def run(
    observations_path: str,
    output_path: str = "geometry_full.parquet",
    min_window: int = 16,
    stride: int = 1,
    include_eigenvectors: bool = True,
    n_eigenvectors: int = 3,
    include_bootstrap_ci: bool = False,
    n_bootstrap: int = 50,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute expanding-window geometry trajectory.

    Args:
        observations_path: Path to observations.parquet
        output_path: Output path for geometry_full.parquet
        min_window: Minimum samples before first eigendecomp
        stride: Compute every N samples (1 = every sample)
        include_eigenvectors: If True, include eigenvector_{pc}_{signal} columns
        n_eigenvectors: Number of principal components to store eigenvectors for
        include_bootstrap_ci: If True, compute bootstrap confidence intervals for eff_dim
        n_bootstrap: Number of bootstrap samples (default 50 for speed)
        verbose: Print progress

    Returns:
        geometry_full DataFrame with eigenvalues, eigenvectors, and optional CI at each I
    """
    if verbose:
        print("=" * 70)
        print("STAGE 20: FULL-SPAN GEOMETRY")
        print("Expanding-window eigendecomposition trajectory")
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
        print(f"Min window: {min_window}, Stride: {stride}")

    results = []

    for cohort_idx, cohort in enumerate(cohorts):
        if verbose and cohort_idx % 10 == 0:
            print(f"  Processing cohort {cohort_idx + 1}/{len(cohorts)}: {cohort}")

        if has_cohort:
            cohort_data = obs.filter(pl.col('cohort') == cohort)
        else:
            cohort_data = obs

        # Pivot to wide format: rows = I, columns = signals
        try:
            wide = cohort_data.pivot(
                values='value',
                index='I',
                on='signal_id',
            ).sort('I')
        except Exception:
            continue

        if wide is None or len(wide) < min_window:
            continue

        i_values = wide['I'].to_numpy()
        signal_cols = [c for c in wide.columns if c != 'I']
        data_matrix = wide.select(signal_cols).to_numpy()

        # Track previous values for continuity
        prev_eff_dim = None
        prev_prev_eff_dim = None
        prev_eigenvectors = None

        # Expanding window: at each I, use all data from 0 to I
        for idx in range(min_window - 1, len(data_matrix), stride):
            # Expanding window: all data up to and including idx
            window = data_matrix[:idx + 1]

            # Remove rows with NaN
            valid_rows = ~np.isnan(window).any(axis=1)
            window = window[valid_rows]

            if len(window) < min_window:
                continue

            # Compute eigendecomp
            eigen = compute_eigendecomp(window)

            if eigen['effective_dim'] is None or np.isnan(eigen['effective_dim']):
                continue

            eff_dim = eigen['effective_dim']
            eigenvalues = eigen['eigenvalues']

            # Compute velocity (first derivative)
            if prev_eff_dim is not None:
                eff_dim_velocity = eff_dim - prev_eff_dim
            else:
                eff_dim_velocity = 0.0

            # Compute acceleration (second derivative)
            if prev_prev_eff_dim is not None and prev_eff_dim is not None:
                eff_dim_acceleration = (eff_dim - prev_eff_dim) - (prev_eff_dim - prev_prev_eff_dim)
            else:
                eff_dim_acceleration = 0.0

            row = {
                'I': int(i_values[idx]),
                'cohort': cohort,
                'effective_dim': eff_dim,
                'effective_dim_velocity': eff_dim_velocity,
                'effective_dim_acceleration': eff_dim_acceleration,
                'eigenvalue_1': float(eigenvalues[0]) if len(eigenvalues) > 0 else None,
                'eigenvalue_2': float(eigenvalues[1]) if len(eigenvalues) > 1 else None,
                'eigenvalue_3': float(eigenvalues[2]) if len(eigenvalues) > 2 else None,
                'explained_1': float(eigen['explained_ratio'][0]) if eigen['explained_ratio'] is not None and len(eigen['explained_ratio']) > 0 else None,
                'ratio_2_1': eigen['ratio_2_1'],
                'condition_number': eigen['condition_number'],
                'total_variance': eigen['total_variance'],
                'n_samples': idx + 1,
            }

            # Add bootstrap confidence intervals
            if include_bootstrap_ci:
                bootstrap_result = bootstrap_effective_dim(window, n_bootstrap=n_bootstrap)
                row['eff_dim_std'] = bootstrap_result['eff_dim_std']
                row['eff_dim_ci_low'] = bootstrap_result['eff_dim_ci_low']
                row['eff_dim_ci_high'] = bootstrap_result['eff_dim_ci_high']

            # Add eigenvectors for visualization projection
            # principal_components is Vt from SVD: shape (min(n,d), d)
            # Each row is a PC, each column is a signal dimension
            if include_eigenvectors and eigen.get('principal_components') is not None:
                pcs = eigen['principal_components']

                # Enforce eigenvector continuity to prevent sign flips
                if prev_eigenvectors is not None and pcs is not None:
                    pcs_T = pcs.T  # Convert to column vectors
                    prev_T = prev_eigenvectors.T
                    pcs_T = enforce_eigenvector_continuity(pcs_T, prev_T)
                    pcs = pcs_T.T  # Convert back to row vectors

                for pc_idx in range(min(n_eigenvectors, len(pcs))):
                    pc_vector = pcs[pc_idx]
                    for sig_idx, sig_name in enumerate(signal_cols[:len(pc_vector)]):
                        row[f'eigenvector_{pc_idx+1}_{sig_name}'] = float(pc_vector[sig_idx])

                prev_eigenvectors = pcs

            results.append(row)

            prev_prev_eff_dim = prev_eff_dim
            prev_eff_dim = eff_dim

    # Build output
    if results:
        result = pl.DataFrame(results)
    else:
        result = pl.DataFrame(schema={
            'I': pl.Int64,
            'cohort': pl.Utf8,
            'effective_dim': pl.Float64,
            'effective_dim_velocity': pl.Float64,
            'effective_dim_acceleration': pl.Float64,
            'eigenvalue_1': pl.Float64,
            'eigenvalue_2': pl.Float64,
            'eigenvalue_3': pl.Float64,
            'explained_1': pl.Float64,
            'ratio_2_1': pl.Float64,
            'condition_number': pl.Float64,
            'total_variance': pl.Float64,
            'n_samples': pl.Int64,
        })

    result.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {result.shape}")

        if len(result) > 0:
            print("\nEffective dimension trajectory stats:")
            print(f"  Start: {result.filter(pl.col('I') == result['I'].min())['effective_dim'].mean():.2f}")
            print(f"  End:   {result.filter(pl.col('I') == result['I'].max())['effective_dim'].mean():.2f}")
            print(f"  Mean velocity: {result['effective_dim_velocity'].mean():.4f}")

        print()
        print("─" * 50)
        print(f"✓ {Path(output_path).absolute()}")
        print("─" * 50)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 20: Full-Span Geometry (Expanding Window)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes eigendecomp at each I using all data from 0 to I.

Unlike windowed state_geometry (local snapshots), this gives the
continuous trajectory of dimensional evolution.

Output schema:
  I, cohort, effective_dim, effective_dim_velocity, effective_dim_acceleration,
  eigenvalue_1, eigenvalue_2, eigenvalue_3, explained_1, ratio_2_1,
  condition_number, total_variance, n_samples

Example:
  python -m engines.entry_points.stage_20_geometry_full \\
      observations.parquet -o geometry_full.parquet --min-window 16
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('-o', '--output', default='geometry_full.parquet',
                        help='Output path (default: geometry_full.parquet)')
    parser.add_argument('--min-window', type=int, default=16,
                        help='Minimum samples before first eigendecomp (default: 16)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Compute every N samples (default: 1)')
    parser.add_argument('--no-eigenvectors', action='store_true',
                        help='Exclude eigenvector columns (reduces output size)')
    parser.add_argument('--n-eigenvectors', type=int, default=3,
                        help='Number of PCs to store eigenvectors for (default: 3)')
    parser.add_argument('--bootstrap-ci', action='store_true',
                        help='Include bootstrap confidence intervals for eff_dim')
    parser.add_argument('--n-bootstrap', type=int, default=50,
                        help='Number of bootstrap samples (default: 50)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        args.output,
        min_window=args.min_window,
        stride=args.stride,
        include_eigenvectors=not args.no_eigenvectors,
        n_eigenvectors=args.n_eigenvectors,
        include_bootstrap_ci=args.bootstrap_ci,
        n_bootstrap=args.n_bootstrap,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
