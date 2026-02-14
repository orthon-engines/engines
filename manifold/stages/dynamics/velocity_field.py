"""
Stage 21: Velocity Field Entry Point
====================================

Computes the velocity vector of the system state at each index I.
Direction, speed, acceleration, and curvature of the trajectory
through state space.

Mathematical Foundation:
    Velocity:     v(I) = x(I+1) - x(I)     [vector - direction + magnitude]
    Speed:        s(I) = |v(I)|             [scalar - how fast]
    Direction:    d(I) = v(I) / |v(I)|      [unit vector - which way]
    Acceleration: a(I) = v(I+1) - v(I)      [vector - speeding up? turning?]
    Curvature:    κ(I) = |a_perp| / |v|²    [scalar - how sharply turning]

Inputs:
    - observations.parquet (z-scored recommended)

Output:
    - velocity_field.parquet

Key outputs:
    - speed: how fast state is changing
    - dominant_motion_signal: which signal drives current motion
    - curvature: how sharply the trajectory is bending
    - motion_dimensionality: how many signals participate in motion
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
from typing import Optional
from scipy.signal import savgol_filter

from manifold.io.writer import write_output, write_sidecar


def run(
    observations_path: str,
    data_path: str = ".",
    smooth: str = 'savgol',
    smooth_window: int = 11,
    include_components: bool = True,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute state-space velocity field.

    Args:
        observations_path: Path to observations.parquet
        data_path: Root data directory for output
        smooth: Smoothing method ('none', 'savgol', 'gaussian')
        smooth_window: Window size for smoothing
        include_components: Include per-signal velocity components (v_{signal} columns).
                           Required for visualization projection. Default: True.
        verbose: Print progress

    Returns:
        velocity_field DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 21: VELOCITY FIELD")
        print("State-space velocity: direction, speed, curvature")
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
        print(f"Smoothing: {smooth}")

    results = []
    component_rows = []  # Narrow sidecar for per-signal velocity components

    for cohort_idx, cohort in enumerate(cohorts):
        if verbose and cohort_idx % 10 == 0:
            print(f"  Processing cohort {cohort_idx + 1}/{len(cohorts)}: {cohort}")

        if has_cohort:
            cohort_data = obs.filter(pl.col('cohort') == cohort)
        else:
            cohort_data = obs

        # Pivot to wide format
        try:
            wide = cohort_data.pivot(
                values='value',
                index='I',
                on='signal_id',
            ).sort('I')
        except Exception:
            continue

        if wide is None or len(wide) < 5:
            continue

        i_values = wide['I'].to_numpy()
        signal_cols = [c for c in wide.columns if c != 'I']
        x = wide.select(signal_cols).to_numpy().astype(float)

        # Z-score normalize each signal
        for j in range(x.shape[1]):
            col = x[:, j]
            valid = ~np.isnan(col)
            if valid.sum() > 1:
                mean = np.nanmean(col)
                std = np.nanstd(col)
                if std > 1e-10:
                    x[:, j] = (col - mean) / std

        # Smooth before differentiation
        if smooth == 'savgol' and len(x) > smooth_window:
            for j in range(x.shape[1]):
                valid = ~np.isnan(x[:, j])
                if valid.sum() > smooth_window:
                    x[valid, j] = savgol_filter(x[valid, j], min(smooth_window, valid.sum() - 1 if valid.sum() % 2 == 0 else valid.sum()), 3)

        # Replace NaN with interpolated values for differentiation
        for j in range(x.shape[1]):
            col = x[:, j]
            nans = np.isnan(col)
            if nans.any() and not nans.all():
                col[nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], col[~nans])
                x[:, j] = col

        # Velocity: first difference
        v = np.diff(x, axis=0)  # (N-1, n_signals)
        speed = np.linalg.norm(v, axis=1)  # (N-1,)

        # Direction: normalized velocity
        direction = np.zeros_like(v)
        nonzero = speed > 1e-12
        direction[nonzero] = v[nonzero] / speed[nonzero, np.newaxis]

        # Acceleration: second difference
        a = np.diff(v, axis=0)  # (N-2, n_signals)
        accel_mag = np.linalg.norm(a, axis=1)  # (N-2,)

        # Process each timestep
        for i in range(len(a)):
            # Decompose acceleration into parallel (speed change) and perpendicular (turning)
            v_hat = direction[i]
            a_parallel_scalar = np.dot(a[i], v_hat)
            a_parallel = a_parallel_scalar * v_hat
            a_perp = a[i] - a_parallel
            a_perp_mag = np.linalg.norm(a_perp)

            # Curvature: |a_perp| / |v|²
            curvature = a_perp_mag / (speed[i] ** 2 + 1e-12)

            # Dominant motion signal: which signal contributes most to velocity
            dominant_idx = np.argmax(np.abs(v[i]))
            dominant_signal = signal_cols[dominant_idx]
            dominant_fraction = np.abs(direction[i, dominant_idx])

            # Dominant acceleration signal
            dominant_accel_idx = np.argmax(np.abs(a[i]))
            dominant_accel_signal = signal_cols[dominant_accel_idx]

            # Motion dimensionality: entropy of squared direction components
            dir_sq = direction[i] ** 2 + 1e-12
            dir_sq = dir_sq / dir_sq.sum()  # Normalize
            motion_dim = np.exp(-np.sum(dir_sq * np.log(dir_sq + 1e-12)))

            row = {
                'I': int(i_values[i + 1]),  # Use I of the second point in difference
                'cohort': cohort,
                'speed': float(speed[i]),
                'acceleration_magnitude': float(accel_mag[i]),
                'acceleration_parallel': float(a_parallel_scalar),
                'acceleration_perpendicular': float(a_perp_mag),
                'curvature': float(curvature),
                'dominant_motion_signal': dominant_signal,
                'dominant_motion_fraction': float(dominant_fraction),
                'dominant_accel_signal': dominant_accel_signal,
                'motion_dimensionality': float(motion_dim),
            }

            # Collect per-signal velocity components into narrow sidecar
            if include_components:
                for j, sig in enumerate(signal_cols):
                    component_rows.append({
                        'I': int(i_values[i + 1]),
                        'cohort': cohort,
                        'signal_id': sig,
                        'velocity': float(v[i, j]),
                    })

            results.append(row)

    # Build output
    if results:
        result = pl.DataFrame(results)
    else:
        result = pl.DataFrame(schema={
            'I': pl.Int64,
            'cohort': pl.Utf8,
            'speed': pl.Float64,
            'acceleration_magnitude': pl.Float64,
            'acceleration_parallel': pl.Float64,
            'acceleration_perpendicular': pl.Float64,
            'curvature': pl.Float64,
            'dominant_motion_signal': pl.Utf8,
            'dominant_motion_fraction': pl.Float64,
            'dominant_accel_signal': pl.Utf8,
            'motion_dimensionality': pl.Float64,
        })

    write_output(result, data_path, 'velocity_field', verbose=verbose)

    # Write velocity components sidecar (narrow schema)
    if component_rows:
        components_df = pl.DataFrame(component_rows)
        write_sidecar(components_df, data_path, 'velocity_field', 'components', verbose=verbose)

    if verbose:
        print(f"Shape: {result.shape}")

        if len(result) > 0:
            print("\nVelocity field stats:")
            print(f"  Mean speed: {result['speed'].mean():.4f}")
            print(f"  Max speed:  {result['speed'].max():.4f}")
            print(f"  Mean curvature: {result['curvature'].mean():.4f}")
            print(f"  Mean motion dimensionality: {result['motion_dimensionality'].mean():.2f}")

            # Most common dominant motion signals
            top_drivers = (
                result
                .group_by('dominant_motion_signal')
                .agg(pl.len().alias('count'))
                .sort('count', descending=True)
                .head(5)
            )
            print("\nTop motion-driving signals:")
            for r in top_drivers.iter_rows(named=True):
                print(f"  {r['dominant_motion_signal']}: {r['count']} timesteps")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 21: Velocity Field (State-Space Motion)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes state-space velocity, acceleration, and curvature at each I.

Key outputs:
  speed:                    How fast state is changing
  acceleration_parallel:    Speeding up (+) or slowing down (-)
  acceleration_perpendicular: Turning force
  curvature:                How sharply trajectory is bending
  dominant_motion_signal:   Which signal drives current motion
  motion_dimensionality:    How many signals participate

Example:
  python -m engines.entry_points.stage_21_velocity_field \\
      observations.parquet -o velocity_field.parquet
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('-d', '--data-path', default='.',
                        help='Root data directory (default: .)')
    parser.add_argument('--smooth', choices=['none', 'savgol', 'gaussian'], default='savgol',
                        help='Smoothing method (default: savgol)')
    parser.add_argument('--smooth-window', type=int, default=11,
                        help='Smoothing window size (default: 11)')
    parser.add_argument('--no-components', action='store_true',
                        help='Exclude per-signal velocity components (reduces output size)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        args.data_path,
        smooth=args.smooth,
        smooth_window=args.smooth_window,
        include_components=not args.no_components,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
