"""
Stage 31: Cohort Velocity Field Entry Point
============================================

Velocity of cohorts through feature space.
Same finite-difference math as stage_21_velocity_field, applied to
cohort trajectories over I windows.

Inputs:
    - cohort_vector.parquet

Output:
    - cohort_velocity_field.parquet
"""

import numpy as np
import polars as pl
from pathlib import Path

from manifold.io.writer import write_output


def run(
    cohort_vector_path: str,
    data_path: str = ".",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute velocity field for cohort trajectories through feature space.

    Args:
        cohort_vector_path: Path to cohort_vector.parquet
        data_path: Root data directory for output
        verbose: Print progress

    Returns:
        Cohort velocity field DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 31: COHORT VELOCITY FIELD")
        print("Velocity, acceleration, curvature of cohorts in feature space")
        print("=" * 70)

    cv = pl.read_parquet(cohort_vector_path)

    if verbose:
        print(f"Loaded cohort_vector: {cv.shape}")

    if len(cv) == 0:
        if verbose:
            print("  Empty cohort_vector — skipping")
        write_output(pl.DataFrame(), data_path, 'cohort_velocity_field', verbose=verbose)
        return pl.DataFrame()

    feature_cols = [c for c in cv.columns if c not in ['cohort', 'I']]
    cohorts = sorted(cv['cohort'].unique().to_list())

    if verbose:
        print(f"Cohorts: {len(cohorts)}")
        print(f"Feature dims: {len(feature_cols)}")

    results = []

    for cohort in cohorts:
        cohort_data = cv.filter(pl.col('cohort') == cohort).sort('I')

        if len(cohort_data) < 3:
            continue

        i_values = cohort_data['I'].to_numpy()
        x = cohort_data.select(feature_cols).to_numpy().astype(float)

        # Replace NaN with column means for differentiation
        for j in range(x.shape[1]):
            col = x[:, j]
            nans = np.isnan(col)
            if nans.any() and not nans.all():
                col[nans] = np.nanmean(col)
                x[:, j] = col
            elif nans.all():
                x[:, j] = 0.0

        # Velocity: first difference
        v = np.diff(x, axis=0)  # (N-1, D)
        speed = np.linalg.norm(v, axis=1)  # (N-1,)

        # Direction: normalized velocity
        direction = np.zeros_like(v)
        nonzero = speed > 1e-12
        direction[nonzero] = v[nonzero] / speed[nonzero, np.newaxis]

        # Acceleration: second difference
        a = np.diff(v, axis=0)  # (N-2, D)
        accel_mag = np.linalg.norm(a, axis=1)

        for i in range(len(a)):
            # Decompose acceleration into parallel and perpendicular
            v_hat = direction[i]
            a_parallel_scalar = np.dot(a[i], v_hat)
            a_perp = a[i] - a_parallel_scalar * v_hat
            a_perp_mag = np.linalg.norm(a_perp)

            # Curvature: |a_perp| / |v|^2
            curvature = a_perp_mag / (speed[i] ** 2 + 1e-12)

            # Dominant motion feature: which feature contributes most to velocity
            dominant_idx = np.argmax(np.abs(v[i]))
            dominant_feature = feature_cols[dominant_idx]

            # Motion dimensionality: entropy of squared direction components
            dir_sq = direction[i] ** 2 + 1e-12
            dir_sq = dir_sq / dir_sq.sum()
            motion_dim = np.exp(-np.sum(dir_sq * np.log(dir_sq + 1e-12)))

            results.append({
                'cohort': cohort,
                'I': int(i_values[i + 1]),  # I of second point in difference
                'speed': float(speed[i]),
                'acceleration_magnitude': float(accel_mag[i]),
                'acceleration_parallel': float(a_parallel_scalar),
                'acceleration_perpendicular': float(a_perp_mag),
                'curvature': float(curvature),
                'dominant_motion_feature': dominant_feature,
                'motion_dimensionality': float(motion_dim),
            })

    result = pl.DataFrame(results) if results else pl.DataFrame()

    write_output(result, data_path, 'cohort_velocity_field', verbose=verbose)

    if verbose:
        print(f"Shape: {result.shape}")
        if len(result) > 0:
            print(f"  Mean speed: {result['speed'].mean():.4f}")
            print(f"  Mean curvature: {result['curvature'].mean():.4f}")
            print(f"  Mean motion dimensionality: {result['motion_dimensionality'].mean():.2f}")

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 31: Cohort Velocity Field",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes velocity, acceleration, and curvature of cohort trajectories
through feature space over I windows.

Example:
  python -m engines.entry_points.stage_31_cohort_velocity_field \\
      cohort_vector.parquet -o cohort_velocity_field.parquet
"""
    )
    parser.add_argument('cohort_vector', help='Path to cohort_vector.parquet')
    parser.add_argument('-d', '--data-path', default='.',
                        help='Root data directory (default: .)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.cohort_vector,
        args.data_path,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
