"""
Stage 23: Ridge Proximity Entry Point
=====================================

The CONVERGENCE point of all three pillars. Combines:
- FTLE field (dynamics) - where are the regime boundaries?
- State velocity (dynamics + geometry) - how fast and which direction?
- FTLE gradient - which way to the ridge?

Computes URGENCY = v · ∇FTLE - rate of approach to regime boundary.

A system can be:
- Near a ridge but moving parallel → High FTLE, low urgency. Stable for now.
- Far from ridge but heading straight for it → Low FTLE, HIGH urgency. Trouble coming.

The WARNING quadrant (low FTLE, positive urgency) is where this engine
earns its keep. Every other diagnostic says "fine". This says "not for long."

Inputs:
    - ftle_rolling.parquet (from stage_22)
    - velocity_field.parquet (from stage_21)

Output:
    - ridge_proximity.parquet

Urgency Classification:
    CRITICAL: Near ridge AND heading toward it
    ELEVATED: Near ridge but stable/retreating
    WARNING:  Far from ridge but heading toward it ← EARLY WARNING
    NOMINAL:  Far from ridge, not approaching
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
from typing import Optional


def run(
    ftle_rolling_path: str,
    velocity_field_path: str,
    output_path: str = "ridge_proximity.parquet",
    ridge_threshold: float = 0.05,
    urgency_threshold: float = 0.001,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute ridge proximity (urgency) metric.

    Args:
        ftle_rolling_path: Path to ftle_rolling.parquet
        velocity_field_path: Path to velocity_field.parquet
        output_path: Output path for ridge_proximity.parquet
        ridge_threshold: FTLE value considered "near ridge"
        urgency_threshold: Minimum urgency to be "approaching"
        verbose: Print progress

    Returns:
        ridge_proximity DataFrame with urgency metrics
    """
    if verbose:
        print("=" * 70)
        print("STAGE 23: RIDGE PROXIMITY")
        print("Urgency = velocity toward FTLE ridge")
        print("=" * 70)

    ftle_rolling = pl.read_parquet(ftle_rolling_path)
    velocity_field = pl.read_parquet(velocity_field_path)

    if verbose:
        print(f"Loaded ftle_rolling: {ftle_rolling.shape}")
        print(f"Loaded velocity_field: {velocity_field.shape}")

    results = []

    # Get unique cohorts
    cohorts = ftle_rolling['cohort'].unique().to_list()

    if verbose:
        print(f"Cohorts: {len(cohorts)}")

    for cohort_idx, cohort in enumerate(cohorts):
        if verbose and cohort_idx % 10 == 0:
            print(f"  Processing cohort {cohort_idx + 1}/{len(cohorts)}: {cohort}")

        cohort_ftle = ftle_rolling.filter(pl.col('cohort') == cohort)
        cohort_vel = velocity_field.filter(pl.col('cohort') == cohort)

        if len(cohort_vel) == 0:
            continue

        # Get I values that exist in both
        ftle_i = set(cohort_ftle['I'].to_list())
        vel_i = set(cohort_vel['I'].to_list())
        common_i = sorted(ftle_i & vel_i)

        if len(common_i) < 3:
            if verbose:
                print(f"  Skipping cohort {cohort}: only {len(common_i)} common I values between ftle_rolling and velocity_field")
            continue

        # Process each signal
        signals = cohort_ftle['signal_id'].unique().to_list()

        for signal_id in signals:
            sig_ftle = (
                cohort_ftle
                .filter(pl.col('signal_id') == signal_id)
                .sort('I')
            )

            if len(sig_ftle) < 3:
                continue

            ftle_values = sig_ftle['ftle'].to_numpy()
            ftle_i_arr = sig_ftle['I'].to_numpy()

            # Compute FTLE gradient (temporal gradient)
            ftle_gradient = np.gradient(ftle_values)

            for idx in range(len(ftle_i_arr)):
                i_val = int(ftle_i_arr[idx])

                # Get velocity at this I
                vel_row = cohort_vel.filter(pl.col('I') == i_val)
                if len(vel_row) == 0:
                    continue

                speed = vel_row['speed'].to_numpy()[0]
                ftle_val = float(ftle_values[idx])
                grad = float(ftle_gradient[idx])

                # Urgency: velocity component along FTLE gradient direction
                # Positive = heading toward higher FTLE = approaching ridge
                urgency = speed * np.sign(grad) * abs(grad)

                # Time-to-ridge estimate
                if urgency > urgency_threshold and abs(grad) > 1e-6:
                    ftle_remaining = max(0, ridge_threshold - ftle_val)
                    time_to_ridge = ftle_remaining / (abs(urgency) + 1e-12)
                else:
                    time_to_ridge = float('inf')

                # FTLE acceleration (is gradient steepening?)
                if idx > 0 and idx < len(ftle_gradient) - 1:
                    ftle_accel = ftle_gradient[idx] - ftle_gradient[idx - 1]
                else:
                    ftle_accel = 0.0

                # Classify urgency
                high_ftle = ftle_val > ridge_threshold
                approaching = urgency > urgency_threshold

                if high_ftle and approaching:
                    urgency_class = 'critical'
                elif high_ftle and not approaching:
                    urgency_class = 'elevated'
                elif not high_ftle and approaching:
                    urgency_class = 'warning'
                else:
                    urgency_class = 'nominal'

                results.append({
                    'I': i_val,
                    'cohort': cohort,
                    'signal_id': signal_id,
                    'ftle_current': ftle_val,
                    'ftle_gradient': grad,
                    'ftle_acceleration': float(ftle_accel),
                    'speed': float(speed),
                    'urgency': float(urgency),
                    'time_to_ridge': float(time_to_ridge) if np.isfinite(time_to_ridge) else float('inf'),
                    'urgency_class': urgency_class,
                })

    # Build output
    if results:
        result = pl.DataFrame(results, infer_schema_length=None)
    else:
        result = pl.DataFrame(schema={
            'I': pl.Int64,
            'cohort': pl.Utf8,
            'signal_id': pl.Utf8,
            'ftle_current': pl.Float64,
            'ftle_gradient': pl.Float64,
            'ftle_acceleration': pl.Float64,
            'speed': pl.Float64,
            'urgency': pl.Float64,
            'time_to_ridge': pl.Float64,
            'urgency_class': pl.Utf8,
        })

    result.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {result.shape}")

        if len(result) > 0:
            print("\nUrgency class distribution:")
            class_dist = result.group_by('urgency_class').agg(pl.len().alias('count')).sort('count', descending=True)
            for r in class_dist.iter_rows(named=True):
                pct = r['count'] / len(result) * 100
                print(f"  {r['urgency_class']:10}: {r['count']:>6} ({pct:.1f}%)")

            # Warning signals (early warning)
            warnings = result.filter(pl.col('urgency_class') == 'warning')
            if len(warnings) > 0:
                print(f"\nEARLY WARNING instances: {len(warnings)}")
                warning_signals = (
                    warnings
                    .group_by('signal_id')
                    .agg(pl.len().alias('count'))
                    .sort('count', descending=True)
                    .head(5)
                )
                print("Top warning signals:")
                for r in warning_signals.iter_rows(named=True):
                    print(f"  {r['signal_id']}: {r['count']} warnings")

        print()
        print("─" * 50)
        print(f"✓ {Path(output_path).absolute()}")
        print("─" * 50)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 23: Ridge Proximity (Urgency)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes urgency = velocity toward FTLE ridge.

Urgency classification:
  CRITICAL: Near ridge AND heading toward it
  ELEVATED: Near ridge but stable/retreating
  WARNING:  Far from ridge but heading toward it (EARLY WARNING)
  NOMINAL:  Far from ridge, not approaching

Requires both ftle_rolling.parquet and velocity_field.parquet.

Example:
  python -m engines.entry_points.stage_23_ridge_proximity \\
      ftle_rolling.parquet velocity_field.parquet -o ridge_proximity.parquet
"""
    )
    parser.add_argument('ftle_rolling', help='Path to ftle_rolling.parquet')
    parser.add_argument('velocity_field', help='Path to velocity_field.parquet')
    parser.add_argument('-o', '--output', default='ridge_proximity.parquet',
                        help='Output path (default: ridge_proximity.parquet)')
    parser.add_argument('--ridge-threshold', type=float, default=0.05,
                        help='FTLE value considered "near ridge" (default: 0.05)')
    parser.add_argument('--urgency-threshold', type=float, default=0.001,
                        help='Minimum urgency to be "approaching" (default: 0.001)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.ftle_rolling,
        args.velocity_field,
        args.output,
        ridge_threshold=args.ridge_threshold,
        urgency_threshold=args.urgency_threshold,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
