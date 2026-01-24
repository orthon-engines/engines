#!/usr/bin/env python3
"""
Signal Typology Entry Point
===========================

Computes Signal Typology for all signals in observations.parquet.

Output:
    - data/signal_typology.parquet (metrics + normalized profile in one file)

Schema:
    - signal_id, entity_id, unit_id, timestamp
    - Raw metrics: hurst_exponent, permutation_entropy, spectral_centroid, ...
    - Normalized profile: memory, information, frequency, volatility, ...
    - Classification: typology_class, dominant_axis, secondary_axis

Usage:
    python -m prism.entry_points.signal_typology
    python -m prism.entry_points.signal_typology --force
    python -m prism.entry_points.signal_typology --testing --limit 100
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl


def main():
    parser = argparse.ArgumentParser(description="Compute Signal Typology")
    parser.add_argument("--force", action="store_true", help="Recompute all signals")
    parser.add_argument("--testing", action="store_true", help="Enable testing mode")
    parser.add_argument("--limit", type=int, default=None, help="[TESTING] Max observations per signal")
    parser.add_argument("--signal", type=str, default=None, help="[TESTING] Only process specific signal")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    obs_path = data_dir / "observations.parquet"
    output_path = data_dir / "signal_typology.parquet"

    # Check for observations
    if not obs_path.exists():
        print(f"ERROR: {obs_path} not found")
        print("Run a fetcher first (e.g., python -m fetchers.hydraulic_fetcher)")
        sys.exit(1)

    print(f"Loading observations from {obs_path}")
    df = pl.read_parquet(obs_path)
    print(f"  {len(df):,} observations loaded")

    # Get unique signals
    signal_ids = df["signal_id"].unique().sort().to_list()
    print(f"  {len(signal_ids)} unique signals")

    # Testing mode filters
    if args.signal:
        if not args.testing:
            print("ERROR: --signal requires --testing flag")
            sys.exit(1)
        signal_ids = [s for s in signal_ids if s in args.signal.split(",")]
        print(f"  [TESTING] Filtered to {len(signal_ids)} signals")

    # Check for existing results
    existing_signals = set()
    if not args.force and output_path.exists():
        existing_df = pl.read_parquet(output_path)
        existing_signals = set(existing_df["signal_id"].unique().to_list())
        print(f"  {len(existing_signals)} signals already computed (use --force to recompute)")

    # Filter to signals that need processing
    signals_to_process = [s for s in signal_ids if s not in existing_signals]

    if not signals_to_process:
        print("All signals already processed")
        return

    print(f"\nProcessing {len(signals_to_process)} signals...")

    # Import orchestrator
    from prism.signal_typology.orchestrator import compute_metrics
    from prism.signal_typology.normalize import metrics_to_profile

    output_rows = []

    for i, signal_id in enumerate(signals_to_process):
        # Extract signal data
        signal_df = df.filter(pl.col("signal_id") == signal_id).sort("timestamp")

        # Get values
        values = signal_df["value"].to_numpy()

        # Apply limit in testing mode
        if args.testing and args.limit:
            values = values[:args.limit]

        n_samples = len(values)

        if n_samples < 30:
            print(f"  [{i+1}/{len(signals_to_process)}] {signal_id}: SKIP (only {n_samples} samples)")
            continue

        # Get entity_id and unit_id if available
        entity_id = "unknown"
        unit_id = "unknown"
        if "entity_id" in signal_df.columns:
            entity_ids = signal_df["entity_id"].unique().to_list()
            if entity_ids:
                entity_id = entity_ids[0]
                unit_id = entity_id  # Default unit_id to entity_id

        if "unit_id" in signal_df.columns:
            unit_ids = signal_df["unit_id"].unique().to_list()
            if unit_ids:
                unit_id = unit_ids[0]

        # Get timestamp range
        timestamps = signal_df["timestamp"].to_list()
        first_ts = timestamps[0] if timestamps else None
        last_ts = timestamps[-1] if timestamps else None

        # Compute metrics
        try:
            metrics = compute_metrics(values, signal_id)

            # Normalize to profile
            profile = metrics_to_profile(metrics)

            # Combine into single row
            row = {
                # Identifiers
                "signal_id": signal_id,
                "entity_id": entity_id,
                "unit_id": unit_id,
                "timestamp_start": first_ts,
                "timestamp_end": last_ts,
                "n_samples": n_samples,
            }

            # Add raw metrics (prefixed with metric_)
            for key, value in metrics.items():
                if key != "signal_id":
                    row[f"metric_{key}"] = value

            # Add normalized profile (axis scores 0-1)
            axis_names = ["memory", "information", "frequency", "volatility",
                         "wavelet", "derivatives", "recurrence", "discontinuity", "momentum"]
            for axis in axis_names:
                row[axis] = profile.get(axis, 0.5)

            # Compute classification
            dominant_axis, dominant_score = _get_dominant_axis(profile, axis_names)
            secondary_axis, secondary_score = _get_secondary_axis(profile, axis_names, dominant_axis)
            typology_class = _classify_typology(profile, axis_names)

            row["typology_class"] = typology_class
            row["dominant_axis"] = dominant_axis
            row["dominant_score"] = dominant_score
            row["secondary_axis"] = secondary_axis
            row["secondary_score"] = secondary_score

            output_rows.append(row)

            # Progress output
            print(f"  [{i+1}/{len(signals_to_process)}] {signal_id}: {n_samples:,} samples | "
                  f"{typology_class} (dom={dominant_axis}:{dominant_score:.2f})")

        except Exception as e:
            print(f"  [{i+1}/{len(signals_to_process)}] {signal_id}: ERROR - {e}")
            continue

    if not output_rows:
        print("\nNo signals processed successfully")
        return

    # Convert to DataFrame
    output_df = pl.DataFrame(output_rows)

    # Merge with existing if not forcing
    if existing_signals and not args.force:
        existing_df = pl.read_parquet(output_path)
        output_df = pl.concat([existing_df, output_df])

    # Write output
    print(f"\nWriting {len(output_df)} rows to {output_path}")
    output_df.write_parquet(output_path)

    # Summary
    print("\n" + "=" * 60)
    print("SIGNAL TYPOLOGY COMPLETE")
    print("=" * 60)
    print(f"  Signals processed: {len(output_rows)}")
    print(f"  Output file: {output_path}")
    print()

    # Show profile summary
    print("Profile Summary (0-1 axis scores):")
    axis_cols = ["memory", "information", "frequency", "volatility",
                 "wavelet", "derivatives", "recurrence", "discontinuity", "momentum"]

    for axis in axis_cols:
        if axis in output_df.columns:
            values = output_df[axis].to_numpy()
            values = values[~np.isnan(values)]
            if len(values) > 0:
                print(f"  {axis:15} mean={np.mean(values):.3f}  std={np.std(values):.3f}  "
                      f"range=[{np.min(values):.3f}, {np.max(values):.3f}]")

    # Show typology distribution
    print("\nTypology Distribution:")
    for tc in output_df["typology_class"].unique().sort().to_list():
        count = len(output_df.filter(pl.col("typology_class") == tc))
        print(f"  {tc}: {count}")


def _get_dominant_axis(profile: dict, axis_names: list) -> tuple:
    """Find the axis with highest deviation from 0.5."""
    max_deviation = 0
    dominant = "balanced"
    dominant_score = 0.5

    for axis in axis_names:
        score = profile.get(axis, 0.5)
        deviation = abs(score - 0.5)
        if deviation > max_deviation:
            max_deviation = deviation
            dominant = axis
            dominant_score = score

    return dominant, dominant_score


def _get_secondary_axis(profile: dict, axis_names: list, exclude: str) -> tuple:
    """Find the second-highest deviation axis."""
    max_deviation = 0
    secondary = "balanced"
    secondary_score = 0.5

    for axis in axis_names:
        if axis == exclude:
            continue
        score = profile.get(axis, 0.5)
        deviation = abs(score - 0.5)
        if deviation > max_deviation:
            max_deviation = deviation
            secondary = axis
            secondary_score = score

    return secondary, secondary_score


def _classify_typology(profile: dict, axis_names: list) -> str:
    """
    Classify signal into typology category.

    Categories based on dominant characteristics:
    - persistent: High memory (>0.7)
    - anti_persistent: Low memory (<0.3)
    - periodic: High frequency (>0.7)
    - chaotic: Low information/predictability (<0.3)
    - volatile: High volatility (>0.7)
    - discontinuous: High discontinuity (>0.7)
    - smooth: Low derivatives (<0.3)
    - complex: Multiple high axes
    - balanced: No strong characteristics
    """
    memory = profile.get("memory", 0.5)
    information = profile.get("information", 0.5)
    frequency = profile.get("frequency", 0.5)
    volatility = profile.get("volatility", 0.5)
    discontinuity = profile.get("discontinuity", 0.5)
    derivatives = profile.get("derivatives", 0.5)

    # Count extreme axes
    high_count = sum(1 for axis in axis_names if profile.get(axis, 0.5) > 0.7)
    low_count = sum(1 for axis in axis_names if profile.get(axis, 0.5) < 0.3)

    # Complex: multiple strong characteristics
    if high_count >= 3:
        return "complex"

    # Specific patterns
    if memory > 0.7:
        return "persistent"
    if memory < 0.3:
        return "anti_persistent"
    if frequency > 0.7:
        return "periodic"
    if information < 0.3:
        return "chaotic"
    if volatility > 0.7:
        return "volatile"
    if discontinuity > 0.7:
        return "discontinuous"
    if derivatives < 0.3:
        return "smooth"

    return "balanced"


if __name__ == "__main__":
    main()
