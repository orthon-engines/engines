#!/usr/bin/env python3
"""
Causal Mechanics Entry Point
============================

Computes Causal Mechanics (physics-inspired analysis) using WINDOWED computation.

Pipeline: signals → signal_typology → structural_geometry → dynamical_systems → causal_mechanics

Input:
    - data/observations.parquet (raw signals)

Output:
    - data/causal_mechanics.parquet (mechanics per signal per window)

Usage:
    python -m prism.entry_points.causal_mechanics
    python -m prism.entry_points.causal_mechanics --force
    python -m prism.entry_points.causal_mechanics --domain hydraulic
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import polars as pl
import yaml


def load_window_config(domain: str = None) -> dict:
    """Load window configuration from stride.yaml."""
    config_path = Path("config/stride.yaml")
    if not config_path.exists():
        return {"window": 200, "stride": 20, "min_obs": 50}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Check for domain-specific config
    if domain and "domain_windows" in config:
        if domain in config["domain_windows"]:
            domain_cfg = config["domain_windows"][domain]
            return {
                "window": domain_cfg.get("window", 200),
                "stride": domain_cfg.get("stride", 20),
                "min_obs": domain_cfg.get("min_obs", 50),
            }

    # Fall back to default
    return {"window": 200, "stride": 20, "min_obs": 50}


def main():
    parser = argparse.ArgumentParser(description="Compute Causal Mechanics (windowed)")
    parser.add_argument("--force", action="store_true", help="Recompute all")
    parser.add_argument("--domain", type=str, default="hydraulic", help="Domain for window config")
    parser.add_argument("--window", type=int, default=None, help="Override window size")
    parser.add_argument("--stride", type=int, default=None, help="Override stride")
    parser.add_argument("--testing", action="store_true", help="Enable testing mode")
    parser.add_argument("--signal", type=str, default=None, help="[TESTING] Only process specific signal")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    obs_path = data_dir / "observations.parquet"
    output_path = data_dir / "causal_mechanics.parquet"

    # Load window config
    win_cfg = load_window_config(args.domain)
    window_size = args.window or win_cfg["window"]
    stride = args.stride or win_cfg["stride"]
    min_obs = win_cfg["min_obs"]

    print(f"Window config: window={window_size}, stride={stride}, min_obs={min_obs}")

    # Check for observations
    if not obs_path.exists():
        print(f"ERROR: {obs_path} not found")
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

    # Import orchestrator
    from prism.causal_mechanics import run_causal_mechanics

    all_results = []

    for signal_id in signal_ids:
        # Extract signal data
        signal_df = df.filter(pl.col("signal_id") == signal_id).sort("timestamp")
        values = signal_df["value"].to_numpy()
        n_total = len(values)

        # Get entity_id
        entity_id = "unknown"
        if "entity_id" in signal_df.columns:
            entity_ids = signal_df["entity_id"].unique().to_list()
            if entity_ids:
                entity_id = entity_ids[0]

        # Calculate windows
        n_windows = max(1, (n_total - window_size) // stride + 1)

        if n_total < min_obs:
            print(f"  {signal_id}: SKIP ({n_total} < {min_obs} min_obs)")
            continue

        print(f"  {signal_id}: {n_total} samples → {n_windows} windows")

        for w in range(n_windows):
            start_idx = w * stride
            end_idx = start_idx + window_size

            if end_idx > n_total:
                break

            window_values = values[start_idx:end_idx]

            if len(window_values) < 30:
                continue

            try:
                result = run_causal_mechanics(
                    window_values,
                    entity_id=entity_id,
                    signal_id=signal_id
                )

                vector = result.get("vector", {})
                row = {
                    "signal_id": signal_id,
                    "entity_id": entity_id,
                    "window_idx": w,
                    "window_start": start_idx,
                    "window_end": end_idx,
                    "n_samples": len(window_values),

                    # Classifications
                    "energy_class": result.get("energy_class", "UNDETERMINED"),
                    "equilibrium_class": result.get("equilibrium_class", "UNDETERMINED"),
                    "flow_class": result.get("flow_class", "UNDETERMINED"),
                    "orbit_class": result.get("orbit_class", "UNDETERMINED"),
                    "dominant_energy": result.get("dominant_energy", "UNDETERMINED"),
                    "motion_class": result.get("motion_class", "UNDETERMINED"),
                    "system_class": result.get("system_class", "Unknown"),

                    # Hamiltonian (Energy)
                    "H_mean": vector.get("H_mean", 0.0),
                    "H_std": vector.get("H_std", 0.0),
                    "H_trend": vector.get("H_trend", 0.0),
                    "H_cv": vector.get("H_cv", 0.0),
                    "T_V_ratio": vector.get("T_V_ratio", 1.0),
                    "energy_conserved": vector.get("energy_conserved", False),

                    # Gibbs (Equilibrium)
                    "G_mean": vector.get("G_mean", 0.0),
                    "G_trend": vector.get("G_trend", 0.0),
                    "delta_G": vector.get("delta_G", 0.0),
                    "spontaneous": vector.get("spontaneous", False),

                    # Angular Momentum (Cycles)
                    "angular_L_mean": vector.get("angular_L_mean", 0.0),
                    "orbit_circularity": vector.get("orbit_circularity", 0.0),

                    # Momentum Flux (Flow)
                    "reynolds_proxy": vector.get("reynolds_proxy", 0.0),
                    "turbulence_intensity": vector.get("turbulence_intensity", 0.0),

                    # Meta
                    "confidence": result.get("confidence", 0.0),
                }

                all_results.append(row)

            except Exception as e:
                if w == 0:
                    print(f"    window 0 ERROR: {e}")
                continue

        # Summary for this signal
        signal_results = [r for r in all_results if r["signal_id"] == signal_id]
        if signal_results:
            energy_classes = [r["energy_class"] for r in signal_results]
            most_common = max(set(energy_classes), key=energy_classes.count)
            print(f"    → {len(signal_results)} windows, dominant: {most_common}")

    if not all_results:
        print("\nNo windows processed successfully")
        return

    # Convert to DataFrame
    results_df = pl.DataFrame(all_results)

    # Write output
    print(f"\nWriting {len(results_df)} rows to {output_path}")
    results_df.write_parquet(output_path)

    # Summary
    print("\n" + "=" * 60)
    print("CAUSAL MECHANICS COMPLETE (WINDOWED)")
    print("=" * 60)
    print(f"  Total windows: {len(results_df)}")
    print(f"  Window size: {window_size}, Stride: {stride}")
    print(f"  Output file: {output_path}")

    # Show distribution
    if len(results_df) > 0:
        print("\nEnergy Class Distribution:")
        for energy in results_df["energy_class"].unique().to_list():
            count = len(results_df.filter(pl.col("energy_class") == energy))
            pct = 100 * count / len(results_df)
            print(f"  {energy}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
