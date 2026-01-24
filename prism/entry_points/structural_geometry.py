#!/usr/bin/env python3
"""
Structural Geometry Entry Point
===============================

Computes Structural Geometry for all entities using WINDOWED computation.

Pipeline: signals → signal_typology → structural_geometry → dynamical_systems → causal_mechanics

Input:
    - data/observations.parquet (raw signals)

Output:
    - data/structural_geometry.parquet (geometry per entity per window)

Usage:
    python -m prism.entry_points.structural_geometry
    python -m prism.entry_points.structural_geometry --force
    python -m prism.entry_points.structural_geometry --domain hydraulic
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
    parser = argparse.ArgumentParser(description="Compute Structural Geometry (windowed)")
    parser.add_argument("--force", action="store_true", help="Recompute all")
    parser.add_argument("--domain", type=str, default="hydraulic", help="Domain for window config")
    parser.add_argument("--window", type=int, default=None, help="Override window size")
    parser.add_argument("--stride", type=int, default=None, help="Override stride")
    parser.add_argument("--testing", action="store_true", help="Enable testing mode")
    parser.add_argument("--entity", type=str, default=None, help="[TESTING] Only process specific entity")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    obs_path = data_dir / "observations.parquet"
    output_path = data_dir / "structural_geometry.parquet"

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

    # Get unique entities
    if "entity_id" not in df.columns:
        print("ERROR: observations.parquet must have entity_id column")
        sys.exit(1)

    entity_ids = df["entity_id"].unique().sort().to_list()
    print(f"  {len(entity_ids)} unique entities")

    # Testing mode filters
    if args.entity:
        if not args.testing:
            print("ERROR: --entity requires --testing flag")
            sys.exit(1)
        entity_ids = [e for e in entity_ids if e in args.entity.split(",")]

    # Import orchestrator
    from prism.structural_geometry import run_structural_geometry

    all_results = []

    for entity_id in entity_ids:
        # Get all signals for this entity
        entity_df = df.filter(pl.col("entity_id") == entity_id)
        signal_ids = entity_df["signal_id"].unique().sort().to_list()
        n_signals = len(signal_ids)

        if n_signals < 2:
            print(f"  {entity_id}: SKIP (only {n_signals} signal)")
            continue

        # Pivot to wide format
        try:
            pivot = entity_df.pivot(
                values="value",
                index="timestamp",
                on="signal_id"
            ).sort("timestamp")
        except Exception as e:
            print(f"  {entity_id}: SKIP (pivot failed: {e})")
            continue

        timestamps = pivot["timestamp"].to_list()
        signal_cols = [c for c in pivot.columns if c != "timestamp"]
        n_total = len(timestamps)

        # Calculate windows
        n_windows = max(1, (n_total - window_size) // stride + 1)
        print(f"  {entity_id}: {n_signals} signals, {n_total} obs → {n_windows} windows")

        for w in range(n_windows):
            start_idx = w * stride
            end_idx = start_idx + window_size

            if end_idx > n_total:
                break

            window_data = pivot.slice(start_idx, window_size)
            window_ts = timestamps[start_idx]

            # Extract signal matrix
            signals = window_data.select(signal_cols).to_numpy().T  # (n_signals, window_size)

            # Remove signals with NaN
            valid_mask = ~np.any(np.isnan(signals), axis=1)
            if valid_mask.sum() < 2:
                continue

            signals_valid = signals[valid_mask]
            signal_ids_valid = [signal_cols[j] for j in range(len(signal_cols)) if valid_mask[j]]

            try:
                result = run_structural_geometry(signals_valid, signal_ids_valid, entity_id)

                vector = result["vector"]
                row = {
                    "entity_id": entity_id,
                    "window_idx": w,
                    "timestamp": window_ts,
                    "window_start": start_idx,
                    "window_end": end_idx,
                    "n_signals": len(signal_ids_valid),

                    # Topology
                    "topology_class": result["topology"],
                    "stability_class": result["stability"],
                    "leadership_class": result["leadership"],

                    # Correlation
                    "mean_correlation": vector.get("mean_correlation", 0.0),
                    "median_correlation": vector.get("median_correlation", 0.0),
                    "correlation_dispersion": vector.get("correlation_dispersion", 0.0),
                    "variance_explained_1": vector.get("variance_explained_1", 0.0),
                    "effective_dimension": vector.get("effective_dimension", 1.0),

                    # Clustering
                    "n_clusters": vector.get("n_clusters", 1),
                    "silhouette_score": vector.get("silhouette_score", 0.0),

                    # Network
                    "network_density": vector.get("network_density", 0.0),
                    "mean_degree": vector.get("mean_degree", 0.0),
                    "transitivity": vector.get("transitivity", 0.0),
                    "n_hubs": vector.get("n_hubs", 0),

                    # Causality
                    "n_causal_pairs": vector.get("n_causal_pairs", 0),
                    "n_bidirectional": vector.get("n_bidirectional", 0),

                    # Decoupling
                    "n_decoupled_pairs": vector.get("n_decoupled_pairs", 0),
                    "decoupling_rate": vector.get("decoupling_rate", 0.0),

                    # Meta
                    "confidence": result.get("confidence", 0.0),
                }

                all_results.append(row)

                if w % 20 == 0 or w == n_windows - 1:
                    print(f"    window {w+1}/{n_windows}: {result['topology']} | {result['stability']}")

            except Exception as e:
                print(f"    window {w}: ERROR - {e}")
                continue

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
    print("STRUCTURAL GEOMETRY COMPLETE (WINDOWED)")
    print("=" * 60)
    print(f"  Windows processed: {len(results_df)}")
    print(f"  Window size: {window_size}, Stride: {stride}")
    print(f"  Output file: {output_path}")


if __name__ == "__main__":
    main()
