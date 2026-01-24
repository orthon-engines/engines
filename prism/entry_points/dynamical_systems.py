#!/usr/bin/env python3
"""
Dynamical Systems Entry Point
=============================

Analyzes temporal evolution of manifold geometry windows.

Pipeline: signals → signal_typology → manifold_geometry → dynamical_systems → causal_mechanics

Input:
    - data/manifold_geometry.parquet (geometry per window)

Output:
    - data/dynamical_systems.parquet (state per entity per window)

Schema includes:
    - entity_id, unit_id, window_idx, timestamp
    - Raw metrics: correlation_level, stability_index, trajectory_speed, ...
    - Categorical states: trajectory, attractor
    - Numeric states: stability, predictability, coupling, memory
    - Classification: dynamics_class, state_string

Transitions are NOT stored separately - they can be identified by:
    1. Comparing consecutive rows (window_idx, window_idx+1)
    2. Applying thresholds from prism.config.thresholds

Usage:
    python -m prism.entry_points.dynamical_systems
    python -m prism.entry_points.dynamical_systems --force
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import polars as pl


# =============================================================================
# Metric Computation
# =============================================================================

def compute_trajectory(geometry_history: List[Dict], window_idx: int) -> str:
    """
    Classify trajectory based on geometry evolution.
    Uses correlation and density trends to determine direction.
    """
    if window_idx < 2:
        return "stationary"

    lookback = min(5, window_idx + 1)
    recent = geometry_history[window_idx - lookback + 1:window_idx + 1]

    if len(recent) < 2:
        return "stationary"

    correlations = [g.get("mean_correlation", 0) for g in recent]
    densities = [g.get("network_density", 0) for g in recent]

    x = np.arange(len(correlations))
    corr_slope = np.polyfit(x, correlations, 1)[0] if len(correlations) > 1 else 0
    density_slope = np.polyfit(x, densities, 1)[0] if len(densities) > 1 else 0

    corr_var = np.var(correlations) if len(correlations) > 1 else 0

    if corr_var > 0.05:
        return "chaotic"
    elif corr_slope > 0.02 and density_slope > 0.02:
        return "converging"
    elif corr_slope < -0.02 and density_slope < -0.02:
        return "diverging"
    elif abs(corr_slope) < 0.01 and corr_var < 0.01:
        if _detect_periodicity(correlations):
            return "periodic"
        return "stationary"
    else:
        return "stationary"


def _detect_periodicity(values: List[float], threshold: float = 0.7) -> bool:
    """Simple periodicity detection via autocorrelation."""
    if len(values) < 6:
        return False

    values = np.array(values)
    values = values - np.mean(values)

    for lag in [2, 3]:
        if len(values) > lag:
            autocorr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
            if abs(autocorr) > threshold:
                return True
    return False


def compute_stability(geometry_history: List[Dict], window_idx: int) -> float:
    """
    Compute stability index based on geometry evolution.
    Returns: -1 to 1, where >0 is stable, <0 is unstable.
    """
    if window_idx < 3:
        return 0.5

    lookback = min(10, window_idx + 1)
    recent = geometry_history[window_idx - lookback + 1:window_idx + 1]

    correlations = [g.get("mean_correlation", 0.5) for g in recent]

    if len(correlations) < 3:
        return 0.5

    diffs = np.diff(correlations)

    growth_rates = []
    for i in range(1, len(diffs)):
        if abs(diffs[i-1]) > 0.001:
            growth_rates.append(diffs[i] / diffs[i-1])

    if not growth_rates:
        return 0.5

    avg_growth = np.mean(growth_rates)
    stability = -np.tanh(avg_growth - 1)

    return float(np.clip(stability, -1, 1))


def compute_attractor(geometry_history: List[Dict], window_idx: int,
                      trajectory: str) -> str:
    """Classify attractor type based on long-term behavior."""
    if window_idx < 5:
        return "none"

    lookback = min(20, window_idx + 1)
    recent = geometry_history[window_idx - lookback + 1:window_idx + 1]

    correlations = [g.get("mean_correlation", 0) for g in recent]
    n_clusters = [g.get("n_clusters", 1) for g in recent]

    if len(correlations) < 5:
        return "none"

    corr_var = np.var(correlations)
    cluster_var = np.var(n_clusters)

    if trajectory == "chaotic" or corr_var > 0.1:
        return "strange"
    elif trajectory == "periodic" or (cluster_var < 0.5 and corr_var < 0.02):
        return "limit_cycle"
    elif trajectory == "converging" and corr_var < 0.01:
        return "fixed_point"
    else:
        return "none"


def compute_predictability(geometry_history: List[Dict], window_idx: int) -> float:
    """Compute predictability via permutation entropy. Returns: 0 to 1."""
    if window_idx < 5:
        return 0.5

    lookback = min(20, window_idx + 1)
    recent = geometry_history[window_idx - lookback + 1:window_idx + 1]

    correlations = [g.get("mean_correlation", 0) for g in recent]

    if len(correlations) < 5:
        return 0.5

    pattern_counts = {}
    for i in range(len(correlations) - 2):
        triplet = correlations[i:i+3]
        pattern = tuple(np.argsort(triplet))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    if not pattern_counts:
        return 0.5

    total = sum(pattern_counts.values())
    probs = [c / total for c in pattern_counts.values()]
    entropy = -sum(p * np.log(p + 1e-10) for p in probs)

    max_entropy = np.log(6)
    normalized_entropy = entropy / max_entropy

    return float(1 - normalized_entropy)


def compute_coupling(geometry_history: List[Dict], window_idx: int) -> float:
    """Compute coupling strength from geometry. Returns: 0 to 1."""
    if window_idx < 0 or window_idx >= len(geometry_history):
        return 0.5

    g = geometry_history[window_idx]

    mean_corr = abs(g.get("mean_correlation", 0.5))
    density = g.get("network_density", 0.5)

    coupling = 0.7 * mean_corr + 0.3 * density

    return float(np.clip(coupling, 0, 1))


def compute_memory(geometry_history: List[Dict], window_idx: int) -> float:
    """Compute memory via Hurst exponent approximation. Returns: 0 to 1."""
    if window_idx < 10:
        return 0.5

    lookback = min(50, window_idx + 1)
    recent = geometry_history[window_idx - lookback + 1:window_idx + 1]

    correlations = [g.get("mean_correlation", 0.5) for g in recent]

    if len(correlations) < 10:
        return 0.5

    series = np.array(correlations)
    n = len(series)

    mean = np.mean(series)
    deviations = series - mean
    cumulative = np.cumsum(deviations)

    R = np.max(cumulative) - np.min(cumulative)
    S = np.std(series)

    if S < 1e-6:
        return 0.5

    RS = R / S

    if RS > 0 and n > 1:
        H = np.log(RS) / np.log(n)
        H = np.clip(H, 0, 1)
    else:
        H = 0.5

    return float(H)


def classify_dynamics(trajectory: str, stability: float, attractor: str,
                      coupling: float, predictability: float) -> str:
    """
    Classify overall dynamics state.

    Categories:
    - stable_coupled: High stability, high coupling
    - stable_decoupled: High stability, low coupling
    - evolving: Moderate stability
    - unstable: Low stability
    - critical: Near bifurcation (stability near 0 with high variance)
    - chaotic: Strange attractor or very low predictability
    """
    if trajectory == "chaotic" or attractor == "strange" or predictability < 0.3:
        return "chaotic"

    if stability > 0.3:
        if coupling > 0.6:
            return "stable_coupled"
        else:
            return "stable_decoupled"
    elif stability > 0:
        return "evolving"
    elif stability > -0.3:
        return "unstable"
    else:
        return "critical"


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compute Dynamical Systems")
    parser.add_argument("--force", action="store_true", help="Recompute all")
    parser.add_argument("--testing", action="store_true", help="Enable testing mode")
    parser.add_argument("--entity", type=str, default=None, help="[TESTING] Only process specific entity")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Try new name first, fall back to old
    geometry_path = data_dir / "manifold_geometry.parquet"
    if not geometry_path.exists():
        geometry_path = data_dir / "structural_geometry.parquet"

    output_path = data_dir / "dynamical_systems.parquet"

    # Check for geometry input
    if not geometry_path.exists():
        print(f"ERROR: {geometry_path} not found")
        print("Run manifold_geometry first:")
        print("  python -m prism.entry_points.manifold_geometry")
        sys.exit(1)

    print(f"Loading geometry from {geometry_path}")
    df = pl.read_parquet(geometry_path)
    print(f"  {len(df):,} geometry windows loaded")

    # Get unique entities
    entity_ids = df["entity_id"].unique().sort().to_list()
    print(f"  {len(entity_ids)} unique entities")

    # Testing mode filters
    if args.entity:
        if not args.testing:
            print("ERROR: --entity requires --testing flag")
            sys.exit(1)
        entity_ids = [e for e in entity_ids if e in args.entity.split(",")]

    output_rows = []

    for entity_id in entity_ids:
        # Get geometry history for this entity (sorted by window)
        entity_df = df.filter(pl.col("entity_id") == entity_id).sort("window_idx")
        n_windows = len(entity_df)

        if n_windows < 2:
            print(f"  {entity_id}: SKIP (only {n_windows} window)")
            continue

        # Get unit_id
        unit_id = entity_id
        if "unit_id" in entity_df.columns:
            unit_ids = entity_df["unit_id"].unique().to_list()
            if unit_ids:
                unit_id = unit_ids[0]

        # Build geometry history as list of dicts
        geometry_history = []
        timestamps = []
        window_indices = []

        for row in entity_df.iter_rows(named=True):
            geometry_history.append({
                "mean_correlation": row.get("mean_correlation", 0.0),
                "network_density": row.get("network_density", 0.0),
                "n_clusters": row.get("n_clusters", 1),
                "n_signals": row.get("n_signals", 0),
                "silhouette_score": row.get("silhouette_score", 0.0),
                "n_hubs": row.get("n_hubs", 0),
                "n_decoupled_pairs": row.get("n_decoupled_pairs", 0),
                "topology_class": row.get("topology_class", ""),
                "stability_class": row.get("stability_class", ""),
                "curvature_forman": row.get("curvature_forman", 0.0),
                "curvature_ollivier": row.get("curvature_ollivier", 0.0),
            })
            timestamps.append(row.get("timestamp"))
            window_indices.append(row.get("window_idx", 0))

        # Compute state at each window
        for w in range(n_windows):
            trajectory = compute_trajectory(geometry_history, w)
            stability = compute_stability(geometry_history, w)
            attractor = compute_attractor(geometry_history, w, trajectory)
            predictability = compute_predictability(geometry_history, w)
            coupling = compute_coupling(geometry_history, w)
            memory = compute_memory(geometry_history, w)

            dynamics_class = classify_dynamics(
                trajectory, stability, attractor, coupling, predictability
            )

            # Build state string for easy querying
            regime = "COUPLED" if coupling > 0.6 else "DECOUPLED" if coupling < 0.4 else "MODERATE"
            stab_str = "STABLE" if stability > 0.3 else "UNSTABLE" if stability < -0.3 else "EVOLVING"
            state_string = f"{regime}.{stab_str}.{trajectory.upper()}.{attractor.upper()}"

            row = {
                # Identifiers
                "entity_id": entity_id,
                "unit_id": unit_id,
                "window_idx": window_indices[w] if w < len(window_indices) else w,
                "timestamp": timestamps[w] if w < len(timestamps) else None,

                # Raw metrics from geometry (for reference)
                "metric_mean_correlation": geometry_history[w].get("mean_correlation", 0.0),
                "metric_network_density": geometry_history[w].get("network_density", 0.0),
                "metric_n_clusters": geometry_history[w].get("n_clusters", 1),
                "metric_curvature_forman": geometry_history[w].get("curvature_forman", 0.0),
                "metric_curvature_ollivier": geometry_history[w].get("curvature_ollivier", 0.0),

                # Categorical states
                "trajectory": trajectory,
                "attractor": attractor,

                # Numeric states (all normalized)
                "stability": stability,
                "predictability": predictability,
                "coupling": coupling,
                "memory": memory,

                # Classification
                "dynamics_class": dynamics_class,
                "state_string": state_string,
            }

            output_rows.append(row)

        # Summary for this entity
        states_for_entity = [r for r in output_rows if r["entity_id"] == entity_id]
        classes = [r["dynamics_class"] for r in states_for_entity[-n_windows:]]
        dominant_class = max(set(classes), key=classes.count)
        print(f"  {entity_id}: {n_windows} windows | dominant: {dominant_class}")

    if not output_rows:
        print("\nNo states computed")
        return

    # Convert to DataFrame
    output_df = pl.DataFrame(output_rows)

    # Write output
    print(f"\nWriting {len(output_df)} rows to {output_path}")
    output_df.write_parquet(output_path)

    # Summary
    print("\n" + "=" * 60)
    print("DYNAMICAL SYSTEMS COMPLETE")
    print("=" * 60)
    print(f"  Entities processed: {len(entity_ids)}")
    print(f"  Total states: {len(output_df)}")
    print(f"  Output file: {output_path}")

    # Show dynamics class distribution
    print("\nDynamics Class Distribution:")
    for dc in output_df["dynamics_class"].unique().sort().to_list():
        count = len(output_df.filter(pl.col("dynamics_class") == dc))
        pct = 100 * count / len(output_df)
        print(f"  {dc}: {count} ({pct:.1f}%)")

    # Show trajectory distribution
    print("\nTrajectory Distribution:")
    for tc in output_df["trajectory"].unique().sort().to_list():
        count = len(output_df.filter(pl.col("trajectory") == tc))
        pct = 100 * count / len(output_df)
        print(f"  {tc}: {count} ({pct:.1f}%)")

    # Numeric state stats
    print("\nNumeric State Statistics:")
    for col in ["stability", "predictability", "coupling", "memory"]:
        vals = output_df[col].to_numpy()
        print(f"  {col:15} mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  "
              f"range=[{np.min(vals):.3f}, {np.max(vals):.3f}]")

    print("\n[Transitions can be identified by comparing consecutive windows")
    print(" using thresholds from prism.config.thresholds.TRANSITION_NUMERIC]")


if __name__ == "__main__":
    main()
