"""
prism/modules/state.py - State layer computation

Pure functions, no I/O. Computes temporal dynamics and regime detection.

Features include:
- hd_slope (coherence loss velocity)
- Trajectory metrics
- Regime change detection
- Cross-correlation dynamics
- Transfer entropy
"""

import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import linregress


def compute_state_features(
    df: pl.DataFrame,
    entity_col: str = "entity_id",
    time_col: str = "timestamp",
    reference_time: Optional[Any] = None,
) -> pl.DataFrame:
    """
    Compute state/trajectory features for each entity.

    Args:
        df: Input DataFrame (geometry features recommended)
        entity_col: Entity identifier column
        time_col: Time/cycle column
        reference_time: Reference point for baseline (None = first timestamp)

    Returns:
        DataFrame with state features per (entity, timestamp)
    """
    results = []

    # Get feature columns
    feature_cols = [
        c for c in df.columns
        if c not in [entity_col, time_col]
        and df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
    ]

    if not feature_cols:
        return df.select([entity_col, time_col])

    entities = df[entity_col].unique().to_list()

    for entity_id in entities:
        entity_df = df.filter(pl.col(entity_col) == entity_id).sort(time_col)

        if len(entity_df) < 5:
            continue

        X = entity_df.select(feature_cols).to_numpy()
        times = entity_df[time_col].to_numpy()

        # Remove columns that are all NaN
        col_mask = ~np.isnan(X).all(axis=0)
        X = X[:, col_mask]

        if X.shape[1] == 0:
            continue

        # Remove rows with NaN (after dropping all-NaN columns)
        row_mask = ~np.isnan(X).any(axis=1)
        X = X[row_mask]
        times = times[row_mask]

        if len(X) < 5:
            continue

        # Baseline state (first observation or reference)
        baseline = X[0]

        # Compute state metrics at each timestamp
        hausdorff_distances = []
        velocities = []

        for i in range(len(X)):
            current = X[i]
            t = times[i]

            # Distance from baseline
            distance = np.linalg.norm(current - baseline)
            hausdorff_distances.append(distance)

            # Velocity (rate of change)
            if i > 0:
                prev = X[i - 1]
                velocity = np.linalg.norm(current - prev)
                velocities.append(velocity)
            else:
                velocities.append(0.0)

            # Compute trajectory metrics up to this point
            metrics = _compute_trajectory_metrics(
                X[:i + 1], times[:i + 1], baseline, hausdorff_distances
            )

            results.append({
                entity_col: entity_id,
                time_col: t,
                'distance_from_baseline': distance,
                'velocity': velocities[-1],
                **metrics,
            })

    if not results:
        return pl.DataFrame()

    return pl.DataFrame(results)


def _compute_trajectory_metrics(
    X: np.ndarray,
    times: np.ndarray,
    baseline: np.ndarray,
    hausdorff_distances: List[float],
) -> Dict[str, float]:
    """Compute trajectory metrics up to current point."""
    metrics = {}
    n = len(X)

    if n < 3:
        return {
            'hd_slope': np.nan,
            'trajectory_length': np.nan,
            'mean_velocity': np.nan,
            'acceleration_magnitude': np.nan,
            'regime_stability': np.nan,
        }

    # hd_slope: Slope of Hausdorff distance over time
    # This is the "coherence loss velocity" - key prognostic
    try:
        # Use numeric time indices if times aren't numeric
        if np.issubdtype(times.dtype, np.number):
            t_numeric = times.astype(float)
        else:
            t_numeric = np.arange(n, dtype=float)

        slope, intercept, r_value, p_value, std_err = linregress(
            t_numeric, hausdorff_distances
        )
        metrics['hd_slope'] = float(slope)
        metrics['hd_r_squared'] = float(r_value ** 2)
    except Exception:
        metrics['hd_slope'] = np.nan
        metrics['hd_r_squared'] = np.nan

    # Trajectory length (total path length)
    try:
        path_lengths = [
            np.linalg.norm(X[i] - X[i - 1])
            for i in range(1, n)
        ]
        metrics['trajectory_length'] = float(np.sum(path_lengths))
    except Exception:
        metrics['trajectory_length'] = np.nan

    # Mean velocity
    try:
        if len(path_lengths) > 0:
            metrics['mean_velocity'] = float(np.mean(path_lengths))
        else:
            metrics['mean_velocity'] = np.nan
    except Exception:
        metrics['mean_velocity'] = np.nan

    # Acceleration (change in velocity)
    try:
        if len(path_lengths) >= 2:
            accelerations = np.diff(path_lengths)
            metrics['acceleration_magnitude'] = float(np.mean(np.abs(accelerations)))
        else:
            metrics['acceleration_magnitude'] = np.nan
    except Exception:
        metrics['acceleration_magnitude'] = np.nan

    # Regime stability (inverse of velocity variance)
    try:
        if len(path_lengths) >= 3:
            velocity_std = np.std(path_lengths)
            metrics['regime_stability'] = float(1.0 / (velocity_std + 1e-10))
        else:
            metrics['regime_stability'] = np.nan
    except Exception:
        metrics['regime_stability'] = np.nan

    # Directedness (how straight is the trajectory)
    try:
        displacement = np.linalg.norm(X[-1] - X[0])
        path_length = metrics.get('trajectory_length', 0)
        if path_length > 0:
            metrics['directedness'] = float(displacement / path_length)
        else:
            metrics['directedness'] = np.nan
    except Exception:
        metrics['directedness'] = np.nan

    return metrics


def compute_hd_slope(
    df: pl.DataFrame,
    entity_col: str = "entity_id",
    time_col: str = "timestamp",
) -> pl.DataFrame:
    """
    Compute hd_slope (coherence loss velocity) per entity.

    This is the key prognostic feature - systems with higher hd_slope
    are losing coherence faster and likely to fail sooner.

    Returns:
        DataFrame with one row per entity: [entity_col, hd_slope, hd_r_squared]
    """
    results = []

    feature_cols = [
        c for c in df.columns
        if c not in [entity_col, time_col]
        and df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
    ]

    if not feature_cols:
        return pl.DataFrame()

    entities = df[entity_col].unique().to_list()

    for entity_id in entities:
        entity_df = df.filter(pl.col(entity_col) == entity_id).sort(time_col)

        if len(entity_df) < 5:
            continue

        X = entity_df.select(feature_cols).to_numpy()
        times = entity_df[time_col].to_numpy()

        # Remove NaN
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        times = times[mask]

        if len(X) < 5:
            continue

        # Baseline
        baseline = X[0]

        # Hausdorff distances
        distances = [np.linalg.norm(X[i] - baseline) for i in range(len(X))]

        # Fit line
        try:
            if np.issubdtype(times.dtype, np.number):
                t_numeric = times.astype(float)
            else:
                t_numeric = np.arange(len(X), dtype=float)

            slope, _, r_value, _, _ = linregress(t_numeric, distances)

            results.append({
                entity_col: entity_id,
                'hd_slope': float(slope),
                'hd_r_squared': float(r_value ** 2),
                'n_observations': len(X),
            })
        except Exception:
            continue

    if not results:
        return pl.DataFrame()

    return pl.DataFrame(results)
