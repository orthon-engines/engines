#!/usr/bin/env python3
"""
PRISM Dynamics Entry Point
==========================

Answers: HOW is it moving through behavioral space?

REQUIRES: vector.parquet AND geometry.parquet

Computes motion through the manifold that geometry discovered.
The key metric is hd_slope: velocity of coherence loss.

Without geometry, there is no manifold.
Without a manifold, hd_slope is meaningless.

Output:
    data/dynamics.parquet - ONE ROW PER ENTITY with:
        - hd_slope: THE KEY METRIC - velocity of coherence loss
        - hd_slope_r_squared: Fit quality (>0.7 = linear degradation)
        - hd_velocity_mean: Average velocity in behavioral space
        - hd_acceleration_mean: Is degradation speeding up?
        - n_regimes: Number of behavioral regimes detected
        - trajectory_path_length: Total distance traveled
        - trajectory_tortuosity: Path length / displacement

hd_slope is ONE number per entity:
    - Computed across ALL signals (not per signal!)
    - Uses Mahalanobis distance (from geometry's covariance)
    - Measures velocity of coherence loss in behavioral space

Usage:
    python -m prism.entry_points.dynamics
    python -m prism.entry_points.dynamics --force
"""

import argparse
import logging
import sys
import time
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import polars as pl
from scipy.stats import linregress

from prism.core.dependencies import check_dependencies
from prism.db.parquet_store import get_path, ensure_directory, VECTOR, GEOMETRY, DYNAMICS
from prism.db.polars_io import read_parquet, write_parquet_atomic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_CONFIG = {
    'min_samples_dynamics': 10,
}


def load_config(data_path: Path) -> Dict[str, Any]:
    """Load config from data directory."""
    config_path = data_path / 'config.yaml'
    config = DEFAULT_CONFIG.copy()

    if config_path.exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f) or {}

        if 'min_samples_dynamics' in user_config:
            config['min_samples_dynamics'] = user_config['min_samples_dynamics']

    return config


# =============================================================================
# BEHAVIORAL VECTOR CONSTRUCTION
# =============================================================================

def build_behavioral_matrix(
    vector_df: pl.DataFrame,
    entity_id: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], list]:
    """
    Build behavioral matrix for one entity.

    Returns:
        (feature_matrix, timestamps, metric_cols) or (None, None, [])

    Shape of feature_matrix: (n_signals, n_metrics)
    Each row is a signal's behavioral fingerprint (155+ metrics)
    """
    entity_data = vector_df.filter(pl.col('entity_id') == entity_id)

    if len(entity_data) == 0:
        return None, None, []

    # Identify metric columns
    id_cols = {'entity_id', 'signal_id', 'window_start', 'window_end', 'n_samples'}
    metric_cols = [c for c in entity_data.columns
                   if c not in id_cols
                   and entity_data[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

    if not metric_cols:
        return None, None, []

    # Build feature matrix
    feature_matrix = entity_data.select(metric_cols).to_numpy()
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Timestamps (signal index as proxy)
    timestamps = np.arange(len(entity_data), dtype=float)

    return feature_matrix, timestamps, metric_cols


def extract_covariance_inverse(geometry_df: pl.DataFrame, entity_id: str) -> Optional[np.ndarray]:
    """
    Extract covariance inverse (precision matrix) from geometry for Mahalanobis distance.
    Supports both binary blob format (new) and JSON format (legacy).
    """
    entity_geom = geometry_df.filter(pl.col('entity_id') == entity_id)

    if len(entity_geom) == 0:
        return None

    # Try binary blob format first (new, efficient)
    if 'precision_matrix_blob' in entity_geom.columns:
        try:
            blob = entity_geom['precision_matrix_blob'][0]
            if blob is None:
                return None
            # Unpack: first 4 bytes = int32 dimension, rest = float64 array
            n = np.frombuffer(blob[:4], dtype=np.int32)[0]
            cov_inv = np.frombuffer(blob[4:], dtype=np.float64).reshape(n, n)
            return cov_inv
        except Exception:
            pass

    # Fall back to JSON format (legacy)
    if 'covariance_inverse_json' in entity_geom.columns:
        try:
            json_str = entity_geom['covariance_inverse_json'][0]
            if json_str is None:
                return None
            cov_inv = np.array(json.loads(json_str))
            return cov_inv
        except Exception:
            pass

    return None


def extract_cluster_centers(geometry_df: pl.DataFrame, entity_id: str) -> Optional[np.ndarray]:
    """Extract cluster centers from geometry for regime detection."""
    entity_geom = geometry_df.filter(pl.col('entity_id') == entity_id)

    if len(entity_geom) == 0:
        return None

    if 'cluster_centers_json' not in entity_geom.columns:
        return None

    try:
        json_str = entity_geom['cluster_centers_json'][0]
        if json_str is None:
            return None
        centers = np.array(json.loads(json_str))
        return centers
    except Exception:
        return None


def extract_pca_components(geometry_df: pl.DataFrame, entity_id: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract PCA components and mean from geometry."""
    entity_geom = geometry_df.filter(pl.col('entity_id') == entity_id)

    if len(entity_geom) == 0:
        return None

    if 'pca_components_json' not in entity_geom.columns:
        return None

    try:
        components_str = entity_geom['pca_components_json'][0]
        mean_str = entity_geom['pca_mean_json'][0]
        if components_str is None or mean_str is None:
            return None
        components = np.array(json.loads(components_str))
        mean = np.array(json.loads(mean_str))
        return components, mean
    except Exception:
        return None


# =============================================================================
# MAHALANOBIS DISTANCE
# =============================================================================

def mahalanobis_distance(x: np.ndarray, y: np.ndarray, cov_inv: np.ndarray) -> float:
    """
    Compute Mahalanobis distance between two points.

    This measures distance ON THE MANIFOLD, not in raw space.

    d² = (x-y)ᵀ Σ⁻¹ (x-y)
    """
    diff = x - y

    # Ensure dimensions match
    if len(diff) != cov_inv.shape[0]:
        # Fall back to Euclidean
        return float(np.linalg.norm(diff))

    try:
        d_squared = np.dot(np.dot(diff, cov_inv), diff)
        return float(np.sqrt(max(0, d_squared)))
    except Exception:
        return float(np.linalg.norm(diff))


# =============================================================================
# HD_SLOPE COMPUTATION
# =============================================================================

def compute_hd_slope(
    entity_id: str,
    vector_df: pl.DataFrame,
    geometry_df: pl.DataFrame,
) -> Dict[str, Any]:
    """
    Compute hd_slope for ONE entity across ALL signals.

    hd_slope = d(distance_from_baseline) / dt

    Where distance is measured IN THE BEHAVIORAL MANIFOLD,
    not in raw signal space.

    This is THE KEY METRIC of PRISM.
    """
    # Get behavioral matrix
    feature_matrix, timestamps, _ = build_behavioral_matrix(vector_df, entity_id)

    if feature_matrix is None or len(feature_matrix) < 2:
        return {}

    n_timestamps, n_features = feature_matrix.shape

    # Get covariance inverse from geometry (for Mahalanobis)
    cov_inv = extract_covariance_inverse(geometry_df, entity_id)
    use_mahalanobis = cov_inv is not None and cov_inv.shape[0] == n_features

    # Baseline = first observation (healthy state)
    baseline = feature_matrix[0]

    # Compute distance from baseline at each timestamp
    distances = np.zeros(n_timestamps)
    for t in range(n_timestamps):
        if use_mahalanobis:
            distances[t] = mahalanobis_distance(feature_matrix[t], baseline, cov_inv)
        else:
            distances[t] = float(np.linalg.norm(feature_matrix[t] - baseline))

    # Fit linear regression: distance = hd_slope * time + intercept
    try:
        slope, intercept, r_value, p_value, std_err = linregress(timestamps, distances)
    except Exception:
        return {}

    # Velocity and acceleration
    velocity = np.gradient(distances, timestamps)
    acceleration = np.gradient(velocity, timestamps)

    return {
        'entity_id': entity_id,
        'hd_slope': float(slope),                    # THE KEY METRIC
        'hd_slope_intercept': float(intercept),
        'hd_slope_r_squared': float(r_value ** 2),
        'hd_slope_p_value': float(p_value),
        'hd_slope_std_err': float(std_err),
        'hd_initial_distance': float(distances[0]),  # Should be ~0
        'hd_final_distance': float(distances[-1]),
        'hd_max_distance': float(np.max(distances)),
        'hd_velocity_mean': float(np.mean(velocity)),
        'hd_velocity_std': float(np.std(velocity)),
        'hd_acceleration_mean': float(np.mean(acceleration)),
        'hd_acceleration_std': float(np.std(acceleration)),
        'manifold_dimension': n_features,
        'distance_metric': 'mahalanobis' if use_mahalanobis else 'euclidean',
    }


# =============================================================================
# REGIME DETECTION
# =============================================================================

def detect_regime_transitions(
    entity_id: str,
    vector_df: pl.DataFrame,
    geometry_df: pl.DataFrame,
) -> Dict[str, Any]:
    """
    Detect when entity transitions between behavioral regimes.

    Regimes are defined by geometry's clustering.
    Transitions are detected by tracking cluster membership over time.
    """
    # Get behavioral matrix
    feature_matrix, timestamps, _ = build_behavioral_matrix(vector_df, entity_id)

    if feature_matrix is None or len(feature_matrix) < 2:
        return {}

    # Get cluster centers from geometry
    cluster_centers = extract_cluster_centers(geometry_df, entity_id)

    if cluster_centers is None or len(cluster_centers) == 0:
        return {'n_regimes': 0, 'n_transitions': 0}

    n_clusters = len(cluster_centers)

    # Check dimension compatibility
    if cluster_centers.shape[1] != feature_matrix.shape[1]:
        return {'n_regimes': 0, 'n_transitions': 0}

    # Assign each timestamp to nearest cluster
    regime_assignments = []
    for v in feature_matrix:
        distances = [np.linalg.norm(v - c) for c in cluster_centers]
        regime = int(np.argmin(distances))
        regime_assignments.append(regime)

    # Detect transitions
    n_transitions = 0
    for i in range(1, len(regime_assignments)):
        if regime_assignments[i] != regime_assignments[i-1]:
            n_transitions += 1

    # Time in each regime
    regime_counts = {}
    for r in regime_assignments:
        regime_counts[r] = regime_counts.get(r, 0) + 1

    return {
        'n_regimes': n_clusters,
        'n_transitions': n_transitions,
        'final_regime': regime_assignments[-1] if regime_assignments else -1,
        'regime_entropy': float(-sum(
            (c/len(regime_assignments)) * np.log(c/len(regime_assignments) + 1e-10)
            for c in regime_counts.values()
        )),
    }


# =============================================================================
# TRAJECTORY ANALYSIS
# =============================================================================

def analyze_trajectory(
    entity_id: str,
    vector_df: pl.DataFrame,
    geometry_df: pl.DataFrame,
) -> Dict[str, Any]:
    """
    Analyze the trajectory through behavioral space.

    Uses geometry's PCA to project onto principal components.
    """
    # Get behavioral matrix
    feature_matrix, timestamps, _ = build_behavioral_matrix(vector_df, entity_id)

    if feature_matrix is None or len(feature_matrix) < 2:
        return {}

    # Get PCA from geometry
    pca_data = extract_pca_components(geometry_df, entity_id)

    if pca_data is not None:
        components, mean = pca_data

        # Check dimension compatibility
        if components.shape[1] == feature_matrix.shape[1]:
            # Project onto principal components
            centered = feature_matrix - mean
            projected = np.dot(centered, components.T)  # (T, n_components)
        else:
            projected = feature_matrix
    else:
        projected = feature_matrix

    # Compute trajectory metrics
    path_segments = np.diff(projected, axis=0)
    path_lengths = np.linalg.norm(path_segments, axis=1)
    path_length = float(np.sum(path_lengths))

    displacement = float(np.linalg.norm(projected[-1] - projected[0]))

    # Tortuosity = path_length / displacement (1 = straight line)
    tortuosity = path_length / displacement if displacement > 1e-10 else float('inf')

    return {
        'trajectory_path_length': path_length,
        'trajectory_displacement': displacement,
        'trajectory_tortuosity': min(tortuosity, 1000.0),  # Cap extreme values
    }


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def compute_dynamics(
    vector_df: pl.DataFrame,
    geometry_df: pl.DataFrame,
    config: Dict[str, Any],
) -> pl.DataFrame:
    """
    Compute all dynamics metrics for all entities.

    Output: ONE ROW PER ENTITY (not per signal!)

    This is where hd_slope is computed - the velocity of coherence loss
    across the FULL behavioral space.
    """
    min_samples = config.get('min_samples_dynamics', 10)

    entities = vector_df.select('entity_id').unique()['entity_id'].to_list()
    n_entities = len(entities)

    logger.info(f"Computing dynamics for {n_entities} entities")
    logger.info("Using geometry's covariance for Mahalanobis distance")

    results = []

    for i, entity_id in enumerate(entities):
        # Check minimum samples
        entity_data = vector_df.filter(pl.col('entity_id') == entity_id)
        if len(entity_data) < min_samples:
            continue

        # Core metric: hd_slope
        hd = compute_hd_slope(entity_id, vector_df, geometry_df)
        if not hd:
            continue

        # Regime detection
        regimes = detect_regime_transitions(entity_id, vector_df, geometry_df)

        # Trajectory analysis
        trajectory = analyze_trajectory(entity_id, vector_df, geometry_df)

        # Combine all metrics
        row = {**hd, **regimes, **trajectory}
        results.append(row)

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i + 1}/{n_entities} entities")

    if not results:
        logger.warning("No entities with sufficient data for dynamics")
        return pl.DataFrame()

    df = pl.DataFrame(results)
    logger.info(f"Dynamics: {len(df)} rows (one per entity), {len(df.columns)} columns")

    # Report distance metric usage
    if 'distance_metric' in df.columns:
        mahal_count = df.filter(pl.col('distance_metric') == 'mahalanobis').height
        logger.info(f"  Mahalanobis distance used: {mahal_count}/{len(df)} entities")

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRISM Dynamics - HOW is it moving? (requires geometry)"
    )
    parser.add_argument('--force', '-f', action='store_true',
                        help='Recompute even if output exists')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PRISM Dynamics Engine")
    logger.info("HOW is it moving through behavioral space?")
    logger.info("=" * 60)

    ensure_directory()
    data_path = get_path(VECTOR).parent

    # Check dependencies (HARD FAIL if geometry missing)
    check_dependencies('dynamics', data_path)

    output_path = get_path(DYNAMICS)

    if output_path.exists() and not args.force:
        logger.info("dynamics.parquet exists, use --force to recompute")
        return 0

    config = load_config(data_path)

    # Load BOTH required inputs
    vector_path = get_path(VECTOR)
    geometry_path = get_path(GEOMETRY)

    vector_df = read_parquet(vector_path)
    geometry_df = read_parquet(geometry_path)

    logger.info(f"Loaded vector.parquet: {len(vector_df):,} rows, {len(vector_df.columns)} columns")
    logger.info(f"Loaded geometry.parquet: {len(geometry_df):,} rows, {len(geometry_df.columns)} columns")

    start = time.time()
    df = compute_dynamics(vector_df, geometry_df, config)
    elapsed = time.time() - start

    if len(df) > 0:
        write_parquet_atomic(df, output_path)
        logger.info(f"Wrote {output_path}")
        logger.info(f"  {len(df):,} rows, {len(df.columns)} columns in {elapsed:.1f}s")

    return 0


if __name__ == '__main__':
    sys.exit(main())
