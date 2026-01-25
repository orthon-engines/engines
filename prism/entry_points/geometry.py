#!/usr/bin/env python3
"""
PRISM Geometry Entry Point
==========================

Answers: WHERE does the entity live in behavioral space?

REQUIRES: vector.parquet

Transforms 24 independent signals into ONE unified behavioral manifold.
Computes the structure that dynamics needs for Mahalanobis distance.

Output:
    data/geometry.parquet - One row per entity with:
        - Covariance structure (for Mahalanobis distance in dynamics)
        - PCA components and variance explained
        - Cluster centers (for regime detection)
        - Effective dimensionality
        - Correlation structure

The manifold concept:
    RAW SIGNAL SPACE (24D):     BEHAVIORAL MANIFOLD:
    - Each axis = one sensor    - Axes = principal components
    - Euclidean misleading      - Distance accounts for correlations
    - Correlated = double-count - Structure reveals healthy vs degraded

Usage:
    python -m prism.entry_points.geometry
    python -m prism.entry_points.geometry --force
"""

import argparse
import logging
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json

import numpy as np
import polars as pl

from prism.core.dependencies import check_dependencies
from prism.db.parquet_store import get_path, ensure_directory, VECTOR, GEOMETRY
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
    'min_samples_geometry': 10,
    'n_pca_components': 10,
    'n_clusters': 3,
}


def load_config(data_path: Path) -> Dict[str, Any]:
    """Load config from data directory."""
    config_path = data_path / 'config.yaml'
    config = DEFAULT_CONFIG.copy()

    if config_path.exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f) or {}

        if 'min_samples_geometry' in user_config:
            config['min_samples_geometry'] = user_config['min_samples_geometry']

    return config


# =============================================================================
# BEHAVIORAL VECTOR CONSTRUCTION
# =============================================================================

def build_behavioral_matrix(
    vector_df: pl.DataFrame,
    entity_id: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], list]:
    """
    Build the behavioral matrix for one entity.

    The behavioral vector at each timestamp is the FULL state across all signals.
    Shape: (n_timestamps, n_signals × n_metrics)

    Args:
        vector_df: Vector metrics with columns [entity_id, signal_id, window_start, ...]
        entity_id: Entity to extract

    Returns:
        (behavioral_matrix, timestamps, column_names) or (None, None, [])
    """
    entity_data = vector_df.filter(pl.col('entity_id') == entity_id)

    if len(entity_data) == 0:
        return None, None, []

    # Identify metric columns (exclude identifiers)
    id_cols = {'entity_id', 'signal_id', 'window_start', 'window_end', 'n_samples'}
    metric_cols = [c for c in entity_data.columns
                   if c not in id_cols
                   and entity_data[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

    if not metric_cols:
        return None, None, []

    # Get unique signals and timestamps
    signals = sorted(entity_data['signal_id'].unique().to_list())
    n_signals = len(signals)

    # Use window_start as timestamp proxy
    if 'window_start' not in entity_data.columns:
        return None, None, []

    # For each signal, we have one vector of metrics
    # The behavioral state is the concatenation across all signals
    # Build: (n_signals, n_metrics) matrix for this entity

    # Actually, in current vector.parquet structure:
    # - Each row is (entity_id, signal_id) with 155+ metrics
    # - We don't have timestamps per row, we have one row per signal

    # So the "behavioral vector" for an entity is:
    # Concatenate all signals' metrics into one big vector

    # Build feature matrix: (n_signals, n_metrics)
    feature_matrix = entity_data.select(metric_cols).to_numpy()
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Column names for reference
    col_names = metric_cols

    # "Timestamp" is signal index for now (until we have proper temporal structure)
    timestamps = np.arange(len(entity_data), dtype=float)

    return feature_matrix, timestamps, col_names


# =============================================================================
# GEOMETRY COMPUTATION
# =============================================================================

def compute_pca(feature_matrix: np.ndarray, n_components: int = 10) -> Dict[str, Any]:
    """
    Compute PCA on feature matrix.

    Returns components, variance explained, and projections.
    """
    n_samples, n_features = feature_matrix.shape
    n_components = min(n_components, n_samples, n_features)

    # Center the data
    mean = np.mean(feature_matrix, axis=0)
    centered = feature_matrix - mean

    # Compute covariance
    cov = np.cov(centered.T)

    # Eigendecomposition
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Take top components
        components = eigenvectors[:, :n_components].T  # (n_components, n_features)
        explained_variance = eigenvalues[:n_components]
        total_variance = np.sum(eigenvalues)

        if total_variance > 1e-10:
            explained_ratio = explained_variance / total_variance
        else:
            explained_ratio = np.zeros(n_components)

        return {
            'pca_components': components,
            'pca_mean': mean,
            'pca_explained_variance': explained_variance,
            'pca_explained_ratio': explained_ratio,
            'pca_total_variance': total_variance,
            'pca_n_components': n_components,
        }
    except Exception:
        return {}


def compute_covariance_structure(feature_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Compute covariance matrix and related metrics.

    This is what dynamics uses for Mahalanobis distance.
    """
    n_samples, n_features = feature_matrix.shape

    try:
        cov = np.cov(feature_matrix.T)

        # Ensure 2D
        if cov.ndim == 0:
            cov = np.array([[cov]])
        elif cov.ndim == 1:
            cov = np.diag(cov)

        # Compute inverse (for Mahalanobis)
        try:
            # Regularize for numerical stability
            cov_reg = cov + 1e-6 * np.eye(cov.shape[0])
            cov_inv = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)

        # Eigenvalues for effective dimensionality
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[eigenvalues > 1e-10]

        # Effective dimensionality (participation ratio)
        if len(eigenvalues) > 0:
            total_var = np.sum(eigenvalues)
            p = eigenvalues / total_var
            entropy = -np.sum(p * np.log(p + 1e-10))
            effective_dim = np.exp(entropy)
        else:
            effective_dim = 0.0

        return {
            'covariance_matrix': cov,
            'covariance_inverse': cov_inv,
            'cov_trace': float(np.trace(cov)),
            'cov_det_log': float(np.log(np.abs(np.linalg.det(cov)) + 1e-10)),
            'cov_condition': float(np.linalg.cond(cov)) if cov.shape[0] > 1 else 1.0,
            'effective_dimensionality': float(effective_dim),
            'n_features': n_features,
        }
    except Exception as e:
        logger.debug(f"Covariance computation failed: {e}")
        return {'n_features': n_features}


def compute_clustering(feature_matrix: np.ndarray, n_clusters: int = 3) -> Dict[str, Any]:
    """
    Compute clustering for regime detection.

    Returns cluster centers and assignments.
    """
    n_samples, n_features = feature_matrix.shape

    if n_samples < n_clusters:
        return {}

    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        # Hierarchical clustering
        distances = pdist(feature_matrix, metric='euclidean')
        linkage_matrix = linkage(distances, method='ward')

        # Get cluster assignments
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1

        # Compute cluster centers
        centers = np.zeros((n_clusters, n_features))
        for i in range(n_clusters):
            mask = labels == i
            if np.any(mask):
                centers[i] = np.mean(feature_matrix[mask], axis=0)

        # Intra-cluster variance
        intra_variance = 0.0
        for i in range(n_clusters):
            mask = labels == i
            if np.any(mask):
                cluster_points = feature_matrix[mask]
                intra_variance += np.sum((cluster_points - centers[i]) ** 2)

        return {
            'cluster_centers': centers,
            'cluster_labels': labels,
            'n_clusters': n_clusters,
            'cluster_intra_variance': float(intra_variance),
        }
    except Exception as e:
        logger.debug(f"Clustering failed: {e}")
        return {}


def compute_correlation_structure(feature_matrix: np.ndarray) -> Dict[str, Any]:
    """Compute correlation metrics."""
    try:
        corr = np.corrcoef(feature_matrix.T)

        # Handle edge cases
        if corr.ndim == 0:
            return {'corr_mean': 0.0}

        # Upper triangle (excluding diagonal)
        n = corr.shape[0]
        if n < 2:
            return {'corr_mean': 0.0}

        upper_tri = corr[np.triu_indices(n, k=1)]
        upper_tri = upper_tri[np.isfinite(upper_tri)]

        if len(upper_tri) == 0:
            return {'corr_mean': 0.0}

        return {
            'corr_mean': float(np.mean(upper_tri)),
            'corr_std': float(np.std(upper_tri)),
            'corr_max': float(np.max(np.abs(upper_tri))),
            'corr_min': float(np.min(upper_tri)),
        }
    except Exception:
        return {}


def compute_distance_metrics(feature_matrix: np.ndarray) -> Dict[str, Any]:
    """Compute pairwise distance metrics."""
    try:
        from scipy.spatial.distance import pdist

        distances = pdist(feature_matrix, metric='euclidean')

        if len(distances) == 0:
            return {}

        # Centroid
        centroid = np.mean(feature_matrix, axis=0)
        centroid_distances = np.linalg.norm(feature_matrix - centroid, axis=1)

        return {
            'dist_mean': float(np.mean(distances)),
            'dist_std': float(np.std(distances)),
            'dist_max': float(np.max(distances)),
            'dist_min': float(np.min(distances)),
            'centroid_dist_mean': float(np.mean(centroid_distances)),
            'centroid_dist_max': float(np.max(centroid_distances)),
        }
    except Exception:
        return {}


def compute_entity_geometry(
    feature_matrix: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute full geometry for one entity.

    Returns flattened dict of scalar metrics plus serialized matrices.
    """
    results = {}

    # PCA
    pca = compute_pca(feature_matrix, config.get('n_pca_components', 10))
    if pca:
        # Serialize matrices as JSON for storage
        if 'pca_components' in pca:
            results['pca_components_json'] = json.dumps(pca['pca_components'].tolist())
            results['pca_mean_json'] = json.dumps(pca['pca_mean'].tolist())

        if 'pca_explained_ratio' in pca:
            for i, ratio in enumerate(pca['pca_explained_ratio'][:5]):
                results[f'pca_var_explained_{i+1}'] = float(ratio)
            results['pca_var_explained_cumsum_3'] = float(np.sum(pca['pca_explained_ratio'][:3]))
            results['pca_var_explained_cumsum_5'] = float(np.sum(pca['pca_explained_ratio'][:5]))

    # Covariance (critical for Mahalanobis)
    cov = compute_covariance_structure(feature_matrix)
    if cov:
        # Store precision matrix as binary blob (not JSON) for efficiency
        # DuckDB can read binary blobs directly
        if 'covariance_inverse' in cov:
            # Store as bytes: float64 array flattened, with shape prefix
            cov_inv = cov['covariance_inverse']
            n = cov_inv.shape[0]
            # Pack as: [n (int32)] + [flattened float64 array]
            results['precision_matrix_blob'] = (
                np.array([n], dtype=np.int32).tobytes() +
                cov_inv.astype(np.float64).tobytes()
            )
            results['precision_matrix_dim'] = n

        for k in ['cov_trace', 'cov_det_log', 'cov_condition', 'effective_dimensionality', 'n_features']:
            if k in cov:
                results[k] = cov[k]

    # Clustering (for regime detection)
    clusters = compute_clustering(feature_matrix, config.get('n_clusters', 3))
    if clusters:
        if 'cluster_centers' in clusters:
            results['cluster_centers_json'] = json.dumps(clusters['cluster_centers'].tolist())
        for k in ['n_clusters', 'cluster_intra_variance']:
            if k in clusters:
                results[k] = clusters[k]

    # Correlation structure
    corr = compute_correlation_structure(feature_matrix)
    results.update(corr)

    # Distance metrics
    dist = compute_distance_metrics(feature_matrix)
    results.update(dist)

    return results


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def compute_geometry(
    vector_df: pl.DataFrame,
    config: Dict[str, Any],
) -> pl.DataFrame:
    """
    Compute geometry for all entities.

    Output: ONE ROW PER ENTITY

    Each row contains the manifold structure that dynamics needs:
    - Covariance matrix (for Mahalanobis distance)
    - PCA components (for trajectory projection)
    - Cluster centers (for regime detection)
    """
    min_samples = config.get('min_samples_geometry', 10)

    entities = vector_df.select('entity_id').unique()['entity_id'].to_list()
    n_entities = len(entities)

    logger.info(f"Computing geometry for {n_entities} entities")

    results = []

    for i, entity_id in enumerate(entities):
        # Build behavioral matrix for this entity
        feature_matrix, timestamps, col_names = build_behavioral_matrix(vector_df, entity_id)

        if feature_matrix is None or len(feature_matrix) < min_samples:
            continue

        # Compute geometry
        geom = compute_entity_geometry(feature_matrix, config)

        if not geom:
            continue

        row_data = {
            'entity_id': entity_id,
            'n_signals': len(feature_matrix),
            'n_metrics': feature_matrix.shape[1] if feature_matrix.ndim > 1 else 1,
        }
        row_data.update(geom)

        results.append(row_data)

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i + 1}/{n_entities} entities")

    if not results:
        logger.warning("No entities with sufficient data for geometry")
        return pl.DataFrame()

    df = pl.DataFrame(results)
    logger.info(f"Geometry: {len(df)} rows (one per entity), {len(df.columns)} columns")

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRISM Geometry - WHERE does it live? (requires vector)"
    )
    parser.add_argument('--force', '-f', action='store_true',
                        help='Recompute even if output exists')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PRISM Geometry Engine")
    logger.info("WHERE does it live in behavioral space?")
    logger.info("=" * 60)

    ensure_directory()
    data_path = get_path(VECTOR).parent

    # Check dependencies (HARD FAIL if vector missing)
    check_dependencies('geometry', data_path)

    output_path = get_path(GEOMETRY)

    if output_path.exists() and not args.force:
        logger.info("geometry.parquet exists, use --force to recompute")
        return 0

    config = load_config(data_path)

    # Load input
    vector_path = get_path(VECTOR)
    vector_df = read_parquet(vector_path)
    logger.info(f"Loaded vector.parquet: {len(vector_df):,} rows, {len(vector_df.columns)} columns")

    start = time.time()
    df = compute_geometry(vector_df, config)
    elapsed = time.time() - start

    if len(df) > 0:
        write_parquet_atomic(df, output_path)
        logger.info(f"Wrote {output_path}")
        logger.info(f"  {len(df):,} rows, {len(df.columns)} columns in {elapsed:.1f}s")

    return 0


if __name__ == '__main__':
    sys.exit(main())
