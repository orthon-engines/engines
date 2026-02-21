"""
ENGINES Signal Geometry Engine

Signal geometry computes each signal's relationship to each engine's state.
This is the SCAFFOLDING - how individual signals relate to the system states.

Computes per signal, per engine, per index:
- Distance to state centroid
- Coherence to first principal component
- Contribution (projection magnitude)
- Residual (orthogonal component)

REQUIRES: signal_vector.parquet + cohort_vector.parquet + cohort_geometry.parquet

ARCHITECTURE: This is an ORCHESTRATOR that delegates all compute to primitives.
All mathematical operations are performed by directly-imported primitive functions.

Pipeline:
    signal_vector + cohort_vector + cohort_geometry → signal_geometry.parquet → dynamics.parquet
"""

import numpy as np
import polars as pl
from typing import List, Dict, Optional, Any, Set

# Import primitives for all mathematical computation
from manifold.core._pmtvs import zscore_normalize, euclidean_distance, cosine_similarity, covariance_matrix, eigendecomposition

# Import configuration
from manifold.config import get_config


# ============================================================
# CONFIGURATION-DRIVEN DEFAULTS
# ============================================================

def _get_feature_groups() -> Dict[str, List[str]]:
    """Get feature groups from config."""
    config = get_config()
    return config.get('geometry.feature_groups', {
        'shape': ['kurtosis', 'skewness', 'crest_factor'],
        'complexity': ['permutation_entropy', 'hurst', 'acf_lag1'],
        'spectral': ['spectral_entropy', 'spectral_centroid', 'band_low_rel', 'band_mid_rel', 'band_high_rel'],
    })


def _get_svd_exclude_features() -> Set[str]:
    """Get features to exclude from SVD."""
    config = get_config()
    return set(config.get('geometry.svd.exclude_features', ['cv', 'range_ratio', 'window_size']))


def _get_alignment_config() -> Dict[str, Any]:
    """Get alignment config."""
    config = get_config()
    min_norm = config.get('geometry.alignment.min_norm', 1e-10)
    if isinstance(min_norm, str):
        min_norm = float(min_norm)
    return {
        'min_norm': min_norm,
    }


# Legacy aliases for compatibility
DEFAULT_FEATURE_GROUPS = _get_feature_groups()
SVD_EXCLUDE_FEATURES = _get_svd_exclude_features()


def normalize_for_svd(matrix: np.ndarray, feature_names: List[str]) -> np.ndarray:
    """
    Z-score normalize features before SVD.

    ARCHITECTURE: Delegates normalization to ENGINES primitive.

    Excludes unbounded features (cv, range_ratio) that explode
    when signals oscillate around zero.

    Args:
        matrix: Feature matrix (N × D)
        feature_names: List of feature names

    Returns:
        Normalized matrix for SVD
    """
    # Exclude problematic features (preprocessing logic)
    svd_exclude = _get_svd_exclude_features()
    keep_idx = [i for i, f in enumerate(feature_names) if f not in svd_exclude]
    if not keep_idx:
        keep_idx = list(range(len(feature_names)))

    matrix = matrix[:, keep_idx]

    # ─────────────────────────────────────────────────
    # Z-SCORE NORMALIZE → ENGINES PRIMITIVE
    # ─────────────────────────────────────────────────
    normalized, _ = zscore_normalize(matrix, axis=0)

    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================
# SIGNAL GEOMETRY COMPUTATION (Python - needs PC projections)
# ============================================================

def compute_signal_geometry_at_index(
    signal_matrix: np.ndarray,
    signal_ids: List[str],
    centroid: np.ndarray,
    principal_components: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Compute geometry for all signals at single index.

    ARCHITECTURE: Pure orchestration - delegates all math to ENGINES primitives.

    Args:
        signal_matrix: N_signals × D_features
        signal_ids: Names of signals
        centroid: D_features centroid from cohort_vector
        principal_components: Principal components from cohort_geometry (optional)

    Returns:
        List of dicts, one per signal
    """
    # Rust-backed pmtvs functions require float64
    signal_matrix = np.asarray(signal_matrix, dtype=np.float64)
    centroid = np.asarray(centroid, dtype=np.float64)
    if principal_components is not None:
        principal_components = np.real(principal_components).astype(np.float64)

    # Get config values
    alignment_config = _get_alignment_config()
    min_norm = alignment_config['min_norm']

    N, D = signal_matrix.shape
    results = []

    _zero = np.zeros_like(centroid)
    centroid_norm = euclidean_distance(centroid, _zero)

    # If we have PCs, use first for coherence
    if principal_components is not None and len(principal_components) > 0:
        pc1 = principal_components[0]
        pc1_norm = euclidean_distance(pc1, np.zeros_like(pc1))
    else:
        # Fall back to centroid direction as PC1 proxy
        pc1 = centroid
        pc1_norm = centroid_norm

    for i, signal_id in enumerate(signal_ids):
        signal = signal_matrix[i]

        # Skip only if ALL features are NaN (no usable data)
        if not np.isfinite(signal).any():
            results.append({
                'signal_id': signal_id,
                'distance': np.nan,
                'coherence': np.nan,
                'contribution': np.nan,
                'residual': np.nan,
                'signal_magnitude': np.nan,
            })
            continue

        # Impute NaN features with centroid values (signal is "at the centroid" for missing features)
        signal_clean = np.where(np.isfinite(signal), signal, centroid)
        signal_norm = euclidean_distance(signal_clean, np.zeros_like(signal_clean))

        # ─────────────────────────────────────────────────
        # DISTANCE to centroid → ENGINES PRIMITIVE
        # ─────────────────────────────────────────────────
        distance = euclidean_distance(signal_clean, centroid)

        # ─────────────────────────────────────────────────
        # COHERENCE to PC1 (or centroid direction) → ENGINES PRIMITIVE
        # How aligned is this signal with the dominant direction?
        # ─────────────────────────────────────────────────
        if signal_norm > min_norm and pc1_norm > min_norm:
            # Center signal first for coherence
            centered = signal_clean - centroid
            centered_norm = euclidean_distance(centered, np.zeros_like(centered))
            if centered_norm > min_norm:
                # Use ENGINES cosine_similarity for coherence
                coherence = cosine_similarity(centered, pc1)
            else:
                coherence = 0.0
        else:
            coherence = 0.0

        # ─────────────────────────────────────────────────
        # CONTRIBUTION (projection onto centroid direction)
        # How much does this signal contribute to the state?
        # ─────────────────────────────────────────────────
        if centroid_norm > min_norm:
            contribution = np.dot(signal_clean, centroid) / centroid_norm
        else:
            contribution = 0.0

        # ─────────────────────────────────────────────────
        # RESIDUAL (component orthogonal to centroid)
        # What part of signal is NOT explained by state?
        # ─────────────────────────────────────────────────
        if centroid_norm > min_norm:
            projection_on_centroid = (np.dot(signal_clean, centroid) / (centroid_norm ** 2)) * centroid
            residual_vector = signal_clean - projection_on_centroid
            residual = euclidean_distance(residual_vector, np.zeros_like(residual_vector))
        else:
            residual = signal_norm

        results.append({
            'signal_id': signal_id,
            'distance': distance,
            'coherence': coherence,
            'contribution': contribution,
            'residual': residual,
            'signal_magnitude': signal_norm,
        })

    return results


def compute_signal_geometry(
    signal_vector: pl.DataFrame,
    cohort_vector: pl.DataFrame,
    cohort_geometry_path: Optional[str] = None,
    feature_groups: Optional[Dict[str, List[str]]] = None,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Compute signal geometry (per signal relationships to each state).

    Args:
        signal_vector: Signal vector DataFrame
        cohort_vector: State vector DataFrame
        cohort_geometry_path: Optional path to cohort_geometry.parquet for PCs
        feature_groups: Dict mapping engine names to feature lists
        verbose: Print progress

    Returns:
        Signal geometry DataFrame
    """
    if verbose:
        print("=" * 70)
        print("SIGNAL GEOMETRY ENGINE")
        print("Per-signal relationships to each state")
        print("=" * 70)

    # Optionally load state geometry for principal components
    # (For now, we'll compute PC1 on the fly or use centroid)
    # TODO: Store PCs in cohort_geometry and load them here

    # Identify features
    meta_cols = ['unit_id', 'signal_0_start', 'signal_0_end', 'signal_0_center', 'signal_id']
    all_features = [c for c in signal_vector.columns if c not in meta_cols]

    # Determine feature groups
    if feature_groups is None:
        default_groups = _get_feature_groups()
        feature_groups = {}
        for name, features in default_groups.items():
            available = [f for f in features if f in all_features]
            if len(available) >= 2:
                feature_groups[name] = available

        if not feature_groups and len(all_features) >= 2:
            feature_groups['full'] = all_features[:3]

    if verbose:
        print(f"Feature groups: {list(feature_groups.keys())}")
        print()

    # I is REQUIRED
    if 'signal_0_end' not in signal_vector.columns:
        raise ValueError("Missing required column 'signal_0_end'. Use temporal signal_vector.")

    # Determine grouping columns - include cohort if present
    has_cohort = 'cohort' in signal_vector.columns
    group_cols = ['cohort', 'signal_0_end'] if has_cohort else ['signal_0_end']

    # Process each (cohort, I) or just I
    results = []
    groups = signal_vector.group_by(group_cols, maintain_order=True)
    n_groups = signal_vector.select(group_cols).unique().height

    if verbose:
        if has_cohort:
            n_cohorts = signal_vector['cohort'].n_unique()
            print(f"Processing {n_groups} (cohort, I) groups across {n_cohorts} cohorts...")
        else:
            print(f"Processing {n_groups} time points...")

    for i, (group_key, group) in enumerate(groups):
        if has_cohort:
            cohort, s0_end = group_key if isinstance(group_key, tuple) else (None, group_key)
        else:
            cohort = None
            s0_end = group_key[0] if isinstance(group_key, tuple) else group_key
        unit_id = group['unit_id'].to_list()[0] if 'unit_id' in group.columns else ''

        # Get state vector for this (cohort, signal_0_end) or just signal_0_end
        if has_cohort and cohort:
            state_row = cohort_vector.filter(
                (pl.col('cohort') == cohort) & (pl.col('signal_0_end') == s0_end)
            )
        else:
            state_row = cohort_vector.filter(pl.col('signal_0_end') == s0_end)

        if len(state_row) == 0:
            continue

        signal_ids = group['signal_id'].to_list()

        # Compute geometry for each engine
        for engine_name, features in feature_groups.items():
            available = [f for f in features if f in group.columns]
            if len(available) < 2:
                continue

            # Get centroid from cohort_vector
            # Try state_{engine}_{feature} first (legacy), then centroid_{feature} (v2)
            centroid_cols = [f'state_{engine_name}_{f}' for f in available]
            centroid_available = [c for c in centroid_cols if c in state_row.columns]

            if len(centroid_available) != len(available):
                # Try v2 naming: centroid_{feature}
                centroid_cols = [f'centroid_{f}' for f in available]
                centroid_available = [c for c in centroid_cols if c in state_row.columns]

            if len(centroid_available) != len(available):
                continue

            centroid = state_row.select(centroid_available).to_numpy().flatten()

            # Get signal matrix
            matrix = group.select(available).to_numpy()

            # Compute PC1 for this engine at this index
            # (Use normalized data to prevent extreme values dominating)
            valid_mask = np.isfinite(matrix).all(axis=1)
            if valid_mask.sum() >= 2:
                valid_matrix = matrix[valid_mask]
                # Normalize before eigendecomposition to prevent cv/range_ratio from dominating
                normalized = normalize_for_svd(valid_matrix, available)
                try:
                    # ─────────────────────────────────────────────────
                    # EIGENDECOMPOSITION → ENGINES PRIMITIVE
                    # ─────────────────────────────────────────────────
                    cov_matrix = covariance_matrix(normalized)
                    eigenvalues, eigenvectors = eigendecomposition(cov_matrix)
                    # Principal components are rows (transpose eigenvectors)
                    principal_components = eigenvectors.T
                except (np.linalg.LinAlgError, ValueError):
                    principal_components = None
            else:
                principal_components = None

            # Compute signal geometry
            geom_results = compute_signal_geometry_at_index(
                matrix, signal_ids, centroid, principal_components
            )

            # Build result rows (narrow schema: one row per signal per engine)
            for geom in geom_results:
                row = {
                    'signal_0_end': s0_end,
                    'signal_id': geom['signal_id'],
                    'engine': engine_name,
                    'distance': geom['distance'],
                    'coherence': geom['coherence'],
                    'contribution': geom['contribution'],
                    'residual': geom['residual'],
                    'magnitude': geom['signal_magnitude'],
                }
                # Include cohort if available
                if cohort:
                    row['cohort'] = cohort
                if unit_id:
                    row['unit_id'] = unit_id
                results.append(row)

        if verbose and (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{n_groups}...")

    # Build DataFrame (already one row per signal per engine per I — no pivot needed)
    result = pl.DataFrame(results)

    return result

