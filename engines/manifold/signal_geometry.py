"""
ENGINES Signal Geometry Engine

Signal geometry computes each signal's relationship to each engine's state.
This is the SCAFFOLDING - how individual signals relate to the system states.

Computes per signal, per engine, per index:
- Distance to state centroid
- Coherence to first principal component
- Contribution (projection magnitude)
- Residual (orthogonal component)

REQUIRES: signal_vector.parquet + state_vector.parquet + state_geometry.parquet

ARCHITECTURE: This is an ORCHESTRATOR that delegates all compute to ENGINES primitives.
All mathematical operations are performed by engines.* functions.

Pipeline:
    signal_vector + state_vector + state_geometry → signal_geometry.parquet → dynamics.parquet
"""

import numpy as np
import polars as pl
import duckdb
from pathlib import Path
from typing import List, Dict, Optional, Any, Set

# Import ENGINES primitives for all mathematical computation
import engines

# Import configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_config


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
    normalized, _ = engines.zscore_normalize(matrix, axis=0)

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
        centroid: D_features centroid from state_vector
        principal_components: Principal components from state_geometry (optional)

    Returns:
        List of dicts, one per signal
    """
    # Get config values
    alignment_config = _get_alignment_config()
    min_norm = alignment_config['min_norm']

    N, D = signal_matrix.shape
    results = []

    centroid_norm = np.linalg.norm(centroid)

    # If we have PCs, use first for coherence
    if principal_components is not None and len(principal_components) > 0:
        pc1 = principal_components[0]
        pc1_norm = np.linalg.norm(pc1)
    else:
        # Fall back to centroid direction as PC1 proxy
        pc1 = centroid
        pc1_norm = centroid_norm

    for i, signal_id in enumerate(signal_ids):
        signal = signal_matrix[i]

        # Skip invalid signals
        if not np.isfinite(signal).all():
            results.append({
                'signal_id': signal_id,
                'distance': np.nan,
                'coherence': np.nan,
                'contribution': np.nan,
                'residual': np.nan,
                'signal_magnitude': np.nan,
            })
            continue

        signal_norm = np.linalg.norm(signal)

        # ─────────────────────────────────────────────────
        # DISTANCE to centroid → ENGINES PRIMITIVE
        # ─────────────────────────────────────────────────
        distance = engines.euclidean_distance(signal, centroid)

        # ─────────────────────────────────────────────────
        # COHERENCE to PC1 (or centroid direction) → ENGINES PRIMITIVE
        # How aligned is this signal with the dominant direction?
        # ─────────────────────────────────────────────────
        if signal_norm > min_norm and pc1_norm > min_norm:
            # Center signal first for coherence
            centered = signal - centroid
            centered_norm = np.linalg.norm(centered)
            if centered_norm > min_norm:
                # Use ENGINES cosine_similarity for coherence
                coherence = engines.cosine_similarity(centered, pc1)
            else:
                coherence = 0.0
        else:
            coherence = 0.0

        # ─────────────────────────────────────────────────
        # CONTRIBUTION (projection onto centroid direction)
        # How much does this signal contribute to the state?
        # ─────────────────────────────────────────────────
        if centroid_norm > min_norm:
            contribution = np.dot(signal, centroid) / centroid_norm
        else:
            contribution = 0.0

        # ─────────────────────────────────────────────────
        # RESIDUAL (component orthogonal to centroid)
        # What part of signal is NOT explained by state?
        # ─────────────────────────────────────────────────
        if centroid_norm > min_norm:
            projection_on_centroid = (np.dot(signal, centroid) / (centroid_norm ** 2)) * centroid
            residual_vector = signal - projection_on_centroid
            residual = np.linalg.norm(residual_vector)
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
    signal_vector_path: str,
    state_vector_path: str,
    output_path: str = "signal_geometry.parquet",
    state_geometry_path: Optional[str] = None,
    feature_groups: Optional[Dict[str, List[str]]] = None,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Compute signal geometry (per signal relationships to each state).

    Args:
        signal_vector_path: Path to signal_vector.parquet
        state_vector_path: Path to state_vector.parquet
        output_path: Output path
        state_geometry_path: Optional path to state_geometry.parquet for PCs
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

    # Load data
    signal_vector = pl.read_parquet(signal_vector_path)
    state_vector = pl.read_parquet(state_vector_path)

    # Optionally load state geometry for principal components
    # (For now, we'll compute PC1 on the fly or use centroid)
    # TODO: Store PCs in state_geometry and load them here

    # Identify features
    meta_cols = ['unit_id', 'I', 'signal_id']
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
    if 'I' not in signal_vector.columns:
        raise ValueError("Missing required column 'I'. Use temporal signal_vector.")

    # Determine grouping columns - include cohort if present
    has_cohort = 'cohort' in signal_vector.columns
    group_cols = ['cohort', 'I'] if has_cohort else ['I']

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
            cohort, I = group_key if isinstance(group_key, tuple) else (None, group_key)
        else:
            cohort = None
            I = group_key[0] if isinstance(group_key, tuple) else group_key
        unit_id = group['unit_id'].to_list()[0] if 'unit_id' in group.columns else ''

        # Get state vector for this (cohort, I) or just I
        if has_cohort and cohort:
            state_row = state_vector.filter(
                (pl.col('cohort') == cohort) & (pl.col('I') == I)
            )
        else:
            state_row = state_vector.filter(pl.col('I') == I)

        if len(state_row) == 0:
            continue

        signal_ids = group['signal_id'].to_list()

        # Compute geometry for each engine
        for engine_name, features in feature_groups.items():
            available = [f for f in features if f in group.columns]
            if len(available) < 2:
                continue

            # Get centroid from state_vector
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
                    cov_matrix = engines.covariance_matrix(normalized)
                    eigenvalues, eigenvectors = engines.eigendecomposition(cov_matrix, sort_descending=True)
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

            # Build result rows
            for geom in geom_results:
                row = {
                    'I': I,
                    'signal_id': geom['signal_id'],
                    'engine': engine_name,
                    f'distance_{engine_name}': geom['distance'],
                    f'coherence_{engine_name}': geom['coherence'],
                    f'contribution_{engine_name}': geom['contribution'],
                    f'residual_{engine_name}': geom['residual'],
                    f'magnitude_{engine_name}': geom['signal_magnitude'],
                }
                # Include cohort if available
                if cohort:
                    row['cohort'] = cohort
                if unit_id:
                    row['unit_id'] = unit_id
                results.append(row)

        if verbose and (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{n_groups}...")

    # Build DataFrame
    result = pl.DataFrame(results)

    # Pivot to have one row per (cohort, I, signal_id, engine) with all columns
    if len(result) > 0:
        # Determine grouping columns
        agg_group_cols = ['I', 'signal_id', 'engine']
        if 'cohort' in result.columns:
            agg_group_cols = ['cohort'] + agg_group_cols

        result = result.group_by(agg_group_cols).agg([
            pl.col(c).first() for c in result.columns
            if c not in agg_group_cols
        ])

    result.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {result.shape}")
        print(f"Columns: {result.columns}")

    return result


# ============================================================
# SQL VERSION (simpler, for basic geometry without PCs)
# ============================================================

SIGNAL_GEOMETRY_SQL = """
-- Signal geometry: per-signal relationships to each state
-- Simpler version that doesn't need principal components

-- Build geometry for shape engine
CREATE OR REPLACE VIEW v_signal_geometry_shape AS
SELECT
    sv.unit_id,
    sv.I,
    sv.signal_id,
    'shape' AS engine,

    -- Distance to centroid
    SQRT(
        POWER(sv.kurtosis - st.state_shape_kurtosis, 2) +
        POWER(sv.skewness - st.state_shape_skewness, 2) +
        POWER(sv.crest_factor - st.state_shape_crest_factor, 2)
    ) AS distance,

    -- Coherence (cosine similarity of centered signal to centroid direction)
    -- Simplified: use dot product normalized
    (
        (sv.kurtosis - st.state_shape_kurtosis) * st.state_shape_kurtosis +
        (sv.skewness - st.state_shape_skewness) * st.state_shape_skewness +
        (sv.crest_factor - st.state_shape_crest_factor) * st.state_shape_crest_factor
    ) / NULLIF(
        SQRT(
            POWER(sv.kurtosis - st.state_shape_kurtosis, 2) +
            POWER(sv.skewness - st.state_shape_skewness, 2) +
            POWER(sv.crest_factor - st.state_shape_crest_factor, 2)
        ) *
        SQRT(
            POWER(st.state_shape_kurtosis, 2) +
            POWER(st.state_shape_skewness, 2) +
            POWER(st.state_shape_crest_factor, 2)
        ),
        0
    ) AS coherence,

    -- Contribution (projection onto centroid)
    (
        sv.kurtosis * st.state_shape_kurtosis +
        sv.skewness * st.state_shape_skewness +
        sv.crest_factor * st.state_shape_crest_factor
    ) / NULLIF(
        SQRT(
            POWER(st.state_shape_kurtosis, 2) +
            POWER(st.state_shape_skewness, 2) +
            POWER(st.state_shape_crest_factor, 2)
        ),
        0
    ) AS contribution,

    -- Signal magnitude
    SQRT(
        POWER(sv.kurtosis, 2) +
        POWER(sv.skewness, 2) +
        POWER(sv.crest_factor, 2)
    ) AS signal_magnitude

FROM signal_vector sv
JOIN state_vector st ON sv.unit_id = st.unit_id AND sv.I = st.I
WHERE sv.kurtosis IS NOT NULL;
"""


def compute_signal_geometry_sql(
    signal_vector_path: str,
    state_vector_path: str,
    output_path: str = "signal_geometry.parquet",
    verbose: bool = True
) -> pl.DataFrame:
    """
    Compute signal geometry using SQL (simpler, faster for basic metrics).

    Args:
        signal_vector_path: Path to signal_vector.parquet
        state_vector_path: Path to state_vector.parquet
        output_path: Output path
        verbose: Print progress

    Returns:
        Signal geometry DataFrame
    """
    if verbose:
        print("=" * 70)
        print("SIGNAL GEOMETRY (SQL)")
        print("=" * 70)

    con = duckdb.connect()

    # Load data
    con.execute(f"CREATE TABLE signal_vector AS SELECT * FROM read_parquet('{signal_vector_path}')")
    con.execute(f"CREATE TABLE state_vector AS SELECT * FROM read_parquet('{state_vector_path}')")

    if verbose:
        n_signals = con.execute("SELECT COUNT(DISTINCT signal_id) FROM signal_vector").fetchone()[0]
        n_indices = con.execute("SELECT COUNT(DISTINCT I) FROM signal_vector").fetchone()[0]
        print(f"Signals: {n_signals}, Indices: {n_indices}")

    # Run geometry SQL
    for statement in SIGNAL_GEOMETRY_SQL.split(';'):
        statement = statement.strip()
        if statement and not statement.startswith('--'):
            try:
                con.execute(statement)
            except Exception as e:
                if verbose:
                    print(f"  Warning: {e}")

    # Export
    try:
        result = con.execute("SELECT * FROM v_signal_geometry_shape ORDER BY unit_id, I, signal_id").pl()
        result.write_parquet(output_path)

        if verbose:
            print(f"\nSaved: {output_path}")
            print(f"Shape: {result.shape}")
    except Exception as e:
        if verbose:
            print(f"Error: {e}")
        result = pl.DataFrame()

    con.close()

    return result


# ============================================================
# COMBINED: Python first, then SQL aggregations
# ============================================================

def compute_full_signal_geometry(
    signal_vector_path: str,
    state_vector_path: str,
    output_dir: str = ".",
    state_geometry_path: Optional[str] = None,
    feature_groups: Optional[Dict[str, List[str]]] = None,
    verbose: bool = True
) -> Dict[str, pl.DataFrame]:
    """
    Compute full signal geometry (Python + SQL).

    Args:
        signal_vector_path: Path to signal_vector.parquet
        state_vector_path: Path to state_vector.parquet
        output_dir: Output directory
        state_geometry_path: Optional path to state_geometry.parquet
        feature_groups: Dict mapping engine names to feature lists
        verbose: Print progress

    Returns:
        Dict of DataFrames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Python computation (with PC projections)
    signal_geom = compute_signal_geometry(
        signal_vector_path,
        state_vector_path,
        str(output_dir / "signal_geometry.parquet"),
        state_geometry_path,
        feature_groups,
        verbose
    )

    return {
        'signal_geometry': signal_geom,
    }


# ============================================================
# CLI
# ============================================================

def main():
    import sys

    usage = """
Signal Geometry Engine - Per-signal relationships to states

Usage:
    python signal_geometry.py <signal_vector.parquet> <state_vector.parquet> [output.parquet]
    python signal_geometry.py --sql <signal_vector.parquet> <state_vector.parquet> [output.parquet]

Computes per signal, per engine, per index:
- Distance to state centroid
- Coherence to principal component
- Contribution (projection magnitude)
- Residual (orthogonal component)

This is the SCAFFOLDING between signals and system states.
"""

    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    if sys.argv[1] == '--sql':
        signal_path = sys.argv[2]
        state_path = sys.argv[3]
        output_path = sys.argv[4] if len(sys.argv) > 4 else "signal_geometry.parquet"
        compute_signal_geometry_sql(signal_path, state_path, output_path)
    else:
        signal_path = sys.argv[1]
        state_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else "signal_geometry.parquet"
        compute_signal_geometry(signal_path, state_path, output_path)


if __name__ == "__main__":
    main()
