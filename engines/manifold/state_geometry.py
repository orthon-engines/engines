"""
ENGINES State Geometry Engine

State geometry computes the SHAPE of the signal distribution around each state.
This is where eigenvalues live - they describe relationships between signals,
not the state position itself.

Computes per engine, per index:
- Eigenvalues (via SVD)
- effective_dim (from eigenvalues)
- Total variance, condition number
- Eigenvalue entropy

REQUIRES: signal_vector.parquet + state_vector.parquet

ARCHITECTURE: This is an ORCHESTRATOR that delegates all compute to ENGINES primitives.
All mathematical operations are performed by engines.* functions.

Pipeline:
    signal_vector.parquet + state_vector.parquet → state_geometry.parquet
"""

import numpy as np
import polars as pl
import duckdb
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

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
        'complexity': ['entropy', 'hurst', 'autocorr'],
        'spectral': ['spectral_entropy', 'spectral_centroid', 'band_ratio_low', 'band_ratio_mid', 'band_ratio_high'],
    })


def _get_eigenvalue_config() -> Dict[str, Any]:
    """Get eigenvalue computation config."""
    config = get_config()
    eigenvalue_threshold = config.get('geometry.effective_dimension.eigenvalue_threshold', 1e-10)
    if isinstance(eigenvalue_threshold, str):
        eigenvalue_threshold = float(eigenvalue_threshold)
    return {
        'min_signals': config.get('geometry.eigenvalue.min_signals', 3),
        'max_eigenvalues': config.get('geometry.eigenvalue.max_eigenvalues', 5),
        'eigenvalue_threshold': eigenvalue_threshold,
        'multimode_ratio_threshold': config.get('geometry.multimode.ratio_threshold', 0.5),
    }


# Legacy alias for compatibility
DEFAULT_FEATURE_GROUPS = _get_feature_groups()


# ============================================================
# EIGENVALUE COMPUTATION (Python - can't do SVD in SQL)
# ============================================================

def compute_eigenvalues(
    signal_matrix: np.ndarray,
    centroid: np.ndarray,
    min_signals: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute eigenvalues of signal distribution around centroid.

    This is the SHAPE of the signal cloud - how signals spread
    around the state (centroid).

    ARCHITECTURE: Pure orchestration - delegates all math to ENGINES primitives.

    Steps:
        1. Center signals around centroid
        2. Z-score normalize (so features are comparable) → engines.zscore_normalize
        3. Covariance matrix → engines.covariance_matrix
        4. Eigendecomposition → engines.eigendecomposition
        5. Derived metrics → engines.effective_dimension, engines.matrix_entropy, etc.

    Args:
        signal_matrix: N_signals × D_features
        centroid: D_features centroid from state_vector
        min_signals: Minimum signals for reliable eigenvalues

    Returns:
        Eigenvalue metrics
    """
    # Get config values
    eigen_config = _get_eigenvalue_config()
    if min_signals is None:
        min_signals = eigen_config['min_signals']
    eigenvalue_threshold = eigen_config['eigenvalue_threshold']

    N, D = signal_matrix.shape

    if N < min_signals:
        return _empty_eigenvalues(D)

    # Remove NaN/Inf
    valid_mask = np.isfinite(signal_matrix).all(axis=1)
    if valid_mask.sum() < min_signals:
        return _empty_eigenvalues(D)

    signal_matrix = signal_matrix[valid_mask]
    N = len(signal_matrix)

    # ─────────────────────────────────────────────────
    # CENTER SIGNALS AROUND CENTROID
    # ─────────────────────────────────────────────────
    centered = signal_matrix - centroid

    # ─────────────────────────────────────────────────
    # Z-SCORE NORMALIZE BEFORE SVD → ENGINES PRIMITIVE
    # Without this, features with large variance dominate
    # eigenvalues, making them incomparable across time
    # ─────────────────────────────────────────────────
    normalized, _ = engines.zscore_normalize(centered, axis=0)

    # ─────────────────────────────────────────────────
    # COVARIANCE MATRIX → ENGINES PRIMITIVE
    # ─────────────────────────────────────────────────
    try:
        cov_matrix = engines.covariance_matrix(normalized)

        # ─────────────────────────────────────────────────
        # EIGENDECOMPOSITION → ENGINES PRIMITIVE
        # ─────────────────────────────────────────────────
        eigenvalues, eigenvectors = engines.eigendecomposition(cov_matrix, sort_descending=True)

        # Principal components (transpose of eigenvectors for consistency)
        principal_components = eigenvectors.T

    except (np.linalg.LinAlgError, ValueError):
        return _empty_eigenvalues(D)

    # ─────────────────────────────────────────────────
    # DERIVED METRICS → ENGINES PRIMITIVES
    # ─────────────────────────────────────────────────
    total_variance = float(np.sum(eigenvalues))

    if total_variance > eigenvalue_threshold:
        # Effective dimension → ENGINES PRIMITIVE
        effective_dim = engines.effective_dimension(eigenvalues)

        # Explained variance ratios
        explained_ratios = eigenvalues / total_variance

        # Eigenvalue entropy → ENGINES PRIMITIVE (matrix_entropy on diagonal)
        nonzero = eigenvalues[eigenvalues > eigenvalue_threshold]
        if len(nonzero) > 1:
            p = nonzero / nonzero.sum()
            eigenvalue_entropy = float(-np.sum(p * np.log(p)))
            max_entropy = np.log(len(nonzero))
            eigenvalue_entropy_normalized = eigenvalue_entropy / max_entropy if max_entropy > 0 else 0
        else:
            eigenvalue_entropy = 0
            eigenvalue_entropy_normalized = 0

        # Condition number → ENGINES PRIMITIVE
        condition_number = engines.condition_number(cov_matrix)

        # Eigenvalue ratios (for multi-mode detection)
        if len(eigenvalues) >= 2 and eigenvalues[0] > eigenvalue_threshold:
            ratio_2_1 = float(eigenvalues[1] / eigenvalues[0])
        else:
            ratio_2_1 = 0

        if len(eigenvalues) >= 3 and eigenvalues[0] > eigenvalue_threshold:
            ratio_3_1 = float(eigenvalues[2] / eigenvalues[0])
        else:
            ratio_3_1 = 0

    else:
        effective_dim = 0
        explained_ratios = np.zeros_like(eigenvalues)
        eigenvalue_entropy = 0
        eigenvalue_entropy_normalized = 0
        condition_number = 1.0
        ratio_2_1 = 0
        ratio_3_1 = 0

    return {
        'eigenvalues': eigenvalues,
        'explained_ratios': explained_ratios,
        'total_variance': total_variance,
        'effective_dim': effective_dim,
        'eigenvalue_entropy': eigenvalue_entropy,
        'eigenvalue_entropy_normalized': eigenvalue_entropy_normalized,
        'condition_number': condition_number,
        'ratio_2_1': ratio_2_1,
        'ratio_3_1': ratio_3_1,
        'principal_components': principal_components,
        'n_signals': N,
        'n_features': D,
    }


def _empty_eigenvalues(D: int) -> Dict[str, Any]:
    """Return empty eigenvalue result for edge cases."""
    return {
        'eigenvalues': np.zeros(D),
        'explained_ratios': np.zeros(D),
        'total_variance': 0.0,
        'effective_dim': 0.0,
        'eigenvalue_entropy': 0.0,
        'eigenvalue_entropy_normalized': 0.0,
        'condition_number': 1.0,
        'ratio_2_1': 0.0,
        'ratio_3_1': 0.0,
        'principal_components': np.eye(D),
        'n_signals': 0,
        'n_features': D,
    }


# ============================================================
# STATE GEOMETRY COMPUTATION
# ============================================================

def compute_state_geometry(
    signal_vector_path: str,
    state_vector_path: str,
    output_path: str = "state_geometry.parquet",
    feature_groups: Optional[Dict[str, List[str]]] = None,
    max_eigenvalues: Optional[int] = None,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Compute state geometry (eigenvalues per engine per index).

    Args:
        signal_vector_path: Path to signal_vector.parquet
        state_vector_path: Path to state_vector.parquet
        output_path: Output path
        feature_groups: Dict mapping engine names to feature lists
        max_eigenvalues: Maximum eigenvalues to store
        verbose: Print progress

    Returns:
        State geometry DataFrame
    """
    # Get config values
    eigen_config = _get_eigenvalue_config()
    if max_eigenvalues is None:
        max_eigenvalues = eigen_config['max_eigenvalues']

    if verbose:
        print("=" * 70)
        print("STATE GEOMETRY ENGINE")
        print("Eigenvalues and shape metrics per engine")
        print("=" * 70)

    # Load data
    signal_vector = pl.read_parquet(signal_vector_path)
    state_vector = pl.read_parquet(state_vector_path)

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

    # Process each I (unit_id is cargo, not a compute key)
    results = []
    groups = signal_vector.group_by(['I'], maintain_order=True)
    n_groups = signal_vector.select(['I']).unique().height

    if verbose:
        print(f"Processing {n_groups} time points...")

    for i, (I_tuple, group) in enumerate(groups):
        I = I_tuple[0] if isinstance(I_tuple, tuple) else I_tuple

        # Get state vector for this I
        state_row = state_vector.filter(pl.col('I') == I)

        if len(state_row) == 0:
            continue

        # Compute eigenvalues for each engine
        for engine_name, features in feature_groups.items():
            available = [f for f in features if f in group.columns]
            if len(available) < 2:
                continue

            # Get centroid from state_vector
            centroid_cols = [f'state_{engine_name}_{f}' for f in available]
            centroid_available = [c for c in centroid_cols if c in state_row.columns]

            if len(centroid_available) != len(available):
                # Centroid not computed for this engine, compute from data
                matrix = group.select(available).to_numpy()
                valid_mask = np.isfinite(matrix).all(axis=1)
                if valid_mask.sum() > 0:
                    centroid = np.mean(matrix[valid_mask], axis=0)
                else:
                    continue
            else:
                centroid = state_row.select(centroid_available).to_numpy().flatten()

            # Get signal matrix
            matrix = group.select(available).to_numpy()

            # Compute eigenvalues
            eigen_result = compute_eigenvalues(matrix, centroid)

            # Build result row (unit_id from group if present, else blank)
            unit_id = group['unit_id'][0] if 'unit_id' in group.columns else ''
            row = {
                'unit_id': unit_id,
                'I': I,
                'engine': engine_name,
                'n_signals': eigen_result['n_signals'],
                'n_features': eigen_result['n_features'],
            }

            # Eigenvalues
            for j in range(min(max_eigenvalues, len(eigen_result['eigenvalues']))):
                row[f'eigenvalue_{j+1}'] = float(eigen_result['eigenvalues'][j])
            for j in range(len(eigen_result['eigenvalues']), max_eigenvalues):
                row[f'eigenvalue_{j+1}'] = 0.0

            # Explained ratios
            for j in range(min(max_eigenvalues, len(eigen_result['explained_ratios']))):
                row[f'explained_{j+1}'] = float(eigen_result['explained_ratios'][j])
            for j in range(len(eigen_result['explained_ratios']), max_eigenvalues):
                row[f'explained_{j+1}'] = 0.0

            # Derived metrics
            row['effective_dim'] = eigen_result['effective_dim']
            row['total_variance'] = eigen_result['total_variance']
            row['eigenvalue_entropy'] = eigen_result['eigenvalue_entropy']
            row['eigenvalue_entropy_norm'] = eigen_result['eigenvalue_entropy_normalized']
            row['condition_number'] = eigen_result['condition_number']
            row['ratio_2_1'] = eigen_result['ratio_2_1']
            row['ratio_3_1'] = eigen_result['ratio_3_1']

            results.append(row)

        if verbose and (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{n_groups}...")

    # Build DataFrame
    result = pl.DataFrame(results)
    result.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {result.shape}")

        # Summary per engine
        for engine_name in feature_groups.keys():
            engine_data = result.filter(pl.col('engine') == engine_name)
            if len(engine_data) > 0:
                print(f"\n{engine_name} engine:")
                print(f"  effective_dim: mean={engine_data['effective_dim'].mean():.2f}, "
                      f"std={engine_data['effective_dim'].std():.2f}")
                print(f"  eigenvalue_1: mean={engine_data['eigenvalue_1'].mean():.4f}")

    return result


# ============================================================
# SQL AGGREGATIONS (after Python eigenvalue computation)
# ============================================================

STATE_GEOMETRY_SQL_AGGREGATIONS = """
-- Aggregate state geometry metrics
-- Run AFTER eigenvalue computation

-- Summary by unit
CREATE OR REPLACE VIEW v_state_geometry_by_unit AS
SELECT
    unit_id,
    engine,
    COUNT(*) AS n_indices,

    AVG(effective_dim) AS mean_effective_dim,
    STDDEV(effective_dim) AS std_effective_dim,
    MIN(effective_dim) AS min_effective_dim,
    MAX(effective_dim) AS max_effective_dim,

    AVG(eigenvalue_1) AS mean_eigenvalue_1,
    AVG(total_variance) AS mean_total_variance,
    AVG(condition_number) AS mean_condition_number,

    -- Detect dimensional collapse
    CORR(I, effective_dim) AS effective_dim_trend,

    -- Count high ratio_2_1 (multimode indicators)
    SUM(CASE WHEN ratio_2_1 > 0.5 THEN 1 ELSE 0 END) AS multimode_count

FROM state_geometry
GROUP BY unit_id, engine;


-- Cross-engine comparison
CREATE OR REPLACE VIEW v_engine_comparison AS
SELECT
    unit_id,
    I,
    MAX(CASE WHEN engine = 'shape' THEN effective_dim END) AS effective_dim_shape,
    MAX(CASE WHEN engine = 'complexity' THEN effective_dim END) AS effective_dim_complexity,
    MAX(CASE WHEN engine = 'spectral' THEN effective_dim END) AS effective_dim_spectral,

    -- Engine disagreement
    MAX(CASE WHEN engine = 'shape' THEN effective_dim END) -
    MIN(CASE WHEN engine = 'complexity' THEN effective_dim END) AS dim_disagreement_shape_complexity

FROM state_geometry
GROUP BY unit_id, I;
"""


def run_sql_aggregations(
    state_geometry_path: str,
    output_dir: str = ".",
    verbose: bool = True
) -> Dict[str, pl.DataFrame]:
    """
    Run SQL aggregations on state geometry.

    Args:
        state_geometry_path: Path to state_geometry.parquet
        output_dir: Output directory
        verbose: Print progress

    Returns:
        Dict of aggregation DataFrames
    """
    if verbose:
        print("\nRunning SQL aggregations...")

    con = duckdb.connect()
    con.execute(f"CREATE TABLE state_geometry AS SELECT * FROM read_parquet('{state_geometry_path}')")

    # Run aggregation views
    for statement in STATE_GEOMETRY_SQL_AGGREGATIONS.split(';'):
        statement = statement.strip()
        if statement and not statement.startswith('--'):
            try:
                con.execute(statement)
            except Exception as e:
                if verbose:
                    print(f"  Warning: {e}")

    results = {}

    # Export aggregations
    output_dir = Path(output_dir)

    try:
        by_unit = con.execute("SELECT * FROM v_state_geometry_by_unit").pl()
        by_unit.write_parquet(output_dir / "state_geometry_by_unit.parquet")
        results['by_unit'] = by_unit
        if verbose:
            print(f"  Saved: state_geometry_by_unit.parquet")
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not export by_unit: {e}")

    try:
        engine_comp = con.execute("SELECT * FROM v_engine_comparison").pl()
        engine_comp.write_parquet(output_dir / "engine_comparison.parquet")
        results['engine_comparison'] = engine_comp
        if verbose:
            print(f"  Saved: engine_comparison.parquet")
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not export engine_comparison: {e}")

    con.close()

    return results


# ============================================================
# CLI
# ============================================================

def main():
    import sys

    usage = """
State Geometry Engine - Eigenvalues and shape metrics

Usage:
    python state_geometry.py <signal_vector.parquet> <state_vector.parquet> [output.parquet]
    python state_geometry.py --aggregate <state_geometry.parquet> [output_dir]

Computes per engine, per index:
- Eigenvalues (via SVD)
- effective_dim (from eigenvalues)
- Total variance, condition number
- Eigenvalue entropy

This is the SHAPE of signal distribution around each state.
"""

    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    if sys.argv[1] == '--aggregate':
        state_geometry_path = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "."
        run_sql_aggregations(state_geometry_path, output_dir)
    else:
        signal_path = sys.argv[1]
        state_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else "state_geometry.parquet"
        compute_state_geometry(signal_path, state_path, output_path)


if __name__ == "__main__":
    main()
