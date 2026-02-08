"""
ENGINES Signal Pairwise Engine

Signal pairwise computes relationships BETWEEN signals at each index.
This captures the internal structure of the signal cloud.

Computes per pair, per engine, per index:
- Correlation (do they move together in feature space?)
- Distance (how far apart?)
- Cosine similarity (same direction?)

REQUIRES: signal_vector.parquet + state_vector.parquet (for context)

N signals → N²/2 unique pairs per index
N ≈ 14 → ~91 pairs per index
TRACTABLE (not the N² across time we avoided)

ARCHITECTURE: This is an ORCHESTRATOR that delegates all compute to ENGINES primitives.
All mathematical operations are performed by engines.* functions.

Pipeline:
    signal_vector + state_vector → signal_pairwise.parquet → dynamics.parquet
"""

import numpy as np
import polars as pl
import duckdb
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from itertools import combinations

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


def _get_pairwise_config() -> Dict[str, Any]:
    """Get pairwise computation config."""
    config = get_config()
    return {
        'high_correlation_threshold': config.get('pairwise.correlation.high_threshold', 0.8),
        'moderate_correlation_threshold': config.get('pairwise.correlation.moderate_threshold', 0.5),
        'min_correlation': config.get('pairwise.coupling.min_correlation', 0.1),
    }


def _get_thresholds() -> Dict[str, Any]:
    """Get numerical thresholds."""
    config = get_config()
    epsilon = config.get('thresholds.numerical.epsilon', 1e-10)
    if isinstance(epsilon, str):
        epsilon = float(epsilon)
    return {
        'epsilon': epsilon,
    }


# Legacy alias for compatibility
DEFAULT_FEATURE_GROUPS = _get_feature_groups()


# ============================================================
# PAIRWISE COMPUTATION (Python)
# ============================================================

def compute_pairwise_at_index(
    signal_matrix: np.ndarray,
    signal_ids: List[str],
    centroid: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Compute pairwise relationships between all signals at single index.

    ARCHITECTURE: Pure orchestration - delegates all math to ENGINES primitives.

    Args:
        signal_matrix: N_signals × D_features
        signal_ids: Names of signals
        centroid: Optional centroid for relative metrics

    Returns:
        List of dicts, one per pair
    """
    # Get config values
    thresholds = _get_thresholds()
    epsilon = thresholds['epsilon']

    N, D = signal_matrix.shape
    results = []

    # Precompute distances to centroid if provided
    if centroid is not None:
        # Use ENGINES primitive for each distance
        centroid_distances = np.array([
            engines.euclidean_distance(signal_matrix[i], centroid)
            for i in range(N)
        ])
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > epsilon:
            # Projections onto centroid direction
            projections = (signal_matrix @ centroid) / centroid_norm
        else:
            projections = np.zeros(N)
    else:
        centroid_distances = None
        projections = None

    # Compute all pairs
    for i, j in combinations(range(N), 2):
        signal_a = signal_matrix[i]
        signal_b = signal_matrix[j]
        name_a = signal_ids[i]
        name_b = signal_ids[j]

        # Skip if either signal is invalid
        if not (np.isfinite(signal_a).all() and np.isfinite(signal_b).all()):
            continue

        # ─────────────────────────────────────────────────
        # DISTANCE (Euclidean) → ENGINES PRIMITIVE
        # ─────────────────────────────────────────────────
        distance = engines.euclidean_distance(signal_a, signal_b)

        # ─────────────────────────────────────────────────
        # COSINE SIMILARITY → ENGINES PRIMITIVE
        # ─────────────────────────────────────────────────
        cosine_similarity = engines.cosine_similarity(signal_a, signal_b)

        # ─────────────────────────────────────────────────
        # CORRELATION → ENGINES PRIMITIVE
        # ─────────────────────────────────────────────────
        if D > 1:
            correlation = engines.correlation_coefficient(signal_a, signal_b)
            # Handle NaN from correlation_coefficient (can happen with constant signals)
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = cosine_similarity  # For 1D, same as cosine

        # ─────────────────────────────────────────────────
        # RELATIVE TO STATE (if centroid provided)
        # ─────────────────────────────────────────────────
        if centroid_distances is not None:
            dist_a_to_state = centroid_distances[i]
            dist_b_to_state = centroid_distances[j]

            # Are both close to state?
            mean_dist = np.mean(centroid_distances[np.isfinite(centroid_distances)])
            both_close = (dist_a_to_state < mean_dist) and (dist_b_to_state < mean_dist)
            both_far = (dist_a_to_state > mean_dist) and (dist_b_to_state > mean_dist)

            # Same side of state? (same sign of projection)
            if projections is not None:
                same_side = (projections[i] * projections[j]) > 0
            else:
                same_side = None

            # Distance difference to state
            state_distance_diff = abs(dist_a_to_state - dist_b_to_state)
        else:
            both_close = None
            both_far = None
            same_side = None
            state_distance_diff = None

        results.append({
            'signal_a': name_a,
            'signal_b': name_b,
            'distance': distance,
            'cosine_similarity': cosine_similarity,
            'correlation': correlation,
            'both_close_to_state': both_close,
            'both_far_from_state': both_far,
            'same_side_of_state': same_side,
            'state_distance_diff': state_distance_diff,
        })

    return results


def compute_signal_pairwise(
    signal_vector_path: str,
    state_vector_path: str,
    output_path: str = "signal_pairwise.parquet",
    feature_groups: Optional[Dict[str, List[str]]] = None,
    state_geometry_path: Optional[str] = None,
    coloading_threshold: float = 0.1,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Compute signal pairwise relationships.

    NEW: Uses eigenvector co-loading to gate expensive pairwise operations.
    If two signals have high co-loading on same PC, run Granger causality.
    Otherwise, correlation/mutual_info can be derived from loadings.

    Args:
        signal_vector_path: Path to signal_vector.parquet
        state_vector_path: Path to state_vector.parquet
        output_path: Output path
        feature_groups: Dict mapping engine names to feature lists
        state_geometry_path: Path to state_geometry.parquet (optional, for eigenvector gating)
        coloading_threshold: Threshold for PC co-loading to trigger Granger (default 0.5)
        verbose: Print progress

    Returns:
        Signal pairwise DataFrame
    """
    if verbose:
        print("=" * 70)
        print("SIGNAL PAIRWISE ENGINE")
        print("Signal-to-signal relationships")
        print("=" * 70)

    # Load data
    signal_vector = pl.read_parquet(signal_vector_path)
    state_vector = pl.read_parquet(state_vector_path)

    # Load eigenvector loadings for gating (if provided)
    eigenvector_gating = {}
    if state_geometry_path is not None:
        try:
            sg = pl.read_parquet(state_geometry_path)
            # Extract PC1 signal loadings for each (cohort, I)
            pc1_cols = [c for c in sg.columns if c.startswith('pc1_signal_')]
            if pc1_cols and verbose:
                print(f"Eigenvector gating enabled: {len(pc1_cols)} signal loadings found")
                # Build lookup: (cohort, I, engine) -> {signal_id: pc1_loading}
                for row in sg.iter_rows(named=True):
                    key = (row.get('cohort'), row.get('I'), row.get('engine'))
                    loadings = {}
                    for col in pc1_cols:
                        sig_id = col.replace('pc1_signal_', '')
                        if row[col] is not None:
                            loadings[sig_id] = row[col]
                    if loadings:
                        eigenvector_gating[key] = loadings
        except Exception as e:
            if verbose:
                print(f"Warning: Could not load eigenvector gating: {e}")

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

    # I is REQUIRED
    if 'I' not in signal_vector.columns:
        raise ValueError("Missing required column 'I'. Use temporal signal_vector.")

    if verbose:
        print(f"Feature groups: {list(feature_groups.keys())}")

        # Estimate output size
        n_signals = signal_vector.select('signal_id').unique().height
        n_pairs = n_signals * (n_signals - 1) // 2
        n_indices = signal_vector.select(['I']).unique().height
        n_engines = len(feature_groups)
        print(f"Signals: {n_signals} → Pairs: {n_pairs}")
        print(f"Indices: {n_indices}")
        print(f"Estimated rows: {n_pairs * n_indices * n_engines:,}")
        print()

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

        signal_ids = group['signal_id'].to_list()

        # Compute pairwise for each engine
        for engine_name, features in feature_groups.items():
            available = [f for f in features if f in group.columns]
            if len(available) < 2:
                continue

            # Get signal matrix
            matrix = group.select(available).to_numpy()

            # Get centroid from state_vector (if available)
            centroid_cols = [f'state_{engine_name}_{f}' for f in available]
            centroid_available = [c for c in centroid_cols if c in state_row.columns]

            if len(state_row) > 0 and len(centroid_available) == len(available):
                centroid = state_row.select(centroid_available).to_numpy().flatten()
            else:
                centroid = np.mean(matrix[np.isfinite(matrix).all(axis=1)], axis=0) if len(matrix) > 0 else None

            # Get eigenvector loadings for this (cohort, I, engine) if available
            gating_key = (cohort, I, engine_name)
            pc1_loadings = eigenvector_gating.get(gating_key, {})

            # Compute pairwise
            pairs = compute_pairwise_at_index(matrix, signal_ids, centroid)

            # Build result rows
            for pair in pairs:
                signal_a = pair['signal_a']
                signal_b = pair['signal_b']

                # Check eigenvector co-loading (if available)
                # High co-loading = both signals load strongly onto same PC
                # This indicates correlation without needing full pairwise compute
                pc1_a = pc1_loadings.get(signal_a, 0.0)
                pc1_b = pc1_loadings.get(signal_b, 0.0)

                # Co-loading: product of absolute loadings (both high = high product)
                coloading = abs(pc1_a * pc1_b)

                # Flag for Granger causality: high co-loading means they're related,
                # but we need Granger to determine direction
                needs_granger = coloading > coloading_threshold

                row = {
                    'I': I,
                    'signal_a': signal_a,
                    'signal_b': signal_b,
                    'engine': engine_name,
                    'distance': pair['distance'],
                    'cosine_similarity': pair['cosine_similarity'],
                    'correlation': pair['correlation'],
                    'pc1_coloading': float(coloading),
                    'needs_granger': needs_granger,
                    'both_close_to_state': pair['both_close_to_state'],
                    'both_far_from_state': pair['both_far_from_state'],
                    'same_side_of_state': pair['same_side_of_state'],
                    'state_distance_diff': pair['state_distance_diff'],
                }
                # Include cohort if available
                if cohort:
                    row['cohort'] = cohort
                if unit_id:
                    row['unit_id'] = unit_id
                results.append(row)

        if verbose and (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{n_groups}...")

    # Build DataFrame
    result = pl.DataFrame(results)
    result.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {result.shape}")

        # Summary
        if len(result) > 0:
            pairwise_config = _get_pairwise_config()
            high_threshold = pairwise_config['high_correlation_threshold']

            print(f"\nCorrelation stats:")
            print(f"  Mean: {result['correlation'].mean():.3f}")
            print(f"  Std:  {result['correlation'].std():.3f}")

            high_corr = (result['correlation'].abs() > high_threshold).sum()
            print(f"  High correlation pairs (|r|>{high_threshold}): {high_corr}")

    return result


# ============================================================
# SQL VERSION (faster for basic metrics)
# ============================================================

SIGNAL_PAIRWISE_SQL = """
-- Signal pairwise: signal-to-signal relationships
-- Self-join on signal_vector

CREATE OR REPLACE VIEW v_signal_pairwise_shape AS
WITH signal_features AS (
    SELECT
        unit_id,
        I,
        signal_id,
        kurtosis,
        skewness,
        crest_factor,
        SQRT(kurtosis*kurtosis + skewness*skewness + crest_factor*crest_factor) AS norm
    FROM signal_vector
    WHERE kurtosis IS NOT NULL
)
SELECT
    a.unit_id,
    a.I,
    a.signal_id AS signal_a,
    b.signal_id AS signal_b,
    'shape' AS engine,

    -- Distance
    SQRT(
        POWER(a.kurtosis - b.kurtosis, 2) +
        POWER(a.skewness - b.skewness, 2) +
        POWER(a.crest_factor - b.crest_factor, 2)
    ) AS distance,

    -- Cosine similarity
    (a.kurtosis * b.kurtosis + a.skewness * b.skewness + a.crest_factor * b.crest_factor) /
    NULLIF(a.norm * b.norm, 0) AS cosine_similarity

FROM signal_features a
JOIN signal_features b
    ON a.unit_id = b.unit_id
    AND a.I = b.I
    AND a.signal_id < b.signal_id;  -- Only upper triangle
"""


def compute_signal_pairwise_sql(
    signal_vector_path: str,
    output_path: str = "signal_pairwise.parquet",
    verbose: bool = True
) -> pl.DataFrame:
    """
    Compute signal pairwise using SQL (faster).

    Args:
        signal_vector_path: Path to signal_vector.parquet
        output_path: Output path
        verbose: Print progress

    Returns:
        Signal pairwise DataFrame
    """
    if verbose:
        print("=" * 70)
        print("SIGNAL PAIRWISE (SQL)")
        print("=" * 70)

    con = duckdb.connect()

    # Load data
    con.execute(f"CREATE TABLE signal_vector AS SELECT * FROM read_parquet('{signal_vector_path}')")

    if verbose:
        n_signals = con.execute("SELECT COUNT(DISTINCT signal_id) FROM signal_vector").fetchone()[0]
        n_pairs = n_signals * (n_signals - 1) // 2
        n_indices = con.execute("SELECT COUNT(DISTINCT I) FROM signal_vector").fetchone()[0]
        print(f"Signals: {n_signals} → Pairs: {n_pairs}")
        print(f"Indices: {n_indices}")

    # Run SQL
    for statement in SIGNAL_PAIRWISE_SQL.split(';'):
        statement = statement.strip()
        if statement and not statement.startswith('--'):
            try:
                con.execute(statement)
            except Exception as e:
                if verbose:
                    print(f"  Warning: {e}")

    # Export
    try:
        result = con.execute("SELECT * FROM v_signal_pairwise_shape ORDER BY unit_id, I, signal_a, signal_b").pl()
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
# AGGREGATIONS
# ============================================================

def compute_pairwise_aggregations(
    pairwise_path: str,
    output_dir: str = ".",
    verbose: bool = True
) -> Dict[str, pl.DataFrame]:
    """
    Compute aggregations on pairwise data.

    Args:
        pairwise_path: Path to signal_pairwise.parquet
        output_dir: Output directory
        verbose: Print progress

    Returns:
        Dict of aggregation DataFrames
    """
    if verbose:
        print("\nComputing pairwise aggregations...")

    con = duckdb.connect()
    con.execute(f"CREATE TABLE pairwise AS SELECT * FROM read_parquet('{pairwise_path}')")

    output_dir = Path(output_dir)
    results = {}

    # Aggregation: Most correlated pairs per unit
    sql = """
    SELECT
        unit_id,
        signal_a,
        signal_b,
        engine,
        AVG(correlation) AS mean_correlation,
        AVG(distance) AS mean_distance,
        AVG(cosine_similarity) AS mean_cosine,
        COUNT(*) AS n_observations
    FROM pairwise
    GROUP BY unit_id, signal_a, signal_b, engine
    ORDER BY mean_correlation DESC
    """

    try:
        by_pair = con.execute(sql).pl()
        by_pair.write_parquet(output_dir / "pairwise_by_pair.parquet")
        results['by_pair'] = by_pair
        if verbose:
            print(f"  Saved: pairwise_by_pair.parquet ({len(by_pair)} rows)")
    except Exception as e:
        if verbose:
            print(f"  Warning: {e}")

    # Aggregation: Coupling strength over time
    sql = """
    SELECT
        unit_id,
        I,
        engine,
        AVG(ABS(correlation)) AS mean_abs_correlation,
        AVG(distance) AS mean_distance,
        STDDEV(correlation) AS std_correlation,
        SUM(CASE WHEN ABS(correlation) > 0.8 THEN 1 ELSE 0 END) AS high_corr_pairs,
        COUNT(*) AS n_pairs
    FROM pairwise
    GROUP BY unit_id, I, engine
    ORDER BY unit_id, I
    """

    try:
        over_time = con.execute(sql).pl()
        over_time.write_parquet(output_dir / "pairwise_over_time.parquet")
        results['over_time'] = over_time
        if verbose:
            print(f"  Saved: pairwise_over_time.parquet ({len(over_time)} rows)")
    except Exception as e:
        if verbose:
            print(f"  Warning: {e}")

    con.close()

    return results


# ============================================================
# CLI
# ============================================================

def main():
    import sys

    usage = """
Signal Pairwise Engine - Signal-to-signal relationships

Usage:
    python signal_pairwise.py <signal_vector.parquet> <state_vector.parquet> [output.parquet]
    python signal_pairwise.py --sql <signal_vector.parquet> [output.parquet]
    python signal_pairwise.py --aggregate <signal_pairwise.parquet> [output_dir]

Computes per pair, per engine, per index:
- Distance (Euclidean)
- Cosine similarity
- Correlation
- Relative position to state

This captures the internal structure of the signal cloud.
"""

    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    if sys.argv[1] == '--sql':
        signal_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else "signal_pairwise.parquet"
        compute_signal_pairwise_sql(signal_path, output_path)

    elif sys.argv[1] == '--aggregate':
        pairwise_path = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "."
        compute_pairwise_aggregations(pairwise_path, output_dir)

    else:
        if len(sys.argv) < 3:
            print("Usage: python signal_pairwise.py <signal_vector.parquet> <state_vector.parquet> [output.parquet]")
            sys.exit(1)
        signal_path = sys.argv[1]
        state_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else "signal_pairwise.parquet"
        compute_signal_pairwise(signal_path, state_path, output_path)


if __name__ == "__main__":
    main()
