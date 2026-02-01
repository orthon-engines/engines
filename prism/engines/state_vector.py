"""
PRISM State Vector Engine (v2)

The state vector IS the discrete Laplace transform representation,
NOW with eigenvalues for full geometry capture.

Key features:
- Eigenvalues encode the SHAPE of the signal cloud (no angular info lost)
- Multi-mode detection (system may have multiple states)
- Per-engine state vectors (different views of the system)
- Exact effective_dim from eigenvalues (not approximated)

For two signals to occupy the same state space, they must spell
exactly the same across ALL ~6 dimensions. Eigenvalues capture this.

Credit: Avery Rudder - insight that Laplace transform IS the state engine.

Pipeline:
    signal_vector.parquet → state_vector.parquet → geometry.parquet → dynamics.parquet
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import warnings


# ============================================================
# DEFAULT ENGINE FEATURE GROUPS
# ============================================================

DEFAULT_FEATURE_GROUPS = {
    'shape': ['kurtosis', 'skewness', 'crest_factor'],
    'complexity': ['entropy', 'hurst', 'autocorr'],
    'spectral': ['spectral_entropy', 'spectral_centroid', 'band_ratio_low', 'band_ratio_mid', 'band_ratio_high'],
}

# Fallback: use all available features as single group
FALLBACK_FEATURES = ['kurtosis', 'skewness', 'crest_factor']


# ============================================================
# CORE STATE COMPUTATION
# ============================================================

def compute_state_at_index(
    signal_matrix: np.ndarray,
    signal_ids: List[str],
    feature_names: List[str],
    min_signals: int = 2
) -> Dict[str, Any]:
    """
    Compute complete state at single index i.

    Args:
        signal_matrix: N_signals × D_features matrix
        signal_ids: Names of signals (rows)
        feature_names: Names of features (columns)
        min_signals: Minimum signals required for computation

    Returns:
        Complete state with centroid, eigenvalues, and geometry
    """
    N, D = signal_matrix.shape

    # Edge case: not enough signals
    if N < min_signals:
        return _empty_state(D, feature_names)

    # Remove any rows with NaN/Inf
    valid_mask = np.isfinite(signal_matrix).all(axis=1)
    if valid_mask.sum() < min_signals:
        return _empty_state(D, feature_names)

    signal_matrix = signal_matrix[valid_mask]
    signal_ids = [n for n, v in zip(signal_ids, valid_mask) if v]
    N = signal_matrix.shape[0]

    # ─────────────────────────────────────────────────────────
    # CENTROID (position in feature space)
    # ─────────────────────────────────────────────────────────
    centroid = np.mean(signal_matrix, axis=0)

    # ─────────────────────────────────────────────────────────
    # EIGENVALUES via SVD (shape of signal cloud)
    # ─────────────────────────────────────────────────────────
    centered = signal_matrix - centroid

    try:
        # SVD of centered data
        # U: N × k, S: k, Vt: k × D where k = min(N, D)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # Eigenvalues of covariance = S² / (N-1)
        eigenvalues = (S ** 2) / max(N - 1, 1)

        # Principal components (rows of Vt)
        principal_components = Vt

    except np.linalg.LinAlgError:
        eigenvalues = np.zeros(min(N, D))
        principal_components = np.eye(D)[:min(N, D)]

    # ─────────────────────────────────────────────────────────
    # DERIVED METRICS
    # ─────────────────────────────────────────────────────────
    total_variance = eigenvalues.sum()

    if total_variance > 1e-10:
        # Effective dimension (participation ratio)
        effective_dim = (total_variance ** 2) / (eigenvalues ** 2).sum()

        # Explained variance ratios
        explained_ratios = eigenvalues / total_variance

        # Eigenvalue entropy (uniformity of spread)
        # High entropy = eigenvalues are uniform = high effective dim
        # Low entropy = one eigenvalue dominates = low effective dim
        nonzero_ev = eigenvalues[eigenvalues > 1e-10]
        if len(nonzero_ev) > 1:
            p = nonzero_ev / nonzero_ev.sum()
            eigenvalue_entropy = -np.sum(p * np.log(p))
            # Normalize by max possible entropy
            max_entropy = np.log(len(nonzero_ev))
            eigenvalue_entropy_normalized = eigenvalue_entropy / max_entropy if max_entropy > 0 else 0
        else:
            eigenvalue_entropy = 0
            eigenvalue_entropy_normalized = 0
    else:
        effective_dim = 0
        explained_ratios = np.ones_like(eigenvalues) / len(eigenvalues) if len(eigenvalues) > 0 else np.array([1])
        eigenvalue_entropy = 0
        eigenvalue_entropy_normalized = 0

    # ─────────────────────────────────────────────────────────
    # MULTI-MODE DETECTION
    # ─────────────────────────────────────────────────────────
    if len(eigenvalues) >= 2 and eigenvalues[0] > 1e-10:
        # Ratio of second to first eigenvalue
        eigenvalue_ratio_2_1 = eigenvalues[1] / eigenvalues[0]

        # Count significant modes (explain > 10% variance)
        n_significant_modes = int(np.sum(explained_ratios > 0.1))

        # Multi-mode if second eigenvalue is substantial
        is_multimode = eigenvalue_ratio_2_1 > 0.5 and n_significant_modes >= 2

        # Condition number (spread of eigenvalues)
        nonzero_ev = eigenvalues[eigenvalues > 1e-10]
        if len(nonzero_ev) > 1:
            condition_number = nonzero_ev[0] / nonzero_ev[-1]
        else:
            condition_number = 1.0
    else:
        eigenvalue_ratio_2_1 = 0
        n_significant_modes = 1
        is_multimode = False
        condition_number = 1.0

    # ─────────────────────────────────────────────────────────
    # SIGNAL-LEVEL GEOMETRY
    # ─────────────────────────────────────────────────────────

    # Distance from each signal to centroid
    distances = np.linalg.norm(centered, axis=1)

    # Coherence of each signal to first principal component
    if len(principal_components) > 0 and np.linalg.norm(principal_components[0]) > 1e-10:
        pc1 = principal_components[0]
        norms = np.linalg.norm(centered, axis=1)
        # Avoid division by zero
        norms = np.where(norms > 1e-10, norms, 1)
        coherences = (centered @ pc1) / norms
    else:
        coherences = np.zeros(N)

    # Projection of each signal onto principal components
    projections = centered @ principal_components.T

    # ─────────────────────────────────────────────────────────
    # ASSEMBLE RESULT
    # ─────────────────────────────────────────────────────────
    return {
        # Metadata
        'n_signals': N,
        'n_features': D,
        'feature_names': feature_names,

        # Position (centroid)
        'centroid': centroid,

        # Shape (eigenvalues)
        'eigenvalues': eigenvalues,
        'explained_ratios': explained_ratios,
        'total_variance': total_variance,
        'effective_dim': effective_dim,
        'eigenvalue_entropy': eigenvalue_entropy,
        'eigenvalue_entropy_normalized': eigenvalue_entropy_normalized,
        'condition_number': condition_number,

        # Multi-mode
        'is_multimode': is_multimode,
        'n_modes': n_significant_modes,
        'eigenvalue_ratio_2_1': eigenvalue_ratio_2_1,

        # Principal directions
        'principal_components': principal_components,

        # Per-signal geometry
        'signal_ids': signal_ids,
        'signal_distances': distances,
        'signal_coherences': coherences,
        'signal_projections': projections,

        # Summary stats
        'mean_distance': np.mean(distances),
        'max_distance': np.max(distances),
        'std_distance': np.std(distances),
        'mean_coherence': np.mean(np.abs(coherences)),
        'coherence_spread': np.std(coherences),
    }


def _empty_state(D: int, feature_names: List[str]) -> Dict[str, Any]:
    """Return empty state for edge cases."""
    return {
        'n_signals': 0,
        'n_features': D,
        'feature_names': feature_names,
        'centroid': np.zeros(D),
        'eigenvalues': np.array([0.0]),
        'explained_ratios': np.array([1.0]),
        'total_variance': 0.0,
        'effective_dim': 0.0,
        'eigenvalue_entropy': 0.0,
        'eigenvalue_entropy_normalized': 0.0,
        'condition_number': 1.0,
        'is_multimode': False,
        'n_modes': 0,
        'eigenvalue_ratio_2_1': 0.0,
        'principal_components': np.eye(D),
        'signal_ids': [],
        'signal_distances': np.array([]),
        'signal_coherences': np.array([]),
        'signal_projections': np.array([]),
        'mean_distance': 0.0,
        'max_distance': 0.0,
        'std_distance': 0.0,
        'mean_coherence': 0.0,
        'coherence_spread': 0.0,
    }


# ============================================================
# ENGINE-SPECIFIC STATE VECTORS
# ============================================================

def compute_engine_states(
    signal_vectors_at_i: pl.DataFrame,
    feature_groups: Dict[str, List[str]]
) -> Dict[str, Dict[str, Any]]:
    """
    Compute state vector for each engine/feature group.

    Different engines give different "views" of the system:
    - shape: kurtosis, skewness, crest → distribution behavior
    - complexity: entropy, hurst → predictability
    - spectral: frequency features → energy distribution

    When engines DISAGREE, that's diagnostic information.

    Args:
        signal_vectors_at_i: DataFrame with signal vectors at single index
        feature_groups: Dict mapping engine name to feature columns

    Returns:
        Dict of {engine_name: state_dict}
    """
    engine_states = {}

    # Get signal name column
    signal_col = 'signal_id' if 'signal_id' in signal_vectors_at_i.columns else 'signal_id'
    signal_ids = signal_vectors_at_i[signal_col].to_list()

    for engine_name, features in feature_groups.items():
        # Check which features are available
        available = [f for f in features if f in signal_vectors_at_i.columns]

        if len(available) < 2:
            continue

        # Extract feature matrix
        matrix = signal_vectors_at_i.select(available).to_numpy()

        # Compute state for this view
        state = compute_state_at_index(matrix, signal_ids, available)
        state['engine_name'] = engine_name

        engine_states[engine_name] = state

    return engine_states


def detect_engine_disagreement(
    engine_states: Dict[str, Dict[str, Any]],
    dim_threshold: float = 2.0,
    mode_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Detect if different engine views disagree.

    Disagreement is diagnostic:
    - Shape says healthy, complexity says degrading → early warning
    - One view sees multimode, another doesn't → transitional state

    Args:
        engine_states: Dict of engine states
        dim_threshold: Max allowed spread in effective_dim
        mode_threshold: Threshold for multimode disagreement

    Returns:
        Disagreement analysis
    """
    if len(engine_states) < 2:
        return {
            'has_disagreement': False,
            'type': None,
            'details': 'Insufficient engines for comparison',
        }

    # Collect metrics across engines
    effective_dims = {
        name: state['effective_dim']
        for name, state in engine_states.items()
    }

    multimode_flags = {
        name: state['is_multimode']
        for name, state in engine_states.items()
    }

    n_modes = {
        name: state['n_modes']
        for name, state in engine_states.items()
    }

    # Analyze disagreement
    dims = list(effective_dims.values())
    dim_spread = max(dims) - min(dims)
    dim_disagreement = dim_spread > dim_threshold

    multimode_any = any(multimode_flags.values())
    multimode_all = all(multimode_flags.values())
    multimode_disagreement = multimode_any and not multimode_all

    # Identify which engines disagree
    if dim_disagreement:
        dim_sorted = sorted(effective_dims.items(), key=lambda x: x[1])
        low_dim_engine = dim_sorted[0][0]
        high_dim_engine = dim_sorted[-1][0]
        dim_detail = f"{low_dim_engine}={dim_sorted[0][1]:.1f} vs {high_dim_engine}={dim_sorted[-1][1]:.1f}"
    else:
        dim_detail = None

    if multimode_disagreement:
        multimode_engines = [n for n, v in multimode_flags.items() if v]
        singlemode_engines = [n for n, v in multimode_flags.items() if not v]
        mode_detail = f"multimode: {multimode_engines}, singlemode: {singlemode_engines}"
    else:
        mode_detail = None

    has_disagreement = dim_disagreement or multimode_disagreement

    return {
        'has_disagreement': has_disagreement,
        'dim_disagreement': dim_disagreement,
        'multimode_disagreement': multimode_disagreement,
        'effective_dim_spread': dim_spread,
        'effective_dims_by_engine': effective_dims,
        'multimode_by_engine': multimode_flags,
        'n_modes_by_engine': n_modes,
        'dim_detail': dim_detail,
        'mode_detail': mode_detail,
        'type': 'dimensional' if dim_disagreement else ('modal' if multimode_disagreement else None),
    }


# ============================================================
# MAIN COMPUTATION
# ============================================================

def compute_state_vector(
    signal_vector_path: str,
    typology_path: str,
    output_path: str = "state_vector.parquet",
    feature_groups: Optional[Dict[str, List[str]]] = None,
    compute_per_engine: bool = True,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Compute full state vector with eigenvalues and multi-mode detection.

    Args:
        signal_vector_path: Path to signal_vector.parquet
        typology_path: Path to typology.parquet
        output_path: Output path for state_vector.parquet
        feature_groups: Dict mapping engine names to feature lists
                       If None, uses defaults + fallback
        compute_per_engine: Whether to compute per-engine states
        verbose: Print progress

    Returns:
        Polars DataFrame with complete state vectors
    """
    if verbose:
        print("=" * 70)
        print("STATE VECTOR ENGINE (v2 - with eigenvalues)")
        print("=" * 70)

    # Load data
    signal_vector = pl.read_parquet(signal_vector_path)
    typology = pl.read_parquet(typology_path)

    # Get signal column name
    signal_col = 'signal_id' if 'signal_id' in signal_vector.columns else 'signal_id'
    typology_signal_col = 'signal_id' if 'signal_id' in typology.columns else 'signal_id'

    # Get active signals (not constant)
    active_signals = typology.filter(
        ~pl.col('is_constant')
    )[typology_signal_col].unique().to_list()

    signal_vector = signal_vector.filter(
        pl.col(signal_col).is_in(active_signals)
    )

    # Identify available feature columns
    meta_cols = ['unit_id', 'I', 'signal_id', 'signal_id', 'n_samples']
    all_features = [c for c in signal_vector.columns if c not in meta_cols]

    if verbose:
        print(f"Active signals: {len(active_signals)}")
        print(f"Available features: {all_features[:10]}{'...' if len(all_features) > 10 else ''}")

    # Determine feature groups
    if feature_groups is None:
        # Use defaults, filtered to available features
        feature_groups = {}
        for name, features in DEFAULT_FEATURE_GROUPS.items():
            available = [f for f in features if f in all_features]
            if len(available) >= 2:
                feature_groups[name] = available

        # Fallback: all features as single group
        if not feature_groups:
            fallback = [f for f in FALLBACK_FEATURES if f in all_features]
            if len(fallback) >= 2:
                feature_groups['full'] = fallback
            else:
                feature_groups['full'] = all_features[:3] if len(all_features) >= 2 else all_features

    if verbose:
        print(f"\nFeature groups:")
        for name, features in feature_groups.items():
            print(f"  {name}: {features}")

    # Composite features (union of all groups)
    composite_features = list(set(f for features in feature_groups.values() for f in features))
    composite_features = [f for f in composite_features if f in all_features]

    if verbose:
        print(f"\nComposite features: {composite_features}")
        print()

    # Process each I - group by I only (unit_id is pass-through, not for compute)
    # I is REQUIRED in canonical schema
    if 'I' not in signal_vector.columns:
        raise ValueError("Missing required column 'I'. Use temporal signal_vector.")

    results = []
    n_multimode = 0
    n_disagreement = 0

    groups = signal_vector.group_by(['I'], maintain_order=True)
    n_groups = signal_vector.select(['I']).unique().height

    if verbose:
        print(f"Processing {n_groups} time points...")

    for i, (group_key, group) in enumerate(groups):
        I = group_key[0] if isinstance(group_key, tuple) else group_key
        # Get unit_id from first row (pass-through only)
        unit_id = group['unit_id'].to_list()[0] if 'unit_id' in group.columns else ''

        # ─────────────────────────────────────────────────
        # COMPOSITE STATE (all features)
        # ─────────────────────────────────────────────────
        signal_ids = group[signal_col].to_list()

        # Build composite matrix
        available_composite = [f for f in composite_features if f in group.columns]
        if len(available_composite) < 2:
            continue

        composite_matrix = group.select(available_composite).to_numpy()
        composite_state = compute_state_at_index(
            composite_matrix, signal_ids, available_composite
        )

        # ─────────────────────────────────────────────────
        # PER-ENGINE STATES
        # ─────────────────────────────────────────────────
        if compute_per_engine:
            engine_states = compute_engine_states(group, feature_groups)
            disagreement = detect_engine_disagreement(engine_states)
        else:
            engine_states = {}
            disagreement = {'has_disagreement': False}

        # ─────────────────────────────────────────────────
        # BUILD RESULT ROW
        # ─────────────────────────────────────────────────
        row = {
            'unit_id': unit_id,
            'n_signals': composite_state['n_signals'],
        }
        if has_I:
            row['I'] = I

        # Centroid (per feature)
        for j, feat in enumerate(available_composite):
            row[f'centroid_{feat}'] = composite_state['centroid'][j]

        # Eigenvalues (top 5)
        for j, ev in enumerate(composite_state['eigenvalues'][:5]):
            row[f'eigenvalue_{j+1}'] = float(ev)
        # Pad with zeros if fewer than 5
        for j in range(len(composite_state['eigenvalues']), 5):
            row[f'eigenvalue_{j+1}'] = 0.0

        # Derived metrics
        row['effective_dim'] = composite_state['effective_dim']
        row['eigenvalue_entropy'] = composite_state['eigenvalue_entropy']
        row['eigenvalue_entropy_normalized'] = composite_state['eigenvalue_entropy_normalized']
        row['total_variance'] = composite_state['total_variance']
        row['condition_number'] = composite_state['condition_number']

        # Multi-mode
        row['is_multimode'] = composite_state['is_multimode']
        row['n_modes'] = composite_state['n_modes']
        row['eigenvalue_ratio_2_1'] = composite_state['eigenvalue_ratio_2_1']

        # Dispersion
        row['mean_distance'] = composite_state['mean_distance']
        row['max_distance'] = composite_state['max_distance']
        row['std_distance'] = composite_state['std_distance']
        row['mean_coherence'] = composite_state['mean_coherence']
        row['coherence_spread'] = composite_state['coherence_spread']

        # Per-engine effective_dim
        if compute_per_engine:
            for engine_name, state in engine_states.items():
                row[f'effective_dim_{engine_name}'] = state['effective_dim']
                row[f'is_multimode_{engine_name}'] = state['is_multimode']

        # Engine disagreement
        row['engine_disagreement'] = disagreement['has_disagreement']
        if disagreement['has_disagreement']:
            row['disagreement_type'] = disagreement['type']
            row['effective_dim_spread'] = disagreement['effective_dim_spread']
            n_disagreement += 1
        else:
            row['disagreement_type'] = None
            row['effective_dim_spread'] = 0.0

        if composite_state['is_multimode']:
            n_multimode += 1

        results.append(row)

        # Progress
        if verbose and (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{n_groups}...")

    # Build DataFrame
    result = pl.DataFrame(results)

    # Save
    result.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {result.shape}")
        print()
        print("SUMMARY")
        print("-" * 50)
        print(f"Units: {len(result)}")
        print(f"Multi-mode detected: {n_multimode} ({100*n_multimode/max(len(result),1):.1f}%)")
        print(f"Engine disagreement: {n_disagreement} ({100*n_disagreement/max(len(result),1):.1f}%)")
        print()
        print("Effective dimension:")
        print(f"  Mean: {result['effective_dim'].mean():.2f}")
        print(f"  Std:  {result['effective_dim'].std():.2f}")
        print(f"  Min:  {result['effective_dim'].min():.2f}")
        print(f"  Max:  {result['effective_dim'].max():.2f}")
        print()
        print("Eigenvalue 1 (dominant):")
        print(f"  Mean: {result['eigenvalue_1'].mean():.4f}")
        print(f"  Max:  {result['eigenvalue_1'].max():.4f}")

    return result


# ============================================================
# CLI
# ============================================================

def main():
    import sys

    usage = """
State Vector Engine (v2) - with eigenvalues and multi-mode detection

Usage:
    python state_vector.py <signal_vector.parquet> <typology.parquet> [output.parquet]

The state vector IS the discrete Laplace transform, now with full geometry:
- Eigenvalues capture the SHAPE of the signal cloud
- Multi-mode detection identifies systems with multiple states
- Per-engine states show different views (shape, complexity, spectral)
- Engine disagreement is diagnostic information

Credit: Avery Rudder - insight that Laplace transform IS the state engine.
"""

    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    signal_path = sys.argv[1]
    typology_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "state_vector.parquet"

    compute_state_vector(signal_path, typology_path, output_path)


if __name__ == "__main__":
    main()
