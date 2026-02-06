"""
02: State Vector Entry Point
============================

Pure orchestration - calls centroid engine from engines/state/centroid.py.
Computes WHERE the system is (centroid = mean position in feature space).

Stages: signal_vector.parquet → state_vector.parquet

The SHAPE (eigenvalues, effective_dim) is computed in 03_state_geometry.py.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Dict, Optional, Any

# Import the actual computation from engine
from prism.engines.state.centroid import compute as compute_centroid_engine


# Feature groups for per-engine centroids
try:
    from prism.engines.geometry.config import DEFAULT_FEATURE_GROUPS, FALLBACK_FEATURES
except ImportError:
    DEFAULT_FEATURE_GROUPS = {
        'shape': ['kurtosis', 'skewness', 'crest_factor'],
        'complexity': ['permutation_entropy', 'hurst', 'acf_lag1'],
        'spectral': ['spectral_entropy', 'spectral_centroid', 'band_low_rel', 'band_mid_rel', 'band_high_rel'],
    }
    FALLBACK_FEATURES = ['kurtosis', 'skewness', 'crest_factor']


def compute_centroid(
    signal_matrix: np.ndarray,
    feature_names: List[str],
    min_signals: int = 2
) -> Dict[str, Any]:
    """
    Wrapper - delegates entirely to engine.

    Args:
        signal_matrix: N_signals × D_features matrix
        feature_names: Names of features (columns)
        min_signals: Minimum signals required

    Returns:
        Dict with centroid and distance statistics
    """
    return compute_centroid_engine(signal_matrix, min_signals=min_signals)


def compute_state_vector(
    signal_vector_path: str,
    typology_path: str,
    output_path: str = "state_vector.parquet",
    feature_groups: Optional[Dict[str, List[str]]] = None,
    compute_per_engine: bool = True,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Compute state vector (centroids) from signal vector.

    The state vector captures WHERE the system is in feature space.
    The SHAPE (eigenvalues, effective_dim) is computed in 03_state_geometry.py.

    Args:
        signal_vector_path: Path to signal_vector.parquet
        typology_path: Path to typology.parquet
        output_path: Output path
        feature_groups: Dict mapping engine names to feature lists
        compute_per_engine: Compute per-engine centroids
        verbose: Print progress

    Returns:
        State vector DataFrame with centroids per I
    """
    if verbose:
        print("=" * 70)
        print("02: STATE VECTOR - Centroids (position in feature space)")
        print("=" * 70)

    # Load data
    signal_vector = pl.read_parquet(signal_vector_path)
    typology = pl.read_parquet(typology_path)

    # Get active signals (non-constant)
    if 'is_constant' in typology.columns:
        active_signals = (
            typology
            .filter(pl.col('is_constant') == False)
            .select('signal_id')
            .to_series()
            .to_list()
        )
    elif 'signal_std' in typology.columns:
        active_signals = (
            typology
            .filter(pl.col('signal_std') > 1e-10)
            .select('signal_id')
            .to_series()
            .to_list()
        )
    else:
        active_signals = typology['signal_id'].unique().to_list()

    # Filter to active signals
    signal_col = 'signal_id' if 'signal_id' in signal_vector.columns else 'signal_name'
    signal_vector = signal_vector.filter(pl.col(signal_col).is_in(active_signals))

    # Identify features
    meta_cols = ['unit_id', 'I', 'signal_id', 'signal_name', 'n_samples', 'window_size', 'cohort']
    all_features = [c for c in signal_vector.columns if c not in meta_cols]

    if verbose:
        print(f"Active signals: {len(active_signals)}")
        print(f"Available features: {len(all_features)}")

    # Determine feature groups
    if feature_groups is None:
        feature_groups = {}
        for name, features in DEFAULT_FEATURE_GROUPS.items():
            available = [f for f in features if f in all_features]
            if len(available) >= 2:
                feature_groups[name] = available

        if not feature_groups:
            fallback = [f for f in FALLBACK_FEATURES if f in all_features]
            if len(fallback) >= 2:
                feature_groups['full'] = fallback
            else:
                feature_groups['full'] = all_features[:3] if len(all_features) >= 2 else all_features

    # Composite features (union of all groups)
    composite_features = list(set(f for features in feature_groups.values() for f in features))
    composite_features = [f for f in composite_features if f in all_features]

    if verbose:
        print(f"Feature groups: {list(feature_groups.keys())}")
        print(f"Composite features: {len(composite_features)}")
        print()

    # Require I column
    if 'I' not in signal_vector.columns:
        raise ValueError("Missing required column 'I'. Use temporal signal_vector.")

    # Determine grouping columns
    has_cohort = 'cohort' in signal_vector.columns
    group_cols = ['cohort', 'I'] if has_cohort else ['I']

    # Process each group
    results = []
    groups = signal_vector.group_by(group_cols, maintain_order=True)
    n_groups = signal_vector.select(group_cols).unique().height

    if verbose:
        print(f"Processing {n_groups} groups...")

    for i, (group_key, group) in enumerate(groups):
        if has_cohort:
            cohort, I = group_key if isinstance(group_key, tuple) else (group_key, None)
        else:
            cohort = None
            I = group_key[0] if isinstance(group_key, tuple) else group_key
        unit_id = group['unit_id'].to_list()[0] if 'unit_id' in group.columns else ''

        # Build composite matrix
        available_composite = [f for f in composite_features if f in group.columns]
        if len(available_composite) < 2:
            continue

        composite_matrix = group.select(available_composite).to_numpy()
        state = compute_centroid(composite_matrix, available_composite)

        # Build result row
        row = {
            'I': I,
            'n_signals': state['n_signals'],
        }
        if cohort:
            row['cohort'] = cohort
        if unit_id:
            row['unit_id'] = unit_id

        # Centroid per feature
        for j, feat in enumerate(available_composite):
            row[f'state_{feat}'] = state['centroid'][j]

        # Dispersion metrics
        row['mean_distance'] = state['mean_distance']
        row['max_distance'] = state['max_distance']
        row['std_distance'] = state['std_distance']

        # Per-engine centroids
        if compute_per_engine:
            for engine_name, features in feature_groups.items():
                available = [f for f in features if f in group.columns]
                if len(available) >= 2:
                    matrix = group.select(available).to_numpy()
                    engine_state = compute_centroid(matrix, available)
                    for j, feat in enumerate(available):
                        row[f'state_{engine_name}_{feat}'] = engine_state['centroid'][j]

        results.append(row)

        if verbose and (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{n_groups}...")

    # Build DataFrame
    result = pl.DataFrame(results)
    result.write_parquet(output_path)

    if verbose:
        print(f"\nShape: {result.shape}")
        print()
        print("─" * 50)
        print(f"✓ {Path(output_path).absolute()}")
        print("─" * 50)

    return result


def main():
    import sys

    if len(sys.argv) < 3:
        print("Usage: python 02_state_vector.py <signal_vector.parquet> <typology.parquet> [output.parquet]")
        sys.exit(1)

    signal_path = sys.argv[1]
    typology_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "state_vector.parquet"

    compute_state_vector(signal_path, typology_path, output_path)


if __name__ == "__main__":
    main()
