"""
03: State Geometry Entry Point
==============================

Pure orchestration - calls eigendecomp engine from engines/state/eigendecomp.py.
Computes the SHAPE of the signal distribution around each state.

Stages: signal_vector.parquet + state_vector.parquet → state_geometry.parquet

Key insight: effective_dim shows 63% importance in predicting RUL.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Dict, Optional, Any

# Import the actual computation from engine
from engines.manifold.state.eigendecomp import (
    compute as compute_eigenvalues_engine,
    enforce_eigenvector_continuity,
    bootstrap_effective_dim,
)


# Feature groups
try:
    from engines.manifold.geometry.config import DEFAULT_FEATURE_GROUPS
except ImportError:
    DEFAULT_FEATURE_GROUPS = {
        'shape': ['kurtosis', 'skewness', 'crest_factor'],
        'complexity': ['permutation_entropy', 'hurst', 'acf_lag1'],
        'spectral': ['spectral_entropy', 'spectral_centroid', 'band_low_rel', 'band_mid_rel', 'band_high_rel'],
    }


def compute_eigenvalues(
    signal_matrix: np.ndarray,
    centroid: np.ndarray,
    min_signals: int = 2,
    normalize: bool = True
) -> Dict[str, Any]:
    """
    Wrapper - delegates entirely to eigendecomp engine.

    Args:
        signal_matrix: N_signals × D_features
        centroid: D_features centroid from state_vector
        min_signals: Minimum signals for reliable eigenvalues (2 = mathematical min)
        normalize: If True, Z-score normalize before SVD

    Returns:
        Eigenvalue metrics including effective_dim
    """
    norm_method = "zscore" if normalize else "none"

    result = compute_eigenvalues_engine(
        signal_matrix,
        centroid=centroid,
        norm_method=norm_method,
        min_signals=min_signals,
    )

    # Map engine output to expected format
    return {
        'eigenvalues': result['eigenvalues'],
        'explained_ratios': result['explained_ratio'],
        'total_variance': result['total_variance'],
        'effective_dim': result['effective_dim'],
        'condition_number': result['condition_number'],
        'eigenvalue_entropy': result['eigenvalue_entropy'],
        'eigenvalue_entropy_normalized': result['eigenvalue_entropy_normalized'],
        'ratio_2_1': result['ratio_2_1'],
        'ratio_3_1': result['ratio_3_1'],
        'n_signals': result['n_signals'],
        'n_features': result['n_features'],
        # Eigenvector loadings - key for pairwise gating
        'principal_components': result['principal_components'],  # Feature loadings (D x D)
        'signal_loadings': result['signal_loadings'],            # Signal loadings on PCs (N x k)
    }


def compute_state_geometry(
    signal_vector_path: str,
    state_vector_path: str,
    output_path: str = "state_geometry.parquet",
    feature_groups: Optional[Dict[str, List[str]]] = None,
    max_eigenvalues: int = 5,
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
    if verbose:
        print("=" * 70)
        print("03: STATE GEOMETRY - Eigenvalues and shape metrics")
        print("=" * 70)

    # Load data
    signal_vector = pl.read_parquet(signal_vector_path)
    state_vector = pl.read_parquet(state_vector_path)

    # Identify features
    meta_cols = ['unit_id', 'I', 'signal_id', 'cohort']
    all_features = [c for c in signal_vector.columns if c not in meta_cols]

    # Determine feature groups
    if feature_groups is None:
        feature_groups = {}
        for name, features in DEFAULT_FEATURE_GROUPS.items():
            available = [f for f in features if f in all_features]
            if len(available) >= 2:
                feature_groups[name] = available

        if not feature_groups and len(all_features) >= 2:
            feature_groups['full'] = all_features[:3]

    if verbose:
        print(f"Feature groups: {list(feature_groups.keys())}")
        print()

    # Determine grouping columns
    has_cohort = 'cohort' in signal_vector.columns
    group_cols = ['cohort', 'I'] if has_cohort else ['I']

    # Process each group
    results = []
    loading_rows = []  # Narrow sidecar for signal loadings
    feature_loading_rows = []  # Narrow sidecar for feature (PC1) loadings
    groups = signal_vector.group_by(group_cols, maintain_order=True)
    n_groups = signal_vector.select(group_cols).unique().height

    # Track previous eigenvectors per (cohort, engine) for continuity enforcement
    prev_loadings_by_key: Dict[str, np.ndarray] = {}

    if verbose:
        print(f"Processing {n_groups} groups...")

    for i, (group_key, group) in enumerate(groups):
        if has_cohort:
            cohort, I = group_key if isinstance(group_key, tuple) else (group_key, None)
        else:
            cohort = None
            I = group_key[0] if isinstance(group_key, tuple) else group_key

        # Get state vector for this group
        if has_cohort and 'cohort' in state_vector.columns:
            state_row = state_vector.filter(
                (pl.col('I') == I) & (pl.col('cohort') == cohort)
            )
        else:
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
                # Compute centroid from data
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

            # Guard: filter NaN rows
            valid_rows = np.isfinite(matrix).all(axis=1)
            matrix_clean = matrix[valid_rows]

            # Guard: need at least 2 signals for eigendecomposition (variance requires N>=2)
            if len(matrix_clean) < 2:
                if verbose and i == 0:
                    print(f"  Skipping {engine_name} at I={I}: only {len(matrix_clean)} valid rows (need >=2)")
                continue

            # Guard: skip if any column is constant (zero variance → degenerate SVD)
            col_std = np.std(matrix_clean, axis=0)
            if np.any(col_std < 1e-12):
                if verbose and i == 0:
                    const_cols = [available[k] for k in range(len(available)) if col_std[k] < 1e-12]
                    print(f"  Skipping {engine_name} at I={I}: constant columns {const_cols}")
                continue

            # Compute eigenvalues
            eigen_result = compute_eigenvalues(matrix_clean, centroid)

            # Build result row
            unit_id = group['unit_id'][0] if 'unit_id' in group.columns else ''
            row = {
                'I': I,
                'engine': engine_name,
                'n_signals': eigen_result['n_signals'],
                'n_features': eigen_result['n_features'],
            }
            if cohort:
                row['cohort'] = cohort
            if unit_id:
                row['unit_id'] = unit_id

            # Eigenvalues (fill missing slots with 0.0 — no variance on that axis)
            n_eigen = len(eigen_result['eigenvalues'])
            for j in range(max_eigenvalues):
                if j < n_eigen:
                    row[f'eigenvalue_{j+1}'] = float(eigen_result['eigenvalues'][j])
                else:
                    row[f'eigenvalue_{j+1}'] = 0.0

            # Explained ratios (fill missing slots with 0.0)
            n_explained = len(eigen_result['explained_ratios'])
            for j in range(max_eigenvalues):
                if j < n_explained:
                    row[f'explained_{j+1}'] = float(eigen_result['explained_ratios'][j])
                else:
                    row[f'explained_{j+1}'] = 0.0

            # Derived metrics
            row['effective_dim'] = eigen_result['effective_dim']
            row['total_variance'] = eigen_result['total_variance']
            row['eigenvalue_entropy'] = eigen_result['eigenvalue_entropy']
            row['eigenvalue_entropy_norm'] = eigen_result['eigenvalue_entropy_normalized']
            row['condition_number'] = eigen_result['condition_number']
            row['ratio_2_1'] = eigen_result['ratio_2_1']
            row['ratio_3_1'] = eigen_result['ratio_3_1']

            # Bootstrap confidence interval for effective_dim
            if len(matrix) >= 5:
                bs = bootstrap_effective_dim(matrix, n_bootstrap=50)
                row['eff_dim_std'] = bs['eff_dim_std']
                row['eff_dim_ci_low'] = bs['eff_dim_ci_low']
                row['eff_dim_ci_high'] = bs['eff_dim_ci_high']

            # Signal loadings on principal components (for pairwise gating)
            # signal_loadings: N_signals x k matrix (U from SVD)
            # High co-loading on same PC = signals are correlated
            signal_loadings = eigen_result.get('signal_loadings')
            signal_ids = group['signal_id'].to_list() if 'signal_id' in group.columns else []

            if signal_loadings is not None and len(signal_ids) > 0:
                # Eigenvector continuity enforcement across sequential windows
                continuity_key = f'{cohort}_{engine_name}' if cohort else engine_name
                flip_count = 0

                if continuity_key in prev_loadings_by_key:
                    prev = prev_loadings_by_key[continuity_key]
                    if prev.shape == signal_loadings.shape:
                        # Count flips before correction
                        n_pcs = min(3, signal_loadings.shape[1])
                        for pc in range(n_pcs):
                            if np.dot(prev[:, pc], signal_loadings[:, pc]) < 0:
                                flip_count += 1
                        # Apply continuity correction
                        signal_loadings = enforce_eigenvector_continuity(
                            signal_loadings, prev
                        )
                prev_loadings_by_key[continuity_key] = signal_loadings

                row['eigenvector_flip_count'] = flip_count

                # Collect per-signal loadings into narrow sidecar (not wide columns)
                n_pcs = min(3, signal_loadings.shape[1])
                for sig_idx, sig_id in enumerate(signal_ids[:len(signal_loadings)]):
                    loading_row = {
                        'I': I,
                        'engine': engine_name,
                        'signal_id': sig_id,
                        'pc1_loading': float(signal_loadings[sig_idx, 0]),
                        'pc2_loading': float(signal_loadings[sig_idx, 1]) if signal_loadings.shape[1] > 1 else 0.0,
                        'pc3_loading': float(signal_loadings[sig_idx, 2]) if signal_loadings.shape[1] > 2 else 0.0,
                    }
                    if cohort:
                        loading_row['cohort'] = cohort
                    loading_rows.append(loading_row)

                # Store signal_ids list for reference
                row['signal_ids'] = ','.join(signal_ids)

            # Feature loadings on PC1 → narrow sidecar (not wide columns)
            principal_components = eigen_result.get('principal_components')
            if principal_components is not None and len(available) > 0:
                pc1_loadings = principal_components[0] if len(principal_components) > 0 else None
                if pc1_loadings is not None:
                    for feat_idx, feat_name in enumerate(available[:len(pc1_loadings)]):
                        feat_row = {
                            'I': I,
                            'engine': engine_name,
                            'feature': feat_name,
                            'pc1_loading': float(pc1_loadings[feat_idx]),
                        }
                        if cohort:
                            feat_row['cohort'] = cohort
                        feature_loading_rows.append(feat_row)

            results.append(row)

        if verbose and (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{n_groups}...")

    # Build DataFrame
    result = pl.DataFrame(results)
    result.write_parquet(output_path)

    # Write signal loadings sidecar file (narrow schema)
    if loading_rows:
        loadings_df = pl.DataFrame(loading_rows)
        loadings_path = str(Path(output_path).parent / 'state_geometry_loadings.parquet')
        loadings_df.write_parquet(loadings_path)
        if verbose:
            print(f"Signal loadings sidecar: {loadings_df.shape} → {loadings_path}")

    # Write feature loadings sidecar file (narrow schema, replaces wide pc1_feat_* columns)
    if feature_loading_rows:
        feat_loadings_df = pl.DataFrame(feature_loading_rows)
        feat_loadings_path = str(Path(output_path).parent / 'state_geometry_feature_loadings.parquet')
        feat_loadings_df.write_parquet(feat_loadings_path)
        if verbose:
            print(f"Feature loadings sidecar: {feat_loadings_df.shape} → {feat_loadings_path}")

    if verbose:
        print(f"\nShape: {result.shape}")
        print()
        print("─" * 50)
        print(f"✓ {Path(output_path).absolute()}")
        print("─" * 50)

        # Summary per engine
        if len(result) > 0 and 'engine' in result.columns:
            for engine_name in feature_groups.keys():
                engine_data = result.filter(pl.col('engine') == engine_name)
                if len(engine_data) > 0:
                    print(f"\n{engine_name} engine:")
                    print(f"  effective_dim: mean={engine_data['effective_dim'].mean():.2f}, "
                          f"std={engine_data['effective_dim'].std():.2f}")

    return result


# Alias for run_pipeline.py compatibility
def run(
    signal_vector_path: str,
    state_vector_path: str,
    output_path: str = "state_geometry.parquet",
    verbose: bool = True,
) -> pl.DataFrame:
    """Run state geometry computation (wrapper for compute_state_geometry)."""
    return compute_state_geometry(
        signal_vector_path,
        state_vector_path,
        output_path,
        verbose=verbose,
    )


def main():
    import sys

    if len(sys.argv) < 3:
        print("Usage: python 03_state_geometry.py <signal_vector.parquet> <state_vector.parquet> [output.parquet]")
        sys.exit(1)

    signal_path = sys.argv[1]
    state_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "state_geometry.parquet"

    compute_state_geometry(signal_path, state_path, output_path)


if __name__ == "__main__":
    main()
