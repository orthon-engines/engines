"""
Correlation Structure Engine

Computes eigenvalue decomposition of correlation matrix.

Definition:
    Correlation matrix R where R_ij = Cov(X_i, X_j) / (σ_i × σ_j)
    Eigenvalues reveal how variance is distributed across principal directions

Math:
    R = D⁻¹ Σ D⁻¹  where D = diag(σ₁, σ₂, ...)
    Eigendecomposition: R = V Λ Vᵀ

Output:
    largest_eigenvalue: λ₁ (dominance of first PC)
    eigenvalue_ratio: λ₁/λₙ (spread of eigenvalues)
    condition_number: Same as ratio for correlation
    explained_by_first: λ₁/Σλ (fraction of variance in first PC)
    explained_by_top3: (λ₁+λ₂+λ₃)/Σλ
"""

import numpy as np
from typing import Dict, Any, Optional
import polars as pl


def compute(
    df: pl.DataFrame = None,
    data: np.ndarray = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Compute correlation structure metrics.

    Args:
        df: DataFrame with signals as columns for one window
        data: Alternative: numpy array (n_observations, n_signals)

    Returns:
        Dict with correlation structure metrics
    """
    # Get data matrix
    if data is not None:
        X = np.asarray(data, dtype=np.float64)
    elif df is not None:
        X = _df_to_matrix(df)
    else:
        return _null_result("No data provided")

    if X.ndim != 2:
        return _null_result("Data must be 2D")

    n_obs, n_signals = X.shape

    if n_obs < 3:
        return _null_result(f"Insufficient observations: {n_obs} < 3")

    if n_signals < 2:
        return _null_result("Need at least 2 signals for correlation")

    # Remove NaN rows
    mask = ~np.any(np.isnan(X), axis=1)
    X_clean = X[mask]

    if len(X_clean) < 3:
        return _null_result(f"Insufficient clean observations: {len(X_clean)}")

    # Compute correlation matrix
    # np.corrcoef expects variables in rows
    corr_matrix = np.corrcoef(X_clean, rowvar=False)

    # Handle NaN in correlation (constant columns)
    if np.any(np.isnan(corr_matrix)):
        # Replace NaN with 0 (no correlation with constant)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        # Ensure diagonal is 1
        np.fill_diagonal(corr_matrix, 1.0)

    # Eigenvalue decomposition (eigvalsh for symmetric matrices)
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order

    # Filter small/negative eigenvalues (numerical precision)
    eigenvalues = np.maximum(eigenvalues, 0)

    n = len(eigenvalues)
    total = eigenvalues.sum()

    if total == 0:
        return _null_result("Sum of eigenvalues is zero")

    # Compute metrics
    largest = eigenvalues[0]
    smallest = eigenvalues[-1] if eigenvalues[-1] > 1e-10 else 1e-10

    return {
        'corr_largest_eigenvalue': float(largest),
        'corr_smallest_eigenvalue': float(eigenvalues[-1]),
        'corr_eigenvalue_ratio': float(largest / smallest),
        'corr_condition_number': float(largest / smallest),
        'corr_explained_first': float(largest / total),
        'corr_explained_top3': float(eigenvalues[:min(3, n)].sum() / total),
        'corr_explained_top5': float(eigenvalues[:min(5, n)].sum() / total) if n >= 5 else None,
        'corr_n_signals': n_signals,
        'corr_n_observations': len(X_clean),
        'corr_mean_abs': float(np.mean(np.abs(corr_matrix[np.triu_indices(n, k=1)]))),
        'corr_max_abs_offdiag': float(np.max(np.abs(corr_matrix[np.triu_indices(n, k=1)]))),
    }


def compute_matrix(
    df: pl.DataFrame = None,
    data: np.ndarray = None,
    **kwargs,
) -> Optional[np.ndarray]:
    """
    Return the full correlation matrix.
    """
    if data is not None:
        X = np.asarray(data, dtype=np.float64)
    elif df is not None:
        X = _df_to_matrix(df)
    else:
        return None

    mask = ~np.any(np.isnan(X), axis=1)
    X_clean = X[mask]

    if len(X_clean) < 3:
        return None

    corr_matrix = np.corrcoef(X_clean, rowvar=False)

    if np.any(np.isnan(corr_matrix)):
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        np.fill_diagonal(corr_matrix, 1.0)

    return corr_matrix


def _df_to_matrix(df: pl.DataFrame) -> np.ndarray:
    """Convert DataFrame to numpy matrix."""
    if 'signal_id' in df.columns and 'value' in df.columns:
        index_col = 'window' if 'window' in df.columns else 'index'
        if index_col not in df.columns:
            df = df.with_row_index(index_col)

        wide = df.pivot(
            index=index_col,
            on='signal_id',
            values='value',
        )
        numeric_cols = [c for c in wide.columns if c != index_col]
        return wide.select(numeric_cols).to_numpy()
    else:
        numeric_cols = [c for c in df.columns
                        if df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
                        and c not in ['window', 'entity_id', 'index']]
        if not numeric_cols:
            raise ValueError("No numeric signal columns found")
        return df.select(numeric_cols).to_numpy()


def _null_result(reason: str) -> Dict[str, Any]:
    """Return null result with reason."""
    return {
        'corr_largest_eigenvalue': None,
        'corr_smallest_eigenvalue': None,
        'corr_eigenvalue_ratio': None,
        'corr_condition_number': None,
        'corr_explained_first': None,
        'corr_explained_top3': None,
        'corr_explained_top5': None,
        'corr_n_signals': None,
        'corr_n_observations': None,
        'corr_mean_abs': None,
        'corr_max_abs_offdiag': None,
        'corr_structure_error': reason,
    }
