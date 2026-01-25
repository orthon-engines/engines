"""
Effective Dimensionality Engine

Computes the effective (intrinsic) dimensionality of multi-signal data.

The key metric is the PARTICIPATION RATIO, which measures how many
dimensions meaningfully contribute to the variance.

Outputs:
    - effective_dim: Participation ratio (primary metric)
    - effective_dim_90: Dimensions needed for 90% variance
    - effective_dim_95: Dimensions needed for 95% variance
    - effective_dim_99: Dimensions needed for 99% variance

Math:
    Participation Ratio (PR):
        PR = (Σλᵢ)² / Σλᵢ²

    Where λᵢ are eigenvalues of the covariance matrix.

    Interpretation:
        - If all eigenvalues equal: PR = n (full dimensionality)
        - If one eigenvalue dominates: PR → 1 (effectively 1D)
        - PR gives "effective number of dimensions"

    This is equivalent to the inverse of the Herfindahl-Hirschman Index
    applied to normalized eigenvalues.

References:
    - Gao et al. (2017) "A theory of multineuronal dimensionality"
    - Standard PCA dimensionality analysis
"""

import numpy as np
from typing import Dict, Any, Optional
import polars as pl


def compute(
    df: pl.DataFrame = None,
    data: np.ndarray = None,
    eigenvalues: np.ndarray = None,
    min_samples: int = 10,
    **kwargs,
) -> Dict[str, Any]:
    """
    Compute effective dimensionality metrics.

    Args:
        df: Polars DataFrame with signals as columns
        data: Alternative: numpy array of shape (n_observations, n_signals)
        eigenvalues: Alternative: pre-computed eigenvalues (skip covariance computation)
        min_samples: Minimum observations required

    Returns:
        Dict with effective dimensionality metrics
    """
    # Get eigenvalues
    if eigenvalues is not None:
        eigs = np.asarray(eigenvalues, dtype=np.float64)
    else:
        eigs = _compute_eigenvalues(df=df, data=data, min_samples=min_samples)

    if eigs is None:
        return _null_result("Could not compute eigenvalues")

    # Filter to positive eigenvalues (numerical precision)
    eigs = eigs[eigs > 1e-10]

    if len(eigs) == 0:
        return _null_result("No positive eigenvalues")

    # Sort descending
    eigs = np.sort(eigs)[::-1]

    # Participation ratio: (Σλ)² / Σλ²
    sum_eigs = np.sum(eigs)
    sum_eigs_sq = np.sum(eigs ** 2)

    if sum_eigs_sq == 0:
        return _null_result("Sum of squared eigenvalues is zero")

    participation_ratio = (sum_eigs ** 2) / sum_eigs_sq

    # Variance explained thresholds
    cumvar = np.cumsum(eigs) / sum_eigs
    dim_90 = int(np.searchsorted(cumvar, 0.90) + 1)
    dim_95 = int(np.searchsorted(cumvar, 0.95) + 1)
    dim_99 = int(np.searchsorted(cumvar, 0.99) + 1)

    # Cap at actual dimensionality
    n_dims = len(eigs)
    dim_90 = min(dim_90, n_dims)
    dim_95 = min(dim_95, n_dims)
    dim_99 = min(dim_99, n_dims)

    # Normalized entropy of eigenvalue distribution (alternative measure)
    # Higher entropy = more uniform = higher effective dim
    eigs_norm = eigs / sum_eigs
    entropy = -np.sum(eigs_norm * np.log(eigs_norm + 1e-10))
    max_entropy = np.log(n_dims)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    return {
        'effective_dim': float(participation_ratio),
        'effective_dim_90': dim_90,
        'effective_dim_95': dim_95,
        'effective_dim_99': dim_99,
        'effective_dim_total': n_dims,
        'effective_dim_ratio': float(participation_ratio / n_dims),
        'effective_dim_entropy': float(normalized_entropy),
        'variance_top1': float(eigs[0] / sum_eigs),
        'variance_top3': float(np.sum(eigs[:3]) / sum_eigs) if n_dims >= 3 else float(np.sum(eigs) / sum_eigs),
    }


def _compute_eigenvalues(
    df: pl.DataFrame = None,
    data: np.ndarray = None,
    min_samples: int = 10,
) -> Optional[np.ndarray]:
    """Compute eigenvalues of covariance matrix."""
    # Get data matrix
    if data is not None:
        X = np.asarray(data, dtype=np.float64)
    elif df is not None:
        X = _df_to_matrix(df)
    else:
        return None

    if X.ndim != 2:
        return None

    n_obs, n_signals = X.shape

    if n_obs < min_samples or n_signals < 1:
        return None

    # Remove NaN rows
    mask = ~np.any(np.isnan(X), axis=1)
    X_clean = X[mask]

    if len(X_clean) < min_samples:
        return None

    # Compute covariance matrix
    cov_matrix = np.cov(X_clean, rowvar=False)

    # Handle scalar case
    if cov_matrix.ndim == 0:
        return np.array([float(cov_matrix)])

    # Eigenvalues only (eigvalsh is faster than eig for symmetric matrices)
    try:
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        return eigenvalues
    except Exception:
        return None


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
        numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found")
        return df.select(numeric_cols).to_numpy()


def _null_result(reason: str) -> Dict[str, Any]:
    """Return null result with reason."""
    return {
        'effective_dim': None,
        'effective_dim_90': None,
        'effective_dim_95': None,
        'effective_dim_99': None,
        'effective_dim_total': None,
        'effective_dim_ratio': None,
        'effective_dim_entropy': None,
        'variance_top1': None,
        'variance_top3': None,
        'effective_dim_error': reason,
    }
