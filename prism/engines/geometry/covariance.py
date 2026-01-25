"""
Covariance Structure Engine

Computes the covariance matrix and derived metrics for multi-signal data.

Outputs:
    - covariance_matrix: Full NxN covariance matrix (as flattened array or stored separately)
    - covariance_det: Determinant of covariance matrix (volume of uncertainty ellipsoid)
    - covariance_trace: Trace (sum of variances)
    - covariance_condition: Condition number (numerical stability indicator)
    - covariance_rank: Effective rank of the covariance matrix

Math:
    Cov(X,Y) = E[(X - μX)(Y - μY)]

    For matrix X with n observations and p variables:
    Σ = (1/(n-1)) * (X - X̄)ᵀ(X - X̄)

References:
    Standard statistical covariance estimation with Bessel's correction (n-1 denominator)
"""

import numpy as np
from typing import Dict, Any, Optional
import polars as pl


def compute(
    df: pl.DataFrame = None,
    data: np.ndarray = None,
    min_samples: int = 10,
    **kwargs,
) -> Dict[str, Any]:
    """
    Compute covariance matrix and derived metrics.

    Args:
        df: Polars DataFrame with signals as columns (wide format)
            OR with signal_id column (long format, will be pivoted)
        data: Alternative: numpy array of shape (n_observations, n_signals)
        min_samples: Minimum observations required

    Returns:
        Dict with covariance metrics
    """
    # Get data matrix
    if data is not None:
        X = np.asarray(data, dtype=np.float64)
    elif df is not None:
        X = _df_to_matrix(df)
    else:
        return _null_result("No data provided")

    # Validate
    if X.ndim != 2:
        return _null_result("Data must be 2D (observations x signals)")

    n_obs, n_signals = X.shape

    if n_obs < min_samples:
        return _null_result(f"Insufficient samples: {n_obs} < {min_samples}")

    if n_signals < 2:
        return _null_result("Need at least 2 signals for covariance")

    # Remove rows with NaN
    mask = ~np.any(np.isnan(X), axis=1)
    X_clean = X[mask]

    if len(X_clean) < min_samples:
        return _null_result(f"Insufficient non-NaN samples: {len(X_clean)} < {min_samples}")

    # Compute covariance matrix
    # np.cov expects variables in rows, observations in columns
    # So we transpose: X_clean.T gives (n_signals, n_obs)
    cov_matrix = np.cov(X_clean, rowvar=False)

    # Handle single signal edge case (np.cov returns scalar)
    if cov_matrix.ndim == 0:
        cov_matrix = np.array([[cov_matrix]])

    # Derived metrics
    det = _safe_det(cov_matrix)
    trace = np.trace(cov_matrix)
    condition = _safe_condition(cov_matrix)
    rank = np.linalg.matrix_rank(cov_matrix)

    # Eigenvalues for additional metrics
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order

    return {
        'covariance_det': float(det) if det is not None else None,
        'covariance_log_det': float(np.log(det)) if det is not None and det > 0 else None,
        'covariance_trace': float(trace),
        'covariance_condition': float(condition) if condition is not None else None,
        'covariance_rank': int(rank),
        'covariance_n_signals': n_signals,
        'covariance_n_samples': len(X_clean),
        'covariance_eigenvalue_max': float(eigenvalues[0]),
        'covariance_eigenvalue_min': float(eigenvalues[-1]),
        'covariance_eigenvalue_ratio': float(eigenvalues[0] / eigenvalues[-1]) if eigenvalues[-1] > 0 else None,
    }


def compute_matrix(
    df: pl.DataFrame = None,
    data: np.ndarray = None,
    **kwargs,
) -> Optional[np.ndarray]:
    """
    Return the full covariance matrix.

    For cases where you need the actual matrix, not just summary statistics.
    """
    if data is not None:
        X = np.asarray(data, dtype=np.float64)
    elif df is not None:
        X = _df_to_matrix(df)
    else:
        return None

    mask = ~np.any(np.isnan(X), axis=1)
    X_clean = X[mask]

    if len(X_clean) < 2:
        return None

    return np.cov(X_clean, rowvar=False)


def _df_to_matrix(df: pl.DataFrame) -> np.ndarray:
    """Convert DataFrame to numpy matrix."""
    # Check if long format (has signal_id)
    if 'signal_id' in df.columns and 'value' in df.columns:
        # Pivot to wide format
        index_col = 'window' if 'window' in df.columns else 'index'
        if index_col not in df.columns:
            # Use row number as index
            df = df.with_row_index(index_col)

        wide = df.pivot(
            index=index_col,
            on='signal_id',
            values='value',
        )
        # Drop index column, keep only signal columns
        numeric_cols = [c for c in wide.columns if c != index_col]
        return wide.select(numeric_cols).to_numpy()
    else:
        # Assume already wide format - use all numeric columns
        numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found")
        return df.select(numeric_cols).to_numpy()


def _safe_det(matrix: np.ndarray) -> Optional[float]:
    """Compute determinant safely."""
    try:
        det = np.linalg.det(matrix)
        if np.isfinite(det):
            return det
        return None
    except Exception:
        return None


def _safe_condition(matrix: np.ndarray) -> Optional[float]:
    """Compute condition number safely."""
    try:
        cond = np.linalg.cond(matrix)
        if np.isfinite(cond):
            return cond
        return None
    except Exception:
        return None


def _null_result(reason: str) -> Dict[str, Any]:
    """Return null result with reason."""
    return {
        'covariance_det': None,
        'covariance_log_det': None,
        'covariance_trace': None,
        'covariance_condition': None,
        'covariance_rank': None,
        'covariance_n_signals': None,
        'covariance_n_samples': None,
        'covariance_eigenvalue_max': None,
        'covariance_eigenvalue_min': None,
        'covariance_eigenvalue_ratio': None,
        'covariance_error': reason,
    }
