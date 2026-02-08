"""
Eigendecomposition Engine (State Geometry).

Computes the SHAPE of the system in behavioral space via eigenvalues.
This is HOW the system is distributed around its centroid.

state_vector = centroid (WHERE)
state_geometry = eigenvalues (SHAPE)

Key insight: effective_dim shows 63% importance in predicting
remaining useful life (RUL). Systems collapse dimensionally
before failure.
"""

import numpy as np
import polars as pl
from typing import Dict, Any, Optional, List, Literal


def compute(
    signal_matrix: np.ndarray,
    centroid: np.ndarray = None,
    norm_method: Literal["zscore", "robust", "mad", "none"] = "zscore",
    min_signals: int = 2,
) -> Dict[str, Any]:
    """
    Compute state geometry (eigenvalues) from signal matrix.

    Args:
        signal_matrix: 2D array (n_signals, n_features)
        centroid: Pre-computed centroid. If None, computed from data.
        norm_method: Normalization before SVD:
            - zscore: (x-mean)/std
            - robust: (x-median)/IQR
            - mad: (x-median)/MAD (most robust)
            - none: raw covariance
        min_signals: Minimum valid signals required (2 = mathematical minimum)

    Returns:
        dict with eigenvalues, effective_dim, derived metrics, loadings
    """
    signal_matrix = np.asarray(signal_matrix)

    if signal_matrix.ndim == 1:
        signal_matrix = signal_matrix.reshape(1, -1)

    N, D = signal_matrix.shape

    if N < min_signals:
        return _empty_result(D)

    # Remove NaN/Inf rows
    valid_mask = np.isfinite(signal_matrix).all(axis=1)
    if valid_mask.sum() < min_signals:
        return _empty_result(D)

    signal_matrix = signal_matrix[valid_mask]
    N = signal_matrix.shape[0]

    # Center around centroid
    if centroid is None:
        centroid = np.mean(signal_matrix, axis=0)
    centered = signal_matrix - centroid

    # Normalize
    if norm_method == "none":
        normalized = centered
    elif norm_method == "robust":
        q75, q25 = np.percentile(centered, [75, 25], axis=0)
        iqr = q75 - q25
        iqr = np.where(iqr < 1e-10, 1.0, iqr)
        normalized = centered / iqr
    elif norm_method == "mad":
        median = np.median(centered, axis=0)
        mad = np.median(np.abs(centered - median), axis=0)
        mad = np.where(mad < 1e-10, 1.0, mad)
        normalized = (centered - median) / mad
    else:  # zscore (default)
        std = np.std(centered, axis=0)
        std = np.where(std < 1e-10, 1.0, std)
        normalized = centered / std

    # SVD
    try:
        U, S, Vt = np.linalg.svd(normalized, full_matrices=False)
        eigenvalues = (S ** 2) / max(N - 1, 1)
    except np.linalg.LinAlgError:
        return _empty_result(D)

    # Derived metrics
    total_var = eigenvalues.sum()

    if total_var > 1e-10:
        effective_dim = (total_var ** 2) / (eigenvalues ** 2).sum()
        explained_ratio = eigenvalues / total_var

        # Eigenvalue entropy
        nonzero = eigenvalues[eigenvalues > 1e-10]
        if len(nonzero) > 1:
            p = nonzero / nonzero.sum()
            entropy = -np.sum(p * np.log(p))
            max_entropy = np.log(len(nonzero))
            entropy_norm = entropy / max_entropy if max_entropy > 0 else 0
        else:
            entropy, entropy_norm = 0.0, 0.0

        # Condition number
        if len(nonzero) >= 2:
            condition_number = nonzero[0] / nonzero[-1]
        else:
            condition_number = 1.0

        # Eigenvalue ratios
        ratio_2_1 = eigenvalues[1] / eigenvalues[0] if len(eigenvalues) >= 2 and eigenvalues[0] > 1e-10 else 0.0
        ratio_3_1 = eigenvalues[2] / eigenvalues[0] if len(eigenvalues) >= 3 and eigenvalues[0] > 1e-10 else 0.0
    else:
        effective_dim = 0.0
        explained_ratio = np.zeros_like(eigenvalues)
        entropy, entropy_norm = 0.0, 0.0
        condition_number = 1.0
        ratio_2_1, ratio_3_1 = 0.0, 0.0

    return {
        'eigenvalues': eigenvalues,
        'explained_ratio': explained_ratio,
        'total_variance': float(total_var),
        'effective_dim': float(effective_dim),
        'eigenvalue_entropy': float(entropy),
        'eigenvalue_entropy_normalized': float(entropy_norm),
        'condition_number': float(condition_number),
        'ratio_2_1': float(ratio_2_1),
        'ratio_3_1': float(ratio_3_1),
        'principal_components': Vt,   # Feature loadings (D x D)
        'signal_loadings': U,          # Signal loadings on PCs (N x min(N,D))
        'n_signals': N,
        'n_features': D,
    }


def _empty_result(D: int) -> Dict[str, Any]:
    """Return empty result for insufficient data."""
    return {
        'eigenvalues': np.full(D, np.nan),
        'explained_ratio': np.full(D, np.nan),
        'total_variance': np.nan,
        'effective_dim': np.nan,
        'eigenvalue_entropy': np.nan,
        'eigenvalue_entropy_normalized': np.nan,
        'condition_number': np.nan,
        'ratio_2_1': np.nan,
        'ratio_3_1': np.nan,
        'principal_components': None,
        'signal_loadings': None,
        'n_signals': 0,
        'n_features': D,
    }


def compute_from_signal_vector(
    signal_vector: pl.DataFrame,
    feature_columns: Optional[List[str]] = None,
    group_cols: List[str] = ['unit_id', 'I'],
    norm_method: Literal["zscore", "robust", "mad", "none"] = "zscore",
    min_signals: int = 3,
) -> pl.DataFrame:
    """
    Compute state geometry from signal_vector.parquet.

    Args:
        signal_vector: DataFrame with signal features
        feature_columns: Which columns to use as features
        group_cols: Columns to group by
        norm_method: Normalization method
        min_signals: Minimum signals per group

    Returns:
        DataFrame with eigenvalues for each group
    """
    if feature_columns is None:
        # Auto-detect numeric feature columns
        feature_columns = [
            col for col in signal_vector.columns
            if col not in ['unit_id', 'I', 'signal_id', 'cohort']
            and signal_vector[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]

    results = []

    # Group and compute geometry
    for group_keys, group_df in signal_vector.group_by(group_cols):
        # Extract feature matrix
        matrix = group_df.select(feature_columns).to_numpy()

        # Compute geometry
        geom = compute(matrix, norm_method=norm_method, min_signals=min_signals)

        # Build result row
        row = dict(zip(group_cols, group_keys if isinstance(group_keys, tuple) else [group_keys]))
        row['effective_dim'] = geom['effective_dim']
        row['total_variance'] = geom['total_variance']
        row['condition_number'] = geom['condition_number']
        row['eigenvalue_entropy'] = geom['eigenvalue_entropy']
        row['eigenvalue_entropy_normalized'] = geom['eigenvalue_entropy_normalized']
        row['ratio_2_1'] = geom['ratio_2_1']
        row['ratio_3_1'] = geom['ratio_3_1']
        row['n_signals'] = geom['n_signals']

        # Add top eigenvalues
        eig = geom['eigenvalues']
        for i in range(min(5, len(eig))):
            row[f'eigenvalue_{i}'] = eig[i] if not np.isnan(eig[i]) else None

        # Add explained ratio for top components
        exp_ratio = geom['explained_ratio']
        for i in range(min(3, len(exp_ratio))):
            row[f'explained_ratio_{i}'] = exp_ratio[i] if not np.isnan(exp_ratio[i]) else None

        results.append(row)

    return pl.DataFrame(results).sort(group_cols)


def enforce_eigenvector_continuity(
    eigenvectors_current: np.ndarray,
    eigenvectors_previous: np.ndarray,
) -> np.ndarray:
    """
    Ensure eigenvectors maintain consistent orientation across windows.

    Eigenvectors can flip sign or swap order between adjacent windows.
    This causes artificial discontinuities in projections and velocity
    decomposition.

    Parameters
    ----------
    eigenvectors_current : np.ndarray
        Current eigenvectors (d x k matrix, columns are PCs)
    eigenvectors_previous : np.ndarray
        Previous window's eigenvectors (same shape)

    Returns
    -------
    np.ndarray
        Corrected eigenvectors with consistent orientation
    """
    if eigenvectors_previous is None:
        return eigenvectors_current

    if eigenvectors_current.shape != eigenvectors_previous.shape:
        return eigenvectors_current

    corrected = eigenvectors_current.copy()
    n_vecs = corrected.shape[1] if corrected.ndim == 2 else 1

    if corrected.ndim == 1:
        # Single eigenvector
        if np.dot(corrected, eigenvectors_previous) < 0:
            corrected *= -1
        return corrected

    # Check each eigenvector for sign flip
    for j in range(n_vecs):
        dot = np.dot(eigenvectors_previous[:, j], corrected[:, j])
        if dot < 0:
            corrected[:, j] *= -1  # Flip sign

    return corrected


def bootstrap_effective_dim(
    signal_matrix: np.ndarray,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """
    Bootstrap confidence interval for effective dimensionality.

    Resample rows of feature matrix, compute eigendecomp each time.
    Return mean, std, and CI of eff_dim.

    Parameters
    ----------
    signal_matrix : np.ndarray
        Signal matrix (n_signals x n_features)
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level for CI (0.95 = 95% CI)

    Returns
    -------
    dict with:
        eff_dim_mean : float
        eff_dim_std : float
        eff_dim_ci_low : float
        eff_dim_ci_high : float
    """
    signal_matrix = np.asarray(signal_matrix)

    if signal_matrix.ndim == 1:
        signal_matrix = signal_matrix.reshape(1, -1)

    N, D = signal_matrix.shape

    if N < 3:
        return {
            'eff_dim_mean': np.nan,
            'eff_dim_std': np.nan,
            'eff_dim_ci_low': np.nan,
            'eff_dim_ci_high': np.nan,
        }

    eff_dims = []

    for _ in range(n_bootstrap):
        # Resample rows with replacement
        idx = np.random.choice(N, size=N, replace=True)
        bootstrap_sample = signal_matrix[idx]

        # Remove duplicates (can cause singular covariance)
        unique_rows = np.unique(bootstrap_sample, axis=0)
        if len(unique_rows) < 2:
            continue

        # Compute covariance and eigenvalues
        try:
            centered = unique_rows - np.mean(unique_rows, axis=0)
            cov = np.cov(centered.T)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            eigenvalues = np.linalg.eigvalsh(cov)[::-1]
        except Exception:
            continue

        # Compute effective dimension
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if len(eigenvalues) == 0:
            continue

        total_var = eigenvalues.sum()
        if total_var > 1e-10:
            eff_dim = (total_var ** 2) / (eigenvalues ** 2).sum()
            eff_dims.append(eff_dim)

    if not eff_dims:
        return {
            'eff_dim_mean': np.nan,
            'eff_dim_std': np.nan,
            'eff_dim_ci_low': np.nan,
            'eff_dim_ci_high': np.nan,
        }

    eff_dims = np.array(eff_dims)
    alpha = 1 - confidence_level
    ci_low = np.percentile(eff_dims, alpha / 2 * 100)
    ci_high = np.percentile(eff_dims, (1 - alpha / 2) * 100)

    return {
        'eff_dim_mean': float(np.mean(eff_dims)),
        'eff_dim_std': float(np.std(eff_dims)),
        'eff_dim_ci_low': float(ci_low),
        'eff_dim_ci_high': float(ci_high),
    }


def enforce_eigenvector_continuity_sequence(
    eigenvectors_sequence: list,
) -> list:
    """
    Ensure eigenvector continuity across a sequence of windows.

    Parameters
    ----------
    eigenvectors_sequence : list
        List of eigenvector matrices from sequential windows

    Returns
    -------
    list
        Corrected eigenvector matrices
    """
    if not eigenvectors_sequence:
        return eigenvectors_sequence

    corrected = [eigenvectors_sequence[0]]

    for i in range(1, len(eigenvectors_sequence)):
        prev = corrected[i - 1]
        curr = eigenvectors_sequence[i]

        if prev is None or curr is None:
            corrected.append(curr)
            continue

        corrected.append(enforce_eigenvector_continuity(curr, prev))

    return corrected


def compute_effective_dim_trend(
    effective_dims: np.ndarray,
) -> Dict[str, float]:
    """
    Compute trend statistics on effective dimension over time.

    Returns numbers only - ORTHON interprets what "collapsing" means.

    Args:
        effective_dims: Array of effective_dim values over time

    Returns:
        dict with slope, r2
    """
    valid = ~np.isnan(effective_dims)
    if np.sum(valid) < 4:
        return {
            'eff_dim_slope': np.nan,
            'eff_dim_r2': np.nan,
        }

    x = np.arange(len(effective_dims))[valid]
    y = effective_dims[valid]

    slope, intercept = np.polyfit(x, y, 1)

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        'eff_dim_slope': float(slope),
        'eff_dim_r2': float(r2),
    }
