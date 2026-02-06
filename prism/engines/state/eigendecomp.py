"""
Eigendecomposition Engine (State Geometry).

Computes the SHAPE of the system in behavioral space via eigenvalues.
This is HOW the system is distributed around its centroid.

state_vector = centroid (WHERE)
state_geometry = eigenvalues (SHAPE)

Key insight (Avery Rudder): effective_dim shows 63% importance
in predicting remaining useful life (RUL). Systems collapse
dimensionally before failure.
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
