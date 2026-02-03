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
from typing import Dict, Any, Optional, List, Tuple


def compute(signal_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Compute state geometry (eigenvalues) from signal matrix.
    
    Args:
        signal_matrix: 2D array of shape (n_signals, n_features)
                      Each row is a signal's feature vector
                      
    Returns:
        dict with eigenvalues, effective_dim, explained_variance_ratio, etc.
    """
    signal_matrix = np.asarray(signal_matrix)
    
    if signal_matrix.ndim == 1:
        signal_matrix = signal_matrix.reshape(1, -1)
    
    n_signals, n_features = signal_matrix.shape
    
    if n_signals < 2:
        return _empty_result(n_features)
    
    # Remove NaN rows
    valid_mask = ~np.any(np.isnan(signal_matrix), axis=1)
    signal_matrix = signal_matrix[valid_mask]
    n_signals = signal_matrix.shape[0]
    
    if n_signals < 2:
        return _empty_result(n_features)
    
    # Center the data
    centroid = np.mean(signal_matrix, axis=0)
    centered = signal_matrix - centroid
    
    # SVD approach (more stable than computing covariance)
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        # Eigenvalues from singular values
        # eigenvalue = (singular_value^2) / (n - 1)
        eigenvalues = (S ** 2) / (n_signals - 1)
        
        # Effective dimension (participation ratio)
        # effective_dim = (sum(lambda))^2 / sum(lambda^2)
        total_var = np.sum(eigenvalues)
        if total_var > 0:
            effective_dim = (total_var ** 2) / np.sum(eigenvalues ** 2)
        else:
            effective_dim = 0.0
        
        # Explained variance ratio
        if total_var > 0:
            explained_ratio = eigenvalues / total_var
        else:
            explained_ratio = np.zeros_like(eigenvalues)
        
        # Condition number (ratio of largest to smallest eigenvalue)
        nonzero_eig = eigenvalues[eigenvalues > 1e-10]
        if len(nonzero_eig) >= 2:
            condition_number = nonzero_eig[0] / nonzero_eig[-1]
        else:
            condition_number = 1.0
        
        return {
            'eigenvalues': eigenvalues,
            'effective_dim': float(effective_dim),
            'total_variance': float(total_var),
            'explained_ratio': explained_ratio,
            'condition_number': float(condition_number),
            'n_signals': n_signals,
            'n_features': n_features,
        }
        
    except np.linalg.LinAlgError:
        return _empty_result(n_features)


def _empty_result(n_features: int) -> Dict[str, Any]:
    """Return empty result."""
    return {
        'eigenvalues': np.full(n_features, np.nan),
        'effective_dim': np.nan,
        'total_variance': np.nan,
        'explained_ratio': np.full(n_features, np.nan),
        'condition_number': np.nan,
        'n_signals': 0,
        'n_features': n_features,
    }


def compute_from_signal_vector(
    signal_vector: pl.DataFrame,
    feature_columns: Optional[List[str]] = None,
    group_cols: List[str] = ['unit_id', 'I'],
) -> pl.DataFrame:
    """
    Compute state geometry from signal_vector.parquet.
    
    Args:
        signal_vector: DataFrame with signal features
        feature_columns: Which columns to use as features
        group_cols: Columns to group by
        
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
        geom = compute(matrix)
        
        # Build result row
        row = dict(zip(group_cols, group_keys if isinstance(group_keys, tuple) else [group_keys]))
        row['effective_dim'] = geom['effective_dim']
        row['total_variance'] = geom['total_variance']
        row['condition_number'] = geom['condition_number']
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
