"""
Centroid Engine (State Vector).

Computes the state vector as the centroid of all signals in feature space.
This is WHERE the system is in behavioral space.

state_vector = centroid
state_geometry = eigenvalues (separate engine)
"""

import numpy as np
import polars as pl
from typing import Dict, Any, Optional, List


def compute(signal_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute state vector (centroid) from signal matrix.
    
    Args:
        signal_matrix: 2D array of shape (n_signals, n_features)
                      Each row is a signal's feature vector
                      
    Returns:
        dict with 'centroid' array of shape (n_features,)
    """
    signal_matrix = np.asarray(signal_matrix)
    
    if signal_matrix.ndim == 1:
        signal_matrix = signal_matrix.reshape(1, -1)
    
    # Handle NaN: compute mean ignoring NaN
    centroid = np.nanmean(signal_matrix, axis=0)
    
    return {
        'centroid': centroid,
        'n_signals': signal_matrix.shape[0],
        'n_features': signal_matrix.shape[1],
    }


def compute_from_signal_vector(
    signal_vector: pl.DataFrame,
    feature_columns: Optional[List[str]] = None,
    group_cols: List[str] = ['unit_id', 'I'],
) -> pl.DataFrame:
    """
    Compute state vector (centroid) from signal_vector.parquet.
    
    Args:
        signal_vector: DataFrame with signal features
        feature_columns: Which columns to use as features
        group_cols: Columns to group by (usually unit_id, I)
        
    Returns:
        DataFrame with centroid for each group
    """
    if feature_columns is None:
        # Auto-detect numeric feature columns
        feature_columns = [
            col for col in signal_vector.columns
            if col not in ['unit_id', 'I', 'signal_id', 'cohort']
            and signal_vector[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]
    
    # Compute mean of each feature across signals
    agg_exprs = [
        pl.col(col).mean().alias(col)
        for col in feature_columns
    ]
    
    # Also count signals
    agg_exprs.append(pl.count().alias('n_signals'))
    
    result = (
        signal_vector
        .group_by(group_cols)
        .agg(agg_exprs)
        .sort(group_cols)
    )
    
    return result


def compute_weighted(
    signal_matrix: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute weighted centroid.
    
    Useful when some signals are more important than others.
    
    Args:
        signal_matrix: 2D array (n_signals, n_features)
        weights: 1D array (n_signals,) of weights
        
    Returns:
        dict with 'centroid' array
    """
    signal_matrix = np.asarray(signal_matrix)
    weights = np.asarray(weights)
    
    if signal_matrix.ndim == 1:
        signal_matrix = signal_matrix.reshape(1, -1)
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Weighted mean
    centroid = np.average(signal_matrix, axis=0, weights=weights)
    
    return {
        'centroid': centroid,
        'n_signals': signal_matrix.shape[0],
        'n_features': signal_matrix.shape[1],
    }
