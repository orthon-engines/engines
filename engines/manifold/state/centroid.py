"""
Centroid Engine (State Vector).

Computes the state vector as the centroid of all signals in feature space.
This is WHERE the system is in behavioral space.

state_vector = centroid + dispersion metrics
state_geometry = eigenvalues (separate engine)
"""

import numpy as np
import polars as pl
from typing import Dict, Any, Optional, List


def compute(signal_matrix: np.ndarray, min_signals: int = 2) -> Dict[str, Any]:
    """
    Compute state vector (centroid) and dispersion metrics.

    Args:
        signal_matrix: 2D array of shape (n_signals, n_features)
        min_signals: Minimum valid signals required

    Returns:
        dict with centroid, n_signals, and distance metrics
    """
    signal_matrix = np.asarray(signal_matrix)

    if signal_matrix.ndim == 1:
        signal_matrix = signal_matrix.reshape(1, -1)

    N, D = signal_matrix.shape

    # Remove NaN/Inf rows
    valid_mask = np.isfinite(signal_matrix).all(axis=1)
    n_valid = valid_mask.sum()

    if n_valid < min_signals:
        return {
            'centroid': np.full(D, np.nan),
            'n_signals': 0,
            'n_features': D,
            'mean_distance': np.nan,
            'max_distance': np.nan,
            'std_distance': np.nan,
        }

    signal_matrix = signal_matrix[valid_mask]

    # Centroid = mean position
    centroid = np.mean(signal_matrix, axis=0)

    # Distance metrics (dispersion around centroid)
    centered = signal_matrix - centroid
    distances = np.linalg.norm(centered, axis=1)

    return {
        'centroid': centroid,
        'n_signals': int(n_valid),
        'n_features': D,
        'mean_distance': float(np.mean(distances)),
        'max_distance': float(np.max(distances)),
        'std_distance': float(np.std(distances)),
    }


def compute_from_signal_vector(
    signal_vector: pl.DataFrame,
    feature_columns: Optional[List[str]] = None,
    group_cols: List[str] = ['unit_id', 'I'],
    min_signals: int = 2,
) -> pl.DataFrame:
    """
    Compute state vector (centroid) from signal_vector.parquet.

    Args:
        signal_vector: DataFrame with signal features
        feature_columns: Which columns to use as features
        group_cols: Columns to group by (usually unit_id, I)
        min_signals: Minimum signals required per group

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

    results = []

    for group_key, group in signal_vector.group_by(group_cols, maintain_order=True):
        matrix = group.select(feature_columns).to_numpy()
        state = compute(matrix, min_signals=min_signals)

        row = dict(zip(group_cols, group_key if isinstance(group_key, tuple) else [group_key]))
        row['n_signals'] = state['n_signals']
        row['mean_distance'] = state['mean_distance']
        row['max_distance'] = state['max_distance']
        row['std_distance'] = state['std_distance']

        # Add centroid values
        for i, col in enumerate(feature_columns):
            row[col] = state['centroid'][i]

        results.append(row)

    return pl.DataFrame(results).sort(group_cols)


def compute_weighted(
    signal_matrix: np.ndarray,
    weights: np.ndarray,
    min_signals: int = 2,
) -> Dict[str, Any]:
    """
    Compute weighted centroid.

    Useful when some signals are more important than others.

    Args:
        signal_matrix: 2D array (n_signals, n_features)
        weights: 1D array (n_signals,) of weights
        min_signals: Minimum valid signals required

    Returns:
        dict with centroid, n_signals, and distance metrics
    """
    signal_matrix = np.asarray(signal_matrix)
    weights = np.asarray(weights)

    if signal_matrix.ndim == 1:
        signal_matrix = signal_matrix.reshape(1, -1)

    N, D = signal_matrix.shape

    # Remove NaN/Inf rows
    valid_mask = np.isfinite(signal_matrix).all(axis=1)
    n_valid = valid_mask.sum()

    if n_valid < min_signals:
        return {
            'centroid': np.full(D, np.nan),
            'n_signals': 0,
            'n_features': D,
            'mean_distance': np.nan,
            'max_distance': np.nan,
            'std_distance': np.nan,
        }

    signal_matrix = signal_matrix[valid_mask]
    weights = weights[valid_mask]

    # Normalize weights
    weights = weights / np.sum(weights)

    # Weighted mean
    centroid = np.average(signal_matrix, axis=0, weights=weights)

    # Distance metrics
    centered = signal_matrix - centroid
    distances = np.linalg.norm(centered, axis=1)

    return {
        'centroid': centroid,
        'n_signals': int(n_valid),
        'n_features': D,
        'mean_distance': float(np.mean(distances)),
        'max_distance': float(np.max(distances)),
        'std_distance': float(np.std(distances)),
    }
