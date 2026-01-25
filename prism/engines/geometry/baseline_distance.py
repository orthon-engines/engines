"""
Baseline Distance Engine (formerly healthy_distance)

Computes Mahalanobis distance from baseline covariance.

Definition:
    Baseline = first N% of windows (configurable)
    Distance = Mahalanobis distance from baseline mean using baseline covariance

Math:
    d(x) = √[(x - μ_baseline)ᵀ Σ_baseline⁻¹ (x - μ_baseline)]

Config:
    baseline_fraction: float (default 0.1) - Use first 10% as baseline
    baseline_windows: int (optional) - Override fraction with explicit count

Output:
    baseline_distance: Mahalanobis distance from baseline
    baseline_windows_used: Number of windows in baseline calculation
"""

import numpy as np
from typing import Dict, Any, Optional, List, Union
import polars as pl


def compute(
    df: pl.DataFrame = None,
    data: np.ndarray = None,
    config: Dict[str, Any] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Compute baseline distance for all windows.

    Args:
        df: DataFrame with signals as columns, rows as windows
        data: Alternative: numpy array (n_windows, n_signals)
        config: Contains baseline_fraction or baseline_windows

    Returns:
        Dict with baseline_distance per window and metadata
    """
    if config is None:
        config = {}

    # Get data matrix
    if data is not None:
        X = np.asarray(data, dtype=np.float64)
    elif df is not None:
        X = _df_to_matrix(df)
    else:
        return _null_result("No data provided")

    if X.ndim != 2:
        return _null_result("Data must be 2D (windows x signals)")

    n_windows, n_signals = X.shape

    if n_windows < 10:
        return _null_result(f"Insufficient windows: {n_windows} < 10")

    # Determine baseline windows
    baseline_windows = config.get('baseline_windows')
    if baseline_windows is None:
        baseline_fraction = config.get('baseline_fraction', 0.1)
        baseline_windows = max(20, int(n_windows * baseline_fraction))

    # Can't use more than half the data for baseline
    baseline_windows = min(baseline_windows, n_windows // 2)

    # Ensure minimum samples for covariance
    baseline_windows = max(baseline_windows, n_signals + 1)

    if baseline_windows >= n_windows:
        return _null_result(f"Not enough windows for baseline: need > {baseline_windows}")

    # Extract baseline data
    baseline_data = X[:baseline_windows]

    # Remove NaN rows from baseline
    baseline_mask = ~np.any(np.isnan(baseline_data), axis=1)
    baseline_clean = baseline_data[baseline_mask]

    if len(baseline_clean) < n_signals + 1:
        return _null_result(f"Insufficient clean baseline samples: {len(baseline_clean)}")

    # Compute baseline statistics
    baseline_mean = np.mean(baseline_clean, axis=0)
    baseline_cov = np.cov(baseline_clean, rowvar=False)

    # Handle scalar covariance (single signal)
    if baseline_cov.ndim == 0:
        baseline_cov = np.array([[baseline_cov]])

    # Regularize covariance if ill-conditioned
    cond = np.linalg.cond(baseline_cov)
    if cond > 1e10:
        baseline_cov += np.eye(len(baseline_cov)) * 1e-6

    # Invert covariance
    try:
        baseline_cov_inv = np.linalg.inv(baseline_cov)
    except np.linalg.LinAlgError:
        return _null_result("Singular covariance matrix")

    # Compute Mahalanobis distance for each window
    distances = []
    for i in range(n_windows):
        x = X[i]
        if np.any(np.isnan(x)):
            distances.append(np.nan)
        else:
            diff = x - baseline_mean
            d_squared = diff @ baseline_cov_inv @ diff
            d = np.sqrt(max(0, d_squared))  # Ensure non-negative
            distances.append(d)

    distances = np.array(distances)

    return {
        'baseline_distance': distances.tolist(),
        'baseline_distance_mean': float(np.nanmean(distances)),
        'baseline_distance_max': float(np.nanmax(distances)),
        'baseline_distance_final': float(distances[-1]) if not np.isnan(distances[-1]) else None,
        'baseline_windows_used': int(len(baseline_clean)),
        'baseline_cov_condition': float(cond),
        'n_windows': n_windows,
        'n_signals': n_signals,
    }


def compute_single(
    x: np.ndarray,
    baseline_mean: np.ndarray,
    baseline_cov_inv: np.ndarray,
) -> float:
    """
    Compute baseline distance for a single observation.

    Args:
        x: Single observation vector
        baseline_mean: Baseline mean vector
        baseline_cov_inv: Inverse of baseline covariance matrix

    Returns:
        Mahalanobis distance
    """
    diff = x - baseline_mean
    d_squared = diff @ baseline_cov_inv @ diff
    return np.sqrt(max(0, d_squared))


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
        'baseline_distance': None,
        'baseline_distance_mean': None,
        'baseline_distance_max': None,
        'baseline_distance_final': None,
        'baseline_windows_used': None,
        'baseline_cov_condition': None,
        'n_windows': None,
        'n_signals': None,
        'baseline_distance_error': reason,
    }
