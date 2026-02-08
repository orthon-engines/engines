"""
Saddle Point Detection Engine.

Detects saddle points and measures proximity to unstable equilibria.
Saddle points indicate where the system can transition between basins.

ENGINES computes metrics. ORTHON interprets:
    - High saddle_score = near unstable equilibrium
    - Approaching saddle = possible regime transition
    - Crossed separatrix = changed attractor basin
"""

import numpy as np
from typing import Dict, Any, Optional, List

from engines.primitives.embedding import (
    time_delay_embedding,
    optimal_delay,
    optimal_dimension,
)
from engines.primitives.dynamical.saddle import (
    detect_saddle_points,
    classify_jacobian_eigenvalues,
    compute_separatrix_distance,
    compute_basin_stability,
)


def compute(
    y: np.ndarray,
    min_samples: int = 100,
    emb_dim: Optional[int] = None,
    emb_tau: Optional[int] = None,
    velocity_threshold: float = 0.1,
    n_neighbors: int = None,
) -> Dict[str, Any]:
    """
    Detect saddle points and compute proximity metrics.

    Args:
        y: Signal values
        min_samples: Minimum samples required
        emb_dim: Embedding dimension (auto if None)
        emb_tau: Embedding delay (auto if None)
        velocity_threshold: Threshold for "near equilibrium"
        n_neighbors: Neighbors for Jacobian estimation

    Returns:
        dict with saddle_score, saddle_indices, stability metrics
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    if n < min_samples:
        return _empty_result()

    try:
        # Auto-detect embedding parameters
        if emb_tau is None:
            emb_tau = optimal_delay(y, max_lag=min(100, n // 10))
        if emb_dim is None:
            emb_dim = optimal_dimension(y, emb_tau, max_dim=10)

        # Embed signal
        embedded = time_delay_embedding(y, dimension=emb_dim, delay=emb_tau)

        if len(embedded) < 50:
            return _empty_result()

        if n_neighbors is None:
            n_neighbors = 2 * emb_dim + 1

        # Detect saddle points
        saddle_score, velocity, saddle_info = detect_saddle_points(
            embedded,
            velocity_threshold=velocity_threshold,
            n_neighbors=n_neighbors,
        )

        # Find saddle indices (score > 0.5)
        saddle_mask = saddle_score > 0.5
        saddle_indices = np.where(saddle_mask)[0]

        # Compute basin stability
        basin_stability = compute_basin_stability(
            embedded,
            saddle_score,
            window=min(50, len(embedded) // 4),
        )

        # Current state metrics
        current_idx = len(saddle_score) - 1
        while current_idx >= 0 and np.isnan(saddle_score[current_idx]):
            current_idx -= 1

        if current_idx >= 0:
            current_score = saddle_score[current_idx]
            current_stability = basin_stability[current_idx] if not np.isnan(basin_stability[current_idx]) else None
            current_info = saddle_info[current_idx] if current_idx < len(saddle_info) else {}
            current_stability_type = current_info.get('stability_type', 'unknown')
        else:
            current_score = None
            current_stability = None
            current_stability_type = 'unknown'

        # Separatrix distance
        if len(saddle_indices) > 0:
            sep_distance = compute_separatrix_distance(embedded, saddle_indices)
            sep_distance_current = float(sep_distance[-1]) if not np.isnan(sep_distance[-1]) else None
        else:
            sep_distance = np.full(len(embedded), np.nan)
            sep_distance_current = None

        # Statistics
        valid_scores = saddle_score[~np.isnan(saddle_score)]
        valid_stability = basin_stability[~np.isnan(basin_stability)]

        return {
            'saddle_score': saddle_score,
            'saddle_score_mean': float(np.mean(valid_scores)) if len(valid_scores) > 0 else None,
            'saddle_score_max': float(np.max(valid_scores)) if len(valid_scores) > 0 else None,
            'saddle_score_current': float(current_score) if current_score is not None else None,
            'velocity': velocity,
            'saddle_indices': saddle_indices,
            'n_saddle_points': len(saddle_indices),
            'saddle_fraction': float(len(saddle_indices) / len(saddle_score)) if len(saddle_score) > 0 else 0.0,
            'basin_stability': basin_stability,
            'basin_stability_mean': float(np.mean(valid_stability)) if len(valid_stability) > 0 else None,
            'basin_stability_current': current_stability,
            'separatrix_distance': sep_distance,
            'separatrix_distance_current': sep_distance_current,
            'current_stability_type': current_stability_type,
            'embedding_dim': emb_dim,
            'embedding_tau': emb_tau,
            'n_valid': int(np.sum(~np.isnan(saddle_score))),
        }

    except Exception:
        return _empty_result()


def compute_multivariate(
    signals: np.ndarray,
    signal_names: Optional[List[str]] = None,
    min_samples: int = 100,
    velocity_threshold: float = 0.1,
    n_neighbors: int = None,
) -> Dict[str, Any]:
    """
    Detect saddle points in multivariate trajectory.

    Args:
        signals: Multi-variate time series (n_points, n_signals)
        signal_names: Names for each signal
        min_samples: Minimum samples required
        velocity_threshold: Threshold for equilibrium detection
        n_neighbors: Neighbors for Jacobian estimation

    Returns:
        dict with saddle metrics for the joint state space
    """
    signals = np.asarray(signals)

    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)

    n_points, n_signals = signals.shape

    if signal_names is None:
        signal_names = [f'signal_{i}' for i in range(n_signals)]

    if n_points < min_samples:
        return _empty_multivariate_result(n_signals, signal_names)

    # Remove NaN rows
    valid_mask = np.all(~np.isnan(signals), axis=1)
    signals_clean = signals[valid_mask]

    if len(signals_clean) < min_samples:
        return _empty_multivariate_result(n_signals, signal_names)

    if n_neighbors is None:
        n_neighbors = 2 * n_signals + 1

    try:
        # Detect saddle points in joint space
        saddle_score, velocity, saddle_info = detect_saddle_points(
            signals_clean,
            velocity_threshold=velocity_threshold,
            n_neighbors=n_neighbors,
        )

        saddle_mask = saddle_score > 0.5
        saddle_indices = np.where(saddle_mask)[0]

        # Basin stability
        basin_stability = compute_basin_stability(
            signals_clean,
            saddle_score,
            window=min(50, len(signals_clean) // 4),
        )

        # Per-variable stability type at current point
        current_idx = len(saddle_info) - 1
        while current_idx >= 0 and saddle_info[current_idx].get('stability_type') == 'unknown':
            current_idx -= 1

        if current_idx >= 0:
            current_info = saddle_info[current_idx]
            eigenvalues = current_info.get('eigenvalues', np.array([]))
            eigenvalues_real = current_info.get('eigenvalues_real', np.array([]))
        else:
            eigenvalues = np.array([])
            eigenvalues_real = np.array([])

        # Build variable stability dict
        variable_stability = {}
        for i, name in enumerate(signal_names):
            if i < len(eigenvalues_real):
                variable_stability[name] = {
                    'eigenvalue_real': float(eigenvalues_real[i]) if not np.isnan(eigenvalues_real[i]) else None,
                    'is_stable_direction': bool(eigenvalues_real[i] < 0) if not np.isnan(eigenvalues_real[i]) else None,
                }
            else:
                variable_stability[name] = {'eigenvalue_real': None, 'is_stable_direction': None}

        valid_scores = saddle_score[~np.isnan(saddle_score)]
        valid_stability = basin_stability[~np.isnan(basin_stability)]

        return {
            'saddle_score': saddle_score,
            'saddle_score_mean': float(np.mean(valid_scores)) if len(valid_scores) > 0 else None,
            'saddle_score_max': float(np.max(valid_scores)) if len(valid_scores) > 0 else None,
            'saddle_score_current': float(saddle_score[-1]) if not np.isnan(saddle_score[-1]) else None,
            'velocity': velocity,
            'saddle_indices': saddle_indices,
            'n_saddle_points': len(saddle_indices),
            'basin_stability': basin_stability,
            'basin_stability_mean': float(np.mean(valid_stability)) if len(valid_stability) > 0 else None,
            'basin_stability_current': float(basin_stability[-1]) if len(basin_stability) > 0 and not np.isnan(basin_stability[-1]) else None,
            'current_stability_type': saddle_info[-1].get('stability_type', 'unknown') if saddle_info else 'unknown',
            'jacobian_eigenvalues': eigenvalues,
            'variable_stability': variable_stability,
            'signal_names': signal_names,
            'n_valid': int(np.sum(~np.isnan(saddle_score))),
        }

    except Exception:
        return _empty_multivariate_result(n_signals, signal_names)


def compute_rolling(
    y: np.ndarray,
    window: int = 200,
    stride: int = 20,
    min_samples: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling saddle metrics.

    Args:
        y: Signal values
        window: Rolling window size
        stride: Step size
        min_samples: Min samples per window

    Returns:
        dict with rolling saddle statistics
    """
    y = np.asarray(y).flatten()
    n = len(y)

    if n < window or window < min_samples:
        return {
            'rolling_saddle_score': np.full(n, np.nan),
            'rolling_basin_stability': np.full(n, np.nan),
        }

    saddle_scores = np.full(n, np.nan)
    basin_stabilities = np.full(n, np.nan)

    for i in range(0, n - window + 1, stride):
        chunk = y[i:i + window]
        result = compute(chunk, min_samples=min_samples)

        idx = i + window - 1
        if result['saddle_score_current'] is not None:
            saddle_scores[idx] = result['saddle_score_current']
        if result['basin_stability_current'] is not None:
            basin_stabilities[idx] = result['basin_stability_current']

    return {
        'rolling_saddle_score': saddle_scores,
        'rolling_basin_stability': basin_stabilities,
    }


def _empty_result() -> Dict[str, Any]:
    """Return empty result for insufficient data."""
    return {
        'saddle_score': None,
        'saddle_score_mean': None,
        'saddle_score_max': None,
        'saddle_score_current': None,
        'velocity': None,
        'saddle_indices': np.array([]),
        'n_saddle_points': 0,
        'saddle_fraction': 0.0,
        'basin_stability': None,
        'basin_stability_mean': None,
        'basin_stability_current': None,
        'separatrix_distance': None,
        'separatrix_distance_current': None,
        'current_stability_type': 'unknown',
        'embedding_dim': None,
        'embedding_tau': None,
        'n_valid': 0,
    }


def _empty_multivariate_result(n_signals: int, signal_names: List[str]) -> Dict[str, Any]:
    """Return empty multivariate result."""
    return {
        'saddle_score': None,
        'saddle_score_mean': None,
        'saddle_score_max': None,
        'saddle_score_current': None,
        'velocity': None,
        'saddle_indices': np.array([]),
        'n_saddle_points': 0,
        'basin_stability': None,
        'basin_stability_mean': None,
        'basin_stability_current': None,
        'current_stability_type': 'unknown',
        'jacobian_eigenvalues': np.array([]),
        'variable_stability': {name: {'eigenvalue_real': None, 'is_stable_direction': None}
                              for name in signal_names},
        'signal_names': signal_names,
        'n_valid': 0,
    }
