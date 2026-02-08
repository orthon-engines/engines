"""
Trajectory Sensitivity Engine.

Computes which variables are most sensitive at current state.
Sensitivity varies with position on the attractor.

ENGINES computes rankings. ORTHON interprets:
    - Variable ranks show real-time importance
    - Rank transitions indicate regime changes
    - High sensitivity entropy = distributed importance
"""

import numpy as np
from typing import Dict, Any, Optional, List

from engines.primitives.embedding import (
    time_delay_embedding,
    optimal_delay,
    optimal_dimension,
)
from engines.primitives.dynamical.sensitivity import (
    compute_variable_sensitivity,
    compute_sensitivity_evolution,
    detect_sensitivity_transitions,
    compute_influence_matrix,
)


def compute(
    signals: np.ndarray,
    signal_names: Optional[List[str]] = None,
    time_horizon: int = 10,
    min_samples: int = 100,
    n_neighbors: int = 10,
) -> Dict[str, Any]:
    """
    Compute trajectory-dependent variable sensitivity.

    Args:
        signals: Multi-variate time series (n_points, n_signals)
        signal_names: Names for each signal
        time_horizon: Steps for sensitivity calculation
        min_samples: Minimum samples required
        n_neighbors: Neighbors for local estimation

    Returns:
        dict with sensitivity scores, ranks, dominant variable
    """
    signals = np.asarray(signals)

    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)

    n_points, n_signals = signals.shape

    if signal_names is None:
        signal_names = [f'signal_{i}' for i in range(n_signals)]

    if n_points < min_samples:
        return _empty_result(n_signals, signal_names)

    # Remove NaN rows
    valid_mask = np.all(~np.isnan(signals), axis=1)
    signals_clean = signals[valid_mask]

    if len(signals_clean) < min_samples:
        return _empty_result(n_signals, signal_names)

    try:
        # Compute sensitivity
        sensitivity, rank = compute_variable_sensitivity(
            signals_clean,
            time_horizon=time_horizon,
            n_neighbors=n_neighbors,
        )

        # Compute evolution metrics
        evolution = compute_sensitivity_evolution(
            sensitivity,
            window=min(50, n_points // 4),
        )

        # Detect transitions
        transition_points, transitions = detect_sensitivity_transitions(
            sensitivity, rank,
            window=20,
        )

        # Compute influence matrix
        influence = compute_influence_matrix(
            signals_clean,
            time_horizon=time_horizon,
            n_neighbors=n_neighbors,
        )

        # Current state (last valid point)
        current_idx = len(sensitivity) - 1
        while current_idx >= 0 and np.all(np.isnan(sensitivity[current_idx])):
            current_idx -= 1

        if current_idx >= 0:
            current_sensitivity = sensitivity[current_idx]
            current_rank = rank[current_idx]
            dominant_idx = np.nanargmax(current_sensitivity)
            dominant_signal = signal_names[dominant_idx]
        else:
            current_sensitivity = np.full(n_signals, np.nan)
            current_rank = np.full(n_signals, np.nan)
            dominant_idx = None
            dominant_signal = None

        # Mean sensitivity per signal
        mean_sensitivity = np.nanmean(sensitivity, axis=0)

        # Build signal sensitivity dict
        signal_sensitivity = {}
        for i, name in enumerate(signal_names):
            signal_sensitivity[name] = {
                'current_sensitivity': float(current_sensitivity[i]) if not np.isnan(current_sensitivity[i]) else None,
                'current_rank': int(current_rank[i]) if not np.isnan(current_rank[i]) else None,
                'mean_sensitivity': float(mean_sensitivity[i]) if not np.isnan(mean_sensitivity[i]) else None,
            }

        return {
            'sensitivity': sensitivity,
            'rank': rank,
            'current_sensitivity': current_sensitivity,
            'current_rank': current_rank,
            'dominant_variable': dominant_signal,
            'dominant_variable_idx': dominant_idx,
            'signal_sensitivity': signal_sensitivity,
            'mean_sensitivity': mean_sensitivity,
            'sensitivity_entropy': evolution['sensitivity_entropy'],
            'sensitivity_entropy_current': float(evolution['sensitivity_entropy'][-1])
                if len(evolution['sensitivity_entropy']) > 0 and not np.isnan(evolution['sensitivity_entropy'][-1]) else None,
            'n_transitions': len(transitions),
            'transition_points': transition_points,
            'transitions': transitions,
            'influence_matrix': influence,
            'signal_names': signal_names,
            'n_valid': int(np.sum(~np.isnan(sensitivity[:, 0]))),
        }

    except Exception:
        return _empty_result(n_signals, signal_names)


def compute_from_signal_vector(
    signal_vector: "pl.DataFrame",
    feature_columns: Optional[List[str]] = None,
    time_horizon: int = 10,
) -> Dict[str, Any]:
    """
    Compute sensitivity from ENGINES signal_vector.parquet.

    Args:
        signal_vector: DataFrame with signal features
        feature_columns: Which columns to use
        time_horizon: Steps for sensitivity

    Returns:
        Sensitivity analysis results
    """
    import polars as pl

    if feature_columns is None:
        # Use all numeric feature columns
        feature_columns = [
            c for c in signal_vector.columns
            if c not in ['unit_id', 'I', 'signal_id', 'cohort']
            and signal_vector[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]

    # Pivot to get signals as columns
    # Each row is a time step (I), columns are signals
    signals_matrix = signal_vector.pivot(
        values=feature_columns[0] if feature_columns else 'value',
        index='I',
        on='signal_id',
    ).sort('I')

    signal_cols = [c for c in signals_matrix.columns if c != 'I']
    matrix = signals_matrix.select(signal_cols).to_numpy()

    return compute(
        matrix,
        signal_names=signal_cols,
        time_horizon=time_horizon,
    )


def compute_rolling(
    signals: np.ndarray,
    signal_names: Optional[List[str]] = None,
    window: int = 100,
    stride: int = 10,
    time_horizon: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling sensitivity statistics.

    Args:
        signals: Multi-variate time series (n_points, n_signals)
        signal_names: Names for each signal
        window: Rolling window size
        stride: Step size
        time_horizon: FTLE time horizon

    Returns:
        dict with rolling sensitivity per signal
    """
    signals = np.asarray(signals)
    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)

    n_points, n_signals = signals.shape

    if signal_names is None:
        signal_names = [f'signal_{i}' for i in range(n_signals)]

    rolling_sensitivity = {name: np.full(n_points, np.nan) for name in signal_names}
    rolling_dominant = np.full(n_points, np.nan)

    if n_points < window:
        return {
            'rolling_sensitivity': rolling_sensitivity,
            'rolling_dominant': rolling_dominant,
        }

    for i in range(0, n_points - window + 1, stride):
        chunk = signals[i:i + window]
        result = compute(chunk, signal_names=signal_names, time_horizon=time_horizon)

        idx = i + window - 1
        if result['dominant_variable_idx'] is not None:
            rolling_dominant[idx] = result['dominant_variable_idx']

        for j, name in enumerate(signal_names):
            if result['mean_sensitivity'] is not None and not np.isnan(result['mean_sensitivity'][j]):
                rolling_sensitivity[name][idx] = result['mean_sensitivity'][j]

    return {
        'rolling_sensitivity': rolling_sensitivity,
        'rolling_dominant': rolling_dominant,
    }


def _empty_result(n_signals: int, signal_names: List[str]) -> Dict[str, Any]:
    """Return empty result for insufficient data."""
    return {
        'sensitivity': None,
        'rank': None,
        'current_sensitivity': np.full(n_signals, np.nan),
        'current_rank': np.full(n_signals, np.nan),
        'dominant_variable': None,
        'dominant_variable_idx': None,
        'signal_sensitivity': {name: {'current_sensitivity': None, 'current_rank': None, 'mean_sensitivity': None}
                              for name in signal_names},
        'mean_sensitivity': np.full(n_signals, np.nan),
        'sensitivity_entropy': None,
        'sensitivity_entropy_current': None,
        'n_transitions': 0,
        'transition_points': np.array([]),
        'transitions': [],
        'influence_matrix': None,
        'signal_names': signal_names,
        'n_valid': 0,
    }
