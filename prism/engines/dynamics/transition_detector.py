"""
Transition Detector Engine

Detects windows where regime changes occur.

Definition:
    A transition occurs when regime[i] != regime[i-1]

Output:
    is_transition: bool per window
    transition_from: regime before transition (null if not transition)
    transition_to: regime after transition (null if not transition)
    transitions: list of (window, from_regime, to_regime) tuples
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import polars as pl


def compute(
    regime_labels: np.ndarray = None,
    df: pl.DataFrame = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Detect transition points from regime sequence.

    Args:
        regime_labels: Array of regime labels per window
        df: Alternative: DataFrame with 'regime' column

    Returns:
        Dict with transition detection results
    """
    # Get regime labels
    if regime_labels is not None:
        regimes = np.asarray(regime_labels)
    elif df is not None and 'regime' in df.columns:
        regimes = df['regime'].to_numpy()
    else:
        return _null_result("No regime labels provided")

    if len(regimes) < 2:
        return _null_result("Need at least 2 windows for transition detection")

    n_windows = len(regimes)

    # Detect transitions
    is_transition = np.zeros(n_windows, dtype=bool)
    transition_from = np.full(n_windows, np.nan)
    transition_to = np.full(n_windows, np.nan)
    transitions = []

    for i in range(1, n_windows):
        if regimes[i] != regimes[i-1]:
            is_transition[i] = True
            transition_from[i] = regimes[i-1]
            transition_to[i] = regimes[i]
            transitions.append({
                'window': i,
                'from_regime': int(regimes[i-1]),
                'to_regime': int(regimes[i]),
            })

    # Compute transition statistics
    n_transitions = len(transitions)

    # Count transition types
    transition_counts = {}
    for t in transitions:
        key = f"{t['from_regime']}_to_{t['to_regime']}"
        transition_counts[key] = transition_counts.get(key, 0) + 1

    # Identify degradation vs recovery transitions
    # Degradation: from_regime < to_regime (moving away from baseline)
    # Recovery: from_regime > to_regime (moving toward baseline)
    n_degradation = sum(1 for t in transitions if t['from_regime'] < t['to_regime'])
    n_recovery = sum(1 for t in transitions if t['from_regime'] > t['to_regime'])

    # First and last transition
    first_transition = transitions[0] if transitions else None
    last_transition = transitions[-1] if transitions else None

    return {
        'is_transition': is_transition.tolist(),
        'transition_from': [int(x) if not np.isnan(x) else None for x in transition_from],
        'transition_to': [int(x) if not np.isnan(x) else None for x in transition_to],
        'transitions': transitions,
        'n_transitions': n_transitions,
        'n_degradation_transitions': n_degradation,
        'n_recovery_transitions': n_recovery,
        'transition_counts': transition_counts,
        'first_transition_window': first_transition['window'] if first_transition else None,
        'last_transition_window': last_transition['window'] if last_transition else None,
        'mean_time_between_transitions': float(n_windows / (n_transitions + 1)),
    }


def compute_sparse(
    regime_labels: np.ndarray = None,
    df: pl.DataFrame = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Return only the transition rows (sparse format).

    Useful for storing in a separate transitions table.
    """
    result = compute(regime_labels=regime_labels, df=df, **kwargs)

    if 'transitions' in result and result['transitions']:
        return result['transitions']
    return []


def _null_result(reason: str) -> Dict[str, Any]:
    """Return null result with reason."""
    return {
        'is_transition': None,
        'transition_from': None,
        'transition_to': None,
        'transitions': [],
        'n_transitions': None,
        'n_degradation_transitions': None,
        'n_recovery_transitions': None,
        'transition_counts': {},
        'first_transition_window': None,
        'last_transition_window': None,
        'mean_time_between_transitions': None,
        'transition_detector_error': reason,
    }
