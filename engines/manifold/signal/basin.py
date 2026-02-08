"""
Basin Engine.

Estimates basin stability and transition probability.
"""

import numpy as np


def compute(y: np.ndarray, n_bins: int = 20) -> dict:
    """
    Estimate basin stability metrics.

    Args:
        y: Signal values
        n_bins: Number of bins for state discretization

    Returns:
        dict with basin_stability, transition_prob, n_attractors
    """
    result = {
        'basin_stability': np.nan,
        'transition_prob': np.nan,
        'n_attractors': np.nan
    }

    if len(y) < 100:
        return result

    try:
        # Discretize into bins
        bins = np.linspace(y.min(), y.max(), n_bins + 1)
        states = np.digitize(y, bins) - 1
        states = np.clip(states, 0, n_bins - 1)

        # Count transitions
        transitions = np.sum(states[1:] != states[:-1])
        transition_prob = transitions / (len(states) - 1)

        # Find dominant states (attractors)
        state_counts = np.bincount(states, minlength=n_bins)
        threshold = len(y) / n_bins * 0.5  # States with > half expected count
        n_attractors = np.sum(state_counts > threshold)

        # Basin stability: fraction of time in dominant state
        basin_stability = state_counts.max() / len(states)

        result = {
            'basin_stability': float(basin_stability),
            'transition_prob': float(transition_prob),
            'n_attractors': int(n_attractors)
        }

    except Exception:
        pass

    return result
