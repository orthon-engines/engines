"""
Pulsation Index Engine.

Measures flow variability: (max - min) / mean
High pulsation indicates pump issues, cavitation, valve problems.
"""

import numpy as np


def compute(y: np.ndarray) -> dict:
    """
    Compute pulsation index.

    Args:
        y: Signal values (typically flow rate)

    Returns:
        dict with pulsation_index, flow_mean, flow_range
    """
    if len(y) < 10:
        return {
            'pulsation_index': np.nan,
            'flow_mean': np.nan,
            'flow_range': np.nan
        }

    mean_val = np.mean(y)
    min_val = np.min(y)
    max_val = np.max(y)
    range_val = max_val - min_val

    # Pulsation index: range / mean (dimensionless)
    pi = range_val / (abs(mean_val) + 1e-10)

    return {
        'pulsation_index': float(pi),
        'flow_mean': float(mean_val),
        'flow_range': float(range_val)
    }
