"""
Rate of Change Engine.

Computes mean and max rate of change.
Used for temperature ramp detection, pressure transients.
"""

import numpy as np
from typing import Dict


def compute(y: np.ndarray, I: np.ndarray = None) -> Dict[str, float]:
    """
    Compute rate of change metrics.

    Args:
        y: Signal values
        I: Index/time values (optional, assumes uniform dt=1 if None)

    Returns:
        dict with mean_rate, max_rate, min_rate, rate_std,
              abs_max_rate (max of absolute rates)
    """
    result = {
        'mean_rate': np.nan,
        'max_rate': np.nan,
        'min_rate': np.nan,
        'rate_std': np.nan,
        'abs_max_rate': np.nan
    }

    # Handle NaN values
    y = np.asarray(y).flatten()

    if I is not None:
        I = np.asarray(I).flatten()
        # Remove pairs with NaN in either y or I
        if len(I) == len(y):
            valid_mask = ~(np.isnan(y) | np.isnan(I))
            y = y[valid_mask]
            I = I[valid_mask]
    else:
        # Remove NaN from y only
        valid_mask = ~np.isnan(y)
        y = y[valid_mask]

    n = len(y)
    if n < 3:
        return result

    try:
        if I is None:
            # Uniform time steps, dt=1
            dy = np.diff(y)
        else:
            # Non-uniform time steps
            dt = np.diff(I)
            dy_raw = np.diff(y)

            # Handle zero or near-zero time intervals
            dt_safe = np.where(np.abs(dt) < 1e-10, 1e-10, dt)
            dy = dy_raw / dt_safe

        # Filter out any remaining NaN/Inf
        dy = dy[np.isfinite(dy)]

        if len(dy) == 0:
            return result

        result = {
            'mean_rate': float(np.mean(dy)),
            'max_rate': float(np.max(dy)),
            'min_rate': float(np.min(dy)),
            'rate_std': float(np.std(dy)),
            'abs_max_rate': float(np.max(np.abs(dy)))
        }

    except Exception:
        pass

    return result
