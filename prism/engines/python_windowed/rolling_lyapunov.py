"""
Rolling Lyapunov Engine.

Computes Lyapunov exponent over sliding windows.
Uses the same algorithm as the signal-level lyapunov engine.

Key insight: Lyapunov trending positive = system losing stability
"""

import numpy as np
from typing import Dict
from ..python import lyapunov


def compute(
    y: np.ndarray,
    window: int = 500,
    stride: int = 50,
    min_samples: int = 200
) -> Dict[str, np.ndarray]:
    """
    Compute rolling Lyapunov exponent.

    Args:
        y: Signal values
        window: Window size (minimum 200 recommended)
        stride: Step size between windows
        min_samples: Minimum samples for valid computation

    Returns:
        dict with:
            - 'rolling_lyapunov': Lyapunov values at each window end
            - 'rolling_stability_class': Stability classification
            - 'window_centers': Center index of each window
    """
    y = np.asarray(y).flatten()
    n = len(y)

    if n < window or window < min_samples:
        return {
            'rolling_lyapunov': np.full(n, np.nan),
            'rolling_stability_class': np.array(['unknown'] * n),
            'window_centers': np.array([]),
        }

    # Output arrays (sparse - only at window ends)
    lyap_values = np.full(n, np.nan)
    stability_classes = np.array(['unknown'] * n, dtype=object)
    window_centers = []

    for i in range(0, n - window + 1, stride):
        chunk = y[i:i + window]
        center = i + window // 2
        end = i + window - 1

        # Use the signal-level engine (same algorithm)
        result = lyapunov.compute(chunk, min_samples=min_samples)

        lyap_values[end] = result['lyapunov']
        stability_classes[end] = result['stability_class']
        window_centers.append(center)

    return {
        'rolling_lyapunov': lyap_values,
        'rolling_stability_class': stability_classes,
        'window_centers': np.array(window_centers),
    }


def compute_trend(lyap_values: np.ndarray) -> Dict[str, float]:
    """
    Detect trend in Lyapunov values.

    Key early warning: Lyapunov trending positive = losing stability

    Args:
        lyap_values: Array of Lyapunov values (with NaN for missing)

    Returns:
        dict with:
            - 'slope': Trend slope
            - 'r_squared': Fit quality
            - 'is_destabilizing': True if significantly trending positive
    """
    valid = ~np.isnan(lyap_values)
    if np.sum(valid) < 4:
        return {
            'slope': np.nan,
            'r_squared': np.nan,
            'is_destabilizing': False
        }

    x = np.arange(len(lyap_values))[valid]
    y = lyap_values[valid]

    # Linear fit
    slope, intercept = np.polyfit(x, y, 1)

    # R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Destabilizing if positive slope with good fit
    is_destabilizing = (
        slope > 0.001 and
        r_squared > 0.3 and
        np.mean(y[-3:]) > np.mean(y[:3])
    )

    return {
        'slope': float(slope),
        'r_squared': float(r_squared),
        'is_destabilizing': bool(is_destabilizing)
    }
