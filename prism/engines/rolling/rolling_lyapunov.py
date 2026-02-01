"""
Rolling Lyapunov Engine.

Computes Lyapunov exponent over sliding windows.
All parameters from manifest via params dict.

Key insight: Lyapunov trending positive = system losing stability
"""

import numpy as np
from typing import Dict, Any
from ..signal import lyapunov


def compute(y: np.ndarray, params: Dict[str, Any] = None) -> dict:
    """
    Compute rolling Lyapunov exponent.

    Args:
        y: Signal values
        params: Parameters from manifest:
            - window: Window size (minimum 200 recommended)
            - stride: Step size between windows
            - min_samples: Minimum samples for valid computation

    Returns:
        dict with:
            - 'rolling_lyapunov': Lyapunov values at each window end
            - 'rolling_stability_class': Stability classification
    """
    params = params or {}
    window = params.get('window', 500)
    # Expensive engine: stride MUST come from manifest, fallback to window//10
    stride = params.get('stride', max(1, window // 10))
    min_samples = params.get('min_samples', 200)

    y = np.asarray(y).flatten()
    n = len(y)

    if n < window or window < min_samples:
        return {
            'rolling_lyapunov': np.full(n, np.nan),
            'rolling_stability_class': np.array(['unknown'] * n, dtype=object),
        }

    lyap_values = np.full(n, np.nan)
    confidence_values = np.full(n, np.nan)

    for i in range(0, n - window + 1, stride):
        chunk = y[i:i + window]
        end = i + window - 1

        result = lyapunov.compute(chunk, min_samples=min_samples)

        lyap_values[end] = result['lyapunov']
        confidence_values[end] = result.get('confidence', np.nan)

    return {
        'rolling_lyapunov': lyap_values,
        'rolling_lyapunov_confidence': confidence_values,
    }


def compute_trend(lyap_values: np.ndarray) -> Dict[str, float]:
    """
    Detect trend in Lyapunov values.

    Key early warning: Lyapunov trending positive = losing stability
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

    slope, intercept = np.polyfit(x, y, 1)

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

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
