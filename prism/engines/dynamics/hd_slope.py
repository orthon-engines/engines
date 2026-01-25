"""
HD Slope - Degradation Rate Engine
===================================

The most important metric for prognosis.

hd_slope = d(||v - v₀||) / dt

Measures how fast the system is drifting from its initial (healthy) state.

Interpretation:
    hd_slope ≈ 0      : System stable, staying near baseline
    hd_slope > 0      : System drifting away (degrading)
    hd_slope >> 0     : Rapid degradation, failure imminent
    hd_slope < 0      : System recovering (rare, usually measurement artifact)

The slope is computed via linear regression of distance-from-baseline over time.
R² indicates how linear the degradation is:
    R² ≈ 1  : Steady, predictable degradation
    R² << 1 : Erratic, non-linear degradation (harder to predict)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from scipy.stats import linregress


@dataclass
class HDSlopeResult:
    """Result of HD slope computation."""
    hd_slope: float           # Degradation rate (primary metric)
    hd_intercept: float       # Initial distance from baseline
    hd_r_squared: float       # How linear is the degradation
    hd_p_value: float         # Statistical significance
    hd_std_err: float         # Standard error of slope
    hd_final_distance: float  # Distance at end of series
    hd_max_distance: float    # Maximum distance observed
    hd_acceleration: float    # Is degradation speeding up?


def compute_hd_slope(values: np.ndarray) -> Dict[str, float]:
    """
    Compute HD slope (degradation rate) from a single signal.

    For a single signal, we compute the slope of cumulative deviation
    from the initial value. This captures monotonic drift.

    Args:
        values: 1D array of signal values (ordered by time/cycle/depth)

    Returns:
        Dict with hd_slope metrics
    """
    n = len(values)
    if n < 10:
        return {}

    # Baseline is the initial value (or mean of first few points for stability)
    baseline = np.mean(values[:min(5, n)])

    # Distance from baseline at each point
    distances = np.abs(values - baseline)

    # Time axis (normalized to [0, 1] for comparable slopes)
    t = np.arange(n) / (n - 1)

    # Linear regression
    try:
        slope, intercept, r_value, p_value, std_err = linregress(t, distances)
    except Exception:
        return {}

    # Acceleration: is the slope increasing over time?
    # Fit quadratic and check second derivative
    try:
        coeffs = np.polyfit(t, distances, 2)
        acceleration = 2 * coeffs[0]  # Second derivative of quadratic
    except Exception:
        acceleration = 0.0

    # Keys without 'hd_' prefix since vector.py adds 'hd_slope_' prefix
    return {
        'slope': float(slope),                    # Primary degradation rate
        'intercept': float(intercept),            # Initial distance from baseline
        'r_squared': float(r_value ** 2),         # How linear is degradation
        'p_value': float(p_value),                # Statistical significance
        'std_err': float(std_err),                # Standard error of slope
        'final_distance': float(distances[-1]),   # Distance at end
        'max_distance': float(np.max(distances)), # Max distance observed
        'acceleration': float(acceleration),      # Is degradation speeding up
    }


def compute_hd_slope_multivariate(
    feature_matrix: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute HD slope from multivariate feature vectors.

    This is the canonical form: track distance in feature space
    from the initial state over time.

    Args:
        feature_matrix: [n_timestamps, n_features] array
        timestamps: Optional time values (uses indices if None)

    Returns:
        Dict with hd_slope metrics
    """
    n_times, n_features = feature_matrix.shape

    if n_times < 10:
        return {}

    # Baseline: first feature vector (or mean of first few)
    baseline = np.mean(feature_matrix[:min(5, n_times)], axis=0)

    # Euclidean distance from baseline at each time
    distances = np.array([
        np.linalg.norm(feature_matrix[i] - baseline)
        for i in range(n_times)
    ])

    # Time axis
    if timestamps is None:
        t = np.arange(n_times) / (n_times - 1)
    else:
        t = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])

    # Linear regression
    try:
        slope, intercept, r_value, p_value, std_err = linregress(t, distances)
    except Exception:
        return {}

    # Acceleration
    try:
        coeffs = np.polyfit(t, distances, 2)
        acceleration = 2 * coeffs[0]
    except Exception:
        acceleration = 0.0

    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_value ** 2),
        'p_value': float(p_value),
        'std_err': float(std_err),
        'final_distance': float(distances[-1]),
        'max_distance': float(np.max(distances)),
        'acceleration': float(acceleration),
        'n_features': float(n_features),
    }


def compute_hd_slope_windowed(
    values: np.ndarray,
    window_size: int = 50,
    stride: int = 10,
) -> np.ndarray:
    """
    Compute HD slope in rolling windows.

    Returns array of local degradation rates, useful for detecting
    when degradation accelerates.

    Args:
        values: 1D array of values
        window_size: Size of rolling window
        stride: Step between windows

    Returns:
        Array of hd_slope values for each window
    """
    n = len(values)
    if n < window_size:
        return np.array([])

    slopes = []
    for start in range(0, n - window_size + 1, stride):
        window = values[start:start + window_size]
        result = compute_hd_slope(window)
        if 'slope' in result:
            slopes.append(result['slope'])

    return np.array(slopes)
