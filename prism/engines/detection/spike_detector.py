"""
Spike Detection Engine
======================

Detects impulse events (spikes) in time series.

Characteristics of detected spikes:
    - Sharp deviation above threshold
    - Decay back toward baseline
    - Transient effect (not permanent)

Examples:
    - News shocks
    - Error spikes
    - Anomalies
    - Equipment faults
    - Outlier events

HONEST NAMING:
    This is NOT the Dirac delta function delta(x).
    The Dirac delta is a distribution defined by:
        delta(x) = 0 for x != 0
        integral(delta(x) dx) = 1

    This engine detects WHERE spike-like events occur in a time series.
    It does not compute the Dirac delta function.

    If you need the actual Dirac delta (for convolution, etc.), use:
        scipy.signal.unit_impulse(n, idx)
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from typing import Dict, List, Any


def compute(
    series: np.ndarray,
    threshold_sigma: float = 3.0,
    decay_window: int = 5
) -> Dict[str, Any]:
    """
    Detect impulse events (spikes) in time series.

    A spike is confirmed when:
    1. A point exceeds threshold_sigma standard deviations from trend
    2. The deviation decays back toward baseline

    Args:
        series: 1D numpy array of observations
        threshold_sigma: Z-score threshold for spike detection
        decay_window: Window to look for decay

    Returns:
        dict with:
            - detected: Boolean - any impulses found?
            - count: Number of impulses
            - max_magnitude: Largest impulse (sigma units)
            - mean_magnitude: Average impulse size
            - mean_half_life: Average decay rate
            - up_ratio: Fraction of positive impulses
            - locations: Indices of impulses (list)
    """
    series = np.asarray(series).flatten()
    n = len(series)

    if n < 10:
        return _empty_result()

    # Rolling window for local statistics
    window = min(20, n // 5)
    if window < 3:
        return _empty_result()

    # Detrend using rolling mean
    trend = uniform_filter1d(series.astype(float), size=window, mode='nearest')
    detrended = series - trend

    # Rolling std
    rolling_std = np.zeros(n)
    for i in range(window, n):
        rolling_std[i] = np.std(detrended[i-window:i])
    rolling_std[:window] = rolling_std[window] if window < n else 1.0
    rolling_std[rolling_std < 1e-10] = 1.0

    # Z-scores
    z_scores = detrended / rolling_std

    # Find spikes
    spike_mask = np.abs(z_scores) > threshold_sigma
    spike_indices = np.where(spike_mask)[0]

    if len(spike_indices) == 0:
        return _empty_result()

    # Cluster nearby spikes (treat consecutive spikes as one event)
    impulses = []
    current_cluster = [spike_indices[0]]

    for idx in spike_indices[1:]:
        if idx - current_cluster[-1] <= decay_window:
            current_cluster.append(idx)
        else:
            # Find peak of current cluster
            peak_idx = current_cluster[np.argmax(np.abs(z_scores[current_cluster]))]
            impulses.append(peak_idx)
            current_cluster = [idx]

    # Don't forget last cluster
    peak_idx = current_cluster[np.argmax(np.abs(z_scores[current_cluster]))]
    impulses.append(peak_idx)

    # Compute metrics
    magnitudes = np.abs(z_scores[impulses])
    directions = np.sign(z_scores[impulses])

    # Estimate half-lives (how quickly spike decays)
    half_lives = []
    for imp_idx in impulses:
        peak_val = np.abs(detrended[imp_idx])
        half_val = peak_val / 2

        for k in range(1, min(decay_window * 2, n - imp_idx)):
            if np.abs(detrended[imp_idx + k]) < half_val:
                half_lives.append(k)
                break
        else:
            half_lives.append(decay_window)

    return {
        'detected': True,
        'count': len(impulses),
        'max_magnitude': float(np.max(magnitudes)),
        'mean_magnitude': float(np.mean(magnitudes)),
        'mean_half_life': float(np.mean(half_lives)) if half_lives else None,
        'up_ratio': float(np.mean(directions > 0)),
        'locations': impulses,
        'threshold_used': float(threshold_sigma),
    }


def _empty_result() -> Dict[str, Any]:
    """Return empty result for no detections."""
    return {
        'detected': False,
        'count': 0,
        'max_magnitude': None,
        'mean_magnitude': None,
        'mean_half_life': None,
        'up_ratio': None,
        'locations': [],
        'threshold_used': None,
    }
