"""
Structural Discontinuity Analysis
=================================

Analyzes the overall pattern of discontinuities:
    - Interval between discontinuities
    - Regularity of occurrences
    - Acceleration (are they getting more frequent?)

This provides meta-information about the discontinuity
process itself.
"""

import numpy as np
from typing import Dict, List, Any

# Use honest names
from prism.engines.detection.spike_detector import compute as detect_spikes
from prism.engines.detection.step_detector import compute as detect_steps


def compute(series: np.ndarray) -> Dict[str, Any]:
    """
    Analyze structural discontinuity patterns.

    Args:
        series: 1D numpy array of observations

    Returns:
        dict with:
            - spikes: Spike detection results
            - steps: Step detection results
            - total_count: Combined discontinuity count
            - mean_interval: Average time between discontinuities
            - interval_cv: Coefficient of variation (regularity)
            - dominant_period: Characteristic period if regular
            - is_accelerating: Are discontinuities getting more frequent?
            - any_detected: Boolean - any discontinuities found?
    """
    # Detect both types
    spike_result = detect_spikes(series)
    step_result = detect_steps(series)

    # Combine locations
    all_locations = sorted(
        spike_result['locations'] + step_result['locations']
    )

    total_count = len(all_locations)
    any_detected = spike_result['detected'] or step_result['detected']

    if total_count < 2:
        return {
            'spikes': spike_result,
            'steps': step_result,
            # Backwards compat
            'dirac': spike_result,
            'heaviside': step_result,
            'total_count': total_count,
            'mean_interval': None,
            'interval_cv': None,
            'dominant_period': None,
            'is_accelerating': False,
            'any_detected': any_detected
        }

    # Analyze intervals
    intervals = np.diff(all_locations)
    mean_interval = float(np.mean(intervals))
    interval_cv = float(np.std(intervals) / mean_interval) if mean_interval > 0 else None

    # Check if accelerating (intervals getting shorter)
    if len(intervals) >= 3:
        first_half = np.mean(intervals[:len(intervals)//2])
        second_half = np.mean(intervals[len(intervals)//2:])
        is_accelerating = second_half < first_half * 0.8
    else:
        is_accelerating = False

    # Dominant period (if regular)
    if interval_cv is not None and interval_cv < 0.5:  # Relatively regular
        dominant_period = mean_interval
    else:
        dominant_period = None

    return {
        'spikes': spike_result,
        'steps': step_result,
        # Backwards compat
        'dirac': spike_result,
        'heaviside': step_result,
        'total_count': total_count,
        'mean_interval': mean_interval,
        'interval_cv': interval_cv,
        'dominant_period': dominant_period,
        'is_accelerating': is_accelerating,
        'any_detected': any_detected
    }
