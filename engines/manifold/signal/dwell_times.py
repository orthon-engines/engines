"""
Dwell Times Engine
==================

Computes time (samples) spent at each discrete level before transitioning.
Designed for DISCRETE, BINARY, STEP, and EVENT signal types.

For continuous signals, values are binned first.

Outputs:
    dwell_mean      - Mean dwell time across all levels
    dwell_std       - Std of dwell times (regularity of state holding)
    dwell_max       - Longest dwell (stuck state detection)
    dwell_min       - Shortest dwell (chattering detection)
    dwell_cv        - Coefficient of variation (dwell_std / dwell_mean)
    n_dwells        - Total number of dwell periods

Physics:
    - High dwell_cv → irregular holding times → degrading control
    - Increasing dwell_max → system getting stuck
    - Decreasing dwell_min → chattering / instability
    - dwell_mean trending down → faster state cycling
"""

import numpy as np
from typing import Dict, Any


MIN_SAMPLES = 4


def compute(y: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """
    Compute dwell time statistics.

    Parameters
    ----------
    y : np.ndarray
        Input signal (discrete or continuous).
    n_bins : int
        Number of bins for continuous signals. Ignored if signal
        has fewer than n_bins unique values (already discrete).

    Returns
    -------
    dict
        Dwell time statistics.
    """
    y = np.asarray(y).flatten()

    if len(y) < MIN_SAMPLES:
        raise ValueError(f"Need {MIN_SAMPLES} samples, got {len(y)}")

    # Discretize if continuous
    unique_vals = np.unique(y[~np.isnan(y)])
    if len(unique_vals) > n_bins:
        # Continuous signal — bin it
        bins = np.linspace(np.nanmin(y), np.nanmax(y), n_bins + 1)
        levels = np.digitize(y, bins, right=True)
    else:
        # Already discrete — use raw values
        # Map to integer labels for consistency
        val_to_label = {v: i for i, v in enumerate(sorted(unique_vals))}
        levels = np.array([val_to_label.get(v, -1) for v in y])

    # Compute dwell times: consecutive runs of same level
    dwells = []
    current_level = levels[0]
    current_dwell = 1

    for i in range(1, len(levels)):
        if levels[i] == current_level:
            current_dwell += 1
        else:
            dwells.append(current_dwell)
            current_level = levels[i]
            current_dwell = 1

    # Don't forget the last run
    dwells.append(current_dwell)

    dwells = np.array(dwells, dtype=float)

    if len(dwells) == 0:
        return {
            'dwell_mean': np.nan,
            'dwell_std': np.nan,
            'dwell_max': np.nan,
            'dwell_min': np.nan,
            'dwell_cv': np.nan,
            'n_dwells': 0,
        }

    mean_dwell = float(np.mean(dwells))
    std_dwell = float(np.std(dwells))

    return {
        'dwell_mean': mean_dwell,
        'dwell_std': std_dwell,
        'dwell_max': float(np.max(dwells)),
        'dwell_min': float(np.min(dwells)),
        'dwell_cv': std_dwell / mean_dwell if mean_dwell > 0 else np.nan,
        'n_dwells': int(len(dwells)),
    }
