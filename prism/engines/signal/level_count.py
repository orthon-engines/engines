"""
Level Count Engine
==================

Counts unique discrete levels and their distribution properties.
Designed for DISCRETE, BINARY, STEP signal types.

For continuous signals, values are binned first.

Outputs:
    n_levels          - Number of unique levels (or bins occupied)
    level_occupancy   - Fraction of bins occupied (n_levels / n_bins)
    dominant_level    - Most frequent level (as fraction of total)
    level_spread      - Range between min and max occupied levels
    level_entropy     - Shannon entropy of level distribution (bits)

Physics:
    - n_levels decreasing → system collapsing to fewer states
    - dominant_level increasing → system spending more time in one state
    - level_entropy decreasing → loss of state diversity
    - level_occupancy dropping → available state space shrinking
"""

import numpy as np
from typing import Dict, Any


MIN_SAMPLES = 4


def compute(y: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """
    Compute level count statistics.

    Parameters
    ----------
    y : np.ndarray
        Input signal.
    n_bins : int
        Number of bins for continuous signals.

    Returns
    -------
    dict
        Level count statistics.
    """
    y = np.asarray(y).flatten()
    y_clean = y[~np.isnan(y)]

    if len(y_clean) < MIN_SAMPLES:
        raise ValueError(f"Need {MIN_SAMPLES} samples, got {len(y_clean)}")

    # Discretize if continuous
    unique_vals = np.unique(y_clean)
    if len(unique_vals) > n_bins:
        bins = np.linspace(np.nanmin(y_clean), np.nanmax(y_clean), n_bins + 1)
        levels = np.digitize(y_clean, bins, right=True)
        total_bins = n_bins
    else:
        val_to_label = {v: i for i, v in enumerate(sorted(unique_vals))}
        levels = np.array([val_to_label[v] for v in y_clean])
        total_bins = len(unique_vals)

    # Count unique occupied levels
    unique_levels, counts = np.unique(levels, return_counts=True)
    n_levels = len(unique_levels)

    # Probabilities for each level
    probs = counts / counts.sum()

    # Shannon entropy in bits
    entropy = -np.sum(probs * np.log2(probs + 1e-12))

    # Dominant level fraction
    dominant = float(np.max(probs))

    # Level spread (normalized)
    if n_levels > 1:
        spread = float((unique_levels.max() - unique_levels.min()) / max(total_bins - 1, 1))
    else:
        spread = 0.0

    return {
        'n_levels': int(n_levels),
        'level_occupancy': float(n_levels / max(total_bins, 1)),
        'dominant_level': dominant,
        'level_spread': spread,
        'level_entropy': float(entropy),
    }
