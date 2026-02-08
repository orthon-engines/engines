"""
Level Histogram Engine
======================

Computes distribution shape metrics across discrete levels.
Returns summary statistics of the histogram, NOT the raw histogram
(which would be variable-length and break parquet schema).

Designed for DISCRETE, BINARY, STEP, EVENT signal types.

Outputs:
    hist_uniformity   - How uniform is the distribution (1.0 = perfectly uniform)
    hist_concentration - Gini coefficient of level distribution (0=equal, 1=concentrated)
    hist_peak_ratio   - Ratio of most common to second most common level
    hist_tail_weight  - Fraction of samples in bottom 20% of levels
    hist_bimodality   - Bimodality coefficient (>0.555 suggests bimodal)

Physics:
    - hist_uniformity dropping → system preferring certain states
    - hist_concentration rising → system collapsing to fewer states
    - hist_peak_ratio spiking → one state dominating
    - hist_bimodality appearing → system oscillating between two attractors
"""

import numpy as np
from typing import Dict, Any


MIN_SAMPLES = 4


def compute(y: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """
    Compute histogram shape statistics.

    Parameters
    ----------
    y : np.ndarray
        Input signal.
    n_bins : int
        Number of bins for continuous signals.

    Returns
    -------
    dict
        Histogram shape statistics.
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
    else:
        val_to_label = {v: i for i, v in enumerate(sorted(unique_vals))}
        levels = np.array([val_to_label[v] for v in y_clean])

    _, counts = np.unique(levels, return_counts=True)
    n = len(y_clean)
    k = len(counts)

    probs = counts / n

    # Uniformity: 1 - normalized KL divergence from uniform
    uniform = np.ones(k) / k
    kl = np.sum(probs * np.log2((probs + 1e-12) / (uniform + 1e-12)))
    max_kl = np.log2(k) if k > 1 else 1.0
    uniformity = float(1.0 - min(kl / max(max_kl, 1e-12), 1.0))

    # Gini coefficient (concentration)
    sorted_probs = np.sort(probs)
    cumulative = np.cumsum(sorted_probs)
    gini = float(1.0 - 2.0 * np.sum(cumulative) / (k * np.sum(probs)) + 1.0 / k) if k > 1 else 0.0

    # Peak ratio: most common / second most common
    sorted_counts = np.sort(counts)[::-1]
    if len(sorted_counts) >= 2 and sorted_counts[1] > 0:
        peak_ratio = float(sorted_counts[0] / sorted_counts[1])
    else:
        peak_ratio = float('inf') if k == 1 else np.nan

    # Tail weight: fraction of samples in least common 20% of levels
    n_tail = max(1, int(k * 0.2))
    tail_levels = np.sort(counts)[:n_tail]
    tail_weight = float(tail_levels.sum() / n)

    # Bimodality coefficient: BC = (skew^2 + 1) / (kurtosis + 3 * (n-1)^2 / ((n-2)(n-3)))
    # Applied to the histogram counts themselves
    if k >= 4:
        m = np.mean(counts.astype(float))
        s = np.std(counts.astype(float), ddof=1)
        if s > 0:
            skew = float(np.mean(((counts - m) / s) ** 3))
            kurt = float(np.mean(((counts - m) / s) ** 4))
            excess_kurt = kurt - 3.0
            bc = (skew ** 2 + 1) / (excess_kurt + 3.0)
        else:
            bc = 0.0
    else:
        bc = np.nan

    return {
        'hist_uniformity': uniformity,
        'hist_concentration': max(0.0, min(1.0, gini)),
        'hist_peak_ratio': peak_ratio,
        'hist_tail_weight': tail_weight,
        'hist_bimodality': float(bc) if not np.isnan(bc) else np.nan,
    }
