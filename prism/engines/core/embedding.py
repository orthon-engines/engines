"""
Embedding Dimension Estimation
==============================

Estimates the proper embedding dimension for phase space
reconstruction using a simplified False Nearest Neighbors approach.

The embedding dimension determines how many delayed coordinates
are needed to properly unfold the attractor.

References:
    Kennel et al. (1992) "Determining embedding dimension for phase-space
                          reconstruction using a geometrical construction"
"""

import numpy as np
from typing import Dict


def compute(series: np.ndarray, max_dim: int = 10) -> Dict[str, int]:
    """
    Estimate embedding dimension.

    Args:
        series: 1D numpy array of observations
        max_dim: Maximum dimension to consider

    Returns:
        dict with:
            - embedding_dimension: Estimated optimal dimension
            - first_zero_crossing: ACF first zero crossing (used as delay)
    """
    n = len(series)

    if n < 50:
        return {
            'embedding_dimension': 2,
            'first_zero_crossing': 1
        }

    # Compute ACF to find delay
    centered = series - np.mean(series)
    acf = np.correlate(centered, centered, mode='full')
    acf = acf[n - 1:] / acf[n - 1]

    # Find first zero crossing
    zero_cross = np.where(acf < 0)[0]
    if len(zero_cross) > 0:
        delay = int(zero_cross[0])
    else:
        delay = n // 10

    delay = max(1, delay)

    # Heuristic: dimension ~ log2(delay) + 2
    # This is a simplified approach
    dim = max(2, min(int(np.log2(delay + 1)) + 2, max_dim))

    return {
        'embedding_dimension': dim,
        'first_zero_crossing': delay
    }
