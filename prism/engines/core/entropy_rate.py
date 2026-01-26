"""
Entropy Rate
============

Measures the rate of change of entropy over time.
Useful for detecting regime changes where complexity is
either increasing or decreasing.

Positive rate: Complexity increasing
Negative rate: Complexity decreasing (structure emerging)
Near zero: Stable complexity
"""

import numpy as np
from typing import Dict, Optional
from .entropy import compute_permutation_entropy as compute_perm_entropy


def compute(
    series: np.ndarray,
    previous_entropy: Optional[float] = None,
    window_size: int = None
) -> Dict[str, float]:
    """
    Compute entropy rate (change in entropy).

    Args:
        series: 1D numpy array of observations
        previous_entropy: Entropy from previous window
        window_size: Window size for internal windowed computation

    Returns:
        dict with:
            - entropy_rate: Rate of change (entropy_now - entropy_prev)
            - current_entropy: Current window's entropy
    """
    # Compute current entropy
    result = compute_perm_entropy(series)
    current_entropy = result['entropy']

    if previous_entropy is not None:
        entropy_rate = current_entropy - previous_entropy
    else:
        entropy_rate = 0.0

    return {
        'entropy_rate': float(entropy_rate),
        'current_entropy': float(current_entropy)
    }
