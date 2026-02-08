"""
Complexity Engine.

Imports from primitives/individual/entropy.py (canonical).
Primitives handle min_samples via config - no redundant checks here.
"""

import numpy as np
from typing import Dict
from engines.primitives.individual.entropy import (
    sample_entropy,
    permutation_entropy,
    approximate_entropy,
)


def compute(y: np.ndarray) -> Dict[str, float]:
    """
    Compute entropy/complexity measures of signal.

    Args:
        y: Signal values

    Returns:
        dict with sample_entropy, permutation_entropy, approximate_entropy
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]

    # Try antropy library first (most accurate)
    try:
        from antropy import sample_entropy as ant_sampen
        from antropy import perm_entropy, app_entropy

        return {
            'sample_entropy': float(ant_sampen(y, order=2, metric='chebyshev')),
            'permutation_entropy': float(perm_entropy(y, order=3, normalize=True)),
            'approximate_entropy': float(app_entropy(y, order=2, metric='chebyshev')),
        }

    except ImportError:
        pass

    # Fall back to primitives (they handle min_samples internally)
    return {
        'sample_entropy': sample_entropy(y, m=2),
        'permutation_entropy': permutation_entropy(y, order=3, normalize=True),
        'approximate_entropy': approximate_entropy(y, m=2),
    }


def compute_sample_entropy(y: np.ndarray, m: int = 2) -> Dict[str, float]:
    """Compute sample entropy only."""
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    return {'sample_entropy': sample_entropy(y, m=m)}


def compute_permutation_entropy(y: np.ndarray, order: int = 3) -> Dict[str, float]:
    """Compute permutation entropy only."""
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    return {'permutation_entropy': permutation_entropy(y, order=order, normalize=True)}


def compute_approximate_entropy(y: np.ndarray, m: int = 2) -> Dict[str, float]:
    """Compute approximate entropy only."""
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    return {'approximate_entropy': approximate_entropy(y, m=m)}
