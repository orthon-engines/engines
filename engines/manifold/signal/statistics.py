"""
Statistics Engine.

Imports from primitives/individual/statistics.py (canonical).
"""

import numpy as np
from typing import Dict
from engines.primitives.individual.statistics import (
    kurtosis,
    skewness,
    crest_factor,
    rms,
    zero_crossings,
    mean_crossings,
)


def compute(y: np.ndarray) -> Dict[str, float]:
    """
    Compute statistical properties of signal.

    Args:
        y: Signal values

    Returns:
        dict with kurtosis, skewness, crest_factor
    """
    result = {
        'kurtosis': np.nan,
        'skewness': np.nan,
        'crest_factor': np.nan,
    }

    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]

    if len(y) < 4:
        return result

    result['kurtosis'] = kurtosis(y, fisher=True)
    result['skewness'] = skewness(y)
    result['crest_factor'] = crest_factor(y)

    return result


def compute_kurtosis(y: np.ndarray) -> Dict[str, float]:
    """Compute kurtosis only."""
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    if len(y) < 4:
        return {'kurtosis': np.nan}
    return {'kurtosis': kurtosis(y, fisher=True)}


def compute_skewness(y: np.ndarray) -> Dict[str, float]:
    """Compute skewness only."""
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    if len(y) < 3:
        return {'skewness': np.nan}
    return {'skewness': skewness(y)}


def compute_crest_factor(y: np.ndarray) -> Dict[str, float]:
    """Compute crest factor only."""
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    if len(y) < 1:
        return {'crest_factor': np.nan}
    return {'crest_factor': crest_factor(y)}
