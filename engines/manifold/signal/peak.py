"""
Peak Engine.

Imports from primitives/individual/statistics.py (canonical).
"""

import numpy as np
from engines.primitives.individual.statistics import peak_to_peak


def compute(y: np.ndarray) -> dict:
    """
    Compute peak value of signal.

    Args:
        y: Signal values

    Returns:
        dict with 'peak' and 'peak_to_peak' keys
    """
    if len(y) < 1:
        return {'peak': np.nan, 'peak_to_peak': np.nan}

    return {
        'peak': float(np.max(np.abs(y))),
        'peak_to_peak': peak_to_peak(y)
    }
