"""
RMS (Root Mean Square) Engine.

Imports from primitives/individual/statistics.py (canonical).
"""

import numpy as np
from engines.primitives.individual.statistics import rms


def compute(y: np.ndarray) -> dict:
    """
    Compute RMS of signal.

    Args:
        y: Signal values

    Returns:
        dict with 'rms' key
    """
    if len(y) < 1:
        return {'rms': np.nan}

    return {'rms': rms(y)}
