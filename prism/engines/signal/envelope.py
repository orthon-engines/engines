"""
Envelope Engine.

Imports from primitives/individual/hilbert.py (canonical).
"""

import numpy as np
from prism.primitives.individual.hilbert import envelope
from prism.primitives.individual.statistics import kurtosis, rms


def compute(y: np.ndarray) -> dict:
    """
    Compute Hilbert envelope metrics of signal.

    Args:
        y: Signal values

    Returns:
        dict with envelope_rms, envelope_peak, envelope_kurtosis
    """
    if len(y) < 10:
        return {
            'envelope_rms': np.nan,
            'envelope_peak': np.nan,
            'envelope_kurtosis': np.nan
        }

    try:
        env = envelope(y)

        return {
            'envelope_rms': rms(env),
            'envelope_peak': float(np.max(env)),
            'envelope_kurtosis': kurtosis(env, fisher=True)
        }
    except Exception:
        return {
            'envelope_rms': np.nan,
            'envelope_peak': np.nan,
            'envelope_kurtosis': np.nan
        }
