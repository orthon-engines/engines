"""
Spectral Engine.

Imports from primitives/individual/spectral.py (canonical).
Adds spectral_slope not in primitives.
"""

import numpy as np
from typing import Dict
from engines.primitives.individual.spectral import (
    psd,
    dominant_frequency,
    spectral_centroid,
    spectral_bandwidth,
    spectral_entropy,
)


def compute(y: np.ndarray, sample_rate: float = 1.0) -> Dict[str, float]:
    """
    Compute spectral properties of signal.

    Args:
        y: Signal values
        sample_rate: Sampling rate in Hz (default: 1.0)

    Returns:
        dict with spectral_slope, dominant_freq, spectral_entropy,
        spectral_centroid, spectral_bandwidth
    """
    result = {
        'spectral_slope': np.nan,
        'dominant_freq': np.nan,
        'spectral_entropy': np.nan,
        'spectral_centroid': np.nan,
        'spectral_bandwidth': np.nan
    }

    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    # Absolute minimum check - real validation happens in runner (PR2)
    if n < 4:
        return result

    # Check for constant signal
    if np.std(y) < 1e-10:
        result['spectral_entropy'] = 0.0
        result['spectral_centroid'] = 0.0
        result['spectral_bandwidth'] = 0.0
        result['spectral_slope'] = 0.0
        result['dominant_freq'] = 0.0
        return result

    try:
        # Use primitives
        result['dominant_freq'] = dominant_frequency(y, fs=sample_rate)
        result['spectral_entropy'] = spectral_entropy(y, fs=sample_rate, normalize=True)
        result['spectral_centroid'] = spectral_centroid(y, fs=sample_rate)
        result['spectral_bandwidth'] = spectral_bandwidth(y, fs=sample_rate)

        # Spectral slope (not in primitives - compute here)
        freqs, Pxx = psd(y, fs=sample_rate)
        mask = freqs > 0
        if np.sum(mask) > 3:
            log_freqs = np.log10(freqs[mask])
            log_psd = np.log10(Pxx[mask] + 1e-10)
            if np.std(log_psd) > 1e-10:
                slope, _ = np.polyfit(log_freqs, log_psd, 1)
                result['spectral_slope'] = float(slope)

    except Exception:
        pass

    return result
