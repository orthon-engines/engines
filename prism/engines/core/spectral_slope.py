"""
Spectral Slope Analysis
=======================

Computes the slope β in S(f) ~ f^-β.

The spectral slope is related to the Hurst exponent:
    β ≈ 2H - 1 for fractional Gaussian noise
    β ≈ 2H + 1 for fractional Brownian motion

Typical values:
    - β ≈ 0: White noise
    - β ≈ 1: Pink noise (1/f)
    - β ≈ 2: Brownian motion (red noise)
"""

import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq
from typing import Dict


def compute(series: np.ndarray) -> Dict[str, float]:
    """
    Compute spectral slope (β in S(f) ~ f^-β).

    Args:
        series: 1D numpy array of observations

    Returns:
        dict with:
            - slope: Spectral slope β
            - r_squared: Fit quality
    """
    n = len(series)

    if n < 16:
        return {'slope': 0.0, 'r_squared': 0.0}

    # FFT
    fft_vals = fft(series - np.mean(series))
    power = np.abs(fft_vals[:n//2]) ** 2
    freqs = fftfreq(n)[:n//2]

    # Exclude DC and very high frequencies
    valid = (freqs > 0.01) & (freqs < 0.4)

    if np.sum(valid) < 5:
        return {'slope': 0.0, 'r_squared': 0.0}

    log_f = np.log(freqs[valid])
    log_p = np.log(power[valid] + 1e-10)

    slope, _, r_value, _, _ = stats.linregress(log_f, log_p)

    return {
        'slope': float(-slope),  # Negative because S ~ f^-β
        'r_squared': float(r_value ** 2)
    }
