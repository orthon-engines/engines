"""
Frequency Bands Engine.

Imports from primitives/individual/spectral.py (canonical).
"""

import numpy as np
from typing import Dict
from engines.primitives.individual.spectral import psd


def compute(y: np.ndarray, sample_rate: float = 1.0, bands: dict = None) -> Dict[str, float]:
    """
    Compute energy in frequency bands.

    Args:
        y: Signal values
        sample_rate: Sampling rate in Hz
        bands: Dict mapping band names to [low, high] frequency ranges
               e.g., {'low': [0, 100], 'mid': [100, 500], 'high': [500, 2000]}
               If None, uses default bands based on Nyquist

    Returns:
        dict with 'band_{name}' keys for each band
    """
    result = {}

    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    if n < 64:
        if bands:
            for name in bands:
                result[f'band_{name}'] = np.nan
                result[f'band_{name}_rel'] = np.nan
        result['total_power'] = np.nan
        return result

    try:
        nyquist = sample_rate / 2

        # Use primitive for PSD
        freqs, power = psd(y, fs=sample_rate, nperseg=min(256, n))

        # Default bands if not specified
        if bands is None:
            bands = {
                'low': [0, nyquist * 0.1],
                'mid': [nyquist * 0.1, nyquist * 0.5],
                'high': [nyquist * 0.5, nyquist]
            }

        total_power = np.sum(power)

        for name, (low, high) in bands.items():
            if low >= nyquist:
                result[f'band_{name}'] = 0.0
                result[f'band_{name}_rel'] = 0.0
                continue

            high = min(high, nyquist)

            if low >= high:
                result[f'band_{name}'] = 0.0
                result[f'band_{name}_rel'] = 0.0
                continue

            mask = (freqs >= low) & (freqs <= high)
            band_power = float(np.sum(power[mask]))
            result[f'band_{name}'] = band_power
            result[f'band_{name}_rel'] = float(band_power / (total_power + 1e-10))

        result['total_power'] = float(total_power)
        result['nyquist'] = float(nyquist)

    except Exception:
        if bands:
            for name in bands:
                result[f'band_{name}'] = np.nan
                result[f'band_{name}_rel'] = np.nan
        result['total_power'] = np.nan

    return result
