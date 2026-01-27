"""
FFT / Spectral Features
=======================

Frequency domain characteristics of the full signal:
    - Centroid: Center of mass of power spectrum
    - Bandwidth: Spread around centroid
    - Rolloff: Frequency below which 85% of energy lies
    - Low/High ratio: Energy distribution

These distinguish:
    - Narrowband: Dominant frequency (periodic)
    - Broadband: Energy spread (noise-like)
    - 1/f: Power-law spectrum (complex)

CANONICAL INTERFACE:
    def compute(observations: pd.DataFrame) -> pd.DataFrame
    Input:  [entity_id, signal_id, I, y]
    Output: [entity_id, signal_id, spectral_centroid, bandwidth, dominant_freq, ...]
"""

ENGINE_TYPE = 'signal'

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from typing import Dict, Any


def _compute_array(series: np.ndarray) -> Dict[str, Any]:
    """
    Compute spectral features on entire signal.

    Args:
        series: 1D numpy array of observations

    Returns:
        centroid: Spectral center of mass [0-0.5 normalized freq]
        bandwidth: Spread around centroid
        low_high_ratio: Low/high frequency energy ratio
        rolloff: 85% energy frequency
        dominant_freq: Frequency of peak power
        total_power: Total spectral power
    """
    series = np.asarray(series).flatten()
    n = len(series)

    if n < 16:
        return _nan_result('Signal too short (need >= 16 samples)')

    # FFT (remove DC component)
    fft_vals = fft(series - np.mean(series))
    power = np.abs(fft_vals[:n//2]) ** 2
    freqs = fftfreq(n)[:n//2]

    # Exclude DC
    power = power[1:]
    freqs = freqs[1:]

    total_power = np.sum(power)
    if len(freqs) == 0 or total_power < 1e-10:
        return _nan_result('No spectral content')

    # Normalize power for distribution metrics
    power_norm = power / total_power

    # Spectral centroid (center of mass)
    centroid = np.sum(freqs * power_norm)

    # Spectral bandwidth (spread around centroid)
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * power_norm))

    # Low/high frequency ratio (split at 0.1 Nyquist)
    low_mask = freqs < 0.1
    high_mask = freqs >= 0.1
    low_power = np.sum(power_norm[low_mask]) if np.any(low_mask) else 0
    high_power = np.sum(power_norm[high_mask]) if np.any(high_mask) else 1e-10
    low_high_ratio = low_power / high_power

    # Rolloff frequency (85% cumulative energy)
    cumsum = np.cumsum(power_norm)
    rolloff_idx = np.searchsorted(cumsum, 0.85)
    rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]

    # Dominant frequency
    dominant_idx = np.argmax(power)
    dominant_freq = freqs[dominant_idx]

    return {
        'centroid': float(centroid),
        'bandwidth': float(bandwidth),
        'low_high_ratio': float(low_high_ratio),
        'rolloff': float(rolloff),
        'dominant_freq': float(dominant_freq),
        'total_power': float(total_power),
        'n_samples': n,
    }


def _nan_result(reason: str) -> Dict[str, Any]:
    """Return NaN result with error reason."""
    return {
        'centroid': float('nan'),
        'bandwidth': float('nan'),
        'low_high_ratio': float('nan'),
        'rolloff': float('nan'),
        'dominant_freq': float('nan'),
        'total_power': float('nan'),
        'n_samples': 0,
        'error': reason,
    }


def compute(observations: pd.DataFrame) -> pd.DataFrame:
    """
    Compute spectral features.

    CANONICAL INTERFACE:
        Input:  observations [entity_id, signal_id, I, y]
        Output: primitives [entity_id, signal_id, spectral_centroid, bandwidth, ...]

    Args:
        observations: DataFrame with columns [entity_id, signal_id, I, y]

    Returns:
        DataFrame with spectral features per entity/signal
    """
    results = []

    for (entity_id, signal_id), group in observations.groupby(['entity_id', 'signal_id']):
        y = group.sort_values('I')['y'].values

        try:
            result = _compute_array(y)
            results.append({
                'entity_id': entity_id,
                'signal_id': signal_id,
                'spectral_centroid': result.get('centroid', np.nan),
                'spectral_bandwidth': result.get('bandwidth', np.nan),
                'spectral_rolloff': result.get('rolloff', np.nan),
                'dominant_frequency': result.get('dominant_freq', np.nan),
                'low_high_ratio': result.get('low_high_ratio', np.nan),
                'total_power': result.get('total_power', np.nan),
            })
        except Exception:
            results.append({
                'entity_id': entity_id,
                'signal_id': signal_id,
                'spectral_centroid': np.nan,
                'spectral_bandwidth': np.nan,
                'spectral_rolloff': np.nan,
                'dominant_frequency': np.nan,
                'low_high_ratio': np.nan,
                'total_power': np.nan,
            })

    return pd.DataFrame(results)
