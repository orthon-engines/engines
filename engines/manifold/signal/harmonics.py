"""
Harmonics Engine.

Computes harmonic analysis (fundamental + harmonics + THD).
Replaces motor_signature, gear_mesh, rotor_dynamics outputs.
"""

import numpy as np
from typing import Dict


def compute(y: np.ndarray, sample_rate: float = 1.0, fundamental: float = None) -> Dict[str, float]:
    """
    Compute harmonic analysis of signal.

    Args:
        y: Signal values
        sample_rate: Sampling rate in Hz
        fundamental: Known fundamental frequency. If None, auto-detected.

    Returns:
        dict with fundamental_freq, fundamental_amplitude, harmonic_2x, harmonic_3x, thd
    """
    result = {
        'fundamental_freq': np.nan,
        'fundamental_amplitude': np.nan,
        'harmonic_2x': np.nan,
        'harmonic_3x': np.nan,
        'thd': np.nan
    }

    # Handle NaN values
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    # Absolute minimum check - real validation happens in runner (PR2)
    if n < 4:
        return result

    # Check for constant signal
    if np.std(y) < 1e-10:
        result['fundamental_freq'] = 0.0
        result['fundamental_amplitude'] = 0.0
        result['thd'] = 0.0
        return result

    try:
        # Compute FFT
        fft = np.fft.rfft(y - np.mean(y))
        freqs = np.fft.rfftfreq(n, d=1/sample_rate)
        mag = np.abs(fft)

        # Nyquist frequency
        nyquist = sample_rate / 2

        # Frequency resolution
        freq_resolution = sample_rate / n

        # Find fundamental
        if fundamental is None:
            # Auto-detect: largest peak (skip DC)
            if len(mag) > 1:
                fund_idx = np.argmax(mag[1:]) + 1
                fundamental = freqs[fund_idx]
            else:
                return result
        else:
            # Find closest frequency bin
            fund_idx = np.argmin(np.abs(freqs - fundamental))

        # Check if fundamental is valid
        if fundamental < freq_resolution:
            # Fundamental below frequency resolution - meaningless
            result['fundamental_freq'] = 0.0
            result['fundamental_amplitude'] = float(mag[0]) if len(mag) > 0 else 0.0
            return result

        f1_amp = mag[fund_idx]

        # Get harmonic amplitudes with Nyquist checking
        def amplitude_at_freq(target_freq):
            if target_freq > nyquist:
                return np.nan  # Above Nyquist - cannot measure
            idx = np.argmin(np.abs(freqs - target_freq))
            return mag[idx] if idx < len(mag) else 0.0

        f2_amp = amplitude_at_freq(fundamental * 2)
        f3_amp = amplitude_at_freq(fundamental * 3)

        # THD: sqrt(sum of harmonics^2) / fundamental
        harmonics_sq = 0.0
        valid_harmonics = 0
        for h in range(2, 11):
            h_freq = fundamental * h
            if h_freq > nyquist:
                break  # Stop at Nyquist
            h_amp = amplitude_at_freq(h_freq)
            if not np.isnan(h_amp):
                harmonics_sq += h_amp ** 2
                valid_harmonics += 1

        if f1_amp > 1e-10 and valid_harmonics > 0:
            thd = np.sqrt(harmonics_sq) / f1_amp * 100  # As percentage
        else:
            thd = 0.0

        result = {
            'fundamental_freq': float(fundamental),
            'fundamental_amplitude': float(f1_amp),
            'harmonic_2x': float(f2_amp) if not np.isnan(f2_amp) else np.nan,
            'harmonic_3x': float(f3_amp) if not np.isnan(f3_amp) else np.nan,
            'thd': float(thd)
        }

    except Exception:
        pass

    return result
