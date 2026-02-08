"""
THD (Total Harmonic Distortion) Engine.

Measures harmonic distortion (purity of periodic signal).
"""

import numpy as np
from typing import Dict


def compute(y: np.ndarray, n_harmonics: int = 5) -> Dict[str, float]:
    """
    Compute total harmonic distortion.

    Args:
        y: Signal values
        n_harmonics: Number of harmonics to measure (default 5)

    Returns:
        dict with thd_percent, thd_db, fundamental_power, n_harmonics_found
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    result = {
        'thd_percent': np.nan,
        'thd_db': np.nan,
        'thd_fundamental_power': np.nan,
        'thd_harmonic_power': np.nan,
        'n_harmonics_found': 0,
    }

    if n < 16:
        return result

    try:
        # Remove DC
        y = y - np.mean(y)

        # FFT
        fft = np.fft.rfft(y)
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(n)

        # Exclude DC
        power_no_dc = power[1:]
        freqs_no_dc = freqs[1:]

        if len(power_no_dc) == 0:
            return result

        # Find fundamental (highest peak)
        fundamental_idx = np.argmax(power_no_dc)
        fundamental_freq = freqs_no_dc[fundamental_idx]
        fundamental_power = power_no_dc[fundamental_idx]

        if fundamental_power == 0 or fundamental_freq == 0:
            return result

        # Find harmonics (2f, 3f, 4f, ...)
        harmonic_powers = []
        harmonics_found = 0

        for h in range(2, n_harmonics + 2):
            harmonic_freq = h * fundamental_freq

            # Find closest bin to harmonic frequency
            freq_idx = np.argmin(np.abs(freqs_no_dc - harmonic_freq))

            # Check if within reasonable range (within 1 bin)
            if abs(freqs_no_dc[freq_idx] - harmonic_freq) < (freqs_no_dc[1] - freqs_no_dc[0]) * 1.5:
                harmonic_powers.append(power_no_dc[freq_idx])
                harmonics_found += 1
            else:
                # Harmonic out of range
                break

        # THD = sqrt(sum(harmonic_powers)) / fundamental_power
        if harmonic_powers:
            harmonic_sum = np.sum(harmonic_powers)
            thd_linear = np.sqrt(harmonic_sum) / np.sqrt(fundamental_power)
            thd_percent = thd_linear * 100
            thd_db = 20 * np.log10(thd_linear) if thd_linear > 0 else -np.inf
        else:
            thd_percent = 0.0
            thd_db = -np.inf
            harmonic_sum = 0.0

        result = {
            'thd_percent': float(thd_percent),
            'thd_db': float(thd_db),
            'thd_fundamental_power': float(fundamental_power),
            'thd_harmonic_power': float(harmonic_sum),
            'n_harmonics_found': harmonics_found,
        }

    except Exception:
        pass

    return result
