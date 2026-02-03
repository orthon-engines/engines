"""
Fundamental Frequency Engine.

Detects lowest dominant frequency (base rhythm of signal).
"""

import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    """
    Detect fundamental frequency.

    Args:
        y: Signal values

    Returns:
        dict with fundamental_freq, fundamental_power, fundamental_ratio, confidence
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    result = {
        'fundamental_freq': np.nan,
        'fundamental_power': np.nan,
        'fundamental_ratio': np.nan,
        'fundamental_confidence': np.nan,
    }

    if n < 8:
        return result

    try:
        # Remove DC
        y = y - np.mean(y)

        # FFT
        fft = np.fft.rfft(y)
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(n)

        # Exclude DC (index 0)
        power_no_dc = power[1:]
        freqs_no_dc = freqs[1:]

        if len(power_no_dc) == 0:
            return result

        total_power = np.sum(power_no_dc)
        if total_power == 0:
            return result

        # Find peaks (local maxima)
        peaks = []
        for i in range(1, len(power_no_dc) - 1):
            if power_no_dc[i] > power_no_dc[i-1] and power_no_dc[i] > power_no_dc[i+1]:
                peaks.append((i, power_no_dc[i], freqs_no_dc[i]))

        if not peaks:
            # No peaks - use max
            idx = np.argmax(power_no_dc)
            fundamental_freq = float(freqs_no_dc[idx])
            fundamental_power = float(power_no_dc[idx])
            confidence = 0.0  # No clear peak
        else:
            # Sort by power descending
            peaks.sort(key=lambda x: x[1], reverse=True)

            # Find lowest frequency among significant peaks (top 3 by power)
            significant_peaks = peaks[:min(3, len(peaks))]
            # Pick lowest frequency among them
            significant_peaks.sort(key=lambda x: x[2])  # Sort by freq

            idx, fundamental_power, fundamental_freq = significant_peaks[0]
            fundamental_power = float(fundamental_power)
            fundamental_freq = float(fundamental_freq)

            # Confidence: peak prominence relative to noise floor
            noise_floor = np.median(power_no_dc)
            if noise_floor > 0:
                prominence = fundamental_power / noise_floor
                confidence = float(min(1.0, prominence / 10.0))  # Normalize to 0-1
            else:
                confidence = 1.0

        fundamental_ratio = fundamental_power / total_power

        result = {
            'fundamental_freq': fundamental_freq,
            'fundamental_power': fundamental_power,
            'fundamental_ratio': float(fundamental_ratio),
            'fundamental_confidence': confidence,
        }

    except Exception:
        pass

    return result
