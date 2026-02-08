"""
SNR (Signal-to-Noise Ratio) Engine.

Quantifies signal vs noise content.
"""

import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    """
    Compute signal-to-noise ratio.

    Uses spectral method: estimates signal as dominant spectral components,
    noise as remainder.

    Args:
        y: Signal values

    Returns:
        dict with snr_db, snr_linear, signal_power, noise_power
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    result = {
        'snr_db': np.nan,
        'snr_linear': np.nan,
        'signal_power': np.nan,
        'noise_power': np.nan,
    }

    if n < 8:
        return result

    try:
        # Remove DC
        y = y - np.mean(y)

        # FFT
        fft = np.fft.rfft(y)
        power = np.abs(fft) ** 2

        # Exclude DC
        power = power[1:]

        if len(power) == 0:
            return result

        total_power = np.sum(power)
        if total_power == 0:
            return result

        # Estimate signal: power above noise floor
        # Noise floor estimated as median power
        noise_floor = np.median(power)

        # Signal components: bins significantly above noise floor
        signal_threshold = noise_floor * 3  # 3x noise floor
        signal_mask = power > signal_threshold

        signal_power = float(np.sum(power[signal_mask]))
        noise_power = float(np.sum(power[~signal_mask]))

        # Handle edge cases
        if noise_power == 0:
            noise_power = 1e-10  # Avoid division by zero

        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else -np.inf

        result = {
            'snr_db': float(snr_db),
            'snr_linear': float(snr_linear),
            'signal_power': signal_power,
            'noise_power': noise_power,
        }

    except Exception:
        pass

    return result
