"""
Phase Coherence Engine.

Measures phase stability across time using Hilbert transform.
"""

import numpy as np
from typing import Dict


def compute(y: np.ndarray, n_segments: int = 4) -> Dict[str, float]:
    """
    Measure phase coherence.

    Args:
        y: Signal values
        n_segments: Number of segments to compare

    Returns:
        dict with mean_coherence, coherence_std, coherence_trend
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    result = {
        'phase_coherence': np.nan,
        'coherence_std': np.nan,
        'coherence_trend': np.nan,
    }

    if n < 32:
        return result

    try:
        from scipy.signal import hilbert

        # Remove DC
        y = y - np.mean(y)

        # Compute analytic signal via Hilbert transform
        analytic = hilbert(y)
        phase = np.angle(analytic)

        # Segment the signal
        seg_len = n // n_segments
        if seg_len < 8:
            return result

        coherences = []
        for i in range(n_segments):
            start = i * seg_len
            end = start + seg_len
            seg_phase = phase[start:end]

            # Phase coherence: mean resultant length (circular statistics)
            # R = |mean(exp(i*phase))|
            mean_vector = np.mean(np.exp(1j * seg_phase))
            coherence = np.abs(mean_vector)
            coherences.append(coherence)

        coherences = np.array(coherences)

        # Overall coherence (using all data)
        mean_vector_all = np.mean(np.exp(1j * phase))
        mean_coherence = float(np.abs(mean_vector_all))

        # Coherence variability
        coherence_std = float(np.std(coherences))

        # Coherence trend (is it increasing or decreasing?)
        if len(coherences) >= 2:
            x = np.arange(len(coherences))
            slope, _ = np.polyfit(x, coherences, 1)
            coherence_trend = float(slope)
        else:
            coherence_trend = 0.0

        result = {
            'phase_coherence': mean_coherence,
            'coherence_std': coherence_std,
            'coherence_trend': coherence_trend,
        }

    except ImportError:
        # scipy not available - simple fallback
        try:
            # Use zero-crossing regularity as proxy
            y_centered = y - np.mean(y)
            crossings = np.where(np.diff(np.signbit(y_centered)))[0]

            if len(crossings) > 2:
                intervals = np.diff(crossings)
                mean_interval = np.mean(intervals)
                if mean_interval > 0:
                    cv = np.std(intervals) / mean_interval  # Coefficient of variation
                    mean_coherence = float(max(0, 1 - cv))  # High regularity = high coherence
                else:
                    mean_coherence = 0.0
            else:
                mean_coherence = 0.0

            result = {
                'phase_coherence': mean_coherence,
                'coherence_std': np.nan,
                'coherence_trend': np.nan,
            }
        except Exception:
            pass

    except Exception:
        pass

    return result
