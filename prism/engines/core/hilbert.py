"""
Hilbert Transform Engine

Point-wise engine for instantaneous amplitude, phase, and frequency.

CANONICAL INTERFACE:
    def compute(observations: pd.DataFrame) -> pd.DataFrame
    Input:  [entity_id, signal_id, I, y]
    Output: [entity_id, signal_id, I, amplitude, phase, frequency]

The Hilbert transform provides an analytic signal representation,
enabling extraction of:
- Instantaneous amplitude (envelope)
- Instantaneous phase
- Instantaneous frequency
"""

ENGINE_TYPE = 'point'

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from typing import Dict, Any


def compute(
    observations: pd.DataFrame,
    unwrap_phase: bool = True,
) -> pd.DataFrame:
    """
    Compute Hilbert transform for all signals.

    CANONICAL INTERFACE:
        Input:  observations [entity_id, signal_id, I, y]
        Output: primitives [entity_id, signal_id, I, amplitude, phase, frequency]

    Parameters
    ----------
    observations : pd.DataFrame
        Must have columns: entity_id, signal_id, I, y
    unwrap_phase : bool, optional
        Whether to unwrap phase (remove 2pi jumps) (default: True)

    Returns
    -------
    pd.DataFrame
        Point-wise Hilbert metrics
    """
    results = []

    for (entity_id, signal_id), group in observations.groupby(['entity_id', 'signal_id']):
        group = group.sort_values('I')
        y = group['y'].values
        I_values = group['I'].values

        if len(y) < 10:
            for I in I_values:
                results.append({
                    'entity_id': entity_id,
                    'signal_id': signal_id,
                    'I': I,
                    'amplitude': np.nan,
                    'phase': np.nan,
                    'frequency': np.nan,
                })
            continue

        try:
            # Compute analytic signal
            analytic = hilbert(y)

            # Instantaneous amplitude (envelope)
            amplitude = np.abs(analytic)

            # Instantaneous phase
            phase = np.angle(analytic)
            if unwrap_phase:
                phase = np.unwrap(phase)

            # Instantaneous frequency (derivative of phase)
            if len(I_values) > 1:
                dt = np.diff(I_values.astype(np.float64))
                dt = np.concatenate([[dt[0]], dt])  # Pad first element
                frequency = np.gradient(phase) / (2 * np.pi * dt + 1e-10)
            else:
                frequency = np.zeros_like(phase)

            for i, I in enumerate(I_values):
                results.append({
                    'entity_id': entity_id,
                    'signal_id': signal_id,
                    'I': I,
                    'amplitude': float(amplitude[i]),
                    'phase': float(phase[i]),
                    'frequency': float(frequency[i]),
                })

        except Exception:
            for I in I_values:
                results.append({
                    'entity_id': entity_id,
                    'signal_id': signal_id,
                    'I': I,
                    'amplitude': np.nan,
                    'phase': np.nan,
                    'frequency': np.nan,
                })

    return pd.DataFrame(results)


def compute_summary(
    observations: pd.DataFrame,
    unwrap_phase: bool = True,
) -> pd.DataFrame:
    """
    Compute summary Hilbert metrics per signal (not point-wise).

    CANONICAL INTERFACE:
        Input:  observations [entity_id, signal_id, I, y]
        Output: primitives [entity_id, signal_id, amplitude_mean, amplitude_std,
                           phase_std, frequency_mean, frequency_std]

    Parameters
    ----------
    observations : pd.DataFrame
        Must have columns: entity_id, signal_id, I, y
    unwrap_phase : bool, optional
        Whether to unwrap phase (default: True)

    Returns
    -------
    pd.DataFrame
        Summary Hilbert metrics per signal
    """
    results = []

    for (entity_id, signal_id), group in observations.groupby(['entity_id', 'signal_id']):
        y = group.sort_values('I')['y'].values
        I_values = group.sort_values('I')['I'].values

        if len(y) < 10:
            results.append({
                'entity_id': entity_id,
                'signal_id': signal_id,
                'amplitude_mean': np.nan,
                'amplitude_std': np.nan,
                'phase_std': np.nan,
                'frequency_mean': np.nan,
                'frequency_std': np.nan,
            })
            continue

        try:
            analytic = hilbert(y)
            amplitude = np.abs(analytic)
            phase = np.angle(analytic)
            if unwrap_phase:
                phase = np.unwrap(phase)

            if len(I_values) > 1:
                dt = np.diff(I_values.astype(np.float64))
                dt = np.concatenate([[dt[0]], dt])
                frequency = np.gradient(phase) / (2 * np.pi * dt + 1e-10)
            else:
                frequency = np.zeros_like(phase)

            results.append({
                'entity_id': entity_id,
                'signal_id': signal_id,
                'amplitude_mean': float(np.mean(amplitude)),
                'amplitude_std': float(np.std(amplitude)),
                'phase_std': float(np.std(phase)),
                'frequency_mean': float(np.mean(frequency)),
                'frequency_std': float(np.std(frequency)),
            })

        except Exception:
            results.append({
                'entity_id': entity_id,
                'signal_id': signal_id,
                'amplitude_mean': np.nan,
                'amplitude_std': np.nan,
                'phase_std': np.nan,
                'frequency_mean': np.nan,
                'frequency_std': np.nan,
            })

    return pd.DataFrame(results)
