"""
Hilbert Transform Engine
========================

Point-wise engine for instantaneous amplitude, phase, and frequency.

The Hilbert transform provides an analytic signal representation,
enabling extraction of:
- Instantaneous amplitude (envelope)
- Instantaneous phase
- Instantaneous frequency

These are point-wise operations - output has same length as input.
"""

import numpy as np
from scipy.signal import hilbert
from typing import Tuple, Optional

from prism.core.signals.types import DenseSignal


# =============================================================================
# HILBERT ENGINE
# =============================================================================

class HilbertEngine:
    """
    Point-wise Hilbert transform engine.

    Computes instantaneous amplitude, phase, and frequency from
    the analytic signal representation.

    Usage:
        engine = HilbertEngine()
        amplitude, phase, frequency = engine.compute(timestamps, values)
    """

    def __init__(self, unwrap_phase: bool = True):
        """
        Initialize Hilbert engine.

        Parameters:
            unwrap_phase: Whether to unwrap phase (remove 2π jumps)
        """
        self.unwrap_phase = unwrap_phase

    def compute(
        self,
        signal_id: str,
        timestamps: np.ndarray,
        values: np.ndarray,
    ) -> Tuple[DenseSignal, DenseSignal, DenseSignal]:
        """
        Compute Hilbert transform outputs.

        Returns:
            (amplitude, phase, frequency) as DenseSignal objects
        """
        # Compute analytic signal
        analytic = hilbert(values)

        # Instantaneous amplitude (envelope)
        amplitude_values = np.abs(analytic)

        # Instantaneous phase
        phase_values = np.angle(analytic)
        if self.unwrap_phase:
            phase_values = np.unwrap(phase_values)

        # Instantaneous frequency (derivative of phase)
        if len(timestamps) > 1:
            # Compute time differences
            dt = np.diff(timestamps.astype(np.float64))
            # Handle non-uniform sampling
            dt = np.concatenate([[dt[0]], dt])  # Pad first element
            frequency_values = np.gradient(phase_values) / (2 * np.pi * dt + 1e-10)
        else:
            frequency_values = np.zeros_like(phase_values)

        # Create output signals
        amplitude = DenseSignal(
            signal_id=f"{signal_id}_hilbert_amplitude",
            timestamps=timestamps,
            values=amplitude_values,
            source_signal=signal_id,
            engine='hilbert',
            parameters={'metric': 'amplitude'},
        )

        phase = DenseSignal(
            signal_id=f"{signal_id}_hilbert_phase",
            timestamps=timestamps,
            values=phase_values,
            source_signal=signal_id,
            engine='hilbert',
            parameters={'metric': 'phase', 'unwrapped': self.unwrap_phase},
        )

        frequency = DenseSignal(
            signal_id=f"{signal_id}_hilbert_frequency",
            timestamps=timestamps,
            values=frequency_values,
            source_signal=signal_id,
            engine='hilbert',
            parameters={'metric': 'instantaneous_frequency'},
        )

        return amplitude, phase, frequency


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_hilbert_amplitude(
    signal_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
) -> DenseSignal:
    """Compute instantaneous amplitude (envelope)."""
    analytic = hilbert(values)
    amplitude_values = np.abs(analytic)

    return DenseSignal(
        signal_id=f"{signal_id}_hilbert_amplitude",
        timestamps=timestamps,
        values=amplitude_values,
        source_signal=signal_id,
        engine='hilbert',
        parameters={'metric': 'amplitude'},
    )


def compute_hilbert_phase(
    signal_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    unwrap: bool = True,
) -> DenseSignal:
    """Compute instantaneous phase."""
    analytic = hilbert(values)
    phase_values = np.angle(analytic)
    if unwrap:
        phase_values = np.unwrap(phase_values)

    return DenseSignal(
        signal_id=f"{signal_id}_hilbert_phase",
        timestamps=timestamps,
        values=phase_values,
        source_signal=signal_id,
        engine='hilbert',
        parameters={'metric': 'phase', 'unwrapped': unwrap},
    )


def compute_hilbert_frequency(
    signal_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
) -> DenseSignal:
    """Compute instantaneous frequency."""
    analytic = hilbert(values)
    phase = np.unwrap(np.angle(analytic))

    if len(timestamps) > 1:
        dt = np.diff(timestamps.astype(np.float64))
        dt = np.concatenate([[dt[0]], dt])
        frequency_values = np.gradient(phase) / (2 * np.pi * dt + 1e-10)
    else:
        frequency_values = np.zeros_like(phase)

    return DenseSignal(
        signal_id=f"{signal_id}_hilbert_frequency",
        timestamps=timestamps,
        values=frequency_values,
        source_signal=signal_id,
        engine='hilbert',
        parameters={'metric': 'instantaneous_frequency'},
    )


# =============================================================================
# DERIVED METRICS
# =============================================================================

def compute_amplitude_modulation(
    signal_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
) -> DenseSignal:
    """
    Compute amplitude modulation rate.

    Rate of change of the envelope - useful for detecting
    transient events and modulation patterns.
    """
    analytic = hilbert(values)
    amplitude = np.abs(analytic)
    modulation = np.gradient(amplitude)

    return DenseSignal(
        signal_id=f"{signal_id}_amplitude_modulation",
        timestamps=timestamps,
        values=modulation,
        source_signal=signal_id,
        engine='hilbert',
        parameters={'metric': 'amplitude_modulation'},
    )


def compute_frequency_modulation(
    signal_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
) -> DenseSignal:
    """
    Compute frequency modulation rate.

    Rate of change of instantaneous frequency - useful for
    detecting frequency sweeps and chirps.
    """
    freq_signal = compute_hilbert_frequency(signal_id, timestamps, values)
    modulation = np.gradient(freq_signal.values)

    return DenseSignal(
        signal_id=f"{signal_id}_frequency_modulation",
        timestamps=timestamps,
        values=modulation,
        source_signal=signal_id,
        engine='hilbert',
        parameters={'metric': 'frequency_modulation'},
    )
