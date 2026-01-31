"""
Koopman Engine

Dynamic Mode Decomposition (DMD) for linear approximation of nonlinear dynamics.
Extracts spatiotemporal modes and their growth/decay rates.
"""

import numpy as np
import polars as pl
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..primitives.matrix import dynamic_mode_decomposition


@dataclass
class KoopmanConfig:
    """Configuration for Koopman/DMD engine."""
    min_samples: int = 50
    rank: Optional[int] = None  # None = automatic rank selection
    dt: float = 1.0  # Time step for frequency computation


class KoopmanEngine:
    """
    Koopman/Dynamic Mode Decomposition Engine.

    DMD approximates nonlinear dynamics with a linear operator,
    extracting spatial modes and temporal frequencies.

    Outputs:
    - modes: DMD spatial modes (complex)
    - eigenvalues: DMD eigenvalues (complex)
    - frequencies: Oscillation frequencies (from eigenvalues)
    - growth_rates: Exponential growth/decay rates
    - amplitudes: Mode amplitudes (importance)
    - reconstruction_error: How well DMD reconstructs data
    - spectral_radius: Largest |eigenvalue| (stability indicator)
    - dominant_frequency: Frequency of strongest mode
    """

    ENGINE_TYPE = "structure"

    def __init__(self, config: Optional[KoopmanConfig] = None):
        self.config = config or KoopmanConfig()

    def compute(
        self,
        signals: Dict[str, np.ndarray],
        unit_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Compute DMD for an entity.

        Parameters
        ----------
        signals : dict
            Dictionary mapping signal_id to numpy array of values
        unit_id : str
            Entity identifier

        Returns
        -------
        dict
            DMD modes, eigenvalues, and derived metrics
        """
        if len(signals) < 1:
            return self._empty_result(unit_id)

        # Build data matrix (time × signals)
        signal_ids = sorted(signals.keys())
        min_len = min(len(signals[s]) for s in signal_ids)

        if min_len < self.config.min_samples:
            return self._empty_result(unit_id)

        # Stack as rows (time) × columns (signals)
        data = np.column_stack([
            np.asarray(signals[s])[:min_len] for s in signal_ids
        ])

        # Remove NaN rows
        valid_rows = ~np.any(np.isnan(data), axis=1)
        data = data[valid_rows]

        if len(data) < self.config.min_samples:
            return self._empty_result(unit_id)

        # DMD on transposed data (signals × time)
        # DMD expects columns to be snapshots
        data_T = data.T

        rank = self.config.rank
        if rank is None:
            # Automatic rank: min(n_signals, n_time - 1)
            rank = min(data_T.shape[0], data_T.shape[1] - 1)

        modes, eigenvalues, dynamics, amplitudes = dynamic_mode_decomposition(data_T, rank=rank)

        # Compute frequencies and growth rates from eigenvalues
        # λ = exp(ω * dt) where ω = growth_rate + i * frequency
        dt = self.config.dt
        omega = np.log(eigenvalues + 1e-10) / dt
        growth_rates = np.real(omega)
        frequencies = np.imag(omega) / (2 * np.pi)

        # Mode amplitudes (already computed by DMD)
        amplitudes = np.abs(amplitudes) if amplitudes.size > 0 else np.array([])

        # Reconstruction error
        if modes.size > 0 and dynamics.size > 0:
            try:
                reconstructed = modes @ dynamics
                error = np.linalg.norm(data_T - reconstructed) / np.linalg.norm(data_T)
            except:
                error = np.nan
        else:
            error = np.nan

        # Spectral radius (stability)
        if len(eigenvalues) > 0:
            spectral_radius = float(np.max(np.abs(eigenvalues)))
        else:
            spectral_radius = np.nan

        # Dominant frequency (largest amplitude mode)
        if len(amplitudes) > 0 and len(frequencies) > 0:
            dominant_idx = np.argmax(amplitudes)
            dominant_frequency = float(np.abs(frequencies[dominant_idx]))
        else:
            dominant_frequency = np.nan

        # Coherence (how well modes explain variance)
        if len(amplitudes) > 0:
            total_amplitude = np.sum(amplitudes)
            if total_amplitude > 0:
                mode_coherence = float(np.max(amplitudes) / total_amplitude)
            else:
                mode_coherence = np.nan
        else:
            mode_coherence = np.nan

        return {
            'unit_id': unit_id,
            'n_signals': len(signal_ids),
            'n_samples': len(data),
            'signal_ids': signal_ids,
            'rank': rank,
            'modes': modes,
            'eigenvalues': eigenvalues,
            'dynamics': dynamics,
            'frequencies': frequencies,
            'growth_rates': growth_rates,
            'amplitudes': amplitudes,
            'reconstruction_error': float(error) if not np.isnan(error) else np.nan,
            'spectral_radius': spectral_radius,
            'dominant_frequency': dominant_frequency,
            'mode_coherence': mode_coherence,
        }

    def _empty_result(self, unit_id: str) -> Dict[str, Any]:
        """Return empty result for insufficient data."""
        return {
            'unit_id': unit_id,
            'n_signals': 0,
            'n_samples': 0,
            'signal_ids': [],
            'rank': 0,
            'modes': np.array([[]]),
            'eigenvalues': np.array([]),
            'dynamics': np.array([[]]),
            'frequencies': np.array([]),
            'growth_rates': np.array([]),
            'amplitudes': np.array([]),
            'reconstruction_error': np.nan,
            'spectral_radius': np.nan,
            'dominant_frequency': np.nan,
            'mode_coherence': np.nan,
        }

    def to_parquet_row(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert result to flat row for parquet output."""
        # Store top frequencies and growth rates
        freq_dict = {}
        for i in range(min(5, len(result['frequencies']))):
            freq_dict[f'frequency_{i+1}'] = float(np.abs(result['frequencies'][i]))
            freq_dict[f'growth_rate_{i+1}'] = float(result['growth_rates'][i])

        return {
            'unit_id': result['unit_id'],
            'n_signals': result['n_signals'],
            'n_samples': result['n_samples'],
            'rank': result['rank'],
            'reconstruction_error': result['reconstruction_error'],
            'spectral_radius': result['spectral_radius'],
            'dominant_frequency': result['dominant_frequency'],
            'mode_coherence': result['mode_coherence'],
            **freq_dict,
        }
