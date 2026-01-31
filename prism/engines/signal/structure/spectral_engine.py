"""
Spectral Engine

Power spectral density, coherence, and cross-spectral analysis.
Maps time domain to frequency domain relationships.
"""

import numpy as np
import polars as pl
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from ..primitives.individual import (
    psd,
    dominant_frequency,
    spectral_centroid,
    spectral_bandwidth,
    spectral_entropy,
)
from ..primitives.pairwise import (
    coherence,
    cross_spectral_density,
    phase_spectrum,
)


@dataclass
class SpectralConfig:
    """Configuration for spectral engine."""
    min_samples: int = 64
    sample_rate: float = 1.0
    nperseg: Optional[int] = None  # None = auto
    coherence_threshold: float = 0.5  # Threshold for significant coherence


class SpectralEngine:
    """
    Spectral Analysis Engine.

    Computes frequency-domain metrics for all signals and pairwise relationships.

    Individual signal outputs:
    - psd: Power spectral density
    - dominant_freq: Frequency of maximum power
    - spectral_centroid: Center of mass of spectrum
    - spectral_bandwidth: Spread of spectrum
    - spectral_entropy: Flatness of spectrum

    Pairwise outputs:
    - coherence_matrix: Coherence at each frequency band
    - mean_coherence: Average coherence across all pairs
    - max_coherence_pair: Pair with highest coherence
    - cross_spectral_density: Complex cross-spectrum
    - phase_coherence: Phase relationship between pairs
    """

    ENGINE_TYPE = "structure"

    def __init__(self, config: Optional[SpectralConfig] = None):
        self.config = config or SpectralConfig()

    def compute(
        self,
        signals: Dict[str, np.ndarray],
        unit_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Compute spectral analysis for an entity.

        Parameters
        ----------
        signals : dict
            Dictionary mapping signal_id to numpy array of values
        unit_id : str
            Entity identifier

        Returns
        -------
        dict
            Spectral metrics for all signals and pairs
        """
        if len(signals) < 1:
            return self._empty_result(unit_id)

        signal_ids = sorted(signals.keys())
        min_len = min(len(signals[s]) for s in signal_ids)

        if min_len < self.config.min_samples:
            return self._empty_result(unit_id)

        # Individual signal metrics
        individual_metrics = {}
        for sig_id in signal_ids:
            sig = np.asarray(signals[sig_id])[:min_len]

            # Remove NaN
            sig = sig[~np.isnan(sig)]
            if len(sig) < self.config.min_samples:
                continue

            # Compute spectral metrics
            freqs, power = psd(sig, fs=self.config.sample_rate)
            dom_freq = dominant_frequency(sig, fs=self.config.sample_rate)
            centroid = spectral_centroid(sig, fs=self.config.sample_rate)
            bandwidth = spectral_bandwidth(sig, fs=self.config.sample_rate)
            entropy = spectral_entropy(sig)

            individual_metrics[sig_id] = {
                'frequencies': freqs,
                'psd': power,
                'dominant_frequency': dom_freq,
                'spectral_centroid': centroid,
                'spectral_bandwidth': bandwidth,
                'spectral_entropy': entropy,
            }

        # Pairwise coherence matrix
        n_signals = len(signal_ids)
        coherence_matrix = np.zeros((n_signals, n_signals))
        phase_matrix = np.zeros((n_signals, n_signals))
        max_coherence = 0.0
        max_coherence_pair = (None, None)

        pairwise_metrics = {}

        if n_signals >= 2:
            for i, sig_i in enumerate(signal_ids):
                for j, sig_j in enumerate(signal_ids):
                    if i >= j:
                        continue

                    x = np.asarray(signals[sig_i])[:min_len]
                    y = np.asarray(signals[sig_j])[:min_len]

                    # Align and clean
                    valid = ~(np.isnan(x) | np.isnan(y))
                    x = x[valid]
                    y = y[valid]

                    if len(x) < self.config.min_samples:
                        continue

                    # Coherence
                    coh_freqs, coh_values = coherence(x, y, fs=self.config.sample_rate)
                    mean_coh = float(np.mean(coh_values))

                    coherence_matrix[i, j] = mean_coh
                    coherence_matrix[j, i] = mean_coh

                    # Phase
                    phase_freqs, phase_values = phase_spectrum(x, y, fs=self.config.sample_rate)
                    mean_phase = float(np.mean(np.abs(phase_values)))
                    phase_matrix[i, j] = mean_phase
                    phase_matrix[j, i] = -mean_phase

                    # Track maximum
                    if mean_coh > max_coherence:
                        max_coherence = mean_coh
                        max_coherence_pair = (sig_i, sig_j)

                    # Cross-spectral density
                    csd_freqs, csd_values = cross_spectral_density(
                        x, y, fs=self.config.sample_rate
                    )

                    pairwise_metrics[(sig_i, sig_j)] = {
                        'coherence': coh_values,
                        'coherence_freqs': coh_freqs,
                        'mean_coherence': mean_coh,
                        'phase': phase_values,
                        'csd': csd_values,
                        'csd_freqs': csd_freqs,
                    }

        # Aggregate metrics
        off_diag = coherence_matrix[np.triu_indices(n_signals, k=1)]
        if len(off_diag) > 0:
            mean_coherence = float(np.mean(off_diag))
            coherence_std = float(np.std(off_diag))
        else:
            mean_coherence = np.nan
            coherence_std = np.nan

        # Count significant coherence pairs
        n_significant = int(np.sum(off_diag > self.config.coherence_threshold))

        return {
            'unit_id': unit_id,
            'n_signals': n_signals,
            'n_samples': min_len,
            'signal_ids': signal_ids,
            'individual_metrics': individual_metrics,
            'pairwise_metrics': pairwise_metrics,
            'coherence_matrix': coherence_matrix,
            'phase_matrix': phase_matrix,
            'mean_coherence': mean_coherence,
            'coherence_std': coherence_std,
            'max_coherence': max_coherence,
            'max_coherence_pair': max_coherence_pair,
            'n_significant_coherence': n_significant,
        }

    def _empty_result(self, unit_id: str) -> Dict[str, Any]:
        """Return empty result for insufficient data."""
        return {
            'unit_id': unit_id,
            'n_signals': 0,
            'n_samples': 0,
            'signal_ids': [],
            'individual_metrics': {},
            'pairwise_metrics': {},
            'coherence_matrix': np.array([[]]),
            'phase_matrix': np.array([[]]),
            'mean_coherence': np.nan,
            'coherence_std': np.nan,
            'max_coherence': np.nan,
            'max_coherence_pair': (None, None),
            'n_significant_coherence': 0,
        }

    def to_parquet_row(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert result to flat row for parquet output."""
        # Aggregate individual metrics
        individual = result['individual_metrics']
        if individual:
            avg_dom_freq = np.mean([m['dominant_frequency'] for m in individual.values()])
            avg_entropy = np.mean([m['spectral_entropy'] for m in individual.values()])
        else:
            avg_dom_freq = np.nan
            avg_entropy = np.nan

        return {
            'unit_id': result['unit_id'],
            'n_signals': result['n_signals'],
            'n_samples': result['n_samples'],
            'mean_coherence': result['mean_coherence'],
            'coherence_std': result['coherence_std'],
            'max_coherence': result['max_coherence'],
            'n_significant_coherence': result['n_significant_coherence'],
            'avg_dominant_frequency': float(avg_dom_freq),
            'avg_spectral_entropy': float(avg_entropy),
        }
