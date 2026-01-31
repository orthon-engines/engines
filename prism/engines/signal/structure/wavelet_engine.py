"""
Wavelet Engine

Wavelet analysis for time-frequency decomposition and coherence.
Captures both transient and stationary patterns.
"""

import numpy as np
import polars as pl
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from ..primitives.pairwise import wavelet_coherence


@dataclass
class WaveletConfig:
    """Configuration for wavelet engine."""
    min_samples: int = 64
    wavelet: str = 'morl'  # Morlet wavelet
    scales: Optional[np.ndarray] = None  # None = auto
    n_scales: int = 32
    coherence_threshold: float = 0.5


class WaveletEngine:
    """
    Wavelet Analysis Engine.

    Performs continuous wavelet transform and wavelet coherence analysis.
    Captures time-frequency structure that Fourier misses.

    Individual signal outputs:
    - wavelet_power: Power at each scale/time
    - scale_averaged_power: Power averaged over time at each scale
    - time_averaged_power: Power averaged over scale at each time
    - dominant_scale: Scale with maximum power
    - wavelet_entropy: Entropy of scale distribution

    Pairwise outputs:
    - wavelet_coherence: Time-frequency coherence
    - mean_coherence: Average coherence
    - coherence_peaks: Times/scales of high coherence
    """

    ENGINE_TYPE = "structure"

    def __init__(self, config: Optional[WaveletConfig] = None):
        self.config = config or WaveletConfig()

    def compute(
        self,
        signals: Dict[str, np.ndarray],
        unit_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Compute wavelet analysis for an entity.

        Parameters
        ----------
        signals : dict
            Dictionary mapping signal_id to numpy array of values
        unit_id : str
            Entity identifier

        Returns
        -------
        dict
            Wavelet metrics for all signals and pairs
        """
        if len(signals) < 1:
            return self._empty_result(unit_id)

        signal_ids = sorted(signals.keys())
        min_len = min(len(signals[s]) for s in signal_ids)

        if min_len < self.config.min_samples:
            return self._empty_result(unit_id)

        # Set up scales
        if self.config.scales is None:
            scales = np.geomspace(2, min_len // 4, self.config.n_scales)
        else:
            scales = self.config.scales

        # Individual signal metrics
        individual_metrics = {}
        for sig_id in signal_ids:
            sig = np.asarray(signals[sig_id])[:min_len]

            # Remove NaN (interpolate for continuity)
            nan_mask = np.isnan(sig)
            if np.any(nan_mask):
                sig = np.interp(
                    np.arange(len(sig)),
                    np.arange(len(sig))[~nan_mask],
                    sig[~nan_mask]
                )

            if len(sig) < self.config.min_samples:
                continue

            # Continuous wavelet transform
            try:
                import pywt
                coefficients, frequencies = pywt.cwt(sig, scales, self.config.wavelet)
                wt_scales = scales
            except ImportError:
                # Fallback: skip wavelet analysis without pywt
                continue

            # Power
            power = np.abs(coefficients) ** 2

            # Scale-averaged power (time series of power at each scale)
            scale_power = np.mean(power, axis=1)

            # Time-averaged power (power at each time point)
            time_power = np.mean(power, axis=0)

            # Dominant scale (highest power)
            if len(scale_power) > 0:
                dominant_scale_idx = np.argmax(scale_power)
                dominant_scale = float(wt_scales[dominant_scale_idx])
            else:
                dominant_scale = np.nan

            # Wavelet entropy (across scales)
            total_power = np.sum(scale_power)
            if total_power > 0:
                p = scale_power / total_power
                p = p[p > 1e-10]
                wavelet_entropy = -np.sum(p * np.log(p))
            else:
                wavelet_entropy = 0.0

            # Temporal variability (std of time-averaged power)
            temporal_variability = float(np.std(time_power)) if len(time_power) > 0 else np.nan

            individual_metrics[sig_id] = {
                'coefficients': coefficients,
                'power': power,
                'scales': wt_scales,
                'scale_power': scale_power,
                'time_power': time_power,
                'dominant_scale': dominant_scale,
                'wavelet_entropy': float(wavelet_entropy),
                'temporal_variability': temporal_variability,
            }

        # Pairwise wavelet coherence
        n_signals = len(signal_ids)
        coherence_matrix = np.zeros((n_signals, n_signals))
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

                    # Interpolate NaN
                    for arr in [x, y]:
                        nan_mask = np.isnan(arr)
                        if np.any(nan_mask):
                            arr[:] = np.interp(
                                np.arange(len(arr)),
                                np.arange(len(arr))[~nan_mask],
                                arr[~nan_mask]
                            )

                    if len(x) < self.config.min_samples:
                        continue

                    # Wavelet coherence
                    wc_scales, wc_time, wc = wavelet_coherence(x, y, scales=scales)
                    mean_coh = float(np.mean(wc))

                    coherence_matrix[i, j] = mean_coh
                    coherence_matrix[j, i] = mean_coh

                    # Track maximum
                    if mean_coh > max_coherence:
                        max_coherence = mean_coh
                        max_coherence_pair = (sig_i, sig_j)

                    # Find coherence peaks (times/scales with high coherence)
                    threshold = self.config.coherence_threshold
                    peak_mask = wc > threshold
                    n_peaks = int(np.sum(peak_mask))

                    pairwise_metrics[(sig_i, sig_j)] = {
                        'coherence': wc,
                        'mean_coherence': mean_coh,
                        'n_coherence_peaks': n_peaks,
                        'peak_fraction': float(n_peaks / wc.size) if wc.size > 0 else 0.0,
                    }

        # Aggregate metrics
        off_diag = coherence_matrix[np.triu_indices(n_signals, k=1)]
        if len(off_diag) > 0:
            mean_coherence = float(np.mean(off_diag))
            coherence_std = float(np.std(off_diag))
        else:
            mean_coherence = np.nan
            coherence_std = np.nan

        # Average wavelet entropy
        if individual_metrics:
            avg_entropy = np.mean([m['wavelet_entropy'] for m in individual_metrics.values()])
            avg_variability = np.mean([m['temporal_variability'] for m in individual_metrics.values()])
        else:
            avg_entropy = np.nan
            avg_variability = np.nan

        return {
            'unit_id': unit_id,
            'n_signals': n_signals,
            'n_samples': min_len,
            'n_scales': len(scales),
            'signal_ids': signal_ids,
            'scales': scales,
            'individual_metrics': individual_metrics,
            'pairwise_metrics': pairwise_metrics,
            'coherence_matrix': coherence_matrix,
            'mean_coherence': mean_coherence,
            'coherence_std': coherence_std,
            'max_coherence': max_coherence,
            'max_coherence_pair': max_coherence_pair,
            'avg_wavelet_entropy': float(avg_entropy),
            'avg_temporal_variability': float(avg_variability),
        }

    def _empty_result(self, unit_id: str) -> Dict[str, Any]:
        """Return empty result for insufficient data."""
        return {
            'unit_id': unit_id,
            'n_signals': 0,
            'n_samples': 0,
            'n_scales': 0,
            'signal_ids': [],
            'scales': np.array([]),
            'individual_metrics': {},
            'pairwise_metrics': {},
            'coherence_matrix': np.array([[]]),
            'mean_coherence': np.nan,
            'coherence_std': np.nan,
            'max_coherence': np.nan,
            'max_coherence_pair': (None, None),
            'avg_wavelet_entropy': np.nan,
            'avg_temporal_variability': np.nan,
        }

    def to_parquet_row(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert result to flat row for parquet output."""
        return {
            'unit_id': result['unit_id'],
            'n_signals': result['n_signals'],
            'n_samples': result['n_samples'],
            'n_scales': result['n_scales'],
            'mean_coherence': result['mean_coherence'],
            'coherence_std': result['coherence_std'],
            'max_coherence': result['max_coherence'],
            'avg_wavelet_entropy': result['avg_wavelet_entropy'],
            'avg_temporal_variability': result['avg_temporal_variability'],
        }
