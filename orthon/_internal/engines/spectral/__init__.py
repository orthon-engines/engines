"""
PRISM Spectral Engines — Frequency domain analysis.

Wavelet microscope for multi-scale degradation detection.
"""

from orthon._internal.engines.spectral.wavelet import (
    WaveletBand,
    compute_wavelet_decomposition,
    compute_band_snr_evolution,
    identify_degradation_band,
    run_wavelet_microscope,
    extract_wavelet_features,
)

__all__ = [
    'WaveletBand',
    'compute_wavelet_decomposition',
    'compute_band_snr_evolution',
    'identify_degradation_band',
    'run_wavelet_microscope',
    'extract_wavelet_features',
]
