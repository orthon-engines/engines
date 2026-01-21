"""
PRISM Spectral Features Engine

Comprehensive frequency domain analysis.

Tier 1 (Always-On):
- FFT-based spectral features
- Spectral shape metrics (centroid, bandwidth, rolloff, flatness)
- Spectral slope (1/f^β characteristics)
- Band power ratios

Tier 2 (Triggered - see wavelet_microscope.py):
- Full time-frequency decomposition
- Transient detection

Compute cost: O(n log n) - essentially free compared to RQA O(n²)

Phase: Unbound
Normalization: Detrend applied internally
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from datetime import date
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import signal, fft

from orthon._internal.engines.engine_base import BaseEngine
from orthon._internal.engines.metadata import EngineMetadata
from orthon._internal.entry_points.derivations.base import Derivation


logger = logging.getLogger(__name__)


# =============================================================================
# Spectral Features Dataclass
# =============================================================================

@dataclass
class SpectralFeatures:
    """FFT-based features computed on every window."""

    # Frequency content
    dominant_freq: Optional[float]           # Frequency with max power
    dominant_freq_power: Optional[float]     # Power at dominant frequency
    secondary_freq: Optional[float]          # Second highest peak
    dominant_period: Optional[float]         # 1/dominant_freq

    # Spectral shape
    spectral_centroid: Optional[float]       # Center of mass of spectrum
    spectral_bandwidth: Optional[float]      # Spread around centroid
    spectral_rolloff: Optional[float]        # Frequency below which 85% of power
    spectral_flatness: Optional[float]       # Wiener entropy (noise vs tonal)

    # Spectral slope (1/f^β characteristics)
    spectral_slope: Optional[float]          # β exponent from log-log fit
    spectral_slope_r2: Optional[float]       # Fit quality

    # Band power ratios
    low_freq_power: Optional[float]          # Power in low band (0-0.1 normalized)
    mid_freq_power: Optional[float]          # Power in mid band (0.1-0.3)
    high_freq_power: Optional[float]         # Power in high band (0.3-0.5)
    low_high_ratio: Optional[float]          # Low/high band ratio

    # Spectral entropy
    spectral_entropy: Optional[float]        # Shannon entropy of spectrum

    # Peak statistics
    n_significant_peaks: Optional[int]       # Number of peaks above threshold
    total_power: Optional[float]             # Total spectral power

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return asdict(self)


# =============================================================================
# Default frequency band configuration
# =============================================================================

DEFAULT_BANDS = {
    'low': (0.0, 0.1),    # Low frequency: 0-10% of Nyquist
    'mid': (0.1, 0.3),    # Mid frequency: 10-30% of Nyquist
    'high': (0.3, 0.5),   # High frequency: 30-50% of Nyquist
}


METADATA = EngineMetadata(
    name="spectral",
    engine_type="vector",
    description="FFT-based spectral features: shape, slope, bands, entropy",
    domains={"signal_topology", "frequency"},
    requires_window=True,
    deterministic=True,
)


# =============================================================================
# Helper Functions
# =============================================================================

def _find_spectral_peaks(psd: np.ndarray, min_prominence: float = 0.1) -> np.ndarray:
    """Find spectral peaks sorted by power descending."""
    threshold = min_prominence * np.max(psd)
    peaks, _ = signal.find_peaks(psd, height=threshold)
    if len(peaks) == 0:
        return np.array([np.argmax(psd)])
    # Sort by power descending
    return peaks[np.argsort(psd[peaks])[::-1]]


def _compute_spectral_slope(freqs: np.ndarray, psd: np.ndarray) -> Tuple[float, float]:
    """
    Fit power spectrum to 1/f^β in log-log space.

    Returns:
        (slope, r2) - β exponent and fit quality
    """
    # Avoid log(0)
    mask = (freqs > 0) & (psd > 0)
    if mask.sum() < 3:
        return 0.0, 0.0

    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask])

    # Linear regression
    slope, intercept = np.polyfit(log_f, log_p, 1)

    # R² for fit quality
    predicted = slope * log_f + intercept
    ss_res = np.sum((log_p - predicted) ** 2)
    ss_tot = np.sum((log_p - np.mean(log_p)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))

    return float(slope), float(max(0, r2))


def _compute_band_power(
    freqs: np.ndarray,
    psd_norm: np.ndarray,
    band: Tuple[float, float]
) -> float:
    """Compute normalized power in frequency band (fraction of Nyquist)."""
    nyquist = freqs[-1] if len(freqs) > 0 else 0.5
    low = band[0] * nyquist
    high = band[1] * nyquist
    mask = (freqs >= low) & (freqs <= high)
    return float(psd_norm[mask].sum()) if mask.any() else 0.0


# =============================================================================
# Vector Engine Contract: Full Spectral Features
# =============================================================================

def compute_spectral_features(
    values: np.ndarray,
    method: str = "welch",
    bands: dict = None
) -> SpectralFeatures:
    """
    Compute comprehensive spectral features for a single signal.

    Args:
        values: Array of observed values
        method: 'welch' or 'periodogram'
        bands: Custom frequency bands dict, or use DEFAULT_BANDS

    Returns:
        SpectralFeatures dataclass with all metrics
    """
    if bands is None:
        bands = DEFAULT_BANDS

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)

    # Null result for insufficient data
    if n < 16:
        return SpectralFeatures(
            dominant_freq=None, dominant_freq_power=None, secondary_freq=None,
            dominant_period=None, spectral_centroid=None, spectral_bandwidth=None,
            spectral_rolloff=None, spectral_flatness=None, spectral_slope=None,
            spectral_slope_r2=None, low_freq_power=None, mid_freq_power=None,
            high_freq_power=None, low_high_ratio=None, spectral_entropy=None,
            n_significant_peaks=None, total_power=None
        )

    # Detrend to avoid DC leakage
    values_detrend = signal.detrend(values)

    # Compute PSD
    if method == "welch":
        freqs, psd = signal.welch(values_detrend, fs=1.0, nperseg=min(n // 2, 256))
    else:
        freqs, psd = signal.periodogram(values_detrend, fs=1.0)

    # Skip DC component
    if len(freqs) > 1:
        freqs = freqs[1:]
        psd = psd[1:]

    if len(psd) == 0:
        return SpectralFeatures(
            dominant_freq=None, dominant_freq_power=None, secondary_freq=None,
            dominant_period=None, spectral_centroid=None, spectral_bandwidth=None,
            spectral_rolloff=None, spectral_flatness=None, spectral_slope=None,
            spectral_slope_r2=None, low_freq_power=None, mid_freq_power=None,
            high_freq_power=None, low_high_ratio=None, spectral_entropy=None,
            n_significant_peaks=None, total_power=None
        )

    # Normalize power to probability distribution
    total_power = psd.sum()
    psd_norm = psd / (total_power + 1e-10)

    # --- Dominant frequencies ---
    peaks = _find_spectral_peaks(psd)
    dominant_idx = peaks[0]
    secondary_idx = peaks[1] if len(peaks) > 1 else dominant_idx

    dominant_freq = float(freqs[dominant_idx])
    dominant_freq_power = float(psd_norm[dominant_idx])
    secondary_freq = float(freqs[secondary_idx])
    dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else None

    # --- Spectral centroid (center of mass) ---
    spectral_centroid = float(np.sum(freqs * psd_norm))

    # --- Spectral bandwidth (spread around centroid) ---
    spectral_bandwidth = float(np.sqrt(
        np.sum(((freqs - spectral_centroid) ** 2) * psd_norm)
    ))

    # --- Spectral rolloff (85% cumulative power) ---
    cumsum = np.cumsum(psd_norm)
    rolloff_idx = np.searchsorted(cumsum, 0.85)
    spectral_rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    # --- Spectral flatness (Wiener entropy: geometric/arithmetic mean) ---
    log_psd = np.log(psd + 1e-10)
    geometric_mean = np.exp(np.mean(log_psd))
    arithmetic_mean = np.mean(psd)
    spectral_flatness = float(geometric_mean / (arithmetic_mean + 1e-10))

    # --- Spectral slope (1/f^β) ---
    spectral_slope, spectral_slope_r2 = _compute_spectral_slope(freqs, psd)

    # --- Band powers ---
    low_freq_power = _compute_band_power(freqs, psd_norm, bands['low'])
    mid_freq_power = _compute_band_power(freqs, psd_norm, bands['mid'])
    high_freq_power = _compute_band_power(freqs, psd_norm, bands['high'])
    low_high_ratio = float(low_freq_power / (high_freq_power + 1e-10))

    # --- Spectral entropy (Shannon entropy) ---
    spectral_entropy = float(-np.sum(psd_norm * np.log(psd_norm + 1e-10)))

    return SpectralFeatures(
        dominant_freq=dominant_freq,
        dominant_freq_power=dominant_freq_power,
        secondary_freq=secondary_freq,
        dominant_period=dominant_period,
        spectral_centroid=spectral_centroid,
        spectral_bandwidth=spectral_bandwidth,
        spectral_rolloff=spectral_rolloff,
        spectral_flatness=spectral_flatness,
        spectral_slope=spectral_slope,
        spectral_slope_r2=spectral_slope_r2,
        low_freq_power=low_freq_power,
        mid_freq_power=mid_freq_power,
        high_freq_power=high_freq_power,
        low_high_ratio=low_high_ratio,
        spectral_entropy=spectral_entropy,
        n_significant_peaks=len(peaks),
        total_power=float(total_power)
    )


def compute_spectral(values: np.ndarray, method: str = "welch") -> dict:
    """
    Compute spectral metrics for a single signal.

    Backward-compatible wrapper around compute_spectral_features().

    Args:
        values: Array of observed values
        method: 'welch' or 'periodogram'

    Returns:
        Dict of spectral metrics (full feature set)
    """
    features = compute_spectral_features(values, method)
    return features.to_dict()


def compute_spectral_entropy_with_derivation(
    values: np.ndarray,
    method: str = "welch",
    signal_id: str = "unknown",
    window_id: str = "unknown",
    window_start: str = "",
    window_end: str = "",
) -> tuple:
    """
    Compute spectral entropy with full mathematical derivation.

    Returns:
        tuple: (result_dict, Derivation object)
    """
    from datetime import datetime

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)

    # Initialize derivation
    derivation = Derivation(
        engine_name="spectral_entropy",
        method_name="Welch's Power Spectral Density",
        signal_id=signal_id,
        window_id=window_id,
        window_start=window_start,
        window_end=window_end,
        sample_size=n,
        generated_at=datetime.now(),
    )

    derivation.purpose = (
        "Spectral entropy measures the flatness/uniformity of the power spectrum. "
        "A white noise signal has maximum spectral entropy (power spread across all frequencies), "
        "while a pure sinusoid has minimum entropy (power concentrated at one frequency)."
    )

    derivation.definition = """
Spectral entropy is the Shannon entropy of the normalized power spectral density:

```
H_s = -∑ P(f) · log(P(f))
```

Where:
- P(f) = normalized power spectral density (sums to 1)
- log = natural logarithm

### Interpretation

| Value Range | Interpretation |
|-------------|----------------|
| Low H_s | Periodic, predictable signal (peaked spectrum) |
| High H_s | Noisy, unpredictable signal (flat spectrum) |
| Max H_s = log(N_f) | Perfect white noise |
"""

    # Store raw data sample
    derivation.raw_data_sample = values[:10].tolist()

    # Step 1: Input data summary
    derivation.add_step(
        title="Input Data Summary",
        equation="X = {x₁, x₂, ..., xₙ}",
        calculation=f"n = {n}\nRange: [{values.min():.4f}, {values.max():.4f}]\nMean: {values.mean():.4f}\nStd: {values.std():.4f}",
        result=n,
        result_name="n",
        notes="Input signal topology for spectral analysis"
    )

    # Step 2: Detrend
    values_detrended = signal.detrend(values)
    derivation.add_step(
        title="Remove Linear Trend",
        equation="y_i = x_i - (a + b·i)  where a,b fit by least squares",
        calculation=f"Original mean: {values.mean():.4f}\nDetrended mean: {values_detrended.mean():.6f} (≈ 0)\nDetrended range: [{values_detrended.min():.4f}, {values_detrended.max():.4f}]",
        result=values_detrended.mean(),
        result_name="detrended_mean",
        notes="Detrending removes low-frequency artifacts from trend"
    )

    # Step 3: Welch's method parameters
    nperseg = min(len(values_detrended) // 2, 256)
    derivation.add_step(
        title="Set Welch's Method Parameters",
        equation="nperseg = min(n/2, 256), overlap = nperseg/2, window = Hann",
        calculation=f"n = {n}\nnperseg = min({n}//2, 256) = {nperseg}\noverlap = {nperseg}//2 = {nperseg // 2}\nn_segments ≈ {2 * n // nperseg - 1}",
        result=nperseg,
        result_name="nperseg",
        notes="Welch's method averages periodograms from overlapping segments"
    )

    # Step 4: Compute PSD
    freqs, psd = signal.welch(values_detrended, fs=1.0, nperseg=nperseg)

    # Skip DC component
    if len(freqs) > 1:
        freqs = freqs[1:]
        psd = psd[1:]

    derivation.add_step(
        title="Compute Power Spectral Density (Welch)",
        equation="For each segment k:\n1. Apply Hann window: w_k[n] = 0.5(1 - cos(2πn/N))·x_k[n]\n2. Compute FFT: X_k(f) = FFT(w_k)\n3. Periodogram: P_k(f) = |X_k(f)|²/N\n\nAverage: PSD(f) = (1/K)∑_k P_k(f)",
        calculation=f"Frequency bins: {len(freqs)}\nFrequency range: [{freqs[0]:.4f}, {freqs[-1]:.4f}]\nTotal power: {psd.sum():.4f}\nPeak power at f = {freqs[np.argmax(psd)]:.4f}",
        result=psd.sum(),
        result_name="total_power",
        notes="Welch's method reduces variance compared to single periodogram"
    )

    # Step 5: Normalize PSD to probability distribution
    psd_norm = psd / (psd.sum() + 1e-10)

    derivation.add_step(
        title="Normalize PSD to Probability Distribution",
        equation="P(f) = PSD(f) / ∑_f PSD(f)",
        calculation=f"∑ PSD(f) = {psd.sum():.4f}\nP(f₁) = {psd[0]:.6f} / {psd.sum():.4f} = {psd_norm[0]:.6f}\nP(f₂) = {psd[1]:.6f} / {psd.sum():.4f} = {psd_norm[1]:.6f}\n...\n∑ P(f) = {psd_norm.sum():.6f} (= 1.0)",
        result=psd_norm.sum(),
        result_name="sum_P",
        notes="Normalization ensures P(f) can be interpreted as probability"
    )

    # Step 6: Compute spectral entropy
    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
    max_entropy = np.log(len(freqs))

    derivation.add_step(
        title="Compute Shannon Entropy of Spectrum",
        equation="H_s = -∑_f P(f) · log(P(f))",
        calculation=f"H_s = -[{psd_norm[0]:.6f}·log({psd_norm[0]:.6f}) + {psd_norm[1]:.6f}·log({psd_norm[1]:.6f}) + ...]\n    = {spectral_entropy:.6f}\n\nMaximum possible (flat spectrum): H_max = log({len(freqs)}) = {max_entropy:.4f}\nNormalized entropy: H_s/H_max = {spectral_entropy/max_entropy:.4f}",
        result=spectral_entropy,
        result_name="H_s",
        notes="Higher entropy indicates more uniform power distribution"
    )

    # Step 7: Find dominant frequency
    peaks_idx, _ = signal.find_peaks(psd, height=np.mean(psd))
    if len(peaks_idx) > 0:
        dominant_idx = peaks_idx[np.argmax(psd[peaks_idx])]
        dominant_freq = freqs[dominant_idx]
        dominant_period = 1 / dominant_freq if dominant_freq > 0 else None
    else:
        dominant_freq = freqs[np.argmax(psd)]
        dominant_period = 1 / dominant_freq if dominant_freq > 0 else None

    derivation.add_step(
        title="Identify Dominant Frequency",
        equation="f_dom = argmax_f PSD(f)  among spectral peaks",
        calculation=f"Number of peaks above mean: {len(peaks_idx)}\nDominant frequency: f = {dominant_freq:.4f}\nDominant period: T = 1/f = {dominant_period:.2f} samples",
        result=dominant_freq,
        result_name="f_dom",
        notes="Dominant frequency indicates strongest periodic component"
    )

    # Set final result
    derivation.final_result = spectral_entropy
    derivation.prism_output = spectral_entropy

    result = {
        "dominant_frequency": float(dominant_freq),
        "dominant_period": float(dominant_period) if dominant_period else None,
        "spectral_entropy": float(spectral_entropy),
        "n_significant_peaks": int(len(peaks_idx)),
        "total_power": float(psd.sum()),
    }

    return result, derivation


class SpectralEngine(BaseEngine):
    """
    Spectral Density engine for cycle detection.

    Identifies dominant periodic patterns in signal topology.

    Outputs:
        - results.geometry_fingerprints: Spectral characteristics
    """

    name = "spectral"
    phase = "derived"
    default_normalization = "diff"  # Detrend via differencing

    @property
    def metadata(self) -> EngineMetadata:
        return METADATA

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        method: str = "welch",
        n_peaks: int = 5,
        **params
    ) -> Dict[str, Any]:
        """
        Run spectral analysis.
        
        Args:
            df: Signal data (preferably detrended)
            run_id: Unique run identifier
            method: 'welch' or 'periodogram'
            n_peaks: Number of spectral peaks to identify
        
        Returns:
            Dict with summary metrics
        """
        df_clean = df
        signals = list(df_clean.columns)
        n_samples = len(df_clean)
        
        window_start = df_clean.index.min().date()
        window_end = df_clean.index.max().date()
        
        results = []
        
        for signal in signals:
            series = df_clean[signal].values
            
            # Compute power spectral density
            if method == "welch":
                freqs, psd = signal.welch(series, nperseg=min(256, n_samples // 4))
            else:
                freqs, psd = signal.periodogram(series)
            
            # Find peaks
            peaks_idx, properties = signal.find_peaks(psd, height=np.median(psd))
            
            # Sort by power
            if len(peaks_idx) > 0:
                peak_powers = psd[peaks_idx]
                sorted_idx = np.argsort(peak_powers)[::-1][:n_peaks]
                top_peaks = peaks_idx[sorted_idx]
                
                dominant_freq = freqs[top_peaks[0]] if len(top_peaks) > 0 else 0
                dominant_period = 1 / dominant_freq if dominant_freq > 0 else np.inf
            else:
                dominant_freq = 0
                dominant_period = np.inf
                top_peaks = []
            
            # Spectral entropy (measure of how spread out the power is)
            psd_norm = psd / (psd.sum() + 1e-10)
            spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
            
            results.append({
                "signal_id": signal,
                "dominant_frequency": float(dominant_freq),
                "dominant_period": float(dominant_period) if not np.isinf(dominant_period) else None,
                "spectral_entropy": float(spectral_entropy),
                "n_significant_peaks": len(top_peaks),
                "total_power": float(psd.sum()),
            })
        
        # Store as geometry fingerprints
        self._store_spectral(results, window_start, window_end, run_id)
        
        # Summary metrics
        df_results = pd.DataFrame(results)
        
        metrics = {
            "n_signals": len(signals),
            "n_samples": n_samples,
            "method": method,
            "avg_spectral_entropy": float(df_results["spectral_entropy"].mean()),
            "avg_n_peaks": float(df_results["n_significant_peaks"].mean()),
        }
        
        logger.info(
            f"Spectral analysis complete: {len(results)} signals, "
            f"avg entropy={metrics['avg_spectral_entropy']:.2f}"
        )
        
        return metrics
    
    def _store_spectral(
        self,
        results: list,
        window_start: date,
        window_end: date,
        run_id: str,
    ):
        """Store spectral characteristics as geometry fingerprints."""
        records = []
        
        for r in results:
            for dim in ["dominant_frequency", "spectral_entropy", "n_significant_peaks"]:
                value = r[dim]
                if value is not None:
                    records.append({
                        "signal_id": r["signal_id"],
                        "window_start": window_start,
                        "window_end": window_end,
                        "dimension": dim,
                        "value": float(value),
                        "run_id": run_id,
                    })
        
        if records:
            df = pd.DataFrame(records)
            self.store_results("geometry_fingerprints", df, run_id)
