"""
PRISM Wavelet Engine
====================
Time-frequency analysis with automatic wavelet selection.

Wavelet families and their uses:
- Morlet: Oscillatory signals (EEG, audio, periodic)
- Mexican Hat: Spike detection, transients
- Haar: Step changes, binary-like data
- Daubechies: General purpose, smooth signals
- Coiflet: Heavy-tailed, smooth signals

Phase: Unbound
Normalization: None (scale-invariant)
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
from datetime import date

import numpy as np
import pandas as pd
from scipy import signal

from orthon._internal.engines.engine_base import BaseEngine
from orthon._internal.engines.metadata import EngineMetadata


logger = logging.getLogger(__name__)


METADATA = EngineMetadata(
    name="wavelet",
    engine_type="vector",
    description="Time-frequency decomposition via wavelet transform",
    domains={"signal_topology", "frequency"},
    requires_window=True,
    deterministic=True,
)


# Try to import pywt for wavelets
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    logger.warning("pywt not installed. Wavelet analysis limited.")


# =============================================================================
# WAVELET SELECTION
# =============================================================================

class WaveletType(Enum):
    """Available wavelet families."""
    MORLET = "morl"          # Oscillatory, complex - good for EEG, periodic
    MEXICAN_HAT = "mexh"     # Spike detection, transients
    HAAR = "haar"            # Step changes, binary-like
    DAUBECHIES_4 = "db4"     # General purpose, compact
    DAUBECHIES_8 = "db8"     # Smoother, longer support
    SYMLET_4 = "sym4"        # Symmetric version of db4
    COIFLET_3 = "coif3"      # Symmetric, smooth


@dataclass
class WaveletProfile:
    """Recommended wavelet configuration based on data characteristics."""
    primary_wavelet: str
    secondary_wavelet: str
    reason: str
    scales: Optional[np.ndarray] = None
    use_dwt: bool = False      # Discrete WT for decomposition
    eeg_bands: bool = False    # Extract EEG frequency bands


# EEG frequency bands (in Hz)
EEG_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100),
}


def select_wavelet(
    values: np.ndarray,
    sampling_rate: Optional[float] = None,
    data_profile: Optional[Dict] = None,
) -> WaveletProfile:
    """
    Auto-select wavelet based on data characteristics.

    Args:
        values: Data array
        sampling_rate: Samples per second (if known)
        data_profile: Optional profile from DataProfiler

    Returns:
        WaveletProfile with recommended configuration
    """
    n = len(values)

    # Extract characteristics from profile or compute
    if data_profile:
        is_periodic = data_profile.get('has_periodicity', False)
        has_jumps = data_profile.get('has_jumps', False)
        is_discrete = data_profile.get('is_discrete', False)
        is_sparse = data_profile.get('is_sparse', False)
        is_heavy_tailed = data_profile.get('is_heavy_tailed', False)
        dominant_period = data_profile.get('dominant_period')
    else:
        # Quick detection
        is_periodic = _detect_periodicity(values)
        has_jumps = _detect_jumps(values)
        is_discrete = len(np.unique(values)) < 20
        is_sparse = np.mean(values == 0) > 0.3
        is_heavy_tailed = _detect_heavy_tails(values)
        dominant_period = None

    # Check for EEG-like data (high sampling rate + oscillatory)
    is_eeg_like = sampling_rate is not None and sampling_rate >= 100 and is_periodic

    # Select wavelet based on characteristics
    if is_eeg_like:
        return WaveletProfile(
            primary_wavelet="morl",
            secondary_wavelet="db4",
            reason="EEG-like: oscillatory signal with high sampling rate",
            eeg_bands=True,
        )

    if is_periodic:
        return WaveletProfile(
            primary_wavelet="morl",
            secondary_wavelet="cmor1.5-1.0",  # Complex Morlet
            reason="Periodic signal - Morlet captures phase and frequency",
        )

    if has_jumps or is_sparse:
        return WaveletProfile(
            primary_wavelet="mexh",
            secondary_wavelet="haar",
            reason="Spiky/sparse data - Mexican Hat detects transients",
        )

    if is_discrete:
        return WaveletProfile(
            primary_wavelet="haar",
            secondary_wavelet="db2",
            reason="Discrete data - Haar captures step changes",
            use_dwt=True,
        )

    if is_heavy_tailed:
        return WaveletProfile(
            primary_wavelet="coif3",
            secondary_wavelet="db4",
            reason="Heavy-tailed - Coiflet handles fat tails",
        )

    # Default: general purpose Daubechies
    return WaveletProfile(
        primary_wavelet="db4",
        secondary_wavelet="morl",
        reason="Continuous data - Daubechies for general decomposition",
        use_dwt=True,
    )


def _detect_periodicity(values: np.ndarray) -> bool:
    """Quick periodicity detection via FFT."""
    if len(values) < 32:
        return False
    try:
        detrended = signal.detrend(values)
        fft = np.fft.fft(detrended)
        power = np.abs(fft[:len(fft)//2]) ** 2

        # Exclude DC and very low frequencies
        power = power[3:]
        if len(power) == 0:
            return False

        mean_power = np.mean(power)
        max_power = np.max(power)

        # Check if there's a clear dominant peak (>10x average)
        # Also check that it's not just noise
        if mean_power == 0:
            return False

        peak_ratio = max_power / mean_power
        return peak_ratio > 10 and max_power > np.var(values)
    except Exception:
        return False


def _detect_jumps(values: np.ndarray) -> bool:
    """Detect sudden jumps in data."""
    if len(values) < 20:
        return False
    returns = np.diff(values)
    std = np.std(returns)
    if std == 0:
        return False
    jumps = np.abs(returns) > 4 * std
    return np.mean(jumps) > 0.01


def _detect_heavy_tails(values: np.ndarray) -> bool:
    """Detect heavy-tailed distribution."""
    if len(values) < 20:
        return False
    from scipy import stats
    kurtosis = stats.kurtosis(values)
    return kurtosis > 3


# =============================================================================
# WAVELET COMPUTATION
# =============================================================================

def compute_wavelets(
    values: np.ndarray,
    wavelet: Optional[str] = None,
    sampling_rate: Optional[float] = None,
    data_profile: Optional[Dict] = None,
) -> dict:
    """
    Compute wavelet properties with automatic wavelet selection.

    Args:
        values: Array of observed values
        wavelet: Override wavelet selection (None = auto)
        sampling_rate: Samples per second (for EEG band extraction)
        data_profile: Profile from DataProfiler (optional)

    Returns:
        Dict of metric_name -> metric_value
    """
    n = len(values)
    if n < 32:
        return {}

    # Auto-select wavelet if not specified
    if wavelet is None:
        profile = select_wavelet(values, sampling_rate, data_profile)
        wavelet = profile.primary_wavelet
        use_eeg_bands = profile.eeg_bands
        use_dwt = profile.use_dwt
        wavelet_reason = profile.reason
    else:
        use_eeg_bands = False
        use_dwt = False
        wavelet_reason = "user-specified"

    results = {
        'wavelet_used': wavelet,
        'wavelet_reason': wavelet_reason,
    }

    try:
        # Define scales (periods from 4 to N/4)
        max_scale = n // 4
        scales = np.geomspace(4, max(max_scale, 5), min(15, max_scale)).astype(int)

        # Compute continuous wavelet transform
        if HAS_PYWT:
            cwt, freqs = pywt.cwt(values, scales, wavelet)
        else:
            # Fallback to scipy's cwt with Ricker wavelet
            cwt = signal.cwt(values, signal.ricker, scales)
            freqs = None

        # Power at each scale (time-averaged)
        power = np.mean(np.abs(cwt) ** 2, axis=1)

        # Normalize to get energy distribution
        total_power = np.sum(power)
        if total_power == 0:
            return results

        power_dist = power / total_power

        # Find dominant scale
        dominant_idx = np.argmax(power)
        dominant_scale = int(scales[dominant_idx])

        # Compute scale entropy (how spread out the energy is)
        power_dist_clean = power_dist[power_dist > 0]
        scale_entropy = float(-np.sum(power_dist_clean * np.log2(power_dist_clean)))

        # Energy in short/mid/long scales
        n_scales = len(scales)
        short_energy = float(np.sum(power_dist[:n_scales // 3]))
        mid_energy = float(np.sum(power_dist[n_scales // 3:2 * n_scales // 3]))
        long_energy = float(np.sum(power_dist[2 * n_scales // 3:]))

        results.update({
            'dominant_scale': float(dominant_scale),
            'scale_entropy': scale_entropy,
            'short_scale_energy': short_energy,
            'mid_scale_energy': mid_energy,
            'long_scale_energy': long_energy,
        })

        # EEG band extraction
        if use_eeg_bands and sampling_rate is not None and HAS_PYWT:
            band_powers = extract_eeg_bands(values, sampling_rate)
            results.update(band_powers)

        # Discrete wavelet transform (DWT) for decomposition
        if use_dwt and HAS_PYWT:
            dwt_results = compute_dwt_decomposition(values, wavelet)
            results.update(dwt_results)

    except Exception as e:
        logger.debug(f"Wavelet computation failed: {e}")

    return results


def compute_wavelets_with_derivation(
    values: np.ndarray,
    signal_id: str = "unknown",
    window_id: str = "0",
    window_start: str = None,
    window_end: str = None,
    wavelet: Optional[str] = None,
    sampling_rate: Optional[float] = None,
) -> tuple:
    """
    Compute wavelet decomposition with full mathematical derivation.

    Returns:
        tuple: (result_dict, Derivation object)
    """
    from orthon._internal.entry_points.derivations.base import Derivation

    deriv = Derivation(
        engine_name="wavelet",
        method_name="Continuous Wavelet Transform (CWT)",
        signal_id=signal_id,
        window_id=window_id,
        window_start=window_start,
        window_end=window_end,
        sample_size=len(values),
        raw_data_sample=values[:10].tolist() if len(values) >= 10 else values.tolist(),
    )

    n = len(values)
    if n < 32:
        deriv.final_result = None
        deriv.interpretation = "Insufficient data (n < 32)"
        return {}, deriv

    # Step 1: Input and wavelet selection
    deriv.add_step(
        title="Input Data Summary",
        equation="X = {x₁, x₂, ..., xₙ}",
        calculation=f"n = {n}\nRange: [{np.min(values):.4f}, {np.max(values):.4f}]\nMean: {np.mean(values):.4f}\nStd: {np.std(values):.4f}",
        result=n,
        result_name="n",
        notes="Signal for time-frequency analysis"
    )

    # Step 2: Auto-select wavelet
    if wavelet is None:
        profile = select_wavelet(values, sampling_rate)
        wavelet = profile.primary_wavelet
        wavelet_reason = profile.reason
    else:
        wavelet_reason = "user-specified"

    deriv.parameters['wavelet'] = wavelet
    deriv.parameters['wavelet_reason'] = wavelet_reason

    deriv.add_step(
        title="Wavelet Selection",
        equation="ψ(t) = chosen mother wavelet",
        calculation=f"Selected wavelet: {wavelet}\nReason: {wavelet_reason}\n\nWavelet families:\n- morl: Morlet (oscillatory)\n- mexh: Mexican Hat (transients)\n- db4: Daubechies-4 (general)",
        result=wavelet,
        result_name="ψ",
        notes="Mother wavelet determines frequency resolution vs time resolution trade-off"
    )

    # Step 3: Generate scales
    max_scale = n // 4
    scales = np.geomspace(4, max(max_scale, 5), min(15, max_scale)).astype(int)

    deriv.add_step(
        title="Define Analysis Scales",
        equation="scales = {s₁, s₂, ..., sₖ} (geometric spacing)",
        calculation=f"Min scale: 4\nMax scale: {max_scale}\nNumber of scales: {len(scales)}\nScales: {scales.tolist()}",
        result=len(scales),
        result_name="n_scales",
        notes="Scales relate to pseudo-frequencies: larger scale → lower frequency"
    )

    # Step 4: Compute CWT
    if HAS_PYWT:
        cwt_coefs, freqs = pywt.cwt(values, scales, wavelet)
        deriv.add_step(
            title="Continuous Wavelet Transform",
            equation="W(s,τ) = ∫ x(t) · (1/√s) · ψ*((t-τ)/s) dt",
            calculation=f"CWT coefficient matrix shape: {cwt_coefs.shape}\n  {len(scales)} scales × {n} time points\n\nPseudo-frequencies (Hz):\n  Scale {scales[0]}: {freqs[0]:.4f} Hz\n  Scale {scales[-1]}: {freqs[-1]:.4f} Hz",
            result=cwt_coefs.shape[0],
            result_name="W",
            notes="W(s,τ) measures similarity between signal and scaled/shifted wavelet"
        )
    else:
        cwt_coefs = signal.cwt(values, signal.ricker, scales)
        freqs = None
        deriv.add_step(
            title="Continuous Wavelet Transform (scipy fallback)",
            equation="W(s,τ) = ∫ x(t) · (1/√s) · ψ*((t-τ)/s) dt",
            calculation=f"CWT coefficient matrix shape: {cwt_coefs.shape}\nUsing Ricker wavelet (Mexican Hat)",
            result=cwt_coefs.shape[0],
            result_name="W",
            notes="pywt not installed, using scipy Ricker wavelet"
        )

    # Step 5: Power spectrum
    power = np.mean(np.abs(cwt_coefs) ** 2, axis=1)
    total_power = np.sum(power)

    if total_power == 0:
        deriv.interpretation = "Zero power - constant signal"
        return {'wavelet_used': wavelet}, deriv

    power_dist = power / total_power

    deriv.add_step(
        title="Compute Wavelet Power Spectrum",
        equation="P(s) = ⟨|W(s,τ)|²⟩_τ  (time-averaged power at each scale)",
        calculation=f"Power at each scale (normalized):\n" + "\n".join([
            f"  Scale {scales[i]}: {power_dist[i]:.4f}" for i in range(min(5, len(scales)))
        ]) + f"\n  ...\nTotal power: {total_power:.4f}",
        result=power_dist,
        result_name="P(s)",
        notes="Power spectrum shows energy distribution across scales"
    )

    # Step 6: Dominant scale
    dominant_idx = np.argmax(power)
    dominant_scale = int(scales[dominant_idx])

    deriv.add_step(
        title="Identify Dominant Scale",
        equation="s* = argmax_s P(s)",
        calculation=f"Maximum power at scale index {dominant_idx}\nDominant scale s* = {dominant_scale}\nPower at dominant scale: {power_dist[dominant_idx]:.4f}",
        result=dominant_scale,
        result_name="s*",
        notes="Dominant scale corresponds to the most energetic frequency component"
    )

    # Step 7: Scale entropy
    power_dist_clean = power_dist[power_dist > 0]
    scale_entropy = float(-np.sum(power_dist_clean * np.log2(power_dist_clean)))

    deriv.add_step(
        title="Compute Scale Entropy",
        equation="H = -Σₛ P(s) · log₂(P(s))",
        calculation=f"Entropy calculation:\nH = -[{power_dist_clean[0]:.4f}×log₂({power_dist_clean[0]:.4f}) + ...]\nH = {scale_entropy:.4f}",
        result=scale_entropy,
        result_name="H_scale",
        notes="High entropy = energy spread across scales; Low = concentrated at dominant scale"
    )

    # Step 8: Energy distribution by scale groups
    n_scales = len(scales)
    short_energy = float(np.sum(power_dist[:n_scales // 3]))
    mid_energy = float(np.sum(power_dist[n_scales // 3:2 * n_scales // 3]))
    long_energy = float(np.sum(power_dist[2 * n_scales // 3:]))

    deriv.add_step(
        title="Energy Distribution by Scale Group",
        equation="E_short + E_mid + E_long = 1",
        calculation=f"Short scales (high freq): {short_energy:.4f}\nMid scales: {mid_energy:.4f}\nLong scales (low freq): {long_energy:.4f}\n\nSum check: {short_energy + mid_energy + long_energy:.4f}",
        result=[short_energy, mid_energy, long_energy],
        result_name="E_groups",
        notes="Shows whether signal energy is in fast (short) or slow (long) variations"
    )

    # Final result
    result = {
        'wavelet_used': wavelet,
        'wavelet_reason': wavelet_reason,
        'dominant_scale': float(dominant_scale),
        'scale_entropy': scale_entropy,
        'short_scale_energy': short_energy,
        'mid_scale_energy': mid_energy,
        'long_scale_energy': long_energy,
    }

    deriv.final_result = scale_entropy
    deriv.prism_output = scale_entropy

    # Interpretation
    if scale_entropy > 3.0:
        interp = f"Scale entropy = {scale_entropy:.3f} indicates **broadband** signal with energy spread across scales."
    elif scale_entropy < 1.5:
        interp = f"Scale entropy = {scale_entropy:.3f} indicates **narrowband** signal concentrated at dominant scale {dominant_scale}."
    else:
        interp = f"Scale entropy = {scale_entropy:.3f} indicates **moderate** scale diversity."

    if long_energy > 0.6:
        interp += " Energy concentrated in **long scales** (slow variations/trends)."
    elif short_energy > 0.6:
        interp += " Energy concentrated in **short scales** (fast variations/noise)."

    deriv.interpretation = interp

    return result, deriv


def extract_eeg_bands(
    values: np.ndarray,
    sampling_rate: float,
    wavelet: str = 'db4',
) -> Dict[str, float]:
    """
    Extract EEG frequency band powers using wavelet decomposition.

    Args:
        values: EEG signal
        sampling_rate: Samples per second (e.g., 256 Hz)
        wavelet: Wavelet for decomposition

    Returns:
        Dict with band powers (delta, theta, alpha, beta, gamma)
    """
    if not HAS_PYWT:
        return {}

    try:
        # Determine max decomposition level
        max_level = pywt.dwt_max_level(len(values), wavelet)
        levels = min(max_level, 8)  # Cap at 8 levels

        # Decompose
        coeffs = pywt.wavedec(values, wavelet, level=levels)

        # Frequency range per level: fs/2^(level+1) to fs/2^level
        results = {}
        total_power = 0
        band_powers = {}

        for level in range(1, levels + 1):
            if level >= len(coeffs):
                continue

            freq_low = sampling_rate / (2 ** (level + 1))
            freq_high = sampling_rate / (2 ** level)

            # Power at this level
            detail = coeffs[level]
            power = np.sum(detail ** 2)
            total_power += power

            # Map to EEG bands
            for band_name, (band_low, band_high) in EEG_BANDS.items():
                if freq_low <= band_high and freq_high >= band_low:
                    band_powers[band_name] = band_powers.get(band_name, 0) + power

        # Normalize
        if total_power > 0:
            for band_name in EEG_BANDS:
                power = band_powers.get(band_name, 0)
                results[f'eeg_{band_name}_power'] = float(power / total_power)
                results[f'eeg_{band_name}_raw'] = float(power)

        # Ratios (common EEG metrics)
        delta = band_powers.get('delta', 0)
        theta = band_powers.get('theta', 0)
        alpha = band_powers.get('alpha', 0)
        beta = band_powers.get('beta', 0)

        if alpha > 0:
            results['eeg_theta_alpha_ratio'] = float(theta / alpha)
        if beta > 0:
            results['eeg_alpha_beta_ratio'] = float(alpha / beta)
        if (alpha + beta) > 0:
            results['eeg_delta_alphabeta_ratio'] = float(delta / (alpha + beta))

        return results

    except Exception as e:
        logger.debug(f"EEG band extraction failed: {e}")
        return {}


def compute_dwt_decomposition(
    values: np.ndarray,
    wavelet: str = 'db4',
) -> Dict[str, float]:
    """
    Compute discrete wavelet transform decomposition metrics.

    Returns energy distribution across approximation and detail levels.
    """
    if not HAS_PYWT:
        return {}

    try:
        max_level = pywt.dwt_max_level(len(values), wavelet)
        levels = min(max_level, 6)

        coeffs = pywt.wavedec(values, wavelet, level=levels)

        # Energy at each level
        energies = [np.sum(c ** 2) for c in coeffs]
        total_energy = sum(energies)

        if total_energy == 0:
            return {}

        results = {
            'dwt_levels': levels,
            'dwt_approx_energy': float(energies[0] / total_energy),
        }

        # Detail energies by level
        for i, energy in enumerate(energies[1:], 1):
            results[f'dwt_detail_{i}_energy'] = float(energy / total_energy)

        # Summary: low-freq vs high-freq energy
        low_freq = sum(energies[:len(energies)//2])
        high_freq = sum(energies[len(energies)//2:])
        results['dwt_low_freq_ratio'] = float(low_freq / total_energy)
        results['dwt_high_freq_ratio'] = float(high_freq / total_energy)

        return results

    except Exception as e:
        logger.debug(f"DWT decomposition failed: {e}")
        return {}


def compute_wavelet_coherence(
    x: np.ndarray,
    y: np.ndarray,
    wavelet: str = 'morl',
    scales: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute wavelet coherence between two series.

    Measures time-frequency synchronization.
    """
    n = min(len(x), len(y))
    if n < 32:
        return {}

    if scales is None:
        max_scale = n // 4
        scales = np.geomspace(4, max(max_scale, 5), min(15, max_scale)).astype(int)

    try:
        if HAS_PYWT:
            cwtx, _ = pywt.cwt(x[:n], scales, wavelet)
            cwty, _ = pywt.cwt(y[:n], scales, wavelet)
        else:
            cwtx = signal.cwt(x[:n], signal.ricker, scales)
            cwty = signal.cwt(y[:n], signal.ricker, scales)

        # Cross-wavelet spectrum
        Wxy = cwtx * np.conj(cwty)

        # Individual power spectra
        Wxx = np.abs(cwtx) ** 2
        Wyy = np.abs(cwty) ** 2

        # Smoothed coherence
        coherence = np.abs(np.mean(Wxy, axis=1)) ** 2 / (
            np.mean(Wxx, axis=1) * np.mean(Wyy, axis=1) + 1e-10
        )

        # Phase (lead/lag)
        phase = np.angle(np.mean(Wxy, axis=1))

        # Summarize by scale groups
        n_scales = len(scales)
        short_idx = slice(0, n_scales // 3)
        mid_idx = slice(n_scales // 3, 2 * n_scales // 3)
        long_idx = slice(2 * n_scales // 3, n_scales)

        return {
            'short_term_coherence': float(np.mean(coherence[short_idx])),
            'mid_term_coherence': float(np.mean(coherence[mid_idx])),
            'long_term_coherence': float(np.mean(coherence[long_idx])),
            'overall_coherence': float(np.mean(coherence)),
            'dominant_coherence_scale': int(scales[np.argmax(coherence)]),
            'short_term_phase': float(np.mean(phase[short_idx])),
            'long_term_phase': float(np.mean(phase[long_idx])),
        }

    except Exception as e:
        logger.debug(f"Wavelet coherence failed: {e}")
        return {}


# =============================================================================
# Legacy Class Interface (for backwards compatibility)
# =============================================================================


class WaveletEngine(BaseEngine):
    """
    Wavelet Coherence engine for time-frequency analysis.

    Captures co-movement at different scales (short-term to long-term).
    Now with automatic wavelet selection based on data characteristics.

    Outputs:
        - results.wavelet_coherence: Coherence summaries
    """

    name = "wavelet"
    phase = "derived"
    default_normalization = None  # Scale-invariant

    @property
    def metadata(self) -> EngineMetadata:
        return METADATA

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        wavelet: str = None,  # None = auto-select
        scales: Optional[np.ndarray] = None,
        sampling_rate: Optional[float] = None,
        **params
    ) -> Dict[str, Any]:
        """
        Run wavelet coherence analysis.

        Args:
            df: Signal data
            run_id: Unique run identifier
            wavelet: Wavelet type (None = auto-select)
            scales: Wavelet scales (default: logarithmic range)
            sampling_rate: Sampling rate in Hz (for EEG band extraction)

        Returns:
            Dict with summary metrics
        """
        df_clean = df
        signals = list(df_clean.columns)
        n = len(signals)
        n_samples = len(df_clean)

        if n_samples < 128:
            logger.warning(
                f"Wavelet analysis works best with 128+ samples. Got {n_samples}."
            )

        window_start = df_clean.index.min().date()
        window_end = df_clean.index.max().date()

        # Default scales (periods from 4 to N/4)
        if scales is None:
            max_scale = n_samples // 4
            scales = np.geomspace(4, max_scale, 20).astype(int)

        # Compute pairwise coherence summaries
        results = []
        wavelets_used = set()

        for i, ind1 in enumerate(signals):
            for j, ind2 in enumerate(signals):
                if i >= j:
                    continue

                x = df_clean[ind1].values
                y = df_clean[ind2].values

                # Auto-select wavelet if needed
                if wavelet is None:
                    profile = select_wavelet(x, sampling_rate)
                    use_wavelet = profile.primary_wavelet
                    wavelets_used.add(use_wavelet)
                else:
                    use_wavelet = wavelet
                    wavelets_used.add(wavelet)

                # Compute wavelet coherence
                coherence_summary = compute_wavelet_coherence(
                    x, y, use_wavelet, scales
                )

                coherence_summary['wavelet_used'] = use_wavelet

                results.append({
                    "signal_id_1": ind1,
                    "signal_id_2": ind2,
                    "window_start": window_start,
                    "window_end": window_end,
                    **coherence_summary,
                    "run_id": run_id,
                })

        # Store results
        if results:
            self._store_coherence(pd.DataFrame(results), run_id)

        # Summary metrics
        df_results = pd.DataFrame(results)

        metrics = {
            "n_signals": n,
            "n_pairs": len(results),
            "n_samples": n_samples,
            "n_scales": len(scales),
            "wavelets_used": list(wavelets_used),
            "avg_short_term_coherence": float(df_results["short_term_coherence"].mean()) if results else 0,
            "avg_long_term_coherence": float(df_results["long_term_coherence"].mean()) if results else 0,
            "has_pywt": HAS_PYWT,
        }

        logger.info(
            f"Wavelet analysis complete: {metrics['n_pairs']} pairs, "
            f"wavelets={metrics['wavelets_used']}, "
            f"short-term coh={metrics['avg_short_term_coherence']:.3f}"
        )

        return metrics

    def _store_coherence(self, df: pd.DataFrame, run_id: str):
        """Store wavelet coherence results."""
        # Would need table: results.wavelet_coherence
        pass
