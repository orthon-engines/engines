"""
PRISM Spectral Analysis Primitives

Pure mathematical functions for frequency domain analysis.
All functions take numpy arrays and return numbers or arrays.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Tuple, Optional, Union
import warnings


def power_spectral_density(
    values: np.ndarray,
    sample_rate: float = 1.0,
    nperseg: Optional[int] = None,
    method: str = 'welch'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density using Welch's method.

    Args:
        values: Input time series
        sample_rate: Sampling frequency
        nperseg: Length of each segment for Welch's method
        method: Method to use ('welch', 'periodogram', 'multitaper')

    Returns:
        Tuple of (frequencies, power_spectral_density)
    """
    values = np.asarray(values, dtype=np.float64)

    if len(values) < 4:
        # Return minimal PSD for very short signals
        freqs = np.array([0, sample_rate/2])
        psd = np.array([np.var(values), 0])
        return freqs, psd

    if nperseg is None:
        nperseg = min(len(values) // 4, 256)

    if method == 'welch':
        freqs, psd = signal.welch(values, fs=sample_rate, nperseg=nperseg)
    elif method == 'periodogram':
        freqs, psd = signal.periodogram(values, fs=sample_rate)
    else:
        raise ValueError(f"Unknown method: {method}")

    return freqs, psd


def dominant_frequency(
    values: np.ndarray,
    sample_rate: float = 1.0,
    exclude_dc: bool = True
) -> float:
    """
    Find the frequency with maximum power.

    Args:
        values: Input time series
        sample_rate: Sampling frequency
        exclude_dc: If True, ignore the DC component (f=0)

    Returns:
        Dominant frequency in Hz
    """
    freqs, psd = power_spectral_density(values, sample_rate)

    if exclude_dc and len(freqs) > 1:
        start_idx = 1
    else:
        start_idx = 0

    max_idx = np.argmax(psd[start_idx:]) + start_idx
    return float(freqs[max_idx])


def spectral_flatness(
    values: np.ndarray,
    sample_rate: float = 1.0
) -> float:
    """
    Compute spectral flatness (Wiener entropy).

    Spectral flatness is the ratio of geometric mean to arithmetic mean
    of the power spectrum. Values close to 1 indicate noise-like signals,
    values close to 0 indicate tonal signals.

    Args:
        values: Input time series
        sample_rate: Sampling frequency

    Returns:
        Spectral flatness (0 to 1)
    """
    freqs, psd = power_spectral_density(values, sample_rate)

    # Avoid log(0) by adding small epsilon
    psd_positive = psd + 1e-12

    # Geometric mean
    log_mean = np.mean(np.log(psd_positive))
    geometric_mean = np.exp(log_mean)

    # Arithmetic mean
    arithmetic_mean = np.mean(psd)

    if arithmetic_mean == 0:
        return 0.0

    flatness = geometric_mean / arithmetic_mean
    return min(flatness, 1.0)


def spectral_entropy(
    values: np.ndarray,
    sample_rate: float = 1.0,
    normalize: bool = True
) -> float:
    """
    Compute spectral entropy (Shannon entropy of the power spectrum).

    Args:
        values: Input time series
        sample_rate: Sampling frequency
        normalize: If True, normalize entropy to [0,1]

    Returns:
        Spectral entropy
    """
    freqs, psd = power_spectral_density(values, sample_rate)

    # Normalize PSD to create probability distribution
    psd_sum = np.sum(psd)
    if psd_sum == 0:
        return 0.0

    psd_norm = psd / psd_sum

    # Shannon entropy
    psd_nonzero = psd_norm[psd_norm > 0]
    entropy = -np.sum(psd_nonzero * np.log2(psd_nonzero))

    if normalize:
        max_entropy = np.log2(len(psd))
        if max_entropy > 0:
            entropy = entropy / max_entropy

    return float(entropy)


def spectral_centroid(
    values: np.ndarray,
    sample_rate: float = 1.0
) -> float:
    """
    Compute spectral centroid (center of mass of spectrum).

    Args:
        values: Input time series
        sample_rate: Sampling frequency

    Returns:
        Spectral centroid frequency
    """
    freqs, psd = power_spectral_density(values, sample_rate)
    total_power = np.sum(psd)

    if total_power == 0:
        return 0.0

    return float(np.sum(freqs * psd) / total_power)


def spectral_bandwidth(
    values: np.ndarray,
    sample_rate: float = 1.0,
    p: int = 2
) -> float:
    """
    Compute spectral bandwidth (spread around centroid).

    Args:
        values: Input time series
        sample_rate: Sampling frequency
        p: Order of bandwidth (2 for variance-based)

    Returns:
        Spectral bandwidth
    """
    freqs, psd = power_spectral_density(values, sample_rate)
    centroid = spectral_centroid(values, sample_rate)
    total_power = np.sum(psd)

    if total_power == 0:
        return 0.0

    return float((np.sum(np.abs(freqs - centroid)**p * psd) / total_power) ** (1/p))


def fundamental_frequency(
    values: np.ndarray,
    sample_rate: float = 1.0,
    min_freq: float = 20.0,
    max_freq: Optional[float] = None
) -> float:
    """
    Estimate fundamental frequency using autocorrelation method.

    Args:
        values: Input time series
        sample_rate: Sampling frequency
        min_freq: Minimum frequency to consider
        max_freq: Maximum frequency to consider (default: sample_rate/4)

    Returns:
        Fundamental frequency in Hz
    """
    values = np.asarray(values, dtype=np.float64)

    if max_freq is None:
        max_freq = sample_rate / 4

    # Compute autocorrelation
    autocorr = np.correlate(values, values, mode='full')
    autocorr = autocorr[autocorr.size // 2:]

    # Convert frequency limits to lag limits
    min_lag = max(1, int(sample_rate / max_freq))
    max_lag = min(len(autocorr) - 1, int(sample_rate / min_freq))

    if min_lag >= max_lag:
        return min_freq

    # Find peak in autocorrelation within lag range
    search_range = autocorr[min_lag:max_lag]
    peak_idx = np.argmax(search_range) + min_lag

    return float(sample_rate / peak_idx)


def harmonic_content(
    values: np.ndarray,
    sample_rate: float = 1.0,
    fundamental: Optional[float] = None,
    num_harmonics: int = 5
) -> np.ndarray:
    """
    Analyze harmonic content relative to fundamental frequency.

    Args:
        values: Input time series
        sample_rate: Sampling frequency
        fundamental: Fundamental frequency (if None, estimated automatically)
        num_harmonics: Number of harmonics to analyze

    Returns:
        Array of harmonic magnitudes (fundamental + harmonics)
    """
    if fundamental is None:
        fundamental = fundamental_frequency(values, sample_rate)

    freqs, psd = power_spectral_density(values, sample_rate)

    harmonics = []
    for h in range(1, num_harmonics + 1):
        harmonic_freq = h * fundamental

        if harmonic_freq > freqs[-1]:
            harmonics.append(0.0)
        else:
            idx = np.argmin(np.abs(freqs - harmonic_freq))
            harmonics.append(psd[idx])

    return np.array(harmonics)


def total_harmonic_distortion(
    values: np.ndarray,
    sample_rate: float = 1.0,
    fundamental: Optional[float] = None,
    num_harmonics: int = 5
) -> float:
    """
    Compute total harmonic distortion (THD).

    THD = sqrt(sum of harmonic powers) / fundamental power

    Args:
        values: Input time series
        sample_rate: Sampling frequency
        fundamental: Fundamental frequency (if None, estimated)
        num_harmonics: Number of harmonics to include

    Returns:
        THD as ratio (0 to infinity, lower is better)
    """
    harmonics = harmonic_content(values, sample_rate, fundamental, num_harmonics + 1)

    if len(harmonics) == 0:
        return 0.0

    fundamental_power = harmonics[0]
    harmonic_powers = harmonics[1:]

    if fundamental_power == 0:
        return float('inf') if np.sum(harmonic_powers) > 0 else 0.0

    thd = np.sqrt(np.sum(harmonic_powers**2)) / fundamental_power
    return float(thd)


def signal_to_noise_ratio(
    values: np.ndarray,
    noise_floor_percentile: float = 10.0
) -> float:
    """
    Estimate signal-to-noise ratio from power spectrum.

    Args:
        values: Input time series
        noise_floor_percentile: Percentile to use as noise floor estimate

    Returns:
        SNR in dB
    """
    freqs, psd = power_spectral_density(values, 1.0)

    noise_floor = np.percentile(psd, noise_floor_percentile)
    signal_power = np.mean(psd)

    if noise_floor <= 0 or signal_power <= 0:
        return 0.0

    snr_db = 10 * np.log10(signal_power / noise_floor)
    return float(snr_db)


def phase_coherence(
    signal1: np.ndarray,
    signal2: np.ndarray,
    sample_rate: float = 1.0
) -> float:
    """
    Compute phase coherence between two signals.

    Args:
        signal1: First time series
        signal2: Second time series
        sample_rate: Sampling frequency

    Returns:
        Phase coherence (0 to 1)
    """
    min_len = min(len(signal1), len(signal2))
    s1 = np.asarray(signal1[:min_len], dtype=np.float64)
    s2 = np.asarray(signal2[:min_len], dtype=np.float64)

    # Compute cross-power spectral density
    freqs, csd = signal.csd(s1, s2, fs=sample_rate)

    # Compute auto-power spectral densities
    _, psd1 = signal.welch(s1, fs=sample_rate)
    _, psd2 = signal.welch(s2, fs=sample_rate)

    # Ensure same length
    min_len = min(len(csd), len(psd1), len(psd2))
    csd = csd[:min_len]
    psd1 = psd1[:min_len]
    psd2 = psd2[:min_len]

    # Coherence = |cross_psd|^2 / (psd1 * psd2)
    coherence = np.abs(csd)**2 / (psd1 * psd2 + 1e-12)

    return float(np.mean(coherence))


def laplace_transform(
    values: np.ndarray,
    sample_rate: float = 1.0,
    s_values: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute discrete Laplace transform approximation.

    Args:
        values: Input time series
        sample_rate: Sampling frequency
        s_values: Complex frequencies to evaluate (default: jω for FFT frequencies)

    Returns:
        Laplace transform values
    """
    values = np.asarray(values, dtype=np.float64)

    if s_values is None:
        freqs = fftfreq(len(values), d=1/sample_rate)
        s_values = 1j * 2 * np.pi * freqs

    dt = 1 / sample_rate
    t = np.arange(len(values)) * dt

    laplace_vals = []
    for s in s_values:
        val = np.sum(values * np.exp(-s * t)) * dt
        laplace_vals.append(val)

    return np.array(laplace_vals)
