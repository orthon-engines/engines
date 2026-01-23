"""
Ørthon Signal Typology: Axis Measurement Engines
=================================================

Computational engines for measuring each of the six orthogonal axes.

Each engine:
    - Takes raw time series data
    - Computes relevant metrics
    - Returns axis dataclass with measurements + classification

Dependencies:
    - numpy
    - scipy (for spectral, statistical tests)
    - statsmodels (for ADF, KPSS, GARCH)
    
Note: Some advanced metrics (RQA, Lyapunov) may require additional packages
or custom implementations. Stubs provided where needed.
"""

import numpy as np
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from typing import Tuple, Optional
import warnings

from .models import (
    MemoryAxis, InformationAxis, RecurrenceAxis,
    VolatilityAxis, FrequencyAxis, DynamicsAxis,
    DiracDiscontinuity, HeavisideDiscontinuity, StructuralDiscontinuity,
    MemoryClass, InformationClass, RecurrenceClass,
    VolatilityClass, FrequencyClass, DynamicsClass,
    ACFDecayType
)


# =============================================================================
# AXIS 1: MEMORY (Hurst, ACF, Spectral Slope)
# =============================================================================

def compute_hurst_rs(series: np.ndarray, min_window: int = 10) -> Tuple[float, float]:
    """
    Compute Hurst exponent using R/S (rescaled range) analysis.
    
    Returns:
        (hurst_exponent, r_squared)
    """
    n = len(series)
    if n < min_window * 2:
        return 0.5, 0.0
    
    # Window sizes (powers of 2 that fit)
    max_k = int(np.floor(np.log2(n / min_window)))
    if max_k < 2:
        return 0.5, 0.0
    
    window_sizes = [int(n / (2**k)) for k in range(max_k + 1)]
    window_sizes = [w for w in window_sizes if w >= min_window]
    
    rs_values = []
    
    for window_size in window_sizes:
        n_windows = n // window_size
        rs_list = []
        
        for i in range(n_windows):
            window = series[i * window_size:(i + 1) * window_size]
            
            # Mean-adjusted cumulative sum
            mean = np.mean(window)
            cumsum = np.cumsum(window - mean)
            
            # Range and standard deviation
            r = np.max(cumsum) - np.min(cumsum)
            s = np.std(window, ddof=1)
            
            if s > 0:
                rs_list.append(r / s)
        
        if rs_list:
            rs_values.append((window_size, np.mean(rs_list)))
    
    if len(rs_values) < 2:
        return 0.5, 0.0
    
    # Log-log regression
    log_n = np.log([x[0] for x in rs_values])
    log_rs = np.log([x[1] for x in rs_values])
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_rs)
    
    return np.clip(slope, 0, 1), r_value ** 2


def compute_hurst_dfa(series: np.ndarray, min_window: int = 10) -> Tuple[float, float]:
    """
    Compute Hurst exponent using Detrended Fluctuation Analysis (DFA).
    More robust to non-stationarity than R/S.
    
    Returns:
        (hurst_exponent, r_squared)
    """
    n = len(series)
    if n < min_window * 4:
        return 0.5, 0.0
    
    # Integrate the series (cumulative sum of deviations from mean)
    y = np.cumsum(series - np.mean(series))
    
    # Window sizes
    max_window = n // 4
    window_sizes = []
    w = min_window
    while w <= max_window:
        window_sizes.append(w)
        w = int(w * 1.5)  # Logarithmic spacing
    
    if len(window_sizes) < 3:
        return 0.5, 0.0
    
    fluctuations = []
    
    for window_size in window_sizes:
        n_windows = n // window_size
        f2_list = []
        
        for i in range(n_windows):
            segment = y[i * window_size:(i + 1) * window_size]
            
            # Fit linear trend
            x = np.arange(window_size)
            slope, intercept = np.polyfit(x, segment, 1)
            trend = slope * x + intercept
            
            # Fluctuation (RMS of detrended segment)
            f2 = np.mean((segment - trend) ** 2)
            f2_list.append(f2)
        
        if f2_list:
            fluctuations.append((window_size, np.sqrt(np.mean(f2_list))))
    
    if len(fluctuations) < 3:
        return 0.5, 0.0
    
    # Log-log regression
    log_n = np.log([x[0] for x in fluctuations])
    log_f = np.log([x[1] for x in fluctuations])
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_f)
    
    return np.clip(slope, 0, 1), r_value ** 2


def compute_acf_decay(series: np.ndarray, max_lag: int = 50) -> Tuple[ACFDecayType, float]:
    """
    Determine if ACF decays exponentially or power-law.
    
    Returns:
        (decay_type, half_life_in_lags)
    """
    n = len(series)
    max_lag = min(max_lag, n // 3)
    
    if max_lag < 5:
        return ACFDecayType.EXPONENTIAL, 1.0
    
    # Compute ACF
    acf = np.correlate(series - np.mean(series), series - np.mean(series), mode='full')
    acf = acf[n-1:n-1+max_lag+1]
    acf = acf / acf[0]
    
    lags = np.arange(1, len(acf))
    acf_values = np.abs(acf[1:])
    
    # Filter out zeros/negatives for log
    valid = acf_values > 0.01
    if np.sum(valid) < 3:
        return ACFDecayType.EXPONENTIAL, 1.0
    
    lags_valid = lags[valid]
    acf_valid = acf_values[valid]
    
    # Fit exponential: log(ACF) = -λ * lag
    log_acf = np.log(acf_valid)
    exp_slope, exp_intercept, exp_r, _, _ = stats.linregress(lags_valid, log_acf)
    exp_r2 = exp_r ** 2
    
    # Fit power law: log(ACF) = -α * log(lag)
    log_lags = np.log(lags_valid)
    pow_slope, pow_intercept, pow_r, _, _ = stats.linregress(log_lags, log_acf)
    pow_r2 = pow_r ** 2
    
    # Half-life from exponential fit
    if exp_slope < 0:
        half_life = -np.log(2) / exp_slope
    else:
        half_life = max_lag
    
    # Better fit wins
    if pow_r2 > exp_r2 + 0.05:  # Power law needs to be notably better
        return ACFDecayType.POWER_LAW, half_life
    else:
        return ACFDecayType.EXPONENTIAL, half_life


def compute_spectral_slope(series: np.ndarray) -> Tuple[float, float]:
    """
    Compute spectral slope (β in S(f) ~ f^-β).
    
    Returns:
        (slope, r_squared)
    """
    n = len(series)
    
    # FFT
    fft_vals = fft(series - np.mean(series))
    power = np.abs(fft_vals[:n//2]) ** 2
    freqs = fftfreq(n)[:n//2]
    
    # Exclude DC and very high frequencies
    valid = (freqs > 0.01) & (freqs < 0.4)
    
    if np.sum(valid) < 5:
        return 0.0, 0.0
    
    log_f = np.log(freqs[valid])
    log_p = np.log(power[valid] + 1e-10)
    
    slope, intercept, r_value, _, _ = stats.linregress(log_f, log_p)
    
    return -slope, r_value ** 2  # Negative because S ~ f^-β


def measure_memory_axis(series: np.ndarray, method: str = 'dfa') -> MemoryAxis:
    """
    Complete measurement of Memory axis.
    """
    # Hurst exponent
    if method == 'dfa':
        hurst, hurst_conf = compute_hurst_dfa(series)
    else:
        hurst, hurst_conf = compute_hurst_rs(series)
    
    # ACF decay
    acf_type, acf_half_life = compute_acf_decay(series)
    
    # Spectral slope
    spec_slope, spec_r2 = compute_spectral_slope(series)
    
    # Classification
    if hurst < 0.45:
        mem_class = MemoryClass.ANTI_PERSISTENT
    elif hurst > 0.55:
        mem_class = MemoryClass.PERSISTENT
    else:
        mem_class = MemoryClass.RANDOM
    
    return MemoryAxis(
        hurst_exponent=hurst,
        hurst_method=method,
        hurst_confidence=hurst_conf,
        acf_decay_type=acf_type,
        acf_half_life=acf_half_life,
        spectral_slope=spec_slope,
        spectral_slope_r2=spec_r2,
        memory_class=mem_class
    )


# =============================================================================
# AXIS 2: INFORMATION (Entropy)
# =============================================================================

def compute_permutation_entropy(series: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """
    Compute permutation entropy (Bandt & Pompe, 2002).
    Normalized to [0, 1].
    """
    n = len(series)
    
    # Create ordinal patterns
    from itertools import permutations
    import math
    
    factorial_order = math.factorial(order)
    
    # All possible permutations
    all_patterns = list(permutations(range(order)))
    pattern_counts = {p: 0 for p in all_patterns}
    
    n_patterns = 0
    for i in range(n - (order - 1) * delay):
        # Extract embedded vector
        indices = [i + j * delay for j in range(order)]
        values = series[indices]
        
        # Get ordinal pattern (rank of each value)
        pattern = tuple(np.argsort(np.argsort(values)))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        n_patterns += 1
    
    if n_patterns == 0:
        return 1.0
    
    # Compute entropy
    probs = np.array([c / n_patterns for c in pattern_counts.values() if c > 0])
    entropy = -np.sum(probs * np.log(probs))
    
    # Normalize by maximum entropy (log of factorial(order))
    max_entropy = np.log(factorial_order)
    
    return entropy / max_entropy if max_entropy > 0 else 0.0


def compute_sample_entropy(series: np.ndarray, m: int = 2, r: float = None) -> float:
    """
    Compute sample entropy.
    
    Args:
        series: Input time series
        m: Embedding dimension
        r: Tolerance (default: 0.2 * std)
    """
    n = len(series)
    
    if r is None:
        r = 0.2 * np.std(series)
    
    if r == 0 or n < m + 2:
        return 0.0
    
    def count_matches(template_length):
        count = 0
        for i in range(n - template_length):
            for j in range(i + 1, n - template_length):
                # Check if templates match within tolerance
                diff = np.abs(series[i:i+template_length] - series[j:j+template_length])
                if np.all(diff <= r):
                    count += 1
        return count
    
    # Count matches for m and m+1
    a = count_matches(m)
    b = count_matches(m + 1)
    
    if a == 0 or b == 0:
        return 0.0
    
    return -np.log(b / a)


def measure_information_axis(
    series: np.ndarray, 
    previous_entropy: float = None
) -> InformationAxis:
    """
    Complete measurement of Information axis.
    """
    # Permutation entropy
    perm_entropy = compute_permutation_entropy(series)
    
    # Sample entropy
    samp_entropy = compute_sample_entropy(series)
    
    # Entropy rate (if previous available)
    if previous_entropy is not None:
        entropy_rate = perm_entropy - previous_entropy
    else:
        entropy_rate = 0.0
    
    # Classification
    if perm_entropy < 0.4:
        info_class = InformationClass.LOW
    elif perm_entropy > 0.7:
        info_class = InformationClass.HIGH
    else:
        info_class = InformationClass.MODERATE
    
    return InformationAxis(
        entropy_permutation=perm_entropy,
        entropy_sample=samp_entropy,
        entropy_rate=entropy_rate,
        information_class=info_class
    )


# =============================================================================
# AXIS 3: RECURRENCE (RQA)
# =============================================================================

def compute_rqa(
    series: np.ndarray, 
    embedding_dim: int = 3, 
    delay: int = 1, 
    threshold: float = None,
    min_diagonal: int = 2,
    min_vertical: int = 2
) -> RecurrenceAxis:
    """
    Compute Recurrence Quantification Analysis metrics.
    
    Simplified implementation - for production use pyrqa or similar.
    """
    n = len(series)
    
    # Default threshold: 10% of max distance
    if threshold is None:
        threshold = 0.1 * (np.max(series) - np.min(series))
    
    # Create embedded vectors
    n_vectors = n - (embedding_dim - 1) * delay
    if n_vectors < 10:
        return RecurrenceAxis()
    
    embedded = np.zeros((n_vectors, embedding_dim))
    for i in range(n_vectors):
        for j in range(embedding_dim):
            embedded[i, j] = series[i + j * delay]
    
    # Compute recurrence matrix (simplified: using Euclidean distance)
    # For large series, this should use efficient implementations
    from scipy.spatial.distance import cdist
    
    if n_vectors > 500:
        # Subsample for large series
        indices = np.linspace(0, n_vectors-1, 500, dtype=int)
        embedded = embedded[indices]
        n_vectors = 500
    
    distances = cdist(embedded, embedded, 'euclidean')
    recurrence_matrix = distances <= threshold
    
    # Recurrence rate
    n_recurrent = np.sum(recurrence_matrix) - n_vectors  # Exclude diagonal
    recurrence_rate = n_recurrent / (n_vectors * (n_vectors - 1))
    
    # Find diagonal lines
    diagonal_lengths = []
    for offset in range(1, n_vectors):
        diag = np.diag(recurrence_matrix, k=offset)
        # Count consecutive True values
        current_length = 0
        for val in diag:
            if val:
                current_length += 1
            else:
                if current_length >= min_diagonal:
                    diagonal_lengths.append(current_length)
                current_length = 0
        if current_length >= min_diagonal:
            diagonal_lengths.append(current_length)
    
    # Determinism
    if diagonal_lengths:
        total_diagonal_points = sum(diagonal_lengths)
        determinism = total_diagonal_points / max(n_recurrent, 1)
        avg_diagonal = np.mean(diagonal_lengths)
        max_diagonal = max(diagonal_lengths)
        
        # Diagonal line entropy
        hist, _ = np.histogram(diagonal_lengths, bins=range(min_diagonal, max(diagonal_lengths)+2))
        hist = hist / hist.sum() if hist.sum() > 0 else hist
        hist = hist[hist > 0]
        entropy_diagonal = -np.sum(hist * np.log(hist)) if len(hist) > 0 else 0
    else:
        determinism = 0.0
        avg_diagonal = 0.0
        max_diagonal = 0
        entropy_diagonal = 0.0
    
    # Find vertical lines (laminarity)
    vertical_lengths = []
    for col in range(n_vectors):
        column = recurrence_matrix[:, col]
        current_length = 0
        for val in column:
            if val:
                current_length += 1
            else:
                if current_length >= min_vertical:
                    vertical_lengths.append(current_length)
                current_length = 0
        if current_length >= min_vertical:
            vertical_lengths.append(current_length)
    
    if vertical_lengths:
        total_vertical_points = sum(vertical_lengths)
        laminarity = total_vertical_points / max(n_recurrent, 1)
        trapping_time = np.mean(vertical_lengths)
    else:
        laminarity = 0.0
        trapping_time = 0.0
    
    # Classification
    if determinism > 0.7:
        rec_class = RecurrenceClass.DETERMINISTIC
    elif determinism < 0.4:
        rec_class = RecurrenceClass.STOCHASTIC
    else:
        rec_class = RecurrenceClass.TRANSITIONAL
    
    return RecurrenceAxis(
        determinism=np.clip(determinism, 0, 1),
        laminarity=np.clip(laminarity, 0, 1),
        entropy_diagonal=entropy_diagonal,
        recurrence_rate=recurrence_rate,
        trapping_time=trapping_time,
        max_diagonal=max_diagonal,
        avg_diagonal=avg_diagonal,
        recurrence_class=rec_class
    )


def measure_recurrence_axis(series: np.ndarray) -> RecurrenceAxis:
    """Complete measurement of Recurrence axis."""
    return compute_rqa(series)


# =============================================================================
# AXIS 4: VOLATILITY (GARCH, Hilbert)
# =============================================================================

def compute_garch_simple(series: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Simple GARCH(1,1) estimation using method of moments.
    For production, use arch package.
    
    Returns:
        (alpha, beta, omega, persistence, unconditional_variance)
    """
    # Returns (assuming series is already returns or changes)
    returns = np.diff(series)
    
    if len(returns) < 20:
        return 0.1, 0.8, 0.001, 0.9, np.var(returns)
    
    # Squared returns
    r2 = returns ** 2
    
    # Sample autocorrelation of squared returns
    mean_r2 = np.mean(r2)
    var_r2 = np.var(r2)
    
    if var_r2 < 1e-10:
        return 0.1, 0.8, 0.001, 0.9, mean_r2
    
    # ACF at lag 1
    acf1 = np.corrcoef(r2[:-1], r2[1:])[0, 1] if len(r2) > 1 else 0
    
    # Method of moments estimates
    # persistence ≈ ACF(1) of squared returns
    persistence = np.clip(acf1, 0, 0.999)
    
    # Typical alpha/beta split
    alpha = np.clip(persistence * 0.15, 0.01, 0.3)
    beta = np.clip(persistence - alpha, 0, 0.99)
    
    # Omega from unconditional variance
    unconditional_var = mean_r2
    omega = unconditional_var * (1 - alpha - beta) if (1 - alpha - beta) > 0 else 0.001
    
    return alpha, beta, omega, alpha + beta, unconditional_var


def compute_hilbert_amplitude(series: np.ndarray) -> Tuple[float, float]:
    """
    Compute amplitude envelope using Hilbert transform.
    
    Returns:
        (mean_amplitude, std_amplitude)
    """
    from scipy.signal import hilbert
    
    # Detrend
    detrended = series - np.mean(series)
    
    # Hilbert transform
    analytic = hilbert(detrended)
    amplitude = np.abs(analytic)
    
    return np.mean(amplitude), np.std(amplitude)


def measure_volatility_axis(series: np.ndarray) -> VolatilityAxis:
    """Complete measurement of Volatility axis."""
    
    # GARCH estimation
    alpha, beta, omega, persistence, unconditional = compute_garch_simple(series)
    
    # Hilbert amplitude
    hilbert_mean, hilbert_std = compute_hilbert_amplitude(series)
    
    # Classification
    if persistence < 0.85:
        vol_class = VolatilityClass.DISSIPATING
    elif persistence >= 0.99:
        vol_class = VolatilityClass.INTEGRATED
    else:
        vol_class = VolatilityClass.PERSISTENT
    
    return VolatilityAxis(
        garch_alpha=alpha,
        garch_beta=beta,
        garch_persistence=persistence,
        garch_omega=omega,
        garch_unconditional=unconditional,
        hilbert_amplitude_mean=hilbert_mean,
        hilbert_amplitude_std=hilbert_std,
        volatility_class=vol_class
    )


# =============================================================================
# AXIS 5: FREQUENCY (Spectral Characteristics)
# =============================================================================

def compute_spectral_features(series: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute spectral features.
    
    Returns:
        (centroid, bandwidth, low_high_ratio, rolloff)
    """
    n = len(series)
    
    # FFT
    fft_vals = fft(series - np.mean(series))
    power = np.abs(fft_vals[:n//2]) ** 2
    freqs = fftfreq(n)[:n//2]
    
    # Exclude DC
    power = power[1:]
    freqs = freqs[1:]
    
    if len(freqs) == 0 or np.sum(power) < 1e-10:
        return 0.25, 0.1, 1.0, 0.25
    
    # Normalize power
    power = power / np.sum(power)
    
    # Spectral centroid (center of mass)
    centroid = np.sum(freqs * power)
    
    # Spectral bandwidth (spread around centroid)
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * power))
    
    # Low/high frequency ratio (split at 0.1 Nyquist)
    low_mask = freqs < 0.1
    high_mask = freqs >= 0.1
    
    low_power = np.sum(power[low_mask]) if np.any(low_mask) else 0
    high_power = np.sum(power[high_mask]) if np.any(high_mask) else 1e-10
    
    low_high_ratio = low_power / high_power
    
    # Rolloff frequency (85% energy)
    cumsum = np.cumsum(power)
    rolloff_idx = np.searchsorted(cumsum, 0.85)
    rolloff = freqs[min(rolloff_idx, len(freqs)-1)]
    
    return centroid, bandwidth, low_high_ratio, rolloff


def measure_frequency_axis(series: np.ndarray) -> FrequencyAxis:
    """Complete measurement of Frequency axis."""
    
    centroid, bandwidth, lh_ratio, rolloff = compute_spectral_features(series)
    
    # Classification
    if bandwidth < 0.08:
        freq_class = FrequencyClass.NARROWBAND
    elif bandwidth > 0.18:
        freq_class = FrequencyClass.BROADBAND
    else:
        freq_class = FrequencyClass.ONE_OVER_F
    
    return FrequencyAxis(
        spectral_centroid=centroid,
        spectral_bandwidth=bandwidth,
        spectral_low_high_ratio=lh_ratio,
        spectral_rolloff=rolloff,
        frequency_class=freq_class
    )


# =============================================================================
# AXIS 6: DYNAMICS (Lyapunov, Embedding)
# =============================================================================

def estimate_embedding_dimension(series: np.ndarray, max_dim: int = 10) -> int:
    """
    Estimate embedding dimension using false nearest neighbors (simplified).
    """
    n = len(series)
    
    if n < 50:
        return 2
    
    # Simple heuristic based on autocorrelation
    acf = np.correlate(series - np.mean(series), series - np.mean(series), mode='full')
    acf = acf[n-1:] / acf[n-1]
    
    # Find first zero crossing
    zero_cross = np.where(acf < 0)[0]
    if len(zero_cross) > 0:
        delay = zero_cross[0]
    else:
        delay = n // 10
    
    # Estimate dimension (heuristic: log2 of delay)
    dim = max(2, min(int(np.log2(delay + 1)) + 2, max_dim))
    
    return dim


def compute_lyapunov_simple(
    series: np.ndarray, 
    embedding_dim: int = 3, 
    delay: int = 1,
    n_neighbors: int = 5
) -> Tuple[float, float]:
    """
    Simplified Lyapunov exponent estimation.
    For production, use nolds or similar package.
    
    Returns:
        (lyapunov_exponent, confidence)
    """
    n = len(series)
    n_vectors = n - (embedding_dim - 1) * delay
    
    if n_vectors < 50:
        return 0.0, 0.0
    
    # Create embedded vectors
    embedded = np.zeros((n_vectors, embedding_dim))
    for i in range(n_vectors):
        for j in range(embedding_dim):
            embedded[i, j] = series[i + j * delay]
    
    # For each point, find nearest neighbor and track divergence
    from scipy.spatial.distance import cdist
    
    if n_vectors > 300:
        indices = np.linspace(0, n_vectors-1, 300, dtype=int)
        embedded = embedded[indices]
        n_vectors = 300
    
    distances = cdist(embedded, embedded, 'euclidean')
    
    # Find nearest neighbors (excluding self and temporal neighbors)
    divergences = []
    
    for i in range(n_vectors - 10):
        dist_row = distances[i].copy()
        dist_row[max(0, i-3):min(n_vectors, i+4)] = np.inf  # Exclude temporal neighbors
        
        nearest_idx = np.argmin(dist_row)
        initial_dist = dist_row[nearest_idx]
        
        if initial_dist < 1e-10:
            continue
        
        # Track divergence over time
        for k in range(1, min(10, n_vectors - max(i, nearest_idx) - 1)):
            if i + k < n_vectors and nearest_idx + k < n_vectors:
                later_dist = np.linalg.norm(embedded[i + k] - embedded[nearest_idx + k])
                if later_dist > 1e-10 and initial_dist > 1e-10:
                    divergences.append((k, np.log(later_dist / initial_dist)))
    
    if len(divergences) < 10:
        return 0.0, 0.0
    
    # Linear regression on divergence vs time
    times = np.array([d[0] for d in divergences])
    log_divs = np.array([d[1] for d in divergences])
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(times, log_divs)
    
    return slope, r_value ** 2


def measure_dynamics_axis(series: np.ndarray) -> DynamicsAxis:
    """Complete measurement of Dynamics axis."""
    
    # Embedding dimension
    embed_dim = estimate_embedding_dimension(series)
    
    # Lyapunov exponent
    lyapunov, lyap_conf = compute_lyapunov_simple(series, embedding_dim=embed_dim)
    
    # Classification
    if lyapunov < -0.05:
        dyn_class = DynamicsClass.STABLE
    elif lyapunov > 0.05:
        dyn_class = DynamicsClass.CHAOTIC
    else:
        dyn_class = DynamicsClass.EDGE_OF_CHAOS
    
    return DynamicsAxis(
        lyapunov_exponent=lyapunov,
        lyapunov_confidence=lyap_conf,
        embedding_dimension=embed_dim,
        correlation_dimension=0.0,  # Would need additional implementation
        dynamics_class=dyn_class
    )


# =============================================================================
# STRUCTURAL DISCONTINUITY DETECTION
# =============================================================================

def detect_dirac_impulses(
    series: np.ndarray, 
    threshold_sigma: float = 3.0,
    decay_window: int = 5
) -> DiracDiscontinuity:
    """
    Detect impulse (Dirac-like) discontinuities.
    
    Impulses are characterized by:
    - Sharp spike above threshold
    - Decay back toward baseline
    """
    n = len(series)
    
    # Compute rolling statistics
    window = min(20, n // 5)
    if window < 3:
        return DiracDiscontinuity()
    
    # Detrend using rolling mean
    from scipy.ndimage import uniform_filter1d
    trend = uniform_filter1d(series.astype(float), size=window, mode='nearest')
    detrended = series - trend
    
    # Rolling std
    rolling_std = np.zeros(n)
    for i in range(window, n):
        rolling_std[i] = np.std(detrended[i-window:i])
    rolling_std[:window] = rolling_std[window] if window < n else 1.0
    rolling_std[rolling_std < 1e-10] = 1.0
    
    # Z-scores
    z_scores = detrended / rolling_std
    
    # Find spikes
    spike_mask = np.abs(z_scores) > threshold_sigma
    spike_indices = np.where(spike_mask)[0]
    
    if len(spike_indices) == 0:
        return DiracDiscontinuity()
    
    # Cluster nearby spikes (within decay_window)
    impulses = []
    current_cluster = [spike_indices[0]]
    
    for idx in spike_indices[1:]:
        if idx - current_cluster[-1] <= decay_window:
            current_cluster.append(idx)
        else:
            # Finalize cluster
            peak_idx = current_cluster[np.argmax(np.abs(z_scores[current_cluster]))]
            impulses.append(peak_idx)
            current_cluster = [idx]
    
    # Don't forget last cluster
    peak_idx = current_cluster[np.argmax(np.abs(z_scores[current_cluster]))]
    impulses.append(peak_idx)
    
    # Compute metrics
    magnitudes = np.abs(z_scores[impulses])
    directions = np.sign(z_scores[impulses])
    
    # Estimate half-lives (how quickly they decay)
    half_lives = []
    for imp_idx in impulses:
        peak_val = np.abs(detrended[imp_idx])
        half_val = peak_val / 2
        
        # Look forward for decay
        for k in range(1, min(decay_window * 2, n - imp_idx)):
            if np.abs(detrended[imp_idx + k]) < half_val:
                half_lives.append(k)
                break
        else:
            half_lives.append(decay_window)
    
    return DiracDiscontinuity(
        detected=True,
        count=len(impulses),
        max_magnitude=float(np.max(magnitudes)),
        mean_magnitude=float(np.mean(magnitudes)),
        mean_half_life=float(np.mean(half_lives)) if half_lives else decay_window,
        up_ratio=float(np.mean(directions > 0)),
        locations=impulses
    )


def detect_heaviside_steps(
    series: np.ndarray,
    threshold_sigma: float = 2.0,
    min_stable_periods: int = 5
) -> HeavisideDiscontinuity:
    """
    Detect step (Heaviside-like) discontinuities.
    
    Steps are characterized by:
    - Permanent level shift
    - New stable level after change
    """
    n = len(series)
    
    if n < min_stable_periods * 3:
        return HeavisideDiscontinuity()
    
    # Compute differences
    diff = np.diff(series)
    
    # Rolling std of original series (for threshold)
    window = min(20, n // 5)
    rolling_std = np.std(series[:window]) if window > 0 else 1.0
    if rolling_std < 1e-10:
        rolling_std = 1.0
    
    threshold = threshold_sigma * rolling_std
    
    # Find large jumps
    jump_mask = np.abs(diff) > threshold
    jump_indices = np.where(jump_mask)[0]
    
    if len(jump_indices) == 0:
        return HeavisideDiscontinuity()
    
    # Verify each jump is a true step (level persists)
    confirmed_steps = []
    
    for idx in jump_indices:
        # Check if level is stable after jump
        post_start = idx + 1
        post_end = min(idx + 1 + min_stable_periods, n)
        
        if post_end - post_start < min_stable_periods:
            continue
        
        post_segment = series[post_start:post_end]
        post_std = np.std(post_segment)
        
        # If post-jump segment is relatively stable, it's a step
        if post_std < rolling_std * 1.5:
            confirmed_steps.append(idx)
    
    if len(confirmed_steps) == 0:
        return HeavisideDiscontinuity()
    
    # Compute metrics
    step_magnitudes = np.abs(diff[confirmed_steps]) / rolling_std
    step_directions = np.sign(diff[confirmed_steps])
    
    return HeavisideDiscontinuity(
        detected=True,
        count=len(confirmed_steps),
        max_magnitude=float(np.max(step_magnitudes)),
        mean_magnitude=float(np.mean(step_magnitudes)),
        up_ratio=float(np.mean(step_directions > 0)),
        locations=confirmed_steps
    )


def measure_discontinuity(series: np.ndarray) -> StructuralDiscontinuity:
    """Complete measurement of structural discontinuities."""
    
    dirac = detect_dirac_impulses(series)
    heaviside = detect_heaviside_steps(series)
    
    # Structural metrics
    all_locations = sorted(dirac.locations + heaviside.locations)
    
    if len(all_locations) >= 2:
        intervals = np.diff(all_locations)
        mean_interval = float(np.mean(intervals))
        interval_cv = float(np.std(intervals) / np.mean(intervals)) if np.mean(intervals) > 0 else 0
        
        # Check if accelerating (intervals getting shorter)
        if len(intervals) >= 3:
            first_half = np.mean(intervals[:len(intervals)//2])
            second_half = np.mean(intervals[len(intervals)//2:])
            is_accelerating = second_half < first_half * 0.8
        else:
            is_accelerating = False
        
        # Dominant period (if regular)
        if interval_cv < 0.5:  # Relatively regular
            dominant_period = mean_interval
        else:
            dominant_period = 0.0
    else:
        mean_interval = 0.0
        interval_cv = 0.0
        is_accelerating = False
        dominant_period = 0.0
    
    return StructuralDiscontinuity(
        dirac=dirac,
        heaviside=heaviside,
        mean_interval=mean_interval,
        interval_cv=interval_cv,
        dominant_period=dominant_period,
        is_accelerating=is_accelerating
    )
