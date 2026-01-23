"""
Derivatives Module
==================

Computes derivations with formulas, intermediate values, and interpretations.
Used by thesis mode to generate publication-ready documentation.

Each derivation returns:
- name: Metric name
- category: Statistical, Complexity, Frequency, Dynamics, Geometry, State
- formula: LaTeX formula
- value: Computed result
- variables: Dict of intermediate values
- interpretation: Plain-English explanation
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np

try:
    from scipy import stats
    from scipy.fft import fft
    from scipy.signal import welch
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class DerivationResult:
    """Result of a derivation computation with full provenance."""
    name: str
    category: str  # Statistical, Complexity, Frequency, Dynamics, Geometry, State
    formula: str   # LaTeX formula
    value: Any     # Computed value
    variables: Dict[str, Any] = field(default_factory=dict)  # Intermediate values
    interpretation: str = ""  # Plain-English explanation


# =============================================================================
# ENGINE LISTS
# =============================================================================

SINGLE_SIGNAL_ENGINES = [
    "mean", "std", "skewness", "kurtosis", "entropy",
    "hurst", "spectral_entropy", "dominant_frequency",
]

PAIRWISE_ENGINES = [
    "correlation", "mutual_information", "dtw_distance",
]

STATE_ENGINES = [
    "hd_slope", "regime_stability", "coherence_velocity",
]


# =============================================================================
# STATISTICAL DERIVATIONS
# =============================================================================

def compute_mean(values: np.ndarray) -> DerivationResult:
    """Compute arithmetic mean with derivation."""
    n = len(values)
    total = np.sum(values)
    mean_val = total / n

    return DerivationResult(
        name="Arithmetic Mean",
        category="Statistical",
        formula=r"\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i",
        value=mean_val,
        variables={
            "n": n,
            "sum": total,
            "x_min": np.min(values),
            "x_max": np.max(values),
        },
        interpretation=f"The central tendency of the signal. Value {mean_val:.4f} indicates the typical magnitude."
    )


def compute_std(values: np.ndarray) -> DerivationResult:
    """Compute standard deviation with derivation."""
    n = len(values)
    mean_val = np.mean(values)
    variance = np.sum((values - mean_val) ** 2) / (n - 1)
    std_val = np.sqrt(variance)

    return DerivationResult(
        name="Standard Deviation",
        category="Statistical",
        formula=r"s = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2}",
        value=std_val,
        variables={
            "n": n,
            "mean": mean_val,
            "variance": variance,
        },
        interpretation=f"Measures spread around the mean. Value {std_val:.4f} indicates {'high' if std_val > mean_val * 0.5 else 'moderate' if std_val > mean_val * 0.2 else 'low'} variability."
    )


def compute_skewness(values: np.ndarray) -> DerivationResult:
    """Compute skewness with derivation."""
    n = len(values)
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)

    if std_val == 0:
        skew = 0.0
    else:
        skew = np.sum(((values - mean_val) / std_val) ** 3) * n / ((n - 1) * (n - 2))

    if skew > 0.5:
        interp = "right-skewed (tail extends right)"
    elif skew < -0.5:
        interp = "left-skewed (tail extends left)"
    else:
        interp = "approximately symmetric"

    return DerivationResult(
        name="Skewness",
        category="Statistical",
        formula=r"\gamma_1 = \frac{n}{(n-1)(n-2)} \sum_{i=1}^{n} \left(\frac{x_i - \bar{x}}{s}\right)^3",
        value=skew,
        variables={
            "n": n,
            "mean": mean_val,
            "std": std_val,
        },
        interpretation=f"Measures asymmetry of distribution. Value {skew:.4f} indicates {interp}."
    )


def compute_kurtosis(values: np.ndarray) -> DerivationResult:
    """Compute excess kurtosis with derivation."""
    n = len(values)
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)

    if std_val == 0:
        kurt = 0.0
    else:
        m4 = np.mean((values - mean_val) ** 4)
        kurt = m4 / (std_val ** 4) - 3  # Excess kurtosis

    if kurt > 1:
        interp = "heavy-tailed (more outliers than normal)"
    elif kurt < -1:
        interp = "light-tailed (fewer outliers than normal)"
    else:
        interp = "approximately normal tails"

    return DerivationResult(
        name="Excess Kurtosis",
        category="Statistical",
        formula=r"\gamma_2 = \frac{m_4}{s^4} - 3 \quad \text{where } m_4 = \frac{1}{n}\sum(x_i - \bar{x})^4",
        value=kurt,
        variables={
            "n": n,
            "mean": mean_val,
            "std": std_val,
            "m4": m4 if std_val != 0 else 0,
        },
        interpretation=f"Measures tail heaviness. Value {kurt:.4f} indicates {interp}."
    )


# =============================================================================
# COMPLEXITY DERIVATIONS
# =============================================================================

def compute_entropy(values: np.ndarray, bins: int = 50) -> DerivationResult:
    """Compute Shannon entropy with derivation."""
    # Discretize into bins
    hist, bin_edges = np.histogram(values, bins=bins, density=True)
    bin_width = bin_edges[1] - bin_edges[0]

    # Compute probabilities
    probs = hist * bin_width
    probs = probs[probs > 0]  # Remove zeros for log

    # Shannon entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(bins)
    normalized = entropy / max_entropy if max_entropy > 0 else 0

    return DerivationResult(
        name="Shannon Entropy",
        category="Complexity",
        formula=r"H(X) = -\sum_{i=1}^{n} p_i \log_2(p_i)",
        value=entropy,
        variables={
            "bins": bins,
            "bin_width": bin_width,
            "max_entropy": max_entropy,
            "normalized_entropy": normalized,
            "non_zero_bins": len(probs),
        },
        interpretation=f"Measures uncertainty/randomness. Normalized entropy {normalized:.2%} of maximum ({max_entropy:.2f} bits)."
    )


def compute_hurst(values: np.ndarray) -> DerivationResult:
    """Compute Hurst exponent using R/S analysis."""
    n = len(values)

    if n < 20:
        return DerivationResult(
            name="Hurst Exponent",
            category="Complexity",
            formula=r"H = \frac{\log(R/S)}{\log(n)}",
            value=np.nan,
            variables={"n": n, "error": "Insufficient data"},
            interpretation="Insufficient data for Hurst calculation (need >= 20 points)."
        )

    # R/S analysis
    mean_val = np.mean(values)
    deviations = values - mean_val
    cumulative = np.cumsum(deviations)

    R = np.max(cumulative) - np.min(cumulative)  # Range
    S = np.std(values, ddof=1)  # Standard deviation

    if S == 0:
        hurst = 0.5
    else:
        RS = R / S
        hurst = np.log(RS) / np.log(n) if RS > 0 else 0.5

    # Clamp to valid range
    hurst = np.clip(hurst, 0, 1)

    if hurst > 0.6:
        interp = "persistent (trending behavior)"
    elif hurst < 0.4:
        interp = "anti-persistent (mean-reverting)"
    else:
        interp = "random walk behavior"

    return DerivationResult(
        name="Hurst Exponent",
        category="Complexity",
        formula=r"H = \frac{\log(R/S)}{\log(n)} \quad \text{where } R = \max(Y_t) - \min(Y_t), \; Y_t = \sum_{i=1}^{t}(x_i - \bar{x})",
        value=hurst,
        variables={
            "n": n,
            "mean": mean_val,
            "R": R,
            "S": S,
            "R/S": R/S if S != 0 else np.nan,
        },
        interpretation=f"Measures long-term memory. Value {hurst:.4f} indicates {interp}."
    )


# =============================================================================
# FREQUENCY DERIVATIONS
# =============================================================================

def compute_spectral_entropy(values: np.ndarray) -> DerivationResult:
    """Compute spectral entropy from power spectrum."""
    if not HAS_SCIPY:
        return DerivationResult(
            name="Spectral Entropy",
            category="Frequency",
            formula=r"H_s = -\sum_{f} p_f \log_2(p_f)",
            value=np.nan,
            variables={"error": "scipy not available"},
            interpretation="Cannot compute: scipy required."
        )

    # Compute power spectrum
    freqs, psd = welch(values, nperseg=min(256, len(values)))

    # Normalize to probability distribution
    psd_norm = psd / np.sum(psd)
    psd_norm = psd_norm[psd_norm > 0]

    # Spectral entropy
    spectral_ent = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    max_ent = np.log2(len(psd_norm))
    normalized = spectral_ent / max_ent if max_ent > 0 else 0

    return DerivationResult(
        name="Spectral Entropy",
        category="Frequency",
        formula=r"H_s = -\sum_{f} \frac{P(f)}{\sum P} \log_2\left(\frac{P(f)}{\sum P}\right)",
        value=spectral_ent,
        variables={
            "n_frequencies": len(freqs),
            "total_power": np.sum(psd),
            "max_entropy": max_ent,
            "normalized": normalized,
        },
        interpretation=f"Measures frequency complexity. Normalized {normalized:.2%} indicates {'broadband noise' if normalized > 0.8 else 'some dominant frequencies' if normalized > 0.5 else 'narrow frequency content'}."
    )


def compute_dominant_frequency(values: np.ndarray, fs: float = 1.0) -> DerivationResult:
    """Find dominant frequency from FFT."""
    n = len(values)

    # FFT
    fft_vals = np.abs(fft(values))[:n//2]
    freqs = np.fft.fftfreq(n, d=1/fs)[:n//2]

    # Find peak
    peak_idx = np.argmax(fft_vals[1:]) + 1  # Skip DC
    dominant_freq = freqs[peak_idx]
    peak_power = fft_vals[peak_idx]
    total_power = np.sum(fft_vals)

    return DerivationResult(
        name="Dominant Frequency",
        category="Frequency",
        formula=r"f_{dom} = \arg\max_f |X(f)|",
        value=dominant_freq,
        variables={
            "n_samples": n,
            "sampling_rate": fs,
            "peak_power": peak_power,
            "total_power": total_power,
            "power_ratio": peak_power / total_power if total_power > 0 else 0,
        },
        interpretation=f"Primary oscillation at {dominant_freq:.4f} Hz, containing {100*peak_power/total_power:.1f}% of spectral power."
    )


# =============================================================================
# PAIRWISE DERIVATIONS
# =============================================================================

def compute_correlation(x: np.ndarray, y: np.ndarray) -> DerivationResult:
    """Compute Pearson correlation with derivation."""
    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    cov_xy = np.sum((x - mean_x) * (y - mean_y)) / (n - 1)
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)

    if std_x == 0 or std_y == 0:
        corr = 0.0
    else:
        corr = cov_xy / (std_x * std_y)

    if abs(corr) > 0.7:
        strength = "strong"
    elif abs(corr) > 0.4:
        strength = "moderate"
    else:
        strength = "weak"

    direction = "positive" if corr > 0 else "negative" if corr < 0 else "no"

    return DerivationResult(
        name="Pearson Correlation",
        category="Geometry",
        formula=r"\rho_{xy} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y} = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{(n-1)\sigma_X\sigma_Y}",
        value=corr,
        variables={
            "n": n,
            "mean_x": mean_x,
            "mean_y": mean_y,
            "std_x": std_x,
            "std_y": std_y,
            "cov_xy": cov_xy,
        },
        interpretation=f"Measures linear relationship. Value {corr:.4f} indicates {strength} {direction} correlation."
    )


def compute_mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 20) -> DerivationResult:
    """Compute mutual information with derivation."""
    # Joint histogram
    hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins)

    # Normalize to joint probability
    pxy = hist_2d / np.sum(hist_2d)

    # Marginals
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))

    # Normalized MI
    hx = -np.sum(px[px > 0] * np.log2(px[px > 0] + 1e-10))
    hy = -np.sum(py[py > 0] * np.log2(py[py > 0] + 1e-10))
    nmi = 2 * mi / (hx + hy) if (hx + hy) > 0 else 0

    return DerivationResult(
        name="Mutual Information",
        category="Geometry",
        formula=r"I(X;Y) = \sum_{x,y} p(x,y) \log_2\frac{p(x,y)}{p(x)p(y)}",
        value=mi,
        variables={
            "bins": bins,
            "H(X)": hx,
            "H(Y)": hy,
            "normalized_MI": nmi,
        },
        interpretation=f"Measures shared information. {mi:.4f} bits shared, normalized MI = {nmi:.2%}."
    )


# =============================================================================
# STATE DERIVATIONS
# =============================================================================

def compute_hd_slope(distances: np.ndarray, timestamps: np.ndarray = None) -> DerivationResult:
    """Compute Hausdorff distance slope (coherence velocity)."""
    if timestamps is None:
        timestamps = np.arange(len(distances))

    n = len(distances)

    # Linear regression: distance = slope * time + intercept
    mean_t = np.mean(timestamps)
    mean_d = np.mean(distances)

    numerator = np.sum((timestamps - mean_t) * (distances - mean_d))
    denominator = np.sum((timestamps - mean_t) ** 2)

    slope = numerator / denominator if denominator != 0 else 0
    intercept = mean_d - slope * mean_t

    # R-squared
    predicted = slope * timestamps + intercept
    ss_res = np.sum((distances - predicted) ** 2)
    ss_tot = np.sum((distances - mean_d) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    if slope > 0.01:
        interp = "geometry diverging (losing coherence)"
    elif slope < -0.01:
        interp = "geometry converging (gaining coherence)"
    else:
        interp = "geometry stable"

    return DerivationResult(
        name="HD Slope (Coherence Velocity)",
        category="State",
        formula=r"\frac{d(\text{HD})}{dt} = \frac{\sum(t_i - \bar{t})(d_i - \bar{d})}{\sum(t_i - \bar{t})^2}",
        value=slope,
        variables={
            "n_points": n,
            "mean_distance": mean_d,
            "intercept": intercept,
            "r_squared": r_squared,
            "start_distance": distances[0],
            "end_distance": distances[-1],
        },
        interpretation=f"Rate of geometry change. Slope {slope:.6f} indicates {interp}. R² = {r_squared:.4f}."
    )


# =============================================================================
# COMPUTE ALL
# =============================================================================

def compute_all(values: np.ndarray) -> List[DerivationResult]:
    """Compute all single-signal derivations."""
    results = []

    # Statistical
    results.append(compute_mean(values))
    results.append(compute_std(values))
    results.append(compute_skewness(values))
    results.append(compute_kurtosis(values))

    # Complexity
    results.append(compute_entropy(values))
    results.append(compute_hurst(values))

    # Frequency
    results.append(compute_spectral_entropy(values))
    results.append(compute_dominant_frequency(values))

    return results
