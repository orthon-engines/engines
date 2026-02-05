"""
PRISM Similarity and Distance Primitives

Pure mathematical functions for computing similarity and distance
between signals and vectors.
"""

import numpy as np
from scipy import stats
from typing import Optional, Tuple


def cosine_similarity(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Compute cosine similarity between two vectors.

    cos(θ) = (x · y) / (||x|| ||y||)

    Args:
        x: First vector
        y: Second vector

    Returns:
        Cosine similarity (-1 to 1)
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()

    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)

    if norm_x == 0 or norm_y == 0:
        return 0.0

    return float(np.dot(x, y) / (norm_x * norm_y))


def euclidean_distance(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Compute Euclidean distance between two vectors.

    d = sqrt(sum((x_i - y_i)^2))

    Args:
        x: First vector
        y: Second vector

    Returns:
        Euclidean distance (>= 0)
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()

    return float(np.linalg.norm(x - y))


def manhattan_distance(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Compute Manhattan (L1) distance between two vectors.

    d = sum(|x_i - y_i|)

    Args:
        x: First vector
        y: Second vector

    Returns:
        Manhattan distance (>= 0)
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()

    return float(np.sum(np.abs(x - y)))


def correlation_coefficient(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Compute Pearson correlation coefficient.

    r = cov(x,y) / (std(x) * std(y))

    Args:
        x: First signal
        y: Second signal

    Returns:
        Correlation coefficient (-1 to 1)
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()

    # Use minimum length
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    # Remove NaN pairs
    valid = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid) < 2:
        return np.nan

    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def spearman_correlation(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Compute Spearman rank correlation.

    Non-parametric measure of monotonic relationship.

    Args:
        x: First signal
        y: Second signal

    Returns:
        Spearman correlation (-1 to 1)
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()

    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    valid = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid) < 2:
        return np.nan

    rho, _ = stats.spearmanr(x[valid], y[valid])
    return float(rho)


def mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Estimate mutual information between two signals.

    MI(X,Y) = H(X) + H(Y) - H(X,Y)

    Args:
        x: First signal
        y: Second signal
        n_bins: Number of bins for histogram estimation

    Returns:
        Mutual information (>= 0, in bits)
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()

    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    # Remove NaN pairs
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]

    if len(x) < n_bins:
        return 0.0

    # Bin the data
    x_binned = np.digitize(x, np.linspace(np.min(x), np.max(x), n_bins))
    y_binned = np.digitize(y, np.linspace(np.min(y), np.max(y), n_bins))

    # Joint histogram
    joint_hist = np.histogram2d(x_binned, y_binned, bins=n_bins)[0]
    joint_hist = joint_hist / np.sum(joint_hist)  # Normalize

    # Marginals
    px = np.sum(joint_hist, axis=1)
    py = np.sum(joint_hist, axis=0)

    # Entropies
    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    h_x = entropy(px)
    h_y = entropy(py)
    h_xy = entropy(joint_hist.flatten())

    mi = h_x + h_y - h_xy
    return float(max(0, mi))


def cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: Optional[int] = None,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cross-correlation between two signals.

    Args:
        x: First signal
        y: Second signal
        max_lag: Maximum lag to compute
        normalize: If True, normalize to [-1, 1]

    Returns:
        Tuple of (lags, cross_correlation_values)
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()

    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    if max_lag is None:
        max_lag = min_len - 1

    # Full cross-correlation
    xcorr = np.correlate(x - np.mean(x), y - np.mean(y), mode='full')

    if normalize:
        norm = np.std(x) * np.std(y) * len(x)
        if norm > 0:
            xcorr = xcorr / norm

    # Extract lags
    n = len(x)
    lags = np.arange(-n + 1, n)

    # Trim to max_lag
    center = n - 1
    start = center - max_lag
    end = center + max_lag + 1

    return lags[start:end], xcorr[start:end]


def lag_at_max_correlation(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: Optional[int] = None
) -> Tuple[int, float]:
    """
    Find the lag at which cross-correlation is maximized.

    Args:
        x: First signal
        y: Second signal
        max_lag: Maximum lag to search

    Returns:
        Tuple of (optimal_lag, max_correlation)
        Positive lag means y leads x
    """
    lags, xcorr = cross_correlation(x, y, max_lag, normalize=True)
    max_idx = np.argmax(xcorr)

    return int(lags[max_idx]), float(xcorr[max_idx])


def dynamic_time_warping(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[int] = None
) -> float:
    """
    Compute Dynamic Time Warping distance.

    DTW finds the optimal alignment between two sequences,
    allowing for non-linear time warping.

    Args:
        x: First signal
        y: Second signal
        window: Sakoe-Chiba band width (None for full)

    Returns:
        DTW distance (>= 0)
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()

    n, m = len(x), len(y)

    if window is None:
        window = max(n, m)

    # DTW cost matrix
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window) + 1

        for j in range(j_start, j_end):
            cost = abs(x[i - 1] - y[j - 1])
            dtw[i, j] = cost + min(
                dtw[i - 1, j],      # Insertion
                dtw[i, j - 1],      # Deletion
                dtw[i - 1, j - 1]   # Match
            )

    return float(dtw[n, m])


def coherence(
    x: np.ndarray,
    y: np.ndarray,
    sample_rate: float = 1.0,
    nperseg: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute magnitude squared coherence between two signals.

    Coherence measures frequency-domain correlation.

    Args:
        x: First signal
        y: Second signal
        sample_rate: Sampling frequency
        nperseg: Segment length for Welch method

    Returns:
        Tuple of (frequencies, coherence_values)
    """
    from scipy import signal as scipy_signal

    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()

    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    if nperseg is None:
        nperseg = min(256, min_len // 4)

    freqs, coh = scipy_signal.coherence(x, y, fs=sample_rate, nperseg=nperseg)

    return freqs, coh


def earth_movers_distance(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Compute Earth Mover's Distance (Wasserstein distance) between distributions.

    Args:
        x: First distribution (as samples)
        y: Second distribution (as samples)

    Returns:
        EMD (>= 0)
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()

    # Sort both arrays
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)

    # Interpolate to same length for fair comparison
    n = max(len(x), len(y))
    x_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x_sorted)
    y_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(y)), y_sorted)

    return float(np.mean(np.abs(x_interp - y_interp)))
