"""
Takens Embedding Engine
=======================

Reconstructs phase space from time series using Takens' theorem.
Unified engine combining embedding and phase space analysis.

Metrics:
    - embedding_dimension: Optimal embedding dimension
    - time_delay: Optimal time delay (tau)
    - correlation_dimension: Fractal dimension of attractor
    - largest_lyapunov: Largest Lyapunov exponent
    - attractor_type: Classified attractor type

Usage:
    from prism.engines.typology.takens import compute_takens_embedding
    result = compute_takens_embedding(values)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy import stats
from scipy.spatial import distance


def compute_takens_embedding(
    values: np.ndarray,
    max_dim: int = 10,
    max_tau: int = 50,
    method: str = "auto"
) -> Dict[str, Any]:
    """
    Compute Takens embedding and phase space metrics.

    Args:
        values: 1D array of time series values
        max_dim: Maximum embedding dimension to test
        max_tau: Maximum time delay to test
        method: Method for parameter selection ('auto', 'ami', 'fnn')

    Returns:
        Dictionary with embedding metrics
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]

    n = len(values)
    if n < 100:
        return _empty_result("Insufficient data (need >= 100 points)")

    # Normalize
    values = (values - np.mean(values)) / (np.std(values) + 1e-10)

    # Find optimal time delay using average mutual information
    tau = _find_optimal_tau(values, max_tau)

    # Find optimal embedding dimension using false nearest neighbors
    dim = _find_optimal_dimension(values, tau, max_dim)

    # Construct embedding
    embedding = _construct_embedding(values, dim, tau)

    if len(embedding) < 50:
        return _empty_result("Embedding too short after reconstruction")

    # Compute correlation dimension
    correlation_dim = _correlation_dimension(embedding)

    # Compute largest Lyapunov exponent
    lyapunov = _largest_lyapunov(embedding, tau)

    # Classify attractor
    attractor_type = _classify_attractor(correlation_dim, lyapunov)

    # Phase space statistics
    centroid = np.mean(embedding, axis=0)
    distances = np.linalg.norm(embedding - centroid, axis=1)

    # Recurrence metrics
    recurrence_rate = _recurrence_rate(embedding, threshold=0.1)

    return {
        "embedding_dimension": int(dim),
        "time_delay": int(tau),
        "correlation_dimension": float(correlation_dim),
        "largest_lyapunov": float(lyapunov),
        "attractor_type": attractor_type,
        "phase_space_volume": float(np.prod(np.std(embedding, axis=0))),
        "mean_distance_from_centroid": float(np.mean(distances)),
        "max_distance_from_centroid": float(np.max(distances)),
        "recurrence_rate": float(recurrence_rate),
        "embedding_length": len(embedding),
        "n_observations": n,
    }


def _find_optimal_tau(values: np.ndarray, max_tau: int) -> int:
    """
    Find optimal time delay using first minimum of average mutual information.
    Falls back to first zero crossing of autocorrelation.
    """
    n = len(values)
    max_tau = min(max_tau, n // 4)

    # Method 1: Average Mutual Information
    ami = _compute_ami(values, max_tau)

    # Find first local minimum
    for i in range(1, len(ami) - 1):
        if ami[i] < ami[i-1] and ami[i] < ami[i+1]:
            return i

    # Fallback: First zero crossing of ACF
    acf = _compute_acf(values, max_tau)
    for i in range(1, len(acf)):
        if acf[i] <= 0:
            return i

    # Default: 1/e decay of ACF
    threshold = acf[0] / np.e
    for i in range(1, len(acf)):
        if acf[i] < threshold:
            return i

    return max_tau // 4


def _compute_ami(values: np.ndarray, max_tau: int) -> np.ndarray:
    """
    Compute Average Mutual Information for each lag.
    """
    n = len(values)
    ami = np.zeros(max_tau + 1)

    # Discretize values for MI estimation
    n_bins = max(10, int(np.sqrt(n / 5)))

    for tau in range(max_tau + 1):
        if tau == 0:
            ami[tau] = np.inf  # Self-information
            continue

        x = values[:-tau]
        y = values[tau:]

        # 2D histogram for joint distribution
        hist_2d, _, _ = np.histogram2d(x, y, bins=n_bins)
        hist_2d = hist_2d / hist_2d.sum()

        # Marginals
        px = hist_2d.sum(axis=1)
        py = hist_2d.sum(axis=0)

        # Mutual information
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if hist_2d[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += hist_2d[i, j] * np.log2(hist_2d[i, j] / (px[i] * py[j]))

        ami[tau] = mi

    return ami


def _compute_acf(values: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Compute autocorrelation function.
    """
    n = len(values)
    mean = np.mean(values)
    var = np.var(values)

    if var == 0:
        return np.ones(max_lag + 1)

    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            acf[lag] = np.sum((values[:-lag] - mean) * (values[lag:] - mean)) / (n * var)

    return acf


def _find_optimal_dimension(values: np.ndarray, tau: int, max_dim: int) -> int:
    """
    Find optimal embedding dimension using False Nearest Neighbors (FNN).
    """
    n = len(values)
    fnn_fractions = []

    for dim in range(1, max_dim + 1):
        embedding = _construct_embedding(values, dim, tau)
        if len(embedding) < 50:
            break

        fnn = _false_nearest_neighbors(embedding, values, dim, tau)
        fnn_fractions.append(fnn)

        # Stop if FNN drops below threshold
        if fnn < 0.01:
            return dim

    # Return dimension where FNN is minimized
    if fnn_fractions:
        return np.argmin(fnn_fractions) + 1
    return 3  # Default


def _false_nearest_neighbors(
    embedding: np.ndarray,
    values: np.ndarray,
    dim: int,
    tau: int
) -> float:
    """
    Compute fraction of false nearest neighbors.
    """
    n_points = len(embedding)
    if n_points < 10:
        return 1.0

    # Sample points for efficiency
    n_sample = min(500, n_points)
    indices = np.random.choice(n_points, n_sample, replace=False)

    # Threshold for FNN
    Rtol = 15.0
    Atol = 2.0

    # Standard deviation for normalization
    std = np.std(values)
    if std == 0:
        return 1.0

    n_fnn = 0
    n_valid = 0

    for idx in indices:
        # Find nearest neighbor in current dimension
        point = embedding[idx]
        distances = np.linalg.norm(embedding - point, axis=1)
        distances[idx] = np.inf  # Exclude self

        nn_idx = np.argmin(distances)
        nn_dist = distances[nn_idx]

        if nn_dist == 0:
            continue

        n_valid += 1

        # Check if this is a false neighbor
        # Need to look at the next dimension
        next_idx = idx + dim * tau
        next_nn_idx = nn_idx + dim * tau

        if next_idx < len(values) and next_nn_idx < len(values):
            # Distance in the (dim+1)th component
            extra_dist = abs(values[next_idx] - values[next_nn_idx])

            # FNN criterion 1: ratio test
            ratio = extra_dist / nn_dist if nn_dist > 0 else np.inf

            # FNN criterion 2: absolute test
            abs_test = extra_dist / std

            if ratio > Rtol or abs_test > Atol:
                n_fnn += 1

    return n_fnn / n_valid if n_valid > 0 else 1.0


def _construct_embedding(values: np.ndarray, dim: int, tau: int) -> np.ndarray:
    """
    Construct delay embedding.
    """
    n = len(values)
    n_vectors = n - (dim - 1) * tau

    if n_vectors < 1:
        return np.array([])

    embedding = np.zeros((n_vectors, dim))
    for i in range(dim):
        embedding[:, i] = values[i * tau:i * tau + n_vectors]

    return embedding


def _correlation_dimension(embedding: np.ndarray, n_samples: int = 500) -> float:
    """
    Estimate correlation dimension using Grassberger-Procaccia algorithm.
    """
    n_points = len(embedding)
    if n_points < 50:
        return np.nan

    # Sample points for efficiency
    n_samples = min(n_samples, n_points)
    indices = np.random.choice(n_points, n_samples, replace=False)
    sample = embedding[indices]

    # Compute pairwise distances
    dists = distance.pdist(sample)
    dists = dists[dists > 0]  # Remove zeros

    if len(dists) < 10:
        return np.nan

    # Multiple radii
    log_dists = np.log(dists)
    r_min = np.percentile(log_dists, 5)
    r_max = np.percentile(log_dists, 95)

    radii = np.exp(np.linspace(r_min, r_max, 20))

    # Correlation integral
    log_r = []
    log_C = []

    for r in radii:
        C = np.sum(dists < r) / len(dists)
        if C > 0:
            log_r.append(np.log(r))
            log_C.append(np.log(C))

    if len(log_r) < 5:
        return np.nan

    # Linear regression for slope
    slope, _, r_value, _, _ = stats.linregress(log_r, log_C)

    # Only trust if R^2 is good
    if r_value ** 2 < 0.9:
        return np.nan

    return slope


def _largest_lyapunov(embedding: np.ndarray, tau: int, n_iter: int = 20) -> float:
    """
    Estimate largest Lyapunov exponent using Rosenstein's algorithm.
    """
    n_points = len(embedding)
    if n_points < 100:
        return np.nan

    # Find nearest neighbors (excluding temporally close points)
    min_temporal_sep = tau * 2

    divergences = []

    for i in range(min(500, n_points - n_iter)):
        point = embedding[i]

        # Find nearest neighbor
        min_dist = np.inf
        nn_idx = -1

        for j in range(n_points):
            if abs(i - j) < min_temporal_sep:
                continue

            dist = np.linalg.norm(embedding[j] - point)
            if dist < min_dist and dist > 0:
                min_dist = dist
                nn_idx = j

        if nn_idx < 0:
            continue

        # Track divergence
        for k in range(1, n_iter + 1):
            idx1 = i + k
            idx2 = nn_idx + k

            if idx1 >= n_points or idx2 >= n_points:
                break

            new_dist = np.linalg.norm(embedding[idx1] - embedding[idx2])
            if new_dist > 0 and min_dist > 0:
                divergences.append((k, np.log(new_dist / min_dist)))

    if len(divergences) < 10:
        return np.nan

    # Average divergence at each time step
    divergences = np.array(divergences)

    # Linear fit to get Lyapunov exponent
    unique_k = np.unique(divergences[:, 0])
    mean_div = []

    for k in unique_k:
        mask = divergences[:, 0] == k
        mean_div.append((k, np.mean(divergences[mask, 1])))

    mean_div = np.array(mean_div)

    if len(mean_div) < 3:
        return np.nan

    # Slope of log(divergence) vs time
    slope, _, _, _, _ = stats.linregress(mean_div[:, 0], mean_div[:, 1])

    return slope


def _recurrence_rate(embedding: np.ndarray, threshold: float = 0.1) -> float:
    """
    Compute recurrence rate in phase space.
    """
    n_points = len(embedding)
    if n_points < 10:
        return None

    # Sample for efficiency
    n_sample = min(500, n_points)
    indices = np.random.choice(n_points, n_sample, replace=False)
    sample = embedding[indices]

    # Normalize threshold by typical distance
    dists = distance.pdist(sample)
    if len(dists) == 0:
        return None

    eps = threshold * np.std(dists)

    # Count recurrences
    n_recur = np.sum(dists < eps)
    n_total = len(dists)

    return n_recur / n_total if n_total > 0 else None


def _classify_attractor(correlation_dim: float, lyapunov: float) -> str:
    """
    Classify attractor type based on dimension and Lyapunov exponent.
    """
    if np.isnan(correlation_dim) or np.isnan(lyapunov):
        return "unknown"

    # Fixed point: dim ~ 0, lyapunov < 0
    if correlation_dim < 0.5 and lyapunov < 0:
        return "fixed_point"

    # Limit cycle: dim ~ 1, lyapunov ~ 0
    if correlation_dim < 1.5 and abs(lyapunov) < 0.1:
        return "limit_cycle"

    # Torus: dim ~ 2, lyapunov ~ 0
    if 1.5 <= correlation_dim < 2.5 and abs(lyapunov) < 0.1:
        return "torus"

    # Strange attractor: fractal dim, lyapunov > 0
    if correlation_dim > 1.5 and lyapunov > 0.1:
        return "strange"

    # Noise: high dim, lyapunov > 0
    if correlation_dim > 4 and lyapunov > 0:
        return "noise"

    return "indeterminate"


def _empty_result(reason: str) -> Dict[str, Any]:
    """Return empty result with reason."""
    return {
        "embedding_dimension": 0,
        "time_delay": 0,
        "correlation_dimension": np.nan,
        "largest_lyapunov": np.nan,
        "attractor_type": "unknown",
        "phase_space_volume": np.nan,
        "mean_distance_from_centroid": np.nan,
        "max_distance_from_centroid": np.nan,
        "recurrence_rate": np.nan,
        "embedding_length": 0,
        "n_observations": 0,
        "error": reason,
    }


def compute_embedding_quality(
    values: np.ndarray,
    dim: int,
    tau: int
) -> Dict[str, Any]:
    """
    Assess quality of a specific embedding configuration.

    Args:
        values: 1D array of time series values
        dim: Embedding dimension
        tau: Time delay

    Returns:
        Dictionary with quality metrics
    """
    values = np.asarray(values, dtype=np.float64)
    values = (values - np.mean(values)) / (np.std(values) + 1e-10)

    embedding = _construct_embedding(values, dim, tau)

    if len(embedding) < 50:
        return {"error": "Embedding too short"}

    # False nearest neighbors
    fnn = _false_nearest_neighbors(embedding, values, dim, tau)

    # Determinism (from recurrence)
    det = _determinism(embedding)

    # Laminarity
    lam = _laminarity(embedding)

    return {
        "embedding_dimension": dim,
        "time_delay": tau,
        "embedding_length": len(embedding),
        "false_nearest_neighbors": float(fnn),
        "determinism": float(det),
        "laminarity": float(lam),
        "quality_score": float(1 - fnn) * float(det),
    }


def _determinism(embedding: np.ndarray, threshold: float = 0.1) -> float:
    """
    Compute determinism from recurrence plot diagonal structures.
    """
    # Simplified: ratio of points on diagonals to total recurrences
    n = len(embedding)
    if n < 20:
        return None

    n_sample = min(200, n)
    indices = np.random.choice(n, n_sample, replace=False)
    sample = embedding[indices]

    # Build recurrence matrix
    dists = distance.cdist(sample, sample)
    eps = threshold * np.std(dists)
    R = (dists < eps).astype(int)

    # Count diagonal points (excluding main diagonal)
    diag_points = 0
    total_recur = np.sum(R) - n_sample  # Exclude main diagonal

    for k in range(1, n_sample):
        diag = np.diag(R, k)
        diag_points += np.sum(diag)

    return diag_points / total_recur if total_recur > 0 else None


def _laminarity(embedding: np.ndarray, threshold: float = 0.1) -> float:
    """
    Compute laminarity from recurrence plot vertical structures.
    """
    n = len(embedding)
    if n < 20:
        return None

    n_sample = min(200, n)
    indices = np.random.choice(n, n_sample, replace=False)
    sample = embedding[indices]

    # Build recurrence matrix
    dists = distance.cdist(sample, sample)
    eps = threshold * np.std(dists)
    R = (dists < eps).astype(int)

    # Count vertical line points
    vert_points = 0
    total_recur = np.sum(R) - n_sample

    for col in range(n_sample):
        # Find vertical lines
        line_len = 0
        for row in range(n_sample):
            if R[row, col] == 1 and row != col:
                line_len += 1
            else:
                if line_len >= 2:
                    vert_points += line_len
                line_len = 0
        if line_len >= 2:
            vert_points += line_len

    return vert_points / total_recur if total_recur > 0 else None
