"""
Dynamical Lyapunov Primitives (86-87)

Lyapunov exponent estimation.
"""

import numpy as np
from typing import Tuple, Optional


def lyapunov_rosenstein(
    signal: np.ndarray,
    dimension: int = None,
    delay: int = None,
    min_tsep: int = None,
    max_iter: int = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate largest Lyapunov exponent using Rosenstein's algorithm.

    Parameters
    ----------
    signal : np.ndarray
        1D time series
    dimension : int, optional
        Embedding dimension (default: auto-detected)
    delay : int, optional
        Time delay (default: auto-detected)
    min_tsep : int, optional
        Minimum temporal separation for neighbors
    max_iter : int, optional
        Maximum number of iterations

    Returns
    -------
    tuple
        (lambda_max, divergence, iterations)
        lambda_max: Largest Lyapunov exponent
        divergence: Mean divergence curve
        iterations: Iteration indices

    Notes
    -----
    λ_max > 0: chaotic
    λ_max ≈ 0: quasi-periodic / edge of chaos
    λ_max < 0: periodic / fixed point

    Algorithm:
    1. Embed the signal
    2. For each point, find nearest neighbor (excluding temporal neighbors)
    3. Track divergence over time
    4. Fit slope to log(divergence) vs time
    """
    from scipy.spatial import KDTree

    signal = np.asarray(signal).flatten()
    n = len(signal)

    # Auto-detect parameters
    if delay is None:
        delay = _auto_delay(signal)
    if dimension is None:
        dimension = _auto_dimension(signal, delay)
    if min_tsep is None:
        min_tsep = delay * dimension
    if max_iter is None:
        max_iter = min(n // 10, 500)

    # Embed
    embedded = _embed(signal, dimension, delay)
    n_points = len(embedded)

    if n_points < min_tsep + max_iter + 10:
        return np.nan, np.array([]), np.array([])

    # Find nearest neighbors using KDTree (O(n log n) instead of O(n²))
    # Query enough neighbors to find one outside temporal exclusion zone
    tree = KDTree(embedded)
    k_query = min(min_tsep + 10, n_points)  # Query enough to find valid neighbor

    nn_indices = np.full(n_points, -1, dtype=int)
    nn_dists = np.full(n_points, np.inf)

    # Query all points at once for efficiency
    dists_all, indices_all = tree.query(embedded, k=k_query)

    for i in range(n_points):
        # Find first neighbor outside temporal exclusion zone
        for k in range(k_query):
            j = indices_all[i, k]
            if abs(i - j) >= min_tsep and dists_all[i, k] > 0:
                nn_indices[i] = j
                nn_dists[i] = dists_all[i, k]
                break

    # Track divergence
    divergence = np.zeros(max_iter)
    counts = np.zeros(max_iter)

    for i in range(n_points - max_iter):
        j = nn_indices[i]
        if j < 0 or j >= n_points - max_iter:
            continue

        for k in range(max_iter):
            if i + k < n_points and j + k < n_points:
                dist = np.linalg.norm(embedded[i + k] - embedded[j + k])
                if dist > 0:
                    divergence[k] += np.log(dist)
                    counts[k] += 1

    # Average divergence
    valid = counts > 0
    divergence[valid] = divergence[valid] / counts[valid]
    divergence[~valid] = np.nan

    iterations = np.arange(max_iter)

    # Fit slope to initial linear region
    # Use first 10-30% where growth is approximately linear
    fit_end = max(10, max_iter // 5)
    fit_mask = np.isfinite(divergence[:fit_end])

    if np.sum(fit_mask) < 3:
        return np.nan, divergence, iterations

    x = iterations[:fit_end][fit_mask]
    y = divergence[:fit_end][fit_mask]

    # Linear regression
    slope, _ = np.polyfit(x, y, 1)
    lambda_max = slope / delay  # Convert to per-sample rate

    return float(lambda_max), divergence, iterations


def lyapunov_kantz(
    signal: np.ndarray,
    dimension: int = None,
    delay: int = None,
    min_tsep: int = None,
    epsilon: float = None,
    max_iter: int = None
) -> Tuple[float, np.ndarray]:
    """
    Estimate largest Lyapunov exponent using Kantz's algorithm.

    Parameters
    ----------
    signal : np.ndarray
        1D time series
    dimension : int, optional
        Embedding dimension
    delay : int, optional
        Time delay
    min_tsep : int, optional
        Minimum temporal separation
    epsilon : float, optional
        Neighborhood radius (default: auto)
    max_iter : int, optional
        Maximum iterations

    Returns
    -------
    tuple
        (lambda_max, divergence)

    Notes
    -----
    Similar to Rosenstein but averages over all neighbors within ε,
    not just the nearest neighbor. More robust but slower.
    """
    signal = np.asarray(signal).flatten()
    n = len(signal)

    # Auto-detect parameters
    if delay is None:
        delay = _auto_delay(signal)
    if dimension is None:
        dimension = _auto_dimension(signal, delay)
    if min_tsep is None:
        min_tsep = delay * dimension
    if max_iter is None:
        max_iter = min(n // 10, 500)

    # Embed
    embedded = _embed(signal, dimension, delay)
    n_points = len(embedded)

    if n_points < min_tsep + max_iter + 10:
        return np.nan, np.array([])

    # Auto epsilon
    if epsilon is None:
        dists = []
        sample_idx = np.random.choice(n_points, min(100, n_points), replace=False)
        for i in sample_idx:
            for j in sample_idx:
                if i != j:
                    dists.append(np.linalg.norm(embedded[i] - embedded[j]))
        epsilon = np.percentile(dists, 10)

    # Track divergence
    divergence = np.zeros(max_iter)
    counts = np.zeros(max_iter)

    for i in range(n_points - max_iter):
        # Find all neighbors within epsilon
        neighbors = []
        for j in range(n_points):
            if abs(i - j) >= min_tsep:
                dist = np.linalg.norm(embedded[i] - embedded[j])
                if 0 < dist < epsilon:
                    neighbors.append(j)

        if len(neighbors) == 0:
            continue

        # Track average divergence from all neighbors
        for k in range(max_iter):
            if i + k >= n_points:
                break

            neighbor_dists = []
            for j in neighbors:
                if j + k < n_points:
                    dist = np.linalg.norm(embedded[i + k] - embedded[j + k])
                    if dist > 0:
                        neighbor_dists.append(np.log(dist))

            if neighbor_dists:
                divergence[k] += np.mean(neighbor_dists)
                counts[k] += 1

    # Average
    valid = counts > 0
    divergence[valid] = divergence[valid] / counts[valid]
    divergence[~valid] = np.nan

    # Fit slope
    fit_end = max(10, max_iter // 5)
    iterations = np.arange(max_iter)
    fit_mask = np.isfinite(divergence[:fit_end])

    if np.sum(fit_mask) < 3:
        return np.nan, divergence

    x = iterations[:fit_end][fit_mask]
    y = divergence[:fit_end][fit_mask]

    slope, _ = np.polyfit(x, y, 1)
    lambda_max = slope / delay

    return float(lambda_max), divergence


def lyapunov_spectrum(
    signal: np.ndarray,
    dimension: int = 3,
    delay: int = 1,
    n_exponents: int = None
) -> np.ndarray:
    """
    Estimate Lyapunov spectrum using QR decomposition method.

    Parameters
    ----------
    signal : np.ndarray
        1D time series
    dimension : int
        Embedding dimension
    delay : int
        Time delay
    n_exponents : int, optional
        Number of exponents to return (default: dimension)

    Returns
    -------
    np.ndarray
        Lyapunov exponents (sorted descending)

    Notes
    -----
    Full spectrum provides more information:
    - Sum of all exponents = system's volume contraction rate
    - Kaplan-Yorke dimension = j + Σλ_i / |λ_{j+1}|

    Warning: Spectrum estimation from scalar time series is difficult
    and may not be reliable. Use with caution.
    """
    signal = np.asarray(signal).flatten()

    if n_exponents is None:
        n_exponents = dimension

    # Embed
    embedded = _embed(signal, dimension, delay)
    n_points = len(embedded)

    if n_points < 100:
        return np.full(n_exponents, np.nan)

    # Initialize orthonormal basis
    Q = np.eye(dimension)
    lyap_sums = np.zeros(dimension)

    # Iterate through trajectory
    n_iter = n_points - 1

    for i in range(n_iter):
        # Estimate local Jacobian using finite differences
        # This is approximate - proper Jacobian requires model
        J = _estimate_jacobian(embedded, i, dimension)

        if np.any(np.isnan(J)):
            continue

        # Evolve tangent vectors
        W = J @ Q

        # QR decomposition
        Q, R = np.linalg.qr(W)

        # Accumulate log of stretching factors
        for k in range(dimension):
            if np.abs(R[k, k]) > 1e-10:
                lyap_sums[k] += np.log(np.abs(R[k, k]))

    # Average
    exponents = lyap_sums / n_iter

    # Sort descending
    exponents = np.sort(exponents)[::-1]

    return exponents[:n_exponents]


# Helper functions

def _embed(signal: np.ndarray, dimension: int, delay: int) -> np.ndarray:
    """Time delay embedding."""
    n = len(signal)
    n_points = n - (dimension - 1) * delay
    embedded = np.zeros((n_points, dimension))
    for d in range(dimension):
        embedded[:, d] = signal[d * delay : d * delay + n_points]
    return embedded


def _auto_delay(signal: np.ndarray) -> int:
    """Auto-detect delay using autocorrelation 1/e decay."""
    n = len(signal)
    signal_centered = signal - np.mean(signal)
    var = np.var(signal_centered)
    if var == 0:
        return 1

    for lag in range(1, n // 4):
        acf = np.mean(signal_centered[:-lag] * signal_centered[lag:]) / var
        if acf < 1 / np.e:
            return lag

    return n // 10


def _auto_dimension(signal: np.ndarray, delay: int) -> int:
    """Auto-detect embedding dimension using false nearest neighbors."""
    # Simplified: try dimensions 2-10, pick where FNN drops
    for dim in range(2, min(11, len(signal) // (3 * delay))):
        fnn_ratio = _fnn_ratio(signal, dim, delay)
        if fnn_ratio < 0.01:
            return dim
    return 5  # Default


def _fnn_ratio(signal: np.ndarray, dimension: int, delay: int) -> float:
    """Compute false nearest neighbor ratio."""
    from scipy.spatial import KDTree

    emb = _embed(signal, dimension, delay)
    emb_next = _embed(signal, dimension + 1, delay)

    n_points = min(len(emb), len(emb_next))
    emb = emb[:n_points]
    emb_next = emb_next[:n_points]

    if n_points < 10:
        return 1.0

    tree = KDTree(emb)
    n_false = 0

    for i in range(min(500, n_points)):
        dists, indices = tree.query(emb[i], k=2)
        if len(indices) < 2:
            continue

        j = indices[1]
        r_d = dists[1]

        if r_d < 1e-10:
            continue

        r_d1 = np.linalg.norm(emb_next[i] - emb_next[j])

        if r_d1 / r_d > 10:
            n_false += 1

    return n_false / min(500, n_points)


def _estimate_jacobian(
    embedded: np.ndarray,
    i: int,
    dimension: int
) -> np.ndarray:
    """Estimate local Jacobian from embedded trajectory."""
    n_points = len(embedded)

    # Find nearby points
    from scipy.spatial import KDTree
    tree = KDTree(embedded)

    n_neighbors = min(2 * dimension + 1, n_points - 1)
    dists, indices = tree.query(embedded[i], k=n_neighbors + 1)

    # Exclude self and last point
    valid = [j for j in indices[1:] if j < n_points - 1]

    if len(valid) < dimension + 1:
        return np.full((dimension, dimension), np.nan)

    # Linear regression: x_{j+1} - x_{i+1} ≈ J @ (x_j - x_i)
    X = np.zeros((len(valid), dimension))
    Y = np.zeros((len(valid), dimension))

    for k, j in enumerate(valid):
        X[k] = embedded[j] - embedded[i]
        Y[k] = embedded[j + 1] - embedded[i + 1]

    # Least squares: Y = X @ J.T
    try:
        J_T = np.linalg.lstsq(X, Y, rcond=None)[0]
        return J_T.T
    except:
        return np.full((dimension, dimension), np.nan)
