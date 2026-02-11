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

    # Minimum viable signal length
    if n < 50:
        return np.nan, np.array([]), np.array([])

    # Auto-detect parameters
    if delay is None:
        delay = _auto_delay(signal)
    if dimension is None:
        dimension = _auto_dimension(signal, delay)

    # Validate embedding parameters and adjust if needed
    min_embed_points = max(50, n // 4)  # Need enough points for statistics
    is_valid, dimension, delay, msg = _validate_embedding_params(
        n, dimension, delay, min_embed_points
    )

    if not is_valid:
        # Signal too short for Lyapunov analysis
        return np.nan, np.array([]), np.array([])

    if min_tsep is None:
        min_tsep = delay * dimension
    if max_iter is None:
        max_iter = min(n // 10, 500)

    # Embed (now safe after validation)
    try:
        embedded = _embed(signal, dimension, delay)
    except ValueError:
        return np.nan, np.array([]), np.array([])

    n_points = len(embedded)

    if n_points < min_tsep + max_iter + 10:
        # Reduce max_iter to fit available points
        max_iter = max(10, n_points - min_tsep - 10)
        if max_iter < 10:
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

    # Minimum viable signal length
    if n < 50:
        return np.nan, np.array([])

    # Auto-detect parameters
    if delay is None:
        delay = _auto_delay(signal)
    if dimension is None:
        dimension = _auto_dimension(signal, delay)

    # Validate embedding parameters and adjust if needed
    min_embed_points = max(50, n // 4)
    is_valid, dimension, delay, msg = _validate_embedding_params(
        n, dimension, delay, min_embed_points
    )

    if not is_valid:
        return np.nan, np.array([])

    if min_tsep is None:
        min_tsep = delay * dimension
    if max_iter is None:
        max_iter = min(n // 10, 500)

    # Embed (now safe after validation)
    try:
        embedded = _embed(signal, dimension, delay)
    except ValueError:
        return np.nan, np.array([])

    n_points = len(embedded)

    if n_points < min_tsep + max_iter + 10:
        max_iter = max(10, n_points - min_tsep - 10)
        if max_iter < 10:
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
    n = len(signal)

    if n_exponents is None:
        n_exponents = dimension

    # Validate embedding parameters
    is_valid, adj_dim, adj_delay, msg = _validate_embedding_params(
        n, dimension, delay, min_points=100
    )

    if not is_valid:
        return np.full(n_exponents, np.nan)

    # Use adjusted parameters if needed
    dimension = adj_dim
    delay = adj_delay

    # Embed (now safe after validation)
    try:
        embedded = _embed(signal, dimension, delay)
    except ValueError:
        return np.full(n_exponents, np.nan)

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


# =============================================================================
# Embedding Dimension Estimation
# =============================================================================

def estimate_embedding_dim_cao(
    signal: np.ndarray,
    max_dim: int = 10,
    tau: int = None
) -> dict:
    """
    Cao's method for embedding dimension estimation.

    Preferred over False Nearest Neighbors (FNN) because:
    - FNN requires a threshold parameter (arbitrary)
    - Cao's is parameter-free
    - Cao's E2 statistic distinguishes deterministic from stochastic signals

    Parameters
    ----------
    signal : np.ndarray
        1D time series
    max_dim : int
        Maximum embedding dimension to test
    tau : int, optional
        Time delay (auto-detected if None)

    Returns
    -------
    dict with:
        optimal_dim : int
            Estimated embedding dimension
        is_deterministic : bool
            True if E2 indicates determinism (FTLE meaningful)
        E1_values : np.ndarray
            E1(d) for d=1..max_dim
        E2_values : np.ndarray
            E2(d) for d=1..max_dim
        E1_ratio : np.ndarray
            E1(d+1)/E1(d) - saturates at correct dimension
        confidence : float
            Confidence in the estimate (0-1)

    Notes
    -----
    E2 distinguishes deterministic from stochastic:
    - Deterministic: E2 ≠ 1 for some d
    - Stochastic: E2 ≈ 1 for all d

    If E2 ≈ 1 for all d, FTLE is not meaningful for this signal.
    """
    signal = np.asarray(signal).flatten()
    n = len(signal)

    if n < 50:
        return {
            'optimal_dim': 3,
            'is_deterministic': None,
            'E1_values': None,
            'E2_values': None,
            'E1_ratio': None,
            'confidence': 0.0,
        }

    if tau is None:
        tau = _auto_delay(signal)

    E1 = np.zeros(max_dim)
    E2 = np.zeros(max_dim)

    for d in range(1, max_dim + 1):
        # Check if embedding is possible before trying
        n_points_d = n - (d - 1) * tau
        n_points_d1 = n - d * tau

        if n_points_d < 10 or n_points_d1 < 10:
            E1[d - 1] = np.nan
            E2[d - 1] = np.nan
            continue

        try:
            embedded_d = _embed(signal, d, tau)
            embedded_d1 = _embed(signal, d + 1, tau)
        except ValueError:
            E1[d - 1] = np.nan
            E2[d - 1] = np.nan
            continue

        N = min(len(embedded_d), len(embedded_d1))

        if N < 10:
            E1[d - 1] = np.nan
            E2[d - 1] = np.nan
            continue

        a_values = []
        a_star_values = []

        # Subsample for efficiency
        sample_size = min(500, N)
        sample_idx = np.random.choice(N, sample_size, replace=False)

        for i in sample_idx:
            # Find nearest neighbor in d-dimensional space using Chebyshev distance
            min_dist = np.inf
            nn_idx = -1

            # Compare against other samples (not full N for speed)
            for j in sample_idx:
                if i == j:
                    continue
                dist = np.max(np.abs(embedded_d[i] - embedded_d[j]))
                if 0 < dist < min_dist:
                    min_dist = dist
                    nn_idx = j

            if nn_idx < 0 or min_dist == 0:
                continue

            # a(i, d) = distance in (d+1)-space / distance in d-space
            dist_d1 = np.max(np.abs(embedded_d1[i] - embedded_d1[nn_idx]))
            a_values.append(dist_d1 / (min_dist + 1e-12))

            # a*(i, d) for E2 calculation
            idx_d1 = min(i + d * tau, len(signal) - 1)
            idx_d1_nn = min(nn_idx + d * tau, len(signal) - 1)
            a_star_values.append(abs(signal[idx_d1] - signal[idx_d1_nn]))

        E1[d - 1] = np.mean(a_values) if a_values else np.nan
        E2[d - 1] = np.mean(a_star_values) if a_star_values else np.nan

    # E1 ratio: E1(d+1) / E1(d)
    # Stops changing when correct dimension is reached
    E1_ratio = np.zeros(max_dim - 1)
    for d in range(1, max_dim):
        if E1[d - 1] > 1e-10:
            E1_ratio[d - 1] = E1[d] / E1[d - 1]
        else:
            E1_ratio[d - 1] = np.nan

    # Find where E1_ratio saturates (stops changing significantly)
    threshold = 0.95
    optimal_dim = max_dim
    for d in range(len(E1_ratio)):
        if np.isfinite(E1_ratio[d]) and E1_ratio[d] > threshold:
            optimal_dim = d + 1
            break

    # E2 distinguishes deterministic from stochastic
    # For deterministic: E2 varies with d
    # For stochastic: E2 ≈ constant for all d
    E2_valid = E2[np.isfinite(E2)]
    if len(E2_valid) >= 2:
        E2_std = np.std(E2_valid)
        E2_mean = np.mean(E2_valid)
        # If E2 varies significantly, signal is deterministic
        is_deterministic = E2_std / (E2_mean + 1e-12) > 0.1
    else:
        is_deterministic = None

    # Confidence based on how sharp the saturation is
    if len(E1_ratio[np.isfinite(E1_ratio)]) >= 2:
        ratio_std = np.std(E1_ratio[np.isfinite(E1_ratio)])
        confidence = min(1.0, 1.0 / (ratio_std + 0.1))
    else:
        confidence = 0.5

    return {
        'optimal_dim': optimal_dim,
        'is_deterministic': is_deterministic,
        'E1_values': E1,
        'E2_values': E2,
        'E1_ratio': E1_ratio,
        'confidence': float(confidence),
    }


def estimate_tau_ami(
    signal: np.ndarray,
    max_tau: int = 50,
    n_bins: int = 64
) -> int:
    """
    Estimate embedding delay using Average Mutual Information.

    First minimum of AMI gives optimal tau.
    Better than autocorrelation for nonlinear systems.

    Parameters
    ----------
    signal : np.ndarray
        1D time series
    max_tau : int
        Maximum delay to test
    n_bins : int
        Number of bins for histogram

    Returns
    -------
    int
        Optimal embedding delay
    """
    signal = np.asarray(signal).flatten()
    n = len(signal)

    if n < max_tau + 10:
        return 1

    ami_values = []

    for tau in range(1, min(max_tau + 1, n // 2)):
        x = signal[:-tau]
        y = signal[tau:]

        # 2D histogram
        try:
            hist_xy, _, _ = np.histogram2d(x, y, bins=n_bins)
            hist_x, _ = np.histogram(x, bins=n_bins)
            hist_y, _ = np.histogram(y, bins=n_bins)
        except Exception:
            ami_values.append(np.nan)
            continue

        # Normalize to probabilities
        p_xy = hist_xy / hist_xy.sum()
        p_x = hist_x / hist_x.sum()
        p_y = hist_y / hist_y.sum()

        # Mutual information
        mi = 0
        for i in range(n_bins):
            for j in range(n_bins):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))

        ami_values.append(mi)

    if not ami_values:
        return 1

    ami_arr = np.array(ami_values)

    # Find first local minimum
    for i in range(1, len(ami_arr) - 1):
        if np.isfinite(ami_arr[i]):
            if ami_arr[i] < ami_arr[i - 1] and ami_arr[i] <= ami_arr[i + 1]:
                return i + 1  # tau (1-indexed)

    # No minimum found — use first below 1/e of initial
    if np.isfinite(ami_arr[0]):
        threshold = ami_arr[0] / np.e
        below = np.where(ami_arr < threshold)[0]
        if len(below) > 0:
            return int(below[0]) + 1

    return max_tau // 4  # fallback


# =============================================================================
# Helper functions
# =============================================================================

def _validate_embedding_params(
    n: int,
    dimension: int,
    delay: int,
    min_points: int = 10
) -> Tuple[bool, int, int, str]:
    """
    Validate embedding parameters and suggest adjustments if invalid.

    Parameters
    ----------
    n : int
        Signal length
    dimension : int
        Proposed embedding dimension
    delay : int
        Proposed time delay
    min_points : int
        Minimum embedded points required

    Returns
    -------
    tuple
        (is_valid, adjusted_dimension, adjusted_delay, message)
        If is_valid is False and adjustments can't help, dimension=0.
    """
    n_points = n - (dimension - 1) * delay

    if n_points >= min_points:
        return True, dimension, delay, "OK"

    # Try reducing dimension first (preserves temporal structure)
    for new_dim in range(dimension - 1, 1, -1):
        new_n_points = n - (new_dim - 1) * delay
        if new_n_points >= min_points:
            return True, new_dim, delay, f"Reduced dimension {dimension}→{new_dim}"

    # Try reducing delay
    for new_delay in range(delay - 1, 0, -1):
        new_n_points = n - (dimension - 1) * new_delay
        if new_n_points >= min_points:
            return True, dimension, new_delay, f"Reduced delay {delay}→{new_delay}"

    # Try reducing both
    for new_dim in range(dimension - 1, 1, -1):
        for new_delay in range(delay - 1, 0, -1):
            new_n_points = n - (new_dim - 1) * new_delay
            if new_n_points >= min_points:
                return True, new_dim, new_delay, f"Reduced dim {dimension}→{new_dim}, delay {delay}→{new_delay}"

    # Signal too short for any valid embedding
    return False, 0, 0, f"Signal too short ({n} samples) for Lyapunov analysis"


def _embed(signal: np.ndarray, dimension: int, delay: int) -> np.ndarray:
    """
    Time delay embedding.

    Raises ValueError if parameters would result in negative embedding length.
    This happens when (dimension - 1) * delay >= len(signal).
    """
    n = len(signal)
    n_points = n - (dimension - 1) * delay

    if n_points <= 0:
        raise ValueError(
            f"Cannot embed: signal length {n} is too short for "
            f"dimension={dimension}, delay={delay}. "
            f"Need at least {(dimension - 1) * delay + 1} samples."
        )

    embedded = np.zeros((n_points, dimension))
    for d in range(dimension):
        embedded[:, d] = signal[d * delay : d * delay + n_points]
    return embedded


def _auto_delay(signal: np.ndarray, method: str = 'ami') -> int:
    """
    Auto-detect delay.

    Parameters
    ----------
    signal : np.ndarray
        1D time series
    method : str
        'ami' for Average Mutual Information (default, better for nonlinear)
        'acf' for autocorrelation 1/e decay (faster, linear systems)

    Returns
    -------
    int
        Optimal embedding delay
    """
    if method == 'ami':
        return estimate_tau_ami(signal)

    # Fallback: autocorrelation 1/e decay
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


def _auto_dimension(signal: np.ndarray, delay: int, method: str = 'cao') -> int:
    """
    Auto-detect embedding dimension.

    Parameters
    ----------
    signal : np.ndarray
        1D time series
    delay : int
        Time delay
    method : str
        'cao' for Cao's method (default, parameter-free)
        'fnn' for False Nearest Neighbors (faster)

    Returns
    -------
    int
        Optimal embedding dimension
    """
    if method == 'cao':
        result = estimate_embedding_dim_cao(signal, tau=delay)
        return result['optimal_dim']

    # Fallback: FNN
    for dim in range(2, min(11, len(signal) // (3 * delay))):
        fnn_ratio = _fnn_ratio(signal, dim, delay)
        if fnn_ratio < 0.01:
            return dim
    return 5  # Default


def _fnn_ratio(signal: np.ndarray, dimension: int, delay: int) -> float:
    """Compute false nearest neighbor ratio."""
    from scipy.spatial import KDTree

    # Check if embedding is possible
    n = len(signal)
    if n - dimension * delay < 10:
        return 1.0

    try:
        emb = _embed(signal, dimension, delay)
        emb_next = _embed(signal, dimension + 1, delay)
    except ValueError:
        return 1.0

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
