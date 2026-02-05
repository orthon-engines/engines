"""
PRISM Dynamical Systems Primitives

Pure mathematical functions for analyzing dynamical systems,
chaos, and attractor reconstruction.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import Tuple, Optional


def lyapunov_exponent(
    values: np.ndarray,
    embed_dim: int = 3,
    tau: int = 1,
    min_separation: int = 10,
    max_steps: int = 10
) -> float:
    """
    Estimate largest Lyapunov exponent using Rosenstein's method.

    Positive Lyapunov exponent indicates chaos.
    Negative indicates stability.
    Zero indicates periodicity or quasi-periodicity.

    Args:
        values: Input time series
        embed_dim: Embedding dimension
        tau: Time delay for embedding
        min_separation: Minimum separation between neighbors
        max_steps: Maximum divergence steps to track

    Returns:
        Largest Lyapunov exponent
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    # Embed the time series
    n_vectors = n - (embed_dim - 1) * tau
    if n_vectors < 2 * min_separation:
        return np.nan

    embedded = np.array([
        values[i:i + embed_dim * tau:tau]
        for i in range(n_vectors)
    ])

    # Find nearest neighbors (excluding temporal neighbors)
    distances = squareform(pdist(embedded))

    # Set temporal neighbors to infinity
    for i in range(len(distances)):
        for j in range(max(0, i - min_separation), min(len(distances), i + min_separation + 1)):
            distances[i, j] = np.inf

    # Find nearest neighbor for each point
    nearest = np.argmin(distances, axis=1)

    # Track divergence
    divergences = []
    for step in range(1, min(max_steps, n_vectors - max(nearest) - 1)):
        step_divergences = []
        for i in range(len(nearest)):
            j = nearest[i]
            if i + step < n_vectors and j + step < n_vectors:
                d0 = distances[i, j]
                d1 = np.linalg.norm(embedded[i + step] - embedded[j + step])
                if d0 > 0 and d1 > 0:
                    step_divergences.append(np.log(d1 / d0))

        if len(step_divergences) > 0:
            divergences.append(np.mean(step_divergences))

    if len(divergences) < 2:
        return np.nan

    # Fit line to estimate Lyapunov exponent
    steps = np.arange(1, len(divergences) + 1)
    slope, _ = np.polyfit(steps, divergences, 1)

    return float(slope)


def largest_lyapunov_exponent(
    values: np.ndarray,
    **kwargs
) -> float:
    """
    Alias for lyapunov_exponent.

    Args:
        values: Input time series
        **kwargs: Arguments passed to lyapunov_exponent

    Returns:
        Largest Lyapunov exponent
    """
    return lyapunov_exponent(values, **kwargs)


def attractor_reconstruction(
    values: np.ndarray,
    embed_dim: int = 3,
    tau: int = 1
) -> np.ndarray:
    """
    Reconstruct attractor using time-delay embedding.

    Takens' embedding theorem: delay embedding preserves
    topological properties of the original attractor.

    Args:
        values: Input time series
        embed_dim: Embedding dimension
        tau: Time delay

    Returns:
        Embedded vectors (n_vectors × embed_dim)
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    n_vectors = n - (embed_dim - 1) * tau
    if n_vectors <= 0:
        return np.array([])

    embedded = np.array([
        values[i:i + embed_dim * tau:tau]
        for i in range(n_vectors)
    ])

    return embedded


def embedding_dimension(
    values: np.ndarray,
    max_dim: int = 10,
    tau: int = 1,
    rtol: float = 10.0
) -> int:
    """
    Estimate optimal embedding dimension using false nearest neighbors.

    Args:
        values: Input time series
        max_dim: Maximum dimension to test
        tau: Time delay
        rtol: Threshold ratio for false neighbors

    Returns:
        Optimal embedding dimension
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    fnn_ratios = []

    for dim in range(1, max_dim):
        n_vectors = n - dim * tau
        if n_vectors < 100:
            break

        # Embed in dimension dim
        embedded = np.array([
            values[i:i + dim * tau:tau]
            for i in range(n_vectors)
        ])

        # Find nearest neighbors
        distances = squareform(pdist(embedded))
        np.fill_diagonal(distances, np.inf)
        nearest = np.argmin(distances, axis=1)

        # Check if neighbors are false
        false_neighbors = 0
        total_neighbors = 0

        for i in range(len(embedded) - tau):
            j = nearest[i]
            if j + tau >= n_vectors:
                continue

            d_k = distances[i, j]
            if d_k == 0:
                continue

            # Distance in next dimension
            d_k1 = abs(values[i + dim * tau] - values[j + dim * tau])

            # Check if false neighbor
            if d_k1 / d_k > rtol:
                false_neighbors += 1
            total_neighbors += 1

        if total_neighbors > 0:
            fnn_ratios.append(false_neighbors / total_neighbors)
        else:
            fnn_ratios.append(1.0)

    # Find dimension where FNN drops below threshold
    for dim, fnn in enumerate(fnn_ratios, start=1):
        if fnn < 0.01:  # Less than 1% false neighbors
            return dim

    return max_dim


def optimal_delay(
    values: np.ndarray,
    max_lag: Optional[int] = None,
    method: str = 'mutual_info'
) -> int:
    """
    Estimate optimal time delay for embedding.

    Args:
        values: Input time series
        max_lag: Maximum lag to consider
        method: 'mutual_info' or 'autocorr'

    Returns:
        Optimal delay
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    if max_lag is None:
        max_lag = n // 4

    if method == 'autocorr':
        # First zero crossing or first minimum of autocorrelation
        acf = np.correlate(values - np.mean(values), values - np.mean(values), mode='full')
        acf = acf[n - 1:]
        acf = acf / acf[0]

        # Find first minimum
        for i in range(1, min(max_lag, len(acf) - 1)):
            if acf[i] < acf[i - 1] and acf[i] < acf[i + 1]:
                return i

        # Or first zero crossing
        for i in range(1, min(max_lag, len(acf))):
            if acf[i] <= 0:
                return i

        return 1

    elif method == 'mutual_info':
        # First minimum of mutual information
        from .similarity import mutual_information

        mi_values = []
        for lag in range(1, max_lag + 1):
            mi = mutual_information(values[:-lag], values[lag:])
            mi_values.append(mi)

        # Find first minimum
        for i in range(1, len(mi_values) - 1):
            if mi_values[i] < mi_values[i - 1] and mi_values[i] < mi_values[i + 1]:
                return i + 1

        return 1

    else:
        raise ValueError(f"Unknown method: {method}")


def recurrence_analysis(
    values: np.ndarray,
    embed_dim: int = 3,
    tau: int = 1,
    threshold: Optional[float] = None
) -> dict:
    """
    Perform recurrence quantification analysis (RQA).

    Args:
        values: Input time series
        embed_dim: Embedding dimension
        tau: Time delay
        threshold: Recurrence threshold (if None, use 10% of max distance)

    Returns:
        Dictionary with RQA measures:
        - recurrence_rate: Fraction of recurrent points
        - determinism: Fraction of recurrent points in diagonal lines
        - laminarity: Fraction of recurrent points in vertical lines
        - mean_diagonal: Average diagonal line length
        - entropy: Shannon entropy of diagonal line distribution
    """
    values = np.asarray(values, dtype=np.float64)

    # Embed
    embedded = attractor_reconstruction(values, embed_dim, tau)

    if len(embedded) < 10:
        return {
            'recurrence_rate': np.nan,
            'determinism': np.nan,
            'laminarity': np.nan,
            'mean_diagonal': np.nan,
            'entropy': np.nan
        }

    # Distance matrix
    distances = squareform(pdist(embedded))

    # Set threshold
    if threshold is None:
        threshold = 0.1 * np.max(distances)

    # Recurrence matrix
    recurrence = distances < threshold

    # Recurrence rate
    n = len(recurrence)
    rr = np.sum(recurrence) / (n * n)

    # Find diagonal lines (excluding main diagonal)
    diag_lengths = []
    for k in range(-n + 2, n - 1):  # Skip main diagonal (k=0)
        diag = np.diag(recurrence, k)
        length = 0
        for val in diag:
            if val:
                length += 1
            elif length >= 2:
                diag_lengths.append(length)
                length = 0
        if length >= 2:
            diag_lengths.append(length)

    # Determinism
    if len(diag_lengths) > 0 and np.sum(recurrence) > 0:
        det = np.sum(diag_lengths) / np.sum(recurrence)
    else:
        det = 0.0

    # Mean diagonal length
    mean_diag = np.mean(diag_lengths) if len(diag_lengths) > 0 else 0.0

    # Entropy of diagonal distribution
    if len(diag_lengths) > 0:
        unique, counts = np.unique(diag_lengths, return_counts=True)
        probs = counts / np.sum(counts)
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
    else:
        entropy = 0.0

    # Find vertical lines
    vert_lengths = []
    for col in range(n):
        length = 0
        for row in range(n):
            if recurrence[row, col]:
                length += 1
            elif length >= 2:
                vert_lengths.append(length)
                length = 0
        if length >= 2:
            vert_lengths.append(length)

    # Laminarity
    if len(vert_lengths) > 0 and np.sum(recurrence) > 0:
        lam = np.sum(vert_lengths) / np.sum(recurrence)
    else:
        lam = 0.0

    return {
        'recurrence_rate': float(rr),
        'determinism': float(det),
        'laminarity': float(lam),
        'mean_diagonal': float(mean_diag),
        'entropy': float(entropy)
    }


def poincare_map(
    values: np.ndarray,
    section_value: Optional[float] = None,
    direction: str = 'positive'
) -> np.ndarray:
    """
    Compute Poincaré map (successive intersections with a surface).

    Args:
        values: Input time series
        section_value: Value defining the Poincaré section (default: mean)
        direction: 'positive' or 'negative' crossing direction

    Returns:
        Array of values at section crossings
    """
    values = np.asarray(values, dtype=np.float64)

    if section_value is None:
        section_value = np.mean(values)

    crossings = []

    for i in range(1, len(values)):
        if direction == 'positive':
            # Crossing from below
            if values[i - 1] < section_value <= values[i]:
                crossings.append(values[i])
        elif direction == 'negative':
            # Crossing from above
            if values[i - 1] > section_value >= values[i]:
                crossings.append(values[i])
        else:
            raise ValueError(f"Unknown direction: {direction}")

    return np.array(crossings)
