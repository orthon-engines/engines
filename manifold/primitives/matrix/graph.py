"""
Matrix Graph Primitives (63-65)

Distance matrix, adjacency matrix, Laplacian matrix.
"""

import numpy as np
from typing import Callable, Optional, Literal

try:
    from rudder_primitives_rs.matrix import (
        laplacian_matrix as _laplacian_matrix_rs,
        laplacian_eigenvalues as _laplacian_eigenvalues_rs,
    )
    _USE_RUST_GRAPH = True
except ImportError:
    _USE_RUST_GRAPH = False


def distance_matrix(
    signals: np.ndarray,
    metric: str = 'euclidean',
    rowvar: bool = False
) -> np.ndarray:
    """
    Compute pairwise distance matrix between signals.

    Parameters
    ----------
    signals : np.ndarray
        Data matrix (n_samples x n_signals) or (n_signals x n_samples)
    metric : str
        Distance metric: 'euclidean', 'manhattan', 'cosine', 'correlation'
    rowvar : bool
        If True, each row is a variable. If False, each column is a variable.

    Returns
    -------
    np.ndarray
        Distance matrix (n_signals x n_signals)

    Notes
    -----
    Output is symmetric with zeros on diagonal (for proper metrics).
    Cosine distance = 1 - cosine_similarity.
    Correlation distance = 1 - |correlation|.
    """
    signals = np.asarray(signals)

    if signals.ndim == 1:
        return np.array([[0.0]])

    if rowvar:
        signals = signals.T  # Now: (n_samples x n_signals)

    n_samples, n_signals = signals.shape
    dist = np.zeros((n_signals, n_signals))

    for i in range(n_signals):
        for j in range(i + 1, n_signals):
            x, y = signals[:, i], signals[:, j]

            # Handle NaN
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 2:
                dist[i, j] = dist[j, i] = np.nan
                continue

            x_clean, y_clean = x[mask], y[mask]

            if metric == 'euclidean':
                d = np.sqrt(np.sum((x_clean - y_clean) ** 2))
            elif metric == 'manhattan':
                d = np.sum(np.abs(x_clean - y_clean))
            elif metric == 'cosine':
                dot = np.dot(x_clean, y_clean)
                norm_x = np.linalg.norm(x_clean)
                norm_y = np.linalg.norm(y_clean)
                if norm_x > 0 and norm_y > 0:
                    d = 1 - dot / (norm_x * norm_y)
                else:
                    d = np.nan
            elif metric == 'correlation':
                if np.std(x_clean) > 0 and np.std(y_clean) > 0:
                    d = 1 - np.abs(np.corrcoef(x_clean, y_clean)[0, 1])
                else:
                    d = np.nan
            else:
                raise ValueError(f"Unknown metric: {metric}")

            dist[i, j] = d
            dist[j, i] = d

    return dist


def adjacency_matrix(
    similarity: np.ndarray,
    threshold: float = None,
    k_nearest: int = None,
    binary: bool = False
) -> np.ndarray:
    """
    Create adjacency matrix from similarity/distance matrix.

    Parameters
    ----------
    similarity : np.ndarray
        Similarity or distance matrix (n x n)
    threshold : float, optional
        Keep edges where similarity > threshold (or distance < threshold)
    k_nearest : int, optional
        Keep only k nearest neighbors per node
    binary : bool
        If True, return binary adjacency (0/1). If False, keep weights.

    Returns
    -------
    np.ndarray
        Adjacency matrix (n x n)

    Notes
    -----
    At least one of threshold or k_nearest must be specified.
    If both specified, edges must satisfy both conditions.
    """
    similarity = np.asarray(similarity)
    n = similarity.shape[0]

    if threshold is None and k_nearest is None:
        # Default: keep all edges
        adj = similarity.copy()
        np.fill_diagonal(adj, 0)
        if binary:
            adj = (adj != 0).astype(float)
        return adj

    adj = np.zeros_like(similarity)

    for i in range(n):
        row = similarity[i, :].copy()
        row[i] = np.nan  # Exclude self

        valid = ~np.isnan(row)

        if not np.any(valid):
            continue

        # Determine if this is similarity (higher = closer) or distance (lower = closer)
        # Heuristic: if diagonal is max or all positive, likely similarity
        diag_val = similarity[i, i] if not np.isnan(similarity[i, i]) else 0

        if k_nearest is not None:
            # Assume distance matrix (lower = closer) unless diagonal suggests otherwise
            if diag_val > np.nanmax(row[valid]):
                # Similarity matrix - keep k highest
                indices = np.where(valid)[0]
                sorted_idx = indices[np.argsort(row[indices])[::-1]][:k_nearest]
            else:
                # Distance matrix - keep k lowest
                indices = np.where(valid)[0]
                sorted_idx = indices[np.argsort(row[indices])][:k_nearest]

            for j in sorted_idx:
                if threshold is None or (diag_val > np.nanmax(row[valid]) and row[j] > threshold) or \
                   (diag_val <= np.nanmax(row[valid]) and row[j] < threshold):
                    adj[i, j] = row[j] if not binary else 1.0
        elif threshold is not None:
            for j in range(n):
                if i != j and not np.isnan(row[j]):
                    # Assume lower values = closer for distance
                    if row[j] < threshold:
                        adj[i, j] = row[j] if not binary else 1.0

    return adj


def laplacian_matrix(
    adjacency: np.ndarray,
    normalized: bool = False
) -> np.ndarray:
    """
    Compute graph Laplacian from adjacency matrix.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix (n x n)
    normalized : bool
        If True, compute normalized Laplacian

    Returns
    -------
    np.ndarray
        Laplacian matrix (n x n)

    Notes
    -----
    Unnormalized: L = D - A
    where D is diagonal degree matrix.

    Normalized: L_norm = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}

    Properties:
    - Symmetric positive semi-definite
    - Smallest eigenvalue is 0
    - Number of zero eigenvalues = number of connected components
    - Second smallest eigenvalue (Fiedler value) measures connectivity
    """
    adjacency = np.asarray(adjacency, dtype=np.float64)

    if _USE_RUST_GRAPH and adjacency.ndim == 2:
        return np.asarray(_laplacian_matrix_rs(adjacency, normalized))

    n = adjacency.shape[0]

    # Degree matrix
    degrees = np.nansum(np.abs(adjacency), axis=1)

    if normalized:
        # D^{-1/2}
        with np.errstate(divide='ignore'):
            d_inv_sqrt = 1.0 / np.sqrt(degrees)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0

        # L_norm = I - D^{-1/2} A D^{-1/2}
        D_inv_sqrt = np.diag(d_inv_sqrt)
        laplacian = np.eye(n) - D_inv_sqrt @ adjacency @ D_inv_sqrt
    else:
        # L = D - A
        laplacian = np.diag(degrees) - adjacency

    return laplacian


def laplacian_eigenvalues(
    laplacian: np.ndarray,
    k: int = None
) -> np.ndarray:
    """
    Compute eigenvalues of graph Laplacian.

    Parameters
    ----------
    laplacian : np.ndarray
        Laplacian matrix (n x n)
    k : int, optional
        Number of smallest eigenvalues to return

    Returns
    -------
    np.ndarray
        Eigenvalues (sorted ascending)

    Notes
    -----
    Eigenvalue interpretation:
    - λ_1 = 0 always
    - λ_2 (algebraic connectivity / Fiedler value): larger = more connected
    - Number of λ = 0: number of connected components
    """
    laplacian = np.asarray(laplacian, dtype=np.float64)

    if _USE_RUST_GRAPH and laplacian.ndim == 2:
        return np.asarray(_laplacian_eigenvalues_rs(laplacian, k))

    try:
        eigenvalues = np.linalg.eigvalsh(laplacian)
    except np.linalg.LinAlgError:
        return np.full(laplacian.shape[0], np.nan)

    eigenvalues = np.sort(eigenvalues.real)

    if k is not None:
        eigenvalues = eigenvalues[:k]

    return eigenvalues


def temporal_distance_matrix(
    signals: np.ndarray,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Compute pairwise distance matrix between all time points.

    Parameters
    ----------
    signals : np.ndarray
        2D array of shape (n_samples, n_signals)
        Each row is a state vector at one time point
    metric : str
        Distance metric: 'euclidean', 'manhattan', 'chebyshev', 'cosine'

    Returns
    -------
    np.ndarray
        Distance matrix of shape (n_samples, n_samples)
        dist[i,j] = distance between state at time i and state at time j

    Notes
    -----
    Treats each row as a point in n_signals-dimensional space.
    Computes distance between all pairs of points.

    Physical interpretation:
    The distance matrix shows how the system's state changes over time.
    - Small dist[i,j]: system was in similar state at times i and j
    - Patterns in distance matrix reveal recurrence, periodicity
    """
    from scipy.spatial.distance import pdist, squareform

    signals = np.asarray(signals)

    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)

    if metric == 'cosine':
        distances = pdist(signals, metric='cosine')
    elif metric == 'chebyshev':
        distances = pdist(signals, metric='chebyshev')
    else:
        distances = pdist(signals, metric=metric)

    return squareform(distances)


def recurrence_matrix(
    signals: np.ndarray,
    threshold: float = None,
    threshold_percentile: float = 10.0,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Compute recurrence matrix (binary version of temporal distance matrix).

    Parameters
    ----------
    signals : np.ndarray
        2D array of shape (n_samples, n_signals)
        Can be raw signals or embedded phase space
    threshold : float, optional
        Distance threshold for recurrence. If None, use percentile.
    threshold_percentile : float
        If threshold is None, use this percentile of distances
    metric : str
        Distance metric

    Returns
    -------
    np.ndarray
        Binary recurrence matrix of shape (n_samples, n_samples)
        R[i,j] = 1 if states at time i and j are "close"
        R[i,j] = 0 otherwise

    Notes
    -----
    R[i,j] = 1 if ||x(i) - x(j)|| < epsilon, else 0

    Properties:
    - Symmetric: R[i,j] = R[j,i]
    - Diagonal is all 1s: R[i,i] = 1 (state is close to itself)
    - Binary: only 0 and 1

    Physical interpretation:
    A recurrence plot (visualization of R) reveals:
    - Diagonal lines: deterministic dynamics
    - Vertical/horizontal lines: laminar states
    - Isolated points: chaos, unpredictability
    - Checkerboard: periodicity
    """
    # Compute temporal distance matrix
    dist = temporal_distance_matrix(signals, metric=metric)

    # Determine threshold
    if threshold is None:
        # Use percentile of non-zero distances
        flat_dist = dist[np.triu_indices_from(dist, k=1)]
        threshold = np.percentile(flat_dist, threshold_percentile)

    # Threshold to get binary matrix
    R = (dist <= threshold).astype(int)

    return R
