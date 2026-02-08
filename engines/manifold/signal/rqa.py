"""
Recurrence Quantification Analysis (RQA) Engines.

Computes recurrence-based complexity measures:
- recurrence_rate: fraction of recurrent points
- determinism: fraction of recurrent points forming diagonal lines
- correlation_dimension: fractal dimension estimate
- embedding_dim: optimal embedding dimension
"""

import numpy as np
from typing import Dict, Any
from scipy.spatial.distance import pdist, squareform


def compute_recurrence_matrix(x: np.ndarray, threshold: float = None, embed_dim: int = 3, delay: int = 1) -> np.ndarray:
    """
    Compute recurrence matrix from time series.

    Args:
        x: 1D time series
        threshold: Distance threshold (default: 10% of std)
        embed_dim: Embedding dimension
        delay: Time delay for embedding

    Returns:
        Binary recurrence matrix
    """
    x = np.asarray(x).flatten()
    x = x[~np.isnan(x)]
    n = len(x)

    if n < embed_dim * delay + 1:
        return np.array([[]])

    # Time-delay embedding
    m = n - (embed_dim - 1) * delay
    embedded = np.zeros((m, embed_dim))
    for i in range(embed_dim):
        embedded[:, i] = x[i * delay:i * delay + m]

    # Distance matrix
    distances = squareform(pdist(embedded))

    # Threshold
    if threshold is None:
        threshold = 0.1 * np.std(x)

    # Recurrence matrix
    R = (distances <= threshold).astype(int)
    return R


def compute_recurrence_rate(x: np.ndarray, threshold: float = None, embed_dim: int = 3, delay: int = 1) -> Dict[str, float]:
    """
    Compute recurrence rate (RR).

    RR = fraction of recurrent points in the recurrence matrix.
    """
    x = np.asarray(x).flatten()
    x = x[~np.isnan(x)]

    if len(x) < 20:
        return {'recurrence_rate': np.nan}

    R = compute_recurrence_matrix(x, threshold, embed_dim, delay)
    if R.size == 0:
        return {'recurrence_rate': np.nan}

    n = R.shape[0]
    if n == 0:
        return {'recurrence_rate': np.nan}

    # Exclude diagonal
    np.fill_diagonal(R, 0)
    rr = R.sum() / (n * (n - 1))

    return {'recurrence_rate': float(rr)}


def compute_determinism(x: np.ndarray, threshold: float = None, embed_dim: int = 3, delay: int = 1, min_line: int = 2) -> Dict[str, float]:
    """
    Compute determinism (DET).

    DET = fraction of recurrent points forming diagonal lines of length >= min_line.
    High DET indicates deterministic dynamics.
    """
    x = np.asarray(x).flatten()
    x = x[~np.isnan(x)]

    if len(x) < 20:
        return {'determinism': np.nan}

    R = compute_recurrence_matrix(x, threshold, embed_dim, delay)
    if R.size == 0:
        return {'determinism': np.nan}

    n = R.shape[0]
    if n < min_line:
        return {'determinism': np.nan}

    # Count diagonal lines
    total_recurrent = 0
    diagonal_recurrent = 0

    # Check all diagonals (excluding main)
    for k in range(1, n):
        diag = np.diag(R, k)
        total_recurrent += diag.sum()

        # Count points in lines >= min_line
        line_length = 0
        for val in diag:
            if val:
                line_length += 1
            else:
                if line_length >= min_line:
                    diagonal_recurrent += line_length
                line_length = 0
        if line_length >= min_line:
            diagonal_recurrent += line_length

    # Symmetric matrix - count lower triangle too
    total_recurrent *= 2
    diagonal_recurrent *= 2

    if total_recurrent == 0:
        return {'determinism': np.nan}

    det = diagonal_recurrent / total_recurrent
    return {'determinism': float(det)}


def compute_correlation_dimension(x: np.ndarray, embed_dims: list = None, delay: int = 1) -> Dict[str, float]:
    """
    Estimate correlation dimension using Grassberger-Procaccia algorithm.

    Returns the slope of log(C(r)) vs log(r) curve.
    """
    x = np.asarray(x).flatten()
    x = x[~np.isnan(x)]

    if len(x) < 50:
        return {'correlation_dimension': np.nan}

    if embed_dims is None:
        embed_dims = [2, 3, 4, 5]

    # Use highest embedding dimension
    embed_dim = max(embed_dims)
    n = len(x)
    m = n - (embed_dim - 1) * delay

    if m < 20:
        return {'correlation_dimension': np.nan}

    # Embed
    embedded = np.zeros((m, embed_dim))
    for i in range(embed_dim):
        embedded[:, i] = x[i * delay:i * delay + m]

    # Compute distances
    distances = pdist(embedded)
    distances = distances[distances > 0]  # Remove zeros

    if len(distances) < 10:
        return {'correlation_dimension': np.nan}

    # Correlation sum for multiple radii
    r_values = np.logspace(np.log10(np.percentile(distances, 1)),
                           np.log10(np.percentile(distances, 50)), 20)

    C_values = []
    for r in r_values:
        C = np.sum(distances < r) / len(distances)
        if C > 0:
            C_values.append((np.log(r), np.log(C)))

    if len(C_values) < 5:
        return {'correlation_dimension': np.nan}

    # Linear fit to get dimension
    log_r = np.array([c[0] for c in C_values])
    log_C = np.array([c[1] for c in C_values])

    slope, _ = np.polyfit(log_r, log_C, 1)

    return {'correlation_dimension': float(slope)}


def compute_embedding_dim(x: np.ndarray, max_dim: int = 10, delay: int = 1, threshold: float = 0.1) -> Dict[str, float]:
    """
    Estimate optimal embedding dimension using False Nearest Neighbors (FNN).

    Returns the dimension where FNN ratio drops below threshold.
    """
    x = np.asarray(x).flatten()
    x = x[~np.isnan(x)]

    if len(x) < 50:
        return {'embedding_dim': np.nan}

    n = len(x)

    for dim in range(1, max_dim + 1):
        m = n - dim * delay
        if m < 20:
            return {'embedding_dim': float(dim - 1) if dim > 1 else np.nan}

        # Embed in dim and dim+1
        embedded_d = np.zeros((m, dim))
        embedded_d1 = np.zeros((m, dim + 1))

        for i in range(dim):
            embedded_d[:, i] = x[i * delay:i * delay + m]
            embedded_d1[:, i] = x[i * delay:i * delay + m]
        embedded_d1[:, dim] = x[dim * delay:dim * delay + m]

        # Find nearest neighbors in dim-dimensional space
        from scipy.spatial import cKDTree
        tree = cKDTree(embedded_d)

        fnn_count = 0
        total = 0

        for i in range(min(m, 500)):  # Sample for speed
            dist, idx = tree.query(embedded_d[i], k=2)
            if idx[1] < m and dist[1] > 1e-10:
                # Check if neighbor is false
                d_d = dist[1]
                d_d1 = np.linalg.norm(embedded_d1[i] - embedded_d1[idx[1]])

                if d_d1 / d_d > 10:  # Standard FNN criterion
                    fnn_count += 1
                total += 1

        if total > 0:
            fnn_ratio = fnn_count / total
            if fnn_ratio < threshold:
                return {'embedding_dim': float(dim)}

    return {'embedding_dim': float(max_dim)}


# Engine compute functions for registry
def compute(x: np.ndarray, metric: str = 'all') -> Dict[str, float]:
    """Compute RQA metrics."""
    if metric == 'recurrence_rate':
        return compute_recurrence_rate(x)
    elif metric == 'determinism':
        return compute_determinism(x)
    elif metric == 'correlation_dimension':
        return compute_correlation_dimension(x)
    elif metric == 'embedding_dim':
        return compute_embedding_dim(x)
    else:
        # All metrics
        result = {}
        result.update(compute_recurrence_rate(x))
        result.update(compute_determinism(x))
        result.update(compute_correlation_dimension(x))
        result.update(compute_embedding_dim(x))
        return result
