"""
PRISM Geometry and Linear Algebra Primitives

Pure mathematical functions for matrix operations and eigenstructure analysis.
Core functions for effective dimension, participation ratio, and eigendecomposition.
"""

import numpy as np
from scipy.linalg import eigh, eigvals, svd
from typing import Tuple, Optional


def covariance_matrix(
    data: np.ndarray,
    bias: bool = False
) -> np.ndarray:
    """
    Compute covariance matrix of multivariate data.

    Args:
        data: Input data matrix (samples × variables)
        bias: If False, use N-1 normalization (sample covariance)
              If True, use N normalization (population covariance)

    Returns:
        Covariance matrix (variables × variables)
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim == 1:
        return np.array([[np.var(data, ddof=0 if bias else 1)]])

    return np.cov(data, rowvar=False, bias=bias)


def correlation_matrix(data: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix of multivariate data.

    Args:
        data: Input data matrix (samples × variables)

    Returns:
        Correlation matrix (variables × variables)
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim == 1:
        return np.array([[1.0]])

    return np.corrcoef(data, rowvar=False)


def eigendecomposition(
    matrix: np.ndarray,
    symmetric: bool = True,
    sort_descending: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues and eigenvectors of a matrix.

    Args:
        matrix: Input square matrix
        symmetric: If True, assume matrix is symmetric (faster computation)
        sort_descending: If True, sort eigenvalues in descending order

    Returns:
        Tuple of (eigenvalues, eigenvectors)
        Eigenvectors are columns of the returned matrix
    """
    matrix = np.asarray(matrix, dtype=np.float64)

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")

    if symmetric:
        eigenvals, eigenvecs = eigh(matrix)
    else:
        eigenvals, eigenvecs = np.linalg.eig(matrix)
        eigenvals = np.real(eigenvals)
        eigenvecs = np.real(eigenvecs)

    if sort_descending:
        idx = np.argsort(np.abs(eigenvals))[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

    return eigenvals, eigenvecs


def effective_dimension(
    eigenvalues: np.ndarray,
    method: str = 'participation_ratio'
) -> float:
    """
    Compute effective dimension from eigenvalues.

    The effective dimension quantifies how many dimensions actively
    contribute to the system's behavior.

    Args:
        eigenvalues: Array of eigenvalues (should be non-negative)
        method: Method to compute effective dimension:
                'participation_ratio' - Standard participation ratio
                'normalized_entropy' - Information-theoretic measure
                'inverse_participation' - Physics-based measure

    Returns:
        Effective dimension (1 to len(eigenvalues))
    """
    eigenvals = np.abs(np.asarray(eigenvalues, dtype=np.float64))
    eigenvals = eigenvals[eigenvals > 1e-12]

    if len(eigenvals) == 0:
        return 0.0

    if method == 'participation_ratio':
        sum_evals = np.sum(eigenvals)
        sum_squared = np.sum(eigenvals**2)

        if sum_squared == 0:
            return 0.0

        return float(sum_evals**2 / sum_squared)

    elif method == 'normalized_entropy':
        probs = eigenvals / np.sum(eigenvals)
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        return float(2**entropy)

    elif method == 'inverse_participation':
        probs = eigenvals / np.sum(eigenvals)
        ipr = 1.0 / np.sum(probs**2)
        return float(ipr)

    else:
        raise ValueError(f"Unknown method: {method}")


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """
    Compute participation ratio from eigenvalues.

    PR = (Σλᵢ)² / (Σλᵢ²)

    Args:
        eigenvalues: Array of eigenvalues

    Returns:
        Participation ratio
    """
    return effective_dimension(eigenvalues, method='participation_ratio')


def condition_number(matrix: np.ndarray) -> float:
    """
    Compute condition number of a matrix.

    The condition number indicates how sensitive the matrix is to
    numerical errors. Large condition numbers indicate ill-conditioning.

    Args:
        matrix: Input matrix

    Returns:
        Condition number (ratio of largest to smallest singular value)
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    singular_vals = svd(matrix, compute_uv=False)
    nonzero_sv = singular_vals[singular_vals > 1e-12]

    if len(nonzero_sv) == 0:
        return float('inf')

    return float(np.max(nonzero_sv) / np.min(nonzero_sv))


def matrix_rank(
    matrix: np.ndarray,
    tolerance: Optional[float] = None
) -> int:
    """
    Compute numerical rank of a matrix.

    Args:
        matrix: Input matrix
        tolerance: Tolerance for determining rank (default: automatic)

    Returns:
        Numerical rank
    """
    matrix = np.asarray(matrix, dtype=np.float64)

    if tolerance is None:
        tolerance = max(matrix.shape) * np.finfo(matrix.dtype).eps * np.max(np.abs(matrix))

    singular_vals = svd(matrix, compute_uv=False)
    rank = int(np.sum(singular_vals > tolerance))

    return rank


def alignment_metric(
    eigenvalues: np.ndarray,
    method: str = 'cosine'
) -> float:
    """
    Compute alignment of eigenvalue distribution.

    Measures how aligned the eigenvalue distribution is with
    a uniform distribution (high alignment = more uniform).

    Args:
        eigenvalues: Array of eigenvalues
        method: 'cosine' or 'kl_divergence'

    Returns:
        Alignment measure
    """
    eigenvals = np.abs(np.asarray(eigenvalues, dtype=np.float64))
    eigenvals = eigenvals[eigenvals > 1e-12]

    if len(eigenvals) <= 1:
        return 1.0

    probs = eigenvals / np.sum(eigenvals)
    uniform = np.ones_like(probs) / len(probs)

    if method == 'cosine':
        dot_product = np.dot(probs, uniform)
        norm_probs = np.linalg.norm(probs)
        norm_uniform = np.linalg.norm(uniform)

        if norm_probs == 0 or norm_uniform == 0:
            return 0.0

        return float(dot_product / (norm_probs * norm_uniform))

    elif method == 'kl_divergence':
        kl_div = np.sum(probs * np.log(probs / (uniform + 1e-12) + 1e-12))
        max_kl = np.log(len(probs))
        alignment = 1.0 - (kl_div / max_kl)
        return float(max(0.0, alignment))

    else:
        raise ValueError(f"Unknown method: {method}")


def eigenvalue_spread(eigenvalues: np.ndarray) -> float:
    """
    Compute spread of eigenvalues (coefficient of variation).

    Args:
        eigenvalues: Array of eigenvalues

    Returns:
        Eigenvalue spread
    """
    eigenvals = np.abs(np.asarray(eigenvalues, dtype=np.float64))
    eigenvals = eigenvals[eigenvals > 1e-12]

    if len(eigenvals) <= 1:
        return 0.0

    mean_eval = np.mean(eigenvals)
    std_eval = np.std(eigenvals)

    if mean_eval == 0:
        return 0.0

    return float(std_eval / mean_eval)


def matrix_entropy(
    matrix: np.ndarray,
    normalize: bool = True
) -> float:
    """
    Compute entropy of a matrix using its eigenvalues.

    Args:
        matrix: Input square matrix
        normalize: If True, normalize entropy to [0,1]

    Returns:
        Matrix entropy
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    eigenvals = np.abs(eigvals(matrix))
    eigenvals = eigenvals[eigenvals > 1e-12]

    if len(eigenvals) == 0:
        return 0.0

    probs = eigenvals / np.sum(eigenvals)
    entropy = -np.sum(probs * np.log2(probs + 1e-12))

    if normalize:
        max_entropy = np.log2(len(probs))
        if max_entropy > 0:
            entropy = entropy / max_entropy

    return float(entropy)


def geometric_mean_eigenvalue(eigenvalues: np.ndarray) -> float:
    """
    Compute geometric mean of eigenvalues.

    Args:
        eigenvalues: Array of eigenvalues

    Returns:
        Geometric mean of eigenvalues
    """
    eigenvals = np.abs(np.asarray(eigenvalues, dtype=np.float64))
    eigenvals = eigenvals[eigenvals > 1e-12]

    if len(eigenvals) == 0:
        return 0.0

    log_mean = np.mean(np.log(eigenvals + 1e-12))
    return float(np.exp(log_mean))


def svd_decomposition(
    matrix: np.ndarray,
    full_matrices: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute singular value decomposition.

    Args:
        matrix: Input matrix
        full_matrices: If True, return full U and Vh matrices

    Returns:
        Tuple of (U, singular_values, Vh)
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    U, s, Vh = svd(matrix, full_matrices=full_matrices)
    return U, s, Vh


def explained_variance_ratio(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Compute explained variance ratio for each eigenvalue.

    Args:
        eigenvalues: Array of eigenvalues

    Returns:
        Array of variance ratios (sum to 1)
    """
    eigenvals = np.abs(np.asarray(eigenvalues, dtype=np.float64))
    total = np.sum(eigenvals)

    if total == 0:
        return np.zeros_like(eigenvals)

    return eigenvals / total


def cumulative_variance_ratio(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Compute cumulative explained variance ratio.

    Args:
        eigenvalues: Array of eigenvalues (should be sorted descending)

    Returns:
        Cumulative variance ratios
    """
    ratios = explained_variance_ratio(eigenvalues)
    return np.cumsum(ratios)
