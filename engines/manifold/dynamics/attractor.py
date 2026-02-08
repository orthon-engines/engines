"""
Attractor Reconstruction Engine.

Discovers hidden dynamical structure using Takens embedding.

Computes:
- Optimal embedding dimension (false nearest neighbors)
- Optimal delay (mutual information)
- Correlation dimension (attractor complexity)
- Attractor properties

Philosophy: Compute once, melt the mac, query forever.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy.spatial.distance import pdist
from scipy.stats import linregress

try:
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from engines.primitives.embedding import (
    time_delay_embedding,
    optimal_delay,
    optimal_dimension,
)


def compute(
    y: np.ndarray,
    min_samples: int = 100,
    max_dim: int = 10,
    max_delay: int = 100,
) -> Dict[str, Any]:
    """
    Reconstruct attractor and compute properties.
    
    Args:
        y: Signal values
        min_samples: Minimum samples required
        max_dim: Maximum embedding dimension to test
        max_delay: Maximum delay to test
        
    Returns:
        dict with embedding_dim, embedding_tau, correlation_dim, etc.
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)
    
    if n < min_samples:
        return _empty_result()
    
    try:
        # Find optimal embedding parameters
        tau = optimal_delay(y, max_lag=min(max_delay, n // 10))
        dim = optimal_dimension(y, tau, max_dim=max_dim)
        
        # Embed signal
        embedded = time_delay_embedding(y, dimension=dim, delay=tau)
        
        if len(embedded) < 50:
            return _empty_result()
        
        # Compute correlation dimension
        corr_dim, corr_dim_r2 = _correlation_dimension(embedded)
        
        return {
            'embedding_dim': dim,
            'embedding_tau': tau,
            'correlation_dim': corr_dim,
            'correlation_dim_r2': corr_dim_r2,
            'n_embedded': len(embedded),
        }
        
    except Exception:
        return _empty_result()


def _empty_result() -> Dict[str, Any]:
    """Return empty result."""
    return {
        'embedding_dim': None,
        'embedding_tau': None,
        'correlation_dim': None,
        'correlation_dim_r2': None,
        'n_embedded': 0,
    }


def _correlation_dimension(
    embedded: np.ndarray,
    n_scales: int = 20,
) -> Tuple[float, float]:
    """
    Compute correlation dimension using Grassberger-Procaccia algorithm.
    
    Returns:
        (correlation_dimension, r_squared)
    """
    n = len(embedded)
    
    if n < 50:
        return np.nan, np.nan
    
    # Compute pairwise distances
    # For large n, subsample
    if n > 1000:
        idx = np.random.choice(n, 1000, replace=False)
        embedded = embedded[idx]
        n = 1000
    
    distances = pdist(embedded)
    distances = distances[distances > 0]
    
    if len(distances) < 100:
        return np.nan, np.nan
    
    # Define scales
    min_r = np.percentile(distances, 1)
    max_r = np.percentile(distances, 50)
    
    if min_r <= 0 or max_r <= min_r:
        return np.nan, np.nan
    
    scales = np.logspace(np.log10(min_r), np.log10(max_r), n_scales)
    
    # Compute correlation sum C(r) for each scale
    n_pairs = len(distances)
    log_r = []
    log_c = []
    
    for r in scales:
        count = np.sum(distances < r)
        if count > 0:
            c_r = count / n_pairs
            log_r.append(np.log(r))
            log_c.append(np.log(c_r))
    
    if len(log_r) < 5:
        return np.nan, np.nan
    
    # Linear fit in scaling region
    log_r = np.array(log_r)
    log_c = np.array(log_c)
    
    slope, intercept, r_value, p_value, std_err = linregress(log_r, log_c)
    
    return float(slope), float(r_value ** 2)


def compute_recurrence_matrix(
    embedded: np.ndarray,
    threshold: float = None,
    threshold_pct: float = 10.0,
) -> np.ndarray:
    """
    Compute recurrence matrix.
    
    Args:
        embedded: Embedded trajectory
        threshold: Fixed threshold distance
        threshold_pct: Percentile of distances for threshold
        
    Returns:
        Boolean recurrence matrix
    """
    n = len(embedded)
    
    # Compute distance matrix
    from scipy.spatial.distance import cdist
    D = cdist(embedded, embedded)
    
    # Determine threshold
    if threshold is None:
        # Use percentile of distances
        threshold = np.percentile(D[D > 0], threshold_pct)
    
    # Recurrence matrix
    R = D < threshold
    
    return R


def compute_rqa_from_matrix(R: np.ndarray) -> Dict[str, float]:
    """
    Compute RQA metrics from recurrence matrix.
    
    Returns:
        dict with recurrence_rate, determinism, laminarity, etc.
    """
    n = R.shape[0]
    
    # Recurrence rate
    rr = np.sum(R) / (n * n)
    
    # Determinism (fraction of recurrence points in diagonal lines)
    det, diag_lengths = _diagonal_lines(R, min_length=2)
    
    # Laminarity (fraction of recurrence points in vertical lines)
    lam, vert_lengths = _vertical_lines(R, min_length=2)
    
    # Entropy of diagonal line lengths
    if len(diag_lengths) > 0:
        p = diag_lengths / np.sum(diag_lengths)
        entropy = -np.sum(p * np.log(p + 1e-10))
    else:
        entropy = 0.0
    
    # Trapping time (average vertical line length)
    if len(vert_lengths) > 0:
        trapping_time = np.mean(vert_lengths)
    else:
        trapping_time = 0.0
    
    return {
        'recurrence_rate': float(rr),
        'determinism': float(det),
        'laminarity': float(lam),
        'rqa_entropy': float(entropy),
        'trapping_time': float(trapping_time),
    }


def _diagonal_lines(R: np.ndarray, min_length: int = 2) -> Tuple[float, np.ndarray]:
    """Extract diagonal line lengths and compute determinism."""
    n = R.shape[0]
    lengths = []
    
    # Check all diagonals
    for k in range(-n + 1, n):
        diag = np.diag(R, k)
        
        # Find consecutive True values
        length = 0
        for val in diag:
            if val:
                length += 1
            else:
                if length >= min_length:
                    lengths.append(length)
                length = 0
        if length >= min_length:
            lengths.append(length)
    
    lengths = np.array(lengths)
    
    if len(lengths) == 0:
        return 0.0, lengths
    
    # Determinism = sum of points in lines / total recurrence points
    total_in_lines = np.sum(lengths)
    total_recurrence = np.sum(R)
    
    det = total_in_lines / total_recurrence if total_recurrence > 0 else 0.0
    
    return det, lengths


def _vertical_lines(R: np.ndarray, min_length: int = 2) -> Tuple[float, np.ndarray]:
    """Extract vertical line lengths and compute laminarity."""
    n = R.shape[0]
    lengths = []
    
    # Check each column
    for j in range(n):
        col = R[:, j]
        
        length = 0
        for val in col:
            if val:
                length += 1
            else:
                if length >= min_length:
                    lengths.append(length)
                length = 0
        if length >= min_length:
            lengths.append(length)
    
    lengths = np.array(lengths)
    
    if len(lengths) == 0:
        return 0.0, lengths
    
    # Laminarity = sum of points in vertical lines / total recurrence points
    total_in_lines = np.sum(lengths)
    total_recurrence = np.sum(R)
    
    lam = total_in_lines / total_recurrence if total_recurrence > 0 else 0.0
    
    return lam, lengths
