"""
Pairwise Distance Primitives (49-51)

DTW, Euclidean, Cosine similarity.
"""

import numpy as np
from typing import Optional

try:
    from rudder_primitives_rs.pairwise import (
        dynamic_time_warping as _dtw_rs,
    )
    _USE_RUST_DISTANCE = True
except ImportError:
    _USE_RUST_DISTANCE = False


def dynamic_time_warping(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    window: int = None,
    return_path: bool = False
) -> float:
    """
    Compute Dynamic Time Warping distance.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Input signals (can be different lengths)
    window : int, optional
        Sakoe-Chiba band width
    return_path : bool
        If True, also return warping path

    Returns
    -------
    float
        DTW distance

    Notes
    -----
    Measures similarity allowing for time warping.
    Good for comparing signals with different speeds/phases.
    """
    sig_a = np.asarray(sig_a).flatten()
    sig_b = np.asarray(sig_b).flatten()

    n, m = len(sig_a), len(sig_b)

    if _USE_RUST_DISTANCE:
        return _dtw_rs(sig_a, sig_b, window)

    if window is None:
        window = max(n, m)

    # Cost matrix
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m + 1, i + window + 1)):
            cost = (sig_a[i-1] - sig_b[j-1]) ** 2
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])

    return float(np.sqrt(dtw[n, m]))


def euclidean_distance(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    normalized: bool = False
) -> float:
    """
    Compute Euclidean distance.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Input signals
    normalized : bool
        If True, normalize by sqrt(n)

    Returns
    -------
    float
        Euclidean distance

    Notes
    -----
    d = sqrt(sum((a_i - b_i)^2))
    """
    sig_a = np.asarray(sig_a).flatten()
    sig_b = np.asarray(sig_b).flatten()

    n = min(len(sig_a), len(sig_b))
    sig_a, sig_b = sig_a[:n], sig_b[:n]

    dist = np.sqrt(np.sum((sig_a - sig_b) ** 2))

    if normalized:
        dist = dist / np.sqrt(n)

    return float(dist)


def cosine_similarity(
    sig_a: np.ndarray,
    sig_b: np.ndarray
) -> float:
    """
    Compute cosine similarity.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Input signals

    Returns
    -------
    float
        Cosine similarity in [-1, 1]

    Notes
    -----
    cos(θ) = (a · b) / (||a|| * ||b||)
    Measures angular similarity, invariant to magnitude.
    """
    sig_a = np.asarray(sig_a).flatten()
    sig_b = np.asarray(sig_b).flatten()

    n = min(len(sig_a), len(sig_b))
    sig_a, sig_b = sig_a[:n], sig_b[:n]

    norm_a = np.linalg.norm(sig_a)
    norm_b = np.linalg.norm(sig_b)

    if norm_a == 0 or norm_b == 0:
        return np.nan

    return float(np.dot(sig_a, sig_b) / (norm_a * norm_b))


def manhattan_distance(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    normalized: bool = False
) -> float:
    """
    Compute Manhattan (L1) distance.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Input signals
    normalized : bool
        If True, normalize by n

    Returns
    -------
    float
        Manhattan distance

    Notes
    -----
    d = sum(|a_i - b_i|)
    More robust to outliers than Euclidean.
    """
    sig_a = np.asarray(sig_a).flatten()
    sig_b = np.asarray(sig_b).flatten()

    n = min(len(sig_a), len(sig_b))
    sig_a, sig_b = sig_a[:n], sig_b[:n]

    dist = np.sum(np.abs(sig_a - sig_b))

    if normalized:
        dist = dist / n

    return float(dist)
