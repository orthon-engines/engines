"""
Embedding Delay Primitives (66-69)

Time delay embedding and parameter estimation.
"""

import numpy as np
from typing import Tuple, Optional

try:
    from rudder_primitives_rs.embedding import (
        optimal_delay as _optimal_delay_rs,
    )
    _USE_RUST_EMBEDDING = True
except ImportError:
    _USE_RUST_EMBEDDING = False


def time_delay_embedding(
    signal: np.ndarray,
    dimension: int = 3,
    delay: int = 1
) -> np.ndarray:
    """
    Construct time delay embedding of a signal.

    Parameters
    ----------
    signal : np.ndarray
        1D time series
    dimension : int
        Embedding dimension (number of coordinates)
    delay : int
        Time delay (lag between coordinates)

    Returns
    -------
    np.ndarray
        Embedded trajectory (n_points x dimension)

    Notes
    -----
    Takens' embedding theorem: for appropriate d and τ,
    the embedding preserves topological properties of the attractor.

    Point i: [x(i), x(i+τ), x(i+2τ), ..., x(i+(d-1)τ)]

    Number of embedded points: n - (d-1) * τ
    """
    signal = np.asarray(signal).flatten()
    n = len(signal)

    n_points = n - (dimension - 1) * delay

    if n_points <= 0:
        raise ValueError(
            f"Signal too short for embedding: {n} samples, "
            f"need at least {(dimension - 1) * delay + 1}"
        )

    embedded = np.zeros((n_points, dimension))

    for d in range(dimension):
        start = d * delay
        end = start + n_points
        embedded[:, d] = signal[start:end]

    return embedded


def optimal_delay(
    signal: np.ndarray,
    max_lag: int = None,
    method: str = 'mutual_info'
) -> int:
    """
    Estimate optimal time delay for embedding.

    Parameters
    ----------
    signal : np.ndarray
        1D time series
    max_lag : int, optional
        Maximum lag to consider (default: n // 4)
    method : str
        'mutual_info': First minimum of mutual information
        'autocorr': First zero crossing of autocorrelation
        'autocorr_e': First 1/e decay of autocorrelation

    Returns
    -------
    int
        Optimal delay (time lag)

    Notes
    -----
    Mutual information method (Fraser & Swinney, 1986):
    - First minimum indicates coordinates are maximally independent
    - Generally preferred over autocorrelation

    Autocorrelation methods:
    - Zero crossing: lag where autocorrelation first crosses zero
    - 1/e decay: lag where autocorrelation falls below 1/e ≈ 0.368
    """
    signal = np.asarray(signal).flatten()
    n = len(signal)

    if max_lag is None:
        max_lag = n // 4

    max_lag = min(max_lag, n // 2)

    if _USE_RUST_EMBEDDING:
        return _optimal_delay_rs(signal, max_lag, method)

    if method == 'autocorr':
        # Autocorrelation zero crossing
        signal_centered = signal - np.mean(signal)
        var = np.var(signal_centered)

        if var == 0:
            return 1

        for lag in range(1, max_lag):
            acf = np.mean(signal_centered[:-lag] * signal_centered[lag:]) / var
            if acf <= 0:
                return lag

        return max_lag

    elif method == 'autocorr_e':
        # Autocorrelation 1/e decay
        signal_centered = signal - np.mean(signal)
        var = np.var(signal_centered)

        if var == 0:
            return 1

        threshold = 1 / np.e

        for lag in range(1, max_lag):
            acf = np.mean(signal_centered[:-lag] * signal_centered[lag:]) / var
            if acf <= threshold:
                return lag

        return max_lag

    elif method == 'mutual_info':
        # Mutual information first minimum
        mi_values = []

        for lag in range(1, max_lag):
            mi = _lagged_mutual_info(signal, lag, bins=16)
            mi_values.append(mi)

            # Check for minimum
            if len(mi_values) >= 3:
                if mi_values[-2] < mi_values[-1] and mi_values[-2] < mi_values[-3]:
                    return lag - 1

        # No clear minimum - use first significant drop
        if mi_values:
            initial_mi = mi_values[0]
            for lag, mi in enumerate(mi_values, 1):
                if mi < 0.5 * initial_mi:
                    return lag

        return max(1, max_lag // 4)

    else:
        raise ValueError(f"Unknown method: {method}")


def optimal_dimension(
    signal: np.ndarray,
    delay: int = None,
    max_dim: int = 10,
    method: str = 'fnn',
    threshold: float = 0.01
) -> int:
    """
    Estimate optimal embedding dimension.

    Parameters
    ----------
    signal : np.ndarray
        1D time series
    delay : int, optional
        Time delay (if None, computed automatically)
    max_dim : int
        Maximum dimension to test
    method : str
        'fnn': False Nearest Neighbors
        'cao': Cao's method
    threshold : float
        Threshold for convergence

    Returns
    -------
    int
        Optimal embedding dimension

    Notes
    -----
    False Nearest Neighbors (Kennel et al., 1992):
    - Points that are neighbors due to projection (not true neighbors)
    - FNN ratio drops when true dimension is reached

    Cao's method:
    - Variant of FNN, more robust to noise
    - E1(d) ≈ 1 for deterministic signals when d is sufficient
    """
    signal = np.asarray(signal).flatten()

    if delay is None:
        delay = optimal_delay(signal)

    if method == 'fnn':
        return _fnn_dimension(signal, delay, max_dim, threshold)
    elif method == 'cao':
        return _cao_dimension(signal, delay, max_dim, threshold)
    else:
        raise ValueError(f"Unknown method: {method}")


def multivariate_embedding(
    signals: np.ndarray,
    dimension: int = 3,
    delay: int = 1
) -> np.ndarray:
    """
    Construct multivariate time delay embedding.

    Parameters
    ----------
    signals : np.ndarray
        Matrix of signals (n_samples x n_signals)
    dimension : int
        Embedding dimension per signal
    delay : int
        Time delay

    Returns
    -------
    np.ndarray
        Embedded trajectory (n_points x (n_signals * dimension))

    Notes
    -----
    For m signals with embedding dimension d:
    - Total embedded dimension: m * d
    - Point i: [x1(i), x1(i+τ), ..., x2(i), x2(i+τ), ...]
    """
    signals = np.asarray(signals)

    if signals.ndim == 1:
        return time_delay_embedding(signals, dimension, delay)

    n_samples, n_signals = signals.shape
    n_points = n_samples - (dimension - 1) * delay

    if n_points <= 0:
        raise ValueError(
            f"Signals too short for embedding: {n_samples} samples, "
            f"need at least {(dimension - 1) * delay + 1}"
        )

    total_dim = n_signals * dimension
    embedded = np.zeros((n_points, total_dim))

    for sig_idx in range(n_signals):
        for d in range(dimension):
            start = d * delay
            end = start + n_points
            col_idx = sig_idx * dimension + d
            embedded[:, col_idx] = signals[start:end, sig_idx]

    return embedded


# Helper functions

def _lagged_mutual_info(signal: np.ndarray, lag: int, bins: int = 16) -> float:
    """Compute mutual information between signal and its lagged version."""
    x = signal[:-lag]
    y = signal[lag:]

    # Joint histogram
    hist_xy, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist_xy / hist_xy.sum()

    # Marginals
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    # MI
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))

    return mi


def _fnn_dimension(
    signal: np.ndarray,
    delay: int,
    max_dim: int,
    threshold: float
) -> int:
    """False Nearest Neighbors method for optimal dimension."""
    from scipy.spatial import KDTree

    for dim in range(1, max_dim):
        emb = time_delay_embedding(signal, dim, delay)
        emb_next = time_delay_embedding(signal, dim + 1, delay)

        n_points = min(len(emb), len(emb_next))
        emb = emb[:n_points]
        emb_next = emb_next[:n_points]

        if n_points < 10:
            return dim

        tree = KDTree(emb)

        n_false = 0
        n_total = 0

        for i in range(min(1000, n_points)):
            # Find nearest neighbor (excluding self)
            dists, indices = tree.query(emb[i], k=2)

            if len(indices) < 2:
                continue

            j = indices[1]
            r_d = dists[1]

            if r_d < 1e-10:
                continue

            # Distance in higher dimension
            r_d1 = np.linalg.norm(emb_next[i] - emb_next[j])

            # Check if false neighbor
            if r_d1 / r_d > 10:  # Threshold ratio
                n_false += 1

            n_total += 1

        if n_total == 0:
            return dim

        fnn_ratio = n_false / n_total

        if fnn_ratio < threshold:
            return dim + 1

    return max_dim


def _cao_dimension(
    signal: np.ndarray,
    delay: int,
    max_dim: int,
    threshold: float
) -> int:
    """Cao's method for optimal dimension (returns dimension only)."""
    result = cao_embedding_analysis(signal, delay, max_dim, threshold)
    return result['dimension']


def cao_embedding_analysis(
    signal: np.ndarray,
    delay: int,
    max_dim: int = 10,
    threshold: float = 0.01,
) -> dict:
    """
    Full Cao's method: embedding dimension + determinism test.

    Returns:
        dict with:
            dimension: optimal embedding dimension
            E1_values: E1(d) ratios (dimension saturates where E1 -> 1)
            E2_values: E2(d) ratios (determinism: E2 != 1 for some d -> deterministic)
            is_deterministic: bool — whether E2 deviates from 1
            E1_saturation_dim: dimension where E1 first saturates
    """
    from scipy.spatial import KDTree

    e_values = []   # E(d) = mean of a(i,d) ratios
    e2_raw = []     # E*(d) = mean of |x(i+(d-1)τ) - x(nn(i)+(d-1)τ)|

    for dim in range(1, max_dim + 1):
        emb = time_delay_embedding(signal, dim, delay)

        if dim > 1:
            emb_prev = time_delay_embedding(signal, dim - 1, delay)
            n_points = min(len(emb), len(emb_prev))
        else:
            n_points = len(emb)
            emb_prev = emb[:, :1]

        emb = emb[:n_points]
        emb_prev = emb_prev[:n_points]

        if n_points < 10:
            break

        tree = KDTree(emb_prev)

        a_values = []
        e2_diffs = []

        n_check = min(500, n_points)
        for i in range(n_check):
            dists, indices = tree.query(emb_prev[i], k=2)

            if len(indices) < 2:
                continue

            j = indices[1]
            r_d = dists[1]

            if r_d < 1e-10:
                continue

            # E1: distance ratio between dimensions
            r_d1 = np.linalg.norm(emb[i] - emb[j])
            a_values.append(r_d1 / r_d)

            # E2: |x(i+(d-1)τ) - x(j+(d-1)τ)| — the extra coordinate distance
            extra_idx_i = i + (dim - 1) * delay
            extra_idx_j = j + (dim - 1) * delay
            if extra_idx_i < len(signal) and extra_idx_j < len(signal):
                e2_diffs.append(abs(signal[extra_idx_i] - signal[extra_idx_j]))

        if a_values:
            e_values.append(np.mean(a_values))
        if e2_diffs:
            e2_raw.append(np.mean(e2_diffs))

    # E1(d) = E(d+1) / E(d) — dimension saturation
    if len(e_values) < 2:
        return {
            'dimension': 2,
            'E1_values': [],
            'E2_values': [],
            'is_deterministic': None,
            'E1_saturation_dim': 2,
        }

    e1_values = [
        e_values[i + 1] / e_values[i] if e_values[i] > 0 else 1.0
        for i in range(len(e_values) - 1)
    ]

    # E2(d) = E*(d+1) / E*(d) — determinism test
    e2_values = []
    if len(e2_raw) >= 2:
        e2_values = [
            e2_raw[i + 1] / e2_raw[i] if e2_raw[i] > 0 else 1.0
            for i in range(len(e2_raw) - 1)
        ]

    # Find saturation dimension (E1 -> 1)
    dim_result = max_dim
    saturation_dim = max_dim
    for d, e1 in enumerate(e1_values, 1):
        if abs(e1 - 1) < threshold:
            dim_result = d + 1
            saturation_dim = d + 1
            break

    # Determinism test: if E2 deviates significantly from 1 for d >= 2,
    # the data is deterministic (stochastic data gives E2 ≈ 1 for all d).
    # Skip d=1 (first E2 value) as it's always unreliable.
    is_deterministic = None
    if len(e2_values) >= 2:
        # Check E2 values for d >= 2 (skip index 0 = d=1)
        e2_stable = e2_values[1:]
        max_deviation = max(abs(e2 - 1.0) for e2 in e2_stable) if e2_stable else 0
        is_deterministic = max_deviation > 0.1

    return {
        'dimension': dim_result,
        'E1_values': [float(v) for v in e1_values],
        'E2_values': [float(v) for v in e2_values],
        'is_deterministic': is_deterministic,
        'E1_saturation_dim': saturation_dim,
    }
