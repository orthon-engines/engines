"""
Recurrence Quantification Analysis (RQA)
========================================

Analyzes the recurrence structure of a time series by:
1. Reconstructing phase space via embedding
2. Computing recurrence matrix
3. Extracting measures from diagonal and vertical line structures

Key measures:
    - DET (Determinism): % points on diagonal lines
    - LAM (Laminarity): % points on vertical lines
    - ENT (Entropy): Complexity of diagonal line distribution
    - TT (Trapping Time): Average vertical line length
    - RR (Recurrence Rate): % of recurrent points

Supports three computation modes:
    - static: Entire signal → single value
    - windowed: Rolling windows → time series
    - point: At time t → single value

References:
    Marwan et al. (2007) "Recurrence plots for the analysis of complex systems"
"""

import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict, Any, Optional


def compute(
    series: np.ndarray,
    mode: str = 'static',
    t: Optional[int] = None,
    window_size: int = 200,
    step_size: int = 20,
    embedding_dim: int = 3,
    delay: int = 1,
    threshold: float = None,
    min_diagonal: int = 2,
    min_vertical: int = 2,
    max_vectors: int = 500,
) -> Dict[str, Any]:
    """
    Compute Recurrence Quantification Analysis metrics.

    Args:
        series: 1D numpy array of observations
        mode: 'static', 'windowed', or 'point'
        t: Time index for point mode
        window_size: Window size for windowed/point modes
        step_size: Step between windows for windowed mode
        embedding_dim: Phase space embedding dimension
        delay: Time delay for embedding
        threshold: Distance threshold for recurrence (default: 10% of range)
        min_diagonal: Minimum diagonal line length
        min_vertical: Minimum vertical line length
        max_vectors: Max vectors to use (subsampling for large series)

    Returns:
        mode='static': {'determinism': float, 'laminarity': float, ...}
        mode='windowed': {'determinism': array, 'laminarity': array, 't': array, ...}
        mode='point': {'determinism': float, 'laminarity': float, 't': int, ...}
    """
    series = np.asarray(series).flatten()

    if mode == 'static':
        return _compute_static(series, embedding_dim, delay, threshold,
                               min_diagonal, min_vertical, max_vectors)
    elif mode == 'windowed':
        return _compute_windowed(series, window_size, step_size, embedding_dim,
                                 delay, threshold, min_diagonal, min_vertical, max_vectors)
    elif mode == 'point':
        return _compute_point(series, t, window_size, embedding_dim, delay,
                              threshold, min_diagonal, min_vertical, max_vectors)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'static', 'windowed', or 'point'.")


def _compute_static(
    series: np.ndarray,
    embedding_dim: int = 3,
    delay: int = 1,
    threshold: float = None,
    min_diagonal: int = 2,
    min_vertical: int = 2,
    max_vectors: int = 500,
) -> Dict[str, Any]:
    """Compute RQA metrics on entire signal."""
    n = len(series)

    # Default threshold: 10% of max distance
    if threshold is None:
        threshold = 0.1 * (np.max(series) - np.min(series))

    # Create embedded vectors
    n_vectors = n - (embedding_dim - 1) * delay

    if n_vectors < 10:
        return _empty_result()

    embedded = np.zeros((n_vectors, embedding_dim))
    for i in range(n_vectors):
        for j in range(embedding_dim):
            embedded[i, j] = series[i + j * delay]

    # Subsample for large series
    if n_vectors > max_vectors:
        indices = np.linspace(0, n_vectors - 1, max_vectors, dtype=int)
        embedded = embedded[indices]
        n_vectors = max_vectors

    # Compute recurrence matrix
    distances = cdist(embedded, embedded, 'euclidean')
    recurrence_matrix = distances <= threshold

    # Recurrence rate
    n_recurrent = np.sum(recurrence_matrix) - n_vectors  # Exclude diagonal
    recurrence_rate = n_recurrent / (n_vectors * (n_vectors - 1))

    # Find diagonal lines
    diagonal_lengths = []
    for offset in range(1, n_vectors):
        diag = np.diag(recurrence_matrix, k=offset)
        current_length = 0
        for val in diag:
            if val:
                current_length += 1
            else:
                if current_length >= min_diagonal:
                    diagonal_lengths.append(current_length)
                current_length = 0
        if current_length >= min_diagonal:
            diagonal_lengths.append(current_length)

    # Determinism and diagonal statistics
    if diagonal_lengths:
        total_diagonal_points = sum(diagonal_lengths)
        determinism = total_diagonal_points / max(n_recurrent, 1)
        avg_diagonal = np.mean(diagonal_lengths)
        max_diagonal = max(diagonal_lengths)

        # Diagonal line entropy
        hist, _ = np.histogram(diagonal_lengths, bins=range(min_diagonal, max(diagonal_lengths) + 2))
        hist = hist / hist.sum() if hist.sum() > 0 else hist
        hist = hist[hist > 0]
        entropy_diagonal = -np.sum(hist * np.log(hist)) if len(hist) > 0 else 0
    else:
        determinism = 0.0
        avg_diagonal = 0.0
        max_diagonal = 0
        entropy_diagonal = 0.0

    # Find vertical lines (laminarity)
    vertical_lengths = []
    for col in range(n_vectors):
        column = recurrence_matrix[:, col]
        current_length = 0
        for val in column:
            if val:
                current_length += 1
            else:
                if current_length >= min_vertical:
                    vertical_lengths.append(current_length)
                current_length = 0
        if current_length >= min_vertical:
            vertical_lengths.append(current_length)

    if vertical_lengths:
        total_vertical_points = sum(vertical_lengths)
        laminarity = total_vertical_points / max(n_recurrent, 1)
        trapping_time = np.mean(vertical_lengths)
    else:
        laminarity = 0.0
        trapping_time = 0.0

    return {
        'determinism': float(np.clip(determinism, 0, 1)),
        'laminarity': float(np.clip(laminarity, 0, 1)),
        'entropy': float(entropy_diagonal),
        'recurrence_rate': float(recurrence_rate),
        'trapping_time': float(trapping_time),
        'max_diagonal': int(max_diagonal),
        'avg_diagonal': float(avg_diagonal)
    }


def _compute_windowed(
    series: np.ndarray,
    window_size: int,
    step_size: int,
    embedding_dim: int = 3,
    delay: int = 1,
    threshold: float = None,
    min_diagonal: int = 2,
    min_vertical: int = 2,
    max_vectors: int = 500,
) -> Dict[str, Any]:
    """Compute RQA metrics over rolling windows."""
    n = len(series)

    if n < window_size:
        return {
            'determinism': np.array([]),
            'laminarity': np.array([]),
            'entropy': np.array([]),
            'recurrence_rate': np.array([]),
            'trapping_time': np.array([]),
            'max_diagonal': np.array([]),
            'avg_diagonal': np.array([]),
            't': np.array([]),
            'window_size': window_size,
            'step_size': step_size,
        }

    t_values = []
    det_values = []
    lam_values = []
    ent_values = []
    rr_values = []
    tt_values = []
    max_diag_values = []
    avg_diag_values = []

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        window = series[start:end]

        result = _compute_static(window, embedding_dim, delay, threshold,
                                 min_diagonal, min_vertical, max_vectors)

        t_values.append(start + window_size // 2)
        det_values.append(result['determinism'])
        lam_values.append(result['laminarity'])
        ent_values.append(result['entropy'])
        rr_values.append(result['recurrence_rate'])
        tt_values.append(result['trapping_time'])
        max_diag_values.append(result['max_diagonal'])
        avg_diag_values.append(result['avg_diagonal'])

    return {
        'determinism': np.array(det_values),
        'laminarity': np.array(lam_values),
        'entropy': np.array(ent_values),
        'recurrence_rate': np.array(rr_values),
        'trapping_time': np.array(tt_values),
        'max_diagonal': np.array(max_diag_values),
        'avg_diagonal': np.array(avg_diag_values),
        't': np.array(t_values),
        'window_size': window_size,
        'step_size': step_size,
    }


def _compute_point(
    series: np.ndarray,
    t: int,
    window_size: int,
    embedding_dim: int = 3,
    delay: int = 1,
    threshold: float = None,
    min_diagonal: int = 2,
    min_vertical: int = 2,
    max_vectors: int = 500,
) -> Dict[str, Any]:
    """Compute RQA metrics at specific time t."""
    if t is None:
        raise ValueError("t is required for point mode")

    n = len(series)

    # Center window on t
    half_window = window_size // 2
    start = max(0, t - half_window)
    end = min(n, start + window_size)

    if end - start < window_size:
        start = max(0, end - window_size)

    window = series[start:end]

    n_vectors = len(window) - (embedding_dim - 1) * delay
    if n_vectors < 10:
        result = _empty_result()
        result['t'] = t
        result['window_start'] = start
        result['window_end'] = end
        return result

    result = _compute_static(window, embedding_dim, delay, threshold,
                             min_diagonal, min_vertical, max_vectors)
    result['t'] = t
    result['window_start'] = start
    result['window_end'] = end

    return result


def _empty_result() -> Dict[str, Any]:
    """Return empty result for insufficient data."""
    return {
        'determinism': 0.0,
        'laminarity': 0.0,
        'entropy': 0.0,
        'recurrence_rate': 0.0,
        'trapping_time': 0.0,
        'max_diagonal': 0,
        'avg_diagonal': 0.0
    }
