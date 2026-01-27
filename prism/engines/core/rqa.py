"""
Recurrence Quantification Analysis (RQA) Engine

Analyzes the recurrence structure of time series.

CANONICAL INTERFACE:
    def compute(observations: pd.DataFrame) -> pd.DataFrame
    Input:  [entity_id, signal_id, I, y]
    Output: [entity_id, signal_id, determinism, laminarity, entropy,
             recurrence_rate, trapping_time, max_diagonal, avg_diagonal]

Key measures:
    - DET (Determinism): % points on diagonal lines
    - LAM (Laminarity): % points on vertical lines
    - ENT (Entropy): Complexity of diagonal line distribution
    - TT (Trapping Time): Average vertical line length
    - RR (Recurrence Rate): % of recurrent points
"""

ENGINE_TYPE = 'signal'

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from typing import Dict, Any


def compute(
    observations: pd.DataFrame,
    embedding_dim: int = 3,
    delay: int = 1,
    threshold: float = None,
    min_diagonal: int = 2,
    min_vertical: int = 2,
    max_vectors: int = 500,
) -> pd.DataFrame:
    """
    Compute RQA metrics for all signals.

    CANONICAL INTERFACE:
        Input:  observations [entity_id, signal_id, I, y]
        Output: primitives [entity_id, signal_id, determinism, laminarity,
                           entropy, recurrence_rate, trapping_time,
                           max_diagonal, avg_diagonal]

    Parameters
    ----------
    observations : pd.DataFrame
        Must have columns: entity_id, signal_id, I, y
    embedding_dim : int, optional
        Phase space embedding dimension (default: 3)
    delay : int, optional
        Time delay for embedding (default: 1)
    threshold : float, optional
        Distance threshold for recurrence (default: 10% of range)
    min_diagonal : int, optional
        Minimum diagonal line length (default: 2)
    min_vertical : int, optional
        Minimum vertical line length (default: 2)
    max_vectors : int, optional
        Max vectors to use for large series (default: 500)

    Returns
    -------
    pd.DataFrame
        RQA metrics per signal
    """
    results = []

    for (entity_id, signal_id), group in observations.groupby(['entity_id', 'signal_id']):
        y = group.sort_values('I')['y'].values

        if len(y) < 50:
            results.append({
                'entity_id': entity_id,
                'signal_id': signal_id,
                'determinism': np.nan,
                'laminarity': np.nan,
                'entropy': np.nan,
                'recurrence_rate': np.nan,
                'trapping_time': np.nan,
                'max_diagonal': np.nan,
                'avg_diagonal': np.nan,
            })
            continue

        try:
            result = _compute_rqa(
                y, embedding_dim, delay, threshold,
                min_diagonal, min_vertical, max_vectors
            )
            results.append({
                'entity_id': entity_id,
                'signal_id': signal_id,
                **result
            })
        except Exception:
            results.append({
                'entity_id': entity_id,
                'signal_id': signal_id,
                'determinism': np.nan,
                'laminarity': np.nan,
                'entropy': np.nan,
                'recurrence_rate': np.nan,
                'trapping_time': np.nan,
                'max_diagonal': np.nan,
                'avg_diagonal': np.nan,
            })

    return pd.DataFrame(results)


def _compute_rqa(
    series: np.ndarray,
    embedding_dim: int,
    delay: int,
    threshold: float,
    min_diagonal: int,
    min_vertical: int,
    max_vectors: int,
) -> Dict[str, float]:
    """Compute RQA metrics on a single series."""
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
        entropy = -np.sum(hist * np.log(hist)) if len(hist) > 0 else 0
    else:
        determinism = 0.0
        avg_diagonal = 0.0
        max_diagonal = 0
        entropy = 0.0

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
        'entropy': float(entropy),
        'recurrence_rate': float(recurrence_rate),
        'trapping_time': float(trapping_time),
        'max_diagonal': int(max_diagonal),
        'avg_diagonal': float(avg_diagonal),
    }


def _empty_result() -> Dict[str, float]:
    """Return empty result for insufficient data."""
    return {
        'determinism': np.nan,
        'laminarity': np.nan,
        'entropy': np.nan,
        'recurrence_rate': np.nan,
        'trapping_time': np.nan,
        'max_diagonal': np.nan,
        'avg_diagonal': np.nan,
    }
