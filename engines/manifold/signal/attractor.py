"""
Attractor Engine.

Imports from primitives/embedding/ and primitives/dynamical/ (canonical).
Returns numbers only - ORTHON classifies attractor types.
"""

import numpy as np
from typing import Dict, Union


def compute(y: np.ndarray, embedding_dim: int = None, delay: int = None) -> Dict[str, Union[float, int]]:
    """
    Compute attractor properties of signal.

    Args:
        y: Signal values
        embedding_dim: Embedding dimension (auto-detected if None)
        delay: Time delay (auto-detected if None)

    Returns:
        dict with embedding_dim, correlation_dim, delay
        ORTHON interprets correlation_dim to classify attractor type.
    """
    result = {
        'embedding_dim': np.nan,
        'correlation_dim': np.nan,
        'delay': np.nan
    }

    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    if n < 100:
        return result

    if np.std(y) < 1e-10:
        result['correlation_dim'] = 0.0
        return result

    try:
        from engines.primitives.embedding.delay import (
            optimal_delay,
            optimal_dimension,
        )
        from engines.primitives.dynamical.dimension import correlation_dimension

        # Auto-detect parameters using primitives
        if delay is None:
            delay = optimal_delay(y, max_lag=min(100, n // 10), method='autocorr')

        if embedding_dim is None:
            embedding_dim = optimal_dimension(y, delay=delay, max_dim=10)

        # Compute correlation dimension using primitive
        corr_dim, log_r, log_C = correlation_dimension(
            y, dimension=embedding_dim, delay=delay
        )

        result = {
            'embedding_dim': int(embedding_dim),
            'correlation_dim': float(corr_dim) if not np.isnan(corr_dim) else np.nan,
            'delay': int(delay)
        }

    except ImportError:
        # Fallback if primitives not available
        result = _compute_fallback(y, embedding_dim, delay)
    except Exception:
        pass

    return result


def _compute_fallback(y: np.ndarray, embedding_dim: int = None, delay: int = None) -> Dict[str, Union[float, int]]:
    """Fallback computation without primitives."""
    from scipy.spatial.distance import pdist

    n = len(y)

    # Estimate delay from autocorrelation
    if delay is None:
        y_centered = y - np.mean(y)
        autocorr = np.correlate(y_centered, y_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr_max = autocorr[0]
        if autocorr_max > 0:
            autocorr = autocorr / autocorr_max

        delay = 1
        for i in range(1, min(len(autocorr), n // 4)):
            if autocorr[i] <= 0:
                delay = i
                break

    if embedding_dim is None:
        embedding_dim = min(5, max(2, n // (delay * 5)))

    # Embed
    m = n - (embedding_dim - 1) * delay
    if m < 50:
        return {
            'embedding_dim': int(embedding_dim),
            'correlation_dim': np.nan,
            'delay': int(delay)
        }

    embedded = np.zeros((m, embedding_dim))
    for i in range(embedding_dim):
        embedded[:, i] = y[i*delay:i*delay+m]

    # Sample for efficiency
    sample_size = min(500, m)
    if m > 500:
        indices = np.random.choice(m, sample_size, replace=False)
    else:
        indices = np.arange(m)

    dists = pdist(embedded[indices])
    dists = dists[dists > 0]

    if len(dists) < 100:
        return {
            'embedding_dim': int(embedding_dim),
            'correlation_dim': np.nan,
            'delay': int(delay)
        }

    # Correlation dimension via Grassberger-Procaccia
    radii = np.percentile(dists, [10, 20, 30, 40, 50])
    log_r, log_c = [], []
    for r in radii:
        c = np.sum(dists < r) / len(dists)
        if c > 0:
            log_r.append(np.log(r))
            log_c.append(np.log(c))

    corr_dim = np.nan
    if len(log_r) >= 3:
        corr_dim, _ = np.polyfit(log_r, log_c, 1)

    return {
        'embedding_dim': int(embedding_dim),
        'correlation_dim': float(corr_dim) if not np.isnan(corr_dim) else np.nan,
        'delay': int(delay)
    }
