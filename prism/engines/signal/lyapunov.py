"""
Lyapunov Exponent Engine.

Computes the largest Lyapunov exponent using Rosenstein's algorithm.
PRISM computes only - no classification.
"""

import numpy as np
from typing import Dict, Any


def compute(y: np.ndarray, min_samples: int = 200) -> Dict[str, Any]:
    """
    Compute Lyapunov exponent of signal.

    Uses Rosenstein's algorithm which tracks divergence of nearby
    trajectories in reconstructed phase space.

    Args:
        y: Signal values
        min_samples: Minimum samples required

    Returns:
        dict with:
            - 'lyapunov': Largest Lyapunov exponent (number only)
            - 'embedding_dim': Used embedding dimension
            - 'embedding_tau': Used time delay
            - 'confidence': Confidence in estimate (0-1)
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    if n < min_samples:
        return _empty_result()

    try:
        # Import primitives from package root
        from prism.primitives.embedding import (
            time_delay_embedding, optimal_delay, optimal_dimension
        )
        from prism.primitives.dynamical.lyapunov import lyapunov_rosenstein

        # Auto-detect embedding parameters
        tau = optimal_delay(y, max_lag=min(100, n // 10))
        dim = optimal_dimension(y, tau, max_dim=10)

        # Embed
        embedded = time_delay_embedding(y, dimension=dim, delay=tau)

        if len(embedded) < 50:
            return _empty_result()

        # Compute Lyapunov using correct algorithm
        lyap, divergence, iterations = lyapunov_rosenstein(embedded)

        # Compute confidence based on iterations and variance
        confidence = min(1.0, iterations / 100) if iterations else 0.5

        return {
            'lyapunov': float(lyap) if not np.isnan(lyap) else None,
            'embedding_dim': dim,
            'embedding_tau': tau,
            'confidence': confidence,
        }

    except ImportError:
        # Fallback if primitives not available - use simplified version
        return _compute_simplified(y)
    except Exception:
        return _empty_result()


def _compute_simplified(y: np.ndarray) -> Dict[str, Any]:
    """
    Simplified Lyapunov estimation (fallback).

    Uses basic nearest-neighbor divergence tracking.
    Less accurate but no dependencies.
    """
    n = len(y)

    # Fixed embedding parameters for simplicity
    dim = 3
    tau = max(1, n // 50)

    # Create embedding
    m = n - (dim - 1) * tau
    if m < 50:
        return _empty_result()

    embedded = np.zeros((m, dim))
    for d in range(dim):
        embedded[:, d] = y[d * tau:d * tau + m]

    # Track divergence of nearest neighbors
    divergences = []
    n_samples = min(100, m - 10)

    for i in range(n_samples):
        # Find nearest neighbor (excluding temporal neighbors)
        min_dist = np.inf
        j_nearest = -1

        for j in range(m):
            if abs(i - j) < dim:  # Exclude temporal neighbors
                continue
            dist = np.linalg.norm(embedded[i] - embedded[j])
            if dist < min_dist and dist > 1e-10:
                min_dist = dist
                j_nearest = j

        if j_nearest < 0:
            continue

        # Track divergence over next few steps
        for k in range(1, min(10, m - max(i, j_nearest))):
            d_k = np.linalg.norm(embedded[i + k] - embedded[j_nearest + k])
            if d_k > 1e-10 and min_dist > 1e-10:
                divergences.append(np.log(d_k / min_dist) / k)

    if len(divergences) < 10:
        return _empty_result()

    lyap = float(np.median(divergences))

    # Confidence based on divergence consistency
    divergence_std = np.std(divergences)
    confidence = 1.0 / (1.0 + divergence_std) if divergence_std < 10 else 0.1

    return {
        'lyapunov': lyap,
        'embedding_dim': dim,
        'embedding_tau': tau,
        'confidence': confidence,
    }


def _empty_result() -> Dict[str, Any]:
    """Return empty result for insufficient data."""
    return {
        'lyapunov': None,
        'embedding_dim': None,
        'embedding_tau': None,
        'confidence': 0.0,
    }
