"""
Persistent Homology Engine
==========================

Computes topological invariants of a state-space trajectory.

Given a point cloud (trajectory segment in feature space), computes
persistence diagrams via Vietoris-Rips filtration:
- H0: connected components (β₀)
- H1: loops/cycles (β₁)

From persistence diagrams, extracts:
- Betti numbers at optimal scale
- Persistence entropy (diversity of feature lifetimes)
- Max/total persistence (dominant structure strength)
- Number of significant features (above noise threshold)

MANIFOLD computes topology. ORTHON interprets what fragmentation means.
"""

import numpy as np
from typing import Dict, Any, Optional


def compute(
    trajectory: np.ndarray,
    min_samples: int = 10,
    max_dim: int = 1,
    subsample: int = 500,
) -> Dict[str, Any]:
    """
    Compute persistent homology of a trajectory (point cloud).

    Args:
        trajectory: (n_points, n_dims) point cloud in feature space
        min_samples: Minimum points required
        max_dim: Maximum homology dimension (1 = H0 + H1)
        subsample: Max points before subsampling (O(n³) complexity)

    Returns:
        dict with keys:
            betti_0, betti_1,
            persistence_entropy_h0, persistence_entropy_h1,
            max_persistence_h0, max_persistence_h1,
            total_persistence_h0, total_persistence_h1,
            n_significant_h0, n_significant_h1
    """
    from manifold.primitives.topology import (
        persistence_diagram,
        betti_numbers,
        persistence_entropy,
    )

    trajectory = np.asarray(trajectory, dtype=np.float64)

    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)

    # Remove rows with any NaN
    valid_mask = ~np.any(np.isnan(trajectory), axis=1)
    trajectory = trajectory[valid_mask]

    n = len(trajectory)
    if n < min_samples:
        return _empty_result()

    # Subsample if too large (persistence is O(n³))
    subsampled = False
    if n > subsample:
        indices = np.random.choice(n, subsample, replace=False)
        indices.sort()
        trajectory = trajectory[indices]
        subsampled = True

    try:
        # Compute persistence diagram
        diagrams = persistence_diagram(trajectory, max_dimension=max_dim)

        # Betti numbers (count features with persistence above noise threshold)
        noise_thresh = _estimate_noise_threshold(diagrams)
        betti = betti_numbers(diagrams, threshold=noise_thresh)

        # Also get raw betti (no threshold) for comparison
        betti_raw = betti_numbers(diagrams)

        # Persistence entropy per dimension
        ent_h0 = persistence_entropy(diagrams, dimension=0)
        ent_h1 = persistence_entropy(diagrams, dimension=1) if max_dim >= 1 else np.nan

        # Summary metrics per dimension
        max_pers_h0, total_pers_h0 = _persistence_summary(diagrams, 0)
        max_pers_h1, total_pers_h1 = _persistence_summary(diagrams, 1) if max_dim >= 1 else (np.nan, np.nan)

        # Significant features (persistence > noise threshold)
        n_sig_h0 = _count_significant(diagrams, 0, noise_thresh)
        n_sig_h1 = _count_significant(diagrams, 1, noise_thresh) if max_dim >= 1 else 0

        return {
            'betti_0': betti.get(0, 0),
            'betti_1': betti.get(1, 0),
            'persistence_entropy_h0': ent_h0,
            'persistence_entropy_h1': ent_h1,
            'max_persistence_h0': max_pers_h0,
            'max_persistence_h1': max_pers_h1,
            'total_persistence_h0': total_pers_h0,
            'total_persistence_h1': total_pers_h1,
            'n_significant_h0': n_sig_h0,
            'n_significant_h1': n_sig_h1,
            'subsampled': subsampled,
            'n_points': len(trajectory),
        }

    except Exception:
        return _empty_result()


def _estimate_noise_threshold(diagrams: Dict[int, np.ndarray]) -> float:
    """
    Estimate noise threshold from persistence diagram.

    Uses the median persistence of H0 features as the noise floor.
    Features with persistence below this are likely noise.
    """
    all_pers = []
    for dim, dgm in diagrams.items():
        if len(dgm) == 0:
            continue
        persistence = dgm[:, 1] - dgm[:, 0]
        finite = persistence[np.isfinite(persistence)]
        if len(finite) > 0:
            all_pers.extend(finite.tolist())

    if not all_pers:
        return 0.0

    return float(np.median(all_pers))


def _persistence_summary(
    diagrams: Dict[int, np.ndarray],
    dimension: int,
) -> tuple:
    """Extract max and total persistence for a given dimension."""
    if dimension not in diagrams or len(diagrams[dimension]) == 0:
        return np.nan, np.nan

    dgm = diagrams[dimension]
    persistence = dgm[:, 1] - dgm[:, 0]
    finite = persistence[np.isfinite(persistence)]

    if len(finite) == 0:
        return np.nan, np.nan

    return float(np.max(finite)), float(np.sum(finite))


def _count_significant(
    diagrams: Dict[int, np.ndarray],
    dimension: int,
    threshold: float,
) -> int:
    """Count features with persistence above threshold."""
    if dimension not in diagrams or len(diagrams[dimension]) == 0:
        return 0

    dgm = diagrams[dimension]
    persistence = dgm[:, 1] - dgm[:, 0]
    finite = persistence[np.isfinite(persistence)]

    return int(np.sum(finite > threshold))


def _empty_result() -> Dict[str, Any]:
    """Return empty result for insufficient data."""
    return {
        'betti_0': None,
        'betti_1': None,
        'persistence_entropy_h0': None,
        'persistence_entropy_h1': None,
        'max_persistence_h0': None,
        'max_persistence_h1': None,
        'total_persistence_h0': None,
        'total_persistence_h1': None,
        'n_significant_h0': None,
        'n_significant_h1': None,
        'subsampled': False,
        'n_points': 0,
    }
