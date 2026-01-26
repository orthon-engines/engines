"""
UMAP Dimensionality Reduction
=============================

Uniform Manifold Approximation and Projection.
System-level engine: operates on state vectors, not signals.

Stream mode: Accumulate all state vectors, compute once at end.
"""

import numpy as np
from typing import Dict, Any, Optional


def compute(
    X: np.ndarray,
    n_components: int = 3,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean'
) -> Dict[str, Any]:
    """
    Compute UMAP embedding.
    
    Args:
        X: Data matrix (n_samples, n_features)
        n_components: Target dimensions
        n_neighbors: Local neighborhood size
        min_dist: Minimum distance in embedding
        metric: Distance metric
    
    Returns:
        embedding: Reduced coordinates (n_samples, n_components)
        
    Note: Requires umap-learn package.
    """
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn required: pip install umap-learn")
    
    X = np.asarray(X)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    if len(X) < n_neighbors:
        n_neighbors = max(2, len(X) - 1)
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42  # Reproducibility
    )
    
    embedding = reducer.fit_transform(X)
    
    return {
        'embedding': embedding,
        'n_samples': len(X),
        'n_components': n_components
    }
