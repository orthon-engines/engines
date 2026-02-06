"""Embedding Dimension Engine."""

import numpy as np
from typing import Dict
from .rqa import compute_embedding_dim as _compute


def compute(y: np.ndarray) -> Dict[str, float]:
    """Compute optimal embedding dimension (False Nearest Neighbors)."""
    return _compute(y)
