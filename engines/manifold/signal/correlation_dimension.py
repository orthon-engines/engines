"""Correlation Dimension Engine."""

import numpy as np
from typing import Dict
from .rqa import compute_correlation_dimension as _compute


def compute(y: np.ndarray) -> Dict[str, float]:
    """Compute correlation dimension (Grassberger-Procaccia)."""
    return _compute(y)
