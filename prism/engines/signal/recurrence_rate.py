"""Recurrence Rate Engine."""

import numpy as np
from typing import Dict
from .rqa import compute_recurrence_rate as _compute


def compute(y: np.ndarray) -> Dict[str, float]:
    """Compute recurrence rate."""
    return _compute(y)
