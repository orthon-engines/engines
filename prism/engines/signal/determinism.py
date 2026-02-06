"""Determinism Engine (RQA)."""

import numpy as np
from typing import Dict
from .rqa import compute_determinism as _compute


def compute(y: np.ndarray) -> Dict[str, float]:
    """Compute determinism from recurrence plot."""
    return _compute(y)
