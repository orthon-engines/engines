"""
Minimal signal types stub for backwards compatibility.
"""

from dataclasses import dataclass
from typing import Optional, Any
import numpy as np


@dataclass
class DenseSignal:
    """Stub for DenseSignal - represents a dense time series."""
    data: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = np.array([])


@dataclass
class SparseSignal:
    """Stub for SparseSignal - represents a sparse time series."""
    data: Optional[np.ndarray] = None
    indices: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = np.array([])
        if self.indices is None:
            self.indices = np.array([])


@dataclass
class LaplaceField:
    """Stub for LaplaceField - not used in stream architecture."""
    data: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = np.array([])
