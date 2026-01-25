"""
Dynamics Axis Engines
=====================

Computation engines for the Dynamics axis:
- lyapunov: Largest Lyapunov exponent
- embedding: Embedding dimension estimation
- phase_space: Phase space reconstruction
- hd_slope: Degradation rate (distance from baseline over time)
"""

from .lyapunov import compute as compute_lyapunov
from .embedding import compute as compute_embedding
from .phase_space import compute as compute_phase_space
from .hd_slope import compute_hd_slope

__all__ = [
    'compute_lyapunov',
    'compute_embedding',
    'compute_phase_space',
    'compute_hd_slope',
]
