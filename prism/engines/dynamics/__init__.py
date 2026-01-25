"""
Dynamics Axis Engines
=====================

Computation engines for the Dynamics axis:
- lyapunov: Largest Lyapunov exponent
- embedding: Embedding dimension estimation
- phase_space: Phase space reconstruction
- hd_slope: Degradation rate (distance from baseline over time)
- attractor: Takens embedding, correlation dimension, Lyapunov (full reconstruction)
- phase_position: Track position on reconstructed attractor
- basin: Basin membership and transition analysis
"""

from .lyapunov import compute as compute_lyapunov
from .embedding import compute as compute_embedding
from .phase_space import compute as compute_phase_space
from .hd_slope import compute_hd_slope
from .regime_detector import compute as compute_regime
from .transition_detector import compute as compute_transitions
from .attractor import compute as compute_attractor, AttractorReconstructor
from .phase_position import compute as compute_phase_position, PhaseTracker
from .basin import compute as compute_basin, BasinAnalyzer, detect_basin_transitions

__all__ = [
    'compute_lyapunov',
    'compute_embedding',
    'compute_phase_space',
    'compute_hd_slope',
    'compute_regime',
    'compute_transitions',
    # Attractor reconstruction
    'compute_attractor',
    'AttractorReconstructor',
    # Phase position tracking
    'compute_phase_position',
    'PhaseTracker',
    # Basin analysis
    'compute_basin',
    'BasinAnalyzer',
    'detect_basin_transitions',
]
