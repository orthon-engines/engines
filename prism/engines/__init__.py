"""
PRISM Engines
=============

Flat directory containing all compute engines.
PRISM computes, ORTHON classifies.

Structure:
    signal/      - Per-signal engines (statistics, memory, complexity, spectral, etc.)
    state/       - State engines (centroid, eigendecomp)
    pairwise/    - Pairwise engines (correlation, causality)
    dynamics/    - Dynamics engines (lyapunov, attractor)
    sql/         - SQL-based engines
    rolling.py   - Generic rolling window wrapper

Legacy (flat files, to be consolidated):
    state_vector.py      - Centroid computation (WHERE)
    state_geometry.py    - Eigenvalue computation (SHAPE)
    signal_geometry.py   - Signal-to-centroid distances
    signal_pairwise.py   - Signal-to-signal relationships
    geometry_dynamics.py - Derivatives of geometry
"""

# Submodules
from prism.engines import signal
from prism.engines import state
from prism.engines import pairwise
from prism.engines import dynamics

# Rolling wrapper
from prism.engines.rolling import compute as rolling_compute

# Legacy flat files (for backwards compatibility)
from prism.engines.state_vector import compute_state_vector, compute_centroid
from prism.engines.state_geometry import compute_state_geometry, compute_eigenvalues
from prism.engines.signal_geometry import compute_signal_geometry
from prism.engines.signal_pairwise import compute_signal_pairwise
from prism.engines.geometry_dynamics import (
    compute_geometry_dynamics,
    compute_signal_dynamics,
    compute_pairwise_dynamics,
    compute_all_dynamics,
    compute_derivatives,
)

__all__ = [
    # Submodules
    'signal',
    'state',
    'pairwise',
    'dynamics',
    # Rolling
    'rolling_compute',
    # Legacy flat files
    'compute_state_vector',
    'compute_centroid',
    'compute_state_geometry',
    'compute_eigenvalues',
    'compute_signal_geometry',
    'compute_signal_pairwise',
    'compute_geometry_dynamics',
    'compute_signal_dynamics',
    'compute_pairwise_dynamics',
    'compute_all_dynamics',
    'compute_derivatives',
]
