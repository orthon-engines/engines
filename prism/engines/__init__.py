"""
PRISM Engines
=============

Self-describing compute engines with configuration.
PRISM computes, ORTHON classifies.

Structure:
    base.py      - BaseEngine class with self-configuration
    registry.py  - EngineRegistry for discovery and loading
    signal/      - Per-signal engines (statistics, memory, complexity, spectral, etc.)
                   Each engine has a .yaml config file alongside its .py file
    state/       - State engines (centroid, eigendecomp)
    pairwise/    - Pairwise engines (correlation, causality)
    dynamics/    - Dynamics engines (lyapunov, attractor)
    sql/         - SQL-based engines
    rolling.py   - Generic rolling window wrapper

Engine Configuration:
    Each engine defines its requirements in a .yaml file:
    - base_window: Default window size
    - min_window: Minimum samples required
    - max_window: Maximum window size
    - scaling: How window scales with window_factor (linear, sqrt, log)
    - outputs: List of output column names

Legacy (flat files, to be consolidated):
    state_vector.py      - Centroid computation (WHERE)
    state_geometry.py    - Eigenvalue computation (SHAPE)
    signal_geometry.py   - Signal-to-centroid distances
    signal_pairwise.py   - Signal-to-signal relationships
    geometry_dynamics.py - Derivatives of geometry
"""

# Registry (new architecture)
from prism.engines.registry import get_registry, EngineRegistry
from prism.engines.base import BaseEngine, EngineConfig, EngineRequirements

# Submodules
from prism.engines import signal
from prism.engines import state
from prism.engines import pairwise
from prism.engines import dynamics

# Rolling wrapper
from prism.engines.rolling import compute as rolling_compute

# Normalization engine
from prism.engines.normalization import (
    normalize,
    compute_zscore,
    compute_robust,
    compute_mad,
    compute_minmax,
    recommend_method,
    inverse_normalize,
    NormMethod,
)

# Legacy flat files (for backwards compatibility)
# Note: compute functions come from the actual engine implementations
from prism.engines.state.centroid import compute as compute_centroid
from prism.engines.state.eigendecomp import compute as compute_eigenvalues
from prism.engines.signal_geometry import compute_signal_geometry
from prism.engines.signal_pairwise import compute_signal_pairwise

# Lazy import to avoid circular dependency
def compute_state_vector(*args, **kwargs):
    """Lazy wrapper - imports from entry_points on first call."""
    import importlib
    ep = importlib.import_module('prism.entry_points.02_state_vector')
    return ep.compute_state_vector(*args, **kwargs)

def compute_state_geometry(*args, **kwargs):
    """Lazy wrapper - imports from entry_points on first call."""
    import importlib
    ep = importlib.import_module('prism.entry_points.03_state_geometry')
    return ep.compute_state_geometry(*args, **kwargs)
from prism.engines.geometry_dynamics import (
    compute_geometry_dynamics,
    compute_signal_dynamics,
    compute_pairwise_dynamics,
    compute_all_dynamics,
    compute_derivatives,
)

__all__ = [
    # Registry (new architecture)
    'get_registry',
    'EngineRegistry',
    'BaseEngine',
    'EngineConfig',
    'EngineRequirements',
    # Submodules
    'signal',
    'state',
    'pairwise',
    'dynamics',
    # Rolling
    'rolling_compute',
    # Normalization
    'normalize',
    'compute_zscore',
    'compute_robust',
    'compute_mad',
    'compute_minmax',
    'recommend_method',
    'inverse_normalize',
    'NormMethod',
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
