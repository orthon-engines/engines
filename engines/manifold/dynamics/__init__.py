"""
Dynamics Engines.

Temporal dynamics and chaos analysis.
- lyapunov: stability/chaos indicator (Rosenstein algorithm)
- attractor: phase space reconstruction
- critical_slowing_down: early warning signals for B-tipping
- formal_definitions: classification framework for stability analysis
- ftle: finite-time Lyapunov exponent (trajectory-dependent stability)
- trajectory_sensitivity: variable importance at current state
- saddle_detection: unstable equilibria detection
"""

from . import lyapunov
from . import attractor
from . import critical_slowing_down
from . import formal_definitions
from . import ftle
from . import trajectory_sensitivity
from . import saddle_detection

# Convenience imports
from .lyapunov import compute as compute_lyapunov
from .critical_slowing_down import compute as compute_csd
from .ftle import compute as compute_ftle
from .trajectory_sensitivity import compute as compute_trajectory_sensitivity
from .saddle_detection import compute as compute_saddle_detection
from .formal_definitions import (
    AttractorType,
    StabilityType,
    FailureMode,
    TippingType,
    SystemTypology,
    GeometryMetrics,
    MassMetrics,
    EarlyWarningSignals,
    FormalAssessment,
    classify_failure_mode,
    classify_tipping_type,
    classify_stability,
)

__all__ = [
    # Modules
    'lyapunov',
    'attractor',
    'critical_slowing_down',
    'formal_definitions',
    'ftle',
    'trajectory_sensitivity',
    'saddle_detection',
    # Functions
    'compute_lyapunov',
    'compute_csd',
    'compute_ftle',
    'compute_trajectory_sensitivity',
    'compute_saddle_detection',
    # Enums
    'AttractorType',
    'StabilityType',
    'FailureMode',
    'TippingType',
    'SystemTypology',
    # Dataclasses
    'GeometryMetrics',
    'MassMetrics',
    'EarlyWarningSignals',
    'FormalAssessment',
    # Classification
    'classify_failure_mode',
    'classify_tipping_type',
    'classify_stability',
]
