"""
Discontinuity Engines
=====================

Structural discontinuity detection:
- step_detector: Step detection (permanent level shifts)
- spike_detector: Spike detection (transient shocks that decay)
- structural: Interval analysis, acceleration detection

HONEST NAMING:
    Previously these were called "dirac" and "heaviside" but that was
    misleading - they detect WHERE events occur, they don't compute
    the actual mathematical functions.

    Now renamed to what they actually do:
    - step_detector: Detects step changes (level shifts)
    - spike_detector: Detects spike events (impulses)
"""

# Import from new honest location
from prism.engines.detection.step_detector import compute as compute_step
from prism.engines.detection.spike_detector import compute as compute_spike
from .structural import compute as compute_structural

# Backwards compatibility aliases (deprecated)
compute_dirac = compute_spike
compute_heaviside = compute_step

__all__ = [
    'compute_step',
    'compute_spike',
    'compute_structural',
    # Deprecated aliases
    'compute_dirac',
    'compute_heaviside',
]
