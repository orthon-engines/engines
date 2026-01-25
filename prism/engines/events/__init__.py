"""
PRISM Event Engines

Hybrid approach:
1. Detect events (steps, spikes) in time series
2. Compute REAL H(t-t0) and delta(t-t0) centered at each event
3. Store for downstream use (convolution, system ID, decomposition)

The functions are computed blind. They become relevant when they do.
"""

from prism.engines.events.heaviside_dirac import (
    # Event classes
    StepEvent,
    SpikeEvent,
    # Detection
    detect_steps,
    detect_spikes,
    # Real Heaviside function
    heaviside,
    heaviside_scaled,
    # Real Dirac delta
    dirac_delta_discrete,
    dirac_delta_gaussian,
    dirac_delta_scaled,
    # Construct basis from events
    construct_heaviside_basis,
    construct_dirac_basis,
    # Signal decomposition
    decompose_signal,
    # Main compute function
    compute,
)

__all__ = [
    'StepEvent',
    'SpikeEvent',
    'detect_steps',
    'detect_spikes',
    'heaviside',
    'heaviside_scaled',
    'dirac_delta_discrete',
    'dirac_delta_gaussian',
    'dirac_delta_scaled',
    'construct_heaviside_basis',
    'construct_dirac_basis',
    'decompose_signal',
    'compute',
]
