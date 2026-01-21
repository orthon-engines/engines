"""
PRISM Point-wise Engines
========================

Engines that produce values at every timestamp (native resolution).

These engines operate on individual observations and produce
DenseSignal outputs with the same length as input.

Engines:
    - hilbert: Instantaneous amplitude, phase, frequency
    - derivatives: Velocity, acceleration, jerk
    - statistical: Z-score, rolling statistics
"""

from orthon._internal.engines.pointwise.hilbert import (
    HilbertEngine,
    compute_hilbert_amplitude,
    compute_hilbert_phase,
    compute_hilbert_frequency,
)

from orthon._internal.engines.pointwise.derivatives import (
    DerivativesEngine,
    compute_velocity,
    compute_acceleration,
    compute_jerk,
)

from orthon._internal.engines.pointwise.statistical import (
    StatisticalEngine,
    compute_zscore,
    compute_rolling_mean,
    compute_rolling_std,
)

__all__ = [
    # Hilbert
    'HilbertEngine',
    'compute_hilbert_amplitude',
    'compute_hilbert_phase',
    'compute_hilbert_frequency',
    # Derivatives
    'DerivativesEngine',
    'compute_velocity',
    'compute_acceleration',
    'compute_jerk',
    # Statistical
    'StatisticalEngine',
    'compute_zscore',
    'compute_rolling_mean',
    'compute_rolling_std',
]
