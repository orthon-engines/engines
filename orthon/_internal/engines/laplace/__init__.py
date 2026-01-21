"""
PRISM Laplace Engines — Laplace domain computation.

The Laplace transform resolves time-varying metrics to field vectors,
enabling direct cross-signal comparison regardless of scale or sampling frequency.
"""

from orthon._internal.engines.laplace.transform import (
    compute_laplace_for_series,
    compute_gradient,
    compute_laplacian,
    compute_divergence_for_signal,
    add_divergence_to_field_rows,
    RunningLaplace,
    compute_laplace_field,
    laplace_gradient,
    laplace_divergence,
    laplace_energy,
    decompose_by_scale,
    frequency_shift,
)
from orthon._internal.engines.laplace.pairwise import (
    run_laplace_pairwise_vectorized,
    run_laplace_pairwise_windowed,
    extract_cohort_from_signal,
)

__all__ = [
    # Transform
    'compute_laplace_for_series',
    'compute_gradient',
    'compute_laplacian',
    'compute_divergence_for_signal',
    'add_divergence_to_field_rows',
    'RunningLaplace',
    'compute_laplace_field',
    'laplace_gradient',
    'laplace_divergence',
    'laplace_energy',
    'decompose_by_scale',
    'frequency_shift',
    # Pairwise
    'run_laplace_pairwise_vectorized',
    'run_laplace_pairwise_windowed',
    'extract_cohort_from_signal',
]
