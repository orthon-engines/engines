"""
Divergence Computation
======================

Compute field divergence from Laplace fields.

Divergence indicates whether the system is:
- Source (+): Energy/information flowing outward
- Sink (-): Energy/information flowing inward
"""

import numpy as np
from typing import Dict

from prism.core.signals.types import LaplaceField


def compute_divergence(
    field_vectors: np.ndarray,
    s_values: np.ndarray,
) -> float:
    """
    Compute divergence in Laplace s-space.

    ∇·F = Σ_i Σ_s ∂²F_i/∂s²

    Args:
        field_vectors: [n_signals, n_s] complex array of F(s) values
        s_values: Laplace s-values

    Returns:
        Scalar divergence value
    """
    n_signals = field_vectors.shape[0]

    # Compute second derivative in s for each signal
    total_div = 0.0

    for i in range(n_signals):
        F_s = field_vectors[i]
        # Second derivative in s-space
        d2F_ds2 = np.gradient(np.gradient(F_s))
        total_div += np.sum(np.real(d2F_ds2))

    return float(total_div / n_signals)


def compute_divergence_from_fields(
    fields: Dict[str, LaplaceField],
    t: float,
) -> float:
    """
    Compute divergence at time t from a collection of fields.

    Args:
        fields: Dict mapping signal_id to LaplaceField
        t: Timestamp to compute divergence at

    Returns:
        Scalar divergence value
    """
    signal_ids = sorted(fields.keys())
    s_values = list(fields.values())[0].s_values

    # Get F(s) for each signal at time t
    field_vectors = np.array([fields[sid].at(t) for sid in signal_ids])

    return compute_divergence(field_vectors, s_values)


def compute_local_divergence(
    field: LaplaceField,
) -> np.ndarray:
    """
    Compute divergence at each timestamp for a single field.

    Args:
        field: LaplaceField for one signal

    Returns:
        Array of divergence values, one per timestamp
    """
    # field.field has shape [n_timestamps, n_s]
    # Compute ∂²F/∂s² and sum over s for each t

    divergence = []
    for t_idx in range(len(field.timestamps)):
        F_s = field.field[t_idx]
        d2F_ds2 = np.gradient(np.gradient(F_s))
        divergence.append(np.sum(np.real(d2F_ds2)))

    return np.array(divergence)


def compute_divergence_trajectory(
    fields: Dict[str, LaplaceField],
    timestamps: np.ndarray,
) -> np.ndarray:
    """
    Compute divergence at each timestamp.

    Args:
        fields: Dict mapping signal_id to LaplaceField
        timestamps: Timestamps to compute divergence at

    Returns:
        Array of divergence values
    """
    divergences = []
    for t in timestamps:
        div = compute_divergence_from_fields(fields, float(t))
        divergences.append(div)

    return np.array(divergences)
