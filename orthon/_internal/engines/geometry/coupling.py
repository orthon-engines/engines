"""
Coupling Computation
====================

Compare signals in Laplace domain with power-weighted coupling.
Signals only compared where BOTH have meaningful content.
"""

import numpy as np
from typing import Dict, List

from orthon._internal.core.signals.types import LaplaceField


def compute_coupling_matrix(
    field_vectors: np.ndarray,
    s_values: np.ndarray,
    fields: Dict[str, LaplaceField] = None,
    signal_ids: List[str] = None,
) -> np.ndarray:
    """
    Compute pairwise coupling weighted by shared information.

    Key insight: If signal A is sparse (low-freq only) and signal B is dense
    (full spectrum), only compare them in the low-freq region where A has content.

    Args:
        field_vectors: [n_signals, n_s] complex array of F(s) values at time t
        s_values: Laplace s-values
        fields: Original fields (for power spectrum info, optional)
        signal_ids: Signal ordering (optional)

    Returns:
        [n_signals, n_signals] coupling matrix
    """
    n_signals = field_vectors.shape[0]
    coupling = np.zeros((n_signals, n_signals))

    # Compute power spectrum for each signal
    power = np.abs(field_vectors) ** 2

    for i in range(n_signals):
        for j in range(i + 1, n_signals):
            # Weight by minimum power at each s
            # This ensures we only compare where BOTH have content
            weight = np.minimum(power[i], power[j])
            weight_sum = weight.sum()

            if weight_sum < 1e-10:
                coupling[i, j] = 0.0
            else:
                # Normalized weighted correlation in s-domain
                F_i = field_vectors[i]
                F_j = field_vectors[j]

                # Weighted inner product
                inner = np.sum(weight * np.real(F_i * np.conj(F_j)))
                norm_i = np.sqrt(np.sum(weight * np.abs(F_i) ** 2))
                norm_j = np.sqrt(np.sum(weight * np.abs(F_j) ** 2))

                if norm_i > 1e-10 and norm_j > 1e-10:
                    coupling[i, j] = inner / (norm_i * norm_j)
                else:
                    coupling[i, j] = 0.0

            coupling[j, i] = coupling[i, j]  # Symmetric

    # Diagonal = 1 (self-coupling)
    np.fill_diagonal(coupling, 1.0)

    return coupling


def compute_affinity_matrix(
    coupling_matrix: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Convert coupling to affinity (similarity) for clustering.

    Args:
        coupling_matrix: Pairwise coupling values
        threshold: Minimum coupling to consider (default 0)

    Returns:
        Affinity matrix (non-negative)
    """
    # Convert correlation-like values to non-negative affinities
    affinity = (coupling_matrix + 1) / 2  # Map [-1, 1] to [0, 1]
    affinity[affinity < threshold] = 0
    return affinity


def compute_distance_matrix(coupling_matrix: np.ndarray) -> np.ndarray:
    """
    Convert coupling to distance for MST/clustering.

    Args:
        coupling_matrix: Pairwise coupling values

    Returns:
        Distance matrix
    """
    # Higher coupling = smaller distance
    # Map from [-1, 1] to [0, 2]
    distance = 1 - coupling_matrix
    return distance
