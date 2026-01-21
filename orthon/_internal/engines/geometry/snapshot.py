"""
Geometry Snapshot
=================

Compute system geometry at a single timestamp from Laplace fields.
"""

import numpy as np
from typing import Dict, List

from orthon._internal.core.signals.types import LaplaceField, GeometrySnapshot
from orthon._internal.engines.geometry.coupling import compute_coupling_matrix
from orthon._internal.engines.geometry.divergence import compute_divergence
from orthon._internal.engines.geometry.modes import discover_modes


def compute_geometry_at_t(
    fields: Dict[str, LaplaceField],
    t: float,
    s_values: np.ndarray = None,
) -> GeometrySnapshot:
    """
    Compute full geometry snapshot at timestamp t.

    Args:
        fields: Dict mapping signal_id to LaplaceField
        t: Timestamp to compute geometry at
        s_values: Laplace s-values for comparison (if None, use from fields)

    Returns:
        GeometrySnapshot with coupling, divergence, modes
    """
    signal_ids = sorted(fields.keys())
    n_signals = len(signal_ids)

    if n_signals == 0:
        return GeometrySnapshot(
            timestamp=t,
            coupling_matrix=np.array([[]]),
            divergence=0.0,
            mode_labels=np.array([]),
            mode_coherence=np.array([]),
            signal_ids=[],
        )

    # Get s_values from first field if not provided
    if s_values is None:
        s_values = list(fields.values())[0].s_values

    # Get F(s) for each signal at time t
    field_vectors = np.zeros((n_signals, len(s_values)), dtype=np.complex128)

    for i, sid in enumerate(signal_ids):
        field_vectors[i] = fields[sid].at(t)

    # Compute pairwise coupling
    coupling_matrix = compute_coupling_matrix(
        field_vectors,
        s_values,
        fields,
        signal_ids,
    )

    # Compute divergence
    divergence = compute_divergence(field_vectors, s_values)

    # Discover modes
    mode_labels, mode_coherence = discover_modes(field_vectors, s_values)

    return GeometrySnapshot(
        timestamp=t,
        coupling_matrix=coupling_matrix,
        divergence=divergence,
        mode_labels=mode_labels,
        mode_coherence=mode_coherence,
        signal_ids=signal_ids,
    )


def compute_geometry_trajectory(
    fields: Dict[str, LaplaceField],
    timestamps: np.ndarray,
    s_values: np.ndarray = None,
) -> List[GeometrySnapshot]:
    """
    Compute geometry at each timestamp.

    Args:
        fields: All Laplace fields
        timestamps: Times to compute geometry
        s_values: Laplace s-values

    Returns:
        List of GeometrySnapshot, one per timestamp
    """
    snapshots = []

    for t in timestamps:
        snapshot = compute_geometry_at_t(fields, float(t), s_values)
        snapshots.append(snapshot)

    return snapshots


def snapshot_to_vector(snapshot: GeometrySnapshot) -> np.ndarray:
    """
    Convert GeometrySnapshot to a fixed-size flat vector for state trajectory.

    Uses geometry metrics to build position vector. Supports both old-style
    (coupling_matrix based) and new-style (engine metrics based) snapshots.

    Returns fixed-size vector for state trajectory computation.
    """
    # Check if snapshot has new-style metrics (from vector-based geometry)
    if hasattr(snapshot, 'pca_var_1'):
        # New-style geometry with engine metrics
        pos = np.array([
            getattr(snapshot, 'pca_var_1', 0.0),
            getattr(snapshot, 'pca_var_2', 0.0),
            getattr(snapshot, 'clustering_silhouette', 0.0),
            getattr(snapshot, 'mst_total_weight', 0.0),
            getattr(snapshot, 'lof_mean', 0.0),
            getattr(snapshot, 'distance_mean', 0.0),
        ])
        return pos

    # Old-style geometry with coupling matrix
    if snapshot.n_signals == 0:
        return np.array([0.0, 0.0, 0.0, 0.0, snapshot.divergence, 0.0])

    # Get coupling matrix statistics (fixed-size summary)
    coupling = snapshot.coupling_matrix
    if coupling.size > 0:
        # Upper triangle (excluding diagonal) for statistics
        n = coupling.shape[0]
        if n > 1:
            upper_tri = coupling[np.triu_indices(n, k=1)]
            coupling_mean = float(np.nanmean(upper_tri)) if len(upper_tri) > 0 else 0.0
            coupling_std = float(np.nanstd(upper_tri)) if len(upper_tri) > 0 else 0.0
            coupling_max = float(np.nanmax(upper_tri)) if len(upper_tri) > 0 else 0.0
        else:
            coupling_mean = 0.0
            coupling_std = 0.0
            coupling_max = 0.0
    else:
        coupling_mean = 0.0
        coupling_std = 0.0
        coupling_max = 0.0

    # Mode coherence summary
    mean_coherence = float(np.mean(snapshot.mode_coherence)) if len(snapshot.mode_coherence) > 0 else 0.0

    # Build fixed-size position vector
    pos = np.array([
        float(snapshot.n_signals),
        coupling_mean,
        coupling_std,
        coupling_max,
        snapshot.divergence,
        mean_coherence,
    ])

    return pos


def get_unified_timestamps(fields: Dict[str, LaplaceField]) -> np.ndarray:
    """
    Get unified timestamp grid from all fields.

    Uses union of all timestamps.
    """
    all_timestamps = set()
    for field in fields.values():
        all_timestamps.update(field.timestamps.astype(float))
    return np.array(sorted(all_timestamps))
