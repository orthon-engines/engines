"""
Transition Matrix Engine
========================

Computes state transition statistics from sequential observations.
The transition matrix T[i,j] = P(state_j at t+1 | state_i at t).

Returns summary statistics of the matrix, NOT the raw matrix
(which would be variable-size and break parquet schema).

Designed for DISCRETE, BINARY, STEP, EVENT signal types.

Outputs:
    trans_entropy       - Entropy of transition matrix (bits). Higher = more random transitions.
    trans_self_loop     - Mean diagonal probability (tendency to stay in same state)
    trans_max_self_loop - Highest self-loop probability (stickiest state)
    trans_asymmetry     - Frobenius norm of (T - T^T), normalized. 0 = perfectly reversible.
    trans_n_active      - Number of transitions that actually occur (out of k^2 possible)
    trans_sparsity      - Fraction of zero entries in transition matrix

Physics:
    - trans_self_loop increasing → system getting stuck, losing mobility
    - trans_entropy decreasing → transitions becoming predictable (degradation)
    - trans_asymmetry increasing → system becoming irreversible (wear direction)
    - trans_sparsity increasing → fewer paths available (state space collapse)
"""

import numpy as np
from typing import Dict, Any


MIN_SAMPLES = 4


def compute(y: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """
    Compute transition matrix statistics.

    Parameters
    ----------
    y : np.ndarray
        Input signal (sequential observations).
    n_bins : int
        Number of bins for continuous signals.

    Returns
    -------
    dict
        Transition matrix summary statistics.
    """
    y = np.asarray(y).flatten()

    if len(y) < MIN_SAMPLES:
        raise ValueError(f"Need {MIN_SAMPLES} samples, got {len(y)}")

    # Remove NaNs but preserve order
    mask = ~np.isnan(y)
    y_clean = y[mask]

    if len(y_clean) < MIN_SAMPLES:
        raise ValueError(f"Need {MIN_SAMPLES} non-NaN samples, got {len(y_clean)}")

    # Discretize if continuous
    unique_vals = np.unique(y_clean)
    if len(unique_vals) > n_bins:
        bins = np.linspace(np.nanmin(y_clean), np.nanmax(y_clean), n_bins + 1)
        levels = np.digitize(y_clean, bins, right=True)
    else:
        val_to_label = {v: i for i, v in enumerate(sorted(unique_vals))}
        levels = np.array([val_to_label[v] for v in y_clean])

    unique_levels = np.unique(levels)
    k = len(unique_levels)

    if k < 2:
        # Only one state — no transitions possible
        return {
            'trans_entropy': 0.0,
            'trans_self_loop': 1.0,
            'trans_max_self_loop': 1.0,
            'trans_asymmetry': 0.0,
            'trans_n_active': 0,
            'trans_sparsity': 1.0,
        }

    # Map levels to 0..k-1
    level_map = {v: i for i, v in enumerate(unique_levels)}
    mapped = np.array([level_map[l] for l in levels])

    # Build count matrix
    T_counts = np.zeros((k, k), dtype=float)
    for i in range(len(mapped) - 1):
        T_counts[mapped[i], mapped[i + 1]] += 1

    # Normalize rows to get probabilities
    row_sums = T_counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid div by zero
    T_prob = T_counts / row_sums

    # Transition entropy: average entropy of each row
    row_entropies = []
    for i in range(k):
        row = T_prob[i]
        row_nonzero = row[row > 0]
        if len(row_nonzero) > 0:
            h = -np.sum(row_nonzero * np.log2(row_nonzero))
            row_entropies.append(h)
    trans_entropy = float(np.mean(row_entropies)) if row_entropies else 0.0

    # Self-loop statistics
    diag = np.diag(T_prob)
    trans_self_loop = float(np.mean(diag))
    trans_max_self_loop = float(np.max(diag))

    # Asymmetry: ||T - T^T||_F / ||T||_F
    diff = T_prob - T_prob.T
    norm_diff = np.sqrt(np.sum(diff ** 2))
    norm_T = np.sqrt(np.sum(T_prob ** 2))
    trans_asymmetry = float(norm_diff / max(norm_T, 1e-12))

    # Active transitions and sparsity
    n_active = int(np.sum(T_counts > 0))
    total_possible = k * k
    trans_sparsity = float(1.0 - n_active / total_possible)

    return {
        'trans_entropy': trans_entropy,
        'trans_self_loop': trans_self_loop,
        'trans_max_self_loop': trans_max_self_loop,
        'trans_asymmetry': trans_asymmetry,
        'trans_n_active': n_active,
        'trans_sparsity': trans_sparsity,
    }
