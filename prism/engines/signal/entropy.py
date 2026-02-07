"""
Discrete Entropy Engine
=======================

Computes entropy measures appropriate for discrete and state-machine signals.
Distinct from complexity engines (sample_entropy, permutation_entropy) which
target continuous signals.

Designed for DISCRETE, BINARY, STEP, EVENT signal types.

Outputs:
    shannon_entropy     - Shannon entropy of level distribution (bits)
    normalized_entropy  - Shannon / log2(n_levels), range [0, 1]
    conditional_entropy - H(X_t+1 | X_t), uncertainty given current state
    entropy_rate        - Limiting entropy per symbol (approx via conditional)
    excess_entropy      - shannon_entropy - conditional_entropy (predictability gain from memory)

Physics:
    - normalized_entropy → 1.0: signal is random across states
    - normalized_entropy → 0.0: signal is stuck in one state
    - conditional_entropy < shannon_entropy: transitions are predictable
    - excess_entropy rising: system developing stronger patterns
    - excess_entropy falling: system losing structure (randomizing)
"""

import numpy as np
from typing import Dict, Any


MIN_SAMPLES = 4


def compute(y: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """
    Compute discrete entropy measures.

    Parameters
    ----------
    y : np.ndarray
        Input signal.
    n_bins : int
        Number of bins for continuous signals.

    Returns
    -------
    dict
        Entropy measures for discrete signals.
    """
    y = np.asarray(y).flatten()
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
    n = len(levels)

    # --- Shannon entropy ---
    _, counts = np.unique(levels, return_counts=True)
    probs = counts / n
    shannon = float(-np.sum(probs * np.log2(probs + 1e-12)))

    # Normalized entropy
    max_entropy = np.log2(k) if k > 1 else 1.0
    normalized = float(shannon / max_entropy) if max_entropy > 0 else 0.0

    # --- Conditional entropy H(X_t+1 | X_t) ---
    if k < 2 or n < 3:
        conditional = 0.0
    else:
        # Build transition counts
        level_map = {v: i for i, v in enumerate(unique_levels)}
        mapped = np.array([level_map[l] for l in levels])

        T_counts = np.zeros((k, k), dtype=float)
        for i in range(len(mapped) - 1):
            T_counts[mapped[i], mapped[i + 1]] += 1

        # H(X_t+1 | X_t) = sum_i P(X_t=i) * H(X_t+1 | X_t=i)
        row_sums = T_counts.sum(axis=1)
        state_probs = row_sums / max(row_sums.sum(), 1e-12)

        cond_h = 0.0
        for i in range(k):
            if row_sums[i] == 0:
                continue
            row_probs = T_counts[i] / row_sums[i]
            row_nonzero = row_probs[row_probs > 0]
            h_row = -np.sum(row_nonzero * np.log2(row_nonzero))
            cond_h += state_probs[i] * h_row

        conditional = float(cond_h)

    # Entropy rate ≈ conditional entropy for stationary Markov chains
    entropy_rate = conditional

    # Excess entropy: how much does knowing the current state reduce uncertainty?
    excess = shannon - conditional

    return {
        'shannon_entropy': shannon,
        'normalized_entropy': normalized,
        'conditional_entropy': conditional,
        'entropy_rate': entropy_rate,
        'excess_entropy': float(excess),
    }
