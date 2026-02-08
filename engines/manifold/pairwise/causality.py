"""
Causality Engine.

Computes directional causal measures (asymmetric):
- Granger causality
- Transfer entropy

Thin wrapper over primitives/pairwise/causality.py.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple

from engines.primitives.pairwise.causality import (
    granger_causality as _granger,
)
from engines.primitives.information import (
    transfer_entropy as _transfer_entropy,
)


def compute_granger(
    source: np.ndarray,
    target: np.ndarray,
    max_lag: int = 5,
) -> Dict[str, float]:
    """
    Test Granger causality from source to target.

    Does past values of source improve prediction of target
    beyond target's own past?

    Args:
        source: Potential cause signal
        target: Potential effect signal
        max_lag: Maximum lag to test

    Returns:
        dict with granger_f, granger_p, optimal_lag

    Note: ORTHON interprets significance (e.g., p < 0.05)
    """
    source = np.asarray(source).flatten()
    target = np.asarray(target).flatten()

    n = min(len(source), len(target))
    source, target = source[:n], target[:n]

    valid = ~(np.isnan(source) | np.isnan(target))
    source, target = source[valid], target[valid]

    if len(source) < max_lag + 20:
        return {
            'granger_f': np.nan,
            'granger_p': np.nan,
            'optimal_lag': np.nan,
        }

    f_stat, p_value, optimal_lag = _granger(source, target, max_lag=max_lag)

    return {
        'granger_f': float(f_stat),
        'granger_p': float(p_value),
        'optimal_lag': int(optimal_lag),
    }


def compute_transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    n_bins: int = 8,
) -> Dict[str, float]:
    """
    Compute transfer entropy from source to target.
    
    Information flow from source to target.
    
    Args:
        source: Source signal
        target: Target signal
        lag: Time lag
        n_bins: Discretization bins
        
    Returns:
        dict with transfer_entropy, normalized_te
    """
    source = np.asarray(source).flatten()
    target = np.asarray(target).flatten()
    
    n = min(len(source), len(target))
    source, target = source[:n], target[:n]
    
    valid = ~(np.isnan(source) | np.isnan(target))
    source, target = source[valid], target[valid]
    
    if len(source) < n_bins * 3:
        return {
            'transfer_entropy': np.nan,
            'normalized_te': np.nan,
        }
    
    try:
        te = _transfer_entropy(source, target, lag=lag, n_bins=n_bins)
        
        # Normalize by target entropy
        h_target = _entropy_1d(target, n_bins)
        normalized = te / h_target if h_target > 0 else 0.0
        
        return {
            'transfer_entropy': float(te),
            'normalized_te': float(normalized),
        }
    except Exception:
        return {
            'transfer_entropy': np.nan,
            'normalized_te': np.nan,
        }


def _entropy_1d(x: np.ndarray, n_bins: int) -> float:
    """Compute entropy of 1D distribution."""
    counts, _ = np.histogram(x, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))


def compute_bidirectional(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    max_lag: int = 5,
) -> Dict[str, Any]:
    """
    Compute bidirectional causality.

    Tests A→B and B→A. Returns numbers only - ORTHON classifies
    the relationship type (mutual, unidirectional, independent).

    Args:
        signal_a, signal_b: Two signals
        max_lag: Maximum lag

    Returns:
        dict with both directions and net causality
    """
    # A → B
    ab = compute_granger(signal_a, signal_b, max_lag)

    # B → A
    ba = compute_granger(signal_b, signal_a, max_lag)

    # Net causality (positive = A drives B, negative = B drives A)
    if not np.isnan(ab['granger_f']) and not np.isnan(ba['granger_f']):
        net_causality = ab['granger_f'] - ba['granger_f']
    else:
        net_causality = np.nan

    return {
        'granger_f_ab': ab['granger_f'],
        'granger_p_ab': ab['granger_p'],
        'granger_f_ba': ba['granger_f'],
        'granger_p_ba': ba['granger_p'],
        'net_causality': float(net_causality) if not np.isnan(net_causality) else None,
    }


def compute_all(
    source: np.ndarray,
    target: np.ndarray,
    max_lag: int = 5,
    te_lag: int = 1,
) -> Dict[str, Any]:
    """
    Compute all causality measures.
    
    Args:
        source, target: Signal values
        max_lag: For Granger
        te_lag: For transfer entropy
        
    Returns:
        Combined dict with all measures
    """
    result = {}
    result.update(compute_granger(source, target, max_lag))
    
    te_result = compute_transfer_entropy(source, target, lag=te_lag)
    for k, v in te_result.items():
        result[k] = v
    
    return result
