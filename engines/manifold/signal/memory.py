"""
Memory Engine.

Computes long-range dependence measures:
- Hurst exponent (R/S and DFA methods)
- Autocorrelation decay

Thin wrapper over primitives/individual/fractal.py and correlation.py.
Primitives handle min_samples via config - no redundant checks here.
"""

import numpy as np
from typing import Dict, Any

from engines.primitives.individual.fractal import (
    hurst_exponent,
    hurst_r2,
    dfa,
)
from engines.primitives.individual.correlation import (
    autocorrelation,
)


def compute(y: np.ndarray, method: str = 'rs') -> Dict[str, float]:
    """
    Compute memory/persistence measures.

    Args:
        y: Signal values
        method: 'rs' for rescaled range, 'dfa' for detrended fluctuation

    Returns:
        dict with hurst, hurst_r2

    Interpretation (ORTHON's job, but for reference):
        H < 0.5: anti-persistent (mean-reverting)
        H = 0.5: random walk
        H > 0.5: persistent (trending)
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]

    return {
        'hurst': hurst_exponent(y, method=method),
        'hurst_r2': hurst_r2(y),
    }


def compute_hurst(y: np.ndarray, method: str = 'rs') -> Dict[str, float]:
    """Compute Hurst exponent only."""
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    return {'hurst': hurst_exponent(y, method=method)}


def compute_dfa(y: np.ndarray) -> Dict[str, float]:
    """Compute DFA exponent."""
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    return {'dfa': dfa(y)}


def compute_acf_decay(y: np.ndarray, max_lag: int = 50) -> Dict[str, Any]:
    """
    Compute autocorrelation decay characteristics.

    Returns:
        dict with acf_lag1, acf_lag10, acf_half_life
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    if n < max_lag:
        max_lag = n // 2
    if max_lag < 2:
        return {
            'acf_lag1': np.nan,
            'acf_lag10': np.nan,
            'acf_half_life': np.nan,
        }

    # Get all ACF values then slice to max_lag
    acf_vals = autocorrelation(y)[:max_lag + 1]

    # Find half-life (lag where ACF < 0.5)
    half_life = np.nan
    for lag, ac in enumerate(acf_vals):
        if ac < 0.5:
            half_life = float(lag)
            break

    return {
        'acf_lag1': float(acf_vals[1]) if len(acf_vals) > 1 else np.nan,
        'acf_lag10': float(acf_vals[10]) if len(acf_vals) > 10 else np.nan,
        'acf_half_life': half_life,
    }
