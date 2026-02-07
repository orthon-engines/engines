"""
FTLE Engine - Finite-Time Lyapunov Exponents
=============================================

Computes finite-time Lyapunov exponents using Rosenstein's algorithm.
FTLE measures rate of divergence of nearby trajectories over finite windows.

Unlike classical Lyapunov exponents (which assume infinite time and ergodicity),
FTLE:
    - Works on finite windows → time-varying field
    - Handles transient, non-stationary data
    - Ridges in FTLE field = Lagrangian Coherent Structures (LCS)
    - LCS = regime boundaries, transition corridors, failure basins

The astrodynamics of your bearings.

PRISM computes FTLE values. ORTHON interprets ridges as regime boundaries.
"""

import numpy as np
from typing import Dict, Any, Optional

from prism.primitives.embedding import (
    optimal_delay,
    optimal_dimension,
)
from prism.primitives.dynamical.lyapunov import (
    lyapunov_rosenstein,
    lyapunov_kantz,
)


def compute(
    y: np.ndarray,
    min_samples: int = 200,
    method: str = 'rosenstein',
    emb_dim: Optional[int] = None,
    emb_tau: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute FTLE (Finite-Time Lyapunov Exponent).
    
    Args:
        y: Signal values
        min_samples: Minimum samples required
        method: 'rosenstein' or 'kantz'
        emb_dim: Embedding dimension (auto if None)
        emb_tau: Embedding delay (auto if None)
        
    Returns:
        dict with ftle, ftle_std, embedding_dim, embedding_tau, confidence
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)
    
    if n < min_samples:
        return _empty_result()
    
    try:
        # Auto-detect embedding parameters if not provided
        if emb_tau is None:
            emb_tau = optimal_delay(y, max_lag=min(100, n // 10))
            # Cap delay to ensure enough embedded points
            max_tau = n // 20  # Ensure at least 20 embedded points
            emb_tau = min(emb_tau, max_tau)
        if emb_dim is None:
            emb_dim = optimal_dimension(y, emb_tau, max_dim=10)

        # Check if embedding would leave enough points
        embedded_length = n - (emb_dim - 1) * emb_tau
        if embedded_length < 50:
            return _empty_result()

        # Compute FTLE - pass raw signal with embedding params
        # (lyapunov_rosenstein does its own embedding internally)
        if method == 'kantz':
            ftle, divergence, iterations = lyapunov_kantz(
                y, dimension=emb_dim, delay=emb_tau
            )
        else:
            ftle, divergence, iterations = lyapunov_rosenstein(
                y, dimension=emb_dim, delay=emb_tau
            )
        
        # Compute std of divergence rates if divergence curve available
        ftle_std = float(np.std(divergence)) if divergence is not None and len(divergence) > 1 else 0.0

        # Confidence based on iterations (iterations is an array of indices)
        if iterations is not None and len(iterations) > 0:
            confidence = min(1.0, len(iterations) / 100)
        else:
            confidence = 0.5

        return {
            'ftle': float(ftle) if not np.isnan(ftle) else None,
            'ftle_std': ftle_std,
            'embedding_dim': emb_dim,
            'embedding_tau': emb_tau,
            'confidence': confidence,
        }
        
    except Exception:
        return _empty_result()


def _empty_result() -> Dict[str, Any]:
    """Return empty result for insufficient data."""
    return {
        'ftle': None,
        'ftle_std': None,
        'embedding_dim': None,
        'embedding_tau': None,
        'confidence': 0.0,
    }


def compute_rolling(
    y: np.ndarray,
    window: int = 500,
    stride: int = 50,
    min_samples: int = 200,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling FTLE field along time axis.
    
    This produces the time-varying FTLE field that enables LCS detection.
    Ridges in the FTLE field correspond to dynamical barriers.
    
    Args:
        y: Signal values
        window: Window size (recommend 500+)
        stride: Step size
        min_samples: Min samples per window
        
    Returns:
        dict with rolling_ftle, rolling_ftle_std, rolling_ftle_confidence
    """
    y = np.asarray(y).flatten()
    n = len(y)
    
    if n < window or window < min_samples:
        return {
            'rolling_ftle': np.full(n, np.nan),
            'rolling_ftle_std': np.full(n, np.nan),
            'rolling_ftle_confidence': np.full(n, np.nan),
        }
    
    ftle_values = np.full(n, np.nan)
    std_values = np.full(n, np.nan)
    conf_values = np.full(n, np.nan)
    
    for i in range(0, n - window + 1, stride):
        chunk = y[i:i + window]
        result = compute(chunk, min_samples=min_samples)
        
        idx = i + window - 1
        if result['ftle'] is not None:
            ftle_values[idx] = result['ftle']
            std_values[idx] = result['ftle_std']
            conf_values[idx] = result['confidence']
    
    return {
        'rolling_ftle': ftle_values,
        'rolling_ftle_std': std_values,
        'rolling_ftle_confidence': conf_values,
    }


def compute_trend(ftle_values: np.ndarray) -> Dict[str, float]:
    """
    Compute trend statistics on FTLE values.

    Returns numbers only - ORTHON interprets what "destabilizing" means.
    """
    valid = ~np.isnan(ftle_values)
    if np.sum(valid) < 4:
        return {
            'ftle_slope': np.nan,
            'ftle_r2': np.nan,
        }

    x = np.arange(len(ftle_values))[valid]
    y = ftle_values[valid]

    slope, intercept = np.polyfit(x, y, 1)

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        'ftle_slope': float(slope),
        'ftle_r2': float(r2),
    }


# Backward compatibility alias
lyapunov = compute
compute_lyapunov = compute
