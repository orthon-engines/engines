"""
Lyapunov Engine.

Computes largest Lyapunov exponent using Rosenstein's algorithm.
Measures rate of divergence of nearby trajectories in phase space.

ENGINES computes, ORTHON interprets:
    λ > 0: Chaos (trajectories diverge)
    λ ≈ 0: Quasi-periodic (trajectories parallel)
    λ < 0: Stable (trajectories converge)
"""

import numpy as np
from typing import Dict, Any, Optional

from engines.primitives.embedding import (
    time_delay_embedding,
    optimal_delay,
    optimal_dimension,
)
from engines.primitives.dynamical.lyapunov import (
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
    Compute Lyapunov exponent.
    
    Args:
        y: Signal values
        min_samples: Minimum samples required
        method: 'rosenstein' or 'kantz'
        emb_dim: Embedding dimension (auto if None)
        emb_tau: Embedding delay (auto if None)
        
    Returns:
        dict with lyapunov, embedding_dim, embedding_tau, confidence
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
        if emb_dim is None:
            emb_dim = optimal_dimension(y, emb_tau, max_dim=10)
        
        # Embed signal
        embedded = time_delay_embedding(y, dimension=emb_dim, delay=emb_tau)
        
        if len(embedded) < 50:
            return _empty_result()
        
        # Compute Lyapunov
        if method == 'kantz':
            lyap, divergence, iterations = lyapunov_kantz(embedded)
        else:
            lyap, divergence, iterations = lyapunov_rosenstein(embedded)
        
        # Confidence based on iterations
        confidence = min(1.0, iterations / 100) if iterations else 0.5
        
        return {
            'lyapunov': float(lyap) if not np.isnan(lyap) else None,
            'embedding_dim': emb_dim,
            'embedding_tau': emb_tau,
            'confidence': confidence,
        }
        
    except Exception:
        return _empty_result()


def _empty_result() -> Dict[str, Any]:
    """Return empty result for insufficient data."""
    return {
        'lyapunov': None,
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
    Compute rolling Lyapunov exponent.
    
    Args:
        y: Signal values
        window: Window size (recommend 500+)
        stride: Step size
        min_samples: Min samples per window
        
    Returns:
        dict with rolling_lyapunov, rolling_lyapunov_confidence
    """
    y = np.asarray(y).flatten()
    n = len(y)
    
    if n < window or window < min_samples:
        return {
            'rolling_lyapunov': np.full(n, np.nan),
            'rolling_lyapunov_confidence': np.full(n, np.nan),
        }
    
    lyap_values = np.full(n, np.nan)
    conf_values = np.full(n, np.nan)
    
    for i in range(0, n - window + 1, stride):
        chunk = y[i:i + window]
        result = compute(chunk, min_samples=min_samples)
        
        idx = i + window - 1
        if result['lyapunov'] is not None:
            lyap_values[idx] = result['lyapunov']
            conf_values[idx] = result['confidence']
    
    return {
        'rolling_lyapunov': lyap_values,
        'rolling_lyapunov_confidence': conf_values,
    }


def compute_trend(lyap_values: np.ndarray) -> Dict[str, float]:
    """
    Compute trend statistics on Lyapunov values.

    Returns numbers only - ORTHON interprets what "destabilizing" means.
    """
    valid = ~np.isnan(lyap_values)
    if np.sum(valid) < 4:
        return {
            'lyapunov_slope': np.nan,
            'lyapunov_r2': np.nan,
        }

    x = np.arange(len(lyap_values))[valid]
    y = lyap_values[valid]

    slope, intercept = np.polyfit(x, y, 1)

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        'lyapunov_slope': float(slope),
        'lyapunov_r2': float(r2),
    }
