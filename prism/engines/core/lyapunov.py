"""
Lyapunov Exponent - REAL Implementation

Largest Lyapunov exponent (lambda) measures the rate of
separation of infinitesimally close trajectories.

    lambda > 0: Chaotic (sensitive dependence on initial conditions)
    lambda = 0: Edge of chaos (critical point)
    lambda < 0: Stable attractor (converging trajectories)

REAL implementation using nolds package with:
- Rosenstein et al. (1993) algorithm for lambda estimation
- Grassberger-Procaccia (1983) for correlation dimension
- Proper Theiler window for temporal decorrelation

References:
    Wolf, Swift, Swinney & Vastano (1985)
    "Determining Lyapunov exponents from a time series"
    Physica D 16, 285-317

    Rosenstein, Collins & De Luca (1993)
    "A practical method for calculating largest Lyapunov exponents"
    Physica D 65, 117-134

    Kantz (1994)
    "A robust method to estimate the maximal Lyapunov exponent"
    Physics Letters A 185, 77-87
"""

import os
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.signal import correlate
from typing import Dict, Any, Optional

VERBOSE = os.getenv('PRISM_VERBOSE', '0') == '1'

# Check for nolds package
try:
    import nolds
    HAS_NOLDS = True
except ImportError:
    HAS_NOLDS = False


def compute(
    series: np.ndarray,
    mode: str = 'static',
    t: Optional[int] = None,
    window_size: int = 200,
    step_size: int = 20,
    embedding_dim: int = None,
    delay: int = None,
    min_tsep: int = None,
    trajectory_len: int = None,
) -> Dict[str, Any]:
    """
    Estimate largest Lyapunov exponent.

    Uses nolds package for REAL implementation.
    Falls back to simplified method only if nolds unavailable.

    Args:
        series: 1D numpy array of observations
        mode: 'static', 'windowed', or 'point'
        t: Time index for point mode
        window_size: Window size for windowed/point modes
        step_size: Step between windows for windowed mode
        embedding_dim: Phase space embedding dimension (auto if None)
        delay: Time delay for embedding (auto if None)
        min_tsep: Theiler window - minimum temporal separation
        trajectory_len: Length over which to track divergence

    Returns:
        mode='static': {'lyapunov_exponent': float, 'is_chaotic': bool, ...}
        mode='windowed': {'lyapunov_exponent': array, 't': array, ...}
        mode='point': {'lyapunov_exponent': float, 't': int, ...}
    """
    series = np.asarray(series).flatten()

    if mode == 'static':
        return _compute_static(series, embedding_dim, delay, min_tsep, trajectory_len)
    elif mode == 'windowed':
        return _compute_windowed(series, window_size, step_size, embedding_dim, delay, min_tsep, trajectory_len)
    elif mode == 'point':
        return _compute_point(series, t, window_size, embedding_dim, delay, min_tsep, trajectory_len)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'static', 'windowed', or 'point'.")


def _estimate_delay(x: np.ndarray) -> int:
    """
    Estimate optimal time delay using first minimum of autocorrelation.

    The delay should be chosen such that coordinates are as
    independent as possible (first minimum of ACF or mutual information).
    """
    n = len(x)
    if n < 10:
        return 1

    # Compute autocorrelation
    x_centered = x - np.mean(x)
    autocorr = correlate(x_centered, x_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]

    # Find first local minimum or zero crossing
    for i in range(1, min(len(autocorr) - 1, n // 4)):
        if autocorr[i] < autocorr[i-1] and autocorr[i] < autocorr[i+1]:
            return max(1, i)
        if autocorr[i] <= 0:
            return max(1, i)

    # Default: 1/4 of decorrelation time
    return max(1, n // 20)


def _compute_lyapunov_nolds(
    x: np.ndarray,
    emb_dim: int,
    lag: int,
    min_tsep: int,
    trajectory_len: int,
) -> Dict[str, Any]:
    """
    Compute Lyapunov exponent using nolds package.

    This is the REAL implementation.
    """
    if not HAS_NOLDS:
        raise ImportError(
            "nolds package required for real Lyapunov estimation.\n"
            "Install with: pip install nolds"
        )

    try:
        # Rosenstein method (more robust for short/noisy data)
        lyap_r = nolds.lyap_r(
            x,
            emb_dim=emb_dim,
            lag=lag,
            min_tsep=min_tsep,
            trajectory_len=trajectory_len,
        )

        # Also try Eckmann method for comparison
        try:
            lyap_e = nolds.lyap_e(
                x,
                emb_dim=emb_dim,
                matrix_dim=min(4, emb_dim),
            )
            lyap_e_val = lyap_e[0] if len(lyap_e) > 0 else None
        except Exception:
            lyap_e_val = None

        return {
            'lyapunov_exponent': float(lyap_r),
            'lyapunov_rosenstein': float(lyap_r),
            'lyapunov_eckmann': float(lyap_e_val) if lyap_e_val is not None else None,
            'is_chaotic': lyap_r > 0.01,
            'is_stable': lyap_r < -0.01,
            'is_critical': abs(lyap_r) <= 0.01,
            'embedding_dimension': emb_dim,
            'embedding_lag': lag,
            'theiler_window': min_tsep,
            'trajectory_length': trajectory_len,
            'n_points': len(x),
            'method': 'nolds_rosenstein',
        }

    except Exception as e:
        return {
            'lyapunov_exponent': None,
            'is_chaotic': None,
            'is_stable': None,
            'is_critical': None,
            'error': str(e),
            'method': 'nolds_failed',
        }


def _compute_lyapunov_simple(
    x: np.ndarray,
    emb_dim: int,
    lag: int,
    max_vectors: int = 300,
) -> Dict[str, Any]:
    """
    Fallback: Simplified nearest-neighbor divergence tracking.

    CLEARLY MARKED AS APPROXIMATION - not the real thing.
    """
    n = len(x)
    n_vectors = n - (emb_dim - 1) * lag

    if n_vectors < 50:
        return {
            'lyapunov_exponent': None,
            'is_chaotic': None,
            'method': 'INSUFFICIENT_DATA',
        }

    # Create embedded vectors
    embedded = np.zeros((n_vectors, emb_dim))
    for i in range(n_vectors):
        for j in range(emb_dim):
            embedded[i, j] = x[i + j * lag]

    # Subsample for efficiency
    if n_vectors > max_vectors:
        indices = np.linspace(0, n_vectors - 1, max_vectors, dtype=int)
        embedded = embedded[indices]
        n_vectors = max_vectors

    distances = cdist(embedded, embedded, 'euclidean')

    # Track divergence from nearest neighbors
    divergences = []

    for i in range(n_vectors - 10):
        dist_row = distances[i].copy()
        # Exclude self and temporal neighbors (Theiler window)
        theiler = max(3, lag)
        dist_row[max(0, i - theiler):min(n_vectors, i + theiler + 1)] = np.inf

        nearest_idx = np.argmin(dist_row)
        initial_dist = dist_row[nearest_idx]

        if initial_dist < 1e-10:
            continue

        # Track divergence over time
        for k in range(1, min(10, n_vectors - max(i, nearest_idx) - 1)):
            if i + k < n_vectors and nearest_idx + k < n_vectors:
                later_dist = np.linalg.norm(embedded[i + k] - embedded[nearest_idx + k])
                if later_dist > 1e-10 and initial_dist > 1e-10:
                    divergences.append((k, np.log(later_dist / initial_dist)))

    if len(divergences) < 10:
        return {
            'lyapunov_exponent': None,
            'is_chaotic': None,
            'method': 'INSUFFICIENT_DIVERGENCES',
        }

    # Linear regression on divergence vs time
    times = np.array([d[0] for d in divergences])
    log_divs = np.array([d[1] for d in divergences])

    slope, intercept, r_value, p_value, std_err = stats.linregress(times, log_divs)

    result = {
        'lyapunov_exponent': float(slope),
        'is_chaotic': slope > 0.01,
        'is_stable': slope < -0.01,
        'is_critical': abs(slope) <= 0.01,
        # Confidence metrics from regression
        'lyapunov_r2': float(r_value ** 2),
        'lyapunov_p_value': float(p_value),
        'lyapunov_std_err': float(std_err),
        'lyapunov_intercept': float(intercept),
        # Backwards compatibility
        'confidence': float(r_value ** 2),
        'embedding_dimension': emb_dim,
        'embedding_lag': lag,
        'n_divergences': len(divergences),
        'method': 'SIMPLIFIED_APPROXIMATION',  # CLEARLY MARKED
    }

    if VERBOSE:
        result['divergence_times'] = times.tolist()
        result['divergence_values'] = log_divs.tolist()

    return result


def _compute_static(
    series: np.ndarray,
    embedding_dim: int = None,
    delay: int = None,
    min_tsep: int = None,
    trajectory_len: int = None,
) -> Dict[str, Any]:
    """Estimate Lyapunov exponent on entire signal."""
    # Remove NaN
    x = series[~np.isnan(series)]

    if len(x) < 100:
        return {
            'lyapunov_exponent': None,
            'is_chaotic': None,
            'error': 'Time series too short (need >= 100 points)',
            'method': 'INSUFFICIENT_DATA',
        }

    # Estimate parameters if not provided
    if delay is None:
        delay = _estimate_delay(x)

    if embedding_dim is None:
        # Conservative default - can be estimated via FNN
        embedding_dim = min(10, max(3, len(x) // 50))

    if min_tsep is None:
        min_tsep = delay  # Theiler window = embedding lag

    if trajectory_len is None:
        trajectory_len = min(500, len(x) // 10)

    # Try nolds first
    if HAS_NOLDS:
        try:
            return _compute_lyapunov_nolds(x, embedding_dim, delay, min_tsep, trajectory_len)
        except Exception:
            pass

    # Fallback to simplified method
    return _compute_lyapunov_simple(x, embedding_dim, delay)


def _compute_windowed(
    series: np.ndarray,
    window_size: int,
    step_size: int,
    embedding_dim: int = None,
    delay: int = None,
    min_tsep: int = None,
    trajectory_len: int = None,
) -> Dict[str, Any]:
    """Estimate Lyapunov exponent over rolling windows."""
    n = len(series)

    if n < window_size:
        return {
            'lyapunov_exponent': np.array([]),
            'is_chaotic': np.array([]),
            't': np.array([]),
            'window_size': window_size,
            'step_size': step_size,
            'method': 'nolds' if HAS_NOLDS else 'simplified',
        }

    t_values = []
    lyap_values = []
    chaotic_values = []

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        window = series[start:end]

        result = _compute_static(window, embedding_dim, delay, min_tsep, trajectory_len)

        t_values.append(start + window_size // 2)
        lyap_values.append(result.get('lyapunov_exponent', np.nan) or np.nan)
        chaotic_values.append(result.get('is_chaotic', False) or False)

    return {
        'lyapunov_exponent': np.array(lyap_values),
        'is_chaotic': np.array(chaotic_values),
        't': np.array(t_values),
        'window_size': window_size,
        'step_size': step_size,
        'method': 'nolds' if HAS_NOLDS else 'simplified',
    }


def _compute_point(
    series: np.ndarray,
    t: int,
    window_size: int,
    embedding_dim: int = None,
    delay: int = None,
    min_tsep: int = None,
    trajectory_len: int = None,
) -> Dict[str, Any]:
    """Estimate Lyapunov exponent at specific time t."""
    if t is None:
        raise ValueError("t is required for point mode")

    n = len(series)

    # Center window on t
    half_window = window_size // 2
    start = max(0, t - half_window)
    end = min(n, start + window_size)

    if end - start < window_size:
        start = max(0, end - window_size)

    window = series[start:end]

    result = _compute_static(window, embedding_dim, delay, min_tsep, trajectory_len)
    result['t'] = t
    result['window_start'] = start
    result['window_end'] = end

    return result


def compute_correlation_dimension(
    series: np.ndarray,
    embedding_dim: int = 10,
) -> Dict[str, Any]:
    """
    Compute correlation dimension (Grassberger-Procaccia algorithm).

    D2 = lim_{r->0} [log C(r) / log r]

    Physical meaning:
        - Fractal dimension of the attractor
        - Integer D2 = regular attractor (limit cycle, torus)
        - Non-integer D2 = strange attractor (chaos)

    Reference:
        Grassberger & Procaccia (1983)
        "Characterization of Strange Attractors"
        Physical Review Letters 50, 346
    """
    if not HAS_NOLDS:
        return {
            'correlation_dimension': None,
            'error': 'nolds package required',
        }

    x = np.asarray(series).flatten()
    x = x[~np.isnan(x)]

    if len(x) < 100:
        return {
            'correlation_dimension': None,
            'error': 'Time series too short',
        }

    try:
        d2 = nolds.corr_dim(x, embedding_dim)

        return {
            'correlation_dimension': float(d2),
            'is_strange_attractor': (d2 % 1) > 0.1,  # Non-integer
            'is_regular_attractor': (d2 % 1) <= 0.1,  # Integer-ish
            'embedding_dimension': embedding_dim,
            'n_points': len(x),
        }
    except Exception as e:
        return {
            'correlation_dimension': None,
            'error': str(e),
        }


def compute_sample_entropy(
    series: np.ndarray,
    m: int = 2,
    r: float = None,
) -> Dict[str, Any]:
    """
    Compute sample entropy using nolds.

    Sample entropy measures regularity/predictability of a time series.
    Higher values = more irregular/complex.

    Reference:
        Richman & Moorman (2000)
        "Physiological time-series analysis using approximate entropy and sample entropy"
    """
    if not HAS_NOLDS:
        return {
            'sample_entropy': None,
            'error': 'nolds package required',
        }

    x = np.asarray(series).flatten()
    x = x[~np.isnan(x)]

    if len(x) < 50:
        return {
            'sample_entropy': None,
            'error': 'Time series too short',
        }

    if r is None:
        r = 0.2 * np.std(x)

    try:
        se = nolds.sampen(x, emb_dim=m, tolerance=r)

        return {
            'sample_entropy': float(se),
            'embedding_dimension': m,
            'tolerance': float(r),
            'n_points': len(x),
        }
    except Exception as e:
        return {
            'sample_entropy': None,
            'error': str(e),
        }
