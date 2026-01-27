"""
Attractor Reconstruction Engine

Discovers hidden dynamical structure using Takens embedding.

CANONICAL INTERFACE:
    def compute(observations: pd.DataFrame) -> pd.DataFrame
    Input:  [entity_id, signal_id, I, y]
    Output: [entity_id, signal_id, embedding_dim, delay, correlation_dim,
             lyapunov_exponent, attractor_type]

Computes:
- Optimal embedding dimension (false nearest neighbors)
- Optimal delay (mutual information)
- Correlation dimension (attractor complexity)
- Largest Lyapunov exponent (stability/chaos)
- Attractor classification
"""

ENGINE_TYPE = 'signal'

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import linregress
from typing import Dict, Any

try:
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def compute(
    observations: pd.DataFrame,
    max_dim: int = 10,
    max_delay: int = 100,
) -> pd.DataFrame:
    """
    Compute attractor reconstruction for all signals.

    CANONICAL INTERFACE:
        Input:  observations [entity_id, signal_id, I, y]
        Output: primitives [entity_id, signal_id, embedding_dim, delay,
                           correlation_dim, lyapunov_exponent, attractor_type]

    Parameters
    ----------
    observations : pd.DataFrame
        Must have columns: entity_id, signal_id, I, y
    max_dim : int, optional
        Maximum embedding dimension to try (default: 10)
    max_delay : int, optional
        Maximum delay to try (default: 100)

    Returns
    -------
    pd.DataFrame
        Attractor metrics per signal
    """
    if not HAS_SKLEARN:
        # Return empty with NaN if sklearn not available
        return pd.DataFrame(columns=[
            'entity_id', 'signal_id', 'embedding_dim', 'delay',
            'correlation_dim', 'lyapunov_exponent', 'attractor_type'
        ])

    results = []

    for (entity_id, signal_id), group in observations.groupby(['entity_id', 'signal_id']):
        y = group.sort_values('I')['y'].values

        if len(y) < 50:
            results.append({
                'entity_id': entity_id,
                'signal_id': signal_id,
                'embedding_dim': np.nan,
                'delay': np.nan,
                'correlation_dim': np.nan,
                'lyapunov_exponent': np.nan,
                'attractor_type': 'insufficient_data',
            })
            continue

        try:
            result = _reconstruct_attractor(y, max_dim, max_delay)
            results.append({
                'entity_id': entity_id,
                'signal_id': signal_id,
                **result
            })
        except Exception:
            results.append({
                'entity_id': entity_id,
                'signal_id': signal_id,
                'embedding_dim': np.nan,
                'delay': np.nan,
                'correlation_dim': np.nan,
                'lyapunov_exponent': np.nan,
                'attractor_type': 'error',
            })

    return pd.DataFrame(results)


def _reconstruct_attractor(
    signal: np.ndarray,
    max_dim: int,
    max_delay: int,
) -> Dict[str, Any]:
    """Reconstruct attractor from a single signal."""
    # Step 1: Find optimal delay via mutual information
    delay = _optimal_delay(signal, max_delay)

    # Step 2: Find optimal embedding dimension via false nearest neighbors
    embedding_dim = _optimal_embedding_dim(signal, delay, max_dim)

    # Step 3: Create embedded trajectory
    embedded = _embed(signal, embedding_dim, delay)

    if len(embedded) < 20:
        return {
            'embedding_dim': embedding_dim,
            'delay': delay,
            'correlation_dim': np.nan,
            'lyapunov_exponent': np.nan,
            'attractor_type': 'insufficient_embedded',
        }

    # Step 4: Compute correlation dimension
    correlation_dim = _correlation_dimension(embedded)

    # Step 5: Compute largest Lyapunov exponent
    lyapunov_exp = _lyapunov_exponent(embedded, delay, embedding_dim)

    # Step 6: Classify attractor type
    attractor_type = _classify_attractor(correlation_dim, lyapunov_exp)

    return {
        'embedding_dim': int(embedding_dim),
        'delay': int(delay),
        'correlation_dim': float(correlation_dim) if not np.isnan(correlation_dim) else np.nan,
        'lyapunov_exponent': float(lyapunov_exp) if not np.isnan(lyapunov_exp) else np.nan,
        'attractor_type': attractor_type,
    }


def _optimal_delay(signal: np.ndarray, max_delay: int) -> int:
    """Find optimal delay using first minimum of mutual information."""
    n = len(signal)
    mi_values = []

    for delay in range(1, min(max_delay, n // 4)):
        x = signal[:-delay]
        y = signal[delay:]

        bins = max(10, int(np.sqrt(len(x) / 5)))
        hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
        hist_sum = hist_2d.sum()
        if hist_sum == 0:
            continue
        hist_2d = hist_2d / hist_sum

        px = hist_2d.sum(axis=1)
        py = hist_2d.sum(axis=0)

        mi = 0
        for i in range(bins):
            for j in range(bins):
                if hist_2d[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += hist_2d[i, j] * np.log(hist_2d[i, j] / (px[i] * py[j]))

        mi_values.append(mi)

        if len(mi_values) >= 3:
            if mi_values[-2] < mi_values[-3] and mi_values[-2] < mi_values[-1]:
                return delay - 1

    if len(mi_values) == 0:
        return 1

    mi_values = np.array(mi_values)
    threshold = mi_values[0] / np.e
    below_threshold = np.where(mi_values < threshold)[0]
    if len(below_threshold) > 0:
        return below_threshold[0] + 1

    return max(1, len(mi_values) // 4)


def _optimal_embedding_dim(signal: np.ndarray, delay: int, max_dim: int) -> int:
    """Find optimal embedding dimension using false nearest neighbors."""
    threshold = 15.0

    for dim in range(1, max_dim + 1):
        embedded = _embed(signal, dim, delay)
        if len(embedded) < 10:
            return dim

        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(embedded)
        distances, indices = nbrs.kneighbors(embedded)

        embedded_plus1 = _embed(signal, dim + 1, delay)
        if len(embedded_plus1) < len(embedded):
            embedded = embedded[:len(embedded_plus1)]
            distances = distances[:len(embedded_plus1)]
            indices = indices[:len(embedded_plus1)]

        if len(embedded_plus1) == 0:
            return dim

        false_neighbors = 0
        total_checked = 0

        for i in range(len(embedded)):
            nn_idx = indices[i, 1]
            if nn_idx >= len(embedded_plus1):
                continue

            d_current = distances[i, 1]
            if d_current < 1e-10:
                continue

            d_next = np.linalg.norm(embedded_plus1[i] - embedded_plus1[nn_idx])

            ratio = abs(d_next - d_current) / d_current
            if ratio > threshold:
                false_neighbors += 1
            total_checked += 1

        if total_checked > 0:
            fnn_ratio = false_neighbors / total_checked
            if fnn_ratio < 0.01:
                return dim

    return max_dim


def _embed(signal: np.ndarray, dim: int, delay: int) -> np.ndarray:
    """Create Takens embedding."""
    n = len(signal)
    n_points = n - (dim - 1) * delay

    if n_points <= 0:
        return np.array([]).reshape(0, dim)

    embedded = np.zeros((n_points, dim))
    for i in range(dim):
        embedded[:, i] = signal[i * delay : i * delay + n_points]

    return embedded


def _correlation_dimension(embedded: np.ndarray, n_scales: int = 20) -> float:
    """Estimate correlation dimension using Grassberger-Procaccia algorithm."""
    if len(embedded) < 50:
        return np.nan

    distances = pdist(embedded)
    distances = distances[distances > 0]

    if len(distances) < 100:
        return np.nan

    r_min = np.percentile(distances, 1)
    r_max = np.percentile(distances, 50)

    if r_min <= 0 or r_max <= r_min:
        return np.nan

    radii = np.logspace(np.log10(r_min), np.log10(r_max), n_scales)

    C_r = []
    for r in radii:
        count = np.sum(distances < r)
        C_r.append(count / len(distances))

    C_r = np.array(C_r)

    valid = (C_r > 0.01) & (C_r < 0.5)
    if np.sum(valid) < 5:
        return np.nan

    log_r = np.log(radii[valid])
    log_C = np.log(C_r[valid])

    slope, _, r_value, _, _ = linregress(log_r, log_C)

    if r_value**2 < 0.9:
        return np.nan

    return slope


def _lyapunov_exponent(embedded: np.ndarray, delay: int, dim: int, dt: float = 1.0) -> float:
    """Estimate largest Lyapunov exponent using Rosenstein's method."""
    if len(embedded) < 100:
        return np.nan

    n = len(embedded)
    min_temporal_separation = delay * dim * 2

    nbrs = NearestNeighbors(n_neighbors=min(n // 10, 50), algorithm='ball_tree').fit(embedded)
    distances, indices = nbrs.kneighbors(embedded)

    nn_indices = np.zeros(n, dtype=int)
    nn_distances = np.zeros(n)

    for i in range(n):
        for j in range(1, len(indices[i])):
            nn_idx = indices[i, j]
            if abs(nn_idx - i) > min_temporal_separation:
                nn_indices[i] = nn_idx
                nn_distances[i] = distances[i, j]
                break

    max_iter = min(n // 4, 200)
    divergence = np.zeros(max_iter)
    counts = np.zeros(max_iter)

    for i in range(n - max_iter):
        j = nn_indices[i]
        if j == 0 or j + max_iter >= n:
            continue

        d0 = nn_distances[i]
        if d0 < 1e-10:
            continue

        for k in range(max_iter):
            if i + k >= n or j + k >= n:
                break
            dk = np.linalg.norm(embedded[i + k] - embedded[j + k])
            if dk > 0:
                divergence[k] += np.log(dk / d0)
                counts[k] += 1

    valid = counts > 10
    if np.sum(valid) < 10:
        return np.nan

    avg_divergence = divergence[valid] / counts[valid]
    time = np.arange(len(avg_divergence)) * dt

    n_fit = min(len(time), 50)
    slope, _, r_value, _, _ = linregress(time[:n_fit], avg_divergence[:n_fit])

    if r_value**2 < 0.8:
        return np.nan

    return slope


def _classify_attractor(correlation_dim: float, lyapunov_exp: float) -> str:
    """Classify attractor type based on computed metrics."""
    if np.isnan(correlation_dim):
        return "unknown"

    lya = lyapunov_exp if not np.isnan(lyapunov_exp) else 0

    if correlation_dim < 0.5:
        return "fixed_point"
    elif correlation_dim < 1.5:
        if lya < 0:
            return "limit_cycle_stable"
        else:
            return "limit_cycle_unstable"
    elif correlation_dim < 2.5:
        if lya < 0:
            return "torus"
        else:
            return "quasi_periodic"
    else:
        if lya > 0.01:
            return "strange_attractor"
        else:
            return "high_dimensional"
