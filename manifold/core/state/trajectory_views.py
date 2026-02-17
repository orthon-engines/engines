"""
Trajectory Views Engine.

Computes per-window views of feature trajectories:
- Fourier view: spectral analysis of feature trajectories
- Hilbert view: envelope (amplitude modulation) of feature trajectories
- Laplacian view: graph coupling structure across signals

Pure computation — numpy/dict in, dict out. No file I/O.
"""

import warnings

import numpy as np
from typing import List, Dict

from manifold.primitives.individual.spectral import spectral_profile
from manifold.primitives.individual.hilbert import envelope
from manifold.primitives.matrix.graph import laplacian_matrix, laplacian_eigenvalues


def compute_fourier_view(
    trajectories: Dict[str, Dict[str, np.ndarray]],
    feature_cols: List[str],
    min_length: int = 8,
) -> Dict[str, float]:
    """
    Fourier view: run spectral_profile on each signal's feature trajectory.
    Aggregate across signals via nanmedian.

    Output keys: fourier_{feature}_dominant_freq, fourier_{feature}_spectral_flatness
    """
    result = {}

    for feat in feature_cols:
        dom_freqs = []
        flatnesses = []

        for sig_trajs in trajectories.values():
            if feat not in sig_trajs:
                continue
            arr = sig_trajs[feat]
            if len(arr) < min_length:
                continue

            sp = spectral_profile(arr)
            dom_freqs.append(sp.get('dominant_frequency', np.nan))
            flatnesses.append(sp.get('spectral_flatness', np.nan))

        result[f'fourier_{feat}_dominant_freq'] = float(np.nanmedian(dom_freqs)) if dom_freqs else np.nan
        result[f'fourier_{feat}_spectral_flatness'] = float(np.nanmedian(flatnesses)) if flatnesses else np.nan

    return result


def compute_hilbert_view(
    trajectories: Dict[str, Dict[str, np.ndarray]],
    feature_cols: List[str],
    min_length: int = 4,
) -> Dict[str, float]:
    """
    Hilbert view: run envelope on each signal's feature trajectory.
    Compute mean, trend, cv. Aggregate across signals via nanmedian.

    Output keys: envelope_{feature}_mean, envelope_{feature}_trend, envelope_{feature}_cv
    """
    result = {}

    for feat in feature_cols:
        means = []
        trends = []
        cvs = []

        for sig_trajs in trajectories.values():
            if feat not in sig_trajs:
                continue
            arr = sig_trajs[feat]
            if len(arr) < min_length:
                continue

            env = envelope(arr)
            m = float(np.mean(env))
            means.append(m)

            if len(env) >= 2:
                trend = np.polyfit(np.arange(len(env)), env, 1)[0]
                trends.append(float(trend))
            else:
                trends.append(np.nan)

            if m > 0:
                cvs.append(float(np.std(env) / m))
            else:
                cvs.append(np.nan)

        result[f'envelope_{feat}_mean'] = float(np.nanmedian(means)) if means else np.nan
        result[f'envelope_{feat}_trend'] = float(np.nanmedian(trends)) if trends else np.nan
        result[f'envelope_{feat}_cv'] = float(np.nanmedian(cvs)) if cvs else np.nan

    return result


def compute_laplacian_view(
    trajectories: Dict[str, Dict[str, np.ndarray]],
    feature_cols: List[str],
    min_signals: int = 2,
) -> Dict[str, float]:
    """
    Laplacian view: build correlation-based adjacency from concatenated
    feature trajectories, compute graph Laplacian spectrum.

    Output keys: laplacian_algebraic_connectivity, laplacian_spectral_gap,
                 laplacian_n_components, laplacian_max_eigenvalue
    """
    result = {
        'laplacian_algebraic_connectivity': np.nan,
        'laplacian_spectral_gap': np.nan,
        'laplacian_n_components': np.nan,
        'laplacian_max_eigenvalue': np.nan,
    }

    if len(trajectories) < min_signals:
        return result

    # For each signal, concatenate all feature trajectories into one vector
    vectors = []
    for sig_id in sorted(trajectories.keys()):
        parts = []
        for feat in feature_cols:
            if feat in trajectories[sig_id]:
                arr = trajectories[sig_id][feat]
                arr = np.where(np.isfinite(arr), arr, 0.0)
                parts.append(arr)
        if parts:
            vectors.append(np.concatenate(parts))

    if len(vectors) < min_signals:
        return result

    # Pad to same length (signals may have different trajectory lengths)
    max_len = max(len(v) for v in vectors)
    padded = np.zeros((len(vectors), max_len))
    for i, v in enumerate(vectors):
        padded[i, :len(v)] = v

    try:
        # Build adjacency: |corrcoef|
        corr = np.corrcoef(padded)
        if corr.ndim < 2:
            return result
        adj = np.abs(corr)
        np.fill_diagonal(adj, 0.0)
        adj = np.where(np.isfinite(adj), adj, 0.0)

        L = laplacian_matrix(adj, normalized=True)
        eigs = laplacian_eigenvalues(L)

        if len(eigs) >= 2 and np.any(np.isfinite(eigs)):
            result['laplacian_algebraic_connectivity'] = float(eigs[1])
            if eigs[-1] > 0:
                result['laplacian_spectral_gap'] = float(eigs[1] / eigs[-1])
            result['laplacian_n_components'] = float(np.sum(eigs < 1e-10))
            result['laplacian_max_eigenvalue'] = float(eigs[-1])
    except (ValueError, np.linalg.LinAlgError):
        pass
    except Exception as e:
        warnings.warn(f"trajectory_views.compute_laplacian_view: {type(e).__name__}: {e}", RuntimeWarning, stacklevel=2)

    return result
