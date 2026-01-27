"""
Dynamic Mode Decomposition (DMD) Engine

Decomposes multivariate time series into dynamic modes.

CANONICAL INTERFACE:
    def compute(observations: pd.DataFrame) -> pd.DataFrame
    Input:  [entity_id, signal_id, I, y]
    Output: [entity_id, mode_idx, eigenvalue_real, eigenvalue_imag, growth_rate,
             frequency, amplitude, energy, is_stable]

DMD extracts coherent spatiotemporal patterns (modes) and their
growth/decay rates and oscillation frequencies.
"""

ENGINE_TYPE = 'signal'  # Entity-level: outputs per-entity DMD modes

import numpy as np
import pandas as pd
from scipy import linalg
from typing import Dict, Any, Tuple, List


def compute(
    observations: pd.DataFrame,
    rank: int = None,
    delay_embedding: int = 1,
) -> pd.DataFrame:
    """
    Compute Dynamic Mode Decomposition for all entities.

    CANONICAL INTERFACE:
        Input:  observations [entity_id, signal_id, I, y]
        Output: primitives [entity_id, mode_idx, eigenvalue_real, eigenvalue_imag,
                           growth_rate, frequency, amplitude, energy, is_stable]

    Parameters
    ----------
    observations : pd.DataFrame
        Must have columns: entity_id, signal_id, I, y
    rank : int, optional
        Truncation rank (None = auto via SVD threshold)
    delay_embedding : int, optional
        Time-delay embedding (1 = no embedding)

    Returns
    -------
    pd.DataFrame
        DMD modes per entity
    """
    results = []

    for entity_id, entity_group in observations.groupby('entity_id'):
        # Pivot to wide format: rows=I (time), cols=signal_id, values=y
        try:
            wide = entity_group.pivot(index='I', columns='signal_id', values='y')
            wide = wide.sort_index().dropna()
        except Exception:
            wide = entity_group.groupby(['I', 'signal_id'])['y'].mean().unstack()
            wide = wide.sort_index().dropna()

        if len(wide) < 10:
            continue

        try:
            # Prepare data matrix (features x time)
            X = wide.values.T  # Shape: (n_features, n_samples)

            # Apply time-delay embedding if requested
            if delay_embedding > 1:
                X = _delay_embed(X, delay_embedding)

            # Run DMD
            eigenvalues, modes, amplitudes, reconstruction_error = _compute_dmd(X, rank)

            # Extract mode characteristics
            mode_info = _analyze_modes(eigenvalues, amplitudes)

            for mode in mode_info:
                results.append({
                    'entity_id': entity_id,
                    'mode_idx': mode['mode_idx'],
                    'eigenvalue_real': mode['eigenvalue_real'],
                    'eigenvalue_imag': mode['eigenvalue_imag'],
                    'magnitude': mode['magnitude'],
                    'growth_rate': mode['growth_rate'],
                    'frequency': mode['frequency'],
                    'amplitude': mode['amplitude'],
                    'energy': mode['energy'],
                    'is_stable': mode['is_stable'],
                    'reconstruction_error': reconstruction_error,
                })

        except Exception:
            # Add NaN result for failed computation
            results.append({
                'entity_id': entity_id,
                'mode_idx': 0,
                'eigenvalue_real': np.nan,
                'eigenvalue_imag': np.nan,
                'magnitude': np.nan,
                'growth_rate': np.nan,
                'frequency': np.nan,
                'amplitude': np.nan,
                'energy': np.nan,
                'is_stable': False,
                'reconstruction_error': np.nan,
            })

    return pd.DataFrame(results)


def _delay_embed(X: np.ndarray, delays: int) -> np.ndarray:
    """
    Time-delay embedding to augment state space.

    Converts X(t) to [X(t), X(t-1), ..., X(t-delays+1)]
    """
    n_features, n_samples = X.shape
    n_embedded = n_samples - delays + 1

    embedded = np.zeros((n_features * delays, n_embedded))

    for d in range(delays):
        start = delays - 1 - d
        end = start + n_embedded
        embedded[d * n_features:(d + 1) * n_features, :] = X[:, start:end]

    return embedded


def _compute_dmd(
    X: np.ndarray,
    rank: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Exact DMD algorithm.

    Given snapshots X = [x_1, ..., x_m], find A such that x_{k+1} ≈ A x_k

    Returns (eigenvalues, modes, amplitudes, reconstruction_error)
    """
    # Split into X (past) and Y (future)
    X_past = X[:, :-1]
    X_future = X[:, 1:]

    # SVD of X_past
    U, s, Vh = linalg.svd(X_past, full_matrices=False)

    # Determine rank
    if rank is None:
        # Auto-select based on energy capture (99%)
        cumulative_energy = np.cumsum(s ** 2) / np.sum(s ** 2)
        rank = np.searchsorted(cumulative_energy, 0.99) + 1
        rank = max(1, min(rank, len(s)))

    rank = min(rank, len(s))

    # Truncate
    U_r = U[:, :rank]
    s_r = s[:rank]
    Vh_r = Vh[:rank, :]

    # Build reduced A matrix: A_tilde = U_r^T @ Y @ V_r @ S_r^{-1}
    A_tilde = U_r.T @ X_future @ Vh_r.T @ np.diag(1 / s_r)

    # Eigendecomposition
    eigenvalues, W = linalg.eig(A_tilde)

    # DMD modes (in original space)
    modes = X_future @ Vh_r.T @ np.diag(1 / s_r) @ W

    # Mode amplitudes (initial condition projection)
    x0 = X[:, 0]
    amplitudes = np.abs(np.linalg.lstsq(modes, x0, rcond=None)[0])

    # Reconstruction error
    X_reconstructed = _reconstruct(modes, eigenvalues, amplitudes, X.shape[1])
    error = np.linalg.norm(X - X_reconstructed, 'fro') / np.linalg.norm(X, 'fro')

    return eigenvalues, modes, amplitudes, error


def _reconstruct(
    modes: np.ndarray,
    eigenvalues: np.ndarray,
    amplitudes: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Reconstruct signal from DMD modes."""
    n_modes = len(eigenvalues)
    n_features = modes.shape[0]

    X_recon = np.zeros((n_features, n_steps), dtype=complex)

    for k in range(n_steps):
        for j in range(n_modes):
            X_recon[:, k] += amplitudes[j] * (eigenvalues[j] ** k) * modes[:, j]

    return np.real(X_recon)


def _analyze_modes(
    eigenvalues: np.ndarray,
    amplitudes: np.ndarray,
) -> List[Dict[str, Any]]:
    """Extract characteristics of each mode."""
    modes = []

    for i, (ev, amp) in enumerate(zip(eigenvalues, amplitudes)):
        # Growth rate (log of magnitude)
        growth_rate = np.log(np.abs(ev)) if np.abs(ev) > 0 else -np.inf

        # Frequency (angle)
        frequency = np.abs(np.angle(ev)) / (2 * np.pi)

        # Stability
        is_stable = np.abs(ev) <= 1.0

        modes.append({
            'mode_idx': i,
            'eigenvalue_real': float(np.real(ev)),
            'eigenvalue_imag': float(np.imag(ev)),
            'magnitude': float(np.abs(ev)),
            'growth_rate': float(growth_rate) if not np.isinf(growth_rate) else np.nan,
            'frequency': float(frequency),
            'amplitude': float(amp),
            'energy': float(amp ** 2),
            'is_stable': is_stable,
        })

    return modes
