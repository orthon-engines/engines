"""
DMD (Dynamic Mode Decomposition) Engine.

Computes dynamic mode decomposition for linear dynamics analysis.

DMD extracts spatiotemporal coherent structures from time series:
- Eigenvalues inside unit circle = stable modes (decay)
- Eigenvalues outside unit circle = unstable modes (growth)
- Eigenvalue phase = oscillation frequency
"""

import numpy as np
from typing import Dict


def compute(y: np.ndarray, rank: int = None, dt: float = 1.0) -> Dict[str, float]:
    """
    Compute DMD of signal.

    Args:
        y: Signal values
        rank: Maximum rank for truncation (default: auto)
        dt: Time step (default: 1.0)

    Returns:
        dict with:
            - 'dmd_dominant_freq': Frequency of dominant mode
            - 'dmd_growth_rate': Growth rate of dominant mode
            - 'dmd_is_stable': True if all modes stable
            - 'dmd_n_modes': Number of significant modes
            - 'dmd_eigenvalues': List of eigenvalue magnitudes
    """
    result = {
        'dmd_dominant_freq': np.nan,
        'dmd_growth_rate': np.nan,
        'dmd_is_stable': True,
        'dmd_n_modes': 0,
        'dmd_eigenvalues': []
    }

    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    if n < 50:
        return result

    # Build time-delay embedding matrix
    n_delays = min(10, n // 5)
    if n_delays < 3:
        return result

    n_snapshots = n - n_delays
    if n_snapshots < n_delays + 5:
        return result

    X = np.zeros((n_delays, n_snapshots))
    for i in range(n_delays):
        X[i, :] = y[i:n - n_delays + i]

    # Check for constant signal
    if np.std(X) < 1e-10:
        return result

    # Split into X1, X2 for DMD
    X1, X2 = X[:, :-1], X[:, 1:]

    try:
        # SVD of X1
        U, S, Vh = np.linalg.svd(X1, full_matrices=False)

        # Check conditioning
        if S[0] < 1e-10:
            return result

        # Determine rank (truncation)
        if rank is None:
            # Keep modes explaining 99% of energy
            energy = np.cumsum(S ** 2) / np.sum(S ** 2)
            r = min(np.searchsorted(energy, 0.99) + 1, len(S), 10)
        else:
            r = min(rank, len(S))

        r = max(1, r)

        # Truncate
        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]

        # Check for near-zero singular values
        if np.min(S_r) < 1e-10:
            # Regularize
            S_r = np.maximum(S_r, 1e-10)

        # Compute reduced DMD operator: A_tilde = U^T * X2 * V * S^{-1}
        A_tilde = U_r.T @ X2 @ Vh_r.T @ np.diag(1 / S_r)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eig(A_tilde)

        # Convert to continuous-time growth rates and frequencies
        # Discrete eigenvalue λ → continuous: λ_c = log(λ) / dt
        eigenvalue_mags = np.abs(eigenvalues)

        # Frequencies from phase
        freqs = np.angle(eigenvalues) / (2 * np.pi * dt)

        # Growth rates from magnitude
        growth_rates = np.log(eigenvalue_mags + 1e-10) / dt

        # Find dominant mode (highest amplitude)
        dominant_idx = np.argmax(eigenvalue_mags)

        # Stability: all eigenvalues inside unit circle
        is_stable = bool(np.all(eigenvalue_mags <= 1.01))

        result = {
            'dmd_dominant_freq': float(np.abs(freqs[dominant_idx])),
            'dmd_growth_rate': float(growth_rates[dominant_idx]),
            'dmd_is_stable': is_stable,
            'dmd_n_modes': int(r),
            'dmd_eigenvalues': [float(m) for m in sorted(eigenvalue_mags, reverse=True)[:5]]
        }

    except np.linalg.LinAlgError:
        pass
    except Exception:
        pass

    return result
