"""
Pairwise Causality Primitives (46, 48)

Granger causality and convergent cross-mapping.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional

try:
    from rudder_primitives_rs.pairwise import (
        granger_causality as _granger_rs,
    )
    _USE_RUST_CAUSALITY = True
except ImportError:
    _USE_RUST_CAUSALITY = False


def granger_causality(
    source: np.ndarray,
    target: np.ndarray,
    max_lag: int = 5
) -> Tuple[float, float, int]:
    """
    Test Granger causality from source to target.

    Parameters
    ----------
    source : np.ndarray
        Potential cause
    target : np.ndarray
        Potential effect
    max_lag : int
        Maximum lag to test

    Returns
    -------
    tuple
        (f_statistic, p_value, optimal_lag)

    Notes
    -----
    X Granger-causes Y if past values of X improve prediction of Y
    beyond Y's own past.
    p < 0.05: significant Granger causality
    """
    source = np.asarray(source).flatten()
    target = np.asarray(target).flatten()

    n = min(len(source), len(target))
    source, target = source[:n], target[:n]

    if n < max_lag + 10:
        return 0.0, 1.0, 1

    if _USE_RUST_CAUSALITY:
        return _granger_rs(source, target, max_lag)

    # Find optimal lag via AIC
    best_aic = np.inf
    optimal_lag = 1

    for lag in range(1, max_lag + 1):
        y = target[max_lag:]
        X_r = np.column_stack([
            np.ones(len(y)),
            *[target[max_lag - i : n - i] for i in range(1, lag + 1)]
        ])

        if len(y) < X_r.shape[1] + 2:
            continue

        try:
            beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
            resid = y - X_r @ beta_r
            ssr = np.sum(resid ** 2)
            aic = len(y) * np.log(ssr / len(y) + 1e-10) + 2 * (lag + 1)

            if aic < best_aic:
                best_aic = aic
                optimal_lag = lag
        except:
            continue

    lag = optimal_lag

    # Restricted model: Y ~ Y_past
    y = target[max_lag:]
    X_r = np.column_stack([
        np.ones(len(y)),
        *[target[max_lag - i : n - i] for i in range(1, lag + 1)]
    ])

    # Unrestricted model: Y ~ Y_past + X_past
    X_u = np.column_stack([
        X_r,
        *[source[max_lag - i : n - i] for i in range(1, lag + 1)]
    ])

    try:
        beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
        beta_u = np.linalg.lstsq(X_u, y, rcond=None)[0]

        ssr_r = np.sum((y - X_r @ beta_r) ** 2)
        ssr_u = np.sum((y - X_u @ beta_u) ** 2)

        df1 = lag
        df2 = len(y) - X_u.shape[1]

        if df2 <= 0 or ssr_u <= 0:
            return 0.0, 1.0, lag

        f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)

        return float(f_stat), float(p_value), lag

    except:
        return 0.0, 1.0, lag


def convergent_cross_mapping(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    embedding_dim: int = 3,
    tau: int = 1,
    library_size: int = None
) -> Tuple[float, float]:
    """
    Convergent Cross-Mapping for nonlinear causality.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Input signals
    embedding_dim : int
        Embedding dimension
    tau : int
        Time delay
    library_size : int, optional
        Library size (default: max available)

    Returns
    -------
    tuple
        (rho_a_causes_b, rho_b_causes_a)

    Notes
    -----
    If A causes B, then B's attractor contains information about A.
    Uses manifold reconstruction to test causality.
    Works for nonlinear, deterministic systems.
    """
    sig_a = np.asarray(sig_a).flatten()
    sig_b = np.asarray(sig_b).flatten()

    n = min(len(sig_a), len(sig_b))
    sig_a, sig_b = sig_a[:n], sig_b[:n]

    max_n = n - (embedding_dim - 1) * tau

    if max_n < embedding_dim + 10:
        return np.nan, np.nan

    if library_size is None:
        library_size = max_n

    library_size = min(library_size, max_n)

    # Embed both signals
    def embed(x, E, tau):
        n_emb = len(x) - (E - 1) * tau
        emb = np.zeros((n_emb, E))
        for i in range(E):
            emb[:, i] = x[i * tau : i * tau + n_emb]
        return emb

    emb_a = embed(sig_a, embedding_dim, tau)
    emb_b = embed(sig_b, embedding_dim, tau)

    # Target values
    target_a = sig_a[(embedding_dim - 1) * tau:]
    target_b = sig_b[(embedding_dim - 1) * tau:]

    def cross_map_skill(manifold_x, target_y, L):
        """Predict target_y using manifold_x"""
        from scipy.spatial import KDTree

        # Use random subsample as library
        lib_idx = np.random.choice(len(manifold_x), min(L, len(manifold_x)), replace=False)
        library = manifold_x[lib_idx]
        tree = KDTree(library)

        predictions = []
        actuals = []

        for i in range(len(manifold_x)):
            if i in lib_idx:
                continue

            # Find E+1 nearest neighbors
            dists, indices = tree.query(manifold_x[i], k=min(embedding_dim + 1, L))

            # Weights
            weights = np.exp(-dists / (dists[0] + 1e-10))
            weights = weights / weights.sum()

            # Predict
            nn_targets = target_y[lib_idx[indices]]
            pred = np.sum(weights * nn_targets)

            predictions.append(pred)
            actuals.append(target_y[i])

        if len(predictions) < 3:
            return np.nan

        rho = np.corrcoef(predictions, actuals)[0, 1]
        return rho if not np.isnan(rho) else 0.0

    # A causes B: B's manifold predicts A
    rho_a_causes_b = cross_map_skill(emb_b, target_a, library_size)

    # B causes A: A's manifold predicts B
    rho_b_causes_a = cross_map_skill(emb_a, target_b, library_size)

    return float(rho_a_causes_b), float(rho_b_causes_a)
