"""
Information Transfer Primitives (113)

Transfer entropy for causal information flow.
"""

import numpy as np
from typing import Optional

try:
    from rudder_primitives_rs.information import (
        transfer_entropy as _transfer_entropy_rs,
    )
    _USE_RUST_TE = True
except ImportError:
    _USE_RUST_TE = False


def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    n_bins: int = 10,
    history_length: int = 1
) -> float:
    """
    Compute transfer entropy from source to target.

    Parameters
    ----------
    source : np.ndarray
        Potential cause signal
    target : np.ndarray
        Potential effect signal
    lag : int
        Time lag for source
    n_bins : int
        Number of bins for discretization
    history_length : int
        How many past values of target to condition on

    Returns
    -------
    float
        Transfer entropy TE(source → target) in bits

    Notes
    -----
    TE(X→Y) = I(Y_future ; X_past | Y_past)
            = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)

    "How much does X's past reduce uncertainty about Y's future,
    beyond what Y's own past tells us?"

    Properties:
    - TE ≥ 0
    - TE(X→Y) ≠ TE(Y→X) in general (asymmetric!)
    - TE = 0 iff X provides no additional information about Y's future

    Physical interpretation:
    "Does signal X CAUSE signal Y?"

    Key insight: Causality is about PREDICTION, not correlation.
    X causes Y if X's past helps predict Y's future.

    TE vs Granger:
    - Granger: Linear causality (F-test)
    - TE: Nonlinear causality (information-theoretic)
    """
    source = np.asarray(source).flatten()
    target = np.asarray(target).flatten()

    # Align lengths
    n = min(len(source), len(target))
    source = source[:n]
    target = target[:n]

    # Remove NaN
    valid = ~(np.isnan(source) | np.isnan(target))
    source = source[valid]
    target = target[valid]

    # Need enough samples for lagged values
    if len(source) < lag + history_length + 10:
        return 0.0

    if _USE_RUST_TE and history_length == 1:
        return _transfer_entropy_rs(source, target, lag, n_bins)

    n_samples = len(source) - lag - history_length

    # Discretize
    def discretize(s):
        s_min, s_max = s.min(), s.max()
        if s_max == s_min:
            return np.zeros(len(s), dtype=int)
        return np.digitize(s, np.linspace(s_min, s_max + 1e-10, n_bins + 1)[:-1]) - 1

    source_d = discretize(source)
    target_d = discretize(target)

    # Build arrays
    y_future = target_d[lag + history_length:][:n_samples]
    y_past = target_d[history_length:history_length + n_samples]
    x_past = source_d[:n_samples]

    # Compute entropies
    def entropy_counts(*arrays):
        combined = np.column_stack(arrays)
        _, counts = np.unique(combined, axis=0, return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log2(p + 1e-10))

    # H(Y_future | Y_past) = H(Y_future, Y_past) - H(Y_past)
    H_yf_yp = entropy_counts(y_future, y_past) - entropy_counts(y_past)

    # H(Y_future | Y_past, X_past) = H(Y_future, Y_past, X_past) - H(Y_past, X_past)
    H_yf_yp_xp = entropy_counts(y_future, y_past, x_past) - entropy_counts(y_past, x_past)

    te = H_yf_yp - H_yf_yp_xp

    return float(max(0, te))


def transfer_entropy_effective(
    source: np.ndarray,
    target: np.ndarray,
    max_lag: int = 10,
    n_bins: int = 10
) -> tuple:
    """
    Compute effective transfer entropy with optimal lag detection.

    Parameters
    ----------
    source : np.ndarray
        Potential cause signal
    target : np.ndarray
        Potential effect signal
    max_lag : int
        Maximum lag to test
    n_bins : int
        Number of bins

    Returns
    -------
    te : float
        Maximum transfer entropy found
    optimal_lag : int
        Lag with maximum TE
    te_curve : np.ndarray
        TE at each lag

    Notes
    -----
    Tests multiple lags and returns the one with maximum information transfer.
    Useful when the causal delay is unknown.
    """
    te_curve = np.zeros(max_lag)

    for lag in range(1, max_lag + 1):
        te_curve[lag - 1] = transfer_entropy(source, target, lag, n_bins)

    optimal_lag = np.argmax(te_curve) + 1
    max_te = te_curve[optimal_lag - 1]

    return float(max_te), int(optimal_lag), te_curve


def net_transfer_entropy(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    lag: int = 1,
    n_bins: int = 10
) -> float:
    """
    Compute net transfer entropy (asymmetry).

    Parameters
    ----------
    sig_a : np.ndarray
        First signal
    sig_b : np.ndarray
        Second signal
    lag : int
        Time lag
    n_bins : int
        Number of bins

    Returns
    -------
    float
        Net TE = TE(A→B) - TE(B→A)

    Notes
    -----
    Positive: A causes B more than B causes A
    Negative: B causes A more than A causes B
    Zero: Symmetric or no causal relationship
    """
    te_ab = transfer_entropy(sig_a, sig_b, lag, n_bins)
    te_ba = transfer_entropy(sig_b, sig_a, lag, n_bins)

    return float(te_ab - te_ba)
