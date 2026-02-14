"""
Pairwise Information Primitives (44-45)

Mutual information and transfer entropy.
"""

import numpy as np
from typing import Optional

try:
    from rudder_primitives_rs.information import (
        mutual_information as _mutual_info_rs,
        transfer_entropy as _transfer_entropy_rs,
    )
    _USE_RUST_INFO = True
except ImportError:
    _USE_RUST_INFO = False


def mutual_information(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    bins: int = 20,
    normalized: bool = False
) -> float:
    """
    Compute mutual information.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Input signals
    bins : int
        Number of histogram bins
    normalized : bool
        If True, normalize to [0, 1]

    Returns
    -------
    float
        Mutual information in bits

    Notes
    -----
    I(A; B) = H(A) + H(B) - H(A, B)
    Measures any (not just linear) statistical dependence.
    """
    sig_a = np.asarray(sig_a).flatten()
    sig_b = np.asarray(sig_b).flatten()

    n = min(len(sig_a), len(sig_b))
    sig_a, sig_b = sig_a[:n], sig_b[:n]

    mask = ~(np.isnan(sig_a) | np.isnan(sig_b))
    sig_a, sig_b = sig_a[mask], sig_b[mask]

    if len(sig_a) < 10:
        return np.nan

    if _USE_RUST_INFO:
        return _mutual_info_rs(sig_a, sig_b, bins, normalized)

    # Joint histogram
    hist_ab, _, _ = np.histogram2d(sig_a, sig_b, bins=bins)
    hist_ab = hist_ab / hist_ab.sum()

    # Marginals
    hist_a = hist_ab.sum(axis=1)
    hist_b = hist_ab.sum(axis=0)

    # Mutual information
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if hist_ab[i, j] > 0 and hist_a[i] > 0 and hist_b[j] > 0:
                mi += hist_ab[i, j] * np.log2(hist_ab[i, j] / (hist_a[i] * hist_b[j]))

    if normalized:
        # H(A) and H(B)
        h_a = -np.sum(hist_a[hist_a > 0] * np.log2(hist_a[hist_a > 0]))
        h_b = -np.sum(hist_b[hist_b > 0] * np.log2(hist_b[hist_b > 0]))
        max_mi = min(h_a, h_b)
        if max_mi > 0:
            mi = mi / max_mi

    return float(max(0.0, mi))


def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    history: int = 1,
    bins: int = 8
) -> float:
    """
    Compute transfer entropy T(source -> target).

    Parameters
    ----------
    source : np.ndarray
        Potential cause signal
    target : np.ndarray
        Potential effect signal
    lag : int
        Prediction horizon
    history : int
        Number of past values to condition on
    bins : int
        Discretization bins

    Returns
    -------
    float
        Transfer entropy in bits

    Notes
    -----
    T(X->Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
    Measures directed information flow.
    T(X->Y) > 0 implies X has predictive information about Y.
    """
    source = np.asarray(source).flatten()
    target = np.asarray(target).flatten()

    n = min(len(source), len(target))
    source, target = source[:n], target[:n]

    start = history
    end = n - lag

    if end <= start + 10:
        return np.nan

    if _USE_RUST_INFO and history == 1:
        return _transfer_entropy_rs(source, target, lag, bins)

    # Build lagged arrays
    y_future = target[start + lag : end + lag]
    y_past = np.column_stack([target[start - i : end - i] for i in range(1, history + 1)])
    x_past = np.column_stack([source[start - i : end - i] for i in range(1, history + 1)])

    # Discretize
    def discretize(arr, bins):
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        result = np.zeros_like(arr, dtype=int)
        for j in range(arr.shape[1]):
            edges = np.linspace(arr[:, j].min() - 1e-10, arr[:, j].max() + 1e-10, bins + 1)
            result[:, j] = np.clip(np.digitize(arr[:, j], edges[:-1]) - 1, 0, bins - 1)
        return result

    y_future_d = discretize(y_future, bins).flatten()
    y_past_d = discretize(y_past, bins)
    x_past_d = discretize(x_past, bins)

    # Combine past indices
    def to_index(arr, bins):
        if arr.ndim == 1:
            return arr
        idx = np.zeros(len(arr), dtype=np.int64)
        for j in range(arr.shape[1]):
            idx = idx * bins + arr[:, j]
        return idx

    y_past_idx = to_index(y_past_d, bins)
    xy_past_idx = y_past_idx * (bins ** history) + to_index(x_past_d, bins)

    # Count probabilities
    def count_joint(idx1, idx2, n1, n2):
        counts = np.zeros((n1, n2))
        for i1, i2 in zip(idx1, idx2):
            if 0 <= i1 < n1 and 0 <= i2 < n2:
                counts[int(i1), int(i2)] += 1
        return counts

    n_y = bins
    n_ypast = bins ** history
    n_xypast = bins ** (2 * history)

    # P(Y_future, Y_past)
    counts_yf_yp = count_joint(y_future_d, y_past_idx, n_y, n_ypast)
    p_yf_yp = counts_yf_yp / max(1, counts_yf_yp.sum())
    p_yp = p_yf_yp.sum(axis=0)

    # P(Y_future, Y_past, X_past)
    counts_yf_xyp = count_joint(y_future_d, xy_past_idx, n_y, n_xypast)
    p_yf_xyp = counts_yf_xyp / max(1, counts_yf_xyp.sum())
    p_xyp = p_yf_xyp.sum(axis=0)

    # Entropies
    def entropy(p):
        p = p.flatten()
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    h_yf_yp = entropy(p_yf_yp)
    h_yp = entropy(p_yp)
    h_yf_given_yp = h_yf_yp - h_yp

    h_yf_xyp = entropy(p_yf_xyp)
    h_xyp = entropy(p_xyp)
    h_yf_given_xyp = h_yf_xyp - h_xyp

    te = h_yf_given_yp - h_yf_given_xyp
    return float(max(0.0, te))
