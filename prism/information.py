"""
PRISM Information Theory Primitives

Pure mathematical functions for information-theoretic analysis
of signals and relationships.
"""

import numpy as np
from typing import Optional, Tuple
from collections import Counter


def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    k: int = 1,
    l: int = 1,
    n_bins: int = 8
) -> float:
    """
    Estimate transfer entropy from source to target.

    Transfer entropy measures the reduction in uncertainty in the target
    given the history of the source, beyond what can be predicted from
    the target's own history.

    TE(X→Y) = H(Y_t | Y_{t-1:t-k}) - H(Y_t | Y_{t-1:t-k}, X_{t-1:t-l})

    Args:
        source: Source time series
        target: Target time series
        k: History length for target
        l: History length for source
        n_bins: Number of bins for discretization

    Returns:
        Transfer entropy (>= 0, in bits)
    """
    source = np.asarray(source, dtype=np.float64).flatten()
    target = np.asarray(target, dtype=np.float64).flatten()

    min_len = min(len(source), len(target))
    source = source[:min_len]
    target = target[:min_len]

    # Discretize
    def discretize(x, bins):
        edges = np.linspace(np.min(x) - 1e-10, np.max(x) + 1e-10, bins + 1)
        return np.digitize(x, edges[1:-1])

    source_disc = discretize(source, n_bins)
    target_disc = discretize(target, n_bins)

    # Build joint samples
    max_lag = max(k, l)
    n = len(target) - max_lag

    if n < 10:
        return 0.0

    # Current target
    y_current = target_disc[max_lag:]

    # Target history
    y_history = tuple(target_disc[max_lag - i - 1:max_lag - i - 1 + n] for i in range(k))

    # Source history
    x_history = tuple(source_disc[max_lag - i - 1:max_lag - i - 1 + n] for i in range(l))

    # Build joint states
    states_y = list(zip(*y_history)) if k > 0 else [()] * n
    states_xy = list(zip(*y_history, *x_history)) if k + l > 0 else [()] * n

    # Count occurrences
    def count_joint(y_curr, states):
        counter = Counter()
        for i in range(len(y_curr)):
            counter[(y_curr[i], states[i])] += 1
        return counter

    joint_y = count_joint(y_current, states_y)
    joint_xy = count_joint(y_current, states_xy)

    state_y_counts = Counter(states_y)
    state_xy_counts = Counter(states_xy)

    # Compute transfer entropy
    te = 0.0
    total = len(y_current)

    for (y, s_xy), count_xy in joint_xy.items():
        # Find corresponding state_y
        s_y = s_xy[:k] if k > 0 else ()

        p_y_xy = count_xy / total
        p_xy = state_xy_counts[s_xy] / total
        p_y_given_xy = count_xy / state_xy_counts[s_xy] if state_xy_counts[s_xy] > 0 else 0

        count_y = joint_y.get((y, s_y), 0)
        p_y_given_y = count_y / state_y_counts[s_y] if state_y_counts[s_y] > 0 else 0

        if p_y_given_xy > 0 and p_y_given_y > 0:
            te += p_y_xy * np.log2(p_y_given_xy / p_y_given_y)

    return float(max(0, te))


def conditional_entropy(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute conditional entropy H(X|Y).

    H(X|Y) = H(X,Y) - H(Y)

    Args:
        x: First variable
        y: Conditioning variable
        n_bins: Number of bins

    Returns:
        Conditional entropy (>= 0, in bits)
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()

    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    # Joint entropy
    h_xy = joint_entropy(x, y, n_bins)

    # Marginal entropy of Y
    h_y = _entropy(y, n_bins)

    return float(h_xy - h_y)


def joint_entropy(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute joint entropy H(X,Y).

    Args:
        x: First variable
        y: Second variable
        n_bins: Number of bins

    Returns:
        Joint entropy (in bits)
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()

    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    # 2D histogram
    hist, _, _ = np.histogram2d(x, y, bins=n_bins)
    hist = hist / np.sum(hist)

    # Entropy
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def _entropy(x: np.ndarray, n_bins: int = 10) -> float:
    """Compute Shannon entropy of a variable."""
    hist, _ = np.histogram(x, bins=n_bins)
    hist = hist / np.sum(hist)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def granger_causality(
    source: np.ndarray,
    target: np.ndarray,
    max_lag: int = 5
) -> Tuple[float, float]:
    """
    Compute Granger causality F-statistic.

    Tests whether source Granger-causes target by comparing
    AR models with and without source history.

    Args:
        source: Potential causal variable
        target: Effect variable
        max_lag: Maximum lag for AR model

    Returns:
        Tuple of (f_statistic, p_value)
    """
    from scipy import stats

    source = np.asarray(source, dtype=np.float64).flatten()
    target = np.asarray(target, dtype=np.float64).flatten()

    min_len = min(len(source), len(target))
    source = source[:min_len]
    target = target[:min_len]

    n = len(target)
    if n < max_lag + 10:
        return np.nan, np.nan

    # Build design matrices
    y = target[max_lag:]

    # Restricted model (target history only)
    X_restricted = np.column_stack([
        target[max_lag - i - 1:-i - 1] for i in range(max_lag)
    ])

    # Unrestricted model (target + source history)
    X_unrestricted = np.column_stack([
        *[target[max_lag - i - 1:-i - 1] for i in range(max_lag)],
        *[source[max_lag - i - 1:-i - 1] for i in range(max_lag)]
    ])

    # Add constant
    X_restricted = np.column_stack([np.ones(len(y)), X_restricted])
    X_unrestricted = np.column_stack([np.ones(len(y)), X_unrestricted])

    # Fit models
    try:
        beta_r = np.linalg.lstsq(X_restricted, y, rcond=None)[0]
        resid_r = y - X_restricted @ beta_r
        ssr_r = np.sum(resid_r ** 2)

        beta_u = np.linalg.lstsq(X_unrestricted, y, rcond=None)[0]
        resid_u = y - X_unrestricted @ beta_u
        ssr_u = np.sum(resid_u ** 2)
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    # F-test
    df1 = max_lag  # Additional parameters in unrestricted
    df2 = len(y) - X_unrestricted.shape[1]

    if df2 <= 0 or ssr_u == 0:
        return np.nan, np.nan

    f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)

    return float(f_stat), float(p_value)


def phase_coupling(
    signal1: np.ndarray,
    signal2: np.ndarray
) -> float:
    """
    Compute phase coupling (phase-locking value) between two signals.

    Uses Hilbert transform to extract instantaneous phase.

    Args:
        signal1: First signal
        signal2: Second signal

    Returns:
        Phase coupling strength (0 to 1)
    """
    from scipy.signal import hilbert

    signal1 = np.asarray(signal1, dtype=np.float64).flatten()
    signal2 = np.asarray(signal2, dtype=np.float64).flatten()

    min_len = min(len(signal1), len(signal2))
    signal1 = signal1[:min_len]
    signal2 = signal2[:min_len]

    # Get instantaneous phase via Hilbert transform
    analytic1 = hilbert(signal1)
    analytic2 = hilbert(signal2)

    phase1 = np.angle(analytic1)
    phase2 = np.angle(analytic2)

    # Phase difference
    phase_diff = phase1 - phase2

    # Phase-locking value (magnitude of mean phase vector)
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))

    return float(plv)


def normalized_transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    k: int = 1,
    l: int = 1,
    n_bins: int = 8
) -> float:
    """
    Compute normalized transfer entropy.

    Normalized to [0, 1] by dividing by entropy of target.

    Args:
        source: Source time series
        target: Target time series
        k: History length for target
        l: History length for source
        n_bins: Number of bins

    Returns:
        Normalized transfer entropy (0 to 1)
    """
    te = transfer_entropy(source, target, k, l, n_bins)
    h_target = _entropy(target, n_bins)

    if h_target == 0:
        return 0.0

    return float(min(1.0, te / h_target))


def information_flow(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 1,
    n_bins: int = 8
) -> Tuple[float, float, float]:
    """
    Compute bidirectional information flow between two signals.

    Args:
        x: First signal
        y: Second signal
        k: History length
        n_bins: Number of bins

    Returns:
        Tuple of (te_x_to_y, te_y_to_x, net_flow)
        net_flow is positive if X drives Y, negative if Y drives X
    """
    te_xy = transfer_entropy(x, y, k, k, n_bins)
    te_yx = transfer_entropy(y, x, k, k, n_bins)

    return float(te_xy), float(te_yx), float(te_xy - te_yx)
