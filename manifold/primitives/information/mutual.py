"""
Information Mutual Information Primitives (115-116)

Mutual information, conditional MI, multivariate MI.
"""

import numpy as np
from typing import List, Optional, Tuple

try:
    from rudder_primitives_rs.information import (
        mutual_information as _mutual_info_rs,
    )
    _USE_RUST_MI = True
except ImportError:
    _USE_RUST_MI = False


def mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = None,
    normalized: bool = False,
    base: float = 2
) -> float:
    """
    Compute mutual information I(X; Y).

    Parameters
    ----------
    x, y : np.ndarray
        Input variables
    bins : int, optional
        Number of bins
    normalized : bool
        If True, normalize to [0, 1]
    base : float
        Logarithm base

    Returns
    -------
    float
        Mutual information

    Notes
    -----
    I(X; Y) = H(X) + H(Y) - H(X, Y)
            = H(X) - H(X|Y)
            = H(Y) - H(Y|X)
            = D_KL(P(X,Y) || P(X)P(Y))

    Measures shared information between X and Y.
    I(X;Y) = 0 iff X and Y are independent.
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    if len(x) < 10:
        return 0.0

    if bins is None:
        bins = max(5, int(np.sqrt(len(x))))

    if _USE_RUST_MI:
        return _mutual_info_rs(x, y, bins, normalized, base)

    # Estimate marginal and joint entropies
    h_x = _entropy(x, bins, base)
    h_y = _entropy(y, bins, base)
    h_xy = _joint_entropy(x, y, bins, base)

    mi = h_x + h_y - h_xy

    # Ensure non-negative (numerical issues can cause small negatives)
    mi = max(0, mi)

    if normalized:
        max_mi = min(h_x, h_y)
        mi = mi / max_mi if max_mi > 0 else 0.0

    return float(mi)


def conditional_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bins: int = None,
    base: float = 2
) -> float:
    """
    Compute conditional mutual information I(X; Y | Z).

    Parameters
    ----------
    x, y, z : np.ndarray
        Input variables
    bins : int, optional
        Number of bins
    base : float
        Logarithm base

    Returns
    -------
    float
        Conditional mutual information

    Notes
    -----
    I(X; Y | Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z)
                = H(X,Z) + H(Y,Z) - H(Z) - H(X,Y,Z)

    Measures information shared between X and Y that is not
    explained by Z.

    If I(X;Y|Z) ≈ 0 and I(X;Y) > 0, then Z "mediates" the
    relationship between X and Y.
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    z = np.asarray(z).flatten()

    n = min(len(x), len(y), len(z))
    x, y, z = x[:n], y[:n], z[:n]

    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[mask], y[mask], z[mask]

    if len(x) < 20:
        return 0.0

    if bins is None:
        bins = max(4, int(len(x) ** (1/4)))

    # Joint entropies
    h_xz = _joint_entropy(x, z, bins, base)
    h_yz = _joint_entropy(y, z, bins, base)
    h_z = _entropy(z, bins, base)
    h_xyz = _joint_entropy_3d(x, y, z, bins, base)

    cmi = h_xz + h_yz - h_z - h_xyz

    return float(max(0, cmi))


def multivariate_mutual_information(
    variables: List[np.ndarray],
    bins: int = None,
    base: float = 2
) -> float:
    """
    Compute multivariate mutual information (co-information).

    Parameters
    ----------
    variables : list of np.ndarray
        List of variables
    bins : int, optional
        Number of bins
    base : float
        Logarithm base

    Returns
    -------
    float
        Multivariate MI / co-information

    Notes
    -----
    For 3 variables:
    I(X; Y; Z) = I(X; Y) - I(X; Y | Z)

    Can be positive (redundancy) or negative (synergy).
    Positive: X, Y share info that Z also has
    Negative: X, Y share info only when combined with Z
    """
    if len(variables) < 2:
        return 0.0

    if len(variables) == 2:
        return mutual_information(variables[0], variables[1], bins, base=base)

    # Align all variables
    n = min(len(v) for v in variables)
    variables = [np.asarray(v).flatten()[:n] for v in variables]

    # For 3 variables: I(X;Y;Z) = I(X;Y) - I(X;Y|Z)
    if len(variables) == 3:
        x, y, z = variables
        mi_xy = mutual_information(x, y, bins, base=base)
        cmi_xy_z = conditional_mutual_information(x, y, z, bins, base)
        return float(mi_xy - cmi_xy_z)

    # For more variables: use inclusion-exclusion
    # This is computationally expensive and has estimation issues
    # Use total correlation instead for practical purposes
    return _inclusion_exclusion_mi(variables, bins, base)


def total_correlation(
    variables: List[np.ndarray],
    bins: int = None,
    base: float = 2
) -> float:
    """
    Compute total correlation (multi-information).

    Parameters
    ----------
    variables : list of np.ndarray
        List of variables
    bins : int, optional
        Number of bins
    base : float
        Logarithm base

    Returns
    -------
    float
        Total correlation

    Notes
    -----
    TC(X_1, ..., X_n) = Σ H(X_i) - H(X_1, ..., X_n)
                      = D_KL(P(X_1,...,X_n) || Π P(X_i))

    Measures total amount of dependence among variables.
    TC = 0 iff all variables are mutually independent.
    TC ≥ 0 always.
    """
    if len(variables) < 2:
        return 0.0

    # Align all variables
    n = min(len(v) for v in variables)
    variables = [np.asarray(v).flatten()[:n] for v in variables]

    mask = np.ones(n, dtype=bool)
    for v in variables:
        mask &= ~np.isnan(v)

    variables = [v[mask] for v in variables]
    n = len(variables[0])

    if n < 20:
        return 0.0

    if bins is None:
        bins = max(3, int(n ** (1 / (len(variables) + 1))))

    # Sum of marginal entropies
    sum_h = sum(_entropy(v, bins, base) for v in variables)

    # Joint entropy
    h_joint = _joint_entropy_nd(variables, bins, base)

    tc = sum_h - h_joint

    return float(max(0, tc))


def interaction_information(
    variables: List[np.ndarray],
    bins: int = None,
    base: float = 2
) -> float:
    """
    Compute interaction information (same as multivariate MI).

    Parameters
    ----------
    variables : list of np.ndarray
        List of variables
    bins : int, optional
        Number of bins
    base : float
        Logarithm base

    Returns
    -------
    float
        Interaction information

    Notes
    -----
    Alias for multivariate_mutual_information.
    Measures higher-order interactions not captured by pairwise MI.
    """
    return multivariate_mutual_information(variables, bins, base)


def dual_total_correlation(
    variables: List[np.ndarray],
    bins: int = None,
    base: float = 2
) -> float:
    """
    Compute dual total correlation (binding information).

    Parameters
    ----------
    variables : list of np.ndarray
        List of variables
    bins : int, optional
        Number of bins
    base : float
        Logarithm base

    Returns
    -------
    float
        Dual total correlation

    Notes
    -----
    B(X_1, ..., X_n) = H(X_1, ..., X_n) - Σ H(X_i | X_{-i})

    Alternative measure of shared information.
    Focuses on unique information each variable contributes.
    """
    if len(variables) < 2:
        return 0.0

    # Align variables
    n = min(len(v) for v in variables)
    variables = [np.asarray(v).flatten()[:n] for v in variables]

    mask = np.ones(n, dtype=bool)
    for v in variables:
        mask &= ~np.isnan(v)

    variables = [v[mask] for v in variables]
    n = len(variables[0])

    if n < 20:
        return 0.0

    if bins is None:
        bins = max(3, int(n ** (1 / (len(variables) + 1))))

    # Joint entropy
    h_joint = _joint_entropy_nd(variables, bins, base)

    # Sum of conditional entropies H(X_i | X_{-i})
    sum_cond_h = 0.0
    for i in range(len(variables)):
        others = [variables[j] for j in range(len(variables)) if j != i]
        h_xi_others = _joint_entropy_nd([variables[i]] + others, bins, base)
        h_others = _joint_entropy_nd(others, bins, base)
        sum_cond_h += h_xi_others - h_others

    dtc = h_joint - sum_cond_h

    return float(max(0, dtc))


# Helper functions

def _entropy(data: np.ndarray, bins: int, base: float) -> float:
    """Compute entropy of single variable."""
    counts, _ = np.histogram(data, bins=bins)
    p = counts / counts.sum()
    p = p[p > 0]

    if base == 2:
        return float(-np.sum(p * np.log2(p)))
    elif base == np.e:
        return float(-np.sum(p * np.log(p)))
    else:
        return float(-np.sum(p * np.log(p)) / np.log(base))


def _joint_entropy(x: np.ndarray, y: np.ndarray, bins: int, base: float) -> float:
    """Compute joint entropy of two variables."""
    hist, _, _ = np.histogram2d(x, y, bins=bins)
    p = hist.flatten() / hist.sum()
    p = p[p > 0]

    if base == 2:
        return float(-np.sum(p * np.log2(p)))
    elif base == np.e:
        return float(-np.sum(p * np.log(p)))
    else:
        return float(-np.sum(p * np.log(p)) / np.log(base))


def _joint_entropy_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bins: int,
    base: float
) -> float:
    """Compute joint entropy of three variables."""
    hist, _ = np.histogramdd([x, y, z], bins=bins)
    p = hist.flatten() / hist.sum()
    p = p[p > 0]

    if base == 2:
        return float(-np.sum(p * np.log2(p)))
    elif base == np.e:
        return float(-np.sum(p * np.log(p)))
    else:
        return float(-np.sum(p * np.log(p)) / np.log(base))


def _joint_entropy_nd(
    variables: List[np.ndarray],
    bins: int,
    base: float
) -> float:
    """Compute joint entropy of n variables."""
    data = np.column_stack(variables)
    hist, _ = np.histogramdd(data, bins=bins)
    p = hist.flatten() / hist.sum()
    p = p[p > 0]

    if base == 2:
        return float(-np.sum(p * np.log2(p)))
    elif base == np.e:
        return float(-np.sum(p * np.log(p)))
    else:
        return float(-np.sum(p * np.log(p)) / np.log(base))


def _inclusion_exclusion_mi(
    variables: List[np.ndarray],
    bins: int,
    base: float
) -> float:
    """Compute multivariate MI using inclusion-exclusion."""
    from itertools import combinations

    n_vars = len(variables)
    result = 0.0

    # Sum over all subsets
    for k in range(1, n_vars + 1):
        sign = (-1) ** (n_vars - k)

        for subset in combinations(range(n_vars), k):
            subset_vars = [variables[i] for i in subset]

            if len(subset_vars) == 1:
                h = _entropy(subset_vars[0], bins, base)
            else:
                h = _joint_entropy_nd(subset_vars, bins, base)

            result += sign * h

    return float(result)
