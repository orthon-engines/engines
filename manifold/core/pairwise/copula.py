"""
Copula Engine
=============

Models the dependency structure between two signals beyond linear
correlation. Copulas separate marginal distributions from the
dependency structure, revealing:

  - Tail dependence: Do signals crash together? (lower tail)
    Do they boom together? (upper tail)
  - Asymmetric dependence: Correlation during crashes vs rallies
  - Non-linear dependence: Relationships that Pearson misses

Method:
  1. Transform both signals to uniform marginals via empirical CDF (rank transform)
  2. Fit parametric copula families (Gaussian, Clayton, Gumbel, Frank)
  3. Select best fit via AIC
  4. Extract tail dependence coefficients

Layer: Causal Mechanics (pairwise)
Used by: signal_pairwise, divergence analysis

References:
    Sklar (1959) "Fonctions de répartition à n dimensions et leurs marges"
    Joe (2014) "Dependence Modeling with Copulas"
    Embrechts, McNeil & Straumann (2002) "Correlation and Dependence in Risk Management"
"""

import warnings

import numpy as np
from typing import Optional


def compute(
    y1: np.ndarray,
    y2: np.ndarray,
    families: Optional[list] = None,
) -> dict:
    """
    Fit copula models to two signals and extract dependency structure.

    Args:
        y1: First signal (1D array)
        y2: Second signal (1D array)
        families: Copula families to test (default: all four)

    Returns:
        dict with:
            - best_family: str ("gaussian", "clayton", "gumbel", "frank")
            - best_aic: float
            - best_param: float (copula parameter theta)
            - kendall_tau: float (rank correlation)
            - spearman_rho: float (Spearman rank correlation)
            - lower_tail_dependence: float (lambda_L, 0-1)
            - upper_tail_dependence: float (lambda_U, 0-1)
            - tail_asymmetry: float (lambda_U - lambda_L)
            - tail_dependence_ratio: float (max/min tail dep)
            - gaussian_param: float (rho for Gaussian copula)
            - gaussian_aic: float
            - clayton_param: float (theta for Clayton)
            - clayton_aic: float
            - gumbel_param: float (theta for Gumbel)
            - gumbel_aic: float
            - frank_param: float (theta for Frank)
            - frank_aic: float
            - empirical_lower_tail: float (observed lower tail concentration)
            - empirical_upper_tail: float (observed upper tail concentration)
            - n_samples: int
    """
    if families is None:
        families = ["gaussian", "clayton", "gumbel", "frank"]

    # Clean inputs
    mask = ~(np.isnan(y1) | np.isnan(y2))
    y1_clean = y1[mask].astype(np.float64)
    y2_clean = y2[mask].astype(np.float64)

    n = len(y1_clean)

    if n < 20:
        import logging
        logging.getLogger(__name__).debug(
            f"Copula skipped: insufficient_data (n={n}, need 20)"
        )
        return _empty_result(n, reason="insufficient_data")

    if np.std(y1_clean) < 1e-10 or np.std(y2_clean) < 1e-10:
        import logging
        logging.getLogger(__name__).debug(
            f"Copula skipped: constant_signal (std_y1={np.std(y1_clean):.2e}, std_y2={np.std(y2_clean):.2e}, n={n})"
        )
        return _empty_result(n, reason="constant_signal")

    # Step 1: Transform to pseudo-observations (uniform marginals)
    u1 = _empirical_cdf(y1_clean)
    u2 = _empirical_cdf(y2_clean)

    # Rank correlations
    kendall_tau = _kendall_tau(y1_clean, y2_clean)
    spearman_rho = _spearman_rho(y1_clean, y2_clean)

    # Empirical tail concentrations
    empirical_lower = _empirical_tail_dependence(u1, u2, quantile=0.10, tail="lower")
    empirical_upper = _empirical_tail_dependence(u1, u2, quantile=0.10, tail="upper")

    # Step 2: Fit each copula family
    results = {}

    for family in families:
        try:
            if family == "gaussian":
                results["gaussian"] = _fit_gaussian(u1, u2, kendall_tau)
            elif family == "clayton":
                results["clayton"] = _fit_clayton(u1, u2, kendall_tau)
            elif family == "gumbel":
                results["gumbel"] = _fit_gumbel(u1, u2, kendall_tau)
            elif family == "frank":
                results["frank"] = _fit_frank(u1, u2, kendall_tau)
        except (ValueError, np.linalg.LinAlgError):
            pass
        except Exception as e:
            warnings.warn(f"copula.compute: fitting {family}: {type(e).__name__}: {e}", RuntimeWarning, stacklevel=2)

    # Step 3: Select best by AIC
    if not results:
        result = _empty_result(n, reason="all_families_failed")
        result["kendall_tau"] = float(kendall_tau)
        result["spearman_rho"] = float(spearman_rho)
        result["empirical_lower_tail"] = float(empirical_lower)
        result["empirical_upper_tail"] = float(empirical_upper)
        return result

    best_family = None
    best_aic = np.inf
    best_param = float("nan")

    for family, res in results.items():
        if res["aic"] < best_aic:
            best_aic = res["aic"]
            best_family = family
            best_param = res["param"]

    # Step 4: Tail dependence from best parametric model
    lower_tail, upper_tail = _parametric_tail_dependence(best_family, best_param)
    tail_asymmetry = upper_tail - lower_tail
    tail_ratio = (
        max(upper_tail, lower_tail) / max(min(upper_tail, lower_tail), 1e-10)
        if max(upper_tail, lower_tail) > 0.001
        else 1.0
    )

    return {
        "best_family": best_family,
        "best_aic": float(best_aic),
        "best_param": float(best_param),
        "kendall_tau": float(kendall_tau),
        "spearman_rho": float(spearman_rho),
        "lower_tail_dependence": float(lower_tail),
        "upper_tail_dependence": float(upper_tail),
        "tail_asymmetry": float(tail_asymmetry),
        "tail_dependence_ratio": float(tail_ratio),
        "gaussian_param": float(results.get("gaussian", {}).get("param", float("nan"))),
        "gaussian_aic": float(results.get("gaussian", {}).get("aic", float("inf"))),
        "clayton_param": float(results.get("clayton", {}).get("param", float("nan"))),
        "clayton_aic": float(results.get("clayton", {}).get("aic", float("inf"))),
        "gumbel_param": float(results.get("gumbel", {}).get("param", float("nan"))),
        "gumbel_aic": float(results.get("gumbel", {}).get("aic", float("inf"))),
        "frank_param": float(results.get("frank", {}).get("param", float("nan"))),
        "frank_aic": float(results.get("frank", {}).get("aic", float("inf"))),
        "empirical_lower_tail": float(empirical_lower),
        "empirical_upper_tail": float(empirical_upper),
        "n_samples": int(n),
    }


# ===========================================================================
# Marginal transformation
# ===========================================================================

def _empirical_cdf(x: np.ndarray) -> np.ndarray:
    """
    Transform to pseudo-observations via rank transform.
    Returns values in (0, 1) — excludes exactly 0 and 1 to
    avoid log(0) in likelihood computations.

    Uses the standard formula: u_i = rank(x_i) / (n + 1)
    """
    n = len(x)
    ranks = np.argsort(np.argsort(x)).astype(np.float64)
    return (ranks + 1.0) / (n + 1.0)


# ===========================================================================
# Rank correlations (pure numpy)
# ===========================================================================

def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall's tau rank correlation. O(n log n) approximation."""
    n = len(x)
    if n < 3:
        return 0.0

    # Use the O(n²) direct method for small n, sample for large n
    if n > 1000:
        idx = np.random.RandomState(42).choice(n, 1000, replace=False)
        x, y = x[idx], y[idx]
        n = 1000

    concordant = 0
    discordant = 0
    for i in range(n - 1):
        dx = x[i + 1 :] - x[i]
        dy = y[i + 1 :] - y[i]
        product = dx * dy
        concordant += np.sum(product > 0)
        discordant += np.sum(product < 0)

    denom = concordant + discordant
    if denom == 0:
        return 0.0

    return float((concordant - discordant) / denom)


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation."""
    n = len(x)
    if n < 3:
        return 0.0

    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)

    mx, my = np.mean(rx), np.mean(ry)
    num = np.sum((rx - mx) * (ry - my))
    den = np.sqrt(np.sum((rx - mx) ** 2) * np.sum((ry - my) ** 2))

    return float(num / den) if den > 1e-15 else 0.0


# ===========================================================================
# Empirical tail dependence
# ===========================================================================

def _empirical_tail_dependence(
    u1: np.ndarray, u2: np.ndarray, quantile: float = 0.10, tail: str = "lower"
) -> float:
    """
    Non-parametric tail dependence estimate.

    Lower tail: P(U2 < q | U1 < q)
    Upper tail: P(U2 > 1-q | U1 > 1-q)
    """
    if tail == "lower":
        in_tail = u1 < quantile
        if np.sum(in_tail) == 0:
            return 0.0
        return float(np.mean(u2[in_tail] < quantile))
    else:
        threshold = 1.0 - quantile
        in_tail = u1 > threshold
        if np.sum(in_tail) == 0:
            return 0.0
        return float(np.mean(u2[in_tail] > threshold))


# ===========================================================================
# Copula fitting — method of moments via Kendall's tau
# ===========================================================================

def _fit_gaussian(u1: np.ndarray, u2: np.ndarray, tau: float) -> dict:
    """
    Gaussian copula. Parameter rho estimated from Kendall's tau.
    rho = sin(pi * tau / 2)
    """
    rho = np.sin(np.pi * tau / 2.0)
    rho = np.clip(rho, -0.999, 0.999)

    # Log-likelihood
    ll = _gaussian_copula_loglik(u1, u2, rho)
    aic = -2 * ll + 2  # 1 parameter

    return {"param": float(rho), "loglik": float(ll), "aic": float(aic)}


def _fit_clayton(u1: np.ndarray, u2: np.ndarray, tau: float) -> dict:
    """
    Clayton copula. Strong lower tail dependence.
    theta = 2 * tau / (1 - tau)  for tau > 0
    """
    if tau <= 0.01:
        return {"param": 0.01, "loglik": 0.0, "aic": np.inf}

    theta = 2.0 * tau / (1.0 - tau)
    theta = max(theta, 0.01)

    ll = _clayton_copula_loglik(u1, u2, theta)
    aic = -2 * ll + 2

    return {"param": float(theta), "loglik": float(ll), "aic": float(aic)}


def _fit_gumbel(u1: np.ndarray, u2: np.ndarray, tau: float) -> dict:
    """
    Gumbel copula. Strong upper tail dependence.
    theta = 1 / (1 - tau)  for tau > 0
    """
    if tau <= 0.01:
        return {"param": 1.0, "loglik": 0.0, "aic": np.inf}

    theta = 1.0 / (1.0 - tau)
    theta = max(theta, 1.001)

    ll = _gumbel_copula_loglik(u1, u2, theta)
    aic = -2 * ll + 2

    return {"param": float(theta), "loglik": float(ll), "aic": float(aic)}


def _fit_frank(u1: np.ndarray, u2: np.ndarray, tau: float) -> dict:
    """
    Frank copula. Symmetric tail dependence (none).
    theta estimated via numerical inversion of tau-theta relationship.
    """
    if abs(tau) < 0.01:
        return {"param": 0.01, "loglik": 0.0, "aic": np.inf}

    # Approximate theta from tau using bisection
    theta = _frank_theta_from_tau(tau)

    ll = _frank_copula_loglik(u1, u2, theta)
    aic = -2 * ll + 2

    return {"param": float(theta), "loglik": float(ll), "aic": float(aic)}


# ===========================================================================
# Copula log-likelihoods (pure numpy)
# ===========================================================================

def _gaussian_copula_loglik(u1: np.ndarray, u2: np.ndarray, rho: float) -> float:
    """Log-likelihood of the Gaussian copula density."""
    from scipy.special import ndtri  # only scipy dependency — inverse normal CDF

    # Transform to normal quantiles
    x1 = ndtri(np.clip(u1, 1e-6, 1 - 1e-6))
    x2 = ndtri(np.clip(u2, 1e-6, 1 - 1e-6))

    rho2 = rho ** 2
    if rho2 > 0.999:
        rho2 = 0.999

    # Gaussian copula density: c(u1,u2) = (1/sqrt(1-rho²)) * exp(-(rho²(x1²+x2²) - 2*rho*x1*x2) / (2*(1-rho²)))
    ll = -0.5 * np.log(1 - rho2) - (rho2 * (x1 ** 2 + x2 ** 2) - 2 * rho * x1 * x2) / (
        2 * (1 - rho2)
    )

    return float(np.sum(ll))


def _clayton_copula_loglik(u1: np.ndarray, u2: np.ndarray, theta: float) -> float:
    """Log-likelihood of the Clayton copula density."""
    # c(u1,u2) = (1+theta) * (u1*u2)^(-1-theta) * (u1^(-theta) + u2^(-theta) - 1)^(-1/theta - 2)
    u1c = np.clip(u1, 1e-6, 1 - 1e-6)
    u2c = np.clip(u2, 1e-6, 1 - 1e-6)

    term = u1c ** (-theta) + u2c ** (-theta) - 1.0
    # Guard against negative values (can happen with very small theta)
    term = np.maximum(term, 1e-10)

    ll = (
        np.log(1.0 + theta)
        + (-1.0 - theta) * (np.log(u1c) + np.log(u2c))
        + (-1.0 / theta - 2.0) * np.log(term)
    )

    # Filter out infs/nans
    ll = ll[np.isfinite(ll)]
    return float(np.sum(ll)) if len(ll) > 0 else -np.inf


def _gumbel_copula_loglik(u1: np.ndarray, u2: np.ndarray, theta: float) -> float:
    """Log-likelihood of the Gumbel copula density (approximate)."""
    u1c = np.clip(u1, 1e-6, 1 - 1e-6)
    u2c = np.clip(u2, 1e-6, 1 - 1e-6)

    lu1 = -np.log(u1c)
    lu2 = -np.log(u2c)

    A = (lu1 ** theta + lu2 ** theta)
    A = np.maximum(A, 1e-10)
    A_inv_theta = A ** (1.0 / theta)

    # Gumbel copula: C(u1,u2) = exp(-A^(1/theta))
    # Log-density is complex; use simplified form
    log_C = -A_inv_theta
    C = np.exp(log_C)

    # Copula density via numerical differentiation (more stable than analytic)
    # c = C * (1/(u1*u2)) * A^(1/theta - 2) * (lu1*lu2)^(theta-1) * (A^(1/theta) + theta - 1)
    ll = (
        log_C
        - np.log(u1c)
        - np.log(u2c)
        + (1.0 / theta - 2.0) * np.log(A)
        + (theta - 1.0) * (np.log(lu1) + np.log(lu2))
        + np.log(A_inv_theta + theta - 1.0)
    )

    ll = ll[np.isfinite(ll)]
    return float(np.sum(ll)) if len(ll) > 0 else -np.inf


def _frank_copula_loglik(u1: np.ndarray, u2: np.ndarray, theta: float) -> float:
    """Log-likelihood of the Frank copula density."""
    u1c = np.clip(u1, 1e-6, 1 - 1e-6)
    u2c = np.clip(u2, 1e-6, 1 - 1e-6)

    if abs(theta) < 1e-6:
        # Independence copula
        return 0.0

    et = np.exp(-theta)
    eu1 = np.exp(-theta * u1c)
    eu2 = np.exp(-theta * u2c)

    # c(u1,u2) = -theta * (et - 1) * exp(-theta*(u1+u2)) / ((et-1) + (eu1-1)*(eu2-1))^2
    numer = -theta * (et - 1.0) * np.exp(-theta * (u1c + u2c))
    denom = ((et - 1.0) + (eu1 - 1.0) * (eu2 - 1.0)) ** 2

    denom = np.maximum(np.abs(denom), 1e-15)
    ratio = np.abs(numer) / denom

    ll = np.log(np.maximum(ratio, 1e-15))
    ll = ll[np.isfinite(ll)]

    return float(np.sum(ll)) if len(ll) > 0 else -np.inf


def _frank_theta_from_tau(tau: float) -> float:
    """
    Numerical inversion of Frank copula tau-theta relationship.
    tau = 1 - 4/theta * (1 - D1(theta))
    where D1 is the first Debye function.

    Uses bisection search.
    """
    sign = 1.0 if tau > 0 else -1.0
    tau_abs = abs(tau)

    # Bisection
    lo, hi = 0.01, 50.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        tau_mid = _frank_tau_from_theta(mid)
        if tau_mid < tau_abs:
            lo = mid
        else:
            hi = mid

    return sign * (lo + hi) / 2.0


def _frank_tau_from_theta(theta: float) -> float:
    """Kendall's tau as function of Frank theta (positive theta only)."""
    if theta < 0.01:
        return 0.0

    # Debye function D1(x) = (1/x) * integral_0^x t/(exp(t)-1) dt
    # Approximate via numerical integration
    t = np.linspace(0.01, theta, 100)
    integrand = t / (np.exp(t) - 1.0)
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    D1 = _trapz(integrand, t) / theta

    return 1.0 - 4.0 / theta * (1.0 - D1)


# ===========================================================================
# Parametric tail dependence
# ===========================================================================

def _parametric_tail_dependence(family: str, param: float) -> tuple:
    """
    Theoretical tail dependence coefficients for each copula family.

    Returns (lower_tail_lambda, upper_tail_lambda).
    """
    if family is None or np.isnan(param):
        return 0.0, 0.0

    if family == "gaussian":
        # Gaussian copula has NO tail dependence (for |rho| < 1)
        # But we report the asymptotic concentration for practical purposes
        return 0.0, 0.0

    elif family == "clayton":
        # Clayton: lambda_L = 2^(-1/theta), lambda_U = 0
        if param > 0:
            lower = 2.0 ** (-1.0 / param)
        else:
            lower = 0.0
        return float(lower), 0.0

    elif family == "gumbel":
        # Gumbel: lambda_L = 0, lambda_U = 2 - 2^(1/theta)
        if param > 1:
            upper = 2.0 - 2.0 ** (1.0 / param)
        else:
            upper = 0.0
        return 0.0, float(upper)

    elif family == "frank":
        # Frank: NO tail dependence (symmetric, light tails)
        return 0.0, 0.0

    return 0.0, 0.0


# ===========================================================================
# Empty result
# ===========================================================================

def _empty_result(n: int, reason: str = "unknown") -> dict:
    """Return empty result when computation cannot proceed."""
    nan = float("nan")
    inf = float("inf")
    return {
        "best_family": None,
        "best_aic": inf,
        "best_param": nan,
        "kendall_tau": nan,
        "spearman_rho": nan,
        "lower_tail_dependence": nan,
        "upper_tail_dependence": nan,
        "tail_asymmetry": nan,
        "tail_dependence_ratio": nan,
        "gaussian_param": nan,
        "gaussian_aic": inf,
        "clayton_param": nan,
        "clayton_aic": inf,
        "gumbel_param": nan,
        "gumbel_aic": inf,
        "frank_param": nan,
        "frank_aic": inf,
        "empirical_lower_tail": nan,
        "empirical_upper_tail": nan,
        "n_samples": int(n),
    }
