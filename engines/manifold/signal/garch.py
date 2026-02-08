"""
GARCH Engine.

Computes GARCH(1,1) volatility model parameters.

GARCH(1,1): σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}
- ω (omega): Long-term variance weight
- α (alpha): Shock persistence (how much yesterday's shock affects today)
- β (beta): Volatility persistence (how much yesterday's volatility persists)
- α + β: Total persistence (should be < 1 for stationarity)
"""

import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    """
    Compute GARCH(1,1) parameters of signal.

    Args:
        y: Signal values

    Returns:
        dict with:
            - 'garch_omega': Long-term variance weight
            - 'garch_alpha': Shock persistence (ARCH effect)
            - 'garch_beta': Volatility persistence (GARCH effect)
            - 'garch_persistence': α + β (total persistence)
    """
    result = {
        'garch_omega': np.nan,
        'garch_alpha': np.nan,
        'garch_beta': np.nan,
        'garch_persistence': np.nan
    }

    if len(y) < 100:
        return result

    returns = np.diff(y)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 50:
        return result

    try:
        # Try arch library first (most accurate)
        from arch import arch_model

        # Scale returns for numerical stability
        scale = np.std(returns)
        if scale < 1e-10:
            return result

        scaled_returns = returns / scale * 100

        model = arch_model(scaled_returns, vol='Garch', p=1, q=1, rescale=False)
        fit = model.fit(disp='off', show_warning=False)

        result['garch_omega'] = float(fit.params.get('omega', np.nan))
        result['garch_alpha'] = float(fit.params.get('alpha[1]', np.nan))
        result['garch_beta'] = float(fit.params.get('beta[1]', np.nan))

        if not np.isnan(result['garch_alpha']) and not np.isnan(result['garch_beta']):
            result['garch_persistence'] = result['garch_alpha'] + result['garch_beta']

    except ImportError:
        # Fallback: moment-based estimation
        result = _estimate_garch_fallback(returns)
    except Exception:
        # Fallback on any arch error
        result = _estimate_garch_fallback(returns)

    return result


def _estimate_garch_fallback(returns: np.ndarray) -> Dict[str, float]:
    """
    Moment-based GARCH(1,1) estimation (fallback).

    Uses autocorrelation of squared returns to estimate parameters.
    More robust than single-lag method.
    """
    result = {
        'garch_omega': np.nan,
        'garch_alpha': np.nan,
        'garch_beta': np.nan,
        'garch_persistence': np.nan
    }

    n = len(returns)
    if n < 30:
        return result

    # Compute squared returns (proxy for volatility)
    sq_returns = returns ** 2
    var = np.var(returns)

    if var < 1e-15:
        return result

    # Use multiple lags to estimate persistence more robustly
    autocorrs = []
    max_lag = min(10, n // 10)

    for lag in range(1, max_lag + 1):
        if n - lag > 10:
            r = np.corrcoef(sq_returns[:-lag], sq_returns[lag:])[0, 1]
            if not np.isnan(r):
                autocorrs.append(abs(r))

    if autocorrs:
        # Average autocorrelation gives estimate of persistence
        avg_autocorr = np.mean(autocorrs)

        # First-lag autocorr estimates alpha + beta*rho
        first_autocorr = autocorrs[0] if autocorrs else 0.1

        # Second-lag gives more info about beta
        if len(autocorrs) > 1:
            second_autocorr = autocorrs[1]
            # Under GARCH(1,1): rho_2/rho_1 ≈ beta
            beta_estimate = second_autocorr / (first_autocorr + 1e-10)
            beta_estimate = max(0.5, min(0.95, beta_estimate))
        else:
            beta_estimate = 0.85

        # Estimate alpha from first autocorrelation
        # rho_1 = (alpha + beta^2*rho_1 + ...) for GARCH
        # Simplified: alpha ≈ rho_1 * (1 - beta)
        alpha_estimate = first_autocorr * (1 - beta_estimate * 0.8)
        alpha_estimate = max(0.02, min(0.20, alpha_estimate))

    else:
        # Default values if no autocorrelation available
        alpha_estimate = 0.05
        beta_estimate = 0.85

    # Ensure stationarity: alpha + beta < 1
    if alpha_estimate + beta_estimate >= 1.0:
        total = alpha_estimate + beta_estimate
        alpha_estimate = alpha_estimate / total * 0.95
        beta_estimate = beta_estimate / total * 0.95

    # Omega from unconditional variance: σ² = ω / (1 - α - β)
    persistence = alpha_estimate + beta_estimate
    omega = var * (1 - persistence) if persistence < 1 else var * 0.05

    result['garch_omega'] = float(max(1e-10, omega))
    result['garch_alpha'] = float(alpha_estimate)
    result['garch_beta'] = float(beta_estimate)
    result['garch_persistence'] = float(persistence)

    return result
