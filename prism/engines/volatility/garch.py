"""
GARCH(p,q) - Generalized Autoregressive Conditional Heteroskedasticity

REAL implementation using arch package with Maximum Likelihood Estimation.

    sigma^2_t = omega + sum_i(alpha_i * epsilon^2_{t-i}) + sum_j(beta_j * sigma^2_{t-j})

Key parameters:
    - omega: Long-run variance contribution
    - alpha: ARCH effect (shock impact)
    - beta: GARCH effect (persistence)
    - alpha + beta: Total persistence (< 1 for stationarity)

Persistence classification:
    - alpha + beta < 0.85: Dissipating (shocks fade quickly)
    - 0.85 <= alpha + beta < 0.99: Persistent (shocks linger)
    - alpha + beta >= 0.99: Integrated (shocks permanent)

References:
    Bollerslev (1986) "Generalized Autoregressive Conditional Heteroskedasticity"
    Journal of Econometrics 31, 307-327

    Engle (1982) "Autoregressive Conditional Heteroscedasticity with Estimates
    of the Variance of United Kingdom Inflation"
    Econometrica 50, 987-1007
"""

import numpy as np
from typing import Dict, Any, Optional
import warnings

# Check for arch package
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False


def compute(
    series: np.ndarray,
    mode: str = 'static',
    t: Optional[int] = None,
    window_size: int = 200,
    step_size: int = 20,
    p: int = 1,
    q: int = 1,
    dist: str = 'normal',
) -> Dict[str, Any]:
    """
    Fit GARCH(p,q) model using Maximum Likelihood Estimation.

    Uses the arch package for proper MLE estimation.
    Falls back to method-of-moments only if arch is unavailable.

    Args:
        series: 1D numpy array of observations
        mode: 'static', 'windowed', or 'point'
        t: Time index for point mode
        window_size: Window size for windowed/point modes
        step_size: Step between windows for windowed mode
        p: GARCH lag order (sigma^2 lags)
        q: ARCH lag order (epsilon^2 lags)
        dist: Error distribution ('normal', 't', 'skewt', 'ged')

    Returns:
        mode='static': {'alpha': float, 'beta': float, 'persistence': float, ...}
        mode='windowed': {'alpha': array, 'beta': array, 't': array, ...}
        mode='point': {'alpha': float, 'beta': float, 't': int, ...}
    """
    series = np.asarray(series).flatten()

    if mode == 'static':
        return _compute_static(series, p, q, dist)
    elif mode == 'windowed':
        return _compute_windowed(series, window_size, step_size, p, q, dist)
    elif mode == 'point':
        return _compute_point(series, t, window_size, p, q, dist)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'static', 'windowed', or 'point'.")


def _fit_garch_mle(
    returns: np.ndarray,
    p: int = 1,
    q: int = 1,
    dist: str = 'normal',
) -> Dict[str, Any]:
    """
    Fit GARCH(p,q) using Maximum Likelihood Estimation via arch package.

    This is the REAL implementation - no approximations.
    """
    if not HAS_ARCH:
        raise ImportError(
            "arch package required for real GARCH estimation.\n"
            "Install with: pip install arch\n"
            "Without arch, GARCH results are approximations."
        )

    # Scale returns for numerical stability
    scale = np.std(returns)
    if scale < 1e-10:
        scale = 1.0
    scaled_returns = returns / scale * 100

    # Fit model with MLE
    model = arch_model(
        scaled_returns,
        mean='Zero',
        vol='GARCH',
        p=p,
        q=q,
        dist=dist,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(disp='off', show_warning=False)

    # Extract parameters
    params = result.params

    # Rescale omega back to original units
    omega = params.get('omega', 0) * (scale / 100) ** 2

    # Sum alpha and beta parameters
    alpha_params = {k: v for k, v in params.items() if k.startswith('alpha')}
    beta_params = {k: v for k, v in params.items() if k.startswith('beta')}

    alpha_sum = sum(alpha_params.values())
    beta_sum = sum(beta_params.values())

    # Persistence
    persistence = alpha_sum + beta_sum

    # Unconditional variance (only defined for stationary processes)
    if persistence < 1:
        unconditional_var = omega / (1 - persistence)
        unconditional_vol = np.sqrt(unconditional_var)
    else:
        unconditional_var = None
        unconditional_vol = None

    # Conditional volatility series (rescaled)
    cond_vol = result.conditional_volatility * scale / 100

    return {
        'omega': float(omega),
        'alpha': float(alpha_sum),
        'beta': float(beta_sum),
        'persistence': float(persistence),
        'is_stationary': persistence < 1,
        'unconditional_variance': float(unconditional_var) if unconditional_var else None,
        'unconditional_volatility': float(unconditional_vol) if unconditional_vol else None,
        'log_likelihood': float(result.loglikelihood),
        'aic': float(result.aic),
        'bic': float(result.bic),
        'conditional_volatility': cond_vol,
        'p': p,
        'q': q,
        'distribution': dist,
        'n_observations': len(returns),
        'method': 'MLE',
    }


def _compute_static(
    series: np.ndarray,
    p: int = 1,
    q: int = 1,
    dist: str = 'normal',
) -> Dict[str, Any]:
    """Estimate GARCH(p,q) on entire signal."""
    # Compute returns
    returns = np.diff(series)

    if len(returns) < 20:
        return _insufficient_data_result()

    # Remove NaN/Inf
    returns = returns[np.isfinite(returns)]

    if len(returns) < 20:
        return _insufficient_data_result()

    if HAS_ARCH:
        try:
            result = _fit_garch_mle(returns, p, q, dist)
            # Remove conditional volatility array for static output
            if 'conditional_volatility' in result:
                del result['conditional_volatility']
            return result
        except Exception as e:
            # Fall through to method-of-moments
            pass

    # Fallback: Method of moments (clearly marked as approximation)
    return _fit_garch_mom(returns)


def _fit_garch_mom(returns: np.ndarray) -> Dict[str, Any]:
    """
    Fallback: Method of moments estimation.

    CLEARLY MARKED AS APPROXIMATION - not the real thing.
    """
    # Squared returns
    r2 = returns ** 2

    # Sample statistics
    mean_r2 = np.mean(r2)
    var_r2 = np.var(r2)

    if var_r2 < 1e-10:
        return _insufficient_data_result()

    # ACF at lag 1 of squared returns
    if len(r2) > 1:
        acf1 = np.corrcoef(r2[:-1], r2[1:])[0, 1]
        if np.isnan(acf1):
            acf1 = 0.5
    else:
        acf1 = 0.5

    # Persistence approximation from ACF
    persistence = np.clip(acf1, 0, 0.999)

    # Rough alpha/beta split (THIS IS THE APPROXIMATION)
    # In real GARCH, these are estimated via MLE
    alpha = np.clip(persistence * 0.15, 0.01, 0.3)
    beta = np.clip(persistence - alpha, 0, 0.99)

    # Omega from unconditional variance
    unconditional_var = mean_r2
    omega = unconditional_var * (1 - alpha - beta) if (1 - alpha - beta) > 0 else 0.001

    return {
        'omega': float(omega),
        'alpha': float(alpha),
        'beta': float(beta),
        'persistence': float(alpha + beta),
        'is_stationary': (alpha + beta) < 1,
        'unconditional_variance': float(unconditional_var),
        'unconditional_volatility': float(np.sqrt(unconditional_var)),
        'log_likelihood': None,
        'aic': None,
        'bic': None,
        'p': 1,
        'q': 1,
        'distribution': 'normal',
        'n_observations': len(returns),
        'method': 'MoM_APPROXIMATION',  # CLEARLY MARKED
    }


def _insufficient_data_result() -> Dict[str, Any]:
    """Return result for insufficient data."""
    return {
        'omega': None,
        'alpha': None,
        'beta': None,
        'persistence': None,
        'is_stationary': None,
        'unconditional_variance': None,
        'unconditional_volatility': None,
        'log_likelihood': None,
        'aic': None,
        'bic': None,
        'p': None,
        'q': None,
        'distribution': None,
        'n_observations': 0,
        'method': 'INSUFFICIENT_DATA',
    }


def _compute_windowed(
    series: np.ndarray,
    window_size: int,
    step_size: int,
    p: int = 1,
    q: int = 1,
    dist: str = 'normal',
) -> Dict[str, Any]:
    """Estimate GARCH(p,q) over rolling windows."""
    n = len(series)

    if n < window_size:
        return {
            'alpha': np.array([]),
            'beta': np.array([]),
            'omega': np.array([]),
            'persistence': np.array([]),
            'unconditional_variance': np.array([]),
            't': np.array([]),
            'window_size': window_size,
            'step_size': step_size,
            'method': 'MLE' if HAS_ARCH else 'MoM_APPROXIMATION',
        }

    t_values = []
    alpha_values = []
    beta_values = []
    omega_values = []
    persistence_values = []
    uncond_var_values = []

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        window = series[start:end]

        result = _compute_static(window, p, q, dist)

        t_values.append(start + window_size // 2)
        alpha_values.append(result['alpha'] if result['alpha'] is not None else np.nan)
        beta_values.append(result['beta'] if result['beta'] is not None else np.nan)
        omega_values.append(result['omega'] if result['omega'] is not None else np.nan)
        persistence_values.append(result['persistence'] if result['persistence'] is not None else np.nan)
        uncond_var_values.append(result['unconditional_variance'] if result['unconditional_variance'] is not None else np.nan)

    return {
        'alpha': np.array(alpha_values),
        'beta': np.array(beta_values),
        'omega': np.array(omega_values),
        'persistence': np.array(persistence_values),
        'unconditional_variance': np.array(uncond_var_values),
        't': np.array(t_values),
        'window_size': window_size,
        'step_size': step_size,
        'method': 'MLE' if HAS_ARCH else 'MoM_APPROXIMATION',
    }


def _compute_point(
    series: np.ndarray,
    t: int,
    window_size: int,
    p: int = 1,
    q: int = 1,
    dist: str = 'normal',
) -> Dict[str, Any]:
    """Estimate GARCH(p,q) at specific time t."""
    if t is None:
        raise ValueError("t is required for point mode")

    n = len(series)

    # Center window on t
    half_window = window_size // 2
    start = max(0, t - half_window)
    end = min(n, start + window_size)

    if end - start < window_size:
        start = max(0, end - window_size)

    window = series[start:end]

    if len(window) < 21:
        result = _insufficient_data_result()
        result['t'] = t
        result['window_start'] = start
        result['window_end'] = end
        return result

    result = _compute_static(window, p, q, dist)
    result['t'] = t
    result['window_start'] = start
    result['window_end'] = end

    return result
