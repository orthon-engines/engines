"""
Time Constant Engine.

Fits exponential decay/rise to estimate thermal time constant.
y(t) = y_final + (y_initial - y_final) * exp(-t/tau)
"""

import numpy as np
from typing import Dict


def compute(y: np.ndarray, I: np.ndarray = None) -> Dict[str, float]:
    """
    Estimate time constant from exponential fit.

    Args:
        y: Signal values (temperature, etc.)
        I: Time/index values (optional)

    Returns:
        dict with time_constant, equilibrium_value, fit_r2, is_decay
    """
    result = {
        'time_constant': np.nan,
        'equilibrium_value': np.nan,
        'fit_r2': np.nan,
        'is_decay': None
    }

    # Handle NaN values
    y = np.asarray(y).flatten()
    valid_mask = ~np.isnan(y)

    if I is not None:
        I = np.asarray(I).flatten()
        if len(I) == len(y):
            valid_mask &= ~np.isnan(I)
            I = I[valid_mask]
    else:
        I = None

    y = y[valid_mask]
    n = len(y)

    if n < 10:
        return result

    if I is None:
        I = np.arange(n, dtype=float)

    # Check if signal is changing enough to fit
    y_range = np.max(y) - np.min(y)
    if y_range < 1e-10:
        # Constant signal - no time constant
        result['equilibrium_value'] = float(y[0])
        result['fit_r2'] = 1.0
        result['time_constant'] = np.inf
        return result

    try:
        from scipy.optimize import curve_fit

        # Normalize time to start at 0
        t = I - I[0]

        # Determine if decay or rise
        y_start = np.mean(y[:max(1, n//10)])
        y_end = np.mean(y[-max(1, n//10):])
        is_decay = y_start > y_end

        # Initial guesses
        y0 = y[0]
        y_final = y[-1]
        tau_guess = max(1.0, (t[-1] - t[0]) / 3)

        def exp_func(t, y_inf, y_0, tau):
            return y_inf + (y_0 - y_inf) * np.exp(-t / np.maximum(tau, 1e-10))

        # Set bounds based on data range
        y_min = np.min(y) - y_range
        y_max = np.max(y) + y_range
        t_range = t[-1] - t[0]

        popt, pcov = curve_fit(
            exp_func, t, y,
            p0=[y_final, y0, tau_guess],
            bounds=([y_min, y_min, 1e-6],
                    [y_max, y_max, t_range * 10]),
            maxfev=2000
        )

        y_inf, y_0, tau = popt

        # Compute R^2
        y_pred = exp_func(t, y_inf, y_0, tau)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot > 1e-10:  # Prevent near-zero overflow
            r2 = 1 - ss_res / ss_tot
            if not np.isfinite(r2):  # Belt and suspenders
                r2 = np.nan
        else:
            r2 = np.nan  # Signal is essentially constant

        # Only accept fit if R^2 is reasonable
        if r2 >= 0.5:
            result = {
                'time_constant': float(abs(tau)),
                'equilibrium_value': float(y_inf),
                'fit_r2': float(r2),
                'is_decay': is_decay
            }
        else:
            # Poor fit - signal may not be exponential
            result['fit_r2'] = float(r2)
            result['is_decay'] = is_decay

    except ImportError:
        # scipy not available - use simple 63.2% method
        try:
            y_start = y[0]
            y_end = y[-1]
            target = y_start + 0.632 * (y_end - y_start)

            # Find time to reach 63.2% of final value
            idx = np.argmin(np.abs(y - target))
            tau = float(I[idx] - I[0])

            result['time_constant'] = tau if tau > 0 else np.nan
            result['equilibrium_value'] = float(y_end)
            result['is_decay'] = y_start > y_end
        except Exception:
            pass

    except Exception:
        pass

    return result
