"""
ADF Stat Engine.

Augmented Dickey-Fuller test for stationarity.
Uses config for min_samples threshold.
"""

import numpy as np
from typing import Dict
from engines.primitives.config import PRIMITIVES_CONFIG as cfg


def compute(y: np.ndarray) -> Dict[str, float]:
    """
    Compute Augmented Dickey-Fuller test statistic.

    Args:
        y: Signal values (1D array)

    Returns:
        dict with:
            adf_stat: Test statistic (more negative = more stationary)
            adf_pvalue: p-value (< 0.05 typically means stationary)
            adf_lags: Number of lags used in the test
            adf_nobs: Number of observations used

    Notes:
        H0 = unit root (non-stationary)
        p < 0.05 → reject H0 → stationary
    """
    result = {
        'adf_stat': np.nan,
        'adf_pvalue': np.nan,
        'adf_lags': np.nan,
        'adf_nobs': np.nan,
    }

    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    if n < cfg.min_samples.stationarity:
        return result

    # Check for constant signal
    if np.std(y) < 1e-10:
        # Constant signal is trivially stationary
        result['adf_stat'] = -np.inf
        result['adf_pvalue'] = 0.0
        result['adf_lags'] = 0.0
        result['adf_nobs'] = float(n)
        return result

    try:
        from statsmodels.tsa.stattools import adfuller
        adf_result = adfuller(y, autolag='AIC')
        # adf_result: (stat, pvalue, lags, nobs, critical_values, icbest)
        result['adf_stat'] = float(adf_result[0])
        result['adf_pvalue'] = float(adf_result[1])
        result['adf_lags'] = float(adf_result[2])
        result['adf_nobs'] = float(adf_result[3])
    except Exception:
        pass

    return result
