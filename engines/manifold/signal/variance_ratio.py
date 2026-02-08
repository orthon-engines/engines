"""
Variance Ratio Engine.

Tests whether variance scales linearly with time (random walk)
or sub/super-linearly (mean-reverting/trending).
"""

import numpy as np
from typing import Dict


def compute(y: np.ndarray, periods: list = None) -> Dict[str, float]:
    """
    Compute variance ratio test.

    Args:
        y: Signal values (1D array)
        periods: List of periods to test (default: [2, 4, 8])

    Returns:
        dict with:
            variance_ratio: Primary ratio (period=2 vs period=1)
            variance_ratio_4: Ratio at period=4
            variance_ratio_8: Ratio at period=8
            variance_ratio_stat: Z-statistic for primary ratio

    Interpretation:
        ratio ~ 1.0: Random walk (variance scales linearly)
        ratio < 1.0: Mean-reverting (negative autocorrelation)
        ratio > 1.0: Trending/persistent (positive autocorrelation)
    """
    result = {
        'variance_ratio': np.nan,
        'variance_ratio_4': np.nan,
        'variance_ratio_8': np.nan,
        'variance_ratio_stat': np.nan,
    }

    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    if n < 20:
        return result

    # Check for constant signal
    if np.std(y) < 1e-10:
        result['variance_ratio'] = 1.0
        result['variance_ratio_4'] = 1.0
        result['variance_ratio_8'] = 1.0
        result['variance_ratio_stat'] = 0.0
        return result

    if periods is None:
        periods = [2, 4, 8]

    try:
        # Compute returns/differences
        returns = np.diff(y)

        if len(returns) < 16:
            return result

        # Variance of 1-period returns
        var_1 = np.var(returns, ddof=1)

        if var_1 < 1e-10:
            return result

        # Compute variance ratios for different periods
        ratios = {}
        for q in periods:
            if len(y) < q + 1:
                continue

            # q-period returns
            returns_q = y[q:] - y[:-q]
            var_q = np.var(returns_q, ddof=1)

            # Under random walk: Var(q-period) = q * Var(1-period)
            # So ratio should be 1.0
            ratio = var_q / (q * var_1)
            ratios[q] = ratio

        if 2 in ratios:
            result['variance_ratio'] = float(ratios[2])

            # Asymptotic z-statistic for VR(2)
            # Under null (random walk), VR ~ N(1, 2/n)
            z_stat = (ratios[2] - 1.0) / np.sqrt(2.0 / n)
            result['variance_ratio_stat'] = float(z_stat)

        if 4 in ratios:
            result['variance_ratio_4'] = float(ratios[4])

        if 8 in ratios:
            result['variance_ratio_8'] = float(ratios[8])

    except Exception:
        pass

    return result
