"""
ACF Decay Engine

Measures autocorrelation decay rate - how quickly a signal forgets its past.

CANONICAL INTERFACE:
    def compute(observations: pd.DataFrame) -> pd.DataFrame
    Input:  [entity_id, signal_id, I, y]
    Output: [entity_id, signal_id, acf_decay_rate, half_life, acf_1, acf_5]

Fast decay = weak memory, slow decay = strong memory/persistence.
"""

ENGINE_TYPE = 'signal'

import numpy as np
import pandas as pd
from typing import Dict, Any


def compute(
    observations: pd.DataFrame,
    max_lag: int = 50,
) -> pd.DataFrame:
    """
    Compute autocorrelation decay rate.

    CANONICAL INTERFACE:
        Input:  observations [entity_id, signal_id, I, y]
        Output: primitives [entity_id, signal_id, acf_decay_rate, half_life,
                           acf_1, acf_5]

    Parameters
    ----------
    observations : pd.DataFrame
        Must have columns: entity_id, signal_id, I, y
    max_lag : int, optional
        Maximum lag to compute ACF (default: 50)

    Returns
    -------
    pd.DataFrame
        ACF decay metrics per signal
    """
    results = []

    for (entity_id, signal_id), group in observations.groupby(['entity_id', 'signal_id']):
        y = group.sort_values('I')['y'].values

        if len(y) < max_lag + 10:
            results.append({
                'entity_id': entity_id,
                'signal_id': signal_id,
                'acf_decay_rate': np.nan,
                'half_life': np.nan,
                'acf_1': np.nan,
                'acf_5': np.nan,
            })
            continue

        try:
            result = _compute_acf_decay(y, max_lag)
            results.append({
                'entity_id': entity_id,
                'signal_id': signal_id,
                **result
            })
        except Exception:
            results.append({
                'entity_id': entity_id,
                'signal_id': signal_id,
                'acf_decay_rate': np.nan,
                'half_life': np.nan,
                'acf_1': np.nan,
                'acf_5': np.nan,
            })

    return pd.DataFrame(results)


def _compute_acf_decay(y: np.ndarray, max_lag: int) -> Dict[str, float]:
    """Compute ACF and fit exponential decay."""
    n = len(y)

    # Compute autocorrelation
    acf_values = _autocorrelation(y, max_lag)

    # Get specific ACF values
    acf_1 = acf_values[1] if len(acf_values) > 1 else np.nan
    acf_5 = acf_values[5] if len(acf_values) > 5 else np.nan

    # Fit exponential decay: acf(k) ~ exp(-k * decay_rate)
    # Linear regression on log(acf) for positive acf values
    positive_mask = acf_values > 0.01
    positive_lags = np.where(positive_mask)[0]

    if len(positive_lags) > 5:
        log_acf = np.log(acf_values[positive_mask])
        lags = positive_lags

        # Simple linear fit
        slope, intercept = np.polyfit(lags, log_acf, 1)
        decay_rate = -slope

        # Half-life: time for ACF to decay to 0.5
        if decay_rate > 0:
            half_life = np.log(2) / decay_rate
        else:
            half_life = np.inf
    else:
        decay_rate = np.nan
        half_life = np.nan

    return {
        'acf_decay_rate': float(decay_rate) if not np.isnan(decay_rate) else np.nan,
        'half_life': float(half_life) if not np.isinf(half_life) and not np.isnan(half_life) else np.nan,
        'acf_1': float(acf_1),
        'acf_5': float(acf_5),
    }


def _autocorrelation(y: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation function up to max_lag."""
    n = len(y)
    y_centered = y - np.mean(y)
    var = np.var(y)

    if var == 0:
        return np.zeros(max_lag + 1)

    acf = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        if k == 0:
            acf[k] = 1.0
        else:
            acf[k] = np.sum(y_centered[k:] * y_centered[:-k]) / ((n - k) * var)

    return acf
