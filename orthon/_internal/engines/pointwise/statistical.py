"""
Statistical Engine
==================

Point-wise engine for statistical transformations.

Computes:
- Z-score normalization
- Rolling mean
- Rolling standard deviation
- Percentile rank
- Bollinger bands

These provide normalized views of the signal relative to
recent history - useful for detecting anomalies.
"""

import numpy as np
from typing import Tuple, Optional

from orthon._internal.core.signals.types import DenseSignal


# =============================================================================
# STATISTICAL ENGINE
# =============================================================================

class StatisticalEngine:
    """
    Point-wise statistical transformations engine.

    Usage:
        engine = StatisticalEngine(window=20)
        zscore, mean, std = engine.compute(timestamps, values)
    """

    def __init__(
        self,
        window: int = 20,
        min_periods: int = 1,
    ):
        """
        Initialize statistical engine.

        Parameters:
            window: Rolling window size
            min_periods: Minimum observations required
        """
        self.window = window
        self.min_periods = min_periods

    def compute(
        self,
        signal_id: str,
        timestamps: np.ndarray,
        values: np.ndarray,
    ) -> Tuple[DenseSignal, DenseSignal, DenseSignal]:
        """
        Compute z-score, rolling mean, and rolling std.

        Returns:
            (zscore, rolling_mean, rolling_std) as DenseSignal objects
        """
        mean_values = self._rolling_mean(values)
        std_values = self._rolling_std(values)

        # Z-score: (x - mean) / std
        zscore_values = (values - mean_values) / (std_values + 1e-10)

        zscore = DenseSignal(
            signal_id=f"{signal_id}_zscore",
            timestamps=timestamps,
            values=zscore_values,
            source_signal=signal_id,
            engine='statistical',
            parameters={'metric': 'zscore', 'window': self.window},
        )

        rolling_mean = DenseSignal(
            signal_id=f"{signal_id}_rolling_mean",
            timestamps=timestamps,
            values=mean_values,
            source_signal=signal_id,
            engine='statistical',
            parameters={'metric': 'rolling_mean', 'window': self.window},
        )

        rolling_std = DenseSignal(
            signal_id=f"{signal_id}_rolling_std",
            timestamps=timestamps,
            values=std_values,
            source_signal=signal_id,
            engine='statistical',
            parameters={'metric': 'rolling_std', 'window': self.window},
        )

        return zscore, rolling_mean, rolling_std

    def _rolling_mean(self, values: np.ndarray) -> np.ndarray:
        """Compute rolling mean."""
        result = np.full_like(values, np.nan)

        for i in range(len(values)):
            start = max(0, i - self.window + 1)
            window_vals = values[start:i + 1]
            if len(window_vals) >= self.min_periods:
                result[i] = np.nanmean(window_vals)

        return result

    def _rolling_std(self, values: np.ndarray) -> np.ndarray:
        """Compute rolling standard deviation."""
        result = np.full_like(values, np.nan)

        for i in range(len(values)):
            start = max(0, i - self.window + 1)
            window_vals = values[start:i + 1]
            if len(window_vals) >= max(2, self.min_periods):
                result[i] = np.nanstd(window_vals, ddof=1)

        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_zscore(
    signal_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    window: int = 20,
) -> DenseSignal:
    """Compute rolling z-score."""
    mean_vals = _rolling_mean(values, window)
    std_vals = _rolling_std(values, window)
    zscore_values = (values - mean_vals) / (std_vals + 1e-10)

    return DenseSignal(
        signal_id=f"{signal_id}_zscore",
        timestamps=timestamps,
        values=zscore_values,
        source_signal=signal_id,
        engine='statistical',
        parameters={'metric': 'zscore', 'window': window},
    )


def compute_rolling_mean(
    signal_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    window: int = 20,
) -> DenseSignal:
    """Compute rolling mean."""
    mean_values = _rolling_mean(values, window)

    return DenseSignal(
        signal_id=f"{signal_id}_rolling_mean",
        timestamps=timestamps,
        values=mean_values,
        source_signal=signal_id,
        engine='statistical',
        parameters={'metric': 'rolling_mean', 'window': window},
    )


def compute_rolling_std(
    signal_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    window: int = 20,
) -> DenseSignal:
    """Compute rolling standard deviation."""
    std_values = _rolling_std(values, window)

    return DenseSignal(
        signal_id=f"{signal_id}_rolling_std",
        timestamps=timestamps,
        values=std_values,
        source_signal=signal_id,
        engine='statistical',
        parameters={'metric': 'rolling_std', 'window': window},
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling mean efficiently."""
    result = np.full_like(values, np.nan)

    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start:i + 1]
        if len(window_vals) >= 1:
            result[i] = np.nanmean(window_vals)

    return result


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling standard deviation."""
    result = np.full_like(values, np.nan)

    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start:i + 1]
        if len(window_vals) >= 2:
            result[i] = np.nanstd(window_vals, ddof=1)

    return result


# =============================================================================
# EXTENDED STATISTICS
# =============================================================================

def compute_percentile_rank(
    signal_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    window: int = 100,
) -> DenseSignal:
    """
    Compute rolling percentile rank.

    At each point, what percentile is this value in the
    rolling window? 0 = lowest, 100 = highest.
    """
    result = np.full_like(values, np.nan)

    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start:i + 1]
        if len(window_vals) >= 2:
            rank = np.sum(window_vals <= values[i]) / len(window_vals)
            result[i] = rank * 100

    return DenseSignal(
        signal_id=f"{signal_id}_percentile",
        timestamps=timestamps,
        values=result,
        source_signal=signal_id,
        engine='statistical',
        parameters={'metric': 'percentile_rank', 'window': window},
    )


def compute_bollinger_bands(
    signal_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    window: int = 20,
    num_std: float = 2.0,
) -> Tuple[DenseSignal, DenseSignal, DenseSignal]:
    """
    Compute Bollinger Bands.

    Returns:
        (upper_band, middle_band, lower_band)
    """
    mean_vals = _rolling_mean(values, window)
    std_vals = _rolling_std(values, window)

    upper = mean_vals + num_std * std_vals
    lower = mean_vals - num_std * std_vals

    upper_band = DenseSignal(
        signal_id=f"{signal_id}_bb_upper",
        timestamps=timestamps,
        values=upper,
        source_signal=signal_id,
        engine='statistical',
        parameters={'metric': 'bollinger_upper', 'window': window, 'num_std': num_std},
    )

    middle_band = DenseSignal(
        signal_id=f"{signal_id}_bb_middle",
        timestamps=timestamps,
        values=mean_vals,
        source_signal=signal_id,
        engine='statistical',
        parameters={'metric': 'bollinger_middle', 'window': window},
    )

    lower_band = DenseSignal(
        signal_id=f"{signal_id}_bb_lower",
        timestamps=timestamps,
        values=lower,
        source_signal=signal_id,
        engine='statistical',
        parameters={'metric': 'bollinger_lower', 'window': window, 'num_std': num_std},
    )

    return upper_band, middle_band, lower_band


def compute_bollinger_pct(
    signal_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    window: int = 20,
    num_std: float = 2.0,
) -> DenseSignal:
    """
    Compute Bollinger %B.

    Position of value within Bollinger bands.
    0 = at lower band, 0.5 = at middle, 1 = at upper band.
    """
    mean_vals = _rolling_mean(values, window)
    std_vals = _rolling_std(values, window)

    upper = mean_vals + num_std * std_vals
    lower = mean_vals - num_std * std_vals

    pct_b = (values - lower) / (upper - lower + 1e-10)

    return DenseSignal(
        signal_id=f"{signal_id}_bb_pct",
        timestamps=timestamps,
        values=pct_b,
        source_signal=signal_id,
        engine='statistical',
        parameters={'metric': 'bollinger_pct', 'window': window, 'num_std': num_std},
    )


def compute_rolling_skew(
    signal_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    window: int = 30,
) -> DenseSignal:
    """
    Compute rolling skewness.

    Positive skew = right tail, Negative skew = left tail.
    """
    from scipy.stats import skew

    result = np.full_like(values, np.nan)

    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start:i + 1]
        if len(window_vals) >= 3:
            result[i] = skew(window_vals, nan_policy='omit')

    return DenseSignal(
        signal_id=f"{signal_id}_rolling_skew",
        timestamps=timestamps,
        values=result,
        source_signal=signal_id,
        engine='statistical',
        parameters={'metric': 'rolling_skew', 'window': window},
    )


def compute_rolling_kurtosis(
    signal_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    window: int = 30,
) -> DenseSignal:
    """
    Compute rolling kurtosis.

    Excess kurtosis: 0 = normal, positive = heavy tails.
    """
    from scipy.stats import kurtosis

    result = np.full_like(values, np.nan)

    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start:i + 1]
        if len(window_vals) >= 4:
            result[i] = kurtosis(window_vals, nan_policy='omit')

    return DenseSignal(
        signal_id=f"{signal_id}_rolling_kurtosis",
        timestamps=timestamps,
        values=result,
        source_signal=signal_id,
        engine='statistical',
        parameters={'metric': 'rolling_kurtosis', 'window': window},
    )
