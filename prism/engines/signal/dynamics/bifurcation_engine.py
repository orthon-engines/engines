"""
Bifurcation Engine

Detects critical slowing down indicators that precede bifurcations
(regime changes, tipping points). Tracks variance, autocorrelation,
and recovery rate trends.

Key insight: Rising variance + rising autocorrelation = approaching tipping point.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from ..primitives.individual import autocorrelation
from ..primitives.tests import mann_kendall


@dataclass
class BifurcationConfig:
    """Configuration for bifurcation engine."""
    dt: float = 1.0  # Time step in seconds
    detrend: bool = True  # Detrend before analysis
    detrend_window: int = 50  # Window for moving average detrending
    min_samples: int = 30
    csd_warning_threshold: float = 2.0  # CSD score threshold for warning


class BifurcationEngine:
    """
    Critical Slowing Down Detection Engine.

    Detects early warning signals that precede bifurcations/tipping points.

    Outputs:
    - variance: Signal variance (increases before bifurcation)
    - autocorr_lag1: Lag-1 autocorrelation (increases before bifurcation)
    - skewness: Distribution asymmetry
    - kurtosis: Distribution tail heaviness
    - variance_trend: Mann-Kendall trend in variance
    - autocorr_trend: Mann-Kendall trend in autocorrelation
    - csd_score: Composite critical slowing down score
    - approaching_bifurcation: True if strong CSD signal
    """

    ENGINE_TYPE = "dynamics"

    def __init__(self, config: Optional[BifurcationConfig] = None):
        self.config = config or BifurcationConfig()
        self.variance_history: Dict[str, List[float]] = {}
        self.autocorr_history: Dict[str, List[float]] = {}

    def compute(
        self,
        signal: np.ndarray,
        unit_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Compute critical slowing down indicators for a signal.

        Parameters
        ----------
        signal : np.ndarray
            Time series data
        unit_id : str
            Entity identifier

        Returns
        -------
        dict
            CSD metrics
        """
        signal = np.asarray(signal).flatten()

        # Remove NaN
        signal = signal[~np.isnan(signal)]

        if len(signal) < self.config.min_samples:
            return self._empty_result(unit_id)

        try:
            # Detrend if requested
            if self.config.detrend and len(signal) > self.config.detrend_window:
                kernel = np.ones(self.config.detrend_window) / self.config.detrend_window
                trend = np.convolve(signal, kernel, mode='same')
                sig_detrended = signal - trend
            else:
                sig_detrended = signal - np.mean(signal)

            # Basic statistics
            var = float(np.var(sig_detrended))
            sig_std = np.std(sig_detrended)

            # Autocorrelation
            ac = autocorrelation(sig_detrended)  # Returns all lags when no lag specified
            ac1 = float(ac[1]) if len(ac) > 1 else 0.0
            ac5 = float(ac[5]) if len(ac) > 5 else 0.0

            # Higher moments
            if sig_std > 0:
                centered = sig_detrended - np.mean(sig_detrended)
                skew = float(np.mean((centered / sig_std) ** 3))
                kurt = float(np.mean((centered / sig_std) ** 4) - 3)
            else:
                skew, kurt = 0.0, 0.0

            # Track histories
            if unit_id not in self.variance_history:
                self.variance_history[unit_id] = []
                self.autocorr_history[unit_id] = []

            self.variance_history[unit_id].append(var)
            self.autocorr_history[unit_id].append(ac1)

            # Trend detection
            var_history = self.variance_history[unit_id]
            ac_history = self.autocorr_history[unit_id]

            if len(var_history) >= 4:
                var_trend, var_p, _, var_slope = mann_kendall(np.array(var_history))
                ac_trend, ac_p, _, ac_slope = mann_kendall(np.array(ac_history))
            else:
                var_trend, var_p, var_slope = 'no trend', 1.0, 0.0
                ac_trend, ac_p, ac_slope = 'no trend', 1.0, 0.0

            # Critical Slowing Down composite score
            # Both variance AND autocorrelation increasing = strong CSD signal
            csd_score = 0.0

            if var_trend == 'increasing' and var_p < 0.1:
                csd_score += 1.0
            if ac_trend == 'increasing' and ac_p < 0.1:
                csd_score += 1.0
            if var_slope > 0:
                csd_score += min(1.0, abs(var_slope) * 10)
            if ac_slope > 0:
                csd_score += min(1.0, abs(ac_slope) * 10)

            # Approaching bifurcation if both indicators trending up significantly
            approaching = (
                var_trend == 'increasing' and var_p < 0.05 and
                ac_trend == 'increasing' and ac_p < 0.05
            )

            # Status
            if approaching:
                csd_status = 'CRITICAL'
            elif csd_score > self.config.csd_warning_threshold:
                csd_status = 'WARNING'
            elif csd_score > 1.0:
                csd_status = 'ELEVATED'
            else:
                csd_status = 'NORMAL'

            return {
                'unit_id': unit_id,
                'n_samples': len(signal),
                'variance': var,
                'autocorr_lag1': ac1,
                'autocorr_lag5': ac5,
                'skewness': skew,
                'kurtosis': kurt,
                'variance_trend': var_trend,
                'variance_trend_p': float(var_p),
                'variance_trend_slope': float(var_slope),
                'autocorr_trend': ac_trend,
                'autocorr_trend_p': float(ac_p),
                'autocorr_trend_slope': float(ac_slope),
                'csd_score': float(csd_score),
                'approaching_bifurcation': approaching,
                'csd_status': csd_status,
            }

        except Exception as e:
            return self._empty_result(unit_id)

    def _empty_result(self, unit_id: str) -> Dict[str, Any]:
        """Return empty result for insufficient data."""
        return {
            'unit_id': unit_id,
            'n_samples': 0,
            'variance': np.nan,
            'autocorr_lag1': np.nan,
            'autocorr_lag5': np.nan,
            'skewness': np.nan,
            'kurtosis': np.nan,
            'variance_trend': 'unknown',
            'variance_trend_p': np.nan,
            'variance_trend_slope': np.nan,
            'autocorr_trend': 'unknown',
            'autocorr_trend_p': np.nan,
            'autocorr_trend_slope': np.nan,
            'csd_score': np.nan,
            'approaching_bifurcation': False,
            'csd_status': 'unknown',
        }

    def to_parquet_row(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert result to flat row for parquet output."""
        return {k: v for k, v in result.items()}


def run_bifurcation_engine(
    observations: pl.DataFrame,
    config: BifurcationConfig,
    signal_columns: List[str],
    output_path: Optional[Path] = None
) -> pl.DataFrame:
    """
    Run bifurcation engine on observations DataFrame.

    Parameters
    ----------
    observations : pl.DataFrame
        Observations with unit_id, signal_id, index, value
    config : BifurcationConfig
        Engine configuration
    signal_columns : list
        Signals to analyze
    output_path : Path, optional
        Path to write output parquet

    Returns
    -------
    pl.DataFrame
        CSD results
    """
    entities = observations.select('unit_id').unique().to_series().to_list()
    all_results = []

    for sig_col in signal_columns:
        engine = BifurcationEngine(config)  # Fresh engine per signal

        for unit_id in entities:
            entity_obs = observations.filter(pl.col('unit_id') == unit_id)

            sig_data = (
                entity_obs
                .filter(pl.col('signal_id') == sig_col)
                .sort('index')
                .select('value')
                .to_series()
                .to_numpy()
            )

            if len(sig_data) > 0:
                result = engine.compute(sig_data, unit_id)
                row = engine.to_parquet_row(result)
                row['signal'] = sig_col
                all_results.append(row)

    df = pl.DataFrame(all_results) if all_results else pl.DataFrame({
        'unit_id': [], 'signal': [], 'csd_score': [], 'csd_status': []
    })

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path)

    return df
