"""
PRISM Transfer Detector Engine
==============================

Detects cross-cohort transmission patterns in structural dynamics.

This engine identifies how structural changes (energy, tension, phase shifts)
propagate between cohorts over time.

Types of Transfer:
    - contagion: Shock in one cohort rapidly spreads to others
    - spillover: Gradual transmission of structural changes
    - co-movement: Simultaneous changes without clear lead/lag

Input: Cohort-level energy/tension signal topology
Output: Transfer metrics between cohort pairs

Key Metrics:
    - transfer_strength: Magnitude of transmission
    - transfer_lag: Days for transmission
    - transfer_direction: 'energy', 'tension', 'both'
    - causality evidence (Granger-like, cross-correlation)

Usage:
    from prism.engines.state.transfer_detector import TransferDetectorEngine

    engine = TransferDetectorEngine()
    result = engine.run(cohort_a_series, cohort_b_series)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from scipy import stats
from scipy.signal import correlate


@dataclass
class TransferResult:
    """Result from transfer detection."""
    cohort_from: str
    cohort_to: str
    transfer_strength: float
    transfer_lag: int
    transfer_direction: str  # 'energy', 'tension', 'both'
    correlation: float
    correlation_lag: int
    granger_fstat: Optional[float]
    granger_pvalue: Optional[float]
    is_significant: bool
    transfer_type: str  # 'contagion', 'spillover', 'co-movement'


class TransferDetectorEngine:
    """
    Detect cross-cohort transmission patterns.

    Uses multiple methods to identify transmission:
    1. Cross-correlation with lag analysis
    2. Granger-like causality tests
    3. Lead-lag relationship detection

    Classifies transfers as:
    - contagion: Strong, rapid (<5 day lag), significant
    - spillover: Moderate, slow (>5 day lag), significant
    - co-movement: Simultaneous (0-2 day lag), moderate correlation
    """

    def __init__(self, max_lag: int = 21, significance_threshold: float = 0.05):
        """
        Initialize detector.

        Args:
            max_lag: Maximum lag to test (days)
            significance_threshold: P-value threshold for significance
        """
        self.max_lag = max_lag
        self.significance_threshold = significance_threshold

    def run(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        cohort_a: str = 'A',
        cohort_b: str = 'B'
    ) -> TransferResult:
        """
        Detect transfer between two cohort signal topology.

        Args:
            series_a: Energy/tension series for cohort A
            series_b: Energy/tension series for cohort B
            cohort_a: Name of cohort A
            cohort_b: Name of cohort B

        Returns:
            TransferResult with transfer metrics
        """
        # Align series
        combined = pd.concat([series_a, series_b], axis=1, keys=['a', 'b'])
        combined = combined.dropna()

        if len(combined) < self.max_lag + 5:
            return self._empty_result(cohort_a, cohort_b)

        a_vals = combined['a'].values
        b_vals = combined['b'].values

        # Cross-correlation analysis
        corr, corr_lag = self._cross_correlation(a_vals, b_vals)

        # Granger-like causality (simple version)
        granger_fstat, granger_pvalue = self._simple_granger(a_vals, b_vals)

        # Determine transfer direction and strength
        direction, strength, from_to = self._determine_transfer(
            corr, corr_lag, granger_fstat, granger_pvalue
        )

        # Classify transfer type
        is_significant = granger_pvalue is not None and granger_pvalue < self.significance_threshold
        transfer_type = self._classify_transfer(corr_lag, strength, is_significant)

        # Determine which cohort leads
        if from_to == 'a_to_b':
            cohort_from, cohort_to = cohort_a, cohort_b
            transfer_lag = abs(corr_lag)
        else:
            cohort_from, cohort_to = cohort_b, cohort_a
            transfer_lag = abs(corr_lag)

        return TransferResult(
            cohort_from=cohort_from,
            cohort_to=cohort_to,
            transfer_strength=strength,
            transfer_lag=transfer_lag,
            transfer_direction=direction,
            correlation=corr,
            correlation_lag=corr_lag,
            granger_fstat=granger_fstat,
            granger_pvalue=granger_pvalue,
            is_significant=is_significant,
            transfer_type=transfer_type
        )

    def _cross_correlation(self, a: np.ndarray, b: np.ndarray) -> Tuple[float, int]:
        """
        Compute cross-correlation and find optimal lag.

        Returns (max_correlation, optimal_lag)
        Positive lag means A leads B.
        """
        # Normalize
        a_norm = (a - np.mean(a)) / (np.std(a) + 1e-9)
        b_norm = (b - np.mean(b)) / (np.std(b) + 1e-9)

        # Full cross-correlation
        xcorr = correlate(a_norm, b_norm, mode='full')
        xcorr = xcorr / len(a)  # Normalize

        # Find lag with maximum correlation
        lags = np.arange(-len(a) + 1, len(a))
        valid_mask = (lags >= -self.max_lag) & (lags <= self.max_lag)

        valid_xcorr = xcorr[valid_mask]
        valid_lags = lags[valid_mask]

        if len(valid_xcorr) == 0:
            return None, None

        max_idx = np.argmax(np.abs(valid_xcorr))
        max_corr = valid_xcorr[max_idx]
        optimal_lag = valid_lags[max_idx]

        return float(max_corr), int(optimal_lag)

    def _simple_granger(self, a: np.ndarray, b: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """
        Simple Granger-like causality test.

        Tests if A helps predict B beyond B's own history.
        Returns (f_statistic, p_value)
        """
        try:
            n = len(a)
            lag = min(5, n // 4)  # Reasonable lag

            if n < lag * 3:
                return None, None

            # Build lagged arrays
            y = b[lag:]
            x_restricted = np.column_stack([b[lag-i-1:-i-1] for i in range(lag)])
            x_full = np.column_stack([
                *[b[lag-i-1:-i-1] for i in range(lag)],
                *[a[lag-i-1:-i-1] for i in range(lag)]
            ])

            # Add constant
            x_restricted = np.column_stack([np.ones(len(y)), x_restricted])
            x_full = np.column_stack([np.ones(len(y)), x_full])

            # OLS for restricted and full models
            _, resid_r, _, _ = np.linalg.lstsq(x_restricted, y, rcond=None)
            _, resid_f, _, _ = np.linalg.lstsq(x_full, y, rcond=None)

            # Calculate SSR
            beta_r = np.linalg.lstsq(x_restricted, y, rcond=None)[0]
            beta_f = np.linalg.lstsq(x_full, y, rcond=None)[0]

            ssr_r = np.sum((y - x_restricted @ beta_r) ** 2)
            ssr_f = np.sum((y - x_full @ beta_f) ** 2)

            # F-statistic
            df_diff = lag
            df_resid = len(y) - x_full.shape[1]

            if ssr_f > 0 and df_resid > 0:
                f_stat = ((ssr_r - ssr_f) / df_diff) / (ssr_f / df_resid)
                p_value = 1 - stats.f.cdf(f_stat, df_diff, df_resid)
                return float(f_stat), float(p_value)

        except Exception:
            pass

        return None, None

    def _determine_transfer(
        self,
        corr: float,
        corr_lag: int,
        granger_fstat: Optional[float],
        granger_pvalue: Optional[float]
    ) -> Tuple[str, float, str]:
        """
        Determine transfer direction, strength, and flow.

        Returns (direction, strength, from_to)
        """
        # Base strength on correlation
        strength = abs(corr)

        # Determine direction based on lag
        # Positive lag: A leads B
        if corr_lag > 2:
            from_to = 'a_to_b'
        elif corr_lag < -2:
            from_to = 'b_to_a'
        else:
            from_to = 'a_to_b'  # Default

        # Boost strength if causally significant
        if granger_pvalue is not None and granger_pvalue < 0.05:
            strength = min(1.0, strength * 1.2)

        # Direction type
        direction = 'energy'  # Default; could be 'tension' or 'both' based on series type

        return direction, strength, from_to

    def _classify_transfer(self, lag: int, strength: float, is_significant: bool) -> str:
        """Classify the type of transfer."""
        abs_lag = abs(lag)

        if is_significant and strength > 0.5 and abs_lag <= 5:
            return 'contagion'
        elif is_significant and abs_lag > 5:
            return 'spillover'
        elif abs_lag <= 2 and strength > 0.3:
            return 'co-movement'
        else:
            return 'weak'

    def _empty_result(self, cohort_a: str, cohort_b: str) -> TransferResult:
        """Return empty result when data is insufficient."""
        return TransferResult(
            cohort_from=cohort_a,
            cohort_to=cohort_b,
            transfer_strength=0.0,
            transfer_lag=0,
            transfer_direction='none',
            correlation=0.0,
            correlation_lag=0,
            granger_fstat=None,
            granger_pvalue=None,
            is_significant=False,
            transfer_type='none'
        )

    def detect_all_transfers(
        self,
        cohort_series: Dict[str, pd.Series]
    ) -> List[TransferResult]:
        """
        Detect transfers between all cohort pairs.

        Args:
            cohort_series: Dict mapping cohort_id -> signal topology

        Returns:
            List of TransferResult for all pairs
        """
        cohorts = list(cohort_series.keys())
        results = []

        for i, cohort_a in enumerate(cohorts):
            for cohort_b in cohorts[i+1:]:
                result = self.run(
                    cohort_series[cohort_a],
                    cohort_series[cohort_b],
                    cohort_a,
                    cohort_b
                )
                results.append(result)

        return results


def detect_transfer(
    series_a: np.ndarray,
    series_b: np.ndarray,
    cohort_a: str = 'A',
    cohort_b: str = 'B'
) -> Dict[str, Any]:
    """
    Functional interface for transfer detection.

    Returns dict with transfer metrics.
    """
    engine = TransferDetectorEngine()
    result = engine.run(
        pd.Series(series_a),
        pd.Series(series_b),
        cohort_a,
        cohort_b
    )

    return {
        'cohort_from': result.cohort_from,
        'cohort_to': result.cohort_to,
        'transfer_strength': result.transfer_strength,
        'transfer_lag': result.transfer_lag,
        'transfer_direction': result.transfer_direction,
        'correlation': result.correlation,
        'correlation_lag': result.correlation_lag,
        'granger_fstat': result.granger_fstat,
        'granger_pvalue': result.granger_pvalue,
        'is_significant': result.is_significant,
        'transfer_type': result.transfer_type
    }
