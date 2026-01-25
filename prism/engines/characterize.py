"""
PRISM Characterization Engine
=============================

Computes 6-axis dynamical classification for signal topology data.
This is the gatekeeper that determines valid metrics and weights.

Axes:
    1. Stationarity: Does it trend or mean-revert?
    2. Memory: Persistent, random, or anti-persistent?
    3. Periodicity: Does it oscillate?
    4. Complexity: Simple or complex?
    5. Determinism: Stochastic or deterministic?
    6. Volatility: Constant or clustered variance?

Usage:
    from prism.engines.characterize import Characterizer

    char = Characterizer()
    result = char.compute(values)
    print(result.dynamical_class)
    print(result.valid_engines)
    print(result.metric_weights)
"""

import logging
import numpy as np
from scipy import stats, signal
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime, date
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Import discontinuity engines
try:
    from prism.engines.state.break_detector import compute_breaks, analyze_break_pattern, get_break_metrics
    HAS_BREAK_DETECTOR = True
except ImportError:
    HAS_BREAK_DETECTOR = False

try:
    from prism.engines.detection.step_detector import compute as get_step_metrics
    # Backwards compat alias
    get_heaviside_metrics = get_step_metrics
    HAS_HEAVISIDE = True
except ImportError:
    HAS_HEAVISIDE = False

try:
    from prism.engines.detection.spike_detector import compute as get_spike_metrics
    # Backwards compat alias
    get_dirac_metrics = get_spike_metrics
    HAS_DIRAC = True
except ImportError:
    HAS_DIRAC = False


# =============================================================================
# STANDALONE DERIVATION FUNCTIONS
# =============================================================================

def compute_dfa_with_derivation(
    values: np.ndarray,
    signal_id: str = "unknown",
    window_id: str = "0",
    window_start: str = None,
    window_end: str = None,
) -> tuple:
    """
    Compute DFA (Detrended Fluctuation Analysis) with full mathematical derivation.

    Returns:
        tuple: (result_dict, Derivation object)
    """
    from prism.entry_points.derivations.base import Derivation

    deriv = Derivation(
        engine_name="dfa",
        method_name="Detrended Fluctuation Analysis (DFA)",
        signal_id=signal_id,
        window_id=window_id,
        window_start=window_start,
        window_end=window_end,
        sample_size=len(values),
        raw_data_sample=values[:10].tolist() if len(values) >= 10 else values.tolist(),
    )

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)

    if n < 50:
        deriv.final_result = None
        deriv.interpretation = "Insufficient data (n < 50)"
        return {"dfa_exponent": None}, deriv

    # Step 1: Data summary
    deriv.add_step(
        title="Input Data Summary",
        equation="X = {x₁, x₂, ..., xₙ}",
        calculation=f"n = {n}\nMean: {np.mean(values):.4f}\nStd: {np.std(values):.4f}",
        result=n,
        result_name="n"
    )

    # Step 2: Integrate the series
    mean_val = np.mean(values)
    y = np.cumsum(values - mean_val)

    deriv.add_step(
        title="Integrate the Series (Cumulative Sum)",
        equation="Y(k) = Σᵢ₌₁ᵏ (xᵢ - x̄)",
        calculation=f"x̄ = {mean_val:.4f}\n\nY(1) = {values[0]:.4f} - {mean_val:.4f} = {y[0]:.4f}\nY(2) = Y(1) + ({values[1]:.4f} - {mean_val:.4f}) = {y[1]:.4f}\n⋮\nY(n) = {y[-1]:.4f}",
        result=y[-1],
        result_name="Y(n)",
        notes="The integrated series represents cumulative deviations from the mean"
    )

    # Step 3: Generate scale sizes
    scales = np.logspace(1, np.log10(n // 4), num=20, dtype=int)
    scales = np.unique(scales)
    scales = scales[scales >= 4]

    deriv.add_step(
        title="Generate Scale Sizes (Log-spaced)",
        equation="s ∈ {s₁, s₂, ..., sₖ}",
        calculation=f"Min scale: 4\nMax scale: n/4 = {n//4}\nScales: {scales.tolist()}",
        result=len(scales),
        result_name="n_scales",
        notes=f"Using {len(scales)} scales for regression"
    )

    # Step 4: Compute fluctuations for each scale
    fluctuations = []
    example_scale = scales[0]
    example_n_segments = n // example_scale

    for scale in scales:
        n_segments = n // scale
        if n_segments < 1:
            continue

        rms_segments = []
        for i in range(n_segments):
            segment = y[i * scale : (i + 1) * scale]
            if len(segment) < 2:
                continue

            # Fit linear trend
            x_axis = np.arange(len(segment))
            coeffs = np.polyfit(x_axis, segment, 1)
            trend = np.polyval(coeffs, x_axis)

            # RMS of detrended segment
            rms = np.sqrt(np.mean((segment - trend) ** 2))
            if rms > 0:
                rms_segments.append(rms)

        if rms_segments:
            fluctuations.append((scale, np.mean(rms_segments)))

    # Show example for first scale
    first_segment = y[:example_scale]
    x_axis = np.arange(example_scale)
    coeffs = np.polyfit(x_axis, first_segment, 1)
    trend = np.polyval(coeffs, x_axis)
    rms_example = np.sqrt(np.mean((first_segment - trend) ** 2))

    deriv.add_step(
        title="Detrend and Compute Fluctuations",
        equation="F(s) = √[(1/s) Σₖ (Y(k) - Yₜᵣₑₙ𝒹(k))²]",
        calculation=f"Example for scale s = {example_scale}:\n  Segment 0: Y[0:{example_scale}]\n  Linear fit: slope = {coeffs[0]:.4f}, intercept = {coeffs[1]:.4f}\n  RMS fluctuation = {rms_example:.4f}\n\n  Segments for s={example_scale}: {example_n_segments}\n  Mean fluctuation F({example_scale}) = {fluctuations[0][1]:.4f}",
        result=fluctuations[0][1],
        result_name=f"F({example_scale})",
        notes="Detrending removes local linear trends from each segment"
    )

    # Step 5: Show all fluctuations
    fluct_str = "\n".join([f"  s={s:4d}: F(s) = {f:.6f}" for s, f in fluctuations[:5]])
    if len(fluctuations) > 5:
        fluct_str += "\n  ⋮"
        fluct_str += f"\n  s={fluctuations[-1][0]:4d}: F(s) = {fluctuations[-1][1]:.6f}"

    deriv.add_step(
        title="Fluctuation Function F(s) for All Scales",
        equation="Compute F(s) for each scale s",
        calculation=fluct_str,
        result=len(fluctuations),
        result_name="n_points"
    )

    # Step 6: Log-log regression
    log_scales = np.log([f[0] for f in fluctuations])
    log_fluct = np.log([f[1] for f in fluctuations])
    slope, intercept = np.polyfit(log_scales, log_fluct, 1)
    alpha = float(np.clip(slope, 0, 2))

    deriv.add_step(
        title="Log-Log Regression for DFA Exponent",
        equation="log(F(s)) = α × log(s) + c",
        calculation=f"log(s) = [{log_scales[0]:.4f}, {log_scales[1]:.4f}, ..., {log_scales[-1]:.4f}]\nlog(F) = [{log_fluct[0]:.4f}, {log_fluct[1]:.4f}, ..., {log_fluct[-1]:.4f}]\n\nLinear fit: y = {slope:.6f}x + {intercept:.4f}",
        result=alpha,
        result_name="α",
        notes="The slope α is the DFA scaling exponent"
    )

    deriv.final_result = alpha
    deriv.prism_output = alpha

    # Interpretation
    if alpha < 0.5:
        interp = f"α = {alpha:.4f} < 0.5 indicates **anti-persistent** (mean-reverting) dynamics."
    elif alpha < 0.6:
        interp = f"α = {alpha:.4f} ≈ 0.5 indicates **uncorrelated** (white noise-like) dynamics."
    elif alpha < 1.0:
        interp = f"α = {alpha:.4f} > 0.5 indicates **persistent** (long-range correlated) dynamics."
    elif alpha < 1.1:
        interp = f"α = {alpha:.4f} ≈ 1.0 indicates **1/f noise** (pink noise) dynamics."
    else:
        interp = f"α = {alpha:.4f} > 1.0 indicates **non-stationary** dynamics (possibly deterministic chaos)."

    deriv.interpretation = interp

    return {"dfa_exponent": alpha}, deriv


# =============================================================================
# CONFIGURATION
# =============================================================================

# Import unified engine mapping (single source of truth)
from prism.engines.engine_mapping import (
    ENGINE_MAPPING,
    get_valid_engines as _get_valid_engines_from_mapping,
    get_metric_weights as _get_metric_weights_from_mapping,
)


@dataclass
class CharacterizationResult:
    """Result of characterization computation."""
    signal_id: str
    window_end: date
    window_size: int

    # 6 Axes (0.0 to 1.0)
    ax_stationarity: float
    ax_memory: float
    ax_periodicity: float
    ax_complexity: float
    ax_determinism: float
    ax_volatility: float

    # Derived
    dynamical_class: str
    valid_engines: List[str]
    metric_weights: Dict[str, float]
    return_method: str  # 'log_return', 'simple_diff', or 'pct_change'

    # Data handling (PR 015)
    frequency: str = 'daily'  # 'daily', 'weekly', 'monthly', 'intraday', 'irregular'
    avg_gap_days: Optional[float] = None
    max_gap_days: Optional[float] = None
    is_step_function: bool = False
    step_duration_mean: Optional[float] = None
    unique_value_ratio: Optional[float] = None
    change_ratio: Optional[float] = None
    quote_convention: Optional[str] = None  # 'per_usd' or 'per_foreign' for FX

    # Metadata
    computed_at: datetime = field(default_factory=datetime.now)
    computation_ms: int = 0
    memory_method: str = 'rs'  # 'rs' (R/S analysis) or 'dfa' (DFA fallback)

    # Discontinuity detection (break_detector, heaviside, dirac)
    n_breaks: int = 0
    break_rate: float = 0.0
    break_pattern: str = 'NONE'  # PERIODIC, ACCELERATING, DECELERATING, IRREGULAR, NONE
    has_steps: bool = False  # Heaviside-like persistent level shifts
    has_impulses: bool = False  # Dirac-like transient shocks
    heaviside_n_steps: int = 0
    heaviside_mean_magnitude: float = 0.0
    dirac_n_impulses: int = 0
    dirac_mean_magnitude: float = 0.0

    def to_dict(self) -> dict:
        return {
            'signal_id': self.signal_id,
            'window_end': self.window_end,
            'window_size': self.window_size,
            'ax_stationarity': self.ax_stationarity,
            'ax_memory': self.ax_memory,
            'ax_periodicity': self.ax_periodicity,
            'ax_complexity': self.ax_complexity,
            'ax_determinism': self.ax_determinism,
            'ax_volatility': self.ax_volatility,
            'dynamical_class': self.dynamical_class,
            'valid_engines': self.valid_engines,
            'metric_weights': self.metric_weights,
            'return_method': self.return_method,
            'frequency': self.frequency,
            'avg_gap_days': self.avg_gap_days,
            'max_gap_days': self.max_gap_days,
            'is_step_function': self.is_step_function,
            'step_duration_mean': self.step_duration_mean,
            'unique_value_ratio': self.unique_value_ratio,
            'change_ratio': self.change_ratio,
            'quote_convention': self.quote_convention,
            'computed_at': self.computed_at,
            'computation_ms': self.computation_ms,
            'memory_method': self.memory_method,
            # Discontinuity detection
            'n_breaks': self.n_breaks,
            'break_rate': self.break_rate,
            'break_pattern': self.break_pattern,
            'has_steps': self.has_steps,
            'has_impulses': self.has_impulses,
            'heaviside_n_steps': self.heaviside_n_steps,
            'heaviside_mean_magnitude': self.heaviside_mean_magnitude,
            'dirac_n_impulses': self.dirac_n_impulses,
            'dirac_mean_magnitude': self.dirac_mean_magnitude,
        }


class Characterizer:
    """
    Computes 6-axis dynamical characterization of signal topology.

    This is a lightweight pass that determines:
    1. What type of dynamical system is this?
    2. Which engines produce meaningful metrics?
    3. How should metrics be weighted?
    """

    # Class-level cache for return_method per signal
    # This ensures consistency across windows for the same signal
    _return_method_cache: Dict[str, str] = {}

    def __init__(self):
        pass

    @classmethod
    def clear_return_method_cache(cls):
        """Clear the return_method cache (useful for testing or reprocessing)."""
        cls._return_method_cache = {}

    @classmethod
    def set_return_method(cls, signal_id: str, method: str):
        """Manually set return_method for an signal (for configuration override)."""
        cls._return_method_cache[signal_id] = method

    def compute(
        self,
        values: np.ndarray,
        signal_id: str = '',
        window_end: Optional[date] = None,
        dates: Optional[np.ndarray] = None,
    ) -> CharacterizationResult:
        """
        Compute 6-axis characterization for a signal topology.

        Args:
            values: Signal values (1D array)
            signal_id: Identifier for this signal
            window_end: End date of window
            dates: Observation dates (optional, for frequency detection)

        Returns:
            CharacterizationResult with axes, valid engines, and weights
        """
        import time
        start = time.time()

        if window_end is None:
            window_end = date.today()

        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]
        n = len(values)

        # Compute 6 axes
        ax_stationarity = self._compute_stationarity(values)
        ax_memory, memory_method = self._compute_memory(values)
        ax_periodicity = self._compute_periodicity(values)
        ax_complexity = self._compute_complexity(values)
        ax_determinism = self._compute_determinism(values)
        ax_volatility = self._compute_volatility(values)

        # Derive class label
        dynamical_class = self._derive_class(
            ax_stationarity, ax_memory, ax_periodicity,
            ax_complexity, ax_determinism, ax_volatility
        )

        # Determine valid engines
        axes = {
            'ax_stationarity': ax_stationarity,
            'ax_memory': ax_memory,
            'ax_periodicity': ax_periodicity,
            'ax_complexity': ax_complexity,
            'ax_determinism': ax_determinism,
            'ax_volatility': ax_volatility,
        }
        valid_engines = self._get_valid_engines(axes)

        # Compute metric weights
        metric_weights = self._compute_weights(axes, valid_engines)

        # Determine return method
        return_method = self._determine_return_method(values, signal_id)

        # Data handling detection (PR 015)
        frequency, avg_gap_days, max_gap_days = self._detect_frequency(dates)
        is_step, step_duration_mean, unique_ratio, change_ratio = self._detect_step_function(values, dates)
        quote_convention = self._detect_quote_convention(signal_id)

        # Gate RQA for step functions - remove from valid engines
        if is_step and 'rqa' in valid_engines:
            valid_engines = [e for e in valid_engines if e != 'rqa']

        # =================================================================
        # DISCONTINUITY DETECTION (break_detector, heaviside, dirac)
        # =================================================================
        n_breaks = 0
        break_rate = 0.0
        break_pattern = 'NONE'
        has_steps = False
        has_impulses = False
        heaviside_n_steps = 0
        heaviside_mean_magnitude = 0.0
        dirac_n_impulses = 0
        dirac_mean_magnitude = 0.0

        if HAS_BREAK_DETECTOR and n >= 50:
            try:
                # Run break detection
                break_metrics = get_break_metrics(values)
                n_breaks = int(break_metrics.get('break_n') or 0)
                break_rate = break_metrics.get('break_rate')
                # Determine pattern from signal flags
                if break_metrics.get('break_is_periodic', 0) > 0:
                    break_pattern = 'PERIODIC'
                elif break_metrics.get('break_is_accelerating', 0) > 0:
                    break_pattern = 'ACCELERATING'
                elif n_breaks > 0:
                    break_pattern = 'IRREGULAR'
                else:
                    break_pattern = 'NONE'

                # If we have breaks, run heaviside and dirac analysis
                if n_breaks > 0:
                    # Add break_detector to valid engines
                    if 'break_detector' not in valid_engines:
                        valid_engines.append('break_detector')

                    # Heaviside (persistent level shifts)
                    if HAS_HEAVISIDE:
                        try:
                            heaviside_metrics = get_heaviside_metrics(values)
                            heaviside_n_steps = heaviside_metrics.get('heaviside_n_steps') or 0
                            heaviside_mean_magnitude = heaviside_metrics.get('heaviside_mean_magnitude')
                            has_steps = heaviside_n_steps > 0
                            if has_steps and 'heaviside' not in valid_engines:
                                valid_engines.append('heaviside')
                        except Exception as e:
                            logger.debug(f"Heaviside detection failed: {e}")

                    # Dirac (transient impulses)
                    if HAS_DIRAC:
                        try:
                            dirac_metrics = get_dirac_metrics(values)
                            dirac_n_impulses = dirac_metrics.get('dirac_n_impulses') or 0
                            dirac_mean_magnitude = dirac_metrics.get('dirac_mean_magnitude')
                            has_impulses = dirac_n_impulses > 0
                            if has_impulses and 'dirac' not in valid_engines:
                                valid_engines.append('dirac')
                        except Exception as e:
                            logger.debug(f"Dirac detection failed: {e}")

            except Exception as e:
                logger.debug(f"Break detection failed: {e}")

        elapsed_ms = int((time.time() - start) * 1000)

        return CharacterizationResult(
            signal_id=signal_id,
            window_end=window_end,
            window_size=n,
            ax_stationarity=ax_stationarity,
            ax_memory=ax_memory,
            ax_periodicity=ax_periodicity,
            ax_complexity=ax_complexity,
            ax_determinism=ax_determinism,
            ax_volatility=ax_volatility,
            dynamical_class=dynamical_class,
            valid_engines=valid_engines,
            metric_weights=metric_weights,
            return_method=return_method,
            frequency=frequency,
            avg_gap_days=avg_gap_days,
            max_gap_days=max_gap_days,
            is_step_function=is_step,
            step_duration_mean=step_duration_mean,
            unique_value_ratio=unique_ratio,
            change_ratio=change_ratio,
            quote_convention=quote_convention,
            computation_ms=elapsed_ms,
            memory_method=memory_method,
            # Discontinuity detection results
            n_breaks=n_breaks,
            break_rate=break_rate,
            break_pattern=break_pattern,
            has_steps=has_steps,
            has_impulses=has_impulses,
            heaviside_n_steps=heaviside_n_steps,
            heaviside_mean_magnitude=heaviside_mean_magnitude,
            dirac_n_impulses=dirac_n_impulses,
            dirac_mean_magnitude=dirac_mean_magnitude,
        )

    # =========================================================================
    # AXIS COMPUTATIONS
    # =========================================================================

    def _compute_stationarity(self, values: np.ndarray) -> float:
        """
        Measure stationarity (0=non-stationary, 1=stationary).
        Uses variance ratio test and trend detection.
        Also checks stationarity of returns for price-like series.
        """
        if len(values) < 20:
            return 0.5

        try:
            n = len(values)

            # Variance ratio across multiple splits
            ratios = []
            for split in [0.25, 0.5, 0.75]:
                idx = int(n * split)
                var1 = np.var(values[:idx])
                var2 = np.var(values[idx:])
                if var1 > 0:
                    ratios.append(var2 / var1)

            if not ratios:
                return 0.5

            # Trending series: variance grows -> ratio > 1
            avg_ratio = np.mean(ratios)

            # Also check for trend using linear regression
            x = np.arange(n)
            slope, intercept = np.polyfit(x, values, 1)
            trend_strength = abs(slope) * n / (np.std(values) + 1e-10)

            # Check returns stationarity for integrated processes
            # If levels are non-stationary but returns are stationary, it's like GARCH
            # Use LOG returns for positive price-like series (simple diff for others)
            if np.all(values > 0):
                with np.errstate(divide='ignore', invalid='ignore'):
                    increments = np.diff(np.log(values))
                    increments = increments[np.isfinite(increments)]
            else:
                increments = np.diff(values)

            inc_ratios = []
            for split in [0.25, 0.5, 0.75]:
                idx = int(len(increments) * split)
                var1 = np.var(increments[:idx])
                var2 = np.var(increments[idx:])
                if var1 > 0:
                    inc_ratios.append(var2 / var1)

            inc_avg_ratio = np.mean(inc_ratios) if inc_ratios else 1.0

            # Determine level stationarity
            if avg_ratio > 2.0:
                level_stationarity = 0.1
            elif avg_ratio > 1.5:
                level_stationarity = 0.2
            elif avg_ratio > 1.3:
                level_stationarity = 0.3
            elif avg_ratio < 0.5:
                level_stationarity = 0.2
            elif avg_ratio < 0.7:
                level_stationarity = 0.3
            else:
                level_stationarity = 0.7

            # Trend reduces stationarity
            if trend_strength > 2.0:
                level_stationarity = min(level_stationarity, 0.2)
            elif trend_strength > 1.0:
                level_stationarity = min(level_stationarity, 0.4)

            # If levels are non-stationary but log returns are stationary, boost score
            # This ONLY applies to price-like series (all positive) where we use log returns
            # Random walks (simple cumsum) should NOT get this boost - they're I(1) processes
            levels_nonstationary = (avg_ratio > 1.5 or avg_ratio < 0.67 or trend_strength > 1.0)
            returns_stationary = (0.7 < inc_avg_ratio < 1.4)
            is_price_series = np.all(values > 0)  # Only boost for positive (price-like) series

            if levels_nonstationary and returns_stationary and is_price_series:
                # GARCH-like process - substantial boost
                # GARCH is covariance-stationary in returns, so treat as weakly stationary
                returns_boost = 0.4
            else:
                returns_boost = 0.0

            stationarity = level_stationarity + returns_boost

            return float(np.clip(stationarity, 0, 1))

        except Exception:
            return 0.5

    def _is_nonstationary(self, values: np.ndarray) -> bool:
        """Quick check if series is non-stationary."""
        n = len(values)
        if n < 20:
            return False

        # Variance ratio test
        mid = n // 2
        var1 = np.var(values[:mid])
        var2 = np.var(values[mid:])
        if var1 > 0:
            ratio = var2 / var1
            if ratio > 1.5 or ratio < 0.67:
                return True

        # Trend test
        x = np.arange(n)
        slope, _ = np.polyfit(x, values, 1)
        trend_strength = abs(slope) * n / (np.std(values) + 1e-10)
        if trend_strength > 1.0:
            return True

        return False

    def _compute_memory(self, values: np.ndarray) -> Tuple[float, str]:
        """
        Measure memory/persistence using Hurst exponent via DFA.

        H < 0.5: Anti-persistent (mean-reverting)
        H = 0.5: Random walk (no memory)
        H > 0.5: Persistent (trending)

        Uses DFA (Detrended Fluctuation Analysis) exclusively for consistency.
        R/S analysis was previously used but creates discontinuities when
        switching methods (R/S often gives H > 1.0 for deterministic chaos,
        triggering a fallback to DFA mid-series). Using DFA exclusively
        ensures consistent behavior and avoids false regime signals in the
        Laplace field from method-switching artifacts.

        Returns tuple of (H clipped to [0.01, 0.99], method used).
        """
        if len(values) < 100:
            return 0.5, 'insufficient_data'

        try:
            # Use DFA exclusively for consistency across all series types
            hurst_dfa = self._compute_hurst_dfa(values)
            hurst = np.clip(hurst_dfa, 0.01, 0.99)
            return float(hurst), 'dfa'

        except Exception:
            # Fallback: use ACF-based estimate
            try:
                acf1 = np.corrcoef(values[:-1], values[1:])[0, 1]
                if np.isnan(acf1):
                    return 0.5, 'acf_fallback'
                # Map ACF to approximate Hurst: H ≈ 0.5 + acf1/4
                return float(np.clip(0.5 + acf1 / 4, 0.01, 0.99)), 'acf_fallback'
            except:
                return 0.5, 'error_fallback'

    def _compute_hurst_dfa(self, values: np.ndarray) -> float:
        """
        Detrended Fluctuation Analysis - robust for deterministic chaos.

        DFA handles non-stationarity and deterministic trends better than R/S.
        Used as fallback when R/S gives out-of-bounds values.
        """
        try:
            n = len(values)

            # Integrate the series (cumulative sum of deviations from mean)
            y = np.cumsum(values - np.mean(values))

            # Scale sizes (log-spaced)
            scales = np.logspace(1, np.log10(n // 4), num=20, dtype=int)
            scales = np.unique(scales)
            scales = scales[scales >= 4]

            if len(scales) < 3:
                return 0.5

            fluctuations = []

            for scale in scales:
                n_segments = n // scale
                if n_segments < 1:
                    continue

                rms_segments = []
                for i in range(n_segments):
                    segment = y[i * scale : (i + 1) * scale]
                    if len(segment) < 2:
                        continue

                    # Fit linear trend
                    x_axis = np.arange(len(segment))
                    coeffs = np.polyfit(x_axis, segment, 1)
                    trend = np.polyval(coeffs, x_axis)

                    # RMS of detrended segment
                    rms = np.sqrt(np.mean((segment - trend) ** 2))
                    if rms > 0:
                        rms_segments.append(rms)

                if rms_segments:
                    fluctuations.append((scale, np.mean(rms_segments)))

            if len(fluctuations) < 3:
                return 0.5

            # Log-log regression to get Hurst exponent
            log_scales = np.log([f[0] for f in fluctuations])
            log_fluct = np.log([f[1] for f in fluctuations])

            slope, _ = np.polyfit(log_scales, log_fluct, 1)

            return float(np.clip(slope, 0, 1))

        except Exception:
            return 0.5

    def _compute_periodicity(self, values: np.ndarray) -> float:
        """
        Measure periodicity (0=aperiodic, 1=periodic).

        True periodicity = consistent peak frequency across multiple windows.
        AR(1) and random walks have peaks that wander. True periodic signals don't.
        """
        if len(values) < 128:  # Need enough for 4 windows of 32
            return None

        try:
            # For non-stationary series, analyze increments to avoid trend artifacts
            if self._is_nonstationary(values):
                work_values = np.diff(values)
            else:
                work_values = signal.detrend(values)

            if len(work_values) < 128:
                return None

            # Split into 4 windows and check peak consistency
            n_windows = 4
            window_size = len(work_values) // n_windows
            peak_freqs = []
            peak_powers = []

            for i in range(n_windows):
                chunk = work_values[i * window_size : (i + 1) * window_size]
                fft = np.abs(np.fft.rfft(chunk))

                # Skip DC, find peak
                if len(fft) < 2:
                    return None
                peak_idx = np.argmax(fft[1:]) + 1
                peak_freqs.append(peak_idx)
                peak_powers.append(fft[peak_idx])

            # True periodic = same peak in all windows (or consistent top-N peaks for multi-frequency)
            # Random/AR(1) = peaks wander randomly
            mean_freq = np.mean(peak_freqs)
            if mean_freq == 0:
                return None
            freq_cv = np.std(peak_freqs) / mean_freq  # Coefficient of variation

            # For truly periodic signals:
            # 1. Peak frequencies should be consistent (low CV)
            # 2. AND peaks should be at meaningful frequencies (not ultra-low like AR(1))
            #
            # AR(1) concentrates power at bins 1-5 (< 2% of Nyquist)
            # but peak location wanders. Not truly periodic.

            # Require CV < 0.2 for single-frequency OR reasonable multi-freq clustering
            if freq_cv > 0.2:
                # Check if it's multi-frequency with consistent peaks
                freq_range = max(peak_freqs) - min(peak_freqs)
                max_possible_range = window_size // 2

                # Even for multi-freq, don't accept if all peaks are ultra-low frequency
                min_peak = min(peak_freqs)
                if min_peak < window_size // 20:  # < 5% of Nyquist
                    return None  # Low-freq peaks that wander = AR(1), not periodic

                # Multi-freq: range should be small relative to max
                if freq_range > 0.3 * max_possible_range:
                    return None

            # Peaks are consistent - now check SNR on full series
            full_fft = np.abs(np.fft.rfft(work_values))
            if len(full_fft) < 2:
                return None

            # Use actual max peak (consistency check already verified it's periodic)
            peak_power = np.max(full_fft[1:])  # Exclude DC
            noise_floor = np.median(full_fft[1:])

            if noise_floor == 0:
                return None

            snr = peak_power / noise_floor

            # Require SNR > 5 for periodicity
            if snr < 5:
                return None

            # Scale: SNR 5-25 maps to 0.3-1.0
            periodicity = np.clip((snr - 5) / 20, 0, 1) * 0.7 + 0.3

            return float(periodicity)

        except Exception:
            return None

    def _compute_complexity(self, values: np.ndarray) -> float:
        """
        Measure complexity (0=simple, 1=complex).

        Complexity peaks at "edge of chaos", not at max entropy.
        Uses LEVELS (not increments) to capture structural simplicity.
        - Trend: low PE on levels (predictable direction) → simple
        - Random: high PE → cap it (randomness != complexity)
        - Chaotic: moderate PE with structure → complex
        """
        if len(values) < 50:
            return 0.5

        try:
            import antropy as ant

            # Use LEVELS for complexity - captures structural patterns
            # (increments of trend = white noise, loses simplicity info)
            work_values = values

            # Permutation entropy (randomness measure)
            pe = ant.perm_entropy(work_values, order=5, normalize=True)

            # Lempel-Ziv complexity (structure measure)
            binary = (work_values > np.median(work_values)).astype(int)
            lz = ant.lziv_complexity(binary, normalize=True)

            # Penalize pure randomness - high PE doesn't mean complex
            if pe > 0.95:
                complexity = 0.65  # Cap it - pure randomness isn't complex
            elif pe > 0.85 and lz > 0.7:
                complexity = 0.70  # High entropy + unstructured = noise
            else:
                # True complexity = entropy WITH structure
                complexity = pe * 0.6 + (1 - abs(pe - lz)) * 0.4

            return float(np.clip(complexity, 0, 1))

        except ImportError:
            # Fallback to permutation entropy only
            return self._compute_complexity_fallback(values)
        except Exception:
            return 0.5

    def _compute_complexity_fallback(self, values: np.ndarray) -> float:
        """Fallback complexity using permutation entropy."""
        try:
            from collections import Counter
            from math import factorial

            if self._is_nonstationary(values):
                work_values = np.diff(values)
            else:
                work_values = values

            dim = 5
            tau = 1
            n = len(work_values)

            patterns = []
            for i in range(n - (dim - 1) * tau):
                window = work_values[i:i + dim * tau:tau]
                pattern = tuple(np.argsort(window))
                patterns.append(pattern)

            if not patterns:
                return 0.5

            counts = Counter(patterns)
            n_patterns = len(patterns)
            probs = [count / n_patterns for count in counts.values()]

            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            max_entropy = np.log2(factorial(dim))

            return float(np.clip(entropy / max_entropy, 0, 1)) if max_entropy > 0 else 0.5

        except Exception:
            return 0.5

    def _compute_determinism(self, values: np.ndarray) -> float:
        """
        Measure determinism (0=stochastic, 1=deterministic).
        Uses RQA determinism: ratio of recurrence points in diagonal lines.

        For non-stationary series, computes on increments.
        """
        if len(values) < 50:
            return 0.5

        try:
            # For non-stationary series, compute on increments
            if self._is_nonstationary(values):
                work_values = np.diff(values)
            else:
                work_values = values

            if len(work_values) < 50:
                return 0.5

            n = len(work_values)

            # Normalize
            values_norm = (work_values - np.mean(work_values)) / (np.std(work_values) + 1e-10)

            # Use sequential sampling to preserve temporal structure
            sample_size = min(n, 200)
            sample = values_norm[:sample_size]

            # Distance matrix
            diff = sample[:, np.newaxis] - sample[np.newaxis, :]
            distances = np.abs(diff)

            # Threshold at 10th percentile of non-zero distances
            nonzero_distances = distances[distances > 0]
            if len(nonzero_distances) == 0:
                return 0.5
            threshold = np.percentile(nonzero_distances, 10)

            # Recurrence matrix
            R = (distances < threshold).astype(int)
            np.fill_diagonal(R, 0)

            # Total recurrence points
            total_recurrent = R.sum()
            if total_recurrent == 0:
                return None

            # Count recurrence points that are part of diagonal lines (length >= 2)
            # RQA DET = sum of diagonal line points / total recurrence points
            diagonal_points = 0

            for k in range(1, sample_size):
                # Check diagonal k
                diag = np.diag(R, k)
                if len(diag) < 2:
                    continue

                # Find runs of consecutive 1s
                runs = []
                run_length = 0
                for val in diag:
                    if val == 1:
                        run_length += 1
                    else:
                        if run_length >= 2:
                            runs.append(run_length)
                        run_length = 0
                if run_length >= 2:
                    runs.append(run_length)

                # Each run of length L contributes L points to DET
                diagonal_points += sum(runs)

            # Also check negative diagonals (below main)
            for k in range(1, sample_size):
                diag = np.diag(R, -k)
                if len(diag) < 2:
                    continue

                runs = []
                run_length = 0
                for val in diag:
                    if val == 1:
                        run_length += 1
                    else:
                        if run_length >= 2:
                            runs.append(run_length)
                        run_length = 0
                if run_length >= 2:
                    runs.append(run_length)

                diagonal_points += sum(runs)

            # RQA determinism ratio
            determinism = diagonal_points / total_recurrent if total_recurrent > 0 else 0.0

            return float(np.clip(determinism, 0, 1))

        except Exception:
            return 0.5

    def _compute_volatility(self, values: np.ndarray) -> float:
        """
        Measure volatility clustering (0=homoscedastic, 1=clustered).
        Uses squared return autocorrelation, filters for ARCH-like behavior.
        """
        if len(values) < 30:
            return None

        try:
            # Compute returns (log differences for price series, simple diff for others)
            if np.all(values > 0):
                with np.errstate(divide='ignore', invalid='ignore'):
                    returns = np.diff(np.log(values))
            else:
                returns = np.diff(values)
                returns = returns / (np.std(returns) + 1e-10)  # Standardize

            returns = returns[np.isfinite(returns)]

            if len(returns) < 20:
                return None

            # If this looks periodic, squared-return autocorrelation
            # is likely from periodicity, not ARCH effects
            if self._compute_periodicity(values) > 0.5:
                return None

            # Check for VERY strong deterministic trends
            # Only cap volatility for extreme trends (trend_strength > 3.0)
            # Moderate trends (1.5-3.0) should let ACF determine true volatility
            if self._is_nonstationary(values):
                n = len(values)
                x = np.arange(n)
                slope, _ = np.polyfit(x, values, 1)
                trend_strength = abs(slope) * n / (np.std(values) + 1e-10)
                if trend_strength > 3.0:
                    # Very strong deterministic trend - cap volatility
                    return 0.35

            # Squared returns (demeaned)
            returns_demean = returns - np.mean(returns)
            r2 = returns_demean ** 2

            # Multiple lag ACF for robustness
            # Use ALL lags (not just positive) to avoid false positives on white noise
            # White noise: some lags positive, some negative, mean near 0
            # GARCH: ALL lags positive, mean around 0.1-0.2
            acf_vals = []
            for lag in range(1, min(6, len(r2))):
                if len(r2) > lag:
                    acf = np.corrcoef(r2[:-lag], r2[lag:])[0, 1]
                    if not np.isnan(acf):
                        acf_vals.append(acf)

            if not acf_vals:
                return None

            avg_acf = np.mean(acf_vals)

            # Check if this is deterministic (chaos) vs stochastic (GARCH)
            determinism = self._compute_determinism(values)

            # Chaotic systems have oscillating ACF (alternating +/-) but still have
            # volatility-like results. Use absolute ACF for chaotic systems.
            if determinism > 0.5:
                # For deterministic/chaotic systems, use mean of absolute ACF
                # This captures the structure even when ACF oscillates
                abs_avg_acf = np.mean(np.abs(acf_vals))
                # Scale down for chaos (it's not true ARCH clustering)
                avg_acf = abs_avg_acf * 0.4
            elif avg_acf < 0:
                # For stochastic systems, negative mean ACF means no clustering
                return None

            # GARCH(1,1) samples typically show mean squared-return ACF ~ 0.1-0.15
            # Scale: ACF 0-0.12 maps to 0-1
            volatility = np.clip(avg_acf / 0.12, 0, 1)

            return float(volatility)

        except Exception:
            return None

    def _determine_return_method(self, values: np.ndarray, signal_id: str = '') -> str:
        """
        Determine appropriate change calculation method.

        IMPORTANT: Once determined for an signal, the method is cached and
        reused for all subsequent windows. This prevents method-flip discontinuities
        where a series fluctuating near threshold boundaries (e.g., values near 0.01)
        would use different methods in different windows, creating artificial
        noise in the Laplace field.

        Returns:
            'log_return': For level series (strictly positive, typical > 1.0)
            'simple_diff': For rates/spreads/unbounded series

        Cache behavior:
            - First call for an signal: compute and cache result
            - Subsequent calls: return cached result
            - Use clear_return_method_cache() to reset
            - Use set_return_method() to override
        """
        # Check cache first - ensures consistency across windows
        if signal_id and signal_id in self._return_method_cache:
            return self._return_method_cache[signal_id]

        # Determine method based on value characteristics
        method = self._compute_return_method(values)

        # Cache for this signal if we have an ID
        if signal_id:
            self._return_method_cache[signal_id] = method
            logger.debug(f"Cached return_method for {signal_id}: {method}")

        return method

    def _compute_return_method(self, values: np.ndarray) -> str:
        """
        Internal method to compute change calculation method.

        More conservative than before to reduce false method switches:
        - Only use log_return for clearly price-like series (min > 10.0)
        - Use simple_diff for anything near zero or with large range ratio
        """
        # Check for zeros or negatives - must use simple diff
        has_zeros = np.any(values == 0)
        has_negatives = np.any(values < 0)

        if has_zeros or has_negatives:
            return 'simple_diff'

        min_value = np.min(values)
        max_value = np.max(values)

        # If min value < 10.0, it could be a rate, percentage, or small-valued series
        # Be conservative: use simple_diff to avoid log issues near zero
        if min_value < 10.0:
            return 'simple_diff'

        # If range ratio is very large (>100x), log is appropriate for scale normalization
        # Otherwise, simple_diff is safer
        if max_value / (min_value + 1e-10) > 100:
            return 'log_return'

        # For moderate positive values (10-1000 range), use log_return
        if min_value >= 10.0:
            return 'log_return'

        return 'simple_diff'

    # =========================================================================
    # DATA HANDLING DETECTION (PR 015)
    # =========================================================================

    def _detect_frequency(
        self,
        dates: Optional[np.ndarray]
    ) -> Tuple[str, Optional[float], Optional[float]]:
        """
        Detect data frequency from observation dates.

        Args:
            dates: Array of observation dates

        Returns:
            Tuple of (frequency, avg_gap_days, max_gap_days)
        """
        if dates is None or len(dates) < 2:
            return 'daily', None, None

        try:
            # Convert to datetime if needed
            if hasattr(dates[0], 'days'):
                # Already timedelta-like
                gaps = np.array([d.days for d in np.diff(dates)])
            else:
                # Convert dates to days
                import pandas as pd
                dates_pd = pd.to_datetime(dates)
                gaps = np.diff(dates_pd).astype('timedelta64[D]').astype(int)

            if len(gaps) == 0:
                return 'daily', None, None

            median_gap = float(np.median(gaps))
            avg_gap = float(np.mean(gaps))
            max_gap = float(np.max(gaps))

            # Classify frequency based on median gap
            # Daily data with gaps (weekends, etc.) has median 1.0-2.0
            if median_gap < 1:
                frequency = 'intraday'
            elif median_gap <= 3:
                frequency = 'daily'  # Accounts for gaps (weekends, holidays, sensor downtime)
            elif median_gap <= 8:
                frequency = 'weekly'
            elif median_gap <= 35:
                frequency = 'monthly'
            else:
                frequency = 'irregular'

            return frequency, avg_gap, max_gap

        except Exception:
            return 'daily', None, None

    def _detect_step_function(
        self,
        values: np.ndarray,
        dates: Optional[np.ndarray] = None,
        min_step_duration: float = 7.0
    ) -> Tuple[bool, Optional[float], Optional[float], Optional[float]]:
        """
        Detect if series behaves like a step function.

        Step functions have:
        - Long periods of constant values (e.g., Fed Funds Rate)
        - Discrete jumps between levels
        - The key is mean duration between changes, not just unique/change ratios

        For policy rates like DFF:
        - Stays constant for weeks/months
        - Jumps in discrete increments (25bp, 50bp)
        - Produces degenerate RQA (constant input)

        Args:
            values: Signal values
            dates: Observation dates (optional)
            min_step_duration: Minimum mean duration (days) between changes to qualify

        Returns:
            Tuple of (is_step_function, step_duration_mean, unique_value_ratio, change_ratio)
        """
        if len(values) < 10:
            return False, None, None, None

        try:
            n = len(values)

            # Count unique values
            unique_values = len(np.unique(values))
            unique_ratio = unique_values / n

            # Count changes (where consecutive values differ)
            changes = np.sum(values[1:] != values[:-1])
            change_ratio = changes / (n - 1) if n > 1 else 0

            # Calculate mean step duration if dates provided
            step_duration_mean = None
            is_step = False

            if dates is not None and len(dates) == n:
                try:
                    import pandas as pd
                    dates_pd = pd.to_datetime(dates)

                    # Find indices where value changes
                    change_indices = np.where(values[1:] != values[:-1])[0] + 1
                    if len(change_indices) > 0:
                        # Add start and end
                        all_indices = np.concatenate([[0], change_indices, [n - 1]])
                        # Calculate durations
                        durations = []
                        for i in range(len(all_indices) - 1):
                            start_date = dates_pd[all_indices[i]]
                            end_date = dates_pd[all_indices[i + 1]]
                            duration = (end_date - start_date).days
                            if duration > 0:
                                durations.append(duration)
                        if durations:
                            step_duration_mean = float(np.mean(durations))
                            max_step_duration = float(np.max(durations))
                            # A series is a step function if:
                            # - Low change ratio (< 50%) AND long max step (> 30 days)
                            #   This catches policy rates that stay constant for months
                            # - OR very few unique values (< 1%)
                            # This distinguishes weekly data (changes every week) from
                            # policy rates (stays constant for weeks/months)
                            is_step = (
                                (change_ratio < 0.5 and max_step_duration > 30) or
                                unique_ratio < 0.01
                            )
                except Exception:
                    pass
            else:
                # Fallback without dates: use ratio-based detection
                # Very low change ratio suggests step function behavior
                is_step = change_ratio < 0.5 and unique_ratio < 0.1

            return is_step, step_duration_mean, unique_ratio, change_ratio

        except Exception:
            return False, None, None, None

    def _detect_quote_convention(self, signal_id: str) -> Optional[str]:
        """
        Detect quote convention from signal ID.

        Args:
            signal_id: The signal identifier

        Returns:
            Quote convention string or None if not applicable
        """
        # Generic implementation - domain-specific quote conventions
        # can be added as needed
        return None

    # =========================================================================
    # DERIVED COMPUTATIONS
    # =========================================================================

    def _derive_class(
        self,
        ax_stationarity: float,
        ax_memory: float,
        ax_periodicity: float,
        ax_complexity: float,
        ax_determinism: float,
        ax_volatility: float,
    ) -> str:
        """Derive compound dynamical class label from axes."""

        labels = []

        # Stationarity
        # Expected ranges: stationary > 0.6, non-stationary < 0.4
        if ax_stationarity >= 0.6:
            labels.append('STATIONARY')
        elif ax_stationarity <= 0.4:
            labels.append('NONSTATIONARY')

        # Memory
        if ax_memory > 0.65:
            labels.append('PERSISTENT')
        elif ax_memory < 0.35:
            labels.append('ANTIPERSISTENT')

        # Periodicity
        if ax_periodicity > 0.5:
            labels.append('OSCILLATORY')
        elif ax_periodicity < 0.2:
            labels.append('APERIODIC')

        # Complexity
        if ax_complexity > 0.6:
            labels.append('COMPLEX')
        elif ax_complexity < 0.3:
            labels.append('SIMPLE')

        # Determinism
        if ax_determinism > 0.7:
            labels.append('DETERMINISTIC')
        elif ax_determinism < 0.3:
            labels.append('STOCHASTIC')

        # Volatility
        if ax_volatility > 0.7:
            labels.append('CLUSTERED_VOL')

        if not labels:
            return 'MIXED'

        return '_'.join(labels)

    def _get_valid_engines(self, axes: Dict[str, float]) -> List[str]:
        """Determine which engines are valid for this characterization."""
        return _get_valid_engines_from_mapping(axes)

    def _compute_weights(
        self,
        axes: Dict[str, float],
        valid_engines: List[str],
    ) -> Dict[str, float]:
        """Compute per-metric weights based on axes."""
        return _get_metric_weights_from_mapping(axes, valid_engines)


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PRISM Characterization Engine')
    parser.add_argument('--test', action='store_true', help='Run tests')
    args = parser.parse_args()

    if args.test:
        print("=" * 60)
        print("PRISM Characterization Engine - Test")
        print("=" * 60)

        char = Characterizer()

        # Test 1: Trending series (e.g., cumulative sensor reading, RUL degradation)
        np.random.seed(42)
        trend = np.cumsum(np.random.randn(500) * 0.1) + 100
        noise = np.random.randn(500) * 2
        trend_series = trend + noise

        result = char.compute(trend_series, signal_id='TREND_001')
        print(f"\n1. Trending series (sensor degradation):")
        print(f"   Class: {result.dynamical_class}")
        print(f"   Axes: stat={result.ax_stationarity:.2f}, mem={result.ax_memory:.2f}, "
              f"per={result.ax_periodicity:.2f}, comp={result.ax_complexity:.2f}, "
              f"det={result.ax_determinism:.2f}, vol={result.ax_volatility:.2f}")
        print(f"   Valid engines: {result.valid_engines}")
        print(f"   Change method: {result.return_method}")

        # Test 2: Oscillatory series (e.g., vibration sensor, periodic measurement)
        t = np.linspace(0, 10 * np.pi, 500)
        oscillatory = np.sin(t * 8) + 0.5 * np.sin(t * 13) + 0.3 * np.random.randn(500)

        result = char.compute(oscillatory, signal_id='VIBRATION_001')
        print(f"\n2. Oscillatory series (vibration sensor):")
        print(f"   Class: {result.dynamical_class}")
        print(f"   Axes: stat={result.ax_stationarity:.2f}, mem={result.ax_memory:.2f}, "
              f"per={result.ax_periodicity:.2f}, comp={result.ax_complexity:.2f}, "
              f"det={result.ax_determinism:.2f}, vol={result.ax_volatility:.2f}")
        print(f"   Valid engines: {result.valid_engines}")

        # Test 3: Mean-reverting series
        mean_rev = np.zeros(500)
        mean_rev[0] = 0
        for i in range(1, 500):
            mean_rev[i] = mean_rev[i-1] * 0.9 + np.random.randn() * 0.5

        result = char.compute(mean_rev, signal_id='MEAN_REV')
        print(f"\n3. Mean-reverting series:")
        print(f"   Class: {result.dynamical_class}")
        print(f"   Axes: stat={result.ax_stationarity:.2f}, mem={result.ax_memory:.2f}, "
              f"per={result.ax_periodicity:.2f}, comp={result.ax_complexity:.2f}, "
              f"det={result.ax_determinism:.2f}, vol={result.ax_volatility:.2f}")
        print(f"   Valid engines: {result.valid_engines}")

        # Test 4: Random walk
        random_walk = np.cumsum(np.random.randn(500))

        result = char.compute(random_walk, signal_id='RANDOM_WALK')
        print(f"\n4. Random walk:")
        print(f"   Class: {result.dynamical_class}")
        print(f"   Axes: stat={result.ax_stationarity:.2f}, mem={result.ax_memory:.2f}, "
              f"per={result.ax_periodicity:.2f}, comp={result.ax_complexity:.2f}, "
              f"det={result.ax_determinism:.2f}, vol={result.ax_volatility:.2f}")
        print(f"   Valid engines: {result.valid_engines}")
        print(f"   Return method: {result.return_method}")

        # Test 5: Bounded positive series (e.g., efficiency ratio, temperature)
        # Values typically in bounded range, can approach zero
        bounded_pos = np.abs(2 + np.cumsum(np.random.randn(500) * 0.1))
        bounded_pos = np.clip(bounded_pos, 0, 10)  # Keep in bounded range
        bounded_pos[100:200] = 0.1  # Near-zero period

        result = char.compute(bounded_pos, signal_id='EFFICIENCY_001')
        print(f"\n5. Bounded positive series (efficiency ratio):")
        print(f"   Class: {result.dynamical_class}")
        print(f"   Values: min={bounded_pos.min():.2f}, max={bounded_pos.max():.2f}")
        print(f"   Change method: {result.return_method}")

        # Test 6: Unbounded series (can be negative, e.g., temperature differential)
        differential = np.random.randn(500) * 0.5 + 0.5  # Can go negative
        result = char.compute(differential, signal_id='TEMP_DIFF_001')
        print(f"\n6. Unbounded series (temperature differential):")
        print(f"   Values: min={differential.min():.2f}, max={differential.max():.2f}")
        print(f"   Change method: {result.return_method}")

        print("\n" + "=" * 60)
        print("Tests complete")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================
# These wrap the Characterizer class for inline use by entry points

# Singleton characterizer instance (stateless, can be reused)
_characterizer = None


def _get_characterizer() -> Characterizer:
    """Get or create singleton Characterizer instance."""
    global _characterizer
    if _characterizer is None:
        _characterizer = Characterizer()
    return _characterizer


def characterize_signal(
    signal_id: str,
    values: np.ndarray,
    dates: Optional[np.ndarray] = None,
    window_end: Optional[date] = None,
) -> CharacterizationResult:
    """
    Characterize a single signal inline.

    This is the primary interface for signal_vector to characterize
    signals as they are processed.

    Args:
        signal_id: The signal identifier
        values: Signal values (1D array, will be cleaned of NaN)
        dates: Optional observation dates (for frequency detection)
        window_end: Optional window end date (defaults to today)

    Returns:
        CharacterizationResult with 6 axes and engine recommendations
    """
    char = _get_characterizer()
    return char.compute(
        values=values,
        signal_id=signal_id,
        window_end=window_end,
        dates=dates,
    )


def get_engines_from_characterization(
    char_result: CharacterizationResult,
    core_engines: Set[str],
    conditional_engines: Set[str],
    discontinuity_engines: Set[str],
) -> Tuple[Set[str], bool]:
    """
    Determine which engines to run based on characterization result.

    Args:
        char_result: Result from characterize_signal()
        core_engines: Set of engines that always run
        conditional_engines: Set of engines that run conditionally
        discontinuity_engines: Set of engines for discontinuity analysis

    Returns:
        Tuple of (engines_to_run, has_discontinuities)
    """
    engines_to_run = core_engines.copy()
    valid_set = set(char_result.valid_engines)

    for engine in conditional_engines:
        if engine in valid_set:
            engines_to_run.add(engine)

    has_discontinuities = char_result.has_steps or char_result.has_impulses

    if has_discontinuities:
        for engine in discontinuity_engines:
            if engine in valid_set:
                engines_to_run.add(engine)

    return engines_to_run, has_discontinuities


def get_characterization_summary(char_result: CharacterizationResult) -> Dict[str, Any]:
    """
    Get a summary dict from characterization result for logging/storage.
    """
    return {
        'signal_id': char_result.signal_id,
        'dynamical_class': char_result.dynamical_class,
        'ax_stationarity': char_result.ax_stationarity,
        'ax_memory': char_result.ax_memory,
        'ax_periodicity': char_result.ax_periodicity,
        'ax_complexity': char_result.ax_complexity,
        'ax_determinism': char_result.ax_determinism,
        'ax_volatility': char_result.ax_volatility,
        'valid_engines': char_result.valid_engines,
        'has_steps': char_result.has_steps,
        'has_impulses': char_result.has_impulses,
        'n_breaks': char_result.n_breaks,
        'return_method': char_result.return_method,
        'frequency': char_result.frequency,
        'is_step_function': char_result.is_step_function,
    }


# =============================================================================
# UNIFIED 6-AXIS SYSTEM (Signal Typology v2.0)
# =============================================================================
# New orthogonal axes for ORTHON framework Layer 1
#
# The 6 Axes:
#   1. Memory        - Temporal persistence (Hurst, ACF decay)
#   2. Periodicity   - Cyclical structure (FFT, wavelets)
#   3. Volatility    - Variance dynamics (GARCH, rolling std)
#   4. Discontinuity - Level shifts / Heaviside (PELT, CUSUM)
#   5. Impulsivity   - Shocks / Dirac (derivative spikes, kurtosis)
#   6. Complexity    - Predictability (entropy)


def compute_all_axes(
    signal: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute all 6 orthogonal axes for Signal Typology.

    This is the main computation function called by the typology orchestrator.

    Args:
        signal: 1D numpy array of observations
        config: Optional configuration dict

    Returns:
        Dict with axis scores [0,1] and optional event details
    """
    signal = np.asarray(signal, dtype=float)
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < 30:
        return {
            'memory': np.nan,
            'periodicity': np.nan,
            'volatility': np.nan,
            'discontinuity': np.nan,
            'impulsivity': np.nan,
            'complexity': np.nan,
        }

    config = config or {}

    # Compute each axis
    memory = _compute_memory_axis(signal)
    periodicity = _compute_periodicity_axis(signal)
    volatility = _compute_volatility_axis(signal)
    discontinuity, disc_events = _compute_discontinuity_axis(signal)
    impulsivity, impulse_events = _compute_impulsivity_axis(signal)
    complexity = _compute_complexity_axis(signal)

    result = {
        'memory': memory,
        'periodicity': periodicity,
        'volatility': volatility,
        'discontinuity': discontinuity,
        'impulsivity': impulsivity,
        'complexity': complexity,
    }

    # Include event details if present
    if disc_events:
        result['discontinuity_events'] = disc_events
    if impulse_events:
        result['impulse_events'] = impulse_events

    return result


def _compute_memory_axis(signal: np.ndarray) -> float:
    """
    Compute Memory axis: temporal persistence / long-range dependence.

    Methods:
        - Hurst exponent via DFA
        - ACF decay rate

    Returns:
        Score [0,1] where 1 = strong memory/persistence
    """
    n = len(signal)
    if n < 50:
        return 0.5

    try:
        # Use existing Hurst computation (DFA)
        char = Characterizer()
        hurst, _ = char._compute_memory(signal)

        # Hurst > 0.5 = persistent, Hurst < 0.5 = anti-persistent
        # Map to [0,1] where 1 = strong memory
        # Anti-persistent (H < 0.5) = low memory, Persistent (H > 0.5) = high memory
        # H = 0.5 = random walk (some memory but no long-range)
        # Scale: H=0 -> 0, H=0.5 -> 0.5, H=1 -> 1
        memory_score = float(np.clip(hurst, 0, 1))

        return memory_score

    except Exception:
        return 0.5


def _compute_periodicity_axis(signal: np.ndarray) -> float:
    """
    Compute Periodicity axis: cyclical/seasonal structure.

    Methods:
        - FFT peak detection
        - Spectral entropy (low = periodic)

    Returns:
        Score [0,1] where 1 = strong periodicity
    """
    n = len(signal)
    if n < 64:
        return None

    try:
        # Use existing periodicity computation
        char = Characterizer()
        periodicity = char._compute_periodicity(signal)

        if periodicity is None:
            return None
        return float(np.clip(periodicity, 0, 1))

    except Exception:
        return None


def _compute_volatility_axis(signal: np.ndarray) -> float:
    """
    Compute Volatility axis: variance dynamics / heteroskedasticity.

    Methods:
        - Rolling std ratio
        - GARCH-like clustering detection

    Returns:
        Score [0,1] where 1 = high volatility clustering
    """
    n = len(signal)
    if n < 30:
        return None

    try:
        # Use existing volatility computation
        char = Characterizer()
        volatility = char._compute_volatility(signal)

        if volatility is None:
            return None
        return float(np.clip(volatility, 0, 1))

    except Exception:
        return None


def _compute_discontinuity_axis(signal: np.ndarray) -> Tuple[float, List[Dict]]:
    """
    Compute Discontinuity axis: level shifts / structural breaks (Heaviside).

    Methods:
        - CUSUM changepoint detection
        - Level shift detection
        - Magnitude analysis

    Returns:
        Tuple of (score [0,1], list of event details)
    """
    n = len(signal)
    events = []

    if n < 50:
        return None, events

    try:
        # Simple CUSUM-based detection
        mean_val = np.mean(signal)
        std_val = np.std(signal)

        if std_val == 0:
            return None, events

        # Standardize
        standardized = (signal - mean_val) / std_val

        # CUSUM
        cusum = np.cumsum(standardized)

        # Find significant deviations (potential changepoints)
        threshold = np.sqrt(n) * 1.5  # Heuristic threshold

        # Detect level shifts by finding where CUSUM changes direction significantly
        d_cusum = np.diff(cusum)
        abs_d_cusum = np.abs(d_cusum)

        # Find peaks in CUSUM derivative (potential changepoints)
        mean_deriv = np.mean(abs_d_cusum)
        std_deriv = np.std(abs_d_cusum)

        if std_deriv > 0:
            z_deriv = (abs_d_cusum - mean_deriv) / std_deriv
            changepoints = np.where(z_deriv > 3.0)[0]  # 3 sigma threshold
        else:
            changepoints = np.array([])

        n_changes = len(changepoints)

        # Also check for level shifts via segment comparison
        n_segments = 4
        segment_size = n // n_segments
        segment_means = []

        for i in range(n_segments):
            start = i * segment_size
            end = (i + 1) * segment_size if i < n_segments - 1 else n
            segment_means.append(np.mean(signal[start:end]))

        # Check for significant mean shifts between segments
        mean_shifts = np.abs(np.diff(segment_means))
        significant_shifts = np.sum(mean_shifts > std_val)

        # Combine metrics
        # More changepoints or significant segment shifts = higher discontinuity
        change_ratio = min(n_changes / 10, 1.0)  # Cap at 10 changes
        shift_ratio = significant_shifts / (n_segments - 1)

        discontinuity_score = 0.6 * change_ratio + 0.4 * shift_ratio

        # Build event list
        for idx in changepoints[:10]:  # Cap at 10 events
            events.append({
                'type': 'level_shift',
                'index': int(idx),
                'magnitude': float(abs_d_cusum[idx]) if idx < len(abs_d_cusum) else 0.0,
            })

        return float(np.clip(discontinuity_score, 0, 1)), events

    except Exception:
        return None, events


def _compute_impulsivity_axis(signal: np.ndarray) -> Tuple[float, List[Dict]]:
    """
    Compute Impulsivity axis: shocks / spikes / impulse events (Dirac).

    Methods:
        - Derivative magnitude spikes
        - Kurtosis (heavy tails)
        - Isolated outlier detection

    Returns:
        Tuple of (score [0,1], list of event details)
    """
    n = len(signal)
    events = []

    if n < 30:
        return None, events

    try:
        # First derivative (velocity)
        d1 = np.diff(signal)

        if len(d1) == 0:
            return None, events

        # Detect spikes in derivative
        d1_mean = np.mean(np.abs(d1))
        d1_std = np.std(d1)

        if d1_std == 0:
            return None, events

        # Z-scores of derivative
        d1_z = np.abs(d1 - np.mean(d1)) / d1_std

        # Impulses are isolated spikes (not sustained changes)
        spike_threshold = 3.0  # 3 sigma
        spike_indices = np.where(d1_z > spike_threshold)[0]

        # Filter to isolated spikes (not consecutive)
        isolated_spikes = []
        for idx in spike_indices:
            # Check if isolated (not part of sustained move)
            if idx > 0 and idx < len(d1_z) - 1:
                is_isolated = d1_z[idx - 1] < spike_threshold and d1_z[idx + 1] < spike_threshold
                if is_isolated or idx == 0 or idx == len(d1_z) - 1:
                    isolated_spikes.append(idx)
            elif idx == 0 or idx == len(d1_z) - 1:
                isolated_spikes.append(idx)

        n_spikes = len(isolated_spikes)

        # Kurtosis (heavy tails indicate impulse-prone)
        kurtosis = stats.kurtosis(signal)
        kurtosis_score = np.clip((kurtosis - 3) / 10, 0, 1)  # Excess kurtosis, scaled

        # Combine metrics
        spike_ratio = min(n_spikes / 5, 1.0)  # Cap at 5 spikes
        impulsivity_score = 0.5 * spike_ratio + 0.5 * kurtosis_score

        # Build event list
        for idx in isolated_spikes[:10]:
            events.append({
                'type': 'impulse',
                'index': int(idx),
                'magnitude': float(d1[idx]) if idx < len(d1) else 0.0,
                'z_score': float(d1_z[idx]) if idx < len(d1_z) else 0.0,
            })

        return float(np.clip(impulsivity_score, 0, 1)), events

    except Exception:
        return None, events


def _compute_complexity_axis(signal: np.ndarray) -> float:
    """
    Compute Complexity axis: predictability / information content.

    Methods:
        - Permutation entropy
        - Sample entropy
        - Lempel-Ziv complexity

    Returns:
        Score [0,1] where 1 = high complexity (unpredictable)
    """
    n = len(signal)
    if n < 50:
        return 0.5

    try:
        # Use existing complexity computation
        char = Characterizer()
        complexity = char._compute_complexity(signal)

        return float(np.clip(complexity, 0, 1))

    except Exception:
        return 0.5
