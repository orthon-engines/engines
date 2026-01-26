"""
PRISM Hurst Exponent Engine

Measures long-term memory and persistence in signal topology.

Measures:
- Hurst exponent (H)
  - H > 0.5: Trending/persistent (momentum)
  - H = 0.5: Random walk
  - H < 0.5: Mean-reverting (anti-persistent)

Phase: Unbound
Normalization: None (works on levels)

Performance: Uses numba JIT compilation for 5-15x speedup on R/S analysis.
"""

import logging
import os
from typing import Dict, Any, Optional, Tuple
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

VERBOSE = os.getenv('PRISM_VERBOSE', '0') == '1'

from prism.engines.engine_base import BaseEngine
from prism.engines.metadata import EngineMetadata

# Numba JIT compilation for performance-critical loops
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


logger = logging.getLogger(__name__)


METADATA = EngineMetadata(
    name="hurst",
    engine_type="vector",
    description="Hurst exponent for long-term memory and persistence",
    domains={"signal_topology", "persistence"},
    requires_window=True,
    deterministic=True,
)


# =============================================================================
# Vector Engine Contract: Simple function interface
# =============================================================================

def compute_hurst(values: np.ndarray) -> dict:
    """
    Measure Hurst exponent of a single signal.

    Args:
        values: Array of observed values (native sampling)

    Returns:
        Dict of metric_name -> metric_value
        Includes confidence metrics: r2, p_value, std_err, intercept
    """
    if len(values) < 20:
        return {}

    try:
        # Use n//3 for max_window to support shorter series (63+ obs)
        h, confidence = _compute_hurst_rs(values, min_window=10, max_window=len(values) // 3)

        result = {
            'hurst_exponent': float(h),
            'persistence': 1.0 if h > 0.5 else -1.0 if h < 0.5 else 0.0,
            # Confidence metrics from log-log regression
            'hurst_r2': confidence['r2'],
            'hurst_p_value': confidence['p_value'],
            'hurst_std_err': confidence['std_err'],
            'hurst_intercept': confidence['intercept'],
        }

        if VERBOSE:
            result['hurst_n_points'] = confidence['n_points']

        return result
    except Exception:
        return {}


def compute_hurst_with_derivation(
    values: np.ndarray,
    signal_id: str = "unknown",
    window_id: str = "0",
    window_start: str = None,
    window_end: str = None,
) -> tuple:
    """
    Compute Hurst exponent with full mathematical derivation.

    Returns:
        tuple: (result_dict, Derivation object)
    """
    from prism.entry_points.derivations.base import Derivation

    deriv = Derivation(
        engine_name="hurst_exponent",
        method_name="Rescaled Range (R/S) Analysis",
        signal_id=signal_id,
        window_id=window_id,
        window_start=window_start,
        window_end=window_end,
        sample_size=len(values),
        raw_data_sample=values[:10].tolist() if len(values) >= 10 else values.tolist(),
    )

    if len(values) < 20:
        deriv.final_result = None
        deriv.interpretation = "Insufficient data (n < 20)"
        return {}, deriv

    n = len(values)
    series = values.astype(np.float64)

    # Step 1: Overview
    deriv.add_step(
        title="Input Data Summary",
        equation="X = {x₁, x₂, ..., xₙ}",
        calculation=f"n = {n}\nRange: [{np.min(series):.4f}, {np.max(series):.4f}]\nMean: {np.mean(series):.4f}\nStd: {np.std(series):.4f}",
        result=n,
        result_name="n",
        notes="The series must have sufficient length for R/S analysis"
    )

    # Step 2: Generate window sizes
    min_window = 10
    max_window = n // 3
    window_sizes = []
    size = min_window
    while size <= min(max_window, n // 2):
        window_sizes.append(size)
        size = int(size * 1.3) if size < 20 else int(size * 1.5)

    deriv.add_step(
        title="Generate Window Sizes (Logarithmically Spaced)",
        equation="sizes = {s₁, s₂, ..., sₖ} where sᵢ₊₁ ≈ 1.5 × sᵢ",
        calculation=f"min_window = {min_window}\nmax_window = {max_window}\nwindow_sizes = {window_sizes}",
        result=window_sizes,
        result_name="window_sizes",
        notes=f"Using {len(window_sizes)} window sizes for regression"
    )

    # Step 3-5: Detailed R/S calculation for first window size
    first_window = window_sizes[0]
    first_series = series[:first_window]
    first_mean = np.mean(first_series)

    deriv.add_step(
        title=f"Example: R/S Calculation for Window Size {first_window}",
        equation="For each window of size s, compute R/S",
        calculation=f"First window: x[0:{first_window}]\nValues: [{first_series[0]:.4f}, {first_series[1]:.4f}, {first_series[2]:.4f}, ...]",
        result=first_window,
        result_name="s",
        notes="Showing detailed calculation for first window"
    )

    # Step 3a: Mean
    deriv.add_step(
        title="Compute Window Mean",
        equation="x̄ = (1/s) Σᵢ xᵢ",
        calculation=f"x̄ = ({first_series[0]:.4f} + {first_series[1]:.4f} + ... + x[{first_window-1}]) / {first_window}\nx̄ = {np.sum(first_series):.4f} / {first_window}",
        result=first_mean,
        result_name="x̄"
    )

    # Step 3b: Mean-adjusted series
    first_deviations = first_series - first_mean
    deriv.add_step(
        title="Compute Mean-Adjusted Series",
        equation="yᵢ = xᵢ - x̄",
        calculation=f"y₀ = {first_series[0]:.4f} - {first_mean:.4f} = {first_deviations[0]:.4f}\ny₁ = {first_series[1]:.4f} - {first_mean:.4f} = {first_deviations[1]:.4f}\ny₂ = {first_series[2]:.4f} - {first_mean:.4f} = {first_deviations[2]:.4f}\n⋮",
        result=first_deviations[:5].tolist(),
        result_name="y"
    )

    # Step 3c: Cumulative deviations
    first_cumsum = np.cumsum(first_deviations)
    deriv.add_step(
        title="Compute Cumulative Deviation Series",
        equation="Zₖ = Σᵢ₌₀ᵏ yᵢ",
        calculation=f"Z₀ = {first_deviations[0]:.4f}\nZ₁ = {first_cumsum[0]:.4f} + {first_deviations[1]:.4f} = {first_cumsum[1]:.4f}\nZ₂ = {first_cumsum[1]:.4f} + {first_deviations[2]:.4f} = {first_cumsum[2]:.4f}\n⋮\nZ_max = {np.max(first_cumsum):.4f}\nZ_min = {np.min(first_cumsum):.4f}",
        result=first_cumsum[-1],
        result_name="Z_s",
        notes="By construction, Z_s ≈ 0 (deviations sum to zero)"
    )

    # Step 3d: Range
    first_R = np.max(first_cumsum) - np.min(first_cumsum)
    deriv.add_step(
        title="Compute the Range",
        equation="R = max(Z) - min(Z)",
        calculation=f"R = {np.max(first_cumsum):.4f} - ({np.min(first_cumsum):.4f})\nR = {first_R:.4f}",
        result=first_R,
        result_name="R"
    )

    # Step 3e: Standard deviation
    first_S = np.std(first_series, ddof=1)
    deriv.add_step(
        title="Compute Standard Deviation",
        equation="S = √[(1/(s-1)) Σᵢ (xᵢ - x̄)²]",
        calculation=f"S = √[({first_deviations[0]:.4f})² + ({first_deviations[1]:.4f})² + ... ] / ({first_window}-1)\nS = √[{np.sum(first_deviations**2):.4f} / {first_window-1}]",
        result=first_S,
        result_name="S"
    )

    # Step 3f: Rescaled range for first window
    first_RS = first_R / first_S if first_S > 0 else 0
    deriv.add_step(
        title="Compute Rescaled Range",
        equation="(R/S) = R / S",
        calculation=f"(R/S) = {first_R:.4f} / {first_S:.4f}",
        result=first_RS,
        result_name="(R/S)₁"
    )

    # Step 4: Compute R/S for all window sizes
    rs_values = []
    for window_size in window_sizes:
        mean_rs = _rs_for_window_size(series, window_size)
        if mean_rs > 0:
            rs_values.append((window_size, mean_rs))

    rs_table = "\n".join([f"  s={s:4d}: (R/S) = {rs:.4f}" for s, rs in rs_values[:5]])
    if len(rs_values) > 5:
        rs_table += "\n  ⋮"
        rs_table += f"\n  s={rs_values[-1][0]:4d}: (R/S) = {rs_values[-1][1]:.4f}"

    deriv.add_step(
        title="Compute Mean R/S for All Window Sizes",
        equation="For each window size s, average R/S across all non-overlapping windows",
        calculation=rs_table,
        result=len(rs_values),
        result_name="n_points",
        notes=f"Computed {len(rs_values)} (s, R/S) pairs for regression"
    )

    # Step 5: Log-log regression
    log_sizes = np.log(np.array([x[0] for x in rs_values]))
    log_rs = np.log(np.array([x[1] for x in rs_values]))
    slope, intercept = np.polyfit(log_sizes, log_rs, 1)

    deriv.add_step(
        title="Log-Log Regression",
        equation="log(R/S) = H × log(s) + c",
        calculation=f"log(s) = [{log_sizes[0]:.4f}, {log_sizes[1]:.4f}, ..., {log_sizes[-1]:.4f}]\nlog(R/S) = [{log_rs[0]:.4f}, {log_rs[1]:.4f}, ..., {log_rs[-1]:.4f}]\n\nLinear fit: y = {slope:.4f}x + {intercept:.4f}",
        result=slope,
        result_name="H",
        notes="The slope of the log-log regression is the Hurst exponent"
    )

    # Final result
    H = slope
    deriv.final_result = H
    deriv.prism_output = H  # Same computation

    # Interpretation
    if H > 1.0:
        interp = f"H = {H:.4f} > 1.0 indicates strong determinism (typical for deterministic chaos). Consider using DFA for more robust estimation."
    elif H > 0.55:
        interp = f"H = {H:.4f} > 0.5 indicates **persistent** (trending) behavior. Past increases predict future increases."
    elif H < 0.45:
        interp = f"H = {H:.4f} < 0.5 indicates **anti-persistent** (mean-reverting) behavior. Past increases predict future decreases."
    else:
        interp = f"H = {H:.4f} ≈ 0.5 indicates **random walk** behavior. No long-range memory."

    deriv.interpretation = interp

    result = {
        'hurst_exponent': float(H),
        'persistence': 1.0 if H > 0.5 else -1.0 if H < 0.5 else 0.0,
    }

    return result, deriv


@jit(nopython=True, cache=True)
def _rs_for_window_size(series: np.ndarray, window_size: int) -> float:
    """
    Numba-JIT compiled R/S calculation for a single window size.

    Returns mean R/S value across all non-overlapping windows.
    5-15x faster than pure Python for typical series lengths.
    """
    n = len(series)
    n_windows = n // window_size
    rs_sum = 0.0
    rs_count = 0

    for i in range(n_windows):
        start = i * window_size
        end = start + window_size

        # Compute mean
        window_sum = 0.0
        for j in range(start, end):
            window_sum += series[j]
        mean = window_sum / window_size

        # Compute cumulative deviations and range
        cumsum = 0.0
        cumsum_min = 0.0
        cumsum_max = 0.0
        for j in range(start, end):
            cumsum += series[j] - mean
            if cumsum < cumsum_min:
                cumsum_min = cumsum
            if cumsum > cumsum_max:
                cumsum_max = cumsum
        R = cumsum_max - cumsum_min

        # Compute std with ddof=1
        var_sum = 0.0
        for j in range(start, end):
            diff = series[j] - mean
            var_sum += diff * diff
        S = np.sqrt(var_sum / (window_size - 1))

        if S > 0:
            rs_sum += R / S
            rs_count += 1

    if rs_count == 0:
        return None
    return rs_sum / rs_count


def _compute_hurst_rs(series: np.ndarray, min_window: int, max_window: int) -> Tuple[float, Dict[str, Any]]:
    """
    Compute Hurst exponent using Rescaled Range (R/S) analysis.

    Delegates inner loop to numba-compiled function for performance.

    Returns:
        Tuple of (hurst_exponent, confidence_metrics)
    """
    n = len(series)
    series = series.astype(np.float64)

    # Generate window sizes (logarithmically spaced)
    # Use 1.3 growth factor for better support of shorter series
    window_sizes = []
    size = min_window
    while size <= min(max_window, n // 2):
        window_sizes.append(size)
        size = int(size * 1.3) if size < 20 else int(size * 1.5)

    if len(window_sizes) < 3:
        raise ValueError("Not enough window sizes for R/S analysis")

    rs_values = []

    for window_size in window_sizes:
        mean_rs = _rs_for_window_size(series, window_size)
        if mean_rs > 0:
            rs_values.append((window_size, mean_rs))

    if len(rs_values) < 3:
        raise ValueError("Not enough valid R/S values")

    log_sizes = np.log(np.array([x[0] for x in rs_values]))
    log_rs = np.log(np.array([x[1] for x in rs_values]))

    # Use scipy.stats.linregress for full regression statistics
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_rs)

    confidence = {
        'r2': float(r_value ** 2),
        'p_value': float(p_value),
        'std_err': float(std_err),
        'intercept': float(intercept),
        'n_points': len(rs_values),
    }

    return slope, confidence


