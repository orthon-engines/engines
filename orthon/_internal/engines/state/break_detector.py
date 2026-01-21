"""
PRISM Break Detector Engine
===========================

Detects regime breaks at OBSERVATION level - no windowing required.
Point precision. Runs FIRST in characterize pipeline.

This engine identifies structural breaks in signal topology using multiple
detection methods, then analyzes break patterns (periodic, accelerating,
irregular) to inform adaptive windowing.

WHAT IT DETECTS:
    - Level shifts (Heaviside-like)
    - Volatility breaks  
    - Trend changes
    - Shock events (Dirac-like)

OUTPUTS:
    Per-observation:
        - break_flag: 0/1 signal
        - break_zscore: magnitude of break
        - break_direction: +1 (up) / -1 (down) / 0 (none)
        - inter_break_interval: observations since last break
    
    Per-signal summary:
        - n_breaks: total breaks detected
        - break_rate: breaks per observation
        - mean_interval: average inter-break interval
        - interval_cv: coefficient of variation (low = periodic)
        - interval_trend: slope of intervals (negative = accelerating)
        - break_pattern: PERIODIC / ACCELERATING / DECELERATING / IRREGULAR
        - dominant_period: FFT-detected period (if periodic)

PHYSICS INTERPRETATION:
    - PERIODIC breaks: Cyclic process (maintenance, seasons)
    - ACCELERATING breaks: System degrading toward failure
    - IRREGULAR breaks: Exogenous shocks, random events

Usage:
    from orthon._internal.engines.break_detector import compute_breaks, analyze_break_pattern
    breaks = compute_breaks(values, threshold=3.0)
    pattern = analyze_break_pattern(breaks)

Author: PRISM Team
"""

import numpy as np
import polars as pl
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy.fft import fft, fftfreq
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    'zscore_threshold': 3.0,      # Standard deviations for break detection
    'rolling_window': 50,          # Lookback for rolling statistics
    'min_breaks_for_pattern': 5,   # Minimum breaks to analyze pattern
    'cusum_threshold': 4.0,        # CUSUM detection threshold
    'volatility_window': 20,       # Window for volatility break detection
    'min_interval_for_fft': 10,    # Minimum intervals for FFT analysis
}

MIN_OBSERVATIONS = 30  # Need at least this many points


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class BreakPoint:
    """Single break point."""
    index: int
    timestamp: Any  # Could be datetime, int cycle, etc.
    zscore: float
    direction: int  # +1, -1, 0
    method: str     # Which detector found it


@dataclass 
class BreakPattern:
    """Pattern analysis of breaks."""
    n_breaks: int
    break_rate: float
    mean_interval: float
    std_interval: float
    cv: float  # Coefficient of variation
    trend: float  # Slope of intervals
    pattern: str  # PERIODIC, ACCELERATING, DECELERATING, IRREGULAR
    dominant_period: Optional[float]
    dominant_power: Optional[float]
    fft_frequencies: Optional[np.ndarray]
    fft_power: Optional[np.ndarray]


@dataclass
class BreakResult:
    """Complete break detection result."""
    signal_id: str
    n_observations: int
    breaks: List[BreakPoint]
    pattern: BreakPattern
    break_flags: np.ndarray
    break_zscores: np.ndarray
    break_directions: np.ndarray
    inter_break_intervals: np.ndarray


# =============================================================================
# BREAK DETECTION METHODS
# =============================================================================

def detect_zscore_breaks(
    values: np.ndarray,
    threshold: float = 3.0,
    window: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect breaks using z-score of delta from previous value.
    
    Uses MAD (median absolute deviation) for robustness.
    
    Returns:
        flags: Binary break signals
        zscores: Z-score at each point
        directions: +1 (up), -1 (down), 0 (no break)
    """
    n = len(values)
    flags = np.zeros(n, dtype=np.int8)
    zscores = np.zeros(n, dtype=np.float64)
    directions = np.zeros(n, dtype=np.int8)
    
    if n < window + 1:
        return flags, zscores, directions
    
    # Compute deltas (point-to-point changes)
    deltas = np.diff(values, prepend=values[0])
    
    # Use global statistics for stable reference
    # MAD is more robust to outliers than std
    median_delta = np.median(deltas[window:])  # Skip burn-in
    mad = np.median(np.abs(deltas[window:] - median_delta))
    
    # Convert MAD to std-equivalent (for normal distribution)
    robust_std = 1.4826 * mad
    
    if robust_std < 1e-10:
        # Fallback to regular std if MAD is zero
        robust_std = np.std(deltas[window:])
    
    if robust_std < 1e-10:
        return flags, zscores, directions
    
    # Detect breaks using global scale
    for i in range(window, n):
        z = (deltas[i] - median_delta) / robust_std
        zscores[i] = z
        
        if abs(z) > threshold:
            flags[i] = 1
            directions[i] = 1 if z > 0 else -1
    
    # Debounce: merge breaks that are too close (within 3 points)
    min_gap = 3
    last_break = -min_gap - 1
    for i in range(n):
        if flags[i] == 1:
            if i - last_break <= min_gap:
                # Keep the larger break
                if abs(zscores[i]) > abs(zscores[last_break]):
                    flags[last_break] = 0
                    last_break = i
                else:
                    flags[i] = 0
            else:
                last_break = i
    
    return flags, zscores, directions


def detect_cusum_breaks(
    values: np.ndarray,
    threshold: float = 4.0,
) -> np.ndarray:
    """
    CUSUM (Cumulative Sum) break detection.
    Detects persistent shifts in mean.
    
    Returns:
        flags: Binary break signals
    """
    n = len(values)
    flags = np.zeros(n, dtype=np.int8)
    
    if n < 20:
        return flags
    
    # Standardize
    mean = np.mean(values)
    std = np.std(values)
    if std < 1e-10:
        return flags
    
    z = (values - mean) / std
    
    # Two-sided CUSUM
    cusum_pos = np.zeros(n)
    cusum_neg = np.zeros(n)
    
    for i in range(1, n):
        cusum_pos[i] = max(0, cusum_pos[i-1] + z[i] - 0.5)
        cusum_neg[i] = max(0, cusum_neg[i-1] - z[i] - 0.5)
        
        # Detect break
        if cusum_pos[i] > threshold or cusum_neg[i] > threshold:
            flags[i] = 1
            # Reset CUSUM after break
            cusum_pos[i] = 0
            cusum_neg[i] = 0
    
    return flags


def detect_volatility_breaks(
    values: np.ndarray,
    window: int = 20,
    threshold: float = 2.0,
) -> np.ndarray:
    """
    Detect breaks in volatility regime.
    
    Returns:
        flags: Binary break signals
    """
    n = len(values)
    flags = np.zeros(n, dtype=np.int8)
    
    if n < 2 * window:
        return flags
    
    # Compute rolling volatility
    returns = np.diff(values, prepend=values[0])
    
    for i in range(2 * window, n):
        vol_recent = np.std(returns[i-window:i])
        vol_prior = np.std(returns[i-2*window:i-window])
        
        if vol_prior > 1e-10:
            ratio = vol_recent / vol_prior
            
            # Significant change in volatility
            if ratio > threshold or ratio < 1/threshold:
                flags[i] = 1
    
    return flags


def detect_trend_breaks(
    values: np.ndarray,
    window: int = 30,
    threshold: float = 2.0,
) -> np.ndarray:
    """
    Detect breaks in trend direction.
    Uses change in rolling regression slope.
    
    Returns:
        flags: Binary break signals
    """
    n = len(values)
    flags = np.zeros(n, dtype=np.int8)
    
    if n < 2 * window:
        return flags
    
    x = np.arange(window)
    slopes = np.zeros(n)
    
    # Compute rolling slopes
    for i in range(window, n):
        y = values[i-window:i]
        slope, _ = np.polyfit(x, y, 1)
        slopes[i] = slope
    
    # Detect slope changes
    slope_std = np.std(slopes[window:])
    if slope_std > 1e-10:
        for i in range(window + 1, n):
            slope_change = abs(slopes[i] - slopes[i-1])
            if slope_change > threshold * slope_std:
                flags[i] = 1
    
    return flags


# =============================================================================
# COMBINED BREAK DETECTION
# =============================================================================

def compute_breaks(
    values: np.ndarray,
    config: Optional[Dict] = None,
    timestamps: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Comprehensive break detection using multiple methods.
    
    Args:
        values: Signal values
        config: Configuration dict (uses DEFAULT_CONFIG if None)
        timestamps: Optional timestamps for each observation
    
    Returns:
        Dict with break detection results:
            - break_flags: Binary signals
            - break_zscores: Z-scores
            - break_directions: Directions
            - inter_break_intervals: Intervals
            - break_indices: Indices where breaks occurred
            - n_breaks: Count
            - methods: Which methods detected each break
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    n = len(values)
    
    if n < MIN_OBSERVATIONS:
        return {
            'break_flags': np.zeros(n, dtype=np.int8),
            'break_zscores': np.zeros(n),
            'break_directions': np.zeros(n, dtype=np.int8),
            'inter_break_intervals': np.arange(n),
            'break_indices': np.array([], dtype=np.int64),
            'n_breaks': 0,
            'methods': {},
        }
    
    # Run all detectors
    zscore_flags, zscores, directions = detect_zscore_breaks(
        values,
        threshold=config['zscore_threshold'],
        window=config['rolling_window'],
    )
    
    cusum_flags = detect_cusum_breaks(
        values,
        threshold=config['cusum_threshold'],
    )
    
    vol_flags = detect_volatility_breaks(
        values,
        window=config['volatility_window'],
    )
    
    trend_flags = detect_trend_breaks(values)
    
    # Smarter combination: use z-score as primary, others as confirmation
    # A break is detected if:
    #   - zscore detects it (primary)
    #   - OR at least 2 other methods agree (consensus)
    
    other_count = cusum_flags.astype(int) + vol_flags.astype(int) + trend_flags.astype(int)
    
    combined_flags = np.where(
        (zscore_flags == 1) | (other_count >= 2),
        1, 0
    ).astype(np.int8)
    
    # Track which methods detected each break
    methods = {}
    break_indices = np.where(combined_flags == 1)[0]
    
    for idx in break_indices:
        detected_by = []
        if zscore_flags[idx]: detected_by.append('zscore')
        if cusum_flags[idx]: detected_by.append('cusum')
        if vol_flags[idx]: detected_by.append('volatility')
        if trend_flags[idx]: detected_by.append('trend')
        methods[idx] = detected_by
    
    # Compute inter-break intervals
    inter_break_intervals = np.zeros(n, dtype=np.int64)
    last_break = -1
    
    for i in range(n):
        if combined_flags[i] == 1:
            inter_break_intervals[i] = i - last_break - 1 if last_break >= 0 else i
            last_break = i
        else:
            inter_break_intervals[i] = i - last_break if last_break >= 0 else i
    
    return {
        'break_flags': combined_flags,
        'break_zscores': zscores,
        'break_directions': directions,
        'inter_break_intervals': inter_break_intervals,
        'break_indices': break_indices,
        'n_breaks': len(break_indices),
        'methods': methods,
        'timestamps': timestamps,
    }


# =============================================================================
# BREAK PATTERN ANALYSIS (FOURIER)
# =============================================================================

def analyze_break_pattern(
    break_result: Dict[str, Any],
    config: Optional[Dict] = None,
) -> BreakPattern:
    """
    Analyze the pattern of breaks using statistics and FFT.
    
    Determines if breaks are:
        - PERIODIC: Regular intervals (cyclic process)
        - ACCELERATING: Intervals decreasing (degradation)
        - DECELERATING: Intervals increasing (stabilization)
        - IRREGULAR: No clear pattern (random shocks)
    
    Args:
        break_result: Output from compute_breaks()
        config: Configuration dict
    
    Returns:
        BreakPattern with statistics and classification
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    break_indices = break_result['break_indices']
    n_breaks = len(break_indices)
    n_obs = len(break_result['break_flags'])
    
    # Handle insufficient breaks
    if n_breaks < 2:
        return BreakPattern(
            n_breaks=n_breaks,
            break_rate=n_breaks / n_obs if n_obs > 0 else 0,
            mean_interval=n_obs if n_breaks <= 1 else 0,
            std_interval=0,
            cv=float('inf'),
            trend=0,
            pattern='INSUFFICIENT_DATA',
            dominant_period=None,
            dominant_power=None,
            fft_frequencies=None,
            fft_power=None,
        )
    
    # Compute inter-break intervals
    intervals = np.diff(break_indices).astype(float)
    
    if len(intervals) < 2:
        return BreakPattern(
            n_breaks=n_breaks,
            break_rate=n_breaks / n_obs,
            mean_interval=intervals[0] if len(intervals) == 1 else 0,
            std_interval=0,
            cv=float('inf'),
            trend=0,
            pattern='INSUFFICIENT_DATA',
            dominant_period=None,
            dominant_power=None,
            fft_frequencies=None,
            fft_power=None,
        )
    
    # Basic statistics
    mean_interval = float(np.mean(intervals))
    std_interval = float(np.std(intervals))
    cv = std_interval / mean_interval if mean_interval > 0 else float('inf')
    
    # Trend in intervals (are breaks accelerating?)
    x = np.arange(len(intervals))
    slope, intercept = np.polyfit(x, intervals, 1)
    trend = float(slope)
    
    # FFT analysis (if enough intervals)
    dominant_period = None
    dominant_power = None
    fft_frequencies = None
    fft_power = None
    
    if len(intervals) >= config['min_interval_for_fft']:
        # FFT on intervals
        intervals_centered = intervals - np.mean(intervals)
        n_fft = len(intervals_centered)
        
        fft_result = fft(intervals_centered)
        freqs = fftfreq(n_fft, d=1.0)  # d=1 means per-interval
        
        # Get positive frequencies
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_power = np.abs(fft_result[pos_mask]) ** 2
        
        if len(pos_power) > 0:
            fft_frequencies = pos_freqs
            fft_power = pos_power
            
            # Dominant frequency
            peak_idx = np.argmax(pos_power)
            dominant_freq = pos_freqs[peak_idx]
            dominant_power = float(pos_power[peak_idx] / np.sum(pos_power))
            
            if dominant_freq > 0:
                dominant_period = float(1.0 / dominant_freq)
    
    # Classify pattern
    pattern = _classify_pattern(cv, trend, mean_interval, dominant_power)
    
    return BreakPattern(
        n_breaks=n_breaks,
        break_rate=n_breaks / n_obs,
        mean_interval=mean_interval,
        std_interval=std_interval,
        cv=cv,
        trend=trend,
        pattern=pattern,
        dominant_period=dominant_period,
        dominant_power=dominant_power,
        fft_frequencies=fft_frequencies,
        fft_power=fft_power,
    )


def _classify_pattern(
    cv: float,
    trend: float,
    mean_interval: float,
    dominant_power: Optional[float],
) -> str:
    """
    Classify break pattern based on statistics.
    
    Logic:
        - CV < 0.3 AND dominant_power > 0.3: PERIODIC
        - trend < -0.1 * mean_interval: ACCELERATING (breaks getting closer)
        - trend > 0.1 * mean_interval: DECELERATING (breaks spreading out)
        - else: IRREGULAR
    """
    # Check for periodicity first
    if cv < 0.3 and dominant_power is not None and dominant_power > 0.3:
        return 'PERIODIC'
    
    # Check for acceleration/deceleration
    trend_threshold = 0.1 * mean_interval if mean_interval > 0 else 1
    
    if trend < -trend_threshold:
        return 'ACCELERATING'
    elif trend > trend_threshold:
        return 'DECELERATING'
    else:
        return 'IRREGULAR'


# =============================================================================
# POLARS INTERFACE (for pipeline integration)
# =============================================================================

def compute_breaks_polars(
    df: pl.DataFrame,
    value_col: str = 'value',
    signal_col: str = 'signal_id',
    time_col: str = 'observed_at',
    config: Optional[Dict] = None,
) -> pl.DataFrame:
    """
    Compute breaks for all signals in a Polars DataFrame.
    
    Args:
        df: DataFrame with observations
        value_col: Column containing values
        signal_col: Column containing signal IDs
        time_col: Column containing timestamps
        config: Configuration dict
    
    Returns:
        DataFrame with break detection results per observation
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    results = []
    
    # Process each signal
    for signal_id in df.select(signal_col).unique().to_series():
        ind_df = df.filter(pl.col(signal_col) == signal_id).sort(time_col)
        
        values = ind_df.select(value_col).to_series().to_numpy()
        timestamps = ind_df.select(time_col).to_series().to_numpy()
        
        # Detect breaks
        break_result = compute_breaks(values, config, timestamps)
        
        # Build result DataFrame
        n = len(values)
        result_df = pl.DataFrame({
            signal_col: [signal_id] * n,
            time_col: timestamps,
            value_col: values,
            'break_flag': break_result['break_flags'],
            'break_zscore': break_result['break_zscores'],
            'break_direction': break_result['break_directions'],
            'inter_break_interval': break_result['inter_break_intervals'],
        })
        
        results.append(result_df)
    
    return pl.concat(results) if results else pl.DataFrame()


def compute_break_summary_polars(
    break_df: pl.DataFrame,
    signal_col: str = 'signal_id',
) -> pl.DataFrame:
    """
    Compute break pattern summary for each signal.
    
    Args:
        break_df: DataFrame from compute_breaks_polars()
        signal_col: Column containing signal IDs
    
    Returns:
        DataFrame with one row per signal containing pattern analysis
    """
    summaries = []
    
    for signal_id in break_df.select(signal_col).unique().to_series():
        ind_df = break_df.filter(pl.col(signal_col) == signal_id)
        
        # Extract break result
        break_result = {
            'break_flags': ind_df.select('break_flag').to_series().to_numpy(),
            'break_indices': np.where(ind_df.select('break_flag').to_series().to_numpy() == 1)[0],
        }
        
        # Analyze pattern
        pattern = analyze_break_pattern(break_result)
        
        summaries.append({
            signal_col: signal_id,
            'n_observations': len(ind_df),
            'n_breaks': pattern.n_breaks,
            'break_rate': pattern.break_rate,
            'mean_interval': pattern.mean_interval,
            'std_interval': pattern.std_interval,
            'interval_cv': pattern.cv,
            'interval_trend': pattern.trend,
            'break_pattern': pattern.pattern,
            'dominant_period': pattern.dominant_period,
            'dominant_power': pattern.dominant_power,
        })
    
    return pl.DataFrame(summaries)


# =============================================================================
# ADAPTIVE WINDOWING HELPER
# =============================================================================

def create_adaptive_windows(
    df: pl.DataFrame,
    break_df: pl.DataFrame,
    signal_col: str = 'signal_id',
    time_col: str = 'observed_at',
) -> pl.DataFrame:
    """
    Create adaptive windows based on detected breaks.
    
    Window boundaries align to break points.
    Each window represents a homogeneous regime.
    
    Args:
        df: Original observations DataFrame
        break_df: DataFrame from compute_breaks_polars()
        signal_col: Column containing signal IDs
        time_col: Column containing timestamps
    
    Returns:
        DataFrame with window_id column added
    """
    results = []
    
    for signal_id in df.select(signal_col).unique().to_series():
        # Get original data
        ind_df = df.filter(pl.col(signal_col) == signal_id).sort(time_col)
        
        # Get breaks for this signal
        ind_breaks = break_df.filter(
            (pl.col(signal_col) == signal_id) &
            (pl.col('break_flag') == 1)
        ).sort(time_col)
        
        break_times = ind_breaks.select(time_col).to_series().to_list()
        
        # Assign window IDs
        n = len(ind_df)
        window_ids = np.zeros(n, dtype=np.int64)
        current_window = 0
        timestamps = ind_df.select(time_col).to_series().to_list()
        
        break_idx = 0
        for i, t in enumerate(timestamps):
            # Check if we've passed a break point
            while break_idx < len(break_times) and t >= break_times[break_idx]:
                current_window += 1
                break_idx += 1
            window_ids[i] = current_window
        
        # Add window_id to dataframe
        result_df = ind_df.with_columns(
            pl.Series('window_id', window_ids)
        )
        
        results.append(result_df)
    
    return pl.concat(results) if results else pl.DataFrame()


# =============================================================================
# DIRECTED COMPUTE HELPER
# =============================================================================

def identify_break_regions(
    break_df: pl.DataFrame,
    buffer: int = 20,
    signal_col: str = 'signal_id',
    time_col: str = 'observed_at',
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Separate observations into break regions and quiet regions.
    
    Break regions: +/- buffer around each break (full engine suite)
    Quiet regions: Everything else (light engine suite)
    
    Args:
        break_df: DataFrame from compute_breaks_polars()
        buffer: Observations before/after break to include
        signal_col: Column containing signal IDs
        time_col: Column containing timestamps
    
    Returns:
        Tuple of (break_regions_df, quiet_regions_df)
    """
    break_regions = []
    quiet_regions = []
    
    for signal_id in break_df.select(signal_col).unique().to_series():
        ind_df = break_df.filter(pl.col(signal_col) == signal_id).sort(time_col)
        
        # Get break indices
        break_mask = ind_df.select('break_flag').to_series().to_numpy() == 1
        break_indices = np.where(break_mask)[0]
        
        # Create region mask
        n = len(ind_df)
        in_break_region = np.zeros(n, dtype=bool)
        
        for idx in break_indices:
            start = max(0, idx - buffer)
            end = min(n, idx + buffer + 1)
            in_break_region[start:end] = True
        
        # Split
        break_region_df = ind_df.filter(pl.Series(in_break_region))
        quiet_region_df = ind_df.filter(pl.Series(~in_break_region))
        
        if len(break_region_df) > 0:
            break_regions.append(break_region_df)
        if len(quiet_region_df) > 0:
            quiet_regions.append(quiet_region_df)
    
    break_df_out = pl.concat(break_regions) if break_regions else pl.DataFrame()
    quiet_df_out = pl.concat(quiet_regions) if quiet_regions else pl.DataFrame()
    
    return break_df_out, quiet_df_out


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_signal_breaks(
    values: np.ndarray,
    signal_id: str = 'unknown',
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    One-shot break detection for a single signal.
    
    Returns dict with all break info suitable for storage.
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    # Detect breaks
    break_result = compute_breaks(values, config)
    
    # Analyze pattern
    pattern = analyze_break_pattern(break_result, config)
    
    return {
        'signal_id': signal_id,
        'n_observations': len(values),
        'n_breaks': pattern.n_breaks,
        'break_rate': pattern.break_rate,
        'mean_interval': pattern.mean_interval,
        'std_interval': pattern.std_interval,
        'interval_cv': pattern.cv,
        'interval_trend': pattern.trend,
        'break_pattern': pattern.pattern,
        'dominant_period': pattern.dominant_period,
        'dominant_power': pattern.dominant_power,
        'break_indices': break_result['break_indices'].tolist(),
        'break_flags': break_result['break_flags'],
        'break_zscores': break_result['break_zscores'],
        'break_directions': break_result['break_directions'],
        'inter_break_intervals': break_result['inter_break_intervals'],
    }


def get_break_metrics(
    values: np.ndarray,
    config: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Get break metrics suitable for inclusion in behavioral vector.
    
    These can be included alongside hurst, entropy, etc.
    """
    result = compute_signal_breaks(values, config=config)
    
    return {
        'break_n': float(result['n_breaks']),
        'break_rate': result['break_rate'],
        'break_mean_interval': result['mean_interval'],
        'break_interval_cv': result['interval_cv'] if np.isfinite(result['interval_cv']) else -1.0,
        'break_interval_trend': result['interval_trend'],
        'break_is_periodic': 1.0 if result['break_pattern'] == 'PERIODIC' else 0.0,
        'break_is_accelerating': 1.0 if result['break_pattern'] == 'ACCELERATING' else 0.0,
        'break_dominant_period': result['dominant_period'] if result['dominant_period'] else -1.0,
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PRISM Break Detector Engine')
    parser.add_argument('--test', action='store_true', help='Run tests')
    parser.add_argument('--demo', action='store_true', help='Run demo on synthetic data')
    
    args = parser.parse_args()
    
    if args.test or args.demo:
        print("=" * 70)
        print("PRISM Break Detector Engine - Demo")
        print("=" * 70)
        
        np.random.seed(42)
        
        # Use higher threshold for cleaner demonstration
        demo_config = DEFAULT_CONFIG.copy()
        demo_config['zscore_threshold'] = 5.0  # Stricter for demo
        demo_config['cusum_threshold'] = 6.0
        
        # Test 1: Periodic breaks (clear regime shifts)
        print("\n1. PERIODIC BREAKS (like maintenance cycles)")
        print("-" * 50)
        n = 500
        period = 50
        values = np.zeros(n)
        for i in range(n):
            regime = i // period
            # Clear level shifts with low noise
            values[i] = regime * 20 + np.random.randn() * 1.0
        
        result = compute_signal_breaks(values, 'periodic_test', demo_config)
        print(f"   N breaks: {result['n_breaks']} (expected ~10)")
        print(f"   Pattern: {result['break_pattern']}")
        print(f"   Mean interval: {result['mean_interval']:.1f}")
        print(f"   CV: {result['interval_cv']:.3f} (low = periodic)")
        if result['dominant_period']:
            print(f"   Dominant period: {result['dominant_period']:.1f} (true = 50)")
        
        # Test 2: Accelerating breaks (degradation toward failure)
        print("\n2. ACCELERATING BREAKS (like system degradation)")
        print("-" * 50)
        values = np.zeros(n)
        # Breaks get closer together: system degrading
        break_times = [50, 120, 180, 230, 270, 300, 320, 335, 345, 352, 357]
        intervals_true = np.diff(break_times)
        print(f"   True intervals: {intervals_true} (decreasing)")
        
        current = 100
        for i in range(n):
            if i in break_times:
                current += 25  # Level shift
            values[i] = current + np.random.randn() * 1.0
        
        result = compute_signal_breaks(values, 'accelerating_test', demo_config)
        print(f"   N breaks: {result['n_breaks']} (expected ~{len(break_times)})")
        print(f"   Pattern: {result['break_pattern']}")
        print(f"   Interval trend: {result['interval_trend']:.2f} (negative = accelerating)")
        
        # Test 3: Irregular breaks (random shocks, varying intervals)
        print("\n3. IRREGULAR BREAKS (like random shocks)")
        print("-" * 50)
        values = np.ones(n) * 100
        # Random shock times with varying magnitudes
        shock_times = sorted(np.random.choice(range(50, 450), 12, replace=False))
        for t in shock_times:
            values[t:] += np.random.choice([-1, 1]) * (15 + np.random.rand() * 10)
        values += np.random.randn(n) * 1.0
        
        result = compute_signal_breaks(values, 'irregular_test', demo_config)
        print(f"   N breaks: {result['n_breaks']}")
        print(f"   Pattern: {result['break_pattern']}")
        print(f"   CV: {result['interval_cv']:.3f} (high = irregular)")
        
        # Test 4: No breaks (stable system)
        print("\n4. NO BREAKS (stable system)")
        print("-" * 50)
        values = 100 + np.random.randn(n) * 2.0  # Just noise
        
        result = compute_signal_breaks(values, 'stable_test', demo_config)
        print(f"   N breaks: {result['n_breaks']} (expected: few/none)")
        print(f"   Pattern: {result['break_pattern']}")
        
        # Test 5: C-MAPSS-like degradation (trending + accelerating breaks)
        print("\n5. C-MAPSS-LIKE DEGRADATION (trend + accelerating breaks)")
        print("-" * 50)
        # Baseline trend (gradual degradation)
        values = np.linspace(100, 60, n)
        # Add accelerating break points (degradation events)
        break_times = [100, 200, 280, 340, 380, 410, 430, 445, 455, 462]
        for t in break_times:
            values[t:] -= 3  # Each break drops level
        values += np.random.randn(n) * 0.5
        
        result = compute_signal_breaks(values, 'cmapss_like', demo_config)
        print(f"   N breaks: {result['n_breaks']}")
        print(f"   Pattern: {result['break_pattern']}")
        print(f"   Interval trend: {result['interval_trend']:.2f}")
        print(f"   Break rate: {result['break_rate']:.4f}")
        
        # Show metrics format
        print("\n6. METRICS FOR BEHAVIORAL VECTOR")
        print("-" * 50)
        metrics = get_break_metrics(values, demo_config)
        for k, v in metrics.items():
            print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
        
        print("\n" + "=" * 70)
        print("Break Detector Engine Demo Complete")
        print("=" * 70)
