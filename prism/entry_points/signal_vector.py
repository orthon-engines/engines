"""
PRISM Vector Runner
===================

Computes behavioral vector metrics from observations using sliding windows.
Uses inline characterization to determine which engines to run per signal.

ENGINE CLASSIFICATION:
    CORE ENGINES (run on ALL signals):
        - hurst:        Long-range dependence and memory
        - entropy:      Information content and complexity
        - rqa:          Recurrence quantification (phase space)
        - realized_vol: Short-window volatility, drawdown, distribution

    CONDITIONAL ENGINES (run based on characterization):
        - spectral:     ax_periodicity > 0.3
        - wavelet:      ax_periodicity > 0.3
        - garch:        ax_volatility > 0.3
        - lyapunov:     ax_complexity > 0.3

    DISCONTINUITY ENGINES (run if breaks detected):
        - break_detector: Always runs to detect breaks
        - heaviside:      Runs if breaks found (persistent steps)
        - dirac:          Runs if breaks found (transient impulses)

PIPELINE:
    observations → characterize → signal_vector → laplace → geometry
                       ↓                 ↓
                 (valid_engines)   (uses valid_engines, chains to laplace)

Storage: Parquet files (no database locks)

Output: data/vector.parquet

Usage:
    # Production run
    python -m prism.entry_points.signal_vector

    # Force recompute
    python -m prism.entry_points.signal_vector --force

    # Adaptive windowing (auto-detect from data)
    python -m prism.entry_points.signal_vector --adaptive

    # Testing mode
    python -m prism.entry_points.signal_vector --testing --limit 100
    python -m prism.entry_points.signal_vector --testing --signal sensor_1,sensor_2
"""

import argparse
import gc
import json
import logging
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

from prism.db.parquet_store import (
    ensure_directory,
    get_data_root,
    get_path,
    OBSERVATIONS,
    VECTOR,
    GEOMETRY,
    STATE,
    COHORTS,
)
# Backwards compatibility
SIGNALS = VECTOR
from prism.db.polars_io import read_parquet, upsert_parquet, write_parquet_atomic
from prism.utils.memory import (
    force_gc,
    get_memory_usage_mb,
    get_size_mb,
    MemoryTracker,
)
from prism.db.scratch import TempParquet, merge_to_table
from prism.engines.utils.parallel import (
    WorkerAssignment,
    divide_by_count,
    generate_temp_path,
    run_workers,
)

# Window/stride configuration
from prism.utils.stride import load_stride_config, get_default_tiers

# Normalization configuration
import yaml

# Inline modules (characterize + laplace)
from prism.engines.characterize import (
    characterize_signal,
    get_engines_from_characterization,
    get_characterization_summary,
)
from prism.engines.laplace.transform import (
    compute_laplace_for_series,
    compute_divergence_for_signal,
    add_divergence_to_field_rows,
)

# V2 Architecture: Pointwise engines (native resolution)
from prism.engines.pointwise import (
    HilbertEngine,
    DerivativesEngine,
    StatisticalEngine,
)
from prism.core.signals.types import DenseSignal, SparseSignal
from prism.engines.laplace.transform import compute_laplace_field as compute_laplace_field_v2

# Domain clock for adaptive windowing
from prism.core.domain_clock import DomainClock, DomainInfo
from prism.config.loader import load_clock_config, load_delta_thresholds

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

MIN_OBSERVATIONS = 15  # Absolute floor for any computation

DEFAULT_ENGINE_MIN_OBS = {
    "hurst": 20,
    "entropy": 30,
    "lyapunov": 30,
    "garch": 50,
    "spectral": 40,
    "wavelet": 40,
    "rqa": 30,
    "realized_vol": 15,  # Short-window: vol, drawdown, distribution (13 metrics)
    "hilbert": 20,  # Hilbert transform: amplitude, phase, inst_freq (7 metrics)
    # Observation-level (discontinuity) engines
    "break_detector": 50,
    "heaviside": 50,
    "dirac": 50,
}

# Key columns for upsert deduplication (new 5-file schema)
# signals.parquet: entity_id, signal_id, source_signal, engine, signal_type, timestamp, value, mode_id
SIGNALS_KEY_COLS = ["entity_id", "signal_id", "timestamp"]

# Legacy key columns (deprecated)
VECTOR_KEY_COLS = ["signal_id", "timestamp", "target_obs", "engine", "metric_name"]


# =============================================================================
# REGIME-AWARE NORMALIZATION
# =============================================================================

def get_normalization_config() -> dict:
    """
    Load normalization parameters.

    Checks domain_info.json first (from adaptive DomainClock),
    then falls back to window.yaml defaults.

    Returns:
        Dict with 'window' and 'min_periods' keys
    """
    # First check for adaptive domain_info (from DomainClock)
    try:
        domain_info_path = get_data_root() / "domain_info.json"
        if domain_info_path.exists():
            import json
            with open(domain_info_path) as f:
                domain_info = json.load(f)
            window = domain_info.get('window_samples')
            if window:
                return {'window': window, 'min_periods': max(10, window // 10)}
    except Exception:
        pass

    # Check window.yaml for defaults
    config_path = Path(__file__).parent.parent.parent / 'config' / 'window.yaml'
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            defaults = config.get('default', {})
            if defaults:
                return {
                    'window': defaults.get('window_size', 252),
                    'min_periods': defaults.get('min_observations', 50),
                }
        except Exception:
            pass

    # Hardcoded fallback
    return {'window': 252, 'min_periods': 50}


def apply_regime_normalization(
    batch_rows: List[Dict],
    window: int,
    min_periods: int = 30,
) -> pl.DataFrame:
    """
    Standardizes signal behavior relative to trailing window.
    Ensures 'Source' detection is calibrated to current volatility regimes.

    New 5-file schema: signals.parquet uses
        entity_id, signal_id, source_signal, engine, signal_type, timestamp, value, mode_id

    Args:
        batch_rows: List of metric dicts from signal processing
        window: Rolling window size (REQUIRED - from domain config)
        min_periods: Minimum observations before computing (default 30)

    Returns:
        DataFrame with original value AND value_norm columns.
        value_norm will be NaN for first min_periods rows (burn-in).

    Philosophy:
        - NaN during burn-in is honest accounting, not a problem to solve
        - Rolling window preserves regime awareness
        - Both raw and normalized preserved for downstream choice
    """
    df = pl.DataFrame(batch_rows, infer_schema_length=None)

    if len(df) == 0:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias("value_norm"))

    # Sort by timestamp within each signal group (required for rolling)
    # In new schema, signal_id is unique per derived signal (source_engine_metric)
    df = df.sort(["signal_id", "timestamp"])

    # Apply rolling Z-score within each signal group
    # Z = (x - rolling_mean) / (rolling_std + epsilon)
    return df.with_columns([
        ((pl.col("value") - pl.col("value").rolling_mean(window, min_periods=min_periods)) /
         (pl.col("value").rolling_std(window, min_periods=min_periods) + 1e-10))
        .over(["signal_id"])
        .alias("value_norm")
    ])

# =============================================================================
# ENGINE CLASSIFICATION
# =============================================================================
# CORE engines run on ALL signals regardless of characterization
# CONDITIONAL engines only run if characterization indicates they're valid

CORE_ENGINES = {"hurst", "entropy", "rqa", "realized_vol", "hilbert"}

CONDITIONAL_ENGINES = {"spectral", "wavelet", "garch", "lyapunov"}

# Discontinuity engines: break_detector always runs, heaviside/dirac run if breaks found
DISCONTINUITY_ENGINES = {"break_detector", "heaviside", "dirac"}

# V2 Architecture: Pointwise engines (native resolution output)
POINTWISE_ENGINES = {"hilbert", "derivatives", "statistical"}


# =============================================================================
# VECTOR ENGINES
# =============================================================================

from prism.engines import VECTOR_ENGINES, OBSERVATION_ENGINES, compute_breaks


# =============================================================================
# V2 ARCHITECTURE: POINTWISE ENGINE COMPUTATION
# =============================================================================

def compute_pointwise_signals(
    signal_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
) -> List[DenseSignal]:
    """
    Compute all pointwise engine outputs for a signal.

    V2 Architecture: These engines produce values at every timestamp
    (native resolution), unlike windowed engines which produce sparse output.

    Args:
        signal_id: Signal identifier
        timestamps: Array of timestamps
        values: Array of signal values

    Returns:
        List of DenseSignal objects (amplitude, phase, freq, velocity, accel, etc.)
    """
    results = []

    if len(values) < 10:
        logger.debug(f"{signal_id}: Insufficient data for pointwise engines ({len(values)})")
        return results

    # 1. Hilbert Engine - instantaneous amplitude, phase, frequency
    try:
        hilbert = HilbertEngine()
        amp, phase, freq = hilbert.compute(signal_id, timestamps, values)
        results.extend([amp, phase, freq])
        logger.debug(f"{signal_id}: Hilbert computed 3 DenseSignals")
    except Exception as e:
        logger.debug(f"{signal_id}: Hilbert failed: {e}")

    # 2. Derivatives Engine - velocity, acceleration, jerk
    try:
        derivatives = DerivativesEngine()
        vel, accel, jerk = derivatives.compute(signal_id, timestamps, values)
        results.extend([vel, accel, jerk])
        logger.debug(f"{signal_id}: Derivatives computed 3 DenseSignals")
    except Exception as e:
        logger.debug(f"{signal_id}: Derivatives failed: {e}")

    # 3. Statistical Engine - zscore, rolling mean, rolling std
    try:
        statistical = StatisticalEngine()
        zscore, rmean, rstd = statistical.compute(signal_id, timestamps, values)
        results.extend([zscore, rmean, rstd])
        logger.debug(f"{signal_id}: Statistical computed 3 DenseSignals")
    except Exception as e:
        logger.debug(f"{signal_id}: Statistical failed: {e}")

    return results


def create_signal_row(
    entity_id: str,
    source_signal: str,
    engine: str,
    metric_name: str,
    timestamp: Any,
    value: float,
    signal_type: str = 'sparse',
) -> Dict[str, Any]:
    """
    Create a row in the new signals.parquet schema.

    New 5-file schema: signals.parquet
        entity_id, signal_id, source_signal, engine, signal_type, timestamp, value, mode_id

    Args:
        entity_id: The entity this signal belongs to
        source_signal: Original signal ID
        engine: Engine that produced this
        metric_name: Metric name
        timestamp: Time value
        value: Signal value
        signal_type: 'dense' or 'sparse'

    Returns:
        Row dictionary matching signals.parquet schema
    """
    # Convert timestamp to float if needed
    ts_value = timestamp
    if hasattr(timestamp, 'date'):
        # datetime/date object - convert to ordinal or timestamp
        import datetime
        if isinstance(timestamp, datetime.date):
            ts_value = float(timestamp.toordinal())
        else:
            ts_value = timestamp.timestamp()
    elif isinstance(timestamp, np.datetime64):
        # numpy.datetime64 - convert to float seconds
        import pandas as pd
        ts_value = float(pd.Timestamp(timestamp).timestamp())
    # numpy.float64 and other numeric types pass through as-is

    return {
        'entity_id': entity_id,
        'signal_id': f"{source_signal}_{engine}_{metric_name}",
        'source_signal': source_signal,
        'engine': engine,
        'signal_type': signal_type,
        'timestamp': float(ts_value) if isinstance(ts_value, (int, float)) else ts_value,
        'value': float(value),
        'mode_id': None,  # Assigned by geometry layer
    }


def dense_signals_to_rows(
    dense_signals: List[DenseSignal],
    entity_id: str,
    computed_at: datetime,
) -> List[Dict[str, Any]]:
    """
    Convert DenseSignal objects to row dictionaries for storage.

    New 5-file schema: signals.parquet
        entity_id, signal_id, source_signal, engine, signal_type, timestamp, value, mode_id

    Args:
        dense_signals: List of DenseSignal objects
        entity_id: The entity this signal belongs to
        computed_at: Computation timestamp

    Returns:
        List of row dictionaries for parquet storage
    """
    rows = []
    for signal in dense_signals:
        if not signal.is_valid:
            continue
        for i, (t, v) in enumerate(zip(signal.timestamps, signal.values)):
            if not np.isfinite(v):
                continue
            # Derive signal_id from engine + metric: e.g., "hilbert_inst_amp"
            metric_name = signal.signal_id.split('_')[-1]  # e.g. 'inst_amp'
            derived_signal_id = f"{signal.engine}_{metric_name}"
            rows.append({
                'entity_id': entity_id,
                'signal_id': derived_signal_id,
                'source_signal': signal.source_signal,
                'engine': signal.engine,
                'signal_type': 'dense',  # Pointwise = native resolution
                'timestamp': float(t) if isinstance(t, (int, float)) else t,
                'value': float(v),
                'mode_id': None,  # Assigned by geometry layer
            })
    return rows


# =============================================================================
# INLINE CHARACTERIZATION
# =============================================================================

def characterize_signal_inline(
    signal_id: str,
    values: np.ndarray,
    dates: Optional[np.ndarray] = None,
) -> Tuple[set, bool, Dict[str, Any]]:
    """
    Characterize an signal inline and determine which engines to run.

    This replaces the pre-computed characterization parquet approach.
    Characterization happens as each signal is processed.

    Args:
        signal_id: The signal to characterize
        values: Signal values (full history, will be cleaned of NaN)
        dates: Optional observation dates

    Returns:
        Tuple of (engines_to_run: set, has_discontinuities: bool, char_summary: dict)
    """
    # Clean values
    values = np.asarray(values, dtype=float)
    valid_mask = ~np.isnan(values)
    clean_values = values[valid_mask]

    if len(clean_values) < 50:
        # Insufficient data - run core engines only
        logger.debug(f"{signal_id}: Insufficient data ({len(clean_values)}), using core engines only")
        return CORE_ENGINES.copy(), False, {'signal_id': signal_id, 'reason': 'insufficient_data'}

    # Clean dates if provided
    clean_dates = None
    if dates is not None:
        dates = np.asarray(dates)
        if len(dates) == len(values):
            clean_dates = dates[valid_mask]

    # Characterize inline using the module
    char_result = characterize_signal(
        signal_id=signal_id,
        values=clean_values,
        dates=clean_dates,
    )

    # Get engines to run using module helper
    engines_to_run, has_discontinuities = get_engines_from_characterization(
        char_result=char_result,
        core_engines=CORE_ENGINES,
        conditional_engines=CONDITIONAL_ENGINES,
        discontinuity_engines=DISCONTINUITY_ENGINES,
    )

    # Get summary for logging/storage
    char_summary = get_characterization_summary(char_result)

    return engines_to_run, has_discontinuities, char_summary


# Legacy function for backwards compatibility (reads from parquet)
def load_characterization() -> Dict[str, Dict[str, Any]]:
    """
    Load pre-computed characterization data from parquet.

    DEPRECATED: Use characterize_signal_inline() instead.
    This is kept for backwards compatibility with parallel workers.
    """
    # Legacy path - characterization not in new 5-file schema
    char_path = get_data_root() / "characterization.parquet"

    if char_path is None or not char_path.exists():
        return {}

    df = pl.read_parquet(char_path)

    char_dict = {}
    for row in df.iter_rows(named=True):
        signal_id = row['signal_id']

        # Parse valid_engines from JSON string
        valid_engines_str = row.get('valid_engines', '[]')
        if isinstance(valid_engines_str, str):
            try:
                valid_engines = json.loads(valid_engines_str)
            except json.JSONDecodeError:
                valid_engines = []
        else:
            valid_engines = valid_engines_str or []

        char_dict[signal_id] = {
            'valid_engines': set(valid_engines),
            'has_discontinuities': row.get('n_breaks', 0) > 0,
        }

    return char_dict


def get_engines_for_signal(
    signal_id: str,
    characterization: Dict[str, Dict[str, Any]],
) -> Tuple[set, bool]:
    """
    Determine which engines to run based on pre-computed characterization.

    DEPRECATED: Use characterize_signal_inline() instead for inline characterization.
    """
    # Start with CORE engines (always run)
    engines_to_run = CORE_ENGINES.copy()

    # Check if we have characterization for this signal
    if signal_id not in characterization:
        engines_to_run.update(CONDITIONAL_ENGINES)
        return engines_to_run, False

    char = characterization[signal_id]
    valid_engines = char.get('valid_engines', set())
    has_discontinuities = char.get('has_discontinuities', False)

    # Add conditional engines if they're in the valid_engines list
    for engine in CONDITIONAL_ENGINES:
        if engine in valid_engines:
            engines_to_run.add(engine)

    return engines_to_run, has_discontinuities


# =============================================================================
# SLIDING WINDOW GENERATION
# =============================================================================


def generate_windows(
    observations: pl.DataFrame,
    target_obs: int,
    min_obs: int,
    stride: int,
) -> List[Tuple[np.ndarray, Any, Any, int]]:
    """
    Generate sliding windows from observations using Polars.

    Args:
        observations: DataFrame with timestamp and value columns
        target_obs: Target number of observations per window
        min_obs: Minimum observations required
        stride: Steps between window endpoints

    Returns:
        List of (values, timestamp, lookback_start, actual_obs) tuples
        timestamp can be float (cycles) or date (calendar time)
    """
    if len(observations) < min_obs:
        return []

    # Sort by timestamp
    df = observations.sort("timestamp")

    # Extract arrays
    timestamps = df["timestamp"].to_numpy()
    values = df["value"].to_numpy()
    n = len(values)

    windows = []
    start_idx = max(min_obs, target_obs) - 1

    for end_idx in range(start_idx, n, stride):
        window_start_idx = max(0, end_idx - target_obs + 1)
        window_len = end_idx - window_start_idx + 1

        if window_len < min_obs:
            continue

        window_values = values[window_start_idx : end_idx + 1]
        timestamp = timestamps[end_idx]
        lookback_start = timestamps[window_start_idx]

        # Keep float timestamps as-is (cycles, seconds, etc.)
        # Only convert datetime types if present (legacy support)
        if hasattr(timestamp, "date") and not isinstance(timestamp, (int, float)):
            timestamp = timestamp.date()
        elif hasattr(timestamp, "astype") and not isinstance(timestamp, (int, float, np.floating)):
            timestamp = pd.Timestamp(timestamp).to_pydatetime().date()

        if hasattr(lookback_start, "date") and not isinstance(lookback_start, (int, float)):
            lookback_start = lookback_start.date()
        elif hasattr(lookback_start, "astype") and not isinstance(lookback_start, (int, float, np.floating)):
            lookback_start = pd.Timestamp(lookback_start).to_pydatetime().date()

        windows.append((window_values, timestamp, lookback_start, window_len))

    return windows


# =============================================================================
# WORKER FUNCTIONS
# =============================================================================


def process_signal_parallel(assignment: WorkerAssignment) -> Dict[str, Any]:
    """
    Worker function for parallel execution.

    Writes to isolated temp parquet file.
    Memory-efficient: reads observations per-signal using lazy scan.
    """
    signals = assignment.items
    config = assignment.config
    temp_path = assignment.temp_path

    window_days = config.get("window_days")
    stride_days = config.get("stride_days")
    engine_min_obs = config.get("engine_min_obs", DEFAULT_ENGINE_MIN_OBS)
    engines_filter = config.get("engines_filter", None)

    # Get path for lazy scanning (read per-signal)
    obs_path = get_path(OBSERVATIONS)

    total_windows = 0
    total_metrics = 0
    failed = 0
    batch_rows = []
    computed_at = datetime.now()

    for signal_id in signals:
        try:
            # Read only this signal's observations - LAZY SCAN with filter pushdown
            obs_df = (
                pl.scan_parquet(obs_path)
                .filter(pl.col("signal_id") == signal_id)
                .collect()
            )

            if len(obs_df) == 0:
                del obs_df
                continue

            # Get entity_id for this signal (from observations)
            entity_id = obs_df["entity_id"][0] if "entity_id" in obs_df.columns and len(obs_df) > 0 else signal_id

            # Generate windows
            min_obs = min(engine_min_obs.values())
            windows = generate_windows(obs_df, window_days, min_obs, stride_days)
            total_windows += len(windows)

            # Process each window
            for values, timestamp, lookback_start, actual_obs in windows:
                # Run each engine
                for engine_name, engine_func in VECTOR_ENGINES.items():
                    if engines_filter and engine_name not in engines_filter:
                        continue

                    min_obs_engine = engine_min_obs.get(engine_name, MIN_OBSERVATIONS)
                    if actual_obs < min_obs_engine:
                        continue

                    # Compute metrics
                    try:
                        try:
                            metrics = engine_func(values, min_obs=min_obs_engine)
                        except TypeError:
                            metrics = engine_func(values)
                    except Exception as e:
                        logger.debug(f"Engine {engine_name} failed for {signal_id}: {e}")
                        continue

                    # Collect metrics (new 5-file schema: signals.parquet)
                    for metric_name, metric_value in metrics.items():
                        try:
                            if metric_value is not None:
                                numeric_value = float(metric_value)
                                if np.isfinite(numeric_value):
                                    batch_rows.append(
                                        create_signal_row(
                                            entity_id=entity_id,
                                            source_signal=signal_id,
                                            engine=engine_name,
                                            metric_name=metric_name,
                                            timestamp=timestamp,
                                            value=numeric_value,
                                            signal_type='sparse',
                                        )
                                    )
                        except (TypeError, ValueError):
                            continue

                # Run OBSERVATION_ENGINES (discontinuity detection)
                # These run on the window but have special handling:
                # - break_detector runs first to find break indices
                # - heaviside/dirac only run if breaks found
                min_obs_break = engine_min_obs.get("break_detector", 50)
                if actual_obs >= min_obs_break:
                    try:
                        # Get break metrics first
                        from prism.engines import get_break_metrics, get_heaviside_metrics, get_dirac_metrics

                        break_metrics = get_break_metrics(values)

                        # Add break_detector metrics
                        for metric_name, metric_value in break_metrics.items():
                            if metric_value is not None:
                                try:
                                    numeric_value = float(metric_value)
                                    if np.isfinite(numeric_value):
                                        batch_rows.append(
                                            create_signal_row(
                                                entity_id=entity_id,
                                                source_signal=signal_id,
                                                engine="break_detector",
                                                metric_name=metric_name,
                                                timestamp=timestamp,
                                                value=numeric_value,
                                                signal_type='sparse',
                                            )
                                        )
                                except (TypeError, ValueError):
                                    continue

                        # If breaks found, run heaviside and dirac
                        if break_metrics.get('break_n', 0) > 0:
                            break_result = compute_breaks(values)
                            break_indices = break_result['break_indices']

                            # Heaviside (step function measurement)
                            heaviside_metrics = get_heaviside_metrics(values, break_indices)
                            for metric_name, metric_value in heaviside_metrics.items():
                                if metric_value is not None:
                                    try:
                                        numeric_value = float(metric_value)
                                        if np.isfinite(numeric_value):
                                            batch_rows.append(
                                                create_signal_row(
                                                    entity_id=entity_id,
                                                    source_signal=signal_id,
                                                    engine="heaviside",
                                                    metric_name=metric_name,
                                                    timestamp=timestamp,
                                                    value=numeric_value,
                                                    signal_type='sparse',
                                                )
                                            )
                                    except (TypeError, ValueError):
                                        continue

                            # Dirac (impulse measurement)
                            dirac_metrics = get_dirac_metrics(values, break_indices)
                            for metric_name, metric_value in dirac_metrics.items():
                                if metric_value is not None:
                                    try:
                                        numeric_value = float(metric_value)
                                        if np.isfinite(numeric_value):
                                            batch_rows.append(
                                                create_signal_row(
                                                    entity_id=entity_id,
                                                    source_signal=signal_id,
                                                    engine="dirac",
                                                    metric_name=metric_name,
                                                    timestamp=timestamp,
                                                    value=numeric_value,
                                                    signal_type='sparse',
                                                )
                                            )
                                    except (TypeError, ValueError):
                                        continue
                    except Exception as e:
                        logger.debug(f"Observation engines failed for {signal_id}: {e}")

            # RELEASE - clean up signal data before next iteration
            del obs_df
            del windows
            gc.collect()

        except Exception as e:
            logger.error(f"Failed processing {signal_id}: {e}")
            failed += 1

    # Write to temp parquet - then RELEASE
    if batch_rows:
        df = pl.DataFrame(batch_rows)
        df.write_parquet(temp_path)
        total_metrics = len(batch_rows)
        # RELEASE
        del df
        del batch_rows
        gc.collect()

    return {
        "processed": len(signals) - failed,
        "failed": failed,
        "windows": total_windows,
        "metrics": total_metrics,
        "temp_path": temp_path,
    }


def process_signal_sequential(
    signal_id: str,
    window_days: int,
    stride_days: int,
    engine_min_obs: Dict[str, int],
    engines_to_run: Optional[set] = None,
    has_discontinuities: bool = False,
    use_inline_characterization: bool = True,
    compute_laplace_inline: bool = True,
    entity_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a single signal for a specific entity sequentially.
    Computes engine metrics based on characterization + vector score for each window.
    Optionally characterizes inline and computes laplace field inline.

    Args:
        signal_id: The signal to process
        window_days: Window size in days
        stride_days: Stride between windows
        engine_min_obs: Minimum observations per engine
        engines_to_run: Set of engine names to run (from characterization)
                       If None, characterizes inline or runs all engines
        has_discontinuities: Whether characterization found discontinuities
        use_inline_characterization: If True and engines_to_run is None,
                                     characterize inline to determine engines
        compute_laplace_inline: If True, compute laplace field inline
        entity_id: The entity to filter for (if None, uses first entity found)

    Memory-efficient: Uses lazy scan to read only this entity+signal's data.
    """
    # Default to all engines if not specified AND inline characterization disabled
    if engines_to_run is None and not use_inline_characterization:
        engines_to_run = CORE_ENGINES | CONDITIONAL_ENGINES
    from prism.engines.vector_score import (
        ENGINE_CONFIGS,
        compute_baselines,
        compute_vector_score,
    )

    try:
        # Read observations - LAZY SCAN with filter pushdown
        # Only reads the specific entity+signal's data from disk
        obs_path = get_path(OBSERVATIONS)
        if entity_id is not None:
            # Filter by both entity_id and signal_id
            obs_df = (
                pl.scan_parquet(obs_path)
                .filter(
                    (pl.col("entity_id") == entity_id) &
                    (pl.col("signal_id") == signal_id)
                )
                .collect()
            )
        else:
            # Legacy mode: filter by signal_id only, use first entity found
            obs_df = (
                pl.scan_parquet(obs_path)
                .filter(pl.col("signal_id") == signal_id)
                .collect()
            )
            # Get entity_id from first row
            entity_id = obs_df["entity_id"][0] if "entity_id" in obs_df.columns and len(obs_df) > 0 else signal_id

        if len(obs_df) == 0:
            return {"signal": signal_id, "windows": 0, "metrics": 0}

        # =================================================================
        # V2 ARCHITECTURE: POINTWISE ENGINE COMPUTATION (native resolution)
        # =================================================================
        # These engines produce values at every timestamp, not windowed
        full_series = obs_df.sort("timestamp")
        full_timestamps = full_series["timestamp"].to_numpy()
        full_values = full_series["value"].to_numpy()

        # Compute pointwise signals (Hilbert, Derivatives, Statistical)
        dense_signals = compute_pointwise_signals(
            signal_id=signal_id,
            timestamps=full_timestamps,
            values=full_values,
        )

        # Convert to row format for storage (pass entity_id for new schema)
        dense_rows = dense_signals_to_rows(dense_signals, entity_id, datetime.now())

        # =================================================================
        # INLINE CHARACTERIZATION: If engines_to_run not provided, characterize now
        char_summary = None
        if engines_to_run is None and use_inline_characterization:
            # Get full series for characterization
            full_values = obs_df.sort("timestamp")["value"].to_numpy()
            full_dates = obs_df.sort("timestamp")["timestamp"].to_numpy()

            engines_to_run, has_discontinuities, char_summary = characterize_signal_inline(
                signal_id=signal_id,
                values=full_values,
                dates=full_dates,
            )
            logger.debug(f"{signal_id}: char={char_summary.get('dynamical_class', 'N/A')}, engines={len(engines_to_run)}")

        # Fallback if still None
        if engines_to_run is None:
            engines_to_run = CORE_ENGINES | CONDITIONAL_ENGINES

        total_windows = 0
        batch_rows = []
        field_rows = []  # For laplace computation
        computed_at = datetime.now()

        # Generate windows
        min_obs = min(engine_min_obs.values())
        windows = generate_windows(obs_df, window_days, min_obs, stride_days)
        total_windows = len(windows)

        # Track metric history for baseline computation (running baselines)
        metric_history: Dict[str, List[float]] = {k: [] for k in ENGINE_CONFIGS.keys()}

        # Process each window
        for values, timestamp, lookback_start, actual_obs in windows:
            # Collect all metrics for this window (for vector score)
            window_metrics: Dict[str, float] = {}

            # Run each engine (only those in engines_to_run based on characterization)
            for engine_name, engine_func in VECTOR_ENGINES.items():
                # Skip engines not in the engines_to_run set
                if engine_name not in engines_to_run:
                    continue

                min_obs_engine = engine_min_obs.get(engine_name, MIN_OBSERVATIONS)
                if actual_obs < min_obs_engine:
                    continue

                # Compute metrics
                try:
                    try:
                        metrics = engine_func(values, min_obs=min_obs_engine)
                    except TypeError:
                        metrics = engine_func(values)
                except Exception as e:
                    logger.debug(f"Engine {engine_name} failed for {signal_id}: {e}")
                    continue

                # Collect metrics (new 5-file schema: signals.parquet)
                for metric_name, metric_value in metrics.items():
                    try:
                        if metric_value is not None:
                            numeric_value = float(metric_value)
                            if np.isfinite(numeric_value):
                                batch_rows.append(
                                    create_signal_row(
                                        entity_id=entity_id,
                                        source_signal=signal_id,
                                        engine=engine_name,
                                        metric_name=metric_name,
                                        timestamp=timestamp,
                                        value=numeric_value,
                                        signal_type='sparse',  # Windowed = sparse
                                    )
                                )
                                # Track for vector score
                                window_metrics[metric_name] = numeric_value
                                if metric_name in metric_history:
                                    metric_history[metric_name].append(numeric_value)
                    except (TypeError, ValueError):
                        continue

            # Run OBSERVATION_ENGINES (discontinuity detection)
            # Only run if characterization indicates discontinuities exist
            # (or if no characterization, always run to detect)
            min_obs_break = engine_min_obs.get("break_detector", 50)
            should_check_discontinuities = has_discontinuities or (engines_to_run == (CORE_ENGINES | CONDITIONAL_ENGINES))

            if actual_obs >= min_obs_break and should_check_discontinuities:
                try:
                    from prism.engines import get_break_metrics, get_heaviside_metrics, get_dirac_metrics

                    break_metrics = get_break_metrics(values)

                    # Add break_detector metrics (new 5-file schema)
                    for metric_name, metric_value in break_metrics.items():
                        if metric_value is not None:
                            try:
                                numeric_value = float(metric_value)
                                if np.isfinite(numeric_value):
                                    batch_rows.append(
                                        create_signal_row(
                                            entity_id=entity_id,
                                            source_signal=signal_id,
                                            engine="break_detector",
                                            metric_name=metric_name,
                                            timestamp=timestamp,
                                            value=numeric_value,
                                            signal_type='sparse',
                                        )
                                    )
                                    window_metrics[metric_name] = numeric_value
                            except (TypeError, ValueError):
                                continue

                    # If breaks found in this window, run heaviside and dirac
                    if break_metrics.get('break_n', 0) > 0:
                        break_result = compute_breaks(values)
                        break_indices = break_result['break_indices']

                        # Heaviside (step function measurement) - new 5-file schema
                        heaviside_metrics = get_heaviside_metrics(values, break_indices)
                        for metric_name, metric_value in heaviside_metrics.items():
                            if metric_value is not None:
                                try:
                                    numeric_value = float(metric_value)
                                    if np.isfinite(numeric_value):
                                        batch_rows.append(
                                            create_signal_row(
                                                entity_id=entity_id,
                                                source_signal=signal_id,
                                                engine="heaviside",
                                                metric_name=metric_name,
                                                timestamp=timestamp,
                                                value=numeric_value,
                                                signal_type='sparse',
                                            )
                                        )
                                        window_metrics[metric_name] = numeric_value
                                except (TypeError, ValueError):
                                    continue

                        # Dirac (impulse measurement) - new 5-file schema
                        dirac_metrics = get_dirac_metrics(values, break_indices)
                        for metric_name, metric_value in dirac_metrics.items():
                            if metric_value is not None:
                                try:
                                    numeric_value = float(metric_value)
                                    if np.isfinite(numeric_value):
                                        batch_rows.append(
                                            create_signal_row(
                                                entity_id=entity_id,
                                                source_signal=signal_id,
                                                engine="dirac",
                                                metric_name=metric_name,
                                                timestamp=timestamp,
                                                value=numeric_value,
                                                signal_type='sparse',
                                            )
                                        )
                                        window_metrics[metric_name] = numeric_value
                                except (TypeError, ValueError):
                                    continue
                except Exception as e:
                    logger.debug(f"Observation engines failed for {signal_id}: {e}")

            # Compute vector score using running baselines (need 10+ observations)
            if len(metric_history.get('hurst_exponent', [])) >= 10:
                # Compute baselines from history
                baselines = {}
                for metric_name, history in metric_history.items():
                    if len(history) >= 10:
                        baselines[metric_name] = compute_baselines(np.array(history))

                # Compute score
                scores = compute_vector_score(window_metrics, baselines)

                # Add score metrics (new 5-file schema)
                for score_name, score_value in scores.items():
                    if score_name in ('n_engines', 'total_weight'):
                        continue  # Skip metadata
                    if score_value is not None and not np.isnan(score_value):
                        batch_rows.append(
                            create_signal_row(
                                entity_id=entity_id,
                                source_signal=signal_id,
                                engine="vector_score",
                                metric_name=score_name,
                                timestamp=timestamp,
                                value=float(score_value),
                                signal_type='sparse',
                            )
                        )

        # INLINE LAPLACE: Compute field vectors if requested
        field_rows = []
        if compute_laplace_inline and len(batch_rows) > 0:
            # Group batch_rows by (engine, signal_id) for laplace computation
            # In new schema, signal_id = "{source_signal}_{engine}_{metric}"
            from collections import defaultdict
            metric_series = defaultdict(list)

            for row in batch_rows:
                # Extract engine and metric from signal_id: "{source}_{engine}_{metric}"
                parts = row['signal_id'].split('_')
                if len(parts) >= 2:
                    engine = row['engine']
                    # metric_name is the part after source and engine
                    metric_name = '_'.join(parts[2:]) if len(parts) > 2 else parts[-1]
                else:
                    engine = row['engine']
                    metric_name = row['signal_id']
                key = (engine, metric_name)
                metric_series[key].append({
                    'timestamp': row['timestamp'],  # New schema uses 'timestamp'
                    'metric_value': row['value'],  # New schema uses 'value'
                })

            # Compute laplace for each metric series
            for (engine, metric_name), series_data in metric_series.items():
                # Sort by date
                series_data.sort(key=lambda x: x['timestamp'])
                dates = [s['timestamp'] for s in series_data]
                values = np.array([s['metric_value'] for s in series_data])

                # Compute laplace field
                laplace_rows = compute_laplace_for_series(
                    signal_id=signal_id,
                    dates=dates,
                    values=values,
                    engine=engine,
                    metric_name=metric_name,
                )
                field_rows.extend(laplace_rows)

            # Compute divergence per window
            if field_rows:
                div_by_window = compute_divergence_for_signal(field_rows)
                field_rows = add_divergence_to_field_rows(field_rows, div_by_window)

        return {
            "signal": signal_id,
            "windows": total_windows,
            "metrics": len(batch_rows),
            "rows": batch_rows,
            "dense_rows": dense_rows,  # V2: Native resolution pointwise output
            "field_rows": field_rows,
            "char_summary": char_summary,
            "n_dense_signals": len(dense_signals),  # V2: Count of DenseSignal objects
        }

    except Exception as e:
        return {"signal": signal_id, "windows": 0, "metrics": 0, "error": str(e)}


def run_window_tier(
    signals: List[Tuple[str, str]],
    window_name: str,
    engine_min_obs: Optional[Dict[str, int]] = None,
    use_inline_characterization: bool = True,
    verbose: bool = True,
    window_days_override: Optional[int] = None,
    stride_days_override: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run vector computation for a specific window tier.

    Uses INLINE CHARACTERIZATION: Each signal is characterized as it's processed,
    rather than reading from a pre-computed characterization parquet.

    Args:
        signals: List of (entity_id, signal_id) tuples to process
        window_name: Window tier name ('anchor', 'bridge', 'scout', 'micro', 'adaptive')
        engine_min_obs: Minimum observations per engine
        use_inline_characterization: If True, characterize each signal inline
        verbose: Print progress
        window_days_override: Override window size (for adaptive mode)
        stride_days_override: Override stride (for adaptive mode)

    Returns:
        Dict with processing statistics
    """
    # Handle adaptive mode with overrides
    if window_days_override is not None and stride_days_override is not None:
        window_days = window_days_override
        stride_days = stride_days_override
    else:
        config = load_stride_config()
        window = config.get_window(window_name)
        window_days = window.window_days
        stride_days = window.stride_days

    if engine_min_obs is None:
        engine_min_obs = DEFAULT_ENGINE_MIN_OBS

    if verbose:
        logger.info(f"Window tier: {window_name} ({window_days}d, stride {stride_days}d)")
        logger.info(f"Entity-Signal pairs: {len(signals)}")
        logger.info(f"Inline characterization: {'enabled' if use_inline_characterization else 'disabled'}")

    # Sequential execution with incremental batch writes and progress tracking
    from prism.db.progress_tracker import ProgressTracker

    tracker = ProgressTracker("vector", "signal")
    # New 5-file schema: all signals go to signals.parquet
    target_path = get_path(SIGNALS)

    # Convert pairs to tracking keys (entity_id:signal_id)
    pair_keys = [f"{e}:{s}" for e, s in signals]

    # Filter to pending pairs (resume capability)
    pending_keys = tracker.get_pending(pair_keys, window_name)
    skipped = len(signals) - len(pending_keys)

    # Convert back to pairs
    pending = [(k.split(":")[0], k.split(":")[1]) for k in pending_keys]

    # Memory tracking
    start_memory = get_memory_usage_mb()

    # Batch accumulator - reduced to 1 for very large datasets (TEP: 175k obs/signal)
    BATCH_SIZE = 1
    batch_rows = []
    batch_field_rows = []  # Field vectors (laplace) - internal, not in 5-file schema
    batch_dense_rows = []  # Dense signals also go to signals.parquet with signal_type='dense'
    batch_signals = []
    total_windows = 0
    total_metrics = 0
    total_field_rows = 0
    total_dense_rows = 0  # V2: Track dense signal rows
    errors = []

    # Skip writing laplace field data - it's intermediate computation for geometry
    # The 5-file schema stores only: observations, vector, geometry, state, cohorts
    field_path = None  # Disabled - laplace data computed inline for geometry
    FIELD_KEY_COLS = None  # Not used
    # Dense signals go to vector.parquet

    if verbose:
        if skipped > 0:
            logger.info(f"Resuming: {skipped} already completed, {len(pending)} pending")
        logger.info(f"Sequential mode: batch writes every {BATCH_SIZE} signals")
        logger.info(f"Starting memory: {start_memory:.1f} MB")

    for i, (entity_id, signal_id) in enumerate(pending):
        pair_key = f"{entity_id}:{signal_id}"
        tracker.mark_started(pair_key, window_name)

        # INLINE CHARACTERIZATION: engines determined inside process_signal_sequential
        # Pass engines_to_run=None to trigger inline characterization
        result = process_signal_sequential(
            signal_id, window_days, stride_days, engine_min_obs,
            engines_to_run=None,  # Triggers inline characterization
            has_discontinuities=False,  # Will be determined inline
            use_inline_characterization=use_inline_characterization,
            compute_laplace_inline=True,
            entity_id=entity_id,  # Process this specific entity
        )

        if "error" in result:
            tracker.mark_failed(pair_key, window_name, result.get("error", ""))
            errors.append(result)
            print(f"  X {entity_id}:{signal_id} (error)", flush=True)
            continue

        if "rows" in result:
            batch_rows.extend(result["rows"])
            batch_signals.append((pair_key, len(result["rows"])))
            total_windows += result.get("windows", 0)

            # Collect field rows (laplace)
            if "field_rows" in result and result["field_rows"]:
                batch_field_rows.extend(result["field_rows"])

            # V2: Collect dense rows (pointwise native resolution)
            if "dense_rows" in result and result["dense_rows"]:
                batch_dense_rows.extend(result["dense_rows"])

            print(f"  {signal_id}", end="", flush=True)  # Just show signal_id for brevity

        # Write batch when full - COMPUTE → WRITE → RELEASE pattern
        if len(batch_signals) >= BATCH_SIZE:
            if batch_rows:
                # WRITE sparse (windowed) signals with regime normalization
                norm_config = get_normalization_config()
                df = apply_regime_normalization(
                    batch_rows,
                    window=norm_config['window'],
                    min_periods=norm_config.get('min_periods', 30),
                )
                upsert_parquet(df, target_path, SIGNALS_KEY_COLS)
                total_metrics += len(batch_rows)

                # SKIP field rows (laplace) - computed inline for geometry, not persisted
                if batch_field_rows and field_path is not None:
                    field_df = pl.DataFrame(batch_field_rows, infer_schema_length=None)
                    upsert_parquet(field_df, field_path, FIELD_KEY_COLS)
                    total_field_rows += len(batch_field_rows)
                    del field_df
                batch_field_rows = []  # Clear regardless

                # WRITE dense (pointwise) signals - also go to signals.parquet
                if batch_dense_rows:
                    dense_df = pl.DataFrame(batch_dense_rows, infer_schema_length=None)
                    upsert_parquet(dense_df, target_path, SIGNALS_KEY_COLS)  # Same file!
                    total_dense_rows += len(batch_dense_rows)
                    del dense_df
                    batch_dense_rows = []

                for ind_id, row_count in batch_signals:
                    tracker.mark_completed(ind_id, window_name, rows=row_count)

                row_count = len(batch_rows)

                # RELEASE - explicit memory cleanup
                del df
                del batch_rows
                batch_rows = []
                batch_signals = []
                force_gc()

                # Report memory
                current_mem = get_memory_usage_mb()
                print(f" -> SAVED {row_count:,} rows ({i+1}/{len(pending)}) [mem: {current_mem:.0f} MB]", flush=True)
            else:
                batch_rows = []
                batch_signals = []
                batch_field_rows = []
                batch_dense_rows = []

        else:
            print(",", end="", flush=True)

    # Final batch write - COMPUTE → WRITE → RELEASE
    if batch_rows:
        # WRITE sparse (windowed) signals with regime normalization
        norm_config = get_normalization_config()
        df = apply_regime_normalization(
            batch_rows,
            window=norm_config['window'],
            min_periods=norm_config.get('min_periods', 30),
        )
        upsert_parquet(df, target_path, SIGNALS_KEY_COLS)
        total_metrics += len(batch_rows)

        # SKIP remaining field rows (laplace) - computed inline for geometry, not persisted
        if batch_field_rows and field_path is not None:
            field_df = pl.DataFrame(batch_field_rows, infer_schema_length=None)
            upsert_parquet(field_df, field_path, FIELD_KEY_COLS)
            total_field_rows += len(batch_field_rows)
            del field_df

        # WRITE remaining dense (pointwise) signals - also to signals.parquet
        if batch_dense_rows:
            dense_df = pl.DataFrame(batch_dense_rows, infer_schema_length=None)
            upsert_parquet(dense_df, target_path, SIGNALS_KEY_COLS)  # Same file!
            total_dense_rows += len(batch_dense_rows)
            del dense_df

        for ind_id, row_count in batch_signals:
            tracker.mark_completed(ind_id, window_name, rows=row_count)

        row_count = len(batch_rows)

        # RELEASE
        del df
        del batch_rows
        force_gc()

        current_mem = get_memory_usage_mb()
        print(f" -> SAVED {row_count:,} rows (FINAL) [mem: {current_mem:.0f} MB]", flush=True)

    # Summary with memory delta
    end_memory = get_memory_usage_mb()
    delta = end_memory - start_memory
    print(f"\n[{window_name}] Complete: {total_metrics:,} windowed rows, {total_dense_rows:,} dense rows, {total_field_rows:,} field rows", flush=True)
    print(f"[{window_name}] Signals: {len(pending)}", flush=True)
    print(f"[{window_name}] Memory: {start_memory:.0f} → {end_memory:.0f} MB (Δ{delta:+.0f} MB)", flush=True)
    if skipped > 0:
        print(f"[{window_name}] Skipped {skipped} already-completed signals", flush=True)

    return {
        "window_name": window_name,
        "window_days": window_days,
        "stride_days": stride_days,
        "signals": len(signals),
        "windows": total_windows,
        "metrics": total_metrics,
        "dense_rows": total_dense_rows,  # V2: Native resolution pointwise output
        "field_rows": total_field_rows,
        "errors": len(errors),
    }


# =============================================================================
# MAIN RUNNER
# =============================================================================


def get_signals(cohort: Optional[str] = None) -> List[str]:
    """Get signal IDs from parquet files (for legacy/cohort mode)."""
    obs_path = get_path(OBSERVATIONS)
    if not obs_path.exists():
        return []

    if cohort:
        # Filter by cohort membership from cohorts.parquet
        cohorts_path = get_path(COHORTS)
        if cohorts_path.exists():
            members = pl.read_parquet(cohorts_path)
            signal_ids = (
                members.filter(pl.col("cohort_id") == cohort)["signal_id"].unique().to_list()
            )
            return signal_ids

    # All signals (unique signal_id from observations)
    df = pl.scan_parquet(obs_path).select("signal_id").unique().collect()
    return df["signal_id"].to_list()


def get_entity_signal_pairs() -> List[Tuple[str, str]]:
    """
    Get unique (entity_id, signal_id) pairs from observations.

    Returns:
        List of (entity_id, signal_id) tuples
    """
    obs_path = get_path(OBSERVATIONS)
    if not obs_path.exists():
        return []

    # Get unique (entity_id, signal_id) pairs
    df = (
        pl.scan_parquet(obs_path)
        .select(["entity_id", "signal_id"])
        .unique()
        .collect()
    )

    pairs = [(row["entity_id"], row["signal_id"]) for row in df.iter_rows(named=True)]
    return pairs


def run_adaptive_vectors(
    signals: Optional[List[str]] = None,
    verbose: bool = True,
    domain: str = "unknown",
) -> Dict[str, Any]:
    """
    Compute sliding window vectors using adaptive (auto-detected) window size.

    Uses DomainClock to analyze signal frequencies and determine optimal window.
    The domain's clock is set by its fastest-changing signal.

    Args:
        signals: List of signal IDs to process (None = all)
        verbose: Print progress
        domain: Domain name for config lookup

    Returns:
        Dict with processing statistics
    """
    import json

    # Ensure directories exist
    ensure_directory()

    # Get signals if not provided
    if signals is None:
        signals = get_signals()

    if not signals:
        if verbose:
            print("No signals found")
        return {"signals": 0, "windows": 0, "metrics": 0, "errors": 0}

    # Get adaptive clock config
    clock_config = load_clock_config(domain)

    if verbose:
        print("=" * 80)
        print("PRISM VECTOR - ADAPTIVE WINDOWING (DomainClock)")
        print("=" * 80)
        print(f"Domain: {domain}")
        print(f"Signals: {len(signals)}")

    # Characterize domain frequency
    if verbose:
        print("\nCharacterizing domain frequency...")

    # min_samples must be >= max engine min_obs requirement (garch=50)
    # to ensure all engines can run on each window
    clock = DomainClock(
        min_cycles=clock_config.get('min_cycles', 3),
        min_samples=clock_config.get('min_samples', 50),
        max_samples=clock_config.get('max_samples', 1000),
    )

    # Sample signals for frequency estimation (use lazy scan with filter pushdown)
    sample_size = min(100, len(signals))
    sample_pairs = signals[:sample_size]
    # Extract unique signal_ids from (entity_id, signal_id) tuples
    sample_signal_ids = list(set(sig for _, sig in sample_pairs))
    obs_path = get_path(OBSERVATIONS)
    sample_obs = (
        pl.scan_parquet(obs_path)
        .filter(pl.col('signal_id').is_in(sample_signal_ids))
        .collect()
    )

    domain_info = clock.characterize(sample_obs)
    window_config = clock.get_window_config()

    # Save domain_info for downstream layers (laplace, geometry, state)
    domain_info_path = get_data_root() / "domain_info.json"
    domain_info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(domain_info_path, 'w') as f:
        json.dump(clock.to_parquet_metadata(), f, indent=2, default=str)
    if verbose:
        print(f"Saved domain_info to {domain_info_path}")

    # Convert window_samples to window_days (for compatibility with run_window_tier)
    # This is approximate - assumes 1 sample ≈ 1 day for date-indexed data
    window_days = window_config['window_samples']
    stride_days = window_config['stride_samples']

    if verbose:
        print(f"\nAdaptive window: {window_days} samples, stride {stride_days} samples")
        print(f"Fastest signal: {window_config['fastest_signal']}")
        print(f"Domain frequency: {window_config['domain_frequency']:.6f} Hz")
        print()

    # Process with adaptive window (single tier)
    engine_min_obs = DEFAULT_ENGINE_MIN_OBS.copy()

    result = run_window_tier(
        signals=signals,
        window_name="adaptive",  # Virtual tier name
        engine_min_obs=engine_min_obs,
        use_inline_characterization=True,
        verbose=verbose,
        # Override with adaptive settings
        window_days_override=window_days,
        stride_days_override=stride_days,
    )

    if verbose:
        print()
        print("=" * 80)
        print("COMPLETE (Adaptive)")
        print("=" * 80)
        print(f"Signals processed: {len(signals)}")
        print(f"Window: {window_days} samples (adaptive)")
        print(f"Stride: {stride_days} samples")
        print(f"Total metrics: {result.get('metrics', 0):,}")
        print(f"Total field rows (laplace): {result.get('field_rows', 0):,}")
        print(f"Errors: {result.get('errors', 0)}")

    return {
        "signals": len(signals),
        "window_samples": window_days,
        "stride_samples": stride_days,
        "domain_frequency": window_config['domain_frequency'],
        "fastest_signal": window_config['fastest_signal'],
        "windows": result.get("windows", 0),
        "metrics": result.get("metrics", 0),
        "field_rows": result.get("field_rows", 0),
        "errors": result.get("errors", 0),
    }


def run_cohort_vector_aggregation(
    cohorts: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Aggregate signal vectors into cohort vectors.

    Reads signal-level vector metrics and aggregates by cohort:
    - mean, std, min, max, count per metric per cohort per window

    This gives cohort-level behavior metrics using the same "language"
    as signal vectors (Hurst, entropy, GARCH, etc.)

    Args:
        cohorts: List of cohort IDs to process (None = all)
        verbose: Print progress

    Returns:
        Summary stats
    """
    from datetime import datetime

    print()
    print("=" * 80)
    print("COHORT VECTOR AGGREGATION")
    print("=" * 80)

    # Load signal vectors
    signal_path = get_path(SIGNALS)
    if not signal_path.exists():
        print(f"ERROR: No signal vectors found at {signal_path}")
        print("Run --signal mode first to compute signal vectors.")
        return {"status": "error", "reason": "no signal vectors"}

    print(f"Reading signal vectors from {signal_path}...")
    vectors_df = pl.read_parquet(signal_path)
    print(f"  Loaded {len(vectors_df):,} signal vector rows")

    # Load cohort membership
    cohorts_path = get_path(COHORTS)
    if not cohorts_path.exists():
        print(f"ERROR: No cohort membership found at {cohorts_path}")
        return {"status": "error", "reason": "no cohort membership"}

    members_df = pl.read_parquet(cohorts_path)
    print(f"  Loaded {len(members_df):,} cohort membership rows")

    # Join vectors with cohort membership
    # vectors: signal_id, timestamp, target_obs, engine, metric_name, metric_value
    # members: signal_id, cohort_id
    joined = vectors_df.join(
        members_df.select(["signal_id", "cohort_id"]),
        on="signal_id",
        how="inner"
    )
    print(f"  Joined: {len(joined):,} rows with cohort assignments")

    # Filter to specific cohorts if requested
    if cohorts:
        joined = joined.filter(pl.col("cohort_id").is_in(cohorts))
        print(f"  Filtered to cohorts {cohorts}: {len(joined):,} rows")

    # Aggregate by cohort + window + engine + metric
    print()
    print("Aggregating by cohort/window/engine/metric...")

    cohort_vectors = joined.group_by(
        ["cohort_id", "timestamp", "target_obs", "engine", "metric_name"]
    ).agg([
        pl.col("metric_value").mean().alias("metric_mean"),
        pl.col("metric_value").std().alias("metric_std"),
        pl.col("metric_value").min().alias("metric_min"),
        pl.col("metric_value").max().alias("metric_max"),
        pl.col("metric_value").count().alias("n_signals"),
    ]).with_columns([
        pl.lit(datetime.now()).alias("computed_at")
    ])

    print(f"  Aggregated to {len(cohort_vectors):,} cohort vector rows")

    # Show sample
    if verbose:
        print()
        print("=== SAMPLE COHORT VECTOR ROW ===")
        sample = cohort_vectors.head(1)
        for col in sample.columns:
            print(f"  {col}: {sample[col][0]}")

    # Write output - cohort vectors go to geometry.parquet
    output_path = get_path(GEOMETRY)
    print()
    print(f"Writing to {output_path}...")

    # Use upsert for incremental updates
    key_cols = ["cohort_id", "timestamp", "target_obs", "engine", "metric_name"]
    upsert_parquet(cohort_vectors, output_path, key_cols)

    print(f"  Wrote {len(cohort_vectors):,} cohort vector rows")

    # Summary stats
    n_cohorts = cohort_vectors["cohort_id"].n_unique()
    n_windows = cohort_vectors.select(["timestamp", "target_obs"]).unique().height
    n_engines = cohort_vectors["engine"].n_unique()

    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Cohorts: {n_cohorts}")
    print(f"Windows: {n_windows}")
    print(f"Engines: {n_engines}")
    print(f"Total rows: {len(cohort_vectors):,}")

    return {
        "status": "success",
        "cohorts": n_cohorts,
        "windows": n_windows,
        "engines": n_engines,
        "rows": len(cohort_vectors),
    }


def run_sliding_vectors(
    signals: Optional[List[Tuple[str, str]]] = None,
    verbose: bool = True,
    adaptive: bool = False,
) -> Dict[str, Any]:
    """
    Compute sliding window vectors for each (entity_id, signal_id) pair.

    Runs anchor + bridge tiers only (structural coverage).
    Scout/micro are for geometry/state delta drill-down.

    INLINE ARCHITECTURE:
    - Characterization: Each signal characterized as it's processed
    - Laplace: Field vectors computed inline after engine metrics
    - No separate characterize.py or laplace.py runner needed

    Args:
        signals: List of (entity_id, signal_id) tuples to process (None = all)
        verbose: Print progress
        adaptive: Use DomainClock to auto-detect window size from data

    Returns:
        Dict with processing statistics
    """
    import os

    # Ensure directories exist
    ensure_directory()

    # Get configuration
    engine_min_obs = DEFAULT_ENGINE_MIN_OBS.copy()
    stride_config = load_stride_config()
    domain = os.environ.get('PRISM_DOMAIN', 'unknown')

    # ADAPTIVE MODE: Use DomainClock to detect window from data
    if adaptive:
        return run_adaptive_vectors(signals, verbose, domain)

    # STANDARD MODE: Use default tiers from config (anchor + bridge)
    # Scout + micro are drilldown tiers - run when delta flags displacement
    window_tiers = get_default_tiers()

    # Get (entity_id, signal_id) pairs if not provided
    if signals is None:
        signals = get_entity_signal_pairs()

    if not signals:
        if verbose:
            print("No entity-signal pairs found")
        return {"signals": 0, "windows": 0, "metrics": 0, "errors": 0}

    if verbose:
        print("=" * 80)
        print("PRISM VECTOR - INLINE CHARACTERIZATION + LAPLACE")
        print("=" * 80)
        print(f"Storage: Parquet files")
        print(f"Entity-Signal pairs: {len(signals)}")
        print(f"Characterization: INLINE (per signal)")
        print(f"Laplace: INLINE (computed with metrics)")
        print(f"Window tiers: {window_tiers}")
        for tier in window_tiers:
            w = stride_config.get_window(tier)
            print(f"  {tier}: {w.window_days}d / {w.stride_days}d stride")
        print(f"Core engines (all): {sorted(CORE_ENGINES)}")
        print(f"Conditional engines: {sorted(CONDITIONAL_ENGINES)}")
        print()

    # Process each window tier
    all_results = []
    for window_name in window_tiers:
        result = run_window_tier(
            signals=signals,
            window_name=window_name,
            engine_min_obs=engine_min_obs,
            use_inline_characterization=True,
            verbose=verbose,
        )
        all_results.append(result)

    # Aggregate results
    total_windows = sum(r["windows"] for r in all_results)
    total_metrics = sum(r["metrics"] for r in all_results)
    total_dense_rows = sum(r.get("dense_rows", 0) for r in all_results)  # V2
    total_field_rows = sum(r.get("field_rows", 0) for r in all_results)
    total_errors = sum(r["errors"] for r in all_results)

    if verbose:
        print()
        print("=" * 80)
        print("COMPLETE")
        print("=" * 80)
        print(f"Signals processed: {len(signals)}")
        print(f"Window tiers processed: {len(window_tiers)}")
        print(f"Total windows: {total_windows:,}")
        print(f"Total windowed metrics: {total_metrics:,}")
        print(f"Total dense rows (pointwise): {total_dense_rows:,}")  # V2
        print(f"Total field rows (laplace): {total_field_rows:,}")
        print(f"Errors: {total_errors}")
        print()
        print(f"Output: vector/signal.parquet (windowed metrics)")
        print(f"Output: vector/signal_dense.parquet (V2: native resolution)")  # V2
        print(f"Output: vector/signal_field.parquet (laplace field)")

    return {
        "signals": len(signals),
        "window_tiers": len(window_tiers),
        "windows": total_windows,
        "metrics": total_metrics,
        "dense_rows": total_dense_rows,  # V2
        "field_rows": total_field_rows,
        "errors": total_errors,
        "results": all_results,
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PRISM Vector Runner - Compute behavioral metrics from observations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output: data/vector.parquet

Examples:
  python -m prism.entry_points.signal_vector              # Production run
  python -m prism.entry_points.signal_vector --adaptive   # Auto-detect window
  python -m prism.entry_points.signal_vector --force      # Force recompute
  python -m prism.entry_points.signal_vector --testing --limit 100
  python -m prism.entry_points.signal_vector --testing --signal s1,s2
"""
    )

    # Production flags
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Use DomainClock to auto-detect window size from data frequency",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear progress tracker and recompute everything",
    )

    # Testing mode
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Enable testing mode (required for --limit and --signal)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="[TESTING] Max observations per signal",
    )
    parser.add_argument(
        "--signal",
        type=str,
        help="[TESTING] Comma-separated signal IDs to process",
    )

    args = parser.parse_args()

    # Guard testing flags
    if not args.testing and (args.limit or args.signal):
        logger.warning("=" * 80)
        logger.warning("TESTING FLAGS IGNORED - --testing not specified")
        logger.warning("Running FULL computation.")
        logger.warning("=" * 80)
        args.limit = None
        args.signal = None

    # Config is law - load or fail
    stride_config = load_stride_config()
    if not stride_config:
        raise RuntimeError("Cannot load stride config - config/stride.yaml required")

    # Handle --force: clear progress tracker
    if args.force:
        from prism.db.progress_tracker import ProgressTracker
        tracker = ProgressTracker("vector", "signal")
        tracker.clear()
        print("Progress cleared (--force)", flush=True)

    # Get (entity_id, signal_id) pairs to process
    if args.signal:
        # Support both "signal" and "entity:signal" format
        items = []
        for i in args.signal.split(","):
            i = i.strip()
            if ":" in i:
                entity, signal = i.split(":", 1)
                items.append((entity, signal))
            else:
                # If only signal provided, get all entities for that signal
                obs_path = get_path(OBSERVATIONS)
                entities = (
                    pl.scan_parquet(obs_path)
                    .filter(pl.col("signal_id") == i)
                    .select("entity_id")
                    .unique()
                    .collect()["entity_id"].to_list()
                )
                items.extend([(e, i) for e in entities])
    else:
        items = get_entity_signal_pairs()

    if not items:
        raise RuntimeError("No entity-signal pairs found to process")

    print(f"Source: data/observations.parquet", flush=True)
    print(f"Destination: data/vector.parquet", flush=True)
    print(f"Entity-Signal pairs to process: {len(items)}", flush=True)

    # Run vector computation
    # Laplace is computed INLINE - no separate chain needed
    # Each signal gets: characterization → engine metrics → laplace field
    run_sliding_vectors(
        signals=items,
        verbose=True,
        adaptive=args.adaptive,
    )
