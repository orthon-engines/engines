#!/usr/bin/env python3
"""
PRISM Vector Entry Point
========================

Computes signal-level metrics using ALL vector engines with proper windowing.

Engines (28 total):
    Memory (4): hurst_dfa, hurst_rs, acf_decay, spectral_slope
    Information (3): permutation_entropy, sample_entropy, entropy_rate
    Frequency (2): spectral, wavelet
    Volatility (4): garch, realized_vol, bipower_variation, hilbert_amplitude
    Recurrence (1): rqa
    Typology (8): cusum, derivative_stats, distribution, rolling_volatility,
                  seasonality, stationarity, takens, trend
    Pointwise (3): derivatives, hilbert, statistical
    Momentum (1): runs_test
    Discontinuity (3): dirac, heaviside, structural

Usage:
    python -m prism.entry_points.vector
    python -m prism.entry_points.vector --force
    python -m prism.entry_points.vector --window 100 --stride 20

Output:
    data/vector.parquet - One row per (entity, signal, window)
"""

import argparse
import logging
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

import numpy as np
import pandas as pd
import polars as pl

from prism.db.parquet_store import get_path, ensure_directory, OBSERVATIONS
from prism.db.polars_io import read_parquet, write_parquet_atomic

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENGINE IMPORTS
# =============================================================================

def import_engines(config: Dict[str, Any]):
    """
    Import vector engines based on config selection.

    Config structure:
        engines:
          vector:
            hurst_dfa: true
            hurst_rs: false
            ...
    """
    engines = {}
    engine_config = config.get('engines', {}).get('vector', {})

    # If no config, enable all engines by default
    if not engine_config:
        engine_config = {k: True for k in [
            'hurst_dfa', 'hurst_rs', 'acf_decay', 'spectral_slope',
            'permutation_entropy', 'sample_entropy', 'entropy_rate',
            'spectral', 'wavelet', 'garch', 'realized_vol',
            'bipower_variation', 'hilbert_amplitude', 'rqa',
            'cusum', 'derivative_stats', 'distribution', 'rolling_volatility',
            'seasonality', 'stationarity', 'takens', 'trend',
            'derivatives', 'hilbert', 'statistical', 'runs_test',
            'dirac', 'heaviside', 'structural',
            'laplace',  # Gradient, laplacian, divergence - key for geometry
        ]}

    # Memory engines
    if engine_config.get('hurst_dfa', True):
        try:
            from prism.engines.memory import compute_hurst_dfa
            engines['hurst_dfa'] = compute_hurst_dfa
        except ImportError:
            pass
    if engine_config.get('hurst_rs', True):
        try:
            from prism.engines.memory import compute_hurst_rs
            engines['hurst_rs'] = compute_hurst_rs
        except ImportError:
            pass
    if engine_config.get('acf_decay', True):
        try:
            from prism.engines.memory import compute_acf_decay
            engines['acf_decay'] = compute_acf_decay
        except ImportError:
            pass
    if engine_config.get('spectral_slope', True):
        try:
            from prism.engines.memory import compute_spectral_slope
            engines['spectral_slope'] = compute_spectral_slope
        except ImportError:
            pass

    # Information engines
    if engine_config.get('permutation_entropy', True):
        try:
            from prism.engines.information import compute_permutation_entropy
            engines['permutation_entropy'] = compute_permutation_entropy
        except ImportError:
            pass
    if engine_config.get('sample_entropy', True):
        try:
            from prism.engines.information import compute_sample_entropy
            engines['sample_entropy'] = compute_sample_entropy
        except ImportError:
            pass
    if engine_config.get('entropy_rate', True):
        try:
            from prism.engines.information import compute_entropy_rate
            engines['entropy_rate'] = compute_entropy_rate
        except ImportError:
            pass

    # Frequency engines
    if engine_config.get('spectral', True):
        try:
            from prism.engines.frequency import compute_spectral
            engines['spectral'] = compute_spectral
        except ImportError:
            pass
    if engine_config.get('wavelet', True):
        try:
            from prism.engines.frequency import compute_wavelet
            engines['wavelet'] = compute_wavelet
        except ImportError:
            pass

    # Volatility engines
    if engine_config.get('garch', True):
        try:
            from prism.engines.volatility import compute_garch
            engines['garch'] = compute_garch
        except ImportError:
            pass
    if engine_config.get('realized_vol', True):
        try:
            from prism.engines.volatility import compute_realized_vol
            engines['realized_vol'] = compute_realized_vol
        except ImportError:
            pass
    if engine_config.get('bipower_variation', True):
        try:
            from prism.engines.volatility import compute_bipower_variation
            engines['bipower_variation'] = compute_bipower_variation
        except ImportError:
            pass
    if engine_config.get('hilbert_amplitude', True):
        try:
            from prism.engines.volatility import compute_hilbert_amplitude
            engines['hilbert_amplitude'] = compute_hilbert_amplitude
        except ImportError:
            pass

    # Recurrence engines
    if engine_config.get('rqa', True):
        try:
            from prism.engines.recurrence import compute_rqa
            engines['rqa'] = compute_rqa
        except ImportError:
            pass

    # Typology engines
    if engine_config.get('cusum', True):
        try:
            from prism.engines.typology import compute_cusum
            engines['cusum'] = compute_cusum
        except ImportError:
            pass
    if engine_config.get('derivative_stats', True):
        try:
            from prism.engines.typology import compute_derivative_stats
            engines['derivative_stats'] = compute_derivative_stats
        except ImportError:
            pass
    if engine_config.get('distribution', True):
        try:
            from prism.engines.typology import compute_distribution
            engines['distribution'] = compute_distribution
        except ImportError:
            pass
    if engine_config.get('rolling_volatility', True):
        try:
            from prism.engines.typology import compute_rolling_volatility
            engines['rolling_volatility'] = compute_rolling_volatility
        except ImportError:
            pass
    if engine_config.get('seasonality', True):
        try:
            from prism.engines.typology import compute_seasonality
            engines['seasonality'] = compute_seasonality
        except ImportError:
            pass
    if engine_config.get('stationarity', True):
        try:
            from prism.engines.typology import compute_stationarity
            engines['stationarity'] = compute_stationarity
        except ImportError:
            pass
    if engine_config.get('takens', True):
        try:
            from prism.engines.typology import compute_takens
            engines['takens'] = compute_takens
        except ImportError:
            pass
    if engine_config.get('trend', True):
        try:
            from prism.engines.typology import compute_trend
            engines['trend'] = compute_trend
        except ImportError:
            pass

    # Pointwise engines
    if engine_config.get('derivatives', True):
        try:
            from prism.engines.pointwise import compute_derivatives
            engines['derivatives'] = compute_derivatives
        except ImportError:
            pass
    if engine_config.get('hilbert', True):
        try:
            from prism.engines.pointwise import compute_hilbert
            engines['hilbert'] = compute_hilbert
        except ImportError:
            pass
    if engine_config.get('statistical', True):
        try:
            from prism.engines.pointwise import compute_statistical
            engines['statistical'] = compute_statistical
        except ImportError:
            pass

    # Momentum engines
    if engine_config.get('runs_test', True):
        try:
            from prism.engines.momentum import compute_runs_test
            engines['runs_test'] = compute_runs_test
        except ImportError:
            pass

    # Discontinuity engines
    if engine_config.get('dirac', True):
        try:
            from prism.engines.discontinuity import compute_dirac
            engines['dirac'] = compute_dirac
        except ImportError:
            pass
    if engine_config.get('heaviside', True):
        try:
            from prism.engines.discontinuity import compute_heaviside
            engines['heaviside'] = compute_heaviside
        except ImportError:
            pass
    if engine_config.get('structural', True):
        try:
            from prism.engines.discontinuity import compute_structural
            engines['structural'] = compute_structural
        except ImportError:
            pass

    # Laplace engines (gradient, laplacian, divergence)
    if engine_config.get('laplace', True):
        try:
            from prism.engines.laplace import compute_gradient, compute_laplacian
            engines['laplace'] = _compute_laplace_metrics
        except ImportError:
            pass

    logger.info(f"  Loaded {len(engines)} engines from config")
    return engines


def _compute_laplace_metrics(values: np.ndarray) -> Dict[str, float]:
    """
    Compute Laplace field metrics for a signal.

    Returns gradient (velocity), laplacian (acceleration), and divergence.
    """
    from prism.engines.laplace import compute_gradient, compute_laplacian

    gradient = compute_gradient(values)
    laplacian = compute_laplacian(values)

    # Filter NaNs
    grad_valid = gradient[~np.isnan(gradient)]
    lap_valid = laplacian[~np.isnan(laplacian)]

    result = {}

    if len(grad_valid) > 0:
        result['gradient_mean'] = float(np.mean(grad_valid))
        result['gradient_std'] = float(np.std(grad_valid))
        result['gradient_max'] = float(np.max(np.abs(grad_valid)))
        result['gradient_sum'] = float(np.sum(grad_valid))

    if len(lap_valid) > 0:
        result['laplacian_mean'] = float(np.mean(lap_valid))
        result['laplacian_std'] = float(np.std(lap_valid))
        result['laplacian_max'] = float(np.max(np.abs(lap_valid)))
        # Divergence = sum of laplacians (SOURCE > 0, SINK < 0)
        result['divergence'] = float(np.sum(lap_valid))
        result['divergence_sign'] = 1.0 if result['divergence'] > 0 else (-1.0 if result['divergence'] < 0 else 0.0)

    return result


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_CONFIG = {
    'min_samples': 50,
    'window_size': None,  # None = use full signal
    'stride': None,       # None = non-overlapping
}


def load_config(data_path: Path) -> Dict[str, Any]:
    """Load config from data directory."""
    config_path = data_path / 'config.yaml'
    config = DEFAULT_CONFIG.copy()

    if config_path.exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f) or {}
        config.update(user_config)
        logger.info(f"Loaded config: min_samples={config['min_samples']}")

    return config


# =============================================================================
# WINDOWING
# =============================================================================

def generate_windows(
    values: np.ndarray,
    timestamps: np.ndarray,
    window_size: Optional[int] = None,
    stride: Optional[int] = None,
    min_samples: int = 50,
) -> List[Dict]:
    """
    Generate overlapping windows from a signal.

    Args:
        values: Signal values
        timestamps: Corresponding timestamps
        window_size: Window size (None = full signal)
        stride: Stride between windows (None = window_size)
        min_samples: Minimum samples per window

    Yields:
        Dict with window_idx, window_start, window_end, values, timestamps
    """
    n = len(values)

    if window_size is None or window_size >= n:
        # Single window (full signal)
        if n >= min_samples:
            yield {
                'window_idx': 0,
                'window_start': float(timestamps[0]),
                'window_end': float(timestamps[-1]),
                'values': values,
                'timestamps': timestamps,
            }
        return

    if stride is None:
        stride = window_size

    window_idx = 0
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        window_values = values[start:end]
        window_timestamps = timestamps[start:end]

        if len(window_values) >= min_samples:
            yield {
                'window_idx': window_idx,
                'window_start': float(window_timestamps[0]),
                'window_end': float(window_timestamps[-1]),
                'values': window_values,
                'timestamps': window_timestamps,
            }
            window_idx += 1


# =============================================================================
# RESULT FLATTENING
# =============================================================================

def flatten_result(result: Any, prefix: str) -> Dict[str, float]:
    """Flatten engine result to dict of floats."""
    flat = {}

    if result is None:
        return flat

    if isinstance(result, dict):
        for k, v in result.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                val = float(v)
                if np.isfinite(val):
                    flat[f"{prefix}_{k}"] = val
            elif isinstance(v, np.ndarray) and v.size == 1:
                val = float(v.item())
                if np.isfinite(val):
                    flat[f"{prefix}_{k}"] = val
    elif hasattr(result, '__dataclass_fields__'):
        # Dataclass
        for k in result.__dataclass_fields__:
            v = getattr(result, k)
            if isinstance(v, (int, float, np.integer, np.floating)):
                val = float(v)
                if np.isfinite(val):
                    flat[f"{prefix}_{k}"] = val
    elif isinstance(result, (int, float, np.integer, np.floating)):
        val = float(result)
        if np.isfinite(val):
            flat[prefix] = val

    return flat


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def compute_vector(
    observations: pl.DataFrame,
    config: Dict[str, Any],
    engines: Dict[str, Any],
    force: bool = False,
) -> pl.DataFrame:
    """
    Compute vector metrics for all signals.

    Args:
        observations: Raw observations DataFrame
        config: Domain configuration
        engines: Dict of engine_name -> compute function
        force: Recompute all

    Returns:
        DataFrame with one row per (entity, signal, window)
    """
    min_samples = config.get('min_samples', 50)
    window_size = config.get('window_size')
    stride = config.get('stride')

    # Group observations by entity+signal
    signals = observations.group_by(['entity_id', 'signal_id']).agg([
        pl.col('value').alias('values'),
        pl.col('timestamp').alias('timestamps'),
    ])

    n_signals = len(signals)
    n_engines = len(engines)
    logger.info(f"Processing {n_signals} signals with {n_engines} engines")

    results = []
    total_windows = 0

    for i, row in enumerate(signals.iter_rows(named=True)):
        entity_id = row['entity_id']
        signal_id = row['signal_id']
        values = np.array(row['values'], dtype=float)
        timestamps = np.array(row['timestamps'], dtype=float)

        # Sort by timestamp
        sort_idx = np.argsort(timestamps)
        values = values[sort_idx]
        timestamps = timestamps[sort_idx]

        # Remove NaN
        valid = ~np.isnan(values)
        values = values[valid]
        timestamps = timestamps[valid]

        if len(values) < min_samples:
            continue

        # Generate windows
        for window in generate_windows(values, timestamps, window_size, stride, min_samples):
            window_values = window['values']

            row_data = {
                'entity_id': entity_id,
                'signal_id': signal_id,
                'window_idx': window['window_idx'],
                'window_start': window['window_start'],
                'window_end': window['window_end'],
                'n_samples': len(window_values),
            }

            # Run all engines
            for engine_name, compute_fn in engines.items():
                try:
                    result = compute_fn(window_values)
                    flat = flatten_result(result, engine_name)
                    row_data.update(flat)
                except Exception as e:
                    # Engine failed for this window, skip silently
                    pass

            results.append(row_data)
            total_windows += 1

        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i + 1}/{n_signals} signals ({total_windows} windows)")

    if not results:
        logger.warning("No signals with sufficient data")
        return pl.DataFrame()

    df = pl.DataFrame(results)
    logger.info(f"Vector: {len(df)} rows, {len(df.columns)} columns")

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRISM Vector - Signal-level metrics computation"
    )
    parser.add_argument('--force', '-f', action='store_true',
                        help='Recompute even if output exists')
    parser.add_argument('--window', type=int, default=None,
                        help='Window size (default: full signal)')
    parser.add_argument('--stride', type=int, default=None,
                        help='Stride between windows (default: window size)')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PRISM Vector Engine")
    logger.info("=" * 60)

    ensure_directory()

    # Load observations
    obs_path = get_path(OBSERVATIONS)
    if not obs_path.exists():
        logger.error(f"observations.parquet not found at {obs_path}")
        sys.exit(1)

    data_path = obs_path.parent
    output_path = data_path / 'vector.parquet'

    if output_path.exists() and not args.force:
        logger.info(f"vector.parquet exists, use --force to recompute")
        return 0

    # Load config
    config = load_config(data_path)
    if args.window:
        config['window_size'] = args.window
    if args.stride:
        config['stride'] = args.stride

    # Load engines from config
    logger.info("Loading engines from config...")
    engines = import_engines(config)
    logger.info(f"Total engines: {len(engines)}")

    # Load data
    observations = read_parquet(obs_path)
    logger.info(f"Loaded {len(observations):,} observations")

    # Compute
    start = time.time()
    df = compute_vector(observations, config, engines, args.force)
    elapsed = time.time() - start

    if len(df) > 0:
        write_parquet_atomic(df, output_path)
        logger.info(f"Wrote {output_path}")
        logger.info(f"  {len(df):,} rows, {len(df.columns)} columns in {elapsed:.1f}s")

    return 0


if __name__ == '__main__':
    sys.exit(main())
