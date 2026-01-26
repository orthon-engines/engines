#!/usr/bin/env python3
"""
PRISM Vector Entry Point
========================

Computes signal-level metrics using ALL vector engines with index-based windowing.

ORTHON Canonical Spec v1.0.0:
    Window k contains all rows where:
        x₀ + k*stride ≤ index < x₀ + k*stride + window_size

    CRITICAL: window_size and stride are in INDEX UNITS (seconds, meters, cycles),
              NOT row counts.

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
    python -m prism.entry_points.vector --window 100.0 --stride 50.0

Output:
    data/vector.parquet - One row per (entity, signal, window)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings


import numpy as np
import pandas as pd
import polars as pl

from prism.db.parquet_store import get_path, ensure_directory, OBSERVATIONS
from prism.db.streaming import StreamingReader, IncrementalWriter, check_memory
from prism.config.validator import ConfigurationError

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

# Available engines - must be explicitly enabled in config
# Updated for core/ directory structure
AVAILABLE_ENGINES = {
    # Memory engines
    'hurst_dfa': 'prism.engines.core.memory.hurst_dfa:compute',
    'hurst_rs': 'prism.engines.core.memory.hurst_rs:compute',
    'acf_decay': 'prism.engines.core.memory.acf_decay:compute',
    'spectral_slope': 'prism.engines.core.memory.spectral_slope:compute',
    # Information engines
    'permutation_entropy': 'prism.engines.core.information.permutation_entropy:compute',
    'sample_entropy': 'prism.engines.core.information.sample_entropy:compute',
    'entropy_rate': 'prism.engines.core.information.entropy_rate:compute',
    # Frequency engines
    'spectral': 'prism.engines.core.frequency.spectral:compute',
    'wavelet': 'prism.engines.core.frequency.wavelet:compute',
    # Volatility engines
    'garch': 'prism.engines.core.volatility.garch:compute',
    'realized_vol': 'prism.engines.core.volatility.realized_vol:compute',
    'bipower_variation': 'prism.engines.core.volatility.bipower_variation:compute',
    'hilbert_amplitude': 'prism.engines.core.volatility.hilbert_amplitude:compute',
    # Recurrence engines
    'rqa': 'prism.engines.core.recurrence.rqa:compute',
    # Typology engines
    'cusum': 'prism.engines.core.typology.cusum:compute',
    'derivative_stats': 'prism.engines.core.typology.derivative_stats:compute',
    'distribution': 'prism.engines.core.typology.distribution:compute',
    'rolling_volatility': 'prism.engines.core.typology.rolling_volatility:compute',
    'seasonality': 'prism.engines.core.typology.seasonality:compute',
    'stationarity': 'prism.engines.core.typology.stationarity:compute',
    'takens': 'prism.engines.core.typology.takens:compute',
    'trend': 'prism.engines.core.typology.trend:compute',
    # Pointwise engines
    'derivatives': 'prism.engines.core.pointwise.derivatives:compute',
    'hilbert': 'prism.engines.core.pointwise.hilbert:compute',
    'statistical': 'prism.engines.core.pointwise.statistical:compute',
    # Momentum engines
    'runs_test': 'prism.engines.core.momentum.runs_test:compute',
    # Discontinuity engines
    'dirac': 'prism.engines.core.detection.spike_detector:compute',
    'heaviside': 'prism.engines.core.detection.step_detector:compute',
    'structural': 'prism.engines.core.discontinuity.structural:compute',
    # Laplace engines
    'laplace': None,  # Special handling
    # Dynamics engines
    'hd_slope': 'prism.engines.core.dynamics.hd_slope:compute_hd_slope',
    # Jacobian (Wolf algorithm)
    'jacobian': 'prism.engines.core.dynamics.jacobian:compute',
}


def import_engines(config: Dict[str, Any]):
    """
    Import vector engines based on EXPLICIT config selection.

    ZERO DEFAULTS POLICY: Engines must be explicitly listed in config.

    Config structure (REQUIRED):
        engines:
          vector:
            enabled:
              - hurst_dfa
              - sample_entropy
              - rqa
            params:
              rqa:
                embedding_dim: 3
                time_delay: 1
                threshold_percentile: 10.0

    Raises:
        ConfigurationError: If engines.vector.enabled not specified
    """
    engines = {}

    if 'engines' not in config or 'vector' not in config.get('engines', {}):
        raise ConfigurationError(
            f"\n{'='*60}\n"
            f"CONFIGURATION ERROR: engines.vector section missing\n"
            f"{'='*60}\n\n"
            f"PRISM requires explicit engine configuration.\n"
            f"Add to config.yaml:\n\n"
            f"  engines:\n"
            f"    vector:\n"
            f"      enabled:\n"
            f"        - hurst_dfa\n"
            f"        - sample_entropy\n"
            f"        - rqa\n\n"
            f"Available engines: {list(AVAILABLE_ENGINES.keys())}\n\n"
            f"NO DEFAULTS. NO FALLBACKS. Configure your domain.\n"
            f"{'='*60}"
        )

    engine_config = config['engines']['vector']

    if 'enabled' not in engine_config:
        raise ConfigurationError(
            f"\n{'='*60}\n"
            f"CONFIGURATION ERROR: engines.vector.enabled not specified\n"
            f"{'='*60}\n\n"
            f"List the engines to run explicitly:\n\n"
            f"  engines:\n"
            f"    vector:\n"
            f"      enabled:\n"
            f"        - hurst_dfa\n"
            f"        - sample_entropy\n"
            f"        - rqa\n\n"
            f"Available engines: {list(AVAILABLE_ENGINES.keys())}\n\n"
            f"NO DEFAULTS. NO FALLBACKS. Configure your domain.\n"
            f"{'='*60}"
        )

    enabled_engines = engine_config['enabled']
    engine_params = engine_config.get('params', {})

    for engine_name in enabled_engines:
        if engine_name not in AVAILABLE_ENGINES:
            logger.warning(f"Unknown engine: {engine_name}")
            continue

        module_path = AVAILABLE_ENGINES[engine_name]
        if module_path is None:
            continue

        try:
            module_name, func_name = module_path.rsplit(':', 1)
            module = __import__(module_name, fromlist=[func_name])
            compute_fn = getattr(module, func_name)

            # Wrap with params if specified
            params = engine_params.get(engine_name, {})
            if params:
                from functools import partial
                engines[engine_name] = partial(compute_fn, **params)
            else:
                engines[engine_name] = compute_fn

        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not load engine {engine_name}: {e}")

    logger.info(f"  Loaded {len(engines)} engines from config: {list(engines.keys())}")
    return engines


# =============================================================================
# CONFIG
# =============================================================================

def load_config(data_path: Path) -> Dict[str, Any]:
    """
    Load config from data directory.

    ORTHON Canonical Spec v1.0.0:
        window.size  - Window width in INDEX UNITS (e.g., seconds, meters, cycles)
        window.stride - Step between windows in INDEX UNITS

    CRITICAL: size and stride are in INDEX UNITS, not row counts.
        Window k contains all rows where:
            x₀ + k*stride ≤ index < x₀ + k*stride + size

    Raises:
        ConfigurationError: If window.size or window.stride not set
    """
    # JSON only - no legacy YAML
    config_path = data_path / 'config.json'

    if not config_path.exists():
        raise ConfigurationError(
            f"\n{'='*60}\n"
            f"CONFIGURATION ERROR: config.json not found\n"
            f"{'='*60}\n"
            f"Location: {data_path / 'config.json'}\n\n"
            f"PRISM requires explicit windowing configuration.\n"
            f"Create config.json with:\n\n"
            f'  {{"window": {{"size": 50.0, "stride": 25.0, "min_observations": 10}}}}\n\n'
            f"CRITICAL: size/stride are in INDEX UNITS (seconds, meters, cycles)\n"
            f"NO DEFAULTS. NO FALLBACKS. Configure your domain.\n"
            f"{'='*60}"
        )

    with open(config_path) as f:
        user_config = json.load(f)

    config = {
        'engines': user_config.get('engines', {}),
    }

    # REQUIRED: window section (ORTHON Canonical Spec v1.0.0)
    if 'window' not in user_config:
        raise ConfigurationError(
            f"\n{'='*60}\n"
            f"CONFIGURATION ERROR: window section missing\n"
            f"{'='*60}\n"
            f"File: {config_path}\n\n"
            f"PRISM requires explicit windowing configuration.\n"
            f"Add to config.yaml:\n\n"
            f"  window:\n"
            f"    size: 50.0           # Window width in INDEX UNITS\n"
            f"    stride: 25.0         # Step between windows in INDEX UNITS\n"
            f"    min_observations: 10 # Minimum rows per window\n\n"
            f"CRITICAL: size/stride are in INDEX UNITS (not row counts)\n"
            f"NO DEFAULTS. NO FALLBACKS. Configure your domain.\n"
            f"{'='*60}"
        )

    window_cfg = user_config['window']

    if window_cfg.get('size') is None:
        raise ConfigurationError(
            f"\n{'='*60}\n"
            f"CONFIGURATION ERROR: window.size not set\n"
            f"{'='*60}\n"
            f"File: {config_path}\n\n"
            f"window.size is REQUIRED (in INDEX UNITS). Example:\n\n"
            f"  window:\n"
            f"    size: 50.0           # Window width in INDEX UNITS\n"
            f"    stride: 25.0         # Step between windows in INDEX UNITS\n"
            f"    min_observations: 10 # Minimum rows per window\n\n"
            f"NO DEFAULTS. NO FALLBACKS. Configure your domain.\n"
            f"{'='*60}"
        )

    if window_cfg.get('stride') is None:
        raise ConfigurationError(
            f"\n{'='*60}\n"
            f"CONFIGURATION ERROR: window.stride not set\n"
            f"{'='*60}\n"
            f"File: {config_path}\n\n"
            f"window.stride is REQUIRED (in INDEX UNITS). Example:\n\n"
            f"  window:\n"
            f"    size: 50.0           # Window width in INDEX UNITS\n"
            f"    stride: 25.0         # Step between windows in INDEX UNITS\n"
            f"    min_observations: 10 # Minimum rows per window\n\n"
            f"NO DEFAULTS. NO FALLBACKS. Configure your domain.\n"
            f"{'='*60}"
        )

    # Use float for index-based windowing
    config['window_size'] = float(window_cfg['size'])
    config['stride'] = float(window_cfg['stride'])
    # Default min_observations = 10 per ORTHON spec
    config['min_observations'] = window_cfg.get('min_observations', 10)

    logger.info(f"Loaded config: window_size={config['window_size']}, stride={config['stride']}, min_observations={config['min_observations']} (INDEX UNITS)")

    return config


# =============================================================================
# WINDOWING (ORTHON Canonical Spec v1.0.0)
# =============================================================================

def generate_windows(
    values: np.ndarray,
    indices: np.ndarray,
    window_size: float,
    stride: float,
    min_observations: int,
) -> List[Dict]:
    """
    Generate overlapping windows from a signal using INDEX-BASED windowing.

    ORTHON Canonical Spec v1.0.0:
        Window k contains all rows where:
            x₀ + k*S ≤ index < x₀ + k*S + W

        Where:
            x₀ = first index value
            W  = window_size (in INDEX UNITS, not row count)
            S  = stride (in INDEX UNITS, not row count)

    Args:
        values: Signal values
        indices: Corresponding sequence indices (time, depth, cycle, etc.)
        window_size: Window width in INDEX UNITS (not row count)
        stride: Step between windows in INDEX UNITS (not row count)
        min_observations: Minimum rows required per window

    Yields:
        Dict with window_idx, window_start, window_end, values, indices
    """
    if len(values) == 0:
        return

    # Get index range
    x0 = float(indices[0])      # First index value
    x_max = float(indices[-1])  # Last index value

    # If total index span is less than window size, skip
    if (x_max - x0) < window_size:
        return

    window_idx = 0
    window_start = x0

    while window_start + window_size <= x_max + stride:  # Allow last partial window
        window_end = window_start + window_size

        # Select rows in this window: window_start ≤ index < window_end
        mask = (indices >= window_start) & (indices < window_end)
        window_values = values[mask]
        window_indices = indices[mask]

        # Only yield if we have enough observations
        if len(window_values) >= min_observations:
            yield {
                'window_idx': window_idx,
                'window_start': window_start,
                'window_end': window_end,
                'values': window_values,
                'indices': window_indices,
            }
            window_idx += 1

        window_start += stride


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
# MAIN COMPUTATION (STREAMING)
# =============================================================================

def compute_vector_streaming(
    obs_path: Path,
    output_path: Path,
    config: Dict[str, Any],
    engines: Dict[str, Any],
) -> int:
    """
    Compute vector metrics for all signals using streaming I/O.

    MEMORY TARGET: < 1GB RAM regardless of input size.

    Uses DuckDB streaming to read signals one at a time, processes windows,
    and writes results incrementally to parquet.

    ORTHON Canonical Spec v1.0.0:
        Windows are defined in INDEX UNITS (not row counts).
        Window k contains all rows where:
            x₀ + k*stride ≤ index < x₀ + k*stride + window_size

    Args:
        obs_path: Path to observations.parquet
        output_path: Path for output vector.parquet
        config: Domain configuration (window_size, stride in INDEX UNITS)
        engines: Dict of engine_name -> compute function

    Returns:
        Total rows written
    """
    check_memory("start")

    min_observations = config['min_observations']
    window_size = config['window_size']  # INDEX UNITS
    stride = config['stride']            # INDEX UNITS

    n_engines = len(engines)

    with StreamingReader(obs_path) as reader:
        n_signals = len(reader.signal_keys)
        logger.info(f"Processing {n_signals} signals with {n_engines} engines")

        with IncrementalWriter(output_path, batch_size=1000) as writer:
            total_windows = 0

            for i, (entity_id, signal_id, values, indices) in enumerate(reader.iter_signals()):
                # Sort by index
                sort_idx = np.argsort(indices)
                values = values[sort_idx]
                indices = indices[sort_idx]

                # Remove NaN
                valid = ~np.isnan(values)
                values = values[valid]
                indices = indices[valid]

                if len(values) < min_observations:
                    continue

                # Generate windows (ORTHON Canonical Spec - index-based)
                for window in generate_windows(values, indices, window_size, stride, min_observations):
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
                        except Exception:
                            # Engine failed for this window, skip silently
                            pass

                    # Write immediately - no accumulation
                    writer.write_row(row_data)
                    total_windows += 1

                if (i + 1) % 10 == 0:
                    logger.info(f"  Processed {i + 1}/{n_signals} signals ({total_windows} windows)")

                # Memory check every 100 signals
                if (i + 1) % 100 == 0:
                    check_memory(f"signal_{i + 1}/{n_signals}")

            check_memory("end")
            rows_written = writer.rows_written

    logger.info(f"Vector: {rows_written} rows written")
    return rows_written


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRISM Vector - Signal-level metrics computation"
    )
    parser.add_argument('--force', '-f', action='store_true',
                        help='Recompute even if output exists')
    # Window/stride in INDEX UNITS - override config (ORTHON Canonical Spec)
    parser.add_argument('--window', type=float,
                        help='Override window.size from config (in INDEX UNITS)')
    parser.add_argument('--stride', type=float,
                        help='Override window.stride from config (in INDEX UNITS)')

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

    # Load config - WILL FAIL LOUDLY if window/stride not set
    try:
        config = load_config(data_path)
    except ConfigurationError as e:
        logger.error(str(e))
        sys.exit(1)

    # CLI overrides (only if explicitly provided)
    if args.window is not None:
        logger.info(f"CLI override: window_size={args.window}")
        config['window_size'] = args.window
    if args.stride is not None:
        logger.info(f"CLI override: stride={args.stride}")
        config['stride'] = args.stride

    # Load engines from config
    logger.info("Loading engines from config...")
    engines = import_engines(config)
    logger.info(f"Total engines: {len(engines)}")

    # Compute using STREAMING (memory-efficient)
    check_memory("before_compute")
    start = time.time()
    rows_written = compute_vector_streaming(obs_path, output_path, config, engines)
    elapsed = time.time() - start
    check_memory("after_compute")

    if rows_written > 0:
        logger.info(f"Wrote {output_path}")
        logger.info(f"  {rows_written:,} rows in {elapsed:.1f}s")
    else:
        logger.warning("No data written - check if signals have sufficient observations")

    return 0


if __name__ == '__main__':
    sys.exit(main())
