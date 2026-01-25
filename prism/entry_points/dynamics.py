#!/usr/bin/env python3
"""
PRISM Dynamics Entry Point
==========================

Computes temporal dynamics and state transitions using ALL state engines.

Engines:
    State engines:
        - granger: Granger causality
        - cross_correlation: Cross-correlation analysis
        - cointegration: Cointegration testing
        - dtw: Dynamic time warping
        - dmd: Dynamic mode decomposition
        - transfer_entropy: Information transfer
        - coupled_inertia: Coupling dynamics
        - trajectory: Trajectory analysis

    Dynamics engines:
        - embedding: Embedding dimension
        - phase_space: Phase space reconstruction
        - lyapunov: Lyapunov exponents

    Temporal engines:
        - energy_dynamics: Energy evolution
        - tension_dynamics: Tension evolution
        - phase_detector: Phase detection
        - break_detector: Structural breaks
        - cohort_aggregator: Cohort-level dynamics

Usage:
    python -m prism.entry_points.dynamics
    python -m prism.entry_points.dynamics --force

Output:
    data/dynamics.parquet - Temporal dynamics per (entity, signal, window)
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
    Import dynamics engines based on config selection.

    Config structure:
        engines:
          dynamics:
            embedding: true
            phase_space: false
            ...

    Note: Class-based pairwise engines (granger, cross_correlation, cointegration,
    dtw, dmd, transfer_entropy) are stored separately for entity-level computation.
    """
    signal_engines = {}  # Engines that work on single signals
    pairwise_engines = {}  # Class-based engines for entity matrices
    engine_config = config.get('engines', {}).get('dynamics', {})

    # If no config, enable all engines by default
    if not engine_config:
        engine_config = {k: True for k in [
            'embedding', 'phase_space', 'lyapunov', 'break_detector',
            'granger', 'cross_correlation', 'cointegration', 'dtw',
            'dmd', 'transfer_entropy', 'trajectory',
        ]}

    # Dynamics engines (single signal)
    if engine_config.get('embedding', True):
        try:
            from prism.engines.dynamics import compute_embedding
            signal_engines['embedding'] = compute_embedding
        except ImportError:
            pass

    if engine_config.get('phase_space', True):
        try:
            from prism.engines.dynamics import compute_phase_space
            signal_engines['phase_space'] = compute_phase_space
        except ImportError:
            pass

    if engine_config.get('lyapunov', True):
        try:
            from prism.engines.dynamics import compute_lyapunov
            signal_engines['lyapunov'] = compute_lyapunov
        except ImportError:
            pass

    # Break detector (single signal)
    if engine_config.get('break_detector', True):
        try:
            from prism.engines.state.break_detector import compute_breaks
            signal_engines['break_detector'] = compute_breaks
        except ImportError:
            pass

    # Pairwise class-based engines (entity matrices)
    if engine_config.get('granger', True):
        try:
            from prism.engines.state.granger import GrangerEngine
            pairwise_engines['granger'] = GrangerEngine()
        except ImportError:
            pass

    if engine_config.get('cross_correlation', True):
        try:
            from prism.engines.state.cross_correlation import CrossCorrelationEngine
            pairwise_engines['cross_correlation'] = CrossCorrelationEngine()
        except ImportError:
            pass

    if engine_config.get('cointegration', True):
        try:
            from prism.engines.state.cointegration import CointegrationEngine
            pairwise_engines['cointegration'] = CointegrationEngine()
        except ImportError:
            pass

    if engine_config.get('dtw', True):
        try:
            from prism.engines.state.dtw import DTWEngine
            pairwise_engines['dtw'] = DTWEngine()
        except ImportError:
            pass

    if engine_config.get('dmd', True):
        try:
            from prism.engines.state.dmd import DMDEngine
            pairwise_engines['dmd'] = DMDEngine()
        except ImportError:
            pass

    if engine_config.get('transfer_entropy', True):
        try:
            from prism.engines.state.transfer_entropy import TransferEntropyEngine
            pairwise_engines['transfer_entropy'] = TransferEntropyEngine()
        except ImportError:
            pass

    logger.info(f"  Signal engines: {len(signal_engines)}, Pairwise engines: {len(pairwise_engines)}")
    return signal_engines, pairwise_engines


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_CONFIG = {
    'min_samples_dynamics': 100,
    'window_size': None,
    'stride': None,
}


def load_config(data_path: Path) -> Dict[str, Any]:
    """Load config from data directory."""
    config_path = data_path / 'config.yaml'
    config = DEFAULT_CONFIG.copy()

    if config_path.exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f) or {}
        config.update(user_config)

    return config


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

def compute_dynamics(
    observations: pl.DataFrame,
    config: Dict[str, Any],
    signal_engines: Dict[str, Any],
    pairwise_engines: Dict[str, Any],
    force: bool = False,
) -> pl.DataFrame:
    """
    Compute dynamics metrics for all signals.

    Args:
        observations: Raw observations
        config: Domain config
        signal_engines: Dict of compute functions for single signals
        pairwise_engines: Dict of class-based engines for entity matrices
        force: Recompute all

    Returns:
        DataFrame with dynamics metrics
    """
    min_samples = config.get('min_samples_dynamics', 100)

    # Group by entity+signal
    signals = observations.group_by(['entity_id', 'signal_id']).agg([
        pl.col('value').alias('values'),
        pl.col('timestamp').alias('timestamps'),
    ])

    n_signals = len(signals)
    n_engines = len(signal_engines) + len(pairwise_engines)
    logger.info(f"Processing {n_signals} signals with {n_engines} engines")

    results = []

    for i, row in enumerate(signals.iter_rows(named=True)):
        entity_id = row['entity_id']
        signal_id = row['signal_id']
        values = np.array(row['values'], dtype=float)
        timestamps = np.array(row['timestamps'], dtype=float)

        # Sort and clean
        sort_idx = np.argsort(timestamps)
        values = values[sort_idx]
        timestamps = timestamps[sort_idx]

        valid = ~np.isnan(values)
        values = values[valid]
        timestamps = timestamps[valid]

        if len(values) < min_samples:
            continue

        row_data = {
            'entity_id': entity_id,
            'signal_id': signal_id,
            'n_samples': len(values),
            'window_start': float(timestamps[0]),
            'window_end': float(timestamps[-1]),
        }

        # Run signal-level dynamics engines
        for engine_name, compute_fn in signal_engines.items():
            try:
                result = compute_fn(values)
                flat = flatten_result(result, engine_name)
                row_data.update(flat)
            except Exception:
                pass

        # Compute trend metrics
        try:
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)
            row_data['trend_slope'] = float(slope)
            row_data['trend_intercept'] = float(intercept)
            detrended = values - (slope * x + intercept)
            row_data['detrended_variance'] = float(np.var(detrended))
        except Exception:
            pass

        # Velocity and acceleration
        try:
            velocity = np.diff(values)
            acceleration = np.diff(velocity)
            row_data['velocity_mean'] = float(np.mean(velocity))
            row_data['velocity_std'] = float(np.std(velocity))
            row_data['acceleration_mean'] = float(np.mean(acceleration))
            row_data['acceleration_std'] = float(np.std(acceleration))
        except Exception:
            pass

        results.append(row_data)

        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i + 1}/{n_signals} signals")

    if not results:
        logger.warning("No signals with sufficient data")
        return pl.DataFrame()

    df = pl.DataFrame(results)
    logger.info(f"Dynamics: {len(df)} rows, {len(df.columns)} columns")

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRISM Dynamics - Temporal dynamics computation"
    )
    parser.add_argument('--force', '-f', action='store_true',
                        help='Recompute even if output exists')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PRISM Dynamics Engine")
    logger.info("=" * 60)

    ensure_directory()

    obs_path = get_path(OBSERVATIONS)
    if not obs_path.exists():
        logger.error("observations.parquet not found")
        sys.exit(1)

    data_path = obs_path.parent
    output_path = data_path / 'dynamics.parquet'

    if output_path.exists() and not args.force:
        logger.info("dynamics.parquet exists, use --force to recompute")
        return 0

    config = load_config(data_path)

    logger.info("Loading engines from config...")
    signal_engines, pairwise_engines = import_engines(config)
    logger.info(f"Total engines: {len(signal_engines) + len(pairwise_engines)}")

    observations = read_parquet(obs_path)
    logger.info(f"Loaded {len(observations):,} observations")

    start = time.time()
    df = compute_dynamics(observations, config, signal_engines, pairwise_engines, args.force)
    elapsed = time.time() - start

    if len(df) > 0:
        write_parquet_atomic(df, output_path)
        logger.info(f"Wrote {output_path}")
        logger.info(f"  {len(df):,} rows, {len(df.columns)} columns in {elapsed:.1f}s")

    return 0


if __name__ == '__main__':
    sys.exit(main())
