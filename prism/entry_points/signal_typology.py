"""
PRISM Signal Typology
=====================

Entry point for signal typology analysis. Layer 1 of the ORTHON framework.

ORTHON Framework:
    Signal Typology    → WHAT is it?     (this entry point)
    Behavioral Geometry → HOW does it behave?
    Dynamical Systems   → WHEN/HOW does it change?
    Causal Mechanics    → WHY does it change?

The Six Orthogonal Axes:
    1. Memory        - Temporal persistence (Hurst, ACF decay)
    2. Periodicity   - Cyclical structure (FFT, wavelets)
    3. Volatility    - Variance dynamics (GARCH, rolling std)
    4. Discontinuity - Level shifts / Heaviside (PELT, CUSUM)
    5. Impulsivity   - Shocks / Dirac (derivative spikes, kurtosis)
    6. Complexity    - Predictability (entropy)

Output: signal_typology.parquet

Pipeline:
    observations.parquet → signal_typology.parquet → behavioral_geometry → ...

Usage:
    # Full run
    python -m prism.entry_points.signal_typology

    # Force recompute
    python -m prism.entry_points.signal_typology --force

    # Adaptive windowing
    python -m prism.entry_points.signal_typology --adaptive

    # Testing mode
    python -m prism.entry_points.signal_typology --testing --limit 100
"""

import argparse
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

# Core imports
from prism.db.parquet_store import (
    ensure_directory,
    get_path,
    OBSERVATIONS,
)
from prism.db.polars_io import read_parquet, write_parquet_atomic

# Typology package (new unified system)
from prism.typology import (
    select_engines,
    get_primary_classification,
    AXIS_NAMES,
)

# Computation module
from prism.engines.characterize import compute_all_axes

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

SIGNAL_TYPOLOGY = 'signal_typology'  # Output file name

# Default window/stride
DEFAULT_WINDOW_SIZE = 252
DEFAULT_STRIDE = 21

# Minimum observations
MIN_OBSERVATIONS = 30


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    entity_col: str = 'entity_id',
    signal_col: str = 'signal_id',
    time_col: str = 'timestamp',
    value_col: str = 'value',
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    force: bool = False,
    adaptive: bool = False,
    testing: bool = False,
    limit: Optional[int] = None,
    signals: Optional[List[str]] = None,
) -> Path:
    """
    Run the full signal typology pipeline.

    Args:
        input_path: Input observations parquet (default: data/observations.parquet)
        output_path: Output parquet (default: data/signal_typology.parquet)
        entity_col: Entity identifier column
        signal_col: Signal identifier column
        time_col: Time/cycle column
        value_col: Value column
        window_size: Rolling window size
        stride: Window stride
        force: Force recompute
        adaptive: Auto-detect window size
        testing: Enable testing mode
        limit: Limit observations per signal (testing)
        signals: Only process these signals (testing)

    Returns:
        Path to output parquet file
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("  SIGNAL TYPOLOGY")
    logger.info("  Six Orthogonal Axes (v2.0)")
    logger.info("  ORTHON Layer 1: WHAT is it?")
    logger.info("=" * 60)
    logger.info(f"  Axes: {', '.join(AXIS_NAMES)}")

    # Resolve paths
    if input_path is None:
        input_path = get_path(OBSERVATIONS)
    else:
        input_path = Path(input_path)

    if output_path is None:
        # Use same directory as observations
        data_dir = Path(input_path).parent
        output_path = data_dir / f"{SIGNAL_TYPOLOGY}.parquet"
    else:
        output_path = Path(output_path)

    # Check existing output
    if output_path.exists() and not force:
        logger.info(f"Output exists: {output_path}")
        logger.info("Use --force to recompute")
        return output_path

    # Load data
    logger.info(f"Loading {input_path}...")
    df = read_parquet(input_path)
    logger.info(f"  Loaded {len(df):,} rows")

    # Detect columns
    entity_col, signal_col, time_col, value_col = _detect_columns(
        df, entity_col, signal_col, time_col, value_col
    )
    logger.info(f"  Columns: entity={entity_col}, signal={signal_col}, time={time_col}, value={value_col}")

    # Testing filters
    if testing:
        if signals:
            df = df.filter(pl.col(signal_col).is_in(signals))
            logger.info(f"  Filtered to signals: {signals}")

        if limit:
            df = df.group_by([entity_col, signal_col]).head(limit)
            logger.info(f"  Limited to {limit} observations per signal")

    # Adaptive windowing
    if adaptive:
        window_size, stride = _auto_detect_window(df, time_col, entity_col)
        logger.info(f"  Adaptive: window={window_size}, stride={stride}")

    # Process all signals
    logger.info(f"\nProcessing with window={window_size}, stride={stride}...")
    results = _process_signals(
        df=df,
        entity_col=entity_col,
        signal_col=signal_col,
        time_col=time_col,
        value_col=value_col,
        window_size=window_size,
        stride=stride,
    )

    if not results:
        logger.warning("No results computed!")
        return output_path

    # Convert to DataFrame
    output_df = pl.DataFrame(results)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(output_df, output_path)

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("")
    logger.info(f"Output: {output_path}")
    logger.info(f"  Rows: {len(output_df):,}")
    logger.info(f"  Columns: {len(output_df.columns)}")
    logger.info(f"  Elapsed: {elapsed:.1f}s")

    # Classification distribution
    if 'classification' in output_df.columns:
        class_counts = output_df.group_by('classification').len().sort('len', descending=True)
        logger.info("")
        logger.info("Classification Distribution:")
        for row in class_counts.iter_rows(named=True):
            logger.info(f"  {row['classification']}: {row['len']:,}")

    return output_path


# =============================================================================
# SIGNAL PROCESSING
# =============================================================================

def _process_signals(
    df: pl.DataFrame,
    entity_col: str,
    signal_col: str,
    time_col: str,
    value_col: str,
    window_size: int,
    stride: int,
) -> List[Dict[str, Any]]:
    """Process all signals and return typology results."""
    results = []

    # Get unique (entity, signal) pairs
    pairs = df.select([entity_col, signal_col]).unique().to_dicts()
    logger.info(f"  Processing {len(pairs)} (entity, signal) pairs...")

    total_windows = 0

    for i, pair in enumerate(pairs):
        entity_id = pair[entity_col]
        signal_id = pair[signal_col]

        # Get signal data
        signal_df = df.filter(
            (pl.col(entity_col) == entity_id) & (pl.col(signal_col) == signal_id)
        ).sort(time_col)

        values = signal_df[value_col].to_numpy()
        times = signal_df[time_col].to_numpy()

        n = len(values)
        if n < window_size:
            continue

        # Process windows
        previous_axes = None

        for start in range(0, n - window_size + 1, stride):
            end = start + window_size
            window_values = values[start:end]
            window_start = times[start]
            window_end = times[end - 1]

            # Compute 6 axes
            axis_result = compute_all_axes(window_values)

            # Get classification and engines
            classification = get_primary_classification(axis_result)
            engines = select_engines(axis_result)

            # Detect regime change
            regime_changed = False
            if previous_axes is not None:
                from prism.typology import detect_regime_change
                change_result = detect_regime_change(previous_axes, axis_result)
                regime_changed = change_result['regime_changed']

            # Build row
            row = {
                entity_col: entity_id,
                signal_col: signal_id,
                'source_signal': signal_id,
                time_col: window_end,
                'window_start': window_start,
                'window_size': window_size,
                'n_observations': len(window_values),

                # 6 Axes
                'memory': axis_result['memory'],
                'periodicity': axis_result['periodicity'],
                'volatility': axis_result['volatility'],
                'discontinuity': axis_result['discontinuity'],
                'impulsivity': axis_result['impulsivity'],
                'complexity': axis_result['complexity'],

                # Classification
                'classification': classification,
                'recommended_engines': ','.join(engines),
                'regime_changed': regime_changed,
            }

            # Add event counts if present
            if 'discontinuity_events' in axis_result:
                row['n_discontinuities'] = len(axis_result['discontinuity_events'])
            if 'impulse_events' in axis_result:
                row['n_impulses'] = len(axis_result['impulse_events'])

            results.append(row)
            total_windows += 1
            previous_axes = axis_result

        if (i + 1) % 50 == 0:
            logger.info(f"    Processed {i + 1}/{len(pairs)} pairs, {total_windows} windows")

    logger.info(f"  Completed: {total_windows} windows from {len(pairs)} signals")
    return results


# =============================================================================
# HELPERS
# =============================================================================

def _detect_columns(
    df: pl.DataFrame,
    entity_col: str,
    signal_col: str,
    time_col: str,
    value_col: str,
) -> Tuple[str, str, str, str]:
    """Detect appropriate columns from DataFrame."""
    columns = df.columns

    # Entity column
    if entity_col not in columns:
        for candidate in ['entity_id', 'unit_id', 'unit', 'id', 'asset_id']:
            if candidate in columns:
                entity_col = candidate
                break

    # Signal column
    if signal_col not in columns:
        for candidate in ['signal_id', 'signal', 'sensor', 'feature', 'column']:
            if candidate in columns:
                signal_col = candidate
                break

    # Time column
    if time_col not in columns:
        for candidate in ['timestamp', 'cycle', 'time', 't', 'datetime', 'date']:
            if candidate in columns:
                time_col = candidate
                break

    # Value column
    if value_col not in columns:
        for candidate in ['value', 'reading', 'measurement', 'y']:
            if candidate in columns:
                value_col = candidate
                break

    return entity_col, signal_col, time_col, value_col


def _auto_detect_window(
    df: pl.DataFrame,
    time_col: str,
    entity_col: str,
) -> Tuple[int, int]:
    """Auto-detect appropriate window size and stride."""
    # Get average observations per entity
    entity_counts = df.group_by(entity_col).len()
    avg_obs = entity_counts['len'].mean()

    if avg_obs is None:
        return DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE

    # Window = ~1/4 of average entity length
    window_size = max(30, min(500, int(avg_obs / 4)))

    # Stride = ~1/10 of window
    stride = max(1, window_size // 10)

    return window_size, stride


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='PRISM Signal Typology - Six Orthogonal Axes Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The Six Orthogonal Axes:
    1. Memory        - Temporal persistence (Hurst exponent)
    2. Periodicity   - Cyclical structure (FFT)
    3. Volatility    - Variance dynamics (GARCH)
    4. Discontinuity - Level shifts / Heaviside
    5. Impulsivity   - Shocks / Dirac
    6. Complexity    - Predictability (entropy)

Signal Classifications:
    PERSISTENT, PERIODIC, VOLATILE, REGIME_SHIFTING,
    IMPULSIVE, CHAOTIC, TRENDING_VOLATILE, etc.
"""
    )

    parser.add_argument('--input', type=str, help='Input observations parquet')
    parser.add_argument('--output', type=str, help='Output signal_typology parquet')
    parser.add_argument('--entity-col', type=str, default='entity_id')
    parser.add_argument('--signal-col', type=str, default='signal_id')
    parser.add_argument('--time-col', type=str, default='timestamp')
    parser.add_argument('--value-col', type=str, default='value')
    parser.add_argument('--window', type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument('--stride', type=int, default=DEFAULT_STRIDE)
    parser.add_argument('--force', action='store_true', help='Force recompute')
    parser.add_argument('--adaptive', action='store_true', help='Auto-detect window size')

    # Testing
    parser.add_argument('--testing', action='store_true', help='Enable testing mode')
    parser.add_argument('--limit', type=int, help='[TESTING] Limit observations per signal')
    parser.add_argument('--signal', type=str, help='[TESTING] Comma-separated signals to process')

    args = parser.parse_args()

    signals = args.signal.split(',') if args.signal else None

    run(
        input_path=Path(args.input) if args.input else None,
        output_path=Path(args.output) if args.output else None,
        entity_col=args.entity_col,
        signal_col=args.signal_col,
        time_col=args.time_col,
        value_col=args.value_col,
        window_size=args.window,
        stride=args.stride,
        force=args.force,
        adaptive=args.adaptive,
        testing=args.testing,
        limit=args.limit,
        signals=signals,
    )


if __name__ == '__main__':
    main()
