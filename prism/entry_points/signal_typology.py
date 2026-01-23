"""
PRISM Signal Typology
=====================

Entry point for signal typology analysis. Orchestrates three stages:

    Stage 1: CHARACTERIZATION (pre-compute)
        → Determine signal type, stationarity, window sizing, engine flags

    Stage 2: ENGINE COMPUTATION
        → Run all applicable engines to compute behavioral metrics

    Stage 3: CLASSIFICATION & INTERPRETATION
        → Assign typology labels, generate summaries, detect shifts

Output: signal_typology.parquet

This is the INFORMATION layer - researchers get answers, not data dumps.

Usage:
    # Full run
    python -m prism.entry_points.signal_typology

    # Force recompute
    python -m prism.entry_points.signal_typology --force

    # Adaptive windowing
    python -m prism.entry_points.signal_typology --adaptive

    # Testing mode
    python -m prism.entry_points.signal_typology --testing --limit 100

Pipeline:
    raw → signal_typology → behavioral_geometry → phase_state → dynamical_systems
"""

import argparse
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

# Core imports
from prism.db.parquet_store import (
    ensure_directory,
    get_data_root,
    get_path,
    OBSERVATIONS,
    VECTOR,
)
from prism.db.polars_io import read_parquet, write_parquet_atomic

# Characterization
from prism.engines.characterize import (
    characterize_signal,
    get_engines_from_characterization,
    get_characterization_summary,
)

# Engine computation
from prism.modules.signal_behavior import (
    compute_all_metrics,
    get_engine_list_for_characterization,
    CORE_ENGINES,
)

# Classification
from prism.modules.typology_classifier import (
    classify,
    generate_summary,
    generate_summary_text,
    detect_shifts,
    compute_typology,
    get_persistence_label,
)

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
# STAGE 1: CHARACTERIZATION
# =============================================================================

def characterize_signals(
    df: pl.DataFrame,
    entity_col: str = 'entity_id',
    signal_col: str = 'signal_id',
    time_col: str = 'timestamp',
    value_col: str = 'value',
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> Dict[str, Dict[str, Any]]:
    """
    Run characterization on all signals.

    Returns dictionary keyed by (entity_id, signal_id) with characterization results.
    """
    logger.info("Stage 1: Characterizing signals...")

    characterizations = {}

    # Get unique (entity, signal) pairs
    pairs = df.select([entity_col, signal_col]).unique().to_dicts()
    logger.info(f"  Found {len(pairs)} unique (entity, signal) pairs")

    for i, pair in enumerate(pairs):
        entity_id = pair[entity_col]
        signal_id = pair[signal_col]

        # Get signal data
        signal_df = df.filter(
            (pl.col(entity_col) == entity_id) & (pl.col(signal_col) == signal_id)
        ).sort(time_col)

        values = signal_df[value_col].to_numpy()

        if len(values) < MIN_OBSERVATIONS:
            continue

        # Characterize
        try:
            char_result = characterize_signal(
                values=values,
                signal_id=f"{entity_id}_{signal_id}",
                window_size=min(window_size, len(values)),
            )
            characterizations[(entity_id, signal_id)] = char_result.to_dict()
        except Exception as e:
            logger.warning(f"  Failed to characterize {entity_id}/{signal_id}: {e}")
            continue

        if (i + 1) % 100 == 0:
            logger.info(f"  Characterized {i + 1}/{len(pairs)} signals")

    logger.info(f"  Completed characterization for {len(characterizations)} signals")
    return characterizations


# =============================================================================
# STAGE 2: ENGINE COMPUTATION
# =============================================================================

def compute_signal_metrics(
    df: pl.DataFrame,
    characterizations: Dict[str, Dict[str, Any]],
    entity_col: str = 'entity_id',
    signal_col: str = 'signal_id',
    time_col: str = 'timestamp',
    value_col: str = 'value',
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
) -> List[Dict[str, Any]]:
    """
    Compute engine metrics for all signals using sliding windows.

    Returns list of result dictionaries (one per window).
    """
    logger.info("Stage 2: Computing engine metrics...")

    results = []

    pairs = df.select([entity_col, signal_col]).unique().to_dicts()
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

        if len(values) < window_size:
            continue

        # Get characterization
        char_key = (entity_id, signal_id)
        characterization = characterizations.get(char_key, {})

        # Determine engines to run
        engines = get_engine_list_for_characterization(characterization)

        # Process windows
        prev_metrics = None

        for start in range(0, len(values) - window_size + 1, stride):
            window_values = values[start:start + window_size]
            window_start = times[start]
            window_end = times[start + window_size - 1]

            # Compute engine metrics
            metrics = compute_all_metrics(window_values, characterization)

            # Build result row
            row = {
                # Identity
                entity_col: entity_id,
                signal_col: signal_id,
                'source_signal': signal_id,
                time_col: window_end,
                'window_start': window_start,
                'window_size': window_size,

                # Characterization
                'signal_type': characterization.get('frequency', 'unknown'),
                'is_stationary': characterization.get('ax_stationarity', 0.5) > 0.5,
                'ax_stationarity': characterization.get('ax_stationarity'),
                'ax_memory': characterization.get('ax_memory'),
                'ax_periodicity': characterization.get('ax_periodicity'),
                'ax_complexity': characterization.get('ax_complexity'),
                'ax_determinism': characterization.get('ax_determinism'),
                'ax_volatility': characterization.get('ax_volatility'),
                'recommended_window': characterization.get('window_size', window_size),

                # All engine metrics
                **metrics,
            }

            # Store this row's metrics for next iteration
            current_metrics = metrics
            results.append(row)
            total_windows += 1

            # Update prev_metrics for next window
            prev_metrics = current_metrics

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i + 1}/{len(pairs)} signals, {total_windows} windows")

    logger.info(f"  Completed: {len(results)} windows from {len(pairs)} signals")
    return results


# =============================================================================
# STAGE 3: CLASSIFICATION & INTERPRETATION
# =============================================================================

def classify_and_interpret(
    results: List[Dict[str, Any]],
    entity_col: str = 'entity_id',
    signal_col: str = 'signal_id',
    time_col: str = 'timestamp',
) -> List[Dict[str, Any]]:
    """
    Add typology classification and interpretation to results.

    For each row, computes:
        - Typology classification
        - Bullet summaries
        - Shift detection (vs previous window)
    """
    logger.info("Stage 3: Classifying and interpreting...")

    # Group by (entity, signal) for shift detection
    from collections import defaultdict
    signal_windows = defaultdict(list)

    for row in results:
        key = (row[entity_col], row[signal_col])
        signal_windows[key].append(row)

    # Sort each signal's windows by time
    for key in signal_windows:
        signal_windows[key].sort(key=lambda r: r[time_col])

    # Process each signal
    enriched_results = []

    for key, windows in signal_windows.items():
        prev_metrics = None

        for row in windows:
            # Extract metrics for classification
            metrics = {k: v for k, v in row.items()
                       if isinstance(v, (int, float)) and not k.startswith('ax_')}

            # Compute typology
            typology = compute_typology(metrics, prev_metrics)

            # Add to row
            row.update(typology)

            enriched_results.append(row)

            # Update prev_metrics
            prev_metrics = metrics

    logger.info(f"  Classified {len(enriched_results)} windows")
    return enriched_results


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
    logger.info("  geometry leads — orthon")
    logger.info("=" * 60)

    # Resolve paths
    if input_path is None:
        input_path = get_path(OBSERVATIONS)
    else:
        input_path = Path(input_path)

    if output_path is None:
        output_path = get_path(SIGNAL_TYPOLOGY)
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

    # === STAGE 1: CHARACTERIZATION ===
    characterizations = characterize_signals(
        df=df,
        entity_col=entity_col,
        signal_col=signal_col,
        time_col=time_col,
        value_col=value_col,
        window_size=window_size,
    )

    # === STAGE 2: ENGINE COMPUTATION ===
    results = compute_signal_metrics(
        df=df,
        characterizations=characterizations,
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

    # === STAGE 3: CLASSIFICATION ===
    enriched_results = classify_and_interpret(
        results=results,
        entity_col=entity_col,
        signal_col=signal_col,
        time_col=time_col,
    )

    # Convert to DataFrame
    output_df = pl.DataFrame(enriched_results)

    # Write output
    ensure_directory(output_path.parent)
    write_parquet_atomic(output_df, output_path)

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("")
    logger.info(f"Output: {output_path}")
    logger.info(f"  Rows: {len(output_df):,}")
    logger.info(f"  Columns: {len(output_df.columns)}")
    logger.info(f"  Elapsed: {elapsed:.1f}s")

    # Typology distribution
    if 'typology' in output_df.columns:
        typology_counts = output_df.group_by('typology').len().sort('len', descending=True)
        logger.info("")
        logger.info("Typology Distribution:")
        for row in typology_counts.iter_rows(named=True):
            logger.info(f"  {row['typology']}: {row['len']:,}")

    return output_path


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
        description='PRISM Signal Typology - Behavioral classification and interpretation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
