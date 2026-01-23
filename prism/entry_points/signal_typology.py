"""
PRISM Signal Typology
=====================

Entry point for signal typology analysis. Uses the six orthogonal axes
framework to classify signals into behavioral archetypes.

The Six Orthogonal Axes:
    1. Memory        - Long-range dependence (Hurst, ACF decay, spectral slope)
    2. Information   - Complexity (permutation entropy, sample entropy)
    3. Recurrence    - Pattern structure (RQA: determinism, laminarity)
    4. Volatility    - Amplitude dynamics (GARCH persistence, Hilbert envelope)
    5. Frequency     - Spectral character (centroid, bandwidth, 1/f character)
    6. Dynamics      - Stability (Lyapunov exponent, embedding dimension)

Plus: Structural Discontinuity Detection (Dirac impulse + Heaviside step)

Output: signal_typology.parquet

Pipeline:
    raw → signal_typology → behavioral_geometry → phase_state → dynamical_systems

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

import numpy as np
import polars as pl

# Core imports
from prism.db.parquet_store import (
    ensure_directory,
    get_data_root,
    get_path,
    OBSERVATIONS,
)
from prism.db.polars_io import read_parquet, write_parquet_atomic

# Typology package
from prism.typology import (
    analyze_signal,
    analyze_windowed,
    SignalTypology,
    ARCHETYPES,
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
    hurst_method: str = 'dfa',
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
        hurst_method: 'dfa' or 'rs' for Hurst estimation

    Returns:
        Path to output parquet file
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("  SIGNAL TYPOLOGY")
    logger.info("  Six Orthogonal Axes + Discontinuity Detection")
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
        hurst_method=hurst_method,
    )

    if not results:
        logger.warning("No results computed!")
        return output_path

    # Convert to DataFrame
    output_df = pl.DataFrame(results)

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

    # Archetype distribution
    if 'archetype' in output_df.columns:
        archetype_counts = output_df.group_by('archetype').len().sort('len', descending=True)
        logger.info("")
        logger.info("Archetype Distribution:")
        for row in archetype_counts.iter_rows(named=True):
            logger.info(f"  {row['archetype']}: {row['len']:,}")

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
    hurst_method: str = 'dfa',
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

        # Analyze using windowed approach
        previous_typology = None

        for start in range(0, n - window_size + 1, stride):
            end = start + window_size
            window_values = values[start:end]
            window_start = times[start]
            window_end = times[end - 1]

            # Analyze this window
            typology = analyze_signal(
                series=window_values,
                entity_id=str(entity_id),
                signal_id=str(signal_id),
                window_start=datetime.now(),  # Could parse from times
                window_end=datetime.now(),
                previous_typology=previous_typology,
                hurst_method=hurst_method,
            )

            # Convert to row
            row = _typology_to_row(
                typology=typology,
                entity_id=entity_id,
                signal_id=signal_id,
                timestamp=window_end,
                window_start=window_start,
                window_size=window_size,
                entity_col=entity_col,
                signal_col=signal_col,
                time_col=time_col,
            )

            results.append(row)
            total_windows += 1
            previous_typology = typology

        if (i + 1) % 50 == 0:
            logger.info(f"    Processed {i + 1}/{len(pairs)} pairs, {total_windows} windows")

    logger.info(f"  Completed: {total_windows} windows from {len(pairs)} signals")
    return results


def _typology_to_row(
    typology: SignalTypology,
    entity_id: Any,
    signal_id: Any,
    timestamp: Any,
    window_start: Any,
    window_size: int,
    entity_col: str,
    signal_col: str,
    time_col: str,
) -> Dict[str, Any]:
    """Convert SignalTypology to a flat dictionary row."""
    row = {
        # Identity
        entity_col: entity_id,
        signal_col: signal_id,
        'source_signal': signal_id,
        time_col: timestamp,
        'window_start': window_start,
        'window_size': window_size,
        'n_observations': typology.n_observations,

        # === AXIS 1: MEMORY ===
        'hurst_exponent': typology.memory.hurst_exponent,
        'hurst_method': typology.memory.hurst_method,
        'hurst_confidence': typology.memory.hurst_confidence,
        'acf_decay_type': typology.memory.acf_decay_type.value,
        'acf_half_life': typology.memory.acf_half_life,
        'spectral_slope': typology.memory.spectral_slope,
        'spectral_slope_r2': typology.memory.spectral_slope_r2,
        'memory_class': typology.memory.memory_class.value,

        # === AXIS 2: INFORMATION ===
        'entropy_permutation': typology.information.entropy_permutation,
        'entropy_sample': typology.information.entropy_sample,
        'entropy_rate': typology.information.entropy_rate,
        'information_class': typology.information.information_class.value,

        # === AXIS 3: RECURRENCE ===
        'determinism': typology.recurrence.determinism,
        'laminarity': typology.recurrence.laminarity,
        'entropy_diagonal': typology.recurrence.entropy_diagonal,
        'recurrence_rate': typology.recurrence.recurrence_rate,
        'trapping_time': typology.recurrence.trapping_time,
        'max_diagonal': typology.recurrence.max_diagonal,
        'avg_diagonal': typology.recurrence.avg_diagonal,
        'recurrence_class': typology.recurrence.recurrence_class.value,

        # === AXIS 4: VOLATILITY ===
        'garch_alpha': typology.volatility.garch_alpha,
        'garch_beta': typology.volatility.garch_beta,
        'garch_persistence': typology.volatility.garch_persistence,
        'garch_omega': typology.volatility.garch_omega,
        'garch_unconditional': typology.volatility.garch_unconditional,
        'hilbert_amplitude_mean': typology.volatility.hilbert_amplitude_mean,
        'hilbert_amplitude_std': typology.volatility.hilbert_amplitude_std,
        'volatility_class': typology.volatility.volatility_class.value,

        # === AXIS 5: FREQUENCY ===
        'spectral_centroid': typology.frequency.spectral_centroid,
        'spectral_bandwidth': typology.frequency.spectral_bandwidth,
        'spectral_low_high_ratio': typology.frequency.spectral_low_high_ratio,
        'spectral_rolloff': typology.frequency.spectral_rolloff,
        'frequency_class': typology.frequency.frequency_class.value,

        # === AXIS 6: DYNAMICS ===
        'lyapunov_exponent': typology.dynamics.lyapunov_exponent,
        'lyapunov_confidence': typology.dynamics.lyapunov_confidence,
        'embedding_dimension': typology.dynamics.embedding_dimension,
        'correlation_dimension': typology.dynamics.correlation_dimension,
        'dynamics_class': typology.dynamics.dynamics_class.value,

        # === STRUCTURAL DISCONTINUITY ===
        'dirac_detected': typology.discontinuity.dirac.detected,
        'dirac_count': typology.discontinuity.dirac.count,
        'dirac_max_magnitude': typology.discontinuity.dirac.max_magnitude,
        'dirac_mean_magnitude': typology.discontinuity.dirac.mean_magnitude,
        'dirac_mean_half_life': typology.discontinuity.dirac.mean_half_life,
        'dirac_up_ratio': typology.discontinuity.dirac.up_ratio,

        'heaviside_detected': typology.discontinuity.heaviside.detected,
        'heaviside_count': typology.discontinuity.heaviside.count,
        'heaviside_max_magnitude': typology.discontinuity.heaviside.max_magnitude,
        'heaviside_mean_magnitude': typology.discontinuity.heaviside.mean_magnitude,
        'heaviside_up_ratio': typology.discontinuity.heaviside.up_ratio,

        'discontinuity_mean_interval': typology.discontinuity.mean_interval,
        'discontinuity_interval_cv': typology.discontinuity.interval_cv,
        'discontinuity_dominant_period': typology.discontinuity.dominant_period,
        'discontinuity_is_accelerating': typology.discontinuity.is_accelerating,

        # === CLASSIFICATION ===
        'archetype': typology.archetype,
        'archetype_distance': typology.archetype_distance,
        'secondary_archetype': typology.secondary_archetype,
        'secondary_distance': typology.secondary_distance,
        'boundary_proximity': typology.boundary_proximity,
        'fingerprint': typology.fingerprint.tolist(),

        # === TRANSITION ===
        'regime_transition': typology.regime_transition.value,
        'axes_moving': typology.axes_moving,
        'axes_stable': typology.axes_stable,

        # === SUMMARY ===
        'summary': typology.summary,
        'confidence': typology.confidence,
        'alerts': typology.alerts,
    }

    return row


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
    1. Memory        - Long-range dependence (Hurst exponent)
    2. Information   - Complexity (permutation/sample entropy)
    3. Recurrence    - Pattern structure (RQA determinism)
    4. Volatility    - Amplitude dynamics (GARCH persistence)
    5. Frequency     - Spectral character (bandwidth)
    6. Dynamics      - Stability (Lyapunov exponent)

Archetypes:
    Stable Trend, Momentum Decay, Trending Volatile,
    Mean Reversion Stable, Mean Reversion Volatile,
    Random Walk, Consolidation, Chaotic, Edge of Chaos,
    Regime Transition, Post-Shock Recovery, Periodic, Quasi-Periodic
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
    parser.add_argument('--hurst-method', type=str, default='dfa', choices=['dfa', 'rs'])
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
        hurst_method=args.hurst_method,
        force=args.force,
        adaptive=args.adaptive,
        testing=args.testing,
        limit=args.limit,
        signals=signals,
    )


if __name__ == '__main__':
    main()
