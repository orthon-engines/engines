#!/usr/bin/env python3
"""
PRISM Pipeline Runner
=====================

Single entry point for the ORTHON four-layer pipeline.

    fetch.py → observations.parquet (standalone)
    run.py   → signal_typology.parquet
             → behavioral_geometry.parquet
             → dynamical_systems.parquet
             → causal_mechanics.parquet

Behavior:
    1. Checks for each layer's parquet file
    2. Validates schema if file exists
    3. Skips completed layers (correct schema)
    4. On schema mismatch: warns, offers to rename old → legacy_*
    5. Runs remaining layers in order

Usage:
    python -m prism.entry_points.run                    # Run full pipeline
    python -m prism.entry_points.run --from geometry    # Start from Layer 2
    python -m prism.entry_points.run --only typology    # Run only Layer 1
    python -m prism.entry_points.run --force            # Recompute all
    python -m prism.entry_points.run --check            # Validate only, no compute

The Four Layers:
    1. Signal Typology      → WHAT is it?
    2. Behavioral Geometry  → HOW does it behave?
    3. Dynamical Systems    → WHEN/HOW does it change?
    4. Causal Mechanics     → WHY does it change?
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# LAYER DEFINITIONS
# =============================================================================

LAYERS = {
    'typology': {
        'name': 'Signal Typology',
        'layer_num': 1,
        'question': 'WHAT is it?',
        'input': 'observations.parquet',
        'output': 'signal_typology.parquet',
        'required_columns': [
            'entity_id', 'signal_id', 'timestamp',
            'memory', 'periodicity', 'volatility',
            'discontinuity', 'impulsivity', 'complexity',
            'classification',
        ],
        'runner': '_run_signal_typology',
    },
    'geometry': {
        'name': 'Behavioral Geometry',
        'layer_num': 2,
        'question': 'HOW does it behave?',
        'input': 'signal_typology.parquet',
        'output': 'behavioral_geometry.parquet',
        'required_columns': [
            'entity_id', 'timestamp',
            'mean_correlation', 'n_clusters', 'network_density',
        ],
        'runner': '_run_behavioral_geometry',
    },
    'dynamics': {
        'name': 'Dynamical Systems',
        'layer_num': 3,
        'question': 'WHEN/HOW does it change?',
        'input': 'behavioral_geometry.parquet',
        'output': 'dynamical_systems.parquet',
        'required_columns': [
            'entity_id', 'timestamp',
            'regime', 'stability',
        ],
        'runner': '_run_dynamical_systems',
    },
    'mechanics': {
        'name': 'Causal Mechanics',
        'layer_num': 4,
        'question': 'WHY does it change?',
        'input': 'dynamical_systems.parquet',
        'output': 'causal_mechanics.parquet',
        'required_columns': [
            'entity_id', 'timestamp',
            'energy_class', 'equilibrium_class',
        ],
        'runner': '_run_causal_mechanics',
    },
}

LAYER_ORDER = ['typology', 'geometry', 'dynamics', 'mechanics']


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================

def validate_schema(path: Path, required_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate parquet file has required columns.

    Returns:
        (is_valid, missing_columns)
    """
    if not path.exists():
        return False, required_columns

    try:
        df = pl.read_parquet(path)
        existing = set(df.columns)
        missing = [col for col in required_columns if col not in existing]
        return len(missing) == 0, missing
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return False, required_columns


def check_layer_status(data_dir: Path, layer_key: str) -> Dict:
    """
    Check status of a layer's output file.

    Returns:
        {
            'exists': bool,
            'valid_schema': bool,
            'missing_columns': list,
            'row_count': int,
            'path': Path,
        }
    """
    layer = LAYERS[layer_key]
    output_path = data_dir / layer['output']

    status = {
        'exists': output_path.exists(),
        'valid_schema': False,
        'missing_columns': [],
        'row_count': 0,
        'path': output_path,
    }

    if status['exists']:
        is_valid, missing = validate_schema(output_path, layer['required_columns'])
        status['valid_schema'] = is_valid
        status['missing_columns'] = missing

        if is_valid:
            try:
                df = pl.read_parquet(output_path)
                status['row_count'] = len(df)
            except:
                pass

    return status


def rename_to_legacy(path: Path) -> Path:
    """Rename file to legacy_* with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    legacy_name = f"legacy_{path.stem}_{timestamp}{path.suffix}"
    legacy_path = path.parent / legacy_name
    path.rename(legacy_path)
    return legacy_path


# =============================================================================
# LAYER RUNNERS
# =============================================================================

def _run_signal_typology(data_dir: Path, config: Dict) -> bool:
    """Run Signal Typology layer."""
    from prism.engines.characterize import compute_all_axes
    from prism.typology import select_engines, get_primary_classification, detect_regime_change

    input_path = data_dir / 'observations.parquet'
    output_path = data_dir / 'signal_typology.parquet'

    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        logger.error("Run: python -m prism.entry_points.fetch first")
        return False

    logger.info(f"Loading {input_path}...")
    df = pl.read_parquet(input_path)
    logger.info(f"  Loaded {len(df):,} observations")

    # Get window/stride from config
    window_size = config.get('window_size', 252)
    stride = config.get('stride', 21)

    # Adaptive windowing
    if config.get('adaptive', False):
        entity_counts = df.group_by('entity_id').len()
        avg_obs = entity_counts['len'].mean()
        if avg_obs:
            window_size = max(30, min(500, int(avg_obs / 4)))
            stride = max(1, window_size // 10)
        logger.info(f"  Adaptive: window={window_size}, stride={stride}")

    # Process signals
    results = []
    pairs = df.select(['entity_id', 'signal_id']).unique().to_dicts()
    logger.info(f"  Processing {len(pairs)} (entity, signal) pairs...")

    total_windows = 0
    for i, pair in enumerate(pairs):
        entity_id = pair['entity_id']
        signal_id = pair['signal_id']

        signal_df = df.filter(
            (pl.col('entity_id') == entity_id) & (pl.col('signal_id') == signal_id)
        ).sort('timestamp')

        values = signal_df['value'].to_numpy()
        times = signal_df['timestamp'].to_numpy()
        n = len(values)

        if n < window_size:
            continue

        previous_axes = None
        for start in range(0, n - window_size + 1, stride):
            end = start + window_size
            window_values = values[start:end]
            window_end = times[end - 1]

            axis_result = compute_all_axes(window_values)
            classification = get_primary_classification(axis_result)
            engines = select_engines(axis_result)

            regime_changed = False
            if previous_axes is not None:
                change_result = detect_regime_change(previous_axes, axis_result)
                regime_changed = change_result['regime_changed']

            row = {
                'entity_id': entity_id,
                'signal_id': signal_id,
                'timestamp': window_end,
                'window_size': window_size,
                'memory': axis_result['memory'],
                'periodicity': axis_result['periodicity'],
                'volatility': axis_result['volatility'],
                'discontinuity': axis_result['discontinuity'],
                'impulsivity': axis_result['impulsivity'],
                'complexity': axis_result['complexity'],
                'classification': classification,
                'recommended_engines': ','.join(engines),
                'regime_changed': regime_changed,
            }
            results.append(row)
            total_windows += 1
            previous_axes = axis_result

        if (i + 1) % 50 == 0:
            logger.info(f"    Processed {i + 1}/{len(pairs)} pairs")

    if not results:
        logger.warning("No results computed!")
        return False

    output_df = pl.DataFrame(results)
    output_df.write_parquet(output_path)
    logger.info(f"  Wrote {len(output_df):,} rows to {output_path}")
    return True


def _run_behavioral_geometry(data_dir: Path, config: Dict) -> bool:
    """Run Behavioral Geometry layer."""
    # Import the layer
    from prism.layers.behavioral_geometry import BehavioralGeometryLayer, analyze_geometry

    input_path = data_dir / 'signal_typology.parquet'
    output_path = data_dir / 'behavioral_geometry.parquet'

    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        return False

    logger.info(f"Loading {input_path}...")
    df = pl.read_parquet(input_path)
    logger.info(f"  Loaded {len(df):,} rows")

    # Group by entity and timestamp, compute geometry
    results = []
    entities = df['entity_id'].unique().to_list()

    for entity_id in entities:
        entity_df = df.filter(pl.col('entity_id') == entity_id)
        timestamps = entity_df['timestamp'].unique().sort().to_list()

        for ts in timestamps:
            window_df = entity_df.filter(pl.col('timestamp') == ts)

            # Extract axis values as signal vectors
            signals = {}
            for row in window_df.iter_rows(named=True):
                sig_id = row['signal_id']
                signals[sig_id] = [
                    row['memory'], row['periodicity'], row['volatility'],
                    row['discontinuity'], row['impulsivity'], row['complexity']
                ]

            if len(signals) < 2:
                continue

            # Compute basic geometry metrics
            import numpy as np
            signal_matrix = np.array(list(signals.values()))

            # Correlation
            if signal_matrix.shape[0] >= 2:
                corr_matrix = np.corrcoef(signal_matrix)
                upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
                mean_corr = np.nanmean(upper_tri) if len(upper_tri) > 0 else 0.0
            else:
                mean_corr = 0.0

            # Simple clustering (number of distinct groups)
            n_clusters = min(3, len(signals))

            # Network density (proportion of strong correlations)
            strong_pairs = np.sum(np.abs(upper_tri) > 0.5) if len(upper_tri) > 0 else 0
            max_pairs = len(upper_tri) if len(upper_tri) > 0 else 1
            network_density = strong_pairs / max_pairs

            results.append({
                'entity_id': entity_id,
                'timestamp': ts,
                'n_signals': len(signals),
                'mean_correlation': float(mean_corr),
                'n_clusters': n_clusters,
                'network_density': float(network_density),
            })

    if not results:
        logger.warning("No results computed!")
        return False

    output_df = pl.DataFrame(results)
    output_df.write_parquet(output_path)
    logger.info(f"  Wrote {len(output_df):,} rows to {output_path}")
    return True


def _run_dynamical_systems(data_dir: Path, config: Dict) -> bool:
    """Run Dynamical Systems layer."""
    input_path = data_dir / 'behavioral_geometry.parquet'
    output_path = data_dir / 'dynamical_systems.parquet'

    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        return False

    logger.info(f"Loading {input_path}...")
    df = pl.read_parquet(input_path)
    logger.info(f"  Loaded {len(df):,} rows")

    # Compute dynamics from geometry evolution
    results = []
    entities = df['entity_id'].unique().to_list()

    for entity_id in entities:
        entity_df = df.filter(pl.col('entity_id') == entity_id).sort('timestamp')

        if len(entity_df) < 2:
            continue

        # Compute changes over time
        corrs = entity_df['mean_correlation'].to_numpy()
        densities = entity_df['network_density'].to_numpy()
        timestamps = entity_df['timestamp'].to_list()

        for i in range(1, len(entity_df)):
            corr_change = corrs[i] - corrs[i-1]
            density_change = densities[i] - densities[i-1]

            # Simple regime detection
            if abs(corr_change) > 0.2:
                regime = 'TRANSITIONING'
            elif corrs[i] > 0.7:
                regime = 'COUPLED'
            elif corrs[i] < 0.3:
                regime = 'DECOUPLED'
            else:
                regime = 'MODERATE'

            # Stability assessment
            if abs(corr_change) < 0.05 and abs(density_change) < 0.05:
                stability = 'STABLE'
            elif abs(corr_change) > 0.15 or abs(density_change) > 0.15:
                stability = 'UNSTABLE'
            else:
                stability = 'EVOLVING'

            results.append({
                'entity_id': entity_id,
                'timestamp': timestamps[i],
                'regime': regime,
                'stability': stability,
                'correlation_change': float(corr_change),
                'density_change': float(density_change),
            })

    if not results:
        logger.warning("No results computed!")
        return False

    output_df = pl.DataFrame(results)
    output_df.write_parquet(output_path)
    logger.info(f"  Wrote {len(output_df):,} rows to {output_path}")
    return True


def _run_causal_mechanics(data_dir: Path, config: Dict) -> bool:
    """Run Causal Mechanics layer."""
    input_path = data_dir / 'dynamical_systems.parquet'
    output_path = data_dir / 'causal_mechanics.parquet'

    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        return False

    logger.info(f"Loading {input_path}...")
    df = pl.read_parquet(input_path)
    logger.info(f"  Loaded {len(df):,} rows")

    # Compute mechanics from dynamics
    results = []
    entities = df['entity_id'].unique().to_list()

    for entity_id in entities:
        entity_df = df.filter(pl.col('entity_id') == entity_id).sort('timestamp')

        for row in entity_df.iter_rows(named=True):
            # Energy class based on change magnitude
            total_change = abs(row['correlation_change']) + abs(row['density_change'])
            if total_change > 0.3:
                energy_class = 'HIGH_ENERGY'
            elif total_change > 0.1:
                energy_class = 'MODERATE_ENERGY'
            else:
                energy_class = 'LOW_ENERGY'

            # Equilibrium class based on stability
            if row['stability'] == 'STABLE':
                equilibrium_class = 'AT_EQUILIBRIUM'
            elif row['regime'] == 'TRANSITIONING':
                equilibrium_class = 'FAR_FROM_EQUILIBRIUM'
            else:
                equilibrium_class = 'NEAR_EQUILIBRIUM'

            results.append({
                'entity_id': entity_id,
                'timestamp': row['timestamp'],
                'regime': row['regime'],
                'stability': row['stability'],
                'energy_class': energy_class,
                'equilibrium_class': equilibrium_class,
            })

    if not results:
        logger.warning("No results computed!")
        return False

    output_df = pl.DataFrame(results)
    output_df.write_parquet(output_path)
    logger.info(f"  Wrote {len(output_df):,} rows to {output_path}")
    return True


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(
    data_dir: Path,
    from_layer: Optional[str] = None,
    only_layer: Optional[str] = None,
    force: bool = False,
    check_only: bool = False,
    config: Optional[Dict] = None,
) -> bool:
    """
    Run the ORTHON pipeline.

    Args:
        data_dir: Directory containing parquet files
        from_layer: Start from this layer (skip earlier)
        only_layer: Run only this layer
        force: Force recompute all
        check_only: Only validate, don't compute
        config: Configuration dict

    Returns:
        True if pipeline completed successfully
    """
    config = config or {}

    logger.info("=" * 60)
    logger.info("  ORTHON PIPELINE")
    logger.info("  Four-Layer Framework")
    logger.info("=" * 60)

    # Determine which layers to run
    if only_layer:
        layers_to_run = [only_layer]
    elif from_layer:
        start_idx = LAYER_ORDER.index(from_layer)
        layers_to_run = LAYER_ORDER[start_idx:]
    else:
        layers_to_run = LAYER_ORDER

    # Check status of all layers
    logger.info("\nLayer Status:")
    logger.info("-" * 50)

    for layer_key in LAYER_ORDER:
        layer = LAYERS[layer_key]
        status = check_layer_status(data_dir, layer_key)

        if status['exists']:
            if status['valid_schema']:
                status_str = f"OK ({status['row_count']:,} rows)"
            else:
                status_str = f"SCHEMA MISMATCH (missing: {status['missing_columns']})"
        else:
            status_str = "NOT FOUND"

        marker = "→" if layer_key in layers_to_run else " "
        logger.info(f"  {marker} Layer {layer['layer_num']}: {layer['name']:<20} {status_str}")

    if check_only:
        logger.info("\n[CHECK ONLY] No computation performed.")
        return True

    # Run layers
    logger.info("\n" + "=" * 60)

    for layer_key in layers_to_run:
        layer = LAYERS[layer_key]
        status = check_layer_status(data_dir, layer_key)

        logger.info(f"\nLayer {layer['layer_num']}: {layer['name']}")
        logger.info(f"  Question: {layer['question']}")

        # Check if we can skip
        if status['exists'] and status['valid_schema'] and not force:
            logger.info(f"  SKIPPING (valid output exists with {status['row_count']:,} rows)")
            continue

        # Handle schema mismatch
        if status['exists'] and not status['valid_schema']:
            logger.warning(f"  Schema mismatch! Missing columns: {status['missing_columns']}")

            # Ask user
            print(f"\n  File {status['path']} has incompatible schema.")
            print(f"  Rename to legacy and create new? [y/N]: ", end="")

            try:
                response = input().strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\nCancelled.")
                return False

            if response == 'y':
                legacy_path = rename_to_legacy(status['path'])
                logger.info(f"  Renamed to {legacy_path}")
            else:
                logger.info("  Skipping layer (user declined)")
                continue

        # Run the layer
        logger.info(f"  Running...")
        runner_name = layer['runner']
        runner_func = globals()[runner_name]

        try:
            success = runner_func(data_dir, config)
            if not success:
                logger.error(f"  Layer {layer['name']} failed!")
                return False
            logger.info(f"  Layer {layer['name']} completed.")
        except Exception as e:
            logger.error(f"  Layer {layer['name']} failed: {e}")
            return False

    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 60)
    return True


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='PRISM Pipeline Runner - ORTHON Four-Layer Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The Four Layers:
    1. typology   - Signal Typology:      WHAT is it?
    2. geometry   - Behavioral Geometry:  HOW does it behave?
    3. dynamics   - Dynamical Systems:    WHEN/HOW does it change?
    4. mechanics  - Causal Mechanics:     WHY does it change?

Examples:
    python -m prism.entry_points.run                    # Full pipeline
    python -m prism.entry_points.run --from geometry    # Start from Layer 2
    python -m prism.entry_points.run --only typology    # Only Layer 1
    python -m prism.entry_points.run --check            # Validate only
    python -m prism.entry_points.run --force            # Recompute all
"""
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Data directory (default: data/)'
    )
    parser.add_argument(
        '--from',
        dest='from_layer',
        choices=LAYER_ORDER,
        help='Start from this layer'
    )
    parser.add_argument(
        '--only',
        dest='only_layer',
        choices=LAYER_ORDER,
        help='Run only this layer'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recompute all layers'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Validate only, no computation'
    )
    parser.add_argument(
        '--adaptive',
        action='store_true',
        help='Auto-detect window size'
    )
    parser.add_argument(
        '--window',
        type=int,
        default=252,
        help='Window size (default: 252)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=21,
        help='Stride (default: 21)'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    config = {
        'adaptive': args.adaptive,
        'window_size': args.window,
        'stride': args.stride,
    }

    success = run_pipeline(
        data_dir=data_dir,
        from_layer=args.from_layer,
        only_layer=args.only_layer,
        force=args.force,
        check_only=args.check,
        config=config,
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
