#!/usr/bin/env python3
"""
PRISM Analyze - Unified Pipeline Runner
=======================================

Thin wrapper that runs all PRISM entry points in sequence.

Usage:
    python -m prism.entry_points.analyze           # Full pipeline
    python -m prism.entry_points.analyze --force   # Recompute all

Outputs:
    data/vector.parquet   - Signal metrics
    data/geometry.parquet - Structural metrics
    data/dynamics.parquet - Temporal metrics
    data/physics.parquet  - Physics metrics
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

from prism.db.parquet_store import get_path, ensure_directory, OBSERVATIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Entry point modules (in dependency order)
LAYERS = ['vector', 'geometry', 'dynamics', 'state', 'physics', 'fields', 'systems']


def run_entry_point(layer: str, force: bool = False) -> bool:
    """
    Run a PRISM entry point.

    Args:
        layer: Entry point name (vector, geometry, dynamics, physics)
        force: Pass --force flag

    Returns:
        True if successful
    """
    module = f"prism.entry_points.{layer}"
    cmd = [sys.executable, "-m", module]

    if force:
        cmd.append("--force")

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to run {layer}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="PRISM Compute - Run all entry points",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Layers:
    vector    - Signal-level metrics (28 engines)
    geometry  - Structural relationships (13 engines)
    dynamics  - Temporal dynamics (11 engines)
    physics   - Energy/momentum metrics (8 engines)

Examples:
    python -m prism.entry_points.compute           # Run all
    python -m prism.entry_points.compute --force   # Force recompute
    python -m prism.entry_points.compute vector    # Vector only
    python -m prism.entry_points.compute geometry dynamics  # Multiple
        """
    )

    parser.add_argument(
        'layers',
        nargs='*',
        choices=LAYERS + [[]],
        default=[],
        help='Specific layers to run (default: all)'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Recompute even if output exists'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PRISM Compute Pipeline")
    logger.info("=" * 60)

    ensure_directory()

    # Check observations exist
    obs_path = get_path(OBSERVATIONS)
    if not obs_path.exists():
        logger.error(f"observations.parquet not found at {obs_path}")
        logger.error("Run: python -m prism.entry_points.fetch")
        return 1

    # Determine which layers to run
    layers_to_run = args.layers if args.layers else LAYERS

    logger.info(f"Layers: {', '.join(layers_to_run)}")
    logger.info(f"Force: {args.force}")
    logger.info("")

    # Run each layer
    start = time.time()
    success = True

    for layer in layers_to_run:
        layer_start = time.time()

        if not run_entry_point(layer, args.force):
            logger.error(f"Layer {layer} failed")
            success = False
        else:
            elapsed = time.time() - layer_start
            logger.info(f"Layer {layer} complete ({elapsed:.1f}s)")

        logger.info("")

    total_elapsed = time.time() - start

    # Summary
    logger.info("=" * 60)
    logger.info("Pipeline Complete")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_elapsed:.1f}s")

    # Show output files
    data_path = obs_path.parent
    for layer in layers_to_run:
        output_path = data_path / f"{layer}.parquet"
        if output_path.exists():
            size_kb = output_path.stat().st_size / 1024
            logger.info(f"  {layer}.parquet: {size_kb:.1f} KB")
        else:
            logger.info(f"  {layer}.parquet: NOT CREATED")

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
