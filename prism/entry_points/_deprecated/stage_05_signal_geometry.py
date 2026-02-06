"""
Stage 05: Signal Geometry Entry Point
=====================================

Pure orchestration - calls engines/signal_geometry.py for computation.

Inputs:
    - signal_vector.parquet
    - state_vector.parquet
    - state_geometry.parquet (optional, for PCs)

Output:
    - signal_geometry.parquet

Computes per-signal relationships to each state:
    - Distance to centroid
    - Coherence to principal component
    - Contribution (projection magnitude)
    - Residual (orthogonal component)
"""

import argparse
import polars as pl
from pathlib import Path
from typing import Optional

from prism.engines.signal_geometry import (
    compute_signal_geometry,
    compute_full_signal_geometry,
)


def run(
    signal_vector_path: str,
    state_vector_path: str,
    output_path: str = "signal_geometry.parquet",
    state_geometry_path: Optional[str] = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run signal geometry computation.

    Args:
        signal_vector_path: Path to signal_vector.parquet
        state_vector_path: Path to state_vector.parquet
        output_path: Output path for signal_geometry.parquet
        state_geometry_path: Optional path to state_geometry.parquet for PCs
        verbose: Print progress

    Returns:
        Signal geometry DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 05: SIGNAL GEOMETRY")
        print("Per-signal relationships to states")
        print("=" * 70)

    result = compute_signal_geometry(
        signal_vector_path,
        state_vector_path,
        output_path,
        state_geometry_path=state_geometry_path,
        verbose=verbose,
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 05: Signal Geometry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes per-signal relationships to each state:
  - Distance to centroid
  - Coherence to principal component
  - Contribution (projection magnitude)
  - Residual (orthogonal component)

Example:
  python -m prism.entry_points.stage_05_signal_geometry \\
      signal_vector.parquet state_vector.parquet -o signal_geometry.parquet
"""
    )
    parser.add_argument('signal_vector', help='Path to signal_vector.parquet')
    parser.add_argument('state_vector', help='Path to state_vector.parquet')
    parser.add_argument('-o', '--output', default='signal_geometry.parquet',
                        help='Output path (default: signal_geometry.parquet)')
    parser.add_argument('--state-geometry', help='Optional path to state_geometry.parquet')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.signal_vector,
        args.state_vector,
        args.output,
        state_geometry_path=args.state_geometry,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
