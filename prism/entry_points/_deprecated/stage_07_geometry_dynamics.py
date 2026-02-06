"""
Stage 07: Geometry Dynamics Entry Point
=======================================

Pure orchestration - calls engines/geometry_dynamics.py for computation.

Inputs:
    - state_geometry.parquet

Output:
    - geometry_dynamics.parquet

Computes differential geometry of state evolution:
    - Velocity (first derivative)
    - Acceleration (second derivative)
    - Jerk (third derivative)
    - Curvature
"""

import argparse
import polars as pl
from pathlib import Path
from typing import Optional

from prism.engines.geometry_dynamics import (
    compute_geometry_dynamics,
    compute_all_dynamics,
)


def run(
    state_geometry_path: str,
    output_path: str = "geometry_dynamics.parquet",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run geometry dynamics computation.

    Args:
        state_geometry_path: Path to state_geometry.parquet
        output_path: Output path for geometry_dynamics.parquet
        verbose: Print progress

    Returns:
        Geometry dynamics DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 07: GEOMETRY DYNAMICS")
        print("Differential geometry of state evolution")
        print("=" * 70)

    # Load state geometry
    state_geom = pl.read_parquet(state_geometry_path)

    if verbose:
        print(f"Loaded state geometry: {len(state_geom)} rows")

    result = compute_geometry_dynamics(
        state_geom,
        output_path=output_path,
        verbose=verbose,
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 07: Geometry Dynamics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes differential geometry of state evolution:
  - Velocity (first derivative)
  - Acceleration (second derivative)
  - Jerk (third derivative)
  - Curvature

Example:
  python -m prism.entry_points.stage_07_geometry_dynamics \\
      state_geometry.parquet -o geometry_dynamics.parquet
"""
    )
    parser.add_argument('state_geometry', help='Path to state_geometry.parquet')
    parser.add_argument('-o', '--output', default='geometry_dynamics.parquet',
                        help='Output path (default: geometry_dynamics.parquet)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.state_geometry,
        args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
