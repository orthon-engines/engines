"""
Stage 17: Backward FTLE Entry Point
===================================

Backward FTLE reveals attracting structures - where trajectories converge TO.

This is a thin wrapper around stage_08_ftle with direction='backward'.
Forward FTLE (stage_08) + Backward FTLE (stage_17) together reveal the
full Lagrangian Coherent Structure.

Forward FTLE:  Repelling structures (where trajectories diverge FROM)
Backward FTLE: Attracting structures (where trajectories converge TO)

Inputs:
    - observations.parquet

Output:
    - ftle_backward.parquet

PRISM computes FTLE values. ORTHON interprets attractors as failure states.
"""

import argparse
from pathlib import Path

from prism.entry_points.stage_08_ftle import run as _run


def run(
    observations_path: str,
    output_path: str = "ftle_backward.parquet",
    min_samples: int = 200,
    method: str = 'rosenstein',
    verbose: bool = True,
    intervention: dict = None,
    direction: str = 'backward',  # Force backward
):
    """
    Compute backward FTLE (attracting structures).

    This is a wrapper around stage_08_ftle with direction='backward'.
    """
    return _run(
        observations_path=observations_path,
        output_path=output_path,
        min_samples=min_samples,
        method=method,
        verbose=verbose,
        intervention=intervention,
        direction='backward',  # Always backward
    )


def main():
    parser = argparse.ArgumentParser(
        description="Stage 17: Backward FTLE (Attracting Structures)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes backward FTLE (attracting structures).

Backward FTLE = Lyapunov exponent on time-reversed trajectory.
Reveals where trajectories converge TO (failure attractors).

Use with stage_08 (forward FTLE) for complete LCS analysis.

Example:
  python -m prism.entry_points.stage_17_ftle_backward \\
      observations.parquet -o ftle_backward.parquet
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('-o', '--output', default='ftle_backward.parquet',
                        help='Output path (default: ftle_backward.parquet)')
    parser.add_argument('--min-samples', type=int, default=200,
                        help='Minimum samples per signal (default: 200)')
    parser.add_argument('--method', choices=['rosenstein', 'kantz'], default='rosenstein',
                        help='Algorithm (default: rosenstein)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        args.output,
        min_samples=args.min_samples,
        method=args.method,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
