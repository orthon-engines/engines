"""
Stage 06: Signal Pairwise Entry Point
=====================================

Pure orchestration - calls engines/signal_pairwise.py for computation.

Inputs:
    - signal_vector.parquet

Output:
    - signal_pairwise.parquet

Computes pairwise relationships between signals:
    - Correlation
    - Mutual information
    - Phase synchronization
    - Coherence changes over time
"""

import argparse
import polars as pl
from pathlib import Path
from typing import Optional

from prism.engines.signal_pairwise import (
    compute_signal_pairwise,
)


def run(
    signal_vector_path: str,
    output_path: str = "signal_pairwise.parquet",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run signal pairwise computation.

    Args:
        signal_vector_path: Path to signal_vector.parquet
        output_path: Output path for signal_pairwise.parquet
        verbose: Print progress

    Returns:
        Signal pairwise DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 06: SIGNAL PAIRWISE")
        print("Pairwise relationships between signals")
        print("=" * 70)

    result = compute_signal_pairwise(
        signal_vector_path,
        output_path,
        verbose=verbose,
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 06: Signal Pairwise",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes pairwise relationships between signals:
  - Correlation
  - Mutual information
  - Phase synchronization

Example:
  python -m prism.entry_points.stage_06_signal_pairwise \\
      signal_vector.parquet -o signal_pairwise.parquet
"""
    )
    parser.add_argument('signal_vector', help='Path to signal_vector.parquet')
    parser.add_argument('-o', '--output', default='signal_pairwise.parquet',
                        help='Output path (default: signal_pairwise.parquet)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.signal_vector,
        args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
