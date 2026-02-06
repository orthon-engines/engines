"""
Stage 12: Z-Score Entry Point
=============================

Pure orchestration - normalizes all prior parquet outputs.

Inputs:
    - All prior parquet files in output directory

Output:
    - zscore.parquet

Computes z-score normalized versions of all numeric columns
across all pipeline outputs for consistent scaling.
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Optional


def run(
    output_dir: str,
    output_path: str = "zscore.parquet",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute z-scores for all numeric columns across pipeline outputs.

    Args:
        output_dir: Directory containing pipeline parquet files
        output_path: Output path for zscore.parquet
        verbose: Print progress

    Returns:
        Z-score normalized DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 12: ZSCORE")
        print("Normalizing all pipeline outputs")
        print("=" * 70)

    output_dir = Path(output_dir)

    # Find all parquet files (exclude zscore itself)
    parquet_files = [
        f for f in output_dir.glob("*.parquet")
        if f.name != "zscore.parquet"
    ]

    if verbose:
        print(f"Found {len(parquet_files)} parquet files")

    results = []

    for parquet_file in parquet_files:
        try:
            df = pl.read_parquet(parquet_file)
            source = parquet_file.stem

            # Get numeric columns
            numeric_cols = [
                c for c in df.columns
                if df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            ]

            if not numeric_cols:
                continue

            # Compute z-scores for each numeric column
            for col in numeric_cols:
                values = df[col].drop_nulls().to_numpy()
                if len(values) < 2:
                    continue

                mean = np.mean(values)
                std = np.std(values)

                if std > 1e-10:
                    results.append({
                        'source': source,
                        'column': col,
                        'mean': float(mean),
                        'std': float(std),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'n_samples': len(values),
                    })

            if verbose:
                print(f"  {source}: {len(numeric_cols)} numeric columns")

        except Exception as e:
            if verbose:
                print(f"  Warning: Could not process {parquet_file.name}: {e}")

    # Build DataFrame
    result = pl.DataFrame(results) if results else pl.DataFrame()

    if len(result) > 0:
        result.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {result.shape}")
        print()
        print("─" * 50)
        print(f"✓ {Path(output_path).absolute()}")
        print("─" * 50)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 12: Z-Score",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes z-score normalization parameters for all pipeline outputs.

Stores mean and std for each numeric column in each source file,
enabling consistent normalization across the pipeline.

Example:
  python -m prism.entry_points.stage_12_zscore output/ -o zscore.parquet
"""
    )
    parser.add_argument('output_dir', help='Directory containing pipeline outputs')
    parser.add_argument('-o', '--output', default='zscore.parquet',
                        help='Output path (default: zscore.parquet)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.output_dir,
        args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
