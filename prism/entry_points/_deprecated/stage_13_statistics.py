"""
Stage 13: Statistics Entry Point
================================

Pure orchestration - calls engines/sql/statistics.sql for computation.

Inputs:
    - observations.parquet

Output:
    - statistics.parquet

Computes summary statistics per signal:
    - mean, std, min, max
    - count, null_count
    - quartiles (25%, 50%, 75%)
"""

import argparse
import polars as pl
import duckdb
from pathlib import Path
from typing import Optional

from prism.engines.sql import get_sql


def run(
    observations_path: str,
    output_path: str = "statistics.parquet",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run statistics computation using SQL engine.

    Args:
        observations_path: Path to observations.parquet
        output_path: Output path for statistics.parquet
        verbose: Print progress

    Returns:
        Statistics DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 13: STATISTICS")
        print("Summary statistics per signal")
        print("=" * 70)

    # Load SQL
    sql = get_sql('statistics')

    # Connect to DuckDB
    con = duckdb.connect()
    con.execute(f"CREATE TABLE observations AS SELECT * FROM read_parquet('{observations_path}')")

    if verbose:
        n_rows = con.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        n_signals = con.execute("SELECT COUNT(DISTINCT signal_id) FROM observations").fetchone()[0]
        print(f"Loaded: {n_rows:,} observations, {n_signals} signals")

    # Execute SQL
    result = con.execute(sql).pl()

    # Write output
    result.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {result.shape}")

    con.close()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 13: Statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes summary statistics per signal:
  - mean, std, min, max
  - count, null_count
  - quartiles

Example:
  python -m prism.entry_points.stage_13_statistics \\
      observations.parquet -o statistics.parquet
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('-o', '--output', default='statistics.parquet',
                        help='Output path (default: statistics.parquet)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
