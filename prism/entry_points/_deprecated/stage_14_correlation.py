"""
Stage 14: Correlation Entry Point
=================================

Pure orchestration - calls engines/sql/correlation.sql for computation.

Inputs:
    - observations.parquet

Output:
    - correlation.parquet

Computes pairwise correlation matrix between signals.
"""

import argparse
import polars as pl
import duckdb
from pathlib import Path
from typing import Optional

from prism.engines.sql import get_sql


def run(
    observations_path: str,
    output_path: str = "correlation.parquet",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run correlation computation using SQL engine.

    Args:
        observations_path: Path to observations.parquet
        output_path: Output path for correlation.parquet
        verbose: Print progress

    Returns:
        Correlation DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 14: CORRELATION")
        print("Pairwise correlation matrix")
        print("=" * 70)

    # Load SQL
    sql = get_sql('correlation')

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

        # Summary
        if 'correlation' in result.columns:
            mean_corr = result['correlation'].mean()
            max_corr = result['correlation'].max()
            print(f"Mean correlation: {mean_corr:.3f}")
            print(f"Max correlation: {max_corr:.3f}")

    con.close()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 14: Correlation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes pairwise correlation matrix between signals.

Example:
  python -m prism.entry_points.stage_14_correlation \\
      observations.parquet -o correlation.parquet
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('-o', '--output', default='correlation.parquet',
                        help='Output path (default: correlation.parquet)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
