"""
Stage 12: Z-Score Entry Point
=============================

Pure orchestration - calls engines/sql/zscore.sql for computation.

Inputs:
    - observations.parquet

Output:
    - zscore.parquet

Computes z-score normalization and anomaly detection:
    - z_score: (value - mean) / std per signal
    - is_anomaly: |z_score| > 3

See docs/NORMALIZATION.md for guidance on z-score vs MAD-based detection.
"""

import argparse
import polars as pl
import duckdb
from pathlib import Path
from typing import Optional

from prism.engines.sql import get_sql


def run(
    observations_path: str,
    output_path: str = "zscore.parquet",
    threshold: float = 3.0,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run z-score computation using SQL engine.

    Args:
        observations_path: Path to observations.parquet
        output_path: Output path for zscore.parquet
        threshold: Anomaly threshold (default: 3.0)
        verbose: Print progress

    Returns:
        Z-score DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 12: Z-SCORE")
        print("Normalization and anomaly detection")
        print("=" * 70)

    # Load SQL
    sql = get_sql('zscore')

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

        # Anomaly summary
        if 'is_anomaly' in result.columns:
            n_anomalies = result.filter(pl.col('is_anomaly')).height
            pct = n_anomalies / len(result) * 100 if len(result) > 0 else 0
            print(f"Anomalies: {n_anomalies} ({pct:.2f}%)")

    con.close()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 12: Z-Score",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes z-score normalization and anomaly detection:
  - z_score: (value - mean) / std per signal
  - is_anomaly: |z_score| > threshold

Note: For robust anomaly detection, consider using MAD-based
      approach instead. See docs/NORMALIZATION.md

Example:
  python -m prism.entry_points.stage_12_zscore \\
      observations.parquet -o zscore.parquet
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('-o', '--output', default='zscore.parquet',
                        help='Output path (default: zscore.parquet)')
    parser.add_argument('--threshold', type=float, default=3.0,
                        help='Anomaly threshold (default: 3.0)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        args.output,
        threshold=args.threshold,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
