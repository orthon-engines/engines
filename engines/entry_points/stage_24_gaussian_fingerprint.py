"""
Stage 24: Gaussian Fingerprint Entry Point
==========================================

Pure orchestration - calls SQL engines for computation.

Inputs:
    - signal_vector.parquet

Outputs:
    - gaussian_fingerprint.parquet (per-signal Gaussian summary)
    - gaussian_similarity.parquet (pairwise Bhattacharyya distance)

Builds probabilistic fingerprints from windowed engine outputs,
then computes pairwise similarity between signals within each cohort.
"""

import argparse
import polars as pl
import duckdb
from pathlib import Path
from typing import Optional

from engines.manifold.sql import get_sql


def run(
    signal_vector_path: str,
    fingerprint_output_path: str = "gaussian_fingerprint.parquet",
    similarity_output_path: str = "gaussian_similarity.parquet",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run Gaussian fingerprint and similarity computation.

    Args:
        signal_vector_path: Path to signal_vector.parquet
        fingerprint_output_path: Output path for gaussian_fingerprint.parquet
        similarity_output_path: Output path for gaussian_similarity.parquet
        verbose: Print progress

    Returns:
        Fingerprint DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 24: GAUSSIAN FINGERPRINT")
        print("Probabilistic signal fingerprints + pairwise similarity")
        print("=" * 70)

    # Load SQL
    fingerprint_sql = get_sql('gaussian_fingerprint')
    similarity_sql = get_sql('gaussian_similarity')

    # Connect to DuckDB
    con = duckdb.connect()
    con.execute(f"CREATE TABLE signal_vector AS SELECT * FROM read_parquet('{signal_vector_path}')")

    if verbose:
        n_rows = con.execute("SELECT COUNT(*) FROM signal_vector").fetchone()[0]
        n_signals = con.execute("SELECT COUNT(DISTINCT signal_id) FROM signal_vector").fetchone()[0]
        print(f"Loaded: {n_rows:,} windows, {n_signals} signals")

    # Step 1: Compute fingerprints
    if verbose:
        print("\nComputing Gaussian fingerprints...")

    fingerprint = con.execute(fingerprint_sql).pl()
    fingerprint.write_parquet(fingerprint_output_path)

    if verbose:
        print(f"  Fingerprints: {fingerprint.shape}")

    # Step 2: Load fingerprints and compute similarity
    if verbose:
        print("Computing pairwise similarity...")

    con.execute(f"CREATE TABLE gaussian_fingerprint AS SELECT * FROM read_parquet('{fingerprint_output_path}')")
    similarity = con.execute(similarity_sql).pl()
    similarity.write_parquet(similarity_output_path)

    if verbose:
        print(f"  Similarity pairs: {similarity.shape}")
        print()
        print("=" * 50)
        print(f"  {Path(fingerprint_output_path).absolute()}")
        print(f"  {Path(similarity_output_path).absolute()}")
        print("=" * 50)

    con.close()

    return fingerprint


def main():
    parser = argparse.ArgumentParser(
        description="Stage 24: Gaussian Fingerprint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Builds probabilistic Gaussian fingerprints from signal_vector.parquet,
then computes pairwise Bhattacharyya similarity within each cohort.

Example:
  python -m engines.entry_points.stage_24_gaussian_fingerprint \\
      signal_vector.parquet \\
      -o gaussian_fingerprint.parquet \\
      --similarity gaussian_similarity.parquet
"""
    )
    parser.add_argument('signal_vector', help='Path to signal_vector.parquet')
    parser.add_argument('-o', '--output', default='gaussian_fingerprint.parquet',
                        help='Output path for fingerprints (default: gaussian_fingerprint.parquet)')
    parser.add_argument('--similarity', default='gaussian_similarity.parquet',
                        help='Output path for similarity (default: gaussian_similarity.parquet)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.signal_vector,
        args.output,
        args.similarity,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
