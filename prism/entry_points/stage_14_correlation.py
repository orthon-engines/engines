"""
Stage 14: Correlation Entry Point
=================================

Pure orchestration - computes correlation matrix for signal_vector.

Inputs:
    - signal_vector.parquet

Output:
    - correlation.parquet

Computes:
    - Correlation matrix between all feature columns
    - Per-cohort correlation if cohort column exists
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
from typing import Optional


def run(
    signal_vector_path: str,
    output_path: str = "correlation.parquet",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute correlation matrix for signal_vector features.

    Args:
        signal_vector_path: Path to signal_vector.parquet
        output_path: Output path for correlation.parquet
        verbose: Print progress

    Returns:
        Correlation DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 14: CORRELATION")
        print("Computing feature correlation matrix")
        print("=" * 70)

    # Load signal vector
    sv = pl.read_parquet(signal_vector_path)

    if verbose:
        print(f"Loaded signal_vector: {sv.shape}")

    # Identify feature columns
    meta_cols = ['unit_id', 'I', 'signal_id', 'cohort']
    feature_cols = [
        c for c in sv.columns
        if c not in meta_cols
        and sv[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
    ]

    if verbose:
        print(f"Feature columns: {len(feature_cols)}")

    has_cohort = 'cohort' in sv.columns
    results = []

    if has_cohort:
        cohorts = sv['cohort'].unique().to_list()
        if verbose:
            print(f"Computing per-cohort correlations ({len(cohorts)} cohorts)...")

        for cohort in cohorts:
            cohort_data = sv.filter(pl.col('cohort') == cohort)
            matrix = cohort_data.select(feature_cols).to_numpy()

            # Remove rows with NaN
            valid_mask = np.isfinite(matrix).all(axis=1)
            matrix = matrix[valid_mask]

            if len(matrix) < 2:
                continue

            # Compute correlation matrix
            corr_matrix = np.corrcoef(matrix.T)

            # Store upper triangle
            for i in range(len(feature_cols)):
                for j in range(i + 1, len(feature_cols)):
                    if np.isfinite(corr_matrix[i, j]):
                        results.append({
                            'cohort': cohort,
                            'feature_a': feature_cols[i],
                            'feature_b': feature_cols[j],
                            'correlation': float(corr_matrix[i, j]),
                        })
    else:
        matrix = sv.select(feature_cols).to_numpy()

        # Remove rows with NaN
        valid_mask = np.isfinite(matrix).all(axis=1)
        matrix = matrix[valid_mask]

        if verbose:
            print(f"Computing global correlation ({len(matrix)} samples)...")

        if len(matrix) >= 2:
            corr_matrix = np.corrcoef(matrix.T)

            # Store upper triangle
            for i in range(len(feature_cols)):
                for j in range(i + 1, len(feature_cols)):
                    if np.isfinite(corr_matrix[i, j]):
                        results.append({
                            'feature_a': feature_cols[i],
                            'feature_b': feature_cols[j],
                            'correlation': float(corr_matrix[i, j]),
                        })

    # Build DataFrame
    result = pl.DataFrame(results) if results else pl.DataFrame()

    if len(result) > 0:
        result.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {result.shape}")

        if len(result) > 0:
            print(f"\nCorrelation stats:")
            print(f"  Mean |r|: {result['correlation'].abs().mean():.3f}")
            high_corr = (result['correlation'].abs() > 0.8).sum()
            print(f"  High correlation pairs (|r| > 0.8): {high_corr}")

        print()
        print("─" * 50)
        print(f"✓ {Path(output_path).absolute()}")
        print("─" * 50)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 14: Correlation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes correlation matrix between feature columns.

Outputs pairwise correlations for all feature combinations.
If cohort column exists, computes per-cohort correlations.

Example:
  python -m prism.entry_points.stage_14_correlation \\
      signal_vector.parquet -o correlation.parquet
"""
    )
    parser.add_argument('signal_vector', help='Path to signal_vector.parquet')
    parser.add_argument('-o', '--output', default='correlation.parquet',
                        help='Output path (default: correlation.parquet)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.signal_vector,
        args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
