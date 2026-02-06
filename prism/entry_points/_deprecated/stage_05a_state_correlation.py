"""
Stage 05a: State Correlation Entry Point
========================================

Pure orchestration - computes correlation between state vectors over time.

Inputs:
    - state_vector.parquet

Output:
    - state_correlation.parquet

Computes temporal correlation of state vector components.
"""

import argparse
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any
from itertools import combinations


def run(
    state_vector_path: str,
    output_path: str = "state_correlation.parquet",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute correlation between state vector components over time.

    Args:
        state_vector_path: Path to state_vector.parquet
        output_path: Output path for state_correlation.parquet
        verbose: Print progress

    Returns:
        State correlation DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 05a: STATE CORRELATION")
        print("Temporal correlation of state vector components")
        print("=" * 70)

    # Load state vector
    sv = pl.read_parquet(state_vector_path)

    if verbose:
        print(f"Loaded state_vector: {sv.shape}")

    # Identify centroid columns (those that are numeric and not I/cohort/etc)
    exclude_cols = {'I', 'cohort', 'unit_id', 'window_idx', 'n_signals'}
    centroid_cols = [c for c in sv.columns if c not in exclude_cols
                     and sv[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

    if verbose:
        print(f"Centroid columns: {len(centroid_cols)}")

    results = []

    # Compute pairwise correlations
    pairs = list(combinations(centroid_cols, 2))
    if verbose:
        print(f"Computing {len(pairs)} pairwise correlations...")

    for col_a, col_b in pairs:
        x = sv[col_a].to_numpy()
        y = sv[col_b].to_numpy()

        # Remove NaN pairs
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 3:
            continue

        # Pearson correlation
        corr = np.corrcoef(x_clean, y_clean)[0, 1]

        # Spearman rank correlation
        from scipy.stats import spearmanr
        spearman_corr, spearman_p = spearmanr(x_clean, y_clean)

        results.append({
            'component_a': col_a,
            'component_b': col_b,
            'pearson_corr': float(corr) if not np.isnan(corr) else np.nan,
            'spearman_corr': float(spearman_corr),
            'spearman_pvalue': float(spearman_p),
            'n_samples': len(x_clean),
        })

    # Build DataFrame
    df = pl.DataFrame(results) if results else pl.DataFrame()

    # Write output
    if len(df) > 0:
        df.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {df.shape}")
        if 'pearson_corr' in df.columns:
            mean_corr = df['pearson_corr'].mean()
            print(f"Mean Pearson correlation: {mean_corr:.3f}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Stage 05a: State Correlation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes pairwise correlation between state vector components.

Example:
  python -m prism.entry_points.stage_05a_state_correlation \\
      state_vector.parquet -o state_correlation.parquet
"""
    )
    parser.add_argument('state_vector', help='Path to state_vector.parquet')
    parser.add_argument('-o', '--output', default='state_correlation.parquet',
                        help='Output path (default: state_correlation.parquet)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.state_vector,
        args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
