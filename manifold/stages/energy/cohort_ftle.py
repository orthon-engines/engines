"""
Stage 30: Cohort FTLE Entry Point
==================================

FTLE on cohort trajectories in feature space.
Each cohort's shape_effective_dim trajectory over I is a 1D time series.
Uses the same FTLE engine as stage_08, with reduced min_samples (cohort
trajectories are short — ~65 windows vs ~500+ signal samples).

Inputs:
    - cohort_vector.parquet

Output:
    - cohort_ftle.parquet
"""

import numpy as np
import polars as pl
from pathlib import Path

from manifold.core.dynamics.ftle import compute as compute_ftle
from manifold.core.dynamics.formal_definitions import classify_stability
from manifold.io.writer import write_output


def run(
    cohort_vector_path: str,
    data_path: str = ".",
    min_samples: int = 30,
    method: str = 'rosenstein',
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute FTLE for cohort trajectories.

    Uses shape_effective_dim as the primary scalar trajectory per cohort.
    Reduced min_samples (30) because cohort trajectories are short.

    Args:
        cohort_vector_path: Path to cohort_vector.parquet
        data_path: Root data directory for output
        min_samples: Minimum samples required (default: 30, reduced from 200)
        method: 'rosenstein' or 'kantz'
        verbose: Print progress

    Returns:
        Cohort FTLE DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 30: COHORT FTLE")
        print("Finite-Time Lyapunov Exponents for cohort trajectories")
        print("=" * 70)

    cv = pl.read_parquet(cohort_vector_path)

    if verbose:
        print(f"Loaded cohort_vector: {cv.shape}")

    if len(cv) == 0:
        if verbose:
            print("  Empty cohort_vector — skipping")
        write_output(pl.DataFrame(), data_path, 'cohort_ftle', verbose=verbose)
        return pl.DataFrame()

    # Find the primary scalar column: shape_effective_dim
    scalar_col = None
    for candidate in ['shape_effective_dim', 'complexity_effective_dim', 'spectral_effective_dim']:
        if candidate in cv.columns:
            scalar_col = candidate
            break

    if scalar_col is None:
        # Fall back to first effective_dim column
        eff_cols = [c for c in cv.columns if c.endswith('_effective_dim')]
        if eff_cols:
            scalar_col = eff_cols[0]
        else:
            if verbose:
                print("  No effective_dim column found — skipping")
            write_output(pl.DataFrame(), data_path, 'cohort_ftle', verbose=verbose)
            return pl.DataFrame()

    if verbose:
        print(f"Primary scalar: {scalar_col}")
        print(f"min_samples: {min_samples}")

    cohorts = sorted(cv['cohort'].unique().to_list())
    results = []

    for cohort in cohorts:
        cohort_data = cv.filter(pl.col('cohort') == cohort).sort('I')
        values = cohort_data[scalar_col].to_numpy().astype(float)

        # Remove NaN
        values_clean = values[np.isfinite(values)]

        if len(values_clean) < min_samples:
            results.append({
                'cohort': cohort,
                'direction': 'forward',
                'ftle': None,
                'ftle_std': None,
                'embedding_dim': None,
                'embedding_tau': None,
                'confidence': 0.0,
                'n_samples': len(values_clean),
                'stability': 'insufficient_data',
            })
            results.append({
                'cohort': cohort,
                'direction': 'backward',
                'ftle': None,
                'ftle_std': None,
                'embedding_dim': None,
                'embedding_tau': None,
                'confidence': 0.0,
                'n_samples': len(values_clean),
                'stability': 'insufficient_data',
            })
            continue

        # Forward FTLE
        fwd = compute_ftle(values_clean, min_samples=min_samples, method=method)
        fwd_val = fwd['ftle']
        if fwd_val is not None:
            fwd_stability = classify_stability(fwd_val).value
        else:
            fwd_stability = 'unknown'

        results.append({
            'cohort': cohort,
            'direction': 'forward',
            'ftle': fwd_val,
            'ftle_std': fwd['ftle_std'],
            'embedding_dim': fwd['embedding_dim'],
            'embedding_tau': fwd['embedding_tau'],
            'confidence': fwd['confidence'],
            'n_samples': len(values_clean),
            'stability': fwd_stability,
        })

        # Backward FTLE
        bwd = compute_ftle(values_clean[::-1].copy(), min_samples=min_samples, method=method)
        bwd_val = bwd['ftle']
        if bwd_val is not None:
            bwd_stability = classify_stability(bwd_val).value
        else:
            bwd_stability = 'unknown'

        results.append({
            'cohort': cohort,
            'direction': 'backward',
            'ftle': bwd_val,
            'ftle_std': bwd['ftle_std'],
            'embedding_dim': bwd['embedding_dim'],
            'embedding_tau': bwd['embedding_tau'],
            'confidence': bwd['confidence'],
            'n_samples': len(values_clean),
            'stability': bwd_stability,
        })

    result = pl.DataFrame(results, infer_schema_length=len(results)) if results else pl.DataFrame()

    write_output(result, data_path, 'cohort_ftle', verbose=verbose)

    if verbose:
        print(f"Shape: {result.shape}")
        if len(result) > 0 and 'ftle' in result.columns:
            for direction in ['forward', 'backward']:
                subset = result.filter(
                    (pl.col('direction') == direction) & pl.col('ftle').is_not_null()
                )
                if len(subset) > 0:
                    print(f"  {direction} FTLE: mean={subset['ftle'].mean():.4f}, "
                          f"n={len(subset)}")

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 30: Cohort FTLE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes forward and backward FTLE for cohort trajectories.
Uses shape_effective_dim as the primary scalar.

Example:
  python -m engines.entry_points.stage_30_cohort_ftle \\
      cohort_vector.parquet -o cohort_ftle.parquet
"""
    )
    parser.add_argument('cohort_vector', help='Path to cohort_vector.parquet')
    parser.add_argument('-d', '--data-path', default='.',
                        help='Root data directory (default: .)')
    parser.add_argument('--min-samples', type=int, default=30,
                        help='Minimum samples per cohort (default: 30)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.cohort_vector,
        args.data_path,
        min_samples=args.min_samples,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
