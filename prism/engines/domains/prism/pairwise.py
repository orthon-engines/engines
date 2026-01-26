"""
PRISM Laplace Pairwise Geometry (Vectorized)
============================================

Polars-native pairwise computation on Laplace field vectors.
No Python loops. Self-join + parallel computation.
"""

import polars as pl
from pathlib import Path
from datetime import datetime
import logging

from prism.db.parquet_store import ensure_directory, get_path, get_data_root, VECTOR, GEOMETRY
from prism.db.polars_io import write_parquet_atomic, get_file_size_mb
from prism.utils.stride import load_stride_config, get_default_tiers, get_drilldown_tiers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_window_config() -> dict:
    """
    Load window configuration from stride.yaml. Fails if not configured.

    Returns dict mapping tier name to window config.
    """
    try:
        config = load_stride_config()
        result = {}
        for tier_name in ['anchor', 'bridge', 'scout', 'micro']:
            try:
                window = config.get_window(tier_name)
                result[tier_name] = {
                    'window_days': window.window_days,
                    'stride_days': window.stride_days,
                    'weight': window.weight,
                }
            except Exception:
                pass
        if result:
            return result
    except Exception as e:
        raise RuntimeError(f"Failed to load stride config: {e}")

    # No fallback - must be configured
    raise RuntimeError(
        "No window tiers configured in config/stride.yaml. "
        "Configure domain-specific window sizes before running."
    )


# Load window config at module level
WINDOW_CONFIG = get_window_config()

DEFAULT_TIERS = get_default_tiers()
DRILLDOWN_TIERS = get_drilldown_tiers()


def extract_cohort_from_signal(signal_id: str) -> str:
    """Extract cohort from signal_id (e.g., CMAPSS_BPR_FD001_U001 -> FD001_U001)."""
    parts = signal_id.split('_')
    if len(parts) >= 4:
        return '_'.join(parts[-2:])
    return signal_id


def run_laplace_pairwise_vectorized(
    cohort_filter: str = None,
    chunk_size: int = 50_000,
) -> dict:
    """
    Vectorized pairwise geometry on Laplace field data.

    Uses Polars self-join instead of Python loops.
    """

    ensure_directory()

    # Load Laplace field data with lazy scan and filter pushdown
    path = get_path(VECTOR)
    if not Path(path).exists():
        raise FileNotFoundError(f"Run laplace.py first: {path}")

    file_size_mb = get_file_size_mb(path)
    logger.info(f"Loading {path} ({file_size_mb:.0f} MB)")

    # Build lazy query with filter pushdown
    lazy_query = pl.scan_parquet(path)

    # If cohort_filter provided, filter by signal_id pattern BEFORE collecting
    # (cohort is derived from signal_id, so we can filter on signal_id pattern)
    if cohort_filter:
        # cohort_filter like "FD001_U001" matches signal_id ending with "_FD001_U001"
        lazy_query = lazy_query.filter(
            pl.col('signal_id').str.ends_with(f'_{cohort_filter}') |
            pl.col('signal_id').str.ends_with(cohort_filter)
        )
        logger.info(f"Applied cohort filter: {cohort_filter}")

    # Collect with filter pushdown applied
    laplace = lazy_query.collect()
    logger.info(f"Loaded {len(laplace):,} rows")

    # Add cohort column derived from signal_id
    laplace = laplace.with_columns(
        pl.col('signal_id').map_elements(extract_cohort_from_signal, return_dtype=pl.Utf8).alias('cohort')
    )

    # Final filter to ensure exact match (in case pattern was too broad)
    if cohort_filter:
        laplace = laplace.filter(pl.col('cohort') == cohort_filter)
        logger.info(f"Filtered to cohort {cohort_filter}: {len(laplace):,} rows")

    # Get unique cohorts
    cohorts = laplace['cohort'].unique().to_list()
    logger.info(f"Processing {len(cohorts)} cohorts")

    all_results = []

    for cohort in cohorts:
        logger.info(f"[{cohort}] Starting...")
        t0 = datetime.now()

        # Filter to this cohort
        cohort_data = laplace.filter(pl.col('cohort') == cohort)

        # Aggregate to signal-level (mean across windows and engines)
        signal_agg = (
            cohort_data
            .group_by('signal_id')
            .agg([
                pl.col('gradient_mean').mean().alias('gradient'),
                pl.col('laplacian_mean').mean().alias('laplacian'),
                pl.col('divergence').mean().alias('divergence'),
                pl.col('field_potential').mean().alias('field_potential'),
                pl.col('cohort').first().alias('cohort'),
            ])
        )

        n_signals = len(signal_agg)
        n_pairs = n_signals * (n_signals - 1) // 2
        logger.info(f"[{cohort}] {n_signals} signals → {n_pairs:,} pairs")

        if n_signals < 2:
            continue

        # Self-join for all pairs
        pairwise = (
            signal_agg
            .join(signal_agg, how='cross', suffix='_b')
            .filter(pl.col('signal_id') < pl.col('signal_id_b'))
            .with_columns([
                # Euclidean distance in field space
                (
                    (pl.col('gradient') - pl.col('gradient_b')).pow(2) +
                    (pl.col('laplacian') - pl.col('laplacian_b')).pow(2) +
                    (pl.col('divergence') - pl.col('divergence_b')).pow(2) +
                    (pl.col('field_potential') - pl.col('field_potential_b')).pow(2)
                ).sqrt().alias('field_distance'),

                # Divergence distance (difference in source/sink behavior)
                (pl.col('divergence') - pl.col('divergence_b')).abs().alias('divergence_distance'),

                # Field sign agreement (same role: source/sink)
                (pl.col('divergence').sign() == pl.col('divergence_b').sign())
                    .cast(pl.Float64).alias('field_sign_agreement'),

                # Divergence product (both sources = +, both sinks = +, mixed = -)
                (pl.col('divergence') * pl.col('divergence_b')).alias('divergence_product'),

                # Potential difference
                (pl.col('field_potential') - pl.col('field_potential_b')).abs().alias('potential_diff'),

                # Field magnitude coupling (captures anti-phase coupling)
                # Product of absolute values: high when both are active regardless of sign
                (pl.col('laplacian').abs() * pl.col('laplacian_b').abs()).alias('laplacian_magnitude_product'),
                (pl.col('divergence').abs() * pl.col('divergence_b').abs()).alias('divergence_magnitude_product'),

                # Magnitude similarity: 1 - normalized difference (1 = same magnitude, 0 = very different)
                (1.0 - (pl.col('laplacian').abs() - pl.col('laplacian_b').abs()).abs() /
                    (pl.col('laplacian').abs() + pl.col('laplacian_b').abs() + 1e-10)
                ).alias('laplacian_magnitude_similarity'),
            ])
            .select([
                'signal_id',
                pl.col('signal_id_b').alias('signal_b'),
                'cohort',
                'field_distance',
                'divergence_distance',
                'field_sign_agreement',
                'divergence_product',
                'potential_diff',
                'laplacian_magnitude_product',
                'divergence_magnitude_product',
                'laplacian_magnitude_similarity',
                pl.col('divergence').alias('divergence_a'),
                pl.col('divergence_b'),
            ])
        )

        elapsed = (datetime.now() - t0).total_seconds()
        logger.info(f"[{cohort}] {len(pairwise):,} pairs in {elapsed:.1f}s")

        all_results.append(pairwise)

    # Combine all cohorts
    if all_results:
        result = pl.concat(all_results)

        output_path = get_path(GEOMETRY)
        write_parquet_atomic(result, output_path)
        logger.info(f"Wrote {len(result):,} rows to {output_path}")

        return {
            'cohorts': len(cohorts),
            'pairs': len(result),
        }

    return {'cohorts': 0, 'pairs': 0}


def run_laplace_pairwise_windowed(
    cohort_filter: str = None,
) -> dict:
    """
    Windowed pairwise - fully vectorized, no Python loops.

    Single Polars pipeline: aggregate → self-join → filter → compute.
    """

    ensure_directory()

    path = get_path(VECTOR)

    file_size_mb = get_file_size_mb(path)
    logger.info(f"Loading {path} ({file_size_mb:.0f} MB)")

    # Build lazy query with filter pushdown
    lazy_query = pl.scan_parquet(path)

    # If cohort_filter provided, filter by signal_id pattern BEFORE collecting
    if cohort_filter:
        lazy_query = lazy_query.filter(
            pl.col('signal_id').str.ends_with(f'_{cohort_filter}') |
            pl.col('signal_id').str.ends_with(cohort_filter)
        )
        logger.info(f"Applied cohort filter: {cohort_filter}")

    # Collect with filter pushdown applied
    laplace = lazy_query.collect()
    logger.info(f"Loaded {len(laplace):,} rows")

    # Add cohort column derived from signal_id
    laplace = laplace.with_columns(
        pl.col('signal_id').map_elements(extract_cohort_from_signal, return_dtype=pl.Utf8).alias('cohort')
    )

    # Final filter to ensure exact match
    if cohort_filter:
        laplace = laplace.filter(pl.col('cohort') == cohort_filter)
        logger.info(f"Filtered to {cohort_filter}: {len(laplace):,} rows")

    # Step 1: Aggregate to (cohort, window_end, signal_id) level
    logger.info("Aggregating to signal-window level...")
    agg = (
        laplace
        .group_by(['cohort', 'window_end', 'signal_id'])
        .agg([
            pl.col('gradient_mean').mean().alias('gradient'),
            pl.col('laplacian_mean').mean().alias('laplacian'),
            pl.col('divergence').mean(),
            pl.col('field_potential').mean(),
        ])
    )
    logger.info(f"Aggregated to {len(agg):,} signal-windows")

    # Step 2: Self-join on (cohort, window_end) to get all pairs
    logger.info("Computing pairwise (cross-join)...")
    pairwise = (
        agg
        .join(
            agg,
            on=['cohort', 'window_end'],
            how='inner',
            suffix='_b'
        )
        .filter(pl.col('signal_id') < pl.col('signal_id_b'))
        .with_columns([
            # Field distance in 4D Laplace space
            (
                (pl.col('gradient') - pl.col('gradient_b')).pow(2) +
                (pl.col('laplacian') - pl.col('laplacian_b')).pow(2) +
                (pl.col('divergence') - pl.col('divergence_b')).pow(2) +
                (pl.col('field_potential') - pl.col('field_potential_b')).pow(2)
            ).sqrt().alias('field_distance'),

            # Divergence distance
            (pl.col('divergence') - pl.col('divergence_b')).abs().alias('divergence_distance'),

            # Field sign agreement
            (pl.col('divergence').sign() == pl.col('divergence_b').sign())
                .cast(pl.Float64).alias('field_sign_agreement'),

            # Divergence product
            (pl.col('divergence') * pl.col('divergence_b')).alias('divergence_product'),

            # Field magnitude coupling (captures anti-phase coupling)
            (pl.col('laplacian').abs() * pl.col('laplacian_b').abs()).alias('laplacian_magnitude_product'),
            (pl.col('divergence').abs() * pl.col('divergence_b').abs()).alias('divergence_magnitude_product'),

            # Magnitude similarity
            (1.0 - (pl.col('laplacian').abs() - pl.col('laplacian_b').abs()).abs() /
                (pl.col('laplacian').abs() + pl.col('laplacian_b').abs() + 1e-10)
            ).alias('laplacian_magnitude_similarity'),
        ])
        .select([
            'signal_id',
            pl.col('signal_id_b').alias('signal_b'),
            'cohort',
            'window_end',
            'field_distance',
            'divergence_distance',
            'field_sign_agreement',
            'divergence_product',
            'laplacian_magnitude_product',
            'divergence_magnitude_product',
            'laplacian_magnitude_similarity',
            pl.col('divergence').alias('div_a'),
            pl.col('divergence_b').alias('div_b'),
        ])
    )

    logger.info(f"Computed {len(pairwise):,} pair-windows")

    # Write output
    output_path = get_path(GEOMETRY)
    write_parquet_atomic(pairwise, output_path)
    logger.info(f"Wrote {len(pairwise):,} rows to {output_path}")

    return {'pairs': len(pairwise)}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', type=str, help='Filter to cohort')
    parser.add_argument('--windowed', action='store_true', help='Keep time dimension')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("PRISM LAPLACE PAIRWISE (VECTORIZED)")
    logger.info("=" * 70)

    if args.windowed:
        stats = run_laplace_pairwise_windowed(args.cohort)
    else:
        stats = run_laplace_pairwise_vectorized(args.cohort)

    logger.info("=" * 70)
    logger.info(f"COMPLETE: {stats}")
    logger.info("=" * 70)
