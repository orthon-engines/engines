"""
PRISM Mode Geometry Runner - Geometry by Behavioral Mode
=========================================================

Computes geometry metrics organized by discovered behavioral modes.

Key insight: Modes are DISCOVERED groupings from Laplace dynamics, unlike
cohorts which are PREDEFINED physical/logical groupings. This runner:

1. Discovers modes from Laplace fingerprints (gradient/divergence patterns)
2. Groups signals by their discovered mode
3. Computes geometry metrics for each mode (like we do for cohorts)

Output Schema (geometry/mode.parquet):
    - domain_id: Domain identifier
    - cohort_id: Original cohort (for reference)
    - mode_id: Discovered behavioral mode
    - window_end: Window date
    - n_signals: Count of signals in this mode
    - mode_affinity_mean: Mean affinity of signals to their mode
    - mode_entropy_mean: Mean entropy (uncertainty) of mode assignments
    - distance_mean: Mean pairwise distance within mode
    - pca_variance_pc1: First principal component variance
    - clustering_silhouette: Cluster quality within mode
    - ... (other geometry metrics)

Usage:
    python -m prism.entry_points.mode_geometry --domain cheme
    python -m prism.entry_points.mode_geometry --domain cheme --max-modes 10
"""

import argparse
import logging
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from orthon._internal.db.parquet_store import (
    get_path,
    get_data_root,
    ensure_directory,
    OBSERVATIONS,
    VECTOR,
    GEOMETRY,
    STATE,
    COHORTS,
)
from orthon._internal.db.polars_io import write_parquet_atomic
from orthon._internal.utils.domain import require_domain
from orthon._internal.engines.geometry.modes import (
    discover_modes,
    extract_laplace_fingerprint,
)

# Geometry engines
from orthon._internal.engines import (
    DistanceEngine,
    PCAEngine,
    ClusteringEngine,
    MSTEngine,
    LOFEngine,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_mode_data_matrix(
    observations: pl.DataFrame,
    signal_ids: List[str],
) -> pd.DataFrame:
    """
    Build data matrix for a mode's signals.

    Args:
        observations: Full observations DataFrame
        signal_ids: Signals in this mode

    Returns:
        DataFrame (rows=time, cols=signals)
    """
    if len(signal_ids) < 2:
        return pd.DataFrame()

    # Filter to mode signals
    filtered = observations.filter(
        pl.col('signal_id').is_in(signal_ids)
    ).select(['obs_date', 'signal_id', 'value'])

    if filtered.is_empty():
        return pd.DataFrame()

    # Deduplicate
    filtered = filtered.group_by(['signal_id', 'obs_date']).agg(
        pl.col('value').last()
    )

    # Pivot to matrix
    pivoted = filtered.pivot(
        on='signal_id',
        index='obs_date',
        values='value'
    ).sort('obs_date').drop_nulls()

    if pivoted.is_empty():
        return pd.DataFrame()

    # Convert to pandas
    dates = pivoted['obs_date'].to_list()
    cols = [c for c in pivoted.columns if c != 'obs_date']
    data = {col: pivoted[col].to_numpy() for col in cols}

    return pd.DataFrame(data, index=pd.DatetimeIndex(dates))


def compute_mode_geometry(matrix: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute geometry metrics for a mode.

    Args:
        matrix: Data matrix (rows=time, cols=signals)

    Returns:
        Dict of geometry metrics
    """
    if matrix.empty or matrix.shape[1] < 2:
        return {}

    results = {}
    run_id = f"mode_{datetime.now().isoformat()}"

    # 1. DISTANCE
    try:
        distance_engine = DistanceEngine()
        distance_result = distance_engine.run(matrix, run_id=run_id)

        if 'distance_matrix_euclidean' in distance_result:
            dist_matrix = distance_result['distance_matrix_euclidean']
            upper_tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
            results['distance_mean'] = float(np.mean(upper_tri))
            results['distance_std'] = float(np.std(upper_tri))
            results['distance_max'] = float(np.max(upper_tri))
    except Exception as e:
        logger.debug(f"Distance failed: {e}")

    # 2. PCA
    try:
        pca_engine = PCAEngine()
        n_comp = min(5, matrix.shape[0], matrix.shape[1] - 1)
        if n_comp >= 1:
            pca_result = pca_engine.run(matrix, run_id=run_id, n_components=n_comp)
            results['pca_variance_pc1'] = pca_result.get('variance_pc1', 0)
            results['pca_variance_pc2'] = pca_result.get('variance_pc2', 0)
            results['pca_cumulative_3'] = pca_result.get('cumulative_variance_3', 0)
            results['pca_effective_dim'] = pca_result.get('effective_dimensionality', 0)
    except Exception as e:
        logger.debug(f"PCA failed: {e}")

    # 3. CLUSTERING
    try:
        n_clusters = min(3, matrix.shape[0] - 1, matrix.shape[1])
        if n_clusters >= 2:
            clustering_engine = ClusteringEngine()
            clustering_result = clustering_engine.run(matrix, run_id=run_id, n_clusters=n_clusters)
            results['clustering_silhouette'] = clustering_result.get('silhouette_score', 0)
    except Exception as e:
        logger.debug(f"Clustering failed: {e}")

    # 4. MST
    try:
        mst_engine = MSTEngine()
        mst_result = mst_engine.run(matrix, run_id=run_id)
        results['mst_total_weight'] = mst_result.get('total_weight', 0)
        results['mst_avg_degree'] = mst_result.get('average_degree', 0)
    except Exception as e:
        logger.debug(f"MST failed: {e}")

    # 5. LOF
    try:
        lof_engine = LOFEngine()
        lof_result = lof_engine.run(matrix, run_id=run_id)
        results['lof_mean'] = lof_result.get('mean_lof', 0)
        results['lof_n_outliers'] = lof_result.get('n_outliers', 0)
    except Exception as e:
        logger.debug(f"LOF failed: {e}")

    results['n_signals'] = matrix.shape[1]
    results['n_observations'] = matrix.shape[0]

    return results


def run_mode_geometry(
    domain: str,
    max_modes: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Discover modes and compute geometry for each.

    Args:
        domain: Domain identifier
        max_modes: Maximum modes to discover
        verbose: Print progress

    Returns:
        Summary statistics
    """
    ensure_directory()

    # Load signal field (Laplace data)
    field_path = get_path(VECTOR)
    if not Path(field_path).exists():
        raise FileNotFoundError(f"Signal field not found: {field_path}")

    if verbose:
        print("=" * 80)
        print("PRISM MODE GEOMETRY - Geometry by Behavioral Mode")
        print("=" * 80)
        print(f"Domain: {domain}")
        print(f"Max modes: {max_modes}")
        print()

    # Load field data
    if verbose:
        print("Step 1: Loading signal field data...")

    field_df = pl.read_parquet(field_path)

    # Get unique signals
    all_signals = field_df['signal_id'].unique().to_list()
    if verbose:
        print(f"  Found {len(all_signals)} signals")

    # Discover modes
    if verbose:
        print()
        print("Step 2: Discovering behavioral modes...")

    modes_df = discover_modes(
        field_df,
        domain_id=domain,
        cohort_id='default',
        signals=all_signals,
        max_modes=max_modes,
    )

    if modes_df is None or len(modes_df) == 0:
        print("  No modes discovered (insufficient data)")
        return {'n_modes': 0, 'n_signals': 0}

    n_modes = modes_df['mode_id'].nunique()
    if verbose:
        print(f"  Discovered {n_modes} modes")
        print()
        print("  Mode distribution:")
        for mode_id in sorted(modes_df['mode_id'].unique()):
            count = len(modes_df[modes_df['mode_id'] == mode_id])
            mean_aff = modes_df[modes_df['mode_id'] == mode_id]['mode_affinity'].mean()
            print(f"    Mode {mode_id}: {count} signals (affinity={mean_aff:.3f})")

    # Load observations for geometry computation (lazy with streaming)
    if verbose:
        print()
        print("Step 3: Loading observations...")

    obs_path = get_path(OBSERVATIONS)
    # Get all signal_ids from modes for filter pushdown
    all_mode_signals = modes_df['signal_id'].unique().tolist()
    observations = (
        pl.scan_parquet(obs_path)
        .filter(pl.col('signal_id').is_in(all_mode_signals))
        .collect()
    )

    if verbose:
        print(f"  Loaded {len(observations):,} observations (filtered to mode signals)")

    # Compute geometry for each mode
    if verbose:
        print()
        print("Step 4: Computing geometry by mode...")

    records = []
    computed_at = datetime.now()

    for mode_id in sorted(modes_df['mode_id'].unique()):
        mode_data = modes_df[modes_df['mode_id'] == mode_id]
        signal_ids = mode_data['signal_id'].tolist()

        if verbose:
            print(f"  Mode {mode_id}: {len(signal_ids)} signals...")

        # Build data matrix for this mode
        matrix = get_mode_data_matrix(observations, signal_ids)

        if matrix.empty or matrix.shape[1] < 2:
            if verbose:
                print(f"    Skipped (insufficient data)")
            continue

        # Compute geometry
        metrics = compute_mode_geometry(matrix)

        # Get window range from observations
        mode_obs = observations.filter(
            pl.col('signal_id').is_in(signal_ids)
        )
        window_end = mode_obs['obs_date'].max()

        # Mode-level stats
        mode_affinity_mean = float(mode_data['mode_affinity'].mean())
        mode_entropy_mean = float(mode_data['mode_entropy'].mean())

        record = {
            'domain_id': domain,
            'cohort_id': 'default',
            'mode_id': int(mode_id),
            'window_end': window_end,
            'n_signals': len(signal_ids),
            'mode_affinity_mean': mode_affinity_mean,
            'mode_entropy_mean': mode_entropy_mean,
            **metrics,
            'computed_at': computed_at,
        }
        records.append(record)

        if verbose:
            pc1 = metrics.get('pca_variance_pc1', 0)
            dist = metrics.get('distance_mean', 0)
            print(f"    Computed: PCA_1={pc1:.3f}, dist={dist:.3f}")

    if not records:
        print("No geometry computed")
        return {'n_modes': n_modes, 'n_signals': len(all_signals)}

    # Save results
    result_df = pl.DataFrame(records, infer_schema_length=None)

    output_path = get_path(GEOMETRY)
    write_parquet_atomic(result_df, output_path)

    # Also save mode assignments
    modes_output_path = get_path(VECTOR)
    modes_pl = pl.from_pandas(modes_df)
    write_parquet_atomic(modes_pl, modes_output_path)

    if verbose:
        print()
        print("=" * 80)
        print("MODE GEOMETRY COMPLETE")
        print("=" * 80)
        print(f"Modes: {n_modes}")
        print(f"Signals: {len(all_signals)}")
        print(f"Geometry records: {len(records)}")
        print(f"Output: {output_path}")
        print(f"Modes: {modes_output_path}")

    return {
        'n_modes': n_modes,
        'n_signals': len(all_signals),
        'n_records': len(records),
        'output_path': str(output_path),
        'modes_path': str(modes_output_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description='PRISM Mode Geometry - Compute geometry by behavioral mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Discovers behavioral modes from Laplace fingerprints and computes
geometry metrics for each mode.

Modes are DISCOVERED groupings (GMM on gradient/divergence patterns),
unlike cohorts which are PREDEFINED groupings.

Output:
  geometry/mode.parquet       - Geometry metrics per mode
  vector/cohort_modes.parquet - Mode assignments for each signal
"""
    )

    parser.add_argument('--domain', type=str, default=None,
                        help='Domain to process (prompts if not specified)')
    parser.add_argument('--max-modes', type=int, default=10,
                        help='Maximum modes to discover (default: 10)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    # Domain selection
    import os
    domain = require_domain(args.domain, "Select domain for mode geometry")
    os.environ["PRISM_DOMAIN"] = domain

    run_mode_geometry(
        domain=domain,
        max_modes=args.max_modes,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
