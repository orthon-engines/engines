"""
PRISM Geometry Runner
=====================

Computes structural geometry from vector signals using Laplace fields.

GEOMETRY ENGINES (9 canonical):
    - distance:            Euclidean/Mahalanobis/cosine distance matrices
    - pca:                 Principal components (dimensionality)
    - clustering:          K-means, hierarchical grouping
    - mutual_information:  Nonlinear dependence
    - copula:              Tail dependence
    - mst:                 Minimum spanning tree (network structure)
    - lof:                 Local outlier factor
    - convex_hull:         Phase space volume
    - barycenter:          Conviction-weighted center of mass

Output: data/geometry.parquet

Usage:
    python -m prism.entry_points.geometry              # Production run
    python -m prism.entry_points.geometry --adaptive   # Auto-detect window
    python -m prism.entry_points.geometry --force      # Force recompute
    python -m prism.entry_points.geometry --testing    # Enable test mode
"""

import argparse
import gc
import logging
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from multiprocessing import Pool, cpu_count
import warnings

from prism.db.parquet_store import (
    ensure_directory,
    get_data_root,
    get_path,
    OBSERVATIONS,
    VECTOR,
    GEOMETRY,
    STATE,
    COHORTS,
)
# Backwards compatibility
SIGNALS = VECTOR
from prism.db.polars_io import (
    read_parquet,
    read_parquet_smart,
    get_file_size_mb,
    upsert_parquet,
    write_parquet_atomic,
)
from prism.utils.memory import force_gc, get_memory_usage_mb
from prism.db.scratch import TempParquet, merge_to_table
from prism.engines.utils.parallel import (
    WorkerAssignment,
    divide_by_count,
    generate_temp_path,
    run_workers,
)

# Canonical engines from registry (9 geometry engines)
from prism.engines import (
    DistanceEngine,
    PCAEngine,
    ClusteringEngine,
    MutualInformationEngine,
    CopulaEngine,
    MSTEngine,
    LOFEngine,
    ConvexHullEngine,
    BarycenterEngine,
    compute_barycenter,
)

# Window/stride configuration
from prism.utils.stride import (
    load_stride_config,
    get_window_dates,
    get_barycenter_weights,
    get_default_tiers,
    get_drilldown_tiers,
    WINDOWS,
)

# Fast config access (Python dicts)
from prism.config.windows import get_window_weight

# Adaptive domain clock integration
from prism.config.loader import load_delta_thresholds
import json

# Bisection analysis
from prism.utils import bisection


def load_domain_info() -> Optional[Dict[str, Any]]:
    """
    Load domain_info from data/domain_info.json if available.

    This is saved by signal_vector when running in --adaptive mode.
    Contains auto-detected window parameters based on domain frequency.
    """
    domain_info_path = get_data_root() / "domain_info.json"
    if domain_info_path.exists():
        try:
            with open(domain_info_path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def get_adaptive_window_config() -> Optional[Tuple[int, int]]:
    """
    Get adaptive window/stride from domain_info if available.

    Returns (window_samples, stride_samples) or None if not available.
    """
    domain_info = load_domain_info()
    if domain_info:
        window = domain_info.get('window_samples')
        if window:
            stride = domain_info.get('stride_samples') or max(1, window // 3)
            return (window, stride)
    return None

# Inline modules for mode discovery and wavelet analysis
from prism.engines.geometry.modes import (
    discover_modes,
    compute_affinity_weighted_features,
)
from prism.engines.spectral.wavelet import (
    run_wavelet_microscope,
    extract_wavelet_features,
)

# V2 Architecture: Geometry from Laplace fields
from prism.engines.geometry.snapshot import (
    compute_geometry_at_t,
    compute_geometry_trajectory,
    snapshot_to_vector,
    get_unified_timestamps,
)
from prism.engines.geometry.coupling import compute_coupling_matrix, compute_affinity_matrix
from prism.engines.geometry.divergence import compute_divergence, compute_divergence_trajectory
from prism.engines.geometry.modes import discover_modes as discover_modes_v2
from prism.core.signals.types import LaplaceField, GeometrySnapshot
from prism.engines.laplace.transform import compute_laplace_field as compute_laplace_field_v2

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# DATA FETCHING (orchestration)
# =============================================================================

def get_curated_signals() -> Optional[set]:
    """
    Get set of curated (non-redundant) signals from filter_deep output.

    Returns None if no filter output exists (run all signals).
    Returns set of signal IDs if filter output exists.
    """
    curated_path = Path('data/filter/deep_curated.parquet')
    if not curated_path.exists():
        return None

    try:
        df = pl.read_parquet(curated_path)
        return set(df['signal_id'].to_list())
    except Exception as e:
        logger.warning(f"Could not read curated signals: {e}")
        return None


def get_redundant_signals() -> set:
    """Get set of redundant signals to exclude."""
    redundant_path = Path('data/filter/deep_redundant.parquet')
    if not redundant_path.exists():
        return set()

    try:
        df = pl.read_parquet(redundant_path)
        return set(df['signal_id'].to_list())
    except Exception as e:
        logger.warning(f"Could not read redundant signals: {e}")
        return set()


def get_date_range() -> Tuple[date, date]:
    """Get available date range from observations (lazy query)."""
    # Use lazy scan - only reads metadata, not full file
    lf = pl.scan_parquet(get_path(OBSERVATIONS))
    # New schema uses 'timestamp' instead of 'obs_date'
    ts_col = 'timestamp' if 'timestamp' in lf.collect_schema().names() else 'obs_date'
    result = lf.select([
        pl.col(ts_col).min().alias('min_date'),
        pl.col(ts_col).max().alias('max_date'),
    ]).collect()
    min_date = result['min_date'][0]
    max_date = result['max_date'][0]
    # Convert to Python date if needed
    if hasattr(min_date, 'date'):
        min_date = min_date.date()
    if hasattr(max_date, 'date'):
        max_date = max_date.date()
    return min_date, max_date


# Cache redundant signals at module load
_REDUNDANT_INDICATORS: Optional[set] = None


def get_cohort_signals(cohort: str) -> List[str]:
    """
    Fetch all signals in a cohort, excluding redundant ones.

    Uses filter_deep output if available to exclude redundant signals.
    """
    global _REDUNDANT_INDICATORS

    # Load redundant signals once
    if _REDUNDANT_INDICATORS is None:
        _REDUNDANT_INDICATORS = get_redundant_signals()
        if _REDUNDANT_INDICATORS:
            logger.info(f"Excluding {len(_REDUNDANT_INDICATORS)} redundant signals from filter_deep")

    cohort_members = pl.read_parquet(get_path(COHORTS))
    # Handle both 'cohort_id' and 'cohort' column names
    cohort_col = 'cohort_id' if 'cohort_id' in cohort_members.columns else 'cohort'
    signals = cohort_members.filter(
        pl.col(cohort_col) == cohort
    ).select('signal_id').sort('signal_id').to_series().to_list()

    # Filter out redundant signals
    if _REDUNDANT_INDICATORS:
        signals = [ind for ind in signals if ind not in _REDUNDANT_INDICATORS]

    return signals


def get_all_cohorts() -> List[str]:
    """Get list of all cohorts."""
    cohort_members = pl.read_parquet(get_path(COHORTS))
    # Handle both 'cohort_id' and 'cohort' column names
    cohort_col = 'cohort_id' if 'cohort_id' in cohort_members.columns else 'cohort'
    return cohort_members.select(cohort_col).unique().sort(cohort_col).to_series().to_list()


def get_cohort_data_matrix(
    cohort: str,
    window_end: date,
    window_days: int
) -> pd.DataFrame:
    """
    Fetch cohort data as a matrix (rows=time, cols=signals).

    Returns DataFrame with DateTimeIndex and signal columns.
    """
    window_start = window_end - timedelta(days=window_days)

    # Get cohort signals
    signals = get_cohort_signals(cohort)

    # Lazy scan with filter pushdown (memory efficient)
    filtered = (
        pl.scan_parquet(get_path(OBSERVATIONS))
        .filter(
            (pl.col('signal_id').is_in(signals)) &
            (pl.col('obs_date') >= window_start) &
            (pl.col('obs_date') <= window_end)
        )
        .select(['obs_date', 'signal_id', 'value'])
        .collect()
    )

    if filtered.is_empty():
        return pd.DataFrame()

    # Deduplicate: take last value for each (signal_id, obs_date) pair
    filtered = filtered.group_by(['signal_id', 'obs_date']).agg(
        pl.col('value').last()
    )

    # Pivot to matrix format using Polars
    pivoted = filtered.pivot(
        on='signal_id',
        index='obs_date',
        values='value'
    ).sort('obs_date')

    # Drop rows with any null (engines need complete data)
    pivoted = pivoted.drop_nulls()

    if pivoted.is_empty():
        return pd.DataFrame()

    # Convert to pandas DataFrame with date index
    # Use numpy conversion to avoid pyarrow dependency
    dates = pivoted['obs_date'].to_list()
    cols = [c for c in pivoted.columns if c != 'obs_date']
    data = {col: pivoted[col].to_numpy() for col in cols}

    matrix = pd.DataFrame(data, index=pd.DatetimeIndex(dates))
    return matrix


def get_pairwise_data(
    ind_a: str,
    ind_b: str,
    window_end: date,
    window_days: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fetch aligned signal topology for two signals.

    Returns (series_a, series_b) as numpy arrays.
    """
    window_start = window_end - timedelta(days=window_days)

    # Lazy scan with filter pushdown (memory efficient)
    obs_path = get_path(OBSERVATIONS)

    # Get data for both signals
    data_a = (
        pl.scan_parquet(obs_path)
        .filter(
            (pl.col('signal_id') == ind_a) &
            (pl.col('obs_date') >= window_start) &
            (pl.col('obs_date') <= window_end)
        )
        .select(['obs_date', 'value'])
        .collect()
        .rename({'value': 'value_a'})
    )

    data_b = (
        pl.scan_parquet(obs_path)
        .filter(
            (pl.col('signal_id') == ind_b) &
            (pl.col('obs_date') >= window_start) &
            (pl.col('obs_date') <= window_end)
        )
        .select(['obs_date', 'value'])
        .collect()
        .rename({'value': 'value_b'})
    )

    # Join on date
    aligned = data_a.join(data_b, on='obs_date', how='inner')
    aligned = aligned.drop_nulls().sort('obs_date')

    if aligned.is_empty():
        return np.array([]), np.array([])

    return aligned['value_a'].to_numpy(), aligned['value_b'].to_numpy()


def get_signal_window_vectors(
    cohort: str,
    window_end: date,
    window_sizes: List[int]
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Build window_vectors dict for barycenter computation.

    For each signal, fetch signal topology for each window size and create
    feature vectors (simple stats as representation).

    Returns:
        Dict mapping signal_id -> {window_days: feature_vector}
    """
    signals = get_cohort_signals(cohort)
    window_vectors = {}

    # Lazy scan with cohort filter pushdown (memory efficient)
    obs_cohort = (
        pl.scan_parquet(get_path(OBSERVATIONS))
        .filter(pl.col('signal_id').is_in(signals))
        .collect()
    )

    for signal_id in signals:
        vectors = {}

        for window_days in window_sizes:
            window_start = window_end - timedelta(days=window_days)

            df = obs_cohort.filter(
                (pl.col('signal_id') == signal_id) &
                (pl.col('obs_date') >= window_start) &
                (pl.col('obs_date') <= window_end)
            ).sort('obs_date')

            if len(df) >= 15:  # Minimum observations
                values = df['value'].to_numpy()
                # Create feature vector from the signal topology
                vectors[window_days] = np.array([
                    np.mean(values),
                    np.std(values),
                    np.min(values),
                    np.max(values),
                    len(values)
                ])

        if vectors:
            window_vectors[signal_id] = vectors

    return window_vectors


# =============================================================================
# ENGINE ORCHESTRATION (delegation to 9 canonical engines)
# =============================================================================

def compute_cohort_geometry(matrix: pd.DataFrame, cohort: str, window_end: date) -> Dict[str, Any]:
    """
    Orchestrate geometry computation on cohort matrix.

    Calls all 9 canonical GEOMETRY engines and extracts metrics.

    Args:
        matrix: DataFrame (rows=time, cols=signals)
        cohort: Cohort identifier
        window_end: Window end date

    Returns:
        Dict of cohort-level metrics
    """
    if matrix.empty or matrix.shape[1] < 2:
        return {}

    results = {}
    run_id = f"{cohort}_{window_end}"

    # 1. DISTANCE ENGINE
    try:
        distance_engine = DistanceEngine()
        distance_result = distance_engine.run(matrix, run_id=run_id)

        # Extract summary metrics
        if 'distance_matrix_euclidean' in distance_result:
            dist_matrix = distance_result['distance_matrix_euclidean']
            results['distance_mean'] = float(np.mean(dist_matrix[np.triu_indices_from(dist_matrix, k=1)]))
            results['distance_std'] = float(np.std(dist_matrix[np.triu_indices_from(dist_matrix, k=1)]))
    except Exception as e:
        logger.warning(f"Distance engine failed: {e}")

    # 2. PCA ENGINE
    try:
        pca_engine = PCAEngine()
        # n_components must be <= min(n_samples, n_features)
        n_comp = min(5, matrix.shape[0], matrix.shape[1] - 1)
        if n_comp < 1:
            raise ValueError(f"Insufficient data for PCA: {matrix.shape}")
        pca_result = pca_engine.run(matrix, run_id=run_id, n_components=n_comp)

        results['pca_variance_pc1'] = pca_result.get('variance_pc1', 0)
        results['pca_variance_pc2'] = pca_result.get('variance_pc2', 0)
        results['pca_variance_pc3'] = pca_result.get('variance_pc3', 0)
        results['pca_cumulative_3'] = pca_result.get('cumulative_variance_3', 0)
        results['pca_effective_dim'] = pca_result.get('effective_dimensionality', 0)
    except Exception as e:
        logger.warning(f"PCA engine failed: {e}")

    # 3. CLUSTERING ENGINE
    try:
        # n_clusters must be < n_samples and reasonable for n_features
        n_clusters = min(5, matrix.shape[0] - 1, matrix.shape[1])
        if n_clusters < 2:
            raise ValueError(f"Insufficient data for clustering: {matrix.shape}")
        clustering_engine = ClusteringEngine()
        clustering_result = clustering_engine.run(matrix, run_id=run_id, n_clusters=n_clusters)

        results['clustering_silhouette'] = clustering_result.get('silhouette_score', 0)
        results['clustering_n_clusters'] = n_clusters
    except Exception as e:
        logger.warning(f"Clustering engine failed: {e}")

    # 4. MST ENGINE
    try:
        mst_engine = MSTEngine()
        mst_result = mst_engine.run(matrix, run_id=run_id)

        results['mst_total_weight'] = mst_result.get('total_weight', 0)
        results['mst_avg_degree'] = mst_result.get('average_degree', 0)
    except Exception as e:
        logger.warning(f"MST engine failed: {e}")

    # 5. LOF ENGINE
    try:
        lof_engine = LOFEngine()
        lof_result = lof_engine.run(matrix, run_id=run_id)

        results['lof_mean'] = lof_result.get('mean_lof', 0)
        results['lof_n_outliers'] = lof_result.get('n_outliers', 0)
    except Exception as e:
        logger.warning(f"LOF engine failed: {e}")

    # 6. CONVEX HULL ENGINE
    try:
        hull_engine = ConvexHullEngine()
        hull_result = hull_engine.run(matrix, run_id=run_id)

        results['hull_volume'] = hull_result.get('volume', 0)
        results['hull_surface_area'] = hull_result.get('surface_area', 0)
    except Exception as e:
        logger.warning(f"Convex hull engine failed: {e}")

    # 7. MUTUAL INFORMATION ENGINE (cohort-level)
    try:
        mi_engine = MutualInformationEngine()
        mi_result = mi_engine.run(matrix, run_id=run_id)
        # Cohort-level MI summary (mean of pairwise)
        if 'mutual_information_matrix' in mi_result:
            mi_matrix = mi_result['mutual_information_matrix']
            results['mi_mean'] = float(np.mean(mi_matrix[np.triu_indices_from(mi_matrix, k=1)]))
    except Exception as e:
        logger.debug(f"MI cohort-level failed: {e}")

    # 8. COPULA ENGINE (cohort-level)
    try:
        copula_engine = CopulaEngine()
        copula_result = copula_engine.run(matrix, run_id=run_id)
        results['copula_upper_mean'] = copula_result.get('upper_tail_dependence', 0)
        results['copula_lower_mean'] = copula_result.get('lower_tail_dependence', 0)
    except Exception as e:
        logger.debug(f"Copula cohort-level failed: {e}")

    # 9. BARYCENTER ENGINE - handled separately with window_vectors

    # Add metadata
    results['n_signals'] = matrix.shape[1]
    results['n_observations'] = matrix.shape[0]

    return results


def compute_barycenter_metrics(
    cohort: str,
    window_end: date,
    weights: Optional[Dict[int, float]] = None
) -> Dict[str, Any]:
    """
    Compute barycenter metrics for cohort using the canonical BarycenterEngine.

    Fetches multi-window vectors and calls the engine.

    Returns:
        Dict with barycenter_mean_dispersion, barycenter_mean_alignment, barycenter_n_computed
    """
    if weights is None:
        weights = get_barycenter_weights()

    # Get window sizes from weights
    window_sizes = sorted(weights.keys())

    # Build window_vectors for each signal
    window_vectors = get_signal_window_vectors(cohort, window_end, window_sizes)

    if not window_vectors:
        return {
            'barycenter_mean_dispersion': None,
            'barycenter_mean_alignment': None,
            'barycenter_n_computed': 0,
        }

    # Call canonical barycenter engine
    barycenter_result = compute_barycenter(window_vectors, weights)

    return {
        'barycenter_mean_dispersion': barycenter_result.get('mean_dispersion'),
        'barycenter_mean_alignment': barycenter_result.get('mean_alignment'),
        'barycenter_n_computed': barycenter_result.get('n_computed', 0),
    }


def compute_pairwise_geometry(
    series_a: np.ndarray,
    series_b: np.ndarray,
    ind_a: str,
    ind_b: str
) -> Dict[str, float]:
    """
    Compute pairwise geometry metrics directly.

    Direct computation without engine class overhead for efficiency.

    Args:
        series_a, series_b: Aligned signal topology
        ind_a, ind_b: Signal names

    Returns:
        Dict of pairwise metrics
    """
    if len(series_a) < 5 or len(series_b) < 5:
        return {}

    results = {}

    # 1. DISTANCE METRICS (direct computation)
    try:
        # Euclidean distance between normalized series
        a_norm = (series_a - np.mean(series_a)) / (np.std(series_a) + 1e-10)
        b_norm = (series_b - np.mean(series_b)) / (np.std(series_b) + 1e-10)
        results['distance_euclidean'] = float(np.linalg.norm(a_norm - b_norm))

        # Correlation distance: 1 - correlation
        corr = np.corrcoef(series_a, series_b)[0, 1]
        if not np.isnan(corr):
            results['distance_correlation'] = float(1.0 - corr)
    except Exception as e:
        logger.debug(f"Distance pairwise failed: {e}")

    # 2. MUTUAL INFORMATION (binned estimation)
    try:
        from sklearn.metrics import mutual_info_score
        # Discretize into 10 bins for MI estimation
        n_bins = min(10, len(series_a) // 5)
        if n_bins >= 2:
            a_binned = np.digitize(series_a, np.histogram_bin_edges(series_a, bins=n_bins)[:-1])
            b_binned = np.digitize(series_b, np.histogram_bin_edges(series_b, bins=n_bins)[:-1])
            mi = mutual_info_score(a_binned, b_binned)
            results['mutual_information'] = float(mi)
    except Exception as e:
        logger.debug(f"MI pairwise failed: {e}")

    # 3. COPULA METRICS (tail dependence and rank correlation)
    try:
        from scipy import stats

        # Convert to uniform marginals (empirical CDF)
        n = len(series_a)
        u = stats.rankdata(series_a) / (n + 1)
        v = stats.rankdata(series_b) / (n + 1)

        # Tail dependence via threshold approach
        thresholds = [0.05, 0.10, 0.15]
        lower_deps, upper_deps = [], []

        for q in thresholds:
            # Lower tail
            mask_lower = u <= q
            if mask_lower.sum() > 0:
                lower_deps.append((v[mask_lower] <= q).mean())
            # Upper tail
            mask_upper = u >= (1 - q)
            if mask_upper.sum() > 0:
                upper_deps.append((v[mask_upper] >= (1 - q)).mean())

        results['copula_lower_tail'] = float(np.mean(lower_deps)) if lower_deps else 0.0
        results['copula_upper_tail'] = float(np.mean(upper_deps)) if upper_deps else 0.0

        # Kendall's tau
        tau, _ = stats.kendalltau(series_a, series_b)
        results['copula_kendall_tau'] = float(tau) if not np.isnan(tau) else 0.0
    except Exception as e:
        logger.debug(f"Copula pairwise failed: {e}")

    return results


# =============================================================================
# MODE & WAVELET MODULES (inline computation)
# =============================================================================

def compute_mode_metrics(
    cohort: str,
    domain_id: str = 'default',
    field_df: Optional[pl.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Compute mode discovery metrics for a cohort.

    Calls the modes module to discover behavioral modes from Laplace signatures.

    Args:
        cohort: Cohort identifier
        domain_id: Domain identifier
        field_df: Optional pre-loaded Laplace field DataFrame

    Returns:
        Dict with mode metrics (n_modes, mode_entropy_mean, etc.)
    """
    results = {}

    # Get cohort signals first (needed for lazy filter)
    signals = get_cohort_signals(cohort)
    if len(signals) < 3:
        return results

    # Try to load Laplace field data if not provided (lazy scan with filter)
    if field_df is None:
        try:
            field_path = get_path(SIGNALS)
            if Path(field_path).exists():
                # Lazy scan with filter pushdown for cohort signals
                field_df = (
                    pl.scan_parquet(field_path)
                    .filter(pl.col('signal_id').is_in(signals))
                    .collect()
                )
            else:
                logger.debug(f"No Laplace field data found for mode discovery")
                return results
        except Exception as e:
            logger.debug(f"Could not load Laplace field: {e}")
            return results

    try:
        # Discover modes using the module
        modes_df = discover_modes(field_df, domain_id, cohort, signals)

        if modes_df is not None and len(modes_df) > 0:
            results['mode_n_discovered'] = int(modes_df['mode_id'].nunique())
            results['mode_affinity_mean'] = float(modes_df['mode_affinity'].mean())
            results['mode_affinity_std'] = float(modes_df['mode_affinity'].std())
            results['mode_entropy_mean'] = float(modes_df['mode_entropy'].mean())
            results['mode_entropy_std'] = float(modes_df['mode_entropy'].std())

            # Mode distribution
            mode_counts = modes_df['mode_id'].value_counts()
            if len(mode_counts) > 0:
                results['mode_dominant_size'] = int(mode_counts.iloc[0])
                results['mode_balance'] = float(mode_counts.std() / (mode_counts.mean() + 1e-10))

            logger.debug(f"Mode discovery for {cohort}: {results['mode_n_discovered']} modes")
    except Exception as e:
        logger.debug(f"Mode discovery failed for {cohort}: {e}")

    return results


def compute_wavelet_metrics(
    cohort: str,
    observations: Optional[pl.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Compute wavelet microscope metrics for a cohort.

    Identifies which frequency bands show earliest degradation.

    Args:
        cohort: Cohort identifier
        observations: Optional pre-loaded observations DataFrame

    Returns:
        Dict with wavelet degradation metrics
    """
    results = {}

    # Try to load observations if not provided (lazy scan with cohort filter)
    if observations is None:
        try:
            # Use lazy scan with filter pushdown for cohort
            lazy_obs = pl.scan_parquet(get_path(OBSERVATIONS))
            schema = lazy_obs.collect_schema()
            if 'cohort_id' in schema.names():
                observations = lazy_obs.filter(pl.col('cohort_id') == cohort).collect()
            else:
                # Filter by signal_id pattern if no cohort_id column
                observations = lazy_obs.filter(
                    pl.col('signal_id').str.starts_with(cohort + '_')
                ).collect()
        except Exception as e:
            logger.debug(f"Could not load observations for wavelet: {e}")
            return results

    try:
        # Run wavelet microscope
        wavelet_df = run_wavelet_microscope(observations, cohort)

        if wavelet_df is not None and len(wavelet_df) > 0:
            # Extract cohort-level features
            wavelet_features = extract_wavelet_features(wavelet_df, cohort)
            results.update(wavelet_features)
            logger.debug(f"Wavelet analysis for {cohort}: {len(results)} features")
    except Exception as e:
        logger.debug(f"Wavelet analysis failed for {cohort}: {e}")

    return results


# =============================================================================
# DATABASE STORAGE (orchestration)
# =============================================================================

def ensure_schema():
    """Ensure geometry directory exists."""
    ensure_directory()


# Key columns for upsert operations
GEOMETRY_COHORT_KEY_COLS = ['cohort_id', 'window_end', 'window_days']
GEOMETRY_PAIRS_KEY_COLS = ['signal_a', 'signal_b', 'window_end', 'window_days']


def store_cohort_geometry_batch(rows: List[Dict[str, Any]], weighted: bool = False):
    """Store batch of cohort geometry metrics to Parquet (both weighted and unweighted)."""
    if not rows:
        return

    df = pl.DataFrame(rows, infer_schema_length=None)

    # New 5-file schema: all geometry goes to geometry.parquet
    upsert_parquet(df, get_path(GEOMETRY), GEOMETRY_COHORT_KEY_COLS)

    logger.info(f"Wrote {len(rows)} cohort geometry rows")


def make_cohort_row(
    cohort: str,
    window_end: date,
    window_days: int,
    metrics: Dict[str, Any],
    include_weight: bool = False
) -> Dict[str, Any]:
    """Create a row dict for cohort geometry."""
    row = {
        'cohort_id': cohort,
        'window_end': window_end,
        'window_days': window_days,
    }
    if include_weight:
        row['window_weight'] = get_window_weight(window_days)
    row.update({
        'n_signals': metrics.get('n_signals', 0),
        'n_observations': metrics.get('n_observations', 0),
        'distance_mean': metrics.get('distance_mean'),
        'distance_std': metrics.get('distance_std'),
        'pca_variance_pc1': metrics.get('pca_variance_pc1'),
        'pca_variance_pc2': metrics.get('pca_variance_pc2'),
        'pca_variance_pc3': metrics.get('pca_variance_pc3'),
        'pca_cumulative_3': metrics.get('pca_cumulative_3'),
        'pca_effective_dim': metrics.get('pca_effective_dim'),
        'clustering_silhouette': metrics.get('clustering_silhouette'),
        'clustering_n_clusters': metrics.get('clustering_n_clusters'),
        'mst_total_weight': metrics.get('mst_total_weight'),
        'mst_avg_degree': metrics.get('mst_avg_degree'),
        'lof_mean': metrics.get('lof_mean'),
        'lof_n_outliers': metrics.get('lof_n_outliers'),
        'hull_volume': metrics.get('hull_volume'),
        'hull_surface_area': metrics.get('hull_surface_area'),
        'barycenter_mean_dispersion': metrics.get('barycenter_mean_dispersion'),
        'barycenter_mean_alignment': metrics.get('barycenter_mean_alignment'),
        'barycenter_n_computed': metrics.get('barycenter_n_computed'),
        # Mode discovery metrics (from prism.engines.modes)
        'mode_n_discovered': metrics.get('mode_n_discovered'),
        'mode_affinity_mean': metrics.get('mode_affinity_mean'),
        'mode_affinity_std': metrics.get('mode_affinity_std'),
        'mode_entropy_mean': metrics.get('mode_entropy_mean'),
        'mode_entropy_std': metrics.get('mode_entropy_std'),
        'mode_dominant_size': metrics.get('mode_dominant_size'),
        'mode_balance': metrics.get('mode_balance'),
        # Wavelet degradation metrics (from prism.engines.wavelet_microscope)
        'wavelet_max_degradation': metrics.get('wavelet_max_degradation'),
        'wavelet_mean_degradation': metrics.get('wavelet_mean_degradation'),
        'wavelet_n_degrading': metrics.get('wavelet_n_degrading'),
        'wavelet_worst_snr_change': metrics.get('wavelet_worst_snr_change'),
        'wavelet_dominant_band': metrics.get('wavelet_dominant_band'),
        'computed_at': datetime.now(),
    })
    return row


def store_pairwise_geometry_batch(rows: List[Dict[str, Any]], weighted: bool = False):
    """Store batch of pairwise geometry metrics to geometry.parquet."""
    if not rows:
        return

    df = pl.DataFrame(rows, infer_schema_length=None)

    # All geometry goes to geometry.parquet
    upsert_parquet(df, get_path(GEOMETRY), GEOMETRY_PAIRS_KEY_COLS)

    logger.info(f"Wrote {len(rows)} pairwise geometry rows")


def make_pairwise_row(
    ind_a: str,
    ind_b: str,
    window_end: date,
    window_days: int,
    metrics: Dict[str, float],
    include_weight: bool = False
) -> Dict[str, Any]:
    """Create a row dict for pairwise geometry."""
    row = {
        'signal_a': ind_a,
        'signal_b': ind_b,
        'window_end': window_end,
        'window_days': window_days,
    }
    if include_weight:
        row['window_weight'] = get_window_weight(window_days)
    row.update({
        'distance_euclidean': metrics.get('distance_euclidean'),
        'distance_correlation': metrics.get('distance_correlation'),
        'mutual_information': metrics.get('mutual_information'),
        'copula_upper_tail': metrics.get('copula_upper_tail'),
        'copula_lower_tail': metrics.get('copula_lower_tail'),
        'copula_kendall_tau': metrics.get('copula_kendall_tau'),
        'computed_at': datetime.now(),
    })
    return row


# =============================================================================
# MAIN RUNNERS
# =============================================================================

def run_cohort_geometry(
    cohort: str,
    window_end: date,
    window_days: int,
    include_weight: bool = False,
    run_bisection: bool = False  # Disabled for now - needs parquet adaptation
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Run geometry for a cohort at a specific window.

    Calls all 9 canonical engines.

    Returns:
        Tuple of (result_dict, row_dict for batch storage)
    """
    matrix = get_cohort_data_matrix(cohort, window_end, window_days)

    if matrix.empty or matrix.shape[1] < 2:
        logger.warning(f"Insufficient data for {cohort} at {window_end} ({window_days}d)")
        return {'status': 'insufficient_data'}, None

    # Compute cohort-level geometry (8 canonical engines)
    metrics = compute_cohort_geometry(matrix, cohort, window_end)

    # Compute barycenters (9th canonical engine)
    barycenter_metrics = compute_barycenter_metrics(cohort, window_end)
    metrics.update(barycenter_metrics)

    # Compute mode discovery metrics (inline module)
    mode_metrics = compute_mode_metrics(cohort)
    metrics.update(mode_metrics)

    # Compute wavelet degradation metrics (inline module)
    wavelet_metrics = compute_wavelet_metrics(cohort)
    metrics.update(wavelet_metrics)

    # Create row for batch storage
    row = make_cohort_row(cohort, window_end, window_days, metrics, include_weight=include_weight)

    logger.info(f"  {cohort} @ {window_end} ({window_days}d): {matrix.shape[1]} signals, "
                f"PCA_1={(metrics.get('pca_variance_pc1') or 0):.3f}, "
                f"barycenter_disp={(metrics.get('barycenter_mean_dispersion') or 0):.3f}")

    return {
        'status': 'success',
        'n_signals': metrics.get('n_signals', 0),
        'n_observations': metrics.get('n_observations', 0),
    }, row


def run_pairwise_geometry(
    cohort: str,
    window_end: date,
    window_days: int,
    include_weight: bool = False
) -> List[Dict[str, Any]]:
    """Run pairwise geometry for all signal pairs in a cohort."""
    signals = get_cohort_signals(cohort)
    rows = []

    if len(signals) < 2:
        logger.warning(f"Cohort {cohort} has fewer than 2 signals")
        return rows

    for i, ind_a in enumerate(signals):
        for ind_b in signals[i+1:]:
            series_a, series_b = get_pairwise_data(ind_a, ind_b, window_end, window_days)

            if len(series_a) < 5:
                continue

            metrics = compute_pairwise_geometry(series_a, series_b, ind_a, ind_b)
            row = make_pairwise_row(ind_a, ind_b, window_end, window_days, metrics, include_weight=include_weight)
            rows.append(row)

    logger.info(f"  Pairwise: {len(rows)} pairs processed for {cohort} @ {window_end} ({window_days}d)")
    return rows


def run_window_tier(
    cohorts: List[str],
    window_name: str,
    start_date: date,
    end_date: date,
    include_pairwise: bool = True
) -> Dict[str, int]:
    """
    Run geometry for all cohorts across a window tier's date range.

    Uses stride from config/stride.yaml.

    Args:
        cohorts: List of cohorts to process
        window_name: Window tier name ('anchor', 'bridge', 'scout', 'micro')
        start_date: Start of date range
        end_date: End of date range
        include_pairwise: Whether to compute pairwise geometry

    Returns:
        Dict with processing stats
    """
    config = load_stride_config()
    window = config.get_window(window_name)
    window_days = window.window_days

    # Generate dates at configured stride
    dates = get_window_dates(window_name, start_date, end_date, config)

    logger.info(f"Window tier: {window_name} ({window_days}d, stride {window.stride_days}d)")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Snapshots: {len(dates)}")

    cohort_rows = []
    pairwise_rows = []

    for window_end in dates:
        for cohort in cohorts:
            result, row = run_cohort_geometry(cohort, window_end, window_days)
            if row:
                cohort_rows.append(row)

            if include_pairwise:
                pair_rows = run_pairwise_geometry(cohort, window_end, window_days)
                pairwise_rows.extend(pair_rows)

    # Batch store results
    if cohort_rows:
        store_cohort_geometry_batch(cohort_rows)
    if pairwise_rows:
        store_pairwise_geometry_batch(pairwise_rows)

    return {
        'cohort_rows': len(cohort_rows),
        'pairwise_rows': len(pairwise_rows),
        'snapshots': len(dates),
    }


# =============================================================================
# PROGRESS TRACKING
# =============================================================================

PROGRESS_PATH = Path('data/geometry/.progress_geometry.parquet')


def get_completed_windows() -> set:
    """Get set of completed (cohort_id, window_end, window_days) tuples."""
    if not PROGRESS_PATH.exists():
        return set()
    try:
        df = pl.read_parquet(PROGRESS_PATH)
        return set(
            (row['cohort_id'], row['window_end'], row['window_days'])
            for row in df.iter_rows(named=True)
        )
    except Exception:
        return set()


def mark_window_complete(cohort_id: str, window_end: date, window_days: int):
    """Mark a window as complete."""
    new_row = pl.DataFrame([{
        'cohort_id': cohort_id,
        'window_end': window_end,
        'window_days': window_days,
        'completed_at': datetime.now(),
    }])

    if PROGRESS_PATH.exists():
        existing = pl.read_parquet(PROGRESS_PATH)
        combined = pl.concat([existing, new_row])
    else:
        combined = new_row

    combined.write_parquet(PROGRESS_PATH)


def clear_progress():
    """Clear progress tracker."""
    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()
        logger.info("Progress cleared (--force)")


# =============================================================================
# V3 ARCHITECTURE: GEOMETRY FROM VECTOR METRICS
# =============================================================================

def load_vector_features() -> Tuple[Optional[pl.DataFrame], Dict]:
    """
    Load vector metrics from parquet and prepare for geometry computation.

    Returns:
        Tuple of (feature_matrix DataFrame, metadata dict)
    """
    vector_path = get_path(VECTOR)
    if not vector_path.exists():
        logger.warning(f"No vector data found at {vector_path}. Run signal_vector first.")
        return None, {}

    # Load vector data
    df = pl.read_parquet(vector_path)

    # Filter to only windowed metrics (exclude dense signals)
    # Windowed metrics have signal_type in ['windowed', 'sparse', null]
    if 'signal_type' in df.columns:
        df = df.filter(
            (pl.col('signal_type').is_null()) |
            (pl.col('signal_type') != 'dense')
        )

    # Get unique entities and timestamps
    entities = df['entity_id'].unique().sort().to_list()
    timestamps = df['timestamp'].unique().sort().to_list()

    logger.info(f"Loaded vector data: {len(entities)} entities, {len(timestamps)} timestamps")

    metadata = {
        'n_entities': len(entities),
        'n_timestamps': len(timestamps),
        'n_rows': len(df),
    }

    return df, metadata


def create_feature_matrix(
    df: pl.DataFrame,
    timestamp: float,
    entity_id: str,
) -> Optional[np.ndarray]:
    """
    Create feature vector for a specific entity at a timestamp.

    Args:
        df: Vector DataFrame
        timestamp: Timestamp to extract
        entity_id: Entity to extract

    Returns:
        1D numpy array of feature values, or None if insufficient data
    """
    # Filter to this entity and timestamp
    subset = df.filter(
        (pl.col('entity_id') == entity_id) &
        (pl.col('timestamp') == timestamp)
    )

    if len(subset) < 5:
        return None

    # Get values as feature vector
    values = subset['value'].to_numpy()
    values = values[~np.isnan(values)]

    if len(values) < 5:
        return None

    return values


def compute_geometry_from_vectors(
    df: pl.DataFrame,
    verbose: bool = True,
) -> List[Dict]:
    """
    Compute geometry metrics from vector data using canonical engines.

    Processes each (entity_id, timestamp) and computes:
    - PCA (dimensionality reduction)
    - Clustering (signal grouping)
    - MST (minimum spanning tree)
    - LOF (outlier detection)
    - Distance metrics
    - Mutual Information
    - Copula (tail dependence)
    - Convex Hull (feature space volume)

    Args:
        df: Vector DataFrame with entity_id, signal_id, engine, timestamp, value
        verbose: Print progress

    Returns:
        List of geometry row dictionaries
    """
    rows = []
    computed_at = datetime.now()

    # Group by entity_id and timestamp
    groups = df.group_by(['entity_id', 'timestamp']).agg([
        pl.col('value').count().alias('n_features'),
        pl.col('signal_id').n_unique().alias('n_signals'),
        pl.col('engine').n_unique().alias('n_engines'),
    ]).sort(['entity_id', 'timestamp'])

    n_groups = len(groups)
    if verbose:
        logger.info(f"Computing geometry for {n_groups} (entity, timestamp) combinations")

    # Process each group
    for idx, row in enumerate(groups.iter_rows(named=True)):
        entity_id = row['entity_id']
        timestamp = row['timestamp']
        n_features = row['n_features']

        # Skip if too few features
        if n_features < 10:
            continue

        # Get feature data for this entity at this timestamp
        subset_df = df.filter(
            (pl.col('entity_id') == entity_id) &
            (pl.col('timestamp') == timestamp)
        )

        # Pivot to matrix: rows=engine, cols=source_signal, values=mean(value)
        # This creates a dense matrix where each row is an engine type
        try:
            pivoted = subset_df.group_by(['engine', 'source_signal']).agg(
                pl.col('value').mean().alias('mean_value')
            ).pivot(
                index='engine',
                on='source_signal',
                values='mean_value'
            )

            # Convert to pandas for engine compatibility
            matrix = pivoted.drop('engine').to_pandas()

            # Fill NaN with 0 (some engines don't compute for all signals)
            matrix = matrix.fillna(0)

            if matrix.shape[0] < 3 or matrix.shape[1] < 3:
                continue

        except Exception as e:
            continue

        run_id = f"{entity_id}_{timestamp}"
        metrics = {}

        # 1. PCA ENGINE
        try:
            pca_engine = PCAEngine()
            n_comp = min(5, matrix.shape[0], matrix.shape[1] - 1)
            if n_comp >= 1:
                pca_result = pca_engine.run(matrix, run_id=run_id, n_components=n_comp)
                metrics['pca_variance_pc1'] = pca_result.get('variance_pc1', 0)
                metrics['pca_variance_pc2'] = pca_result.get('variance_pc2', 0)
                metrics['pca_effective_dim'] = pca_result.get('effective_dimensionality', 0)
        except Exception as e:
            pass

        # 2. CLUSTERING ENGINE
        try:
            n_clusters = min(5, matrix.shape[0] - 1, matrix.shape[1])
            if n_clusters >= 2:
                clustering_engine = ClusteringEngine()
                clustering_result = clustering_engine.run(matrix, run_id=run_id, n_clusters=n_clusters)
                metrics['clustering_silhouette'] = clustering_result.get('silhouette_score', 0)
        except Exception:
            pass

        # 3. MST ENGINE
        try:
            mst_engine = MSTEngine()
            mst_result = mst_engine.run(matrix, run_id=run_id)
            metrics['mst_total_weight'] = mst_result.get('total_weight', 0)
            metrics['mst_avg_degree'] = mst_result.get('average_degree', 0)
        except Exception:
            pass

        # 4. LOF ENGINE
        try:
            lof_engine = LOFEngine()
            lof_result = lof_engine.run(matrix, run_id=run_id)
            metrics['lof_mean'] = lof_result.get('avg_lof_score', 0)
            metrics['lof_n_outliers'] = lof_result.get('n_outliers_auto', 0)
        except Exception:
            pass

        # 5. DISTANCE ENGINE
        try:
            distance_engine = DistanceEngine()
            distance_result = distance_engine.run(matrix, run_id=run_id)
            metrics['distance_mean'] = distance_result.get('avg_euclidean_distance', 0)
            metrics['distance_std'] = distance_result.get('max_euclidean_distance', 0) - distance_result.get('min_euclidean_distance', 0)
        except Exception:
            pass

        # 6. MUTUAL INFORMATION ENGINE
        try:
            mi_engine = MutualInformationEngine()
            mi_result = mi_engine.run(matrix, run_id=run_id)
            metrics['mi_mean'] = mi_result.get('avg_mi', 0)
        except Exception:
            pass

        # 7. COPULA ENGINE
        try:
            copula_engine = CopulaEngine()
            copula_result = copula_engine.run(matrix, run_id=run_id)
            metrics['copula_upper_tail'] = copula_result.get('avg_upper_tail', 0)
            metrics['copula_lower_tail'] = copula_result.get('avg_lower_tail', 0)
        except Exception:
            pass

        # 8. CONVEX HULL ENGINE
        try:
            hull_engine = ConvexHullEngine()
            hull_result = hull_engine.run(matrix, run_id=run_id)
            metrics['hull_volume'] = hull_result.get('hull_volume', 0)
            metrics['hull_centroid_dist'] = hull_result.get('centroid_avg_distance', 0)
        except Exception:
            pass

        # Get signal_ids (metadata only)
        signal_ids_subset = subset_df['signal_id'].unique().to_list()
        signal_ids_str = ','.join(sorted(signal_ids_subset)[:50]) if signal_ids_subset else ''

        # Build row from engine outputs only - no inline calculations
        rows.append({
            'entity_id': entity_id,
            'timestamp': float(timestamp),
            'n_features': int(n_features),
            'n_signals': int(row['n_signals']),
            'n_engines': int(row['n_engines']),
            # PCA engine
            'pca_var_1': float(metrics.get('pca_variance_pc1', 0)),
            'pca_var_2': float(metrics.get('pca_variance_pc2', 0)),
            'pca_effective_dim': float(metrics.get('pca_effective_dim', 0)),
            # Clustering engine
            'clustering_silhouette': float(metrics.get('clustering_silhouette', 0)),
            # MST engine
            'mst_total_weight': float(metrics.get('mst_total_weight', 0)),
            'mst_avg_degree': float(metrics.get('mst_avg_degree', 0)),
            # LOF engine
            'lof_mean': float(metrics.get('lof_mean', 0)),
            'lof_n_outliers': int(metrics.get('lof_n_outliers', 0)),
            # Distance engine
            'distance_mean': float(metrics.get('distance_mean', 0)),
            'distance_std': float(metrics.get('distance_std', 0)),
            # Mutual Information engine
            'mi_mean': float(metrics.get('mi_mean', 0)),
            # Copula engine
            'copula_upper_tail': float(metrics.get('copula_upper_tail', 0)),
            'copula_lower_tail': float(metrics.get('copula_lower_tail', 0)),
            # Convex Hull engine
            'hull_volume': float(metrics.get('hull_volume', 0)),
            'hull_centroid_dist': float(metrics.get('hull_centroid_dist', 0)),
            # Metadata
            'signal_ids': signal_ids_str,
            'computed_at': computed_at,
        })

        if verbose and (idx + 1) % 100 == 0:
            logger.info(f"  Processed {idx + 1}/{n_groups} groups...")

    if verbose:
        logger.info(f"  Computed {len(rows)} geometry snapshots")

    return rows


def compute_geometry_v2(
    fields: Dict[str, LaplaceField],
    verbose: bool = True,
) -> Tuple[List[GeometrySnapshot], Dict]:
    """
    V2 Architecture: Compute geometry snapshots from Laplace fields.

    Uses compute_geometry_at_t for each unified timestamp.

    Args:
        fields: Dict mapping signal_id to LaplaceField
        verbose: Print progress

    Returns:
        (list of GeometrySnapshots, dict of statistics)
    """
    if not fields:
        return [], {'n_snapshots': 0}

    # Get unified timestamps from all fields
    timestamps = get_unified_timestamps(fields)

    if verbose:
        logger.info(f"V2 Geometry: {len(fields)} signals, {len(timestamps)} timestamps")

    # Compute geometry at each timestamp
    snapshots = compute_geometry_trajectory(fields, timestamps)

    if verbose:
        logger.info(f"  Computed {len(snapshots)} geometry snapshots")

    stats = {
        'n_signals': len(fields),
        'n_timestamps': len(timestamps),
        'n_snapshots': len(snapshots),
    }

    return snapshots, stats


def snapshots_to_rows(
    snapshots: List[GeometrySnapshot],
    computed_at: datetime = None,
) -> List[Dict]:
    """
    Convert GeometrySnapshots to row format for parquet storage.

    Args:
        snapshots: List of GeometrySnapshot objects
        computed_at: Computation timestamp

    Returns:
        List of row dictionaries
    """
    if computed_at is None:
        computed_at = datetime.now()

    rows = []
    for snap in snapshots:
        # Store per-snapshot metrics
        rows.append({
            'timestamp': snap.timestamp,
            'n_signals': snap.n_signals,
            'divergence': float(snap.divergence),
            'n_modes': snap.n_modes,
            'mean_mode_coherence': float(np.mean(snap.mode_coherence)) if len(snap.mode_coherence) > 0 else 0.0,
            'mean_coupling': float(np.mean(snap.coupling_matrix)) if snap.coupling_matrix.size > 0 else 0.0,
            'signal_ids': ','.join(snap.signal_ids),
            'computed_at': computed_at,
        })

    return rows


def run_v2_geometry(verbose: bool = True) -> Dict:
    """
    Run V3 geometry computation from vector metrics.

    Loads vector data, computes geometry metrics, saves to parquet.

    Args:
        verbose: Print progress

    Returns:
        Dict with processing statistics
    """
    # Load vector data
    df, metadata = load_vector_features()

    if df is None or len(df) == 0:
        logger.warning("No vector data loaded. Run signal_vector first.")
        return {'snapshots': 0}

    # Compute geometry from vectors
    rows = compute_geometry_from_vectors(df, verbose=verbose)

    if not rows:
        return {'snapshots': 0, **metadata}

    stats = {
        'n_snapshots': len(rows),
        **metadata,
    }

    if verbose:
        logger.info(f"  Saving {len(rows)} geometry rows...")

    # Save to geometry.parquet
    geom_df = pl.DataFrame(rows, infer_schema_length=None)
    geom_path = get_path(GEOMETRY)
    upsert_parquet(geom_df, geom_path, ['entity_id', 'timestamp'])

    if verbose:
        logger.info(f"  Saved: {geom_path}")

    stats['saved_rows'] = len(rows)
    return stats


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PRISM Geometry Runner - Compute structural geometry from vector signals',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output: data/geometry.parquet

Examples:
  python -m prism.entry_points.geometry              # Production run
  python -m prism.entry_points.geometry --adaptive   # Auto-detect window
  python -m prism.entry_points.geometry --force      # Force recompute
  python -m prism.entry_points.geometry --testing    # Enable test mode
"""
    )

    # Production flags
    parser.add_argument('--adaptive', action='store_true',
                        help='Use adaptive windowing from domain_info.json')
    parser.add_argument('--force', action='store_true',
                        help='Clear progress tracker and recompute all')

    # Testing mode
    parser.add_argument('--testing', action='store_true',
                        help='Enable testing mode')

    args = parser.parse_args()

    # Handle --force
    if args.force:
        clear_progress()

    # Ensure directories exist
    ensure_schema()

    # Run V3 geometry (from vector metrics)
    logger.info("=" * 80)
    logger.info("PRISM GEOMETRY - Vector Metrics Architecture")
    logger.info("=" * 80)
    logger.info(f"Source: data/vector.parquet")
    logger.info(f"Destination: data/geometry.parquet")

    result = run_v2_geometry(verbose=True)

    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  Snapshots: {result.get('n_snapshots', 0)}")
    logger.info(f"  Saved rows: {result.get('saved_rows', 0)}")
    return 0



if __name__ == '__main__':
    exit(main())
