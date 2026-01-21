"""
PRISM Bisection Module
======================

Pure analysis module - NOT a runner/entry point.
Called by geometry.py and state.py to analyze deltas and identify dates needing drill-down.

This module is LAYER-AGNOSTIC. It can analyze displacement for:
    - geometry.observations (cohort geometry)
    - state.observations (temporal dynamics)
    - Any table with window_end, window_days, and metric columns

Flow:
    runner.py computes metrics @ window
        ↓
    bisection.analyze(cohort, window, layer) → dates needing drill-down
        ↓
    runner.py computes metrics @ window/2 for those dates only
        ↓
    bisection.analyze(cohort, window/2, layer) → dates needing drill-down
        ↓
    ... until no more dates need drilling or min_window reached

This module:
    - Reads {layer}.observations
    - Computes displacement (Δ) between adjacent observations
    - Stores displacement in {layer}.displacement
    - Returns list of dates that need tighter window analysis
    - NEVER computes metrics itself - that's the runner's job
"""

import logging
import numpy as np
import polars as pl
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from orthon._internal.db.parquet_store import (
    get_path,
    get_data_root,
    OBSERVATIONS,
    SIGNALS,
    GEOMETRY,
    STATE,
    COHORTS,
)
from orthon._internal.db.polars_io import read_parquet, upsert_parquet

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Displacement thresholds - these determine when to drill down
DISPLACEMENT_THRESHOLDS = {
    'delta_total': 0.3,         # Overall displacement
    'delta_pca': 0.15,          # PCA structure shift
    'delta_clustering': 0.2,    # Cluster structure change
    'delta_hull': 0.25,         # Phase space boundary change
}

# Minimum window size (stop drilling at this point)
MIN_WINDOW_DAYS = 16

# Severity classification
SEVERITY_THRESHOLDS = {
    'major': 1.0,
    'significant': 0.5,
    'minor': 0.3,
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GeometryVector:
    """Geometric signature loaded from database."""
    cohort: str
    window_end: date
    window_days: int

    pca_variance_pc1: float
    pca_variance_pc2: float
    pca_effective_dim: float

    clustering_n_clusters: int
    clustering_silhouette: float
    clustering_balance: float

    mst_mean_weight: float

    hull_area: float
    hull_compactness: float

    lof_outlier_ratio: float

    def to_vector(self) -> np.ndarray:
        """Convert to numpy array for distance calculations."""
        return np.array([
            self.pca_variance_pc1,
            self.pca_variance_pc2,
            self.pca_effective_dim / 5,
            self.clustering_n_clusters / 5,
            self.clustering_silhouette,
            self.clustering_balance,
            self.mst_mean_weight,
            self.hull_area,
            self.hull_compactness,
            self.lof_outlier_ratio,
        ])


@dataclass
class StateVector:
    """State/temporal dynamics signature loaded from database."""
    cohort: str
    window_end: date
    window_days: int

    # Granger causality
    granger_f_a_to_b: float
    granger_f_b_to_a: float
    granger_bidirectional: float

    # Cross-correlation
    xcorr_peak_corr: float
    xcorr_zero_corr: float
    xcorr_synchronous: float

    # Cointegration
    coint_is_cointegrated: float
    coint_t_stat: float

    # DMD
    dmd_is_stable: float
    dmd_growth_rate: float
    dmd_spectral_radius: float

    # Transfer entropy
    te_a_to_b: float
    te_b_to_a: float
    te_net: float

    # DTW
    dtw_normalized: float

    def to_vector(self) -> np.ndarray:
        """Convert to numpy array for distance calculations."""
        return np.array([
            self.granger_f_a_to_b / 10,  # Normalize F-stats
            self.granger_f_b_to_a / 10,
            self.granger_bidirectional,
            self.xcorr_peak_corr,
            self.xcorr_zero_corr,
            self.xcorr_synchronous,
            self.coint_is_cointegrated,
            self.coint_t_stat / 5,  # Normalize t-stats
            self.dmd_is_stable,
            self.dmd_growth_rate,
            self.dmd_spectral_radius,
            self.te_a_to_b,
            self.te_b_to_a,
            self.te_net,
            self.dtw_normalized,
        ])


@dataclass
class Displacement:
    """Displacement between two geometry observations."""
    cohort: str
    date_from: date
    date_to: date
    window_days: int

    delta_total: float
    delta_pca: float
    delta_clustering: float
    delta_mst: float
    delta_hull: float
    delta_lof: float

    drill_down: bool
    drill_reason: Optional[str]
    severity: str  # 'major', 'significant', 'minor', 'noise'
    primary_driver: str  # which component drove the displacement

    # Direction signals
    direction: str  # 'stabilizing', 'destabilizing', 'mixed'
    direction_score: float  # composite score (positive = destabilizing)
    pca_direction: float  # PC1 change (positive = stabilizing)
    hull_direction: float  # Area change (positive = destabilizing)
    mst_direction: float  # Weight change (positive = destabilizing)
    cluster_direction: float  # Silhouette change (positive = stabilizing)


# =============================================================================
# DISPLACEMENT COMPUTATION
# =============================================================================

def compute_displacement(g1, g2) -> Displacement:
    """
    Compute displacement between two observations.
    Works with both GeometryVector and StateVector.
    Returns detailed breakdown of what changed.
    """
    v1 = g1.to_vector()
    v2 = g2.to_vector()

    # Total displacement (normalized Euclidean)
    delta_total = np.linalg.norm(v2 - v1) / (np.linalg.norm(v1) + 1e-10)

    # Detect vector type and compute component deltas accordingly
    if isinstance(g1, GeometryVector):
        # Geometry-specific component deltas
        delta_pca = np.sqrt(
            (g2.pca_variance_pc1 - g1.pca_variance_pc1) ** 2 +
            (g2.pca_variance_pc2 - g1.pca_variance_pc2) ** 2 +
            ((g2.pca_effective_dim - g1.pca_effective_dim) / 5) ** 2
        )

        delta_clustering = np.sqrt(
            ((g2.clustering_n_clusters - g1.clustering_n_clusters) / 3) ** 2 +
            (g2.clustering_silhouette - g1.clustering_silhouette) ** 2 +
            (g2.clustering_balance - g1.clustering_balance) ** 2
        )

        delta_mst = abs(g2.mst_mean_weight - g1.mst_mean_weight)

        delta_hull = np.sqrt(
            (g2.hull_area - g1.hull_area) ** 2 +
            (g2.hull_compactness - g1.hull_compactness) ** 2
        )

        delta_lof = abs(g2.lof_outlier_ratio - g1.lof_outlier_ratio)

        # Determine primary driver
        drivers = {
            'pca': delta_pca,
            'clustering': delta_clustering,
            'mst': delta_mst,
            'hull': delta_hull,
            'lof': delta_lof,
        }
        primary_driver = max(drivers, key=drivers.get)

        # Direction: is geometry stabilizing or destabilizing?
        pca_direction = g2.pca_variance_pc1 - g1.pca_variance_pc1
        hull_direction = g2.hull_area - g1.hull_area
        mst_direction = g2.mst_mean_weight - g1.mst_mean_weight
        cluster_direction = g2.clustering_silhouette - g1.clustering_silhouette

        # Composite direction score: positive = destabilizing
        direction_score = (
            -pca_direction +      # PC1 up = stable
            hull_direction +      # Hull up = unstable
            mst_direction +       # MST up = unstable
            -cluster_direction    # Silhouette up = stable
        )

    elif isinstance(g1, StateVector):
        # State-specific component deltas
        delta_granger = np.sqrt(
            (g2.granger_f_a_to_b - g1.granger_f_a_to_b) ** 2 +
            (g2.granger_f_b_to_a - g1.granger_f_b_to_a) ** 2
        ) / 10  # Normalize

        delta_xcorr = np.sqrt(
            (g2.xcorr_peak_corr - g1.xcorr_peak_corr) ** 2 +
            (g2.xcorr_zero_corr - g1.xcorr_zero_corr) ** 2
        )

        delta_coint = abs(g2.coint_is_cointegrated - g1.coint_is_cointegrated)

        delta_dmd = np.sqrt(
            (g2.dmd_growth_rate - g1.dmd_growth_rate) ** 2 +
            (g2.dmd_spectral_radius - g1.dmd_spectral_radius) ** 2
        )

        delta_te = np.sqrt(
            (g2.te_a_to_b - g1.te_a_to_b) ** 2 +
            (g2.te_b_to_a - g1.te_b_to_a) ** 2
        )

        # Map to geometry-like deltas for consistency
        delta_pca = delta_granger
        delta_clustering = delta_xcorr
        delta_mst = delta_te
        delta_hull = delta_dmd
        delta_lof = delta_coint

        # Determine primary driver
        drivers = {
            'granger': delta_granger,
            'xcorr': delta_xcorr,
            'coint': delta_coint,
            'dmd': delta_dmd,
            'te': delta_te,
        }
        primary_driver = max(drivers, key=drivers.get)

        # Direction for state: destabilizing = dynamics becoming unstable
        # dmd_is_stable decreasing = destabilizing
        # dmd_growth_rate increasing = destabilizing
        # coint_is_cointegrated decreasing = destabilizing (loss of equilibrium)
        pca_direction = g2.granger_f_a_to_b - g1.granger_f_a_to_b
        hull_direction = g2.dmd_growth_rate - g1.dmd_growth_rate  # + = destabilizing
        mst_direction = -(g2.dmd_is_stable - g1.dmd_is_stable)    # stable down = destabilizing
        cluster_direction = g2.coint_is_cointegrated - g1.coint_is_cointegrated  # + = stabilizing

        direction_score = (
            hull_direction +      # Growth rate up = unstable
            mst_direction +       # Stability down = unstable
            -cluster_direction    # Cointegration up = stable
        )

    else:
        # Fallback for unknown vector types
        delta_pca = delta_clustering = delta_mst = delta_hull = delta_lof = 0
        primary_driver = 'unknown'
        pca_direction = hull_direction = mst_direction = cluster_direction = 0
        direction_score = 0

    # Severity classification
    if delta_total >= SEVERITY_THRESHOLDS['major']:
        severity = 'major'
    elif delta_total >= SEVERITY_THRESHOLDS['significant']:
        severity = 'significant'
    elif delta_total >= SEVERITY_THRESHOLDS['minor']:
        severity = 'minor'
    else:
        severity = 'noise'

    # Drill down decision
    drill_down, drill_reason = _should_drill_down(
        delta_total, delta_pca, delta_clustering, delta_hull
    )

    # Determine direction label
    if direction_score > 0.1:
        direction = 'destabilizing'
    elif direction_score < -0.1:
        direction = 'stabilizing'
    else:
        direction = 'mixed'

    return Displacement(
        cohort=g2.cohort,
        date_from=g1.window_end,
        date_to=g2.window_end,
        window_days=g2.window_days,
        delta_total=float(delta_total),
        delta_pca=float(delta_pca),
        delta_clustering=float(delta_clustering),
        delta_mst=float(delta_mst),
        delta_hull=float(delta_hull),
        delta_lof=float(delta_lof),
        drill_down=drill_down,
        drill_reason=drill_reason,
        severity=severity,
        primary_driver=primary_driver,
        direction=direction,
        direction_score=float(direction_score),
        pca_direction=float(pca_direction),
        hull_direction=float(hull_direction),
        mst_direction=float(mst_direction),
        cluster_direction=float(cluster_direction),
    )


def _should_drill_down(delta_total: float, delta_pca: float,
                       delta_clustering: float, delta_hull: float) -> Tuple[bool, Optional[str]]:
    """Determine if we should drill down to shorter window."""
    reasons = []

    if delta_total > DISPLACEMENT_THRESHOLDS['delta_total']:
        reasons.append(f"total={delta_total:.3f}")

    if delta_pca > DISPLACEMENT_THRESHOLDS['delta_pca']:
        reasons.append(f"pca={delta_pca:.3f}")

    if delta_clustering > DISPLACEMENT_THRESHOLDS['delta_clustering']:
        reasons.append(f"cluster={delta_clustering:.3f}")

    if delta_hull > DISPLACEMENT_THRESHOLDS['delta_hull']:
        reasons.append(f"hull={delta_hull:.3f}")

    if reasons:
        return True, "; ".join(reasons)

    return False, None


# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def ensure_schema(layer: str = 'geometry'):
    """Ensure schema exists (directories created by parquet_store)."""
    # Parquet files are created on first write
    # This function is a no-op now, kept for backwards compatibility
    pass


def _store_displacement(d: Displacement, layer: str = 'geometry'):
    """Store displacement record to Parquet."""
    path = get_path(GEOMETRY) if layer == 'geometry' else get_path(STATE)

    df = pl.DataFrame({
        'cohort': [d.cohort],
        'date_from': [d.date_from],
        'date_to': [d.date_to],
        'window_days': [d.window_days],
        'delta_total': [d.delta_total],
        'delta_pca': [d.delta_pca],
        'delta_clustering': [d.delta_clustering],
        'delta_mst': [d.delta_mst],
        'delta_hull': [d.delta_hull],
        'delta_lof': [d.delta_lof],
        'direction': [d.direction],
        'direction_score': [d.direction_score],
        'pca_direction': [d.pca_direction],
        'hull_direction': [d.hull_direction],
        'mst_direction': [d.mst_direction],
        'cluster_direction': [d.cluster_direction],
        'drill_down': [d.drill_down],
        'drill_reason': [d.drill_reason],
        'severity': [d.severity],
        'primary_driver': [d.primary_driver],
        'computed_at': [datetime.now()],
    })

    upsert_parquet(
        df, path,
        key_cols=['cohort', 'date_from', 'date_to', 'window_days']
    )


def _store_shift(d: Displacement, depth: int, layer: str = 'geometry'):
    """Store detected shift (major/significant only) to Parquet."""
    if d.severity in ('major', 'significant'):
        path = get_path(GEOMETRY) if layer == 'geometry' else get_path(STATE)

        df = pl.DataFrame({
            'cohort': [d.cohort],
            'shift_date': [d.date_to],
            'window_days': [d.window_days],
            'depth': [depth],
            'delta_total': [d.delta_total],
            'direction': [d.direction],
            'direction_score': [d.direction_score],
            'severity': [d.severity],
            'primary_driver': [d.primary_driver],
            'detected_at': [datetime.now()],
        })

        upsert_parquet(
            df, path,
            key_cols=['cohort', 'shift_date', 'window_days', 'depth']
        )


def _load_observations(cohort: str, window_days: int, layer: str = 'geometry'):
    """Load observations for a cohort at a specific window size.

    Returns List[GeometryVector] for geometry layer, List[StateVector] for state layer.
    """
    path = get_path(GEOMETRY) if layer == 'geometry' else get_path(STATE)
    df = read_parquet(path)

    if df.is_empty():
        return []

    # Filter for cohort and window_days, sort by window_end
    result = df.filter(
        (pl.col('cohort') == cohort) &
        (pl.col('window_days') == window_days)
    ).sort('window_end')

    if result.is_empty():
        return []

    def get_val(row_dict: dict, key: str, default=0):
        """Safely get value from row dict with default."""
        val = row_dict.get(key)
        if val is None:
            return default
        return val

    vectors = []
    for row_dict in result.to_dicts():
        window_end = row_dict['window_end']
        # Convert to date if it's a datetime
        if hasattr(window_end, 'date'):
            window_end = window_end.date()

        if layer == 'geometry':
            vectors.append(GeometryVector(
                cohort=row_dict['cohort'],
                window_end=window_end,
                window_days=row_dict['window_days'],
                pca_variance_pc1=get_val(row_dict, 'pca_variance_pc1', 0),
                pca_variance_pc2=get_val(row_dict, 'pca_variance_pc2', 0),
                pca_effective_dim=get_val(row_dict, 'pca_effective_dim', 1),
                clustering_n_clusters=get_val(row_dict, 'clustering_n_clusters', 1),
                clustering_silhouette=get_val(row_dict, 'clustering_silhouette', 0),
                clustering_balance=get_val(row_dict, 'clustering_balance', 1),
                mst_mean_weight=get_val(row_dict, 'mst_mean_weight', 0),
                hull_area=get_val(row_dict, 'hull_area', 0),
                hull_compactness=get_val(row_dict, 'hull_compactness', 0),
                lof_outlier_ratio=get_val(row_dict, 'lof_outlier_ratio', 0),
            ))
        elif layer == 'state':
            vectors.append(StateVector(
                cohort=row_dict['cohort'],
                window_end=window_end,
                window_days=row_dict['window_days'],
                granger_f_a_to_b=get_val(row_dict, 'granger_f_a_to_b', 0),
                granger_f_b_to_a=get_val(row_dict, 'granger_f_b_to_a', 0),
                granger_bidirectional=get_val(row_dict, 'granger_bidirectional', 0),
                xcorr_peak_corr=get_val(row_dict, 'xcorr_peak_corr', 0),
                xcorr_zero_corr=get_val(row_dict, 'xcorr_zero_corr', 0),
                xcorr_synchronous=get_val(row_dict, 'xcorr_synchronous', 0),
                coint_is_cointegrated=get_val(row_dict, 'coint_is_cointegrated', 0),
                coint_t_stat=get_val(row_dict, 'coint_t_stat', 0),
                dmd_is_stable=get_val(row_dict, 'dmd_is_stable', 0),
                dmd_growth_rate=get_val(row_dict, 'dmd_growth_rate', 0),
                dmd_spectral_radius=get_val(row_dict, 'dmd_spectral_radius', 0),
                te_a_to_b=get_val(row_dict, 'te_a_to_b', 0),
                te_b_to_a=get_val(row_dict, 'te_b_to_a', 0),
                te_net=get_val(row_dict, 'te_net', 0),
                dtw_normalized=get_val(row_dict, 'dtw_normalized', 0),
            ))

    return vectors


# =============================================================================
# MAIN ANALYSIS FUNCTION (called by geometry.py, state.py)
# =============================================================================

def analyze(
    cohort: str,
    window_days: int,
    layer: str = 'geometry',
    depth: int = 0,
    verbose: bool = True
) -> List[date]:
    """
    Analyze displacement for a cohort at a given window size.

    This is the main entry point called by runners (geometry.py, state.py).

    Args:
        cohort: Cohort to analyze
        window_days: Current window size
        layer: Schema layer ('geometry' or 'state')
        depth: Bisection depth (0 = first pass)
        verbose: Print progress

    Returns:
        List of dates that need drill-down at window_days/2
    """
    ensure_schema(layer)

    # Load observations for this window
    observations = _load_observations(cohort, window_days, layer)

    if len(observations) < 2:
        if verbose:
            print(f"  [bisection] {cohort} @ {window_days}d: Not enough observations ({len(observations)})")
        return []

    if verbose:
        print(f"  [bisection] {cohort} @ {window_days}d: Analyzing {len(observations)} observations")

    # Compute displacements between adjacent observations
    dates_to_drill = []
    n_major = 0
    n_significant = 0
    n_minor = 0

    for i in range(1, len(observations)):
        g_prev = observations[i - 1]
        g_curr = observations[i]

        displacement = compute_displacement(g_prev, g_curr)

        # Store displacement
        _store_displacement(displacement, layer)

        # Store shift if significant
        _store_shift(displacement, depth, layer)

        # Track stats
        if displacement.severity == 'major':
            n_major += 1
        elif displacement.severity == 'significant':
            n_significant += 1
        elif displacement.severity == 'minor':
            n_minor += 1

        # Check if we should drill down
        next_window = window_days // 2
        if displacement.drill_down and next_window >= MIN_WINDOW_DAYS:
            dates_to_drill.append(g_curr.window_end)

    if verbose:
        print(f"  [bisection] Results: {n_major} major, {n_significant} significant, {n_minor} minor")
        print(f"  [bisection] Dates needing drill-down: {len(dates_to_drill)}")

    return dates_to_drill


def get_drill_down_window(current_window: int) -> Optional[int]:
    """Get the next window size for drill-down."""
    next_window = current_window // 2
    if next_window >= MIN_WINDOW_DAYS:
        return next_window
    return None


# =============================================================================
# SUMMARY FUNCTIONS
# =============================================================================

def get_shifts_summary(
    cohort: Optional[str] = None,
    layer: str = 'geometry'
) -> pl.DataFrame:
    """Get summary of detected shifts."""
    path = get_path(GEOMETRY) if layer == 'geometry' else get_path(STATE)
    df = read_parquet(path)

    if df.is_empty():
        return df

    if cohort:
        df = df.filter(pl.col('cohort') == cohort)

    return df.sort('delta_total', descending=True)


def get_displacement_stats(
    cohort: Optional[str] = None,
    layer: str = 'geometry'
) -> List[Dict[str, Any]]:
    """Get displacement statistics."""
    path = get_path(GEOMETRY) if layer == 'geometry' else get_path(STATE)
    df = read_parquet(path)

    if df.is_empty():
        return []

    if cohort:
        df = df.filter(pl.col('cohort') == cohort)

    # Group by severity and compute stats
    result = df.group_by('severity').agg([
        pl.len().alias('n'),
        pl.col('delta_total').mean().alias('avg_delta'),
        pl.col('delta_total').max().alias('max_delta'),
    ])

    # Create severity order for sorting
    severity_order = {'major': 1, 'significant': 2, 'minor': 3, 'noise': 4}
    result = result.with_columns(
        pl.col('severity').replace(severity_order, default=5).alias('_sort_order')
    ).sort('_sort_order').drop('_sort_order')

    return result.to_dicts()


def get_top_shifts(
    n: int = 20,
    cohort: Optional[str] = None,
    layer: str = 'geometry'
) -> pl.DataFrame:
    """Get top N shifts by delta."""
    path = get_path(GEOMETRY) if layer == 'geometry' else get_path(STATE)
    df = read_parquet(path)

    if df.is_empty():
        return df

    # Select columns of interest
    columns = ['shift_date', 'cohort', 'window_days', 'delta_total', 'severity', 'primary_driver']
    available_cols = [c for c in columns if c in df.columns]
    df = df.select(available_cols)

    if cohort:
        df = df.filter(pl.col('cohort') == cohort)

    return df.sort('delta_total', descending=True).head(n)
