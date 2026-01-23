"""
PRISM Phase State
=================

Computes temporal state dynamics by analyzing geometry evolution over time.
This is the TEMPORAL layer - it answers "how is the system evolving?"

STATE DYNAMICS ENGINES (5):
    - energy_dynamics:    Energy trends, acceleration, z-scores
    - tension_dynamics:   Dispersion velocity, alignment evolution
    - phase_detector:     Regime shifts, phase classification
    - cohort_aggregator:  Signal-level to cohort-level metrics
    - transfer_detector:  Cross-cohort transmission patterns

Output: data/phase_state.parquet

Pipeline:
    raw → signal_typology → behavioral_geometry → phase_state → dynamical_systems

Usage:
    python -m prism.entry_points.phase_state              # Production run
    python -m prism.entry_points.phase_state --adaptive   # Auto-detect window
    python -m prism.entry_points.phase_state --force      # Force recompute
    python -m prism.entry_points.phase_state --testing    # Enable test mode
"""

import argparse
import logging
import numpy as np
import polars as pl
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings

from prism.db.parquet_store import (
    ensure_directory,
    get_path,
    get_data_root,
    OBSERVATIONS,
    VECTOR,
    GEOMETRY,
    STATE,
    COHORTS,
)
# Backwards compatibility
SIGNALS = VECTOR
from prism.db.polars_io import read_parquet, upsert_parquet
from prism.db.scratch import TempParquet, merge_temp_results
from prism.engines.utils.parallel import (
    WorkerAssignment,
    divide_by_count,
    generate_temp_path,
    run_workers,
)

# Canonical temporal dynamics engines
from prism.engines.state.energy_dynamics import EnergyDynamicsEngine
from prism.engines.state.tension_dynamics import TensionDynamicsEngine
from prism.engines.state.phase_detector import PhaseDetectorEngine
from prism.engines.cohort_aggregator import CohortAggregatorEngine
from prism.engines.state.transfer_detector import TransferDetectorEngine

# V2 Architecture: State trajectory from geometry snapshots
from prism.engines.state.trajectory import (
    compute_state_trajectory,
    detect_failure_acceleration,
    compute_state_metrics,
    find_acceleration_events,
    compute_trajectory_curvature,
)
from prism.engines.geometry.snapshot import (
    compute_geometry_trajectory,
    snapshot_to_vector,
    get_unified_timestamps,
)
from prism.core.signals.types import GeometrySnapshot, StateTrajectory, LaplaceField

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION (with adaptive domain clock integration)
# =============================================================================

from prism.config.loader import load_delta_thresholds
import json


def load_domain_info() -> Optional[Dict[str, Any]]:
    """
    Load domain_info from config/domain_info.json if available.

    This is saved by signal_vector when running in --adaptive mode.
    """
    import os
    domain_info_path = get_data_root() / "domain_info.json"
    if domain_info_path.exists():
        try:
            with open(domain_info_path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


# Load delta thresholds from config/domain.yaml
_delta_thresholds = load_delta_thresholds()

# State layer thresholds (from domain.yaml or defaults)
VELOCITY_THRESHOLD = _delta_thresholds.get('state_velocity', 0.10)
ACCELERATION_THRESHOLD = _delta_thresholds.get('state_acceleration', 0.05)


def get_lookback_window() -> int:
    """
    Get lookback window from domain_info. Fails if not configured.

    Uses window_samples from DomainClock.
    """
    domain_info = load_domain_info()
    if domain_info:
        window = domain_info.get('window_samples')
        if window:
            return max(20, window)  # Ensure minimum for statistics

    # No fallback - must be configured
    raise RuntimeError(
        "No domain_info.json found. "
        "Run signal_vector with --adaptive flag first to auto-detect window parameters."
    )


def get_default_stride() -> int:
    """
    Get default stride from domain_info. Fails if not configured.

    Uses stride_samples from DomainClock.
    """
    domain_info = load_domain_info()
    if domain_info:
        stride = domain_info.get('stride_samples')
        if stride:
            return max(1, stride)

    # No fallback - must be configured
    raise RuntimeError(
        "No domain_info.json found. "
        "Run signal_vector with --adaptive flag first to auto-detect stride parameters."
    )


# MIN_HISTORY is a statistical minimum, not domain-specific
MIN_HISTORY = 20  # Minimum snapshots needed for dynamics

# Key columns for upsert deduplication
SYSTEM_KEY_COLS = ['state_time']
INDICATOR_DYNAMICS_KEY_COLS = ['signal_id', 'state_time']
TRANSFERS_KEY_COLS = ['state_time', 'cohort_from', 'cohort_to']


# =============================================================================
# DATA FETCHING
# =============================================================================

def get_available_dates() -> List[date]:
    """Get all dates with geometry data from parquet files."""
    geometry_path = get_path(GEOMETRY)
    if not geometry_path.exists():
        return []

    df = pl.read_parquet(geometry_path, columns=['window_end'])
    if len(df) == 0:
        return []

    # Get unique dates sorted
    dates = df.select('window_end').unique().sort('window_end')
    return dates['window_end'].to_list()


def get_geometry_history(
    end_date: date,
    lookback: int = None
) -> pl.DataFrame:
    """
    Fetch geometry.structure history for dynamics computation.

    Returns DataFrame with columns from geometry.structure.
    """
    if lookback is None:
        lookback = get_lookback_window()

    geometry_path = get_path(GEOMETRY)
    if not geometry_path.exists():
        return pl.DataFrame()

    start_date = end_date - timedelta(days=lookback * 2)  # Buffer for sparse data

    df = pl.read_parquet(geometry_path)
    df = df.filter(
        (pl.col('window_end') >= start_date) &
        (pl.col('window_end') <= end_date)
    ).select([
        'window_end',
        'n_signals',
        'pca_variance_1',
        'pca_variance_2',
        'pca_variance_3',
        'pca_cumulative_3',
        'n_clusters',
        'total_dispersion',
        'mean_alignment',
        'system_coherence',
        'system_energy',
    ]).sort('window_end')

    return df


def get_displacement_history(
    end_date: date,
    lookback: int = None
) -> pl.DataFrame:
    """
    Fetch geometry.displacement history for dynamics computation.
    """
    if lookback is None:
        lookback = get_lookback_window()

    geometry_path = get_path(GEOMETRY)
    if not geometry_path.exists():
        return pl.DataFrame()

    start_date = end_date - timedelta(days=lookback * 2)

    df = pl.read_parquet(geometry_path)
    df = df.filter(
        (pl.col('window_end_to') >= start_date) &
        (pl.col('window_end_to') <= end_date)
    ).select([
        pl.col('window_end_to').alias('window_end'),
        'days_elapsed',
        'energy_total',
        'energy_63',
        'energy_126',
        'energy_252',
        'anchor_ratio',
        'barycenter_shift_mean',
        'dispersion_delta',
        'dispersion_velocity',
        'regime_conviction',
    ]).sort('window_end')

    return df


def get_signal_geometry(window_end: date) -> pl.DataFrame:
    """
    Fetch geometry.signals for a specific date.
    """
    signals_path = get_path(SIGNALS)
    if not signals_path.exists():
        return pl.DataFrame()

    df = pl.read_parquet(signals_path)
    df = df.filter(pl.col('window_end') == window_end).select([
        'signal_id',
        'barycenter',
        'timescale_dispersion',
        'timescale_alignment',
    ])

    return df


def get_cohort_membership() -> Dict[str, List[str]]:
    """Get cohort membership mapping from parquet."""
    members_path = get_path(COHORTS)
    if not members_path.exists():
        return {}

    df = pl.read_parquet(members_path)
    df = df.filter(pl.col('cohort_id').is_not_null()).sort(['cohort_id', 'signal_id'])

    membership = {}
    for row in df.iter_rows(named=True):
        cohort_id = row['cohort_id']
        if cohort_id not in membership:
            membership[cohort_id] = []
        membership[cohort_id].append(row['signal_id'])

    return membership


# =============================================================================
# STATE COMPUTATION
# =============================================================================

def compute_system_state(
    structure_history: pl.DataFrame,
    displacement_history: pl.DataFrame,
    state_date: date
) -> Dict[str, Any]:
    """
    Compute system-level state for a single date.

    Uses energy_dynamics, tension_dynamics, and phase_detector engines.
    """
    if len(structure_history) == 0 or len(displacement_history) == 0:
        return {}

    # Initialize engines
    energy_engine = EnergyDynamicsEngine()
    tension_engine = TensionDynamicsEngine()
    phase_engine = PhaseDetectorEngine()

    # Convert to pandas for engine compatibility (engines expect pandas Series)
    struct_pd = structure_history.to_pandas().set_index('window_end')
    disp_pd = displacement_history.to_pandas().set_index('window_end')

    # Prepare series
    energy_series = disp_pd['energy_total']
    dispersion_series = struct_pd['total_dispersion']
    alignment_series = struct_pd['mean_alignment']
    coherence_series = struct_pd['system_coherence']

    # Get current values
    current_disp = displacement_history.filter(pl.col('window_end') == state_date)
    current_struct = structure_history.filter(pl.col('window_end') == state_date)

    if len(current_disp) == 0 and len(current_struct) == 0:
        return {}

    # Energy dynamics
    energy_result = energy_engine.run(energy_series)

    # Tension dynamics
    tension_result = tension_engine.run(
        dispersion_series,
        alignment_series,
        coherence_series
    )

    # Get anchor_ratio and regime_conviction from current displacement
    if len(current_disp) > 0:
        anchor_ratio = current_disp['anchor_ratio'][0]
        regime_conviction = current_disp['regime_conviction'][0]
    else:
        anchor_ratio = 0.0
        regime_conviction = 0.0

    # Phase detection
    phase_result = phase_engine.run(
        energy_total=energy_result.energy_total,
        energy_zscore=energy_result.energy_zscore or 0.0,
        energy_trend=energy_result.energy_trend,
        dispersion_total=tension_result.dispersion_total,
        tension_state=tension_result.tension_state,
        alignment=tension_result.alignment_mean,
        anchor_ratio=anchor_ratio,
        regime_conviction=regime_conviction
    )

    # Get PCA concentration
    if len(current_struct) > 0:
        pca_concentration = current_struct['pca_cumulative_3'][0]
    else:
        pca_concentration = 0.0

    return {
        'state_time': state_date,
        'energy_total': energy_result.energy_total,
        'energy_ma5': energy_result.energy_ma5,
        'energy_ma20': energy_result.energy_ma20,
        'energy_acceleration': energy_result.energy_acceleration,
        'energy_zscore': energy_result.energy_zscore,
        'dispersion_total': tension_result.dispersion_total,
        'dispersion_velocity': tension_result.dispersion_velocity,
        'dispersion_acceleration': tension_result.dispersion_acceleration,
        'alignment_mean': tension_result.alignment_mean,
        'coherence': tension_result.coherence,
        'pca_concentration': pca_concentration,
        'regime_conviction': regime_conviction,
        'anchor_ratio': anchor_ratio,
        'phase_score': phase_result.phase_score,
        'phase_label': phase_result.phase_label,
        'is_regime_shift': bool(phase_result.is_regime_shift),
        'shift_confidence': phase_result.shift_confidence,
    }


def compute_signal_dynamics(
    state_date: date,
    prev_date: Optional[date]
) -> List[Dict[str, Any]]:
    """
    Compute signal-level dynamics for a single date.
    """
    current = get_signal_geometry(state_date)

    if len(current) == 0:
        return []

    results = []

    if prev_date:
        previous = get_signal_geometry(prev_date)
        prev_dict = {row['signal_id']: row for row in previous.iter_rows(named=True)}
    else:
        prev_dict = {}

    # Compute system centroid
    barycenters = []
    for row in current.iter_rows(named=True):
        bc = row['barycenter']
        if bc is not None and len(bc) > 0:
            barycenters.append(np.array(bc))

    if barycenters:
        system_centroid = np.mean(barycenters, axis=0)
    else:
        system_centroid = None

    for row in current.iter_rows(named=True):
        ind_id = row['signal_id']
        bc = row['barycenter']
        disp = row['timescale_dispersion']
        align = row['timescale_alignment']

        result = {
            'signal_id': ind_id,
            'state_time': state_date,
            'dispersion': disp,
            'alignment': align,
            'barycenter_shift': 0.0,
            'barycenter_velocity': 0.0,
            'barycenter_acceleration': 0.0,
            'dispersion_delta': 0.0,
            'distance_to_centroid': 0.0,
            'cluster_id': 0,
            'cluster_changed': False,
            'leads_system': False,
            'lags_system': False,
            'lead_lag_days': 0,
        }

        # Compute shift from previous
        if ind_id in prev_dict:
            prev_row = prev_dict[ind_id]
            prev_bc = prev_row['barycenter']

            if bc is not None and prev_bc is not None and len(bc) > 0 and len(prev_bc) > 0:
                from scipy.spatial.distance import euclidean
                result['barycenter_shift'] = euclidean(np.array(bc), np.array(prev_bc))

            if prev_row['timescale_dispersion'] is not None:
                result['dispersion_delta'] = disp - prev_row['timescale_dispersion']

        # Distance to system centroid
        if system_centroid is not None and bc is not None and len(bc) > 0:
            from scipy.spatial.distance import euclidean
            result['distance_to_centroid'] = euclidean(np.array(bc), system_centroid)

        results.append(result)

    return results


def compute_cross_cohort_transfers(
    state_date: date,
    lookback: int = 30
) -> List[Dict[str, Any]]:
    """
    Compute cross-cohort transfer metrics.
    """
    # Get cohort membership
    membership = get_cohort_membership()

    if len(membership) < 2:
        return []

    # Get energy history by cohort
    start_date = state_date - timedelta(days=lookback * 2)

    # Get displacement data
    geometry_path = get_path(GEOMETRY)
    if not geometry_path.exists():
        return []

    disp_df = pl.read_parquet(geometry_path)
    disp_df = disp_df.filter(
        (pl.col('window_end_to') >= start_date) &
        (pl.col('window_end_to') <= state_date)
    ).select([
        pl.col('window_end_to').alias('window_end'),
        'energy_total',
    ]).sort('window_end')

    if len(disp_df) == 0 or len(disp_df) < 10:
        return []

    # For simplified version, detect transfers between cohorts
    # using system-level metrics as proxy
    # Full implementation would use cohort-specific aggregates

    results = []
    cohorts = list(membership.keys())
    transfer_engine = TransferDetectorEngine()

    # Convert to pandas for engine compatibility
    disp_pd = disp_df.to_pandas().set_index('window_end')
    energy_series = disp_pd['energy_total']

    for i, cohort_a in enumerate(cohorts):
        for cohort_b in cohorts[i+1:]:
            # Simplified: use same series for demo
            # Real implementation: aggregate signal-level metrics per cohort
            result = transfer_engine.run(
                energy_series,
                energy_series,  # Would be cohort_b's series
                cohort_a,
                cohort_b
            )

            results.append({
                'state_time': state_date,
                'cohort_from': result.cohort_from,
                'cohort_to': result.cohort_to,
                'transfer_strength': result.transfer_strength,
                'transfer_lag': result.transfer_lag,
                'transfer_direction': result.transfer_direction,
                'granger_fstat': result.granger_fstat,
                'granger_pvalue': result.granger_pvalue,
                'te_net': None,  # Would come from TransferEntropyEngine
                'correlation': result.correlation,
                'correlation_lag': result.correlation_lag,
                'is_significant': result.is_significant,
                'transfer_type': result.transfer_type,
            })

    return results


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_state_snapshot(
    state_date: date,
    prev_date: Optional[date] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute state for a single snapshot date.

    Returns dict with results to store.
    """
    results = {'system': None, 'signals': [], 'transfers': []}

    # Get history
    structure_history = get_geometry_history(state_date)
    displacement_history = get_displacement_history(state_date)

    if len(structure_history) == 0 or len(displacement_history) == 0:
        if verbose:
            logger.warning(f"  {state_date}: No geometry data")
        return results

    # 1. System state
    system_state = compute_system_state(structure_history, displacement_history, state_date)
    if system_state:
        results['system'] = system_state

    # 2. Signal dynamics
    signal_dynamics = compute_signal_dynamics(state_date, prev_date)
    if signal_dynamics:
        results['signals'] = signal_dynamics

    # 3. Cross-cohort transfers
    transfers = compute_cross_cohort_transfers(state_date)
    if transfers:
        results['transfers'] = transfers

    if verbose and system_state:
        phase = system_state.get('phase_label', 'unknown')
        shift = system_state.get('is_regime_shift', False)
        logger.info(f"  {state_date}: phase={phase}, shift={shift}, signals={len(signal_dynamics)}")

    return results


def run_state_range(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    stride: int = None,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Run state computation for a date range.
    """
    if stride is None:
        stride = get_default_stride()

    ensure_directory()

    # Get available dates
    available_dates = get_available_dates()

    if not available_dates:
        logger.error("No geometry data found. Run geometry runner first.")
        return {'system': 0, 'signals': 0, 'transfers': 0}

    # Filter to range
    if start_date:
        available_dates = [d for d in available_dates if d >= start_date]
    if end_date:
        available_dates = [d for d in available_dates if d <= end_date]

    if not available_dates:
        logger.error(f"No data in range {start_date} to {end_date}")
        return {'system': 0, 'signals': 0, 'transfers': 0}

    # Apply stride
    if stride > 1:
        strided_dates = available_dates[::stride]
    else:
        strided_dates = available_dates

    if verbose:
        logger.info("=" * 80)
        logger.info("PRISM STATE RUNNER - TEMPORAL DYNAMICS")
        logger.info("=" * 80)
        logger.info(f"Storage: Parquet files")
        logger.info(f"Date range: {strided_dates[0]} to {strided_dates[-1]}")
        logger.info(f"Snapshots: {len(strided_dates)} (stride={stride})")
        logger.info("")

    totals = {'system': 0, 'signals': 0, 'transfers': 0}
    prev_date = None
    computed_at = datetime.now()

    # Collect all results
    system_rows = []
    signal_rows = []
    transfer_rows = []

    for i, state_date in enumerate(strided_dates):
        results = run_state_snapshot(state_date, prev_date, verbose)

        if results['system']:
            results['system']['computed_at'] = computed_at
            system_rows.append(results['system'])
            totals['system'] += 1

        for ind in results['signals']:
            ind['computed_at'] = computed_at
            signal_rows.append(ind)
        totals['signals'] += len(results['signals'])

        for tr in results['transfers']:
            tr['computed_at'] = computed_at
            transfer_rows.append(tr)
        totals['transfers'] += len(results['transfers'])

        prev_date = state_date

        # Periodic write (every 50 dates)
        if (i + 1) % 50 == 0:
            if system_rows:
                df = pl.DataFrame(system_rows)
                upsert_parquet(df, get_path(STATE), SYSTEM_KEY_COLS)
                system_rows = []
            if signal_rows:
                df = pl.DataFrame(signal_rows)
                upsert_parquet(df, get_path(STATE), INDICATOR_DYNAMICS_KEY_COLS)
                signal_rows = []
            if transfer_rows:
                df = pl.DataFrame(transfer_rows)
                upsert_parquet(df, get_path(STATE), TRANSFERS_KEY_COLS)
                transfer_rows = []

    # Final write
    if system_rows:
        df = pl.DataFrame(system_rows)
        upsert_parquet(df, get_path(STATE), SYSTEM_KEY_COLS)
    if signal_rows:
        df = pl.DataFrame(signal_rows)
        upsert_parquet(df, get_path(STATE), INDICATOR_DYNAMICS_KEY_COLS)
    if transfer_rows:
        df = pl.DataFrame(transfer_rows)
        upsert_parquet(df, get_path(STATE), TRANSFERS_KEY_COLS)

    if verbose:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"COMPLETE: {totals['system']} system states, "
                    f"{totals['signals']} signal dynamics, "
                    f"{totals['transfers']} transfers")
        logger.info("=" * 80)

    return totals


# =============================================================================
# PARALLEL WORKER
# =============================================================================

def process_state_parallel(assignment: WorkerAssignment) -> Dict[str, Any]:
    """Worker function for parallel state computation."""
    dates = assignment.items
    temp_path = assignment.temp_path
    config = assignment.config

    totals = {'system': 0, 'signals': 0, 'transfers': 0}
    prev_date = None
    computed_at = datetime.now()

    # Collect all results
    system_rows = []
    signal_rows = []
    transfer_rows = []

    try:
        for state_date in dates:
            # Get history
            structure_history = get_geometry_history(state_date)
            displacement_history = get_displacement_history(state_date)

            if len(structure_history) == 0 or len(displacement_history) == 0:
                continue

            # Compute system state
            system_state = compute_system_state(structure_history, displacement_history, state_date)
            if system_state:
                system_state['computed_at'] = computed_at
                system_rows.append(system_state)
                totals['system'] += 1

            # Compute signal dynamics
            signal_dynamics = compute_signal_dynamics(state_date, prev_date)
            for ind in signal_dynamics:
                ind['computed_at'] = computed_at
                signal_rows.append(ind)
            totals['signals'] += len(signal_dynamics)

            # Cross-cohort transfers (less frequently)
            if totals['system'] % 5 == 0:
                transfers = compute_cross_cohort_transfers(state_date)
                for tr in transfers:
                    tr['computed_at'] = computed_at
                    transfer_rows.append(tr)
                totals['transfers'] += len(transfers)

            prev_date = state_date

        # Write all results to temp parquet
        # We write a combined dataframe with a 'table_type' column to distinguish
        all_rows = []

        for row in system_rows:
            row['_table'] = 'system'
            all_rows.append(row)

        for row in signal_rows:
            row['_table'] = 'signal_dynamics'
            all_rows.append(row)

        for row in transfer_rows:
            row['_table'] = 'transfers'
            all_rows.append(row)

        if all_rows:
            # Write combined data to temp path
            df = pl.DataFrame(all_rows, infer_schema_length=None)
            df.write_parquet(temp_path)

    except Exception as e:
        logger.error(f"Worker error: {e}")
        raise

    return totals


def merge_state_results(temp_paths: List[Path], verbose: bool = True) -> Dict[str, int]:
    """
    Merge worker temp files into main state parquet files.
    """
    totals = {'system': 0, 'signal_dynamics': 0, 'transfers': 0}

    # Read all temp files
    all_dfs = []
    for path in temp_paths:
        if path.exists():
            try:
                df = pl.read_parquet(path)
                if len(df) > 0:
                    all_dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to read temp file {path}: {e}")

    if not all_dfs:
        return totals

    combined = pl.concat(all_dfs, how='diagonal_relaxed')

    # Split by table type and write to respective parquet files
    for table_name, key_cols in [
        ('system', SYSTEM_KEY_COLS),
        ('signal_dynamics', INDICATOR_DYNAMICS_KEY_COLS),
        ('transfers', TRANSFERS_KEY_COLS),
    ]:
        table_df = combined.filter(pl.col('_table') == table_name).drop('_table')

        if len(table_df) > 0:
            target_path = get_path(STATE)
            upsert_parquet(table_df, target_path, key_cols)
            totals[table_name] = len(table_df)

            if verbose:
                logger.info(f"  Merged {len(table_df):,} rows to state.{table_name}")

    # Cleanup temp files
    for path in temp_paths:
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass

    return totals


# =============================================================================
# V2 ARCHITECTURE: STATE TRAJECTORY FROM GEOMETRY SNAPSHOTS
# =============================================================================

def load_geometry_snapshots_v2() -> Dict[str, List[GeometrySnapshot]]:
    """
    Load V2 GeometrySnapshots from parquet storage, grouped by entity.

    Returns:
        Dict mapping entity_id -> List of GeometrySnapshot objects sorted by timestamp
    """
    geom_path = get_path(GEOMETRY)
    if not geom_path.exists():
        logger.warning(f"No V2 geometry snapshots at {geom_path}. Run geometry --v2 first.")
        return {}

    df = pl.read_parquet(geom_path).sort(['entity_id', 'timestamp'])

    # Check if coupling columns exist (signal_a, signal_b, coupling)
    has_coupling_columns = all(col in df.columns for col in ['signal_a', 'signal_b', 'coupling'])

    # Group by entity_id
    snapshots_by_entity = {}

    for row in df.iter_rows(named=True):
        entity_id = row.get('entity_id', 'unknown')
        ts = row['timestamp']
        signal_ids = row.get('signal_ids', '')
        signal_ids = signal_ids.split(',') if signal_ids else []
        n_signals = row.get('n_signals', len(signal_ids))

        # Use identity matrix when no pairwise coupling data
        if n_signals > 0:
            coupling_matrix = np.eye(n_signals)
        else:
            coupling_matrix = np.array([[]])

        # Handle both n_modes and mode_count column names
        n_modes = row.get('n_modes', row.get('mode_count', 1))
        mode_labels = np.zeros(n_signals, dtype=int) if n_signals > 0 else np.array([])
        # Default coherence to 1.0 if not available
        mean_coherence = row.get('mean_mode_coherence', 1.0)
        mode_coherence = np.array([mean_coherence]) if n_modes > 0 else np.array([])

        snapshot = GeometrySnapshot(
            timestamp=float(ts) if isinstance(ts, (int, float)) else ts.timestamp() if hasattr(ts, 'timestamp') else 0.0,
            coupling_matrix=coupling_matrix,
            divergence=row.get('divergence', 0.0),
            mode_labels=mode_labels,
            mode_coherence=mode_coherence,
            signal_ids=signal_ids,
        )

        if entity_id not in snapshots_by_entity:
            snapshots_by_entity[entity_id] = []
        snapshots_by_entity[entity_id].append(snapshot)

    total_snapshots = sum(len(snaps) for snaps in snapshots_by_entity.values())
    logger.info(f"Loaded {total_snapshots} GeometrySnapshots from {geom_path} ({len(snapshots_by_entity)} entities)")
    return snapshots_by_entity


def state_trajectory_to_rows(
    trajectory: StateTrajectory,
    computed_at: datetime = None,
) -> List[Dict]:
    """
    Convert StateTrajectory to row format for parquet storage.

    Args:
        trajectory: StateTrajectory object
        computed_at: Computation timestamp

    Returns:
        List of row dictionaries
    """
    if computed_at is None:
        computed_at = datetime.now()

    rows = []
    n_timestamps = len(trajectory.timestamps)

    for i in range(n_timestamps):
        # Compute scalar metrics at this timestamp
        speed = trajectory.speed[i] if hasattr(trajectory, 'speed') else np.linalg.norm(trajectory.velocity[i])
        accel_mag = trajectory.acceleration_magnitude[i] if hasattr(trajectory, 'acceleration_magnitude') else np.linalg.norm(trajectory.acceleration[i])

        rows.append({
            'timestamp': trajectory.timestamps[i],
            'speed': float(speed),
            'acceleration_magnitude': float(accel_mag),
            'position_dim': int(trajectory.position.shape[1]) if len(trajectory.position.shape) > 1 else 1,
            'computed_at': computed_at,
        })

    return rows


def run_v2_state(
    verbose: bool = True,
) -> Dict:
    """
    Run V2 state trajectory computation PER ENTITY.

    Loads geometry metrics directly from parquet, computes state trajectory
    (position, velocity, acceleration) for each entity, saves to parquet.

    Args:
        verbose: Print progress

    Returns:
        Dict with processing statistics
    """
    computed_at = datetime.now()

    # Load geometry directly from parquet
    geom_path = get_path(GEOMETRY)
    if not geom_path.exists():
        logger.warning(f"No geometry data at {geom_path}. Run geometry first.")
        return {'snapshots': 0}

    geom_df = pl.read_parquet(geom_path).sort(['entity_id', 'timestamp'])

    # Feature columns to use for position vector
    feature_cols = [
        'pca_var_1', 'pca_var_2', 'clustering_silhouette',
        'mst_total_weight', 'lof_mean', 'distance_mean'
    ]
    # Filter to columns that exist
    feature_cols = [c for c in feature_cols if c in geom_df.columns]

    if not feature_cols:
        logger.warning("No geometry feature columns found.")
        return {'snapshots': 0}

    n_entities = geom_df['entity_id'].n_unique()
    total_snapshots = len(geom_df)

    if verbose:
        logger.info("=" * 80)
        logger.info("V2 ARCHITECTURE: State Trajectory from Geometry (Per-Entity)")
        logger.info("=" * 80)
        logger.info(f"  Entities: {n_entities}")
        logger.info(f"  Total snapshots: {total_snapshots}")
        logger.info(f"  Feature columns: {feature_cols}")

    # Compute state trajectory for each entity
    all_rows = []
    total_failure_timestamps = 0
    entities_processed = 0

    for entity_id in geom_df['entity_id'].unique().sort().to_list():
        entity_df = geom_df.filter(pl.col('entity_id') == entity_id).sort('timestamp')

        if len(entity_df) < 2:
            continue

        timestamps = entity_df['timestamp'].to_numpy()
        positions = entity_df.select(feature_cols).to_numpy()

        # Compute velocity and acceleration using numpy gradient
        velocity = np.gradient(positions, timestamps, axis=0)
        acceleration = np.gradient(velocity, timestamps, axis=0)

        # Compute speed (magnitude of velocity)
        speed = np.linalg.norm(velocity, axis=1)
        accel_mag = np.linalg.norm(acceleration, axis=1)

        # Compute curvature (change in direction)
        curvature = np.zeros(len(timestamps))
        for i in range(1, len(timestamps) - 1):
            v1 = velocity[i - 1]
            v2 = velocity[i]
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            if norm_v1 > 1e-10 and norm_v2 > 1e-10:
                cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                curvature[i] = np.arccos(cos_angle)

        # Detect failure signatures (high velocity + positive acceleration)
        mean_speed = np.mean(speed)
        failure_mask = (speed > mean_speed) & (accel_mag > np.mean(accel_mag))
        n_failure = int(np.sum(failure_mask))
        total_failure_timestamps += n_failure

        # Summary metrics
        mean_velocity = float(np.nanmean(speed))
        mean_acceleration = float(np.nanmean(accel_mag))

        # Get mode_id for this entity (from geometry)
        mode_ids = entity_df['mode_id'].to_numpy() if 'mode_id' in entity_df.columns else np.zeros(len(timestamps), dtype=int)

        # Create rows for each timestamp
        for i in range(len(timestamps)):
            all_rows.append({
                'entity_id': entity_id,
                'timestamp': float(timestamps[i]),
                'speed': float(speed[i]),
                'acceleration_magnitude': float(accel_mag[i]),
                'position_dim': len(feature_cols),
                'is_failure_signature': bool(failure_mask[i]),
                'curvature': float(curvature[i]),
                'mean_velocity': mean_velocity,
                'mean_acceleration': mean_acceleration,
                'mode_id': int(mode_ids[i]),
                'computed_at': computed_at,
            })

        entities_processed += 1

    if verbose:
        logger.info(f"  Trajectories computed: {entities_processed} entities")
        logger.info(f"  Total state rows: {len(all_rows)}")
        logger.info(f"  Failure signature timestamps: {total_failure_timestamps}")

    if not all_rows:
        logger.warning("No state rows generated. Check geometry data.")
        return {'snapshots': total_snapshots, 'saved_rows': 0}

    if verbose:
        logger.info(f"\n  Saving {len(all_rows)} state trajectory rows...")

    # Save trajectory to parquet
    df = pl.DataFrame(all_rows, infer_schema_length=None)

    # Add mode_transition and mode_delta columns
    df = df.sort(['entity_id', 'timestamp'])
    df = df.with_columns([
        pl.col('mode_id').shift(1).over('entity_id').alias('_mode_prev'),
    ])
    df = df.with_columns([
        # Did mode change from previous timestamp?
        (pl.col('mode_id') != pl.col('_mode_prev')).fill_null(False).alias('mode_transition'),
        # Direction of change: +1 = degrading, -1 = recovering, 0 = stable
        (pl.col('mode_id') - pl.col('_mode_prev')).fill_null(0).cast(pl.Int64).alias('mode_delta'),
    ])
    df = df.drop('_mode_prev')

    # Count mode transitions
    n_transitions = df.filter(pl.col('mode_transition')).height
    if verbose:
        logger.info(f"  Mode transitions detected: {n_transitions}")

    state_path = get_path(STATE)
    df.write_parquet(state_path)

    if verbose:
        logger.info(f"  Saved: {state_path}")

    return {
        'snapshots': total_snapshots,
        'entities': entities_processed,
        'failure_timestamps': total_failure_timestamps,
        'saved_rows': len(all_rows),
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PRISM State Runner - Temporal dynamics from geometry',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output: data/state.parquet

Examples:
  python -m prism.entry_points.state              # Production run
  python -m prism.entry_points.state --adaptive   # Auto-detect window
  python -m prism.entry_points.state --force      # Force recompute
  python -m prism.entry_points.state --testing    # Enable test mode
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

    # Always use V2 architecture
    ensure_directory()

    logger.info("=" * 80)
    logger.info("PRISM STATE - Temporal Dynamics")
    logger.info("=" * 80)
    logger.info(f"Source: data/geometry.parquet")
    logger.info(f"Destination: data/state.parquet")

    result = run_v2_state(verbose=True)

    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  Snapshots processed: {result.get('snapshots', 0)}")
    logger.info(f"  Timestamps: {result.get('timestamps', 0)}")
    logger.info(f"  Failure signatures: {result.get('failure_timestamps', 0)}")
    logger.info(f"  Acceleration events: {result.get('events', 0)}")
    logger.info(f"  Saved rows: {result.get('saved_rows', 0)}")
    return 0




if __name__ == '__main__':
    exit(main())
