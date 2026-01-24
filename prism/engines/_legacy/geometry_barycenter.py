"""
PRISM Geometry Engine - Barycenter Edition
===========================================

Computes the structural manifold using conviction-weighted barycenters.

Core Concept:
    - Short windows (63d) are scouts - noisy, early warning
    - Medium windows (126d) are bridges - confirmation
    - Long windows (252d) are anchors - structural truth

Weights: 63d (1x), 126d (2x), 252d (4x)

When anchors move, it's a regime shift.
When only scouts move, it's noise.

Output Tables:
    geometry.signals   - Per signal: barycenter, dispersion, alignment
    geometry.pairs        - Per pair: weighted distance, correlation
    geometry.structure    - Per snapshot: PCA, clusters, system dispersion
    geometry.displacement - Per transition: kinetic energy, conviction

Usage:
    from prism.engines.geometry_barycenter import GeometryEngine

    engine = GeometryEngine(stride=21)
    engine.run_range(date(2020, 1, 1), date(2024, 12, 31))

    # Or single snapshot
    engine.run_snapshot(date(2024, 1, 15))
"""

import logging
import numpy as np
import pandas as pd
import polars as pl
from datetime import date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from scipy.spatial.distance import euclidean, pdist
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings

from prism.db.parquet_store import (
    get_path,
    get_data_root,
    file_exists,
    OBSERVATIONS,
    SIGNALS,
    GEOMETRY,
    STATE,
    COHORTS,
)
from prism.db.polars_io import read_parquet, upsert_parquet

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class GeometryEngine:
    """
    PRISM Geometry Engine - Barycenter-based

    Computes the conviction-weighted structural manifold.

    The key insight: not all timeframes are equal.
    A 252-day window shifting is 4x more significant than a 63-day window shifting.
    This is encoded directly into the geometry via weighting.
    """

    def __init__(self, stride: int = 21, n_clusters: int = 5, weights: dict = None):
        """
        Initialize the geometry engine.

        Args:
            stride: Days between snapshots (default 21 = monthly)
            n_clusters: Number of clusters for structural analysis
            weights: Optional window weights dict. Loads from config if not provided.
        """
        self.stride = stride
        self.n_clusters = n_clusters

        # Conviction weights: longer windows = more weight
        # Load from config or use provided weights
        if weights is not None:
            self.weights = weights
        else:
            self.weights = self._load_weights()
        self.total_weight = sum(self.weights.values())

        # Feature columns (discovered dynamically)
        self.feature_cols = None

        # Paths to parquet files
        self._vectors_path = get_path(SIGNALS)
        self._geometry_signals_path = get_path(GEOMETRY)
        self._geometry_pairs_path = get_path(GEOMETRY)
        self._geometry_structure_path = get_path(GEOMETRY)
        self._geometry_displacement_path = get_path(GEOMETRY)

    def _load_weights(self) -> dict:
        """Load barycenter weights from config. Fails if not configured."""
        try:
            from prism.utils.stride import get_barycenter_weights
            weights = get_barycenter_weights()
            if weights:
                return weights
        except Exception as e:
            raise RuntimeError(f"Failed to load barycenter weights: {e}")

        raise RuntimeError(
            "No barycenter weights configured in config/stride.yaml. "
            "Configure domain-specific window weights before running."
        )

    def _discover_feature_columns(self) -> List[str]:
        """Dynamically discover numeric feature columns from vectors table."""
        if self.feature_cols is not None:
            return self.feature_cols

        try:
            if not self._vectors_path.exists():
                logger.warning("Vectors parquet file not found")
                return []

            # Read schema from parquet file using Polars lazy scan
            lf = pl.scan_parquet(self._vectors_path)
            schema = lf.schema

            metadata_cols = {
                'signal_id', 'window_days', 'window_end', 'window_start',
                'cohort', 'n_observations', 'computed_at', 'run_id',
                'obs_date', 'target_obs', 'engine', 'metric_name', 'metric_value'
            }

            # Get numeric columns that aren't metadata
            numeric_types = {pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8}
            self.feature_cols = [
                col_name
                for col_name, col_type in schema.items()
                if col_name not in metadata_cols
                and col_type in numeric_types
            ]

            logger.info(f"Discovered {len(self.feature_cols)} feature columns")
            return self.feature_cols

        except Exception as e:
            logger.error(f"Failed to discover feature columns: {e}")
            return []

    def get_vector_matrix(self, window_end: date) -> pd.DataFrame:
        """
        Fetch behavioral vectors for all signals at a given snapshot.

        Returns DataFrame with columns: signal_id, window_days, vector (np.array)
        """
        feature_cols = self._discover_feature_columns()
        if not feature_cols:
            return pd.DataFrame()

        if not self._vectors_path.exists():
            return pd.DataFrame()

        # Read and filter using Polars - LAZY with pushdown filter
        df_pl = (
            pl.scan_parquet(self._vectors_path)
            .filter(pl.col('window_end') == window_end)
            .collect()
        )

        if len(df_pl) == 0:
            return pd.DataFrame()

        # Select required columns plus feature columns
        available_features = [c for c in feature_cols if c in df_pl.columns]
        if not available_features:
            return pd.DataFrame()

        select_cols = ['signal_id', 'window_days'] + available_features
        df_pl = df_pl.select([c for c in select_cols if c in df_pl.columns])

        # Sort by signal_id and window_days
        df_pl = df_pl.sort(['signal_id', 'window_days'])

        # Convert to pandas for scipy/sklearn compatibility
        df = df_pl.to_pandas()

        if df.empty:
            return df

        # Combine feature columns into single vector array
        # Handle NaN by filling with 0 (or could use column mean)
        feature_matrix = df[available_features].fillna(0).values
        df['vector'] = [np.array(row) for row in feature_matrix]

        return df[['signal_id', 'window_days', 'vector']]

    def compute_barycenter(self, vectors: Dict[int, np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float]]:
        """
        Calculate conviction-weighted barycenter and tension metrics.

        Args:
            vectors: Dict mapping window_days -> vector array

        Returns:
            (barycenter, dispersion, alignment)
            - barycenter: weighted center of mass
            - dispersion: distance between shortest and longest window (tension)
            - alignment: coherence of all windows (1 = perfect agreement)
        """
        # Need all configured windows
        required = set(self.weights.keys())
        available = set(vectors.keys()) & required
        if len(available) < 2:
            return None, None, None

        # Get sorted windows for consistent processing
        sorted_windows = sorted(available)
        first_window = sorted_windows[0]

        # 1. Weighted Barycenter (Center of Mass)
        weighted_sum = np.zeros_like(vectors[first_window])
        active_weight = 0.0
        for win in sorted_windows:
            if win in self.weights:
                weighted_sum += vectors[win] * self.weights[win]
                active_weight += self.weights[win]

        if active_weight == 0:
            return None, None, None

        barycenter = weighted_sum / active_weight

        # 2. Timescale Dispersion (Tension between scouts and anchors)
        shortest = min(sorted_windows)
        longest = max(sorted_windows)
        dispersion = euclidean(vectors[shortest], vectors[longest])

        # 3. Timescale Alignment (How coherent are all windows?)
        # Low variance in distances to barycenter = high alignment
        distances = [euclidean(vectors[w], barycenter) for w in sorted_windows]
        variance = np.var(distances)
        alignment = 1.0 / (1.0 + variance)

        return barycenter, float(dispersion), float(alignment)

    def run_snapshot(self, window_end: date) -> Dict[str, Any]:
        """
        Compute full geometry manifold for a single snapshot.

        Returns summary statistics for logging.
        """
        df = self.get_vector_matrix(window_end)
        if df.empty:
            logger.warning(f"No vectors found for {window_end}")
            return {'status': 'no_data'}

        # Group vectors by signal
        signal_groups: Dict[str, Dict[int, np.ndarray]] = {}
        for _, row in df.iterrows():
            iid = row['signal_id']
            if iid not in signal_groups:
                signal_groups[iid] = {}
            signal_groups[iid][row['window_days']] = row['vector']

        # =====================================================================
        # PHASE 1: INDICATOR GEOMETRY
        # =====================================================================
        signal_data: Dict[str, Dict] = {}
        barycenters: List[np.ndarray] = []
        signal_records = []

        for iid, windows in signal_groups.items():
            barycenter, dispersion, alignment = self.compute_barycenter(windows)

            if barycenter is None:
                continue

            signal_data[iid] = {
                'barycenter': barycenter,
                'dispersion': dispersion,
                'alignment': alignment,
                'vectors': windows
            }
            barycenters.append(barycenter)

            # Store individual window vectors for later analysis
            v63 = windows.get(63, np.zeros_like(barycenter)).tolist()
            v126 = windows.get(126, np.zeros_like(barycenter)).tolist()
            v252 = windows.get(252, np.zeros_like(barycenter)).tolist()

            signal_records.append({
                'signal_id': iid,
                'window_end': window_end,
                'barycenter': barycenter.tolist(),
                'timescale_dispersion': dispersion,
                'timescale_alignment': alignment,
                'vector_63': v63,
                'vector_126': v126,
                'vector_252': v252,
            })

        n_signals = len(signal_data)
        if n_signals < 3:
            logger.warning(f"Only {n_signals} complete signals at {window_end}")
            return {'status': 'insufficient_data', 'n_signals': n_signals}

        # Write signal geometry to parquet
        if signal_records:
            signals_df = pl.DataFrame(signal_records)
            upsert_parquet(
                signals_df,
                self._geometry_signals_path,
                key_cols=['signal_id', 'window_end']
            )

        # =====================================================================
        # PHASE 2: PAIRWISE GEOMETRY
        # =====================================================================
        ids = list(signal_data.keys())
        n_pairs = 0
        pair_records = []

        for i, id_a in enumerate(ids):
            for id_b in ids[i+1:]:
                bc_a = signal_data[id_a]['barycenter']
                bc_b = signal_data[id_b]['barycenter']

                # Distance between weighted centers
                bc_distance = euclidean(bc_a, bc_b)

                # Weighted correlation (using barycenters as representative vectors)
                if np.std(bc_a) > 0 and np.std(bc_b) > 0:
                    corr = np.corrcoef(bc_a, bc_b)[0, 1]
                else:
                    corr = 0.0

                # Co-movement: do their dispersions move together?
                co_movement = self._compute_co_movement(
                    signal_data[id_a]['vectors'],
                    signal_data[id_b]['vectors']
                )

                pair_records.append({
                    'signal_a': id_a,
                    'signal_b': id_b,
                    'window_end': window_end,
                    'barycenter_distance': float(bc_distance),
                    'correlation_weighted': float(corr) if not np.isnan(corr) else 0.0,
                    'co_movement': float(co_movement),
                })
                n_pairs += 1

        # Write pairs to parquet
        if pair_records:
            pairs_df = pl.DataFrame(pair_records)
            upsert_parquet(
                pairs_df,
                self._geometry_pairs_path,
                key_cols=['signal_a', 'signal_b', 'window_end']
            )

        # =====================================================================
        # PHASE 3: SYSTEM STRUCTURE
        # =====================================================================
        matrix = np.array(barycenters)

        # PCA on weighted barycenters
        pca = PCA(n_components=min(5, n_signals - 1))
        try:
            pca.fit(matrix)
            pca_var = pca.explained_variance_ratio_
        except Exception:
            pca_var = [0, 0, 0]

        # Clustering
        n_clust = min(self.n_clusters, n_signals - 1)
        try:
            kmeans = KMeans(n_clusters=n_clust, n_init=10, random_state=42)
            labels = kmeans.fit_predict(matrix)
            cluster_sizes = [int(np.sum(labels == i)) for i in range(n_clust)]
        except Exception:
            cluster_sizes = [n_signals]
            n_clust = 1

        # System metrics
        total_dispersion = float(np.mean([d['dispersion'] for d in signal_data.values()]))
        mean_alignment = float(np.mean([d['alignment'] for d in signal_data.values()]))

        # System coherence: inverse of average pairwise distance
        if len(barycenters) > 1:
            pairwise_dists = pdist(matrix)
            system_coherence = 1.0 / (1.0 + np.mean(pairwise_dists))
        else:
            system_coherence = 1.0

        # System energy: total displacement energy from latest transition
        # Query the most recent displacement ending at this window_end
        system_energy = 0.0
        if self._geometry_displacement_path.exists():
            # LAZY with pushdown filter - only load matching rows
            disp_filtered = (
                pl.scan_parquet(self._geometry_displacement_path)
                .filter(pl.col('window_end_to') == window_end)
                .collect()
            )
            if len(disp_filtered) > 0:
                # Get the most recent one
                energy_row = disp_filtered.sort('window_end_to', descending=True).head(1)
                if len(energy_row) > 0:
                    e63 = energy_row['energy_63'][0] if 'energy_63' in energy_row.columns else 0
                    e126 = energy_row['energy_126'][0] if 'energy_126' in energy_row.columns else 0
                    e252 = energy_row['energy_252'][0] if 'energy_252' in energy_row.columns else 0
                    system_energy = float(e63 + e126 + e252)

        structure_record = {
            'window_end': window_end,
            'n_signals': n_signals,
            'pca_variance_1': float(pca_var[0]) if len(pca_var) > 0 else 0.0,
            'pca_variance_2': float(pca_var[1]) if len(pca_var) > 1 else 0.0,
            'pca_variance_3': float(pca_var[2]) if len(pca_var) > 2 else 0.0,
            'pca_cumulative_3': float(sum(pca_var[:3])) if len(pca_var) >= 3 else float(sum(pca_var)),
            'n_clusters': n_clust,
            'cluster_sizes': cluster_sizes,
            'total_dispersion': total_dispersion,
            'mean_alignment': mean_alignment,
            'system_coherence': system_coherence,
            'system_energy': system_energy,
        }

        structure_df = pl.DataFrame([structure_record])
        upsert_parquet(
            structure_df,
            self._geometry_structure_path,
            key_cols=['window_end']
        )

        # =====================================================================
        # PHASE 4: DISPLACEMENT (Temporal Physics)
        # =====================================================================
        self._compute_displacement(window_end, signal_groups)

        return {
            'status': 'success',
            'window_end': window_end,
            'n_signals': n_signals,
            'n_pairs': n_pairs,
            'total_dispersion': total_dispersion,
            'mean_alignment': mean_alignment,
            'pca_var_1': float(pca_var[0]) if len(pca_var) > 0 else 0
        }

    def _compute_co_movement(self, vectors_a: Dict[int, np.ndarray],
                             vectors_b: Dict[int, np.ndarray]) -> float:
        """
        Compute co-movement between two signals across timescales.

        High co-movement = their window vectors move in similar patterns.
        """
        correlations = []
        for win in [63, 126, 252]:
            if win in vectors_a and win in vectors_b:
                va, vb = vectors_a[win], vectors_b[win]
                if np.std(va) > 0 and np.std(vb) > 0:
                    corr = np.corrcoef(va, vb)[0, 1]
                    if not np.isnan(corr):
                        # Weight by window importance
                        correlations.append(corr * self.weights[win])

        if correlations:
            return sum(correlations) / self.total_weight
        return 0.0

    def _compute_displacement(self, t_now: date, current_vectors: Dict[str, Dict[int, np.ndarray]]):
        """
        Compute the physics of movement between this snapshot and the previous one.

        Key metrics:
        - energy_by_window: How much did each timescale contribute to movement?
        - anchor_ratio: Did anchors move more than scouts? (>1 = structural shift)
        - regime_conviction: High energy into tight structure = confirmed regime change
        """
        # Find previous snapshot from geometry.signals
        if not self._geometry_signals_path.exists():
            return

        # LAZY - only get unique dates before t_now
        prev_dates = (
            pl.scan_parquet(self._geometry_signals_path)
            .filter(pl.col('window_end') < t_now)
            .select('window_end')
            .unique()
            .collect()
        )

        if len(prev_dates) == 0:
            return

        t_prev = prev_dates.sort('window_end', descending=True)['window_end'][0]

        # Convert both to pandas Timestamp for consistent subtraction
        t_now_ts = pd.Timestamp(t_now)
        t_prev_ts = pd.Timestamp(t_prev)
        days_elapsed = (t_now_ts - t_prev_ts).days

        # Get previous vectors
        prev_df = self.get_vector_matrix(t_prev)
        if prev_df.empty:
            return

        prev_vectors: Dict[str, Dict[int, np.ndarray]] = {}
        for _, row in prev_df.iterrows():
            iid = row['signal_id']
            if iid not in prev_vectors:
                prev_vectors[iid] = {}
            prev_vectors[iid][row['window_days']] = row['vector']

        # Calculate energy by window
        energy = {63: 0.0, 126: 0.0, 252: 0.0}
        barycenter_shifts = []
        n_processed = 0

        # Read geometry signals for barycenter lookups (lazy with filter pushdown)
        geom_ind_df = (
            pl.scan_parquet(self._geometry_signals_path)
            .filter(pl.col('window_end').is_in([t_now, t_prev]))
            .collect()
        )

        for iid, windows in current_vectors.items():
            if iid not in prev_vectors:
                continue

            # Energy contribution from each window
            for win in [63, 126, 252]:
                if win in windows and win in prev_vectors[iid]:
                    dist = euclidean(windows[win], prev_vectors[iid][win])
                    energy[win] += dist * self.weights[win]

            # Barycenter shift
            bc_now_row = geom_ind_df.filter(
                (pl.col('signal_id') == iid) & (pl.col('window_end') == t_now)
            )
            bc_prev_row = geom_ind_df.filter(
                (pl.col('signal_id') == iid) & (pl.col('window_end') == t_prev)
            )

            if len(bc_now_row) > 0 and len(bc_prev_row) > 0:
                bc_now = bc_now_row['barycenter'][0]
                bc_prev = bc_prev_row['barycenter'][0]
                if bc_now is not None and bc_prev is not None:
                    shift = euclidean(np.array(bc_now), np.array(bc_prev))
                    barycenter_shifts.append(shift)

            n_processed += 1

        if n_processed == 0:
            return

        # Aggregate metrics
        energy_total = sum(energy.values())

        # Anchor ratio: did the 252d move more than the 63d?
        # > 1.0 means anchors moved more (structural shift)
        # < 1.0 means scouts moved more (noise)
        anchor_ratio = energy[252] / (energy[63] + 1e-9)

        # Barycenter statistics
        bc_shift_mean = np.mean(barycenter_shifts) if barycenter_shifts else 0
        bc_shift_max = np.max(barycenter_shifts) if barycenter_shifts else 0

        # Dispersion change
        disp_now_df = geom_ind_df.filter(pl.col('window_end') == t_now)
        disp_prev_df = geom_ind_df.filter(pl.col('window_end') == t_prev)

        disp_now = disp_now_df['timescale_dispersion'].mean() if len(disp_now_df) > 0 else 0
        disp_prev = disp_prev_df['timescale_dispersion'].mean() if len(disp_prev_df) > 0 else 0

        disp_now = float(disp_now) if disp_now is not None else 0.0
        disp_prev = float(disp_prev) if disp_prev is not None else 0.0

        dispersion_delta = disp_now - disp_prev

        # Dispersion velocity: rate of change in system dispersion
        dispersion_velocity = dispersion_delta / days_elapsed if days_elapsed > 0 else 0.0

        # Regime conviction: high energy + low dispersion = high conviction
        # High energy + high dispersion = chaos, not regime shift
        regime_conviction = energy_total / (disp_now + 1e-9)

        displacement_record = {
            'window_end_from': t_prev,
            'window_end_to': t_now,
            'days_elapsed': days_elapsed,
            'energy_total': energy_total,
            'energy_63': energy[63],
            'energy_126': energy[126],
            'energy_252': energy[252],
            'anchor_ratio': anchor_ratio,
            'barycenter_shift_mean': float(bc_shift_mean),
            'barycenter_shift_max': float(bc_shift_max),
            'dispersion_delta': dispersion_delta,
            'dispersion_velocity': dispersion_velocity,
            'regime_conviction': regime_conviction,
            'n_signals': n_processed,
        }

        displacement_df = pl.DataFrame([displacement_record])
        upsert_parquet(
            displacement_df,
            self._geometry_displacement_path,
            key_cols=['window_end_from', 'window_end_to']
        )

    def run_range(self, start_date: date, end_date: date, verbose: bool = True):
        """
        Process geometry for a date range.

        Args:
            start_date: First snapshot date
            end_date: Last snapshot date
            verbose: Print progress
        """
        # Find available snapshot dates from vectors table
        if not self._vectors_path.exists():
            logger.warning(f"Vectors parquet file not found at {self._vectors_path}")
            return

        # Lazy scan with filter pushdown - only load dates in range
        dates_df = (
            pl.scan_parquet(self._vectors_path)
            .filter(
                (pl.col('window_end') >= start_date) &
                (pl.col('window_end') <= end_date)
            )
            .select('window_end')
            .unique()
            .sort('window_end')
            .collect()
        )

        if len(dates_df) == 0:
            logger.warning(f"No vector data found between {start_date} and {end_date}")
            return

        available_dates = dates_df['window_end'].to_list()

        # Apply stride
        if self.stride > 1:
            strided_dates = available_dates[::self.stride]
        else:
            strided_dates = available_dates

        logger.info(f"Processing {len(strided_dates)} snapshots from {strided_dates[0]} to {strided_dates[-1]}")

        for i, snapshot_date in enumerate(strided_dates):
            result = self.run_snapshot(snapshot_date)

            if verbose and result.get('status') == 'success':
                logger.info(
                    f"[{i+1}/{len(strided_dates)}] {snapshot_date}: "
                    f"{result['n_signals']} signals, "
                    f"dispersion={result['total_dispersion']:.3f}, "
                    f"alignment={result['mean_alignment']:.3f}"
                )

        logger.info("Geometry computation complete")

    def get_regime_shifts(self, min_conviction: float = 100,
                          min_anchor_ratio: float = 1.5) -> pd.DataFrame:
        """
        Query for detected regime shifts.

        Args:
            min_conviction: Minimum regime_conviction value
            min_anchor_ratio: Minimum anchor_ratio (>1 = anchors moved more than scouts)

        Returns:
            DataFrame of regime shift events
        """
        if not self._geometry_displacement_path.exists():
            return pd.DataFrame()

        # Lazy scan with filter pushdown for regime shifts
        result = (
            pl.scan_parquet(self._geometry_displacement_path)
            .filter(
                (pl.col('regime_conviction') > min_conviction) &
                (pl.col('anchor_ratio') > min_anchor_ratio)
            )
            .select([
            'window_end_from',
            'window_end_to',
            'days_elapsed',
            pl.col('energy_total').round(2).alias('energy'),
            pl.col('anchor_ratio').round(2).alias('anchor_ratio'),
            pl.col('regime_conviction').round(2).alias('conviction'),
            pl.col('barycenter_shift_mean').round(4).alias('shift_mean'),
            pl.col('dispersion_delta').round(4).alias('disp_delta'),
            'n_signals',
            ])
            .sort('regime_conviction', descending=True)
            .collect()
        )

        return result.to_pandas()

    def close(self):
        """Close resources (no-op for Parquet-based storage)."""
        pass
