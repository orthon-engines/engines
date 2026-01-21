"""
Ørthon Geometry Learner

Learn geometric structure from training data:
- Operating regimes
- Healthy baseline per regime
- Coupling/correlation structure per regime
- Mode centroids in geometry space
- Failure trajectory patterns

Usage:
    python -m prism.entry_points.geometry_learn --data data/FD002 --output models/fd002_geometry.pkl
"""

import polars as pl
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

DATA_PATH = Path(os.environ.get("ORTHON_DATA_PATH", "data/FD002"))


@dataclass
class LearnedGeometry:
    """Container for all learned geometric structure."""

    # Regime structure
    n_regimes: int = 6
    regime_centers: np.ndarray = None
    regime_scaler: StandardScaler = None
    regime_signals: List[str] = field(default_factory=list)

    # Healthy baseline per regime
    healthy_baselines: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    healthy_stats: Dict[int, Dict[str, Dict]] = field(default_factory=dict)

    # Coupling structure per regime
    coupling_matrices: Dict[int, np.ndarray] = field(default_factory=dict)
    signal_names: List[str] = field(default_factory=list)

    # Mode structure
    n_modes: int = 5
    mode_centroids: np.ndarray = None
    mode_scaler: StandardScaler = None
    geometry_features: List[str] = field(default_factory=list)

    # Failure trajectories
    failure_trajectories: Dict[str, np.ndarray] = field(default_factory=dict)
    trajectory_window: int = 20

    # Metadata
    train_entities: List[str] = field(default_factory=list)

    def save(self, path: Path):
        """Save learned structure to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Saved learned geometry to {path}")

    @classmethod
    def load(cls, path: Path) -> 'LearnedGeometry':
        """Load learned structure from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class GeometryLearner:
    """
    Learn geometric structure from training data.
    """

    def __init__(self, n_regimes: int = 6, n_modes: int = 5, trajectory_window: int = 20):
        self.n_regimes = n_regimes
        self.n_modes = n_modes
        self.trajectory_window = trajectory_window
        self.learned = LearnedGeometry(
            n_regimes=n_regimes,
            n_modes=n_modes,
            trajectory_window=trajectory_window
        )

    def fit(self, obs: pl.DataFrame, vec: pl.DataFrame,
            geom: Optional[pl.DataFrame] = None,
            state: Optional[pl.DataFrame] = None) -> 'GeometryLearner':
        """
        Learn all geometric structure from training data.
        """
        print("="*70)
        print("GEOMETRY LEARNER: Fitting on training data")
        print("="*70)

        self.learned.train_entities = obs["entity_id"].unique().to_list()
        print(f"\nTraining entities: {len(self.learned.train_entities)}")

        # Step 1: Cluster operating conditions into regimes
        print("\n[1/5] Clustering operating regimes...")
        self._cluster_operating_conditions(obs)

        # Step 2: Compute healthy baseline per regime
        print("\n[2/5] Computing healthy baselines per regime...")
        self._compute_healthy_baselines(obs)

        # Step 3: Learn coupling structure per regime
        print("\n[3/5] Learning coupling structure per regime...")
        self._learn_coupling_structure(obs)

        # Step 4: Cluster geometry space into modes
        print("\n[4/5] Clustering geometry space into modes...")
        if geom is not None:
            self._cluster_geometry_space(geom)
        else:
            self._cluster_geometry_from_vector(vec)

        # Step 5: Extract failure trajectories
        print("\n[5/5] Extracting failure trajectory patterns...")
        if state is not None:
            self._extract_failure_trajectories(state, geom)
        else:
            self._extract_trajectories_from_vector(vec)

        print("\n" + "="*70)
        print("LEARNING COMPLETE")
        print("="*70)

        return self

    def _cluster_operating_conditions(self, obs: pl.DataFrame):
        """Cluster operating conditions into regimes."""
        # Find operating condition signals
        op_signals = ["altitude", "op_mach", "mach", "operating_condition"]
        available = [s for s in op_signals if s in obs["signal_id"].unique().to_list()]

        if len(available) >= 2:
            self.learned.regime_signals = available[:2]

            # Pivot operating conditions
            op_conditions = obs.filter(
                pl.col("signal_id").is_in(self.learned.regime_signals)
            ).pivot(
                values="value",
                index=["entity_id", "timestamp"],
                on="signal_id"
            ).drop_nulls()

            # Cluster
            op_matrix = op_conditions.select(self.learned.regime_signals).to_numpy()

            self.learned.regime_scaler = StandardScaler()
            op_scaled = self.learned.regime_scaler.fit_transform(op_matrix)

            kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
            kmeans.fit(op_scaled)

            self.learned.regime_centers = kmeans.cluster_centers_

            print(f"  Found {len(available)} operating signals: {available[:2]}")
            print(f"  Clustered into {self.n_regimes} regimes")
            for i, center in enumerate(self.learned.regime_centers):
                # Inverse transform to get original scale
                orig = self.learned.regime_scaler.inverse_transform([center])[0]
                print(f"    Regime {i}: {self.learned.regime_signals[0]}={orig[0]:.1f}, "
                      f"{self.learned.regime_signals[1]}={orig[1]:.3f}")
        else:
            print("  No operating condition signals found, using single regime")
            self.learned.n_regimes = 1
            self.learned.regime_centers = np.array([[0, 0]])

    def _compute_healthy_baselines(self, obs: pl.DataFrame):
        """Compute healthy baseline statistics per regime."""
        # Get signal names
        self.learned.signal_names = sorted(obs["signal_id"].unique().to_list())

        if len(self.learned.regime_signals) < 2:
            # Single regime - use first 20% of data as "healthy"
            for entity in self.learned.train_entities[:5]:  # Sample entities
                entity_data = obs.filter(pl.col("entity_id") == entity)
                max_ts = entity_data["timestamp"].max()
                healthy_data = entity_data.filter(pl.col("timestamp") < max_ts * 0.2)

                if 0 not in self.learned.healthy_baselines:
                    self.learned.healthy_baselines[0] = {}
                    self.learned.healthy_stats[0] = {}

                for signal in self.learned.signal_names:
                    sig_data = healthy_data.filter(pl.col("signal_id") == signal)["value"].to_numpy()
                    if len(sig_data) > 0:
                        if signal not in self.learned.healthy_stats[0]:
                            self.learned.healthy_stats[0][signal] = {
                                'means': [], 'stds': []
                            }
                        self.learned.healthy_stats[0][signal]['means'].append(np.mean(sig_data))
                        self.learned.healthy_stats[0][signal]['stds'].append(np.std(sig_data))

            # Aggregate
            for signal in self.learned.signal_names:
                if signal in self.learned.healthy_stats[0]:
                    stats = self.learned.healthy_stats[0][signal]
                    self.learned.healthy_baselines[0][signal] = {
                        'mean': np.mean(stats['means']),
                        'std': np.mean(stats['stds']) + np.std(stats['means'])
                    }

            print(f"  Computed baseline for {len(self.learned.healthy_baselines[0])} signals (single regime)")
            return

        # Multi-regime: compute per regime
        # Join observations with regime assignments
        op_conditions = obs.filter(
            pl.col("signal_id").is_in(self.learned.regime_signals)
        ).pivot(
            values="value",
            index=["entity_id", "timestamp"],
            on="signal_id"
        ).drop_nulls()

        op_matrix = op_conditions.select(self.learned.regime_signals).to_numpy()
        op_scaled = self.learned.regime_scaler.transform(op_matrix)

        # Assign regimes
        distances = np.linalg.norm(
            op_scaled[:, np.newaxis, :] - self.learned.regime_centers[np.newaxis, :, :],
            axis=2
        )
        regime_assignments = np.argmin(distances, axis=1)

        op_conditions = op_conditions.with_columns(
            pl.Series("regime", regime_assignments)
        )

        # For each regime, compute healthy baseline (first 20% of each entity's data)
        for regime in range(self.n_regimes):
            self.learned.healthy_baselines[regime] = {}
            self.learned.healthy_stats[regime] = {}

            regime_ops = op_conditions.filter(pl.col("regime") == regime)

            for entity in self.learned.train_entities[:10]:
                entity_regime_data = regime_ops.filter(pl.col("entity_id") == entity)
                if len(entity_regime_data) < 5:
                    continue

                # Get timestamps for this entity in this regime
                timestamps = entity_regime_data["timestamp"].to_list()
                healthy_ts = timestamps[:len(timestamps)//5] if len(timestamps) >= 5 else timestamps

                # Get observations for healthy timestamps
                for signal in self.learned.signal_names:
                    if signal in self.learned.regime_signals:
                        continue

                    sig_data = obs.filter(
                        (pl.col("entity_id") == entity) &
                        (pl.col("signal_id") == signal) &
                        (pl.col("timestamp").is_in(healthy_ts))
                    )["value"].to_numpy()

                    if len(sig_data) > 0:
                        if signal not in self.learned.healthy_stats[regime]:
                            self.learned.healthy_stats[regime][signal] = {'means': [], 'stds': []}
                        self.learned.healthy_stats[regime][signal]['means'].append(np.mean(sig_data))
                        self.learned.healthy_stats[regime][signal]['stds'].append(np.std(sig_data))

            # Aggregate
            for signal in list(self.learned.healthy_stats[regime].keys()):
                stats = self.learned.healthy_stats[regime][signal]
                if len(stats['means']) > 0:
                    self.learned.healthy_baselines[regime][signal] = {
                        'mean': np.mean(stats['means']),
                        'std': np.mean(stats['stds']) + np.std(stats['means'])
                    }

        total_signals = sum(len(b) for b in self.learned.healthy_baselines.values())
        print(f"  Computed baselines for {total_signals} (signal, regime) pairs")

    def _learn_coupling_structure(self, obs: pl.DataFrame):
        """Learn correlation structure between signals per regime."""
        signals = [s for s in self.learned.signal_names if s not in self.learned.regime_signals]

        for regime in range(self.learned.n_regimes):
            # Collect data for this regime
            all_corrs = []

            for entity in self.learned.train_entities[:20]:
                # Get entity data
                entity_data = obs.filter(pl.col("entity_id") == entity)

                # Pivot to wide format
                try:
                    pivot = entity_data.pivot(
                        values="value",
                        index="timestamp",
                        on="signal_id"
                    ).sort("timestamp")
                except:
                    continue

                # Build matrix
                available_signals = [s for s in signals if s in pivot.columns]
                if len(available_signals) < 5:
                    continue

                matrix = np.column_stack([
                    pivot[s].to_numpy() for s in available_signals
                ])
                matrix = np.nan_to_num(matrix, nan=0.0)

                if matrix.std() > 0:
                    scaler = StandardScaler()
                    matrix_norm = scaler.fit_transform(matrix)
                    corr = np.corrcoef(matrix_norm.T)
                    corr = np.nan_to_num(corr, nan=0.0)
                    all_corrs.append(corr)

            if all_corrs:
                # Average correlation matrix
                self.learned.coupling_matrices[regime] = np.mean(all_corrs, axis=0)

        print(f"  Learned coupling matrices for {len(self.learned.coupling_matrices)} regimes")

    def _cluster_geometry_space(self, geom: pl.DataFrame):
        """Cluster geometry space into modes using geometry.parquet."""
        # Select numeric geometry features
        numeric_cols = [c for c in geom.columns
                       if c not in ['entity_id', 'timestamp', 'mode_id', 'computed_at']
                       and geom[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

        self.learned.geometry_features = numeric_cols[:20]  # Limit features

        if len(self.learned.geometry_features) < 3:
            print("  Not enough geometry features, using PCA on vector data")
            return

        # Build feature matrix
        geom_matrix = geom.select(self.learned.geometry_features).to_numpy()
        geom_matrix = np.nan_to_num(geom_matrix, nan=0.0)

        # Scale and cluster
        self.learned.mode_scaler = StandardScaler()
        geom_scaled = self.learned.mode_scaler.fit_transform(geom_matrix)

        kmeans = KMeans(n_clusters=self.n_modes, random_state=42, n_init=10)
        kmeans.fit(geom_scaled)

        self.learned.mode_centroids = kmeans.cluster_centers_

        print(f"  Clustered into {self.n_modes} modes using {len(self.learned.geometry_features)} features")

    def _cluster_geometry_from_vector(self, vec: pl.DataFrame):
        """Fallback: cluster using vector features."""
        # Aggregate vector features per (entity, timestamp)
        agg = vec.group_by(["entity_id", "timestamp"]).agg([
            pl.col("value").mean().alias("mean"),
            pl.col("value").std().alias("std"),
            pl.col("value").max().alias("max"),
        ])

        feature_matrix = agg.select(["mean", "std", "max"]).to_numpy()
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)

        self.learned.geometry_features = ["mean", "std", "max"]
        self.learned.mode_scaler = StandardScaler()
        scaled = self.learned.mode_scaler.fit_transform(feature_matrix)

        kmeans = KMeans(n_clusters=self.n_modes, random_state=42, n_init=10)
        kmeans.fit(scaled)

        self.learned.mode_centroids = kmeans.cluster_centers_

        print(f"  Clustered into {self.n_modes} modes using vector aggregates")

    def _extract_failure_trajectories(self, state: pl.DataFrame, geom: pl.DataFrame):
        """Extract failure trajectory patterns from state data."""
        # Look for failure signatures
        if 'is_failure_signature' in state.columns:
            failure_entities = state.filter(
                pl.col('is_failure_signature') == True
            )['entity_id'].unique().to_list()
        else:
            # Use last N timestamps as "near failure"
            failure_entities = self.learned.train_entities

        trajectories = []

        for entity in failure_entities[:20]:
            entity_geom = geom.filter(pl.col("entity_id") == entity).sort("timestamp")

            if len(entity_geom) < self.trajectory_window:
                continue

            # Get last N geometry snapshots
            last_n = entity_geom.tail(self.trajectory_window)

            # Extract trajectory in geometry space
            if len(self.learned.geometry_features) > 0:
                available = [f for f in self.learned.geometry_features if f in last_n.columns]
                if available:
                    traj = last_n.select(available).to_numpy()
                    traj = np.nan_to_num(traj, nan=0.0)
                    trajectories.append(traj)

        if trajectories:
            # Compute average failure trajectory
            min_len = min(t.shape[0] for t in trajectories)
            aligned = [t[-min_len:] for t in trajectories]
            self.learned.failure_trajectories['mean'] = np.mean(aligned, axis=0)
            self.learned.failure_trajectories['std'] = np.std(aligned, axis=0)

        print(f"  Extracted failure trajectories from {len(trajectories)} entities")

    def _extract_trajectories_from_vector(self, vec: pl.DataFrame):
        """Fallback: extract trajectories from vector data."""
        trajectories = []

        for entity in self.learned.train_entities[:20]:
            entity_vec = vec.filter(pl.col("entity_id") == entity)

            # Aggregate per timestamp
            agg = entity_vec.group_by("timestamp").agg([
                pl.col("value").mean().alias("mean"),
                pl.col("value").std().alias("std"),
            ]).sort("timestamp")

            if len(agg) < self.trajectory_window:
                continue

            last_n = agg.tail(self.trajectory_window)
            traj = last_n.select(["mean", "std"]).to_numpy()
            traj = np.nan_to_num(traj, nan=0.0)
            trajectories.append(traj)

        if trajectories:
            min_len = min(t.shape[0] for t in trajectories)
            aligned = [t[-min_len:] for t in trajectories]
            self.learned.failure_trajectories['mean'] = np.mean(aligned, axis=0)
            self.learned.failure_trajectories['std'] = np.std(aligned, axis=0)

        print(f"  Extracted failure trajectories from {len(trajectories)} entities")

    def save(self, path: Path):
        """Save learned structure."""
        self.learned.save(path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Learn geometry structure from training data")
    parser.add_argument("--data", type=str, default=str(DATA_PATH), help="Data directory")
    parser.add_argument("--output", type=str, default=None, help="Output model path")
    parser.add_argument("--regimes", type=int, default=6, help="Number of operating regimes")
    parser.add_argument("--modes", type=int, default=5, help="Number of geometry modes")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output) if args.output else data_path / "learned_geometry.pkl"

    print(f"Data path: {data_path}")
    print(f"Output: {output_path}")

    # Load data
    print("\nLoading data...")
    obs = pl.read_parquet(data_path / "observations.parquet")
    vec = pl.read_parquet(data_path / "vector.parquet")

    geom_path = data_path / "geometry.parquet"
    geom = pl.read_parquet(geom_path) if geom_path.exists() else None

    state_path = data_path / "state.parquet"
    state = pl.read_parquet(state_path) if state_path.exists() else None

    print(f"  Observations: {len(obs):,}")
    print(f"  Vector: {len(vec):,}")
    print(f"  Geometry: {len(geom):,}" if geom is not None else "  Geometry: None")
    print(f"  State: {len(state):,}" if state is not None else "  State: None")

    # Learn
    learner = GeometryLearner(n_regimes=args.regimes, n_modes=args.modes)
    learner.fit(obs, vec, geom, state)

    # Save
    learner.save(output_path)


if __name__ == "__main__":
    main()
