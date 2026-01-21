"""
Ørthon Geometry Applier

Apply learned geometric structure to test data:
- Assign operating regime
- Compute distance from healthy baseline
- Assign geometry mode
- Compute trajectory alignment with failure patterns
- Output ML-ready features

Usage:
    python -m prism.entry_points.geometry_apply --model models/fd002_geometry.pkl --data data/FD002_test
"""

import polars as pl
import numpy as np
from pathlib import Path
from scipy.special import softmax
from scipy.spatial.distance import cdist
import pickle
import os
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

# Import learned structure
from prism.entry_points.geometry_learn import LearnedGeometry

DATA_PATH = Path(os.environ.get("ORTHON_DATA_PATH", "data/FD002"))


class GeometryApplier:
    """
    Apply learned geometry structure to test data.
    """

    def __init__(self):
        self.learned: Optional[LearnedGeometry] = None

    def load(self, path: Path) -> 'GeometryApplier':
        """Load learned structure from disk."""
        self.learned = LearnedGeometry.load(path)
        print(f"Loaded learned geometry from {path}")
        print(f"  Regimes: {self.learned.n_regimes}")
        print(f"  Modes: {self.learned.n_modes}")
        print(f"  Signals: {len(self.learned.signal_names)}")
        print(f"  Geometry features: {len(self.learned.geometry_features)}")
        return self

    def assign_regime(self, ops: np.ndarray) -> int:
        """Assign operating regime based on conditions."""
        if self.learned.regime_scaler is None or ops is None:
            return 0

        ops_scaled = self.learned.regime_scaler.transform([ops])[0]
        distances = np.linalg.norm(
            ops_scaled - self.learned.regime_centers,
            axis=1
        )
        return int(np.argmin(distances))

    def distance_from_baseline(self, regime: int, signals: Dict[str, float]) -> Dict[str, float]:
        """Compute distance from healthy baseline for each signal."""
        if regime not in self.learned.healthy_baselines:
            regime = 0

        baseline = self.learned.healthy_baselines.get(regime, {})
        distances = {}

        for signal, value in signals.items():
            if signal in baseline:
                mean = baseline[signal]['mean']
                std = baseline[signal]['std']
                if std > 0:
                    distances[signal] = abs(value - mean) / std
                else:
                    distances[signal] = abs(value - mean)
            else:
                distances[signal] = 0.0

        return distances

    def assign_mode(self, geometry_vector: np.ndarray) -> tuple:
        """Assign geometry mode and return probabilities."""
        if self.learned.mode_scaler is None or self.learned.mode_centroids is None:
            return 0, np.array([1.0])

        # Scale
        vec_scaled = self.learned.mode_scaler.transform([geometry_vector])[0]

        # Compute distances to all mode centroids
        distances = np.linalg.norm(
            vec_scaled - self.learned.mode_centroids,
            axis=1
        )

        # Convert to probabilities via softmax of negative distances
        probabilities = softmax(-distances)

        mode_id = int(np.argmin(distances))

        return mode_id, probabilities

    def trajectory_alignment(self, recent_trajectory: np.ndarray) -> Dict[str, float]:
        """Compute alignment with learned failure trajectories.

        Handles different feature dimensions by comparing temporal evolution
        patterns rather than direct element-wise comparison.
        """
        if 'mean' not in self.learned.failure_trajectories:
            return {'alignment': 0.0, 'distance': 0.0}

        failure_traj = self.learned.failure_trajectories['mean']
        failure_std = self.learned.failure_trajectories.get('std', np.ones_like(failure_traj))

        # Align time dimension (first axis)
        min_time = min(recent_trajectory.shape[0], failure_traj.shape[0])
        if min_time < 2:
            return {'alignment': 0.0, 'distance': 0.0}

        recent = recent_trajectory[-min_time:]
        failure = failure_traj[-min_time:]
        std = failure_std[-min_time:]

        # Compute per-timestep summary statistics for comparison
        # This allows comparing trajectories with different numbers of features
        recent_means = np.mean(recent, axis=1) if recent.ndim > 1 else recent
        recent_stds = np.std(recent, axis=1) if recent.ndim > 1 else np.zeros_like(recent)

        failure_means = np.mean(failure, axis=1) if failure.ndim > 1 else failure
        failure_stds = np.std(failure, axis=1) if failure.ndim > 1 else np.zeros_like(failure)
        std_means = np.mean(std, axis=1) if std.ndim > 1 else std

        # Compute correlation on temporal evolution of means
        if np.std(recent_means) > 0 and np.std(failure_means) > 0:
            corr_means = np.corrcoef(recent_means, failure_means)[0, 1]
            corr_means = 0.0 if np.isnan(corr_means) else corr_means
        else:
            corr_means = 0.0

        # Also correlate the variance evolution
        if np.std(recent_stds) > 0 and np.std(failure_stds) > 0:
            corr_stds = np.corrcoef(recent_stds, failure_stds)[0, 1]
            corr_stds = 0.0 if np.isnan(corr_stds) else corr_stds
        else:
            corr_stds = 0.0

        # Combined alignment score
        alignment = 0.7 * corr_means + 0.3 * corr_stds

        # Compute normalized distance on temporal means
        std_means = std_means + 1e-10
        distance = np.mean(np.abs(recent_means - failure_means) / std_means)

        return {
            'alignment': float(alignment),
            'distance': float(distance)
        }

    def coupling_changes(self, regime: int, signals: Dict[str, float]) -> Dict[str, float]:
        """Compute changes in signal coupling from baseline."""
        if regime not in self.learned.coupling_matrices:
            return {}

        baseline_coupling = self.learned.coupling_matrices[regime]

        # Get signal names that are in the coupling matrix
        n_signals = baseline_coupling.shape[0]
        available_signals = [s for s in self.learned.signal_names
                           if s not in self.learned.regime_signals][:n_signals]

        # Build current correlation estimate (simplified - just use distances from mean)
        current_values = np.array([
            signals.get(s, 0.0) for s in available_signals
        ])

        # Compute pairwise products (proxy for correlation)
        if len(current_values) > 1 and np.std(current_values) > 0:
            normalized = (current_values - np.mean(current_values)) / np.std(current_values)
            current_coupling = np.outer(normalized, normalized)

            # Compute coupling delta
            delta = np.abs(current_coupling - baseline_coupling)
            mean_delta = np.mean(delta)
            max_delta = np.max(delta)
        else:
            mean_delta = 0.0
            max_delta = 0.0

        return {
            'coupling_delta_mean': float(mean_delta),
            'coupling_delta_max': float(max_delta)
        }

    def transform(self, obs: pl.DataFrame, vec: Optional[pl.DataFrame] = None,
                  geom: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        Transform test observations into ML-ready features.

        For each (entity, timestamp), compute:
        - regime_id: Operating regime assignment
        - healthy_distance_*: Distance from baseline per signal
        - mode_id: Geometry mode assignment
        - mode_probability_*: Softmax probabilities for each mode
        - trajectory_alignment: Correlation with failure trajectory
        - trajectory_distance: Distance from failure trajectory
        - coupling_delta_mean/max: Changes in signal coupling
        """
        print("="*70)
        print("GEOMETRY APPLIER: Transforming test data")
        print("="*70)

        entities = obs["entity_id"].unique().to_list()
        print(f"\nTest entities: {len(entities)}")

        results = []

        for entity in entities:
            entity_obs = obs.filter(pl.col("entity_id") == entity)
            timestamps = sorted(entity_obs["timestamp"].unique().to_list())

            # Build entity trajectory for trajectory alignment
            entity_trajectory = []

            for ts in timestamps:
                ts_obs = entity_obs.filter(pl.col("timestamp") == ts)

                # Get signal values
                signals = {
                    row['signal_id']: row['value']
                    for row in ts_obs.iter_rows(named=True)
                }

                # 1. Assign regime
                if len(self.learned.regime_signals) >= 2:
                    ops = np.array([
                        signals.get(self.learned.regime_signals[0], 0),
                        signals.get(self.learned.regime_signals[1], 0)
                    ])
                    regime_id = self.assign_regime(ops)
                else:
                    regime_id = 0

                # 2. Distance from healthy baseline
                healthy_distances = self.distance_from_baseline(regime_id, signals)
                mean_healthy_dist = np.mean(list(healthy_distances.values())) if healthy_distances else 0

                # 3. Assign mode (using geometry if available, else use signals)
                if geom is not None:
                    entity_geom = geom.filter(
                        (pl.col("entity_id") == entity) &
                        (pl.col("timestamp") == ts)
                    )
                    if len(entity_geom) > 0:
                        available_feats = [f for f in self.learned.geometry_features
                                          if f in entity_geom.columns]
                        if available_feats:
                            geom_vec = entity_geom.select(available_feats).to_numpy().flatten()
                            geom_vec = np.nan_to_num(geom_vec, nan=0.0)
                            # Pad or truncate to expected size
                            expected_size = self.learned.mode_centroids.shape[1] if self.learned.mode_centroids is not None else len(geom_vec)
                            if len(geom_vec) < expected_size:
                                geom_vec = np.pad(geom_vec, (0, expected_size - len(geom_vec)))
                            elif len(geom_vec) > expected_size:
                                geom_vec = geom_vec[:expected_size]
                            mode_id, mode_probs = self.assign_mode(geom_vec)
                        else:
                            mode_id, mode_probs = 0, np.array([1.0])
                    else:
                        mode_id, mode_probs = 0, np.array([1.0])
                else:
                    # Use signal values as proxy
                    signal_vals = np.array(list(signals.values()))
                    if len(signal_vals) > 0:
                        geom_vec = np.array([
                            np.mean(signal_vals),
                            np.std(signal_vals),
                            np.max(signal_vals)
                        ])
                        if self.learned.mode_centroids is not None:
                            expected_size = self.learned.mode_centroids.shape[1]
                            if len(geom_vec) < expected_size:
                                geom_vec = np.pad(geom_vec, (0, expected_size - len(geom_vec)))
                            elif len(geom_vec) > expected_size:
                                geom_vec = geom_vec[:expected_size]
                        mode_id, mode_probs = self.assign_mode(geom_vec)
                    else:
                        mode_id, mode_probs = 0, np.array([1.0])

                # Track trajectory
                entity_trajectory.append(list(signals.values())[:10])

                # 4. Trajectory alignment (using recent window)
                if len(entity_trajectory) >= self.learned.trajectory_window:
                    recent_traj = np.array(entity_trajectory[-self.learned.trajectory_window:])
                    traj_result = self.trajectory_alignment(recent_traj)
                else:
                    traj_result = {'alignment': 0.0, 'distance': 0.0}

                # 5. Coupling changes
                coupling = self.coupling_changes(regime_id, signals)

                # Build result row
                row = {
                    'entity_id': entity,
                    'timestamp': ts,
                    'regime_id': regime_id,
                    'healthy_distance_mean': mean_healthy_dist,
                    'mode_id': mode_id,
                    'mode_probability_max': float(np.max(mode_probs)),
                    'mode_entropy': float(-np.sum(mode_probs * np.log(mode_probs + 1e-10))),
                    'trajectory_alignment': traj_result['alignment'],
                    'trajectory_distance': traj_result['distance'],
                    'coupling_delta_mean': coupling.get('coupling_delta_mean', 0.0),
                    'coupling_delta_max': coupling.get('coupling_delta_max', 0.0),
                }

                # Add top healthy distances
                sorted_dists = sorted(healthy_distances.items(), key=lambda x: -x[1])[:5]
                for i, (sig, dist) in enumerate(sorted_dists):
                    row[f'healthy_dist_top{i+1}'] = dist

                # Add mode probabilities
                for i, prob in enumerate(mode_probs[:5]):
                    row[f'mode_prob_{i}'] = float(prob)

                results.append(row)

        # Convert to DataFrame
        result_df = pl.DataFrame(results)

        print(f"\nGenerated {len(result_df):,} feature rows")
        print(f"Features: {result_df.columns}")

        return result_df

    def save_features(self, df: pl.DataFrame, path: Path):
        """Save ML features to parquet."""
        df.write_parquet(path)
        print(f"Saved features to {path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Apply learned geometry to test data")
    parser.add_argument("--model", type=str, required=True, help="Path to learned geometry model")
    parser.add_argument("--data", type=str, default=str(DATA_PATH), help="Test data directory")
    parser.add_argument("--output", type=str, default=None, help="Output features path")
    args = parser.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)
    output_path = Path(args.output) if args.output else data_path / "ml_geometry_features.parquet"

    # Load model
    applier = GeometryApplier()
    applier.load(model_path)

    # Load test data
    print(f"\nLoading test data from {data_path}...")
    obs = pl.read_parquet(data_path / "observations.parquet")
    print(f"  Observations: {len(obs):,}")

    vec_path = data_path / "vector.parquet"
    vec = pl.read_parquet(vec_path) if vec_path.exists() else None

    geom_path = data_path / "geometry.parquet"
    geom = pl.read_parquet(geom_path) if geom_path.exists() else None

    # Transform
    features = applier.transform(obs, vec, geom)

    # Save
    applier.save_features(features, output_path)

    # Show sample
    print("\nSample features:")
    print(features.head(5))


if __name__ == "__main__":
    main()
