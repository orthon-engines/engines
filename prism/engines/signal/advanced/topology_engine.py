"""
Topology Engine

Computes persistent homology metrics: Betti numbers, persistence entropy,
topological complexity. Detects attractor fragmentation and structural changes.

Key insight: Topology reveals structure that geometry misses.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..primitives.embedding import (
    time_delay_embedding, optimal_delay, optimal_dimension
)
from ..primitives.topology import (
    persistence_diagram, betti_numbers, persistence_entropy,
    wasserstein_distance
)


@dataclass
class TopologyConfig:
    """Configuration for topology engine."""
    dt: float = 1.0
    embedding_dim: Optional[int] = None
    embedding_tau: Optional[int] = None
    max_dimension: int = 2
    max_points: int = 300  # Subsample for speed
    min_samples: int = 100


class TopologyEngine:
    """
    Topological Data Analysis Engine.

    Computes persistent homology for signal structure analysis.

    Outputs:
    - betti_0, betti_1, betti_2: Betti numbers (components, loops, voids)
    - persistence_entropy_h0, h1: Entropy of persistence diagrams
    - total_persistence_h1: Total persistence in H1
    - topological_complexity: Combined complexity measure
    - fragmentation: True if betti_0 > 1 (disconnected attractor)
    - topology_change: Wasserstein distance from baseline
    """

    ENGINE_TYPE = "advanced"

    def __init__(self, config: Optional[TopologyConfig] = None):
        self.config = config or TopologyConfig()
        self.baseline_diagrams: Dict[str, Dict] = {}

    def compute(
        self,
        signal: np.ndarray,
        unit_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Compute topological features for a signal.

        Parameters
        ----------
        signal : np.ndarray
            Time series data
        unit_id : str
            Entity identifier

        Returns
        -------
        dict
            Topological metrics
        """
        signal = np.asarray(signal).flatten()
        signal = signal[~np.isnan(signal)]

        if len(signal) < self.config.min_samples:
            return self._empty_result(unit_id)

        try:
            # Embed the signal
            if self.config.embedding_tau is None:
                tau = optimal_delay(signal, max_lag=len(signal) // 4)
            else:
                tau = self.config.embedding_tau

            if self.config.embedding_dim is None:
                dim = optimal_dimension(signal, tau, max_dim=10)
            else:
                dim = self.config.embedding_dim

            embedded = time_delay_embedding(signal, dimension=dim, delay=tau)

            if len(embedded) < 30:
                return self._empty_result(unit_id)

            # Subsample for computational efficiency
            if len(embedded) > self.config.max_points:
                idx = np.random.choice(len(embedded), self.config.max_points, replace=False)
                embedded = embedded[idx]

            # Compute persistence diagram
            diagrams = persistence_diagram(embedded, max_dimension=self.config.max_dimension)

            # Compute threshold for Betti numbers (median pairwise distance)
            from scipy.spatial.distance import pdist
            sample_size = min(100, len(embedded))
            threshold = float(np.median(pdist(embedded[:sample_size])))

            # Betti numbers at threshold
            betti = betti_numbers(diagrams, threshold=threshold)

            # Persistence entropy
            h0_ent = persistence_entropy(diagrams, dimension=0)
            h1_ent = persistence_entropy(diagrams, dimension=1) if 1 in diagrams else 0.0

            # H1 statistics (loops)
            h1_diag = diagrams.get(1, np.array([]).reshape(-1, 2))
            if len(h1_diag) > 0:
                pers = h1_diag[:, 1] - h1_diag[:, 0]
                total_h1 = float(np.sum(pers))
                max_h1 = float(np.max(pers))
            else:
                total_h1, max_h1 = 0.0, 0.0

            # Topological complexity
            complexity = self._compute_complexity(diagrams)

            # Fragmentation detection
            fragmentation = betti.get(0, 1) > 1

            # Compare to baseline
            if unit_id not in self.baseline_diagrams:
                self.baseline_diagrams[unit_id] = diagrams
                topo_change = 0.0
            else:
                try:
                    topo_change = wasserstein_distance(
                        self.baseline_diagrams[unit_id], diagrams, dimension=1
                    )
                except Exception:
                    topo_change = 0.0

            return {
                'unit_id': unit_id,
                'n_samples': len(signal),
                'embedding_dim': dim,
                'embedding_tau': tau,
                'betti_0': int(betti.get(0, 1)),
                'betti_1': int(betti.get(1, 0)),
                'betti_2': int(betti.get(2, 0)),
                'persistence_entropy_h0': float(h0_ent),
                'persistence_entropy_h1': float(h1_ent),
                'total_persistence_h1': float(total_h1),
                'max_persistence_h1': float(max_h1),
                'topological_complexity': float(complexity),
                'fragmentation': fragmentation,
                'topology_change': float(topo_change),
            }

        except Exception as e:
            return self._empty_result(unit_id)

    def _compute_complexity(self, diagrams: Dict) -> float:
        """
        Compute topological complexity from persistence diagrams.
        Sum of all persistence values across dimensions.
        """
        total = 0.0
        for dim, diag in diagrams.items():
            if len(diag) > 0:
                pers = diag[:, 1] - diag[:, 0]
                # Remove infinite persistence
                pers = pers[np.isfinite(pers)]
                total += np.sum(pers)
        return total

    def _empty_result(self, unit_id: str) -> Dict[str, Any]:
        """Return empty result for insufficient data."""
        return {
            'unit_id': unit_id,
            'n_samples': 0,
            'embedding_dim': None,
            'embedding_tau': None,
            'betti_0': 1,
            'betti_1': 0,
            'betti_2': 0,
            'persistence_entropy_h0': np.nan,
            'persistence_entropy_h1': np.nan,
            'total_persistence_h1': np.nan,
            'max_persistence_h1': np.nan,
            'topological_complexity': np.nan,
            'fragmentation': False,
            'topology_change': np.nan,
        }

    def to_parquet_row(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert result to flat row for parquet output."""
        return {k: v for k, v in result.items()}


def run_topology_engine(
    observations: pl.DataFrame,
    config: TopologyConfig,
    signal_column: str,
    output_path: Optional[Path] = None
) -> pl.DataFrame:
    """
    Run topology engine on observations DataFrame.

    Parameters
    ----------
    observations : pl.DataFrame
        Observations with unit_id, signal_id, index, value
    config : TopologyConfig
        Engine configuration
    signal_column : str
        Signal to analyze
    output_path : Path, optional
        Path to write output parquet

    Returns
    -------
    pl.DataFrame
        Topology results
    """
    engine = TopologyEngine(config)

    entities = observations.select('unit_id').unique().to_series().to_list()
    results = []

    for unit_id in entities:
        entity_obs = observations.filter(pl.col('unit_id') == unit_id)

        sig_data = (
            entity_obs
            .filter(pl.col('signal_id') == signal_column)
            .sort('index')
            .select('value')
            .to_series()
            .to_numpy()
        )

        if len(sig_data) > 0:
            result = engine.compute(sig_data, unit_id)
            results.append(engine.to_parquet_row(result))

    df = pl.DataFrame(results) if results else pl.DataFrame({
        'unit_id': [], 'betti_0': [], 'betti_1': [], 'topological_complexity': []
    })

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path)

    return df
