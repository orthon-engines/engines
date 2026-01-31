"""
Causality Engine

Computes causal networks using Granger causality and Transfer Entropy.
Identifies drivers, sinks, feedback loops, and causal hierarchy.

Key insight: Understanding causation reveals intervention points.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

from ..primitives.pairwise import granger_causality, transfer_entropy
from ..primitives.network import (
    threshold_matrix, network_density, centrality_betweenness
)


@dataclass
class CausalityConfig:
    """Configuration for causality engine."""
    dt: float = 1.0
    max_lag: int = 5
    threshold_percentile: float = 75.0
    min_samples: int = 50


class CausalityEngine:
    """
    Causal Network Analysis Engine.

    Computes pairwise causality and network metrics.

    Outputs:
    - Pairwise: granger_f, granger_p, transfer_entropy, is_significant
    - Network: density, hierarchy, n_feedback_loops, top_driver, top_sink
    """

    ENGINE_TYPE = "advanced"

    def __init__(self, config: Optional[CausalityConfig] = None):
        self.config = config or CausalityConfig()

    def compute(
        self,
        signals: np.ndarray,
        signal_names: List[str],
        unit_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Compute causal network for multivariate signals.

        Parameters
        ----------
        signals : np.ndarray
            2D array (n_samples, n_signals)
        signal_names : list
            Names for each signal column
        unit_id : str
            Entity identifier

        Returns
        -------
        dict
            Contains 'edges' (list of dicts) and 'network' (dict)
        """
        signals = np.asarray(signals)
        if signals.ndim == 1:
            signals = signals.reshape(-1, 1)

        n_samples, n_signals = signals.shape

        if n_samples < self.config.min_samples:
            return self._empty_result(unit_id, signal_names)

        if n_signals < 2:
            return self._empty_result(unit_id, signal_names)

        try:
            # Build causality matrices
            granger_mat = np.zeros((n_signals, n_signals))
            granger_p_mat = np.ones((n_signals, n_signals))
            te_mat = np.zeros((n_signals, n_signals))

            edges = []

            for i in range(n_signals):
                for j in range(n_signals):
                    if i == j:
                        continue

                    # Granger: does i cause j?
                    try:
                        f_stat, p_val, opt_lag = granger_causality(
                            signals[:, i], signals[:, j], max_lag=self.config.max_lag
                        )
                        granger_mat[i, j] = f_stat
                        granger_p_mat[i, j] = p_val
                    except Exception:
                        f_stat, p_val = 0.0, 1.0
                        granger_mat[i, j] = 0.0
                        granger_p_mat[i, j] = 1.0

                    # Transfer entropy: i -> j
                    try:
                        te = transfer_entropy(
                            signals[:, i], signals[:, j], lag=self.config.max_lag
                        )
                        te_mat[i, j] = te
                    except Exception:
                        te = 0.0
                        te_mat[i, j] = 0.0

                    edges.append({
                        'source': signal_names[i],
                        'target': signal_names[j],
                        'granger_f': float(granger_mat[i, j]),
                        'granger_p': float(granger_p_mat[i, j]),
                        'transfer_entropy': float(te_mat[i, j]),
                        'is_significant': granger_p_mat[i, j] < 0.05,
                    })

            # Network metrics from TE matrix
            adj = threshold_matrix(te_mat, percentile=self.config.threshold_percentile)
            density = network_density(adj, directed=True)

            # Hierarchy: based on transitivity of causal flow
            hierarchy = self._compute_hierarchy(adj)

            # Feedback loops
            loops = self._find_feedback_loops(adj, max_length=4)

            # Driver/sink (net outflow of TE)
            outflow = te_mat.sum(axis=1) - te_mat.sum(axis=0)
            driver_idx = int(np.argmax(outflow))
            sink_idx = int(np.argmin(outflow))

            # Bottleneck (highest betweenness centrality)
            between = centrality_betweenness(adj)
            bottleneck_idx = int(np.argmax(between))

            # Mean TE among significant edges
            significant_te = te_mat[te_mat > 0]
            mean_te = float(np.mean(significant_te)) if len(significant_te) > 0 else 0.0

            network_metrics = {
                'unit_id': unit_id,
                'n_samples': n_samples,
                'n_signals': n_signals,
                'density': float(density),
                'hierarchy': float(hierarchy),
                'n_feedback_loops': len(loops),
                'top_driver': signal_names[driver_idx],
                'top_driver_flow': float(outflow[driver_idx]),
                'top_sink': signal_names[sink_idx],
                'top_sink_flow': float(outflow[sink_idx]),
                'bottleneck': signal_names[bottleneck_idx],
                'bottleneck_centrality': float(between[bottleneck_idx]),
                'mean_te': mean_te,
                'n_significant_edges': sum(1 for e in edges if e['is_significant']),
            }

            return {
                'unit_id': unit_id,
                'edges': edges,
                'network': network_metrics,
            }

        except Exception as e:
            return self._empty_result(unit_id, signal_names)

    def _compute_hierarchy(self, adj: np.ndarray) -> float:
        """
        Compute network hierarchy based on flow asymmetry.
        Hierarchy = 1 means perfect tree structure (no feedback)
        Hierarchy = 0 means fully symmetric/circular
        """
        n = adj.shape[0]
        if n < 2:
            return 0.0

        # Count bidirectional edges
        bidirectional = 0
        unidirectional = 0

        for i in range(n):
            for j in range(i + 1, n):
                has_ij = adj[i, j] > 0
                has_ji = adj[j, i] > 0
                if has_ij and has_ji:
                    bidirectional += 1
                elif has_ij or has_ji:
                    unidirectional += 1

        total = bidirectional + unidirectional
        if total == 0:
            return 0.0

        # Hierarchy = fraction of edges that are unidirectional
        return unidirectional / total

    def _find_feedback_loops(self, adj: np.ndarray, max_length: int = 4) -> List[List[int]]:
        """Find all feedback loops up to max_length."""
        n = adj.shape[0]
        loops = []

        def dfs(start: int, current: int, path: List[int], visited: set):
            if len(path) > max_length:
                return

            for next_node in range(n):
                if adj[current, next_node] > 0:
                    if next_node == start and len(path) >= 2:
                        # Found a loop
                        loops.append(path.copy())
                    elif next_node not in visited:
                        visited.add(next_node)
                        path.append(next_node)
                        dfs(start, next_node, path, visited)
                        path.pop()
                        visited.remove(next_node)

        for start in range(n):
            dfs(start, start, [start], {start})

        return loops

    def _empty_result(self, unit_id: str, signal_names: List[str]) -> Dict[str, Any]:
        """Return empty result for insufficient data."""
        return {
            'unit_id': unit_id,
            'edges': [],
            'network': {
                'unit_id': unit_id,
                'n_samples': 0,
                'n_signals': len(signal_names),
                'density': np.nan,
                'hierarchy': np.nan,
                'n_feedback_loops': 0,
                'top_driver': None,
                'top_driver_flow': np.nan,
                'top_sink': None,
                'top_sink_flow': np.nan,
                'bottleneck': None,
                'bottleneck_centrality': np.nan,
                'mean_te': np.nan,
                'n_significant_edges': 0,
            },
        }


def run_causality_engine(
    observations: pl.DataFrame,
    config: CausalityConfig,
    signal_columns: List[str],
    output_path: Optional[Path] = None
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Run causality engine on observations DataFrame.

    Parameters
    ----------
    observations : pl.DataFrame
        Observations with unit_id, signal_id, index, value
    config : CausalityConfig
        Engine configuration
    signal_columns : list
        Signals to analyze for causality
    output_path : Path, optional
        Base path for output (will create _edges.parquet and _network.parquet)

    Returns
    -------
    tuple of pl.DataFrame
        (edges_df, network_df)
    """
    engine = CausalityEngine(config)

    entities = observations.select('unit_id').unique().to_series().to_list()
    all_edges = []
    all_network = []

    for unit_id in entities:
        entity_obs = observations.filter(pl.col('unit_id') == unit_id)

        # Pivot to get signals as columns
        signals_data = []
        for sig_col in signal_columns:
            sig_data = (
                entity_obs
                .filter(pl.col('signal_id') == sig_col)
                .sort('index')
                .select('value')
                .to_series()
                .to_numpy()
            )
            signals_data.append(sig_data)

        if len(signals_data) == 0:
            continue

        # Align to minimum length
        min_len = min(len(s) for s in signals_data)
        if min_len < config.min_samples:
            continue

        signals_matrix = np.column_stack([s[:min_len] for s in signals_data])

        result = engine.compute(signals_matrix, signal_columns, unit_id)

        # Add edges
        for edge in result['edges']:
            edge['unit_id'] = unit_id
            all_edges.append(edge)

        # Add network metrics
        all_network.append(result['network'])

    # Create DataFrames
    df_edges = pl.DataFrame(all_edges) if all_edges else pl.DataFrame({
        'unit_id': [], 'source': [], 'target': [], 'granger_f': [],
        'granger_p': [], 'transfer_entropy': [], 'is_significant': []
    })

    df_network = pl.DataFrame(all_network) if all_network else pl.DataFrame({
        'unit_id': [], 'density': [], 'hierarchy': [], 'n_feedback_loops': []
    })

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_edges.write_parquet(output_path.parent / f"{output_path.stem}_edges.parquet")
        df_network.write_parquet(output_path.parent / f"{output_path.stem}_network.parquet")

    return df_edges, df_network
