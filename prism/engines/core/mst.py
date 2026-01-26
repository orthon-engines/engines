"""
PRISM Minimum Spanning Tree Engine

Builds the minimum spanning tree of signals in behavioral space.

Measures:
- Total MST weight (sum of edge lengths)
- Average edge length
- Max edge length (weakest link)
- Hub signals (highest degree in MST)
- MST diameter (longest path)

Phase: Structure
Normalization: Z-score preferred

Interpretation:
- Contracting MST (shorter edges) = behavioral convergence
- Expanding MST (longer edges) = behavioral divergence
- High-degree nodes = "bridge" signals connecting clusters
"""

import logging
from typing import Dict, Any, List, Tuple
from datetime import date

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import Counter

from prism.engines.engine_base import BaseEngine, get_window_dates
from prism.engines.metadata import EngineMetadata


logger = logging.getLogger(__name__)


METADATA = EngineMetadata(
    name="mst",
    engine_type="geometry",
    description="Minimum spanning tree structure in behavioral space",
    domains={"structure", "network"},
    requires_window=True,
    deterministic=True,
)


def _build_mst(distance_matrix: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
    """
    Build minimum spanning tree from distance matrix.
    
    Returns:
        mst_matrix: Sparse MST adjacency matrix
        edges: List of (i, j, weight) tuples
    """
    # scipy's minimum_spanning_tree expects a dense or sparse matrix
    # It returns a sparse matrix with only MST edges
    mst_sparse = minimum_spanning_tree(distance_matrix)
    mst_matrix = mst_sparse.toarray()
    
    # Extract edges
    edges = []
    rows, cols = np.where(mst_matrix > 0)
    for i, j in zip(rows, cols):
        edges.append((i, j, mst_matrix[i, j]))
    
    return mst_matrix, edges


def _compute_node_degrees(edges: List[Tuple[int, int, float]], n_nodes: int) -> Dict[int, int]:
    """Compute degree of each node in the MST."""
    degree_count = Counter()
    for i, j, _ in edges:
        degree_count[i] += 1
        degree_count[j] += 1
    
    # Ensure all nodes are represented
    return {node: degree_count.get(node, 0) for node in range(n_nodes)}


def _compute_mst_diameter(mst_matrix: np.ndarray) -> Tuple[int, List[int]]:
    """
    Compute MST diameter (longest shortest path between any two nodes).
    
    Uses BFS from each node to find the farthest node.
    """
    n = mst_matrix.shape[0]
    
    if n <= 1:
        return 0, []
    
    # Make symmetric for traversal
    mst_symmetric = mst_matrix + mst_matrix.T
    
    def bfs_farthest(start: int) -> Tuple[int, int, List[int]]:
        """BFS to find farthest node and path length from start."""
        visited = {start}
        queue = [(start, 0, [start])]
        farthest_node = start
        max_dist = 0
        best_path = [start]
        
        while queue:
            node, dist, path = queue.pop(0)
            if dist > max_dist:
                max_dist = dist
                farthest_node = node
                best_path = path
            
            for neighbor in range(n):
                if mst_symmetric[node, neighbor] > 0 and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1, path + [neighbor]))
        
        return farthest_node, max_dist, best_path
    
    # Find one end of diameter by BFS from node 0
    end1, _, _ = bfs_farthest(0)
    
    # Find other end and diameter by BFS from end1
    end2, diameter, path = bfs_farthest(end1)
    
    return diameter, path


class MSTEngine(BaseEngine):
    """
    Minimum Spanning Tree engine for behavioral space.
    
    Builds the MST connecting all signals with minimum total
    distance in behavioral space. Reveals structural relationships
    and "bridge" signals.
    
    Outputs:
        - results.mst_edges: MST edge list
        - results.mst_metrics: Summary metrics
    """
    
    name = "mst"
    phase = "structure"
    default_normalization = "zscore"

    @property
    def metadata(self) -> EngineMetadata:
        return METADATA

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        distance_metric: str = "euclidean",
        **params
    ) -> Dict[str, Any]:
        """
        Run MST analysis on behavioral space.
        
        Args:
            df: Behavioral vectors (rows=dimensions, cols=signals)
            run_id: Unique run identifier
            distance_metric: Distance metric for scipy.pdist
        
        Returns:
            Dict with summary metrics
        """
        signals = list(df.columns)
        n_signals = len(signals)
        
        if n_signals < 2:
            raise ValueError(f"Need at least 3 signals for MST, got {n_signals}")
        
        window_start, window_end = get_window_dates(df)
        
        # Compute pairwise distances
        # df.T gives us (n_signals, n_dimensions) for pdist
        X = df.T.values  # Each row is an signal's behavioral vector
        distances = pdist(X, metric=distance_metric)
        distance_matrix = squareform(distances)
        
        # Build MST
        mst_matrix, edges = _build_mst(distance_matrix)
        
        if not edges:
            logger.warning("MST has no edges")
            return {"n_signals": n_signals, "n_edges": 0}
        
        # Compute metrics
        edge_weights = [w for _, _, w in edges]
        total_weight = sum(edge_weights)
        avg_weight = np.mean(edge_weights)
        max_weight = max(edge_weights)
        min_weight = min(edge_weights)
        std_weight = np.std(edge_weights)
        
        # Node degrees (connectivity in MST)
        degrees = _compute_node_degrees(edges, n_signals)
        max_degree = max(degrees.values())
        hub_nodes = [i for i, d in degrees.items() if d == max_degree]
        hub_signals = [signals[i] for i in hub_nodes]
        
        # Leaf nodes (degree 1)
        leaf_count = sum(1 for d in degrees.values() if d == 1)
        
        # MST diameter (longest path)
        diameter, diameter_path = _compute_mst_diameter(mst_matrix)
        
        # Store edge list
        self._store_edges(
            edges, signals, window_start, window_end, run_id
        )
        
        # Store hub information
        self._store_hubs(
            degrees, signals, window_start, window_end, run_id
        )
        
        metrics = {
            "n_signals": n_signals,
            "n_edges": len(edges),  # Should be n_signals - 1
            "total_weight": float(total_weight),
            "avg_edge_weight": float(avg_weight),
            "max_edge_weight": float(max_weight),
            "min_edge_weight": float(min_weight),
            "std_edge_weight": float(std_weight),
            "max_degree": int(max_degree),
            "n_hubs": len(hub_nodes),
            "n_leaves": int(leaf_count),
            "diameter": int(diameter),
            "distance_metric": distance_metric,
        }
        
        logger.info(
            f"MST complete: {n_signals} signals, "
            f"total_weight={total_weight:.4f}, "
            f"diameter={diameter}, "
            f"hubs={hub_signals[:3]}"
        )
        
        return metrics
    
    def _store_edges(
        self,
        edges: List[Tuple[int, int, float]],
        signals: List[str],
        window_start: date,
        window_end: date,
        run_id: str,
    ):
        """Store MST edges."""
        records = []
        for i, j, weight in edges:
            records.append({
                "signal_1": signals[i],
                "signal_2": signals[j],
                "window_start": window_start,
                "window_end": window_end,
                "edge_weight": float(weight),
                "run_id": run_id,
            })
        
        if records:
            df = pd.DataFrame(records)
            self.store_results("mst_edges", df, run_id)
    
    def _store_hubs(
        self,
        degrees: Dict[int, int],
        signals: List[str],
        window_start: date,
        window_end: date,
        run_id: str,
    ):
        """Store node degree information."""
        records = []
        for node_idx, degree in degrees.items():
            records.append({
                "signal_id": signals[node_idx],
                "window_start": window_start,
                "window_end": window_end,
                "mst_degree": int(degree),
                "is_hub": degree >= 3,
                "is_leaf": degree == 1,
                "run_id": run_id,
            })
        
        if records:
            df = pd.DataFrame(records)
            self.store_results("mst_nodes", df, run_id)
