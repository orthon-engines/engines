"""
PRISM Convex Hull Engine

Measures the geometric extent of signals in behavioral space.

Measures:
- Convex hull volume (overall dispersion)
- Surface area
- Number of vertices (boundary signals)
- Centroid location
- Distance from centroid (per signal)
- Inradius / circumradius ratio (shape regularity)

Phase: Structure
Normalization: Z-score required

Interpretation:
- Contracting volume over time = behavioral convergence (crisis?)
- Expanding volume = behavioral divergence
- Boundary signals = extreme behavioral signatures
- Centroid distance = how "mainstream" vs "unusual" each signal is

Note: Full convex hull only computable in low dimensions.
For high-D behavioral space, we use PCA reduction or approximate methods.
"""

import logging
from typing import Dict, Any, List, Tuple
from datetime import date

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, distance
from sklearn.decomposition import PCA

from prism.engines.engine_base import BaseEngine, get_window_dates
from prism.engines.metadata import EngineMetadata


logger = logging.getLogger(__name__)


METADATA = EngineMetadata(
    name="convex_hull",
    engine_type="geometry",
    description="Convex hull analysis for behavioral space extent",
    domains={"structure", "geometry"},
    requires_window=True,
    deterministic=True,
)


def _compute_centroid_distances(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute centroid and distances from each point to centroid.
    
    Returns:
        centroid: (n_dims,) array
        distances: (n_points,) array of distances to centroid
    """
    centroid = X.mean(axis=0)
    distances = np.linalg.norm(X - centroid, axis=1)
    return centroid, distances


def _compute_pairwise_extent(X: np.ndarray) -> Dict[str, float]:
    """
    Compute extent metrics that work in any dimension.
    """
    # Pairwise distances
    n = X.shape[0]
    
    if n < 2:
        return {
            "max_pairwise_distance": 0.0,
            "avg_pairwise_distance": 0.0,
            "min_pairwise_distance": 0.0,
        }
    
    # Compute all pairwise distances
    dists = distance.pdist(X)
    
    return {
        "max_pairwise_distance": float(np.max(dists)),
        "avg_pairwise_distance": float(np.mean(dists)),
        "min_pairwise_distance": float(np.min(dists)),
        "std_pairwise_distance": float(np.std(dists)),
    }


class ConvexHullEngine(BaseEngine):
    """
    Convex Hull engine for behavioral space geometry.
    
    Measures the geometric extent and shape of the signal cloud
    in behavioral space. Useful for tracking convergence/divergence
    and identifying boundary signals.
    
    For high-dimensional spaces, uses PCA projection to compute
    hull metrics in reduced dimensions.
    
    Outputs:
        - results.hull_metrics: Overall hull statistics
        - results.signal_centrality: Per-signal distance from centroid
    """
    
    name = "convex_hull"
    phase = "structure"
    default_normalization = "zscore"

    @property
    def metadata(self) -> EngineMetadata:
        return METADATA

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        max_hull_dims: int = 6,
        **params
    ) -> Dict[str, Any]:
        """
        Run convex hull analysis on behavioral space.
        
        Args:
            df: Behavioral vectors (rows=dimensions, cols=signals)
            run_id: Unique run identifier
            max_hull_dims: Maximum dimensions for hull computation (default 6)
        
        Returns:
            Dict with summary metrics
        """
        signals = list(df.columns)
        n_signals = len(signals)
        n_dims = len(df)
        
        if n_signals < 4:
            raise ValueError(f"Need at least 4 signals for hull, got {n_signals}")
        
        window_start, window_end = get_window_dates(df)
        
        # Prepare data: (n_signals, n_dimensions)
        X = df.T.values
        
        # Compute centroid and distances (works in any dimension)
        centroid, centroid_distances = _compute_centroid_distances(X)
        
        # Pairwise extent metrics (work in any dimension)
        extent_metrics = _compute_pairwise_extent(X)
        
        # Store signal centrality
        self._store_centrality(
            signals, centroid_distances, window_start, window_end, run_id
        )
        
        # For hull computation, reduce dimensionality if needed
        hull_metrics = {}
        
        if n_dims <= max_hull_dims and n_signals > n_dims:
            # Can compute hull directly
            hull_metrics = self._compute_hull_metrics(X, signals)
            hull_metrics["hull_dimensionality"] = n_dims
            hull_metrics["used_pca_projection"] = False
        else:
            # Project to lower dimensions for hull
            n_components = min(max_hull_dims, n_dims, n_signals - 1)
            
            if n_components >= 2:
                pca = PCA(n_components=n_components)
                X_projected = pca.fit_transform(X)
                
                hull_metrics = self._compute_hull_metrics(X_projected, signals)
                hull_metrics["hull_dimensionality"] = n_components
                hull_metrics["used_pca_projection"] = True
                hull_metrics["pca_variance_captured"] = float(sum(pca.explained_variance_ratio_))
        
        # Combine all metrics
        metrics = {
            "n_signals": n_signals,
            "n_dimensions": n_dims,
            "centroid_avg_distance": float(np.mean(centroid_distances)),
            "centroid_max_distance": float(np.max(centroid_distances)),
            "centroid_min_distance": float(np.min(centroid_distances)),
            "centroid_std_distance": float(np.std(centroid_distances)),
            **extent_metrics,
            **hull_metrics,
        }
        
        # Identify boundary vs interior signals
        if "hull_vertices" in hull_metrics:
            metrics["n_boundary_signals"] = hull_metrics.get("n_vertices", 0)
            metrics["boundary_ratio"] = hull_metrics.get("n_vertices", 0) / n_signals
        
        logger.info(
            f"Convex hull complete: {n_signals} signals in {n_dims}D, "
            f"avg centroid dist={metrics['centroid_avg_distance']:.4f}, "
            f"max pairwise={metrics['max_pairwise_distance']:.4f}"
        )
        
        return metrics
    
    def _compute_hull_metrics(
        self,
        X: np.ndarray,
        signals: List[str],
    ) -> Dict[str, Any]:
        """
        Compute convex hull metrics for point cloud.
        
        Args:
            X: (n_points, n_dims) array
            signals: List of signal names
        
        Returns:
            Dict with hull metrics
        """
        n_points, n_dims = X.shape
        
        if n_points <= n_dims:
            logger.warning(f"Too few points ({n_points}) for {n_dims}D hull")
            return {}
        
        try:
            hull = ConvexHull(X)
            
            # Vertices are the boundary signals
            vertex_indices = hull.vertices
            vertex_signals = [signals[i] for i in vertex_indices]
            
            metrics = {
                "hull_volume": float(hull.volume) if n_dims >= 2 else 0.0,
                "hull_area": float(hull.area) if n_dims >= 2 else 0.0,
                "n_vertices": len(vertex_indices),
                "n_facets": len(hull.simplices),
                "hull_vertices": vertex_signals,
            }
            
            # Compute approximate "sphericity" - how round is the hull?
            # Compare volume to volume of enclosing sphere
            if n_dims >= 2 and hull.volume > 0:
                # Approximate radius from max centroid distance
                centroid = X.mean(axis=0)
                max_radius = np.max(np.linalg.norm(X[vertex_indices] - centroid, axis=1))
                
                if n_dims == 2:
                    sphere_area = np.pi * max_radius ** 2
                    metrics["sphericity"] = float(hull.volume / sphere_area) if sphere_area > 0 else 0
                elif n_dims == 3:
                    sphere_vol = (4/3) * np.pi * max_radius ** 3
                    metrics["sphericity"] = float(hull.volume / sphere_vol) if sphere_vol > 0 else 0
                else:
                    # Generalized: ratio of hull volume to hypersphere volume
                    # This is approximate
                    metrics["sphericity"] = None
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Hull computation failed: {e}")
            return {"hull_error": str(e)}
    
    def _store_centrality(
        self,
        signals: List[str],
        distances: np.ndarray,
        window_start: date,
        window_end: date,
        run_id: str,
    ):
        """Store per-signal centrality (distance from centroid)."""
        # Normalize distances to [0, 1] range
        max_dist = distances.max() if distances.max() > 0 else 1.0
        normalized = distances / max_dist
        
        records = []
        for signal, dist, norm_dist in zip(signals, distances, normalized):
            records.append({
                "signal_id": signal,
                "window_start": window_start,
                "window_end": window_end,
                "centroid_distance": float(dist),
                "normalized_distance": float(norm_dist),
                "centrality_rank": None,  # Will compute below
                "run_id": run_id,
            })
        
        # Add rank (1 = closest to centroid = most central)
        sorted_records = sorted(records, key=lambda x: x["centroid_distance"])
        for rank, record in enumerate(sorted_records, 1):
            record["centrality_rank"] = rank
        
        if records:
            df = pd.DataFrame(records)
            self.store_results("signal_centrality", df, run_id)
