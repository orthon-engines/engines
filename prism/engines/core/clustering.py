"""
PRISM Clustering Engine

Groups signals by behavioral similarity.

Measures:
- Cluster assignments
- Cluster centroids
- Silhouette scores
- Optimal cluster count

Phase: Structure
Normalization: Z-score
"""

import logging
from typing import Dict, Any, Optional
from datetime import date

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from prism.engines.engine_base import BaseEngine, get_window_dates
from prism.engines.metadata import EngineMetadata


logger = logging.getLogger(__name__)


METADATA = EngineMetadata(
    name="clustering",
    engine_type="geometry",
    description="Behavioral grouping via clustering algorithms",
    domains={"structure", "grouping"},
    requires_window=True,
    deterministic=False,  # K-means has random initialization
)


class ClusteringEngine(BaseEngine):
    """
    Clustering engine for behavioral grouping.
    
    Groups signals based on correlation structure or return patterns.
    
    Outputs:
        - results.clusters: Signal cluster assignments
        - results.centroids: Cluster centroids
    """
    
    name = "clustering"
    phase = "structure"
    default_normalization = "zscore"

    @property
    def metadata(self) -> EngineMetadata:
        return METADATA

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        n_clusters: Optional[int] = None,
        method: str = "kmeans",
        max_clusters: int = 10,
        use_correlation: bool = True,
        **params
    ) -> Dict[str, Any]:
        """
        Run clustering analysis.
        
        Args:
            df: Normalized signal data
            run_id: Unique run identifier
            n_clusters: Number of clusters (None = auto-detect)
            method: 'kmeans' or 'hierarchical'
            max_clusters: Max clusters for auto-detection
            use_correlation: If True, cluster on correlation matrix
        
        Returns:
            Dict with summary metrics
        """
        df_clean = df
        signals = list(df_clean.columns)
        n_signals = len(signals)
        
        if n_signals < 2:
            raise ValueError(f"Need at least 3 signals for clustering, got {n_signals}")
        
        window_start, window_end = get_window_dates(df_clean)
        
        # Prepare feature matrix
        if use_correlation:
            # Use correlation structure as features
            # Drop constant columns (zero variance) before computing correlation
            df_var = df_clean.loc[:, df_clean.std() > 0]
            if df_var.shape[1] < 3:
                raise ValueError(f"Need at least 3 non-constant dimensions, got {df_var.shape[1]}")
            corr_matrix = df_var.corr()
            # Fill any remaining NaN with 0 (zero correlation)
            corr_matrix = corr_matrix.fillna(0)
            X = corr_matrix.values
        else:
            # Use transposed data (signals as samples)
            # Drop constant columns first
            df_var = df_clean.loc[:, df_clean.std() > 0]
            X = df_var.T.values
        
        # Auto-detect optimal clusters if not specified
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(X, max_clusters, method)
        
        n_clusters = min(n_clusters, n_signals - 1)
        
        # Fit clustering
        if method == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(X)
            centroids = model.cluster_centers_
        else:  # hierarchical
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(X)
            # Compute centroids manually
            centroids = np.array([
                X[labels == i].mean(axis=0) for i in range(n_clusters)
            ])
        
        # Compute quality metrics
        if n_clusters > 1 and n_clusters < n_signals:
            silhouette = silhouette_score(X, labels)
            calinski = calinski_harabasz_score(X, labels)
        else:
            silhouette = 0
            calinski = 0
        
        # Store results
        self._store_clusters(
            signals, labels, window_start, window_end, run_id
        )
        self._store_centroids(
            centroids, window_start, window_end, run_id
        )
        
        # Cluster composition
        cluster_sizes = pd.Series(labels).value_counts().sort_index().to_dict()
        
        metrics = {
            "n_signals": n_signals,
            "n_clusters": int(n_clusters),
            "method": method,
            "silhouette_score": float(silhouette),
            "calinski_harabasz": float(calinski),
            "cluster_sizes": cluster_sizes,
            "use_correlation": use_correlation,
        }
        
        logger.info(
            f"Clustering complete: {n_clusters} clusters, "
            f"silhouette={silhouette:.3f}"
        )
        
        return metrics
    
    def _find_optimal_clusters(
        self,
        X: np.ndarray,
        max_clusters: int,
        method: str
    ) -> int:
        """Find optimal cluster count using silhouette score."""
        n_samples = X.shape[0]
        max_k = min(max_clusters, n_samples - 1)
        
        if max_k < 2:
            return 2
        
        scores = []
        for k in range(2, max_k + 1):
            if method == "kmeans":
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
            else:
                model = AgglomerativeClustering(n_clusters=k)
            
            labels = model.fit_predict(X)
            score = silhouette_score(X, labels)
            scores.append((k, score))
        
        # Return k with highest silhouette
        best_k = max(scores, key=lambda x: x[1])[0]
        return best_k
    
    def _store_clusters(
        self,
        signals: list,
        labels: np.ndarray,
        window_start: date,
        window_end: date,
        run_id: str,
    ):
        """Store cluster assignments to results.clusters."""
        records = []
        for signal, label in zip(signals, labels):
            records.append({
                "cluster_id": f"cluster_{label}",
                "window_start": window_start,
                "window_end": window_end,
                "signal_id": signal,
                "membership": 1.0,  # Hard clustering
                "run_id": run_id,
            })
        
        df = pd.DataFrame(records)
        self.store_results("clusters", df, run_id)
    
    def _store_centroids(
        self,
        centroids: np.ndarray,
        window_start: date,
        window_end: date,
        run_id: str,
    ):
        """Store cluster centroids to results.centroids."""
        records = []
        for i, centroid in enumerate(centroids):
            for j, value in enumerate(centroid):
                records.append({
                    "cluster_id": f"cluster_{i}",
                    "window_start": window_start,
                    "window_end": window_end,
                    "dimension": f"dim_{j}",
                    "value": float(value),
                    "run_id": run_id,
                })
        
        df = pd.DataFrame(records)
        self.store_results("centroids", df, run_id)
