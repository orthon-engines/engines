"""
Geometry Stage - SQL + Python Engines

SQL handles: correlation, lag analysis, covariance
Python engines: LOF, clustering, PCA, MST
"""

from .base import StageOrchestrator
import numpy as np
import pandas as pd


class GeometryStage(StageOrchestrator):
    """Behavioral geometry - coupling, correlation, networks, outliers."""

    SQL_FILE = '05_geometry.sql'

    VIEWS = [
        'v_correlation_matrix',
        'v_lagged_correlation',
        'v_optimal_lag',
        'v_lead_lag',
        'v_coupling_network',
        'v_node_degree',
        'v_directional_degree',
        'v_correlation_clusters',
        'v_derivative_correlation',
        'v_covariance_matrix',
        'v_partial_correlation_proxy',
        'v_mutual_info_proxy',
        'v_geometry_complete',
    ]

    TABLES = [
        't_lof_scores',
        't_clusters',
        't_pca',
        't_mst',
    ]

    DEPENDS_ON = ['v_base', 'v_d2y', 'v_stats_global']

    def _run_engines(self) -> None:
        """Call Python engines for geometry computations."""
        try:
            self._compute_lof()
        except Exception as e:
            print(f"    LOF failed: {e}")

        try:
            self._compute_clustering()
        except Exception as e:
            print(f"    Clustering failed: {e}")

        try:
            self._compute_pca()
        except Exception as e:
            print(f"    PCA failed: {e}")

    def _compute_lof(self) -> None:
        """Compute Local Outlier Factor scores."""
        from prism.engines.core import lof

        # Get observations
        obs = self.conn.execute("""
            SELECT entity_id, signal_id, I, y
            FROM observations
            ORDER BY entity_id, signal_id, I
        """).fetchdf()

        if len(obs) == 0:
            return

        # Compute LOF
        result = lof.compute(obs)

        if len(result) > 0:
            # Add is_outlier column
            if 'lof_score' in result.columns:
                result['is_outlier'] = result['lof_score'] > 1.5
            else:
                result['lof_score'] = 1.0
                result['is_outlier'] = False

            self._insert_df('t_lof_scores', result)
            print(f"    ✓ LOF: {len(result)} signals")

    def _compute_clustering(self) -> None:
        """Compute signal clustering."""
        from prism.engines.core import clustering

        obs = self.conn.execute("""
            SELECT entity_id, signal_id, I, y
            FROM observations
            ORDER BY entity_id, signal_id, I
        """).fetchdf()

        if len(obs) == 0:
            return

        result = clustering.compute(obs)

        if len(result) > 0:
            self._insert_df('t_clusters', result)
            print(f"    ✓ Clustering: {len(result)} points")

    def _compute_pca(self) -> None:
        """Compute PCA for dimensionality."""
        from prism.engines.core import pca

        obs = self.conn.execute("""
            SELECT entity_id, signal_id, I, y
            FROM observations
            ORDER BY entity_id, signal_id, I
        """).fetchdf()

        if len(obs) == 0:
            return

        result = pca.compute(obs)

        if len(result) > 0:
            self._insert_df('t_pca', result)
            print(f"    ✓ PCA: {len(result)} components")
