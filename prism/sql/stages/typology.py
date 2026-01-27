"""
Typology Stage - SQL + Python Engines

SQL handles: trend detection, mean reversion, stationarity proxies
Python engines: Hurst exponent (DFA), GARCH
"""

from .base import StageOrchestrator
import numpy as np
import pandas as pd


class TypologyStage(StageOrchestrator):
    """Behavioral typology: trending, mean-reverting, chaotic, random."""

    SQL_FILE = '04_typology.sql'

    VIEWS = [
        'v_trend_detection',       # Trend analysis
        'v_mean_reversion',        # Mean reversion detection
        'v_stationarity_test',     # Stationarity proxy
        'v_chaos_proxy',           # Chaos indicators
        'v_volatility_clustering', # GARCH-like clustering
        'v_signal_typology',       # Final typology
        'v_prism_requests',        # PRISM work order generation
    ]

    TABLES = [
        't_hurst',
        't_garch',
    ]

    DEPENDS_ON = ['v_base', 'v_signal_class', 'v_stats_global', 'v_autocorrelation']

    def _run_engines(self) -> None:
        """Call Python engines for typology computations."""
        try:
            self._compute_hurst()
        except Exception as e:
            print(f"    Hurst failed: {e}")

        try:
            self._compute_garch()
        except Exception as e:
            print(f"    GARCH failed: {e}")

    def _compute_hurst(self) -> None:
        """Compute Hurst exponent (DFA method)."""
        from prism.engines.core import hurst

        obs = self.conn.execute("""
            SELECT entity_id, signal_id, I, y
            FROM observations
            ORDER BY entity_id, signal_id, I
        """).fetchdf()

        if len(obs) == 0:
            return

        result = hurst.compute(obs)

        if len(result) > 0:
            self._insert_df('t_hurst', result)
            print(f"    ✓ Hurst: {len(result)} signals")

    def _compute_garch(self) -> None:
        """Compute GARCH volatility clustering."""
        from prism.engines.core import garch

        obs = self.conn.execute("""
            SELECT entity_id, signal_id, I, y
            FROM observations
            ORDER BY entity_id, signal_id, I
        """).fetchdf()

        if len(obs) == 0:
            return

        result = garch.compute(obs)

        if len(result) > 0:
            self._insert_df('t_garch', result)
            print(f"    ✓ GARCH: {len(result)} signals")

    def get_prism_work_order(self) -> list:
        """
        Query the PRISM work order view.

        PURE: Just queries the view. Logic is in SQL.
        """
        return self.query('v_prism_requests').to_dict(orient='records')
