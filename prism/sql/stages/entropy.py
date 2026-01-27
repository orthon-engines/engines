"""
Entropy Stage - SQL + Python Engines

SQL handles: binned entropy proxies
Python engines: Sample entropy, permutation entropy, mutual information
"""

from .base import StageOrchestrator
import numpy as np
import pandas as pd


class EntropyStage(StageOrchestrator):
    """Shannon entropy, permutation entropy, mutual information."""

    SQL_FILE = '08_entropy.sql'

    VIEWS = [
        'v_shannon_entropy',          # Shannon entropy (binned)
        'v_permutation_entropy',      # Permutation entropy
        'v_spectral_entropy_proxy',   # Spectral entropy approximation
        'v_mutual_information_pairwise', # Pairwise mutual information
        'v_conditional_entropy_proxy', # Conditional entropy
        'v_entropy_complete',         # Combined view
    ]

    TABLES = [
        't_entropy',
        't_mutual_info',
    ]

    DEPENDS_ON = ['v_base', 'v_dy', 'v_stats_global']

    def _run_engines(self) -> None:
        """Call Python engines for entropy computations."""
        try:
            self._compute_entropy()
        except Exception as e:
            print(f"    Entropy engine failed: {e}")

        try:
            self._compute_mutual_info()
        except Exception as e:
            print(f"    Mutual info failed: {e}")

    def _compute_entropy(self) -> None:
        """Compute sample and permutation entropy."""
        from prism.engines.core import entropy

        obs = self.conn.execute("""
            SELECT entity_id, signal_id, I, y
            FROM observations
            ORDER BY entity_id, signal_id, I
        """).fetchdf()

        if len(obs) == 0:
            return

        result = entropy.compute(obs)

        if len(result) > 0:
            self._insert_df('t_entropy', result)
            print(f"    ✓ Entropy: {len(result)} signals")

    def _compute_mutual_info(self) -> None:
        """Compute pairwise mutual information."""
        from prism.engines.core import mutual_info

        obs = self.conn.execute("""
            SELECT entity_id, signal_id, I, y
            FROM observations
            ORDER BY entity_id, signal_id, I
        """).fetchdf()

        if len(obs) == 0:
            return

        result = mutual_info.compute(obs)

        if len(result) > 0:
            self._insert_df('t_mutual_info', result)
            print(f"    ✓ Mutual info: {len(result)} pairs")
