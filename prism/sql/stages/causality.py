"""
Causality Stage - SQL + Python Engines

SQL handles: correlation-based proxies, lag relationships
Python engines: Granger, Transfer Entropy, Cointegration
"""

from .base import StageOrchestrator
import numpy as np
import pandas as pd


class CausalityStage(StageOrchestrator):
    """Causal mechanics - Granger, transfer entropy, causal roles."""

    SQL_FILE = '07_causality.sql'

    VIEWS = [
        'v_granger_proxy',
        'v_bidirectional_causality',
        'v_transfer_entropy_proxy',
        'v_causal_roles',
        'v_causal_chains',
        'v_causal_timing',
        'v_intervention_effects',
        'v_root_cause_candidates',
        'v_causal_strength',
        'v_causal_graph',
        'v_causality_complete',
        'v_system_causal_structure',
    ]

    TABLES = [
        't_granger',
        't_transfer_entropy',
        't_cointegration',
        't_causal_graph',
    ]

    DEPENDS_ON = ['v_base', 'v_optimal_lag', 'v_regime_changes', 'v_dy']

    def _run_engines(self) -> None:
        """Call Python engines for causality computations."""
        try:
            self._compute_granger()
        except Exception as e:
            print(f"    Granger failed: {e}")

        try:
            self._compute_transfer_entropy()
        except Exception as e:
            print(f"    Transfer entropy failed: {e}")

        try:
            self._compute_cointegration()
        except Exception as e:
            print(f"    Cointegration failed: {e}")

        try:
            self._build_causal_graph()
        except Exception as e:
            print(f"    Causal graph failed: {e}")

    def _compute_granger(self) -> None:
        """Compute Granger causality for all signal pairs."""
        from prism.engines.core import granger

        obs = self.conn.execute("""
            SELECT entity_id, signal_id, I, y
            FROM observations
            ORDER BY entity_id, signal_id, I
        """).fetchdf()

        if len(obs) == 0:
            return

        result = granger.compute(obs)

        if len(result) > 0:
            # Rename columns for consistency
            if 'source_id' in result.columns:
                result = result.rename(columns={
                    'source_id': 'source_signal',
                    'target_id': 'target_signal'
                })
            self._insert_df('t_granger', result)
            print(f"    ✓ Granger: {len(result)} pairs")

    def _compute_transfer_entropy(self) -> None:
        """Compute transfer entropy for all signal pairs."""
        from prism.engines.core import transfer_entropy

        obs = self.conn.execute("""
            SELECT entity_id, signal_id, I, y
            FROM observations
            ORDER BY entity_id, signal_id, I
        """).fetchdf()

        if len(obs) == 0:
            return

        result = transfer_entropy.compute(obs)

        if len(result) > 0:
            if 'source_id' in result.columns:
                result = result.rename(columns={
                    'source_id': 'source_signal',
                    'target_id': 'target_signal'
                })
            self._insert_df('t_transfer_entropy', result)
            print(f"    ✓ Transfer entropy: {len(result)} pairs")

    def _compute_cointegration(self) -> None:
        """Compute cointegration for all signal pairs."""
        from prism.engines.core import cointegration

        obs = self.conn.execute("""
            SELECT entity_id, signal_id, I, y
            FROM observations
            ORDER BY entity_id, signal_id, I
        """).fetchdf()

        if len(obs) == 0:
            return

        result = cointegration.compute(obs)

        if len(result) > 0:
            self._insert_df('t_cointegration', result)
            print(f"    ✓ Cointegration: {len(result)} pairs")

    def _build_causal_graph(self) -> None:
        """Build causal graph from granger and transfer entropy results."""
        # Try to read granger results
        try:
            granger_df = self.conn.execute("SELECT * FROM t_granger").fetchdf()
        except:
            granger_df = pd.DataFrame()

        # Try to read transfer entropy results
        try:
            te_df = self.conn.execute("SELECT * FROM t_transfer_entropy").fetchdf()
        except:
            te_df = pd.DataFrame()

        edges = []

        # Add significant Granger edges
        if len(granger_df) > 0:
            sig_col = 'granger_significant' if 'granger_significant' in granger_df.columns else None
            p_col = 'granger_p' if 'granger_p' in granger_df.columns else 'p_value' if 'p_value' in granger_df.columns else None

            for _, row in granger_df.iterrows():
                is_sig = False
                if sig_col and row.get(sig_col):
                    is_sig = True
                elif p_col and pd.notna(row.get(p_col)) and row[p_col] < 0.05:
                    is_sig = True

                if is_sig:
                    edges.append({
                        'entity_id': row.get('entity_id', 'default'),
                        'from_signal': row.get('source_signal', row.get('signal_a', '')),
                        'to_signal': row.get('target_signal', row.get('signal_b', '')),
                        'edge_type': 'granger',
                        'edge_weight': row.get('granger_f', row.get('f_stat', 0)),
                        'p_value': row.get('granger_p', row.get('p_value', 1)),
                        'lag': row.get('granger_lag', row.get('optimal_lag', 0)),
                    })

        # Add significant TE edges
        if len(te_df) > 0:
            te_col = 'transfer_entropy' if 'transfer_entropy' in te_df.columns else 'te' if 'te' in te_df.columns else None

            if te_col:
                for _, row in te_df.iterrows():
                    te_val = row.get(te_col, 0) or 0
                    if te_val > 0.05:  # Threshold for significance
                        edges.append({
                            'entity_id': row.get('entity_id', 'default'),
                            'from_signal': row.get('source_signal', row.get('signal_a', '')),
                            'to_signal': row.get('target_signal', row.get('signal_b', '')),
                            'edge_type': 'transfer_entropy',
                            'edge_weight': te_val,
                            'p_value': np.nan,
                            'lag': 0,
                        })

        if edges:
            causal_graph = pd.DataFrame(edges)
            self._insert_df('t_causal_graph', causal_graph)
            print(f"    ✓ Causal graph: {len(causal_graph)} edges")
