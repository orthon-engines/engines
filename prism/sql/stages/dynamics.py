"""
Dynamics Stage - SQL + Python Engines

SQL handles: regime detection, transitions, stability metrics
Python engines: Lyapunov, attractor reconstruction, basin analysis, DMD
"""

from .base import StageOrchestrator
import numpy as np
import pandas as pd


class DynamicsStage(StageOrchestrator):
    """Regime detection, transitions, stability, basins, attractors."""

    SQL_FILE = '06_dynamics.sql'

    VIEWS = [
        'v_rolling_regime_stats',  # Rolling stats for regime detection
        'v_regime_changes',        # Change point detection
        'v_regime_boundaries',     # Regime boundary identification
        'v_regime_assignment',     # Per-point regime assignment
        'v_regime_stats',          # Per-regime statistics
        'v_regime_transitions',    # Transition characterization
        'v_transition_matrix',     # Transition probabilities
        'v_stability',             # Local stability analysis
        'v_basins',                # Basin of attraction detection
        'v_attractors',            # Attractor identification
        'v_recurrence_proxy',      # Recurrence quantification
        'v_bifurcation_candidates', # Bifurcation detection
        'v_phase_velocity',        # Phase space velocity
        'v_dynamics_complete',     # Combined view
        'v_system_regime',         # System-level regime changes
    ]

    TABLES = [
        't_lyapunov',
        't_attractor',
        't_basin',
        't_dmd',
    ]

    DEPENDS_ON = ['v_base', 'v_curvature', 'v_d2y', 'v_local_extrema', 'v_stats_global']

    def _run_engines(self) -> None:
        """Call Python engines for dynamics computations."""
        try:
            self._compute_lyapunov()
        except Exception as e:
            print(f"    Lyapunov failed: {e}")

        try:
            self._compute_attractor()
        except Exception as e:
            print(f"    Attractor failed: {e}")

        try:
            self._compute_basin()
        except Exception as e:
            print(f"    Basin failed: {e}")

        try:
            self._compute_dmd()
        except Exception as e:
            print(f"    DMD failed: {e}")

    def _compute_lyapunov(self) -> None:
        """Compute Lyapunov exponents (with subsampling for speed)."""
        from prism.engines.core import lyapunov

        obs = self.conn.execute("""
            SELECT entity_id, signal_id, I, y
            FROM observations
            ORDER BY entity_id, signal_id, I
        """).fetchdf()

        if len(obs) == 0:
            return

        # Subsample per signal for expensive computation
        subsampled = []
        for (entity_id, signal_id), group in obs.groupby(['entity_id', 'signal_id']):
            group = group.sort_values('I')
            if len(group) > self.MAX_SAMPLES:
                indices = np.linspace(0, len(group) - 1, self.MAX_SAMPLES, dtype=int)
                group = group.iloc[indices]
            subsampled.append(group)

        if not subsampled:
            return

        obs_sub = pd.concat(subsampled, ignore_index=True)
        result = lyapunov.compute(obs_sub)

        if len(result) > 0:
            self._insert_df('t_lyapunov', result)
            print(f"    ✓ Lyapunov: {len(result)} signals")

    def _compute_attractor(self) -> None:
        """Compute attractor reconstruction (with subsampling)."""
        from prism.engines.core import attractor

        obs = self.conn.execute("""
            SELECT entity_id, signal_id, I, y
            FROM observations
            ORDER BY entity_id, signal_id, I
        """).fetchdf()

        if len(obs) == 0:
            return

        # Subsample per signal
        subsampled = []
        for (entity_id, signal_id), group in obs.groupby(['entity_id', 'signal_id']):
            group = group.sort_values('I')
            if len(group) > self.MAX_SAMPLES:
                indices = np.linspace(0, len(group) - 1, self.MAX_SAMPLES, dtype=int)
                group = group.iloc[indices]
            subsampled.append(group)

        if not subsampled:
            return

        obs_sub = pd.concat(subsampled, ignore_index=True)
        result = attractor.compute(obs_sub)

        if len(result) > 0:
            self._insert_df('t_attractor', result)
            print(f"    ✓ Attractor: {len(result)} signals")

    def _compute_basin(self) -> None:
        """Compute basin membership (with subsampling)."""
        from prism.engines.core import basin

        obs = self.conn.execute("""
            SELECT entity_id, signal_id, I, y
            FROM observations
            ORDER BY entity_id, signal_id, I
        """).fetchdf()

        if len(obs) == 0:
            return

        # Subsample per signal
        subsampled = []
        for (entity_id, signal_id), group in obs.groupby(['entity_id', 'signal_id']):
            group = group.sort_values('I')
            if len(group) > self.MAX_SAMPLES:
                indices = np.linspace(0, len(group) - 1, self.MAX_SAMPLES, dtype=int)
                group = group.iloc[indices]
            subsampled.append(group)

        if not subsampled:
            return

        obs_sub = pd.concat(subsampled, ignore_index=True)
        result = basin.compute(obs_sub)

        if len(result) > 0:
            self._insert_df('t_basin', result)
            print(f"    ✓ Basin: {len(result)} signals")

    def _compute_dmd(self) -> None:
        """Compute Dynamic Mode Decomposition."""
        from prism.engines.core import dmd

        obs = self.conn.execute("""
            SELECT entity_id, signal_id, I, y
            FROM observations
            ORDER BY entity_id, signal_id, I
        """).fetchdf()

        if len(obs) == 0:
            return

        result = dmd.compute(obs)

        if len(result) > 0:
            self._insert_df('t_dmd', result)
            print(f"    ✓ DMD: {len(result)} modes")
