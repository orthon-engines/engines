"""
PRISM Coupled Inertia Engine

Measures resistance to co-movement change between two signals.

Core Concept:
    Coupled inertia = stability of coupling over multiple timescales.

    High inertia = pair resists decoupling (strongly linked, stable relationship)
    Low inertia = pair easily drifts apart (weakly linked, volatile relationship)

Computation:
    1. Compute rolling correlation across multiple windows (63, 126, 252 days)
    2. Measure variance in correlation across timescales
    3. inertia = 1 / (correlation_variance + epsilon)

    Additionally:
    - Coupling strength = mean absolute correlation
    - Coupling direction = sign of mean correlation (positive/negative coupling)

Phase: Temporal (requires time history)
Category: State Engine (analyzes how relationships evolve)
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from orthon._internal.engines.engine_base import BaseEngine
from orthon._internal.engines.metadata import EngineMetadata

logger = logging.getLogger(__name__)


METADATA = EngineMetadata(
    name="coupled_inertia",
    engine_type="state",
    description="Measures resistance to co-movement change via correlation stability",
    domains={"coupling", "inertia", "stability"},
    requires_window=True,
    deterministic=True,
)


class CoupledInertiaEngine(BaseEngine):
    """
    Coupled Inertia Engine - Measures stability of coupling between signals.

    Analyzes how resistant a pair's co-movement is to change by measuring
    correlation variance across multiple timescales.
    """

    metadata = METADATA

    def __init__(self, windows: list = None):
        """
        Initialize coupled inertia engine.

        Args:
            windows: List of window sizes for rolling correlation.
                     If not provided, loads from stride.yaml config.
        """
        super().__init__()
        if windows is None:
            windows = self._load_windows()
        self.windows = windows

    def _load_windows(self) -> list:
        """Load window sizes from config. Fails if not configured."""
        try:
            from orthon._internal.utils.stride import get_barycenter_weights
            weights = get_barycenter_weights()
            if weights:
                return sorted(weights.keys())
        except Exception as e:
            raise RuntimeError(f"Failed to load window config: {e}")

        raise RuntimeError(
            "No window sizes configured in config/stride.yaml. "
            "Configure domain-specific window sizes before running."
        )

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        **params
    ) -> Dict[str, Any]:
        """
        Compute coupled inertia for a pair of signals.

        Args:
            df: DataFrame with 2 columns (the signal pair)
            run_id: Unique run identifier
            **params: Additional parameters

        Returns:
            Dict with:
                - coupled_inertia: Resistance to decoupling (0-inf, higher = more stable)
                - coupling_strength: Mean absolute correlation across windows
                - coupling_direction: Sign of mean correlation (+1/-1)
                - correlation_variance: Variance in correlation across windows
                - correlation_mean: Mean correlation across windows
                - correlations: Dict of correlations per window
        """
        if df.shape[1] != 2:
            logger.warning(f"Coupled inertia requires exactly 2 columns, got {df.shape[1]}")
            return {}

        if len(df) < max(self.windows):
            logger.warning(f"Insufficient data: need {max(self.windows)} obs, got {len(df)}")
            return {}

        # Get column names
        col_a, col_b = df.columns[0], df.columns[1]
        series_a = df[col_a].values
        series_b = df[col_b].values

        # Remove NaN
        mask = np.isfinite(series_a) & np.isfinite(series_b)
        series_a = series_a[mask]
        series_b = series_b[mask]

        if len(series_a) < max(self.windows):
            return {}

        # Compute rolling correlations for each window
        correlations = {}
        for window in self.windows:
            if len(series_a) < window:
                continue

            # Use last 'window' observations
            a_window = series_a[-window:]
            b_window = series_b[-window:]

            if np.std(a_window) > 0 and np.std(b_window) > 0:
                corr = np.corrcoef(a_window, b_window)[0, 1]
                if np.isfinite(corr):
                    correlations[f'window_{window}'] = float(corr)

        if len(correlations) < 2:
            return {}

        # Extract correlation values
        corr_values = np.array(list(correlations.values()))

        # Compute metrics
        correlation_mean = float(np.mean(corr_values))
        correlation_variance = float(np.var(corr_values))

        # Coupled inertia: inverse of variance (high variance = low inertia)
        # Add small epsilon to prevent division by zero
        coupled_inertia = 1.0 / (correlation_variance + 1e-6)

        # Coupling strength: mean absolute correlation
        coupling_strength = float(np.mean(np.abs(corr_values)))

        # Coupling direction: sign of mean correlation
        coupling_direction = 1 if correlation_mean > 0 else -1

        return {
            'coupled_inertia': float(coupled_inertia),
            'coupling_strength': coupling_strength,
            'coupling_direction': coupling_direction,
            'correlation_variance': correlation_variance,
            'correlation_mean': correlation_mean,
            'correlations': correlations,
            'n_windows': len(correlations),
        }
