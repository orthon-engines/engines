"""
Lyapunov Engine

Computes Lyapunov exponents and stability classification.
Tracks stability trends over time for early warning detection.

Key insight: Lyapunov exponent trending positive = losing stability
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..primitives.embedding import (
    time_delay_embedding, optimal_delay, optimal_dimension
)
from ..primitives.dynamical import (
    lyapunov_rosenstein, lyapunov_kantz
)
from ..primitives.tests import mann_kendall, surrogate_test


@dataclass
class LyapunovConfig:
    """Configuration for Lyapunov engine."""
    dt: float = 1.0  # Time step in seconds
    embedding_dim: Optional[int] = None  # None = auto-detect
    embedding_tau: Optional[int] = None  # None = auto-detect
    method: str = 'rosenstein'  # 'rosenstein' | 'kantz'
    n_surrogates: int = 50  # For significance testing
    min_samples: int = 100


class LyapunovEngine:
    """
    Lyapunov Exponent Engine.

    Computes the largest Lyapunov exponent to characterize system stability.

    Outputs:
    - lyapunov: Largest Lyapunov exponent
    - stability_class: 'chaotic', 'marginal', 'periodic', 'stable'
    - is_significant: Whether result differs from surrogates
    - surrogate_p_value: P-value from surrogate test
    - embedding_dim: Used embedding dimension
    - embedding_tau: Used embedding delay
    """

    ENGINE_TYPE = "dynamics"

    def __init__(self, config: Optional[LyapunovConfig] = None):
        self.config = config or LyapunovConfig()
        self.lyapunov_history: Dict[str, List[float]] = {}

    def compute(
        self,
        signal: np.ndarray,
        unit_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Compute Lyapunov exponent for a signal.

        Parameters
        ----------
        signal : np.ndarray
            Time series data
        unit_id : str
            Entity identifier

        Returns
        -------
        dict
            Lyapunov metrics
        """
        signal = np.asarray(signal).flatten()

        # Remove NaN
        signal = signal[~np.isnan(signal)]

        if len(signal) < self.config.min_samples:
            return self._empty_result(unit_id)

        try:
            # Auto-detect embedding parameters
            if self.config.embedding_tau is None:
                tau = optimal_delay(signal, max_lag=len(signal) // 4)
            else:
                tau = self.config.embedding_tau

            if self.config.embedding_dim is None:
                dim = optimal_dimension(signal, tau, max_dim=10)
            else:
                dim = self.config.embedding_dim

            # Embed the signal
            embedded = time_delay_embedding(signal, dimension=dim, delay=tau)

            if len(embedded) < 50:
                return self._empty_result(unit_id)

            # Compute Lyapunov exponent
            if self.config.method == 'rosenstein':
                lyap_result = lyapunov_rosenstein(embedded)
                lyap = lyap_result[0] if isinstance(lyap_result, tuple) else lyap_result
            else:
                lyap_result = lyapunov_kantz(embedded)
                lyap = lyap_result[0] if isinstance(lyap_result, tuple) else lyap_result

            # Surrogate test for significance (computed, not classification)
            try:
                def compute_lyap(sig):
                    emb = time_delay_embedding(sig, dimension=dim, delay=tau)
                    if len(emb) < 30:
                        return np.nan
                    result = lyapunov_rosenstein(emb)
                    return result[0] if isinstance(result, tuple) else result

                _, p_surr, z_surr = surrogate_test(
                    signal,
                    compute_lyap,
                    n_surrogates=min(self.config.n_surrogates, 30),
                    surrogate_type='phase_randomized'
                )
                is_significant = p_surr < 0.05
            except Exception:
                p_surr = np.nan
                z_surr = np.nan
                is_significant = False

            # Track history for trend detection
            if unit_id not in self.lyapunov_history:
                self.lyapunov_history[unit_id] = []
            self.lyapunov_history[unit_id].append(lyap)

            # Trend detection
            history = self.lyapunov_history[unit_id]
            if len(history) >= 4:
                trend, p_trend, tau_mk, slope = mann_kendall(np.array(history))
            else:
                trend, p_trend, slope = 'no trend', 1.0, 0.0

            return {
                'unit_id': unit_id,
                'n_samples': len(signal),
                'lyapunov': float(lyap),
                'surrogate_p_value': float(p_surr) if not np.isnan(p_surr) else None,
                'surrogate_z_score': float(z_surr) if not np.isnan(z_surr) else None,
                'embedding_dim': dim,
                'embedding_tau': tau,
                'lyapunov_trend_slope': float(slope),
                'lyapunov_trend_p': float(p_trend),
            }

        except Exception as e:
            return self._empty_result(unit_id)

    def _empty_result(self, unit_id: str) -> Dict[str, Any]:
        """Return empty result for insufficient data."""
        return {
            'unit_id': unit_id,
            'n_samples': 0,
            'lyapunov': np.nan,
            'surrogate_p_value': None,
            'surrogate_z_score': None,
            'embedding_dim': None,
            'embedding_tau': None,
            'lyapunov_trend_slope': np.nan,
            'lyapunov_trend_p': np.nan,
        }

    def to_parquet_row(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert result to flat row for parquet output."""
        return {k: v for k, v in result.items()}


def run_lyapunov_engine(
    observations: pl.DataFrame,
    config: LyapunovConfig,
    signal_column: str,
    output_path: Optional[Path] = None
) -> pl.DataFrame:
    """
    Run Lyapunov engine on observations DataFrame.

    Parameters
    ----------
    observations : pl.DataFrame
        Observations with unit_id, signal_id, index, value
    config : LyapunovConfig
        Engine configuration
    signal_column : str
        Signal to analyze
    output_path : Path, optional
        Path to write output parquet

    Returns
    -------
    pl.DataFrame
        Lyapunov results
    """
    engine = LyapunovEngine(config)

    entities = observations.select('unit_id').unique().to_series().to_list()
    results = []

    for unit_id in entities:
        entity_obs = observations.filter(pl.col('unit_id') == unit_id)

        # Get signal
        sig_data = (
            entity_obs
            .filter(pl.col('signal_id') == signal_column)
            .sort('index')
            .select('value')
            .to_series()
            .to_numpy()
        )

        if len(sig_data) > 0:
            result = engine.compute(sig_data, unit_id)
            results.append(engine.to_parquet_row(result))

    df = pl.DataFrame(results) if results else pl.DataFrame({
        'unit_id': [], 'lyapunov': []
    })

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path)

    return df
