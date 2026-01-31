"""
Recurrence Engine

Computes Recurrence Quantification Analysis (RQA) metrics.
Tracks determinism, laminarity, entropy, and recurrence patterns.

Key insight: Determinism dropping = system becoming unpredictable.
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
    recurrence_matrix, recurrence_rate, determinism,
    laminarity, trapping_time, entropy_rqa, divergence_rqa
)
from ..primitives.tests import mann_kendall


@dataclass
class RecurrenceConfig:
    """Configuration for recurrence engine."""
    dt: float = 1.0  # Time step in seconds
    embedding_dim: Optional[int] = None  # None = auto-detect
    embedding_tau: Optional[int] = None  # None = auto-detect
    recurrence_threshold: float = 0.1  # Threshold for recurrence
    min_line_length: int = 2  # Minimum diagonal/vertical line
    min_samples: int = 100


class RecurrenceEngine:
    """
    Recurrence Quantification Analysis Engine.

    Computes RQA metrics from recurrence plots.

    Outputs:
    - recurrence_rate: Density of recurrence points
    - determinism: Fraction of points in diagonal lines
    - laminarity: Fraction of points in vertical lines
    - trapping_time: Average vertical line length
    - entropy: Shannon entropy of diagonal line distribution
    - divergence: Inverse of longest diagonal line
    """

    ENGINE_TYPE = "dynamics"

    def __init__(self, config: Optional[RecurrenceConfig] = None):
        self.config = config or RecurrenceConfig()
        self.determinism_history: Dict[str, List[float]] = {}

    def compute(
        self,
        signal: np.ndarray,
        unit_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Compute RQA metrics for a signal.

        Parameters
        ----------
        signal : np.ndarray
            Time series data
        unit_id : str
            Entity identifier

        Returns
        -------
        dict
            RQA metrics
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

            # Embed
            embedded = time_delay_embedding(signal, dimension=dim, delay=tau)

            if len(embedded) < 30:
                return self._empty_result(unit_id)

            # Compute recurrence matrix
            R = recurrence_matrix(embedded, threshold=self.config.recurrence_threshold)

            # Compute RQA metrics
            rr = recurrence_rate(R)
            det = determinism(R, min_line=self.config.min_line_length)
            lam = laminarity(R, min_line=self.config.min_line_length)
            tt = trapping_time(R, min_line=self.config.min_line_length)
            ent = entropy_rqa(R, min_line=self.config.min_line_length)
            div = divergence_rqa(R, min_line=self.config.min_line_length)

            # Track determinism history
            if unit_id not in self.determinism_history:
                self.determinism_history[unit_id] = []
            self.determinism_history[unit_id].append(det)

            # Trend detection
            history = self.determinism_history[unit_id]
            if len(history) >= 4:
                trend, p_trend, tau_mk, slope = mann_kendall(np.array(history))
            else:
                trend, p_trend, slope = 'no trend', 1.0, 0.0

            # Status classifications
            if det < 0.5:
                det_status = 'LOW'
            elif det < 0.7:
                det_status = 'MODERATE'
            else:
                det_status = 'HIGH'

            if lam > 0.7:
                lam_status = 'HIGH_STICKING'
            elif lam > 0.5:
                lam_status = 'MODERATE'
            else:
                lam_status = 'NORMAL'

            return {
                'unit_id': unit_id,
                'n_samples': len(signal),
                'recurrence_rate': float(rr),
                'determinism': float(det),
                'laminarity': float(lam),
                'trapping_time': float(tt),
                'entropy': float(ent),
                'divergence': float(div),
                'embedding_dim': dim,
                'embedding_tau': tau,
                'det_status': det_status,
                'lam_status': lam_status,
                'det_trend': trend,
                'det_trend_p': float(p_trend),
                'det_trend_slope': float(slope),
            }

        except Exception as e:
            return self._empty_result(unit_id)

    def _empty_result(self, unit_id: str) -> Dict[str, Any]:
        """Return empty result for insufficient data."""
        return {
            'unit_id': unit_id,
            'n_samples': 0,
            'recurrence_rate': np.nan,
            'determinism': np.nan,
            'laminarity': np.nan,
            'trapping_time': np.nan,
            'entropy': np.nan,
            'divergence': np.nan,
            'embedding_dim': None,
            'embedding_tau': None,
            'det_status': 'unknown',
            'lam_status': 'unknown',
            'det_trend': 'unknown',
            'det_trend_p': np.nan,
            'det_trend_slope': np.nan,
        }

    def to_parquet_row(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert result to flat row for parquet output."""
        return {k: v for k, v in result.items()}


def run_recurrence_engine(
    observations: pl.DataFrame,
    config: RecurrenceConfig,
    signal_column: str,
    output_path: Optional[Path] = None
) -> pl.DataFrame:
    """
    Run recurrence engine on observations DataFrame.

    Parameters
    ----------
    observations : pl.DataFrame
        Observations with unit_id, signal_id, index, value
    config : RecurrenceConfig
        Engine configuration
    signal_column : str
        Signal to analyze
    output_path : Path, optional
        Path to write output parquet

    Returns
    -------
    pl.DataFrame
        RQA results
    """
    engine = RecurrenceEngine(config)

    entities = observations.select('unit_id').unique().to_series().to_list()
    results = []

    for unit_id in entities:
        entity_obs = observations.filter(pl.col('unit_id') == unit_id)

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
        'unit_id': [], 'determinism': [], 'det_status': []
    })

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path)

    return df
