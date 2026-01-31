"""
Attractor Engine

Computes attractor characteristics: correlation dimension,
effective dimension, and detects attractor changes.

Key insight: Dimension collapsing often precedes failure.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..primitives.embedding import (
    time_delay_embedding, optimal_delay, optimal_dimension
)
from ..primitives.dynamical import correlation_dimension
from ..primitives.matrix import covariance_matrix, eigendecomposition
from ..primitives.tests import marchenko_pastur_test


@dataclass
class AttractorConfig:
    """Configuration for attractor engine."""
    dt: float = 1.0  # Time step in seconds
    embedding_dim: Optional[int] = None  # None = auto-detect
    embedding_tau: Optional[int] = None  # None = auto-detect
    min_samples: int = 50
    collapse_threshold: float = 0.7  # 30% drop = collapse


class AttractorEngine:
    """
    Attractor Characterization Engine.

    Computes attractor dimension and tracks structural changes.

    Outputs:
    - correlation_dimension: Fractal dimension of attractor
    - effective_dimension: Participation ratio from eigenvalues
    - n_significant_modes: Modes above Marchenko-Pastur threshold
    - dimension_collapse: True if dimension dropped significantly
    - attractor_change: Relative change from baseline
    """

    ENGINE_TYPE = "dynamics"

    def __init__(self, config: Optional[AttractorConfig] = None):
        self.config = config or AttractorConfig()
        self.dimension_history: Dict[str, List[float]] = {}

    def compute(
        self,
        signal: np.ndarray,
        unit_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Compute attractor characteristics for a signal.

        Parameters
        ----------
        signal : np.ndarray
            Time series data (can be 1D or 2D multivariate)
        unit_id : str
            Entity identifier

        Returns
        -------
        dict
            Attractor metrics
        """
        signal = np.asarray(signal)

        # Handle 1D vs 2D input
        if signal.ndim == 1:
            signal = signal.flatten()
            signal = signal[~np.isnan(signal)]
            is_multivariate = False
        else:
            # Remove rows with NaN
            valid_rows = ~np.any(np.isnan(signal), axis=1)
            signal = signal[valid_rows]
            is_multivariate = True

        if len(signal) < self.config.min_samples:
            return self._empty_result(unit_id)

        try:
            # Embed if univariate
            if not is_multivariate:
                if self.config.embedding_tau is None:
                    tau = optimal_delay(signal, max_lag=len(signal) // 4)
                else:
                    tau = self.config.embedding_tau

                if self.config.embedding_dim is None:
                    dim = optimal_dimension(signal, tau, max_dim=10)
                else:
                    dim = self.config.embedding_dim

                embedded = time_delay_embedding(signal, dimension=dim, delay=tau)
            else:
                embedded = signal
                dim = signal.shape[1]
                tau = 1

            if len(embedded) < 30:
                return self._empty_result(unit_id)

            # Correlation dimension
            try:
                D2_result = correlation_dimension(embedded)
                D2 = D2_result[0] if isinstance(D2_result, tuple) else D2_result
            except Exception:
                D2 = np.nan

            # Covariance-based effective dimension
            cov = covariance_matrix(embedded)
            eigenvalues, eigenvectors = eigendecomposition(cov)

            # Sort eigenvalues descending
            eigenvalues = np.sort(eigenvalues)[::-1]

            # Effective dimension (participation ratio)
            total_var = np.sum(eigenvalues)
            if total_var > 0:
                p = eigenvalues / total_var
                p = p[p > 0]
                eff_dim = 1.0 / np.sum(p ** 2)  # Participation ratio
            else:
                eff_dim = 0

            # Marchenko-Pastur significance
            sig_mask, mp_bound = marchenko_pastur_test(
                eigenvalues, len(embedded), embedded.shape[1]
            )
            n_significant = int(np.sum(sig_mask))

            # Track history
            if unit_id not in self.dimension_history:
                self.dimension_history[unit_id] = []
            self.dimension_history[unit_id].append(eff_dim)

            # Dimension collapse detection
            history = self.dimension_history[unit_id]
            if len(history) > 1:
                prev_dim = history[-2]
                dimension_change = eff_dim - prev_dim
                dimension_collapse = (eff_dim < prev_dim * self.config.collapse_threshold)
            else:
                dimension_change = 0
                dimension_collapse = False

            # Attractor change relative to baseline
            if len(history) > 1:
                baseline_dim = history[0]
                attractor_change = abs(eff_dim - baseline_dim) / (baseline_dim + 1e-10)
            else:
                attractor_change = 0

            # Eigenvalue ratio (first/second)
            if len(eigenvalues) > 1 and eigenvalues[1] > 0:
                ev_ratio = float(eigenvalues[0] / eigenvalues[1])
            else:
                ev_ratio = np.nan

            return {
                'unit_id': unit_id,
                'n_samples': len(signal) if signal.ndim == 1 else len(embedded),
                'correlation_dimension': float(D2) if not np.isnan(D2) else None,
                'effective_dimension': float(eff_dim),
                'n_significant_modes': n_significant,
                'mp_upper_bound': float(mp_bound),
                'dimension_change': float(dimension_change),
                'dimension_collapse': dimension_collapse,
                'attractor_change': float(attractor_change),
                'embedding_dim': dim,
                'embedding_tau': tau,
                'eigenvalue_1': float(eigenvalues[0]) if len(eigenvalues) > 0 else None,
                'eigenvalue_2': float(eigenvalues[1]) if len(eigenvalues) > 1 else None,
                'eigenvalue_ratio': ev_ratio,
            }

        except Exception as e:
            return self._empty_result(unit_id)

    def _empty_result(self, unit_id: str) -> Dict[str, Any]:
        """Return empty result for insufficient data."""
        return {
            'unit_id': unit_id,
            'n_samples': 0,
            'correlation_dimension': None,
            'effective_dimension': np.nan,
            'n_significant_modes': 0,
            'mp_upper_bound': np.nan,
            'dimension_change': np.nan,
            'dimension_collapse': False,
            'attractor_change': np.nan,
            'embedding_dim': None,
            'embedding_tau': None,
            'eigenvalue_1': None,
            'eigenvalue_2': None,
            'eigenvalue_ratio': np.nan,
        }

    def to_parquet_row(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert result to flat row for parquet output."""
        return {k: v for k, v in result.items()}


def run_attractor_engine(
    observations: pl.DataFrame,
    config: AttractorConfig,
    signal_column: str,
    output_path: Optional[Path] = None
) -> pl.DataFrame:
    """
    Run attractor engine on observations DataFrame.

    Parameters
    ----------
    observations : pl.DataFrame
        Observations with unit_id, signal_id, index, value
    config : AttractorConfig
        Engine configuration
    signal_column : str
        Signal to analyze
    output_path : Path, optional
        Path to write output parquet

    Returns
    -------
    pl.DataFrame
        Attractor results
    """
    engine = AttractorEngine(config)

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
        'unit_id': [], 'effective_dimension': [], 'dimension_collapse': []
    })

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path)

    return df
