"""
Covariance Engine

Computes covariance and correlation matrices from signal data.
Outputs entity-level structural summaries.
"""

import numpy as np
import polars as pl
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from ..primitives.matrix import (
    covariance_matrix,
    correlation_matrix,
)


@dataclass
class CovarianceConfig:
    """Configuration for covariance engine."""
    min_samples: int = 30
    regularization: float = 0.0  # Ridge regularization for ill-conditioned matrices
    compute_partial: bool = False  # Compute partial correlations


class CovarianceEngine:
    """
    Covariance and Correlation Matrix Engine.

    Computes structural relationships between all signals within an entity.

    Outputs:
    - covariance_matrix: Full covariance matrix (n_signals × n_signals)
    - correlation_matrix: Full correlation matrix
    - mean_correlation: Average absolute correlation
    - max_correlation: Maximum absolute correlation (off-diagonal)
    - min_correlation: Minimum absolute correlation (off-diagonal)
    - condition_number: Matrix condition number (ill-conditioning indicator)
    - determinant: Matrix determinant (volume of ellipsoid)
    - trace: Matrix trace (total variance)
    """

    ENGINE_TYPE = "structure"

    def __init__(self, config: Optional[CovarianceConfig] = None):
        self.config = config or CovarianceConfig()

    def compute(
        self,
        signals: Dict[str, np.ndarray],
        unit_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Compute covariance structure for an entity.

        Parameters
        ----------
        signals : dict
            Dictionary mapping signal_id to numpy array of values
        unit_id : str
            Entity identifier

        Returns
        -------
        dict
            Covariance metrics and matrices
        """
        if len(signals) < 2:
            return self._empty_result(unit_id)

        # Build data matrix (align all signals)
        signal_ids = sorted(signals.keys())
        min_len = min(len(signals[s]) for s in signal_ids)

        if min_len < self.config.min_samples:
            return self._empty_result(unit_id)

        # Stack signals as columns
        data = np.column_stack([
            np.asarray(signals[s])[:min_len] for s in signal_ids
        ])

        # Remove rows with NaN
        valid_rows = ~np.any(np.isnan(data), axis=1)
        data = data[valid_rows]

        if len(data) < self.config.min_samples:
            return self._empty_result(unit_id)

        # Compute matrices
        cov_mat = covariance_matrix(data)
        corr_mat = correlation_matrix(data)

        # Apply regularization if needed
        if self.config.regularization > 0:
            n = cov_mat.shape[0]
            cov_mat = cov_mat + self.config.regularization * np.eye(n)

        # Extract metrics from correlation matrix
        n_signals = corr_mat.shape[0]

        # Off-diagonal elements
        mask = ~np.eye(n_signals, dtype=bool)
        off_diag = np.abs(corr_mat[mask])

        mean_corr = float(np.mean(off_diag))
        max_corr = float(np.max(off_diag))
        min_corr = float(np.min(off_diag))

        # Matrix properties
        try:
            eigenvalues = np.linalg.eigvalsh(cov_mat)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            if len(eigenvalues) > 0:
                condition_number = float(eigenvalues.max() / eigenvalues.min())
            else:
                condition_number = np.inf
        except:
            condition_number = np.nan

        try:
            determinant = float(np.linalg.det(cov_mat))
        except:
            determinant = np.nan

        trace = float(np.trace(cov_mat))

        # Partial correlations if requested
        partial_corr_mat = None
        if self.config.compute_partial:
            partial_corr_mat = self._partial_correlation_matrix(corr_mat)

        return {
            'unit_id': unit_id,
            'n_signals': n_signals,
            'n_samples': len(data),
            'signal_ids': signal_ids,
            'covariance_matrix': cov_mat,
            'correlation_matrix': corr_mat,
            'partial_correlation_matrix': partial_corr_mat,
            'mean_correlation': mean_corr,
            'max_correlation': max_corr,
            'min_correlation': min_corr,
            'condition_number': condition_number,
            'determinant': determinant,
            'trace': trace,
        }

    def _partial_correlation_matrix(self, corr_mat: np.ndarray) -> np.ndarray:
        """Compute partial correlation matrix from correlation matrix."""
        try:
            # Partial correlation = -P_ij / sqrt(P_ii * P_jj)
            # where P = inv(correlation_matrix)
            precision = np.linalg.inv(corr_mat)
            d = np.sqrt(np.diag(precision))
            partial = -precision / np.outer(d, d)
            np.fill_diagonal(partial, 1.0)
            return partial
        except:
            return np.full_like(corr_mat, np.nan)

    def _empty_result(self, unit_id: str) -> Dict[str, Any]:
        """Return empty result for insufficient data."""
        return {
            'unit_id': unit_id,
            'n_signals': 0,
            'n_samples': 0,
            'signal_ids': [],
            'covariance_matrix': None,
            'correlation_matrix': None,
            'partial_correlation_matrix': None,
            'mean_correlation': np.nan,
            'max_correlation': np.nan,
            'min_correlation': np.nan,
            'condition_number': np.nan,
            'determinant': np.nan,
            'trace': np.nan,
        }

    def to_parquet_row(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert result to flat row for parquet output."""
        return {
            'unit_id': result['unit_id'],
            'n_signals': result['n_signals'],
            'n_samples': result['n_samples'],
            'mean_correlation': result['mean_correlation'],
            'max_correlation': result['max_correlation'],
            'min_correlation': result['min_correlation'],
            'condition_number': result['condition_number'],
            'determinant': result['determinant'],
            'trace': result['trace'],
        }
