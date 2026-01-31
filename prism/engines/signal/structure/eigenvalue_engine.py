"""
Eigenvalue Engine

Eigenvalue decomposition with Marchenko-Pastur significance testing.
Separates signal from noise in correlation structure.
"""

import numpy as np
import polars as pl
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..primitives.matrix import (
    correlation_matrix,
    eigendecomposition,
    pca_loadings,
)
from ..primitives.tests import marchenko_pastur_test


@dataclass
class EigenvalueConfig:
    """Configuration for eigenvalue engine."""
    min_samples: int = 30
    significance_level: float = 0.05
    n_components: Optional[int] = None  # None = all components


class EigenvalueEngine:
    """
    Eigenvalue Decomposition Engine.

    Performs eigenvalue decomposition on correlation matrix and tests
    significance against Marchenko-Pastur (random matrix) distribution.

    Outputs:
    - eigenvalues: All eigenvalues (sorted descending)
    - eigenvectors: Corresponding eigenvectors
    - n_significant: Number of eigenvalues above MP threshold
    - explained_variance_ratio: Fraction of variance explained
    - participation_ratio: Effective dimensionality
    - spectral_entropy: Entropy of eigenvalue distribution
    - mp_threshold: Marchenko-Pastur upper bound
    - tracy_widom_pvalue: P-value for largest eigenvalue
    """

    ENGINE_TYPE = "structure"

    def __init__(self, config: Optional[EigenvalueConfig] = None):
        self.config = config or EigenvalueConfig()

    def compute(
        self,
        signals: Dict[str, np.ndarray],
        unit_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Compute eigenvalue decomposition for an entity.

        Parameters
        ----------
        signals : dict
            Dictionary mapping signal_id to numpy array of values
        unit_id : str
            Entity identifier

        Returns
        -------
        dict
            Eigenvalue metrics and decomposition
        """
        if len(signals) < 2:
            return self._empty_result(unit_id)

        # Build data matrix
        signal_ids = sorted(signals.keys())
        min_len = min(len(signals[s]) for s in signal_ids)

        if min_len < self.config.min_samples:
            return self._empty_result(unit_id)

        data = np.column_stack([
            np.asarray(signals[s])[:min_len] for s in signal_ids
        ])

        # Remove NaN rows
        valid_rows = ~np.any(np.isnan(data), axis=1)
        data = data[valid_rows]

        if len(data) < self.config.min_samples:
            return self._empty_result(unit_id)

        n_samples, n_features = data.shape

        # Compute correlation matrix
        corr_mat = correlation_matrix(data)

        # Eigendecomposition of correlation matrix
        eigenvalues, eigenvectors = eigendecomposition(corr_mat)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Marchenko-Pastur test
        significant_mask, mp_upper = marchenko_pastur_test(eigenvalues, n_samples, n_features)
        n_significant = int(np.sum(significant_mask))

        # Explained variance
        total_var = np.sum(eigenvalues)
        if total_var > 0:
            explained_ratio = eigenvalues / total_var
            cumulative_ratio = np.cumsum(explained_ratio)
        else:
            explained_ratio = np.zeros_like(eigenvalues)
            cumulative_ratio = np.zeros_like(eigenvalues)

        # Participation ratio (effective dimensionality)
        # PR = (Σλ)² / Σλ² = 1 / Σp²  where p = λ/Σλ
        if total_var > 0:
            p = eigenvalues / total_var
            participation_ratio = 1.0 / np.sum(p ** 2)
        else:
            participation_ratio = 0.0

        # Spectral entropy
        # H = -Σ p log(p)
        if total_var > 0:
            p = eigenvalues / total_var
            p = p[p > 1e-10]  # Remove zeros
            spectral_entropy = -np.sum(p * np.log(p))
        else:
            spectral_entropy = 0.0

        # Tracy-Widom p-value for largest eigenvalue
        # Simplified approximation
        tw_pvalue = self._tracy_widom_pvalue(
            eigenvalues[0], n_samples, n_features
        )

        # PCA loadings for significant components
        n_comp = min(n_significant, len(eigenvalues)) if n_significant > 0 else 1
        loadings = pca_loadings(data, n_comp)

        return {
            'unit_id': unit_id,
            'n_signals': n_features,
            'n_samples': n_samples,
            'signal_ids': signal_ids,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'explained_variance_ratio': explained_ratio,
            'cumulative_variance_ratio': cumulative_ratio,
            'n_significant': n_significant,
            'mp_threshold': mp_upper,
            'significant_mask': significant_mask,
            'participation_ratio': float(participation_ratio),
            'spectral_entropy': float(spectral_entropy),
            'tracy_widom_pvalue': tw_pvalue,
            'loadings': loadings,
        }

    def _tracy_widom_pvalue(
        self,
        largest_eigenvalue: float,
        n_samples: int,
        n_features: int
    ) -> float:
        """
        Approximate Tracy-Widom p-value for largest eigenvalue.

        Uses the standardized largest eigenvalue under null hypothesis
        of uncorrelated Gaussian data.
        """
        # Under null: largest eigenvalue follows Tracy-Widom distribution
        # Centering and scaling parameters
        gamma = n_features / n_samples
        mu = (1 + np.sqrt(gamma)) ** 2
        sigma = (1 + np.sqrt(gamma)) * (1/np.sqrt(n_samples) + 1/np.sqrt(n_features)) ** (1/3)

        if sigma <= 0:
            return 1.0

        # Standardized statistic
        s = (largest_eigenvalue - mu) / sigma

        # Tracy-Widom approximation using Gaussian tail
        # This is a rough approximation; exact TW requires special functions
        from scipy import stats
        # TW1 has heavier right tail than Gaussian
        # Use shifted/scaled Gaussian as approximation
        p_value = 1 - stats.norm.cdf(s)

        return float(np.clip(p_value, 0, 1))

    def _empty_result(self, unit_id: str) -> Dict[str, Any]:
        """Return empty result for insufficient data."""
        return {
            'unit_id': unit_id,
            'n_signals': 0,
            'n_samples': 0,
            'signal_ids': [],
            'eigenvalues': np.array([]),
            'eigenvectors': np.array([[]]),
            'explained_variance_ratio': np.array([]),
            'cumulative_variance_ratio': np.array([]),
            'n_significant': 0,
            'mp_threshold': np.nan,
            'significant_mask': np.array([]),
            'participation_ratio': np.nan,
            'spectral_entropy': np.nan,
            'tracy_widom_pvalue': np.nan,
            'loadings': np.array([[]]),
        }

    def to_parquet_row(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert result to flat row for parquet output."""
        # Store top eigenvalues as separate columns
        eigenvalues = result['eigenvalues']
        ev_dict = {}
        for i in range(min(10, len(eigenvalues))):
            ev_dict[f'eigenvalue_{i+1}'] = float(eigenvalues[i])

        return {
            'unit_id': result['unit_id'],
            'n_signals': result['n_signals'],
            'n_samples': result['n_samples'],
            'n_significant': result['n_significant'],
            'mp_threshold': result['mp_threshold'],
            'participation_ratio': result['participation_ratio'],
            'spectral_entropy': result['spectral_entropy'],
            'tracy_widom_pvalue': result['tracy_widom_pvalue'],
            **ev_dict,
        }
