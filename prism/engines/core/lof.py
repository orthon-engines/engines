"""
Local Outlier Factor (LOF) Engine

Detects anomalous signals using density-based analysis.

CANONICAL INTERFACE:
    def compute(observations: pd.DataFrame) -> pd.DataFrame
    Input:  [entity_id, signal_id, I, y]
    Output: [entity_id, signal_id, lof_score, is_outlier]

LOF compares the local density of a point to the local densities
of its neighbors. Points with substantially lower density than
their neighbors are considered outliers.
"""

ENGINE_TYPE = 'signal'

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any


def compute(
    observations: pd.DataFrame,
    n_neighbors: int = 5,
    contamination: str = "auto",
    max_lof: float = 10.0,
) -> pd.DataFrame:
    """
    Compute Local Outlier Factor for all signals within each entity.

    CANONICAL INTERFACE:
        Input:  observations [entity_id, signal_id, I, y]
        Output: primitives [entity_id, signal_id, lof_score, is_outlier]

    Parameters
    ----------
    observations : pd.DataFrame
        Must have columns: entity_id, signal_id, I, y
    n_neighbors : int, optional
        Number of neighbors for LOF (default: 5)
    contamination : str, optional
        Contamination parameter (default: "auto")
    max_lof : float, optional
        Maximum LOF value to clip to (default: 10.0)

    Returns
    -------
    pd.DataFrame
        LOF scores per signal
    """
    results = []

    for entity_id, entity_group in observations.groupby('entity_id'):
        # Pivot to wide format: rows=I (time), cols=signal_id, values=y
        try:
            wide = entity_group.pivot(index='I', columns='signal_id', values='y')
            wide = wide.sort_index().dropna()
        except Exception:
            wide = entity_group.groupby(['I', 'signal_id'])['y'].mean().unstack()
            wide = wide.sort_index().dropna()

        signals = list(wide.columns)
        n_signals = len(signals)

        if n_signals < 3 or len(wide) < 10:
            for signal_id in signals:
                results.append({
                    'entity_id': entity_id,
                    'signal_id': signal_id,
                    'lof_score': np.nan,
                    'is_outlier': False,
                })
            continue

        try:
            # Transpose: each signal is a row, time points are features
            X = wide.T.values  # (n_signals, n_timepoints)

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Filter constant features
            feature_std = np.std(X_scaled, axis=0)
            valid_features = feature_std > 1e-10
            if valid_features.sum() < 2:
                for signal_id in signals:
                    results.append({
                        'entity_id': entity_id,
                        'signal_id': signal_id,
                        'lof_score': 1.0,
                        'is_outlier': False,
                    })
                continue

            X_filtered = X_scaled[:, valid_features]

            # Adjust neighbors if needed
            effective_neighbors = min(n_neighbors, n_signals - 1)
            if effective_neighbors < 2:
                for signal_id in signals:
                    results.append({
                        'entity_id': entity_id,
                        'signal_id': signal_id,
                        'lof_score': 1.0,
                        'is_outlier': False,
                    })
                continue

            # Compute LOF
            lof = LocalOutlierFactor(
                n_neighbors=effective_neighbors,
                contamination=contamination if contamination != "auto" else "auto",
                novelty=False,
            )
            labels = lof.fit_predict(X_filtered)
            raw_scores = -lof.negative_outlier_factor_

            # Bound scores
            lof_scores = np.clip(raw_scores, 0.0, max_lof)
            lof_scores = np.where(np.isfinite(lof_scores), lof_scores, max_lof)

            for i, signal_id in enumerate(signals):
                results.append({
                    'entity_id': entity_id,
                    'signal_id': signal_id,
                    'lof_score': float(lof_scores[i]),
                    'is_outlier': labels[i] == -1,
                })

        except Exception:
            for signal_id in signals:
                results.append({
                    'entity_id': entity_id,
                    'signal_id': signal_id,
                    'lof_score': np.nan,
                    'is_outlier': False,
                })

    return pd.DataFrame(results)
