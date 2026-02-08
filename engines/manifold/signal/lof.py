"""
LOF Engine.

Local Outlier Factor for anomaly detection in phase space.
"""

import numpy as np
from typing import Dict


def compute(y: np.ndarray, n_neighbors: int = 20, embedding_dim: int = 3, delay: int = 1) -> Dict[str, float]:
    """
    Compute Local Outlier Factor scores.

    Args:
        y: Signal values
        n_neighbors: Number of neighbors for LOF
        embedding_dim: Embedding dimension for phase space
        delay: Time delay for embedding (default: 1)

    Returns:
        dict with lof_max, lof_mean, lof_std, outlier_fraction, n_outliers
    """
    result = {
        'lof_max': np.nan,
        'lof_mean': np.nan,
        'lof_std': np.nan,
        'outlier_fraction': np.nan,
        'n_outliers': 0
    }

    # Handle NaN values
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    # Need enough points for embedding + neighbors
    min_points = (embedding_dim - 1) * delay + n_neighbors * 2 + 10
    if n < min_points:
        return result

    # Check for constant signal
    if np.std(y) < 1e-10:
        result['lof_max'] = 1.0
        result['lof_mean'] = 1.0
        result['lof_std'] = 0.0
        result['outlier_fraction'] = 0.0
        return result

    try:
        from sklearn.neighbors import LocalOutlierFactor

        # Create time-delay embedding
        m = n - (embedding_dim - 1) * delay
        if m < n_neighbors + 1:
            return result

        X = np.zeros((m, embedding_dim))
        for i in range(embedding_dim):
            X[:, i] = y[i*delay:i*delay+m]

        # Normalize embedding for better LOF performance
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std < 1e-10] = 1.0
        X_norm = (X - X_mean) / X_std

        # Fit LOF
        k = min(n_neighbors, m - 1)
        lof = LocalOutlierFactor(n_neighbors=k, contamination='auto', novelty=False)
        labels = lof.fit_predict(X_norm)

        # Get LOF scores (higher = more anomalous)
        # negative_outlier_factor_ is negative, so negate it
        scores = -lof.negative_outlier_factor_

        # Count outliers (labels == -1)
        n_outliers = int(np.sum(labels == -1))
        outlier_fraction = n_outliers / len(labels)

        result = {
            'lof_max': float(np.max(scores)),
            'lof_mean': float(np.mean(scores)),
            'lof_std': float(np.std(scores)),
            'outlier_fraction': float(outlier_fraction),
            'n_outliers': n_outliers
        }

    except ImportError:
        # sklearn not available - use simple distance-based fallback
        try:
            # Create embedding
            m = n - (embedding_dim - 1) * delay
            X = np.zeros((m, embedding_dim))
            for i in range(embedding_dim):
                X[:, i] = y[i*delay:i*delay+m]

            # Simple outlier detection: points far from centroid
            centroid = np.mean(X, axis=0)
            distances = np.sqrt(np.sum((X - centroid) ** 2, axis=1))

            threshold = np.mean(distances) + 2 * np.std(distances)
            n_outliers = int(np.sum(distances > threshold))

            result = {
                'lof_max': float(np.max(distances)),
                'lof_mean': float(np.mean(distances)),
                'lof_std': float(np.std(distances)),
                'outlier_fraction': float(n_outliers / len(distances)),
                'n_outliers': n_outliers
            }

        except Exception:
            pass

    except Exception:
        pass

    return result
