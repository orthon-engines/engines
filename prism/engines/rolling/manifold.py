"""
Manifold Engine.

Computes PCA projection at every time slice (cross-signal).
All parameters from manifest via params dict.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.decomposition import PCA


def compute(observations: pd.DataFrame, params: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Compute manifold projection at each time slice.

    Args:
        observations: DataFrame with unit_id, signal_id, I, value columns
        params: Parameters from manifest:
            - n_components: Number of PCA components (default: 3)

    Returns:
        DataFrame with manifold_x, manifold_y, manifold_z, manifold_velocity, manifold_acceleration
    """
    params = params or {}
    n_components = params.get('n_components', 3)

    results = []

    for unit_id, group in observations.groupby('unit_id'):
        pivot = group.pivot(index='I', columns='signal_id', values='value').dropna()
        if len(pivot) < 10 or len(pivot.columns) < 2:
            continue

        n_comp = min(n_components, len(pivot.columns))
        pca = PCA(n_components=n_comp)
        projected = pca.fit_transform(pivot.values)

        # Pad to 3 components
        if projected.shape[1] < 3:
            pad = np.zeros((projected.shape[0], 3 - projected.shape[1]))
            projected = np.hstack([projected, pad])

        # Manifold velocity
        velocity = np.zeros(len(projected))
        velocity[1:] = np.sqrt(np.sum(np.diff(projected, axis=0)**2, axis=1))

        # Manifold acceleration
        acceleration = np.zeros(len(projected))
        acceleration[1:] = np.diff(velocity)

        for i, I_val in enumerate(pivot.index):
            results.append({
                'unit_id': unit_id,
                'I': I_val,
                'manifold_x': projected[i, 0],
                'manifold_y': projected[i, 1],
                'manifold_z': projected[i, 2],
                'manifold_velocity': velocity[i],
                'manifold_acceleration': acceleration[i],
            })

    return pd.DataFrame(results)
