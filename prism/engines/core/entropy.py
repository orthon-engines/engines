"""
Entropy Measures
================

Information-theoretic complexity measures:
- Sample Entropy (SampEn)
- Permutation Entropy (PE)
- Entropy Rate

CANONICAL INTERFACE:
    def compute(observations: pd.DataFrame) -> pd.DataFrame
    Input:  [entity_id, signal_id, I, y]
    Output: [entity_id, signal_id, sample_entropy, permutation_entropy]
"""

ENGINE_TYPE = 'signal'

import numpy as np
import pandas as pd
from typing import Dict, Any
from math import factorial


def compute_sample_entropy(y: np.ndarray, m: int = 2, r: float = 0.2) -> Dict[str, Any]:
    """
    Sample Entropy (SampEn) - template matching complexity.
    
    Args:
        y: Signal array
        m: Embedding dimension
        r: Tolerance (as fraction of std)
    
    Returns:
        sample_entropy: SampEn value (higher = more complex)
    """
    y = np.asarray(y).flatten()
    n = len(y)
    
    if n < m + 2:
        return {'sample_entropy': np.nan}
    
    # Tolerance as fraction of std
    tolerance = r * np.std(y, ddof=1)
    if tolerance == 0:
        return {'sample_entropy': 0.0}
    
    def count_matches(dim):
        count = 0
        templates = np.array([y[i:i + dim] for i in range(n - dim)])
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) < tolerance:
                    count += 1
        return count
    
    A = count_matches(m + 1)
    B = count_matches(m)
    
    if B == 0:
        return {'sample_entropy': np.nan}
    
    return {'sample_entropy': float(-np.log(A / B)) if A > 0 else np.inf}


def compute_permutation_entropy(y: np.ndarray, order: int = 3, delay: int = 1) -> Dict[str, Any]:
    """
    Permutation Entropy - ordinal pattern complexity.
    
    Args:
        y: Signal array
        order: Embedding dimension (pattern length)
        delay: Time delay
    
    Returns:
        permutation_entropy: PE value (normalized to [0, 1])
    """
    y = np.asarray(y).flatten()
    n = len(y)
    
    if n < order * delay:
        return {'permutation_entropy': np.nan}
    
    # Extract ordinal patterns
    n_patterns = n - (order - 1) * delay
    patterns = {}
    
    for i in range(n_patterns):
        indices = [i + j * delay for j in range(order)]
        values = y[indices]
        pattern = tuple(np.argsort(values))
        patterns[pattern] = patterns.get(pattern, 0) + 1
    
    # Compute entropy
    probs = np.array(list(patterns.values())) / n_patterns
    entropy = -np.sum(probs * np.log2(probs))
    
    # Normalize
    max_entropy = np.log2(factorial(order))
    normalized = entropy / max_entropy if max_entropy > 0 else 0
    
    return {
        'permutation_entropy': float(normalized),
        'n_patterns': len(patterns)
    }


def _compute_array(y: np.ndarray, method: str = 'sample', **kwargs) -> Dict[str, Any]:
    """Internal: compute entropy from numpy array."""
    if method == 'sample':
        return compute_sample_entropy(y, **kwargs)
    elif method == 'permutation':
        return compute_permutation_entropy(y, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute(observations: pd.DataFrame, method: str = 'both', **kwargs) -> pd.DataFrame:
    """
    Compute entropy measures.

    CANONICAL INTERFACE:
        Input:  observations [entity_id, signal_id, I, y]
        Output: primitives [entity_id, signal_id, sample_entropy, permutation_entropy]

    Args:
        observations: DataFrame with columns [entity_id, signal_id, I, y]
        method: 'sample', 'permutation', or 'both' (default)

    Returns:
        DataFrame with entropy values per entity/signal
    """
    results = []

    for (entity_id, signal_id), group in observations.groupby(['entity_id', 'signal_id']):
        y = group.sort_values('I')['y'].values

        row = {
            'entity_id': entity_id,
            'signal_id': signal_id,
        }

        try:
            if method in ('sample', 'both'):
                result = compute_sample_entropy(y, **kwargs)
                row['sample_entropy'] = result.get('sample_entropy', np.nan)

            if method in ('permutation', 'both'):
                result = compute_permutation_entropy(y, **kwargs)
                row['permutation_entropy'] = result.get('permutation_entropy', np.nan)

        except Exception:
            if method in ('sample', 'both'):
                row['sample_entropy'] = np.nan
            if method in ('permutation', 'both'):
                row['permutation_entropy'] = np.nan

        results.append(row)

    return pd.DataFrame(results)
