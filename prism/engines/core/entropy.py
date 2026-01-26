"""
Entropy Measures
================

Information-theoretic complexity measures:
- Sample Entropy (SampEn)
- Permutation Entropy (PE)
- Entropy Rate

Stream mode: Accumulate signal, compute when complete.
"""

import numpy as np
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


def compute(y: np.ndarray, method: str = 'sample', **kwargs) -> Dict[str, Any]:
    """
    Compute entropy measure.
    
    Args:
        y: Signal array
        method: 'sample' or 'permutation'
    """
    if method == 'sample':
        return compute_sample_entropy(y, **kwargs)
    elif method == 'permutation':
        return compute_permutation_entropy(y, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
