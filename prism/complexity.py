"""
PRISM Complexity and Entropy Primitives

Pure mathematical functions for measuring signal complexity, entropy,
and information content.
"""

import numpy as np
from typing import Optional, Union
from scipy.spatial.distance import pdist
from collections import Counter
import math


def permutation_entropy(
    values: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True
) -> float:
    """
    Compute permutation entropy of a time series.

    Permutation entropy measures the complexity of a time series by
    looking at the relative order of values in embedding vectors.

    Args:
        values: Input time series
        order: Embedding dimension (length of ordinal patterns)
        delay: Time delay for embedding
        normalize: If True, normalize entropy to [0,1]

    Returns:
        Permutation entropy
    """
    values = np.asarray(values, dtype=np.float64)

    if len(values) < order:
        return 0.0

    # Create embedded vectors
    embedded = []
    for i in range(len(values) - (order - 1) * delay):
        vector = [values[i + j * delay] for j in range(order)]
        embedded.append(vector)

    if len(embedded) == 0:
        return 0.0

    # Convert each vector to ordinal pattern
    ordinal_patterns = []
    for vector in embedded:
        pattern = tuple(np.argsort(np.argsort(vector)))
        ordinal_patterns.append(pattern)

    # Count frequencies of ordinal patterns
    pattern_counts = Counter(ordinal_patterns)

    # Convert to probabilities
    total_patterns = len(ordinal_patterns)
    probabilities = np.array(list(pattern_counts.values())) / total_patterns

    # Shannon entropy
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))

    if normalize:
        max_entropy = np.log2(math.factorial(order))
        if max_entropy > 0:
            entropy = entropy / max_entropy

    return float(entropy)


def sample_entropy(
    values: np.ndarray,
    m: int = 2,
    r: Optional[float] = None,
    relative_tolerance: bool = True
) -> float:
    """
    Compute sample entropy of a time series.

    Sample entropy measures the complexity of a time series by quantifying
    the probability that patterns that are similar remain similar for the
    next point.

    Args:
        values: Input time series
        m: Pattern length
        r: Tolerance for matching (if None, use 0.2 * std(values))
        relative_tolerance: If True, r is relative to signal std

    Returns:
        Sample entropy (higher values indicate more complexity)
    """
    values = np.asarray(values, dtype=np.float64)

    if len(values) < m + 1:
        return 0.0

    if r is None:
        r = 0.2 * np.std(values)
    elif relative_tolerance:
        r = r * np.std(values)

    def _maxdist(xi, xj):
        return max(abs(ua - va) for ua, va in zip(xi, xj))

    def _phi(m_val):
        patterns = [values[i:i + m_val] for i in range(len(values) - m_val + 1)]

        matches = 0
        total_pairs = 0

        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                total_pairs += 1
                if _maxdist(patterns[i], patterns[j]) <= r:
                    matches += 1

        if total_pairs == 0:
            return 0.0

        return matches / total_pairs

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    if phi_m == 0 or phi_m1 == 0:
        return 0.0

    return float(-np.log(phi_m1 / phi_m))


def approximate_entropy(
    values: np.ndarray,
    m: int = 2,
    r: Optional[float] = None,
    relative_tolerance: bool = True
) -> float:
    """
    Compute approximate entropy of a time series.

    Similar to sample entropy but includes self-matches in the calculation.

    Args:
        values: Input time series
        m: Pattern length
        r: Tolerance for matching (if None, use 0.2 * std(values))
        relative_tolerance: If True, r is relative to signal std

    Returns:
        Approximate entropy
    """
    values = np.asarray(values, dtype=np.float64)

    if len(values) < m + 1:
        return 0.0

    if r is None:
        r = 0.2 * np.std(values)
    elif relative_tolerance:
        r = r * np.std(values)

    def _maxdist(xi, xj):
        return max(abs(ua - va) for ua, va in zip(xi, xj))

    def _phi(m_val):
        patterns = [values[i:i + m_val] for i in range(len(values) - m_val + 1)]

        phi_values = []
        for i in range(len(patterns)):
            matches = sum(1 for j in range(len(patterns))
                         if _maxdist(patterns[i], patterns[j]) <= r)
            phi_values.append(matches / len(patterns))

        return np.mean([np.log(phi + 1e-12) for phi in phi_values])

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    return float(phi_m - phi_m1)


def multiscale_entropy(
    values: np.ndarray,
    m: int = 2,
    r: Optional[float] = None,
    max_scale: int = 10,
    entropy_type: str = 'sample'
) -> np.ndarray:
    """
    Compute multiscale entropy across different time scales.

    Args:
        values: Input time series
        m: Pattern length for entropy calculation
        r: Tolerance for matching
        max_scale: Maximum scale factor to compute
        entropy_type: Type of entropy ('sample' or 'approximate')

    Returns:
        Array of entropy values at each scale
    """
    values = np.asarray(values, dtype=np.float64)
    entropies = []

    for scale in range(1, max_scale + 1):
        # Coarse-grain the time series
        if scale == 1:
            coarse_grained = values
        else:
            n_windows = len(values) // scale
            if n_windows == 0:
                entropies.append(0.0)
                continue

            coarse_grained = np.array([
                np.mean(values[i*scale:(i+1)*scale])
                for i in range(n_windows)
            ])

        # Compute entropy at this scale
        if entropy_type == 'sample':
            ent = sample_entropy(coarse_grained, m, r)
        elif entropy_type == 'approximate':
            ent = approximate_entropy(coarse_grained, m, r)
        else:
            raise ValueError(f"Unknown entropy type: {entropy_type}")

        entropies.append(ent)

    return np.array(entropies)


def lempel_ziv_complexity(
    values: np.ndarray,
    threshold: Optional[float] = None,
    normalize: bool = True
) -> float:
    """
    Compute Lempel-Ziv complexity of a time series.

    First binarizes the signal, then computes the number of distinct
    subsequences using the Lempel-Ziv algorithm.

    Args:
        values: Input time series
        threshold: Threshold for binarization (if None, use median)
        normalize: If True, normalize by theoretical maximum

    Returns:
        Lempel-Ziv complexity
    """
    values = np.asarray(values, dtype=np.float64)

    if len(values) < 2:
        return 0.0

    # Binarize the signal
    if threshold is None:
        threshold = np.median(values)

    binary_string = ''.join(['1' if x >= threshold else '0' for x in values])

    # Lempel-Ziv algorithm
    i = 0
    complexity = 1

    while i < len(binary_string):
        substring = binary_string[i]
        j = i + 1

        while j <= len(binary_string):
            if binary_string[:i].find(substring) == -1:
                break
            if j < len(binary_string):
                substring += binary_string[j]
            j += 1

        complexity += 1
        i = j - 1

        if i >= len(binary_string):
            break

    if normalize:
        n = len(binary_string)
        max_complexity = n / np.log2(n) if n > 1 else 1
        complexity = complexity / max_complexity

    return float(complexity)


def fractal_dimension(
    values: np.ndarray,
    method: str = 'box_counting'
) -> float:
    """
    Estimate fractal dimension of a time series.

    Args:
        values: Input time series
        method: Method to use ('box_counting', 'correlation_dimension')

    Returns:
        Estimated fractal dimension
    """
    if method == 'box_counting':
        return _box_counting_dimension(values)
    elif method == 'correlation_dimension':
        return _correlation_dimension(values)
    else:
        raise ValueError(f"Unknown method: {method}")


def _box_counting_dimension(values: np.ndarray) -> float:
    """Estimate fractal dimension using box counting method."""
    values = np.asarray(values, dtype=np.float64)

    # Normalize signal to [0, 1]
    val_range = np.max(values) - np.min(values)
    if val_range == 0:
        return 1.0

    normalized = (values - np.min(values)) / val_range

    # Try different box sizes
    box_sizes = [2**(-k) for k in range(1, 8)]
    box_counts = []

    for box_size in box_sizes:
        n_boxes = int(1.0 / box_size) + 1
        boxes_occupied = set()

        for i, y in enumerate(normalized):
            x_box = int(i / len(normalized) * n_boxes)
            y_box = int(y * n_boxes)
            boxes_occupied.add((x_box, y_box))

        box_counts.append(len(boxes_occupied))

    if len(box_counts) < 2:
        return 1.0

    # Fit line to log-log plot
    log_sizes = np.log(box_sizes)
    log_counts = np.log(box_counts)

    slope, _ = np.polyfit(log_sizes, log_counts, 1)

    return float(-slope)


def _correlation_dimension(values: np.ndarray, embed_dim: int = 5) -> float:
    """Estimate correlation dimension using embedding."""
    values = np.asarray(values, dtype=np.float64)

    if len(values) < embed_dim * 2:
        return 1.0

    # Create embedding
    embedded = np.array([
        values[i:i + embed_dim]
        for i in range(len(values) - embed_dim + 1)
    ])

    # Compute pairwise distances
    distances = pdist(embedded, metric='euclidean')

    if len(distances) == 0:
        return 1.0

    # Try different radius values
    max_dist = np.max(distances)
    radii = np.logspace(np.log10(max_dist/100), np.log10(max_dist/2), 10)
    correlations = []

    for r in radii:
        count = np.sum(distances < r)
        total_pairs = len(distances)
        correlation = count / total_pairs if total_pairs > 0 else 0
        correlations.append(correlation)

    # Remove zeros for log calculation
    valid_indices = np.array(correlations) > 0
    if np.sum(valid_indices) < 2:
        return 1.0

    valid_radii = radii[valid_indices]
    valid_correlations = np.array(correlations)[valid_indices]

    # Fit line in log-log space
    log_radii = np.log(valid_radii)
    log_correlations = np.log(valid_correlations)

    slope, _ = np.polyfit(log_radii, log_correlations, 1)

    return float(max(0.1, slope))


def entropy_rate(
    values: np.ndarray,
    block_length: int = 3
) -> float:
    """
    Compute entropy rate of a time series.

    Args:
        values: Input time series
        block_length: Length of blocks to analyze

    Returns:
        Entropy rate (bits per symbol)
    """
    values = np.asarray(values, dtype=np.float64)

    if len(values) < block_length:
        return 0.0

    # Discretize signal if continuous
    if np.issubdtype(values.dtype, np.floating):
        val_range = np.max(values) - np.min(values)
        if val_range == 0:
            return 0.0
        quantized = np.floor(8 * (values - np.min(values)) / (val_range + 1e-12))
        quantized = np.clip(quantized, 0, 7).astype(int)
        values = quantized

    # Extract blocks
    blocks = [
        tuple(values[i:i + block_length])
        for i in range(len(values) - block_length + 1)
    ]

    # Count block frequencies
    block_counts = Counter(blocks)
    total_blocks = len(blocks)

    # Compute entropy of blocks
    block_probs = np.array(list(block_counts.values())) / total_blocks
    block_entropy = -np.sum(block_probs * np.log2(block_probs + 1e-12))

    return float(block_entropy / block_length)
