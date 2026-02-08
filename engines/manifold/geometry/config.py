"""
ENGINES Geometry Engine Configuration

Single source of truth for feature groups used across geometry engines.
This replaces duplicated DEFAULT_FEATURE_GROUPS in:
- state_geometry.py
- state_vector.py
- signal_geometry.py
- signal_pairwise.py

Feature names MUST match actual signal_vector.parquet columns.
"""

from typing import Dict, List

# ============================================================
# DEFAULT FEATURE GROUPS
# ============================================================
# Feature names must match signal_vector output columns exactly.
# See signal_vector.parquet schema for available features.

DEFAULT_FEATURE_GROUPS: Dict[str, List[str]] = {
    # Shape: Distribution characteristics
    'shape': ['kurtosis', 'skewness', 'crest_factor'],

    # Complexity: Predictability and memory
    'complexity': ['permutation_entropy', 'hurst', 'acf_lag1'],

    # Spectral: Frequency domain characteristics
    'spectral': [
        'spectral_entropy',
        'spectral_centroid',
        'band_low_rel',
        'band_mid_rel',
        'band_high_rel'
    ],
}

# Fallback features if no groups match available columns
FALLBACK_FEATURES: List[str] = ['kurtosis', 'skewness', 'crest_factor']

# Minimum features required per group
MIN_FEATURES_PER_GROUP: int = 2


def get_available_feature_groups(
    available_columns: List[str],
    custom_groups: Dict[str, List[str]] = None
) -> Dict[str, List[str]]:
    """
    Get feature groups filtered to available columns.

    Args:
        available_columns: List of available column names
        custom_groups: Optional custom feature groups (overrides defaults)

    Returns:
        Dict mapping group names to lists of available features
    """
    groups = custom_groups if custom_groups else DEFAULT_FEATURE_GROUPS
    result = {}

    for name, features in groups.items():
        available = [f for f in features if f in available_columns]
        if len(available) >= MIN_FEATURES_PER_GROUP:
            result[name] = available

    # If no groups available, use fallback
    if not result and available_columns:
        fallback = [f for f in FALLBACK_FEATURES if f in available_columns]
        if len(fallback) >= MIN_FEATURES_PER_GROUP:
            result['full'] = fallback
        elif len(available_columns) >= MIN_FEATURES_PER_GROUP:
            result['full'] = available_columns[:3]

    return result
