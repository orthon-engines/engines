"""
ENGINES Geometry Engines Configuration

Single source of truth for feature groups across all geometry engines.
"""

from .config import (
    DEFAULT_FEATURE_GROUPS,
    FALLBACK_FEATURES,
    MIN_FEATURES_PER_GROUP,
    get_available_feature_groups,
)

__all__ = [
    'DEFAULT_FEATURE_GROUPS',
    'FALLBACK_FEATURES',
    'MIN_FEATURES_PER_GROUP',
    'get_available_feature_groups',
]
