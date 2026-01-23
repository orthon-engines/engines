"""
prism/modules/ - Pure computation modules

No I/O, no side effects. Just computation.

Pipeline modules (research-facing names):
    signal_behavior:      Compute all engine metrics (was: vector)
    typology_classifier:  Classification and interpretation
    geometry:             Structural relationships
    state:                Temporal dynamics
    discovery:            Cohort/system discovery

Usage:
    from prism.modules.signal_behavior import compute_all_metrics
    from prism.modules.typology_classifier import classify, generate_summary
    from prism.modules.geometry import compute_geometry_features
    from prism.modules.state import compute_state_features
    from prism.modules.discovery import discover_cohorts
"""

# New typology modules
from prism.modules.signal_behavior import (
    compute_engines,
    compute_all_metrics,
    CORE_ENGINES,
    CONDITIONAL_ENGINES,
    DISCONTINUITY_ENGINES,
    ALL_ENGINES,
)
from prism.modules.typology_classifier import (
    classify,
    generate_summary,
    generate_summary_text,
    detect_shifts,
    compute_typology,
    TypologyResult,
    ShiftResult,
)

# Existing modules
from prism.modules.geometry import compute_geometry_features
from prism.modules.state import compute_state_features
from prism.modules.discovery import discover_cohorts, compute_cohort_summary

# Legacy alias (backwards compatibility)
from prism.modules.vector import compute_vector_features

__all__ = [
    # Signal behavior (new)
    "compute_engines",
    "compute_all_metrics",
    "CORE_ENGINES",
    "CONDITIONAL_ENGINES",
    "DISCONTINUITY_ENGINES",
    "ALL_ENGINES",

    # Typology classifier (new)
    "classify",
    "generate_summary",
    "generate_summary_text",
    "detect_shifts",
    "compute_typology",
    "TypologyResult",
    "ShiftResult",

    # Existing
    "compute_geometry_features",
    "compute_state_features",
    "discover_cohorts",
    "compute_cohort_summary",

    # Legacy
    "compute_vector_features",
]
