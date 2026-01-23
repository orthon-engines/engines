"""
Ørthon Signal Typology
======================

Signal Typology classifies time series through measurement across
**six orthogonal axes** combined with **structural discontinuity detection**.

The Six Orthogonal Axes:
    1. Memory        - Long-range dependence (Hurst, ACF decay, spectral slope)
    2. Information   - Complexity (permutation entropy, sample entropy)
    3. Recurrence    - Pattern structure (RQA: determinism, laminarity, trapping)
    4. Volatility    - Amplitude dynamics (GARCH persistence, Hilbert envelope)
    5. Frequency     - Spectral character (centroid, bandwidth, 1/f character)
    6. Dynamics      - Stability (Lyapunov exponent, embedding dimension)

Plus: Structural Discontinuity Detection
    - Dirac (impulse): Transient shocks that decay
    - Heaviside (step): Permanent level shifts

Key Insight:
    "When axes measure different things, disagreement isn't noise — it's discovery.
    The fingerprint combination reveals what the signal actually IS."

Usage:
    >>> from prism.typology import analyze_signal, analyze_windowed
    >>>
    >>> # Quick analysis
    >>> typology = analyze_signal(my_series)
    >>> print(typology.archetype, typology.confidence)
    >>>
    >>> # Windowed analysis with transition detection
    >>> typologies = analyze_windowed(my_series, window_size=50, step_size=10)
    >>> for t in typologies:
    ...     if t.regime_transition.value != "none":
    ...         print(f"Warning: {t.archetype}: {t.summary}")

Academic Advancement:
    Traditional approach: Single metric (Hurst) → single label
    Ørthon approach: 6 orthogonal axes + discontinuity → fingerprint → archetype

    This enables:
    - Multi-dimensional regime characterization
    - Early warning via axis divergence
    - Differential diagnosis ("what changed, what didn't")
    - Structural discontinuity as first-class citizen
"""

__version__ = "0.1.0"
__author__ = "Ørthon Project"

# Core analysis functions
from .analyzer import (
    analyze_signal,
    analyze_windowed,
    analyze_batch,
    compare_typologies,
    find_regime_changes,
    quick_typology,
    typologies_to_dataframe,
    typologies_to_parquet
)

# Data models
from .models import (
    SignalTypology,
    MemoryAxis,
    InformationAxis,
    RecurrenceAxis,
    VolatilityAxis,
    FrequencyAxis,
    DynamicsAxis,
    StructuralDiscontinuity,
    DiracDiscontinuity,
    HeavisideDiscontinuity,
    Archetype,
    # Enums
    MemoryClass,
    InformationClass,
    RecurrenceClass,
    VolatilityClass,
    FrequencyClass,
    DynamicsClass,
    ACFDecayType,
    TransitionType
)

# Archetype library
from .archetypes import (
    ARCHETYPES,
    match_archetype,
    compute_fingerprint,
    diagnose_differential,
    generate_summary,
    compute_confidence,
    compute_boundary_proximity,
    # Classification helpers
    classify_memory,
    classify_information,
    classify_recurrence,
    classify_volatility,
    classify_frequency,
    classify_dynamics,
)

# Low-level engines (for custom analysis)
from .engines import (
    measure_memory_axis,
    measure_information_axis,
    measure_recurrence_axis,
    measure_volatility_axis,
    measure_frequency_axis,
    measure_dynamics_axis,
    measure_discontinuity,
    # Individual computations
    compute_hurst_dfa,
    compute_hurst_rs,
    compute_permutation_entropy,
    compute_sample_entropy,
    compute_rqa,
    detect_dirac_impulses,
    detect_heaviside_steps
)

__all__ = [
    # Version
    "__version__",

    # Main API
    "analyze_signal",
    "analyze_windowed",
    "analyze_batch",
    "compare_typologies",
    "find_regime_changes",
    "quick_typology",

    # Export
    "typologies_to_dataframe",
    "typologies_to_parquet",

    # Models
    "SignalTypology",
    "MemoryAxis",
    "InformationAxis",
    "RecurrenceAxis",
    "VolatilityAxis",
    "FrequencyAxis",
    "DynamicsAxis",
    "StructuralDiscontinuity",
    "DiracDiscontinuity",
    "HeavisideDiscontinuity",
    "Archetype",

    # Enums
    "MemoryClass",
    "InformationClass",
    "RecurrenceClass",
    "VolatilityClass",
    "FrequencyClass",
    "DynamicsClass",
    "ACFDecayType",
    "TransitionType",

    # Archetypes
    "ARCHETYPES",
    "match_archetype",
    "compute_fingerprint",
    "diagnose_differential",
    "generate_summary",
    "compute_confidence",
    "compute_boundary_proximity",
    "classify_memory",
    "classify_information",
    "classify_recurrence",
    "classify_volatility",
    "classify_frequency",
    "classify_dynamics",

    # Engines
    "measure_memory_axis",
    "measure_information_axis",
    "measure_recurrence_axis",
    "measure_volatility_axis",
    "measure_frequency_axis",
    "measure_dynamics_axis",
    "measure_discontinuity",
    "compute_hurst_dfa",
    "compute_hurst_rs",
    "compute_permutation_entropy",
    "compute_sample_entropy",
    "compute_rqa",
    "detect_dirac_impulses",
    "detect_heaviside_steps"
]
