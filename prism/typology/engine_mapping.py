"""
Engine Mapping
==============

Maps Signal Typology axis scores to recommended analysis engines.

The 6 Orthogonal Axes:
    1. Memory        - Temporal persistence
    2. Periodicity   - Cyclical structure
    3. Volatility    - Variance dynamics
    4. Discontinuity - Level shifts (Heaviside)
    5. Impulsivity   - Shocks (Dirac)
    6. Complexity    - Predictability (entropy)

Usage:
    from prism.typology.engine_mapping import select_engines

    axis_scores = {
        'memory': 0.7,
        'periodicity': 0.2,
        'volatility': 0.8,
        'discontinuity': 0.4,
        'impulsivity': 0.1,
        'complexity': 0.5,
    }

    engines = select_engines(axis_scores)
    # ['granger', 'var', 'garch', 'ewma', 'changepoint', ...]
"""

from typing import Dict, List, Tuple


# =============================================================================
# THRESHOLDS
# =============================================================================

# Engine activation thresholds
# Lower thresholds for discontinuity/impulsivity - structural events are important
THRESHOLDS = {
    'memory': 0.5,
    'periodicity': 0.5,
    'volatility': 0.5,
    'discontinuity': 0.3,   # Lower - breaks are critical
    'impulsivity': 0.3,     # Lower - shocks are critical
    'complexity': 0.5,
}

# High-confidence thresholds (for primary classification)
HIGH_THRESHOLDS = {
    'memory': 0.7,
    'periodicity': 0.7,
    'volatility': 0.7,
    'discontinuity': 0.5,
    'impulsivity': 0.5,
    'complexity': 0.7,
}


# =============================================================================
# ENGINE RECOMMENDATIONS
# =============================================================================

# Single-axis engine recommendations
ENGINE_MAP = {
    'memory': [
        'granger',           # Granger causality (requires memory)
        'var',               # Vector autoregression
        'vecm',              # Vector error correction
        'arima',             # ARIMA modeling
        'acf_analysis',      # Autocorrelation analysis
    ],
    'periodicity': [
        'wavelet',           # Wavelet decomposition
        'fourier',           # Fourier analysis
        'spectral',          # Spectral analysis
        'seasonal_decompose', # Seasonal decomposition
        'harmonic_regression',
    ],
    'volatility': [
        'garch',             # GARCH modeling
        'ewma',              # Exponential weighted moving average
        'rolling_correlation',
        'realized_volatility',
        'bipower_variation',
    ],
    'discontinuity': [
        'changepoint',       # Changepoint detection (PELT, BOCPD)
        'regime_hmm',        # Hidden Markov Model regime detection
        'structural_break',  # Structural break tests (Chow, CUSUM)
        'level_shift',       # Level shift detection
    ],
    'impulsivity': [
        'outlier_detection', # Outlier/anomaly detection
        'event_study',       # Event study analysis
        'impulse_response',  # Impulse response functions
        'spike_detection',   # Spike detection
        'kurtosis_analysis',
    ],
    'complexity': [
        'entropy',           # Entropy measures
        'lyapunov',          # Lyapunov exponent
        'recurrence_plot',   # Recurrence quantification
        'permutation_entropy',
        'sample_entropy',
    ],
}

# Compound engine recommendations (when multiple axes are active)
COMPOUND_ENGINES = {
    # Memory + Periodicity: time-frequency analysis
    ('memory', 'periodicity'): [
        'wavelet_coherence',
        'cross_spectral',
        'dynamic_time_warping',
    ],

    # Discontinuity + Volatility: regime-switching models
    ('discontinuity', 'volatility'): [
        'regime_switching_garch',
        'markov_switching',
        'threshold_garch',
    ],

    # Impulsivity + Memory: self-exciting processes
    ('impulsivity', 'memory'): [
        'hawkes_process',
        'self_exciting_point_process',
    ],

    # Periodicity + Discontinuity: seasonal structural breaks
    ('periodicity', 'discontinuity'): [
        'seasonal_structural_break',
        'bai_perron',
    ],

    # Volatility + Complexity: chaotic volatility
    ('volatility', 'complexity'): [
        'stochastic_volatility',
        'rough_volatility',
    ],

    # Memory + Complexity: long-memory chaos
    ('memory', 'complexity'): [
        'fractional_brownian',
        'arfima',
    ],

    # Discontinuity + Impulsivity: shock-induced regime changes
    ('discontinuity', 'impulsivity'): [
        'shock_regime_detector',
        'compound_event_analysis',
    ],
}


# =============================================================================
# CLASSIFICATION
# =============================================================================

# Primary signal type classifications based on dominant axes
SIGNAL_TYPES = {
    'PERSISTENT': ('memory',),
    'PERIODIC': ('periodicity',),
    'VOLATILE': ('volatility',),
    'REGIME_SHIFTING': ('discontinuity',),
    'IMPULSIVE': ('impulsivity',),
    'CHAOTIC': ('complexity',),
    'TRENDING_VOLATILE': ('memory', 'volatility'),
    'SEASONAL_TRENDING': ('memory', 'periodicity'),
    'REGIME_VOLATILE': ('discontinuity', 'volatility'),
    'SHOCK_PRONE': ('discontinuity', 'impulsivity'),
}


# =============================================================================
# FUNCTIONS
# =============================================================================

def select_engines(axis_scores: Dict[str, float]) -> List[str]:
    """
    Select recommended engines based on axis scores.

    Args:
        axis_scores: Dict of {axis_name: score [0,1]}

    Returns:
        Prioritized list of recommended engines
    """
    # Only consider the 6 standard axes (ignore event lists)
    STANDARD_AXES = {'memory', 'periodicity', 'volatility', 'discontinuity', 'impulsivity', 'complexity'}

    engines = []
    active_axes = []

    # Single-axis recommendations
    for axis, score in axis_scores.items():
        if axis not in STANDARD_AXES:
            continue
        if not isinstance(score, (int, float)):
            continue
        threshold = THRESHOLDS.get(axis, 0.5)
        if score >= threshold:
            engines.extend(ENGINE_MAP.get(axis, []))
            active_axes.append(axis)

    # Compound recommendations (when multiple axes fire)
    active_set = set(active_axes)
    for axis_combo, combo_engines in COMPOUND_ENGINES.items():
        if all(a in active_set for a in axis_combo):
            engines.extend(combo_engines)

    # Deduplicate while preserving order
    seen = set()
    prioritized = []
    for e in engines:
        if e not in seen:
            seen.add(e)
            prioritized.append(e)

    return prioritized


def get_primary_classification(axis_scores: Dict[str, float]) -> str:
    """
    Classify signal into primary type based on dominant axes.

    Args:
        axis_scores: Dict of {axis_name: score [0,1]}

    Returns:
        Primary signal type classification
    """
    # Only consider the 6 standard axes (ignore event lists)
    STANDARD_AXES = {'memory', 'periodicity', 'volatility', 'discontinuity', 'impulsivity', 'complexity'}

    # Find axes above high threshold
    high_axes = []
    for axis, score in axis_scores.items():
        if axis not in STANDARD_AXES:
            continue
        if not isinstance(score, (int, float)):
            continue
        threshold = HIGH_THRESHOLDS.get(axis, 0.7)
        if score >= threshold:
            high_axes.append((axis, score))

    if not high_axes:
        # Find axes above normal threshold
        for axis, score in axis_scores.items():
            if axis not in STANDARD_AXES:
                continue
            if not isinstance(score, (int, float)):
                continue
            threshold = THRESHOLDS.get(axis, 0.5)
            if score >= threshold:
                high_axes.append((axis, score))

    if not high_axes:
        return 'UNDETERMINED'

    # Sort by score (descending)
    high_axes.sort(key=lambda x: x[1], reverse=True)
    high_axis_names = [a[0] for a in high_axes]

    # Match to signal types
    for signal_type, required_axes in SIGNAL_TYPES.items():
        if all(ax in high_axis_names for ax in required_axes):
            return signal_type

    # Default to dominant axis
    dominant = high_axes[0][0].upper()
    return dominant


def get_axis_weights(axis_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Compute normalized weights for each axis.

    Args:
        axis_scores: Dict of {axis_name: score [0,1]}

    Returns:
        Normalized weights summing to 1.0
    """
    total = sum(axis_scores.values())
    if total == 0:
        return {ax: 0.0 for ax in axis_scores}

    return {ax: score / total for ax, score in axis_scores.items()}


def get_engine_priority(
    axis_scores: Dict[str, float],
    engine_name: str,
) -> float:
    """
    Get priority score for a specific engine.

    Higher scores = more recommended for this signal.

    Args:
        axis_scores: Dict of {axis_name: score [0,1]}
        engine_name: Name of engine to score

    Returns:
        Priority score [0, 1]
    """
    priority = 0.0

    for axis, axis_engines in ENGINE_MAP.items():
        if engine_name in axis_engines:
            axis_score = axis_scores.get(axis, 0.0)
            priority = max(priority, axis_score)

    # Boost for compound recommendations
    for axis_combo, combo_engines in COMPOUND_ENGINES.items():
        if engine_name in combo_engines:
            combo_score = min(axis_scores.get(ax, 0.0) for ax in axis_combo)
            priority = max(priority, combo_score * 1.2)  # 20% boost

    return min(priority, 1.0)


def should_run_engine(
    axis_scores: Dict[str, float],
    engine_name: str,
    min_priority: float = 0.3,
) -> bool:
    """
    Check if an engine should be run for this signal.

    Args:
        axis_scores: Dict of {axis_name: score [0,1]}
        engine_name: Name of engine
        min_priority: Minimum priority threshold

    Returns:
        True if engine should be run
    """
    return get_engine_priority(axis_scores, engine_name) >= min_priority
