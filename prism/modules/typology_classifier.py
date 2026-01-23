"""
prism/modules/typology_classifier.py - Signal Typology Classification

Converts engine outputs into meaningful classifications and human-readable summaries.
This is the INTERPRETATION layer - it answers "what type of signal is this?"

Typology Categories:
    - Trending-Structured: Persistent momentum with recognizable patterns
    - Trending-Chaotic: Persistent momentum with unpredictable dynamics
    - Mean-Reverting: Movements tend to reverse
    - Random-Walk: No directional memory
    - Transitional: Undergoing behavioral shift

Usage:
    from prism.modules.typology_classifier import classify, generate_summary, detect_shifts

    typology, confidence = classify(engine_metrics)
    summaries = generate_summary(engine_metrics, typology)
    shifts = detect_shifts(current_metrics, previous_metrics)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np


# =============================================================================
# TYPOLOGY CLASSIFICATION RULES
# =============================================================================

TYPOLOGY_RULES = {
    'Trending-Structured': {
        'description': 'Persistent momentum with recognizable patterns',
        'rules': {
            'hurst_exponent': ('>', 0.6),
            'determinism': ('>', 0.7),
            'sample_entropy': ('<', 1.5),
        },
        'required': ['hurst_exponent'],  # Must have this metric
        'min_match': 2,  # Need at least 2 rules to match
    },
    'Trending-Chaotic': {
        'description': 'Persistent momentum with unpredictable dynamics',
        'rules': {
            'hurst_exponent': ('>', 0.6),
            'determinism': ('<', 0.5),
            'sample_entropy': ('>', 2.0),
        },
        'required': ['hurst_exponent'],
        'min_match': 2,
    },
    'Mean-Reverting': {
        'description': 'Movements tend to reverse toward an equilibrium',
        'rules': {
            'hurst_exponent': ('<', 0.45),
            'determinism': ('>', 0.5),
        },
        'required': ['hurst_exponent'],
        'min_match': 1,
    },
    'Random-Walk': {
        'description': 'No directional memory - unpredictable',
        'rules': {
            'hurst_exponent': ('between', 0.45, 0.55),
            'determinism': ('<', 0.5),
        },
        'required': [],
        'min_match': 1,
    },
    'Transitional': {
        'description': 'Undergoing behavioral regime shift',
        'rules': {
            'delta_hurst': ('abs>', 0.15),  # Absolute change threshold
        },
        'required': ['delta_hurst'],
        'min_match': 1,
    },
}

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    'high': 0.8,
    'medium': 0.6,
    'low': 0.4,
}

# Shift detection thresholds
SHIFT_THRESHOLDS = {
    'hurst_exponent': 0.15,
    'determinism': 0.15,
    'sample_entropy': 0.3,
    'laminarity': 0.15,
    'garch_persistence': 0.1,
    'spectral_centroid': 0.1,
    'lyapunov_exponent': 0.1,
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TypologyResult:
    """Result of typology classification."""
    typology: str
    confidence: float
    confidence_level: str  # 'high', 'medium', 'low'
    matched_rules: List[str]
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'typology': self.typology,
            'typology_confidence': self.confidence,
            'typology_confidence_level': self.confidence_level,
            'typology_matched_rules': self.matched_rules,
            'typology_description': self.description,
        }


@dataclass
class ShiftResult:
    """Result of shift detection."""
    shift_detected: bool
    shifts: List[Dict[str, Any]]
    shift_description: str
    delta_hurst: Optional[float] = None
    delta_determinism: Optional[float] = None
    delta_entropy: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'shift_detected': self.shift_detected,
            'shift_description': self.shift_description,
            'delta_hurst': self.delta_hurst,
            'delta_determinism': self.delta_determinism,
            'delta_entropy': self.delta_entropy,
        }


# =============================================================================
# CLASSIFICATION FUNCTIONS
# =============================================================================

def _check_rule(value: float, rule: tuple) -> bool:
    """Check if a value satisfies a rule."""
    if value is None or np.isnan(value):
        return False

    op = rule[0]

    if op == '>':
        return value > rule[1]
    elif op == '<':
        return value < rule[1]
    elif op == '>=':
        return value >= rule[1]
    elif op == '<=':
        return value <= rule[1]
    elif op == 'between':
        return rule[1] <= value <= rule[2]
    elif op == 'abs>':
        return abs(value) > rule[1]
    elif op == 'abs<':
        return abs(value) < rule[1]

    return False


def classify(
    metrics: Dict[str, float],
    prev_metrics: Optional[Dict[str, float]] = None,
) -> TypologyResult:
    """
    Assign typology label based on engine outputs.

    Args:
        metrics: Current window engine metrics
        prev_metrics: Previous window metrics (for shift detection)

    Returns:
        TypologyResult with typology, confidence, and matched rules
    """
    # Compute deltas if we have previous metrics
    if prev_metrics:
        metrics = {**metrics}  # Copy to avoid mutation
        for key in ['hurst_exponent', 'determinism', 'sample_entropy', 'laminarity']:
            if key in metrics and key in prev_metrics:
                current = metrics.get(key)
                previous = prev_metrics.get(key)
                if current is not None and previous is not None:
                    if not np.isnan(current) and not np.isnan(previous):
                        metrics[f'delta_{key}'] = current - previous

    best_match = None
    best_score = 0.0
    best_matched = []

    for typology_name, config in TYPOLOGY_RULES.items():
        rules = config['rules']
        required = config.get('required', [])
        min_match = config.get('min_match', 1)

        # Check required metrics
        has_required = all(
            m in metrics and metrics[m] is not None and not np.isnan(metrics.get(m, np.nan))
            for m in required
        )
        if not has_required:
            continue

        # Count matching rules
        matched_rules = []
        for metric_name, rule in rules.items():
            value = metrics.get(metric_name)
            if value is not None and _check_rule(value, rule):
                matched_rules.append(f"{metric_name}: {value:.3f}")

        # Calculate score
        if len(matched_rules) >= min_match:
            score = len(matched_rules) / len(rules)

            if score > best_score:
                best_score = score
                best_match = typology_name
                best_matched = matched_rules

    # Default to Random-Walk if no match
    if best_match is None:
        best_match = 'Random-Walk'
        best_score = 0.5
        best_matched = ['default classification']

    # Determine confidence level
    if best_score >= CONFIDENCE_THRESHOLDS['high']:
        confidence_level = 'high'
    elif best_score >= CONFIDENCE_THRESHOLDS['medium']:
        confidence_level = 'medium'
    else:
        confidence_level = 'low'

    description = TYPOLOGY_RULES.get(best_match, {}).get(
        'description',
        'Unknown behavioral pattern'
    )

    return TypologyResult(
        typology=best_match,
        confidence=best_score,
        confidence_level=confidence_level,
        matched_rules=best_matched,
        description=description,
    )


# =============================================================================
# BULLET SUMMARY GENERATION
# =============================================================================

def generate_summary(
    metrics: Dict[str, float],
    typology_result: Optional[TypologyResult] = None,
    prev_metrics: Optional[Dict[str, float]] = None,
) -> List[str]:
    """
    Generate human-readable bullet summaries.

    Universal templates that apply to any signal, any domain.

    Args:
        metrics: Engine output metrics
        typology_result: Classification result (optional, will compute if not provided)
        prev_metrics: Previous window metrics for shift detection

    Returns:
        List of summary strings
    """
    summaries = []

    # Get typology if not provided
    if typology_result is None:
        typology_result = classify(metrics, prev_metrics)

    # === TREND CHARACTER ===
    hurst = metrics.get('hurst_exponent')
    if hurst is not None and not np.isnan(hurst):
        if hurst > 0.6:
            summaries.append("Trending: movements tend to persist")
        elif hurst < 0.45:
            summaries.append("Mean-reverting: movements tend to reverse")
        else:
            summaries.append("Random walk: no directional memory")

    # === STRUCTURAL CHARACTER ===
    determinism = metrics.get('determinism')
    if determinism is not None and not np.isnan(determinism):
        if determinism > 0.7:
            summaries.append("Structured: behavior follows recognizable patterns")
        elif determinism < 0.5:
            summaries.append("Chaotic: behavior is difficult to predict")

    entropy = metrics.get('sample_entropy')
    if entropy is not None and not np.isnan(entropy):
        if entropy > 2.0:
            summaries.append("Complex: high information content")
        elif entropy < 1.0:
            summaries.append("Simple: low information content")

    # === STABILITY CHARACTER ===
    lyapunov = metrics.get('lyapunov_exponent')
    if lyapunov is not None and not np.isnan(lyapunov):
        if lyapunov < 0:
            summaries.append("Stable: perturbations decay")
        elif lyapunov > 0.1:
            summaries.append("Sensitive: perturbations amplify")

    garch_persistence = metrics.get('garch_persistence')
    if garch_persistence is None:
        # Compute from alpha + beta if available
        alpha = metrics.get('garch_alpha')
        beta = metrics.get('garch_beta')
        if alpha is not None and beta is not None:
            if not np.isnan(alpha) and not np.isnan(beta):
                garch_persistence = alpha + beta

    if garch_persistence is not None and not np.isnan(garch_persistence):
        if garch_persistence > 0.9:
            summaries.append("Volatility clusters: calm and storm periods")

    # === TEMPORAL CHARACTER ===
    spectral_centroid = metrics.get('spectral_centroid')
    if spectral_centroid is not None and not np.isnan(spectral_centroid):
        if spectral_centroid < 0.2:
            summaries.append("Dominated by slow cycles (low frequency)")
        elif spectral_centroid > 0.5:
            summaries.append("Dominated by fast fluctuations (high frequency)")

    # === BREAK CHARACTER ===
    dirac_detected = metrics.get('dirac_detected')
    dirac_n_impulses = metrics.get('dirac_n_impulses', 0)
    if dirac_detected or dirac_n_impulses > 0:
        summaries.append(f"Impulse activity: {int(dirac_n_impulses)} spikes detected")

    heaviside_detected = metrics.get('heaviside_detected')
    heaviside_n_steps = metrics.get('heaviside_n_steps', 0)
    if heaviside_detected or heaviside_n_steps > 0:
        summaries.append(f"Step changes: {int(heaviside_n_steps)} level shifts detected")

    break_detected = metrics.get('break_detected')
    break_is_accelerating = metrics.get('break_is_accelerating')
    if break_detected and break_is_accelerating:
        summaries.append("Warning: Break acceleration - discontinuities increasing")

    # === SHIFT ALERTS ===
    if prev_metrics:
        shifts = detect_shifts(metrics, prev_metrics)
        for shift in shifts.shifts:
            summaries.append(f"Warning: {shift['description']}")

    return summaries


def generate_summary_text(summaries: List[str]) -> str:
    """Convert summary list to narrative text."""
    if not summaries:
        return "No significant behavioral characteristics detected."

    # Separate warnings from observations
    warnings = [s for s in summaries if s.startswith('Warning:')]
    observations = [s for s in summaries if not s.startswith('Warning:')]

    parts = []
    if observations:
        parts.append('. '.join(observations) + '.')
    if warnings:
        parts.append(' '.join(warnings))

    return ' '.join(parts)


# =============================================================================
# SHIFT DETECTION
# =============================================================================

def detect_shifts(
    current: Dict[str, float],
    previous: Dict[str, float],
    thresholds: Optional[Dict[str, float]] = None,
) -> ShiftResult:
    """
    Detect significant changes between windows.

    Args:
        current: Current window metrics
        previous: Previous window metrics
        thresholds: Override default shift thresholds

    Returns:
        ShiftResult with detected shifts
    """
    thresholds = thresholds or SHIFT_THRESHOLDS

    shifts = []
    delta_hurst = None
    delta_determinism = None
    delta_entropy = None

    for metric, threshold in thresholds.items():
        current_val = current.get(metric)
        prev_val = previous.get(metric)

        if current_val is None or prev_val is None:
            continue
        if np.isnan(current_val) or np.isnan(prev_val):
            continue

        delta = current_val - prev_val

        # Store key deltas
        if metric == 'hurst_exponent':
            delta_hurst = delta
        elif metric == 'determinism':
            delta_determinism = delta
        elif metric == 'sample_entropy':
            delta_entropy = delta

        if abs(delta) > threshold:
            direction = 'rose' if delta > 0 else 'dropped'

            # Human-readable metric names
            readable_names = {
                'hurst_exponent': 'Hurst',
                'determinism': 'predictability',
                'sample_entropy': 'complexity',
                'laminarity': 'structure',
                'garch_persistence': 'volatility persistence',
                'spectral_centroid': 'frequency profile',
                'lyapunov_exponent': 'sensitivity',
            }

            readable = readable_names.get(metric, metric)

            shifts.append({
                'metric': metric,
                'delta': delta,
                'threshold': threshold,
                'description': f"Behavioral shift: {readable} {direction} {abs(delta):.2f}",
            })

    shift_detected = len(shifts) > 0

    # Combine descriptions
    if shifts:
        descriptions = [s['description'] for s in shifts]
        shift_description = '; '.join(descriptions)
    else:
        shift_description = ''

    return ShiftResult(
        shift_detected=shift_detected,
        shifts=shifts,
        shift_description=shift_description,
        delta_hurst=delta_hurst,
        delta_determinism=delta_determinism,
        delta_entropy=delta_entropy,
    )


# =============================================================================
# PERSISTENCE LABEL
# =============================================================================

def get_persistence_label(hurst: Optional[float]) -> str:
    """Get human-readable persistence label from Hurst exponent."""
    if hurst is None or np.isnan(hurst):
        return 'unknown'

    if hurst > 0.6:
        return 'trending'
    elif hurst < 0.45:
        return 'mean-reverting'
    else:
        return 'random'


# =============================================================================
# STRUCTURAL LABEL
# =============================================================================

def get_structural_label(determinism: Optional[float], entropy: Optional[float]) -> str:
    """Get human-readable structural label from RQA determinism and entropy."""
    det_high = determinism is not None and not np.isnan(determinism) and determinism > 0.7
    det_low = determinism is not None and not np.isnan(determinism) and determinism < 0.5
    ent_high = entropy is not None and not np.isnan(entropy) and entropy > 2.0
    ent_low = entropy is not None and not np.isnan(entropy) and entropy < 1.0

    if det_high and ent_low:
        return 'structured-simple'
    elif det_high and ent_high:
        return 'structured-complex'
    elif det_low and ent_high:
        return 'chaotic-complex'
    elif det_low and ent_low:
        return 'chaotic-simple'
    else:
        return 'mixed'


# =============================================================================
# FULL TYPOLOGY COMPUTATION
# =============================================================================

def compute_typology(
    metrics: Dict[str, float],
    prev_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute full typology classification with summaries.

    This is the main entry point for the typology classifier.

    Args:
        metrics: Engine output metrics for current window
        prev_metrics: Previous window metrics (optional)

    Returns:
        Dictionary with all typology fields for the output schema
    """
    # Classify
    typology_result = classify(metrics, prev_metrics)

    # Generate summaries
    summaries = generate_summary(metrics, typology_result, prev_metrics)
    summary_text = generate_summary_text(summaries)

    # Detect shifts
    shift_result = ShiftResult(
        shift_detected=False,
        shifts=[],
        shift_description='',
    )
    if prev_metrics:
        shift_result = detect_shifts(metrics, prev_metrics)

    # Get labels
    persistence_label = get_persistence_label(metrics.get('hurst_exponent'))
    structural_label = get_structural_label(
        metrics.get('determinism'),
        metrics.get('sample_entropy')
    )

    return {
        # Classification
        **typology_result.to_dict(),

        # Labels
        'hurst_persistence': persistence_label,
        'structural_character': structural_label,

        # Shifts
        **shift_result.to_dict(),

        # Summaries
        'summary_bullets': summaries,
        'summary_text': summary_text,
    }
