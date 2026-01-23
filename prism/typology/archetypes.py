"""
Ørthon Signal Typology: Archetype Library
==========================================

The Archetype Library defines known behavioral fingerprint patterns.
Each archetype represents a distinct region in 6D typology space.

Fingerprints are matched to archetypes to provide:
    - Human-interpretable classification
    - Distance to regime boundaries
    - Transition probability estimation

The insight: When axes measure different things, "disagreement" is discovery.
The fingerprint combination reveals what the signal actually IS.
"""

from typing import List, Tuple, Dict
import numpy as np

from .models import (
    Archetype, SignalTypology, 
    MemoryClass, InformationClass, RecurrenceClass,
    VolatilityClass, FrequencyClass, DynamicsClass,
    TransitionType
)


# =============================================================================
# PRIMARY ARCHETYPES
# =============================================================================

ARCHETYPES: List[Archetype] = [
    
    # -------------------------------------------------------------------------
    # TRENDING FAMILY
    # -------------------------------------------------------------------------
    
    Archetype(
        name="Stable Trend",
        description="Persistent momentum with predictable structure. High confidence continuation.",
        memory_range=(0.60, 1.0),       # High Hurst = persistent
        information_range=(0.0, 0.4),   # Low entropy = structured
        recurrence_range=(0.65, 1.0),   # High determinism = patterns repeat
        volatility_range=(0.0, 0.85),   # Low-moderate volatility
        frequency_range=(0.0, 0.4),     # Narrowband = dominant frequency
        dynamics_range=(0.0, 0.45),     # Negative Lyapunov = stable
        expects_dirac=False,
        expects_heaviside=False
    ),
    
    Archetype(
        name="Momentum Decay",
        description="Trend persists but structure dissolving. Predictability collapsing.",
        memory_range=(0.55, 0.85),      # Still persistent
        information_range=(0.5, 0.8),   # Rising entropy
        recurrence_range=(0.3, 0.6),    # Falling determinism
        volatility_range=(0.6, 0.95),   # Rising volatility
        frequency_range=(0.4, 0.7),     # Broadening spectrum
        dynamics_range=(0.4, 0.6),      # Lyapunov approaching zero
        expects_dirac=True,             # Impulses appearing
        expects_heaviside=False
    ),
    
    Archetype(
        name="Trending Volatile",
        description="Strong momentum with amplitude shocks. Direction clear, magnitude uncertain.",
        memory_range=(0.60, 0.90),      # High persistence
        information_range=(0.3, 0.6),   # Moderate entropy
        recurrence_range=(0.5, 0.8),    # Moderate-high determinism
        volatility_range=(0.85, 1.0),   # High volatility persistence
        frequency_range=(0.2, 0.5),     # Moderate bandwidth
        dynamics_range=(0.3, 0.55),     # Near stable
        expects_dirac=True,             # Amplitude shocks
        expects_heaviside=False
    ),
    
    # -------------------------------------------------------------------------
    # MEAN-REVERTING FAMILY
    # -------------------------------------------------------------------------
    
    Archetype(
        name="Mean Reversion Stable",
        description="Oscillation around equilibrium with low volatility. Classic range-bound.",
        memory_range=(0.0, 0.42),       # Low Hurst = anti-persistent
        information_range=(0.2, 0.5),   # Low-moderate entropy
        recurrence_range=(0.65, 1.0),   # High determinism
        volatility_range=(0.0, 0.75),   # Low volatility persistence
        frequency_range=(0.0, 0.35),    # Narrowband (regular oscillation)
        dynamics_range=(0.0, 0.4),      # Stable attractor
        expects_dirac=False,
        expects_heaviside=False
    ),
    
    Archetype(
        name="Mean Reversion Volatile", 
        description="Reverting but with amplitude shocks. Returns to mean, path is rough.",
        memory_range=(0.0, 0.45),       # Anti-persistent
        information_range=(0.3, 0.6),   # Moderate entropy
        recurrence_range=(0.5, 0.8),    # Moderate determinism
        volatility_range=(0.80, 1.0),   # High volatility
        frequency_range=(0.2, 0.5),     # Moderate bandwidth
        dynamics_range=(0.2, 0.5),      # Near stable
        expects_dirac=True,             # Volatility spikes
        expects_heaviside=False
    ),
    
    # -------------------------------------------------------------------------
    # RANDOM / NEUTRAL FAMILY
    # -------------------------------------------------------------------------
    
    Archetype(
        name="Random Walk",
        description="No memory, no structure. Classic efficient market hypothesis.",
        memory_range=(0.45, 0.55),      # Hurst ≈ 0.5
        information_range=(0.6, 0.9),   # High entropy
        recurrence_range=(0.1, 0.4),    # Low determinism
        volatility_range=(0.4, 0.7),    # Moderate volatility
        frequency_range=(0.5, 0.8),     # Broadband (flat spectrum)
        dynamics_range=(0.45, 0.55),    # Lyapunov ≈ 0
        expects_dirac=False,
        expects_heaviside=False
    ),
    
    Archetype(
        name="Consolidation",
        description="Low movement, high structure. Compression before expansion.",
        memory_range=(0.42, 0.58),      # Near random
        information_range=(0.1, 0.4),   # Low entropy
        recurrence_range=(0.7, 1.0),    # High determinism
        volatility_range=(0.0, 0.5),    # Low volatility
        frequency_range=(0.0, 0.3),     # Very narrowband
        dynamics_range=(0.0, 0.4),      # Stable
        expects_dirac=False,
        expects_heaviside=False
    ),
    
    # -------------------------------------------------------------------------
    # CHAOTIC / UNSTABLE FAMILY
    # -------------------------------------------------------------------------
    
    Archetype(
        name="Chaotic",
        description="Sensitive dependence on initial conditions. Short-term predictable, long-term impossible.",
        memory_range=(0.3, 0.7),        # Variable (chaos isn't about memory)
        information_range=(0.7, 1.0),   # High entropy
        recurrence_range=(0.1, 0.4),    # Low determinism
        volatility_range=(0.7, 1.0),    # High volatility
        frequency_range=(0.6, 1.0),     # Broadband
        dynamics_range=(0.6, 1.0),      # Positive Lyapunov
        expects_dirac=True,
        expects_heaviside=False
    ),
    
    Archetype(
        name="Edge of Chaos",
        description="Critical state between order and disorder. Maximum information processing.",
        memory_range=(0.4, 0.6),        # Near critical
        information_range=(0.5, 0.7),   # Moderate-high entropy
        recurrence_range=(0.35, 0.55),  # Transitional determinism
        volatility_range=(0.5, 0.8),    # Moderate-high volatility
        frequency_range=(0.4, 0.6),     # 1/f-like spectrum
        dynamics_range=(0.45, 0.55),    # Lyapunov ≈ 0
        expects_dirac=True,
        expects_heaviside=True
    ),
    
    # -------------------------------------------------------------------------
    # TRANSITION / REGIME CHANGE FAMILY
    # -------------------------------------------------------------------------
    
    Archetype(
        name="Regime Transition",
        description="Active regime change in progress. Multiple axes shifting simultaneously.",
        memory_range=(0.3, 0.7),        # Shifting
        information_range=(0.6, 0.9),   # Spiking entropy
        recurrence_range=(0.2, 0.5),    # Dropping determinism
        volatility_range=(0.7, 1.0),    # Spiking volatility
        frequency_range=(0.4, 0.8),     # Shifting spectrum
        dynamics_range=(0.4, 0.7),      # Lyapunov crossing zero
        expects_dirac=True,
        expects_heaviside=True          # Step change detected
    ),
    
    Archetype(
        name="Post-Shock Recovery",
        description="System returning to equilibrium after discontinuity.",
        memory_range=(0.4, 0.7),        # Recovering
        information_range=(0.4, 0.7),   # Falling from spike
        recurrence_range=(0.4, 0.7),    # Rebuilding structure
        volatility_range=(0.5, 0.85),   # Elevated but falling
        frequency_range=(0.3, 0.6),     # Narrowing
        dynamics_range=(0.3, 0.55),     # Stabilizing
        expects_dirac=False,
        expects_heaviside=True          # Recent step detected
    ),
    
    # -------------------------------------------------------------------------
    # STRUCTURED OSCILLATION FAMILY
    # -------------------------------------------------------------------------
    
    Archetype(
        name="Periodic",
        description="Regular, predictable cycles. Seasonal or cyclical patterns.",
        memory_range=(0.2, 0.5),        # Can be anti-persistent or random
        information_range=(0.0, 0.35),  # Very low entropy
        recurrence_range=(0.85, 1.0),   # Very high determinism
        volatility_range=(0.0, 0.5),    # Low volatility
        frequency_range=(0.0, 0.2),     # Very narrowband
        dynamics_range=(0.0, 0.35),     # Very stable
        expects_dirac=False,
        expects_heaviside=False
    ),
    
    Archetype(
        name="Quasi-Periodic",
        description="Multiple incommensurate frequencies. Complex but structured oscillation.",
        memory_range=(0.3, 0.6),        # Variable
        information_range=(0.2, 0.5),   # Low-moderate entropy
        recurrence_range=(0.6, 0.85),   # High determinism
        volatility_range=(0.2, 0.6),    # Low-moderate volatility
        frequency_range=(0.25, 0.5),    # Multiple peaks (moderate bandwidth)
        dynamics_range=(0.1, 0.45),     # Stable
        expects_dirac=False,
        expects_heaviside=False
    ),
]


# =============================================================================
# ARCHETYPE MATCHING
# =============================================================================

def compute_fingerprint(typology: SignalTypology) -> np.ndarray:
    """
    Compute 6D fingerprint from SignalTypology.
    
    Each dimension normalized to [0, 1]:
        0: Memory (Hurst exponent)
        1: Information (Permutation entropy)
        2: Recurrence (Determinism)
        3: Volatility (GARCH persistence)
        4: Frequency (Spectral bandwidth, normalized)
        5: Dynamics (Lyapunov exponent, shifted and scaled)
    """
    fingerprint = np.array([
        typology.memory.hurst_exponent,
        typology.information.entropy_permutation,
        typology.recurrence.determinism,
        typology.volatility.garch_persistence,
        np.clip(typology.frequency.spectral_bandwidth / 0.25, 0, 1),
        np.clip((typology.dynamics.lyapunov_exponent + 0.5) / 1.0, 0, 1)
    ])
    return np.clip(fingerprint, 0, 1)


def match_archetype(
    fingerprint: np.ndarray,
    discontinuity_dirac: bool = False,
    discontinuity_heaviside: bool = False,
    archetypes: List[Archetype] = None
) -> Tuple[Archetype, float, Archetype, float]:
    """
    Match fingerprint to nearest archetype.
    
    Returns:
        (primary_archetype, distance, secondary_archetype, secondary_distance)
    """
    if archetypes is None:
        archetypes = ARCHETYPES
    
    distances = []
    
    for archetype in archetypes:
        # Base distance from centroid
        dist = archetype.distance_to(fingerprint)
        
        # Penalty for discontinuity mismatch
        if discontinuity_dirac and not archetype.expects_dirac:
            dist += 0.15  # Penalize if we see Dirac but archetype doesn't expect it
        if discontinuity_heaviside and not archetype.expects_heaviside:
            dist += 0.20  # Heavier penalty for unexpected step change
        if archetype.expects_heaviside and not discontinuity_heaviside:
            dist += 0.10  # Archetype expects step but we don't see one
            
        distances.append((archetype, dist))
    
    # Sort by distance
    distances.sort(key=lambda x: x[1])
    
    primary = distances[0]
    secondary = distances[1] if len(distances) > 1 else distances[0]
    
    return primary[0], primary[1], secondary[0], secondary[1]


def compute_boundary_proximity(
    fingerprint: np.ndarray,
    primary_archetype: Archetype,
    secondary_archetype: Archetype,
    primary_distance: float,
    secondary_distance: float
) -> float:
    """
    Compute proximity to regime boundary (0 = at boundary, 1 = far from boundary).
    
    If primary and secondary archetypes are very close in distance,
    we're near a regime boundary.
    """
    if secondary_distance < 0.001:
        return 0.0  # Prevent division by zero
    
    # Ratio of distances
    distance_ratio = primary_distance / secondary_distance
    
    # Convert to proximity: when ratio ≈ 1, we're at boundary
    # When ratio << 1, we're far from boundary
    proximity = 1.0 - np.clip(distance_ratio, 0, 1)
    
    return proximity


# =============================================================================
# DIFFERENTIAL DIAGNOSIS (The Discovery Matrix)
# =============================================================================

DISCOVERY_MATRIX: Dict[Tuple[str, str], str] = {
    # (what_moved, what_stable) -> interpretation
    
    # Entropy changes
    ("information", "memory,recurrence"): 
        "Structure dissolving but patterns persist — instability emerging",
    ("information", "memory"): 
        "Complexity increasing without memory change — noise injection",
    
    # Memory changes  
    ("memory", "recurrence,information"):
        "Memory shortening but patterns still repeat — scale shift",
    ("memory", "volatility"):
        "Persistence changing without volatility change — trend character shift",
    
    # Recurrence changes
    ("recurrence", "memory,information"):
        "Determinism dropping but memory intact — approaching chaos",
    ("recurrence", "memory"):
        "Pattern structure changing — regime evolution",
    
    # Volatility changes
    ("volatility", "memory,recurrence,information"):
        "Pure amplitude shock — structure intact, magnitude uncertain",
    ("volatility", "memory"):
        "Volatility spike without trend change — temporary instability",
    
    # Dynamics changes
    ("dynamics", "memory,recurrence"):
        "Lyapunov crossing zero — critical transition",
    ("dynamics", "volatility"):
        "Stability changing without volatility — attractor deformation",
    
    # Frequency changes
    ("frequency", "memory,recurrence"):
        "Spectral shift without structural change — timescale migration",
    ("frequency", "dynamics"):
        "Frequency structure changing — new dominant rhythms",
    
    # Multi-axis changes
    ("memory,information", "volatility"):
        "Trend and structure both shifting — regime change likely",
    ("recurrence,dynamics", "memory"):
        "Determinism and stability both dropping — chaos onset",
    ("volatility,dynamics", "memory"):
        "Volatility and stability both changing — system stress",
    
    # All axes
    ("all", "none"):
        "All axes moving — full regime transition in progress",
}


def diagnose_differential(
    current: SignalTypology,
    previous: SignalTypology,
    threshold: float = 0.1
) -> Tuple[List[str], List[str], str]:
    """
    Identify which axes moved vs stayed stable between windows.
    
    Returns:
        (axes_moving, axes_stable, diagnosis_text)
    """
    axes_moving = []
    axes_stable = []
    
    # Compare each axis
    if abs(current.memory.hurst_exponent - previous.memory.hurst_exponent) > threshold:
        axes_moving.append("memory")
    else:
        axes_stable.append("memory")
        
    if abs(current.information.entropy_permutation - previous.information.entropy_permutation) > threshold:
        axes_moving.append("information")
    else:
        axes_stable.append("information")
        
    if abs(current.recurrence.determinism - previous.recurrence.determinism) > threshold:
        axes_moving.append("recurrence")
    else:
        axes_stable.append("recurrence")
        
    if abs(current.volatility.garch_persistence - previous.volatility.garch_persistence) > threshold:
        axes_moving.append("volatility")
    else:
        axes_stable.append("volatility")
        
    if abs(current.frequency.spectral_bandwidth - previous.frequency.spectral_bandwidth) > threshold * 0.25:
        axes_moving.append("frequency")
    else:
        axes_stable.append("frequency")
        
    if abs(current.dynamics.lyapunov_exponent - previous.dynamics.lyapunov_exponent) > threshold:
        axes_moving.append("dynamics")
    else:
        axes_stable.append("dynamics")
    
    # Generate diagnosis
    if len(axes_moving) == 0:
        diagnosis = "All axes stable — regime unchanged"
    elif len(axes_moving) == 6:
        diagnosis = "All axes moving — full regime transition in progress"
    else:
        # Look up in discovery matrix
        moving_key = ",".join(sorted(axes_moving))
        stable_key = ",".join(sorted(axes_stable))
        
        diagnosis = DISCOVERY_MATRIX.get(
            (moving_key, stable_key),
            f"Axes {', '.join(axes_moving)} shifting while {', '.join(axes_stable)} stable"
        )
    
    return axes_moving, axes_stable, diagnosis


# =============================================================================
# CLASSIFICATION
# =============================================================================

def classify_memory(hurst: float, acf_decay_power_law: bool = False) -> MemoryClass:
    """Classify memory axis"""
    if hurst < 0.45:
        return MemoryClass.ANTI_PERSISTENT
    elif hurst > 0.55:
        return MemoryClass.PERSISTENT
    else:
        return MemoryClass.RANDOM


def classify_information(entropy: float, entropy_rate: float = 0.0) -> InformationClass:
    """Classify information axis"""
    if entropy < 0.4:
        return InformationClass.LOW
    elif entropy > 0.7:
        return InformationClass.HIGH
    else:
        return InformationClass.MODERATE


def classify_recurrence(determinism: float) -> RecurrenceClass:
    """Classify recurrence axis"""
    if determinism > 0.7:
        return RecurrenceClass.DETERMINISTIC
    elif determinism < 0.4:
        return RecurrenceClass.STOCHASTIC
    else:
        return RecurrenceClass.TRANSITIONAL


def classify_volatility(persistence: float) -> VolatilityClass:
    """Classify volatility axis"""
    if persistence < 0.85:
        return VolatilityClass.DISSIPATING
    elif persistence >= 0.99:
        return VolatilityClass.INTEGRATED
    else:
        return VolatilityClass.PERSISTENT


def classify_frequency(centroid: float, bandwidth: float) -> FrequencyClass:
    """Classify frequency axis"""
    if bandwidth < 0.1:
        return FrequencyClass.NARROWBAND
    elif bandwidth > 0.2:
        return FrequencyClass.BROADBAND
    else:
        return FrequencyClass.ONE_OVER_F


def classify_dynamics(lyapunov: float) -> DynamicsClass:
    """Classify dynamics axis"""
    if lyapunov < -0.05:
        return DynamicsClass.STABLE
    elif lyapunov > 0.05:
        return DynamicsClass.CHAOTIC
    else:
        return DynamicsClass.EDGE_OF_CHAOS


# =============================================================================
# SUMMARY GENERATION
# =============================================================================

def generate_summary(
    typology: SignalTypology,
    previous: SignalTypology = None
) -> str:
    """Generate human-readable summary of signal typology."""
    
    parts = []
    alerts = []
    
    # Primary classification
    parts.append(f"**{typology.archetype}**")
    
    # Confidence indicator
    if typology.boundary_proximity < 0.3:
        parts.append(f"(boundary proximity: {typology.boundary_proximity:.0%})")
        alerts.append(f"⚠️ Near regime boundary with {typology.secondary_archetype}")
    
    # Key characteristics from dominant axes
    mem = typology.memory.memory_class.value.replace("_", "-")
    rec = typology.recurrence.recurrence_class.value
    vol = typology.volatility.volatility_class.value
    
    parts.append(f"| {mem} memory | {rec} structure | {vol} volatility")
    
    # Discontinuity alerts
    if typology.discontinuity.heaviside.detected:
        n = typology.discontinuity.heaviside.count
        mag = typology.discontinuity.heaviside.max_magnitude
        alerts.append(f"⚠️ Step discontinuity detected: {n}x, max {mag:.1f}σ")
        
    if typology.discontinuity.dirac.detected:
        n = typology.discontinuity.dirac.count
        mag = typology.discontinuity.dirac.max_magnitude
        alerts.append(f"📍 Impulse detected: {n}x, max {mag:.1f}σ")
    
    # Differential diagnosis if previous available
    if previous is not None:
        axes_moving, axes_stable, diagnosis = diagnose_differential(typology, previous)
        if axes_moving:
            alerts.append(f"📊 {diagnosis}")
            typology.axes_moving = axes_moving
            typology.axes_stable = axes_stable
    
    # Combine
    summary = " ".join(parts)
    if alerts:
        summary += "\n" + "\n".join(alerts)
    
    typology.summary = summary
    typology.alerts = alerts
    
    return summary


def compute_confidence(typology: SignalTypology) -> float:
    """
    Compute overall classification confidence.
    
    Based on:
        - Distance to archetype (closer = higher confidence)
        - Boundary proximity (further = higher confidence)
        - Axes agreement (more stable = higher confidence)
    """
    # Distance factor: closer to archetype = higher confidence
    dist_factor = 1.0 / (1.0 + typology.archetype_distance)
    
    # Boundary factor: further from boundary = higher confidence
    boundary_factor = typology.boundary_proximity
    
    # Stability factor: more stable axes = higher confidence
    n_stable = len(typology.axes_stable)
    stability_factor = n_stable / 6.0
    
    # Combine (weighted average)
    confidence = (
        0.4 * dist_factor +
        0.35 * boundary_factor +
        0.25 * stability_factor
    )
    
    return np.clip(confidence, 0, 1)
