"""
Formal Definitions for Dynamical Systems Assessment

This module contains the formal terminology and classification framework
for stability analysis, tipping points, and failure modes.

Based on:
- Strogatz (2015) Nonlinear Dynamics and Chaos
- Scheffer (2009) Critical Transitions in Nature and Society
- Ashwin et al. (2012) Tipping points in open systems

ENGINES computes metrics. ORTHON interprets failure modes.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS: CLASSIFICATION CATEGORIES
# ═══════════════════════════════════════════════════════════════════════════════

class AttractorType(str, Enum):
    """Type of attractor in phase space."""
    FIXED_POINT = "fixed_point"           # Single stable equilibrium
    LIMIT_CYCLE = "limit_cycle"           # Periodic orbit
    TORUS = "torus"                       # Quasi-periodic (multiple frequencies)
    STRANGE_ATTRACTOR = "strange_attractor"  # Chaotic (deterministic chaos)
    NO_ATTRACTOR = "no_attractor"         # Transient / unbounded


class StabilityType(str, Enum):
    """Stability classification based on Lyapunov exponent."""
    ASYMPTOTICALLY_STABLE = "asymptotically_stable"  # λ < -0.1
    STABLE = "stable"                                 # -0.1 < λ < -0.01
    MARGINALLY_STABLE = "marginally_stable"          # -0.01 < λ < 0.01
    WEAKLY_UNSTABLE = "weakly_unstable"              # 0.01 < λ < 0.1
    UNSTABLE = "unstable"                            # λ > 0.1 or diverging
    CHAOTIC = "chaotic"                              # λ > 0.1 with strange attractor


class FailureMode(str, Enum):
    """
    How the system is failing (or approaching failure).

    Structure = Geometry × Mass
    - Geometry: eigenstructure (relationships between signals)
    - Mass: total variance (energy in the system)
    """
    # Geometry-driven (relationships changing)
    GEOMETRY_COLLAPSE = "geometry_collapse"       # eff_dim decreasing, mass stable
    GEOMETRY_EXPLOSION = "geometry_explosion"     # eff_dim increasing chaotically
    ALIGNMENT_LOSS = "alignment_loss"             # PC1 alignment dropping

    # Mass-driven (energy changing)
    MASS_ACCUMULATION = "mass_accumulation"       # Variance growing
    MASS_DEPLETION = "mass_depletion"             # Variance shrinking
    MASS_OSCILLATION = "mass_oscillation"         # Variance cycling

    # Combined
    STRUCTURE_COLLAPSE = "structure_collapse"     # Both geometry and mass failing
    DECOUPLING = "decoupling"                     # Signals becoming independent

    # Healthy
    HEALTHY = "healthy"                           # No failure mode detected


class TippingType(str, Enum):
    """
    Classification of tipping point mechanisms.

    B-tipping: Bifurcation-induced
        - Gradual parameter change crosses threshold
        - Early warning: YES (critical slowing down)
        - Example: fold bifurcation in pump

    R-tipping: Rate-induced
        - Parameter changes too fast for system to track
        - Early warning: NO (or very limited)
        - Example: rate-induced collapse in climate

    N-tipping: Noise-induced
        - Large fluctuation kicks system over barrier
        - Early warning: SOME (increased variance)
        - Example: market crash from news shock
    """
    BIFURCATION = "bifurcation"           # B-tipping (gradual, detectable)
    RATE_INDUCED = "rate_induced"         # R-tipping (fast, NO early warning)
    NOISE_INDUCED = "noise_induced"       # N-tipping (stochastic)
    BIFURCATION_RATE = "bifurcation_rate" # Combined B + R
    UNKNOWN = "unknown"                   # Cannot determine


class SystemTypology(str, Enum):
    """
    System behavior typology (from ORTHON).

    Guides interpretation of metrics.
    """
    DEGRADATION = "degradation"       # System wears down (bearing, battery)
    ACCUMULATION = "accumulation"     # Something builds up (fouling, drift)
    CONSERVATION = "conservation"     # Should be stable (control system)
    OSCILLATORY = "oscillatory"       # Natural cycles (HVAC, seasonal)
    NETWORK = "network"               # Coupled systems (power grid)


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES: METRIC CONTAINERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GeometryMetrics:
    """
    Metrics describing the GEOMETRY (shape/structure) of the system.

    Geometry = relationships between signals.
    """
    eff_dim: float                           # Effective dimension (participation ratio)
    alignment: float                         # PC1 alignment (how dominant is first mode)
    mean_abs_correlation: float              # Mean absolute pairwise correlation
    max_abs_correlation: float = 0.0         # Max pairwise correlation
    coupling_fraction: float = 0.0           # Fraction of coupled pairs
    condition_number: float = 1.0            # Eigenvalue ratio (degeneracy)

    # Trends
    eff_dim_slope: Optional[float] = None    # Trend in eff_dim
    alignment_slope: Optional[float] = None  # Trend in alignment

    def is_collapsing(self) -> bool:
        """Check if geometry is collapsing (dimensional reduction)."""
        if self.eff_dim_slope is not None:
            return self.eff_dim_slope < -0.01  # Significant negative trend
        return False

    def is_decoupling(self) -> bool:
        """Check if signals are becoming independent."""
        return self.mean_abs_correlation < 0.2 and self.coupling_fraction < 0.3


@dataclass
class MassMetrics:
    """
    Metrics describing the MASS (energy/variance) of the system.

    Mass = total variance in the system.
    """
    total_variance: float                    # Sum of eigenvalues
    dominant_signal_mean: float = 0.0        # Mean of dominant signal
    dominant_signal_std: float = 0.0         # Std of dominant signal

    # Trends
    variance_slope: Optional[float] = None   # Trend in total variance
    drift_rate: float = 0.0                  # Rate of drift in mean

    def is_accumulating(self) -> bool:
        """Check if mass is accumulating."""
        if self.variance_slope is not None:
            return self.variance_slope > 0.01
        return False

    def is_depleting(self) -> bool:
        """Check if mass is depleting."""
        if self.variance_slope is not None:
            return self.variance_slope < -0.01
        return False


@dataclass
class EarlyWarningSignals:
    """
    Critical Slowing Down indicators.

    Near a bifurcation:
    - Autocorrelation increases (memory grows)
    - Variance increases (fluctuations grow)
    - Recovery rate decreases (system sluggish)

    IMPORTANT: Only applies to B-tipping!
    R-tipping may show NO early warning.
    """
    autocorrelation_lag1: float              # AR(1) coefficient
    variance: float                          # Current variance
    skewness: float = 0.0                    # Asymmetry
    recovery_rate: Optional[float] = None    # -ln(ρ)

    # Trends
    autocorr_trend: Optional[float] = None   # Slope of rolling autocorr
    variance_trend: Optional[float] = None   # Slope of rolling variance

    # Detection
    csd_score: float = 0.0                   # Composite score (0-1)
    critical_slowing_detected: bool = False  # Boolean flag

    def has_warning_signs(self) -> bool:
        """Check if early warning signs are present."""
        return (
            self.autocorrelation_lag1 > 0.7 or
            (self.autocorr_trend is not None and self.autocorr_trend > 0.01) or
            self.csd_score > 0.5
        )


@dataclass
class FormalAssessment:
    """
    Complete dynamical systems assessment.

    Combines:
    - Attractor classification
    - Stability analysis
    - Failure mode detection
    - Tipping point classification
    - Early warning signals
    """
    # Classification
    attractor_type: AttractorType
    stability_type: StabilityType
    failure_mode: FailureMode
    tipping_type: TippingType
    system_typology: Optional[SystemTypology] = None

    # Metrics
    geometry: Optional[GeometryMetrics] = None
    mass: Optional[MassMetrics] = None
    ews: Optional[EarlyWarningSignals] = None

    # Lyapunov
    lyapunov_exponent: Optional[float] = None
    lyapunov_confidence: float = 0.0

    # Prognostics
    bifurcation_proximity: float = 0.0       # 0 = far, 1 = at bifurcation
    transition_probability: float = 0.0      # P(transition in next window)
    expected_severity: float = 0.0           # Expected impact (0-1)
    lead_time_estimate: Optional[float] = None  # Time until transition

    def to_report(self) -> str:
        """Generate human-readable report."""
        lines = [
            "═" * 70,
            "DYNAMICAL SYSTEMS ASSESSMENT",
            "═" * 70,
            "",
            "SYSTEM CLASSIFICATION",
            f"├── Attractor type: {self.attractor_type.value}",
            f"├── Stability: {self.stability_type.value}",
            f"├── System typology: {self.system_typology.value if self.system_typology else 'Unknown'}",
            f"└── Failure mode: {self.failure_mode.value}",
        ]

        if self.geometry:
            lines.extend([
                "",
                "GEOMETRY (Structure)",
                f"├── Effective dimension: {self.geometry.eff_dim:.2f}",
                f"├── Alignment (PC1 dominance): {self.geometry.alignment:.3f}",
                f"├── Mean correlation: {self.geometry.mean_abs_correlation:.3f}",
                f"└── Dimensional collapse: {'DETECTED' if self.geometry.is_collapsing() else 'Not detected'}",
            ])

        if self.mass:
            lines.extend([
                "",
                "MASS (Slow Variable)",
                f"├── Total variance: {self.mass.total_variance:.2e}",
                f"├── Drift rate: {self.mass.drift_rate:+.4f}",
                f"└── Mass dynamics: {'Accumulating' if self.mass.is_accumulating() else 'Depleting' if self.mass.is_depleting() else 'Stable'}",
            ])

        if self.ews:
            lines.extend([
                "",
                "EARLY WARNING SIGNALS",
                f"├── Autocorrelation (lag-1): {self.ews.autocorrelation_lag1:.3f}",
                f"├── Variance trend: {self.ews.variance_trend:.3f}" if self.ews.variance_trend else "├── Variance trend: N/A",
                f"└── Critical slowing down: {'DETECTED' if self.ews.critical_slowing_detected else 'Not detected'}",
            ])

        lines.extend([
            "",
            "TIPPING CLASSIFICATION",
            f"├── Type: {self.tipping_type.value}",
            f"├── Mechanism: {_tipping_mechanism(self.tipping_type)}",
        ])

        lines.extend([
            "",
            "ASSESSMENT",
            f"├── Bifurcation proximity: {self.bifurcation_proximity * 100:.1f}%",
            f"├── Transition probability: {self.transition_probability * 100:.1f}%",
            f"├── Expected severity: {self.expected_severity * 100:.1f}%",
            f"└── Lead time: {self.lead_time_estimate if self.lead_time_estimate else 'Unknown'}",
        ])

        lines.append("")
        lines.append("═" * 70)

        return "\n".join(lines)


def _tipping_mechanism(tipping_type: TippingType) -> str:
    """Get description of tipping mechanism."""
    mechanisms = {
        TippingType.BIFURCATION: "Gradual parameter drift crossing threshold (early warning available)",
        TippingType.RATE_INDUCED: "Rate of change too fast for system to track (NO early warning)",
        TippingType.NOISE_INDUCED: "Large fluctuation pushing system over barrier",
        TippingType.BIFURCATION_RATE: "Combined bifurcation and rate effects",
        TippingType.UNKNOWN: "Cannot determine mechanism",
    }
    return mechanisms.get(tipping_type, "Unknown")


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def classify_failure_mode(
    geometry: Optional[GeometryMetrics],
    mass: Optional[MassMetrics],
    ews: Optional[EarlyWarningSignals],
    system_type: Optional[SystemTypology] = None,
) -> FailureMode:
    """
    Classify the failure mode based on metrics.

    Structure = Geometry × Mass
    Both can fail independently or together.
    """
    if geometry is None and mass is None:
        return FailureMode.HEALTHY

    geometry_failing = False
    mass_failing = False

    # Check geometry
    if geometry:
        if geometry.is_collapsing():
            geometry_failing = True
        elif geometry.is_decoupling():
            return FailureMode.DECOUPLING
        elif geometry.eff_dim < 1.5:
            geometry_failing = True  # Already collapsed

    # Check mass
    if mass:
        if mass.is_accumulating():
            mass_failing = True
        elif mass.is_depleting():
            return FailureMode.MASS_DEPLETION

    # Combine
    if geometry_failing and mass_failing:
        return FailureMode.STRUCTURE_COLLAPSE
    elif geometry_failing:
        return FailureMode.GEOMETRY_COLLAPSE
    elif mass_failing:
        return FailureMode.MASS_ACCUMULATION

    # Check for alignment loss specifically
    if geometry and geometry.alignment < 0.5 and geometry.alignment_slope and geometry.alignment_slope < -0.01:
        return FailureMode.ALIGNMENT_LOSS

    return FailureMode.HEALTHY


def classify_tipping_type(
    failure_mode: FailureMode,
    geometry: Optional[GeometryMetrics],
    ews: Optional[EarlyWarningSignals],
) -> TippingType:
    """
    Classify the tipping mechanism.

    B-tipping: Early warning via CSD
    R-tipping: No early warning, fast geometry change
    N-tipping: Stochastic, variance-driven
    """
    if failure_mode == FailureMode.HEALTHY:
        return TippingType.UNKNOWN

    has_csd = ews and ews.critical_slowing_detected
    fast_geometry_change = geometry and geometry.eff_dim_slope and abs(geometry.eff_dim_slope) > 0.1

    if has_csd and not fast_geometry_change:
        return TippingType.BIFURCATION
    elif fast_geometry_change and not has_csd:
        return TippingType.RATE_INDUCED
    elif has_csd and fast_geometry_change:
        return TippingType.BIFURCATION_RATE
    elif ews and ews.variance_trend and ews.variance_trend > 0.1:
        return TippingType.NOISE_INDUCED

    return TippingType.UNKNOWN


def classify_stability(lyapunov: float) -> StabilityType:
    """Classify stability based on Lyapunov exponent."""
    if lyapunov < -0.1:
        return StabilityType.ASYMPTOTICALLY_STABLE
    elif lyapunov < -0.01:
        return StabilityType.STABLE
    elif lyapunov < 0.01:
        return StabilityType.MARGINALLY_STABLE
    elif lyapunov < 0.1:
        return StabilityType.WEAKLY_UNSTABLE
    else:
        return StabilityType.CHAOTIC
