"""
Dynamical Systems Layer
=======================

Orchestrates dynamics engines to answer:
    "When and how does the system change?"

Sub-questions:
    - Regime: What state is it in?
    - Stability: Is it stable or transitioning?
    - Trajectory: Where is it heading?
    - Attractors: What states does it tend toward?

This is a PURE ORCHESTRATOR - no computation here.
All dynamics calculations delegated to engines/.

Output:
    - DynamicsVector: Numerical measurements for downstream layers
    - DynamicsTypology: Classification for interpretation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class RegimeClass(Enum):
    """Dynamical regime classification"""
    COUPLED = "coupled"                 # Strong correlation, moving together
    DECOUPLED = "decoupled"             # Weak correlation, independent
    MODERATE = "moderate"               # Intermediate coupling
    TRANSITIONING = "transitioning"     # Active regime change


class StabilityClass(Enum):
    """System stability classification"""
    STABLE = "stable"                   # Minimal change
    EVOLVING = "evolving"               # Gradual change
    UNSTABLE = "unstable"               # Rapid change
    CRITICAL = "critical"               # Near bifurcation


class TrajectoryClass(Enum):
    """Trajectory classification"""
    CONVERGING = "converging"           # Moving toward attractor
    DIVERGING = "diverging"             # Moving away from attractor
    OSCILLATING = "oscillating"         # Periodic motion
    WANDERING = "wandering"             # No clear direction


class AttractorClass(Enum):
    """Attractor type classification"""
    FIXED_POINT = "fixed_point"         # Single stable state
    LIMIT_CYCLE = "limit_cycle"         # Periodic attractor
    STRANGE = "strange"                 # Chaotic attractor
    NONE = "none"                       # No clear attractor


# =============================================================================
# OUTPUT DATACLASSES
# =============================================================================

@dataclass
class DynamicsVector:
    """
    Numerical measurements from dynamical systems analysis.
    This is the DATA output - consumed by downstream layers.
    """

    # === IDENTIFICATION ===
    timestamp: datetime = field(default_factory=datetime.now)
    entity_id: str = ""

    # === REGIME METRICS ===
    correlation_level: float = 0.0      # Current coupling strength
    correlation_change: float = 0.0     # Rate of correlation change
    regime_duration: int = 0            # Time in current regime

    # === STABILITY METRICS ===
    stability_index: float = 0.0        # Overall stability [0,1]
    volatility_trend: float = 0.0       # Change in volatility
    density_change: float = 0.0         # Network density evolution

    # === TRAJECTORY METRICS ===
    trajectory_direction: float = 0.0   # Angle in phase space
    trajectory_speed: float = 0.0       # Rate of movement
    trajectory_curvature: float = 0.0   # Change in direction

    # === ATTRACTOR METRICS ===
    attractor_distance: float = 0.0     # Distance from nearest attractor
    basin_depth: float = 0.0            # Stability of current basin

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'entity_id': self.entity_id,
            'correlation_level': self.correlation_level,
            'correlation_change': self.correlation_change,
            'regime_duration': self.regime_duration,
            'stability_index': self.stability_index,
            'volatility_trend': self.volatility_trend,
            'density_change': self.density_change,
            'trajectory_direction': self.trajectory_direction,
            'trajectory_speed': self.trajectory_speed,
            'trajectory_curvature': self.trajectory_curvature,
            'attractor_distance': self.attractor_distance,
            'basin_depth': self.basin_depth,
        }


@dataclass
class DynamicsTypology:
    """
    Classification output from dynamical systems analysis.
    This is the INTERPRETATION output - consumed by humans/reports.
    """

    # === CLASSIFICATIONS ===
    regime_class: RegimeClass = RegimeClass.MODERATE
    stability_class: StabilityClass = StabilityClass.STABLE
    trajectory_class: TrajectoryClass = TrajectoryClass.WANDERING
    attractor_class: AttractorClass = AttractorClass.NONE

    # === SUMMARY ===
    summary: str = ""
    confidence: float = 0.0
    alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'regime_class': self.regime_class.value,
            'stability_class': self.stability_class.value,
            'trajectory_class': self.trajectory_class.value,
            'attractor_class': self.attractor_class.value,
            'summary': self.summary,
            'confidence': self.confidence,
            'alerts': self.alerts,
        }


@dataclass
class DynamicalSystemsOutput:
    """Combined output from Dynamical Systems layer."""
    vector: DynamicsVector
    typology: DynamicsTypology
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# LAYER CLASS
# =============================================================================

class DynamicalSystemsLayer:
    """
    Dynamical Systems Layer - PURE ORCHESTRATOR.

    Analyzes temporal evolution of behavioral geometry to detect:
    - Regime changes
    - Stability transitions
    - Trajectory patterns
    - Attractor structures

    This layer contains NO computation - it delegates to engines.
    """

    def __init__(
        self,
        entity_id: str = "",
        config: Optional[Dict] = None,
    ):
        """
        Initialize layer.

        Args:
            entity_id: Entity being analyzed
            config: Optional configuration overrides
        """
        self.entity_id = entity_id
        self.config = config or {}

    def analyze(
        self,
        geometry_history: List[Dict[str, float]],
        timestamps: Optional[List[datetime]] = None,
    ) -> DynamicalSystemsOutput:
        """
        Analyze dynamical systems from geometry history.

        Args:
            geometry_history: List of geometry measurements over time
            timestamps: Optional timestamps for each measurement

        Returns:
            DynamicalSystemsOutput with vector and typology
        """
        if len(geometry_history) < 2:
            return self._empty_output()

        # Extract time series
        correlations = [g.get('mean_correlation', 0.0) for g in geometry_history]
        densities = [g.get('network_density', 0.0) for g in geometry_history]

        # Compute changes
        corr_changes = np.diff(correlations)
        density_changes = np.diff(densities)

        # Current values
        current_corr = correlations[-1]
        current_density = densities[-1]
        recent_corr_change = corr_changes[-1] if len(corr_changes) > 0 else 0.0
        recent_density_change = density_changes[-1] if len(density_changes) > 0 else 0.0

        # Classify regime
        regime_class = self._classify_regime(current_corr, recent_corr_change)

        # Classify stability
        stability_class = self._classify_stability(corr_changes, density_changes)

        # Classify trajectory
        trajectory_class = self._classify_trajectory(correlations, densities)

        # Build vector
        vector = DynamicsVector(
            timestamp=timestamps[-1] if timestamps else datetime.now(),
            entity_id=self.entity_id,
            correlation_level=current_corr,
            correlation_change=recent_corr_change,
            stability_index=self._compute_stability_index(corr_changes, density_changes),
            density_change=recent_density_change,
            trajectory_speed=np.sqrt(recent_corr_change**2 + recent_density_change**2),
        )

        # Build typology
        typology = DynamicsTypology(
            regime_class=regime_class,
            stability_class=stability_class,
            trajectory_class=trajectory_class,
            summary=self._generate_summary(regime_class, stability_class, trajectory_class),
            confidence=self._compute_confidence(correlations, densities),
        )

        return DynamicalSystemsOutput(
            vector=vector,
            typology=typology,
            metadata={'n_observations': len(geometry_history)},
        )

    def _classify_regime(self, correlation: float, change: float) -> RegimeClass:
        """Classify current regime."""
        if abs(change) > 0.2:
            return RegimeClass.TRANSITIONING
        elif correlation > 0.7:
            return RegimeClass.COUPLED
        elif correlation < 0.3:
            return RegimeClass.DECOUPLED
        else:
            return RegimeClass.MODERATE

    def _classify_stability(
        self,
        corr_changes: np.ndarray,
        density_changes: np.ndarray,
    ) -> StabilityClass:
        """Classify system stability."""
        total_volatility = np.std(corr_changes) + np.std(density_changes)

        if total_volatility < 0.05:
            return StabilityClass.STABLE
        elif total_volatility < 0.15:
            return StabilityClass.EVOLVING
        elif total_volatility < 0.3:
            return StabilityClass.UNSTABLE
        else:
            return StabilityClass.CRITICAL

    def _classify_trajectory(
        self,
        correlations: List[float],
        densities: List[float],
    ) -> TrajectoryClass:
        """Classify trajectory pattern."""
        if len(correlations) < 3:
            return TrajectoryClass.WANDERING

        # Check for trend
        corr_trend = np.polyfit(range(len(correlations)), correlations, 1)[0]

        # Check for oscillation
        corr_diff = np.diff(correlations)
        sign_changes = np.sum(np.diff(np.sign(corr_diff)) != 0)
        oscillation_ratio = sign_changes / len(corr_diff) if len(corr_diff) > 0 else 0

        if oscillation_ratio > 0.5:
            return TrajectoryClass.OSCILLATING
        elif corr_trend > 0.01:
            return TrajectoryClass.CONVERGING
        elif corr_trend < -0.01:
            return TrajectoryClass.DIVERGING
        else:
            return TrajectoryClass.WANDERING

    def _compute_stability_index(
        self,
        corr_changes: np.ndarray,
        density_changes: np.ndarray,
    ) -> float:
        """Compute stability index [0,1] where 1 = stable."""
        if len(corr_changes) == 0:
            return 0.5

        volatility = np.std(corr_changes) + np.std(density_changes)
        # Map volatility to stability (inverse relationship)
        stability = 1.0 / (1.0 + volatility * 5)
        return float(np.clip(stability, 0, 1))

    def _compute_confidence(
        self,
        correlations: List[float],
        densities: List[float],
    ) -> float:
        """Compute confidence in classification."""
        n = len(correlations)
        if n < 5:
            return 0.3
        elif n < 10:
            return 0.5
        elif n < 20:
            return 0.7
        else:
            return 0.9

    def _generate_summary(
        self,
        regime: RegimeClass,
        stability: StabilityClass,
        trajectory: TrajectoryClass,
    ) -> str:
        """Generate human-readable summary."""
        return f"{regime.value.title()} regime, {stability.value} system, {trajectory.value} trajectory"

    def _empty_output(self) -> DynamicalSystemsOutput:
        """Return empty output for insufficient data."""
        return DynamicalSystemsOutput(
            vector=DynamicsVector(entity_id=self.entity_id),
            typology=DynamicsTypology(summary="Insufficient data"),
            metadata={'error': 'insufficient_data'},
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_dynamics(
    geometry_history: List[Dict[str, float]],
    entity_id: str = "",
    timestamps: Optional[List[datetime]] = None,
    config: Optional[Dict] = None,
) -> DynamicalSystemsOutput:
    """
    Convenience function for dynamical systems analysis.

    Args:
        geometry_history: List of geometry measurements
        entity_id: Entity identifier
        timestamps: Optional timestamps
        config: Optional configuration

    Returns:
        DynamicalSystemsOutput
    """
    layer = DynamicalSystemsLayer(entity_id=entity_id, config=config)
    return layer.analyze(geometry_history, timestamps)
