"""
Causal Mechanics Layer
======================

Orchestrates physics engines to answer:
    "Why does this system change? What are the causal drivers?"

Sub-questions:
    - Energy: Is energy conserved? (Hamiltonian)
    - Motion: What are the equations of motion? (Lagrangian)
    - Equilibrium: Spontaneous or forced? (Gibbs)
    - Flow: How does momentum propagate? (Navier-Stokes inspired)
    - Cycles: What are the rotational dynamics? (Angular momentum)

This is a PURE ORCHESTRATOR - no computation here.
All physics calculations delegated to engines/physics/.

Part of ORTHON four-layer framework:
    - Signal Typology: What is it?
    - Behavioral Geometry: How does it behave?
    - Dynamical Systems: When/how does it change?
    - Causal Mechanics: Why does it change? (this layer)

Output:
    - MechanicsVector: Numerical measurements for downstream layers
    - MechanicsTypology: Classification for interpretation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class EnergyClass(Enum):
    """Energy regime classification (from Hamiltonian)"""
    CONSERVATIVE = "conservative"   # H is constant - closed system
    DRIVEN = "driven"               # H increasing - energy injection
    DISSIPATIVE = "dissipative"     # H decreasing - energy loss
    FLUCTUATING = "fluctuating"     # H varies irregularly


class EquilibriumClass(Enum):
    """Equilibrium tendency classification (from Gibbs)"""
    APPROACHING = "approaching"       # Moving toward equilibrium (dG < 0)
    AT_EQUILIBRIUM = "at_equilibrium" # At equilibrium (G stable)
    DEPARTING = "departing"           # Moving away from equilibrium
    FORCED = "forced"                 # Externally driven (non-spontaneous)


class FlowClass(Enum):
    """Flow regime classification (from momentum flux)"""
    LAMINAR = "laminar"             # Smooth, predictable flow
    TRANSITIONAL = "transitional"   # Between laminar and turbulent
    TURBULENT = "turbulent"         # Chaotic, unpredictable flow


class OrbitClass(Enum):
    """Phase space orbit classification (from angular momentum)"""
    CIRCULAR = "circular"           # Constant |L|, perfect oscillation
    ELLIPTICAL = "elliptical"       # Varying |L|, regular oscillation
    IRREGULAR = "irregular"         # Complex orbit shape
    LINEAR = "linear"               # No rotation, L ~ 0


class DominanceClass(Enum):
    """Energy dominance classification"""
    KINETIC = "kinetic"             # Motion dominates
    POTENTIAL = "potential"         # Position dominates
    BALANCED = "balanced"           # Equal contribution


# =============================================================================
# OUTPUT DATACLASSES
# =============================================================================

@dataclass
class MechanicsVector:
    """
    Numerical measurements from causal mechanics.
    This is the DATA output - consumed by downstream layers.
    """

    # === IDENTIFICATION ===
    entity_id: str = "unknown"
    signal_id: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)

    # === HAMILTONIAN (Energy) ===
    H_mean: float = 0.0               # Mean total energy
    H_std: float = 0.0                # Energy variability
    H_trend: float = 0.0              # dH/dt (energy injection rate)
    H_cv: float = 0.0                 # Coefficient of variation
    T_mean: float = 0.0               # Mean kinetic energy
    V_mean: float = 0.0               # Mean potential energy
    T_V_ratio: float = 1.0            # Kinetic / Potential ratio
    energy_conserved: bool = True     # Is H approximately constant?

    # === LAGRANGIAN (Motion) ===
    L_mean: float = 0.0               # Mean Lagrangian
    action: float = 0.0               # Total action (integral L dt)
    action_rate: float = 0.0          # Action per unit time
    kinetic_dominant_fraction: float = 0.5  # Fraction of time T > V

    # === GIBBS (Equilibrium) ===
    G_mean: float = 0.0               # Mean Gibbs free energy
    G_trend: float = 0.0              # dG/dt
    delta_G: float = 0.0              # G_final - G_initial
    temperature_mean: float = 1.0     # Mean "temperature" (volatility)
    entropy_mean: float = 0.0         # Mean entropy
    spontaneous: bool = False         # Is dG < 0?

    # === ANGULAR MOMENTUM (Cycles) ===
    angular_L_mean: float = 0.0       # Mean angular momentum (signed)
    angular_L_abs_mean: float = 0.0   # Mean |L|
    sign_change_rate: float = 0.0     # Direction reversals per period
    orbit_circularity: float = 0.0    # 0 = linear, 1 = circular
    orbit_stability: float = 0.0      # Consistency of |L|
    L_conserved: bool = True          # Is angular momentum conserved?

    # === MOMENTUM FLUX (Flow) ===
    p_mean: float = 0.0               # Mean momentum
    p_std: float = 0.0                # Momentum variability
    flux_mean: float = 0.0            # Mean dp/dt
    flux_std: float = 0.0             # Force variability
    reynolds_proxy: float = 1.0       # Inertial / viscous ratio
    turbulence_intensity: float = 0.0 # Flux variability measure
    inertial: bool = False            # Momentum persistent?
    viscous: bool = False             # Drag present?
    forced: bool = False              # External forcing?
    turbulent: bool = False           # Turbulent regime?

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, np.ndarray):
                result[key] = value.tolist()
            else:
                result[key] = value
        return result


@dataclass
class MechanicsTypology:
    """
    Classification from causal mechanics.
    This is the INFORMATION output - human-interpretable.
    """

    # === IDENTIFICATION ===
    entity_id: str = "unknown"
    signal_id: str = "unknown"

    # === ENERGY REGIME ===
    energy_class: EnergyClass = EnergyClass.CONSERVATIVE
    dominant_energy: DominanceClass = DominanceClass.BALANCED

    # === MOTION REGIME ===
    motion_class: DominanceClass = DominanceClass.BALANCED

    # === EQUILIBRIUM REGIME ===
    equilibrium_class: EquilibriumClass = EquilibriumClass.AT_EQUILIBRIUM
    spontaneous: bool = False

    # === CYCLICAL REGIME ===
    rotation_direction: str = "mixed"    # counterclockwise | clockwise | mixed
    orbit_class: OrbitClass = OrbitClass.LINEAR

    # === FLOW REGIME ===
    flow_class: FlowClass = FlowClass.LAMINAR
    forcing_type: str = "mixed"          # inertial | viscous | forced | mixed

    # === COMPOSITE ===
    system_class: str = "Unknown"        # Overall system classification

    # === HUMAN-READABLE ===
    summary: str = ""
    alerts: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'entity_id': self.entity_id,
            'signal_id': self.signal_id,
            'energy_class': self.energy_class.value,
            'dominant_energy': self.dominant_energy.value,
            'motion_class': self.motion_class.value,
            'equilibrium_class': self.equilibrium_class.value,
            'spontaneous': self.spontaneous,
            'rotation_direction': self.rotation_direction,
            'orbit_class': self.orbit_class.value,
            'flow_class': self.flow_class.value,
            'forcing_type': self.forcing_type,
            'system_class': self.system_class,
            'summary': self.summary,
            'alerts': self.alerts,
            'confidence': self.confidence
        }


@dataclass
class CausalMechanicsOutput:
    """Complete output from Causal Mechanics layer"""
    vector: MechanicsVector
    typology: MechanicsTypology

    def to_dict(self) -> Dict:
        """Combined dictionary for full output."""
        result = self.vector.to_dict()
        result.update(self.typology.to_dict())
        return result


# Backwards compatibility aliases
PhysicsVector = MechanicsVector
PhysicsTypology = MechanicsTypology
SystemPhysicsOutput = CausalMechanicsOutput


# =============================================================================
# LAYER IMPLEMENTATION
# =============================================================================

class CausalMechanicsLayer:
    """
    Causal Mechanics Layer: Why does this system change?

    Pure orchestrator. Calls physics engines, classifies results.
    Contains ZERO computation - only coordination and classification logic.

    Answers:
        - Is energy conserved? (Hamiltonian)
        - What type of motion? (Lagrangian)
        - Is it spontaneously equilibrating? (Gibbs)
        - What are the cyclical dynamics? (Angular momentum)
        - How does momentum flow? (Momentum flux)
    """

    def __init__(self, config: dict = None):
        self.config = config or {}

    def analyze(
        self,
        series: np.ndarray,
        entity_id: str = "unknown",
        signal_id: str = "unknown",
        previous: Optional[CausalMechanicsOutput] = None
    ) -> CausalMechanicsOutput:
        """
        Analyze causal mechanics of a signal.

        Args:
            series: 1D numpy array
            entity_id: Entity identifier
            signal_id: Signal identifier
            previous: Previous window output (for transition detection)

        Returns:
            CausalMechanicsOutput with vector and typology
        """
        # Import engines (deferred to avoid circular imports)
        from ..engines.physics import (
            hamiltonian, lagrangian, kinetic_energy, potential_energy,
            gibbs_free_energy, angular_momentum, momentum_flux
        )

        series = np.asarray(series).flatten()

        # === CALL ALL ENGINES ===
        H_result = hamiltonian.compute(series)
        L_result = lagrangian.compute(series)
        KE_result = kinetic_energy.compute(series)
        PE_result = potential_energy.compute(series)
        G_result = gibbs_free_energy.compute(series)
        AM_result = angular_momentum.compute(series)
        MF_result = momentum_flux.compute(series)

        # === BUILD VECTOR ===
        vector = self._build_vector(
            entity_id, signal_id,
            H_result, L_result, KE_result, PE_result,
            G_result, AM_result, MF_result
        )

        # === CLASSIFY ===
        typology = self._classify(
            entity_id, signal_id, vector,
            H_result, L_result, G_result, AM_result, MF_result
        )

        return CausalMechanicsOutput(vector=vector, typology=typology)

    def _build_vector(
        self, entity_id, signal_id,
        H, L, KE, PE, G, AM, MF
    ) -> MechanicsVector:
        """Assemble MechanicsVector from engine outputs."""

        return MechanicsVector(
            entity_id=entity_id,
            signal_id=signal_id,
            timestamp=datetime.now(),

            # Hamiltonian
            H_mean=H.H_mean,
            H_std=H.H_std,
            H_trend=H.H_trend,
            H_cv=H.H_cv,
            T_mean=H.T_mean,
            V_mean=H.V_mean,
            T_V_ratio=H.T_V_ratio,
            energy_conserved=H.conserved,

            # Lagrangian
            L_mean=L.L_mean,
            action=L.action,
            action_rate=L.action_rate,
            kinetic_dominant_fraction=L.kinetic_dominant_fraction,

            # Gibbs
            G_mean=G.G_mean,
            G_trend=G.G_trend,
            delta_G=G.delta_G,
            temperature_mean=G.T_mean,
            entropy_mean=G.S_mean,
            spontaneous=G.spontaneous,

            # Angular momentum
            angular_L_mean=AM.L_mean,
            angular_L_abs_mean=AM.L_abs_mean,
            sign_change_rate=AM.sign_change_rate,
            orbit_circularity=AM.orbit_circularity,
            orbit_stability=AM.orbit_stability,
            L_conserved=AM.L_conserved,

            # Momentum flux
            p_mean=MF.p_mean,
            p_std=MF.p_std,
            flux_mean=MF.flux_mean,
            flux_std=MF.flux_std,
            reynolds_proxy=MF.reynolds_proxy,
            turbulence_intensity=MF.turbulence_intensity,
            inertial=MF.inertial,
            viscous=MF.viscous,
            forced=MF.forced,
            turbulent=MF.turbulent
        )

    def _classify(
        self, entity_id, signal_id, vector,
        H, L, G, AM, MF
    ) -> MechanicsTypology:
        """Convert measurements to classification."""

        # Energy class (from Hamiltonian)
        energy_class = EnergyClass(H.regime)

        # Dominant energy
        if H.dominant_energy == "kinetic":
            dominant_energy = DominanceClass.KINETIC
        elif H.dominant_energy == "potential":
            dominant_energy = DominanceClass.POTENTIAL
        else:
            dominant_energy = DominanceClass.BALANCED

        # Motion class (from Lagrangian)
        if L.dominance == "kinetic":
            motion_class = DominanceClass.KINETIC
        elif L.dominance == "potential":
            motion_class = DominanceClass.POTENTIAL
        else:
            motion_class = DominanceClass.BALANCED

        # Equilibrium class (from Gibbs)
        equilibrium_class = EquilibriumClass(G.equilibrium_class)

        # Rotation direction
        rotation_direction = AM.rotation_direction

        # Orbit class
        if AM.orbit_circularity > 0.8:
            orbit_class = OrbitClass.CIRCULAR
        elif AM.orbit_circularity > 0.4:
            orbit_class = OrbitClass.ELLIPTICAL
        elif AM.L_abs_mean > 0.1:
            orbit_class = OrbitClass.IRREGULAR
        else:
            orbit_class = OrbitClass.LINEAR

        # Flow class
        flow_class = FlowClass(MF.flow_regime)

        # Forcing type
        forcing_type = MF.forcing_type

        # System class (composite)
        system_class = self._derive_system_class(
            energy_class, equilibrium_class, flow_class, orbit_class
        )

        # Summary and alerts
        summary, alerts = self._generate_summary(
            energy_class, equilibrium_class, flow_class, H, G
        )

        # Confidence
        confidence = self._compute_confidence(H, G, MF)

        return MechanicsTypology(
            entity_id=entity_id,
            signal_id=signal_id,
            energy_class=energy_class,
            dominant_energy=dominant_energy,
            motion_class=motion_class,
            equilibrium_class=equilibrium_class,
            spontaneous=G.spontaneous,
            rotation_direction=rotation_direction,
            orbit_class=orbit_class,
            flow_class=flow_class,
            forcing_type=forcing_type,
            system_class=system_class,
            summary=summary,
            alerts=alerts,
            confidence=confidence
        )

    def _derive_system_class(
        self, energy: EnergyClass, equilibrium: EquilibriumClass,
        flow: FlowClass, orbit: OrbitClass
    ) -> str:
        """Derive overall system classification from components."""

        # Priority-based classification
        if energy == EnergyClass.CONSERVATIVE:
            if orbit == OrbitClass.CIRCULAR:
                return "Stable Oscillator"
            elif orbit == OrbitClass.ELLIPTICAL:
                return "Quasi-Periodic"
            elif flow == FlowClass.LAMINAR:
                return "Conservative Laminar"
            else:
                return "Conservative"

        elif energy == EnergyClass.DRIVEN:
            if flow == FlowClass.TURBULENT:
                return "Turbulent Driven"
            elif equilibrium == EquilibriumClass.DEPARTING:
                return "Forced Unstable"
            else:
                return "Driven System"

        elif energy == EnergyClass.DISSIPATIVE:
            if equilibrium == EquilibriumClass.APPROACHING:
                return "Damped Equilibrating"
            else:
                return "Dissipative"

        elif flow == FlowClass.TURBULENT:
            return "Turbulent"

        elif equilibrium == EquilibriumClass.APPROACHING:
            return "Equilibrating"

        elif equilibrium == EquilibriumClass.DEPARTING:
            return "Destabilizing"

        else:
            return "Transitional"

    def _generate_summary(
        self, energy: EnergyClass, equilibrium: EquilibriumClass,
        flow: FlowClass, H, G
    ) -> tuple:
        """Generate human-readable summary and alerts."""

        alerts = []

        # Energy alerts
        if not H.conserved:
            if H.regime == "driven":
                alerts.append("Energy injection - system is being driven")
            elif H.regime == "dissipative":
                alerts.append("Energy dissipating - momentum fading")

        # Equilibrium alerts
        if equilibrium == EquilibriumClass.DEPARTING:
            alerts.append("Moving away from equilibrium - instability risk")
        elif equilibrium == EquilibriumClass.FORCED:
            alerts.append("Forced regime - unsustainable without external input")

        # Flow alerts
        if flow == FlowClass.TURBULENT:
            alerts.append("Turbulent flow - high unpredictability")

        # Gibbs spontaneity
        if G.spontaneous and G.moving_toward_equilibrium:
            alerts.append("Spontaneously equilibrating")

        summary = f"**{energy.value.title()}** | {equilibrium.value.replace('_', ' ')} | {flow.value} flow"

        return summary, alerts

    def _compute_confidence(self, H, G, MF) -> float:
        """Compute overall classification confidence."""

        # Higher confidence when measurements are clear
        energy_conf = 1.0 / (1.0 + H.H_cv) if H.H_cv < float('inf') else 0.5
        gibbs_conf = min(abs(G.G_trend) * 100 + 0.3, 1.0)
        flow_conf = 0.3 if MF.turbulent else 0.8

        confidence = (energy_conf + gibbs_conf + flow_conf) / 3

        return float(np.clip(confidence, 0, 1))


# Backwards compatibility alias
SystemPhysicsLayer = CausalMechanicsLayer


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def analyze_mechanics(
    series: np.ndarray,
    entity_id: str = "unknown",
    signal_id: str = "unknown"
) -> CausalMechanicsOutput:
    """
    Convenience function for quick causal mechanics analysis.

    Example:
        result = analyze_mechanics(my_series)
        print(result.typology.summary)
        print(result.typology.system_class)
    """
    layer = CausalMechanicsLayer()
    return layer.analyze(series, entity_id, signal_id)


# Backwards compatibility alias
analyze_physics = analyze_mechanics
