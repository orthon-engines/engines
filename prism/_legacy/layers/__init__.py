"""
PRISM Analytical Layers
=======================

Pure orchestrators that call engines and produce meaning.
Contains ZERO computation - all computation lives in engines/.

ORTHON Four-Layer Framework:
    1. Signal Typology: What is it?
    2. Behavioral Geometry: How does it behave?
    3. Dynamical Systems: When/how does it change?
    4. Causal Mechanics: Why does it change?

Rule: If you see `np.` or `scipy.` in a layer file, STOP.
      That computation belongs in an engine.

Usage:
    from prism.layers import (
        SignalTypologyLayer,
        BehavioralGeometryLayer,
        DynamicalSystemsLayer,
        CausalMechanicsLayer,
    )
"""

# Layer 1: Signal Typology
from .signal_typology import SignalTypologyLayer

# Layer 2: Behavioral Geometry
from .behavioral_geometry import (
    BehavioralGeometryLayer,
    BehavioralGeometryOutput,
    GeometryVector,
    GeometryTypology,
    analyze_geometry,
    TopologyClass,
    StabilityClass,
    LeadershipClass,
)

# Layer 3: Dynamical Systems
from .dynamical_systems import (
    DynamicalSystemsLayer,
    DynamicalSystemsOutput,
    DynamicsVector,
    DynamicsTypology,
    analyze_dynamics,
    RegimeClass,
    StabilityClass as DynamicsStabilityClass,
    TrajectoryClass,
    AttractorClass,
)

# Layer 4: Causal Mechanics
from .causal_mechanics import (
    CausalMechanicsLayer,
    CausalMechanicsOutput,
    MechanicsVector,
    MechanicsTypology,
    analyze_mechanics,
    EnergyClass,
    EquilibriumClass,
    FlowClass,
    OrbitClass,
    DominanceClass,
)

__all__ = [
    # Layer 1: Signal Typology
    'SignalTypologyLayer',

    # Layer 2: Behavioral Geometry
    'BehavioralGeometryLayer',
    'BehavioralGeometryOutput',
    'GeometryVector',
    'GeometryTypology',
    'analyze_geometry',
    'TopologyClass',
    'StabilityClass',
    'LeadershipClass',

    # Layer 3: Dynamical Systems
    'DynamicalSystemsLayer',
    'DynamicalSystemsOutput',
    'DynamicsVector',
    'DynamicsTypology',
    'analyze_dynamics',
    'RegimeClass',
    'DynamicsStabilityClass',
    'TrajectoryClass',
    'AttractorClass',

    # Layer 4: Causal Mechanics
    'CausalMechanicsLayer',
    'CausalMechanicsOutput',
    'MechanicsVector',
    'MechanicsTypology',
    'analyze_mechanics',
    'EnergyClass',
    'EquilibriumClass',
    'FlowClass',
    'OrbitClass',
    'DominanceClass',
]
