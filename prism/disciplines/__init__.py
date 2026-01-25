"""
PRISM Disciplines

User-selectable domain-specific analysis modules.

Core analysis (vector, geometry, dynamics, state, systems) always runs.
Disciplines add domain-specific physics calculations when:
1. User selects the discipline
2. Required signals/constants are available

Available Disciplines:
- thermodynamics: Energy, entropy, equations of state
- transport: Heat, mass, and momentum transfer
- reaction: Kinetics, reactor design, yields
- controls: Dynamics, stability, feedback
- mechanics: Vibration, fatigue, stress
- electrical: Impedance, power, batteries
- fluid_dynamics: Velocity fields, vorticity, turbulence
"""

from prism.disciplines.registry import DISCIPLINES
from prism.disciplines.requirements import (
    get_requirements_text,
    check_requirements,
    get_available_engines,
)

__all__ = [
    'DISCIPLINES',
    'get_requirements_text',
    'check_requirements',
    'get_available_engines',
]
