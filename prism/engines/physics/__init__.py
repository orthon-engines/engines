"""
PRISM Physics Engines
=====================

Process engineering calculations requiring iterative solvers:
- Phase equilibria (VLE, LLE, flash)
- Reaction kinetics (Arrhenius, rate laws)
- Separation processes (distillation, absorption)
- Electrochemistry (Butler-Volmer)
"""

from . import phase_equilibria
from . import activity_models
from . import reaction_kinetics
from . import separations
from . import electrochemistry

__all__ = [
    'phase_equilibria',
    'activity_models',
    'reaction_kinetics',
    'separations',
    'electrochemistry',
]
