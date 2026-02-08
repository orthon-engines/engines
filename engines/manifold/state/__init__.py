"""
State Engines.

System-level computations.
- centroid: state_vector (WHERE the system is)
- eigendecomp: state_geometry (SHAPE of the system)
"""

from . import centroid
from . import eigendecomp

__all__ = ['centroid', 'eigendecomp']
