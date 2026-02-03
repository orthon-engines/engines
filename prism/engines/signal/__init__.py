"""
PRISM Signal Engines.

Each engine computes ONE thing. Imports from primitives.
PRISM computes, ORTHON classifies.
"""

from . import statistics  # kurtosis, skewness, crest_factor
from . import memory      # hurst, dfa, acf_decay
from . import complexity  # sample_entropy, permutation_entropy, approximate_entropy
from . import spectral    # dominant_freq, spectral_entropy, spectral_centroid
from . import trend       # trend_slope, mann_kendall, rate_of_change
from . import rms
from . import peak
from . import envelope
from . import frequency_bands
from . import harmonics
from . import hurst       # legacy alias for memory
from . import lyapunov
from . import garch
from . import attractor
from . import dmd
from . import physics_stack
from . import pulsation_index
from . import rate_of_change
from . import time_constant
from . import cycle_counting
from . import basin
from . import lof

__all__ = [
    'statistics',
    'memory',
    'complexity',
    'spectral',
    'trend',
    'rms',
    'peak',
    'envelope',
    'frequency_bands',
    'harmonics',
    'hurst',
    'lyapunov',
    'garch',
    'attractor',
    'dmd',
    'physics_stack',
    'pulsation_index',
    'rate_of_change',
    'time_constant',
    'cycle_counting',
    'basin',
    'lof',
]
