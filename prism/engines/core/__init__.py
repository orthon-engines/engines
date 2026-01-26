"""
PRISM Core Engines - Irreducible Algorithms
============================================

These engines cannot be expressed in SQL. They require:
- Iterative algorithms (hurst, lyapunov, garch)
- Matrix decomposition (pca, dmd)
- Optimization (cointegration, granger)
- Complex number arithmetic (fft, hilbert)
- Pattern matching (entropy, rqa)
- Graph algorithms (mst, clustering)
- Density estimation (lof)
- Geometric algorithms (convex_hull)
"""

# Memory / Long-range dependence
from . import hurst
from . import acf_decay
from . import spectral_slope

# Dynamics / Nonlinear
from . import lyapunov
from . import embedding
from . import attractor
from . import basin

# Frequency / Spectral
from . import fft
from . import wavelet
from . import hilbert

# Information / Complexity
from . import entropy
from . import entropy_rate

# Volatility
from . import garch

# Recurrence
from . import rqa

# State / Pairwise
from . import granger
from . import transfer_entropy
from . import cointegration
from . import dtw
from . import dmd

# Geometry / Structure
from . import pca
from . import umap
from . import clustering
from . import mst
from . import mutual_info
from . import copula
from . import divergence
from . import convex_hull
from . import lof

__all__ = [
    # Memory
    'hurst',
    'acf_decay',
    'spectral_slope',
    # Dynamics
    'lyapunov',
    'embedding',
    'attractor',
    'basin',
    # Frequency
    'fft',
    'wavelet',
    'hilbert',
    # Information
    'entropy',
    'entropy_rate',
    # Volatility
    'garch',
    # Recurrence
    'rqa',
    # State
    'granger',
    'transfer_entropy',
    'cointegration',
    'dtw',
    'dmd',
    # Geometry
    'pca',
    'umap',
    'clustering',
    'mst',
    'mutual_info',
    'copula',
    'divergence',
    'convex_hull',
    'lof',
]
