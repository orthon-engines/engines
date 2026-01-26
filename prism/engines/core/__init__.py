"""
PRISM Core Engines - Irreducible Algorithms
============================================

These engines cannot be expressed in SQL. They require:
- Iterative algorithms (hurst, lyapunov, garch)
- Matrix decomposition (pca, dmd)
- Optimization (cointegration, granger)
- Complex number arithmetic (fft)
- Pattern matching (entropy, rqa)
- Graph algorithms (mst, clustering)
"""

from . import hurst
from . import lyapunov
from . import fft
from . import pca
from . import umap
from . import garch
from . import entropy
from . import wavelet
from . import rqa
from . import granger
from . import transfer_entropy
from . import cointegration
from . import dtw
from . import dmd
from . import embedding
from . import mutual_info
from . import clustering
from . import mst
from . import copula
from . import divergence

__all__ = [
    'hurst',
    'lyapunov',
    'fft',
    'pca',
    'umap',
    'garch',
    'entropy',
    'wavelet',
    'rqa',
    'granger',
    'transfer_entropy',
    'cointegration',
    'dtw',
    'dmd',
    'embedding',
    'mutual_info',
    'clustering',
    'mst',
    'copula',
    'divergence',
]
