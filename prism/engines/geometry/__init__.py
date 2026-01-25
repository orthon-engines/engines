"""
PRISM Geometry Engines
======================

Structural relationship analysis between signals.

Class-based Engines (BaseEngine):
    - PCAEngine: Principal Component Analysis (dimensionality, loadings)
    - MSTEngine: Minimum Spanning Tree (network topology)
    - ClusteringEngine: Hierarchical/K-means clustering
    - LOFEngine: Local Outlier Factor (anomaly detection)
    - DistanceEngine: Euclidean/Mahalanobis/Cosine distances
    - ConvexHullEngine: Geometric enclosure metrics
    - CopulaEngine: Dependency structure beyond correlation
    - MutualInformationEngine: Non-linear dependence (bits)
    - BarycenterEngine: Centroid and dispersion metrics

Function-based Engines:
    - compute_coupling_matrix: Signal coupling in Laplace domain
    - compute_divergence: Distribution divergence (KL, JS)
    - discover_modes: Behavioral mode discovery
    - compute_snapshot: Point-in-time structural snapshot
"""

# Class-based engines
from .pca import PCAEngine, METADATA as PCA_METADATA
from .mst import MSTEngine, METADATA as MST_METADATA
from .clustering import ClusteringEngine, METADATA as CLUSTERING_METADATA
from .lof import LOFEngine, METADATA as LOF_METADATA
from .distance import DistanceEngine, METADATA as DISTANCE_METADATA
from .convex_hull import ConvexHullEngine, METADATA as HULL_METADATA
from .copula import CopulaEngine, METADATA as COPULA_METADATA
from .mutual_information import MutualInformationEngine, METADATA as MI_METADATA
from .barycenter import BarycenterEngine, METADATA as BARYCENTER_METADATA

# Function-based engines
from .coupling import compute_coupling_matrix
from .divergence import compute_divergence
from .modes import discover_modes, extract_laplace_fingerprint, extract_cohort_fingerprints
from .snapshot import compute_geometry_at_t as compute_snapshot

# All class-based geometry engines
GEOMETRY_ENGINES = [
    PCAEngine,
    MSTEngine,
    ClusteringEngine,
    LOFEngine,
    DistanceEngine,
    ConvexHullEngine,
    CopulaEngine,
    MutualInformationEngine,
    BarycenterEngine,
]

__all__ = [
    # Class-based Engines
    'PCAEngine',
    'MSTEngine',
    'ClusteringEngine',
    'LOFEngine',
    'DistanceEngine',
    'ConvexHullEngine',
    'CopulaEngine',
    'MutualInformationEngine',
    'BarycenterEngine',

    # Function-based engines
    'compute_coupling_matrix',
    'compute_divergence',
    'discover_modes',
    'extract_laplace_fingerprint',
    'extract_cohort_fingerprints',
    'compute_snapshot',

    # Metadata
    'PCA_METADATA',
    'MST_METADATA',
    'CLUSTERING_METADATA',
    'LOF_METADATA',
    'DISTANCE_METADATA',
    'HULL_METADATA',
    'COPULA_METADATA',
    'MI_METADATA',
    'BARYCENTER_METADATA',

    # Engine list
    'GEOMETRY_ENGINES',
]
