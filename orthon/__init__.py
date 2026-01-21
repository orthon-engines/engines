"""
Orthon - Behavioral Geometry Engine for Industrial Signal Topology Analysis.

Orthon computes intrinsic properties, relational structure, and temporal dynamics
of sensor data from turbofans, bearings, hydraulic systems, and chemical processes.

Quick Start:
    import orthon

    # Read sensor data
    df = orthon.read_parquet("observations.parquet")

    # Compute behavioral metrics
    metrics = orthon.compute_hurst(signal_array)

    # Use geometry engines
    pca = orthon.PCAEngine()
    result = pca.compute(data_matrix)

For more details, see: https://github.com/prism-engines/diagnostics
"""

__version__ = "0.1.0"

# =============================================================================
# Core Types
# =============================================================================
from orthon._internal.core import DomainClock, DomainInfo, auto_detect_window

# =============================================================================
# I/O Layer
# =============================================================================
from orthon._internal.db import (
    # Path management
    get_data_root,
    get_path,
    ensure_directory,
    file_exists,
    # File constants
    OBSERVATIONS,
    VECTOR,
    SIGNALS,
    GEOMETRY,
    STATE,
    COHORTS,
    # Polars I/O
    read_parquet,
    write_parquet_atomic,
    upsert_parquet,
    append_parquet,
    # Query utilities
    describe_table,
    table_stats,
)

# =============================================================================
# Engine API
# =============================================================================
from orthon._internal.engines import (
    # Registries
    ENGINES,
    VECTOR_ENGINES,
    GEOMETRY_ENGINES,
    STATE_ENGINES,
    # Lookup functions
    get_engine,
    get_vector_engine,
    get_geometry_engine,
    get_state_engine,
    list_engines,
    list_vector_engines,
    list_geometry_engines,
    list_state_engines,
)

# =============================================================================
# Vector Engines (single-signal analysis)
# =============================================================================
from orthon._internal.engines import (
    compute_hurst,
    compute_entropy,
    compute_wavelets,
    compute_spectral,
    compute_garch,
    compute_rqa,
    compute_lyapunov,
    compute_realized_vol,
    compute_hilbert_amplitude,
    compute_hilbert_phase,
    compute_hilbert_frequency,
)

# =============================================================================
# Geometry Engines (multi-signal structure)
# =============================================================================
from orthon._internal.engines import (
    PCAEngine,
    DistanceEngine,
    ClusteringEngine,
    MutualInformationEngine,
    CopulaEngine,
    MSTEngine,
    LOFEngine,
    ConvexHullEngine,
    BarycenterEngine,
)

# =============================================================================
# State Engines (temporal dynamics)
# =============================================================================
from orthon._internal.engines import (
    CointegrationEngine,
    CrossCorrelationEngine,
    DMDEngine,
    DTWEngine,
    GrangerEngine,
    TransferEntropyEngine,
    CoupledInertiaEngine,
)

# =============================================================================
# Base Classes
# =============================================================================
from orthon._internal.engines import BaseEngine, EngineResult, EngineMetadata

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Version
    "__version__",
    # Core types
    "DomainClock",
    "DomainInfo",
    "auto_detect_window",
    # I/O
    "get_data_root",
    "get_path",
    "ensure_directory",
    "file_exists",
    "OBSERVATIONS",
    "VECTOR",
    "SIGNALS",
    "GEOMETRY",
    "STATE",
    "COHORTS",
    "read_parquet",
    "write_parquet_atomic",
    "upsert_parquet",
    "append_parquet",
    "describe_table",
    "table_stats",
    # Engine registries
    "ENGINES",
    "VECTOR_ENGINES",
    "GEOMETRY_ENGINES",
    "STATE_ENGINES",
    # Engine lookup
    "get_engine",
    "get_vector_engine",
    "get_geometry_engine",
    "get_state_engine",
    "list_engines",
    "list_vector_engines",
    "list_geometry_engines",
    "list_state_engines",
    # Vector engines
    "compute_hurst",
    "compute_entropy",
    "compute_wavelets",
    "compute_spectral",
    "compute_garch",
    "compute_rqa",
    "compute_lyapunov",
    "compute_realized_vol",
    "compute_hilbert_amplitude",
    "compute_hilbert_phase",
    "compute_hilbert_frequency",
    # Geometry engines
    "PCAEngine",
    "DistanceEngine",
    "ClusteringEngine",
    "MutualInformationEngine",
    "CopulaEngine",
    "MSTEngine",
    "LOFEngine",
    "ConvexHullEngine",
    "BarycenterEngine",
    # State engines
    "CointegrationEngine",
    "CrossCorrelationEngine",
    "DMDEngine",
    "DTWEngine",
    "GrangerEngine",
    "TransferEntropyEngine",
    "CoupledInertiaEngine",
    # Base classes
    "BaseEngine",
    "EngineResult",
    "EngineMetadata",
]
