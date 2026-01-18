"""
PRISM Engines Package.

Unified engine registry for all PRISM analysis engines.
Engines are callable tools that can be invoked by an orchestrator.

Engine Categories:
- Vector: Single-signal signal topology analysis (entropy, hurst, garch, etc.)
- Geometry: Multi-signal relational structure (pca, clustering, distance, etc.)
- State: Temporal dynamics and causality (granger, cointegration, dtw, etc.)

Usage:
    from prism.engines import get_engine, list_engines, ENGINES

    # Get a specific engine
    compute_fn = get_engine("hurst")
    metrics = compute_fn(values_array)

    # List all available engines
    for name in list_engines():
        print(name)
"""

from typing import Dict, Callable, Type, List, Union, Any
import numpy as np

# Base classes
from prism.engines.engine_base import BaseEngine, EngineResult, get_window_dates
from prism.engines.metadata import EngineMetadata

# State vector dataclass
from prism.engines.state.state_vector import StateVector

# =============================================================================
# Vector Engines (functional interface)
# =============================================================================
from prism.engines.windowed.hurst import compute_hurst, HurstEngine
from prism.engines.windowed.entropy import compute_entropy
from prism.engines.windowed.wavelet import compute_wavelets, WaveletEngine
from prism.engines.windowed.spectral import compute_spectral, SpectralEngine
from prism.engines.windowed.garch import compute_garch, GARCHEngine
from prism.engines.windowed.rqa import compute_rqa, RQAEngine
from prism.engines.windowed.lyapunov import compute_lyapunov, LyapunovEngine
from prism.engines.windowed.realized_vol import compute_realized_vol, RealizedVolEngine
from prism.engines.pointwise.hilbert import (
    HilbertEngine,
    compute_hilbert_amplitude,
    compute_hilbert_phase,
    compute_hilbert_frequency,
)

# =============================================================================
# Geometry Engines (class interface)
# =============================================================================
from prism.engines.geometry.pca import PCAEngine
from prism.engines.geometry.distance import DistanceEngine
from prism.engines.geometry.clustering import ClusteringEngine
from prism.engines.geometry.mutual_information import MutualInformationEngine
from prism.engines.geometry.copula import CopulaEngine
from prism.engines.geometry.mst import MSTEngine
from prism.engines.geometry.lof import LOFEngine
from prism.engines.geometry.convex_hull import ConvexHullEngine
from prism.engines.geometry.barycenter import BarycenterEngine, compute_barycenter

# =============================================================================
# State Engines (class interface)
# =============================================================================
from prism.engines.state.cointegration import CointegrationEngine
from prism.engines.state.cross_correlation import CrossCorrelationEngine
from prism.engines.state.dmd import DMDEngine
from prism.engines.state.dtw import DTWEngine
from prism.engines.state.granger import GrangerEngine
from prism.engines.state.transfer_entropy import TransferEntropyEngine
from prism.engines.state.coupled_inertia import CoupledInertiaEngine

# =============================================================================
# Temporal Dynamics Engines (analyze geometry evolution)
# =============================================================================
from prism.engines.state.energy_dynamics import EnergyDynamicsEngine, compute_energy_dynamics
from prism.engines.state.tension_dynamics import TensionDynamicsEngine, compute_tension_dynamics
from prism.engines.state.phase_detector import PhaseDetectorEngine, detect_phase
from prism.engines.cohort_aggregator import CohortAggregatorEngine, aggregate_cohort
from prism.engines.state.transfer_detector import TransferDetectorEngine, detect_transfer

# =============================================================================
# Observation-Level Engines (run BEFORE windowing, point precision)
# =============================================================================
from prism.engines.state.break_detector import (
    compute_breaks,
    compute_breaks_polars,
    compute_break_summary_polars,
    analyze_break_pattern,
    get_break_metrics,
    create_adaptive_windows,
    identify_break_regions,
    BreakPattern,
    DEFAULT_CONFIG as BREAK_DETECTOR_CONFIG,
)

from prism.engines.heaviside import (
    compute_heaviside,
    get_heaviside_metrics,
    identify_steps,
    reconstruct_step_function,
    compute_step_residual,
    StepEvent,
    DEFAULT_CONFIG as HEAVISIDE_CONFIG,
)

from prism.engines.dirac import (
    compute_dirac,
    get_dirac_metrics,
    identify_impulses,
    reconstruct_impulse_response,
    compute_impulse_residual,
    ImpulseEvent,
    DEFAULT_CONFIG as DIRAC_CONFIG,
)


# =============================================================================
# Engine Registries
# =============================================================================

# Vector engines: name -> compute function (9 canonical)
VECTOR_ENGINES: Dict[str, Callable[[np.ndarray], dict]] = {
    "hurst": compute_hurst,
    "entropy": compute_entropy,
    "wavelet": compute_wavelets,
    "spectral": compute_spectral,
    "garch": compute_garch,
    "rqa": compute_rqa,
    "lyapunov": compute_lyapunov,
    "realized_vol": compute_realized_vol,  # 13 metrics: vol, drawdown, distribution
    "hilbert_amplitude": compute_hilbert_amplitude,  # Instantaneous amplitude
    "hilbert_phase": compute_hilbert_phase,  # Instantaneous phase
    "hilbert_frequency": compute_hilbert_frequency,  # Instantaneous frequency
}

# Geometry engines: name -> class (9 canonical engines)
GEOMETRY_ENGINES: Dict[str, Type[BaseEngine]] = {
    "pca": PCAEngine,
    "distance": DistanceEngine,
    "clustering": ClusteringEngine,
    "mutual_information": MutualInformationEngine,
    "copula": CopulaEngine,
    "mst": MSTEngine,
    "lof": LOFEngine,
    "convex_hull": ConvexHullEngine,
    "barycenter": BarycenterEngine,
}

# State engines: name -> class
STATE_ENGINES: Dict[str, Type[BaseEngine]] = {
    "cointegration": CointegrationEngine,
    "cross_correlation": CrossCorrelationEngine,
    "dmd": DMDEngine,
    "dtw": DTWEngine,
    "granger": GrangerEngine,
    "transfer_entropy": TransferEntropyEngine,
    "coupled_inertia": CoupledInertiaEngine,
}

# Temporal dynamics engines: name -> class
# These analyze geometry evolution over time
TEMPORAL_DYNAMICS_ENGINES: Dict[str, Type] = {
    "energy_dynamics": EnergyDynamicsEngine,
    "tension_dynamics": TensionDynamicsEngine,
    "phase_detector": PhaseDetectorEngine,
    "cohort_aggregator": CohortAggregatorEngine,
    "transfer_detector": TransferDetectorEngine,
}

# Observation-level engines: name -> compute function
# These run BEFORE windowing at point precision
# Discontinuity engines form a hierarchy:
#   break_detector -> screens for ALL discontinuities
#   heaviside -> measures PERSISTENT level shifts (steps)
#   dirac -> measures TRANSIENT shocks (impulses)
OBSERVATION_ENGINES: Dict[str, Callable] = {
    "break_detector": get_break_metrics,
    "heaviside": get_heaviside_metrics,
    "dirac": get_dirac_metrics,
}

# Unified registry: all engines
ENGINES: Dict[str, Union[Callable, Type[BaseEngine]]] = {
    **VECTOR_ENGINES,
    **GEOMETRY_ENGINES,
    **STATE_ENGINES,
    **TEMPORAL_DYNAMICS_ENGINES,
    **OBSERVATION_ENGINES,
}


# =============================================================================
# Public API
# =============================================================================

def get_engine(name: str) -> Union[Callable[[np.ndarray], dict], Type[BaseEngine]]:
    """
    Get an engine by name.

    Args:
        name: Engine name (e.g., 'hurst', 'pca', 'granger')

    Returns:
        For vector engines: compute function (callable)
        For geometry/state engines: engine class

    Raises:
        KeyError: If engine not found
    """
    if name not in ENGINES:
        available = ", ".join(sorted(ENGINES.keys()))
        raise KeyError(f"Unknown engine: {name}. Available: {available}")
    return ENGINES[name]


def get_vector_engine(name: str) -> Callable[[np.ndarray], dict]:
    """Get a vector engine compute function by name."""
    if name not in VECTOR_ENGINES:
        available = ", ".join(sorted(VECTOR_ENGINES.keys()))
        raise KeyError(f"Unknown vector engine: {name}. Available: {available}")
    return VECTOR_ENGINES[name]


def get_geometry_engine(name: str) -> Type[BaseEngine]:
    """Get a geometry engine class by name."""
    if name not in GEOMETRY_ENGINES:
        available = ", ".join(sorted(GEOMETRY_ENGINES.keys()))
        raise KeyError(f"Unknown geometry engine: {name}. Available: {available}")
    return GEOMETRY_ENGINES[name]


def get_state_engine(name: str) -> Type[BaseEngine]:
    """Get a state engine class by name."""
    if name not in STATE_ENGINES:
        available = ", ".join(sorted(STATE_ENGINES.keys()))
        raise KeyError(f"Unknown state engine: {name}. Available: {available}")
    return STATE_ENGINES[name]


def list_engines() -> List[str]:
    """Get sorted list of all available engine names."""
    return sorted(ENGINES.keys())


def list_vector_engines() -> List[str]:
    """Get sorted list of vector engine names."""
    return sorted(VECTOR_ENGINES.keys())


def list_geometry_engines() -> List[str]:
    """Get sorted list of geometry engine names."""
    return sorted(GEOMETRY_ENGINES.keys())


def list_state_engines() -> List[str]:
    """Get sorted list of state engine names."""
    return sorted(STATE_ENGINES.keys())


def get_all_engines() -> Dict[str, Union[Callable, Type[BaseEngine]]]:
    """Get all engines as a dict."""
    return ENGINES.copy()


def get_all_vector_engines() -> Dict[str, Callable[[np.ndarray], dict]]:
    """Get all vector engines as a dict."""
    return VECTOR_ENGINES.copy()


def get_all_geometry_engines() -> Dict[str, Type[BaseEngine]]:
    """Get all geometry engines as a dict."""
    return GEOMETRY_ENGINES.copy()


def get_all_state_engines() -> Dict[str, Type[BaseEngine]]:
    """Get all state engines as a dict."""
    return STATE_ENGINES.copy()


# =============================================================================
# Backwards Compatibility (deprecated - use new API)
# =============================================================================

# These match the old prism.vector_engines API
def get_vector_engines() -> Dict[str, Callable[[np.ndarray], dict]]:
    """Deprecated: use get_all_vector_engines()"""
    return get_all_vector_engines()


# These match the old prism.geometry_engines API
def get_geometry_engines() -> Dict[str, Type[BaseEngine]]:
    """Deprecated: use get_all_geometry_engines()"""
    return get_all_geometry_engines()


def get_behavioral_engines() -> Dict[str, Type[BaseEngine]]:
    """Deprecated: use get_all_geometry_engines()"""
    return get_all_geometry_engines()


# These match the old prism.state_engines API
def get_state_engines() -> Dict[str, Type[BaseEngine]]:
    """Deprecated: use get_all_state_engines()"""
    return get_all_state_engines()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base classes
    "BaseEngine",
    "EngineResult",
    "EngineMetadata",
    "StateVector",
    "get_window_dates",

    # Registries
    "ENGINES",
    "VECTOR_ENGINES",
    "GEOMETRY_ENGINES",
    "STATE_ENGINES",
    "TEMPORAL_DYNAMICS_ENGINES",

    # API functions
    "get_engine",
    "get_vector_engine",
    "get_geometry_engine",
    "get_state_engine",
    "list_engines",
    "list_vector_engines",
    "list_geometry_engines",
    "list_state_engines",
    "get_all_engines",
    "get_all_vector_engines",
    "get_all_geometry_engines",
    "get_all_state_engines",

    # Backwards compatibility
    "get_vector_engines",
    "get_geometry_engines",
    "get_behavioral_engines",
    "get_state_engines",

    # Vector engine functions (9 canonical)
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
    "HilbertEngine",

    # Vector engine classes (legacy)
    "HurstEngine",
    "WaveletEngine",
    "SpectralEngine",
    "GARCHEngine",
    "RQAEngine",
    "LyapunovEngine",
    "RealizedVolEngine",

    # Geometry engine classes (9 canonical)
    "PCAEngine",
    "DistanceEngine",
    "ClusteringEngine",
    "MutualInformationEngine",
    "CopulaEngine",
    "MSTEngine",
    "LOFEngine",
    "ConvexHullEngine",
    "BarycenterEngine",
    "compute_barycenter",

    # State engine classes
    "CointegrationEngine",
    "CrossCorrelationEngine",
    "DMDEngine",
    "DTWEngine",
    "GrangerEngine",
    "TransferEntropyEngine",
    "CoupledInertiaEngine",

    # Temporal dynamics engines
    "TEMPORAL_DYNAMICS_ENGINES",
    "EnergyDynamicsEngine",
    "TensionDynamicsEngine",
    "PhaseDetectorEngine",
    "CohortAggregatorEngine",
    "TransferDetectorEngine",

    # Temporal dynamics functions
    "compute_energy_dynamics",
    "compute_tension_dynamics",
    "detect_phase",
    "aggregate_cohort",
    "detect_transfer",

    # Observation-level engines (discontinuity detection)
    "OBSERVATION_ENGINES",

    # Break detector
    "compute_breaks",
    "compute_breaks_polars",
    "compute_break_summary_polars",
    "analyze_break_pattern",
    "get_break_metrics",
    "create_adaptive_windows",
    "identify_break_regions",
    "BreakPattern",
    "BREAK_DETECTOR_CONFIG",

    # Heaviside (step function measurement)
    "compute_heaviside",
    "get_heaviside_metrics",
    "identify_steps",
    "reconstruct_step_function",
    "compute_step_residual",
    "StepEvent",
    "HEAVISIDE_CONFIG",

    # Dirac (impulse measurement)
    "compute_dirac",
    "get_dirac_metrics",
    "identify_impulses",
    "reconstruct_impulse_response",
    "compute_impulse_residual",
    "ImpulseEvent",
    "DIRAC_CONFIG",
]
