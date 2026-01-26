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

# =============================================================================
# Vector Engines (functional interface) - Core engines
# =============================================================================
from prism.engines.core.windowed.hurst import compute_hurst
from prism.engines.core.windowed.entropy import compute_entropy
from prism.engines.core.windowed.wavelet import compute_wavelets
from prism.engines.core.windowed.spectral import compute_spectral
from prism.engines.core.windowed.garch import compute_garch
from prism.engines.core.windowed.rqa import compute_rqa
from prism.engines.core.windowed.lyapunov import compute_lyapunov
from prism.engines.core.windowed.realized_vol import compute_realized_vol
from prism.engines.core.pointwise.hilbert import (
    compute_hilbert_amplitude,
    compute_hilbert_phase,
    compute_hilbert_frequency,
)

# =============================================================================
# Geometry Engines (class interface) - Core engines
# =============================================================================
from prism.engines.core.geometry.pca import PCAEngine
from prism.engines.core.geometry.distance import DistanceEngine
from prism.engines.core.geometry.clustering import ClusteringEngine
from prism.engines.core.geometry.mutual_information import MutualInformationEngine
from prism.engines.core.geometry.copula import CopulaEngine
from prism.engines.core.geometry.mst import MSTEngine
from prism.engines.core.geometry.lof import LOFEngine
from prism.engines.core.geometry.convex_hull import ConvexHullEngine

# Domain-specific geometry (PRISM degradation model)
from prism.engines.domains.prism.barycenter import BarycenterEngine, compute_barycenter

# =============================================================================
# State Engines (class interface) - Core engines
# =============================================================================
from prism.engines.core.state.cointegration import CointegrationEngine
from prism.engines.core.state.cross_correlation import CrossCorrelationEngine
from prism.engines.core.state.dmd import DMDEngine
from prism.engines.core.state.dtw import DTWEngine
from prism.engines.core.state.granger import GrangerEngine
from prism.engines.core.state.transfer_entropy import TransferEntropyEngine
from prism.engines.core.state.coupled_inertia import CoupledInertiaEngine

# =============================================================================
# Temporal Dynamics Engines (PRISM domain - analyze geometry evolution)
# =============================================================================
from prism.engines.domains.prism.energy_dynamics import EnergyDynamicsEngine, compute_energy_dynamics
from prism.engines.domains.prism.tension_dynamics import TensionDynamicsEngine, compute_tension_dynamics

# Detection engines (honestly named) - Core engines
from prism.engines.core.detection.step_detector import compute as compute_step
from prism.engines.core.detection.spike_detector import compute as compute_spike



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
}

# Observation-level engines: name -> compute function
# These run BEFORE windowing at point precision
# Discontinuity engines:
#   heaviside -> measures PERSISTENT level shifts (steps)
#   dirac -> measures TRANSIENT shocks (impulses)
OBSERVATION_ENGINES: Dict[str, Callable] = {
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
# Domain Engine Loading API
# =============================================================================

from prism.engines.core import CORE_ENGINE_CATEGORIES
from prism.engines.domains import AVAILABLE_DOMAINS


def get_domain_engines(domain: str) -> Dict[str, Any]:
    """
    Get engines from a specific domain.

    Args:
        domain: Domain name (e.g., 'prism', 'battery', 'fluid', 'chemical')

    Returns:
        Dict mapping engine names to engine functions/classes

    Raises:
        ValueError: If domain not recognized
    """
    if domain not in AVAILABLE_DOMAINS:
        raise ValueError(
            f"Unknown domain: {domain}. "
            f"Available: {', '.join(AVAILABLE_DOMAINS)}"
        )

    # Dynamic import of domain module
    import importlib
    module = importlib.import_module(f'prism.engines.domains.{domain}')

    # Extract all exportable items from domain
    engines = {}
    for name in getattr(module, '__all__', []):
        engines[name] = getattr(module, name)

    return engines


def list_domain_engines(domain: str) -> List[str]:
    """List engine names available in a domain."""
    engines = get_domain_engines(domain)
    return sorted(engines.keys())


def list_available_domains() -> List[str]:
    """List all available domain names."""
    return AVAILABLE_DOMAINS.copy()


def load_engines_for_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load engines based on configuration.

    Always loads core engines. Optionally loads domain engines
    if config specifies a domain.

    Args:
        config: Configuration dict with optional 'domain' key

    Returns:
        Dict mapping engine names to engine functions/classes

    Example:
        config = {'domain': 'prism'}
        engines = load_engines_for_config(config)
        # Returns all core engines + prism domain engines
    """
    # Start with all currently registered engines (core)
    engines = ENGINES.copy()

    # Optionally add domain-specific engines
    domain = config.get('domain')
    if domain:
        domain_engines = get_domain_engines(domain)
        engines.update(domain_engines)

    return engines


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base classes
    "BaseEngine",
    "EngineResult",
    "EngineMetadata",
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

    # Domain engine API
    "CORE_ENGINE_CATEGORIES",
    "AVAILABLE_DOMAINS",
    "get_domain_engines",
    "list_domain_engines",
    "list_available_domains",
    "load_engines_for_config",

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

    # Temporal dynamics functions
    "compute_energy_dynamics",
    "compute_tension_dynamics",

    # Observation-level engines (discontinuity detection)
    "OBSERVATION_ENGINES",

    # Heaviside (step function measurement)
    "compute_heaviside",
    "get_heaviside_metrics",

    # Dirac (impulse measurement)
    "compute_dirac",
    "get_dirac_metrics",
]
