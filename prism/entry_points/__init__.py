"""
PRISM Entry Points Registry
===========================

CLI entry points for the PRISM analysis pipeline.
Each entry point has a defined goal, inputs, outputs, and documentation.

Storage: Polars + Parquet (data/ directory)

Pipeline Architecture:

    CORE PIPELINE (6 entry points):
        fetch → signal_vector → geometry → state → physics
                                   ↓
                            (laplace computed
                             internally here)

    Entry points orchestrate. Modules compute.

Usage:
    # Core pipeline
    python -m prism.db.fetch --cmapss
    python -m prism.entry_points.signal_vector --signal --domain cmapss
    python -m prism.entry_points.geometry --signal --domain cmapss
    python -m prism.entry_points.state
    python -m prism.entry_points.physics

    # V2 Architecture (Laplace-based)
    python -m prism.entry_points.signal_vector --signal --domain cmapss
    python -m prism.laplace.compute --v2 --domain cmapss
    python -m prism.entry_points.geometry --v2 --domain cmapss
    python -m prism.entry_points.state --v2

    # Dynamical systems (moved to prism.testing)
    python -m prism.testing.dynamical --system lorenz
    python -m prism.testing.dynamic_vector
"""

from typing import Dict, Any, Optional

# =============================================================================
# ENTRY POINT REGISTRY
# =============================================================================

ENTRY_POINT_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ==========================================================================
    # CORE PIPELINE (6 entry points)
    # ==========================================================================
    'fetch': {
        'module': 'prism.db.fetch',
        'goal': 'Fetch data from external sources (USGS, climate, C-MAPSS, etc.)',
        'inputs': ['APIs', 'fetchers/yaml/*.yaml'],
        'outputs': ['raw/observations.parquet'],
    },

    'characterize': {
        'module': 'prism.modules.characterize',
        'goal': 'Inline characterization for signals (called by signal_vector)',
        'inputs': ['signal values array'],
        'outputs': ['CharacterizationResult (in-memory)'],
        'note': 'Module, not CLI entry point. Use from prism.modules.characterize import characterize_signal',
    },

    'signal_vector': {
        'module': 'prism.entry_points.signal_vector',
        'goal': 'Compute vector metrics for each signal (point-wise + windowed)',
        'inputs': ['raw/observations.parquet', 'raw/characterization.parquet'],
        'outputs': ['vector/signal.parquet', 'vector/signal_dense.parquet'],
    },

    'geometry': {
        'module': 'prism.entry_points.geometry',
        'goal': 'Compute cohort geometry (PCA, MST, clustering, LOF) + modes + wavelet. Calls laplace internally.',
        'inputs': ['vector/signal_field.parquet'],
        'outputs': ['geometry/cohort.parquet', 'geometry/signal_pair.parquet', 'geometry/snapshots_v2.parquet'],
    },

    'state': {
        'module': 'prism.entry_points.state',
        'goal': 'Derive query-time state for signals and cohorts',
        'inputs': ['geometry/cohort.parquet'],
        'outputs': ['state/signal.parquet', 'state/cohort.parquet', 'state/trajectory_v2.parquet'],
    },

    'physics': {
        'module': 'prism.entry_points.physics',
        'goal': 'Validate physics laws (energy conservation, entropy increase, least action)',
        'inputs': ['state/signal.parquet'],
        'outputs': ['physics/conservation.parquet'],
    },

    'hybrid': {
        'module': 'prism.entry_points.hybrid',
        'goal': 'Combine PRISM features with ML models for supervised prediction',
        'inputs': ['vector/signal_field.parquet', 'geometry/cohort.parquet'],
        'outputs': ['predictions (in-memory)'],
    },

    # ==========================================================================
    # LAPLACE (moved to prism.laplace package - internal, not user-facing)
    # ==========================================================================
    'laplace': {
        'module': 'prism.laplace.compute',
        'goal': 'Compute Laplace field (internal - called by geometry)',
        'inputs': ['vector/signal.parquet'],
        'outputs': ['vector/signal_field.parquet', 'vector/laplace_field_v2.parquet'],
        'note': 'Use --v2 flag for Running Laplace transform',
    },

    'laplace_pairwise': {
        'module': 'prism.laplace.pairwise',
        'goal': 'Compute pairwise geometry in Laplace field space',
        'inputs': ['vector/signal_field.parquet'],
        'outputs': ['geometry/laplace_pair.parquet'],
    },

    # ==========================================================================
    # TESTING UTILITIES (moved to prism.testing package)
    # ==========================================================================
    'generate_dynamical': {
        'module': 'prism.testing.dynamical',
        'goal': 'Generate test data from dynamical systems (Lorenz, Rossler, etc.)',
        'inputs': [],
        'outputs': ['raw/observations.parquet (dynamical)'],
    },

    'generate_pendulum_regime': {
        'module': 'prism.testing.pendulum',
        'goal': 'Generate double pendulum regime data for testing',
        'inputs': [],
        'outputs': ['raw/observations.parquet (pendulum)'],
    },

    'dynamic_vector': {
        'module': 'prism.testing.dynamic_vector',
        'goal': 'Compute vector metrics for dynamical system data',
        'inputs': ['raw/observations.parquet'],
        'outputs': ['vector/signal.parquet'],
    },

    'dynamic_state': {
        'module': 'prism.testing.dynamic_state',
        'goal': 'Compute state metrics for dynamical system data',
        'inputs': ['vector/signal.parquet'],
        'outputs': ['state/signal.parquet'],
    },

    # ==========================================================================
    # GEOMETRY MODES (moved to prism.geometry package)
    # ==========================================================================
    'mode_geometry': {
        'module': 'prism.geometry.mode_runner',
        'goal': 'Compute geometry metrics organized by discovered behavioral modes',
        'inputs': ['vector/signal_field.parquet'],
        'outputs': ['geometry/mode.parquet'],
    },

    # ==========================================================================
    # COHORT STATE (moved to prism.state package)
    # ==========================================================================
    'cohort_state': {
        'module': 'prism.state.cohort',
        'goal': 'Compute cohort-level state dynamics',
        'inputs': ['geometry/cohort.parquet'],
        'outputs': ['state/cohort.parquet'],
    },
}


def list_entry_points() -> None:
    """List all entry points."""
    print("=" * 70)
    print("PRISM ENTRY POINTS")
    print("=" * 70)
    for name, info in ENTRY_POINT_REGISTRY.items():
        print(f"\n{name}")
        print(f"  Goal: {info['goal']}")
        print(f"  Inputs: {info['inputs']}")
        print(f"  Outputs: {info['outputs']}")


def get_entry_point_info(name: str) -> Optional[Dict[str, Any]]:
    """Get info for a specific entry point."""
    return ENTRY_POINT_REGISTRY.get(name)


# =============================================================================
# PUBLIC API - Core Pipeline
# =============================================================================
try:
    from prism.entry_points.signal_vector import (
        UnivariateVector,
        UnivariateResult,
        compute_univariate,
        process_signal,
        process_all_signals,
        store_results,
    )
except ImportError:
    pass

try:
    from prism.entry_points.geometry import (
        load_cohort_members,
    )
except ImportError:
    pass

try:
    from prism.modules.laplace_transform import compute_laplace_field
    from prism.modules.laplace_compute import WindowConfig
except ImportError:
    pass
