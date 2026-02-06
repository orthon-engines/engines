"""
PRISM Entry Points - Orchestration Layer
=========================================

Entry points are pure orchestration: read parquet → call engines → write parquet.
No computation logic lives here - that belongs in engines or primitives.

Core Pipeline (01-09):
    stage_01_signal_vector      observations → signal_vector.parquet
    stage_02_state_vector       signal_vector → state_vector.parquet
    stage_03_state_geometry     → state_geometry.parquet
    stage_04_cohorts            → cohorts.parquet
    stage_05_breaks             → breaks.parquet

Extended Analysis (10-19):
    stage_10_pairwise           correlation, causality, mutual info
    stage_11_dynamics           Lyapunov, attractors, phase space
    stage_12_information_flow   transfer entropy, causal networks
    stage_13_topology           TDA, persistence diagrams

Postprocessing (20-29):
    stage_21_statistics         summary statistics

Usage:
    python -m prism.entry_points.stage_01_signal_vector manifest.yaml
    python -m prism.entry_points.stage_02_state_vector signal_vector.parquet typology.parquet
    python -m prism.entry_points.stage_03_state_geometry signal_vector.parquet state_vector.parquet
"""

# Core pipeline
from . import stage_01_signal_vector
from . import stage_02_state_vector
from . import stage_03_state_geometry
from . import stage_04_cohorts
from . import stage_05_breaks

# Extended analysis
from . import stage_10_pairwise
from . import stage_11_dynamics
from . import stage_12_information_flow
from . import stage_13_topology

# Postprocessing
from . import stage_21_statistics

# Aliases for convenience
signal_vector = stage_01_signal_vector
state_vector = stage_02_state_vector
state_geometry = stage_03_state_geometry
cohorts = stage_04_cohorts
breaks = stage_05_breaks
pairwise = stage_10_pairwise
dynamics = stage_11_dynamics
information_flow = stage_12_information_flow
topology = stage_13_topology
statistics = stage_21_statistics

# Re-export main functions for backward compatibility
run = stage_01_signal_vector.run
run_from_manifest = stage_01_signal_vector.run_from_manifest
compute_state_vector = stage_02_state_vector.compute_state_vector
compute_centroid = stage_02_state_vector.compute_centroid
compute_state_geometry = stage_03_state_geometry.compute_state_geometry
compute_eigenvalues = stage_03_state_geometry.compute_eigenvalues

__all__ = [
    # Core pipeline modules
    'stage_01_signal_vector',
    'stage_02_state_vector',
    'stage_03_state_geometry',
    'stage_04_cohorts',
    'stage_05_breaks',
    # Extended analysis
    'stage_10_pairwise',
    'stage_11_dynamics',
    'stage_12_information_flow',
    'stage_13_topology',
    # Postprocessing
    'stage_21_statistics',
    # Aliases
    'signal_vector',
    'state_vector',
    'state_geometry',
    'cohorts',
    'breaks',
    'pairwise',
    'dynamics',
    'information_flow',
    'topology',
    'statistics',
    # Backward-compatible functions
    'run',
    'run_from_manifest',
    'compute_state_vector',
    'compute_centroid',
    'compute_state_geometry',
    'compute_eigenvalues',
]
