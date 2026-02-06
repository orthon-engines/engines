"""
PRISM Entry Points - Ordered Pipeline Stages
=============================================

Thin orchestrators that call engines for computation.
Entry points do NOT contain compute logic - only orchestration.

Pipeline Order:
    stage_01_signal_vector      → Per-signal metrics (combines 3a-3e)
    stage_02_state_vector       → Cross-signal centroid
    stage_02a_observations_windowed → Windowed observations
    stage_03_state_geometry     → Eigenstructure
    stage_03a_signal_statistics → Per-signal statistics (kurtosis, skewness)
    stage_03b_signal_temporal   → Per-signal temporal (trend, rate of change)
    stage_03c_signal_spectral   → Per-signal spectral (frequency, entropy)
    stage_03d_signal_complexity → Per-signal complexity (entropy measures)
    stage_03e_signal_stationarity → Per-signal stationarity (ADF, variance ratio)
    stage_04_cohorts            → Aggregated summaries
    stage_05_signal_geometry    → Per-signal to state relationships
    stage_05a_state_correlation → State vector component correlations
    stage_05b_signal_pairwise_detail → Detailed pairwise analysis
    stage_05c_state_aggregate   → Aggregated state statistics
    stage_06_signal_pairwise    → Pairwise signal relationships
    stage_07_geometry_dynamics  → Derivatives of geometry
    stage_08_lyapunov           → Per-signal Lyapunov exponents
    stage_08a_cohort_discovery  → Cluster signals into cohorts
    stage_08b_cohort_membership → Track cohort membership over time
    stage_08c_cohort_evolution  → Track cohort dynamics
    stage_09_dynamics           → Full dynamics (Lyapunov + attractor)
    stage_09a_cohort_thermodynamics → Thermodynamic analogs
    stage_10_information_flow   → Pairwise causality
    stage_11_topology           → Topological features
    stage_12_zscore             → Z-score normalization
    stage_13_statistics         → Summary statistics
    stage_14_correlation        → Correlation matrix

Each entry point:
1. Reads manifest/input data
2. Calls appropriate engines
3. Writes output parquet

PRISM computes numbers. ORTHON classifies.

Usage:
    python -m prism.entry_points.stage_01_signal_vector manifest.yaml
    python -m prism.entry_points.stage_02_state_vector signal_vector.parquet typology.parquet
    python -m prism.entry_points.stage_03a_signal_statistics observations.parquet
    ...
"""

# Backward compatibility - original unordered modules
from . import signal_vector

# For backward compatibility, export main functions
from .signal_vector import run, run_from_manifest

# Aliases for old names → new stages
from . import stage_02_state_vector as state_vector
from . import stage_03_state_geometry as state_geometry

# Stage 00: Break detection (runs before signal_vector)
from . import stage_00_breaks

# Core ordered stages (original 14)
from . import stage_01_signal_vector
from . import stage_02_state_vector
from . import stage_03_state_geometry
from . import stage_04_cohorts
from . import stage_05_signal_geometry
from . import stage_06_signal_pairwise
from . import stage_07_geometry_dynamics
from . import stage_08_lyapunov
from . import stage_09_dynamics
from . import stage_10_information_flow
from . import stage_11_topology
from . import stage_12_zscore
from . import stage_13_statistics
from . import stage_14_correlation

# Granular stages (substages)
from . import stage_02a_observations_windowed
from . import stage_03a_signal_statistics
from . import stage_03b_signal_temporal
from . import stage_03c_signal_spectral
from . import stage_03d_signal_complexity
from . import stage_03e_signal_stationarity
from . import stage_05a_state_correlation
from . import stage_05b_signal_pairwise_detail
from . import stage_05c_state_aggregate
from . import stage_08a_cohort_discovery
from . import stage_08b_cohort_membership
from . import stage_08c_cohort_evolution
from . import stage_09a_cohort_thermodynamics

# Stability pipeline (combines multiple stages)
from . import stability_pipeline
from .stability_pipeline import run_stability_pipeline

__all__ = [
    # Backward compatibility
    'run',
    'run_from_manifest',
    'signal_vector',
    'state_vector',
    'state_geometry',
    # Break detection (stage 00)
    'stage_00_breaks',
    # Core ordered stages
    'stage_01_signal_vector',
    'stage_02_state_vector',
    'stage_03_state_geometry',
    'stage_04_cohorts',
    'stage_05_signal_geometry',
    'stage_06_signal_pairwise',
    'stage_07_geometry_dynamics',
    'stage_08_lyapunov',
    'stage_09_dynamics',
    'stage_10_information_flow',
    'stage_11_topology',
    'stage_12_zscore',
    'stage_13_statistics',
    'stage_14_correlation',
    # Granular substages
    'stage_02a_observations_windowed',
    'stage_03a_signal_statistics',
    'stage_03b_signal_temporal',
    'stage_03c_signal_spectral',
    'stage_03d_signal_complexity',
    'stage_03e_signal_stationarity',
    'stage_05a_state_correlation',
    'stage_05b_signal_pairwise_detail',
    'stage_05c_state_aggregate',
    'stage_08a_cohort_discovery',
    'stage_08b_cohort_membership',
    'stage_08c_cohort_evolution',
    'stage_09a_cohort_thermodynamics',
    # Stability pipeline
    'stability_pipeline',
    'run_stability_pipeline',
]
