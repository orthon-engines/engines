"""
PRISM Conditional Execution

Main pipeline module with stage definitions, conditional logic, and execution.

Output Parquet Mapping:
    Stage          Output            Key Columns
    1. Characterize  data.parquet      signal_type, is_stationary, distribution
    2. Vector        vector.parquet    hurst, entropy, spectral, n_steps, n_spikes
    3. Geometry      geometry.parquet  eff_dim, PC variances, cluster_id
    4. Dynamics      dynamics.parquet  hd_slope, tortuosity, n_regimes
    5. Physics       physics.parquet   T, V, H, L, p, G (with units/is_specific)
    6. Systems       systems.parquet   poles, zeros, bandwidth, stability
    7. Fields        fields.parquet    Re, TKE, epsilon, omega, E(k) slope

Conditional Execution:
    - Stages 1-5: Always run (core pipeline)
    - Stage 6 (Systems): Only if has_io_pairs AND has_events
    - Stage 7 (Fields): Only if has_velocity_field

Usage:
    from prism.conditional import run, run_all

    for result in run(config):
        print(result)

    # Or run all and get list
    results = run_all(config)
"""

import logging
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass

import polars as pl

from prism.db.parquet_store import (
    DATA, VECTOR, GEOMETRY, DYNAMICS, PHYSICS, SYSTEMS, FIELDS,
    get_path, ensure_directory,
)
from prism.utils.capability import detect_capabilities, DataSpec, Capability, SpatialType

logger = logging.getLogger(__name__)


# =============================================================================
# STAGE DEFINITIONS
# =============================================================================

@dataclass
class Stage:
    """Pipeline stage definition."""
    name: str
    output: str
    description: str
    key_columns: List[str]
    always_run: bool = True
    requires: Optional[List[str]] = None


STAGES: Dict[str, Stage] = {
    'data': Stage(
        name='data',
        output=DATA,
        description='Observations + numeric characterization',
        key_columns=['signal_type', 'is_stationary', 'distribution'],
        always_run=True,
    ),
    'vector': Stage(
        name='vector',
        output=VECTOR,
        description='Signal-level metrics (memory, frequency, volatility)',
        key_columns=['hurst', 'entropy', 'spectral', 'n_steps', 'n_spikes'],
        always_run=True,
    ),
    'geometry': Stage(
        name='geometry',
        output=GEOMETRY,
        description='Pairwise relationships (correlation, distance)',
        key_columns=['eff_dim', 'pc_variance_1', 'pc_variance_2', 'cluster_id'],
        always_run=True,
    ),
    'dynamics': Stage(
        name='dynamics',
        output=DYNAMICS,
        description='State/transition metrics (granger, dtw)',
        key_columns=['hd_slope', 'tortuosity', 'n_regimes'],
        always_run=True,
    ),
    'physics': Stage(
        name='physics',
        output=PHYSICS,
        description='Energy/momentum metrics (T, V, H, L, p, G)',
        key_columns=['kinetic_energy', 'potential_energy', 'hamiltonian',
                     'lagrangian', 'momentum', 'gibbs', 'is_specific', 'units'],
        always_run=False,
        requires=['labeled_signals_or_constants'],
    ),
    'systems': Stage(
        name='systems',
        output=SYSTEMS,
        description='Transfer functions (poles, zeros, bandwidth)',
        key_columns=['poles', 'zeros', 'bandwidth', 'gain', 'phase_margin', 'is_stable'],
        always_run=False,
        requires=['io_pairs', 'events'],
    ),
    'fields': Stage(
        name='fields',
        output=FIELDS,
        description='Navier-Stokes field analysis (vorticity, TKE)',
        key_columns=['reynolds_number', 'mean_tke', 'mean_dissipation',
                     'mean_vorticity', 'energy_spectrum_slope'],
        always_run=False,
        requires=['velocity_field'],
    ),
}


# =============================================================================
# STAGE RESULT
# =============================================================================

@dataclass
class StageResult:
    """Result from running a pipeline stage."""
    stage: str
    output_file: str
    n_rows: int
    n_cols: int
    success: bool
    error: Optional[str] = None


# =============================================================================
# CONDITIONAL CHECKS
# =============================================================================

def has_io_pairs(spec: DataSpec) -> bool:
    """Check if data has input/output signal pairs."""
    return spec.has_io_pair()


def has_events(vector_df: Optional[pl.DataFrame]) -> bool:
    """Check if data has detected step/spike events."""
    if vector_df is None:
        return False

    event_cols = ['n_steps', 'n_spikes', 'step_count', 'spike_count']
    for col in event_cols:
        if col in vector_df.columns:
            total_events = vector_df[col].sum()
            if total_events is not None and total_events > 0:
                return True
    return False


def has_velocity_field(spec: DataSpec) -> bool:
    """Check if data has 3D velocity field."""
    return spec.spatial.type == SpatialType.VELOCITY_FIELD


def has_labeled_signals_or_constants(spec: DataSpec) -> bool:
    """
    Check if we can compute any physics.

    Requires EITHER:
    - Labeled signals (velocity, position, temperature, etc.) for specific physics
    - Physical constants (mass, spring_constant, etc.) for real physics
    """
    # Check for labeled signals
    has_labels = spec.has_labeled_signals()

    # Check for any physics-relevant constants
    c = spec.constants
    has_constants = any([
        c.mass is not None,
        c.spring_constant is not None,
        c.moment_of_inertia is not None,
        c.Cp is not None,
        c.kinematic_viscosity is not None,
        c.density is not None,
    ])

    # Check for pipe network (enables pipe physics)
    has_pipes = spec.has_pipe_network()

    return has_labels or has_constants or has_pipes


def check_requirement(
    requirement: str,
    spec: DataSpec,
    vector_df: Optional[pl.DataFrame] = None,
) -> bool:
    """Check if a single requirement is met."""
    if requirement == 'io_pairs':
        return has_io_pairs(spec)
    elif requirement == 'events':
        return has_events(vector_df)
    elif requirement == 'velocity_field':
        return has_velocity_field(spec)
    elif requirement == 'labeled_signals_or_constants':
        return has_labeled_signals_or_constants(spec)
    else:
        logger.warning(f"Unknown requirement: {requirement}")
        return False


def should_run_stage(
    stage_name: str,
    spec: DataSpec,
    vector_df: Optional[pl.DataFrame] = None,
) -> bool:
    """Determine if a stage should run based on data availability."""
    stage = STAGES.get(stage_name)
    if stage is None:
        logger.warning(f"Unknown stage: {stage_name}")
        return False

    if stage.always_run:
        return True

    if stage.requires:
        for req in stage.requires:
            if not check_requirement(req, spec, vector_df):
                return False

    return True


# =============================================================================
# STAGE RUNNERS
# =============================================================================

def run_data(config: Dict, force: bool = False) -> StageResult:
    """Stage 1: Characterize observations."""
    output_path = get_path(DATA)

    if output_path.exists() and not force:
        df = pl.read_parquet(output_path)
        return StageResult('data', DATA, len(df), len(df.columns), True)

    try:
        from prism.engines.characterize import compute as characterize

        observations_path = get_path('observations')
        if not observations_path.exists():
            return StageResult('data', DATA, 0, 0, False, "observations.parquet not found")

        obs_df = pl.read_parquet(observations_path)
        results = []
        signal_cols = [c for c in obs_df.columns
                       if c not in ['entity_id', 'timestamp', 'window_id']]

        for signal_id in signal_cols:
            values = obs_df[signal_id].to_numpy()
            char_result = characterize(values)
            char_result['signal_id'] = signal_id
            results.append(char_result)

        df = pl.DataFrame(results)
        df.write_parquet(output_path)
        return StageResult('data', DATA, len(df), len(df.columns), True)

    except Exception as e:
        logger.error(f"Stage data failed: {e}")
        return StageResult('data', DATA, 0, 0, False, str(e))


def run_vector(config: Dict, force: bool = False) -> StageResult:
    """Stage 2: Compute signal-level metrics."""
    output_path = get_path(VECTOR)

    if output_path.exists() and not force:
        df = pl.read_parquet(output_path)
        return StageResult('vector', VECTOR, len(df), len(df.columns), True)

    try:
        logger.info("Running vector stage...")
        # Actual implementation calls vector engines
        return StageResult('vector', VECTOR, 0, 0, True)

    except Exception as e:
        logger.error(f"Stage vector failed: {e}")
        return StageResult('vector', VECTOR, 0, 0, False, str(e))


def run_geometry(config: Dict, force: bool = False) -> StageResult:
    """Stage 3: Compute pairwise relationships."""
    output_path = get_path(GEOMETRY)

    if output_path.exists() and not force:
        df = pl.read_parquet(output_path)
        return StageResult('geometry', GEOMETRY, len(df), len(df.columns), True)

    try:
        logger.info("Running geometry stage...")
        return StageResult('geometry', GEOMETRY, 0, 0, True)

    except Exception as e:
        logger.error(f"Stage geometry failed: {e}")
        return StageResult('geometry', GEOMETRY, 0, 0, False, str(e))


def run_dynamics(config: Dict, force: bool = False) -> StageResult:
    """Stage 4: Compute state/transition metrics."""
    output_path = get_path(DYNAMICS)

    if output_path.exists() and not force:
        df = pl.read_parquet(output_path)
        return StageResult('dynamics', DYNAMICS, len(df), len(df.columns), True)

    try:
        logger.info("Running dynamics stage...")
        return StageResult('dynamics', DYNAMICS, 0, 0, True)

    except Exception as e:
        logger.error(f"Stage dynamics failed: {e}")
        return StageResult('dynamics', DYNAMICS, 0, 0, False, str(e))


def run_physics(config: Dict, spec: DataSpec, force: bool = False) -> StageResult:
    """Stage 5: Compute energy/momentum metrics."""
    output_path = get_path(PHYSICS)

    if output_path.exists() and not force:
        df = pl.read_parquet(output_path)
        return StageResult('physics', PHYSICS, len(df), len(df.columns), True)

    try:
        logger.info("Running physics stage...")
        report = detect_capabilities(config)

        physics_caps = [
            Capability.KINETIC_ENERGY,
            Capability.POTENTIAL_ENERGY,
            Capability.HAMILTONIAN,
            Capability.LAGRANGIAN,
            Capability.MOMENTUM,
            Capability.GIBBS_FREE_ENERGY,
        ]

        available = [c for c in physics_caps if c in report.available]
        logger.info(f"Physics capabilities: {[c.name for c in available]}")

        # Actual implementation calls physics engines
        return StageResult('physics', PHYSICS, 0, 0, True)

    except Exception as e:
        logger.error(f"Stage physics failed: {e}")
        return StageResult('physics', PHYSICS, 0, 0, False, str(e))


def run_systems(config: Dict, spec: DataSpec, force: bool = False) -> StageResult:
    """Stage 6: Compute transfer functions (CONDITIONAL)."""
    output_path = get_path(SYSTEMS)

    if output_path.exists() and not force:
        df = pl.read_parquet(output_path)
        return StageResult('systems', SYSTEMS, len(df), len(df.columns), True)

    try:
        logger.info("Running systems stage (transfer functions)...")
        # Actual implementation calls transfer function engines
        return StageResult('systems', SYSTEMS, 0, 0, True)

    except Exception as e:
        logger.error(f"Stage systems failed: {e}")
        return StageResult('systems', SYSTEMS, 0, 0, False, str(e))


def run_fields(config: Dict, spec: DataSpec, force: bool = False) -> StageResult:
    """Stage 7: Navier-Stokes field analysis (CONDITIONAL)."""
    output_path = get_path(FIELDS)

    if output_path.exists() and not force:
        df = pl.read_parquet(output_path)
        return StageResult('fields', FIELDS, len(df), len(df.columns), True)

    try:
        logger.info("Running fields stage (Navier-Stokes)...")
        # Actual implementation calls Navier-Stokes engine
        return StageResult('fields', FIELDS, 0, 0, True)

    except Exception as e:
        logger.error(f"Stage fields failed: {e}")
        return StageResult('fields', FIELDS, 0, 0, False, str(e))


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run(
    config: Dict,
    force: bool = False,
    stages: Optional[List[str]] = None,
) -> Generator[StageResult, None, None]:
    """
    Run the PRISM calculation pipeline.

    Args:
        config: Configuration dictionary
        force: If True, recompute even if output exists
        stages: Optional list of specific stages to run

    Yields:
        StageResult for each completed stage
    """
    ensure_directory()
    spec = DataSpec.from_config(config)

    stages_to_run = stages or list(STAGES.keys())

    # Load vector.parquet if exists (for event detection)
    vector_path = get_path(VECTOR)
    vector_df = pl.read_parquet(vector_path) if vector_path.exists() else None

    # Core pipeline (stages 1-5: always run)
    if 'data' in stages_to_run:
        yield run_data(config, force)

    if 'vector' in stages_to_run:
        yield run_vector(config, force)
        if vector_path.exists():
            vector_df = pl.read_parquet(vector_path)

    if 'geometry' in stages_to_run:
        yield run_geometry(config, force)

    if 'dynamics' in stages_to_run:
        yield run_dynamics(config, force)

    if 'physics' in stages_to_run:
        yield run_physics(config, spec, force)

    # Conditional stages (6-7)
    if 'systems' in stages_to_run:
        if should_run_stage('systems', spec, vector_df):
            yield run_systems(config, spec, force)
        else:
            logger.info("Skipping systems stage (requires I/O pairs + events)")

    if 'fields' in stages_to_run:
        if should_run_stage('fields', spec, vector_df):
            yield run_fields(config, spec, force)
        else:
            logger.info("Skipping fields stage (requires velocity field)")


def run_all(config: Dict, force: bool = False) -> List[StageResult]:
    """Run all stages and return results."""
    return list(run(config, force))


# =============================================================================
# UTILITIES
# =============================================================================

def get_stages_to_run(
    spec: DataSpec,
    vector_df: Optional[pl.DataFrame] = None,
    requested: Optional[List[str]] = None,
) -> List[Stage]:
    """Get list of stages that should run."""
    stage_names = requested or list(STAGES.keys())
    return [STAGES[n] for n in stage_names if should_run_stage(n, spec, vector_df)]


def get_stage_summary(
    spec: DataSpec,
    vector_df: Optional[pl.DataFrame] = None
) -> Dict[str, Any]:
    """Get summary of which stages will run."""
    summary = {
        'always_run': [],
        'conditional_will_run': [],
        'conditional_will_skip': [],
    }

    for stage in STAGES.values():
        if stage.always_run:
            summary['always_run'].append(stage.name)
        elif should_run_stage(stage.name, spec, vector_df):
            summary['conditional_will_run'].append(stage.name)
        else:
            summary['conditional_will_skip'].append(stage.name)

    summary['stages_to_run'] = summary['always_run'] + summary['conditional_will_run']
    return summary


def print_stage_info():
    """Print stage information."""
    print("\nPRISM Pipeline Stages")
    print("=" * 80)
    print(f"{'Stage':<12} {'Output':<20} {'Condition':<20} Description")
    print("-" * 80)

    for stage in STAGES.values():
        cond = "always" if stage.always_run else ", ".join(stage.requires or [])
        print(f"{stage.name:<12} {stage.output + '.parquet':<20} {cond:<20} {stage.description}")

    print("=" * 80)


# =============================================================================
# CLI (when run directly)
# =============================================================================

if __name__ == "__main__":
    print_stage_info()
