"""
Reader — all parquet reads go through here.

No other module should call pl.read_parquet directly.
"""

import polars as pl
from pathlib import Path
from typing import Optional


# Output directory mapping (29 files -> 6 directories)
STAGE_DIRS = {
    'signal_vector':        '1_signal_features',
    'signal_geometry':      '1_signal_features',
    'signal_stability':     '1_signal_features',

    'state_vector':         '2_system_state',
    'state_geometry':       '2_system_state',
    'geometry_dynamics':    '2_system_state',
    'sensor_eigendecomp':   '2_system_state',

    'breaks':               '3_health_scoring',
    'cohort_baseline':      '3_health_scoring',
    'observation_geometry': '3_health_scoring',

    'signal_pairwise':      '4_signal_relationships',
    'information_flow':     '4_signal_relationships',
    'segment_comparison':   '4_signal_relationships',
    'info_flow_delta':      '4_signal_relationships',

    'ftle':                 '5_evolution',
    'lyapunov':             '5_evolution',
    'cohort_thermodynamics': '5_evolution',
    'ftle_field':           '5_evolution',
    'ftle_backward':        '5_evolution',
    'velocity_field':       '5_evolution',
    'ftle_rolling':         '5_evolution',
    'ridge_proximity':          '5_evolution',
    'persistent_homology':      '5_evolution',

    'cohort_vector':                 '6_fleet',
    'system_geometry':           '6_fleet',
    'cohort_pairwise':           '6_fleet',
    'cohort_information_flow':   '6_fleet',
    'cohort_ftle':               '6_fleet',
    'cohort_velocity_field':     '6_fleet',
}


def load_observations(data_path: str) -> pl.DataFrame:
    """Load observations.parquet from a data directory."""
    p = Path(data_path)
    if p.is_file() and p.suffix == '.parquet':
        return pl.read_parquet(str(p))
    obs_path = p / 'observations.parquet'
    if obs_path.exists():
        return pl.read_parquet(str(obs_path))
    raise FileNotFoundError(f"No observations.parquet in {data_path}")


def load_output(data_path: str, name: str) -> Optional[pl.DataFrame]:
    """
    Load a stage output by name.

    Searches output/<subdir>/<name>.parquet first,
    then output/<name>.parquet (flat fallback).
    """
    output_dir = _get_output_dir(data_path)
    filename = f"{name}.parquet"

    # Try subdirectory first
    subdir = STAGE_DIRS.get(name, '')
    if subdir:
        path = output_dir / subdir / filename
        if path.exists():
            return pl.read_parquet(str(path))

    # Flat fallback
    path = output_dir / filename
    if path.exists():
        return pl.read_parquet(str(path))

    return None


def output_path(data_path: str, name: str) -> Path:
    """Get the output path for a stage output by name."""
    output_dir = _get_output_dir(data_path)
    filename = f"{name}.parquet"
    subdir = STAGE_DIRS.get(name, '')
    if subdir:
        d = output_dir / subdir
        d.mkdir(parents=True, exist_ok=True)
        return d / filename
    return output_dir / filename


def _get_output_dir(data_path: str) -> Path:
    """Resolve the output directory from a data path."""
    p = Path(data_path)
    if p.is_file():
        p = p.parent
    output_dir = p / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
