"""
Reader — all parquet reads go through here.

No other module should call pl.read_parquet directly.
"""

import polars as pl
from pathlib import Path
from typing import Optional


# Output directory mapping (28 files -> 6 directories)
STAGE_DIRS = {
    # signal/ — per-signal features
    'signal_vector':              'signal',
    'signal_geometry':            'signal',
    'signal_stability':           'signal',

    # parameterization/ — derived rankings for Prime
    'signal_dominance':           'parameterization',

    # cohort/ — per-cohort geometry & relationships
    'cohort_geometry':            'cohort',
    'cohort_vector':              'cohort',
    'cohort_signal_positions':    'cohort',
    'cohort_feature_loadings':    'cohort',
    'cohort_pairwise':            'cohort',
    'cohort_information_flow':    'cohort',

    # cohort/cohort_dynamics/ — per-cohort dynamics
    'breaks':                     'cohort/cohort_dynamics',
    'geometry_dynamics':          'cohort/cohort_dynamics',
    'ftle':                       'cohort/cohort_dynamics',
    'lyapunov':                   'cohort/cohort_dynamics',
    'thermodynamics':             'cohort/cohort_dynamics',
    'ftle_field':                 'cohort/cohort_dynamics',
    'ftle_backward':              'cohort/cohort_dynamics',
    'velocity_field':             'cohort/cohort_dynamics',
    'ftle_rolling':               'cohort/cohort_dynamics',
    'ridge_proximity':            'cohort/cohort_dynamics',
    'persistent_homology':        'cohort/cohort_dynamics',

    # system/ — fleet-level geometry & relationships
    'system_geometry':            'system',
    'system_vector':              'system',
    'system_cohort_positions':    'system',
    'system_pairwise':            'system',
    'system_information_flow':    'system',
    'trajectory_signatures':      'system',
    'trajectory_library':         'system',
    'trajectory_match':           'system',

    # system/system_dynamics/ — fleet-level dynamics
    'system_ftle':                'system/system_dynamics',
    'system_velocity_field':      'system/system_dynamics',
}

# Internal names that map to different output filenames
# (avoids collision with cohort-level files in different directories)
STAGE_FILENAMES = {
    'system_ftle':           'ftle.parquet',
    'system_velocity_field': 'velocity_field.parquet',
}


def load_observations(data_path: str) -> pl.DataFrame:
    """Load observations.parquet from a data directory, sorted by signal_0."""
    p = Path(data_path)
    if p.is_file() and p.suffix == '.parquet':
        df = pl.read_parquet(str(p))
    elif (p / 'observations.parquet').exists():
        df = pl.read_parquet(str(p / 'observations.parquet'))
    else:
        raise FileNotFoundError(f"No observations.parquet in {data_path}")
    return df.sort('signal_0')


def load_output(data_path: str, name: str) -> Optional[pl.DataFrame]:
    """
    Load a stage output by name.

    Searches output/<subdir>/<filename>.parquet first,
    then output/<filename>.parquet (flat fallback).
    """
    output_dir = _get_output_dir(data_path)
    filename = STAGE_FILENAMES.get(name, f"{name}.parquet")

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
    filename = STAGE_FILENAMES.get(name, f"{name}.parquet")
    subdir = STAGE_DIRS.get(name, '')
    if subdir:
        d = output_dir / subdir
        d.mkdir(parents=True, exist_ok=True)
        return d / filename
    return output_dir / filename


def _get_output_dir(data_path: str) -> Path:
    """Resolve the output directory from a data path.

    data_path IS the output directory (e.g. output_time/).
    Stages write directly into subdirectories of data_path.
    """
    p = Path(data_path)
    if p.is_file():
        p = p.parent
    p.mkdir(parents=True, exist_ok=True)
    return p
