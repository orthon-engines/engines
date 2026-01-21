"""
PRISM Parquet Storage Layer
===========================

5 files. No more, no less.

Directory Structure:
    data/
        observations.parquet        # Raw sensor data
        vector.parquet              # All behavioral signals (dense + sparse)
        geometry.parquet            # System structure at each timestamp
        state.parquet               # Dynamics at each timestamp
        cohorts.parquet             # Discovered entity groupings

Entity Hierarchy:
    entity (engine_47)              # Fails, gets RUL, joins cohort
    └── signal (sensor_1)           # Measures entity
        └── derived (inst_freq)     # Computed from signal

File Schemas:
    observations: entity_id, signal_id, timestamp, value
    vector:       entity_id, signal_id, source_signal, engine, signal_type, timestamp, value, mode_id
    geometry:     entity_id, timestamp, divergence, mode_count, coupling_mean, transition_flag, regime, ...
    state:        entity_id, timestamp, position_*, velocity_*, acceleration_*, failure_signature, ...
    cohorts:      entity_id, cohort_id, trajectory_similarity, failure_mode, ...

Usage:
    from orthon._internal.db.parquet_store import get_path, OBSERVATIONS, VECTOR, GEOMETRY, STATE, COHORTS

    # Get path to a file
    obs_path = get_path(OBSERVATIONS)  # -> data/observations.parquet
"""

import os
from pathlib import Path
from typing import List, Optional

# =============================================================================
# THE 5 FILES
# =============================================================================

OBSERVATIONS = "observations"   # Raw sensor data
VECTOR = "vector"               # All behavioral signals (was SIGNALS)
SIGNALS = VECTOR                # Backwards compatibility alias
GEOMETRY = "geometry"           # System structure at each t
STATE = "state"                 # Dynamics at each t
COHORTS = "cohorts"             # Discovered entity groupings

# Intermediate cohort files
COHORTS_RAW = "cohorts_raw"     # Cohorts discovered from raw observations
COHORTS_VECTOR = "cohorts_vector"  # Cohorts discovered from vector signals

# ML Accelerator files
ML_FEATURES = "ml_features"     # Denormalized feature table for ML
ML_RESULTS = "ml_results"       # Model predictions vs actuals
ML_IMPORTANCE = "ml_importance" # Feature importance rankings
ML_MODEL = "ml_model"           # Serialized model (actually .pkl)

# All valid file names
FILES = [OBSERVATIONS, VECTOR, GEOMETRY, STATE, COHORTS]
ML_FILES = [ML_FEATURES, ML_RESULTS, ML_IMPORTANCE, ML_MODEL]
ALL_FILES = FILES + [COHORTS_RAW, COHORTS_VECTOR] + ML_FILES


# =============================================================================
# PATH FUNCTIONS
# =============================================================================

def get_data_root() -> Path:
    """
    Return the root data directory.

    Returns:
        Path to data directory (e.g., data/)
    """
    env_path = os.environ.get("PRISM_DATA_PATH")
    if env_path:
        return Path(env_path)
    return Path(os.path.expanduser("~/prism-mac/data"))


def get_path(file: str) -> Path:
    """
    Return the path to a PRISM output file.

    Args:
        file: File name (OBSERVATIONS, VECTOR, GEOMETRY, STATE, COHORTS)

    Returns:
        Path to parquet file

    Examples:
        >>> get_path(OBSERVATIONS)
        PosixPath('.../data/observations.parquet')

        >>> get_path(VECTOR)
        PosixPath('.../data/vector.parquet')
    """
    if file not in ALL_FILES:
        raise ValueError(f"Unknown file: {file}. Valid files: {ALL_FILES}")

    return get_data_root() / f"{file}.parquet"


def ensure_directory() -> Path:
    """
    Create data directory if it doesn't exist.

    Returns:
        Path to data directory
    """
    root = get_data_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def file_exists(file: str) -> bool:
    """Check if a PRISM output file exists."""
    return get_path(file).exists()


def get_file_size(file: str) -> Optional[int]:
    """Get file size in bytes, or None if doesn't exist."""
    path = get_path(file)
    if path.exists():
        return path.stat().st_size
    return None


def delete_file(file: str) -> bool:
    """Delete a file. Returns True if deleted, False if didn't exist."""
    path = get_path(file)
    if path.exists():
        path.unlink()
        return True
    return False


def list_files() -> List[str]:
    """List all existing PRISM output files."""
    return [f for f in ALL_FILES if file_exists(f)]


def get_status() -> dict:
    """
    Get status of all PRISM output files.

    Returns:
        Dict with file status and sizes
    """
    status = {}
    for f in ALL_FILES:
        path = get_path(f)
        if path.exists():
            size = path.stat().st_size
            status[f] = {"exists": True, "size_bytes": size, "size_mb": size / 1024 / 1024}
        else:
            status[f] = {"exists": False, "size_bytes": 0, "size_mb": 0}
    return status


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PRISM Storage - 5 Files")
    parser.add_argument("--init", action="store_true", help="Create data directory")
    parser.add_argument("--list", action="store_true", help="List files")
    parser.add_argument("--status", action="store_true", help="Show file status")

    args = parser.parse_args()

    if args.init:
        path = ensure_directory()
        print(f"Created: {path}")
        print("\nExpected files:")
        for f in FILES:
            print(f"  {f}.parquet")

    elif args.list:
        files = list_files()
        if files:
            print("Files:")
            for f in files:
                size = get_file_size(f)
                print(f"  {f}.parquet ({size:,} bytes)")
        else:
            print("No files found")

    elif args.status:
        status = get_status()
        print("Status:")
        print("-" * 50)
        for f, info in status.items():
            if info["exists"]:
                print(f"  ✓ {f}.parquet ({info['size_mb']:.2f} MB)")
            else:
                print(f"  ✗ {f}.parquet (missing)")

    else:
        parser.print_help()
