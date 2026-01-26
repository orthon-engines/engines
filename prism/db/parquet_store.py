"""
PRISM Parquet Storage Layer
===========================

Core storage for PRISM diagnostics pipeline.

Directory Structure:
    data/
        observations.parquet  # Raw sensor data
        data.parquet         # Observations + characterization
        vector.parquet       # Signal-level metrics
        geometry.parquet     # Pairwise relationships
        dynamics.parquet     # State/transition metrics
        fields.parquet       # Navier-Stokes field analysis (3D velocity fields)
        cohorts.parquet      # User-defined entity groupings

PRISM Seven-File Architecture:
    Pure numerical output - no labels, no classification.

    1. data.parquet     - observations + numeric characterization
    2. vector.parquet   - signal-level metrics (memory, frequency, volatility)
    3. geometry.parquet - pairwise relationships (correlation, distance)
    4. dynamics.parquet - state/transition metrics (granger, dtw)
    5. physics.parquet  - energy/momentum metrics (T, V, H, L, p, G with units)
    6. systems.parquet  - transfer functions (poles, zeros, bandwidth, stability)
    7. fields.parquet   - Navier-Stokes field analysis (vorticity, TKE, dissipation)

Usage:
    from prism.db.parquet_store import get_path, OBSERVATIONS, DATA, VECTOR
    from prism.db.parquet_store import GEOMETRY, DYNAMICS, FIELDS

    # Get path to a file
    obs_path = get_path(OBSERVATIONS)  # -> data/observations.parquet
    vector_path = get_path(VECTOR)     # -> data/vector.parquet
"""

import os
from pathlib import Path
from typing import List, Optional

# =============================================================================
# CORE FILES
# =============================================================================

OBSERVATIONS = "observations"   # Raw sensor data

# =============================================================================
# PRISM PURE CALCULATION OUTPUT (7 files)
# =============================================================================

DATA = "data"           # Observations + numeric characterization
VECTOR = "vector"       # Signal-level metrics (memory, frequency, volatility)
GEOMETRY = "geometry"   # Pairwise relationships (correlation, distance)
DYNAMICS = "dynamics"   # State/transition metrics (granger, dtw)
PHYSICS = "physics"     # Energy/momentum metrics (T, V, H, L, p, G with units)
SYSTEMS = "systems"     # Transfer functions (poles, zeros, bandwidth, stability)
FIELDS = "fields"       # Navier-Stokes field analysis (vorticity, TKE, dissipation)

# PRISM deliverables - the seven parquet files users receive
PRISM_FILES = [DATA, VECTOR, GEOMETRY, DYNAMICS, PHYSICS, SYSTEMS, FIELDS]

COHORTS = "cohorts"             # User-defined entity groupings

# =============================================================================
# ML ACCELERATOR FILES
# =============================================================================

ML_FEATURES = "ml_features"     # Denormalized feature table for ML
ML_RESULTS = "ml_results"       # Model predictions vs actuals
ML_IMPORTANCE = "ml_importance" # Feature importance rankings
ML_MODEL = "ml_model"           # Serialized model (actually .pkl)

# =============================================================================
# FILE LISTS
# =============================================================================

# Core pipeline files
FILES = [OBSERVATIONS] + PRISM_FILES + [COHORTS]

# ML files
ML_FILES = [ML_FEATURES, ML_RESULTS, ML_IMPORTANCE, ML_MODEL]

# All valid file names
ALL_FILES = FILES + ML_FILES


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
