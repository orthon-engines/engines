"""
PRISM Parquet Storage Layer
===========================

Core storage for PRISM diagnostics pipeline.

Directory Structure:
    data/
        observations.parquet        # Raw sensor data
        signal_typology.parquet     # ORTHON Layer 1: Signal classification
        manifold_geometry.parquet   # ORTHON Layer 2: Structural geometry
        dynamical_systems.parquet   # ORTHON Layer 3: System dynamics
        causal_mechanics.parquet    # ORTHON Layer 4: Physics-inspired analysis
        cohorts.parquet             # User-defined entity groupings

ORTHON Four-Layer Architecture:
    Each layer produces ONE parquet file with:
    - Identifiers: entity_id, unit_id, signal_id, window_idx, timestamp
    - Raw metrics: metric_* columns from engine computations
    - Classifications: categorical states (e.g., topology_class, dynamics_class)
    - Numeric states: normalized scores (0-1 or -1 to 1)

    Transitions are NOT stored separately - they are identified by comparing
    consecutive windows using thresholds from prism.config.thresholds.

Usage:
    from prism.db.parquet_store import get_path, OBSERVATIONS, SIGNAL_TYPOLOGY
    from prism.db.parquet_store import MANIFOLD_GEOMETRY, DYNAMICAL_SYSTEMS, CAUSAL_MECHANICS

    # Get path to a file
    obs_path = get_path(OBSERVATIONS)  # -> data/observations.parquet
    typology_path = get_path(SIGNAL_TYPOLOGY)  # -> data/signal_typology.parquet
"""

import os
from pathlib import Path
from typing import List, Optional

# =============================================================================
# CORE FILES
# =============================================================================

OBSERVATIONS = "observations"   # Raw sensor data

# =============================================================================
# ORTHON FOUR-LAYER ARCHITECTURE (one parquet per layer)
# =============================================================================

SIGNAL_TYPOLOGY = "signal_typology"         # Layer 1: 9-axis profile + classification
MANIFOLD_GEOMETRY = "manifold_geometry"     # Layer 2: Structural geometry + curvature
STRUCTURAL_GEOMETRY = "structural_geometry" # Layer 2 alias (backwards compat)
DYNAMICAL_SYSTEMS = "dynamical_systems"     # Layer 3: 6 dynamics metrics + classification
CAUSAL_MECHANICS = "causal_mechanics"       # Layer 4: 4 mechanics metrics + classification

# ORTHON deliverables - the four parquet files users receive
ORTHON_FILES = [SIGNAL_TYPOLOGY, MANIFOLD_GEOMETRY, DYNAMICAL_SYSTEMS, CAUSAL_MECHANICS]

# =============================================================================
# LEGACY FILES (kept for backwards compatibility)
# =============================================================================

VECTOR = "vector"               # Legacy: behavioral signals
SIGNALS = VECTOR                # Legacy alias
GEOMETRY = "geometry"           # Legacy: system structure
STATE = "state"                 # Legacy: dynamics
COHORTS = "cohorts"             # User-defined entity groupings

# Intermediate cohort files
COHORTS_RAW = "cohorts_raw"     # Cohorts discovered from raw observations
COHORTS_VECTOR = "cohorts_vector"  # Cohorts discovered from vector signals

# Signal States (unified state-based architecture)
SIGNAL_STATES = "signal_states"     # Unified signal states across all layers
COHORT_MEMBERS = "cohort_members"   # User-defined cohort memberships
CORPUS_CLASS = "corpus_class"       # Corpus-level classifications

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
FILES = [OBSERVATIONS] + ORTHON_FILES + [COHORTS]

# ML files
ML_FILES = [ML_FEATURES, ML_RESULTS, ML_IMPORTANCE, ML_MODEL]

# State architecture files
STATE_FILES = [SIGNAL_STATES, COHORT_MEMBERS, CORPUS_CLASS]

# Legacy files (still supported)
LEGACY_FILES = [VECTOR, GEOMETRY, STATE, STRUCTURAL_GEOMETRY]

# All valid file names
ALL_FILES = FILES + [COHORTS_RAW, COHORTS_VECTOR] + ML_FILES + STATE_FILES + LEGACY_FILES


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
