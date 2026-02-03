"""
Pipeline Stage Prerequisites

Validates that required files exist before running a pipeline stage.
Enforces the stage order: observations -> typology -> manifest -> signal_vector -> ...

PRINCIPLE: "Check before compute, not after failure"

Usage:
    from prism.validation import check_prerequisites, PrerequisiteError

    try:
        check_prerequisites('signal_vector', data_dir='/path/to/data')
    except PrerequisiteError as e:
        print(f"Missing prerequisites: {e}")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any


class PrerequisiteError(Exception):
    """Raised when pipeline stage prerequisites are not met."""

    def __init__(
        self,
        stage: str,
        missing_files: List[str],
        message: Optional[str] = None,
    ):
        self.stage = stage
        self.missing_files = missing_files

        if message is None:
            message = (
                f"Cannot run '{stage}' stage. Missing required files:\n"
                + "\n".join(f"  - {f}" for f in missing_files)
                + "\n\nRun the prerequisite stages first (ORTHON generates these)."
            )

        super().__init__(message)


@dataclass
class StagePrerequisites:
    """
    Definition of prerequisites for a pipeline stage.

    Attributes:
        stage_name: Name of this stage
        required_files: List of files that must exist before this stage runs
        produces: List of files this stage produces
        depends_on: List of stages that must complete before this one
        description: Human-readable description of what this stage does
    """
    stage_name: str
    required_files: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    description: str = ""

    def check(self, data_dir: Path) -> List[str]:
        """
        Check if all prerequisites exist.

        Args:
            data_dir: Directory containing pipeline files

        Returns:
            List of missing files (empty if all present)
        """
        missing = []
        for filename in self.required_files:
            if not (data_dir / filename).exists():
                missing.append(filename)
        return missing

    def is_satisfied(self, data_dir: Path) -> bool:
        """Check if prerequisites are satisfied."""
        return len(self.check(data_dir)) == 0


# =============================================================================
# STAGE DEFINITIONS
# =============================================================================
# Pipeline order: observations -> typology -> manifest -> signal_vector -> ...
# ORTHON produces: observations.parquet, typology.parquet, manifest.yaml
# PRISM produces: signal_vector.parquet, state_vector.parquet, etc.
# =============================================================================

STAGE_PREREQUISITES: Dict[str, StagePrerequisites] = {
    'signal_vector': StagePrerequisites(
        stage_name='signal_vector',
        required_files=['observations.parquet', 'typology.parquet', 'manifest.yaml'],
        produces=['signal_vector.parquet'],
        depends_on=['typology'],
        description='Compute per-signal features per window',
    ),

    'state_vector': StagePrerequisites(
        stage_name='state_vector',
        required_files=['signal_vector.parquet', 'manifest.yaml'],
        produces=['state_vector.parquet', 'state_geometry.parquet', 'signal_geometry.parquet'],
        depends_on=['signal_vector'],
        description='Compute state centroids and geometry',
    ),

    'geometry_pairwise': StagePrerequisites(
        stage_name='geometry_pairwise',
        required_files=['signal_vector.parquet', 'manifest.yaml'],
        produces=['signal_pairwise.parquet'],
        depends_on=['signal_vector'],
        description='Compute pairwise signal relationships',
    ),

    'geometry_laplace': StagePrerequisites(
        stage_name='geometry_laplace',
        required_files=['state_vector.parquet', 'manifest.yaml'],
        produces=['geometry_dynamics.parquet', 'lyapunov.parquet', 'dynamics.parquet'],
        depends_on=['state_vector'],
        description='Compute dynamics and chaos measures',
    ),
}


def check_prerequisites(
    stage: str,
    data_dir: str,
    raise_on_missing: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Check that prerequisites are met for a pipeline stage.

    Args:
        stage: Name of the stage to check (e.g., 'signal_vector')
        data_dir: Directory containing pipeline files
        raise_on_missing: If True, raise PrerequisiteError on missing files
        verbose: If True, print status

    Returns:
        Dict with:
            - satisfied: bool
            - missing: List[str] of missing files
            - present: List[str] of present files

    Raises:
        PrerequisiteError: If prerequisites not met and raise_on_missing=True
        ValueError: If stage is not recognized
    """
    if stage not in STAGE_PREREQUISITES:
        raise ValueError(
            f"Unknown stage: '{stage}'. "
            f"Valid stages: {list(STAGE_PREREQUISITES.keys())}"
        )

    prereqs = STAGE_PREREQUISITES[stage]
    data_path = Path(data_dir)

    missing = prereqs.check(data_path)
    present = [f for f in prereqs.required_files if f not in missing]
    satisfied = len(missing) == 0

    if verbose:
        print(f"Stage: {stage}")
        print(f"  Description: {prereqs.description}")
        print(f"  Data directory: {data_path}")
        print(f"  Required files:")
        for f in prereqs.required_files:
            status = "OK" if f in present else "MISSING"
            print(f"    [{status}] {f}")
        print(f"  Prerequisites satisfied: {satisfied}")

    if not satisfied and raise_on_missing:
        raise PrerequisiteError(stage, missing)

    return {
        'satisfied': satisfied,
        'missing': missing,
        'present': present,
        'stage': stage,
        'description': prereqs.description,
    }


def check_all_stages(data_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Check prerequisites for all stages.

    Args:
        data_dir: Directory containing pipeline files

    Returns:
        Dict of {stage_name: check_result}
    """
    results = {}
    for stage in STAGE_PREREQUISITES:
        results[stage] = check_prerequisites(
            stage, data_dir, raise_on_missing=False
        )
    return results


def get_next_stage(data_dir: str) -> Optional[str]:
    """
    Determine the next stage that can be run.

    Args:
        data_dir: Directory containing pipeline files

    Returns:
        Name of next runnable stage, or None if all complete or stuck
    """
    data_path = Path(data_dir)

    for stage, prereqs in STAGE_PREREQUISITES.items():
        # Check if outputs already exist
        outputs_exist = all(
            (data_path / f).exists() for f in prereqs.produces
        )
        if outputs_exist:
            continue

        # Check if prerequisites are met
        if prereqs.is_satisfied(data_path):
            return stage

    return None


def print_pipeline_status(data_dir: str) -> None:
    """Print human-readable pipeline status."""
    data_path = Path(data_dir)

    print("=" * 60)
    print("PRISM PIPELINE STATUS")
    print(f"Data directory: {data_path}")
    print("=" * 60)

    for stage, prereqs in STAGE_PREREQUISITES.items():
        # Check prerequisites
        missing = prereqs.check(data_path)
        prereqs_ok = len(missing) == 0

        # Check outputs
        outputs_exist = all(
            (data_path / f).exists() for f in prereqs.produces
        )

        # Determine status
        if outputs_exist:
            status = "COMPLETE"
            symbol = "[OK]"
        elif prereqs_ok:
            status = "READY"
            symbol = "[->]"
        else:
            status = "BLOCKED"
            symbol = "[  ]"

        print(f"\n{symbol} {stage}: {status}")
        print(f"     {prereqs.description}")

        if missing:
            print(f"     Missing: {', '.join(missing)}")

    print("\n" + "=" * 60)

    next_stage = get_next_stage(data_dir)
    if next_stage:
        print(f"Next stage ready: {next_stage}")
    else:
        print("Pipeline complete or blocked.")
