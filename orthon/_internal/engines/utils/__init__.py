"""
PRISM Engine Utilities
======================

Shared infrastructure for PRISM engines and runners.

Modules:
    parallel: Orchestration utilities for parallel processing with scratch DBs
"""

from orthon._internal.engines.utils.parallel import (
    Orchestrator,
    WorkerAssignment,
    WorkerResult,
    divide_by_count,
    divide_by_date_range,
    divide_by_cohort,
    provision_scratch_db,
    merge_scratch_to_main,
    run_workers,
    parse_date,
    get_available_snapshots,
    get_signals,
)

__all__ = [
    "Orchestrator",
    "WorkerAssignment",
    "WorkerResult",
    "divide_by_count",
    "divide_by_date_range",
    "divide_by_cohort",
    "provision_scratch_db",
    "merge_scratch_to_main",
    "run_workers",
    "parse_date",
    "get_available_snapshots",
    "get_signals",
]
