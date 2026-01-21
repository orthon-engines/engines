"""
PRISM Parallel Orchestration Utilities
======================================

Shared infrastructure for parallel processing across all runners.
Each runner (vectors.py, geometry.py, state.py) is an orchestrator that:
  1. Plans work division
  2. Creates worker assignments with temp parquet paths
  3. Dispatches workers (each writes to isolated temp file)
  4. Merges temp results to main parquet
  5. Cleans up

Uses Parquet for storage (no database locks).

Usage:
    from orthon._internal.engines.utils.parallel import ParquetOrchestrator

    class GeometryOrchestrator(ParquetOrchestrator):
        schema = 'geometry'
        table = 'signals'
        key_cols = ['cohort_id', 'window_end', 'window_days']

        def get_work_items(self):
            return list_of_items_to_process

        def worker_task(self, assignment):
            # Process items, return dict with 'processed', 'failed', 'temp_path'
            ...

    if __name__ == '__main__':
        GeometryOrchestrator.main()
"""

import argparse
import logging
import os
import tempfile
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import polars as pl

from orthon._internal.db.parquet_store import ensure_directory, get_path, OBSERVATIONS, SIGNALS, GEOMETRY, STATE, COHORTS
from orthon._internal.db.polars_io import read_parquet, upsert_parquet, write_parquet_atomic
from orthon._internal.db.scratch import TempParquet, merge_temp_results

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class WorkerAssignment:
    """Work package for a single worker."""

    worker_id: int
    temp_path: Path  # Path to worker's temp parquet file
    items: List[Any]  # dates, signals, cohorts - whatever the unit of work is
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerResult:
    """Result from a worker."""

    worker_id: int
    temp_path: Path
    status: str  # 'success', 'error', 'partial'
    items_processed: int
    items_failed: int
    error_message: Optional[str] = None
    duration_seconds: float = 0.0


# =============================================================================
# TEMP PATH MANAGEMENT
# =============================================================================


def generate_temp_path(worker_id: int, prefix: str = "worker") -> Path:
    """Generate unique temp parquet path for a worker."""
    unique = uuid.uuid4().hex[:8]
    return Path(tempfile.gettempdir()) / f"{prefix}_{worker_id}_{unique}.parquet"


def cleanup_temp_files(temp_paths: List[Path]) -> None:
    """Remove temporary parquet files."""
    for path in temp_paths:
        try:
            if path.exists():
                path.unlink()
                logger.debug(f"Deleted temp file: {path}")
        except Exception as e:
            logger.warning(f"Failed to delete temp file {path}: {e}")


# =============================================================================
# WORK DIVISION STRATEGIES
# =============================================================================


def divide_by_count(items: List[Any], n_workers: int) -> List[List[Any]]:
    """Divide items evenly across workers."""
    if n_workers <= 0:
        return [items]

    chunks = [[] for _ in range(n_workers)]
    for i, item in enumerate(items):
        chunks[i % n_workers].append(item)

    # Filter out empty chunks
    return [c for c in chunks if c]


def divide_by_date_range(
    start_date: date, end_date: date, n_workers: int
) -> List[Tuple[date, date]]:
    """Divide a date range into chunks for workers."""
    total_days = (end_date - start_date).days
    days_per_worker = max(1, total_days // n_workers)

    ranges = []
    current = start_date

    for i in range(n_workers):
        if current >= end_date:
            break

        chunk_end = min(current + timedelta(days=days_per_worker), end_date)

        # Last worker gets the remainder
        if i == n_workers - 1:
            chunk_end = end_date

        ranges.append((current, chunk_end))
        current = chunk_end

    return ranges


def get_cohorts_from_parquet() -> List[str]:
    """Get list of cohorts from cohorts.parquet."""
    cohorts_path = get_path(COHORTS)
    if not cohorts_path.exists():
        return []

    df = pl.read_parquet(cohorts_path)
    if "cohort_id" in df.columns:
        return df["cohort_id"].unique().to_list()
    return []


def get_signals_from_parquet(cohort: Optional[str] = None) -> List[str]:
    """Get signal IDs from parquet files."""
    obs_path = get_path(OBSERVATIONS)
    if not obs_path.exists():
        return []

    if cohort:
        # Filter by cohort membership from cohorts.parquet
        cohorts_path = get_path(COHORTS)
        if cohorts_path.exists():
            members = pl.read_parquet(cohorts_path)
            signal_ids = (
                members.filter(pl.col("cohort_id") == cohort)["signal_id"]
                .unique()
                .to_list()
            )
            return signal_ids

    # All signals
    df = pl.scan_parquet(obs_path).select("signal_id").unique().collect()
    return df["signal_id"].to_list()


# =============================================================================
# WORKER EXECUTION
# =============================================================================


def _worker_wrapper(args: Tuple[Callable, WorkerAssignment]) -> WorkerResult:
    """
    Wrapper for worker execution with error handling.
    Called by multiprocessing Pool.
    """
    task_fn, assignment = args
    start_time = datetime.now()

    try:
        result = task_fn(assignment)
        duration = (datetime.now() - start_time).total_seconds()

        return WorkerResult(
            worker_id=assignment.worker_id,
            temp_path=result.get("temp_path", assignment.temp_path),
            status="success",
            items_processed=result.get("processed", len(assignment.items)),
            items_failed=result.get("failed", 0),
            duration_seconds=duration,
        )

    except Exception as e:
        import traceback

        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Worker {assignment.worker_id} failed: {e}")
        logger.debug(traceback.format_exc())

        return WorkerResult(
            worker_id=assignment.worker_id,
            temp_path=assignment.temp_path,
            status="error",
            items_processed=0,
            items_failed=len(assignment.items),
            error_message=str(e),
            duration_seconds=duration,
        )


def run_workers(
    task_fn: Callable[[WorkerAssignment], Dict],
    assignments: List[WorkerAssignment],
    n_workers: int,
) -> List[WorkerResult]:
    """
    Execute workers in parallel.

    Args:
        task_fn: Function to execute for each assignment
        assignments: List of work assignments
        n_workers: Number of parallel workers

    Returns:
        List of worker results
    """
    if not assignments:
        return []

    # Prepare args for pool
    pool_args = [(task_fn, a) for a in assignments]

    if n_workers == 1:
        # Sequential execution for debugging
        results = [_worker_wrapper(args) for args in pool_args]
    else:
        with Pool(processes=n_workers) as pool:
            results = pool.map(_worker_wrapper, pool_args)

    return results


# =============================================================================
# PARQUET ORCHESTRATOR BASE CLASS
# =============================================================================


class ParquetOrchestrator(ABC):
    """
    Base class for all PRISM orchestrators using Parquet storage.

    Subclasses must define:
        - file: str (OBSERVATIONS, SIGNALS, GEOMETRY, STATE, or COHORTS)
        - key_cols: List[str] (columns for upsert deduplication)
        - get_work_items(): Get items to process
        - worker_task(): Process a single assignment

    Workers write to isolated temp parquet files, then results are
    merged to the main parquet file using upsert semantics.
    """

    file: str = None  # File constant (SIGNALS, GEOMETRY, etc.)
    key_cols: List[str] = []

    def __init__(
        self,
        workers: int = 1,
        config: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
    ):
        self.workers = min(workers, cpu_count())
        self.config = config or {}
        self.dry_run = dry_run
        self._temp_paths: List[Path] = []

        # Ensure data directories exist
        if not dry_run:
            ensure_directory()

    @abstractmethod
    def get_work_items(self) -> List[Any]:
        """
        Get the items to be processed (dates, signals, etc.).
        Must be implemented by subclass.
        """
        pass

    @abstractmethod
    def worker_task(self, assignment: WorkerAssignment) -> Dict[str, Any]:
        """
        Process a single worker assignment.
        Must be implemented by subclass.

        Args:
            assignment: WorkerAssignment with items to process

        Returns:
            Dict with:
                - 'processed': int (items successfully processed)
                - 'failed': int (items that failed)
                - 'temp_path': Path (path to temp parquet with results)
        """
        pass

    def create_assignments(self, items: List[Any]) -> List[WorkerAssignment]:
        """Create worker assignments from work items."""
        chunks = divide_by_count(items, self.workers)
        assignments = []

        for i, chunk in enumerate(chunks):
            temp_path = generate_temp_path(i, prefix=f"{self.schema}_{self.table}")
            self._temp_paths.append(temp_path)

            assignments.append(
                WorkerAssignment(
                    worker_id=i,
                    temp_path=temp_path,
                    items=chunk,
                    config=self.config,
                )
            )

        return assignments

    def merge_results(self, results: List[WorkerResult]) -> int:
        """Merge all temp parquet files into main parquet."""
        # Collect successful temp paths
        temp_paths = []
        for result in results:
            if result.status == "error":
                logger.warning(f"Skipping failed worker {result.worker_id}")
                continue
            if result.temp_path.exists():
                temp_paths.append(result.temp_path)

        if not temp_paths:
            logger.info("No temp files to merge")
            return 0

        # Merge to main parquet
        target_path = get_path(self.file)
        total_rows = merge_temp_results(
            temp_paths, target_path, key_cols=self.key_cols if self.key_cols else None
        )

        logger.info(f"Merged {total_rows:,} rows to {self.file}")
        return total_rows

    def cleanup(self):
        """Remove any remaining temp files."""
        cleanup_temp_files(self._temp_paths)
        self._temp_paths.clear()

    def run(self) -> Dict[str, Any]:
        """
        Execute the full orchestration pipeline.

        Returns:
            Summary statistics
        """
        start_time = datetime.now()

        # 1. Plan
        logger.info(f"Planning work for {self.__class__.__name__}...")
        items = self.get_work_items()

        if not items:
            logger.warning("No work items found")
            return {"status": "empty", "items": 0}

        logger.info(f"Found {len(items)} work items, {self.workers} workers")

        if self.dry_run:
            chunks = divide_by_count(items, self.workers)
            return {
                "status": "dry_run",
                "items": len(items),
                "workers": len(chunks),
                "items_per_worker": [len(c) for c in chunks],
            }

        # 2. Create Assignments
        logger.info("Creating worker assignments...")
        assignments = self.create_assignments(items)

        # 3. Dispatch
        logger.info(f"Dispatching {len(assignments)} workers...")
        results = run_workers(self.worker_task, assignments, self.workers)

        # 4. Report
        successful = sum(1 for r in results if r.status == "success")
        failed = sum(1 for r in results if r.status == "error")
        total_processed = sum(r.items_processed for r in results)
        total_failed = sum(r.items_failed for r in results)

        logger.info(f"Workers complete: {successful} success, {failed} failed")
        logger.info(f"Items: {total_processed} processed, {total_failed} failed")

        # 5. Merge
        logger.info("Merging results to main parquet...")
        total_rows = self.merge_results(results)

        # 6. Cleanup
        logger.info("Cleaning up temp files...")
        self.cleanup()

        duration = (datetime.now() - start_time).total_seconds()

        summary = {
            "status": "complete",
            "workers": self.workers,
            "workers_successful": successful,
            "workers_failed": failed,
            "items_total": len(items),
            "items_processed": total_processed,
            "items_failed": total_failed,
            "rows_merged": total_rows,
            "duration_seconds": duration,
        }

        logger.info(f"Complete in {duration:.1f}s: {total_rows:,} rows merged")

        return summary

    @classmethod
    def main(cls):
        """Standard CLI entry point for orchestrators."""
        parser = argparse.ArgumentParser(description=f"PRISM {cls.__name__}")
        parser.add_argument(
            "--workers", type=int, default=1, help="Number of parallel workers"
        )
        parser.add_argument("--dry-run", action="store_true", help="Plan but do not execute")
        parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

        # Allow subclasses to add arguments
        cls.add_arguments(parser)

        args = parser.parse_args()

        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Build config from args
        config = cls.build_config(args)

        orchestrator = cls(workers=args.workers, config=config, dry_run=args.dry_run)

        result = orchestrator.run()

        if args.dry_run:
            print("\n=== DRY RUN ===")
            print(f"Items: {result['items']}")
            print(f"Workers: {result['workers']}")
            print(f"Items per worker: {result['items_per_worker']}")
        else:
            print(f"\n=== COMPLETE ===")
            print(f"Duration: {result['duration_seconds']:.1f}s")
            print(f"Rows merged: {result['rows_merged']:,}")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Override to add orchestrator-specific arguments."""
        pass

    @classmethod
    def build_config(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Override to build config from parsed arguments."""
        return {}


# =============================================================================
# UTILITIES
# =============================================================================


def parse_date(s: str) -> date:
    """Parse date string in various formats."""
    for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"]:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {s}")


def get_available_snapshots(
    file: str = SIGNALS,
    start: Optional[date] = None,
    end: Optional[date] = None,
    date_col: str = "timestamp",
) -> List[date]:
    """Get available snapshot dates from a parquet file."""
    path = get_path(file)
    if not path.exists():
        return []

    lf = pl.scan_parquet(path)

    if date_col not in lf.schema:
        return []

    expr = pl.col(date_col).unique().sort()

    if start:
        expr = pl.col(date_col).filter(pl.col(date_col) >= start)
    if end:
        expr = pl.col(date_col).filter(pl.col(date_col) <= end)

    result = lf.select(pl.col(date_col).unique().sort()).collect()
    return result[date_col].to_list()


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

# Keep old names for backward compatibility but mark as deprecated


class Orchestrator(ParquetOrchestrator):
    """DEPRECATED: Use ParquetOrchestrator instead."""

    def __init__(self, db_path: str = None, **kwargs):
        import warnings

        warnings.warn(
            "Orchestrator is deprecated. Use ParquetOrchestrator instead. "
            "The db_path argument is ignored (data is stored in Parquet files).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)


# Legacy function compatibility
def provision_scratch_db(*args, **kwargs):
    """DEPRECATED: Scratch DBs no longer used."""
    import warnings

    warnings.warn(
        "provision_scratch_db is deprecated. Workers write directly to temp parquet files.",
        DeprecationWarning,
        stacklevel=2,
    )


def merge_scratch_to_main(*args, **kwargs):
    """DEPRECATED: Use merge_temp_results instead."""
    import warnings

    warnings.warn(
        "merge_scratch_to_main is deprecated. Use merge_temp_results from orthon._internal.db.scratch.",
        DeprecationWarning,
        stacklevel=2,
    )
    return 0


def generate_scratch_path(worker_id: int, prefix: str = "scratch") -> str:
    """DEPRECATED: Use generate_temp_path instead."""
    import warnings

    warnings.warn(
        "generate_scratch_path is deprecated. Use generate_temp_path instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return str(generate_temp_path(worker_id, prefix))
