"""
PRISM Temporary Parquet Storage

Simple temp storage pattern for parallel workers:
1. Each worker writes to isolated temp parquet file
2. After processing, results merged to main storage
3. Auto-cleanup

Usage:
    from orthon._internal.db.scratch import TempParquet

    with TempParquet(prefix='vector_worker') as temp:
        # Compute results
        results_df = compute_metrics(...)
        temp.write(results_df)

    # Auto-deleted on exit

For parallel workers:
    def worker(assignment):
        with TempParquet(prefix=f'worker_{assignment.id}') as temp:
            results = process(assignment.items)
            temp.write(pl.DataFrame(results))
            return temp.path  # Return path for merge

    # After all workers complete
    merge_temp_results(temp_paths, target_path, key_cols)
"""

import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Union

import polars as pl

from orthon._internal.db.parquet_store import get_path, OBSERVATIONS, SIGNALS, GEOMETRY, STATE, COHORTS
from orthon._internal.db.polars_io import read_parquet, upsert_parquet, write_parquet_atomic

logger = logging.getLogger(__name__)


class TempParquet:
    """
    Temporary parquet file for parallel workers.

    Creates an isolated temp parquet file that is automatically
    cleaned up when the context manager exits.

    Usage:
        with TempParquet(prefix='worker_0') as temp:
            df = pl.DataFrame({'a': [1, 2, 3]})
            temp.write(df)
            # or append multiple times
            temp.append(more_data)

        # File auto-deleted

    Attributes:
        path: Path to temp parquet file
        df: Last written DataFrame (if any)
    """

    def __init__(
        self,
        prefix: str = "temp",
        keep_on_exit: bool = False,
    ):
        """
        Initialize temporary parquet storage.

        Args:
            prefix: Prefix for temp filename
            keep_on_exit: If True, don't delete file on close (for debugging)
        """
        self.keep_on_exit = keep_on_exit
        self._batches: List[pl.DataFrame] = []

        # Generate unique temp path
        unique_id = uuid.uuid4().hex[:8]
        self.path = Path(tempfile.gettempdir()) / f"{prefix}_{unique_id}.parquet"

        logger.debug(f"TempParquet created: {self.path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def write(self, df: pl.DataFrame) -> int:
        """
        Write DataFrame to temp parquet file.

        Overwrites any existing data.

        Args:
            df: Polars DataFrame to write

        Returns:
            Number of rows written
        """
        if len(df) == 0:
            return 0

        df.write_parquet(self.path)
        self._batches = [df]
        logger.debug(f"TempParquet wrote {len(df):,} rows to {self.path}")
        return len(df)

    def append(self, df: pl.DataFrame) -> int:
        """
        Append DataFrame to batch (writes on flush or close).

        For efficiency, data is accumulated in memory and written
        once when flush() is called or the context exits.

        Args:
            df: Polars DataFrame to append

        Returns:
            Number of rows appended
        """
        if len(df) == 0:
            return 0

        self._batches.append(df)
        return len(df)

    def flush(self) -> int:
        """
        Write all accumulated batches to temp file.

        Returns:
            Total number of rows written
        """
        if not self._batches:
            return 0

        combined = pl.concat(self._batches, how="diagonal_relaxed")
        combined.write_parquet(self.path)

        total = len(combined)
        logger.debug(f"TempParquet flushed {total:,} rows to {self.path}")
        return total

    def read(self) -> pl.DataFrame:
        """
        Read back data from temp parquet file.

        Returns:
            Polars DataFrame (empty if file doesn't exist)
        """
        if self.path.exists():
            return pl.read_parquet(self.path)
        elif self._batches:
            return pl.concat(self._batches, how="diagonal_relaxed")
        else:
            return pl.DataFrame()

    def row_count(self) -> int:
        """Get number of rows in temp file."""
        if self.path.exists():
            return pl.scan_parquet(self.path).select(pl.len()).collect().item()
        return sum(len(df) for df in self._batches)

    def exists(self) -> bool:
        """Check if temp file exists."""
        return self.path.exists()

    def close(self):
        """Flush batches and cleanup temp file."""
        # Flush any remaining batches
        if self._batches and not self.path.exists():
            self.flush()

        # Cleanup
        if not self.keep_on_exit and self.path.exists():
            self.path.unlink()
            logger.debug(f"TempParquet deleted: {self.path}")
        elif self.keep_on_exit:
            logger.debug(f"TempParquet kept: {self.path}")

        self._batches = []


def merge_temp_results(
    temp_paths: List[Path],
    target_path: Path,
    key_cols: Optional[List[str]] = None,
    delete_temps: bool = True,
) -> int:
    """
    Merge multiple temp parquet files into target.

    Used after parallel workers complete to combine results.

    Args:
        temp_paths: List of temp parquet file paths
        target_path: Target output file
        key_cols: If provided, upsert using these as key columns
        delete_temps: If True, delete temp files after merge

    Returns:
        Number of rows in merged result

    Example:
        >>> temp_paths = [Path('/tmp/worker_0.parquet'), Path('/tmp/worker_1.parquet')]
        >>> merge_temp_results(
        ...     temp_paths,
        ...     get_path(SIGNALS),
        ...     key_cols=['entity_id', 'signal_id', 'timestamp']
        ... )
    """
    # Read all temp files
    dfs = []
    for path in temp_paths:
        if path.exists():
            try:
                df = pl.read_parquet(path)
                if len(df) > 0:
                    dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to read temp file {path}: {e}")

    if not dfs:
        logger.info("No data to merge from temp files")
        return 0

    # Combine temp results
    combined = pl.concat(dfs, how="diagonal_relaxed")
    logger.info(f"Merged {len(combined):,} rows from {len(dfs)} temp files")

    # Write to target
    if key_cols:
        result = upsert_parquet(combined, target_path, key_cols)
    else:
        result = write_parquet_atomic(combined, target_path)

    # Cleanup temp files
    if delete_temps:
        for path in temp_paths:
            if path.exists():
                try:
                    path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {path}: {e}")

    return result


def merge_to_table(
    temp_paths: List[Path],
    file: str,
    key_cols: Optional[List[str]] = None,
    delete_temps: bool = True,
) -> int:
    """
    Merge temp files into a PRISM file.

    Convenience wrapper around merge_temp_results.

    Args:
        temp_paths: List of temp parquet file paths
        file: Target file (OBSERVATIONS, SIGNALS, GEOMETRY, STATE, COHORTS)
        key_cols: Key columns for upsert
        delete_temps: Delete temp files after merge

    Returns:
        Number of rows in merged result

    Example:
        >>> merge_to_table(
        ...     temp_paths,
        ...     SIGNALS,
        ...     key_cols=['entity_id', 'signal_id', 'timestamp']
        ... )
    """
    target_path = get_path(file)
    return merge_temp_results(temp_paths, target_path, key_cols, delete_temps)


class ParquetBatchWriter:
    """
    Efficient batch writer for accumulating results before write.

    Collects DataFrames in memory and writes in a single operation.
    More efficient than writing many small files.

    Usage:
        with ParquetBatchWriter() as writer:
            for signal in signals:
                metrics = compute(signal)
                writer.append(pl.DataFrame(metrics))

            # Write all at once
            writer.write_to(get_path(SIGNALS),
                           key_cols=['entity_id', 'signal_id', 'timestamp'])
    """

    def __init__(self, max_batch_size: int = 100_000):
        """
        Initialize batch writer.

        Args:
            max_batch_size: Max rows to hold in memory before auto-flush
        """
        self._batches: List[pl.DataFrame] = []
        self._row_count = 0
        self.max_batch_size = max_batch_size

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._batches = []
        return False

    def append(self, df: pl.DataFrame) -> int:
        """
        Append DataFrame to batch.

        Args:
            df: Polars DataFrame

        Returns:
            Current total row count
        """
        if len(df) == 0:
            return self._row_count

        self._batches.append(df)
        self._row_count += len(df)
        return self._row_count

    def get_combined(self) -> pl.DataFrame:
        """Get all batches combined into single DataFrame."""
        if not self._batches:
            return pl.DataFrame()
        return pl.concat(self._batches, how="diagonal_relaxed")

    def write_to(
        self,
        path: Union[str, Path],
        key_cols: Optional[List[str]] = None,
        mode: str = "upsert",
    ) -> int:
        """
        Write accumulated data to parquet file.

        Args:
            path: Target parquet path
            key_cols: Key columns for upsert mode
            mode: 'replace', 'append', or 'upsert'

        Returns:
            Number of rows written
        """
        if not self._batches:
            return 0

        combined = self.get_combined()
        path = Path(path)

        if mode == "replace":
            return write_parquet_atomic(combined, path)
        elif mode == "append":
            existing = read_parquet(path)
            if len(existing) > 0:
                combined = pl.concat([existing, combined], how="diagonal_relaxed")
            return write_parquet_atomic(combined, path)
        elif mode == "upsert":
            if key_cols is None:
                raise ValueError("key_cols required for upsert mode")
            return upsert_parquet(combined, path, key_cols)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def write_to_file(
        self,
        file: str,
        key_cols: Optional[List[str]] = None,
        mode: str = "upsert",
    ) -> int:
        """
        Write to PRISM file.

        Args:
            file: Target file (OBSERVATIONS, SIGNALS, GEOMETRY, STATE, COHORTS)
            key_cols: Key columns for upsert
            mode: 'replace', 'append', or 'upsert'

        Returns:
            Number of rows written
        """
        path = get_path(file)
        return self.write_to(path, key_cols, mode)

    @property
    def row_count(self) -> int:
        """Total rows accumulated."""
        return self._row_count

    def clear(self):
        """Clear accumulated batches."""
        self._batches = []
        self._row_count = 0


# Legacy compatibility - ScratchDB is deprecated
class ScratchDB:
    """
    DEPRECATED: Use TempParquet instead.

    This class exists for backward compatibility only.
    """

    def __init__(self, *args, **kwargs):
        import warnings

        warnings.warn(
            "ScratchDB is deprecated. Use TempParquet for temp storage, "
            "or use polars_io functions directly for reading/writing.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise NotImplementedError(
            "ScratchDB has been removed. Use TempParquet instead:\n\n"
            "  from orthon._internal.db.scratch import TempParquet\n\n"
            "  with TempParquet(prefix='worker') as temp:\n"
            "      temp.write(results_df)\n"
        )
