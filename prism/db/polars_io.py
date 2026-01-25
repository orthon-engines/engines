"""
PRISM Polars I/O Utilities

Atomic writes, upsert operations, and safe I/O for Parquet files.
Ensures parallel-safe operations without database locks.

Key Functions:
    read_parquet(path) - Read parquet, return empty DataFrame if missing
    write_parquet_atomic(df, path) - Write to temp file, rename (atomic)
    upsert_parquet(df, path, key_cols) - INSERT OR REPLACE semantics
    append_parquet(df, path) - Simple append to existing file
"""

import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Union

import polars as pl

from prism.db.parquet_store import get_path, OBSERVATIONS, VECTOR, GEOMETRY, DYNAMICS, PHYSICS, COHORTS
# Legacy aliases
SIGNALS = VECTOR
STATE = DYNAMICS


def read_parquet(
    path: Union[str, Path],
    columns: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Read a parquet file, returning empty DataFrame if file doesn't exist.

    Args:
        path: Path to parquet file
        columns: Optional list of columns to read (projection pushdown)

    Returns:
        Polars DataFrame (empty if file doesn't exist)

    Example:
        >>> df = read_parquet(get_path(OBSERVATIONS))
    """
    path = Path(path)
    if not path.exists():
        return pl.DataFrame()

    return pl.read_parquet(path, columns=columns)


# Size threshold for lazy processing (500 MB)
LAZY_THRESHOLD_MB = 500


def get_file_size_mb(path: Union[str, Path]) -> Optional[float]:
    """Get file size in MB, None if doesn't exist."""
    path = Path(path)
    if not path.exists():
        return None
    return path.stat().st_size / 1e6


def read_parquet_smart(
    path: Union[str, Path],
    columns: Optional[List[str]] = None,
    threshold_mb: float = LAZY_THRESHOLD_MB,
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Smart parquet reader: uses lazy loading for large files.

    For files > threshold_mb, uses scan_parquet + collect for memory efficiency.
    For smaller files, uses direct read_parquet.

    Args:
        path: Path to parquet file
        columns: Optional columns to select (projection pushdown)
        threshold_mb: Size threshold for lazy mode (default 500 MB)
        verbose: Print which mode is being used

    Returns:
        Polars DataFrame (empty if file doesn't exist)

    Example:
        >>> df = read_parquet_smart('data/vector/signal_field.parquet')
        # Automatically uses lazy mode if file is > 500 MB
    """
    path = Path(path)
    if not path.exists():
        return pl.DataFrame()

    size_mb = get_file_size_mb(path)

    if size_mb > threshold_mb:
        if verbose:
            print(f"  [LAZY] {path.name}: {size_mb:.0f} MB > {threshold_mb:.0f} MB threshold")
        lf = pl.scan_parquet(path)
        if columns:
            lf = lf.select(columns)
        return lf.collect()
    else:
        if verbose:
            print(f"  [EAGER] {path.name}: {size_mb:.0f} MB")
        return pl.read_parquet(path, columns=columns)


def iter_parquet_windows(
    path: Union[str, Path],
    window_col: str,
    columns: Optional[List[str]] = None,
    verbose: bool = False,
):
    """
    Iterate over a parquet file window-by-window (memory efficient).

    Yields one window at a time using lazy filtering.

    Args:
        path: Path to parquet file
        window_col: Column to partition by (e.g., 'window_end', 'obs_date')
        columns: Optional columns to select
        verbose: Print progress

    Yields:
        (window_value, DataFrame) tuples

    Example:
        >>> for window_end, df in iter_parquet_windows(
        ...     'data/vector/signal_field.parquet',
        ...     window_col='window_end'
        ... ):
        ...     process(df)
        ...     # df is automatically released after each iteration
    """
    import gc

    path = Path(path)
    if not path.exists():
        return

    # Get unique windows (metadata query)
    windows = (
        pl.scan_parquet(path)
        .select(window_col)
        .unique()
        .sort(window_col)
        .collect()[window_col]
        .to_list()
    )

    if verbose:
        print(f"  Streaming {len(windows)} windows from {path.name}")

    for i, window in enumerate(windows):
        # Load only this window
        lf = pl.scan_parquet(path).filter(pl.col(window_col) == window)
        if columns:
            lf = lf.select(columns)
        df = lf.collect()

        yield window, df

        # Release memory
        del df
        gc.collect()

        if verbose and (i + 1) % 50 == 0:
            print(f"    Processed {i + 1}/{len(windows)} windows...")


def iter_parquet_batches(
    path: Union[str, Path],
    batch_size: int = 100000,
    columns: Optional[List[str]] = None,
):
    """
    Iterate over a parquet file in row batches (memory efficient).

    Uses Polars streaming to read in chunks.

    Args:
        path: Path to parquet file
        batch_size: Number of rows per batch
        columns: Optional columns to select

    Yields:
        DataFrame batches

    Example:
        >>> for batch in iter_parquet_batches(
        ...     'data/raw/observations.parquet',
        ...     batch_size=50000
        ... ):
        ...     process(batch)
    """
    import gc

    path = Path(path)
    if not path.exists():
        return

    lf = pl.scan_parquet(path)
    if columns:
        lf = lf.select(columns)

    # Use streaming collect with batch processing
    total_rows = lf.select(pl.len()).collect().item()
    offset = 0

    while offset < total_rows:
        batch = (
            pl.scan_parquet(path)
            .slice(offset, batch_size)
        )
        if columns:
            batch = batch.select(columns)
        df = batch.collect()

        if len(df) == 0:
            break

        yield df

        offset += len(df)
        del df
        gc.collect()


def scan_parquet(
    path: Union[str, Path],
) -> Optional[pl.LazyFrame]:
    """
    Create a lazy scan of a parquet file.

    Args:
        path: Path to parquet file

    Returns:
        Polars LazyFrame, or None if file doesn't exist

    Example:
        >>> lf = scan_parquet('data/raw/observations.parquet')
        >>> if lf is not None:
        ...     result = lf.filter(pl.col('signal_id') == 'SENSOR_01').collect()
    """
    path = Path(path)
    if not path.exists():
        return None

    return pl.scan_parquet(path)


def write_parquet_atomic(
    df: pl.DataFrame,
    path: Union[str, Path],
    compression: str = "zstd",
) -> int:
    """
    Atomically write a DataFrame to a parquet file.

    Writes to a temporary file first, then renames to target path.
    This ensures the target file is never in a partial/corrupt state.

    Args:
        df: Polars DataFrame to write
        path: Target path for parquet file
        compression: Compression algorithm (zstd, snappy, lz4, etc.)

    Returns:
        Number of rows written

    Example:
        >>> write_parquet_atomic(df, 'data/raw/observations.parquet')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (for atomic rename)
    temp_path = path.with_suffix(".parquet.tmp")

    try:
        df.write_parquet(temp_path, compression=compression)
        temp_path.rename(path)
        return len(df)
    except Exception:
        # Clean up temp file on failure
        if temp_path.exists():
            temp_path.unlink()
        raise


def append_parquet(
    df: pl.DataFrame,
    path: Union[str, Path],
    compression: str = "zstd",
) -> int:
    """
    Append a DataFrame to an existing parquet file.

    Reads existing data, concatenates new data, writes atomically.

    Args:
        df: Polars DataFrame to append
        path: Path to parquet file
        compression: Compression algorithm

    Returns:
        Total number of rows after append

    Example:
        >>> append_parquet(new_observations, 'data/raw/observations.parquet')
    """
    path = Path(path)

    if path.exists():
        existing = pl.read_parquet(path)
        combined = pl.concat([existing, df], how="diagonal_relaxed")
    else:
        combined = df

    return write_parquet_atomic(combined, path, compression)


def upsert_parquet(
    df: pl.DataFrame,
    path: Union[str, Path],
    key_cols: List[str],
    compression: str = "zstd",
) -> int:
    """
    Upsert (INSERT OR REPLACE) a DataFrame to a parquet file.

    For rows with matching key columns, new data replaces old.
    For rows with new keys, data is appended.

    Args:
        df: Polars DataFrame with new/updated data
        path: Path to parquet file
        key_cols: Columns that form the unique key
        compression: Compression algorithm

    Returns:
        Total number of rows after upsert

    Example:
        >>> upsert_parquet(
        ...     metrics_df,
        ...     'data/vector/signals.parquet',
        ...     key_cols=['signal_id', 'obs_date', 'engine', 'metric_name']
        ... )
    """
    path = Path(path)

    if not path.exists():
        # No existing data, just write
        return write_parquet_atomic(df, path, compression)

    existing = pl.read_parquet(path)

    if len(existing) == 0:
        return write_parquet_atomic(df, path, compression)

    # Anti-join: keep existing rows that don't match new keys
    kept = existing.join(df.select(key_cols), on=key_cols, how="anti")

    # Combine kept rows with new data
    combined = pl.concat([kept, df], how="diagonal_relaxed")

    return write_parquet_atomic(combined, path, compression)


def delete_rows(
    path: Union[str, Path],
    filter_expr: pl.Expr,
    compression: str = "zstd",
) -> int:
    """
    Delete rows matching a filter expression.

    Args:
        path: Path to parquet file
        filter_expr: Polars expression for rows to DELETE
        compression: Compression algorithm

    Returns:
        Number of rows remaining after deletion

    Example:
        >>> delete_rows(
        ...     'data/raw/observations.parquet',
        ...     pl.col('signal_id') == 'OLD_INDICATOR'
        ... )
    """
    path = Path(path)

    if not path.exists():
        return 0

    df = pl.read_parquet(path)
    kept = df.filter(~filter_expr)

    return write_parquet_atomic(kept, path, compression)


def merge_parquet_files(
    input_paths: List[Path],
    output_path: Path,
    key_cols: Optional[List[str]] = None,
    delete_inputs: bool = True,
    compression: str = "zstd",
) -> int:
    """
    Merge multiple parquet files into one.

    Used for combining parallel worker outputs.

    Args:
        input_paths: List of parquet files to merge
        output_path: Target output file
        key_cols: If provided, deduplicate on these columns
        delete_inputs: If True, delete input files after merge
        compression: Compression algorithm

    Returns:
        Number of rows in merged file

    Example:
        >>> merge_parquet_files(
        ...     [Path('/tmp/worker_0.parquet'), Path('/tmp/worker_1.parquet')],
        ...     Path('data/vector/signals.parquet'),
        ...     key_cols=['signal_id', 'obs_date', 'engine', 'metric_name']
        ... )
    """
    dfs = []
    for path in input_paths:
        if path.exists():
            dfs.append(pl.read_parquet(path))

    if not dfs:
        return 0

    combined = pl.concat(dfs, how="diagonal_relaxed")

    if key_cols:
        # Deduplicate, keeping last occurrence
        combined = combined.unique(subset=key_cols, keep="last")

    result = write_parquet_atomic(combined, output_path, compression)

    if delete_inputs:
        for path in input_paths:
            if path.exists():
                path.unlink()

    return result


def get_parquet_schema(path: Union[str, Path]) -> Optional[dict]:
    """
    Get the schema of a parquet file without reading data.

    Args:
        path: Path to parquet file

    Returns:
        Dict mapping column names to dtypes, or None if file doesn't exist

    Example:
        >>> schema = get_parquet_schema('data/raw/observations.parquet')
        >>> print(schema)
        {'signal_id': String, 'obs_date': Date, 'value': Float64}
    """
    path = Path(path)
    if not path.exists():
        return None

    # Use scan to get schema without reading data
    lf = pl.scan_parquet(path)
    return dict(lf.schema)


def get_row_count(path: Union[str, Path]) -> int:
    """
    Get the row count of a parquet file efficiently.

    Args:
        path: Path to parquet file

    Returns:
        Number of rows, or 0 if file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        return 0

    # Use lazy scan with count for efficiency
    return pl.scan_parquet(path).select(pl.len()).collect().item()


# Convenience functions for common operations


def read_file(file: str, columns: Optional[List[str]] = None) -> pl.DataFrame:
    """
    Read a file from the PRISM data store.

    Args:
        file: File constant (OBSERVATIONS, SIGNALS, GEOMETRY, STATE, COHORTS)
        columns: Optional list of columns to read

    Returns:
        Polars DataFrame

    Example:
        >>> obs = read_file(OBSERVATIONS)
        >>> signals = read_file(SIGNALS, columns=['entity_id', 'signal_id', 'value'])
    """
    path = get_path(file)
    return read_parquet(path, columns=columns)


def write_file(
    df: pl.DataFrame,
    file: str,
    mode: str = "replace",
    key_cols: Optional[List[str]] = None,
) -> int:
    """
    Write a DataFrame to the PRISM data store.

    Args:
        df: Polars DataFrame to write
        file: File constant (OBSERVATIONS, SIGNALS, GEOMETRY, STATE, COHORTS)
        mode: 'replace' (overwrite), 'append', or 'upsert'
        key_cols: Required for 'upsert' mode

    Returns:
        Number of rows written/total

    Example:
        >>> write_file(observations, OBSERVATIONS, mode='upsert',
        ...            key_cols=['entity_id', 'signal_id', 'timestamp'])
    """
    path = get_path(file)

    if mode == "replace":
        return write_parquet_atomic(df, path)
    elif mode == "append":
        return append_parquet(df, path)
    elif mode == "upsert":
        if key_cols is None:
            raise ValueError("key_cols required for upsert mode")
        return upsert_parquet(df, path, key_cols)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'replace', 'append', or 'upsert'")
