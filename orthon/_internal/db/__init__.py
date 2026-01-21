"""
PRISM Database Layer
====================

5-file Parquet storage with Polars I/O.

Files:
    observations.parquet  - Raw sensor data
    signals.parquet       - Behavioral signals (dense + sparse)
    geometry.parquet      - System structure at each timestamp
    state.parquet         - Dynamics at each timestamp
    cohorts.parquet       - Discovered entity groupings

Usage:
    from orthon._internal.db import get_path, OBSERVATIONS, SIGNALS, GEOMETRY, STATE, COHORTS

    # Get path to a file
    obs_path = get_path(OBSERVATIONS)

    # Read with Polars
    import polars as pl
    obs = pl.read_parquet(get_path(OBSERVATIONS))
"""

# Path management
from orthon._internal.db.parquet_store import (
    # Core functions
    get_data_root,
    get_path,
    ensure_directory,
    file_exists,
    get_file_size,
    delete_file,
    list_files,
    get_status,
    # File constants
    OBSERVATIONS,
    VECTOR,
    SIGNALS,  # Alias for VECTOR
    GEOMETRY,
    STATE,
    COHORTS,
    FILES,
)

# Polars I/O
from orthon._internal.db.polars_io import (
    read_parquet,
    write_parquet_atomic,
    upsert_parquet,
    append_parquet,
    read_file,
    write_file,
    get_row_count,
    get_parquet_schema,
)

# Query utilities
from orthon._internal.db.query import (
    describe_table,
    table_stats,
)

# Temporary storage
from orthon._internal.db.scratch import (
    TempParquet,
    ParquetBatchWriter,
    merge_temp_results,
    merge_to_table,
)

__all__ = [
    # Core path functions
    "get_data_root",
    "get_path",
    "ensure_directory",
    "file_exists",
    "get_file_size",
    "delete_file",
    "list_files",
    "get_status",
    # File constants
    "OBSERVATIONS",
    "VECTOR",
    "SIGNALS",
    "GEOMETRY",
    "STATE",
    "COHORTS",
    "FILES",
    # polars_io
    "read_parquet",
    "write_parquet_atomic",
    "upsert_parquet",
    "append_parquet",
    "read_file",
    "write_file",
    "get_row_count",
    "get_parquet_schema",
    # query
    "describe_table",
    "table_stats",
    # scratch
    "TempParquet",
    "ParquetBatchWriter",
    "merge_temp_results",
    "merge_to_table",
]
