"""
PRISM Streaming Data Access - DuckDB-based

Stream data from parquet files without loading into RAM.
Target: < 1GB RAM regardless of input size.

Usage:
    from prism.db.streaming import StreamingReader, IncrementalWriter

    # Read windows one at a time
    reader = StreamingReader(parquet_path)
    for entity_id, signal_id, values, indices in reader.iter_signals():
        for window in generate_windows(values, indices, ...):
            result = process(window)
            writer.write_row(result)

    # Write results incrementally
    writer = IncrementalWriter(output_path, schema)
    writer.write_row(result_dict)
    writer.close()
"""

import os
from pathlib import Path
from typing import Iterator, Tuple, Dict, Any, List, Optional
import numpy as np

# Try DuckDB first, fall back to polars chunks
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


def check_memory(label: str = "") -> float:
    """Check current memory usage. Returns GB."""
    try:
        import psutil
        mem_gb = psutil.Process().memory_info().rss / 1e9
        if os.getenv('PRISM_VERBOSE') == '1':
            print(f"[MEM {label}] {mem_gb:.2f} GB")
        if mem_gb > 1.0:
            print(f"WARNING: Memory exceeding 1GB budget at {label}")
        return mem_gb
    except ImportError:
        return 0.0


class StreamingReader:
    """
    Stream signals from parquet without loading entire file.

    Uses DuckDB for efficient parquet scanning.
    """

    def __init__(self, parquet_path: Path, chunk_size: int = 100):
        """
        Initialize streaming reader.

        Args:
            parquet_path: Path to observations.parquet
            chunk_size: Number of signals to load per batch
        """
        self.parquet_path = Path(parquet_path)
        self.chunk_size = chunk_size

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        if HAS_DUCKDB:
            self.con = duckdb.connect()
            self._init_duckdb()
        else:
            self.con = None
            self._init_polars()

    def _init_duckdb(self):
        """Initialize DuckDB connection and get metadata."""
        # Get basic stats without loading data
        stats = self.con.execute(f"""
            SELECT
                COUNT(*) as total_rows,
                COUNT(DISTINCT entity_id) as n_entities,
                COUNT(DISTINCT signal_id) as n_signals
            FROM '{self.parquet_path}'
        """).fetchone()

        self.total_rows = stats[0]
        self.n_entities = stats[1]
        self.n_signals = stats[2]

        # Get unique entity/signal combinations
        self.signal_keys = self.con.execute(f"""
            SELECT DISTINCT entity_id, signal_id
            FROM '{self.parquet_path}'
            ORDER BY entity_id, signal_id
        """).fetchall()

    def _init_polars(self):
        """Initialize with polars lazy scanning."""
        # Use lazy frame to get metadata
        lf = pl.scan_parquet(self.parquet_path)

        # Get counts via lazy evaluation
        stats = lf.select([
            pl.count().alias('total_rows'),
            pl.col('entity_id').n_unique().alias('n_entities'),
            pl.col('signal_id').n_unique().alias('n_signals'),
        ]).collect()

        self.total_rows = stats['total_rows'][0]
        self.n_entities = stats['n_entities'][0]
        self.n_signals = stats['n_signals'][0]

        # Get unique combinations
        keys = lf.select(['entity_id', 'signal_id']).unique().collect()
        self.signal_keys = list(zip(keys['entity_id'].to_list(), keys['signal_id'].to_list()))

    def iter_signals(self) -> Iterator[Tuple[str, str, np.ndarray, np.ndarray]]:
        """
        Iterate over signals, yielding one at a time.

        Yields:
            (entity_id, signal_id, values_array, indices_array)
        """
        if HAS_DUCKDB:
            yield from self._iter_signals_duckdb()
        else:
            yield from self._iter_signals_polars()

    def _iter_signals_duckdb(self) -> Iterator[Tuple[str, str, np.ndarray, np.ndarray]]:
        """Stream signals via DuckDB."""
        # Determine index column
        columns = self.con.execute(f"""
            SELECT column_name FROM (DESCRIBE SELECT * FROM '{self.parquet_path}')
        """).fetchall()
        columns = [c[0] for c in columns]

        index_col = 'index' if 'index' in columns else 'timestamp'

        for entity_id, signal_id in self.signal_keys:
            # Fetch only this signal's data
            result = self.con.execute(f"""
                SELECT {index_col} as idx, value
                FROM '{self.parquet_path}'
                WHERE entity_id = ? AND signal_id = ?
                ORDER BY {index_col}
            """, [entity_id, signal_id]).fetchnumpy()

            indices = result['idx'].astype(np.float64)
            values = result['value'].astype(np.float64)

            yield entity_id, signal_id, values, indices

    def _iter_signals_polars(self) -> Iterator[Tuple[str, str, np.ndarray, np.ndarray]]:
        """Stream signals via polars lazy frames."""
        lf = pl.scan_parquet(self.parquet_path)

        # Determine index column
        schema = lf.collect_schema()
        index_col = 'index' if 'index' in schema else 'timestamp'

        for entity_id, signal_id in self.signal_keys:
            # Fetch only this signal
            signal_df = (
                lf
                .filter(
                    (pl.col('entity_id') == entity_id) &
                    (pl.col('signal_id') == signal_id)
                )
                .select([index_col, 'value'])
                .sort(index_col)
                .collect()
            )

            indices = signal_df[index_col].to_numpy().astype(np.float64)
            values = signal_df['value'].to_numpy().astype(np.float64)

            yield entity_id, signal_id, values, indices

    def get_bounds(self) -> Tuple[float, float]:
        """Get min/max index values."""
        if HAS_DUCKDB:
            # Determine index column
            columns = self.con.execute(f"""
                SELECT column_name FROM (DESCRIBE SELECT * FROM '{self.parquet_path}')
            """).fetchall()
            columns = [c[0] for c in columns]
            index_col = 'index' if 'index' in columns else 'timestamp'

            result = self.con.execute(f"""
                SELECT MIN({index_col}), MAX({index_col})
                FROM '{self.parquet_path}'
            """).fetchone()
            return result[0], result[1]
        else:
            lf = pl.scan_parquet(self.parquet_path)
            schema = lf.collect_schema()
            index_col = 'index' if 'index' in schema else 'timestamp'

            result = lf.select([
                pl.col(index_col).min().alias('min'),
                pl.col(index_col).max().alias('max'),
            ]).collect()
            return result['min'][0], result['max'][0]

    def close(self):
        """Clean up resources."""
        if self.con is not None:
            self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class IncrementalWriter:
    """
    Write results incrementally to parquet.

    Writes in batches to avoid accumulating in RAM.
    """

    def __init__(
        self,
        output_path: Path,
        batch_size: int = 1000,
        schema: Optional[pa.Schema] = None
    ):
        """
        Initialize incremental writer.

        Args:
            output_path: Path for output parquet
            batch_size: Rows to accumulate before writing
            schema: Optional PyArrow schema (inferred from first batch if not provided)
        """
        self.output_path = Path(output_path)
        self.batch_size = batch_size
        self.schema = schema

        self.buffer: List[Dict[str, Any]] = []
        self.writer: Optional[pq.ParquetWriter] = None
        self.rows_written = 0
        self._column_order: Optional[List[str]] = None

    def write_row(self, row: Dict[str, Any]):
        """Add a row to the buffer, flush if full."""
        self.buffer.append(row)

        if len(self.buffer) >= self.batch_size:
            self._flush()

    def _flush(self):
        """Write buffer to parquet."""
        if not self.buffer:
            return

        # Convert to polars DataFrame
        df = pl.DataFrame(self.buffer)

        # Track all columns seen (for schema evolution)
        if self.writer is None:
            # First batch - sort columns alphabetically for consistent ordering
            # Keep id columns first, then sort the rest
            id_cols = ['entity_id', 'signal_id', 'window_idx', 'window_start', 'window_end', 'n_samples']
            other_cols = sorted([c for c in df.columns if c not in id_cols])
            col_order = [c for c in id_cols if c in df.columns] + other_cols
            df = df.select(col_order)

            self._column_order = col_order
            table = df.to_arrow()
            self.schema = table.schema
            self.writer = pq.ParquetWriter(str(self.output_path), self.schema)
        else:
            # Subsequent batches - ensure same column order
            # Add missing columns as null, drop extra columns
            for col in self._column_order:
                if col not in df.columns:
                    df = df.with_columns(pl.lit(None).alias(col))

            # Select in exact same order
            df = df.select(self._column_order)
            table = df.to_arrow()

        # Write batch
        self.writer.write_table(table)
        self.rows_written += len(self.buffer)

        # Clear buffer
        self.buffer = []

        if os.getenv('PRISM_VERBOSE') == '1':
            check_memory(f"after_flush_{self.rows_written}")

    def close(self) -> int:
        """Flush remaining and close writer. Returns total rows written."""
        self._flush()

        if self.writer is not None:
            self.writer.close()

        return self.rows_written

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def stream_process_signals(
    parquet_path: Path,
    output_path: Path,
    process_fn,
    batch_size: int = 1000,
) -> int:
    """
    Stream process signals from parquet.

    Args:
        parquet_path: Input observations.parquet
        output_path: Output results.parquet
        process_fn: Function(entity_id, signal_id, values, indices) -> List[Dict]
        batch_size: Rows per write batch

    Returns:
        Total rows written
    """
    check_memory("start")

    with StreamingReader(parquet_path) as reader:
        with IncrementalWriter(output_path, batch_size) as writer:
            n_signals = len(reader.signal_keys)

            for i, (entity_id, signal_id, values, indices) in enumerate(reader.iter_signals()):
                # Process this signal
                results = process_fn(entity_id, signal_id, values, indices)

                # Write results immediately
                for row in results:
                    writer.write_row(row)

                if (i + 1) % 100 == 0:
                    check_memory(f"signal_{i+1}/{n_signals}")

            check_memory("end")
            return writer.rows_written
