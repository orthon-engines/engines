"""
PRISM Incremental Writer

Reusable utility for batch writing results with progress tracking.
Used by all runners (vector, geometry, state) to prevent data loss.

Features:
- Batch accumulation with periodic writes
- Integration with window_schedule for progress tracking
- Resume capability (skip already-computed items)
- Safe atomic writes via polars

Usage:
    writer = IncrementalWriter(
        schema="vector",
        table="signals",
        key_cols=["signal_id", "obs_date", "target_obs", "engine", "metric_name"],
        batch_size=10,  # Write every 10 items
    )

    for signal_id in signals:
        rows = process_signal(signal_id)
        writer.add_rows(rows, item_id=signal_id, tier="anchor")

    writer.flush()  # Final write
    stats = writer.get_stats()
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import polars as pl

from orthon._internal.db.parquet_store import (
    get_path,
    get_data_root,
    OBSERVATIONS,
    SIGNALS,
    GEOMETRY,
    STATE,
    COHORTS,
)
from orthon._internal.db.polars_io import upsert_parquet

logger = logging.getLogger(__name__)


@dataclass
class WriterStats:
    """Statistics for incremental writer."""
    items_processed: int = 0
    rows_written: int = 0
    batches_written: int = 0
    items_skipped: int = 0  # Already computed
    errors: int = 0
    start_time: datetime = field(default_factory=datetime.now)

    def elapsed_seconds(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()

    def rows_per_second(self) -> float:
        elapsed = self.elapsed_seconds()
        return self.rows_written / elapsed if elapsed > 0 else 0


class IncrementalWriter:
    """
    Incremental batch writer with progress tracking.

    Accumulates rows and writes periodically to prevent data loss.
    Integrates with window_schedule for resume capability.
    """

    def __init__(
        self,
        schema: str,
        table: str,
        key_cols: List[str],
        batch_size: int = 10,
        use_schedule: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize incremental writer.

        Args:
            schema: Parquet schema (e.g., "vector", "geometry", "state")
            table: Table name (e.g., "signals", "cohorts", "pairs")
            key_cols: Columns that form unique key for upsert
            batch_size: Number of items to accumulate before writing
            use_schedule: Whether to use window_schedule for tracking
            verbose: Print progress messages
        """
        self.schema = schema
        self.table = table
        self.key_cols = key_cols
        self.batch_size = batch_size
        self.use_schedule = use_schedule
        self.verbose = verbose

        # Map schema to file constant
        FILE_MAP = {
            'raw': OBSERVATIONS,
            'observations': OBSERVATIONS,
            'vector': SIGNALS,
            'signals': SIGNALS,
            'geometry': GEOMETRY,
            'state': STATE,
            'config': COHORTS,
            'cohorts': COHORTS,
        }
        file_const = FILE_MAP.get(schema, SIGNALS)
        self.target_path = get_path(file_const)
        self.buffer: List[Dict[str, Any]] = []
        self.items_in_batch: List[Tuple[str, str]] = []  # (item_id, tier) pairs
        self.stats = WriterStats()

        # Load already-computed items for resume capability
        self.computed_keys: Set[Tuple] = set()
        if use_schedule:
            self._load_computed_keys()

    def _load_computed_keys(self):
        """Load already-computed items from schedule or existing data."""
        try:
            schedule_path = get_data_root() / "window_schedule.parquet"
            if schedule_path.exists():
                schedule = pl.read_parquet(schedule_path)
                computed = schedule.filter(pl.col("status") == "computed")
                for row in computed.iter_rows(named=True):
                    key = (row["signal_id"], row["tier"], row["window_end"])
                    self.computed_keys.add(key)
                if self.verbose and len(self.computed_keys) > 0:
                    logger.info(f"Loaded {len(self.computed_keys):,} already-computed windows from schedule")
        except Exception as e:
            logger.debug(f"Could not load schedule: {e}")

    def is_computed(self, item_id: str, tier: str, window_end: Optional[Any] = None) -> bool:
        """Check if an item/window is already computed."""
        if not self.use_schedule:
            return False
        if window_end is not None:
            return (item_id, tier, window_end) in self.computed_keys
        # Check if ANY window for this item/tier is computed
        return any(k[0] == item_id and k[1] == tier for k in self.computed_keys)

    def add_rows(
        self,
        rows: List[Dict[str, Any]],
        item_id: str,
        tier: str,
        window_end: Optional[Any] = None,
    ):
        """
        Add rows to buffer and write if batch is full.

        Args:
            rows: List of row dicts to add
            item_id: Identifier for this item (e.g., signal_id)
            tier: Window tier name
            window_end: Optional specific window end date
        """
        if not rows:
            return

        self.buffer.extend(rows)
        self.items_in_batch.append((item_id, tier))
        self.stats.items_processed += 1

        # Check if batch is full
        if len(self.items_in_batch) >= self.batch_size:
            self._write_batch()

    def _write_batch(self):
        """Write current batch to parquet."""
        if not self.buffer:
            return

        try:
            df = pl.DataFrame(self.buffer, infer_schema_length=None)
            upsert_parquet(df, self.target_path, self.key_cols)

            rows_written = len(self.buffer)
            self.stats.rows_written += rows_written
            self.stats.batches_written += 1

            if self.verbose:
                elapsed = self.stats.elapsed_seconds()
                rate = self.stats.rows_written / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Batch {self.stats.batches_written}: wrote {rows_written:,} rows "
                    f"(total: {self.stats.rows_written:,}, {rate:.0f} rows/sec)"
                )

            # Update schedule if using it
            if self.use_schedule:
                self._update_schedule()

            # Clear buffer
            self.buffer = []
            self.items_in_batch = []

        except Exception as e:
            logger.error(f"Failed to write batch: {e}")
            self.stats.errors += 1
            raise

    def _update_schedule(self):
        """Update window schedule with completed items."""
        try:
            from orthon._internal.db.window_schedule import mark_computed

            for item_id, tier in self.items_in_batch:
                # Mark all windows for this item/tier as computed
                # In practice, we'd want more granular tracking by window_end
                pass  # TODO: Implement granular schedule updates

        except ImportError:
            pass  # Schedule module not available

    def flush(self):
        """Write any remaining buffered rows."""
        if self.buffer:
            self._write_batch()

    def get_stats(self) -> WriterStats:
        """Get current statistics."""
        return self.stats

    def summary(self) -> str:
        """Get summary string."""
        s = self.stats
        elapsed = s.elapsed_seconds()
        return (
            f"Items: {s.items_processed:,} processed, {s.items_skipped:,} skipped | "
            f"Rows: {s.rows_written:,} written in {s.batches_written} batches | "
            f"Rate: {s.rows_per_second():.0f} rows/sec | "
            f"Time: {elapsed:.1f}s | "
            f"Errors: {s.errors}"
        )


class TierRunner:
    """
    Base class for running computations across a tier.

    Handles:
    - Loading signals/items to process
    - Checking what's already computed
    - Running computation with incremental writes
    - Progress tracking and resume
    """

    def __init__(
        self,
        schema: str,
        table: str,
        key_cols: List[str],
        tier: str,
        batch_size: int = 10,
        verbose: bool = True,
    ):
        self.schema = schema
        self.table = table
        self.key_cols = key_cols
        self.tier = tier
        self.batch_size = batch_size
        self.verbose = verbose

        self.writer = IncrementalWriter(
            schema=schema,
            table=table,
            key_cols=key_cols,
            batch_size=batch_size,
            use_schedule=True,
            verbose=verbose,
        )

    def get_pending_items(self, all_items: List[str]) -> List[str]:
        """Filter to items that haven't been fully computed."""
        pending = []
        for item_id in all_items:
            if not self.writer.is_computed(item_id, self.tier):
                pending.append(item_id)
            else:
                self.writer.stats.items_skipped += 1

        if self.verbose and self.writer.stats.items_skipped > 0:
            logger.info(
                f"Skipping {self.writer.stats.items_skipped} already-computed items, "
                f"{len(pending)} pending"
            )

        return pending

    def run(
        self,
        items: List[str],
        compute_fn,
        skip_computed: bool = True,
    ) -> WriterStats:
        """
        Run computation across items with incremental writes.

        Args:
            items: List of item IDs to process
            compute_fn: Function(item_id) -> List[Dict] that computes rows
            skip_computed: Whether to skip already-computed items

        Returns:
            WriterStats with summary
        """
        if skip_computed:
            items = self.get_pending_items(items)

        if not items:
            if self.verbose:
                logger.info("No items to process")
            return self.writer.get_stats()

        if self.verbose:
            logger.info(f"Processing {len(items)} items for tier {self.tier}")

        for i, item_id in enumerate(items):
            try:
                rows = compute_fn(item_id)
                self.writer.add_rows(rows, item_id=item_id, tier=self.tier)

                if self.verbose and (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i+1}/{len(items)} items")

            except Exception as e:
                logger.error(f"Error processing {item_id}: {e}")
                self.writer.stats.errors += 1

        self.writer.flush()

        if self.verbose:
            logger.info(f"Complete: {self.writer.summary()}")

        return self.writer.get_stats()
