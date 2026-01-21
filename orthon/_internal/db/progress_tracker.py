"""
PRISM Progress Tracker

Simple progress tracking for incremental computation.
Tracks completion at signal+tier level (not per-window).

Schema:
    schema: str (vector, geometry, state)
    table: str (signals, cohorts, pairs)
    item_id: str (signal_id or cohort_id)
    tier: str (anchor, bridge, scout, micro)
    status: str (pending, in_progress, completed, failed)
    rows_written: int
    started_at: datetime
    completed_at: datetime (nullable)

Usage:
    tracker = ProgressTracker("vector", "signals")

    # Check what's already done
    completed = tracker.get_completed("anchor")
    pending = tracker.get_pending(all_items, "anchor")

    # Mark progress
    tracker.mark_started("SENSOR_01", "anchor")
    tracker.mark_completed("SENSOR_01", "anchor", rows=1234)
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import polars as pl

from orthon._internal.db.parquet_store import ensure_directory, get_data_root

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Track computation progress at item+tier level."""

    def __init__(self, schema: str, table: str):
        """
        Initialize progress tracker.

        Args:
            schema: e.g., "vector", "geometry", "state"
            table: e.g., "signals", "cohorts", "pairs"
        """
        self.schema = schema
        self.table = table
        # Progress files stored in domain's .progress directory
        self.progress_path = get_data_root() / ".progress" / f"progress_{schema}_{table}.parquet"
        self._cache: Optional[pl.DataFrame] = None
        self._load()

    def _load(self):
        """Load existing progress data."""
        if self.progress_path.exists():
            try:
                self._cache = pl.read_parquet(self.progress_path)
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}")
                self._cache = self._empty_df()
        else:
            self._cache = self._empty_df()

    def _empty_df(self) -> pl.DataFrame:
        """Create empty progress dataframe."""
        return pl.DataFrame({
            "schema": pl.Series([], dtype=pl.Utf8),
            "table": pl.Series([], dtype=pl.Utf8),
            "item_id": pl.Series([], dtype=pl.Utf8),
            "tier": pl.Series([], dtype=pl.Utf8),
            "status": pl.Series([], dtype=pl.Utf8),
            "rows_written": pl.Series([], dtype=pl.Int64),
            "started_at": pl.Series([], dtype=pl.Datetime),
            "completed_at": pl.Series([], dtype=pl.Datetime),
        })

    def _save(self):
        """Save progress data."""
        ensure_directory()
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache.write_parquet(self.progress_path)

    def get_completed(self, tier: str) -> Set[str]:
        """Get set of completed item IDs for a tier."""
        if self._cache is None or len(self._cache) == 0:
            return set()

        completed = self._cache.filter(
            (pl.col("schema") == self.schema) &
            (pl.col("table") == self.table) &
            (pl.col("tier") == tier) &
            (pl.col("status") == "completed")
        )
        return set(completed["item_id"].to_list())

    def get_pending(self, all_items: List[str], tier: str) -> List[str]:
        """Get items that haven't been completed for a tier."""
        completed = self.get_completed(tier)
        return [item for item in all_items if item not in completed]

    def get_in_progress(self, tier: str) -> Set[str]:
        """Get items currently in progress (may need recovery)."""
        if self._cache is None or len(self._cache) == 0:
            return set()

        in_progress = self._cache.filter(
            (pl.col("schema") == self.schema) &
            (pl.col("table") == self.table) &
            (pl.col("tier") == tier) &
            (pl.col("status") == "in_progress")
        )
        return set(in_progress["item_id"].to_list())

    def mark_started(self, item_id: str, tier: str):
        """Mark an item as started/in_progress."""
        now = datetime.now()

        # Remove existing record for this item+tier
        self._cache = self._cache.filter(
            ~((pl.col("schema") == self.schema) &
              (pl.col("table") == self.table) &
              (pl.col("item_id") == item_id) &
              (pl.col("tier") == tier))
        )

        # Add new record
        new_row = pl.DataFrame({
            "schema": [self.schema],
            "table": [self.table],
            "item_id": [item_id],
            "tier": [tier],
            "status": ["in_progress"],
            "rows_written": [0],
            "started_at": [now],
            "completed_at": [None],
        })
        self._cache = pl.concat([self._cache, new_row])
        self._save()

    def mark_completed(self, item_id: str, tier: str, rows: int = 0):
        """Mark an item as completed."""
        now = datetime.now()

        # Get started_at from existing record
        existing = self._cache.filter(
            (pl.col("schema") == self.schema) &
            (pl.col("table") == self.table) &
            (pl.col("item_id") == item_id) &
            (pl.col("tier") == tier)
        )
        started_at = existing["started_at"].to_list()[0] if len(existing) > 0 else now

        # Remove existing record
        self._cache = self._cache.filter(
            ~((pl.col("schema") == self.schema) &
              (pl.col("table") == self.table) &
              (pl.col("item_id") == item_id) &
              (pl.col("tier") == tier))
        )

        # Add completed record
        new_row = pl.DataFrame({
            "schema": [self.schema],
            "table": [self.table],
            "item_id": [item_id],
            "tier": [tier],
            "status": ["completed"],
            "rows_written": [rows],
            "started_at": [started_at],
            "completed_at": [now],
        })
        self._cache = pl.concat([self._cache, new_row])
        self._save()

    def mark_failed(self, item_id: str, tier: str, error: str = ""):
        """Mark an item as failed."""
        now = datetime.now()

        # Get started_at from existing record
        existing = self._cache.filter(
            (pl.col("schema") == self.schema) &
            (pl.col("table") == self.table) &
            (pl.col("item_id") == item_id) &
            (pl.col("tier") == tier)
        )
        started_at = existing["started_at"].to_list()[0] if len(existing) > 0 else now

        # Remove existing record
        self._cache = self._cache.filter(
            ~((pl.col("schema") == self.schema) &
              (pl.col("table") == self.table) &
              (pl.col("item_id") == item_id) &
              (pl.col("tier") == tier))
        )

        # Add failed record
        new_row = pl.DataFrame({
            "schema": [self.schema],
            "table": [self.table],
            "item_id": [item_id],
            "tier": [tier],
            "status": ["failed"],
            "rows_written": [0],
            "started_at": [started_at],
            "completed_at": [now],
        })
        self._cache = pl.concat([self._cache, new_row])
        self._save()

    def reset_in_progress(self, tier: Optional[str] = None):
        """Reset in_progress items to pending (for crash recovery)."""
        filter_cond = (
            (pl.col("schema") == self.schema) &
            (pl.col("table") == self.table) &
            (pl.col("status") == "in_progress")
        )
        if tier:
            filter_cond = filter_cond & (pl.col("tier") == tier)

        # Remove in_progress records (they'll be recomputed)
        self._cache = self._cache.filter(~filter_cond)
        self._save()

    def get_stats(self, tier: Optional[str] = None) -> Dict[str, int]:
        """Get progress statistics."""
        if self._cache is None or len(self._cache) == 0:
            return {"completed": 0, "in_progress": 0, "failed": 0, "total_rows": 0}

        filter_cond = (
            (pl.col("schema") == self.schema) &
            (pl.col("table") == self.table)
        )
        if tier:
            filter_cond = filter_cond & (pl.col("tier") == tier)

        filtered = self._cache.filter(filter_cond)

        return {
            "completed": len(filtered.filter(pl.col("status") == "completed")),
            "in_progress": len(filtered.filter(pl.col("status") == "in_progress")),
            "failed": len(filtered.filter(pl.col("status") == "failed")),
            "total_rows": filtered["rows_written"].sum() or 0,
        }

    def clear(self, tier: Optional[str] = None):
        """Clear progress data (for fresh run)."""
        if tier:
            self._cache = self._cache.filter(
                ~((pl.col("schema") == self.schema) &
                  (pl.col("table") == self.table) &
                  (pl.col("tier") == tier))
            )
        else:
            self._cache = self._cache.filter(
                ~((pl.col("schema") == self.schema) &
                  (pl.col("table") == self.table))
            )
        self._save()
