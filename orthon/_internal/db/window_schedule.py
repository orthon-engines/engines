"""
PRISM Window Schedule

Persists the window computation schedule so we can:
- Track progress across runs
- Resume interrupted computations
- Know expected workload upfront
- Detect missing windows

Schema:
    signal_id: str
    tier: str (anchor, bridge, scout, micro)
    window_end: date
    lookback_start: date
    target_obs: int
    status: str (pending, computed, skipped, failed)
    computed_at: datetime (nullable)

Usage:
    from orthon._internal.db.window_schedule import (
        generate_schedule,
        get_pending_windows,
        mark_computed,
        get_schedule_stats,
    )

    # Generate full schedule (run once or when data changes)
    generate_schedule()

    # Get windows that need computation
    pending = get_pending_windows(tier='anchor')

    # Mark windows as computed
    mark_computed(signal_id='SENSOR_01', tier='anchor', window_end=date(2024,1,15))
"""

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from orthon._internal.db.parquet_store import (
    ensure_directory,
    get_path,
    get_data_root,
    OBSERVATIONS,
    SIGNALS,
    GEOMETRY,
    STATE,
    COHORTS,
)
from orthon._internal.db.polars_io import read_parquet, upsert_parquet, write_parquet_atomic
from orthon._internal.utils.stride import load_stride_config

logger = logging.getLogger(__name__)


def _get_schedule_path() -> Path:
    """Get path to window schedule file."""
    return get_data_root() / "window_schedule.parquet"


def _compute_windows_for_signal(
    obs_dates: np.ndarray,
    target_obs: int,
    stride: int,
    min_obs: int,
) -> List[Tuple[date, date, int]]:
    """
    Compute window endpoints for a single signal.

    Returns list of (window_end, lookback_start, actual_obs) tuples.
    """
    n = len(obs_dates)
    if n < min_obs:
        return []

    windows = []
    start_idx = max(min_obs, target_obs) - 1

    for end_idx in range(start_idx, n, stride):
        window_start_idx = max(0, end_idx - target_obs + 1)
        window_len = end_idx - window_start_idx + 1

        if window_len < min_obs:
            continue

        window_end = obs_dates[end_idx]
        lookback_start = obs_dates[window_start_idx]

        # Convert numpy datetime64 to Python date
        if hasattr(window_end, 'astype'):
            window_end = pl.Series([window_end]).cast(pl.Date).to_list()[0]
        if hasattr(lookback_start, 'astype'):
            lookback_start = pl.Series([lookback_start]).cast(pl.Date).to_list()[0]

        windows.append((window_end, lookback_start, window_len))

    return windows


def generate_schedule(
    tiers: Optional[List[str]] = None,
    signals: Optional[List[str]] = None,
    force: bool = False,
) -> Dict[str, int]:
    """
    Generate the full window schedule for all signals and tiers.

    Args:
        tiers: Specific tiers to generate (default: all)
        signals: Specific signals (default: all)
        force: Overwrite existing schedule (default: False, only add new)

    Returns:
        Dict with counts by tier
    """
    ensure_directory()

    # Load config
    stride_config = load_stride_config()
    if tiers is None:
        tiers = stride_config.list_windows()

    # Load observations
    obs_path = get_path(OBSERVATIONS)
    if not obs_path.exists():
        logger.error("No observations found")
        return {}

    obs = pl.read_parquet(obs_path)

    if signals:
        obs = obs.filter(pl.col("signal_id").is_in(signals))

    # Group by signal
    signal_groups = obs.group_by("signal_id").agg([
        pl.col("obs_date").sort().alias("dates")
    ])

    # Load existing schedule if not forcing
    existing = set()
    if not force and _get_schedule_path().exists():
        try:
            existing_df = pl.read_parquet(_get_schedule_path())
            for row in existing_df.select(["signal_id", "tier", "window_end"]).iter_rows():
                existing.add((row[0], row[1], row[2]))
        except Exception:
            pass

    # Generate schedule rows
    rows = []
    counts = {tier: 0 for tier in tiers}

    for row in signal_groups.iter_rows(named=True):
        signal_id = row["signal_id"]
        dates = np.array(row["dates"])

        for tier in tiers:
            window_config = stride_config.get_window(tier)
            target_obs = window_config.window_days
            stride = window_config.stride_days
            min_obs = window_config.min_observations

            windows = _compute_windows_for_signal(dates, target_obs, stride, min_obs)

            for window_end, lookback_start, actual_obs in windows:
                # Skip if already exists
                key = (signal_id, tier, window_end)
                if key in existing:
                    continue

                rows.append({
                    "signal_id": signal_id,
                    "tier": tier,
                    "window_end": window_end,
                    "lookback_start": lookback_start,
                    "target_obs": target_obs,
                    "actual_obs": actual_obs,
                    "status": "pending",
                    "computed_at": None,
                })
                counts[tier] += 1

    if not rows:
        logger.info("No new windows to add to schedule")
        return counts

    # Create DataFrame
    df = pl.DataFrame(rows)

    # Write or append
    if force or not _get_schedule_path().exists():
        df.write_parquet(_get_schedule_path())
        logger.info(f"Created schedule with {len(df):,} windows")
    else:
        # Append to existing
        existing_df = pl.read_parquet(_get_schedule_path())
        combined = pl.concat([existing_df, df])
        combined.write_parquet(_get_schedule_path())
        logger.info(f"Added {len(df):,} windows to schedule (total: {len(combined):,})")

    return counts


def get_schedule_stats() -> pl.DataFrame:
    """
    Get summary statistics of the window schedule.

    Returns DataFrame with counts by tier and status.
    """
    if not _get_schedule_path().exists():
        return pl.DataFrame({
            "tier": [],
            "status": [],
            "count": [],
        })

    df = pl.read_parquet(_get_schedule_path())

    return df.group_by(["tier", "status"]).agg([
        pl.len().alias("count")
    ]).sort(["tier", "status"])


def get_pending_windows(
    tier: Optional[str] = None,
    signal_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> pl.DataFrame:
    """
    Get windows that are pending computation.

    Args:
        tier: Filter by tier
        signal_id: Filter by signal
        limit: Max number of windows to return

    Returns:
        DataFrame of pending windows
    """
    if not _get_schedule_path().exists():
        return pl.DataFrame()

    df = pl.read_parquet(_get_schedule_path())
    df = df.filter(pl.col("status") == "pending")

    if tier:
        df = df.filter(pl.col("tier") == tier)

    if signal_id:
        df = df.filter(pl.col("signal_id") == signal_id)

    df = df.sort(["tier", "signal_id", "window_end"])

    if limit:
        df = df.head(limit)

    return df


def mark_computed(
    signal_id: str,
    tier: str,
    window_end: date,
    status: str = "computed",
) -> bool:
    """
    Mark a window as computed (or failed/skipped).

    Args:
        signal_id: Signal ID
        tier: Window tier
        window_end: Window end date
        status: New status (computed, failed, skipped)

    Returns:
        True if updated, False if not found
    """
    if not _get_schedule_path().exists():
        return False

    df = pl.read_parquet(_get_schedule_path())

    # Find matching row
    mask = (
        (pl.col("signal_id") == signal_id) &
        (pl.col("tier") == tier) &
        (pl.col("window_end") == window_end)
    )

    # Update status and computed_at
    df = df.with_columns([
        pl.when(mask).then(pl.lit(status)).otherwise(pl.col("status")).alias("status"),
        pl.when(mask).then(pl.lit(datetime.now())).otherwise(pl.col("computed_at")).alias("computed_at"),
    ])

    df.write_parquet(_get_schedule_path())
    return True


def mark_batch_computed(
    records: List[Tuple[str, str, date]],
    status: str = "computed",
) -> int:
    """
    Mark multiple windows as computed in a single operation.

    Args:
        records: List of (signal_id, tier, window_end) tuples
        status: New status

    Returns:
        Number of records updated
    """
    if not _get_schedule_path().exists() or not records:
        return 0

    df = pl.read_parquet(_get_schedule_path())

    # Create lookup set
    to_update = set(records)
    now = datetime.now()

    # Build update expressions
    def should_update(row):
        return (row["signal_id"], row["tier"], row["window_end"]) in to_update

    # More efficient: create a DataFrame of updates and join
    updates_df = pl.DataFrame({
        "signal_id": [r[0] for r in records],
        "tier": [r[1] for r in records],
        "window_end": [r[2] for r in records],
        "_new_status": [status] * len(records),
        "_new_computed_at": [now] * len(records),
    })

    # Left join and coalesce
    df = df.join(
        updates_df,
        on=["signal_id", "tier", "window_end"],
        how="left",
    ).with_columns([
        pl.coalesce(["_new_status", "status"]).alias("status"),
        pl.coalesce(["_new_computed_at", "computed_at"]).alias("computed_at"),
    ]).drop(["_new_status", "_new_computed_at"])

    df.write_parquet(_get_schedule_path())
    return len(records)


def mark_in_progress(
    signal_id: str,
    tier: str,
    window_end: date,
) -> bool:
    """
    Mark a window as in-progress (being computed).

    Call this BEFORE starting computation. If process crashes,
    we can detect orphaned in-progress windows.
    """
    return mark_computed(signal_id, tier, window_end, status="in_progress")


def mark_batch_in_progress(
    records: List[Tuple[str, str, date]],
) -> int:
    """Mark multiple windows as in-progress."""
    return mark_batch_computed(records, status="in_progress")


def get_orphaned_windows(stale_minutes: int = 60) -> pl.DataFrame:
    """
    Find windows that are stuck in 'in_progress' status.

    These are computations that started but never completed -
    likely due to a crash or interruption.

    Args:
        stale_minutes: Consider in_progress stale after this many minutes

    Returns:
        DataFrame of orphaned windows
    """
    if not _get_schedule_path().exists():
        return pl.DataFrame()

    df = pl.read_parquet(_get_schedule_path())
    df = df.filter(pl.col("status") == "in_progress")

    if len(df) == 0:
        return df

    # Filter by computed_at being stale
    cutoff = datetime.now() - timedelta(minutes=stale_minutes)
    df = df.filter(pl.col("computed_at") < cutoff)

    return df


def reset_orphaned_to_pending(stale_minutes: int = 60) -> int:
    """
    Reset orphaned in_progress windows back to pending.

    Call this at startup to recover from crashes.

    Returns:
        Number of windows reset
    """
    if not _get_schedule_path().exists():
        return 0

    df = pl.read_parquet(_get_schedule_path())
    cutoff = datetime.now() - timedelta(minutes=stale_minutes)

    # Find orphaned
    orphaned_mask = (
        (pl.col("status") == "in_progress") &
        (pl.col("computed_at") < cutoff)
    )

    count = df.filter(orphaned_mask).height

    if count > 0:
        df = df.with_columns([
            pl.when(orphaned_mask)
            .then(pl.lit("pending"))
            .otherwise(pl.col("status"))
            .alias("status")
        ])
        df.write_parquet(_get_schedule_path())
        logger.info(f"Reset {count} orphaned windows to pending")

    return count


def reconcile_with_results() -> Dict[str, int]:
    """
    Reconcile schedule with actual computed results.

    Marks windows as 'computed' if they exist in vector/signals.parquet
    but are still marked 'pending' in the schedule.

    Returns:
        Dict with counts of reconciled windows by tier
    """
    if not _get_schedule_path().exists():
        return {}

    vector_path = get_path(SIGNALS)
    if not vector_path.exists():
        return {}

    schedule = pl.read_parquet(_get_schedule_path())
    results = pl.read_parquet(vector_path)

    # Get unique computed windows from results
    computed = results.select([
        "signal_id",
        pl.col("obs_date").alias("window_end"),
        "target_obs",
    ]).unique()

    # Map target_obs to tier name
    stride_config = load_stride_config()
    obs_to_tier = {
        stride_config.get_window(name).window_days: name
        for name in stride_config.list_windows()
    }

    # Add tier to computed results using map_dict
    computed = computed.with_columns([
        pl.col("target_obs").replace_strict(obs_to_tier, default=None).alias("tier")
    ])

    # Find pending windows that are actually computed
    pending = schedule.filter(pl.col("status") == "pending")

    to_update = pending.join(
        computed.select(["signal_id", "window_end", "tier"]),
        on=["signal_id", "window_end", "tier"],
        how="inner",
    )

    if len(to_update) == 0:
        return {}

    # Update schedule
    update_keys = set(
        (r["signal_id"], r["tier"], r["window_end"])
        for r in to_update.iter_rows(named=True)
    )

    now = datetime.now()
    schedule = schedule.with_columns([
        pl.when(
            pl.struct(["signal_id", "tier", "window_end"]).map_elements(
                lambda x: (x["signal_id"], x["tier"], x["window_end"]) in update_keys,
                return_dtype=pl.Boolean
            )
        ).then(pl.lit("computed")).otherwise(pl.col("status")).alias("status"),
        pl.when(
            pl.struct(["signal_id", "tier", "window_end"]).map_elements(
                lambda x: (x["signal_id"], x["tier"], x["window_end"]) in update_keys,
                return_dtype=pl.Boolean
            )
        ).then(pl.lit(now)).otherwise(pl.col("computed_at")).alias("computed_at"),
    ])

    schedule.write_parquet(_get_schedule_path())

    # Count by tier
    counts = to_update.group_by("tier").len().to_dicts()
    return {r["tier"]: r["len"] for r in counts}


def get_missing_windows(tier: Optional[str] = None) -> pl.DataFrame:
    """
    Compare schedule against actual computed results to find missing windows.

    Returns windows that are marked 'computed' but have no data in vector/signals.parquet.
    """
    if not _get_schedule_path().exists():
        return pl.DataFrame()

    schedule = pl.read_parquet(_get_schedule_path())
    schedule = schedule.filter(pl.col("status") == "computed")

    if tier:
        schedule = schedule.filter(pl.col("tier") == tier)

    # Load actual results
    vector_path = get_path(SIGNALS)
    if not vector_path.exists():
        return schedule  # All are missing

    results = pl.read_parquet(vector_path)

    # Get unique (signal, window_end, target_obs) from results
    computed = results.select([
        "signal_id",
        "obs_date",  # This is window_end in results
        "target_obs",
    ]).unique()

    # Map tier to target_obs
    stride_config = load_stride_config()
    tier_to_obs = {
        name: stride_config.get_window(name).window_days
        for name in stride_config.list_windows()
    }

    # Add target_obs to schedule for joining
    schedule = schedule.with_columns([
        pl.col("tier").replace(tier_to_obs).alias("target_obs_check")
    ])

    # Find missing by anti-join
    missing = schedule.join(
        computed,
        left_on=["signal_id", "window_end", "target_obs_check"],
        right_on=["signal_id", "obs_date", "target_obs"],
        how="anti",
    )

    return missing.drop("target_obs_check")


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PRISM Window Schedule Management")
    parser.add_argument("--generate", action="store_true", help="Generate schedule for all signals")
    parser.add_argument("--force", action="store_true", help="Force regenerate (overwrite existing)")
    parser.add_argument("--stats", action="store_true", help="Show schedule statistics")
    parser.add_argument("--pending", action="store_true", help="Show pending windows")
    parser.add_argument("--tier", type=str, help="Filter by tier")
    parser.add_argument("--limit", type=int, default=20, help="Limit output rows")

    args = parser.parse_args()

    if args.generate:
        print("Generating window schedule...")
        counts = generate_schedule(force=args.force)
        print("\nWindows by tier:")
        for tier, count in counts.items():
            print(f"  {tier}: {count:,}")

    elif args.stats:
        print("=== WINDOW SCHEDULE STATS ===")
        stats = get_schedule_stats()
        if len(stats) == 0:
            print("No schedule found. Run --generate first.")
        else:
            print(stats)
            print()
            total = stats["count"].sum()
            pending = stats.filter(pl.col("status") == "pending")["count"].sum()
            computed = stats.filter(pl.col("status") == "computed")["count"].sum()
            print(f"Total: {total:,}")
            print(f"Pending: {pending:,}")
            print(f"Computed: {computed:,}")

    elif args.pending:
        pending = get_pending_windows(tier=args.tier, limit=args.limit)
        print(f"=== PENDING WINDOWS ({len(pending):,} shown) ===")
        print(pending)

    else:
        parser.print_help()
