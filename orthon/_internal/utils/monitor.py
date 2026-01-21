"""
PRISM Run Monitor

Monitor parquet file growth and progress during computation runs.

Usage:
    # From command line
    python -m prism.utils.monitor

    # Or in Python
    from orthon._internal.utils.monitor import monitor_progress
    monitor_progress()
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

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


def get_file_stats(path: Path) -> dict:
    """Get file statistics."""
    if not path.exists():
        return {"exists": False, "size_mb": 0, "rows": 0}

    size_mb = path.stat().st_size / (1024 * 1024)
    try:
        # Use lazy scan to get row count without loading full file
        rows = pl.scan_parquet(path).select(pl.len()).collect().item()
    except Exception:
        rows = 0

    return {"exists": True, "size_mb": size_mb, "rows": rows}


def get_progress_stats(schema: str, table: str) -> dict:
    """Get progress tracker statistics."""
    progress_path = get_data_root() / f"progress_{schema}_{table}.parquet"
    if not progress_path.exists():
        return {"completed": 0, "in_progress": 0, "failed": 0}

    try:
        df = pl.read_parquet(progress_path)
        df = df.filter(
            (pl.col("schema") == schema) & (pl.col("table") == table)
        )
        return {
            "completed": len(df.filter(pl.col("status") == "completed")),
            "in_progress": len(df.filter(pl.col("status") == "in_progress")),
            "failed": len(df.filter(pl.col("status") == "failed")),
        }
    except Exception:
        return {"completed": 0, "in_progress": 0, "failed": 0}


def monitor_once() -> dict:
    """Get current status of all PRISM data files."""
    results = {}

    # Signals
    signals_path = get_path(SIGNALS)
    signals_stats = get_file_stats(signals_path)
    signals_progress = get_progress_stats("vector", "signals")
    results["signals"] = {**signals_stats, **signals_progress}

    # Geometry
    geometry_path = get_path(GEOMETRY)
    results["geometry"] = get_file_stats(geometry_path)

    # State
    state_path = get_path(STATE)
    results["state"] = get_file_stats(state_path)

    # Cohorts
    cohorts_path = get_path(COHORTS)
    results["cohorts"] = get_file_stats(cohorts_path)

    # Observations
    obs_path = get_path(OBSERVATIONS)
    results["observations"] = get_file_stats(obs_path)

    return results


def print_status():
    """Print current status."""
    status = monitor_once()
    print(f"\n{'='*60}")
    print(f"PRISM Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    for name, stats in status.items():
        if stats.get("exists", False):
            progress_str = ""
            if "completed" in stats:
                progress_str = f" | completed={stats['completed']}, in_progress={stats['in_progress']}, failed={stats['failed']}"
            print(f"{name}: {stats['size_mb']:.1f} MB, {stats['rows']:,} rows{progress_str}")
        else:
            print(f"{name}: (not created)")

    print(f"{'='*60}\n")


def monitor_progress(interval: int = 30, duration: Optional[int] = None):
    """
    Continuously monitor progress.

    Args:
        interval: Seconds between updates
        duration: Total duration in seconds (None = run forever)
    """
    start = time.time()
    prev_stats = {}

    while True:
        current = monitor_once()

        # Calculate deltas
        print(f"\n{'='*60}")
        print(f"PRISM Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        for name, stats in current.items():
            if not stats.get("exists", False):
                continue

            delta_rows = 0
            delta_mb = 0
            if name in prev_stats:
                delta_rows = stats["rows"] - prev_stats[name].get("rows", 0)
                delta_mb = stats["size_mb"] - prev_stats[name].get("size_mb", 0)

            progress_str = ""
            if "completed" in stats:
                progress_str = f" | {stats['completed']} done, {stats['in_progress']} active"

            delta_str = ""
            if delta_rows > 0:
                delta_str = f" (+{delta_rows:,} rows, +{delta_mb:.1f} MB)"

            print(f"{name}: {stats['size_mb']:.1f} MB, {stats['rows']:,} rows{delta_str}{progress_str}")

        prev_stats = current.copy()

        if duration and (time.time() - start) >= duration:
            break

        time.sleep(interval)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor PRISM computation progress")
    parser.add_argument("--interval", type=int, default=30, help="Update interval in seconds")
    parser.add_argument("--once", action="store_true", help="Print status once and exit")
    args = parser.parse_args()

    if args.once:
        print_status()
    else:
        print("Monitoring PRISM progress. Press Ctrl+C to stop.")
        try:
            monitor_progress(interval=args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
