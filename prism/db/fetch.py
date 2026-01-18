#!/usr/bin/env python3
"""
PRISM Fetch Module

Source-agnostic orchestrator for fetching raw observations to Parquet.

Usage:
    python -m prism.db.fetch --cmapss
    python -m prism.db.fetch --climate
    python -m prism.db.fetch fetchers/yaml/usgs.yaml

Fetchers are loaded dynamically from repo_root/fetchers/{source}_fetcher.py.
Results are written to data/raw/observations.parquet
"""

import argparse
import importlib.util
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import polars as pl
import yaml

from prism.db.parquet_store import ensure_directories, get_parquet_path
from prism.db.polars_io import upsert_parquet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# DEFAULT YAML PATHS (shortcuts)
# =============================================================================

DEFAULT_YAMLS = {
    "usgs": "fetchers/yaml/usgs.yaml",
    "climate": "fetchers/yaml/climate.yaml",
    "ecology": "fetchers/yaml/ecology.yaml",
    "delphi": "fetchers/yaml/delphi.yaml",
    "cmapss": "fetchers/yaml/cmapss.yaml",
    "tep": "fetchers/yaml/tep.yaml",
}


def find_repo_root() -> Path:
    """Find repository root by looking for fetchers/ directory."""
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "fetchers").exists():
            return parent
    return current


def resolve_yaml_path(yaml_arg: Optional[str], source_shortcut: Optional[str]) -> Path:
    """Resolve YAML path from argument or shortcut."""
    repo_root = find_repo_root()

    if yaml_arg:
        path = Path(yaml_arg)
        if not path.is_absolute():
            path = repo_root / path
        return path

    if source_shortcut:
        if source_shortcut not in DEFAULT_YAMLS:
            available = ", ".join(sorted(DEFAULT_YAMLS.keys()))
            raise ValueError(f"Unknown source: {source_shortcut}. Available: {available}")
        return repo_root / DEFAULT_YAMLS[source_shortcut]

    raise ValueError("Must specify YAML file or source shortcut (--cmapss, --climate, etc.)")


def load_fetcher(source: str) -> Callable:
    """
    Dynamically load a fetcher module and return its fetch function.

    Fetchers are expected at: repo_root/fetchers/{source}_fetcher.py
    """
    repo_root = find_repo_root()
    fetcher_path = repo_root / "fetchers" / f"{source}_fetcher.py"

    if not fetcher_path.exists():
        raise FileNotFoundError(f"Fetcher not found: {fetcher_path}")

    spec = importlib.util.spec_from_file_location(f"{source}_fetcher", fetcher_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{source}_fetcher"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "fetch"):
        raise AttributeError(f"Fetcher {source} must have a 'fetch(config)' function")

    return module.fetch


def fetch_to_parquet(
    yaml_path: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    signals: Optional[List[str]] = None,
) -> int:
    """
    Fetch data using config and write to Parquet.

    Args:
        yaml_path: Path to YAML config file
        start_date: Override start date
        end_date: Override end date
        signals: Override signal list

    Returns:
        Number of observations written
    """
    # Load config
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    source = config.get("source")
    if not source:
        raise ValueError("Config must specify 'source' field")

    # Apply overrides
    if start_date:
        config["start_date"] = start_date
    if end_date:
        config["end_date"] = end_date
    if signals:
        config["signals"] = signals

    logger.info(f"Fetching from {source}...")
    logger.info(f"Config: {yaml_path}")
    if "signals" in config:
        logger.info(f"Signals: {len(config['signals'])}")

    # Load fetcher and fetch data
    fetch_func = load_fetcher(source)
    observations = fetch_func(config)

    if not observations:
        logger.warning("No observations returned")
        return 0

    logger.info(f"Fetched {len(observations):,} observations")

    # Convert to Polars DataFrame
    df = pl.DataFrame(observations)

    # Normalize column names
    if "observed_at" in df.columns:
        df = df.rename({"observed_at": "obs_date"})

    # Ensure required columns
    required_cols = ["signal_id", "obs_date", "value"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Select and cast columns
    df = df.select(
        [
            pl.col("signal_id").cast(pl.Utf8),
            pl.col("obs_date").cast(pl.Date),
            pl.col("value").cast(pl.Float64),
        ]
    )

    # Get domain from config (defaults to active domain)
    domain = config.get("domain")

    # Ensure directories exist for this domain
    ensure_directories(domain)

    # Write to Parquet (upsert on signal_id + obs_date)
    target_path = get_parquet_path("raw", "observations", domain=domain)
    total_rows = upsert_parquet(df, target_path, key_cols=["signal_id", "obs_date"])

    logger.info(f"Wrote {total_rows:,} rows to {target_path}")

    return total_rows


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PRISM Fetch Runner - Fetch data to Parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("yaml_file", nargs="?", help="Path to YAML config file")

    # Source shortcuts
    parser.add_argument("--usgs", action="store_true", help="Fetch from USGS")
    parser.add_argument("--climate", action="store_true", help="Fetch climate data")
    parser.add_argument("--ecology", action="store_true", help="Fetch ecology data")
    parser.add_argument("--cmapss", action="store_true", help="Fetch NASA C-MAPSS turbofan data")
    parser.add_argument("--tep", action="store_true", help="Fetch Tennessee Eastman process data")

    # Options
    parser.add_argument("--start-date", type=str, help="Override start date")
    parser.add_argument("--end-date", type=str, help="Override end date")
    parser.add_argument("--signals", type=str, help="Comma-separated signal list")

    args = parser.parse_args()

    # Determine source shortcut
    source_shortcut = None
    for source in DEFAULT_YAMLS.keys():
        if getattr(args, source.replace("-", "_"), False):
            source_shortcut = source
            break

    try:
        yaml_path = resolve_yaml_path(args.yaml_file, source_shortcut)
    except ValueError as e:
        parser.error(str(e))

    # Parse signals if provided
    signals = None
    if args.signals:
        signals = [i.strip() for i in args.signals.split(",")]

    # Run fetch
    try:
        count = fetch_to_parquet(
            yaml_path=yaml_path,
            start_date=args.start_date,
            end_date=args.end_date,
            signals=signals,
        )
        print(f"\nFetched {count:,} observations to Parquet")
    except Exception as e:
        logger.error(f"Fetch failed: {e}")
        sys.exit(1)
