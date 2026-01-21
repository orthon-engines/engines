#!/usr/bin/env python3
"""
PRISM Fetch Entry Point
=======================

Fetch raw observations to observations.parquet.

Output Schema:
    entity_id   | String   | Engine, bearing, unit identifier
    signal_id   | String   | Sensor name
    timestamp   | Float64  | Time (cycles, seconds, etc.)
    value       | Float64  | Raw measurement

Usage:
    python -m prism.entry_points.fetch           # Interactive picker
    python -m prism.entry_points.fetch cmapss    # Direct source name

Results are written to data/observations.parquet
"""

import importlib.util
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import polars as pl
import yaml

from orthon._internal.db.parquet_store import get_path, ensure_directory, OBSERVATIONS
from orthon._internal.db.polars_io import upsert_parquet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_repo_root() -> Path:
    """Find repository root by looking for fetchers/ directory."""
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "fetchers").exists():
            return parent
    return current


def list_yaml_configs() -> List[Path]:
    """List all available YAML configs in fetchers/yaml/."""
    repo_root = find_repo_root()
    yaml_dir = repo_root / "fetchers" / "yaml"
    if not yaml_dir.exists():
        return []
    return sorted([p for p in yaml_dir.glob("*.yaml") if not p.name.startswith(".")])


def interactive_picker() -> Optional[Path]:
    """Interactive picker for YAML config files."""
    configs = list_yaml_configs()

    if not configs:
        logger.error("No YAML configs found in fetchers/yaml/")
        return None

    print("\nAvailable data sources:")
    print("-" * 40)
    for i, config in enumerate(configs, 1):
        # Load config to get description
        with open(config) as f:
            cfg = yaml.safe_load(f)
        desc = cfg.get("description", cfg.get("source", ""))
        print(f"  [{i}] {config.stem:<20} {desc}")
    print()

    while True:
        try:
            choice = input("Select source (number or name): ").strip()

            # Try as number
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(configs):
                    return configs[idx]
                print(f"Invalid number. Enter 1-{len(configs)}")
                continue

            # Try as name match
            for config in configs:
                if choice.lower() in config.stem.lower():
                    return config

            print(f"No match found for '{choice}'")

        except (KeyboardInterrupt, EOFError):
            print("\nCancelled")
            return None


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


def fetch_to_parquet(yaml_path: Path) -> int:
    """
    Fetch data using config and write to observations.parquet.

    Args:
        yaml_path: Path to YAML config file

    Returns:
        Number of observations written
    """
    # Load config
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    source = config.get("source")
    if not source:
        raise ValueError("Config must specify 'source' field")

    logger.info(f"Fetching from {source}...")
    logger.info(f"Config: {yaml_path}")

    # Load fetcher and fetch data
    fetch_func = load_fetcher(source)
    observations = fetch_func(config)

    if not observations:
        logger.warning("No observations returned")
        return 0

    logger.info(f"Fetched {len(observations):,} observations")

    # Convert to Polars DataFrame
    df = pl.DataFrame(observations)

    # Normalize column names to new schema
    column_mappings = {
        "unit_id": "entity_id",
        "engine_id": "entity_id",
        "bearing_id": "entity_id",
        "run_id": "entity_id",
        "obs_date": "timestamp",
        "observed_at": "timestamp",
        "time": "timestamp",
        "cycle": "timestamp",
        "t": "timestamp",
        "sensor_id": "signal_id",
        "indicator_id": "signal_id",
    }

    for old_name, new_name in column_mappings.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename({old_name: new_name})

    # If no entity_id, try to infer from signal_id or use default
    if "entity_id" not in df.columns:
        default_entity = config.get("entity_id", config.get("domain", "unit_1"))
        df = df.with_columns(pl.lit(default_entity).alias("entity_id"))
        logger.info(f"No entity_id found, using default: {default_entity}")

    # Ensure required columns
    required_cols = ["entity_id", "signal_id", "timestamp", "value"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {df.columns}")

    # Select and cast columns to final schema
    df = df.select([
        pl.col("entity_id").cast(pl.Utf8),
        pl.col("signal_id").cast(pl.Utf8),
        pl.col("timestamp").cast(pl.Float64),
        pl.col("value").cast(pl.Float64),
    ])

    # Ensure directory exists
    ensure_directory()

    # Write to observations.parquet (upsert on entity_id + signal_id + timestamp)
    target_path = get_path(OBSERVATIONS)
    total_rows = upsert_parquet(
        df,
        target_path,
        key_cols=["entity_id", "signal_id", "timestamp"]
    )

    logger.info(f"Wrote {total_rows:,} rows to {target_path}")

    # Write domain metadata to config/domain.yaml
    write_domain_config(config, yaml_path, total_rows)

    return total_rows


def write_domain_config(config: dict, yaml_path: Path, row_count: int):
    """Write domain metadata after successful fetch."""
    repo_root = find_repo_root()
    domain_yaml = repo_root / "config" / "domain.yaml"

    metadata = {
        "source": config.get("source"),
        "source_config": str(yaml_path),
        "fetched_at": datetime.now().isoformat(),
        "row_count": row_count,
        "domain": config.get("domain", config.get("source")),
        "description": config.get("description", ""),
    }

    with open(domain_yaml, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)

    logger.info(f"Wrote domain metadata to {domain_yaml}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PRISM Fetch - Raw observations to observations.parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output: data/observations.parquet

Schema:
  entity_id   | String   | The thing that fails (engine, bearing, unit)
  signal_id   | String   | The measurement (sensor_1, temp, vibration)
  timestamp   | Float64  | Time (cycles, seconds, etc.)
  value       | Float64  | Raw measurement value

Examples:
  python -m prism.entry_points.fetch              # Interactive picker
  python -m prism.entry_points.fetch cmapss       # Fetch C-MAPSS data
  python -m prism.entry_points.fetch tep          # Fetch TEP data
"""
    )

    parser.add_argument(
        "source",
        nargs="?",
        help="Source name (matches yaml filename) or path to yaml config"
    )

    args = parser.parse_args()

    # Determine yaml path
    yaml_path = None

    if args.source:
        # Check if it's a path
        path = Path(args.source)
        if path.exists():
            yaml_path = path
        else:
            # Search for matching yaml
            configs = list_yaml_configs()
            for config in configs:
                if args.source.lower() in config.stem.lower():
                    yaml_path = config
                    break

            if not yaml_path:
                logger.error(f"No config found matching '{args.source}'")
                print("\nAvailable sources:")
                for c in configs:
                    print(f"  - {c.stem}")
                sys.exit(1)
    else:
        # Interactive picker
        yaml_path = interactive_picker()
        if not yaml_path:
            sys.exit(0)

    # Run fetch
    try:
        count = fetch_to_parquet(yaml_path)
        print(f"\n✓ Fetched {count:,} observations to data/observations.parquet")
    except Exception as e:
        logger.error(f"Fetch failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
