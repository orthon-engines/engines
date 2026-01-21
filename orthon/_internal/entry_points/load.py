#!/usr/bin/env python3
"""
PRISM Load Entry Point
======================

Load raw data files directly into observations.parquet using the universal Loader.

Supports CSV, TXT, and Parquet files with automatic schema detection.
Special handling for C-MAPSS format (auto-detected from filename).

Output Schema:
    entity_id   | String   | Engine, bearing, unit identifier
    signal_id   | String   | Sensor name
    timestamp   | Float64  | Time (cycles, seconds, etc.)
    value       | Float64  | Raw measurement

Usage:
    python -m prism.entry_points.load data/train_FD004.txt
    python -m prism.entry_points.load data/my_data.csv --entity-col unit_id --timestamp-col time

Results are written to data/loader/observations.parquet by default (isolated from main pipeline).
Use --output to specify a different location.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl
import yaml

from orthon._internal.core.loader import Loader
from orthon._internal.db.parquet_store import get_data_root
from orthon._internal.db.polars_io import write_parquet_atomic

# Default output directory for loader (isolated from main pipeline)
DEFAULT_OUTPUT_DIR = "loader"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_repo_root() -> Path:
    """Find repository root by looking for config/ directory."""
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "config").exists():
            return parent
    return current


def write_loader_metadata(output_dir: Path, source_path: str, row_count: int, loader: Loader):
    """Write loader metadata to the output directory."""
    metadata_path = output_dir / "metadata.yaml"

    # Extract domain name from filename
    source_name = Path(source_path).stem

    metadata = {
        "source": "loader",
        "source_file": source_path,
        "loaded_at": datetime.now().isoformat(),
        "row_count": row_count,
        "domain": source_name,
        "description": f"Loaded from {Path(source_path).name}",
        "schema": {
            "entity_col": loader.entity_col,
            "timestamp_col": loader.timestamp_col,
            "sensor_count": len(loader.sensor_cols),
            "op_count": len(loader._schema.get("op_cols", [])),
        },
    }

    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)

    logger.info(f"Wrote metadata to {metadata_path}")


def load_to_parquet(
    path: str,
    output_dir: Optional[str] = None,
    entity_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    sensor_cols: Optional[str] = None,
    schema_file: Optional[str] = None,
    entity_prefix: Optional[str] = None,
) -> tuple:
    """
    Load data file to observations.parquet.

    Args:
        path: Path to data file
        output_dir: Output directory (default: data/loader/)
        entity_col: Column name or index for entity ID
        timestamp_col: Column name or index for timestamp
        sensor_cols: Comma-separated list of sensor column names/indices
        schema_file: Path to schema YAML file
        entity_prefix: Prefix to add to entity IDs (e.g., "FD004")

    Returns:
        Tuple of (row_count, output_path)
    """
    path = Path(path)
    logger.info(f"Loading: {path}")

    # Determine output directory
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = get_data_root() / DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse sensor_cols if provided
    parsed_sensor_cols = None
    if sensor_cols:
        parsed_sensor_cols = [s.strip() for s in sensor_cols.split(",")]
        # Convert numeric strings to ints
        parsed_sensor_cols = [
            int(s) if s.isdigit() else s for s in parsed_sensor_cols
        ]

    # Parse entity_col and timestamp_col
    parsed_entity_col = None
    if entity_col:
        parsed_entity_col = int(entity_col) if entity_col.isdigit() else entity_col

    parsed_timestamp_col = None
    if timestamp_col:
        parsed_timestamp_col = int(timestamp_col) if timestamp_col.isdigit() else timestamp_col

    # Load with Loader
    loader = Loader.from_file(
        path,
        schema=schema_file,
        entity_col=parsed_entity_col,
        timestamp_col=parsed_timestamp_col,
        sensor_cols=parsed_sensor_cols,
    )

    # Print summary
    loader.summary()

    # Get observations and convert to PRISM schema
    obs = loader.observations

    # Rename 'signal' to 'signal_id' to match PRISM schema
    obs = obs.rename({"signal": "signal_id"})

    # Add entity prefix if specified (e.g., "FD004_1" instead of "1")
    if entity_prefix:
        obs = obs.with_columns(
            (pl.lit(entity_prefix + "_") + pl.col("entity_id")).alias("entity_id")
        )

    # Ensure correct column order and types
    obs = obs.select([
        pl.col("entity_id").cast(pl.Utf8),
        pl.col("signal_id").cast(pl.Utf8),
        pl.col("timestamp").cast(pl.Float64),
        pl.col("value").cast(pl.Float64),
    ])

    # Write to observations.parquet in output directory
    target_path = out_dir / "observations.parquet"
    write_parquet_atomic(obs, target_path)

    logger.info(f"Wrote {len(obs):,} observations to {target_path}")

    # Write metadata
    write_loader_metadata(out_dir, str(path), len(obs), loader)

    return len(obs), target_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PRISM Load - Load data files to observations.parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output: data/loader/observations.parquet (isolated from main pipeline)

Schema:
  entity_id   | String   | The thing that fails (engine, bearing, unit)
  signal_id   | String   | The measurement (sensor_1, temp, vibration)
  timestamp   | Float64  | Time (cycles, seconds, etc.)
  value       | Float64  | Raw measurement value

Examples:
  # Auto-detect C-MAPSS format
  python -m prism.entry_points.load data/train_FD004.txt

  # With entity prefix
  python -m prism.entry_points.load data/train_FD004.txt --prefix FD004

  # Explicit column mapping
  python -m prism.entry_points.load data/sensors.csv --entity-col machine_id --timestamp-col time

  # Use saved schema
  python -m prism.entry_points.load data/test.csv --schema my_schema.yaml

  # Output to specific directory
  python -m prism.entry_points.load data/train_FD004.txt --output data/my_test/
"""
    )

    parser.add_argument(
        "path",
        help="Path to data file (CSV, TXT, or Parquet)"
    )

    parser.add_argument(
        "--output", "-o",
        help="Output directory (default: data/loader/)"
    )

    parser.add_argument(
        "--entity-col",
        help="Column name or index for entity ID"
    )

    parser.add_argument(
        "--timestamp-col",
        help="Column name or index for timestamp"
    )

    parser.add_argument(
        "--sensor-cols",
        help="Comma-separated list of sensor column names or indices"
    )

    parser.add_argument(
        "--schema",
        help="Path to schema YAML file"
    )

    parser.add_argument(
        "--prefix",
        help="Prefix to add to entity IDs (e.g., 'FD004' -> 'FD004_1')"
    )

    args = parser.parse_args()

    # Validate path
    if not Path(args.path).exists():
        logger.error(f"File not found: {args.path}")
        sys.exit(1)

    # Run load
    try:
        count, output_path = load_to_parquet(
            args.path,
            output_dir=args.output,
            entity_col=args.entity_col,
            timestamp_col=args.timestamp_col,
            sensor_cols=args.sensor_cols,
            schema_file=args.schema,
            entity_prefix=args.prefix,
        )
        print(f"\nLoaded {count:,} observations to {output_path}")
    except Exception as e:
        logger.error(f"Load failed: {e}")
        raise


if __name__ == "__main__":
    main()
