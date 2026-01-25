"""
PRISM Pipeline Runner (CLI wrapper)

Simple CLI interface for prism.conditional module.

Usage:
    python -m prism.entry_points.run --config config.yaml
    python -m prism.entry_points.run --config config.yaml --stage vector
    python -m prism.entry_points.run --config config.yaml --force
    python -m prism.entry_points.run --list
"""

import argparse
import logging
import yaml

from prism.utils.conditional import run, STAGES, print_stage_info


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description="PRISM Pipeline Runner")
    parser.add_argument("--config", help="Path to config YAML")
    parser.add_argument("--stage", help="Run specific stage only")
    parser.add_argument("--force", action="store_true", help="Force recompute")
    parser.add_argument("--list", action="store_true", help="List stages")

    args = parser.parse_args()

    if args.list:
        print_stage_info()
        return

    if not args.config:
        parser.error("--config is required unless using --list")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Run pipeline
    stages = [args.stage] if args.stage else None

    print("\nPRISM Pipeline")
    print("=" * 60)

    for result in run(config, force=args.force, stages=stages):
        status = "[OK]" if result.success else "[FAIL]"
        print(f"  {status} {result.stage:12} -> {result.output_file}.parquet")
        if result.error:
            print(f"       Error: {result.error}")
        if result.n_rows > 0:
            print(f"       {result.n_rows} rows, {result.n_cols} columns")

    print("=" * 60)


if __name__ == "__main__":
    main()
