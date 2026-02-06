# prism/entry_points/run_pipeline.py
"""
PRISM Pipeline Runner

Orchestrates stages 01-05 (core) with optional extended analysis.

Usage:
    python -m prism.entry_points.run_pipeline manifest.yaml
    python -m prism.entry_points.run_pipeline manifest.yaml --stages 01,02,03
    python -m prism.entry_points.run_pipeline manifest.yaml --all
"""

import argparse
from pathlib import Path


def run(manifest_path: str, stages: list = None, verbose: bool = True):
    """Run pipeline stages."""
    raise NotImplementedError("Pipeline runner not yet implemented. Run stages individually.")


def main():
    parser = argparse.ArgumentParser(description="PRISM Pipeline Runner")
    parser.add_argument('manifest', help='Path to manifest.yaml')
    parser.add_argument('--stages', help='Comma-separated stage numbers (e.g., 01,02,03)')
    parser.add_argument('--all', action='store_true', help='Run all stages')
    parser.add_argument('-q', '--quiet', action='store_true')

    args = parser.parse_args()

    stages = args.stages.split(',') if args.stages else None
    run(args.manifest, stages=stages, verbose=not args.quiet)


if __name__ == '__main__':
    main()
