"""
Orthon CLI - Command line interface.

Usage:
    orthon --version
    orthon --help
"""

import argparse
import sys

from orthon import __version__


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="orthon",
        description="Orthon - Behavioral Geometry Engine for Industrial Signal Analysis",
    )
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"orthon {__version__}",
    )
    parser.add_argument(
        "--list-engines",
        action="store_true",
        help="List all available engines",
    )

    args = parser.parse_args()

    if args.list_engines:
        from orthon import list_engines, list_vector_engines, list_geometry_engines, list_state_engines

        print("Vector Engines:")
        for name in list_vector_engines():
            print(f"  - {name}")

        print("\nGeometry Engines:")
        for name in list_geometry_engines():
            print(f"  - {name}")

        print("\nState Engines:")
        for name in list_state_engines():
            print(f"  - {name}")

        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
