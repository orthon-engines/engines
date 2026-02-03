"""
PRISM Command Line Interface

Usage:
    python -m prism <command> [args]

Commands:
    validate    Check prerequisites and validate input files
    signal      Compute signal vector from manifest
    status      Show pipeline status

Examples:
    python -m prism validate /path/to/data
    python -m prism signal /path/to/manifest.yaml
    python -m prism status /path/to/data
"""

import argparse
import sys
from pathlib import Path


def cmd_validate(args):
    """Validate prerequisites and input data."""
    from prism.validation import (
        check_prerequisites,
        validate_input,
        PrerequisiteError,
        ValidationError,
    )

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        return 1

    print(f"Validating: {data_dir}")
    print()

    errors = []

    # Check prerequisites for signal_vector stage
    try:
        result = check_prerequisites(
            'signal_vector',
            str(data_dir),
            raise_on_missing=False,
            verbose=True,
        )
        if not result['satisfied']:
            errors.append(f"Missing prerequisites: {result['missing']}")
    except Exception as e:
        errors.append(f"Prerequisite check failed: {e}")

    print()

    # Validate input data (if prerequisites present)
    if not errors or args.force:
        try:
            report = validate_input(
                str(data_dir),
                raise_on_error=False,
                verbose=True,
            )
            if not report.valid:
                errors.extend(report.errors)
        except Exception as e:
            errors.append(f"Input validation failed: {e}")

    # Summary
    if errors:
        print("\nValidation FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    else:
        print("\nValidation PASSED")
        return 0


def cmd_status(args):
    """Show pipeline status."""
    from prism.validation.prerequisites import print_pipeline_status

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        return 1

    print_pipeline_status(str(data_dir))
    return 0


def cmd_signal(args):
    """Compute signal vector."""
    from prism.entry_points.signal_vector import run_from_manifest

    manifest_path = Path(args.manifest)

    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        return 1

    try:
        run_from_manifest(
            str(manifest_path),
            verbose=not args.quiet,
        )
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


def main():
    """PRISM CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='prism',
        description='PRISM Signal Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m prism validate /path/to/data
    python -m prism signal /path/to/manifest.yaml
    python -m prism status /path/to/data
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # validate command
    validate_parser = subparsers.add_parser(
        'validate',
        help='Check prerequisites and validate input files',
    )
    validate_parser.add_argument(
        'data_dir',
        help='Directory containing pipeline files',
    )
    validate_parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Continue validation even if prerequisites missing',
    )

    # status command
    status_parser = subparsers.add_parser(
        'status',
        help='Show pipeline status',
    )
    status_parser.add_argument(
        'data_dir',
        help='Directory containing pipeline files',
    )

    # signal command
    signal_parser = subparsers.add_parser(
        'signal',
        help='Compute signal vector from manifest',
    )
    signal_parser.add_argument(
        'manifest',
        help='Path to manifest.yaml',
    )
    signal_parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output',
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # Dispatch to command handler
    handlers = {
        'validate': cmd_validate,
        'status': cmd_status,
        'signal': cmd_signal,
    }

    handler = handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
