"""
ENGINES Command Line Interface

Usage:
    python -m engines <command> [args]

Commands:
    validate    Check prerequisites and validate input files
    signal      Compute signal vector from manifest
    status      Show pipeline status

Examples:
    python -m engines validate /path/to/data
    python -m engines signal /path/to/manifest.yaml
    python -m engines status /path/to/data

SafeCLI:
    Standardized argument parsing for entry points with safety checks:
    1. Named arguments (no positional ambiguity)
    2. Input file validation (must exist)
    3. Output file protection (can't overwrite inputs)
    4. Overwrite confirmation for non-default outputs
    5. Clear help text with INPUT/OUTPUT labels

    Usage in entry points:
        from engines.cli import SafeCLI

        cli = SafeCLI("State Geometry Engine")
        cli.add_input('signal_vector', '-s', help='signal_vector.parquet')
        cli.add_input('state_vector', '-t', help='state_vector.parquet')
        cli.add_output('output', default='state_geometry.parquet')
        args = cli.parse()
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Set


# ============================================================
# SAFE CLI FOR ENTRY POINTS
# ============================================================

class SafeCLI:
    """
    Safe command-line interface with input/output validation.

    Prevents accidental data destruction by:
    - Validating input files exist
    - Preventing output from overwriting inputs
    - Confirming overwrites of existing files
    """

    def __init__(self, description: str, allow_overwrite: bool = False):
        """
        Initialize CLI parser.

        Args:
            description: Program description for --help
            allow_overwrite: If True, skip overwrite confirmation (for scripts)
        """
        self.parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.defaults: dict = {}
        self.allow_overwrite = allow_overwrite

        # Add global flags
        self.parser.add_argument(
            '-y', '--yes',
            action='store_true',
            help='Skip confirmation prompts (for automated scripts)'
        )
        self.parser.add_argument(
            '-v', '--verbose',
            action='store_true',
            default=True,
            help='Verbose output (default: True)'
        )
        self.parser.add_argument(
            '-q', '--quiet',
            action='store_true',
            help='Suppress output'
        )

    def add_input(
        self,
        name: str,
        flag: Optional[str] = None,
        help: str = '',
        required: bool = True
    ):
        """
        Add an input file argument.

        Args:
            name: Argument name (e.g., 'signal_vector')
            flag: Optional short flag (e.g., '-s')
            help: Help text
            required: Whether argument is required
        """
        self.inputs.append(name)

        # Convert to flag format
        flag_name = f"--{name.replace('_', '-')}"
        flags = [flag, flag_name] if flag else [flag_name]

        self.parser.add_argument(
            *flags,
            required=required,
            metavar='FILE',
            help=f'[INPUT] {help}'
        )

    def add_output(
        self,
        name: str = 'output',
        default: str = 'output.parquet',
        help: str = ''
    ):
        """
        Add an output file argument.

        Args:
            name: Argument name
            default: Default output filename
            help: Help text (auto-generated if empty)
        """
        self.outputs.append(name)
        self.defaults[name] = default

        flag_name = f"--{name.replace('_', '-')}"

        if not help:
            help = f'Output path (default: {default})'

        self.parser.add_argument(
            '-o' if name == 'output' else flag_name,
            f'--{name.replace("_", "-")}' if name != 'output' else '--output',
            default=default,
            metavar='FILE',
            help=f'[OUTPUT] {help}'
        )

    def add_flag(self, name: str, help: str = '', short: Optional[str] = None):
        """Add a boolean flag."""
        flags = [f'--{name.replace("_", "-")}']
        if short:
            flags.insert(0, short)
        self.parser.add_argument(*flags, action='store_true', help=help)

    def add_option(
        self,
        name: str,
        default=None,
        type=str,
        help: str = '',
        choices: Optional[List] = None
    ):
        """Add an option with a value."""
        self.parser.add_argument(
            f'--{name.replace("_", "-")}',
            default=default,
            type=type,
            choices=choices,
            help=help
        )

    def parse(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """
        Parse arguments with safety validation.

        Args:
            args: Arguments to parse (default: sys.argv)

        Returns:
            Parsed arguments namespace

        Raises:
            SystemExit: On validation failure
        """
        parsed = self.parser.parse_args(args)

        # Handle quiet flag
        if parsed.quiet:
            parsed.verbose = False

        # Collect input paths
        input_paths: Set[str] = set()
        for input_name in self.inputs:
            path = getattr(parsed, input_name, None)
            if path:
                # Resolve to absolute path for comparison
                abs_path = str(Path(path).resolve())
                input_paths.add(abs_path)

                # Validate input exists
                if not Path(path).exists():
                    self._error(f"Input file not found: {path}")

        # Validate outputs
        for output_name in self.outputs:
            path = getattr(parsed, output_name, None)
            if path:
                abs_path = str(Path(path).resolve())

                # Check: output can't be an input
                if abs_path in input_paths:
                    self._error(
                        f"Output '{path}' matches an input file!\n"
                        f"       This would destroy your input data.\n"
                        f"       Use -o/--output to specify a different output path."
                    )

                # Check: warn before overwriting existing non-default file
                default = self.defaults.get(output_name)
                if (
                    Path(path).exists()
                    and path != default
                    and not self.allow_overwrite
                    and not parsed.yes
                ):
                    self._confirm_overwrite(path)

        return parsed

    def _error(self, message: str):
        """Print error and exit."""
        print(f"\n❌ ERROR: {message}", file=sys.stderr)
        sys.exit(1)

    def _confirm_overwrite(self, path: str):
        """Ask user to confirm overwrite."""
        print(f"\n⚠️  WARNING: Output file '{path}' already exists.")
        try:
            response = input("   Overwrite? [y/N]: ")
            if response.lower() != 'y':
                print("   Aborted.")
                sys.exit(0)
        except EOFError:
            # Non-interactive mode
            self._error(
                f"Output file '{path}' exists and running non-interactively.\n"
                f"       Use -y/--yes to overwrite, or choose a different output path."
            )


# ============================================================
# ENGINES MAIN CLI
# ============================================================


def cmd_validate(args):
    """Validate prerequisites and input data."""
    from engines.validation import (
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
    from engines.validation.prerequisites import print_pipeline_status

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        return 1

    print_pipeline_status(str(data_dir))
    return 0


def cmd_signal(args):
    """Compute signal vector."""
    from engines.entry_points.signal_vector import run_from_manifest

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


# ============================================================
# ATLAS STAGE COMMANDS (stages 16-23)
# ============================================================

def cmd_atlas_stage(args):
    """Run a single atlas stage."""
    import importlib
    import yaml

    data_dir = Path(args.data_dir)
    output_dir = data_dir / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = data_dir / 'manifest.yaml'
    manifest = None
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

    obs_path = data_dir / 'observations.parquet'
    verbose = not getattr(args, 'quiet', False)

    stage_map = {
        'break-sequence': ('stage_16_break_sequence', lambda m: {
            'args': [str(output_dir / 'breaks.parquet'), str(output_dir / 'break_sequence.parquet')],
            'kwargs': {
                'reference_index': (m or {}).get('intervention', {}).get('event_index') if m else None,
                'verbose': verbose,
            },
        }),
        'ftle-backward': ('stage_17_ftle_backward', lambda m: {
            'args': [str(obs_path), str(output_dir / 'ftle_backward.parquet')],
            'kwargs': {
                'verbose': verbose,
                'intervention': (m or {}).get('intervention'),
                'direction': 'backward',
            },
        }),
        'segment-comparison': ('stage_18_segment_comparison', lambda m: {
            'args': [str(obs_path), str(output_dir / 'segment_comparison.parquet')],
            'kwargs': {
                'segments': _get_segments(m),
                'verbose': verbose,
            },
        }),
        'info-flow-delta': ('stage_19_info_flow_delta', lambda m: {
            'args': [str(obs_path), str(output_dir / 'info_flow_delta.parquet')],
            'kwargs': {
                'segments': _get_segments(m),
                'verbose': verbose,
            },
        }),
        'geometry-full': ('stage_20_geometry_full', lambda m: {
            'args': [str(obs_path), str(output_dir / 'geometry_full.parquet')],
            'kwargs': {'verbose': verbose},
        }),
        'velocity-field': ('stage_21_velocity_field', lambda m: {
            'args': [str(obs_path), str(output_dir / 'velocity_field.parquet')],
            'kwargs': {'verbose': verbose},
        }),
        'ftle-rolling': ('stage_22_ftle_rolling', lambda m: {
            'args': [str(obs_path), str(output_dir / 'ftle_rolling.parquet')],
            'kwargs': {'verbose': verbose},
        }),
        'ridge-proximity': ('stage_23_ridge_proximity', lambda m: {
            'args': [
                str(output_dir / 'ftle_rolling.parquet'),
                str(output_dir / 'velocity_field.parquet'),
                str(output_dir / 'ridge_proximity.parquet'),
            ],
            'kwargs': {'verbose': verbose},
        }),
    }

    stage_name, build_call = stage_map[args.stage_command]
    call = build_call(manifest)

    try:
        module = importlib.import_module(f'engines.entry_points.{stage_name}')
        module.run(*call['args'], **call['kwargs'])
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


def cmd_atlas(args):
    """Run the full atlas pipeline in dependency order."""
    import importlib
    import yaml

    data_dir = Path(args.data_dir)
    output_dir = data_dir / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = data_dir / 'manifest.yaml'
    manifest = None
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

    obs_path = data_dir / 'observations.parquet'
    verbose = not args.quiet

    # Atlas stages in dependency order
    stages = [
        ('stage_16_break_sequence', 'break_sequence.parquet', lambda: {
            'args': [str(output_dir / 'breaks.parquet'), str(output_dir / 'break_sequence.parquet')],
            'kwargs': {
                'reference_index': (manifest or {}).get('intervention', {}).get('event_index') if manifest else None,
                'verbose': verbose,
            },
            'requires': ['breaks.parquet'],
        }),
        ('stage_17_ftle_backward', 'ftle_backward.parquet', lambda: {
            'args': [str(obs_path), str(output_dir / 'ftle_backward.parquet')],
            'kwargs': {
                'verbose': verbose,
                'intervention': (manifest or {}).get('intervention'),
                'direction': 'backward',
            },
            'requires': ['observations.parquet'],
        }),
        ('stage_22_ftle_rolling', 'ftle_rolling.parquet', lambda: {
            'args': [str(obs_path), str(output_dir / 'ftle_rolling.parquet')],
            'kwargs': {'verbose': verbose},
            'requires': ['observations.parquet'],
        }),
        ('stage_21_velocity_field', 'velocity_field.parquet', lambda: {
            'args': [str(obs_path), str(output_dir / 'velocity_field.parquet')],
            'kwargs': {'verbose': verbose},
            'requires': ['observations.parquet'],
        }),
        ('stage_20_geometry_full', 'geometry_full.parquet', lambda: {
            'args': [str(obs_path), str(output_dir / 'geometry_full.parquet')],
            'kwargs': {'verbose': verbose},
            'requires': ['observations.parquet'],
        }),
        ('stage_23_ridge_proximity', 'ridge_proximity.parquet', lambda: {
            'args': [
                str(output_dir / 'ftle_rolling.parquet'),
                str(output_dir / 'velocity_field.parquet'),
                str(output_dir / 'ridge_proximity.parquet'),
            ],
            'kwargs': {'verbose': verbose},
            'requires': ['ftle_rolling.parquet', 'velocity_field.parquet'],
        }),
        ('stage_18_segment_comparison', 'segment_comparison.parquet', lambda: {
            'args': [str(obs_path), str(output_dir / 'segment_comparison.parquet')],
            'kwargs': {
                'segments': _get_segments(manifest),
                'verbose': verbose,
            },
            'requires': ['observations.parquet'],
        }),
        ('stage_19_info_flow_delta', 'info_flow_delta.parquet', lambda: {
            'args': [str(obs_path), str(output_dir / 'info_flow_delta.parquet')],
            'kwargs': {
                'segments': _get_segments(manifest),
                'verbose': verbose,
            },
            'requires': ['observations.parquet'],
        }),
    ]

    if verbose:
        print("=" * 60)
        print("DYNAMICAL ATLAS")
        print("=" * 60)
        print(f"Data:   {data_dir}")
        print(f"Output: {output_dir}")
        print()

    completed = 0
    skipped = 0

    for stage_name, output_file, build_call in stages:
        call = build_call()

        # Check prerequisites
        missing = [r for r in call['requires'] if not (
            (data_dir / r).exists() or (output_dir / r).exists()
        )]
        if missing:
            if verbose:
                print(f"  SKIP {stage_name}: missing {', '.join(missing)}")
            skipped += 1
            continue

        if verbose:
            print(f"  {stage_name}...")

        try:
            module = importlib.import_module(f'engines.entry_points.{stage_name}')
            module.run(*call['args'], **call['kwargs'])
            completed += 1
        except Exception as e:
            print(f"  ERROR in {stage_name}: {e}")
            if not args.continue_on_error:
                return 1

    if verbose:
        print()
        print("=" * 60)
        print(f"ATLAS COMPLETE: {completed} stages, {skipped} skipped")
        print("=" * 60)

    return 0


def _get_segments(manifest):
    """Extract segments from manifest intervention config."""
    if not manifest:
        return None
    segments = manifest.get('segments')
    if segments:
        return segments
    intervention = manifest.get('intervention')
    if intervention and intervention.get('enabled'):
        event_idx = intervention.get('event_index', 20)
        return [
            {'name': 'pre', 'range': [0, event_idx - 1]},
            {'name': 'post', 'range': [event_idx, None]},
        ]
    return None


def cmd_run(args):
    """Run ENGINES pipeline on input data (user-facing entry point)."""
    from engines.input_loader import load_input, detect_data_characteristics, generate_auto_manifest
    import yaml

    input_path = Path(args.input_path)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input data
    print(f"Loading {input_path}...")
    observations = load_input(input_path)

    # Analyze data
    chars = detect_data_characteristics(observations)
    print(f"  {chars['n_signals']} signals, {chars['n_cohorts']} cohorts, "
          f"{chars['min_samples']} samples/signal")

    if chars['constant_signals']:
        print(f"  {len(chars['constant_signals'])} constant signals will be skipped")

    if chars['nan_count'] > 0:
        print(f"  {chars['nan_count']} NaN values found — rows removed")

    # Load or generate manifest
    if args.manifest:
        manifest_path = Path(args.manifest)
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        print(f"  Manifest: {manifest_path}")
    else:
        # Parse segment strings
        segments = None
        if args.segments:
            segments = []
            for seg_str in args.segments:
                parts = seg_str.split(':')
                if len(parts) != 3:
                    print(f"  Error: segment format is name:start:end (got '{seg_str}')")
                    return 1
                name = parts[0]
                start = int(parts[1])
                end = int(parts[2]) if parts[2] != '' else None
                segments.append({'name': name, 'range': [start, end]})

        manifest = generate_auto_manifest(chars, atlas=args.atlas, segments=segments)
        print(f"  Manifest: auto-generated (window={manifest['system']['window']}, "
              f"stride={manifest['system']['stride']})")

    # Warnings
    if chars['min_samples'] < 50:
        print(f"  Warning: Very short signals ({chars['min_samples']} samples) — limited analysis")
    if not chars['ftle_viable']:
        print(f"  Warning: FTLE requires >=200 samples (have {chars['min_samples']})")
    elif args.atlas and not chars['rolling_ftle_viable']:
        print(f"  Warning: Rolling FTLE requires >=400 samples (have {chars['min_samples']})")

    # Save observations.parquet to output directory
    obs_output = output_dir / 'observations.parquet'
    observations.write_parquet(obs_output)

    # Generate minimal typology.parquet (required by state_vector stage)
    import polars as pl
    typology_rows = []
    for sig_id in chars['signal_list']:
        is_const = sig_id in set(chars['constant_signals'])
        typology_rows.append({
            'signal_id': sig_id,
            'temporal_pattern': 'CONSTANT' if is_const else 'STATIONARY',
            'spectral': 'NONE' if is_const else 'BROADBAND',
            'stationarity': 'STATIONARY',
            'continuity': 'CONTINUOUS',
            'complexity': 'MEDIUM',
            'is_constant': is_const,
        })
    typology_df = pl.DataFrame(typology_rows)
    typology_output = output_dir / 'typology.parquet'
    typology_df.write_parquet(typology_output)

    # Save manifest
    manifest_output = output_dir / 'manifest.yaml'
    manifest['paths'] = {
        'observations': 'observations.parquet',
        'typology': 'typology.parquet',
        'output_dir': 'output/',
    }
    with open(manifest_output, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False)

    # Determine stages to run
    stage_list = None
    if args.atlas:
        # Core + atlas stages
        stage_list = [f'{i:02d}' for i in range(15)] + [f'{i:02d}' for i in range(16, 24)]
    # else: default (core stages only)

    # Run pipeline
    print(f"\nRunning pipeline...")
    from engines.entry_points.run_pipeline import run as run_pipeline
    run_pipeline(str(manifest_output), stages=stage_list, verbose=not args.quiet)

    # Output summary
    _print_summary(output_dir / 'output', manifest)

    return 0


def cmd_inspect(args):
    """Inspect input data and suggest configuration."""
    from engines.input_loader import load_input, detect_data_characteristics

    input_path = Path(args.input_path)
    print(f"Inspecting {input_path}...\n")

    observations = load_input(input_path)
    chars = detect_data_characteristics(observations)

    print(f"Signals:      {chars['n_signals']}")
    print(f"Cohorts:      {chars['n_cohorts']}")
    print(f"Samples:      {chars['min_samples']} min, {chars['max_samples']} max (per signal)")
    print(f"Total rows:   {chars['total_rows']:,}")
    print(f"NaN values:   {chars['nan_count']}")
    print(f"Constants:    {len(chars['constant_signals'])}")

    # Window recommendation
    raw_window = max(16, min(128, chars['min_samples'] // 15))
    import numpy as np
    window = int(2 ** round(np.log2(max(raw_window, 1))))
    print(f"\nRecommended window: {window}")
    print(f"FTLE viable:        {'yes' if chars['ftle_viable'] else 'no (need >=200 samples)'}")
    print(f"Rolling FTLE:       {'yes' if chars['rolling_ftle_viable'] else 'no (need >=400)'}")
    print(f"Granger viable:     {'yes' if chars['granger_viable'] else 'no (need <=100 signals, >=50 samples)'}")

    # Signal preview
    signals = chars['signal_list']
    print(f"\nSignals: {', '.join(signals[:10])}")
    if len(signals) > 10:
        print(f"  ... and {len(signals) - 10} more")

    if chars['constant_signals']:
        print(f"\nConstant signals (will be skipped):")
        for s in chars['constant_signals'][:10]:
            print(f"  - {s}")

    if chars['n_cohorts'] > 1:
        cohorts = chars['cohorts']
        print(f"\nCohorts: {', '.join(str(c) for c in cohorts[:10])}")
        if len(cohorts) > 10:
            print(f"  ... and {len(cohorts) - 10} more")

    print(f"\nRun pipeline:")
    print(f"  engines run {input_path}")
    print(f"  engines run {input_path} --atlas")

    return 0


def cmd_explore(args):
    """Launch the ORTHON explorer on pipeline output."""
    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        print(f"Error: Directory not found: {output_dir}")
        return 1

    parquets = list(output_dir.glob('*.parquet'))
    if not parquets:
        # Check output/ subdirectory
        sub_dir = output_dir / 'output'
        if sub_dir.exists():
            parquets = list(sub_dir.glob('*.parquet'))
            if parquets:
                output_dir = sub_dir

    if not parquets:
        print(f"No parquet files found in {output_dir}")
        return 1

    print(f"Found {len(parquets)} parquet files in {output_dir}")

    # Try ORTHON explorer first
    try:
        from orthon.explorer.server import serve
        print(f"Starting ORTHON explorer on http://localhost:{args.port}")
        serve(str(output_dir), port=args.port)
        return 0
    except ImportError:
        pass

    # Fallback: simple HTTP server serving the directory
    import http.server
    import os
    print(f"Starting file server on http://localhost:{args.port}")
    print(f"Serving: {output_dir}")
    print("Press Ctrl+C to stop\n")

    os.chdir(output_dir)
    handler = http.server.SimpleHTTPRequestHandler
    server = http.server.HTTPServer(('', args.port), handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


def _print_summary(output_dir: Path, manifest: dict):
    """Print summary of pipeline results."""
    import polars as pl

    if not output_dir.exists():
        output_dir = output_dir.parent  # try parent if output/ subdir doesn't exist

    files = list(output_dir.glob('*.parquet'))

    print(f"\n{'='*60}")
    print(f"  Pipeline Complete")
    print(f"{'='*60}")
    print(f"  Output: {output_dir}")
    print(f"  Files:  {len(files)} parquet files")

    # Quick stats from key files
    try:
        geo_path = output_dir / 'state_geometry.parquet'
        if geo_path.exists():
            geo = pl.read_parquet(geo_path, columns=['engine', 'effective_dim'])
            shape = geo.filter(pl.col('engine') == 'shape')
            if len(shape) > 0:
                eff = shape['effective_dim'].drop_nulls()
                if len(eff) > 0:
                    print(f"  eff_dim: {eff.mean():.2f} mean "
                          f"({eff.min():.2f} - {eff.max():.2f})")
    except Exception:
        pass

    try:
        ftle_path = output_dir / 'ftle.parquet'
        if ftle_path.exists():
            ftle = pl.read_parquet(ftle_path, columns=['ftle'])
            unstable = ftle.filter(pl.col('ftle') > 0.01)
            print(f"  FTLE:   {len(unstable)}/{len(ftle)} signals with positive FTLE")
    except Exception:
        pass

    try:
        ridge_path = output_dir / 'ridge_proximity.parquet'
        if ridge_path.exists():
            ridge = pl.read_parquet(ridge_path, columns=['urgency_class'])
            for cls in ['critical', 'warning']:
                n = len(ridge.filter(pl.col('urgency_class') == cls))
                if n > 0:
                    print(f"  Urgency: {n} {cls} samples")
    except Exception:
        pass

    try:
        breaks_path = output_dir / 'breaks.parquet'
        if breaks_path.exists():
            breaks = pl.read_parquet(breaks_path)
            print(f"  Breaks: {len(breaks)} regime changes detected")
    except Exception:
        pass

    print(f"\n  View results:")
    print(f"    engines explore {output_dir}")
    print(f"{'='*60}")


def main():
    """ENGINES CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='engines',
        description='Engines — Dynamical systems computation engines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick start:
    engines run sensor_data.csv              # Run core pipeline
    engines run sensor_data.csv --atlas      # Run with full atlas
    engines inspect sensor_data.csv          # Analyze input data
    engines explore ./engines_output/        # Launch visualization

Pipeline commands:
    engines validate /path/to/data
    engines signal /path/to/manifest.yaml
    engines status /path/to/data
    engines atlas /path/to/data
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # --- User-facing commands ---

    # run command
    run_parser = subparsers.add_parser(
        'run',
        help='Run ENGINES pipeline on input data (CSV, parquet, or directory)',
    )
    run_parser.add_argument(
        'input_path',
        help='Input file (CSV, parquet) or directory with observations.parquet',
    )
    run_parser.add_argument(
        '--output', '-o', default='./engines_output',
        help='Output directory for parquet files (default: ./engines_output)',
    )
    run_parser.add_argument(
        '--manifest', '-m', default=None,
        help='Path to manifest.yaml (auto-generated if not provided)',
    )
    run_parser.add_argument(
        '--atlas', action='store_true',
        help='Enable all atlas engines (velocity, FTLE rolling, ridge proximity)',
    )
    run_parser.add_argument(
        '--segments', '-s', action='append', default=None,
        help='Segment boundaries as name:start:end (e.g., pre:0:19 post:20:)',
    )
    run_parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress verbose output',
    )

    # inspect command
    inspect_parser = subparsers.add_parser(
        'inspect',
        help='Inspect input data and suggest configuration',
    )
    inspect_parser.add_argument(
        'input_path',
        help='Input file to inspect (CSV, parquet, or directory)',
    )

    # explore command
    explore_parser = subparsers.add_parser(
        'explore',
        help='Launch visualization server on pipeline output',
    )
    explore_parser.add_argument(
        'output_dir',
        help='Directory containing parquet output files',
    )
    explore_parser.add_argument(
        '--port', '-p', type=int, default=8080,
        help='Port for local server (default: 8080)',
    )

    # --- Pipeline commands ---

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

    # atlas composite command
    atlas_parser = subparsers.add_parser(
        'atlas',
        help='Run the full Dynamical Atlas pipeline (stages 16-23)',
    )
    atlas_parser.add_argument(
        'data_dir',
        help='Directory containing observations.parquet and manifest.yaml',
    )
    atlas_parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output',
    )
    atlas_parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue running stages even if one fails',
    )

    # Individual atlas stage commands
    atlas_stages = {
        'break-sequence': 'Break propagation order (stage 16)',
        'ftle-backward': 'Backward FTLE / attracting structures (stage 17)',
        'segment-comparison': 'Per-segment geometry deltas (stage 18)',
        'info-flow-delta': 'Per-segment Granger deltas (stage 19)',
        'geometry-full': 'Expanding window eigendecomp trajectory (stage 20)',
        'velocity-field': 'State-space velocity field (stage 21)',
        'ftle-rolling': 'Rolling FTLE stability evolution (stage 22)',
        'ridge-proximity': 'Urgency = velocity toward FTLE ridge (stage 23)',
    }

    for stage_cmd, help_text in atlas_stages.items():
        stage_parser = subparsers.add_parser(stage_cmd, help=help_text)
        stage_parser.add_argument(
            'data_dir',
            help='Directory containing observations.parquet and manifest.yaml',
        )
        stage_parser.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress output',
        )
        # Store which stage this is for dispatch
        stage_parser.set_defaults(stage_command=stage_cmd)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # Dispatch to command handler
    handlers = {
        'run': cmd_run,
        'inspect': cmd_inspect,
        'explore': cmd_explore,
        'validate': cmd_validate,
        'status': cmd_status,
        'signal': cmd_signal,
        'atlas': cmd_atlas,
    }

    # Add atlas stage handlers
    for stage_cmd in atlas_stages:
        handlers[stage_cmd] = cmd_atlas_stage

    handler = handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
