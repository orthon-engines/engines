"""
PRISM Pipeline Runner
=====================

Orchestrates all pipeline stages in dependency order.
Pure orchestration - no computation here.

Usage:
    python -m prism.entry_points.run_pipeline manifest.yaml
    python -m prism.entry_points.run_pipeline manifest.yaml --stages 01,02,03
    python -m prism.entry_points.run_pipeline manifest.yaml --skip 08,09
"""

import argparse
import importlib
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any


# Canonical stage order - reflects dependency graph
CORE_STAGES = [
    'stage_00_breaks',
    'stage_01_signal_vector',
    'stage_02_state_vector',
    'stage_03_state_geometry',
    'stage_04_cohorts',
    'stage_05_signal_geometry',
    'stage_06_signal_pairwise',
    'stage_07_geometry_dynamics',
    'stage_08_ftle',
    'stage_09_dynamics',
    'stage_10_information_flow',
    'stage_11_topology',
    'stage_12_zscore',
    'stage_13_statistics',
    'stage_14_correlation',
    # stage_15_ftle_field is OPTIONAL - requires --stages 15 to run
]

# Advanced/optional stages (not run by default)
ADVANCED_STAGES = [
    'stage_15_ftle_field',         # Local FTLE fields around centroids (LCS detection)
    'stage_16_break_sequence',     # Break propagation order (requires breaks.parquet)
    'stage_17_ftle_backward',      # Backward FTLE (attracting structures)
    'stage_18_segment_comparison', # Per-segment geometry deltas
    'stage_19_info_flow_delta',    # Per-segment Granger deltas
    'stage_20_geometry_full',      # Expanding window eigendecomp trajectory
    'stage_21_velocity_field',     # State-space velocity: direction, speed, curvature
    'stage_22_ftle_rolling',       # FTLE at each timestep
    'stage_23_ridge_proximity',    # Urgency = velocity toward FTLE ridge
]

# Stage dependencies for validation
STAGE_DEPS = {
    'stage_00_breaks': [],
    'stage_01_signal_vector': ['observations.parquet', 'manifest.yaml'],
    'stage_02_state_vector': ['signal_vector.parquet'],
    'stage_03_state_geometry': ['signal_vector.parquet', 'state_vector.parquet'],
    'stage_04_cohorts': ['signal_vector.parquet', 'state_vector.parquet'],
    'stage_05_signal_geometry': ['signal_vector.parquet', 'state_vector.parquet'],
    'stage_06_signal_pairwise': ['signal_vector.parquet', 'state_vector.parquet'],
    'stage_07_geometry_dynamics': ['state_geometry.parquet'],
    'stage_08_ftle': ['observations.parquet'],
    'stage_09_dynamics': ['ftle.parquet'],
    'stage_10_information_flow': ['signal_pairwise.parquet'],
    'stage_11_topology': ['state_geometry.parquet', 'dynamics.parquet'],
    'stage_12_zscore': [],  # Reads from output_dir
    'stage_13_statistics': [],  # Reads from output_dir
    'stage_14_correlation': ['signal_vector.parquet'],
    'stage_15_ftle_field': ['state_vector.parquet', 'state_geometry.parquet'],
    'stage_16_break_sequence': ['breaks.parquet'],
    'stage_17_ftle_backward': ['observations.parquet'],
    'stage_18_segment_comparison': ['observations.parquet'],
    'stage_19_info_flow_delta': ['observations.parquet'],
    'stage_20_geometry_full': ['observations.parquet'],
    'stage_21_velocity_field': ['observations.parquet'],
    'stage_22_ftle_rolling': ['observations.parquet'],
    'stage_23_ridge_proximity': ['ftle_rolling.parquet', 'velocity_field.parquet'],
}


def load_manifest(manifest_path: str) -> Dict[str, Any]:
    """Load manifest.yaml."""
    with open(manifest_path) as f:
        return yaml.safe_load(f)


def get_stage_number(stage_name: str) -> str:
    """Extract stage number from name (e.g., 'stage_01_signal_vector' -> '01')."""
    parts = stage_name.split('_')
    if len(parts) >= 2:
        return parts[1]
    return ''


def run(
    manifest_path: str,
    stages: Optional[List[str]] = None,
    skip: Optional[List[str]] = None,
    verbose: bool = True,
) -> None:
    """
    Run pipeline stages in dependency order.

    Args:
        manifest_path: Path to manifest.yaml
        stages: Specific stage numbers to run (e.g., ['01', '02', '03'])
        skip: Stage numbers to skip (e.g., ['08', '09'])
        verbose: Print progress
    """
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)

    # Determine output directory
    output_dir = manifest_path.parent / manifest.get('paths', {}).get('output_dir', 'output')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which stages to run
    # Include advanced stages only when explicitly requested
    all_stages = CORE_STAGES + ADVANCED_STAGES
    if stages:
        run_stages = [
            s for s in all_stages
            if get_stage_number(s) in stages
        ]
    else:
        run_stages = CORE_STAGES.copy()  # Default excludes advanced stages

    # Apply skip filter
    if skip:
        run_stages = [
            s for s in run_stages
            if get_stage_number(s) not in skip
        ]

    if verbose:
        print("=" * 70)
        print("PRISM PIPELINE")
        print("=" * 70)
        print(f"Manifest: {manifest_path}")
        print(f"Output:   {output_dir}")
        print(f"Stages:   {len(run_stages)}")
        print()

    # Run each stage
    for stage_name in run_stages:
        if verbose:
            print(f"─── {stage_name} ───")

        try:
            module = importlib.import_module(f'prism.entry_points.{stage_name}')

            # Get the run function
            if not hasattr(module, 'run'):
                if verbose:
                    print(f"  Warning: {stage_name} has no run() function, skipping")
                continue

            # Call run with appropriate arguments based on stage
            stage_num = get_stage_number(stage_name)

            if stage_num == '00':
                # Breaks
                obs_path = manifest_path.parent / manifest['paths']['observations']
                module.run(str(obs_path), str(output_dir / 'breaks.parquet'), verbose=verbose)

            elif stage_num == '01':
                # Signal vector - run() takes (observations_path, output_path, manifest_dict)
                obs_path = manifest_path.parent / manifest['paths']['observations']
                typology_path = manifest_path.parent / manifest['paths'].get('typology', 'typology.parquet')
                module.run(
                    str(obs_path),
                    str(output_dir / 'signal_vector.parquet'),
                    manifest,
                    verbose=verbose,
                    typology_path=str(typology_path) if typology_path.exists() else None,
                )

            elif stage_num == '02':
                # State vector
                typology_path = manifest_path.parent / manifest['paths'].get('typology', 'typology.parquet')
                module.run(
                    str(output_dir / 'signal_vector.parquet'),
                    str(output_dir / 'state_vector.parquet'),
                    typology_path=str(typology_path) if typology_path.exists() else None,
                    verbose=verbose,
                )

            elif stage_num == '03':
                # State geometry
                module.run(
                    str(output_dir / 'signal_vector.parquet'),
                    str(output_dir / 'state_vector.parquet'),
                    str(output_dir / 'state_geometry.parquet'),
                    verbose=verbose,
                )

            elif stage_num == '04':
                # Cohorts
                module.run(
                    str(output_dir / 'signal_vector.parquet'),
                    str(output_dir / 'state_vector.parquet'),
                    str(output_dir / 'cohorts.parquet'),
                    state_geometry_path=str(output_dir / 'state_geometry.parquet'),
                    verbose=verbose,
                )

            elif stage_num == '05':
                # Signal geometry
                module.run(
                    str(output_dir / 'signal_vector.parquet'),
                    str(output_dir / 'state_vector.parquet'),
                    str(output_dir / 'signal_geometry.parquet'),
                    state_geometry_path=str(output_dir / 'state_geometry.parquet'),
                    verbose=verbose,
                )

            elif stage_num == '06':
                # Signal pairwise
                module.run(
                    str(output_dir / 'signal_vector.parquet'),
                    str(output_dir / 'state_vector.parquet'),
                    str(output_dir / 'signal_pairwise.parquet'),
                    state_geometry_path=str(output_dir / 'state_geometry.parquet'),
                    verbose=verbose,
                )

            elif stage_num == '07':
                # Geometry dynamics
                module.run(
                    str(output_dir / 'state_geometry.parquet'),
                    str(output_dir / 'geometry_dynamics.parquet'),
                    verbose=verbose,
                )

            elif stage_num == '08':
                # FTLE (Finite-Time Lyapunov Exponents)
                obs_path = manifest_path.parent / manifest['paths']['observations']
                intervention = manifest.get('intervention')
                module.run(
                    str(obs_path),
                    str(output_dir / 'ftle.parquet'),
                    verbose=verbose,
                    intervention=intervention,
                )

            elif stage_num == '09':
                # Dynamics - reads observations (like Lyapunov)
                # TODO: This is redundant with stage_08. Stage_09 should do
                # attractor metrics, RQA, etc. For now, just run it.
                obs_path = manifest_path.parent / manifest['paths']['observations']
                module.run(str(obs_path), str(output_dir / 'dynamics.parquet'), verbose=verbose)

            elif stage_num == '10':
                # Information flow - uses signal_pairwise for Granger gating + observations for time series
                obs_path = manifest_path.parent / manifest['paths']['observations']
                module.run(
                    str(obs_path),
                    str(output_dir / 'signal_pairwise.parquet'),
                    str(output_dir / 'information_flow.parquet'),
                    verbose=verbose,
                )

            elif stage_num == '11':
                # Topology - current implementation takes signal_vector
                module.run(
                    str(output_dir / 'signal_vector.parquet'),
                    str(output_dir / 'topology.parquet'),
                    verbose=verbose,
                )

            elif stage_num == '12':
                # Zscore
                module.run(str(output_dir), str(output_dir / 'zscore.parquet'), verbose=verbose)

            elif stage_num == '13':
                # Statistics - reads observations
                obs_path = manifest_path.parent / manifest['paths']['observations']
                module.run(str(obs_path), str(output_dir / 'statistics.parquet'), verbose=verbose)

            elif stage_num == '14':
                # Correlation
                module.run(
                    str(output_dir / 'signal_vector.parquet'),
                    str(output_dir / 'correlation.parquet'),
                    verbose=verbose,
                )

            elif stage_num == '15':
                # FTLE Field - Local FTLE around centroids (LCS detection)
                module.run(
                    str(output_dir / 'state_vector.parquet'),
                    str(output_dir / 'state_geometry.parquet'),
                    str(output_dir / 'ftle_field.parquet'),
                    verbose=verbose,
                )

            elif stage_num == '16':
                # Break Sequence - propagation order
                intervention = manifest.get('intervention')
                ref_index = intervention.get('event_index', 0) if intervention else None
                module.run(
                    str(output_dir / 'breaks.parquet'),
                    str(output_dir / 'break_sequence.parquet'),
                    reference_index=ref_index,
                    verbose=verbose,
                )

            elif stage_num == '17':
                # Backward FTLE - attracting structures
                obs_path = manifest_path.parent / manifest['paths']['observations']
                intervention = manifest.get('intervention')
                module.run(
                    str(obs_path),
                    str(output_dir / 'ftle_backward.parquet'),
                    verbose=verbose,
                    intervention=intervention,
                    direction='backward',
                )

            elif stage_num == '18':
                # Segment comparison - per-segment geometry
                obs_path = manifest_path.parent / manifest['paths']['observations']
                intervention = manifest.get('intervention')
                segments_config = manifest.get('segments')

                # Build segments from intervention or manifest config
                if segments_config:
                    segments = segments_config
                elif intervention and intervention.get('enabled'):
                    event_idx = intervention.get('event_index', 20)
                    segments = [
                        {'name': 'pre', 'range': [0, event_idx - 1]},
                        {'name': 'post', 'range': [event_idx, None]},
                    ]
                else:
                    segments = None  # Use default

                module.run(
                    str(obs_path),
                    str(output_dir / 'segment_comparison.parquet'),
                    segments=segments,
                    verbose=verbose,
                )

            elif stage_num == '19':
                # Information flow delta - per-segment Granger
                obs_path = manifest_path.parent / manifest['paths']['observations']
                intervention = manifest.get('intervention')
                segments_config = manifest.get('segments')

                if segments_config:
                    segments = segments_config
                elif intervention and intervention.get('enabled'):
                    event_idx = intervention.get('event_index', 20)
                    segments = [
                        {'name': 'pre', 'range': [0, event_idx - 1]},
                        {'name': 'post', 'range': [event_idx, None]},
                    ]
                else:
                    segments = None

                module.run(
                    str(obs_path),
                    str(output_dir / 'info_flow_delta.parquet'),
                    segments=segments,
                    verbose=verbose,
                )

            elif stage_num == '20':
                # Full-span geometry - expanding window eigendecomp
                obs_path = manifest_path.parent / manifest['paths']['observations']
                module.run(
                    str(obs_path),
                    str(output_dir / 'geometry_full.parquet'),
                    verbose=verbose,
                )

            elif stage_num == '21':
                # Velocity field - state-space motion
                obs_path = manifest_path.parent / manifest['paths']['observations']
                module.run(
                    str(obs_path),
                    str(output_dir / 'velocity_field.parquet'),
                    verbose=verbose,
                )

            elif stage_num == '22':
                # Rolling FTLE - stability evolution
                obs_path = manifest_path.parent / manifest['paths']['observations']
                module.run(
                    str(obs_path),
                    str(output_dir / 'ftle_rolling.parquet'),
                    verbose=verbose,
                )

            elif stage_num == '23':
                # Ridge proximity - urgency metric
                # Requires ftle_rolling and velocity_field
                module.run(
                    str(output_dir / 'ftle_rolling.parquet'),
                    str(output_dir / 'velocity_field.parquet'),
                    str(output_dir / 'ridge_proximity.parquet'),
                    verbose=verbose,
                )

            else:
                if verbose:
                    print(f"  Warning: Unknown stage number {stage_num}")

        except Exception as e:
            if verbose:
                print(f"  Error in {stage_name}: {e}")
            raise

        if verbose:
            print()

    if verbose:
        print("=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="PRISM Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Runs pipeline stages in canonical dependency order.

Stage Order:
  00: breaks           - Break detection
  01: signal_vector    - Per-signal features
  02: state_vector     - Centroids
  03: state_geometry   - Eigenvalues, PCs
  04: cohorts          - Cohort aggregation
  05: signal_geometry  - Signal-to-state relationships
  06: signal_pairwise  - Pairwise with PC gating
  07: geometry_dynamics - d1/d2/d3 of geometry
  08: ftle             - Per-signal FTLE (Finite-Time Lyapunov)
  09: dynamics         - Full dynamics
  10: information_flow - Granger causality
  11: topology         - Topological features
  12: zscore           - Normalization
  13: statistics       - Summary stats
  14: correlation      - Correlation matrix

Advanced (opt-in via --stages 15,16,17,18,19):
  15: ftle_field         - Local FTLE fields (LCS detection)
  16: break_sequence     - Propagation order (uses intervention.event_index)
  17: ftle_backward      - Backward FTLE (attracting structures)
  18: segment_comparison - Per-segment geometry with deltas
  19: info_flow_delta    - Per-segment Granger with link changes

Examples:
  python -m prism.entry_points.run_pipeline manifest.yaml
  python -m prism.entry_points.run_pipeline manifest.yaml --stages 01,02,03
  python -m prism.entry_points.run_pipeline manifest.yaml --skip 08,09
"""
    )
    parser.add_argument('manifest', help='Path to manifest.yaml')
    parser.add_argument('--stages', help='Comma-separated stage numbers to run (e.g., 01,02,03)')
    parser.add_argument('--skip', help='Comma-separated stage numbers to skip (e.g., 08,09)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    stages = args.stages.split(',') if args.stages else None
    skip = args.skip.split(',') if args.skip else None

    run(args.manifest, stages=stages, skip=skip, verbose=not args.quiet)


if __name__ == '__main__':
    main()
