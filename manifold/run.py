"""
Manifold Sequencer
==================

Orchestrates all 28 pipeline stages in dependency order.
Pure orchestration — no computation here.

All stages always run. No opt-in. No tiers.

Architecture: Manifold computes, Prime interprets.
    If it's linear algebra → Manifold.
    If it's SQL → Prime.

Output: 28 parquet files in 6 named directories.

Usage:
    python -m manifold domains/rossler
    python -m manifold domains/rossler --stages 01,02,03
    python -m manifold domains/rossler --skip 08,09a
"""

import argparse
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any

from manifold.io.reader import STAGE_DIRS


# ═══════════════════════════════════════════════════════════════
# STAGE REGISTRY — 28 stages, all always-on
# ═══════════════════════════════════════════════════════════════

# (module_path, stage_id) — module_path relative to manifold.stages
ALL_STAGES = [
    # 1_signal_features + 2_system_state (core geometry)
    ('vector.breaks',                    '00'),
    ('vector.signal_vector',             '01'),
    ('geometry.state_vector',            '02'),
    ('geometry.state_geometry',          '03'),
    ('geometry.signal_geometry',         '05'),
    ('information.signal_pairwise',      '06'),
    ('geometry.geometry_dynamics',       '07'),

    # 5_evolution (trajectory analysis)
    ('dynamics.ftle',                    '08'),
    ('dynamics.lyapunov',                '08_lyapunov'),
    ('dynamics.cohort_thermodynamics',   '09a'),

    # 4_signal_relationships (coupling)
    ('information.information_flow',     '10'),

    # 5_evolution (fields)
    ('dynamics.ftle_field',              '15'),
    ('dynamics.ftle_backward',           '17'),

    # 4_signal_relationships (segments)
    ('information.segment_comparison',   '18'),
    ('information.info_flow_delta',      '19'),

    # 2_system_state (rolling)
    ('geometry.sensor_eigendecomp',      '20'),

    # 5_evolution (motion + rolling + urgency)
    ('dynamics.velocity_field',          '21'),
    ('dynamics.ftle_rolling',            '22'),
    ('dynamics.ridge_proximity',         '23'),
    ('dynamics.persistent_homology',     '36'),

    # 6_fleet (requires n_cohorts >= 2 + cohort_vector from Prime SQL)
    ('energy.system_geometry',           '26'),
    ('energy.cohort_pairwise',           '27'),
    ('energy.cohort_information_flow',   '28'),
    ('energy.cohort_ftle',              '30'),
    ('energy.cohort_velocity_field',     '31'),

    # 1_signal_features (stability)
    ('vector.signal_stability',          '33'),

    # 3_health_scoring (baseline + scoring)
    ('geometry.cohort_baseline',         '34'),
    ('geometry.observation_geometry',    '35'),
]


def _out(output_dir: Path, filename: str) -> str:
    """Resolve output path: output_dir / subdir / filename."""
    stem = filename.replace('.parquet', '')
    subdir = STAGE_DIRS.get(stem, '')
    if subdir:
        d = output_dir / subdir
        d.mkdir(parents=True, exist_ok=True)
        return str(d / filename)
    return str(output_dir / filename)


def run(
    data_path: str,
    stages: Optional[List[str]] = None,
    skip: Optional[List[str]] = None,
    verbose: bool = True,
) -> None:
    """
    Run pipeline stages in dependency order.

    All 27 stages run by default. Use --stages or --skip for debugging only.

    Args:
        data_path: Path to data directory (must contain manifest.yaml)
        stages: Specific stage identifiers to run (e.g., ['01', '02', '08_lyapunov'])
        skip: Stage identifiers to skip
        verbose: Print progress
    """
    import importlib

    data_path = Path(data_path)
    manifest_path = data_path / 'manifest.yaml'

    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.yaml in {data_path}")

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    # Store manifest path for downstream use
    manifest['_manifest_path'] = str(manifest_path)
    manifest['_data_dir'] = str(data_path)

    # Determine output directory
    output_dir = data_path / manifest.get('paths', {}).get('output_dir', 'output')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output subdirectories
    for subdir in sorted(set(STAGE_DIRS.values())):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    _write_readmes(output_dir)

    # Filter stages
    run_stages = ALL_STAGES.copy()

    if stages:
        run_stages = [
            (mod, sid) for mod, sid in run_stages
            if sid in stages
        ]

    if skip:
        run_stages = [
            (mod, sid) for mod, sid in run_stages
            if sid not in skip
        ]

    if verbose:
        print("=" * 70)
        print("MANIFOLD PIPELINE")
        print("=" * 70)
        print(f"Data:     {data_path}")
        print(f"Output:   {output_dir}")
        print(f"Stages:   {len(run_stages)}")
        print()

    obs_path = str(data_path / manifest['paths']['observations'])

    # Run each stage
    for module_path, stage_id in run_stages:
        stage_label = f"{stage_id} ({module_path.split('.')[-1]})"
        if verbose:
            print(f"--- {stage_label} ---")

        try:
            module = importlib.import_module(f'manifold.stages.{module_path}')

            if not hasattr(module, 'run'):
                if verbose:
                    print(f"  Warning: {module_path} has no run() function, skipping")
                continue

            _dispatch(module, module_path, stage_id, obs_path, output_dir, manifest, verbose)

        except Exception as e:
            if verbose:
                print(f"  Error in {stage_label}: {e}")
            raise

        if verbose:
            print()

    if verbose:
        # Report output
        total = 0
        for subdir in sorted(set(STAGE_DIRS.values())):
            files = list((output_dir / subdir).glob('*.parquet'))
            total += len(files)
            print(f"  {subdir}/ ({len(files)} files)")
        print(f"\n  Total: {total} parquet files")
        print()
        print("=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)


def _dispatch(
    module,
    module_path: str,
    stage_id: str,
    obs_path: str,
    output_dir: Path,
    manifest: Dict[str, Any],
    verbose: bool,
) -> None:
    """Dispatch a stage with the correct arguments and output path."""

    intervention = manifest.get('intervention')
    stage_name = module_path.split('.')[-1]  # e.g., 'ftle', 'breaks'

    # ── vector ──

    if stage_name == 'breaks':
        module.run(obs_path, _out(output_dir, 'breaks.parquet'), verbose=verbose)

    elif stage_name == 'signal_vector':
        typology_path = _find_typology(manifest, output_dir)
        module.run(
            obs_path,
            _out(output_dir, 'signal_vector.parquet'),
            manifest,
            verbose=verbose,
            typology_path=typology_path,
        )

    elif stage_name == 'signal_stability':
        module.run(obs_path, _out(output_dir, 'signal_stability.parquet'), verbose=verbose)

    # ── geometry ──

    elif stage_name == 'state_vector':
        typology_path = _find_typology(manifest, output_dir)
        module.run(
            _out(output_dir, 'signal_vector.parquet'),
            _out(output_dir, 'state_vector.parquet'),
            typology_path=typology_path,
            verbose=verbose,
        )

    elif stage_name == 'state_geometry':
        module.run(
            _out(output_dir, 'signal_vector.parquet'),
            _out(output_dir, 'state_vector.parquet'),
            _out(output_dir, 'state_geometry.parquet'),
            verbose=verbose,
        )

    elif stage_name == 'signal_geometry':
        module.run(
            _out(output_dir, 'signal_vector.parquet'),
            _out(output_dir, 'state_vector.parquet'),
            _out(output_dir, 'signal_geometry.parquet'),
            state_geometry_path=_out(output_dir, 'state_geometry.parquet'),
            verbose=verbose,
        )

    elif stage_name == 'geometry_dynamics':
        module.run(
            _out(output_dir, 'state_geometry.parquet'),
            _out(output_dir, 'geometry_dynamics.parquet'),
            verbose=verbose,
        )

    elif stage_name == 'sensor_eigendecomp':
        se_config = manifest.get('sensor_eigendecomp', {})
        module.run(
            obs_path,
            _out(output_dir, 'sensor_eigendecomp.parquet'),
            agg_window=se_config.get('agg_window', 30),
            agg_stride=se_config.get('agg_stride', 5),
            lookback=se_config.get('lookback', 30),
            verbose=verbose,
        )

    elif stage_name == 'cohort_baseline':
        module.run(obs_path, _out(output_dir, 'cohort_baseline.parquet'), verbose=verbose)

    elif stage_name == 'observation_geometry':
        baseline_path = _out(output_dir, 'cohort_baseline.parquet')
        if Path(baseline_path).exists():
            module.run(
                obs_path,
                baseline_path,
                _out(output_dir, 'observation_geometry.parquet'),
                verbose=verbose,
            )
        else:
            if verbose:
                print("  Skipped (cohort_baseline.parquet not found -- run stage 34 first)")

    # ── information ──

    elif stage_name == 'signal_pairwise':
        module.run(
            _out(output_dir, 'signal_vector.parquet'),
            _out(output_dir, 'state_vector.parquet'),
            _out(output_dir, 'signal_pairwise.parquet'),
            state_geometry_path=_out(output_dir, 'state_geometry.parquet'),
            verbose=verbose,
        )

    elif stage_name == 'information_flow':
        import polars as _pl
        pairwise_file = Path(_out(output_dir, 'signal_pairwise.parquet'))
        if pairwise_file.exists() and len(_pl.read_parquet(str(pairwise_file))) > 0:
            module.run(
                obs_path,
                str(pairwise_file),
                _out(output_dir, 'information_flow.parquet'),
                verbose=verbose,
            )
        else:
            if verbose:
                print("  Skipped (empty signal_pairwise)")
            _pl.DataFrame().write_parquet(_out(output_dir, 'information_flow.parquet'))

    elif stage_name == 'segment_comparison':
        segments = _get_segments(manifest)
        module.run(
            obs_path,
            _out(output_dir, 'segment_comparison.parquet'),
            segments=segments,
            verbose=verbose,
        )

    elif stage_name == 'info_flow_delta':
        segments = _get_segments(manifest)
        module.run(
            obs_path,
            _out(output_dir, 'info_flow_delta.parquet'),
            segments=segments,
            verbose=verbose,
        )

    # ── dynamics ──

    elif stage_name == 'ftle':
        module.run(
            obs_path,
            _out(output_dir, 'ftle.parquet'),
            verbose=verbose,
            intervention=intervention,
        )

    elif stage_name == 'lyapunov':
        module.run(obs_path, _out(output_dir, 'lyapunov.parquet'), verbose=verbose)

    elif stage_name == 'cohort_thermodynamics':
        sg_path = _out(output_dir, 'state_geometry.parquet')
        if Path(sg_path).exists():
            module.run(sg_path, _out(output_dir, 'cohort_thermodynamics.parquet'), verbose=verbose)
        else:
            if verbose:
                print("  Skipped (state_geometry.parquet not found)")

    elif stage_name == 'ftle_field':
        module.run(
            _out(output_dir, 'state_vector.parquet'),
            _out(output_dir, 'state_geometry.parquet'),
            _out(output_dir, 'ftle_field.parquet'),
            verbose=verbose,
        )

    elif stage_name == 'ftle_backward':
        module.run(
            obs_path,
            _out(output_dir, 'ftle_backward.parquet'),
            verbose=verbose,
            intervention=intervention,
            direction='backward',
        )

    elif stage_name == 'velocity_field':
        module.run(obs_path, _out(output_dir, 'velocity_field.parquet'), verbose=verbose)

    elif stage_name == 'ftle_rolling':
        module.run(obs_path, _out(output_dir, 'ftle_rolling.parquet'), verbose=verbose)

    elif stage_name == 'ridge_proximity':
        module.run(
            _out(output_dir, 'ftle_rolling.parquet'),
            _out(output_dir, 'velocity_field.parquet'),
            _out(output_dir, 'ridge_proximity.parquet'),
            verbose=verbose,
        )

    elif stage_name == 'persistent_homology':
        sv_path = _out(output_dir, 'state_vector.parquet')
        if Path(sv_path).exists():
            module.run(sv_path, _out(output_dir, 'persistent_homology.parquet'), verbose=verbose)
        else:
            if verbose:
                print("  Skipped (state_vector.parquet not found)")

    # ── energy (fleet) ──

    elif stage_name in (
        'system_geometry', 'cohort_pairwise', 'cohort_information_flow',
        'cohort_ftle', 'cohort_velocity_field',
    ):
        _dispatch_fleet(module, stage_name, output_dir, verbose)

    else:
        if verbose:
            print(f"  Warning: Unknown stage {stage_name}")


def _dispatch_fleet(
    module,
    stage_name: str,
    output_dir: Path,
    verbose: bool,
) -> None:
    """
    Dispatch fleet-scale stages (26-31).

    These require cohort_vector.parquet (produced by Prime SQL stage 25).
    Guard: skip if cohort_vector missing or n_cohorts < 2.
    """
    import polars as _pl

    # cohort_vector could be in output root (from Prime) or in 6_fleet
    cv_path = output_dir / 'cohort_vector.parquet'
    if not cv_path.exists():
        cv_path = output_dir / '6_fleet' / 'cohort_vector.parquet'
    if not cv_path.exists():
        if verbose:
            print("  Skipped (cohort_vector.parquet not found -- produced by Prime SQL)")
        return

    cv = _pl.read_parquet(str(cv_path))
    if len(cv) == 0:
        if verbose:
            print("  Skipped (cohort_vector.parquet is empty)")
        return
    if 'cohort' not in cv.columns or cv['cohort'].n_unique() < 2:
        n = cv['cohort'].n_unique() if 'cohort' in cv.columns else 0
        if verbose:
            print(f"  Skipped (n_cohorts={n} < 2)")
        return

    if stage_name == 'system_geometry':
        module.run(str(cv_path), _out(output_dir, 'system_geometry.parquet'), verbose=verbose)

    elif stage_name == 'cohort_pairwise':
        loadings_path = output_dir / '6_fleet' / 'system_geometry_loadings.parquet'
        if not loadings_path.exists():
            loadings_path = output_dir / 'system_geometry_loadings.parquet'
        module.run(
            str(cv_path),
            _out(output_dir, 'cohort_pairwise.parquet'),
            system_geometry_loadings_path=str(loadings_path) if loadings_path.exists() else None,
            verbose=verbose,
        )

    elif stage_name == 'cohort_information_flow':
        pairwise_path = Path(_out(output_dir, 'cohort_pairwise.parquet'))
        if pairwise_path.exists() and len(_pl.read_parquet(str(pairwise_path))) > 0:
            module.run(
                str(cv_path),
                str(pairwise_path),
                _out(output_dir, 'cohort_information_flow.parquet'),
                verbose=verbose,
            )
        else:
            if verbose:
                print("  Skipped (empty cohort_pairwise)")
            _pl.DataFrame().write_parquet(_out(output_dir, 'cohort_information_flow.parquet'))

    elif stage_name == 'cohort_ftle':
        module.run(str(cv_path), _out(output_dir, 'cohort_ftle.parquet'), verbose=verbose)

    elif stage_name == 'cohort_velocity_field':
        module.run(str(cv_path), _out(output_dir, 'cohort_velocity_field.parquet'), verbose=verbose)


def _find_typology(manifest: Dict[str, Any], output_dir: Path) -> Optional[str]:
    """Find typology.parquet path from manifest."""
    manifest_parent = Path(manifest.get('_manifest_path', '')).parent if '_manifest_path' in manifest else output_dir.parent
    typology_path = manifest_parent / manifest.get('paths', {}).get('typology', 'typology.parquet')
    return str(typology_path) if typology_path.exists() else None


def _get_segments(manifest: Dict[str, Any]) -> Optional[list]:
    """Extract segments config from manifest."""
    segments_config = manifest.get('segments')
    if segments_config:
        return segments_config

    intervention = manifest.get('intervention')
    if intervention and intervention.get('enabled'):
        event_idx = intervention.get('event_index', 20)
        return [
            {'name': 'pre', 'range': [0, event_idx - 1]},
            {'name': 'post', 'range': [event_idx, None]},
        ]
    return None


_READMES = {
    '1_signal_features': """\
# 1. Signal Features

**What does each sensor look like?**

Per-signal windowed analysis: statistics, spectral, entropy, stability.

| File | Grain | Description |
|------|-------|-------------|
| signal_vector.parquet | (signal_id, I) | Per-signal engine outputs |
| signal_geometry.parquet | (signal_id, I) | Per-signal eigendecomposition |
| signal_stability.parquet | (signal_id, I) | Hilbert + wavelet stability |
""",

    '2_system_state': """\
# 2. System State

**What is the system's geometric structure?**

Cross-signal eigendecomposition: the core operation of Manifold.

| File | Grain | Description |
|------|-------|-------------|
| state_vector.parquet | (cohort, I) | Cross-signal centroid per window |
| state_geometry.parquet | (cohort, I) | Eigenvalues, effective dimension |
| geometry_dynamics.parquet | (cohort, I) | Velocity/acceleration/jerk of eigenvalue trajectories |
| sensor_eigendecomp.parquet | (cohort, I) | Rolling 2-level SVD |
""",

    '3_health_scoring': """\
# 3. Health Scoring

**How healthy is this system right now?**

| File | Grain | Description |
|------|-------|-------------|
| breaks.parquet | (cohort, I) | Change-point detection |
| cohort_baseline.parquet | (cohort) | SVD on early-life observations |
| observation_geometry.parquet | (cohort, I) | Per-observation scoring against baseline |
""",

    '4_signal_relationships': """\
# 4. Signal Relationships

**How are sensors connected to each other?**

| File | Grain | Description |
|------|-------|-------------|
| signal_pairwise.parquet | (cohort, signal_a, signal_b, I) | Covariance, correlation, MI |
| information_flow.parquet | (source, target, I) | Transfer entropy, Granger causality |
| segment_comparison.parquet | (cohort, signal_id) | Early vs late statistical tests |
| info_flow_delta.parquet | (source, target) | Delta in information flow between segments |
""",

    '5_evolution': """\
# 5. Evolution

**How is the system changing over time?**

| File | Grain | Description |
|------|-------|-------------|
| ftle.parquet | (cohort, I) | Finite-Time Lyapunov Exponents |
| lyapunov.parquet | (signal_id) | Largest Lyapunov exponent per signal |
| cohort_thermodynamics.parquet | (cohort) | Shannon entropy from eigenvalue spectrum |
| ftle_field.parquet | (cohort, grid_x, grid_y) | Local FTLE grid |
| ftle_backward.parquet | (cohort, I) | Backward FTLE |
| velocity_field.parquet | (cohort, I) | Speed, direction in state space |
| ftle_rolling.parquet | (cohort, I) | Time-varying FTLE windows |
| ridge_proximity.parquet | (cohort, I) | Urgency = v . grad(FTLE) |
| persistent_homology.parquet | (cohort, I) | Betti numbers, persistence entropy |
""",

    '6_fleet': """\
# 6. Fleet

**How does this compare across the fleet?**

Requires `cohort_vector.parquet` (from Prime SQL) and n_cohorts >= 2.

| File | Grain | Description |
|------|-------|-------------|
| system_geometry.parquet | (cohort) | Fleet-level eigendecomposition |
| cohort_pairwise.parquet | (cohort_a, cohort_b) | Distance between cohort vectors |
| cohort_information_flow.parquet | (source, target) | Transfer entropy at fleet scale |
| cohort_ftle.parquet | (cohort, I) | FTLE on cohort trajectories |
| cohort_velocity_field.parquet | (cohort, I) | Drift at fleet scale |
""",
}


def _write_readmes(output_dir: Path) -> None:
    """Write README.md into each output subdirectory (idempotent)."""
    for subdir, content in _READMES.items():
        readme_path = output_dir / subdir / 'README.md'
        if not readme_path.exists():
            readme_path.write_text(content)


def main():
    parser = argparse.ArgumentParser(
        description="Manifold Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
28 stages. All always-on.

Usage:
  python -m manifold domains/rossler
  python -m manifold domains/rossler --stages 01,02,03
  python -m manifold domains/rossler --skip 08,09a
"""
    )
    parser.add_argument('data_path', help='Path to data directory (must contain manifest.yaml)')
    parser.add_argument('--stages', help='Comma-separated stage IDs to run (debugging only)')
    parser.add_argument('--skip', help='Comma-separated stage IDs to skip')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    stages_list = args.stages.split(',') if args.stages else None
    skip_list = args.skip.split(',') if args.skip else None

    run(args.data_path, stages=stages_list, skip=skip_list, verbose=not args.quiet)


if __name__ == '__main__':
    main()
