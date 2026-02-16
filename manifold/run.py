"""
Manifold Sequencer
==================

Orchestrates all 29 pipeline stages in dependency order.
Pure orchestration — no computation here.

All stages always run. No opt-in. No tiers.

Architecture: Manifold computes, Prime interprets.
    If it's linear algebra → Manifold.
    If it's SQL → Prime.

Output: 29 parquet files in 6 named directories.

Usage:
    python -m manifold domains/rossler
    python -m manifold domains/rossler --stages 01,02,03
    python -m manifold domains/rossler --skip 08,09a
"""

import argparse
import copy
import os
import shutil
import time
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any

from manifold.io.reader import STAGE_DIRS


# ═══════════════════════════════════════════════════════════════
# STAGE REGISTRY — 29 stages, all always-on
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

    # 6_fleet (cohort_vector built from state_geometry, then fleet stages)
    ('energy.cohort_vector',             '25'),
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

# Stages that must run globally before parallel split (column pruning is cross-cohort)
GLOBAL_FIRST_IDS = {'00', '01'}

# Stage IDs that can run independently per cohort (after global stages)
COHORT_PARALLEL_IDS = {
    '02', '03', '05', '06', '07', '08', '08_lyapunov',
    '09a', '10', '17', '18', '19', '20', '21', '22', '23', '36', '33',
}


# ═══════════════════════════════════════════════════════════════
# PARALLEL COHORT PROCESSING
# ═══════════════════════════════════════════════════════════════

def _filter_manifest_for_cohort(manifest: dict, cohort: str) -> dict:
    """Create a manifest copy with only the specified cohort's signal configs."""
    filtered = copy.deepcopy(manifest)
    cohorts_cfg = filtered.get('cohorts', {})
    if isinstance(cohorts_cfg, dict) and cohort in cohorts_cfg:
        filtered['cohorts'] = {cohort: cohorts_cfg[cohort]}
    elif isinstance(cohorts_cfg, dict):
        filtered['cohorts'] = {}
    return filtered


def _presplit_observations(
    obs_path: str,
    output_dir: Path,
    manifest: dict,
) -> Optional[dict]:
    """
    Split observations by cohort into per-cohort temp directories.

    Returns dict mapping cohort -> {'data_path', 'obs_path', 'output_dir', 'manifest'}
    or None if parallel execution is not possible.
    """
    import polars as pl

    obs = pl.read_parquet(obs_path)
    if 'cohort' not in obs.columns:
        return None

    cohorts = sorted(obs['cohort'].unique().to_list())
    if len(cohorts) < 2:
        return None

    tmp_dir = output_dir / '_parallel_tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cohort_map = {}
    for cohort in cohorts:
        cohort_str = str(cohort)
        cohort_dir = tmp_dir / cohort_str
        cohort_dir.mkdir(parents=True, exist_ok=True)

        # Write per-cohort observations
        cohort_obs = obs.filter(pl.col('cohort') == cohort)
        cohort_obs_path = str(cohort_dir / 'observations.parquet')
        cohort_obs.write_parquet(cohort_obs_path)

        # Create output subdirectories
        cohort_output = cohort_dir / 'output'
        for subdir in set(STAGE_DIRS.values()):
            (cohort_output / subdir).mkdir(parents=True, exist_ok=True)

        # Filtered manifest
        cohort_manifest = _filter_manifest_for_cohort(manifest, cohort_str)

        cohort_map[cohort_str] = {
            'data_path': str(cohort_dir),
            'obs_path': cohort_obs_path,
            'output_dir': str(cohort_output),
            'manifest': cohort_manifest,
        }

    return cohort_map


def _run_cohort_worker(
    cohort: str,
    cohort_data_path: str,
    cohort_obs_path: str,
    cohort_output_dir: str,
    manifest: dict,
    run_stages: list,
) -> dict:
    """
    Run cohort-parallel stages for a single cohort. Top-level for pickling.

    Returns {'cohort': str, 'status': 'ok'|'error', 'error': str|None}
    """
    import importlib

    # Limit internal parallelism (signal_vector uses joblib)
    os.environ['LOKY_MAX_CPU_COUNT'] = '2'

    output_dir = Path(cohort_output_dir)

    for module_path, stage_id in run_stages:
        if stage_id not in COHORT_PARALLEL_IDS:
            continue
        try:
            module = importlib.import_module(f'manifold.stages.{module_path}')
            if not hasattr(module, 'run'):
                continue
            _dispatch(
                module, module_path, stage_id,
                cohort_obs_path, output_dir, manifest,
                verbose=False,
                data_path_str=cohort_data_path,
            )
        except Exception as e:
            return {'cohort': cohort, 'status': 'error', 'error': f'{stage_id}: {e}'}

    return {'cohort': cohort, 'status': 'ok', 'error': None}


def _merge_cohort_outputs(
    cohort_map: dict,
    final_output_dir: Path,
    verbose: bool = True,
) -> None:
    """Concatenate per-cohort parquet outputs into the final output directory."""
    import polars as pl

    if verbose:
        print("--- Merging cohort outputs ---")

    merged_count = 0

    for subdir in sorted(set(STAGE_DIRS.values())):
        final_subdir = final_output_dir / subdir

        # Discover all parquet files across cohort outputs
        all_files = set()
        for info in cohort_map.values():
            cohort_subdir = Path(info['output_dir']) / subdir
            if cohort_subdir.exists():
                for f in cohort_subdir.glob('*.parquet'):
                    all_files.add(f.name)

        for filename in sorted(all_files):
            parts = []
            for info in cohort_map.values():
                part_path = Path(info['output_dir']) / subdir / filename
                if part_path.exists():
                    try:
                        df = pl.read_parquet(str(part_path))
                        if len(df) > 0:
                            parts.append(df)
                    except Exception:
                        pass

            if parts:
                merged = pl.concat(parts, how='diagonal_relaxed')
                merged.write_parquet(str(final_subdir / filename))
                merged_count += 1
                if verbose:
                    print(f"  {subdir}/{filename}: {len(merged)} rows ({len(parts)} cohorts)")

    if verbose:
        print(f"  Merged {merged_count} files")
        print()


def _cleanup_parallel_tmp(output_dir: Path) -> None:
    """Remove the _parallel_tmp directory tree."""
    import shutil
    tmp_dir = output_dir / '_parallel_tmp'
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)


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
    observations_path: str,
    manifest_path: str,
    output_dir: str,
    stages: Optional[List[str]] = None,
    skip: Optional[List[str]] = None,
    verbose: bool = True,
) -> None:
    """
    Run pipeline stages in dependency order.

    All 29 stages run by default. Use stages/skip for debugging only.

    Args:
        observations_path: Path to observations.parquet
        manifest_path: Path to manifest.yaml
        output_dir: Where to write output parquets
        stages: Specific stage identifiers to run (e.g., ['01', '02', '08_lyapunov'])
        skip: Stage identifiers to skip
        verbose: Print progress
    """
    import importlib

    observations_path = Path(observations_path)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)

    if not observations_path.exists():
        raise FileNotFoundError(f"observations not found: {observations_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    # data_path = parent of output_dir (writer resolves data_path / 'output')
    data_path = output_dir.parent

    manifest['_manifest_path'] = str(manifest_path)
    manifest['_data_dir'] = str(data_path)

    # Safety: refuse to wipe if this looks like a domain root
    if (output_dir / 'observations.parquet').exists():
        raise ValueError(
            f"output_dir ({output_dir}) contains observations.parquet. "
            f"Pass the output/ subdirectory, not the domain root."
        )

    # Fresh start — remove old outputs
    if output_dir.exists():
        shutil.rmtree(output_dir)

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

    # Determine parallelism
    n_workers = int(os.environ.get('MANIFOLD_WORKERS', '4'))
    if n_workers == 0:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    if verbose:
        print("=" * 70)
        print("MANIFOLD PIPELINE")
        print("=" * 70)
        print(f"Input:    {observations_path}")
        print(f"Manifest: {manifest_path}")
        print(f"Output:   {output_dir}")
        print(f"Stages:   {len(run_stages)}")
        print(f"Workers:  {n_workers} ({'parallel' if n_workers > 1 else 'sequential'})")
        print()

    obs_path = str(observations_path)

    # Try parallel execution
    cohort_map = None
    if n_workers > 1:
        cohort_map = _presplit_observations(obs_path, output_dir, manifest)

    if cohort_map is not None:
        # ═══ PARALLEL PATH ═══
        import polars as pl
        from concurrent.futures import ProcessPoolExecutor, as_completed

        n_cohorts = len(cohort_map)
        effective_workers = min(n_workers, n_cohorts)
        global_stages = [(m, s) for m, s in run_stages if s in GLOBAL_FIRST_IDS]
        parallel_stages = [(m, s) for m, s in run_stages if s in COHORT_PARALLEL_IDS]
        fleet_stages_list = [(m, s) for m, s in run_stages
                             if s not in COHORT_PARALLEL_IDS and s not in GLOBAL_FIRST_IDS]

        # Phase 0.5: Run global stages (00 breaks, 01 signal_vector) on full data
        if global_stages:
            if verbose:
                print(f"Phase 0: {len(global_stages)} global stages (sequential)")
                print()

            for module_path, stage_id in global_stages:
                stage_label = f"{stage_id} ({module_path.split('.')[-1]})"
                if verbose:
                    print(f"--- {stage_label} ---")
                try:
                    module = importlib.import_module(f'manifold.stages.{module_path}')
                    if not hasattr(module, 'run'):
                        if verbose:
                            print(f"  Warning: {module_path} has no run() function, skipping")
                        continue
                    _dispatch(module, module_path, stage_id, obs_path, output_dir, manifest, verbose, str(data_path))
                except Exception as e:
                    if verbose:
                        print(f"  Error in {stage_label}: {e}")
                    raise
                if verbose:
                    print()

        # Phase 0.75: Seed per-cohort temp dirs with signal_vector slices
        # (so stages 02+ read the global column schema per-cohort)
        sv_path = _out(output_dir, 'signal_vector.parquet')
        if Path(sv_path).exists():
            sv_all = pl.read_parquet(sv_path)
            has_sv_cohort = 'cohort' in sv_all.columns
            for cohort, info in cohort_map.items():
                cohort_output = Path(info['output_dir'])
                if has_sv_cohort:
                    sv_cohort = sv_all.filter(pl.col('cohort') == cohort)
                else:
                    sv_cohort = sv_all
                sv_dest = cohort_output / STAGE_DIRS.get('signal_vector', '') / 'signal_vector.parquet'
                sv_dest.parent.mkdir(parents=True, exist_ok=True)
                sv_cohort.write_parquet(str(sv_dest))

        breaks_path = _out(output_dir, 'breaks.parquet')
        if Path(breaks_path).exists():
            br_all = pl.read_parquet(breaks_path)
            has_br_cohort = 'cohort' in br_all.columns
            for cohort, info in cohort_map.items():
                cohort_output = Path(info['output_dir'])
                if has_br_cohort:
                    br_cohort = br_all.filter(pl.col('cohort') == cohort)
                else:
                    br_cohort = br_all
                br_dest = cohort_output / STAGE_DIRS.get('breaks', '') / 'breaks.parquet'
                br_dest.parent.mkdir(parents=True, exist_ok=True)
                br_cohort.write_parquet(str(br_dest))

        if verbose:
            print(f"Phase 1: {n_cohorts} cohorts x {len(parallel_stages)} stages on {effective_workers} workers")
            print()

        # Phase 1: Parallel per-cohort processing
        start_time = time.time()
        completed = 0
        failed_cohorts = []

        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            futures = {}
            for cohort, info in cohort_map.items():
                future = executor.submit(
                    _run_cohort_worker,
                    cohort=cohort,
                    cohort_data_path=info['data_path'],
                    cohort_obs_path=info['obs_path'],
                    cohort_output_dir=info['output_dir'],
                    manifest=info['manifest'],
                    run_stages=parallel_stages,
                )
                futures[future] = cohort

            for future in as_completed(futures):
                cohort = futures[future]
                completed += 1
                try:
                    result = future.result()
                    elapsed = time.time() - start_time
                    if result['status'] == 'error':
                        failed_cohorts.append((cohort, result['error']))
                        if verbose:
                            print(f"  [{completed}/{n_cohorts}] {cohort} FAILED: {result['error']} ({elapsed:.1f}s)")
                    else:
                        if verbose:
                            rate = completed / elapsed if elapsed > 0 else 0
                            eta = (n_cohorts - completed) / rate if rate > 0 else 0
                            print(f"  [{completed}/{n_cohorts}] {cohort} ({elapsed:.1f}s, {rate:.1f}/s, ETA {eta:.0f}s)")
                except Exception as e:
                    failed_cohorts.append((cohort, str(e)))
                    if verbose:
                        print(f"  [{completed}/{n_cohorts}] {cohort} FAILED: {e}")

        if failed_cohorts and verbose:
            print(f"\n  WARNING: {len(failed_cohorts)} cohort(s) failed:")
            for c, err in failed_cohorts:
                print(f"    {c}: {err}")
            print()

        if verbose:
            elapsed = time.time() - start_time
            print(f"  Phase 1 complete: {completed - len(failed_cohorts)}/{n_cohorts} cohorts in {elapsed:.1f}s")
            print()

        # Phase 2: Merge per-cohort outputs
        _merge_cohort_outputs(cohort_map, output_dir, verbose=verbose)

        # Phase 3: Fleet + cross-cohort stages (sequential)
        if fleet_stages_list:
            if verbose:
                print(f"Phase 3: {len(fleet_stages_list)} fleet/cross-cohort stages (sequential)")
                print()

            for module_path, stage_id in fleet_stages_list:
                stage_label = f"{stage_id} ({module_path.split('.')[-1]})"
                if verbose:
                    print(f"--- {stage_label} ---")

                try:
                    module = importlib.import_module(f'manifold.stages.{module_path}')

                    if not hasattr(module, 'run'):
                        if verbose:
                            print(f"  Warning: {module_path} has no run() function, skipping")
                        continue

                    _dispatch(module, module_path, stage_id, obs_path, output_dir, manifest, verbose, str(data_path))

                except Exception as e:
                    if verbose:
                        print(f"  Error in {stage_label}: {e}")
                    raise

                if verbose:
                    print()

        # Cleanup temp files
        _cleanup_parallel_tmp(output_dir)

    else:
        # ═══ SEQUENTIAL PATH (existing behavior) ═══
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

                _dispatch(module, module_path, stage_id, obs_path, output_dir, manifest, verbose, str(data_path))

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
    data_path_str: str = '',
) -> None:
    """Dispatch a stage with the correct arguments and output path."""

    intervention = manifest.get('intervention')
    stage_name = module_path.split('.')[-1]  # e.g., 'ftle', 'breaks'

    # ── vector ──

    if stage_name == 'breaks':
        module.run(obs_path, data_path=data_path_str, verbose=verbose)

    elif stage_name == 'signal_vector':
        typology_path = _find_typology(manifest, output_dir)
        module.run(
            obs_path,
            data_path=data_path_str,
            manifest=manifest,
            verbose=verbose,
            typology_path=typology_path,
        )

    elif stage_name == 'signal_stability':
        module.run(obs_path, data_path=data_path_str, verbose=verbose)

    # ── geometry ──

    elif stage_name == 'state_vector':
        typology_path = _find_typology(manifest, output_dir)
        module.run(
            _out(output_dir, 'signal_vector.parquet'),
            data_path=data_path_str,
            typology_path=typology_path,
            verbose=verbose,
        )

    elif stage_name == 'state_geometry':
        module.run(
            _out(output_dir, 'signal_vector.parquet'),
            _out(output_dir, 'state_vector.parquet'),
            data_path=data_path_str,
            verbose=verbose,
        )

    elif stage_name == 'signal_geometry':
        module.run(
            _out(output_dir, 'signal_vector.parquet'),
            _out(output_dir, 'state_vector.parquet'),
            data_path=data_path_str,
            state_geometry_path=_out(output_dir, 'state_geometry.parquet'),
            verbose=verbose,
        )

    elif stage_name == 'geometry_dynamics':
        module.run(
            _out(output_dir, 'state_geometry.parquet'),
            data_path=data_path_str,
            verbose=verbose,
        )

    elif stage_name == 'sensor_eigendecomp':
        se_config = manifest.get('sensor_eigendecomp', {})
        module.run(
            obs_path,
            data_path=data_path_str,
            agg_window=se_config.get('agg_window', 30),
            agg_stride=se_config.get('agg_stride', 5),
            lookback=se_config.get('lookback', 30),
            verbose=verbose,
        )

    elif stage_name == 'cohort_baseline':
        module.run(obs_path, data_path=data_path_str, verbose=verbose)

    elif stage_name == 'observation_geometry':
        baseline_path = _out(output_dir, 'cohort_baseline.parquet')
        if Path(baseline_path).exists():
            module.run(
                obs_path,
                baseline_path,
                data_path=data_path_str,
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
            data_path=data_path_str,
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
                data_path=data_path_str,
                verbose=verbose,
            )
        else:
            if verbose:
                print("  Skipped (empty signal_pairwise)")
            from manifold.io.writer import write_output as _write
            _write(_pl.DataFrame(), data_path_str, 'information_flow', verbose=verbose)

    elif stage_name == 'segment_comparison':
        segments = _get_segments(manifest)
        module.run(
            obs_path,
            data_path=data_path_str,
            segments=segments,
            verbose=verbose,
        )

    elif stage_name == 'info_flow_delta':
        segments = _get_segments(manifest)
        module.run(
            obs_path,
            data_path=data_path_str,
            segments=segments,
            verbose=verbose,
        )

    # ── dynamics ──

    elif stage_name == 'ftle':
        module.run(
            obs_path,
            data_path=data_path_str,
            verbose=verbose,
            intervention=intervention,
        )

    elif stage_name == 'lyapunov':
        module.run(obs_path, data_path=data_path_str, verbose=verbose)

    elif stage_name == 'cohort_thermodynamics':
        sg_path = _out(output_dir, 'state_geometry.parquet')
        if Path(sg_path).exists():
            module.run(sg_path, data_path=data_path_str, verbose=verbose)
        else:
            if verbose:
                print("  Skipped (state_geometry.parquet not found)")

    elif stage_name == 'ftle_field':
        module.run(
            _out(output_dir, 'state_vector.parquet'),
            _out(output_dir, 'state_geometry.parquet'),
            data_path=data_path_str,
            verbose=verbose,
        )

    elif stage_name == 'ftle_backward':
        module.run(
            obs_path,
            data_path=data_path_str,
            verbose=verbose,
            intervention=intervention,
            direction='backward',
        )

    elif stage_name == 'velocity_field':
        module.run(obs_path, data_path=data_path_str, verbose=verbose)

    elif stage_name == 'ftle_rolling':
        module.run(obs_path, data_path=data_path_str, verbose=verbose)

    elif stage_name == 'ridge_proximity':
        module.run(
            _out(output_dir, 'ftle_rolling.parquet'),
            _out(output_dir, 'velocity_field.parquet'),
            data_path=data_path_str,
            verbose=verbose,
        )

    elif stage_name == 'persistent_homology':
        sv_path = _out(output_dir, 'state_vector.parquet')
        if Path(sv_path).exists():
            module.run(sv_path, data_path=data_path_str, verbose=verbose)
        else:
            if verbose:
                print("  Skipped (state_vector.parquet not found)")

    # ── energy (fleet) ──

    elif stage_name == 'cohort_vector':
        sg_path = _out(output_dir, 'state_geometry.parquet')
        if Path(sg_path).exists():
            module.run(sg_path, data_path=data_path_str, verbose=verbose)
        else:
            if verbose:
                print("  Skipped (state_geometry.parquet not found)")

    elif stage_name in (
        'system_geometry', 'cohort_pairwise', 'cohort_information_flow',
        'cohort_ftle', 'cohort_velocity_field',
    ):
        _dispatch_fleet(module, stage_name, output_dir, verbose, data_path_str)

    else:
        if verbose:
            print(f"  Warning: Unknown stage {stage_name}")


def _dispatch_fleet(
    module,
    stage_name: str,
    output_dir: Path,
    verbose: bool,
    data_path_str: str = '',
) -> None:
    """
    Dispatch fleet-scale stages (26-31).

    These require cohort_vector.parquet (produced by stage 25).
    Guard: skip if cohort_vector missing or n_cohorts < 2.
    """
    import polars as _pl

    # cohort_vector could be in output root (from Prime) or in 6_fleet
    cv_path = output_dir / 'cohort_vector.parquet'
    if not cv_path.exists():
        cv_path = output_dir / '6_fleet' / 'cohort_vector.parquet'
    if not cv_path.exists():
        if verbose:
            print("  Skipped (cohort_vector.parquet not found -- run stage 25 first)")
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
        module.run(str(cv_path), data_path=data_path_str, verbose=verbose)

    elif stage_name == 'cohort_pairwise':
        loadings_path = output_dir / '6_fleet' / 'system_geometry_loadings.parquet'
        if not loadings_path.exists():
            loadings_path = output_dir / 'system_geometry_loadings.parquet'
        module.run(
            str(cv_path),
            data_path=data_path_str,
            system_geometry_loadings_path=str(loadings_path) if loadings_path.exists() else None,
            verbose=verbose,
        )

    elif stage_name == 'cohort_information_flow':
        pairwise_path = Path(_out(output_dir, 'cohort_pairwise.parquet'))
        if pairwise_path.exists() and len(_pl.read_parquet(str(pairwise_path))) > 0:
            module.run(
                str(cv_path),
                str(pairwise_path),
                data_path=data_path_str,
                verbose=verbose,
            )
        else:
            if verbose:
                print("  Skipped (empty cohort_pairwise)")
            from manifold.io.writer import write_output as _write
            _write(_pl.DataFrame(), data_path_str, 'cohort_information_flow', verbose=verbose)

    elif stage_name == 'cohort_ftle':
        module.run(str(cv_path), data_path=data_path_str, verbose=verbose)

    elif stage_name == 'cohort_velocity_field':
        module.run(str(cv_path), data_path=data_path_str, verbose=verbose)


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

Requires `cohort_vector.parquet` (from stage 25) and n_cohorts >= 2.

| File | Grain | Description |
|------|-------|-------------|
| cohort_vector.parquet | (cohort, I) | Wide-format cohort features pivoted from state_geometry |
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
    """CLI entry point. Resolves data_path into explicit paths and calls run()."""
    parser = argparse.ArgumentParser(
        description="Manifold Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
29 stages. All always-on.

Usage:
  python -m manifold ~/domains/rossler
  python -m manifold ~/domains/rossler --stages 01,02,03
  python -m manifold ~/domains/rossler --skip 08,09a
"""
    )
    parser.add_argument('data_path', help='Path to data directory (must contain manifest.yaml)')
    parser.add_argument('--stages', help='Comma-separated stage IDs to run (debugging only)')
    parser.add_argument('--skip', help='Comma-separated stage IDs to skip')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    data_path = Path(args.data_path)
    manifest_path = data_path / 'manifest.yaml'

    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.yaml in {data_path}")

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    obs_rel = manifest.get('paths', {}).get('observations', 'observations.parquet')
    out_rel = manifest.get('paths', {}).get('output_dir', 'output')

    stages_list = args.stages.split(',') if args.stages else None
    skip_list = args.skip.split(',') if args.skip else None

    run(
        observations_path=str(data_path / obs_rel),
        manifest_path=str(manifest_path),
        output_dir=str(data_path / out_rel),
        stages=stages_list,
        skip=skip_list,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
