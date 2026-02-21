"""
Manifold Sequencer
==================

Orchestrates all 25 pipeline stages in dependency order.
Pure orchestration — no computation here.

All stages always run. No opt-in. No tiers.

Architecture: Manifold computes, Prime interprets.
    If it's linear algebra → Manifold.
    If it's SQL → Prime.

Output: 28 parquet files in 6 directories (signal/, cohort/, cohort/cohort_dynamics/, system/, system/system_dynamics/, parameterization/).

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
from pathlib import Path
from typing import List, Optional, Dict, Any

from manifold.io.manifest import load_manifest
from manifold.io.reader import STAGE_DIRS, STAGE_FILENAMES


def validate_manifest_paths(manifest: dict) -> list[str]:
    """Validate all input paths in the manifest exist before running any stages.

    Called once at Manifold startup, before any computation.
    Returns list of errors. Empty list = all paths valid.
    """
    errors = []
    paths = manifest.get('paths', {})

    # Input files that must exist
    obs = paths.get('observations')
    if obs and not Path(obs).exists():
        errors.append(f"observations file not found: {obs}")

    # Output directory must be creatable
    out = paths.get('output_dir')
    if out:
        out_path = Path(out)
        if not out_path.parent.exists():
            errors.append(f"output_dir parent does not exist: {out_path.parent}")

    # Warn on relative paths
    for key, val in paths.items():
        if val and isinstance(val, str) and not Path(val).is_absolute():
            errors.append(f"paths.{key} is relative (should be absolute): {val}")

    return errors


# ═══════════════════════════════════════════════════════════════
# STAGE REGISTRY — 25 stages, all always-on
# ═══════════════════════════════════════════════════════════════

# (module_path, stage_id) — module_path relative to manifold.stages
ALL_STAGES = [
    # signal/ + cohort/ (core geometry)
    ('vector.breaks',                    '00'),
    ('vector.signal_vector',             '01'),
    ('geometry.state_vector',            '02'),
    ('geometry.state_geometry',          '03'),
    ('geometry.signal_dominance',       '03b'),
    ('geometry.signal_geometry',         '05'),
    ('information.signal_pairwise',      '06'),
    ('geometry.geometry_dynamics',       '07'),

    # cohort/cohort_dynamics/ (trajectory analysis)
    ('dynamics.ftle',                    '08'),
    ('dynamics.lyapunov',                '08_lyapunov'),
    ('dynamics.cohort_thermodynamics',   '09a'),

    # cohort/ (coupling)
    ('information.information_flow',     '10'),

    # cohort/cohort_dynamics/ (fields)
    ('dynamics.ftle_field',              '15'),
    ('dynamics.ftle_backward',           '17'),

    # cohort/cohort_dynamics/ (motion + rolling + urgency)
    ('dynamics.velocity_field',          '21'),
    ('dynamics.ftle_rolling',            '22'),
    ('dynamics.ridge_proximity',         '23'),
    ('dynamics.persistent_homology',     '36'),

    # system/ (system_vector built from cohort_geometry, then fleet stages)
    ('energy.cohort_vector',             '25'),
    ('energy.system_geometry',           '26'),
    ('energy.cohort_pairwise',           '27'),
    ('energy.cohort_information_flow',   '28'),
    ('energy.cohort_ftle',              '30'),
    ('energy.cohort_velocity_field',     '31'),
    ('energy.trajectory_signature',      '32'),

    # signal/ (stability)
    ('vector.signal_stability',          '33'),
]

# System-level stage IDs (fleet comparison, requires n_cohorts >= 2)
SYSTEM_STAGE_IDS = {'25', '26', '27', '28', '30', '31', '32'}

# Output subdirectories that belong to system-level computation
SYSTEM_SUBDIRS = {'system', 'system/system_dynamics'}

# Stages that must run globally before parallel split (column pruning is cross-cohort)
GLOBAL_FIRST_IDS = {'00', '01'}

# Stage IDs that can run independently per cohort (after global stages)
COHORT_PARALLEL_IDS = {
    '02', '03', '05', '06', '07', '08', '08_lyapunov',
    '09a', '10', '17', '21', '22', '23', '36', '33',
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
            'data_path': str(cohort_output),
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
    actual_filename = STAGE_FILENAMES.get(stem, filename)
    if subdir:
        d = output_dir / subdir
        d.mkdir(parents=True, exist_ok=True)
        return str(d / actual_filename)
    return str(output_dir / actual_filename)


def _resolve_system_mode(manifest: dict, obs_path: str, verbose: bool) -> bool:
    """
    Determine whether to skip system-level stages.

    Reads system.mode from manifest (auto|force|skip).
    When auto: skip if n_cohorts < 2.

    Returns True if system stages should be skipped.
    """
    import polars as pl

    mode = manifest.get('system', {}).get('mode', 'auto')
    if mode not in ('auto', 'force', 'skip'):
        if verbose:
            print(f"  Warning: unknown system.mode '{mode}', defaulting to 'auto'")
        mode = 'auto'

    if mode == 'skip':
        if verbose:
            print("System mode: skip — skipping system-level computation")
            print()
        return True

    if mode == 'force':
        if verbose:
            print("System mode: force — computing system level regardless of cohort count")
            print()
        return False

    # mode == 'auto': count cohorts
    obs = pl.read_parquet(obs_path)
    if 'cohort' not in obs.columns:
        n_cohorts = 1
    else:
        n_cohorts = obs['cohort'].n_unique()

    if n_cohorts < 2:
        if verbose:
            print(f"Single cohort detected, skipping system-level computation (system.mode=auto)")
            print()
        return True

    return False


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

    All 24 stages run by default. Use stages/skip for debugging only.

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

    manifest = load_manifest(str(manifest_path))

    # Validate manifest paths before any computation
    path_errors = validate_manifest_paths(manifest)
    if path_errors:
        msg = "MANIFEST PATH ERRORS:\n" + "\n".join(f"  - {e}" for e in path_errors)
        raise FileNotFoundError(msg)

    # data_path = output_dir itself (writer writes directly into subdirs)
    data_path = output_dir
    manifest['_data_dir'] = str(data_path)

    # Safety: refuse to wipe if this looks like a domain root
    if (output_dir / 'observations.parquet').exists():
        raise ValueError(
            f"output_dir ({output_dir}) contains observations.parquet. "
            f"Pass the output/ subdirectory, not the domain root."
        )

    # Determine system-level mode before wiping output
    skip_system = _resolve_system_mode(manifest, str(observations_path), verbose)
    system_mode = manifest.get('system', {}).get('mode', 'auto')

    # Fresh start — remove old outputs
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output subdirectories (skip system/ when not needed)
    for subdir in sorted(set(STAGE_DIRS.values())):
        if skip_system and subdir in SYSTEM_SUBDIRS:
            continue
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

    # Remove system stages when skipping system-level computation
    if skip_system:
        run_stages = [
            (mod, sid) for mod, sid in run_stages
            if sid not in SYSTEM_STAGE_IDS
        ]

    # Determine parallelism from MANIFOLD_WORKERS env var (0 = auto-detect)
    _env_workers = os.environ.get("MANIFOLD_WORKERS", "")
    if _env_workers:
        n_workers = int(_env_workers) or (os.cpu_count() or 2)
    else:
        n_workers = os.cpu_count() or 2

    if verbose:
        # Resolve backend info
        try:
            import pmtvs
            _use_rust = getattr(pmtvs, 'USE_RUST', False)
            backend_info = f"pmtvs ({'Rust' if _use_rust else 'Python'})"
        except ImportError:
            backend_info = "Python (pmtvs not installed)"

        print("=" * 70)
        print("MANIFOLD PIPELINE")
        print("=" * 70)
        print(f"Input:    {observations_path}")
        print(f"Manifest: {manifest_path}")
        print(f"Output:   {output_dir}")
        print(f"Stages:   {len(run_stages)}")
        print(f"Workers:  {n_workers} ({'parallel' if n_workers > 1 else 'sequential'})")
        print(f"Backend:  {backend_info}")
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
                    _dispatch(module, module_path, stage_id, obs_path, output_dir, manifest, verbose, str(data_path), system_mode)
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

                    _dispatch(module, module_path, stage_id, obs_path, output_dir, manifest, verbose, str(data_path), system_mode)

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

                _dispatch(module, module_path, stage_id, obs_path, output_dir, manifest, verbose, str(data_path), system_mode)

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
            subdir_path = output_dir / subdir
            if not subdir_path.exists():
                continue
            files = list(subdir_path.glob('*.parquet'))
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
    system_mode: str = 'auto',
) -> None:
    """Dispatch a stage with the correct arguments and output path."""
    from manifold.io.manifest import get_signal_0_metadata

    intervention = manifest.get('intervention')
    stage_name = module_path.split('.')[-1]  # e.g., 'ftle', 'breaks'
    s0_meta = get_signal_0_metadata(manifest)

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
            _out(output_dir, 'cohort_vector.parquet'),
            data_path=data_path_str,
            verbose=verbose,
            signal_0_name=s0_meta['name'],
            signal_0_unit=s0_meta['unit'],
        )

    elif stage_name == 'signal_geometry':
        module.run(
            _out(output_dir, 'signal_vector.parquet'),
            _out(output_dir, 'cohort_vector.parquet'),
            data_path=data_path_str,
            cohort_geometry_path=_out(output_dir, 'cohort_geometry.parquet'),
            verbose=verbose,
        )

    elif stage_name == 'signal_dominance':
        module.run(
            _out(output_dir, 'cohort_signal_positions.parquet'),
            data_path=data_path_str,
            verbose=verbose,
        )

    elif stage_name == 'geometry_dynamics':
        module.run(
            _out(output_dir, 'cohort_geometry.parquet'),
            data_path=data_path_str,
            verbose=verbose,
            signal_0_name=s0_meta['name'],
            signal_0_unit=s0_meta['unit'],
        )

    # ── information ──

    elif stage_name == 'signal_pairwise':
        module.run(
            _out(output_dir, 'signal_vector.parquet'),
            _out(output_dir, 'cohort_vector.parquet'),
            data_path=data_path_str,
            cohort_geometry_path=_out(output_dir, 'cohort_geometry.parquet'),
            verbose=verbose,
        )

    elif stage_name == 'information_flow':
        import polars as _pl
        pairwise_file = Path(_out(output_dir, 'cohort_pairwise.parquet'))
        if pairwise_file.exists() and len(_pl.read_parquet(str(pairwise_file))) > 0:
            module.run(
                obs_path,
                str(pairwise_file),
                data_path=data_path_str,
                verbose=verbose,
            )
        else:
            if verbose:
                print("  Skipped (empty cohort_pairwise)")
            from manifold.io.writer import write_output as _write
            _write(_pl.DataFrame(), data_path_str, 'cohort_information_flow', verbose=verbose)

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
        sg_path = _out(output_dir, 'cohort_geometry.parquet')
        if Path(sg_path).exists():
            module.run(sg_path, data_path=data_path_str, verbose=verbose)
        else:
            if verbose:
                print("  Skipped (cohort_geometry.parquet not found)")

    elif stage_name == 'ftle_field':
        module.run(
            _out(output_dir, 'cohort_vector.parquet'),
            _out(output_dir, 'cohort_geometry.parquet'),
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
        module.run(
            obs_path,
            data_path=data_path_str,
            verbose=verbose,
            signal_0_name=s0_meta['name'],
            signal_0_unit=s0_meta['unit'],
        )

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
        sv_path = _out(output_dir, 'cohort_vector.parquet')
        if Path(sv_path).exists():
            module.run(sv_path, data_path=data_path_str, verbose=verbose)
        else:
            if verbose:
                print("  Skipped (cohort_vector.parquet not found)")

    # ── energy (fleet) ──

    elif stage_name == 'cohort_vector':
        sg_path = _out(output_dir, 'cohort_geometry.parquet')
        if Path(sg_path).exists():
            module.run(sg_path, data_path=data_path_str, verbose=verbose)
        else:
            if verbose:
                print("  Skipped (cohort_geometry.parquet not found)")

    elif stage_name in (
        'system_geometry', 'cohort_pairwise', 'cohort_information_flow',
        'cohort_ftle', 'cohort_velocity_field',
    ):
        _dispatch_fleet(module, stage_name, output_dir, verbose, data_path_str, system_mode)

    elif stage_name == 'trajectory_signature':
        _dispatch_trajectory_signature(module, output_dir, verbose, data_path_str, system_mode)

    else:
        if verbose:
            print(f"  Warning: Unknown stage {stage_name}")


def _dispatch_fleet(
    module,
    stage_name: str,
    output_dir: Path,
    verbose: bool,
    data_path_str: str = '',
    system_mode: str = 'auto',
) -> None:
    """
    Dispatch fleet-scale stages (26-31).

    These read cohort_geometry.parquet directly and pivot internally.
    Guard: skip if cohort_geometry missing or n_cohorts < 2 (unless system.mode=force).
    """
    import polars as _pl

    sg_path = Path(_out(output_dir, 'cohort_geometry.parquet'))
    if not sg_path.exists():
        if verbose:
            print("  Skipped (cohort_geometry.parquet not found)")
        return

    sg = _pl.read_parquet(str(sg_path))
    if len(sg) == 0:
        if verbose:
            print("  Skipped (cohort_geometry.parquet is empty)")
        return
    if system_mode != 'force':
        if 'cohort' not in sg.columns or sg['cohort'].n_unique() < 2:
            n = sg['cohort'].n_unique() if 'cohort' in sg.columns else 0
            if verbose:
                print(f"  Skipped (n_cohorts={n} < 2)")
            return

    if stage_name == 'system_geometry':
        module.run(str(sg_path), data_path=data_path_str, verbose=verbose)

    elif stage_name == 'cohort_pairwise':
        loadings_path = Path(_out(output_dir, 'system_cohort_positions.parquet'))
        module.run(
            str(sg_path),
            data_path=data_path_str,
            system_geometry_loadings_path=str(loadings_path) if loadings_path.exists() else None,
            verbose=verbose,
        )

    elif stage_name == 'cohort_information_flow':
        pairwise_path = Path(_out(output_dir, 'system_pairwise.parquet'))
        if pairwise_path.exists() and len(_pl.read_parquet(str(pairwise_path))) > 0:
            module.run(
                str(sg_path),
                str(pairwise_path),
                data_path=data_path_str,
                verbose=verbose,
            )
        else:
            if verbose:
                print("  Skipped (empty system_pairwise)")
            from manifold.io.writer import write_output as _write
            _write(_pl.DataFrame(), data_path_str, 'system_information_flow', verbose=verbose)

    elif stage_name == 'cohort_ftle':
        module.run(str(sg_path), data_path=data_path_str, verbose=verbose)

    elif stage_name == 'cohort_velocity_field':
        module.run(str(sg_path), data_path=data_path_str, verbose=verbose)


def _dispatch_trajectory_signature(
    module,
    output_dir: Path,
    verbose: bool,
    data_path_str: str = '',
    system_mode: str = 'auto',
) -> None:
    """
    Dispatch stage 32 (trajectory signature library).

    Reads cohort_geometry, geometry_dynamics, velocity_field.
    Same fleet guard as _dispatch_fleet (n_cohorts >= 2 unless force).
    """
    import polars as _pl

    sg_path = Path(_out(output_dir, 'cohort_geometry.parquet'))
    if not sg_path.exists():
        if verbose:
            print("  Skipped (cohort_geometry.parquet not found)")
        return

    sg = _pl.read_parquet(str(sg_path))
    if len(sg) == 0:
        if verbose:
            print("  Skipped (cohort_geometry.parquet is empty)")
        return
    if system_mode != 'force':
        if 'cohort' not in sg.columns or sg['cohort'].n_unique() < 2:
            n = sg['cohort'].n_unique() if 'cohort' in sg.columns else 0
            if verbose:
                print(f"  Skipped (n_cohorts={n} < 2)")
            return

    module.run(
        cohort_geometry_path=str(sg_path),
        geometry_dynamics_path=_out(output_dir, 'geometry_dynamics.parquet'),
        velocity_field_path=_out(output_dir, 'velocity_field.parquet'),
        data_path=data_path_str,
        verbose=verbose,
    )


def _find_typology(manifest: Dict[str, Any], output_dir: Path) -> Optional[str]:
    """Find typology.parquet path from manifest."""
    manifest_parent = Path(manifest.get('_manifest_path', '')).parent if '_manifest_path' in manifest else output_dir.parent
    typology_path = manifest_parent / manifest.get('paths', {}).get('typology', 'typology.parquet')
    return str(typology_path) if typology_path.exists() else None


_READMES = {
    'signal': """\
# Signal

**What does each sensor look like?**

Per-signal windowed analysis: statistics, spectral, entropy, stability.

| File | Grain | Description |
|------|-------|-------------|
| signal_vector.parquet | (signal_id, signal_0_end) | Per-signal engine outputs |
| signal_geometry.parquet | (signal_id, signal_0_end) | Per-signal eigendecomposition |
| signal_stability.parquet | (signal_id, signal_0_end) | Hilbert + wavelet stability |
""",

    'cohort': """\
# Cohort

**What is the system's geometric structure and signal relationships?**

Cross-signal eigendecomposition and pairwise metrics per cohort.

| File | Grain | Description |
|------|-------|-------------|
| cohort_geometry.parquet | (cohort, signal_0_end) | Eigenvalues, effective dimension |
| cohort_vector.parquet | (cohort, signal_0_end) | Cross-signal centroid per window |
| cohort_signal_positions.parquet | (cohort, signal_0_end, signal_id) | Signal loadings on PCs |
| cohort_feature_loadings.parquet | (cohort, signal_0_end, feature) | Feature loadings on PC1 |
| cohort_pairwise.parquet | (cohort, signal_a, signal_b, signal_0_end) | Covariance, correlation, MI |
| cohort_information_flow.parquet | (source, target, signal_0_end) | Transfer entropy, Granger causality |
""",

    'cohort/cohort_dynamics': """\
# Cohort Dynamics

**How is each cohort changing over time?**

| File | Grain | Description |
|------|-------|-------------|
| breaks.parquet | (cohort, signal_0) | Change-point detection |
| geometry_dynamics.parquet | (cohort, signal_0_end) | Velocity/acceleration/jerk of eigenvalue trajectories |
| ftle.parquet | (cohort, signal_0_end) | Finite-Time Lyapunov Exponents |
| lyapunov.parquet | (signal_id) | Largest Lyapunov exponent per signal |
| thermodynamics.parquet | (cohort) | Shannon entropy from eigenvalue spectrum |
| ftle_field.parquet | (cohort, grid_x, grid_y) | Local FTLE grid |
| ftle_backward.parquet | (cohort, signal_0_end) | Backward FTLE |
| velocity_field.parquet | (cohort, signal_0_end) | Speed, direction in state space |
| ftle_rolling.parquet | (cohort, signal_0_end) | Time-varying FTLE windows |
| ridge_proximity.parquet | (cohort, signal_0_end) | Urgency = v . grad(FTLE) |
| persistent_homology.parquet | (cohort, signal_0_end) | Betti numbers, persistence entropy |
""",

    'system': """\
# System

**How does this compare across the fleet?**

Requires `cohort_geometry.parquet` and n_cohorts >= 2.

| File | Grain | Description |
|------|-------|-------------|
| system_geometry.parquet | (signal_0_end) | Fleet-level eigendecomposition |
| system_vector.parquet | (signal_0_end) | Cross-cohort centroid per window |
| system_cohort_positions.parquet | (signal_0_end, cohort) | Cohort loadings on PCs |
| system_pairwise.parquet | (cohort_a, cohort_b) | Distance between cohort vectors |
| system_information_flow.parquet | (source, target) | Transfer entropy at fleet scale |
""",

    'system/system_dynamics': """\
# System Dynamics

**How is the fleet evolving over time?**

| File | Grain | Description |
|------|-------|-------------|
| ftle.parquet | (cohort, signal_0_end) | FTLE on cohort trajectories |
| velocity_field.parquet | (cohort, signal_0_end) | Drift at fleet scale |
""",
}


def _write_readmes(output_dir: Path) -> None:
    """Write README.md into each output subdirectory (idempotent)."""
    for subdir, content in _READMES.items():
        subdir_path = output_dir / subdir
        if not subdir_path.exists():
            continue
        readme_path = subdir_path / 'README.md'
        if not readme_path.exists():
            readme_path.write_text(content)


def main():
    """CLI entry point. Resolves data_path into explicit paths and calls run()."""
    parser = argparse.ArgumentParser(
        description="Manifold Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
25 stages. All always-on.

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

    manifest = load_manifest(str(manifest_path))

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
