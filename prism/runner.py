"""
PRISM Manifest Runner (Orchestrator)

Reads manifest from ORTHON and runs ALL engines.

FULL COMPUTE. RAM OPTIMIZED. NO EXCEPTIONS.

- ALL engines run, always
- Insufficient data → NaN, never skip
- RAM managed via entity batching
- Parallel execution controlled here (not in individual runners)
- Writes directly to data/ directory
"""

import json
import gc
import os
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import polars as pl

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

from prism.python_runner import PythonRunner, SIGNAL_ENGINES, PAIR_ENGINES, SYMMETRIC_PAIR_ENGINES, WINDOWED_ENGINES
from prism.sql_runner import SQLRunner, SQL_ENGINES


# ALL engines - always enabled, no exceptions
ALL_SIGNAL_ENGINES = SIGNAL_ENGINES
ALL_PAIR_ENGINES = PAIR_ENGINES
ALL_SYMMETRIC_PAIR_ENGINES = SYMMETRIC_PAIR_ENGINES
ALL_WINDOWED_ENGINES = WINDOWED_ENGINES
ALL_SQL_ENGINES = SQL_ENGINES


def load_manifest(manifest_path: str) -> dict:
    """
    Load manifest from YAML or JSON file.

    Supports both ORTHON format (YAML) and legacy format (JSON).
    """
    manifest_path = Path(manifest_path)

    with open(manifest_path) as f:
        content = f.read()

    # Try YAML first (ORTHON format), fall back to JSON (legacy)
    if manifest_path.suffix in ['.yaml', '.yml']:
        if not HAS_YAML:
            raise ImportError("PyYAML required for YAML manifests: pip install pyyaml")
        manifest = yaml.safe_load(content)
    elif manifest_path.suffix == '.json':
        manifest = json.loads(content)
    else:
        # Try both
        try:
            if HAS_YAML:
                manifest = yaml.safe_load(content)
            else:
                manifest = json.loads(content)
        except:
            manifest = json.loads(content)

    return normalize_manifest(manifest, manifest_path.parent)


def normalize_manifest(manifest: dict, manifest_dir: Path) -> dict:
    """
    Normalize manifest to internal format.

    Handles both ORTHON format and legacy format.
    Always enables ALL engines.
    """

    # Check if this is ORTHON format (has 'data' and/or 'prism' keys)
    if 'data' in manifest or 'prism' in manifest:
        return _normalize_orthon_manifest(manifest, manifest_dir)
    else:
        # Legacy format - normalize and enable all engines
        return _normalize_legacy_manifest(manifest)


def _normalize_orthon_manifest(manifest: dict, manifest_dir: Path) -> dict:
    """Normalize ORTHON YAML manifest to internal format."""

    data_config = manifest.get('data', {})
    prism_config = manifest.get('prism', {})

    # Resolve observations path
    obs_path = data_config.get('observations_path', data_config.get('output_path', 'observations.parquet'))
    if not Path(obs_path).is_absolute():
        obs_path = manifest_dir / obs_path
    obs_path = Path(obs_path)

    # Output directory is same as observations directory
    output_dir = obs_path.parent

    # Window/stride params
    window_size = prism_config.get('window_size', 100)
    stride = prism_config.get('stride', window_size)

    # Build params dict
    params = {
        'window_size': window_size,
        'stride': stride,
    }

    # Add any engine-specific params from prism config
    for key, value in prism_config.items():
        if key not in ['engines', 'ram', 'parallel', 'window_size', 'stride', 'compute']:
            params[key] = value

    # RAM and parallel config
    ram_config = prism_config.get('ram', {})
    parallel_config = prism_config.get('parallel', {})

    # ALL ENGINES - NO EXCEPTIONS
    normalized = {
        'observations_path': str(obs_path),
        'output_dir': str(output_dir),
        'engines': {
            'signal': ALL_SIGNAL_ENGINES,
            'pair': ALL_PAIR_ENGINES,
            'symmetric_pair': ALL_SYMMETRIC_PAIR_ENGINES,
            'windowed': ALL_WINDOWED_ENGINES,
            'sql': ALL_SQL_ENGINES,
            'dynamics': True,
            'topology': True,
            'information_flow': True,
            'physics': True,
        },
        'params': params,
        'parallel': parallel_config,
        'ram': ram_config,
        'metadata': manifest.get('dataset', {}),
    }

    return normalized


def _normalize_legacy_manifest(manifest: dict) -> dict:
    """Normalize legacy JSON manifest to internal format."""

    # Override engine config - ALL engines always
    manifest['engines'] = {
        'signal': ALL_SIGNAL_ENGINES,
        'pair': ALL_PAIR_ENGINES,
        'symmetric_pair': ALL_SYMMETRIC_PAIR_ENGINES,
        'windowed': ALL_WINDOWED_ENGINES,
        'sql': ALL_SQL_ENGINES,
        'dynamics': True,
        'topology': True,
        'information_flow': True,
        'physics': True,
    }

    if 'parallel' not in manifest:
        manifest['parallel'] = {}
    if 'ram' not in manifest:
        manifest['ram'] = {}

    return manifest


def get_ram_stats() -> dict:
    """Get current RAM statistics."""
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'used_pct': mem.percent,
        }
    return {'total_gb': 0, 'available_gb': 0, 'used_pct': 0}


class ManifestRunner:
    """
    Orchestrates manifest execution.

    FULL COMPUTE. RAM OPTIMIZED. NO EXCEPTIONS.

    - Runs ALL engines
    - Parallel execution controlled here (n_jobs parameter)
    - Returns NaN for insufficient data, never skips
    """

    def __init__(self, manifest: dict):
        self.manifest = manifest
        self.observations_path = Path(manifest['observations_path'])
        self.output_dir = Path(manifest['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.engine_config = manifest.get('engines', {})
        self.params = manifest.get('params', {})
        self.parallel_config = manifest.get('parallel', {})

        # Parallel settings: -1 = all cores, 1 = sequential
        self.n_jobs = self.parallel_config.get('n_jobs', 1)
        if self.n_jobs == -1:
            self.n_jobs = os.cpu_count() or 1

        # Load observations
        print(f"Loading observations from {self.observations_path}")
        self.obs_pd = pd.read_parquet(self.observations_path)
        self.obs_pl = pl.from_pandas(self.obs_pd)
        print(f"  Loaded {len(self.obs_pd):,} observations")

        # Get entities
        self.entities = self.obs_pl.select('entity_id').unique().to_series().to_list()
        self.n_entities = len(self.entities)
        print(f"  Entities: {self.n_entities}")
        print(f"  Parallel jobs: {self.n_jobs}")

        self._check_sampling()
        self.results: Dict[str, Any] = {}

    def _check_sampling(self):
        """Check and report on sampling uniformity."""
        try:
            I_values = self.obs_pl.select('I').unique().sort('I').to_series().to_numpy()
            if len(I_values) > 1:
                dI = np.diff(I_values)
                cv = np.std(dI) / (np.mean(dI) + 1e-10)
                if cv > 0.1:
                    print(f"  Note: Non-uniform sampling (CV={cv:.2f})")
        except Exception:
            pass

    def _clear_ram(self):
        """Clear RAM and report status."""
        gc.collect()
        if HAS_PSUTIL:
            stats = get_ram_stats()
            print(f"  RAM: {stats['used_pct']:.1f}% used ({stats['available_gb']:.1f}GB available)")

    def run(self) -> dict:
        """Execute the manifest - FULL COMPUTE, ALL ENGINES."""
        print("=" * 60)
        print("PRISM MANIFEST RUNNER - FULL COMPUTE")
        print("=" * 60)
        print(f"Input:  {self.observations_path}")
        print(f"Output: {self.output_dir}")
        print(f"Mode:   ALL engines, {'PARALLEL' if self.n_jobs > 1 else 'SEQUENTIAL'}")

        if HAS_PSUTIL:
            stats = get_ram_stats()
            print(f"RAM:    {stats['available_gb']:.1f}GB available")

        # 1. Python signal/pair engines
        python_engines = {
            'signal': self.engine_config.get('signal', ALL_SIGNAL_ENGINES),
            'pair': self.engine_config.get('pair', ALL_PAIR_ENGINES),
            'symmetric_pair': self.engine_config.get('symmetric_pair', ALL_SYMMETRIC_PAIR_ENGINES),
            'windowed': self.engine_config.get('windowed', ALL_WINDOWED_ENGINES),
        }

        print("\n" + "-" * 60)
        print("PYTHON RUNNER (ALL ENGINES)")
        print("-" * 60)

        python_runner = PythonRunner(
            obs=self.obs_pd,
            output_dir=self.output_dir,
            engines=python_engines,
            params=self.params
        )
        self.results['python'] = python_runner.run()
        self._clear_ram()

        # 2. SQL engines
        sql_engines = self.engine_config.get('sql', ALL_SQL_ENGINES)

        print("\n" + "-" * 60)
        print("SQL RUNNER (ALL ENGINES)")
        print("-" * 60)

        sql_runner = SQLRunner(
            observations_path=self.observations_path,
            output_dir=self.output_dir,
            engines=sql_engines,
            params=self.params
        )
        self.results['sql'] = sql_runner.run()
        self._clear_ram()

        # 3. Dynamics (parallel in orchestrator)
        self._run_dynamics_parallel()
        self._clear_ram()

        # 4. Topology (parallel in orchestrator)
        self._run_topology_parallel()
        self._clear_ram()

        # 5. Information flow (parallel in orchestrator)
        self._run_information_flow_parallel()
        self._clear_ram()

        # 6. Physics
        self._run_physics()
        self._clear_ram()

        print("\n" + "=" * 60)
        print("COMPLETE - FULL COMPUTE")
        print("=" * 60)
        self._print_summary()

        return {
            'status': 'complete',
            'output_dir': str(self.output_dir),
            'results': self.results
        }

    def _run_dynamics_parallel(self):
        """Compute dynamics with optional parallel execution."""
        print("\n" + "-" * 60)
        print(f"DYNAMICS ENGINE {'(PARALLEL)' if self.n_jobs > 1 else '(SEQUENTIAL)'}")
        print("-" * 60)

        try:
            from prism.engines.dynamics_runner import process_entity_dynamics

            dynamics_params = self.params.get('dynamics', {})
            print(f"  Processing {self.n_entities} entities on {self.n_jobs} workers...")

            if self.n_jobs > 1 and HAS_JOBLIB:
                results_nested = Parallel(n_jobs=self.n_jobs)(
                    delayed(process_entity_dynamics)(
                        entity_id,
                        self.obs_pl.filter(pl.col('entity_id') == entity_id),
                        dynamics_params
                    )
                    for entity_id in self.entities
                )
                all_results = [r for entity_results in results_nested for r in entity_results]
            else:
                all_results = []
                for entity_id in self.entities:
                    entity_obs = self.obs_pl.filter(pl.col('entity_id') == entity_id)
                    entity_results = process_entity_dynamics(entity_id, entity_obs, dynamics_params)
                    all_results.extend(entity_results)

            if not all_results:
                print("  Warning: no dynamics data computed")
                self.results['dynamics'] = {'rows': 0, 'cols': 0}
                return

            df = pl.DataFrame(all_results)
            output_path = self.output_dir / 'dynamics.parquet'
            df.write_parquet(output_path)
            print(f"  dynamics.parquet: {len(df):,} rows x {len(df.columns)} cols")
            self.results['dynamics'] = {'rows': len(df), 'cols': len(df.columns)}

        except Exception as e:
            print(f"  Error in dynamics engine: {e}")
            self.results['dynamics'] = {'error': str(e)}

    def _run_topology_parallel(self):
        """Compute topology with optional parallel execution."""
        print("\n" + "-" * 60)
        print(f"TOPOLOGY ENGINE {'(PARALLEL)' if self.n_jobs > 1 else '(SEQUENTIAL)'}")
        print("-" * 60)

        try:
            from prism.engines.topology_runner import process_entity_topology

            topology_params = self.params.get('topology', {})
            print(f"  Processing {self.n_entities} entities on {self.n_jobs} workers...")

            if self.n_jobs > 1 and HAS_JOBLIB:
                results_nested = Parallel(n_jobs=self.n_jobs)(
                    delayed(process_entity_topology)(
                        entity_id,
                        self.obs_pl.filter(pl.col('entity_id') == entity_id),
                        topology_params
                    )
                    for entity_id in self.entities
                )
                all_results = [r for entity_results in results_nested for r in entity_results]
            else:
                all_results = []
                for entity_id in self.entities:
                    entity_obs = self.obs_pl.filter(pl.col('entity_id') == entity_id)
                    entity_results = process_entity_topology(entity_id, entity_obs, topology_params)
                    all_results.extend(entity_results)

            if not all_results:
                print("  Warning: no topology data computed")
                self.results['topology'] = {'rows': 0, 'cols': 0}
                return

            df = pl.DataFrame(all_results)
            output_path = self.output_dir / 'topology.parquet'
            df.write_parquet(output_path)
            print(f"  topology.parquet: {len(df):,} rows x {len(df.columns)} cols")
            self.results['topology'] = {'rows': len(df), 'cols': len(df.columns)}

        except Exception as e:
            print(f"  Error in topology engine: {e}")
            self.results['topology'] = {'error': str(e)}

    def _run_information_flow_parallel(self):
        """Compute information flow with optional parallel execution."""
        print("\n" + "-" * 60)
        print(f"INFORMATION FLOW ENGINE {'(PARALLEL)' if self.n_jobs > 1 else '(SEQUENTIAL)'}")
        print("-" * 60)

        try:
            from prism.engines.information_flow_runner import process_entity_information_flow

            info_params = self.params.get('information_flow', {})
            print(f"  Processing {self.n_entities} entities on {self.n_jobs} workers...")

            if self.n_jobs > 1 and HAS_JOBLIB:
                results_nested = Parallel(n_jobs=self.n_jobs)(
                    delayed(process_entity_information_flow)(
                        entity_id,
                        self.obs_pl.filter(pl.col('entity_id') == entity_id),
                        info_params
                    )
                    for entity_id in self.entities
                )
                all_results = [r for entity_results in results_nested for r in entity_results]
            else:
                all_results = []
                for entity_id in self.entities:
                    entity_obs = self.obs_pl.filter(pl.col('entity_id') == entity_id)
                    entity_results = process_entity_information_flow(entity_id, entity_obs, info_params)
                    all_results.extend(entity_results)

            if not all_results:
                print("  Warning: no information flow data computed")
                self.results['information_flow'] = {'rows': 0, 'cols': 0}
                return

            df = pl.DataFrame(all_results)
            output_path = self.output_dir / 'information_flow.parquet'
            df.write_parquet(output_path)
            print(f"  information_flow.parquet: {len(df):,} rows x {len(df.columns)} cols")
            self.results['information_flow'] = {'rows': len(df), 'cols': len(df.columns)}

        except Exception as e:
            print(f"  Error in information flow engine: {e}")
            self.results['information_flow'] = {'error': str(e)}

    def _run_physics(self):
        """Compute physics stack."""
        print("\n" + "-" * 60)
        print("PHYSICS STACK")
        print("-" * 60)

        obs_enriched_path = self.output_dir / 'observations_enriched.parquet'

        try:
            from prism.engines.signal.physics_stack import compute_physics_for_all_entities

            physics_params = self.params.get('physics', {})

            if obs_enriched_path.exists():
                obs_enriched = pd.read_parquet(obs_enriched_path)
                print(f"  Using observations_enriched.parquet ({len(obs_enriched):,} rows)")
            else:
                print("  Using raw observations")
                obs_enriched = self.obs_pd.copy()
                if 'y' not in obs_enriched.columns and 'value' in obs_enriched.columns:
                    obs_enriched['y'] = obs_enriched['value']

            physics_df = compute_physics_for_all_entities(
                obs_enriched=obs_enriched,
                n_baseline=physics_params.get('n_baseline', 100),
                coherence_window=physics_params.get('coherence_window', 50),
            )

            if not physics_df.empty:
                output_path = self.output_dir / 'physics.parquet'
                physics_df.to_parquet(output_path, index=False)
                print(f"  physics.parquet: {len(physics_df):,} rows x {len(physics_df.columns)} cols")
                self.results['physics'] = {'rows': len(physics_df), 'cols': len(physics_df.columns)}
            else:
                print("  Warning: no physics data computed")
                self.results['physics'] = {'rows': 0, 'cols': 0}

        except Exception as e:
            print(f"  Error in physics engine: {e}")
            self.results['physics'] = {'error': str(e)}

    def _print_summary(self):
        """Print summary of outputs."""
        print("\nOutput files:")
        for f in sorted(self.output_dir.glob('*.parquet')):
            size = f.stat().st_size
            if size > 1_000_000:
                size_str = f"{size / 1_000_000:.1f} MB"
            elif size > 1_000:
                size_str = f"{size / 1_000:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"  {f.name}: {size_str}")


def run_manifest(manifest_path: str) -> dict:
    """Load and run a manifest from a YAML or JSON file."""
    manifest = load_manifest(manifest_path)
    runner = ManifestRunner(manifest)
    return runner.run()


def run_manifest_dict(manifest: dict) -> dict:
    """Run a manifest from a dictionary."""
    if 'data' in manifest or 'prism' in manifest:
        manifest = normalize_manifest(manifest, Path('.'))
    else:
        manifest = _normalize_legacy_manifest(manifest)
    runner = ManifestRunner(manifest)
    return runner.run()


__all__ = [
    'ManifestRunner',
    'run_manifest',
    'run_manifest_dict',
    'load_manifest',
    'normalize_manifest',
    'SIGNAL_ENGINES',
    'PAIR_ENGINES',
    'SYMMETRIC_PAIR_ENGINES',
    'WINDOWED_ENGINES',
    'SQL_ENGINES',
    'ALL_SIGNAL_ENGINES',
    'ALL_PAIR_ENGINES',
    'ALL_SYMMETRIC_PAIR_ENGINES',
    'ALL_WINDOWED_ENGINES',
    'ALL_SQL_ENGINES',
]
