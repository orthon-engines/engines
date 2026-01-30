"""
PRISM Runner

One command. All engines. No logic.
Read manifest. Compute everything. Write parquets.

FULL COMPUTE. RAM OPTIMIZED. NO EXCEPTIONS.

Data lives in data/
- observations.parquet (ORTHON creates, PRISM reads)
- manifest.yaml (ORTHON creates, PRISM reads)
- *.parquet (PRISM writes, overwrites if exists)

Framework:
    GEOMETRY    → What is the structure?
    DYNAMICS    → How is structure changing?
    ENERGY      → What drives the change?
    SQL         → Normalize and summarize
"""

import json
import gc
from pathlib import Path
from typing import Dict, Any
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

from prism.python_runner import PythonRunner, SIGNAL_ENGINES, PAIR_ENGINES, SYMMETRIC_PAIR_ENGINES, WINDOWED_ENGINES
from prism.sql_runner import SQLRunner, SQL_ENGINES
from prism.ram_manager import RAMManager, MemoryStats, streaming_parquet_writer, combine_parquet_batches

# Canonical data directory
DATA_DIR = Path(__file__).parent.parent / 'data'


def run(manifest_path: Path = None) -> dict:
    """
    Run PRISM from manifest.

    No CLI flags.
    No conditional logic.
    Just compute everything.

    All data lives in data/
    """

    # Use canonical data directory
    data_dir = DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest from data/
    if manifest_path is None:
        manifest_path = data_dir / 'manifest.yaml'
    else:
        manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if manifest_path.suffix in ['.yaml', '.yml']:
        if not HAS_YAML:
            raise ImportError("PyYAML required: pip install pyyaml")
        manifest = yaml.safe_load(manifest_path.read_text())
    else:
        manifest = json.loads(manifest_path.read_text())

    # Observations always in data/
    observations_path = data_dir / 'observations.parquet'

    if not observations_path.exists():
        raise FileNotFoundError(
            f"observations.parquet not found in {data_dir}\n"
            "ORTHON must create observations.parquet before running PRISM."
        )

    # Output goes to data/ (same directory, overwrites existing)
    output_dir = data_dir

    # Extract params from manifest
    prism_config = manifest.get('prism', {})
    params = {
        'window_size': prism_config.get('window_size', prism_config.get('windows', {}).get('default', 100)),
        'stride': prism_config.get('stride', prism_config.get('windows', {}).get('stride', 50)),
    }

    # RAM config
    ram_config = prism_config.get('ram', {})
    ram_manager = RAMManager(
        max_memory_pct=ram_config.get('max_memory_pct', 0.8),
        min_batch_size=ram_config.get('min_batch_size', 10),
        max_batch_size=ram_config.get('max_batch_size', 500),
    )

    print("=" * 60)
    print("PRISM - FULL COMPUTE")
    print("=" * 60)
    print(f"Data:   {data_dir}")

    if HAS_PSUTIL:
        print(f"RAM:    {MemoryStats.current()}")

    # Load observations (ORTHON created this)
    print(f"\nLoading observations...")
    obs_pd = pd.read_parquet(observations_path)
    obs_pl = pl.from_pandas(obs_pd)
    print(f"  {len(obs_pd):,} observations")

    entities = obs_pl.select('entity_id').unique().to_series().to_list()
    print(f"  {len(entities)} entities")

    results = {}

    # ═══════════════════════════════════════════════════════════════
    # GEOMETRY: What is the structure?
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "═" * 60)
    print("GEOMETRY: What is the structure?")
    print("═" * 60)

    # Signal engines → primitives.parquet
    # Pair engines → primitives_pairs.parquet
    # Symmetric engines → geometry.parquet
    # Windowed engines → observations_enriched.parquet, manifold.parquet
    python_runner = PythonRunner(
        obs=obs_pd,
        output_dir=output_dir,
        engines={},  # PythonRunner runs ALL engines internally
        params=params
    )
    results['python'] = python_runner.run()
    _clear_ram()

    # Topology → topology.parquet
    print("\n[TOPOLOGY]")
    try:
        from prism.engines.topology_runner import process_entity_topology

        batch_dir = output_dir / '_topology_batches'
        batch_dir.mkdir(exist_ok=True)

        def process_topology_batch(batch_entities):
            batch_results = []
            for entity_id in batch_entities:
                entity_obs = obs_pl.filter(pl.col('entity_id') == entity_id)
                entity_results = process_entity_topology(entity_id, entity_obs, params)
                batch_results.extend(entity_results)
            return batch_results

        writer = streaming_parquet_writer(batch_dir, prefix='topology')
        ram_manager.process_in_batches(
            items=entities,
            process_func=process_topology_batch,
            write_func=writer,
            bytes_per_item=5_000_000,
        )

        output_path = output_dir / 'topology.parquet'
        n_rows = combine_parquet_batches(batch_dir, output_path, prefix='topology')
        results['topology'] = {'rows': n_rows}
        if batch_dir.exists():
            batch_dir.rmdir()
    except Exception as e:
        print(f"  Error: {e}")
        results['topology'] = {'error': str(e)}
    _clear_ram()

    # ═══════════════════════════════════════════════════════════════
    # DYNAMICS: How is structure changing?
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "═" * 60)
    print("DYNAMICS: How is structure changing?")
    print("═" * 60)

    # Dynamics → dynamics.parquet (Lyapunov, RQA, Hurst)
    print("\n[DYNAMICS - Lyapunov, RQA]")
    try:
        from prism.engines.dynamics_runner import process_entity_dynamics

        batch_dir = output_dir / '_dynamics_batches'
        batch_dir.mkdir(exist_ok=True)

        def process_dynamics_batch(batch_entities):
            batch_results = []
            for entity_id in batch_entities:
                entity_obs = obs_pl.filter(pl.col('entity_id') == entity_id)
                entity_results = process_entity_dynamics(entity_id, entity_obs, params)
                batch_results.extend(entity_results)
            return batch_results

        writer = streaming_parquet_writer(batch_dir, prefix='dynamics')
        ram_manager.process_in_batches(
            items=entities,
            process_func=process_dynamics_batch,
            write_func=writer,
            bytes_per_item=5_000_000,
        )

        output_path = output_dir / 'dynamics.parquet'
        n_rows = combine_parquet_batches(batch_dir, output_path, prefix='dynamics')
        results['dynamics'] = {'rows': n_rows}
        if batch_dir.exists():
            batch_dir.rmdir()
    except Exception as e:
        print(f"  Error: {e}")
        results['dynamics'] = {'error': str(e)}
    _clear_ram()

    # Information flow → information_flow.parquet
    print("\n[INFORMATION FLOW - Transfer Entropy, Granger]")
    try:
        from prism.engines.information_flow_runner import process_entity_information_flow

        batch_dir = output_dir / '_info_flow_batches'
        batch_dir.mkdir(exist_ok=True)

        def process_info_flow_batch(batch_entities):
            batch_results = []
            for entity_id in batch_entities:
                entity_obs = obs_pl.filter(pl.col('entity_id') == entity_id)
                entity_results = process_entity_information_flow(entity_id, entity_obs, params)
                batch_results.extend(entity_results)
            return batch_results

        writer = streaming_parquet_writer(batch_dir, prefix='info_flow')
        ram_manager.process_in_batches(
            items=entities,
            process_func=process_info_flow_batch,
            write_func=writer,
            bytes_per_item=5_000_000,
        )

        output_path = output_dir / 'information_flow.parquet'
        n_rows = combine_parquet_batches(batch_dir, output_path, prefix='info_flow')
        results['information_flow'] = {'rows': n_rows}
        if batch_dir.exists():
            batch_dir.rmdir()
    except Exception as e:
        print(f"  Error: {e}")
        results['information_flow'] = {'error': str(e)}
    _clear_ram()

    # ═══════════════════════════════════════════════════════════════
    # ENERGY: What drives the change?
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "═" * 60)
    print("ENERGY: What drives the change?")
    print("═" * 60)

    # Physics → physics.parquet
    print("\n[PHYSICS - Entropy, Energy, Free Energy]")
    try:
        from prism.engines.signal.physics_stack import compute_physics_for_all_entities

        obs_enriched_path = output_dir / 'observations_enriched.parquet'
        if obs_enriched_path.exists():
            obs_enriched = pd.read_parquet(obs_enriched_path)
        else:
            obs_enriched = obs_pd.copy()

        physics_df = compute_physics_for_all_entities(
            obs_enriched=obs_enriched,
            n_baseline=params.get('n_baseline', 100),
            coherence_window=params.get('coherence_window', 50),
        )

        if not physics_df.empty:
            output_path = output_dir / 'physics.parquet'
            physics_df.to_parquet(output_path, index=False)
            print(f"  physics.parquet: {len(physics_df):,} rows")
            results['physics'] = {'rows': len(physics_df)}
        else:
            results['physics'] = {'rows': 0}
    except Exception as e:
        print(f"  Error: {e}")
        results['physics'] = {'error': str(e)}
    _clear_ram()

    # ═══════════════════════════════════════════════════════════════
    # SQL RECONCILIATION: Normalize and summarize
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "═" * 60)
    print("SQL RECONCILIATION: Normalize and summarize")
    print("═" * 60)

    # SQL engines → zscore, statistics, correlation, regime_assignment
    sql_runner = SQLRunner(
        observations_path=observations_path,
        output_dir=output_dir,
        engines=SQL_ENGINES,
        params=params
    )
    results['sql'] = sql_runner.run()
    _clear_ram()

    # ═══════════════════════════════════════════════════════════════
    # COMPLETE
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "═" * 60)
    print("✅ PRISM COMPLETE")
    print("═" * 60)

    _print_summary(output_dir)

    return {
        'status': 'complete',
        'output_dir': str(output_dir),
        'results': results
    }


def _clear_ram():
    """Clear RAM and report status."""
    gc.collect()
    if HAS_PSUTIL:
        print(f"  {MemoryStats.current()}")


def _print_summary(output_dir: Path):
    """Print summary of output files."""
    print("\nOutput files:")
    for f in sorted(output_dir.glob('*.parquet')):
        if f.name == 'observations.parquet':
            continue  # Don't list input file
        size = f.stat().st_size
        if size > 1_000_000:
            size_str = f"{size / 1_000_000:.1f} MB"
        elif size > 1_000:
            size_str = f"{size / 1_000:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"  {f.name}: {size_str}")


# Direct execution
if __name__ == "__main__":
    run()
