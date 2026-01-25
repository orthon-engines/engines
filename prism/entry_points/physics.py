#!/usr/bin/env python3
"""
PRISM Physics Entry Point
=========================

Orchestrates energy and momentum calculations using physics engines.

REQUIRES: vector.parquet (for velocity/position signals)

Engines (7 total):
    Energy: kinetic_energy, potential_energy, hamiltonian, lagrangian
    Momentum: linear_momentum, angular_momentum
    Thermodynamics: gibbs_free_energy, work_energy

Usage:
    python -m prism.entry_points.physics
    python -m prism.entry_points.physics --force

Output:
    data/physics.parquet
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import polars as pl

from prism.db.parquet_store import get_path, ensure_directory, VECTOR, PHYSICS
from prism.db.polars_io import read_parquet, write_parquet_atomic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENGINE REGISTRY
# =============================================================================

ENGINES = {
    'kinetic': 'prism.engines.physics:compute_kinetic',
    'potential': 'prism.engines.physics:compute_potential',
    'hamiltonian': 'prism.engines.physics:compute_hamilton',
    'lagrangian': 'prism.engines.physics:compute_lagrange',
    'momentum': 'prism.engines.physics:compute_momentum',
    'gibbs': 'prism.engines.physics:compute_gibbs',
    'work_energy': 'prism.engines.physics:compute_work_energy',
}


def load_engine(spec: str):
    """Load engine function from module:function spec."""
    module_path, func_name = spec.rsplit(':', 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


# =============================================================================
# CONFIG
# =============================================================================

def load_config(data_path: Path) -> Dict[str, Any]:
    """Load config.json or config.yaml from data directory."""
    import json

    config_path = data_path / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)

    yaml_path = data_path / 'config.yaml'
    if yaml_path.exists():
        import yaml
        with open(yaml_path) as f:
            return yaml.safe_load(f)

    return {}


# =============================================================================
# ORCHESTRATOR
# =============================================================================

def run_physics_engines(
    vector_df: pl.DataFrame,
    config: Dict[str, Any],
    engines: List[str] = None,
) -> pl.DataFrame:
    """
    Orchestrate physics engine execution.

    Pure orchestration - no computation here, just routing.
    """
    if engines is None:
        engines = list(ENGINES.keys())

    # Get constants from config
    constants = config.get('global_constants', {})

    results = []
    entities = vector_df['entity_id'].unique().to_list() if 'entity_id' in vector_df.columns else ['default']

    for entity_id in entities:
        entity_df = vector_df.filter(pl.col('entity_id') == entity_id) if 'entity_id' in vector_df.columns else vector_df

        row = {'entity_id': entity_id}

        for engine_name in engines:
            if engine_name not in ENGINES:
                continue

            try:
                engine = load_engine(ENGINES[engine_name])
                result = engine(df=entity_df, constants=constants)

                # Flatten engine output into row
                if isinstance(result, dict):
                    for k, v in result.items():
                        if isinstance(v, (int, float, bool)) or v is None:
                            row[f"{engine_name}_{k}"] = v

            except Exception as e:
                logger.debug(f"{engine_name} skipped: {e}")

        results.append(row)

    return pl.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PRISM Physics")
    parser.add_argument('--force', '-f', action='store_true')
    parser.add_argument('--engines', '-e', nargs='+', choices=list(ENGINES.keys()))
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PRISM Physics")
    logger.info("=" * 60)

    ensure_directory()
    data_path = get_path(PHYSICS).parent

    # Check dependency
    vector_path = get_path(VECTOR)
    if not vector_path.exists():
        logger.error("vector.parquet required - run: python -m prism.entry_points.vector")
        return 1

    output_path = get_path(PHYSICS)
    if output_path.exists() and not args.force:
        logger.info("physics.parquet exists, use --force to recompute")
        return 0

    # Load
    vector_df = read_parquet(vector_path)
    config = load_config(data_path)

    logger.info(f"Vector: {len(vector_df)} rows")
    logger.info(f"Engines: {args.engines or list(ENGINES.keys())}")

    # Run
    start = time.time()
    physics_df = run_physics_engines(vector_df, config, args.engines)

    logger.info(f"Complete: {time.time() - start:.1f}s")
    logger.info(f"Output: {len(physics_df)} rows, {len(physics_df.columns)} columns")

    # Save
    write_parquet_atomic(physics_df, output_path)
    logger.info(f"Saved: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
